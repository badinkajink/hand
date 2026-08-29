/*
 * 8-axis independent open-loop step generator.
 *
 * Design choice: each axis's timer runs purely as an interrupt time-base
 * (update event only, no hardware output-compare/PWM). The STEP pin is
 * toggled in software from the ISR. This means the timer instance assigned
 * to an axis does NOT need to be the one whose alternate function reaches
 * that axis's STEP pin -- any general-purpose timer works as a tick source,
 * which sidesteps needing to verify AF/timer-channel routing for 8 pins.
 *
 * Motion model: independent trapezoidal velocity profile per axis, recomputed
 * every 1ms by Stepper_SupervisorTick() (driven by TIM6). Each axis's own
 * timer ARR/PSC is rewritten to match its current commanded step rate.
 *
 * Timer kernel clock: at this project's 550MHz/VOS0 SystemClock_Config,
 * D2PPRE1 and D2PPRE2 are both /2 (APB1=APB2=137.5MHz), and STM32H7 timer
 * kernel clocks are 2x their APB bus when that bus's prescaler != 1 --
 * so every timer used here (APB1: TIM2,3,4,5,12,13,14; APB2: TIM15) gets a
 * uniform 275MHz kernel clock. If you change SystemClock_Config, update
 * TIMER_KERNEL_CLOCK_HZ to match.
 */

#include <stdlib.h>
#include <math.h>
#include "stepper.h"
#include "tmc5160_spi.h"

#define TIMER_KERNEL_CLOCK_HZ 275000000UL
#define SUPERVISOR_HZ         1000U
/* Crude minimum pulse-width guarantee: ~600 cycles at this project's 550MHz
 * core clock is a little over 1us high time, comfortably over TMC2209's
 * minimum STEP pulse width spec and still negligible against any step
 * period this firmware will actually command. The original value here (20,
 * ~36ns) was untested against real hardware and likely too short for the
 * driver to reliably register every edge. */
#define STEP_PULSE_NOPS       600U

typedef struct {
    GPIO_TypeDef *port;
    uint16_t      pin;
} Pin_t;

typedef struct {
    Pin_t step, dir, en;
    TIM_TypeDef *tim;
    IRQn_Type    irq;
} AxisHw_t;

/* From docs/pinout.md. Timer assignment is this firmware's own choice (see
 * header comment above) -- not tied to BTT's board design.
 *
 * NOTE: TIM12/13/14 do NOT have standalone NVIC vectors on this chip --
 * their interrupts are combined onto TIM8's BRK/UP/TRG_COM lines
 * (TIM8_BRK_TIM12_IRQn etc., see stm32h723xx.h's IRQn_Type enum). A build
 * caught this (undeclared TIM12_IRQn/TIM13_IRQn/TIM14_IRQn) -- rather than
 * deal with shared-vector semantics, J4-J6 use TIM7/TIM16/TIM17 instead,
 * which all have genuinely standalone vectors. */
static const AxisHw_t axis_hw[STEPPER_NUM_AXES] = {
    /* J0 / M1 */ { {GPIOE, GPIO_PIN_6}, {GPIOE, GPIO_PIN_5}, {GPIOC, GPIO_PIN_14}, TIM2,  TIM2_IRQn  },
    /* J1 / M2 */ { {GPIOE, GPIO_PIN_2}, {GPIOE, GPIO_PIN_1}, {GPIOE, GPIO_PIN_4},  TIM3,  TIM3_IRQn  },
    /* J2 / M3 */ { {GPIOB, GPIO_PIN_8}, {GPIOB, GPIO_PIN_7}, {GPIOE, GPIO_PIN_0},  TIM4,  TIM4_IRQn  },
    /* J3 / M4 */ { {GPIOB, GPIO_PIN_4}, {GPIOB, GPIO_PIN_3}, {GPIOB, GPIO_PIN_6},  TIM5,  TIM5_IRQn  },
    /* J4 / M5 */ { {GPIOG, GPIO_PIN_13},{GPIOG, GPIO_PIN_12},{GPIOG, GPIO_PIN_15}, TIM7,  TIM7_IRQn  },
    /* J5 / M6 */ { {GPIOG, GPIO_PIN_9}, {GPIOD, GPIO_PIN_7}, {GPIOG, GPIO_PIN_11}, TIM16, TIM16_IRQn },
    /* J6 / M7 */ { {GPIOD, GPIO_PIN_4}, {GPIOD, GPIO_PIN_3}, {GPIOD, GPIO_PIN_6},  TIM17, TIM17_IRQn },
    /* J7 / M8 */ { {GPIOC, GPIO_PIN_7}, {GPIOC, GPIO_PIN_8}, {GPIOD, GPIO_PIN_2},  TIM15, TIM15_IRQn },
};

static TIM_HandleTypeDef axis_tim[STEPPER_NUM_AXES];
static Stepper_Status axis_st[STEPPER_NUM_AXES];
static uint8_t axis_running[STEPPER_NUM_AXES]; /* tracks set_axis_rate's stopped/running state */

static void gpio_clock_enable_for_port(GPIO_TypeDef *port)
{
    if (port == GPIOB) __HAL_RCC_GPIOB_CLK_ENABLE();
    else if (port == GPIOC) __HAL_RCC_GPIOC_CLK_ENABLE();
    else if (port == GPIOD) __HAL_RCC_GPIOD_CLK_ENABLE();
    else if (port == GPIOE) __HAL_RCC_GPIOE_CLK_ENABLE();
    else if (port == GPIOF) __HAL_RCC_GPIOF_CLK_ENABLE();
    else if (port == GPIOG) __HAL_RCC_GPIOG_CLK_ENABLE();
}

/* DIAG pin per axis, for sensorless homing (see Stepper_Home). From BTT's
 * own reference Klipper config for this board (generic-bigtreetech-manta-
 * m8p-V2_0.cfg): Motor1/J0 diag_pin=PF4, Motor2/J1 diag_pin=PF3, Motor3/J2
 * diag_pin=PF2, Motor4/J3 diag_pin=PF1 -- confirmed against the physical
 * board's DIAG jumpers (see docs/bringup.md), not re-derived from the
 * schematic ourselves. J0-J3 have both a jumper installed and a confirmed
 * pin as of this writing. J4/J5 (Motor5/Motor6) also have jumpers installed
 * but NOT a confirmed pin -- the same reference cfg has no diag_pin lines
 * for them at all, only "Endstop5/6" tied to filament-sensor switch_pins
 * (PF0/PC15), which may or may not be the same physical net as those
 * motors' DIAG output. Guessing wrong here isn't dangerous (it's just a
 * GPIO input read) but could silently conflict with that documented
 * filament-sensor use, so {NULL, 0} for J4/J5 until that's confirmed rather
 * than assumed. Extend this table (and the EXTI wiring in Stepper_Init/the
 * EXTI IRQ handlers in stm32h7xx_it.c) once it is. */
typedef struct {
    GPIO_TypeDef *port;
    uint16_t      pin;
} DiagPin_t;

static const DiagPin_t axis_diag[STEPPER_NUM_AXES] = {
    /* J0 / M1 */ { GPIOF, GPIO_PIN_4 },
    /* J1 / M2 */ { GPIOF, GPIO_PIN_3 },
    /* J2 / M3 */ { GPIOF, GPIO_PIN_2 },
    /* J3 / M4 */ { GPIOF, GPIO_PIN_1 },
    /* J4 / M5 */ { NULL, 0 },
    /* J5 / M6 */ { NULL, 0 },
    /* J6 / M7 */ { NULL, 0 },
    /* J7 / M8 */ { NULL, 0 },
};

static uint32_t axis_homing_deadline_ms[STEPPER_NUM_AXES];
static uint32_t axis_homing_grace_until_ms[STEPPER_NUM_AXES];

static void timer_clock_enable(TIM_TypeDef *tim)
{
    if (tim == TIM2) __HAL_RCC_TIM2_CLK_ENABLE();
    else if (tim == TIM3) __HAL_RCC_TIM3_CLK_ENABLE();
    else if (tim == TIM4) __HAL_RCC_TIM4_CLK_ENABLE();
    else if (tim == TIM5) __HAL_RCC_TIM5_CLK_ENABLE();
    else if (tim == TIM7) __HAL_RCC_TIM7_CLK_ENABLE();
    else if (tim == TIM15) __HAL_RCC_TIM15_CLK_ENABLE();
    else if (tim == TIM16) __HAL_RCC_TIM16_CLK_ENABLE();
    else if (tim == TIM17) __HAL_RCC_TIM17_CLK_ENABLE();
}

void Stepper_Init(void)
{
    /* SYSCFG clock needed before HAL_GPIO_Init can route any GPIO_MODE_IT_*
     * pin's EXTI line (it writes SYSCFG->EXTICR) -- must run before the
     * per-axis DIAG pin init inside the loop below, not after it. */
    __HAL_RCC_SYSCFG_CLK_ENABLE();

    for (uint8_t a = 0; a < STEPPER_NUM_AXES; a++) {
        const AxisHw_t *hw = &axis_hw[a];

        gpio_clock_enable_for_port(hw->step.port);
        gpio_clock_enable_for_port(hw->dir.port);
        gpio_clock_enable_for_port(hw->en.port);

        GPIO_InitTypeDef gi = {0};
        gi.Mode = GPIO_MODE_OUTPUT_PP;
        gi.Pull = GPIO_NOPULL;
        gi.Speed = GPIO_SPEED_FREQ_HIGH;

        gi.Pin = hw->step.pin;
        HAL_GPIO_Init(hw->step.port, &gi);
        HAL_GPIO_WritePin(hw->step.port, hw->step.pin, GPIO_PIN_RESET);

        gi.Pin = hw->dir.pin;
        HAL_GPIO_Init(hw->dir.port, &gi);
        HAL_GPIO_WritePin(hw->dir.port, hw->dir.pin, GPIO_PIN_RESET);

        gi.Pin = hw->en.pin;
        HAL_GPIO_Init(hw->en.port, &gi);
        HAL_GPIO_WritePin(hw->en.port, hw->en.pin, GPIO_PIN_SET); /* active-low: SET = disabled */

        timer_clock_enable(hw->tim);

        TIM_HandleTypeDef *htim = &axis_tim[a];
        htim->Instance = hw->tim;
        htim->Init.Prescaler = TIMER_KERNEL_CLOCK_HZ / 1000000U - 1U; /* 1MHz tick, retuned per-move below */
        htim->Init.CounterMode = TIM_COUNTERMODE_UP;
        htim->Init.Period = 0xFFFF;
        htim->Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
        htim->Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
        HAL_TIM_Base_Init(htim);

        HAL_NVIC_SetPriority(hw->irq, 2, 0);
        HAL_NVIC_EnableIRQ(hw->irq);

        axis_st[a] = (Stepper_Status){0};
        axis_st[a].max_velocity_sps = 2000;
        axis_st[a].accel_sps2 = 4000;

        if (axis_diag[a].port != NULL) {
            gpio_clock_enable_for_port(axis_diag[a].port);
            GPIO_InitTypeDef di = {0};
            di.Pin = axis_diag[a].pin;
            di.Mode = GPIO_MODE_IT_FALLING; /* DIAG pulls this line low on a stall, not high --
                                                confirmed via SWD/live testing on J1's TMC5160: the
                                                board's "M#-DIAG" jumper routes DIAG through the
                                                same "M#-STOP-Det" net as the endstop header, which
                                                has its own onboard +5V pull-up for standard NC/NO
                                                switch wiring (idle high, pulled low on trigger).
                                                Previously configured rising-edge/pulldown, which
                                                would never see the actual falling-edge stall event
                                                -- likely why DIAG never fired for any axis all the
                                                way back to the original TMC2209 testing. */
            di.Pull = GPIO_PULLUP; /* defensive: reads a clean inactive HIGH rather than floating
                                       if ever disconnected, instead of risking a false stall
                                       trigger (the board's own onboard pull-up on this net should
                                       normally make this redundant). */
            HAL_GPIO_Init(axis_diag[a].port, &di);
        }
    }

    /* EXTI1-4 each have their own standalone NVIC vector on this chip (only
     * EXTI5-9 and EXTI10-15 are combined) -- one line per DIAG pin here, no
     * shared-vector demuxing needed for J0-J3. Priority 0 (highest on this
     * chip, lower number = higher): reacting to a stall fast is the entire
     * point, and it must preempt the axis step timers (priority 2) and the
     * supervisor tick (priority 1) to stop cleanly. Left enabled
     * unconditionally rather than armed/disarmed per Stepper_Home call --
     * Stepper_DiagIRQ itself checks whether the axis is actually homing and
     * is a no-op otherwise, which avoids any enable/disable race with a
     * pulse arriving right at the start/end of a homing move. */
    if (axis_diag[0].port != NULL) {
        HAL_NVIC_SetPriority(EXTI4_IRQn, 0, 0);
        HAL_NVIC_EnableIRQ(EXTI4_IRQn);
    }
    if (axis_diag[1].port != NULL) {
        HAL_NVIC_SetPriority(EXTI3_IRQn, 0, 0);
        HAL_NVIC_EnableIRQ(EXTI3_IRQn);
    }
    if (axis_diag[2].port != NULL) {
        HAL_NVIC_SetPriority(EXTI2_IRQn, 0, 0);
        HAL_NVIC_EnableIRQ(EXTI2_IRQn);
    }
    if (axis_diag[3].port != NULL) {
        HAL_NVIC_SetPriority(EXTI1_IRQn, 0, 0);
        HAL_NVIC_EnableIRQ(EXTI1_IRQn);
    }

    /* TIM6: 1kHz supervisory tick, APB1 basic timer, also gets the x2 kernel
     * clock multiplier (275MHz) under this project's clock config. */
    __HAL_RCC_TIM6_CLK_ENABLE();
    TIM_HandleTypeDef htim6 = {0};
    htim6.Instance = TIM6;
    htim6.Init.Prescaler = (TIMER_KERNEL_CLOCK_HZ / 1000000U) - 1U; /* 1MHz */
    htim6.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim6.Init.Period = (1000000U / SUPERVISOR_HZ) - 1U;            /* 1ms */
    htim6.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
    HAL_TIM_Base_Init(&htim6);
    HAL_NVIC_SetPriority(TIM6_DAC_IRQn, 1, 0);
    HAL_NVIC_EnableIRQ(TIM6_DAC_IRQn);
    HAL_TIM_Base_Start_IT(&htim6);
}

void Stepper_Enable(uint8_t axis)
{
    if (axis >= STEPPER_NUM_AXES) return;
    HAL_GPIO_WritePin(axis_hw[axis].en.port, axis_hw[axis].en.pin, GPIO_PIN_RESET);
    axis_st[axis].enabled = 1;
}

void Stepper_Disable(uint8_t axis)
{
    if (axis >= STEPPER_NUM_AXES) return;
    HAL_GPIO_WritePin(axis_hw[axis].en.port, axis_hw[axis].en.pin, GPIO_PIN_SET);
    axis_st[axis].enabled = 0;
    axis_st[axis].moving = 0;
    axis_st[axis].current_velocity_sps = 0;
    /* Disabling mid-homing leaves the axis un-driven with the DIAG interrupt
     * still logically "armed" (homing_result==1) otherwise -- Stepper_Home
     * checks enabled before starting, but nothing re-checks it once a JOG is
     * already running, so cancel here explicitly rather than leave stale
     * homing state around. */
    axis_st[axis].homing_result = 0;
    HAL_TIM_Base_Stop_IT(&axis_tim[axis]);
    axis_running[axis] = 0;
}

void Stepper_SetInvertDir(uint8_t axis, uint8_t invert)
{
    if (axis >= STEPPER_NUM_AXES) return;
    axis_st[axis].invert_dir = invert ? 1 : 0;
}

void Stepper_Zero(uint8_t axis)
{
    if (axis >= STEPPER_NUM_AXES) return;
    axis_st[axis].position = 0;
    axis_st[axis].target = 0;
}

void Stepper_Move(uint8_t axis, int32_t abs_steps, uint32_t max_vel_sps, uint32_t accel_sps2)
{
    if (axis >= STEPPER_NUM_AXES) return;
    Stepper_Status *s = &axis_st[axis];
    /* Motion-start current kick: see TMC5160_StartMotionKick's doc comment.
     * current_velocity_sps==0 means genuinely at rest (not just idle between
     * two already-in-motion legs), which is when static friction is highest
     * and this axis's normal running current has been confirmed (real
     * hardware, J0, 2026-08-24) to sometimes not be enough to start turning
     * at all -- with the firmware unable to tell, since step counting is
     * open-loop. No-op on axes without a TMC5160 (checked inside). */
    if (s->current_velocity_sps == 0) TMC5160_StartMotionKick(axis);
    s->target = abs_steps;
    s->max_velocity_sps = max_vel_sps ? max_vel_sps : 1;
    s->accel_sps2 = accel_sps2 ? accel_sps2 : 1;
    s->jog_velocity = 0;
    s->moving = (s->position != s->target);
}

/* steps-per-mm per axis, 0 = uncalibrated. RAM only -- lost on reset, same as
 * position itself (see Stepper_Zero). Re-run SETSCALE (and re-home) after
 * every power cycle rather than treating either as durable across one. */
static float axis_steps_per_mm[STEPPER_NUM_AXES];

void Stepper_SetStepsPerMM(uint8_t axis, float steps_per_mm)
{
    if (axis >= STEPPER_NUM_AXES) return;
    axis_steps_per_mm[axis] = steps_per_mm;
}

uint8_t Stepper_MoveMM(uint8_t axis, float mm, uint32_t max_vel_sps, uint32_t accel_sps2)
{
    if (axis >= STEPPER_NUM_AXES || axis_steps_per_mm[axis] == 0.0f) return 0;
    Stepper_Move(axis, (int32_t)lroundf(mm * axis_steps_per_mm[axis]), max_vel_sps, accel_sps2);
    return 1;
}

void Stepper_Jog(uint8_t axis, int32_t vel_sps, uint32_t accel_sps2)
{
    if (axis >= STEPPER_NUM_AXES) return;
    Stepper_Status *s = &axis_st[axis];
    /* See Stepper_Move's comment on the motion-start kick -- vel_sps!=0 check
     * excludes a JOG 0 stop request from (harmlessly but pointlessly)
     * kicking current on an axis that's being told to decelerate, not start. */
    if (s->current_velocity_sps == 0 && vel_sps != 0) TMC5160_StartMotionKick(axis);
    s->jog_velocity = vel_sps;
    s->accel_sps2 = accel_sps2 ? accel_sps2 : 1;
    s->max_velocity_sps = (uint32_t)abs(vel_sps);
    s->moving = 1; /* supervisor tick clears this once decelerated to 0 for a JOG 0 stop */
}

void Stepper_Stop(uint8_t axis)
{
    if (axis >= STEPPER_NUM_AXES) return;
    Stepper_Status *s = &axis_st[axis];
    s->jog_velocity = 0;
    s->target = s->position; /* supervisor tick will decelerate to a stop, not snap to 0 */
}

uint8_t Stepper_Home(uint8_t axis, int8_t direction, uint32_t vel_sps, uint32_t accel_sps2,
                      uint32_t timeout_ms)
{
    if (axis >= STEPPER_NUM_AXES || !TMC5160_HasAxis(axis)) return 0;
    Stepper_Status *s = &axis_st[axis];
    if (!s->enabled) return 0;

    /* Grace period before polling trusts the stall flag: at the moment this is called the
     * axis may already be sitting stopped against a hard limit from a PREVIOUS home in the
     * opposite direction, and StallGuard2 has no fresh, reliable data below full speed -- at
     * true standstill it just holds whatever it last computed (e.g. "stalled" from having
     * been driven into that same limit), and during the low-speed part of acceleration the
     * back-EMF signal it depends on is still weak/marginal/noisy. Computed from this call's
     * own accel profile (time to reach vel_sps at accel_sps2) rather than a fixed guess,
     * since callers can pass very different speed/accel combinations, plus an 800ms margin
     * (originally 200ms -- confirmed on real hardware, J0, 2026-08-23, that 200ms lets the
     * grace period expire right as the axis is JUST reaching vel_sps, i.e. right at the
     * tail of the ramp's noisy low-speed regime rather than safely after it: every first
     * poll landed there and falsely tripped, at a near-identical step count across multiple
     * SGT values and even after a 5s idle settling pause, ruling out both a sensitivity
     * problem and stale/leftover data as the cause). 800ms gives real time running at
     * steady cruise speed, where SG_RESULT is known to behave, before the first poll can
     * count. */
    uint32_t ramp_ms = (accel_sps2 > 0) ? (vel_sps * 1000U) / accel_sps2 : 0;
    axis_homing_grace_until_ms[axis] = HAL_GetTick() + ramp_ms + 800;

    s->homing_result = 1; /* armed -- Stepper_HomingPollTick now polls this axis */
    axis_homing_deadline_ms[axis] = HAL_GetTick() + timeout_ms;
    Stepper_Jog(axis, direction >= 0 ? (int32_t)vel_sps : -(int32_t)vel_sps, accel_sps2);
    return 1;
}

void Stepper_DiagIRQ(uint8_t axis)
{
    if (axis >= STEPPER_NUM_AXES) return;
    Stepper_Status *s = &axis_st[axis];
    if (s->homing_result != 1) return; /* not (or no longer) homing -- ignore */

    Stepper_Stop(axis);
    s->homing_result = 2; /* stalled */
}

#define HOMING_POLL_PERIOD_MS 3 /* SPI read of DRV_STATUS takes a few ms (bit-banged, two
                                    5-byte transactions -- see TMC5160_ReadReg's comment on
                                    the read pipeline); rate-limit so polling several homing
                                    axes at once doesn't monopolize the main loop. Runs from
                                    the main loop, not an ISR, so this never blocks step
                                    pulses on any axis -- those are hardware-timer-driven and
                                    always preempt main-loop code regardless of how long a
                                    poll takes. */

void Stepper_HomingPollTick(void)
{
    static uint32_t last_poll_ms[STEPPER_NUM_AXES] = {0};
    uint32_t now = HAL_GetTick();

    for (uint8_t a = 0; a < STEPPER_NUM_AXES; a++) {
        Stepper_Status *s = &axis_st[a];
        if (s->homing_result != 1) continue; /* not currently homing */
        if ((int32_t)(now - axis_homing_grace_until_ms[a]) < 0) continue; /* stale stall flag
                                                                              from before this
                                                                              move started --
                                                                              see Stepper_Home's
                                                                              comment */
        if ((uint32_t)(now - last_poll_ms[a]) < HOMING_POLL_PERIOD_MS) continue;
        last_poll_ms[a] = now;

        uint32_t drv_status = TMC5160_ReadReg(a, TMC5160_REG_DRV_STATUS);
        if (s->homing_result != 1) continue; /* the poll itself took a few ms -- re-check
                                                  in case a timeout fired while we were in it */
        if (drv_status & TMC5160_DRV_STATUS_STALLGUARD) {
            /* Single-poll trigger -- was briefly a 3-consecutive-poll debounce (added
             * 2026-08-24 alongside the 800ms grace-period extension, both aimed at the
             * same ramp-noise false-positive bug), reverted here. This session's only
             * confirmed-genuine stall detections (before today, same SGT=17/8000sps
             * config) both had a noticeable real lag before triggering -- the user
             * described "kept grinding for a while" and confirmed homing_result==2 was
             * still a real stall, not a fluke. That's the signature of a signal that's
             * intermittent/noisy even during a genuine stall, not just during the ramp --
             * which a 3-in-a-row requirement could plausibly prevent from ever completing.
             * The grace-period extension alone should still cover the original ramp-noise
             * case; the debounce was very possibly net-harmful. Re-add/re-tune from here
             * if ramp-noise false positives reappear, rather than assuming more debounce
             * is always safer. */
            Stepper_Stop(a);
            s->homing_result = 2; /* stalled */
        }
    }
}

void Stepper_StopAll(void)
{
    for (uint8_t a = 0; a < STEPPER_NUM_AXES; a++) Stepper_Stop(a);
}

uint8_t Stepper_GetStatus(uint8_t axis, Stepper_Status *out)
{
    if (axis >= STEPPER_NUM_AXES || !out) return 0;
    *out = axis_st[axis];
    return 1;
}

/* Reprogram axis timer to pulse at `sps` steps/sec (0 = stop the timer).
 *
 * Bug fixed here: this is called every 1ms from Stepper_SupervisorTick()
 * for every enabled/moving axis, including every tick of a smooth
 * accel/decel ramp, not just when the rate actually needs to change. The
 * original version unconditionally reset the counter to 0 on every call --
 * which, for any commanded rate under ~1000 steps/sec (i.e. a period longer
 * than the 1ms tick that keeps calling this), means the counter was wiped
 * back to 0 before it could ever count up to ARR and fire the update event.
 * STEP never toggled at all for any realistic step rate as a result --
 * confirmed on real hardware via TMC2209's MSCNT and this firmware's own
 * Stepper_TimerIRQ-incremented position both staying frozen during an
 * active JOG. Only reset+(re)start the counter on the stopped->running
 * transition; while already running, just update the period and let the
 * counter keep counting instead of restarting it every tick. */
static void set_axis_rate(uint8_t axis, uint32_t sps)
{
    TIM_HandleTypeDef *htim = &axis_tim[axis];
    if (sps == 0) {
        HAL_TIM_Base_Stop_IT(htim);
        axis_running[axis] = 0;
        return;
    }
    /* Aim for an ARR in a comfortable mid-range for resolution; pick PSC so
     * tick_freq / sps lands near 1000-60000 counts. */
    uint32_t psc = TIMER_KERNEL_CLOCK_HZ / (sps * 2000UL);
    if (psc < 1) psc = 1;
    if (psc > 65535) psc = 65535;
    uint32_t tick_freq = TIMER_KERNEL_CLOCK_HZ / psc;
    uint32_t arr = tick_freq / sps;
    if (arr < 2) arr = 2;
    if (arr > 65535) arr = 65535;

    __HAL_TIM_SET_PRESCALER(htim, psc - 1);
    __HAL_TIM_SET_AUTORELOAD(htim, arr - 1);
    if (!axis_running[axis]) {
        __HAL_TIM_SET_COUNTER(htim, 0);
        HAL_TIM_Base_Start_IT(htim);
        axis_running[axis] = 1;
    }
}

void Stepper_TimerIRQ(uint8_t axis)
{
    if (axis >= STEPPER_NUM_AXES) return;
    const AxisHw_t *hw = &axis_hw[axis];
    Stepper_Status *s = &axis_st[axis];

    int8_t dir = 0;
    if (s->jog_velocity != 0) dir = (s->jog_velocity > 0) ? 1 : -1;
    else if (s->target != s->position) dir = (s->target > s->position) ? 1 : -1;
    if (dir == 0) return;

    uint8_t dir_level = (dir > 0) ? 1 : 0;
    if (s->invert_dir) dir_level ^= 1;
    HAL_GPIO_WritePin(hw->dir.port, hw->dir.pin, dir_level ? GPIO_PIN_SET : GPIO_PIN_RESET);

    HAL_GPIO_WritePin(hw->step.port, hw->step.pin, GPIO_PIN_SET);
    for (volatile uint32_t i = 0; i < STEP_PULSE_NOPS; i++) { __NOP(); }
    HAL_GPIO_WritePin(hw->step.port, hw->step.pin, GPIO_PIN_RESET);

    s->position += dir;
}

void Stepper_SupervisorTick(void)
{
    for (uint8_t a = 0; a < STEPPER_NUM_AXES; a++) {
        Stepper_Status *s = &axis_st[a];
        if (!s->enabled) continue;

        /* Homing timeout: signed-difference comparison rather than a plain
         * >= so this stays correct across HAL_GetTick()'s eventual uint32_t
         * wraparound (~49.7 days uptime), not just for the deadlines this
         * feature will actually see. */
        if (s->homing_result == 1 &&
            (int32_t)(HAL_GetTick() - axis_homing_deadline_ms[a]) >= 0) {
            Stepper_Stop(a);
            s->homing_result = 3; /* timed out, no stall seen */
        }

        int32_t remaining = s->jog_velocity ? 0x7FFFFFF : (s->target - s->position);
        uint32_t target_speed = s->max_velocity_sps;

        if (s->jog_velocity == 0 && remaining == 0 && s->current_velocity_sps == 0) {
            s->moving = 0;
            continue;
        }

        /* Distance needed to decelerate to 0 from current speed. */
        uint32_t decel_dist = (s->current_velocity_sps * s->current_velocity_sps) /
                               (2U * (s->accel_sps2 ? s->accel_sps2 : 1U));
        uint32_t abs_remaining = (uint32_t)((remaining < 0) ? -remaining : remaining);

        uint32_t step_v = s->accel_sps2 / SUPERVISOR_HZ;
        if (step_v == 0) step_v = 1;

        if (s->jog_velocity == 0 && abs_remaining <= decel_dist) {
            /* Decelerate */
            s->current_velocity_sps = (s->current_velocity_sps > step_v)
                                           ? (s->current_velocity_sps - step_v) : 0;
        } else if (s->current_velocity_sps < target_speed) {
            s->current_velocity_sps += step_v;
            if (s->current_velocity_sps > target_speed) s->current_velocity_sps = target_speed;
        } else if (s->current_velocity_sps > target_speed) {
            s->current_velocity_sps = (s->current_velocity_sps > step_v)
                                           ? (s->current_velocity_sps - step_v) : target_speed;
        }

        s->moving = (s->current_velocity_sps != 0) || (remaining != 0);
        set_axis_rate(a, s->current_velocity_sps);
    }
}
