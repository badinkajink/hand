#ifndef STEPPER_H
#define STEPPER_H

#include <stdint.h>
#include "stm32h7xx_hal.h"

#define STEPPER_NUM_AXES 8

/* homing_result: 0=idle (never homed / not currently homing), 1=homing in
 * progress, 2=stalled (StallGuard2 flag seen via SPI polling -- treat
 * `position` at the moment of the stall as the home reference; nothing here
 * auto-zeroes it, call ZERO explicitly if that's what you want), 3=timed out
 * with no stall seen (bad SGT/TCOOLTHRS tuning, no mechanical limit in this
 * direction, or a wiring problem -- position is NOT to be trusted as a home
 * reference). */
typedef struct {
    int32_t  position;    /* current step count, signed, open-loop */
    int32_t  target;      /* target step count for the active MOVE, if any */
    int32_t  jog_velocity; /* nonzero => continuous JOG in progress (steps/s, signed) */
    uint32_t max_velocity_sps;
    uint32_t accel_sps2;
    uint32_t current_velocity_sps; /* magnitude, current point on the trapezoid */
    uint8_t  moving;
    uint8_t  enabled;
    uint8_t  invert_dir;
    uint8_t  homing_result;
} Stepper_Status;

void Stepper_Init(void);
void Stepper_Enable(uint8_t axis);
void Stepper_Disable(uint8_t axis);
void Stepper_SetInvertDir(uint8_t axis, uint8_t invert);
void Stepper_Zero(uint8_t axis);
void Stepper_Move(uint8_t axis, int32_t abs_steps, uint32_t max_vel_sps, uint32_t accel_sps2);

/* Calibration for Stepper_MoveMM, RAM only (see that function's comment).
 * 0 (the power-on default) means uncalibrated. */
void Stepper_SetStepsPerMM(uint8_t axis, float steps_per_mm);

/* Same as Stepper_Move but takes an absolute position in mm, converted via
 * this axis's Stepper_SetStepsPerMM calibration -- 0mm is wherever this
 * axis's step-position 0 currently is (i.e. wherever Stepper_Zero/a HOME's
 * stall point last set it), so home before relying on this. Returns 0
 * (does nothing) if the axis has no calibration set yet or is out of range,
 * same convention as Stepper_Home. */
uint8_t Stepper_MoveMM(uint8_t axis, float mm, uint32_t max_vel_sps, uint32_t accel_sps2);

void Stepper_Jog(uint8_t axis, int32_t vel_sps, uint32_t accel_sps2);
void Stepper_Stop(uint8_t axis);
void Stepper_StopAll(void);
uint8_t Stepper_GetStatus(uint8_t axis, Stepper_Status *out);

/* Sensorless homing via the TMC5160's StallGuard2, read over SPI --
 * deliberately NOT the DIAG pin/EXTI interrupt: connecting DIAG to this
 * board's DIAG net cost us a TMC5160 module and, on a second attempt, a
 * whole STM32 (see the project history/memory for the full account) --
 * likely sustained current through the MCU pin's ESD protection from an
 * under-characterized external circuit. Stepper_HomingPollTick() polls
 * DRV_STATUS's stallguard flag from the main loop instead, over the same
 * CS/MOSI/MISO/SCK bus that's never caused any damage. Requires SGT
 * (COOLCONF, register 0x6D) and TCOOLTHRS (0x14) already configured and
 * empirically tuned on this axis first, and the axis present in
 * tmc5160_cs[] (tmc5160_spi.c) -- others return 0 and do nothing. Starts a
 * JOG at vel_sps in the given direction; the next poll tick that sees the
 * stall flag (or a timeout_ms deadline, whichever comes first) stops it and
 * sets homing_result. Non-blocking -- poll Stepper_GetStatus for the
 * result, same pattern as every other move command here. */
uint8_t Stepper_Home(uint8_t axis, int8_t direction, uint32_t vel_sps, uint32_t accel_sps2,
                      uint32_t timeout_ms);

/* Called from each axis's TIMx update-interrupt handler (stm32h7xx_it.c). */
void Stepper_TimerIRQ(uint8_t axis);

/* Called from a DIAG pin's EXTI interrupt handler (stm32h7xx_it.c). Currently
 * unused/dead code -- nothing is wired to any DIAG pin (see Stepper_Home's
 * doc comment for why) -- kept only in case a future, properly-characterized
 * DIAG connection makes interrupt-based detection safe to revisit. No-op if
 * this axis isn't currently homing -- safe to call unconditionally from an
 * ISR. */
void Stepper_DiagIRQ(uint8_t axis);

/* Poll every axis currently mid-Stepper_Home() call for StallGuard2's stall
 * flag over SPI, rate-limited internally so this is cheap to call every main
 * loop iteration. See Stepper_Home's doc comment for why this replaced
 * DIAG/EXTI-based detection. */
void Stepper_HomingPollTick(void);

/* Called from the 1kHz supervisory timer (TIM6) interrupt -- recomputes each
 * active axis's target step rate along its trapezoidal profile and rewrites
 * that axis's timer ARR/PSC accordingly. */
void Stepper_SupervisorTick(void);

#endif /* STEPPER_H */
