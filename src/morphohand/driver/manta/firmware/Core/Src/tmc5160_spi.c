/*
 * Bit-banged SPI for TMC5160 drivers, mode 3 (CPOL=1, CPHA=1 -- idle clock
 * high, data setup on the falling edge, sampled on the rising edge),
 * matching the TMC5160 datasheet's SPI interface section.
 *
 * MOSI/SCK/MISO are one bus shared across every axis's socket (PG6/PG8/PG7
 * -- see generic-bigtreetech-manta-m8p-V2_0.cfg's commented [tmc2130
 * stepper_y] block; confirmed via the datasheet these pins have no plain
 * SPI-peripheral AF on this chip, only OCTOSPI-manager functions, which is
 * why Klipper also bit-bangs this bus rather than using a hardware
 * peripheral). Only the per-axis CS pin differs, matching each axis's "CS"
 * row from the manual's M1-M8 pin table -- currently only J0 (PC13) and J1
 * (PE3) are populated below; add more axes here as they get wired up.
 *
 * Datagram format (5 bytes, MSB first): byte0 = 7-bit register address with
 * bit7 = write(1)/read(0); bytes1-4 = 32-bit data (write) or don't-care
 * (read request). The reply is pipelined by one transaction -- the data
 * clocked out during a given transaction is the PREVIOUS transaction's
 * addressed register, not this one's. TMC5160_ReadReg sends the same read
 * request twice to get a same-call, correct result at the cost of a second
 * transaction. Originally only ever called at setup/config time -- that's
 * no longer true, Stepper_HomingPollTick() (stepper.c) now calls
 * TMC5160_ReadReg repeatedly from the main loop during homing, which is
 * exactly why spi_transact() masks interrupts below.
 */

#include "tmc5160_spi.h"
#include "stm32h7xx_hal.h"
#include "stepper.h" /* for STEPPER_NUM_AXES */

#define TMC5160_MOSI_PORT GPIOG
#define TMC5160_MOSI_PIN  GPIO_PIN_6
#define TMC5160_MISO_PORT GPIOG
#define TMC5160_MISO_PIN  GPIO_PIN_7
#define TMC5160_SCK_PORT  GPIOG
#define TMC5160_SCK_PIN   GPIO_PIN_8

typedef struct {
    GPIO_TypeDef *port;
    uint16_t      pin;
} TmcCsPin_t;

/* NULL port = no TMC5160 wired up on this axis's socket (still bit-bang
 * UART/TMC2209 or unpopulated) -- TMC5160_ReadReg/WriteReg are no-ops for
 * those. Pins from the manual's M1-M8 "CS" row, same table used for the
 * UART axes' CS/PDN_UART pin (same physical net either way -- see
 * generic-bigtreetech-manta-m8p-V2_0.cfg's per-axis cs_pin lines). Only
 * J0-J5 populated: this project needs 6 axes (see main.c/README), J6/J7
 * left NULL rather than guessed. */
static const TmcCsPin_t tmc5160_cs[STEPPER_NUM_AXES] = {
    /* J0 */ { GPIOC, GPIO_PIN_13 },
    /* J1 */ { GPIOE, GPIO_PIN_3  },
    /* J2 */ { GPIOB, GPIO_PIN_9  },
    /* J3 */ { GPIOB, GPIO_PIN_5  },
    /* J4 */ { GPIOG, GPIO_PIN_14 },
    /* J5 */ { GPIOG, GPIO_PIN_10 },
    /* J6 */ { NULL,  0 },
    /* J7 */ { NULL,  0 },
};

/* Was 20000 (~10kHz bit rate, ~8ms per TMC5160_ReadReg's two-transaction
 * pipelined read) for a conservative first bring-up -- that comment said
 * "tighten later once basic reads are confirmed working", and this is that
 * moment. Confirmed on real hardware (J0, 2026-08-24) that 8ms was actually
 * the root cause of the day's entire stall-detection saga, not any SGT/
 * current/TCOOLTHRS tuning: Stepper_HomingPollTick() polls every 3ms during
 * active homing, and each poll's spi_transact() masks the step-timer and
 * 1kHz supervisor-tick interrupts (BASEPRI, see spi_transact's comment) for
 * the read's full duration -- at 8ms per read against a 3ms poll period,
 * that's ~90% masked, which was starving BOTH this axis's own step
 * generation (observed effective rate ~850sps against an 8000sps command)
 * and the supervisor tick's homing-timeout check (which is why timeouts
 * stopped firing too, not just stalls). 800000 brings a read down to
 * ~200us, leaving real unmasked time between 3ms-spaced polls. */
#define SPI_HALF_BIT_CYCLES_DIVISOR 800000U

static inline void spi_delay(void)
{
    uint32_t cycles = SystemCoreClock / SPI_HALF_BIT_CYCLES_DIVISOR;
    uint32_t start = DWT->CYCCNT;
    while ((uint32_t)(DWT->CYCCNT - start) < cycles) { }
}

static void gpio_clock_enable_for_port(GPIO_TypeDef *port)
{
    if (port == GPIOB) __HAL_RCC_GPIOB_CLK_ENABLE();
    else if (port == GPIOC) __HAL_RCC_GPIOC_CLK_ENABLE();
    else if (port == GPIOD) __HAL_RCC_GPIOD_CLK_ENABLE();
    else if (port == GPIOE) __HAL_RCC_GPIOE_CLK_ENABLE();
    else if (port == GPIOG) __HAL_RCC_GPIOG_CLK_ENABLE();
}

void TMC5160_Init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    __HAL_RCC_GPIOG_CLK_ENABLE();

    GPIO_InitTypeDef gi = {0};
    gi.Mode = GPIO_MODE_OUTPUT_PP;
    gi.Pull = GPIO_NOPULL;
    gi.Speed = GPIO_SPEED_FREQ_LOW;

    gi.Pin = TMC5160_MOSI_PIN;
    HAL_GPIO_Init(TMC5160_MOSI_PORT, &gi);

    gi.Pin = TMC5160_SCK_PIN;
    HAL_GPIO_Init(TMC5160_SCK_PORT, &gi);
    HAL_GPIO_WritePin(TMC5160_SCK_PORT, TMC5160_SCK_PIN, GPIO_PIN_SET); /* mode 3: idle high */

    gi.Mode = GPIO_MODE_INPUT;
    gi.Pull = GPIO_PULLDOWN; /* defined level if CS deselected/module absent */
    gi.Pin = TMC5160_MISO_PIN;
    HAL_GPIO_Init(TMC5160_MISO_PORT, &gi);

    for (uint8_t a = 0; a < STEPPER_NUM_AXES; a++) {
        const TmcCsPin_t *cs = &tmc5160_cs[a];
        if (cs->port == NULL) continue;

        gpio_clock_enable_for_port(cs->port);
        gi.Mode = GPIO_MODE_OUTPUT_PP;
        gi.Pull = GPIO_NOPULL;
        gi.Pin = cs->pin;
        HAL_GPIO_Init(cs->port, &gi);
        HAL_GPIO_WritePin(cs->port, cs->pin, GPIO_PIN_SET); /* idle deselected */
    }

    /* CHOPCONF resets to TOFF=0 (driver stage fully disabled) on every TMC5160
     * -- confirmed on real hardware that a freshly-populated axis (J1) sits
     * completely silent/motionless at that reset value despite otherwise-normal
     * STEP/DIR/EN signaling. Push a known-good CHOPCONF and a current-limited
     * IHOLD_IRUN to every populated axis here so this doesn't have to get
     * rediscovered per-axis. See the constants' doc comments in tmc5160_spi.h. */
    for (uint8_t a = 0; a < STEPPER_NUM_AXES; a++) {
        if (tmc5160_cs[a].port == NULL) continue;
        TMC5160_WriteReg(a, TMC5160_REG_CHOPCONF, TMC5160_DEFAULT_CHOPCONF);
        TMC5160_WriteReg(a, TMC5160_REG_TCOOLTHRS, TMC5160_DEFAULT_TCOOLTHRS);
        TMC5160_SetRunCurrent(a, TMC5160_DEFAULT_IHOLD_IRUN); /* also remembers it for
                                                                   TMC5160_StartMotionKick
                                                                   to restore to */
    }
}

/* Same BASEPRI-masking pattern already used in tmc2209_uart.c's
 * sw_uart_transact() (see that file for the __NVIC_PRIO_BITS/priority-value
 * reasoning) -- and the same real-hardware motivation: this bit-banged
 * transaction's timing comes from cycle-counted spin-waits (spi_delay()),
 * with nothing else guarding it. During active homing at 8000sps, this
 * axis's own STEP timer ISR fires every 125us; if it preempts mid-bit here,
 * it corrupts the transaction's timing and can hand back garbage in place
 * of a real DRV_STATUS read. Confirmed on real hardware (J0, 2026-08-23):
 * StallGuard's poll-based stall detection failed to register three separate,
 * user-confirmed real hardstop contacts in a row, across every current/SGT
 * combination tried -- consistent with the read itself being unreliable
 * during active motion rather than a sensitivity/tuning problem. The DIAG
 * EXTI path this priority scheme was originally designed around is unused
 * dead code now (see Stepper_Home's doc comment in stepper.h), so there's
 * no longer a reason to leave priority 0 unmasked here the way the UART
 * driver still does -- but reusing the identical mask keeps both bit-banged
 * drivers' interrupt behavior easy to reason about together. */
#define SPI_BASEPRI_MASK 0x10U

static void spi_transact(const TmcCsPin_t *cs, const uint8_t *tx, uint8_t *rx)
{
    uint32_t saved_basepri = __get_BASEPRI();
    __set_BASEPRI(SPI_BASEPRI_MASK);

    HAL_GPIO_WritePin(cs->port, cs->pin, GPIO_PIN_RESET);
    spi_delay();

    for (uint8_t b = 0; b < 5; b++) {
        uint8_t txbyte = tx[b];
        uint8_t rxbyte = 0;

        for (int8_t bit = 7; bit >= 0; bit--) {
            HAL_GPIO_WritePin(TMC5160_MOSI_PORT, TMC5160_MOSI_PIN,
                               (txbyte & (1U << bit)) ? GPIO_PIN_SET : GPIO_PIN_RESET);
            HAL_GPIO_WritePin(TMC5160_SCK_PORT, TMC5160_SCK_PIN, GPIO_PIN_RESET); /* leading edge: data setup */
            spi_delay();

            HAL_GPIO_WritePin(TMC5160_SCK_PORT, TMC5160_SCK_PIN, GPIO_PIN_SET); /* trailing edge: sample */
            if (HAL_GPIO_ReadPin(TMC5160_MISO_PORT, TMC5160_MISO_PIN) == GPIO_PIN_SET) {
                rxbyte = (uint8_t)(rxbyte | (1U << bit));
            }
            spi_delay();
        }
        rx[b] = rxbyte;
    }

    HAL_GPIO_WritePin(cs->port, cs->pin, GPIO_PIN_SET);
    spi_delay();

    __set_BASEPRI(saved_basepri);
}

uint32_t TMC5160_ReadReg(uint8_t axis, uint8_t reg)
{
    if (axis >= STEPPER_NUM_AXES || tmc5160_cs[axis].port == NULL) return 0;
    const TmcCsPin_t *cs = &tmc5160_cs[axis];

    uint8_t tx[5] = { (uint8_t)(reg & 0x7F), 0, 0, 0, 0 };
    uint8_t rx[5];

    spi_transact(cs, tx, rx);   /* primes the pipeline */
    spi_transact(cs, tx, rx);   /* this reply now holds reg's data */

    return ((uint32_t)rx[1] << 24) | ((uint32_t)rx[2] << 16) |
           ((uint32_t)rx[3] << 8) | (uint32_t)rx[4];
}

void TMC5160_WriteReg(uint8_t axis, uint8_t reg, uint32_t data)
{
    if (axis >= STEPPER_NUM_AXES || tmc5160_cs[axis].port == NULL) return;
    const TmcCsPin_t *cs = &tmc5160_cs[axis];

    uint8_t tx[5] = {
        (uint8_t)(reg | 0x80),
        (uint8_t)(data >> 24), (uint8_t)(data >> 16), (uint8_t)(data >> 8), (uint8_t)data
    };
    uint8_t rx[5];

    spi_transact(cs, tx, rx);
}

uint8_t TMC5160_HasAxis(uint8_t axis)
{
    return (axis < STEPPER_NUM_AXES && tmc5160_cs[axis].port != NULL) ? 1 : 0;
}

/* See TMC5160_SetRunCurrent/TMC5160_StartMotionKick/TMC5160_KickPollTick's
 * doc comments in tmc5160_spi.h. axis_run_ihold_irun holds what to restore
 * to after a kick -- deliberately separate from whatever's live on the chip
 * at any given moment (which TMC5160_WriteReg can't read back anyway, since
 * IHOLD_IRUN is write-only on this part). */
static uint32_t axis_run_ihold_irun[STEPPER_NUM_AXES];
static uint32_t axis_kick_until_ms[STEPPER_NUM_AXES];
static uint8_t  axis_kick_active[STEPPER_NUM_AXES];

void TMC5160_SetRunCurrent(uint8_t axis, uint32_t ihold_irun)
{
    if (axis >= STEPPER_NUM_AXES) return;
    axis_run_ihold_irun[axis] = ihold_irun;
    TMC5160_WriteReg(axis, TMC5160_REG_IHOLD_IRUN, ihold_irun);
}

void TMC5160_StartMotionKick(uint8_t axis)
{
    if (!TMC5160_HasAxis(axis)) return;
    TMC5160_WriteReg(axis, TMC5160_REG_IHOLD_IRUN, TMC5160_KICK_IHOLD_IRUN);
    axis_kick_until_ms[axis] = HAL_GetTick() + TMC5160_KICK_DURATION_MS;
    axis_kick_active[axis] = 1;
}

void TMC5160_KickPollTick(void)
{
    uint32_t now = HAL_GetTick();
    for (uint8_t a = 0; a < STEPPER_NUM_AXES; a++) {
        if (!axis_kick_active[a]) continue;
        if ((int32_t)(now - axis_kick_until_ms[a]) < 0) continue;
        TMC5160_WriteReg(a, TMC5160_REG_IHOLD_IRUN, axis_run_ihold_irun[a]);
        axis_kick_active[a] = 0;
    }
}
