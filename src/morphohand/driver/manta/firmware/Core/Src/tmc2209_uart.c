/*
 * TMC2209 single-wire UART driver. All 8 axes currently use the software
 * bit-banged path (see the "Software bit-banged UART" section below) --
 * J0/J5 because their pins (PC13/PG10) have no hardware UART/USART
 * alternate function at all on the STM32H723 (confirmed against the
 * datasheet's Table 7); J1/J2/J4/J6/J7 because their datasheet-correct
 * hardware peripheral either hit a real-hardware dead end (J1/PE3: both
 * documented AFs for USART10 failed to even complete the half-duplex echo
 * readback; J2/PB9: same failure on UART4's assumed AF) or was never worth
 * testing given that pattern (J4/PG14, J6/PD5: the same category of
 * unverified "standard AF" guess that had already failed twice elsewhere),
 * or conflicted with another axis needing the same peripheral (J7/PC6 vs.
 * J4/PG14, both datasheet-showing USART6). The `TmcUartPin_t.inst`/`af`
 * fields for the hardware-UART machinery below are kept because nothing
 * rules out a future axis (or board revision) actually landing on a pin
 * with a working hardware AF -- a NULL `inst` in tmc_pin[] is what
 * currently marks every axis as software-driven instead.
 *
 * TMC2209 UART datagram format and CRC8 algorithm are from the TMC2209
 * datasheet ("UART interface" chapter) -- reproduced here, not invented.
 * Since each driver is on its own dedicated single-wire UART (not a shared
 * multi-drop bus), the address bytes are fixed constants (0x00 for
 * datagrams we send, drivers reply stamped with 0xFF) rather than a
 * configurable per-driver node address.
 *
 * Half-duplex quirk (hardware-UART axes only, currently none): STM32's
 * HAL_HalfDuplex UART mode loops transmitted bytes back onto its own RX --
 * so after sending a request, this driver must read and discard the echo
 * of what it just sent before capturing the TMC2209's actual reply. The
 * software bit-banged path has no such loopback (it only reads the pin
 * after it's done driving it), so it doesn't need an echo-discard step.
 */

#include <string.h>
#include "tmc2209_uart.h"
#include "tmc5160_spi.h"
#include "stm32h7xx_hal.h"
#include "stepper.h"  /* for STEPPER_NUM_AXES */

#define TMC_SYNC       0x05
#define TMC_ADDR_TX    0x00
#define TMC_RSENSE_OHM 0.11f   /* BTT TMC2209 module default; matches their Klipper cfg */
#define TMC_VFS_VOLTS  0.325f  /* VSENSE=0, POR default, not touched by this driver */
#define TMC_UART_TIMEOUT_MS 5

typedef struct {
    USART_TypeDef *inst;
    GPIO_TypeDef  *port;
    uint16_t       pin;
    uint8_t        af;
} TmcUartPin_t;

/* From docs/pinout.md's "TMC UART peripheral" column. inst == NULL means
 * "no hardware UART on this pin -- use the software bit-banged path". */
static const TmcUartPin_t tmc_pin[STEPPER_NUM_AXES] = {
    /* J0 */ { NULL, GPIOC, GPIO_PIN_13, 0 }, /* PC13: no hardware UART/USART AF exists on this
                                                   pin at all (datasheet Table 7 -- only EVENTOUT/
                                                   RTC_TAMP1/RTC_TS/WKUP4). Software bit-bang. */
    /* J1 */ { NULL, GPIOE, GPIO_PIN_3, 0 }, /* PE3's real peripheral is USART10 per the datasheet
                                                 (Table 7), but neither documented AF for USART10
                                                 (AF11, AF4) got even as far as a successful echo
                                                 readback on real hardware (rc=-4) -- confirmed via
                                                 live SWD register inspection that the MCU genuinely
                                                 never puts a signal on the physical pin, despite
                                                 USART10's own registers reporting a successful
                                                 transmit. Software bit-bang instead, same proven
                                                 path as J0. */
    /* J2 */ { NULL, GPIOB, GPIO_PIN_9, 0 }, /* PB9's real peripheral is UART4 per the datasheet
                                                 (Table 7: UART4_TX is in its function list, no
                                                 USART3 at all), but AF8 (the assumed "standard"
                                                 AF for UART4 on this chip) failed hardware test
                                                 the same way both USART10 guesses did for J1
                                                 (rc=-4, echo readback never completed). Rather
                                                 than try a third counted-from-the-garbled-PDF AF
                                                 guess -- a method already proven unreliable on J1
                                                 -- moved to software bit-bang like J0/J1/J5/J7. */
    /* J3 */ { NULL, GPIOB, GPIO_PIN_5, 0 }, /* UART5 is confirmed as PB5's peripheral in the
                                                 datasheet by name coincidence with the original
                                                 code, but only UART5_RX was ever confirmed in its
                                                 function list, not UART5_TX -- given AF8 has now
                                                 failed on two other pins that also listed their
                                                 UARTx_TX function plainly (J1, J2), and this pin's
                                                 TX capability was already the least-certain entry
                                                 in this whole table, not worth spending a real
                                                 hardware test on before falling back anyway.
                                                 Software bit-bang like the rest. */
    /* J4 */ { NULL, GPIOG, GPIO_PIN_14, 0 }, /* PG14's real peripheral is USART6 per the
                                                 datasheet (Table 7: USART6_TX in its function
                                                 list, no UART4), which also conflicts with J7/PC6
                                                 datasheet-showing the same USART6 -- moot now,
                                                 since AF7 (the assumed standard AF for USART6) is
                                                 the same category of guess that's already failed
                                                 twice (J1, J2). Software bit-bang, same as J7. */
    /* J5 */ { NULL, GPIOG, GPIO_PIN_10, 0 }, /* PG10: same situation as J0 -- full datasheet
                                                  function list for this pin has no UART/USART
                                                  entry at all. Software bit-bang, same as J0. */
    /* J6 */ { NULL, GPIOD, GPIO_PIN_5, 0 }, /* PD5's real peripheral is USART2 per the datasheet
                                                 (Table 7: USART2_TX in its function list, no
                                                 UART8), but AF7 (the assumed standard AF for
                                                 USART2) is the same category of guess that's
                                                 already failed twice (J1, J2) -- not worth a real
                                                 hardware test before falling back anyway. Software
                                                 bit-bang like the rest. */
    /* J7 */ { NULL, GPIOC, GPIO_PIN_6, 0 }, /* PC6 datasheet-shows USART6 (not the original
                                                 USART2 guess), but J4/PG14 already claims the
                                                 chip's only USART6 instance -- one physical
                                                 peripheral can't serve two pins/joints at once.
                                                 Software bit-bang instead, same proven approach as
                                                 J0/J5, rather than leaving this joint unusable. */
};

static UART_HandleTypeDef tmc_huart[STEPPER_NUM_AXES];

static uint8_t tmc_crc8(const uint8_t *data, uint8_t len)
{
    uint8_t crc = 0;
    for (uint8_t i = 0; i < len; i++) {
        uint8_t cur = data[i];
        for (uint8_t j = 0; j < 8; j++) {
            if ((crc >> 7) ^ (cur & 0x01)) {
                crc = (uint8_t)((crc << 1) ^ 0x07);
            } else {
                crc = (uint8_t)(crc << 1);
            }
            cur >>= 1;
        }
    }
    return crc;
}

static void gpio_clock_enable_for_port(GPIO_TypeDef *port)
{
    if (port == GPIOB) __HAL_RCC_GPIOB_CLK_ENABLE();
    else if (port == GPIOC) __HAL_RCC_GPIOC_CLK_ENABLE();
    else if (port == GPIOD) __HAL_RCC_GPIOD_CLK_ENABLE();
    else if (port == GPIOE) __HAL_RCC_GPIOE_CLK_ENABLE();
    else if (port == GPIOG) __HAL_RCC_GPIOG_CLK_ENABLE();
}

static void usart_clock_enable(USART_TypeDef *u)
{
    if (u == USART1) __HAL_RCC_USART1_CLK_ENABLE();
    else if (u == USART2) __HAL_RCC_USART2_CLK_ENABLE();
    else if (u == USART3) __HAL_RCC_USART3_CLK_ENABLE();
    else if (u == USART6) __HAL_RCC_USART6_CLK_ENABLE();
    else if (u == UART4) __HAL_RCC_UART4_CLK_ENABLE();
    else if (u == UART5) __HAL_RCC_UART5_CLK_ENABLE();
    else if (u == UART7) __HAL_RCC_UART7_CLK_ENABLE();
    else if (u == UART8) __HAL_RCC_UART8_CLK_ENABLE();
    else if (u == USART10) __HAL_RCC_USART10_CLK_ENABLE();
}

/* ---- Software bit-banged UART (currently all 8 axes -- see the file
 * header comment for why each one ended up here) ----
 *
 * Plain GPIO, open-drain + pull-up (same electrical setup TMC2209's
 * single-wire PDN_UART needs regardless of who's driving it). Bit timing
 * comes from the Cortex-M7's DWT cycle counter, which is free-running
 * hardware present on every M7 core -- no extra timer peripheral needed.
 *
 * ~2400 baud (see SW_UART_BIT_US below) rather than the originally-planned
 * hardware-UART axes' 115200: TMC2209 autobauds off the sync byte's edges,
 * so any reasonable rate works, and a slower rate gives a busy-wait
 * bit-bang loop much more timing margin.
 *
 * A full datagram transaction runs with interrupts masked via BASEPRI (see
 * sw_uart_transact), NOT __disable_irq()/PRIMASK, so the 1kHz step-profile
 * tick (priority 2) and the supervisor tick (priority 1) can't jitter bit
 * timing mid-byte -- but a DIAG stall interrupt (priority 0, the highest on
 * this chip, set in Stepper_Init) can still preempt an in-progress UART
 * transaction and stop the motor immediately. This is deliberate: sensorless
 * homing's whole point is reacting to a stall the instant it happens, and a
 * corrupted UART byte is cheap (the caller just sees a bad reply/timeout and
 * can retry) next to missing a stall because some *other* axis's config
 * write happened to be in flight. PRIMASK-based masking (the original
 * approach) blocked DIAG too, since it disables every exception regardless
 * of priority -- BASEPRI masks by priority level instead, so DIAG's
 * priority-0 slot is deliberately left outside what gets blocked.
 *
 * This still stalls step pulses on every axis, not just its own, for the
 * duration of a config transaction (worst case ~15ms for a ReadReg's 4-byte
 * write + 8-byte reply) -- don't call TMC_SetCurrent/TMC_SetMicrosteps for
 * any joint while another axis is actively moving. */
#define SW_UART_BIT_US           417U   /* ~2400 baud -- slowed from ~9600 (104us) after J2/PB9
                                            showed a perfectly reproducible stuck-at-1 bit in its
                                            write datagrams (same bit position every time,
                                            regardless of data), while J0/PC13 ran the identical
                                            code cleanly at the faster rate -- consistent with a
                                            timing margin that's fine on one pin's electrical
                                            characteristics but marginal on another's. TMC2209
                                            autobauds off the sync byte's edges, so any reasonable
                                            rate works; this just trades a bit of latency (config
                                            commands only, not the hot path) for a lot more margin. */
/* start-bit wait per received byte. Every single TMC read (RREG, any axis,
 * any register, even fully idle) has been observed failing with rc=-1 (no
 * start bit ever seen) on real hardware. Widening this made no difference
 * at any value tried, including an extreme 2,000,000 (2 full seconds,
 * elapsed-time-confirmed via the host round trip) -- ruling out SENDDELAY
 * margin, or any timing explanation at all: the TMC2209 either genuinely
 * never replies within a window far longer than any plausible real
 * turnaround time, or something earlier in this receive path never
 * observes it even when it's sent. Root cause NOT YET FOUND -- writes
 * appear to work fine (motor motion, current changes, all visibly
 * correct), so whatever's wrong is specific to the receive/reply side, not
 * a general link failure. Left at a middling value here: enough margin to
 * rule out a merely-tight window on some future axis, without a broken
 * read stalling every other interrupt (steppers, USB) for seconds at a
 * time. Don't spend more time adjusting this specific number -- that knob
 * has been fully explored and isn't the answer. */
#define SW_UART_BYTE_TIMEOUT_US  15000U

static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

/* Busy-wait until DWT->CYCCNT reaches (anchor + offset_cycles) -- an
 * ABSOLUTE point on the timeline, not a relative "wait N more cycles from
 * whenever this happens to be called" like a chained delay_us() loop would
 * give. That distinction is the whole point: see the comment above
 * sw_uart_send_bytes for why relative delays were a real, reproducible bug
 * here, not just a theoretical concern. */
static inline void wait_until(uint32_t anchor, uint32_t offset_cycles)
{
    while ((uint32_t)(DWT->CYCCNT - anchor) < offset_cycles) { }
}

/* Send a full multi-byte datagram (start/8-data/stop framing per byte, back
 * to back with no inter-byte gap) against a SINGLE time anchor taken once
 * at the very start of the whole transmission, with every bit edge's target
 * time computed as an absolute offset from that one anchor.
 *
 * The original version called a per-byte send function in a loop, and that
 * function timed each bit with delay_us() -- which restarts its own
 * reference point on every call. The real, uncompensated overhead of each
 * HAL_GPIO_WritePin() call and loop bookkeeping between delay_us() calls
 * was never accounted for, so it just kept accumulating, bit after bit,
 * byte after byte, with nothing to reset it until the whole transmission
 * ended. On real hardware this reproducibly corrupted the exact same bit
 * position (the first data bit of the 4th byte -- ~30 bit-periods deep into
 * a continuous 80-bit-period 8-byte write) on more than one axis, which is
 * the signature of systematic drift, not per-pin electrical noise: the
 * failure point was identical regardless of which physical pin was
 * involved. A single shared anchor with absolute per-bit targets can't
 * accumulate error this way -- whatever overhead happens between bits just
 * eats into the slack before the next absolute target, instead of pushing
 * that target itself later. */
static void sw_uart_send_bytes(GPIO_TypeDef *port, uint16_t pin, const uint8_t *data, uint8_t len)
{
    uint32_t bit_cycles = SW_UART_BIT_US * (SystemCoreClock / 1000000U);
    uint32_t anchor = DWT->CYCCNT;
    uint32_t n = 0; /* bit-period index since anchor, across the whole multi-byte send */

    for (uint8_t b = 0; b < len; b++) {
        uint8_t byte = data[b];

        wait_until(anchor, n * bit_cycles);
        HAL_GPIO_WritePin(port, pin, GPIO_PIN_RESET); /* start bit */
        n++;

        for (uint8_t i = 0; i < 8; i++) {
            wait_until(anchor, n * bit_cycles);
            HAL_GPIO_WritePin(port, pin, (byte & (1U << i)) ? GPIO_PIN_SET : GPIO_PIN_RESET);
            n++;
        }

        wait_until(anchor, n * bit_cycles);
        HAL_GPIO_WritePin(port, pin, GPIO_PIN_SET); /* stop bit */
        n++;
    }
    wait_until(anchor, n * bit_cycles); /* hold through the last stop bit before returning */
}

/* Same anchored-absolute-timing approach for receive, scoped to one byte:
 * each byte in a multi-byte reply gets its own fresh anchor at its actual
 * observed start-bit edge (the reply's real electrical signal, not
 * something we control), so there's no equivalent whole-transaction drift
 * to accumulate here the way there was on the send side -- but the 8
 * within-byte sample points still need to be absolute offsets from that
 * edge rather than chained delay_us() calls, for the same reason. */
static int sw_uart_recv_byte(GPIO_TypeDef *port, uint16_t pin, uint8_t *out)
{
    uint32_t bit_cycles = SW_UART_BIT_US * (SystemCoreClock / 1000000U);
    uint32_t timeout_cycles = SW_UART_BYTE_TIMEOUT_US * (SystemCoreClock / 1000000U);
    uint32_t t0 = DWT->CYCCNT;
    while (HAL_GPIO_ReadPin(port, pin) == GPIO_PIN_SET) {
        if ((uint32_t)(DWT->CYCCNT - t0) > timeout_cycles) return -1; /* no start bit */
    }
    uint32_t anchor = DWT->CYCCNT; /* the real start-bit falling edge */

    uint8_t byte = 0;
    for (uint8_t i = 0; i < 8; i++) {
        /* land mid-bit, not on the edge: 0.5 bit periods to the center of
         * the start bit, plus (i+1) more full periods to the center of
         * data bit i. */
        wait_until(anchor, bit_cycles / 2 + (uint32_t)(i + 1) * bit_cycles);
        if (HAL_GPIO_ReadPin(port, pin) == GPIO_PIN_SET) byte |= (uint8_t)(1U << i);
    }
    wait_until(anchor, bit_cycles / 2 + 9U * bit_cycles); /* stop bit -- not checked */
    *out = byte;
    return 0;
}

/* Raw NVIC priority register value for "preempt priority 1" on this chip
 * (__NVIC_PRIO_BITS == 4, all 4 bits allocated to preempt priority since
 * nothing here ever calls HAL_NVIC_SetPriorityGrouping -- the reset default,
 * NVIC_PRIORITYGROUP_4, is what's in effect, matching every
 * HAL_NVIC_SetPriority(irq, N, 0) call in stepper.c). BASEPRI masks any
 * exception whose priority register value is numerically >= BASEPRI, so
 * 0x10 blocks priority 1 (0x10, the supervisor tick) and priority 2 (0x20,
 * the step timers) while leaving priority 0 (0x00, DIAG) unmasked. */
#define UART_BASEPRI_MASK 0x10U

static int sw_uart_transact(const TmcUartPin_t *p, const uint8_t *tx, uint8_t txlen,
                             uint8_t *rx, uint8_t rxlen)
{
    uint32_t saved_basepri = __get_BASEPRI();
    __set_BASEPRI(UART_BASEPRI_MASK);

    sw_uart_send_bytes(p->port, p->pin, tx, txlen);

    int rc = 0;
    for (uint8_t i = 0; i < rxlen; i++) {
        if (sw_uart_recv_byte(p->port, p->pin, &rx[i]) != 0) { rc = -1; break; }
    }

    __set_BASEPRI(saved_basepri);
    return rc;
}

static int tmc_write_verified(uint8_t axis, uint8_t reg, uint32_t data);

void TMC_Init(void)
{
    dwt_init();

    for (uint8_t a = 0; a < STEPPER_NUM_AXES; a++) {
        const TmcUartPin_t *p = &tmc_pin[a];

        gpio_clock_enable_for_port(p->port);

        GPIO_InitTypeDef gi = {0};
        gi.Pin = p->pin;
        gi.Pull = GPIO_PULLUP;
        gi.Speed = GPIO_SPEED_FREQ_HIGH;

        if (p->inst == NULL) {
            /* Software bit-banged axis: plain open-drain GPIO, no AF. */
            gi.Mode = GPIO_MODE_OUTPUT_OD;
            HAL_GPIO_Init(p->port, &gi);
            HAL_GPIO_WritePin(p->port, p->pin, GPIO_PIN_SET); /* idle high */
            continue;
        }

        usart_clock_enable(p->inst);

        gi.Mode = GPIO_MODE_AF_OD; /* open-drain: matches TMC2209 PDN_UART single-wire usage */
        gi.Alternate = p->af;
        HAL_GPIO_Init(p->port, &gi);

        UART_HandleTypeDef *h = &tmc_huart[a];
        h->Instance = p->inst;
        /* 9600 briefly tried as a diagnostic (weak-pull-up/RC-rise-time theory for the
         * framing error) -- ruled out, ISR.FE was byte-identical at both baud rates. Back
         * to full speed. */
        h->Init.BaudRate = 115200;
        h->Init.WordLength = UART_WORDLENGTH_8B;
        h->Init.StopBits = UART_STOPBITS_1;
        h->Init.Parity = UART_PARITY_NONE;
        h->Init.Mode = UART_MODE_TX_RX;
        h->Init.HwFlowCtl = UART_HWCONTROL_NONE;
        h->Init.OverSampling = UART_OVERSAMPLING_16;
        HAL_HalfDuplex_Init(h);
    }

    /* Give the TMC2209s time to finish their own power-up before the first
     * UART transaction -- a write issued via TMC_Init() (milliseconds after
     * MCU reset) was observed silently not sticking on real hardware, while
     * the identical write issued later by hand over the command protocol
     * worked fine. Most likely explanation: the driver isn't ready to
     * reliably receive UART yet this early. TMC2209 writes get no
     * acknowledgment datagram at all, so on top of the delay, read back and
     * retry rather than trust a single fire-and-forget write. */
    HAL_Delay(50);

    /* GCONF: pdn_disable=1 (bit6, "set this bit when using the UART
     * interface" per the TMC2209 datasheet -- otherwise PDN_UART also
     * doubles as an analog config-strapping pin) and mstep_reg_select=1
     * (bit7, otherwise microstep resolution is taken from the MS1/MS2
     * hardware pins and TMC_SetMicrosteps's CHOPCONF write has no effect
     * at all). Best-effort: axes whose UART link isn't working yet will
     * just fail this too, same as every other write for them. GCONF is
     * readable, so this goes through tmc_write_verified rather than a
     * plain TMC_WriteReg. */
    for (uint8_t a = 0; a < STEPPER_NUM_AXES; a++) {
        if (TMC5160_HasAxis(a)) continue; /* SPI/TMC5160 axis now (see tmc5160_spi.c) -- this
                                              UART write would just be noise on its CS/MOSI/SCK
                                              pins before TMC5160_Init() takes them over. */
        tmc_write_verified(a, TMC_REG_GCONF, 0xC0);
    }

    /* IHOLD_IRUN safety-net default -- without this, an axis that gets
     * .enable()'d before anyone calls TMC_SetCurrent() for it runs on
     * whatever this chip's power-on-reset IHOLD_IRUN value happens to be,
     * which real hardware testing confirmed is high enough to overheat a
     * small NEMA8 (this joint's motor is rated 0.2A/phase -- confirmed on
     * the datasheet/label, not guessed) within minutes just sitting
     * enabled and idle. 100mA run / 50mA hold is comfortably below every
     * NEMA8 rating this project is expected to see, not tuned to any one
     * motor -- callers still need their own TMC_SetCurrent() for real
     * torque, this only makes "forgot to call it" inert instead of
     * dangerous. IHOLD_IRUN is write-only (see tmc_write_verified's
     * comment for why that register can't be read back), so plain
     * TMC_WriteReg, not verified. */
    for (uint8_t a = 0; a < STEPPER_NUM_AXES; a++) {
        if (TMC5160_HasAxis(a)) continue; /* see the skip above */
        TMC_SetCurrent(a, 100, 50);
    }
}

static int send_and_collect_reply(uint8_t axis, const uint8_t *tx, uint8_t txlen,
                                   uint8_t *rx, uint8_t rxlen)
{
    UART_HandleTypeDef *h = &tmc_huart[axis];

    /* DIAGNOSTIC: no EnableTransmitter/EnableReceiver toggling. That toggling produced a
     * byte-identical framing error (ISR.FE=1, RXNE=0) on every single attempt regardless
     * of baud rate (115200 vs 9600) or whether anything was even plugged into J1 -- fully
     * deterministic and electrically baud-independent, which doesn't fit a signal-timing
     * explanation. Current theory: the TE-off/RE-on handover itself glitches the receiver.
     * HAL_HalfDuplex_Init() already configures both TE and RE together via Init.Mode =
     * UART_MODE_TX_RX; this leaves them both enabled the whole time instead of switching
     * direction between transmit and receive. */
    if (HAL_UART_Transmit(h, (uint8_t *)tx, txlen, TMC_UART_TIMEOUT_MS) != HAL_OK) return -2;

    if (rxlen == 0) return 0;

    if (HAL_UART_Receive(h, rx, rxlen, TMC_UART_TIMEOUT_MS) != HAL_OK) return -4;
    return 0;
}

static int tmc_transact(uint8_t axis, const uint8_t *tx, uint8_t txlen,
                         uint8_t *rx, uint8_t rxlen)
{
    const TmcUartPin_t *p = &tmc_pin[axis];
    if (p->inst == NULL) return sw_uart_transact(p, tx, txlen, rx, rxlen);
    return send_and_collect_reply(axis, tx, txlen, rx, rxlen);
}

/* Plain fire-and-forget write -- TMC2209 writes get no acknowledgment
 * datagram at all, so this can't detect one that didn't stick (a real
 * failure mode: CHOPCONF's MRES field was observed silently staying at its
 * old value across a fresh USTEP on real hardware). NOT safe to wrap in a
 * generic read-back verification at this level: several TMC2209 registers
 * (IHOLD_IRUN among them) are write-only and simply don't reflect what was
 * written when read back, so a blind "read it back and retry until it
 * matches" was tried here previously and made every CUR command fail
 * (rc=-8) even when the write genuinely worked. Callers that write a
 * register known to be readable (CHOPCONF, GCONF) verify explicitly
 * themselves via tmc_write_verified below -- see TMC_SetMicrosteps and
 * TMC_Init. */
int TMC_WriteReg(uint8_t axis, uint8_t reg, uint32_t data)
{
    if (axis >= STEPPER_NUM_AXES) return -100;
    uint8_t dg[8];
    dg[0] = TMC_SYNC;
    dg[1] = TMC_ADDR_TX;
    dg[2] = reg | 0x80;
    dg[3] = (uint8_t)(data >> 24);
    dg[4] = (uint8_t)(data >> 16);
    dg[5] = (uint8_t)(data >> 8);
    dg[6] = (uint8_t)(data);
    dg[7] = tmc_crc8(dg, 7);
    return tmc_transact(axis, dg, 8, NULL, 0);
}

/* Write a register known to support read-back, verifying and retrying since
 * a single fire-and-forget write has been observed on real hardware to
 * sometimes silently not stick. Do NOT use this for write-only registers
 * (IHOLD_IRUN, TPOWERDOWN, TPWMTHRS, ...) -- the read-back will never match
 * regardless of write success and this will always exhaust its retries. */
static int tmc_write_verified(uint8_t axis, uint8_t reg, uint32_t data)
{
    for (uint8_t attempt = 0; attempt < 5; attempt++) {
        int rc = TMC_WriteReg(axis, reg, data);
        if (rc != 0) return rc; /* local/link error -- not a verification failure, don't retry */

        HAL_Delay(20);
        uint32_t readback;
        if (TMC_ReadReg(axis, reg, &readback) == 0 && readback == data) return 0;
        HAL_Delay(20);
    }
    return -8; /* write never verified against a read-back */
}

/* DIAGNOSTIC: lets the caller override the slave-address byte sent in a
 * read request, instead of always using TMC_ADDR_TX (0x00). Real hardware
 * testing found every read failing (no reply ever seen, confirmed via
 * oscilloscope -- our own request transmits cleanly, the driver simply
 * never responds) while writes always work. The TMC2209's own UART address
 * is set by MS1/MS2 pin strapping (datasheet: MS1_ADDR0/MS2_ADDR1) --
 * writes don't require an address match ("no addressing is required" for
 * write-only access per the datasheet's multi-slave section), but a read
 * does, since the driver needs to know it's the one that should reply.
 * If these modules are strapped to a nonzero address, TMC_ADDR_TX=0x00
 * would explain this exactly. Sweep 0-3 (the only possible values --
 * 2-bit field) via this function before assuming anything else is wrong. */
int TMC_ReadRegAddr(uint8_t axis, uint8_t slave_addr, uint8_t reg, uint32_t *data_out)
{
    if (axis >= STEPPER_NUM_AXES || !data_out) return -100;
    uint8_t dg[4];
    dg[0] = TMC_SYNC;
    dg[1] = slave_addr;
    dg[2] = reg & 0x7F;
    dg[3] = tmc_crc8(dg, 3);

    uint8_t reply[8];
    int rc = tmc_transact(axis, dg, 4, reply, 8);
    if (rc != 0) return rc;

    if (tmc_crc8(reply, 7) != reply[7]) return -6;
    if (reply[0] != TMC_SYNC || reply[2] != (reg & 0x7F)) return -7;

    *data_out = ((uint32_t)reply[3] << 24) | ((uint32_t)reply[4] << 16) |
                ((uint32_t)reply[5] << 8) | (uint32_t)reply[6];
    return 0;
}

int TMC_ReadReg(uint8_t axis, uint8_t reg, uint32_t *data_out)
{
    if (axis >= STEPPER_NUM_AXES || !data_out) return -100;
    uint8_t dg[4];
    dg[0] = TMC_SYNC;
    dg[1] = TMC_ADDR_TX;
    dg[2] = reg & 0x7F;
    dg[3] = tmc_crc8(dg, 3);

    uint8_t reply[8];
    int rc = tmc_transact(axis, dg, 4, reply, 8);
    if (rc != 0) return rc;

    if (tmc_crc8(reply, 7) != reply[7]) return -6;
    if (reply[0] != TMC_SYNC || reply[2] != (reg & 0x7F)) return -7;

    *data_out = ((uint32_t)reply[3] << 24) | ((uint32_t)reply[4] << 16) |
                ((uint32_t)reply[5] << 8) | (uint32_t)reply[6];
    return 0;
}

static uint8_t current_to_cs(uint16_t ma)
{
    float irms = ma / 1000.0f;
    float cs = 32.0f * irms * 1.41421356f * (TMC_RSENSE_OHM + 0.02f) / TMC_VFS_VOLTS - 1.0f;
    if (cs < 0) cs = 0;
    if (cs > 31) cs = 31;
    return (uint8_t)(cs + 0.5f);
}

int TMC_SetCurrent(uint8_t axis, uint16_t run_ma, uint16_t hold_ma)
{
    uint8_t irun = current_to_cs(run_ma);
    uint8_t ihold = current_to_cs(hold_ma);
    uint32_t reg = ((uint32_t)ihold) | ((uint32_t)irun << 8) | ((uint32_t)4 << 16); /* IHOLDDELAY=4 */
    return TMC_WriteReg(axis, TMC_REG_IHOLD_IRUN, reg);
}

int TMC_SetMicrosteps(uint8_t axis, uint16_t usteps)
{
    /* CHOPCONF MRES field: 0=256,1=128,...,8=1 microstep (log2 countdown from 256) */
    uint8_t mres = 8;
    uint16_t v = 256;
    while (v > usteps && mres > 0) { v >>= 1; mres--; }

    uint32_t chopconf;
    int rc = TMC_ReadReg(axis, TMC_REG_CHOPCONF, &chopconf);
    if (rc != 0) return rc;
    HAL_Delay(20); /* turnaround gap before the write -- see tmc_write_verified's comment */
    chopconf &= ~(0xFUL << 24);
    chopconf |= ((uint32_t)mres << 24);
    return tmc_write_verified(axis, TMC_REG_CHOPCONF, chopconf);
}
