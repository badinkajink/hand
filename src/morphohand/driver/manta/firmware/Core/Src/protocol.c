#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "protocol.h"
#include "stepper.h"
#include "tmc2209_uart.h"
#include "tmc5160_spi.h"
#include "usbd_cdc_if.h"

#define RXBUF_SIZE 256
#define LINE_MAX   96

static volatile uint8_t rxbuf[RXBUF_SIZE];
static volatile uint16_t rx_head = 0, rx_tail = 0;

void Protocol_Init(void)
{
    rx_head = rx_tail = 0;
}

void Protocol_OnUsbRxByte(uint8_t byte)
{
    uint16_t next = (uint16_t)((rx_head + 1) % RXBUF_SIZE);
    if (next != rx_tail) { /* drop byte on overflow rather than corrupt the buffer */
        rxbuf[rx_head] = byte;
        rx_head = next;
    }
}

/* CDC_Transmit_FS returns USBD_BUSY (and sends nothing) if the previous
 * packet hasn't finished transmitting yet -- observed in practice: STATALL's
 * 8 back-to-back lines occasionally dropped one before this retry loop was
 * added, since the original code ignored the return value entirely. Retries
 * with a bounded spin count rather than blocking forever, in case the host
 * side genuinely stops reading. */
static void cdc_send_blocking(const uint8_t *buf, uint16_t len)
{
    uint32_t attempts = 0;
    while (CDC_Transmit_FS((uint8_t *)buf, len) == USBD_BUSY) {
        if (++attempts > 200000U) return; /* give up rather than hang the main loop forever */
    }
}

static void reply(const char *s)
{
    cdc_send_blocking((const uint8_t *)s, (uint16_t)strlen(s));
}

static void replyf(const char *fmt, ...)
{
    char buf[LINE_MAX];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf) - 2, fmt, ap);
    va_end(ap);
    if (n < 0) n = 0;
    buf[n++] = '\r'; buf[n++] = '\n';
    cdc_send_blocking((const uint8_t *)buf, (uint16_t)n);
}

static int parse_joint(const char *tok)
{
    if (!tok || (tok[0] != 'J' && tok[0] != 'j')) return -1;
    int j = atoi(tok + 1);
    if (j < 0 || j >= STEPPER_NUM_AXES) return -1;
    return j;
}

static void report_joint(uint8_t j, char *out, size_t outsz)
{
    Stepper_Status s;
    Stepper_GetStatus(j, &s);
    snprintf(out, outsz, "%ld %ld %u %u %u", (long)s.position, (long)s.target,
             s.moving ? 1U : 0U, s.enabled ? 1U : 0U, (unsigned)s.homing_result);
}

static void execute_line(char *line)
{
    char *save = NULL;
    char *cmd = strtok_r(line, " \t", &save);
    if (!cmd) return;

    if (!strcmp(cmd, "STATALL")) {
        reply("OK\r\n");
        for (uint8_t j = 0; j < STEPPER_NUM_AXES; j++) {
            char body[64];
            report_joint(j, body, sizeof(body));
            replyf("%s", body);
        }
        return;
    }
    if (!strcmp(cmd, "STOPALL")) { Stepper_StopAll(); reply("OK\r\n"); return; }

    char *jtok = strtok_r(NULL, " \t", &save);
    int j = parse_joint(jtok);

    if (!strcmp(cmd, "STAT")) {
        if (j < 0) { reply("ERR RANGE\r\n"); return; }
        char body[64];
        report_joint((uint8_t)j, body, sizeof(body));
        replyf("OK %s", body);
        return;
    }
    if (!strcmp(cmd, "EN"))  { if (j < 0) { reply("ERR RANGE\r\n"); return; } Stepper_Enable((uint8_t)j);  reply("OK\r\n"); return; }
    if (!strcmp(cmd, "DIS")) { if (j < 0) { reply("ERR RANGE\r\n"); return; } Stepper_Disable((uint8_t)j); reply("OK\r\n"); return; }
    if (!strcmp(cmd, "STOP")){ if (j < 0) { reply("ERR RANGE\r\n"); return; } Stepper_Stop((uint8_t)j);    reply("OK\r\n"); return; }
    if (!strcmp(cmd, "ZERO")){ if (j < 0) { reply("ERR RANGE\r\n"); return; } Stepper_Zero((uint8_t)j);    reply("OK\r\n"); return; }

    if (!strcmp(cmd, "CUR")) {
        char *a1 = strtok_r(NULL, " \t", &save), *a2 = strtok_r(NULL, " \t", &save);
        if (j < 0 || !a1 || !a2) { reply("ERR BADARG\r\n"); return; }
        int rc = TMC_SetCurrent((uint8_t)j, (uint16_t)atoi(a1), (uint16_t)atoi(a2));
        if (rc == 0) reply("OK\r\n"); else replyf("ERR TMC J%d rc=%d", j, rc);
        return;
    }
    if (!strcmp(cmd, "USTEP")) {
        char *a1 = strtok_r(NULL, " \t", &save);
        if (j < 0 || !a1) { reply("ERR BADARG\r\n"); return; }
        int rc = TMC_SetMicrosteps((uint8_t)j, (uint16_t)atoi(a1));
        if (rc == 0) reply("OK\r\n"); else replyf("ERR TMC J%d rc=%d", j, rc);
        return;
    }
    if (!strcmp(cmd, "RREG")) {
        /* Raw TMC register read for debugging -- reg given in hex, no 0x
         * prefix (e.g. "RREG J0 6C" for CHOPCONF). */
        char *a1 = strtok_r(NULL, " \t", &save);
        if (j < 0 || !a1) { reply("ERR BADARG\r\n"); return; }
        uint8_t reg = (uint8_t)strtoul(a1, NULL, 16);
        uint32_t val = 0;
        int rc = TMC_ReadReg((uint8_t)j, reg, &val);
        if (rc == 0) replyf("OK %08lX", (unsigned long)val);
        else replyf("ERR TMC J%d rc=%d", j, rc);
        return;
    }
    if (!strcmp(cmd, "RREGA")) {
        /* DIAGNOSTIC: same as RREG but with an explicit slave-address byte
         * (decimal, 0-3) instead of the fixed TMC_ADDR_TX -- see
         * TMC_ReadRegAddr's comment in tmc2209_uart.c. Usage:
         * "RREGA J0 <addr> <reg-hex>", e.g. "RREGA J0 1 06" tries address 1. */
        char *a1 = strtok_r(NULL, " \t", &save), *a2 = strtok_r(NULL, " \t", &save);
        if (j < 0 || !a1 || !a2) { reply("ERR BADARG\r\n"); return; }
        uint8_t addr = (uint8_t)atoi(a1);
        uint8_t reg = (uint8_t)strtoul(a2, NULL, 16);
        uint32_t val = 0;
        int rc = TMC_ReadRegAddr((uint8_t)j, addr, reg, &val);
        if (rc == 0) replyf("OK %08lX", (unsigned long)val);
        else replyf("ERR TMC J%d rc=%d", j, rc);
        return;
    }
    if (!strcmp(cmd, "RREG5160")) {
        /* DIAGNOSTIC: TMC5160 over bit-banged SPI -- see tmc5160_spi.c. Any axis with a
         * populated CS pin in tmc5160_cs[] (currently J0, J1) works; others are a no-op
         * returning 0. "RREG5160 J0 <reg-hex>", e.g. "RREG5160 J0 00" for GCONF. No
         * transport-level failure mode (self-clocked SPI, not a UART timeout), so always OK. */
        char *a1 = strtok_r(NULL, " \t", &save);
        if (j < 0 || !a1) { reply("ERR BADARG\r\n"); return; }
        uint8_t reg = (uint8_t)strtoul(a1, NULL, 16);
        uint32_t val = TMC5160_ReadReg((uint8_t)j, reg);
        replyf("OK %08lX", (unsigned long)val);
        return;
    }
    if (!strcmp(cmd, "WREG5160")) {
        /* "WREG5160 J0 <reg-hex> <val-hex>" -- raw register write. For
         * IHOLD_IRUN (reg 10) specifically, prefer SETRUN5160 instead: a raw
         * write here takes effect immediately but ISN'T remembered as this
         * axis's steady-state current, so the automatic motion-start kick
         * (see TMC5160_StartMotionKick) will restore whatever SETRUN5160 (or
         * the boot default) last set, clobbering a raw WREG5160 write here
         * the next time this axis starts moving from a stop. */
        char *a1 = strtok_r(NULL, " \t", &save), *a2 = strtok_r(NULL, " \t", &save);
        if (j < 0 || !a1 || !a2) { reply("ERR BADARG\r\n"); return; }
        uint8_t reg = (uint8_t)strtoul(a1, NULL, 16);
        uint32_t val = (uint32_t)strtoul(a2, NULL, 16);
        TMC5160_WriteReg((uint8_t)j, reg, val);
        reply("OK\r\n");
        return;
    }
    if (!strcmp(cmd, "SETRUN5160")) {
        /* "SETRUN5160 J<i> <ihold_irun-hex>" -- sets this axis's steady-state
         * IHOLD_IRUN, applied immediately and remembered as what to restore
         * to after a motion-start current kick (see TMC5160_StartMotionKick,
         * stepper.c's Stepper_Move/Stepper_Jog). This is the normal way to
         * change an axis's running current now -- see WREG5160's comment for
         * why a raw register write doesn't interact well with the kick. */
        char *a1 = strtok_r(NULL, " \t", &save);
        if (j < 0 || !a1) { reply("ERR BADARG\r\n"); return; }
        uint32_t val = (uint32_t)strtoul(a1, NULL, 16);
        TMC5160_SetRunCurrent((uint8_t)j, val);
        reply("OK\r\n");
        return;
    }
    if (!strcmp(cmd, "WREG")) {
        /* Raw TMC register write for debugging -- reg and value both in hex,
         * no 0x prefix (e.g. "WREG J0 00 C0" for GCONF=0xC0). TMC2209 writes
         * get no acknowledgment datagram at all, so "OK" here only means the
         * bytes went out locally without a HAL/bit-bang error -- follow with
         * RREG on the same register to confirm it actually stuck. */
        char *a1 = strtok_r(NULL, " \t", &save), *a2 = strtok_r(NULL, " \t", &save);
        if (j < 0 || !a1 || !a2) { reply("ERR BADARG\r\n"); return; }
        uint8_t reg = (uint8_t)strtoul(a1, NULL, 16);
        uint32_t val = (uint32_t)strtoul(a2, NULL, 16);
        int rc = TMC_WriteReg((uint8_t)j, reg, val);
        if (rc == 0) reply("OK\r\n"); else replyf("ERR TMC J%d rc=%d", j, rc);
        return;
    }
    if (!strcmp(cmd, "MOVE")) {
        char *a1 = strtok_r(NULL, " \t", &save), *a2 = strtok_r(NULL, " \t", &save), *a3 = strtok_r(NULL, " \t", &save);
        if (j < 0 || !a1 || !a2 || !a3) { reply("ERR BADARG\r\n"); return; }
        Stepper_Move((uint8_t)j, atol(a1), (uint32_t)atol(a2), (uint32_t)atol(a3));
        reply("OK\r\n");
        return;
    }
    if (!strcmp(cmd, "SETSCALE")) {
        /* SETSCALE J<n> <steps_per_mm> -- RAM-only calibration for MOVEMM, see
         * Stepper_SetStepsPerMM's doc comment. Re-run after every power cycle. */
        char *a1 = strtok_r(NULL, " \t", &save);
        if (j < 0 || !a1) { reply("ERR BADARG\r\n"); return; }
        Stepper_SetStepsPerMM((uint8_t)j, (float)atof(a1));
        reply("OK\r\n");
        return;
    }
    if (!strcmp(cmd, "MOVEMM")) {
        /* MOVEMM J<n> <mm> <max_vel_sps> <accel_sps2> -- same semantics as
         * MOVE but in mm, via this axis's SETSCALE calibration. 0mm is
         * wherever this axis's step-position 0 currently is, so home (or at
         * least ZERO) first. */
        char *a1 = strtok_r(NULL, " \t", &save), *a2 = strtok_r(NULL, " \t", &save), *a3 = strtok_r(NULL, " \t", &save);
        if (j < 0 || !a1 || !a2 || !a3) { reply("ERR BADARG\r\n"); return; }
        if (!Stepper_MoveMM((uint8_t)j, (float)atof(a1), (uint32_t)atol(a2), (uint32_t)atol(a3))) {
            reply("ERR UNCALIBRATED\r\n");
            return;
        }
        reply("OK\r\n");
        return;
    }
    if (!strcmp(cmd, "JOG")) {
        char *a1 = strtok_r(NULL, " \t", &save), *a2 = strtok_r(NULL, " \t", &save);
        if (j < 0 || !a1 || !a2) { reply("ERR BADARG\r\n"); return; }
        Stepper_Jog((uint8_t)j, atol(a1), (uint32_t)atol(a2));
        reply("OK\r\n");
        return;
    }
    if (!strcmp(cmd, "HOME")) {
        /* HOME J<n> <direction: 1 or -1> <vel_sps> <accel_sps2> <timeout_ms>
         * -- non-blocking, same pattern as MOVE/JOG: poll STAT's 5th field
         * (homing_result) for the outcome. Requires SGTHRS/TCOOLTHRS already
         * configured and tuned on this axis via WREG (registers 0x40, 0x14)
         * and a DIAG pin wired for it (currently J0/J1 only) -- see
         * docs/bringup.md. */
        char *a1 = strtok_r(NULL, " \t", &save), *a2 = strtok_r(NULL, " \t", &save);
        char *a3 = strtok_r(NULL, " \t", &save), *a4 = strtok_r(NULL, " \t", &save);
        if (j < 0 || !a1 || !a2 || !a3 || !a4) { reply("ERR BADARG\r\n"); return; }
        int dir = atoi(a1);
        if (Stepper_Home((uint8_t)j, dir >= 0 ? 1 : -1, (uint32_t)atol(a2),
                          (uint32_t)atol(a3), (uint32_t)atol(a4))) {
            reply("OK\r\n");
        } else {
            reply("ERR NODIAG\r\n"); /* no DIAG pin wired for this axis, or axis not enabled */
        }
        return;
    }

    reply("ERR BADCMD\r\n");
}

void Protocol_Poll(void)
{
    static char line[LINE_MAX];
    static uint16_t linelen = 0;

    while (rx_tail != rx_head) {
        uint8_t b = rxbuf[rx_tail];
        rx_tail = (uint16_t)((rx_tail + 1) % RXBUF_SIZE);

        if (b == '\n' || b == '\r') {
            if (linelen > 0) {
                line[linelen] = '\0';
                execute_line(line);
                linelen = 0;
            }
        } else if (linelen < LINE_MAX - 1) {
            line[linelen++] = (char)b;
        } else {
            linelen = 0; /* overlong line: drop it */
        }
    }
}
