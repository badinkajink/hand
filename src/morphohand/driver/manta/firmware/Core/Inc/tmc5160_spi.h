#ifndef TMC5160_SPI_H
#define TMC5160_SPI_H

#include <stdint.h>

/* TMC5160 register addresses actually used here (subset -- add more as needed) */
#define TMC5160_REG_GCONF      0x00
#define TMC5160_REG_GSTAT      0x01
#define TMC5160_REG_IHOLD_IRUN 0x10
#define TMC5160_REG_TPOWERDOWN 0x11
#define TMC5160_REG_TCOOLTHRS  0x14
#define TMC5160_REG_COOLCONF   0x6D
#define TMC5160_REG_CHOPCONF   0x6C
#define TMC5160_REG_DRV_STATUS 0x6F

/* DRV_STATUS bit24 -- live StallGuard2 flag, set when SG_RESULT (bits 9:0 of
 * the same register) has been clamped to 0 by the COOLCONF.SGT bias, i.e. a
 * real stall. Confirmed via real hardware testing (SPI-only, no DIAG pin
 * involved) to respond correctly: reads ~1023 while spinning freely, drops
 * under ~150 during an actual forced stall, with this bit set once it hits
 * 0. Used for polling-based stall detection -- see Stepper_HomingPollTick()
 * in stepper.c. */
#define TMC5160_DRV_STATUS_STALLGUARD (1UL << 24)

/* Default CHOPCONF written to every populated axis at init. TOFF (bits3:0)
 * is the critical field here: TMC5160 resets with CHOPCONF's TOFF=0, which
 * per the datasheet means the driver stage is fully disabled -- no output
 * on any bridge, regardless of STEP/DIR/EN. Confirmed on real hardware: J1
 * sat at TOFF=0 (silent, no buzz, no movement, but clean GSTAT and normal
 * step-counting in firmware) until this same value (copied from J0, which
 * had been configured by hand earlier and does work) was written, at which
 * point it moved immediately. Every axis needs this written explicitly --
 * nothing else in this driver does it implicitly. */
#define TMC5160_DEFAULT_CHOPCONF 0x140100C3UL

/* Default IHOLD_IRUN: IHOLD=0, IRUN=1, IHOLDDELAY=6. With this board's
 * 75mOhm sense resistors and the project's 200mA/phase-rated motors, IRUN=1
 * computes (via I_RMS = (CS+1)/32 * (0.325V/Rsense) / sqrt(2)) to ~192mA
 * RMS, just under rated max. IHOLD=0 gives ~96mA holding current. Pick a
 * different value deliberately if a given axis's motor has a different
 * rating -- don't assume this fits every axis on this project. */
#define TMC5160_DEFAULT_IHOLD_IRUN 0x00060100UL

/* Default TCOOLTHRS (0x14): max 20-bit value, i.e. maximally permissive.
 * TCOOLTHRS gates StallGuard/CoolStep activity by comparing against TSTEP
 * (measured, inversely proportional to speed) -- StallGuard is only "live"
 * when TSTEP <= TCOOLTHRS, so TCOOLTHRS=0 (this chip's power-on-reset
 * default) means the condition is essentially never true, i.e. StallGuard
 * is NEVER active at any real speed. This register is write-only, like
 * IHOLD_IRUN, so it can't be read back to confirm -- and it was never
 * explicitly written anywhere in this project until now, on the assumption
 * (unverified, apparently wrong) that its reset default was permissive
 * enough. Prime suspect for why every stall-detection attempt on
 * 2026-08-23/24 failed to register a single real, confirmed hardstop
 * contact regardless of current/SGT/grace-period/debounce tuning -- all of
 * that tuning was likely operating on a detector that was never armed at
 * all. Set generously permissive here (max value) rather than computing an
 * exact clock-derived threshold, since the actual need is "active across
 * this project's whole tested speed range (300-8000sps)", not a precise
 * cutoff. */
#define TMC5160_DEFAULT_TCOOLTHRS 0x000FFFFFUL

/* Motion-start current kick: IHOLD=1, IRUN=7 (~766mA at this board's 75mOhm
 * sense resistors). Confirmed on real hardware (J0, 2026-08-24): a MOVE/JOG
 * starting from a true standstill can silently fail to turn the motor at
 * all -- at the axis's normal running current (~192-383mA), static friction
 * at the very start of motion can exceed available torque, and firmware has
 * no way to detect this (open-loop step counting reports success regardless
 * of whether the motor actually turned). A brief burst at this current
 * before dropping to the configured running current reliably breaks that
 * static friction. Duration needed isn't a fixed number, though -- 250ms
 * worked starting from a random mid-rail position but was NOT enough
 * starting right from a hardstop (silent non-move, same bug this exists to
 * prevent, just with the kick present and still too short), where
 * mechanical preload against the wall apparently needs more. Bumped to
 * 500ms; re-tune again if a 500ms kick still isn't enough at a hardstop --
 * this may need to be even longer, or the wrong knob entirely (compare
 * against just raising TMC5160_KICK_IHOLD_IRUN instead) if 500ms doesn't
 * hold up. See TMC5160_StartMotionKick(). */
#define TMC5160_KICK_IHOLD_IRUN 0x00060701UL
#define TMC5160_KICK_DURATION_MS 500U

/* Bit-banged SPI, one bus shared across axes (PG6=MOSI, PG7=MISO, PG8=SCK --
 * see generic-bigtreetech-manta-m8p-V2_0.cfg's commented [tmc2130 stepper_y]
 * block; confirmed via datasheet these pins have no plain SPI-peripheral AF
 * on this chip, only OCTOSPI-manager functions, which is why Klipper also
 * bit-bangs this bus). Only the per-axis CS pin differs -- see tmc5160_spi.c's
 * cs table. Requires the board's SPI-mode jumpers (4, per the user manual's
 * "TMC Drivers - SPI Mode" diagram) set for that axis's socket. Returns
 * silently (does nothing / returns 0) for an axis with no CS pin defined. */
void TMC5160_Init(void);

uint32_t TMC5160_ReadReg(uint8_t axis, uint8_t reg);
void TMC5160_WriteReg(uint8_t axis, uint8_t reg, uint32_t data);

/* True if this axis has a CS pin populated in tmc5160_cs[] -- i.e. it's
 * safe/meaningful to call TMC5160_ReadReg/WriteReg on it. */
uint8_t TMC5160_HasAxis(uint8_t axis);

/* This axis's steady-state IHOLD_IRUN -- what TMC5160_StartMotionKick()
 * restores after the kick window, and what a fresh kick's write doesn't
 * disturb (kick writes bypass this so kicking never clobbers the intended
 * running current). Applies immediately via TMC5160_WriteReg as well as
 * remembering the value; use this (not a raw WREG5160 IHOLD_IRUN write) any
 * time you want the kick logic to restore back to it correctly. Defaults
 * to TMC5160_DEFAULT_IHOLD_IRUN via TMC5160_Init(). */
void TMC5160_SetRunCurrent(uint8_t axis, uint32_t ihold_irun);

/* Call when a MOVE/JOG is starting motion from a true standstill (see
 * Stepper_Move/Stepper_Jog in stepper.c, which call this automatically --
 * not normally something to call directly). Writes TMC5160_KICK_IHOLD_IRUN
 * immediately and arms a one-shot timer; the next TMC5160_KickPollTick()
 * call after TMC5160_KICK_DURATION_MS restores this axis's
 * TMC5160_SetRunCurrent value. No-op if this axis isn't populated. */
void TMC5160_StartMotionKick(uint8_t axis);

/* Poll every axis with an armed motion-start kick and restore its running
 * current once TMC5160_KICK_DURATION_MS has elapsed. Cheap no-op when
 * nothing's armed -- call unconditionally from the main loop, same pattern
 * as Stepper_HomingPollTick(). */
void TMC5160_KickPollTick(void);

#endif /* TMC5160_SPI_H */
