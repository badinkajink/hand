# Bring-up

Bring-up procedure for the current build: TMC5160 SPI-driven stepper axes
with StallGuard2 homing, plus a separate U2D2 servo subsystem.
`host/examples/hand_control.py` is the primary interface.

## 1. Physical setup

Four independent electrical/data paths, all needed before the hand is
usable. TODO: confirm exact connector types/labels against the board
silkscreen and add photos -- the descriptions below are accurate to how
this build is wired but not yet cross-checked against BTT's own connector
labeling.

1. **Motor/logic power -- 19V DC into the Manta M8P's main screw terminal.**
   This is the same rail that powers the M8P board itself; there is no
   separate "board on" step -- applying 19V here powers on the M8P (and,
   per the board's onboard regulator, backfeeds power downstream) and
   energizes the TMC5160 drivers. See `docs/pinout.md`/`tmc5160_spi.h` for
   the current default running/holding current settings this firmware
   assumes at this voltage -- re-verify if you change it.
2. **Separate 5V power** feeds the CB1 independently of the 19V motor rail.
   Keep both connected before expecting the CB1 to be reachable over
   SSH/network -- the CB1 will not reliably stay up on backfed power alone
   under load. TODO: confirm which connector/screw terminal this is and
   whether the two supplies need to be sequenced (5V first, then 19V, or
   either order) -- not yet verified either way.
3. **U2D2 servo link -- micro-USB from the CB1 to the U2D2 adapter.**
   Completely separate from the M8P's own USB connection to the CB1 (see
   below) -- this is the link the 6 Feetech SCS0009 servos (2 per finger)
   talk over, and none of it goes through the STM32 firmware at all. Shows
   up on the CB1 as `/dev/ttyUSB0`. See `host/manta_hand/servos.py`'s module
   docstring for the servo-side protocol details. TODO: confirm the actual
   servo power source/rail and whether it needs its own sequencing relative
   to the 19V/5V supplies above -- the U2D2's USB link alone is data-only,
   servos draw far more current than USB can supply.

   **Servo models and power** (manufacturer/reseller-published specs, not
   yet measured on this hardware):

   | Spec | SCS0009 (current, all 6 installed) | STS3215 (added next hardware version) |
   |---|---|---|
   | Operating voltage | 4-7.4V | 4-14V (variant-dependent) |
   | No-load current | 150mA @ 6V | 150mA @ 6V / 180mA @ 12V |
   | Stall current | 1.0A @ 6V | 2A @ 6V / 2.7A @ 12V |
   | Stall torque | 2.3 kg·cm @ 6V | 19.5 kg·cm @ 6V / 30 kg·cm @ 12V |
   | Rated torque | 0.7 kg·cm @ 6V | 6.5 kg·cm @ 6V / 10 kg·cm @ 12V |
   | Weight | 13.2g | 55g |

   Sources: [evelta.com SCS0009](https://evelta.com/scs0009-6v-2-3kg-300deg-serial-bus-servo-motor/),
   [RobotShop STS3215 12V](https://www.robotshop.com/products/feetech-12v-30kgcm-magnetic-encoding-servo-sts3215),
   [servodatabase.com STS3215](https://servodatabase.com/servo/feetech/sts3215).
   STS3215's stall current is 2-2.7x SCS0009's -- size the servo power
   supply for the *new* 9-servo (3/finger x 3 fingers) hand around
   STS3215's numbers, not SCS0009's, once real servo-to-joint assignments
   are decided (see `docs/servos.md`).
4. **M8P <-> CB1 link -- micro-USB, CDC-ACM.** Shows up on the CB1 as
   `/dev/ttyACM0` once the STM32 has booted its application firmware (not
   DFU mode) -- see "Startup / power-up" in the project README for the
   BOOT0/RESET cycle this currently requires every time that device drops
   out.
5. **Per-axis SPI-mode jumpers.** Each of the M8P's motor driver sockets
   needs its SPI-mode jumpers set (4 per socket, per BTT's own "TMC Drivers
   - SPI Mode" diagram in the board's user manual) for that axis's TMC5160
   to be reachable over the shared bit-banged SPI bus this firmware uses
   (see `tmc5160_spi.c`'s header comment) -- an axis with these jumpers in
   the wrong position won't respond to any `WREG5160`/`RREG5160`/`HOME`
   command for that joint, and won't necessarily fail loudly (writes over
   SPI to an unpopulated/misconfigured axis can silently no-op).
6. **CB1 network -- Ethernet or WiFi.** The CB1 has a static IP on the lab
   network, `10.99.99.2` (SSH user `irlab`), reachable over either
   interface. Nothing about this project's normal operation requires a
   local display/keyboard on the CB1 -- everything in this doc and in
   `hand_control.py` runs over SSH.

## 2. Toolchain (once, on the CB1)

```sh
sudo apt install gcc-arm-none-eabi dfu-util make
```

## 3. Firmware build + flash

```sh
cd firmware
make
```

Flash via the BOOT0/RESET DFU cycle described in the README, then:

```sh
make flash
```

Every reflash needs that same physical BOOT0+RESET sequence -- there's no
software-triggered bootloader re-entry in this build.

## 4. USB enumeration check

After flashing (and after any power-on -- see the README's startup
section), confirm both serial links show up on the CB1:

```sh
ls /dev/ttyACM0 /dev/ttyUSB0
```

`ttyACM0` is the M8P (STM32 firmware); `ttyUSB0` is the U2D2 servo adapter.
If `ttyACM0` is missing, do the BOOT0/RESET DFU cycle from the README
before assuming anything else is wrong -- this is by far the most common
cause and isn't specific to a fresh flash.

## 5. Protocol sanity check, no Python yet

```sh
screen /dev/ttyACM0 115200
STATALL
```

Expect `OK` followed by 8 lines of status. `ERR BADCMD` verbatim usually
means line-ending mismatch (`screen` sends `\r`, which this firmware
accepts, but double check if using a different terminal).

## 6. Per-axis StallGuard2 tuning

This is the step most likely to need real iteration, and the one most
sensitive to drift over time -- heat, current, mechanical wear, and even a
DFU reset can all shift what SGT value actually works for a given axis.
There is no shortcut around doing this live against real hardware per
axis.

**Live-grind method** -- the fastest reliable way to find a working SGT
before committing to a real `HOME` test:

1. Press the axis against a *confirmed real* hardstop with a plain `JOG`
   (no stall detection involved) at the speed you intend to home at.
2. Wait past the ramp-up (roughly `HOME_VELOCITY / HOME_ACCEL` seconds,
   plus ~1s margin) before trusting any reading -- StallGuard2's raw
   measurement is unreliable at low speed, and trusting it too early
   produces false positives that look identical to a real stall.
3. Repeatedly read `RREG5160 J<n> 6F` (DRV_STATUS) directly: bits 9:0 are
   `sg_result` (near-max while spinning freely, should clamp to a
   *sustained* 0 at real contact), bit 24 is the live stall flag.
4. Compare the free-running baseline against the stall value at several
   candidate SGT settings (`WREG5160 J<n> 6D <SGT<<16, hex>`) before
   picking one -- a good value shows a clean, large gap between "spinning
   freely" and "stalled"; a bad one shows the same noisy low range in both
   states (this has happened on more than one axis this session and was
   NOT fixable by picking a different SGT -- it needed a mechanical fix:
   re-oiling the leadscrew, recentering a coupling, reseating a bearing).
5. Once a candidate looks clean, validate it with several real
   back-off-and-rehome cycles (`host/examples/hand_control.py`'s
   `home_axis`/`home_all_axes`) before trusting it -- a single successful
   home is not enough evidence; this project has repeatedly seen an SGT
   value pass once and then fail intermittently on a later run.

Record each axis's working SGT in `hand_control.py`'s `HOME_COOLCONF`
dict, not in firmware -- it's intentionally a per-axis, easily-edited
Python constant, not compiled in.

**If StallGuard won't reliably distinguish a real stall from noise no
matter what SGT you try**, don't keep iterating on SGT alone -- check the
mechanism itself first (leadscrew lubrication, coupling alignment, bearing
seating). Every persistent case of this on this build turned out to be
mechanical, not a tuning problem.

## 7. Run it

```sh
cd host/examples
python3 hand_control.py
```

This homes all 6 stepper axes (backing off first if an axis is already at
its hardstop) and cycles the fingers through flexion/extension before
accepting any command -- see the script's own docstring for the full text
command language.

## Recommendations for further work

- **Document the physical setup with photos/labels**, not just prose --
  the power/jumper/connector section above is accurate to how this unit is
  wired but hasn't been cross-checked against the board's own silkscreen or
  BTT's manual diagrams. Worth doing once, carefully, so a new person (or
  you, in six months) doesn't have to re-derive it.
- **Characterize why StallGuard2 detection is intermittent** rather than
  continuing to treat each axis's SGT value as a one-time tuning exercise.
  This session's `_home_timeout_ms` guaranteed-coverage fallback in
  `hand_control.py` makes an undetected stall harmless (the timeout proves
  the axis reached its hardstop regardless), but it doesn't explain *why*
  the same SGT value passes some homing attempts and not others on the same
  axis with no hardware change in between. Worth a real investigation
  (current ripple? SPI bus contention with a neighboring axis? thermal?)
  if this project keeps scaling.
- ~~**Calibrate remaining fingers' adduction/abduction range**~~ -- DONE
  (2026-08-29). All three fingers have a real, wide, bidirectional `aa`
  range in `FINGER_JOINTS`, from the assembled torque-free sweep: finger 0
  (-70.02, +74.71), finger 1 (-79.69, +74.41), finger 2 (-75.29, +75.29).
  Every finger yaws. There is no longer an `ADDUCTION_RANGE` symbol and no
  ±44° cap -- that described a superseded state, as did the claim that
  fingers 1/2 raise on nonzero adduction. The three differ by up to ~10° at
  the bounds (per-servo assembled hardstops, not a design difference); the
  tightest range common to all three is (-70.02, +74.41), i.e. ±70
  symmetric, which is what a policy commanding all three fingers uniformly
  should stay inside.
- **Ruler-verify the non-J0 axes' `STEPS_PER_MM` magnitudes** -- only J0's
  is directly ruler-verified; the rest were back-calculated from a
  known-good 10mm move. Probably fine, but worth confirming once mechanical
  work on those axes settles down.
- **Firmware-level fix for the grace-period-at-hardstop case**, if it comes
  up often enough to matter -- `hand_control.py`'s `PRE_HOME_BACKOFF_MM`
  works around it at the Python layer (back off before homing if already
  close to home), but the underlying cause (HOME's grace period skips
  stall-checking during ramp-up regardless of starting position) still
  lives in firmware and could be fixed at the source if it turns out to
  matter for the eventual sim-to-real handoff to the MorphoHand policy.
- **This repo's relationship to the MorphoHand simulation/RL repo** should
  get written down somewhere durable (this file, or a top-level doc) once
  you've decided how the two are organized relative to each other --
  right now that connection only exists in conversation history.
