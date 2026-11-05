# Gantry overnight results (2026-11-04)

These are the results from running the gantry on the 24 V / 6 A supply.

## 1. Stall detection at 24 V

**Free running is fine.** The current SGT values (8 and 10) give a steady load reading with no false flags on any axis, at 12000 steps/s and IRUN 1.

**Homes fail at the wall.** When an axis hits its wall, the reading no longer collapses toward 0 the way it did at 19 V. It steps down by about 20 % and stays there. The firmware's stall flag needs a reading near 0, so it never trips at the wall. Any SGT sensitive enough to trip there (6) also flags 5–29 % of the time while the axis runs freely.

| Axis (SGT 10, 12000 sps) | free sg | at wall | drop |
|---|---|---|---|
| J0 | 110 | 87 | 21 % |
| J1 | 76 | 54 | 29 % (ripple, see below) |
| J2 | 104 | 83 | 20 % |
| J3 | 94 | 74 | 21 % |
| J4 | 86 | 50 | 42 % |
| J5 | 102 | 77 | 25 % |

At 6000 and 4000 steps/s the drop is still about 20 counts. The free-running noise grows as speed falls, though, so the wall is harder to separate at 4000.

**Relative-StallGuard home** (`sysid/sg_home.py`, run from the CB1):
1. Once the axis has been at cruise speed for 0.3 s, take the median of the next 0.4 s of readings as its baseline.
2. Stop when the median of the last 40 ms falls below 0.85 × baseline.
3. J1 and J4 have a once-per-turn ripple, so they use a window of one full turn (0.27 s) and a threshold of 0.8 × baseline.

Each axis had been zeroed at its wall beforehand. Single-axis results:
- **J0, J2, J3, J5:** 12 of 12 homes stopped within ±0.1 mm of the wall, each in 6.5 s.
- **J4:** 4 of 4 caught the wall, stopping 0.5–1.3 mm past it because of the longer window.
- **J1:** not reliable. It false-stopped at 12 mm, inside its known tight zone, and other runs stopped 1–3 mm late. J1 stays on its wall-push home (`home_all.py`), which was camera-verified 10 of 10.

## 2. Multi-axis step budget

The firmware generates about **24,000 steps/s in total**, however that is split across axes (`sysid/step_budget.py`, 30 mm out and back, positions exact in every run).

| axes × speed each | asked total | achieved total |
|---|---|---|
| 1 × 12000 | 12k | 11.9k |
| 2 × 12000 | 24k | 23.8k |
| 3 × 12000 | 36k | 20.1k |
| 6 × 12000 | 72k | 24.2k |
| 6 × 8000 | 48k | 23.8k |
| 6 × 6000 | 36k | 18.1k |
| 6 × 4000 | 24k | 23.7k |

**Rule:** speed per axis ≤ 24000 ÷ number of axes moving. Asking for more than that can lower the total. Commands sent while the board is over budget often get no reply.

## 3. Why the ceiling is so low (firmware, from reading the source; nothing flashed)

- **The caches are never enabled.** `main.c` has no `SCB_EnableICache()` or `SCB_EnableDCache()`. The H723 runs at 550 MHz from flash with 4 wait states, so every instruction fetch is slow.
- **The step interrupt holds the processor for a long time.** Each step spends a 600-iteration `volatile` NOP loop (`STEP_PULSE_NOPS`) holding the pulse high. That is several thousand cycles, against the TMC5160's minimum pulse of about 100 ns. It also uses `HAL_GPIO_WritePin` calls rather than direct BSRR writes.
- **24k steps/s at saturation works out to about 40 µs of processor time per step.**
- **Stall-register reads block stepping.** Each driver-chip read is bit-banged SPI that masks the step interrupts for about 200 µs. Homing polls each homing axis every 3 ms, so six axes homing at once would mask stepping about 40 % of the time.

**Suggested firmware changes, roughly in order of payoff (for review before flashing):**
1. Call `SCB_EnableICache()` at startup. The D-cache needs care with any DMA buffers, so start with the I-cache only.
2. Replace the NOP loop with a DWT-cycle wait of about 0.2 µs, and drive STEP and DIR through GPIO BSRR.
3. Move the TMC5160 reads to the hardware SPI peripheral, or at least stop masking the step interrupts during reads.
4. Implement the relative-StallGuard stop in `Stepper_HomingPollTick` (baseline median, then a windowed median below a ratio), so homing works at 24 V without the host.

Changes 1 and 2 alone should raise the ceiling several-fold. Six axes at full speed, and simultaneous homing, would then become realistic.

## 4. Simultaneous homing (tested 2026-11-04 daytime)

**Grouped StallGuard homing does not work on this firmware.** I ran `sg_home.py J0 J2 J3` together, with no zeroing, and measured every stop against fresh one-axis zeros:

| Setting | Good stops | False or late |
|---|---|---|
| 8000 steps/s, 40 ms window | 5 of 9 | 4, all false stops near 17 mm |
| 6000 steps/s, 150 ms window, ratio 0.8 | 6 of 12 | 6 (J2 false or 1.7 mm late) |

Traces (`sg_home.py --log`) show why:
- **The axes never hold speed.** At three axes × 8000 the board is at its step budget, so J2 and J3 only reach about 7200 steps/s, and their speed shifts whenever another axis changes.
- **The reading gets noisy.** That uneven step timing makes sg swing ±15 % between 0.1 s slices, compared with about ±5 % for one axis alone. That swing buries the 20 % wall drop.
- **Late stops walk the zero.** A late stop skips steps at the wall, so the count runs ahead of the carriage. Over four grouped cycles J0's apparent wall moved from −0.2 mm to −1.0 mm.

**The simultaneous wall push works** (`sysid/push_home_all.py`):
1. All six axes move together to 3 mm.
2. All six are driven together to −1.5 mm at 3500 steps/s, about 21k steps/s in total. Each stalls at its wall and grinds for about 1.4 s at low current; J1 runs at IRUN 2.
3. All six are disabled and zeroed. The whole home takes about 9 s.

I checked it with one-axis StallGuard homes (no zeroing) after each of 3 cycles. The spread across cycles was ≤ 0.15 mm on every axis:

| Axis | StallGuard stop relative to the push zero |
|---|---|
| J0 | +0.05 to +0.07 mm |
| J2 | +0.09 to +0.18 mm |
| J3 | +0.29 to +0.35 mm |
| J4 | −0.61 to −0.76 mm (StallGuard's known lateness on J4) |
| J5 | +0.03 to +0.07 mm |

J1 uses the same push it has always used, which was camera-verified before. It wasn't re-verified separately here.

**Limit:** the push needs the current counts to be right within 1.5 mm. That holds as long as the stepper board hasn't been power-cycled; a CB1 reboot doesn't reset it. After a power cycle, home one axis at a time: `home_all.py J1`, then `sg_home.py --zero` and a `--trusted` verify for each of the rest. One-axis `--zero` homes also need that verify: J3 made one false stop, 2.1 mm short, this morning.

Fully simultaneous StallGuard homing still needs firmware changes 1–4 above.

## 5. Hand state when contact was lost overnight (since re-homed; see section 4)

- **Zeros:** all six zeros were valid at the wall when that last grouped home started (J1 by wall push, the rest by relative StallGuard, and J4 checked against its wall drop).
- **The last test:** after its first cycle, the grouped home re-zeros J0, J2 and J3 wherever each stops. If any of those stops was false, that axis's zero is off.
- **Re-home everything after reconnecting:** with `home_all.py` for J1, and `sg_home.py` for the rest.

## 6. Maximum speeds (tested 2026-11-04 daytime)

1 mm = 3216 steps. Every move was a 40 mm out-and-back (30 mm for the six-axis runs) timed by `sysid/step_budget.py`.

**All six moving together.** These runs used acceleration 8000.

| Speed per axis | Total asked | Achieved total | Notes |
|---|---|---|---|
| 3000 | 18.0k | 17.9k | exact |
| 3500 | 21.0k | 21.0k | exact |
| 3800 | 22.8k | 22.6k | |
| 4000 | 24.0k | 23.7k | **recommended** |
| 4200 | 25.2k | 24.9k | the peak |
| 4500 | 27.0k | 20.5k | over the budget, so throughput collapses |

Use **4000 steps/s per axis (1.24 mm/s)** when all six move together. A 30 mm move then takes about 25 s. The limit is the firmware's step generation (section 3), not the motors. After the sweep, the J0, J2, J3 and J5 walls were unchanged (±0.1 mm).

**One axis alone.** Each axis ran a 40 mm out-and-back from 10 mm, so any lost steps would leave it displaced instead of being hidden by grinding back into the wall. I then checked for lost steps: StallGuard walls for J0 and J2–J5; for J1, the camera, where 0.5 mm of displacement is detectable.

- **Every axis ran 32000 steps/s at acceleration 8000 with no lost steps.**
- **J2 alone also ran 32000 steps/s at acceleration 16000, 32000 and 64000 with no lost steps.**
- **The actual cruise rate tops out at about 26,500 steps/s (8.2 mm/s), whatever speed is asked.** That is the firmware's single-axis ceiling.
- **At acceleration 64000, a 40 mm move takes 5.5 s,** against 8.3 s at 8000.

**Takeaways:**
- **The motors have plenty of margin.** One axis goes 6–7× faster than each axis can when all six move.
- **Firmware changes 1–2 (section 3) are the lever for faster six-axis motion.** If they lifted the total ceiling to the one-axis rate on each axis, six axes could move together at about 8 mm/s.
- **Acceleration can rise to at least 32000.** That shortens short moves, but has only been tested on J2.

**Notes:**
- **The camera view has moved.** After the CB1 reboot the RealSense colour stream is `/dev/video4`; `snap.sh` now defaults to it, overridable with `SNAP_DEV`. J1's finger now sits at the right edge of the frame, rows 156–479 and columns 534–639. `track_shift.py` accepts a `TRACK_REGION` setting, but the plain image difference in that region was more sensitive.
- **J4's verify homes walk its zero.** Each J4 StallGuard home stops about 0.7 mm late and skips that much at the wall. After any J4 `--trusted` check, re-zero it, for example with `push_home_all.py`.

## 7. J1 StallGuard threshold sweep at 24 V (2026-11-04 daytime)

**Result: no threshold, current or speed gives J1 a usable wall signal. J1 stays on the wall-push home.**

Method: `wall_signature.py` drives J1 from 20 mm into the wall and 4 mm past it, logging sg the whole way. I analysed the traces offline with the relative detector, averaging over one leadscrew turn (about 1 mm). Raw traces are in `/tmp/j1sgt/` on the CB1, which is wiped on reboot.

- **Single runs at 12000 steps/s:**
  - IRUN 2 at SGT 0–20: sg is pinned at 0 for SGT ≤ 10, and stall flags fire during free running for SGT ≤ 15.
  - IRUN 1 at SGT 8–16.
- **3 repeats of the 5 best settings** (IRUN 2: SGT 12, 13, 14; IRUN 1: SGT 12, 14). The lowest free-running one-turn average and the at-wall level overlap. The margin flips sign from run to run (−0.27 to +0.49 of baseline), and the baseline drifts 15–25 % between runs.
  - **Best:** IRUN 2 SGT 13 caught the wall in 2 of 3 runs.
  - **Common failures:** other settings would stop falsely at 2–5 mm or 15 mm, or miss the wall.
- **2 repeats each at 20000 and 6000 steps/s:**
  - At 20000 the wall often reads *higher* than free running. The best case, IRUN 2 SGT 18, has a margin of only +0.07–0.09 and would trip about 2 mm past the wall.
  - At 6000 the reading is saturated or very noisy.
- **Interpretation:** this matches the earlier camera check. J1's rotor stops at the wall, but its SG_RESULT keeps reading like free running. J1's motor or wiring may respond differently from the other five, which is worth checking against another axis's motor and cable if J1 ever needs StallGuard.
- **J1 zero after the sweep:** I re-homed J1 with the wall push (`home_all.py J1 --j1-near`). Camera position scans at 8–12 mm matched the morning reference best at 10.0 mm (±0.25 mm).

## 8. The `home` command on the CB1 (set up 2026-11-04)

After SSHing into the CB1, run:

- **`home`**: the fast home. All six axes go to 3 mm together, then are pushed 1.5 mm past their walls and zeroed. It takes about 9 s and leaves every axis at 0 mm, disabled. If the stepper board has been reset since the last home, it moves nothing and asks for `home --full` instead.
- **`home --full`**: the original sequential firmware home, the same one `hand_control.py` runs at startup. It takes about 4 minutes, and each axis grinds at its wall until its timeout. It is safe from any position, so use it after every power-up or DFU reset, and after moving a carriage by hand.
- **`home --check`**: says which of the two is allowed, without moving anything.

**How it detects a reset:** after each successful home it sets a RAM-only marker on the unused J7 (`SETSCALE J7 7.0`). The STM32 loses that marker on any reset, which is exactly when it also loses the step counts. It reads the marker back with `MOVEMM J7 0`, which returns `ERR UNCALIBRATED` once the marker is gone.

**Where it lives:**
- `~/.local/bin/home`: a wrapper with a 330 s wall-clock timeout that disables every axis on any failure.
- `~/manta-hand/sysid/home.py`.

The sysid scripts now import their helpers from their own folder instead of `/tmp`, so they survive a CB1 reboot.

**Tested 2026-11-04:**
- The old home still homes: all six by timeout in 221 s. StallGuard checks of J0, J2, J3 and J5 agreed with the push-home zeros to about 0.15 mm.
- `home` refused correctly while the marker was unset.
- After the marker was set, `home` ran the fast home in 9 s, with all drivers OK.
