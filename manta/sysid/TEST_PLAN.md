# SR² Hand servo system identification: test plan

*Draft for review, 2026-10-27. Hardware: 9 × Feetech SCS0009 finger servos on the CB1, driven
through the U2D2 USB adapter. Notes and raw results: [README.md](README.md).*

**Goal:** measure the SCS0009's friction and load behaviour well enough to fit a BAM
(Better Actuator Models) actuator model. Then check that simulated servo load matches the real
hand before training policy C (the policy that uses load feedback).

---

## Done so far

| Test | Script | Main result |
|---|---|---|
| Health check (read-only) | `servo_health_check.py` | All 9 servos healthy. 6 of 9 have **no EEPROM (on-servo memory) angle limits**, so the software limits are the only range protection |
| Step tests, index fe1 | `single_joint_step_test.py`, `step_press_test.py` | `present_load` works. Moves are captured at ~330 Hz via sync writes. Stick-slip: the joint sticks, then jumps |
| Hand press, index fe1 | `step_press_test.py` | Load reacts to an outside push. Load stays non-zero with **no contact** when friction holds the joint off its target |
| Scale calibration, index fe1 (3 runs) | `load_scale_calibration.py` | **Contact force follows joint position, not load.** Friction held ~170 g with load at 0. At slip, ~0.24 mN·m of torque per load unit (lever arm 35.9 mm, from the v3 CAD meshes) |
| Friction sweep, all 9 joints (2026-10-29) | `friction_sweep.py` | Stick-slip on every joint. Flexion breakaway load 60–90 (~15–22 mN·m), lag ≤1.8°. Adduction joints stickier: 90–210, lag up to 3.3°, reversal dead band up to 4.3° |
| Full-range friction sweep (2026-10-29) | `full_range_sweep.py` | Friction mostly flat across range (48–103 on fe joints); thumb fe2 doubles 78→151 by 40°. Gravity shows only on fe1 (to −46). Middle fe1 has a tight spot at 20–24° (friction ~3.5×, confirmed by rerun); aa held to ±10° |
| Full-range repeat pass (2026-10-29) | `full_range_sweep.py`, `compare_full_range_passes.py` | Friction repeats (medians within ~7%). Sticking spots fixed in angle (print errors): middle fe1 21–27°, thumb fe2 banded 22–48° / 60–80° |
| Full-range third pass (2026-10-29) | same | All medians within 5% (index aa <10%). Middle fe1 spot and thumb fe2 bands repeat; upper-limit stiffening on thumb fe1 / index fe2 did not |
| Full-hand loop rate (2026-10-29) | `loop_rate_test.py`, `combined_read_check.py` | Position + load tops out at 48 Hz with separate reads (50 Hz unreachable; 30 Hz solid). One combined 6-byte read per servo gives position + speed + load at ~111 Hz read-only, so 50 Hz is comfortable with it |
| Servo gain sweep, index fe1 (2026-10-29) | `gain_sweep.py` | P 30–45 cuts tracking error ~35–45% and holding error to ≤1 count; I gain doesn't help. Reference only: **factory gains kept** for training |
| Gain optimization, rig + 4 hand joints (2026-10-29) | `double_pendulum_record.py`, `compare_gains.py`, `gain_sweep.py` | Higher P cuts tracking error 20–40% but raises direction reversals 1.5–2× (rougher to the eye); P ≥ 30 oscillates on index aa / rig joint 1. **Keep factory gains**; use friction feed-forward lead for tracking |

Takeaway so far: load measures how hard the servo is trying, not how hard the finger presses.
The gearbox is effectively non-backdrivable at these forces.

---

## Proposed tests, in order

### 1. Friction sweep: free motion, all 9 joints — DONE 2026-10-29 (see README)
- **Does:** each joint in turn follows a slow triangle wave, ±4° at 1°/s. Only that one servo has torque on.
- **Measures, per joint and direction:**
  - breakaway load, the load just before each jump;
  - lag and dead band while stuck;
  - jump size.
- **Why:** these are the friction numbers for the BAM fit. It also shows whether every joint
  behaves like index fe1.
- **Safety:**
  - sweep kept ≥1° inside the software limits;
  - jam / collision abort if the joint is >4° behind its target for 0.6 s;
  - abort if |load| > 500 or the temperature goes over 45 °C;
  - hard torque-off for each joint.
- **Time:** ~15 s per joint, ~3 min total.
- **Needs:** scale moved out from under the index finger.

### 2. Speed and step response
- **Does:** steps of 5°, 10° and 20° at several `goal_speed` settings.
- **Measures:** top speed, acceleration, overshoot and settling time.
- **Why:** BAM identifies the servo's internal controller from this data.
- **Scope:** index fe1 and fe2 first, then all joints if they look consistent.
- **Needs:** FTDI latency back at 1 ms.

### 3. Backdrive threshold: torque off, hanging weights
- **Does:** hang increasing known weights from the index fingertip, with torque off, until the joint slips.
- **Measures:** the outside torque the joint holds with zero motor effort. This is the "lock-in"
  the simulator must reproduce. The scale runs suggest ≥ ~60 mN·m.
- **Needs:** a string and a few objects weighed on the scale.

### 4. BAM pendulum: the standard BAM identification test
- **Does:** a known weight on a known arm swings on one servo, first with torque off, then under
  small commands.
- **Measures:** friction and inertia in the form BAM's fitting expects.
- **Needs:** a rigid or 3D-printed arm plus a weight. Run on a single servo.
- **Rig designed 2026-10-28:** [bam_pendulum/](bam_pendulum/README.md). One print set fits both the
  SCS0009 and the STS3032, as a single pendulum (identification) or a double pendulum (validation).
  One identification arm (holes at 45 / 60 / 90 mm) stays mounted per servo, so BAM's single zero
  offset holds. Its real mass properties are logged. Weight stacks are split across the arm to keep
  the load near the output bearing.

### 5. Scale calibration on 2–3 other joints
- **Does:** the same protocol as the index fe1 runs, on index fe2 and on fe1 of the middle finger or thumb.
- **Why:** to check whether 0.24 mN·m per load unit holds across servos, or needs a value for each one.

### 6. Stall heating
- **Does:** hold one servo at load ~300 against the scale for up to 60 s, logging temperature,
  with a 45 °C abort.
- **Why:** it sets a safe grip duration and load limit for the policy.

### 7. Full-hand logging rate — DONE 2026-10-29 (see README)
- **Does:** a realistic control loop: sync goal writes plus position and load reads on all 9 servos.
- **Measures:** whether the ~55 Hz position + load rate holds with commands mixed in.
- **Needs:** FTDI latency at 1 ms.

### 8. Fit and check the model
- Fit BAM friction (static friction plus lock-in) to tests 1–5.
- Replay the recorded commands in MuJoCo and compare simulated position and load against the
  real logs.
- **This is the gate for trusting simulated load in policy C.**

---

## Rules for every moving test
- The operator OKs each test before it runs.
- **Safety built into every script:**
  - goal set to the current position before torque goes on;
  - targets checked against the software limits;
  - jam, load and temperature aborts;
  - a wall-clock torque-off backstop independent of the script's logic;
  - torque off in `finally`.
- Fingers stay spread on the gantries, with the outer corners homed.
- No adduction (aa) move larger than ±10° until the limit question below is settled.

## Open questions
1. **Adduction limits:** the driver allows ±85°, but older manta-hand measurements were
   narrower (e.g. −70…+75°). Which is current?
2. **Tests 3 and 4 need simple props,** a weighted string and a pendulum arm. Do them, or fit
   from tests 1, 2 and 5 only?
3. **FTDI latency** is back at 16 ms since the U2D2 reconnected. Tests 2 and 7 need 1 ms (run on
   the CB1; it needs `sudo`):
   `echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer`
4. Add or drop anything? For example, TPU (thermoplastic polyurethane) pad contact tests for grasping.
