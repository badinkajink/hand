# The servo the simulator does not have — 236 bench runs, 2026-09-02

Reproduce with

```bash
python3 scripts/fit_servo_compliance.py \
  --logs docs/experiments/20260902-cb1-log-archive/logs \
  --out  docs/experiments/20260902-servo-compliance/fit.json
```

Source: every run log the CB1 has kept, 2026-08-30 to 2026-09-01, archived to
[`../20260902-cb1-log-archive/logs`](../20260902-cb1-log-archive/logs). 236 of 244 runs carry
servo feedback, over 11 morphologies; 64 of them also carry `servo_load`, over 6.

## 1. What the sim assumes

Every `real_v1` scene actuates its fingers with

```xml
<position class="ctrl" kp="30" kv="0.5" gear="1" forcerange="-10 10" ctrlrange="-1 1" />
```

and no MJCF in the repo carries a `frictionloss`.

> **Corrected 2026-09-02.** An earlier version of this file quoted `kp="4000"` here. That is the
> `pose` class, which drives the palm/arm actuators; the nine finger actuators are `class="ctrl"`
> and ship at **`kp=30`**, confirmed from the compiled model's `actuator_gainprm` in the deploy
> scene, `hand_frozen_morphology.xml` and `real_hand.xml` alike. The conclusion is unchanged and
> the factor is smaller: calibrating against the bench's own deficits puts the effective gain at
> about **0.5**, so the shipped model is roughly **60×** too stiff, not 250×. See
> `../20260902-servo-sysid/kp_calibration.json`. The exported plans are therefore written in commanded units and
assume the hand arrives; [`../20260830-real_v1_bench_sobol`](../20260830-real_v1_bench_sobol)
already established that it does not.

## 2. The servo is a spring, and one spring for the whole hand

Deflection is measured as `commanded - achieved` at the terminal hold — an absolute angle, not
the `achieved_fraction` ratio of deltas that `real_v1_bench_arrival.py` reports, because a
stiffness is an absolute quantity and a joint already sagging at `before` cancels part of its own
sag out of a delta. The `after` telemetry is taken a median 1.92 s after the last command
(minimum 1.52 s), so these are settled values, not tracking lag.

With the tripped samples of §3 removed:

| joint | n | k (deg/load) | b (deg) | R² | rms |
|---|---|---|---|---|---|
| `thumb_yaw` | 64 | 0.01820 | −0.43 | 0.966 | 0.46 |
| `thumb_mcp` | 64 | 0.01769 | +0.01 | 0.954 | 0.46 |
| `index_yaw` | 56 | 0.01883 | −0.17 | 0.938 | 0.53 |
| `middle_yaw` | 37 | 0.01981 | −1.08 | 0.987 | 0.56 |
| `index_pip` | 64 | 0.00991 | +0.29 | 0.826 | 0.19 |
| `thumb_pip` | 64 | 0.01231 | +0.14 | 0.678 | 0.35 |

Four joints that carry real load agree on **0.0177–0.0198 deg per load unit** with R² 0.94–0.99
and rms about half a degree. This is one servo characteristic, not nine. The remaining joints
(`index_mcp`, `middle_mcp`, `middle_pip`) sit at or near zero load — `middle_pip` commands a
median 48.7° and arrives within 0.23° — so their fits are unidentifiable rather than different.

**Intercepts run −1.08 to +0.54 deg.** There is no stiction or backlash term to model. An earlier
pooled fit over all joints reported a 2.27° intercept; that was the tripped samples of §3 leaking
into a regression that had no term for them, and it is withdrawn.

`servo_load` is the SCS0009's PWM-duty proxy on a 0–1000 scale, not torque in N·m, so `k` is not
yet a MuJoCo `kp`. It becomes one by matching the observable: run a design's own plan in sim,
sweep `kp`, and take the value whose simulated `ctrl − qpos` at the hold reproduces that design's
measured deflection. That is one scalar, fitted against a quantity both sides can see, and it
never needs the unit conversion.

## 3. Two of nine joints are not sagging — they have quit

Reported load takes 51 distinct values across the dataset. Fifty of them are multiples of 15.
The exception is **200**, which is `protective_torque` = 20 percent of the 0–1000 scale: a
register, not a reading. The SCS0009 drops its output there after sustained overload, so a servo
reading exactly 200 has unloaded itself and its joint is being pushed by whatever else is
touching it.

| joint | trip rate | \|defl\| untripped | \|defl\| tripped | sd untripped | sd tripped |
|---|---|---|---|---|---|
| `middle_yaw` | **42 %** | 12.16° | 21.37° | 4.82 | 8.65 |
| `index_yaw` | 12 % | 6.54° | 13.74° | 2.13 | 0.97 |

`middle_yaw` — the joint that drives the turn — is in overload protection on **42 % of bench
runs**. Tripping nearly doubles its deflection and roughly doubles the scatter.

Excluding those samples is what takes `middle_yaw` from R² 0.025 to **0.987** and `index_yaw`
from 0.044 to 0.938. The two regimes are not one noisy phenomenon; they are a spring and a
cliff, and pooling them hides both.

This also answers, from data already on disk, the 2×2 that
[`../20260830-real_v1_bench_suite/README_open_loop.md`](../20260830-real_v1_bench_suite/README_open_loop.md)
proposed and never ran. Its four readings were mutually exclusive; the measurement says the
ceiling is **both** of the first two, separated by the trip point. Below the trip the joint is a
linear spring, entirely predictable, and commanding further does buy travel. At the trip the
actuator is gone and no additional command reaches the joint. The register write test
(`protective_torque` 20 → 40) is still worth running, and now has a specific prediction: it
should move `middle_yaw`'s 42 % of runs onto the linear branch, where the same load buys
0.0198 deg/unit instead of an uncontrolled 21°.

All 64 load-carrying runs are from 2026-08-30 at stock settings — maximum observed load 990 —
so nothing here observed a raised register. The 2026-08-31 and 09-01 transfer runs did not record
`servo_load`; **that field should be turned on for every future session**, since without it the
tripped and untripped branches cannot be told apart after the fact.

## 4. What a calibration can fix, and what needs feedback

Deflection variance split into between-design (a per-design feedforward correction removes it)
and within-design (it varies trial to trial on the same hand, so only feedback catches it):

| joint | sd total | sd within | sd between | % within |
|---|---|---|---|---|
| `index_yaw` | 2.57 | 1.12 | 2.32 | 19 % |
| `middle_yaw` | 7.65 | 4.19 | 6.40 | 30 % |
| `middle_mcp` | 2.00 | 1.09 | 1.68 | 30 % |
| `thumb_mcp` | 2.41 | 1.69 | 1.72 | 49 % |
| `thumb_yaw` | 1.76 | 1.36 | 1.11 | 60 % |
| `index_mcp` | 0.71 | 0.67 | 0.22 | 90 % |

The large deflections are mostly between-design, so most of "the hand performs half its commanded
turn" is recoverable by a feedforward correction that costs nothing at run time.

What survives that correction is the case for a closed loop: **4.19° of `middle_yaw` scatter
remains trial-to-trial on the same hand**, on the joint that drives the turn, and no open-loop
plan can see it. That is the same failure the transfer study measured from the other end — six
drops that reached a mean peak of +0.608 and then collapsed to +0.109, losing a turn that had
already been won.

## 5. Consequences

1. The plant is three parameters, all measured: a linear stiffness (`kp`), a torque ceiling
   (`forcerange`, currently ±1000 and therefore absent), and nothing else. No `frictionloss`.
2. A rigid deterministic simulator makes open-loop optimal, which is the honest reading of
   `b33` ignoring its observations under a replay ablation. A compliant, saturating, per-design
   plant does not, and the disturbance it creates is visible in `joint_pos`, which the bench
   reads at 111 Hz.
3. Turning `servo_load` on for every session is free and it is the field the whole of §3 rests on.
