# The first bench session on the Sobol hands — 20 runs, and what the servos actually did

**2026-08-30, pulled off the CB1 the same evening.** Three of the promoted Sobol-128 designs were
run open loop on the hardware: `sv1_u0060_b75` (6 runs), `sv1_u0100_b70` (12), `rv05_manual_b85`
(2). Every run completed. The user's verdict on the session was that **`rv05_manual` and
`sv1_u0100` "worked really well"**.

**No run carries a rotation angle.** `manual_score` is null on all 20, which by the bench
protocol's own rule ([`../20260830-real_v1_bench_suite/README.md`](../20260830-real_v1_bench_suite/README.md))
means the session produced no *result* — a rotation number with a repeat count and a stated
measurement method. What it produced instead is 20 complete command/telemetry traces, and those
turn out to say something the eyeball could not.

## The hand performs half to three-quarters of the turn it is told to

The exported plan is open loop, which is exactly the statement that commanded == achieved. The
station logs both sides of that assumption. Per joint, over the whole turn:

    achieved fraction = (measured after − measured before) / (commanded end − commanded start)

| design | runs | thumb yaw | thumb mcp | index yaw | index mcp | middle yaw | middle mcp | middle pip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sv1_u0060` | 6 | 0.90 | **−2.09** | 0.75 | 0.66 | **0.86** | 0.77 | 1.05 |
| `rv05_manual` | 2 | 0.22 | −0.34 | 0.44 | 0.75 | **0.69** | 0.54 | 1.04 |
| `sv1_u0100` | 12 | **−0.00** | −0.08 | 0.55 | 0.60 | **0.56** | 0.69 | 0.99 |

(medians; per-run values in [`commanded_vs_achieved.json`](commanded_vs_achieved.json),
regenerate with `scripts/real_v1_bench_arrival.py --logs logs/`.)

Three things in that table matter.

**The pip joints arrive and the yaw joints do not.** Every `pip` sits at 1.00 ± 0.05. Yaw ranges
from 0.44 to 0.90, and yaw is the joint the turn is built out of.

**It is not speed.** The session swept `speed_ratio` from 0.10 to 1.50. `sv1_u0100`'s middle yaw
reads 0.56 at 0.10 and 0.56 at 1.50. A shortfall invariant to how slowly you ask for it is not a
following error.

**It is torque.** The terminal `servo_load` on middle yaw (servo id 6) is 405–495 on `sv1_u0060`
and **900–915** on `sv1_u0100` — against the `overload_torque = 80%` trip that the 2026-08-30
protection experiment put near 800
([`TORQUE_MODEL.md`](../20260830-real_v1-sobol4096/TORQUE_MODEL.md)). One `sv1_u0100` run sits at
exactly 200.0, the
`protective_torque = 20%` plateau this program has seen before. The joint that falls furthest
short is the joint that is stalled.

## What that does to the plan

The residual clip is a per-joint cap on commanded travel, and on this evidence roughly half of the
commanded middle-yaw travel is not delivered. The clip that the *hand* executes is therefore well
below the clip in the plan's metadata, by a design-dependent factor — 0.86 for `sv1_u0060`, 0.56
for `sv1_u0100`.

That is an order of magnitude larger than the 4–6° of yaw droop measured on 2026-08-29, which is
what the current ship rule's `+0.10 rad` margin was sized against. The margin is not wrong, it is
small. Nothing here says what the right correction is: an achieved-travel factor is not a clip,
because the clip binds different joints at different points of the ramp.

## What this session cannot say

`sv1_u0060` achieves the most of its commanded turn (0.86) and has the best simulated alignment of
the three; it is not one of the two the user called out. `sv1_u0100` achieves the least, has a
completely idle thumb, and is. With no rotation measurement on any run, that comparison has no
data in it — which is the argument for the AprilTag vane instrument the bench suite already
specifies, not an argument about the hands.

## Files

- [`logs/`](logs/) — the 20 runs as pulled from `irlab@10.99.99.2:~/hand/logs/hardware`, JSONL
  plus per-run summary. **The CB1's clock is ~14.5 h behind** (no RTC, no NTP on that subnet), so
  the timestamps in the filenames order the runs correctly but do not date them.
- [`commanded_vs_achieved.json`](commanded_vs_achieved.json) — per run, per joint: commanded
  travel, achieved fraction, and the terminal servo loads.
