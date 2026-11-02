# SCS0009 BAM (Better Actuator Models) identification — results, 2026-10-29

## Pipeline
| Step | Command (from this folder) |
|---|---|
| Process logs | `../.venv-bam/bin/python prepare_logs.py` → `data_processed/single/` (81 logs) |
| Fit | `../.venv-bam/bin/python fit_bam.py --logdir data_processed/single --actuator scs0009 --model m6 --output fits/m6.json --trials 20000 --validation_kp 4` |
| Score single pendulum | `../.venv-bam/bin/python eval_fits.py --fits fits_best --plot fits_best/compare.png` |
| Double-pendulum replay (MuJoCo) | `../.venv-bam/bin/python double_replay.py --fits fits_best --plot fits_best/double_replay.png` |

BAM itself is the unmodified upstream checkout at `../bam_upstream` (commit b9e1111), in the Python 3.12 env `../.venv-bam`.
`bam_scs0009.py` adds the actuator and a pendulum testbench that uses measured mass properties.
`fits_best/` holds the better of a 5,000-trial run and a 20,000-trial run for each model, chosen by training score.
The optimizer (CMA-ES, Covariance Matrix Adaptation Evolution Strategy) varies from run to run, so rerun a model if it looks stuck.

## Firmware control law (measured, not fitted)
The load register is the PWM duty × 1000. Across 205k samples it fits:

`duty = 0.189·kp·err + 0.036·sign(err)`

- The duty is zero inside the 1-step dead zone and clips at ±1.
- The 0.036 term is the "minimum startup force" setting (45).
- The fit's residual is 1.6% of full PWM.

## Single pendulum (servo 20, 48.08 g at 30–90 mm, D = 0)
Mean absolute position error in degrees. The P gain 4 logs were held out of every fit.

| Model | Train | Held-out P gain 4 |
|---|---|---|
| M1 Coulomb + viscous (what MuJoCo can do natively) | 1.32 | 1.12 |
| M2 + Stribeck | 1.08 | 0.87 |
| M3 load-dependent | 1.10 | 0.89 |
| M4 Stribeck load-dependent | 1.07 | 0.85 |
| M5 + directional | 1.07 | 0.83 |
| M6 + quadratic | 0.95 | **0.64** (−43% against M1) |

## Double pendulum (servo 20 + servo 21, D = 0, nothing fitted on this data)
Both joints use servo 20's fitted model, so joint 2 also tests transfer to another unit. Zero offsets come from the hanging readings.

| Model | Mean tip error over all 16 logs (mm) |
|---|---|
| M1 | 3.17 |
| M2 | 3.61 |
| M3 | 2.51 |
| M4 | **2.45** |
| M5 | 2.62 |
| M6 | 2.71 |

- At P gain 15, every model is about 1.5–1.8 mm.
- The differences show at P gain 4, and in the elbow-up set where joint 1 carries more load.
- There, load-dependent friction (M3–M6) cuts the error by 30–40% against M1.

There are two test sets:
- `double_v2`: elbow down, the intended paths.
- `double_v2_elbow_up`: servo 21's direction was mirrored, recorded before the sign fix in the recorder.

## Caveats
- **Motor constants:** the fitted kt and R aren't individually physical; for example, M6 implies a 0.5 N·m stall against the 0.18 N·m spec.
  - The load reached at most ~26% of stall, so motor torque and load-dependent friction trade off against each other.
  - A heavier mass (~100 g) would pin them down.
  - The fitted models still predict motion well inside the tested range.
- **Joint zeros:** these are weakly determined, because friction (~45 mN·m) is about the arm's peak gravity torque, so the arm can rest several degrees off plumb.
  - Fitting the offsets on the double pendulum moved them by about 10° and barely changed the ranking.
- **D gain:** the hand runs the servo default D = 15, and these models are D = 0 only.
  - The firmware D term still has to be identified, for example from single-pendulum runs at D = 15.

## 2026-10-29: double pendulum at higher P gains (D = 0)
Question: the hand's index fe1 tracked better at P 30–45 (D 15; sysid/README gain sweep). Does that
hold on the pendulum rig, and do the BAM fits (single pendulum, P ≤ 15, D 0) predict it?
Recorded `data_raw/double_v2_highkp/` (4 paths × P 30 and P 45, D 0, elbow down, ~270 Hz; log
`data_raw/double_v2_highkp_record_log.txt`). No aborts; servo 20 warmed 21 → 37 °C over the 8 runs.
Replay: `../.venv-bam/bin/python double_replay.py --fits fits_best --sets double_v2_highkp double_v2
--plot fits_best/double_replay_highkp.png` (the table now lists every P in the logs).

Real tracking (RMS goal − position during the path; D = 0 throughout):

| P | joint 1 (servo 20, carries link 2) | joint 2 (servo 21) | j1 load at 1000 (full PWM) | j1 direction reversals |
|---|---|---|---|---|
| 4 | 4.66° | 8.80° | 0% | 2.9 /s |
| 15 | **1.80°** | 2.78° | 0% | 4.5 /s |
| 30 | 2.02° | **1.59°** | 0% | 9.7 /s |
| 45 | 3.58° | 1.63° | 4.4% | 18.1 /s |

BAM replay, tip error (mm), all six models within 0.1 mm of each other: P 15 1.8, P 4 3.9–4.4,
**P 30 3.5, P 45 8.0**.

- **At D = 0, raising P makes joint 1 oscillate.** At P 30 it limit-cycles by about ±5° wherever
  it hangs near plumb with little gravity load (start and end of every path); where gravity loads
  it (q1 −30…−60°) it tracks cleanly. At P 45 the oscillation is worse, the servo hits full PWM,
  and it heats. Classic backlash limit cycle: unloaded, the gear train rattles across its slack.
- **Joint 2 does improve** (2.8° → 1.6° at P 30), like index fe1 on the hand.
- **The fits do not predict the oscillation** (no backlash in the BAM models), so their error grows
  from 1.8 mm at P 15 to 3.5 / 8.0 mm. The fits are only trustworthy at the gains they were fitted on.
- The hand runs **D = 15**, which damps exactly this, and index fe1 at P 30/45 D 15 showed no
  buzzing. So these D = 0 runs don't rule higher P out for the hand; they show it needs D = 15
  data to judge, which is also what the BAM fit needs to model the hand's real controller.

## 2026-10-29: firmware D term (the hand runs D = 15)
Model: `duty = 0.189·kp·err + 0.036·sign(err) − 0.0015·kd·dq` (dq in rad/s), zero inside the dead
zone, clipped to ±1. `bam_scs0009.py` (`D_GAIN`, model parameter `d_gain`, not fitted by default;
`kd` read from each log, 0 when absent, so the single-pendulum fits are unchanged).
- **Measured** from the load register (`d_term_check.py`): the residual of the P law tracks joint
  velocity (not error rate) with scale 0.0011–0.0022 across P 15–45 / D 15–150.
- **Confirmed by replay** (`d_gain_check.py`, M4, high-D logs P 22–30 / D 45–150, MuJoCo):
  mean joint error j1/j2 1.02/0.70° with no D term → **0.85/0.36° at 0.0015** (optimum 0.0015–0.00175).
- At P 15 / D 15 the D term changes simulated position little (j2 0.51 → 0.41° MAE), tip 1.9 mm.
- **Simulated load vs real** (P 15 / D 15, 50 ms smoothing): joint 2 correlation 0.98–0.99, mean
  abs error ~29 load units on a mean |load| of ~135; joint 1 correlation 0.86–0.93, error ~45 on ~95.
  First evidence that simulated load can stand in for the real signal, on the rig.
