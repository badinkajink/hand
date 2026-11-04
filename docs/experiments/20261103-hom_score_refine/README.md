# Grasp, force and contact-mode refinement on the HOM score: runbook (2026-11-03)

Owner, 2026-11-03 ~18:00: "can't we score and refine a grasp based off how well the grasp (+morphology) executes the
HOM control? from what i understand we are doing pretty random sampling"; "is it possible to co-design HOM? are there
hyperparameters / parameters that we can change within it?"; 18:1x: "yea, let's send it on these tasks ... we can try
out multiple scores, set up different autopilot jobs to continue promising work. make sure things are documented well!"

## What runs

`scripts/hom_grasp_refine.py` scores each sampled layout three ways (the module docstring has the details):

| score | HOM | grasp | searched by |
|---|---|---|---|
| S1 | defaults (`hom_tasks.PRM`) | best of the layout's grasp set (up to 16) | the grids of `hom_plant_transfer.py` |
| S2 | defaults | refined | CEM from S1's grasp, 24 candidates x 8 iterations |
| S3 | forces and contact modes refined | refined | CEM from S2's grasp, 32 candidates x 8 iterations |

The search scores every candidate at seeds 0-2, as the grids scored S1; `validate` runs the three picks of each layout
at seeds 20-24, which no grid or search used, and those scores are the ones to compare. The contact modes are Eq. (2)'s
row weights by finger role (`hom_tasks.mode_weights`, new keys in `hom_tasks.PRM` with the old values as defaults); the
forces are `f_tripod`, `f_pinch`, `f_drive` and `f_press`. `hom_gpu.Batch` takes them per world (a world's `prm`).

The queue `queue.sh` (a copy of `logs/20261103-hom_score_refine/queue.sh`, which runs) does, one GPU job at a time:

1. `sv25` (the SCS0009 with pads at 2.5 MPa, the hand's servos): 60 layouts per task (30 best by S1 and 30 at random),
   levels grasp and full, validation, report. About 5.2 h.
2. `k21` (the STS3032 at kp 2.1 N m/rad, friction 1.0), then `k21mu0p4` (the same at friction 0.4, TPU on bare PLA) on
   the same 30 layouts (20 best by k21's S1, 10 at random), 6 iterations each. About 2 h each.
3. The picks tuned at each friction run at the other (`validate --picks-from`): what knowing the friction is worth when
   the forces and modes can be retuned. About 20 min.

## Files

- `rows/<plant>/<task>_layouts.json`: the layout sample, with S1 and its grasp.
- `rows/<plant>/<level>/<task>_d<d>.jsonl`: every rollout, in `hom_gpu.py`'s format; the key holds `call`, `it`,
  `cand`, `v` (the toy's grasp variables), `prm` and `x` (the CEM variables). `_gens.jsonl`: one line per layout and
  iteration (scores, elite mean, best so far, the CEM mean and spread). `_timing.jsonl`: one line per GPU batch.
- `rows/<plant>/val/` and `rows/<plant>/val_from_<other>/`: the picks at seeds 20-24.
- `rows/<plant>/<date>-score_levels.json`: the report (`report`): per layout S1, S2 and S3 at search and validation
  seeds, the twist's ramp angle, the tuned forces and modes; per task medians, Spearman's rho between levels, the best
  layout under each score.
- `rows_smoke/`: the smoke test of 18:22 (two twist layouts, two iterations); not results.

## Rerun or resume

```bash
nohup setsid bash logs/20261103-hom_score_refine/queue.sh >> logs/20261103-hom_score_refine/queue.log 2>&1 &
```

A finished step leaves `logs/20261103-hom_score_refine/Q_<step>.done`; a stopped step resumes from its rows (the
iterations are replayed with the same random draws, and a replay that draws another candidate stops with an error).
Each step goes through `resguard.sh run --mem 5G` and retries while resguard refuses it.

## Compute tracking

Owner, 2026-11-03 19:0x: "let's make sure we're timing this all for compute-tracking purposes and throughput
measuring, including cpu vs gpu speedups". Three records, all written as the work runs:

- each GPU batch: one line in `<rows>_timing.jsonl` beside its rows (`hom_gpu.batch_child`: worlds, `build_s`, `run_s`,
  `score_s`, `max_mem_gb`, `when`);
- each search iteration: the line `HH:MM:SS <task> <level> it k: ... rollouts run, draw X s, all Y s` in
  `logs/20261103-hom_score_refine/Q_<step>.log`;
- each queue step: `peak <kB> kB, <s> s` from `/usr/bin/time` at the end of the same log.

`scripts/hom_compute.py` reads them. `ledger` sums every HOM GPU timing line since 2026-10-29 (this study and the grids
and searches under `docs/experiments/20261029-foundational_pose_review/rows/`) into `compute/<date>-ledger.json`.
`cpu-bench` times the CPU runner on one core on the same layouts, grasps, plant and seed, in two variants: as the CPU
checks run it, and with the GPU's contact filter and solver settings. `gpu-bench` times GPU batches of 64 worlds up to
the queue's batch size, and runs only while no GPU queue does. `report` writes `compute/<date>-throughput.json` with the
speedups of one GPU against one CPU core at three clocks: the rollout loop, the whole batch, and the whole queue step.
Ledger of 2026-11-03 18:5x: 3,270,293 HOM rollouts on the GPU since 10-29, in 2,442 batches and 35.2 h of batch time.

## Checks

- `logs/20261103-hom_score_refine/check_prm.py` (18:2x): default worlds reproduce the grid's rows within the GPU's
  run-to-run spread (twist 1.0 = 1.0; pin-push within 0.036); a world with changed forces and modes scores the same in a
  mixed batch and alone (within 0.01); the CPU runner with the same prm agrees (pin-push 0.130 vs 0.132, twist ramp 91.8
  vs 94.9 deg).
- The probe (`hom_grasp_refine.py probe`): a feasibility draw costs about 3 ms and an open pose 1-2 ms; 19-53 % of
  draws at 0.25 of the feasible widths pass.

## Outcome (2026-11-04)

Page: `20261103-hom_score_refine.html` beside this file, built by `scripts/hom_score_refine_page.py` (`figs`, `films`,
then the page). The results quote the elitist picks (`hom_grasp_refine.elitist`; fields `s2e_*`, `s3e_*` and
`below_start_*` in the reports), because the night's searches scored draws around each start and never the start
itself: 7 of 240 grasp searches and 5 of 240 full searches on the SCS0009 picked a candidate below their start.

| SCS0009, 30 best layouts, seeds 20-24 | S1 | S2e | S3e | rho(S1, S3e) over 60 |
|---|---|---|---|---|
| twist | 0.89 | 0.95 | 1.00 | 0.75 |
| roll | 0.56 | 0.58 | 0.60 | 0.56 |
| pin-push | 0.70 | 0.72 | 0.75 | 0.71 |
| press | 0.70 | 0.75 | 0.91 | 0.67 |

The best layout moves under S3e on the twist, the roll and the press. The S3 search lowered the press's pinch force
from 3 N to a median 1.8 N and the pusher's roll weight from 1 to 0.38. On the STS3032 the picks tuned at friction 1.0
score a median 0 at friction 0.4 on the twist, the roll and the pin-push; retuning at 0.4 gains a median 0.43, 0.51,
0.53 and 0.18. The pose toy's contact normals of both sets lie inside the 0.4 friction cone, and the friction needed for
static balance is the same in both sets, so the drops have no named cause (`media/20261103-friction_check.json`).

From 2026-11-04 the search scores its start as candidate 0 of iteration 0, so its pick is elitist by construction, and
each iteration writes one line to `<task>_d<d>_iters.jsonl` (draw, open-pose and GPU-batch seconds, batches,
rollouts). The compute ledger, the GPU and CPU benchmarks and the speedups are in `compute/`
(`scripts/hom_compute.py`; `report` keeps the newest CPU session per plant, task and variant).

## Worst-case score over three frictions (2026-11-04)

Owner, 2026-11-03: "maybe do need to do some online sysid to measure friction, and/or to have robustness to a band of
physical parameters as a scoring parameter in hand selection"; "worst-case sampling makes sense, i saw it in spider
too". Autopilot job `hand-20261104-hom_worst_case_score`; the page's Section 6.

`hom_grasp_refine.py --band k21mu0p4,k21mu0p7,k21` (the band `band_k21mu` of `BANDS`) runs each candidate at seeds 0-2
on the STS3032 at kp 2.1 at scene friction 0.4, 0.7 and 1.0 (scene caches `logs/20261102-hom_servo_plant/scenes_mu0.4`,
`scenes_mu0.7` and `scenes`) and scores its lowest three-seed mean; ties go to the larger margin at that friction. The
open pose does not depend on the friction, so one open pose per candidate is used on all three plants. The parent sets
`HOS_PLANT`, `HOS_SCENES` and `HOM_ROWS_TAG` before it spawns each `hom_gpu.batch_child`, and stops if a row reports
another scene cache. `queue_band.sh` is a copy of `logs/20261104-hom_worst_case/queue.sh`, which ran 04:22-08:54:

```bash
B=k21mu0p4,k21mu0p7,k21
python3 scripts/hom_grasp_refine.py validate --task twist --band $B --picks-from k21        # 11-03 picks at 0.7
python3 scripts/hom_grasp_refine.py run --task twist --level grasp --band $B --layouts-from k21 --iters 6
python3 scripts/hom_grasp_refine.py run --task twist --level full --band $B --layouts-from k21 --iters 6
python3 scripts/hom_grasp_refine.py validate --task twist --band $B                         # S1w, S2w, S3w
python3 scripts/hom_grasp_refine.py report --band $B
nohup setsid bash logs/20261104-hom_worst_case/queue.sh >> logs/20261104-hom_worst_case/queue.log 2>&1 &
```

Files under `rows/band_k21mu/`:

- `<task>_layouts.json`: the k21 sample (20 best by S1 at 1.0, 10 at random) with S1w, the grasp of the grasp set with
  the best worst-case score on the three grids, and each plant's mean.
- `<level>/<plant>/<task>_d<d>.jsonl` and `_timing.jsonl`: each plant's rollouts and GPU batches;
  `<level>/<task>_d<d>_gens.jsonl` (per layout and iteration, with `best_plants`) and `_iters.jsonl` (per iteration,
  with `gpu_s_plants`).
- `val/<plant>/`: S1w, S2w and S3w at seeds 20-24. `val_from_k21/` and `val_from_k21mu0p4/`: the 11-03 picks at 0.7;
  their rows at 0.4 and 1.0 are the 11-03 ones (read, never appended to).
- `20261104-band_levels.json`: the report. `20261104-cpu_check.jsonl`: every pick on the CPU plant at seed 20, at 0.4
  and 1.0 (`band_cpu_check.py` beside this file, a copy of `logs/20261104-hom_worst_case/cpu_check.py`).
  `20261104-queue_steps.json`: each queue step's wall clock and peak RSS.

Checks before the queue (`logs/20261104-hom_worst_case/check_band.py`, `check*.log`; rows in `rows_smoke/`, not
results): candidate 0 of iteration 0 against the grid rows, at most 0.025 apart over 18 rollouts. A plant's band rows
against a single-plant run of the same candidates: three-seed means a median 0.0004-0.0012 apart, at most 0.24. An
identical batch against its repeat: a median 0.0001-0.0010 apart, at most 0.22. The large differences are the GPU's own
spread on marginal twist candidates.

Outcome, at seeds 20-24 over the 30 layouts per task (the last column at seed 20, the lower of the scores at 0.4 and
1.0):

| task | lowest score over 0.4-1.0: S3w | picks tuned at 1.0 | picks tuned at 0.4 | at 1.0: S3w less the 1.0 picks | CPU, every contact: S3w less the 1.0 picks (GPU) |
|---|---|---|---|---|---|
| twist | 0.28 | 0.00 | 0.17 | -0.46 | +0.00 (+0.27) |
| roll | 0.57 | 0.00 | 0.53 | -0.05 | +0.01 (+0.47) |
| pin-push | 0.59 | 0.00 | 0.63 | -0.12 | +0.55 (+0.54) |
| press | 0.92 | 0.42 | 0.88 | -0.02 | +0.10 (+0.11) |

The paired change of the lowest score, S3w less the picks tuned at 1.0, has a median of +0.26, +0.32, +0.56 and +0.21.
Against the picks tuned at 0.4 it is +0.08, +0.00, +0.00 and +0.01. Friction 0.7 gives the lowest validated score of 1
to 4 of the 30 worst-case picks per task. The layouts are reordered: Spearman's rho between the lowest score under S3w
and the score at 1.0 under the picks tuned there is 0.27 to 0.51, and the best layout moves on the roll, the pin-push
and the press. On the CPU plant with every contact, S3w scores 0 at 0.4 on 16 of the 30 twist layouts at seed 20, 7 on
the GPU; with the GPU's contact filter the CPU's twist change is +0.28. The runs took 371,871 GPU rollouts in 308
batches, 4.30 h of batch time in 4.53 h of queue steps (34 ms run, 42 ms batch and 44 ms step time per rollout). The
CPU runner at friction 0.7 takes 1.8 to 3.2 s a rollout on one core, 72 to 82 cores per GPU.
