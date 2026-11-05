# Grasp, force and contact-mode refinement on the HOM score: runbook (2026-11-03)

Owner, 2026-11-03 ~18:00: "can't we score and refine a grasp based off how well the grasp (+morphology) executes the
HOM control? from what i understand we are doing pretty random sampling"; "is it possible to co-design HOM? are there
hyperparameters / parameters that we can change within it?"; 18:1x: "yea, let's send it on these tasks ... we can try
out multiple scores, set up different autopilot jobs to continue promising work. make sure things are documented well!"

## What runs

`scripts/hom/hom_grasp_refine.py` scores each sampled layout three ways (the module docstring has the details):

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

`scripts/hom/hom_compute.py` reads them. `ledger` sums every HOM GPU timing line since 2026-10-29 (this study and the grids
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

Page: `20261103-hom_score_refine.html` beside this file, built by `scripts/hom/hom_score_refine_page.py` (`figs`, `films`,
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
(`scripts/hom/hom_compute.py`; `report` keeps the newest CPU session per plant, task and variant).

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
python3 scripts/hom/hom_grasp_refine.py validate --task twist --band $B --picks-from k21        # 11-03 picks at 0.7
python3 scripts/hom/hom_grasp_refine.py run --task twist --level grasp --band $B --layouts-from k21 --iters 6
python3 scripts/hom/hom_grasp_refine.py run --task twist --level full --band $B --layouts-from k21 --iters 6
python3 scripts/hom/hom_grasp_refine.py validate --task twist --band $B                         # S1w, S2w, S3w
python3 scripts/hom/hom_grasp_refine.py report --band $B
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

## Forces and contact modes shared by all layouts on the SCS0009 (2026-11-04)

Owner, 2026-11-03: "is it possible to co-design HOM? are there hyperparameters / parameters that we can change within
it?"; HOM fixed "gives us a fixed reference to evaluate morphologies". Autopilot job `hand-20261104-hom_shared_controller`;
the page's Section 7.

`hom_grasp_refine.py run --level shared` searches one set of commanded forces and contact-mode weights per task, the
shared set, by CEM over S3's force and mode variables (their ranges, starting spread and floor) with no grasp variable:
24 candidates, 8 iterations, HOM's defaults as candidate 0 of iteration 0. Every candidate runs on 40 layouts of the
sv25 sample with S1's grasps at seeds 0-2: the sample's 30 best by S1 and 10 of its 30 random layouts. The sample stores
its random layouts in lattice order, so the ten are drawn from the 30 with numpy seed 0 (`shared_layouts`). A
candidate's score is the mean of the layouts' three-seed means; the lowest layout's mean is recorded. Then
`hom_gpu.py run --prm JSON` rescores each task's grid with the pick (every world gets that set; each row records it in
its key), `validate --shared` runs the pick on the 60 sample layouts at seeds 20-24 with S1's grasp (S1sh) and with the
best grasp of the layout's set on the rescored grid (Sh), and `report --shared` compares the two grids as
`hom_plant_transfer.compare` does. `queue_shared.sh` is a copy of `logs/20261104-hom_shared_controller/queue.sh`:

```bash
python3 scripts/hom/hom_grasp_refine.py run --task press --level shared
P=$(python3 scripts/hom/hom_grasp_refine.py prm --task press)          # the pick, from rows/sv25/shared/press_d25_pick.json
HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25_shared \
  python3 scripts/hom/hom_gpu.py run --task press --layouts grid2dff_size --d 25 --seeds 0 1 2 --variants hold \
  --batch 1536 --out press_d25 --prm "$P" \
  --rows-dir docs/experiments/20261029-foundational_pose_review/rows/grid2dff_v4b_sv25_shared
python3 scripts/hom/hom_grasp_refine.py validate --task press --shared
python3 scripts/hom/hom_grasp_refine.py report --shared
nohup setsid bash logs/20261104-hom_shared_controller/queue.sh >> logs/20261104-hom_shared_controller/queue.log 2>&1 &
```

Files:

- `rows/sv25/shared/<task>_d<d>.jsonl` and `_timing.jsonl`: every rollout and GPU batch of the search; `_gens.jsonl`, one
  line per iteration (every candidate's mean and lowest layout score, the elites' mean, the best so far, the CEM mean
  and spread); `_iters.jsonl`, one line per iteration (draw, open-pose and GPU seconds); `_pick.json`, the pick and its
  start with each layout's mean.
- `rows/sv25/val_shared/`: S1sh and Sh at seeds 20-24. `rows/sv25/20261104-shared_levels.json`: the report.
- `docs/experiments/20261029-foundational_pose_review/rows/grid2dff_v4b_sv25_shared/`: the rescored grids.
- `rows/sv25/shared/cpu/`: the CPU runner with the shared set (`shared_cpu_bench.py` beside this file, which runs
  `hom_compute.py`'s cpu-worker with `hom_tasks.PRM` updated); `rows/sv25/shared/20261104-queue_steps.json`: each queue
  step's wall clock and peak RSS.

Smoke test 09:33-09:36 (`logs/20261104-hom_shared_controller/smoke.sh`; rows in `rows_smoke/`, not results): `--prm`
with HOM's default values reproduces the default grid's 96 rollouts of the same batch (largest difference 0.0004); the
search's start reproduces S1's three-seed means within the GPU's batch spread.

Outcome (queue 09:36-11:52 on 11-04, no failed step; medians at seeds 20-24; S1sh is S1's grasp and Sh the rescored
grid's best grasp, both with the shared set):

| task | shared set | search score, defaults and shared | 30 best: S1, Sh, S3e | 30 random: S1, Sh, S3e | grids: rho, best tenth kept |
|---|---|---|---|---|---|
| twist | HOM's defaults | 0.745, 0.745 | 0.89, 0.89, 1.00 | 0.23, 0.23, 0.62 | 0.96, 0.90 |
| roll | HOM's defaults | 0.540, 0.540 | 0.56, 0.56, 0.60 | 0.45, 0.45, 0.53 | 0.98, 0.92 |
| pin-push | pinch 2.89 N, drive 1.56 N, pusher roll 0.84 | 0.644, 0.648 | 0.70, 0.70, 0.75 | 0.45, 0.47, 0.67 | 0.96, 0.67 |
| press | pinch 1.73 N, press 0.53 N, pusher roll 0.35 | 0.621, 0.776 | 0.70, 0.89, 0.91 | 0.35, 0.72, 0.83 | 0.47, 0.28 |

On the press one shared set reaches 94 % of the per-layout gain on the 30 best layouts (Sh less S1 +0.19, S3e less S1
+0.20) and 67 % on the 30 random ones; it lowers the pinch force from 3 N to 1.73 N and the pusher's roll weight from 1
to 0.35, near the S3 picks' medians of 1.80 N and 0.38. It reorders the press landscape: rho 0.47 between the grids,
the median best-of-16 score from 0.42 to 0.78, the best layout from (70, 150) to (65, 140). On the twist and the roll no
candidate beat HOM's defaults, and the pin-push's pick scores 0.004 above them. The twist's and the roll's rescored
grids are therefore reruns of the default grids: 5 to 13 % of their rollouts move by more than 0.02 and up to 0.6 % by
more than 0.5. Their rho is 0.96 and 0.98, with 90 and 92 % of the best tenth kept and 79 and 82 % of the best grasps
kept: the evaluator's run-to-run spread for a whole landscape. Compute: 183,840 GPU rollouts, 2.07 h of batch
time in 2.17 h of queue steps (33, 40 and 43 ms per rollout of run, batch and step time). The CPU runner with the shared
sets takes 3.0 s (pin-push) and 2.0 s (press) a rollout on one core, 68 to 69 cores per GPU; with every contact the CPU
scores all 6 press rollouts within 0.05 of the GPU.

## Evolutionary search over layout and grasp on the SCS0009 (2026-11-04)

Owner, 2026-11-03: "can't we score and refine a grasp based off how well the grasp (+morphology) executes the HOM
control? from what i understand we are doing pretty random sampling". Autopilot job `hand-20261104-hom_grasp_genome_ea`;
the page's Section 8.

`scripts/hom/hom_ea_grasp.py` runs the evolutionary search of `scripts/hom/hom_ea.py` (the layout EA: an individual is a lattice
layout, its fitness S1, the best of its grasp set at seeds 0-2, up to 48 rollouts) with one grasp per individual (the
layout-and-grasp EA). Generation 0 is hom_ea's 64 layouts for the same search seed, each with its foundational pose. A
child keeps its parent's layout with probability 0.5; otherwise the layout moves as hom_ea's children do and the pose
toy re-solves the parent's grasp there (`fp_sr2_toy.Problem.solve` started from it; the layout's foundational pose if
that fails). The grasp then steps by N(0, (0.25 x the feasible widths)^2), redrawn until feasible
(`hom_grasp_refine.draw_layout`'s rule). The fitness is the grasp's mean score at seeds 0-2, 3 rollouts. Selection,
order and ties are hom_ea's; a search stops after the generation in which it reaches 6,000 rollouts. The three search
seeds of a task share one GPU batch per generation (about 288 worlds). `queue_ea_grasp.sh` is a copy of
`logs/20261104-hom_grasp_genome_ea/queue.sh`:

```bash
python3 scripts/hom/hom_ea_grasp.py run --task twist                     # search seeds 0-2, population 64, 6,000 rollouts
HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25 \
  python3 scripts/hom/hom_ea.py run --task twist --d 40 --seed 0 --budget 6000          # and seeds 1, 2
python3 scripts/hom/hom_ea_grasp.py validate --task twist                # both EAs' incumbents at seeds 20-24
python3 scripts/hom/hom_ea_grasp.py cpu --task twist                     # ... on the CPU plant at seed 20, one core
python3 scripts/hom/hom_ea_grasp.py report                               # rows/ea_grasp/sv25/<date>-ea_grasp.json
nohup setsid bash logs/20261104-hom_grasp_genome_ea/queue.sh >> logs/20261104-hom_grasp_genome_ea/queue.log 2>&1 &
```

Files under `rows/ea_grasp/sv25/`:

- `<task>_d<d>.jsonl` and `_timing.jsonl`: every rollout of the three searches (the key holds `pick`, the individual
  `s<seed>g<gen>c<child>`, with `run`, `gen`, `child`, `op`, `parent`, `resolved` and `v`) and every GPU batch;
- `<task>_d<d>_s<seed>_gens.jsonl`: one line per generation of a search (the population with its fitness, the
  incumbent, the rollouts so far, the children made by each operator), as hom_ea's gens files;
- `<task>_d<d>_iters.jsonl`: one line per lockstep generation (re-solve and draw, open-pose and GPU seconds);
- `val/`: both EAs' last incumbents at seeds 20-24; `cpu/cpu_check.jsonl`: the same on the CPU plant at seed 20;
- `<date>-ea_grasp.json`: the report (the grid's best and its replay of hom_ea, each search's curve, incumbent,
  rollouts to within 0.05 of the grid's best, validated and CPU scores; compute; the queue steps).

hom_ea's own searches write to `docs/experiments/20261029-foundational_pose_review/rows/ea_live_sv25/`.

Smoke test 12:22-12:26 on the roll (population 8, budget 60 and 90; rows in `rows_smoke/ea_grasp/`, not results): run,
resume by replay (the replayed children matched their rows), validate, the CPU check and the report.

Outcome (queue 12:27-16:13 on 11-04, no failed step; from 13:02 to 13:34 resguard refused its steps on swap 15.1 GB >
15 GB, the desktop's pages). Rollouts until a search's incumbent scores within 0.05 of the grid's best at seeds 0-2,
and each search's last incumbent at seeds 20-24, three searches per EA:

| task | grid's best, seeds 0-2 (its layout at 20-24) | rollouts to within 0.05: layout EA | layout-and-grasp EA | incumbents at 20-24: layout EA | layout-and-grasp EA |
|---|---|---|---|---|---|
| twist | 1.00 (1.00) | 5,472, 1,728, 3,216 | 1,479, 1,410, 747 | 1.00, 1.00, 1.00 | 1.00, 1.00, 1.00 |
| roll | 0.65 (0.66) | 5,184, 5,424, 3,648 | none, none, none | 0.65, 0.33, 0.65 | 0.58, 0.60, 0.58 |
| pin-push | 0.74 (0.80) | 768, 2,064, 576 | 621, 1,167, 237 | 0.78, 0.68, 0.69 | 0.75, 0.81, 0.73 |
| press | 0.82 (0.86) | 1,008, 3,168, 288 | 2,124, 1,575, none | 0.81, 0.61, 0.61 | 0.78, 0.78, 0.75 |

The medians at seeds 20-24 differ by +0.00, -0.07, +0.06 and +0.17 (layout-and-grasp EA less layout EA). The
layout-and-grasp EA's incumbents lie where each layout's foundational pose scores well, because its generation 0 scores
each layout by that pose and its grasp steps stay near it: Spearman's rho between the foundational pose's score and the
best of 16 over the grid is 0.58, 0.70, 0.68 and 0.49. On the roll the foundational pose peaks at (80, 60), 0.56, and
scores 0.11 at the grid's best layout (85, 170), whose best grasp scores 0.65; on the press it scores 0 at (70, 150),
the grid's best (0.82). Every layout-and-grasp incumbent kept its search score within 0.011 at seeds 20-24; the layout
EA's lost up to 0.28 (roll, (90, 130)) and 0.15 (press, (85, 65) in two searches). On the CPU plant with every contact
at seed 20, 11 of the 12 layout-and-grasp incumbents and 7 of the 12 layout EA incumbents score within 0.05 of the GPU
at that seed; the twist's (85, 70) layout-and-grasp incumbent drops the sphere there (GPU 1.00), and the layout EA's
twist incumbent at (105, 105) scores 0.37 (GPU 1.00).
Three layout EA searches stopped at hom_ea's cap of 40 generations, at 4,704 to 5,328 rollouts. Compute: the
layout-and-grasp EA ran 72,561 GPU rollouts in 278 batches of up to 483 worlds, 1.59 h of batch time in 2.05 h of queue
steps (61, 79 and 102 ms per rollout of run, batch and step time); the layout EA 70,944 in 110 batches, 1.03 h in 1.10 h
(41, 52 and 56 ms). The CPU runner took a median 2.5 s a rollout on one core in the CPU check: one GPU equals 40 cores on
the layout-and-grasp EA's rollout loop and 24 over its steps, and 60 and 44 on the layout EA's.
