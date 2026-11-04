# One finger layout or several for the HOM tasks: runbook (2026-11-03/04)

Owner, 2026-11-03 20:3x: "how can we accommodate tasks with conflicting morphology objectives? can we linearly
interpolate? if not, is there some rule we can establish (pareto front) for when blending, however we do it, is
optimal, vs when we really should switch morphologies". 22:3x: "we shouldnt consider time, only performance. for desired
performance, we should be able to mathematically formulate the blend vs switch decision, given the current layout".

The page is `20261104-hom_task_conflict.html` beside this file, built by `scripts/hom_task_conflict_page.py`.

## What runs

`scripts/hom_task_conflict.py` (its docstring defines every quantity):

| command | what it does | output |
|---|---|---|
| `scope --plant P [--val]` | score matrices of the lattice grid (S1, seeds 0-2), specialists, J_K, coverage (generalists, K*, alpha*_K, incompatible tasks), the decision at a layout, pair curves and segments; with `--val` the same choices scored at seeds 20-24, chosen at 20-24 alone, and on 3 x 3 smoothed scores | `<date>-conflict_<P>.json` |
| `scope --grid sizes` | the same coverage on the size grid (16 tasks, 8,182 Sobol layouts, work plant, seeds 0-2) | `20261103-conflict_sizes.json` |
| `validate --plant P` | every reached layout's S1 grasp at seeds 20-24 on the GPU (`hom_grasp_refine.evaluate`, one `hom_gpu.batch_child` per batch) | `rows/<P>/val/<task>_d<d>.jsonl` |
| `refine layouts\|run\|validate\|report` | `hom_grasp_refine.py`'s S2 and S3 searches on 10 shared layouts (`shared_layouts()`), its rows root set to `rows_refine/` | `rows_refine/sv25/` |
| `refined --plant sv25` | J_K and coverage on the shared layouts at S1, S2 and S3 | `<date>-conflict_refined_sv25.json` |

Plants: `sv25` (SCS0009, pads 2.5 MPa, the hand), `k21` (STS3032 at kp 2.1, friction 1.0), `k21mu0p4` (the same at
friction 0.4), `work10` (the work plant of 10-30, seeds 0-2 only).

## Queues

Both run one GPU job at a time, each step through `~/.claude/bin/resguard.sh run --mem 5G --cpu 500`, retrying while
resguard refuses (exit 3); a finished step leaves `logs/20261103-hom_task_conflict/Q_<step>.done`, and a stopped step
resumes from its rows.

```bash
nohup setsid bash docs/experiments/20261103-hom_task_conflict/queue.sh >> logs/20261103-hom_task_conflict/queue.log 2>&1 &
nohup setsid bash docs/experiments/20261103-hom_task_conflict/queue_refine.sh >> logs/20261103-hom_task_conflict/queue_refine.log 2>&1 &
```

`queue.sh` validates the three plants (9,300 rollouts each; 2026-11-04 02:20-02:41). `queue_refine.sh` waits for it,
then runs per task the S2 search, the S3 search and the validation of the three picks on the shared layouts, then the
report. Do not edit `scripts/hom_task_conflict.py` while a queue runs: every step and every spawned batch child
imports it.

## Files

- `20261103-conflict_<plant>.json`: the window's scope of 11-03 at seeds 0-2. `20261104-conflict_<plant>.json`: the
  same with `val` (choices, coverage and decisions at seeds 20-24, smoothed variant, the score matrices).
- `rows/<plant>/val/<task>_d<d>.jsonl`: one row per rollout in `hom_gpu.py`'s format (key: call, pick "S1", grasp, v);
  `_timing.jsonl` one line per GPU batch; `_runs.jsonl` one line per run (open-pose and GPU seconds, rollouts, wall).
- `rows_refine/sv25/`: `<task>_layouts.json` (the shared layouts with S1 and its grasp; `roles` gives why each was
  chosen), `grasp/` and `full/` (search rows, `_gens.jsonl`, `_iters.jsonl` per iteration), `val/` (the picks at seeds
  20-24), `<date>-score_levels.json` (the report).
- `media/`: the page's figures and films (`20261104-films.json` the clips' scores).

## Compute tracking

Every GPU batch writes its timing line beside its rows; each validation run and each search iteration writes a JSON
line (`_runs.jsonl`, `_iters.jsonl`); each queue step ends its log with `peak <kB> kB, <s> s`. `scripts/hom_compute.py
ledger` includes `rows/` and `rows_refine/` from 2026-11-04 (`task_conflict_study` in the ledger).

## Outcome (2026-11-04)

Page: `20261104-hom_task_conflict.html` beside this file. Validated at seeds 20-24, with every choice made at seeds 0-2:

| plant | J_1 | J_4 | J_4/J_1 | alpha*_1 | K*(0.9) |
|---|---|---|---|---|---|
| SCS0009, friction 1.0 | 0.75 | 0.82 | 1.10 | 0.84 | 3 |
| STS3032, friction 1.0 | 0.76 | 0.84 | 1.11 | 0.87 | 2 |
| STS3032, friction 0.4 | 0.72 | 0.83 | 1.15 | 0.81 | 2 |

On the SCS0009 the four specialists and the best single layout (70, 110) are the same at seeds 0-2 and 20-24, and the
layouts' order repeats (Spearman's rho 0.89-0.93 per task). The straight segment between two specialists is beaten on
both tasks by up to 0.61; on the 3 x 3 smoothed scores by 0.02-0.11. Every pair of tasks shares a layout at 0.85-0.95
of their bests, every three at 0.84-0.85. On the ten shared layouts, per-task S3 refinement raises J_1 to 0.82 and J_4
to 0.90 (J_4/J_1 1.09), and two layouts suffice at 0.9 where S1 needs three; the pin-push gap persists (0.73 against
0.94). The size grid (16 tasks): alpha*_1 0.52, K* 7 at 0.9 and 11 at 0.95.
