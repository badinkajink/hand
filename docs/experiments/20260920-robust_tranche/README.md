# Robust continuation tranche (2026-09-20)

Driver: `nohup setsid python3 scripts/hands_tranche_queue.py --queue docs/experiments/20260920-robust_tranche/queue.json >> logs/20260920-robust_tranche.log 2>&1 &`
(started 09:23 MDT; 11 jobs, ~38 min each). Same per-job outputs as `20260919-hands_tranche/` plus
`<id>_eval_j3dr.json` (64 rollouts at ±3 mm / ±10° spawn jitter with friction DR).

## What changed against the 2026-09-19 tranche

| item | 2026-09-19 | here | why |
|---|---|---|---|
| tool spawn | one nominal pose | ±3 mm x/y, ±10° yaw | the 09-19 policies collapse under that jitter (D7 0/64, D6 clip s1 8/64, D5 26/64, D1 20/64) |
| tool | 25 mm cylinder | + screw-tip mesh (+1 g) | the chain's and the real tool have it; `make_tipmesh_morphology_run.py` |
| floor | mjlab plane (μ 1, solref 0.02) | the scene's floor (μ 1.8, solref 0.006), `--scene-floor` | the tool lies on it through closure and lift |
| tool contact | MuJoCo defaults (solref 0.02, solimp 0.9–0.95) | the scene's (0.006, 0.97–0.995) | `env_build.make_object_spec_from_frozen` had dropped them since the first mjlab run |
| warm start | D6 60 M checkpoint | each hand/arm's own 09-19 checkpoint | 20 M steps of continuation on the matched plant |

## Decision log
- 09:23 launched; order: D6 clipsep, D3, D5, D1, D6 (clip s2), D4, D7, D2 clip, then D3/D5/D1 clipsep.
