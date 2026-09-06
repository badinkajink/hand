# Data: deploy-screen reproduction, bench trial table, closure re-fit

Results live in the page, not here: `../20260906-screen_vs_chain/20260906-screen_vs_chain.html`.

| file | produced by |
|---|---|
| `repro_band_D1_D2_D7.json` | `real_v1_plan_band.py --deploy-dir docs/experiments/20260829-real_v1_deploy/deploy --plans sv1_w6689_b060,sv1_w2360_b075,rv05_manual_b85 --budgets 0.60,0.85,0.05 --reps 4` |
| `bench_trials_D1_D8.json` | `real_v1_bench_trials.py` (reads `../20260902-cb1-log-archive/logs/`) |
| `grasp_refit.json` | `real_v1_grasp_refit.py` |
| `carry_rv05_manual_stored_b*_k*.json` | `rv05_carry_band.sh` |
| `20260906-films/` | `real_v1_chain_films.py --only sv1_w6689_b060,rv05_manual_b85` |

`carry_*` and `20260906-rv05_k025*` are the rv05 axis-height sweep on `rv05_manual_stored`.
Its cos 0.971 cell sits at 75-81 deg of turn, above every measured hardware drop angle, so it
is an overshoot cell and is not a result. See the page.
