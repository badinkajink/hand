# Method comparison: power_drill_short_proximal / open_flat
Matched CEM budget (12 iters x 24 pop = 288 evals/seed); settle/lift/hold = 120/80/40 sim steps; backend=MuJoCo native.

## Summary (sorted by oracle score, baseline objective)
Note: `in-method` scores are NOT directly comparable across methods because each method optimizes a different objective. The `oracle` column re-scores the best grasp from each run under the baseline objective for an apples-to-apples comparison.

| Method | seeds | oracle (baseline obj) mean ± std | min | max | in-method score | wall (s/seed) | cube_lift | tip_contacts | fc_q1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `contact_map` | 3 | 8.759 ± 0.363 | 8.462 | 9.270 | 7.213 | 8.19 | 0.0528 | 10.7 | nan |
| `baseline` | 3 | 8.457 ± 0.293 | 8.089 | 8.806 | 8.457 | 8.23 | 0.0524 | 10.3 | nan |
| `synergy_k4` | 3 | 7.091 ± 0.561 | 6.674 | 7.885 | 7.091 | 6.54 | 0.0560 | 6.7 | nan |
| `force_closure` | 3 | 6.411 ± 0.706 | 5.412 | 6.919 | 5.861 | 7.45 | 0.0522 | 5.0 | 0.008 |
| `synergy_k3` | 3 | 3.334 ± 1.841 | 1.771 | 5.918 | 3.334 | 5.47 | 0.0527 | 2.3 | nan |
| `combined_k4+ct+fc` | 3 | 0.837 ± 0.645 | 0.380 | 1.749 | -1.287 | 4.52 | 0.0463 | 0.3 | inf |

## Diagnostics means
| Method | all_finger_contact_persistence | contact_target_mean_distance | contact_target_reward | cube_lift | cube_tip_contacts | cube_z_drop_from_peak | fc_fingers_engaged | fc_normal_balance | fc_q1_distance | fc_wrench_spread | mean_tip_distance | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `contact_map` | 1.000 | 0.077 | 0.000 | 0.053 | 10.667 | 0.003 | — | — | — | — | 0.000 | 7.213 |
| `baseline` | 1.000 | 0.000 | 0.000 | 0.052 | 10.333 | 0.003 | — | — | — | — | 0.000 | 8.457 |
| `synergy_k4` | 1.000 | 0.000 | 0.000 | 0.056 | 6.667 | 0.004 | — | — | — | — | 0.000 | 7.091 |
| `force_closure` | 1.000 | 0.000 | 0.000 | 0.052 | 5.000 | 0.004 | 3.000 | 0.797 | 0.008 | 0.223 | 0.000 | 5.861 |
| `synergy_k3` | 0.333 | 0.000 | 0.000 | 0.053 | 2.333 | 0.004 | — | — | — | — | 0.000 | 3.334 |
| `combined_k4+ct+fc` | 0.000 | 0.087 | 0.000 | 0.046 | 0.333 | 0.003 | 0.333 | 0.333 | inf | 0.335 | 0.000 | -1.287 |
