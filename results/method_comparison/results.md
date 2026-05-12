# Method comparison: power_drill_short_proximal / open_flat
Matched CEM budget (12 iters x 24 pop = 288 evals/seed); settle/lift/hold = 120/80/40 sim steps; backend=MuJoCo native.

## Summary (sorted by oracle score, baseline objective)
Note: `in-method` scores are NOT directly comparable across methods because each method optimizes a different objective. The `oracle` column re-scores the best grasp from each run under the baseline objective for an apples-to-apples comparison.

| Method | seeds | oracle (baseline obj) mean ± std | min | max | in-method score | wall (s/seed) | cube_lift | tip_contacts | fc_q1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 3 | 8.326 ± 0.503 | 7.816 | 9.011 | 8.326 | 2.28 | 0.0499 | 10.3 | nan |
| `contact_map` | 3 | 8.152 ± 0.608 | 7.705 | 9.011 | 6.620 | 2.27 | 0.0492 | 10.0 | nan |
| `force_closure` | 3 | 6.508 ± 0.234 | 6.249 | 6.815 | 5.541 | 2.13 | 0.0506 | 5.7 | 0.009 |
| `synergy_k3` | 3 | 3.850 ± 1.525 | 2.267 | 5.910 | 3.850 | 1.67 | 0.0522 | 3.3 | nan |
| `synergy_k4` | 3 | 3.533 ± 2.222 | 1.680 | 6.657 | 3.533 | 1.61 | 0.0538 | 3.3 | nan |
| `combined_k4+ct+fc` | 3 | 0.812 ± 0.662 | 0.335 | 1.749 | -1.301 | 1.36 | 0.0461 | 0.3 | inf |

## Diagnostics means
| Method | all_finger_contact_persistence | contact_target_mean_distance | contact_target_reward | cube_lift | cube_tip_contacts | cube_z_drop_from_peak | fc_fingers_engaged | fc_normal_balance | fc_q1_distance | fc_wrench_spread | mean_tip_distance | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 1.000 | 0.000 | 0.000 | 0.050 | 10.333 | 0.002 | — | — | — | — | 0.000 | 8.326 |
| `contact_map` | 1.000 | 0.077 | 0.000 | 0.049 | 10.000 | 0.002 | — | — | — | — | 0.000 | 6.620 |
| `force_closure` | 1.000 | 0.000 | 0.000 | 0.051 | 5.667 | 0.003 | 3.000 | 0.717 | 0.009 | 0.046 | 0.000 | 5.541 |
| `synergy_k3` | 0.425 | 0.000 | 0.000 | 0.052 | 3.333 | 0.004 | — | — | — | — | 0.000 | 3.850 |
| `synergy_k4` | 0.333 | 0.000 | 0.000 | 0.054 | 3.333 | 0.004 | — | — | — | — | 0.000 | 3.533 |
| `combined_k4+ct+fc` | 0.000 | 0.086 | 0.000 | 0.046 | 0.333 | 0.003 | 0.333 | 0.333 | inf | 0.335 | 0.000 | -1.301 |
