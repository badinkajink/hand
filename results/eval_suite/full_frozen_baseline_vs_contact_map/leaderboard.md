# Eval suite leaderboard

All scores are re-evaluated under the **baseline objective** for apples-to-apples comparison. Higher is better.

## Oracle score per (benchmark, method)

| Benchmark | baseline | contact_map |
|---|---:|---:|
| `cube` | 6.70 ± 0.17 | 6.98 ± 0.19 |
| `power_drill` | 7.37 ± 0.62 | 6.20 ± 0.48 |
| `power_drill_short_proximal` | 7.89 ± 0.47 | 7.84 ± 0.42 |
| `prism` | 2.31 ± 1.10 | 5.57 ± 0.08 |
| `screwdriver_medium_90vertical` | 6.20 ± 0.52 | 6.27 ± 0.57 |
| `screwdriver_medium_flat` | 5.52 ± 0.22 | 5.85 ± 0.31 |
| `screwdriver_medium_vertical` | 5.83 ± 0.17 | 5.72 ± 0.01 |
| `screwdriver_small_flat` | -0.06 ± 0.00 | -0.07 ± 0.01 |

## Δ vs baseline per benchmark

| Benchmark | Δ contact_map |
|---|---:|
| `cube` | +0.28 |
| `power_drill` | -1.17 |
| `power_drill_short_proximal` | -0.05 |
| `prism` | +3.26 |
| `screwdriver_medium_90vertical` | +0.07 |
| `screwdriver_medium_flat` | +0.33 |
| `screwdriver_medium_vertical` | -0.11 |
| `screwdriver_small_flat` | -0.01 |

### Mean Δ across benchmarks

- **contact_map**: mean Δ = +0.32, median Δ = +0.03, wins 4/8 benchmarks
