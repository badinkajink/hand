# Eval suite leaderboard

All scores are re-evaluated under the **baseline objective** for apples-to-apples comparison. Higher is better.

## Oracle score per (benchmark, method)

| Benchmark | baseline | contact_map |
|---|---:|---:|
| `cube` | 6.64 ± 0.13 | 6.82 ± 0.31 |
| `power_drill` | 7.39 ± 0.16 | 7.06 ± 0.17 |
| `power_drill_short_proximal` | 8.46 ± 0.29 | 8.76 ± 0.36 |
| `prism` | 1.00 ± 0.17 | 4.16 ± 1.96 |
| `screwdriver_medium_90vertical` | 6.24 ± 0.41 | 6.21 ± 0.11 |
| `screwdriver_medium_flat` | 5.67 ± 0.00 | 5.39 ± 0.32 |
| `screwdriver_medium_vertical` | 5.70 ± 0.00 | 5.71 ± 0.00 |
| `screwdriver_small_flat` | -0.06 ± 0.01 | -0.08 ± 0.01 |

## Δ vs baseline per benchmark

| Benchmark | Δ contact_map |
|---|---:|
| `cube` | +0.18 |
| `power_drill` | -0.33 |
| `power_drill_short_proximal` | +0.30 |
| `prism` | +3.16 |
| `screwdriver_medium_90vertical` | -0.04 |
| `screwdriver_medium_flat` | -0.28 |
| `screwdriver_medium_vertical` | +0.01 |
| `screwdriver_small_flat` | -0.01 |

### Mean Δ across benchmarks

- **contact_map**: mean Δ = +0.37, median Δ = -0.00, wins 4/8 benchmarks
