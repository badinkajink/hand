# Run 7 Analysis and Summary

Date: 2026-04-23

## Scope
This document summarizes the completed Run 7 strict comparison between:
- Sphere tips: [results/phase1/run7_strict_spheres](results/phase1/run7_strict_spheres)
- Capsule tips: [results/phase1/run7_strict_capsules](results/phase1/run7_strict_capsules)

Both runs used the same keyframes and strict vertical constraints:
- open_flat
- open_vertical
- open_90vertical

## Completion Status
Both runs completed successfully with expected output markers:
- [results/phase1/run7_strict_spheres/all_candidates_multitask.csv](results/phase1/run7_strict_spheres/all_candidates_multitask.csv)
- [results/phase1/run7_strict_spheres/summary.json](results/phase1/run7_strict_spheres/summary.json)
- [results/phase1/run7_strict_spheres/top5_candidates.csv](results/phase1/run7_strict_spheres/top5_candidates.csv)
- [results/phase1/run7_strict_capsules/all_candidates_multitask.csv](results/phase1/run7_strict_capsules/all_candidates_multitask.csv)
- [results/phase1/run7_strict_capsules/summary.json](results/phase1/run7_strict_capsules/summary.json)
- [results/phase1/run7_strict_capsules/top5_candidates.csv](results/phase1/run7_strict_capsules/top5_candidates.csv)

Generated media:
- Sphere GIF count: 24
- Capsule GIF count: 24

## Headline Result
Under this strict setting, sphere tips outperformed capsule tips on feasibility and average score.

## Core Metrics
| Metric | Spheres | Capsules | Delta (Capsules - Spheres) |
|---|---:|---:|---:|
| Samples | 1200 | 1200 | 0 |
| All-task feasible count | 191 | 43 | -148 |
| All-task feasible rate | 15.92% | 3.58% | -12.34 pp |
| Mean feasible tasks per sample | 1.765 | 1.285 | -0.480 |
| Mean aggregate score | 2.8313 | 2.0910 | -0.7403 |
| P95 aggregate score | 5.4087 | 3.8398 | -1.5689 |
| Best aggregate score | 72.1350 | 135.9551 | +63.8201 |
| Best min-task score (robust best) | 6.0733 | 5.0978 | -0.9755 |

Interpretation:
- Sphere tips gave much better consistency across all keyframes.
- Capsule tips produced higher outlier peaks, but weaker robust performance.
- The robust best metric (max of aggregate_min_score) favors spheres.

## Per-Keyframe Feasibility
| Keyframe | Spheres | Capsules | Delta (Capsules - Spheres) |
|---|---:|---:|---:|
| open_flat | 90.92% | 88.58% | -2.33 pp |
| open_vertical | 55.83% | 27.00% | -28.83 pp |
| open_90vertical | 29.75% | 12.92% | -16.83 pp |

Interpretation:
- Flat scene is similar between tip geometries.
- The main gap appears in vertical scenes, where spheres are substantially more feasible.

## Vertical-Only Summary
| Metric | Spheres | Capsules | Delta (Capsules - Spheres) |
|---|---:|---:|---:|
| Mean vertical score (open_vertical + open_90vertical)/2 over all samples | 1.7996 | 1.0988 | -0.7008 |
| Both vertical tasks feasible count | 201 | 46 | -155 |
| Both vertical tasks feasible rate | 16.75% | 3.83% | -12.92 pp |

Conclusion for vertical anti-twist objective:
- Sphere tips are clearly stronger in this run configuration.

## Best Candidates
Top candidate from each run:

- Spheres (candidate 440):
  - aggregate_score_mean: 6.3159
  - aggregate_min_score: 6.0733
  - scene: [results/phase1/run7_strict_spheres/generated_mjcf/scene_multi_tn0d0001_p0d0251_p0d0000_ip0d0005_n0d0033_p0d0000_mp0d0029_p0d0061_p0d0000.xml](results/phase1/run7_strict_spheres/generated_mjcf/scene_multi_tn0d0001_p0d0251_p0d0000_ip0d0005_n0d0033_p0d0000_mp0d0029_p0d0061_p0d0000.xml)

- Capsules (candidate 952):
  - aggregate_score_mean: 5.4277
  - aggregate_min_score: 5.0978
  - scene: [results/phase1/run7_strict_capsules/generated_mjcf/scene_multi_tp0d0051_p0d0257_p0d0000_ip0d0124_n0d0190_p0d0103_mp0d0205_p0d0037_p0d0015.xml](results/phase1/run7_strict_capsules/generated_mjcf/scene_multi_tp0d0051_p0d0257_p0d0000_ip0d0124_n0d0190_p0d0103_mp0d0205_p0d0037_p0d0015.xml)

## Artifact Index
Primary run summaries:
- [results/phase1/run7_strict_spheres/summary.json](results/phase1/run7_strict_spheres/summary.json)
- [results/phase1/run7_strict_capsules/summary.json](results/phase1/run7_strict_capsules/summary.json)

Per-run analysis outputs:
- [results/phase1/run7_strict_spheres/analysis/run6_analysis_summary.md](results/phase1/run7_strict_spheres/analysis/run6_analysis_summary.md)
- [results/phase1/run7_strict_capsules/analysis/run6_analysis_summary.md](results/phase1/run7_strict_capsules/analysis/run6_analysis_summary.md)

Top-level analysis folders:
- [results/phase1/run7_strict_spheres/analysis](results/phase1/run7_strict_spheres/analysis)
- [results/phase1/run7_strict_capsules/analysis](results/phase1/run7_strict_capsules/analysis)

## Recommendation
If the immediate target is robust multi-keyframe performance with strict vertical anti-drift constraints, keep sphere tips as the baseline.

Capsule tips may still be useful to explore, but likely need retuning before direct competition with spheres, for example:
- Dedicated capsule-specific objective weights
- Capsule-specific feasibility thresholds
- Longer adaptation budget or different perturbation schedule
