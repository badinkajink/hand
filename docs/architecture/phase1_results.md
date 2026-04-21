# Phase 1 Results Summary

This page summarizes the current Phase 1 results that are reflected in the README and the
manuscript.

## Current Composition

Phase 1 now combines three pieces:

1. foundational pose search per keyframe,
2. morphology sampling around a base morphology,
3. FP adaptation and feasibility-gated ranking.

The current medium-screwdriver benchmark is the clearest example of this pipeline because
it has three distinct keyframes and a full run6 multitask evaluation path.

## Foundational Pose Search

The foundational pose searches on `assets/mjcf/scene_screwdriver_medium.xml` produced the
following best scores:

| keyframe | best score | best cube lift | best contacts |
|---|---:|---:|---:|
| `open_flat` | 5.702777 | 0.049613 | 3 |
| `open_vertical` | 6.906849 | 0.050089 | 5 |
| `open_90vertical` | 7.091068 | 0.069566 | 5 |

These are the baselines reused in the screwdriver multitask run.

## Combined Screwdriver Sweep

The combined multitask driver evaluates all three keyframes per sampled morphology and
tracks both per-task feasibility and aggregate score.

Smoke test summary (`run6_combined_smoke2`):

| keyframe | total | feasible | feasible rate | mean score | max score |
|---|---:|---:|---:|---:|---:|
| `open_flat` | 3 | 3 | 1.000 | 4.3617 | 6.1590 |
| `open_vertical` | 3 | 3 | 1.000 | 6.9368 | 6.9850 |
| `open_90vertical` | 3 | 3 | 1.000 | 4.9689 | 6.1868 |

Larger sweep summary (`run6_combined_1000`):

| keyframe | total | feasible | feasible rate | mean score |
|---|---:|---:|---:|---:|
| `open_flat` | 1000 | 994 | 0.994 | 5.4831 |
| `open_vertical` | 1000 | 884 | 0.884 | 3.9379 |
| `open_90vertical` | 1000 | 696 | 0.696 | 2.4647 |

The main trend is that pose sensitivity is real: the same morphology distribution is much
more consistently feasible in `open_flat` than in `open_90vertical`.

## Where to Look

- Run outputs: `results/phase1/`
- Foundational pose reports: `results/phase1/run6_foundational/`
- Combined multitask reports: `results/phase1/run6_combined_1000/`
- Analysis artifacts: `analysis/run6_analysis_summary.md` and the generated plots in each
  run directory