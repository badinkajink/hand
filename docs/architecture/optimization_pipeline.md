# Optimization Pipeline

## Bi-Level Structure

- Outer loop: explores morphology vectors via Pollard-style sampling and multi-scene sweeps.
- Inner loop: optimizes grasp controls for each morphology using the Phase 1 evaluator.

The long-term outer-loop plan is still MAP-Elites, but the current implementation is the
sampling-and-adaptation pipeline around Phase 1.

## Parameterization

For three fingers:

- Morphology vector $\theta \in \mathbb{R}^9$: $(x_i, y_i, l_i)_{i=1}^3$
- Control vector $q \in \mathbb{R}^9$: $(\psi_i, q^{mcp}_i, q^{pip}_i)_{i=1}^3$

This matches the current code layout in `src/morphohand/sampling/morphology.py` and
`src/morphohand/optimization/phase1_common.py`.

## Inner Objective

Maximize a grasp quality proxy $\epsilon(\theta, q)$ over $q$:

```
score = 35.0 * lift
      + 2.4 * min_finger_contact_persistence
      + 0.4 * contact_count
      + 0.8 * contact_persistence
      - 2.0 * mean_tip_distance
      - 1.2 * finger_persistence_imbalance
      - 1.0 * finger_yaw_drift
      - 0.5 * finger_flex_drift
      - 6.0 * cube_xy_drift
      - 12.0 * cube_z_drop_from_peak
      - 0.15 * cube_velocity
```

The exact weights are configurable in `scripts/phase1_optimize_grasp.py` and the combined
run6 driver. The important point is the current shape of the objective: lift, contact, and
stability terms dominate, while drift and control-regularization terms suppress degenerate
solutions.

## Outer Objective

For each $\theta$, evaluate the best available grasp control and rank by feasibility-first,
then score.

In the current pipeline, this is done in two places:

- single-scene foundational pose search for each keyframe;
- morphology sweeps that adapt or reuse those foundational poses.

## Foundational Pose Strategy

The foundational pose (FP) is the finger control vector that produces a good grasp. Finding it is the inner-loop optimization problem.

### FP Adaptation (Run 5 finding)

Fixed FPs degrade as morphologies diverge from the base. Tested strategies:

| Strategy | How it works | Cost | Gain |
|----------|-------------|------|------|
| `none` | Use pre-computed FPs unchanged | 0 | baseline |
| `interval-initial-fp` | Re-run CEM every N morphologies, warm-started | 5s per trigger | +311 |
| `interval-open` | Re-run CEM every N morphologies, from scratch | 5s per trigger | +23 |
| `sparse-per-morph` | 5 random perturbations per morphology | 0.1s per morph | **+316** |
| `local-perturbation` | ±delta on each control dim per morphology | 0.35s per morph | +358 |

**Recommended default**: `sparse-per-morph` with 5 samples. Best cost-effectiveness.

Key insight: warm-starting matters enormously (interval-fp +311 vs interval-open +23).
The current FP is usually a better starting point than zero, especially in the screwdriver
multi-keyframe run.

## Canonical Palm Poses

`scene.xml` adds 6 pose joints for palm placement but these are not optimized in early experiments.

Current protocol:

1. Define fixed palm approach per object class.
2. Keep palm pose fixed per evaluation run.
3. Optimize only morphology and finger joint controls.

For the current medium-screwdriver benchmark, the three keyframes are `open_flat`,
`open_vertical`, and `open_90vertical`.

## Feasibility Gating

A morphology-scene evaluation is feasible if:

- `mean_tip_distance` <= threshold (cube: 0.012m, prism: 0.02m)
- `cube_tip_contacts` >= threshold (4 for both)
- `min_finger_contact_persistence` >= threshold (cube: 0.55, prism: 0.45)
- `finger_yaw_drift` <= threshold (cube: 0.30, prism: 0.40)

The run6 screwdriver combined sweep uses a slightly looser shared gate to make the
multi-keyframe aggregation more informative:

- `max_mean_tip_distance = 0.022`
- `min_contacts = 2`

## Current Scripts

- `scripts/phase1_optimize_grasp.py`: foundational pose search and single-scene reports.
- `scripts/phase1_pollard_multiscene.py`: cube + prism morphology sampling sweep.
- `scripts/run6_combined_multitask.py`: screwdriver multi-keyframe multitask evaluation.
- `scripts/run6_analysis.py`: embeddings and feature-metric plots for sampled morphologies.
