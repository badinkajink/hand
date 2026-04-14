# Optimization Pipeline

## Bi-Level Structure

- Outer loop: explores morphology vectors (currently Pollard-style sampling, planned MAP-Elites).
- Inner loop: optimizes grasp controls for each morphology (CEM-based).

## Parameterization

For three fingers:

- Morphology vector $\theta \in \mathbb{R}^9$: $(x_i, y_i, l_i)_{i=1}^3$
- Control vector $q \in \mathbb{R}^9$: $(\psi_i, q^{mcp}_i, q^{pip}_i)_{i=1}^3$

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

## Outer Objective

For each $\theta$, evaluate $\max_q \epsilon(\theta, q)$ and rank by feasibility-first, then score.

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

Key insight: warm-starting matters enormously (interval-fp +311 vs interval-open +23). The current FP is always a better starting point than zero.

## Canonical Palm Poses

`scene.xml` adds 6 pose joints for palm placement but these are not optimized in early experiments.

Current protocol:

1. Define fixed palm approach per object class.
2. Keep palm pose fixed per evaluation run.
3. Optimize only morphology and finger joint controls.

## Feasibility Gating

A morphology-scene evaluation is feasible if:

- `mean_tip_distance` <= threshold (cube: 0.012m, prism: 0.02m)
- `cube_tip_contacts` >= threshold (4 for both)
- `min_finger_contact_persistence` >= threshold (cube: 0.55, prism: 0.45)
- `finger_yaw_drift` <= threshold (cube: 0.30, prism: 0.40)
