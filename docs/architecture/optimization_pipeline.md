# Optimization Pipeline

## Bi-Level Structure

- Outer loop (`MAP-Elites`): explores morphology vectors.
- Inner loop (gradient-based): optimizes grasp controls for each morphology.

## Parameterization

For three fingers:

- Morphology vector $\theta \in \mathbb{R}^9$: $(x_i, y_i, l_i)_{i=1}^3$
- Control vector $q \in \mathbb{R}^9$: $(\psi_i, q^{mcp}_i, q^{pip}_i)_{i=1}^3$

## Inner Objective

Maximize a grasp quality proxy $\epsilon(\theta, q)$ over $q$ under joint and contact constraints.

## Outer Objective

For each $\theta$, evaluate $\max_q \epsilon(\theta, q)$ and place it in the archive by behavioral descriptor.

## Canonical Palm Poses

`scene.xml` adds 6 pose joints for palm placement but these are not optimized in early experiments.

Recommended early protocol:

1. Define 3-5 canonical palm poses for each object class.
2. Keep those fixed per evaluation run.
3. Optimize only morphology and finger joint controls.
