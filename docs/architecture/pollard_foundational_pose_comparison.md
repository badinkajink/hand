# Differential MCP Actuation and Pollard FP Method: Implications for MorphoHand

Date: 2026-04-13

## 1) Validating your differential-mechanism intuition

Short answer: your intuition is directionally strong for your stated constraints, with one correction.

### Correction to the independence point

A two-motor differential does not necessarily remove independent control in the kinematic sense. If the mechanism is approximately linear,

- MCP flex/extend is proportional to sum of motor motions.
- MCP ab/adduction is proportional to difference of motor motions.

Then commanding the two motors can still span both joint coordinates. In that sense, the two DOFs remain controllable.

### Where your concern is still valid

In practice, differential couplings often introduce useful but real constraints:

- Coupled actuator limits: feasible motion is often a tilted/diamond region in task space, not a full rectangle.
- Torque-speed tradeoffs between the two virtual axes (sum and difference channels share hardware limits).
- Sensitivity to friction, backlash, and tendon compliance that can make one axis disturb the other.
- Control complexity and calibration burden without clear benefit when packaging is not constrained.

So your core argument stands for this project: if packaging, cable routing, and distal inertia are not first-order constraints, independent actuators per DOF are usually cleaner for system identification, optimization, and interpretation.

### Why papers still use differentials

Common reasons include:

- Packaging and remote actuation simplicity.
- Mechanical robustness and force amplification in the dominant grasp direction.
- Built-in synergy priors that regularize behavior in limited-control applications.
- Anthropomorphic inspiration (where perfect MCP decoupling is not physiologically realistic anyway).

For your research goals (quality diversity, object-subset specialization, and reconfigurability), these benefits are usually secondary to controllability and interpretability.

## 2) What Pollard et al. actually do in the Foundational Pose paper

Paper: Wang, Oh, Pollard. The Foundational Pose as a Selection Mechanism for the Design of Tool-Wielding Multi-Finger Robotic Hands.

### Core idea

They define a small set of foundational poses (FPs) for tool use (carve, poke, press style mechanisms). A design is promising if it can reach all required FPs while satisfying contact and static-equilibrium constraints.

### Their optimization/evaluation flow

It is best described as feasibility-guided sampling plus post-hoc multi-objective evaluation:

1. Parameterize hand design by a low-dimensional vector d (their template uses 6D in full experiments).
2. Start from one known-feasible design.
3. Sample new designs using RRT in standardized design space (not gradient descent).
4. For each sample, solve a nonlinear optimization to move the hand/tool into all FPs under collision and statics constraints.
5. Accept as candidate only if it is feasible and sufficiently different from existing candidates.
6. For each accepted candidate, plan tool-wielding paths from each FP using a QP-based contact-sliding minimization procedure.
7. Score candidates with metrics (tool motion range, mean sliding speed, max finger torque), aggregate across FPs, and analyze Pareto fronts/clusters.

### Important nuance

They explicitly note a limitation: the step that enforces FP feasibility does not optimize the final evaluation metrics directly. So feasibility is a strong filter, but not the same as directly maximizing task metrics.

## 3) How this differs from your proposed pipeline

## Your proposed method (documents)

- Bi-level optimization architecture:
  - Outer loop: MAP-Elites over morphology variables.
  - Inner loop: optimize controls for each morphology.
- Control and morphology are explicitly split by role.
- Multi-backend plan: differentiable inner loop with MJX-family, high-throughput outer evaluation with MJWarp/ComFree.
- Canonical palm poses are fixed early to reduce confounds.

## What is actually implemented now

Implemented:

- A real Phase 1 inner-loop evaluator with MuJoCo simulation rollout objective (distance, contact count, lift, stability proxy).
- Two working inner-loop optimizers:
  - CEM.
  - MJX-autodiff surrogate-gradient variant.
- End-to-end artifacts (trace CSV, plots, rollout arrays, report, optional GIF).

Not yet implemented (or placeholder):

- True MAP-Elites outer loop (current outer module is an initialization stub).
- Backend abstraction fully driving experiments (Phase 1 currently uses direct MuJoCo/MJX path in the grasp module).
- DiffMJX-lite and CTR-style additions are still plan-stage.

## 4) Should you simplify toward Pollard style?

Yes, but as a baseline lane, not a replacement.

For your questions, fully collapsing into Pollard style would lose key capabilities. But adding a Pollard-like lane can reduce complexity and de-risk experimentation.

### What to borrow now (high value, low risk)

1. Feasibility-first screening:
   - Add cheap filters before expensive optimization (joint-limit feasibility, collision-free initial closure, simple static support proxies).
2. Candidate diversity during sampling:
   - Keep a minimum-distance acceptance rule in morphology space to avoid near-duplicate evaluations.
3. Pareto and cluster analysis as first-class outputs:
   - Report tradeoff fronts and morphology families, not only single best score.
4. Multi-start from a foundational-pose-like set:
   - Your canonical palm poses already map well to this idea.

### What to keep from your current architecture

1. Bi-level decomposition:
   - Needed to isolate morphology quality from controller optimization effort.
2. MAP-Elites framing:
   - Needed for quality diversity and object-subset specialization.
3. Reconfigurable-platform-aware analysis:
   - Your central contribution depends on identifying families of good designs, not one local optimum.
4. Differentiable inner loop option:
   - Essential if you want to study gradient quality and compare MJX vs DiffMJX-style fixes.

## 5) Recommended practical compromise

Run two tracks in parallel:

- Track A: Pollard-like baseline
  - Sampling-based morphology generation with diversity filter.
  - Short-horizon inner optimization from fixed initial pose(s).
  - Pareto/cluster analysis on resulting metric vectors.

- Track B: Full MorphoHand pipeline
  - MAP-Elites outer loop with inner best response.
  - Backend comparisons and quality-diversity/object-subset experiments.

This gives you a clean ablation: how much insight/performance is gained by your extra structure.

## 6) Specific answer to your original claim

Your claim is mostly correct for your project context:

- If you do not need packaging-driven coupling, independent MCP flexion and ab/adduction actuation is usually preferable.
- The only correction is that a two-motor differential can still be independently controllable in principle via sum/difference coordinates.
- The practical question is not binary controllability, but whether coupling-induced constraints and complexity are worth it. For your stated goals, likely no.
