# Architecture Overview

## Design Principles

1. Separate morphology and control optimization concerns.
2. Keep the working Phase 1 pipeline explicit and reproducible.
3. Treat backend adapters as optional runtime lanes, not the main abstraction.
4. Keep model definitions inspectable and generated from the same morphology data.

## Core Modules

- `morphohand.sampling`: morphology perturbation, foundational pose loading, feasibility gating,
  scene patching, and CSV / plot helpers.
- `morphohand.optimization.phase1_common`: shared Phase 1 evaluator, objective, and metrics.
- `morphohand.optimization.phase1_strategy_cem`: CEM lane for foundational poses and adaptation.
- `morphohand.optimization.phase1_strategy_mjx_autodiff`: MJX autodiff lane.
- `morphohand.optimization.phase1_strategy_diffmjx`: DiffMJX-style MVP lane.
- `morphohand.tools.morphology_xml`: MJCF generation and morphology parameter handling.
- `morphohand.backends`: backend protocol plus adapter shells.
- `assets/mjcf`: canonical hand, scene, and task-specific evaluation XML assets.
- `scripts/phase1_pollard_multiscene.py`: cube + prism morphology sampling pipeline.
- `scripts/run6_combined_multitask.py`: screwdriver multi-keyframe combined evaluation.

## DOF Split

Per finger DOFs:

- Morphology (sampling space): `x`, `y`, `len` — 3 DOF/finger, 9D total
- Control (foundational pose / grasp control): `yaw`, `mcp`, `pip` — 3 DOF/finger, 9D total

This supports object classes where approach angle sensitivity is high, especially elongated
objects and the screwdriver orientations used in run6.

## Current Pipeline

1. Foundational pose search per keyframe using `scripts/phase1_optimize_grasp.py`.
2. Morphology sampling around a base morphology using `morphohand.sampling.sample_morphologies`.
3. FP adaptation per morphology using the chosen strategy lane.
4. Feasibility gating and ranking with the shared Phase 1 evaluator.
5. Post-hoc analysis and visualization using `scripts/run6_analysis.py`.

MuJoCo is the default runtime backend for this loop. Warp-backed evaluation paths remain
available as throughput experiments, but they are not required for the current workflow.

## Current Status

- Phase 1 sampling and evaluation are implemented.
- Outer-loop MAP-Elites remains a skeleton.
- Backend adapter modules are present, but the default documented path is still MuJoCo.

## Trajectory-Based Optimization (April 30, 2026)

Recent work added **time-varying finger-control trajectories** to Phase 1 evaluation:

- `Phase1GraspEvaluator.evaluate_trajectory()` — Evaluates piecewise-linear finger control trajectories (N keypoints) over grasp→lift→pivot→hold phases.
- `Phase1GraspEvaluator.rollout_trajectory()` and `render_rollout_gif_trajectory()` — Generate full rollout data and visualization.
- `optimize_finger_control_trajectory()` in `phase1_strategy_cem` — CEM-based search over trajectory parameter space (flattened N×9 controls).
- CLI flag `--traj-phases` in `phase1_optimize_grasp.py` — Activates trajectory optimization when > 1.

This enables dynamic finger control adaptation during reorientation maneuvers. See [4-30 Experiments](4-30_experiments.md) for debugging history, results, and next steps on contact-count optimization.

## Recent Findings

- 2026-04-13: Foundational pose search on the medium screwdriver scene produced distinct
  best scores for `open_flat`, `open_vertical`, and `open_90vertical`.
- 2026-04-21: Run6 combined multitask sweeps show that feasibility depends strongly on
  screwdriver orientation, with `open_flat` highest and `open_90vertical` lowest among the
  current 1000-sample sweep.
- 2026-04-30: Trajectory-based optimization enables palm reorientation (rz=90°) with full contact persistence through lift and tilt phases. See [4-30 Experiments](4-30_experiments.md) for the current numbers.
- See [Phase 1 results summary](phase1_results.md) for earlier milestone numbers.
