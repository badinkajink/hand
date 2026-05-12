# MorphoHand Docs

This documentation tracks the current Phase 1 composition of MorphoHand: morphology
sampling, foundational pose search, grasp evaluation, and analysis around the screwdriver
and prism benchmarks.

## Scope

- Phase 1 evaluation pipeline and current scripts
- Morphology generation and keyframe handling
- Grasp specification methods beyond raw joint-space CEM
- Cross-object eval suite with GIF artifacts per benchmark
- Backend strategy for MuJoCo and optional Warp lanes
- Current results summaries and roadmap notes

## Start Here

- [Architecture overview](architecture/overview.md)
- [Optimization pipeline](architecture/optimization_pipeline.md)
- [Phase 1 inner-loop implementation](architecture/phase1_inner_loop.md)
- [Phase 1 results summary](architecture/phase1_results.md)
- [Backend strategy](simulators/backends.md)
- [GPU MJX usage guide](architecture/gpu_mjx_usage.md)
- [DiffMJX implementation plan](simulators/diffmjx_plan.md)
- [4-14 Experiments: Warp Throughput vs Metric Fidelity](architecture/4-14_experiments.md)
- [Warp-MuJoCo sync bottleneck and mitigation](architecture/warp_mujoco_bottleneck.md)
- [Milestones](roadmap/phases.md)

## Grasp specification methods

Beyond the baseline raw-9D CEM. Each method is opt-in via a config knob
that defaults to off.

- [Grasp methods overview](grasp_methods.md) — when to use which
- [Eigengrasp / synergy subspace CEM](eigengrasp.md)
- [Contact-target patches](contact_targets.md)
- [Force-closure energy](force_closure.md)
- [Empirical comparison on the drill scene](method_comparison.md)

## Evaluation & protocol

- [Eval suite](eval_suite.md) — benchmark × method × seed harness with GIFs
- [Frozen-scene protocol](frozen_scene_protocol.md) — **mandatory** for any
  grasp eval; base scenes have morph DOFs that drift
