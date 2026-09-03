# MorphoHand Docs

> **Current state (2026-09-03).** The sections below describe the Phase 1 *simulation*
> pipeline and are still accurate for that. They do not cover the hardware program. For where
> the project actually is:
>
> - **[Gates on a deployable residual](experiments/20260903-sim2real-gates/page.html)** —
>   current state, what blocks a sim2real residual RL policy, and the full index of hardware
>   report pages. **Start here.**
> - [Servo system identification](experiments/20260902-servo-sysid/servo-sysid.html) — the
>   measured plant and the ranking negative result.
> - `docs/rl/reorientation.md` — chronological engineering log, RL source of truth.
> - `CLAUDE.md` — durable conventions and the gotcha list.

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
