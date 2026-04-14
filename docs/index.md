# MorphoHand Docs

This documentation tracks implementation details for the simulation-first morphology optimization stack.

## Scope

- Multi-backend simulator architecture (`mjx`, `diffmjx-lite`, `mjwarp`, `comfree-warp`)
- Bi-level optimization pipeline
- Baseline MJCF assets and reproducible setup with `uv`


## Start Here

- Architecture overview: `architecture/overview.md`
- Optimization pipeline: `architecture/optimization_pipeline.md`
- Phase 1 inner-loop implementation: `architecture/phase1_inner_loop.md`
- GPU MJX usage guide: `architecture/gpu_mjx_usage.md`
- Backend strategy: `simulators/backends.md`
- DiffMJX implementation plan: `simulators/diffmjx_plan.md`
- [4-14 Experiments: Warp Throughput vs Metric Fidelity](architecture/4-14_experiments.md)
- [Warp-MuJoCo sync bottleneck and mitigation](architecture/warp_mujoco_bottleneck.md)
- Milestones: `roadmap/phases.md`
