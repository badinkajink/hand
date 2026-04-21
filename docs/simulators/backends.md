# Backend Strategy

## Roles

- `mujoco`: default backend for the current Phase 1 evaluation loop.
- `mjwarp`: optional high-throughput backend for throughput experiments.
- `comfree-warp`: optional Warp-backed backend for analytical-contact experiments.
- `mjx-native`: autodiff lane for the optimizer, not the default runtime backend.
- `diffmjx-lite`: planned MJX variant with smooth collision and CFD gradient path.

## Recommended Usage

- Current docs and runs: use `mujoco` first.
- Optimization experiments: compare `mjx-native` and `diffmjx-lite` only when the gradient
	path is the focus.
- Throughput experiments: try `mjwarp` or `comfree-warp` only if you need GPU-backed batch
	evaluation.

## Failure Handling

When a morphology causes instability on `comfree-warp`:

1. mark evaluation as unstable,
2. rerun same morphology in `mjwarp`,
3. record backend used in experiment logs.

When comparing backends in the docs, treat MuJoCo as the reference implementation for the
current Phase 1 reports.
