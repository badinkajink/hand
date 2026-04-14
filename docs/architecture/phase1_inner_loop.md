# Phase 1 Inner Loop

Phase 1 implements grasp synthesis for fixed morphology and fixed palm approach.

## Scope

- Optimize only finger controls: `(yaw, mcp, pip)` for thumb/index/middle.
- Keep palm pose controlled by scene pose actuators from keyframe.
- Use one rigid morphology scene at a time.

## Objective

The objective combines:

- fingertip-to-cube distance minimization,
- cube-tip contact count encouragement,
- lift reward by increasing palm `z` target after closing,
- cube velocity penalty for unstable outcomes.

This is a practical proxy for force-closure style grasp quality in the current setup.

## Optimizers

Phase 1 now supports three strategy lanes over the 9-dimensional finger control vector:

1. `cem`: derivative-free cross-entropy search.
2. `mjx-autodiff`: MJX smooth surrogate optimized with gradient ascent.
3. `diffmjx-mvp`: differentiable MVP lane with contact-from-distance proxies and periodic full MuJoCo evaluation.

Implementation modules:

- Evaluator and shared config: `src/morphohand/optimization/phase1_common.py`
- CEM strategy: `src/morphohand/optimization/phase1_strategy_cem.py`
- MJX autodiff strategy: `src/morphohand/optimization/phase1_strategy_mjx_autodiff.py`
- DiffMJX MVP strategy: `src/morphohand/optimization/phase1_strategy_diffmjx.py`
- Backward-compatible re-export shim: `src/morphohand/optimization/phase1_grasp.py`

For CEM specifically, the optimization loop remains:

1. sample control candidates,
2. rollout each candidate,
3. keep elite set,
4. update mean and variance,
5. repeat for N iterations.

## Backend speed controls

Phase 1 runners support explicit backend/runtime controls for large sweeps:

- `--backend {mujoco,mjwarp,comfree-warp}`
- `--speed-mode {accurate,balanced,aggressive}`
- `--metric-collection-mode {sampled,terminal}`
- `--backend-sync-interval N`
- `--metric-sample-interval N`

`accurate` corresponds to frequent synchronization/metric sampling (`N=1`).
`balanced` and `aggressive` reduce synchronization frequency to improve throughput.

For fixed foundational controls in current open-loop Phase 1 runs, `mjwarp` with moderate intervals preserved core metrics while reducing wall time. See `4-14_experiments.md` for measurements.

## Artifacts

The runner emits:

- optimization trace CSV,
- objective and metric plots,
- best rollout trajectories (`npz`),
- animated GIF of best rollout,
- markdown run report.

## Command

```bash
uv run python scripts/phase1_optimize_grasp.py \
  --scene-xml assets/mjcf/generated/scene_tp0d0000p0d0200p0d0000_ip0d0100n0d0123p0d0000_mp0d0100p0d0153p0d0000.xml \
  --optimizer diffmjx-mvp \
  --iterations 12
```
