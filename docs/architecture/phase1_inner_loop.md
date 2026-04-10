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

## Optimizer

Phase 1 uses CEM over the 9-dimensional control vector:

1. sample control candidates,
2. rollout each candidate,
3. keep elite set,
4. update mean and variance,
5. repeat for N iterations.

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
  --scene-xml assets/mjcf/generated/scene_tp0d0000p0d0200p0d0000_ip0d0100n0d0123p0d0000_mp0d0100p0d0153p0d0000.xml
```
