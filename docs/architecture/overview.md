# Architecture Overview

## Design Principles

1. Separate morphology and control optimization concerns.
2. Keep backend adapters thin and interchangeable.
3. Prefer one codebase with optional backend dependencies.
4. Keep model definitions inspectable and reproducible.

## Core Modules

- `morphohand.backends`: simulator abstraction and backend-specific adapters.
- `morphohand.optimization.inner_loop`: gradient-based grasp synthesis.
- `morphohand.optimization.outer_loop`: MAP-Elites morphology search.
- `assets/mjcf`: canonical hand and scene XML assets.

## Recent Findings

- 2026-04-14 throughput/fidelity comparison for warp backends and sync intervals: `4-14_experiments.md`
- Runtime bottleneck analysis for frequent warp->MuJoCo synchronization: `warp_mujoco_bottleneck.md`

## DOF Split

Per finger DOFs:

- Morphology (outer loop): `x`, `y`, `len`
- Control (inner loop): `yaw`, `mcp`, `pip`

This supports object classes where approach angle sensitivity is high (elongated and flat objects).
