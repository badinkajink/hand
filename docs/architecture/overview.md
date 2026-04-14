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
- `morphohand.optimization.phase1_common`: grasp evaluator, objective function, and metrics.
- `morphohand.optimization.phase1_strategy_cem`: CEM optimizer for foundational poses and FP adaptation.
- `morphohand.tools.morphology_xml`: MJCF generation and morphology parameter handling.
- `assets/mjcf`: canonical hand and scene XML assets.
- `scripts/phase1_pollard_multiscene.py`: main morphology sampling pipeline with FP adaptation.

## DOF Split

Per finger DOFs:

- Morphology (outer loop): `x`, `y`, `len` — 3 DOF/finger, 9D total
- Control (inner loop): `yaw`, `mcp`, `pip` — 3 DOF/finger, 9D total

This supports object classes where approach angle sensitivity is high (elongated and flat objects).

## Backend Decision

CPU MuJoCo is the production backend for Phase 1 evaluation. GPU backends (mjwarp, comfree-warp) were benchmarked but provide no advantage for the current host-driven per-morphology evaluation loop. See `4-14_experiments.md` for details.

## Current Pipeline

```
1. Foundational Pose Search (one-time per scene)
   CEM: 40 iter × 72 pop × 3 seeds → best finger controls per object

2. Morphology Sampling (500 candidates)
   9D perturbation around base → optional distance-sort

3. Per-Morphology Evaluation
   For each morphology × each scene:
   a. Generate scene XML with morphology
   b. Adapt FP (sparse-per-morph recommended: 5 random perturbations)
   c. Also evaluate original FPs, pick best feasible
   d. Run settle → lift → hold simulation (600 steps)
   e. Score: lift + contacts + persistence - drift penalties
   f. Gate: min contacts, min persistence, max drift, max tip distance

4. Ranking & Refinement
   Feasibility filtering → Pareto front → top-k CEM refinement → GIF export
```

## Recent Findings

- 2026-04-10: FP adaptation study (Run 5): cheap per-morphology FP refinement recovers +22% feasibility. See `4-14_experiments.md`.
- 2026-04-14: Warp throughput/fidelity comparison. CPU MuJoCo preferred. See `4-14_experiments.md`.
- 2026-04-13: Pollard sampling progression (Runs 1-3). See `4-13_pollard_sampling.md`.
