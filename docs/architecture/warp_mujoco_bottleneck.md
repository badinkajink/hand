# Warp-MuJoCo Interaction and Bottleneck Breakdown

Date: 2026-04-14

## Why this exists

Phase 1 CEM currently uses MuJoCo-structured metrics (contacts, kinematics, persistence, drift/drop). When using `mjwarp` / `comfree-warp`, physics can run on GPU quickly, but metric extraction and reporting still require some state in MuJoCo host memory.

This document explains the bottleneck, speed modes, and fidelity tradeoffs.

## Dataflow before speed modes

Per simulation step in candidate evaluation:

1. Push control into warp backend data.
2. Step physics on GPU (`mjwarp.step` / `comfree_warp.step`).
3. Synchronize and copy full backend state to MuJoCo (`get_data_into`).
4. Compute metrics from MuJoCo fields.

This is expensive for CEM because each candidate rollout has hundreds of steps and CEM evaluates many candidates.

## Core bottleneck

Main cost is repeated device-to-host synchronization and large structured copies (`get_data_into`) at high frequency.

Even if kernels are cached and fast, this repeated transfer dominates runtime in small-world CEM loops.

## Current speed architecture

`Phase1GraspEvaluator` now supports:

- `backend_sync_interval`: sync warp -> MuJoCo every N steps.
- `metric_sample_interval`: sample dynamic metrics every N steps.
- `speed_mode`: preset that controls both intervals and metric mode.
- `metric_collection_mode`: dynamic metric collection strategy.

### Speed presets

- `accurate`
  - sync/sample every step
  - `metric_collection_mode=sampled`
- `balanced`
  - sync/sample at coarser stride (>=4)
  - `metric_collection_mode=sampled`
- `aggressive`
  - large stride (>=16)
  - `metric_collection_mode=terminal`

## Metric collection modes

### `sampled`

- Dynamics are stepped continuously in warp.
- State is sampled at sync points.
- Dynamic terms (contact persistence, min-finger persistence, etc.) are approximated by sampled values weighted by chunk length.

Pros:
- Keeps objective structure close to original.
- Large speedup versus per-step sync.

Cons:
- Misses very brief transients between sampling points.

### `terminal`

- Dynamics run in warp through entire lift/hold phase.
- One final sync to MuJoCo for dynamic terms.
- Dynamic persistence-like terms are terminal approximations.

Pros:
- Maximum throughput.
- Minimal sync overhead.

Cons:
- Loses temporal detail by design.
- Not appropriate as final truth metric for ranking/reporting.

## Recommended workflow

1. Search phase:
   - Use `--speed-mode balanced` or `--speed-mode aggressive`.
2. Verification phase:
   - Re-evaluate top candidates with `--speed-mode accurate`.
3. Rendering:
   - Replay best control in MuJoCo for GIF/report only.

This separates optimization throughput from final reporting fidelity.

## Why we still touch MuJoCo at all

Current metric code and rendering/report stack are MuJoCo-based. Full elimination of MuJoCo from optimization requires implementing all score terms directly on backend-native arrays and returning only compact summary metrics.

That is feasible, but is a larger engineering step than interval-based speed modes.

## Practical note on N sensitivity

Empirical short-run tests in this workspace showed substantial gains from increasing N from 1 to 4/8 with near-identical early optimization outcomes. Diminishing returns start after moderate N, while metric fidelity drops as N becomes very large.

Use moderate N for search, then accurate recheck for final selection.
