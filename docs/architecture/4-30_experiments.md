# April 30, 2026 – Grasp+Lift+Reorientation Experiments

## Overview
This document captures the iterative debugging and refinement of the Phase 1 grasp-optimization pipeline to achieve robust three-finger grasping of a power drill with controlled lift, tilt, and reorientation. The core challenge was ensuring all three fingertips remain in contact throughout a smooth kinematic trajectory while the hand transitions from a flat palm pose to a 90° palm-down orientation (via rz rotation).

## Problem Statement & Observations

### Initial Issues (Runs 8–9)
1. **Fingers sliding on palm** — During rollout, fingers were observed sliding relative to the palm due to live morphology joints remaining active in the scene.
   - Root cause: Live `morph_*` joints (thumb_x, index_x, middle_x) were not baked out; optimizer was evaluating against a non-rigid morphology.
   - Impact: Contact metrics were unstable and fingers could drift across the palm surface.

2. **Wrong tilt direction** — The initial launcher used `pivot_delta_ry = -1.20`, but the intended behavior was to rotate the palm ~90° down (positive rz), not to tilt about the Y-axis.
   - Root cause: Confusion over pivot axis and sign convention. The drill head should face downward after the maneuver.
   - Impact: Grasps appeared to succeed but the roll/tilt direction was incorrect.

3. **Low fingertip contact count** — Despite optimization, only 6 contacts achieved (out of max 9):
   - Thumb: contacted  
   - Index: contacted  
   - Middle: **not contacted** throughout dynamic phases (lift/pivot/hold).
   - Root cause: Objective function weights did not strongly penalize missing contacts during pivot. Finger control was static and did not adapt.

### Solution Strategy
We implemented a **time-varying finger-control trajectory** (4 keypoints) over the grasp, lift, pivot, and hold phases, allowing the optimizer to adjust grip strength and finger angles dynamically rather than holding a fixed posture.

## Implemented Changes

### 1. Frozen Morphology Scene Generation
**File:** `src/morphohand/tools/morphology_xml.py` (existing utilities leveraged)

Added support in `phase1_optimize_grasp.py` to:
- Extract the morphology poses from the `open_flat` keyframe
- Bake those poses into a new rigid scene XML (`frozen_scene.xml`) with all morph joints removed
- Use the frozen scene for all subsequent evaluations

**Benefit:** Eliminates live morph-joint artifacts; contact metrics become stable and repeatable.

### 2. Trajectory Evaluation in Phase1GraspEvaluator
**File:** `src/morphohand/optimization/phase1_common.py`

Added methods to `Phase1GraspEvaluator`:

**`evaluate_trajectory(finger_ctrl_traj)` (uses TrajectoryInterpolator)**
- Core evaluation loop for a trajectory of shape `(phases, n_f)`.
- Follows the same settle→lift→pivot→hold structure as `evaluate()` but interpolates finger controls at each step via `TrajectoryInterpolator.at_dynamic_step()`.
- Computes the same contact, persistence, and kinematic metrics.
- **Note:** The interpolation logic was previously embedded in `_interp_finger_ctrl_for_dynamic()` but has been refactored into a dedicated `TrajectoryInterpolator` class for reusability and testability.

**`rollout_trajectory(finger_ctrl_traj)` & `render_rollout_gif_trajectory(...)`**
- Produce full trajectory data and GIF animation using interpolated finger controls.
- Essential for final visualization and metric collection.

### 3. Trajectory-Capable CEM Optimizer
**File:** `src/morphohand/optimization/phase1_strategy_cem.py`

Added `optimize_finger_control_trajectory()` function:
- Treats the parameter space as a flattened trajectory (phases × n_f).
- Samples, evaluates, and fits distributions over trajectory keypoints, not just a single fixed control.
- Generates `[CEM-TRAJ]` log output to distinguish from scalar-control optimization.

### 4. CLI and Script Integration
**File:** `scripts/phase1_optimize_grasp.py`

Changes:
- Added `--traj-phases` argument (default=1 for backward compatibility).
- When `traj_phases > 1`, routes to `optimize_finger_control_trajectory()`.
- Automatically selects `rollout_trajectory()` and `render_rollout_gif_trajectory()` for multi-phase runs.
- Saves trajectory as shape `(phases, n_f)` in summary.json.

### 5. Launcher & Pivot Axis Fix
**File:** `scripts/run8_power_drill_all_in_one.sh`

- Flipped `pivot_delta_ry = -1.20` → `pivot_delta_ry = 1.20` (prior run).
- Updated to use `pivot_delta_rz = 1.5708` (90° rotation about Z-axis to achieve palm-down orientation).
- Later, transitioned to `rz` for cleaner reorientation semantics.

## Experimental Results

### Run 8 (Baseline – Static Control, negative ry tilt)
- Scene: `scene_power_drill_short_proximal.xml` (live morph)
- Iterations: 32, Population: 48
- Pivot: `ry = -1.20`
- **Result:** Best score = 0.841254, contacts = 6.0
- **Issue:** Fingers sliding on live morph joints; wrong tilt axis.

### Run 9 (Frozen Scene, Positive ry tilt)
- Scene: Frozen from Run 8 (`frozen_scene.xml`)
- Iterations: 32, Population: 48
- Pivot: `ry = 1.20` (corrected sign)
- **Result:** Best score = 6.783975, contacts = 7.0
- **Improvement:** Score increased due to contact persistence weighting; still only 7 contacts.

### Run 10 (Static Control, Contact Focus, rz axis)
- Scene: Frozen from Run 9
- Iterations: 80, Population: 128
- Pivot: `rz = 1.5708` (Z-axis rotation)
- Objective weights: Increased `contact_persistence = 3.0`, `min_finger_persistence = 6.0`
- **Result:** Best score = 13.238693, contacts = 7.0
- **Observation:** Still only 7 contacts despite higher weights; static control is insufficient to maintain middle finger contact through reorientation.

### Run 11 (Trajectory Control, 4 Phases, rz axis)
- Scene: Frozen from Run 10
- Iterations: 120, Population: 160
- **Settings:**
  - `traj_phases = 4` (settle, lift, pivot, hold)
  - `pivot_delta_rz = 1.5708`
  - Objective weights: `contact_persistence = 4.0`, `min_finger_persistence = 8.0`, `finger_persistence_imbalance_penalty = 0.1`
  - Softer early termination tuning

- **Result:** Best score = 11.698247, contacts = 4.0
- **Lift:** 0.116878 (much higher than previous runs)
- **Issues:**
  - Contact count regressed to 4 (worst yet among trajectory runs).
  - Lift significantly increased (positive side effect).
  - Hypothesis: Optimizer prioritized lift over contact persistence; flexibility in finger control during pivot caused fingers to disengage to reduce grip forces.

## Current Status

✅ **Achieved:**
- Frozen morphology scene generation and persistent use in optimization.
- Correct pivot axis (`rz = 1.5708` for palm-down reorientation).
- Time-varying finger-control trajectory optimization infrastructure.
- Smooth grasp+lift+reorientation motion (visible in GIFs).
- Modular trajectory evaluation and CEM routines.

⚠️ **Outstanding:**
- **Contact count degradation in Run 11:** Despite higher finger-persistence weights, trajectory optimization led to fewer simultaneous contacts. Likely causes:
  - Objective function balance: lift reward (35×) vastly outweighs contact penalties.
  - Finger control flexibility: Allowing per-phase adjustments may admit solutions that reduce grip mid-trajectory.
  - Elite selection: Population of 160 may be too large; convergence slower.

🔧 **Next Steps:**
1. **Rebalance objectives** — Increase penalty on finger-persistence imbalance; reduce lift reward during non-lift phases.
2. **Constrain trajectory smoothness** — Add regularization/smoothness term to prevent wild finger control changes across phases.
3. **Tune CEM parameters** — Try smaller population + longer iterations (exploration vs. exploitation trade-off).
4. **Increase minimum contact target** — Use explicit contact-count objective (e.g., penalty for < 9 contacts).
5. **Scene geometry inspection** — Verify that open_flat pose actually permits 9-contact stable grasp.

## Code Quality: Completed Refactoring

✅ **Refactoring Completed:**
1. **`TrajectoryInterpolator` utility** (in `phase1_trajectory.py`) — Moved interpolation logic from the embedded `_interp_finger_ctrl_for_dynamic()` method into a dedicated, reusable class.
2. **Factory function** `build_trajectory_interpolator()` — Creates interpolators from evaluator configs for clean initialization.
3. **Removed embedding** — Deleted `_interp_finger_ctrl_for_dynamic()` from `Phase1GraspEvaluator`; all dynamic phases now use `interp.at_dynamic_step(t)`.

### Benefits
- **Testable:** TrajectoryInterpolator can be unit-tested independently.
- **Reusable:** Can be imported and used in other optimization strategies or analysis tools.
- **Decoupled:** Interpolation logic no longer mixed with evaluator state management.
- **Cleaner:** Phase loops in `execute_sequential()` are more readable without interpolation branching.

### Remaining Challenges
- `Phase1GraspEvaluator` still ~1500+ lines with both scalar and trajectory evaluation paths.
- Branching in `phase1_optimize_grasp.py` for trajectory vs. scalar control could use strategy pattern.

### Future Refactoring (if needed)
1. Extract trajectory-specific evaluator logic into a `Phase1TrajectoryEvaluator` subclass.
2. Create a `TrajectoryCEM` strategy module to keep trajectory-specific optimizer separate.
3. Simplify `phase1_optimize_grasp.py` with cleaner branching using strategy pattern.

## Files Modified

- `src/morphohand/optimization/phase1_common.py` — Added trajectory evaluation methods.
- `src/morphohand/optimization/phase1_strategy_cem.py` — Added `optimize_finger_control_trajectory()`.
- `scripts/phase1_optimize_grasp.py` — Added `--traj-phases` argument; integrated trajectory branch.
- `scripts/run8_power_drill_all_in_one.sh` — Updated pivot axis and signs.
- `README.md` — Updated protocol to require frozen-scene XML.
- `docs/architecture/4-30_experiments.md` (this file) — New documentation.
