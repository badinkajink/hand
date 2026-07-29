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

**`rollout_trajectory(finger_ctrl_traj)` & `render_rollout_trajectory(...)`**
- Produce full trajectory data and MP4 animation using interpolated finger controls.
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
- Automatically selects `rollout_trajectory()` and `render_rollout_trajectory()` for multi-phase runs.
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

### Run 12 (Trajectory Control, Contact-Focused Weights)
- Scene: Frozen from Run 11
- Iterations: 120, Population: 160
- **Settings:**
  - `traj_phases = 4`, `pivot_delta_rz = 1.5708`, `lift_delta_z = 0.060`
  - Boosted contact weights: `contact_persistence = 6.0`, `min_finger_persistence = 10.0`
  - Reduced cube-orientation penalties: `cube_yaw_drift = 3.0`, `cube_axis_tilt = 2.0`, `cube_ang_drift = 1.0`

- **Result:** Best score = 16.287, contacts = 4.0, lift = 0.117
- **What improved:** all-finger persistence reached 1.0 (every fingertip stays in contact for 100% of dynamic steps). `mean_tip_distance ≈ 0`, `finger_persistence_imbalance = 0`.
- **What did not:** raw contact count is still 4, and `cube_axis_tilt = 1.166 rad (~67°)` plus `cube_xy_drift = 0.083 m`. With the new sphere-shell power-drill collision proxy (8 spheres on one body), "contacts = 4" no longer means "fingers detached" — the persistence metrics show all three tips engaged throughout. The drill itself is rolling under the grip during the pivot. **The contact-count metric is now ambiguous for multi-sphere objects and should be retired in favor of per-finger persistence.**

## Current Status

✅ **Achieved:**
- Frozen morphology scene generation and persistent use in optimization.
- Correct pivot axis (`rz = 1.5708` for palm-down reorientation).
- Time-varying finger-control trajectory optimization infrastructure.
- Smooth grasp+lift+reorientation motion (visible in GIFs).
- Modular trajectory evaluation and CEM routines.

⚠️ **Outstanding:**
- **Contact-count metric is ambiguous** under the new sphere-shell collision proxy on the power drill. Per-finger persistence is the truer signal and reads 1.0 for all tips in Run 12.
- **Drill is rotating under the grip during pivot** (`cube_axis_tilt ≈ 67°`, `cube_xy_drift ≈ 8 cm`). The objective prioritized hand-side persistence; the cube-orientation penalties were intentionally reduced and need to come back up.
- **No smoothness constraint on the finger trajectory** — adjacent phase keypoints can diverge sharply, which encourages mid-pivot grip changes that destabilize the object.

🔧 **Next Steps:**
1. **Retire raw contact count from the score**; keep it as a diagnostic but score on per-finger persistence + cube tilt.
2. **Rebalance objectives** — push `cube_axis_tilt_penalty` and `cube_ang_drift_penalty` back up while keeping high finger-persistence weights.
3. **Add trajectory smoothness regularizer** — penalize `||traj[i+1] - traj[i]||` to suppress mid-pivot grip jumps.
4. **Inspect drill collision shell** — the 8-sphere approximation may be letting fingertips slip into seams; check sphere coverage near the grip ring.

## Code Quality: Completed Refactoring (post-cleanup pass)

After the initial trajectory work, the evaluator and CEM strategy carried significant
duplication between scalar and trajectory paths. Cleaned up:

1. **`Phase1GraspEvaluator` deduped** (was ~1075 lines, now ~819):
   - `_pose_scales_for_dynamic_step(dynamic_t)` — single source of truth for lift/pivot ramping.
   - `_ctrl_for_dynamic_step(dynamic_t, get_finger_ctrl)` — combines scales + `_build_pose_ctrl`.
   - `_compute_score_and_metrics(...)` — single scoring formula (was duplicated in `evaluate` and `evaluate_trajectory`).
   - `_run_dynamic_loop_sampled(get_finger_ctrl)` — single sampled lift→pivot→hold loop.
   - `_run_dynamic_loop_terminal(get_finger_ctrl, z_before)` — preserves the warp-backend terminal-mode optimization.
   - `_evaluate_with_provider`, `_rollout_with_provider`, `_render_rollout_with_provider` — three shared bodies; the public `evaluate*`, `rollout*`, `render_rollout*` methods are thin wrappers that pass either a constant `lambda _t: finger_ctrl` or `interp.at_dynamic_step` as the provider.
   - Numerically equivalent: `evaluate(fc) == evaluate_trajectory(tile(fc, (4,1)))` to <1e-9 on score and all metrics; rollouts agree to ~1e-12.
2. **CEM deduped** — `_run_cem(evaluate_sample, ...)` is the shared loop; the two public optimizers reshape inputs/outputs around it.
3. **Asset-path rebaser deduped** — `rebase_asset_file_paths` lives once in `morphohand.tools.morphology_xml` and is reused by `morphohand.sampling.scene`.
4. **Dead code removed** — unused `_build_lift_ctrl`, `_step_chunk_with_ctrl`, `TrajectoryInterpolator.lift_ramp_steps`/`pivot_ramp_steps` (never read), and a dangling `dims = n_f * phases` in CEM.

## Files Modified

- `src/morphohand/optimization/phase1_common.py` — Trajectory evaluation methods, then deduped against scalar path via shared providers.
- `src/morphohand/optimization/phase1_strategy_cem.py` — Trajectory CEM, then collapsed onto shared `_run_cem` loop.
- `src/morphohand/optimization/phase1_trajectory.py` — `TrajectoryInterpolator` extracted into its own module; unused ramp params removed.
- `src/morphohand/sampling/{scene.py,feasibility.py,foundational.py}` — Cube-orientation drift fields, vertical-keyframe criteria, recursive seed-folder traversal.
- `src/morphohand/tools/morphology_xml.py` — Promoted `rebase_asset_file_paths`, added `_strip_scene_morph_qpos` for rigid keyframes.
- `scripts/phase1_optimize_grasp.py` — `--traj-phases`, frozen-scene generation, pivot/cube-drift CLI flags.
- `scripts/run6_combined_multitask.py`, `scripts/run6_screwdriver_multikey_sampling.py`, `scripts/run6_analysis.py` — Per-keyframe feasibility criteria, pivot/cube-drift flags, rigid-scene emission.
- `scripts/run12_contact_focused_traj.sh` — Contact-focused weight sweep.
- `assets/mjcf/baseline/scenes/scene_power_drill.xml` — Sphere-shell collision proxy and `hand`/`object` collision classes.
- `README.md`, `docs/architecture/overview.md`, `docs/architecture/4-30_experiments.md` (this file).
