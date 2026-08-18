"""Facade over the split mjlab term modules (CODEBASE_AUDIT.md step 3).

The implementation now lives in:

  - `terms_obs.py`     — observation terms
  - `terms_reward.py`  — reward terms + curriculum weight/DR anneals
  - `terms_event.py`   — event terms (bank injection, compliance DR) + terminations
  - `terms_common.py`  — shared package-private helpers
  - `math.py`          — batched quaternion ops

Every public (and historically-poked private) name is re-exported here so
existing references — env-build wiring, scripts, archived experiments, and any
pickled cfg that resolves `morphohand.rl.mjlab_terms.<fn>` — keep working.
Prefer importing from the specific module in new code.
"""
from __future__ import annotations

from morphohand.rl.math import quat_mul as _quat_mul  # noqa: F401  (historic name)
from morphohand.rl.math import quat_rotate as _quat_rotate  # noqa: F401
from morphohand.rl.terms_common import (  # noqa: F401
    _FINGER_JOINT_NAMES,
    _alignment_cos,
    _alignment_hold_counter,
    _contact_force_mag,
    _contact_gate,
    _get_env_time,
    _get_finger_action,
    _get_finger_joint_ids,
    _get_finger_qpos,
    _get_ref,
    _get_ref_batch,
    _get_step_dt,
    _in_lift_phase,
    _spawn_pose,
    _track,
)
from morphohand.rl.terms_event import (  # noqa: F401
    inject_handoff_bank_at_onset,
    randomize_geom_solimp,
    reset_from_handoff_bank,
    terminate_alignment_success,
    terminate_any_tip_lost,
    terminate_finger_slip,
    terminate_low_tilt_velocity,
    terminate_object_drop,
    terminate_object_floor_proximity,
    terminate_object_orientation_slip,
    terminate_object_slip,
    terminate_tip_lost,
)
from morphohand.rl.terms_obs import (  # noqa: F401
    object_pose_rel_palm,
    ref_finger_qpos,
    ref_object_pose,
    target_axis_misalignment,
)
from morphohand.rl.terms_reward import (  # noqa: F401
    alignment_speed_bonus,
    alignment_success_bonus,
    anneal_cube_spawn_jitter,
    anneal_smoothness_weights,
    anneal_spawn_tilt_z,
    anneal_target_axis_alpha,
    anneal_tracking_weights,
    finger_drift_from_grip,
    finger_drift_from_grip_gated,
    fingertip_contact_mean,
    fingertip_contact_min,
    fingertip_to_object_distance,
    grip_force,
    grip_force_excess,
    grip_force_spread,
    handoff_target_proximity,
    object_ang_acc_l2,
    object_axial_slip,
    object_axial_slip_gated,
    object_drop_indicator,
    object_lateral_drift,
    object_lateral_drift_gated,
    object_lift_height,
    object_orientation_drift,
    object_orientation_drift_gated,
    object_xy_drift,
    object_xy_drift_gated,
    palm_brace_distance,
    palm_brace_force,
    reorient_time_cost,
    target_axis_alignment,
    thumb_brace_force,
    target_axis_progress,
    track_finger_ctrl_anchor,
    track_finger_qpos,
    track_object_pos,
    track_object_quat,
)
