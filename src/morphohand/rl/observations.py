"""Observation extractors as pure functions.

Each `obs_*` function takes numpy arrays (the env state slice) and returns
a fixed-length numpy array, normalized + clipped per the plan
(`docs/rl/architecture.md` Phase 4). The mjlab `ObsTerm` wrappers in
`env_cfg.py` adapt these to torch tensors over the vectorised env.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Standard clip bounds for everything except joint pos (which is range-normalized).
_OBS_CLIP = 5.0
_VEL_CLIP = 10.0


def normalize_to_range(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Map x in [lo, hi] -> [-1, 1]. Out-of-range x is clipped to [-1.1, 1.1]."""
    span = np.maximum(hi - lo, 1e-9)
    y = 2.0 * (np.asarray(x) - lo) / span - 1.0
    return np.clip(y, -1.1, 1.1)


def obs_finger_joint_pos(finger_qpos: np.ndarray,
                          finger_lo: np.ndarray,
                          finger_hi: np.ndarray) -> np.ndarray:
    """9-d. Range-normalized finger joint positions."""
    return normalize_to_range(finger_qpos, finger_lo, finger_hi)


def obs_finger_joint_vel(finger_qvel: np.ndarray) -> np.ndarray:
    """9-d. Clipped to ±VEL_CLIP rad/s."""
    return np.clip(np.asarray(finger_qvel), -_VEL_CLIP, _VEL_CLIP)


def obs_palm_joint_pos(palm_qpos: np.ndarray,
                        palm_lo: np.ndarray,
                        palm_hi: np.ndarray) -> np.ndarray:
    """6-d. Informational (palm is scripted in MVP; still useful for the policy)."""
    return normalize_to_range(palm_qpos, palm_lo, palm_hi)


def obs_object_pos_rel_palm(object_pos: np.ndarray, palm_pos: np.ndarray) -> np.ndarray:
    """3-d. Object position expressed in the palm base frame (no rotation)."""
    return np.clip(np.asarray(object_pos) - np.asarray(palm_pos), -_OBS_CLIP, _OBS_CLIP)


def obs_object_quat(object_quat: np.ndarray) -> np.ndarray:
    """4-d. Unit quaternion (wxyz). Already bounded."""
    return np.asarray(object_quat)


def obs_object_lin_vel(object_lin_vel: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(object_lin_vel), -_OBS_CLIP, _OBS_CLIP)


def obs_object_ang_vel(object_ang_vel: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(object_ang_vel), -_OBS_CLIP, _OBS_CLIP)


def obs_fingertip_contact(per_tip_distance: np.ndarray,
                           per_tip_contact: np.ndarray,
                           per_tip_normal_z: np.ndarray) -> np.ndarray:
    """9-d. 3 tips × {distance, contact_flag, contact_normal_z}."""
    out = np.stack([per_tip_distance, per_tip_contact, per_tip_normal_z], axis=-1)
    return np.clip(out.reshape(-1), -_OBS_CLIP, _OBS_CLIP)


def obs_time_phase(t: float, duration: float) -> np.ndarray:
    """1-d. Episode phase in [0, 1]."""
    if duration <= 0.0:
        return np.array([0.0])
    return np.array([float(np.clip(t / duration, 0.0, 1.0))])


def obs_previous_action(action_prev: np.ndarray) -> np.ndarray:
    """9-d."""
    return np.clip(np.asarray(action_prev), -1.1, 1.1)


# ---- composite assembler used by tests + the env wrapper ---------------

@dataclass
class ObsState:
    finger_qpos: np.ndarray
    finger_qvel: np.ndarray
    palm_qpos: np.ndarray
    object_pos: np.ndarray
    palm_pos: np.ndarray
    object_quat: np.ndarray
    object_lin_vel: np.ndarray
    object_ang_vel: np.ndarray
    per_tip_distance: np.ndarray
    per_tip_contact: np.ndarray
    per_tip_normal_z: np.ndarray
    t: float
    duration: float
    action_prev: np.ndarray
    # Lookahead refs (resolved from ReferenceTrajectory in the env wrapper).
    finger_qpos_ref_lookahead: np.ndarray
    object_pose_ref_lookahead: np.ndarray   # 7 = pos(3) + quat(4)
    # Bounds.
    finger_lo: np.ndarray
    finger_hi: np.ndarray
    palm_lo: np.ndarray
    palm_hi: np.ndarray


def compute_obs_vector(s: ObsState) -> np.ndarray:
    """Concatenate all obs terms into a single flat vector.

    Matches the layout documented in docs/rl/architecture.md (≈ 85-dim).
    """
    parts = [
        obs_finger_joint_pos(s.finger_qpos, s.finger_lo, s.finger_hi),        # 9
        obs_finger_joint_vel(s.finger_qvel),                                  # 9
        obs_palm_joint_pos(s.palm_qpos, s.palm_lo, s.palm_hi),                # 6
        obs_object_pos_rel_palm(s.object_pos, s.palm_pos),                    # 3
        obs_object_quat(s.object_quat),                                       # 4
        obs_object_lin_vel(s.object_lin_vel),                                 # 3
        obs_object_ang_vel(s.object_ang_vel),                                 # 3
        obs_fingertip_contact(s.per_tip_distance, s.per_tip_contact,
                              s.per_tip_normal_z),                            # 9
        obs_time_phase(s.t, s.duration),                                      # 1
        obs_previous_action(s.action_prev),                                   # 9
        s.finger_qpos_ref_lookahead,                                          # 9
        s.object_pose_ref_lookahead,                                          # 7
    ]
    out = np.concatenate(parts, axis=0)
    return np.nan_to_num(out, nan=0.0, posinf=_OBS_CLIP, neginf=-_OBS_CLIP)


# Expected total length — used by tests for shape assertions.
OBS_DIM = 9 + 9 + 6 + 3 + 4 + 3 + 3 + 9 + 1 + 9 + 9 + 7  # = 72; add lookahead variants if extending
