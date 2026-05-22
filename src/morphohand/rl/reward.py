"""Reward terms for the RL policy, as pure functions over (state, ref).

Two families:

- *Tracking* terms — Opt2Skill-style exp(-alpha * ||x - x_ref||^2) ∈ (0, 1].
- *Task* terms — port the Phase1 evaluator's per-step quantities (contact
  persistence, lift, finger anchor, object drift) into per-step rewards.

Per the plan (`docs/rl/architecture.md`), these functions are
*pure*: they take numpy/torch arrays and return scalars/arrays. The mjlab
RewTerm wiring lives in `env_cfg.py`; reward formulas live here so they're
testable without an env.

Default weights (per Phase 3 table) live in `DEFAULT_REWARD_WEIGHTS`.
A curriculum scales the (tracking, task) splits over training; see
`env_cfg.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

# Per-term defaults — keep in sync with docs/rl/architecture.md table.
# Each entry: (kind, weight, alpha). kind ∈ {"track", "penalty", "linear", "bonus"}.
DEFAULT_REWARD_WEIGHTS: dict[str, tuple[str, float, float]] = {
    # Tracking — exp(-alpha * ||.||^2).
    "track_finger_qpos":         ("track",   4.0, 20.0),
    "track_object_pos":          ("track",   6.0, 200.0),
    "track_object_quat":         ("track",   2.0, 10.0),
    "track_finger_ctrl_anchor":  ("track",   1.0, 4.0),
    # Task — inherits from phase1_common.py weights.
    "task_contact_persistence":  ("track",   0.8, 1.0),
    "task_min_finger_persist":   ("track",   2.0, 1.0),
    "task_lift_height":          ("linear", 35.0, 0.0),
    "task_distance_palm_to_obj": ("track",   2.0, 15.0),
    # Penalties — clipped linear or sparse indicator.
    "pen_object_velocity":       ("penalty", 0.15, 0.0),
    "pen_xy_drift":              ("penalty", 6.0,  0.0),
    "pen_drop":                  ("penalty", 12.0, 0.0),
    "pen_finger_yaw_drift":      ("penalty", 0.8,  0.0),
    "pen_finger_flex_drift":     ("penalty", 0.4,  0.0),
    "pen_cube_yaw_drift":        ("penalty", 4.0,  0.0),
    "pen_cube_axis_tilt":        ("penalty", 6.0,  0.0),
    "pen_cube_ang_drift":        ("penalty", 2.0,  0.0),
    "pen_action_rate":           ("penalty", 0.05, 0.0),
    "pen_action_l2":             ("penalty", 0.01, 0.0),
    # Survival.
    "bonus_alive":               ("bonus",   0.1,  0.0),
}


# ---- pure reward functions ---------------------------------------------

def _track(diff: np.ndarray, alpha: float) -> float:
    """Opt2Skill-style tracking reward: exp(-alpha * mean(diff^2))."""
    return float(np.exp(-alpha * float(np.mean(np.asarray(diff) ** 2))))


def track_finger_qpos(finger_qpos: np.ndarray, finger_qpos_ref: np.ndarray,
                       alpha: float = 20.0) -> float:
    return _track(finger_qpos - finger_qpos_ref, alpha)


def track_object_pos(object_pos: np.ndarray, object_pos_ref: np.ndarray,
                      alpha: float = 200.0) -> float:
    return _track(object_pos - object_pos_ref, alpha)


def track_object_quat(object_quat: np.ndarray, object_quat_ref: np.ndarray,
                       alpha: float = 10.0) -> float:
    """Geodesic squared distance on unit quaternions: 1 - <q, qref>^2."""
    dot = float(np.dot(object_quat, object_quat_ref))
    geo_sq = max(0.0, 1.0 - dot * dot)
    return float(np.exp(-alpha * geo_sq))


def track_finger_ctrl_anchor(action: np.ndarray, action_ref: np.ndarray,
                              alpha: float = 4.0) -> float:
    return _track(action - action_ref, alpha)


# ---- task terms (port of Phase1 reward signals) ------------------------

def task_contact_persistence(per_tip_contact: np.ndarray, alpha: float = 1.0) -> float:
    """per_tip_contact: (3,) booleans/floats in [0, 1]. Reward = exp(-alpha*(1 - mean))."""
    mean = float(np.mean(per_tip_contact))
    return float(np.exp(-alpha * (1.0 - mean)))


def task_min_finger_persist(per_tip_contact: np.ndarray, alpha: float = 1.0) -> float:
    """Reward dominated by the *worst* finger — discourages 2-finger grips."""
    worst = float(np.min(per_tip_contact))
    return float(np.exp(-alpha * (1.0 - worst)))


def task_lift_height(object_z: float, object_z_initial: float,
                      target_lift: float = 0.05) -> float:
    """Clipped lift: linear up to `target_lift` above the initial settle z."""
    lift = float(object_z - object_z_initial)
    return float(max(0.0, min(lift, target_lift)))


def task_distance_palm_to_obj(palm_pos: np.ndarray, object_pos: np.ndarray,
                               alpha: float = 15.0) -> float:
    d2 = float(np.sum((np.asarray(palm_pos) - np.asarray(object_pos)) ** 2))
    return float(np.exp(-alpha * d2))


# ---- penalties ---------------------------------------------------------

def pen_object_velocity(object_vel: np.ndarray) -> float:
    return float(np.linalg.norm(object_vel))


def pen_xy_drift(object_xy: np.ndarray, object_xy_initial: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(object_xy) - np.asarray(object_xy_initial)))


def pen_drop(object_z: float, object_z_initial: float, drop_threshold: float = 0.02) -> float:
    return 1.0 if object_z < object_z_initial - drop_threshold else 0.0


def pen_finger_yaw_drift(finger_yaw_now: np.ndarray, finger_yaw_initial: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(finger_yaw_now) - np.asarray(finger_yaw_initial))))


def pen_finger_flex_drift(finger_flex_now: np.ndarray, finger_flex_initial: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(finger_flex_now) - np.asarray(finger_flex_initial))))


def pen_cube_yaw_drift(object_quat_now: np.ndarray, object_quat_initial: np.ndarray) -> float:
    """Approx yaw-only drift via z-axis projection (cheap, OK as a penalty)."""
    import numpy as np
    def yaw_from_quat(q: np.ndarray) -> float:
        w, x, y, z = q
        # ZYX yaw extraction.
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return float(abs(yaw_from_quat(np.asarray(object_quat_now))
                     - yaw_from_quat(np.asarray(object_quat_initial))))


def pen_cube_axis_tilt(object_quat: np.ndarray) -> float:
    """How far the object's z-axis tilts from world-z. Angle in radians."""
    w, x, y, z = np.asarray(object_quat)
    cos_tilt = float(1.0 - 2.0 * (x * x + y * y))
    cos_tilt = max(-1.0, min(1.0, cos_tilt))
    return float(np.arccos(cos_tilt))


def pen_cube_ang_drift(object_angvel: np.ndarray) -> float:
    return float(np.linalg.norm(object_angvel))


def pen_action_rate(action: np.ndarray, action_prev: np.ndarray) -> float:
    return float(np.mean((np.asarray(action) - np.asarray(action_prev)) ** 2))


def pen_action_l2(action: np.ndarray) -> float:
    return float(np.mean(np.asarray(action) ** 2))


# ---- composite scorer (used in tests + eval) ---------------------------

@dataclass
class RewardState:
    """The minimum slice of state needed to compute the full per-step reward."""
    finger_qpos: np.ndarray
    finger_qpos_ref: np.ndarray
    object_pos: np.ndarray
    object_pos_ref: np.ndarray
    object_quat: np.ndarray
    object_quat_ref: np.ndarray
    action: np.ndarray
    action_ref: np.ndarray
    action_prev: np.ndarray
    per_tip_contact: np.ndarray
    object_z_initial: float
    palm_pos: np.ndarray
    object_vel: np.ndarray
    object_xy_initial: np.ndarray
    finger_yaw_initial: np.ndarray
    finger_flex_initial: np.ndarray
    object_quat_initial: np.ndarray
    object_angvel: np.ndarray


def compute_total_reward(s: RewardState,
                         weights: Mapping[str, tuple[str, float, float]] | None = None
                         ) -> tuple[float, dict[str, float]]:
    """Apply DEFAULT_REWARD_WEIGHTS (or overrides) to a RewardState.

    Returns (total, per-term-dict). per-term values are post-weighted
    contributions so they sum to total.
    """
    w = dict(DEFAULT_REWARD_WEIGHTS if weights is None else weights)

    def W(k: str) -> tuple[str, float, float]:
        return w[k]

    finger_yaw_idx = np.array([0, 3, 6])
    finger_flex_idx = np.array([1, 2, 4, 5, 7, 8])

    parts: dict[str, float] = {}
    parts["track_finger_qpos"]         = W("track_finger_qpos")[1] * track_finger_qpos(
        s.finger_qpos, s.finger_qpos_ref, W("track_finger_qpos")[2])
    parts["track_object_pos"]          = W("track_object_pos")[1] * track_object_pos(
        s.object_pos, s.object_pos_ref, W("track_object_pos")[2])
    parts["track_object_quat"]         = W("track_object_quat")[1] * track_object_quat(
        s.object_quat, s.object_quat_ref, W("track_object_quat")[2])
    parts["track_finger_ctrl_anchor"]  = W("track_finger_ctrl_anchor")[1] * track_finger_ctrl_anchor(
        s.action, s.action_ref, W("track_finger_ctrl_anchor")[2])
    parts["task_contact_persistence"]  = W("task_contact_persistence")[1] * task_contact_persistence(
        s.per_tip_contact, W("task_contact_persistence")[2])
    parts["task_min_finger_persist"]   = W("task_min_finger_persist")[1] * task_min_finger_persist(
        s.per_tip_contact, W("task_min_finger_persist")[2])
    parts["task_lift_height"]          = W("task_lift_height")[1] * task_lift_height(
        float(s.object_pos[2]), s.object_z_initial)
    parts["task_distance_palm_to_obj"] = W("task_distance_palm_to_obj")[1] * task_distance_palm_to_obj(
        s.palm_pos, s.object_pos, W("task_distance_palm_to_obj")[2])
    parts["pen_object_velocity"]       = -W("pen_object_velocity")[1] * pen_object_velocity(s.object_vel)
    parts["pen_xy_drift"]              = -W("pen_xy_drift")[1] * pen_xy_drift(
        s.object_pos[:2], s.object_xy_initial)
    parts["pen_drop"]                  = -W("pen_drop")[1] * pen_drop(
        float(s.object_pos[2]), s.object_z_initial)
    parts["pen_finger_yaw_drift"]      = -W("pen_finger_yaw_drift")[1] * pen_finger_yaw_drift(
        s.finger_qpos[finger_yaw_idx], s.finger_yaw_initial)
    parts["pen_finger_flex_drift"]     = -W("pen_finger_flex_drift")[1] * pen_finger_flex_drift(
        s.finger_qpos[finger_flex_idx], s.finger_flex_initial)
    parts["pen_cube_yaw_drift"]        = -W("pen_cube_yaw_drift")[1] * pen_cube_yaw_drift(
        s.object_quat, s.object_quat_initial)
    parts["pen_cube_axis_tilt"]        = -W("pen_cube_axis_tilt")[1] * pen_cube_axis_tilt(s.object_quat)
    parts["pen_cube_ang_drift"]        = -W("pen_cube_ang_drift")[1] * pen_cube_ang_drift(s.object_angvel)
    parts["pen_action_rate"]           = -W("pen_action_rate")[1] * pen_action_rate(s.action, s.action_prev)
    parts["pen_action_l2"]             = -W("pen_action_l2")[1] * pen_action_l2(s.action)
    parts["bonus_alive"]               = W("bonus_alive")[1]
    return float(sum(parts.values())), parts
