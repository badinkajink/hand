"""Tests for `morphohand.rl.reward`.

Two flavors:
- Unit tests for individual reward terms on synthetic inputs.
- A semi-integration test that constructs a `RewardState` matching the
  *reference itself* (state = reference) and asserts the tracking terms
  saturate near their max. This is the property the curriculum relies on:
  a policy that exactly tracks the reference should get full tracking
  reward — if it can't, the reference is broken (or the env state diverges
  from CPU MuJoCo, see the Warp-vs-CPU risk in docs/rl/training.md).
"""
from __future__ import annotations

import numpy as np
import pytest

from morphohand.rl.reward import (
    DEFAULT_REWARD_WEIGHTS, RewardState,
    compute_total_reward,
    track_finger_qpos, track_object_pos, track_object_quat,
    task_lift_height, task_contact_persistence, task_min_finger_persist,
    pen_drop, pen_action_rate, pen_cube_axis_tilt,
)


# ---- individual term sanity checks ---------------------------------------

def test_track_finger_qpos_max_at_zero_diff():
    x = np.array([0.5, -0.3, 1.2, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1])
    assert track_finger_qpos(x, x) == pytest.approx(1.0)


def test_track_finger_qpos_decays_with_diff():
    x = np.zeros(9)
    y = np.full(9, 0.5)
    r0 = track_finger_qpos(x, x)
    r1 = track_finger_qpos(x, y)
    assert 0.0 <= r1 < r0 == pytest.approx(1.0)


def test_track_object_quat_geodesic_is_one_at_identity():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert track_object_quat(q, q) == pytest.approx(1.0)


def test_track_object_quat_handles_double_cover():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    qneg = -q
    assert track_object_quat(q, qneg) == pytest.approx(1.0, rel=1e-6)


def test_task_lift_height_clips_to_target():
    assert task_lift_height(1.0, 0.0, target_lift=0.05) == pytest.approx(0.05)
    assert task_lift_height(-0.5, 0.0, target_lift=0.05) == 0.0
    assert task_lift_height(0.03, 0.0, target_lift=0.05) == pytest.approx(0.03)


def test_task_contact_persistence_max_at_all_in_contact():
    assert task_contact_persistence(np.ones(3)) == pytest.approx(1.0)
    assert task_contact_persistence(np.zeros(3)) < task_contact_persistence(np.ones(3))


def test_task_min_finger_persist_dominated_by_worst():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0, 1.0, 0.0])
    assert task_min_finger_persist(a) > task_min_finger_persist(b)


def test_pen_drop_is_indicator():
    assert pen_drop(0.0, 1.0) == 1.0
    assert pen_drop(0.99, 1.0) == 0.0
    assert pen_drop(0.97, 1.0, drop_threshold=0.02) == 1.0


def test_pen_action_rate_zero_for_static_action():
    a = np.array([0.5, -0.3, 0.0])
    assert pen_action_rate(a, a) == 0.0


def test_pen_cube_axis_tilt_zero_at_identity():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert abs(pen_cube_axis_tilt(q)) < 1e-9


# ---- composite scorer ----------------------------------------------------

def _make_reference_state() -> RewardState:
    """state = reference: tracking terms should saturate near 1.0."""
    finger_q = np.array([0.083, 1.896, -0.932, -0.086, 1.371, 1.100, 0.034, 1.375, 1.112])
    return RewardState(
        finger_qpos=finger_q,
        finger_qpos_ref=finger_q,
        object_pos=np.array([0.0, 0.0, 0.07]),
        object_pos_ref=np.array([0.0, 0.0, 0.07]),
        object_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        object_quat_ref=np.array([1.0, 0.0, 0.0, 0.0]),
        action=np.zeros(9),
        action_ref=np.zeros(9),
        action_prev=np.zeros(9),
        per_tip_contact=np.ones(3),
        object_z_initial=0.02,
        palm_pos=np.array([0.0, 0.0, 0.085]),
        object_vel=np.zeros(3),
        object_xy_initial=np.array([0.0, 0.0]),
        finger_yaw_initial=finger_q[[0, 3, 6]],
        finger_flex_initial=finger_q[[1, 2, 4, 5, 7, 8]],
        object_quat_initial=np.array([1.0, 0.0, 0.0, 0.0]),
        object_angvel=np.zeros(3),
    )


def test_total_reward_on_reference_is_positive_and_dominated_by_tracking():
    s = _make_reference_state()
    total, parts = compute_total_reward(s)
    assert total > 0.0
    # All tracking terms should saturate at ~weight × 1.0
    assert parts["track_finger_qpos"] == pytest.approx(DEFAULT_REWARD_WEIGHTS["track_finger_qpos"][1])
    assert parts["track_object_pos"] == pytest.approx(DEFAULT_REWARD_WEIGHTS["track_object_pos"][1])
    assert parts["track_object_quat"] == pytest.approx(DEFAULT_REWARD_WEIGHTS["track_object_quat"][1])
    # Penalties should be ~0 (state matches reference).
    assert abs(parts["pen_drop"]) < 1e-9
    assert abs(parts["pen_action_rate"]) < 1e-9


def test_total_reward_drops_when_object_falls():
    """Drop the object: pen_drop fires, lift_height collapses, total reward drops."""
    good = _make_reference_state()
    bad = _make_reference_state()
    bad.object_pos = np.array([0.0, 0.0, -0.01])  # well below initial 0.02
    total_good, _ = compute_total_reward(good)
    total_bad, parts_bad = compute_total_reward(bad)
    assert parts_bad["pen_drop"] < 0  # active penalty
    assert total_bad < total_good
