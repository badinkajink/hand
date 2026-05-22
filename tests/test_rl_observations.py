"""Tests for `morphohand.rl.observations`."""
from __future__ import annotations

import numpy as np

from morphohand.rl.observations import (
    OBS_DIM, ObsState, compute_obs_vector,
    obs_finger_joint_pos, obs_object_quat, normalize_to_range,
)


def test_normalize_to_range_endpoints():
    lo = np.array([0.0, -1.0, 5.0])
    hi = np.array([2.0,  1.0, 10.0])
    np.testing.assert_allclose(normalize_to_range(lo, lo, hi), -1.0)
    np.testing.assert_allclose(normalize_to_range(hi, lo, hi),  1.0)
    np.testing.assert_allclose(normalize_to_range((lo + hi) / 2, lo, hi), 0.0)


def test_normalize_to_range_clips_out_of_bounds():
    lo = np.array([0.0])
    hi = np.array([1.0])
    assert normalize_to_range(np.array([2.0]), lo, hi)[0] <= 1.1 + 1e-9
    assert normalize_to_range(np.array([-1.0]), lo, hi)[0] >= -1.1 - 1e-9


def _make_obs_state() -> ObsState:
    return ObsState(
        finger_qpos=np.zeros(9),
        finger_qvel=np.zeros(9),
        palm_qpos=np.zeros(6),
        object_pos=np.array([0.0, 0.0, 0.02]),
        palm_pos=np.array([0.0, 0.0, 0.134]),
        object_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        object_lin_vel=np.zeros(3),
        object_ang_vel=np.zeros(3),
        per_tip_distance=np.zeros(3),
        per_tip_contact=np.ones(3),
        per_tip_normal_z=np.array([1.0, 1.0, 1.0]),
        t=0.0,
        duration=1.2,
        action_prev=np.zeros(9),
        finger_qpos_ref_lookahead=np.zeros(9),
        object_pose_ref_lookahead=np.zeros(7),
        finger_lo=np.full(9, -1.0),
        finger_hi=np.full(9,  1.0),
        palm_lo=np.full(6, -1.0),
        palm_hi=np.full(6,  1.0),
    )


def test_compute_obs_vector_no_nans_and_correct_length():
    s = _make_obs_state()
    v = compute_obs_vector(s)
    assert v.shape == (OBS_DIM,)
    assert np.isfinite(v).all()


def test_obs_object_quat_passthrough():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(obs_object_quat(q), q)


def test_obs_finger_joint_pos_zeros_when_centered():
    lo = np.full(9, -1.0)
    hi = np.full(9,  1.0)
    out = obs_finger_joint_pos(np.zeros(9), lo, hi)
    np.testing.assert_allclose(out, 0.0)


def test_obs_vector_contains_no_nans_when_input_has_nans():
    """The composite assembler nan-fills to zeros so the policy never sees NaN."""
    s = _make_obs_state()
    s.object_lin_vel = np.array([np.nan, 0.0, 0.0])
    v = compute_obs_vector(s)
    assert np.isfinite(v).all()
