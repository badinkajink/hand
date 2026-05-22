"""Tests for `morphohand.rl.reference_trajectory`.

Uses the real on-disk `best_rollout.npz` from `run18_final/foundational/cube`
since the file is small and deterministic. Skipped if it isn't present
(e.g. fresh clone before any CEM run).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/phase1/run18_final/foundational/cube/run_20260521_161817"
NPZ = RUN / "best_rollout.npz"
SUMMARY = RUN / "summary.json"


@pytest.fixture(scope="module")
def fixture_model():
    pytest.importorskip("mujoco")
    if not (RUN / "frozen_scene.xml").exists():
        pytest.skip("frozen_scene.xml not present under run dir")
    import mujoco
    return mujoco.MjModel.from_xml_path(str(RUN / "frozen_scene.xml"))


@pytest.fixture(scope="module")
def trajectory(fixture_model):
    if not NPZ.exists() or not SUMMARY.exists():
        pytest.skip("reference rollout not on disk")
    from morphohand.rl.reference_trajectory import ReferenceTrajectory
    return ReferenceTrajectory.from_run_dir(RUN, fixture_model, dt=0.002)


def test_qpos_shape_matches_model(trajectory, fixture_model):
    assert trajectory.qpos.ndim == 2
    assert trajectory.qpos.shape[1] == fixture_model.nq
    assert trajectory.qvel.shape[1] == fixture_model.nv


def test_n_steps_and_duration(trajectory):
    assert trajectory.n_steps == 600
    assert abs(trajectory.duration - 1.2) < 1e-9


def test_finger_ctrl_shape(trajectory):
    assert trajectory.finger_ctrl.shape == (9,)


def test_palm_pose_at_zero_matches_qpos_row(trajectory):
    """palm_qpos_at(0) must equal qpos[0, palm_ids] exactly (no interp)."""
    palm_ids = list(trajectory.layout.palm)
    expected = trajectory.qpos[0, palm_ids]
    got = trajectory.palm_qpos_at(0.0)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_finger_qpos_at_zero_matches_qpos_row(trajectory):
    finger_ids = list(trajectory.layout.fingers)
    expected = trajectory.qpos[0, finger_ids]
    got = trajectory.finger_qpos_at(0.0)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_object_quat_unit_norm_at_intermediate_t(trajectory):
    for t in (0.0, 0.0011, 0.5, 1.199):
        q = trajectory.object_quat_at(t)
        assert q.shape == (4,)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6


def test_object_pos_lerp_midpoint(trajectory):
    """object_pos_at midpoint between two steps = average of the two rows."""
    # midpoint between step 100 and 101 -> t = 100.5 * dt
    t = 100.5 * trajectory.dt
    got = trajectory.object_pos_at(t)
    expected = 0.5 * (trajectory.qpos[100, 0:3] + trajectory.qpos[101, 0:3])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_out_of_range_clamps(trajectory):
    """t outside [0, duration] clamps to endpoint, never raises."""
    a = trajectory.object_pos_at(-1.0)
    b = trajectory.object_pos_at(trajectory.duration + 5.0)
    np.testing.assert_allclose(a, trajectory.qpos[0, 0:3])
    np.testing.assert_allclose(b, trajectory.qpos[-1, 0:3])


def test_batch_lookup_shapes(trajectory):
    ts = [0.0, 0.5, 1.0]
    out = trajectory.batch_at(ts)
    assert out["object_pos"].shape == (3, 3)
    assert out["object_quat"].shape == (3, 4)
    assert out["palm_qpos"].shape == (3, 6)
    assert out["finger_qpos"].shape == (3, 9)
    assert out["finger_ctrl"].shape == (3, 9)
