"""Pins for scripts/real_v1_d1_continuation.py, the module shared by the DiffMJX and contact-gait scripts.

The plan schedule, the score and the held criterion decide every number of the two result pages of 2026-10-07 and of
the audit of 2026-10-11; a silent change to any of them moves those numbers without an error. The scene test builds
the single-sphere D1 continuation in a temporary folder and checks the collision masks the module promises.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import real_v1_d1_continuation as d1  # noqa: E402

YAW = np.array([0, 3, 6])
FLEX = {"index": np.array([4, 5]), "middle": np.array([7, 8])}
CTRL0 = np.linspace(0.1, 0.9, 9)
LOW = np.full(9, -1.0)
HIGH = np.full(9, 1.0)


def test_vertical_cos_from_quaternion():
    assert d1.vertical_cos(np.array([1.0, 0, 0, 0])) == pytest.approx(1.0)
    h = math.sqrt(0.5)
    assert d1.vertical_cos(np.array([h, h, 0, 0])) == pytest.approx(0.0, abs=1e-12)    # 90 deg about x
    assert d1.vertical_cos(np.array([0.0, 1.0, 0, 0])) == pytest.approx(-1.0)          # upside down
    assert d1.vertical_cos(np.array([h, 0, 0, h])) == pytest.approx(1.0)               # yaw keeps the axis vertical


def test_hold_vector_and_forced_release():
    hold = d1.hold_vector()
    assert hold.shape == (d1.N_PARAM,) and np.all(hold[9:11] == -1) and not hold[[*range(9), 11, 12]].any()
    assert d1.hold_vector(True)[9] == 1.0
    x = d1.hold_vector()
    y = d1.apply_forced_release(x, True)
    assert y[9] == 1.0 and x[9] == -1.0                                                 # a copy, not in place


def test_hold_plan_keeps_the_grip_controls():
    for k in d1.CONTROL_CHANGE_MS:
        np.testing.assert_array_equal(d1.plan_ctrl(d1.hold_vector(), k, CTRL0, LOW, HIGH, YAW, FLEX), CTRL0)


def test_plan_phases_opening_and_closing():
    x = np.zeros(d1.N_PARAM)
    x[0:9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    x[9], x[10], x[11], x[12] = 1.0, -1.0, 0.5, -0.5
    for k, phase in ((0, 0), (120, 1), (180, 1), (240, 2)):
        c = d1.plan_ctrl(x, k, CTRL0, LOW, HIGH, YAW, FLEX)
        np.testing.assert_allclose(c[YAW], np.clip(CTRL0[YAW] + d1.YAW_SCALE_RAD * x[3 * phase:3 * phase + 3],
                                                   LOW[YAW], HIGH[YAW]))
    early, late = (d1.plan_ctrl(x, k, CTRL0, LOW, HIGH, YAW, FLEX) for k in (120, 180))
    np.testing.assert_allclose(early[FLEX["index"]], LOW[FLEX["index"]])                 # fully open until 180 ms
    np.testing.assert_allclose(early[FLEX["middle"]], CTRL0[FLEX["middle"]])            # amplitude 0 at -1
    np.testing.assert_allclose(late[FLEX["index"]], CTRL0[FLEX["index"]] + d1.CLOSE_SCALE_RAD * 0.5)
    np.testing.assert_allclose(late[FLEX["middle"]], CTRL0[FLEX["middle"]] - d1.CLOSE_SCALE_RAD * 0.5)
    assert d1.plan_ctrl(np.ones(d1.N_PARAM) * 1.0, 0, np.full(9, 0.95), LOW, HIGH, YAW, FLEX)[0] == 1.0  # clipped


def test_control_changes_follow_the_phases():
    changes = [k for k in range(d1.HORIZON_MS)
               if k == 0 or d1.plan_phase(k) != d1.plan_phase(k - 1) or k == d1.OPEN_UNTIL_MS]
    assert tuple(changes) == d1.CONTROL_CHANGE_MS


def test_score_and_held_criterion():
    forces = {"thumb": 1.0, "index": 1.0, "middle": 1.0}
    row = d1.score_and_held(0.1, 0.1, 0.3, 0.1, forces)
    assert row["held_final"] and row["score"] == pytest.approx(0.2) and row["cos_gain"] == pytest.approx(0.2)
    descent = d1.score_and_held(0.1, 0.1, 0.3, 0.1 - 0.005, forces)                   # 5 mm: 2 mm beyond the free 3
    assert descent["held_final"] and descent["score"] == pytest.approx(0.2 - 70 * 0.002)
    weak = d1.score_and_held(0.1, 0.1, 0.3, 0.1, {**forces, "index": 0.3})
    assert not weak["held_final"] and weak["score"] == pytest.approx(0.2 - 0.7 * 0.2)
    dropped = d1.score_and_held(0.1, 0.1, 0.9, 0.1 - 0.009, forces)
    assert not dropped["held_final"]


def test_release_statistics():
    trace = [{"finger_normal_force_N": {"thumb": 1.0, "index": f, "middle": 1.0}}
             for f in (1.0, 0.01, 0.02, 0.0, 0.8, 0.9)]
    out = d1.release_stats(trace, {"thumb": 1.0, "index": 0.9, "middle": 1.0})
    assert out["index_release_ms"] == 30 and out["index_recontact"]
    assert out["middle_release_ms"] == 0 and not out["middle_recontact"]


def test_central_difference_is_exact_on_a_quadratic():
    f = lambda u: np.array([u[0] ** 2 + 3 * u[1], u[0] * u[1]])  # noqa: E731
    jac = d1.central_difference(f, np.array([1.0, 2.0]), np.eye(2), 1e-3)
    np.testing.assert_allclose(jac, [[2.0, 3.0], [2.0, 1.0]], rtol=1e-9)


def test_sphere_scene_masks_and_held_state(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    scene, meta, model, held = d1.held_scene("sphere", "pt", seed=1, out_dir=tmp_path)
    plate = d1.palm_plate_geom(model)
    assert model.geom_contype[plate] == 0 and model.geom_conaffinity[plate] == 0
    post = model.body("tool_platform").id
    assert all(model.geom_contype[g] == 0 for g in range(model.ngeom) if model.geom_bodyid[g] == post)
    summary = d1.contact_summary(model, held)
    assert summary["palm_plate_contacts"] == 0
    assert min(summary["finger_normal_force_N"].values()) > 0.1                       # every finger grips
    qadr, vadr = d1.tool_addresses(model)
    assert abs(held.qpos[qadr + 2] - meta["tool7"][2]) < 0.005
    m0 = d1.cpu_metrics(model, held, 0)
    assert m0[0] == pytest.approx(d1.vertical_cos(held.qpos[qadr + 3:qadr + 7]))
    assert model.opt.timestep == 0.001 and model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
