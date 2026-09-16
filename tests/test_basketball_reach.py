"""Pins for the basketball reach map in scripts/real_v1_basketball_grasp.py.

The map is a closed-form forward kinematics of the real_v1 finger evaluated on a joint grid;
if it drifts from the MJCF the fitter reports contacts the finger cannot make.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import real_v1_basketball_grasp as G  # noqa: E402

SCENE = ROOT / "assets/mjcf/real_v1/scenes/scene_basketball.xml"


@pytest.mark.parametrize("finger", ["thumb", "index", "middle"])
def test_finger_fk_matches_mujoco(finger):
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    for _ in range(20):
        yaw = rng.uniform(*G.YAW_RANGE)
        mcp = rng.uniform(*G.MCP_RANGE)
        pip = rng.uniform(*G.PIP_RANGE)
        d.qpos[:] = 0
        d.qpos[:7] = [0, 0, 0.12, 1, 0, 0, 0]
        for j, v in zip(G.FINGERS[finger], (yaw, mcp, pip)):
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
        mujoco.mj_forward(m, d)
        pad, segs = G.finger_fk(finger, np.array(yaw), np.array(mcp), np.array(pip))
        mount = d.body(f"{finger}_mount").xpos
        assert np.allclose(pad + mount, d.body(G.TIPS[finger]).xpos, atol=1e-9)
        assert np.allclose(segs[1][0] + mount, d.body(f"{finger}_mcp_frame").xpos, atol=1e-9)
        assert np.allclose(segs[2][0] + mount, d.body(f"{finger}_pip_frame").xpos, atol=1e-9)


def test_reach_map_finds_the_straight_finger():
    """A pad hanging straight down from a mount 78.66 mm above a sphere's surface touches it."""
    R = 0.12
    centre = np.array([0.0, 0.0, R])
    reach = 2 * G.JOINT_SPACING + G.PAD_CENTRE
    # mount directly above the ball at height so the straight finger's pad touches the pole
    mounts = {f: np.array([0.0, 0.0, 2 * R + G.PAD_RADIUS + reach]) for f in G.FINGER_NAMES}
    rm = G.reach_map(mounts, centre, R)
    for f in G.FINGER_NAMES:
        assert rm[f]["theta"].size > 0
        assert rm[f]["theta"].min() < 3.0


def test_choose_contacts_is_symmetric_about_the_thumb():
    rm = {f: {"theta": np.array([60.0, 60.0, 60.0]), "az": np.array(a),
              "yaw": np.zeros(3), "mcp": np.zeros(3), "pip": np.zeros(3)}
          for f, a in (("thumb", [180.0, 170.0, 100.0]), ("index", [60.0, 20.0, 85.0]),
                       ("middle", [-60.0, -20.0, -85.0]))}
    sep, pose = G.choose_contacts(rm, 60.0)
    assert sep == pytest.approx(120.0)
    assert pose["thumb"][3] == 180.0
    assert pose["index"][3] == 60.0 and pose["middle"][3] == -60.0
    # a thumb 40 deg off its meridian is not a candidate even if it is the only one at theta
    rm["thumb"]["az"] = np.array([140.0, 130.0, 100.0])
    assert G.choose_contacts(rm, 60.0) is None
