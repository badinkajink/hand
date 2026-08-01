"""Unit tests for morphohand.tools.keyframe_ik (CPU-only; tiny synthetic MJCF)."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco

from morphohand.tools import keyframe_ik

TINY_SCENE = """
<mujoco model="tiny">
  <worldbody>
    <body name="b1">
      <joint name="j1" type="hinge" axis="0 0 1" range="-1 1"/>
      <geom type="sphere" size="0.01"/>
    </body>
  </worldbody>
  <actuator><position joint="j1"/></actuator>
</mujoco>
"""


def test_has_joint():
    m = mujoco.MjModel.from_xml_string(TINY_SCENE)
    assert keyframe_ik.has_joint(m, "j1")
    assert not keyframe_ik.has_joint(m, "nope")


def test_actuator_ctrl_from_qpos():
    m = mujoco.MjModel.from_xml_string(TINY_SCENE)
    d = mujoco.MjData(m)
    d.qpos[0] = 0.42
    assert keyframe_ik.actuator_ctrl_from_qpos(m, d) == [0.42]


def test_inject_keyframe_creates_and_replaces(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text(TINY_SCENE)
    keyframe_ik.inject_keyframe(scene, "open_ik", "0.1", "0.1")
    keyframe_ik.inject_keyframe(scene, "open_ik", "0.2", "0.2")  # replace, not duplicate
    keys = ET.parse(scene).getroot().findall("keyframe/key")
    assert [(k.get("name"), k.get("qpos")) for k in keys] == [("open_ik", "0.2")]
    # the scene must still load with the injected keyframe
    m = mujoco.MjModel.from_xml_path(str(scene))
    assert m.key("open_ik").id >= 0


# --- guard: an IK-retargeted keyframe must never be silently discarded ---------------
# Regression for the 2026-08-01 defect that wiped out two 5-design queues: the launcher
# retargeted `open_ik` into every scene and the env then ignored it, because
# open_finger_from_keyframe defaults to False. Nothing errored; it surfaced as
# "ungraspable morphologies" and "unstable scene". See configs/recipes/perp_single.yaml.

import pytest

from morphohand.rl.env_build import _assert_keyframe_not_silently_discarded


class _Cfg:
    def __init__(self, keyframe_name, open_finger_from_keyframe, open_finger_qpos):
        self.keyframe_name = keyframe_name
        self.open_finger_from_keyframe = open_finger_from_keyframe
        self.open_finger_qpos = open_finger_qpos


PERP_OPEN_IK = (0.0, 1.9086, -1.8, 0.0, 1.1382, 0.7285, 0.0, 1.1382, 0.7285)
BASELINE_OPEN = (0.0, 3.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_raises_when_ik_keyframe_would_be_discarded():
    cfg = _Cfg("open_ik", False, BASELINE_OPEN)
    with pytest.raises(ValueError, match="would be DISCARDED"):
        _assert_keyframe_not_silently_discarded(cfg, {"finger_joint_pos": PERP_OPEN_IK})


def test_silent_when_flag_is_set():
    cfg = _Cfg("open_ik", True, BASELINE_OPEN)
    _assert_keyframe_not_silently_discarded(cfg, {"finger_joint_pos": PERP_OPEN_IK})


def test_silent_for_non_ik_keyframe():
    """Baseline-hand runs legitimately use the authored open pose."""
    cfg = _Cfg("open", False, BASELINE_OPEN)
    _assert_keyframe_not_silently_discarded(cfg, {"finger_joint_pos": PERP_OPEN_IK})


def test_silent_when_poses_agree():
    """Zero morphology change: nothing is being discarded, so do not cry wolf."""
    cfg = _Cfg("open_ik", False, PERP_OPEN_IK)
    _assert_keyframe_not_silently_discarded(cfg, {"finger_joint_pos": PERP_OPEN_IK})
