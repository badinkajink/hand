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
