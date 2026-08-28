"""Pins for the `real_v1` (CAD-matched hardware) topology.

Every number here was measured off the physical hand and handed over in
`assets/mjcf/real_v1/real_finger_kinematics.pdf`. The point of the file is that a later edit to
the scene, the builder or the shared generator cannot quietly move the model away from the
hardware — a run against slightly-wrong geometry produces a number that looks real.

Four classes of defect these cover, all of which have actually happened in this repo:
  * the yaw link collapsing onto the yaw AXIS, so yaw and MCP become coincident joints;
  * a fingertip body with no geom, which silently zeroes every contact-based RL reward;
  * by-construction link overlap showing up as hundreds of newtons of self-collision;
  * the generator's fixed [x, y, yaw, mcp, len, pip] qpos block drifting out of sync.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(ROOT / "scripts"))
import build_real_v1_scenes as B  # noqa: E402

HAND = ROOT / "assets/mjcf/real_v1/real_hand.xml"
SCENE = ROOT / "assets/mjcf/real_v1/scenes/scene_screwdriver_medium.xml"
ACTUATED = ROOT / "assets/mjcf/real_v1/real_hand_morphology_actuated.xml"
FINGERS = ("thumb", "index", "middle")

MM = 1e-3


@pytest.fixture(scope="module")
def scene_model():
    return mujoco.MjModel.from_xml_path(str(SCENE))


def _fwd(m, keyframe: str):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(keyframe).id)
    mujoco.mj_forward(m, d)
    return d


def test_files_match_the_builder():
    """The three MJCFs are generated; a hand edit to one of them would be silently overwritten."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts/build_real_v1_scenes.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_cad_link_lengths(scene_model):
    """Kinematic joint spacing is the CAD length MINUS the 12.70 mm overhang, and the visible
    link still ends at the CAD length. Conflating the two is what made every published finger
    length in this program name a parameter rather than a link."""
    assert B.JOINT_SPACING == pytest.approx(20.75 * MM)
    assert B.LINK_YAW - B.OVERHANG == pytest.approx(B.JOINT_SPACING)

    m = scene_model
    d = _fwd(m, "open")
    for f in FINGERS:
        yaw = d.body(f"{f}_yaw_frame").xpos
        mcp = d.body(f"{f}_mcp_frame").xpos
        pip = d.body(f"{f}_pip_frame").xpos
        tip = d.body(f"{f}_tip").xpos
        assert np.linalg.norm(mcp - yaw) == pytest.approx(B.JOINT_SPACING, abs=1e-9)
        assert np.linalg.norm(pip - mcp) == pytest.approx(B.JOINT_SPACING, abs=1e-9)
        # PIP -> pad SURFACE is the CAD distal length; the tip body is the pad CENTRE.
        assert np.linalg.norm(tip - pip) + B.R_PHALANGE == pytest.approx(37.16 * MM, abs=1e-9)


def test_yaw_link_is_perpendicular_to_the_yaw_axis(scene_model):
    """The 32-ish mm yaw link must OFFSET the MCP joint from the yaw axis.

    Built along the axis it lies ON it: yaw cannot move the MCP joint and the two joints are
    coincident in everything but their body origins. Zero here is that bug. Body origins
    differing is NOT sufficient evidence — that was the trap in the 2026-08-27 length study.
    """
    m = scene_model
    d = _fwd(m, "open")
    for f in FINGERS:
        jid = m.joint(f"{f}_yaw").id
        axis = d.xaxis[jid] / np.linalg.norm(d.xaxis[jid])
        anchor = d.xanchor[jid]
        v = d.body(f"{f}_mcp_frame").xpos - anchor
        perp = np.linalg.norm(v - np.dot(v, axis) * axis)
        assert perp == pytest.approx(B.JOINT_SPACING, abs=1e-6), f"{f}: yaw/MCP coincident"


def test_rom_matches_cad(scene_model):
    m = scene_model
    for f in FINGERS:
        lo, hi = m.jnt_range[m.joint(f"{f}_yaw").id]
        assert np.rad2deg(lo) == pytest.approx(-85.0, abs=0.01)
        assert np.rad2deg(hi) == pytest.approx(85.0, abs=0.01)
        lo, hi = m.jnt_range[m.joint(f"{f}_mcp").id]
        assert np.rad2deg(lo) == pytest.approx(-15.0, abs=0.01)
        assert np.rad2deg(hi) == pytest.approx(92.0, abs=0.01)
        lo, hi = m.jnt_range[m.joint(f"{f}_pip").id]
        assert np.rad2deg(lo) == pytest.approx(-18.0, abs=0.01)
        assert np.rad2deg(hi) == pytest.approx(92.0, abs=0.01)


def test_thumb_flexion_opposes_the_pair(scene_model):
    """Positive flexion has to close all three fingers toward the shaft, which means the thumb's
    hinge axis is mirrored. Get this wrong and the thumb opens as the pair closes."""
    m = scene_model
    assert list(m.jnt_axis[m.joint("thumb_mcp").id]) == [0, -1, 0]
    assert list(m.jnt_axis[m.joint("index_mcp").id]) == [0, 1, 0]
    assert list(m.jnt_axis[m.joint("middle_mcp").id]) == [0, 1, 0]


def test_workspace_matches_the_cad_registration(scene_model):
    """Mount positions and gantry travel, against XY_space.png: thumb 60x110 mm, index/middle
    60x60 mm, 40 mm between the thumb band and the pair, 50 mm between index and middle."""
    m = scene_model
    pos = {f: m.body(f"{f}_mount").pos for f in FINGERS}
    assert pos["thumb"][:2] == pytest.approx([-50 * MM, 0.0])
    assert pos["index"][:2] == pytest.approx([50 * MM, 55 * MM])
    assert pos["middle"][:2] == pytest.approx([50 * MM, -55 * MM])

    box = {}
    for f in FINGERS:
        for ax in ("x", "y"):
            lo, hi = m.jnt_range[m.joint(f"{f}_{ax}").id]
            box[(f, ax)] = (pos[f][0 if ax == "x" else 1] + lo,
                            pos[f][0 if ax == "x" else 1] + hi)
    # thumb band vs pair band, along X
    assert box[("index", "x")][0] - box[("thumb", "x")][1] == pytest.approx(40 * MM)
    # index vs middle, along Y
    assert box[("index", "y")][0] - box[("middle", "y")][1] == pytest.approx(50 * MM)
    assert box[("thumb", "y")][1] - box[("thumb", "y")][0] == pytest.approx(110 * MM)
    assert box[("index", "x")][1] - box[("index", "x")][0] == pytest.approx(60 * MM)


def test_len_slides_are_frozen(scene_model):
    """The hardware has no proximal-length DoF. The `_len` slides exist only so the shared
    generator finds its [x, y, yaw, mcp, len, pip] block; zero travel is what makes them inert."""
    m = scene_model
    for f in FINGERS:
        lo, hi = m.jnt_range[m.joint(f"{f}_len").id]
        assert (lo, hi) == (0.0, 0.0)

    from morphohand.sampling.morphology import REAL_V1_WORKSPACE
    assert REAL_V1_WORKSPACE.len_min == REAL_V1_WORKSPACE.len_max == 0.0


def test_fingertip_bodies_carry_a_geom(scene_model):
    """`_build_sensors` matches contacts by BODY on ("thumb_tip","index_tip","middle_tip").

    A tip body with no geom can never appear in a contact, so the fingertip-contact reward, the
    tip-loss termination and the grip-force scorecard would all read zero for a whole training
    run and look exactly like a policy that never touched the object.
    """
    m = scene_model
    for f in FINGERS:
        bid = m.body(f"{f}_tip").id
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
        assert geoms, f"{f}_tip has no geom — every contact reward would read zero"
        assert m.geom_type[geoms[0]] == mujoco.mjtGeom.mjGEOM_SPHERE
        assert m.geom_size[geoms[0], 0] == pytest.approx(B.R_PHALANGE)


@pytest.mark.parametrize("keyframe", ["open", "open_ik"])
def test_no_self_collision_at_rest(scene_model, keyframe):
    """Only the pads may touch the shaft, and nothing may touch anything else.

    The links genuinely overlap (12.70 mm of it) and the yaw capsule's cap sits inside the palm
    plate, so without the <contact><exclude> block the shipped hand reports 223 N palm
    interpenetration and 38 N per finger before a policy has done anything.
    """
    m = scene_model
    d = _fwd(m, keyframe)
    offenders = []
    for c in d.contact[: d.ncon]:
        b1 = m.body(m.geom_bodyid[c.geom1]).name
        b2 = m.body(m.geom_bodyid[c.geom2]).name
        pair = {b1, b2}
        if "screwdriver_medium" in pair or "world" in pair:
            continue
        if c.dist < -1e-4:
            offenders.append((b1, b2, float(c.dist)))
    assert not offenders, offenders


def test_hand_and_scene_agree_on_the_finger(scene_model):
    """The generator needs a hand/scene PAIR. If they drift apart, a design generated for one is
    evaluated on the other (`tests/test_perp_hand_scene_parity.py` pins the same thing for perp)."""
    mh = mujoco.MjModel.from_xml_path(str(HAND))
    ms = scene_model
    for f in FINGERS:
        for part in ("mount", "yaw_frame", "mcp_frame", "len_frame", "pip_frame", "tip"):
            assert mh.body(f"{f}_{part}").pos == pytest.approx(ms.body(f"{f}_{part}").pos)
        for j in ("x", "y", "len", "yaw", "mcp", "pip"):
            assert mh.jnt_range[mh.joint(f"{f}_{j}").id] == pytest.approx(
                ms.jnt_range[ms.joint(f"{f}_{j}").id])


def test_actuated_variant_drives_the_gantries():
    """The explorer file is the only one whose morph slides are actuated; the base pair's are
    unactuated on purpose (they must not be driven, and they must not be optimised over)."""
    ma = mujoco.MjModel.from_xml_path(str(ACTUATED))
    names = {ma.actuator(i).name for i in range(ma.nu)}
    assert {"a_thumb_x", "a_index_y", "a_middle_x"} <= names
    ms = mujoco.MjModel.from_xml_path(str(SCENE))
    assert not any(ms.actuator(i).name.endswith(("_x", "_y", "_len")) for i in range(ms.nu))


def test_generator_bakes_a_design_and_strips_the_morph_joints(tmp_path):
    """The end-to-end contract the whole pipeline rests on: a 6-dim XY design goes in, a rigid
    scene with 22 qpos and baked mounts comes out, and the <contact> excludes survive."""
    from morphohand.sampling.morphology import real_v1_compact_design
    from morphohand.tools.morphology_xml import create_rigid_hand_and_scene_xmls

    design = real_v1_compact_design(1.0, 1.0, 1.0)
    _, scene = create_rigid_hand_and_scene_xmls(HAND, SCENE, design, tmp_path)
    m = mujoco.MjModel.from_xml_path(str(scene))

    assert m.nq == 22, "13 (object freejoint + palm pose) + 9 articulation"
    assert not [m.joint(i).name for i in range(m.njnt)
                if m.joint(i).name.endswith(("_x", "_y", "_len"))]
    assert m.body("thumb_mount").pos[0] == pytest.approx(-50 * MM + design.thumb_x)
    assert m.body("index_mount").pos[1] == pytest.approx(55 * MM + design.index_y)
    assert m.nexclude == mujoco.MjModel.from_xml_path(str(SCENE)).nexclude


# --- the reorient mechanism (2026-08-28) --------------------------------------------------
#
# The four trained real_v1 designs held the shaft perfectly and turned it by 0.9-4 degrees.
# The cause is geometric, not a training accident: a fixed-contact rotation of theta about the
# pinch axis drives the DESCENDING pair contact down by straddle * sin(theta), which its finger
# pays for by extending — and `fit_real_v1_pose` picks the deepest reachable palm, so every
# design grasps within 1.3-9.6 mm of full extension. These pin the two things that fixed it.


def test_axial_offset_slides_the_whole_contact_ring():
    """`--axial-offset` moves the grip along the shaft without changing the straddle.

    An off-centre grip is what buys back the palm clearance that a shallower, less-extended
    grasp would cost: the stub above the grip is (half_len - offset), not half_len.
    """
    import fit_real_v1_pose as F

    centre = np.zeros(3)
    a = F.tip_targets(centre, 0.0125, 0.001, 0.030, 0.0, 0.0)
    b = F.tip_targets(centre, 0.0125, 0.001, 0.030, 0.0, 0.025)
    for f in FINGERS:
        assert b[f][1] - a[f][1] == pytest.approx(0.025)
        assert b[f][0] == pytest.approx(a[f][0])
        assert b[f][2] == pytest.approx(a[f][2])
    assert b["index"][1] - b["middle"][1] == pytest.approx(0.060)


def test_finger_radial_workspace_is_a_thin_shell(scene_model):
    """The pad's distance from its mount is confined to a band, and the TOP of that band is
    what a fixed-contact rotation spends. 68.11 mm is the straight-chain reach: yaw link
    20.75 + mcp-to-pip 20.75 + pip-to-pad 26.61."""
    d = mujoco.MjData(scene_model)
    mujoco.mj_forward(scene_model, d)
    reach = (B.JOINT_SPACING + B.JOINT_SPACING + B.PAD_CENTER)
    assert reach == pytest.approx(0.06811, abs=1e-5)
    for f in FINGERS:
        root = d.body(f"{f}_yaw_frame").xpos
        tip = d.body(f"{f}_tip").xpos
        assert np.linalg.norm(tip - root) == pytest.approx(reach, abs=1e-4)


def test_hold_anchor_can_open_on_a_schedule():
    """`hold_switch_from_sim_step` exists and defaults off.

    The alignment gate alone is circular on this hand: the anchor only moves once the policy
    rotates the shaft, and the grasp is a rotational lock, so it never opens. The step gate
    turns the reorient into a residual problem instead of an exploration one."""
    from morphohand.rl.actions import LerpFingerActionCfg
    from morphohand.rl.env_cfg import MorphoHandEnvCfg

    assert LerpFingerActionCfg.hold_switch_from_sim_step == 0
    assert MorphoHandEnvCfg.hold_switch_from_sim_step == 0


def test_target_axis_rewards_can_be_gated_on_height():
    """`target_axis_min_lift` exists and is off by default.

    `target_axis_alignment` is weight 100 and saturates near 1; the whole `lift_height` reward is
    80 * clip(z - spawn, 0, 0.10) and tops out at 8. So putting the object DOWN costs at most 8
    per step and buys up to 98, and once the scheduled anchor made a large rotation reachable the
    policy took that trade: alignment 0.64 -> 19.9 while episode-mean object height fell
    0.114 -> 0.061 and the handoff ended with the shaft on the table."""
    from morphohand.rl.env_cfg import MorphoHandEnvCfg
    from morphohand.rl.terms_reward import _lift_gate

    assert MorphoHandEnvCfg.target_axis_min_lift == 0.0
    assert callable(_lift_gate)
