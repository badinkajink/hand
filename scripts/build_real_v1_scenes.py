"""Build the `real_v1` base topology from the CAD-measured finger geometry.

`assets/mjcf/real_v1/` arrived as two hand-authored files: a 15-DoF morphology-ACTUATED scene
carrying the CAD finger and the CAD XY workspace, and one frozen example design. Neither is a
generation source — the pipeline needs the pair described in the `morphology-scenes` skill
(hand + scene, morph joints present but UNACTUATED) — and both carry three defects that are
invisible in a metric and fatal in a rollout. This script emits the base pair (plus a matching
actuated explorer) from one spec, so the CAD numbers live in exactly one place.

WHAT THE CAD GIVES US (do not "improve" these — they are measurements)

    phalange half-width                 r = 10.55 mm
    yaw / middle link, axis to surface      33.45 mm
    distal link, axis to pad surface        37.16 mm
    parent link overhang past child axis    12.70 mm  =>  joint spacing 33.45 - 12.70 = 20.75 mm
    yaw ROM   +/- 85 deg      (0 = straight, + = abduction)
    MCP ROM   -15 .. +92 deg  (0 = straight, + = flexion)
    PIP ROM   -18 .. +92 deg
    mounts (palm frame)  thumb (-50, 0)  index (+50, +55)  middle (+50, -55) mm
    workspaces           thumb 60 x 110  index/middle 60 x 60 mm   (assets/.../XY_space.png)

WHAT THIS SCRIPT FIXES, AND WHY EACH ONE MATTERS

1.  `<f>_tip` HAS NO GEOM.  The RL fingertip contact sensor is `ContactMatch(mode="body",
    pattern=("thumb_tip","index_tip","middle_tip"))` (rl/env_build.py `_build_sensors`). A tip
    body with no geom can never register a contact, so every fingertip-contact reward, the
    tip-loss termination and the grip-force scorecard would read identically zero all training
    and the run would look like a policy failure. We move the distal capsule's terminal cap into
    the tip body as a sphere of the same radius at the same place: the collision envelope still
    ends 37.16 mm from the PIP axis, but the pad is now its own body. This is exactly the
    baseline hand's convention (capsule on the link, sphere on `<f>_tip`).

2.  THE FINGER INTERPENETRATES THE PALM AND ITSELF.  At the home pose the shipped scene reports
    223 N between `thumb_yaw_frame` and `palm_pose`, and 38 N between each `<f>_yaw_frame` and
    its own `<f>_pip_frame`. Both are consequences of correct geometry, not bugs in it: the yaw
    capsule's cap sits at z=0, which is inside the palm plate, and the 12.70 mm overhang means
    the yaw and distal links genuinely overlap in space. MuJoCo only auto-excludes direct
    parent/child pairs, so grandparent pairs collide. 21 `<contact><exclude>` entries, mirroring
    what `assets/mjcf/perp/perp_hand.xml` already carries for the same reason.

3.  THE PALM IS AT THE OLD HAND'S HEIGHT.  0.134 m was fitted to a 117 mm finger. This finger
    reaches 78.66 mm below its mount, so at 0.134 the fingertips stop 30 mm above a shaft lying
    on the table and no grasp exists at all. `PALM_Z` below is solved for this finger (see
    `--fit-palm-z`), which is legitimate: palm height is the arm's pose, not a hardware
    parameter. Everything the CAD does fix -- link lengths, ROM, mounts, workspace -- is taken
    from the user's model verbatim, per LINK_LENGTH_GATE §6.

THE `len` SHIM.  This hand has no proximal-length DoF: the links are CAD parts. But the shared
generator (`morphohand.tools.morphology_xml`) indexes a fixed [x, y, yaw, mcp, len, pip] block
per finger and looks up `<f>_len_frame`, and so do the sweep, the IK retarget and the keyframe
parsers. Rather than fork all of that for one topology, each finger carries a zero-length
`<f>_len_frame` and a `<f>_len` slide pinned to range="0 0". The design space is the 6 XY dims;
`REAL_V1_WORKSPACE` sets len_min = len_max = 0 so no sampler can move it.

Usage:
    uv run python scripts/build_real_v1_scenes.py            # write the three files
    uv run python scripts/build_real_v1_scenes.py --check    # verify without writing
    MUJOCO_GL=egl uv run python scripts/build_real_v1_scenes.py --fit-palm-z
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "mjcf" / "real_v1"

# --- CAD measurements (metres / radians) ---------------------------------------------------
R_PHALANGE = 0.010550          # axis to outer surface
LINK_YAW = 0.033450            # yaw axis -> yaw link surface
LINK_MIDDLE = 0.033450         # MCP axis -> middle link surface
LINK_DISTAL = 0.037160         # PIP axis -> pad surface
OVERHANG = 0.012700            # link surface past the NEXT joint axis
JOINT_SPACING = LINK_YAW - OVERHANG        # 0.020750, yaw->MCP and MCP->PIP
CAP_YAW = LINK_YAW - R_PHALANGE            # 0.022900, capsule medial axis
CAP_MIDDLE = LINK_MIDDLE - R_PHALANGE      # 0.022900
CAP_DISTAL = LINK_DISTAL - 2 * R_PHALANGE  # 0.016060, stops short so the PAD is the tip sphere
PAD_CENTER = LINK_DISTAL - R_PHALANGE      # 0.026610, sphere centre == old capsule end

YAW_RANGE = 1.483530           # +/- 85 deg
MCP_RANGE = (-0.261799, 1.605703)   # -15 .. +92 deg
PIP_RANGE = (-0.314159, 1.605703)   # -18 .. +92 deg

# --- workspace registration (palm frame), from assets/mjcf/real_v1/XY_space.png ------------
# CAD envelope x=[-30,140] y=[0,160] mm, bbox centre (55,80) taken as the palm origin, rotated
# into MuJoCo as x_MJ = -(y_CAD - 80), y_MJ = -(x_CAD - 55).
MOUNTS = {
    "thumb": (-0.050, 0.000),
    "index": (0.050, 0.055),
    "middle": (0.050, -0.055),
}
WORKSPACE = {                  # half-extents of each mount's rectangle
    "thumb": (0.030, 0.055),
    "index": (0.030, 0.030),
    "middle": (0.030, 0.030),
}

# Palm plate: half-extents chosen to cover the union of the three workspaces (x +/-80,
# y +/-85 mm) with a 5 mm lip, so a mount is never hanging off the plate it is bolted to.
PALM_HALF = (0.085, 0.090, 0.0015)

# Palm height above the table, solved by --fit-palm-z for the WIDEST design in the workspace
# (all six morph slides at 0: thumb at x=-50, pair at x=+50, 100 mm apart). At 0.0625 the tripod
# grasp on the shaft's mid-height works each finger at 0.80-0.86 of its flexing reach, which
# keeps MCP/PIP inside their ROM with squeeze authority in both directions; any higher and the
# spread-out design cannot reach the shaft at all (utilisation > 1 by 0.0775).
#
# This is the BASE height only. Grip depth below the mounting plane is 50.0 mm here against a
# 50 mm shaft half-length, i.e. the LINK_LENGTH_GATE clearance ceiling on held-cos is exactly
# 1.0 with ZERO headroom -- a vertical shaft's top end just grazes the palm. Compact designs
# buy that headroom back (a fully-inboard layout reaches 62-66 mm of depth), which is why every
# generated design gets its own palm pose from `scripts/fit_real_v1_pose.py` rather than
# inheriting this one. Dropping a short-fingered hand at another hand's palm height is trap #1
# in docs/experiments/8-27_LINK_LENGTH_GATE.
PALM_Z = 0.0625

# Thumb flexion axes are mirrored so that positive flexion opposes index/middle.
FLEX_AXIS = {"thumb": "0 -1 0", "index": "0 1 0", "middle": "0 1 0"}

FINGERS = ("thumb", "index", "middle")


def _finger_body(f: str, morph_joints: bool, indent: str) -> str:
    mx, my = MOUNTS[f]
    hx, hy = WORKSPACE[f]
    ax = FLEX_AXIS[f]
    i = indent
    morph = ""
    if morph_joints:
        morph = (
            f'{i}  <joint class="morph" name="{f}_x" axis="1 0 0" range="{-hx:.6f} {hx:.6f}" />\n'
            f'{i}  <joint class="morph" name="{f}_y" axis="0 1 0" range="{-hy:.6f} {hy:.6f}" />\n'
        )
    # The `len` shim (see module docstring): a slide pinned to zero travel, purely so the
    # shared generator finds the [x, y, yaw, mcp, len, pip] block it indexes by position. A
    # body carrying a joint needs mass, hence the token inertial.
    len_joint = (
        f'{i}        <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6" />\n'
        f'{i}        <joint class="morph" name="{f}_len" axis="1 0 0" range="0 0" />\n'
        if morph_joints else ""
    )
    return f"""{i}<body name="{f}_mount" pos="{mx:.6f} {my:.6f} 0">
{i}  <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6" />
{morph}{i}  <body name="{f}_yaw_frame">
{i}    <joint class="ctrl" name="{f}_yaw" axis="1 0 0" range="{-YAW_RANGE:.6f} {YAW_RANGE:.6f}" />
{i}    <geom type="capsule" fromto="0 0 0 0 0 {-CAP_YAW:.6f}" size="{R_PHALANGE:.6f}" material="finger_mat" />
{i}    <body name="{f}_mcp_frame" pos="0 0 {-JOINT_SPACING:.6f}">
{i}      <joint class="ctrl" name="{f}_mcp" axis="{ax}" range="{MCP_RANGE[0]:.6f} {MCP_RANGE[1]:.6f}" />
{i}      <geom type="capsule" fromto="0 0 0 0 0 {-CAP_MIDDLE:.6f}" size="{R_PHALANGE:.6f}" material="finger_mat" />
{i}      <body name="{f}_len_frame" pos="0 0 0">
{len_joint}{i}        <body name="{f}_pip_frame" pos="0 0 {-JOINT_SPACING:.6f}">
{i}          <joint class="ctrl" name="{f}_pip" axis="{ax}" range="{PIP_RANGE[0]:.6f} {PIP_RANGE[1]:.6f}" />
{i}          <geom type="capsule" fromto="0 0 0 0 0 {-CAP_DISTAL:.6f}" size="{R_PHALANGE:.6f}" material="finger_mat" friction="2.4 0.2 0.02" />
{i}          <body name="{f}_tip" pos="0 0 {-PAD_CENTER:.6f}">
{i}            <geom type="sphere" size="{R_PHALANGE:.6f}" material="finger_mat" friction="2.4 0.2 0.02" />
{i}          </body>
{i}        </body>
{i}      </body>
{i}    </body>
{i}  </body>
{i}</body>"""


def _excludes(indent: str) -> str:
    """The 21 pairs that are in permanent contact by construction, not by accident.

    15 x palm <-> every finger body: the yaw capsule's cap is centred ON the mounting plane, so
    it is inside the palm plate, and a flexed finger sweeps back through it.
    3 x <f>_yaw_frame <-> <f>_pip_frame: consecutive links overlap by 12.70 mm, and MuJoCo only
    filters direct parent/child pairs, so the grandparent pair collides at 38 N in every pose.
    3 x <f>_mcp_frame <-> <f>_pip_frame: the same overlap one link further down. These two ARE
    parent and child kinematically, but the zero-travel `len_frame` shim sits between them in
    the body tree, so MuJoCo's parent filter no longer sees the pair -- 21 mm of penetration
    that exists only because of the shim. Nothing here excludes a pair that could separate in
    a normal pose, so a real self-collision still registers.
    """
    lines = []
    for f in FINGERS:
        for part in ("yaw_frame", "mcp_frame", "len_frame", "pip_frame", "tip"):
            lines.append(f'{indent}<exclude body1="palm_pose" body2="{f}_{part}" />')
        lines.append(f'{indent}<exclude body1="{f}_yaw_frame" body2="{f}_pip_frame" />')
        lines.append(f'{indent}<exclude body1="{f}_mcp_frame" body2="{f}_pip_frame" />')
    return "\n".join(lines)


HEADER = """<mujoco model="{model}">
  <compiler angle="radian" autolimits="true" />
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" />
  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1" />
    <map znear=".01" />
    <quality shadowsize="2048" />
    <global elevation="-15" />
  </visual>
  <default>
    <geom friction="1.8 0.15 0.01" solref="0.006 1" solimp="0.97 0.995 0.0005" />
    <motor ctrlrange="-1 1" ctrllimited="true" />
    <default class="pose">
      <joint damping="2.0" armature="0.01" />
      <position kp="4000" kv="100" forcerange="-1000 1000" />
    </default>
    <default class="morph">
      <joint type="slide" damping="2000" armature="0.0001" />
    </default>
    <default class="ctrl">
      <joint type="hinge" damping="0.5" armature="0.001" />
      <position kp="30" kv="0.5" gear="1" forcerange="-10 10" ctrlrange="-1 1" />
    </default>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
    <texture name="grid" type="2d" builtin="checker" rgb1="0.15 0.15 0.15" rgb2="0.25 0.25 0.25" width="256" height="256" />
    <material name="table_mat" texture="grid" texrepeat="3 3" reflectance="0.2" />
    <material name="palm_mat" rgba="0.18 0.2 0.24 1" />
    <material name="finger_mat" rgba="0.8 0.72 0.6 1" />
    <material name="object_mat" rgba="0.6 0.6 0.6 1" />
  </asset>
"""

CTRL_ACTUATORS = """    <position class="ctrl" name="a_thumb_yaw" joint="thumb_yaw" ctrlrange="{yl:.6f} {yh:.6f}" />
    <position class="ctrl" name="a_thumb_mcp" joint="thumb_mcp" ctrlrange="{ml:.6f} {mh:.6f}" />
    <position class="ctrl" name="a_thumb_pip" joint="thumb_pip" ctrlrange="{pl:.6f} {ph:.6f}" />
    <position class="ctrl" name="a_index_yaw" joint="index_yaw" ctrlrange="{yl:.6f} {yh:.6f}" />
    <position class="ctrl" name="a_index_mcp" joint="index_mcp" ctrlrange="{ml:.6f} {mh:.6f}" />
    <position class="ctrl" name="a_index_pip" joint="index_pip" ctrlrange="{pl:.6f} {ph:.6f}" />
    <position class="ctrl" name="a_middle_yaw" joint="middle_yaw" ctrlrange="{yl:.6f} {yh:.6f}" />
    <position class="ctrl" name="a_middle_mcp" joint="middle_mcp" ctrlrange="{ml:.6f} {mh:.6f}" />
    <position class="ctrl" name="a_middle_pip" joint="middle_pip" ctrlrange="{pl:.6f} {ph:.6f}" />""".format(
    yl=-YAW_RANGE, yh=YAW_RANGE, ml=MCP_RANGE[0], mh=MCP_RANGE[1], pl=PIP_RANGE[0], ph=PIP_RANGE[1]
)

POSE_ACTUATORS = """    <position class="pose" name="a_palm_px" joint="palm_px" ctrlrange="-0.20 0.20" />
    <position class="pose" name="a_palm_py" joint="palm_py" ctrlrange="-0.20 0.20" />
    <position class="pose" name="a_palm_pz" joint="palm_pz" ctrlrange="-0.15 0.35" />
    <position class="pose" name="a_palm_rx" joint="palm_rx" ctrlrange="-1.57 1.57" />
    <position class="pose" name="a_palm_ry" joint="palm_ry" ctrlrange="-1.57 1.57" />
    <position class="pose" name="a_palm_rz" joint="palm_rz" ctrlrange="-3.14 3.14" />"""

MORPH_ACTUATORS = "\n".join(
    f'    <position class="morph" name="a_{f}_{ax}" joint="{f}_{ax}" '
    f'ctrlrange="{-WORKSPACE[f][k]:.6f} {WORKSPACE[f][k]:.6f}" kp="30" kv="0.5" forcerange="-10 10" />'
    for f in FINGERS
    for k, ax in enumerate(("x", "y"))
)

# The shipped `open` key is a PLACEHOLDER, not a grasp: every morph slide at 0 (each finger at
# its workspace centre) and a mild uniform flexion that keeps the fingertips clear of the table.
# Straight fingers (all zeros) would put every pad 16 mm BELOW the floor at this palm height,
# which makes the key useless as a reset state and as a freeze pose. The grasp keyframe is
# authored per design by `scripts/fit_real_v1_pose.py`, which writes `open_ik`.
OPEN_MCP = 0.55
OPEN_PIP = 0.55
WORKSPACE_SITES = """      <!-- Mount workspaces from XY_space.png, visualisation only (sites never collide). -->
      <site name="workspace_thumb" type="box" pos="-0.050 0 0.003" size="0.030 0.055 0.0005" rgba="0.85 0.30 0.30 0.14" />
      <site name="workspace_index" type="box" pos="0.050 0.055 0.003" size="0.030 0.030 0.0005" rgba="0.30 0.55 0.85 0.14" />
      <site name="workspace_middle" type="box" pos="0.050 -0.055 0.003" size="0.030 0.030 0.0005" rgba="0.30 0.75 0.45 0.14" />"""


def _keyframe(is_scene: bool) -> str:
    """A straight-finger `open` key in the layout the shared generator expects.

    Per-finger qpos block is [x, y, yaw, mcp, len, pip]; a scene prefixes the 7-value object
    freejoint and the 6 palm-pose joints. The generator strips the morph entries when it bakes
    a design, so this layout is load-bearing.
    """
    finger_q = []
    for _ in FINGERS:
        #          x    y    yaw  mcp        len  pip
        finger_q += [0.0, 0.0, 0.0, OPEN_MCP, 0.0, OPEN_PIP]
    if is_scene:
        qpos = [0, 0, 0.0125, 0.70711, 0.70711, 0, 0] + [0.0] * 6 + finger_q
        ctrl = [0.0] * 6 + [0.0, OPEN_MCP, OPEN_PIP] * 3
    else:
        qpos = finger_q
        ctrl = [0.0, OPEN_MCP, OPEN_PIP] * 3
    return (
        '  <keyframe>\n'
        f'    <key name="open" qpos="{" ".join(f"{v:g}" for v in qpos)}"'
        f' ctrl="{" ".join(f"{v:g}" for v in ctrl)}" />\n'
        '  </keyframe>\n'
    )


def build_hand() -> str:
    bodies = "\n".join(_finger_body(f, morph_joints=True, indent="      ") for f in FINGERS)
    return (
        HEADER.format(model="real_v1_hand")
        + '  <worldbody>\n'
        '    <light pos="0 0 1.5" dir="0 0 -1" directional="true" diffuse=".6 .6 .6" specular=".2 .2 .2" />\n'
        '    <geom name="floor" type="plane" size="1.5 1.5 0.1" material="table_mat" contype="0" conaffinity="0" />\n'
        f'    <body gravcomp="1" name="palm_pose" pos="0 0 {PALM_Z:.6f}">\n'
        f'      <geom type="box" size="{PALM_HALF[0]:.4f} {PALM_HALF[1]:.4f} {PALM_HALF[2]:.4f}" material="palm_mat" />\n'
        f'{WORKSPACE_SITES}\n'
        f'{bodies}\n'
        '    </body>\n'
        '  </worldbody>\n'
        '  <contact>\n'
        f'{_excludes("    ")}\n'
        '  </contact>\n'
        '  <actuator>\n'
        f'{CTRL_ACTUATORS}\n'
        '  </actuator>\n'
        + _keyframe(is_scene=False)
        + '</mujoco>\n'
    )


def build_scene(actuated: bool = False) -> str:
    bodies = "\n".join(_finger_body(f, morph_joints=True, indent="      ") for f in FINGERS)
    model = "real_v1_scene_actuated" if actuated else "real_v1_scene_screwdriver_medium"
    actuators = POSE_ACTUATORS + "\n" + (MORPH_ACTUATORS + "\n" if actuated else "") + CTRL_ACTUATORS
    return (
        HEADER.format(model=model)
        + '  <worldbody>\n'
        '    <light pos="0 0 1.5" dir="0 0 -1" directional="true" diffuse=".6 .6 .6" specular=".2 .2 .2" />\n'
        '    <light pos="0.5 0.5 1.0" dir="-1 -1 -1" directional="true" diffuse=".4 .4 .4" specular=".1 .1 .1" />\n'
        '    <geom name="floor" type="plane" size="1.5 1.5 0.1" material="groundplane" />\n'
        '    <body name="screwdriver_medium" pos="0.0 0.0 0.0125" quat="0.70711 0.70711 0 0">\n'
        '      <freejoint />\n'
        '      <geom type="cylinder" size="0.0125 0.05" density="500" material="object_mat" friction="2.4 0.2 0.02" />\n'
        '    </body>\n'
        f'    <body gravcomp="1" name="palm_pose" pos="0 0 {PALM_Z:.6f}">\n'
        '      <joint class="pose" name="palm_px" type="slide" axis="1 0 0" range="-0.20 0.20" />\n'
        '      <joint class="pose" name="palm_py" type="slide" axis="0 1 0" range="-0.20 0.20" />\n'
        '      <joint class="pose" name="palm_pz" type="slide" axis="0 0 1" range="-0.15 0.35" />\n'
        '      <joint class="pose" name="palm_rx" type="hinge" axis="1 0 0" range="-1.57 1.57" />\n'
        '      <joint class="pose" name="palm_ry" type="hinge" axis="0 1 0" range="-1.57 1.57" />\n'
        '      <joint class="pose" name="palm_rz" type="hinge" axis="0 0 1" range="-3.14 3.14" />\n'
        f'      <geom type="box" size="{PALM_HALF[0]:.4f} {PALM_HALF[1]:.4f} {PALM_HALF[2]:.4f}" material="palm_mat" />\n'
        f'{WORKSPACE_SITES}\n'
        f'{bodies}\n'
        '    </body>\n'
        '  </worldbody>\n'
        '  <contact>\n'
        f'{_excludes("    ")}\n'
        '  </contact>\n'
        '  <actuator>\n'
        f'{actuators}\n'
        '  </actuator>\n'
        + _keyframe(is_scene=True)
        + '</mujoco>\n'
    )


def _without_keyframes(xml: str) -> str:
    """`xml` with its <keyframe> block removed, for structure-only comparison.

    Trailing whitespace goes too: `inject_keyframe` rewrites the file through ElementTree and
    does not restore the final newline, which is a one-byte difference and not drift.
    """
    return re.sub(r"\s*<keyframe>.*?</keyframe>", "", xml, flags=re.DOTALL).rstrip()


TARGETS = {
    "real_hand.xml": build_hand,
    "scenes/scene_screwdriver_medium.xml": lambda: build_scene(actuated=False),
    "real_hand_morphology_actuated.xml": lambda: build_scene(actuated=True),
}


def fit_palm_z(lo: float = 0.055, hi: float = 0.115, step: float = 0.0025) -> None:
    """Report, per candidate palm height, how hard the tripod grasp works each finger.

    Utilisation = |mount -> grasp target| / flexing reach. Too high and the finger is straight
    (no squeeze authority in either direction); too low and it has to fold past its ROM. m05's
    working hand sat at 0.75-0.84, which is the band we aim for.
    """
    import numpy as np

    reach = JOINT_SPACING + JOINT_SPACING + LINK_DISTAL     # 0.078660, mount -> pad surface
    # A tripod on the shaft: thumb from -x, index/middle from +x, all at the shaft mid-height.
    targets = {
        "thumb": np.array([-0.0125, 0.0, 0.0125]),
        "index": np.array([0.0125, 0.030, 0.0125]),
        "middle": np.array([0.0125, -0.030, 0.0125]),
    }
    print(f"flexing reach {reach*1000:.2f} mm   (shaft half-length 50.0 mm)")
    print(f"{'palm_z':>8} {'depth':>7} {'thumb':>7} {'index':>7} {'middle':>7}   verdict")
    z = lo
    while z <= hi + 1e-9:
        us = []
        for f, t in targets.items():
            mount = np.array([MOUNTS[f][0], MOUNTS[f][1], z])
            us.append(float(np.linalg.norm(t - mount)) / reach)
        depth = z - 0.0125
        ok = all(0.70 <= u <= 0.90 for u in us) and depth >= 0.050
        print(f"{z:8.4f} {depth*1000:6.1f}mm {us[0]:7.3f} {us[1]:7.3f} {us[2]:7.3f}   "
              f"{'OK' if ok else ''}")
        z += step


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="compile and report; fail if any file on disk differs from the build")
    ap.add_argument("--fit-palm-z", action="store_true",
                    help="print the palm-height utilisation table instead of building")
    args = ap.parse_args()

    if args.fit_palm_z:
        fit_palm_z()
        return 0

    stale = []
    for rel, builder in TARGETS.items():
        path = OUT_DIR / rel
        text = builder()
        if args.check:
            # Compare everything EXCEPT <keyframe>. The shipped `open` key is a placeholder;
            # scripts/fit_real_v1_pose.py overwrites it and adds `open_ik` once a scene has a
            # solved grasp, and that is the intended flow -- it must not read as drift.
            if not path.exists() or _without_keyframes(path.read_text()) != _without_keyframes(text):
                stale.append(rel)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path.relative_to(ROOT)}")

    if args.check and stale:
        print("STALE (rebuild with `uv run python scripts/build_real_v1_scenes.py`):")
        for rel in stale:
            print(f"  {rel}")
        return 1

    try:
        import mujoco
        import numpy as np
    except ImportError:
        return 0

    for rel in TARGETS:
        path = OUT_DIR / rel
        m = mujoco.MjModel.from_xml_path(str(path))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        # mj_contactForce, not efc_force[c.efc_address]: a pose with no active constraints
        # leaves efc empty and every efc_address at -1, which indexes the last row instead of
        # raising.
        wrench = np.zeros(6)
        peak = 0.0
        for k in range(d.ncon):
            mujoco.mj_contactForce(m, d, k, wrench)
            peak = max(peak, float(np.linalg.norm(wrench[:3])))
        print(f"  {rel:42} nq={m.nq:3d} nu={m.nu:3d} ncon={d.ncon} "
              f"peak-contact={peak:.3f} N")
    return 0


if __name__ == "__main__":
    sys.exit(main())
