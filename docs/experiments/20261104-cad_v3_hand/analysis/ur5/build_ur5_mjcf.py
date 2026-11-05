#!/usr/bin/env python3
"""Draft MJCF of the lab's UR5 (CB3) for the cad_v3 scenes, from ur_description's ur5.urdf (ros-industrial
universal_robot at 6507bb67, as copied into rh56_controller/magpie_force_control): the URDF's kinematic chain, joint
limits and inertials; the visual meshes converted from DAE to OBJ, one per material (dae_to_obj.py); collision
primitives fitted to the URDF's collision meshes (fit_collision.py); six position actuators; a tool0 site and body.

Frames. The root body base_link_inertia is the controller's Base frame B (URDF base_link_inertia and base coincide: both
are base_link turned by pi about z), so a site's pose in it is what the pendant reports with a zero TCP. tool0 is
wrist_3_link itself (the URDF's flange and tool0 joints compose to the identity): z out of the flange.

    .venv/bin/python logs/20261104-cad_v3/ur5/build_ur5_mjcf.py          writes ur5_cb3.xml and scene.xml here
"""
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path("/home/humanoid/Programs/hand")
sys.path.insert(0, str(ROOT / "src"))
from sr2hand.geometry import rotations as R  # noqa: E402

HERE = Path(__file__).resolve().parent
UR = Path.home() / "Programs/rh56_controller/magpie_force_control/src/ur_description"
URDF = UR / "ur5.urdf"
CHAIN = ("base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link",
         "wrist_3_link")
STEM = {"base_link_inertia": "base", "shoulder_link": "shoulder", "upper_arm_link": "upperarm", "forearm_link": "forearm",
        "wrist_1_link": "wrist1", "wrist_2_link": "wrist2", "wrist_3_link": "wrist3"}
ACT = {"shoulder_pan_joint": ("shoulder_pan", "size3"), "shoulder_lift_joint": ("shoulder_lift", "size3"),
       "elbow_joint": ("elbow", "size3_limited"), "wrist_1_joint": ("wrist_1", "size1"),
       "wrist_2_joint": ("wrist_2", "size1"), "wrist_3_joint": ("wrist_3", "size1")}
GAIN = 1.6          # the DAE's diffuse colours are dark for MuJoCo's headlight; all scaled by one factor, clipped at 1
# keyframes (rad), from the lab's records: docs/experiments/20261022-ur5_tip_calibration/lab/poses.jsonl and cell.json
KEYS = {
    "lab_home": [0.8428878870702559, -1.5053337486788703, 1.9119157214318552, -0.43316500622212184,
                 0.8263340452986379, 0.11221380904316902],                    # 'home_after_2x4_rotation', 2026-10-22 18:28
    "calib_home_v3id1": [1.0381939841128691, -1.3252233945171437, 2.0997127654259735, -0.7786831271834114,
                         1.0839976772818047, 0.031856119229146995],             # cell.json tool q_home (v3 ID 1, 10-27)
    "tool_down": None,                                                         # 'path_home_to_press_3_over_seat1', 11-02
    "hand_down": None,                  # computed: tool_down's flange xy at z_B 400 mm, tool axis exactly -z_B (DH IK)
    "zero": [0.0] * 6,
}


def fmt(v, nd=9):
    out = []
    for x in np.atleast_1d(v):
        x = float(x)
        if abs(x) < 10 ** -nd:
            x = 0.0
        out.append(f"{x:.{nd}g}")
    return " ".join(out)


def pose_attr(T):
    s = ""
    if np.linalg.norm(T[:3, 3]) > 1e-12:
        s += f' pos="{fmt(T[:3, 3])}"'
    q = R.mat_to_quat(T[:3, :3])
    if R.rot_angle(np.eye(3), T[:3, :3]) > 1e-9:
        s += f' quat="{fmt(q)}"'
    return s


def origin(el):
    o = el.find("origin") if el is not None else None
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0.0] * 3
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0.0] * 3
    return R.tf(R.euler_to_mat(rpy, "xyz"), np.array(xyz))


def parse():
    root = ET.parse(URDF).getroot()
    links = {ln.get("name"): ln for ln in root.findall("link")}
    joints = {}
    for j in root.findall("joint"):
        joints[j.find("child").get("link")] = j
    return links, joints


def lab_tool_down():
    rows = [json.loads(l) for l in open(ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab/poses.jsonl") if l.strip()]
    r = next(r for r in rows if r["label"] == "path_home_to_press_3_over_seat1")
    return r["q_rad"]


def hand_down(q0):
    """The tool axis straight down (-z_B) at tool_down's flange x, y and z_B 400 mm, x_tool0 along the horizontal part
    of tool_down's: the joints by the nominal DH's IK nearest q0 (a computed pose, never measured on the robot)."""
    sys.path.insert(0, str(ROOT / "src/sr2hand/driver/reskin5x/host"))
    import ur5_tip_calibration_sim as SIM
    T0 = SIM.fk(np.asarray(q0, float))
    x = T0[:3, 0].copy()
    x[2] = 0.0
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, -1.0])
    T = R.tf(np.c_[x, np.cross(z, x), z], np.r_[T0[0, 3], T0[1, 3], 400.0])
    q = SIM.ik(T, q0)
    if q is None:
        raise RuntimeError("no IK for hand_down")
    return [float(v) for v in q]


def geom_xml(pr, cls="collision"):
    if pr["type"] == "box":
        return (f'<geom name="{pr["name"]}" class="{cls}" type="box" pos="{fmt(pr["pos"], 6)}" '
                f'size="{fmt(pr["half"], 6)}"/>')
    if pr["type"] == "capsule":
        return (f'<geom name="{pr["name"]}" class="{cls}" type="capsule" fromto="{fmt(pr["fromto"], 6)}" '
                f'size="{fmt(pr["r"], 6)}"/>')
    quat = {"z": "", "x": ' quat="0.7071068 0 0.7071068 0"', "y": ' quat="0.7071068 -0.7071068 0 0"'}[pr["axis"]]
    return (f'<geom name="{pr["name"]}" class="{cls}" type="cylinder" pos="{fmt(pr["pos"], 6)}"{quat} '
            f'size="{fmt([pr["r"], pr["half"]], 6)}"/>')


def main():
    links, joints = parse()
    mats = json.loads((HERE / "meshes/visual/materials.json").read_text())
    prims = json.loads((HERE / "collision_primitives.json").read_text())["links"]
    KEYS["tool_down"] = lab_tool_down()
    KEYS["hand_down"] = hand_down(KEYS["tool_down"])
    # collision STLs copied beside the visual meshes (not used by the MJCF; for point-sample clearance checks)
    (HERE / "meshes/collision").mkdir(parents=True, exist_ok=True)
    for st in STEM.values():
        shutil.copy2(UR / f"meshes/ur5/collision/{st}.stl", HERE / f"meshes/collision/{st}.stl")
    material_xml, mesh_xml = [], []
    seen = set()
    for st in STEM.values():
        for p in mats[st]["parts"]:
            name = p["material"]
            if name not in seen:
                rgb = np.clip(np.array(p["rgba"][:3]) * GAIN, 0.0, 1.0)
                material_xml.append(f'<material name="{name}" class="ur5" rgba="{fmt(rgb, 3)} 1"/>')
                seen.add(name)
            mesh_xml.append(f'<mesh name="{p["file"][:-4]}" file="visual/{p["file"]}"/>')

    def body(i, indent):
        ln = CHAIN[i]
        L = links[ln]
        pad = " " * indent
        lines = []
        if i == 0:
            lines.append(f'{pad}<body name="{ln}" childclass="ur5">')
            lines.append(f'{pad}  <!-- the controller\'s Base frame B (URDF base_link_inertia = base = base_link turned by pi '
                         f'about z); z up -->')
            lines.append(f'{pad}  <site name="base" class="frame"/>')
            lines.append(f'{pad}  <site name="base_link" class="frame" quat="0 0 0 1"/>'
                         f'  <!-- ROS REP-103 base_link: x forward -->')
        else:
            j = joints[ln]
            T = origin(j)
            lines.append(f'{pad}<body name="{ln}"{pose_attr(T)} gravcomp="1">')
            lim = j.find("limit")
            lo, hi = float(lim.get("lower")), float(lim.get("upper"))
            act, cls = ACT[j.get("name")]
            rng = "" if cls != "size3_limited" else ""
            lines.append(f'{pad}  <joint name="{j.get("name")}" class="{cls}"/>'
                         f'  <!-- URDF limit {math.degrees(lo):.0f}..{math.degrees(hi):.0f} deg, '
                         f'effort {float(lim.get("effort")):.0f} N m, velocity {float(lim.get("velocity")):.4f} rad/s -->')
        inl = L.find("inertial")
        Ti = origin(inl)
        I = inl.find("inertia")
        lines.append(f'{pad}  <inertial mass="{fmt(float(inl.find("mass").get("value")))}"'
                     f' pos="{fmt(Ti[:3, 3])}"{(" quat=" + chr(34) + fmt(R.mat_to_quat(Ti[:3, :3])) + chr(34)) if R.rot_angle(np.eye(3), Ti[:3, :3]) > 1e-9 else ""}'
                     f' diaginertia="{fmt([float(I.get(k)) for k in ("ixx", "iyy", "izz")])}"/>')
        Tv = origin(L.find("visual"))
        for p in mats[STEM[ln]]["parts"]:
            lines.append(f'{pad}  <geom class="visual" mesh="{p["file"][:-4]}" material="{p["material"]}"{pose_attr(Tv)}/>')
        for pr in prims[ln]:
            lines.append(f"{pad}  {geom_xml(pr)}")
        if ln == "wrist_3_link":
            lines.append(f'{pad}  <!-- tool0 = wrist_3_link: z out of the flange face (z 0); the visual mesh has the four M6 '
                         f'holes at +-45 and +-135 deg from x, no dowel hole, and the tool connector on -y -->')
            lines.append(f'{pad}  <site name="tool0" class="frame"/>')
            lines.append(f'{pad}  <body name="tool0">')
            lines.append(f'{pad}    <!-- attach an end effector here (mjSpec attach, or <attach> in XML) -->')
            lines.append(f'{pad}  </body>')
        if i + 1 < len(CHAIN):
            lines += body(i + 1, indent + 2)
        lines.append(f"{pad}</body>")
        return lines

    kin = "\n".join(body(0, 4))
    keys = "\n".join(f'    <key name="{k}" qpos="{fmt(v, 8)}" ctrl="{fmt(v, 8)}"/>' for k, v in KEYS.items())
    acts = "\n".join(f'    <position class="{c}" name="{a}" joint="{j}"/>' for j, (a, c) in ACT.items())
    xml = f"""<mujoco model="ur5_cb3">
  <!-- The lab's UR5 (CB3, PolyScope 3.15.8), drafted for cad_v3 on 2026-11-04 by
       logs/20261104-cad_v3/ur5/build_ur5_mjcf.py from ur_description's ur5.urdf (ros-industrial/universal_robot
       6507bb67, copy in rh56_controller/magpie_force_control/src/ur_description; the package declares a BSD licence).
       Kinematics: UR's nominal DH (d1 89.159, a2 -425, a3 -392.25, d4 109.15, d5 94.65, d6 82.3 mm); the controller's
       own calibration is not in it. Metres, radians. -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>
  <option integrator="implicitfast"/>

  <default>
    <default class="ur5">
      <material specular="0.4" shininess="0.3"/>
      <joint axis="0 0 1" range="-6.283185 6.283185" armature="0.1"/>
      <position ctrlrange="-6.283185 6.283185" kp="2000" kv="400" forcerange="-150 150"/>
      <default class="size3">
        <default class="size3_limited">
          <joint range="-3.141593 3.141593"/>
          <position ctrlrange="-3.141593 3.141593"/>
        </default>
      </default>
      <default class="size1">
        <position kp="500" kv="100" forcerange="-28 28"/>
      </default>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom group="3" rgba="0.9 0.5 0.2 0.35"/>
      </default>
      <default class="frame">
        <site type="sphere" size="0.004" rgba="0.9 0.1 0.1 1" group="4"/>
      </default>
    </default>
  </default>

  <asset>
    {chr(10).join("    " + m for m in material_xml).strip()}
    {chr(10).join("    " + m for m in mesh_xml).strip()}
  </asset>

  <worldbody>
{kin}
  </worldbody>

  <contact>
    <!-- base_link_inertia is welded to the world, so MuJoCo's parent-child filter does not cover the shoulder's
         housing, which rests 0.04 mm into the base disc's top (z_B 24 mm) -->
    <exclude body1="base_link_inertia" body2="shoulder_link"/>
  </contact>

  <actuator>
{acts}
  </actuator>

  <keyframe>
    <!-- lab_home: the calibration cell's home, 2026-10-22 18:28 (flange along base -y);
         calib_home_v3id1: cell.json's tool registration (v3 ID 1 sphere pass 1, 2026-10-27);
         tool_down: the inverted rig's path over seat 1, 2026-11-02 14:20 (tool axis 3.8 deg from straight down);
         hand_down: computed by IK, not recorded: tool_down's flange x, y at z_B 400 mm, tool axis exactly -z_B -->
{keys}
  </keyframe>
</mujoco>
"""
    (HERE / "ur5_cb3.xml").write_text(xml)
    scene = """<mujoco model="ur5_cb3 scene">
  <include file="ur5_cb3.xml"/>
  <statistic center="0 -0.3 0.35" extent="1.0"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="120" elevation="-20" offwidth="1600" offheight="1200"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.85 0.88 0.92" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.86 0.86 0.86" rgb2="0.80 0.80 0.80" width="300" height="300"
      mark="edge" markrgb="0.7 0.7 0.7"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.6 1.8" dir="-0.1 0.3 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <!-- the robot's mounting plane is z 0 of B; the lab's mat table top stands at z_B about 32.8 mm (lab README) -->
    <geom name="floor" type="plane" size="1.5 1.5 0.05" material="floor"/>
  </worldbody>
</mujoco>
"""
    (HERE / "scene.xml").write_text(scene)
    print("wrote", HERE / "ur5_cb3.xml", HERE / "scene.xml")


if __name__ == "__main__":
    main()
