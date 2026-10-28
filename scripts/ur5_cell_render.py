#!/usr/bin/env python3
"""Rendered stills of the lab's UR5 (CB3) calibration cell as it stood on 2026-10-27 (owner, 2026-10-27: "render the scene
with the four indenters mounted relative to the UR5, preferably in MuJoCo, with some colour realism"). MuJoCo draws the
robot's visual meshes, the HEX-E, mount v7 and the finger, the four fixtures at their registered poses on two 2x4s with
their clamps, the cutting-mat board under the 2x4s and the narrow table, under six cool-white overhead lights with soft
shadows, in a room with green walls and a grey cement floor.

Three images at 1600 x 1000 go to docs/experiments/20261027-v3_tip_calibration_and_crosstalk/media/:
  20261027-ur5_cell_overview.png        the cell from the fixtures' end of the table (-y_B) and to the robot's side,
                                        the robot at pose 2 (--overview-pose seated for pose 1)
  20261027-ur5_cell_holder_hover.png    the fingertip holder, the robot's pad 2 mm over the +y held tip (pose 2)
  20261027-ur5_cell_holder_seated.png   the same camera, the mount's socket seated on the holder's key (pose 1)

Sources (mm; B the robot's base frame, I a fixture's frame, E the flange, M the mount, F the finger). Each fixture's T_BI
comes from the registration that lab/cell.json names for it (owner, 2026-10-27: renders that follow the registrations
through the night), the tool's T_EM and q_home from its "tool" entry; on 2026-10-27 at 20:43:
  low sphere fixture   lab/20261026-192158-v2id1 (touch-offs); ur5_cb3_cell.INDENTER_STL["sphere"]
  rod fixture          lab/20261027-rod-seed_3 (seated by hand at 19:42, 96 mm further along -x_B than before), or
                       --rod-T-json; ur5_cb3_cell.INDENTER_STL["rod"]
  tall indenter        lab/20261027-143529-tallv2id1 (touch-offs); 20261026-tall_indenter/cad/*.stl
  fingertip holder     lab/20261027-holder-apex-plusy (level, through the +y apex that touch-offs measured);
                       20261027-fingertip_holder/cad/*.stl; each held tip's cassette (cover), trimmed cap, screw heads
                       (button heads) and pad from ur5_cb3_cell.part_F
                       placed by tiphold_T_IF(i) (both tips up, the +y one turned 180 deg about the vertical: TIPHOLD's
                       stations and turn_deg, owner 19:4x and 20:00), its board (FR-4, J1, plug, ribbon stub) from
                       held_board_boxes_F
  robot                the visual meshes (.dae, read here) of the URDF that ur5_cb3_cell.urdf_model() reads, posed by
                       link_frames(q); the HEX-E (hex_e_mesh) and its cable port (PORT), and TOOL_PARTS by finger_mesh()
                       placed by T_BE T_EM; the finger's own board as the held ones
  pose 1 (seated)      cell.json's "seated" of lab/poses.jsonl (holder_seated_5): the joints that put the URDF's flange at
                       its T_tcp_mm (IK from its q_rad)
  pose 2 (hover)       cell.json's "hover" of its "hover_plan": T_BE = T_BI T_IF(at_I, twist) T_FM T_ME
                       (fingertip_holder.T_IF_moving) with T_BI the frame the program homes to (the plan's holder
                       registration), joints by ur5_cb3_cell.SIM.ik from the tool's q_home
  2x4s, clamps, mat    from the lab's overhead D435 frames of 2026-10-22 18:30 (lab/1830_after_rotation_color.png) and
                       2026-10-27 19:49 (logs/corvus/cellcam/cell_1949_color.png), read through the fixtures' known
                       places: 38 x 89 mm pine, the near one from x_B -545 to +85 (top z_B 70.35, the low sphere's and the
                       rod's bases' far ends at its far edge, the rod at its place before 19:3x, from checks.other of
                       lab/20261027-plan_sphere_lab_check_shear_v2id1.json), the far one from x_B -527 to +125 (its
                       right end is out of the frame; top at the tall indenter's z_B 71.2, centred under the tall
                       indenter and the holder); grey C-clamps at the near one's ends (the owner's "metal handles") and
                       blue bar clamps at the far one's; an 18 x 24 in grey mat with a 10 mm white grid on a board whose
                       top is the near 2x4's underside (z_B 32.35)
  table                (owner, 19:4x) narrow, just wider than the mat board, the robot's base on one corner: top at z_B 0,
                       its edges 10 mm past the mat board and 15 mm past the base; blue frame; floor 762 mm below the top
Colours (owner, 2026-10-27 19:33 and 19:4x): the four fixtures and the mount (adapter, standoff fork, clamp plate) purple
PLA; the fingertips on the robot and on the holder (DIP link, cassette, cap, pad) black; the boards green FR-4 with white
connectors; the HEX-E dark grey; the UR5 aluminium with the light blue joint caps and black rings of its visual meshes;
the 2x4s pale pine, the table warm birch with a blue frame, green walls, grey cement floor.

    MUJOCO_GL=egl NUMPY_MADVISE_HUGEPAGE=0 ~/.claude/bin/resguard.sh run --mem 4G --cpu 200 -- \\
        .venv/bin/python scripts/ur5_cell_render.py [--rod-T-json PATH] [--views overview,hover,seated] \\
        [--overview-pose hover|seated] [--size 1600x1000] [--out DIR] [--work DIR]

--rod-T-json PATH reads the rod fixture's 4x4 T_BI (mm) as a bare JSON list or under the key "T_BI" (a registration.json).
--views also takes "plan", a view from above (not rendered by default).
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
os.environ.setdefault("UR5_MOUNT", "v7")            # the mount on the robot since R25 (2026-10-26)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_holder as FH  # noqa: E402
import ur5_cb3_cell as CELL  # noqa: E402

C = CELL.C
LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
MEDIA = ROOT / "docs/experiments/20261027-v3_tip_calibration_and_crosstalk/media"
STEM = "20261027-ur5_cell"
CELL_JSON = LAB / "cell.json"                         # each fixture's registration, updated as fixtures are registered
SRC = dict(rod_plan=LAB / "20261027-plan_sphere_lab_check_shear_v2id1.json", poses=LAB / "poses.jsonl")
TALL_STL = ROOT / "docs/experiments/20261026-tall_indenter/cad/20261026-tall_indenter.stl"
HOLDER_STL = ROOT / "docs/experiments/20261027-fingertip_holder/cad/20261027-fingertip_holder.stl"


def registry():
    """lab/cell.json: each fixture's registration, the tool's, and the seating and the hover that pose the robot."""
    return json.loads(CELL_JSON.read_text())

# The 2x4s, their clamps, the mat board and the table (mm, B); see the docstring for where each number comes from
LUMBER = dict(w=89.0, h=38.0)
BOARD_ENDS = dict(near=(-545.0, 85.0), far=(-527.0, 125.0))           # x_B of each 2x4's ends
C_CLAMPS = ((-480.0, -579.0), (30.0, 129.0))                          # near 2x4: (jaw, spine) x_B
BAR_CLAMPS = ((-482.0, -548.0), (55.0, 140.0))                        # far 2x4: (jaw, bar) x_B
MAT_BOARD = dict(x=(-521.0, 89.0), y=(-906.0, -449.0), border=25.0, grid=10.0, bold=50.0)
TABLE = dict(past_mat=10.0, past_base=15.0, thick=38.0, floor=-762.0, leg=50.0, inset=15.0, apron=70.0)
ROOM = dict(x=(-2.2, 2.0), y=(-3.4, 1.65), height=2.75)               # walls (m, B), the ceiling height over the floor

# Materials: rgba, specular, shininess
MAT = {
    "alu": ((0.80, 0.81, 0.83, 1.0), 0.55, 0.55), "alu_joint": ((0.75, 0.76, 0.78, 1.0), 0.50, 0.50),
    "ur_blue": ((0.47, 0.66, 0.85, 1.0), 0.35, 0.50), "ur_black": ((0.10, 0.10, 0.11, 1.0), 0.30, 0.40),
    "ur_dark": ((0.22, 0.22, 0.23, 1.0), 0.40, 0.50), "hex_e": ((0.24, 0.24, 0.26, 1.0), 0.45, 0.60),
    "purple": ((0.42, 0.26, 0.68, 1.0), 0.30, 0.35), "black_pla": ((0.10, 0.10, 0.11, 1.0), 0.35, 0.45),
    "black_tpu": ((0.07, 0.07, 0.075, 1.0), 0.15, 0.25), "fr4": ((0.10, 0.42, 0.19, 1.0), 0.30, 0.50),
    "connector": ((0.94, 0.94, 0.91, 1.0), 0.20, 0.30), "cable": ((0.10, 0.10, 0.10, 1.0), 0.20, 0.30),
    "steel": ((0.64, 0.65, 0.68, 1.0), 0.80, 0.80), "frame_blue": ((0.13, 0.31, 0.62, 1.0), 0.45, 0.55),
    "clamp_grey": ((0.58, 0.60, 0.61, 1.0), 0.45, 0.45), "clamp_blue": ((0.10, 0.30, 0.68, 1.0), 0.35, 0.45),
    "clamp_yellow": ((0.96, 0.76, 0.10, 1.0), 0.30, 0.40), "rubber": ((0.06, 0.06, 0.06, 1.0), 0.10, 0.20),
    "wall": ((0.60, 0.72, 0.57, 1.0), 0.00, 0.10), "baseboard": ((0.28, 0.30, 0.28, 1.0), 0.10, 0.20),
}
EMISSION = dict(wall=0.30, baseboard=0.05)
DAE_CLASS = (("blau", "ur_blue"), ("scheibe", "ur_black"), ("verbindung", "alu_joint"), ("rohr", "alu"))


# ------------------------------------------------------------------------------------------------ sources
def load_T(path, key="T_BI"):
    """A 4x4 transform (mm) from a JSON file: a bare 4x4 list or the value under key."""
    d = json.loads(Path(path).read_text())
    T = np.array(d[key] if isinstance(d, dict) else d, float)
    if T.shape != (4, 4):
        raise ValueError(f"{path}: expected a 4x4 {key}, got shape {T.shape}")
    return T


def fixtures(rod_json=None):
    """T_BI of the four fixtures (lab/cell.json), the rod's place before 19:3x (rod_old, which with the low sphere's places
    the near 2x4) and the tool's registration (T_EM, q_home)."""
    cell = registry()
    fx = {k: load_T(LAB / v["registration"]) for k, v in cell["fixtures"].items()}
    if rod_json:
        fx["rod"] = load_T(rod_json)
    plan = json.loads(SRC["rod_plan"].read_text())
    fx["rod_old"] = np.array(next(o for o in plan["checks"]["other"] if o["kind"] == "rod")["T_BI"], float)
    tool = json.loads((LAB / cell["tool"]["registration"]).read_text())
    return dict(fx, T_EM=np.array(tool["T_EM"], float), q_home=np.array(tool["q_home"], float))


def robot_poses(fx):
    """Joints (rad) of pose 1 (seated on the holder's key) and pose 2 (the hover), with their flange poses."""
    poses = {}
    for line in SRC["poses"].read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            poses[r["label"]] = r
    cell = registry()
    seated, hover = cell["seated"], cell["hover"]
    p = poses[seated]
    T1 = np.array(p["T_tcp_mm"], float)
    q1 = CELL.SIM.ik(T1, np.array(p["q_rad"], float))
    if q1 is None:
        raise RuntimeError(f"no IK for {seated}'s T_tcp_mm")
    plan = json.loads((LAB / cell["hover_plan"]).read_text())
    act = next(a for a in plan["actions"] if a["id"] == hover)
    T_prog = fx["holder"]                                   # the frame the program homes to and the plan's at_I is in
    if plan.get("holder", {}).get("registration"):
        T_prog = load_T(ROOT / plan["holder"]["registration"] / "registration.json")
    T_IM = FH.T_IF_moving(act["at_I"], act["twist"]) @ np.linalg.inv(CELL.T_M_FROM_F)
    T2 = T_prog @ T_IM @ C.inv(fx["T_EM"])
    q2 = CELL.SIM.ik(T2, fx["q_home"])
    if q2 is None:
        raise RuntimeError(f"no IK for {hover}")
    return dict(seated=dict(q=q1, T_BE=T1, q_rec=np.array(p["q_rad"], float)), hover=dict(q=q2, T_BE=T2, act=act))


# ------------------------------------------------------------------------------------------------ meshes
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def read_dae(path):
    """Triangles of a COLLADA file as [(material, vertices (n, 3) in metres, faces (m, 3))], node transforms applied.
    Covers what the UR5 visual meshes use: <triangles> on <vertices> positions, nested nodes with <matrix>."""
    root = ET.parse(path).getroot()
    unit = root.find("c:asset/c:unit", NS)
    scale = float(unit.get("meter", "1")) if unit is not None else 1.0
    geoms = {}
    for g in root.findall("c:library_geometries/c:geometry", NS):
        mesh = g.find("c:mesh", NS)
        src = {}
        for s in mesh.findall("c:source", NS):
            stride = int(s.find(".//c:accessor", NS).get("stride", "1"))
            src[s.get("id")] = np.array(s.find("c:float_array", NS).text.split(), float).reshape(-1, stride)
        pos = src[mesh.find("c:vertices/c:input[@semantic='POSITION']", NS).get("source")[1:]]
        prims = []
        for tri in mesh.findall("c:triangles", NS):
            ins = tri.findall("c:input", NS)
            stride = max(int(i.get("offset")) for i in ins) + 1
            vo = next(int(i.get("offset")) for i in ins if i.get("semantic") == "VERTEX")
            idx = np.array(tri.find("c:p", NS).text.split(), int).reshape(-1, stride)[:, vo].reshape(-1, 3)
            prims.append((tri.get("material"), idx))
        geoms[g.get("id")] = (pos, prims)
    out = []

    def walk(node, T):
        m = node.find("c:matrix", NS)
        if m is not None:
            T = T @ np.array(m.text.split(), float).reshape(4, 4)
        for ig in node.findall("c:instance_geometry", NS):
            pos, prims = geoms[ig.get("url")[1:]]
            bind = {im.get("symbol"): im.get("target")[1:] for im in ig.findall(".//c:instance_material", NS)}
            P = (pos @ T[:3, :3].T + T[:3, 3]) * scale
            out.extend((bind.get(mat, mat), P, idx) for mat, idx in prims)
        for ch in node.findall("c:node", NS):
            walk(ch, T)

    for vs in root.findall("c:library_visual_scenes/c:visual_scene", NS):
        for n in vs.findall("c:node", NS):
            walk(n, np.eye(4))
    return out


def urdf_visuals():
    """{link: (T_origin (mm), .dae path)} of the URDF's visual elements."""
    root = ET.parse(CELL.URDF).getroot()
    out = {}
    for ln in root.findall("link"):
        v = ln.find("visual")
        if v is None:
            continue
        o = v.find("origin")
        xyz = [float(s) * 1e3 for s in o.get("xyz", "0 0 0").split()] if o is not None else [0.0] * 3
        rpy = [float(s) for s in o.get("rpy", "0 0 0").split()] if o is not None else [0.0] * 3
        fn = v.find("geometry/mesh").get("filename").replace("package://", "")
        out[ln.get("name")] = (C.Tm(CELL.rpy_R(*rpy), xyz), CELL.UR_DESC.parent / fn)
    return out


def tm(V, F):
    import trimesh
    return trimesh.Trimesh(np.asarray(V, float), np.asarray(F, int), process=True)


def transformed(m, T):
    m = m.copy()
    m.apply_transform(T)
    return m


def box_mesh(lo, hi):
    import trimesh
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    return m


def cylinder(a, b, r, sections=32):
    import trimesh
    return trimesh.creation.cylinder(radius=r, segment=[np.asarray(a, float), np.asarray(b, float)], sections=sections)


def cap_parts_F():
    """The tip cap trimmed at CAP_TRIM_X without its screw heads, and the four heads, in F."""
    m = CELL._load(CELL.TOOL_PARTS["cap"][0])
    if CELL.CAP_TRIM_X is not None:
        m = CELL.trim_x(m, float(CELL.CAP_TRIM_X))
    return m, (CELL.heads_mesh_F() if CELL.SCREW_HEADS is not None else None)


@lru_cache(maxsize=1)
def tip_parts_F():
    """One fingertip in F: [(name, mesh, material)]: cassette, cap, screw heads, pad and the board's parts."""
    cap, heads = cap_parts_F()
    out = [("cover", CELL.part_F("cover"), "black_pla"), ("cap", cap, "black_pla"),
           ("pad", CELL.part_F("pad"), "black_tpu")]
    if heads is not None:
        out.append(("heads", heads, "steel"))
    board_mat = dict(pcb="fr4", pcb_tab="fr4", j1="connector", plug="connector", cable="cable")
    for name, lo, hi in (CELL.held_board_boxes_F() if CELL.HELD_BOARD else []):
        out.append((f"board_{name}", box_mesh(lo, hi), board_mat[name]))
    return out


def port_mesh_E():
    """The HEX-E's cable port and its bend (ur5_cb3_cell.PORT) as two cylinders in E."""
    import trimesh
    r0, P = CELL.HEX["diameter_mm"] / 2, CELL.PORT
    a = np.array([0.0, r0 - 1.0, P["z_mm"]])
    b = a + [0.0, P["radial_mm"] + 1.0, 0.0]
    c = b + [-P["bend_mm"], 0.0, 0.0]
    ball = trimesh.creation.icosphere(subdivisions=2, radius=P["r_mm"])
    ball.apply_translation(b)
    return trimesh.util.concatenate([cylinder(a, b, P["r_mm"], 24), cylinder(b - [P["r_mm"], 0, 0], c, P["r_mm"], 24), ball])


# ------------------------------------------------------------------------------------------------ 2x4s, clamps, table
def lumber_frame(Ts, x_ref, z_top, ends):
    """Pose (4x4, mm) of a 2x4's top centre and its length: along the mean horizontal y_I of the fixtures Ts, its
    centreline through the mean of the points x_ref mm along each fixture's x_I, its ends at x_B ends, z at z_top."""
    u = np.mean([T[:3, 1] * [1.0, 1.0, 0.0] for T in Ts], axis=0)
    u /= np.linalg.norm(u)
    p = np.mean([C.apply(T, np.array([[x_ref, 0.0, 0.0]]))[0] for T in Ts], axis=0)
    p = p + u * ((ends[0] + ends[1]) / 2 - p[0]) / u[0]
    T = np.eye(4)
    T[:3, 0], T[:3, 2] = u, [0.0, 0.0, 1.0]
    T[:3, 1] = np.cross(T[:3, 2], u)
    T[:3, 3] = [p[0], p[1], z_top]
    return T, (ends[1] - ends[0]) / u[0]


def _axial(T, x_B):
    """The coordinate along a 2x4 (its frame T) of the point of its centreline at x_B."""
    return (x_B - T[0, 3]) / T[0, 0]


def c_clamp(T, jaw_x, spine_x, below):
    """A C-clamp gripping the 2x4 (top frame T) to the table: the fixed jaw's pad on the 2x4 at x_B jaw_x, the frame's
    upper arm out to the spine at spine_x outside the table's edge, the lower arm under the table top, the screw with its
    swivel pad pressing the top's underside (below mm under the 2x4's top) and the sliding T-handle. [(mesh, material)]"""
    sj, ss = _axial(T, jaw_x), _axial(T, spine_x)
    o = math.copysign(1.0, ss - sj)
    zu = -below                                         # the table top's underside
    lo = zu - 48.0                                      # the lower arm's underside
    parts = [(cylinder((sj, 0, 0), (sj, 0, 9), 11.0), "clamp_grey"),                     # the fixed jaw's pad
             (box_mesh((sj, -8, 9), (ss + o * 9, 8, 27)), "clamp_grey"),                 # upper arm
             (box_mesh((ss - o * 9, -10, lo), (ss + o * 9, 10, 27)), "clamp_grey"),      # spine
             (box_mesh((sj - o * 12, -8, lo), (ss, 8, lo + 18)), "clamp_grey"),          # lower arm
             (cylinder((sj, 0, lo - 28), (sj, 0, zu - 4), 5.5, 20), "steel"),           # screw
             (cylinder((sj, 0, zu - 4), (sj, 0, zu), 10.0), "steel"),                    # swivel pad
             (cylinder((sj, -55, lo - 22), (sj, 55, lo - 22), 4.0, 16), "steel")]        # sliding T-handle
    return [(transformed(m, T), mat) for m, mat in parts]


def bar_clamp(T, jaw_x, bar_x, below):
    """A one-handed bar clamp (blue arms, black pads, yellow grip) gripping the 2x4 (top frame T) to the table: the upper
    jaw's pad on the 2x4 at x_B jaw_x, the steel bar at bar_x outside the table's edge, the lower jaw under the top (below
    mm under the 2x4's top) and the pistol grip under it. [(mesh, material)]"""
    sj, sb = _axial(T, jaw_x), _axial(T, bar_x)
    o = math.copysign(1.0, sb - sj)
    zt = -below
    parts = [(box_mesh((sj - 16, -13, 0), (sj + 16, 13, 7)), "rubber"),
             (box_mesh((sj - o * 14, -11, 7), (sb - o * 3, 11, 33)), "clamp_blue"),
             (box_mesh((sb - 3, -10, zt - 150), (sb + 3, 10, 70)), "steel"),
             (box_mesh((sj - 16, -13, zt - 7), (sj + 16, 13, zt)), "rubber"),
             (box_mesh((sj - o * 14, -11, zt - 33), (sb - o * 3, 11, zt - 7)), "clamp_blue"),
             (box_mesh((sb - o * 48, -14, zt - 128), (sb - o * 3, 14, zt - 33)), "clamp_yellow"),
             (box_mesh((sb - o * 70, -8, zt - 112), (sb - o * 48, 8, zt - 50)), "clamp_blue")]
    return [(transformed(m, T), mat) for m, mat in parts]


def table_extent(base_xy):
    """The table top's x and y ranges (mm): 10 mm past the mat board, and 15 mm past the robot's base on the corner."""
    t, mb = TABLE, MAT_BOARD
    (bx0, by0), (bx1, by1) = base_xy
    return ((min(mb["x"][0], bx0) - t["past_mat"], max(mb["x"][1] + t["past_mat"], bx1 + t["past_base"])),
            (min(mb["y"][0], by0) - t["past_mat"], max(mb["y"][1] + t["past_mat"], by1 + t["past_base"])))


def table_frame(tx, ty):
    """The table's blue frame: four legs, an apron under the top and stretchers near the floor, as one mesh."""
    import trimesh
    t = TABLE
    g, s, a, ztop = t["leg"], t["inset"], t["apron"], -t["thick"]
    x0, x1, y0, y1 = tx[0] + s, tx[1] - s, ty[0] + s, ty[1] - s
    parts = [box_mesh((x, y, t["floor"]), (x + g, y + g, ztop)) for x in (x0, x1 - g) for y in (y0, y1 - g)]
    parts += [box_mesh((x0, y0, ztop - a), (x1, y0 + 25, ztop)), box_mesh((x0, y1 - 25, ztop - a), (x1, y1, ztop)),
              box_mesh((x0, y0, ztop - a), (x0 + 25, y1, ztop)), box_mesh((x1 - 25, y0, ztop - a), (x1, y1, ztop))]
    zs = t["floor"] + 150.0
    parts += [box_mesh((x0 + 5, y0 + g, zs), (x0 + g - 5, y1 - g, zs + 40)),
              box_mesh((x1 - g + 5, y0 + g, zs), (x1 - 5, y1 - g, zs + 40)),
              box_mesh((x0 + g, y1 - g + 5, zs), (x1 - g, y1 - 5, zs + 40))]
    return trimesh.util.concatenate(parts)


# ------------------------------------------------------------------------------------------------ textures
def _noise(h, w, cy, cx, rng):
    """Smooth noise in about [-1, 1] on an h x w grid from a cy x cx random lattice (bicubic)."""
    from PIL import Image
    lat = rng.uniform(-1.0, 1.0, (max(cy, 2), max(cx, 2))).astype(np.float32)
    return np.asarray(Image.fromarray(lat, "F").resize((w, h), Image.BICUBIC), np.float32)


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _save(rgb, path):
    from PIL import Image
    Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB").save(path)
    return path


def _grid(mm_w, mm_h, px_mm):
    W, H = int(mm_w * px_mm), int(mm_h * px_mm)
    return W, H, ((np.arange(W, dtype=np.float32) + 0.5) / px_mm)[None, :], \
        ((np.arange(H, dtype=np.float32) + 0.5) / px_mm)[:, None]


def pine_long(path, seed, mm_w, mm_h, px_mm=6.0):
    """Pale flat-sawn pine for a 2x4's long faces: growth rings as stripes along the length (u), warped by noise, with
    faint latewood bands, fibre streaks and two to four knots; mm_h of faces across (v)."""
    rng = np.random.default_rng(seed)
    W, H, X, Y = _grid(mm_w, mm_h, px_mm)
    r = Y + 7.0 * _noise(H, W, 4, 5, rng) + 2.2 * _noise(H, W, 10, 22, rng) + 0.5 * _noise(H, W, 40, 120, rng)
    knots = []
    for _ in range(int(rng.integers(2, 5))):
        kx, ky, kr = rng.uniform(40, mm_w - 40), rng.uniform(5, mm_h - 5), rng.uniform(3.0, 7.0)
        d2 = ((X - kx) / 1.6) ** 2 + (Y - ky) ** 2
        r = r + 2.5 * kr * np.exp(-d2 / (2 * (2.2 * kr) ** 2))
        knots.append((kr, d2))
    phase = np.mod(r / 4.2 + 0.18 * _noise(H, W, 12, 30, rng), 1.0)
    late = _smooth(0.70, 0.84, phase) * (1.0 - _smooth(0.93, 1.0, phase))
    early, latec = np.array([0.90, 0.82, 0.63], np.float32), np.array([0.81, 0.68, 0.47], np.float32)
    rgb = early + (latec - early) * (0.70 * late)[..., None]
    rgb *= (1.0 + 0.035 * _noise(H, W, 6, 9, rng) + 0.02 * _noise(H, W, max(H // 6, 2), max(W // 60, 2), rng))[..., None]
    for kr, d2 in knots:
        k = np.exp(-(d2 / kr ** 2) ** 2)[..., None]
        ring = 0.5 + 0.5 * np.cos(np.sqrt(d2) * 2.4)
        kc = np.array([0.66, 0.48, 0.28], np.float32) * (0.85 + 0.15 * ring)[..., None]
        rgb = rgb * (1 - k) + kc * k
    return _save(rgb, path)


def pine_end(path, seed, px_mm=8.0):
    """End grain of a 2x4 (89 x 38 mm): rings about a pith below the board, a little darker than the faces."""
    rng = np.random.default_rng(seed)
    W, H, X, Y = _grid(LUMBER["w"], LUMBER["h"], px_mm)
    px, py = rng.uniform(25, 65), -rng.uniform(25, 60)
    r = np.sqrt((X - px) ** 2 + (Y - py) ** 2) + 0.6 * _noise(H, W, 6, 12, rng)
    phase = np.mod(r / 4.2, 1.0)
    late = _smooth(0.6, 0.8, phase) * (1.0 - _smooth(0.9, 1.0, phase))
    early, latec = np.array([0.84, 0.73, 0.54], np.float32), np.array([0.70, 0.56, 0.37], np.float32)
    rgb = early + (latec - early) * late[..., None]
    rgb *= (1.0 + 0.04 * _noise(H, W, max(H // 3, 2), 4, rng))[..., None]
    return _save(rgb, path)


def birch(path, seed, mm_w, mm_h, px_mm=2.4):
    """A warm birch butcher-block top: strips 44 mm wide along v (the table's length) with their own tone and fine
    grain."""
    rng = np.random.default_rng(seed)
    W, H, X, Y = _grid(mm_w, mm_h, px_mm)
    strip = np.floor(X / 44.0).astype(int)
    tone = rng.normal(0.0, 0.03, int(strip.max()) + 1).astype(np.float32)[strip]
    r = X + 2.5 * _noise(H, W, 10, 8, rng) + 0.7 * _noise(H, W, 60, 30, rng)
    grain = 0.5 + 0.5 * np.sin(r * 2 * np.pi / 2.4)
    base = np.array([0.84, 0.67, 0.45], np.float32)
    shade = 1.0 + tone - 0.05 * grain ** 3 + 0.025 * _noise(H, W, 40, 14, rng)
    seam = 1.0 - 0.10 * np.exp(-((X - 44.0 * np.round(X / 44.0)) / 0.35) ** 2)
    return _save(base * (shade * seam)[..., None], path)


def cutting_mat(path, px_mm=5.0):
    """The grey self-healing mat of the D435 frames: white lines every 10 mm, bolder every 50 mm, a plain border."""
    b = MAT_BOARD
    mw, mh = b["x"][1] - b["x"][0], b["y"][1] - b["y"][0]
    W, H, X, Y = _grid(mw, mh, px_mm)
    m = b["border"]
    inside = (X > m) & (X < mw - m) & (Y > m) & (Y < mh - m)
    gx, gy = X - m, Y - m

    def line(g, step, half):
        d = np.abs(g - step * np.round(g / step))
        return 1.0 - _smooth(half, half + 0.25, d)

    lines = np.maximum(np.maximum(line(gx, b["grid"], 0.30), line(gy, b["grid"], 0.30)),
                       np.maximum(line(gx, b["bold"], 0.55), line(gy, b["bold"], 0.55)))
    mat, white, border = (np.array([0.55, 0.57, 0.58], np.float32), np.array([0.93, 0.94, 0.95], np.float32),
                          np.array([0.60, 0.62, 0.63], np.float32))
    rgb = np.where(inside[..., None], mat + (white - mat) * lines[..., None], border)
    rgb = rgb * (1.0 + 0.015 * _noise(H, W, 20, 24, np.random.default_rng(3)))[..., None]
    return _save(rgb, path)


def cement(path, px_mm=0.8, size=1500.0):
    """Grey cement: mottled tone at several scales and fine aggregate speckle; size mm square, tiling."""
    rng = np.random.default_rng(17)
    n = int(size * px_mm)
    a = np.zeros((n, n), np.float32)
    for cells, amp in ((4, 0.05), (12, 0.035), (40, 0.02), (n // 3, 0.03)):
        lat = rng.uniform(-1.0, 1.0, (cells, cells)).astype(np.float32)
        lat = np.pad(lat, 2, mode="wrap")                  # tiling: the lattice wraps
        from PIL import Image
        big = np.asarray(Image.fromarray(lat, "F").resize((int(n * (cells + 4) / cells), int(n * (cells + 4) / cells)),
                                                          Image.BICUBIC), np.float32)
        o = int(round(2 * n / cells))
        a += amp * big[o:o + n, o:o + n]
    speck = rng.uniform(0.0, 1.0, (n, n)).astype(np.float32)
    a += np.where(speck > 0.985, -0.06, 0.0) + np.where(speck < 0.01, 0.05, 0.0)
    return _save(np.array([0.57, 0.57, 0.55], np.float32) * (1.0 + a)[..., None], path)


def box_obj(path, lo, hi, uv_fn, faces=("+z", "-z", "+y", "-y", "+x", "-x")):
    """An axis-aligned box (mm) as an OBJ with four vertices per face, face normals and the texture coordinates that
    uv_fn(face, corners (4, 3)) gives."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    quads = {"+z": ([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]),
             "-z": ([x0, y1, z0], [x1, y1, z0], [x1, y0, z0], [x0, y0, z0]),
             "+y": ([x1, y1, z0], [x0, y1, z0], [x0, y1, z1], [x1, y1, z1]),
             "-y": ([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]),
             "+x": ([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]),
             "-x": ([x0, y1, z0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1])}
    normals = {"+z": (0, 0, 1), "-z": (0, 0, -1), "+y": (0, 1, 0), "-y": (0, -1, 0), "+x": (1, 0, 0), "-x": (-1, 0, 0)}
    vs, vts, vns, fs = [], [], [], []
    for k, f in enumerate(faces):
        q = np.array(quads[f], float)
        uv = np.asarray(uv_fn(f, q), float)
        for j in range(4):
            vs.append("v {:.4f} {:.4f} {:.4f}".format(*q[j]))
            vts.append("vt {:.6f} {:.6f}".format(*uv[j]))
        vns.append("vn {} {} {}".format(*normals[f]))
        a = 4 * k + 1
        fs += [f"f {a}/{a}/{k + 1} {a + 1}/{a + 1}/{k + 1} {a + 2}/{a + 2}/{k + 1}",
               f"f {a}/{a}/{k + 1} {a + 2}/{a + 2}/{k + 1} {a + 3}/{a + 3}/{k + 1}"]
    Path(path).write_text("\n".join(vs + vts + vns + fs) + "\n")
    return path


def write_obj(m, path, smooth_deg=30.0):
    import trimesh
    sm = trimesh.graph.smooth_shade(m, angle=np.radians(smooth_deg), facet_minarea=None)
    Path(path).write_text(trimesh.exchange.obj.export_obj(sm, include_normals=True, include_texture=False))
    return path


# ------------------------------------------------------------------------------------------------ the scene
def quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).reshape(9))
    return q


def fmt(v, k=6):
    return " ".join(f"{float(x):.{k}g}" for x in v)


class Scene:
    """MJCF assembly: meshes in mm (scaled to m), static bodies for the cell, mocap bodies for the robot and the tool."""

    def __init__(self, work):
        self.work = Path(work)
        self.assets, self.world, self.bodies, self.n = [], [], {}, 0

    def mesh(self, m, material, body="world", smooth_deg=30.0, name=None):
        self.n += 1
        name = name or f"m{self.n:03d}"
        p = write_obj(m, self.work / f"{name}.obj", smooth_deg)
        self.assets.append(f'<mesh name="{name}" file="{p.name}" scale="0.001 0.001 0.001" inertia="shell"/>')
        self._geom(f'<geom type="mesh" mesh="{name}" material="{material}"/>', body)

    def obj(self, path, material, body="world", T=None):
        name = Path(path).stem
        self.assets.append(f'<mesh name="{name}" file="{Path(path).name}" scale="0.001 0.001 0.001" inertia="shell"/>')
        if T is None:
            self._geom(f'<geom type="mesh" mesh="{name}" material="{material}"/>', body)
        else:
            self._geom(f'<geom type="mesh" mesh="{name}" material="{material}" pos="{fmt(T[:3, 3] * 1e-3)}" '
                       f'quat="{fmt(quat(T[:3, :3]))}"/>', body)

    def raw(self, xml, body="world"):
        self._geom(xml, body)

    def _geom(self, xml, body):
        (self.world if body == "world" else self.bodies.setdefault(body, [])).append(xml)

    def xml(self, w, h, cameras, lights, materials):
        bodies = "".join(f'<body name="{b}" mocap="true">{"".join(g)}</body>' for b, g in self.bodies.items())
        return f"""<mujoco model="ur5_cell">
<compiler meshdir="{self.work}" texturedir="{self.work}" inertiafromgeom="false"/>
<statistic extent="2.0" center="-0.2 -0.4 0.1"/>
<visual><global offwidth="{w}" offheight="{h}" fovy="35"/><quality shadowsize="8192" offsamples="16"/>
<map znear="0.004" zfar="40" shadowscale="1.0" shadowclip="2"/><rgba haze="0.60 0.72 0.57 1"/>
<headlight ambient="0.19 0.195 0.205" diffuse="0.06 0.06 0.065" specular="0 0 0"/></visual>
<default><geom contype="0" conaffinity="0" group="1"/></default>
<asset><texture name="sky" type="skybox" builtin="gradient" rgb1="0.86 0.86 0.84" rgb2="0.70 0.70 0.69" width="64"
height="64"/>{materials}{''.join(self.assets)}</asset>
<worldbody>{lights}{cameras}{''.join(self.world)}{bodies}</worldbody></mujoco>"""


def materials_xml(tex):
    out = []
    for k, (rgba, spec, shin) in MAT.items():
        out.append(f'<material name="{k}" rgba="{fmt(rgba, 4)}" specular="{spec}" shininess="{shin}" '
                   f'emission="{EMISSION.get(k, 0.0)}"/>')
    for k, (f, spec, shin, extra) in tex.items():
        out.append(f'<texture name="t_{k}" type="2d" file="{Path(f).name}"/>'
                   f'<material name="{k}" texture="t_{k}" specular="{spec}" shininess="{shin}" {extra}/>')
    return "".join(out)


def lights_xml():
    """Six cool-white spots under the ceiling, 2.7 m over the floor, pointing down, in two rows along x_B as tube
    fixtures would hang; the three nearest the fixtures cast soft shadows (bulb radius 0.12 m). Shadows from all six
    put acne on the printed parts' edges at grazing light."""
    out = []
    for x in (-1.00, -0.20, 0.60):
        for y in (-1.15, -0.05):
            cast = "true" if (x, y) in ((-1.00, -0.05), (-0.20, -1.15), (-0.20, -0.05)) else "false"
            out.append(f'<light type="spot" pos="{x} {y} 1.95" dir="0 0 -1" cutoff="40" exponent="1.2" '
                       f'diffuse="0.175 0.183 0.200" specular="0.10 0.10 0.11" castshadow="{cast}" bulbradius="0.12"/>')
    return "".join(out)


def look_at(pos, target, fovy, name):
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = (target - pos) / np.linalg.norm(target - pos)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return f'<camera name="{name}" pos="{fmt(pos * 1e-3)}" xyaxes="{fmt(right)} {fmt(up)}" fovy="{fovy}"/>'


# Cameras (mm in B): position, target, vertical field of view (deg)
CAMERAS = dict(overview=((433.0, -1839.0, 1145.0), (-215.0, -420.0, 60.0), 34.0),
               holder=((-11.0, -905.0, 205.0), (-236.0, -680.0, 120.0), 33.0),
               plan=((-230.0, -640.0, 2600.0), (-230.0, -639.0, 0.0), 24.0))
VIEWS = dict(overview=("overview", None, f"{STEM}_overview.png"),       # pose from --overview-pose
             hover=("holder", "hover", f"{STEM}_holder_hover.png"),
             seated=("holder", "seated", f"{STEM}_holder_seated.png"),
             plan=("plan", "hover", f"{STEM}_plan.png"))         # a view from above, not rendered by default


def build(work, fx, w, h):
    """The scene's MJCF, a log of where each part came from, and the 2x4s' frames."""
    import trimesh
    S = Scene(work)
    log = {}
    # robot links: visual meshes split by material
    base_xy = None
    for ln, (T0, path) in urdf_visuals().items():
        groups = {}
        for mat, P, F in read_dae(path):
            cls = next((c for k, c in DAE_CLASS if mat.lower().startswith(k)), "ur_dark")
            groups.setdefault(cls, []).append(tm(P * 1e3, F))
        for cls, ms in groups.items():
            S.mesh(transformed(trimesh.util.concatenate(ms), T0), cls, body=ln, name=f"{ln}_{cls}")
        if ln == "base_link_inertia":                    # the base's footprint in B (the base link does not move)
            V = np.vstack([transformed(trimesh.util.concatenate(ms), T0).vertices for ms in groups.values()])
            VB = C.apply(CELL.link_frames(np.zeros(6))[ln], V)
            base_xy = (VB[:, :2].min(0), VB[:, :2].max(0))
        log[ln] = dict(file=str(path), groups={c: int(sum(len(m.faces) for m in ms)) for c, ms in groups.items()})
    # the tool on the flange (E): HEX-E, its port, the mount, the finger and its board
    T_EM = fx["T_EM"]
    S.mesh(transformed(CELL.hex_e_mesh(None), T_EM), "hex_e", body="tool", name="hex_e", smooth_deg=40.0)
    S.mesh(port_mesh_E(), "ur_black", body="tool", name="hex_e_port", smooth_deg=50.0)
    for name in ("adapter", "fork", "clamp_plate"):
        S.mesh(transformed(CELL.finger_mesh(name), T_EM), "purple", body="tool", name=f"tool_{name}")
    S.mesh(transformed(CELL.finger_mesh("dip_link"), T_EM), "black_pla", body="tool", name="tool_dip_link")
    T_EF = T_EM @ CELL.T_M_FROM_F
    for name, m, mat in tip_parts_F():
        S.mesh(transformed(m, T_EF), mat, body="tool", name=f"tool_{name}")
    log["tool"] = dict(mount=os.environ.get("UR5_MOUNT"), parts=[str(CELL.TOOL_PARTS[k][0].name) for k in CELL.TOOL_PARTS])
    # the fixtures
    for key, path in (("sphere", CELL.INDENTER_STL["sphere"]), ("rod", CELL.INDENTER_STL["rod"]), ("tall", TALL_STL),
                      ("holder", HOLDER_STL)):
        S.mesh(transformed(trimesh.load(path, force="mesh"), fx[key]), "purple", name=f"fix_{key}")
        log[key] = dict(file=str(Path(path).relative_to(ROOT)), origin_B=np.round(fx[key][:3, 3], 1).tolist())
    for i in range(len(CELL.TIPHOLD["stations"])):
        T = fx["holder"] @ CELL.tiphold_T_IF(i=i)
        for name, m, mat in tip_parts_F():
            S.mesh(transformed(m, T), mat, name=f"held{i}_{name}")
    log["held_tips"] = [f"{y:+g} mm {mode}" for y, mode in CELL.TIPHOLD["stations"]]
    # the 2x4s
    boards = dict(near=lumber_frame([fx["sphere"], fx["rod_old"]], 35.0 - LUMBER["w"] / 2,
                                    float(np.mean([fx["sphere"][2, 3], fx["rod_old"][2, 3]])), BOARD_ENDS["near"]),
                  far=lumber_frame([fx["tall"], fx["holder"]], 0.0, float(fx["tall"][2, 3]), BOARD_ENDS["far"]))
    Wd, Hh = LUMBER["w"], LUMBER["h"]
    span = 2 * Wd + 2 * Hh
    tex = {}
    for k, (bname, (T, L)) in enumerate(boards.items()):
        tex[f"pine{k}"] = (pine_long(work / f"pine{k}.png", 5 + k, L, span), 0.06, 0.15, "")
        tex[f"pine_end{k}"] = (pine_end(work / f"pine_end{k}.png", 9 + k), 0.04, 0.10, "")

        def uv_long(f, q, L=L):
            u = (q[:, 0] + L / 2) / L
            if f in ("+z", "-z"):
                v = ((0.0 if f == "+z" else Wd) + q[:, 1] + Wd / 2) / span
            else:
                v = ((2 * Wd if f == "+y" else 2 * Wd + Hh) + q[:, 2] + Hh) / span
            return np.c_[u, v]

        def uv_end(f, q):
            return np.c_[(q[:, 1] + Wd / 2) / Wd, (q[:, 2] + Hh) / Hh]

        lo, hi = (-L / 2, -Wd / 2, -Hh), (L / 2, Wd / 2, 0.0)
        S.obj(box_obj(work / f"lumber_{bname}.obj", lo, hi, uv_long, ("+z", "-z", "+y", "-y")), f"pine{k}", T=T)
        S.obj(box_obj(work / f"lumber_{bname}_ends.obj", lo, hi, uv_end, ("+x", "-x")), f"pine_end{k}", T=T)
        log[f"2x4_{bname}"] = dict(top_centre_B=np.round(T[:3, 3], 1).tolist(), length=round(float(L), 1),
                                   yaw_deg=round(math.degrees(math.atan2(T[1, 0], T[0, 0])), 2))
    # the clamps: grey C-clamps on the near 2x4, blue bar clamps on the far one
    clamps = []
    T, _ = boards["near"]
    for jaw, spine in C_CLAMPS:
        clamps += c_clamp(T, jaw, spine, T[2, 3] + TABLE["thick"])
    T, _ = boards["far"]
    for jaw, bar in BAR_CLAMPS:
        clamps += bar_clamp(T, jaw, bar, T[2, 3] + TABLE["thick"])
    for mat in sorted({m for _, m in clamps}):
        S.mesh(trimesh.util.concatenate([m for m, mm in clamps if mm == mat]), mat, name=f"clamps_{mat}", smooth_deg=40.0)
    # the mat board and the table
    mb = MAT_BOARD
    z_mat = boards["near"][0][2, 3] - Hh
    tex["mat"] = (cutting_mat(work / "mat.png"), 0.10, 0.25, "")
    mw, mh = mb["x"][1] - mb["x"][0], mb["y"][1] - mb["y"][0]

    def uv_mat(f, q):
        if f == "+z":
            return np.c_[(q[:, 0] - mb["x"][0]) / mw, (q[:, 1] - mb["y"][0]) / mh]
        return np.full((4, 2), 0.004)

    S.obj(box_obj(work / "mat_board.obj", (mb["x"][0], mb["y"][0], 0.0), (mb["x"][1], mb["y"][1], z_mat), uv_mat),
          "mat")
    log["mat_board"] = dict(x=mb["x"], y=mb["y"], top=round(float(z_mat), 2))
    tx, ty = table_extent(base_xy)
    tw, th = tx[1] - tx[0], ty[1] - ty[0]
    tex["birch"] = (birch(work / "birch.png", 21, tw, th), 0.14, 0.30, "")

    def uv_table(f, q):
        if f in ("+z", "-z"):
            return np.c_[(q[:, 0] - tx[0]) / tw, (q[:, 1] - ty[0]) / th]
        if f in ("+y", "-y"):
            return np.c_[(q[:, 0] - tx[0]) / tw, 0.30 + (q[:, 2] + TABLE["thick"]) / th * 0.25]
        return np.c_[0.40 + (q[:, 2] + TABLE["thick"]) / tw * 0.25, (q[:, 1] - ty[0]) / th]

    S.obj(box_obj(work / "table_top.obj", (tx[0], ty[0], -TABLE["thick"]), (tx[1], ty[1], 0.0), uv_table), "birch")
    S.mesh(table_frame(tx, ty), "frame_blue", name="table_frame", smooth_deg=10.0)
    log["table"] = dict(x=np.round(tx, 1).tolist(), y=np.round(ty, 1).tolist(), base_xy=np.round(base_xy, 1).tolist())
    # the room: cement floor, green walls with baseboards
    tex["floor"] = (cement(work / "cement.png"), 0.08, 0.20, 'texuniform="true" texrepeat="1.3333 1.3333"')
    f = TABLE["floor"] * 1e-3
    (rx0, rx1), (ry0, ry1), rh = ROOM["x"], ROOM["y"], ROOM["height"]
    cx, cy, hx, hy = (rx0 + rx1) / 2, (ry0 + ry1) / 2, (rx1 - rx0) / 2, (ry1 - ry0) / 2
    S.raw(f'<geom type="plane" size="{hx} {hy} 0.1" pos="{cx} {cy} {f}" material="floor"/>')
    for (px, py, sx, sy) in ((cx, ry1 + 0.05, hx, 0.05), (cx, ry0 - 0.05, hx, 0.05),
                             (rx0 - 0.05, cy, 0.05, hy), (rx1 + 0.05, cy, 0.05, hy)):
        S.raw(f'<geom type="box" size="{sx} {sy} {rh / 2}" pos="{px} {py} {f + rh / 2}" material="wall"/>')
    for (px, py, sx, sy) in ((cx, ry1 - 0.006, hx, 0.006), (cx, ry0 + 0.006, hx, 0.006),
                             (rx0 + 0.006, cy, 0.006, hy), (rx1 - 0.006, cy, 0.006, hy)):
        S.raw(f'<geom type="box" size="{sx} {sy} 0.05" pos="{px} {py} {f + 0.05}" material="baseboard"/>')
    cams = "".join(look_at(p, t, fv, name) for name, (p, t, fv) in CAMERAS.items())
    return S.xml(w, h, cams, lights_xml(), materials_xml(tex)), log, boards


def set_pose(m, d, q):
    import mujoco
    LF = CELL.link_frames(q)
    for ln in list(urdf_visuals()) + ["tool"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ln)
        if bid < 0:
            continue
        k = m.body_mocapid[bid]
        T = LF["tool0"] if ln == "tool" else LF[ln]
        d.mocap_pos[k] = T[:3, 3] * 1e-3
        d.mocap_quat[k] = quat(T[:3, :3])
    mujoco.mj_forward(m, d)
    return LF


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rod-T-json", default=None, help="JSON with the rod fixture's 4x4 T_BI in mm (a bare list or under "
                    "'T_BI'); default: lab/cell.json's rod entry")
    ap.add_argument("--views", default="overview,hover,seated")
    ap.add_argument("--overview-pose", default="hover", choices=("seated", "hover"),
                    help="the robot's pose in the overview: hover (pose 2, default) or seated (pose 1); at the rod's place "
                         "before 19:3x the wrist hides it at pose 2")
    ap.add_argument("--size", default="1600x1000", help="W x H in pixels")
    ap.add_argument("--out", default=str(MEDIA))
    ap.add_argument("--work", default=None, help="keep the converted meshes and textures here (default: a temp dir)")
    a = ap.parse_args()
    import mujoco
    from PIL import Image
    w, h = (int(v) for v in a.size.lower().split("x"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fx = fixtures(a.rod_T_json)
    poses = robot_poses(fx)
    tmp = None
    if a.work:
        work = Path(a.work)
        work.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="ur5_cell_render_")
        work = Path(tmp.name)
    xml, log, boards = build(work, fx, w, h)
    (work / "scene.xml").write_text(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    print(f"model: {m.nmesh} meshes, {m.ngeom} geoms, {m.ntex} textures, {m.nlight} lights ({time.time() - t0:.1f} s)")
    ren = mujoco.Renderer(m, h, w)
    try:
        for v in a.views.split(","):
            cam, pose, fname = VIEWS[v]
            pose = pose or a.overview_pose
            set_pose(m, d, poses[pose]["q"])
            ren.update_scene(d, camera=cam)
            Image.fromarray(ren.render()).save(out / fname)
            print(f"{v}: {cam} camera, pose {pose} -> {out / fname}")
    finally:
        ren.close()
    for k in ("seated", "hover"):
        p = poses[k]
        print(f"pose {k}: joints {np.round(np.degrees(p['q']), 2).tolist()} deg, "
              f"flange {np.round(p['T_BE'][:3, 3], 1).tolist()}")
    print("q_rad of", registry()["seated"], np.round(np.degrees(poses["seated"]["q_rec"]), 2).tolist())
    for k, v in log.items():
        if not k.endswith("_link"):
            print(k, v)
    print("fixtures from", a.rod_T_json or "", {k: v["registration"] for k, v in registry()["fixtures"].items()},
          f"{time.time() - t0:.1f} s")
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
