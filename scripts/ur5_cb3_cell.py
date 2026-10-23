#!/usr/bin/env python3
"""Geometry of the lab's UR5 (CB3) cell for the fingertip's calibration plan and film: the robot's links (the UR5 URDF of
ur_description in rh56_controller/magpie_force_control, its collision meshes), the tool stack on the flange (the
OptoForce HEX-E, UR5 mount v3, DIP link b v4 with cover I15H3 3T, cap TW15H3 and the pad), the printed indenters on the
plywood, and the clearances between them. Millimetres; B is the controller's base frame, E the flange (tool0), M the mount
(origin on the HEX-E's tool face), F the DIP study's finger frame, P the pad's tip frame, I the indenter (z up from its
base on the plywood).

    .venv/bin/python scripts/ur5_cb3_cell.py check        tool0 of the URDF against the DH model, point counts
"""
from __future__ import annotations

import math
import os
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "src/morphohand/driver/reskin5x/host"
sys.path.insert(0, str(HOST))
import ur5_tip_calibration as C  # noqa: E402
import ur5_tip_calibration_sim as SIM  # noqa: E402

UR_DESC = next((p for p in (Path.home() / "Programs/rh56_controller/magpie_force_control/src/ur_description",
                            Path.home() / "workspace/rh56_controller/magpie_force_control/src/ur_description")   # UR5's PC
                if p.exists()), Path.home() / "Programs/rh56_controller/magpie_force_control/src/ur_description")
URDF = UR_DESC / "ur5.urdf"
# The mount: v3 of 2026-10-21 by default; UR5_MOUNT=v4 takes v4 of 2026-10-23 (v3 with the stop plate behind the tongue,
# the same frames and PIP_M), whose parts the builder writes to their own folder; UR5_MOUNT=v5 takes v4's parts and the PLA
# wedge in the stop plate's gap, as the space it takes over all its seats (stop_wedge_swept)
_MOUNT_STEM = {"v3": "20261021-ur5_calibration_mount_v3", "v4": "20261023-ur5_calibration_mount_v4",
               "v5": "20261023-ur5_calibration_mount_v5"}[os.environ.get("UR5_MOUNT", "v3")]
MOUNT = ROOT / "docs/experiments" / _MOUNT_STEM / "cad"
FINGER = ROOT / "docs/experiments/20261018-dip_link_board_mount/v4/b/cad"
TOOL_PARTS = {   # name: (file, frame)
    "adapter": (MOUNT / f"{_MOUNT_STEM}_adapter.stl", "M"),
    "fork": (MOUNT / f"{_MOUNT_STEM}_standoff_fork.stl", "M"),
    "clamp_plate": (MOUNT / f"{_MOUNT_STEM}_clamp_plate.stl", "M"),
    "dip_link": (FINGER / "20261021-dip_link_v4b_dip_link.stl", "F"),
    "cover": (FINGER / "20261022-dip_link_v4b_cover_I15H3_3T.stl", "F"),
    "cap": (FINGER / "20261021-dip_link_v4b_tip_cap_TW15H3.stl", "F"),
    "pad": (FINGER / "20261021-dip_link_v4b_pad_ell11_5.stl", "F"),
}
if (MOUNT / f"{_MOUNT_STEM}_stop_wedge_swept.stl").exists():
    TOOL_PARTS["stop_wedge"] = (MOUNT / f"{_MOUNT_STEM}_stop_wedge_swept.stl", "M")
INDENTER_STL = {"sphere": MOUNT / f"{_MOUNT_STEM}_indenter_sphere.stl",
                "rod": MOUNT / f"{_MOUNT_STEM}_indenter_rod.stl"}
HEX = dict(height_mm=35.0, diameter_mm=70.0, across_connector_mm=78.5, mass_kg=0.260)   # OptoForce HEX-E V1.0 data sheet
T_M_FROM_F = np.array([[-1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, -1.0, 5.45], [0, 0, 0, 1.0]])   # mount v3 (PIP_M 57)
T_M_FROM_P = C.Tm(None, [2.05, 0.0, 82.50])                      # the pad's tip frame in M (b v4 on mount v3)


def rpy_R(r, p, y):
    return C.exp_so3([0.0, 0.0, y]) @ C.exp_so3([0.0, p, 0.0]) @ C.exp_so3([r, 0.0, 0.0])


@lru_cache(maxsize=1)
def urdf_model():
    """Joints (name, parent, child, T_origin (mm), axis or None) and links' collision meshes (T_origin, file)."""
    root = ET.parse(URDF).getroot()
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) * 1e3 for v in (o.get("xyz", "0 0 0").split())] if o is not None else [0.0, 0.0, 0.0]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0").split())] if o is not None else [0.0, 0.0, 0.0]
        ax = j.find("axis")
        joints.append(dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                           child=j.find("child").get("link"), T=C.Tm(rpy_R(*rpy), xyz),
                           axis=None if ax is None else np.array([float(v) for v in ax.get("xyz").split()])))
    meshes = {}
    for ln in root.findall("link"):
        col = ln.find("collision")
        if col is None:
            continue
        o = col.find("origin")
        xyz = [float(v) * 1e3 for v in (o.get("xyz", "0 0 0").split())] if o is not None else [0.0, 0.0, 0.0]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0").split())] if o is not None else [0.0, 0.0, 0.0]
        fn = col.find("geometry/mesh").get("filename").replace("package://", "")
        meshes[ln.get("name")] = (C.Tm(rpy_R(*rpy), xyz), UR_DESC.parent / fn)
    return joints, meshes


ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def link_frames(q):
    """Every URDF link's frame in the controller's base frame ('base'), mm."""
    joints, _ = urdf_model()
    Tw = {"base_link": np.eye(4)}
    qd = dict(zip(ARM_JOINTS, q))
    pending = list(joints)
    while pending:
        rest = []
        for j in pending:
            if j["parent"] not in Tw:
                rest.append(j)
                continue
            T = Tw[j["parent"]] @ j["T"]
            if j["type"] == "revolute" and j["name"] in qd:
                T = T @ C.Tm(C.exp_so3(j["axis"] * qd[j["name"]]))
            Tw[j["child"]] = T
        if len(rest) == len(pending):
            break
        pending = rest
    T_base = Tw["base"]
    Ti = C.inv(T_base)
    return {k: Ti @ v for k, v in Tw.items()}


def _load(path):
    import trimesh
    m = trimesh.load(path, force="mesh")
    if np.abs(m.bounds).max() < 5.0:                   # metres (the URDF's meshes)
        m.apply_scale(1e3)
    return m


@lru_cache(maxsize=4)
def robot_points(spacing=6.0):
    """Surface samples of the links near the tool (link frame, mm)."""
    import trimesh
    _, meshes = urdf_model()
    out = {}
    for ln in ("forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"):
        T, f = meshes[ln]
        m = _load(f)
        n = int(max(200, m.area / spacing ** 2))
        p, _ = trimesh.sample.sample_surface_even(m, n, seed=0)
        out[ln] = C.apply(T, p)
    return out


def finger_mesh(name):
    """A finger or mount part as a mesh in M."""
    f, frame = TOOL_PARTS[name]
    m = _load(f)
    if frame == "F":
        m.apply_transform(T_M_FROM_F)
    return m


def hex_e_mesh(T_EM):
    """The HEX-E as a cylinder in M (its tool face at z_M 0)."""
    import trimesh
    m = trimesh.creation.cylinder(radius=HEX["diameter_mm"] / 2, height=HEX["height_mm"], sections=48)
    m.apply_translation([0.0, 0.0, -HEX["height_mm"] / 2])
    return m


@lru_cache(maxsize=4)
def tool_points(spacing=2.0):
    """Surface samples of the tool parts (M, mm), the pad's from the FE surface's own points too."""
    import trimesh
    out = {}
    for name in TOOL_PARTS:
        m = finger_mesh(name)
        sp = 1.0 if name in ("pad", "cover", "cap") else 0.5 if name == "stop_wedge" else spacing
        n = int(min(12000, max(300, m.area / sp ** 2)))
        p, _ = trimesh.sample.sample_surface_even(m, n, seed=0)
        out[name] = p
    hm = hex_e_mesh(None)
    out["hex_e"], _ = trimesh.sample.sample_surface_even(hm, 1500, seed=0)
    return out


# ------------------------------------------------------------------------------------------- the tool against the robot
# The HEX-E's cable port (owner, 2026-10-22, after the port was pressed against the forearm with joint 5 at 173.5 deg): a
# cylinder about 1 in long and 5/16 in across, radial from the sensor's side along the flange's +y (its top at homing),
# then a right-angle continuation 0.75 in long toward the flange's -x. In the flange frame E, at the HEX-E's mid-height.
PORT = dict(r_mm=3.97, z_mm=17.5, radial_mm=25.4, bend_mm=19.05)
SELF_LINKS = ("forearm_link", "wrist_1_link", "wrist_2_link")
SELF_CLEAR = dict(port=15.0, hex_e=10.0, tool=15.0)      # mm; the HEX-E's body stands 34.8 mm from wrist 2 in every pose


def _tube(a, b, r, n=20, m=10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    d /= L
    u = np.cross(d, [0.0, 0.0, 1.0])
    u = u if np.linalg.norm(u) > 1e-6 else np.cross(d, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    s_ = np.linspace(0.0, L, n)[:, None, None]
    th = np.linspace(0.0, 2 * math.pi, m, endpoint=False)[None, :, None]
    return (a + s_ * d + r * (np.cos(th) * u + np.sin(th) * w)).reshape(-1, 3)


@lru_cache(maxsize=1)
def port_points():
    """Surface samples of the HEX-E's cable port and its bend (E frame, mm)."""
    r0 = HEX["diameter_mm"] / 2
    a = np.array([0.0, r0, PORT["z_mm"]])
    b = a + [0.0, PORT["radial_mm"], 0.0]
    c = b + [-PORT["bend_mm"], 0.0, 0.0]
    return np.vstack([_tube(a, b, PORT["r_mm"]), _tube(b, c, PORT["r_mm"])])


@lru_cache(maxsize=1)
def _self_trees():
    from scipy.spatial import cKDTree
    return {ln: cKDTree(robot_points()[ln]) for ln in SELF_LINKS}


@lru_cache(maxsize=1)
def _self_tool_points():
    tp = tool_points()
    rest = np.vstack([tp[k][::8] for k in TOOL_PARTS])
    return tp["hex_e"][::2], rest


def self_clearance(q, T_EM):
    """Least distances (mm) of the HEX-E, its cable port and the mount and finger to the robot's own forearm and wrist
    links at joints q. The cell's checks (clearance) cover the indenters, the board and the table; this one covers the
    tool folding back onto the arm, as with joint 5 near 180 deg."""
    LF = link_frames(q)
    T_BE = LF["tool0"]
    hex_M, rest_M = _self_tool_points()
    parts = dict(hex_e=C.apply(T_BE @ T_EM, hex_M), port=C.apply(T_BE, port_points()), tool=C.apply(T_BE @ T_EM, rest_M))
    out = {}
    for name, PB in parts.items():
        out[name] = min(float(tree.query(C.apply(C.inv(LF[ln]), PB))[0].min()) for ln, tree in _self_trees().items())
    return out


def self_ok(q, T_EM, need=SELF_CLEAR):
    sc = self_clearance(q, T_EM)
    return all(sc[k] >= need[k] for k in need), sc


# ------------------------------------------------------------------------------------------- indenters (I frame)
def sd_box(P, lo, hi):
    c = (np.asarray(lo) + np.asarray(hi)) / 2
    h = (np.asarray(hi) - np.asarray(lo)) / 2
    q = np.abs(P - c) - h
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


def sd_cyl_z(P, r, z0, z1, c=(0.0, 0.0)):
    rho = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
    zc, h = (z0 + z1) / 2, (z1 - z0) / 2
    d = np.c_[rho - r, np.abs(P[:, 2] - zc) - h]
    return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)


def sd_cyl_y(P, r, y0, y1, xz=(0.0, 30.0)):
    rho = np.hypot(P[:, 0] - xz[0], P[:, 2] - xz[1])
    yc, h = (y0 + y1) / 2, (y1 - y0) / 2
    d = np.c_[rho - r, np.abs(P[:, 1] - yc) - h]
    return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)


def sd_capped_cone(P, z0, z1, r0, r1):
    """Exact distance to a cone frustum about z (radius r0 at z0, r1 at z1); Quilez's sdCappedCone."""
    qx = np.hypot(P[:, 0], P[:, 1])
    h = (z1 - z0) / 2
    qy = P[:, 2] - (z0 + z1) / 2
    k1 = np.array([r1, h])
    k2 = np.array([r1 - r0, 2 * h])
    cax = qx - np.minimum(qx, np.where(qy < 0, r0, r1))
    cay = np.abs(qy) - h
    t = np.clip(((k1[0] - qx) * k2[0] + (k1[1] - qy) * k2[1]) / (k2 @ k2), 0.0, 1.0)
    cbx = qx - k1[0] + k2[0] * t
    cby = qy - k1[1] + k2[1] * t
    s = np.where((cbx < 0) & (cay < 0), -1.0, 1.0)
    return s * np.sqrt(np.minimum(cax ** 2 + cay ** 2, cbx ** 2 + cby ** 2))


def indenter_sdf(P, kind="sphere", tip=True):
    """Signed distance (mm) to the printed indenter (I frame); tip=False leaves out the part that presses (the sphere,
    or the rod's round top), for the pad's points around its contact."""
    P = np.asarray(P, float)
    parts = [sd_box(P, (-35.0, -12.0, 0.0), (35.0, 12.0, 6.0)),                 # base
             sd_box(P, (-56.58, -12.0, 0.0), (-30.22, 12.0, 6.0)),               # arm to the homing post
             sd_box(P, (-56.58, 12.0, 0.0), (-32.14, 22.63, 6.0)),
             sd_box(P, (-56.58, 22.63, 0.0), (-32.14, 37.37, 49.05))]            # homing post and key
    if kind == "sphere":
        parts += [sd_cyl_z(P, 10.0, 5.99, 36.0), sd_capped_cone(P, 36.0, 41.0, 10.0, 5.0)]
        if tip:
            parts.append(np.linalg.norm(P - np.array([0.0, 0.0, 41.0]), axis=1) - 5.0)
    else:
        parts.append(sd_box(P, (-12.5, -10.0, 5.99), (12.5, 10.0, 30.0)))
        if tip:
            parts.append(sd_cyl_y(P, 12.5, -10.0, 10.0))
    return np.min(np.stack(parts), axis=0)


def tip_sdf(P, kind="sphere"):
    if kind == "sphere":
        return np.linalg.norm(P - np.array([0.0, 0.0, 41.0]), axis=1) - 5.0
    return sd_cyl_y(P, 12.5, -10.0, 10.0)


BOARD = dict(half_mm=200.0, thick_mm=18.0, riser_half_mm=150.0, block_mm=0.0, block_half_mm=(50.0, 30.0))
# the plywood around the indenter; a riser under it when it stands above the table; a block under the indenter's base
# (block_mm tall, block_half_mm in x_I, y_I), which lifts the indenter above the plywood


def cell_sdf(P_I, T_BI, board=None):
    """Distance (mm) to the block under the indenter, the plywood board (400 x 400 x 18 mm about the indenter), the riser
    under it (when the board stands above the table) and the table (the plane through the robot's base, z_B = 0)."""
    b = dict(BOARD, **(board or {}))
    h, t, hb = b["half_mm"], b["thick_mm"], b["block_mm"]
    parts = [sd_box(P_I, (-h, -h, -hb - t), (h, h, -hb))]
    if hb > 0:
        bx, by = b["block_half_mm"]
        parts.append(sd_box(P_I, (-bx, -by, -hb), (bx, by, 0.0)))
    z_table_I = -(T_BI[2, 3])                                 # the table's height in I (T_BI's rotation keeps z up)
    if -hb - t - z_table_I > 1.0:
        r = b["riser_half_mm"]
        parts.append(sd_box(P_I, (-r, -r, z_table_I), (r, r, -hb - t)))
    parts.append(P_I[:, 2] - z_table_I)
    return np.min(np.stack(parts), axis=0)


OTHER = []   # other indenters standing on the table, as obstacles: (T_BI, kind); the lab's rod beside the sphere (2026-10-22)


def other_sdf(P_B):
    """Least distance (mm) of base-frame points to the other indenters (1e9 when there are none)."""
    if not OTHER:
        return np.full(len(P_B), 1e9)
    return np.min([indenter_sdf(C.apply(C.inv(T), P_B), k) for T, k in OTHER], axis=0)


def clearance(q, T_BE, T_BI, T_EM, kind="sphere", contact_P=None, excl_mm=4.0, ply=True, contact_M=None):
    """Least clearances (mm) of each part to the indenter and to the plywood, its riser and the table for joints q and
    flange pose T_BE, and to the OTHER indenters. The pad's points within excl_mm of contact_P (its target, P frame) are
    kept from the indenter's tip (they touch it)."""
    T_IB = C.inv(T_BI)
    out = {}
    LF = link_frames(q)
    for ln, p in robot_points().items():
        PI = C.apply(T_IB @ LF[ln], p)
        out[ln] = min(float(indenter_sdf(PI, kind).min()), float(cell_sdf(PI, T_BI).min()) if ply else 1e9,
                      float(other_sdf(C.apply(LF[ln], p)).min()))
    T_IM = T_IB @ T_BE @ T_EM
    for name, p in tool_points().items():
        PI = C.apply(T_IM, p)
        if (name == "pad" and contact_P is not None) or (name == "fork" and contact_M is not None):
            if name == "pad":
                near = np.linalg.norm(C.apply(C.inv(T_M_FROM_P), p) - np.asarray(contact_P), axis=1) < excl_mm
            else:                                             # the column's flat face around the touch: a plane on the sphere
                near = (np.abs(p[:, 0] - contact_M[0]) < 0.05) & (np.linalg.norm(p - np.asarray(contact_M), axis=1) < 12.0)
            d = np.minimum(indenter_sdf(PI[~near], kind).min() if (~near).any() else 1e9,
                           indenter_sdf(PI[near], kind, tip=False).min() if near.any() else 1e9)
        else:
            d = float(indenter_sdf(PI, kind).min())
        out[name] = min(float(d), float(cell_sdf(PI, T_BI).min()) if ply else 1e9,
                        float(other_sdf(C.apply(T_BE @ T_EM, p)).min()))
    return out


def check():
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-math.pi, math.pi, 6)
        Tu = link_frames(q)["tool0"]
        Td = SIM.fk(q)
        print("URDF tool0 vs DH:", np.round(C.pose_error(Tu, Td), 6))
    print({k: len(v) for k, v in robot_points().items()}, {k: len(v) for k, v in tool_points().items()})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
