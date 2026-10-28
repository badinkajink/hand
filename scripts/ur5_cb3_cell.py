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
               "v5": "20261023-ur5_calibration_mount_v5", "v6": "20261023-ur5_calibration_mount_v6",
               "v7": "20261025-ur5_calibration_mount_v7"}[os.environ.get("UR5_MOUNT", "v3")]
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
# The tip cap trimmed flush with the cassette's side walls (owner, 2026-10-27: "i trimmed the top cap so that there is no
# "fingertip" coming out to the surface of the fingertip now, the top cap is more or less flush across the surface of the
# side walls"): the part of TW15H3 at x_F < CAP_TRIM_X is removed, the tongue over the pad's distal end (|y_F| < 2.9 mm,
# x_F -5.26 to -0.10, z_F -100.06 to -98.46), and the cut is closed by a face in the plane. None takes the cap as printed
# before 10-27, to recheck a plan as it was built; UR5_CAP_TRIM=none (or an x_F in mm) sets it for a process.
_CAP_TRIM = os.environ.get("UR5_CAP_TRIM", "-0.10").strip().lower()
CAP_TRIM_X = None if _CAP_TRIM in ("", "none", "old") else float(_CAP_TRIM)
_AS_SET = object()


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


def _inside_2d(Q, S):
    """Even-odd test of points Q (k, 2) against the closed boundary given as segments S (m, 2, 2)."""
    a, b = S[:, 0], S[:, 1]
    za, zb = a[None, :, 1], b[None, :, 1]
    cross = (za > Q[:, None, 1]) != (zb > Q[:, None, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        y = a[None, :, 0] + (Q[:, None, 1] - za) * (b[None, :, 0] - a[None, :, 0]) / (zb - za)
    return (cross & (y > Q[:, None, 0])).sum(axis=1) % 2 == 1


def _seg_dist_2d(Q, S):
    """Least distance of points Q (k, 2) to segments S (m, 2, 2)."""
    a, d = S[:, 0], S[:, 1] - S[:, 0]
    t = np.clip(np.einsum("kmj,mj->km", Q[:, None, :] - a[None], d) / np.maximum((d * d).sum(1), 1e-18)[None], 0.0, 1.0)
    return np.linalg.norm(Q[:, None, :] - (a[None] + t[..., None] * d[None]), axis=2).min(axis=1)


def trim_x(m, x0, sign=1.0, h=0.1, grid=0.3):
    """The part of the closed mesh m at sign (x - x0) >= 0, closed by its cut face in the plane x = x0. The cut face is
    triangulated from the boundary that the cut leaves, divided into pieces of at most h mm, and a grid of points inside it
    (grid mm apart, h or more from the boundary): of their Delaunay triangles, those whose centroids lie inside the boundary
    (even-odd). With every boundary piece shorter than any point's distance to it, each piece is an edge of the triangulation."""
    import trimesh
    from scipy.spatial import Delaunay
    from trimesh.intersections import slice_faces_plane
    nrm = np.array([sign, 0.0, 0.0])
    dots = (m.vertices[:, 0] - x0) * sign
    dots[np.abs(dots) < 1e-5] = 0.0                     # an STL's float32 -0.1 lies on the plane x = -0.10
    V, F, _ = slice_faces_plane(m.vertices, m.faces, nrm, np.array([x0, 0.0, 0.0]), cached_dots=dots)
    k = trimesh.Trimesh(V, F, process=True)
    E, n_use = np.unique(k.edges_sorted, axis=0, return_counts=True)
    E = E[n_use == 1]
    E = E[(np.abs(k.vertices[E, 0] - x0) < 1e-4).all(axis=1)]
    if not len(E):
        return k
    S = k.vertices[E][:, :, 1:]                        # (y, z) of the boundary's segments
    pts = [S[:, 0], S[:, 1]]
    for a, b in S:
        n = int(math.ceil(np.linalg.norm(b - a) / h))
        if n > 1:
            pts.append(a + (b - a) * (np.arange(1, n) / n)[:, None])
    lo, hi = S.reshape(-1, 2).min(0), S.reshape(-1, 2).max(0)
    G = np.stack(np.meshgrid(np.arange(lo[0] + grid / 2, hi[0], grid), np.arange(lo[1] + grid / 2, hi[1], grid),
                             indexing="ij"), -1).reshape(-1, 2)
    if len(G):
        G = G[_inside_2d(G, S)]
        G = G[_seg_dist_2d(G, S) >= h] if len(G) else G
    Q = np.unique(np.round(np.vstack(pts + [G]), 7), axis=0)
    tri = Delaunay(Q).simplices
    tri = tri[_inside_2d(Q[tri].mean(axis=1), S)]
    a, b, c = Q[tri[:, 0]], Q[tri[:, 1]], Q[tri[:, 2]]
    cr = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])   # x of the normal
    flip = sign * cr > 0.0                              # the cut face's normal points out of the kept part, -sign x
    tri[flip] = tri[flip][:, ::-1]
    capV = np.c_[np.full(len(Q), float(x0)), Q]
    return trimesh.Trimesh(np.vstack([k.vertices, capV]), np.vstack([k.faces, tri + len(k.vertices)]), process=True)


# The four M2 x 8 screws that hold the cassette and the cap have button heads (owner, 2026-10-27 20:2x: "they are
# buttonheads"; ISO 7380-1, 3.5 mm across and 1.3 mm tall in Accu's listing of SSB-M2), seated on the walls' and the cap's
# palmar faces at x_F -0.10, so their domes stand up to 1.3 mm proud of the trimmed cap (found 2026-10-27,
# scripts/ur5_tip_calibration_trimmed_cap_reach.py). Each is drawn as a spherical cap over its base. They go with the
# cap's mesh and so with the finger's group in every check. UR5_SCREW_HEADS=cheese gives the flat DIN 84 heads (3.8 x 1.3
# mm) that the checks of 2026-10-27 before 20:30 used, a cylinder around the dome; none leaves them out, to recheck a
# plan built before 5049f2c5 as it was built.
HEAD_KINDS = dict(button=dict(kind="button", d=3.5, h=1.3, seat_x_F=-0.10),
                  cheese=dict(kind="cheese", d=3.8, h=1.3, seat_x_F=-0.10))
_HEADS = os.environ.get("UR5_SCREW_HEADS", "button").strip().lower()
SCREW_HEADS = None if _HEADS in ("", "none", "off") else HEAD_KINDS["button" if _HEADS == "on" else _HEADS]


def heads_mesh_F(spec=None):
    """The four screw heads (spec, default SCREW_HEADS) as one mesh in F, their axes along x_F at TIP_F's holes, each
    rising toward -x_F from its seat."""
    import trimesh
    s = spec or SCREW_HEADS
    a, h = s["d"] / 2, s["h"]
    if s.get("kind") == "button":           # a spherical cap: its sphere's radius from the base's radius and the height
        R = (a * a + h * h) / (2 * h)
        phi = np.linspace(math.asin(a / R), 0.0, 13)
        head = trimesh.creation.revolve(np.r_[[[0.0, 0.0]], np.c_[R * np.sin(phi), h - R + R * np.cos(phi)]], sections=36)
        head.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [0.0, 1.0, 0.0]))   # z to -x
        x0 = s["seat_x_F"]
    else:
        head = trimesh.creation.cylinder(radius=a, height=h, sections=36)
        head.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0.0, 1.0, 0.0]))    # axis along x
        x0 = s["seat_x_F"] - h / 2
    out = []
    for y, z in TIP_F["holes"]:
        c = head.copy()
        c.apply_translation([x0, y, z])
        out.append(c)
    return trimesh.util.concatenate(out)


def part_F(name, trim=_AS_SET):
    """A tool part as a mesh in its own frame (F for the finger's parts, M for the mount's), the cap trimmed at x_F
    CAP_TRIM_X and carrying the four screw heads (SCREW_HEADS); trim gives another plane, or None for the cap as printed
    before 2026-10-27."""
    m = _load(TOOL_PARTS[name][0])
    x0 = CAP_TRIM_X if trim is _AS_SET else trim
    if name == "cap" and x0 is not None:
        m = trim_x(m, float(x0))
    if name == "cap" and SCREW_HEADS is not None:
        import trimesh
        m = trimesh.util.concatenate([m, heads_mesh_F()])
    return m


def finger_mesh(name, trim=_AS_SET):
    """A finger or mount part as a mesh in M (the cap trimmed as part_F)."""
    m = part_F(name, trim)
    if TOOL_PARTS[name][1] == "F":
        m.apply_transform(T_M_FROM_F)
    return m


def hex_e_mesh(T_EM):
    """The HEX-E as a cylinder in M (its tool face at z_M 0)."""
    import trimesh
    m = trimesh.creation.cylinder(radius=HEX["diameter_mm"] / 2, height=HEX["height_mm"], sections=48)
    m.apply_translation([0.0, 0.0, -HEX["height_mm"] / 2])
    return m


def part_points(name, m, spacing=2.0):
    """Surface samples of tool part `name` given as mesh m (as tool_points samples it)."""
    import trimesh
    sp = 1.0 if name in ("pad", "cover", "cap") else 0.5 if name == "stop_wedge" else spacing
    n = int(min(12000, max(300, m.area / sp ** 2)))
    return trimesh.sample.sample_surface_even(m, n, seed=0)[0]


@lru_cache(maxsize=4)
def tool_points(spacing=2.0):
    """Surface samples of the tool parts (M, mm), the cap trimmed at CAP_TRIM_X (its cut face sampled as surface)."""
    import trimesh
    out = {name: part_points(name, finger_mesh(name), spacing) for name in TOOL_PARTS}
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


# The tall indenter of 2026-10-26 (docs/experiments/20261026-tall_indenter, scripts/tall_indenter.py builds it from these
# numbers): a 5 mm sphere on a neck of radius neck_r that widens at taper_deg into a column of radius col_r, on a base plate
# screwed to the 2x4; the homing seat on a strut, its mouth plane mouth_below under the sphere's centre and key_y to +y.
# I frame: z up from the base's underside (the 2x4's top), x along the tool axis toward the tip at homing.
TALL = dict(H=160.0, R=5.0, neck_r=3.0, neck_z=4.0, taper_deg=15.0, col_r=30.0,
            base=(-40.0, 40.0, -45.0, 70.0, 8.0),       # x0, x1, y0, y1, thickness
            key_y=40.0, mouth_below=55.0,
            a_M=(10.55, 0.0, 93.36), mouth_M=(17.0, 0.0, 49.0))   # the pad's contact point and the socket's mouth in M


def tall_frame(t=None):
    """The homing geometry of the tall indenter (I frame): the socket's mouth, the contact point at homing and the map
    from M at homing (z_M -> x_I, y_M -> y_I, x_M -> -z_I)."""
    t = dict(TALL, **(t or {}))
    a = np.array(t["a_M"])
    m = np.array(t["mouth_M"])
    mouth = np.array([0.0, t["key_y"], t["H"] - t["mouth_below"]])
    mouth[0] = t.get("mouth_x", m[2] - a[2])                # -44.36: the mouth this far behind the contact along x_I
    contact = mouth + np.array([a[2] - m[2], 0.0, m[0] - a[0]])
    R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    T_IM = np.eye(4)
    T_IM[:3, :3] = R
    T_IM[:3, 3] = contact - R @ a
    return dict(mouth=mouth, contact=contact, T_IM_home=T_IM)


def tall_boxes(t=None):
    """The tall indenter's homing seat as boxes and convex solids in I (shared by the SDF and the CAD): the seat block, its
    floor recessed 0.6 mm under three lands, the guide walls along the mount's column, the strut to the column and the key.
    Optional keys of t (the fingertip holder's seat): wall_gap and wall_h, the walls' clearance and their four heights over
    the mouth plane (-y and +y below z_M 45.5, -y and +y above); stop, a wall facing the fork's palmar end at z_M 72.95
    (dict of gap, thickness t and height h), the block running under it."""
    t = dict(TALL, **(t or {}))
    f = tall_frame(t)
    mx, my, mz = f["mouth"]
    xM = lambda zm: zm - t["mouth_M"][2] + mx          # noqa: E731  (z_M -> x_I at homing)
    land = 2.0                                          # half-size of a land
    lands = [(xM(36.0), my - 16.5), (xM(36.0), my + 9.0), (xM(66.0), my - 3.5)]
    gap, wt = t.get("wall_gap", 0.4), 2.4               # guide walls: clearance to the column, thickness
    wh = t.get("wall_h", (4.0, 4.0, 4.0, 4.0))
    stop = t.get("stop")
    y_lo, y_hi = my - 23.05 - gap - wt, my + 17.2 + gap + wt
    x_end = xM(72.95 + stop["gap"] + stop["t"]) if stop else xM(70.5)
    walls = [((xM(31.0), my - 21.0 - gap - wt, mz - 0.61), (xM(45.0), my - 21.0 - gap, mz + wh[0])),   # the column
             ((xM(31.0), my + 13.3 + gap, mz - 0.61), (xM(45.0), my + 13.3 + gap + wt, mz + wh[1])),    # steps out at 45.45
             ((xM(46.5), y_lo, mz - 0.61), (xM(59.5), my - 23.05 - gap, mz + wh[2])),   # the clamp plate from z_M 60.4
             ((xM(46.5), my + 17.2 + gap, mz - 0.61), (x_end, y_hi, mz + wh[3]))]
    if stop:
        walls.append(((xM(72.95 + stop["gap"]), y_lo, mz - 0.61), (x_end, y_hi, mz + stop["h"])))
    out = dict(
        block=((xM(31.0), y_lo, mz - 10.0), (x_end, y_hi, mz - 0.6)),
        lands=[((x - land, y - land, mz - 0.61), (x + land, y + land, mz)) for x, y in lands],
        walls=walls,
        strut=[(x, y, mz - 9.5) for x, y in ((xM(31.0), y_lo), (x_end, y_lo), (x_end, y_hi), (xM(31.0), y_hi))]
              + [(x, y, mz - 85.0) for x, y in ((-18.0, 0.0), (2.0, 0.0), (2.0, 20.0), (-18.0, 20.0))],
        lands_xy=lands, mouth=f["mouth"])
    return out


def key_halves(d, t=None):
    """Half-sizes (along x_I, along y_I) of the tall indenter's key d mm above the mouth plane: the mount's socket (8 x 18 mm
    at the mouth, drafted 20 deg; scripts/ur5_calibration_mount.py home_halves) less 0.05 mm on every face."""
    tn = math.tan(math.radians(20.0))
    return (9.0 - d * tn - 0.05, 4.0 - d * tn - 0.05)


def _convex_sdf(P, pts):
    """Lower bound of the signed distance to the convex hull of pts (the largest of its faces' plane distances)."""
    from scipy.spatial import ConvexHull
    eq = ConvexHull(np.asarray(pts, float)).equations          # n . p + d <= 0 inside, n unit
    return (P @ eq[:, :3].T + eq[:, 3]).max(axis=1)


def key_points(t=None):
    """The key's frustum: its section at the seat's floor (0.6 mm under the mouth plane) and at its top (key_top, default
    4.5 mm above). With key_r its four edges along the socket's axis are rounded to that radius (7 points a corner), so that
    the faces bear in a printed socket whose corners are rounded."""
    t = dict(TALL, **(t or {}))
    mx, my, mz = tall_frame(t)["mouth"]
    r = t.get("key_r", 0.0)
    out = []
    for d in (-0.61, t.get("key_top", 4.5)):
        hx, hy = key_halves(d, t)
        if r <= 0.0:
            out += [(mx + sx * hx, my + sy * hy, mz + d) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            continue
        for k, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):     # corners counter-clockwise
            for j in range(7):
                a = math.radians(90.0 * k + 15.0 * j)
                out.append((mx + sx * (hx - r) + r * math.cos(a), my + sy * (hy - r) + r * math.sin(a), mz + d))
    return out


def tall_sdf(P, tip=True, t=None):
    t = dict(TALL, **(t or {}))
    H, tn = t["H"], math.tan(math.radians(t["taper_deg"]))
    z_neck = H - t["neck_z"]                                         # the neck meets the sphere here (radius neck_r)
    z_col = z_neck - (t["col_r"] - t["neck_r"]) / tn                 # the cone reaches col_r here
    x0, x1, y0, y1, tb = t["base"]
    b = tall_boxes(t)
    parts = [sd_box(P, (x0, y0, 0.0), (x1, y1, tb)),
             sd_cyl_z(P, t["col_r"], tb - 0.01, z_col),
             sd_capped_cone(P, z_col, z_neck + 0.5, t["col_r"], t["neck_r"] - 0.5 * tn),
             sd_box(P, *b["block"]), _convex_sdf(P, b["strut"]), _convex_sdf(P, key_points(t))]
    parts += [sd_box(P, lo, hi) for lo, hi in b["lands"] + b["walls"]]
    if tip:
        parts.append(np.linalg.norm(P - np.array([0.0, 0.0, H]), axis=1) - t["R"])
    return np.min(np.stack(parts), axis=0)


# The fingertip holder of 2026-10-27 (docs/experiments/20261027-fingertip_holder; scripts/fingertip_holder.py builds it from
# these numbers): two tips (pad, cassette 3T, top cover TW15H3 trimmed as CAP_TRIM_X, and the board), each screwed by its
# four M2 screws to a station block, the two blocks opposite each other on arms from one column (a T). On the -y station the
# tip faces down from the block's underside (side by side); on the +y station it faces up from the top face (the pads
# facing). Each block's four 1.9 mm holes pass through it, so either face takes a tip. The homing seat is the tall
# indenter's with the seat options of tall_boxes (taller and closer walls, a stop at the fork's palmar end, a key with
# rounded edges reaching 5.0 mm into the socket), on a tower of its own, its mouth at (mouth_x, key_y, mouth_z). TIP_F gives
# the tip in the DIP link's frame F: the cassette's back face (the board's component face) at x_F 3.80, the face's centre at
# z_F -87.91.
# Both held tips stand on their blocks' top faces, facing up (owner, 2026-10-27 19:50: "neither is underneath"); the design
# put the -y tip under its block for side-by-side poses, which "down" restores.
# The +y tip is turned 180 deg about the vertical, its end toward -x_I and its board's connector toward +x_I (owner,
# 2026-10-27 19:59: "the left side fingertip, the one closer to the sphere indenters, needs to be rotated 180 degrees about
# the Z axis"); tiphold_T_IF turns it about its screw pattern's centre (0.09 mm from the face's centre).
TIPHOLD = dict(stations=((-28.5, "up"), (28.5, "up")), turn_deg=(0.0, 180.0), x_s=0.0, z_bot=40.0, T=12.0, col_r=14.0,
               base=(-40.0, 40.0, -30.0, 98.0, 8.0), key_y=75.0, mouth_x=-22.0, mouth_z=30.0,
               seat=dict(wall_gap=0.2, wall_h=(14.0, 14.0, 11.5, 9.5), stop=dict(gap=0.1, t=4.0, h=9.5),
                         key_r=0.8, key_top=5.0))
# UR5_TIPHOLD_STATIONS="down,up" (the -y station's tip, then the +y station's) sets the stations for a process: the
# side-by-side plan of 2026-10-28 has the -y tip under its block (scripts/ur5_tip_calibration_holder_paths.py takes them
# from a plan's holder.stations).
_ST = [x.strip().lower() for x in os.environ.get("UR5_TIPHOLD_STATIONS", "up,up").split(",")]
if len(_ST) == 2 and set(_ST) <= {"up", "down"}:
    TIPHOLD["stations"] = ((-28.5, _ST[0]), (28.5, _ST[1]))
TIP_F = dict(x_seat=3.80, z_c=-87.91, apex_x=-10.55, z_ends=(-98.46, -77.05), y_half=10.5,
             holes=((8.0, -79.82), (-8.0, -79.82), (8.0, -95.82), (-8.0, -95.82)))
# The held tips' ReSkin 5X boards (owner, 2026-10-27: "ive added the magnetometer board and will have their data streaming
# to the ur5 computer ... model the slightly extra height, as well as the qwiic connector port sticking out"). Each board
# lies between its station block and the cassette as on the finger's tongue: its component face (chips, J1) at the
# cassette's back face x_F 3.80, its bare face on the block t mm further, so the tip stands t higher on the +y station and
# t lower on the -y station. Board frame (u, v, w) of the fabrication files (scripts/dip_link_geometry.py board): u = y_F,
# v = z_u1 - z_F (U2 toward the tip's end, the tab with J1 toward the PIP joint), w out of the component face, x_F = x_comp -
# w. The outline, J1's body and the chips are from the fabrication files (docs/experiments/20261018-dip_link_board_mount/
# spec.json, board); J1's header with the mated SHR-04V-S-B (7.0 mm across the tab, 2.0 mm out of the header, 2.96 mm
# tall) as dip_link_concept_a.j1_parts_board; the cable's first 3 mm as the flat 4 x 1 mm ribbon (owner, 2026-10-19) at
# the wires' mid-height. FR-4 0.6 mm is the ReSkin README's rigid order, not measured on the owner's boards; the chips'
# sensing points 0.42 mm under the package top (MLX90393 data sheet), the package 0.9 mm (the cassette's web). HELD_BOARD
# None models the tips without boards, as the holder was designed.
HELD_BOARD = dict(t=0.6, x_comp=3.80, z_u1=-87.91, chip_h=0.9, sense_below_top=0.42,
                  boxes=(("pcb", (-10.0, -10.10, -0.6), (10.0, 10.05, 0.0)),
                         ("pcb_tab", (-5.0, -19.10, -0.6), (5.0, -10.10, 0.0)),
                         ("j1", (-2.95, -18.09, 0.0), (3.05, -13.14, 2.9)),
                         ("plug", (-3.45, -20.09, 0.0), (3.55, -18.09, 2.96)),
                         ("cable", (-1.95, -23.09, 0.9), (2.05, -20.09, 1.9))),
                  chips=(("U1", 0.0, 0.0), ("U2", 0.0, 7.0), ("U3", 7.0, 0.0), ("U4", 0.0, -7.0), ("U5", -7.0, 0.05)))


def board_to_F(uvw, b=None):
    """Board coordinates (u, v, w) to F for a board at HELD_BOARD's place (as on the finger)."""
    b = b or HELD_BOARD
    uvw = np.asarray(uvw, float)
    return np.stack([b["x_comp"] - uvw[..., 2], uvw[..., 0], b["z_u1"] - uvw[..., 1]], axis=-1)


def held_board_boxes_F(b=None):
    """The board's parts as (name, lo, hi) boxes in F."""
    b = b or HELD_BOARD
    out = []
    for name, lo, hi in b["boxes"]:
        c = board_to_F(np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]), b)
        out.append((name, c.min(0), c.max(0)))
    return out


def held_chips_F(b=None):
    """The sensing points of U1-U5 in F."""
    b = b or HELD_BOARD
    w = b["chip_h"] - b["sense_below_top"]
    return {u: board_to_F([cu, cv, w], b) for u, cu, cv in b["chips"]}


def tiphold_T_IF(t=None, i=0):
    """The frame F of the tip on station i in I: pad up on the block's top face, or down from its underside; with
    HELD_BOARD the board's thickness between the block and the cassette."""
    t = dict(TIPHOLD, **(t or {}))
    ys, mode = t["stations"][i]
    zc, xs = TIP_F["z_c"], TIP_F["x_seat"] + (HELD_BOARD["t"] if HELD_BOARD else 0.0)
    T = np.eye(4)
    if mode == "up":                                       # x_F -> -z_I, y_F -> -y_I, z_F -> -x_I
        T[:3, :3] = [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]]
        T[:3, 3] = (t["x_s"] + zc, ys, t["z_bot"] + t["T"] + xs)
    else:                                                  # x_F -> z_I, y_F -> y_I, z_F -> -x_I
        T[:3, :3] = [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        T[:3, 3] = (t["x_s"] + zc, ys, t["z_bot"] - xs)
    turn = float((t.get("turn_deg") or (0.0,) * len(t["stations"]))[i])
    if turn:                                               # about x_F (the vertical) through the screw pattern's centre
        th = math.radians(turn)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(th), -math.sin(th)], [0.0, math.sin(th), math.cos(th)]])
        c = np.array([0.0, 0.0, float(np.mean([z for _, z in TIP_F["holes"]]))])
        Tc = np.eye(4)
        Tc[:3, :3], Tc[:3, 3] = Rx, c - Rx @ c
        T = T @ Tc
    return T


def tiphold_parts(t=None):
    """The holder in I as boxes and convex solids (shared by the SDF and the CAD): each station with its arm into the column
    and a 45 deg gusset under the arm, the column, and the seat (block, lands, walls, stop, key) on a tower."""
    t = dict(TIPHOLD, **(t or {}))
    ts = dict(TALL, H=t["mouth_z"] + TALL["mouth_below"], key_y=t["key_y"], mouth_x=t["mouth_x"], **t["seat"])
    b = tall_boxes(ts)
    tb = t["base"][4]
    xs, zb, zt = t["x_s"], t["z_bot"], t["z_bot"] + t["T"]
    zc, (zd, zp), yh = TIP_F["z_c"], TIP_F["z_ends"], TIP_F["y_half"]
    sx0, sx1 = xs - (zp - zc), xs - (zd - zc) - 0.1        # the cassette's ends; 0.1 mm short of the cover's end plate
    ya = 0.4 * t["col_r"]                                   # the arms and gussets reach this far into the column
    stations = []
    for ys, mode in t["stations"]:
        sg = 1.0 if ys > 0 else -1.0
        yi = ys - sg * yh                                   # the block's inner side, toward the column
        stations.append(dict(
            y_s=ys, mode=mode, block=((sx0, ys - yh, zb), (sx1, ys + yh, zt)),
            arm=((sx0, min(yi + sg * 0.1, sg * ya), zb), (sx1, max(yi + sg * 0.1, sg * ya), zt)),
            gusset=[(x, y, z) for x in (sx0, sx1) for y, z in ((yi, zb + 0.5), (sg * ya, zb + 0.5),
                                                               (sg * ya, zb - abs(yi - sg * ya)), (yi, zb))],
            holes=[(xs - (z - zc), ys + y) for y, z in TIP_F["holes"]]))
    (bx0, by0, _), (bx1, by1, _) = b["block"]
    return dict(stations=stations, z_top=zt, block=b["block"], lands=b["lands"], walls=b["walls"], key=key_points(ts),
                strut=[(x, y, z) for z in (tb - 0.01, b["block"][0][2] + 0.5)
                       for x, y in ((bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1))],
                lands_xy=b["lands_xy"], mouth=b["mouth"], frame=tall_frame(ts))


_HELD = {}


def _held_tip():
    """Surface samples (F) of the held tip's cassette, top cover (trimmed as CAP_TRIM_X) and pad, with their normals, in
    a k-d tree."""
    if not _HELD:
        import trimesh
        from scipy.spatial import cKDTree
        pts, nrm = [], []
        for name in ("cover", "cap", "pad"):
            m = part_F(name)
            n = int(min(20000, max(500, m.area / 0.4 ** 2)))
            p, fi = trimesh.sample.sample_surface_even(m, n, seed=1)
            pts.append(p)
            nrm.append(m.face_normals[fi])
        _HELD.update(p=np.vstack(pts), n=np.vstack(nrm))
        _HELD["tree"] = cKDTree(_HELD["p"])
    return _HELD


def held_tip_sdf(P_I, t=None):
    """Signed distance (mm, I) to the held tips and their boards. Beyond 3 mm of a tip's bounding box, the distance to the
    box (a lower bound); nearer, the distance to the nearest surface sample, signed by the eight nearest samples' normals
    together. The board's FR-4, J1, its plug and the cable's start are boxes (HELD_BOARD)."""
    t = dict(TIPHOLD, **(t or {}))
    zd, zp = TIP_F["z_ends"]
    best = None
    for i in range(len(t["stations"])):
        T = tiphold_T_IF(t, i)
        PF = (np.asarray(P_I, float) - T[:3, 3]) @ T[:3, :3]
        out = sd_box(PF, (-10.6, -10.5, zd - 1.6), (8.0, 10.5, zp))
        near = out < 3.0
        if near.any():
            h = _held_tip()
            Q = PF[near]
            d, j = h["tree"].query(Q, k=8)
            s = np.einsum("ijk,ijk->i", Q[:, None, :] - h["p"][j], h["n"][j])
            out[near] = d[:, 0] * np.where(s < 0.0, -1.0, 1.0)
        if HELD_BOARD:
            out = np.min(np.stack([out] + [sd_box(PF, lo, hi) for _, lo, hi in held_board_boxes_F()]), axis=0)
        best = out if best is None else np.minimum(best, out)
    return best


def tiphold_sdf(P, t=None):
    t = dict(TIPHOLD, **(t or {}))
    g = tiphold_parts(t)
    x0, x1, y0, y1, tb = t["base"]
    parts = [sd_box(P, (x0, y0, 0.0), (x1, y1, tb)), sd_cyl_z(P, t["col_r"], tb - 0.01, g["z_top"]),
             sd_box(P, *g["block"]), _convex_sdf(P, g["strut"]), _convex_sdf(P, g["key"]), held_tip_sdf(P, t)]
    for st in g["stations"]:
        parts += [sd_box(P, *st["block"]), sd_box(P, *st["arm"]), _convex_sdf(P, st["gusset"])]
    parts += [sd_box(P, lo, hi) for lo, hi in g["lands"] + g["walls"]]
    return np.min(np.stack(parts), axis=0)


def indenter_sdf(P, kind="sphere", tip=True):
    """Signed distance (mm) to the printed indenter (I frame); tip=False leaves out the part that presses (the sphere,
    or the rod's round top), for the pad's points around its contact."""
    P = np.asarray(P, float)
    if kind == "tall":
        return tall_sdf(P, tip)
    if kind == "tiphold":                                  # no contact: the held tip stays in
        return tiphold_sdf(P)
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
    if kind == "tiphold":
        return held_tip_sdf(P)
    if kind == "tall":
        return np.linalg.norm(P - np.array([0.0, 0.0, TALL["H"]]), axis=1) - TALL["R"]
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
    old, cut = part_F("cap", None), part_F("cap")
    print(f"cap trimmed at x_F {CAP_TRIM_X}: volume {old.volume:.1f} -> {cut.volume:.1f} mm^3, x_F from "
          f"{old.bounds[0, 0]:.2f} -> {cut.bounds[0, 0]:.2f}, area {old.area:.1f} -> {cut.area:.1f} mm^2")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
