#!/usr/bin/env python3
"""Assemble the cad_v1 finger from its part STLs and write a MuJoCo model of it.

The parts in assets/cad_v1/ are exported one per file, each in its own frame, in millimetres.
Every joint joins a 10.2 mm bearing bore in the parent (an MR105 bearing, 5 x 10 x 4 mm, sits in
it) to a 5.2 mm slotted bore in the child, through a shared 5 mm shaft. The parent also carries
the SCS0009 servo whose output is coaxial with the bearing; the child is a clevis whose second
arm is screwed to the servo horn. The assembly is read from the parts:

  joint axis      the parent's bearing axis and the child's slotted-bore axis coincide;
  axial position  the inner face of the child's slotted arm lies flush on the outer face of the
                  parent's bearing wall (the bearing is flush in that 4 mm wall);
  servo seat      the servo's output is coaxial with the bearing, its base rests on the inner
                  face of the bearing wall, its tab holes are turned toward the midpoint of the
                  parent's two 1.8 mm pilot holes (the tab-hole residuals then test the 5.70 mm
                  spline offset and the 28.50 mm pitch);
  zero angle      the finger is straight: the child's long direction continues the parent's
                  (yaw link: yaw axis to MCP axis; MCP link: MCP axis to PIP axis; DIP link:
                  parallel to the pad face, away from the PIP axis); the carriage hangs from the
                  gantry with its long arm up;
  fingertip       the TPU tip's pose on the DIP link is registered from the fused STL
                  fingertip_and_DIP_link.stl.

The real_v1 conventions are kept: body names, joint names and axes, qpos layout, actuators and
mount positions of assets/mjcf/real_v1/real_hand.xml. All bodies are world-aligned at q = 0;
the thumb is the same finger turned 180 degrees about the vertical, applied to its geoms.

    .venv/bin/python scripts/build_cad_v1_finger.py              # derive, report, write the model
    .venv/bin/python scripts/build_cad_v1_finger.py --report     # derive and report only
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from morphohand.geometry import cad_features as cf  # noqa: E402

CAD = REPO / "assets/cad_v1"
OUT = REPO / "assets/mjcf/experimental/20261013-cad_v1"
FILES = {
    "carriage": "carriage.stl",
    "yaw_link": "mcp_mount_YAW-link.stl",
    "mcp_link": "pip_mount_MCP_link.stl",
    "dip_link": "fingertip_mount_DIP_link.stl",
    "tip": "fingertip.stl",
    "servo": "feetech scs0009.stl",
    "tip_and_dip": "fingertip_and_DIP_link.stl",
}
BEARING_D, SLOT_D, PILOT_D, TAB_D, SPLINE_D = 10.2, 5.2, 1.8, 2.3, 4.0
SERVO_TAB_PITCH = 28.5  # mm between the SCS0009 tab holes (measured from its STL below)
# parent -> child, with the joint name of real_v1
JOINTS = [("carriage", "yaw_link", "yaw"), ("yaw_link", "mcp_link", "mcp"), ("mcp_link", "dip_link", "pip")]
SERVO_IN = {"carriage": "yaw", "yaw_link": "mcp", "mcp_link": "pip"}


def load(name: str) -> trimesh.Trimesh:
    return trimesh.load(CAD / FILES[name], force="mesh")


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def perp(v, a):
    v = np.asarray(v, float)
    return v - (v @ a) * a


def design_rotation(mesh: trimesh.Trimesh) -> np.ndarray:
    """Rotation that brings a part exported slightly rotated back onto its design axes.

    The carriage STL is rotated 0.925 deg about x: its bores AND its large planar faces share the
    tilt, so it is an export rotation, not a design feature. Take the largest planar facet whose
    normal is within 3 deg of a coordinate axis and rotate it onto that axis.
    """
    order = np.argsort(-mesh.facets_area)
    for i in order[:5]:
        n = mesh.facets_normal[i]
        k = int(np.argmax(np.abs(n)))
        target = np.zeros(3)
        target[k] = np.sign(n[k])
        ang = np.degrees(np.arccos(np.clip(n @ target, -1, 1)))
        if ang < 3.0:
            if ang < 1e-4:
                return np.eye(3)
            ax = unit(np.cross(n, target))
            return trimesh.transformations.rotation_matrix(np.radians(ang), ax)[:3, :3]
    return np.eye(3)


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return cf.transform(R, cb - R @ ca)


def register_copy(src: trimesh.Trimesh, tgt: trimesh.Trimesh):
    """Rigid transform taking src onto tgt when tgt is the same tessellation moved (vertex ICP)."""
    A, B = src.vertices, tgt.vertices
    tree = cKDTree(B)
    ea = np.linalg.eigh(np.cov((A - A.mean(0)).T))[1]
    eb = np.linalg.eigh(np.cov((B - B.mean(0)).T))[1]
    best = None
    for s in itertools.product([1, -1], repeat=3):
        R = eb @ np.diag(s) @ ea.T
        if np.linalg.det(R) < 0:
            continue
        T = cf.transform(R, B.mean(0) - R @ A.mean(0))
        for _ in range(60):
            _, idx = tree.query(trimesh.transform_points(A, T))
            T = kabsch(A, B[idx])
        d, _ = tree.query(trimesh.transform_points(A, T))
        if best is None or d.max() < best[1]:
            best = (T, float(d.max()))
    return best


# ---------------------------------------------------------------------------------------------
# features


def bore(mesh, cyls, diameter, what):
    c = cf.find(cyls, diameter, "hole", tol=0.06)
    if len(c) != 1:
        raise RuntimeError(f"{what}: expected one {diameter} mm bore, found {len(c)}")
    return c[0]


def pilot_pair(cyls, axis):
    """The two 1.8 mm servo pilot holes parallel to the bearing axis, SERVO_TAB_PITCH apart."""
    c = [x for x in cf.find(cyls, PILOT_D, "hole", tol=0.06) if abs(abs(x.axis @ axis) - 1) < 1e-3]
    for p, q in itertools.combinations(c, 2):
        d = perp(q.point - p.point, axis)
        if abs(np.linalg.norm(d) - SERVO_TAB_PITCH) < 0.15:
            return p, q
    raise RuntimeError("no pilot-hole pair at the servo tab pitch")


def wall_interval(mesh, point, axis, r=5.8):
    """Axial extent of the bearing wall: the interval containing the bearing centre, median over rays."""
    rows = cf.axial_intervals(mesh, point, axis, r, n_angles=16)
    lo, hi = [], []
    for _, ivs in rows:
        for a, b in ivs:
            if a <= 0.0 <= b:
                lo.append(a)
                hi.append(b)
    return float(np.median(lo)), float(np.median(hi))


def clevis(mesh, slot, r=4.0):
    """Slotted arm interval and the direction from it to the opposite clevis arm."""
    rows = cf.axial_intervals(mesh, slot.point, slot.axis, r, n_angles=16)
    arm, other = [], []
    for _, ivs in rows:
        for a, b in ivs:
            (arm if a <= 0.0 <= b else other).append((a, b))
    arm_lo, arm_hi = float(np.median([a for a, _ in arm])), float(np.median([b for _, b in arm]))
    mids = [0.5 * (a + b) for a, b in other]
    sign = 1.0 if np.median(mids) > 0 else -1.0
    return arm_lo, arm_hi, sign


def servo_features(servo):
    cyls = cf.cylinders(servo)
    spline = [c for c in cyls if c.kind == "boss" and abs(c.diameter - SPLINE_D) < 0.06][0]
    tabs = cf.find(cyls, TAB_D, "hole", tol=0.06)
    a = unit(spline.axis)
    if a @ (spline.point - servo.centroid) < 0:
        a = -a  # output points away from the body
    lo = (servo.vertices - spline.point) @ a
    base = spline.point + a * lo.min()  # the base plane, on the output axis
    tab_lo = float(min(((t.point - base) @ a) - t.length / 2 for t in tabs))
    tab_hi = float(max(((t.point - base) @ a) + t.length / 2 for t in tabs))
    x = unit(perp(np.mean([t.point for t in tabs], axis=0) - spline.point, a))  # output -> body centre
    pitch = float(np.linalg.norm(perp(tabs[0].point - tabs[1].point, a)))
    top = float(lo.max() - lo.min())
    return dict(axis=a, base=base, x=x, tab_lo=tab_lo, tab_hi=tab_hi, pitch=pitch, height=top,
                spline=spline, tabs=tabs, boss_top=None)


# ---------------------------------------------------------------------------------------------
# assembly


def derive() -> dict:
    meshes = {k: load(k) for k in FILES}
    report: dict = {"parts": {}, "joints": {}, "servos": {}, "checks": {}}

    # export rotation of the carriage
    R0 = design_rotation(meshes["carriage"])
    report["parts"]["carriage_export_rotation_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(R0) - 1) / 2, -1, 1))))
    T_design = {k: np.eye(4) for k in meshes}
    T_design["carriage"] = cf.transform(R0, np.zeros(3))
    for k in meshes:
        meshes[k] = meshes[k].copy()
        meshes[k].apply_transform(T_design[k])

    cyl = {k: cf.cylinders(meshes[k]) for k in ("carriage", "yaw_link", "mcp_link", "dip_link")}
    sv = servo_features(meshes["servo"])
    # boss top of the servo case: the 12.1 mm round boss under the spline
    boss = [c for c in cf.cylinders(meshes["servo"]) if c.kind == "boss" and abs(c.diameter - 12.1) < 0.1][0]
    sv["boss_top"] = float((boss.point - sv["base"]) @ sv["axis"] + boss.length / 2)
    report["servo"] = dict(height=sv["height"], tab_face=sv["tab_lo"], tab_top=sv["tab_hi"], boss_top=sv["boss_top"],
                           tab_pitch=sv["pitch"], spline_offset_from_tab_centre=float(np.linalg.norm(perp(
                               np.mean([t.point for t in sv["tabs"]], axis=0) - sv["spline"].point, sv["axis"]))))

    parent_info = {}
    for part in ("carriage", "yaw_link", "mcp_link"):
        m = meshes[part]
        b = bore(m, cyl[part], BEARING_D, part)
        a = unit(b.axis)
        p1, p2 = pilot_pair(cyl[part], a)
        if ((0.5 * (p1.point + p2.point)) - b.point) @ a < 0:
            a = -a  # a points from the bearing wall to the servo cavity
        lo, hi = wall_interval(m, b.point, a)
        parent_info[part] = dict(bore=b, a=a, wall=(lo, hi), pilots=(p1, p2))

    child_info = {}
    for part in ("yaw_link", "mcp_link", "dip_link"):
        m = meshes[part]
        s = bore(m, cyl[part], SLOT_D, part)
        arm_lo, arm_hi, sign = clevis(m, s)
        a = unit(s.axis) * sign  # from the slotted arm to the horn arm
        inner = arm_hi if sign > 0 else -arm_lo  # inner face, measured along a
        child_info[part] = dict(slot=s, a=a, inner=inner, arm=(arm_lo, arm_hi))

    # long directions
    def common_perp(p1, a1, p2, a2):
        n = np.cross(a1, a2)
        if np.linalg.norm(n) < 1e-6:
            return unit(perp(p2 - p1, a1)), float(np.linalg.norm(perp(p2 - p1, a1)))
        n = unit(n)
        if (p2 - p1) @ n < 0:
            n = -n
        return n, float(abs((p2 - p1) @ n))

    long_dir, spacing = {}, {}
    for part in ("yaw_link", "mcp_link"):
        s = child_info[part]["slot"]
        b = parent_info[part]["bore"]
        long_dir[part], spacing[part] = common_perp(s.point, unit(s.axis), b.point, unit(b.axis))
    # DIP: perpendicular to the PIP axis and to the pad face normal, pointing to the tip
    T_fused_dip, res_dip = register_copy(meshes["dip_link"], [p for p in meshes["tip_and_dip"].split(only_watertight=False)
                                                              if abs(p.volume - meshes["dip_link"].volume) < 1.0][0])
    T_fused_tip, res_tip = register_copy(meshes["tip"], [p for p in meshes["tip_and_dip"].split(only_watertight=False)
                                                         if abs(p.volume - meshes["tip"].volume) < 1.0][0])
    T_dip_tip = np.linalg.inv(T_fused_dip) @ T_fused_tip
    report["parts"]["tip_registration_residual_mm"] = dict(dip=res_dip, tip=res_tip)
    tip_in_dip = meshes["tip"].copy()
    tip_in_dip.apply_transform(T_dip_tip)
    s = child_info["dip_link"]["slot"]
    pip_axis = unit(s.axis)
    # long direction: the tip block's longest bounding-box axis perpendicular to the PIP axis,
    # pointing away from the PIP axis; the pad normal is the remaining box axis, towards the flat pad
    obb = tip_in_dip.bounding_box_oriented
    R_obb, ext = obb.primitive.transform[:3, :3], obb.primitive.extents
    cand = [(ext[k], R_obb[:, k]) for k in range(3) if abs(R_obb[:, k] @ pip_axis) < 0.05]
    u = unit(max(cand, key=lambda e: e[0])[1])
    if (tip_in_dip.centroid - s.point) @ u < 0:
        u = -u
    long_dir["dip_link"] = u
    pad_normal = unit(np.cross(pip_axis, u))
    # the flat pad face lies farther from the block centre than the filleted-free back face? decide
    # by the facet: the pad is the planar facet perpendicular to pad_normal with the smaller area
    faces_n = [(tip_in_dip.facets_area[i], tip_in_dip.facets_normal[i]) for i in range(len(tip_in_dip.facets))
               if abs(abs(tip_in_dip.facets_normal[i] @ pad_normal) - 1) < 1e-3]
    if faces_n:
        back = max(faces_n, key=lambda e: e[0])[1]  # the square back face is the larger of the two
        if back @ pad_normal > 0:
            pad_normal = -pad_normal
    report["parts"]["pad_normal_in_dip"] = pad_normal.round(6).tolist()
    report["parts"]["tip_block_extent_mm"] = sorted(np.round(ext, 3).tolist())

    # carriage: down = from the rail-block holes towards the bearing
    pc = parent_info["carriage"]
    rail = [c for c in cf.find(cyl["carriage"], 2.4, "hole", tol=0.06) if abs(c.axis @ pc["a"]) < 1e-3]
    down_c = unit(perp(pc["bore"].point - np.mean([c.point for c in rail], axis=0), pc["a"]))

    # joint transforms at q = 0: T_parent_child maps child design coordinates to parent design coordinates
    T = {}
    for parent, child, jn in JOINTS:
        P, C = parent_info[parent], child_info[child]
        d_p = down_c if parent == "carriage" else long_dir[parent]
        R = cf.rotation_from_frames(C["a"], long_dir[child], P["a"], d_p)
        anchor_c = C["slot"].point + C["inner"] * C["a"]
        anchor_p = P["bore"].point + P["wall"][0] * P["a"]
        T[(parent, child)] = cf.transform(R, anchor_p - R @ anchor_c)
        horn_gap = None
        report["joints"][jn] = dict(
            parent=parent, child=child,
            bearing_wall_mm=[round(P["wall"][0], 3), round(P["wall"][1], 3)],
            slotted_arm_mm=[round(C["arm"][0], 3), round(C["arm"][1], 3)],
            axis_parent=P["a"].round(6).tolist(), axis_child=C["a"].round(6).tolist())

    # servos: output coaxial with the bearing, base on the wall's inner face, tabs on the pilots
    T_servo = {}
    for parent, jn in SERVO_IN.items():
        P = parent_info[parent]
        base_p = P["bore"].point + P["wall"][1] * P["a"]
        pil_mid = 0.5 * (P["pilots"][0].point + P["pilots"][1].point)
        x_p = unit(perp(pil_mid - P["bore"].point, P["a"]))
        R = cf.rotation_from_frames(sv["axis"], sv["x"], P["a"], x_p)
        Ts = cf.transform(R, base_p - R @ sv["base"])
        T_servo[parent] = Ts
        tab_pts = [trimesh.transform_points([t.point], Ts)[0] for t in sv["tabs"]]
        miss = []
        for tp in tab_pts:
            miss.append(min(cf.line_distance(tp, P["a"], q.point, unit(q.axis)) for q in P["pilots"]))
        tab_face = float((tab_pts[0] - P["bore"].point) @ P["a"]) - 0.0
        pil_ends = [float(((q.point - P["bore"].point) @ P["a"]) + q.length / 2) for q in P["pilots"]]
        report["servos"][jn] = dict(
            parent=parent, tab_hole_to_pilot_axis_mm=[round(m, 4) for m in miss],
            servo_base_axial_mm=round(P["wall"][1], 3),
            tab_face_axial_mm=round(P["wall"][1] + sv["tab_lo"], 3),
            pilot_hole_far_ends_axial_mm=[round(e, 3) for e in pil_ends],
            boss_top_axial_mm=round(P["wall"][1] + sv["boss_top"], 3),
            spline_top_axial_mm=round(P["wall"][1] + sv["height"], 3))

    # horn stack: from the servo boss top to the child's horn-side arm, along the parent axis
    for parent, child, jn in JOINTS:
        P = parent_info[parent]
        child_m = meshes[child].copy()
        child_m.apply_transform(T[(parent, child)])
        rows = cf.axial_intervals(child_m, P["bore"].point, P["a"], 4.0, n_angles=16)
        boss_top = P["wall"][1] + sv["boss_top"]
        firsts = [iv[0] for _, ivs in rows for iv in ivs if iv[0] > P["wall"][1] + 1.0]
        report["joints"][jn]["horn_side_inner_face_axial_mm"] = round(float(np.min(firsts)), 3) if firsts else None
        report["joints"][jn]["servo_boss_top_axial_mm"] = round(boss_top, 3)
        if firsts:
            report["joints"][jn]["boss_to_horn_side_mm"] = round(float(np.min(firsts)) - boss_top, 3)
        # material of the horn-side arm along the axis, the median over the rays that see the most common pattern
        beyond = [[iv for iv in ivs if iv[0] > P["wall"][1] + 1.0] for _, ivs in rows]
        lens = [len(b) for b in beyond if b]  # rays through the horn's own hole see no material
        k = max(set(lens), key=lens.count) if lens else 0
        same = np.array([b for b in beyond if len(b) == k], float)
        report["joints"][jn]["horn_side_material_axial_mm"] = mat = np.round(np.median(same, axis=0), 3).tolist() if k else []
        # horn seat: the horn's plate lies on the arm's inner face, or in a slot between the arm's first two layers
        if len(mat) >= 2:
            report["joints"][jn]["horn_slot_axial_mm"] = [mat[0][1], mat[1][0]]
            report["joints"][jn]["horn_seat_outer_axial_mm"] = mat[1][0]
            report["joints"][jn]["horn_clearance_mm"] = round(mat[1][0] - mat[0][1] - HORN_MM, 3)
        elif mat:
            report["joints"][jn]["horn_seat_outer_axial_mm"] = mat[0][0]
        if mat:
            report["joints"][jn]["boss_to_horn_seat_mm"] = round(report["joints"][jn]["horn_seat_outer_axial_mm"] - boss_top, 3)

    # chain into the carriage frame
    T_c = {"carriage": np.eye(4)}
    T_c["yaw_link"] = T[("carriage", "yaw_link")]
    T_c["mcp_link"] = T_c["yaw_link"] @ T[("yaw_link", "mcp_link")]
    T_c["dip_link"] = T_c["mcp_link"] @ T[("mcp_link", "dip_link")]
    T_c["tip"] = T_c["dip_link"] @ T_dip_tip
    T_c["servo_yaw"] = T_servo["carriage"]
    T_c["servo_mcp"] = T_c["yaw_link"] @ T_servo["yaw_link"]
    T_c["servo_pip"] = T_c["mcp_link"] @ T_servo["mcp_link"]

    def axis_in_c(part_frame, info):
        Tm = T_c[part_frame]
        return trimesh.transform_points([info.point], Tm)[0], unit(Tm[:3, :3] @ info.axis)

    yaw_p, yaw_a = axis_in_c("carriage", parent_info["carriage"]["bore"])
    yaw_a = pc["a"]
    mcp_p, mcp_a = axis_in_c("yaw_link", parent_info["yaw_link"]["bore"])
    pip_p, pip_a = axis_in_c("mcp_link", parent_info["mcp_link"]["bore"])
    # the yaw centre: foot of the common perpendicular from the MCP axis onto the yaw axis
    n = unit(np.cross(yaw_a, mcp_a))
    A = np.column_stack([yaw_a, -mcp_a, n])
    t = np.linalg.solve(A, mcp_p - yaw_p)
    yaw_c = yaw_p + t[0] * yaw_a
    mcp_c = mcp_p + t[1] * mcp_a
    # finger mid-plane: the plane through the yaw axis perpendicular to the MCP axis
    pip_c = pip_p + ((yaw_c - pip_p) @ pip_a) * pip_a

    # finger frame F (index finger): x = yaw axis towards the dorsal side, z = up
    x_F = yaw_a
    z_F = -down_c
    R_cF = np.column_stack([x_F, np.cross(z_F, x_F), z_F])  # F axes in carriage coordinates
    T_cF = cf.transform(R_cF, yaw_c)
    T_Fc = np.linalg.inv(T_cF)
    T_F = {k: T_Fc @ v for k, v in T_c.items()}
    centres_F = {k: trimesh.transform_points([p], T_Fc)[0] for k, p in (("yaw", yaw_c), ("mcp", mcp_c), ("pip", pip_c))}
    axes_F = {"yaw": T_Fc[:3, :3] @ yaw_a, "mcp": T_Fc[:3, :3] @ mcp_a, "pip": T_Fc[:3, :3] @ pip_a}
    tip_F = meshes["tip"].copy()
    tip_F.apply_transform(T_F["tip"])
    pad_F = T_F["dip_link"][:3, :3] @ pad_normal

    report["kinematics_mm"] = dict(
        yaw_centre=centres_F["yaw"].round(4).tolist(), mcp_centre=centres_F["mcp"].round(4).tolist(),
        pip_centre=centres_F["pip"].round(4).tolist(),
        yaw_to_mcp=round(float(np.linalg.norm(centres_F["mcp"] - centres_F["yaw"])), 4),
        mcp_to_pip=round(float(np.linalg.norm(centres_F["pip"] - centres_F["mcp"])), 4),
        pip_to_tip_end=round(float(centres_F["pip"][2] - tip_F.bounds[0][2]), 4),
        axes={k: v.round(6).tolist() for k, v in axes_F.items()},
        pad_normal=pad_F.round(6).tolist(),
        tip_bounds=tip_F.bounds.round(3).tolist(),
        yaw_link_spacing_from_part=round(spacing["yaw_link"], 4), mcp_link_spacing_from_part=round(spacing["mcp_link"], 4))
    return dict(meshes=meshes, T_F=T_F, T_design=T_design, centres_F=centres_F, axes_F=axes_F, report=report,
                parent_info=parent_info, child_info=child_info)


# ---------------------------------------------------------------------------------------------
# model

# real_v1 values (assets/mjcf/real_v1/real_hand.xml), kept so the model is a drop-in candidate
MOUNTS_MM = {"thumb": (-50.0, 0.0), "index": (50.0, 55.0), "middle": (50.0, -55.0)}
MORPH_RANGE_MM = {"thumb": (30.0, 55.0), "index": (30.0, 30.0), "middle": (30.0, 30.0)}
ROM = {"yaw": (-1.483530, 1.483530), "mcp": (-0.261799, 1.605703), "pip": (-0.314159, 1.605703)}
PALM_Z_MM, PALM_PLATE_Z_MM = 62.5, 25.0
TIP_BODY_MM = 26.61  # real_v1 places <f>_tip on the finger axis this far below the PIP axis
REAL_V1_REACH_MM = 20.75 + 20.75 + 37.16  # real_v1 yaw centre to fingertip end, straight finger
# masses: SCS0009 13.1 +- 1 g (Switch Science 8042; owner, 2026-10-13: 13 g). PLA parts as printed at 15 % infill
# (owner, 2026-10-13): their PrusaSlicer filament mass from scripts/cad_v1_print_mass.py, spread uniformly over the part;
# solid PLA where print_mass.json lacks a part. The TPU tip is solid.
SERVO_MASS_KG = 0.0131
DENSITY = {"link": 1240.0, "tip": 1210.0}  # PLA and TPU, kg/m^3, solid
PRINT_MASS = OUT / "print_mass.json"
HORN_MM = 1.6  # thickness of the servo horn (owner, 2026-10-13)


def link_density(part: str) -> float:
    """kg/m^3 of a PLA part: its sliced mass over its CAD volume when print_mass.json has it, else solid PLA."""
    if PRINT_MASS.exists():
        rows = json.loads(PRINT_MASS.read_text())["parts"]
        if part in rows:
            return float(rows[part]["density_kg_m3"])
    return DENSITY["link"]
BODY_PARTS = {"mount": ["carriage", "servo_yaw"], "yaw_frame": ["yaw_link", "servo_mcp"],
              "mcp_frame": ["mcp_link", "servo_pip"], "pip_frame": ["dip_link"], "tip": ["tip"]}
PART_SOURCE = {"servo_yaw": "servo", "servo_mcp": "servo", "servo_pip": "servo"}
RGBA = {"link": "0.42 0.32 0.72 1", "servo": "0.10 0.10 0.11 1", "tip": "0.72 0.55 0.32 1"}


def part_kind(part: str) -> str:
    return "servo" if part.startswith("servo") else ("tip" if part == "tip" else "link")


def body_origins_mm(A) -> dict:
    c = A["centres_F"]
    pip = c["pip"]
    return {"mount": c["yaw"], "yaw_frame": c["yaw"], "mcp_frame": c["mcp"], "pip_frame": pip,
            "tip": pip + np.array([0.0, 0.0, -TIP_BODY_MM])}


def body_meshes(A) -> dict:
    """Each part as a trimesh in its body frame, metres."""
    org = body_origins_mm(A)
    out = {}
    for body, parts in BODY_PARTS.items():
        for p in parts:
            m = A["meshes"][PART_SOURCE.get(p, p)].copy()
            m.apply_transform(A["T_F"][p])
            m.apply_translation(-org[body])
            m.apply_scale(0.001)
            out[p] = (body, m)
    return out


def inertial(meshes_in_body: list[tuple[str, trimesh.Trimesh]]) -> dict:
    """Mass, centre of mass and full inertia of the parts of one body (body frame, SI)."""
    mass, com_acc, I_acc = 0.0, np.zeros(3), np.zeros((3, 3))
    items = []
    for p, m in meshes_in_body:
        k = part_kind(p)
        vol = float(m.volume)
        if k == "servo":
            mk, rho = SERVO_MASS_KG, SERVO_MASS_KG / vol
        else:
            rho = link_density(p) if k == "link" else DENSITY[k]
            mk = rho * vol
        mp = m.copy()
        mp.density = rho
        items.append((mk, mp.center_mass, mp.moment_inertia))  # inertia about the part's own COM
        mass += mk
        com_acc += mk * mp.center_mass
    com = com_acc / mass
    for mk, c, Ic in items:
        d = c - com
        I_acc += Ic + mk * (d @ d * np.eye(3) - np.outer(d, d))
    return dict(mass=mass, pos=com, fullinertia=[I_acc[0, 0], I_acc[1, 1], I_acc[2, 2], I_acc[0, 1], I_acc[0, 2], I_acc[1, 2]])


def fmt(v, nd=6):
    return " ".join(f"{x:.{nd}f}".rstrip("0").rstrip(".") if abs(x) > 0 else "0" for x in np.atleast_1d(v))


def finger_xml(f: str, A, inert, collision, mirror: bool, ind: str, morph: bool, pads: dict | None = None) -> str:
    """Body tree of one finger. mirror turns its geoms 180 deg about z (the thumb). collision holds the
    link spheres per body in the index finger's body frames; pads holds this finger's tip pads as
    replace_tips built them (centres already in this finger's tip frame) and their contact attributes."""
    org = body_origins_mm(A)
    q = ' quat="0 0 0 1"' if mirror else ""
    sgn = -1.0 if mirror else 1.0
    flex = "0 -1 0" if mirror else "0 1 0"

    def geoms(body):
        out = []
        for p in BODY_PARTS[body]:
            k = part_kind(p)
            out.append(f'<geom name="{f}_{p}_visual" class="cad_visual" mesh="cad_{p}" material="cad_{k}"{q}/>')
        for i, (c, r) in enumerate(collision.get(body, [])):
            cc = np.array(c) * np.array([sgn, sgn, 1.0])
            out.append(f'<geom name="{f}_{body}_s{i}" class="cad_collision" pos="{fmt(cc)}" size="{r:.6f}"/>')
        if body == "tip" and pads:
            at = " ".join(f'{k}="{v}"' for k, v in pads["attrs"].items())
            for i, c in enumerate(pads["centres"]):         # solimp per pad: builder v2 gives each pad its own d0
                out.append(f'<geom name="{f}_pad{i}" class="cad_pad" pos="{fmt(c, 7)}" size="{pads["rs"]}" {at} '
                           f'solimp="{pads["solimp"][i]}"/>')
        return out

    def inertial_xml(body):
        I = inert[body]
        pos = I["pos"] * np.array([sgn, sgn, 1.0])
        fi = np.array(I["fullinertia"])
        if mirror:  # rotation by 180 deg about z flips the sign of Ixz and Iyz
            fi = fi * np.array([1, 1, 1, 1, -1, -1])
        return f'<inertial pos="{fmt(pos)}" mass="{I["mass"]:.6f}" fullinertia="{fmt(fi, 9)}"/>'

    mx, my = MOUNTS_MM[f]
    rx, ry = MORPH_RANGE_MM[f]
    d_mcp = (org["mcp_frame"] - org["yaw_frame"]) / 1000.0
    d_pip = (org["pip_frame"] - org["mcp_frame"]) / 1000.0
    d_tip = (org["tip"] - org["pip_frame"]) / 1000.0
    pad = "  "
    L = []
    L.append(f'<body name="{f}_mount" pos="{fmt(np.array([mx, my, 0.0]) / 1000)}">')
    L.append(pad + inertial_xml("mount"))
    if morph:
        L.append(pad + f'<joint class="morph" name="{f}_x" axis="1 0 0" range="{-rx / 1000:.6f} {rx / 1000:.6f}"/>')
        L.append(pad + f'<joint class="morph" name="{f}_y" axis="0 1 0" range="{-ry / 1000:.6f} {ry / 1000:.6f}"/>')
    L += [pad + g for g in geoms("mount")]
    L.append(pad + f'<body name="{f}_yaw_frame">')
    L.append(pad * 2 + inertial_xml("yaw_frame"))
    L.append(pad * 2 + f'<joint class="ctrl" name="{f}_yaw" axis="1 0 0" range="{ROM["yaw"][0]:.6f} {ROM["yaw"][1]:.6f}"/>')
    L += [pad * 2 + g for g in geoms("yaw_frame")]
    L.append(pad * 2 + f'<body name="{f}_mcp_frame" pos="{fmt(d_mcp)}">')
    L.append(pad * 3 + inertial_xml("mcp_frame"))
    L.append(pad * 3 + f'<joint class="ctrl" name="{f}_mcp" axis="{flex}" range="{ROM["mcp"][0]:.6f} {ROM["mcp"][1]:.6f}"/>')
    L += [pad * 3 + g for g in geoms("mcp_frame")]
    L.append(pad * 3 + f'<body name="{f}_len_frame">')
    L.append(pad * 4 + '<inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>')
    if morph:  # real_v1's len slide (range 0 0 compiles unlimited, along the pad normal); kept for its qpos layout
        L.append(pad * 4 + f'<joint class="morph" name="{f}_len" axis="1 0 0" range="0 0"/>')
    L.append(pad * 4 + f'<body name="{f}_pip_frame" pos="{fmt(d_pip)}">')
    L.append(pad * 5 + inertial_xml("pip_frame"))
    L.append(pad * 5 + f'<joint class="ctrl" name="{f}_pip" axis="{flex}" range="{ROM["pip"][0]:.6f} {ROM["pip"][1]:.6f}"/>')
    L += [pad * 5 + g for g in geoms("pip_frame")]
    L.append(pad * 5 + f'<body name="{f}_tip" pos="{fmt(d_tip)}">')
    L.append(pad * 6 + inertial_xml("tip"))
    L += [pad * 6 + g for g in geoms("tip")]
    L.append(pad * 5 + "</body>")
    L.append(pad * 4 + "</body>")
    L.append(pad * 3 + "</body>")
    L.append(pad * 2 + "</body>")
    L.append(pad + "</body>")
    L.append("</body>")
    return "\n".join(ind + s for s in L)


HEADER = """<mujoco model="{model}">
  <!-- Generated by scripts/build_cad_v1_finger.py from assets/cad_v1/*.stl. Do not edit by hand. -->
  <compiler angle="radian" autolimits="true" meshdir="meshes"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
    <map znear=".01"/>
    <quality shadowsize="2048"/>
    <global elevation="-15" offwidth="1600" offheight="1200"/>
  </visual>
  <default>
    <geom friction="1.8 0.15 0.01" solref="0.006 1" solimp="0.97 0.995 0.0005"/>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
    <default class="morph">
      <joint type="slide" damping="2000" armature="0.0001"/>
    </default>
    <default class="ctrl">
      <joint type="hinge" damping="0.5" armature="0.001"/>
      <position kp="30" kv="0.5" gear="1" forcerange="-10 10" ctrlrange="-1 1"/>
    </default>
    <default class="cad_visual">
      <geom type="mesh" group="2" contype="0" conaffinity="0" mass="0"/>
    </default>
    <default class="cad_collision">
      <geom type="sphere" group="3" rgba="0.95 0.45 0.2 0.45" mass="0"/>
    </default>
    <default class="cad_pad">
      <geom type="sphere" group="4" rgba="0.2 0.75 0.35 0.6" mass="0"/>
    </default>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.15 0.15 0.15" rgb2="0.25 0.25 0.25" width="256" height="256"/>
    <material name="table_mat" texture="grid" texrepeat="3 3" reflectance="0.2"/>
    <material name="palm_mat" rgba="0.18 0.2 0.24 1"/>
    <material name="cad_link" rgba="{link}"/>
    <material name="cad_servo" rgba="{servo}"/>
    <material name="cad_tip" rgba="{tip}"/>
{meshes}
  </asset>
"""


CAD_DEFAULTS = [
    '<default class="cad_visual"><geom type="mesh" group="2" contype="0" conaffinity="0" mass="0"/></default>',
    '<default class="cad_collision"><geom type="sphere" group="3" rgba="0.95 0.45 0.2 0.45" mass="0"/></default>',
    '<default class="cad_pad"><geom type="sphere" group="4" rgba="0.2 0.75 0.35 0.6" mass="0"/></default>',
]
PAD_MU = 2.4  # reorient_backends.MU_SCENE: the real_v1 deploy scenes' finger and tool friction
# Body pairs of one finger that never collide: neighbours (their spheres overlap at every bearing) and pairs whose CAD
# parts never touch over the measured range of motion (scripts/cad_v1_interference.py: of the non-neighbours only the
# carriage and the MCP link meet, at yaw <= -79 deg, so mount/mcp_frame stays active). Excluding them keeps the
# broadphase from testing link spheres against the 1016 pads of the same finger's tip.
FINGER_EXCLUDES = [("mount", "yaw_frame"), ("mount", "pip_frame"), ("mount", "tip"), ("yaw_frame", "pip_frame"),
                   ("yaw_frame", "tip"), ("mcp_frame", "pip_frame"), ("mcp_frame", "tip")]


def cad_tip_geometry(A):
    """Point fingertip_geometry at the CAD insert (21.41 mm long, not its 22.0 mm), in this process only."""
    sys.path.insert(0, str(REPO / "scripts"))
    import fingertip_geometry as G
    tip = A["meshes"]["tip"].copy()
    tip.apply_transform(A["T_F"]["tip"])
    ext = tip.extents / 1000.0  # x: depth, y: width, z: length
    G.DEPTH, G.WIDTH, G.LENGTH = float(ext[0]), float(ext[1]), float(ext[2])
    G.CENTRE = np.array([G.REACH - G.DEPTH / 2, 0.0, -G.END + G.LENGTH / 2])
    G.HALF = np.array([G.DEPTH / 2, G.WIDTH / 2, G.LENGTH / 2])
    G.FOUNDATION = G.DEPTH / 2
    # the CAD block in the tip frame must be the block fingertip_geometry describes
    org = body_origins_mm(A)["tip"]
    lo, hi = (tip.bounds - org) / 1000.0
    want_lo = G.CENTRE * np.array([-1.0, 1, 1]) - G.HALF  # index finger: palmar face towards -x
    want_hi = G.CENTRE * np.array([-1.0, 1, 1]) + G.HALF
    err = float(max(np.abs(lo - want_lo).max(), np.abs(hi - want_hi).max()))
    if err > 1e-5:
        raise RuntimeError(f"CAD tip block differs from fingertip_geometry's block by {err * 1000:.3f} mm")
    return G


def palm_raise_m(A) -> float:
    """real_v1's keyframes were fitted to a finger 10.05 mm shorter (yaw centre to tip end 78.66 mm); the CAD hand's
    palm is raised by the difference so the fingertips start near real_v1's."""
    km = A["report"]["kinematics_mm"]
    return (km["yaw_to_mcp"] + km["mcp_to_pip"] + km["pip_to_tip_end"] - REAL_V1_REACH_MM) / 1000.0


def write_scene(A, collision: dict, out_dir: Path, inert: dict, builder: str | None = None) -> dict:
    """scene_screwdriver_medium.xml for the CAD hand: real_v1's scene with the finger bodies replaced,
    then the tips replaced by 1 mm TPU pads with reorient_backends.replace_tips (21.41 mm block; `builder` the pads'
    stiffness builder, reorient_backends.PAD_BUILDER when None)."""
    import xml.etree.ElementTree as ET
    G = cad_tip_geometry(A)
    import reorient_backends as RB
    root = ET.parse(REPO / "assets/mjcf/real_v1/scenes/scene_screwdriver_medium.xml").getroot()
    root.set("model", "cad_v1_scene_screwdriver_medium")
    comp = root.find("compiler")
    comp.set("meshdir", str((out_dir / "meshes").resolve()))
    default = root.find("default")
    for d in CAD_DEFAULTS:
        default.append(ET.fromstring(d))
    asset = root.find("asset")
    for k in ("link", "servo", "tip"):
        ET.SubElement(asset, "material", name=f"cad_{k}", rgba=RGBA[k])
    for p in body_meshes(A):
        ET.SubElement(asset, "mesh", name=f"cad_{p}", file=f"{p}.stl")
    palm = root.find(".//body[@name='palm_pose']")
    for f in ("thumb", "index", "middle"):
        palm.remove(palm.find(f"body[@name='{f}_mount']"))
        palm.append(ET.fromstring(finger_xml(f, A, inert, collision, mirror=(f == "thumb"), ind="", morph=True)))
    contact = root.find("contact")
    for f in ("thumb", "index", "middle"):
        ET.SubElement(contact, "exclude", body1="palm_pose", body2=f"{f}_mount")
        have = {(e.get("body1"), e.get("body2")) for e in contact.findall("exclude")}
        for a, b in FINGER_EXCLUDES:
            if (f"{f}_{a}", f"{f}_{b}") not in have:
                ET.SubElement(contact, "exclude", body1=f"{f}_{a}", body2=f"{f}_{b}")
    # the palm is raised by palm_raise_m; open_ik needs a refit
    dz = palm_raise_m(A)
    kf = root.find("keyframe")
    for key in list(kf):
        if key.get("name") == "open_ik":  # a grasp fitted to real_v1's fingers; on these it presses the pads into the tool
            kf.remove(key)
    for key in root.iter("key"):
        qpos = [float(x) for x in key.get("qpos").split()]
        ctrl = [float(x) for x in key.get("ctrl").split()]
        qpos[9] += dz  # object 7, then palm px, py, pz
        ctrl[2] += dz  # a_palm_px, a_palm_py, a_palm_pz
        key.set("qpos", " ".join(f"{x:.6g}" for x in qpos))
        key.set("ctrl", " ".join(f"{x:.6g}" for x in ctrl))
    _, meta = RB.replace_tips(root, "tpu2.7", "pads", PAD_MU, out_dir, tool=RB.OBJ,
                              **({"builder": builder} if builder else {}))
    meta["palm_raised_mm"] = round(dz * 1000, 3)
    # pads in geom group 4 as in finger.xml and hand.xml, so the viewers toggle them with the same key
    for g in root.iter("geom"):
        if any(g.get("name", "").startswith(f"{f}_pad") for f in ("thumb", "index", "middle")):
            g.set("group", "4")
    # replace_tips compiles with absolute paths; the written scene resolves its meshes relative to itself
    comp.set("meshdir", "meshes")
    mdir = (out_dir / "meshes").resolve()
    for mesh in asset.findall("mesh"):
        f = Path(mesh.get("file"))
        if f.is_absolute():
            mesh.set("file", str(f.resolve().relative_to(mdir)))
    ET.indent(root, "  ")
    (out_dir / "scene_screwdriver_medium.xml").write_text(ET.tostring(root, encoding="unicode") + "\n")
    pads = {}
    for f in ("thumb", "index", "middle"):
        tipb = root.find(f".//body[@name='{f}_tip']")
        gs = [g for g in tipb.findall("geom") if g.get("name", "").startswith(f"{f}_pad")]
        pads[f] = dict(centres=[np.array([float(x) for x in g.get("pos").split()]) for g in gs], rs=gs[0].get("size"),
                       attrs={k: gs[0].get(k) for k in ("condim", "priority", "friction", "solref")},
                       solimp=[g.get("solimp") for g in gs])
    meta["block_mm"] = [G.DEPTH * 1000, G.WIDTH * 1000, G.LENGTH * 1000]
    return dict(pads=pads, meta=meta)


def write_model(A, collision: dict, out_dir: Path, with_pads: bool = True, builder: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meshes").mkdir(exist_ok=True)
    bm = body_meshes(A)
    for p, (body, m) in bm.items():
        m.export(out_dir / "meshes" / f"{p}.stl")
    inert = {b: inertial([(p, m) for p, (bb, m) in bm.items() if bb == b]) for b in BODY_PARTS}
    scene = write_scene(A, collision, out_dir, inert, builder) if with_pads else dict(pads={}, meta={})
    pads = scene["pads"]
    mesh_xml = "\n".join(f'    <mesh name="cad_{p}" file="{p}.stl"/>' for p in bm)
    head = HEADER.format(model="{model}", link=RGBA["link"], servo=RGBA["servo"], tip=RGBA["tip"], meshes=mesh_xml)

    # one finger hanging from a fixed carriage, for inspection
    fing = finger_xml("index", A, inert, collision, mirror=False, ind="    ", morph=False, pads=pads.get("index"))
    fing = fing.replace('<body name="index_mount" pos="0.05 0.055 0">', '<body name="index_mount" pos="0 0 0.12">')
    single = head.replace("{model}", "cad_v1_finger") + f"""  <worldbody>
    <light pos="0 -0.3 0.5" dir="0 0.5 -1" directional="true" diffuse=".6 .6 .6" specular=".2 .2 .2"/>
    <geom name="floor" type="plane" size="0.3 0.3 0.01" material="table_mat" contype="0" conaffinity="0"/>
{fing}
  </worldbody>
  <contact>
{chr(10).join(f'    <exclude body1="index_{a}" body2="index_{b}"/>' for a, b in FINGER_EXCLUDES)}
  </contact>
  <actuator>
    <position class="ctrl" name="a_index_yaw" joint="index_yaw" ctrlrange="{ROM['yaw'][0]:.6f} {ROM['yaw'][1]:.6f}"/>
    <position class="ctrl" name="a_index_mcp" joint="index_mcp" ctrlrange="{ROM['mcp'][0]:.6f} {ROM['mcp'][1]:.6f}"/>
    <position class="ctrl" name="a_index_pip" joint="index_pip" ctrlrange="{ROM['pip'][0]:.6f} {ROM['pip'][1]:.6f}"/>
  </actuator>
  <keyframe>
    <key name="straight" qpos="0 0 0" ctrl="0 0 0"/>
    <key name="open" qpos="0 0.55 0.55" ctrl="0 0.55 0.55"/>
    <key name="curled" qpos="0 1.4 1.4" ctrl="0 1.4 1.4"/>
    <key name="yawed" qpos="0.8 0.55 0.55" ctrl="0.8 0.55 0.55"/>
  </keyframe>
</mujoco>
"""
    (out_dir / "finger.xml").write_text(single)

    # three fingers on the real_v1 mounts and palm plate
    bodies = []
    for f in ("thumb", "index", "middle"):
        bodies.append(finger_xml(f, A, inert, collision, mirror=(f == "thumb"), ind="      ", morph=True, pads=pads.get(f)))
    excl = []
    for f in ("thumb", "index", "middle"):
        for b in ("mount", "yaw_frame", "mcp_frame", "len_frame", "pip_frame", "tip"):
            excl.append(f'    <exclude body1="palm_pose" body2="{f}_{b}"/>')
        excl += [f'    <exclude body1="{f}_{a}" body2="{f}_{b}"/>' for a, b in FINGER_EXCLUDES]
    acts = []
    for f in ("thumb", "index", "middle"):
        for j in ("yaw", "mcp", "pip"):
            acts.append(f'    <position class="ctrl" name="a_{f}_{j}" joint="{f}_{j}" ctrlrange="{ROM[j][0]:.6f} {ROM[j][1]:.6f}"/>')
    key = " ".join(["0 0 0 0.55 0 0.55"] * 3)
    hand = head.replace("{model}", "cad_v1_hand") + f"""  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true" diffuse=".6 .6 .6" specular=".2 .2 .2"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.1" material="table_mat" contype="0" conaffinity="0"/>
    <body gravcomp="1" name="palm_pose" pos="0 0 {PALM_Z_MM / 1000 + palm_raise_m(A):.6f}">
      <geom name="palm_plate" type="box" pos="0 0 {PALM_PLATE_Z_MM / 1000:.4f}" size="0.0850 0.0900 0.0015" material="palm_mat"/>
      <site name="workspace_thumb" type="box" pos="-0.050 0 0.0280" size="0.030 0.055 0.0005" rgba="0.85 0.30 0.30 0.14"/>
      <site name="workspace_index" type="box" pos="0.050 0.055 0.0280" size="0.030 0.030 0.0005" rgba="0.30 0.55 0.85 0.14"/>
      <site name="workspace_middle" type="box" pos="0.050 -0.055 0.0280" size="0.030 0.030 0.0005" rgba="0.30 0.75 0.45 0.14"/>
{chr(10).join(bodies)}
    </body>
  </worldbody>
  <contact>
{chr(10).join(excl)}
  </contact>
  <actuator>
{chr(10).join(acts)}
  </actuator>
  <keyframe>
    <key name="open" qpos="{key}" ctrl="0 0.55 0.55 0 0.55 0.55 0 0.55 0.55"/>
  </keyframe>
</mujoco>
"""
    (out_dir / "hand.xml").write_text(hand)
    return inert, scene["meta"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="derive and print the assembly report only")
    ap.add_argument("--spheres", type=Path, default=None,
                    help="JSON of collision spheres per body, {body: [[x, y, z, r], ...]} in metres, body frames")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--builder", default=None, choices=["v1", "v2"],
                    help="pad stiffness builder (reorient_backends.replace_tips): v2 per-pad E a/h with the local depth "
                         "(default, reorient_backends.PAD_BUILDER), v1 one stiffness per finger as before 2026-10-15")
    args = ap.parse_args()
    A = derive()
    print(json.dumps(A["report"], indent=1))
    if args.report:
        return
    collision = {}
    if args.spheres:
        raw = json.loads(args.spheres.read_text())
        collision = {b: [(s[:3], s[3]) for s in v] for b, v in raw.items()}
    inert, pad_meta = write_model(A, collision, args.out, builder=args.builder)
    asm = dict(report=A["report"], body_origins_mm={k: v.round(4).tolist() for k, v in body_origins_mm(A).items()},
               T_F_from_part_design_mm={k: np.round(v, 6).tolist() for k, v in A["T_F"].items()},
               T_part_design_from_stl={k: np.round(v, 9).tolist() for k, v in A["T_design"].items()},
               inertia={b: dict(mass_kg=round(I["mass"], 6), com_m=np.round(I["pos"], 6).tolist(),
                                fullinertia=[float(f"{x:.4e}") for x in I["fullinertia"]]) for b, I in inert.items()},
               masses_note=("SCS0009 13.1 g (Switch Science 8042; owner 13 g); PLA parts as sliced at 15 % infill "
                            "(print_mass.json, scripts/cad_v1_print_mass.py) where present, else 1240 kg/m^3; tip TPU 1210 kg/m^3, solid"),
               link_density_kg_m3={p: link_density(p) for p in ("carriage", "yaw_link", "mcp_link", "dip_link")},
               pads=pad_meta, spheres_file=str(args.spheres) if args.spheres else None)
    (args.out / "assembly.json").write_text(json.dumps(asm, indent=1))
    print(f"wrote {args.out}/finger.xml, hand.xml, scene_screwdriver_medium.xml, assembly.json, meshes/")


if __name__ == "__main__":
    main()
