#!/usr/bin/env python3
"""Fingertip holder for the UR5 cell (owner, 2026-10-27): two fingertips (pad, cassette 3T, top cover TW15H3 trimmed
flush with the walls, and the ReSkin 5X board) held still while v2 ID 1 moves around them without contact, to measure the
crosstalk between tips at known poses; the held tips' boards stream too. Each tip's four M2 screws pass through the cassette
and the board and form their threads in 1.9 mm holes that pass through a station block. The two blocks stand opposite each
other on arms from one column (a T): on the -y station the tip faces down (side by side), on the +y station it faces up (the
pads facing), so a night run needs no change of tips. The board lies between the block and the cassette as on the finger's
tongue, so each tip stands its thickness off the block, and its tab with J1, the plug and the Qwiic cable stands out past
the block's end. The homing seat is the tall indenter's (docs/experiments/20261026-tall_indenter) with taller and closer
walls, a stop at the fork's palmar end and a key with rounded edges, on a tower of its own. The numbers are
ur5_cb3_cell.TIPHOLD, TIP_F and HELD_BOARD, which the cell's clearance checks read as well.

    E=~/miniconda3/envs/femtip/bin/python
    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --"
    $RG $E scripts/fingertip_holder.py build                    # STL and parts.json
    $RG env -u PYTHONPATH -u VIRTUAL_ENV PATH=/usr/bin:/bin FHOLD_CMD=step FHOLD_REPO=$PWD freecadcmd scripts/fingertip_holder.py
    $RG .venv/bin/python scripts/fingertip_holder.py boards      # the held boards in parts.json (chips, J1, screws)
    $RG env UR5_MOUNT=v7 .venv/bin/python scripts/fingertip_holder.py reach --test facing|side
    $RG env UR5_MOUNT=v7 .venv/bin/python scripts/fingertip_holder.py homing
    $RG env UR5_MOUNT=v7 .venv/bin/python scripts/fingertip_holder.py farfield
    $RG env UR5_MOUNT=v7 .venv/bin/python scripts/fingertip_holder.py figures
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:                                   # freecadcmd executes the file without __file__
    REPO = Path(os.environ["FHOLD_REPO"])
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "docs/experiments/20261027-fingertip_holder"
CAD = OUT / "cad"
STEM = "20261027-fingertip_holder"
PARTS_JSON = OUT / f"{STEM}_parts.json"
APEX_F = np.array([-10.55, 0.0, -87.91])            # the pad's apex, the centre of its face, in F
T_PF = np.array([[-1.0, 0, 0, -2.05], [0, 1.0, 0, 0], [0, 0, -1.0, -77.05], [0, 0, 0, 1.0]])   # F -> the pad frame P
TESTS = dict(facing=1, side=0)                      # the station each test uses


# ------------------------------------------------------------------------------------------------ geometry
def geometry():
    """The resolved geometry in I (mm): everything the CAD needs, as plain numbers."""
    import ur5_cb3_cell as CELL
    t = dict(CELL.TIPHOLD)
    g = CELL.tiphold_parts(t)
    f = g.pop("frame")
    g["mouth"] = np.asarray(g["mouth"]).tolist()
    return dict(params=t, tip_F=CELL.TIP_F, **g, hole_d=1.9, key_ring=len(g["key"]) // 2,
                fillet=dict(r0=t["col_r"] + 6.0, h=6.0),
                screws=[(30.0, -20.0), (-30.0, -20.0), (30.0, 75.0), (-30.0, 30.0)], screw_hole=4.6, screw_cb=(9.0, 3.0),
                contact_home=f["contact"].tolist(), T_IM_home=f["T_IM_home"].round(4).tolist())


def _loft(K, pts, n):
    """The convex solid between two rings of n points each (the key with its rounded edges)."""
    if getattr(K, "name", "") == "brep":
        wires = [K.Part.makePolygon([K.V(q) for q in ring] + [K.V(ring[0])]) for ring in (pts[:n], pts[n:])]
        return K.Part.makeLoft(wires, True, True)
    return K.hull(pts)


def solid(K, g):
    """The holder as one solid with kernel K (scripts/ur5_calibration_mount.py MeshK or BrepK)."""
    t = g["params"]
    x0, x1, y0, y1, tb = t["base"]
    base = K.box((x0, y0, 0.0), (x1, y1, tb))
    d, (cbd, cbh) = g["screw_hole"], g["screw_cb"]
    holes = []
    for x, y in g["screws"]:
        holes += [K.cyl("z", (x, y), d / 2, -1.0, tb + 1.0), K.cyl("z", (x, y), cbd / 2, tb - cbh, tb + 1.0)]
    base = K.cut_all(base, holes)
    zb, zt = t["z_bot"], g["z_top"]
    parts = [base]
    for st in g["stations"]:
        s = K.union([K.box(*st["block"]), K.box(*st["arm"]), K.hull(st["gusset"])])
        parts.append(K.cut_all(s, [K.cyl("z", (x, y), g["hole_d"] / 2, zb - 1.0, zt + 1.0) for x, y in st["holes"]]))
    parts += [K.cyl("z", (0.0, 0.0), g["fillet"]["r0"], tb - 0.01, tb + g["fillet"]["h"], r1=t["col_r"]),
              K.cyl("z", (0.0, 0.0), t["col_r"], tb - 0.01, zt)]
    parts += [K.box(*g["block"])] + [K.box(lo, hi) for lo, hi in g["lands"] + g["walls"]]
    parts += [K.hull(g["strut"]), _loft(K, g["key"], g["key_ring"])]
    return K.union(parts)


def cmd_build(a):
    os.environ.setdefault("UR5M_REV", "v7")
    import ur5_calibration_mount as UM
    g = geometry()
    CAD.mkdir(parents=True, exist_ok=True)
    tm = UM.to_trimesh(solid(UM.MeshK(), g))
    dst = CAD / f"{STEM}.stl"
    tm.export(dst)
    vol = float(tm.volume)
    rec = dict(geometry=g, stl=str(dst.relative_to(REPO)), volume_cm3=round(vol / 1e3, 1),
               mass_g_solid=round(vol / 1e3 * 1.24, 0), bounds_mm=np.round(tm.bounds, 2).tolist(),
               watertight=bool(tm.is_watertight), when=time.strftime("%Y-%m-%d %H:%M"))
    if PARTS_JSON.exists() and "held_boards" in json.loads(PARTS_JSON.read_text()):
        rec["held_boards"] = json.loads(PARTS_JSON.read_text())["held_boards"]
    PARTS_JSON.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("geometry", "held_boards")}, indent=1))


# ------------------------------------------------------------------------------------------------ the held tips' boards
def _first_hit_x(m, y, z, r, n=12, x0=-20.0):
    """Least x_F at which rays along +x_F, on a circle of radius r about the line (y, z), first enter mesh m."""
    tri = m.triangles
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    d = np.array([1.0, 0.0, 0.0])
    p = np.cross(d, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    best = np.inf
    for a in np.radians(np.arange(n) * 360.0 / n):
        s = np.array([x0, y + r * math.cos(a), z + r * math.sin(a)]) - v0
        u = (s * p).sum(1) * inv
        q = np.cross(s, e1)
        v = (q @ d) * inv
        t = (e2 * q).sum(1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        if hit.any():
            best = min(best, x0 + float(t[hit].min()))
    return best


def held_boards():
    """The held tips' boards in I (mm): the FR-4, J1, its plug and the cable's start as boxes, the five chips' sensing
    points and the board's axes, the tip frame P, where J1 stands past the station block, and the M2 x 8 screws' reach."""
    import ur5_cb3_cell as CELL
    b = CELL.HELD_BOARD
    g = CELL.tiphold_parts()
    T_FP = np.linalg.inv(T_PF)
    head_r = 1.6                                    # between the 2.4 mm hole and the cheese head's 3.8 mm
    seats = []
    for y, z in CELL.TIP_F["holes"]:
        x = min(_first_hit_x(CELL.part_F(n), y, z, head_r) for n in ("cover", "cap"))
        seats.append(dict(y_F=y, z_F=z, head_seat_x_F=round(x, 3), tip_x_F=round(x + 8.0, 3),
                          into_block_mm=round(x + 8.0 - (b["x_comp"] + b["t"]), 2),
                          into_block_without_board_mm=round(x + 8.0 - b["x_comp"], 2)))
    out = dict(
        source="ur5_cb3_cell.HELD_BOARD: outline, J1 and chips from the ReSkin 5X fabrication files "
               "(docs/experiments/20261018-dip_link_board_mount/spec.json, board); J1's header, the mated SHR-04V-S-B and the "
               "cable's first 3 mm as dip_link_concept_a.j1_parts_board, the cable as a flat 4 x 1 mm ribbon; FR-4 0.6 mm "
               "nominal (ReSkin README, rigid order), not measured on the owner's boards",
        thickness_mm=b["t"], board_frame="u = y_F, v = z_u1 - z_F (U2 toward the tip's end, the tab toward the PIP joint), "
                                         "w out of the component face toward the pad; x_F = x_comp - w",
        sensing_points="0.42 mm under each MLX90393's top, the package 0.9 mm tall on the component face (x_F 3.32)",
        screws=dict(length_mm=8.0, holes=seats), stations=[])
    for i, st in enumerate(g["stations"]):
        T = CELL.tiphold_T_IF(i=i)
        R = T[:3, :3]
        boxes = {}
        for name, lo, hi in CELL.held_board_boxes_F():
            c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            cI = CELL.C.apply(T, c)
            boxes[name] = [np.round(cI.min(0), 3).tolist(), np.round(cI.max(0), 3).tolist()]
        (bx0, _, bz0), (bx1, _, bz1) = st["block"]
        j1x = (boxes["j1"][0][0], boxes["j1"][1][0])
        out["stations"].append(dict(
            y_s=st["y_s"], mode=st["mode"], T_IF=np.round(T, 4).tolist(), T_IP=np.round(T @ T_FP, 4).tolist(),
            chips_I={u: np.round(CELL.C.apply(T, p[None])[0], 3).tolist() for u, p in CELL.held_chips_F().items()},
            board_axes_I=dict(u=np.round(R @ [0.0, 1.0, 0.0], 3).tolist(), v=np.round(R @ [0.0, 0.0, -1.0], 3).tolist(),
                              w=np.round(R @ [-1.0, 0.0, 0.0], 3).tolist()),
            boxes_I=boxes,
            block_I=dict(x=[round(bx0, 3), round(bx1, 3)], z=[round(bz0, 3), round(bz1, 3)]),
            past_block_end_mm=dict(
                end=("-x_I (toward the seat tower and the robot)" if boxes["pcb_tab"][0][0] < bx0 else "+x_I"),
                tab=round(bx0 - boxes["pcb_tab"][0][0], 2), j1_from=round(bx0 - j1x[1], 2), j1_to=round(bx0 - j1x[0], 2),
                plug_rear_face=round(bx0 - boxes["plug"][0][0], 2), cable_stub=round(bx0 - boxes["cable"][0][0], 2)),
            connector_z_I=[min(boxes[k][0][2] for k in ("j1", "plug")), max(boxes[k][1][2] for k in ("j1", "plug"))],
            apex_I=np.round(held_apex(CELL, i), 3).tolist()))
    return out


def cmd_boards(a):
    rec = json.loads(PARTS_JSON.read_text())
    rec["held_boards"] = held_boards()
    rec["held_boards"]["when"] = time.strftime("%Y-%m-%d %H:%M")
    PARTS_JSON.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec["held_boards"], indent=1))


def freecad_step():
    """Inside freecadcmd: the exact solid from parts.json, written as STEP and read back."""
    os.environ.setdefault("UR5M_REV", "v7")
    import FreeCAD
    import Import
    import Part
    import ur5_calibration_mount as UM
    g = json.loads(PARTS_JSON.read_text())["geometry"]
    so = solid(UM.BrepK(), g)
    dst = CAD / f"{STEM}.step"
    doc = FreeCAD.newDocument("fhold")
    o = doc.addObject("Part::Feature", "fingertip_holder")
    o.Shape = so
    Import.export([o], str(dst))
    FreeCAD.closeDocument("fhold")
    back = Part.read(str(dst))
    rec = json.loads(PARTS_JSON.read_text())
    rec["step"] = dict(file=str(dst.relative_to(REPO)), valid=bool(so.isValid()), solids=len(so.Solids),
                       volume_mm3=round(so.Volume, 1), readback_valid=bool(back.isValid()),
                       readback_volume_mm3=round(back.Volume, 1))
    PARTS_JSON.write_text(json.dumps(rec, indent=1))
    print(rec["step"])


# ------------------------------------------------------------------------------------------------ poses of v2 ID 1
def T_IF_moving(p, tw):
    """v2 ID 1's frame F in I with its apex at p, pad down, turned tw deg about the vertical (0: its tip toward +x_I)."""
    th = math.radians(tw)
    R = np.array([[0.0, -math.sin(th), -math.cos(th)], [0.0, math.cos(th), -math.sin(th)], [1.0, 0.0, 0.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p, float) - R @ APEX_F
    return T


def held_apex(CELL, i):
    return (CELL.tiphold_T_IF(i=i) @ np.r_[APEX_F, 1.0])[:3]


def test_poses(test, apex):
    """Facing (the +y station, tip up): v2 ID 1's apex W mm over the held tip's apex, turned tw about the common normal.
    Side by side (the -y station, tip down): v2 ID 1 at -y of the held tip with g mm between the pads' flanks (14.8 mm wide),
    moved s mm toward its tip end, turned tw about the vertical through its apex."""
    out = []
    if test == "facing":
        for W in (0.5, 1, 2, 3, 5, 10, 15, 20, 30):
            for tw in (-20, -10, 0, 10, 20):
                out.append(dict(W=W, tw=tw, p=apex + np.array([0.0, 0.0, W])))
    else:
        for g in (9, 10, 11, 12, 13, 14, 15, 20, 25):
            for s in (0, 2, 5):
                for tw in (-20, -10, 0, 10):
                    out.append(dict(g=g, s=s, tw=tw, p=apex + np.array([s, -(14.8 + g), 0.0])))
    return out


def _cell():
    import tall_indenter as TI
    CELL, C, geo, reg = TI._setup("sphere", None)      # cb3t's registration: the holder's base where the sphere's was
    return TI, CELL, C, reg


def cmd_reach(a):
    TI, CELL, C, reg = _cell()
    S = CELL.SIM
    qh = np.asarray(reg.q_home, float)
    apex = held_apex(CELL, TESTS[a.test])
    rows, t0 = [], time.time()
    for k in test_poses(a.test, apex):
        T_IM = T_IF_moving(k.pop("p"), k["tw"]) @ np.linalg.inv(CELL.T_M_FROM_F)
        T_BE = reg.T_BI @ T_IM @ C.inv(reg.T_EM)
        ok, worst, j = TI._pose_ok(CELL, C, S, reg, "tiphold", T_BE, None, qh)
        k.update(ok=bool(ok), joints_56=None if j is None else [round(float(v), 1) for v in j],
                 clear=None if worst is None else {n: round(float(v), 2) for n, v in worst.items()})
        rows.append(k)
    out = OUT / "data" / f"{STEM}_reach_{a.test}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(test=a.test, station=TESTS[a.test], held_apex_I=apex.round(2).tolist(), J5=TI.J5,
                                   J6=TI.J6, need=TI.NEED, registration=str(TI.REG.relative_to(REPO)), poses=rows,
                                   seconds=round(time.time() - t0, 1)), indent=0))
    key = ("W", "tw") if a.test == "facing" else ("g", "s", "tw")
    for r in rows:
        print(" ".join(f"{n} {r[n]:>4}" for n in key), "ok" if r["ok"] else "--", r["joints_56"], r["clear"])
    print(f"{sum(r['ok'] for r in rows)}/{len(rows)} poses pass -> {out.relative_to(REPO)} in {time.time() - t0:.0f} s")


def seat_check(CELL, C, reg, T_BE):
    """Least distance (mm) from the fork, clamp plate and DIP link at homing to each feature of the seat, from surface
    samples 0.3 mm apart, with the slotted arm's M3 head and DIN 9021 washer (9 mm across) as a cylinder at the PIP axis."""
    import trimesh
    g = CELL.tiphold_parts()
    T_IM = C.inv(reg.T_BI) @ T_BE @ reg.T_EM
    feats = dict(lands=g["lands"], walls_low_z=g["walls"][:2], walls_high_z=g["walls"][2:4], stop=g["walls"][4:])
    pts = {}
    for name in ("fork", "clamp_plate", "dip_link"):
        m = CELL.finger_mesh(name)
        p, _ = trimesh.sample.sample_surface_even(m, int(min(60000, m.area / 0.3 ** 2)), seed=2)
        pts[name] = C.apply(T_IM, p)
    w = trimesh.creation.cylinder(radius=4.5, height=3.8, sections=36)      # washer and head from the arm's face at 17.4
    w.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    w.apply_translation([0.0, 17.4 + 1.9, 57.0])
    p, _ = trimesh.sample.sample_surface_even(w, 3000, seed=2)
    pts["arm_screw"] = C.apply(T_IM, p)
    out = {}
    for fname, boxes in feats.items():
        for pname, P in pts.items():
            out[f"{fname}~{pname}"] = round(min(float(CELL.sd_box(P, lo, hi).min()) for lo, hi in boxes), 3)
    for pname, P in pts.items():
        out[f"key~{pname}"] = round(float(CELL._convex_sdf(P, g["key"]).min()), 3)
    return out


def cmd_homing(a):
    """The homing pose on the holder: IK on cb3t's branch, joints, the clearances of each group and the seat check."""
    TI, CELL, C, reg = _cell()
    T_IM = CELL.tiphold_parts()["frame"]["T_IM_home"]
    T_BE = reg.T_BI @ T_IM @ C.inv(reg.T_EM)
    q = CELL.SIM.ik(T_BE, np.asarray(reg.q_home, float))
    cl = CELL.clearance(q, T_BE, reg.T_BI, reg.T_EM, "tiphold")
    worst = {}
    for k, v in cl.items():
        grp = next((gname for gname, names in TI.GROUPS.items() if k in names), "robot")
        worst[grp] = round(min(worst.get(grp, 1e9), v), 2)
    rec = dict(q_deg=np.degrees(q).round(1).tolist(), clear=worst, self=CELL.self_clearance(q, reg.T_EM),
               flange_B=np.round(T_BE[:3, 3], 1).tolist(), seat=seat_check(CELL, C, reg, T_BE))
    print(json.dumps(rec, indent=1, default=float))
    p = OUT / "data" / f"{STEM}_homing.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1, default=float))


def cmd_farfield(a):
    """The field (uT, the norm over v2 ID 1's 15 chip components, as in the crosstalk budget) of the held tip under test
    and of the other station's tip, at the test poses that pass with s 0 and tw 0, for a five-magnet and a four-magnet pad
    on the held tips (the budget's model of v2 ID 1, scripts/fingertip_crosstalk.py)."""
    import fingertip_crosstalk as X
    import ur5_cb3_cell as CELL
    W = X.W
    cal = W.Calib.load(X.V2 / "calib_rest.json")
    model = W.Model(cal)
    srcs = {"five": X.source(X.five_row(), cal.poles, cal.k), "four": X.source(X.four_row(), X.FOUR_POLES, cal.k)}
    T_FP = np.linalg.inv(T_PF)
    rows = []
    for test, near in TESTS.items():
        far = 1 - near
        apex = held_apex(CELL, near)
        reach = json.loads((OUT / "data" / f"{STEM}_reach_{test}.json").read_text())["poses"]
        for r in reach:
            if not r["ok"] or r["tw"] != 0 or r.get("s", 0) != 0:
                continue
            p = apex + (np.array([0.0, 0.0, r["W"]]) if test == "facing" else np.array([0.0, -(14.8 + r["g"]), 0.0]))
            T_IP = T_IF_moving(p, 0.0) @ T_FP
            rec = dict(test=test, gap_mm=r.get("W", r.get("g")))
            for which, i in (("near", near), ("far", far)):
                T_rel = np.linalg.inv(T_IP) @ CELL.tiphold_T_IF(i=i) @ T_FP
                for sname, src in srcs.items():
                    B = X.reading(model, src, T_rel[:3, :3], T_rel[:3, 3])
                    rec[f"{which}_{sname}_uT"] = round(float(np.linalg.norm(B)), 2)
            rows.append(rec)
            print(rec)
    p = OUT / "data" / f"{STEM}_farfield.json"
    p.write_text(json.dumps(dict(model="fingertip_crosstalk budget: v2 ID 1 calib_rest, five N52 (its poles) or four "
                                       "N52 in a 5 mm square", rows=rows), indent=1))
    print("->", p.relative_to(REPO))


# ------------------------------------------------------------------------------------------------ figure
BOARD_COLOURS = dict(pcb="#228833", pcb_tab="#228833", j1="#e8e8e8", plug="#f4f4f4", cable="#555555")


def board_meshes(CELL, T_IF):
    """A held tip's board (FR-4, J1, plug, cable's start) as meshes in I, faces at most 1 mm across for the clip."""
    import trimesh
    out = []
    for name, lo, hi in (CELL.held_board_boxes_F() if CELL.HELD_BOARD else []):
        m = trimesh.creation.box(extents=np.subtract(hi, lo))
        m.apply_translation((np.asarray(lo) + np.asarray(hi)) / 2)
        m = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=1.0))
        m.apply_transform(T_IF)
        out.append((m, BOARD_COLOURS[name]))
    return out


def cmd_figures(a):
    """The holder with both held tips; the pads facing 2 mm apart on the +y station; side by side at the least gap that
    passes with s 0 and tw 0 on the -y station. v2 ID 1's finger in blue with its fork cut at z_M 60, the held tips in
    orange with their boards in green and J1 with its plug in white."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import ur5_cb3_cell as CELL
    hold = trimesh.load(CAD / f"{STEM}.stl", force="mesh")
    hold = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(hold.vertices, hold.faces, max_edge=3.0))   # for the clip
    side = json.loads((OUT / "data" / f"{STEM}_reach_side.json").read_text())["poses"]
    g_min = min(r["g"] for r in side if r["ok"] and r["s"] == 0 and r["tw"] == 0)
    plt.rcParams.update({"font.family": "serif", "font.size": 13})
    fig = plt.figure(figsize=(17, 6.4))
    panels = ((None, None, "The holder and both held tips", (-45, 45), (-45, 100), (0, 80), (24, -130)),
              ("facing", 2.0, "Pads facing, 2 mm apart", (-45, 35), (-10, 70), (20, 100), (14, -28)),
              ("side", g_min, f"Side by side, {g_min:g} mm between the pads", (-45, 35), (-85, -5), (0, 80), (14, -28)))
    for k, (test, gap, ttl, xlim, ylim, zlim, (el, az)) in enumerate(panels):
        meshes = [(hold, "#9a9a9a")]
        for i in range(len(CELL.TIPHOLD["stations"])):
            T_IF = CELL.tiphold_T_IF(i=i)
            for name in ("cover", "cap", "pad"):
                m = CELL.part_F(name)
                m.apply_transform(T_IF)
                meshes.append((m, "#ee7733"))
            meshes += board_meshes(CELL, T_IF)
        if test is not None:
            apex = held_apex(CELL, TESTS[test])
            p = apex + (np.array([0.0, 0.0, gap]) if test == "facing" else np.array([0.0, -(14.8 + gap), 0.0]))
            T_IM = T_IF_moving(p, 0.0) @ np.linalg.inv(CELL.T_M_FROM_F)
            for name in ("dip_link", "cover", "cap", "pad", "fork"):
                m = CELL.finger_mesh(name)
                if name == "fork":
                    m = trimesh.Trimesh(m.vertices, m.faces[(m.vertices[m.faces][:, :, 2] > 60.0).all(axis=1)])
                m.apply_transform(T_IM)
                meshes.append((m, "#4477aa" if name != "fork" else "#9fb8d6"))
        tri = np.vstack([m.vertices[m.faces] for m, _ in meshes])
        nrm = np.vstack([m.face_normals for m, _ in meshes])
        rgb = np.vstack([np.tile(matplotlib.colors.to_rgb(c), (len(m.faces), 1)) for m, c in meshes])
        lo, hi = np.array([xlim[0], ylim[0], zlim[0]]), np.array([xlim[1], ylim[1], zlim[1]])
        keep = ((tri >= lo) & (tri <= hi)).all(axis=(1, 2))          # faces inside the panel's box
        tri, nrm, rgb = tri[keep], nrm[keep], rgb[keep]
        light = np.clip(nrm @ np.array([0.45, -0.45, 0.77]), 0.12, 1.0)
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tri, edgecolor="none", facecolors=np.clip(rgb * (0.4 + 0.6 * light[:, None]), 0, 1)))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
        ax.view_init(el, az)
        ax.set_xlabel("x_I (mm)", labelpad=8); ax.set_ylabel("y_I (mm)", labelpad=8); ax.set_zlabel("z_I (mm)", labelpad=6)
        ax.set_title(ttl)
    fig.tight_layout()
    media = OUT / "media"
    media.mkdir(exist_ok=True)
    fig.savefig(media / f"{STEM}_poses.png", dpi=100, bbox_inches="tight")
    print("figure ->", (media / f"{STEM}_poses.png").relative_to(REPO))


if os.environ.get("FHOLD_CMD") == "step":           # freecadcmd runs the file as a module or as __main__
    freecad_step()
elif __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("build", "boards", "reach", "homing", "farfield", "figures"))
    ap.add_argument("--test", default="facing", choices=tuple(TESTS))
    a = ap.parse_args()
    dict(build=cmd_build, boards=cmd_boards, reach=cmd_reach, homing=cmd_homing, farfield=cmd_farfield,
         figures=cmd_figures)[a.cmd](a)
