#!/usr/bin/env python3
"""Fingertip holder for the UR5 cell (owner, 2026-10-27): a second v2 fingertip (pad, cassette 3T and top cover TW15H3, no
board) held still while v2 ID 1 moves around it without contact, to measure the crosstalk between two tips at known poses.
The tip's four M2 screws form their threads in 1.9 mm holes that pass through a station block. The block stands out from a
column on an arm, so the tip faces up from its top face (mode "up": the pads facing) or down from its underside (mode
"down": side by side). The homing seat is the tall indenter's (docs/experiments/20261026-tall_indenter): three raised flats,
the key with 0.05 mm on every face and guide walls, on a tower of its own. The numbers are ur5_cb3_cell.TIPHOLD and
TIP_F, which the cell's clearance checks read as well.

    E=~/miniconda3/envs/femtip/bin/python
    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --"
    $RG $E scripts/fingertip_holder.py build                    # STL and parts.json
    $RG env -u PYTHONPATH -u VIRTUAL_ENV PATH=/usr/bin:/bin FHOLD_CMD=step FHOLD_REPO=$PWD freecadcmd scripts/fingertip_holder.py
    $RG env UR5_MOUNT=v7 .venv/bin/python scripts/fingertip_holder.py reach --mode up|down
    $RG env UR5_MOUNT=v7 .venv/bin/python scripts/fingertip_holder.py homing
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


# ------------------------------------------------------------------------------------------------ geometry
def geometry():
    """The resolved geometry in I (mm): everything the CAD needs, as plain numbers."""
    import ur5_cb3_cell as CELL
    t = dict(CELL.TIPHOLD)
    g = CELL.tiphold_parts(t)
    f = g.pop("frame")
    return dict(params=t, tip_F=CELL.TIP_F, **{k: (np.asarray(v).tolist() if k in ("mouth",) else v) for k, v in g.items()},
                hole_d=1.9, fillet=dict(r0=t["col_r"] + 6.0, h=6.0),
                screws=[(30.0, -20.0), (-30.0, -20.0), (30.0, 60.0), (-30.0, 10.0)], screw_hole=4.6, screw_cb=(9.0, 3.0),
                contact_home=f["contact"].tolist(), T_IM_home=f["T_IM_home"].round(4).tolist())


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
    station = K.union([K.box(*g["station"]), K.box(*g["arm"]), K.hull(g["gusset"])])
    station = K.cut_all(station, [K.cyl("z", (x, y), g["hole_d"] / 2, zb - 1.0, zt + 1.0) for x, y in g["holes"]])
    col = [K.cyl("z", (0.0, 0.0), g["fillet"]["r0"], tb - 0.01, tb + g["fillet"]["h"], r1=t["col_r"]),
           K.cyl("z", (0.0, 0.0), t["col_r"], tb - 0.01, zt)]
    seat = [K.box(*g["block"])] + [K.box(lo, hi) for lo, hi in g["lands"] + g["walls"]]
    seat += [K.hull(g["strut"]), K.hull(g["key"])]
    return K.union([base, station] + col + seat)


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
    PARTS_JSON.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "geometry"}, indent=1))


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


def test_poses(mode, held_apex):
    """Facing (mode up): v2 ID 1's apex W mm over the held tip's apex, turned tw about the common normal. Side by side
    (mode down): v2 ID 1 at -y of the held tip with g mm between the pads' flanks (14.8 mm wide), moved s mm toward its
    tip end, turned tw about the vertical through its apex."""
    out = []
    if mode == "up":
        for W in (0.5, 1, 2, 3, 5, 10, 15, 20, 30):
            for tw in (-20, -10, 0, 10, 20, 30, 40):
                out.append(dict(W=W, tw=tw, p=held_apex + np.array([0.0, 0.0, W])))
    else:
        for g in (9, 10, 11, 12, 13, 14, 15, 20, 25):
            for s in (0, 2, 5):
                for tw in (-20, -10, 0, 10):
                    out.append(dict(g=g, s=s, tw=tw, p=held_apex + np.array([s, -(14.8 + g), 0.0])))
    return out


def _cell(mode):
    import tall_indenter as TI
    CELL, C, geo, reg = TI._setup("sphere", None)      # cb3t's registration: the holder's base where the sphere's was
    CELL.TIPHOLD["mode"] = mode
    return TI, CELL, C, reg


def cmd_reach(a):
    TI, CELL, C, reg = _cell(a.mode)
    S = CELL.SIM
    qh = np.asarray(reg.q_home, float)
    held_apex = C.apply(CELL.tiphold_T_IF(), APEX_F[None])[0]
    rows, t0 = [], time.time()
    for k in test_poses(a.mode, held_apex):
        T_IM = T_IF_moving(k.pop("p"), k["tw"]) @ np.linalg.inv(CELL.T_M_FROM_F)
        T_BE = reg.T_BI @ T_IM @ C.inv(reg.T_EM)
        ok, worst, j = TI._pose_ok(CELL, C, S, reg, "tiphold", T_BE, None, qh)
        k.update(ok=bool(ok), joints_56=None if j is None else [round(float(v), 1) for v in j],
                 clear=None if worst is None else {n: round(float(v), 2) for n, v in worst.items()})
        rows.append(k)
    out = OUT / "data" / f"{STEM}_reach_{a.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(mode=a.mode, held_apex_I=held_apex.round(2).tolist(), J5=TI.J5, J6=TI.J6, need=TI.NEED,
                                   registration=str(TI.REG.relative_to(REPO)), poses=rows,
                                   seconds=round(time.time() - t0, 1)), indent=0))
    key = ("W", "tw") if a.mode == "up" else ("g", "s", "tw")
    for r in rows:
        print(" ".join(f"{n} {r[n]:>4}" for n in key), "ok" if r["ok"] else "--", r["joints_56"], r["clear"])
    print(f"{sum(r['ok'] for r in rows)}/{len(rows)} poses pass -> {out.relative_to(REPO)} in {time.time() - t0:.0f} s")


def cmd_homing(a):
    """The homing pose on the holder, with the tip in either mode: IK on cb3t's branch, joints and clearances."""
    recs = {}
    for mode in ("up", "down"):
        TI, CELL, C, reg = _cell(mode)
        T_IM = CELL.tiphold_parts()["frame"]["T_IM_home"]
        T_BE = reg.T_BI @ T_IM @ C.inv(reg.T_EM)
        q = CELL.SIM.ik(T_BE, np.asarray(reg.q_home, float))
        cl = CELL.clearance(q, T_BE, reg.T_BI, reg.T_EM, "tiphold")
        worst = {}
        for k, v in cl.items():
            grp = next((gname for gname, names in TI.GROUPS.items() if k in names), "robot")
            worst[grp] = round(min(worst.get(grp, 1e9), v), 2)
        recs[mode] = dict(q_deg=np.degrees(q).round(1).tolist(), clear=worst, self=CELL.self_clearance(q, reg.T_EM),
                          flange_B=np.round(T_BE[:3, 3], 1).tolist())
    print(json.dumps(recs, indent=1, default=float))
    p = OUT / "data" / f"{STEM}_homing.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(recs, indent=1, default=float))


# ------------------------------------------------------------------------------------------------ figure
def cmd_figures(a):
    """The holder with the held tip up; the pads facing 2 mm apart (mode up); side by side at the least gap that passes
    with s 0 and tw 0 (mode down). v2 ID 1's finger in blue with its fork cut at z_M 60, the held tip in orange."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import ur5_cb3_cell as CELL
    hold = trimesh.load(CAD / f"{STEM}.stl", force="mesh")
    hold = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(hold.vertices, hold.faces, max_edge=3.0))   # for the clip
    side = json.loads((OUT / "data" / f"{STEM}_reach_down.json").read_text())["poses"]
    g_min = min(r["g"] for r in side if r["ok"] and r["s"] == 0 and r["tw"] == 0)
    plt.rcParams.update({"font.family": "serif", "font.size": 13})
    fig = plt.figure(figsize=(17, 6.4))
    panels = (("up", None, "The holder, tip up", (-45, 35, 30), (-45, 75), (0, 80), (24, -130)),
              ("up", 2.0, "Pads facing, 2 mm apart", (-45, 35, 30), (-65, 15), (20, 100), (14, -28)),
              ("down", g_min, f"Side by side, {g_min:g} mm between the pads", (-45, 35, 30), (-85, -5), (0, 80), (14, -28)))
    for k, (mode, gap, ttl, (x0, x1, _), ylim, zlim, (el, az)) in enumerate(panels):
        CELL.TIPHOLD["mode"] = mode
        T_IF = CELL.tiphold_T_IF()
        meshes = [(hold, "#9a9a9a")]
        for name in ("cover", "cap", "pad"):
            m = CELL._load(CELL.TOOL_PARTS[name][0])
            m.apply_transform(T_IF)
            meshes.append((m, "#ee7733"))
        if gap is not None:
            apex = (T_IF @ np.r_[APEX_F, 1.0])[:3]
            p = apex + (np.array([0.0, 0.0, gap]) if mode == "up" else np.array([0.0, -(14.8 + gap), 0.0]))
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
        lo, hi = np.array([x0, ylim[0], zlim[0]]), np.array([x1, ylim[1], zlim[1]])
        keep = ((tri >= lo) & (tri <= hi)).all(axis=(1, 2))          # faces inside the panel's box
        tri, nrm, rgb = tri[keep], nrm[keep], rgb[keep]
        light = np.clip(nrm @ np.array([0.45, -0.45, 0.77]), 0.12, 1.0)
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tri, edgecolor="none", facecolors=np.clip(rgb * (0.4 + 0.6 * light[:, None]), 0, 1)))
        ax.set_xlim(x0, x1); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_box_aspect((x1 - x0, ylim[1] - ylim[0], zlim[1] - zlim[0]))
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
    ap.add_argument("cmd", choices=("build", "reach", "homing", "figures"))
    ap.add_argument("--mode", default="up", choices=("up", "down"))
    a = ap.parse_args()
    dict(build=cmd_build, reach=cmd_reach, homing=cmd_homing, figures=cmd_figures)[a.cmd](a)
