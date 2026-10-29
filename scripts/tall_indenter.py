#!/usr/bin/env python3
"""Tall indenter for the UR5 cell (owner, 2026-10-26; specification docs/experiments/20261026-tall_indenter/
20261026-tall_indenter_spec.md): a 5 mm PLA sphere on a neck of radius 3 mm that widens at 15 deg into a column of radius
30 mm, on a base plate screwed to the 2x4, printed in one piece. The homing seat stands on a strut 55 mm under the sphere's
centre: three lands under the mount's palmar face (its depth and tilt), the key of the present indenters with 0.05 mm on
every face (its place along and across the tool axis and its turn), and walls along the mount's column at 0.4 mm. The
numbers are ur5_cb3_cell.TALL, which the cell's clearance checks and the plan builder (--indenter tall) read as well.

    E=~/miniconda3/envs/femtip/bin/python
    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --"
    $RG $E scripts/tall_indenter.py build                  # STL and parts.json (the resolved geometry)
    $RG env TALL_CMD=step TALL_REPO=$PWD freecadcmd scripts/tall_indenter.py   # STEP from parts.json
    $RG .venv/bin/python scripts/tall_indenter.py stiffness
    $RG .venv/bin/python scripts/tall_indenter.py reach --kind sphere|tall [--H 160]   # acceptance test 1
    $RG .venv/bin/python scripts/tall_indenter.py figures
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts") if "__file__" in globals() else ".")

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:                                   # freecadcmd executes the file without __file__
    REPO = Path(os.environ["TALL_REPO"])
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "docs/experiments/20261026-tall_indenter"
CAD = OUT / "cad"
STEM = "20261026-tall_indenter"
PARTS_JSON = OUT / f"{STEM}_parts.json"


# ------------------------------------------------------------------------------------------------ geometry
def geometry(t=None):
    """The resolved geometry in I (mm) from ur5_cb3_cell.TALL: everything the CAD needs, as plain numbers."""
    import ur5_cb3_cell as CELL
    t = dict(CELL.TALL, **(t or {}))
    tn = math.tan(math.radians(t["taper_deg"]))
    z_neck = t["H"] - t["neck_z"]
    z_col = z_neck - (t["col_r"] - t["neck_r"]) / tn
    b = CELL.tall_boxes(t)
    f = CELL.tall_frame(t)
    x0, x1, y0, y1, tb = t["base"]
    return dict(params=t, z_neck=z_neck, z_col=round(z_col, 3), sphere_centre=[0.0, 0.0, t["H"]],
                base=[x0, x1, y0, y1, tb], fillet=dict(r0=t["col_r"] + 6.0, h=6.0),
                # wood screws (4 mm, steel: more than 100 mm under the sphere): clear of the column and the seat above them
                screws=[(30.0, -35.0), (-30.0, -35.0), (30.0, 35.0), (-5.0, 38.0)], screw_hole=4.6, screw_cb=(9.0, 3.0),
                block=b["block"], lands=b["lands"], walls=b["walls"], strut=b["strut"], key=CELL.key_points(t),
                lands_xy=b["lands_xy"], mouth=f["mouth"].tolist(), contact_home=f["contact"].tolist(),
                T_IM_home=f["T_IM_home"].round(4).tolist())


def solid(K, g):
    """The indenter as one solid with kernel K (scripts/ur5_calibration_mount.py MeshK or BrepK)."""
    t = g["params"]
    x0, x1, y0, y1, tb = g["base"]
    base = K.box((x0, y0, 0.0), (x1, y1, tb))
    holes = []
    d, (cbd, cbh) = g["screw_hole"], g["screw_cb"]
    for x, y in g["screws"]:
        holes += [K.cyl("z", (x, y), d / 2, -1.0, tb + 1.0), K.cyl("z", (x, y), cbd / 2, tb - cbh, tb + 1.0)]
    base = K.cut_all(base, holes)
    col = [K.cyl("z", (0.0, 0.0), g["fillet"]["r0"], tb - 0.01, tb + g["fillet"]["h"], r1=t["col_r"]),
           K.cyl("z", (0.0, 0.0), t["col_r"], tb - 0.01, g["z_col"] + 0.01),
           K.cyl("z", (0.0, 0.0), t["col_r"], g["z_col"], g["z_neck"] + 0.5,
                 r1=t["neck_r"] - 0.5 * math.tan(math.radians(t["taper_deg"]))),
           K.sphere(g["sphere_centre"], t["R"])]
    seat = [K.box(*g["block"])] + [K.box(lo, hi) for lo, hi in g["lands"] + g["walls"]]
    seat += [K.hull(g["strut"]), K.hull(g["key"])]
    return K.union([base] + col + seat)


def cmd_build(a):
    sys.path.insert(0, str(REPO / "scripts"))
    os.environ.setdefault("UR5M_REV", "v7")
    import ur5_calibration_mount as UM
    g = geometry()
    K = UM.MeshK()
    CAD.mkdir(parents=True, exist_ok=True)
    tm = UM.to_trimesh(solid(K, g))
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
    doc = FreeCAD.newDocument("tall")
    o = doc.addObject("Part::Feature", "tall_indenter")
    o.Shape = so
    Import.export([o], str(dst))
    FreeCAD.closeDocument("tall")
    back = Part.read(str(dst))
    rec = json.loads(PARTS_JSON.read_text())
    rec["step"] = dict(file=str(dst.relative_to(REPO)), valid=bool(so.isValid()), solids=len(so.Solids),
                       volume_mm3=round(so.Volume, 1), readback_valid=bool(back.isValid()),
                       readback_volume_mm3=round(back.Volume, 1))
    PARTS_JSON.write_text(json.dumps(rec, indent=1))
    print(rec["step"])


# ------------------------------------------------------------------------------------------------ stiffness
_trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def cmd_stiffness(a):
    """Lateral and axial compliance at the sphere's centre: a cantilever of the column's profile (Euler-Bernoulli, shear
    neglected) on a rigid base, PLA at E = 2.0 and 2.5 GPa; and the present indenter's for comparison."""
    import ur5_cb3_cell as CELL
    out = {}
    for name, H, prof in (("tall", None, None), ("present", 41.0, None)):
        t = dict(CELL.TALL)
        if name == "tall":
            H = t["H"]
            tn = math.tan(math.radians(t["taper_deg"]))
            z_neck, tb = H - t["neck_z"], t["base"][4]

            def r_of(z):
                return np.minimum(t["col_r"], t["neck_r"] + (z_neck - z) * tn)
            z = np.linspace(tb, z_neck, 4000)
        else:                                       # column r 10 from 6 to 36 mm, cone to r 5 at 41 (the sphere's equator)
            def r_of(z):
                return np.where(z < 36.0, 10.0, 10.0 - (z - 36.0))
            z = np.linspace(6.0, 41.0, 4000)
        r = r_of(z)
        I, A = math.pi * r ** 4 / 4, math.pi * r ** 2
        rec = dict(H_mm=H)
        for E in (2000.0, 2500.0):
            rec[f"lateral_mm_per_N_E{E / 1e3:.1f}"] = round(float(_trap((H - z) ** 2 / (E * I), z)), 5)
            rec[f"axial_mm_per_N_E{E / 1e3:.1f}"] = round(float(_trap(1.0 / (E * A), z)), 6)
        if name == "tall":                          # where the lateral compliance comes from
            dz = (H - z) ** 2 / (2000.0 * I)
            c = np.cumsum(dz * np.gradient(z))
            rec["share_from_top_60mm"] = round(float(1.0 - np.interp(H - 60.0, z, c) / c[-1]), 3)
            # the neck alone at r 3 mm (the most it bends locally): stress under 10 N at the sphere's side, 20 mm down
            s = 20.0
            r20 = float(r_of(H - s))
            rec["bending_stress_MPa_10N_20mm_below"] = round(10.0 * s * r20 / (math.pi * r20 ** 4 / 4), 1)
        out[name] = rec
    (OUT / f"{STEM}_stiffness.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_figures(a):
    """Figure 1: the tip's FE surface points by the roll of their normal about the tool axis and their place along it,
    marked by which indenter reaches them (presses, then twists). Figure 2: the part, and the mount's palmar face and key
    at homing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = OUT / "data"
    b = np.load(REPO / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-b_v2_surface_basis.npz")
    P, N = b["pts"], b["normals"]
    old = {r["i"]: r for r in json.loads((D / f"{STEM}_reach_sphere.json").read_text())["points"]}
    new = {r["i"]: r for r in json.loads((D / f"{STEM}_reach_tall_H160.json").read_text())["points"]}
    x0 = 0.5 * (P[:, 0].max() + P[:, 0].min())                   # the middle of the pad's section along x_P
    roll = np.degrees(np.arctan2(P[:, 1], P[:, 0] - x0))
    plt.rcParams.update({"font.family": "serif", "font.size": 13})
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, key, title in ((axs[0], "press", "Presses"), (axs[1], "twist", "Twists to 20 deg")):
        ok = lambda r: bool(r["turns_ok"]) if key == "press" else bool(r["twist_ok"])   # noqa: E731
        cats = {"Both indenters": [], "Tall indenter only": [], "Present indenter only": [], "Neither": []}
        for i in new:
            a_, b_ = ok(old[i]), ok(new[i])
            cats["Both indenters" if a_ and b_ else "Tall indenter only" if b_ else "Present indenter only" if a_
                 else "Neither"].append(i)
        for (lab, idx), col in zip(cats.items(), ("#4477aa", "#ee6677", "#ccbb44", "#bbbbbb")):
            if idx:
                ax.scatter(roll[idx], P[idx, 2], s=14, c=col, label=f"{lab} ({len(idx)})", edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("Angle about the pad's long axis (deg, 0 at the face)")
        ax.legend(loc="lower center", fontsize=11, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.42))
        ax.set_xlim(-135, 135)
    axs[0].set_ylabel("Place along the tool axis, z_P (mm)")
    fig.tight_layout()
    MEDIA = OUT / "media"
    MEDIA.mkdir(exist_ok=True)
    fig.savefig(MEDIA / f"{STEM}_reach.png", dpi=110, bbox_inches="tight")
    # the part
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    tm = trimesh.load(CAD / f"{STEM}.stl", force="mesh")
    g = json.loads(PARTS_JSON.read_text())["geometry"]
    fig = plt.figure(figsize=(13, 6.5))
    for k, (el, az, ttl) in enumerate(((22, -60, "From the robot's side"), (8, 0, "Along the tool axis at homing"))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        tri = tm.vertices[tm.faces]
        light = np.clip(tm.face_normals @ np.array([0.4, -0.5, 0.75]), 0.15, 1.0)
        pc = Poly3DCollection(tri, facecolors=plt.cm.Greys(0.35 + 0.5 * light), edgecolor="none")
        ax.add_collection3d(pc)
        mx, my, mz = g["mouth"]
        for x, y in g["lands_xy"]:
            ax.scatter([x], [y], [mz + 0.3], c="#ee6677", s=30, depthshade=False)
        cx, cy, cz = g["contact_home"]
        ax.scatter([cx], [cy], [cz], c="#4477aa", s=30, depthshade=False)
        ax.set_xlim(-80, 50); ax.set_ylim(-60, 70); ax.set_zlim(0, 170)
        ax.set_box_aspect((130, 130, 170))
        ax.view_init(el, az)
        ax.set_xlabel("x_I (mm)"); ax.set_ylabel("y_I (mm)"); ax.set_zlabel("z_I (mm)")
        ax.set_title(ttl)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{STEM}_part.png", dpi=100, bbox_inches="tight")
    print("figures ->", MEDIA.relative_to(REPO))


# ------------------------------------------------------------------------------------------------ reach map
REG = REPO / "docs/experiments/20261022-ur5_tip_calibration/lab/20261026-134522-cb3t/registration.json"
BASE_PLAN = REPO / "docs/experiments/20261022-ur5_tip_calibration/lab/20261026-plan_sphere_lab_r25_pass1_cb3t.json"
TURNS = tuple(range(-20, 45, 5))
J5, J6 = (34.0, 98.0), (-65.0, 50.0)
NEED = dict(mount=3.0, finger=0.8, hex_e=6.0, pad=0.3, robot=8.0)
GROUPS = dict(mount=("adapter", "fork", "clamp_plate"), finger=("dip_link", "cover", "cap"), hex_e=("hex_e",), pad=("pad",))


def _setup(kind, H):
    """The plan geometry for kind on cb3t's registration (or --registration's): the indenter's base where the sphere's
    base was."""
    import ur5_cb3_cell as CELL
    import ur5_tip_calibration_plan as PB
    C = CELL.C
    if H is not None:
        CELL.TALL["H"] = float(H)
    base = json.loads(BASE_PLAN.read_text())
    ck = base["checks"]
    CELL.BOARD.update(ck["board"])
    CELL.BOARD["half_mm"] = 400.0                     # the mat table under the wrist (top at z_B 32.8), not only 200 mm out
    CELL.OTHER[:] = [(np.array(o["T_BI"], float), o["kind"]) for o in ck.get("other", [])]
    g = PB.nominal(kind, base["geometry"].get("mount_turn_deg", ck.get("mount_turn_deg", -92.15)))
    g["T_EM"] = base["geometry"]["T_EM"]
    plan = json.loads(json.dumps(dict(geometry=g, indenter=kind), default=C.jsonable))
    geo = C.Geometry(plan)
    reg = C.Registration(geo).load(json.loads(REG.read_text()))
    return CELL, C, geo, reg


def _pose_ok(CELL, C, S, reg, kind, T_BE, contact_P, q0):
    """Joints within the windows and every group's least clearance (mm) at flange pose T_BE."""
    q = S.ik(T_BE, q0)
    if q is None:
        return False, None, None
    j5 = math.degrees(q[4])
    j6 = math.degrees(q0[5] + (q[5] - q0[5] + math.pi) % (2 * math.pi) - math.pi)
    if not (J5[0] <= j5 <= J5[1] and J6[0] <= j6 <= J6[1]):
        return False, None, (j5, j6)
    if CELL.self_clearance(q, reg.T_EM)["port"] < 15.0:
        return False, dict(port=0.0), (j5, j6)
    cl = CELL.clearance(q, T_BE, reg.T_BI, reg.T_EM, kind, contact_P=contact_P)
    worst = {}
    for k, v in cl.items():
        grp = next((gname for gname, names in GROUPS.items() if k in names), "robot")
        worst[grp] = min(worst.get(grp, 1e9), v)
    ok = all(worst[gname] >= NEED[gname] for gname in worst)
    return ok, worst, (j5, j6)


def cmd_reach(a):
    global J6
    if a.j6:
        J6 = tuple(float(v) for v in a.j6.split(","))
    CELL, C, geo, reg = _setup(a.kind, a.H)
    S = CELL.SIM
    import ur5_tip_calibration_r24_pass3 as R24
    reg_names = [R24.region(n) or "" for n in geo.nrm]
    qh = np.asarray(reg.q_home, float)
    T_EP = reg.T_EM @ reg.T_MP
    idx = [i for i in range(len(geo.pts)) if reg_names[i] and (not a.regions or reg_names[i] in a.regions.split(","))]
    if a.limit:
        idx = idx[::max(1, len(idx) // a.limit)]
    st = 1.5
    rows = []
    t0 = time.time()
    for n_done, i in enumerate(idx):
        p, n = geo.pts[i], geo.nrm[i]
        mask, why = [], {}
        for tw in TURNS:
            T_IP = geo.T_IP(p, n, float(tw), st)
            good = True
            for dz in (0.0, st + 1.0):
                T_BE = reg.T_BI @ C.Tm(T_IP[:3, :3], T_IP[:3, 3] - C.EZ * dz) @ C.inv(T_EP)
                ok, worst, j = _pose_ok(CELL, C, S, reg, a.kind, T_BE, p, qh)
                if not ok:
                    good = False
                    if j is None:
                        why[tw] = "ik"
                    elif worst is None:
                        why[tw] = "j5" if not J5[0] <= j[0] <= J5[1] else "j6"
                    else:
                        why[tw] = ",".join(sorted(k for k, v in worst.items() if v < NEED.get(k, 15.0)))
                    break
            mask.append(good)
        rec = dict(i=i, region=str(reg_names[i]), y=round(float(p[1]), 3), turns_ok=[tw for tw, m in zip(TURNS, mask) if m])
        if rec["turns_ok"]:                         # slides: the deep pose turned 2.3 deg about four horizontal axes
            tw = min(rec["turns_ok"], key=lambda v: abs(v - 10))
            T_IP = geo.T_IP(p, n, float(tw), st)
            deep = reg.T_BI @ C.Tm(T_IP[:3, :3], T_IP[:3, 3] - C.EZ * (st + 1.0)) @ C.inv(T_EP)
            c_B = C.apply(reg.T_BI, geo.c_I)
            sl = True
            for ang in (0.0, 90.0, 180.0, 270.0):
                u = reg.T_BI[:3, :3] @ np.cross(C.EZ, [math.cos(math.radians(ang)), math.sin(math.radians(ang)), 0.0])
                for dg in (2.3, -2.3):
                    ok, _, _ = _pose_ok(CELL, C, S, reg, a.kind, C.turn_about(deep, c_B, u, dg), p, qh)
                    sl &= ok
                    if not sl:
                        break
                if not sl:
                    break
            rec["slide_ok"] = bool(sl)
        rec["why"] = {str(k): v for k, v in why.items()}
        ok_set = set(rec["turns_ok"])
        rec["twist_ok"] = any(all(t + 5 * k in ok_set for k in range(5)) for t in TURNS)
        rows.append(rec)
        if n_done % 25 == 0:
            print(f"{n_done}/{len(idx)} points, {time.time() - t0:.0f} s", flush=True)
    summ = summarize(rows)
    tag = (a.kind if a.kind == "sphere" else f"tall_H{CELL.TALL['H']:.0f}") + (f"_j6_{J6[0]:.0f}_{J6[1]:.0f}" if a.j6 else "")
    out = Path(a.out) if a.out else OUT / "data" / f"{STEM}_reach_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(kind=a.kind, H=CELL.TALL["H"] if a.kind == "tall" else 41.0, turns=TURNS, J5=J5, J6=J6,
                                   need=NEED, registration=str(REG.relative_to(REPO)), summary=summ, points=rows,
                                   seconds=round(time.time() - t0, 1)), indent=0))
    print(json.dumps(summ, indent=1))
    print(f"-> {out.relative_to(REPO)} in {time.time() - t0:.0f} s")


def summarize(rows):
    out = {}
    for reg in ("face", "edge", "flank", "tip"):
        for side, lo, hi in (("y<-1.5", -1e9, -1.5), ("between", -1.5, 1.5), ("y>1.5", 1.5, 1e9), ("all", -1e9, 1e9)):
            sel = [r for r in rows if r["region"] == reg and lo <= r["y"] < hi]
            if not sel:
                continue
            out[f"{reg} {side}"] = dict(n=len(sel), press=sum(bool(r["turns_ok"]) for r in sel),
                                        slide=sum(bool(r.get("slide_ok")) for r in sel),
                                        twist=sum(bool(r["twist_ok"]) for r in sel))
    return out


def cmd_homing(a):
    """The homing pose on the tall indenter: IK on cb3t's branch, joints, and the clearances of the robot, the HEX-E, the
    finger and the pad (the mount touches the seat by design)."""
    CELL, C, geo, reg = _setup("tall", a.H)
    T_BE = reg.T_BI @ geo.T_IM_home @ C.inv(reg.T_EM)
    q = CELL.SIM.ik(T_BE, np.asarray(reg.q_home, float))
    cl = CELL.clearance(q, T_BE, reg.T_BI, reg.T_EM, "tall")
    worst = {}
    for k, v in cl.items():
        grp = next((gname for gname, names in GROUPS.items() if k in names), "robot")
        worst[grp] = round(min(worst.get(grp, 1e9), v), 2)
    rec = dict(H=CELL.TALL["H"], q_deg=np.degrees(q).round(1).tolist(), clear=worst,
               self=CELL.self_clearance(q, reg.T_EM), flange_B=np.round(T_BE[:3, 3], 1).tolist())
    print(json.dumps(rec, indent=1, default=float))
    p = OUT / "data" / f"{STEM}_homing_H{CELL.TALL['H']:.0f}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1, default=float))


if os.environ.get("TALL_CMD") == "step":           # freecadcmd runs the file as a module or as __main__
    freecad_step()
elif __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("build", "stiffness", "reach", "homing", "figures"))
    ap.add_argument("--kind", default="tall")
    ap.add_argument("--H", type=float, default=None)
    ap.add_argument("--j6", default="", help="reach: joint 6's window, lo,hi deg (default -65,50: the HEX-E's cable)")
    ap.add_argument("--limit", type=int, default=0, help="reach: about this many points (a quick look)")
    ap.add_argument("--registration", default=None, help="reach: a session's registration.json in place of cb3t's")
    ap.add_argument("--base-plan", default=None, help="reach: the plan whose checks (board, other indenters) apply")
    ap.add_argument("--regions", default="", help="reach: only these regions, e.g. tip,edge")
    ap.add_argument("--out", default=None, help="reach: the output JSON")
    a = ap.parse_args()
    if a.registration:
        REG = Path(a.registration).resolve()
    if a.base_plan:
        BASE_PLAN = Path(a.base_plan).resolve()
    dict(build=cmd_build, stiffness=cmd_stiffness, reach=cmd_reach, homing=cmd_homing, figures=cmd_figures)[a.cmd](a)
