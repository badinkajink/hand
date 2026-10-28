#!/usr/bin/env python3
"""Indenter array for the UR5 cell (owner, 2026-10-27: "put all four of these indenters on one print, and that one piece to
have multiple indexers so that we can do registration on multiple indexers on the same part"; the back row "a good bit
higher than the first row", the first row "generally higher than the current low indenters"; "the really elaborate and
higher wall indexer actually produced noisier homes than the simple trapezoidal volume"). One PLA print: a 6 mm base, homing
keys of mount v3's simple design at the front (the drafted key on a post; its long faces seat in the mount's socket and the
palmar face stays 1 mm clear of the post), and four indenters in two rows. In front, the 5 mm sphere and the 12.5 mm half
rod have their tops at 100 mm. At the back, a 12.5 mm sphere (in place of the tall indenter's second 5 mm sphere) and the
fingertip holder's two stations have their tops at 160 mm. The keys stand at different yaws, so that homing on two or three
of them fixes x, y and the yaw of the one part.

Frame A: z up from the base's underside, x away from the robot (the tool axis at homing on a key of yaw 0), origin at the
base's centre. The checks place A where cb3t's registration put the sphere's frame I, on 2x4s under the whole base.

    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --"
    $RG .venv/bin/python scripts/indenter_array.py build                              # STL and parts.json
    $RG env UR5_MOUNT=v7 .venv/bin/python scripts/indenter_array.py homing [--scan]   # IK and clearances at each key
    $RG env UR5_MOUNT=v7 .venv/bin/python scripts/indenter_array.py press             # presses on each indenter
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("UR5M_REV", "v7")
OUT = REPO / "docs/experiments/20261027-indenter_array"
CAD = OUT / "cad"
STEM = "20261027-indenter_array"
PARTS_JSON = OUT / f"{STEM}_parts.json"
TAN15 = math.tan(math.radians(15.0))
MOUTH_M = (17.0, 0.0, 49.0)                         # the homing socket's mouth on mount v7's palmar face (M)

ARRAY = dict(
    base=(-110.0, 110.0, -118.0, 118.0, 6.0),       # x0, x1, y0, y1, thickness
    screws=((-100.0, -108.0), (-100.0, 0.0), (-100.0, 108.0), (100.0, -108.0), (100.0, 0.0), (100.0, 108.0)),
    screw_hole=4.6, screw_cb=(9.0, 3.0),            # wood screws, counterbored 3 mm
    mouth_z=30.0,                                   # the socket's mouth at homing, as on the fingertip holder's seat
    keys=((-95.0, -80.0, -30.0), (-95.0, 0.0, 0.0), (-95.0, 80.0, 15.0)),   # mouth centre x, y and yaw (deg) of each key;
    #                                               joint 5 stays in 34-98 deg at -30..0, -15..0 and 0..+15 deg there
    sphere5=dict(c=(-15.0, -42.0), R=5.0, top=120.0, neck_r=3.0, neck_dz=4.0, col_r=24.0),
    rod=dict(c=(-15.0, 45.0), R=12.5, half_len=10.0, top=120.0, box_dz=27.5, col_r=24.0),
    sphere12=dict(c=(70.0, -45.0), R=12.5, top=180.0, neck_r=7.5, neck_dz=10.0, col_r=26.0),
    holder=dict(c=(70.0, 50.0), z_bot=153.0, col_r0=26.0, cone_below=15.0),   # the held apexes at 180 mm
)
NAMES = ("sphere5", "rod", "sphere12", "holder")


def _Rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def key_dims():
    """The v3 key (scripts/ur5_calibration_mount.py indenter): its half-sizes (along the tool axis, across) at the post's
    top, 1.01 mm under the mouth, and at its top, 4.5 mm over it; the post 3 mm wider on every side."""
    import ur5_calibration_mount as UM
    P = UM.P
    k0 = UM.home_halves(-P["home_key_below"] - 0.01, key=True)[::-1]
    k1 = UM.home_halves(P["home_depth"] - P["home_floor_gap"], key=True)[::-1]
    return dict(k0=[float(v) for v in k0], k1=[float(v) for v in k1], below=P["home_key_below"],
                top=P["home_depth"] - P["home_floor_gap"], margin=P["post_margin"])


def geometry(a=None):
    """The resolved geometry in A (mm), as plain numbers."""
    import ur5_cb3_cell as CELL
    a = dict(ARRAY, **(a or {}))
    tb, mz = a["base"][4], a["mouth_z"]
    kd = key_dims()
    keys = []
    for x, y, yaw in a["keys"]:
        R = _Rz(yaw) @ np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])   # z_M -> tool axis, x_M -> -z
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, np.array([x, y, mz]) - R @ np.array(MOUTH_M)
        keys.append(dict(c=(x, y), yaw=yaw, body_top=mz - kd["below"], pin_top=mz + kd["top"], T_AM_home=T.round(4).tolist()))
    out = dict(params=a, key=kd, keys=keys)
    for name in ("sphere5", "sphere12"):
        s = dict(a[name])
        s["z_neck"] = s["top"] - s["R"] - s["neck_dz"]
        s["z_col"] = max(tb, s["z_neck"] - (s["col_r"] - s["neck_r"]) / TAN15)
        s["r_col"] = s["neck_r"] + (s["z_neck"] - s["z_col"]) * TAN15
        out[name] = s
    r = dict(a["rod"])
    r["z_axis"] = r["top"] - r["R"]
    r["box_z0"] = r["z_axis"] - r["box_dz"]
    r["r_top"] = math.hypot(r["R"], r["half_len"])                     # the column's top meets the box's corners
    out["rod"] = r
    h = dict(a["holder"])
    t = dict(CELL.TIPHOLD, z_bot=h["z_bot"])
    g = CELL.tiphold_parts(t)
    hx, hy = h["c"]
    sh = lambda p: (p[0] + hx, p[1] + hy, p[2])                         # noqa: E731
    lo = lambda p: (p[0] + hx, p[1] + hy, tb - 0.01)                   # noqa: E731  (owner, 10-27: the blocks down to the
    #                                                                   plate, so they print without support)
    h["stations"] = [dict(y_s=st["y_s"] + hy, block=[lo(st["block"][0]), sh(st["block"][1])],
                          arm=[sh(st["arm"][0]), sh(st["arm"][1])], gusset=[sh(p) for p in st["gusset"]],
                          holes=[(x + hx, y + hy) for x, y in st["holes"]]) for st in g["stations"]]
    h["z_top"], h["col_r"], h["tiphold"] = g["z_top"], t["col_r"], dict(z_bot=h["z_bot"])
    h["hole_depth"] = 10.0                                             # blind M2 holes from the top: M2 x 8 reaches 4.1 mm
    h["apex"] = [((CELL.tiphold_T_IF(t, i) @ np.r_[-10.55, 0.0, -87.91, 1.0])[:3] + [hx, hy, 0.0]).round(3).tolist()
                 for i in range(len(t["stations"]))]
    out["holder"] = h
    out["hole_d"] = 1.9
    return out


# ------------------------------------------------------------------------------------------------ CAD
def solid(K, g):
    import ur5_calibration_mount as UM
    a = g["params"]
    x0, x1, y0, y1, tb = a["base"]
    base = K.rbox_z(x0, x1, y0, y1, 0.0, tb, 6.0)
    d, (cbd, cbh) = a["screw_hole"], a["screw_cb"]
    holes = []
    for x, y in a["screws"]:
        holes += [K.cyl("z", (x, y), d / 2, -1.0, tb + 1.0), K.cyl("z", (x, y), cbd / 2, tb - cbh, tb + 1.0)]
    parts = [K.cut_all(base, holes)]
    kd = g["key"]
    bx, by = kd["k0"][0] + kd["margin"], kd["k0"][1] + kd["margin"]
    for k in g["keys"]:
        local = K.union([K.rbox_z(-bx, bx, -by, by, tb - 0.01, k["body_top"], 2.0),
                         UM.frustum(K, "z", (0.0, 0.0), kd["k0"], kd["k1"], k["body_top"] - 0.01, k["pin_top"])])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = _Rz(k["yaw"]), [k["c"][0], k["c"][1], 0.0]
        parts.append(K.transform(local, T))
    for name in ("sphere5", "sphere12"):
        s = g[name]
        c = s["c"]
        if s["z_col"] > tb:
            parts.append(K.cyl("z", c, s["col_r"], tb - 0.01, s["z_col"] + 0.01))
        parts += [K.cyl("z", c, s["r_col"], s["z_col"], s["z_neck"], r1=s["neck_r"]),
                  K.sphere((c[0], c[1], s["top"] - s["R"]), s["R"])]
    r = g["rod"]
    (cx, cy), hl = r["c"], r["half_len"]
    parts += [K.cyl("z", (cx, cy), r["col_r"], tb - 0.01, r["box_z0"] + 0.01, r1=r["r_top"]),
              K.box((cx - r["R"], cy - hl, r["box_z0"]), (cx + r["R"], cy + hl, r["z_axis"])),
              K.cyl("y", (cx, r["z_axis"]), r["R"], cy - hl, cy + hl)]
    h = g["holder"]
    (hx, hy), zb, zt = h["c"], h["z_bot"], h["z_top"]
    zc = zb - h["cone_below"]
    parts += [K.cyl("z", (hx, hy), h["col_r0"], tb - 0.01, zc, r1=h["col_r"]), K.cyl("z", (hx, hy), h["col_r"], zc - 0.01, zt)]
    for st in h["stations"]:
        s = K.union([K.box(*st["block"]), K.box(*st["arm"]), K.hull(st["gusset"])])
        parts.append(K.cut_all(s, [K.cyl("z", (x, y), g["hole_d"] / 2, zt - 10.0, zt + 1.0) for x, y in st["holes"]]))
    return K.union(parts)


def cmd_build(a):
    import ur5_calibration_mount as UM
    g = geometry()
    CAD.mkdir(parents=True, exist_ok=True)
    tm = UM.to_trimesh(solid(UM.MeshK(), g))
    dst = CAD / f"{STEM}.stl"
    tm.export(dst)
    vol = float(tm.volume)
    rec = dict(geometry=g, stl=str(dst.relative_to(REPO)), volume_cm3=round(vol / 1e3, 1),
               mass_g_solid=round(vol / 1e3 * 1.24, 0), bounds_mm=np.round(tm.bounds, 2).tolist(),
               watertight=bool(tm.is_watertight), bodies=len(tm.split(only_watertight=False)),
               when=time.strftime("%Y-%m-%d %H:%M"))
    PARTS_JSON.write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: v for k, v in rec.items() if k != "geometry"}, indent=1))


# ------------------------------------------------------------------------------------------------ the cell model
_G = {}
PRESSED = [None]                                   # the indenter whose tip a press touches (left out with tip=False)
EXCL_MM = 12.0                                     # the pad's points this near its apex meet the pressed tip (the flat rod)
SKIP_KEY = [None]                                  # the key the mount is seated on at homing (its contact is the seat)


def array_sdf(P, tip=True):
    """Signed distance (mm) to the array and the holder's held tips (A frame)."""
    import ur5_cb3_cell as CELL
    if not _G:
        _G.update(geometry())
    g = _G
    P = np.asarray(P, float)
    a = g["params"]
    x0, x1, y0, y1, tb = a["base"]
    parts = [CELL.sd_box(P, (x0, y0, 0.0), (x1, y1, tb))]
    kd = g["key"]
    bx, by = kd["k0"][0] + kd["margin"], kd["k0"][1] + kd["margin"]
    for i, k in enumerate(g["keys"]):
        if i == SKIP_KEY[0]:
            continue
        L = (P - [k["c"][0], k["c"][1], 0.0]) @ _Rz(k["yaw"])
        pts = [(sx * hx, sy * hy, z) for (hx, hy), z in ((kd["k0"], k["body_top"]), (kd["k1"], k["pin_top"]))
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        parts += [CELL.sd_box(L, (-bx, -by, tb), (bx, by, k["body_top"])), CELL._convex_sdf(L, pts)]
    for name in ("sphere5", "sphere12"):
        s = g[name]
        Q = P - [s["c"][0], s["c"][1], 0.0]
        parts += [CELL.sd_cyl_z(Q, s["col_r"], tb, s["z_col"]), CELL.sd_capped_cone(Q, s["z_col"], s["z_neck"], s["r_col"], s["neck_r"])]
        if tip or PRESSED[0] != name:
            parts.append(np.linalg.norm(Q - [0.0, 0.0, s["top"] - s["R"]], axis=1) - s["R"])
    r = g["rod"]
    (cx, cy), hl = r["c"], r["half_len"]
    Q = P - [cx, cy, 0.0]
    parts += [CELL.sd_capped_cone(Q, tb, r["box_z0"], r["col_r"], r["r_top"]),
              CELL.sd_box(P, (cx - r["R"], cy - hl, r["box_z0"]), (cx + r["R"], cy + hl, r["z_axis"]))]
    if tip or PRESSED[0] != "rod":
        parts.append(CELL.sd_cyl_y(P, r["R"], cy - hl, cy + hl, xz=(cx, r["z_axis"])))
    h = g["holder"]
    (hx, hy), zb, zt = h["c"], h["z_bot"], h["z_top"]
    Q = P - [hx, hy, 0.0]
    zc = zb - h["cone_below"]
    parts += [CELL.sd_capped_cone(Q, tb, zc, h["col_r0"], h["col_r"]), CELL.sd_cyl_z(Q, h["col_r"], zc, zt)]
    for st in h["stations"]:
        parts += [CELL.sd_box(P, *st["block"]), CELL.sd_box(P, *st["arm"]), CELL._convex_sdf(P, st["gusset"])]
    parts.append(CELL.held_tip_sdf(Q, dict(CELL.TIPHOLD, **h["tiphold"])))
    return np.min(np.stack(parts), axis=0)


def _setup():
    """cb3t's registration with A where the sphere's frame I was, on 2x4s under the whole base, the rod fixture gone; the
    cell's indenter kind 'array'."""
    import tall_indenter as TI
    CELL, C, geo, reg = TI._setup("sphere", None)
    CELL.OTHER[:] = []
    x0, x1, y0, y1, _ = ARRAY["base"]
    CELL.BOARD["block_half_mm"] = [max(-x0, x1) + 10.0, 500.0]
    ind, tip = CELL.indenter_sdf, CELL.tip_sdf
    CELL.indenter_sdf = lambda P, kind="sphere", tip=True: array_sdf(P, tip) if kind == "array" else ind(P, kind, tip)  # noqa: E731
    CELL.tip_sdf = lambda P, kind="sphere": array_sdf(P) if kind == "array" else tip(P, kind)                      # noqa: E731
    cl = CELL.clearance
    CELL.clearance = lambda *a, **k: cl(*a, **dict(k, excl_mm=EXCL_MM))                                           # noqa: E731
    return TI, CELL, C, reg


def _groups(TI, cl):
    worst = {}
    for k, v in cl.items():
        grp = next((gname for gname, names in TI.GROUPS.items() if k in names), "robot")
        worst[grp] = round(min(worst.get(grp, 1e9), float(v)), 2)
    return worst


def homing_at(TI, CELL, C, reg, x, y, yaw, i=None):
    """IK and clearances with the mount seated on a key at (x, y, yaw); i, that key's index in ARRAY (the other keys stay
    as obstacles), or None for a key alone in place of ARRAY's."""
    _G.clear()
    _G.update(geometry() if i is not None else geometry(dict(keys=((x, y, yaw),))))
    SKIP_KEY[0] = i if i is not None else 0
    T_AM = np.array(_G["keys"][i or 0]["T_AM_home"])
    T_BE = reg.T_BI @ T_AM @ C.inv(reg.T_EM)
    qh = np.asarray(reg.q_home, float)
    q = CELL.SIM.ik(T_BE, qh)
    if q is None:
        return dict(x=x, y=y, yaw=yaw, ok=False, why="ik")
    j5 = math.degrees(q[4])
    j6 = math.degrees(qh[5] + (q[5] - qh[5] + math.pi) % (2 * math.pi) - math.pi)
    worst = _groups(TI, CELL.clearance(q, T_BE, reg.T_BI, reg.T_EM, "array"))
    port = float(CELL.self_clearance(q, reg.T_EM)["port"])
    ok = (TI.J5[0] <= j5 <= TI.J5[1] and TI.J6[0] <= j6 <= TI.J6[1] and port >= 15.0
          and all(worst[k] >= TI.NEED.get(k, 15.0) for k in worst))
    return dict(x=x, y=y, yaw=yaw, ok=bool(ok), q_deg=np.degrees(q).round(1).tolist(), j5=round(j5, 1), j6=round(j6, 1),
                port=round(port, 1), clear=worst)


def cmd_homing(a):
    TI, CELL, C, reg = _setup()
    rows = []
    cands = ([(x, y, yaw) for x, y in ((-95.0, -80.0), (-95.0, 0.0), (-95.0, 80.0)) for yaw in range(-60, 61, 15)]
             if a.scan else ARRAY["keys"])
    for i, (x, y, yaw) in enumerate(cands):
        r = homing_at(TI, CELL, C, reg, x, y, float(yaw), None if a.scan else i)
        rows.append(r)
        print(json.dumps(r, default=float), flush=True)
    if not a.scan:
        p = OUT / "data" / f"{STEM}_homing.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(registration=str(TI.REG.relative_to(REPO)), J5=TI.J5, J6=TI.J6, need=TI.NEED,
                                     keys=rows), indent=1, default=float))


# ------------------------------------------------------------------------------------------------ presses
APEX_F = np.array([-10.55, 0.0, -87.91])
APEX_P = np.array([8.5, 0.0, 10.86])               # the same point in the pad frame P


def T_AF_press(p, n, tw):
    """F with the pad's apex at p, its face normal along -n (pressing along -n), turned tw deg about n (0: the tip toward
    +x as far as n allows)."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    z = np.array([-math.cos(math.radians(tw)), -math.sin(math.radians(tw)), 0.0])
    z = z - n * (z @ n)
    if np.linalg.norm(z) < 1e-6:
        z = np.array([0.0, -1.0, 0.0]) - n * (-n[1])
    z /= np.linalg.norm(z)
    R = np.column_stack([n, np.cross(z, n), z])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(p, float) - R @ APEX_F
    return T


def press_points(g):
    """Contact points and outward normals on each indenter: the spheres at polar 0, 30, 60 and 90 deg, eight azimuths; the
    rod's top and its round at 45 and 90 deg each side; the holder's facing hovers W 2 and 30 mm over each held apex."""
    out = []
    for name in ("sphere5", "sphere12"):
        s = g[name]
        c = np.array([s["c"][0], s["c"][1], s["top"] - s["R"]])
        for pol in (0, 30, 60, 90):
            for az in ((0,) if pol == 0 else range(0, 360, 45)):
                n = np.array([math.sin(math.radians(pol)) * math.cos(math.radians(az)),
                              math.sin(math.radians(pol)) * math.sin(math.radians(az)), math.cos(math.radians(pol))])
                out.append(dict(ind=name, pol=pol, az=az, p=c + s["R"] * n, n=n))
    r = g["rod"]
    c = np.array([r["c"][0], r["c"][1], r["z_axis"]])
    for pol in (-90, -45, 0, 45, 90):
        n = np.array([math.sin(math.radians(pol)), 0.0, math.cos(math.radians(pol))])
        out.append(dict(ind="rod", pol=pol, az=0, p=c + r["R"] * n, n=n))
    for i, ap in enumerate(g["holder"]["apex"]):
        for W in (2.0, 30.0):
            out.append(dict(ind="holder", pol=0, az=i, p=np.array(ap) + [0.0, 0.0, W], n=np.array([0.0, 0.0, 1.0]), W=W))
    return out


def cmd_press(a):
    TI, CELL, C, reg = _setup()
    g = geometry()
    S = CELL.SIM
    qh = np.asarray(reg.q_home, float)
    rows, t0 = [], time.time()
    _G.clear()
    _G.update(g)
    SKIP_KEY[0] = None
    for k in press_points(g):
        PRESSED[0] = k["ind"] if k["ind"] != "holder" else None
        ok_tw, why = [], {}
        for tw in range(-20, 45, 10):
            good = True
            for dz in ((-1.5, 1.0) if k["ind"] != "holder" else (0.0,)):   # 1.5 mm off the surface, then 1 mm into it
                T_AF = T_AF_press(k["p"] - k["n"] * dz, k["n"], tw)
                T_BE = reg.T_BI @ T_AF @ C.inv(CELL.T_M_FROM_F) @ C.inv(reg.T_EM)
                ok, worst, j = TI._pose_ok(CELL, C, S, reg, "array", T_BE, APEX_P if k["ind"] != "holder" else None, qh)
                if not ok:
                    good = False
                    why[tw] = ("ik" if j is None else "j5/j6" if worst is None
                               else ",".join(sorted(n for n, v in worst.items() if v < TI.NEED.get(n, 15.0))))
                    break
            if good:
                ok_tw.append(tw)
        rows.append(dict(ind=k["ind"], pol=k["pol"], az=k["az"], W=k.get("W"), twists_ok=ok_tw, why=why))
        print(rows[-1], flush=True)
    summ = {}
    for name in NAMES:
        sel = [r for r in rows if r["ind"] == name]
        summ[name] = dict(points=len(sel), pressed=sum(bool(r["twists_ok"]) for r in sel),
                          at_twist_0=sum(0 in r["twists_ok"] for r in sel))
    p = OUT / "data" / f"{STEM}_press{a.tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(registration=str(TI.REG.relative_to(REPO)), array=ARRAY, summary=summ, rows=rows,
                                 seconds=round(time.time() - t0, 1)), indent=0, default=float))
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    h = sub.add_parser("homing")
    h.add_argument("--scan", action="store_true")
    pr = sub.add_parser("press")
    pr.add_argument("--tag", default="")
    for sp in (h, pr):
        sp.add_argument("--mouth-z", type=float, help="the keys' mouth height (mm)")
        sp.add_argument("--rows", help="front,back tops (mm)")
    a = ap.parse_args()
    if getattr(a, "mouth_z", None):
        ARRAY["mouth_z"] = a.mouth_z
    if getattr(a, "rows", None):
        f, b = (float(v) for v in a.rows.split(","))
        for name, top in (("sphere5", f), ("rod", f), ("sphere12", b)):
            ARRAY[name] = dict(ARRAY[name], top=top)
        ARRAY["holder"] = dict(ARRAY["holder"], z_bot=b - 14.95 - 12.0)
    dict(build=cmd_build, homing=cmd_homing, press=cmd_press)[a.cmd](a)


if __name__ == "__main__":
    main()
