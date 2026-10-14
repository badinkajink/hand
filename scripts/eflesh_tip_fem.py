#!/usr/bin/env python3
"""Stiffness and magnet travel of a printed fingertip from its G-code, on the GPU (2026-10-13).

The tip is the solid that PrusaSlicer extrudes: every layer's beads (scripts/gcode_audit.py) drawn with their widths at
0.025 mm, averaged onto 0.1 mm voxels, and each 0.2 mm layer stacked as two voxel layers; the magnet's cavity is filled
and taken as bonded TPU. The voxels go to the matrix-free solver of scripts/tpu_tip_fem_gpu.py (hex8 with incompatible
modes, one solid TPU, back and mount faces fixed, frictionless contact of the 12.5 mm screwdriver shaft at the contact
point) at increasing approach. Outputs per tip: approach at 0.5, 1 and 3 N, the pressure-law modulus E* fitted as in
scripts/tpu_tip_magnet.py, the twist arm, the magnet's travel per newton of normal load, and its field and force noise at
an MLX90393 behind the back face (scripts/tpu_tip_print_plan.py). The same path run on the gyroid tip's G-code checks it
against the finite-element rows of magnet_runs.jsonl.

    .venv/bin/python scripts/eflesh_tip_fem.py --tips gyroid20 eflesh_all eflesh_top
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
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gcode_audit as GA  # noqa: E402
import tpu_tip_design as TD  # noqa: E402
import tpu_tip_fem_gpu as GT  # noqa: E402
import tpu_tip_fem_tip as T  # noqa: E402

H = 0.1
SUB = 4                                          # raster pixels per voxel side (0.025 mm)
PART = (TD.Z_END, 2 * TD.Y_HALF, TD.X_FACE - TD.X_BACK)   # the tip printed face up: x' along the finger, y' across, z' up
OUT = ROOT / "docs/experiments/20261013-eflesh_printability/fem.jsonl"
B = ROOT / "logs/20261013-eflesh_build"
TIPS = {
    "gyroid20": (ROOT / "logs/20261013-eflesh_slice/tip_gyroid20_d5.gcode", "SR2 tip, gyroid 20 %, 2 walls, 4 top and 4 bottom layers"),
    "gyroid10": (ROOT / "logs/20261013-eflesh_slice/tip_gyroid10_d5.gcode", "SR2 tip, gyroid 10 %, 2 walls, 4 top and 4 bottom layers"),
    "eflesh_all": (B / "sr2tip_c8_E0035/20261013-sr2tip_c8_E0035.gcode", "SR2 tip, eFlesh lattice E 0.0035, 8 mm cells, 0.9 mm skin on every face"),
    "eflesh_open": (B / "sr2tip_c8_E0035_open/20261013-sr2tip_c8_E0035_open.gcode",
                    "SR2 tip, eFlesh lattice E 0.0035, 8 mm cells, open on every face as eFlesh prints it, solid base"),
    "eflesh_gap": (B / "sr2tip_c8_E0035_gapwalls/20261013-sr2tip_c8_E0035_gapwalls.gcode",
                   "SR2 tip, eFlesh lattice E 0.0035, 8 mm cells, palmar skin and side walls stopping 1.5 mm below it"),
    "eflesh_top": (B / "sr2tip_c8_E0035_topskin/20261013-sr2tip_c8_E0035_topskin.gcode",
                   "SR2 tip, eFlesh lattice E 0.0035, 8 mm cells, 0.9 mm skin on the palmar face only"),
}
MAG = ("d4.76x1.59", 5.0)                       # magnet and the depth of its top face below the palmar face


def raster(seg, lo, nx, ny):
    from PIL import ImageDraw
    img = Image.new("L", (nx * SUB, ny * SUB), 0)
    dr = ImageDraw.Draw(img)
    r = H / SUB
    for x0, y0, x1, y1, w, _, _ in seg:
        a = ((x0 - lo[0]) / r, (y0 - lo[1]) / r)
        b = ((x1 - lo[0]) / r, (y1 - lo[1]) / r)
        q = 0.5 * w / r
        dr.line([a, b], fill=255, width=max(1, int(round(w / r))))
        dr.ellipse((a[0] - q, a[1] - q, a[0] + q, a[1] + q), fill=255)
        dr.ellipse((b[0] - q, b[1] - q, b[0] + q, b[1] + q), fill=255)
    A = np.asarray(img, np.float32) / 255.0                     # rows y, columns x
    return A.reshape(ny, SUB, nx, SUB).mean((1, 3)).T >= 0.5    # (nx, ny)


def voxels(gcode):
    """Occupancy in the tip's frame (x palmar normal, y across, z along from the mount face), origin and magnet mask."""
    L, _ = GA.parse(gcode, 0)
    s = np.vstack([l["seg"] for l in L])
    cx = 0.5 * (min(s[:, 0].min(), s[:, 2].min()) + max(s[:, 0].max(), s[:, 2].max()))
    cy = 0.5 * (min(s[:, 1].min(), s[:, 3].min()) + max(s[:, 1].max(), s[:, 3].max()))
    lo = (cx - PART[0] / 2, cy - PART[1] / 2)
    ni, nj = int(math.ceil(PART[0] / H)), int(math.ceil(PART[1] / H))
    nk = int(round(max(l["z"] for l in L) / H))
    occ = np.zeros((ni, nj, nk), bool)
    for l in L:
        k1 = int(round(l["z"] / H)); k0 = int(round((l["z"] - (l["h"] or 0.2)) / H))
        occ[:, :, k0:k1] |= raster(l["seg"], lo, ni, nj)[:, :, None]
    # magnet cavity filled (bonded magnet); x' = Z_END - z, y' = y + Y_HALF, z' = x - X_BACK. The G-code was sliced on
    # 2026-10-13 from tips with the clearance pocket (TD.FE_CLEAR), before the press-fit pockets of bacc7750.
    pk, mg = TD.pocket(MAG[0], MAG[1], *TD.FE_CLEAR)
    xi = (np.arange(ni) + 0.5) * H; yj = (np.arange(nj) + 0.5) * H; zk = (np.arange(nk) + 0.5) * H
    X, Y, Z = np.meshgrid(xi, yj, zk, indexing="ij")
    rr = (X - (TD.Z_END - TD.Z_C)) ** 2 + (Y - TD.Y_HALF) ** 2
    cav = (rr <= (pk["r"] + 0.05) ** 2) & (Z >= pk["x_lo"] - TD.X_BACK) & (Z <= pk["x_hi"] - TD.X_BACK)
    mag = (rr <= mg["r"] ** 2) & (Z >= mg["x_lo"] - TD.X_BACK) & (Z <= mg["x_hi"] - TD.X_BACK)
    occ |= cav
    occ_t = np.ascontiguousarray(np.transpose(occ[::-1], (2, 1, 0)))
    mag_t = np.ascontiguousarray(np.transpose(mag[::-1], (2, 1, 0)))
    origin = np.array([TD.X_BACK, -TD.Y_HALF, TD.Z_END - ni * H])
    return occ_t, mag_t, origin, dict(offset_mm=[round(lo[0], 3), round(lo[1], 3)], layers=len(L))


def surface_gaps(Xf):
    """Gap to the cylinder from the analytic palmar surface where a face node lies within half a voxel of it."""
    X = Xf.copy()
    xt = TD.surface_x(X[:, 1], X[:, 2])
    use = np.abs(xt - X[:, 0]) <= 0.5 * H + 1e-6
    X[:, 0] = np.where(use, xt, X[:, 0])
    dz = np.clip(np.abs(X[:, 2] - TD.Z_C), 0, T.R_TOOL)
    return (TD.X_FACE + (T.R_TOOL - np.sqrt(T.R_TOOL ** 2 - dz ** 2))) - X[:, 0]


def rigid(Xm, um):
    c = Xm.mean(0); r = Xm - c; t = um.mean(0)
    A = np.zeros((3 * len(r), 3))
    for i in range(3):
        e = np.zeros(3); e[i] = 1.0
        A[:, i] = np.cross(e, r).ravel()
    w, *_ = np.linalg.lstsq(A, (um - t).ravel(), rcond=None)
    return t, w


def run(name, E=30.0, deltas=(0.02, 0.04, 0.08, 0.16, 0.32, 0.64), Fstop=3.5, approaches=None):
    import tpu_tip_magnet as TM
    import tpu_tip_print_plan as PP
    gcode, what = TIPS[name]
    t0 = time.time()
    occ, mag, origin, vinfo = voxels(gcode)
    tip = GT.GpuTip(occ, origin, H, E, 0.45)
    fixed_n = tip.fixed_dof[0::3]
    nodes = GT.face_nodes(tip.X, fixed_n)
    nodes = nodes[(tip.X[nodes, 0] >= TD.X_FACE - 1.0)]                  # the palmar face, not strut tops inside the lattice
    g0 = surface_gaps(tip.X[nodes])
    # magnet nodes: the nodes of its voxels
    idx = np.argwhere(mag)
    sn = (occ.shape[0] + 1, occ.shape[1] + 1, occ.shape[2] + 1)
    import tpu_tip_fem_core as C
    en = C.element_nodes(idx, sn, periodic=False).ravel()
    gidx = np.argwhere(occ)
    allnodes = np.unique(C.element_nodes(gidx, sn, periodic=False).ravel())
    mnodes = np.searchsorted(allnodes, np.unique(en))
    print(f"{name}: {tip.n_el} elements, {tip.ndof} DOF, {len(nodes)} face nodes, {len(mnodes)} magnet nodes, solid "
          f"{occ.mean():.3f} of the box, setup {tip.t_setup:.1f}s", flush=True)
    g_min = float(g0.min())                    # first touch: an open lattice meets the shaft at a strut end, not at the face
    # the pads' law on the analytic face (every point of the shape, not only strut ends)
    ys = np.arange(-TD.Y_HALF + H / 2, TD.Y_HALF, H); zs = np.arange(H / 2, TD.Z_END, H)
    Yl, Zl = np.meshgrid(ys, zs, indexing="ij")
    dzl = np.clip(np.abs(Zl - TD.Z_C), 0, T.R_TOOL)
    g_law = ((TD.X_FACE + (T.R_TOOL - np.sqrt(T.R_TOOL ** 2 - dzl ** 2))) - TD.surface_x(Yl, Zl)).ravel()
    A = np.full(len(g_law), H * H)
    rows, x, active = [], None, None
    plan = [g_min + deltas[0]]                 # 20 um past first touch, then three steps aimed at 0.55, 1.1 and 3.3 N
    fixed = approaches is not None
    if fixed:                                  # explicit approaches from first touch (an open lattice stiffens as struts touch)
        plan = [g_min + a for a in approaches]
    while plan and (fixed or len(rows) < 7):
        dl = plan.pop(0)
        ts = time.time()
        p, x, active, info = GT.contact(tip, nodes, g0, dl, x0=x, active0=active, tol=1e-6)
        m = GT.metrics(tip.X, nodes, p, g0, dl)
        u = x.view(-1, 3).cpu().numpy()
        t, w = rigid(tip.X[mnodes], u[mnodes])
        rows.append(dict(delta=dl - g_min, travel=dl, F=m["F"], arm=m["arm"], t_um=(t * 1e3).tolist(), w_mrad=(w * 1e3).tolist(),
                         cg=info["cg_iterations"], s=round(time.time() - ts, 1)))
        print(f"  delta {dl * 1e3:.0f} um: F {m['F']:.3f} N, arm {m['arm']:.3f} mm, magnet {t[0] * 1e3:.2f} um, "
              f"CG {info['cg_iterations']} in {time.time() - ts:.0f}s", flush=True)
        if m["F"] > Fstop:
            break
        a0 = dl - g_min
        if fixed:
            continue
        if len(rows) == 1 and m["F"] > 0:      # F ~ approach^1.1 from the first point
            plan = [g_min + min(1.5, a0 * (Ft / m["F"]) ** (1 / 1.1)) for Ft in (0.55, 1.1, 3.3)]
            plan = [d for d in plan if d - g_min > a0 * 1.05]
        elif not plan and m["F"] < 3.0:        # short of 3 N: extend along the last two points' exponent
            r0, r1 = rows[-2], rows[-1]
            n = max(0.8, math.log(r1["F"] / r0["F"]) / math.log(r1["delta"] / r0["delta"]))
            plan = [g_min + min(2.0, r1["delta"] * (3.3 / r1["F"]) ** (1 / n))]
    rows.sort(key=lambda r: r["delta"])
    Fs, ds = np.array([r["F"] for r in rows]), np.array([r["delta"] for r in rows])
    per = {}
    for F in (0.5, 1.0, 3.0):
        if Fs.max() < F:
            continue
        d = float(np.exp(np.interp(math.log(F), np.log(Fs), np.log(ds))))
        arm = float(np.interp(F, Fs, [r["arm"] for r in rows]))
        tvec = [float(np.interp(F, Fs, [r["t_um"][i] for r in rows])) / F for i in range(3)]
        wvec = [float(np.interp(F, Fs, [r["w_mrad"][i] for r in rows])) / F for i in range(3)]
        per[f"{F:g}"] = dict(delta_um=1e3 * d, arm_mm=arm, magnet_t_um=tvec, magnet_w_mrad=wvec)
    E_best, worst = TM.best_modulus(lambda dd: T.law_force(A, g_law, dd, 1.0), [float(k) for k in per],
                                    [per[k]["delta_um"] * 1e-3 for k in per])
    row = dict(tip=name, what=what, gcode=str(Path(gcode).relative_to(ROOT)), first_touch_mm=g_min, law="analytic face", E_s=E, nu=0.45,
               h_mm=H, n_el=tip.n_el,
               n_dof=tip.ndof, solid_of_box=float(occ.mean()), voxels=vinfo, steps=rows, per_force=per, E_best_MPa=E_best,
               approach_err_max=worst, magnet=MAG[0], depth_mm=MAG[1], t_total_s=round(time.time() - t0, 1),
               gpu_mem_GB=torch.cuda.max_memory_allocated() / 1e9, script="scripts/eflesh_tip_fem.py", when=time.strftime("%Y-%m-%d %H:%M"))
    if "1" in per:
        fr = dict(magnet=MAG[0], depth_mm=MAG[1], per_force={"1": per["1"]},
                  shear_z_magnet_t_um_per_N=[0, 0, 0], shear_z_magnet_w_mrad_per_N=[0, 0, 0],
                  shear_y_magnet_t_um_per_N=[0, 0, 0], shear_y_magnet_w_mrad_per_N=[0, 0, 0])
        row["field"] = {f"{g:g}": {k: v for k, v in PP.field(fr, g).items() if k in ("B0_mT", "S_normal_norm", "sigF_normal_N")}
                        for g in (1.0, 4.0)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n"); fh.flush(); os.fsync(fh.fileno())
    print(f"{name}: E* {E_best:.2f} MPa (approach error {worst * 100:.0f} %), approach at 1 N "
          f"{per.get('1', {}).get('delta_um', float('nan')):.1f} um, magnet {per.get('1', {}).get('magnet_t_um', [float('nan')])[0]:.2f} um/N, "
          f"noise {row.get('field', {}).get('1', {}).get('sigF_normal_N', float('nan')):.3f} N at 1 mm; {row['t_total_s']:.0f}s", flush=True)
    del tip
    torch.cuda.empty_cache()
    return row


def sections(names, out, px=16):
    """Section of each printed tip through the contact point (y = 0), palmar face up, as the solver sees it: TPU black,
    the bonded magnet red, one panel per tip with its label."""
    from PIL import ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", 28)
    panels = []
    for n in names:
        occ, mag, origin, _ = voxels(TIPS[n][0])
        j = int(round((0.0 - origin[1]) / H - 0.5))
        sec, msec = occ[:, j, :], mag[:, j, :]                       # (x, z)
        img = np.full(sec.shape + (3,), 255, np.uint8)
        img[sec] = (20, 20, 20)
        img[msec] = (200, 20, 40)
        img = np.flipud(img)                                          # palmar face up
        im = Image.fromarray(img).resize((int(sec.shape[1] * H * px), int(sec.shape[0] * H * px)), Image.NEAREST)
        canvas = Image.new("RGB", (im.width + 20, im.height + 52), (255, 255, 255))
        canvas.paste(im, (10, 46))
        ImageDraw.Draw(canvas).text((10, 8), SEC_LBL[n], fill=(0, 0, 0), font=font)
        panels.append(canvas)
    per_row = 3
    rows_ = [panels[i:i + per_row] for i in range(0, len(panels), per_row)]
    W = max(sum(p.width for p in r) + 20 * (len(r) - 1) for r in rows_)
    Hr = max(p.height for p in panels)
    Hh = Hr * len(rows_) + 10 * (len(rows_) - 1)
    sheet = Image.new("RGB", (W + W % 2, Hh + 40 + (Hh + 40) % 2), (255, 255, 255))
    for ri, r in enumerate(rows_):
        x = 0
        for p_ in r:
            sheet.paste(p_, (x, ri * (Hr + 10))); x += p_.width + 20
    d = ImageDraw.Draw(sheet)
    small = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", 26)
    d.line((12, Hh + 18, 12 + 5 * px, Hh + 18), fill=(0, 0, 0), width=3)
    d.text((20 + 5 * px, Hh + 6), "5 mm", fill=(0, 0, 0), font=small)
    sheet.save(out)
    print(out, sheet.size)


SEC_LBL = {"gyroid20": "Gyroid 20 %", "gyroid10": "Gyroid 10 %", "eflesh_all": "eFlesh + skin, every face",
           "eflesh_top": "eFlesh + palmar skin", "eflesh_open": "eFlesh, open",
           "eflesh_gap": "eFlesh + skin, gapped walls"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tips", nargs="+", default=list(TIPS))
    ap.add_argument("--sections", action="store_true", help="write the section image instead of solving")
    ap.add_argument("--approach", type=float, nargs="*", help="approaches from first touch, mm, in place of the aimed steps")
    a = ap.parse_args()
    if a.sections:
        sections(a.tips, ROOT / "docs/experiments/20261013-eflesh_printability/media/20261013-tip_sections.png")
    else:
        for n in a.tips:
            run(n, approaches=a.approach)
