#!/usr/bin/env python3
"""One magnet against four in the printed fingertip (2026-10-13): finite elements of the whole tip with several embedded
magnets, under the screwdriver at the contact point and under small patch loads over the face.

Owner, 2026-10-13: "i'd like to embed ~4 magnets per finger. please analyze if smaller, more numerous magnets would be any
better than the 1 bigger magnet you've designed around, in terms of sensing precision/accuracy and material effect."

Model as scripts/tpu_tip_magnet.py (95A, gyroid 20 % homogenized from cells.jsonl, two walls and four skin layers around
every outline, each pocket's included; magnets NdFeB, bonded, which a press fit approaches; back and mount faces fixed),
on one grid for every layout so that layouts compare on equal terms. Loads:
  cylinder  the screwdriver shaft (R 12.5 mm) at the contact point, 0.5, 1 and 3 N: approach and the pad modulus E*;
  patches   1 N spread over the face as a Gaussian of sigma 0.6 mm at 15 positions (y -3, 0, 3 mm; z_c -4 ... +4 mm),
            normal, and tangential along y and z: each magnet's rigid motion (translation um/N, rotation mrad/N).
Rows: docs/experiments/20261013-magnet_layouts/fem.jsonl (one process per layout; the sparse factor is 6-9 GB).

    ~/miniconda3/envs/femtip/bin/python scripts/tpu_tip_magnet_layouts.py run --layout four_d3.18x1.59_d5
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_tip_design as TD  # noqa: E402
import tpu_tip_fem_core as C  # noqa: E402
import tpu_tip_fem_tip as T  # noqa: E402
import tpu_tip_magnet as TM  # noqa: E402

OUT = ROOT / "docs/experiments/20261013-magnet_layouts/fem.jsonl"
GRID = dict(
    x=[(-8.5, -7.6, 0.45), (-7.6, 1.0, 0.8), (1.0, 7.0, 0.3), (7.0, 8.5, 0.25)],
    y=[(-7.4, 7.4, 0.35)],
    z=[(0.0, 0.9, 0.45), (0.9, 5.0, 0.8), (5.0, 17.0, 0.35), (17.0, 18.0, 0.5), (18.0, 21.41, 0.35)])
RECIPE = TM.RECIPES["w95_20"]
PITCH_Y, PITCH_Z = 6.0, 7.0                         # 2 x 2 grid centred on the contact point


def grid4(d, t):
    return [(d, t, sy * PITCH_Y / 2, TD.Z_C + sz * PITCH_Z / 2) for sy in (-1, 1) for sz in (-1, 1)]


LAYOUTS = {   # magnets (diameter, thickness, y, z) with their top faces `depth` below the palmar face
    "none": dict(depth=None, mags=[]),
    "one_d4.76x1.59_d5": dict(depth=5.0, mags=[(4.76, 1.59, 0.0, TD.Z_C)]),
    "four_d3.18x1.59_d5": dict(depth=5.0, mags=grid4(3.18, 1.59)),
    "four_d3x1_d5": dict(depth=5.0, mags=grid4(3.0, 1.0)),
    "four_d3.18x1.59_d3": dict(depth=3.0, mags=grid4(3.18, 1.59)),
    "four_d4.76x1.59_d5": dict(depth=5.0, mags=grid4(4.76, 1.59)),
}
PATCH_Y = (-3.0, 0.0, 3.0)
PATCH_DZ = (-4.0, -2.0, 0.0, 2.0, 4.0)
SIGMA = 0.6


def magnet_cyl(m, depth):
    d, t, y, z = m
    hi = TD.X_FACE - depth
    return dict(r=d / 2, x_lo=hi - t, x_hi=hi, y=y, z=z)


def in_mag(X, Y, Z, c):
    return (X >= c["x_lo"]) & (X <= c["x_hi"]) & ((Y - c["y"]) ** 2 + (Z - c["z"]) ** 2 <= c["r"] ** 2)


def occupancy(h, lay, pad=2):
    lo = np.array([TD.X_BACK, -TD.Y_HALF, 0.0]) - pad * h
    hi = np.array([TD.X_FACE, TD.Y_HALF, TD.Z_END]) + pad * h
    n = np.ceil((hi - lo) / h).astype(int)
    xs, ys, zs = (lo[k] + h * (np.arange(n[k]) + 0.5) for k in range(3))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    occ = TD.inside_body(X, Y, Z).astype(np.uint8)
    for m in lay["mags"]:
        occ[in_mag(X, Y, Z, magnet_cyl(m, lay["depth"])) & (occ > 0)] = 2
    return occ, lo


def classes(lay):
    """As scripts/tpu_tip_magnet.classes, for any set of pockets (build direction x, the face up)."""
    real = TD.occupancy
    TD.occupancy = lambda h, magnet=None, depth=None, pad=2: occupancy(h, lay, pad)
    try:
        return TM.classes(None, None, "x")
    finally:
        TD.occupancy = real


def build(lay):
    grid = T.Grid(T.zoned_axis(GRID["x"]), T.zoned_axis(GRID["y"]), T.zoned_axis(GRID["z"]))
    cls, o = classes(lay)
    cc = T.sample_cells(grid, cls, o, TM.H_AUX)
    mats, cmap = T.materials(RECIPE)
    mats = list(mats) + [C.iso_D(TM.E_MAG, TM.NU_MAG)]
    cmap = dict(cmap)
    cmap[4] = 2
    mat_id = -np.ones(cc.shape, int)
    for k, v in cmap.items():
        mat_id[cc == k] = v
    return T.TipModel(grid, mat_id, mats, verbose=True)


def magnet_nodes(model, lay):
    X = model.X
    out = []
    for m in lay["mags"]:
        c = magnet_cyl(m, lay["depth"])
        c = dict(c, r=c["r"] + 1e-6, x_lo=c["x_lo"] - 1e-6, x_hi=c["x_hi"] + 1e-6)
        out.append(np.where(in_mag(X[:, 0], X[:, 1], X[:, 2], c))[0])
    return out


def motions(model, mn, u):
    return [[(1e3 * t).tolist(), (1e3 * w).tolist()] for t, w in (TM.rigid_fit(model.X[ids], u[ids]) for ids in mn)]


def run(name, out=OUT):
    lay = LAYOUTS[name]
    t0 = time.time()
    model = build(lay)
    mn = magnet_nodes(model, lay)
    # cylinder contact at the contact point: approach, E*, magnet motions at 1 N
    nodes = model.face_nodes(zlim=(TD.Z_C - 3.5, TD.Z_C + 3.5))
    W = model.compliance(nodes)
    g0 = TM.gaps(model, nodes)
    A = T.tributary_areas(model, nodes)
    per, d_fe = {}, []
    for F in (0.5, 1.0, 3.0):
        d, p, _ = T.force_controlled(W, g0, F)
        m = T.patch_metrics(model, nodes, p, g0, d)
        rec = dict(delta_um=1e3 * d, arm_mm=m["arm"])
        if mn:
            act = np.where(p > 0)[0]
            Fn = np.zeros((len(act), 3))
            Fn[:, 0] = -p[act]
            rec["magnets"] = motions(model, mn, model.solve_full((nodes[act], Fn)))
        per[f"{F:g}"] = rec
        d_fe.append(d)
    E_best, worst = TM.best_modulus(lambda dd: T.law_force(A, g0, dd, 1.0), (0.5, 1.0, 3.0), d_fe)
    # patch loads over the face
    face = model.face_nodes(zlim=(0.5, TD.Z_END - 0.5))
    Xf = model.X[face]
    patches = []
    if True:                                   # every layout, the one without magnets too: the face's own compliance
        for y in PATCH_Y:
            for dz in PATCH_DZ:
                z = TD.Z_C + dz
                wgt = np.exp(-((Xf[:, 1] - y) ** 2 + (Xf[:, 2] - z) ** 2) / (2 * SIGMA ** 2))
                wgt[wgt < 1e-4 * wgt.max()] = 0
                wgt /= wgt.sum()
                sel = wgt > 0
                rec = dict(y=y, z=z)
                for key, (k, sgn) in (("normal", (0, -1.0)), ("shear_y", (1, 1.0)), ("shear_z", (2, 1.0))):
                    Fv = np.zeros((sel.sum(), 3))
                    Fv[:, k] = sgn * wgt[sel]
                    u = model.solve_full((face[sel], Fv))
                    rec[key] = motions(model, mn, u) if mn else []
                    rec[key + "_face_um_per_N"] = float(1e3 * sgn * (u[face[sel], k] * wgt[sel]).sum())   # load-weighted
                patches.append(rec)
    row = dict(layout=name, depth_mm=lay["depth"], magnets=[list(m) for m in lay["mags"]], recipe="w95_20", grid=GRID,
               n_dof=len(model.free), n_el=model.n_el, n_magnet_nodes=[int(len(i)) for i in mn], per_force=per,
               E_best_MPa=E_best, approach_err_max=worst, patches=patches, sigma_mm=SIGMA, t_total_s=time.time() - t0,
               script="scripts/tpu_tip_magnet_layouts.py", when=time.strftime("%Y-%m-%d %H:%M"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    print(f"{name}: E* {E_best:.2f} MPa, approach at 1 N {per['1']['delta_um']:.1f} um, {len(patches)} patch positions, "
          f"{row['n_dof']} dof, {row['t_total_s']:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--layout", required=True, choices=list(LAYOUTS))
    a = ap.parse_args()
    done = {json.loads(l)["layout"] for l in open(OUT)} if OUT.exists() else set()
    if a.layout in done:
        print(a.layout, "done")
    else:
        run(a.layout)
