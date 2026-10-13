#!/usr/bin/env python3
"""Finite elements of the whole printed fingertip with an embedded magnet (2026-10-12): contact stiffness and the
pressure-law modulus with and without the magnet, and the magnet's motion under normal and shear load at the contact
point, which a magnetometer behind the tip reads.

Geometry and pocket: scripts/tpu_tip_design.py (the whole block, no mount slot; owner 2026-10-12). Materials as in
scripts/tpu_tip_fem_tip.py: two 0.45 mm perimeters and four 0.2 mm skin layers around every outline, the pocket's
included (the slicer walls an internal void the same way), gyroid core homogenized from cells.jsonl, solid TPU
isotropic; the magnet NdFeB, E 160 GPa, nu 0.24, bonded to its pocket. Back face (x = -8.5) and mount face (z = 0) fixed.
Contact: the screwdriver cylinder (R 12.5 mm, axis along y) at z_c, frictionless normal contact on the face compliance;
shear: the 1 N contact's nodal forces turned tangential (along z, the finger, and y, across it), 1 N in total, the
contact sticking (Goodman, no normal-tangential coupling).

    .venv/bin/python scripts/tpu_tip_magnet.py run --recipes w95_20 --magnets none d4.76x1.59@3
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

OUT = ROOT / "docs/experiments/20261009-tpu_tip_fem/magnet_runs.jsonl"
H_AUX = 0.05
E_MAG, NU_MAG = 160e3, 0.24                       # NdFeB, MPa
GRID = dict(
    x=[(-8.5, -7.6, 0.45), (-7.6, 1.0, 0.8), (1.0, 7.0, 0.25), (7.0, 8.5, 0.2)],
    y=[(-7.4, -3.5, 0.3), (-3.5, 3.5, 0.3), (3.5, 7.4, 0.3)],
    z=[(0.0, 0.9, 0.45), (0.9, 7.0, 1.0), (7.0, 8.4, 0.35), (8.4, 13.3, 0.2), (13.3, 15.0, 0.35), (15.0, 18.0, 0.75),
       (18.0, 21.41, 0.35)])
RECIPES = {
    "w95_20": dict(core="homog", rho=0.2, E_s=30.0, nu=0.45, build="x"),
    "w95_10": dict(core="homog", rho=0.1, E_s=30.0, nu=0.45, build="x"),
    "w85_50": dict(core="homog", rho=0.5, E_s=12.0, nu=0.45, build="x"),
    "w85_30": dict(core="homog", rho=0.3, E_s=12.0, nu=0.45, build="x"),
}


def classes(magnet, depth, build="x"):
    """Classes on the auxiliary voxel grid: 0 void, 1 wall, 2 skin, 3 core, 4 magnet. Walls and skins follow every
    outline of the printed TPU, the pocket's included."""
    from scipy import ndimage
    occ, o = TD.occupancy(H_AUX, magnet, depth)
    tpu = occ == 1
    ax = {"x": 0, "y": 1, "z": 2}[build]
    dist = np.zeros(tpu.shape, np.float32)
    for k in range(tpu.shape[ax]):
        sl = [slice(None)] * 3
        sl[ax] = k
        s = tpu[tuple(sl)]
        if s.any():
            dist[tuple(sl)] = ndimage.distance_transform_edt(np.pad(s, 1))[1:-1, 1:-1] * H_AUX
    wall = tpu & (dist <= T.WALL + 1e-9)
    n = tpu.shape[ax]
    m = np.moveaxis(tpu, ax, -1)
    up = np.zeros(m.shape, np.float32)
    down = np.zeros(m.shape, np.float32)
    last = np.full(m.shape[:-1], -1, int)
    for k in range(n):
        last = np.where(~m[..., k], k, last)
        down[..., k] = np.where(m[..., k], (k - last) * H_AUX, 0)
    nxt = np.full(m.shape[:-1], n, int)
    for k in range(n - 1, -1, -1):
        nxt = np.where(~m[..., k], k, nxt)
        up[..., k] = np.where(m[..., k], (nxt - k) * H_AUX, 0)
    skin = np.moveaxis(m & ((up <= T.N_SKIN * T.LAYER + 1e-9) | (down <= T.N_SKIN * T.LAYER + 1e-9)), -1, ax) & ~wall
    cls = np.zeros(tpu.shape, np.uint8)
    cls[tpu] = 3
    cls[skin] = 2
    cls[wall] = 1
    cls[occ == 2] = 4
    return cls, o


def build(recipe, magnet, depth):
    grid = T.Grid(T.zoned_axis(GRID["x"]), T.zoned_axis(GRID["y"]), T.zoned_axis(GRID["z"]))
    cls, o = classes(magnet, depth, recipe.get("build", "x"))
    cc = T.sample_cells(grid, cls, o, H_AUX)
    mats, cmap = T.materials(recipe)
    mats = list(mats) + [C.iso_D(E_MAG, NU_MAG)]
    cmap = dict(cmap)
    cmap[4] = 2
    mat_id = -np.ones(cc.shape, int)
    for k, v in cmap.items():
        mat_id[cc == k] = v
    return T.TipModel(grid, mat_id, mats, verbose=True), cc


def gaps(model, nodes):
    """Initial gap to the cylinder from the analytic palmar surface (the grid's staircase on the fillets replaced by the
    surface height where a node lies within half a cell of it)."""
    X = model.X[nodes].copy()
    xt = TD.surface_x(X[:, 1], X[:, 2])
    xs = model.g.xs
    k = np.clip(np.searchsorted(xs, X[:, 0] - 1e-9), 1, len(xs) - 1)
    use = np.abs(xt - X[:, 0]) <= 0.5 * (xs[k] - xs[k - 1]) + 1e-6
    X[:, 0] = np.where(use, xt, X[:, 0])
    dz = np.clip(np.abs(X[:, 2] - TD.Z_C), 0, T.R_TOOL)
    return (TD.X_FACE + (T.R_TOOL - np.sqrt(T.R_TOOL ** 2 - dz ** 2))) - X[:, 0]


def rigid_fit(X, u):
    """Least-squares rigid motion of a node set: translation of the centroid (mm) and small rotation vector (rad)."""
    c = X.mean(0)
    r = X - c
    t = u.mean(0)
    A = np.zeros((3 * len(r), 3))
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        A[:, i] = np.cross(e, r).ravel()
    w, *_ = np.linalg.lstsq(A, (u - t).ravel(), rcond=None)
    return t, w


def best_modulus(L_of_d, forces, d_fe):
    """Pressure-law modulus minimizing the RMS log error of the law's approach against the FE approach."""
    def d_law(F, E):
        lo, hi = 0.0, 3.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if E * L_of_d(mid) < F else (lo, mid)
        return 0.5 * (lo + hi)

    def err(lnE):
        return float(np.sqrt(np.mean([(math.log(d_law(F, math.exp(lnE)) / d)) ** 2 for F, d in zip(forces, d_fe)])))
    a, b = math.log(0.3), math.log(300.0)
    g = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        c1, c2 = b - g * (b - a), a + g * (b - a)
        if err(c1) < err(c2):
            b = c2
        else:
            a = c1
    E = math.exp(0.5 * (a + b))
    worst = max(abs(d_law(F, E) / d - 1) for F, d in zip(forces, d_fe))
    return E, worst


def run(name, recipe, magnet, depth, forces=(0.5, 1.0, 3.0), out=OUT):
    t0 = time.time()
    model, cc = build(recipe, magnet, depth)
    nodes = model.face_nodes(zlim=(TD.Z_C - 3.5, TD.Z_C + 3.5))
    tc = time.time()
    W = model.compliance(nodes)
    tc = time.time() - tc
    g0 = gaps(model, nodes)
    A = T.tributary_areas(model, nodes)
    if magnet is not None:
        _, mg = TD.pocket(magnet, depth)
        X = model.X
        mnodes = np.where(TD.in_cyl(X[:, 0], X[:, 1], X[:, 2], dict(mg, r=mg["r"] + 1e-6, x_lo=mg["x_lo"] - 1e-6,
                                                                       x_hi=mg["x_hi"] + 1e-6)))[0]
    else:
        mnodes = np.array([], int)
    row = dict(recipe=name, **recipe, magnet=magnet or "none", depth_mm=depth, grid="magnet", n_dof=len(model.free),
               n_el=model.n_el, n_magnet_nodes=int(len(mnodes)), t_factor_s=model.t_factor, t_compliance_s=tc,
               script="scripts/tpu_tip_magnet.py", when=time.strftime("%Y-%m-%d %H:%M"))
    d_fe, per = [], {}
    p1 = None
    for F in forces:
        d, p, _ = T.force_controlled(W, g0, F)
        m = T.patch_metrics(model, nodes, p, g0, d)
        E_fit = F / max(T.law_force(A, g0, d, 1.0), 1e-30)
        rec = dict(delta_um=1e3 * d, E_law_fit_MPa=E_fit, arm_mm=m["arm"], r_rms_mm=m["r_rms"], half_z_mm=m["half_z"])
        if len(mnodes):
            act = np.where(p > 0)[0]
            Fn = np.zeros((len(act), 3))
            Fn[:, 0] = -p[act]
            u = model.solve_full((nodes[act], Fn))
            t, w = rigid_fit(model.X[mnodes], u[mnodes])
            rec.update(magnet_t_um=(1e3 * t).tolist(), magnet_w_mrad=(1e3 * w).tolist())
        per[f"{F:g}"] = rec
        d_fe.append(d)
        if abs(F - 1.0) < 1e-9:
            p1 = p
    E_best, worst = best_modulus(lambda dd: T.law_force(A, g0, dd, 1.0), forces, d_fe)
    row.update(per_force=per, E_best_MPa=E_best, approach_err_max=worst)
    if len(mnodes) and p1 is not None:
        act = np.where(p1 > 0)[0]
        for axis, k in (("z", 2), ("y", 1)):
            Ft = np.zeros((len(act), 3))
            Ft[:, k] = p1[act] / p1[act].sum()
            u = model.solve_full((nodes[act], Ft))
            t, w = rigid_fit(model.X[mnodes], u[mnodes])
            row[f"shear_{axis}_magnet_t_um_per_N"] = (1e3 * t).tolist()
            row[f"shear_{axis}_magnet_w_mrad_per_N"] = (1e3 * w).tolist()
            # the contact point's own tangential compliance, for comparison
            row[f"shear_{axis}_face_um_per_N"] = float(1e3 * (u[nodes[act], k] * p1[act]).sum() / p1[act].sum())
    row["t_total_s"] = time.time() - t0
    with open(out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    msg = f"{name} {row['magnet']}@{depth}: E* {E_best:.2f} MPa (worst {100 * worst:.0f} %), approach 1 N {per['1']['delta_um']:.1f} um"
    if len(mnodes):
        msg += (f", magnet x {per['1']['magnet_t_um'][0]:.1f} um/N at 1 N, shear z {row['shear_z_magnet_t_um_per_N'][2]:.1f}, "
                f"y {row['shear_y_magnet_t_um_per_N'][1]:.1f} um/N")
    print(msg + f" ({row['t_total_s']:.0f} s, {row['n_dof']} dof)", flush=True)
    return row


def parse_magnet(s):
    if s == "none":
        return None, None
    m, d = s.split("@")
    return m, float(d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--recipes", nargs="+", default=list(RECIPES))
    ap.add_argument("--magnets", nargs="+", default=["none", "d4.76x1.59@1.5", "d4.76x1.59@3", "d4.76x1.59@5",
                                                     "d3.18x1.59@3"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    done = set()
    if Path(a.out).exists():
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["recipe"], r["magnet"], r["depth_mm"]))
    for name in a.recipes:
        for ms in a.magnets:
            mag, dep = parse_magnet(ms)
            if (name, mag or "none", dep) in done:
                continue
            run(name, RECIPES[name], mag, dep, out=a.out)
