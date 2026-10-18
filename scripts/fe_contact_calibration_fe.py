#!/usr/bin/env python3
"""Finite elements of the four 85A printed fingertips under the calibration probes (phase A of
docs/experiments/20261018-fe_contact_calibration/plan.json).

Model: scripts/tpu_tip_magnet.py's printed tip (two 0.45 mm perimeters and four 0.2 mm skins of solid TPU around every
outline, the pockets' included, over the homogenized gyroid core; bonded NdFeB magnets) with recipe w85_30 (solid 12 MPa,
NinjaFlex 85A standing in for Siraya 85A, Poisson 0.45, gyroid 30 %) in each tip's print orientation (block: back face on
the bed, layers across x; capsule and half-ellipsoids upright, layers across z). The shape's geometry hooks come from
scripts/curved_tips_fe.install; the four 3 x 2 mm magnets sit at y +-3 mm, z_c +-3.5 mm with their tops at the print
sheet's depth (5, 5, 5.5, 7 mm) through scripts/tpu_tip_magnet_layouts.occupancy. Back face (x = -8.5 mm) and mount face
(z = 0) fixed. Grid: scripts/curved_tips_stiffness.fe_grid with its fine y zone widened over both pocket columns
(y -4.6 to 6.0 mm at 0.35 mm); `--grid fine` refines the contact region of the capsule (0.2 mm over |y| <= 2 mm and
|z - z_c| <= 2.5 mm) for the grid check.

Probes, rigid and frictionless, pressed along -x (the palmar normal) from first touch of the design surface; tip-frame
positions (y across the width, z toward the mount, the contact point at 0):
  s_*        sphere of radius 2 mm at the apex, y 2.6, 4.1 and 5.2 mm, z +4, -4 and -7 mm
  cx<R>_z*   cylinder across the finger (axis along y) of radius 5, 8 and 12.5 mm at z 0, +4 and -4 mm
  cz<R>_y0   cylinder along the finger (axis along z) of radius 8 and 12.5 mm at y 0
Loads 0.5, 1, 3 and 6 N. Readings per probe and load: approach (um), local stiffness dF/d(approach) (N/mm, from 0.95
and 1.05 of the load), contact nodes, pressure centre, pressure-weighted RMS radius r_rms and mean radius r_bar about
the centre (mm), half-extents in y and z. Under the 12.5 mm rod at z 0 at 1 N also the face's tangential compliance along
y and z (sticking contact, Goodman) and each magnet's rigid motion.

    ~/.claude/bin/resguard.sh run --mem 6G --cpu 900 -- taskset -c 4-11 env OMP_NUM_THREADS=8 \\
        OPENBLAS_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0 /usr/bin/time -v ~/miniconda3/envs/femtip/bin/python \\
        scripts/fe_contact_calibration_fe.py fe --tip cap_g30
    ~/miniconda3/envs/femtip/bin/python scripts/fe_contact_calibration_fe.py fe --tip cap_g30 --count
Rows: docs/experiments/20261018-fe_contact_calibration/fe/fe.jsonl (one fsynced row per tip, grid and probe),
fe/meta.jsonl (grid, dof, timings, peak memory), fe/patch_<tip>_<grid>.npz (nodal forces under the rods at 1 N).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tip_codesign_geometry as TG  # noqa: E402

OUT = ROOT / "docs/experiments/20261018-fe_contact_calibration/fe"
SCRIPT = "scripts/fe_contact_calibration_fe.py"
TIPS = {   # name: (finite-element key, sheet depth of the magnets' tops in mm, print orientation)
    "block4_g30": ("tcd_rs3_re3", 5.0, "x"),
    "block_td": ("tpu_tip_design", 5.0, "x"),     # the readiness study's block surface (r_side 2.7, r_end 3.0 mm)
    "cap_g30": ("tcd_cap8.5", 5.0, "z"),
    "ell16_g30": ("tcd_ell7.4x8.5x16x32", 5.5, "z"),
    "ell11_g30": ("tcd_ell7.4x8.5x11x22", 7.0, "z"),
}
MAGNET = (3.0, 2.0)
FORCES = (0.5, 1.0, 3.0, 6.0)
Z_C = 21.41 - 10.55


def probes():
    """(name, kind, radius mm, y_tip mm, z_tip mm); kind 'sphere', 'cx' (axis along y) or 'cz' (axis along z)."""
    out = [(f"s_{n}", "sphere", 2.0, y, z) for n, y, z in
           (("apex", 0.0, 0.0), ("y2.6", 2.6, 0.0), ("y4.1", 4.1, 0.0), ("y5.2", 5.2, 0.0), ("z+4", 0.0, 4.0),
            ("z-4", 0.0, -4.0), ("z-7", 0.0, -7.0))]
    for R in (5.0, 8.0, 12.5):
        for z in (0.0, 4.0, -4.0):
            out.append((f"cx{R:g}_z{z:+g}".replace("+0", "0"), "cx", R, 0.0, z))
    for R in (8.0, 12.5):
        out.append((f"cz{R:g}_y0", "cz", R, 0.0, 0.0))
    return out


def grid(kind="base"):
    """Zones per axis (insert frame, mm): curved_tips_stiffness.fe_grid for the fixed back face at x = -8.5 mm with the
    fine y zone over both pocket columns."""
    g = dict(x=[(-8.5, -7.6, 0.45), (-7.6, 1.0, 1.1), (1.0, 5.4, 0.55), (5.4, 8.5, 0.25)],
             y=[(-7.4, -4.6, 0.7), (-4.6, 6.0, 0.35), (6.0, 7.4, 0.35)],
             z=[(0.0, 6.0, 0.8), (6.0, 18.5, 0.3), (18.5, 21.41, 0.4)])
    if kind == "fine":
        g["y"] = [(-7.4, -4.6, 0.7), (-4.6, -2.0, 0.35), (-2.0, 2.0, 0.2), (2.0, 6.0, 0.35), (6.0, 7.4, 0.35)]
        g["z"] = [(0.0, 6.0, 0.8), (6.0, Z_C - 2.5, 0.3), (Z_C - 2.5, Z_C + 2.5, 0.2), (Z_C + 2.5, 18.5, 0.3),
                  (18.5, 21.41, 0.4)]
    return g


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def rows(path):
    out = []
    if Path(path).exists():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


# ------------------------------------------------------------------------------------------ probe geometry (insert frame)

def first_touch(sh, kind, R, y0, z0):
    """Probe reference height (insert frame, mm) at first touch of the analytic surface: the sphere's centre, or a point
    on the cylinder's axis, on a 0.02 mm grid."""
    if kind == "sphere":
        g = np.arange(-R, R + 1e-9, 0.02)
        Y, Z = np.meshgrid(y0 + g, z0 + g, indexing="ij")
        rho2 = (Y - y0) ** 2 + (Z - z0) ** 2
        ok = rho2 < R ** 2
        xs = sh.fe_surface_x(Y[ok], Z[ok])
        return float(np.nanmax(xs + np.sqrt(R ** 2 - rho2[ok])))
    if kind == "cx":
        h = min(R, 6.0)
        zz = z0 + np.arange(-h, h + 1e-9, 0.02)
        yy = np.arange(-7.4, 7.4 + 1e-9, 0.1)
        Y, Z = np.meshgrid(yy, zz, indexing="ij")
        return float(np.nanmax(sh.fe_surface_x(Y, Z) + np.sqrt(np.clip(R ** 2 - (Z - z0) ** 2, 0, None))))
    h = min(R, 7.4)
    yy = y0 + np.arange(-h, h + 1e-9, 0.02)
    zz = np.arange(0.0, 21.41 + 1e-9, 0.1)
    Y, Z = np.meshgrid(yy, zz, indexing="ij")
    return float(np.nanmax(sh.fe_surface_x(Y, Z) + np.sqrt(np.clip(R ** 2 - (Y - y0) ** 2, 0, None))))


def probe_surface(kind, R, xc, y0, z0, Y, Z):
    """x (insert frame, mm) of the probe's lower surface above (Y, Z); +inf outside its footprint."""
    out = np.full(np.shape(Y), np.inf)
    if kind == "sphere":
        r2 = (Y - y0) ** 2 + (Z - z0) ** 2
    elif kind == "cx":
        r2 = (Z - z0) ** 2
    else:
        r2 = (Y - y0) ** 2
    ok = r2 < R ** 2
    out[ok] = xc - np.sqrt(R ** 2 - r2[ok])
    return out


# ------------------------------------------------------------------------------------------ model

def layout(depth):
    return dict(depth=depth, mags=[(MAGNET[0], MAGNET[1], sy * 3.0, Z_C + sz * 3.5) for sy in (-1, 1) for sz in (-1, 1)])


def classes(lay, build):
    """tpu_tip_magnet.classes (walls and skins along every outline, the pockets' included) for the four pockets of
    `lay`, in the print orientation `build`."""
    import tpu_tip_design as TD
    import tpu_tip_magnet as TM
    import tpu_tip_magnet_layouts as TL
    real = TD.occupancy
    TD.occupancy = lambda h, magnet=None, depth=None, pad=2: TL.occupancy(h, lay, pad)
    try:
        return TM.classes(None, None, build)
    finally:
        TD.occupancy = real


class _TDBlock:
    """tpu_tip_design's own block (the readiness study's finite-element surface) with the shape interface used here."""

    def __init__(self):
        import tpu_tip_design as TD
        self.fe_inside, self.fe_surface_x = TD.inside_body, TD.surface_x


def mat_ids(name, gkind):
    """(grid, mat_id, mats, layout) of a tip; the shape's hooks installed in tpu_tip_design."""
    import curved_tips_fe as CF
    import tpu_tip_fem_core as C
    import tpu_tip_fem_tip as T
    import tpu_tip_magnet as TM
    key, depth, build = TIPS[name]
    sh = _TDBlock() if key == "tpu_tip_design" else TG.shape(key)
    if key != "tpu_tip_design":
        CF.install(sh)
    g = grid(gkind)
    G = T.Grid(T.zoned_axis(g["x"]), T.zoned_axis(g["y"]), T.zoned_axis(g["z"]))
    lay = layout(depth)
    cls, o = classes(lay, build)
    counts = {k: int((cls == v).sum()) for k, v in (("wall", 1), ("skin", 2), ("core", 3), ("magnet", 4))}
    cc = T.sample_cells(G, cls, o, TM.H_AUX)
    del cls
    mats, cmap = T.materials(dict(TM.RECIPES["w85_30"], build=build))
    mats = list(mats) + [C.iso_D(TM.E_MAG, TM.NU_MAG)]
    cmap = dict(cmap)
    cmap[4] = 2
    mat_id = -np.ones(cc.shape, int)
    for k, v in cmap.items():
        mat_id[cc == k] = v
    return sh, G, mat_id, mats, lay, counts


def count_dof(G, mat_id):
    import tpu_tip_fem_core as C
    idx = np.argwhere(mat_id >= 0)
    en = C.element_nodes(idx, G.nn, periodic=False)
    used = np.unique(en)
    gi = used // (G.nn[1] * G.nn[2])
    gk = used % G.nn[2]
    x, z = G.xs[gi], G.zs[gk]
    fixed = (x <= -8.5 + 1e-9) | (z <= 1e-9)
    return int(3 * (~fixed).sum()), int(len(idx))


def magnet_nodes(model, lay):
    import tpu_tip_magnet_layouts as TL
    return TL.magnet_nodes(model, lay)


def solve_probe(W, X, g0, F):
    """(approach mm, nodal forces) at total load F, and the local stiffness N/mm from 0.95 F and 1.05 F."""
    import tpu_tip_fem_tip as T
    d, p, _ = T.force_controlled(W, g0, F)
    dlo = T.force_controlled(W, g0, 0.95 * F)[0]
    dhi = T.force_controlled(W, g0, 1.05 * F)[0]
    k = 0.1 * F / (dhi - dlo) if dhi - dlo > 1e-9 else float("nan")
    return d, p, k


def patch(X, p):
    act = p > 1e-12 * p.sum()
    w = p[act]
    yz = X[act, 1:]
    c = (w[:, None] * yz).sum(0) / w.sum()
    r = np.linalg.norm(yz - c, axis=1)
    return dict(n_contact=int(act.sum()), centre_yz_mm=c.tolist(), r_rms_mm=float(np.sqrt((w * r ** 2).sum() / w.sum())),
                r_bar_mm=float((w * r).sum() / w.sum()), half_y_mm=float(np.abs(yz[:, 0] - c[0]).max()),
                half_z_mm=float(np.abs(yz[:, 1] - c[1]).max()))


def cmd_fe(a):
    import tpu_tip_fem_tip as T
    import tpu_tip_magnet as TM
    name = a.tip
    have = {r["probe"] for r in rows(OUT / "fe.jsonl") if r["tip"] == name and r["grid"] == a.grid}
    t0 = time.time()
    sh, G, mat_id, mats, lay, counts = mat_ids(name, a.grid)
    n_dof, n_el = count_dof(G, mat_id)
    t_cls = time.time() - t0
    print(f"{name} grid {a.grid}: {n_el} elements, {n_dof} dof, aux voxels {counts}, classes {t_cls:.0f} s", flush=True)
    if a.count:
        return
    if n_dof > a.max_dof:
        raise SystemExit(f"{n_dof} dof above --max-dof {a.max_dof}")
    todo = [p for p in probes() if p[0] not in have]
    if not todo:
        print(name, a.grid, "done", flush=True)
        return
    model = T.TipModel(G, mat_id, mats, verbose=True)
    del mat_id
    nodes = model.face_nodes(zlim=(0.4, 21.41))
    X = model.X[nodes].copy()
    xt = sh.fe_surface_x(X[:, 1], X[:, 2])
    stair = np.where(np.isfinite(xt), X[:, 0] - xt, np.nan)
    X[:, 0] = np.where(np.isfinite(xt), xt, X[:, 0])
    tc = time.time()
    W = model.compliance(nodes, block=100)
    tc = time.time() - tc
    A = T.tributary_areas(model, nodes)
    mn = magnet_nodes(model, lay)
    key, depth, build = TIPS[name]
    base = dict(tip=name, fe_key=key, grid=a.grid, recipe="w85_30", build=build, magnets="four 3 x 2 mm",
                magnet_top_depth_mm=depth, n_dof=len(model.free), n_el=model.n_el, n_face_nodes=len(nodes),
                n_magnet_nodes=[int(len(i)) for i in mn], staircase_mm=[float(np.nanpercentile(stair, q)) for q in (5, 50, 95)],
                script=SCRIPT)
    print(f"{name}: {len(nodes)} face nodes, compliance {tc:.0f} s, magnet nodes {base['n_magnet_nodes']}", flush=True)
    maps = {}
    for pname, kind, R, y_t, z_t in todo:
        z_ins = TG.Z_MOUNT * 1e3 - z_t
        xc = first_touch(sh, kind, R, y_t, z_ins)
        surf = probe_surface(kind, R, xc, y_t, z_ins, X[:, 1], X[:, 2])
        g0 = np.maximum(surf - X[:, 0], 0.0)
        cand = np.where(g0 < 1.5)[0]
        Wc, gc, Xc = W[np.ix_(cand, cand)], g0[cand], X[cand]
        res = {}
        t1 = time.time()
        for F in FORCES:
            d, p, k = solve_probe(Wc, Xc, gc, F)
            res[f"{F:g}"] = dict(approach_um=1e3 * d, k_N_per_mm=k, **patch(Xc, p))
            if abs(F - 1.0) < 1e-9 and kind == "cx" and z_t == 0.0 and R in (8.0, 12.5):
                act = p > 1e-12
                maps[f"cx{R:g}"] = (cand[act], p[act])
            if abs(F - 1.0) < 1e-9 and kind == "cx" and z_t == 0.0 and R == 12.5:
                act = np.where(p > 1e-12)[0]
                ids = nodes[cand[act]]
                extra = {}
                for axis, kk in (("y", 1), ("z", 2)):
                    Ft = np.zeros((len(act), 3))
                    Ft[:, kk] = p[act] / p[act].sum()
                    u = model.solve_full((ids, Ft))
                    extra[f"shear_{axis}_face_um_per_N"] = float(1e3 * (u[ids, kk] * p[act]).sum() / p[act].sum())
                    extra[f"shear_{axis}_magnets"] = [[(1e3 * t).tolist(), (1e3 * w).tolist()]
                                                      for t, w in (TM.rigid_fit(model.X[i], u[i]) for i in mn)]
                Fn = np.zeros((len(act), 3))
                Fn[:, 0] = -p[act]
                u = model.solve_full((ids, Fn))
                extra["normal_magnets_um_mrad"] = [[(1e3 * t).tolist(), (1e3 * w).tolist()]
                                                   for t, w in (TM.rigid_fit(model.X[i], u[i]) for i in mn)]
                res["1"].update(extra)
        row = dict(base, probe=pname, kind=kind, R_mm=R, y_mm=y_t, z_mm=z_t, x_first_touch_mm=xc,
                   n_candidates=int(len(cand)), per_force=res, t_probe_s=round(time.time() - t1, 1),
                   when=time.strftime("%Y-%m-%d %H:%M"))
        append(OUT / "fe.jsonl", row)
        print(f"  {pname:10s} " + "  ".join(f"{F}: {v['approach_um']:6.1f} um {v['k_N_per_mm']:5.2f} N/mm "
                                           f"rbar {v['r_bar_mm']:.2f}" for F, v in res.items()), flush=True)
    if maps:
        np.savez(OUT / f"patch_{name}_{a.grid}.npz", y=X[:, 1], z=X[:, 2], A=A,
                 **{f"{k}_idx": v[0] for k, v in maps.items()}, **{f"{k}_p": v[1] for k, v in maps.items()})
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    append(OUT / "meta.jsonl", dict(base, grid_zones=grid(a.grid), aux_voxels=counts, t_classes_s=t_cls,
                                    t_assemble_s=model.t_assemble, t_factor_s=model.t_factor, t_compliance_s=tc,
                                    t_total_s=time.time() - t0, peak_rss_GB=peak, when=time.strftime("%Y-%m-%d %H:%M")))
    print(f"{name} {a.grid}: {time.time() - t0:.0f} s, peak RSS {peak:.2f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fe")
    f.add_argument("--tip", required=True, choices=list(TIPS))
    f.add_argument("--grid", default="base", choices=["base", "fine"])
    f.add_argument("--count", action="store_true", help="voxel classes and dof only, no solve")
    f.add_argument("--max-dof", type=int, default=190000)
    a = ap.parse_args()
    {"fe": cmd_fe}[a.cmd](a)


if __name__ == "__main__":
    main()
