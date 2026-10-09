#!/usr/bin/env python3
"""Periodic infill cells of printed TPU and their homogenized elasticity (step 2 of the TPU tip job).

Cells (voxels of side h, periodic in x, y, z; z is the build direction):
  slicer   the gyroid as a slicer prints it: in each layer of height t_layer the toolpath is the zero set of
           g = sin X cos Y + sin Y cos Z + sin Z cos X at the layer's mid-height, X = 2.44 rho x / s (PrusaSlicer and
           OrcaSlicer FillGyroid: DensityAdjust 2.44, s the line spacing), a bead of width w around it, extruded over
           the layer. Period L = 2 pi s / (2.44 rho), rounded to whole layers. Beads of successive layers bond only
           where they overlap, so the cell is layered and anisotropic.
  sheet    the ideal sheet gyroid of the same period, |g| / |grad g| <= t/2, t set for the slicer cell's solid fraction.
  eflesh   one cut-cell of eFlesh's pattern 0646 from its inflator (stitch_cells_cli) at given (E_rel, nu) targets.

    python3 scripts/tpu_tip_fem_cells.py slicer --rho 0.2 --h 0.05
    python3 scripts/tpu_tip_fem_cells.py sweep --out docs/experiments/20261009-tpu_tip_fem/cells.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_tip_fem_core as C  # noqa: E402

LINE_W, LAYER, SPACING, ADJ = 0.45, 0.2, 0.407, 2.44     # mm, mm, mm, -
EFLESH = Path.home() / "wxie_workspace/eFlesh/microstructure"


def gyroid(X, Y, Z):
    return np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)


def slicer_period(rho, s=SPACING, layer=LAYER):
    L = 2 * math.pi * s / (ADJ * rho)
    return max(layer, round(L / layer) * layer), L


def slicer_layer_mask(L, z, h, w=LINE_W, sub=4):
    """In-plane occupancy (n x n) of one layer at height z: voxel centres within w/2 of the zero set of g(., ., z)."""
    from skimage.measure import find_contours
    from scipy.spatial import cKDTree
    n = int(round(L / h))
    k = 2 * math.pi / L
    hf = h / sub
    pad = int(math.ceil((w + 2 * h) / hf))
    m = n * sub
    u = (np.arange(-pad, m + pad) + 0.5) * hf
    U, V = np.meshgrid(u, u, indexing="ij")
    G = gyroid(k * U, k * V, k * z)
    pts = []
    for c in find_contours(G, 0.0):
        pts.append((c + 0.5 - pad) * hf)          # contour coordinates in mm (row = x, col = y)
    xc = (np.arange(n) + 0.5) * h
    XC, YC = np.meshgrid(xc, xc, indexing="ij")
    if not pts:
        return np.zeros((n, n), bool)
    P = np.concatenate(pts)
    d, _ = cKDTree(P).query(np.stack([XC.ravel(), YC.ravel()], 1))
    return (d <= w / 2).reshape(n, n)


def slicer_cell(rho, h=0.05, w=LINE_W, layer=LAYER, s=SPACING):
    """Voxel cell (n, n, n) of the printed gyroid at requested density rho. Returns (occ, L)."""
    L, L_exact = slicer_period(rho, s, layer)
    n = int(round(L / h))
    per = int(round(layer / h))
    if abs(n * h - L) > 1e-9 or abs(per * h - layer) > 1e-9 or n % per:
        raise ValueError(f"h {h} must divide the layer {layer} and the period {L}")
    occ = np.zeros((n, n, n), bool)
    for l in range(n // per):
        zc = (l + 0.5) * layer
        occ[:, :, l * per:(l + 1) * per] = slicer_layer_mask(L, zc, h, w)[:, :, None]
    return occ, L


def sheet_cell(L, rho_target, h=0.05):
    """Ideal sheet gyroid of period L at solid fraction rho_target (first-order distance |g| / |grad g|)."""
    n = int(round(L / h))
    k = 2 * math.pi / L
    x = (np.arange(n) + 0.5) * h * k
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    g = gyroid(X, Y, Z)
    gx = np.cos(X) * np.cos(Y) - np.sin(Z) * np.sin(X)
    gy = -np.sin(X) * np.sin(Y) + np.cos(Y) * np.cos(Z)
    gz = -np.sin(Y) * np.sin(Z) + np.cos(Z) * np.cos(X)
    dist = np.abs(g) / np.maximum(np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) * k, 1e-12)
    t = np.quantile(dist, rho_target)
    return dist <= t, 2 * t


def eflesh_cell(E_rel=0.0035, nu=0.09, cell=8.0, h=0.1, work=None, res=50):
    """One eFlesh pattern-0646 cell from the inflator, voxelized on [0, cell)^3."""
    from tpu_tip_fem_voxel import voxelize
    import trimesh
    work = Path(work or "/tmp")
    work.mkdir(parents=True, exist_ok=True)
    m2g_dir = EFLESH / "matopt/tools/material2geometry"
    sys.path.insert(0, str(m2g_dir))
    from material2geometry import Material2Geometry
    m2g = Material2Geometry(in_path=str(m2g_dir / "0646_geo_1_coeffs.txt"))
    p = [float(v) for v in m2g.evaluate(nu, E_rel)]
    pat = EFLESH / "microstructure_inflators/data/patterns/3D/reference_wires/pattern0646.wire"
    js = work / f"eflesh_{E_rel:g}_{nu:g}.json"
    js.write_text(json.dumps([{"params": p, "symmetry": "Cubic", "pattern": str(pat), "index": [0, 0, 0]}]))
    obj = work / f"eflesh_{E_rel:g}_{nu:g}_{cell:g}.obj"
    cli = EFLESH / "microstructure_inflators/build/isosurface_inflator/stitch_cells_cli"
    subprocess.run([str(cli), "-p", str(js), "--gridSize", f"{cell:g}", "-o", str(obj), "-r", str(res)], check=True,
                   stdout=subprocess.DEVNULL)
    mesh = trimesh.load(obj, force="mesh")
    n = int(round(cell / h))
    occ, _ = voxelize(np.asarray(mesh.vertices), np.asarray(mesh.faces), h, origin=np.zeros(3), shape=(n, n, n))
    return occ, dict(params=p, obj=str(obj), watertight=bool(mesh.is_watertight))


def summarize(C6, E_s=1.0):
    e = C.engineering(C6)
    d = {k: float(v) for k, v in e.items()}
    d.update(E45_xz=float(C.directional_E(C6, [1, 0, 1])), E111=float(C.directional_E(C6, [1, 1, 1])))
    d["zener_xy"] = float(C6[5, 5] / (0.5 * (C6[0, 0] - C6[0, 1]))) if abs(C6[0, 0] - C6[0, 1]) > 0 else None
    return d


def run_case(kind, rho=None, h=0.05, nu=0.45, out=None, E_rel=0.0035, nu_t=0.09, cell=8.0, work=None, solver="cholmod"):
    t0 = time.time()
    info = {}
    if kind == "slicer":
        occ, L = slicer_cell(rho, h)
        info.update(rho_req=rho, period_mm=L)
    elif kind == "sheet":
        L, _ = slicer_period(rho)
        occ_s, _ = slicer_cell(rho, h)
        occ, t = sheet_cell(L, occ_s.mean(), h)
        info.update(rho_req=rho, period_mm=L, sheet_t_mm=t)
    elif kind == "eflesh":
        occ, meta = eflesh_cell(E_rel, nu_t, cell, h, work)
        info.update(E_rel_target=E_rel, nu_target=nu_t, cell_mm=cell, **meta)
    else:
        raise ValueError(kind)
    t_geo = time.time() - t0
    r = C.homogenize(occ, h, 1.0, nu, solver=solver)
    row = dict(kind=kind, h_mm=h, nu_s=nu, rho=float(occ.mean()), rho_kept=r["rho"], n_el=r["n_el"], n_dof=r["n_dof"],
               nnz_L=r.get("nnz_L"), t_geometry_s=t_geo, t_assemble_s=r["t_assemble"], t_factor_s=r["t_factor"],
               t_solve_s=r["t_solve"], solver=solver, amg=r.get("amg"), C=r["C"].tolist(), **summarize(r["C"]), **info,
               script="scripts/tpu_tip_fem_cells.py")
    try:
        row["git_rev"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()
    except Exception:
        pass
    print(f"{kind} rho_req {rho} h {h}: rho {row['rho']:.4f} (kept {row['rho_kept']:.4f}), Ex {row['Ex']:.5f} Ey "
          f"{row['Ey']:.5f} Ez {row['Ez']:.5f} Gxy {row['Gxy']:.5f} Gxz {row['Gxz']:.5f} nu_xy {row['nu_xy']:.3f} "
          f"nu_xz {row['nu_xz']:.3f} (per unit E_s); {row['n_dof']} dof, nnz(L) {row.get('nnz_L')}, "
          f"geometry {t_geo:.1f}s total {time.time() - t0:.1f}s", flush=True)
    if out:
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["slicer", "sheet", "eflesh"])
    ap.add_argument("--rho", type=float, nargs="*", default=[0.2])
    ap.add_argument("--h", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=0.45)
    ap.add_argument("--E-rel", type=float, default=0.0035)
    ap.add_argument("--nu-target", type=float, default=0.09)
    ap.add_argument("--cell", type=float, default=8.0)
    ap.add_argument("--work", default=str(ROOT / "assets/fem/20261009-tpu_tip"))
    ap.add_argument("--out")
    ap.add_argument("--solver", choices=["cholmod", "amg"], default="cholmod")
    a = ap.parse_args()
    if a.kind == "eflesh":
        run_case("eflesh", h=a.h, nu=a.nu, out=a.out, E_rel=a.E_rel, nu_t=a.nu_target, cell=a.cell, work=a.work,
                 solver=a.solver)
        return
    for rho in a.rho:
        run_case(a.kind, rho, a.h, a.nu, a.out, solver=a.solver)


if __name__ == "__main__":
    main()
