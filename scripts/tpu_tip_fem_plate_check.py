#!/usr/bin/env python3
"""Plate stiffness of the solid printed insert with quadratic tetrahedra: an independent check on the two hexahedral
tip models, the graded grid of tpu_tip_fem_tip.py and the 0.1 mm voxels of tpu_tip_fem_gpu.py, which give 680 and
845 N/mm for the solid 95A insert under a flat plate (E 30 MPa, nu 0.45, contact gaps from the STL surface).

Mesh: tetgen on `fingertip (3).stl` with a quality bound (radius-edge ratio) and a volume bound; quadratic Lagrange displacement (dolfinx
assembles the stiffness, CHOLMOD solves); the back face (x = -8.5 mm) and the mount face (z = 0) clamped as in both hex
models; every node of the flat palmar face (x = +8.5 mm) pushed in by delta along -x and free to slide (a frictionless
plate with the whole flat face in contact); stiffness = total reaction / delta. Rows go to plate_check.jsonl.

    ~/miniconda3/envs/femtip/bin/python scripts/tpu_tip_fem_plate_check.py --minratio 3.0 2.0
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
STL = Path.home() / "Downloads/fingertip (3).stl"
OUT = ROOT / "docs/experiments/20261009-tpu_tip_fem/plate_check.jsonl"


def tet_mesh(minratio, maxvol=5.0):
    """tetgen on the STL (its 6.5k surface triangles kept, quality bound minratio on the radius-edge ratio, volume
    bound maxvol mm^3); returns (nodes, tets). gmsh's surface remesh of this STL hangs in its 2D step."""
    import tetgen
    import trimesh
    m = trimesh.load(STL, force="mesh")
    tg = tetgen.TetGen(np.asarray(m.vertices, float), np.asarray(m.faces, np.int32))
    nodes, elems = tg.tetrahedralize(quality=True, minratio=minratio, maxvolume=maxvol)[:2]
    return np.asarray(nodes, float), np.asarray(elems, np.int64)


def stiffness(nodes, elems, E, nu, order):
    """Stiffness matrix (scipy CSR, unrolled DOFs 3 n + c) and node coordinates of the Lagrange space of the order."""
    from mpi4py import MPI
    import basix.ufl
    import ufl
    from dolfinx import fem, mesh as dm
    e = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
    dom = dm.create_mesh(MPI.COMM_WORLD, elems, e, nodes)
    V = fem.functionspace(dom, ("Lagrange", order, (3,)))
    mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))  # noqa: E731
    a = ufl.inner(2 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(3), eps(v)) * ufl.dx
    A = fem.assemble_matrix(fem.form(a))
    K = A.to_scipy().tocsr()
    X = V.tabulate_dof_coordinates()
    return K, X, dom.topology.index_map(3).size_local


def plate(K, X, delta=0.002, tol=1e-4):
    """The STL's planes: back x = -8.5, mount z = 0, flat palmar face x = +8.5 for |y| <= 4.4 (fillets beyond)."""
    from tpu_tip_fem_core import factor
    n = len(X)
    clamp = (X[:, 0] < -8.5 + tol) | (X[:, 2] < tol)
    face = (np.abs(X[:, 0] - 8.5) < tol) & ~clamp
    pres = np.zeros(3 * n, bool)
    val = np.zeros(3 * n)
    for c in range(3):
        pres[3 * np.where(clamp)[0] + c] = True
    pres[3 * np.where(face)[0]] = True
    val[3 * np.where(face)[0]] = -delta
    free = np.where(~pres)[0]
    p = np.where(pres)[0]
    t0 = time.time()
    Kff = K[free][:, free].tocsc()
    fac = factor(Kff)
    uf = fac(-(K[free][:, p] @ val[p]))
    u = val.copy()
    u[free] = uf
    r = K @ u
    rf = r[3 * np.where(face)[0]]
    F = -rf.sum()
    return dict(k_N_per_mm=float(F / delta), F=float(F), delta_mm=delta, n_face_nodes=int(face.sum()),
                tensile_share=float(rf[rf > 0].sum() / max(-rf[rf < 0].sum(), 1e-30)),
                face_y=[float(X[face, 1].min()), float(X[face, 1].max())], face_z=[float(X[face, 2].min()), float(X[face, 2].max())],
                t_solve_s=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minratio", type=float, nargs="+", default=[3.0, 2.0], help="tetgen radius-edge bound")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--E", type=float, default=30.0)
    ap.add_argument("--nu", type=float, default=0.45)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    for mv in a.minratio:
        t0 = time.time()
        nodes, elems = tet_mesh(mv)
        t1 = time.time()
        K, X, ntet = stiffness(nodes, elems, a.E, a.nu, a.order)
        t2 = time.time()
        r = plate(K, X)
        row = dict(case="solid_plate", minratio=mv, maxvol_mm3=5.0, order=a.order, E=a.E, nu=a.nu, n_tet=int(ntet), n_dof=3 * len(X),
                   volume_mm3=float(_volume(nodes, elems)), t_mesh_s=t1 - t0, t_assemble_s=t2 - t1,
                   rss_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, script="scripts/tpu_tip_fem_plate_check.py", **r)
        print(f"minratio {mv}: {ntet} tets, {3 * len(X)} dof, volume {row['volume_mm3']:.0f} mm3: plate {r['k_N_per_mm']:.1f} N/mm "
              f"({r['n_face_nodes']} face nodes, y {r['face_y']}, z {r['face_z']}, tensile share {r['tensile_share']:.3f}), "
              f"solve {r['t_solve_s']:.0f} s, rss {row['rss_GB']:.1f} GB", flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())


def _volume(nodes, elems):
    P = nodes[elems]
    return np.abs(np.einsum("ij,ij->i", np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), P[:, 3] - P[:, 0])).sum() / 6


if __name__ == "__main__":
    main()
