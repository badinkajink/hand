#!/usr/bin/env python3
"""Matrix-free GPU solver for the voxelized printed fingertip (resolved gyroid), with frictionless contact of a rigid
indenter under displacement control (active set). Runs in the project venv (torch with CUDA); reads the voxels written by
tpu_tip_fem_tip.resolved_voxels.

Each voxel is a hex8 element with Wilson-Taylor incompatible modes (tpu_tip_fem_core.hex_im), all of one solid
material; K u is computed element by element on the GPU in float32 (dot products in float64), and conjugate gradients
with a Jacobi preconditioner solve the free DOFs. Mount: back face (x = -8.5 mm) and mount face (z = 0) fixed. Contact
nodes: the palmar surface, normal direction x; an active node has its x displacement prescribed to the indenter
surface, and the active set is updated from reactions and penetrations.

    .venv/bin/python scripts/tpu_tip_fem_gpu.py VOXELS.npz --indenter cyl --delta 0.02 0.05 0.1 --out rows.jsonl
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_tip_fem_core as C  # noqa: E402

X_FACE, X_BACK, Z_END = 8.5, -8.5, 21.41
Z_C = Z_END - 10.55
R_TOOL = 12.5


class GpuTip:
    def __init__(self, occ, origin, h, E=30.0, nu=0.45, dev="cuda"):
        t0 = time.time()
        self.dev = dev
        idx = np.argwhere(occ)
        nx, ny, nz = occ.shape
        sn = (nx + 1, ny + 1, nz + 1)
        en = C.element_nodes(idx, sn, periodic=False)
        used, inv = np.unique(en, return_inverse=True)
        en = inv.reshape(en.shape)
        self.nn = len(used)
        gi = used // (sn[1] * sn[2])
        gj = (used // sn[2]) % sn[1]
        gk = used % sn[2]
        self.X = np.stack([origin[0] + gi * h, origin[1] + gj * h, origin[2] + gk * h], 1)
        Ke, _ = C.hex_im(h, h, h, nu=nu)
        self.Ke = torch.tensor(Ke * E, dtype=torch.float32, device=dev)
        dof = (3 * en[:, :, None] + np.arange(3)[None, None, :]).reshape(len(en), 24)
        self.dof = torch.tensor(dof, dtype=torch.int64, device=dev)
        self.ndof = 3 * self.nn
        self.n_el = len(en)
        diag = np.zeros(self.ndof)
        np.add.at(diag, dof.ravel(), np.tile(np.diag(Ke * E), len(en)))
        fixed = (self.X[:, 0] <= self.X[:, 0].min() + 1e-6) | (self.X[:, 2] <= self.X[:, 2].min() + 1e-6)
        self.n_fixed = int(fixed.sum())
        self.fixed_dof = np.repeat(fixed, 3)
        self.diag = torch.tensor(diag, dtype=torch.float64, device=dev)
        self.t_setup = time.time() - t0

    def matvec(self, u):
        ue = u.float()[self.dof]                       # (n_e, 24)
        fe = ue @ self.Ke                              # Ke symmetric
        f = torch.zeros(self.ndof, dtype=torch.float32, device=self.dev)
        f.index_add_(0, self.dof.reshape(-1), fe.reshape(-1))
        return f.double()

    def solve(self, presc_mask, presc_val, x0=None, tol=1e-6, maxit=40000):
        """K x = 0 on free DOFs with x = presc_val on presc_mask (bool tensor); PCG with Jacobi."""
        free = ~presc_mask
        x = torch.zeros(self.ndof, dtype=torch.float64, device=self.dev) if x0 is None else x0.clone()
        x[presc_mask] = presc_val[presc_mask]
        r = -self.matvec(x)
        r[presc_mask] = 0
        Minv = torch.where(free, 1.0 / self.diag, torch.zeros_like(self.diag))
        z = Minv * r
        p = z.clone()
        rz = torch.dot(r, z)
        b0 = torch.linalg.norm(self.matvec(torch.where(presc_mask, x, torch.zeros_like(x)))[free])
        it = 0
        for it in range(maxit):
            Ap = self.matvec(p)
            Ap[presc_mask] = 0
            alpha = rz / torch.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            if it % 50 == 0:
                rn = torch.linalg.norm(r)
                if rn <= tol * b0:
                    break
            z = Minv * r
            rz_new = torch.dot(r, z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        return x, it + 1


def face_nodes(X, fixed):
    key = np.round(X[:, 1], 6) * 1e6 + np.round(X[:, 2], 6)
    order = np.lexsort((X[:, 0], key))
    ks = key[order]
    last = np.r_[ks[1:] != ks[:-1], True]
    top = order[last]
    return top[(X[top, 0] >= X_FACE - 4.0) & ~fixed[top]]


def gap0(X, indenter, z_c=Z_C, y_c=0.0):
    if indenter == "cyl":
        dz = np.clip(np.abs(X[:, 2] - z_c), 0, R_TOOL)
        surf = X_FACE + (R_TOOL - np.sqrt(R_TOOL ** 2 - dz ** 2))
    elif indenter == "plate":
        surf = np.full(len(X), X_FACE)
    elif indenter.startswith("sphere"):
        R = float(indenter[6:])
        r2 = (X[:, 1] - y_c) ** 2 + (X[:, 2] - z_c) ** 2
        surf = X_FACE + (R - np.sqrt(np.clip(R ** 2 - r2, 0, None)))
    return surf - X[:, 0]


def contact(tip, nodes, g0, delta, x0=None, active0=None, tol=1e-6, max_outer=30, verbose=False):
    """Displacement-controlled frictionless contact; returns (p, x, active, info)."""
    dev = tip.dev
    d = delta - g0                                   # required inward displacement (> 0 where the bodies overlap)
    active = (d > 0) if active0 is None else (active0 | (d > 0))
    xdof = torch.tensor(3 * nodes, device=dev)
    fixed = torch.tensor(tip.fixed_dof, device=dev)
    x = x0
    its_total = 0
    for outer in range(max_outer):
        mask = fixed.clone()
        val = torch.zeros(tip.ndof, dtype=torch.float64, device=dev)
        act_dofs = xdof[torch.tensor(active, device=dev)]
        mask[act_dofs] = True
        val[act_dofs] = -torch.tensor(d[active], dtype=torch.float64, device=dev)
        x, its = tip.solve(mask, val, x0=x, tol=tol)
        its_total += its
        R = tip.matvec(x)
        p = -R[xdof].cpu().numpy()                   # compressive contact force on each candidate node
        p[~active] = 0.0
        ux = x[xdof].cpu().numpy()
        neg = active & (p < 0)
        pen = (~active) & (-ux < d - 1e-9)            # the surface stayed outside the indenter: penetration
        if verbose:
            print(f"    outer {outer}: CG {its}, active {active.sum()}, F {p.sum():.4f} N, release {neg.sum()}, "
                  f"add {pen.sum()}", flush=True)
        if not neg.any() and not pen.any():
            break
        active = (active & ~neg) | pen
    return p, x, active, dict(cg_iterations=its_total, outer=outer + 1)


def metrics(X, nodes, p, g0, delta):
    Xn = X[nodes]
    F = p.sum()
    c = (p[:, None] * Xn[:, 1:]).sum(0) / F
    r = np.linalg.norm(Xn[:, 1:] - c, axis=1)
    m = p > 1e-9 * F
    return dict(F=float(F), cy=float(c[0]), cz=float(c[1]), r_rms=float(np.sqrt((p * r ** 2).sum() / F)),
                arm=float((p * r).sum() / F), half_y=float(np.abs(Xn[m, 1] - c[0]).max()),
                half_z=float(np.abs(Xn[m, 2] - c[1]).max()), n_contact=int(m.sum()),
                outside_overlap=float(p[(delta - g0) <= 0].sum() / F))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("voxels")
    ap.add_argument("--indenter", default="cyl")
    ap.add_argument("--delta", type=float, nargs="+", required=True)
    ap.add_argument("--E", type=float, default=30.0)
    ap.add_argument("--nu", type=float, default=0.45)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out")
    ap.add_argument("--save")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    D = np.load(a.voxels, allow_pickle=True)
    occ, origin, h = D["occ"], D["origin"], float(D["h"])
    recipe = json.loads(str(D["recipe"])) if "recipe" in D else {}
    t0 = time.time()
    tip = GpuTip(occ, origin, h, a.E, a.nu)
    nodes = face_nodes(tip.X, tip.fixed_dof[0::3])
    g0 = gap0(tip.X[nodes], a.indenter)
    print(f"{a.voxels}: {tip.n_el} el, {tip.ndof} dof, {len(nodes)} face nodes, {tip.n_fixed} fixed nodes, x range "
          f"{tip.X[:, 0].min():.3f}..{tip.X[:, 0].max():.3f}, setup {tip.t_setup:.1f}s", flush=True)
    x, active = None, None
    saved = {}
    for dl in a.delta:
        ts = time.time()
        torch.cuda.synchronize()
        p, x, active, info = contact(tip, nodes, g0, dl, x0=x, active0=active, tol=a.tol, verbose=True)
        torch.cuda.synchronize()
        m = metrics(tip.X, nodes, p, g0, dl)
        row = dict(voxels=a.voxels, tag=a.tag, recipe=recipe, shift=D["shift"].tolist() if "shift" in D else None,
                   indenter=a.indenter, delta_mm=dl, E_s=a.E, nu=a.nu, h_mm=h, n_el=tip.n_el, n_dof=tip.ndof,
                   t_solve_s=time.time() - ts, gpu_mem_GB=torch.cuda.max_memory_allocated() / 1e9,
                   script="scripts/tpu_tip_fem_gpu.py", **info, **m)
        print(f"  delta {dl * 1e3:.1f} um: F {m['F']:.4f} N, r_rms {m['r_rms']:.3f} mm, arm {m['arm']:.3f} mm, half y/z "
              f"{m['half_y']:.2f}/{m['half_z']:.2f}, CG {info['cg_iterations']} in {row['t_solve_s']:.1f}s", flush=True)
        saved[f"p_{dl:g}"] = p
        if a.out:
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
    if a.save:
        u = x.view(-1, 3).cpu().numpy()
        np.savez_compressed(a.save, X_face=tip.X[nodes], g0=g0, ux_face=u[nodes, 0], **saved)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
