#!/usr/bin/env python3
"""Large-deformation check of the voxel tip (step 18 of the TPU tip job): compressible Neo-Hookean, the energy of
PolyFEM's NeoHookean,

    Psi = mu/2 (tr C - 3) - mu ln J + lambda/2 (ln J)^2,   P = mu (F - F^-T) + lambda ln J F^-T,

on plain hex8 voxels (2 x 2 x 2 Gauss points), total Lagrangian, solved by Newton with the element tangent matrices
stored on the GPU and Jacobi-preconditioned conjugate gradients. The same code with the linear material
(P = lambda tr(eps) I + 2 mu eps) gives the plain-hex8 linear reference, so the ratio of the two forces is the effect of
large deformation alone.

    PY=.venv/bin/python
    $PY scripts/tpu_tip_fem_gpu_nl.py block --h 0.25 --nu 0.45 --delta 0.05 0.5 1.0     # frictionless plate on a block
    $PY scripts/tpu_tip_fem_gpu_nl.py tip assets/fem/20261009-tpu_tip/voxels/g05x_s0.npz --indenter cyl --delta 0.1 0.2 0.29

Tip: back face and mount face fixed, frictionless rigid indenter under displacement control with the active set of
tpu_tip_fem_gpu.py (the indenter surface is taken at the undeformed y, z of each face node). Rows go to --out.
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
import tpu_tip_fem_gpu as G  # noqa: E402


def gauss_dN(h):
    """Shape-function gradients (8 Gauss points, 8 nodes, 3) of a cube of side h, node a at CORNERS[a] * h."""
    g = 1 / math.sqrt(3)
    xi_a = 2 * C.CORNERS - 1
    out = []
    for gp in [(a, b, c) for a in (-g, g) for b in (-g, g) for c in (-g, g)]:
        gp = np.array(gp)
        dN = np.zeros((8, 3))
        for k in range(3):
            o = [l for l in range(3) if l != k]
            dN[:, k] = xi_a[:, k] / 8 * (1 + xi_a[:, o[0]] * gp[o[0]]) * (1 + xi_a[:, o[1]] * gp[o[1]])
        out.append(dN * (2 / h))
    return np.array(out)


class NLVoxels:
    def __init__(self, occ, origin, h, E=30.0, nu=0.45, dev="cuda", chunk=40_000):
        t0 = time.time()
        self.dev, self.chunk = dev, chunk
        idx = np.argwhere(occ)
        sn = tuple(s + 1 for s in occ.shape)
        en = C.element_nodes(idx, sn, periodic=False)
        used, inv = np.unique(en, return_inverse=True)
        en = inv.reshape(en.shape)
        self.nn, self.n_el = len(used), len(en)
        gi, gj, gk = used // (sn[1] * sn[2]), (used // sn[2]) % sn[1], used % sn[2]
        self.X = np.stack([origin[0] + gi * h, origin[1] + gj * h, origin[2] + gk * h], 1)
        self.ndof = 3 * self.nn
        dof = (3 * en[:, :, None] + np.arange(3)[None, None, :]).reshape(len(en), 24)
        self.dof = torch.tensor(dof, dtype=torch.int64, device=dev)
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))
        self.w = h ** 3 / 8
        self.dN = torch.tensor(gauss_dN(h), dtype=torch.float64, device=dev)          # (8 g, 8 a, 3)
        self.Ke = torch.empty((self.n_el, 24, 24), dtype=torch.float32, device=dev)
        self.diag = torch.zeros(self.ndof, dtype=torch.float64, device=dev)
        self.t_setup = time.time() - t0

    def _stress_tangent(self, Hg, linear):
        """First Piola stress P (m, 8, 3, 3) and tangent dP/dF (m, 8, 3, 3, 3, 3) at the Gauss points."""
        I = torch.eye(3, dtype=Hg.dtype, device=Hg.device)
        lam, mu = self.lam, self.mu
        if linear:
            eps = 0.5 * (Hg + Hg.transpose(-1, -2))
            tr = eps.diagonal(dim1=-2, dim2=-1).sum(-1)
            P = lam * tr[..., None, None] * I + 2 * mu * eps
            A = (lam * torch.einsum("ij,kl->ijkl", I, I) + mu * (torch.einsum("ik,jl->ijkl", I, I)
                                                                  + torch.einsum("il,jk->ijkl", I, I)))
            return P, A.expand(*Hg.shape[:2], 3, 3, 3, 3), None
        F = I + Hg
        J = torch.linalg.det(F)
        FmT = torch.linalg.inv(F).transpose(-1, -2)
        lnJ = torch.log(torch.clamp(J, min=1e-12))
        P = mu * (F - FmT) + lam * lnJ[..., None, None] * FmT
        A = (mu * torch.einsum("ik,jl->ijkl", I, I).expand(*Hg.shape[:2], 3, 3, 3, 3)
             + (mu - lam * lnJ)[..., None, None, None, None] * torch.einsum("...il,...kj->...ijkl", FmT, FmT)
             + lam * torch.einsum("...ij,...kl->...ijkl", FmT, FmT))
        return P, A, J

    def assemble(self, x, linear=False):
        """Element tangents into self.Ke, their diagonal into self.diag; returns the internal force f_int(x) and the
        smallest Jacobian determinant at the Gauss points."""
        f = torch.zeros(self.ndof, dtype=torch.float64, device=self.dev)
        self.diag.zero_()
        Jmin = float("inf")
        dN = self.dN
        for s in range(0, self.n_el, self.chunk):
            dof = self.dof[s:s + self.chunk]
            U = x[dof].view(-1, 8, 3)                                    # (m, a, i)
            Hg = torch.einsum("mai,gaj->mgij", U, dN)                   # grad u at the Gauss points
            P, A, J = self._stress_tangent(Hg, linear)
            if J is not None:
                Jmin = min(Jmin, float(J.min()))
            fe = self.w * torch.einsum("mgij,gaj->mai", P, dN).reshape(-1, 24)
            # K_(ai)(bk) = w sum_g dN_ga,j A_ijkl dN_gb,l (in float32: the tangent only steers Newton)
            dN32 = dN.float()
            T = torch.einsum("gaj,mgijkl->mgaikl", dN32, A.float())
            Ke = self.w * torch.einsum("mgaikl,gbl->maibk", T, dN32).reshape(-1, 24, 24)
            del T
            self.Ke[s:s + self.chunk] = Ke
            f.index_add_(0, dof.reshape(-1), fe.reshape(-1))
            self.diag.index_add_(0, dof.reshape(-1), torch.diagonal(Ke, dim1=1, dim2=2).reshape(-1).double())
            del U, Hg, P, A, fe, Ke
        return f, Jmin

    def matvec(self, v, chunk=500_000):
        out = torch.zeros(self.ndof, dtype=torch.float32, device=self.dev)
        v32 = v.float()
        for s in range(0, self.n_el, chunk):
            dof = self.dof[s:s + chunk]
            fe = torch.bmm(self.Ke[s:s + chunk], v32[dof].unsqueeze(-1)).squeeze(-1)
            out.index_add_(0, dof.reshape(-1), fe.reshape(-1))
        return out.double()

    def cg(self, rhs, free, tol, maxit=60000):
        """K_t dx = rhs on the free DOFs (dx = 0 elsewhere), Jacobi-preconditioned CG."""
        Minv = torch.where(free, 1.0 / torch.clamp(self.diag, min=1e-30), torch.zeros_like(self.diag))
        x = torch.zeros_like(rhs)
        r = torch.where(free, rhs, torch.zeros_like(rhs))
        b0 = torch.linalg.norm(r)
        z = Minv * r
        p = z.clone()
        rz = torch.dot(r, z)
        it = 0
        for it in range(maxit):
            Ap = self.matvec(p)
            Ap[~free] = 0
            alpha = rz / torch.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            if it % 25 == 0 and torch.linalg.norm(r) <= tol * b0:
                break
            z = Minv * r
            rz_new = torch.dot(r, z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        return x, it + 1

    def _increment(self, presc, val, x0, linear, tol, cg_tol, max_newton, verbose):
        """One load increment to x = val on the prescribed DOFs: a predictor through the tangent at x0, then Newton
        with a backtracking step that keeps every Gauss point's J > 0. Returns (x, f, info, converged)."""
        free = ~presc
        f0, _ = self.assemble(x0, linear)
        dp = torch.where(presc, val - x0, torch.zeros_like(x0))
        rhs = -torch.where(free, f0 + self.matvec(dp), torch.zeros_like(f0))
        dfree, cg_total = self.cg(rhs, free, cg_tol)
        x = x0 + dp + dfree
        f, Jmin = self.assemble(x, linear)
        hist = []
        it = 0
        for it in range(max_newton):
            if not (Jmin > 0) or not torch.isfinite(f).all():
                return x, f, dict(newton=it, cg=cg_total, residual=float("nan"), J_min=Jmin), False
            R = torch.where(free, f, torch.zeros_like(f))
            rn = float(torch.linalg.norm(R))
            scale = float(torch.linalg.norm(f[presc])) + 1e-30
            hist.append(rn / scale)
            if verbose:
                print(f"      newton {it}: |R|/|f_presc| {rn / scale:.2e}, J_min {Jmin:.4f}", flush=True)
            if rn <= tol * scale:
                return x, f, dict(newton=it + 1, cg=cg_total, residual=hist[-1], J_min=Jmin), True
            dx, its = self.cg(-R, free, cg_tol)
            cg_total += its
            step = 1.0
            for _ in range(10):
                xt = x + step * dx
                ft, Jt = self.assemble(xt, linear)
                Rt = float(torch.linalg.norm(torch.where(free, ft, torch.zeros_like(ft))))
                if Jt > 0 and math.isfinite(Rt) and (Rt < rn or linear):
                    break
                step *= 0.5
            x, f, Jmin = xt, ft, Jt
        return x, f, dict(newton=it + 1, cg=cg_total, residual=hist[-1] if hist else float("nan"), J_min=Jmin), False

    def newton(self, presc, val, x0=None, linear=False, tol=1e-6, cg_tol=1e-5, max_newton=15, min_frac=1 / 128,
               verbose=True):
        """Equilibrium with x = val on the prescribed DOFs, reached in load increments that halve when an increment
        fails to converge (inverted elements or no convergence). Returns (x, f_int, info)."""
        x = torch.zeros(self.ndof, dtype=torch.float64, device=self.dev) if x0 is None else x0.clone()
        start = torch.where(presc, x, torch.zeros_like(x))
        target = torch.where(presc, val, torch.zeros_like(val))
        done, frac = 0.0, 1.0
        tot = dict(newton=0, cg=0, increments=0)
        f, info = None, {}
        while done < 1.0 - 1e-12:
            s = min(1.0, done + frac)
            vs = start + s * (target - start)
            xt, ft, info, ok = self._increment(presc, vs, x, linear, tol, cg_tol, max_newton, verbose)
            tot["newton"] += info["newton"]
            tot["cg"] += info["cg"]
            if ok:
                x, f, done = xt, ft, s
                tot["increments"] += 1
                frac = min(2 * frac, 1.0 - done) if done < 1 else frac
            else:
                frac *= 0.5
                if verbose:
                    print(f"      increment to {s:.3f} failed; step {frac:.4f}", flush=True)
                if frac < min_frac:
                    raise RuntimeError("load increment below the minimum fraction")
        return x, f, dict(tot, residual=info["residual"], J_min=info["J_min"])


def tip_contact(m, nodes, g0, delta, fixed_dof, x0=None, active0=None, linear=False, max_outer=30, verbose=True, **kw):
    """Displacement-controlled frictionless contact as in tpu_tip_fem_gpu.contact, each active set solved by Newton."""
    dev = m.dev
    d = delta - g0
    active = (d > 0) if active0 is None else (active0 | (d > 0))
    xdof = torch.tensor(3 * nodes, device=dev)
    fixed = torch.tensor(fixed_dof, device=dev)
    x, info = x0, {}
    newton_total, cg_total = 0, 0
    for outer in range(max_outer):
        mask = fixed.clone()
        val = torch.zeros(m.ndof, dtype=torch.float64, device=dev)
        act = xdof[torch.tensor(active, device=dev)]
        mask[act] = True
        val[act] = -torch.tensor(d[active], dtype=torch.float64, device=dev)
        x, f, info = m.newton(mask, val, x0=x, linear=linear, verbose=verbose, **kw)
        newton_total += info["newton"]
        cg_total += info["cg"]
        p = -f[xdof].cpu().numpy()
        p[~active] = 0.0
        ux = x[xdof].cpu().numpy()
        neg = active & (p < 0)
        pen = (~active) & (-ux < d - 1e-9)
        if verbose:
            print(f"    outer {outer}: active {active.sum()}, F {p.sum():.4f} N, release {neg.sum()}, add {pen.sum()}",
                  flush=True)
        if not neg.any() and not pen.any():
            break
        active = (active & ~neg) | pen
    return p, x, active, dict(outer=outer + 1, newton=newton_total, cg=cg_total, J_min=info.get("J_min"))


def write_row(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", choices=["block", "tip"])
    ap.add_argument("voxels", nargs="?")
    ap.add_argument("--h", type=float, default=0.25, help="block: voxel side, mm")
    ap.add_argument("--E", type=float, default=None, help="MPa; block 1, tip 30")
    ap.add_argument("--nu", type=float, default=0.45)
    ap.add_argument("--indenter", default="cyl")
    ap.add_argument("--delta", type=float, nargs="+", required=True)
    ap.add_argument("--materials", nargs="+", default=["linear", "neohookean"])
    ap.add_argument("--surface", action="store_true")
    ap.add_argument("--warm", action="store_true", help="tip: start each material from the previous one's solution")
    ap.add_argument("--cg-tol", type=float, default=1e-5)
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="Newton: |R| / |f_presc| at convergence; the float32 tangent sets a floor near 3e-6 on the eFlesh lattice")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(ROOT / "docs/experiments/20261009-tpu_tip_fem/gpu_nl_runs.jsonl"))
    a = ap.parse_args()
    if a.case == "block":
        E = a.E or 1.0
        n = (int(round(10 / a.h)), int(round(10 / a.h)), int(round(5 / a.h)))
        m = NLVoxels(np.ones(n, bool), np.zeros(3), a.h, E, a.nu)
        X = m.X
        bot, top = X[:, 2] < 1e-9, X[:, 2] > 5 - 1e-9
        presc = np.zeros(m.ndof, bool)
        presc[0::3] |= bot
        presc[1::3] |= bot
        presc[2::3] |= bot | top
        presc_t = torch.tensor(presc, device=m.dev)
        top_z = torch.tensor(np.where(top)[0] * 3 + 2, device=m.dev)
        print(f"block {n}: {m.n_el} el, {m.ndof} dof, setup {m.t_setup:.1f}s", flush=True)
        for mat in a.materials:
            x = None
            for dl in a.delta:
                ts = time.time()
                val = torch.zeros(m.ndof, dtype=torch.float64, device=m.dev)
                val[top_z] = -dl
                x, f, info = m.newton(presc_t, val, x0=x, linear=(mat == "linear"), tol=a.tol)
                F = float(-f[top_z].sum())
                row = dict(case="block", tag=a.tag, material=mat, h_mm=a.h, E=E, nu=a.nu, delta_mm=dl, F=F,
                           F_EAdH=E * 100 * dl / 5, n_el=m.n_el, t_s=time.time() - ts, newton_tol=a.tol,
                           script="scripts/tpu_tip_fem_gpu_nl.py", **info)
                print(f"  {mat} delta {dl}: F {F:.5f} N (E A d/H {E * 100 * dl / 5:.4f}), {info}", flush=True)
                write_row(a.out, row)
        return
    E = a.E or 30.0
    D = np.load(a.voxels, allow_pickle=True)
    occ, origin, h = D["occ"], D["origin"], float(D["h"])
    m = NLVoxels(occ, origin, h, E, a.nu)
    fixed = (m.X[:, 0] <= m.X[:, 0].min() + 1e-6) | (m.X[:, 2] <= m.X[:, 2].min() + 1e-6)
    fixed_dof = np.repeat(fixed, 3)
    nodes = G.face_nodes(m.X, fixed)
    Xg = m.X[nodes].copy()
    if a.surface:
        Xg[:, 0], _ = G.true_surface_x(Xg)
    g0 = G.gap0(Xg, a.indenter)
    print(f"{a.voxels}: {m.n_el} el, {m.ndof} dof, {len(nodes)} face nodes, setup {m.t_setup:.1f}s, "
          f"GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    prev = {}
    for mat in a.materials:
        x, active = None, None
        for dl in a.delta:
            ts = time.time()
            if a.warm and dl in prev:
                x, active = prev[dl]
            p, x, active, info = tip_contact(m, nodes, g0, dl, fixed_dof, x0=x, active0=active, linear=(mat == "linear"),
                                             cg_tol=a.cg_tol, tol=a.tol)
            prev[dl] = (x.clone(), active.copy())
            met = G.metrics(m.X, nodes, p, g0, dl)
            row = dict(case="tip", voxels=a.voxels, tag=a.tag, material=mat, indenter=a.indenter, delta_mm=dl, E_s=E, nu=a.nu,
                       h_mm=h, n_el=m.n_el, surface=a.surface, t_s=time.time() - ts, newton_tol=a.tol, cg_tol=a.cg_tol,
                       gpu_mem_GB=torch.cuda.max_memory_allocated() / 1e9, script="scripts/tpu_tip_fem_gpu_nl.py", **info, **met)
            print(f"  {mat} delta {dl * 1e3:.0f} um: F {met['F']:.4f} N, arm {met['arm']:.3f} mm, {info}, "
                  f"{row['t_s']:.0f}s", flush=True)
            write_row(a.out, row)


if __name__ == "__main__":
    main()
