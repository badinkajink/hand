#!/usr/bin/env python3
"""Voxel finite elements for printed TPU: periodic homogenization of an infill cell and the stiffness of a
voxelized fingertip, small strain, linear elastic, isotropic solid.

Element. Every voxel is a box of sides (hx, hy, hz) carrying an eight-node hexahedron with Wilson-Taylor
incompatible modes (nine internal bubble modes condensed out per element), which represents bending of a wall
two to four voxels thick without the shear locking of the plain trilinear brick. All voxels of one size share
one element matrix, scaled by the voxel's Young's modulus.

Homogenization. A cell periodic in x, y and z with solid voxels `occ`: for each of the six unit macroscopic
strains eps_i (Voigt order xx, yy, zz, yz, xz, xy, engineering shears) the periodic fluctuation u_i solves
K u_i = -f_i with f_i = sum_e B_e^T D eps_i V_e, and C_ij = (V_s D_ij + f_i . u_j) / |Y|.

    python3 scripts/tpu_tip_fem_core.py selftest
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp

VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
CORNERS = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)])   # local node a -> (i, j, k)


def iso_D(E=1.0, nu=0.45):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[3:, 3:] = np.eye(3) * mu
    return D


def _B_rows(dN):
    """Strain-displacement rows for gradients dN (n_nodes x 3) -> (6, 3 n)."""
    n = dN.shape[0]
    B = np.zeros((6, 3 * n))
    for a in range(n):
        dx, dy, dz = dN[a]
        c = 3 * a
        B[0, c] = dx
        B[1, c + 1] = dy
        B[2, c + 2] = dz
        B[3, c + 1], B[3, c + 2] = dz, dy
        B[4, c], B[4, c + 2] = dz, dx
        B[5, c], B[5, c + 1] = dy, dx
    return B


def hex_im(hx, hy, hz, nu=0.45, D=None, incompatible=True):
    """Element stiffness (24 x 24) for E = 1 (or the given D), and the integral of B over the element (6 x 24).

    Node a of the element sits at CORNERS[a] * (hx, hy, hz)."""
    if D is None:
        D = iso_D(1.0, nu)
    g = 1 / math.sqrt(3)
    h = np.array([hx, hy, hz])
    xi_a = 2 * CORNERS - 1                                     # (8, 3) in {-1, 1}
    Kuu = np.zeros((24, 24))
    Kua = np.zeros((24, 9))
    Kaa = np.zeros((9, 9))
    Bint = np.zeros((6, 24))
    w = np.prod(h) / 8                                         # det J for the 8 Gauss points of weight 1
    for gp in [(a, b, c) for a in (-g, g) for b in (-g, g) for c in (-g, g)]:
        gp = np.array(gp)
        # dN/dxi_k = xi_a_k/8 * prod_{l != k} (1 + xi_a_l gp_l)
        dN = np.zeros((8, 3))
        for k in range(3):
            o = [l for l in range(3) if l != k]
            dN[:, k] = xi_a[:, k] / 8 * (1 + xi_a[:, o[0]] * gp[o[0]]) * (1 + xi_a[:, o[1]] * gp[o[1]])
        dN = dN * (2 / h)                                      # d/dx = (2/h) d/dxi
        B = _B_rows(dN)
        # bubble modes P_m = 1 - xi_m^2: dP_m/dx_m = -2 xi_m (2/h_m), times each displacement component
        dP = np.zeros((3, 3))
        for m in range(3):
            dP[m, m] = -2 * gp[m] * 2 / h[m]
        G = _B_rows(dP)                                        # (6, 9): mode m, component c at column 3 m + c
        Kuu += B.T @ D @ B * w
        Kua += B.T @ D @ G * w
        Kaa += G.T @ D @ G * w
        Bint += B * w
    if incompatible:
        K = Kuu - Kua @ np.linalg.solve(Kaa, Kua.T)
    else:
        K = Kuu
    return 0.5 * (K + K.T), Bint


# ------------------------------------------------------------------------------------------ assembly

def element_nodes(idx, shape_nodes, periodic):
    """Global node ids (n_e, 8) of voxels idx (n_e, 3) on a node grid of shape_nodes (periodic wraps)."""
    nxn, nyn, nzn = shape_nodes
    out = np.empty((len(idx), 8), np.int64)
    for a, (i, j, k) in enumerate(CORNERS):
        I, J, K = idx[:, 0] + i, idx[:, 1] + j, idx[:, 2] + k
        if periodic:
            I, J, K = I % nxn, J % nyn, K % nzn
        out[:, a] = (I * nyn + J) * nzn + K
    return out


def assemble(enodes, Ke_list, scale, n_nodes):
    """Sparse stiffness (3 n x 3 n) from per-element node lists, element matrices Ke_list[t] for the element type
    index t of each element (scale is (n_e,) moduli and an (n_e,) type index), assembled one local node pair at a
    time to bound memory."""
    E_e, typ = scale
    rows_all, cols_all, vals_all = [], [], []
    K = None
    for a in range(8):
        for b in range(8):
            blk = np.stack([Ke[3 * a:3 * a + 3, 3 * b:3 * b + 3] for Ke in Ke_list])     # (n_types, 3, 3)
            v = E_e[:, None, None] * blk[typ]                                           # (n_e, 3, 3)
            r = (3 * enodes[:, a])[:, None, None] + np.arange(3)[None, :, None]
            c = (3 * enodes[:, b])[:, None, None] + np.arange(3)[None, None, :]
            r, c = np.broadcast_to(r, v.shape), np.broadcast_to(c, v.shape)
            M = sp.coo_matrix((v.ravel(), (r.ravel(), c.ravel())), shape=(3 * n_nodes, 3 * n_nodes)).tocsr()
            K = M if K is None else K + M
    return K


def largest_component(occ, periodic=False):
    """Keep the largest face-connected (6-connected) component of the solid voxels."""
    from scipy import ndimage
    if periodic:
        # label on a 2x tiled grid would be exact; approximate by merging labels across the wrapped faces
        lab, n = ndimage.label(occ)
        if n <= 1:
            return occ
        parent = np.arange(n + 1)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for ax in range(3):
            a = np.take(lab, 0, axis=ax)
            b = np.take(lab, -1, axis=ax)
            m = (a > 0) & (b > 0)
            for p, q in zip(a[m], b[m]):
                rp, rq = find(p), find(q)
                if rp != rq:
                    parent[rp] = rq
        roots = np.array([find(i) for i in range(n + 1)])
        lab = roots[lab]
        lab[~occ] = 0
    else:
        lab, n = ndimage.label(occ)
        if n <= 1:
            return occ
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == np.argmax(counts)


class _Factor:
    """CHOLMOD factor (scikit-sparse 0.5) callable as solve(b)."""

    def __init__(self, K):
        from sksparse.cholmod import cho_factor
        self.f = cho_factor(K.tocsc())

    def __call__(self, b):
        return self.f.solve(b)

    @property
    def nnz(self):
        return int(self.f.nnz)


def factor(K):
    """Sparse Cholesky (CHOLMOD) of a symmetric positive definite matrix; call the result to solve."""
    return _Factor(K)


# ------------------------------------------------------------------------------------------ homogenization

def homogenize(occ, h, E=1.0, nu=0.45, Emap=None, incompatible=True, verbose=True):
    """Effective 6x6 elasticity tensor of a periodic voxel cell. occ: bool (nx, ny, nz); h: voxel side (scalar
    or 3-vector); Emap: optional per-voxel modulus array. Returns dict(C, rho, n_dof, times)."""
    t0 = time.time()
    occ = largest_component(occ, periodic=True)
    h = np.broadcast_to(np.asarray(h, float), (3,))
    nx, ny, nz = occ.shape
    idx = np.argwhere(occ)
    ne = len(idx)
    D1 = iso_D(1.0, nu)
    Ke, Bint = hex_im(*h, nu=nu, D=D1, incompatible=incompatible)
    E_e = np.full(ne, E) if Emap is None else Emap[occ]
    en = element_nodes(idx, (nx, ny, nz), periodic=True)
    used = np.unique(en)
    remap = -np.ones(nx * ny * nz, np.int64)
    remap[used] = np.arange(len(used))
    en = remap[en]
    nn = len(used)
    K = assemble(en, [Ke], (E_e, np.zeros(ne, int)), nn)
    t1 = time.time()
    # load vectors: f_i = sum_e E_e Bint^T D1 eps_i (assembled), 6 columns
    F = np.zeros((3 * nn, 6))
    BD = Bint.T @ D1                                            # (24, 6)
    for a in range(8):
        for c in range(3):
            np.add.at(F, 3 * en[:, a] + c, E_e[:, None] * BD[3 * a + c][None, :])
    # remove the rigid translation: fix node 0
    keep = np.ones(3 * nn, bool)
    keep[:3] = False
    Kr = K[keep][:, keep]
    fac = factor(Kr)
    t2 = time.time()
    U = np.zeros((3 * nn, 6))
    U[keep] = fac(-F[keep])
    t3 = time.time()
    Vcell = nx * ny * nz * np.prod(h)
    Vs = E_e.sum() * np.prod(h) / max(E, 1e-30) if Emap is None else None
    # C_ij = (sum_e V_e E_e D1_ij + f_i . u_j) / |Y|
    C = (np.prod(h) * E_e.sum() * D1 + F.T @ U) / Vcell
    C = 0.5 * (C + C.T)
    out = dict(C=C, rho=ne / (nx * ny * nz), n_dof=int(keep.sum()), n_el=ne,
               t_assemble=t1 - t0, t_factor=t2 - t1, t_solve=t3 - t2, nnz_K=int(K.nnz))
    out["nnz_L"] = fac.nnz
    if verbose:
        print(f"  homogenize: {ne} el, {out['n_dof']} dof, rho {out['rho']:.4f}, assemble {t1 - t0:.1f}s factor "
              f"{t2 - t1:.1f}s solve {t3 - t2:.1f}s")
    return out


def engineering(C):
    """Young's moduli, shear moduli and Poisson ratios of a 6x6 stiffness (Voigt, engineering shear)."""
    S = np.linalg.inv(C)
    E = 1 / np.diag(S)[:3]
    G = 1 / np.diag(S)[3:]
    nu = np.array([-S[0, 1] * E[0], -S[0, 2] * E[0], -S[1, 2] * E[1]])   # nu_xy, nu_xz, nu_yz
    return dict(Ex=E[0], Ey=E[1], Ez=E[2], Gyz=G[0], Gxz=G[1], Gxy=G[2], nu_xy=nu[0], nu_xz=nu[1], nu_yz=nu[2])


def directional_E(C, d):
    S = np.linalg.inv(C)
    d = np.asarray(d, float) / np.linalg.norm(d)
    # strain-stress for uniaxial stress along d
    s = np.array([d[0] ** 2, d[1] ** 2, d[2] ** 2, 2 * d[1] * d[2], 2 * d[0] * d[2], 2 * d[0] * d[1]])
    sig = np.array([d[0] ** 2, d[1] ** 2, d[2] ** 2, d[1] * d[2], d[0] * d[2], d[0] * d[1]])
    return 1.0 / (s @ S @ sig)


# ------------------------------------------------------------------------------------------ self-test

def selftest():
    nu = 0.45
    print("1. solid cell: C must equal D")
    occ = np.ones((4, 4, 4), bool)
    r = homogenize(occ, 0.1, 1.0, nu, verbose=False)
    print("   max |C - D| =", float(np.abs(r["C"] - iso_D(1, nu)).max()))
    print("2. laminate (layers normal to z, E 1 and 0.1): Ez against the exact series value")
    occ = np.ones((2, 2, 8), bool)
    Em = np.ones(occ.shape)
    Em[:, :, 4:] = 0.1
    r = homogenize(occ, 0.1, 1.0, nu, Emap=Em, verbose=False)
    C = r["C"]
    # exact C33 for a laminate: harmonic mean of the constrained moduli M = E(1-nu)/((1+nu)(1-2nu))
    M = lambda E: E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))  # noqa: E731
    exact = 1 / (0.5 / M(1) + 0.5 / M(0.1))
    print(f"   C33 {C[2, 2]:.6f} exact {exact:.6f}; C11 {C[0, 0]:.6f}")
    print("3. cantilever of 40 x 4 x 4 voxels, tip load: deflection against Euler-Bernoulli plus shear")
    for incompatible in (True, False):
        n = (40, 4, 4)
        occ = np.ones(n, bool)
        h = 0.25
        idx = np.argwhere(occ)
        Ke, _ = hex_im(h, h, h, nu=nu, incompatible=incompatible)
        sn = (n[0] + 1, n[1] + 1, n[2] + 1)
        en = element_nodes(idx, sn, periodic=False)
        nn = np.prod(sn)
        K = assemble(en, [Ke], (np.ones(len(idx)), np.zeros(len(idx), int)), nn)
        node_i = np.arange(nn) // (sn[1] * sn[2])
        fixed = np.zeros(3 * nn, bool)
        for c in range(3):
            fixed[3 * np.where(node_i == 0)[0] + c] = True
        f = np.zeros(3 * nn)
        tip = np.where(node_i == n[0])[0]
        P = 1e-3
        f[3 * tip + 2] = -P / len(tip)
        free = ~fixed
        u = np.zeros(3 * nn)
        u[free] = factor(K[free][:, free])(f[free])
        L, b = n[0] * h, n[1] * h
        I = b ** 4 / 12
        Gs = 1 / (2 * (1 + nu))
        w_eb = P * L ** 3 / (3 * 1.0 * I) + P * L / (5 / 6 * Gs * b * b)
        print(f"   incompatible={incompatible}: tip deflection {-u[3 * tip + 2].mean():.5f} beam theory {w_eb:.5f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
