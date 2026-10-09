#!/usr/bin/env python3
"""The printed SR2 TPU fingertip insert as a finite-element model, and its contact with the bed's indenters
(step 3 of the TPU tip job). Small strain, linear elastic, rigid mount, frictional contact solved on the
compliance of the palmar face.

Geometry: `fingertip (3).stl` (STL frame: palmar face x = +8.5 mm, back face x = -8.5 mm on the mount plate with two
sockets, width y in [-7.4, 7.4], length z in [0, 21.41] with z = 0 on the mount and the end at z = 21.41). The contact
point is the tip-body origin projected on the palmar face, z_c = 21.41 - 10.55 = 10.86 mm.

Mesh: a rectilinear grid of boxes, fine where the contact and the walls are and coarse elsewhere (piecewise uniform
zones per axis), on which every cell takes the material at its centre:
  wall   perimeters, cells within 2 x 0.45 mm of the part outline in the layer plane;
  skin   top and bottom solid layers, cells within 4 x 0.2 mm of an exposed surface along the build direction;
  core   the rest: homogenized gyroid (orthotropic tensor of the printed cell, tpu_tip_fem_cells.py, rotated from the
         print frame), the solid (100 % infill), or a resolved structure given as voxels on the same grid.
Build direction 'x' puts the layers across the load (palmar face printed as a top skin), 'z' along it (palmar face
printed as perimeter walls, part standing on its mount face).

Contact: candidate nodes on the palmar face; the normal compliance of those nodes (columns of K^-1 from one CHOLMOD
factor) gives a dense problem solved by an active set (frictionless, rigid indenter: the screwdriver cylinder R 12.5 mm
with its axis along y, or a flat plate). Tangential stick and slip use the tangential compliance of the nodes in
contact (Goodman: no normal-tangential coupling).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_tip_fem_core as C  # noqa: E402

TIP_STL = Path.home() / "Downloads/fingertip (3).stl"
X_FACE, X_BACK, Y_HALF, Z_END = 8.5, -8.5, 7.4, 21.41
Z_C = Z_END - 10.55
R_TOOL = 12.5
WALL, LAYER, N_SKIN = 2 * 0.45, 0.2, 4


# ------------------------------------------------------------------------------------------ grid

def zoned_axis(zones):
    """Node coordinates from zones [(start, end, spacing), ...] covering an interval; spacing adjusted per zone."""
    pts = [zones[0][0]]
    for a, b, h in zones:
        n = max(1, int(round((b - a) / h)))
        pts.extend(list(a + (b - a) * (np.arange(1, n + 1) / n)))
    return np.array(pts)


GRIDS = {
    # medium: about 0.4 M dof
    "medium": dict(
        x=[(-8.5, -7.6, 0.3), (-7.6, -1.0, 0.66), (-1.0, 5.5, 0.5), (5.5, 7.0, 0.25), (7.0, 8.5, 0.15)],
        y=[(-7.4, -6.5, 0.3), (-6.5, -3.5, 0.2), (-3.5, 3.5, 0.25), (3.5, 6.5, 0.2), (6.5, 7.4, 0.3)],
        z=[(0.0, 0.9, 0.3), (0.9, 6.5, 0.7), (6.5, 8.0, 0.3), (8.0, 13.7, 0.15), (13.7, 15.5, 0.3), (15.5, 18.0, 0.5),
           (18.0, 21.41, 0.25)]),
    # coarse with 0.05 mm along z within 1 mm of the contact line (the solid tip's 0.2 mm half-width)
    "coarse_zfine": dict(
        x=[(-8.5, -7.6, 0.45), (-7.6, 5.5, 0.8), (5.5, 7.0, 0.3), (7.0, 8.5, 0.2)],
        y=[(-7.4, -3.5, 0.3), (-3.5, 3.5, 0.35), (3.5, 7.4, 0.3)],
        z=[(0.0, 0.9, 0.45), (0.9, 7.0, 1.0), (7.0, 8.4, 0.35), (8.4, 9.86, 0.2), (9.86, 11.86, 0.05),
           (11.86, 13.3, 0.2), (13.3, 15.0, 0.35), (15.0, 18.0, 0.75), (18.0, 21.41, 0.35)]),
    "coarse": dict(
        x=[(-8.5, -7.6, 0.45), (-7.6, 5.5, 0.8), (5.5, 7.0, 0.3), (7.0, 8.5, 0.2)],
        y=[(-7.4, -3.5, 0.3), (-3.5, 3.5, 0.35), (3.5, 7.4, 0.3)],
        z=[(0.0, 0.9, 0.45), (0.9, 7.0, 1.0), (7.0, 8.4, 0.35), (8.4, 13.3, 0.2), (13.3, 15.0, 0.35), (15.0, 18.0, 0.75),
           (18.0, 21.41, 0.35)]),
}


class Grid:
    def __init__(self, xs, ys, zs):
        self.xs, self.ys, self.zs = xs, ys, zs
        self.nc = (len(xs) - 1, len(ys) - 1, len(zs) - 1)
        self.nn = (len(xs), len(ys), len(zs))
        self.cx = 0.5 * (xs[1:] + xs[:-1])
        self.cy = 0.5 * (ys[1:] + ys[:-1])
        self.cz = 0.5 * (zs[1:] + zs[:-1])
        self.hx, self.hy, self.hz = np.diff(xs), np.diff(ys), np.diff(zs)

    @classmethod
    def named(cls, name):
        g = GRIDS[name]
        return cls(zoned_axis(g["x"]), zoned_axis(g["y"]), zoned_axis(g["z"]))

    def node_id(self, i, j, k):
        return (i * self.nn[1] + j) * self.nn[2] + k


# ------------------------------------------------------------------------------------------ print classification

def classify(build="x", h_aux=0.05, stl=TIP_STL, cache=None):
    """Material class on a uniform auxiliary grid of the insert: 0 void, 1 wall, 2 skin, 3 core. Returns
    (cls, origin, h_aux)."""
    from scipy import ndimage
    from tpu_tip_fem_voxel import load_stl, voxelize
    if cache and Path(cache).exists():
        d = np.load(cache)
        return d["cls"], d["origin"], float(d["h"])
    V, F = load_stl(stl)
    occ, o = voxelize(V, F, h_aux)
    ax = {"x": 0, "y": 1, "z": 2}[build]
    # in-plane distance to the outside, layer by layer (planes normal to the build axis)
    dist = np.zeros(occ.shape, np.float32)
    for k in range(occ.shape[ax]):
        sl = [slice(None)] * 3
        sl[ax] = k
        s = occ[tuple(sl)]
        if s.any():
            padded = np.pad(s, 1)
            dist[tuple(sl)] = ndimage.distance_transform_edt(padded)[1:-1, 1:-1] * h_aux
    wall = occ & (dist <= WALL + 1e-9)
    # distance along the build axis to the outside, both ways (top and bottom skins)
    n = occ.shape[ax]
    occ_m = np.moveaxis(occ, ax, -1)
    up = np.full(occ_m.shape, np.inf, np.float32)
    down = np.full(occ_m.shape, np.inf, np.float32)
    last = np.full(occ_m.shape[:-1], -1, int)
    for k in range(n):
        last = np.where(~occ_m[..., k], k, last)
        down[..., k] = np.where(occ_m[..., k], (k - last) * h_aux, 0)
    nxt = np.full(occ_m.shape[:-1], n, int)
    for k in range(n - 1, -1, -1):
        nxt = np.where(~occ_m[..., k], k, nxt)
        up[..., k] = np.where(occ_m[..., k], (nxt - k) * h_aux, 0)
    skin_m = occ_m & ((up <= N_SKIN * LAYER + 1e-9) | (down <= N_SKIN * LAYER + 1e-9))
    skin = np.moveaxis(skin_m, -1, ax) & ~wall
    cls = np.zeros(occ.shape, np.uint8)
    cls[occ] = 3
    cls[skin] = 2
    cls[wall] = 1
    if cache:
        np.savez_compressed(cache, cls=cls, origin=o, h=h_aux)
    return cls, o, h_aux


def sample_cells(grid, cls, origin, h_aux):
    """Class at each cell centre of the rectilinear grid (nearest auxiliary voxel)."""
    def idx(c, k):
        return np.clip(np.floor((c - origin[k]) / h_aux).astype(int), 0, cls.shape[k] - 1)
    I, J, K = np.meshgrid(idx(grid.cx, 0), idx(grid.cy, 1), idx(grid.cz, 2), indexing="ij")
    out = cls[I, J, K]
    # outside the auxiliary box -> void
    for c, k, ax in ((grid.cx, 0, 0), (grid.cy, 1, 1), (grid.cz, 2, 2)):
        bad = (c < origin[k]) | (c >= origin[k] + cls.shape[k] * h_aux)
        if bad.any():
            sl = [slice(None)] * 3
            sl[ax] = bad
            out[tuple(sl)] = 0
    return out


# ------------------------------------------------------------------------------------------ material

def rotate_C(C6, build):
    """Stiffness of a cell computed with its build axis along z, expressed in the tip frame with the build axis along
    `build` (cyclic permutation keeps the handedness)."""
    if build == "z":
        return C6.copy()
    perm = {"x": (1, 2, 0), "y": (2, 0, 1)}[build]      # cell axis a -> tip axis perm[a]
    T = np.zeros((3, 3, 3, 3))
    pairs = C.VOIGT
    for I, (i, j) in enumerate(pairs):
        for J, (k, l) in enumerate(pairs):
            v = C6[I, J]
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, l), (l, k)):
                    T[a, b, c, d] = v
    R = np.zeros((3, 3))
    for a in range(3):
        R[perm[a], a] = 1.0
    Tr = np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, T)
    out = np.zeros((6, 6))
    for I, (i, j) in enumerate(pairs):
        for J, (k, l) in enumerate(pairs):
            out[I, J] = Tr[i, j, k, l]
    return out


# ------------------------------------------------------------------------------------------ model

class TipModel:
    """Assembled stiffness of the tip on a grid with per-cell material tensors (6x6, MPa); mount faces fixed."""

    def __init__(self, grid, mat_id, mats, fix_back=True, fix_top=True, verbose=True):
        t0 = time.time()
        self.g = grid
        solid = mat_id >= 0
        idx = np.argwhere(solid)
        ne = len(idx)
        # element types: (hx, hy, hz) zone indices and material
        hx, hy, hz = grid.hx[idx[:, 0]], grid.hy[idx[:, 1]], grid.hz[idx[:, 2]]
        keys = np.stack([np.round(hx, 9), np.round(hy, 9), np.round(hz, 9), mat_id[solid].astype(float)], 1)
        uniq, typ = np.unique(keys, axis=0, return_inverse=True)
        Ke_list = []
        for u in uniq:
            Ke, _ = C.hex_im(u[0], u[1], u[2], D=mats[int(u[3])])
            Ke_list.append(Ke)
        en = C.element_nodes(idx, grid.nn, periodic=False)
        used = np.unique(en)
        nn_all = int(np.prod(grid.nn))
        remap = -np.ones(nn_all, np.int64)
        remap[used] = np.arange(len(used))
        self.node_of = used                     # compact -> grid node id
        self.remap = remap
        en = remap[en]
        nn = len(used)
        K = C.assemble(en, Ke_list, (np.ones(ne), typ.ravel()), nn)
        # node coordinates
        gi = used // (grid.nn[1] * grid.nn[2])
        gj = (used // grid.nn[2]) % grid.nn[1]
        gk = used % grid.nn[2]
        self.X = np.stack([grid.xs[gi], grid.ys[gj], grid.zs[gk]], 1)
        fixed = np.zeros(nn, bool)
        if fix_back:
            fixed |= self.X[:, 0] <= X_BACK + 1e-9
        if fix_top:
            fixed |= self.X[:, 2] <= 1e-9
        self.fixed_nodes = fixed
        fd = np.repeat(fixed, 3)
        self.free = np.where(~fd)[0]
        self.dof_map = -np.ones(3 * nn, np.int64)
        self.dof_map[self.free] = np.arange(len(self.free))
        self.K = K[self.free][:, self.free].tocsc()
        self.n_el, self.nn, self.n_types = ne, nn, len(uniq)
        t1 = time.time()
        self.fac = C.factor(self.K)
        t2 = time.time()
        self.t_assemble, self.t_factor = t1 - t0, t2 - t1
        if verbose:
            print(f"  tip model: {ne} el ({len(uniq)} types), {len(self.free)} dof, assemble {t1 - t0:.1f}s, factor "
                  f"{t2 - t1:.1f}s, nnz(L) {self.fac.nnz / 1e6:.0f} M", flush=True)

    def face_nodes(self, zlim=None, ylim=None):
        """Nodes on the palmar surface: for each (y, z) grid column the solid node with the largest x, if that x is
        within 4 mm of the face (fillets included); returns compact node ids."""
        X = self.X
        key = np.round(X[:, 1], 6) * 1e6 + np.round(X[:, 2], 6)
        order = np.lexsort((X[:, 0], key))
        k_sorted = key[order]
        last = np.r_[k_sorted[1:] != k_sorted[:-1], True]
        top = order[last]
        top = top[(X[top, 0] >= X_FACE - 4.0) & ~self.fixed_nodes[top]]
        if zlim is not None:
            top = top[(X[top, 2] >= zlim[0]) & (X[top, 2] <= zlim[1])]
        if ylim is not None:
            top = top[(X[top, 1] >= ylim[0]) & (X[top, 1] <= ylim[1])]
        return top

    def compliance(self, nodes, comps=(0,), block=200):
        """Dense compliance among the given node DOF components: W[a, b] = displacement of DOF a under a unit force
        at DOF b (both along the axes in comps)."""
        dofs = np.array([3 * n + c for c in comps for n in nodes])
        fr = self.dof_map[dofs]
        if (fr < 0).any():
            raise ValueError("candidate contact DOF is fixed")
        m = len(fr)
        W = np.zeros((m, m))
        nfree = len(self.free)
        for s in range(0, m, block):
            cols = fr[s:s + block]
            B = np.zeros((nfree, len(cols)))
            B[cols, np.arange(len(cols))] = 1.0
            Xs = self.fac(B)
            W[:, s:s + block] = Xs[fr]
        return 0.5 * (W + W.T)

    def solve_full(self, node_forces):
        """Full displacement (nn, 3) under nodal forces {compact node id: (fx, fy, fz)} given as (ids, F)."""
        ids, F = node_forces
        f = np.zeros(len(self.free))
        for c in range(3):
            d = self.dof_map[3 * ids + c]
            ok = d >= 0
            np.add.at(f, d[ok], F[ok, c])
        u = np.zeros(3 * self.nn)
        u[self.free] = self.fac(f)
        return u.reshape(-1, 3)


# ------------------------------------------------------------------------------------------ contact

def tributary_areas(model, nodes):
    """Area of the face associated with each face node (half the spacing to the neighbours in y and z)."""
    g = model.g
    y, z = model.X[nodes, 1], model.X[nodes, 2]

    def half(c, axis):
        k = np.searchsorted(axis, c)
        k = np.clip(k, 0, len(axis) - 1)
        lo = np.where(k > 0, axis[np.maximum(k - 1, 0)], axis[k])
        hi = np.where(k < len(axis) - 1, axis[np.minimum(k + 1, len(axis) - 1)], axis[k])
        return 0.5 * (hi - lo)
    return half(y, g.ys) * half(z, g.zs)


def gap0(model, nodes, indenter, z_c=Z_C):
    """Initial normal gap (mm, along -x) between each face node and the indenter at zero approach, the indenter
    touching the face at x = X_FACE."""
    X = model.X[nodes]
    if indenter == "cyl":
        dz = np.clip(np.abs(X[:, 2] - z_c), 0, R_TOOL)
        surf = X_FACE + (R_TOOL - np.sqrt(R_TOOL ** 2 - dz ** 2))
    elif indenter == "plate":
        surf = np.full(len(X), X_FACE)
    elif indenter.startswith("sphere"):
        R = float(indenter[6:])
        r2 = (X[:, 1]) ** 2 + (X[:, 2] - z_c) ** 2
        surf = X_FACE + (R - np.sqrt(np.clip(R ** 2 - r2, 0, None)))
    else:
        raise ValueError(indenter)
    return surf - X[:, 0]


def normal_contact(W, g0, delta, p0=None, tol=1e-10, maxit=200):
    """Frictionless contact of a rigid indenter at approach delta: nodal compressive forces p >= 0 and inward
    displacements w = W p with w_i >= delta - g0_i, equality where p_i > 0; active set. Returns (p, w, iterations)."""
    d = delta - g0
    S = (d > 0) if p0 is None else ((p0 > 0) | (d > 0))
    for it in range(maxit):
        p = np.zeros(len(d))
        if S.any():
            Ss = np.where(S)[0]
            p[Ss] = np.linalg.solve(W[np.ix_(Ss, Ss)], d[Ss])
        neg = S & (p < -tol)
        w = W @ p
        pen = (~S) & (w < d - 1e-12)
        if not neg.any() and not pen.any():
            return p, w, it
        if neg.any():
            # drop the most negative ones (all negatives at once converges for these problems)
            S &= ~neg
        if pen.any():
            S |= pen
    raise RuntimeError("active set did not converge")


def force_controlled(W, g0, F_target, delta_hi=2.0):
    """Approach giving total force F_target (bisection on delta); returns (delta, p, w)."""
    lo, hi = 0.0, delta_hi
    p_prev = None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p, w, _ = normal_contact(W, g0, mid, p_prev)
        if p.sum() < F_target:
            lo = mid
        else:
            hi = mid
            p_prev = p
        if hi - lo < 1e-7 * max(hi, 1e-3):
            break
    p, w, _ = normal_contact(W, g0, 0.5 * (lo + hi), p_prev)
    return 0.5 * (lo + hi), p, w


def patch_metrics(model, nodes, p, g0, delta):
    """Contact-patch quantities for the solved pressure: force, centre, RMS radius, full-slip twist arm about the
    normal through the force centre, extents, and the load outside the undeformed overlap."""
    X = model.X[nodes]
    F = p.sum()
    c = (p[:, None] * X[:, 1:]).sum(0) / F
    r = np.linalg.norm(X[:, 1:] - c, axis=1)
    m = p > 1e-12 * F
    out = dict(F=float(F), cy=float(c[0]), cz=float(c[1]), r_rms=float(np.sqrt((p * r ** 2).sum() / F)),
               arm=float((p * r).sum() / F), half_y=float(np.abs(X[m, 1] - c[0]).max()),
               half_z=float(np.abs(X[m, 2] - c[1]).max()), n_contact=int(m.sum()),
               outside_overlap=float(p[(delta - g0) <= 0].sum() / F))
    return out


# ------------------------------------------------------------------------------------------ recipes and runs

AUX_CACHE = ROOT / "assets/fem/20261009-tpu_tip"


def cell_tensor(rho, h=0.1, kind="slicer", path=ROOT / "docs/experiments/20261009-tpu_tip_fem/cells.jsonl"):
    """Homogenized 6x6 tensor (per unit solid modulus, print frame) of a cell from the cells.jsonl rows; the finest
    voxel size available for that density is used."""
    best = None
    for line in open(path):
        r = json.loads(line)
        if r["kind"] == kind and r.get("rho_req") is not None and abs(r["rho_req"] - rho) < 1e-9:
            if best is None or r["h_mm"] < best["h_mm"]:
                best = r
    if best is None:
        raise KeyError(f"no {kind} cell at rho {rho}")
    return np.array(best["C"]), best


def materials(recipe):
    """(mats list of 6x6 tensors in MPa, class -> material index map) for a recipe."""
    E_s, nu = recipe.get("E_s", 30.0), recipe.get("nu", 0.45)
    D = C.iso_D(E_s, nu)
    core = recipe.get("core", "solid")
    if core == "solid":
        Dc = D
    elif core == "homog":
        C6, _ = cell_tensor(recipe["rho"], kind=recipe.get("cell", "slicer"))
        Dc = rotate_C(C6, recipe.get("build", "x")) * E_s
    else:
        raise ValueError(core)
    return [D, Dc], {1: 0, 2: 0, 3: 1}


def build(recipe, grid_name="medium", verbose=True):
    grid = Grid.named(grid_name)
    build_dir = recipe.get("build", "x")
    AUX_CACHE.mkdir(parents=True, exist_ok=True)
    cls, o, h = classify(build_dir, cache=AUX_CACHE / f"classes_build{build_dir}_h0.05.npz")
    cc = sample_cells(grid, cls, o, h)
    mats, cmap = materials(recipe)
    mat_id = -np.ones(cc.shape, int)
    for k, v in cmap.items():
        mat_id[cc == k] = v
    if recipe.get("core") == "solid":
        mat_id[cc > 0] = 0
    return TipModel(grid, mat_id, mats, verbose=verbose), cc


def run_indentation(model, indenter, forces=(0.5, 1.0, 3.0), zwin=3.5, extra=None):
    nodes = model.face_nodes(zlim=(Z_C - zwin, Z_C + zwin) if indenter != "plate" else None)
    t0 = time.time()
    W = model.compliance(nodes)
    t1 = time.time()
    g0 = gap0(model, nodes, indenter)
    rows = []
    for F in forces:
        d, p, w = force_controlled(W, g0, F)
        m = patch_metrics(model, nodes, p, g0, d)
        m.update(indenter=indenter, delta_mm=float(d), k_contact_N_per_mm=None, n_candidates=len(nodes),
                 t_compliance_s=t1 - t0)
        rows.append(m)
    # secant and tangent stiffness from a finer force sweep
    return rows, dict(nodes=nodes, W=W, g0=g0)


def hertz_sphere(F, R, E, nu):
    Es = E / (1 - nu ** 2)
    a = (3 * F * R / (4 * Es)) ** (1 / 3)
    return a, a * a / R


def validate(grid_name="medium", R=40.0, E=30.0, nu=0.45):
    """Solid tip under a sphere: force-controlled approach and contact radius against Hertz."""
    model, _ = build(dict(core="solid", E_s=E, nu=nu), grid_name)
    rows, aux = run_indentation(model, f"sphere{R:g}", forces=(0.5, 1.0, 3.0), zwin=3.5)
    for r in rows:
        a, d = hertz_sphere(r["F"], R, E, nu)
        # contact radius from the RMS radius of a Hertz pressure (r_rms = sqrt(2/5) a)
        print(f"  sphere R {R:g} mm, F {r['F']:.2f} N: approach {r['delta_mm'] * 1e3:.2f} um (Hertz {d * 1e3:.2f}), "
              f"RMS radius {r['r_rms']:.3f} mm (Hertz {math.sqrt(0.4) * a:.3f}), arm {r['arm']:.3f} mm "
              f"(Hertz 3 pi a/16 = {3 * math.pi * a / 16:.3f}), {r['n_contact']} nodes in contact")
    return rows


def tangential_cycle(Wt, p, mu=1.0, frac=0.5, n_leg=40):
    """Tool translation U (mm) along y against the pad's tangential force Q (N) for the cycle 0 -> Q* -> -Q* -> Q*,
    Q* = frac mu N, on the nodes in contact with normal forces p (fixed, Goodman). Wt: y-y compliance of those nodes
    (mm/N). Incremental stick and slip (Mindlin-Deresiewicz path dependence through the slip offsets)."""
    n = len(p)
    N = p.sum()
    Qs = frac * mu * N
    path = np.concatenate([np.linspace(0, Qs, n_leg + 1), np.linspace(Qs, -Qs, 2 * n_leg + 1)[1:],
                           np.linspace(-Qs, Qs, 2 * n_leg + 1)[1:]])
    off = np.zeros(n)                       # slip offsets: stick means u_i = U - off_i
    q = np.zeros(n)
    U_prev = 0.0
    out = []
    lim = mu * p
    for Q in path:
        slip = np.zeros(n, bool)
        sgn = np.zeros(n)
        for _ in range(200):
            S = np.where(~slip)[0]
            L = np.where(slip)[0]
            qL = lim[L] * sgn[L]
            m = len(S)
            A = np.zeros((m + 1, m + 1))
            A[:m, :m] = Wt[np.ix_(S, S)]
            A[:m, m] = -1.0
            A[m, :m] = 1.0
            rhs = np.concatenate([-off[S] - Wt[np.ix_(S, L)] @ qL, [Q - qL.sum()]])
            sol = np.linalg.solve(A, rhs)
            qq = np.zeros(n)
            qq[S], qq[L] = sol[:m], qL
            U = sol[m]
            u = Wt @ qq
            over = (~slip) & (np.abs(qq) > lim * (1 + 1e-9))
            # a slipping node whose slip increment opposes its friction direction sticks again
            inc = (U - u) - off
            back = slip & (inc * sgn < -1e-12)
            if not over.any() and not back.any():
                break
            if over.any():
                slip[over] = True
                sgn[over] = np.sign(qq[over])
            if back.any() and not over.any():
                slip[back] = False
        off = np.where(slip, U - u, off)
        q = qq
        out.append((float(Q), float(U), float(slip.mean())))
        U_prev = U
    arr = np.array(out)
    Qp, Up = arr[:, 0], arr[:, 1]
    k1 = n_leg
    u_peak = Up[k1]
    # first return to Q = 0 on the unloading leg
    leg2 = slice(k1, k1 + 2 * n_leg + 1)
    i0 = k1 + n_leg
    loop = float(np.trapezoid(Qp[k1:], Up[k1:]))
    return dict(Q_star=float(Qs), u_presliding_mm=float(u_peak), u_after_unload_mm=float(Up[i0]),
                u_recovered_mm=float(u_peak - Up[i0]), loop_area_mJ=abs(loop), k_t0_N_per_mm=float(Qp[1] / Up[1]),
                slip_fraction_at_peak=float(arr[k1, 2]), path=arr.tolist())


def face_profile(model, u, line="z"):
    """Normal displacement of the palmar surface along the line y = 0 (line 'z') or z = z_c (line 'y'), from the
    full solution u (nn, 3); returns (coordinate, u_x)."""
    nodes = model.face_nodes()
    X = model.X[nodes]
    if line == "z":
        sel = np.abs(X[:, 1]) < 1e-6 + 0.5 * np.min(np.diff(model.g.ys))
        c = X[sel, 2]
    else:
        zz = model.g.zs[np.argmin(np.abs(model.g.zs - Z_C))]
        sel = np.abs(X[:, 2] - zz) < 1e-9
        c = X[sel, 1]
    o = np.argsort(c)
    return c[o], -u[nodes[sel][o], 0]


def law_force(nodes_A, g0, delta, E, h_f=8.5):
    """Winkler pressure law of the pads (p = E d / h_f, h_f the foundation depth of fingertip_geometry) on the face
    nodes: total force at approach delta (E in MPa, lengths in mm -> N)."""
    return float((E / h_f * nodes_A * np.clip(delta - g0, 0, None)).sum())


def run_recipe(name, recipe, grid_name="coarse", forces=(0.25, 0.5, 1.0, 2.0, 3.0, 4.0), out=None, data_dir=None,
               cycle_N=(0.5, 1.0, 3.0)):
    t0 = time.time()
    model, cc = build(recipe, grid_name)
    rows = []
    base = dict(recipe=name, **{k: v for k, v in recipe.items()}, grid=grid_name, n_dof=len(model.free),
                n_el=model.n_el, t_assemble_s=model.t_assemble, t_factor_s=model.t_factor, nnz_L=model.fac.nnz,
                script="scripts/tpu_tip_fem_tip.py")
    saved = {}
    for ind in ("cyl", "plate", "sphere2"):
        zwin = 3.5 if ind == "cyl" else (2.5 if ind == "sphere2" else None)
        ylim = (-2.5, 2.5) if ind == "sphere2" else None
        nodes = model.face_nodes(zlim=(Z_C - zwin, Z_C + zwin) if zwin else None, ylim=ylim)
        tc = time.time()
        W = model.compliance(nodes)
        tc = time.time() - tc
        g0 = gap0(model, nodes, ind)
        A = tributary_areas(model, nodes)
        fl = forces if ind != "sphere2" else (0.5, 1.0, 3.0)
        for F in fl:
            d, p, w = force_controlled(W, g0, F)
            m = patch_metrics(model, nodes, p, g0, d)
            # pressure-law modulus that reproduces this approach at this force
            E_fit = F / max(law_force(A, g0, d, 1.0), 1e-30)
            row = dict(base, indenter=ind, F_target=F, delta_mm=float(d), E_law_fit_MPa=float(E_fit),
                       t_compliance_s=tc, n_candidates=len(nodes), **m)
            if abs(F - 1.0) < 1e-9 or abs(F - 3.0) < 1e-9 or abs(F - 0.5) < 1e-9:
                saved[f"{ind}_{F:g}"] = dict(p=p, X=model.X[nodes], A=A, g0=g0, delta=d)
            if ind == "cyl" and F in cycle_N:
                # tangential cycle at half the slip force on the nodes in contact
                on = np.where(p > 1e-9 * F)[0]
                Wt = model.compliance(nodes[on], comps=(1,))
                cyc = tangential_cycle(Wt, p[on])
                row.update({k: v for k, v in cyc.items() if k != "path"})
                saved[f"cycle_{F:g}"] = dict(path=np.array(cyc["path"]))
            if abs(F - 1.0) < 1e-9 and ind in ("cyl", "sphere2"):
                act = np.where(p > 0)[0]
                Fn = np.zeros((len(act), 3))
                Fn[:, 0] = -p[act]
                u = model.solve_full((nodes[act], Fn))
                for line in ("z", "y"):
                    cz, uz = face_profile(model, u, line)
                    saved[f"profile_{ind}_{line}"] = dict(c=cz, u=uz)
            rows.append(row)
            if out:
                row["t_total_s"] = time.time() - t0
                with open(out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            print(f"  {name} {ind} F {F:g} N: approach {d * 1e3:.1f} um, r_rms {m['r_rms']:.3f} mm, arm {m['arm']:.3f} mm, "
                  f"half y/z {m['half_y']:.2f}/{m['half_z']:.2f} mm, E_law {E_fit:.2f} MPa"
                  + (f", presliding {row['u_presliding_mm'] * 1e3:.2f} um, loop {row['loop_area_mJ'] * 1e3:.3f} uJ"
                     if 'u_presliding_mm' in row else ""), flush=True)
    if data_dir:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        flat = {}
        for k, v in saved.items():
            for kk, vv in v.items():
                flat[f"{k}__{kk}"] = np.asarray(vv)
        np.savez_compressed(Path(data_dir) / f"{name}.npz", **flat)
    return rows


RECIPES = {
    "solid95": dict(core="solid", E_s=30.0, nu=0.45, build="x"),
    "g20_across": dict(core="homog", rho=0.2, E_s=30.0, nu=0.45, build="x"),
    "g20_along": dict(core="homog", rho=0.2, E_s=30.0, nu=0.45, build="z"),
    "g10_across": dict(core="homog", rho=0.1, E_s=30.0, nu=0.45, build="x"),
    "g10_along": dict(core="homog", rho=0.1, E_s=30.0, nu=0.45, build="z"),
}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["validate", "size", "run"])
    ap.add_argument("--grid", default="coarse")
    ap.add_argument("--recipes", nargs="*", default=list(RECIPES))
    ap.add_argument("--out", default=str(ROOT / "docs/experiments/20261009-tpu_tip_fem/tip_runs.jsonl"))
    ap.add_argument("--data", default=str(ROOT / "docs/experiments/20261009-tpu_tip_fem/data"))
    a = ap.parse_args()
    if a.cmd == "validate":
        validate(a.grid)
    elif a.cmd == "size":
        g = Grid.named(a.grid)
        print("cells", g.nc, int(np.prod(g.nc)), "nodes", int(np.prod(g.nn)))
    else:
        for name in a.recipes:
            run_recipe(name, RECIPES[name], a.grid, out=a.out, data_dir=a.data)


# ------------------------------------------------------------------------------------------ resolved print voxels

def resolved_voxels(recipe, h=0.1, shift=(0.0, 0.0), out=None):
    """Uniform voxels (side h) of the printed insert: walls and skins solid, the core filled with the slicer's gyroid
    beads layer by layer in the print frame (build axis recipe['build']), the gyroid phase shifted in-plane by `shift`
    (mm, along the two in-plane tip axes in increasing order). Returns (solid bool array, origin)."""
    import tpu_tip_fem_cells as CE
    build_dir = recipe.get("build", "x")
    cls, o, ha = classify(build_dir, cache=AUX_CACHE / f"classes_build{build_dir}_h0.05.npz")
    step = int(round(h / ha))
    c = cls[step // 2::step, step // 2::step, step // 2::step]
    origin = np.asarray(o, float).copy()                     # coarse voxel k covers auxiliary voxels step k .. step k + step - 1
    solid = (c == 1) | (c == 2)
    core = c == 3
    rho = recipe.get("rho")
    if recipe.get("core") == "solid" or rho is None or rho >= 0.999:
        return solid | core, origin
    ax = {"x": 0, "y": 1, "z": 2}[build_dir]
    inplane = [a for a in range(3) if a != ax]
    L, _ = CE.slicer_period(rho)
    n = int(round(L / h))
    per = int(round(CE.LAYER / h))
    shp = c.shape
    # coordinates along the build axis of each voxel layer, measured from the part's first layer
    nb = shp[ax]
    beads = np.zeros(shp, bool)
    cache = {}
    for kb in range(nb):
        layer = kb // per
        zc = (layer + 0.5) * CE.LAYER
        key = round(zc % L, 6)
        if key not in cache:
            cache[key] = CE.slicer_layer_mask(L, zc, h)          # periodic n x n
        m = cache[key]
        # tile to the in-plane extent with the phase shift
        s0 = int(round(shift[0] / h)) % n
        s1 = int(round(shift[1] / h)) % n
        mm = np.roll(np.roll(m, s0, 0), s1, 1)
        reps = (shp[inplane[0]] // n + 2, shp[inplane[1]] // n + 2)
        tiled = np.tile(mm, reps)[:shp[inplane[0]], :shp[inplane[1]]]
        sl = [slice(None)] * 3
        sl[ax] = kb
        beads[tuple(sl)] = tiled
    occ = solid | (core & beads)
    occ = C.largest_component(occ)
    if out:
        np.savez_compressed(out, occ=occ, origin=origin, h=h, recipe=json.dumps(recipe), shift=np.array(shift))
    return occ, origin
