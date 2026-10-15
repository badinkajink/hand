#!/usr/bin/env python3
"""Parametric convex fingertips of the co-design study (docs/notes/20261014-tip_codesign_plan.md, step 4): shape,
hull meshes, surface samples for the contact laws, sphere pads, and the as-printed surface. numpy and scipy only;
trimesh is imported inside the STL and comparison functions.

Frame: the tip body of fingertip_geometry (palmar normal +x for palmar sign +1, y along the PIP axis, z toward the PIP,
the finger running along -z). The envelope is the CAD insert's (assets/cad_v1/fingertip.stl): 17.0 mm deep (x),
14.8 mm wide (y), 21.41 mm long (z); palmar face at x = 10.55 mm (the reach), back face at x = -6.45 mm, end face at
z = -10.55 mm, mount face at z = 10.86 mm. The bench grasp's contact point is the tip origin's (y, z) = (0, 0).

Keys:
  tcd_rs<a>_re<b>      flat palmar face; the palmar long edges rounded by r_side = a mm, the palmar end edge and the
                       side-end edges by r_end = b mm (the side-end edges by at most the 7.4 mm half-width); back and
                       mount edges square. With a = b the palmar corners are
                       spherical (the as-built insert is tcd_rs3_re3). With a != b the body is the box rounded by
                       min(a, b) with the larger fillet cut into it, which leaves a crease at each palmar corner.
  ..._cr<R>            the same intersected with a ball of radius R mm whose surface passes through the apex
                       (10.55, 0, 0) mm: a spherical crown (R 10.55 is the real_v1 sphere tip's surface)
  ..._p<h>             as printed palmar face up at layer height h mm: the convex hull of the per-layer outlines,
                       each slice taken at mid-layer and placed at the layer's top (PrusaSlicer's slicing)
  tpu<r>               fingertip_geometry's block (every edge r); callers route these keys to that module

Curved tips (CurvedTip, the 2026-10-15 curved-tip study), each the envelope box intersected with one convex solid, so the
back face (on the DIP link's back plate) and the mount face (z = 10.86 mm, on the DIP link) stay flat where the solid
reaches them:
  tcd_cap<r>           capsule: a cylinder of radius r mm along the finger, its axis r below the palmar reach, capped by
                       a hemisphere at the distal end; behind the axis the sides run straight back to the back face, so
                       the section across the finger is a D of width 2 r (r > 7.4 mm: an arc of radius r across the
                       14.8 mm width, cut by the side faces)
  tcd_ell<ay>x<ax>x<az>[x<azp>]
                       half-ellipsoid: the palmar half of an ellipsoid with semi-axes ay across the width, ax deep and az
                       along the finger, apex at the palmar reach over the contact point (y, z) = (0, 0); distal of the
                       apex the semi-axis is az, proximal of it azp (default az); behind its equator the section runs
                       straight back to the back face
  tcd_sph<R>           the whole sphere of radius R mm through the apex, no envelope: tcd_sph10.55 is the real_v1 sphere
                       tip, the limit against which the curved-tip machinery is tested
  ..._u<h>             as printed upright (mount face on the bed, layers of h mm across the finger): the convex hull of
                       the per-layer outlines, each slice taken at mid-layer and placed at the layer's top and bottom
  ..._p<h>             as printed palmar face up, as for the block family

The shape is the intersection of convex primitives (inside()). Surface points are exact (radial bisection from the
box centre); the hull meshes are convex hulls of those points; pads are 0.75 mm spheres on a 1 mm voxel grid of the
surface, each with its own area share, normal, inset centre (clamped into the body shrunk by rs) and K = E a / h, with
h = 8.5 mm (fixed, as fingertip_geometry) or n . (p - c_V) (local, Drake's field to first order).

    .venv/bin/python scripts/tip_codesign_geometry.py tcd_rs3_re3 tcd_rs3_re3_p0.2 --info
"""
from __future__ import annotations

import argparse
import math
import os
import re
from functools import cached_property
from pathlib import Path

import numpy as np

REACH, END = 0.01055, 0.01055
DEPTH, WIDTH, LENGTH = 0.0170, 0.0148, 0.02141
FOUNDATION = DEPTH / 2
X_FACE, X_BACK = REACH, REACH - DEPTH
Z_END, Z_MOUNT = -END, -END + LENGTH
C0 = np.array([X_FACE - DEPTH / 2, 0.0, Z_END + LENGTH / 2])     # box centre, the bisection origin
KEY_RE = re.compile(r"^tcd_rs(?P<a>[0-9.]+)_re(?P<b>[0-9.]+)(?:_cr(?P<R>[0-9.]+))?(?:_p(?P<h>[0-9.]+))?$")


def _fib(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    th = math.pi * (3 - math.sqrt(5)) * i
    return np.column_stack([r * np.cos(th), r * np.sin(th), z])


DEGENERATE = 1e-14       # m^2, |(b - a) x (c - a)| below which a hull facet is degenerate (see TipShape._planes)


def _hull(P):
    from scipy.spatial import ConvexHull
    h = ConvexHull(P)
    V = h.points[h.vertices]
    remap = np.full(len(P), -1)
    remap[h.vertices] = np.arange(len(h.vertices))
    F = remap[h.simplices]
    cen = V.mean(0)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), a - cen) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return V, F


def _subsample(V, F, a_target=1e-8):
    """Centroids, normals and areas of a uniform subdivision of every triangle into n^2 parts, n = ceil(sqrt(A /
    a_target)): flat faces of a convex hull are a few large triangles, which this spreads into ~a_target pieces."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    A = 0.5 * np.linalg.norm(cr, axis=1)
    N = cr / np.maximum(2 * A[:, None], 1e-30)
    nsub = np.maximum(1, np.ceil(np.sqrt(A / a_target))).astype(int)
    P_, N_, A_ = [], [], []
    for n in np.unique(nsub):
        k = nsub == n
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        up = ii + jj <= n - 1
        dn = ii + jj <= n - 2
        U = np.concatenate([(ii[up] + 1 / 3) / n, (ii[dn] + 2 / 3) / n])
        W = np.concatenate([(jj[up] + 1 / 3) / n, (jj[dn] + 2 / 3) / n])
        ak, bk, ck = a[k], b[k], c[k]
        Q = ak[:, None, :] + U[None, :, None] * (bk - ak)[:, None, :] + W[None, :, None] * (ck - ak)[:, None, :]
        P_.append(Q.reshape(-1, 3))
        N_.append(np.repeat(N[k], len(U), axis=0))
        A_.append(np.repeat(A[k] / (n * n), len(U)))
    return np.concatenate(P_), np.concatenate(N_), np.concatenate(A_)


def _voxel_pick(P, w, cell):
    """One representative per occupied voxel: the w-weighted mean of its points. Returns (indices of the groups,
    representative points)."""
    q = np.floor(P / cell).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    W = np.bincount(inv, w, n)
    M = np.column_stack([np.bincount(inv, w * P[:, k], n) for k in range(3)]) / W[:, None]
    return inv, M


class TipShape:
    def __init__(self, key: str):
        m = KEY_RE.match(key)
        if not m:
            raise ValueError(f"not a tcd key: {key!r}")
        self.key = key
        self.a = float(m["a"]) * 1e-3
        self.b = float(m["b"]) * 1e-3
        self.R = float(m["R"]) * 1e-3 if m["R"] else None
        self.layer = float(m["h"]) * 1e-3 if m["h"] else None
        if self.a > WIDTH / 2 + 1e-12 or self.b > min(DEPTH, LENGTH / 2):
            raise ValueError(f"{key}: fillet larger than the envelope")
        self.reach = X_FACE
        self.top_z = Z_MOUNT
        self.design = TipShape(key.rsplit("_p", 1)[0]) if self.layer else None

    # ------------------------------------------------------------------ the body
    def inside_design(self, P, tol=0.0):
        """Points (n, 3) inside the design surface (before printing)."""
        x, y, z = P[:, 0], np.abs(P[:, 1]), P[:, 2]
        a, b = self.a, self.b
        r0 = min(a, b)
        ok = (x >= X_BACK - tol) & (z <= Z_MOUNT + tol) & (x <= X_FACE + tol) & (y <= WIDTH / 2 + tol) & (z >= Z_END - tol)
        dx = np.maximum(x - (X_FACE - r0), 0)
        dy = np.maximum(y - (WIDTH / 2 - r0), 0)
        dz = np.maximum((Z_END + r0) - z, 0)
        ok &= np.sqrt(dx * dx + dy * dy + dz * dz) <= r0 + tol
        if a > r0:
            dx1, dy1 = np.maximum(x - (X_FACE - a), 0), np.maximum(y - (WIDTH / 2 - a), 0)
            ok &= np.hypot(dx1, dy1) <= a + tol
        if b > r0:
            dx2, dz2 = np.maximum(x - (X_FACE - b), 0), np.maximum((Z_END + b) - z, 0)
            ok &= np.hypot(dx2, dz2) <= b + tol
            b3 = min(b, WIDTH / 2)                        # the side-end edges: at most the half-width
            if b3 > r0:
                dy3, dz3 = np.maximum(y - (WIDTH / 2 - b3), 0), np.maximum((Z_END + b3) - z, 0)
                ok &= np.hypot(dy3, dz3) <= b3 + tol
        if self.R:
            c = np.array([X_FACE - self.R, 0.0, 0.0])
            ok &= np.linalg.norm(P - c, axis=1) <= self.R + tol
        return ok

    def inside(self, P, tol=0.0):
        if not self.layer:
            return self.inside_design(P, tol)
        n, d = self._hull_planes
        out = np.empty(len(P), bool)
        ch = max(64, int(4e6 // len(n)))                  # points x facets per chunk <= 4e6 (32 MB a temporary)
        for i in range(0, len(P), ch):
            out[i:i + ch] = np.all(P[i:i + ch] @ n.T - d[None, :] <= tol, axis=1)
        return out

    @cached_property
    def _hull_planes(self):
        V, F = self.hull_fine
        return self._planes(V, F)

    @staticmethod
    def _planes(V, F):
        """Outward unit normals and offsets of the hull's facets. qhull's triangulation of the terraces leaves
        degenerate facets (|cross| about 1e-19 m^2, collinear outline points) whose normalised normals point anywhere;
        they are dropped (|cross| < 1e-14 m^2; real facets have 5e-12 or more), which leaves the hull's region as is."""
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1)
        k = ln > DEGENERATE
        n, a = n[k] / ln[k, None], a[k]
        return n, np.einsum("ij,ij->i", n, a)

    def _bisect(self, D, inside, t_max=0.03, it=42):
        lo, hi = np.zeros(len(D)), np.full(len(D), t_max)
        for _ in range(it):
            mid = 0.5 * (lo + hi)
            ins = inside(C0 + mid[:, None] * D)
            lo, hi = np.where(ins, mid, lo), np.where(ins, hi, mid)
        return C0 + lo[:, None] * D

    # ------------------------------------------------------------------ printed surface
    def _outline(self, x, n_ang=1440):
        """The design shape's cross-section outline at height x (points in y-z), by radial bisection."""
        th = (np.arange(n_ang) + 0.5) / n_ang * 2 * math.pi
        D2 = np.column_stack([np.zeros(n_ang), np.cos(th), np.sin(th)])
        o = np.array([x, 0.0, C0[2]])
        lo, hi = np.zeros(n_ang), np.full(n_ang, 0.03)
        if not self.design.inside_design(o[None])[0]:
            o = np.array([x, 0.0, 0.0])
            if not self.design.inside_design(o[None])[0]:
                return np.zeros((0, 3))
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            ins = self.design.inside_design(o + mid[:, None] * D2)
            lo, hi = np.where(ins, mid, lo), np.where(ins, hi, mid)
        return o + lo[:, None] * D2

    @cached_property
    def hull_fine(self):
        """(V, F) of a fine hull: the design surface from 300k rays, or for an as-printed key the hull of the layers'
        top outlines (and their bottom outlines, which only the lowest layer contributes)."""
        if not self.layer:
            P = self._bisect(_fib(300_000), self.inside_design)
            return _hull(P)
        h = self.layer
        nl = int(round(DEPTH / h))
        pts = []
        for k in range(nl):
            xm = X_BACK + (k + 0.5) * h
            O = self._outline(xm)
            if not len(O):
                continue
            for xs in (X_BACK + (k + 1) * h, X_BACK + k * h):
                Q = O.copy()
                Q[:, 0] = xs
                pts.append(Q)
        return _hull(np.concatenate(pts))

    # ------------------------------------------------------------------ outputs
    @cached_property
    def dense(self):
        """Exact surface points with normals and area shares: the fine hull's triangles cut into pieces of about
        0.01 mm2, each centroid pushed onto the surface along the ray from the box centre."""
        V, F = self.hull_fine
        M, N, A = _subsample(V, F, 1e-8)
        if not self.layer:
            D = M - C0
            D /= np.linalg.norm(D, axis=1, keepdims=True)
            M = self._bisect(D, self.inside_design)
        keep = A > 0
        return M[keep], N[keep], A[keep]

    def surface(self, s=None):
        """Points, normals and areas of the palmar region (n_x > 0.05, no mount-side region), for the law."""
        P, N, A = self.dense
        k = (N[:, 0] > 0.05) & (N[:, 2] < 0.5)
        return P[k], N[k], A[k]

    def _terrace_corners(self, eps=None):
        """The fine hull's vertices (the layers' outline corners of an as-printed key); with eps (m), each layer's
        outline simplified by Douglas-Peucker so no kept point lies within eps of the polyline through its neighbours:
        collinear runs of outline points are what give qhull, and Drake's own hull of a mesh, degenerate facets."""
        V0, _ = self.hull_fine
        if not eps:
            return V0
        out = []
        for x in np.unique(np.round(V0[:, 0], 9)):
            Q = V0[np.abs(V0[:, 0] - x) < 5e-10]
            if len(Q) < 4:
                out.append(Q)
                continue
            c = Q.mean(0)
            Q = Q[np.argsort(np.arctan2(Q[:, 2] - c[2], Q[:, 1] - c[1]))]
            i0 = int(np.argmax(np.linalg.norm(Q - Q[0], axis=1)))
            keep = np.zeros(len(Q), bool)
            keep[[0, i0]] = True
            stack = [(0, i0), (i0, len(Q))]                     # index len(Q) closes the loop at point 0
            while stack:
                a, b = stack.pop()
                if b - a < 2:
                    continue
                A, B = Q[a, 1:], Q[b % len(Q), 1:]
                P = Q[a + 1:b, 1:]
                t = B - A
                dist = np.abs(t[0] * (P[:, 1] - A[1]) - t[1] * (P[:, 0] - A[0])) / max(np.linalg.norm(t), 1e-15)
                j = int(np.argmax(dist))
                if dist[j] > eps:
                    keep[a + 1 + j] = True
                    stack += [(a, a + 1 + j), (a + 1 + j, b)]
            out.append(Q[keep])
        return np.concatenate(out)

    def hull(self, res=0.0005, sign=1.0, eps=None):
        """Convex hull of the surface points thinned to one per res voxel (the 0.5 mm resolution of
        fingertip_geometry.hull_mesh); x mirrored for sign -1."""
        P, _, A = self.dense
        _, M = _voxel_pick(P, A, res)
        if self.layer:                                   # keep the terraces' corners
            M = np.concatenate([M, self._terrace_corners(eps)])
        else:
            D = M - C0
            D /= np.linalg.norm(D, axis=1, keepdims=True)
            M = self._bisect(D, self.inside_design)
        V, F = _hull(M * np.array([sign, 1.0, 1.0]))
        if sign < 0:
            F = F[:, [0, 2, 1]]
            V2, F2 = _hull(V)
            return V2, F2
        return V, F

    @cached_property
    def volume_centroid(self):
        V, F = self.hull_fine
        a, b, c = V[F[:, 0]] - C0, V[F[:, 1]] - C0, V[F[:, 2]] - C0
        vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
        cen = (a + b + c) / 4.0
        return C0 + (vol[:, None] * cen).sum(0) / vol.sum(), float(vol.sum())

    def centroid(self):
        return self.volume_centroid[0].copy()

    def pads(self, E=1e7, s=0.001, rs=0.00075, sign=1.0, h="fixed", K="each", full=False):
        """Pad sphere centres, normals, areas and stiffness (sign -1 mirrors x). With full=False returns (C, K) for the
        law module; K='median' gives every pad the median K, as reorient_backends sets one d0 per finger."""
        P, N, A = self.dense
        m = (N[:, 0] > -0.5) & (N[:, 2] < 0.5)               # fingertip_geometry.pad_mask: no back or mount pads
        P, N, A = P[m], N[m], A[m]
        inv, _ = _voxel_pick(P, A, s)
        n = inv.max() + 1
        Aw = np.bincount(inv, A, n)
        Pm = np.column_stack([np.bincount(inv, A * P[:, k], n) for k in range(3)]) / Aw[:, None]
        Nm = np.column_stack([np.bincount(inv, A * N[:, k], n) for k in range(3)])
        Nm /= np.linalg.norm(Nm, axis=1, keepdims=True)
        # drop slivers (voxels that clip the surface): their points join the nearest kept pad
        keep = Aw >= 0.3 * s * s
        from scipy.spatial import cKDTree
        idx = np.flatnonzero(keep)
        _, near = cKDTree(Pm[idx]).query(P)
        Aw = np.bincount(near, A, len(idx))
        Pm = np.column_stack([np.bincount(near, A * P[:, k], len(idx)) for k in range(3)]) / Aw[:, None]
        Nm = np.column_stack([np.bincount(near, A * N[:, k], len(idx)) for k in range(3)])
        Nm /= np.linalg.norm(Nm, axis=1, keepdims=True)
        if not self.layer:                                   # the area-weighted mean lies inside: back onto the surface
            D = Pm - C0
            D /= np.linalg.norm(D, axis=1, keepdims=True)
            Pm = self._bisect(D, self.inside_design)
        C = self._clamp(Pm - rs * Nm, rs)
        cV = self.centroid()
        hh = np.full(len(C), FOUNDATION) if h == "fixed" else np.maximum(np.einsum("ij,ij->i", Nm, Pm - cV), 1e-3)
        Ks = E / hh * Aw
        if K == "median":
            Ks = np.full(len(Ks), float(np.median(Ks)))
        flip = np.array([sign, 1.0, 1.0])
        if full:
            return C * flip, Nm * flip, Aw, Ks
        return C * flip, Ks

    def _clamp(self, C, rs, it=12):
        """Push centres into the body shrunk by rs (half-spaces of the 0.5 mm hull), so no sphere stands proud of an
        adjacent face next to an edge sharper than rs."""
        V, F = self.hull(0.0005)
        n, d = self._planes(V, F)
        C = C.copy()
        for _ in range(it):
            viol = C @ n.T - (d - rs)[None, :]
            k = np.argmax(viol, axis=1)
            v = viol[np.arange(len(C)), k]
            bad = v > 1e-9
            if not bad.any():
                break
            C[bad] -= v[bad, None] * n[k[bad]]
        return C

    def mesh_paths(self, out_dir, eps=None, res=0.0005):
        """One OBJ per palmar sign, named by the key, written through a temporary file; with eps (m) the terrace
        outlines are simplified first (_terrace_corners), with res (m) the surface points are thinned to that voxel
        in place of 0.5 mm; the files are then named <key>[_dp<eps in um>][_r<res in mm>]_<sign>.obj."""
        out = {}
        tag = (f"_dp{eps * 1e6:g}" if eps else "") + (f"_r{res * 1e3:g}" if res != 0.0005 else "")
        for name, sign in (("pos", 1.0), ("neg", -1.0)):
            V, F = self.hull(res, sign, eps=eps)
            p = Path(out_dir) / f"{self.key}{tag}_{name}.obj"
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
                with open(tmp, "w") as fh:
                    fh.write(f"# {self.key} (scripts/tip_codesign_geometry.py)\n")
                    fh.writelines(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n" for v in V)
                    fh.writelines(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n" for f in F)
                os.replace(tmp, p)
            out[sign] = p
        return out

    # ------------------------------------------------------------------ FE frame and STL
    def fe_inside(self, x, y, z):
        """tpu_tip_design's insert frame (palmar +x at 8.5 mm, mount face z = 0, end face z = 21.41 mm), in mm."""
        xb, yb, zb = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
        xf, yf, zf = xb.ravel(), yb.ravel(), zb.ravel()
        out = np.empty(xf.size, bool)
        for i in range(0, xf.size, 1 << 21):              # 2M points at a time: the FE grids reach 43M points
            sl = slice(i, i + (1 << 21))
            P = np.column_stack([xf[sl] * 1e-3 + (REACH - 0.0085), yf[sl] * 1e-3, Z_MOUNT - zf[sl] * 1e-3])
            out[sl] = self.inside(P)
        return out.reshape(xb.shape)

    def fe_surface_x(self, y, z):
        """Height of the palmar surface (mm, insert frame) above each (y, z) in mm, by bisection along +x from the
        back face; nan outside the footprint."""
        yy, zz = np.broadcast_arrays(np.asarray(y, float), np.asarray(z, float))
        base = np.column_stack([np.full(yy.size, X_BACK + 1e-5), yy.ravel() * 1e-3, Z_MOUNT - zz.ravel() * 1e-3])
        ok = self.inside(base)
        lo, hi = np.zeros(yy.size), np.full(yy.size, DEPTH)
        for _ in range(44):
            mid = 0.5 * (lo + hi)
            Q = base.copy()
            Q[:, 0] += mid
            ins = self.inside(Q)
            lo, hi = np.where(ins, mid, lo), np.where(ins, hi, mid)
        xs = (base[:, 0] + lo - (REACH - 0.0085)) * 1e3
        xs[~ok] = np.nan
        return xs.reshape(yy.shape)

    def stl_insert_frame(self, path, res=0.00025):
        """The design (or printed) body as an STL in the insert frame (mm)."""
        import trimesh
        V, F = self.hull(res)
        W = np.column_stack([(V[:, 0] - (REACH - 0.0085)) * 1e3, V[:, 1] * 1e3, (Z_MOUNT - V[:, 2]) * 1e3])
        m = trimesh.Trimesh(W, F[:, [0, 2, 1]], process=True)
        trimesh.repair.fix_normals(m)
        m.export(path)
        return m


CURVED_RE = re.compile(
    r"^tcd_(?:cap(?P<r>[0-9.]+)|ell(?P<ay>[0-9.]+)x(?P<ax>[0-9.]+)x(?P<az>[0-9.]+)(?:x(?P<azp>[0-9.]+))?|sph(?P<R>[0-9.]+))"
    r"(?:_u(?P<u>[0-9.]+)|_p(?P<p>[0-9.]+))?$")


class CurvedTip(TipShape):
    """The capsule, half-ellipsoid and sphere keys (module docstring). The body is the envelope box intersected with
    q(p) <= 1, q = ((x - x_c)+ / a_x)^2 + (y / a_y)^2 + (d_z)^2 with x_c = X_FACE - a_x: d_z = (z_c - z)+ / r for the
    capsule (z_c = Z_END + r, the start of the cap) and (z - z_c) / a_z distally, (z - z_c) / a_zp proximally for the
    half-ellipsoid (z_c = 0). The positive part (x - x_c)+ continues the section straight back to the back face. An
    as-printed key's surface is the convex hull of its layers, like the block family's (upright: layers across z)."""

    def __init__(self, key: str):
        m = CURVED_RE.match(key)
        if not m:
            raise ValueError(f"not a curved tcd key: {key!r}")
        self.key, self.R, self.b = key, None, None
        if m["R"]:
            self.family = "sph"
            self.ax = self.ay = self.az = self.azp = float(m["R"]) * 1e-3
            self.zc = 0.0
        elif m["r"]:
            self.family = "cap"
            self.ax = self.ay = self.az = float(m["r"]) * 1e-3
            self.azp = math.inf
            self.zc = Z_END + self.az
        else:
            self.family = "ell"
            self.ay, self.ax, self.az = (float(m[k]) * 1e-3 for k in ("ay", "ax", "az"))
            self.azp = float(m["azp"]) * 1e-3 if m["azp"] else self.az
            self.zc = 0.0
        if self.family != "sph" and (self.ax > DEPTH + 1e-12 or self.az > LENGTH + 1e-12 or self.zc > Z_MOUNT):
            raise ValueError(f"{key}: semi-axis larger than the envelope")
        self.xc = X_FACE - self.ax
        h = m["u"] or m["p"]
        self.layer = float(h) * 1e-3 if h else None
        self.axis = "z" if m["u"] else ("x" if m["p"] else None)
        self.design = CurvedTip(key.rsplit("_", 1)[0]) if self.layer else None
        self.reach = X_FACE
        self.top_z = Z_MOUNT if self.family != "sph" else self.az
        self.a = min(self.ay, WIDTH / 2)            # the bed spec's nominal tpu<r> field (a label only)

    def inside_design(self, P, tol=0.0):
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        if self.family == "sph":
            c = np.array([self.xc, 0.0, self.zc])
            return np.linalg.norm(P - c, axis=1) <= self.ax + tol
        ok = (x >= X_BACK - tol) & (z <= Z_MOUNT + tol) & (x <= X_FACE + tol) & (np.abs(y) <= WIDTH / 2 + tol) \
            & (z >= Z_END - tol)
        dx = np.maximum(x - self.xc, 0) / self.ax
        dy = y / self.ay
        if self.family == "cap":
            dz = np.maximum(self.zc - z, 0) / self.az
        else:
            dz = np.where(z < self.zc, (z - self.zc) / self.az, (z - self.zc) / self.azp)
        return ok & (dx * dx + dy * dy + dz * dz <= (1.0 + tol / min(self.ax, self.ay, self.az)) ** 2)

    # ------------------------------------------------------------------ upright print: layers across z
    def _outline_z(self, z, n_ang=360):
        """The design's cross-section outline at height z (points in x-y), by radial bisection from an interior point
        on y = 0 (the sections are convex)."""
        xs = np.linspace(X_BACK, X_FACE, 341)
        ins = self.design.inside_design(np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, z)]))
        if not ins.any():
            return np.zeros((0, 3))
        o = np.array([xs[ins].mean(), 0.0, z])
        th = (np.arange(n_ang) + 0.5) / n_ang * 2 * math.pi
        D2 = np.column_stack([np.cos(th), np.sin(th), np.zeros(n_ang)])
        lo, hi = np.zeros(n_ang), np.full(n_ang, 0.03)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            ins = self.design.inside_design(o + mid[:, None] * D2)
            lo, hi = np.where(ins, mid, lo), np.where(ins, hi, mid)
        return o + lo[:, None] * D2

    @cached_property
    def hull_fine(self):
        if self.axis != "z":
            return super().hull_fine
        h = self.layer
        nl = int(math.floor(LENGTH / h + 1e-9))          # whole layers from the bed (the mount face)
        pts = []
        for k in range(nl):
            O = self._outline_z(Z_MOUNT - (k + 0.5) * h)
            if not len(O):
                continue
            for zs in (Z_MOUNT - k * h, Z_MOUNT - (k + 1) * h):
                Q = O.copy()
                Q[:, 2] = zs
                pts.append(Q)
        return _hull(np.concatenate(pts))

    def mesh_paths(self, out_dir, eps=None, res=0.00025):
        """TipShape.mesh_paths at 0.25 mm: a curved tip's contact patch at 1 N spans 2-3 mm, and Drake's field on the
        0.5 mm mesh of the whole 10.55 mm sphere gives a 1.5 % larger approach than the analytic sphere (0.4 % at
        0.25 mm; curved_tips_law.py)."""
        return super().mesh_paths(out_dir, eps=eps, res=res)

    def hull(self, res=0.0005, sign=1.0, eps=None):
        """TipShape.hull; for an upright print without eps, the fine hull's vertices (the layers' outline corners, which
        lie on the printed surface) thinned to the one farthest from the box centre in each res voxel: the layer steps
        near the contact are a few um, and the full corner set has 8-40 thousand vertices."""
        if self.axis != "z" or eps:
            return super().hull(res, sign, eps)
        V0, _ = self.hull_fine
        _, inv = np.unique(np.floor(V0 / res).astype(np.int64), axis=0, return_inverse=True)
        inv = inv.ravel()
        order = np.lexsort((-np.linalg.norm(V0 - C0, axis=1), inv))
        first = order[np.r_[True, inv[order][1:] != inv[order][:-1]]]
        V, F = _hull(V0[first] * np.array([sign, 1.0, 1.0]))
        if sign < 0:
            return _hull(V)
        return V, F

    def _terrace_corners(self, eps=None):
        if self.axis != "z" or not eps:
            return super()._terrace_corners(eps)
        V0, _ = self.hull_fine                           # TipShape._terrace_corners with the layers across z
        out = []
        for zl in np.unique(np.round(V0[:, 2], 9)):
            Q = V0[np.abs(V0[:, 2] - zl) < 5e-10]
            if len(Q) < 4:
                out.append(Q)
                continue
            c = Q.mean(0)
            Q = Q[np.argsort(np.arctan2(Q[:, 1] - c[1], Q[:, 0] - c[0]))]
            i0 = int(np.argmax(np.linalg.norm(Q - Q[0], axis=1)))
            keep = np.zeros(len(Q), bool)
            keep[[0, i0]] = True
            stack = [(0, i0), (i0, len(Q))]
            while stack:
                a, b = stack.pop()
                if b - a < 2:
                    continue
                A, B = Q[a, :2], Q[b % len(Q), :2]
                P = Q[a + 1:b, :2]
                t = B - A
                dist = np.abs(t[0] * (P[:, 1] - A[1]) - t[1] * (P[:, 0] - A[0])) / max(np.linalg.norm(t), 1e-15)
                j = int(np.argmax(dist))
                if dist[j] > eps:
                    keep[a + 1 + j] = True
                    stack += [(a, a + 1 + j), (a + 1 + j, b)]
            out.append(Q[keep])
        return np.concatenate(out)


def shape(key: str) -> TipShape:
    return CurvedTip(key) if CURVED_RE.match(key) else TipShape(key)


def compare_cad(key="tcd_rs3_re3", stl=None):
    """Symmetric Hausdorff distance (mm) and volume ratio of a shape against the CAD insert STL (insert frame)."""
    import trimesh
    stl = stl or Path(__file__).resolve().parents[1] / "assets/cad_v1/fingertip.stl"
    cad = trimesh.load(stl)
    sh = shape(key)
    V, F = sh.hull_fine
    W = np.column_stack([(V[:, 0] - (REACH - 0.0085)) * 1e3, V[:, 1] * 1e3, (Z_MOUNT - V[:, 2]) * 1e3])
    mine = trimesh.Trimesh(W, F, process=True)
    a = trimesh.sample.sample_surface(mine, 60000, seed=0)[0]
    b = trimesh.sample.sample_surface(cad, 60000, seed=0)[0]
    d_ab = np.abs(trimesh.proximity.signed_distance(cad, a))
    d_ba = np.abs(trimesh.proximity.signed_distance(mine, b))
    return {"hausdorff_mm": float(max(d_ab.max(), d_ba.max())), "mean_mm": float(0.5 * (d_ab.mean() + d_ba.mean())),
            "volume_mm3": float(mine.volume), "cad_volume_mm3": float(cad.volume),
            "volume_ratio": float(mine.volume / cad.volume), "cad_bounds": cad.bounds.tolist(),
            "bounds": mine.bounds.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--cad", action="store_true", help="compare with assets/cad_v1/fingertip.stl")
    a = ap.parse_args()
    import time
    for k in a.keys:
        t0 = time.time()
        sh = shape(k)
        V, F = sh.hull(0.0005)
        C, Nn, A, K = sh.pads(full=True)
        cV, vol = sh.volume_centroid
        print(f"{k}: hull {len(V)} vertices; volume {vol * 1e9:.1f} mm3; c_V {np.round(cV * 1e3, 3)} mm; pads {len(C)} "
              f"(area {A.sum() * 1e6:.0f} mm2, a {A.min() * 1e6:.2f}-{A.max() * 1e6:.2f} mm2, K median {np.median(K):.0f} N/m); "
              f"{time.time() - t0:.1f} s")
        if a.cad:
            print(compare_cad(k))


if __name__ == "__main__":
    main()
