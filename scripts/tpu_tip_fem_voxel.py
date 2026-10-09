#!/usr/bin/env python3
"""Low-memory voxelization of a closed triangle mesh by vertical scanlines (no ray-tracing library).

For each (x, y) column of the grid the script intersects the vertical line with every triangle whose 2D
projection contains the column centre, sorts the crossing heights and fills between pairs. Memory is
bounded by the column chunk (default 256 columns x all triangles).

    python3 scripts/tpu_tip_fem_voxel.py MESH.stl --h 0.2 --out grid.npz
"""
from __future__ import annotations

import argparse

import numpy as np


def load_stl(path):
    import trimesh
    m = trimesh.load(path, force="mesh")
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def voxelize(V, F, h, origin=None, shape=None, chunk=256):
    """Boolean occupancy of voxel centres origin + (i + 1/2) h. Returns (occ, origin)."""
    lo, hi = V.min(0), V.max(0)
    if origin is None:
        origin = lo - 1e-6
    if shape is None:
        shape = tuple(int(np.ceil((hi[k] - origin[k]) / h)) for k in range(3))
    nx, ny, nz = shape
    xs = origin[0] + (np.arange(nx) + 0.5) * h
    ys = origin[1] + (np.arange(ny) + 0.5) * h
    zs = origin[2] + (np.arange(nz) + 0.5) * h
    T = V[F]                                          # (nt, 3, 3)
    # tiny deterministic jitter of the column centres avoids hitting edges and vertices exactly
    jx, jy = 1.37e-7 * h, 2.11e-7 * h
    X, Y = np.meshgrid(xs + jx, ys + jy, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], 1)
    occ = np.zeros((nx * ny, nz), bool)
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    tmin = np.minimum(np.minimum(a[:, :2], b[:, :2]), c[:, :2])
    tmax = np.maximum(np.maximum(a[:, :2], b[:, :2]), c[:, :2])
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    ok = np.abs(det) > 1e-18
    a, b, c, tmin, tmax, det = a[ok], b[ok], c[ok], tmin[ok], tmax[ok], det[ok]
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk]
        inbox = ((p[:, None, 0] >= tmin[None, :, 0]) & (p[:, None, 0] <= tmax[None, :, 0])
                 & (p[:, None, 1] >= tmin[None, :, 1]) & (p[:, None, 1] <= tmax[None, :, 1]))
        ci, ti = np.nonzero(inbox)
        if len(ci) == 0:
            continue
        px, py = p[ci, 0], p[ci, 1]
        A, B, C, D = a[ti], b[ti], c[ti], det[ti]
        l1 = ((B[:, 0] - px) * (C[:, 1] - py) - (C[:, 0] - px) * (B[:, 1] - py)) / D
        l2 = ((C[:, 0] - px) * (A[:, 1] - py) - (A[:, 0] - px) * (C[:, 1] - py)) / D
        l3 = 1.0 - l1 - l2
        hit = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        z = l1 * A[:, 2] + l2 * B[:, 2] + l3 * C[:, 2]
        ci, z = ci[hit], z[hit]
        order = np.lexsort((z, ci))
        ci, z = ci[order], z[order]
        for col in np.unique(ci):
            zc = z[ci == col]
            if len(zc) % 2:
                zc = zc[:-1]
            for z0, z1 in zip(zc[0::2], zc[1::2]):
                occ[s + col] |= (zs >= z0) & (zs <= z1)
    return occ.reshape(nx, ny, nz), np.asarray(origin, float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("--h", type=float, default=0.2)
    ap.add_argument("--out")
    a = ap.parse_args()
    V, F = load_stl(a.mesh)
    occ, o = voxelize(V, F, a.h)
    print(f"{a.mesh}: grid {occ.shape}, solid {occ.mean():.4f}, volume {occ.sum() * a.h ** 3:.1f} mm3")
    if a.out:
        np.savez_compressed(a.out, occ=occ, origin=o, h=a.h)


if __name__ == "__main__":
    main()
