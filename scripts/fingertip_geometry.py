#!/usr/bin/env python3
"""The SR2 fingertip as built: a printed TPU block with rounded front edges, its sphere-packed pad, and
writers for MuJoCo (mesh or sphere pads), Drake (compliant convex) and Newton (mesh SDF).

Geometry, from the CAD `fingertip (3).stl` of 2026-08-27 (the TPU insert) on `fingertip_mount (2).stl`:
the block is 17.0 mm deep (flexion direction), 14.8 mm wide (along the PIP axis) and 22.0 mm long
(along the finger). Its palmar face lies 10.55 mm from the finger axis, flush with the link capsule of
`real_hand.xml`, its end face 37.16 mm from the PIP axis (the paper's distal length), and its back face
sits on the mount's 3 mm back plate. The CAD rounds the front and end edges with r = 2.7-3.0 mm; the
user gave 6 mm for the printed tips on 2026-10-04, so the fillet radius is a parameter.

Frame: the tip body of real_hand.xml, origin 26.61 mm below the PIP axis, +z toward the PIP, the finger
running along -z, y along the PIP axis, and the palmar face toward +x for the thumb and -x for index and
middle (`scene_mutate.Scene.FACE_SIGN`). In that frame the block spans x in sgn*[-6.45, 10.55] mm,
y in [-7.4, 7.4] mm and z in [-10.55, 11.45] mm.

The rounded block is the Minkowski sum of a box and a sphere of radius r (every edge rounded; the back
and top edges never touch an object). Pad spheres of radius rs sit on the surface inset by rs, so each
sphere is tangent to the true surface, on every region except the back face and the top face (the mount
side). Each sphere carries the stiffness of its share of a Winkler layer, K_s = E / h * A_s, with h the
depth from the palmar face to the block's centroid (8.5 mm), which is the depth over which Drake's
pressure field for a compliant convex shape rises from zero to E.

    python3 scripts/fingertip_geometry.py --r 6 --s 1 --png /tmp/tip.png      # sample and draw one tip
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

DEPTH, WIDTH, LENGTH = 0.0170, 0.0148, 0.0220
REACH = 0.01055                     # finger axis to the palmar face
END = 0.01055                       # tip-body origin to the end face (37.16 - 26.61 mm)
CENTRE = np.array([REACH - DEPTH / 2, 0.0, -END + LENGTH / 2])     # (+2.05, 0, +0.45) mm, sign +1
HALF = np.array([DEPTH / 2, WIDTH / 2, LENGTH / 2])
FACE_SIGN = {"thumb": +1.0, "index": -1.0, "middle": -1.0}
PIP_TO_TIP = 0.02661                # pip frame to tip body origin, along -z
CAPSULE_R = 0.01055
FOUNDATION = DEPTH / 2              # palmar face to centroid


def _corner_dirs(n):
    """Fibonacci directions on the positive octant (unit vectors with all components >= 0)."""
    out = []
    k = 0
    N = 8 * n
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(N):
        z = 1.0 - (i + 0.5) / N * 2.0
        rr = math.sqrt(max(0.0, 1.0 - z * z))
        th = ga * i
        v = (rr * math.cos(th), rr * math.sin(th), z)
        if min(v) >= 0:
            out.append(v)
            k += 1
    return np.array(out)


def surface_samples(r: float, s: float, sign: float = 1.0):
    """Points, outward normals and area shares on the rounded block's surface at spacing ~s.

    The surface is 6 flat faces (the box shrunk by r on every side), 12 quarter-cylinder edges of radius r
    and 8 sphere octants. Returns (p, n, a) in the tip-body frame for a finger with FACE_SIGN `sign`."""
    a = HALF - r
    if np.any(a < -1e-12):
        raise ValueError(f"fillet {r * 1e3:.2f} mm exceeds a half-dimension of the block")
    a = np.maximum(a, 0.0)
    P, N, A = [], [], []

    def grid(lo, hi, step):
        L = hi - lo
        n = max(1, int(round(L / step))) if L > 1e-9 else 0
        if n == 0:
            return np.array([]), 0.0
        return lo + (np.arange(n) + 0.5) * L / n, L / n

    # faces: axis k, side +-1
    for k in range(3):
        i, j = [x for x in range(3) if x != k]
        ui, di = grid(-a[i], a[i], s)
        uj, dj = grid(-a[j], a[j], s)
        if len(ui) == 0 or len(uj) == 0:
            continue
        for sd in (-1.0, 1.0):
            for x in ui:
                for y in uj:
                    p = np.zeros(3)
                    p[i], p[j], p[k] = x, y, sd * (a[k] + r)
                    nn = np.zeros(3)
                    nn[k] = sd
                    P.append(p), N.append(nn), A.append(di * dj)
    if r > 1e-9:
        # edges: along axis k, at corner (si, sj) of the other two axes
        n_arc = max(1, int(round(0.5 * math.pi * r / s)))
        for k in range(3):
            i, j = [x for x in range(3) if x != k]
            uk, dk = grid(-a[k], a[k], s)
            if len(uk) == 0:
                continue
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    for m in range(n_arc):
                        th = (m + 0.5) / n_arc * 0.5 * math.pi
                        nn = np.zeros(3)
                        nn[i], nn[j] = si * math.cos(th), sj * math.sin(th)
                        for z in uk:
                            p = np.zeros(3)
                            p[i], p[j], p[k] = si * a[i], sj * a[j], z
                            P.append(p + r * nn), N.append(nn), A.append(dk * r * 0.5 * math.pi / n_arc)
        # corners
        n_c = max(1, int(round(0.5 * math.pi * r * r / (s * s))))
        dirs = _corner_dirs(n_c)
        if len(dirs):
            area_each = 0.5 * math.pi * r * r / len(dirs)
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        sv = np.array([sx, sy, sz])
                        for dv in dirs:
                            nn = dv * sv
                            P.append(a * sv + r * nn), N.append(nn), A.append(area_each)
    P, N, A = np.array(P), np.array(N), np.array(A)
    P = P + CENTRE
    flip = np.array([sign, 1.0, 1.0])
    return P * flip, N * flip, A


def pad_mask(n, sign: float = 1.0):
    """Regions that carry pad spheres: everything but the back face (toward the mount plate) and the top
    face (the mount side). n are outward normals in the tip frame of a finger with FACE_SIGN `sign`."""
    nx = n[:, 0] * sign
    return (nx > -0.5) & (n[:, 2] < 0.5)


def pad_spheres(r: float, s: float, rs: float, sign: float = 1.0, E: float = 1e7):
    """Sphere centres (inset by rs), outward normals, area shares and per-sphere stiffness K_s = E/h A_s.

    Centres are clamped into the block shrunk by rs, so that next to an edge sharper than rs (the sharp box)
    a side-face sphere cannot stand proud of the adjacent face."""
    p, n, a = surface_samples(r, s, sign)
    keep = pad_mask(n, sign)
    p, n, a = p[keep], n[keep], a[keep]
    flip = np.array([sign, 1.0, 1.0])
    c0 = CENTRE * flip
    lim = np.maximum(HALF - rs, 0.0)
    centres = np.clip(p - rs * n, c0 - lim, c0 + lim)
    K = E / FOUNDATION * a
    return centres, n, a, K


def hull_mesh(r: float, sign: float = 1.0, res: float = 0.0005):
    """Vertices and triangles of the rounded block (convex hull of its surface samples)."""
    from scipy.spatial import ConvexHull
    p, _, _ = surface_samples(r, res, sign)
    if r <= 1e-9:                                  # a sharp box: its 8 corners suffice
        c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * HALF + CENTRE
        p = c * np.array([sign, 1.0, 1.0])
    h = ConvexHull(p)
    V = h.points[h.vertices]
    remap = {old: new for new, old in enumerate(h.vertices)}
    F = np.array([[remap[i] for i in simp] for simp in h.simplices])
    # orient outward
    cen = V.mean(0)
    for t in range(len(F)):
        a_, b_, c_ = V[F[t]]
        if np.dot(np.cross(b_ - a_, c_ - a_), a_ - cen) < 0:
            F[t] = F[t][[0, 2, 1]]
    return V, F


def write_obj(path: Path, V, F):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# SR2 TPU fingertip, rounded block (scripts/fingertip_geometry.py)\n")
        for v in V:
            fh.write(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n")
        for f in F:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")
    return path


def tag(r: float) -> str:
    return f"r{r * 1e3:.1f}".replace(".", "p")


def mesh_paths(out_dir: Path, r: float):
    """One OBJ per palmar sign: the thumb's (+x) and the index/middle's (-x)."""
    out = {}
    for name, sign in (("pos", 1.0), ("neg", -1.0)):
        V, F = hull_mesh(r, sign)
        out[sign] = write_obj(Path(out_dir) / f"tpu_tip_{tag(r)}_{name}.obj", V, F)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r", type=float, default=6.0, help="fillet radius, mm")
    ap.add_argument("--s", type=float, default=1.0, help="pad spacing, mm")
    ap.add_argument("--rs", type=float, default=0.75, help="pad sphere radius, mm")
    ap.add_argument("--png", type=Path)
    a = ap.parse_args()
    r, s, rs = a.r * 1e-3, a.s * 1e-3, a.rs * 1e-3
    c, n, ar, K = pad_spheres(r, s, rs)
    V, F = hull_mesh(r)
    front = n[:, 0] > 0.99
    print(f"fillet {a.r} mm, spacing {a.s} mm: {len(c)} pad spheres, padded area {ar.sum() * 1e6:.0f} mm2, "
          f"flat palmar face {front.sum()} spheres {ar[front].sum() * 1e6:.0f} mm2; K_s median {np.median(K):.0f} N/m; "
          f"hull {len(V)} vertices, {len(F)} faces, extents {np.ptp(V, 0) * 1e3}")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fig = plt.figure(figsize=(10, 5))
        for k, (el, az) in enumerate(((20, -60), (15, 30))):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            ax.add_collection3d(Poly3DCollection(V[F] * 1e3, facecolors=(0.75, 0.55, 0.35, 0.25), edgecolor="none"))
            ax.scatter(*(c * 1e3).T, s=2, c=n[:, 0], cmap="coolwarm")
            ax.set_box_aspect(np.ptp(V, 0))
            ax.view_init(el, az)
            ax.set_xlabel("x (mm)"), ax.set_ylabel("y (mm)"), ax.set_zlabel("z (mm)")
        fig.suptitle(f"TPU fingertip, fillet {a.r} mm, {len(c)} pad spheres at {a.s} mm")
        fig.savefig(a.png, dpi=90)
        print("wrote", a.png)


if __name__ == "__main__":
    main()
