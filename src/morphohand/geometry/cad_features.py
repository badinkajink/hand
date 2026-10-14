"""Cylindrical features of tessellated CAD meshes, and material intervals along an axis.

A CAD export tessellates a cylinder of radius r into rectangular chords. Two adjacent chords
meet at a ruling edge that is parallel to the cylinder axis, both normals are perpendicular to
that edge, and the dihedral angle theta between them fixes the radius from the chord width w:

    r = w / (2 sin(theta / 2))

The axis lies at distance r from the shared edge along the bisector of the two normals: toward
the normals for a hole (concave), away from them for a boss (convex). Grouping ruling pairs by
axis direction, axis line and radius recovers every hole and boss of the part; the grouping does
not grow patches across faces, so it does not leak into tangent fillets or planar faces.

``axial_intervals`` casts rays parallel to an axis at a fixed radius and returns the material
intervals along each ray. Bearing faces, servo seats and clevis arm faces are read from these
intervals, so an assembly can be derived from the parts alone.

All lengths are in the units of the mesh (millimetres for the cad_v1 STLs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh


@dataclass
class Cylinder:
    kind: str  # "hole" or "boss"
    radius: float
    axis: np.ndarray  # unit direction, sign canonicalised
    point: np.ndarray  # point on the axis at the middle of the feature's axial extent
    length: float  # axial extent of the wall faces
    cover_deg: float  # angular coverage of the wall normals, in 5 degree bins
    area: float
    faces: np.ndarray = field(repr=False)

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius


def _canonical(e: np.ndarray) -> np.ndarray:
    return e * np.sign(e @ np.array([1e-3, 1e-2, 1.0]))[:, None]


def _ruling_pairs(mesh: trimesh.Trimesh, min_edge: float = 0.4):
    V, F, N = mesh.vertices, mesh.faces, mesh.face_normals
    fa, th, ed = mesh.face_adjacency, mesh.face_adjacency_angles, mesh.face_adjacency_edges
    e = V[ed[:, 1]] - V[ed[:, 0]]
    L = np.linalg.norm(e, axis=1)
    e = e / L[:, None]
    n1, n2 = N[fa[:, 0]], N[fa[:, 1]]
    ok = (th > np.radians(1.5)) & (th < np.radians(46)) & (L > min_edge)
    ok &= (np.abs((n1 * e).sum(1)) < 0.02) & (np.abs((n2 * e).sum(1)) < 0.02)
    fa, th, ed, e, n1, n2 = fa[ok], th[ok], ed[ok], e[ok], n1[ok], n2[ok]
    mid = 0.5 * (V[ed[:, 0]] + V[ed[:, 1]])

    def width(fi, n):
        t = np.cross(e, n)
        s = np.einsum("kij,kj->ki", V[F[fi]], t)
        return s.max(1) - s.min(1)

    w = 0.5 * (width(fa[:, 0], n1) + width(fa[:, 1], n2))
    r = w / (2 * np.sin(th / 2))
    bis = n1 + n2
    bis /= np.linalg.norm(bis, axis=1, keepdims=True)
    c1, c2 = mesh.triangles_center[fa[:, 0]], mesh.triangles_center[fa[:, 1]]
    concave = ((n2 - n1) * (c2 - c1)).sum(1) < 0
    centre = mid + np.where(concave, 1.0, -1.0)[:, None] * r[:, None] * bis
    return fa, _canonical(e), r, centre, concave


def cylinders(mesh: trimesh.Trimesh, rmax: float = 30.0, min_pairs: int = 3) -> list[Cylinder]:
    """Every cylindrical hole and boss of ``mesh`` with radius below ``rmax``."""
    fa, ax, r, centre, concave = _ruling_pairs(mesh)
    used = np.zeros(len(r), bool)
    out: list[Cylinder] = []
    for i in np.argsort(-r):
        if used[i] or r[i] > rmax:
            continue
        a = ax[i]
        same = (~used) & (np.abs(ax @ a) > np.cos(np.radians(1.5))) & (concave == concave[i])
        same &= np.abs(r - r[i]) < 0.03 * r[i] + 0.05
        d = centre - centre[i]
        d -= (d @ a)[:, None] * a[None, :]
        same &= np.linalg.norm(d, axis=1) < 0.03 * r[i] + 0.08
        idx = np.where(same)[0]
        if len(idx) < min_pairs:
            continue
        used[idx] = True
        faces = np.unique(fa[idx].ravel())
        A = ax[idx].mean(0)
        A /= np.linalg.norm(A)
        C = np.median(centre[idx], axis=0)
        P = mesh.vertices[mesh.faces[faces]].reshape(-1, 3)
        s = (P - C) @ A
        t1 = np.cross(A, [1.0, 0, 0]) if abs(A[0]) < 0.9 else np.cross(A, [0, 1.0, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(A, t1)
        Nf = mesh.face_normals[faces]
        ang = np.degrees(np.arctan2(Nf @ t2, Nf @ t1))
        cover = 5.0 * (np.histogram(ang, bins=72, range=(-180, 180))[0] > 0).sum()
        out.append(Cylinder(
            kind="hole" if concave[i] else "boss", radius=float(np.median(r[idx])), axis=A,
            point=C + A * (s.min() + s.max()) / 2, length=float(s.max() - s.min()), cover_deg=float(cover),
            area=float(mesh.area_faces[faces].sum()), faces=faces))
    return out


def find(cyls: list[Cylinder], diameter: float, kind: str = "hole", tol: float = 0.05,
         min_cover: float = 0.0) -> list[Cylinder]:
    return [c for c in cyls if c.kind == kind and abs(c.diameter - diameter) < tol and c.cover_deg >= min_cover]


def _ray_all(tri: np.ndarray, o: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Ray parameters of every crossing of the line o + t d with the triangles (Moller-Trumbore)."""
    v0 = tri[:, 0]
    e1, e2 = tri[:, 1] - v0, tri[:, 2] - v0
    p = np.cross(d, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - v0
    u = (s * p).sum(1) * inv
    q = np.cross(s, e1)
    v = (q @ d) * inv
    t = (e2 * q).sum(1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1)
    ts = np.sort(t[hit])
    if len(ts):
        ts = ts[np.r_[True, np.diff(ts) > 1e-6]]
    return ts


def perpendicular_basis(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    b1 = np.cross(a, t)
    b1 /= np.linalg.norm(b1)
    return b1, np.cross(a, b1)


def axial_intervals(mesh: trimesh.Trimesh, point, axis, radius: float, n_angles: int = 8,
                    reach: float = 200.0) -> list[tuple[float, list[tuple[float, float]]]]:
    """Material intervals along rays parallel to ``axis`` at ``radius`` from it.

    Returns one (angle_deg, [(lo, hi), ...]) entry per ray; lo and hi are axial coordinates
    measured from ``point`` along ``axis``. The basis that fixes angle 0 is
    ``perpendicular_basis(axis)``.
    """
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    p0 = np.asarray(point, float)
    b1, b2 = perpendicular_basis(a)
    rows = []
    for k in range(n_angles if radius > 0 else 1):
        th = 2 * np.pi * k / n_angles
        o = p0 + radius * (np.cos(th) * b1 + np.sin(th) * b2) - reach * a
        ts = _ray_all(mesh.triangles, o, a) - reach
        rows.append((float(np.degrees(th)), [(float(ts[i]), float(ts[i + 1])) for i in range(0, len(ts) - 1, 2)]))
    return rows


def interval_ends(rows, near: float, side: str) -> float:
    """The interval end nearest to ``near`` over all rays: side 'lo' or 'hi'."""
    ends = [iv[0] if side == "lo" else iv[1] for _, ivs in rows for iv in ivs]
    if not ends:
        raise ValueError("no material along any ray")
    ends = np.asarray(ends)
    return float(ends[np.argmin(np.abs(ends - near))])


def rotation_from_frames(src_a, src_b, dst_a, dst_b) -> np.ndarray:
    """Rotation R with R @ src_a = dst_a and R @ src_b = dst_b (both pairs orthonormalised)."""
    def frame(a, b):
        a = np.asarray(a, float) / np.linalg.norm(a)
        b = np.asarray(b, float) - (np.asarray(b, float) @ a) * a
        b /= np.linalg.norm(b)
        return np.column_stack([a, b, np.cross(a, b)])
    return frame(dst_a, dst_b) @ frame(src_a, src_b).T


def transform(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@dataclass
class SolidGrid:
    """Occupancy of a closed mesh on a regular grid: voxel (i, j, k) has centre origin + pitch*(i, j, k)."""
    origin: np.ndarray
    pitch: float
    occ: np.ndarray  # bool, shape (nx, ny, nz)

    def centres(self) -> np.ndarray:
        idx = np.argwhere(self.occ)
        return self.origin + self.pitch * idx

    def contains(self, P: np.ndarray) -> np.ndarray:
        ijk = np.rint((P - self.origin) / self.pitch).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < np.array(self.occ.shape)), axis=1)
        out = np.zeros(len(P), bool)
        out[ok] = self.occ[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        return out

    @property
    def volume(self) -> float:
        return float(self.occ.sum()) * self.pitch ** 3


def voxelize_solid(mesh: trimesh.Trimesh, pitch: float, margin: int = 2) -> SolidGrid:
    """Solid occupancy by z-scanlines: rasterise every triangle onto the column centres it covers,
    record the crossing heights, and fill between consecutive crossings of each column.

    Memory is the grid plus one crossing per (triangle, covered column), a few MB for a finger part
    at 0.3 mm; trimesh's subdivision voxeliser needs gigabytes at that pitch. The grid origin carries
    an irrational offset so that column centres do not fall on the round-number edges of CAD parts.
    Columns with an odd number of crossings (a centre on an edge) are left empty.
    """
    lo, hi = mesh.bounds
    jitter = np.array([0.1234567, 0.2345671, 0.0]) * pitch
    origin = lo - margin * pitch + jitter
    shape = np.ceil((hi - origin) / pitch).astype(int) + margin + 1
    nx, ny, nz = shape
    tri = mesh.triangles
    cols, zs = [], []
    for t in tri:
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = t
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(det) < 1e-14:
            continue  # vertical in xy: no z-ray crosses it
        i0 = int(np.ceil((min(x0, x1, x2) - origin[0]) / pitch))
        i1 = int(np.floor((max(x0, x1, x2) - origin[0]) / pitch))
        j0 = int(np.ceil((min(y0, y1, y2) - origin[1]) / pitch))
        j1 = int(np.floor((max(y0, y1, y2) - origin[1]) / pitch))
        if i1 < i0 or j1 < j0:
            continue
        I, J = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        X = origin[0] + pitch * I.ravel()
        Y = origin[1] + pitch * J.ravel()
        l0 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / det
        l1 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / det
        l2 = 1.0 - l0 - l1
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not inside.any():
            continue
        cols.append(I.ravel()[inside] * ny + J.ravel()[inside])
        zs.append((l0 * z0 + l1 * z1 + l2 * z2)[inside])
    occ = np.zeros((nx, ny, nz), bool)
    if not cols:
        return SolidGrid(origin, pitch, occ)
    cols = np.concatenate(cols)
    zs = np.concatenate(zs)
    order = np.lexsort((zs, cols))
    cols, zs = cols[order], zs[order]
    starts = np.r_[0, np.flatnonzero(np.diff(cols)) + 1]
    ends = np.r_[starts[1:], len(cols)]
    kz = origin[2] + pitch * np.arange(nz)
    for s, e in zip(starts, ends):
        z = zs[s:e]
        z = z[np.r_[True, np.diff(z) > 1e-9]]  # a ray through a shared edge hits both triangles
        if len(z) % 2:
            continue
        c = cols[s]
        i, j = divmod(int(c), ny)
        for a, b in zip(z[0::2], z[1::2]):
            k0 = int(np.ceil((a - origin[2]) / pitch))
            k1 = int(np.floor((b - origin[2]) / pitch))
            if k1 >= k0:
                occ[i, j, max(k0, 0):min(k1 + 1, nz)] = True
    del kz
    return SolidGrid(origin, pitch, occ)


def line_distance(p1, a1, p2, a2) -> float:
    """Shortest distance between two lines (point, direction)."""
    n = np.cross(a1, a2)
    if np.linalg.norm(n) < 1e-9:
        d = np.asarray(p2) - np.asarray(p1)
        return float(np.linalg.norm(d - (d @ a1) * np.asarray(a1)))
    return float(abs((np.asarray(p2) - np.asarray(p1)) @ (n / np.linalg.norm(n))))
