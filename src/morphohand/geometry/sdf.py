"""Signed distances of primitives, negative inside, over numpy and scipy.

P is an array of points (n, 3) in any length unit. numpy and scipy have no analytic signed distances. sd_capped_cone is
Quilez's formula; the others follow from the distance to a box in the coordinates that the primitive makes natural. The
unsigned distance is np.maximum(sd, 0). The scalar closest points of the fingertip surfaces are in
morphohand.control.surfaces.
"""
import numpy as np

__all__ = ["sd_box", "sd_oriented_box", "sd_cylinder", "sd_capped_cone", "sd_convex", "sampled_sdf"]

_HULL_PLANES = {}


def sd_box(P, lo, hi):
    """Signed distance to the axis-aligned box with corners lo and hi."""
    c = (np.asarray(lo) + np.asarray(hi)) / 2
    h = (np.asarray(hi) - np.asarray(lo)) / 2
    q = np.abs(P - c) - h
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


def sd_oriented_box(P, pos, R, half):
    """Signed distance to the box of half sizes half, centred at pos, with its axes the columns of R."""
    q = np.abs((P - pos) @ R) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


def sd_cylinder(P, r, lo, hi, axis=2, centre=(0.0, 0.0)):
    """Signed distance to the solid cylinder of radius r along coordinate `axis` (0, 1, 2) from lo to hi; centre gives
    the other two coordinates of its axis, in increasing order."""
    i, j = [k for k in range(3) if k != axis]
    rho = np.hypot(P[:, i] - centre[0], P[:, j] - centre[1])
    mid, h = (lo + hi) / 2, (hi - lo) / 2
    d = np.c_[rho - r, np.abs(P[:, axis] - mid) - h]
    return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)


def sd_capped_cone(P, z0, z1, r0, r1):
    """Exact signed distance to a cone frustum about z (radius r0 at z0, r1 at z1); Quilez's sdCappedCone."""
    qx = np.hypot(P[:, 0], P[:, 1])
    h = (z1 - z0) / 2
    qy = P[:, 2] - (z0 + z1) / 2
    k1 = np.array([r1, h])
    k2 = np.array([r1 - r0, 2 * h])
    cax = qx - np.minimum(qx, np.where(qy < 0, r0, r1))
    cay = np.abs(qy) - h
    t = np.clip(((k1[0] - qx) * k2[0] + (k1[1] - qy) * k2[1]) / (k2 @ k2), 0.0, 1.0)
    cbx = qx - k1[0] + k2[0] * t
    cby = qy - k1[1] + k2[1] * t
    s = np.where((cbx < 0) & (cay < 0), -1.0, 1.0)
    return s * np.sqrt(np.minimum(cax ** 2 + cay ** 2, cbx ** 2 + cby ** 2))


def sd_convex(P, pts):
    """A lower bound of the signed distance to the convex hull of pts: the largest of its faces' plane distances, exact
    inside and on the faces' side. The planes of a point set are kept between calls."""
    pts = np.asarray(pts, float)
    key = pts.tobytes()
    eq = _HULL_PLANES.get(key)
    if eq is None:
        from scipy.spatial import ConvexHull
        eq = _HULL_PLANES[key] = ConvexHull(pts).equations          # n . p + d <= 0 inside, n unit
    return (P @ eq[:, :3].T + eq[:, 3]).max(axis=1)


def sampled_sdf(tree, samples, normals, Q, k=8):
    """Signed distance of points Q to a surface given by samples with outward normals and a scipy cKDTree of the samples:
    the distance to the nearest sample, signed by the summed offsets of the k nearest samples along their normals."""
    d, j = tree.query(Q, k=k)
    s = np.einsum("ijk,ijk->i", Q[:, None, :] - samples[j], normals[j])
    return d[:, 0] * np.where(s < 0.0, -1.0, 1.0)
