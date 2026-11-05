"""Distances between points, segments, lines and triangles, over numpy.

Every function takes one value or a batch over leading axes, and arguments broadcast, in any length unit. The module
imports numpy only. numpy, scipy and trimesh have no routine for point to segment, segment to segment or line to line;
for a point to a triangle mesh use trimesh.triangles.closest_point in chunks of 4,096 points, and for geoms in a MuJoCo
model mujoco.mj_geomDistance.
"""
import numpy as np

__all__ = ["point_segment", "segment_segment", "point_line", "line_line", "ray_triangles"]


def _arr(x):
    return np.asarray(x, dtype=float)


def point_segment(P, a, b):
    """Distance from points P to the segments a-b; a zero-length segment is a point. Works in 2-D and 3-D."""
    P, a, b = _arr(P), _arr(a), _arr(b)
    ab = b - a
    t = np.clip(np.sum((P - a) * ab, axis=-1) / np.maximum(np.sum(ab * ab, axis=-1), 1e-18), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[..., None] * ab), axis=-1)


def segment_segment(p1, q1, p2, q2):
    """Closest distance between the segments p1-q1 and p2-q2; zero-length segments are allowed (Ericson, Real-Time
    Collision Detection, 5.1.9)."""
    p1, q1, p2, q2 = _arr(p1), _arr(q1), _arr(p2), _arr(q2)
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
    tiny = 1e-14
    A, E = np.where(a > tiny, a, 1.0), np.where(e > tiny, e, 1.0)
    denom = a * e - b * b
    s = np.where(denom > tiny, np.clip((b * f - c * e) / np.where(denom > tiny, denom, 1.0), 0, 1), 0.0)
    t = (b * s + f) / E
    s = np.where(t < 0, np.clip(-c / A, 0, 1), np.where(t > 1, np.clip((b - c) / A, 0, 1), s))
    t = np.clip(t, 0, 1)
    s, t = np.where(e <= tiny, np.clip(-c / A, 0, 1), s), np.where(e <= tiny, 0.0, t)   # the second a point
    s, t = np.where(a <= tiny, 0.0, s), np.where(a <= tiny, np.clip(f / E, 0, 1), t)    # the first a point
    return np.linalg.norm(p1 + d1 * s[..., None] - (p2 + d2 * t[..., None]), axis=-1)


def point_line(P, a, n):
    """Distance from points P to the line through a along n (any length but zero)."""
    d, n = _arr(P) - _arr(a), _arr(n)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    return np.linalg.norm(d - np.sum(d * n, axis=-1, keepdims=True) * n, axis=-1)


def line_line(p1, a1, p2, a2):
    """Shortest distance between the line through p1 along a1 and the line through p2 along a2 (any lengths but zero);
    lines within 1e-9 rad of parallel take the point-to-line distance."""
    p1, a1, p2, a2 = _arr(p1), _arr(a1), _arr(p2), _arr(a2)
    a1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    d = p2 - p1
    n = np.cross(a1, a2)
    sin = np.linalg.norm(n, axis=-1)
    skew = np.abs(np.sum(d * n, axis=-1)) / np.where(sin < 1e-9, 1.0, sin)
    return np.where(sin < 1e-9, point_line(p2, p1, a1), skew)


def ray_triangles(tri, o, d, merge=1e-6):
    """Sorted parameters t of every crossing of the line o + t d with the triangles tri (n, 3, 3), Moller-Trumbore;
    crossings closer than merge are one crossing (an edge shared by two triangles)."""
    tri, o, d = _arr(tri), _arr(o), _arr(d)
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
    ts = np.sort(t[ok & (u >= 0) & (v >= 0) & (u + v <= 1)])
    if len(ts) and merge > 0:
        ts = ts[np.r_[True, np.diff(ts) > merge]]
    return ts
