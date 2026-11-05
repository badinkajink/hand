"""Directions and vectors, over numpy: unit vectors, angles, components along and across a normal, and the pinhole
projection of points to pixels.

Every function takes one value or a batch over leading axes, and arguments broadcast. The module imports numpy only, so
it runs in every environment of the repository and on the lab computer corvus. Rotations and frames are in
sr2hand.geometry.rotations.
"""
import numpy as np

__all__ = ["unit", "angle", "deg_from_cos", "reject", "reject_norm", "project_look_at", "project_free_camera"]


def _arr(x):
    return np.asarray(x, dtype=float)


def unit(v, zero="raise", eps=1e-12):
    """v / |v| along the last axis. A vector shorter than eps raises ValueError (zero="raise"), comes back unchanged
    (zero="keep") or gives NaN (zero="nan")."""
    v = _arr(v)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    short = n < eps
    if zero == "raise":
        if np.any(short):
            raise ValueError("cannot normalize a zero-length vector")
        return v / n
    if zero == "keep":
        return np.where(short, v, v / np.where(short, 1.0, n))
    if zero == "nan":
        return np.where(short, np.nan, v / np.where(short, 1.0, n))
    raise ValueError(f"unit: zero must be raise, keep or nan, not {zero!r}")


def angle(a, b):
    """Angle (rad, in [0, pi]) between the vectors a and b, of any length, as atan2(|a x b|, a . b). It needs no clip and
    keeps its accuracy near 0 and pi; with a zero vector it is 0."""
    a, b = _arr(a), _arr(b)
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))


def deg_from_cos(c):
    """Angle in degrees of a recorded cosine, clipped to [-1, 1]; a signed cosine is never folded with abs()."""
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def reject(v, n):
    """The part of v perpendicular to the unit vector n: v - (v . n) n."""
    v, n = _arr(v), _arr(n)
    return v - np.sum(v * n, axis=-1, keepdims=True) * n


def reject_norm(v, n):
    """Length of the part of v perpendicular to the unit vector n."""
    return np.linalg.norm(reject(v, n), axis=-1)


def _pinhole(P, eye, fwd, right, up, fovy_deg, w, h):
    f = (h / 2) / np.tan(np.radians(fovy_deg) / 2)
    Q = _arr(P) - eye
    zc = Q @ fwd
    return np.stack([w / 2 + f * (Q @ right) / zc, h / 2 - f * (Q @ up) / zc], axis=-1)


def project_look_at(P, eye, target, fovy_deg, w, h, up=(0.0, 0.0, 1.0)):
    """Pixels (u right, v down) of points P (..., 3) in the w x h image of a camera at eye looking at target. A camera
    that looks along up has no image axes and gives NaN."""
    eye, target = _arr(eye), _arr(target)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, _arr(up))
    right = right / np.linalg.norm(right)
    return _pinhole(P, eye, fwd, right, np.cross(right, fwd), fovy_deg, w, h)


def project_free_camera(P, lookat, distance, azimuth_deg, elevation_deg, fovy_deg, w, h):
    """Pixels (u right, v down) of points P (..., 3) in the w x h image of MuJoCo's free camera."""
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    up = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    return _pinhole(P, _arr(lookat) - distance * fwd, fwd, np.cross(fwd, up), up, fovy_deg, w, h)
