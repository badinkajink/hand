"""Analytic convex surfaces for the contact estimator of the hand-object mechanism (HOM) controller.

Each surface is given in its own frame and answers the queries that the estimator of arXiv 2609.25619v2
(Sec. III-D and III-E) makes of a body: signed distance, closest point, outward unit normal, shape operator and the
osculating-circle step that approximates a short geodesic. The paper queries convex decompositions of meshes with
python-fcl and fcpw and interpolates vertex curvatures with libigl. The bodies of this project (the printed TPU
fingertip block with rounded edges, cylinders, spheres and the table) have these quantities in closed form.

Shape operator. For a tangent displacement dp at a surface point with outward normal n, the normal changes by
dn = S dp. S is symmetric, S n = 0, and its eigenvalues on the tangent plane are the principal curvatures, positive
on a convex surface (a sphere of radius r has S = (I - n n^T) / r).
"""
from __future__ import annotations

import math

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-15 else v


class Surface:
    """A convex surface in its own frame."""

    def closest(self, p: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """(signed distance, closest surface point, outward unit normal there) for a point p."""
        raise NotImplementedError

    def shape_operator(self, q: np.ndarray, n: np.ndarray) -> np.ndarray:
        """3x3 shape operator at the surface point q with outward normal n."""
        raise NotImplementedError

    def sdf(self, p: np.ndarray) -> float:
        return self.closest(p)[0]

    def project(self, p: np.ndarray) -> np.ndarray:
        return self.closest(p)[1]

    def normal(self, p: np.ndarray) -> np.ndarray:
        return self.closest(p)[2]

    def geodesic_step(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """The paper's approximate geodesic tracer: from the surface point q, follow the arc of the osculating circle
        of the normal section in the direction of the tangent displacement v, for the arc length |v|; the end point is
        projected back on the surface so that the error of the approximation does not accumulate."""
        s = float(np.linalg.norm(v))
        if s < 1e-15:
            return q.copy()
        _, q, n = self.closest(q)
        t = unit(v - (v @ n) * n)
        k = float(t @ self.shape_operator(q, n) @ t)
        if abs(k * s) < 1e-9:
            e = q + s * t
        else:
            e = q + (math.sin(k * s) / k) * t - ((1.0 - math.cos(k * s)) / k) * n
        return self.closest(e)[1]


class Sphere(Surface):
    def __init__(self, r: float):
        self.r = float(r)

    def closest(self, p):
        rho = float(np.linalg.norm(p))
        n = p / rho if rho > 1e-15 else np.array([1.0, 0.0, 0.0])
        return rho - self.r, self.r * n, n

    def shape_operator(self, q, n):
        return (np.eye(3) - np.outer(n, n)) / self.r


class Cylinder(Surface):
    """Solid cylinder of radius r and half-length hl about the z axis, with flat end caps."""

    def __init__(self, r: float, hl: float):
        self.r, self.hl = float(r), float(hl)

    def _radial(self, p):
        rho = math.hypot(p[0], p[1])
        e = np.array([p[0] / rho, p[1] / rho, 0.0]) if rho > 1e-15 else np.array([1.0, 0.0, 0.0])
        return rho, e

    def closest(self, p):
        rho, e = self._radial(p)
        z = float(p[2])
        if rho <= self.r and abs(z) <= self.hl:                      # inside: nearest of the side and the caps
            if self.r - rho <= self.hl - abs(z):
                return -(self.r - rho), np.array([self.r * e[0], self.r * e[1], z]), e
            sz = math.copysign(1.0, z) if z != 0 else 1.0
            return -(self.hl - abs(z)), np.array([p[0], p[1], sz * self.hl]), np.array([0.0, 0.0, sz])
        q = np.array([min(rho, self.r) * e[0], min(rho, self.r) * e[1], min(max(z, -self.hl), self.hl)])
        d = p - q
        dist = float(np.linalg.norm(d))
        return dist, q, d / dist

    def shape_operator(self, q, n):
        if abs(n[2]) > 1e-6:                                         # cap (or rim): flat
            return np.zeros((3, 3))
        rho, e = self._radial(q)
        et = np.array([-e[1], e[0], 0.0])
        return np.outer(et, et) / self.r


class RoundedBox(Surface):
    """Minkowski sum of a box (half extents half - r) and a sphere of radius r: a box whose every edge and corner is
    rounded with radius r (the printed TPU fingertip block of scripts/fingertip_geometry.py)."""

    def __init__(self, half, r: float, centre=(0.0, 0.0, 0.0)):
        self.half = np.asarray(half, float)
        self.r = float(r)
        self.c = np.asarray(centre, float)
        self.a = np.maximum(self.half - self.r, 0.0)

    def closest(self, p):
        x = np.asarray(p, float) - self.c
        core = np.clip(x, -self.a, self.a)
        d = x - core
        dist = float(np.linalg.norm(d))
        if dist > 1e-12:
            n = d / dist
            return dist - self.r, self.c + core + self.r * n, n
        q = np.abs(x) - self.a                                        # inside the core box: nearest face
        k = int(np.argmax(q))
        sgn = math.copysign(1.0, x[k]) if x[k] != 0 else 1.0
        n = np.zeros(3)
        n[k] = sgn
        s = x.copy()
        s[k] = sgn * (self.a[k] + self.r)
        return float(q[k]) - self.r, self.c + s, n

    def shape_operator(self, qpt, n):
        if self.r <= 0:
            return np.zeros((3, 3))
        x = np.asarray(qpt, float) - self.c - self.r * n            # the core point the surface point sits over
        active = np.abs(x) >= self.a - 1e-9
        act = [k for k in range(3) if active[k] and abs(n[k]) > 1e-9]
        if len(act) <= 1:                                            # flat face
            return np.zeros((3, 3))
        P = np.eye(3) - np.outer(n, n)
        if len(act) == 2:                                            # edge: a cylinder of radius r about the free axis
            k = [i for i in range(3) if i not in act][0]
            e = np.zeros(3)
            e[k] = 1.0
            return (P - np.outer(e, e)) / self.r
        return P / self.r                                            # corner: a sphere of radius r


class Plane(Surface):
    """Half-space n . p <= offset (the table); the surface is n . p = offset with outward normal n."""

    def __init__(self, n=(0.0, 0.0, 1.0), offset: float = 0.0):
        self.n = unit(np.asarray(n, float))
        self.offset = float(offset)

    def closest(self, p):
        d = float(self.n @ p) - self.offset
        return d, p - d * self.n, self.n.copy()

    def shape_operator(self, q, n):
        return np.zeros((3, 3))


class Posed:
    """A surface placed in a body frame: pose (pos, R) of the surface frame in the body frame."""

    def __init__(self, surface: Surface, pos=(0.0, 0.0, 0.0), R=None):
        self.s = surface
        self.pos = np.asarray(pos, float)
        self.R = np.eye(3) if R is None else np.asarray(R, float)

    def closest(self, p_body):
        d, q, n = self.s.closest(self.R.T @ (np.asarray(p_body, float) - self.pos))
        return d, self.pos + self.R @ q, self.R @ n

    def shape_operator(self, q_body, n_body):
        S = self.s.shape_operator(self.R.T @ (q_body - self.pos), self.R.T @ n_body)
        return self.R @ S @ self.R.T

    def geodesic_step(self, q_body, v_body):
        q = self.s.geodesic_step(self.R.T @ (q_body - self.pos), self.R.T @ v_body)
        return self.pos + self.R @ q
