"""Closed-form sphere packings for the primitive collision shapes this repo actually uses.

WHY THIS EXISTS. Our hand is 9 capsules + a few spheres and the screwdriver is one cylinder,
so a closed grasp emits **5 contact points** total -- one per sphere/capsule-vs-cylinder pair.
Every force reading, reward and grip diagnostic in the repo is a function of those 5 scalars,
which is the concrete sense in which contact here is illegible: the contact *patch* has no
representation at all. Replacing a primitive with a dense chain/shell of spheres trades
simulation cost for contact resolution, because MuJoCo emits one contact per sphere pair.

This module is deliberately NOT a mesh approximator (that is MorphIt's job, and it is the
second arm of the experiment). Our shapes are analytic, so the packing is analytic: exact,
free, and controlled by a single interpretable knob.

THE KNOB IS A SURFACE TOLERANCE, NOT A SPHERE COUNT.
A chain of radius-R spheres whose centres are spaced ``d`` apart has a scalloped surface: it
bulges to R at each centre and pinches to ``sqrt(R^2 - d^2/4)`` halfway between. Asking for
"20 spheres" says nothing about how badly that scallop distorts the shape, and the distortion
is what changes the physics. So the caller passes a relative tolerance ``eps`` and we place
the surface symmetrically about the true one:

    peak radius   R      = r * (1 + eps/2)
    valley radius R_min  = r * (1 - eps/2)
    spacing       d      = 2*sqrt(R^2 - R_min^2) = 2*r*sqrt(2*eps)

The inflation matters. Pure inscription (R = r) would make every packed shape uniformly
*thinner* than its source, which reads downstream as a looser grasp -- a physics change
masquerading as a representation change. Splitting the error puts the mean surface on the
original.

MASS IS NOT HANDLED HERE, AND THAT IS A TRAP.
In the frozen scenes the finger links and the screwdriver carry NO explicit ``<inertial>``:
their mass is computed by the compiler from geom volume and density. Overlapping spheres each
contribute their full volume, so swapping one capsule for 18 spheres inflates the link mass
roughly 18x. Callers MUST bake the source body's inertia into an explicit ``<inertial>``
element before substituting geoms -- see ``scripts/generate_sphere_packed_scene.py``, which
reads the compiled ``model.body_mass`` / ``body_inertia`` of the *unpacked* scene and writes
them out. This module only produces shapes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Sphere",
    "chain_spacing",
    "pack_capsule",
    "pack_sphere",
    "pack_cylinder",
    "pack_box",
]


@dataclass(frozen=True)
class Sphere:
    """One packed sphere, in the frame of the geom it replaces."""

    pos: tuple[float, float, float]
    radius: float


def chain_spacing(radius: float, eps: float) -> tuple[float, float]:
    """Return ``(sphere_radius, centre_spacing)`` for a scallop tolerance ``eps``.

    ``eps`` is relative to ``radius``: the packed surface stays within ``+/- eps*radius/2``
    of the true surface. See the module docstring for the derivation.
    """
    if not 0.0 < eps < 2.0:
        raise ValueError(f"eps must be in (0, 2), got {eps}")
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    r_peak = radius * (1.0 + eps / 2.0)
    r_valley = radius * (1.0 - eps / 2.0)
    spacing = 2.0 * math.sqrt(r_peak**2 - r_valley**2)
    return r_peak, spacing


def _lateral_spacing(sphere_radius: float, allowed_dip: float) -> float:
    """Max centre-to-centre spacing of same-radius spheres whose valley dips ``allowed_dip``
    below their common peak. Inverts ``dip = R - sqrt(R^2 - (d/2)^2)``."""
    R = sphere_radius
    if allowed_dip >= R:
        return 2.0 * R
    return 2.0 * math.sqrt(max(R**2 - (R - allowed_dip) ** 2, 0.0))


def pack_sphere(radius: float, eps: float) -> list[Sphere]:
    """A sphere is already a sphere. Returned as-is so callers can treat all geoms uniformly.

    ``eps`` is accepted and ignored: there is no approximation to control.
    """
    del eps
    return [Sphere((0.0, 0.0, 0.0), radius)]


def pack_capsule(
    from_to: tuple[float, float, float, float, float, float],
    radius: float,
    eps: float,
) -> list[Sphere]:
    """Pack a capsule (MJCF ``fromto`` + ``size``) as a chain of spheres along its axis.

    A capsule is exactly a swept sphere, so a chain reproduces it *including the hemispherical
    end caps* with no special-casing -- placing centres at both endpoints inclusive gives caps
    of radius ``R``, which is the capsule's own cap to within ``eps/2``. This is the one shape
    where sphere packing is nearly free of approximation error, and it covers all 9 phalanges
    plus the tips.
    """
    p0 = np.asarray(from_to[:3], dtype=float)
    p1 = np.asarray(from_to[3:], dtype=float)
    r_sphere, spacing = chain_spacing(radius, eps)

    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-12:  # degenerate capsule == sphere
        return [Sphere(tuple(p0), r_sphere)]

    n_gaps = max(1, int(math.ceil(length / spacing)))
    ts = np.linspace(0.0, 1.0, n_gaps + 1)
    return [Sphere(tuple(p0 + t * axis), r_sphere) for t in ts]


def pack_cylinder(
    radius: float,
    half_length: float,
    eps: float,
    *,
    sphere_radius_frac: float = 0.5,
) -> list[Sphere]:
    """Pack a cylinder (MJCF ``size="radius half_length"``, axis +Z) as a surface shell.

    Only the shell is packed. Nothing ever penetrates far enough for interior spheres to
    matter, and they would multiply contact count for no gain.

    ``sphere_radius_frac`` sets the shell sphere radius as a fraction of ``radius``; the ring
    radius is ``radius - sphere_radius`` so the outer surface lands on ``radius``. It trades
    the two things we want against each other and there is no free lunch: LARGER spheres reach
    a given tolerance with far fewer of them (cheap), SMALLER spheres give a finer contact map
    (legible, which is the whole point of arm B). 0.5 is a deliberate middle.

    KNOWN APPROXIMATION: a cylinder's flat cap meets its side in a sharp rim, and a union of
    spheres cannot be sharp. The packed cylinder has a rim rounded at roughly the shell sphere
    radius. For the screwdriver this is a real shape change at the two ends, and it is exactly
    why arm B needs the free-roll fidelity probe before any policy result is believed.
    """
    if not 0.0 < sphere_radius_frac < 1.0:
        raise ValueError(f"sphere_radius_frac must be in (0, 1), got {sphere_radius_frac}")

    r_sphere = radius * sphere_radius_frac
    rho = radius - r_sphere  # ring radius, so rho + r_sphere == radius
    allowed_dip = eps * radius
    max_spacing = _lateral_spacing(r_sphere, allowed_dip)

    spheres: list[Sphere] = []

    # --- lateral surface: rings of spheres, stacked along the axis -------------------
    # Rings sit at |z| <= half_length - r_sphere so the sphere tops stay inside the caps.
    z_extent = max(half_length - r_sphere, 0.0)
    n_rings = max(1, int(math.ceil(2.0 * z_extent / max_spacing)) + 1)
    zs = np.linspace(-z_extent, z_extent, n_rings) if n_rings > 1 else np.array([0.0])

    # Circumferential count from the same dip tolerance: chord 2*rho*sin(pi/N) <= max_spacing.
    if rho > 1e-12:
        ratio = min(1.0, max_spacing / (2.0 * rho))
        n_theta = max(3, int(math.ceil(math.pi / math.asin(ratio))) if ratio < 1.0 else 3)
    else:
        n_theta = 1
    thetas = np.arange(n_theta) * (2.0 * math.pi / n_theta)

    for k, z in enumerate(zs):
        offset = (math.pi / n_theta) if (k % 2 and n_theta > 1) else 0.0  # stagger rings
        for th in thetas + offset:
            spheres.append(
                Sphere((rho * math.cos(th), rho * math.sin(th), float(z)), r_sphere)
            )

    # --- end caps: hex-lattice discs at both ends ----------------------------------
    cap_z = max(half_length - r_sphere, 0.0)
    cap_reach = max(rho, 0.0)
    if cap_reach > 1e-12:
        step = max_spacing
        row_step = step * math.sqrt(3.0) / 2.0
        n_rows = int(math.floor(cap_reach / row_step)) if row_step > 0 else 0
        for iy in range(-n_rows, n_rows + 1):
            y = iy * row_step
            span = math.sqrt(max(cap_reach**2 - y**2, 0.0))
            n_cols = int(math.floor(span / step)) if step > 0 else 0
            x_off = (step / 2.0) if (iy % 2) else 0.0
            for ix in range(-n_cols, n_cols + 1):
                x = ix * step + x_off
                if x * x + y * y > cap_reach**2 + 1e-12:
                    continue
                for sign in (-1.0, 1.0):
                    spheres.append(Sphere((x, y, sign * cap_z), r_sphere))

    return _dedupe(spheres)


def pack_box(
    half_extents: tuple[float, float, float],
    eps: float,
    *,
    sphere_radius_frac: float = 0.5,
) -> list[Sphere]:
    """Pack a box (MJCF ``size`` = half extents) as a surface shell of spheres.

    Same shell-only rationale as ``pack_cylinder``, and the same corner/edge caveat: a union of
    spheres rounds every edge at roughly the shell sphere radius. For the palm slab -- whose
    thickness is 1 mm, far below any sensible sphere -- the packing degenerates to a single
    plate of spheres, which is the honest outcome: a 1 mm slab has no interesting contact
    patch to resolve in the thickness direction.
    """
    if not 0.0 < sphere_radius_frac < 1.0:
        raise ValueError(f"sphere_radius_frac must be in (0, 1), got {sphere_radius_frac}")

    he = np.asarray(half_extents, dtype=float)
    r_sphere = float(min(he.min(), he.max() * sphere_radius_frac))
    r_sphere = max(r_sphere, 1e-5)
    allowed_dip = eps * float(he.max())
    step = _lateral_spacing(r_sphere, allowed_dip)

    inner = np.maximum(he - r_sphere, 0.0)
    axes = [
        (np.linspace(-inner[i], inner[i], max(1, int(math.ceil(2 * inner[i] / step)) + 1))
         if inner[i] > 1e-12 else np.array([0.0]))
        for i in range(3)
    ]

    spheres: list[Sphere] = []
    for x in axes[0]:
        for y in axes[1]:
            for z in axes[2]:
                # shell only: keep a point if it is extremal on at least one axis
                on_shell = any(
                    abs(v) >= inner[i] - 1e-12 for i, v in enumerate((x, y, z))
                )
                if on_shell:
                    spheres.append(Sphere((float(x), float(y), float(z)), r_sphere))
    return _dedupe(spheres)


def _dedupe(spheres: list[Sphere], tol: float = 1e-9) -> list[Sphere]:
    """Drop coincident centres (rings of 1, degenerate caps) so counts are honest."""
    kept: list[Sphere] = []
    seen: set[tuple[int, int, int]] = set()
    q = 1.0 / max(tol, 1e-12)
    for s in spheres:
        key = tuple(int(round(c * q)) for c in s.pos)
        if key in seen:
            continue
        seen.add(key)  # type: ignore[arg-type]
        kept.append(s)
    return kept
