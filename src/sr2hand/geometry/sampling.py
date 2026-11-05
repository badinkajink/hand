"""Fibonacci lattices on the unit sphere. numpy and scipy have none, and trimesh.sample.sample_surface_sphere is random.

The repository's result pages depend on the exact points of its lattices, so the two spellings of the golden angle stay
apart. pi (3 - sqrt 5) is the default; mirror=True uses pi (1 + sqrt 5) = 4 pi - pi (3 - sqrt 5), which gives the mirror
image in y of the same lattice.
"""
import math

import numpy as np

__all__ = ["fibonacci", "cap_pad"]


def fibonacci(n, cos_min=-1.0, mirror=False):
    """n near-uniform unit vectors (n, 3) with z from 1 down to cos_min (the whole sphere by default; a cap of half-angle
    a about +z with cos_min = cos a): z_k = 1 - (1 - cos_min) (k + 1/2) / n, turning by the golden angle at each k."""
    k = np.arange(n) + 0.5
    z = 1.0 - (1.0 - cos_min) * k / n
    r = np.sqrt(1.0 - z ** 2)
    th = math.pi * (1.0 + math.sqrt(5.0)) * k if mirror else math.pi * (3.0 - math.sqrt(5.0)) * k
    return np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)


def cap_pad(radius, sphere_radius, spacing, cap, modulus):
    """(count, nominal area of a sphere, stiffness of a sphere) of a pad of spheres of `sphere_radius`, about `spacing`
    apart on a cap of half-angle `cap` (rad) of a fingertip of `radius`, for the elastic foundation p = E depth / R. The
    sphere centres lie on the inner sphere R - rs, (R / (R - rs))^2 more densely than on the envelope, so the area of a
    sphere is scaled down by that factor."""
    area = 2 * math.pi * radius ** 2 * (1 - math.cos(cap))
    count = max(1, int(round(area / spacing ** 2)))
    nominal = area / count
    return count, nominal, modulus / radius * nominal * ((radius - sphere_radius) / radius) ** 2
