"""Triangle meshes and mass properties, over numpy and scipy; trimesh where a function says so.

The meshes of the repository go through these functions where the result must keep its vertex order or its file bytes
from run to run. Everything else calls trimesh itself: trimesh.creation.box(bounds=[lo, hi]), trimesh.triangles.normals,
trimesh.triangles.area, trimesh.sample.sample_surface, and a Trimesh's volume, center_mass and moment_inertia.
"""
import os
from pathlib import Path

import numpy as np

__all__ = ["convex_hull", "write_obj", "surface_distance", "parallel_axis", "combine_inertia", "box_inertia",
           "cylinder_inertia", "polygon_section"]


def convex_hull(P):
    """(V, F) of the convex hull of points P (n, 3): the hull's vertices in scipy's order and its triangles wound
    outward. trimesh.convex.convex_hull gives the same surface with the vertices in another order."""
    from scipy.spatial import ConvexHull
    P = np.asarray(P, float)
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


def write_obj(path, V, F, header=None, atomic=True):
    """Write a mesh as OBJ with 7 decimals. With atomic=True it goes through a temporary file, so a process that compiles
    a scene never reads a mesh that a parallel process is rewriting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp") if atomic else path
    with open(tmp, "w") as fh:
        if header:
            fh.write(f"# {header}\n")
        for v in V:
            fh.write(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n")
        for f in F:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")
    if atomic:
        os.replace(tmp, path)
    return path


def surface_distance(mesh, points, chunk=4096):
    """Unsigned distance of points to a trimesh's surface, by trimesh.proximity.closest_point in chunks: on about 670,000
    points at once trimesh reached 24 GB (2026-10-14)."""
    import trimesh
    points = np.asarray(points, float)
    return np.concatenate([trimesh.proximity.closest_point(mesh, points[i:i + chunk])[1]
                           for i in range(0, len(points), chunk)])


def parallel_axis(I, m, d):
    """The inertia I about a body's centre of mass moved to a point at offset d from it: I + m ((d . d) E - d d^T)."""
    d = np.asarray(d, float)
    return np.asarray(I, float) + m * ((d @ d) * np.eye(3) - np.outer(d, d))


def combine_inertia(items):
    """(mass, centre of mass, inertia about it) of parts given as [(mass, centre, inertia about its own centre)]."""
    M = sum(i[0] for i in items)
    c = sum(i[0] * np.asarray(i[1], float) for i in items) / M
    J = np.zeros((3, 3))
    for m, ci, I in items:
        J += parallel_axis(I, m, np.asarray(ci, float) - c)
    return M, c, J


def box_inertia(m, sides):
    """Inertia about the centre of a solid box of mass m with the side lengths (a, b, c)."""
    a, b, c = np.asarray(sides, float) ** 2
    return m / 12.0 * np.diag([b + c, a + c, a + b])


def cylinder_inertia(m, r, h, axis=2):
    """Inertia about the centre of a solid cylinder of mass m, radius r and length h along coordinate `axis`
    (trimesh.inertia.cylinder_inertia for axis 2)."""
    d = np.full(3, m * (3 * r * r + h * h) / 12.0)
    d[axis] = 0.5 * m * r * r
    return np.diag(d)


def polygon_section(polygons):
    """(area, cx, cy, Ixx, Iyy) of shapely polygons taken together, holes subtracted, by
    trimesh.path.polygons.second_moments: Ixx is the integral of (y - cy)^2 dA and Iyy of (x - cx)^2 dA (bending about the
    x and the y axis through the centroid). Each ring goes to trimesh on its own, counter-clockwise: for a polygon with a
    clockwise hole, the usual winding, second_moments adds the hole's moments instead of subtracting them."""
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    from trimesh.path.polygons import second_moments
    A = Sx = Sy = Ixx = Iyy = 0.0
    for p in polygons:
        A += p.area
        Sx += p.area * p.centroid.x
        Sy += p.area * p.centroid.y
        for ring, sign in [(p.exterior, 1.0)] + [(h, -1.0) for h in p.interiors]:
            m = second_moments(orient(Polygon(ring), 1.0))
            Ixx += sign * m[0]
            Iyy += sign * m[1]
    cx, cy = Sx / A, Sy / A
    return A, cx, cy, Ixx - A * cy * cy, Iyy - A * cx * cx
