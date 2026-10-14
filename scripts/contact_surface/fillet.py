"""Front face of a rectangular tip with a measured 6 mm edge roundover.

Analytic patches: flat rectangle, four quarter cylinders, four sphere octants.
Width/length 14.8 mm follow the old hardware-facing pad defaults; length is an
assumption, not CAD verification. The back is outside this contact-only model.
The material foundation K stays E / 10.55 mm: it is not E / local curvature.
"""

import math

import numpy as np

from contact_surface.geometry import Surface


def filleted_surface(spacing_mm=1.0, relaxation=0.02, width=0.0148, length=0.0148, fillet=0.006):
    base = Surface(spacing_mm=spacing_mm, relaxation=relaxation).parameters()
    a, b = width / 2 - fillet, length / 2 - fillet
    if min(a, b) <= 0:
        raise ValueError("This patch construction requires width and length > twice the fillet")
    points, normals, weights, corrections, labels = [], [], [], [], []
    s = spacing_mm * 1e-3
    z0 = base["radius"] - fillet
    rs = base["sphere_radius"]

    def add(point, normal, area, correction, label):
        points.append(point)
        normals.append(normal)
        weights.append(area)
        corrections.append(correction)
        labels.append(label)

    nx, ny = max(1, round(2 * a / s)), max(1, round(2 * b / s))
    for x in (np.arange(nx) + 0.5) / nx * 2 * a - a:
        for y in (np.arange(ny) + 0.5) / ny * 2 * b - b:
            add([x, y, base["radius"]], [0, 0, 1], 4 * a * b / (nx * ny), 1.0, "face")
    for along_x, half in [(True, a), (False, b)]:
        nl, nt = max(1, round(2 * half / s)), max(1, round(math.pi * fillet / (2 * s)))
        area = 2 * half * fillet * math.pi / 2 / (nl * nt)
        for side in [-1, 1]:
            for u in (np.arange(nl) + 0.5) / nl * 2 * half - half:
                for theta in (np.arange(nt) + 0.5) / nt * math.pi / 2:
                    n = (
                        np.array([0, side * math.sin(theta), math.cos(theta)])
                        if along_x
                        else np.array([side * math.sin(theta), 0, math.cos(theta)])
                    )
                    core = np.array([u, side * b, z0]) if along_x else np.array([side * a, u, z0])
                    add(core + fillet * n, n, area, 1 - rs / fillet, "edge")
    ncorner = max(1, round(math.pi * fillet**2 / (2 * s * s)))
    # Equal-area sequence in cos(theta), low-discrepancy azimuth inside each octant.
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for i in range(ncorner):
                z = (i + 0.5) / ncorner
                phi = ((i + 0.5) * (math.sqrt(5) - 1) / 2 % 1) * math.pi / 2
                radial = math.sqrt(1 - z * z)
                n = np.array([sx * radial * math.cos(phi), sy * radial * math.sin(phi), z])
                add(
                    np.array([sx * a, sy * b, z0]) + fillet * n,
                    n,
                    math.pi * fillet**2 / (2 * ncorner),
                    (1 - rs / fillet) ** 2,
                    "corner",
                )
    weights, corrections = np.array(weights), np.array(corrections)
    stiffness = base["modulus"] / base["radius"] * weights * corrections
    impedance = 1 - 1 / ((relaxation / 2) ** 2 * stiffness * base["nominal_inverse_inertia"])
    if np.min(impedance) <= 0.0001:
        raise ValueError("Unrepresentable fillet samples: declare a longer relaxation time")
    samples = [
        dict(
            nominal_area=float(area),
            corrected_area=float(area * corr),
            stiffness=float(k),
            solimp=[float(d), float(d), 0.001, 0.5, 2],
            region=label,
        )
        for area, corr, k, d, label in zip(weights, corrections, stiffness, impedance, labels)
    ]
    base.update(
        tip_shape="filleted_prism",
        fillet_radius=fillet,
        width=width,
        length=length,
        modeled_area=float(weights.sum()),
        count=len(samples),
        sample_parameters=samples,
        geometry_provenance="User: 6 mm front fillet. Width 14.8 mm from scene_mutate; "
        "14.8 mm pad length assumed from its defaults. Analytic geometry, not an imported CAD mesh.",
        area_method="Exact patch areas; offset correction 1, (1-rs/rf), or (1-rs/rf)^2",
    )
    centers = np.round(np.array(points) - rs * np.array(normals), 7)
    return centers, base


def support_point(p, tilt_degrees, azimuth_degrees):
    theta, phi = np.radians([tilt_degrees, azimuth_degrees])
    n = np.array(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
    )
    r = p["fillet_radius"]
    core = np.array(
        [
            np.sign(n[0]) * (p["width"] / 2 - r) if abs(n[0]) > 1e-12 else 0.0,
            np.sign(n[1]) * (p["length"] / 2 - r) if abs(n[1]) > 1e-12 else 0.0,
            p["radius"] - r,
        ]
    )
    point = core + r * n
    quaternion = [
        math.cos(theta / 2),
        -math.sin(phi) * math.sin(theta / 2),
        math.cos(phi) * math.sin(theta / 2),
        0,
    ]
    return n, point, quaternion
