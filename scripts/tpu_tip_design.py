#!/usr/bin/env python3
"""The printed SR2 fingertip as a whole block with an embedded magnet (2026-10-12), for printing and for the finite-element
model of scripts/tpu_tip_magnet.py.

Owner, 2026-10-12: the tips are printed whole, without the mount slot and sockets of `fingertip (3).stl` ("you can have the
fingers relatively whole (except for the magnet embedding), i will figure out attachment separately"), and carry magnets
like eFlesh's fingertips, inserted at a print pause, with gyroid infill in place of eFlesh's cut cells.

Frame (the insert's STL frame used by scripts/tpu_tip_fem_tip.py): palmar face x = +8.5 mm, back face x = -8.5 mm,
width y in [-7.4, 7.4] mm, length z in [0, 21.41] mm with z = 0 on the mount side; the contact point of the bed and the
policies lies on the palmar face at z_c = 21.41 - 10.55 = 10.86 mm. The block is the box rounded on its palmar long
edges (r 2.7 mm, along z) and its palmar end edge (r 3.0 mm, along y), the CAD's radii; the back and the side-end edges
stay square. The solid is the intersection of the box with two cylinder half-spaces, so it is convex.

The magnet is an axially magnetized N52 disc (eFlesh's fingertip magnet: 3/16 in diameter, 1/16 in thick), lying in the
layer plane under the contact point with its axis along x; `depth` is the distance from the palmar face to the magnet's
top face. The pocket is the owner's press fit (2026-10-13: "we have to undersize the holes a little bit ... roughly 90% of
diameter and maybe 95% of height"): 90 % of the magnet's diameter and 95 % of its thickness, its floor at the magnet's bottom
face. Printed face up (the part on its back face, layers across the load), the slicer walls the pocket with perimeters and
closes it with solid layers; the print pauses at the layer that first covers the pocket, the magnet is pressed in, and the
print resumes. The slicer prints the pocket as whole layers, those whose mid-height lies inside it, so its printed height is
a multiple of the layer height; the print sheet gives it.

    .venv/bin/python scripts/tpu_tip_design.py stl --magnet 4.76 1.59 --depth 3.0 --out /tmp/tip.stl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

X_FACE, X_BACK, Y_HALF, Z_END = 8.5, -8.5, 7.4, 21.41
Z_C = Z_END - 10.55
R_SIDE, R_END = 2.7, 3.0
LAYER, FIRST_LAYER = 0.2, 0.2
MAGNETS = {"d4.76x1.59": (4.76, 1.59), "d3.18x1.59": (3.18, 1.59)}   # eFlesh fingertip size; 1/8 x 1/16 in
BR_N52 = 1.45                    # remanence of N52, T (1.42-1.48 on supplier sheets)
FIT_D, FIT_T = 0.90, 0.95        # press-fit pocket over the magnet's diameter and thickness (owner, 2026-10-13)
FE_CLEAR = (0.10, 0.10)          # radial and axial clearance of the pocket the finite elements fill with magnet


def surface_x(y, z):
    """x of the palmar surface over (y, z) inside the footprint (the face is flat at X_FACE between the fillets)."""
    y, z = np.asarray(y, float), np.asarray(z, float)
    dy = np.clip(np.abs(y) - (Y_HALF - R_SIDE), 0, R_SIDE)
    dz = np.clip(z - (Z_END - R_END), 0, R_END)
    xs = (X_FACE - R_SIDE) + np.sqrt(np.clip(R_SIDE ** 2 - dy ** 2, 0, None))
    xe = (X_FACE - R_END) + np.sqrt(np.clip(R_END ** 2 - dz ** 2, 0, None))
    return np.minimum(np.minimum(xs, xe), X_FACE)


def inside_body(x, y, z):
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    box = (x >= X_BACK) & (np.abs(y) <= Y_HALF) & (z >= 0) & (z <= Z_END)
    return box & (x <= surface_x(y, z))


def pocket(magnet, depth, clear_r=None, clear_t=None):
    """Pocket cylinder (radius, x_lo, x_hi) and magnet cylinder (radius, x_lo, x_hi), axis along x through (y=0, z_c).
    Default: the press fit, FIT_D of the diameter and FIT_T of the thickness with its floor at the magnet's bottom face, so the
    magnet stands (1 - FIT_T) of its thickness above it. With clear_r and clear_t: a clearance pocket centred on the magnet,
    which the finite elements take as magnet (FE_CLEAR; the print files until 2026-10-12 had this pocket too)."""
    D, t = MAGNETS[magnet] if isinstance(magnet, str) else magnet
    m_hi = X_FACE - depth
    m_lo = m_hi - t
    mg = dict(r=D / 2, x_lo=m_lo, x_hi=m_hi)
    if clear_r is None:
        return dict(r=FIT_D * D / 2, x_lo=m_lo, x_hi=m_lo + FIT_T * t), mg
    return dict(r=D / 2 + clear_r, x_lo=m_lo - clear_t / 2, x_hi=m_hi + clear_t / 2), mg


def in_cyl(x, y, z, c):
    return (np.asarray(x) >= c["x_lo"]) & (np.asarray(x) <= c["x_hi"]) & ((np.asarray(y) ** 2 + (np.asarray(z) - Z_C) ** 2)
                                                                         <= c["r"] ** 2)


def occupancy(h, magnet=None, depth=None, pad=2):
    """Voxel classes on a uniform grid of side h (mm): 0 outside, 1 printed TPU, 2 magnet (the pocket voxels are the
    magnet's, the clearance taken as filled). Returns (occ uint8, origin)."""
    lo = np.array([X_BACK, -Y_HALF, 0.0]) - pad * h
    hi = np.array([X_FACE, Y_HALF, Z_END]) + pad * h
    n = np.ceil((hi - lo) / h).astype(int)
    xs, ys, zs = (lo[k] + h * (np.arange(n[k]) + 0.5) for k in range(3))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    occ = inside_body(X, Y, Z).astype(np.uint8)
    if magnet is not None:
        pk, _ = pocket(magnet, depth, *FE_CLEAR)
        occ[in_cyl(X, Y, Z, pk) & (occ > 0)] = 2
    return occ, lo


def print_sheet(magnet, depth, clear_r=None, clear_t=None, layer=LAYER):
    """Pause height for the face-up print (part on its back face): print height h = x - X_BACK. The slicer voids the layers
    whose mid-height lies inside the pocket, so the printed pocket is a whole number of layers."""
    pk, mg = pocket(magnet, depth, clear_r, clear_t)
    top = pk["x_hi"] - X_BACK
    bottom = pk["x_lo"] - X_BACK
    n_top = int(math.ceil(top / layer + 0.5 - 1e-9))     # first layer whose mid-height lies above the pocket's top
    n_bot = int(math.ceil(bottom / layer + 0.5 - 1e-9))  # first layer whose mid-height lies inside it
    D, t = MAGNETS[magnet] if isinstance(magnet, str) else magnet
    printed = (n_top - n_bot) * layer
    return dict(magnet=magnet, magnet_diameter_mm=D, magnet_thickness_mm=t, grade="N52, axially magnetized",
                magnet_top_depth_mm=depth, magnet_centre_x_mm=0.5 * (mg["x_lo"] + mg["x_hi"]),
                pocket_fit=(f"press fit, {FIT_D:.0%} of the diameter and {FIT_T:.0%} of the thickness" if clear_r is None
                            else f"clearance {clear_r} mm radial, {clear_t} mm axial"),
                pocket_diameter_mm=round(2 * pk["r"], 3), pocket_height_mm=round(pk["x_hi"] - pk["x_lo"], 3),
                pocket_bottom_print_height_mm=round(bottom, 3), pocket_top_print_height_mm=round(top, 3),
                pocket_layers=n_top - n_bot, printed_pocket_height_mm=round(printed, 3),
                printed_pocket_print_heights_mm=[round((n_bot - 1) * layer, 3), round((n_top - 1) * layer, 3)],
                pause_before_layer_at_mm=round(n_top * layer, 3), pause_layer_number=n_top,
                note=("Print face up, the part on its back face. The slicer walls the pocket and closes it with solid layers; "
                      "pause before the first layer printed above the pocket's top, press the magnet in with its north pole "
                      "toward the palmar face, fix it with a drop of cyanoacrylate (the model takes the magnet as bonded), and "
                      "resume."))


def mesh(magnet=None, depth=None, h=0.12, sections=96, clear_r=None, clear_t=None):
    """Watertight STL mesh: the convex hull of the block's surface samples, and the pocket as an inner shell with its
    faces reversed (a slicer prints it as an internal void)."""
    import trimesh
    ys = np.unique(np.r_[np.arange(-Y_HALF, Y_HALF, h), Y_HALF])
    zs = np.unique(np.r_[np.arange(0.0, Z_END, h), Z_END])
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    front = np.stack([surface_x(Y, Z).ravel(), Y.ravel(), Z.ravel()], 1)
    back = np.array([[X_BACK, sy * Y_HALF, sz] for sy in (-1, 1) for sz in (0.0, Z_END)])
    body = trimesh.convex.convex_hull(np.vstack([front, back]))
    parts = [body]
    if magnet is not None:
        pk, _ = pocket(magnet, depth, clear_r, clear_t)
        cyl = trimesh.creation.cylinder(radius=pk["r"], height=pk["x_hi"] - pk["x_lo"], sections=sections)
        R = trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0])        # axis z -> x
        cyl.apply_transform(R)
        cyl.apply_translation([0.5 * (pk["x_lo"] + pk["x_hi"]), 0.0, Z_C])
        cyl.invert()
        parts.append(cyl)
    m = trimesh.util.concatenate(parts)
    return m


def dipole_B(p_sensor, p_magnet, m_vec):
    """Field (T) of a point dipole m_vec (A m^2) at p_magnet, evaluated at p_sensor (positions in m)."""
    r = np.asarray(p_sensor, float) - np.asarray(p_magnet, float)
    d = np.linalg.norm(r)
    rh = r / d
    return 1e-7 * (3 * rh * (rh @ m_vec) - m_vec) / d ** 3


def magnet_moment(magnet, Br=BR_N52):
    D, t = MAGNETS[magnet] if isinstance(magnet, str) else magnet
    V = math.pi * (D / 2e3) ** 2 * (t / 1e3)
    return Br * V / (4e-7 * math.pi)                    # A m^2


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["stl", "sheet"])
    ap.add_argument("--magnet", default="d4.76x1.59", choices=list(MAGNETS) + ["none"])
    ap.add_argument("--depth", type=float, default=3.0, help="palmar face to the magnet's top face, mm")
    ap.add_argument("--out", required=False)
    a = ap.parse_args()
    mag = None if a.magnet == "none" else a.magnet
    if a.cmd == "stl":
        m = mesh(mag, a.depth)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        m.export(a.out)
        print(f"{a.out}: {len(m.faces)} faces, volume {m.volume:.1f} mm^3, watertight {m.is_watertight}")
    else:
        print(json.dumps(print_sheet(mag, a.depth), indent=1))
