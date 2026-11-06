#!/usr/bin/env python3
"""MJCF pieces of the object seating and tracking plan (docs/notes/20261105-object_seating_and_tracking/), shared by
scripts/hardware/object_tracking_rollouts.py (physics, the repository's .venv) and scripts/hardware/object_tracking_sim.py
(rendering and detection, ~/miniconda3/bin/python). Only xml.etree: no MuJoCo import.

The as-built object (plan, "Objects and tag blocks", with the owner's changes of 2026-11-05): the HOM object (25 mm x
100 mm cylinder or 40 mm sphere), a 4 mm PLA stem from the middle of its underside or its lower pole, and the hollow
hexagonal tag block (30 mm across flats, 17 mm tall, 0.8 mm walls). No ballast: the body's infill makes the total 25 g,
the HOM scenes' mass, so the stem and block lower the centre of mass. The fixed seat stays on the table and the arm lifts
the hand at 1.6 s (owner, 2026-11-05: "just move the ur5 up more"). The block's lower end rests in a 4 mm pocket of the
seat, hexagonal for the cylinder and round for the sphere; the plan's 10 mm cone boss in a socket with a 20 mm mouth is
not modelled, because that socket's upper half would lie inside the block.

Frames. The object body's frame is the HOM object frame O; at the start of every task its +y points up (palm +z) and
its +z along palm -y (hom_tasks.SPHERE_Q). The block frame B has z_B along O's +y and x_B along O's x, so at the start B
is parallel to the palm frame.
"""
from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for _paths
import _paths  # noqa: E402,F401
from sr2hand.geometry.rotations import mat_to_quat  # noqa: E402
from sr2hand.tracking import tags as TG  # noqa: E402

PLA = 1.24e-3                  # g/mm^3
STEM_D = 4.0                   # mm
BLOCK = TG.HexBlock()          # 30 mm across flats, 17 mm tall, 13 mm tag36h11
BLOCK_G = 3.0                  # g: hollow, 0.8 mm walls (plan)
MASS_G = 25.0                  # the HOM scenes' object mass
IDS = {"cylinder": (10, 11, 12, 13, 14, 15), "sphere": (16, 17, 18, 19, 20, 21)}
POCKET_DEPTH = 4.0             # mm: the seat's pocket around the block's lower end
POCKET_CLEAR = 0.3             # mm per side
POCKET_WALL = 3.0              # mm
FRICTION_PLA = "0.35 0.005 0.0001"
R_OB = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])   # columns: x_B, y_B, z_B in O


def block_map(shape):
    """The tag map of the shape's block in B (mm)."""
    return TG.HexBlock(ids=IDS[shape]).tag_map()


def block_centre_O(r_mm, stem_mm):
    """B's origin in O (mm): below the object by the stem and half the block."""
    return np.array([0.0, -(r_mm + stem_mm + BLOCK.height / 2), 0.0])


def T_OB(r_mm, stem_mm):
    T = np.eye(4)
    T[:3, :3] = R_OB
    T[:3, 3] = block_centre_O(r_mm, stem_mm)
    return T


def fmt(v, k=7):
    return " ".join(f"{x:.{k}f}" for x in np.ravel(v))


def quat_of(R):
    """MJCF quaternion of a rotation matrix; a reflection (det -1) is refused, because scipy would return some nearby
    rotation and tilt the geom (2026-11-05: a hand tag's backing plate)."""
    R = np.asarray(R, float)
    if np.linalg.det(R) < 0:
        raise ValueError("quat_of: improper matrix (det < 0)")
    return fmt(mat_to_quat(R), 8)


def add_matte(root):
    """A matte material (no specular highlight), as printed PLA and tag labels are; added once."""
    asset = root.find("asset")
    if asset.find("material[@name='otrack_matte']") is None:
        ET.SubElement(asset, "material", name="otrack_matte", specular="0", shininess="0", reflectance="0")
    return "otrack_matte"


def add_tag_sites(body, tag_map, prefix, T=np.eye(4), raised=0.02, rgba="0.03 0.03 0.03 1", group=None,
                  material=None, white="0.92 0.92 0.92 1"):
    """Every tag of tag_map drawn as a flat mosaic of box sites, its frame mapped into body's by T (mm): each row's
    runs of black and white cells, and a white quiet zone one cell wide round the tag, all 2 x raised (0.04 mm) thick
    on the face. Black cells alone stood proud of the face and showed their side walls at oblique views, which moved
    the detected corners 0.23 px outward; thinner black cells z-fought with the face (2026-11-05). MuJoCo units are
    metres."""
    R, t = T[:3, :3], T[:3, 3]
    for tid, (c, r, d, n) in tag_map.frames.items():
        cell = tag_map.size / TG.n_cells(tag_map.family)
        h = tag_map.size / 2.0
        # (right, down, -normal) is right-handed: a tag map's (right, down, normal) is not, because the viewer looks
        # along -normal with right x down pointing away from the viewer
        Rs = np.stack([R @ r, R @ d, -(R @ n)], 1)
        bits = TG.cells(tag_map.family, tid)
        boxes = []                                       # (centre in the tag's (right, down), half sizes, black)
        for i, row in enumerate(bits):
            j = 0
            while j < len(row):
                k = j
                while k + 1 < len(row) and row[k + 1] == row[j]:
                    k += 1
                boxes.append((((j + k + 1) / 2.0 * cell - h, (i + 0.5) * cell - h), ((k - j + 1) * cell / 2.0, cell / 2.0),
                              bool(row[j])))
                j = k + 1
        q = h + cell / 2.0                               # the quiet zone's strips
        boxes += [((0.0, -q), (h + cell, cell / 2.0), False), ((0.0, q), (h + cell, cell / 2.0), False),
                  ((-q, 0.0), (cell / 2.0, h), False), ((q, 0.0), (cell / 2.0, h), False)]
        for k, ((u, v), (hu, hv), black) in enumerate(boxes):
            p = R @ (c + u * r + v * d + raised * n) + t
            kw = dict(name=f"{prefix}{tid}_{k}", type="box", pos=fmt(p / 1000.0, 8), quat=quat_of(Rs),
                      size=fmt(np.array([hu, hv, raised]) / 1000.0, 8), rgba=rgba if black else white)
            if group is not None:
                kw["group"] = str(group)
            if material:
                kw["material"] = material
            ET.SubElement(body, "site", **kw)


def prism_mesh(asset, name, block=BLOCK):
    v = block.vertices() / 1000.0
    ET.SubElement(asset, "mesh", name=name, vertex=fmt(v, 7))


def masses(shape, r_mm, stem_mm):
    """(body g, stem g, block g): the stem solid PLA, the block 3 g, the body the rest of 25 g."""
    stem = PLA * math.pi * (STEM_D / 2) ** 2 * (stem_mm + 1.0)
    return MASS_G - stem - BLOCK_G, stem, BLOCK_G


def add_assembly(root, obj_name, shape, r_mm, stem_mm, tags=True):
    """Stem and block (with its tags as sites) on the object body of an HOM scene; the body geom's mass is lowered so
    the total stays 25 g. Returns the names of the added geoms."""
    asset = root.find("asset")
    prism_mesh(asset, "otrack_block")
    mat = add_matte(root)
    ob = root.find(f".//body[@name='{obj_name}']")
    g0 = ob.find("geom")
    m_body, m_stem, m_block = masses(shape, r_mm, stem_mm)
    g0.set("mass", f"{m_body / 1000.0:.6f}")
    contype, conaff = g0.get("contype", "1"), g0.get("conaffinity", "1")
    common = dict(contype=contype, conaffinity=conaff, friction=FRICTION_PLA)
    L = stem_mm + 1.0                                           # 1 mm into the body
    yc = -(r_mm - 1.0 + L / 2)
    ET.SubElement(ob, "geom", name="otrack_stem", type="cylinder", size=fmt(np.array([STEM_D / 2, L / 2]) / 1000.0),
                  pos=fmt(np.array([0.0, yc, 0.0]) / 1000.0), quat=quat_of(R_OB), mass=f"{m_stem / 1000.0:.6f}",
                  rgba="0.85 0.85 0.85 1", **common)
    bc = block_centre_O(r_mm, stem_mm)
    ET.SubElement(ob, "geom", name="otrack_block", type="mesh", mesh="otrack_block", pos=fmt(bc / 1000.0),
                  quat=quat_of(R_OB), mass=f"{m_block / 1000.0:.6f}", rgba="0.92 0.92 0.92 1", material=mat, **common)
    if tags:
        add_tag_sites(ob, block_map(shape), "otag", T_OB(r_mm, stem_mm), material=mat)
    return ["otrack_stem", "otrack_block"]


def block_bottom_O(r_mm, stem_mm):
    return block_centre_O(r_mm, stem_mm) + R_OB @ np.array([0.0, 0.0, -BLOCK.height / 2])


def add_seat(root, shape, pos_world_mm, yaw_rad=0.0, pedestal_mm=60.0):
    """The fixed seat as a mocap body (the arm's lift moves it down in the hand's frame), its origin on the pocket's
    floor under the block's centre: a pocket POCKET_DEPTH deep with POCKET_CLEAR of clearance, hexagonal round the
    cylinder's block (its yaw fixed) and round (16 wall pieces) round the sphere's (any yaw), on a pedestal."""
    wb = root.find("worldbody")
    c, s_ = math.cos(yaw_rad), math.sin(yaw_rad)
    seat = ET.SubElement(wb, "body", name="otrack_seat", mocap="true", pos=fmt(np.asarray(pos_world_mm) / 1000.0),
                         quat=fmt([math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)], 8))
    common = dict(contype="1", conaffinity="1", friction=FRICTION_PLA, rgba="0.55 0.6 0.7 1")
    half_w = BLOCK.across_corners / 2 + POCKET_CLEAR + POCKET_WALL + 2.0
    ET.SubElement(seat, "geom", name="otrack_floor", type="box", size=fmt(np.array([half_w, half_w, 1.5]) / 1000.0),
                  pos=fmt(np.array([0.0, 0.0, -1.5]) / 1000.0), **common)
    if shape == "cylinder":
        n, r_in = 6, BLOCK.across_flats / 2 + POCKET_CLEAR
        width = BLOCK.face_width + 2 * POCKET_WALL
        a0 = 0.0
    else:
        n, r_in = 16, BLOCK.across_corners / 2 + POCKET_CLEAR
        width = 2 * math.pi * (r_in + POCKET_WALL) / n * 1.15
        a0 = 0.0
    for k in range(n):
        a = a0 + 2 * math.pi * k / n
        radial = np.array([math.cos(a), math.sin(a), 0.0])
        R = np.stack([np.array([-math.sin(a), math.cos(a), 0.0]), np.array([0.0, 0.0, 1.0]), radial], 1)
        centre = radial * (r_in + POCKET_WALL / 2) + np.array([0.0, 0.0, POCKET_DEPTH / 2])
        ET.SubElement(seat, "geom", name=f"otrack_wall{k}", type="box", pos=fmt(centre / 1000.0, 8), quat=quat_of(R),
                      size=fmt(np.array([width / 2, POCKET_DEPTH / 2, POCKET_WALL / 2]) / 1000.0, 8), **common)
    ET.SubElement(seat, "geom", name="otrack_pedestal", type="cylinder", contype="0", conaffinity="0",
                  size=fmt(np.array([half_w * 0.8, pedestal_mm / 2]) / 1000.0),
                  pos=fmt(np.array([0.0, 0.0, -3.0 - pedestal_mm / 2]) / 1000.0), rgba="0.55 0.6 0.7 1")
    return seat
