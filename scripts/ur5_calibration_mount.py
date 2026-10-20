#!/usr/bin/env python3
"""UR5e calibration mount for the DIP link with the ReSkin 5X board (hand-off docs/handoff/20261018-ur5_calibration_mount.md).

The mount holds a DIP link as printed (the CAD hand's original link, or v2 a or b of the DIP study: all three keep the clevis
arms of assets/cad_v1/fingertip_mount_DIP_link.stl) on the tool side of an OnRobot HEX-E force-torque sensor on a UR5e, so the
five magnetometers can be calibrated against the HEX-E while the robot presses the 85A tip on a fixed indenter. Three PLA
parts: an adapter disc (the ISO 9409-1-50-4-M6 interface: spigot, dowel, four M6 in counterbores), a standoff-fork (foot,
column, and a fork that fills the clevis: an inner block between the arms, a key in the horn slot, ears for the clamp screws,
a cable slot), and a clamp plate on the horn-side arm. Screws are M3 into 2.75 mm holes in PLA (self-forming, owner) and M6
into the flange. Two optional table indenters (a 5 mm sphere and the 25 mm rod).

Frames, mm. F: the finger frame of the DIP study (scripts/dip_link_geometry.py): x toward the dorsal side, y along the PIP
axis, z up; the PIP axis passes through (0, y, -51.55); the horn-side arm lies at y -23.25..-17.05, the slotted arm at
13.40..17.40. M: the mount frame: origin at the centre of the flange face (the HEX-E QC tool plate's or the UR5e's ISO face),
z_M along the tool axis toward the tip, x_M toward the palmar side (the contact normal), y_M along the PIP axis:
x_M = -x_F, y_M = y_F - Y_AXIS, z_M = -z_F + Z_OFF. The DIP link STL frame D of the DIP study (x_D = -x_F, y_D = y_F + 21.25,
z_D = -z_F - 51.55) goes into M by a translation only: M = D + (0, -21.25 - Y_AXIS, PIP_M). Parts are built in F and exported
in M.

    E=~/miniconda3/envs/femtip/bin/python
    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 400 -- taskset -c 12-15 env NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1"
    $RG $E scripts/ur5_calibration_mount.py interfaces    # DIP link PIP interface from the STLs + flange data -> interfaces.json
    $RG $E scripts/ur5_calibration_mount.py build         # parts as meshes -> cad/*.stl, parts.json
    $RG env UR5M_CMD=step UR5M_REPO=$PWD freecadcmd scripts/ur5_calibration_mount.py   # exact STEP per part -> cad/*.step
    $RG $E scripts/ur5_calibration_mount.py check         # clearances, insertion, access, walls, board distance -> check.json
    $RG $E scripts/ur5_calibration_mount.py estimate      # deflection, HEX-E, wrist field, payload -> estimate.json
    $RG .venv/bin/python scripts/ur5_calibration_mount.py render   # figures -> media/
    $RG $E scripts/ur5_calibration_mount.py print         # PrusaSlicer project for the XL, sliced and checked -> print/
    $E scripts/ur5_calibration_mount.py docs              # parts list, assembly steps -> assembly.json, README.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:                                   # freecadcmd executes the file without __file__
    REPO = Path(os.environ["UR5M_REPO"])
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

# UR5M_REV=v2: the revision of 2026-10-20 after the owner printed v1 (no spigot, one-way foot, the finger on the tool axis,
# a cradle on the DIP link's tongue in place of the key, homing socket and posts); unset: v1 of 2026-10-18 as committed.
REV = os.environ.get("UR5M_REV", "v1")
DATE = "20261018" if REV == "v1" else "20261020"
OUT = REPO / ("docs/experiments/20261018-ur5_calibration_mount" if REV == "v1" else
              "docs/experiments/20261020-ur5_calibration_mount_v2")
CAD, MEDIA, PRINT = OUT / "cad", OUT / "media", OUT / "print"
PFX = "20261018-ur5_calibration_mount" if REV == "v1" else "20261020-ur5_calibration_mount_v2"
WORK = REPO / ("logs/20261018-ur5_calibration_mount" if REV == "v1" else "logs/20261020-ur5_calibration_mount_v2")
DL = REPO / "logs/20261018-ur5_calibration_mount/downloads"   # the data sheets of `interfaces`
DIP = REPO / "docs/experiments/20261018-dip_link_board_mount"
V2 = DIP / "v2"

# ------------------------------------------------------------------------------------------------ frame and placement
PIP_ZF = -51.55                 # z_F of the PIP axis
PIP_M = 46.0                    # PIP axis above the flange face (z_M): v2 boards' nearest edge >= 62 mm, chips >= 74 mm
Y_AXIS = -3.85 if REV == "v1" else 0.0
# y_F of the tool axis: v1 the centre of the fork's footprint (y_F -21.0..13.3); v2 the finger's mid-plane (owner,
# 2026-10-20: "offset the ur5 mount block for the dip finger such that the finger is inline with the center")
Z_OFF = PIP_M + PIP_ZF          # z_M = -z_F + Z_OFF


def F_to_M(p):
    p = np.asarray(p, float)
    return np.stack([-p[..., 0], p[..., 1] - Y_AXIS, -p[..., 2] + Z_OFF], axis=-1)


def M_to_F(p):
    p = np.asarray(p, float)
    return np.stack([-p[..., 0], p[..., 1] + Y_AXIS, -p[..., 2] + Z_OFF], axis=-1)


T_M_FROM_F = np.array([[-1.0, 0, 0, 0], [0, 1.0, 0, -Y_AXIS], [0, 0, -1.0, Z_OFF], [0, 0, 0, 1.0]])
T_M_FROM_D = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, -21.25 - Y_AXIS], [0, 0, 1.0, PIP_M], [0, 0, 0, 1.0]])


def zF(z_m):
    return -z_m + Z_OFF


# ------------------------------------------------------------------------------------------------ the DIP link (measured)
# Values measured from assets/cad_v1/fingertip_mount_DIP_link.stl by `interfaces` (interfaces.json repeats them with the
# measurement); v2 a and b keep the arms within 0.01 mm.
ARM_HORN = (-23.25, -17.05)     # horn-side arm: outer layer, horn slot, inner layer
OUTER = (-23.25, -21.25)
SLOT = (-21.25, -19.25)
INNER = (-19.25, -17.05)
ARM_SLOTTED = (13.40, 17.40)
SLOT_TOP = -53.5                # the slot layer's walls start below this z_F; above it the hub region is open
SLOT_END_Z, SLOT_END_R = -66.05, 2.5        # the slot's round end (bottom at -68.55)
SLOT_TAPER = 0.0345             # half-width = 2.5 + 0.0345 (z_F + 66.05): 2.92 mm at z_F -54
INNER_TOP = -53.6               # the inner layer's highest points (two bumps beside the R 3.2 arc)
DIP_X = (-10.55, 10.748)        # the DIP link's extent in x_F (all three versions)
BRIDGE_R = {"orig": 12.747, "a": 13.35, "b": 13.45}         # least radius of the bridge's top about the PIP axis


def slot_half(z):
    return SLOT_END_R + SLOT_TAPER * (z - SLOT_END_Z)


# ------------------------------------------------------------------------------------------------ parameters
P = dict(
    # adapter disc: the tool side of ISO 9409-1-50-4-M6 (UR5e flange and OnRobot quick changer tool plate)
    ad_d=63.0,              # disc diameter = the flange's 63 mm
    ad_t=11.0,              # disc thickness (z_M 0..11)
    spigot_d=31.40,         # into the 31.50 H7 recess (6.2 mm deep on the UR5e, 7.1 on the QC tool plate)
    spigot_h=5.5, spigot_ch=0.5,
    dowel_d=5.95,           # printed dowel into the 6 H7 hole (6.2 deep UR5e, 6.1 QC)
    dowel_h=5.0, dowel_ch=0.4, dowel_r=25.0,
    dowel_ang=0.0,          # deg in M: on +x_M, the palmar side (the contact normal points toward the dowel)
    bolt_r=25.0,            # 50 mm bolt circle; bolts at 45 deg from the dowel
    m6_hole=6.4, m6_cb_d=13.0,
    m6_under=3.0,           # PLA under the M6 head; M6 x 12 + 1.6 mm washer engage 7.4 mm (UR5e allows 8, QC 9)
    # standoff-fork: foot, column, fork
    foot_t=6.0,             # z_M 11..17
    m3_r=25.5, m3_ang0=22.5,  # four M3 x 12 through the foot into the adapter (2.75 mm self-forming holes, 8 deep)
    m3_clear=3.4, m3_pilot=2.75, m3_depth=8.0,
    fp_x=17.0,              # footprint of the column and fork in F: x +-17.0, y -21.0..13.3, corners R 3
    fp_y=(-21.0, 13.30), fp_r=3.0,
    ib_y=(-16.95, 13.30),   # inner block between the arms: 0.10 mm to each arm face
    ib_r=12.5,              # its lower boundary about the PIP axis: the bridge's top lies at R 12.75 (orig) to 13.45 (b)
    ib_x=11.1,              # half-width of the inner block; the DIP link reaches |x_F| 10.75
    hub_z0=-53.2,           # the hub band (y_F -21.0..-16.95) ends here: slot walls start at -53.5, inner layer at -53.6
    key_y=(-21.0, -19.4),   # 1.6 mm key in the 2.0 mm slot (-21.25..-19.25): the horn side is the y datum (see assembly)
    key_flank=0.10,         # per flank, x
    key_end_r=2.40,         # its round end in the slot's R 2.5 end: key bottom at z_F -68.45, 0.10 above the slot's
    ear_z=-66.0,            # ears |x_F| 11.1..17.0 reach down to here
    cp_t=8.0,               # clamp plate on the horn-side arm's outer face (y_F -31.25..-23.25)
    cp_z=(-65.5, -55.0), cp_screw_x=14.05, cp_screw_z=-60.0, cp_cb_d=6.5, cp_cb_t=3.0, cp_hole=3.3,
    ear_hole_depth=10.0,    # M3 x 16 into the ears from y_F -21.0
    sl_hole_depth=9.0,      # M3 x 12 + DIN 9021 washer through the slotted arm's 5.2 mm bore into the inner block
    cab_y=2.75, cab_x0=1.4, cab_top=-46.0,      # cable slot in the inner block (open dorsally), y_F +-2.75
    groove_x0=13.0,         # cable groove on the column's dorsal face, x_F 13.0..17.0
    tie_zm=(23.0, 35.0), tie_h=3.2, tie_x=(10.0, 11.6), tie_side=(5.75, 7.5),   # zip-tie tunnels under the groove
)
# DIP link concept b (v2 as printed on 2026-10-19 and v3 of 2026-10-20 alike), measured from their STL files: the tongue
# behind the board, the neck's back, the bridge's highest point between the arms, the arms' lowest point, the contact point
B_LINK = dict(tongue_back=8.0, tongue_half=10.5, tongue_end=-98.46, neck_back=9.45, mount_z=-77.05, arms_low=-70.05,
              bridge_top=-61.75, z_contact=-87.91)
if REV == "v2":
    P.update(
        spigot_h=0.0,               # owner: "theres a 30mm diameter disc coming out of the ur5 adaptor, remove this. the hex-e
        #                             i have does not have a bore for that. only the little 5.5mm diameter nub is usable"
        m3_angles=(22.5, 112.5, 202.5, 280.0),   # not 90 deg apart: the foot goes on one way, the palmar side toward the
        #                                          dowel (owner: "the axis from the center of the mount to the 5.5mm nub i
        #                                          believe is the x-axis, i would use that as a reference")
        ear_z=-70.2,                # the dorsal ear reaches down to the cradle, 0.15 below the arms' lowest point
        ear_x_low=9.95, ear_x_low_z=-62.0,       # below z_F -62 the ear stands 0.5 mm behind the bridge's back
        ib_flat_z=-61.70,           # the inner block's flat bottom, 0.05 above the bridge's highest point: the link goes in
        #                             along x under it and the M3 in the lip pushes the bridge onto it (the z datum)
        cab_z0=-68.5,               # the cable slot starts under the v2 link's cable window (z_F -67.9)
        cr_clear_x=0.15,            # tongue's back to the cradle's back plate (the x datum: the normal force presses it on)
        cr_neck_clear=0.5,          # the neck's back to the back plate (v2: the tongue alone is the datum)
        cr_clear_y=0.20, cr_rail_t=2.5, cr_x0=3.0, cr_x1=15.0,   # rails beside the tongue and the neck
        cr_lip_gap=0.6, cr_lip_t=6.0, cr_lip_x0=4.5, cr_screw_x=7.0,   # lip under the tongue's end, M3 x 8 pusher
        home_d=6.3, home_depth=8.0, home_ch=0.6, home_zm=42.0,   # homing socket in the column's palmar face
        post_d=6.0, post_base_d=12.0, post_pin_h=8.5, post_dy=30.0, home_dz=5.0,   # homing post on the indenters' bases;
        #                                          the pin bottoms in the socket with its shoulder 0.5 mm short of the column
    )
MATERIAL = dict(pla_density_g_cm3=1.24, pla_E_MPa=2500.0, pla_nu=0.35,
                src="PLA: density 1.24 g/cm3 (filament); E 2.5 GPa for 100 % printed PLA, the lower end of 2.5-3.5 GPa")
SCREWS = {
    "M6x12_ISO4762": dict(qty=4, what="M6 x 12 socket head cap screw, steel 8.8, with DIN 125 washer (12 mm), into the flange",
                          mass_g=5.4),
    "M3x12_foot": dict(qty=4, what="M3 x 12 socket head cap screw, foot into the adapter (2.75 mm holes)", mass_g=1.2),
    "M3x16_clamp": dict(qty=2, what="M3 x 16 socket head cap screw, brass or A2, clamp plate into the ears", mass_g=1.5),
    "M3x12_slotted": dict(qty=1, what="M3 x 12 socket head cap screw, brass or A2, with DIN 9021 washer 3.2 x 9 x 0.8, "
                                      "through the slotted arm's bore", mass_g=1.4),
}


def fname(stem, ext="stl", sub=CAD):
    return sub / f"{PFX}_{stem}.{ext}"


def rel(p):
    p = Path(p)
    return str(p.relative_to(REPO)) if REPO in p.parents else str(p)


# ------------------------------------------------------------------------------------------------ geometry kernels
def _axis_T(axis):
    """4x4 matrix that maps a solid built along +z onto the given axis ('x', 'y', 'z')."""
    if axis == "z":
        return np.eye(4)
    if axis == "x":            # z -> x, x -> y, y -> z
        return np.array([[0, 0, 1.0, 0], [1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])
    return np.array([[0, 1.0, 0, 0], [0, 0, 1.0, 0], [1.0, 0, 0, 0], [0, 0, 0, 1.0]])   # z -> y, x -> z, y -> x


class MeshK:
    """manifold3d: meshes for the STL files and the checks."""
    name = "mesh"
    SEG = 96

    def __init__(self):
        import manifold3d
        self.M = manifold3d

    def box(self, lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        return self.M.Manifold.cube(list(hi - lo)).translate(list(lo))

    def cyl(self, axis, c, r, a0, a1, r1=None):
        """Cylinder (cone to r1) along axis from a0 to a1; c = the other two coordinates in xyz order."""
        r1 = r if r1 is None else r1
        s = self.M.Manifold.cylinder(a1 - a0, r, r1, self.SEG).translate([0, 0, a0])
        T = _axis_T(axis)
        s = s.transform(T[:3, :])
        if axis == "z":
            return s.translate([c[0], c[1], 0])
        if axis == "x":
            return s.translate([0, c[0], c[1]])
        return s.translate([c[0], 0, c[1]])

    def sphere(self, c, r):
        return self.M.Manifold.sphere(r, self.SEG).translate(list(c))

    def prism_xz(self, poly, y0, y1):
        """Polygon in the x-z plane (list of (x, z)) extruded along y from y0 to y1."""
        poly = np.asarray(poly, float)
        a = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
        if a < 0:
            poly = poly[::-1]
        # extrude (u, v) = (x, -z) along w and map (u, v, w) -> (u, y0 + w, -v): a rotation (det +1)
        s = self.M.Manifold.extrude(self.M.CrossSection([np.column_stack([poly[:, 0], -poly[:, 1]])[::-1]]), y1 - y0)
        T = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, y0], [0, -1.0, 0, 0]])
        return s.transform(T)

    def union(self, parts):
        return self.M.Manifold.batch_boolean(list(parts), self.M.OpType.Add)

    def sub(self, a, b):
        return a - b

    def inter(self, a, b):
        return a ^ b

    def transform(self, s, T):
        return s.transform(np.asarray(T, float)[:3, :])

    def cut_all(self, s, tools):
        return s - self.union(tools)

    def rbox_z(self, x0, x1, y0, y1, z0, z1, r):
        """Box with its four edges along z rounded with radius r."""
        parts = [self.box((x0 + r, y0, z0), (x1 - r, y1, z1)), self.box((x0, y0 + r, z0), (x1, y1 - r, z1))]
        for x in (x0 + r, x1 - r):
            for y in (y0 + r, y1 - r):
                parts.append(self.cyl("z", (x, y), r, z0, z1))
        return self.union(parts)

    def rbox_y(self, x0, x1, y0, y1, z0, z1, r):
        """Box with its four edges along y rounded with radius r."""
        parts = [self.box((x0 + r, y0, z0), (x1 - r, y1, z1)), self.box((x0, y0, z0 + r), (x1, y1, z1 - r))]
        for x in (x0 + r, x1 - r):
            for z in (z0 + r, z1 - r):
                parts.append(self.cyl("y", (x, z), r, y0, y1))
        return self.union(parts)


class BrepK(MeshK):
    """FreeCAD Part, inside freecadcmd: exact solids for the STEP files."""
    name = "brep"

    def __init__(self):
        import FreeCAD
        import Part
        self.F, self.Part = FreeCAD, Part

    def V(self, q):
        return self.F.Vector(*(float(c) for c in q))

    def box(self, lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        d = hi - lo
        return self.Part.makeBox(float(d[0]), float(d[1]), float(d[2]), self.V(lo))

    def cyl(self, axis, c, r, a0, a1, r1=None):
        d = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
        base = {"x": (a0, c[0], c[1]), "y": (c[0], a0, c[1]), "z": (c[0], c[1], a0)}[axis]
        h = float(a1 - a0)
        if r1 is None or abs(r1 - r) < 1e-9:
            return self.Part.makeCylinder(float(r), h, self.V(base), self.V(d))
        return self.Part.makeCone(float(r), float(r1), h, self.V(base), self.V(d))

    def sphere(self, c, r):
        return self.Part.makeSphere(float(r), self.V(c))

    def prism_xz(self, poly, y0, y1):
        pts = [self.V((x, y0, z)) for x, z in poly]
        face = self.Part.Face(self.Part.makePolygon(pts + [pts[0]]))
        return face.extrude(self.V((0, y1 - y0, 0)))

    def union(self, parts):
        parts = list(parts)
        s = parts[0].fuse(parts[1:]) if len(parts) > 1 else parts[0]
        return s.removeSplitter()

    def sub(self, a, b):
        return a.cut(b).removeSplitter()

    def inter(self, a, b):
        return a.common(b).removeSplitter()

    def transform(self, s, T):
        s = s.copy()
        s.transformShape(self.F.Matrix(*[float(v) for v in np.asarray(T, float).ravel()]))
        return s

    def cut_all(self, s, tools):
        """One tool at a time: OpenCASCADE fails on the union of overlapping, coplanar tool boxes."""
        for t in tools:
            s = s.cut(t).removeSplitter()
        return s


# ------------------------------------------------------------------------------------------------ parts (built in F)
def _m_xy(r, ang_deg):
    """(x_F, y_F) of a point at radius r and angle ang (deg, from +x_M toward +y_M) about the tool axis."""
    a = math.radians(ang_deg)
    xm, ym = r * math.cos(a), r * math.sin(a)
    return -xm, ym + Y_AXIS


def bolt_angles(p=P):
    return [p["dowel_ang"] + 45.0 + 90.0 * k for k in range(4)]


def m3_angles(p=P):
    if "m3_angles" in p:
        return [p["dowel_ang"] + a for a in p["m3_angles"]]
    return [p["dowel_ang"] + p["m3_ang0"] + 90.0 * k for k in range(4)]


def adapter(K, p=P):
    """Adapter disc on the flange: spigot and dowel on the robot side (z_M < 0), four M6 counterbored from the tool side,
    four 2.75 mm holes for the foot's M3 screws."""
    c = (0.0, Y_AXIS)
    z0, z1 = zF(p["ad_t"]), zF(0.0)                      # tool face, robot face (z_F)
    body = [K.cyl("z", c, p["ad_d"] / 2, z0, z1)]
    if p["spigot_h"] > 0:                                  # v2 has none: the owner's HEX-E has no recess for it
        sr = p["spigot_d"] / 2
        zs = zF(-p["spigot_h"])
        body.append(K.cyl("z", c, sr, z1 - 0.01, zs - p["spigot_ch"]))
        body.append(K.cyl("z", c, sr, zs - p["spigot_ch"], zs, r1=sr - p["spigot_ch"]))
    dx, dy = _m_xy(p["dowel_r"], p["dowel_ang"])
    dr = p["dowel_d"] / 2
    zd = zF(-p["dowel_h"])
    body.append(K.cyl("z", (dx, dy), dr, z1 - 0.01, zd - p["dowel_ch"]))
    body.append(K.cyl("z", (dx, dy), dr, zd - p["dowel_ch"], zd, r1=dr - p["dowel_ch"]))
    s = K.union(body)
    cut = []
    for a in bolt_angles(p):
        x, y = _m_xy(p["bolt_r"], a)
        cut.append(K.cyl("z", (x, y), p["m6_hole"] / 2, z0 - 1, z1 + 1))
        cut.append(K.cyl("z", (x, y), p["m6_cb_d"] / 2, z0 - 1, zF(p["m6_under"])))
    for a in m3_angles(p):
        x, y = _m_xy(p["m3_r"], a)
        cut.append(K.cyl("z", (x, y), p["m3_pilot"] / 2, z0 - 1, zF(p["ad_t"] - p["m3_depth"])))
    return K.cut_all(s, cut)


def key_poly(p=P, z_top=None):
    """The key's outline in the x-z plane (without its round end): flanks parallel to the slot's, key_flank inside them."""
    z_top = p["hub_z0"] + 0.3 if z_top is None else z_top
    h = lambda z: slot_half(z) - p["key_flank"]                     # noqa: E731
    return [(-h(SLOT_END_Z), SLOT_END_Z), (h(SLOT_END_Z), SLOT_END_Z), (h(z_top), z_top), (-h(z_top), z_top)]


def cradle(K, p=P):
    """v2: a U-channel below the dorsal ear that takes the DIP link's tongue (owner, 2026-10-20: "a better choice is an
    exterior mold that constrains the outer volume rather than this little rod"): a back plate on the tongue's back face,
    the x datum, onto which the normal force presses the tongue; rails beside the tongue and the neck; a lip under the
    tongue's distal end with an M3 that pushes the link toward the flange until the bridge bears on the inner block.
    The DIP link goes in along +x_F (from the palmar side)."""
    L = B_LINK
    x1 = p["cr_x1"]
    yr = L["tongue_half"] + p["cr_clear_y"]
    yo = yr + p["cr_rail_t"]
    zt = p["ear_z"]
    zb = L["tongue_end"] - p["cr_lip_gap"]
    xbt = L["tongue_back"] + p["cr_clear_x"]
    back_t = K.box((xbt, -yo, zb), (x1, yo, L["mount_z"] - 0.3))           # 0.3 mm under the neck's step face
    back_n = K.box((L["neck_back"] + p["cr_neck_clear"], -yo, L["mount_z"] - 0.31), (x1, yo, zt))
    rails = [K.box((p["cr_x0"], yr, zb), (x1, yo, zt)), K.box((p["cr_x0"], -yo, zb), (x1, -yr, zt))]
    xl0 = p["cr_lip_x0"]
    lip = K.prism_xz([(x1, zb + 0.01), (xbt, zb + 0.01), (xl0, zb - (xbt - xl0)), (xl0, zb - p["cr_lip_t"]),
                      (x1, zb - p["cr_lip_t"])], -yo, yo)
    return [back_t, back_n, lip] + rails


def standoff_fork_v2(K, p=P):
    """v2: foot, column, the dorsal ear, the inner block with a flat bottom and the cradle; no key and no palmar ear, so
    the DIP link goes in along +x_F; the homing socket in the column's palmar face."""
    c = (0.0, Y_AXIS)
    fx, (fy0, fy1), r = p["fp_x"], p["fp_y"], p["fp_r"]
    z_foot0, z_foot1 = zF(p["ad_t"] + p["foot_t"]), zF(p["ad_t"])
    foot = K.cyl("z", c, p["ad_d"] / 2, z_foot0, z_foot1)
    top = K.rbox_z(-fx, fx, fy0, fy1, p["hub_z0"], z_foot0 + 0.01, r)
    ear = K.sub(K.rbox_z(-fx, fx, fy0, fy1, p["ear_z"], p["hub_z0"] + 0.01, r),       # the dorsal ear: x_F ib_x..fx
                K.box((-fx - 1.0, fy0 - 1.0, p["ear_z"] - 1.0), (p["ib_x"], fy1 + 1.0, p["hub_z0"] + 1.0)))
    ear_low = K.box((p["ear_x_low"], fy0, p["ear_z"]), (fx, fy1, p["ear_x_low_z"]))
    ib = K.box((-p["ib_x"] - 0.01, p["ib_y"][0], p["ib_flat_z"]), (p["ib_x"] + 0.01, p["ib_y"][1], p["hub_z0"] + 0.01))
    s = K.union([foot, top, ear, ear_low, ib] + cradle(K, p))
    cut = []
    cy = p["cab_y"]
    cut.append(K.box((p["cab_x0"], -cy, p["cab_z0"]), (fx + 1, cy, p["cab_top"])))
    cut.append(K.box((p["groove_x0"], -cy, p["cab_top"] - 0.01), (fx + 1, cy, z_foot1 + 1)))
    cut.append(K.box((p["groove_x0"], -cy, z_foot0 - 0.01), (p["ad_d"], cy, z_foot1 + 1)))
    for zm in p["tie_zm"]:
        z = zF(zm)
        za, zb = z - p["tie_h"] / 2, z + p["tie_h"] / 2
        cut.append(K.box((p["tie_x"][0], -p["tie_side"][1], za), (p["tie_x"][1], p["tie_side"][1], zb)))
        for sgn in (-1, 1):
            ya, yb = sorted((sgn * p["tie_side"][0], sgn * p["tie_side"][1]))
            cut.append(K.box((p["tie_x"][0], ya, za), (fx + 1, yb, zb)))
    cut.append(K.cyl("y", (0.0, PIP_ZF), p["m3_pilot"] / 2, p["ib_y"][1] - p["sl_hole_depth"], p["ib_y"][1] + 1))
    for a in m3_angles(p):
        x, y = _m_xy(p["m3_r"], a)
        cut.append(K.cyl("z", (x, y), p["m3_clear"] / 2, z_foot0 - 1, z_foot1 + 1))
    # the lip's pusher: M3 into a 2.75 mm hole along z
    zb_lip = B_LINK["tongue_end"] - p["cr_lip_gap"]
    cut.append(K.cyl("z", (p["cr_screw_x"], 0.0), p["m3_pilot"] / 2, zb_lip - p["cr_lip_t"] - 1.0, zb_lip + 0.5))
    # homing socket: along x_F into the column's palmar face, on the finger's mid-plane
    zh, rh = zF(p["home_zm"]), p["home_d"] / 2
    cut.append(K.cyl("x", (Y_AXIS, zh), rh, -fx - 1.0, -fx + p["home_depth"]))
    cut.append(K.cyl("x", (Y_AXIS, zh), rh + p["home_ch"] + 1.0, -fx - 1.0, -fx + p["home_ch"], r1=rh))
    return K.cut_all(s, cut)


def standoff_fork(K, p=P):
    """Foot, column and fork in one part."""
    if REV == "v2":
        return standoff_fork_v2(K, p)
    c = (0.0, Y_AXIS)
    fx, (fy0, fy1), r = p["fp_x"], p["fp_y"], p["fp_r"]
    z_foot0, z_foot1 = zF(p["ad_t"] + p["foot_t"]), zF(p["ad_t"])        # foot: z_F -22.55..-16.55
    foot = K.cyl("z", c, p["ad_d"] / 2, z_foot0, z_foot1)
    top = K.rbox_z(-fx, fx, fy0, fy1, p["hub_z0"], z_foot0 + 0.01, r)                 # column, hub and the fork's top
    ears = K.sub(K.rbox_z(-fx, fx, fy0, fy1, p["ear_z"], p["hub_z0"] + 0.01, r),
                 K.box((-p["ib_x"], fy0 - 1, p["ear_z"] - 1), (p["ib_x"], fy1 + 1, p["hub_z0"] + 1)))
    ib = K.inter(K.box((-p["ib_x"] - 0.01, p["ib_y"][0], p["ear_z"]), (p["ib_x"] + 0.01, p["ib_y"][1], p["hub_z0"] + 0.01)),
                 K.cyl("y", (0.0, PIP_ZF), p["ib_r"], p["ib_y"][0] - 1, p["ib_y"][1] + 1))
    y0k, y1k = p["key_y"]
    key = K.union([K.prism_xz(key_poly(p), y0k, y1k),
                   K.cyl("y", (0.0, SLOT_END_Z), p["key_end_r"], y0k, y1k)])
    s = K.union([foot, top, ears, ib, key])
    cut = []
    # cable: slot through the inner block (open dorsally), groove on the column, notch through the foot
    cy = p["cab_y"]
    cut.append(K.box((p["cab_x0"], -cy, p["ear_z"] - 2), (fx + 1, cy, p["cab_top"])))
    cut.append(K.box((p["groove_x0"], -cy, p["cab_top"] - 0.01), (fx + 1, cy, z_foot1 + 1)))
    cut.append(K.box((p["groove_x0"], -cy, z_foot0 - 0.01), (p["ad_d"], cy, z_foot1 + 1)))
    # zip-tie tunnels under the groove, opening into two side pockets
    for zm in p["tie_zm"]:
        z = zF(zm)
        za, zb = z - p["tie_h"] / 2, z + p["tie_h"] / 2
        cut.append(K.box((p["tie_x"][0], -p["tie_side"][1], za), (p["tie_x"][1], p["tie_side"][1], zb)))
        for sgn in (-1, 1):
            ya, yb = sorted((sgn * p["tie_side"][0], sgn * p["tie_side"][1]))
            cut.append(K.box((p["tie_x"][0], ya, za), (fx + 1, yb, zb)))
    # clamp screws into the ears (from the horn side), the slotted-side screw into the inner block
    for sx in (-1, 1):
        cut.append(K.cyl("y", (sx * p["cp_screw_x"], p["cp_screw_z"]), p["m3_pilot"] / 2, fy0 - 1,
                         fy0 + p["ear_hole_depth"]))
    cut.append(K.cyl("y", (0.0, PIP_ZF), p["m3_pilot"] / 2, p["ib_y"][1] - p["sl_hole_depth"], p["ib_y"][1] + 1))
    # foot screws
    for a in m3_angles(p):
        x, y = _m_xy(p["m3_r"], a)
        cut.append(K.cyl("z", (x, y), p["m3_clear"] / 2, z_foot0 - 1, z_foot1 + 1))
    return K.cut_all(s, cut)


def clamp_plate(K, p=P):
    """Plate on the horn-side arm's outer face; two counterbored M3 holes into the ears."""
    x = p["fp_x"]
    y1 = ARM_HORN[0]
    y0 = y1 - p["cp_t"]
    s = K.rbox_y(-x, x, y0, y1, p["cp_z"][0], p["cp_z"][1], 1.5)
    cut = []
    for sx in (-1, 1):
        c = (sx * p["cp_screw_x"], p["cp_screw_z"])
        cut.append(K.cyl("y", c, p["cp_hole"] / 2, y0 - 1, y1 + 1))
        cut.append(K.cyl("y", c, p["cp_cb_d"] / 2, y0 - 1, y0 + p["cp_cb_t"]))
    return K.cut_all(s, cut)


def indenter(K, kind):
    """Table indenter in its own frame (z up, base on the table): a 70 x 24 x 6 mm base with two slots for M6 bolts 50 mm
    apart (an optical table's 25 mm grid), a 20 mm post and either a 5 mm sphere ('sphere', top at 46 mm) or a half rod of
    radius 12.5 mm, the bench screwdriver's ('rod', top at 42.5 mm, its axis along y)."""
    base = K.box((-35.0, -12.0, 0.0), (35.0, 12.0, 6.0))
    slots = []
    for sx in (-1, 1):
        x0, x1 = sorted((sx * 20.0, sx * 30.0))
        slots += [K.box((x0, -3.3, -1), (x1, 3.3, 7)), K.cyl("z", (x0, 0.0), 3.3, -1, 7), K.cyl("z", (x1, 0.0), 3.3, -1, 7)]
    base = K.cut_all(base, slots)
    if kind == "sphere":
        top = K.union([K.cyl("z", (0, 0), 10.0, 5.99, 36.0), K.cyl("z", (0, 0), 10.0, 36.0, 41.0, r1=5.0),
                       K.sphere((0, 0, 41.0), 5.0)])
    else:
        top = K.union([K.box((-12.5, -10.0, 5.99), (12.5, 10.0, 30.0)), K.cyl("y", (0.0, 30.0), 12.5, -10.0, 10.0)])
        top = K.inter(top, K.box((-13, -11, 5.9), (13, 11, 43)))
    if REV != "v2":
        return K.union([base, top])
    # v2: the homing post (owner, 2026-10-20: "add homing geometry so i can zero the ur5 mount to the indenter. dont
    # overthink that one. it can literally be a slot/rod mating"), its pin seated in the mount's socket when the tip's
    # contact point stands home_dz above the indenter's top and post_dy to the side (the table frame: x along the tool axis
    # toward the tip, z up)
    h = homing(kind)
    xp, yp, zt = h["post_xy"][0], h["post_xy"][1], h["pin_top_z"]
    rb, rp_ = P["post_base_d"] / 2, P["post_d"] / 2
    arm = [K.box((xp - rb - 2.0, -12.0, 0.0), (-34.99, 12.0, 6.0)),
           K.box((xp - rb - 2.0, 11.99, 0.0), (xp + rb + 2.0, yp + rb + 2.0, 6.0))]
    post = [K.cyl("z", (xp, yp), rb, 5.99, zt - P["post_pin_h"]), K.cyl("z", (xp, yp), rp_, zt - P["post_pin_h"] - 0.01, zt - 0.5),
            K.cyl("z", (xp, yp), rp_, zt - 0.5, zt, r1=rp_ - 0.5)]
    return K.union([base, top] + arm + post)


TIP_TOP = {"sphere": 46.0, "rod": 42.5}         # the indenters' tops above the table, mm


def homing(kind):
    """v2: where the homing post stands on an indenter's base and the move from homing to the contact. With the palmar side
    down and the tool axis along the base's +x, the socket's floor lies d from the tip's contact point (M -> table: z_M -> x,
    y_M -> y, x_M -> -z); at homing the contact point stands post_dy to +y and home_dz above the indenter's top, so the pin's
    top is at that point + d."""
    a_M = F_to_M(np.array([-10.55, 0.0, B_LINK["z_contact"]]))
    s_M = F_to_M(np.array([-P["fp_x"] + P["home_depth"], Y_AXIS, zF(P["home_zm"])]))
    d = s_M - a_M
    d_tab = np.array([d[2], d[1], -d[0]])
    tip = np.array([0.0, 0.0, TIP_TOP[kind]])
    contact_home = tip + np.array([0.0, P["post_dy"], P["home_dz"]])
    pin = contact_home + d_tab
    return dict(contact_M=np.round(a_M, 3).tolist(), socket_floor_M=np.round(s_M, 3).tolist(),
                contact_to_socket_table=np.round(d_tab, 3).tolist(), post_xy=[round(float(pin[0]), 3), round(float(pin[1]), 3)],
                pin_top_z=round(float(pin[2]), 3), contact_at_homing_table=np.round(contact_home, 3).tolist(),
                move_to_touch_table=[0.0, -P["post_dy"], -P["home_dz"]], indenter_top_z=TIP_TOP[kind])


PARTS = ("adapter", "standoff_fork", "clamp_plate") if REV == "v1" else ("adapter", "standoff_fork")
TABLE_PARTS = ("indenter_sphere", "indenter_rod")


def build_parts(K, p=P):
    """The printed parts in M (and the indenters in their table frame)."""
    out = {}
    for nm, fn in (("adapter", adapter), ("standoff_fork", standoff_fork), ("clamp_plate", clamp_plate)):
        if nm in PARTS:
            out[nm] = K.transform(fn(K, p), T_M_FROM_F)
    out["indenter_sphere"] = indenter(K, "sphere")
    out["indenter_rod"] = indenter(K, "rod")
    return out


# ------------------------------------------------------------------------------------------------ mesh helpers
def to_trimesh(m):
    import trimesh
    mesh = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)


def from_trimesh(tm):
    import manifold3d
    tm = tm.copy()
    tm.merge_vertices()
    mesh = manifold3d.Mesh(vert_properties=np.asarray(tm.vertices, np.float32), tri_verts=np.asarray(tm.faces, np.uint32))
    m = manifold3d.Manifold(mesh)
    if m.status() != manifold3d.Error.NoError or m.volume() <= 0:
        raise RuntimeError(f"manifold import failed: {m.status()}")
    return m


def dip_links():
    """The three DIP links in F: the CAD hand's (orig) and v2 a and b of the DIP study, with their boards' and chips'
    positions and the contact point."""
    import trimesh
    import dip_link_geometry as G
    out = {}
    orig = G.load_part("dip_link")
    out["orig"] = dict(mesh=orig, label="CAD hand's DIP link (fingertip_mount_DIP_link.stl)", z_c=-78.16)
    for k, stem, L in (("a", "a/cad/20261018-dip_link_v2a_dip_link.stl", 8.5), ("b", "b/cad/20261018-dip_link_v2b_dip_link.stl", 9.75)):
        m = trimesh.load(V2 / stem, force="mesh")
        m.merge_vertices()
        out[k] = dict(mesh=m, label=f"v2 {k} of the DIP study ({rel(V2 / stem)})", z_c=-78.16 - L)
    # boards: chips U1-U5 at (u, v) of the 5X board, u = y_F, v = -(z_F - z_c); the board's proximal edge (tab end or J1)
    chips_uv = [(0.0, 0.0), (0.0, 7.0), (7.0, 0.0), (0.0, -7.0), (-7.0, 0.05)]
    for k, d in out.items():
        d["chips_F"] = [(None, u, d["z_c"] - v) for u, v in chips_uv]
    out["orig"]["board_proximal_z"] = -78.16 + 19.10        # the whole 9 mm tab, board on the plate (readiness scenes)
    out["a"]["board_proximal_z"] = -67.56                   # v2a parts.json: board with J1 to z_F -67.56
    out["b"]["board_proximal_z"] = -69.82                   # v2b evaluation.json: J1 header's top
    return out


# ------------------------------------------------------------------------------------------------ commands
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd_interfaces(a):
    """The flange interfaces with their sources, and the DIP link's PIP interface measured from the STLs."""
    import dip_link_geometry as G
    dips = dip_links()
    L = G.line_material
    orig = dips["orig"]["mesh"]
    meas = {}
    meas["y_lines_x0"] = {str(z): L(orig, "y", {"x": 0.0, "z": z}) for z in (-50.0, -55.0, -60.0)}
    sw = {}
    for z in (-54.0, -56.0, -58.05, -60.0, -62.0, -64.0, -66.05):
        iv = L(orig, "x", {"y": -20.25, "z": z})
        neg = [b for lo, b in iv if b < 0]
        pos = [lo for lo, b in iv if lo > 0]
        sw[str(z)] = dict(slot_half=round((min(pos) - max(neg)) / 2, 3), wall_outer_x=round(max(b for lo, b in iv), 3),
                          wall_mm=round(max(b for lo, b in iv) - min(pos), 3))
    meas["horn_slot"] = sw
    meas["slot_bottom_z"] = L(orig, "z", {"x": 0.0, "y": -20.25})
    meas["outer_layer_z_x0"] = L(orig, "z", {"x": 0.0, "y": -22.45})
    meas["inner_layer_z_x0"] = L(orig, "z", {"x": 0.0, "y": -18.2})
    meas["slotted_arm_z_x0"] = L(orig, "z", {"x": 0.0, "y": 15.4})
    meas["slotted_arm_x_at_axis"] = L(orig, "x", {"y": 15.4, "z": PIP_ZF})
    meas["slotted_arm_x_at_z-55"] = L(orig, "x", {"y": 15.4, "z": -55.0})
    meas["cap_y_lines"] = {f"x{x}_z{z}": L(orig, "y", {"x": x, "z": z}) for x, z in ((0.0, -50.0), (3.0, -51.55), (0.0, -55.0))}
    same = {}
    for k in ("a", "b"):
        dm = dips[k]["mesh"]
        rows = []
        for (x, z) in ((0, -50.0), (0, -55.0), (0, -60.0), (3, -51.55), (5, -62.0), (-5, -58.0)):
            rows.append(dict(x=x, z=z, orig=L(orig, "y", {"x": x, "z": z}), v2=L(dm, "y", {"x": x, "z": z})))
        dev = max(abs(a_ - b_) for r_ in rows for ia, ib in zip(r_["orig"], r_["v2"]) for a_, b_ in zip(ia, ib))
        same[k] = dict(rows=rows, max_deviation_mm=round(dev, 3))
    rec = dict(
        title="Interfaces of the UR5e calibration mount: flange side and the DIP link's PIP joint",
        when=time.strftime("%Y-%m-%d %H:%M"),
        flange=dict(
            ur5e=dict(standard="EN ISO 9409-1-50-4-M6", outer_d_mm=63.0, outer_tol="H8 (as drawn)", bolt_circle_mm=50.0,
                      bolt_circle_tol="+-0.1", bolts="4 x M6-6H, thread depth 8 mm (tap drill 5 mm, 8 deep)",
                      dowel_hole="6 H7, depth 6.20 +-0.20 mm, on the bolt circle 45 deg from the bolts, on the tool connector's side",
                      recess="31.50 H7, depth 6.20 mm", flange_length_mm=41.70, connector="Lumberg RKMW 8-354 (M8, 8 pin)",
                      screws="M6 strength class 8.8, ISO 4762, 8.0 N m on the UR5e; no bolt more than 8 mm into the flange",
                      sources=["Universal Robots, UR5e technical specification, updated November 2023, drawing 'Tool flange for "
                               "UR3e, UR5e, UR10e and UR16e' (https://adamscorp.com/hubfs/ur5e_adams.pdf; copy "
                               f"{rel(DL / 'ur5e/ur5e_adams.pdf')}, sha256 {_sha(DL / 'ur5e/ur5e_adams.pdf')[:16]})",
                               "Universal Robots support article 'End effector mounting on tool output flange' (bolt class, "
                               "torque, 8 mm limit), https://www.universal-robots.com/articles/ur/application-installation/"
                               "end-effector-mounting-on-tool-output-flange/"]),
            hex_e_qc=dict(product="OnRobot HEX-E QC 6-axis force-torque sensor", height_flange_to_tool_mm=50.0,
                          dimensions_mm="50 x 71 x 93 (H x W x L; 72 mm diameter, 56 mm with the robot-side spigot)",
                          mass_kg=0.347, tool_side="OnRobot Quick Changer v2 robot side: a tool attaches through a Quick "
                          "Changer tool plate (or an OnRobot tool with its own)", tcp_cog="TCP z 50 mm, CoG z 20 mm, y 5 mm",
                          capacity="Fxy 200 N, Fz 200 N, Txy 10 N m, Tz 6.5 N m",
                          deformation_at_capacity="Fxy +-1.7 mm, Fz +-0.3 mm, Txy +-2.5 deg, Tz +-5 deg (typical)",
                          signal_noise="Fxy 0.035 N, Fz 0.15 N, Txy 0.002 N m, Tz 0.001 N m (1 sigma of a 1 s no-load signal)",
                          noise_free_resolution="Fxy 0.2 N, Fz 0.8 N, Txy 0.01 N m, Tz 0.002 N m (typical)",
                          nonlinearity="< 2 %", hysteresis="< 2 % (measured on Fz)", crosstalk="< 5 %",
                          sources=["OnRobot, Datasheet HEX-E/H QC v1.5 (https://onrobot.com/storage/datasheets/hex/"
                                   f"datasheet_hex-e_h_qc_v1.5_en.pdf; copy {rel(DL / 'hex/datasheet_hex-e_h_qc_v1.5_en.pdf')})",
                                   "OnRobot, User Manual for UR robots, HEX-E/H QC v1.17.0, sec. 8.2.1.1 (mechanical drawing) and "
                                   "8.3.2 (TCP, CoG) (copy " + rel(DL / "hexman/User_Manual_For_UR_Robots_HEX-E_H_QC_v1.17.0_EN.pdf")
                                   + ")"]),
            qc_tool_side=dict(product="OnRobot Quick Changer - Tool side", standard="ISO 9409-1-50-4-M6 face for the tool",
                              outer_d_mm="63 h8", bolt_circle_mm=50.0, bolts="4 x M6, 9 mm deep",
                              dowel_hole="6 H7, 6.10 mm deep, position 0.04 mm to A and B",
                              recess="31.50 H7, 7.10 mm deep", rim_to_tool_face_mm=15.15, overall_mm=28.40, mass_kg=0.14,
                              sources=["OnRobot, Datasheet Quick Changers v2.0, sec. 1.7 (https://onrobot.com/storage/datasheets/"
                                       f"quick-changers/datasheet_quick_changers_v2.0_en.pdf; copy {rel(DL / 'qcds/datasheet_quick_changers_v2.0_en.pdf')})"]),
            mount_side="The adapter fits both: spigot 31.40 mm x 5.5 mm (recess 6.2 or 7.1 deep), dowel 5.95 x 5.0 mm (hole "
                       "6.2 or 6.1 deep), M6 x 12 with a 1.6 mm washer through 3.0 mm of PLA engage 7.4 mm (8 or 9 allowed).",
            stack_mm=dict(hex_e_qc=50.0, qc_tool_side=15.15, ur_flange_to_mount_face=65.15)),
        dip_link=dict(
            frame="F of the DIP study (x dorsal, y along the PIP axis, z up; PIP axis through (0, y, -51.55))",
            horn_side_arm=dict(y=ARM_HORN, outer_layer=OUTER, horn_slot=SLOT, inner_layer=INNER,
                               text="outer layer 2.0 mm (1.6 mm within 3.2 mm of the PIP axis: a 0.4 mm recess of 6.4 mm "
                                    "diameter on its inner face) with the 2.3 mm centre hole on the PIP axis; the horn slot 2.0 mm "
                                    "wide in y for the 1.6 mm SCS0009 horn, shaped like the horn's arm: open toward the PIP axis "
                                    "above z_F -53.5, half-width 2.92 mm at z_F -54 tapering to 2.50 mm, a round end of radius "
                                    "2.5 mm about z_F -66.05 (bottom -68.55); inner layer 2.2 mm, its top an arc of radius 3.2 mm "
                                    "about the PIP axis with two bumps to z_F -53.6, where the servo's output boss passes; four "
                                    "1.6 mm holes through all three layers at x 0, z_F -58.05, -61.05, -63.55, -66.05"),
            slotted_arm=dict(y=ARM_SLOTTED, text="4.0 mm; a 5.2 mm bore on the PIP axis with a 2.0 mm slot from the bore toward "
                                                  "the tip to z_F -56.95; boss radius 5.8 mm about the axis (top at z_F -45.75)"),
            between_arms_mm=round(ARM_SLOTTED[0] - ARM_HORN[1], 2),
            bridge_top_radius_mm=BRIDGE_R,
            on_the_hand=["2.3 mm centre hole: the horn's centre screw into the SCS0009's output spline",
                         "horn slot: the SCS0009 horn (1.6 mm), its hub on the output boss, which enters through the inner "
                         "layer's arc (the servo's boss: radius 2-3 mm, y_F -19.05..-15.85)",
                         "four 1.6 mm holes: through the horn's arm (fasteners not in the CAD)",
                         "5.2 mm bore: the 5 mm shaft of the PIP joint, carried by the MR105 bearing (5 x 10 x 4 mm) in the "
                         "MCP link at y_F 9.4..13.3"],
            measured=meas, v2_same_arms=same),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{PFX}_interfaces.json").write_text(json.dumps(rec, indent=1, default=float))
    print("interfaces ->", rel(OUT / f"{PFX}_interfaces.json"), "| v2 arms deviate",
          {k: v["max_deviation_mm"] for k, v in same.items()}, "mm", flush=True)


def cmd_build(a):
    K = MeshK()
    t0 = time.time()
    parts = build_parts(K)
    CAD.mkdir(parents=True, exist_ok=True)
    rec = {}
    for nm, s in parts.items():
        tm = to_trimesh(s)
        dst = fname(nm)
        tm.export(dst)
        b = np.asarray(tm.bounds)
        rec[nm] = dict(file=rel(dst), volume_mm3=round(float(s.volume()), 1), bounds=np.round(b, 3).tolist(),
                       watertight=bool(tm.is_watertight), triangles=int(len(tm.faces)), status=str(s.status()),
                       frame="M" if nm in PARTS else "table (z up, base on the table)")
        print(f"{nm}: {s.volume():.0f} mm3, bounds {np.round(b, 2).tolist()}", flush=True)
    (OUT / f"{PFX}_parts.json").write_text(json.dumps(dict(parameters=P, frame=dict(
        F="finger frame of the DIP study", M="mount frame: origin at the flange face's centre, z along the tool axis toward "
        "the tip, x toward the palmar side, y along the PIP axis", T_M_from_F=T_M_FROM_F.tolist(),
        T_M_from_D=T_M_FROM_D.tolist(), pip_axis_M=[0.0, None, PIP_M], y_axis_F=Y_AXIS), parts=rec,
        when=time.strftime("%Y-%m-%d %H:%M")), indent=1, default=float))
    print(f"build {time.time() - t0:.1f} s", flush=True)


def freecad_step():
    """Inside freecadcmd: exact STEP of each part, read back for validity, solids and volume."""
    K = BrepK()
    import FreeCAD
    import Import
    import Part
    parts = build_parts(K)
    rec = {}
    for nm, so in parts.items():
        dst = fname(nm, "step")
        # through a document object: Shape.exportStep wrote the rotated clamp plate with a wrong volume on read-back
        doc = FreeCAD.newDocument("ur5m")
        o = doc.addObject("Part::Feature", nm)
        o.Label = nm
        o.Shape = so
        Import.export([o], str(dst))
        FreeCAD.closeDocument("ur5m")
        back = Part.read(str(dst))
        size = dst.stat().st_size
        rec[nm] = dict(file=rel(dst), bytes=size, valid=bool(so.isValid()), solids=len(so.Solids), faces=len(so.Faces),
                       volume_mm3=round(so.Volume, 1), readback_valid=bool(back.isValid()), readback_solids=len(back.Solids),
                       readback_volume_mm3=round(back.Volume, 1))
        print(nm, rec[nm], flush=True)
    (OUT / f"{PFX}_step.json").write_text(json.dumps(dict(parts=rec, note="exact solids from FreeCAD Part (OpenCASCADE) in "
                                                           "the mount frame M (indenters: table frame), mm", when=time.strftime(
                                                           "%Y-%m-%d %H:%M")), indent=1))


# ------------------------------------------------------------------------------------------------ checks
def _load_part(nm):
    import trimesh
    return trimesh.load(fname(nm), force="mesh")


def _sample(tm, spacing=0.15):
    import trimesh
    n = int(max(2000, tm.area / spacing ** 2))
    P_, _ = trimesh.sample.sample_surface_even(tm, n, seed=0)
    return np.asarray(P_)


def cmd_check(a):
    import trimesh
    from scipy.spatial import cKDTree
    K = MeshK()
    t0 = time.time()
    sf_M = from_trimesh(_load_part("standoff_fork"))
    cp_M = from_trimesh(_load_part("clamp_plate"))
    sf = K.transform(sf_M, np.linalg.inv(T_M_FROM_F))           # back to F for the checks
    cp = K.transform(cp_M, np.linalg.inv(T_M_FROM_F))
    sf_tm, cp_tm = to_trimesh(sf), to_trimesh(cp)
    sf_pts = _sample(sf_tm, 0.12)
    sf_tree = cKDTree(sf_pts)
    dips = dip_links()
    res = {}
    for k, d in dips.items():
        dm = from_trimesh(d["mesh"])
        r = {}
        # assembled pose: the DIP link pushed +0.10 in y (horn side on the inner block) and +0.10 in z (slot end on the key)
        for pose, off in (("nominal", (0.0, 0.0, 0.0)), ("assembled", (0.0, 0.10, 0.10))):
            dmo = dm.translate(list(off))
            ov = (dmo ^ sf).volume()
            ov_cp = (dmo ^ cp.translate([0.0, off[1] - 0.02, 0.0])).volume()
            pts = _sample(to_trimesh(dmo), 0.12)
            dd, ii = sf_tree.query(pts)
            near = dd < 1.5
            # gaps by region of the DIP link (F)
            regions = {
                "key_flanks_in_slot": (pts[:, 1] > SLOT[0] + off[1] - 0.01) & (pts[:, 1] < SLOT[1] + off[1] + 0.01) & (pts[:, 2] < SLOT_TOP + off[2]) & (pts[:, 2] > SLOT_END_Z + off[2]),
                "slot_end_under_key": (pts[:, 1] > SLOT[0] + off[1]) & (pts[:, 1] < SLOT[1] + off[1]) & (pts[:, 2] < SLOT_END_Z + off[2] - 1.5),
                "horn_arm_inner_face": (np.abs(pts[:, 1] - (ARM_HORN[1] + off[1])) < 0.02),
                "slotted_arm_inner_face": (np.abs(pts[:, 1] - (ARM_SLOTTED[0] + off[1])) < 0.02),
                "bridge_under_inner_block": (pts[:, 1] > ARM_HORN[1] + off[1] + 0.05) & (pts[:, 1] < ARM_SLOTTED[0] + off[1] - 0.05),
                "horn_arm_tops_under_hub": (pts[:, 1] > SLOT[0] + off[1]) & (pts[:, 1] < INNER[1] + off[1]) & (pts[:, 2] > -54.5 + off[2]) & (np.abs(pts[:, 0]) > 2.0),
            }
            g = {}
            for nm_, m_ in regions.items():
                m_ = m_ & near
                g[nm_] = round(float(dd[m_].min()), 3) if m_.any() else None
            r[pose] = dict(offset_F=off, overlap_fork_mm3=round(float(ov), 4), overlap_clamp_plate_mm3=round(float(ov_cp), 4),
                           least_gap_mm=round(float(dd.min()), 3), gaps_by_region_mm=g)
        # insertion along +z_F from 30 mm below (assembled y offset)
        ins = []
        for dz in np.arange(-30.0, 0.0001, 0.25):
            v = (dm.translate([0.0, 0.10, dz]) ^ sf).volume()
            if v > 1e-4:
                ins.append((round(float(dz), 2), round(float(v), 4)))
        r["insertion_overlaps"] = ins
        r["insertion_clear"] = not ins
        # board and chips from the flange face (z_M)
        chips_zm = [float(F_to_M(np.array([0.0, u, z]))[2]) for _, u, z in d["chips_F"]]
        r["chips_from_flange_mm"] = [round(c, 2) for c in chips_zm]
        r["nearest_chip_from_flange_mm"] = round(min(chips_zm), 2)
        r["board_proximal_edge_from_flange_mm"] = round(float(-d["board_proximal_z"] + Z_OFF), 2)
        r["contact_point_M"] = np.round(F_to_M(np.array([-10.55, 0.0, d["z_c"]])), 3).tolist()
        res[k] = r
        print(k, {p_: (r[p_]["overlap_fork_mm3"], r[p_]["least_gap_mm"]) for p_ in ("nominal", "assembled")},
              "insertion clear", r["insertion_clear"], "nearest chip", r["nearest_chip_from_flange_mm"], flush=True)
    # hex-key access over the foot's M3 heads: a 3.0 mm key column from the head to 150 mm
    acc = []
    for ang in m3_angles():
        x, y = _m_xy(P["m3_r"], ang)
        col = K.cyl("z", (x, y), 1.5, zF(150.0), zF(P["ad_t"] + P["foot_t"] + 3.0))
        v = (col ^ sf).volume() + sum((col ^ from_trimesh(d["mesh"])).volume() for d in dips.values())
        acc.append(dict(angle_deg=ang, overlap_mm3=round(float(v), 4)))
    # the clamp screws' path through the air beside the arm: no DIP link material within the screw's radius + 0.5
    scr = []
    for sx in (-1, 1):
        c = K.cyl("y", (sx * P["cp_screw_x"], P["cp_screw_z"]), 1.5 + 0.5, ARM_HORN[0] - 9, P["fp_y"][0] + 1)
        scr.append(round(float(sum((c ^ from_trimesh(d["mesh"])).volume() for d in dips.values())), 4))
    # walls (analytic)
    walls = dict(
        key_thickness=round(P["key_y"][1] - P["key_y"][0], 3),
        ear_wall_inner=round(P["cp_screw_x"] - P["m3_pilot"] / 2 - P["ib_x"], 3),
        ear_wall_outer=round(P["fp_x"] - P["cp_screw_x"] - P["m3_pilot"] / 2, 3),
        ear_hole_below=round(P["cp_screw_z"] - P["m3_pilot"] / 2 - P["ear_z"], 3),
        tie_floor=round(P["groove_x0"] - P["tie_x"][1], 3),
        slotted_hole_to_cable_slot=round(P["ib_y"][1] - P["sl_hole_depth"] - P["cab_y"], 3),
        foot_m3_to_m6_counterbore=round(_dist_m(P["m3_r"], m3_angles()[0], P["bolt_r"], bolt_angles()[0]) - P["m6_cb_d"] / 2
                                        - P["m3_pilot"] / 2, 3),
        inner_block_to_bridge={k: round(v - P["ib_r"], 3) for k, v in BRIDGE_R.items()},
        ears_to_dip_link_x=round(P["ib_x"] - DIP_X[1], 3),
        hub_to_slot_walls=round(SLOT_TOP - P["hub_z0"], 3) * -1,
        key_end_to_slot_end=round(SLOT_END_R - P["key_end_r"], 3),
    )
    rec = dict(dip_links=res, hex_key_access=acc, clamp_screw_paths_overlap_mm3=scr, walls_mm=walls,
               note="F frame; overlaps by manifold3d booleans of the meshes; gaps by nearest neighbours between surface "
                    "samples 0.12 mm apart (within 0.1 mm)", when=time.strftime("%Y-%m-%d %H:%M"))
    (OUT / f"{PFX}_check.json").write_text(json.dumps(rec, indent=1, default=float))
    print("access", acc, "screw paths", scr, "walls", walls, f"[{time.time() - t0:.0f} s]", flush=True)


def b_links_v2():
    """The DIP links of concept b that the v2 mount takes, in F: v2 as printed on 2026-10-19 and v3 of 2026-10-20, with the
    pad and the cover of v3 (the cradle meets the tongue, which is the same in both)."""
    import trimesh
    out = {}
    for k, f in (("v2b", "v2/b/cad/20261018-dip_link_v2b_dip_link.stl"), ("v3b", "v3/b/cad/20261020-dip_link_v3b_dip_link.stl")):
        m = trimesh.load(DIP / f, force="mesh")
        m.merge_vertices()
        out[k] = m
    extra = {}
    for k, f in (("cover_v3b", "v3/b/cad/20261020-dip_link_v3b_cover.stl"), ("pad", "v2/b/cad/20261018-dip_link_v2b_pad_ell11_5.stl")):
        m = trimesh.load(DIP / f, force="mesh")
        m.merge_vertices()
        extra[k] = m
    return out, extra


T_TABLE_FROM_M = np.array([[0.0, 0, 1.0, 0], [0, 1.0, 0, 0], [-1.0, 0, 0, 0], [0, 0, 0, 1.0]])   # palmar down, tool axis +x


def cmd_check_v2(a):
    """v2: the b links in the mount (overlap, the insertion along +x_F, gaps at each datum), the foot's screws, the homing
    socket and posts, and the mount, link and pad against each indenter at the homing and the touching poses."""
    import trimesh
    from scipy.spatial import cKDTree
    K = MeshK()
    t0 = time.time()
    sf = K.transform(from_trimesh(_load_part("standoff_fork")), np.linalg.inv(T_M_FROM_F))      # F
    ad = K.transform(from_trimesh(_load_part("adapter")), np.linalg.inv(T_M_FROM_F))
    sf_tree = cKDTree(_sample(to_trimesh(sf), 0.1))
    links, extra = b_links_v2()
    L = B_LINK
    res = {}
    for k, m in links.items():
        dm = from_trimesh(m)
        r = dict(overlap_fork_mm3=round(float((dm ^ sf).volume()), 4))
        ins = [(round(float(dx), 2), round(float((dm.translate([dx, 0.0, 0.0]) ^ sf).volume()), 4))
               for dx in np.arange(-30.0, 0.0001, 0.25)]
        r["insertion_along_x_overlaps"] = [x for x in ins if x[1] > 1e-4]
        r["insertion_clear"] = not r["insertion_along_x_overlaps"]
        pts = _sample(m, 0.1)
        dd, _ = sf_tree.query(pts)
        reg = {"tongue_back_to_back_plate": (np.abs(pts[:, 0] - L["tongue_back"]) < 0.03) & (pts[:, 2] < L["mount_z"] - 0.5),
               "tongue_sides_to_rails": (np.abs(np.abs(pts[:, 1]) - L["tongue_half"]) < 0.03) & (pts[:, 2] < L["mount_z"] - 0.5)
               & (pts[:, 0] > 4.0),
               "tongue_end_to_lip": (np.abs(pts[:, 2] - L["tongue_end"]) < 0.03) & (pts[:, 0] > 4.0),
               "bridge_top_to_inner_block": (pts[:, 1] > -16.9) & (pts[:, 1] < 13.3) & (pts[:, 2] > -63.0) & (pts[:, 2] < -60.0)
               & (np.abs(pts[:, 0]) < 11.0),
               "bridge_back_to_ear": (np.abs(pts[:, 0] - 9.45) < 0.03) & (pts[:, 2] > -70.0) & (pts[:, 2] < -62.0),
               "neck_back_to_back_plate": (np.abs(pts[:, 0] - L["neck_back"]) < 0.03) & (pts[:, 2] < -70.3) & (pts[:, 2] > L["mount_z"]),
               "horn_arm_inner_face": np.abs(pts[:, 1] - ARM_HORN[1]) < 0.03,
               "slotted_arm_inner_face": np.abs(pts[:, 1] - ARM_SLOTTED[0]) < 0.03,
               "horn_arm_outer_layer_to_column": (np.abs(pts[:, 1] - OUTER[1]) < 0.03) & (pts[:, 2] > -53.0)}
        r["gaps_by_region_mm"] = {n: (round(float(dd[s_].min()), 3) if s_.any() else None) for n, s_ in reg.items()}
        r["least_gap_mm"] = round(float(dd.min()), 3)
        res[k] = r
        print(k, "overlap", r["overlap_fork_mm3"], "insertion clear", r["insertion_clear"], r["gaps_by_region_mm"], flush=True)
    acc = []
    for ang in m3_angles():
        x, y = _m_xy(P["m3_r"], ang)
        col = K.cyl("z", (x, y), 1.5, zF(150.0), zF(P["ad_t"] + P["foot_t"] + 3.0))
        acc.append(dict(angle_deg=ang, overlap_mm3=round(float((col ^ sf).volume() + (col ^ from_trimesh(links["v3b"])).volume()), 4)))
    # the M3's head beside the column: 5.5 mm head, distance from its axis to the column's footprint
    heads = []
    for ang in m3_angles():
        x, y = _m_xy(P["m3_r"], ang)
        dx_ = max(abs(x) - P["fp_x"], 0.0)
        dy_ = max(P["fp_y"][0] - y, y - P["fp_y"][1], 0.0)
        heads.append(dict(angle_deg=ang, axis_to_column_mm=round(math.hypot(dx_, dy_), 2), head_r_mm=2.75))
    # the mount, the link and the pad against each indenter, at homing and touching (table frame)
    T_FM = T_M_FROM_F
    parts_M = {"standoff_fork": _load_part("standoff_fork"), "adapter": _load_part("adapter")}
    for k_, m in (("link_v3b", links["v3b"]), ("cover_v3b", extra["cover_v3b"]), ("pad", extra["pad"])):
        t = m.copy()
        t.apply_transform(T_FM)
        parts_M[k_] = t
    ctx = trimesh.creation.cylinder(radius=36.0, height=50.0, sections=96)
    ctx.apply_translation([0.0, 0.0, -25.0])
    parts_M["hex_e"] = ctx
    poses = {}
    for kind in ("sphere", "rod"):
        h = homing(kind)
        ind = from_trimesh(_load_part(f"indenter_{kind}"))
        a_M = np.array(h["contact_M"])
        base_off = np.array([0.0, 0.0, h["indenter_top_z"]]) - (T_TABLE_FROM_M[:3, :3] @ a_M)
        for pose, extra_off in (("touch", np.zeros(3)), ("homing", np.array([0.0, P["post_dy"], P["home_dz"]]))):
            T = T_TABLE_FROM_M.copy()
            T[:3, 3] = base_off + extra_off
            ov = {}
            for nm, tm in parts_M.items():
                t = tm.copy()
                t.apply_transform(T)
                v = (from_trimesh(t) ^ ind).volume() if nm != "hex_e" else (from_trimesh(t) ^ ind).volume()
                ov[nm] = round(float(v), 4)
            # the pin in the socket at homing: the socket's floor against the pin's top
            s_tab = T[:3, :3] @ np.array(h["socket_floor_M"]) + T[:3, 3]
            ov["socket_floor_minus_pin_top_mm"] = (np.round(s_tab - np.array([*h["post_xy"], h["pin_top_z"]]), 3).tolist()
                                                   if pose == "homing" else None)
            poses[f"{kind}_{pose}"] = ov
            print(kind, pose, ov, flush=True)
        poses[f"{kind}_homing_geometry"] = h
    walls = dict(back_plate_at_tongue=round(P["cr_x1"] - L["tongue_back"] - P["cr_clear_x"], 2),
                 back_plate_at_neck=round(P["cr_x1"] - L["neck_back"] - P["cr_neck_clear"], 2),
                 rail=P["cr_rail_t"], lip_thread_mm=round(P["cr_lip_t"] - (L["tongue_back"] + P["cr_clear_x"] - P["cr_screw_x"]), 2),
                 inner_block_height=round(P["hub_z0"] - P["ib_flat_z"], 2),
                 socket_to_column_side=round(P["fp_x"] - P["home_depth"], 2))
    rec = dict(rev="v2", b_links=res, hex_key_access=acc, foot_screw_heads=heads, poses_table=poses, walls_mm=walls,
               note="F frame for the links; table frame for the poses (x along the tool axis toward the tip, z up, the palmar "
                    "side down); overlaps by manifold3d booleans of the meshes (the pad's overlap at 'touch' is the contact's "
                    "tessellation); gaps by nearest neighbours between surface samples 0.1 mm apart",
               when=time.strftime("%Y-%m-%d %H:%M"))
    (OUT / f"{PFX}_check.json").write_text(json.dumps(rec, indent=1, default=float))
    print("access", acc, "heads", heads, "walls", walls, f"[{time.time() - t0:.0f} s]", flush=True)


def cmd_payload_v2(a):
    """v2: mass and centre of gravity of everything past the UR5e's flange for set_payload: the HEX-E (50 mm, 0.347 kg, CoG
    20 mm from its robot face, its data sheet), the adapter and the standoff-fork at the plastic the P1S plate lays down
    (print/20261020-p1s_pla_calibration.json: volume x fill ratio x 1.24 g/cm3), DIP link b v3 at its sliced density with
    the cover, board, pad, magnets and brass screws (v3/b/evaluation.json), and the steel screws."""
    import trimesh
    plate = json.loads((PRINT / "20261020-p1s_pla_calibration.json").read_text())
    fill = {o["id"]: o["fill_ratio"] for o in plate["objects"]}
    ev = json.loads((DIP / "v3/b/evaluation.json").read_text())["mass_g"]
    items = []
    for nm, pid in (("adapter", "U1"), ("standoff_fork", "U2")):
        tm = _load_part(nm)
        items.append((nm, float(tm.volume) * 1e-3 * MATERIAL["pla_density_g_cm3"] * fill[pid], np.asarray(tm.center_mass)))
    link = trimesh.load(DIP / "v3/b/cad/20261020-dip_link_v3b_dip_link.stl", force="mesh")
    link.apply_transform(T_M_FROM_F)
    c_link = np.asarray(link.center_mass)
    c_tip = F_to_M(np.array([-2.0, 0.0, B_LINK["z_contact"]]))
    items += [("dip_link_b_v3", ev["dip_link_at_sliced_density"], c_link),
              ("cover_board_screws", ev["cover_solid"] + ev["board"] + ev["screws_brass"], F_to_M(np.array([3.5, 0.0, -88.0]))),
              ("pad_and_magnets", 2.01 + 5 * 0.106, c_tip),
              ("m6_screws_washers", 4 * SCREWS["M6x12_ISO4762"]["mass_g"], np.array([0.0, 0.0, 6.0])),
              ("m3_screws", 4 * SCREWS["M3x12_foot"]["mass_g"] + 2.4, np.array([0.0, 0.0, 20.0])),
              ("hex_e", 347.0, np.array([0.0, 5.0, -50.0 + 20.0]))]
    m = sum(i[1] for i in items)
    cog_M = sum(i[1] * i[2] for i in items) / m
    rec = dict(items=[dict(item=n, mass_g=round(w, 1), cog_M_mm=np.round(c, 1).tolist()) for n, w, c in items],
               total_g=round(m, 1), cog_M_mm=np.round(cog_M, 1).tolist(),
               cog_from_ur_flange_mm=np.round(cog_M + np.array([0.0, 0.0, 50.0]), 1).tolist(),
               tcp_contact_from_ur_flange_mm=np.round(F_to_M(np.array([-10.55, 0.0, B_LINK["z_contact"]])) + [0, 0, 50.0], 2).tolist(),
               note="the adapter straight on the HEX-E's tool face (50 mm from the UR5e flange; a Quick Changer tool plate in "
                    "between adds 15.15 mm and 0.14 kg); M axes; the HEX-E's CoG from its data sheet",
               when=time.strftime("%Y-%m-%d %H:%M"))
    (OUT / f"{PFX}_payload.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: rec[k] for k in ("total_g", "cog_from_ur_flange_mm", "tcp_contact_from_ur_flange_mm")}), flush=True)


def _dist_m(r1, a1, r2, a2):
    a1, a2 = math.radians(a1), math.radians(a2)
    return math.hypot(r1 * math.cos(a1) - r2 * math.cos(a2), r1 * math.sin(a1) - r2 * math.sin(a2))


# ------------------------------------------------------------------------------------------------ estimates
def section_props(tm, z_M, shell=None):
    """Area (mm2), centroid (x, y) and second moments Ixx, Iyy (mm4, about the centroid) of a mesh's section at z_M, in
    the plane's (x_M, y_M). shell: count only the band within this distance of the boundary (printed perimeters)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    s = tm.section(plane_origin=[0, 0, z_M], plane_normal=[0, 0, 1])
    if s is None:
        return None
    p2, _ = s.to_2D(to_2D=np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, -z_M], [0, 0, 0, 1.0]]))
    polys = [Polygon(p.exterior.coords, [h.coords for h in p.interiors]) for p in p2.polygons_full]
    if not polys:
        return None
    g = unary_union(polys)
    if shell:
        g = g.difference(g.buffer(-shell))
    if g.is_empty:
        return None
    gs = list(g.geoms) if hasattr(g, "geoms") else [g]
    A = Sx = Sy = Ixx0 = Iyy0 = 0.0
    for poly in gs:
        rings = [np.asarray(poly.exterior.coords)] + [np.asarray(h.coords) for h in poly.interiors]
        for k_, ring in enumerate(rings):
            x, y = ring[:, 0], ring[:, 1]
            x1, y1 = np.roll(x, -1), np.roll(y, -1)
            cr = x * y1 - x1 * y
            sgn = 1.0
            a_ = 0.5 * cr.sum()
            if (k_ == 0 and a_ < 0) or (k_ > 0 and a_ > 0):
                sgn = -1.0
            A += sgn * a_
            Sx += sgn * ((x + x1) * cr).sum() / 6
            Sy += sgn * ((y + y1) * cr).sum() / 6
            Iyy0 += sgn * ((x * x + x * x1 + x1 * x1) * cr).sum() / 12        # integral of x^2 dA
            Ixx0 += sgn * ((y * y + y * y1 + y1 * y1) * cr).sum() / 12        # integral of y^2 dA
    cx, cy = Sx / A, Sy / A
    return dict(A=A, cx=cx, cy=cy, I_about_y=Iyy0 - A * cx * cx, I_about_x=Ixx0 - A * cy * cy)


def cmd_estimate(a):
    import trimesh
    t0 = time.time()
    E = MATERIAL["pla_E_MPa"]
    G_ = E / (2 * (1 + MATERIAL["pla_nu"]))
    sf = _load_part("standoff_fork")
    ad = _load_part("adapter")
    cp = _load_part("clamp_plate")
    dips = dip_links()
    out = {}
    # ---------------- deflection of the contact point: Euler-Bernoulli and shear along z_M, sections from the meshes
    extras_files = {"orig": [], "a": ["a/cad/20261018-dip_link_v2a_clamp_plate.stl", "a/cad/20261018-dip_link_v2a_board.stl"],
                    "b": ["b/cad/20261018-dip_link_v2b_cover.stl"]}
    for k, d in dips.items():
        dm = d["mesh"].copy()
        dm.apply_transform(T_M_FROM_F)
        extras = []
        for f in extras_files[k]:
            e = trimesh.load(V2 / f, force="mesh")
            e.apply_transform(T_M_FROM_F)
            extras.append(e)
        C = F_to_M(np.array([-10.55, 0.0, d["z_c"]]))
        zc = float(C[2])
        zs = np.arange(P["ad_t"] + P["foot_t"] + 0.25, zc, 0.5)
        rows = []
        for z in zs:
            parts_here = [sf, cp, dm] if z > PIP_M - 6.0 else [sf]
            comb = trimesh.util.concatenate(parts_here)
            sp = section_props(comb, z)
            if z > PIP_M - 6.0:
                sp_bond = section_props(trimesh.util.concatenate([dm] + extras), z)
                sep = [section_props(t, z) for t in [dm] + extras]
                sep = [x for x in sep if x]
                sp_sep = dict(A=sum(x["A"] for x in sep), I_about_y=sum(x["I_about_y"] for x in sep),
                              I_about_x=sum(x["I_about_x"] for x in sep)) if sep else None
            else:
                sp_bond = sp_sep = None
            sp_mount = section_props(trimesh.util.concatenate([sf, cp]), z)
            rows.append((z, sp, sp_sep, sp_bond, sp_mount))

        def integrate(sel, lo, hi):
            """Deflection (mm per N) of C along x_M and y_M from bending over z_M lo..hi (cantilever from the foot) and
            shear; sel picks the section dict from a row."""
            ux = uy = sx = sy = 0.0
            for row in rows:
                z = row[0]
                if not (lo <= z < hi):
                    continue
                sp = sel(row)
                if sp is None or sp["A"] <= 0:
                    return None
                arm = zc - z
                ux += arm * arm / (E * sp["I_about_y"]) * 0.5         # load along x_M bends about y_M
                uy += arm * arm / (E * sp["I_about_x"]) * 0.5
                sx += 0.5 / (G_ * 0.83 * sp["A"])
            return dict(bend_x=ux, bend_y=uy, shear=sx)
        z_end_mount = PIP_M + 14.0              # the fork's lowest clamp face (ears, key) in z_M
        mount = integrate(lambda r_: r_[1] if r_[0] < z_end_mount else r_[1], P["ad_t"] + P["foot_t"], z_end_mount)
        mount_only = integrate(lambda r_: r_[4], P["ad_t"] + P["foot_t"], PIP_M)
        dip_sep = integrate(lambda r_: r_[2], z_end_mount, zc)
        dip_bond = integrate(lambda r_: r_[3], z_end_mount, zc)
        Fn, Ft = 10.0, 5.0

        def comp(rr):
            if rr is None:
                return None
            return dict(normal_10N_mm=round(Fn * (rr["bend_x"] + rr["shear"]), 4),
                        tangential_5N_y_mm=round(Ft * (rr["bend_y"] + rr["shear"]), 4))
        # HEX-E: deformation at nominal capacity, linear; force at its tool plane plus the moment over the lever
        L_hex = zc + 15.15                           # HEX-E QC tool plane to the contact point, mm
        Cf, Cfz = 1.7 / 200.0, 0.3 / 200.0           # mm/N
        Ct = math.radians(2.5) / 10000.0             # rad per N mm
        hex_n = Fn * (Cf + Ct * L_hex ** 2)
        hex_t = Ft * (Cf + Ct * L_hex ** 2)
        hex_z = Ft * Cfz + Ft * Ct * L_hex * 10.55
        out[k] = dict(contact_point_M=np.round(C, 3).tolist(), lever_flange_to_contact_mm=round(zc, 2),
                      mount_to_clamp=comp(mount), mount_alone_to_pip=comp(mount_only),
                      dip_link_parts_separate=comp(dip_sep), dip_link_parts_bonded=comp(dip_bond),
                      dip_link_parts=["DIP link"] + [Path(f).stem for f in extras_files[k]],
                      hex_e=dict(lever_mm=round(L_hex, 2), normal_10N_mm=round(hex_n, 3), tangential_5N_y_mm=round(hex_t, 3),
                                 tangential_5N_z_mm=round(hex_z, 4),
                                 per_newton_lateral_mm=round(Cf + Ct * L_hex ** 2, 4)))
        print(k, json.dumps(out[k]), flush=True)
    # ---------------- masses, CoG, payload
    rho = MATERIAL["pla_density_g_cm3"] / 1000.0          # g per mm3
    mass = {}
    cog = {}
    for nm, tm in (("adapter", ad), ("standoff_fork", sf), ("clamp_plate", cp)):
        mass[nm] = round(tm.volume * rho, 1)
        cog[nm] = np.round(tm.center_mass, 2).tolist()
    screws_g = sum(v["qty"] * v["mass_g"] for v in SCREWS.values())
    dlink = dict(a=dict(dip_g=11.29, tip_g=3.0), b=dict(dip_g=10.86, tip_g=2.5))
    zc_a = float(F_to_M(np.array([-10.55, 0.0, dips["a"]["z_c"]]))[2])
    items = [("HEX-E QC", 347.0, np.array([0.0, 5.0, 20.0 - 65.15])),
             ("Quick Changer tool side", 140.0, np.array([0.0, 0.0, -15.15 / 2])),
             ("adapter", mass["adapter"], np.array(cog["adapter"])),
             ("standoff_fork", mass["standoff_fork"], np.array(cog["standoff_fork"])),
             ("clamp_plate", mass["clamp_plate"], np.array(cog["clamp_plate"])),
             ("screws and washers", screws_g, np.array([0.0, 0.0, 20.0])),
             ("DIP link v2 a with board, clamp plate, screws, 11/22 tip", dlink["a"]["dip_g"] + dlink["a"]["tip_g"],
              np.array([0.0, 3.85, (PIP_M + zc_a) / 2]))]
    tot = sum(m for _, m, _ in items)
    c_uf = sum(m * (c + np.array([0, 0, 65.15])) for _, m, c in items) / tot      # from the UR5e flange
    payload = dict(items=[dict(item=n, mass_g=round(m, 1), cog_M_mm=np.round(c, 1).tolist()) for n, m, c in items],
                   total_kg=round(tot / 1000, 3), cog_from_ur_flange_mm=np.round(c_uf, 1).tolist(),
                   note="UR5e flange frame along the tool axis; x, y in the mount frame M; HEX-E CoG from OnRobot's table "
                        "(z 20 mm above the UR flange, y 5 mm); the Quick Changer tool side as a disc's centroid")
    # ---------------- wrist field at the chips: dipoles along the tool axis
    mu0_4pi = 1e-7
    d_wrist = 60.0          # wrist 3's motor and brake behind the UR5e flange face (d6 = 99.6 mm in UR's kinematics)
    field = {}
    for k, d in dips.items():
        zs = [float(F_to_M(np.array([0.0, u, z]))[2]) for _, u, z in d["chips_F"]]
        r_near, r_far = (min(zs) + 65.15 + d_wrist) / 1000.0, (max(zs) + 65.15 + d_wrist) / 1000.0
        rows = {}
        for src, (mlo, mhi) in (("brake_solenoid", (0.01, 0.06)), ("rotor_residual", (0.005, 0.03))):
            Bn = [2 * mu0_4pi * m / r_near ** 3 * 1e6 for m in (mlo, mhi)]
            Bf = [2 * mu0_4pi * m / r_far ** 3 * 1e6 for m in (mlo, mhi)]
            rows[src] = dict(moment_A_m2=[mlo, mhi], B_nearest_chip_uT=[round(v, 2) for v in Bn],
                             spread_across_chips_uT=[round(n - f, 3) for n, f in zip(Bn, Bf)])
        # without the HEX-E: the mount on the UR flange
        r0 = (min(zs) + d_wrist) / 1000.0
        rows["mount_on_the_ur_flange_brake_high"] = round(2 * mu0_4pi * 0.06 / r0 ** 3 * 1e6, 2)
        field[k] = dict(r_nearest_chip_mm=round(r_near * 1000, 1), r_farthest_chip_mm=round(r_far * 1000, 1), sources=rows)
    field_note = ("Point dipoles on the tool axis, on-axis field B = mu0 m / (2 pi r^3). Brake: a solenoid that holds the "
                  "wrist's brake released while the arm is powered, 50-120 ampere-turns over about 1 cm2 with its iron 2-5 "
                  "times the air-core moment: 0.01-0.06 A m2. Rotor: the residual dipole of a multipole magnet ring, 2-5 % of "
                  "its magnets' total moment: 0.005-0.03 A m2; it turns with the motor. Distances: wrist 3's motor and brake "
                  "60 mm behind the UR5e flange face (d6 = 99.6 mm), HEX-E QC and tool plate 65.15 mm, the chips' z_M.")
    rec = dict(deflection=out, payload=payload, masses_g=mass, cog_M=cog, field=field, field_model=field_note,
               material=MATERIAL, loads="10 N normal at the contact point along -x_M (into the tip), 5 N tangential along "
               "y_M (across the finger) or z_M (along it)", when=time.strftime("%Y-%m-%d %H:%M"))
    (OUT / f"{PFX}_estimate.json").write_text(json.dumps(rec, indent=1, default=float))
    print("payload", payload["total_kg"], "kg, CoG", payload["cog_from_ur_flange_mm"], "| masses", mass,
          f"[{time.time() - t0:.0f} s]", flush=True)


# ------------------------------------------------------------------------------------------------ figures
COLORS = {"adapter": "0.80 0.62 0.36 1", "standoff_fork": "0.36 0.56 0.85 1", "clamp_plate": "0.25 0.45 0.75 1",
          "dip_link": "0.90 0.90 0.86 1", "dip_clamp": "0.62 0.70 0.82 1", "board": "0.12 0.48 0.24 1",
          "qwiic": "0.95 0.95 0.92 1", "fasteners": "0.80 0.65 0.25 1", "tip": "0.92 0.78 0.55 1",
          "screws": "0.55 0.55 0.58 1", "brass": "0.80 0.65 0.25 1", "hex": "0.72 0.73 0.76 1", "qc": "0.40 0.42 0.46 1",
          "ur": "0.30 0.52 0.78 1", "ur_grey": "0.86 0.86 0.86 1", "indenter": "0.82 0.82 0.80 1"}
W_IMG, H_IMG, FOVY = 1600, 1200, 26.0
T_W_FROM_M = np.diag([1.0, -1.0, -1.0, 1.0])           # display frame: the finger points down


def _mesh_file(tm, stem):
    import trimesh
    dst = WORK / "render" / f"{stem}.obj"
    dst.parent.mkdir(parents=True, exist_ok=True)
    sm = trimesh.graph.smooth_shade(tm, angle=np.radians(30.0), facet_minarea=None)
    dst.write_text(trimesh.exchange.obj.export_obj(sm, include_normals=True, include_texture=False))
    return str(dst)


def _mjcf(items, w, h):
    assets, geoms = [], []
    for k, (path, rgba) in enumerate(items):
        assets.append(f'<mesh name="m{k}" file="{path}" scale="0.001 0.001 0.001"/>')
        geoms.append(f'<geom type="mesh" mesh="m{k}" rgba="{rgba}" contype="0" conaffinity="0"/>')
    return f"""<mujoco><compiler inertiafromgeom="false"/>
<visual><global offwidth="{w}" offheight="{h}" fovy="{FOVY}"/><quality shadowsize="4096" offsamples="8"/>
<headlight ambient="0.42 0.42 0.42" diffuse="0.48 0.48 0.48" specular="0.05 0.05 0.05"/></visual>
<asset><texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/>{''.join(assets)}</asset>
<worldbody><light pos="0.15 -0.2 0.3" dir="-0.4 0.6 -0.6" diffuse="0.40 0.40 0.40" castshadow="false"/>
<light pos="0.2 0.2 -0.1" dir="-0.5 -0.5 0.2" diffuse="0.22 0.22 0.22" castshadow="false"/>
{''.join(geoms)}</worldbody></mujoco>"""


def _render(meshes, cam, dst, w=W_IMG, h=H_IMG):
    """meshes: list of (trimesh in W, rgba, stem). cam: (lookat in W mm, distance mm, azimuth, elevation)."""
    import mujoco
    from PIL import Image
    items = [(_mesh_file(tm, stem), rgba) for tm, rgba, stem in meshes]
    m = mujoco.MjModel.from_xml_string(_mjcf(items, w, h))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    c = mujoco.MjvCamera()
    lookat, distance, azimuth, elevation = cam
    c.lookat[:] = np.asarray(lookat) * 1e-3
    c.distance, c.azimuth, c.elevation = distance * 1e-3, azimuth, elevation
    with mujoco.Renderer(m, h, w) as ren:
        ren.update_scene(d, camera=c)
        img = ren.render()
    Image.fromarray(img).save(dst, optimize=True)
    return dst


def _project(Pw, cam, w=W_IMG, h=H_IMG):
    lookat, dist, az, el = cam
    a, e = math.radians(az), math.radians(el)
    fwd = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    up = np.array([-math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), math.cos(e)])
    right = np.cross(fwd, up)
    eye = np.asarray(lookat, float) - dist * fwd
    f = (h / 2) / math.tan(math.radians(FOVY) / 2)
    Q = np.atleast_2d(Pw) - eye
    zc = Q @ fwd
    return np.column_stack([w / 2 + f * (Q @ right) / zc, h / 2 - f * (Q @ up) / zc])


def _annotate(path, cam, labels, title=None, callout=True):
    """Numbered callouts at points (in W, mm) with a legend under the image; the image cropped to its content."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    img = Image.open(path).convert("RGB")
    W0, H0 = img.size
    a_ = np.asarray(img)
    rows = np.where((a_ < 245).any(axis=(1, 2)))[0]
    top = max(0, int(rows.min()) - 70) if len(rows) else 0
    cut = int(min(a_.shape[0], rows.max() + 30)) if len(rows) else a_.shape[0]
    img = img.crop((0, top, a_.shape[1], cut))
    W, H = img.size
    line = 44.0
    extra = (len(labels) + 0.5) * line if (callout and labels) else 0.0
    fig = plt.figure(figsize=(W / 150, (H + extra) / 150), dpi=150)
    ax = fig.add_axes([0, extra / (H + extra), 1, H / (H + extra)])
    ax.imshow(img)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    box = dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85)
    for k, (text, pw) in enumerate(labels, 1):
        u, v = _project(np.array([pw]), cam, W0, H0)[0]
        v -= top
        ax.plot([u], [v], "o", ms=25, mfc="white", mec="k", mew=1.5)
        ax.text(u, v, str(k), ha="center", va="center", fontsize=16, family="serif", weight="bold")
    if labels:
        lg = fig.add_axes([0, 0, 1, extra / (H + extra)])
        lg.set_xlim(0, W)
        lg.set_ylim(extra, 0)
        lg.axis("off")
        for k, (text, _) in enumerate(labels, 1):
            lg.text(0.03 * W, (k - 0.35) * line, f"{k}   {text}", fontsize=16, family="serif", va="center")
    if title:
        ax.text(0.02 * W, 0.03 * H, title, fontsize=18, family="serif", va="top", bbox=box)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _screw(axis_dir, head_c, length, d, head_d, head_k, washer=None):
    """A socket head screw as a mesh in M: head centred at head_c on the axis, the shank along axis_dir (unit)."""
    import trimesh
    parts = []
    u = np.asarray(axis_dir, float) / np.linalg.norm(axis_dir)

    def cyl(r, a0, a1):
        c = trimesh.creation.cylinder(radius=r, height=a1 - a0, sections=48)
        z = np.array([0, 0, 1.0])
        R = trimesh.geometry.align_vectors(z, u)
        c.apply_transform(R)
        c.apply_translation(np.asarray(head_c) + u * (a0 + a1) / 2)
        return c
    parts.append(cyl(head_d / 2, -head_k, 0.0))
    parts.append(cyl(d / 2 * 0.96, 0.0, length))
    if washer:
        parts.append(cyl(washer[0] / 2, 0.0, washer[1]))
    return trimesh.util.concatenate(parts)


def _context_M():
    """The HEX-E QC, the quick changer's tool plate and the UR5e's flange and wrist housing as plain cylinders in M (context
    for the figures; main dimensions from the data sheets)."""
    import trimesh

    def cyl(r, z0, z1, x=0.0, y=0.0):
        c = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=96)
        c.apply_translation([x, y, (z0 + z1) / 2])
        return c
    qc = cyl(31.5, -15.15, 0.0)
    hexe = trimesh.util.concatenate([cyl(36.0, -65.15, -15.15),
                                     trimesh.creation.box(extents=[18, 16, 22], transform=trimesh.transformations.translation_matrix([-40.0, 0.0, -35.0]))])
    ur = trimesh.util.concatenate([cyl(31.5, -65.15 - 41.7, -65.15), cyl(45.0, -65.15 - 100.0, -65.15 - 41.7)])
    return dict(qc=qc, hexe=hexe, ur=ur)


def _dip_meshes_M(k="a"):
    import trimesh
    out = {}
    if k == "a":
        files = dict(dip_link="a/cad/20261018-dip_link_v2a_dip_link.stl", dip_clamp="a/cad/20261018-dip_link_v2a_clamp_plate.stl",
                     board="a/cad/20261018-dip_link_v2a_board.stl", qwiic="a/cad/20261018-dip_link_v2a_qwiic.stl",
                     fasteners="a/cad/20261018-dip_link_v2a_fasteners.stl", tip="a/cad/20261018-dip_link_v2a_tip_ell11_5_F.stl")
    else:
        files = dict(dip_link="b/cad/20261018-dip_link_v2b_dip_link.stl", dip_clamp="b/cad/20261018-dip_link_v2b_cover.stl",
                     tip="b/cad/20261018-dip_link_v2b_pad_ell11_5.stl")
    for nm, f in files.items():
        m = trimesh.load(V2 / f, force="mesh")
        m.apply_transform(T_M_FROM_F)
        out[nm] = m
    return out


def _screws_M(shift_y=0.0):
    """The mount's screws in M: 4 M6 in the adapter, 4 M3 in the foot, 2 M3 in the clamp plate, 1 M3 in the slotted arm."""
    import trimesh
    out = []
    for a in bolt_angles():
        x, y = _m_xy(P["bolt_r"], a)
        c = F_to_M(np.array([x, y, zF(P["m6_under"] + 1.6)]))
        out.append(_screw((0, 0, -1), c, 12.0, 6.0, 10.0, 6.0, washer=(12.0, 1.6)))
    for a in m3_angles():
        x, y = _m_xy(P["m3_r"], a)
        c = F_to_M(np.array([x, y, zF(P["ad_t"] + P["foot_t"])]))
        out.append(_screw((0, 0, -1), c, 12.0, 3.0, 5.5, 3.0))
    for sx in (-1, 1):
        c = F_to_M(np.array([sx * P["cp_screw_x"], ARM_HORN[0] - P["cp_t"] + P["cp_cb_t"] + shift_y, P["cp_screw_z"]]))
        out.append(_screw((0, 1, 0), c, 16.0, 3.0, 5.5, 3.0))
    c = F_to_M(np.array([0.0, ARM_SLOTTED[1] + 0.8 + shift_y, PIP_ZF]))
    out.append(_screw((0, -1, 0), c, 12.0, 3.0, 5.5, 3.0, washer=None))
    w = trimesh.creation.cylinder(radius=4.5, height=0.8, sections=48)
    w.apply_transform(trimesh.geometry.align_vectors([0, 0, 1.0], [0, 1.0, 0]))
    w.apply_translation(F_to_M(np.array([0.0, ARM_SLOTTED[1] + 0.4 + shift_y, PIP_ZF])))
    out.append(w)
    return trimesh.util.concatenate(out)


def cmd_render(a):
    import trimesh
    MEDIA.mkdir(parents=True, exist_ok=True)
    C_ = COLORS
    parts = {nm: trimesh.load(fname(nm), force="mesh") for nm in PARTS}
    dipa = _dip_meshes_M("a")
    ctx = _context_M()
    scr = _screws_M()

    def W(tm):
        t = tm.copy()
        t.apply_transform(T_W_FROM_M)
        return t

    def pw(pM):
        return (T_W_FROM_M[:3, :3] @ np.asarray(pM, float))

    def pf(pF):
        return pw(F_to_M(np.asarray(pF, float)))
    asm = [(W(parts["adapter"]), C_["adapter"], "adapter"), (W(parts["standoff_fork"]), C_["standoff_fork"], "standoff_fork"),
           (W(parts["clamp_plate"]), C_["clamp_plate"], "clamp_plate"), (W(scr), C_["screws"], "screws")]
    dip = [(W(dipa["dip_link"]), C_["dip_link"], "dip_link_a"), (W(dipa["dip_clamp"]), C_["dip_clamp"], "dip_clamp_a"),
           (W(dipa["board"]), C_["board"], "board_a"), (W(dipa["qwiic"]), C_["qwiic"], "qwiic_a"),
           (W(dipa["fasteners"]), C_["fasteners"], "fasteners_a"), (W(dipa["tip"]), C_["tip"], "tip_a")]
    cx = [(W(ctx["qc"]), C_["qc"], "ctx_qc"), (W(ctx["hexe"]), C_["hex"], "ctx_hexe"), (W(ctx["ur"]), C_["ur_grey"], "ctx_ur")]
    # 1. the mount with the DIP link on the HEX-E and the UR5e flange, from the dorsal and horn side
    vis = lambda r_, zm: pw([-0.7071 * r_, -0.7071 * r_, zm])          # noqa: E731  (a cylinder's side facing the camera)
    cam = (pw([0.0, 0.0, 10.0]), 470, 315, -12)
    img = _render(cx + asm + dip, cam, MEDIA / f"{PFX}_on_the_hex_e.png")
    _annotate(img, cam, [
        ("UR5e tool flange and wrist 3 (context)", vis(31.5, -75.0)),
        ("OnRobot HEX-E QC, 50 mm (context)", vis(36.0, -40.0)),
        ("Quick Changer tool plate, 15.15 mm (context)", vis(31.5, -7.5)),
        ("Adapter disc: spigot, dowel, four M6 x 12", vis(31.5, 5.5)),
        ("Standoff-fork: foot with four M3 x 12, column", pw([-17.0, -12.0, 30.0])),
        ("Cable groove and zip-tie pockets", pf([17.0, 0.0, -34.0])),
        ("Clamp plate on the horn-side arm", pf([-8.0, -31.25, -60.0])),
        ("DIP link v2 a with the board and the 11/22 tip", pf([6.0, -7.0, -84.0]))],
        title="Calibration mount on the HEX-E QC, dorsal side, finger down")
    # 2. close-up of the fork with the DIP link, from the dorsal and horn side
    cam = (pf([0.0, -4.0, -62.0]), 200, 315, -10)
    img = _render(asm + dip, cam, MEDIA / f"{PFX}_fork_closeup.png")
    _annotate(img, cam, [
        ("Clamp plate: two M3 x 16 into the ears", pf([-14.05, -31.25, -60.0])),
        ("Horn-side arm between the clamp plate and the inner block", pf([-5.0, -23.25, -64.0])),
        ("Cable slot and groove, open on the dorsal side", pf([17.0, 0.0, -47.0])),
        ("Zip-tie pockets beside the groove", pf([17.0, -6.6, -40.55])),
        ("J1 and the Qwiic plug in the DIP link's slot", pf([7.3, 0.0, -68.5])),
        ("Board under a's clamp plate", pf([10.9, -4.0, -86.0]))],
        title="The fork holding the DIP link, dorsal and horn side")
    # 3. the standoff-fork alone, tool axis up, from above its fork end (dorsal and horn side)
    sfm = parts["standoff_fork"]
    cam = (F_to_M(np.array([0.0, -4.0, -58.0])), 165, 45, -42)
    img = _render([(sfm, C_["standoff_fork"], "standoff_fork_alone")], cam, MEDIA / f"{PFX}_fork_alone.png")
    pm = lambda pF: F_to_M(np.asarray(pF, float))                     # noqa: E731
    _annotate(img, cam, [
        ("Key for the horn slot: 1.6 mm, round end R 2.4", pm([0.0, -20.2, -67.5])),
        ("Hub: ends 0.3 mm short of the slot walls", pm([-6.0, -21.0, -53.2])),
        ("Inner block: fills the 30.45 mm between the arms, R 12.5 about the PIP axis", pm([-8.0, 0.0, -58.0])),
        ("Ears with the clamp-screw holes (M3, 2.75 mm)", pm([14.05, -21.0, -60.0])),
        ("Cable slot, open on the dorsal side", pm([9.0, 0.0, -60.0])),
        ("Cable groove down the column", pm([17.0, 0.0, -35.0]))],
        title="Standoff-fork without the DIP link, its fork end up")
    # 4. exploded along the tool axis and across the arm
    off = {"adapter": -25.0, "standoff_fork": 0.0, "dip": 30.0}
    exp = [(W(parts["adapter"].copy().apply_translation([0, 0, off["adapter"]])), C_["adapter"], "ex_adapter"),
           (W(parts["standoff_fork"]), C_["standoff_fork"], "ex_sf"),
           (W(parts["clamp_plate"].copy().apply_translation([0, -22.0, 0])), C_["clamp_plate"], "ex_cp")]
    for nm, (tm, rgba, stem) in zip(["dip_link", "dip_clamp", "board", "qwiic", "fasteners", "tip"], dip):
        t = dipa[nm].copy()
        t.apply_translation([0, 0, off["dip"]])
        exp.append((W(t), rgba, "ex_" + stem))
    cam = (pw([0.0, -6.0, 30.0]), 430, 315, -14)
    img = _render(exp, cam, MEDIA / f"{PFX}_exploded.png")
    _annotate(img, cam, [
        ("Adapter: spigot and dowel up, M6 x 12 into the flange first", pw([-22.3, -22.3, -5.0 + off["adapter"]])),
        ("Standoff-fork: four M3 x 12 into the adapter", pw([-17.0, -12.0, 25.0])),
        ("DIP link: slides on along the tool axis, the key into the horn slot", pf([-4.0, -23.25, -60.0 - off["dip"]])),
        ("Clamp plate", pw(F_to_M(np.array([-8.0, -31.25, -60.0])) + np.array([0, -22.0, 0])))],
        title="Exploded: adapter, standoff-fork, DIP link, clamp plate")
    # 5. indenters
    ind = []
    for k, nm in enumerate(TABLE_PARTS):
        t = trimesh.load(fname(nm), force="mesh")
        t.apply_translation([0.0, -30.0 + 60.0 * k, 0.0])
        ind.append((t, C_["indenter"], nm))
    cam = ((0.0, 0.0, 22.0), 260, 300, -25)
    img = _render(ind, cam, MEDIA / f"{PFX}_indenters.png")
    _annotate(img, cam, [("Sphere R 5 mm, top 46 mm above the table", (0.0, -30.0, 46.0)),
                         ("Half rod R 12.5 mm (the bench screwdriver's), top 42.5 mm", (0.0, 30.0, 42.5)),
                         ("Slots for M6 on a 25 mm grid, 50 mm apart", (25.0, -30.0, 6.0))],
                title="Table indenters (optional)")
    print("renders ->", rel(MEDIA), flush=True)


def _screws_M_v2():
    """v2's screws in M: four M6 in the adapter, four M3 in the foot, the M3 with its washer through the slotted arm, the
    M3 pusher in the cradle's lip."""
    import trimesh
    out = []
    for a_ in bolt_angles():
        x, y = _m_xy(P["bolt_r"], a_)
        out.append(_screw((0, 0, -1), F_to_M(np.array([x, y, zF(P["m6_under"] + 1.6)])), 12.0, 6.0, 10.0, 6.0, washer=(12.0, 1.6)))
    for a_ in m3_angles():
        x, y = _m_xy(P["m3_r"], a_)
        out.append(_screw((0, 0, -1), F_to_M(np.array([x, y, zF(P["ad_t"] + P["foot_t"])])), 12.0, 3.0, 5.5, 3.0))
    out.append(_screw((0, -1, 0), F_to_M(np.array([0.0, ARM_SLOTTED[1] + 0.8, PIP_ZF])), 12.0, 3.0, 5.5, 3.0))
    w = trimesh.creation.cylinder(radius=4.5, height=0.8, sections=48)
    w.apply_transform(trimesh.geometry.align_vectors([0, 0, 1.0], [0, 1.0, 0]))
    w.apply_translation(F_to_M(np.array([0.0, ARM_SLOTTED[1] + 0.4, PIP_ZF])))
    out.append(w)
    zb = B_LINK["tongue_end"] - P["cr_lip_gap"] - P["cr_lip_t"]
    out.append(_screw((0, 0, -1), F_to_M(np.array([P["cr_screw_x"], 0.0, zb - 1.2])), 8.0, 3.0, 5.5, 3.0))
    return trimesh.util.concatenate(out)


def cmd_render_v2(a):
    """v2 figures: the mount with DIP link b v3 on the HEX-E from the palmar and the dorsal side, the standoff-fork alone,
    the link going in along x, and the homing and touching poses over the rod indenter."""
    import trimesh
    MEDIA.mkdir(parents=True, exist_ok=True)
    C_ = dict(COLORS, cover="0.36 0.52 0.86 1", cable="0.20 0.30 0.85 1", post="0.75 0.55 0.30 1")
    parts = {nm: trimesh.load(fname(nm), force="mesh") for nm in PARTS}
    links, extra = b_links_v2()
    dip = {}
    for k_, m in (("dip_link", links["v3b"]), ("cover", extra["cover_v3b"]), ("tip", extra["pad"]),
                  ("cable", trimesh.load(DIP / "v3/b/cad/20261020-dip_link_v3b_cable.stl", force="mesh"))):
        t = m.copy()
        t.apply_transform(T_M_FROM_F)
        dip[k_] = t
    hexe = trimesh.creation.cylinder(radius=36.0, height=50.0, sections=96)
    hexe.apply_translation([0.0, 0.0, -25.0])
    ur = trimesh.util.concatenate([trimesh.creation.cylinder(radius=31.5, height=41.7, sections=96),
                                   trimesh.creation.cylinder(radius=45.0, height=58.3, sections=96)])
    ur.vertices[:len(ur.vertices) // 2, 2] -= 50.0 + 20.85
    ur.vertices[len(ur.vertices) // 2:, 2] -= 50.0 + 41.7 + 29.15
    scr = _screws_M_v2()

    def W(tm, T=T_W_FROM_M):
        t = tm.copy()
        t.apply_transform(T)
        return t

    def pw(pM):
        return T_W_FROM_M[:3, :3] @ np.asarray(pM, float)

    def pf(pF):
        return pw(F_to_M(np.asarray(pF, float)))
    asm = [(W(parts["adapter"]), C_["adapter"], "v2_adapter"), (W(parts["standoff_fork"]), C_["standoff_fork"], "v2_fork"),
           (W(scr), C_["screws"], "v2_screws")]
    dl = [(W(dip["dip_link"]), C_["dip_link"], "v2_dip"), (W(dip["cover"]), C_["cover"], "v2_cover"),
          (W(dip["tip"]), C_["tip"], "v2_tip"), (W(dip["cable"]), C_["cable"], "v2_cable")]
    cx = [(W(hexe), C_["hex"], "v2_hexe"), (W(ur), C_["ur_grey"], "v2_ur")]
    # 1. on the HEX-E, palmar side
    cam = (pw([0.0, 0.0, 30.0]), 430, 160, -10)
    img = _render(cx + asm + dl, cam, MEDIA / f"{PFX}_on_the_hex_e_palmar.png")
    _annotate(img, cam, [
        ("HEX-E between the UR5e flange and the adapter (context)", pw([36.0 * 0.94, -36.0 * 0.34, -25.0])),
        ("Adapter: dowel only, four M6 x 12", pw([31.5 * 0.94, -31.5 * 0.34, 5.5])),
        ("Homing socket, 6.3 mm, in the column's palmar face", pw([P["fp_x"], 0.0, P["home_zm"]])),
        ("DIP link b v3 in the cradle; the column under the pad", pf([-8.0, 0.0, -72.0])),
        ("85A pad on the cover with its side walls", pf([-10.55, 0.0, B_LINK["z_contact"]]))],
        title="Mount v2 on the HEX-E with DIP link b v3, palmar side, finger down")
    # 2. dorsal side
    cam = (pw([0.0, 0.0, 55.0]), 330, 315, -12)
    img = _render(asm + dl, cam, MEDIA / f"{PFX}_dorsal.png")
    _annotate(img, cam, [
        ("Cradle: back plate on the tongue's back face (the datum)", pf([15.0, 0.0, -88.0])),
        ("Rails beside the tongue, 0.2 mm", pf([12.0, -11.95, -84.0])),
        ("Lip with the M3 pusher under the tongue's end", pf([10.0, 0.0, -103.0])),
        ("Horn-side arm beside the column: no key in its slot", pf([0.0, -23.25, -60.0])),
        ("Cable slot and groove on the column's dorsal face", pf([17.0, 0.0, -40.0]))],
        title="Mount v2 with DIP link b v3, dorsal side")
    # 3. the standoff-fork alone, its fork end up, from the palmar side
    sfm = parts["standoff_fork"]
    cam = (F_to_M(np.array([0.0, -2.0, -78.0])), 230, 200, -38)
    pm = lambda pF: F_to_M(np.asarray(pF, float))                     # noqa: E731
    img = _render([(sfm, C_["standoff_fork"], "v2_fork_alone")], cam, MEDIA / f"{PFX}_fork_alone.png")
    _annotate(img, cam, [
        ("Inner block with a flat bottom: the bridge bears on it (z datum)", pm([-6.0, -2.0, P["ib_flat_z"]])),
        ("Back plate of the cradle (x datum on the tongue)", pm([B_LINK["tongue_back"] + 0.15, 0.0, -90.0])),
        ("Rails, 0.2 mm beside the tongue", pm([4.0, 12.0, -88.0])),
        ("Lip and the M3 pusher's hole", pm([5.0, 0.0, -101.0])),
        ("Homing socket", pm([-P["fp_x"], 0.0, zF(P["home_zm"])]))],
        title="Standoff-fork v2 alone, its fork end up: no key, no palmar ear")
    # 4. the link going in along x (dorsally), 16 mm short of its place
    shift = np.eye(4)
    shift[0, 3] = 16.0                                                  # +x_M = toward the palmar side
    dl2 = [(W(_shifted(t, shift)), c, st + "_in") for t, c, st in
           [(dip["dip_link"], C_["dip_link"], "v2_dip"), (dip["cover"], C_["cover"], "v2_cover"), (dip["tip"], C_["tip"], "v2_tip")]]
    cam = (pw([10.0, 0.0, 72.0]), 260, 75, -6)
    img = _render(asm + dl2, cam, MEDIA / f"{PFX}_insertion.png")
    _annotate(img, cam, [("DIP link b, 16 mm short of its place", pw(F_to_M(np.array([-16.0 - 10.55, 0.0, -88.0])))),
                         ("Inner block between the clevis arms", pf([0.0, 0.0, -57.0])),
                         ("Cradle, open toward the palmar side", pf([12.0, 0.0, -95.0]))],
              title="The DIP link goes into the mount dorsally (along -x of the mount)")
    # 5. homing and touching over the rod indenter (table frame: x along the tool axis, z up)
    h = homing("rod")
    ind = trimesh.load(fname("indenter_rod"), force="mesh")
    for pose, off in (("homing", np.array([0.0, P["post_dy"], P["home_dz"]])), ("touch", np.zeros(3))):
        T = T_TABLE_FROM_M.copy()
        T[:3, 3] = np.array([0.0, 0.0, h["indenter_top_z"]]) - T_TABLE_FROM_M[:3, :3] @ np.array(h["contact_M"]) + off
        sc = [(W(t, T), c, f"v2_{pose}_{st}") for t, c, st in
              [(parts["standoff_fork"], C_["standoff_fork"], "fork"), (parts["adapter"], C_["adapter"], "adapter"),
               (hexe, C_["hex"], "hexe"), (dip["dip_link"], C_["dip_link"], "dip"), (dip["cover"], C_["cover"], "cover"),
               (dip["tip"], C_["tip"], "tip")]] + [(ind, C_["indenter"], "v2_ind_rod")]
        cam = (np.array([-30.0, 15.0, 45.0]), 330, 240, 12)
        img = _render(sc, cam, MEDIA / f"{PFX}_{pose}.png")
        lab = ([("Homing: the post's pin bottoms in the socket", np.array([*h["post_xy"], h["pin_top_z"]])),
                ("The pad's contact point 30 mm to the side and 5 mm above the rod", np.array(h["contact_at_homing_table"]))]
               if pose == "homing" else [("Touching: from homing, move -30 mm in y and -5 mm in z", np.array([0.0, 0.0, 42.5]))])
        _annotate(img, cam, lab, title=f"Rod indenter with its homing post: {pose}")
    print("renders ->", rel(MEDIA), flush=True)


def _shifted(t, T):
    u = t.copy()
    u.apply_transform(T)
    return u


def cmd_sections(a):
    """Sections through the clevis (femtip: trimesh sections and shapely): x_F = 0 (the y-z plane through the PIP axis and
    the key) and y_F = -20.2 (the horn slot's mid-plane), with the DIP links a and b and the mount."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly
    import trimesh
    MEDIA.mkdir(parents=True, exist_ok=True)
    sf = _load_part("standoff_fork")
    cp = _load_part("clamp_plate")
    for t in (sf, cp):
        t.apply_transform(np.linalg.inv(T_M_FROM_F))
    dips = dip_links()
    plt.rcParams.update({"font.family": "serif", "font.size": 14})
    fig, axs = plt.subplots(1, 2, figsize=(14.0, 7.6), dpi=130, gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    cols = {"mount": "#5b8fd9", "clamp": "#3f6fb0", "dip": "#d8d6cc", "dip_b": "#b9b39a"}

    def draw(ax, tm, origin, normal, uv, color, alpha=1.0, ec="k", lw=0.8, z=1):
        s = tm.section(plane_origin=origin, plane_normal=normal)
        if s is None:
            return
        for e in s.discrete:
            P2 = e[:, uv]
            ax.add_patch(MPoly(P2, closed=True, fc=color, ec=ec, lw=lw, alpha=alpha, zorder=z))
    # x = 0 plane: (y, z)
    ax = axs[0]
    draw(ax, dips["a"]["mesh"].copy().apply_translation([0, 0.1, 0.1]), [0, 0, 0], [1, 0, 0], [1, 2], cols["dip"], 0.9, z=1)
    draw(ax, sf, [0, 0, 0], [1, 0, 0], [1, 2], cols["mount"], 0.85, z=2)
    draw(ax, cp, [0, 0, 0], [1, 0, 0], [1, 2], cols["clamp"], 0.85, z=2)
    ax.plot([-30, 22], [PIP_ZF, PIP_ZF], "k-.", lw=0.8)
    ax.text(21.5, PIP_ZF + 0.6, "PIP axis", ha="right", fontsize=13)
    ax.set_xlim(-33, 23)
    ax.set_ylim(-72, -40)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$y_\mathrm{F}$ (mm), along the PIP axis")
    ax.set_ylabel(r"$z_\mathrm{F}$ (mm), the tip below")
    ax.set_title(r"Section $x_\mathrm{F} = 0$: fork (blue), DIP link v2 a (grey)", fontsize=14)
    for txt, xy in (("Key in the horn slot", (-20.2, -63.0)), ("Hub", (-19.0, -50.0)), ("Inner block", (-3.0, -57.0)),
                    ("Clamp plate", (-27.2, -62.0)), ("Slotted arm", (15.4, -60.0))):
        ax.annotate(txt, xy=xy, xytext=(xy[0] + (6 if xy[0] < 10 else -2), xy[1] + (4 if txt != "Clamp plate" else -3)),
                    fontsize=12, arrowprops=dict(arrowstyle="-", lw=0.8))
    # y = -20.2: (x, z) through the slot layer
    ax = axs[1]
    for k, c_ in (("a", cols["dip"]),):
        draw(ax, dips[k]["mesh"].copy().apply_translation([0, 0.1, 0.1]), [0, -20.2, 0], [0, 1, 0], [0, 2], c_, 0.9, z=1)
    draw(ax, sf, [0, -20.2, 0], [0, 1, 0], [0, 2], cols["mount"], 0.85, z=2)
    ax.plot([-15, 15], [PIP_ZF, PIP_ZF], "k-.", lw=0.8)
    ax.set_xlim(-17.5, 17.5)
    ax.set_ylim(-72, -40)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_\mathrm{F}$ (mm), dorsal to the right")
    ax.set_title(r"Section $y_\mathrm{F} = -20.2$: the key in the horn slot", fontsize=14)
    for txt, xy, dxy in (("Slot walls of the arm", (3.6, -56.0), (5, 5)), ("Key, 0.1 mm per flank", (0.0, -60.0), (5, -6)),
                         ("Ear", (-14.0, -58.0), (-1, 9))):
        ax.annotate(txt, xy=xy, xytext=(xy[0] + dxy[0], xy[1] + dxy[1]), fontsize=12, arrowprops=dict(arrowstyle="-", lw=0.8))
    fig.tight_layout()
    fig.savefig(MEDIA / f"{PFX}_sections.png")
    plt.close(fig)
    print("sections ->", rel(MEDIA / f"{PFX}_sections.png"), flush=True)


# ------------------------------------------------------------------------------------------------ print project
PRINT_SPEC = {
    # part: (rotation rows = print axes in M (or the table frame), settings, label)
    "adapter": ([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]],
                dict(layer_height="0.15", perimeters="5", fill_density="40%", fill_pattern="gyroid", top_solid_layers="6",
                     bottom_solid_layers="6", seam_position="rear"),
                "Adapter disc, tool face on the bed (spigot and dowel up)"),
    "standoff_fork": ([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
                      dict(layer_height="0.15", perimeters="5", fill_density="40%", fill_pattern="gyroid",
                           top_solid_layers="6", bottom_solid_layers="6", seam_position="rear"),
                      "Standoff-fork, foot on the bed (fork end up)"),
    "clamp_plate": ([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]],
                    dict(layer_height="0.15", perimeters="4", fill_density="100%", fill_pattern="rectilinear",
                         seam_position="rear"),
                    "Clamp plate, the face on the arm on the bed"),
    "indenter_sphere": ([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
                        dict(layer_height="0.15", perimeters="4", fill_density="40%", fill_pattern="gyroid",
                             seam_position="rear"), "Indenter, 5 mm sphere"),
    "indenter_rod": ([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
                     dict(layer_height="0.15", perimeters="4", fill_density="40%", fill_pattern="gyroid",
                          seam_position="rear"), "Indenter, 12.5 mm half rod"),
}
FORK_SOLID_FROM_ZM = 38.0       # the fork (inner block, hub, key, ears) at 100 % infill: a modifier from this z_M up


def cmd_print(a):
    """PrusaSlicer project for the Prusa XL on the owner's v17 template (PLA, tool 3), sliced and checked: the three mount
    parts on one plate and the two indenters on a second, through dip_link_print_project's writer and slicer."""
    import trimesh
    import dip_link_print_project as PP
    PRINT.mkdir(parents=True, exist_ok=True)
    work = WORK / "print"
    plates = {f"{PFX}_mount_pla_t3": list(PARTS), f"{PFX}_indenters_pla_t3": list(TABLE_PARTS)}
    report = {}
    for name, keys in plates.items():
        objs = []
        for k in keys:
            R, st, label = PRINT_SPEC[k]
            tm = trimesh.load(fname(k), force="mesh")
            m, Rt = PP.to_print(tm, np.asarray(R))
            o = dict(key=k, name=k, label=label, short=label.split(",")[0], mesh=m, Rt=Rt, extruder=PP.PLA_TOOL,
                     settings=dict(st), volumes=[])
            if k == "standoff_fork":
                z0 = FORK_SOLID_FROM_ZM - float(np.asarray(tm.bounds)[0][2])
                lo, hi = np.asarray(m.bounds)
                o["volumes"].append(dict(name="fork_100pct", type="ParameterModifier",
                                         mesh=PP.box_mesh((lo[0] - 1, lo[1] - 1, z0), (hi[0] + 1, hi[1] + 1, hi[2] + 1)),
                                         settings=dict(fill_density="100%", fill_pattern="rectilinear")))
            objs.append(o)
        for o, t in zip(objs, PP.layout([o["mesh"] for o in objs])):
            o["translation"] = t
        proj = PRINT / f"{name}.3mf"
        cfg = PP.write_project(proj, objs)
        PP.bed_figure(objs, PRINT / f"{name}_bed.png", "Prusa XL bed: " + ("mount parts" if "mount" in name else "indenters")
                      + ", PLA on tool 3")
        rec = dict(project=rel(proj), template=rel(PP.TEMPLATE), config_set=PP.GLOBAL,
                   objects=[dict(part=o["key"], label=o["label"], settings=o["settings"],
                                 rotation_print_from_M=np.asarray(o["Rt"][0]).tolist(),
                                 bed_translation_mm=np.round(o["translation"], 2).tolist(),
                                 size_mm=np.round(o["mesh"].extents, 2).tolist(), volume_mm3=round(float(o["mesh"].volume), 1),
                                 volumes=[dict(name=v["name"], type=v["type"], settings=v.get("settings", {}),
                                               bounds_print_mm=np.round(v["mesh"].bounds, 2).tolist()) for v in o["volumes"]])
                            for o in objs])
        try:
            gc, log = PP.slice_project(proj, cfg, work / name)
            summ = PP.gcode_summary(gc)
            chk = {}
            for o in objs:
                lab = summ["labels"].get(o["name"])           # the M486 object index
                if lab is None:
                    chk[o["key"]] = dict(error="object label not in the G-code", labels=list(summ["labels"]))
                    continue
                S, T = PP.object_segments(gc, lab)
                tid = {v: k for k, v in T.items()}
                sup = [tid[k] for k in ("Support material", "Support material interface") if k in tid]
                zt = float(S[:, 0].max()) if len(S) else None
                chk[o["key"]] = dict(label=lab, layers=int(len(np.unique(S[:, 0]))), top_z=round(zt, 2) if zt else None,
                                     part_height=round(float(o["mesh"].extents[2]), 2),
                                     support_segments=int(np.isin(S[:, 5], sup).sum()) if sup else 0,
                                     types=sorted(T.values()))
            rec["slice"] = dict(gcode=rel(gc), summary={k: v for k, v in summ.items() if k != "labels"}, checks=chk,
                                passes=all(c.get("support_segments", 1) == 0 and c.get("top_z") is not None
                                           and abs(c["top_z"] - c["part_height"]) < 0.3 for c in chk.values()))
        except SystemExit as e:
            rec["slice"] = dict(error=str(e)[-800:])
        rec["when"] = time.strftime("%Y-%m-%d %H:%M")
        (PRINT / f"{name}.json").write_text(json.dumps(rec, indent=1, default=float))
        report[name] = rec.get("slice", {})
        print(name, json.dumps(rec.get("slice", {}).get("summary", rec.get("slice")))[:400],
              "passes", rec.get("slice", {}).get("passes"), flush=True)


# ------------------------------------------------------------------------------------------------ parts list, assembly, runbook
def cmd_docs(a):
    est = json.loads((OUT / f"{PFX}_estimate.json").read_text())
    step = json.loads((OUT / f"{PFX}_step.json").read_text())["parts"]
    pr = {n: json.loads((PRINT / f"{PFX}_{n}_pla_t3.json").read_text()) for n in ("mount", "indenters")}
    m = est["masses_g"]
    parts_list = [
        dict(item="Adapter disc", qty=1, material="PLA", file=rel(fname("adapter", "step")), stl=rel(fname("adapter")),
             mass_g_solid=m["adapter"], note="spigot 31.40 x 5.5 mm and printed dowel 5.95 x 5.0 mm on the robot side"),
        dict(item="Standoff-fork", qty=1, material="PLA", file=rel(fname("standoff_fork", "step")), stl=rel(fname("standoff_fork")),
             mass_g_solid=m["standoff_fork"], note="foot, column and fork; the fork end prints at 100 % (modifier)"),
        dict(item="Clamp plate", qty=1, material="PLA", file=rel(fname("clamp_plate", "step")), stl=rel(fname("clamp_plate")),
             mass_g_solid=m["clamp_plate"], note="on the horn-side arm's outer face"),
        dict(item="Indenter, 5 mm sphere (optional)", qty=1, material="PLA", file=rel(fname("indenter_sphere", "step")),
             stl=rel(fname("indenter_sphere"))),
        dict(item="Indenter, 12.5 mm half rod (optional)", qty=1, material="PLA", file=rel(fname("indenter_rod", "step")),
             stl=rel(fname("indenter_rod"))),
    ] + [dict(item=v["what"], qty=v["qty"], key=k) for k, v in SCREWS.items()] + [
        dict(item="DIN 125 washer M6 (6.4 x 12 x 1.6)", qty=4), dict(item="DIN 9021 washer M3 (3.2 x 9 x 0.8), brass or A2", qty=1),
        dict(item="Cable ties, 2.5 mm wide", qty=2),
        dict(item="OnRobot Quick Changer tool plate (if the HEX-E is the QC version)", qty=1,
             note="the adapter bolts to its ISO 9409-1-50-4-M6 face; on a UR5e flange or a non-QC HEX-E's tool flange directly"),
        dict(item="DIP link as printed (CAD hand's, v2 a or v2 b) with its board, Qwiic cable and tip", qty=1)]
    steps = [
        "Print the two plates (print/*.3mf); check the adapter's spigot and dowel in the tool plate's recess and hole before "
        "anything else: the spigot slides in by hand, the dowel enters without force. A tight spigot: sand it or rebuild with "
        "spigot_d 0.05 mm smaller.",
        "Adapter on the tool plate (or the UR5e flange): dowel into the 6 mm hole, four M6 x 12 with DIN 125 washers into the "
        "counterbores, 1.5 N m (PLA under the heads; the UR5e's 8.0 N m is for metal tools). Each bolt enters the flange 7.4 mm "
        "(the UR5e allows 8, the tool plate 9).",
        "Standoff-fork on the adapter: four M3 x 12 through the foot into the adapter's 2.75 mm holes; they form their thread.",
        "Plug the Qwiic cable into J1 on the board; mount the board and the tip on the DIP link as its own README says.",
        "Slide the DIP link onto the fork along the tool axis: the arms straddle the inner block, the key enters the horn slot "
        "from its open end, the cable drops into the slot on the dorsal side. Push until the slot's round end seats on the key "
        "(the link then sits 0.1 mm past nominal; the hub stays 0.2 mm clear of the slot walls).",
        "Clamp plate on the horn-side arm's outer face; two M3 x 16 into the ears. Push the tip toward its back (the board "
        "side) while tightening, so the key bears on the slot's palmar flank, which is the flank the normal force loads; snug "
        "(about 0.3 N m). The plate presses the arm onto the inner block: the horn side is the datum.",
        "Slotted arm: M3 x 12 with the DIN 9021 washer through the 5.2 mm bore into the inner block; snug. The arm bends up to "
        "0.2 mm onto the block's face.",
        "Cable: along the groove on the column's dorsal face, a tie through each tunnel, out through the notch in the foot; "
        "leave no tension on J1.",
        "UR5e: declare the payload (set_payload) with the mass and centre of gravity of this page's payload table; TCP at the "
        "tip's contact point.",
    ]
    rec = dict(title="Parts list and assembly of the UR5e calibration mount", parts_list=parts_list, assembly=steps,
               print=dict((n, dict(project=v["project"], time=v["slice"]["summary"]["print_time"],
                                   filament_g=v["slice"]["summary"]["filament_g"], passes=v["slice"]["passes"]))
                          for n, v in pr.items()),
               step_files={k: dict(file=v["file"], bytes=v["bytes"], valid=v["readback_valid"]) for k, v in step.items()},
               when=time.strftime("%Y-%m-%d %H:%M"))
    (OUT / f"{PFX}_assembly.json").write_text(json.dumps(rec, indent=1, default=float))
    md = [f"# UR5e calibration mount: files and how to rebuild them", "",
          "Mount for a DIP link as printed (the CAD hand's, v2 a or v2 b of the DIP study: all keep the clevis arms) on the tool "
          "side of the HEX-E QC on the UR5e, built by `scripts/ur5_calibration_mount.py`. The page is "
          f"`{PFX}.html` in this folder.", "",
          "## Files", "",
          "- `cad/*.step`: exact solids per part, in the mount frame M (origin at the flange face's centre, z along the tool axis "
          "toward the tip, x toward the palmar side, y along the PIP axis toward the slotted arm); the indenters in a table frame "
          "(z up). `cad/*.stl`: the same as meshes.",
          "- `print/*.3mf`: PrusaSlicer projects for the Prusa XL on the owner's v17 template, PLA on tool 3, sliced and checked "
          "(`print/*.json`).",
          f"- `{PFX}_interfaces.json` (flange and DIP-link interfaces with sources), `_parts.json`, `_step.json`, `_check.json`, "
          "`_estimate.json` (deflection, payload, wrist field), `_assembly.json` (parts list, assembly).", "",
          "## Opening the STEP files in Inventor", "",
          "File > Open, file type STEP, Options > Convert Model: each part arrives as one editable solid body. The DIP study's STEP "
          "files (frame D of `fingertip_mount_DIP_link.stl`) go into this frame by a translation only: "
          f"x_M = x_D, y_M = y_D - {21.25 + Y_AXIS:.2f}, z_M = z_D + {PIP_M:.1f}.", "",
          "## Rebuild", "",
          "    E=~/miniconda3/envs/femtip/bin/python",
          "    RG=\"$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 400 -- taskset -c 12-15 env NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1\"",
          "    $RG $E scripts/ur5_calibration_mount.py interfaces",
          "    $RG $E scripts/ur5_calibration_mount.py build",
          "    $RG env UR5M_CMD=step UR5M_REPO=$PWD freecadcmd scripts/ur5_calibration_mount.py",
          "    $RG $E scripts/ur5_calibration_mount.py check",
          "    $RG $E scripts/ur5_calibration_mount.py estimate",
          "    $RG env MUJOCO_GL=egl .venv/bin/python scripts/ur5_calibration_mount.py render",
          "    $RG $E scripts/ur5_calibration_mount.py sections",
          "    $RG $E scripts/ur5_calibration_mount.py print",
          "    $E scripts/ur5_calibration_mount.py docs",
          "", "Sizes and clearances are the `P` dictionary at the top of the script; change one there and rerun from `build`.", ""]
    (OUT / "README.md").write_text("\n".join(md))
    print("docs ->", rel(OUT / f"{PFX}_assembly.json"), rel(OUT / "README.md"), flush=True)


# ------------------------------------------------------------------------------------------------ main
def main():
    if os.environ.get("UR5M_CMD") == "step":
        freecad_step()
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["interfaces", "build", "check", "estimate", "render", "sections", "print", "docs", "payload"])
    a = ap.parse_args()
    {"interfaces": cmd_interfaces, "build": cmd_build, "check": cmd_check if REV == "v1" else cmd_check_v2, "estimate": cmd_estimate, "render": cmd_render if REV == "v1" else cmd_render_v2,
     "sections": cmd_sections, "print": cmd_print, "docs": cmd_docs, "payload": cmd_payload_v2}[a.cmd](a)


if __name__ == "__main__":
    main()
elif os.environ.get("UR5M_CMD") == "step":         # freecadcmd runs the file as a module
    freecad_step()
