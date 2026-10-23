#!/usr/bin/env python3
"""Printed parts that put the lab's ATI Mini45 between the UR5's HEX-E and calibration mount v3, and a hook cross for its
calibration with a weighed mass (owner, 2026-10-23: "we can probably calibrate the ATI with just a 3dp mount to the HEX-E,
with the current indenter setup"; later: "design a mount for the ATI for the HEX-E and build up a calibration plan").

Revision 2 (2026-10-23, 11:30), from ATI's drawing 9230-05-1094 rev 20 (Mini45 transducer), fetched from
www.ati-ia.com/app_content/Documents/9230-05-1094.auto.pdf on 2026-10-23. Its sensing frame: origin at the centre of the
tool adapter plate's face, +Z from the mounting side to the tool side, +X and +Y as drawn. Mounting side (adapter plate
5.7 mm): six M3 x 0.5 tapped 5.0 deep on a 39.24 mm circle, in pairs 20 deg apart about 60, 180 and 300 deg from +X; a
3 mm slip-fit dowel hole 5 deep at +X on the same circle; a 9.5 mm recess 3 deep. Tool side (adapter plate 3.7 mm): six
M3 x 0.5 tapped 3.0 deep on a 28.2 mm circle, in pairs 40 deg apart about 60, 180 and 300 deg; two 3 mm dowel holes 3.4
deep at 0 and 120 deg on 28.194 mm. Height 15.7 mm, 45 mm across. The Mini45-R's cable leaves the mounting plate
radially at 110 deg from +X (20 deg past +Y), its fitting about as long as the plate is thick. Note 3: "DO NOT EXCEED
INTERFACE DEPTH, MAY CAUSE DAMAGE". Note 5: nothing may touch both adapter plates.

Stack, from the flange (M: mount v3's frame, origin on the HEX-E's tool face, z along the tool axis):
  HEX-E | v3's adapter (z_M 0..15; its M6 x 20 tightened first, heads on its top face) | ATI base (z_M 15..35: 70 mm
  across over the M6 heads, a 44 mm pad 2 mm high under the ATI so that the cable's fitting clears the disc, a channel and a
  zip-tie tunnel for the cable; four M3 x 20 into the adapter's inserts; six M3 x 12 from below into the ATI's mounting
  side, 4.0 mm of thread of its 5.0; a 3 x 8 dowel pin optional) | Mini45 (z_M 35..50.7) | ATI top (z_M 50.7..61.7, 63 mm:
  six M3 x 10 into the ATI's tool side, 2.5 mm of thread of its 3.0; two 3 x 6 dowel pins optional; four brass inserts in
  v3's foot pattern) | v3's standoff-fork unchanged, or for the calibration the hook cross.
The ATI turns 114 deg about the tool axis from x_M (the clock of thickest walls between the base's holes, 3.1 mm); its
cable then leaves toward 224 deg in M, 44 deg from the HEX-E's cable port (-x_M).

Hook cross: a 6 mm plate on the top's inserts, four arms 12 x 16 mm to 80 mm along +-x_M, +-y_M and a post 16 mm across
to 118 mm. Each eye is a 1.2 mm hole through a 1.6 mm end wall, with a slot behind it for the stopper knot of the line
from which the mass hangs. The line runs from the knot along gravity, or bends over the hole's outer edge when gravity is
more than atan(0.6 / 1.6) from the eye's axis; scripts/ati_calibration_analysis.py takes both into account.

Mass holder (12:52): the owner's weight (100 g, a body 25 mm across and 25 mm tall, a knob on top) in a cup with three crush
ribs, under a cap screwed on with two M3 x 12 that presses on its top face and covers the knob. Cap 2 (15:34, the knob
measured at 14 mm across and 11 mm tall): a pocket that centres the knob, and in place of 12:52's slot a 2 mm tie hole on
the axis; the line from the eye is tied through it, and the holder hangs from the hole's top. Both parts print upright
without supports.

    .venv/bin/python scripts/ati_stack_mount.py [build] [--sensor mini45|mini40] [--clock DEG]   STLs, parts JSON
                                                                           (the Mini45's build also writes the cross and holder)
    .venv/bin/python scripts/ati_stack_mount.py holder                     the holder's STLs and record only
    .venv/bin/python scripts/ati_stack_mount.py render [--figures stack,cross,holder]   the figures
(each through ~/.claude/bin/resguard.sh run --mem 3G: they load meshes)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("UR5M_REV", "v3")
import ur5_calibration_mount as UM  # noqa: E402  (MeshK: manifold booleans; v3's patterns; the renderer)

OUT = ROOT / "docs/experiments/20261023-ati_ft_calibration"
STEM = "20261023-ati_stack"
REV = "v2 2026-10-23 (ATI drawing 9230-05-1094 rev 20)"
# --- the Mini45 (ATI drawing 9230-05-1094 rev 20) ---
ATI_D, ATI_H, ATI_MOUNT_T, ATI_TOOL_T = 45.0, 15.7, 5.7, 3.7
ATI_MOUNT_BC, ATI_TOOL_BC, ATI_DOWEL_TOOL_BC = 39.24, 28.2, 28.194
MOUNT_PAIRS = (60.0, 180.0, 300.0)                 # deg from the ATI's +X; holes +-10 deg about each
MOUNT_PAIR_HALF, TOOL_PAIR_HALF = 10.0, 20.0
THREAD_MOUNT, THREAD_TOOL = 5.0, 3.0                # tapped depth (mm); do not exceed (note 3)
DOWEL_MOUNT, DOWEL_TOOL = (0.0,), (0.0, 120.0)      # deg from +X; 3 mm slip fit, 5.0 and 3.4 deep
DOWEL_DEPTH_MOUNT, DOWEL_DEPTH_TOOL = 5.0, 3.4
CABLE_DEG = 110.0                                   # Mini45-R cable exit, deg from +X, on the mounting plate
CABLE_FITTING = dict(d=6.0, out=2.0)                # its fitting about 6 mm across, 2 mm proud of the 45 mm body
# --- mount v3's interfaces (scripts/ur5_calibration_mount.py, UR5M_REV=v3) ---
ADAPTER_TOP = 15.0                                  # z_M of the adapter's top face
M6_ANG, M6_R, M6_HEAD_K = (45.0, 135.0, 225.0, 315.0), 25.0, 6.0
M3_ANG, M3_R = (0.0, 90.0, 198.0, 270.0), 25.5      # the foot's M3 into the adapter's inserts (and now the top's)
POCKET_D, POCKET_H = 12.5, 6.5
INS_HOLE, INS_DEPTH, INS_CH = 4.0, 9.0, 0.3         # M3 x 5.7 brass heat-set insert bore (v3's adapter)
# --- the parts ---
BASE_D, BASE_T, PAD_D, PAD_H = 70.0, 18.0, 44.0, 2.0
TOP_D, TOP_T = 63.0, 11.0
CB_D, CLR_D = 6.2, 3.4                              # M3 socket head counterbore and clearance
SCREW = dict(base_to_adapter=20.0, ati_mount=12.0, ati_tool=10.0, cross_to_top=12.0)   # M3 lengths under the head
ENGAGE_MOUNT, ENGAGE_TOOL = 4.0, 2.5                # mm of the ATI's thread the screws take (of 5.0 and 3.0)
ADAPTER_CB = 3.5                                    # the base's counterbores for its M3 x 20: 5.5 mm into the 5.7 insert
DOWEL_HOLE = 3.0                                    # printed 3.0: a press fit for a 3 mm pin
CHANNEL_W, CHANNEL_DEPTH = 7.0, 3.0
TIE_R, TIE_Z = (28.0, 31.0), (11.2, 13.2)           # the zip-tie tunnel under the channel (radius, z of the base)
# --- hook cross ---
ARM_L, ARM_W, ARM_H, PLATE_T = 80.0, 12.0, 16.0, 6.0
POST_D, POST_H = 16.0, 118.0
EYE_D, EYE_WALL, KNOT_SLOT = 1.2, 1.6, (8.0, 4.0)    # hole, end wall, slot behind it (length, width)
CROSS_CB_Z = 5.5                                     # the cross's M3 heads 5.5 mm above its underside
# --- the sensors: the Mini45 above, and the Mini40 (owner, 2026-10-23 17:36: "i only want to do this once, so can you set
# up a mount for the ati mini 40 too"), from ATI's drawings 9230-05-1278 rev 11 (Mini40-R and -A) and 9230-05-1314 rev 09
# (Mini40-E), fetched 2026-10-23. Both Mini40 drawings give the same interface: mounting side three M3 x 0.5 tapped 5.0
# deep on a 33 mm circle at 82.5, 202.5 and 322.5 deg from +X, two 3 mm slip-fit dowel holes on it at 37.5 and 277.5 deg
# (2.5 deep on -R, 5.0 on -E), a 6.35 mm centre hole; tool side three M3 x 0.5 tapped 3.0 deep on 33 mm at 60, 180 and
# 300 deg, two 3 mm dowel holes 2.5 deep on a 35 mm circle at 0 and 120 deg. 40.1 mm across; 12.27 mm tall (-R, -A) or
# 14.0 (-E). The cable leaves the mounting plate radially at 150 deg (-R, a fitting) or 157.5 deg (-E, a 17.3 mm strain
# relief); -A's leaves the mounting face's centre and is not provided for. Origin at the tool face's centre, as the Mini45.
SENSORS = {
    "mini45": dict(key="mini45", name="Mini45", stem=STEM, rev=REV, H=ATI_H, D=ATI_D, mount_t=ATI_MOUNT_T,
                   drawing="ATI drawing 9230-05-1094 rev 20, Mini45 transducer (www.ati-ia.com, fetched 2026-10-23)",
                   mount=[(ATI_MOUNT_BC / 2, c + s_ * MOUNT_PAIR_HALF) for c in MOUNT_PAIRS for s_ in (-1, 1)],
                   tool=[(ATI_TOOL_BC / 2, c + s_ * TOOL_PAIR_HALF) for c in MOUNT_PAIRS for s_ in (-1, 1)],
                   thread_mount=THREAD_MOUNT, thread_tool=THREAD_TOOL,
                   dowel_mount=[(ATI_MOUNT_BC / 2, a) for a in DOWEL_MOUNT], dowel_mount_hole=5.0, dowel_mount_pin="3 x 8",
                   dowel_tool=[(ATI_DOWEL_TOOL_BC / 2, a) for a in DOWEL_TOOL],
                   cable_deg=CABLE_DEG, cable=CABLE_FITTING, pad_d=PAD_D, channel_w=CHANNEL_W, mass_g=92.0),
    "mini40": dict(key="mini40", name="Mini40", stem=STEM + "_mini40", H=12.27, D=40.1, mount_t=5.5, H_E=14.0,
                   rev="v1 2026-10-23 (ATI drawings 9230-05-1278 rev 11 and 9230-05-1314 rev 09)",
                   drawing="ATI drawings 9230-05-1278 rev 11, Mini40 transducer radial and axial cable exit, and 9230-05-1314 "
                           "rev 09, Mini40-E (www.ati-ia.com, fetched 2026-10-23)",
                   mount=[(16.5, a) for a in (82.5, 202.5, 322.5)], tool=[(16.5, a) for a in (60.0, 180.0, 300.0)],
                   thread_mount=5.0, thread_tool=3.0,
                   dowel_mount=[(16.5, 37.5), (16.5, 277.5)], dowel_mount_hole=4.0, dowel_mount_pin="3 x 6",
                   dowel_tool=[(17.5, 0.0), (17.5, 120.0)],
                   cable_deg=153.75, cable_variants={"-R": 150.0, "-E": 157.5}, cable=dict(d=6.0, out=3.0),
                   pad_d=38.0, channel_w=10.0, cable_toward=224.0, cable_tol=10.0, mass_g=50.0),
}
S = SENSORS["mini45"]                                # the sensor being built; main() sets it
# --- the mass holder (owner, 2026-10-23 12:52: "a 100g mass on a 25mm diameter 25mm height form factor. theres a little
# knobby thing coming out. i think we'd need to built a tight-fit enclosure. i also dont have a hook") ---
WEIGHT_D, WEIGHT_H = 25.0, 25.0                      # the weight's body as the owner measured it
KNOB = (14.0, 11.0)                                  # the knob, across and tall (owner, 15:34: "7mm in radius, 11mm in height")
KNOB_POCKET = (14.6, 11.5)                           # the cap's pocket for it (12:52's cap: 18 x 13, the knob unmeasured)
HOLD_BORE, HOLD_WALL, HOLD_FLOOR, PUSH_D = 25.6, 2.4, 2.4, 8.0
RIB_TIP_D, RIB_HALF, RIB_ANG = 24.8, 0.7, (90.0, 210.0, 330.0)    # three crush ribs 0.4 mm proud of the bore
HOLD_PROUD = 0.4                                     # the weight stands this far above the cup's rim: the cap clamps it
EAR_R, EAR_D, PILOT_D, PILOT_DEPTH = 16.0, 8.0, 2.75, 10.0         # the cap's two M3 x 12 into self-formed threads
FLANGE_T, DOME_WALL = 4.0, 2.4
TAB_T, TAB_W = 5.0, 12.0                             # the tab on the dome: thickness, width (its top a half-disc)
TIE_HOLE_D, TIE_HOLE_Z = 2.0, 4.0                    # its tie hole on the axis: across, centre above the dome (15:34: the
                                                     # line is tied through the cap, the slot of 12:52 dropped)
PIGTAIL_MM = 60.0                                    # the eye to the cap's tie hole along the line
HALVES = ROOT / "logs/20261023-ati_mass_holder"      # cut-away halves for the figure (written by the holder build)


def z_base0():
    return ADAPTER_TOP


def z_ati0():
    return ADAPTER_TOP + BASE_T + PAD_H              # the ATI's mounting face (z_M)


def z_ati_origin():
    return z_ati0() + S["H"]                         # its sensing origin: the tool face's centre


def z_top1():
    return z_ati_origin() + TOP_T                    # the top's upper face: where the fork's or cross's foot sits


def xy(r, a):
    return r * math.cos(math.radians(a)), r * math.sin(math.radians(a))


def mount_holes(clock):
    return [xy(r, clock + a) for r, a in S["mount"]]


def tool_holes(clock):
    return [xy(r, clock + a) for r, a in S["tool"]]


def walls(clock):
    """Least wall (mm) between the base's holes for a clock of the ATI (deg of its +X from x_M): the ATI screws'
    counterbores (z 0..12 of the base) and clearance holes (12..20) against the base's own M3 holes into the adapter and
    their counterbores (14.5..18); the dowel hole and the cable channel against those counterbores."""
    w = []
    for p in mount_holes(clock):
        for a in M3_ANG:
            q = xy(M3_R, a)
            d = math.hypot(p[0] - q[0], p[1] - q[1])
            w += [d - CB_D / 2 - CLR_D / 2, d - CLR_D / 2 - 3.25]
    for r0, a0 in S["dowel_mount"]:
        p = xy(r0, clock + a0)
        for a in M3_ANG:
            q = xy(M3_R, a)
            w.append(math.hypot(p[0] - q[0], p[1] - q[1]) - DOWEL_HOLE / 2 - 3.25)
    u = np.array(xy(1.0, clock + S["cable_deg"]))
    n = np.array([-u[1], u[0]])
    cw = S["channel_w"]
    for a in M3_ANG:
        q = np.array(xy(M3_R, a))
        if 18.0 < q @ u < 38.0:
            w.append(abs(q @ n) - cw / 2 - 3.25)
        if TIE_R[0] - 3.0 < q @ u < TIE_R[1] + 3.0:               # the tie's tunnel and slots against the through holes
            w.append(abs(q @ n) - cw / 2 - 3.0 - CLR_D / 2)
    for p in mount_holes(clock):                                  # and against the ATI screws' counterbores (z 0..12)
        q = np.array(p)
        if TIE_R[0] - 4.0 < q @ u < TIE_R[1] + 4.0:
            w.append(abs(q @ n) - cw / 2 - 3.0 - CB_D / 2)
    return min(w)


def base(K, clock):
    """z from the adapter's top face (z_M 15)."""
    pad_d, cw = S["pad_d"], S["channel_w"]
    s = K.union([K.cyl("z", (0.0, 0.0), BASE_D / 2, 0.0, BASE_T), K.cyl("z", (0.0, 0.0), pad_d / 2, BASE_T - 0.01, BASE_T + PAD_H)])
    top = BASE_T + PAD_H
    seat = top + ENGAGE_MOUNT - SCREW["ati_mount"]                # 12.0: the ATI screws' heads from below
    cut = []
    for a in M6_ANG:                                              # the M6 heads under the base
        cut.append(K.cyl("z", xy(M6_R, a), POCKET_D / 2, -1.0, POCKET_H))
    for a in M3_ANG:                                              # M3 x 20 into the adapter's inserts, heads beside the ATI
        c = xy(M3_R, a)
        cut += [K.cyl("z", c, CLR_D / 2, -1.0, BASE_T + 1.0), K.cyl("z", c, 3.25, BASE_T - ADAPTER_CB, BASE_T + 1.0)]
    for c in mount_holes(clock):                                  # M3 x 12 up into the ATI; counterbores from below
        cut += [K.cyl("z", c, CLR_D / 2, -1.0, top + 1.0), K.cyl("z", c, CB_D / 2, -1.0, seat)]
    for r0, a0 in S["dowel_mount"]:                               # Mini45: a 3 x 8 pin, 5 mm in the base, 3 into the ATI's 5.0;
        cut.append(K.cyl("z", xy(r0, clock + a0), DOWEL_HOLE / 2,     # Mini40: 3 x 6, 4 mm in the base, 2 into its 2.5
                         top - S["dowel_mount_hole"], top + 1.0))
    # the cable: a channel from under the ATI's rim to the base's edge, 3 mm deep, and a zip-tie tunnel under its floor
    a = clock + S["cable_deg"]
    R = UM._axis_T("z")
    ch = K.box((pad_d / 2 - 1.5, -cw / 2, BASE_T - CHANNEL_DEPTH), (BASE_D / 2 + 1.0, cw / 2, top + 1.0))
    t0, t1 = TIE_Z                                                # a 1.8 mm bridge under the channel's floor
    tie = [K.box((TIE_R[0], -cw / 2 - 3.0, t0), (TIE_R[1], cw / 2 + 3.0, t1)),
           K.box((TIE_R[0], cw / 2 + 0.8, t0), (TIE_R[1], cw / 2 + 3.0, BASE_T + 1.0)),
           K.box((TIE_R[0], -cw / 2 - 3.0, t0), (TIE_R[1], -cw / 2 - 0.8, BASE_T + 1.0))]
    rot = np.eye(4)
    rot[:2, :2] = [[math.cos(math.radians(a)), -math.sin(math.radians(a))], [math.sin(math.radians(a)), math.cos(math.radians(a))]]
    cut += [K.transform(t, rot @ R) for t in [ch] + tie]
    return K.cut_all(s, cut)


def top(K, clock):
    """z from the ATI's tool face."""
    s = K.cyl("z", (0.0, 0.0), TOP_D / 2, 0.0, TOP_T)
    seat = SCREW["ati_tool"] - ENGAGE_TOOL                        # 7.5: the heads 3.5 mm under the top face
    cut = []
    for c in tool_holes(clock):
        cut += [K.cyl("z", c, CLR_D / 2, -1.0, TOP_T + 1.0), K.cyl("z", c, CB_D / 2, seat, TOP_T + 1.0)]
    for r0, a0 in S["dowel_tool"]:                                # 3 x 6 pins: 4 mm in the top, 2 mm into the ATI's 3.4 (2.5)
        cut.append(K.cyl("z", xy(r0, clock + a0), DOWEL_HOLE / 2, -1.0, 4.0))
    for a in M3_ANG:                                              # inserts for the fork's foot (or the hook cross)
        c = xy(M3_R, a)
        cut += [K.cyl("z", c, INS_HOLE / 2, TOP_T - INS_DEPTH, TOP_T + 1.0),
                K.cyl("z", c, INS_HOLE / 2, TOP_T - INS_CH, TOP_T + 0.01, r1=INS_HOLE / 2 + INS_CH + 0.01)]
    return K.cut_all(s, cut)


def eyes():
    """The five eyes in the cross's frame (z from its underside, axes of M): exit point, outward direction."""
    zc = ARM_H / 2
    out = []
    for name, u in (("+x", (1, 0, 0)), ("-x", (-1, 0, 0)), ("+y", (0, 1, 0)), ("-y", (0, -1, 0))):
        u = np.array(u, float)
        out.append(dict(name=name, p=(u * ARM_L + [0.0, 0.0, zc]).tolist(), u=u.tolist()))
    out.append(dict(name="+z", p=[0.0, 0.0, POST_H], u=[0.0, 0.0, 1.0]))
    return out


def cross(K):
    """On the top's inserts: a 6 mm plate, four arms to 80 mm (on the bed, 16 mm high), a post to 118 mm; each eye a
    1.2 mm hole through a 1.6 mm end wall with a slot behind it for the line's stopper knot."""
    parts = [K.cyl("z", (0.0, 0.0), TOP_D / 2, 0.0, PLATE_T)]
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        x0, x1 = sorted((0.0, sx * ARM_L)) if sx else (-ARM_W / 2, ARM_W / 2)
        y0, y1 = sorted((0.0, sy * ARM_L)) if sy else (-ARM_W / 2, ARM_W / 2)
        parts.append(K.box((x0, y0, 0.0), (x1, y1, ARM_H)))
    parts.append(K.cyl("z", (0.0, 0.0), POST_D / 2, 0.0, POST_H))
    s = K.union(parts)
    cut = []
    for a in M3_ANG:                                              # M3 x 12 into the top's inserts, keys from above
        c = xy(M3_R, a)
        cut += [K.cyl("z", c, CLR_D / 2, -1.0, POST_H), K.cyl("z", c, 3.25, CROSS_CB_Z, POST_H)]
    L, W = KNOT_SLOT
    zc = ARM_H / 2
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        e = ARM_L - EYE_WALL
        if sx:
            x0, x1 = sorted((sx * (e - L), sx * e))
            cut.append(K.box((x0, -W / 2, -1.0), (x1, W / 2, ARM_H + 1.0)))
            cut.append(K.cyl("x", (0.0, zc), EYE_D / 2, sx * (e - 1.0) if sx > 0 else -ARM_L - 1.0,
                             ARM_L + 1.0 if sx > 0 else -(e - 1.0)))
        else:
            y0, y1 = sorted((sy * (e - L), sy * e))
            cut.append(K.box((-W / 2, y0, -1.0), (W / 2, y1, ARM_H + 1.0)))
            cut.append(K.cyl("y", (0.0, zc), EYE_D / 2, sy * (e - 1.0) if sy > 0 else -ARM_L - 1.0,
                             ARM_L + 1.0 if sy > 0 else -(e - 1.0)))
    e = POST_H - EYE_WALL                                         # the post's eye: an axial hole, the slot across the post
    cut.append(K.box((-W / 2, -POST_D, e - L), (W / 2, POST_D, e)))
    cut.append(K.cyl("z", (0.0, 0.0), EYE_D / 2, e - 1.0, POST_H + 1.0))
    return K.cut_all(s, cut)


def _prism_z(K, poly, z0, z1):
    """Polygon in the x-y plane (list of (x, y)) extruded along z from z0 to z1."""
    poly = np.asarray(poly, float)
    a = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    return K.M.Manifold.extrude(K.M.CrossSection([poly if a > 0 else poly[::-1]]), z1 - z0).translate([0.0, 0.0, z0])


def cup_h():
    return HOLD_FLOOR + WEIGHT_H - HOLD_PROUD


def cap_top():
    return KNOB_POCKET[1] + DOME_WALL                # the dome's top above the cap's underside


def holder_cup(K):
    """The weight's cup, on the bed as it hangs: a 25.6 mm bore with three crush ribs to 24.8 mm, their tops tapered, a floor
    with an 8 mm hole to push the weight out, and two ears with 2.75 mm holes for the cap's M3 x 12."""
    H, R = cup_h(), HOLD_BORE / 2 + HOLD_WALL
    s = K.union([K.cyl("z", (0.0, 0.0), R, 0.0, H)] + [K.cyl("z", xy(EAR_R, a), EAR_D / 2, 0.0, H) for a in (0.0, 180.0)])
    s = K.cut_all(s, [K.cyl("z", (0.0, 0.0), HOLD_BORE / 2, HOLD_FLOOR, H + 1.0),
                      K.cyl("z", (0.0, 0.0), PUSH_D / 2, -1.0, HOLD_FLOOR + 1.0)]
                  + [K.cyl("z", xy(EAR_R, a), PILOT_D / 2, H - PILOT_DEPTH, H + 1.0) for a in (0.0, 180.0)])
    rt, rw = RIB_TIP_D / 2, HOLD_BORE / 2
    ribs = []
    for a in RIB_ANG:
        c, sn = math.cos(math.radians(a)), math.sin(math.radians(a))
        pts = [(rt, 0.0), (rw, RIB_HALF), (rw + 0.4, RIB_HALF), (rw + 0.4, -RIB_HALF), (rw, -RIB_HALF)]
        ribs.append(_prism_z(K, [(r * c - t * sn, r * sn + t * c) for r, t in pts], HOLD_FLOOR, H))
    s = K.union([s] + ribs)
    # the ribs' tops tapered over 3 mm, the rim chamfered 0.8 mm at 45 deg
    return K.cut_all(s, [K.cyl("z", (0.0, 0.0), rt - 0.01, H - 3.8, H - 0.8, r1=rw + 0.01),
                         K.cyl("z", (0.0, 0.0), rw + 0.01, H - 0.8, H + 0.01, r1=rw + 0.8)])


def holder_cap(K):
    """The cap, on the bed as it hangs: a 4 mm flange that presses on the weight's top face, held by two M3 x 12 through
    its ears; a dome over the knob, its pocket 0.3 mm wider and 0.5 mm deeper than the knob, so the knob centres it; on
    the dome a tab 5 mm thick with a 2 mm tie hole on the axis. The line from the eye is tied through the hole, and the
    holder hangs from the hole's top."""
    kd, kh = KNOB_POCKET
    top, w = cap_top(), TAB_W / 2
    parts = [K.cyl("z", (0.0, 0.0), HOLD_BORE / 2 + HOLD_WALL, 0.0, FLANGE_T),
             K.cyl("z", (0.0, 0.0), kd / 2 + DOME_WALL, 0.0, top)]
    parts += [K.cyl("z", xy(EAR_R, a), EAR_D / 2, 0.0, FLANGE_T) for a in (0.0, 180.0)]
    parts.append(K.prism_xz([(-w, top - 0.5), (w, top - 0.5), (w, top + TIE_HOLE_Z), (-w, top + TIE_HOLE_Z)], -TAB_T / 2, TAB_T / 2))
    parts.append(K.cyl("y", (0.0, top + TIE_HOLE_Z), w, -TAB_T / 2, TAB_T / 2))
    s = K.union(parts)
    cut = [K.cyl("z", (0.0, 0.0), kd / 2, -1.0, kh),
           K.cyl("z", (0.0, 0.0), kd / 2 + 0.6, -0.01, 0.6, r1=kd / 2 - 0.01)]          # a 0.6 mm lead-in at the pocket
    cut += [K.cyl("z", xy(EAR_R, a), CLR_D / 2, -1.0, FLANGE_T + 1.0) for a in (0.0, 180.0)]
    cut.append(K.cyl("y", (0.0, top + TIE_HOLE_Z), TIE_HOLE_D / 2, -TAB_T, TAB_T))
    return K.cut_all(s, cut)


def holder_geometry(plan_mass=None):
    """Heights in the hanging holder (mm, from the cup's underside) and the line lengths that keep it inside the plan's
    envelope for the mass: a thin line for line_mm below the eye, then a cylinder r_mm across and h_mm long."""
    env = plan_mass or dict(line_mm=45.0, r_mm=25.0, h_mm=80.0)
    z_cap = cup_h() + HOLD_PROUD
    seat = z_cap + cap_top() + TIE_HOLE_Z + TIE_HOLE_D / 2      # the line bears on the tie hole's top
    tab_top = z_cap + cap_top() + TIE_HOLE_Z + TAB_W / 2
    com = HOLD_FLOOR + WEIGHT_H / 2                   # the weight's centre, about the whole holder's
    return dict(seat_mm=round(seat, 2), top_mm=round(tab_top, 2), radius_mm=EAR_R + EAR_D / 2,
                com_below_seat_mm=round(seat - com, 1),
                seat_below_eye_mm=[round(env["line_mm"] + tab_top - seat, 1), round(env["line_mm"] + env["h_mm"] - seat, 1)],
                envelope=env, fits=bool(EAR_R + EAR_D / 2 <= env["r_mm"]))


def holder_build(K=None):
    K = K or UM.MeshK()
    try:
        env = json.loads((OUT / "plan/20261023-ati_plan_mass.json").read_text())["mass"]
    except (OSError, ValueError, KeyError):
        env = None
    rec = dict(when=time.strftime("%Y-%m-%d %H:%M"), revision="cap 2 (15:34): pocket for the measured knob, tie hole",
               weight=dict(d_mm=WEIGHT_D, h_mm=WEIGHT_H, knob_mm=KNOB, knob_pocket_mm=KNOB_POCKET, tie_hole_mm=TIE_HOLE_D),
               geometry=holder_geometry(env), parts={},
               screws="2 x M3 x 12 socket head through the cap's ears into the cup's 2.75 mm holes (threads formed in the PLA)")
    for nm, s in (("mass_cup", holder_cup(K)), ("mass_cap", holder_cap(K))):
        tm = UM.to_trimesh(s)
        f = OUT / f"cad/20261023-ati_{nm}.stl"
        tm.export(f)
        rec["parts"][nm] = dict(file=str(f.relative_to(ROOT)), volume_mm3=round(float(s.volume()), 1),
                                bounds=np.round(np.asarray(tm.bounds), 2).tolist(), watertight=bool(tm.is_watertight))
        print(f"{nm}: {s.volume():.0f} mm3, bounds {np.round(np.asarray(tm.bounds), 1).tolist()}", flush=True)
        HALVES.mkdir(parents=True, exist_ok=True)                 # the halves on y <= 0 for the cut-away figure
        UM.to_trimesh(K.inter(s, K.box((-60.0, -60.0, -1.0), (60.0, 0.0, 100.0)))).export(HALVES / f"{nm}_half.stl")
    (OUT / "20261023-ati_mass_holder.json").write_text(json.dumps(rec, indent=1) + "\n")
    g = rec["geometry"]
    print(f"holder: seat {g['seat_mm']} mm above the cup's underside, top {g['top_mm']}, {g['radius_mm']} mm across the ears "
          f"(radius); the seat {g['seat_below_eye_mm'][0]}-{g['seat_below_eye_mm'][1]} mm below the eye keeps it in the plan's "
          f"envelope", flush=True)
    return rec


def frames(clock):
    """The ATI's sensing frame in M (4 x 4) and the eyes in M and in the ATI frame."""
    c, s_ = math.cos(math.radians(clock)), math.sin(math.radians(clock))
    T_MA = np.eye(4)
    T_MA[:3, :3] = [[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]
    T_MA[:3, 3] = [0.0, 0.0, z_ati_origin()]
    R_AM, t = T_MA[:3, :3].T, T_MA[:3, 3]
    ev = []
    for e in eyes():
        pM = np.array(e["p"]) + [0.0, 0.0, z_top1()]
        ev.append(dict(name=e["name"], p_M=np.round(pM, 3).tolist(), u_M=e["u"],
                       p_A=np.round(R_AM @ (pM - t), 3).tolist(), u_A=np.round(R_AM @ np.array(e["u"]), 6).tolist()))
    return T_MA, ev


def best_clock():
    """The clock of thickest walls; for a sensor with cable_toward, among the clocks that point its cable within cable_tol
    of that direction in M (the Mini40's toward the Mini45's, so that both cables take one route along the arm)."""
    cl = np.arange(0.0, 360.0, 1.0)
    if "cable_toward" in S:
        d = np.abs((cl + S["cable_deg"] - S["cable_toward"] + 180.0) % 360.0 - 180.0)
        cl = cl[d <= S["cable_tol"]]
    return max(cl, key=walls)


def main(clock=None, sensor="mini45"):
    global S
    S = SENSORS[sensor]
    if clock is None:
        clock = best_clock()
    K = UM.MeshK()
    (OUT / "cad").mkdir(parents=True, exist_ok=True)
    T_MA, ev = frames(clock)
    nm_, nd_ = len(S["dowel_mount"]), len(S["dowel_tool"])
    rec = dict(sensor=S["name"], revision=S["rev"], when=time.strftime("%Y-%m-%d %H:%M"), clock_deg=float(clock),
               wall_mm=round(walls(clock), 2), cable_deg_M=float((clock + S["cable_deg"]) % 360.0),
               source=S["drawing"], ati=dict(d_mm=S["D"], h_mm=S["H"], mass_g=S["mass_g"], mass_note="data sheet; weigh it"),
               z_M=dict(adapter_top=ADAPTER_TOP, base_top=ADAPTER_TOP + BASE_T, ati_mount_face=z_ati0(),
                        ati_origin=z_ati_origin(), top_face=z_top1(), cross_post_eye=z_top1() + POST_H),
               added_length_mm=z_top1() - ADAPTER_TOP, T_MA=np.round(T_MA, 6).tolist(), eyes=ev,
               eye=dict(hole_d=EYE_D, wall=EYE_WALL),
               screws=[dict(qty=4, what=f"M3 x {SCREW['base_to_adapter']:.0f} socket head, base into v3's adapter inserts "
                                        f"({SCREW['base_to_adapter'] - (BASE_T - ADAPTER_CB):.1f} mm into the 5.7 mm insert)"),
                       dict(qty=len(S["mount"]), what=f"M3 x {SCREW['ati_mount']:.0f} socket head, from the base's underside "
                                        f"into the ATI's mounting side: {ENGAGE_MOUNT:.1f} mm of its {S['thread_mount']:.1f} mm "
                                        "thread"),
                       dict(qty=len(S["tool"]), what=f"M3 x {SCREW['ati_tool']:.0f} socket head, through the top into the ATI's "
                                        f"tool side: {ENGAGE_TOOL:.1f} mm of its {S['thread_tool']:.1f} mm thread; never longer"),
                       dict(qty=4, what="M3 x 5.7 brass heat-set inserts in the top (v3's foot pattern)"),
                       dict(qty=4, what=f"M3 x {SCREW['cross_to_top']:.0f} socket head, the hook cross into the top's "
                                        "inserts (the fork takes its own M3 x 12)"),
                       dict(qty=nm_ + nd_, what=f"3 mm dowel pins, optional: {'one' if nm_ == 1 else 'two'} "
                                                f"{S['dowel_mount_pin']} in the base, two 3 x 6 in the top")]
               if sensor == "mini45" else
               [dict(qty=4, what=f"M3 x {SCREW['base_to_adapter']:.0f} socket head, base into v3's adapter inserts "
                                 f"({SCREW['base_to_adapter'] - (BASE_T - ADAPTER_CB):.1f} mm into the 5.7 mm insert)"),
                dict(qty=len(S["mount"]), what=f"M3 x {SCREW['ati_mount']:.0f} socket head, from the base's underside into the "
                                               f"ATI's mounting side: {ENGAGE_MOUNT:.1f} mm of its {S['thread_mount']:.1f} mm thread"),
                dict(qty=len(S["tool"]), what=f"M3 x {SCREW['ati_tool']:.0f} socket head, through the top into the ATI's tool "
                                              f"side: {ENGAGE_TOOL:.1f} mm of its {S['thread_tool']:.1f} mm thread; never longer"),
                dict(qty=4, what="M3 x 5.7 brass heat-set inserts in the top (v3's foot pattern)"),
                dict(qty=4, what=f"M3 x {SCREW['cross_to_top']:.0f} socket head, the hook cross into the top's inserts"),
                dict(qty=nm_ + nd_, what=f"3 mm dowel pins, optional: two {S['dowel_mount_pin']} in the base (2 mm into the "
                                         "Mini40's 2.5 mm holes), two 3 x 6 in the top")],
               parts={})
    builds = [("ati_base", lambda: base(K, clock)), ("ati_top", lambda: top(K, clock))]
    if sensor == "mini45":
        builds.append(("hook_cross", lambda: cross(K)))
    else:                                                         # the cross and the mass holder are shared
        rec["parts"]["hook_cross"] = json.loads((OUT / f"{STEM}_parts.json").read_text())["parts"]["hook_cross"]
        rec["variants"] = dict(height_mm={"-R": S["H"], "-A": S["H"], "-E": S["H_E"]}, cable_deg=S["cable_variants"],
                               note="the -E stands 1.73 mm taller; the eyes keep their places in the ATI's frame, so the "
                                    "plan and the analysis hold for both")
    for nm, fn in builds:
        s = fn()
        tm = UM.to_trimesh(s)
        f = OUT / f"cad/{S['stem']}_{nm}.stl"
        tm.export(f)
        rec["parts"][nm] = dict(file=str(f.relative_to(ROOT)), volume_mm3=round(float(s.volume()), 1),
                                bounds=np.round(np.asarray(tm.bounds), 2).tolist(), watertight=bool(tm.is_watertight))
        print(f"{nm}: {s.volume():.0f} mm3, bounds {np.round(np.asarray(tm.bounds), 1).tolist()}", flush=True)
    (OUT / f"{S['stem']}_parts.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{S['name']}: clock {clock:.0f} deg: {rec['wall_mm']:.2f} mm least wall; cable toward {rec['cable_deg_M']:.0f} deg "
          f"in M; the tool {rec['added_length_mm']:.1f} mm longer; ATI origin at z_M {z_ati_origin():.2f}")
    if sensor == "mini45":
        holder_build(K)
    return rec


# ------------------------------------------------------------------------------------------------ figures
NUM = {2: "two", 3: "three", 4: "four", 6: "six"}


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def ati_mesh(clock):
    """A stand-in of the Mini45 in M: the body, the cable's fitting and a stub of cable."""
    import trimesh
    parts = [trimesh.creation.cylinder(radius=S["D"] / 2, height=S["H"], sections=96)]
    parts[0].apply_translation([0.0, 0.0, z_ati0() + S["H"] / 2])
    a = math.radians(clock + S["cable_variants"]["-R"] if "cable_variants" in S else clock + S["cable_deg"])
    u = np.array([math.cos(a), math.sin(a), 0.0])
    z = z_ati0() + S["mount_t"] / 2
    D, cf = S["D"], S["cable"]
    fit = trimesh.creation.cylinder(radius=cf["d"] / 2, segment=[u * (D / 2 - 1.0) + [0, 0, z], u * (D / 2 + cf["out"]) + [0, 0, z]])
    cab = trimesh.creation.cylinder(radius=2.6, segment=[u * (D / 2 + 1.0) + [0, 0, z], u * (BASE_D / 2 + 6.0) + [0, 0, z - 4.0]])
    return trimesh.util.concatenate(parts + [fit, cab])


def render(which=("stack", "cross", "holder")):
    import trimesh
    rec = json.loads((OUT / f"{S['stem']}_parts.json").read_text())
    clock = rec["clock_deg"]
    (OUT / "media").mkdir(parents=True, exist_ok=True)
    mnt = UM.OUT.parent / "20261021-ur5_calibration_mount_v3/cad"
    ad = trimesh.load(mnt / "20261021-ur5_calibration_mount_v3_adapter.stl", force="mesh")
    fork = trimesh.load(mnt / "20261021-ur5_calibration_mount_v3_standoff_fork.stl", force="mesh")
    base_m = trimesh.load(OUT / f"cad/{S['stem']}_ati_base.stl", force="mesh")
    top_m = trimesh.load(OUT / f"cad/{S['stem']}_ati_top.stl", force="mesh")
    cr = trimesh.load(OUT / f"cad/{STEM}_hook_cross.stl", force="mesh")              # shared by both sensors
    W = UM.T_W_FROM_M
    gap = 22.0
    items, labels = [], []
    dz = dict(adapter=0.0, base=gap, ati=2 * gap, top=3 * gap, fork=4 * gap)
    fork_z = z_top1() - (ADAPTER_TOP)                             # the fork's foot from the adapter's face to the top's
    for name, m, z, col, lab in (
            ("adapter", ad, dz["adapter"], UM.COLORS["adapter"], "Mount v3's adapter on the HEX-E, its M6 x 20 tightened first"),
            ("base", base_m, ADAPTER_TOP + dz["base"], "0.55 0.78 0.50 1", f"ATI base: over the M6 heads, a {S['pad_d']:.0f} mm "
             "pad, a cable channel; four M3 x 20"),
            ("ati", ati_mesh(clock), dz["ati"], "0.80 0.81 0.84 1", f"ATI {S['name']}: {NUM[len(S['mount'])]} M3 x 12 from below "
             f"(4.0 mm of {S['thread_mount']:.1f})"),
            ("top", top_m, z_ati_origin() + dz["top"], "0.55 0.78 0.50 1", f"ATI top: {NUM[len(S['tool'])]} M3 x 10 into the ATI "
             f"(2.5 mm of {S['thread_tool']:.1f}), four inserts"),
            ("fork", fork, fork_z + dz["fork"], UM.COLORS["standoff_fork"], "Mount v3's standoff-fork, unchanged")):
        mm = m.copy()
        mm.apply_translation([0.0, 0.0, z])
        mm.apply_transform(W)
        items.append((mm, col, f"ati_{name}"))
        c = mm.bounds.mean(axis=0)
        labels.append((lab, [c[0] + 30.0, c[1] - 20.0, c[2]]))
    ctr = np.mean([mm.bounds.mean(axis=0) for mm, _, _ in items], axis=0)
    cam = (ctr.tolist(), 700.0, 210.0, 12.0)
    if "stack" in which:
        dst = UM._render(items, cam, OUT / f"media/{S['stem']}_exploded.png")
        UM._annotate(dst, cam, labels, title=f"The ATI {S['name']} between the HEX-E and mount v3, exploded along the tool axis")
    # the hook cross on the stack with a line and a mass hanging from the +x eye, the tool horizontal
    items, labels = [], []
    for name, m, z, col in (("adapter", ad, 0.0, UM.COLORS["adapter"]), ("base", base_m, ADAPTER_TOP, "0.55 0.78 0.50 1"),
                            ("ati", ati_mesh(clock), 0.0, "0.80 0.81 0.84 1"), ("top", top_m, z_ati_origin(), "0.55 0.78 0.50 1"),
                            ("cross", cr, z_top1(), "0.88 0.55 0.30 1")):
        mm = m.copy()
        mm.apply_translation([0.0, 0.0, z])
        items.append((mm, col, f"ati_c_{name}"))
    ev = rec["eyes"]
    e0 = np.array(ev[0]["p_M"])
    g = np.array([0.5, 0.0, math.sqrt(0.75)])                     # the tool 30 deg from vertical, the +x arm 30 deg down
    hg = holder_geometry()
    L = PIGTAIL_MM
    line = trimesh.creation.cylinder(radius=0.6, segment=[e0, e0 + g * L])
    zax = np.array([0.0, 0.0, 1.0])
    v = np.cross(zax, -g)
    Rm = np.eye(4)
    Rm[:3, :3] = np.eye(3) + _skew(v) + _skew(v) @ _skew(v) / (1.0 + zax @ -g)  # the holder's z -> up, against gravity
    Rm[:3, 3] = e0 + g * (L + hg["seat_mm"])                      # its tie hole at the line's end
    items.append((line, "0.15 0.15 0.15 1", "ati_c_line"))
    for nm, mm, col in holder_meshes():
        mm.apply_transform(Rm)
        items.append((mm, col, f"ati_c_{nm}"))
    Wd = np.eye(4)                                                # display: gravity straight down the picture
    a = -g
    v = np.cross(a, [0.0, 0.0, 1.0])
    Wd[:3, :3] = np.eye(3) + _skew(v) + _skew(v) @ _skew(v) / (1.0 + a @ [0.0, 0.0, 1.0])   # -g -> +z_W (up)
    for k, (mm, col, stem) in enumerate(items):
        mm.apply_transform(Wd)
    names = {"+x": "Eye on the +x arm, 80 mm out: a 1.2 mm hole in a 1.6 mm wall",
             "+z": "Eye on the post, 118 mm above the top", "-y": "Eye on the -y arm (four arms alike)"}
    for e in ev:
        if e["name"] in names:
            labels.append((names[e["name"]], (Wd @ np.r_[e["p_M"], 1.0])[:3]))
    labels.append((f"The 100 g weight in its holder, tied {L:.0f} mm below the eye",
                   (Wd @ np.r_[e0 + g * (L + hg["seat_mm"] - 20.0), 1.0])[:3]))
    labels.append(("Hook cross on the top's inserts, in the fork's place",
                   (Wd @ np.r_[[0.0, -40.0, z_top1() + 8.0], 1.0])[:3]))
    ctr = np.mean([mm.bounds.mean(axis=0) for mm, _, _ in items], axis=0)
    cam = (ctr.tolist(), 720.0, 235.0, 18.0)
    if "cross" in which:
        dst = UM._render(items, cam, OUT / "media/20261023-ati_hook_cross.png")
        UM._annotate(dst, cam, labels, title="Calibration with a weighed mass: the hook cross on the ATI")
    if "holder" in which:
        render_holder()
    print("figures in", OUT / "media")


def weight_mesh():
    """The owner's weight as measured: the body and the knob as cylinders."""
    import trimesh
    parts = [trimesh.creation.cylinder(radius=WEIGHT_D / 2, height=WEIGHT_H, sections=96),
             trimesh.creation.cylinder(radius=KNOB[0] / 2, height=KNOB[1], sections=64)]
    for p, z in zip(parts, (WEIGHT_H / 2, WEIGHT_H + KNOB[1] / 2)):
        p.apply_translation([0.0, 0.0, z])
    return trimesh.util.concatenate(parts)


def screw_mesh(length):
    import trimesh
    sh = trimesh.creation.cylinder(radius=1.5, height=length, sections=32)
    sh.apply_translation([0.0, 0.0, -length / 2])
    hd = trimesh.creation.cylinder(radius=2.75, height=3.0, sections=32)
    hd.apply_translation([0.0, 0.0, 1.5])
    return trimesh.util.concatenate([sh, hd])


def holder_meshes(explode=0.0, section=False):
    """The holder as it hangs (z up, the cup's underside at 0): cup, weight, cap and the two screws, moved apart along z by
    explode; with section, the printed parts cut in half on the tab's plane."""
    import trimesh
    src = (lambda nm: HALVES / f"{nm}_half.stl") if section else (lambda nm: OUT / f"cad/20261023-ati_{nm}.stl")
    cup = trimesh.load(src("mass_cup"), force="mesh")
    cap = trimesh.load(src("mass_cap"), force="mesh")
    z_cap = cup_h() + HOLD_PROUD
    w = weight_mesh()
    w.apply_translation([0.0, 0.0, HOLD_FLOOR + explode])
    cap.apply_translation([0.0, 0.0, z_cap + 2.0 * explode])
    out = [("mass_cup", cup, "0.55 0.78 0.50 1"), ("mass_weight", w, "0.62 0.64 0.68 1"), ("mass_cap", cap, "0.55 0.78 0.50 1")]
    for k, a in enumerate((0.0, 180.0)):
        s = screw_mesh(12.0)
        s.apply_translation([*xy(EAR_R, a), z_cap + FLANGE_T + 3.0 * explode])
        out.append((f"mass_screw{k}", s, "0.25 0.25 0.27 1"))
    return out


def render_holder():
    """The holder exploded, its printed parts cut in half on the tab's plane."""
    ex = 34.0
    items = [(m, col, f"holder_{nm}") for nm, m, col in holder_meshes(explode=ex, section=True)]
    z_cap = cup_h() + HOLD_PROUD + 2.0 * ex
    top = z_cap + cap_top()
    labels = [("Cup: 25.6 mm bore, three crush ribs to 24.8 mm, an 8 mm hole in the floor", [-HOLD_BORE / 2 - 1.0, 0.0, 12.0]),
              ("The weight as measured: body 25 x 25 mm, knob 14 mm across and 11 mm tall", [-WEIGHT_D / 2, -6.0, ex + 14.0]),
              ("Cap: 4 mm flange on the weight, knob pocket 14.6 mm across and 11.5 mm deep", [-kd_r() - 1.0, 0.0, z_cap + 6.0]),
              ("Tie hole 2 mm across on the axis: the line from the eye is tied through it", [-TAB_W / 2, 0.0, top + TIE_HOLE_Z]),
              ("Two M3 x 12 into 2.75 mm holes in the cup", [EAR_R, 0.0, cup_h() + HOLD_PROUD + FLANGE_T + 3.0 * ex - 6.0])]
    ctr = np.mean([m.bounds.mean(axis=0) for m, _, _ in items], axis=0)
    cam = (ctr.tolist(), 470.0, -72.0, -12.0)
    dst = UM._render(items, cam, OUT / "media/20261023-ati_mass_holder.png")
    UM._annotate(dst, cam, labels, title="Holder for the 100 g weight, exploded and cut through its axis")


def kd_r():
    return KNOB_POCKET[0] / 2 + DOME_WALL


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", nargs="?", default="build", choices=("build", "render", "holder"))
    ap.add_argument("--clock", type=float, default=None)
    ap.add_argument("--sensor", default="mini45", choices=sorted(SENSORS))
    ap.add_argument("--figures", default="stack,cross,holder", help="render: which figures")
    a = ap.parse_args()
    S = SENSORS[a.sensor]
    if a.cmd == "render":
        render(tuple(a.figures.split(",")))
    else:
        holder_build() if a.cmd == "holder" else main(a.clock, a.sensor)
