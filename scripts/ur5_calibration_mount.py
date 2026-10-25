#!/usr/bin/env python3
"""UR5e calibration mount for the DIP link with the ReSkin 5X board (hand-off docs/handoff/20261018-ur5_calibration_mount.md).

The mount holds a DIP link as printed (the CAD hand's original link, or v2 a or b of the DIP study: all three keep the clevis
arms of assets/cad_v1/fingertip_mount_DIP_link.stl) on the tool side of an OnRobot HEX-E force-torque sensor on a UR5e, so the
five magnetometers can be calibrated against the HEX-E while the robot presses the 85A tip on a fixed indenter. Three PLA
parts: an adapter disc (the ISO 9409-1-50-4-M6 interface: spigot, dowel, four M6 in counterbores), a standoff-fork (foot,
column, and a fork that fills the clevis: an inner block between the arms, a key in the horn slot, ears for the clamp screws,
a cable slot), and a clamp plate on the horn-side arm. Screws are M3 into 2.75 mm holes in PLA (self-forming, owner) and M6
into the flange; in v3 since 2026-10-22 the foot's four M3 go into brass heat-set inserts in the adapter. Two optional table
indenters (a 5 mm sphere and the 25 mm rod).

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
    $RG $E scripts/ur5_calibration_mount.py payload       # mass and centre of gravity past the flange -> payload.json
    $RG $E scripts/ur5_calibration_mount.py clearance     # v4 on: the lab plans' clearances with this mount -> clearance.json

UR5M_REV picks the revision (v1 when unset). v5 (2026-10-23) is v4's fork as printed with a PLA wedge in the stop plate's
gap: build writes stop_wedge (seated at the nominal gap) and stop_wedge_swept (all its seats, for the clearance checks);
check adds the wedge's seats, overlaps, path in, friction and the plate's bending; its print plate is
scripts/v3b_ur5v2_pla_plate.py mount_v5_wedge.
v7 (2026-10-25) has a rigid back below the dorsal ear with its face on the tongue and v3's leg walls; build adds
v7_checks (the link's way in, the back's stiffness) to parts.json.
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
# UR5M_REV=v3: the revision of 2026-10-21 for M6 x 20 screws (owner, 17:54: "i only have 20mm long m6 screws. so we need to
# pop everything up a little. there is not much thread on the hex e ft sensor ... make the holes accessible from the top
# plate ... i can pre-place the screws through the dip link interface ... should still be able to tighten with a ball-end
# allen key"): the adapter 15 mm thick, the heads captive in pockets of a 13 mm foot with key holes from above, everything
# above the foot 11 mm higher than v2. 2026-10-22: v3's adapter takes brass heat-set inserts for the foot's M3 (m3_seat
# "insert"; `inserts` checks the bores); UR5M_ONLY=adapter rebuilds that part alone.
REV = os.environ.get("UR5M_REV", "v1")
V2UP = REV in ("v2", "v3", "v4", "v5", "v6", "v7")
V3UP = REV in ("v3", "v4", "v5", "v6", "v7")   # v4 (2026-10-23) is v3 with a stop behind the DIP link's tongue (stop_* in its P block);
#                                    v5 (2026-10-23) is v4's fork as printed with a PLA wedge in place of the set screws
DATE = {"v1": "20261018", "v2": "20261020", "v3": "20261021", "v4": "20261023", "v5": "20261023", "v6": "20261023", "v7": "20261025"}[REV]
PFX = {"v1": "20261018-ur5_calibration_mount", "v2": "20261020-ur5_calibration_mount_v2",
       "v3": "20261021-ur5_calibration_mount_v3", "v4": "20261023-ur5_calibration_mount_v4",
       "v5": "20261023-ur5_calibration_mount_v5", "v6": "20261023-ur5_calibration_mount_v6",
       "v7": "20261025-ur5_calibration_mount_v7"}[REV]
OUT = REPO / "docs/experiments" / PFX
CAD, MEDIA, PRINT = OUT / "cad", OUT / "media", OUT / "print"
WORK = REPO / "logs" / PFX
DL = REPO / "logs/20261018-ur5_calibration_mount/downloads"   # the data sheets of `interfaces`
DIP = REPO / "docs/experiments/20261018-dip_link_board_mount"
V2 = DIP / "v2"

# ------------------------------------------------------------------------------------------------ frame and placement
PIP_ZF = -51.55                 # z_F of the PIP axis
PIP_M = 57.0 if V3UP else 46.0   # PIP axis above the flange face (z_M): v2 boards' nearest edge >= 62 mm, chips >= 74
#                                         mm; v3 11 mm higher (the foot's top at z_M 28 in place of 17)
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
if V2UP:
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
        # homing (owner, 2026-10-20 18:55: "the slot and shaft shouldnt be round, this leads to orientation uncertainty. we
        # need to be able to clock the finger exactly"): a rectangular socket in the column's palmar face and a key on each
        # indenter's post, every face drafted home_draft so the key seats on its faces; the key's ends home_end_relief short
        # of the socket's, so its long faces (y_F) always seat: they fix y and the turn about the pin's axis
        home_key=(8.0, 18.0),       # socket mouth: width along y_F, length along the tool axis (mm)
        home_draft=20.0, home_end_relief=0.15,
        home_depth=5.3, home_floor_gap=0.8, home_key_below=1.0, home_zm=38.0, post_margin=3.0,
        fork_style="mold",          # "mold" (owner, 18:55: "i thought you should try adding a perimeter volume so that it also
        #                             'clocks' the dip link. the set screw i again am worried could affix the dip link
        #                             incorrectly"): the ears fork with walls around both legs' outlines and no set screw;
        #                             "ears": v1's fork, slid on along the tool axis, with the clamp plate, no key, the ears an
        #                             outside mold on the link's base (owner, 15:40: "what the hell happened to the standoff
        #                             fork? i have no idea how the dip link is supposed to mount to this now"); "cradle": the
        #                             U-channel on the tongue of 2026-10-20 14:30, the link going in along x
        ear_gap=0.2,                # the ears' inner faces this far from the link's palmar (x_F -10.55) and dorsal (9.45) faces
        ear_z_low=-67.5,            # the ears reach 0.2 below the bridge's underside
        ib_r_b=13.40,               # inner block's arc about the PIP axis: 0.05 above b's bridge (13.45), the z stop
        push_y=-8.0, push_z=-65.5,  # M3 set screw through the dorsal ear onto the bridge's back (fork_style "ears" only)
        post_dy=30.0, home_dz=5.0,  # at homing the contact point stands post_dy to the side of the indenter and home_dz above it
        mold_gap=0.15,              # fork_style "mold": its walls this far from each leg's outline in the leg's plane (x-z),
        #                             the outline swept along the tool axis, so the link still slides on
        mold_short=0.2,             # the walls stop this far short of each leg's outer face: the clamp plate and the slotted
        #                             arm's screw press the legs onto the inner block (the y datum), not onto the walls
        mold_close=3.2,             # notches in a leg's outline narrower than twice this are closed first (the horn slot,
        #                             5.0-5.8 mm: no key in it)
        mold_z=(-67.5, -40.0),      # z_F range of the walls
    )
if V3UP:
    P.update(
        ad_t=15.0,                  # an M6 x 20 seated on the adapter's top face stands 5.0 mm out of its face into the HEX-E
        m6_seat="top",              # the head on the adapter's top face (no counterbore), the hole's mouth chamfered
        m6_len=20.0, m6_head_d=10.0, m6_head_k=6.0, m6_ch=0.5,   # ISO 4762 M6 x 20, 5 mm hex socket
        foot_t=13.0, foot_d=70.0,   # the foot z_M 15..28, 70 mm across: 3.75 mm of wall outside the pockets (the adapter
        #                             stays 63 mm on the HEX-E's face; the HEX-E is 72 mm across)
        m6_pocket_d=12.5,           # each head in a pocket of the foot, open toward the adapter (a DIN 125 washer fits)
        m6_pocket_top=26.5,         # the head's top rises to z_M 26 when the screw's tip is flush with the adapter's face
        key_d=7.0,                  # holes for a 5 mm hex key (5.77 mm across corners) above each pocket: the heads stay captive
        key_tilt=20.0,              # the two screws under the column's corners (y_M -17.68): a ball-end key this far from the
        #                             tool axis, leaning out along x_M; the two at y_M +17.68 take a straight key
        key_sweep=(21.0, 26.0),     # z_M of the head's top, seated without a washer .. lifted: the key reaches it anywhere between
        m3_angles=(0.0, 90.0, 198.0, 270.0),     # foot screws clear of the pockets and the cable notch; one way round
        m3_cb_d=6.5, m3_under=4.0, m3_depth=9.0,  # M3 x 12: counterbored, 4 mm of foot under the head, 8 mm into the adapter
        home_zm=49.0,               # the homing socket 11 mm up with the rest: the indenters' posts fit unchanged
        tie_zm=(34.0, 46.0),
        # 2026-10-22 (owner: "let's not go for self-threaded m3 holes there and just put four brass insert bores in there. make
        # sure to size the bore correctly for the brass insert"): the foot's M3 x 12 go into brass heat-set inserts pressed into
        # the adapter's top face. CNC Kitchen's table (poster "Dimensions & Design Guidelines") for its M3 x 5.7 and M3 x 3
        # inserts: 4.6 mm across the knurls, a 4.0 mm hole, walls of at least 1.6 mm, a blind hole at least 1 mm longer than
        # the insert; its guide asks for a straight hole. Inserts of other sizes take other holes: 3.6 mm for the generic
        # M3 x 4 x 5 (4.0 mm across, JLC3DP), 4.4 mm for the Voron M3 x 5 x 4 (5.0 mm across, CNC Kitchen)
        m3_seat="insert",
        ins_hole=4.0, ins_len=5.7, ins_od=4.6,   # the bore, straight; the insert it is sized for (length, across the knurls)
        ins_ch=0.3,                 # 45 deg chamfer at the bore's mouth, which prints on the bed: the first layer's squeeze and
        #                             the plastic the insert pushes up stay below the face the foot sits on
        #                             (m3_depth 9.0 stays: the bore reaches 3.3 mm past the insert and 1.0 mm past the screw's tip)
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
if V3UP:
    SCREWS["M6x20_ISO4762"] = dict(qty=4, what="M6 x 20 socket head cap screw, steel, captive in the foot, into the HEX-E (no "
                                               "washer: 5.0 mm of thread; a DIN 125 washer under the head: 3.4 mm)", mass_g=6.9)
    del SCREWS["M6x12_ISO4762"]
    SCREWS["M3x12_foot"] = dict(qty=4, what="M3 x 12 socket head cap screw, foot into the adapter's heat-set inserts", mass_g=1.2)
    SCREWS["M3_insert"] = dict(qty=4, what="M3 x 5.7 brass heat-set insert, 4.6 mm across the knurls (CNC Kitchen's M3 x 5.7 or "
                                           "the like), in the adapter's 4.0 mm bores", mass_g=0.4)   # mass from its size

if REV in ("v4", "v5"):
    # 2026-10-23 (the owner, through the lab session, after a hand check: the DIP link moves a little in v3's fork when the
    # pad is pressed hard; the UR5 run sphere2 measured 0.33 mm + 0.20 mm/N of travel beyond the FE pad, 0.21 mm/N on the
    # face; docs/experiments/20261022-ur5_tip_calibration/QUEUE.md R16, R18). In v3 the legs alone hold the link: their walls
    # stand 0.15 mm off, so with the clamp loose the link turns 1.0 deg about the PIP axis before it meets the fork (check.json
    # turn_to_contact_deg), 0.64 mm at the contact point 36.4 mm below the axis, and a press on the face turns it that way.
    # v4 adds a stop plate behind the tongue, hanging from the dorsal ear: three brass M3 set screws in heat-set inserts are
    # turned onto the tongue's back face (x_F 8.0) once the link is clamped, then locked with jam nuts, so the normal force
    # of a press on the face or an edge pushes the tongue onto them 5 to 8 mm behind the contact. The screws only touch:
    # the walls and the clamp still place the link. The slotted arm's M3 goes into a brass insert in the inner block.
    P.update(
        stop_y=9.5,                 # the plate's half-width (y_F); the tongue's back face spans |y_F| <= 8.95 below z_F -79
        stop_x=(9.95, 16.0),        # its inner face 0.5 mm off the neck's back (x_F 9.45), which slides past it as the link
        #                             goes on along the tool axis, so 1.95 mm off the tongue's back (x_F 8.0); its outer face
        stop_neck_x=9.95,           # the same face beside the neck; the board's cable stays under x_F 9.30 in its groove
        stop_step_z=-77.6,          # the neck ends at z_F -77.6 (the tongue's back below it at x_F 8.0)
        stop_top_z=-67.0,           # 0.5 mm into the dorsal ear, whose bottom is at ear_z_low -67.5
        stop_bot_z=-92.0,           # the plate's lower end, 2.1 mm below the tip cap's top (z_F -89.86), 1.0 mm off its back
        stop_r=1.0,                 # its vertical edges rounded
        stop_screws=((0.0, -80.5), (-5.0, -87.0), (5.0, -87.0)),   # (y_F, z_F) of the three set screws: a three-point seat
        #                                                            behind chips U2 (z_F -80.9) and U1, U3, U5 (-87.9)
        stop_ins_hole=4.0, stop_ins_len=5.7, stop_ins_ch=0.3,     # CNC Kitchen's M3 x 5.7 bore, pressed from the outer face
        stop_clear=3.4,             # the screw's hole through the rest of the plate
        sl_seat="insert",           # the slotted arm's M3 x 12 into an M3 x 5.7 brass insert in the inner block (v3: a 2.75 mm
        sl_ins_depth=8.5,           # self-forming hole); the bore reaches 1.3 mm past the screw's tip
    )
    SCREWS["M3_stop"] = dict(qty=3, what="M3 x 12 set screw, flat point, brass (A2 if brass is not at hand), with an M3 brass "
                                         "nut as jam nut, turned onto the tongue's back through the stop plate", mass_g=0.9)
    SCREWS["M3_insert_stop"] = dict(qty=4, what="M3 x 5.7 brass heat-set insert (4.6 mm across the knurls): three in the stop "
                                               "plate, one in the inner block for the slotted arm's screw", mass_g=0.4)
if REV == "v5":
    # 2026-10-23 (owner, after sending v4's fork to the P1S: "i dont have any of that hardware except for the heat inserts. we can
    # try devising a constraining v5 that doesnt require so many fasteners. brass hardware is not really stocked in the lab. but
    # i think this rigid stop should work well already"): v4's fork as printed, its three stop bores left empty, and a PLA wedge
    # pushed into the 1.95 mm gap between the stop plate and the tongue's back from the tip end. The wedge's flat face lies on
    # the tongue's back (x_F 8.0, flat over |y_F| <= 10 from z_F -77.05 to the tongue's end at -98.46); its inclined face meets
    # the plate's lower edge (x_F 9.95, z_F -92.0), where the wedge jams, at whatever thickness the gap has. A 10 deg taper over
    # 9.5 mm takes gaps of 1.2 to 2.9 mm (the plate's print, and the link anywhere in its 1 deg of play) and locks by friction
    # (tan 10 deg = 0.18 against 0.4 to 0.8 for PLA on PLA at both faces). No metal near the board's chips.
    P.update(
        wedge_y=9.0,                # half-width (y_F); the plate's inner face is flat over |y_F| <= 8.5, rounded to 9.5
        wedge_t0=1.2,               # thickness at the leading end, mm
        wedge_deg=10.0,             # taper of the inclined face against the flat one
        wedge_len=9.5,              # length of the incline (z_F): 1.2 to 2.875 mm thick
        wedge_fin_len=2.5,          # the fin at the trailing end, pushed by a finger and pulled by its edges (z_F)
        wedge_fin_x=13.5,           # the fin's dorsal face (x_F), inside the plate's outer face (16.0)
    )
    del SCREWS["M3_stop"]
    SCREWS["M3_insert_stop"] = dict(qty=1, what="M3 x 5.7 brass heat-set insert (4.6 mm across the knurls) in the inner block "
                                               "for the slotted arm's screw; the stop plate's three bores stay empty", mass_g=0.4)


if REV == "v6":
    # 2026-10-23 evening (owner, after v5's wedge failed: "the brass inserts were not a good idea at all. all you really needed
    # to do was to make the back surface flush with the fingertip/magnetometer backing ... we could've wiggled it in to be flush
    # + added a fillet ... v3 was almost fine"; "generate a few versions that require a variable amount of squishing in"): v3's
    # fork (no inserts in it, the slotted arm's 2.75 mm self-forming hole) with a thin plate hanging from the dorsal ear whose
    # lower pad presses on the tongue's back with flush_interf of interference. The neck is 1.45 mm proud of the tongue's back
    # and the legs are in their walls when it passes, so the plate bends 1.45 mm + flush_interf as the link slides on: the pad
    # sits low (z_F -90 to -96) on a 2.5 mm plate, where that bend keeps the root's stress near 26 MPa (beam estimate), with a
    # 45 deg lead-in at its lower edge and a fillet at the plate's root on its outer side. flush_set: the printed variants,
    # told apart by 1 to 4 notches in the pad's outer face.
    P.update(
        flush_y=9.5,                # the plate's half-width (y_F), as v4's
        flush_x_in=9.65,            # the plate's inner face above the pad: the dorsal ear's inner face (0.2 mm off the neck)
        flush_h=2.5,                # the plate's thickness (x_F), the part that bends
        flush_top_z=-67.0,          # 0.5 mm into the dorsal ear (its bottom at ear_z_low -67.5)
        flush_pad_z=(-96.0, -90.0), # the pad on the tongue's back (flat at x_F 8.0 from z_F -77.05 to -98.46)
        flush_interf=0.25,          # pad's face at x_F 8.0 - flush_interf
        flush_set=(0.1, 0.25, 0.4, 0.55),
        flush_root_r=3.0,           # fillet between the plate's outer face and the ear's bottom face
    )


def flush_plate(K, p=P):
    """v6: the plate from the dorsal ear and its pad (F), the pad's 45 deg lead-in at its lower edge, the root fillet."""
    x_in, h, yh = p["flush_x_in"], p["flush_h"], p["flush_y"]
    x_out = x_in + h
    zb, zt = p["flush_pad_z"]
    x_f = B_LINK["tongue_back"] - p["flush_interf"]
    upper = K.box((x_in, -yh, zt - 0.01), (x_out, yh, p["flush_top_z"]))
    pad = K.prism_xz([(x_out, zb), (x_out, zt), (x_f, zt), (x_f, zb + (x_in - x_f)), (x_in, zb)], -yh, yh)
    R, z0 = p["flush_root_r"], p["ear_z_low"]
    fillet = K.sub(K.box((x_out - 0.01, -yh, z0 - R), (x_out + R, yh, z0 + 0.01)), K.cyl("y", (x_out + R, z0 - R), R, -yh - 1, yh + 1))
    return [upper, pad, fillet]


def flush_notches(K, p=P):
    """v6: one notch per step of flush_set in the pad's outer face, 0.6 mm deep: the variant's index + 1."""
    x_out = p["flush_x_in"] + p["flush_h"]
    k = list(p["flush_set"]).index(p["flush_interf"]) if p["flush_interf"] in p["flush_set"] else 0
    zb = p["flush_pad_z"][0]
    return [K.box((x_out - 0.6, -4.5 + 3.0 * j - 0.6, zb - 1.0), (x_out + 1.0, -4.5 + 3.0 * j + 0.6, zb + 3.0)) for j in range(k + 1)]


if REV == "v7":
    # 2026-10-25 (owner, after v6: "this lip wont do very much ... we need more flush surface ... as long as there is some gap
    # at the bottom to allow the dip link to snap in at the wider base ... extend down another full 8-10mm ... heavily shore
    # up the back, extending the supporting beams/volume to better prevent flex"): v3's fork with a solid back below the
    # dorsal ear, as thick as the ear and as wide as the fork, whose face lies on the tongue's back from z_F -97.5 to -83.5
    # (v6's pad: 3.9 mm flat), with a relief above it for the neck (1.45 mm proud of the tongue). The legs' walls are v3's
    # (owner, 10-25, after a first v7 with the channels widened palmar: "you can add back all the leg constraints. it slots
    # in fine, and it will slot in fine with the tighter fit").
    P.update(
        back_x=(9.65, 17.0),        # the back: the dorsal ear's inner face (0.2 mm off the neck) to its outer face
        back_taper=(-85.0, 12.0),   # its outer face runs from x_F 17 at z_F -85 to 12 at its bottom
        face_y=10.5,                # the face spans the tongue's width (flat at x_F 8.0 over |y_F| <= 10.5)
        face_z=(-97.5, -83.5),      # the flat face; 45 deg chamfers below (lead-in) and above (to the relief), printable
        face_interf=0.2,            # the face at x_F 8.0 - face_interf
    )


def back_block(K, p=P):
    """v7: the back below the dorsal ear (as wide as the fork, its outer edges rounded as the ears') and its face on the
    tongue's back, with 45 deg chamfers to the relief at both ends (F)."""
    x0, x1 = p["back_x"]
    xf = B_LINK["tongue_back"] - p["face_interf"]
    zb, zt = p["face_z"]
    zch, zbot = zt + (x0 - xf), zb - (x0 - xf)
    zt0, xbot = p["back_taper"]
    fy0, fy1 = p["fp_y"]
    zr = p["ear_z_low"] + 0.01
    back = K.inter(K.prism_xz([(x0, zbot), (xbot, zbot), (x1, zt0), (x1, zr), (x0, zr)], fy0, fy1),
                   K.rbox_z(-p["fp_x"], p["fp_x"], fy0, fy1, zbot - 1, zr + 1, p["fp_r"]))
    face = K.prism_xz([(x0 + 0.5, zbot), (x0 + 0.5, zch), (x0, zch), (xf, zt), (xf, zb), (x0, zbot)], -p["face_y"], p["face_y"])
    return [back, face]


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

    def hull(self, pts):
        """Convex solid of two parallel quadrilaterals pts[:4] and pts[4:] (corners in the same order)."""
        return self.M.Manifold.hull_points([[float(v) for v in q] for q in pts])


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

    def hull(self, pts):
        wires = [self.Part.makePolygon([self.V(q) for q in quad] + [self.V(quad[0])]) for quad in (pts[:4], pts[4:])]
        return self.Part.makeLoft(wires, True)

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
    four holes for the foot's M3 screws: 2.75 mm and self-forming, or (m3_seat "insert", v3 since 2026-10-22) straight bores
    for brass heat-set inserts with a chamfered mouth."""
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
        if p.get("m6_seat") == "top":                      # v3: the head on the top face, the mouth chamfered
            cut.append(K.cyl("z", (x, y), p["m6_hole"] / 2 + p["m6_ch"] + 0.01, z0 - 0.01, z0 + p["m6_ch"], r1=p["m6_hole"] / 2))
        else:
            cut.append(K.cyl("z", (x, y), p["m6_cb_d"] / 2, z0 - 1, zF(p["m6_under"])))
    for a in m3_angles(p):
        x, y = _m_xy(p["m3_r"], a)
        if p.get("m3_seat") == "insert":
            r_ = p["ins_hole"] / 2
            cut.append(K.cyl("z", (x, y), r_, z0 - 1, zF(p["ad_t"] - p["m3_depth"])))
            cut.append(K.cyl("z", (x, y), r_ + p["ins_ch"] + 0.01, z0 - 0.01, z0 + p["ins_ch"], r1=r_))
        else:
            cut.append(K.cyl("z", (x, y), p["m3_pilot"] / 2, z0 - 1, zF(p["ad_t"] - p["m3_depth"])))
    return K.cut_all(s, cut)


def key_poly(p=P, z_top=None):
    """The key's outline in the x-z plane (without its round end): flanks parallel to the slot's, key_flank inside them."""
    z_top = p["hub_z0"] + 0.3 if z_top is None else z_top
    h = lambda z: slot_half(z) - p["key_flank"]                     # noqa: E731
    return [(-h(SLOT_END_Z), SLOT_END_Z), (h(SLOT_END_Z), SLOT_END_Z), (h(z_top), z_top), (-h(z_top), z_top)]


def frustum(K, axis, c, h0, h1, a0, a1):
    """Rectangular frustum along axis from a0 to a1, its half-sizes h0 at a0 and h1 at a1 along the other two coordinates (in
    xyz order), centred on c (those two coordinates)."""
    pts = []
    for a, (hu, hv) in ((a0, h0), (a1, h1)):
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            u, v = c[0] + su * hu, c[1] + sv * hv
            pts.append({"x": (a, u, v), "y": (u, a, v), "z": (u, v, a)}[axis])
    return K.hull(pts)


def home_halves(d, key=False, p=P):
    """Half-sizes (across the finger, along the tool axis) of the homing socket, or of the key seated in it, d mm into the
    column from its palmar face (d < 0: outside the column)."""
    t = math.tan(math.radians(p["home_draft"]))
    w, l_ = p["home_key"]
    return (w / 2 - d * t, l_ / 2 - d * t - (p["home_end_relief"] if key else 0.0))


def homing_socket(K, p=P):
    """The homing socket in F: along x_F from 1 mm outside the column's palmar face to home_depth into it."""
    fx = p["fp_x"]
    return frustum(K, "x", (Y_AXIS, zF(p["home_zm"])), home_halves(-1.0, p=p), home_halves(p["home_depth"], p=p),
                   -fx - 1.0, -fx + p["home_depth"])


LEG_LAYERS = (   # the legs' layers: name, y_F range of the link's sections, which wall face stops short of the leg's outer face
    ("horn_outer", (OUTER[0], OUTER[1]), "y0"),
    ("horn_slot", (SLOT[0], SLOT[1]), None),
    ("horn_inner", (INNER[0], -16.95), None),        # to the inner block's face
    ("slotted", (13.30, ARM_SLOTTED[1]), "y1"),      # from the inner block's face
)


def leg_molds(p=P):
    """fork_style "mold": x-z polygons of the walls around the DIP link's two legs, per layer of the legs (F). In each layer
    the sections of both b links (v2 and v3; their legs are the CAD hand link's) at 0.1 mm steps in y are united with their
    holes filled, the horn slot filled to its walls' top (no key in it) and other notches narrower than 2 mold_close closed,
    the outline swept down along -z_F
    (the link slides on along +z_F, so the walls never meet it on the way), widened by mold_gap and cut from the block
    |x_F| <= fp_x, z_F in mold_z. Cached in cad/<PFX>_leg_molds.json, which the STEP build reads (freecadcmd has neither
    trimesh nor shapely)."""
    path = CAD / f"{PFX}_leg_molds.json"
    key = dict(gap=p["mold_gap"], short=p["mold_short"], close=p["mold_close"], z=list(p["mold_z"]), fp_x=p["fp_x"],
               slot_filled=True)
    if path.exists():
        d = json.loads(path.read_text())
        if d.get("params") == key:
            return d["layers"]
    import trimesh
    from shapely import affinity
    from shapely.geometry import Polygon, box as sbox
    from shapely.ops import unary_union
    srcs = ("v2/b/cad/20261018-dip_link_v2b_dip_link.stl", "v3/b/cad/20261020-dip_link_v3b_dip_link.stl") + (
        ("v4/b/cad/20261021-dip_link_v4b_dip_link.stl",) if V3UP else ())
    links = [trimesh.load(DIP / f, force="mesh") for f in srcs]
    z0, z1 = p["mold_z"]
    drop = (z1 - z0) + 40.0
    layers = []
    for name, ys, short in LEG_LAYERS:
        loops = []
        for m in links:
            for y in np.arange(ys[0] + 0.02, ys[1] - 0.019, 0.1):
                sec = m.section(plane_origin=[0.0, float(y), 0.0], plane_normal=[0.0, 1.0, 0.0])
                if sec is None:
                    continue
                for ent in sec.entities:
                    q = sec.vertices[ent.points][:, [0, 2]]
                    if len(q) >= 4:
                        g = Polygon(q).buffer(0)
                        if g.area > 0.01:
                            loops.append(g)                       # outer loops and holes alike: the union fills the holes
        if name == "horn_slot":                           # the horn slot filled to its walls' top: nothing enters it
            hw = slot_half(SLOT_TOP) + 0.05
            loops.append(sbox(-hw, SLOT_END_Z - SLOT_END_R - 0.05, hw, SLOT_TOP))
        U = unary_union(loops)
        r = p["mold_close"]
        U = U.buffer(r, join_style=1).buffer(-r, join_style=1)
        sweep = []
        for g in (U.geoms if hasattr(U, "geoms") else [U]):
            ext = np.asarray(g.exterior.coords)
            sweep += [Polygon(ext), affinity.translate(Polygon(ext), 0.0, -drop)]
            for a_, b_ in zip(ext[:-1], ext[1:]):
                if abs(b_[0] - a_[0]) > 1e-9:
                    sweep.append(Polygon([tuple(a_), tuple(b_), (b_[0], b_[1] - drop), (a_[0], a_[1] - drop)]).buffer(0))
        S = unary_union(sweep).buffer(p["mold_gap"], join_style=1)
        walls = sbox(-p["fp_x"], z0, p["fp_x"], z1).difference(S)
        polys = []
        for g in (walls.geoms if hasattr(walls, "geoms") else [walls]):
            if g.area < 0.5:
                continue
            g = g.simplify(0.005, preserve_topology=True)
            if len(g.interiors):
                raise RuntimeError(f"{name}: a wall with a hole")
            polys.append(np.round(np.asarray(g.exterior.coords)[:-1], 4).tolist())
        y0, y1 = ys
        if short == "y0":
            y0 = ARM_HORN[0] + p["mold_short"]
        elif short == "y1":
            y1 = ARM_SLOTTED[1] - p["mold_short"]
        layers.append(dict(name=name, sections_y=[round(v, 3) for v in ys], walls_y=[round(y0, 3), round(y1, 3)], polys=polys,
                           area_mm2=round(float(walls.area), 1)))
        print(f"leg mold {name}: walls y {y0:.2f}..{y1:.2f}, {len(polys)} pieces, {walls.area:.1f} mm2", flush=True)
    path.write_text(json.dumps(dict(params=key, layers=layers, sources=[str(Path("docs/experiments/20261018-dip_link_board_mount")
                                                                            / f) for f in srcs],
                                    when=time.strftime("%Y-%m-%d %H:%M")), indent=1) + "\n")
    return layers


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
    cut.append(homing_socket(K, p))                       # homing socket: along x_F into the column's palmar face
    return K.cut_all(s, cut)


def key_channel(K, x_m, y_m, sx, p=P, r=None, length=110.0):
    """v3: the way of a ball-end hex key to the socket of an M6 head under the column's corner, in F. The key's axis passes
    through the socket's centre, key_tilt from the tool axis, leaning out along sx * x_M; the socket's centre lies anywhere
    on the bolt's axis between z_M key_sweep[0] (seated) and key_sweep[1] (lifted), so the channel is the prism of a stadium:
    two cylinders of radius r about the axes through the two ends and the slab between them."""
    th = math.radians(p["key_tilt"])
    r = p["key_d"] / 2 if r is None else r
    d = np.array([sx * math.sin(th), 0.0, math.cos(th)])
    u = np.array([-sx * math.cos(th), 0.0, math.sin(th)])
    w = np.cross(d, u)
    z0, z1 = p["key_sweep"]
    lu = (z1 - z0) * math.sin(th)
    s0, s1 = -3.0, length
    loc = K.union([K.box((0.0, -r, s0), (lu, r, s1)), K.cyl("z", (0.0, 0.0), r, s0, s1), K.cyl("z", (lu, 0.0), r, s0, s1)])
    T = np.eye(4)
    T[:3, :3] = np.column_stack([u, w, d])
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-9
    T[:3, 3] = [x_m, y_m, z0]
    return K.transform(loc, np.linalg.inv(T_M_FROM_F) @ T)


def m6_ways(K, p=P):
    """v3: the cuts in the foot for the four M6 x 20: a pocket around each head from the adapter's face up to m6_pocket_top,
    a vertical key hole above the two heads at y_M > 0 (clear of the column; it also notches the corners of the walls around
    the slotted leg) and a tilted key channel above the two under the column's corners (y_M < 0)."""
    out = []
    for a in bolt_angles(p):
        x, y = _m_xy(p["bolt_r"], a)                        # F
        x_m, y_m = -x, y - Y_AXIS
        out.append(K.cyl("z", (x, y), p["m6_pocket_d"] / 2, zF(p["m6_pocket_top"]), zF(p["ad_t"]) + 1.0))
        if y_m > 0:
            out.append(K.cyl("z", (x, y), p["key_d"] / 2, zF(140.0), zF(p["m6_pocket_top"] - 0.5)))
        else:
            out.append(key_channel(K, x_m, y_m, 1.0 if x_m > 0 else -1.0, p))
    return out


def stop_plate(K, p=P):
    """v4: the plate behind the DIP link's tongue, hanging from the dorsal ear: x_F stop_neck_x..stop_x[1] beside the neck,
    stop_x[0]..stop_x[1] behind the tongue, |y_F| <= stop_y (F)."""
    x0, x1 = p["stop_x"]
    yh, r = p["stop_y"], p["stop_r"]
    upper = K.rbox_z(p["stop_neck_x"], x1, -yh, yh, p["stop_step_z"] - 0.01, p["stop_top_z"], r)
    lower = K.rbox_z(x0, x1, -yh, yh, p["stop_bot_z"], p["stop_step_z"], r)
    return [upper, lower]


def stop_bores(K, p=P):
    """v4: for each set screw an insert bore from the plate's outer face (x_F stop_x[1]), chamfered at its mouth, and a
    clearance hole through the rest of the plate onto the tongue."""
    x0, x1 = p["stop_x"]
    r_, L, ch = p["stop_ins_hole"] / 2, p["stop_ins_len"], p["stop_ins_ch"]
    out = []
    for y, z in p["stop_screws"]:
        out.append(K.cyl("x", (y, z), r_, x1 - L - 0.5, x1 + 1.0))
        out.append(K.cyl("x", (y, z), r_, x1 - ch, x1 + 0.01, r1=r_ + ch + 0.01))
        out.append(K.cyl("x", (y, z), p["stop_clear"] / 2, x0 - 1.0, x1 - L - 0.49))
    return out


def wedge_seat(t_edge, p=P):
    """v5: z_F of the wedge's leading end when its thickness at the plate's lower edge is t_edge (mm)."""
    return p["stop_bot_z"] + (t_edge - p["wedge_t0"]) / math.tan(math.radians(p["wedge_deg"]))


def stop_wedge(K, p=P, t_edge=None, lift=0.0):
    """v5: the wedge in F, its flat face lift above the tongue's back (x_F 8.0), seated where its inclined face meets the
    plate's lower edge (x_F stop_x[0], z_F stop_bot_z); t_edge None: at the gap of the nominal link (1.95 mm less lift).
    The profile in x-z (leading end toward the PIP axis, the fin at the tip end) is extruded over |y_F| <= wedge_y."""
    xb = B_LINK["tongue_back"] + lift
    t_edge = p["stop_x"][0] - xb if t_edge is None else t_edge
    tan_a = math.tan(math.radians(p["wedge_deg"]))
    t0, L, fl = p["wedge_t0"], p["wedge_len"], p["wedge_fin_len"]
    z0 = wedge_seat(t_edge, p)
    zh = z0 - L
    poly = [(xb, z0), (xb + t0, z0), (xb + t0 + L * tan_a, zh), (p["wedge_fin_x"], zh), (p["wedge_fin_x"], zh - fl), (xb, zh - fl)]
    return K.prism_xz(poly, -p["wedge_y"], p["wedge_y"])


def standoff_fork_v2_ears(K, p=P):
    """v2, fork_style "ears" and "mold": v1's foot, column, hub and inner block, the DIP link sliding on along the tool axis
    between the two ears as in v1, without the key: the ears' inner faces ear_gap from the bridge's palmar and dorsal faces
    down to below the bridge, the inner block's arc 0.05 mm above b's bridge (the z stop), and the homing socket. "ears" adds
    an M3 set screw through the dorsal ear that pushes the bridge onto the palmar ear; "mold" adds walls mold_gap around
    both legs' outlines (leg_molds), which locate the link in x and in its turns, and has no set screw."""
    c = (0.0, Y_AXIS)
    fx, (fy0, fy1), r = p["fp_x"], p["fp_y"], p["fp_r"]
    z_foot0, z_foot1 = zF(p["ad_t"] + p["foot_t"]), zF(p["ad_t"])
    foot = K.cyl("z", c, p.get("foot_d", p["ad_d"]) / 2, z_foot0, z_foot1)
    top = K.rbox_z(-fx, fx, fy0, fy1, p["hub_z0"], z_foot0 + 0.01, r)
    xp, xd = -10.55 - p["ear_gap"], 9.45 + p["ear_gap"]
    ears = K.sub(K.rbox_z(-fx, fx, fy0, fy1, p["ear_z_low"], p["hub_z0"] + 0.01, r),
                 K.box((xp, fy0 - 1, p["ear_z_low"] - 1), (xd, fy1 + 1, p["hub_z0"] + 1)))
    ib = K.inter(K.box((-p["ib_x"] - 0.01, p["ib_y"][0], p["ear_z_low"]), (p["ib_x"] + 0.01, p["ib_y"][1], p["hub_z0"] + 0.01)),
                 K.cyl("y", (0.0, PIP_ZF), p["ib_r_b"], p["ib_y"][0] - 1, p["ib_y"][1] + 1))
    mold = p.get("fork_style") == "mold"
    # each layer's walls reach slightly into the next layer, whose walls cover that region too (the outer layer's lie inside
    # the slot layer's, the slot layer's inside the inner layer's), and into the inner block: no coincident faces
    ext = {"horn_outer": (0.0, 0.02), "horn_slot": (0.0, 0.02), "horn_inner": (0.0, 0.05), "slotted": (-0.05, 0.0)}
    walls = [K.prism_xz(poly, ly["walls_y"][0] + ext[ly["name"]][0], ly["walls_y"][1] + ext[ly["name"]][1])
             for ly in leg_molds(p) for poly in ly["polys"]] if mold else []
    stop = ((stop_plate(K, p) if "stop_x" in p else []) + (flush_plate(K, p) if "flush_h" in p else [])
            + (back_block(K, p) if "back_x" in p else []))
    s = K.union([foot, top, ears, ib] + walls + stop)
    cut = []
    cy = p["cab_y"]
    # the cable slot through the dorsal ear; with the stop plate below the ear it ends at the ear's bottom
    cut.append(K.box((p["cab_x0"], -cy, p["ear_z_low"] - (0.01 if stop else 2)), (fx + 1, cy, p["cab_top"])))
    if "stop_x" in p:
        cut += stop_bores(K, p)
    if "flush_h" in p:
        cut += flush_notches(K, p)
    cut.append(K.box((p["groove_x0"], -cy, p["cab_top"] - 0.01), (fx + 1, cy, z_foot1 + 1)))
    cut.append(K.box((p["groove_x0"], -cy, z_foot0 - 0.01), (p["ad_d"], cy, z_foot1 + 1)))
    for zm in p["tie_zm"]:
        z = zF(zm)
        za, zb = z - p["tie_h"] / 2, z + p["tie_h"] / 2
        cut.append(K.box((p["tie_x"][0], -p["tie_side"][1], za), (p["tie_x"][1], p["tie_side"][1], zb)))
        for sgn in (-1, 1):
            ya, yb = sorted((sgn * p["tie_side"][0], sgn * p["tie_side"][1]))
            cut.append(K.box((p["tie_x"][0], ya, za), (fx + 1, yb, zb)))
    ys0 = min(fy0, ARM_HORN[0] + p["mold_short"]) if mold else fy0
    for sx in (-1, 1):                                   # the clamp plate's screws into the ears (through the horn walls)
        cut.append(K.cyl("y", (sx * p["cp_screw_x"], p["cp_screw_z"]), p["m3_pilot"] / 2, ys0 - 1, fy0 + p["ear_hole_depth"]))
    if p.get("sl_seat") == "insert":                     # v4: a brass insert for the slotted arm's screw, chamfered mouth
        r_, y1 = p["stop_ins_hole"] / 2, p["ib_y"][1]
        cut.append(K.cyl("y", (0.0, PIP_ZF), r_, y1 - p["sl_ins_depth"], y1 + 1))
        cut.append(K.cyl("y", (0.0, PIP_ZF), r_, y1 - p["stop_ins_ch"], y1 + 0.01, r1=r_ + p["stop_ins_ch"] + 0.01))
    else:
        cut.append(K.cyl("y", (0.0, PIP_ZF), p["m3_pilot"] / 2, p["ib_y"][1] - p["sl_hole_depth"], p["ib_y"][1] + 1))
    for a in m3_angles(p):
        x, y = _m_xy(p["m3_r"], a)
        cut.append(K.cyl("z", (x, y), p["m3_clear"] / 2, z_foot0 - 1, z_foot1 + 1))
        if "m3_cb_d" in p:                               # v3: counterbored, m3_under of foot under the head
            cut.append(K.cyl("z", (x, y), p["m3_cb_d"] / 2, z_foot0 - 1, zF(p["ad_t"] + p["m3_under"])))
    if "m6_pocket_d" in p:                               # v3: the M6 heads captive in the foot, keys from above
        cut += m6_ways(K, p)
    if not mold:                                         # the set screw of "ears"
        cut.append(K.cyl("x", (p["push_y"], p["push_z"]), p["m3_pilot"] / 2, xd - 0.5, fx + 1.0))
    cut.append(homing_socket(K, p))
    return K.cut_all(s, cut)


def standoff_fork(K, p=P):
    """Foot, column and fork in one part."""
    if V2UP:
        return standoff_fork_v2_ears(K, p) if p.get("fork_style") in ("ears", "mold") else standoff_fork_v2(K, p)
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
    if not V2UP:
        return K.union([base, top])
    # v2: the homing post (owner, 2026-10-20: "add homing geometry so i can zero the ur5 mount to the indenter. dont
    # overthink that one. it can literally be a slot/rod mating"), its pin seated in the mount's socket when the tip's
    # contact point stands home_dz above the indenter's top and post_dy to the side (the table frame: x along the tool axis
    # toward the tip, z up)
    # 18:55 the key: a frustum, its long side along the tool axis (x), on a block home_key_below under the socket's mouth
    h = homing(kind)
    xp, yp, zb, zt = h["post_xy"][0], h["post_xy"][1], h["body_top_z"], h["pin_top_z"]
    k0 = home_halves(-P["home_key_below"] - 0.01, key=True)[::-1]           # (along x, along y) at the block's top
    k1 = home_halves(P["home_depth"] - P["home_floor_gap"], key=True)[::-1]  # at the key's top
    bx, by = k0[0] + P["post_margin"], k0[1] + P["post_margin"]
    arm = [K.box((xp - bx, -12.0, 0.0), (-34.99, 12.0, 6.0)), K.box((xp - bx, 11.99, 0.0), (xp + bx, yp + by, 6.0))]
    post = [K.rbox_z(xp - bx, xp + bx, yp - by, yp + by, 5.99, zb, 2.0), frustum(K, "z", (xp, yp), k0, k1, zb - 0.01, zt)]
    return K.union([base, top] + arm + post)


TIP_TOP = {"sphere": 46.0, "rod": 42.5}         # the indenters' tops above the table, mm


def homing(kind):
    """v2: where the homing post stands on an indenter's base and the move from homing to the contact. With the palmar side
    down and the tool axis along the base's +x (M -> table: z_M -> x, y_M -> y, x_M -> -z), the socket's mouth on the
    column's palmar face lies d from the tip's contact point; at homing the contact point stands post_dy to +y and home_dz
    above the indenter's top, so the mouth is at that point + d. The key's long faces seat where its section equals the
    socket's, at the mouth's plane; its top stays home_floor_gap short of the socket's floor, its block home_key_below under
    the mouth, and its ends home_end_relief inside the socket's."""
    a_M = F_to_M(np.array([-10.55, 0.0, B_LINK["z_contact"]]))
    m_M = F_to_M(np.array([-P["fp_x"], Y_AXIS, zF(P["home_zm"])]))
    s_M = F_to_M(np.array([-P["fp_x"] + P["home_depth"], Y_AXIS, zF(P["home_zm"])]))
    tab = lambda v: np.array([v[2], v[1], -v[0]])                               # noqa: E731
    tip = np.array([0.0, 0.0, TIP_TOP[kind]])
    contact_home = tip + np.array([0.0, P["post_dy"], P["home_dz"]])
    mouth = contact_home + tab(m_M - a_M)
    w, l_ = P["home_key"]
    return dict(contact_M=np.round(a_M, 3).tolist(), socket_mouth_M=np.round(m_M, 3).tolist(), socket_floor_M=np.round(s_M, 3).tolist(),
                contact_to_mouth_table=np.round(tab(m_M - a_M), 3).tolist(),
                post_xy=[round(float(mouth[0]), 3), round(float(mouth[1]), 3)], mouth_z=round(float(mouth[2]), 3),
                pin_top_z=round(float(mouth[2] + P["home_depth"] - P["home_floor_gap"]), 3),
                body_top_z=round(float(mouth[2] - P["home_key_below"]), 3),
                socket_mouth_mm=[w, l_], key_at_mouth_mm=[w, round(l_ - 2 * P["home_end_relief"], 3)], draft_deg=P["home_draft"],
                contact_at_homing_table=np.round(contact_home, 3).tolist(),
                move_to_touch_table=[0.0, -P["post_dy"], -P["home_dz"]], indenter_top_z=TIP_TOP[kind])


PARTS = (("adapter", "standoff_fork", "clamp_plate") if REV == "v1" or P.get("fork_style") in ("ears", "mold")
         else ("adapter", "standoff_fork")) + (("stop_wedge",) if "wedge_t0" in P else ())
TABLE_PARTS = ("indenter_sphere", "indenter_rod")
MOLD = V2UP and P.get("fork_style") == "mold"


ONLY = tuple(s for s in os.environ.get("UR5M_ONLY", "").split(",") if s)   # e.g. UR5M_ONLY=adapter: build and STEP these alone,
#                                                                           the other parts' files and records stay as they are


def build_parts(K, p=P):
    """The printed parts in M (and the indenters in their table frame)."""
    out = {}
    for nm, fn in (("adapter", adapter), ("standoff_fork", standoff_fork), ("clamp_plate", clamp_plate),
                   ("stop_wedge", stop_wedge)):     # v5: the wedge seated at the nominal link's gap
        if nm in PARTS and (not ONLY or nm in ONLY):
            out[nm] = K.transform(fn(K, p), T_M_FROM_F)
    for kind in ("sphere", "rod"):
        if not ONLY or f"indenter_{kind}" in ONLY:
            out[f"indenter_{kind}"] = indenter(K, kind)
    return out


def _merge_parts(path, rec):
    """With UR5M_ONLY, the records of the parts not rebuilt are kept from the file."""
    if ONLY and path.exists():
        return dict(json.loads(path.read_text())["parts"], **rec)
    return rec


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


def v7_checks(K):
    """v7: overlap volumes (mm3) of the fork (F) with the DIP link and with its cover, pad and cap together, the assembly
    lowered d mm along the tool axis. Beam estimates of the stiffness at the contact (E 2 GPa, solid sections, a cantilever
    from the hub's bottom through the dorsal ear without its cable slot) of v7's back and of v6's plate (load at its pad's
    middle)."""
    import trimesh
    p = P
    fork = standoff_fork(K, p)
    v4b = DIP / "v4/b/cad"
    link = from_trimesh(trimesh.load(v4b / "20261021-dip_link_v4b_dip_link.stl", force="mesh"))
    rest = K.union([from_trimesh(trimesh.load(v4b / f, force="mesh")) for f in (
        "20261022-dip_link_v4b_cover_I15H3_3S.stl", "20261021-dip_link_v4b_pad_block5.stl", "20261022-dip_link_v4b_tip_cap_TW15H3S.stl")])
    rows = []
    for d in np.round(np.arange(0.0, 26.01, 0.25), 2):
        v = [float(K.inter(fork, m.translate([0.0, 0.0, -d])).volume()) for m in (link, rest)]
        rows.append([float(d), round(v[0], 2), round(v[1], 2)])
    out = dict(path=rows, columns=["d_mm", "link_mm3", "cover_pad_cap_mm3"])
    hit = [r for r in rows if r[1] > 0.05 or r[2] > 0.05]
    print(f"v7 path: overlap at d {[r[0] for r in hit][:1]}..{[r[0] for r in hit][-1:]} mm, link max "
          f"{max(r[1] for r in rows):.1f} mm3, cover/pad/cap max {max(r[2] for r in rows):.2f} mm3; at d 0: {rows[0][1:]}",
          flush=True)

    def I_of(rects):              # (x0, x1, width): bending with the deflection along x
        A = sum((b - a) * w for a, b, w in rects)
        xc = sum((b - a) * w * (a + b) / 2 for a, b, w in rects) / A
        return sum(w * (b - a) ** 3 / 12 + (b - a) * w * ((a + b) / 2 - xc) ** 2 for a, b, w in rects)

    def k_at(I, z_load, z_root=p["hub_z0"], E=2000.0, n=600):
        a = z_root - z_load
        u = np.linspace(0.0, a, n)
        trap = getattr(np, "trapezoid", None) or np.trapz
        return 1.0 / trap((a - u) ** 2 / (E * np.array([I(z_root - x) for x in u])), u)

    w_ear, w_fork = p["fp_y"][1] - p["fp_y"][0] - 2 * p["cab_y"], p["fp_y"][1] - p["fp_y"][0]
    x0, x1 = p["back_x"]
    xf = B_LINK["tongue_back"] - p["face_interf"]
    zb, zt = p["face_z"]
    zch = zt + (x0 - xf)
    zt0, xbot = p["back_taper"]
    zbot = zb - (x0 - xf)

    def I7(z):
        if z > p["ear_z_low"]:
            return I_of([(x0, x1, w_ear)])
        xo = x1 if z >= zt0 else xbot + (x1 - xbot) * (z - zbot) / (zt0 - zbot)
        r = [(x0, xo, w_fork)]
        if z < zch:
            r.append((max(xf, x0 - (zch - z)), x0, 2 * p["face_y"]))
        return I_of(r)

    def I6(z):                    # v6: the plate x_F 9.65..12.15, |y_F| <= 9.5, below the ear
        return I_of([(x0, x1, w_ear)]) if z > p["ear_z_low"] else I_of([(9.65, 12.15, 19.0)])
    zc = B_LINK["z_contact"]
    out["beam"] = dict(v7_at_contact_N_per_mm=round(k_at(I7, zc), 1), v7_at_face_bottom_N_per_mm=round(k_at(I7, zb), 1),
                       v6_at_pad_N_per_mm=round(k_at(I6, -92.05), 1),
                       note="E 2 GPa, solid sections, cantilever from the hub's bottom (z_F -53.2) through the dorsal ear "
                            "(cable slot removed); v7 at the contact z_F -87.91 and the face's lower end; v6 at its pad's "
                            "flat middle")
    print("v7 beam", out["beam"], flush=True)
    return out


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
    if "flush_set" in P and (not ONLY or "standoff_fork" in ONLY):     # v6: one fork per interference
        E_, b_, h_ = 2000.0, 2 * P["flush_y"], P["flush_h"]
        I_ = b_ * h_ ** 3 / 12
        zb, zt = P["flush_pad_z"]
        L_top, L_mid = P["ear_z_low"] - zt, P["ear_z_low"] - 0.5 * (zt + zb + (P["flush_x_in"] - 8.0))
        for i in P["flush_set"]:
            s = K.transform(standoff_fork(K, dict(P, flush_interf=i)), T_M_FROM_F)
            tm = to_trimesh(s)
            nm = f"standoff_fork_i{round(i * 100):02d}"
            tm.export(fname(nm))
            d_in = B_LINK["neck_back"] - B_LINK["tongue_back"] + i          # the pad's bend as the neck passes it
            rec[nm] = dict(file=rel(fname(nm)), volume_mm3=round(float(s.volume()), 1), watertight=bool(tm.is_watertight),
                           interference_mm=i, notches=list(P["flush_set"]).index(i) + 1,
                           preload_N=round(3 * E_ * I_ / L_mid ** 3 * i, 1),
                           push_on_N=round(3 * E_ * I_ / L_top ** 3 * d_in, 1),
                           root_stress_MPa=round(3 * E_ * h_ * d_in / (2 * L_top ** 2), 1),
                           note="beam estimates, E 2 GPa: preload at the flush face's middle; push-on force and root stress "
                                "with the neck on the pad's top edge")
            print(nm, rec[nm], flush=True)
    if "back_x" in P and (not ONLY or "standoff_fork" in ONLY):      # v7: the link's way in and the back's stiffness
        rec["v7_checks"] = v7_checks(K)
    if "stop_wedge" in parts:                            # v5: every seat the wedge can take, for the clearance checks
        t1 = P["wedge_t0"] + P["wedge_len"] * math.tan(math.radians(P["wedge_deg"]))
        sw = K.union([stop_wedge(K, P, t_edge=t) for t in np.linspace(P["wedge_t0"], t1, 9)])
        tm = to_trimesh(K.transform(sw, T_M_FROM_F))
        dst = fname("stop_wedge_swept")
        tm.export(dst)
        rec["stop_wedge_swept"] = dict(file=rel(dst), volume_mm3=round(float(sw.volume()), 1),
                                       bounds=np.round(tm.bounds, 3).tolist(), watertight=bool(tm.is_watertight), frame="M",
                                       note=f"the wedge at nine seats, thickness at the plate's edge {P['wedge_t0']:.2f} to "
                                            f"{t1:.3f} mm: the space it can take, for the clearance checks; not a part")
    rec = _merge_parts(OUT / f"{PFX}_parts.json", rec)
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
    rec = _merge_parts(OUT / f"{PFX}_step.json", rec)
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
    for k, f in (("v2b", "v2/b/cad/20261018-dip_link_v2b_dip_link.stl"), ("v3b", "v3/b/cad/20261020-dip_link_v3b_dip_link.stl")) + (
            (("v4b", "v4/b/cad/20261021-dip_link_v4b_dip_link.stl"),) if V3UP else ()):
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


def turn_play(dm, sf, centre=(0.0, -1.8, -62.0), angles=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)):
    """How far the link turns about each axis of F through `centre` (the bridge's middle) before it meets the fork: the
    least angle (deg, each sense) at which their overlap exceeds 1e-3 mm3; None: still clear at the largest angle. With
    the clamp plate and the slotted arm's screw off, the turn the fork's own shape allows."""
    c = np.asarray(centre, float)
    out = {}
    for ax, v in (("about_x", (1.0, 0.0, 0.0)), ("about_y_pip", (0.0, 1.0, 0.0)), ("about_z_tool", (0.0, 0.0, 1.0))):
        for sgn in (1, -1):
            hit = None
            for a in angles:
                R = _rot(np.asarray(v), math.radians(sgn * a))
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = c - R @ c
                if float((dm.transform(T[:3, :]) ^ sf).volume()) > 1e-3:
                    hit = a
                    break
            out[f"{ax}_{'+' if sgn > 0 else '-'}"] = hit
    return out


def _rot(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K_ = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K_ + (1 - math.cos(ang)) * K_ @ K_


WEDGE_MU = (0.2, 0.3, 0.4)        # PLA on PLA, dry: the range for the wedge's friction at both faces


def wedge_mechanics(alpha_deg, mu):
    """v5: the wedge pushed in along +z_F with force F onto the tongue (normal N1, friction mu) and the plate's edge (normal
    N2 on the incline, friction mu): N1 per newton of push, and the pull that takes it out again per newton of N1 (> 0:
    it locks)."""
    a = math.radians(alpha_deg)
    c, s = math.cos(a), math.sin(a)
    n1_per_push = (c - mu * s) / (s + 2 * mu * c - mu * mu * s)
    pull_per_n1 = (mu * c - s + mu * (c + mu * s)) / (c + mu * s)
    return dict(mu=mu, preload_per_push=round(n1_per_push, 3), pull_out_per_preload=round(pull_per_n1, 3))


def wedge_checks(K, sf):
    """v5: the wedge's seats over its range of gaps, its overlaps at the nominal seat with DIP link b v4 (cover 3S, pad, tip
    cap TW15H3S), the fork and the clamp plate, its path in from 25 mm beyond the seat, its friction and the plate's bending."""
    import trimesh
    V4B = DIP / "v4/b/cad"
    lk = {k_: from_trimesh(trimesh.load(V4B / f, force="mesh")) for k_, f in (
        ("dip_link", "20261021-dip_link_v4b_dip_link.stl"), ("cover_3S", "20261022-dip_link_v4b_cover_I15H3_3S.stl"),
        ("pad", "20261021-dip_link_v4b_pad_block5.stl"), ("cap_TW15H3S", "20261022-dip_link_v4b_tip_cap_TW15H3S.stl"))}
    lk["fork"] = sf
    lk["clamp_plate"] = K.transform(from_trimesh(_load_part("clamp_plate")), np.linalg.inv(T_M_FROM_F))
    tan_a = math.tan(math.radians(P["wedge_deg"]))
    t0, L = P["wedge_t0"], P["wedge_len"]
    t1 = t0 + L * tan_a
    gap0 = P["stop_x"][0] - B_LINK["tongue_back"]
    z_end_finger = -100.06                               # the tip cap's end, the finger's last point toward the tip (z_F)
    seats = []
    for lab, t in (("thinnest gap", t0), ("nominal link", gap0), ("thickest gap", t1)):
        z0 = wedge_seat(t)
        zf = z0 - L - P["wedge_fin_len"]
        seats.append(dict(case=lab, gap_at_edge_mm=round(t, 3), leading_end_z_F=round(z0, 2), fin_end_z_F=round(zf, 2),
                          leading_end_short_of_neck_step_mm=round(B_LINK["mount_z"] - z0, 2),
                          past_plate_end_mm=round(P["stop_bot_z"] - zf, 2), past_finger_end_mm=round(z_end_finger - zf, 2)))
    wd = stop_wedge(K, P, lift=0.02)
    over = {k_: round(float((wd ^ m).volume()), 4) for k_, m in lk.items()}
    # the path in: 0.03 mm off the tongue and 0.03 mm short of the plate's edge at the seat, from 25 mm out
    w3 = stop_wedge(K, P, t_edge=P["stop_x"][0] - B_LINK["tongue_back"] - 0.06, lift=0.03)
    obst = K.union(list(lk.values()))
    path = [(round(float(d), 2), round(float((w3.translate([0.0, 0.0, -d]) ^ obst).volume()), 4)) for d in np.arange(0.0, 25.01, 0.25)]
    # the plate as a cantilever from the dorsal ear (z_F -67.5) to its edge, loaded there; E of printed PLA 2 GPa (v4's estimate)
    E, b, h = 2000.0, 2 * P["stop_y"], P["stop_x"][1] - P["stop_x"][0]
    Lc = P["ear_z_low"] - P["stop_bot_z"]
    k_plate = Lc ** 3 / (3 * E * b * h ** 3 / 12)
    arm_c, arm_w = -62.0 - B_LINK["z_contact"], -62.0 - P["stop_bot_z"]      # about the bridge's middle (turn_play's centre)
    rec = dict(geometry=dict(width_mm=2 * P["wedge_y"], t0_mm=t0, t1_mm=round(t1, 3), taper_deg=P["wedge_deg"], incline_mm=L,
                             fin_len_mm=P["wedge_fin_len"], fin_x_F=P["wedge_fin_x"], gap_nominal_mm=round(gap0, 3)),
               seats=seats, overlaps_at_seat_mm3=over,
               path_in_overlaps=[x for x in path if x[1] > 1e-4], path_in_clear=all(x[1] <= 1e-4 for x in path),
               friction=[wedge_mechanics(P["wedge_deg"], mu) for mu in WEDGE_MU],
               plate=dict(cantilever_mm=round(Lc, 2), width_mm=b, thickness_mm=round(h, 2), E_MPa=E,
                          bend_at_edge_mm_per_N=round(k_plate, 5),
                          contact_travel_mm_per_N=round(k_plate * (arm_c / arm_w) ** 2, 5),
                          note="the edge's load per newton of a face press at the contact point is arm_c / arm_w about the "
                               "bridge's middle (z_F -62); the contact moves arm_c / arm_w of the edge's bend"))
    print("wedge", json.dumps(rec, default=float), flush=True)
    return rec


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
        ears = P.get("fork_style") in ("ears", "mold")  # ears, mold: slid on along the tool axis (+z_F); cradle: along +x_F
        step_ = (lambda d: [0.0, 0.0, d]) if ears else (lambda d: [d, 0.0, 0.0])
        ins = [(round(float(d), 2), round(float((dm.translate(step_(d)) ^ sf).volume()), 4)) for d in np.arange(-30.0, 0.0001, 0.25)]
        r["insertion_axis"] = "+z_F (along the tool axis)" if ears else "+x_F (from the palmar side)"
        r["insertion_overlaps"] = [x for x in ins if x[1] > 1e-4]
        r["insertion_clear"] = not r["insertion_overlaps"]
        if ears:
            cpm = K.transform(from_trimesh(_load_part("clamp_plate")), np.linalg.inv(T_M_FROM_F))
            r["overlap_clamp_plate_mm3"] = round(float((dm ^ cpm).volume()), 4)
        pts = _sample(m, 0.1)
        dd, _ = sf_tree.query(pts)
        reg = {"bridge_palmar_face_to_palmar_ear": (np.abs(pts[:, 0] + 10.55) < 0.03) & (pts[:, 2] > -67.3) & (pts[:, 2] < -63.0),
               "tongue_back_to_back_plate": (np.abs(pts[:, 0] - L["tongue_back"]) < 0.03) & (pts[:, 2] < L["mount_z"] - 0.5),
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
        if P.get("fork_style") == "mold":               # the legs' outlines (their side and top faces) to the walls
            off_y = lambda ys: np.all([np.abs(pts[:, 1] - y) > 0.03 for y in ys], axis=0)          # noqa: E731
            reg["horn_leg_outline_to_walls"] = ((pts[:, 1] > ARM_HORN[0] + 0.03) & (pts[:, 1] < ARM_HORN[1] - 0.03)
                                                & off_y(SLOT) & (pts[:, 2] > -66.0)
                                                & (np.abs(pts[:, 0]) > slot_half(np.clip(pts[:, 2], -66.05, -53.5)) + 0.05))
            reg["slotted_leg_outline_to_walls"] = ((pts[:, 1] > ARM_SLOTTED[0] + 0.03) & (pts[:, 1] < ARM_SLOTTED[1] - 0.03)
                                                   & (pts[:, 2] > -63.0))
        r["gaps_by_region_mm"] = {n: (round(float(dd[s_].min()), 3) if s_.any() else None) for n, s_ in reg.items()}
        r["least_gap_mm"] = round(float(dd.min()), 3)
        r["turn_to_contact_deg"] = turn_play(dm, sf)
        if "stop_x" in P:                                # v4: the three set screws turned in to 0.02 mm off the tongue
            scr = K.union([K.cyl("x", (y, z), 1.5, B_LINK["tongue_back"] + 0.02, P["stop_x"][1] + 3.0)
                           for y, z in P["stop_screws"]])
            r["overlap_set_screws_mm3"] = round(float((dm ^ scr).volume()), 4)
            r["turn_to_contact_deg_screws_set"] = turn_play(dm, K.union([sf, scr]))
            print(k, "turn with the stop screws set", r["turn_to_contact_deg_screws_set"], flush=True)
        if "wedge_t0" in P:                              # v5: the wedge 0.02 mm off the tongue, its incline on the plate's edge
            wd = stop_wedge(K, P, lift=0.02)
            r["overlap_wedge_mm3"] = round(float((dm ^ wd).volume()), 4)
            r["turn_to_contact_deg_wedge_set"] = turn_play(dm, K.union([sf, wd]))
            print(k, "turn with the wedge set", r["turn_to_contact_deg_wedge_set"], flush=True)
        res[k] = r
        print(k, "overlap", r["overlap_fork_mm3"], "insertion clear", r["insertion_clear"], r["gaps_by_region_mm"],
              "turn", r["turn_to_contact_deg"], flush=True)
    if MOLD:                                             # the same turns on the fork of 15:40 (ears and set screw)
        res["v3b"]["turn_to_contact_deg_ears_fork"] = turn_play(from_trimesh(links["v3b"]),
                                                                standoff_fork_v2_ears(K, dict(P, fork_style="ears")))
        print("ears fork turn", res["v3b"]["turn_to_contact_deg_ears_fork"], flush=True)
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
    if "wedge_t0" in P:                                  # v5: every seat of the wedge
        parts_M["stop_wedge_swept"] = _load_part("stop_wedge_swept")
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
    rec = dict(wedge=wedge_checks(K, sf)) if "wedge_t0" in P else {}
    rec.update(rev=REV, b_links=res, hex_key_access=acc, foot_screw_heads=heads, poses_table=poses, walls_mm=walls,
               note="F frame for the links; table frame for the poses (x along the tool axis toward the tip, z up, the palmar "
                    "side down); overlaps by manifold3d booleans of the meshes (the pad's overlap at 'touch' is the contact's "
                    "tessellation); gaps by nearest neighbours between surface samples 0.1 mm apart",
               when=time.strftime("%Y-%m-%d %H:%M"))
    (OUT / f"{PFX}_check.json").write_text(json.dumps(rec, indent=1, default=float))
    print("access", acc, "heads", heads, "walls", walls, f"[{time.time() - t0:.0f} s]", flush=True)


def cmd_check_v3_screws(a):
    """v3: the four M6 x 20 against the adapter and the foot, seated (with and without a DIN 125 washer) and lifted until the
    tip is flush with the adapter's face; a 5 mm hex key (a cylinder of the key's 5.77 mm across corners, and 1 mm wider for
    clearance) from each head's socket along its hole or channel, against the mount, the clamp plate, DIP links b v3 and v4,
    their covers and the pad. Overlaps by manifold3d booleans, mm3."""
    import trimesh
    t0 = time.time()
    parts = {nm: from_trimesh(_load_part(nm)) for nm in ("adapter", "standoff_fork", "clamp_plate")}
    others = {}
    for k_, f in (("link_v3b", "v3/b/cad/20261020-dip_link_v3b_dip_link.stl"), ("link_v4b", "v4/b/cad/20261021-dip_link_v4b_dip_link.stl"),
                  ("cover_v3b", "v3/b/cad/20261020-dip_link_v3b_cover.stl"), ("cover_I15H4_v4b", "v4/b/cad/20261021-dip_link_v4b_cover_I15H4.stl"),
                  ("pad", "v2/b/cad/20261018-dip_link_v2b_pad_ell11_5.stl")):
        m = trimesh.load(DIP / f, force="mesh")
        m.apply_transform(T_M_FROM_F)
        others[k_] = from_trimesh(m)
    K = MeshK()
    k_, L, hd = P["m6_head_k"], P["m6_len"], P["m6_head_d"]
    seats = {"seated": P["ad_t"] + k_, "seated_washer": P["ad_t"] + 1.6 + k_,
             "lifted": L + k_}                            # lifted: the tip flush with the adapter's face (z_M 0)
    rows = []
    for a_ in bolt_angles():
        xF, yF = _m_xy(P["bolt_r"], a_)
        x_m, y_m = -xF, yF - Y_AXIS
        under = y_m < 0
        for pose, zt in seats.items():
            head = K.cyl("z", (x_m, y_m), hd / 2, zt - k_, zt)
            shank = K.cyl("z", (x_m, y_m), 3.0, zt - k_ - L, zt - k_)
            r = dict(bolt_deg=a_, xy_M=[round(x_m, 2), round(y_m, 2)], pose=pose, head_top_zM=round(zt, 2),
                     protrusion_mm=round(L - (zt - k_), 2),
                     overlap_mm3={nm: round(float(((head + shank) ^ s_).volume()), 4) for nm, s_ in parts.items()})
            keys = {}
            tilts = (15.0, 20.0, 25.0) if under else (0.0, 10.0)
            for tdeg in tilts:
                th = math.radians(tdeg)
                sx = 1.0 if x_m > 0 else -1.0
                d = np.array([sx * math.sin(th), 0.0, math.cos(th)]) if under else np.array([sx * math.sin(th) / math.sqrt(2), math.sin(th) / math.sqrt(2), math.cos(th)])
                for rk, tag in ((5.77 / 2, "key"), (5.77 / 2 + 0.5, "key_plus_0.5")):
                    cyl = K.cyl("z", (0.0, 0.0), rk, 0.5, 130.0)
                    R = trimesh.geometry.align_vectors([0, 0, 1.0], d)
                    R[:3, 3] = [x_m, y_m, zt]
                    kc = cyl.transform(R[:3, :])
                    ov = {nm: round(float((kc ^ s_).volume()), 3) for nm, s_ in list(parts.items()) + list(others.items())}
                    keys[f"{tag}_tilt{tdeg:.0f}"] = {nm: v for nm, v in ov.items() if v > 0} or "clear"
            r["key_overlaps_mm3"] = keys
            rows.append(r)
            print(a_, pose, r["protrusion_mm"], r["overlap_mm3"], keys, flush=True)
    # the head cannot leave its pocket: the head lifted 0.6 mm past the pocket's top meets the foot at every bolt
    capt = []
    for a_ in bolt_angles():
        xF, yF = _m_xy(P["bolt_r"], a_)
        h = K.cyl("z", (-xF, yF - Y_AXIS), hd / 2, P["m6_pocket_top"] + 0.1, P["m6_pocket_top"] + 0.1 + k_)
        capt.append(dict(bolt_deg=a_, overlap_above_pocket_mm3=round(float((h ^ parts["standoff_fork"]).volume()), 2)))
    rec = dict(rev=REV, screws="ISO 4762 M6 x 20; seated: head on the adapter's top face (z_M %.1f)" % P["ad_t"],
               poses=rows, captive=capt, pocket=dict(d=P["m6_pocket_d"], top_zM=P["m6_pocket_top"]),
               key="5 mm hex key as a cylinder of 2.885 mm radius (and +0.5 mm) from 0.5 mm above the head's top, 129.5 mm long; "
                   "+y bolts straight (0 deg) and 10 deg outward, -y bolts 15/20/25 deg outward along x_M",
               when=time.strftime("%Y-%m-%d %H:%M"), seconds=round(time.time() - t0, 1))
    (OUT / f"{PFX}_screws.json").write_text(json.dumps(rec, indent=1, default=float))
    print("captive", capt, f"[{time.time() - t0:.0f} s]", flush=True)


def cmd_payload_v2(a):
    """v2: mass and centre of gravity of everything past the UR5e's flange for set_payload: the HEX-E (50 mm, 0.347 kg, CoG
    20 mm from its robot face, its data sheet), the adapter and the standoff-fork at the plastic the P1S plate lays down
    (print/20261020-p1s_pla_calibration.json, the fork from print/20261020-p1s_pla_fork_and_indenters.json when it exists:
    volume x fill ratio x 1.24 g/cm3), DIP link b v3 at its sliced density with
    the cover, board, pad, magnets and brass screws (v3/b/evaluation.json), and the steel screws."""
    import trimesh
    p2 = REPO / "docs/experiments/20261020-ur5_calibration_mount_v2/print"
    plate = json.loads((p2 / "20261020-p1s_pla_calibration.json").read_text())
    fill = {o["id"]: o["fill_ratio"] for o in plate["objects"]}
    newer = p2 / "20261020-p1s_pla_fork_and_indenters.json"         # the fork of 19:10 at 15 % grid, when sliced
    if newer.exists():
        fill.update({o["id"]: o["fill_ratio"] for o in json.loads(newer.read_text())["objects"] if o["id"] == "U2"})
    if V3UP:                                            # v3's adapter and fork from their own plate; the clamp plate is v2's
        v3p = REPO / "docs/experiments/20261021-ur5_calibration_mount_v3/print/20261021-ur5_calibration_mount_v3_p1s_pla.json"
        fill.update({o["id"]: o["fill_ratio"] for o in json.loads(v3p.read_text())["objects"]})
        p4 = REPO / "docs/experiments/20261023-ur5_calibration_mount_v4/print/20261023-ur5_calibration_mount_v4_p1s_pla.json"
        if REV in ("v4", "v5") and p4.exists():          # v4's fork from its own plate (v5 is v4's fork as printed)
            fill.update({o["id"]: o["fill_ratio"] for o in json.loads(p4.read_text())["objects"]})
    ev = json.loads((DIP / "v3/b/evaluation.json").read_text())["mass_g"]
    items = []
    for nm, pid in (("adapter", "U1"), ("standoff_fork", "U2")) + ((("clamp_plate", "U3"),) if "clamp_plate" in PARTS else ()):
        tm = _load_part(nm)
        items.append((nm, float(tm.volume) * 1e-3 * MATERIAL["pla_density_g_cm3"] * fill[pid], np.asarray(tm.center_mass)))
    lk = "v4" if V3UP else "v3"                  # v3 of the mount carries DIP link b v4 (its sliced mass, v4 checks)
    link = trimesh.load(DIP / ("v4/b/cad/20261021-dip_link_v4b_dip_link.stl" if lk == "v4" else
                               "v3/b/cad/20261020-dip_link_v3b_dip_link.stl"), force="mesh")
    link.apply_transform(T_M_FROM_F)
    c_link = np.asarray(link.center_mass)
    c_tip = F_to_M(np.array([-2.0, 0.0, B_LINK["z_contact"]]))
    m_link = (json.loads((DIP / "v4/b/20261021-dip_link_v4b_checks.json").read_text())["mass_g"]["v4"]["sliced"] if lk == "v4"
              else ev["dip_link_at_sliced_density"])
    m6 = (4 * SCREWS["M6x20_ISO4762"]["mass_g"], np.array([0.0, 0.0, 11.0])) if V3UP else \
        (4 * SCREWS["M6x12_ISO4762"]["mass_g"], np.array([0.0, 0.0, 6.0]))
    items += [(f"dip_link_b_{lk}", m_link, c_link),
              ("cover_board_screws", ev["cover_solid"] + ev["board"] + ev["screws_brass"], F_to_M(np.array([3.5, 0.0, -88.0]))),
              ("pad_and_magnets", 2.01 + 5 * 0.106, c_tip),
              ("m6_screws" if V3UP else "m6_screws_washers", m6[0], m6[1]),
              ("m3_screws", 4 * SCREWS["M3x12_foot"]["mass_g"] + 2.4, np.array([0.0, 0.0, 24.0 if V3UP else 20.0])),
              ("hex_e", 347.0, np.array([0.0, 5.0, -50.0 + 20.0]))]
    if P.get("m3_seat") == "insert":
        items.append(("m3_inserts", 4 * SCREWS["M3_insert"]["mass_g"], np.array([0.0, 0.0, P["ad_t"] - P["ins_len"] / 2])))
    if "wedge_t0" in P:                                  # v5: the wedge at its plate's fill, the slotted arm's insert
        wp = json.loads((PRINT / f"{PFX}_p1s_pla_wedge.json").read_text())
        tw = _load_part("stop_wedge")
        items.append(("stop_wedge", float(tw.volume) * 1e-3 * MATERIAL["pla_density_g_cm3"] * wp["objects"][0]["fill_ratio"],
                      np.asarray(tw.center_mass)))
        items.append(("slotted_arm_insert", SCREWS["M3_insert_stop"]["mass_g"],
                      F_to_M(np.array([0.0, P["ib_y"][1] - P["stop_ins_len"] / 2, PIP_ZF]))))
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


CLEARANCE_RUNS = (   # the lab's sessions and the plans they ran (docs/experiments/20261022-ur5_tip_calibration/lab)
    ("sphere2", "20261022-210852-sphere2", "20261022-plan_sphere_lab_turns_checked_reg2144.json"),
    ("rod1", "20261023-104526-rod1", "20261023-plan_rod_lab_reg_rod1.json"))


def cmd_clearance(a):
    """v4 and later: the least clearance of each tool group (v5: the wedge's space over all its seats as its own group) to
    the indenter, the cell and the other indenter over each lab plan, with its session's registration and nominal, through
    scripts/ur5_tip_calibration_check.py --clearance with UR5_MOUNT set to this revision."""
    import re
    import subprocess
    lab = REPO / "docs/experiments/20261022-ur5_tip_calibration/lab"
    pat = re.compile(r"^\s+(\w+)\s+least\s+([-\d.]+) mm registered \(([\w-]+)\),\s+([-\d.]+) nominal \(([\w-]+)\); needs ([\d.]+)")
    rec = dict(note=f"scripts/ur5_tip_calibration_check.py SESSION --clearance PLAN with UR5_MOUNT={REV}: press poses at rest "
                    "and 1 mm past contact, every 1 deg of every turn; tool parts sampled about 2 mm apart, the wedge's "
                    "space 0.5 mm apart", plans={}, when=time.strftime("%Y-%m-%d %H:%M"))
    for name, sess, plan in CLEARANCE_RUNS:
        r = subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "scripts/ur5_tip_calibration_check.py"), str(lab / sess),
                            "--clearance", str(lab / plan)], capture_output=True, text=True, env=dict(os.environ, UR5_MOUNT=REV),
                           cwd=REPO, check=True)
        groups, under = {}, None
        for ln in r.stdout.splitlines():
            m = pat.match(ln)
            if m:
                groups[m.group(1)] = dict(registered_mm=float(m.group(2)), registered_at=m.group(3),
                                          nominal_mm=float(m.group(4)), nominal_at=m.group(5), needs_mm=float(m.group(6)))
            if "under the margins with the registration" in ln:
                under = ln.strip()
        rec["plans"][name] = dict(session=rel(lab / sess), plan=rel(lab / plan), groups=groups, under=under)
        print(name, json.dumps(groups), under, flush=True)
    (OUT / f"{PFX}_clearance.json").write_text(json.dumps(rec, indent=1) + "\n")


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


def cmd_render_v2_ears(a):
    """v2 with the ears fork: the assembly exploded along the tool axis (the link slides on as in v1), the link on the
    fork from the horn side, and the fork alone."""
    import trimesh
    MEDIA.mkdir(parents=True, exist_ok=True)
    C_ = dict(COLORS, cover="0.36 0.52 0.86 1", cable="0.20 0.30 0.85 1")
    parts = {nm: trimesh.load(fname(nm), force="mesh") for nm in PARTS}
    links, extra = b_links_v2()
    dip = {}
    for k_, m in (("dip_link", links["v3b"]), ("cover", extra["cover_v3b"]), ("tip", extra["pad"])):
        t = m.copy()
        t.apply_transform(T_M_FROM_F)
        dip[k_] = t
    scr = _screws_M()

    def W(tm, T=T_W_FROM_M):
        t = tm.copy()
        t.apply_transform(T)
        return t

    def pw(pM):
        return T_W_FROM_M[:3, :3] @ np.asarray(pM, float)

    def pf(pF):
        return pw(F_to_M(np.asarray(pF, float)))
    dz = 34.0
    sh = lambda t, v: _shifted(t, np.array([[1, 0, 0, v[0]], [0, 1, 0, v[1]], [0, 0, 1, v[2]], [0, 0, 0, 1.0]]))   # noqa
    exp = [(W(parts["adapter"]), C_["adapter"], "e_ad"), (W(parts["standoff_fork"]), C_["standoff_fork"], "e_sf"),
           (W(sh(parts["clamp_plate"], [0, -24.0, dz])), C_["clamp_plate"], "e_cp")]
    exp += [(W(sh(dip[k], [0, 0, dz])), C_[c], "e_" + k) for k, c in (("dip_link", "dip_link"), ("cover", "cover"), ("tip", "tip"))]
    cam = (pw([0.0, -8.0, 62.0]), 330, 300, -12)
    img = _render(exp, cam, MEDIA / f"{PFX}_exploded.png")
    _annotate(img, cam, [
        ("Adapter: dowel only, four M6 x 12", pw([-22.3, -22.3, 5.5])),
        ("Standoff-fork: walls around both legs, the ears and the inner block; no key, no set screw" if MOLD else
         "Standoff-fork: the ears and the inner block as v1, no key", pf([-12.0, -10.0, -60.0])),
        ("DIP link: slides on along the tool axis, arms either side of the inner block", pf([-10.55, -2.0, -66.0 - dz])),
        ("Clamp plate on the horn-side arm, two M3 x 16 into the ears", pf([0.0, -31.25 - 24.0, -60.0 - dz]))],
        title="Mount v2 exploded along the tool axis: the DIP link slides on as in v1")
    asm = [(W(parts["adapter"]), C_["adapter"], "a_ad"), (W(parts["standoff_fork"]), C_["standoff_fork"], "a_sf"),
           (W(parts["clamp_plate"]), C_["clamp_plate"], "a_cp"), (W(scr), C_["screws"], "a_scr")]
    asm += [(W(dip[k]), C_[c], "a_" + k) for k, c in (("dip_link", "dip_link"), ("cover", "cover"), ("tip", "tip"))]
    cam = (pf([0.0, -6.0, -66.0]), 190, 315, -10)
    img = _render(asm, cam, MEDIA / f"{PFX}_fork_closeup.png")
    _annotate(img, cam, [
        ("Horn-side arm between the clamp plate and the inner block", pf([-5.0, -23.25, -64.0])),
        ("Walls around the horn leg, 0.15 mm off its outline" if MOLD else
         "Dorsal ear with the M3 set screw onto the bridge's back",
         pf([8.0, -22.6, -50.0]) if MOLD else pf([17.0, P["push_y"], P["push_z"]])),
        ("Clamp plate: horn-side arm pressed onto the inner block", pf([-8.0, -31.25, -60.0])),
        ("Cable slot and groove, open on the dorsal side", pf([17.0, 0.0, -47.0]))],
        title="DIP link b on the mount v2 fork, horn and dorsal side")
    sfm = parts["standoff_fork"]
    cam = (F_to_M(np.array([0.0, -4.0, -60.0])), 170, 45, -42)
    pm = lambda pF: F_to_M(np.asarray(pF, float))                     # noqa: E731
    img = _render([(sfm, C_["standoff_fork"], "fork_alone_" + P["fork_style"])], cam, MEDIA / f"{PFX}_fork_alone.png")
    _annotate(img, cam, [
        ("Inner block: its arc 0.05 mm above b's bridge (the stop along the tool axis)", pm([-6.0, 0.0, -64.5])),
        ("Palmar ear, 0.2 mm off the bridge's palmar face", pm([-10.75, -10.0, -66.0]))]
        + ([("Walls around the horn leg: its outline 0.15 mm off, the slot left open", pm([6.0, -21.5, -60.0])),
            ("Walls around the slotted leg", pm([-7.5, 15.0, -56.0]))] if MOLD else
           [("Dorsal ear, 0.2 mm off the bridge's back, with the set-screw hole", pm([9.65, P["push_y"], P["push_z"]])),
            ("No key: the horn slot stays empty", pm([0.0, -20.2, -60.0]))])
        + [("Cable slot, open on the dorsal side", pm([9.0, 0.0, -60.0]))],
        title="Standoff-fork v2 alone, its fork end up")
    if MOLD:                                             # the slotted leg in its walls, from the slotted side
        cam = (pf([0.0, 10.0, -58.0]), 150, 135, -8)
        img = _render(asm, cam, MEDIA / f"{PFX}_fork_slotted_side.png")
        _annotate(img, cam, [
            ("Slotted leg in its walls, 0.15 mm off its outline", pf([-6.0, 17.4, -55.0])),
            ("M3 x 12 with its washer through the leg's bore into the inner block", pf([0.0, 17.4, PIP_ZF])),
            ("Walls stop 0.2 mm short of the leg's outer face", pf([-12.0, 17.2, -48.0]))],
            title="DIP link b on the mount v2 fork, slotted side")
    hexe = trimesh.creation.cylinder(radius=36.0, height=50.0, sections=96)
    hexe.apply_translation([0.0, 0.0, -25.0])
    cam = (pw([0.0, 0.0, 30.0]), 430, 160, -10)
    img = _render([(W(hexe), C_["hex"], "p_hexe")] + asm, cam, MEDIA / f"{PFX}_on_the_hex_e_palmar.png")
    _annotate(img, cam, [
        ("HEX-E (context)", pw([36.0 * 0.94, -36.0 * 0.34, -25.0])),
        ("Adapter: dowel only, four M6 x 12", pw([31.5 * 0.94, -31.5 * 0.34, 5.5])),
        (f"Homing socket, {P['home_key'][0]:.0f} x {P['home_key'][1]:.0f} mm at its mouth, faces at {P['home_draft']:.0f} deg",
         pw([P["fp_x"], 0.0, P["home_zm"]])),
        ("85A pad on the cover with its side walls", pf([-10.55, 0.0, B_LINK["z_contact"]]))],
        title="Mount v2 on the HEX-E with DIP link b v3, palmar side, finger down")
    h = homing("rod")
    ind = trimesh.load(fname("indenter_rod"), force="mesh")
    T = T_TABLE_FROM_M.copy()
    T[:3, 3] = (np.array([0.0, 0.0, h["indenter_top_z"]]) - T_TABLE_FROM_M[:3, :3] @ np.array(h["contact_M"])
                + np.array([0.0, P["post_dy"], P["home_dz"]]))
    sc = [(W(t, T), c, f"h_{st}") for t, c, st in
          [(parts["standoff_fork"], C_["standoff_fork"], "fork"), (parts["adapter"], C_["adapter"], "adapter"),
           (parts["clamp_plate"], C_["clamp_plate"], "cp"), (hexe, C_["hex"], "hexe"), (dip["dip_link"], C_["dip_link"], "dip"),
           (dip["cover"], C_["cover"], "cover"), (dip["tip"], C_["tip"], "tip")]] + [(ind, C_["indenter"], "h_ind_rod")]
    cam = (np.array([-30.0, 15.0, 45.0]), 330, 240, 12)
    img = _render(sc, cam, MEDIA / f"{PFX}_homing.png")
    _annotate(img, cam, [("Homing: the key's long faces seat in the socket", np.array([*h["post_xy"], h["mouth_z"]])),
                         ("The pad's contact point 30 mm to the side and 5 mm above the rod",
                          np.array(h["contact_at_homing_table"]))], title="Rod indenter with its homing post: homing")
    print("renders ->", rel(MEDIA), flush=True)


def _screws_M_v3(lift=0.0):
    """v3's screws in M: four M6 x 20 seated on the adapter (raised by lift), four M3 x 12 in the foot's counterbores, the
    clamp plate's two M3 x 16, the slotted arm's M3 x 12 with its washer."""
    import trimesh
    out = []
    for a_ in bolt_angles():
        x, y = _m_xy(P["bolt_r"], a_)
        out.append(_screw((0, 0, -1), np.array([-x, y - Y_AXIS, P["ad_t"] + lift]), P["m6_len"], 6.0, P["m6_head_d"], P["m6_head_k"]))
    for a_ in m3_angles():
        x, y = _m_xy(P["m3_r"], a_)
        out.append(_screw((0, 0, -1), np.array([-x, y - Y_AXIS, P["ad_t"] + P["m3_under"]]), 12.0, 3.0, 5.5, 3.0))
    for sx in (-1, 1):
        c = F_to_M(np.array([sx * P["cp_screw_x"], ARM_HORN[0] - P["cp_t"] + P["cp_cb_t"], P["cp_screw_z"]]))
        out.append(_screw((0, 1, 0), c, 16.0, 3.0, 5.5, 3.0))
    out.append(_screw((0, -1, 0), F_to_M(np.array([0.0, ARM_SLOTTED[1] + 0.8, PIP_ZF])), 12.0, 3.0, 5.5, 3.0))
    w = trimesh.creation.cylinder(radius=4.5, height=0.8, sections=48)
    w.apply_transform(trimesh.geometry.align_vectors([0, 0, 1.0], [0, 1.0, 0]))
    w.apply_translation(F_to_M(np.array([0.0, ARM_SLOTTED[1] + 0.4, PIP_ZF])))
    out.append(w)
    return trimesh.util.concatenate(out)


def _inserts_M(dz=0.0):
    """v3 since 2026-10-22: the four M3 heat-set inserts in M, flush with the adapter's top face (raised by dz), as tubes of
    the knurls' diameter around a 2.5 mm bore."""
    import trimesh
    out = []
    for a_ in m3_angles():
        x, y = _m_xy(P["m3_r"], a_)
        t = trimesh.creation.annulus(r_min=1.25, r_max=P["ins_od"] / 2, height=P["ins_len"], sections=48)
        t.apply_translation([-x, y - Y_AXIS, P["ad_t"] - P["ins_len"] / 2 + dz])
        out.append(t)
    return trimesh.util.concatenate(out)


def _sliced_bores(proj, rows):
    """The insert bores as Bambu Studio sliced them: in the adapter's plate, the wall arcs (G2/G3 of the outer and inner
    walls) around each bore, the least radius per layer; the hole's diameter is twice that radius less one outer-wall line
    width. A bore's centre on the plate is the median centre of the wall arcs of radius 2.1-2.3 mm within 4 mm of where the
    layout put it (Bambu Studio may move the part, here about 2 mm); the arcs within 0.3 mm of that centre count. The adapter
    prints tool face down (x, -y, -z of M)."""
    import re
    import zipfile
    meta = json.loads(proj.with_suffix(".json").read_text())
    b = np.array(meta["objects"][0]["bounds_mm"])
    cx, cy = (b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2
    g = zipfile.ZipFile(proj).read("Metadata/plate_1.gcode").decode(errors="replace")
    w = float(re.search(r"; outer_wall_line_width = ([\d.]+)", g).group(1))
    arcs, x, y, z, wall = [], 0.0, 0.0, 0.0, False
    for line in g.splitlines():
        if line.startswith("; FEATURE:"):
            wall = line.split(":", 1)[1].strip() in ("Outer wall", "Inner wall")
        s = line.split(";", 1)[0].split()
        if not s or s[0] not in ("G0", "G1", "G2", "G3"):
            continue
        d = {t[0]: float(t[1:]) for t in s[1:] if t[0] in "XYZEIJ" and len(t) > 1}
        if wall and s[0] in ("G2", "G3") and d.get("E", 0.0) > 0 and \
                math.hypot(d.get("X", x) - x, d.get("Y", y) - y) > 1.0:      # loop arcs; seam pieces of 0.3 mm fit stray centres
            ox, oy = x + d.get("I", 0.0), y + d.get("J", 0.0)
            arcs.append((z, ox, oy, math.hypot(x - ox, y - oy)))
        x, y, z = d.get("X", x), d.get("Y", y), d.get("Z", z)
    A = np.array(arcs)
    out = []
    for r_ in rows:
        ex, ey = cx + r_["centre_M"][0], cy - r_["centre_M"][1]
        c0 = A[(np.hypot(A[:, 1] - ex, A[:, 2] - ey) < 4.0) & (A[:, 3] > 2.1) & (A[:, 3] < 2.3)]
        ox, oy = float(np.median(c0[:, 1])), float(np.median(c0[:, 2]))
        m = np.hypot(A[:, 1] - ox, A[:, 2] - oy) < 0.3
        per = {}
        for zz, rr in A[m][:, [0, 3]]:
            per[round(zz, 2)] = min(rr, per.get(round(zz, 2), 9.0))
        zs = sorted(per)
        body = [per[k] for k in zs if k > zs[0] + 0.01]
        out.append(dict(angle_deg=r_["angle_deg"], centre_on_plate_mm=[round(ox, 2), round(oy, 2)],
                        layout_offset_mm=[round(ox - ex, 2), round(oy - ey, 2)],
                        first_layer_hole_d_mm=round(2 * (per[zs[0]] - w / 2), 2),
                        hole_d_mm=round(2 * (float(np.median(body)) - w / 2), 2),
                        hole_d_range_mm=[round(2 * (min(body) - w / 2), 2), round(2 * (max(body) - w / 2), 2)],
                        last_layer_z_mm=zs[-1], layers=len(zs)))
    return dict(project=rel(proj), outer_wall_line_width_mm=w, bores=out,
                note="from the extrusion arcs' centres and radii; Bambu Studio may move the part on the plate (here about 2 mm)")


def cmd_inserts(a):
    """v3 since 2026-10-22: the adapter's four bores for M3 heat-set inserts. On the exported adapter mesh: each bore's
    diameter (twice the mean distance of its section's vertices from the bore's axis) from the mouth to the bottom, the depth
    at which it ends (sections 0.02 mm apart), and the least wall from the bore to the disc's rim or another hole; from the
    parameters: the room in the bore below the insert for the plastic the knurls displace, the M3 x 12's reach and the
    clearance of its tip; by manifold3d: the screw's overlap with the adapter and the fork (zero when its path is clear) and
    the insert's with the adapter (the plastic its knurls displace, as an envelope). Writes <PFX>_inserts.json and a figure:
    the adapter's top face and a section along the tool axis through the screw at 90 deg (media/<PFX>_inserts.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon as MPoly, Rectangle
    t0 = time.time()
    tm_a, tm_f = _load_part("adapter"), _load_part("standoff_fork")
    zt, L, Dh, ch, dep, od = P["ad_t"], P["ins_len"], P["ins_hole"], P["ins_ch"], P["m3_depth"], P["ins_od"]
    reach = 12.0 - P["m3_under"]                          # the M3 x 12's tip below the adapter's top face

    def loops(z):
        sec = tm_a.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        return [e[:, :2] for e in sec.discrete] if sec is not None else []

    K = MeshK()
    ma, mf = from_trimesh(tm_a), from_trimesh(tm_f)
    rows = []
    for a_ in m3_angles():
        xF, yF = _m_xy(P["m3_r"], a_)
        c = np.array([-xF, yF - Y_AXIS])
        diam = {}
        for d in (0.02, ch / 2, ch + 0.05, 1.0, 3.0, L, L + 1.0, reach, dep - 0.05):
            near = [e for e in loops(zt - d) if np.linalg.norm(e.mean(axis=0) - c) < 1.0]
            diam[f"{d:.2f}"] = round(2 * float(np.hypot(*(near[0] - c).T).mean()), 3) if near else None
        end = None
        for d in np.arange(dep - 0.3, dep + 0.3, 0.02):
            if not [e for e in loops(zt - d) if np.linalg.norm(e.mean(axis=0) - c) < 1.0]:
                end = round(float(d), 2)
                break
        ls = loops(zt - 3.0)
        own = [e for e in ls if np.linalg.norm(e.mean(axis=0) - c) < 1.0][0]
        walls = []
        for e in ls:
            if e is own:
                continue
            dd = float(np.min(np.linalg.norm(own[:, None, :] - e[None, :, :], axis=2)))
            r_e = float(np.hypot(*(e - e.mean(axis=0)).T).mean())          # the loop's own radius tells it apart
            walls.append((round(dd, 2), "rim" if r_e > 20 else ("M6 hole" if r_e > 2.6 else "insert bore")))
        walls.sort()
        scr = K.union([K.cyl("z", tuple(c), 1.5, zt - reach, zt + P["m3_under"]),
                       K.cyl("z", tuple(c), 2.75, zt + P["m3_under"], zt + P["m3_under"] + 3.0)])
        ins = K.sub(K.cyl("z", tuple(c), od / 2, zt - L, zt), K.cyl("z", tuple(c), 1.25, zt - L - 1, zt + 1))
        rows.append(dict(angle_deg=a_, centre_M=np.round(c, 2).tolist(), diameter_mm_at_depth=diam, bore_ends_at_depth_mm=end,
                         least_wall_mm=walls[0][0], least_wall_to=walls[0][1],
                         knurl_wall_mm=round(walls[0][0] - (od - Dh) / 2, 2),
                         screw_overlap_mm3=dict(adapter=round(float((scr ^ ma).volume()), 4),
                                                standoff_fork=round(float((scr ^ mf).volume()), 4)),
                         insert_overlap_adapter_mm3=round(float((ins ^ ma).volume()), 2)))
        print(rows[-1], flush=True)
    env = math.pi / 4 * (od ** 2 - Dh ** 2) * L
    proj = PRINT / "20261022-ur5_calibration_mount_v3_p1s_pla_adapter_inserts.3mf"
    sliced = _sliced_bores(proj, rows) if proj.exists() else None
    rec = dict(rev=REV, insert=dict(what=SCREWS["M3_insert"]["what"], length_mm=L, knurl_d_mm=od), as_sliced=sliced,
               bore=dict(d_mm=Dh, depth_mm=dep, mouth_chamfer_mm=ch, mouth_d_mm=Dh + 2 * ch, straight=True),
               source="CNC Kitchen, 'Dimensions & Design Guidelines' (poster, 2025): M3 x 5.7 and M3 x 3, insert 4.6 mm, hole "
                      "4.0 mm, wall at least 1.6 mm, blind hole at least L + 1 mm; its guide: straight holes, no chamfer "
                      "needed, the insert pressed flush",
               rule=dict(min_wall_mm=1.6, min_depth_mm=L + 1.0),
               screw=dict(what="M3 x 12 from the foot's counterbore", reach_below_face_mm=reach, engaged_mm=min(reach, L),
                          tip_to_bore_end_mm=round(dep - reach, 2)),
               displaced_envelope_mm3=round(env, 1), room_below_insert_mm3=round(math.pi / 4 * Dh ** 2 * (dep - L), 1),
               bores=rows, when=time.strftime("%Y-%m-%d %H:%M"), seconds=round(time.time() - t0, 1))
    (OUT / f"{PFX}_inserts.json").write_text(json.dumps(rec, indent=1, default=float))

    # figure: the top face with the bores and the M6 holes; a section along the tool axis through the screw at 90 deg
    MEDIA.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 17})        # the page shows it about 860 px wide
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 6.4), dpi=130, gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    for e in loops(zt - 3.0):
        rim = float(np.hypot(*(e - e.mean(axis=0)).T).mean()) > 20
        ax0.add_patch(MPoly(e, closed=True, fc="#ddb07a" if rim else "white", ec="k", lw=0.7, zorder=1 if rim else 2))
    for r_ in rows:
        cx, cy = r_["centre_M"]
        ax0.add_patch(Circle((cx, cy), od / 2, fc="none", ec="#a87b17", lw=1.4, ls="--", zorder=3))
        lab = f"{r_['angle_deg']:.0f}{chr(176)}"
        ax0.annotate(lab, xy=(cx, cy), xytext=(cx * 0.66, cy * 0.66), ha="center", va="center", fontsize=16,
                     arrowprops=dict(arrowstyle="-", lw=0.7))
    ax0.set_xlim(-35, 35)
    ax0.set_ylim(-35, 35)
    ax0.set_aspect("equal")
    ax0.set_title("Adapter, 3 mm below its top face", fontsize=17)
    ax0.set_xlabel("x$_M$ (mm), toward the dowel")
    ax0.set_ylabel("y$_M$ (mm)")
    yc = P["m3_r"]
    cols = {"adapter": "#ddb07a", "standoff_fork": "#7da6e3"}
    for nm, tm in (("adapter", tm_a), ("standoff_fork", tm_f)):
        sec = tm.section(plane_origin=[0, 0, 0], plane_normal=[1, 0, 0])
        for e in sec.discrete:
            ax1.add_patch(MPoly(e[:, [1, 2]], closed=True, fc=cols[nm], ec="k", lw=0.6))
    for sx in (-1, 1):
        ax1.add_patch(Rectangle((yc + sx * 1.5 if sx > 0 else yc - od / 2, zt - L), od / 2 - 1.5, L, fc="#c9a227", ec="k", lw=0.8,
                                hatch="////"))
    ax1.add_patch(Rectangle((yc - 1.5, zt - reach), 3.0, reach + P["m3_under"], fc="0.62", ec="k", lw=0.8))
    ax1.add_patch(Rectangle((yc - 2.75, zt + P["m3_under"]), 5.5, 3.0, fc="0.45", ec="k", lw=0.8))
    zb = zt - dep                                         # the bore's end; its width dimensioned below it
    for sx in (-1, 1):
        ax1.plot([yc + sx * Dh / 2] * 2, [zb, zb - 1.6], color="k", lw=0.6)
    ax1.annotate("", xy=(yc - Dh / 2, zb - 1.2), xytext=(yc + Dh / 2, zb - 1.2), arrowprops=dict(arrowstyle="<->", lw=0.9))
    ax1.text(yc, zb - 2.0, f"{Dh:.1f} mm", ha="center", va="top", fontsize=16)
    ax1.annotate("", xy=(yc + 4.2, zt), xytext=(yc + 4.2, zb), arrowprops=dict(arrowstyle="<->", lw=0.9))
    ax1.plot([yc + Dh / 2, yc + 4.6], [zb, zb], color="k", lw=0.6)
    ax1.text(yc + 4.6, zt - dep / 2, f"{dep:.1f} mm", rotation=90, va="center", ha="left", fontsize=16)
    notes = [(f"Foot: counterbore {P['m3_cb_d']:.1f} mm,\n{P['m3_under']:.0f} mm under the head",
              (yc + 3.25, zt + P["m3_under"] + 1.5), 24.0),
             (f"Chamfer {ch:.1f} mm{chr(215)}45{chr(176)}", (yc + Dh / 2 + ch / 2, zt - 0.1), 16.5),
             (f"Insert M3{chr(215)}{L:.1f}, {od:.1f} mm\nknurls, pressed flush", (yc + od / 2 - 0.3, zt - 2.0), 11.0),
             (f"M3{chr(215)}12: tip {dep - reach:.1f} mm\nabove the bore's end", (yc + 1.5, zt - reach + 0.3), 3.5)]
    for text, xy, yt in notes:
        ax1.annotate(text, xy=xy, xytext=(36.5, yt), fontsize=15, va="center", arrowprops=dict(arrowstyle="-", lw=0.7))
    ax1.set_xlim(18.0, 62.0)
    ax1.set_ylim(-0.5, 29.5)
    ax1.set_aspect("equal")
    ax1.set_xticks([20, 25, 30, 35])
    ax1.set_title(f"Section through the screw at 90{chr(176)}", fontsize=17)
    ax1.set_xlabel("y$_M$ (mm)")
    ax1.set_ylabel("z$_M$ (mm), along the tool axis")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{PFX}_inserts.png")
    plt.close(fig)
    print("inserts ->", rel(OUT / f"{PFX}_inserts.json"), rel(MEDIA / f"{PFX}_inserts.png"), f"[{time.time() - t0:.0f} s]", flush=True)


def _hex_key(tip, d, length=110.0, short=30.0, ball=False):
    """A 5 mm hex L-key in M: the long arm from tip along d (unit), the short arm bent at its far end; a ball end at the tip."""
    import trimesh
    d = np.asarray(d, float) / np.linalg.norm(d)
    e = np.cross(d, [0.0, 1.0, 0.0])
    e = e / np.linalg.norm(e) if np.linalg.norm(e) > 1e-6 else np.array([1.0, 0, 0])

    def bar(a, b):
        c = trimesh.creation.cylinder(radius=5.0 / math.sqrt(3), height=float(np.linalg.norm(b - a)), sections=6)
        T = trimesh.geometry.align_vectors([0, 0, 1.0], (b - a) / np.linalg.norm(b - a))
        T[:3, 3] = (a + b) / 2
        c.apply_transform(T)
        return c
    tip = np.asarray(tip, float)
    p0 = tip + d * (3.5 if ball else 0.0)
    p1 = tip + d * length
    parts = [bar(p0, p1), bar(p1 - d * 2.5, p1 - d * 2.5 + e * short)]
    if ball:
        b_ = trimesh.creation.icosphere(subdivisions=3, radius=2.5)
        b_.apply_translation(tip + d * 2.0)
        neck = trimesh.creation.cylinder(radius=1.6, height=2.0, sections=24)
        T = trimesh.geometry.align_vectors([0, 0, 1.0], d)
        T[:3, 3] = tip + d * 3.3
        neck.apply_transform(T)
        parts += [b_, neck]
    return trimesh.util.concatenate(parts)


def cmd_render_v3(a):
    """v3 figures: the mount on the HEX-E with DIP link b v4, the M6 x 20 and a key in each kind of access (tool axis up, the
    view of someone tightening from above), and the parts exploded along the tool axis in the order they go together."""
    import trimesh
    MEDIA.mkdir(parents=True, exist_ok=True)
    C_ = dict(COLORS, cover="0.36 0.52 0.86 1", key="0.15 0.15 0.17 1", keyb="0.75 0.25 0.20 1")
    parts = {nm: trimesh.load(fname(nm), force="mesh") for nm in PARTS}
    dip = {}
    for k_, f in (("dip_link", "v4/b/cad/20261021-dip_link_v4b_dip_link.stl"), ("cover", "v4/b/cad/20261021-dip_link_v4b_cover.stl"),
                  ("tip", "v4/b/cad/20261021-dip_link_v4b_pad_ell11_5.stl")):
        t = trimesh.load(DIP / f, force="mesh")
        t.apply_transform(T_M_FROM_F)
        dip[k_] = t
    scr = _screws_M_v3()
    hexe = trimesh.creation.cylinder(radius=36.0, height=22.0, sections=96)      # the HEX-E's top 22 mm (context)
    hexe.apply_translation([0.0, 0.0, -11.0])
    b45 = np.array([P["bolt_r"] * math.cos(math.radians(45.0)), P["bolt_r"] * math.sin(math.radians(45.0)), 0.0])
    b315 = np.array([b45[0], -b45[1], 0.0])
    zt = P["ad_t"] + P["m6_head_k"]
    th = math.radians(P["key_tilt"])
    key_s = _hex_key(b45 + [0, 0, zt], [0.0, 0.0, 1.0], length=85.0, short=24.0)
    key_b = _hex_key(b315 + [0, 0, zt], [math.sin(th), 0.0, math.cos(th)], length=85.0, short=24.0, ball=True)
    asm = [(hexe, C_["hex"], "v3_hexe"), (parts["adapter"], C_["adapter"], "v3_ad"), (parts["standoff_fork"], C_["standoff_fork"], "v3_sf"),
           (parts["clamp_plate"], C_["clamp_plate"], "v3_cp"), (scr, C_["screws"], "v3_scr"), (key_s, C_["key"], "v3_keys"),
           (key_b, C_["keyb"], "v3_keyb")] + [(dip[k], C_[c], "v3_" + k) for k, c in (("dip_link", "dip_link"), ("cover", "cover"), ("tip", "tip"))]
    cam = (np.array([8.0, 0.0, 50.0]), 380, 140, -22)
    img = _render(asm, cam, MEDIA / f"{PFX}_keys.png")
    _annotate(img, cam, [
        ("Adapter, 15 mm: an M6 x 20 seated on it enters the HEX-E 5.0 mm", np.array([31.0, -6.0, 8.0])),
        ("Foot, 13 mm: each M6 head in a pocket, captive", np.array([34.0, 6.0, 22.0])),
        ("Straight 5 mm key from above, slotted-arm side", b45 + [0, 0, 70.0]),
        (f"Ball-end key at {P['key_tilt']:.0f} deg beside the column's corner", b315 + np.array([math.sin(th), 0, math.cos(th)]) * 70.0),
        ("DIP link b v4 on the fork as in v2; clamp plate unchanged", F_to_M(np.array([-10.0, -2.0, -75.0])))],
        title="Mount v3 on the HEX-E, tool axis up: the M6 x 20 tighten from above")
    # exploded along the tool axis: adapter with the M6 dropped in, the fork lowered over the heads, the link, the clamp plate
    sh = lambda t, dz: _shifted(t, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, dz], [0, 0, 0, 1.0]]))   # noqa: E731
    m6 = trimesh.util.concatenate([_screw((0, 0, -1), np.array([*(P["bolt_r"] * np.array([math.cos(math.radians(a_)), math.sin(math.radians(a_))])),
                                                                  P["ad_t"]]), P["m6_len"], 6.0, P["m6_head_d"], P["m6_head_k"])
                                   for a_ in bolt_angles()])
    d1, d2 = 45.0, 95.0
    exp = [(parts["adapter"], C_["adapter"], "x_ad"), (m6, C_["screws"], "x_m6"), (sh(parts["standoff_fork"], d1), C_["standoff_fork"], "x_sf"),
           (sh(parts["clamp_plate"], d2), C_["clamp_plate"], "x_cp")] + [(sh(dip[k], d2), C_[c], "x_" + k)
                                                                         for k, c in (("dip_link", "dip_link"), ("cover", "cover"), ("tip", "tip"))]
    labels = [("First: the four M6 x 20 into the adapter's holes, heads on its top face", np.array([b45[0], b45[1], zt + 2.0])),
              ("Then the fork over the heads, four M3 x 12 into the adapter", np.array([30.0, 8.0, 28.0 + d1]))]
    if P.get("m3_seat") == "insert":                     # 2026-10-22: the inserts go in first, shown above their bores
        dz_i = 22.0
        exp.append((_inserts_M(dz_i), C_["fasteners"], "x_ins"))
        x0, y0 = _m_xy(P["m3_r"], m3_angles()[2])         # the one at 198 deg: the callout clear of the M6 heads'
        labels = [("First: four M3 heat-set inserts into the adapter's top face (above their bores)",
                   np.array([-x0, y0 - Y_AXIS, P["ad_t"] + dz_i + 9.0])),      # the callout's disc above the insert, not on it
                  ("Then the four M6 x 20 into the adapter's holes, heads on its top face", labels[0][1]),
                  ("Then the fork over the heads, four M3 x 12 into the inserts", labels[1][1])]
    cam = (np.array([0.0, 0.0, 95.0]), 660, 130, -12)
    img = _render(exp, cam, MEDIA / f"{PFX}_exploded.png")
    _annotate(img, cam, labels + [
        ("Last: the DIP link onto the fork along the tool axis, then the clamp plate", F_to_M(np.array([-10.55, 0.0, -66.0])) + [0, 0, d2])],
        title="Mount v3 exploded along the tool axis, in assembly order", callout=True)
    print("renders ->", rel(MEDIA), flush=True)


def _stop_hardware_M():
    """v4: the three set screws (from the tongue's back to 4 mm past the plate's outer face) and their jam nuts, in M."""
    import trimesh
    scr, nuts = [], []
    x1 = P["stop_x"][1]
    for y, z in P["stop_screws"]:
        a_, b_ = F_to_M(np.array([B_LINK["tongue_back"], y, z])), F_to_M(np.array([x1 + 4.0, y, z]))
        for r, p0, p1, sec, out in ((1.5, a_, b_, 32, scr),
                                    (5.5 / math.sqrt(3), F_to_M(np.array([x1, y, z])), F_to_M(np.array([x1 + 2.4, y, z])), 6, nuts)):
            c = trimesh.creation.cylinder(radius=r, height=float(np.linalg.norm(p1 - p0)), sections=sec)
            c.apply_transform(trimesh.geometry.align_vectors([0, 0, 1.0], (p1 - p0) / np.linalg.norm(p1 - p0)))
            c.apply_translation((p0 + p1) / 2)
            out.append(c)
    return trimesh.util.concatenate(scr), trimesh.util.concatenate(nuts)


def cmd_render_v4(a):
    """v4 figures: the stop plate behind the DIP link's tongue with its three set screws and jam nuts, seen from the dorsal
    side, and sections through the screws with a press on the face."""
    import trimesh
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MEDIA.mkdir(parents=True, exist_ok=True)
    C_ = dict(COLORS, cover="0.36 0.52 0.86 1")
    parts = {nm: trimesh.load(fname(nm), force="mesh") for nm in PARTS}
    V4B = DIP / "v4/b/cad"
    dip = {}
    for k_, f in (("dip_link", "20261021-dip_link_v4b_dip_link.stl"), ("cover", "20261022-dip_link_v4b_cover_I15H3_3S.stl"),
                  ("tip", "20261021-dip_link_v4b_pad_block5.stl"), ("cap", "20261022-dip_link_v4b_tip_cap_TW15H3S.stl")):
        t = trimesh.load(V4B / f, force="mesh")
        t.apply_transform(T_M_FROM_F)
        dip[k_] = t
    scr, nuts = _stop_hardware_M()
    asm = [(parts["adapter"], C_["adapter"], "v4_ad"), (parts["standoff_fork"], C_["standoff_fork"], "v4_sf"),
           (parts["clamp_plate"], C_["clamp_plate"], "v4_cp"), (scr, C_["brass"], "v4_scr"), (nuts, C_["brass"], "v4_nut"),
           (dip["dip_link"], C_["dip_link"], "v4_link"), (dip["cover"], C_["cover"], "v4_cover"), (dip["tip"], C_["tip"], "v4_tip"),
           (dip["cap"], "0.30 0.60 0.40 1", "v4_cap")]
    look = F_to_M(np.array([12.0, 0.0, -78.0]))
    cam = (look, 260, -35, -12)
    img = _render(asm, cam, MEDIA / f"{PFX}_stop.png")
    y0, z0 = P["stop_screws"][0]
    _annotate(img, cam, [
        ("Stop plate behind the tongue, part of the standoff-fork", F_to_M(np.array([P["stop_x"][1], 7.5, -84.0]))),
        ("Three brass M3 set screws in heat-set inserts, turned onto the tongue's back", F_to_M(np.array([P["stop_x"][1] + 3.0, y0, z0]))),
        ("Jam nuts lock the screws once they touch", F_to_M(np.array([P["stop_x"][1] + 1.2, 5.0, -87.0]))),
        ("Legs in their walls and the clamp plate, as in v3", F_to_M(np.array([0.0, -27.0, -60.0]))),
        ("Pad and cover of DIP link b v4", F_to_M(np.array([-8.0, 0.0, -90.0])))],
        title="Mount v4: the stop plate behind the DIP link's tongue, from the dorsal side")
    # sections through the screws, x_M (palmar to the right) against z_M (the tool axis, tip up)
    plt.rcParams.update({"font.family": "serif", "font.size": 13})
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 6.4), dpi=110)
    cols = {"standoff_fork": "#5b8fd9", "clamp_plate": "#3c6fb5", "dip_link": "#d8d8d2", "cover": "#5c84db", "tip": "#ebc78c",
            "cap": "#4d9966", "screws": "#cca640", "nuts": "#cca640"}
    meshes = dict(standoff_fork=parts["standoff_fork"], clamp_plate=parts["clamp_plate"], dip_link=dip["dip_link"],
                  cover=dip["cover"], tip=dip["tip"], cap=dip["cap"], screws=scr, nuts=nuts)
    zc = F_to_M(np.array([-10.55, 0.0, B_LINK["z_contact"]]))
    for ax, yF in zip(axs, (0.0, 5.0)):
        for nm, m in meshes.items():
            sec = m.section(plane_origin=[0.0, yF - Y_AXIS, 0.0], plane_normal=[0.0, 1.0, 0.0])
            if sec is None:
                continue
            for e in sec.entities:
                q = sec.vertices[e.points]
                ax.fill(q[:, 0], q[:, 2], color=cols[nm], ec="k", lw=0.5)
        ax.annotate("", xy=(zc[0], zc[2]), xytext=(zc[0] + 9.0, zc[2]), arrowprops=dict(arrowstyle="-|>", color="#b03030", lw=2.0))
        ax.text(zc[0] + 1.0, zc[2] + 2.0, "Press on the face", color="#b03030", fontsize=12)
        ax.set_xlim(-24, 24)
        ax.set_ylim(55, 108)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x$_M$ (mm), palmar side to the right")
        ax.set_title(f"Through y$_F$ = {yF:+.0f} mm" + (": the screw at z$_F$ -80.5" if yF == 0 else ": a screw at z$_F$ -87.0"))
    axs[0].set_ylabel("z$_M$ (mm), along the tool axis")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{PFX}_stop_sections.png")
    plt.close(fig)
    # a turntable film about the tool axis, the link on: 36 frames, 10 deg apart, three seconds at 12 frames per second
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(36):
            _render(asm, (F_to_M(np.array([0.0, 0.0, -47.0])), 290, -35 + 10 * k, -14), Path(tmp) / f"f{k:03d}.png", w=960, h=720)
        film = MEDIA / f"{PFX}_turntable.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i", str(Path(tmp) / "f%03d.png"),
                        "-vf", "crop=960:720:0:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", str(film)], check=True)
        import shutil
        shutil.copy(Path(tmp) / "f000.png", MEDIA / f"{PFX}_turntable.png")
    print("render v4:", MEDIA / f"{PFX}_stop.png", MEDIA / f"{PFX}_stop_sections.png", film, flush=True)

def cmd_render_v5(a):
    """v5 figures: the wedge seated in the stop plate's gap (the nominal link), seen from the dorsal side; a section through
    the middle of the link with a close view of the plate's edge on the incline; a film of the wedge going in."""
    import subprocess
    import tempfile
    import shutil
    import trimesh
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MEDIA.mkdir(parents=True, exist_ok=True)
    C_ = dict(COLORS, cover="0.36 0.52 0.86 1", wedge="0.92 0.42 0.12 1")
    parts = {nm: trimesh.load(fname(nm), force="mesh") for nm in PARTS}
    V4B = DIP / "v4/b/cad"
    dip = {}
    for k_, f in (("dip_link", "20261021-dip_link_v4b_dip_link.stl"), ("cover", "20261022-dip_link_v4b_cover_I15H3_3S.stl"),
                  ("tip", "20261021-dip_link_v4b_pad_block5.stl"), ("cap", "20261022-dip_link_v4b_tip_cap_TW15H3S.stl")):
        t = trimesh.load(V4B / f, force="mesh")
        t.apply_transform(T_M_FROM_F)
        dip[k_] = t
    wedge = parts["stop_wedge"]

    def asm(dz=0.0, tag="v5"):
        w = wedge.copy()
        w.apply_translation([0.0, 0.0, dz])                 # +z_M: back out toward the tip
        return [(parts["adapter"], C_["adapter"], f"{tag}_ad"), (parts["standoff_fork"], C_["standoff_fork"], f"{tag}_sf"),
                (parts["clamp_plate"], C_["clamp_plate"], f"{tag}_cp"), (w, C_["wedge"], f"{tag}_w"),
                (dip["dip_link"], C_["dip_link"], f"{tag}_link"), (dip["cover"], C_["cover"], f"{tag}_cover"),
                (dip["tip"], C_["tip"], f"{tag}_tip"), (dip["cap"], "0.30 0.60 0.40 1", f"{tag}_cap")]
    cam = (F_to_M(np.array([12.0, 0.0, -84.0])), 230, -35, -12)
    img = _render(asm(), cam, MEDIA / f"{PFX}_wedge.png")
    x1 = P["stop_x"][1]
    _annotate(img, cam, [
        ("Stop plate of v4's fork as printed; its three insert bores stay empty", F_to_M(np.array([x1, 7.5, -80.0]))),
        ("PLA wedge, pushed in from the tip end until it stops", F_to_M(np.array([9.6, 5.0, -95.5]))),
        ("Fin to push the wedge in and to pull it out", F_to_M(np.array([P["wedge_fin_x"], -4.0, -98.5]))),
        ("Legs in their walls and the clamp plate, as in v3", F_to_M(np.array([0.0, -27.0, -60.0]))),
        ("Pad and cover of DIP link b v4", F_to_M(np.array([-6.0, 7.4, -86.0])))],
        title="Mount v5: a PLA wedge in the stop plate's gap, from the dorsal side")
    # section through the link's middle (y_F 0): the whole tool end, and the plate's edge on the incline
    plt.rcParams.update({"font.family": "serif", "font.size": 13})
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 6.4), dpi=110, gridspec_kw=dict(width_ratios=(1.0, 1.0)))
    cols = {"standoff_fork": "#5b8fd9", "clamp_plate": "#3c6fb5", "dip_link": "#d8d8d2", "cover": "#5c84db", "tip": "#ebc78c",
            "cap": "#4d9966", "stop_wedge": "#eb6b1f"}
    meshes = dict(standoff_fork=parts["standoff_fork"], clamp_plate=parts["clamp_plate"], dip_link=dip["dip_link"],
                  cover=dip["cover"], tip=dip["tip"], cap=dip["cap"], stop_wedge=wedge)
    zc = F_to_M(np.array([-10.55, 0.0, B_LINK["z_contact"]]))
    edge = F_to_M(np.array([P["stop_x"][0], 0.0, P["stop_bot_z"]]))
    for ax, (xl, zl) in zip(axs, (((-24, 24), (55, 112)), ((-15.5, -5.5), (91.0, 104.0)))):
        for nm, m in meshes.items():
            sec = m.section(plane_origin=[0.0, -Y_AXIS, 0.0], plane_normal=[0.0, 1.0, 0.0])
            if sec is None:
                continue
            for e in sec.entities:
                q = sec.vertices[e.points]
                ax.fill(q[:, 0], q[:, 2], color=cols[nm], ec="k", lw=0.5)
        ax.set_xlim(*xl)
        ax.set_ylim(*zl)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x$_M$ (mm), palmar side to the right")
    axs[0].annotate("", xy=(zc[0], zc[2]), xytext=(zc[0] + 9.0, zc[2]), arrowprops=dict(arrowstyle="-|>", color="#b03030", lw=2.0))
    axs[0].text(zc[0] + 1.0, zc[2] + 2.0, "Press on the face", color="#b03030", fontsize=12)
    axs[0].add_patch(plt.Rectangle((-15.5, 91.0), 10.0, 13.0, fill=False, ec="#b03030", lw=1.2, ls="--"))
    axs[0].set_title("Through y$_F$ = 0: the wedge in the gap")
    axs[0].set_ylabel("z$_M$ (mm), along the tool axis")
    axs[1].plot([edge[0]], [edge[2]], "o", ms=9, mfc="none", mec="#b03030", mew=2.0)
    axs[1].annotate("plate's edge on the incline", xy=(edge[0], edge[2]), xytext=(edge[0] - 5.2, edge[2] + 4.4),
                    color="#b03030", fontsize=12, arrowprops=dict(arrowstyle="-", color="#b03030", lw=1.0))
    axs[1].annotate("tongue's back", xy=(-8.0, 100.5), xytext=(-7.6, 102.6), fontsize=12,
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.8))
    axs[1].set_title("The jam at the plate's lower edge")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{PFX}_wedge_sections.png")
    plt.close(fig)
    # the film: the wedge from 18 mm beyond its seat to the seat, then held; 960 x 720 at 12 frames per second
    cam_f = (F_to_M(np.array([10.0, 0.0, -88.0])), 170, -40, -10)
    dzs = list(np.linspace(18.0, 0.0, 36)) + [0.0] * 12
    with tempfile.TemporaryDirectory() as tmp:
        for k, dz in enumerate(dzs):
            if k > 36:                                   # the held frames repeat the last render
                shutil.copy(Path(tmp) / "f035.png", Path(tmp) / f"f{k:03d}.png")
                continue
            _render(asm(dz, tag=f"v5f{k}"), cam_f, Path(tmp) / f"f{k:03d}.png", w=960, h=720)
        film = MEDIA / f"{PFX}_wedge_in.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i", str(Path(tmp) / "f%03d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", str(film)], check=True)
        shutil.copy(Path(tmp) / "f035.png", MEDIA / f"{PFX}_wedge_in.png")
    print("render v5:", MEDIA / f"{PFX}_wedge.png", MEDIA / f"{PFX}_wedge_sections.png", film, flush=True)


def cmd_sections_v3(a):
    """v3: sections through the M6 axes in the x-z plane of M, y_M +17.68 (the two screws on the slotted-arm side, a straight
    key) and -17.68 (the two under the column's corners, a ball-end key), the screws seated and lifted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly, Rectangle
    MEDIA.mkdir(parents=True, exist_ok=True)
    tms = {nm: _load_part(nm) for nm in ("adapter", "standoff_fork", "clamp_plate")}
    cols = {"adapter": "#ddb07a", "standoff_fork": "#7da6e3", "clamp_plate": "#4f78b8"}
    plt.rcParams.update({"font.family": "serif", "font.size": 15})
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 6.4), dpi=130, sharey=True)
    yb = P["bolt_r"] * math.sin(math.radians(45.0))
    zt, k_, L = P["ad_t"] + P["m6_head_k"], P["m6_head_k"], P["m6_len"]
    for ax, yv, ttl in ((axs[0], yb, "Slotted-arm side, y$_M$ = +17.7 mm"), (axs[1], -yb, "Under the column, y$_M$ = $-$17.7 mm")):
        ax.add_patch(Rectangle((-36, -9), 72, 9, fc="0.85", ec="0.5", hatch="//", lw=0.6))
        ax.text(0, -5.5, "HEX-E", ha="center", va="center", fontsize=14)
        for nm, tm in tms.items():
            sec = tm.section(plane_origin=[0, yv, 0], plane_normal=[0, 1, 0])
            if sec is None:
                continue
            for e in sec.discrete:
                ax.add_patch(MPoly(e[:, [0, 2]], closed=True, fc=cols[nm], ec="k", lw=0.6))
        for sx in (1, -1):
            x = sx * yb
            ax.add_patch(Rectangle((x - 5, zt - k_), 10, k_, fc="0.45", ec="k", lw=0.8))
            ax.add_patch(Rectangle((x - 3, zt - k_ - L), 6, L, fc="0.62", ec="k", lw=0.8))
            ax.add_patch(Rectangle((x - 5, L), 10, k_, fc="none", ec="k", lw=1.0, ls="--"))
            tilt = 0.0 if yv > 0 else math.radians(P["key_tilt"])
            dvec = np.array([sx * math.sin(tilt), math.cos(tilt)])
            nvec = np.array([dvec[1], -dvec[0]])
            p0 = np.array([x, zt]) + dvec * 0.5
            p1 = p0 + dvec * 52.0
            ax.add_patch(MPoly(np.array([p0 + nvec * 2.5, p1 + nvec * 2.5, p1 - nvec * 2.5, p0 - nvec * 2.5]), closed=True,
                               fc=(0.75, 0.2, 0.15, 0.55), ec=(0.6, 0.1, 0.1), lw=0.8))
        ax.set_xlim(-36, 36)
        ax.set_ylim(-9, 76)
        ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=15)
        ax.set_xlabel("x$_M$ (mm), palmar side to the right")
    axs[0].set_ylabel("z$_M$ (mm), along the tool axis")
    axs[0].annotate("Key hole 7 mm", xy=(yb, 27.3), xytext=(1.0, 40.0), fontsize=14, arrowprops=dict(arrowstyle="-", lw=0.8))
    axs[0].annotate("Pocket 12.5 mm", xy=(-yb + 5.6, 24.0), xytext=(-12.0, 33.0), fontsize=14, arrowprops=dict(arrowstyle="-", lw=0.8))
    axs[0].annotate("5.0 mm into the HEX-E", xy=(yb + 3.0, -2.5), xytext=(10.0, -8.5), fontsize=13, va="bottom",
                    arrowprops=dict(arrowstyle="-", lw=0.8))
    axs[1].annotate(f"Channel at {P['key_tilt']:.0f}°", xy=(yb + 6.0, 34.0), xytext=(14.0, 52.0), fontsize=14,
                    arrowprops=dict(arrowstyle="-", lw=0.8))
    axs[0].annotate("Head lifted, tip flush\nwith the adapter (dashed)", xy=(-yb + 4.5, 25.5), xytext=(-12.0, 56.0), fontsize=13,
                    arrowprops=dict(arrowstyle="-", lw=0.8))
    fig.tight_layout()
    fig.savefig(MEDIA / f"{PFX}_sections.png")
    plt.close(fig)
    print("sections ->", rel(MEDIA / f"{PFX}_sections.png"), flush=True)


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
    ap.add_argument("cmd", choices=["interfaces", "build", "check", "estimate", "render", "sections", "print", "docs", "payload",
                                    "screws", "inserts", "clearance"])
    a = ap.parse_args()
    {"interfaces": cmd_interfaces, "build": cmd_build, "check": cmd_check if REV == "v1" else cmd_check_v2, "estimate": cmd_estimate, "render": cmd_render if REV == "v1" else (cmd_render_v4 if REV == "v4" else cmd_render_v5 if REV == "v5" else cmd_render_v3 if V3UP else cmd_render_v2_ears if P.get("fork_style") in ("ears", "mold") else cmd_render_v2),
     "sections": cmd_sections_v3 if V3UP else cmd_sections, "print": cmd_print, "docs": cmd_docs, "payload": cmd_payload_v2, "screws": cmd_check_v3_screws, "clearance": cmd_clearance,
     "inserts": cmd_inserts}[a.cmd](a)


if __name__ == "__main__":
    main()
elif os.environ.get("UR5M_CMD") == "step":         # freecadcmd runs the file as a module
    freecad_step()
