#!/usr/bin/env python3
"""Inverted calibration rig for the UR5 cell: a probe on the HEX-E presses fingertips held still on the indenter plate
(owner, 2026-10-29 13:0x: "should we consider adding a rigid indenter mount for the ur5 and adding a third seat on the
fingertip holder to calibrate these contact directions? ... maybe we can get easier, denser testing with that config?";
"designing the adapter, modifying the v2d plate one more time, running the plan would be helpful. if the tall back row of
the indenter plate is just three mounting spots for the finger, that's fine too ... maybe flat surfaces? ... if we invert
the tools like we're describing, then we can spam whatever kind of tool we want").

The probe base takes the standoff fork's place on mount v7's adapter disc: v7's foot (four M3 x 12 into the disc's brass
inserts, pockets for the four M6 heads with key holes over them), a column with the fork's homing socket, so that the
plate's keys home the probe as they home the fork, and a tool interface on the column's top: a drafted square boss and
four M3 pilot holes. A tool has a flange with the matching socket, two M3 x 10 on a diagonal, a stem and its end: a
5 mm sphere on a neck (the tall indenter's), a half rod R 12.5 across the tool axis, or a flat face. Every tool's end
stands at z_M = PROBE["tip_z"]. The plate v2e keeps v2d's base, keys and front row (the 5 mm sphere where v2d had the
12.5 mm sphere; the rod) and has three tip seats in its back row: a station block of the fingertip holder on a support,
the tip screwed to the block by its four M2 x 8 with its board between them, as on the holder.

Frames (mm): M the mount frame (origin on the HEX-E's tool face, z_M along the tool axis toward the tool's end, x_M
toward the dowel and the homing socket); A the plate's (z up from the base's underside, x away from the robot, origin at
the base's centre), where the low sphere's frame stood at v3 ID 1's registration (lab/20261027-225213-spherev3id1), as in
scripts/indenter_plate_layout.py; F and P the held tip's finger and pad frames (scripts/ur5_cb3_cell.py).

A press: the tool's end moves along a push direction d from a rest standoff mm short of contact to deep mm past it.
For the sphere, d is the force's direction on the pad: along the inward normal, or theta from it (oblique). The stem's
direction is free for the sphere and the flat face, so the search tries stem directions in order of preference and takes
the first whose rest and deep poses pass. For each pose it then turns the tool about its own axis to bring joint 6 inside
its window. A pose passes when the UR5 (the DH model's analytic inverse kinematics, the branch nearest the homing pose on
key 2) keeps |sin q5| >= 0.5, |q3| >= 20 deg, the wrist 150 mm from joint 1's axis, every joint 10 deg inside +-360,
joint 5 in 30..100 deg and joint 6 in -62..45 deg (60 with the owner present). The HEX-E's cable port must stay 15 mm
from the arm. The clearances must hold against the plate, the seats, the held tips, the 2x4s and the table: robot links
8, HEX-E 6, adapter and probe base 3, tool stem 1.0 and tool end 0.3 mm. The tool end is checked against everything but
the pressed pad within 4 mm of the contact (12 for the rod, 8 for the flat).

    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 200 --"
    .venv/bin/python scripts/ur5_inverted_rig.py ik-check                       # analytic IK against the simulator's fk
    $RG ~/miniconda3/envs/femtip/bin/python scripts/ur5_inverted_rig.py build    # STL of the probe base, tools, plate v2e
    $RG .venv/bin/python scripts/ur5_inverted_rig.py reach --orient front_up    # pad points and force directions reached
    $RG .venv/bin/python scripts/ur5_inverted_rig.py replay                      # a plan of probe presses, every sample
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("UR5_MOUNT", "v7")
os.environ.setdefault("UR5M_REV", "v7")
import ur5_cb3_cell as CELL  # noqa: E402
import indenter_array as IA  # noqa: E402

C, SIM = CELL.C, CELL.SIM
LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
OUT = ROOT / "docs/experiments/20261029-inverted_calibration_rig"
STEM = "20261029-inverted_rig"
BLOCK_OUT = ROOT / "docs/experiments/20261030-inverted_rig_seat_block"   # --plate block
BLOCK_STEM = "20261030-seat_block"
TOOLREG = LAB / "20261027-225213-spherev3id1/registration.json"
SURFACE_PLAN = LAB / "20261029-plan_sphere_lab_pass1_night_v3id3.json"      # the pad's FE surface: 812 points in P
BOARD_PLAN = LAB / "superseded/20261027-plan_sphere_lab_stage2_v3id1.json"   # the 2x4s under the plate (checks.board)
FINGER_REACH = ROOT / "docs/experiments/20261026-tall_indenter/data/20261027-trimmed_cap_reach.json"

# ------------------------------------------------------------------------------------------------ the probe (M, mm)
PROBE = dict(
    tip_z=120.0,                    # every tool's end on the tool axis; the finger's pad apex stands at z_M 93.4, 10.6 off it
    foot=dict(z0=15.0, z1=28.0, r=35.0),                     # v7's foot on the adapter disc (z_M 0..15)
    col=dict(x0=-13.0, x1=17.0, y0=-13.0, y1=13.0, z1=64.0, r=3.0),   # x1: the fork's palmar face, where its socket opens
    boss=dict(side=10.0, draft=20.0, h=3.0, seat_gap=0.3,    # on the column's top. Since 2026-10-30 the tool's socket is the
              mouth_ch=0.4),                                 # boss's taper made 0.11 mm smaller a face, so the screws pull
    #                                                          the four tapered faces home and the flange stops 0.3 mm over
    #                                                          the column's top (until then 0.05 mm of play a face and the
    #                                                          flange on the top); its mouth chamfered 0.4 mm
    screws=((8.5, 8.5), (-8.5, -8.5), (8.5, -8.5), (-8.5, 8.5)), m3_hole=3.4, hole_depth=9.0,
    nut=dict(af=5.8, t=2.7, below=3.6),     # an M3 nut captive under each screw (pocket 5.8 mm across flats, 2.7 mm high, its
    #                                         top 3.6 mm under the column's top), slid in from the side: a tool's screws then
    #                                         turn in steel, not in PLA, however often it is changed
    flange=dict(z1=70.0, r=17.0), m3_clear=3.4, m3_cb_d=6.5, m3_under=3.0,   # each tool's flange, M3 x 10 on a diagonal
    stem_r=8.0,
    sphere=dict(R=5.0, neck_r=3.0, neck_dz=4.0, taper_deg=15.0),           # the tall indenter's sphere and neck
    sphere2=dict(R=2.0, neck_r=1.3, neck_dz=1.6, taper_deg=15.0),          # a 2 mm sphere for the flanks by the walls
    rod=dict(R=12.5, half_len=10.0, box_dz=12.5, hull_dz=7.0, hull_half=8.0),   # the plate's rod: axis along y_M
    flat=dict(r=7.0, t=3.0, ch=0.5, neck_r=4.5, neck_l=4.0),   # a disc 14 mm across and 3 mm thick on a 9 mm neck
    flat_tilt=30.0,                 # flat30: the same disc turned 30 deg about y_M through its face's centre, so that its
#                                     stem leans off the pad's normal as a sphere's may (a pad-up face from above)
    fingertip=dict(apex_z=140.0),   # (owner, 2026-10-30) a fingertip on the probe by its cassette, as on a seat: its pad's
)                                   # apex on the tool axis, the palmar face along +z_M, its end toward +x_M. The apex
#                                     stands 20 mm past the other tools' ends, so that 43 mm lie between the flange and the
#                                     station block and a hex key reaches the flange's M3 at about 5 deg past the block
TOOLS = ("sphere5", "sphere2", "rod", "flat", "flat30", "fingertip")
FLATS = dict(flat=0.0, flat30=PROBE["flat_tilt"])         # the flat tools and their face's tilt from the tool axis (deg)
SPHERES = dict(sphere5="sphere", sphere2="sphere2")      # the sphere tools and their PROBE entries
NEED = dict(robot=8.0, hex_e=6.0, mount=3.0, stem=1.0, tip=0.3)
GROUPS = dict(mount=("adapter", "probe_base"), stem=("tool_stem",), tip=("tool_tip",), hex_e=("hex_e",))
EXCL = dict(sphere5=4.0, sphere2=3.0, rod=12.0, flat=20.0, flat30=20.0, fingertip=25.0)   # the tool end's points this near the contact meet the pad
#                                                         (the 2 mm sphere's 3 mm: a push 35 deg off the normal drags it 0.6 mm)
J5_WIN, J6_WIN, J6_OWNER = (30.0, 100.0), (-62.0, 45.0), 60.0
J5_SEARCH = (35.0, 97.0)                                  # a press's own poses keep this much inside joint 5's window
Q3_SEARCH = 30.0                                          # and the elbow this far from straight (the check's 20 deg)
BR = [None]                                               # the homing pose's branch of the inverse kinematics
PORT_MM = 15.0
ST, DEEP = 1.5, 1.0                                       # rest standoff and depth past contact (mm)
APEX_F = np.array([-10.55, 0.0, -87.91])                  # the pad's apex in F
T_FP = C.inv(CELL.T_M_FROM_F) @ CELL.T_M_FROM_P           # the pad frame P in F: x_F = -x_P - 2.05, z_F = -77.05 - z_P


def tool_end_z(tool, p=PROBE):
    """z_M of the tool's reference point: the sphere's centre, the rod's axis, the flat face's centre, the fingertip's
    pad apex."""
    if tool == "fingertip":
        return p["fingertip"]["apex_z"]
    return p["tip_z"] - (p[SPHERES[tool]]["R"] if tool in SPHERES else p["rod"]["R"] if tool == "rod" else 0.0)


def fingertip_boxes(tip=False):
    """(lo, hi) in M of the fingertip tool's station block (BLOCK_F), or of the tip on it (cassette and pad)."""
    lo, hi = ((-10.8, -10.5, -98.46), (3.8, 10.5, -77.05)) if tip else BLOCK_F
    P = C.apply(fingertip_T_MF(), np.array([[u, v, w] for u in (lo[0], hi[0]) for v in (lo[1], hi[1]) for w in (lo[2], hi[2])]))
    return P.min(axis=0), P.max(axis=0)


def _toward_box(u, lo, hi):
    """The point of the rectangle [lo, hi] (2-D, about the origin) in the direction u."""
    s = min((hi[k] if u[k] > 0 else lo[k]) / u[k] for k in (0, 1) if abs(u[k]) > 1e-9)
    return u * s


def fingertip_T_MF(p=PROBE):
    """The fingertip tool's tip frame F in M: the pad's apex on the axis at apex_z, the palmar face along +z_M, the tip's
    end toward +x_M (its board's plug and cable toward -x_M, the side away from the homing socket)."""
    xF, zF = np.array([0.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0])
    R = np.column_stack([xF, np.cross(zF, xF), zF])
    return C.Tm(R, np.array([0.0, 0.0, p["fingertip"]["apex_z"]]) - R @ APEX_F)


# ------------------------------------------------------------------------------------------------ the seats (A, mm)
_S2 = 1.0 / math.sqrt(2.0)
ORIENT = dict(   # the pad's face normal and the tip's distal direction in A
    up_y=((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),            # pad up, the tip's end toward +y
    up_xm=((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)),          # pad up, the tip's end toward the robot
    front_up=((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),       # pad toward the robot, the tip's end up (a finger pointing up)
    front_y=((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),        # pad toward the robot, the tip's end toward +y
    tilt_up=((-_S2, 0.0, _S2), (_S2, 0.0, _S2)),        # pad 45 deg up from the robot's direction, the tip's end up and back
    side_up=((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),        # pad toward -y, the tip's end up
)
BLOCK_F = ((4.4, -10.5, -98.36), (16.4, 10.5, -77.05))    # the station block behind the board (0.6 mm) as on the holder
HOLES_F = ((8.0, -79.82), (-8.0, -79.82), (8.0, -95.82), (-8.0, -95.82))   # (y_F, z_F) of the M2 x 8
LAYOUT = dict(       # plate v2e: v2d's base, keys and front row (the 5 mm sphere where v2d had the 12.5 mm one), three seats
    sphere5=dict(c=(-15.0, 60.0), top=100.0), rod=dict(c=(0.0, -85.0), top=100.0),
    seats=((70.0, -80.0), (70.0, 0.0), (70.0, 80.0)), seat_h=160.0, orient="up_xm",   # pad up, end toward the robot
#                                             (front_up, a finger pointing up, until 2026-10-29 16:00: the same reach, but a
#                                             face press bent its support at 13 um/N where this one's column takes it along
#                                             its axis at 0.2 um/N, as printed)
    support=dict(depth=35.0, half_w=22.0),   # each seat's support flares to 35 mm deep (x) and 44 mm wide at the base:
#                                             9 um/N along x at the apex (PLA beam), against 161 for the block's own section
    rail=dict(t=12.0, front=5.0),            # the base 12 mm thick under the seats' row, from 5 mm before the supports
#                                             to its back edge: 8 times the 6 mm base's stiffness where the supports pull up
    cable=((3.4, -3.95, -63.91), (44.4, 4.05, -58.91)))   # F: a slot through each support just past the board's plug, so
#                                             the held board's cable leaves by the support's back, away from the probe
SEAT_BLOCK = dict(   # --plate block (owner, 2026-10-30): v2e's three seats alone where v2e stood, the only thing on the
#                      table (11:5x the low sphere and the rod stayed and the probe homed on the low sphere's key; 12:0x
#                      "add the homing indexers to the sensor blocks. i want to replace everything on the table with just
#                      three sensor mounts"; 12:3x "we only have about 300g of filament left ... can we drastically reduce the
#                      height and material usage of these parts? ... there's no need for these fingertip holders to come so
#                      high up off the ground anymore"). Until 12:3x: a base 135 x 220 mm over both 2x4s, a 12 mm rail under
#                      the seats, supports 145 mm tall, the keys at x_A 20 mm and seven screws (commit 1690e3df).
    base=(50.0, 85.0, -108.0, 108.0, 5.0),                  # x0, x1, y0, y1, thickness (A): a strip on the tall's 2x4
#                                                             (about x_A 55 to 144), its front 5 mm over the joint
    seat_h=40.0,                                            # the pads' apexes over the strip's underside (160 until 12:3x;
#                                                             the middle seat's plan replays alike at 40 and 50)
    support=dict(depth=0.0, half_w=0.0),                    # each seat's station block on a pedestal of its own footprint
    notch=None,
    keys=((65.0, -40.0, 0.0), (65.0, 40.0, 0.0)),           # v2d's keys on the strip between the seats: at homing the
#                                                             probe's column stands between two held tips
    screws=((75.0, -100.0), (75.0, -21.5), (75.0, 21.5), (75.0, 100.0)),   # all four into the tall's 2x4, 20 mm behind
#                                                             the joint
    under_head=3.0,                                         # mm of PLA under each screw's head
    others=())                                              # nothing else stays on the table
HOME_KEY = [0]                                              # the block's key the probe homes on (the replay takes the
#                                                             nearer of the seat)
PLATE = ["v2e"]                                             # what build, reach, replay and film take (--plate)


def T_AF(orient, apex):
    """The held tip's frame F in A: its pad's apex at apex, the face and the distal end as ORIENT gives them."""
    face, end = (np.asarray(v, float) / np.linalg.norm(v) for v in ORIENT[orient])
    xF, zF = -face, -end
    R = np.column_stack([xF, np.cross(zF, xF), zF])
    return C.Tm(R, np.asarray(apex, float) - R @ APEX_F)


def plate(layout=None):
    """Plate v2e in A: indenter_array.geometry's base, keys, 5 mm sphere and rod, and the seats with their tip frames; with
    --plate block the seats alone on SEAT_BLOCK's base, with no key."""
    block = PLATE[0] == "block"
    lay = dict(LAYOUT, **({k: SEAT_BLOCK[k] for k in ("seat_h", "support")} if block else {}))
    lay.update(layout or {})
    a = copy.deepcopy(IA.ARRAY)
    a["sphere5"] = dict(a["sphere5"], **lay["sphere5"])
    a["rod"] = dict(a["rod"], **lay["rod"])
    tb = SEAT_BLOCK["base"][4] if block else a["base"][4]
    seats = []
    orients = lay["orient"] if isinstance(lay["orient"], (list, tuple)) else [lay["orient"]] * len(lay["seats"])
    sp = lay["support"]
    for (x, y), o in zip(lay["seats"], orients):
        T = T_AF(o, (x, y, lay["seat_h"]))
        lo, hi = BLOCK_F
        corners = C.apply(T, np.array([[u, v, w] for u in (lo[0], hi[0]) for v in (lo[1], hi[1]) for w in (lo[2], hi[2])]))
        x0 = float(corners[:, 0].min())                 # the support: the block's hull down to a footprint on the base's
        x1 = max(float(corners[:, 0].max()), x0 + sp["depth"])            # top, from the block's front face back
        yc = float(corners[:, 1].mean())
        y0, y1 = min(float(corners[:, 1].min()), yc - sp["half_w"]), max(float(corners[:, 1].max()), yc + sp["half_w"])
        foot = np.array([[xx, yy, tb - 0.01] for xx in (x0, x1) for yy in (y0, y1)])
        seats.append(dict(apex=[x, y, lay["seat_h"]], orient=o, T_AF=T.round(6).tolist(),
                          support=np.vstack([corners, foot])))
    if block:
        a["base"], a["screws"], a["keys"] = tuple(SEAT_BLOCK["base"]), tuple(SEAT_BLOCK["screws"]), tuple(SEAT_BLOCK["keys"])
        a["screw_cb"] = (a["screw_cb"][0], tb - SEAT_BLOCK["under_head"])
        xb = a["base"][1]
    else:
        xb = max(a["base"][1], max(float(st["support"][:, 0].max()) for st in seats) + 2.0)
        a["base"] = (a["base"][0], xb, a["base"][2], a["base"][3], tb)      # the base reaches 2 mm past the supports
        a["screws"] = (tuple(s_ for s_ in a["screws"] if s_[0] < 0)
                       + tuple((xb - 10.0, yy) for yy in (-110.0, -40.0, 40.0, 110.0)))   # between the supports' footprints
    g = IA.geometry(a)
    for k in ("sphere12", "holder") + (("sphere5", "rod") if block else ()):
        g.pop(k, None)
    g["notch"] = SEAT_BLOCK["notch"] if block else None
    g["seats"], g["layout"] = seats, lay
    g["rail"] = (min(float(st["support"][:, 0].min()) for st in seats) - lay["rail"]["front"], xb, a["base"][2], a["base"][3],
                 lay["rail"]["t"])                                  # x0, x1, y0, y1, top
    if block:                                                       # the low seats need no rail
        g["rail"] = tuple(a["base"][:4]) + (tb,)
    g["zmax"] = float(max([np.max(st["support"][:, 2]) + 25.0 for st in seats] + [g[k]["top"] for k in ("sphere5", "rod") if k in g]
                          + [k["pin_top"] for k in g["keys"]]))     # the highest part (the tips within 25 mm of their blocks)
    return g


# ------------------------------------------------------------------------------------------------ analytic IK (DH model)
_D, _A_, _AL = SIM.DH_CB3["d"], SIM.DH_CB3["a"], SIM.DH_CB3["alpha"]


def _dh(i, th):
    th = np.asarray(th, float)
    ct, st = np.cos(th), np.sin(th)
    ca, sa = math.cos(_AL[i]), math.sin(_AL[i])
    A = np.zeros(th.shape + (4, 4))
    A[..., 0, 0], A[..., 0, 1], A[..., 0, 2], A[..., 0, 3] = ct, -st * ca, st * sa, _A_[i] * ct
    A[..., 1, 0], A[..., 1, 1], A[..., 1, 2], A[..., 1, 3] = st, ct * ca, -ct * sa, _A_[i] * st
    A[..., 2, 1], A[..., 2, 2], A[..., 2, 3] = sa, ca, _D[i]
    A[..., 3, 3] = 1.0
    return A


def _inv(T):
    R, t = T[..., :3, :3], T[..., :3, 3]
    out = np.zeros_like(T)
    out[..., :3, :3] = np.swapaxes(R, -1, -2)
    out[..., :3, 3] = -np.einsum("...ji,...j->...i", R, t)
    out[..., 3, 3] = 1.0
    return out


def fk_many(Q):
    """Flange poses (N, 4, 4) of joints Q (N, 6), the DH model's."""
    Q = np.atleast_2d(Q)
    T = _dh(0, Q[:, 0])
    for i in range(1, 6):
        T = T @ _dh(i, Q[:, i])
    return T


def ik_all(T):
    """Every solution (N, 8, 6) of the DH model for flange poses T (N, 4, 4) in B; NaN where a branch has none."""
    T = np.asarray(T, float).reshape(-1, 4, 4)
    N = len(T)
    d4, d6, a2, a3 = _D[3], _D[5], _A_[1], _A_[2]
    Q = np.full((N, 8, 6), np.nan)
    R, p = T[:, :3, :3], T[:, :3, 3]
    p05 = p - d6 * R[:, :, 2]
    r = np.hypot(p05[:, 0], p05[:, 1])
    ok1 = r > abs(d4) + 1e-9
    phi = np.arccos(np.clip(abs(d4) / np.maximum(r, 1e-9), -1.0, 1.0))
    psi = np.arctan2(p05[:, 1], p05[:, 0])
    k = 0
    for s1 in (1.0, -1.0):
        q1 = psi + s1 * phi + math.pi / 2
        S1, C1 = np.sin(q1), np.cos(q1)
        c5 = (p[:, 0] * S1 - p[:, 1] * C1 - d4) / d6
        ok5 = ok1 & (np.abs(c5) <= 1.0)
        a5 = np.arccos(np.clip(c5, -1.0, 1.0))
        for s5 in (1.0, -1.0):
            q5 = s5 * a5
            S5 = np.sin(q5)
            S5 = np.where(np.abs(S5) < 1e-9, 1e-9, S5)
            q6 = np.arctan2((-R[:, 0, 1] * S1 + R[:, 1, 1] * C1) / S5, (R[:, 0, 0] * S1 - R[:, 1, 0] * C1) / S5)
            T14 = _inv(_dh(0, q1)) @ T @ _inv(_dh(4, q5) @ _dh(5, q6))
            x, y = T14[:, 0, 3], T14[:, 1, 3]
            c3 = (x * x + y * y - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
            ok3 = ok5 & (np.abs(c3) <= 1.0)
            b3 = np.arccos(np.clip(c3, -1.0, 1.0))
            for s3 in (1.0, -1.0):
                q3 = s3 * b3
                q2 = np.arctan2(y, x) - np.arctan2(a3 * np.sin(q3), a2 + a3 * np.cos(q3))
                T34 = _inv(_dh(1, q2) @ _dh(2, q3)) @ T14
                q4 = np.arctan2(T34[:, 1, 0], T34[:, 0, 0])
                sol = np.stack([q1, q2, q3, q4, q5, q6], axis=1)
                sol[~ok3] = np.nan
                Q[:, k] = sol
                k += 1
    return Q


def branch_of(q):
    """The index (0..7) in ik_all's order of the branch (shoulder, wrist, elbow) that joints q lie on."""
    Q = ik_all(fk_many(np.asarray(q, float)[None]))[0]
    d = np.abs((Q - q + math.pi) % (2.0 * math.pi) - math.pi).max(axis=1)
    return int(np.nanargmin(np.where(np.isnan(d), np.inf, d)))


def ik_near(T, q_ref, tol=0.01, free6=False, branch=None):
    """The solution nearest q_ref (largest joint difference, each joint within pi of q_ref's; joints 1 to 5 alone with
    free6, for a tool that may turn about its axis) for each pose of T; rows of NaN where none exists. Each is checked
    against the forward kinematics (tol mm)."""
    T = np.asarray(T, float).reshape(-1, 4, 4)
    Q = ik_all(T)
    qr = np.asarray(q_ref, float).reshape(-1, 1, 6)
    Qw = qr + (Q - qr + math.pi) % (2.0 * math.pi) - math.pi
    dist = np.abs(Qw - qr)[..., :5 if free6 else 6].max(axis=2)
    if branch is not None:                              # one branch only: the homing pose's
        keep = np.zeros(8, bool)
        keep[branch] = True
        dist = np.where(keep[None, :], dist, np.inf)
    dist = np.where(np.isnan(dist), np.inf, dist)
    best = np.argmin(dist, axis=1)
    out = Qw[np.arange(len(T)), best]
    bad = ~np.isfinite(dist[np.arange(len(T)), best])
    good = ~bad
    if good.any():
        err = np.linalg.norm(fk_many(out[good])[:, :3, 3] - T[good, :3, 3], axis=1)
        idx = np.flatnonzero(good)[err > tol]
        bad[idx] = True
    out[bad] = np.nan
    return out


def cmd_ik_check(a):
    rng = np.random.default_rng(0)
    worst, n_near = 0.0, 0
    for _ in range(a.n):
        q = rng.uniform(-math.pi, math.pi, 6)
        T = SIM.fk(q)
        Q = ik_all(T[None])[0]
        errs = [float(np.linalg.norm(SIM.fk(s) - T)) for s in Q if np.isfinite(s).all()]
        worst = max(worst, max(errs))
        qn = ik_near(T[None], q + rng.normal(0, 0.05, 6))[0]
        n_near += bool(np.allclose(qn, q, atol=1e-6))
    print(f"{a.n} random poses: worst |fk(ik) - T| {worst:.2e} (8 branches each); ik_near returned the generating joints "
          f"for {n_near} of {a.n} (start 0.05 rad away)")


# ------------------------------------------------------------------------------------------------ the probe's surface
def _ring_pts(r, z, sp, c=(0.0, 0.0)):
    nt = max(8, int(math.ceil(2 * math.pi * r / sp)))
    th = np.linspace(0.0, 2 * math.pi, nt, endpoint=False)
    return np.c_[c[0] + r * np.cos(th), c[1] + r * np.sin(th), np.full(nt, z)]


def _disc_pts(r, z, sp, c=(0.0, 0.0)):
    out = [np.array([[c[0], c[1], z]])]
    for rho in np.arange(sp, r + 1e-9, sp):
        out.append(_ring_pts(rho, z, sp, c))
    return np.vstack(out)


def _frustum_pts(r0, r1, z0, z1, sp, caps=True):
    nz = max(2, int(math.ceil(abs(z1 - z0) / sp)) + 1)
    out = [_ring_pts(r0 + (r1 - r0) * s, z0 + (z1 - z0) * s, sp) for s in np.linspace(0.0, 1.0, nz)]
    if caps:
        out += [_disc_pts(r0, z0, sp), _disc_pts(r1, z1, sp)]
    return np.vstack(out)


def _box_pts(lo, hi, sp):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ax = [np.linspace(lo[i], hi[i], max(2, int(math.ceil((hi[i] - lo[i]) / sp)) + 1)) for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    on = np.isclose(G, lo).any(axis=1) | np.isclose(G, hi).any(axis=1)
    return G[on]


def _sphere_pts(c, R, sp):
    n = max(50, int(4 * math.pi * R * R / sp ** 2))
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), math.pi * (1 + 5 ** 0.5) * i
    return np.asarray(c) + R * np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def _cyl_y_pts(xz, R, y0, y1, sp, half=False):
    nt = max(12, int(math.ceil(2 * math.pi * R / sp)))
    th = np.linspace(0.0, 2 * math.pi, nt, endpoint=False)
    if half:
        th = th[np.sin(th) >= -1e-9]
    ys = np.linspace(y0, y1, max(2, int(math.ceil((y1 - y0) / sp)) + 1))
    P = np.array([[xz[0] + R * math.cos(t), y, xz[1] + R * math.sin(t)] for t in th for y in ys])
    return P


@lru_cache(maxsize=8)
def _flat_R(tool):
    """The turn of a flat tool's disc: FLATS[tool] deg about y_M."""
    a = math.radians(FLATS.get(tool, 0.0))
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def _turn_flat(P, Rt, tz):
    """Points of the straight flat's disc turned by Rt about the face's centre (0, 0, tz)."""
    c = np.array([0.0, 0.0, tz])
    return (np.asarray(P, float) - c) @ Rt.T + c


def probe_points(tool, sp=1.5):
    """Surface samples (M) of the probe by group: the adapter disc, the probe base (foot, column), the tool's stem
    (flange, stem, neck or the rod's box) and its end (sphere, half rod, flat face), and the HEX-E."""
    p = PROBE
    f, c, fl = p["foot"], p["col"], p["flange"]
    out = dict(hex_e=_frustum_pts(35.0, 35.0, -35.0, 0.0, 3.0), adapter=_frustum_pts(31.5, 31.5, 0.0, 15.0, 3.0),
               probe_base=np.vstack([_frustum_pts(f["r"], f["r"], f["z0"], f["z1"], 2.5),
                                     _box_pts((c["x0"], c["y0"], f["z1"]), (c["x1"], c["y1"], c["z1"]), 2.0)]))
    sr, tz = p["stem_r"], p["tip_z"]
    stem = [_frustum_pts(fl["r"], fl["r"], c["z1"], fl["z1"], 1.5)]
    if tool in SPHERES:
        s = p[SPHERES[tool]]
        z_neck = tz - s["neck_dz"]
        z_c0 = z_neck - (sr - s["neck_r"]) / math.tan(math.radians(s["taper_deg"]))
        stem += [_frustum_pts(sr, sr, fl["z1"], z_c0, sp, caps=False), _frustum_pts(sr, s["neck_r"], z_c0, z_neck, 0.7, False)]
        tip = _sphere_pts((0.0, 0.0, tz - s["R"]), s["R"], 0.5)
    elif tool == "rod":
        r = p["rod"]
        za = tz - r["R"]
        zb0 = za - r["box_dz"]
        zh0 = zb0 - r["hull_dz"]
        stem += [_frustum_pts(sr, sr, fl["z1"], zh0, sp, caps=False), _box_pts((-r["hull_half"], -r["hull_half"], zh0),
                                                                              (r["hull_half"], r["hull_half"], zb0), sp),
                 _box_pts((-r["R"], -r["half_len"], zb0), (r["R"], r["half_len"], za), sp)]
        tip = np.vstack([_cyl_y_pts((0.0, za), r["R"], -r["half_len"], r["half_len"], 0.7, half=True)])
    elif tool == "fingertip":                            # the stem's hull up to the station block, the block, the tip
        blo, bhi = fingertip_boxes()
        th = np.linspace(0.0, 2 * math.pi, 48, endpoint=False)
        ring = np.c_[sr * np.cos(th), sr * np.sin(th)]
        rect = np.array([_toward_box(u, blo[:2], bhi[:2]) for u in np.c_[np.cos(th), np.sin(th)]])
        stem += [np.c_[ring + (rect - ring) * f, np.full(len(th), fl["z1"] + (blo[2] - fl["z1"]) * f)]
                 for f in np.linspace(0.0, 1.0, 16)]
        stem.append(_box_pts(blo, bhi, sp))
        tlo, thi = fingertip_boxes(tip=True)
        tip = _box_pts(tlo, thi, 0.7)
    else:
        fp = p["flat"]
        zd = tz - fp["t"]                                # the disc's back face; the neck behind it, then a cone to the stem
        zn = zd - fp["neck_l"]
        zc0 = zn - (sr - fp["neck_r"]) / math.tan(math.radians(30.0))
        Rt = _flat_R(tool)
        neck = _frustum_pts(fp["neck_r"], fp["neck_r"], zn, zd, 0.7, False)
        stem += [_frustum_pts(sr, sr, fl["z1"], zc0, sp, caps=False), _frustum_pts(sr, fp["neck_r"], zc0, zn, 0.7, False)]
        if FLATS[tool]:                                  # the neck bends from the cone's top to the turned disc's back
            ring0, ring1 = _ring_pts(fp["neck_r"], zn, 0.7), _turn_flat(_ring_pts(fp["neck_r"], zd, 0.7), Rt, tz)
            stem += [ring0 + (ring1 - ring0) * f for f in np.linspace(0.0, 1.0, 8)]
        else:
            stem.append(neck)
        tip = _turn_flat(np.vstack([_frustum_pts(fp["r"], fp["r"], zd, tz - fp["ch"], 0.5, caps=False),
                                    _disc_pts(fp["r"], zd, 0.5),
                                    _frustum_pts(fp["r"], fp["r"] - fp["ch"], tz - fp["ch"], tz, 0.5, caps=False),
                                    _disc_pts(fp["r"] - fp["ch"], tz, 0.5)]), Rt, tz)
    stem = np.vstack(stem)
    if tool in SPHERES:                                  # the neck's end lies inside the sphere: those points are not stem
        s = p[SPHERES[tool]]
        stem = stem[np.linalg.norm(stem - [0.0, 0.0, tz - s["R"]], axis=1) > s["R"] + 0.05]
    out["tool_stem"] = stem
    out["tool_tip"] = tip
    return out


# ------------------------------------------------------------------------------------------------ the CAD (MeshK, M)
def _m_xy(r, ang_deg):
    a = math.radians(ang_deg)
    return r * math.cos(a), r * math.sin(a)


def probe_base_solid(K):
    """The probe base in M: v7's foot and its cuts for the disc's M3 and M6, the column with v7's homing socket, the boss
    on its top, and the four M3 clearance holes with a captive nut under each, slid in from the column's +-y face."""
    import ur5_calibration_mount as UM
    P, p = UM.P, PROBE
    f, c, b, nt = p["foot"], p["col"], p["boss"], p["nut"]
    t = math.tan(math.radians(b["draft"]))
    h0 = b["side"] / 2
    zt = c["z1"] - b["seat_gap"]                          # the column's top: the tool's flange stops seat_gap over it
    parts = [K.cyl("z", (0.0, 0.0), f["r"], f["z0"], f["z1"]),
             K.rbox_z(c["x0"], c["x1"], c["y0"], c["y1"], f["z1"] - 0.01, zt, c["r"]),
             UM.frustum(K, "z", (0.0, 0.0), (h0, h0), (h0 - b["h"] * t, h0 - b["h"] * t), zt - 0.01, zt + b["h"])]
    cut = []
    for ang in UM.m3_angles(P):            # the foot's M3 x 12 into the disc's inserts, counterbored from the foot's top
        xy = _m_xy(P["m3_r"], ang)
        cut += [K.cyl("z", xy, P["m3_clear"] / 2, f["z0"] - 1.0, f["z1"] + 1.0),
                K.cyl("z", xy, P["m3_cb_d"] / 2, P["ad_t"] + P["m3_under"], f["z1"] + 1.0)]
    for ang in UM.bolt_angles(P):          # pockets round the M6 heads on the disc, a hex key's way over each
        xy = _m_xy(P["bolt_r"], ang)
        cut += [K.cyl("z", xy, P["m6_pocket_d"] / 2, f["z0"] - 1.0, P["m6_pocket_top"]),
                K.cyl("z", xy, P["key_d"] / 2, P["m6_pocket_top"] - 0.5, f["z1"] + 1.0)]
    zn1 = zt - nt["below"]
    zn0 = zn1 - nt["t"]
    for x, y in p["screws"]:               # the hex's flats face +-x, so the nut slides in along y from the nearer face
        cut.append(K.cyl("z", (x, y), p["m3_hole"] / 2, zt - p["hole_depth"], zt + b["h"] + 1.0))
        ye = c["y1"] + 1.0 if y > 0 else c["y0"] - 1.0
        hexa = K.hull([(x + nt["af"] / 2 * math.cos(math.radians(a_)) / math.cos(math.radians(30.0)),
                        y + nt["af"] / 2 * math.sin(math.radians(a_)) / math.cos(math.radians(30.0)), z)
                       for a_ in range(30, 360, 60) for z in (zn0, zn1)])
        cut += [hexa, K.box((x - nt["af"] / 2, min(y, ye), zn0), (x + nt["af"] / 2, max(y, ye), zn1))]
    fx = c["x1"]                           # the socket: v7's (8 x 18 mm at the mouth, 20 deg draft, 5.3 deep) at z_M 49
    cut.append(UM.frustum(K, "x", (0.0, P["home_zm"]), UM.home_halves(-1.0, p=P), UM.home_halves(P["home_depth"], p=P),
                          fx + 1.0, fx - P["home_depth"]))
    return K.cut_all(K.union(parts), cut)


def tool_solid(K, tool):
    """A tool in M: its flange with the boss's socket and two counterbored M3 on a diagonal, the stem and the end."""
    import ur5_calibration_mount as UM
    p = PROBE
    c, fl, b = p["col"], p["flange"], p["boss"]
    sr, tz = p["stem_r"], p["tip_z"]
    t = math.tan(math.radians(b["draft"]))
    parts = [K.cyl("z", (0.0, 0.0), fl["r"], c["z1"], fl["z1"])]
    if tool in SPHERES:
        s = p[SPHERES[tool]]
        z_neck = tz - s["neck_dz"]
        z_c0 = z_neck - (sr - s["neck_r"]) / math.tan(math.radians(s["taper_deg"]))
        parts += [K.cyl("z", (0.0, 0.0), sr, fl["z1"] - 0.01, z_c0 + 0.01), K.cyl("z", (0.0, 0.0), sr, z_c0, z_neck, r1=s["neck_r"]),
                  K.sphere((0.0, 0.0, tz - s["R"]), s["R"])]
    elif tool == "rod":
        r = p["rod"]
        za = tz - r["R"]
        zb0 = za - r["box_dz"]
        zh0 = zb0 - r["hull_dz"]
        hh, R, hl = r["hull_half"], r["R"], r["half_len"]
        parts += [K.cyl("z", (0.0, 0.0), sr, fl["z1"] - 0.01, zh0 + 0.01),
                  K.hull([(x, y, zh0) for x, y in ((-hh, -hh), (hh, -hh), (hh, hh), (-hh, hh))]
                         + [(x, y, zb0 + 0.01) for x, y in ((-R, -hl), (R, -hl), (R, hl), (-R, hl))]),
                  K.box((-R, -hl, zb0), (R, hl, za)), K.cyl("y", (0.0, za), R, -hl, hl)]
    elif tool == "fingertip":                            # the stem's hull up to the station block, which takes the tip's
        blo, bhi = fingertip_boxes()                     # four M2 x 8 in 1.9 mm holes as a seat's block does
        T = fingertip_T_MF()
        th = np.linspace(0.0, 2 * math.pi, 48, endpoint=False)
        disc = [(sr * math.cos(a_), sr * math.sin(a_), fl["z1"] - 0.01) for a_ in th]
        corners = [(x, y, blo[2] + 0.01) for x in (blo[0], bhi[0]) for y in (blo[1], bhi[1])]
        lo_ = BLOCK_F[0]
        holes = [K.transform(K.cyl("x", (y, z), 0.95, lo_[0] - 1.0, lo_[0] + 10.0), T) for y, z in HOLES_F]
        parts.append(K.cut_all(K.union([K.hull(disc + corners), K.box(tuple(blo), tuple(bhi))]), holes))
    else:
        fp = p["flat"]
        zd = tz - fp["t"]
        zn = zd - fp["neck_l"]
        zc0 = zn - (sr - fp["neck_r"]) / math.tan(math.radians(30.0))
        disc = K.union([K.cyl("z", (0.0, 0.0), fp["r"], zd, tz - fp["ch"] + 0.01),
                        K.cyl("z", (0.0, 0.0), fp["r"], tz - fp["ch"], tz, r1=fp["r"] - fp["ch"])])
        parts += [K.cyl("z", (0.0, 0.0), sr, fl["z1"] - 0.01, zc0 + 0.01), K.cyl("z", (0.0, 0.0), sr, zc0, zn + 0.01, r1=fp["neck_r"])]
        if FLATS[tool]:                                  # the disc turned about its face's centre; a hull bends the neck
            Rt = _flat_R(tool)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = Rt, np.array([0.0, 0.0, tz]) - Rt @ np.array([0.0, 0.0, tz])
            th = np.linspace(0.0, 2 * math.pi, 64, endpoint=False)
            ring0 = np.c_[fp["neck_r"] * np.cos(th), fp["neck_r"] * np.sin(th), np.full(64, zn - 0.01)]
            ring1 = _turn_flat(np.c_[fp["neck_r"] * np.cos(th), fp["neck_r"] * np.sin(th), np.full(64, zd + 0.01)], Rt, tz)
            parts += [K.hull([tuple(v) for v in np.vstack([ring0, ring1])]), K.transform(disc, T)]
        else:
            parts += [K.cyl("z", (0.0, 0.0), fp["neck_r"], zn, zd + 0.01), disc]
    h0 = b["side"] / 2 - b["seat_gap"] * t     # the boss's half side seat_gap over the column's top: the faces meet there
    ch = b["mouth_ch"]                     # 45 deg at the socket's mouth, which prints on the bed: the first layer's squeeze
    cut = [UM.frustum(K, "z", (0.0, 0.0), (h0 + t, h0 + t), (h0 - (b["h"] + 0.3) * t, h0 - (b["h"] + 0.3) * t),
                      c["z1"] - 1.0, c["z1"] + b["h"] + 0.3),
           UM.frustum(K, "z", (0.0, 0.0), (h0 + ch + 1.0, h0 + ch + 1.0), (h0, h0), c["z1"] - 1.0, c["z1"] + ch)]
    for xy in p["screws"][:2]:
        cut += [K.cyl("z", xy, p["m3_clear"] / 2, c["z1"] - 1.0, fl["z1"] + 1.0),
                K.cyl("z", xy, p["m3_cb_d"] / 2, c["z1"] + p["m3_under"], fl["z1"] + 1.0)]
    return K.cut_all(K.union(parts), cut)


def plate_solid(K, g):
    """Plate v2e (A): indenter_array.solid's base, keys, 5 mm sphere and rod, and each seat's block on its support with
    the four 1.9 mm holes for the tip's M2 x 8 (10 mm deep from the block's face)."""
    a = g["params"]
    x0, x1, y0, y1, tb = a["base"]
    rx0, rx1, ry0, ry1, rt = g["rail"]
    base = K.union([K.rbox_z(x0, x1, y0, y1, 0.0, tb, 6.0), K.rbox_z(rx0, rx1, ry0, ry1, 0.0, rt, 6.0)])
    d, (cbd, cbh) = a["screw_hole"], a["screw_cb"]
    holes = []
    if g.get("notch"):                         # the seat block: clear of the low sphere's base
        (nx0, nx1), (ny0, ny1) = g["notch"]
        holes.append(K.box((nx0 - 1.0, ny0, -1.0), (nx1, ny1, rt + 1.0)))
    for x, y in a["screws"]:                   # on the rail: the counterbore from its top, the same PLA under the head
        top = rt if rx0 <= x <= rx1 else tb
        holes += [K.cyl("z", (x, y), d / 2, -1.0, top + 1.0), K.cyl("z", (x, y), cbd / 2, tb - cbh, top + 1.0)]
    parts = [K.cut_all(base, holes)]
    import ur5_calibration_mount as UM
    kd = g["key"]
    bx, by = kd["k0"][0] + kd["margin"], kd["k0"][1] + kd["margin"]
    for k in g["keys"]:
        local = K.union([K.rbox_z(-bx, bx, -by, by, tb - 0.01, k["body_top"], 2.0),
                         UM.frustum(K, "z", (0.0, 0.0), kd["k0"], kd["k1"], k["body_top"] - 0.01, k["pin_top"])])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = IA._Rz(k["yaw"]), [k["c"][0], k["c"][1], 0.0]
        parts.append(K.transform(local, T))
    if "sphere5" in g:
        s = g["sphere5"]
        cc = s["c"]
        if s["z_col"] > tb:
            parts.append(K.cyl("z", cc, s["col_r"], tb - 0.01, s["z_col"] + 0.01))
        parts += [K.cyl("z", cc, s["r_col"], s["z_col"], s["z_neck"], r1=s["neck_r"]),
                  K.sphere((cc[0], cc[1], s["top"] - s["R"]), s["R"])]
    if "rod" in g:
        r = g["rod"]
        (cx, cy), hl = r["c"], r["half_len"]
        parts += [K.cyl("z", (cx, cy), r["col_r"], tb - 0.01, r["box_z0"] + 0.01, r1=r["r_top"]),
                  K.box((cx - r["R"], cy - hl, r["box_z0"]), (cx + r["R"], cy + hl, r["z_axis"])),
                  K.cyl("y", (cx, r["z_axis"]), r["R"], cy - hl, cy + hl)]
    for st in g["seats"]:
        T = np.asarray(st["T_AF"], float)
        lo, hi = BLOCK_F
        blk = K.transform(K.box(lo, hi), T)
        sup = K.hull([tuple(v) for v in st["support"]])
        holes = [K.transform(K.cyl("x", (y, z), 0.95, lo[0] - 1.0, lo[0] + 10.0), T) for y, z in HOLES_F]
        if st["orient"] == "front_up":             # the cable leaves a pad-up tip away from the robot, over the column
            holes.append(K.transform(K.box(*g["layout"]["cable"]), T))
        parts.append(K.cut_all(K.union([blk, sup]), holes))
    return K.union(parts)


def cmd_build(a):
    import ur5_calibration_mount as UM
    K = UM.MeshK()
    cad = OUT / "cad"
    cad.mkdir(parents=True, exist_ok=True)
    rec = {}
    pname = "seat_block" if PLATE[0] == "block" else "plate_v2e"
    solids = dict(probe_base=probe_base_solid(K), **{f"tool_{t}": tool_solid(K, t) for t in TOOLS},
                  **{pname: plate_solid(K, plate())})
    for name, s in solids.items():
        tm = UM.to_trimesh(s)
        dst = cad / f"{STEM}_{name}.stl"
        tm.export(dst)
        com = tm.center_mass if tm.is_watertight else tm.centroid
        rec[name] = dict(stl=str(dst.relative_to(ROOT)), frame="A" if name == pname else "M",
                         volume_cm3=round(float(tm.volume) / 1e3, 2), mass_g_solid=round(float(tm.volume) / 1e3 * 1.24, 1),
                         com_mm=np.round(com, 2).tolist(), bounds_mm=np.round(tm.bounds, 2).tolist(),
                         watertight=bool(tm.is_watertight), bodies=len(tm.split(only_watertight=False)))
        print(name, json.dumps(rec[name]), flush=True)
    g = plate()
    (cad / f"{STEM}_parts.json").write_text(json.dumps(dict(
        parts=rec, probe=PROBE, layout={k: v for k, v in g["layout"].items()},
        seats=[{k: v for k, v in s.items() if k in ("apex", "orient", "T_AF")} for s in g["seats"]],
        when=time.strftime("%Y-%m-%d %H:%M")), indent=1, default=float) + "\n")


def cmd_seat_beam(a):
    """The seat's compliance at the pad's apex for each hold, by beam theory over the support's height (the hull of the
    block and its footprint, from the rail's top): bending along x and y with the force at the apex's height (shear
    neglected) and the axial compliance under the apex. Solid PLA at E 2.5 GPa, and as printed: a shell of a.wall mm (5
    walls) with the core's sparse infill counted at its density^1.5. Needs trimesh and shapely (the femtip env)."""
    import trimesh
    from shapely.geometry import MultiPoint

    def props(poly):
        x, y = np.asarray(poly.exterior.coords).T
        cr = x[:-1] * y[1:] - x[1:] * y[:-1]
        A = 0.5 * cr.sum()
        cx, cy = ((x[:-1] + x[1:]) * cr).sum() / (6 * A), ((y[:-1] + y[1:]) * cr).sum() / (6 * A)
        xc, yc = x - cx, y - cy
        cr = xc[:-1] * yc[1:] - xc[1:] * yc[:-1]
        Ixx = ((yc[:-1] ** 2 + yc[:-1] * yc[1:] + yc[1:] ** 2) * cr).sum() / 12
        Iyy = ((xc[:-1] ** 2 + xc[:-1] * xc[1:] + xc[1:] ** 2) * cr).sum() / 12
        return abs(A), abs(Ixx), abs(Iyy)

    E = 2500.0
    out = dict(what="the seat's compliance at the pad's apex (um/N) by beam theory over the support, PLA E 2.5 GPa; x is "
                    "away from the robot, z up; 'printed' a shell of 2 mm with 25 % gyroid inside, counted at 0.25^1.5",
               E_MPa=E, wall_mm=a.wall, infill=a.infill, holds={})
    for o in a.orient.split(","):
        g = plate(dict(orient=o))
        st = g["seats"][1]
        apex = np.asarray(st["apex"], float)
        blk = trimesh.creation.box(bounds=[BLOCK_F[0], BLOCK_F[1]])
        blk.apply_transform(np.asarray(st["T_AF"], float))
        hull = trimesh.convex.convex_hull(np.vstack([st["support"], blk.vertices]))
        zs = np.linspace(g["rail"][4] + 0.25, hull.bounds[1][2] - 0.25, 400)
        res = {}
        for label, wall, fill in (("solid", None, 1.0), ("printed", a.wall, a.infill)):
            c = np.zeros(3)
            for z0, z1 in zip(zs[:-1], zs[1:]):
                z = 0.5 * (z0 + z1)
                sec = hull.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
                if sec is None:
                    continue
                poly = MultiPoint([tuple(v[:2]) for v in sec.vertices]).convex_hull
                A, Ixx, Iyy = props(poly)
                if wall is not None:
                    core = poly.buffer(-wall, join_style=2)
                    if not core.is_empty:
                        a2, ix2, iy2 = props(core)
                        k = fill ** 1.5
                        A, Ixx, Iyy = A - (1 - k) * a2, Ixx - (1 - k) * ix2, Iyy - (1 - k) * iy2
                arm, dz = max(apex[2] - z, 0.0), z1 - z0
                c += [arm ** 2 / (E * Iyy) * dz, arm ** 2 / (E * Ixx) * dz, (dz / (E * A) if z < apex[2] else 0.0)]
            res[label] = dict(zip(("x", "y", "z"), np.round(c * 1e3, 3).tolist()))
        face = np.asarray(ORIENT[o][0], float)
        res["face_axis"] = "xyz"[int(np.argmax(np.abs(face)))] if np.max(np.abs(face)) > 0.99 else "tilted"
        res["support_top_mm"] = round(float(hull.bounds[1][2]), 2)
        out["holds"][o] = res
        print(o, json.dumps(res), flush=True)
    dst = OUT / "data" / f"{STEM}_seat_beam.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(f"-> {dst}")


# ------------------------------------------------------------------------------------------------ the scene's SDF (A)
_TREES = {}
_HULLS = {}


def tip_trees():
    """k-d trees of the held tip's surface in F: the pad, and the cassette with the trimmed cap and its screw heads."""
    if not _TREES:
        import trimesh
        from scipy.spatial import cKDTree
        for key, names in (("pad", ("pad",)), ("body", ("cover", "cap"))):
            pts, nrm = [], []
            for name in names:
                m = CELL.part_F(name)
                n = int(min(20000, max(500, m.area / 0.4 ** 2)))
                pp, fi = trimesh.sample.sample_surface_even(m, n, seed=1)
                pts.append(pp)
                nrm.append(m.face_normals[fi])
            P, Nn = np.vstack(pts), np.vstack(nrm)
            _TREES[key] = (cKDTree(P), P, Nn)
    return _TREES


def _tree_sdf(PF, key):
    tree, P, Nn = tip_trees()[key]
    d, j = tree.query(PF, k=8)
    s = np.einsum("ijk,ijk->i", PF[:, None, :] - P[j], Nn[j])
    return d[:, 0] * np.where(s < 0.0, -1.0, 1.0)


def hull_sdf(P, pts):
    """ur5_cb3_cell._convex_sdf (a lower bound of the distance to the convex hull of pts), its planes kept between calls."""
    pts = np.asarray(pts, float)
    key = pts.round(4).tobytes()
    eq = _HULLS.get(key)
    if eq is None:
        from scipy.spatial import ConvexHull
        eq = _HULLS[key] = ConvexHull(pts).equations
    return (P @ eq[:, :3].T + eq[:, 3]).max(axis=1)


def seat_sdfs(PA, st, cut_near=None):
    """Distances (mm) of points PA (A) to one seat: its block and support, its tip's body (cassette, cap, screw heads) with
    the board, and its pad. cut_near (point in A, radius): the pad's distance left out for points that near the point."""
    T = np.asarray(st["T_AF"], float)
    PF = (PA - T[:3, 3]) @ T[:3, :3]
    out = dict(block=np.minimum(CELL.sd_box(PF, *BLOCK_F), hull_sdf(PA, st["support"])))
    zd, zp = CELL.TIP_F["z_ends"]
    box = CELL.sd_box(PF, (-10.8, -10.7, zd - 1.6), (8.0, 10.7, zp))
    near = box < 3.0
    body, pad = box.copy(), box.copy()
    if near.any():
        body[near] = _tree_sdf(PF[near], "body")
        pad[near] = _tree_sdf(PF[near], "pad")
    out["tip body"] = np.min(np.stack([body] + [CELL.sd_box(PF, lo, hi) for _, lo, hi in CELL.held_board_boxes_F()]), axis=0)
    if cut_near is not None:
        pad = np.where(np.linalg.norm(PA - cut_near[0], axis=1) < cut_near[1], 1e9, pad)
    out["pad"] = pad
    return out


def plate_sdfs(PA, g, T_BA, contact=None, reach=20.0):
    """{part: distances (mm)} of points PA (A) to the plate v2e's parts, the seats and their tips, the 2x4s and the table.
    contact = (seat index, contact point in A, radius): that seat's pad left out near the contact. Points more than reach
    mm outside the plate's bounding box get the 2x4s and the table only (1e9 for the rest)."""
    a = g["params"]
    x0, x1, y0, y1, tb = a["base"]
    out = {"2x4s, table": CELL.cell_sdf(PA, T_BA)}
    if CELL.OTHER:                             # the seat block: the low sphere and the rod stay on the table
        out["fixtures"] = CELL.other_sdf(C.apply(T_BA, PA))
    m = ((PA[:, 0] > x0 - reach) & (PA[:, 0] < x1 + reach) & (PA[:, 1] > y0 - reach) & (PA[:, 1] < y1 + reach)
         & (PA[:, 2] < g["zmax"] + reach))
    if not m.any():
        return out
    P = PA[m]
    rx0, rx1, ry0, ry1, rt = g["rail"]
    parts = dict(base=np.minimum(CELL.sd_box(P, (x0, y0, 0.0), (x1, y1, tb)), CELL.sd_box(P, (rx0, ry0, 0.0), (rx1, ry1, rt))))
    if g.get("notch"):
        (nx0, nx1), (ny0, ny1) = g["notch"]
        parts["base"] = np.maximum(parts["base"], -CELL.sd_box(P, (nx0 - 1.0, ny0, -1.0), (nx1, ny1, rt + 1.0)))
    kd = g["key"]
    bx, by = kd["k0"][0] + kd["margin"], kd["k0"][1] + kd["margin"]
    for i, k in enumerate(g["keys"]):
        L = (P - [k["c"][0], k["c"][1], 0.0]) @ IA._Rz(k["yaw"])
        pts = [(sx * hx_, sy * hy_, z) for (hx_, hy_), z in ((kd["k0"], k["body_top"]), (kd["k1"], k["pin_top"]))
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        parts[f"key {i + 1}"] = np.minimum(CELL.sd_box(L, (-bx, -by, tb), (bx, by, k["body_top"])), hull_sdf(L, pts))
    if "sphere5" in g:
        s = g["sphere5"]
        Q = P - [s["c"][0], s["c"][1], 0.0]
        parts["sphere5"] = np.minimum.reduce([CELL.sd_cyl_z(Q, s["col_r"], tb, s["z_col"]),
                                              CELL.sd_capped_cone(Q, s["z_col"], s["z_neck"], s["r_col"], s["neck_r"]),
                                              np.linalg.norm(Q - [0.0, 0.0, s["top"] - s["R"]], axis=1) - s["R"]])
    if "rod" in g:
        r = g["rod"]
        (cx, cy), hl = r["c"], r["half_len"]
        Q = P - [cx, cy, 0.0]
        parts["rod"] = np.minimum.reduce([CELL.sd_capped_cone(Q, tb, r["box_z0"], r["col_r"], r["r_top"]),
                                          CELL.sd_box(P, (cx - r["R"], cy - hl, r["box_z0"]), (cx + r["R"], cy + hl, r["z_axis"])),
                                          CELL.sd_cyl_y(P, r["R"], cy - hl, cy + hl, xz=(cx, r["z_axis"]))])
    for i, st in enumerate(g["seats"]):
        cut = (contact[1], contact[2]) if contact is not None and contact[0] == i else None
        for k, v in seat_sdfs(P, st, cut_near=cut).items():
            parts[f"seat {i + 1} {k}"] = v
    for k, v in parts.items():
        full = np.full(len(PA), 1e9)
        full[m] = v
        out[k] = full
    return out


# ------------------------------------------------------------------------------------------------ the context
def context(layout=None, tool="sphere5"):
    g = plate(layout)
    reg = json.loads(TOOLREG.read_text())
    T_BA, T_EM = np.array(reg["T_BI"], float), np.array(reg["T_EM"], float)
    U_, _, Vt = np.linalg.svd(T_BA[:3, :3])                           # the registration's rotation, rounded in its file,
    T_BA[:3, :3] = U_ @ Vt                                            # is 8e-5 off orthonormal
    CELL.BOARD.update(json.loads(BOARD_PLAN.read_text())["checks"]["board"])
    x0, x1, y0, y1, _ = g["params"]["base"]
    CELL.BOARD["half_mm"] = 400.0
    CELL.BOARD["block_half_mm"] = [max(-x0, x1, 145.0) + 10.0, 500.0]      # both 2x4s (x_A about -35 to 144)
    CELL.OTHER[:] = [(np.array(json.loads((LAB / d / "registration.json").read_text())["T_BI"], float), k)
                     for k, d in (SEAT_BLOCK["others"] if PLATE[0] == "block" else ())]
    T_AM = np.array(g["keys"][HOME_KEY[0] if PLATE[0] == "block" else 1]["T_AM_home"], float)   # the probe's socket is the
    T_BE_home = T_BA @ T_AM @ C.inv(T_EM)                                                      # fork's
    q_home = SIM.ik(T_BE_home, np.asarray(reg["q_home"], float))
    BR[0] = branch_of(q_home)
    surf = json.loads(SURFACE_PLAN.read_text())["geometry"]["surface"]
    return dict(g=g, T_BA=T_BA, T_EM=T_EM, q_home=q_home, T_BE_home=T_BE_home, tool=tool,
                pts_P=np.array(surf["pts"], float), nrm_P=np.array(surf["normals"], float))


def region(n):
    """The plan builder's regions of a pad normal in P (scripts/ur5_tip_calibration_r24_pass3.region)."""
    if n[2] > 0.7:
        return "tip"
    if abs(n[1]) > 0.7:
        return "flank"
    return "face" if n[0] > 0.9 else ("edge" if n[0] > 0.5 else None)


def joint_flags(q, owner=False):
    """The path checker's limits on joints q (rad): a list of what fails (empty when none)."""
    j = np.degrees(q)
    why = []
    if abs(math.sin(q[4])) < 0.5:
        why.append("sin5")
    q3 = abs(math.degrees(math.atan2(math.sin(q[2]), math.cos(q[2]))))
    if q3 < 20.0:
        why.append("q3")
    if float(np.min(360.0 - np.abs(j))) < 10.0:
        why.append("limit")
    if not J5_WIN[0] <= j[4] <= J5_WIN[1]:
        why.append("j5")
    if not J6_WIN[0] <= j[5] <= (J6_OWNER if owner else J6_WIN[1]):
        why.append("j6")
    if not why:
        w = CELL.link_frames(q)["wrist_1_link"][:3, 3]
        if math.hypot(w[0], w[1]) < 150.0:
            why.append("shoulder")
    return why


_ROBOT = {}


def robot_pts():
    if not _ROBOT:
        _ROBOT.update(CELL.robot_points())
    return _ROBOT


def clearance(q, ctx, contact=None, groups=None):
    """{group: (least distance (mm), the part)} of the robot's links, the HEX-E, the adapter and probe base, the tool's
    stem and end at joints q, and the port's distance to the arm."""
    g, T_BA, T_EM = ctx["g"], ctx["T_BA"], ctx["T_EM"]
    T_AB = C.inv(T_BA)
    LF = CELL.link_frames(q)
    res = {}

    def upd(grp, d):
        for part, v in d.items():
            m = float(v.min())
            if m < res.get(grp, (1e9, ""))[0]:
                res[grp] = (m, part)

    for ln, p in robot_pts().items():
        upd("robot", plate_sdfs(C.apply(T_AB @ LF[ln], p), g, T_BA, contact))
    T_AM = T_AB @ LF["tool0"] @ T_EM
    for name, p in probe_points(ctx["tool"]).items():
        grp = next((k for k, v in GROUPS.items() if name in v), "robot")
        if groups and grp not in groups:
            continue
        PA = C.apply(T_AM, p)
        sd = plate_sdfs(PA, g, T_BA, contact if name == "tool_tip" else None)
        upd(grp, sd)
    pe = CELL.port_points()
    port = min(float(tree.query(C.apply(C.inv(LF[ln]) @ LF["tool0"], pe))[0].min()) for ln, tree in CELL._self_trees().items())
    res["port"] = (port, "arm")
    return res


def clear_flags(cl):
    why = [f"{grp} {cl[grp][0]:.2f} mm to the {cl[grp][1]}" for grp, need in NEED.items() if cl.get(grp, (1e9,))[0] < need]
    if cl.get("port", (1e9,))[0] < PORT_MM:
        why.append(f"port {cl['port'][0]:.1f} mm to the arm")
    return why


# ------------------------------------------------------------------------------------------------ the search
def fib_dirs(n=400):
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), math.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


U_ALL = fib_dirs(400)
U_ROBOT = np.array([-1.0, 0.0, 1.2]) / math.hypot(1.0, 1.2)     # a stem toward the robot and up
Q6_TRY = (0.0, -30.0, 30.0, -55.0, 40.0)


def frames_for(u, xref=np.array([0.0, 0.0, 1.0])):
    """Rotations (K, 3, 3) of M in A with z_M = -u and x_M the nearest to xref."""
    z = -np.atleast_2d(u)
    x = xref[None, :] - (z @ xref)[:, None] * z
    bad = np.linalg.norm(x, axis=1) < 1e-6
    x[bad] = np.array([1.0, 0.0, 0.0]) - (z[bad] @ np.array([1.0, 0.0, 0.0]))[:, None] * z[bad]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=2)


def pose_B(ctx, R_AM, c_A, zc):
    """Flange poses (K, 4, 4) in B for M at rotations R_AM with the tool's reference point (z_M zc) at c_A."""
    K = len(R_AM)
    T = np.tile(np.eye(4), (K, 1, 1))
    T[:, :3, :3] = R_AM
    T[:, :3, 3] = np.asarray(c_A, float).reshape(-1, 3) - R_AM[:, :, 2] * zc
    return ctx["T_BA"][None] @ T @ C.inv(ctx["T_EM"])[None]


def turned(T_BE, T_EM, deg):
    """The flange pose with the tool turned deg about its own axis z_M (the registered mount stands 0.4 mm and 0.3 deg off
    the flange's axis, so a turn of joint 6 alone would move the tool's end)."""
    return T_BE @ T_EM @ C.Tm(C.rot([0.0, 0.0, 1.0], deg)) @ C.inv(T_EM)


def try_poses(ctx, T_rest, T_deep, sym, owner, contact, need_clear=True):
    """Joints and clearances of one press's rest and deep flange poses. sym: the tool may turn about its axis; the turns
    that bring joint 6 to each of Q6_TRY are tried in that order. (ok, info) with the poses and joints that passed."""
    q0 = ik_near(T_rest[None], ctx["q_home"], free6=sym, branch=BR[0])[0]
    if not np.isfinite(q0).all():
        return False, dict(why=["ik"])
    targets = Q6_TRY if sym else (math.degrees(q0[5]),)
    last = None
    for tgt in targets:
        tdeg = tgt - math.degrees(q0[5])
        Tr, Td = (turned(T_rest, ctx["T_EM"], tdeg), turned(T_deep, ctx["T_EM"], tdeg)) if sym else (T_rest, T_deep)
        ref = q0.copy()
        ref[5] = math.radians(tgt)
        qr = ik_near(Tr[None], ref, branch=BR[0])[0]
        qd = ik_near(Td[None], qr, branch=BR[0])[0] if np.isfinite(qr).all() else qr
        if not (np.isfinite(qr).all() and np.isfinite(qd).all()):
            last = dict(why=["ik"])
            continue
        why = sorted(set(joint_flags(qr, owner) + joint_flags(qd, owner)))
        if not why and not all(J5_SEARCH[0] <= math.degrees(v[4]) <= J5_SEARCH[1] for v in (qr, qd)):
            why = ["j5"]
        if not why and not all(abs(math.degrees(math.atan2(math.sin(v[2]), math.cos(v[2])))) >= Q3_SEARCH for v in (qr, qd)):
            why = ["q3"]
        if why:
            last = dict(why=why)
            continue
        info = dict(T_rest=Tr, T_deep=Td, q=qr, qd=qd, j5=round(math.degrees(qr[4]), 1), j6=round(math.degrees(qr[5]), 1))
        if not need_clear:
            return True, info
        cl, why = {}, []
        for qq in (qr, qd):
            for k, v in clearance(qq, ctx, contact).items():
                if v[0] < cl.get(k, (1e9, ""))[0]:
                    cl[k] = v
            why = clear_flags(cl)
            if why:
                break
        info["clear"] = {k: [round(v[0], 2), v[1]] for k, v in cl.items()}
        if not why:
            return True, info
        last = dict(why=why, clear=info["clear"])
        if not all(w.startswith("port") for w in why):            # another turn of joint 6 moves only the port
            return False, last
    return False, last or dict(why=["ik"])


def search_sphere(ctx, seat_i, p, n, d, owner=False, max_clear=6, need_clear=True):
    """The first stem direction, in the order of preference, at which a sphere press at pad point p (A) with outward
    normal n and push direction d passes. The stems tried point away from the pad (u.n >= -0.2) and not steeply down
    (u_z >= -0.5); the preferred one lies between the normal and U_ROBOT."""
    zc = tool_end_z(ctx["tool"])
    c = p + PROBE[SPHERES[ctx["tool"]]]["R"] * n
    keep = (U_ALL @ n >= -0.2) & (U_ALL[:, 2] >= -0.5)
    U = U_ALL[keep]
    pref = n + U_ROBOT
    pref /= np.linalg.norm(pref)
    U = U[np.argsort(-(U @ pref))]
    Rs = frames_for(U)
    T_rest = pose_B(ctx, Rs, np.tile(c - ST * d, (len(U), 1)), zc)
    T_deep = pose_B(ctx, Rs, np.tile(c + DEEP * d, (len(U), 1)), zc)
    Q = ik_near(T_rest, ctx["q_home"], free6=True, branch=BR[0])
    fin = np.isfinite(Q).all(axis=1)
    j5 = np.degrees(np.where(fin, Q[:, 4], 0.0))
    cand = np.flatnonzero(fin & (j5 >= J5_SEARCH[0]) & (j5 <= J5_SEARCH[1]))
    if not len(cand):
        return False, dict(why={"j5" if fin.any() else "ik": int(len(U))}, n_u=0)
    why, n_try = {}, 0
    contact = (seat_i, p, EXCL[ctx["tool"]])
    for k in cand:
        ok, info = try_poses(ctx, T_rest[k], T_deep[k], True, owner, contact, need_clear)
        if ok:
            info.update(u=np.round(U[k], 3).tolist(), c=c, n_u=int(len(cand)), tried=n_try + 1)
            return True, info
        for w in info.get("why", []):
            key = w.split(" mm to the ")[0].split()[0] if " mm to the " in w else w.split()[0]
            why[key] = why.get(key, 0) + 1
        if "clear" in info:
            n_try += 1
            if n_try >= max_clear:
                break
    return False, dict(why=why, n_u=int(len(cand)))


FLAT_OFFSETS = ((0.0, 0.0), (2.5, 0.0), (-2.5, 0.0), (0.0, 2.5), (0.0, -2.5), (4.0, 0.0), (-4.0, 0.0), (0.0, 4.0), (0.0, -4.0))


def search_flat(ctx, seat_i, p, n, d, owner=False, max_clear=6, need_clear=True):
    """A flat press: the face (z_M = -n) tangent to the pad at p, its centre at p or up to 4 mm from it across the face
    (FLAT_OFFSETS), the tool turned about its axis as the sphere's; the first that passes. The face is left out of the
    pressed pad's distance (it meets the pad anywhere on it), not out of the tip's body's."""
    tool = ctx["tool"]
    zc = tool_end_z(tool)
    t1, t2 = tangent_basis(n, np.array([0.0, 0.0, 1.0]))
    R0 = frames_for(n[None])[0]
    tilt = FLATS[tool]
    if tilt:                                             # the face's normal is -n; the stem leans tilt deg off n, its
        Ry = _flat_R(tool).T                             # azimuth about n searched, those toward U_ROBOT first
        Rs = [C.rot(n, az) @ R0 @ Ry for az in np.arange(0.0, 360.0, 15.0)]
        Rs.sort(key=lambda R_: -float(-R_[:, 2] @ U_ROBOT))
    else:
        Rs = [R0]
    why, n_try = {}, 0
    contact = (seat_i, p, EXCL[tool])
    for R1 in Rs:
        for o1, o2 in FLAT_OFFSETS:
            c = p + o1 * t1 + o2 * t2
            T_rest = pose_B(ctx, R1[None], (c - ST * d)[None], zc)[0]
            T_deep = pose_B(ctx, R1[None], (c + DEEP * d)[None], zc)[0]
            ok, info = try_poses(ctx, T_rest, T_deep, not tilt, owner, contact, need_clear)
            if ok:
                info.update(u=np.round(-R1[:, 2], 3).tolist(), c=c, offset=[o1, o2], n_u=len(Rs), tried=n_try + 1)
                return True, info
            for w in info.get("why", []):
                key = w.split(" mm to the ")[0].split()[0] if " mm to the " in w else w.split()[0]
                why[key] = why.get(key, 0) + 1
            if "clear" in info:
                n_try += 1
                if n_try >= max_clear * (2 if tilt else 1):
                    return False, dict(why=why, n_u=len(Rs) * len(FLAT_OFFSETS))
    return False, dict(why=why, n_u=len(Rs) * len(FLAT_OFFSETS))


def search(ctx, seat_i, p, n, d, **kw):
    return (search_flat if ctx["tool"] in FLATS else search_sphere)(ctx, seat_i, p, n, d, **kw)


def twist_ok(ctx, seat_i, p, n, info, deg, owner=False, need_clear=True):
    """A found press turned deg about the pad's normal through the contact at its depth (the sphere spins in place)."""
    R_BA = ctx["T_BA"][:3, :3]
    c_B = C.apply(ctx["T_BA"], info["c"] + DEEP * (-n))
    Td = C.turn_about(info["T_deep"], c_B, R_BA @ n, deg)
    ref = info["qd"].copy()
    ref[5] += math.radians(deg) * float(np.sign((R_BA @ n) @ info["T_deep"][:3, 2]) or 1.0)
    qd = ik_near(Td[None], ref, branch=BR[0])[0]
    if not np.isfinite(qd).all() or joint_flags(qd, owner):
        return False
    if not J5_SEARCH[0] <= math.degrees(qd[4]) <= J5_SEARCH[1]:          # the press's own margin on joint 5
        return False
    if not need_clear:
        return True
    return not clear_flags(clearance(qd, ctx, (seat_i, p, EXCL[ctx["tool"]])))


def tangent_basis(n, axis):
    """Two unit tangents at a pad point with outward normal n: along the projection of the tip's axis, and across."""
    t1 = axis - (axis @ n) * n
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.cross(n, [1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.cross(n, [0.0, 1.0, 0.0])
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1)


def thin_idx(pts, mm):
    if mm <= 0:
        return np.arange(len(pts))
    keep = []
    for i, p in enumerate(pts):
        if all(np.linalg.norm(p - pts[j]) >= mm for j in keep):
            keep.append(i)
    return np.array(keep)


def cmd_reach(a):
    """Which pad points and force directions a sphere press reaches on a seat (the middle seat by default) for each
    orientation of --orient."""
    t0 = time.time()
    orients = a.orient.split(",")
    thetas = [float(v) for v in a.thetas.split(",") if v]
    azim = np.arange(a.azimuths) * 360.0 / a.azimuths
    res = dict(what=f"probe presses ({a.tool}) on the held tip of one seat of plate v2e, by pad point and push direction",
               tool=a.tool,
               probe=PROBE, need=NEED, windows=dict(j5=J5_WIN, j6=J6_WIN, j6_owner=J6_OWNER, port_mm=PORT_MM),
               standoff_mm=ST, deep_mm=DEEP, thetas=thetas, azimuths=azim.tolist(), thin_mm=a.thin, seat=a.seat,
               joints_only=bool(a.joints_only), h=a.h, orients={})
    tag = ("_joints" if a.joints_only else "") + ("" if a.tool == "sphere5" else f"_{a.tool}")
    dst = (Path(a.out) if a.out else OUT / "data" / f"{STEM}_reach{tag}_h{a.h:g}.json").resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and a.resume:
        res["orients"] = json.loads(dst.read_text()).get("orients", {})
    for o in orients:
        if o in res["orients"]:
            print(f"{o}: done before, kept", flush=True)
            continue
        ctx = context(dict(orient=o, seat_h=a.h), tool=a.tool)
        st = ctx["g"]["seats"][a.seat]
        T = np.asarray(st["T_AF"], float) @ T_FP
        P_A, N_A = C.apply(T, ctx["pts_P"]), ctx["nrm_P"] @ T[:3, :3].T
        axis = -np.asarray(st["T_AF"], float)[:3, 2]                     # the tip's distal direction in A
        regs = [region(v) for v in ctx["nrm_P"]]
        idx = [i for i in thin_idx(ctx["pts_P"], a.thin) if regs[i]]
        if a.limit:
            idx = idx[::max(1, len(idx) // a.limit)]
        if a.part:                                       # k/n: every n-th point from the k-th, for parallel jobs
            k_, n_ = (int(v) for v in a.part.split("/"))
            idx = idx[k_::n_]
        rows = []
        for m, i in enumerate(idx):
            p, n = P_A[i], N_A[i]
            ok, info = search(ctx, a.seat, p, n, -n, need_clear=not a.joints_only)
            row = dict(i=int(i), region=regs[i], p_P=np.round(ctx["pts_P"][i], 3).tolist(), normal=bool(ok),
                       normal_info={k: v for k, v in info.items() if k in ("u", "j5", "j6", "clear", "why", "n_u", "tried")})
            if ok and not a.no_twist:
                row["twist"] = [tw for tw in (-15.0, 15.0)
                                if twist_ok(ctx, a.seat, p, n, info, tw, need_clear=not a.joints_only)]
            t1, t2 = tangent_basis(n, axis)
            obl = {}
            for th in thetas:
                hits = []
                for az in azim:
                    t = math.cos(math.radians(az)) * t1 + math.sin(math.radians(az)) * t2
                    d = -(math.cos(math.radians(th)) * n + math.sin(math.radians(th)) * t)
                    hits.append(bool(search(ctx, a.seat, p, n, d, need_clear=not a.joints_only, max_clear=3)[0]))
                obl[f"{th:g}"] = hits
            row["oblique"] = obl
            rows.append(row)
            if m % 20 == 0:
                print(f"{o}: {m + 1}/{len(idx)} points, {time.time() - t0:.0f} s", flush=True)
        summ = {}
        for rg in ("face", "edge", "flank", "tip"):
            sel = [r for r in rows if r["region"] == rg]
            if not sel:
                continue
            s = dict(n=len(sel), normal=sum(r["normal"] for r in sel), twist=sum(len(r.get("twist", [])) == 2 for r in sel))
            for th in thetas:
                k = f"{th:g}"
                s[f"oblique{k}_any"] = sum(any(r["oblique"][k]) for r in sel)
                s[f"oblique{k}_all"] = sum(all(r["oblique"][k]) for r in sel)
                s[f"oblique{k}_share"] = round(float(np.mean([np.mean(r["oblique"][k]) for r in sel])), 3)
            summ[rg] = s
        res["orients"][o] = dict(seat=st["apex"], summary=summ, rows=rows, seconds=round(time.time() - t0, 1))
        print(o, json.dumps(summ), flush=True)
        dst.write_text(json.dumps(res, default=float) + "\n")
    print(f"-> {dst} in {time.time() - t0:.0f} s")


# ------------------------------------------------------------------------------------------------ a plan and its moves
RETRACT, SAFE = 12.0, 45.0          # mm back along the stem between near presses, and before a joint move
LIFT = (30.0, 150.0)                # mm along -x_M (up, out of the key) before the first joint move from the homing:
#                                     the low lift first, the high one over the seats' blocks when the low one meets them


def _slerp(T0, T1, s):
    from scipy.spatial.transform import Rotation, Slerp
    r = Slerp([0.0, 1.0], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))(s).as_matrix()
    return C.Tm(r, (1 - s) * T0[:3, 3] + s * T1[:3, 3])


def segment_samples(seg, q0):
    """Joints along a move ('L' straight in the flange's pose every 2 mm and 1 deg, 'J' in joints every 1 deg of the
    largest joint's travel), each from the previous; None where the inverse kinematics has no solution."""
    kind, T0, T1 = seg[:3]
    if kind == "J":                                     # to the solution nearest the hint (the action's own branch)
        q1 = ik_near(T1[None], seg[3] if len(seg) > 3 else q0, branch=BR[0])[0]
        if not np.isfinite(q1).all():
            return None
        n = max(2, int(math.ceil(math.degrees(np.max(np.abs(q1 - q0))))) + 1)
        return [(None, q0 + (q1 - q0) * s) for s in np.linspace(0.0, 1.0, n)]
    mm, deg = C.pose_error(T0, T1)
    n = max(2, int(math.ceil(max(mm / 2.0, deg))) + 1)
    out, q = [], q0
    for s in np.linspace(0.0, 1.0, n):
        T = _slerp(T0, T1, s)
        q = ik_near(T[None], q, branch=BR[0])[0]
        if not np.isfinite(q).all():
            return None
        out.append((T, q))
    return out


def back_along(T_BE, ctx, mm):
    """The flange pose moved mm back along the tool's axis (toward the HEX-E)."""
    z_B = (T_BE @ ctx["T_EM"])[:3, 2]
    return C.Tm(T_BE[:3, :3], T_BE[:3, 3] - z_B * mm)


def build_plan(ctx, seat_i, a):
    """Probe presses on one seat's tip: normal presses at the pad points a.spacing mm apart that a press reaches, oblique
    presses (a.thetas at a.azimuths) at every a.obl_every-th of them, twists of +-15 deg at every 4th face and edge
    point, and the reference press at the point nearest the face's apex first, every 25 actions and last."""
    st = ctx["g"]["seats"][seat_i]
    T = np.asarray(st["T_AF"], float) @ T_FP
    P_A, N_A = C.apply(T, ctx["pts_P"]), ctx["nrm_P"] @ T[:3, :3].T
    axis = -np.asarray(st["T_AF"], float)[:3, 2]
    regs = [region(v) for v in ctx["nrm_P"]]
    idx = [i for i in thin_idx(ctx["pts_P"], a.spacing) if regs[i]]
    i_ref = int(np.argmin(np.linalg.norm(ctx["pts_P"] - [8.5, 0.0, 10.86], axis=1)))
    thetas = [float(v) for v in a.thetas.split(",") if v]
    azim = np.arange(a.azimuths) * 360.0 / a.azimuths
    acts, k_face = [], 0

    def add(aid, kind, i, d, theta=0.0, az=0.0, twist=0.0):
        ok, info = search_sphere(ctx, seat_i, P_A[i], N_A[i], d, need_clear=not a.joints_only)
        if not ok or (twist and not twist_ok(ctx, seat_i, P_A[i], N_A[i], info, twist, need_clear=not a.joints_only)):
            return False
        acts.append(dict(id=aid, kind=kind, i=int(i), region=regs[i], theta=theta, azimuth=az, twist=twist,
                         p_P=np.round(ctx["pts_P"][i], 3).tolist(), n_P=np.round(ctx["nrm_P"][i], 4).tolist(),
                         n_A=np.round(N_A[i], 5).tolist(), d_A=np.round(d, 4).tolist(), u_A=info["u"], T_rest=info["T_rest"], T_deep=info["T_deep"],
                         q_rest=info["q"],
                         levels=[0.5, 1.0, 2.0] if regs[i] in ("flank", "tip") else [0.5, 1.0, 2.0, 3.0]))
        return True

    for m, i in enumerate(idx):
        n = N_A[i]
        if not add(f"probe-{regs[i]}-{i:03d}", "press", i, -n):
            continue
        t1, t2 = tangent_basis(n, axis)
        if m % a.obl_every == 0:
            for th in thetas:
                for az in azim:
                    t = math.cos(math.radians(az)) * t1 + math.sin(math.radians(az)) * t2
                    d = -(math.cos(math.radians(th)) * n + math.sin(math.radians(th)) * t)
                    add(f"probe-{regs[i]}-{i:03d}-o{th:g}-a{az:g}", "oblique", i, d, th, az)
        if regs[i] in ("face", "edge"):
            k_face += 1
            if k_face % 4 == 0:
                for tw in (-15.0, 15.0):
                    add(f"probe-{regs[i]}-{i:03d}-t{tw:+g}", "twist", i, -n, twist=tw)
    # order: by the stem's direction (its azimuth about the vertical, then its elevation), the reference press between
    key = [(round(math.degrees(math.atan2(x["u_A"][1], x["u_A"][0])) / 20.0), x["u_A"][2]) for x in acts]
    acts = [acts[j] for j in sorted(range(len(acts)), key=lambda j: key[j])]
    ref = acts.pop() if add("probe-ref", "reference", i_ref, -N_A[i_ref]) else None
    out = []
    for j, x in enumerate(acts):
        if ref is not None and j % 25 == 0:
            out.append(dict(ref, id=f"probe-ref-{j // 25:02d}"))
        out.append(x)
    if ref is not None:
        out.append(dict(ref, id="probe-ref-final"))
    return out, P_A, N_A


def own_moves(ctx, act):
    """An action's own segments from its rest: the press to its depth, a twist there (about the pad's normal through the
    sphere's centre, in 16 steps, and back) and back to the rest."""
    Tr, Td = act["T_rest"], act["T_deep"]
    segs = [("L", Tr, Td)]
    if act["kind"] == "twist":
        n_B = ctx["T_BA"][:3, :3] @ np.asarray(act["n_A"], float)
        TM = Td @ ctx["T_EM"]
        c_B = TM[:3, 3] + TM[:3, 2] * tool_end_z(ctx["tool"])
        T = Td
        for s in np.linspace(0.0, 1.0, 16)[1:]:
            T1 = C.turn_about(Td, c_B, n_B, act["twist"] * s)
            segs.append(("L", T, T1))
            T = T1
        segs.append(("L", T, Td))
    segs.append(("L", Td, Tr))
    return segs


def travel_options(ctx, T_prev, q_prev, act):
    """The ways from the previous rest (or the homing pose) to an action's rest, in the order they are tried: straight
    (back RETRACT mm along the stem, across, in) when the stems differ by 20 deg or less and the rests by 30 mm or less; a
    joint move between poses SAFE mm back along each stem; that joint move by way of the hub over the seat."""
    Tr, qa = act["T_rest"], act["q_rest"]
    a1 = back_along(Tr, ctx, SAFE)
    Th, qh = ctx["hub"]
    if T_prev is None:                                  # from the homing: out of the key along -x_M (up) first
        T0 = fk_many(q_prev[None])[0]
        up = [C.Tm(T0[:3, :3], T0[:3, 3] + T0[:3, :3] @ ctx["T_EM"][:3, :3] @ np.array([-mm, 0.0, 0.0])) for mm in LIFT]
        return ([("joint", [("L", T0, u), ("J", u, a1, qa), ("L", a1, Tr)]) for u in up]
                + [("hub", [("L", T0, up[-1]), ("J", up[-1], Th, qh), ("J", Th, a1, qa), ("L", a1, Tr)])])
    out = []
    mm, deg = C.pose_error(T_prev, Tr)
    if deg <= 20.0 and mm <= 30.0:
        b0, b1 = back_along(T_prev, ctx, RETRACT), back_along(Tr, ctx, RETRACT)
        out.append(("straight", [("L", T_prev, b0), ("L", b0, b1), ("L", b1, Tr)]))
    a0 = back_along(T_prev, ctx, SAFE)
    out += [("joint", [("L", T_prev, a0), ("J", a0, a1, qa), ("L", a1, Tr)]),
            ("hub", [("L", T_prev, a0), ("J", a0, Th, qh), ("J", Th, a1, qa), ("L", a1, Tr)])]
    return out


def sample_all(segs, q, owner):
    """Samples of consecutive segments from joints q: [(segment index, kind, k, n, T or None, q)], the joint flags."""
    out, why = [], []
    for si, seg in enumerate(segs):
        sm = segment_samples(seg, q)
        if sm is None:
            return out, why + [f"no IK on segment {si}"]
        for k, (T, qq) in enumerate(sm):
            why += [f"{w} on segment {si}" for w in joint_flags(qq, owner)]
            out.append((si, seg[0], k, len(sm), T, qq))
        q = sm[-1][1]
    return out, sorted(set(why))


def clear_of(ctx, sms, contact, joints_only=False):
    """The least clearance of each group over samples (every straight-move sample, every 5th of a joint move)."""
    cl = {}
    if joints_only:
        return cl
    for si, kind, k, n, T, qq in sms:
        if kind == "L" or k % 5 == 0 or k == n - 1:
            for g_, v in clearance(qq, ctx, contact).items():
                if v[0] < cl.get(g_, (1e9, ""))[0]:
                    cl[g_] = v
    return cl


def pick_travel(ctx, T_prev, q, act, owner=False, joints_only=False):
    """The first of travel_options whose joints and clearances pass (else the first): (way, samples, flags, clearances)."""
    best = None
    for w, segs in travel_options(ctx, T_prev, q, act):
        sm, why = sample_all(segs, q, owner)
        cl = clear_of(ctx, sm, None, joints_only) if not why else {}
        why = list(why) + clear_flags(cl)
        if best is None or not why:
            best = (w, sm, why, cl)
        if not why:
            break
    return best


def hub(ctx, seat_i):
    """A pose over the seat for joint moves that would pass a singularity or leave a window: the sphere's centre 90 mm over
    the pad's apex and 60 mm toward the robot, the stem up and back toward the robot, joint 6 near 0; (T_BE, q)."""
    apex = np.asarray(ctx["g"]["seats"][seat_i]["apex"], float)
    for dx, dz, ux in ((-60.0, 90.0, -0.5), (-40.0, 120.0, -0.3), (-80.0, 70.0, -0.7), (0.0, 110.0, 0.0)):
        u = np.array([ux, 0.0, 1.0]) / math.hypot(ux, 1.0)
        T = pose_B(ctx, frames_for(u[None]), (apex + [dx, 0.0, dz])[None], tool_end_z(ctx["tool"]))[0]
        ok, info = try_poses(ctx, T, T, True, False, None, need_clear=False)
        if ok:
            return info["T_rest"], info["q"]
    raise RuntimeError("no hub pose over the seat")


def start_q(ctx):
    """The joints 12 mm out of key 2 along -x_M, where the program's freedrive homing leaves the robot (as
    scripts/ur5_tip_calibration_holder_paths.py starts)."""
    T = np.asarray(ctx["T_BE_home"], float)
    T = C.Tm(T[:3, :3], T[:3, 3] + T[:3, :3] @ ctx["T_EM"][:3, :3] @ np.array([-12.0, 0.0, 0.0]))
    return ik_near(T[None], ctx["q_home"], branch=BR[0])[0]


def cmd_replay(a):
    """A probe plan on one seat (build_plan) replayed move by move: the joints at every sample, the clearances at every
    sample of a straight move and every 5th of a joint move, from 12 mm out of key 2 (on the seat block, out of the key
    nearer the seat)."""
    t0 = time.time()
    HOME_KEY[0] = 1 if a.seat == 2 else 0
    ctx = context(dict(orient=a.orient, seat_h=a.h), tool=a.tool)
    ctx["seat"] = a.seat
    ctx["hub"] = hub(ctx, a.seat)
    acts, P_A, N_A = build_plan(ctx, a.seat, a)
    print(f"{len(acts)} actions on seat {a.seat + 1} ({a.orient}), built in {time.time() - t0:.0f} s", flush=True)
    q = start_q(ctx)
    T_prev, rows, n_way = None, [], {}
    for x in acts:
        contact = (a.seat, P_A[x["i"]], EXCL[ctx["tool"]])
        w, tsm, why, cl = pick_travel(ctx, T_prev, q, x, a.owner, a.joints_only)
        n_way[w] = n_way.get(w, 0) + 1
        q_t = tsm[-1][5] if tsm else q
        osm, why_o = sample_all(own_moves(ctx, x), q_t, a.owner)
        why = [v for v in why if "mm to" not in v] + [f"{v} (own)" for v in why_o]
        for g_, v in clear_of(ctx, osm, contact, a.joints_only).items():
            if v[0] < cl.get(g_, (1e9, ""))[0]:
                cl[g_] = v
        j5r, j6r = [1e9, -1e9], [1e9, -1e9]
        for sms in (tsm, osm):
            for si, kind, k, n, T, qq in sms:
                j5r = [min(j5r[0], math.degrees(qq[4])), max(j5r[1], math.degrees(qq[4]))]
                j6r = [min(j6r[0], math.degrees(qq[5])), max(j6r[1], math.degrees(qq[5]))]
        q = osm[-1][5] if osm else q_t
        why = why + clear_flags(cl)
        rows.append(dict(id=x["id"], kind=x["kind"], region=x["region"], travel=w, flags=why,
                         clear={k: [round(v[0], 2), v[1]] for k, v in cl.items()},
                         j5_deg=np.round(j5r, 1).tolist(), j6_deg=np.round(j6r, 1).tolist()))
        if why:
            print(f"  {x['id']}: {'; '.join(why)}", flush=True)
        T_prev = x["T_rest"]
    least = {}
    for r in rows:
        for k, (v, part) in r["clear"].items():
            if v < least.get(k, (1e9, ""))[0]:
                least[k] = (v, part)
    kinds = {}
    for x in acts:
        kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
    plan = dict(version=1, created=time.strftime("%Y-%m-%d"), kind="probe", seat=a.seat, orient=a.orient,
                seat_apex_A=ctx["g"]["seats"][a.seat]["apex"], T_BA=ctx["T_BA"].round(6).tolist(),
                T_EM=ctx["T_EM"].round(6).tolist(), probe=PROBE, tool=ctx["tool"],
                notes=[f"{time.strftime('%Y-%m-%d %H:%M')}, workstation: scripts/ur5_inverted_rig.py replay. A probe plan for "
                       "the inverted rig, written for the path check and the page; the lab program has no probe mode yet."],
                actions=[{k: (np.round(v, 5).tolist() if isinstance(v, np.ndarray) else v) for k, v in x.items()}
                         for x in acts])
    odir = Path(a.out_dir).resolve() if a.out_dir else OUT / "data"
    tt = "" if a.tool == "sphere5" else f"_{a.tool}"
    dst = odir / f"{STEM}_plan_{a.orient}_seat{a.seat + 1}{tt}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(plan, default=float) + "\n")
    out = dict(plan=str(dst.relative_to(ROOT)) if ROOT in dst.parents else str(dst), n=len(rows), kinds=kinds, flagged=sum(bool(r["flags"]) for r in rows),
               travel=n_way,
               least={k: [round(v[0], 2), v[1]] for k, v in least.items()}, joints_only=bool(a.joints_only),
               j5_deg=[min(r["j5_deg"][0] for r in rows), max(r["j5_deg"][1] for r in rows)],
               j6_deg=[min(r["j6_deg"][0] for r in rows), max(r["j6_deg"][1] for r in rows)],
               seconds=round(time.time() - t0, 1), rows=rows)
    dst2 = odir / f"{STEM}_paths_{a.orient}_seat{a.seat + 1}{tt}{'_joints' if a.joints_only else ''}.json"
    dst2.write_text(json.dumps(out, default=float) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)
    print(f"-> {dst}, {dst2}")


def cmd_home_check(a):
    """Homing with each tool on each of the plate's keys used: the least clearances at the program's start (12 mm out of
    the key) and after each lift of LIFT, written to data/<stem>_homing.json."""
    keys = (0, 1) if PLATE[0] == "block" else (1,)
    rows = []
    for key in keys:
        HOME_KEY[0] = key
        for tool in TOOLS:
            ctx = context(tool=tool)
            q = start_q(ctx)
            T0 = fk_many(q[None])[0]
            for mm in (0.0,) + LIFT:
                T = C.Tm(T0[:3, :3], T0[:3, 3] + T0[:3, :3] @ ctx["T_EM"][:3, :3] @ np.array([-mm, 0.0, 0.0]))
                qq = ik_near(T[None], q, branch=BR[0])[0]
                cl = clearance(qq, ctx) if np.isfinite(qq).all() else {}
                least = min(((v[0], g_, v[1]) for g_, v in cl.items() if g_ != "port"), default=(None, None, None))
                rows.append(dict(key=key + 1, tool=tool, lift_mm=mm, ik=bool(cl), least_mm=least[0], group=least[1],
                                 part=least[2], flags=clear_flags(cl) if cl else ["no IK"],
                                 q_deg=np.round(np.degrees(qq), 1).tolist() if cl else None))
            print(json.dumps(rows[-1]), flush=True)
    dst = OUT / "data" / f"{STEM}_homing.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(dict(lift_mm=LIFT, rows=rows), default=float) + "\n")
    print("->", dst)


# ------------------------------------------------------------------------------------------------ the film
FILM_PICKS = (("press", "face", 0.0, "Press on the face"), ("press", "edge", 0.0, "Press on an edge"),
              ("press", "flank", 0.0, "Press on a flank"), ("press", "tip", 0.0, "Press on the tip's end"),
              ("oblique", "face", 35.0, "Push 35 deg off the normal"), ("twist", None, 0.0, "Twist of 15 deg at depth"))


def cell_scene(ctx, w, h, work):
    """The cell's MuJoCo scene (scripts/ur5_cell_render.py) with plate v2e where the low sphere stood, its three held
    tips on the seats, and the probe base and its tool on the HEX-E in place of the fork and the finger."""
    import trimesh
    import ur5_cell_render as RN
    cad = OUT / "cad"
    dummy = ROOT / "logs/20261029-inverted_rig/dummy_0p01mm.stl"     # inside the repository: the render logs paths relative to it
    dummy.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(0.01, 0.01, 0.01)).apply_translation((0.0, 0.0, -3000.0)).export(dummy)
    fx = RN.fixtures()
    fx["sphere"], fx["T_EM"] = ctx["T_BA"], ctx["T_EM"]
    CELL.INDENTER_STL["sphere"] = cad / f"{STEM}_plate_v2e.stl"
    CELL.INDENTER_STL["rod"], RN.TALL_STL, RN.HOLDER_STL = dummy, dummy, dummy
    probe = {"fork": trimesh.load(cad / f"{STEM}_probe_base.stl", force="mesh"),
             "clamp_plate": trimesh.load(cad / f"{STEM}_tool_{ctx['tool']}.stl", force="mesh"),
             "dip_link": trimesh.load(dummy, force="mesh")}
    orig_fm, orig_tp, orig_T, orig_st = CELL.finger_mesh, RN.tip_parts_F, CELL.tiphold_T_IF, CELL.TIPHOLD["stations"]
    calls = [0]

    def tp():
        calls[0] += 1
        return [] if calls[0] == 1 else orig_tp()           # the robot carries no tip; the seats do

    seats = [np.asarray(s["T_AF"], float) for s in ctx["g"]["seats"]]
    CELL.finger_mesh = lambda name, trim=None: probe[name].copy() if name in probe else orig_fm(name)   # noqa: E731
    RN.tip_parts_F = tp
    CELL.TIPHOLD["stations"] = tuple((0.0, "up") for _ in seats)
    CELL.tiphold_T_IF = lambda t=None, i=0: C.inv(fx["holder"]) @ ctx["T_BA"] @ seats[i]        # noqa: E731
    apex = np.asarray(ctx["g"]["seats"][ctx["seat"]]["apex"], float)
    RN.CAMERAS["plate"] = (tuple(C.apply(ctx["T_BA"], np.array([-60.0, -640.0, 430.0]))),
                           tuple(C.apply(ctx["T_BA"], np.array([0.0, 0.0, 110.0]))), 42.0)
    RN.CAMERAS["seat"] = (tuple(C.apply(ctx["T_BA"], apex + [-230.0, -300.0, 160.0])), tuple(C.apply(ctx["T_BA"], apex)), 30.0)
    try:
        xml, _, _ = RN.build(work, fx, w, h)
    finally:
        CELL.finger_mesh, RN.tip_parts_F, CELL.tiphold_T_IF = orig_fm, orig_tp, orig_T
        CELL.TIPHOLD["stations"] = orig_st
    return xml, RN


def cmd_film(a):
    """One press of each kind on the middle seat, from the homing pose, every move as the replay makes it, filmed from
    beside the plate and close to the seat, each under a label bar."""
    import shutil
    import subprocess
    import tempfile
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    t0 = time.time()
    ctx = context(dict(orient=a.orient, seat_h=a.h), tool=a.tool)
    ctx["seat"] = a.seat
    ctx["hub"] = hub(ctx, a.seat)
    a.spacing, a.thetas, a.azimuths, a.obl_every, a.joints_only = 3.0, "35", 4, 6, False
    acts, P_A, N_A = build_plan(ctx, a.seat, a)
    picks = []
    for kind, reg, th, label in FILM_PICKS:
        hit = [x for x in acts if x["kind"] == kind and (reg is None or x["region"] == reg) and x["theta"] == th
               and x["twist"] >= 0.0]
        if hit:
            picks.append((hit[len(hit) // 2], label))
    print(f"{len(picks)} actions picked in {time.time() - t0:.0f} s", flush=True)
    ctx["q_home"] = start_q(ctx)
    work = Path(tempfile.mkdtemp(prefix="inverted_rig_film_"))
    w, hh = (int(v) for v in a.size.lower().split("x"))
    xml, RN = cell_scene(ctx, w, hh, work)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    frames = work / "frames"
    frames.mkdir()
    q = np.asarray(ctx["q_home"], float)
    T_prev, k = None, 0
    ren = mujoco.Renderer(m, hh, w)
    try:
        for x, label in picks:
            tsm = pick_travel(ctx, T_prev, q, x)[1]            # as the replay moves
            qt = tsm[-1][5] if tsm else q
            osm, _ = sample_all(own_moves(ctx, x), qt, False)
            qs = [s[5] for s in tsm[::max(1, len(tsm) // 40)]] + [s[5] for s in osm[::max(1, len(osm) // 40)]] + [osm[-1][5]] * 6
            for qq in qs:
                RN.set_pose(m, d, qq)
                ims = []
                for cam in ("plate", "seat"):
                    ren.update_scene(d, camera=cam)
                    ims.append(Image.fromarray(ren.render()))
                bar = Image.new("RGB", (2 * w, 40), (255, 255, 255))
                ImageDraw.Draw(bar).text((12, 6), label, fill=(0, 0, 0), font=font)
                out = Image.new("RGB", (2 * w, hh + 40), (255, 255, 255))
                out.paste(bar, (0, 0))
                out.paste(ims[0], (0, 40))
                out.paste(ims[1], (w, 40))
                out.save(frames / f"f{k:05d}.png")
                k += 1
            print(f"{label}: {x['id']}, {len(qs)} frames", flush=True)
            q = osm[-1][5]
            T_prev = x["T_rest"]
    finally:
        ren.close()
    media = OUT / "media"
    media.mkdir(parents=True, exist_ok=True)
    dst = media / f"{STEM}_film.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(frames / "f%05d.png"), "-vf",
                    f"scale={2 * w}:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", str(dst)], check=True)
    shutil.copy(frames / f"f{k // 2:05d}.png", media / f"{STEM}_film_poster.png")
    shutil.rmtree(work, ignore_errors=True)
    print(f"{k} frames -> {dst.relative_to(ROOT)} in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plate", default="v2e", choices=("v2e", "block"),
                    help="plate v2e (2026-10-29), or the seat block (2026-10-30, its own folder)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ik = sub.add_parser("ik-check")
    ik.add_argument("--n", type=int, default=200)
    sub.add_parser("build")
    sub.add_parser("home-check")
    sb = sub.add_parser("seat-beam")
    sb.add_argument("--orient", default=",".join(ORIENT))
    sb.add_argument("--wall", type=float, default=2.0)
    sb.add_argument("--infill", type=float, default=0.25)
    fm = sub.add_parser("film")
    fm.add_argument("--orient", default=LAYOUT["orient"])
    fm.add_argument("--h", type=float, default=None)
    fm.add_argument("--seat", type=int, default=1)
    fm.add_argument("--tool", default="sphere5", choices=tuple(SPHERES))
    fm.add_argument("--size", default="720x560")
    fm.add_argument("--fps", type=int, default=20)
    rp = sub.add_parser("replay")
    rp.add_argument("--orient", default=LAYOUT["orient"])
    rp.add_argument("--tool", default="sphere5", choices=tuple(SPHERES))
    rp.add_argument("--h", type=float, default=None)
    rp.add_argument("--seat", type=int, default=1)
    rp.add_argument("--spacing", type=float, default=2.0, help="normal presses at pad points this far apart (mm)")
    rp.add_argument("--thetas", default="20,35")
    rp.add_argument("--azimuths", type=int, default=4)
    rp.add_argument("--obl-every", type=int, default=3)
    rp.add_argument("--owner", action="store_true", help="joint 6 to +60 deg (the owner present)")
    rp.add_argument("--joints-only", action="store_true")
    rp.add_argument("--out-dir", default="", help="where the plan and its check go (default the experiment's data/)")
    r = sub.add_parser("reach")
    r.add_argument("--orient", default=",".join(ORIENT))
    r.add_argument("--tool", default="sphere5", choices=tuple(SPHERES) + tuple(FLATS))
    r.add_argument("--h", type=float, default=None, help="the seats' apex height over the base's underside (mm)")
    r.add_argument("--seat", type=int, default=1, help="which seat's tip is pressed (0, 1, 2 from -y)")
    r.add_argument("--thin", type=float, default=1.0, help="pad points at least this far apart (mm); 0 for all 812")
    r.add_argument("--thetas", default="20,35", help="push directions this far from the inward normal (deg)")
    r.add_argument("--azimuths", type=int, default=6)
    r.add_argument("--no-twist", action="store_true")
    r.add_argument("--joints-only", action="store_true", help="joints and windows only, no clearances (no meshes)")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--out", default="")
    r.add_argument("--resume", action="store_true", help="keep the orientations already in --out")
    r.add_argument("--part", default="", help="k/n: this job's share of the points (merge the parts with merge-reach)")
    a = ap.parse_args()
    if a.plate == "block":
        global OUT, STEM
        PLATE[0] = "block"
        OUT, STEM = BLOCK_OUT, BLOCK_STEM
    if getattr(a, "h", 0.0) is None:                    # the plate's own seat height
        a.h = (SEAT_BLOCK if PLATE[0] == "block" else LAYOUT)["seat_h"]
    dict(**{"ik-check": cmd_ik_check, "seat-beam": cmd_seat_beam, "home-check": cmd_home_check}, build=cmd_build,
         reach=cmd_reach, replay=cmd_replay, film=cmd_film)[a.cmd](a)


if __name__ == "__main__":
    main()
