#!/usr/bin/env python3
"""Printed parts that put an ATI Mini45 between the UR5's HEX-E and calibration mount v3, and a hook cross for its
calibration with known masses (owner, 2026-10-23: "we can probably calibrate the ATI with just a 3dp mount to the HEX-E,
with the current indenter setup"; the handoff docs/experiments/20261023-ati_ft_calibration/).

Stack, from the flange: HEX-E, mount v3's adapter (its M6 x 20 tightened first, heads on its top face), the ATI base
(18 mm, over the M6 heads, four M3 x 20 into the adapter's heat-set inserts outside the ATI's footprint), the Mini45
(six M3 x 10 from the base's underside into its mounting side, their heads 12 mm up in counterbores above the M6 pockets), the ATI top (six M3 into the ATI's tool side, four brass inserts in mount
v3's foot pattern), then v3's standoff-fork unchanged. For the calibration the hook cross takes the fork's place on the
ATI top: hook points 60 mm out along +-x and +-y and one on the axis.

The Mini45's interface is from ATI's drawing 9230-05-1094 as quoted by a web search (ATI's site refused connections on
2026-10-23): mounting side six M3 x 0.5 on a 39.24 mm bolt circle, tool side six M3 x 0.5 on 28.2 mm, 3 mm dowel holes, a
9.4 mm bore; the six holes stand in pairs. THE PAIRS' ANGLES, THE DOWELS' PLACES, THE THREAD DEPTHS AND THE HEIGHT ARE NOT
CHECKED: measure them on the transducer and set ATI_* below before printing. Frame: z along the tool axis from the v3
adapter's top face, as mount v3's M frame minus 15 mm.

    .venv/bin/python scripts/ati_stack_mount.py [--clock DEG]     writes docs/experiments/20261023-ati_ft_calibration/cad/*.stl
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
import ur5_calibration_mount as UM  # noqa: E402  (MeshK: manifold booleans; v3's patterns)

OUT = ROOT / "docs/experiments/20261023-ati_ft_calibration"
# --- the Mini45 (to be measured) ---
ATI_D = 45.0                    # mm, outer diameter
ATI_H = 15.7                    # mm, mounting face to tool face
ATI_MOUNT_BC, ATI_TOOL_BC = 39.24, 28.2
ATI_PAIR_DEG = 20.0             # the angle between the two holes of a pair (three pairs, 120 deg apart): UNMEASURED
ATI_THREAD = 4.0                # mm of M3 thread in the ATI's faces: UNMEASURED
# --- mount v3's interfaces ---
M6_ANG = [45.0, 135.0, 225.0, 315.0]                        # its M6 x 20 on a 50 mm circle, heads on the adapter's top
M6_R, M6_HEAD_D, M6_HEAD_K = 25.0, 10.0, 6.0
M3_ANG, M3_R = [0.0, 90.0, 198.0, 270.0], 25.5              # the foot's M3 into the adapter's inserts
DISC_D = 63.0
BASE_T, TOP_T = 18.0, 11.0
ATI_CB_UP = 12.0                # the ATI screws' heads 12 mm up from the base's underside, above the M6 pockets (6.5)
POCKET_D, POCKET_H = 12.5, 6.5
CB_D, CB_H = 6.5, 3.5                                       # M3 socket head counterbore
INS_HOLE, INS_DEPTH, INS_CH = 4.0, 9.0, 0.3                 # M3 x 5.7 brass insert (v3's adapter)


def ati_holes(bc, clock_deg):
    r = bc / 2
    return [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
            for k in range(3) for a in (clock_deg + 120 * k - ATI_PAIR_DEG / 2, clock_deg + 120 * k + ATI_PAIR_DEG / 2)]


def xy(r, a):
    return r * math.cos(math.radians(a)), r * math.sin(math.radians(a))


def base(K, clock):
    s = K.cyl("z", (0.0, 0.0), DISC_D / 2, 0.0, BASE_T)
    cut = []
    for a in M6_ANG:                                         # the M6 heads under the base
        cut.append(K.cyl("z", xy(M6_R, a), POCKET_D / 2, -1.0, POCKET_H))
    for a in M3_ANG:                                         # M3 x 16 into the adapter's inserts, heads outside the ATI
        c = xy(M3_R, a)
        cut += [K.cyl("z", c, 1.7, -1.0, BASE_T + 1.0), K.cyl("z", c, CB_D / 2, BASE_T - CB_H, BASE_T + 1.0)]
    for c in ati_holes(ATI_MOUNT_BC, clock):                 # M3 x 10 up into the ATI's mounting side; deep counterbores,
        cut += [K.cyl("z", c, 1.7, -1.0, BASE_T + 1.0), K.cyl("z", c, 3.1, -1.0, ATI_CB_UP)]   # merging with the pockets
    return K.cut_all(s, cut)


def top(K, clock):
    s = K.cyl("z", (0.0, 0.0), DISC_D / 2, 0.0, TOP_T)
    cut = []
    for c in ati_holes(ATI_TOOL_BC, clock):                  # M3 down into the ATI's tool side, heads flush
        cut += [K.cyl("z", c, 1.7, -1.0, TOP_T + 1.0), K.cyl("z", c, 3.1, TOP_T - CB_H, TOP_T + 1.0)]
    for a in M3_ANG:                                         # inserts for the fork's foot (or the hook cross)
        c = xy(M3_R, a)
        cut += [K.cyl("z", c, INS_HOLE / 2, TOP_T - INS_DEPTH, TOP_T + 1.0),
                K.cyl("z", c, INS_HOLE / 2, TOP_T - INS_CH, TOP_T + 0.01, r1=INS_HOLE / 2 + INS_CH + 0.01)]
    return K.cut_all(s, cut)


HOOKS_MM = [(60.0, 0.0, 30.0), (-60.0, 0.0, 30.0), (0.0, 60.0, 30.0), (0.0, -60.0, 30.0), (0.0, 0.0, 100.0)]


def cross(K):
    """On the ATI top's inserts: a 6 mm plate, four arms 12 x 10 mm to 70 mm out, a 16 mm post to 105 mm; a 4.5 mm
    hole through each arm at 60 mm and through the post at 100 mm for an S hook (heights above the plate's underside)."""
    parts = [K.cyl("z", (0.0, 0.0), DISC_D / 2, 0.0, 6.0)]
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        x0, x1 = sorted((0.0, sx * 70.0)) if sx else (-6.0, 6.0)
        y0, y1 = sorted((0.0, sy * 70.0)) if sy else (-6.0, 6.0)
        parts.append(K.box((x0, y0, 5.99), (x1, y1, 35.0)))
    parts.append(K.cyl("z", (0.0, 0.0), 8.0, 5.99, 105.0))
    s = K.union(parts)
    cut = []
    for a in M3_ANG:
        c = xy(M3_R, a)
        cut += [K.cyl("z", c, 1.7, -1.0, 40.0), K.cyl("z", c, CB_D / 2, 6.0 - 0.5, 40.0)]
    for hx, hy, hz in HOOKS_MM[:4]:
        ax = "y" if hx else "x"
        cut.append(K.cyl(ax, (hx, hz) if ax == "y" else (hy, hz), 2.25, -80.0, 80.0))
    cut.append(K.cyl("x", (0.0, 100.0), 2.25, -10.0, 10.0))
    return K.cut_all(s, cut)


def clock_check(clock):
    """Least wall (mm) between an ATI mounting screw's deep counterbore and the base's own M3 holes into the adapter."""
    d = []
    for c in ati_holes(ATI_MOUNT_BC, clock):
        for a in M3_ANG:
            p = xy(M3_R, a)
            d.append(math.hypot(c[0] - p[0], c[1] - p[1]) - 3.1 - 1.7)
    return min(d)


def main(clock=None):
    K = UM.MeshK()
    if clock is None:                                        # the clock of most room between counterbores and pockets
        clock = max(np.arange(0.0, 120.0, 1.0), key=clock_check)
    room = clock_check(clock)
    (OUT / "cad").mkdir(parents=True, exist_ok=True)
    rec = dict(clock_deg=float(clock), counterbore_wall_to_base_m3_mm=round(room, 2), when=time.strftime("%Y-%m-%d %H:%M"),
               unmeasured=dict(ATI_PAIR_DEG=ATI_PAIR_DEG, ATI_THREAD=ATI_THREAD, ATI_H=ATI_H, dowels="not modelled"),
               stack_mm=dict(base=BASE_T, ati=ATI_H, top=TOP_T, total_added=BASE_T + ATI_H + TOP_T),
               hooks_above_cross_underside_mm=HOOKS_MM, parts={})
    for nm, s in (("ati_base", base(K, clock)), ("ati_top", top(K, clock)), ("hook_cross", cross(K))):
        tm = UM.to_trimesh(s)
        f = OUT / f"cad/20261023-ati_stack_{nm}.stl"
        tm.export(f)
        rec["parts"][nm] = dict(file=str(f.relative_to(ROOT)), volume_mm3=round(float(s.volume()), 1),
                                bounds=np.round(np.asarray(tm.bounds), 2).tolist(), watertight=bool(tm.is_watertight))
        print(f"{nm}: {s.volume():.0f} mm3, bounds {np.round(np.asarray(tm.bounds), 1).tolist()}", flush=True)
    (OUT / "20261023-ati_stack_parts.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"clock {clock:.0f} deg: {room:.2f} mm of wall between the ATI screws' counterbores and the base's M3 holes")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clock", type=float, default=None)
    main(ap.parse_args().clock)
