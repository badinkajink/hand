#!/usr/bin/env python3
"""What a PrusaSlicer G-code file extrudes, layer by layer (2026-10-13).

Reads an ASCII G-code file from PrusaSlicer (relative extrusion, G1 moves and G2/G3 arcs with centre offsets I, J split
into 0.1 mm chords (2026-10-21; before that arcs were skipped), `;LAYER_CHANGE`, `;Z:`, `;TYPE:`, `;WIDTH:` comments and
`M486 S<n>` object labels) and returns, for one object, the extrusion paths of every layer: their segments, bead widths,
feature types, the extruded volume, the travels and the retractions. A path is a run of extruding moves without a
travel or a retraction between them.

    ~/miniconda3/envs/femtip/bin/python scripts/gcode_audit.py logs/20261013-eflesh_slice/v17.gcode --object 0
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict

import numpy as np

FIL_AREA = math.pi * (1.75 / 2) ** 2          # mm^2 of 1.75 mm filament
_G1 = re.compile(r"([XYZEFIJ])(-?\d*\.?\d+)")
ARC_CHORD = 0.1                                # mm: G2/G3 arcs (arc_fitting = emit_center) are split into chords this long


def arc_points(x, y, nx, ny, i, j, cw):
    """Points along a G2 (cw) or G3 arc from (x, y) to (nx, ny) about the centre (x + i, y + j), the start excluded."""
    cx, cy = x + i, y + j
    r = math.hypot(i, j)
    a0, a1 = math.atan2(y - cy, x - cx), math.atan2(ny - cy, nx - cx)
    sweep = a1 - a0
    if cw and sweep >= 0:
        sweep -= 2 * math.pi
    elif not cw and sweep <= 0:
        sweep += 2 * math.pi
    n = max(1, int(math.ceil(abs(sweep) * r / ARC_CHORD)))
    pts = [(cx + r * math.cos(a0 + sweep * k / n), cy + r * math.sin(a0 + sweep * k / n)) for k in range(1, n)]
    return pts + [(nx, ny)]


def parse(path, obj=0, skip_types=("Custom", "Wipe tower", "Skirt/Brim", "Skirt")):
    """Per-layer records for object `obj`: z, height, segments (x0, y0, x1, y1, width, type id, path id), volume,
    travels, retractions."""
    layers = []
    cur = None
    z = h = 0.0
    typ, width, label = "", 0.45, None
    x = y = 0.0
    path_id = -1
    in_path = False
    types = {}
    for line in open(path):
        if line.startswith(";"):
            if line.startswith(";LAYER_CHANGE"):
                cur = dict(z=None, h=None, seg=[], vol=0.0, travels=0, retracts=0)
                layers.append(cur)
                in_path = False
            elif line.startswith(";Z:"):
                z = float(line[3:]); cur["z"] = z if cur is not None else z
            elif line.startswith(";HEIGHT:"):
                h = float(line[8:])
                if cur is not None:
                    cur["h"] = h
            elif line.startswith(";TYPE:"):
                typ = line[6:].strip(); in_path = False
            elif line.startswith(";WIDTH:"):
                width = float(line[7:])
            continue
        if line.startswith("M486 S"):
            v = int(line.split()[1][1:])
            label = None if v < 0 else v
            in_path = False
            continue
        arc = line.startswith("G2 ") or line.startswith("G3 ")
        if not line.startswith("G1 ") and not line.startswith("G0 ") and not arc:
            continue
        code = line.split(";", 1)[0]
        f = dict((k, float(v)) for k, v in _G1.findall(code))
        nx, ny = f.get("X", x), f.get("Y", y)
        e = f.get("E", 0.0)
        moved = ("X" in f) or ("Y" in f) or arc
        mine = cur is not None and label == obj and typ not in skip_types
        if moved and e > 0 and mine:
            if not in_path:
                path_id += 1; in_path = True
            tid = types.setdefault(typ, len(types))
            pts = arc_points(x, y, nx, ny, f.get("I", 0.0), f.get("J", 0.0), line[1] == "2") if arc else [(nx, ny)]
            px, py = x, y
            for qx, qy in pts:
                cur["seg"].append((px, py, qx, qy, width, tid, path_id))
                px, py = qx, qy
            cur["vol"] += e * FIL_AREA
        elif moved and mine:
            if in_path:
                cur["travels"] += 1
            in_path = False
        elif e < 0 and mine:
            cur["retracts"] += 1; in_path = False
        x, y = nx, ny
    for L in layers:
        L["seg"] = np.array(L["seg"], float).reshape(-1, 7)
    return [L for L in layers if len(L["seg"])], {v: k for k, v in types.items()}


def summary(layers, types):
    rows = []
    for L in layers:
        s = L["seg"]
        ln = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
        rows.append(dict(z=L["z"], h=L["h"], paths=int(len(np.unique(s[:, 6]))), travels=L["travels"],
                         retracts=L["retracts"], length_mm=float(ln.sum()), volume_mm3=L["vol"],
                         width_median=float(np.median(s[:, 4])) if len(s) else None,
                         by_type={types[int(t)]: float(ln[s[:, 5] == t].sum()) for t in np.unique(s[:, 5])}))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gcode")
    ap.add_argument("--object", type=int, default=0)
    a = ap.parse_args()
    L, T = parse(a.gcode, a.object)
    S = summary(L, T)
    tot = defaultdict(float)
    for r in S:
        for k, v in r["by_type"].items():
            tot[k] += v
    print(json.dumps(dict(layers=len(S), paths=sum(r["paths"] for r in S), retracts=sum(r["retracts"] for r in S),
                          volume_mm3=round(sum(r["volume_mm3"] for r in S), 1), length_by_type_mm={k: round(v) for k, v in tot.items()}),
                     indent=1))
