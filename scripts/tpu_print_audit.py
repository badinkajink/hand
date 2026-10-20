#!/usr/bin/env python3
"""Travel, retraction and per-part mass audit of PrusaSlicer G-code for the 85A prints (2026-10-19 night).

Owner, 2026-10-19 20:20: "the stringing is quite bad right now on the tpu prusa print. i worry this will harm
reproducibility ... maybe it's because we're printing so many in sequence?" Per plate: layers, retractions (a G1 with
negative E and no move), retractions per layer, retracted travels longer than 20 mm (the moves that string across open
air), the longest lift of a travel, and each object's extruded mass (G1, G2 and G3 moves with positive E inside the object's
M486 label; relative E, 1.75 mm filament, 1.26 g/cm3), with the slicer's own total. The plates: the owner's good print of
2026-10-18 (sliced again from its project), the v2 tip-and-pad plate and the fit-size plate of 2026-10-19.

    python3 scripts/tpu_print_audit.py
Out: docs/experiments/20261019-v2_tip_fe_checks/print_audit.json
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/experiments/20261019-v2_tip_fe_checks/print_audit.json"
PLATES = [
    ("good_20261018", "the owner's good print of 2026-10-18 (four tips: 11/22, 16/32, block, capsule)",
     "docs/experiments/20261016-sim2real_fingertip_readiness/print/20261018-siraya85a_plate_block_cap8p5_ell16x32_ell11x22_seat2p9_g30.3mf",
     "logs/20261019-tpu_stringing/20261018-siraya85a_plate_block_cap8p5_ell16x32_ell11x22_seat2p9_g30.gcode"),
    ("tip_pad_20261019", "a's 11/22 tip and b's 11/22 pad with magnets",
     "docs/experiments/20261018-dip_link_board_mount/v2/print/20261019-dip_link_v2_tip_a_pad_b_ell11_5_85a_t2_g30.3mf",
     "logs/20261019-v2_tips_xl/20261019-dip_link_v2_tip_a_pad_b_ell11_5_85a_t2_g30.gcode"),
    ("fit_sizes_20261019", "the fit sizes A00-A20 and B00-B15, no magnets",
     "docs/experiments/20261018-dip_link_board_mount/v2/print/20261019-dip_link_v2_fit_sizes_tips_a_pads_b_85a_t2_g30.3mf",
     "logs/20261019-v2_tip_fit_sizes/20261019-dip_link_v2_fit_sizes_tips_a_pads_b_85a_t2_g30.gcode"),
]
SETTINGS = ("temperature", "first_layer_temperature", "filament_retract_length", "retract_length", "retract_speed",
            "deretract_speed", "filament_retract_lift", "filament_wipe", "wipe", "retract_before_travel", "travel_speed",
            "avoid_crossing_perimeters", "filament_travel_ramping_lift", "filament_travel_max_lift", "travel_slope",
            "extrusion_multiplier", "filament_max_volumetric_speed", "filament_settings_id", "complete_objects",
            "min_fan_speed", "max_fan_speed", "slowdown_below_layer_time")


def audit(gc, density=1.26, dia=1.75):
    area = math.pi * dia ** 2 / 4
    x = y = z = 0.0
    zl = 0.0
    cur, names, ext = None, {}, {}
    layers = retracts = 0
    pending = None
    long_tr, max_lift, total_e = 0, 0.0, 0.0
    settings = {}
    for line in open(gc):
        if line.startswith("; ") and " = " in line:
            k, v = line[2:].split(" = ", 1)
            if k in SETTINGS:
                settings[k] = v.strip()
            continue
        if line.startswith("M486 S"):
            v = int(line.split()[1][1:])
            cur = v if v >= 0 else None
            continue
        if line.startswith("M486 A") and cur is not None:
            names[cur] = line[6:].strip()
            continue
        if line.startswith(";LAYER_CHANGE"):
            layers += 1
            continue
        if line.startswith(";Z:"):
            zl = float(line[3:])
            continue
        if line[:3] not in ("G0 ", "G1 ", "G2 ", "G3 "):
            continue
        code = line.split(";")[0]
        d = {m.group(1): float(m.group(2)) for m in re.finditer(r"([XYZEIJ])([-\d.]+)", code)}
        nx, ny, nz = d.get("X", x), d.get("Y", y), d.get("Z", z)
        moved = ("X" in d or "Y" in d) and (nx != x or ny != y or line[:3] in ("G2 ", "G3 "))
        e = d.get("E", 0.0)
        if e < 0 and not moved:
            retracts += 1
            pending = 0.0
        elif e > 0 and not moved:
            if pending is not None and pending > 20.0:
                long_tr += 1
            pending = None
        elif e > 0 and moved:
            total_e += e
            if cur is not None:
                ext[cur] = ext.get(cur, 0.0) + e
        elif moved and pending is not None:
            pending += math.hypot(nx - x, ny - y)
            max_lift = max(max_lift, nz - zl)
        x, y, z = nx, ny, nz
    g = lambda e_mm: round(e_mm * area * density / 1000.0, 3)      # noqa: E731
    return dict(layers=layers, objects=len(names), retractions=retracts, retractions_per_layer=round(retracts / max(layers, 1), 1),
                retracted_travels_over_20mm=long_tr, travels_over_20mm_per_layer=round(long_tr / max(layers, 1), 2),
                max_travel_lift_mm=round(max_lift, 2), mass_g_by_object={names.get(k, str(k)): g(v) for k, v in sorted(ext.items())},
                mass_g_objects=g(sum(ext.values())), mass_g_all_moves=g(total_e), settings=settings)


def main():
    out = dict(owner="2026-10-19 20:20: 'the stringing is quite bad right now on the tpu prusa print'", plates={},
               script="scripts/tpu_print_audit.py", when=time.strftime("%Y-%m-%d %H:%M"))
    for key, label, proj, gc in PLATES:
        r = audit(ROOT / gc)
        text = (ROOT / gc).read_text(errors="ignore")
        m = re.search(r"^; total filament used \[g\] = ([\d.]+)", text, re.M)
        t = re.search(r"^; estimated printing time \(normal mode\) = (.*)$", text, re.M)
        r.update(label=label, project=proj, gcode=gc, slicer_total_g=float(m.group(1)) if m else None,
                 slicer_time=t.group(1).strip() if t else None)
        out["plates"][key] = r
        print(key, {k: r[k] for k in ("objects", "layers", "retractions_per_layer", "travels_over_20mm_per_layer",
                                      "mass_g_objects", "slicer_total_g")}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("->", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
