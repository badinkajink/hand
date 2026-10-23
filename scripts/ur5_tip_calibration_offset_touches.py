#!/usr/bin/env python3
"""Offset touch-offs added to a plan's pad stage, so that the registration sees the sphere's place along the finger.

Every touch-off of the plans so far meets the sphere at its top, with the contact normal along -z_I, and so sees no
horizontal shift of the sphere's centre; that position rests on the homing seating's prior (0.25 mm along the finger).
An offset touch-off places its target point beside the sphere's top by (ox, oy) in I (the action's "offset", which the
program's target_pose already applies) and approaches vertically, so that a curved part of the pad meets the sphere off
its top. scripts/ur5_tip_calibration_touch_information.py --offset-mm 3 picks the targets and offsets; on sphere2's
registration twelve of them lower the contact point's standard deviation along the finger from 0.27 to 0.07 mm.

The pad meets the sphere at another travel than the target point would: the onset standoff s_on (the target point's
height above the sphere's top when the pad first meets the sphere, from the plan's nominal geometry) sets the action's
standoff to --touch-standoff + s_on and its beyond_mm to 1.2 - s_on, so that the onset comes after the same free travel as
a plain touch-off's and the approach ends 1.2 mm past it. The program prints an offset touch-off's onset relative to the
target's standoff: it should read near -s_on, which the action keeps as onset_shift_mm.

--touch-standoff also sets the plain pad touch-offs' standoff (5.0 mm for a pad newly seated: sph3t's plan).

    .venv/bin/python scripts/ur5_tip_calibration_offset_touches.py <plan.json> <touch_information.json> [--n 12]
        [--touch-standoff 5.0] --out <plan.json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_tip_calibration as C  # noqa: E402


def onset_standoff(geo, p, n, offset):
    """The target's standoff (mm above the sphere's top) at which the pad first meets the sphere: nominal geometry."""
    def gap(s):
        T_IP = geo.T_IP(p, n, 0.0, s, offset)
        return geo.surface_distance(C.apply(C.inv(T_IP), geo.c_I))[0] - geo.R
    lo, hi = -4.0, 4.0
    if gap(hi) <= 0 or gap(lo) >= 0:
        return None
    for _ in range(50):
        m = 0.5 * (lo + hi)
        lo, hi = (lo, m) if gap(m) > 0 else (m, hi)
    return hi


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plan")
    ap.add_argument("info")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--touch-standoff", type=float, default=None, help="mm of free travel before every pad touch-off's "
                    "onset (default: the plan's first pad touch-off's standoff)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    info = json.loads(Path(a.info).read_text())
    geo = C.Geometry(plan)
    pad = [x for x in plan["actions"] if x.get("stage") == "pad"]
    base = a.touch_standoff if a.touch_standoff is not None else float(pad[0].get("standoff", 1.5))
    for x in pad:
        x["standoff"] = base
    new = []
    for k, e in enumerate(info["added"][:a.n]):
        i, off = int(e["index"]), [float(v) for v in e["offset_I"]]
        p, n = geo.pts[i], geo.nrm[i]
        s_on = onset_standoff(geo, p, n, off)
        if s_on is None:
            print(f"  point {i} offset {off}: no onset within 4 mm, left out")
            continue
        new.append(dict(id=f"pad-off-{k:02d}", stage="pad", kind="touch", role="offset",
                        target=dict(frame="P", p=[round(float(v), 4) for v in p], n=[round(float(v), 5) for v in n],
                                    index=i, region=e.get("region", "")),
                        twist=0.0, offset=off, standoff=round(base + s_on, 3), beyond_mm=round(1.2 - s_on, 3),
                        v=float(pad[0].get("v", 0.00015)), stop_N=float(pad[0].get("stop_N", 0.5)),
                        fit_N=float(pad[0].get("fit_N", 0.4)), onset_shift_mm=round(-s_on, 3)))
        print(f"  pad-off-{k:02d}: point {i} ({e.get('region', '')}) offset {off} mm, onset standoff {s_on:+.2f} mm")
    last = max(j for j, x in enumerate(plan["actions"]) if x.get("stage") == "pad")
    plan["actions"][last + 1:last + 1] = new
    note = (f"{time.strftime('%Y-%m-%d %H:%M')}, workstation: {len(new)} offset touch-offs added after the pad stage from "
            f"{Path(a.info).name} (scripts/ur5_tip_calibration_offset_touches.py), every pad touch-off at {base:g} mm of free "
            f"travel; from {Path(a.plan).name}")
    plan["notes"] = (plan.get("notes") or []) + [note] if isinstance(plan.get("notes"), list) else [plan.get("notes"), note]
    Path(a.out).write_text(json.dumps(plan))
    print(f"{len(pad)} plain and {len(new)} offset touch-offs -> {a.out}")


if __name__ == "__main__":
    main()
