#!/usr/bin/env python3
"""The +y held apex from the holder plan's apex search (owner, 2026-10-27: the first touch-offs found no contact 1.2 mm past
the expected apex). Each touch-off came down at an offset (dx, dy) in I from the apex the plan expected; its onset past the
expected contact, delta (mm, positive deeper), puts the contact at z = z_plan - delta. Two pads' domes meet lower the
farther the robot's apex stands from the held apex: z = z0 - (dx - x0)^2 / (2 Rx) - (dy - y0)^2 / (2 Ry), with Ry the
sum of the two pads' radii across the tip at the apex (2 x 7.4^2 / 8.5 = 12.9 mm) and Rx along it (fitted when five or
more touches are in, else 2 x 11^2 / 8.5 = 28.5 mm). Least squares give the true apex (x0, y0, z0 - z_plan) in I, the
shift to pass to the plan builder (--z-offset, and the lateral shift for a corrected seed).

    .venv/bin/python scripts/ur5_tip_calibration_holder_apex_fit.py SESSION_FOLDER [--plan PLAN]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RY = 2 * 7.4 ** 2 / 8.5
RX0 = 2 * 11.0 ** 2 / 8.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--plan", default=None, help="the plan with the touch-offs' offsets (default: the session's plan.json)")
    a = ap.parse_args()
    sd = Path(a.session)
    touches = json.loads((sd / "touches.json").read_text())
    plan = json.loads(Path(a.plan or sd / "plan.json").read_text())
    acts = {x["id"]: x for x in plan["actions"]}
    rows = []
    for t in touches:
        x = acts.get(t["id"])
        if x is None or not t["id"].startswith("hold-touch"):
            continue
        off = x.get("offset", (0.0, 0.0))
        dx, dy = -float(off[0]), -float(off[1])                    # the contact point moves by -offset
        delta = float(t["s0"]) - float(t.get("expected_mm", x.get("standoff", 5.0)))
        rows.append((t["id"], dx, dy, delta))
        print(f"{t['id']}: offset in I ({dx:+.2f}, {dy:+.2f}) mm, onset {delta:+.3f} mm past the expected contact")
    if not rows:
        sys.exit("no hold-touch onsets in this session")
    D = np.array([r[1:] for r in rows], float)
    z = -D[:, 2]                                                    # contact height relative to the plan's apex
    best = None
    for rx in ([RX0] if len(rows) < 5 else np.linspace(10.0, 120.0, 111)):
        A = np.c_[np.ones(len(z)), D[:, 0] / rx, D[:, 1] / RY]
        y = z + D[:, 0] ** 2 / (2 * rx) + D[:, 1] ** 2 / (2 * RY)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = float(np.sqrt(np.mean((A @ c - y) ** 2)))
        if best is None or r < best[0]:
            best = (r, rx, c)
    r, rx, c = best
    x0, y0 = c[1], c[2]
    z0 = c[0] + x0 ** 2 / (2 * rx) + y0 ** 2 / (2 * RY)
    print(f"apex fit (Ry {RY:.1f} mm, Rx {rx:.1f} mm, rms {r:.3f} mm): the held apex stands {x0:+.2f} mm along x_I, "
          f"{y0:+.2f} mm along y_I and {z0:+.2f} mm in z from the plan's")
    print(f"rebuild with --z-offset {z0:+.2f} (and the seed shifted by ({x0:+.2f}, {y0:+.2f}) mm in I)")


if __name__ == "__main__":
    main()
