#!/usr/bin/env python3
"""Pass 3 of R24 on the UR5 (owner, 2026-10-25): left-right twins of the grid, slides, twists at 5 N, and a final reference.

Owner, 11:4x: "more wrench calibration at the end of all this"; 11:5x: contact detection "pretty good on the fingertip's
right side, but pretty bad on the left ... the contact direction is in general pretty noisy". The builder's grid has 30
targets at y < -1.5 mm and 19 at y > 1.5 mm, so the two sides are calibrated unequally. Each grid target of pass 2 whose
mirror about y = 0 (the nearest surface point of the same region within 1 mm) has no target within 1 mm gets a twin
press at the same levels, in candidate turns about the vertical (0, then +-30, +-60 deg) for the clearance check to choose.
The slides of the builder keep only their pre-sliding steps (0.1 to 0.3 mm), since with the slip steps (0.5 and 1 mm)
six of seven came within the margins on mount v6. Twists at 5 N repeat pass 2's at its four face points, with 1 s holds.

    .venv/bin/python scripts/ur5_tip_calibration_r24_pass3.py [--out lab/20261025-plan_sphere_lab_r24_pass3.json]
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "docs/experiments/20261022-ur5_tip_calibration"
BASIS = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-b_v2_surface_basis.npz"
KEEP = ("id", "stage", "kind", "target", "twist", "standoff", "press_N", "moves", "hold_s", "role", "levels", "beyond_mm",
        "v", "stop_N", "fit_N", "max_depth_mm")
TWINS_TURN = (0.0, 30.0, -30.0, 60.0, -60.0)


def clean(a):
    return {k: copy.deepcopy(v) for k, v in a.items() if k in KEEP}


def region(n):
    """The builder's regions (scripts/ur5_tip_calibration_plan.py): face, edge, flank, tip end."""
    if n[2] > 0.7:
        return "tip"
    if abs(n[1]) > 0.7:
        return "flank"
    return "face" if n[0] > 0.9 else ("edge" if n[0] > 0.5 else None)


def twins(grid, pts, nrm):
    """For each target, its mirror about y = 0 on the surface, unless a target already lies within 1 mm of it."""
    have = np.array([a["target"]["p"] for a in grid])
    out = []
    for a in grid:
        p = np.array(a["target"]["p"]) * [1, -1, 1]
        d = np.linalg.norm(pts - p, axis=1)
        i = int(np.argmin(d))
        if d[i] > 1.0 or region(nrm[i]) != a["target"]["region"] or np.min(np.linalg.norm(have - pts[i], axis=1)) < 1.0:
            continue
        for tw in TWINS_TURN:
            b = clean(a)
            b.update(id=f"twin-{a['id'][5:]}-t{tw:+.0f}", twist=tw, twin_of=a["id"],
                     target=dict(frame="P", p=pts[i].tolist(), n=nrm[i].tolist(), index=i, region=a["target"]["region"]))
            out.append(b)
    return out


def run(out):
    pass1 = json.loads((EXP / "lab/20261025-plan_sphere_lab_r24_pass1.json").read_text())
    pass2 = json.loads((EXP / "lab/20261023-plan_sphere_lab_r22_pass2.json").read_text())
    full = json.loads((EXP / "plan/20261022-plan_sphere.json").read_text())
    b = np.load(BASIS)
    grid = [a for a in full["actions"] if a["stage"] == "grid" and a.get("role") != "ref"]
    acts = twins(grid, b["pts"], b["normals"])
    for a in full["actions"]:
        if a["stage"] == "shear":
            s = clean(a)
            s["moves"] = [m for m in s["moves"] if max(abs(d) for d in m["degs"]) < 4.0]      # 0.1 to 0.3 mm on R 5 mm
            acts.append(s)
    tw2 = {a["id"]: a for a in pass2["actions"] if a["stage"] == "twist"}
    twf = {a["id"]: a for a in full["actions"] if a["stage"] == "twist"}
    for k in range(4):
        t = clean(tw2.get(f"twist-{k}-2.5N") or twf[f"twist-{k}-2.5N"])
        t.update(id=f"twist-{k}-5.0N", press_N=5.0, hold_s=1.0, moves=[dict(degs=[5.0, 10.0, 15.0, 10.0, 5.0])])
        acts.append(t)
    acts.append(clean([a for a in pass2["actions"] if a["id"] == "final-ref"][0]))
    plan = {k: copy.deepcopy(v) for k, v in pass1.items() if k != "actions"}
    plan["actions"] = acts
    plan["notes"] = list(pass1.get("notes") or []) + [
        "2026-10-25, workstation: R24 pass 3 (scripts/ur5_tip_calibration_r24_pass3.py): left-right twins of the grid in "
        "candidate turns (keep the first that clears per twin), the builder's slides with their pre-sliding steps only, "
        "twists at 5 N at pass 2's face points, final-ref. Payload and frames of R24 pass 1 (mount v6)."]
    Path(out).write_text(json.dumps(plan, indent=1) + "\n")
    n_tw = len({a["twin_of"] for a in acts if "twin_of" in a})
    print(f"{n_tw} twins ({sum('twin_of' in a for a in acts)} candidates), "
          f"{sum(a['stage'] == 'shear' for a in acts)} slides, {sum(a['stage'] == 'twist' for a in acts)} twists -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(EXP / "lab/20261025-plan_sphere_lab_r24_pass3.json"))
    run(ap.parse_args().out)
