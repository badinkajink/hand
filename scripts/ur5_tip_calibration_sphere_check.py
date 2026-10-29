#!/usr/bin/env python3
"""Reference presses on the sphere fixture after R29 (owner, 2026-10-27 18:2x, through the lab: "we can build it and try
to run it after r29 finishes"). On the tall indenter, v2 ID 1's forces had a shear part along the finger of 0.08 to
0.13 N per newton, against -0.02 to +0.02 on the sphere fixture (scripts/ur5_tip_calibration_tall_r29.py). Three presses
at the face's centre (index 788, the sphere session's reference ladder 0.5, 1 and 2 N) run twice on the sphere fixture,
each set in its own session folder:

  a   on 192158-v2id1's registration as it stands. With the tool unchanged they reproduce the sphere session's references.
  b   on a hybrid: the sphere's centre (T_BI) and everything else from 192158-v2id1, the mount and pad (T_EM, T_MP) from
      143529-tallv2id1. Its pad pose differs from a's by 2.0 deg and 0.4 to 0.6 mm.

Shear along z_P in both sets means the finger or the mount changed; in b alone, the tall registration's orientation; in
neither, the tall indenter (its flex) or the arm's pose there. The plan is the sphere session's header with the tall
indenter (kind "tall", 143529's T_BI) next to the rod in checks.other and the travel's lift at 30 mm (owner, 10-27).

    python3 scripts/ur5_tip_calibration_sphere_check.py        # writes the plan and the two session folders
    L=docs/experiments/20261022-ur5_tip_calibration/lab
    .venv/bin/python scripts/ur5_tip_calibration_check.py $L/20261027-sphchk-b-hybrid --clearance --clearance-stages grid

On corvus each folder is resumed with the plan, stages preflight and grid (no homing, no touch-offs):
    $PY $P --plan $L/20261027-plan_sphere_lab_check_shear_v2id1.json $FT --resume $L/20261027-sphchk-a-v2id1reg \
        --stages preflight,grid
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
SPH, TALL = LAB / "20261026-192158-v2id1", LAB / "20261027-143529-tallv2id1"
PLAN = LAB / "superseded/20261027-plan_sphere_lab_check_shear_v2id1.json"
FOLDERS = dict(a=LAB / "20261027-sphchk-a-v2id1reg", b=LAB / "20261027-sphchk-b-hybrid")


def main():
    base = json.loads((SPH / "plan_resume_2000.json").read_text())
    ref = next(a for a in base["actions"] if a["id"] == "ref-00")
    tall_reg = json.loads((TALL / "registration.json").read_text())
    plan = {k: copy.deepcopy(v) for k, v in base.items() if k != "actions"}
    plan["created"] = "2026-10-27 18:00"
    plan["travel"]["lift_mm"] = 30.0
    other = [o for o in plan["checks"].get("other", []) if o["kind"] != "tall"]
    other.append(dict(kind="tall", T_BI=tall_reg["T_BI"]))
    plan["checks"]["other"] = other
    plan["notes"] = list(plan.get("notes", [])) + [
        "2026-10-27 18:00, workstation: scripts/ur5_tip_calibration_sphere_check.py. Three presses at index 788 as the sphere "
        "session's ref-00, run in two session folders: 20261027-sphchk-a-v2id1reg (192158's registration) and "
        "20261027-sphchk-b-hybrid (T_EM and T_MP from 143529). Stages preflight and grid only: no homing or touch-offs, so "
        "each folder's registration stays as written. The tall indenter is an obstacle at 143529's T_BI; travel lift 30 mm."]
    plan["actions"] = []
    for k in (1, 2, 3):
        a = copy.deepcopy(ref)
        a["id"] = f"sphchk-ref-{k}"
        a["stage"], a["role"] = "grid", "ref"
        plan["actions"].append(a)
    PLAN.write_text(json.dumps(plan, indent=1) + "\n")
    print("->", PLAN.relative_to(ROOT), len(plan["actions"]), "actions")

    reg = json.loads((SPH / "registration.json").read_text())
    for name, folder in FOLDERS.items():
        r = copy.deepcopy(reg)
        if name == "b":
            r["T_EM"], r["T_MP"] = tall_reg["T_EM"], tall_reg["T_MP"]
            r.setdefault("notes", {})
            if isinstance(r["notes"], dict):
                r["notes"]["hybrid"] = ("2026-10-27 18:00: T_EM and T_MP from 20261027-143529-tallv2id1's registration; "
                                        "T_BI and the rest from 20261026-192158-v2id1's")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "registration.json").write_text(json.dumps(r, indent=1) + "\n")
        (folder / "touches.json").write_text((SPH / "touches.json").read_text())
        (folder / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")   # a resume does not write it; the analysis reads it
        (folder / "events.jsonl").touch()                                     # no action done yet
        print("->", folder.relative_to(ROOT))


if __name__ == "__main__":
    main()
