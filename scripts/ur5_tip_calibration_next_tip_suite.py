#!/usr/bin/env python3
"""The night suite for the next four-magnet tip on the UR5 cell (owner, 2026-10-28 13:4x: runs done before 19:00 "so i can
swap fingertips for the other 4 magnet fingertips"): v3 ID 1's plan files as they ran or were checked on 10-28, for a tip
with the same pad (05R4), in the order the lab runs them unattended:

  1 sphere pass 1    superseded/20261027-plan_sphere_lab_pass1_v3id1.json (18 touch-offs): the tip's registration on the sphere
  2 sphere stage 2   superseded/20261027-plan_sphere_lab_stage2_v3id1.json without its speed variants (fast-*, rate-*: press speed
                     from 0.045 to 2 mm/s did not change the GP on cb2t) and without the three actions under the mount's
                     margin on mount v7 (dense-700-t+10, dense-795-t+15 and their slides); after the
                     grid, presses at 0.25 to 5 N on the four FE surface nodes within 2 mm of the apex over U1 that no
                     press of the suite targets (10 of the 14 are targets already)
  3 tall pass 1      superseded/20261028-plan_tall_lab_pass1_19_v3id1.json (19 touch-offs)
  4 tall stage 2     superseded/20261028-plan_tall_lab_stage2_ts1_v3id1.json (102)
  5 rod pass 1       superseded/20261028-plan_rod_lab_pass1_search_ts1_v3id1.json's 21 touch-offs and 5 pivots with the normal
                     approach (3.5 mm above to 1.2 mm past, travel.slow_mm_touch 2.5, as v2 ID 1's rod pass 1): the
                     program resumes a seed folder of scripts/ur5_tip_calibration_seed_registration.py (the rod's T_BI
                     from 102701-rodv3id1, the tip's tool from its sphere pass 1) with --stages preflight,pad,pivot,register
  6 rod stage 2      superseded/20261028-plan_rod_lab_stage2_v7_v3id1.json (94) with superseded/20261028-plan_rod_lab_morning6_v3id1.json's 6
                     presses before its final reference

Every plan gets travel.speed 1.0 and checks.other from lab/cell.json without its own indenter; the tall plans lift
20 mm between actions in place of 60 (owner, 2026-10-28 14:4x: "maybe dont go up so high on these tall indenter moves"). The plans that need the
owner (joint 6 past +45 deg: the tall and sphere morning plans; the top edge) are not part of the night.

    .venv/bin/python scripts/ur5_tip_calibration_next_tip_suite.py [--tip v3id2] [--created ...]
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
SPEED = ("fast-", "rate-")
UNDER_MARGIN_V7 = {"dense-700-t+10", "slide-dense-700-t+10-1N", "slide-dense-795-t+15-1N", "dense-795-t+15"}
BASIS = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-b_v2_surface_basis.npz"
CENTRE_NEW = (799, 775, 777, 765)                       # the free face nodes within 2 mm of the apex, nearest first
SOURCES = [
    ("sphere", "pass1", ["superseded/20261027-plan_sphere_lab_pass1_v3id1.json"]),
    ("sphere", "stage2", ["superseded/20261027-plan_sphere_lab_stage2_v3id1.json"]),
    ("tall", "pass1", ["superseded/20261028-plan_tall_lab_pass1_19_v3id1.json"]),
    ("tall", "stage2", ["superseded/20261028-plan_tall_lab_stage2_ts1_v3id1.json"]),
    ("rod", "pass1", ["superseded/20261028-plan_rod_lab_pass1_search_ts1_v3id1.json"]),
    ("rod", "stage2", ["superseded/20261028-plan_rod_lab_stage2_v7_v3id1.json", "superseded/20261028-plan_rod_lab_morning6_v3id1.json"]),
]


def cell_other(own):
    c = json.loads((LAB / "cell.json").read_text())
    out = []
    for kind, f in c["fixtures"].items():
        k = "tiphold" if kind == "holder" else kind
        if k != own:
            out.append(dict(kind=k, T_BI=json.loads((LAB / f["registration"]).read_text())["T_BI"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tip", default="v3id2")
    ap.add_argument("--created", default=time.strftime("%Y-%m-%d %H:%M"))
    a = ap.parse_args()
    for n, (kind, stage, files) in enumerate(SOURCES, 1):
        plans = [json.loads((LAB / f).read_text()) for f in files]
        p = copy.deepcopy(plans[0])
        acts = list(plans[0]["actions"])
        for q in plans[1:]:                             # more actions go before the first plan's final reference
            k = len(acts) - 1 if "final-ref" in acts[-1]["id"] else len(acts)
            acts = acts[:k] + q["actions"] + acts[k:]
        note = f"{a.created}, workstation: the next tip's night suite (scripts/ur5_tip_calibration_next_tip_suite.py), " \
               f"block {n} of {len(SOURCES)}, from {' and '.join(files)}"
        if kind == "sphere" and stage == "stage2":
            n0 = len(acts)
            speed = [x["id"] for x in acts if x["id"].startswith(SPEED)]
            acts = [x for x in acts if not x["id"].startswith(SPEED) and x["id"] not in UNDER_MARGIN_V7]
            b = np.load(BASIS)
            tpl = next(x for x in acts if x["id"] == "grid-008")
            new = []
            for i in CENTRE_NEW:
                x = {k: copy.deepcopy(v) for k, v in tpl.items() if k not in ("check", "joint6_deg")}
                x["id"] = f"centre-{i:03d}"
                x["target"] = dict(frame="P", p=np.round(b["pts"][i], 4).tolist(), n=np.round(b["normals"][i], 5).tolist(),
                                   index=int(i), region="face")
                new.append(x)
            k = max(j for j, x in enumerate(acts) if x["id"].startswith("grid-")) + 1
            acts = acts[:k] + new + acts[k:]
            note += (f": {n0} to {len(acts)} actions, without the {len(speed)} speed variants and the four under the "
                     f"mount's margin on v7, with {len(new)} presses at 0.25 to 5 N on the free FE nodes within 2 mm "
                     f"of the apex ({', '.join(x['id'] for x in new)}) after the grid")
        if kind == "rod" and stage == "pass1":
            for x in acts:
                if x["kind"] == "touch":
                    x["standoff"], x["beyond_mm"] = 3.5, 1.2
            p.setdefault("travel", {})["slow_mm_touch"] = 2.5
            note += (": the normal approach (3.5 mm above to 1.2 mm past, slow_mm_touch 2.5) in place of the search; "
                     "run it on a seed folder (scripts/ur5_tip_calibration_seed_registration.py) with --stages "
                     "preflight,pad,pivot,register")
        if kind == "tall":
            p.setdefault("travel", {})["lift_mm"] = 20.0
            note += "; travel.lift_mm 20 (the owner, 14:4x)"
        p["actions"] = acts
        p["created"] = a.created
        if isinstance(p.get("suite"), dict):            # the template's label ("v3 ID 1") until 2026-10-29; nothing reads it
            p["suite"]["tip"] = a.tip.replace("id", " ID ")
        p.setdefault("travel", {})["speed"] = 1.0
        p.setdefault("checks", {})["other"] = cell_other(p.get("indenter", kind))
        p.setdefault("notes", [])
        if not isinstance(p["notes"], list):
            p["notes"] = [p["notes"]]
        p["notes"].append(note)
        out = LAB / f"20261028-plan_{kind}_lab_{stage}_night_{a.tip}.json"
        out.write_text(json.dumps(p, indent=1) + "\n")
        print(f"{n} {out.name}: {len(acts)} actions")


if __name__ == "__main__":
    main()
