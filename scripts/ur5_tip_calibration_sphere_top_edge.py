#!/usr/bin/env python3
"""A short top-edge pass on the low sphere (owner, 2026-10-28 18:0x, through the lab: "i think you should do the top edge
calbiration on the lower sphere, we are close to joint limits on the tall indenter"; "ok try again on the lower sphere?").

Targets are the FE surface points of the tip's end (and, with --regions, edges and flanks) that the sphere reaches on a
session's registration in scripts/tall_indenter.py reach --kind sphere (--registration, --base-plan, --j6, --regions):
at a turn about the vertical (the plan's "twist") with joint 5 and joint 6 inside their windows and the cell's
clearances at the rest and 1 mm past the surface. Points within --gap mm of a press target of --cover's plans are left
out, and the rest are thinned to --spacing mm. Each press takes the turn nearest 10 deg of its reachable ones and the
tip's levels, 0.25 to 2 N (edges to 3 N). The plan opens with a reference press at the face centre (index 788, as every
sphere plan), keeps the presses in --order, and closes with the same reference. Ids start with "sph-top-", which no
earlier plan used. The header (geometry, checks, travel) is --base-plan's, with travel.lift_mm --lift and travel.speed 1.0.

    .venv/bin/python scripts/ur5_tip_calibration_sphere_top_edge.py REACH.json --base-plan lab/<v3 ID 1 sphere plan> \\
        [--max 10] [--order id1,id2,...] --out lab/<date>-plan_sphere_lab_top_edge_v3id1.json
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
BASIS = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-b_v2_surface_basis.npz"
LEVELS = dict(tip=[0.25, 0.5, 1.0, 2.0], flank=[0.25, 0.5, 1.0, 2.0], edge=[0.25, 0.5, 1.0, 2.0, 3.0],
              face=[0.25, 0.5, 1.0, 2.0, 3.5, 5.0])
REF_INDEX = 788


def press(i, region, twist, b, role=None):
    lv = LEVELS[region]
    a = dict(id=f"sph-top-{region}-{i:03d}", stage="grid", kind="press",
             target=dict(frame="P", p=np.round(b["pts"][i], 4).tolist(), n=np.round(b["normals"][i], 5).tolist(),
                         index=int(i), region=region),
             twist=float(twist), standoff=1.5, levels=lv, stop_N=lv[-1], max_depth_mm=3.0 if region == "edge" else 2.5,
             hold_s=0.6)
    if role:
        a["role"] = role
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reach")
    ap.add_argument("--base-plan", required=True)
    ap.add_argument("--cover", default="", help="plans whose press targets count as covered (comma-separated)")
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--regions", default="tip")
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--order", default="", help="press ids in the order to run (after the reference)")
    ap.add_argument("--paths", default=None, help="a path check of an earlier build (holder_paths --out): its presses "
                    "with the elbow under --elbow-min deg go, and the rest run nearest a limit first (the least of the elbow "
                    "over --elbow-min and joint 5's and joint 6's margins)")
    ap.add_argument("--elbow-min", type=float, default=20.0)
    ap.add_argument("--lift", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = json.loads(Path(a.reach).read_text())
    b = np.load(BASIS)
    covered = []
    for f in filter(None, a.cover.split(",")):
        for x in json.loads(Path(f).read_text())["actions"]:
            if x.get("kind") == "press" and "target" in x:
                covered.append(x["target"]["p"])
    covered = np.array(covered, float).reshape(-1, 3)
    regions = a.regions.split(",")
    pts = [r for r in R["points"] if r["region"] in regions and r["turns_ok"]]
    keep = []
    for r in sorted(pts, key=lambda r: (regions.index(r["region"]), -len(r["turns_ok"]))):
        p = b["pts"][r["i"]]
        if len(covered) and np.min(np.linalg.norm(covered - p, axis=1)) < a.gap:
            continue
        if any(np.linalg.norm(b["pts"][k["i"]] - p) < a.spacing for k in keep):
            continue
        keep.append(r)
    acts = {}
    for r in keep:
        tw = min(r["turns_ok"], key=lambda v: abs(v - 10))
        x = press(r["i"], r["region"], tw, b)
        acts[x["id"]] = x
    ids = [i for i in a.order.split(",") if i in acts] if a.order else []
    if a.paths:
        rows = {r["id"]: r for r in json.loads(Path(a.paths).read_text())["rows"]}
        low = [i for i in acts if i in rows and rows[i]["q3"] < a.elbow_min]
        for i in low:
            acts.pop(i)
        near = lambda i: min(rows[i]["q3"] - a.elbow_min, rows[i]["j5_margin"], rows[i]["j6_margin"])  # noqa: E731
        ids = sorted([i for i in acts if i in rows], key=near)
        print(f"--paths: {len(low)} presses with the elbow under {a.elbow_min:g} deg left out ({', '.join(low)})")
    ids += [i for i in acts if i not in ids]
    ids = ids[: a.max]
    base = json.loads(Path(a.base_plan).read_text())
    p = {k: copy.deepcopy(v) for k, v in base.items() if k not in ("actions", "suite")}
    p["created"] = time.strftime("%Y-%m-%d %H:%M")
    p.setdefault("travel", {})["lift_mm"] = a.lift
    p["travel"]["speed"] = 1.0
    ref = press(REF_INDEX, "face", 0.0, b, role="ref")
    ref.update(id="sph-top-ref-a", levels=[0.5, 1.0, 2.0], stop_N=2.0, max_depth_mm=2.5)
    fin = dict(ref, id="sph-top-final-ref")
    p["actions"] = [ref] + [acts[i] for i in ids] + [fin]
    p.setdefault("notes", [])
    if not isinstance(p["notes"], list):
        p["notes"] = [p["notes"]]
    p["notes"].append(f"{p['created']}, workstation: the sphere's top-edge pass (scripts/ur5_tip_calibration_sphere_top_edge.py) "
                      f"from {a.reach}: {len(pts)} reachable points in {a.regions}, {len(keep)} after the {a.gap} mm cover "
                      f"and {a.spacing} mm thinning, {len(ids)} pressed; header from {a.base_plan}, lift {a.lift:g} mm")
    Path(a.out).write_text(json.dumps(p, indent=1) + "\n")
    print(f"{len(pts)} reachable, {len(keep)} kept, {len(ids)} in the plan -> {a.out}")
    for x in p["actions"]:
        print(f"  {x['id']:24s} twist {x['twist']:5.1f}  levels {x['levels']}")


if __name__ == "__main__":
    main()
