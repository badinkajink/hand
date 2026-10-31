#!/usr/bin/env python3
"""Why the pose toy reaches more layouts since the finger clearance moved into its program (2026-10-30, 21d3d2877).
Before, the toy solved a pose from one start with no finger clearance in the program, then rejected the solution when two
fingers' skeletons came within 1 mm. Now the plant's spheres must stay 2 mm apart as a constraint of the program. This
script takes layouts that only the new toy reaches, solves them again the old way (the constraint dropped, the
skeleton's clearance checked after solving) and records why each old solve is rejected.

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python scripts/hom_toy_reach_check.py
Rows: docs/experiments/20261029-foundational_pose_review/rows/grid2dff_v4b/reach_check.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fp_sr2_toy as FP  # noqa: E402

PAIRS = (("sobol8k_tripod40_v4b", "sobol8kff_tripod40_v4b"), ("grid2d_tripod40_v4b", "grid2dff_tripod40_v4b"))
N = 40


def rows(name):
    return {json.loads(x)["call"]: json.loads(x) for x in (FP.ROWS / f"{name}.jsonl").read_text().splitlines()
            if x.strip()}


def ok(r):
    return bool((r["fp"].get("tripod") or {}).get("ok"))


def old_clearances(fp, v, M, branch):
    """fp_sr2_toy.clearances before 21d3d2877: the skeleton's finger clearance and the links' clearance to the object."""
    obj, tg, n, _pc, _ = FP.geometry(fp, v, M)
    world, q = {}, {}
    for f in FP.FINGERS:
        q[f] = FP.finger_state(f, tg[f], M[f], branch[f])[0]
        world[f] = {k: p + M[f] for k, p in FP.K.fk(f, q[f]).items() if k != "palmar"}
    ob = float(min(FP.K.object_clearance(world[f], obj) for f in FP.FINGERS))
    ob = min(ob, min(float(FP.rigid_clearance(f, q[f], M[f], obj, n[f])) for f in FP.FINGERS))
    return float(FP.K.finger_clearance(world)), ob, np.inf


def skeleton_clearance(r):
    """The skeleton's finger clearance (mm) of a row's tripod pose."""
    rec = r["fp"]["tripod"]
    M = FP.mounts6(np.asarray(r["d"], float))
    return 1000 * old_clearances("tripod", np.asarray(rec["v"], float), M, rec["branch"])[0]


def main():
    FP.set_sizes(sph_d_mm=40.0)
    rng = np.random.default_rng(1)
    out = {}
    for old_name, new_name in PAIRS:
        old, new = rows(old_name), rows(new_name)
        gained = [c for c in old if not ok(old[c]) and ok(new[c])]
        lost = [c for c in old if ok(old[c]) and not ok(new[c])]
        calls = rng.choice(gained, min(N, len(gained)), replace=False)
        skel = [skeleton_clearance(new[int(c)]) for c in calls]
        new_fns = FP.ff_clearance_v4b, FP.clearances, FP.FF_CLEAR, FP.clear_ok
        seen = []
        FP.ff_clearance_v4b = lambda qs, M: FP.FF_CAP          # no finger clearance in the program
        FP.clearances, FP.FF_CLEAR = old_clearances, 0.001
        FP.clear_ok = lambda clear, f=new_fns[3]: (seen.append(clear), f(clear))[1]
        why = {"reached": 0, "fingers": 0, "object": 0, "no solution": 0}
        try:
            for c in calls:
                d = np.asarray(new[int(c)]["d"], float)
                seen.clear()
                rec = FP.solve_pose("tripod", d, [FP.grid_start("tripod", d)[0]])
                why["reached" if rec["ok"] else "no solution" if not seen else
                    "fingers" if seen[-1][0] < FP.FF_CLEAR else "object"] += 1
        finally:
            FP.ff_clearance_v4b, FP.clearances, FP.FF_CLEAR, FP.clear_ok = new_fns
        out[new_name] = {"old": old_name, "reached_old": sum(ok(r) for r in old.values()),
                         "reached_new": sum(ok(r) for r in new.values()), "gained": len(gained), "lost": len(lost),
                         "tested": len(calls), "old_path": why,
                         "new_pose_skeleton_mm_median": round(float(np.median(skel)), 2),
                         "new_pose_skeleton_under_1mm": int(sum(s < 1.0 for s in skel))}
        print(new_name, out[new_name], flush=True)
    path = FP.ROWS / "grid2dff_v4b" / "reach_check.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
