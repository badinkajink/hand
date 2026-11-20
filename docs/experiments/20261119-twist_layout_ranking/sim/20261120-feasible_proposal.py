#!/usr/bin/env python3
"""More layouts for the hand from the predicted-good part of the feasible set (hand-4a, 2026-11-20 08:2x; owner, 08:0x:
"We don't have to run the full infeasible set, we should get more coverage among the probably good and feasible set of
hands first").

Pool: the layouts of data/20261120-feasible_ranking.json with a feasible grasp, taken at that grasp (each layout's best
grasp whose simulated twist kept the fingers apart, sim/20261120-feasible_grasps.py). A layout enters when its least
finger gap on the CAD collision spheres at seed 0 is at least --clear mm under both models, its sphere stayed in the
grasp at seeds 0-2 under both models, its predicted tracking (mean of seeds 0-2 under both models, the ranking's default
forces) is at least --min, and it is not yet in the plan list (data/20261119-trial_plans.json). Up to --n layouts are
taken, each in turn the one farthest on the lattice from every layout of the plan list and every layout already taken,
so the new layouts fill the gaps of the morning list.

    python3 docs/experiments/20261119-twist_layout_ranking/sim/20261120-feasible_proposal.py [--n 20 --min 0.42]
Writes data/20261120-trial_proposal_feasible.json and prints the vet command.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
DATA = STUDY / "data"
ROWS = Path(__file__).resolve().parents[4] / "logs/20261119-twist_layout_ranking/rows_FG.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min", type=float, default=0.42, help="least predicted tracking")
    ap.add_argument("--clear", type=float, default=15.0, help="least finger gap (mm) under both models")
    ap.add_argument("--out", default=str(DATA / "20261120-trial_proposal_feasible.json"))
    a = ap.parse_args()
    F = json.loads((DATA / "20261120-feasible_ranking.json").read_text())["layouts"]
    lst = json.loads((DATA / "20261119-trial_plans.json").read_text())["plans"]
    have_xy = {tuple(e["xy"]) for e in lst}
    rr = [json.loads(x) for x in ROWS.read_text().splitlines() if x.strip()]
    least, held = {}, {}
    for r in rr:
        key = (r["call"], r["k"])
        if r["seed"] == 0:
            least.setdefault(key, {})[r["model"]] = r["least_mm"]
        held.setdefault(key, []).append(bool(r.get("held")) and not r.get("error"))
    pool = []
    for f in F:
        key = (f["call"], f["k"])
        if f["k"] is None or tuple(f["xy"]) in have_xy or f["seeds012"] is None:
            continue
        gap = min(least[key].values())
        if gap >= a.clear and len(held[key]) == 6 and all(held[key]) and f["seeds012"] >= a.min:
            pool.append({**f, "cad_least_mm": round(gap, 1)})
    taken = [np.array(e["xy"], float) for e in lst]
    new = []
    while pool and len(new) < a.n:
        d = [min(np.linalg.norm(np.array(p["xy"]) - t) for t in taken) for p in pool]
        i = int(np.argmax(d))
        p = pool.pop(i)
        p["spread_mm"] = round(float(d[i]), 1)
        new.append(p)
        taken.append(np.array(p["xy"], float))
    out = {"what": __doc__.split("\n\n")[0].replace("\n", " "), "written": time.strftime("%F %H:%M"),
           "options": vars(a), "pool_left": len(pool), "layouts": new}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    v = np.array([p["seeds012"] for p in new])
    print(f"{len(new)} layouts, predicted {v.min():.3f}-{v.max():.3f} (median {np.median(v):.3f}); "
          f"{len(pool)} more in the pool -> {a.out}")
    for p in sorted(new, key=lambda p: -p["seeds012"]):
        print(f"  ({p['xy'][0]:g}, {p['xy'][1]:g}) g{p['k']}: predicted {p['seeds012']:.3f}, gap {p['cad_least_mm']} mm, "
              f"default grasp g{p['default_k']}, {p['grasps_feasible']}/16 grasps feasible, {p['spread_mm']} mm from the list")
    print("vet:", " ".join(f"{p['xy'][0]:g},{p['xy'][1]:g}:{p['k']}" for p in new))


if __name__ == "__main__":
    main()
