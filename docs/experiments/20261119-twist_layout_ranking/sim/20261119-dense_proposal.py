#!/usr/bin/env python3
"""A denser spread of layouts for the hand over the whole predicted range (hand-4a, 2026-11-19 23:3x; owner, 23:2x: "i
highly suspect that the top 11 plans will not have a good clean correspondence. i think we should shoot for higher
density of evaluation (which is just more time, but we have to accept it) to see if we can first ascertain a rough
correspondence, including simulation-predicted bad morphologies").

Pool: every layout whose mounts fit the gantries' travel boxes, with its best grasp over both models (the ranking's
"candidates") and its least finger gap on the CAD collision spheres along the simulated twist (seed 0, both models;
data/20261119-cad_gap_screen.json). Layouts with a run that ended in HOM's error are left out. The predicted range
0-0.55 is cut into 0.05-wide bins (above 0.55 every fitting layout touches; block 2 has two of them). Each bin is filled
up to --per-bin layouts, counting block 1's layouts, in tiers: first layouts at least --clear mm apart, then at least
10 mm, then any, the touching ones flagged; within a tier, in turn the layout farthest on the lattice from every layout
already chosen. Below 0.35 almost every fitting layout touches (2 of 95 keep 15 mm), so the bad end of the range is
mostly touching layouts. --drops N adds N layouts whose sphere dropped in a simulated run, spread over their scores,
for the end of the block (the lab's series recovers from some drops, not all).

    python3 docs/experiments/20261119-twist_layout_ranking/sim/20261119-dense_proposal.py [--per-bin 3 --clear 15]
Writes data/20261119-trial_proposal_dense.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
DATA = STUDY / "data"
BIN = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bin", type=int, default=3)
    ap.add_argument("--clear", type=float, default=15.0)
    ap.add_argument("--drops", type=int, default=3, help="layouts whose sphere dropped in a simulated run, for the end")
    ap.add_argument("--max-new", type=int, default=22)
    ap.add_argument("--out", default=str(DATA / "20261119-trial_proposal_dense.json"))
    a = ap.parse_args()
    S = json.loads((DATA / "20261119-twist_ranking_summary.json").read_text())
    scr = {(x["call"], x["k"]): x for x in json.loads((DATA / "20261119-cad_gap_screen.json").read_text())["layouts"]}
    runs = json.loads((DATA / "20261119-cad_gap_screen.json").read_text())["runs"]
    errs = {(r["call"], r["k"]) for r in runs if r.get("track") is None}
    lst = json.loads((DATA / "20261119-trial_plans.json").read_text())["plans"]
    have = [e for e in lst if e["role"] in ("ranked", "anchor") and not e.get("prm") and "_cem_" not in e["plan"]
            and not e.get("cad_touch")]
    have_xy = {tuple(e["xy"]) for e in lst}
    fit = sorted((b for b in S["candidates"] if b["fits"]), key=lambda b: -b["score"])
    rank_fit = {(b["call"], b["k"]): i + 1 for i, b in enumerate(fit)}
    pool, droppers = [], []
    for b in fit:
        s = scr.get((b["call"], b["k"]))
        if s is None or tuple(b["xy"]) in have_xy or (b["call"], b["k"]) in errs:
            continue
        rec = {**b, "cad_least_mm": s["least_mm_both_models"], "rank_among_fit": rank_fit[(b["call"], b["k"])]}
        (pool if b["held_all"] else droppers).append(rec)
    chosen = [{"xy": e["xy"], "score": e.get("score") or 0.0} for e in have]
    new = []
    nb = int(round(0.55 / BIN))

    def tier(p):
        g = p["cad_least_mm"]
        return 0 if g >= a.clear else 1 if g >= 10.0 else 2
    for r_ in range(a.per_bin):
        for i in range(nb):
            lo, hi = i * BIN, (i + 1) * BIN
            if len([c for c in chosen if lo <= c["score"] < hi]) > r_ or len(new) >= a.max_new:
                continue
            cand = [p for p in pool if lo <= p["score"] < hi and p not in new]
            if not cand:
                continue
            best_t = min(tier(p) for p in cand)
            cand = [p for p in cand if tier(p) == best_t]
            pts = np.array([c["xy"] for c in chosen], float)
            far = max(cand, key=lambda p: float(np.min(np.linalg.norm(pts - np.asarray(p["xy"], float), axis=1))))
            new.append(far)
            chosen.append({"xy": far["xy"], "score": far["score"]})
    new.sort(key=lambda p: -p["score"])
    drop_pick = []
    if a.drops and droppers:
        ds = sorted(droppers, key=lambda p: -p["score"])
        idx = np.unique(np.round(np.linspace(0, len(ds) - 1, a.drops)).astype(int))
        drop_pick = [ds[i] for i in idx]
    out = {"what": __doc__.split("\n\n")[0].replace("\n", " "), "written": time.strftime("%F %H:%M"),
           "rule": {"bin": BIN, "per_bin": a.per_bin, "clear_mm": a.clear, "drops": a.drops},
           "pick": [{"call": p["call"], "xy": p["xy"], "k": p["k"], "score": p["score"], "held_all": p["held_all"],
                     "im_gap_min_mm": p["im_gap_min_mm"], "cad_least_mm": p["cad_least_mm"],
                     "cad_touch": p["cad_least_mm"] < 2.0, "rank_among_fit": p["rank_among_fit"],
                     "rank_among_deployable": None, "fits": True, "group": "dropper" if p in drop_pick else "dense"}
                    for p in new + drop_pick], "anchors": []}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for i in range(nb):
        lo, hi = i * BIN, (i + 1) * BIN
        o = [c for c in have if lo <= (c.get("score") or 0) < hi]
        n_ = [p for p in new if lo <= p["score"] < hi]
        pl = len([p for p in pool if lo <= p["score"] < hi])
        print(f"{lo:.2f}-{hi:.2f}: block 1 {len(o)}, new {len(n_)} " +
              " ".join(f"({p['xy'][0]:g},{p['xy'][1]:g})g{p['k']} {p['cad_least_mm']:.0f}mm" for p in n_) + f"  [pool {pl}]")
    for p in drop_pick:
        print(f"dropper ({p['xy'][0]:g},{p['xy'][1]:g}) g{p['k']} score {p['score']:.3f} CAD {p['cad_least_mm']:.1f} mm")
    print(f"{len(new)} new layouts and {len(drop_pick)} droppers -> {a.out}")


if __name__ == "__main__":
    main()
