#!/usr/bin/env python3
"""Each plant's prediction for the proposed layouts' default-force plans (hand-4a, 2026-11-19 21:2x), so that each plant
can be scored against the trials on the hand: per proposed layout and its grasp k (data/20261119-trial_proposal.json),
the mean tracking over seeds 0-2 under point contact and the pads (stages A-B) and under stage D's four plants (stage D
where it ran the grasp, stage P of sim/20261119-twist_ranking_search.py for the rest).

    python3 docs/experiments/20261119-twist_layout_ranking/sim/20261119-plant_predictions.py
Writes data/20261119-plant_predictions.json.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
LOGS = ROOT / "logs/20261119-twist_layout_ranking"
PLANTS = ("point", "pads", "pads_stiff", "point_mu0.6", "pads_mu0.6", "point_5v")


def score(r):
    return float(r.get("track") or 0.0) if "error" not in r else 0.0


def main():
    prop = json.loads((STUDY / "data/20261119-trial_proposal.json").read_text())
    want = {(int(p["call"]), int(p["k"])) for p in prop["pick"] + prop["anchors"]}
    sc = defaultdict(dict)                     # (plant, call, k) -> {seed: score}
    for st in ("A", "B", "D", "P"):
        p = LOGS / f"rows_{st}.jsonl"
        for x in p.read_text().splitlines() if p.exists() else []:
            if not x.strip():
                continue
            r = json.loads(x)
            c, k, s = int(r["call"]), int(r["k"]), int(r["seed"])
            if (c, k) in want and s in (0, 1, 2) and r["model"] in PLANTS:
                sc[(r["model"], c, k)][s] = score(r)
    out = []
    for p in prop["pick"] + prop["anchors"]:
        c, k = int(p["call"]), int(p["k"])
        pred = {}
        for m in PLANTS:
            v = sc.get((m, c, k), {})
            pred[m] = round(sum(v.values()) / 3, 4) if len(v) == 3 else None
        out.append({"call": c, "xy": p["xy"], "k": k, "rank_among_deployable": p.get("rank_among_deployable"),
                    "plan": f"20261119-rank_x{p['xy'][0]:g}_y{p['xy'][1]:g}_twist_gr_g{k}_plan.json",
                    "predicted": pred})
    path = STUDY / "data/20261119-plant_predictions.json"
    path.write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "), "written": time.strftime("%F %H:%M"),
                                "plants": {"point": "point contact, 7.5 V, friction 1.0", "pads": "pads, 7.5 V, friction 1.0",
                                           "pads_stiff": "pads at impratio 100", "point_mu0.6": "point contact, friction 0.6",
                                           "pads_mu0.6": "pads, friction 0.6", "point_5v": "point contact, servos at 5 V"},
                                "layouts": out}, indent=1) + "\n")
    for o in out:
        print(o["plan"], {m: v for m, v in o["predicted"].items()})
    print("wrote", path)


if __name__ == "__main__":
    main()
