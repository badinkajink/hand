#!/usr/bin/env python3
"""Per-cell turn scores of the hand-object scale study under each controller and protocol, for the comparison page.

A cell's score is the landscape's rule (hand_object_scale_landscape.py): the median held turn over the placement seeds
at the candidate grasp with the largest median. Prints one table per plant and writes cells.json beside the rows.

  .venv/bin/python scripts/hand_object_scale_controllers.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/experiments/20261013-scale_turn_controllers"
SOURCES = {   # (plant, controller/protocol) -> rows
    ("1 N m", "HOM, post"): ROOT / "docs/experiments/20261008-hand_object_scale/turn.jsonl",
    ("1 N m", "HOM, no post"): OUT / "hom_postoff/turn.jsonl",
    ("1 N m", "MPC"): OUT / "mpc/turn.jsonl",
    ("servo", "HOM, post"): ROOT / "docs/experiments/20261011-bench_plan/servo_plant/turn.jsonl",
    ("servo", "HOM, no post"): OUT / "hom_postoff_servo/turn.jsonl",
    ("servo", "MPC"): OUT / "mpc_servo/turn.jsonl",
}


def scores(path: Path) -> dict:
    by = defaultdict(lambda: defaultdict(list))
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("status") == "ok" and r.get("model", "pads") == "pads":
            by[(r["shape"], r["d_mm"], r["t"])][r["cand"]].append(r["held_turn_deg"])
    out = {}
    for k, cc in by.items():
        ci = max(sorted(cc), key=lambda c: float(np.median(cc[c])))
        out[k] = {"cand": ci, "score": float(np.median(cc[ci])), "seeds": cc[ci],
                  "all": {int(c): float(np.median(v)) for c, v in cc.items()}}
    return out


def main() -> int:
    S = {k: scores(p) for k, p in SOURCES.items()}
    cells = []
    for plant in ("1 N m", "servo"):
        keys = sorted(set().union(*[set(S[(plant, c)]) for c in ("HOM, post", "HOM, no post", "MPC")]))
        if plant == "servo":
            keys = [k for k in keys if k[0] == "cylinder" and k[1] in (10.0, 16.0, 25.0)]
        print(f"== {plant} plant: {len(keys)} cells")
        for k in keys:
            row = {"plant": plant, "shape": k[0], "d_mm": k[1], "t": k[2]}
            for c in ("HOM, post", "HOM, no post", "MPC"):
                v = S[(plant, c)].get(k)
                row[c] = None if v is None else {"cand": v["cand"], "score": round(v["score"], 2), "seeds": v["seeds"]}
            cells.append(row)
        for shape in ("cylinder", "sphere"):
            ks = [k for k in keys if k[0] == shape]
            if not ks:
                continue
            ds = sorted({k[1] for k in ks})
            ts = sorted({k[2] for k in ks})
            for c in ("HOM, post", "HOM, no post", "MPC"):
                if not S[(plant, c)]:
                    continue
                print(f"  {shape}, {c}: rows t, columns d {ds}")
                for t in ts:
                    vals = [S[(plant, c)].get((shape, d, t), {}).get("score") for d in ds]
                    print(f"    {t:+.3f} " + " ".join("   ." if v is None else f"{v:4.0f}" for v in vals))
    (OUT / "cells.json").write_text(json.dumps(cells, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
