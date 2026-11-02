#!/usr/bin/env python3
"""The 10-30 searches' best layouts on three plants, from the CPU checks of hom_top_cpu_check.py (2026-11-02).

The layout study of 10-30/31 ran on the work plant (kp 4 N m/rad, 1 N m) with pads at 10 MPa. Its best layouts were then
run on the CPU plant at 2.5 MPa on the work plant (rows tagged _w25) and on the servo-limited plant (kp 0.72 N m/rad,
0.18 N m; tagged _sv25), from the same best grasp and seed 0 (logs/20261102-hom_servo_plant/cpu_queue.sh, lanes W and S).
For each task and size this prints, per plant, the CPU score of the search's best layout and the median over its checked
layouts, with all contacts and with the GPU's collision filter, and the share of layouts whose score stayed within 0.05
of the 10 MPa run (the trends page's test: "the trends hold if the best layouts' CPU scores move by less than 0.05").

    python3 scripts/hom_plant_compare.py            # table on stdout; rows/cpu_best/20261102-plant_compare.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / "docs/experiments/20261029-foundational_pose_review/rows/cpu_best"
DEFAULT = {"twist": 40.0, "roll": 40.0, "pinpush": 25.0, "press": 25.0}
SIZES = {"twist": (25, 30, 35, 40, 45, 50, 56, 63, 71, 80), "roll": (25, 30, 35, 40, 45, 50, 56, 63, 71, 80),
         "pinpush": (8, 10, 13, 16, 20, 25, 32, 40, 50), "press": (8, 10, 13, 16, 20, 25, 32, 40, 50)}
PLANTS = (("work, 10 MPa", ""), ("work, 2.5 MPa", "_w25_fromwork"), ("servo, 2.5 MPa", "_sv25_fromwork"))
N = 10                                   # the checked layouts per search (the 10-30 grid checks ran 40; their best 10 here)


def rows(task, d, suffix):
    ea = "" if DEFAULT[task] == d else "_ea"
    p = CPU / f"top_{task}_d{d:g}_ff{ea}{suffix}.jsonl"
    out = {}
    for x in (p.read_text().splitlines() if p.exists() else []):
        if x.strip():
            r = json.loads(x)
            out[(int(r["call"]), bool(r["gpu_filter"]))] = r
    return out


def main():
    table = {}
    print(f"{'task':8s} {'d':>4s}  " + "  ".join(f"{n:>26s}" for n, _ in PLANTS))
    for task, sizes in SIZES.items():
        for d in map(float, sizes):
            base = rows(task, d, "")
            calls = sorted({c for c, _ in base}, key=lambda c: -base[(c, False)]["gpu_best"] if (c, False) in base else 0)
            calls = calls[:N]
            if not calls:
                continue
            best = calls[0]
            entry = {"best_call": best, "best_xy": [base[(best, False)]["x_sep"], base[(best, False)]["y_sep"]],
                     "gpu_best": base[(best, False)]["gpu_best"], "plants": {}}
            cells = []
            for name, suffix in PLANTS:
                rr = rows(task, d, suffix)
                per = {}
                for gf in (False, True):
                    sc = {c: rr[(c, gf)]["score"] for c in calls if (c, gf) in rr}
                    ref = {c: base[(c, gf)]["score"] for c in calls if (c, gf) in base}
                    same = [abs(sc[c] - ref[c]) < 0.05 for c in sc if c in ref]
                    per["filter" if gf else "all"] = {
                        "n": len(sc), "best": sc.get(best), "median": float(np.median(list(sc.values()))) if sc else None,
                        "held": int(sum(bool(rr[(c, gf)].get("held")) for c in sc)),
                        "within_0.05_of_10MPa": (int(sum(same)), len(same))}
                entry["plants"][name] = per
                a = per["all"]
                cells.append("–" if not a["n"] else f"best {a['best'] if a['best'] is not None else float('nan'):.2f} "
                                                    f"med {a['median']:.2f} n{a['n']:>2d}")
            table[f"{task}_d{d:g}"] = entry
            print(f"{task:8s} {d:4g}  " + "  ".join(f"{c:>26s}" for c in cells), flush=True)
    out = CPU / "20261102-plant_compare.json"
    out.write_text(json.dumps({"n_layouts": N, "plants": [p for p, _ in PLANTS], "searches": table}, indent=1))
    print("wrote", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
