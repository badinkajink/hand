#!/usr/bin/env python3
"""Pick one (grasp cell, residual clip) operating point per morphology, and say what the clip
recovered.

The 4,096-hand retention screen -- like the 108-hand search and the 128-hand pilot before it --
planned every turn at `budget = 0.5` rad, a number inherited from Policy B's residual ACTION
budget.  A design keeps the tool only inside a contiguous band of clips, so screening at one
value scores each hand at a point that may be nowhere near its own band.  This reads the 2-D
(cell x clip) re-screen and answers the question that matters: which hands pass the retention
gate at SOME clip, and how many of them were invisible at 0.5.

It also enforces the servo-command gate, which the original screen had no way to apply: a
trajectory the driver would refuse is not a candidate, however well it scores.

    python3 scripts/real_v1_select_budget.py \
        --band-dir docs/experiments/20260830-real_v1-budget-rescreen \
        --out-dir  docs/experiments/20260830-real_v1-budget-rescreen/selected
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--band-dir", type=Path, required=True)
    ap.add_argument("--glob", default="band_s*.json")
    ap.add_argument("--table-dir", type=Path, default=None,
                    help="where retention_table_<cell>.json live; defaults to "
                         "<band-dir>/../20260830-real_v1-sobol4096/retention")
    ap.add_argument("--min-cos", type=float, default=0.7)
    ap.add_argument("--min-clearance-mm", type=float, default=5.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    table_dir = a.table_dir or (a.band_dir.parent / "20260830-real_v1-sobol4096" / "retention")
    passing: list[tuple[dict, dict, str]] = []
    seen_designs: set[str] = set()
    evaluated = 0
    for path_str in sorted(glob.glob(str(a.band_dir / a.glob))):
        path = Path(path_str)
        cell = path.stem.removeprefix("band_")
        table = {r["design"]: r for r in
                 json.loads((table_dir / f"retention_table_{cell}.json").read_text())}
        for row in json.loads(path.read_text()):
            evaluated += 1
            seen_designs.add(row["design"])
            clearance = row.get("min_finger_clearance_mm")
            if (not row.get("pose") or not row.get("nom_kept")
                    or row.get("nom_cos", 0.0) < a.min_cos
                    or clearance is None or clearance < a.min_clearance_mm
                    # a plan the driver would refuse is not a candidate, however it scores
                    or row.get("servo_short_deg", 0.0) > 0.0):
                continue
            passing.append((row, table[row["design"]], cell))

    best: dict[str, tuple] = {}
    budgets_that_pass: dict[str, set[float]] = defaultdict(set)
    for row, table_row, cell in passing:
        budgets_that_pass[row["design"]].add(row["budget_rad"])
        key = (row["nom_cos"], -row.get("max_proof_slip_mm", 1e9),
               row["min_finger_clearance_mm"])
        if row["design"] not in best or key > best[row["design"]][0]:
            best[row["design"]] = (key, row, table_row, cell)

    rows, groups = [], defaultdict(list)
    for design, (_, row, table_row, cell) in sorted(best.items()):
        out = dict(table_row)
        out.update(selected_from=cell, axis_k=row["axis_k"], angle_deg=row["angle_deg"],
                   squeeze_mm=row["squeeze_mm"], straddle_mm=row["straddle_mm"],
                   thumb_axial_mm=row["thumb_axial_mm"], budget_rad=row["budget_rad"],
                   single_trial=row,
                   passing_budgets=sorted(budgets_that_pass[design]))
        rows.append(out)
        groups[row["budget_rad"]].append(design)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "selected_table.json").write_text(json.dumps(rows, indent=1) + "\n")
    for budget, designs in sorted(groups.items()):
        (a.out_dir / f"designs_b{round(budget * 100):03d}.txt").write_text(
            ",".join(sorted(designs)) + "\n")

    only_off_half = sorted(d for d in best if 0.5 not in budgets_that_pass[d])
    summary = {
        "cells_evaluated": evaluated,
        "designs_evaluated": len(seen_designs),
        "designs_passing_at_some_clip": len(best),
        "designs_passing_at_0.5": sum(1 for d in best if 0.5 in budgets_that_pass[d]),
        "designs_invisible_at_0.5": len(only_off_half),
        "by_selected_budget": {str(b): len(d) for b, d in sorted(groups.items())},
        "designs_invisible_at_0.5_head": only_off_half[:40],
    }
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    print(json.dumps(summary, indent=1))
    print(f"\nwrote {a.out_dir}/selected_table.json ({len(rows)} designs) and "
          f"{len(groups)} per-clip design lists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
