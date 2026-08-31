#!/usr/bin/env python3
"""Select one strict retention operating point per morphology for confirmation."""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retention-dir", type=Path, required=True)
    ap.add_argument("--strict", default="strict_s*.json")
    ap.add_argument("--min-cos", type=float, default=0.7)
    ap.add_argument("--min-clearance-mm", type=float, default=5.0)
    ap.add_argument("--out-table", type=Path, required=True)
    ap.add_argument("--out-designs", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    args = ap.parse_args()

    passing = []
    evaluated = 0
    for result_path_str in sorted(glob.glob(str(args.retention_dir / args.strict))):
        result_path = Path(result_path_str)
        suffix = result_path.stem.removeprefix("strict_")
        table_path = args.retention_dir / f"retention_table_{suffix}.json"
        table = {row["design"]: row for row in json.loads(table_path.read_text())}
        results = json.loads(result_path.read_text())
        evaluated += len(results)
        for result in results:
            clearance = result.get("min_finger_clearance_mm")
            if (not result.get("nom_kept") or result.get("nom_cos", 0.0) < args.min_cos
                    or clearance is None or clearance < args.min_clearance_mm):
                continue
            if result["design"] not in table:
                raise SystemExit(f"{result['design']} missing from {table_path}")
            passing.append((result, table[result["design"]], suffix))

    # Prefer the best-aligned passing grasp, then less slip and more clearance. Confirmation
    # repeats will determine whether this optimistic single trial is real.
    best = {}
    for result, table_row, suffix in passing:
        key = (result["nom_cos"], -result.get("max_proof_slip_mm", 1e9),
               result["min_finger_clearance_mm"])
        design = result["design"]
        if design not in best or key > best[design][0]:
            best[design] = (key, result, table_row, suffix)

    rows = []
    selected_cells = []
    for design, (_, result, table_row, suffix) in sorted(best.items()):
        row = dict(table_row)
        row["selected_from"] = suffix
        row["axis_k"] = result["axis_k"]
        row["angle_deg"] = result["angle_deg"]
        row["squeeze_mm"] = result["squeeze_mm"]
        row["single_trial"] = result
        rows.append(row)
        selected_cells.append(result)

    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    args.out_table.write_text(json.dumps(rows, indent=2) + "\n")
    args.out_designs.write_text("\n".join(row["design"] for row in rows) + "\n")
    source_counts = Counter(row.get("sampling", {}).get("source", "known") for row in rows)
    summary = {
        "cells_evaluated": evaluated,
        "passing_cells": len(passing),
        "unique_designs_selected": len(rows),
        "min_cos": args.min_cos,
        "min_clearance_mm": args.min_clearance_mm,
        "selected_source_counts": dict(sorted(source_counts.items())),
        "selected_cells": selected_cells,
    }
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "selected_cells"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
