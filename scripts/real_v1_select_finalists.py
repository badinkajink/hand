#!/usr/bin/env python3
"""Choose high-performing plus morphology-diverse hands for independent full-error trials."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirmation", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--core", type=int, default=24)
    ap.add_argument("--diverse", type=int, default=8)
    ap.add_argument("--min-clearance-mm", type=float, default=8.0)
    ap.add_argument("--out-table", type=Path, required=True)
    ap.add_argument("--out-designs", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    args = ap.parse_args()

    confirmation = json.loads(args.confirmation.read_text())
    table = {row["design"]: row for row in json.loads(args.table.read_text())}
    manifest = {row["design"]: row for row in
                json.loads(args.manifest.read_text())["designs"]}
    eligible = [row for row in confirmation
                if row.get("nom_kept") == row.get("n_nom")
                and row.get("nom_cos", 0.0) >= 0.7
                and (row.get("min_finger_clearance_mm") or -1.0)
                >= args.min_clearance_mm]
    eligible.sort(key=lambda row: (row.get("ens_win", 0.0), row.get("ens_kept", 0),
                                   row["nom_cos"], row["min_finger_clearance_mm"],
                                   -row.get("max_proof_slip_mm", 1e9)), reverse=True)
    core = eligible[:args.core]

    # Greedy farthest-point additions in the actual six slide coordinates keep the final test
    # from becoming 32 local variants of one compact tripod. Coordinates are standardized over
    # the eligible pool so X and Y travel contribute comparably.
    slide_indices = np.asarray([0, 1, 3, 4, 6, 7])
    vectors = np.asarray([np.asarray(manifest[row["design"]]["vector_m"])[slide_indices]
                          for row in eligible])
    scale = np.ptp(vectors, axis=0)
    scale[scale == 0.0] = 1.0
    vectors = (vectors - np.mean(vectors, axis=0)) / scale
    name_to_i = {row["design"]: i for i, row in enumerate(eligible)}
    chosen = list(core)
    chosen_names = {row["design"] for row in chosen}
    for _ in range(min(args.diverse, len(eligible) - len(chosen))):
        chosen_indices = [name_to_i[name] for name in chosen_names]
        best_key = None
        best_row = None
        for i, row in enumerate(eligible):
            if row["design"] in chosen_names:
                continue
            distance = float(np.min(np.linalg.norm(vectors[i] - vectors[chosen_indices],
                                                   axis=1)))
            candidate = (distance, row["nom_cos"])
            if best_key is None or candidate > best_key:
                best_key, best_row = candidate, row
        assert best_row is not None
        chosen.append(best_row)
        chosen_names.add(best_row["design"])

    rows = []
    for result in chosen:
        row = dict(table[result["design"]])
        row["confirmation"] = result
        row["finalist_reason"] = "performance" if result in core else "morphology_diversity"
        rows.append(row)
    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    args.out_table.write_text(json.dumps(rows, indent=2) + "\n")
    args.out_designs.write_text("\n".join(row["design"] for row in rows) + "\n")
    summary = {
        "eligible_5_of_5_clear": len(eligible),
        "performance_finalists": len(core),
        "diversity_finalists": len(chosen) - len(core),
        "finalists": len(chosen),
        "min_clearance_mm": args.min_clearance_mm,
        "source_counts": dict(sorted(Counter(
            row.get("sampling", {}).get("source", "known") for row in rows).items())),
        "designs": [row["design"] for row in rows],
    }
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
