#!/usr/bin/env python3
"""Merge population grasp shards and preserve each grasp's exact fitted geometry.

The retention evaluator consumes one row per design, while the coarse screen tests several
straddle/thumb placements.  Writing a separate table for every placement prevents a valid grasp
from accidentally inheriting another placement's fitted depth.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True, help="glob for grasp shard JSON files")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="hardware-filtered manifest defining the expected population")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.shards)
                   if not p.endswith("_manifest.json"))
    if not paths:
        raise SystemExit(f"no shards match {args.shards!r}")
    rows = []
    for path in paths:
        rows.extend(json.loads(path.read_text()))
    rows.sort(key=lambda row: row["design"])

    manifest = json.loads(args.manifest.read_text())
    expected = {row["design"] for row in manifest["designs"]}
    names = [row["design"] for row in rows]
    duplicates = sorted(name for name, count in Counter(names).items() if count != 1)
    unexpected = sorted(set(names) - expected)
    missing = sorted(expected - set(names))
    if duplicates or unexpected or (missing and not args.allow_partial):
        raise SystemExit(json.dumps({"duplicates": duplicates, "unexpected": unexpected,
                                     "missing_count": len(missing), "missing_head": missing[:10]},
                                    indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "grasp_screen.json").write_text(json.dumps(rows, indent=2) + "\n")

    cells: dict[tuple[float, float], list[dict]] = {}
    for row in rows:
        for grasp in row.get("grasps", []):
            if not grasp.get("pose"):
                continue
            key = (float(grasp["straddle_mm"]), float(grasp["thumb_axial_mm"]))
            scores = grasp["scores"]
            cells.setdefault(key, []).append({
                "design": row["design"],
                "graspable": True,
                "straddle_mm": key[0],
                "thumb_axial_mm": key[1],
                "depth_req_mm": grasp.get("depth_req_mm"),
                "depth_fit_mm": scores["depth_fit_mm"],
                "squeeze_mm": grasp["squeeze_mm"],
                "axis_k": 0.15,
                "angle_deg": -80.0,
                "coarse_scores": scores,
                "sampling": row.get("sampling", {}),
                "mounts_mm": row.get("mounts_mm"),
            })

    cell_counts = {}
    for (straddle, thumb), table in sorted(cells.items()):
        suffix = f"s{straddle:g}_t{thumb:g}".replace(".", "p")
        table.sort(key=lambda row: row["design"])
        (args.out_dir / f"retention_table_{suffix}.json").write_text(
            json.dumps(table, indent=2) + "\n")
        (args.out_dir / f"retention_designs_{suffix}.txt").write_text(
            "\n".join(row["design"] for row in table) + "\n")
        cell_counts[suffix] = len(table)

    source_counts = Counter(row.get("sampling", {}).get("source", "known") for row in rows)
    graspable = sum(any(g.get("pose") for g in row.get("grasps", [])) for row in rows)
    summary = {
        "shards": len(paths),
        "screened": len(rows),
        "expected": len(expected),
        "missing": len(missing),
        "graspable_any": graspable,
        "graspable_any_fraction": round(graspable / max(1, len(rows)), 4),
        "source_screened": dict(sorted(source_counts.items())),
        "retention_cells_by_grasp": cell_counts,
        "retention_cells_total": sum(cell_counts.values()),
    }
    (args.out_dir / "grasp_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
