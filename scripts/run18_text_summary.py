"""Plain-text summary of a run18 sweep.

Reads `<run_dir>/analysis_filtered/summary.json` plus `<run_dir>/<task>/all_candidates_filtered.csv`
and emits a readable text report on stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def _to_float(rows: list[dict], k: str) -> np.ndarray:
    return np.asarray([float(r.get(k, "0")) for r in rows], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()

    rd = args.run_dir
    summary_p = rd / "analysis_filtered" / "summary.json"
    cross_p = rd / "all_candidates_multi_filtered.csv"

    if not summary_p.exists() or not cross_p.exists():
        print(f"missing {summary_p} or {cross_p}")
        return

    summary = json.loads(summary_p.read_text())
    cross_rows = _read_csv(cross_p)
    score_mean = _to_float(cross_rows, "score_mean")

    print("=" * 72)
    print(f"run18 SUMMARY — {rd.name}")
    print("=" * 72)
    print()
    print(f"Candidates (after blowup filter): {summary['n_candidates']}")
    print(f"score_mean    min={summary['score_mean_distribution']['min']:+.2f}  "
          f"median={summary['score_mean_distribution']['median']:+.2f}  "
          f"max={summary['score_mean_distribution']['max']:+.2f}")
    print()

    print("Per-task TOP scores + real-contact diagnostics:")
    print(f"{'task':30s} {'best':>7s} {'median':>7s} {'real_grasps':>12s} {'top_persist':>11s} {'top_lift':>9s}")
    print("-" * 84)
    for task, best_row in summary['best_morphology_per_task'].items():
        per_task_csv = rd / task / "all_candidates_filtered.csv"
        rows = _read_csv(per_task_csv)
        if not rows:
            continue
        scores = _to_float(rows, "score")
        persist = _to_float(rows, "min_finger_contact_persistence")
        lift = _to_float(rows, "cube_lift")
        # "real grasps" = candidates where ALL three fingers had >=0.5 persistence
        real_mask = persist > 0.5
        best_i = int(np.argmax(scores))
        print(f"{task:30s} {scores[best_i]:+7.2f} {np.median(scores):+7.2f} "
              f"{real_mask.sum():>5d}/{len(rows):<5d}  {persist[best_i]:>11.3f} {lift[best_i]*1000:>7.1f}mm")
    print()

    # Cross-set top-3
    print("Cross-set TOP-3 morphologies (by score_mean):")
    top3_idx = np.argsort(-score_mean)[:3]
    for rank, idx in enumerate(top3_idx):
        r = cross_rows[idx]
        print(f"  #{rank+1}: candidate_id={r['candidate_id']}  score_mean={float(r['score_mean']):+.2f}")
        for k, v in r.items():
            if k.startswith("score_") and k != "score_mean":
                print(f"      {k[6:]:30s} {float(v):+.2f}")
    print()

    print("Specialisation (variance of score_task - score_mean — higher = task-specific):")
    for t, v in summary["specialization_score_per_task"].items():
        print(f"  {t:30s} {v:>10.2f}")
    print()

    # Artifact pointers
    print("Artifacts:")
    print(f"  CSV (filtered):       {cross_p.relative_to(rd.parent)}")
    print(f"  Analysis:             {rd.name}/analysis_filtered/")
    print(f"  Labeled TSNE:         {rd.name}/analysis_filtered/labeled_tsne.png")
    print(f"  Spearman matrix:      {rd.name}/analysis_filtered/similarity/spearman_matrix.png")
    print(f"  Videos (per task):    {rd.name}/videos_filtered/per_task/<task>/top_NN_*.mp4")
    print(f"  Videos (cross-set):   {rd.name}/videos_filtered/cross_set/top_NN_on_<task>/*.mp4")
    print()


if __name__ == "__main__":
    main()
