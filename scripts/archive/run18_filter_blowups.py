"""Filter the run18 candidates CSV to drop physics blowups.

A "blowup" is a per-task diagnostics row whose object behaviour is clearly
non-physical (object shot to z>1m, velocity > 100 m/s, etc). Their score
swamps the aggregate and the top-K rankings even though they don't
represent real grasps.

Writes:
  <run_dir>/all_candidates_multi_filtered.csv
  <run_dir>/<task>/all_candidates_filtered.csv
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

# Filter thresholds (per-task diagnostics)
BLOWUP_KEYS = {
    "cube_vel_norm": 100.0,        # m/s; real grasps are < a few m/s
    "cube_z_peak":   0.3,          # m; real lifts are < 0.2 m (drill goes to 0.11 + start z)
    "cube_lift":     0.3,          # m; tightened from 1.0 to catch slow whiffs
}


def is_blowup(row: dict[str, str]) -> bool:
    for k, thr in BLOWUP_KEYS.items():
        v = row.get(k)
        if v is None:
            continue
        try:
            if abs(float(v)) > thr:
                return True
        except ValueError:
            continue
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=[
        "cube", "prism", "power_drill",
        "screwdriver_medium_flat", "screwdriver_medium_vertical",
        "screwdriver_medium_90vert", "screwdriver_small_flat",
    ])
    args = ap.parse_args()

    # Build per-task blowup set keyed by candidate_id
    blowup_ids: set[str] = set()
    per_task_blowups: dict[str, int] = {}
    for t in args.tasks:
        p = args.run_dir / t / "all_candidates.csv"
        if not p.exists():
            continue
        with p.open("r") as f:
            rows = list(csv.DictReader(f))
        bids = [r["candidate_id"] for r in rows if is_blowup(r)]
        per_task_blowups[t] = len(bids)
        blowup_ids.update(bids)
        # Write filtered per-task CSV
        kept = [r for r in rows if r["candidate_id"] not in set(bids)]
        out = args.run_dir / t / "all_candidates_filtered.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(kept)

    # Filter cross-task multi CSV — drop any candidate that blew up on ANY task
    cross = args.run_dir / "all_candidates_multi.csv"
    with cross.open("r") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if r["candidate_id"] not in blowup_ids]
    out = args.run_dir / "all_candidates_multi_filtered.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(kept)

    print(f"Blowup candidate IDs (union across tasks): {len(blowup_ids)}")
    for t, n in sorted(per_task_blowups.items()):
        print(f"  {t:30s} {n:4d} blowups")
    print(f"Cross-task CSV: {len(rows)} -> {len(kept)} after filter")
    print(f"Filtered output: {out}")


if __name__ == "__main__":
    main()
