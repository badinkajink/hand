#!/usr/bin/env python3
"""Write the (design, seed) manifest that deltaai_seed_array.slurm consumes.

One line per array task, TAB-separated:  TAG  MORPH_RUN  SEED  EXTRA_ARGS

The point of the seeds column: per-design reorient quality is seed-dominated
(per-draw sd 0.3-0.5), so one run per design cannot resolve morphology
differences — the local single-GPU budget forced n=1 and that is precisely what
made the 9-dim landscape unreadable. On the cluster n=4..8 costs the same wall
clock as n=1, so the design comparison is finally a comparison of means.

  # 5 designs x 4 seeds = 20 tasks
  scripts/cluster/make_seed_manifest.py \
      --designs results/phase1/morph_sweep/H06_04 results/phase1/morph_sweep/H06_06 \
      --seeds 4 > manifest.tsv

  # seeds of ONE design, with recipe flags passed through
  scripts/cluster/make_seed_manifest.py --designs results/phase1/perp_thumb_engage/sp25_manual \
      --seeds 8 --extra "--recipe perp_single --lift-delta-z 0.14" > manifest.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", nargs="+", required=True,
                    help="morphology run dirs (paths AS THEY EXIST ON THE CLUSTER)")
    ap.add_argument("--seeds", type=int, default=4, help="seeds per design")
    ap.add_argument("--seed0", type=int, default=0, help="first seed value")
    ap.add_argument("--extra", default="", help="extra trainer flags, applied to every task")
    ap.add_argument("--tag-prefix", default="", help="prepended to each design's tag")
    ap.add_argument("--check-local", action="store_true",
                    help="fail if a design dir is missing locally (only valid when the "
                         "cluster paths mirror this checkout)")
    args = ap.parse_args()

    lines = []
    for d in args.designs:
        p = Path(d)
        if args.check_local and not p.is_dir():
            print(f"!! missing design dir: {d}", file=sys.stderr)
            return 1
        tag = f"{args.tag_prefix}{p.name}" if args.tag_prefix else p.name
        for s in range(args.seed0, args.seed0 + args.seeds):
            lines.append(f"{tag}\t{d}\t{s}\t{args.extra}")

    print("\n".join(lines))
    n = len(lines)
    # ~25M timesteps at the measured local rate is ~0.7 GPU-h; re-check against
    # the steps/s the smoke job actually printed on a GH200.
    print(f"# {n} tasks ~= {n * 0.7:.0f} GPU-hours at the local reference rate",
          file=sys.stderr)
    print(f"# sbatch --array=1-{n}%32 --account=<yours> "
          f"scripts/cluster/deltaai_seed_array.slurm manifest.tsv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
