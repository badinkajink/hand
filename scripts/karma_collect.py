"""Collect a KaRMA sweep's per-run outputs into one analysis table.

The sweep's JSONL is a progress log; the run directories under ``--work`` are the source
of truth, one ``current.yaml`` each. This reads those, joins them to the labelled design
sample, and writes a single JSON the analysis and the page builder both read.

Beyond the three published scores it records what the seed pinch actually is, which is
what a 3-finger hand needs to know about a 2-finger metric: how deep below the palm the
metric chose to pinch, and how far the pinch sits from the hand's own grasp depth.

    python scripts/karma_collect.py --sample .../sample.json --work <workdir> \\
        --out .../karma_table.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


def collect_run(rdir: Path) -> dict | None:
    res = rdir / "current.yaml"
    if not res.exists():
        return None
    y = yaml.safe_load(res.read_text()) or {}
    sm = y.get("summary") or {}
    ss = y.get("seed_sensitivity") or {}
    sd = y.get("seed") or {}
    if sm.get("translational_score") is None:
        return None
    centre = sd.get("centre_world") or [0.0, 0.0, 0.0]
    return {
        "karma_t": float(sm["translational_score"]),
        "karma_r": float(sm["global_rotational_score"]),
        "karma_s": (float(ss["karma_s"]) if ss.get("karma_s") is not None else None),
        "n_voxels": int(sm["n_voxels_reached"]),
        "n_states": int(sm["n_states_reached"]),
        "l_ref_mm": round(float(sm["l_ref_m"]) * 1e3, 3),
        "volume_mm3": round(float(sm["translational_volume_m3"]) * 1e9, 2),
        "n_seeds": ss.get("n_seeds_evaluated"),
        "best_voxels": ss.get("best"),
        "median_voxels": ss.get("median"),
        # Where the metric chose to pinch, in the palm frame. The task's own grasp sits
        # 50-66 mm below the mounting plane; a much shallower pinch is a different grasp.
        "seed_depth_mm": round(-float(centre[2]) * 1e3, 2),
        "seed_radius_mm": round(math.hypot(centre[0], centre[1]) * 1e3, 2),
        "seed_contact": "+".join(sd.get("contact_pair") or []),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sample = {r["design"]: r for r in json.loads(args.sample.read_text())}
    rows: list[dict] = []
    missing = 0
    for rdir in sorted(args.work.iterdir()):
        if not rdir.is_dir():
            continue
        name = rdir.name
        for variant in ("scaled", "absolute"):
            if not name.endswith("_" + variant):
                continue
            stem = name[: -len(variant) - 1]
            for pair in ("thumb_index", "thumb_middle", "index_middle"):
                if not stem.endswith("_" + pair):
                    continue
                design = stem[: -len(pair) - 1]
                if design not in sample:
                    continue
                run = collect_run(rdir)
                if run is None:
                    missing += 1
                    continue
                rows.append({**sample[design], "pair": pair.replace("_", "-"),
                             "variant": variant, **run})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))

    by = {}
    for r in rows:
        by.setdefault((r["variant"], r["pair"]), []).append(r)
    print(f"{len(rows)} runs collected ({missing} incomplete) -> {args.out}")
    for k in sorted(by):
        v = by[k]
        vox = sorted(r["n_voxels"] for r in v)
        print(f"  {k[0]:9s} {k[1]:13s} n={len(v):4d}  voxels min {vox[0]:4d} "
              f"med {vox[len(vox) // 2]:4d} max {vox[-1]:4d}  "
              f"retained {sum(r['retained'] for r in v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
