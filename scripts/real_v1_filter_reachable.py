#!/usr/bin/env python3
"""Keep only the sampled hands whose three mounts fit the MEASURED gantry travel.

    python3 scripts/real_v1_filter_reachable.py \
        --manifest docs/experiments/<pop>/grasp_screen_manifest.json \
        --out      docs/experiments/<pop>/hardware_manifest.json

A sampler draws mount positions from the design box; the rails do not reach all of it. This is the
first and cheapest filter in the funnel -- it rejected 784 of 4,102 in the 2026-08-30 population --
and it is the one step of that pipeline that had never been committed, so a later population could
not be screened without rewriting it.

`mount_violations` is the same call `HandPlan.validate` makes, so a design that survives here
cannot fail the deployment gate for its mounts (it can still fail on joint range or clearance).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/morphohand/driver/manta/host"))

from manta_hand.plan import mount_violations  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from morphohand.sampling.morphology import REAL_V1_MOUNTS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    manifest = json.loads(a.manifest.read_text())
    kept, rejected = [], []
    for row in manifest["designs"]:
        # From the VECTOR, not from geometry.mounts_mm: the manifest rounds mounts to 0.1 mm and
        # seven of the 4,102 designs in the 2026-08-30 population sit within 0.05 mm of a rail
        # limit, where that rounding flips the verdict.
        vec = row["vector_m"]
        mounts = {"thumb": (REAL_V1_MOUNTS["thumb"][0] + vec[0],
                            REAL_V1_MOUNTS["thumb"][1] + vec[1]),
                  "index": (REAL_V1_MOUNTS["index"][0] + vec[3],
                            REAL_V1_MOUNTS["index"][1] + vec[4]),
                  "middle": (REAL_V1_MOUNTS["middle"][0] + vec[6],
                             REAL_V1_MOUNTS["middle"][1] + vec[7])}
        bad = []
        for finger, (xm, ym) in mounts.items():
            x, y = xm * 1000.0, ym * 1000.0
            bad += [{"finger": v.finger, "axis": v.axis, "value_mm": v.value,
                     "short_mm": v.short} for v in mount_violations(finger, x, y)]
        (kept if not bad else rejected).append(row if not bad else
                                               {"design": row["design"],
                                                "source": row.get("source"),
                                                "violations": bad})

    out = dict(manifest)
    out["designs"] = kept
    out["raw_candidate_count"] = len(manifest["designs"])
    out["candidate_count"] = len(kept)
    out["hardware_reachability"] = {
        "model": "measured manta_hand travel envelope",
        "accepted": len(kept), "rejected": len(rejected), "rejections": rejected,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(kept)}/{len(manifest['designs'])} designs are inside the measured rails "
          f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
