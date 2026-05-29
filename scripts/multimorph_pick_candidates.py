"""Pick K candidates from a phase1 all_candidates.csv for multi-morphology eval.

Per docs/rl/multimorphology.md (Path A): we want id=0 (sanity), the nearest
neighbor, the median-distance candidate, and the p95-distance candidate by
9-dim Euclidean distance from candidate_id=0's morphology vector.

For each pick, also resolve the generated scene XML path so the downstream
CEM run can take `--scene-xml <path>` directly. Generated MJCFs live under
`<run18_root>/generated_mjcf/<object>_<morph_suffix>.xml`.

Outputs JSON to stdout:
  [
    {"candidate_id": 0,    "rank_label": "id0",      "distance": 0.0,    "scene_xml": "..."},
    {"candidate_id": ...,  "rank_label": "nearest",  "distance": ...,    "scene_xml": "..."},
    {"candidate_id": ...,  "rank_label": "p50",      "distance": ...,    "scene_xml": "..."},
    {"candidate_id": ...,  "rank_label": "p95",      "distance": ...,    "scene_xml": "..."},
  ]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from morphohand.sampling.morphology import MorphologyValues, morph_suffix  # noqa: E402

DIMS = (
    "thumb_x", "thumb_y", "thumb_len",
    "index_x", "index_y", "index_len",
    "middle_x", "middle_y", "middle_len",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates-csv", type=Path, required=True,
                   help="phase1 all_candidates.csv (e.g. run18_final/cube/all_candidates.csv)")
    p.add_argument("--generated-mjcf-dir", type=Path, required=True,
                   help="Directory containing generated scene XMLs.")
    p.add_argument("--object-prefix", type=str, default="cube",
                   help="Scene XML filename prefix (e.g. 'cube' for cube_<suffix>.xml).")
    args = p.parse_args()

    import csv
    rows = []
    with args.candidates_csv.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)

    morphs = np.array([[float(r[d]) for d in DIMS] for r in rows])
    ref = morphs[0]
    dists = np.linalg.norm(morphs - ref[None, :], axis=1)

    order = np.argsort(dists)
    nearest_idx = int(order[1])
    p50_idx = int(order[len(order) // 2])
    p95_idx = int(order[int(0.95 * len(order))])

    picks = [
        (0, "id0", float(dists[0])),
        (nearest_idx, "nearest", float(dists[nearest_idx])),
        (p50_idx, "p50", float(dists[p50_idx])),
        (p95_idx, "p95", float(dists[p95_idx])),
    ]

    out = []
    for cid, label, dist in picks:
        r = rows[cid]
        m = MorphologyValues(
            thumb_x=float(r["thumb_x"]),  thumb_y=float(r["thumb_y"]),  thumb_len=float(r["thumb_len"]),
            index_x=float(r["index_x"]),  index_y=float(r["index_y"]),  index_len=float(r["index_len"]),
            middle_x=float(r["middle_x"]),middle_y=float(r["middle_y"]),middle_len=float(r["middle_len"]),
        )
        suffix = morph_suffix(m)
        scene_xml = args.generated_mjcf_dir / f"{args.object_prefix}_{suffix}.xml"
        out.append({
            "candidate_id": cid,
            "rank_label": label,
            "distance": dist,
            "scene_xml": str(scene_xml),
            "scene_xml_exists": scene_xml.exists(),
            "morphology": {d: float(r[d]) for d in DIMS},
        })

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
