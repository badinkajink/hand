#!/usr/bin/env python3
"""The tag map of a printed object in the object's HOM frame O, for the tracker of the HOM trials
(docs/experiments/20261115-hom_bench_trials/lab/hom_trial.py --map; owner, 2026-11-15: "make sure we are ready to start
collecting data and calibrating from real trials of HOM").

The fitted map of an object's block (sr2hand.tracking.pose.bundle_adjust_multi from about 40 two-camera views, written
with TagMap.to_json, in a frame of the fit's own choice) is moved into O by the rigid transform that puts its corners
closest to the CAD map's (object_tracking_mjcf.object_map: the block at the printed stem below the object), over the
tags both maps have (sr2hand.geometry.rotations.kabsch). The corners keep their fitted shape, and the residual per corner
is how far the printed tag layout lies from the CAD's (on the test block of 2026-11-07, 0.33 mm RMS: the tags sit about
0.3 mm inside the faces). A map of the block's tags alone cannot show where the block sits on the object: the fit puts it
where the CAD does, so the stem's length and the glue's turn about it are taken as printed. On the cylinder the key and
slot set the turn, and equal gaps at both ends of a plate laid on the face of tag 10 or 13 check it (2026-11-13). Tags
the print does not carry (the cylinder's end tags, left off on 2026-11-13) are left out.

    .venv/bin/python scripts/hardware/object_tag_map.py --object cylinder --fitted FITTED.json --out MAP_O.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

import numpy as np  # noqa: E402

import object_tracking_mjcf as OM  # noqa: E402
from sr2hand.geometry.rotations import kabsch, log_so3  # noqa: E402
from sr2hand.tracking import tags as TG  # noqa: E402


def to_object_frame(fitted: TG.TagMap, shape: str):
    """(the fitted map in O, report): the map moved by the Kabsch fit of its corners onto the CAD object map's."""
    cad = OM.object_map(shape)
    ids = sorted(set(fitted.corners) & set(cad.corners))
    if len(ids) < 2:
        raise SystemExit(f"{shape}: the fitted map shares {len(ids)} tag(s) with the CAD map ({sorted(cad.corners)})")
    A, B = fitted.stacked(ids), cad.stacked(ids)
    R, t = kabsch(A, B)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    out = fitted.transformed(T)
    d = np.linalg.norm(out.stacked(ids) - B, axis=1)
    per_tag = {i: float(np.linalg.norm(out.corners[i] - cad.corners[i], axis=1).max()) for i in ids}
    rep = {"shape": shape, "tags": ids, "rms_mm": float(np.sqrt(np.mean(d ** 2))), "max_mm": float(d.max()),
           "per_tag_max_mm": per_tag, "rotation_deg": float(np.degrees(np.linalg.norm(log_so3(R)))),
           "translation_mm": t.round(3).tolist(), "left_out": sorted(set(fitted.corners) - set(cad.corners))}
    return out, rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--object", required=True, choices=sorted(OM.OBJECTS))
    ap.add_argument("--fitted", required=True, help="the block's fitted map (TagMap.to_json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = json.loads(Path(a.fitted).read_text())
    fitted = TG.TagMap.from_corners(m["family"], m["size_mm"], {int(i): c for i, c in m["corners_mm"].items()})
    out, rep = to_object_frame(fitted, a.object)
    keep = {i: out.corners[i] for i in rep["tags"]}
    res = TG.TagMap.from_corners(out.family, out.size, keep).to_json()
    res["frame"] = (f"O, the {a.object}'s HOM frame (object_tracking_mjcf.object_map), from {a.fitted} by a Kabsch fit "
                    "of the shared tags' corners onto the CAD map")
    res["fit"] = rep
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(rep, indent=1))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
