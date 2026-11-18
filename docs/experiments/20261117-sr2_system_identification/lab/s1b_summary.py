"""S1(b) of 2026-11-18 into the study's data. The owner lowered the arm 43 mm, switched it off and moved the sphere aside
("the middle finger tag should be visible from at least one camera now, when it is curled into the tag-visible joint
config. if you yaw the thumb finger toward +x and also step the thumb +x, it should be visible by both"). Four sets ran
with s1b_tags.py at that arm pose: middle and index (each twice), thumb with its gantry 12 mm toward +x (twice) and thumb
at its home mount (once). Each set's poses and rows are copied to data/s1/20261118-<set>_poses.json and _rows.jsonl, and
data/s1/20261118-s1b_summary.json gives per set: the poses seen and by which cameras, the tag's offset from the model
(median and range per axis of P), the measured normal's turn from the model's about the finger's flexion axis (sign:
toward more flexion) and out of its flexion plane, and the difference between the two runs of the same commanded pose.

    /usr/bin/python3 s1b_summary.py [LOGDIR]
"""
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "docs/experiments/20261115-hom_bench_trials/lab")]
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs/20261118-sr2_s1b_tags"
OUT = HERE.parent / "data/s1"
SETS = ("middle", "index", "thumb", "thumb_home")


def flex_split(FT, f, mount, q, n_pred, n_meas):
    """(turn about the flexion axis toward more flexion, turn out of the flexion plane), deg."""
    n0 = FT.fk(f, mount, q)[1]
    n1 = FT.fk(f, mount, (q[0], q[1], q[2] + 1.0))[1]
    ax = np.cross(n0, n1)
    ax /= np.linalg.norm(ax)                       # a turn about ax by a positive angle is more flexion
    n_pred, n_meas = np.asarray(n_pred), np.asarray(n_meas)
    flex = math.degrees(math.atan2(float(np.cross(n_pred, n_meas) @ ax), float(n_pred @ n_meas)))
    out = math.degrees(math.asin(float(np.clip(n_meas @ ax, -1, 1))))
    return flex, out


def main():
    sys.argv = sys.argv[:1]
    import fingertip_tag_check as FT
    summary = {"what": __doc__.split("\n\n")[0], "log": str(LOG.relative_to(ROOT)) if LOG.is_relative_to(ROOT) else str(LOG),
               "sets": {}}
    for s in SETS:
        d = LOG / s
        if not (d / "rows.jsonl").exists():
            continue
        shutil.copy(d / "poses.json", OUT / f"20261118-{s}_poses.json")
        shutil.copy(d / "rows.jsonl", OUT / f"20261118-{s}_rows.jsonl")
        rows = [json.loads(x) for x in (d / "rows.jsonl").read_text().splitlines() if x.strip()]
        poses = json.loads((d / "poses.json").read_text())
        seen = [r for r in rows if "d_mm" in r]
        f = rows[0]["finger"]
        D = np.array([r["d_mm"] for r in seen])
        splits = [flex_split(FT, f, r["mounts_P"], r["q_meas"], r["n_pred"], r["n_meas"]) for r in seen]
        cams = defaultdict(int)
        for r in seen:
            for k, v in r.get("cams", {}).items():
                cams[k] += v
        by_q = defaultdict(list)
        for r in rows:
            by_q[tuple(r["q_cmd"])].append(r)
        rep = [float(np.linalg.norm(np.subtract(v[1]["d_mm"], v[0]["d_mm"]))) for v in by_q.values()
               if len(v) >= 2 and all("d_mm" in x for x in v[:2])]
        unseen = sorted({tuple(r["q_cmd"]) for r in rows if "d_mm" not in r})
        rec = {"finger": f, "runs": len(rows), "poses": len(by_q), "seen_runs": len(seen),
               "station_mounts": poses["station_mounts"][f], "mount_P": poses["mounts_P"][f],
               "palm_origin_B_mm": np.round(np.array(poses["T_BP"])[:3, 3], 1).tolist(), "arm_mode": poses.get("arm_mode"),
               "samples_by_cameras": dict(cams), "unseen_q_cmd": [list(u) for u in unseen],
               "d_mm_median": np.round(np.median(D, axis=0), 2).tolist(), "d_mm_min": np.round(D.min(axis=0), 2).tolist(),
               "d_mm_max": np.round(D.max(axis=0), 2).tolist(),
               "abs_d_mm_median": round(float(np.median(np.linalg.norm(D, axis=1))), 2),
               "normal_deg_median": round(float(np.median([r["normal_deg"] for r in seen])), 1),
               "normal_flex_deg_median": round(float(np.median([x[0] for x in splits])), 1),
               "normal_out_of_plane_deg_median": round(float(np.median([x[1] for x in splits])), 1),
               "repeat_mm": {"n": len(rep), "median": round(float(np.median(rep)), 2) if rep else None,
                             "max": round(max(rep), 2) if rep else None},
               "sd_within_pose_mm_median": round(float(np.median([np.linalg.norm(r["sd_mm"]) for r in seen])), 2)}
        summary["sets"][s] = rec
        print(f"{s:10s} {f:6s} mount {rec['station_mounts']}: {len(seen)}/{len(rows)} runs seen "
              f"({', '.join(f'{k} {v}' for k, v in cams.items())}); offset median {rec['d_mm_median']} mm "
              f"(|{rec['abs_d_mm_median']}|), range x {rec['d_mm_min'][0]}..{rec['d_mm_max'][0]}, y {rec['d_mm_min'][1]}.."
              f"{rec['d_mm_max'][1]}, z {rec['d_mm_min'][2]}..{rec['d_mm_max'][2]}; normal {rec['normal_deg_median']} deg "
              f"({rec['normal_flex_deg_median']:+.1f} about the flexion axis, {rec['normal_out_of_plane_deg_median']:+.1f} "
              f"out of plane); repeat {rec['repeat_mm']}; sd in a pose {rec['sd_within_pose_mm_median']} mm"
              + (f"; unseen {rec['unseen_q_cmd']}" if unseen else ""))
    (OUT / "20261118-s1b_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {OUT / '20261118-s1b_summary.json'}")


if __name__ == "__main__":
    main()
