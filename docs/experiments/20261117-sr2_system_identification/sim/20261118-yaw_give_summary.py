"""Summary of the yaw-give replays (sim/20261118-yaw_give_replay.py; W3 of the identification queue, 2026-11-18) beside
the hand and the rigid replays of data/20261118-night_trials_replay.json, for the same tips-off trials of the night.

Per contact model and spring stiffness K_C (rigid: the clutch plant), the means over the trials of: HOM's tracking score
(lab's hom_trial_tracking), the sphere's twist at the end and its largest rise; the twist trace's RMS against the hand's
from 2.0 s; the joint RMS against the read-backs in free air (0-0.6 s), in the grasp (0.6-2.0 s) and in the task (from
2.0 s); the index's and middle's yaw shortfall (how far the replay's reading stops short of the hand's, toward the
sphere; the bias of the yaw against the read-backs, sign set so that positive is short); and the jumps (the largest
twist rise over 0.2 s from 2.0 s, and the number of 0.2 s windows that rose more than 8 deg).

    python3 sim/20261118-yaw_give_summary.py logs/20261118-night_replay/yaw_give.jsonl \\
        [--out data/20261118-yaw_give_replay.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
SHORT = {"index.aa": -1.0, "middle.aa": 1.0}          # servo-frame bias sign that points away from the sphere


def mean(v):
    v = [x for x in v if x is not None]
    return round(float(np.mean(v)), 2) if v else None


def from_give(r):
    jr = r["joint_rms_deg"]
    return {"track": r["replay"]["track"].get("score"), "twist_end_deg": r["replay"]["motion"].get("twist_end_deg"),
            "rise_max_mm": r["replay"]["motion"].get("rise_max_mm"), "twist_trace_rms_deg": r["twist_trace_rms_deg"],
            "joint_rms_deg": {w: jr[w]["all"] for w in ("approach", "grasp", "task") if w in jr},
            "yaw_short_grasp_deg": {k: round(s * jr["grasp"]["bias"][k], 2) for k, s in SHORT.items()},
            "yaw_short_task_deg": {k: round(s * jr["task"]["bias"][k], 2) for k, s in SHORT.items()},
            "jumps": r.get("replay_jumps"), "object_task": r["object_task"]}


def from_rigid(x):
    return {"track": x["track"], "twist_end_deg": x["twist_end_deg"], "rise_max_mm": x["rise_max_mm"],
            "twist_trace_rms_deg": x["twist_trace_rms_deg"], "joint_rms_deg": x["joint_rms_deg"],
            "yaw_short_grasp_deg": None,
            "yaw_short_task_deg": {k: round(s * x["joint_bias_task_deg"][k], 2) for k, s in SHORT.items()},
            "jumps": x["jumps"], "object_task": x["object_task"]}


def agg(ms):
    out = {k: mean([m[k] for m in ms]) for k in ("track", "twist_end_deg", "rise_max_mm", "twist_trace_rms_deg")}
    out["joint_rms_deg"] = {w: mean([m["joint_rms_deg"].get(w) for m in ms]) for w in ("approach", "grasp", "task")}
    for key in ("yaw_short_grasp_deg", "yaw_short_task_deg"):
        out[key] = None if ms[0][key] is None else {k: mean([m[key][k] for m in ms]) for k in SHORT}
    out["jump_max_deg"] = mean([m["jumps"][0] for m in ms])
    out["jumps_over_8deg"] = mean([m["jumps"][1] for m in ms])
    out["object_rot_rms_deg"] = mean([m["object_task"]["rot_rms_deg"] for m in ms])
    out["trials"] = len(ms)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--out", default=str(STUDY / "data/20261118-yaw_give_replay.json"))
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(a.rows) if x.strip()]
    bad = [r for r in rows if "error" in r or r.get("diverged")]
    rows = [r for r in rows if "error" not in r and not r.get("diverged")]
    rep = json.loads((STUDY / "data/20261118-night_trials_replay.json").read_text())
    trials = {t["stamp"]: t for t in rep["trials"]}
    stamps = sorted({r["stamp"] for r in rows})
    hand = agg([{**{k: trials[s]["hand"][k] for k in ("track", "twist_end_deg", "rise_max_mm")},
                 "twist_trace_rms_deg": None, "joint_rms_deg": {}, "yaw_short_grasp_deg": None,
                 "yaw_short_task_deg": None, "jumps": trials[s]["hand_jumps"], "object_task": {"rot_rms_deg": None}}
                for s in stamps])
    table = {"hand": hand}
    for c in sorted({r["contact"] for r in rows}):
        ms = [from_rigid(trials[s]["replays"][f"{c}/7v5"]) for s in stamps if f"{c}/7v5" in trials[s]["replays"]]
        table[f"{c} rigid"] = agg(ms)
        for k in sorted({r["k_c"] for r in rows if r["contact"] == c}):
            table[f"{c} kc{k:g}"] = agg([from_give(r) for r in rows if r["contact"] == c and r["k_c"] == k])
    for name, v in table.items():
        ys = v["yaw_short_task_deg"]
        print(f"{name:18s} n {v['trials']:2d} track {v['track']} twist {v['twist_end_deg']} rise {v['rise_max_mm']} "
              f"trace {v['twist_trace_rms_deg']} joints {v['joint_rms_deg']} yaw short task "
              f"{ys and (ys['index.aa'], ys['middle.aa'])} jumps {v['jump_max_deg']}/{v['jumps_over_8deg']}")
    out = {"what": __doc__.split("\n\n")[0].replace("\n", " "),
           "scripts": ["docs/experiments/20261117-sr2_system_identification/sim/20261118-yaw_give_replay.py",
                       "docs/experiments/20261117-sr2_system_identification/sim/20261118-yaw_give_summary.py"],
           "rows": a.rows, "trials": stamps, "failed": [(r["stamp"], r["contact"], r.get("k_c")) for r in bad],
           "table": table,
           "per_trial": [{"stamp": r["stamp"], "contact": r["contact"], "k_c": r["k_c"], **from_give(r),
                          "hand": trials[r["stamp"]]["hand"], "hand_jumps": trials[r["stamp"]]["hand_jumps"]}
                         for r in sorted(rows, key=lambda r: (r["contact"], r["k_c"], r["stamp"]))]}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
