"""The night's g0 twists beside the simulation: the hand, the blind closed-loop predictions and the open-loop replays per
contact model and plant (hand-4a, 2026-11-18; the replays by night_replay.py). Writes
docs/experiments/20261117-sr2_system_identification/data/20261118-night_trials_replay.json and prints the tables.

    python3 logs/20261117-g13_open_pose/night_replay_summary.py logs/20261118-night_replay/replays.jsonl
"""
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.home() / "Programs/hand"
STUDY = ROOT / "docs/experiments/20261117-sr2_system_identification"
OUT = STUDY / "data/20261118-night_trials_replay.json"
PRED = STUDY / "data/20261117-g0_x70y110_twist_predictions.json"
LIMIT = STUDY / "data/20261117-twist_limit_and_trajectories.json"
JUMP_DEG = 8.0                  # deg: a 0.2 s rise of the twist above this counts as a jump


def mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return round(float(np.mean(xs)), 3) if xs else None


def jumps(t, tw):
    """The largest rise of the twist over 0.2 s from the task's start (2.0 s), and the number of 0.2 s windows that rose
    more than JUMP_DEG (traces every 0.1 s)."""
    t, tw = np.asarray(t, float), np.asarray(tw, float)
    ok = np.isfinite(tw) & (t >= 2.0)
    d = tw[ok][2:] - tw[ok][:-2]
    return (round(float(d.max()), 1), int((d > JUMP_DEG).sum())) if len(d) else (None, 0)


def hand_metrics(r):
    tr, mo = r["hand"]["track"], r["hand"]["motion"]
    return {"track": tr.get("score"), "twist_task_end_deg": tr.get("twist_task_end_deg"),
            "twist_end_deg": mo.get("twist_end_deg"), "rise_max_mm": mo.get("rise_max_mm"),
            "tilt_max_deg": mo.get("tilt_max_deg"), "drift_max_mm": mo.get("drift_max_mm")}


def replay_metrics(r):
    tr, mo = r["replay"]["track"], r["replay"]["motion"]
    jr = r["joint_rms_deg"]
    return {"track": tr.get("score"), "twist_task_end_deg": tr.get("twist_task_end_deg"),
            "twist_end_deg": mo.get("twist_end_deg"), "rise_max_mm": mo.get("rise_max_mm"),
            "tilt_max_deg": mo.get("tilt_max_deg"), "drift_max_mm": mo.get("drift_max_mm"),
            "twist_trace_rms_deg": r["twist_trace_rms_deg"],
            "joint_rms_deg": {w: jr[w]["all"] for w in jr},
            "joint_bias_task_deg": {j: jr["task"]["bias"][j] for j in ("index.aa", "middle.aa", "thumb.fe1")},
            "object_task": r["object_task"]}


def blind():
    """The predictions at 7.5 V committed at 2026-11-17 18:30 (before the night's first trial, 19:42), and the no-slip
    closed-loop runs of 22:28 (after both blocks)."""
    p = json.loads(PRED.read_text())
    out = []
    for s in p["summary"]:
        if s["supply_V"] != 7.5:
            continue
        out.append({"contact": s["contact_model"], "contacts_to_HOM": s["contacts_to_HOM"], "when": "blind, 2026-11-17 18:30",
                    "runs": s["runs"], "held": s["held"], "track": s["track_mean"], "twist_end_deg": s["twist_end_deg"],
                    "rise_max_mm": s["rise_max_mm"], "drift_max_mm": s["drift_max_mm"]})
    lim = json.loads(LIMIT.read_text())
    by = collections.defaultdict(list)
    for r in lim["noslip_sim_g0_7v5"]:
        by[r["contact"]].append(r)
    for c, rs in by.items():
        out.append({"contact": c, "contacts_to_HOM": False, "when": "closed loop after the trials, 2026-11-17 22:28",
                    "runs": len(rs), "held": sum(r["held"] for r in rs), "track": mean([r["track"] for r in rs]),
                    "twist_end_deg": mean([r["twist_end"] for r in rs]), "rise_max_mm": mean([r["rise_max"] for r in rs]),
                    "drift_max_mm": mean([r["drift_max"] for r in rs])})
    return out


def main():
    rows = [json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
    failed = [{k: r[k] for k in ("stamp", "contact", "volts", "error")} for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r and not r.get("diverged")]
    trials = {}
    for r in rows:
        tr = trials.setdefault(r["stamp"], {"stamp": r["stamp"], "block": r["block"], "tips": r["tips"],
                                            "flags": r["flags"], "ended": r["ended"], "hand": hand_metrics(r),
                                            "hand_jumps": jumps(r["trace"]["t"], r["trace"]["twist_hand"]),
                                            "replays": {}, "trace": {"t": r["trace"]["t"][::2],
                                                                     "twist_hand": r["trace"]["twist_hand"][::2]}})
        key = f"{r['contact']}/{r['volts']}"
        tr["replays"][key] = {**replay_metrics(r), "jumps": jumps(r["trace"]["t"], r["trace"]["twist_replay"])}
        tr["trace"][f"twist_{key}"] = r["trace"]["twist_replay"][::2]
        if key in ("pads/7v5", "pads/7v5y", "pads_stiff/7v5"):
            for j in ("index.aa", "middle.aa", "thumb.fe1"):
                tr["trace"].setdefault(j, {"hand": r["trace"][j]["hand"][::2], "target": r["trace"][j]["target"][::2]})
                tr["trace"][j][key] = r["trace"][j]["replay"][::2]
    # arms: tips and flags
    arms = collections.defaultdict(list)
    for tr in trials.values():
        arms[(tr["tips"], tr["flags"])].append(tr)
    arm_rows = []
    for (tips, flags), ts in sorted(arms.items()):
        arm_rows.append({"tips": tips, "flags": flags, "trials": sorted(t["stamp"] for t in ts),
                         "hand": {k: mean([t["hand"][k] for t in ts]) for k in ts[0]["hand"]},
                         "hand_track_min": min(t["hand"]["track"] for t in ts),
                         "hand_track_max": max(t["hand"]["track"] for t in ts)})
    # replay summaries by tips, contact and plant
    summ = []
    by = collections.defaultdict(list)
    for tr in trials.values():
        for key, rm in tr["replays"].items():
            by[(tr["tips"], key)].append((tr, rm))
    for (tips, key), xs in sorted(by.items()):
        hm = [t["hand"] for t, _ in xs]
        rm = [m for _, m in xs]
        summ.append({"tips": tips, "contact": key.split("/")[0], "plant": key.split("/")[1], "n": len(xs),
                     "hand": {k: mean([h[k] for h in hm]) for k in ("track", "twist_end_deg", "rise_max_mm",
                                                                   "tilt_max_deg", "drift_max_mm")},
                     "hand_jump_max_deg": mean([t["hand_jumps"][0] for t, _ in xs]),
                     "hand_jumps_n": mean([t["hand_jumps"][1] for t, _ in xs]),
                     "replay": {k: mean([m[k] for m in rm]) for k in ("track", "twist_end_deg", "rise_max_mm",
                                                                     "tilt_max_deg", "drift_max_mm",
                                                                     "twist_trace_rms_deg")},
                     "track_abs_diff": mean([abs(m["track"] - t["hand"]["track"]) for t, m in xs]),
                     "replay_jump_max_deg": mean([m["jumps"][0] for m in rm]),
                     "replay_jumps_n": mean([m["jumps"][1] for m in rm]),
                     "joint_rms_deg": {w: mean([m["joint_rms_deg"].get(w) for m in rm]) for w in ("approach", "grasp", "task")},
                     "joint_bias_task_deg": {j: mean([m["joint_bias_task_deg"][j] for m in rm])
                                             for j in ("index.aa", "middle.aa", "thumb.fe1")},
                     "object_task": {k: mean([m["object_task"][k] for m in rm]) for k in ("rot_rms_deg", "pos_rms_mm")}})
    meta = {
        "what": "The night's g0 twists at (70, 110) mm (2026-11-17, A/B 1 at 19:42-20:08, A/B 2 at 21:06-21:25; 20 scored "
                "trials, the 5 one-period fragments left out) beside the simulation: the blind closed-loop predictions "
                "and the open-loop replays of each trial's recorded targets, per contact model and plant.",
        "asked": "owner, 2026-11-18 09:1x: 'when you report these trial results, also, please make sure to also report the "
                 "simulation prediction and replay ... across multiple contact model configurations'; 11:0x: 'ok yea "
                 "please do that'",
        "hand_scores": "docs/experiments/20261115-hom_bench_trials/data/20261117-night_repeats_tracking.json (the lab's; "
                       "recomputed here with hom_trial_tracking.track and motion, identical)",
        "replay": "sim/20261118-night_trials_replay.py: from the hand's start (first measured joints; the object at the "
                  "plan's height with the tracked x-y and heading, on the table under its block), every period's "
                  "recorded targets 12 ms after their state, the table removed at corvus's lift start; rev_clutch2's "
                  "plant (gears the load cannot backdrive, c 0.6, stiff joint friction), MCPs stopped at S1(a)'s "
                  "-10.3/-8.5 deg, friction 1.0",
        "plants": {"7v5": "kp 0.825, tau_max 0.27, index/middle yaw kp 0.525 (5 V x 1.5: the blind predictions' plant)",
                   "5v": "kp 0.55, tau_max 0.18, yaw 0.35 (fitted on 2026-11-16 at 5 V)",
                   "7v5y": "7v5 with the yaws at 0.825"},
        "contacts": {"pads": "1 mm sphere pads, soft tangential contact (impratio 1)", "point": "point contact, impratio 1",
                     "pads_stiff": "pads, impratio 100", "pads_stiff_ns": "pads, impratio 100, noslip 30 iterations"},
        "metrics": {"track": "max(0, 1 - err/ref) over 2.0 s to t_end (7.8 s), err the RMS angle between the tracked "
                             "rotation since the window's start and the commanded one (the trial's w_ref integrated); "
                             "the replay's object sampled at the tracker's capture times",
                    "twist_end_deg": "rotation about z_P from the first tracked sample, at the record's end (8.3 s)",
                    "rise_max_mm": "largest rise along z_P from the first sample", "tilt_max_deg": "largest tilt from "
                    "the first sample", "drift_max_mm": "largest 3-D distance from the first sample",
                    "twist_trace_rms_deg": "RMS of the replay's minus the hand's twist from 2.0 s",
                    "jumps": f"(largest rise of the twist over 0.2 s from 2.0 s, deg; number of 0.2 s windows rising "
                             f"more than {JUMP_DEG} deg)",
                    "joint_rms_deg": "RMS of the replayed minus the measured joint angle (servo deg) at the read-backs' "
                                     "times: approach 0-0.6 s (free air), grasp 0.6-2.0 s, task from 2.0 s",
                    "joint_bias_task_deg": "mean replayed minus measured angle, task window (servo deg)",
                    "object_task": "RMS orientation (deg) and position (mm) error against the tracked pose, task window"},
        "check": "sim/20261118-night_trials_replay_check.py: HOM's closed-loop twist in simulation from the start of "
                 "19:42, written as a server record and replayed with no delay on its own plant: identical tracking, "
                 "twist, rise, joints and object pose (0.0 deg and 0.0 mm RMS) with soft and with stiff pads at 7v5",
        "failed": failed, "written": time.strftime("%Y-%m-%d %H:%M"),
        "scripts": ["sim/20261118-night_trials_replay.py", "sim/20261118-night_trials_replay_summary.py",
                    "sim/20261118-night_trials_replay_check.py"]}
    out = {"meta": meta, "arms": arm_rows, "blind": blind(), "replay_summary": summ,
           "trials": sorted(trials.values(), key=lambda t: t["stamp"])}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")
    for a in arm_rows:
        print(a["tips"], a["flags"], len(a["trials"]), a["hand"])
    for b in out["blind"]:
        print("blind", b)
    for s in summ:
        print(s["tips"], s["contact"], s["plant"], s["n"], s["replay"], s["track_abs_diff"], s["joint_rms_deg"],
              s["joint_bias_task_deg"], s["replay_jump_max_deg"], s["replay_jumps_n"], s["hand_jump_max_deg"],
              s["hand_jumps_n"])


if __name__ == "__main__":
    main()
