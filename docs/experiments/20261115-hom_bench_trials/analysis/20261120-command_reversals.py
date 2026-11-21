#!/usr/bin/env python3
"""The commands HOM's server sent in every twist trial under the torque budget, by HOM's force and the server's smoothing
(hand-4a, 2026-11-20 20:3x; owner, 20:2x: "let's get ready to trt smoothing i guess. i imagine there was much less
jitter on the 1.0 N trials").

Each HOM server record (runs/<stamp>_twist_d40_trial<n>.jsonl) from --since on is read as sent, without a replay: the
targets the server sent (targets_sim_deg) and the joints it read, over the periods after the lift (the phases after
open, close, squeeze and lift), with the metrics of 20261120-command_jitter_replay.py:
    rev %    the share of periods in which a joint's command reverses its direction of travel (steps over 0.1 deg), over
             the nine joints
    hf       RMS of the command about its 7-period moving average, mean over the nine joints, deg
    step     RMS of the command's change per period over the nine joints, deg
and the share of periods in which the budget cut HOM's targets (scale below 1). The plan's force comes from the plan
list (data/20261119-trial_plans.json of the twist ranking): "2.5 N" for the default plans, "1.0 N" (or 1.25 N) for the
plans of stage E, "CEM" for the plans with a CEM-refined force schedule. The smoothing is the record header's
budget_joint_tau and target_tau (hom_server since af94c6028). The camera ran at 60 frames/s from 2026-11-20 11:50 on.

    python3 docs/experiments/20261115-hom_bench_trials/analysis/20261120-command_reversals.py [--since STAMP]
Writes 20261120-command_reversals.json next to this script.
"""
import argparse
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parents[2]
RUNS = BENCH / "runs"
OUT = HERE / "20261120-command_reversals.json"
PLANS = ROOT / "docs/experiments/20261119-twist_layout_ranking/data/20261119-trial_plans.json"
FPS60_FROM = "20261120-115000"
spec = importlib.util.spec_from_file_location("replay_metrics", HERE / "20261120-command_jitter_replay.py")
RM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RM)


def force_class(prm):
    if not prm:
        return "2.5 N"
    if "f_tripod_knots" in prm or "w_slide_tripod" in prm:
        return "CEM"
    f = float(prm.get("f_tripod", 2.5))
    return f"{f:.1f} N" if abs(10 * f - round(10 * f)) < 1e-9 else f"{f:g} N"


def summary(rs):
    def m(key, sub):
        return round(float(np.mean([r[key][sub] for r in rs])), 2)
    return {"trials": len(rs), "cmd_rev_pct": m("cmd", "rev_pct"), "cmd_hf_deg": m("cmd", "hf_deg"),
            "cmd_step_deg": m("cmd", "step_deg"), "servo_rev_pct": m("servo", "rev_pct"),
            "budget_below_1_pct": round(float(np.mean([r["budget_below_1_pct"] for r in rs])), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="20261117-170000", help="first record stamp (the 0.56 N m budget from 11-17)")
    a = ap.parse_args()
    lst = {Path(e["plan"]).name: e for e in json.loads(PLANS.read_text())["plans"]}
    recs = []
    for p in sorted(RUNS.glob("*_twist_*trial*.jsonl")):
        if p.name[:15] < a.since:
            continue
        head, rows = RM.load(p)
        task = [i for i, r in enumerate(rows) if r.get("phase") not in RM.PRE]
        if head.get("task") != "twist" or len(task) < 30:
            continue
        name = Path(head["plan"]).name
        e = lst.get(name) or {}
        X = np.array([rows[i]["targets_sim_deg"] for i in task], float)
        J = np.array([[float(v) for f in ("thumb", "index", "middle") for v in rows[i]["joints"][f].values()]
                      for i in task], float)
        sc = np.array([rows[i].get("budget_scale") if rows[i].get("budget_scale") is not None else 1.0 for i in task])
        recs.append({"record": p.name, "plan": name, "in_plan_list": bool(e), "xy": e.get("xy"), "k": e.get("k"),
                     "prm": e.get("prm"), "force": force_class(e.get("prm")) if e else None,
                     "budget_nm": head.get("torque_budget"), "budget_joint_tau": head.get("budget_joint_tau"),
                     "target_tau": head.get("target_tau"),
                     "smoothed": bool(head.get("budget_joint_tau") or head.get("target_tau")),
                     "fps60": p.name[:15] >= FPS60_FROM, "periods": len(task), "cmd": RM.metrics(X),
                     "servo": RM.metrics(J), "budget_below_1_pct": round(100 * float(np.mean(sc < 0.999)), 1)})
    recs = [r for r in recs if r["cmd"] and r["servo"]]
    by = defaultdict(list)
    for r in recs:
        if r["force"] and r["budget_nm"]:
            by[(r["force"], r["smoothed"])].append(r)
    groups = [{"force": f, "smoothed": s, **summary(rs)} for (f, s), rs in sorted(by.items())]
    lay = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r["fps60"] and r["force"] in ("2.5 N", "1.0 N") and r["budget_nm"] and not r["smoothed"]:
            lay[tuple(r["xy"])][r["force"]].append(r)
    paired = [{"xy": list(xy), **{f: summary(v[f]) for f in ("2.5 N", "1.0 N")}} for xy, v in sorted(lay.items())
              if "2.5 N" in v and "1.0 N" in v]
    OUT.write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "), "generated": time.strftime("%F %H:%M"),
                               "since": a.since, "groups": groups, "layouts_at_both_forces_60fps": paired,
                               "records": recs}, indent=1) + "\n")
    for g in groups:
        print(f"{g['force']:6s} smoothed {g['smoothed']!s:5s} n={g['trials']:3d}  rev {g['cmd_rev_pct']:5.1f} %  hf "
              f"{g['cmd_hf_deg']:.2f} deg  budget<1 {g['budget_below_1_pct']:5.1f} %")
    for x in paired:
        print(f"  {tuple(x['xy'])}: 2.5 N rev {x['2.5 N']['cmd_rev_pct']} % hf {x['2.5 N']['cmd_hf_deg']} ({x['2.5 N']['trials']})"
              f"  1.0 N rev {x['1.0 N']['cmd_rev_pct']} % hf {x['1.0 N']['cmd_hf_deg']} ({x['1.0 N']['trials']})")
    print("wrote", OUT, len(recs), "records")


if __name__ == "__main__":
    main()
