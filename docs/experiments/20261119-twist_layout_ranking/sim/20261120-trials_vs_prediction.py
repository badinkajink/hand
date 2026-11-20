#!/usr/bin/env python3
"""The twist layout ranking's trials on the hand against the simulation's predictions (hand-4a, 2026-11-19/20; owner,
2026-11-19 17:2x: deploy many layouts "and seeing how well our real results match with sim").

Each of HOM server's records in docs/experiments/20261115-hom_bench_trials/runs/ from --since on is scored by
scripts/hardware/hom_trial_tracking.py's score_record, the scorer of the 11-16 and 11-17 trials: track =
max(0, 1 - err / ref) over the task's periods from 2.0 s to the schedule's end, err the RMS angle between the tracked
rotation and the commanded one. A record counts for a plan when its header names a plan of data/20261119-trial_plans.json.
Two measured scores per trial: the scorer's track over the periods that ran, and the one comparable with the
simulation, which scores 0 when the sphere left the grasp: track, or 0 when the trial did not end by the schedule's
end (the drift guard, a lost stream, a controller error). Rho is given for both; the first is the comparison.

Per layout of the default plans (roles ranked and anchor), the prediction is the mean tracking over both contact models
at seeds 0-2 (the proposal's score) and per model; the measurement is the mean over the plan's trials. Spearman's rho
between them, with a two-sided p from scipy and an interval from resampling each plan's trials (5000 draws). The CEM and
stage-E plans are compared with their own layout's default plan.

    .venv/bin/python docs/experiments/20261119-twist_layout_ranking/sim/20261120-trials_vs_prediction.py [--since STAMP]
Writes data/20261120-trials_vs_prediction.json and prints the table.
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from sr2hand.numerics.stats import spearman  # noqa: E402

RUNS = ROOT / "docs/experiments/20261115-hom_bench_trials/runs"
PLANS = STUDY / "data/20261119-trial_plans.json"
OUT = STUDY / "data/20261120-trials_vs_prediction.json"
SINCE = "20261119-210000"
DRAWS = 5000


def scorer():
    spec = importlib.util.spec_from_file_location("hom_trial_tracking", ROOT / "scripts/hardware/hom_trial_tracking.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    return M


def predicted(entry):
    """(both models, point, pads) predicted tracking at seeds 0-2 of a plan list entry."""
    pr = entry.get("prediction") or {}
    if "point" in pr:
        pt, pd = pr["point"].get("track_seeds012"), pr["pads"].get("track_seeds012")
        both = entry.get("score")
        if both is None and pt is not None and pd is not None:
            both = (pt + pd) / 2
        return both, pt, pd
    return pr.get("both_models_seeds012"), None, None


def rho_ci(layouts, key_pred):
    """Spearman's rho of the measured means against key_pred over the layouts, its scipy p, and the 95 % interval of
    rho over resampled trials (each layout's trials drawn with replacement)."""
    L = [x for x in layouts if x[key_pred] is not None and x["n"] > 0]
    if len(L) < 3:
        return {"layouts": len(L), "rho": None}
    pred = np.array([x[key_pred] for x in L])
    meas = np.array([x["measured_mean"] for x in L])
    r = spearman(pred, meas)
    p = float(stats.spearmanr(pred, meas)[1]) if np.isfinite(r) else None
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(DRAWS):
        m = [float(np.mean(rng.choice(x["scores"], len(x["scores"])))) for x in L]
        draws.append(spearman(pred, m))
    d = np.array(draws)
    d = d[np.isfinite(d)]
    return {"layouts": len(L), "rho": round(r, 3) if np.isfinite(r) else None, "p_two_sided": p,
            "rho_trials_resampled_95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]
            if len(d) else None,
            "pearson_r": round(float(np.corrcoef(pred, meas)[0, 1]), 3),
            "mean_abs_error": round(float(np.mean(np.abs(pred - meas))), 3),
            "mean_measured_minus_predicted": round(float(np.mean(meas - pred)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=SINCE, help="first trial stamp (YYYYMMDD-HHMMSS)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--plans", default=str(PLANS), help="the plan list (a test: another list)")
    a = ap.parse_args()
    M = scorer()
    lst = json.loads(Path(a.plans).read_text())["plans"]
    by_plan = {Path(e["plan"]).name: e for e in lst}
    cidx = M.corvus_index()
    trials = defaultdict(list)
    skipped = []
    for p in sorted(RUNS.glob("*_trial*.jsonl")):
        if p.name[:15] < a.since:
            continue
        rec = M.score_record(p, cidx)
        plan = Path((rec.get("plan") or {}).get("path") or "").name
        if plan not in by_plan:
            skipped.append({"file": p.name, "why": f"plan {plan or '?'} not in the list"})
            continue
        if not rec.get("scored") or rec.get("mode") != "live":
            skipped.append({"file": p.name, "why": rec.get("unscorable") or rec.get("mode")})
            continue
        tr = rec["track"].get("score")
        ended = rec["ended"].get("class")
        trials[plan].append({"stamp": rec["stamp"], "track": tr, "ended": ended,
                             "track_held": (tr if ended == "end of schedule" else 0.0) if tr is not None else None,
                             "twist_task_end_deg": rec["track"].get("twist_task_end_deg"),
                             "twist_ref_end_deg": rec["track"].get("twist_ref_end_deg"),
                             "notes": rec.get("notes"), "file": rec["file"]})
    plans = []
    for name, e in by_plan.items():
        tt = trials.get(name, [])
        both, pt, pd = predicted(e)
        sc = [t["track_held"] for t in tt if t["track_held"] is not None]
        plans.append({"plan": name, "role": e["role"], "call": e["call"], "xy": e["xy"], "k": e["k"],
                      "rank_among_deployable": e.get("rank_among_deployable"), "prm": e.get("prm"),
                      "predicted_both": both, "predicted_point": pt, "predicted_pads": pd,
                      "n": len(sc), "scores": sc, "measured_mean": round(float(np.mean(sc)), 3) if sc else None,
                      "scores_partial": [t["track"] for t in tt if t["track"] is not None],
                      "measured_sd": round(float(np.std(sc, ddof=1)), 3) if len(sc) > 1 else None,
                      "measured_partial_mean": round(float(np.mean([t["track"] for t in tt if t["track"] is not None])), 3)
                      if tt else None, "trials": tt})
    default = [x for x in plans if x["role"] in ("ranked", "anchor") and not x["prm"]]
    out = {"what": __doc__.split("\n\n")[0].replace("\n", " "), "script": str(Path(__file__).relative_to(ROOT)),
           "generated": time.strftime("%F %H:%M"), "since": a.since,
           "trials_scored": sum(x["n"] for x in plans), "skipped": skipped,
           "rho": {"both models": rho_ci(default, "predicted_both"), "point": rho_ci(default, "predicted_point"),
                   "pads": rho_ci(default, "predicted_pads"),
                   "both models, partial trials scored": rho_ci(
                       [{**x, "scores": x["scores_partial"], "n": len(x["scores_partial"]),
                         "measured_mean": x["measured_partial_mean"]} for x in default], "predicted_both")},
           "plans": plans}
    # refined plans against their layout's default plan
    base = {(x["call"], x["k"]): x for x in default}
    out["refined_vs_default"] = [
        {"plan": x["plan"], "role": x["role"], "xy": x["xy"], "measured_mean": x["measured_mean"], "n": x["n"],
         "predicted_both": x["predicted_both"],
         "default_measured_mean": (base.get((x["call"], x["k"])) or {}).get("measured_mean"),
         "default_predicted_both": (base.get((x["call"], x["k"])) or {}).get("predicted_both")}
        for x in plans if x not in default]
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'plan':58s} {'role':10s} {'pred':>6s} {'meas':>6s} {'sd':>6s} n")
    for x in sorted(plans, key=lambda x: -(x["predicted_both"] or 0)):
        print(f"{x['plan'][:58]:58s} {x['role'][:10]:10s} {x['predicted_both'] or float('nan'):6.3f} "
              f"{x['measured_mean'] if x['measured_mean'] is not None else float('nan'):6.3f} "
              f"{x['measured_sd'] if x['measured_sd'] is not None else float('nan'):6.3f} {x['n']}")
    print(json.dumps(out["rho"], indent=1))
    print(f"{out['trials_scored']} trials scored, {len(skipped)} records skipped -> {a.out}")


if __name__ == "__main__":
    main()
