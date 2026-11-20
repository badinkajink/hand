#!/usr/bin/env python3
"""The fingertip forces of the twist at (70, 110) grasp 0 on the hand, read by the fingertip boards, against the contact
forces of the ranking's simulation for the same grasp (hand-4a, 2026-11-19 21:3x; owner: "dont we literally have
fingertip force readings to verify?").

Hand: every HOM server record of 2026-11-17 for the plan 20261116-film_x70_y110_twist_gr_plan.json. Each period's
"contacts" holds, per finger in contact, the boards' estimate of the contact point and its normal force (N), from
tip_contacts on corvus, logged whether or not HOM used them (tips on or off). Simulation: the ranking's run
(sim/20261119-twist_ranking_search.py's _run, the calibrated 7.5 V plant) of call 162 grasp 0 at seeds 0-2 under point
contact and the pads, whose stream has the same field from the plant's contacts. Per finger over the task window
(2.0-7.8 s): the share of periods with a contact above 0.02 N and the median force over those periods. HOM commands
f_tripod = 2.5 N per finger in both.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 CUDA_VISIBLE_DEVICES= .venv/bin/python \\
        docs/experiments/20261119-twist_layout_ranking/sim/20261119-tip_forces_vs_sim.py
Writes data/20261119-tip_forces_vs_sim.json.
"""
import glob
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
spec = importlib.util.spec_from_file_location("twist_search", HERE / "20261119-twist_ranking_search.py")
SR = importlib.util.module_from_spec(spec)
sys.modules["twist_search"] = SR
spec.loader.exec_module(SR)
import numpy as np  # noqa: E402

PLAN = "20261116-film_x70_y110_twist_gr_plan.json"
FINGERS = ("thumb", "index", "middle")
T0, T1, F_MIN = 2.0, 7.8, 0.02


def summarize(samples):
    """Per finger: share of task periods in contact and median force while in contact."""
    task = [s for s in samples if T0 <= float(s["t"]) <= T1]
    out = {}
    for f in FINGERS:
        F = np.array([float((s.get("contacts") or {}).get(f, [None, 0.0])[1] or 0.0) for s in task])
        on = F > F_MIN
        out[f] = {"share": round(float(on.mean()), 3) if len(F) else None,
                  "median_N": round(float(np.median(F[on])), 3) if on.any() else 0.0,
                  "mean_N_all_periods": round(float(F.mean()), 3) if len(F) else None}
    return out


def hand():
    rows = []
    for p in sorted(glob.glob(str(ROOT / "docs/experiments/20261115-hom_bench_trials/runs/20261117-*_trial*.jsonl"))):
        L = [json.loads(x) for x in open(p) if x.strip()]
        if not L or L[0].get("kind") != "header" or not str(L[0].get("plan", "")).endswith(PLAN):
            continue
        smp = [r for r in L[1:] if "t" in r]
        if len(smp) < 300 or int(smp[0].get("seq", 0)) != 0:              # fragments and short records
            continue
        rows.append({"file": Path(p).name, "tips": L[0].get("tips"), **summarize(smp)})
    return rows


def sim():
    R = SR.load_replan()
    R.RG.gear_class = R.RC2.clutch_class
    R.RC2.STIFF, R.RC2.BREAKAWAY = True, 0.0
    R.RG.RF.RL.rows_file()
    SR._R["mod"] = R
    SR._init()
    from sr2hand.control import hom_bench as HB
    sim0 = HB.simulate
    got = {}

    def keep(*a, **k):
        stream, res = sim0(*a, **k)
        got["stream"] = stream
        return stream, res
    HB.simulate = keep
    xy = next(xy for c, xy, n in SR.layouts() if c == 162)
    out = []
    for m in SR.MODELS:
        for seed in (0, 1, 2):
            job = {**SR.base_job(162, xy, m, contact=m), "k": 0, "seed": seed}
            r = SR._run(job)
            out.append({"model": m, "seed": seed, "track": r.get("track"), "held": r.get("held"),
                        **summarize(got.get("stream") or [])})
            print(m, seed, r.get("track"), {f: out[-1][f] for f in FINGERS}, flush=True)
    HB.simulate = sim0
    return out


def main():
    H = hand()
    S = sim()

    def agg(rows):
        return {f: {"share": round(float(np.mean([r[f]["share"] for r in rows])), 3),
                    "median_N": round(float(np.median([r[f]["median_N"] for r in rows])), 3)} for f in FINGERS}
    res = {"what": __doc__.split("\n\n")[0].replace("\n", " "), "written": time.strftime("%F %H:%M"),
           "commanded_N": 2.5, "hand_trials": H, "sim_runs": S,
           "hand": {t: agg([r for r in H if r["tips"] == t]) for t in ("off", "on") if any(r["tips"] == t for r in H)},
           "sim": {m: agg([r for r in S if r["model"] == m]) for m in SR.MODELS}}
    (STUDY / "data/20261119-tip_forces_vs_sim.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in list(res["hand"].items()) + list(res["sim"].items()):
        print(f"{k:5s} " + "  ".join(f"{f} {v[f]['median_N']:.2f} N in {v[f]['share']:.0%}" for f in FINGERS))
    print(f"hand trials {len(H)}, sim runs {len(S)}")


if __name__ == "__main__":
    main()
