#!/usr/bin/env python3
"""HOM's commanded targets in the simulation of the bench plans, closed loop (hand-4a, 2026-11-20 12:0x; the lab's
20261120-command_jitter.py and the replay 20261120-command_jitter_replay.py found the commands on the hand reversing in
39-62 % of periods under the 0.56 N m torque budget, against 5-34 % for HOM's own commands; owner, 11:5x: "this jitter
just seems highly problematic. smooth control enables more predictable control, im highly confident about this").

Each plan of the twist ranking's plan list that the hand ran is simulated as the ranking predicted it (the calibrated
7.5 V plant of 2026-11-17 with its non-backdrivable gears and MCP stops, the pads, seed 0; the replan module's _run1),
with HOM's stream kept, under variants of what HOM sees and sends:
    budget   the bench's torque budget, 0.56 N m (as predicted)
    none     no budget
    quant    the joints HOM reads rounded to the SCS0009's step, 300/1024 deg (servo_limits)
    delay    the joints HOM reads delay periods old (the bus: a period's targets act after the joints were read)
    noise    white noise of NOISE_DEG on each joint reading every period before the encoder step (the hand's
             readings flicker by 0.4-0.5 deg RMS about a 7-period mean and reverse in 62-70 % of periods)
    j, jt    hom_bench.HomBench's smoothing: the budget's offsets from the joints low-passed at budget_joint_tau, and
             (jt) HOM's targets low-passed at target_tau before the budget
Metrics over the periods from 2.0 s (after the lift) to the end, on the targets sent (sim degrees), as in the replay:
rev % (the share of periods where a joint's command reverses, steps over 0.1 deg), hf (RMS about a 7-period moving
average, deg), step (RMS change per period, deg); with the run's tracking score and whether it held the sphere.

    CUDA_VISIBLE_DEVICES= HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 .venv/bin/python \\
        docs/experiments/20261115-hom_bench_trials/analysis/20261120-command_jitter_sim.py [--variants V ...] [--workers N]
Writes 20261120-command_jitter_sim.json next to this script.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parents[2]
RANK = ROOT / "docs/experiments/20261119-twist_layout_ranking"
OUT = HERE / "20261120-command_jitter_sim.json"
STEP_DEG = 300.0 / 1024.0
NOISE_DEG = 0.45                   # deg: white noise on each joint reading, then the encoder step (the hand's flicker)
VARIANTS = {"budget": {"budget": 0.56}, "none": {"budget": None},
            "budget_quant": {"budget": 0.56, "quant": True}, "none_quant": {"budget": None, "quant": True},
            "budget_delay": {"budget": 0.56, "delay": 1}, "none_delay": {"budget": None, "delay": 1},
            "budget_delay_quant": {"budget": 0.56, "delay": 1, "quant": True},
            "budget_delay2": {"budget": 0.56, "delay": 2},
            "budget_noise": {"budget": 0.56, "noise": True, "quant": True},
            "none_noise": {"budget": None, "noise": True, "quant": True},
            "budget_noise_j": {"budget": 0.56, "noise": True, "quant": True, "budget_joint_tau": 0.1},
            "budget_noise_jt": {"budget": 0.56, "noise": True, "quant": True, "budget_joint_tau": 0.1,
                                "target_tau": 0.04},
            "budget_noise_jt8": {"budget": 0.56, "noise": True, "quant": True, "budget_joint_tau": 0.1,
                                 "target_tau": 0.08},
            "budget_j": {"budget": 0.56, "budget_joint_tau": 0.1},
            "budget_jt": {"budget": 0.56, "budget_joint_tau": 0.1, "target_tau": 0.04}}

spec = importlib.util.spec_from_file_location("replay_metrics", HERE / "20261120-command_jitter_replay.py")
RM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RM)
spec = importlib.util.spec_from_file_location("twist_search", RANK / "sim/20261119-twist_ranking_search.py")
SR = importlib.util.module_from_spec(spec)
sys.modules["twist_search"] = SR
spec.loader.exec_module(SR)


def plans_all():
    """The plan list's default and stage E plans (roles ranked and anchor, no CEM plans)."""
    lst = json.loads((RANK / "data/20261119-trial_plans.json").read_text())["plans"]
    return [e for e in lst if e.get("role") in ("ranked", "anchor") and "_cem_" not in e["plan"]]


def plans_run():
    """The plan list's entries whose plans the hand ran under the 0.56 N m budget (server records from 11-17 17:00)."""
    names = set()
    for p in sorted((BENCH / "runs").glob("*_twist_*trial*.jsonl")):
        if p.name[:15] < "20261117-170000":
            continue
        h = json.loads(p.open().readline())
        names.add(Path(h["plan"]).name)
    lst = json.loads((RANK / "data/20261119-trial_plans.json").read_text())["plans"]
    return [e for e in lst if Path(e["plan"]).name in names and "_cem_" not in e["plan"]]


def run(item):
    """One closed-loop run of a plan entry under a variant; the stream's metrics."""
    from sr2hand.control import hom_bench as HB
    v = VARIANTS[item["variant"]]
    e = item["entry"]
    sim0, bs0, Bench0 = HB.simulate, HB.bench_sample, HB.HomBench
    got = {}

    def keep(*a, **k):
        stream, res = sim0(*a, **k)
        got["stream"] = stream
        return stream, res

    hist = []
    rng = np.random.default_rng(20261120)

    def sample(bench, s):
        smp = bs0(bench, s)
        if v.get("noise"):
            smp["joints"] = {f: {sv: float(x) + float(rng.normal(0.0, NOISE_DEG)) for sv, x in js.items()}
                             for f, js in smp["joints"].items()}
        if v.get("quant"):
            smp["joints"] = {f: {sv: round(float(x) / STEP_DEG) * STEP_DEG for sv, x in js.items()}
                             for f, js in smp["joints"].items()}
        if v.get("delay"):
            hist.append(smp["joints"])
            smp["joints"] = hist[max(0, len(hist) - 1 - int(v["delay"]))]
        return smp
    prm = e.get("prm") or {}
    smooth = {k: v[k] for k in ("budget_joint_tau", "target_tau") if v.get(k)}

    class Bench(Bench0):
        def __init__(self, *a, **k):
            if prm:
                k["task_prm"] = {**(k.get("task_prm") or {}), **prm}
            k.update(smooth)
            super().__init__(*a, **k)
    HB.simulate = keep
    if v.get("quant") or v.get("delay") or v.get("noise"):
        HB.bench_sample = sample
    if prm or smooth:
        HB.HomBench = Bench
    try:
        model, seed = item.get("model", "pads"), int(item.get("seed", 0))
        job = {**SR.base_job(int(e["call"]), tuple(e["xy"]), model, contact=model), "k": int(e["k"]), "seed": seed,
               "budget": v["budget"]}
        r = SR._run(job)
    finally:
        HB.simulate, HB.bench_sample, HB.HomBench = sim0, bs0, Bench0
    st = got.get("stream") or []
    T = np.array([np.asarray(HB.sim_from_servo(s["targets"])[f], float) for s in st if s["t"] >= 2.0 - 1e-9
                  for f in HB.FINGERS], float).reshape(-1, 9)
    T = np.degrees(T)
    return {"plan": Path(e["plan"]).name, "xy": e["xy"], "k": e["k"], "prm": e.get("prm"), "variant": item["variant"],
            "model": item.get("model", "pads"), "seed": int(item.get("seed", 0)),
            "track": r.get("track"), "held": r.get("held"), "error": r.get("error"), "periods": len(T),
            "cmd": RM.metrics(T) if len(T) >= 20 else None,
            "budget_below_1_pct": round(100 * float(np.mean([(s.get("budget_scale") or 1.0) < 0.999 for s in st
                                                              if s["t"] >= 2.0 - 1e-9])), 1) if st else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["budget", "none", "budget_quant", "none_quant"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="every default and stage E plan of the plan list")
    ap.add_argument("--predict", action="store_true",
                    help="both models at seeds 0-2 per plan and variant; rows to PRED_ROWS (resumable), the means to "
                         "the twist ranking's data/20261120-smoothed_predictions.json")
    a = ap.parse_args()
    R = SR.load_replan()
    R.RG.gear_class = R.RC2.clutch_class
    R.RC2.STIFF, R.RC2.BREAKAWAY = True, 0.0
    R.RG.RF.RL.rows_file()
    SR._R["mod"] = R
    es = plans_all() if a.all else plans_run()
    if a.limit:
        es = es[:a.limit]
    if a.predict:
        return predict(es, a)
    items = [{"entry": e, "variant": v} for e in es for v in a.variants]
    print(f"{len(es)} plans x {len(a.variants)} variants", flush=True)
    import multiprocessing as mp
    old = json.loads(OUT.read_text())["runs"] if OUT.exists() else []
    keyf = (lambda r: (r["plan"], r["variant"]))
    have = {keyf(r): r for r in old}
    res = []
    with mp.get_context("fork").Pool(a.workers, initializer=SR._init) as pool:
        for r in pool.imap_unordered(run, items, chunksize=1):
            res.append(r)
            have[keyf(r)] = r
            c = r["cmd"] or {}
            print(f"  {r['plan'][:44]:44s} {r['variant']:20s} rev% {c.get('rev_pct')} hf {c.get('hf_deg')} "
                  f"track {r['track']} held {r['held']}", flush=True)
    OUT.write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "), "generated": time.strftime("%F %H:%M"),
                               "runs": sorted(have.values(), key=lambda r: (r["plan"], r["variant"]))}, indent=1) + "\n")
    for v in a.variants:
        rs = [r for r in have.values() if r["variant"] == v and r["cmd"]]
        if rs:
            print(f"{v:20s} {len(rs)} plans: rev% median {np.median([r['cmd']['rev_pct'] for r in rs]):.1f}, "
                  f"hf median {np.median([r['cmd']['hf_deg'] for r in rs]):.2f}, track mean "
                  f"{np.mean([r['track'] or 0 for r in rs]):.3f}, held {sum(bool(r['held']) for r in rs)}/{len(rs)}")
    print("wrote", OUT)


PRED_ROWS = ROOT / "logs/20261120-controller_rematch/rows_smoothed_pred.jsonl"
PRED_OUT = RANK / "data/20261120-smoothed_predictions.json"


def predict(es, a):
    """Predictions of the plans under the variants, written before any trial of a variant (the twist ranking's score:
    the mean over both models at seeds 0-2, 0 for a run that dropped the sphere)."""
    import multiprocessing as mp
    import os
    done = {}
    if PRED_ROWS.exists():
        for x in PRED_ROWS.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                done[(r["plan"], r["variant"], r["model"], r["seed"])] = r
    items = [{"entry": e, "variant": v, "model": m, "seed": s} for e in es for v in a.variants for m in SR.MODELS
             for s in (0, 1, 2) if (Path(e["plan"]).name, v, m, s) not in done]
    print(f"{len(items)} runs to go ({len(done)} done)", flush=True)
    PRED_ROWS.parent.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(a.workers, initializer=SR._init) as pool, PRED_ROWS.open("a") as fh:
        for n, r in enumerate(pool.imap_unordered(run, items, chunksize=1), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            done[(r["plan"], r["variant"], r["model"], r["seed"])] = r
            if n % 50 == 0 or n == len(items):
                print(f"  {n}/{len(items)} {time.strftime('%H:%M:%S')}", flush=True)
    plans = []
    for e in es:
        name = Path(e["plan"]).name
        rec = {"plan": name, "xy": e["xy"], "k": e["k"], "prm": e.get("prm"), "role": e.get("role"),
               "group": e.get("group"), "prediction_before": e.get("score"), "variants": {}}
        for v in a.variants:
            per = {}
            for m in SR.MODELS:
                rs = [done.get((name, v, m, s)) for s in (0, 1, 2)]
                if all(rs):
                    per[m] = float(np.mean([(r["track"] or 0.0) if r["held"] and not r.get("error") else 0.0
                                            for r in rs]))
            rec["variants"][v] = {"both": round(float(np.mean(list(per.values()))), 4) if len(per) == 2 else None,
                                  **{m: round(x, 4) for m, x in per.items()}}
        plans.append(rec)
    PRED_OUT.write_text(json.dumps({"what": predict.__doc__.replace("\n", " "), "written": time.strftime("%F %H:%M"),
                                    "variants": {v: VARIANTS[v] for v in a.variants}, "noise_deg": NOISE_DEG,
                                    "script": str(Path(__file__).relative_to(ROOT)), "plans": plans}, indent=1) + "\n")
    print("wrote", PRED_OUT)


if __name__ == "__main__":
    main()
