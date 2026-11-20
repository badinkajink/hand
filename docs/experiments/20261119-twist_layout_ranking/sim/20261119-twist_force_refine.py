#!/usr/bin/env python3
"""HOM's twist forces refined per layout on the calibrated bench simulation (stage E of the twist layout ranking;
hand-4a, 2026-11-19; owner 17:2x: "improve our controls as much as we feasibly can" and deploy "top-N controls").

On the hand the twist stalls where the index and middle yaws and the thumb MCP reach the torque budget's cap: the
grip's normal forces leave the yaw servos too little torque to carry the fingertips round the sphere
(docs/experiments/20261117-sr2_system_identification/QUEUE.md, "W3, why the twist tracks about 0.45"). The tripod's
commanded normal force f_tripod (hom_tasks.PRM, 2.5 N) and the weight of the spin rows in Eq. (2) (w_spin_tripod, 1.0)
set that balance. Here each proposed layout's grasp runs on a grid of the two:
  f_tripod        1.0, 1.25, 1.5, 2.0, 2.5, 3.0 N
  w_spin_tripod   0.25, 0.5, 1.0, 2.0
at seeds 0-2 under both contact models, on the plant and controller of sim/20261119-twist_ranking_search.py (HOM gets
the setting as the plan's prm, as hom_bench_world.bench_world merges it into hom_tasks.PRM on the bench). The setting
with the best mean over both models and seeds is then run at seeds 10-12, and it and the default at friction 0.6 with
point contact (the pads slid at 0.31-0.78 on the tape on the UR5; a lighter grip slips sooner). Rows go to
logs/20261119-twist_layout_ranking/rows_E.jsonl (fsynced, resumable) and the choices to data/20261119-force_refine.json.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 OMP_NUM_THREADS=1 .venv/bin/python \\
        docs/experiments/20261119-twist_layout_ranking/sim/20261119-twist_force_refine.py [--workers 8]
"""
import argparse
import importlib.util
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
spec = importlib.util.spec_from_file_location("twist_search", HERE / "20261119-twist_ranking_search.py")
SR = importlib.util.module_from_spec(spec)
sys.modules["twist_search"] = SR
spec.loader.exec_module(SR)                     # sets HOS_PLANT and the thread variables before any hom import

import numpy as np  # noqa: E402

F_TRIPOD = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
W_SPIN = (0.25, 0.5, 1.0, 2.0)
DEFAULT = {"f_tripod": 2.5, "w_spin_tripod": 1.0}
SEEDS, FRESH = (0, 1, 2), (10, 11, 12)


def prm_key(p):
    return json.dumps(p, sort_keys=True)


def _run(job):
    """The search driver's run with HomBench given the job's task_prm."""
    from sr2hand.control import hom_bench as HB
    orig = HB.HomBench
    prm = job["prm"]

    class Bench(orig):
        def __init__(self, *a, **k):
            k["task_prm"] = {**(k.get("task_prm") or {}), **prm}
            super().__init__(*a, **k)
    HB.HomBench = Bench
    try:
        r = SR._run(job)
    finally:
        HB.HomBench = orig
    return {**r, "prm": prm, "stage": "E"}


def jobs(picks, done, fresh=None):
    out = []
    for p in picks:
        c, k, xy = int(p["call"]), int(p["k"]), tuple(p["xy"])
        grid = [{"f_tripod": f, "w_spin_tripod": w} for f, w in itertools.product(F_TRIPOD, W_SPIN)]
        seeds = SEEDS
        if fresh is not None:
            if c not in fresh:
                continue
            grid, seeds = [fresh[c]], FRESH
        for m in SR.MODELS:
            for prm in grid:
                for s in seeds:
                    key = (m, c, k, s, prm_key(prm))
                    if key not in done:
                        out.append({**SR.base_job(c, xy, m, contact=m), "k": k, "seed": s, "prm": prm})
        if fresh is not None:                          # the choice and the default at friction 0.6, point contact
            for prm in (fresh[c], DEFAULT):
                for s in SEEDS:
                    if ("point_mu0.6", c, k, s, prm_key(prm)) not in done:
                        out.append({**SR.base_job(c, xy, "point_mu0.6", contact="point", mu=0.6), "k": k, "seed": s,
                                    "prm": prm})
    return out


def score(r):
    return SR.score(r)


def run(job_list, workers, out):
    import multiprocessing as mp
    if not job_list:
        return
    with mp.get_context("fork").Pool(workers, initializer=SR._init) as pool, out.open("a") as fh:
        it = pool.imap_unordered(_run, job_list, chunksize=1)
        k = 0
        while True:
            try:
                r = it.next(timeout=SR.STALL_S)
            except StopIteration:
                break
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            k += 1
            SR.heartbeat("E", k, len(job_list), 0)
            if k % 100 == 0:
                print(f"  E {k}/{len(job_list)} {time.strftime('%H:%M:%S')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--proposal", default=str(STUDY / "data/20261119-trial_proposal.json"))
    a = ap.parse_args()
    R = SR.load_replan()
    R.RG.gear_class = R.RC2.clutch_class
    R.RC2.STIFF, R.RC2.BREAKAWAY = True, 0.0
    R.RG.RF.RL.rows_file()
    SR._R["mod"] = R
    prop = json.loads(Path(a.proposal).read_text())
    picks = prop["pick"] + prop["anchors"]
    out = SR.LOGS / "rows_E.jsonl"

    def load():
        rs = [json.loads(x) for x in out.read_text().splitlines() if x.strip()] if out.exists() else []
        return rs, {(r["model"], int(r["call"]), int(r["k"]), int(r["seed"]), prm_key(r["prm"])) for r in rs}
    rs, done = load()
    js = jobs(picks, done)
    print(f"stage E grid: {len(js)} runs to do ({len(done)} written before)", flush=True)
    run(js, a.workers, out)
    rs, done = load()
    best = {}
    table = defaultdict(lambda: defaultdict(list))
    for r in rs:
        if int(r["seed"]) in SEEDS and r["model"] in SR.MODELS:
            table[int(r["call"])][prm_key(r["prm"])].append(score(r))
    for p in picks:
        c = int(p["call"])
        full = {pk: float(np.mean(v)) for pk, v in table[c].items() if len(v) == len(SEEDS) * len(SR.MODELS)}
        if full:
            pk = max(full, key=full.get)
            best[c] = json.loads(pk)
    js = jobs(picks, done, fresh=best)
    print(f"stage E fresh seeds: {len(js)} runs", flush=True)
    run(js, a.workers, out)
    rs, done = load()
    res = []
    default = prm_key(DEFAULT)

    def mean(c, pk, seeds, models):
        v = [score(r) for r in rs if int(r["call"]) == c and prm_key(r["prm"]) == pk and int(r["seed"]) in seeds
             and r["model"] in models]
        return round(float(np.mean(v)), 4) if v else None
    for p in picks:
        c = int(p["call"])
        if c not in best:
            continue
        pk = prm_key(best[c])
        res.append({"call": c, "xy": p["xy"], "k": int(p["k"]), "best_prm": best[c],
                    "best_seeds012": mean(c, pk, SEEDS, SR.MODELS),
                    "best_seeds101112": mean(c, pk, FRESH, SR.MODELS),
                    "default_seeds012": mean(c, default, SEEDS, SR.MODELS),
                    "best_mu06": mean(c, pk, SEEDS, ("point_mu0.6",)),
                    "default_mu06": mean(c, default, SEEDS, ("point_mu0.6",)),
                    "grid": {k2: round(float(np.mean(v)), 4) for k2, v in table[c].items()}})
        print(f"  call {c} {tuple(p['xy'])} g{p['k']}: default {res[-1]['default_seeds012']} best {best[c]} "
              f"{res[-1]['best_seeds012']} fresh {res[-1]['best_seeds101112']}; at mu 0.6 default "
              f"{res[-1]['default_mu06']} best {res[-1]['best_mu06']}", flush=True)
    (STUDY / "data/20261119-force_refine.json").write_text(json.dumps(
        {"what": __doc__.split("\n\n")[0].replace("\n", " "), "f_tripod": F_TRIPOD, "w_spin_tripod": W_SPIN,
         "written": time.strftime("%F %H:%M"), "layouts": res}, indent=1) + "\n")
    SR.heartbeat("E", len(rs), len(rs), 0, "done")


if __name__ == "__main__":
    main()
