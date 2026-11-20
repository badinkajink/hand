#!/usr/bin/env python3
"""Each layout judged by its best feasible twist: among its ground-clear grasps, the best whose simulated twist keeps the
fingers apart on the CAD collision spheres (hand-4a, 2026-11-20 00:0x; owner, 23:5x: "we should not be filtering bad
hands because of collisions from infeasible trajectories, we should be judging them by their feasible trajectories").

Pass 1 runs every grasp of each layout at seed 0 under point contact and the pads with the CAD check
(sim/20261119-cad_gap_screen.py's job: the ranking's run, every second period's joints on the cad_v3 collision spheres
with the tip pad, tip end and PIP balls). A grasp is feasible when its least finger gap under both models is at least
--clear mm and the sphere stayed in the grasp. Pass 2 runs each layout's best feasible grasp (by the mean of both models
at seed 0) at seeds 1 and 2. Rows go to logs/20261119-twist_layout_ranking/rows_FG.jsonl (fsynced, resumable); the
layouts named with --first run before the rest.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 CUDA_VISIBLE_DEVICES= .venv/bin/python \\
        docs/experiments/20261119-twist_layout_ranking/sim/20261120-feasible_grasps.py run --workers 12 [--first X,Y ...]
    ... summary     writes data/20261120-feasible_ranking.json
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
spec = importlib.util.spec_from_file_location("cad_gap_screen", HERE / "20261119-cad_gap_screen.py")
CS = importlib.util.module_from_spec(spec)
sys.modules["cad_gap_screen"] = CS
spec.loader.exec_module(CS)
SR = CS.SR
import numpy as np  # noqa: E402

ROWS = SR.LOGS / "rows_FG.jsonl"


def rows():
    out = []
    if ROWS.exists():
        for x in ROWS.read_text().splitlines():
            if x.strip():
                try:
                    out.append(json.loads(x))
                except json.JSONDecodeError:
                    pass
    return out


def job(item):
    if item.get("seed", 0) == 0:
        r = CS.job(item)
        return {k: r[k] for k in ("call", "xy", "k", "model", "track", "held", "least_mm")} | {
            "seed": 0, "gaps": {p: v["gap_mm"] for p, v in r["gaps"].items()}}
    r = SR._run({**SR.base_job(item["call"], tuple(item["xy"]), item["model"], contact=item["model"]), "k": item["k"],
                 "seed": item["seed"]})
    return {"call": item["call"], "xy": item["xy"], "k": item["k"], "model": item["model"], "seed": item["seed"],
            "track": r.get("track"), "held": r.get("held"), "error": r.get("error")}


def feasible(rr, clear):
    """{(call, k): mean seed-0 score over both models} of the grasps feasible under both models."""
    by = {}
    for r in rr:
        if r["seed"] == 0:
            by.setdefault((r["call"], r["k"]), {})[r["model"]] = r
    out = {}
    for key, m in by.items():
        if set(m) == set(SR.MODELS) and all(v["least_mm"] >= clear and v["held"] for v in m.values()):
            out[key] = float(np.mean([SR.score(v) for v in m.values()]))
    return out


def run_pass(items, workers):
    import multiprocessing as mp
    R = SR.load_replan()
    R.RG.gear_class = R.RC2.clutch_class
    R.RC2.STIFF, R.RC2.BREAKAWAY = True, 0.0
    R.RG.RF.RL.rows_file()
    SR._R["mod"] = R
    n = 0
    with mp.get_context("fork").Pool(workers, initializer=SR._init) as pool, ROWS.open("a") as fh:
        for r in pool.imap(job, items, chunksize=1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            n += 1
            if n % 100 == 0 or n == len(items):
                print(f"  {n}/{len(items)} {time.strftime('%H:%M:%S')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summary"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--clear", type=float, default=10.0)
    ap.add_argument("--first", nargs="*", default=[], help="layouts X,Y to run before the rest")
    ap.add_argument("--only-first", action="store_true")
    a = ap.parse_args()
    S = json.loads((STUDY / "data/20261119-twist_ranking_summary.json").read_text())
    fit = {(c["call"]) for c in S["candidates"] if c["fits"]}
    lays = [(c, xy, n) for c, xy, n in SR.layouts() if c in fit]
    first = {tuple(float(v) for v in s.split(",")) for s in a.first}
    lays.sort(key=lambda t: (tuple(t[1]) not in first, t[0]))
    if a.only_first:
        lays = [t for t in lays if tuple(t[1]) in first]
    if a.cmd == "run":
        done = {(r["call"], r["k"], r["model"], r["seed"]) for r in rows()}
        p1 = [{"call": c, "xy": list(xy), "k": k, "model": m, "seed": 0, "score": None, "rank_among_deployable": None}
              for c, xy, n in lays for k in range(n) for m in SR.MODELS if (c, k, m, 0) not in done]
        print(f"pass 1: {len(p1)} runs ({len(lays)} layouts)", flush=True)
        if p1:
            run_pass(p1, a.workers)
        rr = rows()
        fz = feasible(rr, a.clear)
        best = {}
        for (c, k), v in fz.items():
            if c not in best or v > best[c][1]:
                best[c] = (k, v)
        xy = {c: xy for c, xy, n in lays}
        done = {(r["call"], r["k"], r["model"], r["seed"]) for r in rr}
        p2 = [{"call": c, "xy": list(xy[c]), "k": k, "model": m, "seed": s} for c, (k, v) in best.items() if c in xy
              for m in SR.MODELS for s in (1, 2) if (c, k, m, s) not in done]
        print(f"pass 2: {len(p2)} runs", flush=True)
        if p2:
            run_pass(p2, a.workers)
    rr = rows()
    fz = feasible(rr, a.clear)
    sc = {}
    for r in rr:
        sc.setdefault((r["call"], r["k"]), {}).setdefault(r["model"], {})[r["seed"]] = SR.score(r)
    out = []
    xy = {c: xy for c, xy, n in SR.layouts()}
    cand = {c["call"]: c for c in S["candidates"]}
    for c in sorted({c for c, k in fz} | {r["call"] for r in rr}):
        ks = {k: v for (cc, k), v in fz.items() if cc == c}
        n_all = len({r["k"] for r in rr if r["call"] == c and r["seed"] == 0})
        rec = {"call": c, "xy": list(xy[c]), "grasps_run": n_all, "grasps_feasible": len(ks),
               "default_k": cand.get(c, {}).get("k"), "default_score": cand.get(c, {}).get("score")}
        if ks:
            k = max(ks, key=ks.get)
            s012 = [sc[(c, k)].get(m, {}).get(s) for m in SR.MODELS for s in (0, 1, 2)]
            rec.update({"k": k, "seed0": round(ks[k], 4),
                        "seeds012": round(float(np.mean(s012)), 4) if all(v is not None for v in s012) else None})
        else:
            rec.update({"k": None, "seed0": 0.0, "seeds012": 0.0})
        out.append(rec)
    path = STUDY / "data/20261120-feasible_ranking.json"
    path.write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "), "written": time.strftime("%F %H:%M"),
                                "clear_mm": a.clear, "layouts": out}, indent=1) + "\n")
    print(f"{len(out)} layouts, {sum(1 for o in out if o['k'] is not None)} with a feasible grasp -> {path}")
    for o in sorted(out, key=lambda o: -(o["seeds012"] or o["seed0"] or 0))[:15]:
        print(o)


if __name__ == "__main__":
    main()
