#!/usr/bin/env python3
"""Twist layout ranking on the calibrated bench simulation, for the test of the simulation's ranking on the hand
(hand-4a, 2026-11-19; owner 17:2x: re-run the task-space search for the sphere twist, deploy many layouts with their best
controls on the hand, and measure how well the real results match the simulation's).

The simulation is the one whose blind predictions of 2026-11-17 came nearest the hand's g0 twists at (70, 110): tracking
0.437 with point contact against the hand's mean of 0.447 over the night's tips-off trials
(docs/experiments/20261117-sr2_system_identification/QUEUE.md, "W3, A/B"). It is closed-loop HOM through
hom_bench.simulate on the CPU plant of sim/20261117-g13_open_pose_replan.py, whose run function is imported unchanged:
  servos   7.5 V: kp 0.825 N m/rad, damping 0.011, torque limit 0.27 N m; index and middle yaw gains 0.525 N m/rad
  gears    the load cannot backdrive a joint (rev_clutch2, forward friction 0.6 |tau|, held joints do not creep)
  stops    the MCP stops measured on 2026-11-17 (index -11.4, middle -10.0 deg)
  object   the 40 mm sphere on its stem and block, a table under the block until the lift; soft tangential contact
           (impratio 1) at friction 1.0, with point contact on each tip ("point") or the sphere pads ("pads")
  HOM      forces converted to servo offsets at the server's 0.55 N m/rad (HOS_PLANT), under the bench's torque budget
           0.56 N m (HOM's 5 V units), from the layout's own 8 mm open pose; contacts not given to HOM
A run's row is the imported function's: tracking (1 - RMS orientation error / RMS reference, 0 when the sphere drops),
the twist at the end, the rise, the drift, held, the closest index-middle approach of the joint states and of HOM's
targets, plus the stage, the model and the wall time.

Layouts: the symmetric tripods' 5 mm lattice (fp_sr2_toy grid2d, x_sep 40-160 mm, y_sep 50-170 mm, 625 layouts); the
482 with ground-clear tripod grasps (rows grid2dgr_tripod40_v4b.jsonl), each with its 16 grasps.

Stages. Rows go to logs/20261119-twist_layout_ranking/rows_<stage>.jsonl, appended and fsynced per run; a restart
skips the runs already written, so a stopped stage resumes where it stopped.
  A   every grasp at seed 0, point and pads                                          482 x 16 x 2 = 15,424 runs
  B   seeds 1 and 2 for each layout's three best grasps of stage A, per model         up to 5,784 runs
  C   fresh seeds 10-12 for the 60 best layouts' two best grasps (mean of seeds 0-2), per model
  D   other plants on the 150 best layouts' best two grasps (mean over both models), seeds 0-2: stiff pads (impratio
      100), friction 0.6 (the drags' range 0.31-0.78), 5 V (the servos before 11-17)
The heartbeat file logs/20261119-twist_layout_ranking/heartbeat.json is rewritten after every run (time, stage, runs
done and due, errors). A stage stops with exit code 2 and writes FAILED_<stage> when 60 of its last 100 runs ended in an
error (an import or scene failure would do that; a dropped sphere is a row, and a run that fails in HOM's least squares
after a drop is about 1 % of runs), and with exit code 4 when no run finished for 15 min (the pool is then terminated;
rerunning the stage resumes it).

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 OMP_NUM_THREADS=1 .venv/bin/python \\
        docs/experiments/20261119-twist_layout_ranking/sim/20261119-twist_ranking_search.py run --stage A [--workers 14]
    .venv/bin/python .../20261119-twist_ranking_search.py jobs --stage B      # count a stage's runs, run nothing
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HOS_PLANT = "kp0.55_kv0_fr0.18_fl0_dp0.011"           # hom_server's --plant measured: HOM's force-to-offset gain
if os.environ.get("HOS_PLANT", HOS_PLANT) != HOS_PLANT:
    sys.exit(f"HOS_PLANT={os.environ['HOS_PLANT']}: this study runs HOM at the server's {HOS_PLANT}")
os.environ["HOS_PLANT"] = HOS_PLANT
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
LOGS = ROOT / "logs/20261119-twist_layout_ranking"
FPR = ROOT / "docs/experiments/20261029-foundational_pose_review/rows"
REPLAN = ROOT / "docs/experiments/20261117-sr2_system_identification/sim/20261117-g13_open_pose_replan.py"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

MODELS = ("point", "pads")
SEEDS_B, SEEDS_C = (1, 2), (10, 11, 12)
TOP_B, TOP_C, GRASPS_C, TOP_D = 3, 60, 2, 150
STALL_S, FAIL_WINDOW, FAIL_LIMIT = 900.0, 100, 60
# stage D's plants: (name, contact model, volts, friction); "pads_stiff" is replan.SCENES' impratio-100 scene
PLANTS_D = (("pads_stiff", "pads_stiff", 7.5, 1.0), ("point_mu0.6", "point", 7.5, 0.6), ("pads_mu0.6", "pads", 7.5, 0.6),
            ("point_5v", "point", 5.0, 1.0))


def load_replan():
    spec = importlib.util.spec_from_file_location("replan_1117", REPLAN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["replan_1117"] = mod
    spec.loader.exec_module(mod)
    return mod


def layouts():
    """[(call, (x_sep, y_sep), number of ground-clear grasps)] of the lattice, the layouts with grasps only."""
    out = []
    for x in (FPR / "grid2dgr_tripod40_v4b.jsonl").read_text().splitlines():
        if not x.strip():
            continue
        r = json.loads(x)
        rec = r["fp"].get("tripod") or {}
        n = len(rec.get("grasps") or []) if rec.get("ok") else 0
        if n:
            out.append((int(r["call"]), (float(r["xy"][0]), float(r["xy"][1])), n))
    return out


def rows(stage):
    p = LOGS / f"rows_{stage}.jsonl"
    out = []
    if p.exists():
        for x in p.read_text().splitlines():
            if x.strip():
                try:
                    out.append(json.loads(x))
                except json.JSONDecodeError:                 # a line cut by a kill: skipped, its run is redone
                    pass
    return out


def key(r):
    return (r["model"], int(r["call"]), int(r["k"]), int(r["seed"]))


def base_job(call, xy, model, contact="pads", volts=7.5, mu=1.0):
    from sr2hand.robot import sr2_layouts as SL
    return {"cand": "screen", "q_deg": None, "volts": volts, "contacts": False, "contact": contact, "stops": True,
            "budget": 0.56, "call": call, "xy": list(xy), "d6": [float(v) for v in SL.sym6(*xy)], "src": "grid2dgr",
            "model": model, "mu": mu}


def score(r):
    return float(r.get("track") or 0.0) if "error" not in r else 0.0


def best_grasps(rs, model, n, seeds=None):
    """{call: [k, ...]}: per layout its n best grasps under one model by the mean score over the given seeds (all seeds
    present for that grasp when seeds is None)."""
    sc = defaultdict(lambda: defaultdict(list))
    for r in rs:
        if r["model"] == model and (seeds is None or int(r["seed"]) in seeds):
            sc[int(r["call"])][int(r["k"])].append(score(r))
    return {c: sorted(g, key=lambda k: -np.mean(g[k]))[:n] for c, g in sc.items()}


def ranked_grasps(rs, model, seeds, n):
    """{call: [k, ...]}: per layout its n best grasps under one model by the mean over the seeds, among the grasps run
    at every one of them."""
    sc = defaultdict(lambda: defaultdict(dict))
    for r in rs:
        if r["model"] == model and int(r["seed"]) in seeds:
            sc[int(r["call"])][int(r["k"])][int(r["seed"])] = score(r)
    return {c: sorted((k for k in g if len(g[k]) == len(seeds)), key=lambda k: -np.mean(list(g[k].values())))[:n]
            for c, g in sc.items()}


def layout_scores(rs, model, seeds=(0, 1, 2)):
    """{call: (best grasp, its mean score over the seeds)} over grasps run on every one of the seeds."""
    sc = defaultdict(lambda: defaultdict(dict))
    for r in rs:
        if r["model"] == model and int(r["seed"]) in seeds:
            sc[int(r["call"])][int(r["k"])][int(r["seed"])] = score(r)
    out = {}
    for c, g in sc.items():
        full = {k: np.mean(list(v.values())) for k, v in g.items() if len(v) == len(seeds)}
        if full:
            k = max(full, key=full.get)
            out[c] = (k, float(full[k]))
    return out


def jobs_for(stage):
    lays = {c: (xy, n) for c, xy, n in layouts()}
    if stage == "A":
        return [{**base_job(c, xy, m, contact=m), "k": k, "seed": 0}
                for m in MODELS for c, (xy, n) in sorted(lays.items()) for k in range(n)]
    ra = rows("A")
    if stage == "B":
        out = []
        for m in MODELS:
            for c, ks in sorted(best_grasps(ra, m, TOP_B).items()):
                out += [{**base_job(c, lays[c][0], m, contact=m), "k": k, "seed": s} for k in ks for s in SEEDS_B]
        return out
    rab = ra + rows("B")
    if stage == "C":
        out = []
        for m in MODELS:
            ls = layout_scores(rab, m)
            top = sorted(ls, key=lambda c: -ls[c][1])[:TOP_C]
            bg = ranked_grasps(rab, m, (0, 1, 2), GRASPS_C)
            for c in top:
                out += [{**base_job(c, lays[c][0], m, contact=m), "k": k, "seed": s} for k in bg.get(c, [])
                        for s in SEEDS_C]
        return out
    if stage == "D":
        both = defaultdict(list)
        for m in MODELS:
            for c, (k, v) in layout_scores(rab, m).items():
                both[c].append(v)
        top = sorted((c for c in both if len(both[c]) == len(MODELS)), key=lambda c: -np.mean(both[c]))[:TOP_D]
        mean_k = defaultdict(lambda: defaultdict(list))         # grasps by their mean over both models and seeds 0-2
        for r in rab:
            if int(r["call"]) in top and int(r["seed"]) in (0, 1, 2):
                mean_k[int(r["call"])][int(r["k"])].append(score(r))
        out = []
        for name, contact, volts, mu in PLANTS_D:
            for c in top:
                g = mean_k[c]
                ks = sorted((k for k in g if len(g[k]) >= 2 * 3), key=lambda k: -np.mean(g[k]))[:2]
                out += [{**base_job(c, lays[c][0], name, contact=contact, volts=volts, mu=mu), "k": k, "seed": s}
                        for k in ks for s in (0, 1, 2)]
        return out
    if stage == "P":           # stage D's plants on the proposed layouts' grasps (hand-4a, 11-19 21:2x), D's runs skipped
        prop = json.loads((STUDY / "data/20261119-trial_proposal.json").read_text())
        have = {key(r) for r in rows("D")}
        out = []
        for name, contact, volts, mu in PLANTS_D:
            for p in prop["pick"] + prop["anchors"]:
                c, k = int(p["call"]), int(p["k"])
                out += [j for j in ({**base_job(c, lays[c][0], name, contact=contact, volts=volts, mu=mu), "k": k,
                                     "seed": s} for s in (0, 1, 2)) if (name, c, k, j["seed"]) not in have]
        return out
    raise ValueError(stage)


_R = {}


def _init():
    R = _R["mod"]
    R.RG.init(str(R.RG.RF.SCENE))


def _run(job):
    """One run in a worker. The imported run builds its plant as RG._G['GP'](scene, state, mounts, plant_params, ...),
    with plant_params' friction fixed at 1.0; a run at another friction wraps that call and replaces it."""
    R = _R["mod"]
    mu = float(job.get("mu", 1.0))
    GP = R.RG._G["GP"]
    seen = {}

    def plant(scene, st, mxy, params, *a, **k):
        params = {**params, "mu": mu}
        seen["mu"] = params["mu"]
        return GP(scene, st, mxy, params, *a, **k)
    R.RG._G["GP"] = plant
    w0 = time.time()
    try:
        r = R._run1(job)
    except Exception as e:                                   # noqa: BLE001  HOM's least squares after a drop
        r = {**{k: job.get(k) for k in ("cand", "volts", "contacts", "contact", "stops", "seed", "budget", "src",
                                         "call", "k", "xy")}, "error": repr(e)[:200]}
    finally:
        R.RG._G["GP"] = GP
    r.pop("trace", None)
    return {**r, "model": job["model"], "mu": mu, "plant_mu": seen.get("mu"), "pid": os.getpid(),
            "t_end": round(time.time(), 1), "wall_total_s": round(time.time() - w0, 1)}


def heartbeat(stage, done, due, errors, note=""):
    """Rewrite heartbeat.json; a failed write is printed and never stops the run (at 17:53 on 11-19 a second process
    writing the same temporary file made os.replace fail and ended stage A)."""
    hb = LOGS / "heartbeat.json"
    tmp = hb.with_name(f"heartbeat.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps({"time": time.strftime("%F %T"), "stage": stage, "done": done, "due": due,
                                   "errors": errors, "note": note, "pid": os.getpid()}) + "\n")
        os.replace(tmp, hb)
    except OSError as e:
        print(f"heartbeat not written: {e!r}", flush=True)


def run(stage, workers, limit=0):
    import multiprocessing as mp
    R = load_replan()
    src = REPLAN.read_text()                                 # _run's friction wrapper assumes these two lines
    assert 'plant_params = {**(PLANT_75 if v75 else RR.SERVER), "mu": 1.0}' in src, "replan's plant line changed"
    assert "p = GP(str(scene), st, mxy, plant_params, 0.0, c=0.6)" in src, "replan's plant call changed"
    R.RG.gear_class = R.RC2.clutch_class
    R.RC2.STIFF, R.RC2.BREAKAWAY = True, 0.0
    R.RG.RF.RL.rows_file()
    _R["mod"] = R
    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / f"rows_{stage}.jsonl"
    done = {key(r) for r in rows(stage)}
    jobs = [j for j in jobs_for(stage) if (j["model"], j["call"], j["k"], j["seed"]) not in done]
    if limit:
        jobs = jobs[:limit]
    due = len(jobs)
    print(f"stage {stage}: {due} runs to do ({len(done)} written before), {workers} workers, HOS_PLANT "
          f"{os.environ['HOS_PLANT']}", flush=True)
    heartbeat(stage, 0, due, 0, "start")
    if not jobs:
        return 0
    recent, errors, k = [], 0, 0
    pool = mp.get_context("fork").Pool(workers, initializer=_init)
    try:
        it = pool.imap_unordered(_run, jobs, chunksize=1)
        with out.open("a") as fh:
            while True:
                try:
                    r = it.next(timeout=STALL_S)
                except StopIteration:
                    break
                except mp.TimeoutError:
                    heartbeat(stage, k, due, errors, f"no run finished for {STALL_S:.0f} s; pool terminated")
                    print(f"stage {stage}: no run finished for {STALL_S:.0f} s; terminating the pool", flush=True)
                    pool.terminate()
                    return 4
                r["stage"] = stage
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                k += 1
                bad = "error" in r
                errors += bad
                recent = (recent + [bad])[-FAIL_WINDOW:]
                heartbeat(stage, k, due, errors)
                if k % 200 == 0 or k == due:
                    print(f"  {stage} {k}/{due} ({errors} errors) {time.strftime('%H:%M:%S')}", flush=True)
                if len(recent) == FAIL_WINDOW and sum(recent) >= FAIL_LIMIT:
                    msg = f"{sum(recent)} of the last {FAIL_WINDOW} runs ended in an error; last: {r.get('error')}"
                    (LOGS / f"FAILED_{stage}").write_text(time.strftime("%F %T ") + msg + "\n")
                    heartbeat(stage, k, due, errors, "FAILED: " + msg)
                    print("FAILED: " + msg, flush=True)
                    pool.terminate()
                    return 2
        pool.close()
        pool.join()
    finally:
        pool.terminate()
    heartbeat(stage, k, due, errors, "done")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "jobs"))
    ap.add_argument("--stage", choices=("A", "B", "C", "D", "P"), required=True)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N runs due (a size and memory check)")
    a = ap.parse_args()
    if a.cmd == "jobs":
        js = jobs_for(a.stage)
        print(f"stage {a.stage}: {len(js)} runs; layouts {len({j['call'] for j in js})}")
        return 0
    return run(a.stage, a.workers, a.limit)


if __name__ == "__main__":
    sys.exit(main())
