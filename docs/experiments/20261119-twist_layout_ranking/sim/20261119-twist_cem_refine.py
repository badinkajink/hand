#!/usr/bin/env python3
"""Stage F of the twist layout ranking: each proposed layout's grasp, HOM's commanded force and its contact modes refined
together by the cross-entropy method (CEM) on the calibrated bench simulation (hand-4a, 2026-11-19; owner 18:1x:
"shouldnt we also do the CEM refinement on the grasps and HOM? our analysis showed that it was quite helpful"; Sunyu's
suggestion, relayed: impedance force targets per control step, smoothed).

Two levels, as scripts/hom/hom_grasp_refine.py's level "full" (2026-11-03) on the GPU plant:
  full    the toy's grasp variables v (fp_sr2_toy.bounds), the tripod's force f_tripod (log, 1-5 N) and the mode
          variables u of Eq. (2)'s spin, roll and slide rows of the tripod (w = u^2)
  knots   the same, with the force a schedule: f_tripod at 1.6 s (the lift), 3.4 s (the twist ramp's start), 4.9 s and
          6.4 s (its middle and end), interpolated linearly and held before the first and after the last knot
          (hom_tasks.force_at, prm f_tripod_knots), each knot in log 1-5 N
A candidate grasp must pass the pose toy's constraints with the ground on (fp_sr2_toy.set_ground: the plant's spheres
above the plane through the sphere's lowest point less 1 mm, at the touch and the open pose), the constraints the
ground-clear grasp sets (rows grid2dgr) were drawn under, so the contacts stay near the equator; draws come from
hom_grasp_refine.draw_layout (infeasible draws drawn again, up to 40 tries). CEM per layout and level: 24 candidates per
iteration, the start (the layout's grasp k with HOM's defaults) as candidate 0 of iteration 0, the best quarter by score
as elites, mean and spread moved 0.8 of the way to the elites', hom_grasp_refine's floors on the spread, 6 iterations;
the pick is the best candidate seen. A candidate's score is its mean tracking over seeds 0-2 under both contact models
(the ranking's plant and controller, sim/20261119-twist_ranking_search.py). The picks then run at the fresh seeds 10-12
under both models and at friction 0.6 with point contact, beside the start.

Rows: logs/20261119-twist_layout_ranking/rows_F.jsonl (fsynced; a restart replays the CEM from them, since each
iteration's draws are seeded by the layout, the level and the iteration). Picks: data/20261119-cem_refine.json.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 OMP_NUM_THREADS=1 .venv/bin/python \\
        docs/experiments/20261119-twist_layout_ranking/sim/20261119-twist_cem_refine.py [--levels full knots] [--workers 8]
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
spec = importlib.util.spec_from_file_location("twist_search", HERE / "20261119-twist_ranking_search.py")
SR = importlib.util.module_from_spec(spec)
sys.modules["twist_search"] = SR
spec.loader.exec_module(SR)                     # sets HOS_PLANT and the thread variables before any hom import

import numpy as np  # noqa: E402

ROOT = SR.ROOT
POP, ITERS, ELITE, SMOOTH = 24, 6, 0.25, 0.8
KNOT_T = (1.6, 3.4, 4.9, 6.4)
SEEDS, FRESH = (0, 1, 2), (10, 11, 12)
OUT_ROWS = Path(os.environ.get("CEM_ROWS", str(SR.LOGS / "rows_F.jsonl")))      # CEM_ROWS: a test's own file


def refine_module():
    sys.path.insert(0, str(ROOT / "scripts/hom"))
    import hom_grasp_refine as RF
    return RF


class Space:
    """hom_grasp_refine.Space at level full, with (knots) the force replaced by KNOT_T's log forces."""

    def __init__(self, RF, widths, knots):
        self.base = RF.Space("twist", "full", widths)
        self.nv, self.knots = self.base.nv, knots
        b = self.base
        if not knots:
            self.lo, self.hi, self.width = b.lo, b.hi, b.width
            self.default_prm = b.default_prm
            return
        nf = len(b.forces)                                     # f_tripod: one log force
        lo_f, hi_f = b.lo[self.nv], b.hi[self.nv]
        k = len(KNOT_T)
        self.lo = np.r_[b.lo[:self.nv], np.full(k, lo_f), b.lo[self.nv + nf:]]
        self.hi = np.r_[b.hi[:self.nv], np.full(k, hi_f), b.hi[self.nv + nf:]]
        self.width = b.width
        dp = b.default_prm
        self.default_prm = np.r_[np.full(k, dp[0]), dp[nf:]]

    def start(self, v0):
        mean = np.r_[np.asarray(v0, float), self.default_prm]
        std = np.r_[0.15 * self.width, np.full(len(self.default_prm), 0.3)]
        return mean, std

    def floor(self):
        return np.r_[0.02 * self.width, np.full(len(self.default_prm), 0.03)]

    def prm(self, x):
        if not self.knots:
            return self.base.prm(x)
        k = len(KNOT_T)
        f = [round(float(np.exp(x[self.nv + i])), 4) for i in range(k)]
        out = {"f_tripod": f[0], "f_tripod_knots": [[t, fi] for t, fi in zip(KNOT_T, f)]}
        for i, name in enumerate(self.base.modes):
            out[name] = round(float(np.clip(x[self.nv + k + i], 0.0, 1.0) ** 2), 4)
        return out


def setup_from_v(v, d6):
    """hom_tasks setup, mounts and layout record of grasp v on layout d6 (mm), as sr2_hom_replay.toy_setup builds them
    from a row (its sphere branch)."""
    import fp_sr2_toy as FP
    import hand_object_scale_sim as S
    import hom_tasks as T
    FP.set_sizes(sph_d_mm=40.0)
    M = FP.mounts6(d6)
    v = np.asarray(v, float)
    obj, _tg, n, _pc, _ax = FP.geometry("tripod", v, M)
    c = np.asarray(obj["centre"], float)
    n_out = {f: np.asarray(n[f], float) for f in T.FINGERS}
    r = 40.0 / 2000.0
    setup = {"task": "twist", "shape": "sphere", "d_mm": 40.0, "r": r, "hl": 0.0,
             "palm": np.array([-c[0], -c[1], S.Z_OBJ - c[2]]), "quat": np.array(T.SPHERE_Q), "design": {},
             "targets": {f: c + (r + T.K.R) * n_out[f] for f in T.FINGERS}, "n_out": n_out, "mounts": M, "psi": 0.0}
    mxy = {f: np.asarray(M[f][:2], float) for f in T.FINGERS}
    return setup, mxy, {"rows": "cem", "call": -1, "grasp": -1, "d6_mm": [float(x) for x in d6]}


def _run(job):
    """The ranking's run with the job's grasp v and HOM prm in place of the row's grasp and HOM's defaults."""
    from sr2hand.control import hom_bench as HB
    R = SR._R["mod"]
    RP = R.RG.RF._G["RP"]
    orig_setup, orig_bench = RP.toy_setup, HB.HomBench
    prm = job["prm"]
    su = setup_from_v(job["v"], job["d6"])
    RP.toy_setup = lambda *a, **k: su

    class Bench(orig_bench):
        def __init__(self, *a, **k):
            k["task_prm"] = {**(k.get("task_prm") or {}), **prm}
            super().__init__(*a, **k)
    HB.HomBench = Bench
    try:
        r = SR._run(job)
    finally:
        RP.toy_setup, HB.HomBench = orig_setup, orig_bench
    return {**r, "level": job["level"], "it": job["it"], "cand": job["cand"], "x": job["x"], "v": job["v"],
            "prm": prm, "stage": "F"}


def rows():
    return [json.loads(x) for x in OUT_ROWS.read_text().splitlines() if x.strip()] if OUT_ROWS.exists() else []


def key(r):
    return (r["level"], int(r["call"]), int(r["it"]), int(r["cand"]), r["model"], int(r["seed"]))


def run_jobs(jobs, workers):
    import multiprocessing as mp
    done = {key(r) for r in rows()}
    todo = [j for j in jobs if (j["level"], j["call"], j["it"], j["cand"], j["model"], j["seed"]) not in done]
    if not todo:
        return
    with mp.get_context("fork").Pool(workers, initializer=SR._init) as pool, OUT_ROWS.open("a") as fh:
        it = pool.imap_unordered(_run, todo, chunksize=1)
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
            SR.heartbeat("F", k, len(todo), 0)


def cand_scores(rs, level, call, it):
    """{cand: (mean score over seeds 0-2 and both models, x, prm, v)} of one iteration, complete candidates only."""
    by = {}
    for r in rs:
        if r["level"] == level and int(r["call"]) == call and int(r["it"]) == it and int(r["seed"]) in SEEDS \
                and r["model"] in SR.MODELS:
            by.setdefault(int(r["cand"]), []).append(r)
    out = {}
    for c, rr in by.items():
        if len(rr) == len(SEEDS) * len(SR.MODELS):
            out[c] = (float(np.mean([SR.score(r) for r in rr])), rr[0]["x"], rr[0]["prm"], rr[0]["v"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["full", "knots"], choices=("full", "knots"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--proposal", default=str(STUDY / "data/20261119-trial_proposal.json"))
    ap.add_argument("--calls", type=int, nargs="*", default=None, help="only these layouts (a test)")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--pop", type=int, default=POP)
    a = ap.parse_args()
    R = SR.load_replan()
    R.RG.gear_class = R.RC2.clutch_class
    R.RC2.STIFF, R.RC2.BREAKAWAY = True, 0.0
    R.RG.RF.RL.rows_file()
    SR._R["mod"] = R
    RF = refine_module()
    import fp_sr2_toy as FP
    FP.set_ground(True)                                     # the ground-clear grasp sets' constraints
    prop = json.loads(Path(a.proposal).read_text())
    picks = prop["pick"] + prop["anchors"]
    if a.calls:
        picks = [p for p in picks if int(p["call"]) in a.calls] or \
                [{"call": c, "xy": list(dict((cc, xy) for cc, xy, n in SR.layouts())[c]), "k": 0} for c in a.calls]
    toy = {}
    for x in (SR.FPR / "grid2dgr_tripod40_v4b.jsonl").read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            toy[int(r["call"])] = r
    from sr2hand.robot import sr2_layouts as SL
    state = {}
    for level in a.levels:
        for p in picks:
            c, k = int(p["call"]), int(p["k"])
            rec = toy[c]["fp"]["tripod"]
            d6 = [float(v) for v in SL.sym6(*p["xy"])]
            sp = Space(RF, rec["grasps"][0]["widths"], level == "knots")
            mean, std = sp.start(rec["grasps"][k]["v"])
            state[(level, c)] = {"sp": sp, "mean": mean, "std": std, "d6": d6, "xy": p["xy"], "k": k, "best": None}
    for it in range(a.iters):
        jobs = []
        for (level, c), st in state.items():
            sp = st["sp"]
            _call, xs, tried = RF.draw_layout(("twist", "f" + level, c, it, st["d6"], st["mean"], st["std"], sp.lo,
                                               sp.hi, sp.nv, a.pop))
            st["xs"] = xs
            for ci, x in enumerate(xs):
                if x is None:
                    continue
                prm = sp.prm(x)
                for m in SR.MODELS:
                    for s in SEEDS:
                        jobs.append({**SR.base_job(c, tuple(st["xy"]), m, contact=m), "k": st["k"], "seed": s,
                                     "level": level, "it": it, "cand": ci, "x": [float(v) for v in x],
                                     "v": [float(v) for v in x[:sp.nv]], "prm": prm})
        print(f"iteration {it}: {len(jobs)} runs {time.strftime('%H:%M:%S')}", flush=True)
        run_jobs(jobs, a.workers)
        rs = rows()
        for (level, c), st in state.items():
            sc = cand_scores(rs, level, c, it)
            if not sc:
                continue
            order = sorted(sc, key=lambda ci: -sc[ci][0])
            for ci in order[:1]:
                if st["best"] is None or sc[ci][0] > st["best"][0]:
                    st["best"] = (sc[ci][0], sc[ci][1], sc[ci][2], sc[ci][3], it, ci)
            el = np.array([sc[ci][1] for ci in order[:max(2, int(round(ELITE * len(order))))]], float)
            sp = st["sp"]
            st["mean"] = (1 - SMOOTH) * st["mean"] + SMOOTH * el.mean(0)
            st["std"] = np.maximum((1 - SMOOTH) * st["std"] + SMOOTH * el.std(0), sp.floor())
            s0 = sc.get(0)[0] if it == 0 and 0 in sc else None
            print(f"  {level:5s} call {c} {tuple(st['xy'])}: iteration best {sc[order[0]][0]:.3f}, best so far "
                  f"{st['best'][0]:.3f}" + (f", start {s0:.3f}" if s0 is not None else ""), flush=True)
    # validation of the picks and the starts at fresh seeds and friction 0.6
    jobs = []
    for (level, c), st in state.items():
        if st["best"] is None:
            continue
        sc, x, prm, v, it_b, ci_b = st["best"]
        for m in SR.MODELS:
            for s in FRESH:
                jobs.append({**SR.base_job(c, tuple(st["xy"]), m, contact=m), "k": st["k"], "seed": s, "level": level,
                             "it": 900, "cand": 0, "x": x, "v": v, "prm": prm})
        for s in SEEDS:
            jobs.append({**SR.base_job(c, tuple(st["xy"]), "point_mu0.6", contact="point", mu=0.6), "k": st["k"],
                         "seed": s, "level": level, "it": 901, "cand": 0, "x": x, "v": v, "prm": prm})
    run_jobs(jobs, a.workers)
    rs = rows()
    out = []
    for (level, c), st in state.items():
        if st["best"] is None:
            continue
        sc, x, prm, v, it_b, ci_b = st["best"]
        start = cand_scores(rs, level, c, 0).get(0)
        fresh = [SR.score(r) for r in rs if r["level"] == level and int(r["call"]) == c and int(r["it"]) == 900]
        mu06 = [SR.score(r) for r in rs if r["level"] == level and int(r["call"]) == c and int(r["it"]) == 901]
        out.append({"level": level, "call": c, "xy": st["xy"], "k": st["k"], "start_seeds012": start[0] if start else None,
                    "best_seeds012": round(sc, 4), "best_seeds101112": round(float(np.mean(fresh)), 4) if fresh else None,
                    "best_mu06": round(float(np.mean(mu06)), 4) if mu06 else None, "best_it": it_b, "best_cand": ci_b,
                    "v": v, "prm": prm,
                    "lat_deg": round(math.degrees(float(v[6])), 2) if len(v) > 6 else None,
                    "lat_start_deg": round(math.degrees(float(toy[c]["fp"]["tripod"]["grasps"][st["k"]]["v"][6])), 2)})
        o = out[-1]
        print(f"{level:5s} call {c} {tuple(st['xy'])} g{st['k']}: start {o['start_seeds012']}, pick {o['best_seeds012']} "
              f"(fresh {o['best_seeds101112']}, mu 0.6 {o['best_mu06']}), latitude {o['lat_start_deg']} -> {o['lat_deg']}, "
              f"prm {prm}", flush=True)
    path = STUDY / "data/20261119-cem_refine.json"
    if a.calls:
        path = SR.LOGS / "cem_refine_test.json"
    path.write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "), "written": time.strftime("%F %H:%M"),
                                "pop": a.pop, "iters": a.iters, "knot_t": KNOT_T, "picks": out}, indent=1) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
