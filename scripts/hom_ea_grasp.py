#!/usr/bin/env python3
"""The evolutionary search of scripts/hom_ea.py with the grasp in the genome (owner, 2026-11-03: "can't we score and
refine a grasp based off how well the grasp (+morphology) executes the HOM control? from what i understand we are doing
pretty random sampling"). An individual is a layout of the symmetric tripods' 5 mm lattice and one grasp v on it (the
pose toy's variables, fp_sr2_toy.bounds). HOM scores that one grasp, where hom_ea.py scores the best grasp of the
layout's grasp set (up to 16).

The search is hom_ea.search's elitist, mutation-only EA: a generation keeps its best half, takes its best eighth as
parents and fills the other half with children. Generation 0 is hom_ea's (numpy generator 1000 x seed + 7: the same 64
layouts), each layout with its foundational pose (grasp 0 of its set). A child of the parent (layout, v) is made in one
of two ways, each with probability 0.5:
  layout move: the layout moves by U(-delta/2, delta/2) mm per coordinate, rounded to the lattice and clipped to the
      gantries' travel (hom_ea's children), and v is re-solved on the new layout by fp_sr2_toy.Problem.solve started
      from it (the new layout's foundational pose when the solve fails);
  grasp step: the layout is kept.
Then v steps by N(0, (0.25 x the layout's feasible widths)^2) until the pose toy's constraints and clearances pass
(hom_grasp_refine.draw_layout's rule: 40 draws, the last 20 at half the spread). A grasp step with no feasible draw is
left out of the generation; a layout move with none keeps its re-solved grasp. A child on a layout whose pose the toy
did not reach scores 0 at no rollout, as in hom_ea. The fitness is the grasp's task score (hom_landscape.rollout_score)
averaged over seeds 0-2: 3 rollouts a child, against up to 48 for a layout of hom_ea. A search stops after the
generation in which its rollouts reach the budget. Order and ties follow hom_ea: a stable sort by fitness, the
incumbent the first individual seen with the highest fitness.

The search seeds of one task run in lockstep: each generation's children of every seed are one GPU batch
(hom_gpu.batch_child through hom_grasp_refine.evaluate; about 288 worlds, where one seed's 96 would cost the same process
start and model build). The children's grasps are drawn by hom_grasp_refine.WORKERS processes.

    python3 scripts/hom_ea_grasp.py run --task twist [--plant sv25] [--seeds 0 1 2] [--pop 64] [--budget 6000]
    python3 scripts/hom_ea_grasp.py validate --task twist     # each search's incumbent, and hom_ea's, at seeds 20-24
    python3 scripts/hom_ea_grasp.py cpu --task twist          # the incumbents on the CPU plant at seed 20, timed
    python3 scripts/hom_ea_grasp.py report                    # rows/ea_grasp/<plant>/<date>-ea_grasp.json
    python3 scripts/hom_ea_grasp.py run ... --smoke           # rows under rows_smoke/ea_grasp/
Rows under docs/experiments/20261103-hom_score_refine/rows/ea_grasp/<plant>/: <task>_d<d>.jsonl, every rollout in
hom_gpu.py's format, the key with call, xy, pick (the individual's name s<seed>g<gen>c<child>), run (the search seed),
gen, child, op, parent, resolved and v; _timing.jsonl, one line per GPU batch; _iters.jsonl, one line per generation of
the lockstep (draw, open-pose and GPU-batch seconds, rollouts); <task>_d<d>_s<seed>_gens.jsonl, one line per generation
of a search (the population with its fitness, the incumbent, the rollouts so far), as hom_ea's gens files. val/: the
incumbents of both searches at seeds 20-24; cpu/: the CPU plant at seed 20. A stopped run resumes: the generations are
replayed with the same draws against the rows, and a replay that makes another child stops with an error.
hom_ea.py's own runs at the same budget (HOS_PLANT, HOS_SCENES, HOM_ROWS_TAG of the plant) write to its rows,
docs/experiments/20261029-foundational_pose_review/rows/ea_live<tag>/.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import json  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_grasp_refine as R  # noqa: E402

ROWS_ROOT = R.OUT / "rows" / "ea_grasp"          # --smoke: R.OUT / "rows_smoke" / "ea_grasp"
P_GRASP = 0.5                                    # a child's chance of keeping its parent's layout
STD = R.STD0["grasp"]                            # the grasp step's spread, x the layout's feasible width per variable
GENS = 400                                       # generation cap; the budget ends a search first
NEAR = 0.05                                      # a search has come near the grid's best within this score
LOOKUP_RUNS = 200                                # hom_ea's search replayed on the stored grid (hom_landscape.ea_search)
BUDGETS = (2000, 4000, 6000)


def rows_dir(plant):
    return ROWS_ROOT / plant


def stem(task):
    return f"{task}_d{R.TASKS[task]:g}"


def size_kw(task):
    return {"sph_d_mm": R.TASKS[task]} if R.POSE[task] == "tripod" else {"cyl_d_mm": R.TASKS[task]}


def pick_name(run, gen, child):
    return f"s{run}g{gen}c{child}"


def lattice_info(task):
    """{(x_sep, y_sep): (call, d6 (mm), the foundational pose v or None, the feasible widths or None)} over hom_ea's
    lattice, from the pose toy's grid rows: v and the widths of the grasp set where the toy reached the task's pose."""
    import fp_sr2_toy as FP
    import hom_ea as E
    toy = R.toy_rows(task)
    out = {}
    for xy, c in E.lattice().items():
        if c in toy:
            g0 = toy[c]["fp"][R.POSE[task]]["grasps"][0]
            out[xy] = (c, toy[c]["d"], g0["v"], g0["widths"])
        else:
            out[xy] = (c, FP.layout2(*xy).round(3).tolist(), None, None)
    return out


def make_child(job):
    """A child's grasp (a worker's job): the re-solve of a layout move, then the grasp step, numpy's generator seeded by
    (task, search seed, generation, child), so a replayed generation makes the same child. Returns (pick, v or None,
    resolved (None for a grasp step), draws tried, seconds)."""
    import fp_sr2_toy as FP
    task, pick, ident, op, d6, v_par, v_fp, widths = job
    FP.set_sizes(**size_kw(task))
    t0 = time.perf_counter()
    b = np.array(FP.bounds(R.POSE[task]), float)
    lo, hi = b[:, 0], b[:, 1]
    rng = np.random.default_rng([sum(map(ord, task + "ea_grasp")), *ident])
    resolved = None
    if op == "layout":
        if v_par is None:                           # a parent on an unreached layout: the new layout's pose
            base = np.asarray(v_fp, float)
        else:
            pr = FP.Problem(R.POSE[task], d6)
            ok, v, _nit, _clear, _sl = pr.solve(np.asarray(v_par, float))
            resolved = bool(ok) and R.feasible(task, d6, np.asarray(v, float))
            base = np.asarray(v, float) if resolved else np.asarray(v_fp, float)
    else:
        base = np.asarray(v_par, float)
    std = STD * np.maximum(np.asarray(widths, float), 1e-4 * (hi - lo))
    tried = 0
    for t in range(R.TRIES):
        s = std if t < R.TRIES // 2 else 0.5 * std
        y = np.clip(base + s * rng.standard_normal(len(base)), lo, hi)
        tried += 1
        if R.feasible(task, d6, y):
            return pick, [round(float(x), 7) for x in y], resolved, tried, time.perf_counter() - t0
    keep = [round(float(x), 7) for x in base] if op == "layout" else None
    return pick, keep, resolved, tried, time.perf_counter() - t0


def tail(path, pos):
    """The rows appended to path after byte offset pos, and the new offset (a partial last line waits)."""
    if not path.exists():
        return [], pos
    with open(path, "rb") as fh:
        fh.seek(pos)
        data = fh.read()
    end = data.rfind(b"\n") + 1
    return [json.loads(x) for x in data[:end].decode().splitlines() if x.strip()], pos + end


class Search:
    """One search seed's state: hom_ea.cmd_run's generator, the population (sorted by fitness after each generation),
    every individual evaluated in order, and the stop flag."""

    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(1000 * seed + 7)
        self.P, self.keep, self.seen, self.done = [], [], [], False


def born(lat, se, gen, child, xy, op, parent):
    """A new individual: its name, layout and parent; v and the job of its grasp follow."""
    c, d6, v_fp, widths = lat[xy]
    return {"pick": pick_name(se.seed, gen, child), "run": se.seed, "gen": gen, "child": child, "xy": list(xy),
            "call": c, "d": d6, "op": op, "parent": parent["pick"] if parent else None,
            "v_par": parent["v"] if parent else None, "v_fp": v_fp, "widths": widths, "v": None, "resolved": None,
            "tried": 0, "fit": 0.0, "n": 0, "out": None}


def cmd_run(a):
    R.setenv(a.plant)
    import multiprocessing as mp
    import fp_sr2_toy as FP
    import hom_ea as E
    import hom_gpu as HG
    import hom_landscape as L
    task = a.task
    FP.set_sizes(**size_kw(task))
    lat = lattice_info(task)
    keys = list(lat)
    out = rows_dir(a.plant)
    out.mkdir(parents=True, exist_ok=True)
    path, ipath = out / f"{stem(task)}.jsonl", out / f"{stem(task)}_iters.jsonl"
    gpath = {s: out / f"{stem(task)}_s{s}_gens.jsonl" for s in a.seeds}
    gdone = {s: {int(g["gen"]) for g in R.read_rows(gpath[s])} for s in a.seeds}
    S = {s: Search(s) for s in a.seeds}
    acc, vrow, pos = {}, {}, 0                  # per pick: {seed: score} and the v of its rows
    rows, pos = tail(path, pos)
    for r in rows:
        acc.setdefault(r["pick"], {})[int(r["seed"])] = L.rollout_score(task, r)
        vrow[r["pick"]] = r["v"]
    scene = HG.Scene("sphere" if R.POSE[task] == "tripod" else "cylinder", R.TASKS[task])
    ctx = mp.get_context("spawn")
    with ctx.Pool(R.WORKERS) as pool:
        for g in range(a.gens + 1):
            act = [S[s] for s in a.seeds if not S[s].done]
            if not act:
                break
            t0 = time.time()
            new = []
            for se in act:
                if g == 0:
                    idx = se.rng.choice(len(keys), a.pop, replace=False)
                    kids = [born(lat, se, 0, k, keys[i], "start", None) for k, i in enumerate(idx)]
                    for ind in kids:
                        ind["v"] = ind["v_fp"]
                else:
                    se.keep, parents = se.P[:a.pop // 2], se.P[:max(1, a.pop // 8)]
                    kids = []
                    for i in range(a.pop - len(se.keep)):
                        par = parents[i % len(parents)]
                        if se.rng.random() < P_GRASP:
                            kids.append(born(lat, se, g, i, tuple(par["xy"]), "grasp", par))
                        else:
                            m = np.asarray(par["xy"], float) + se.rng.uniform(-a.delta / 2, a.delta / 2, 2)
                            m = np.clip(E.STEP * np.round(m / E.STEP), E.LO, E.HI)
                            kids.append(born(lat, se, g, i, (float(m[0]), float(m[1])), "layout", par))
                new += kids
            jobs = []
            for ind in new:
                if ind["op"] == "start":
                    continue
                if ind["v_fp"] is None:
                    ind["out"] = "unreached"            # scores 0 at no rollout, as hom_ea's unreached layouts
                elif ind["op"] == "grasp" and ind["v_par"] is None:
                    ind["out"] = "unreached"
                else:
                    jobs.append((task, ind["pick"], (ind["run"], ind["gen"], ind["child"]), ind["op"], ind["d"],
                                 ind["v_par"], ind["v_fp"], ind["widths"]))
            res = {p: (v, rs, tr, sec) for p, v, rs, tr, sec in pool.map(make_child, jobs)}
            t_draw = time.time() - t0
            by = {ind["pick"]: ind for ind in new}
            for p, (v, rs, tr, _sec) in res.items():
                ind = by[p]
                ind["v"], ind["resolved"], ind["tried"] = v, rs, tr
                if v is None:
                    ind["out"] = "no_draw"
            for ind in new:                             # a replayed generation must make what its rows hold
                if ind["v"] is not None and ind["pick"] in vrow and \
                        np.abs(np.asarray(vrow[ind["pick"]]) - np.asarray(ind["v"])).max() > 1e-5:
                    raise SystemExit(f"replay of {task} {ind['pick']} made another grasp")
            items = []
            for ind in new:
                if ind["v"] is None:
                    continue
                key = {"d": ind["d"], "call": ind["call"], "xy": ind["xy"], "pick": ind["pick"], "run": ind["run"],
                       "gen": ind["gen"], "child": ind["child"], "op": ind["op"], "parent": ind["parent"],
                       "resolved": ind["resolved"], "v": ind["v"], "prm": {}}
                items.append((key, ind["d"], np.asarray(ind["v"], float), {}))
            tm = {}
            n_run = R.evaluate(task, a.plant, path, items, R.SEARCH_SEEDS, scene, HG, E, tm)
            rows, pos = tail(path, pos)
            for r in rows:
                acc.setdefault(r["pick"], {})[int(r["seed"])] = L.rollout_score(task, r)
                vrow[r["pick"]] = r["v"]
            for ind in new:
                if ind["v"] is None:
                    continue
                sc = acc.get(ind["pick"], {})
                if all(sd in sc for sd in R.SEARCH_SEEDS):
                    ind["fit"], ind["n"] = float(np.mean([sc[sd] for sd in R.SEARCH_SEEDS])), len(R.SEARCH_SEEDS)
                else:
                    ind["out"] = "open_pose"            # hom_grasp_refine.evaluate left it out
            for se in act:
                kids = [ind for ind in new if ind["run"] == se.seed]
                live = [ind for ind in kids if ind["out"] in (None, "unreached")]
                se.seen += live
                se.P = sorted(se.keep + live, key=lambda ind: -ind["fit"])
                inc = max(se.seen, key=lambda ind: ind["fit"])
                spent = sum(ind["n"] for ind in se.seen)
                ops = [ind["op"] for ind in kids]
                line = {"gen": g, "rollouts": spent, "individuals": len(se.seen),
                        "layouts": len({tuple(ind["xy"]) for ind in se.seen if ind["n"]}),
                        "incumbent": {k: inc[k] for k in ("pick", "xy", "call", "op", "fit", "v")},
                        "incumbent_fit": round(inc["fit"], 4),
                        "children": {"start": ops.count("start"), "grasp": ops.count("grasp"),
                                     "layout": ops.count("layout"),
                                     "resolved": sum(ind["resolved"] is True for ind in kids),
                                     "resolve_failed": sum(ind["resolved"] is False for ind in kids),
                                     "unreached": sum(ind["out"] == "unreached" for ind in kids),
                                     "no_draw": sum(ind["out"] == "no_draw" for ind in kids),
                                     "open_pose": sum(ind["out"] == "open_pose" for ind in kids),
                                     "draws": sum(ind["tried"] for ind in kids)},
                        "best_child": round(max([ind["fit"] for ind in live] or [0.0]), 4),
                        "pop": [[ind["xy"][0], ind["xy"][1], round(ind["fit"], 4), ind["pick"]] for ind in se.P]}
                if g not in gdone[se.seed]:
                    append_row(gpath[se.seed], line)
                print(f"{time.strftime('%H:%M:%S')} {task} ea_grasp s{se.seed} gen {g}: {spent} rollouts, "
                      f"{len(se.seen)} individuals, incumbent {inc['pick']} {tuple(inc['xy'])} {inc['fit']:.3f}",
                      flush=True)
                if spent >= a.budget or g == a.gens:
                    se.done = True
            t_all = time.time() - t0
            sec = [r_[3] for r_ in res.values()]
            append_row(ipath, {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "task": task, "plant": a.plant, "gen": g,
                             "searches": len(act), "children": len(new), "jobs": len(jobs),
                             "resolves": sum(1 for j in jobs if j[3] == "layout" and j[5] is not None),
                             "draws": sum(r_[2] for r_ in res.values()), "rollouts": n_run, "replay": n_run == 0,
                             "draw_s": round(t_draw, 2), "child_cpu_s": round(float(np.sum(sec)), 2),
                             "child_cpu_s_max": round(float(np.max(sec)), 2) if sec else 0.0, **tm,
                             "total_s": round(t_all, 2), "workers": R.WORKERS})
            print(f"{time.strftime('%H:%M:%S')} {task} ea_grasp gen {g}: {len(new)} children, {n_run} rollouts run, "
                  f"draw {t_draw:.0f} s, all {t_all:.0f} s", flush=True)


def append_row(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ------------------------------------------------------------------------------------------------ the incumbents

def incumbents(task, plant, seeds):
    """[(method, search seed, {"call", "xy", "d", "v", "grasp", "fit", "rollouts", "gens"})] of both searches: the last
    generation's incumbent of each; for hom_ea the best grasp of its layout's set at seeds 0-2 (as its fitness)."""
    import hom_ea as E
    toy = R.toy_rows(task)
    out = []
    for s in seeds:
        g = R.read_rows(rows_dir(plant) / f"{stem(task)}_s{s}_gens.jsonl")
        if g:
            inc = g[-1]["incumbent"]
            out.append(("ea_grasp", s, {"call": inc["call"], "xy": inc["xy"], "d": toy[inc["call"]]["d"],
                                        "v": inc["v"], "grasp": None, "pick": inc["pick"], "op": inc["op"],
                                        "fit": inc["fit"], "rollouts": g[-1]["rollouts"], "gens": len(g)}))
    lat = E.lattice()
    for s in seeds:
        gp = E.OUT / f"{stem(task)}_s{s}_gens.jsonl"
        g = R.read_rows(gp)
        if not g:
            continue
        xy = tuple(g[-1]["incumbent"])
        c = lat[xy]
        j, f = R.best_of_set(task, E.OUT / f"{stem(task)}_s{s}.jsonl")[c]
        out.append(("ea", s, {"call": c, "xy": list(xy), "d": toy[c]["d"],
                              "v": toy[c]["fp"][R.POSE[task]]["grasps"][j]["v"], "grasp": j, "pick": None, "op": None,
                              "fit": f, "rollouts": g[-1]["rollouts"], "gens": len(g)}))
    return out


def cmd_validate(a):
    """Each search's incumbent, of this script and of hom_ea, at the validation seeds; rows val/<task>_d<d>.jsonl."""
    R.setenv(a.plant)
    import fp_sr2_toy as FP
    import hom_ea as E
    import hom_gpu as HG
    task = a.task
    FP.set_sizes(**size_kw(task))
    items = []
    for method, s, inc in incumbents(task, a.plant, a.seeds):
        key = {"d": inc["d"], "call": inc["call"], "xy": inc["xy"], "pick": f"{method}_s{s}", "method": method,
               "run": s, "grasp": inc["grasp"], "ea_pick": inc["pick"], "v": [round(float(x), 7) for x in inc["v"]],
               "prm": {}}
        items.append((key, inc["d"], np.asarray(inc["v"], float), {}))
    path = rows_dir(a.plant) / "val" / f"{stem(task)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = HG.Scene("sphere" if R.POSE[task] == "tripod" else "cylinder", R.TASKS[task])
    tm = {}
    n = R.evaluate(task, a.plant, path, items, R.VAL_SEEDS, scene, HG, E, tm)
    print(f"{task}: {len(items)} incumbents, {n} rollouts run ({tm})", flush=True)


def cmd_cpu(a):
    """The incumbents of both searches on the CPU plant with every contact (hom_compute's 'plant' variant: the scene's
    solver, no contact filter) at seed a.seed, each rollout timed on one core; rows cpu/cpu_check.jsonl beside the GPU's
    score of the same seed (the validation rows). Run through resguard, one BLAS thread, no GPU."""
    from dataclasses import replace
    R.setenv(a.plant)
    import mujoco
    import fp_sr2_toy as FP
    import hom_compute as HC
    import hom_gpu as HG
    import hom_landscape as L
    T, H, S_ = HG.T, HG.H, HG.S
    task = a.task
    FP.set_sizes(**size_kw(task))
    scene = HG.Scene("sphere" if R.POSE[task] == "tripod" else "cylinder", R.TASKS[task])
    var = "hold" if task in ("pinpush", "press") else "base"
    over, noise = T.VARIANTS[var]
    gpu = {(r["pick"], int(r["seed"])): L.rollout_score(task, r)
           for r in R.read_rows(rows_dir(a.plant) / "val" / f"{stem(task)}.jsonl")}
    out = rows_dir(a.plant) / "cpu" / "cpu_check.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done ={(r["task"], r["pick"], int(r["seed"])) for r in R.read_rows(out)}
    for method, s, inc in incumbents(task, a.plant, a.seeds):
        pk = f"{method}_s{s}"
        if (task, pk, a.seed) in done:
            continue
        t0 = time.perf_counter()
        setup = R.setup_of(HG, task, inc["d"], np.asarray(inc["v"], float))
        bm, q_open = scene.prepare(HG.mount_xy(inc["d"]), setup)
        plant = T.Plant(scene.pads_path, setup)
        m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
        for m in (plant.m, m_pt):
            for f in HG.FINGERS:
                m.body_pos[m.body(f"{f}_mount").id] = bm[f]
        S_.set_palm(m_pt, setup["palm"])
        fs, os_ = T.sr2_specs(setup)
        mirror = H.Mirror(m_pt, fs, os_)
        sch = T.Schedule(task, setup, T.PRM)
        t_setup = time.perf_counter() - t0
        ld = HC.load1()
        c0, t1 = time.process_time(), time.perf_counter()
        res, _tr, _ = T.rollout(task, setup, plant, mirror, q_open, a.seed, T.PRM,
                                replace(H.HomParams(dt=T.DT_C), **over), None, noise,
                                T.PLANT_VARIANTS.get(var, {}).get("hold_s", 0.0))
        wall, cpu = time.perf_counter() - t1, time.process_time() - c0
        row = {"task": task, "plant": a.plant, "hos_plant": R.PLANTS[a.plant][0], "variant": "plant", "pick": pk,
               "method": method, "run": s, "call": inc["call"], "xy": inc["xy"], "grasp": inc["grasp"], "seed": a.seed,
               "wall_s": round(wall, 3), "cpu_s": round(cpu, 3), "setup_s": round(t_setup, 3),
               "sim_s": round(float(sch.t_end), 3), "score": round(L.rollout_score(task, res), 4),
               "held": bool(res.get("held")), "gpu_score_seed": gpu.get((pk, a.seed)), "load_before": ld,
               "load_after": HC.load1(), "host": socket.gethostname(), "when": time.strftime("%Y-%m-%d %H:%M:%S"),
               "threads": {k: os.environ.get(k) for k in HC.THREADS_1}}
        append_row(out, row)
        print(f"{task} {pk} call {inc['call']}: {wall:.2f} s wall for {sch.t_end:.1f} s simulated, score "
              f"{row['score']:.2f} (GPU {gpu.get((pk, a.seed))})", flush=True)


# ------------------------------------------------------------------------------------------------ the report

def grid_best(task, plant):
    """The plant's grid at the task's default object: the best layout's best-of-set score at seeds 0-2, its layouts
    (ties), the grid's rollouts and the layouts within NEAR of the best."""
    s1 = R.s1_table(task, plant)
    best = max(f for _j, f in s1.values())
    p = R.GRID_ROWS / f"grid2dff_v4b{R.PLANTS[plant][2]}" / f"{stem(task)}.jsonl"
    n = sum(1 for r in R.read_rows(p) if int(r["seed"]) in R.SEARCH_SEEDS)
    return {"best": round(best, 4), "best_calls": sorted(c for c, (_j, f) in s1.items() if f >= best - 1e-12),
            "rollouts": n, "layouts_run": len(s1), "layouts_near": sum(f >= best - NEAR for _j, f in s1.values())}


def curve_grasp(task, plant, s):
    """[(rollouts so far, incumbent fitness)] after each child of search s, from the rows in their order (a child's
    three rows lie together)."""
    import hom_landscape as L
    acc, order = {}, []
    for r in R.read_rows(rows_dir(plant) / f"{stem(task)}.jsonl"):
        if int(r["run"]) != s or int(r["seed"]) not in R.SEARCH_SEEDS:
            continue
        if r["pick"] not in acc:
            order.append(r["pick"])
        acc.setdefault(r["pick"], []).append(L.rollout_score(task, r))
    out, cost, inc = [], 0, -1.0
    for p in order:
        if len(acc[p]) < len(R.SEARCH_SEEDS):
            continue
        cost += len(acc[p])
        inc = max(inc, float(np.mean(acc[p])))
        out.append((cost, inc))
    return out


def curve_layout(task, s):
    """[(rollouts so far, incumbent fitness)] after each layout of hom_ea's search s, from its rows in their order."""
    import hom_ea as E
    import hom_landscape as L
    acc, order = {}, []
    for r in R.read_rows(E.OUT / f"{stem(task)}_s{s}.jsonl"):
        c = int(r["call"])
        if c not in acc:
            order.append(c)
        acc.setdefault(c, {}).setdefault(int(r.get("grasp", 0)), []).append(L.rollout_score(task, r))
    out, cost, inc = [], 0, -1.0
    for c in order:
        cost += sum(len(v) for v in acc[c].values())
        inc = max(inc, max(float(np.mean(v)) for v in acc[c].values()))
        out.append((cost, inc))
    return out


def first_within(curve, target):
    return next((n for n, f in curve if f >= target - 1e-12), None)


def lookup_ea(task, plant, best):
    """hom_ea's search replayed LOOKUP_RUNS times on the plant's stored grid (hom_landscape.ea_search, population 64, the
    fitness of every layout looked up): the share of runs within NEAR of the grid's best by each of BUDGETS rollouts,
    the median rollouts to get there and the median curve of the incumbent's score."""
    import hom_landscape as L
    L.GRID_SRC, L.GRID_ROWS = "grid2dff", R.GRID_ROWS / f"grid2dff_v4b{R.PLANTS[plant][2]}"
    look, _basin, _top, _b, total, xs, ys = L.ea_landscape(task, R.TASKS[task])
    hits, curves = [], []
    for run in range(LOOKUP_RUNS):
        rng = np.random.default_rng(1000 * run + 7)
        hist, _seen = L.ea_search(look["best"], rng, xs, ys, pop=64, budget=total)
        inc, cv = None, []
        for cost, k in hist:
            if inc is None or look["best"][k][0] > look["best"][inc][0]:
                inc = k
            cv.append((cost, look["best"][inc][0]))
        curves.append(cv)
        hits.append(first_within(cv, best - NEAR))
    pts = np.linspace(0, 8000, 81)
    med = [float(np.median([max([f for c, f in cv if c <= x] or [0.0]) for cv in curves])) for x in pts]
    h = np.array([x if x is not None else np.inf for x in hits], float)
    return {"runs": LOOKUP_RUNS, "grid_rollouts": int(total),
            "share_within_by": {str(b): round(float(np.mean(h <= b)), 3) for b in BUDGETS},
            "median_to_within": float(np.median(h)) if np.isfinite(np.median(h)) else None,
            "curve": {"rollouts": pts.round(0).tolist(), "median_best": [round(x, 4) for x in med]}}


def cmd_report(a):
    """rows/ea_grasp/<plant>/<date>-ea_grasp.json: per task the grid's best (seeds 0-2) and its best layout's S1 grasp at
    seeds 20-24; hom_ea replayed on the stored grid; per search of each method its incumbent curve, the rollouts to come
    within NEAR of the grid's best, the incumbent with its validated score (seeds 20-24) and CPU check, its
    generations and rollouts; the medians over searches; the compute (timing lines and iteration records)."""
    R.setenv(a.plant)
    import hom_ea as E
    import hom_landscape as L
    out = {"made": time.strftime("%Y-%m-%d %H:%M"), "plant": a.plant, "hos_plant": R.PLANTS[a.plant][0],
           "scenes": R.PLANTS[a.plant][1], "pop": 64, "budget": 6000, "p_grasp": P_GRASP, "std": STD,
           "tries": R.TRIES, "near": NEAR, "search_seeds": R.SEARCH_SEEDS, "val_seeds": R.VAL_SEEDS,
           "seeds": a.seeds, "tasks": {}}
    lev = sorted((R.OUT / "rows" / a.plant).glob("*-score_levels.json"))
    lev = json.loads(lev[-1].read_text()) if lev else None
    cpu = R.read_rows(rows_dir(a.plant) / "cpu" / "cpu_check.jsonl")
    for task in R.TASKS:
        gp = rows_dir(a.plant) / f"{stem(task)}.jsonl"
        if not gp.exists():
            continue
        gb = grid_best(task, a.plant)
        t = {"grid": gb, "lookup_ea": lookup_ea(task, a.plant, gb["best"]), "runs": {}}
        val = {}
        for r in R.read_rows(rows_dir(a.plant) / "val" / f"{stem(task)}.jsonl"):
            val.setdefault(r["pick"], []).append(r)
        ref = {}                                         # the grid's best layout's S1 grasp at seeds 20-24
        for r in R.read_rows(R.rows_dir(a.plant) / "val" / f"{stem(task)}.jsonl"):
            if r.get("pick") == "S1" and int(r["call"]) in gb["best_calls"]:
                ref.setdefault(int(r["call"]), []).append(L.rollout_score(task, r))
        t["grid"]["best_val"] = {str(c): round(float(np.mean(v)), 4) for c, v in ref.items()}
        if lev and task in lev.get("tasks", {}):
            lays = lev["tasks"][task]["layouts"]
            for k in ("s1_val", "s2e_val", "s3e_val"):
                ok = [e for e in lays if e.get(k) is not None]
                if ok:
                    e = max(ok, key=lambda e: (e[k], -e["call"]))
                    t[f"sample_best_{k}"] = {"call": e["call"], "xy": e["xy"], "score": round(e[k], 4)}
        for method, s, inc in incumbents(task, a.plant, a.seeds):
            cv = curve_grasp(task, a.plant, s) if method == "ea_grasp" else curve_layout(task, s)
            vr = val.get(f"{method}_s{s}", [])
            sc = [L.rollout_score(task, r) for r in vr]
            cc = [r for r in cpu if r["task"] == task and r["pick"] == f"{method}_s{s}"]
            gens = R.read_rows((rows_dir(a.plant) if method == "ea_grasp" else E.OUT) / f"{stem(task)}_s{s}_gens.jsonl")
            rec = {"incumbent": inc, "rollouts": cv[-1][0] if cv else 0, "gens": len(gens),
                   "to_within": first_within(cv, gb["best"] - NEAR), "final_fit": round(inc["fit"], 4),
                   "val": round(float(np.mean(sc)), 4) if sc else None, "val_n": len(sc),
                   "val_held": sum(bool(r.get("held")) for r in vr),
                   "cpu": [{k: c[k] for k in ("seed", "score", "held", "gpu_score_seed", "wall_s")} for c in cc],
                   "curve": [[n, round(f, 4)] for n, f in cv]}
            if method == "ea_grasp":
                ch = [g_["children"] for g_ in gens]
                rec["children"] = {k: int(sum(c.get(k, 0) for c in ch)) for k in
                                   ("grasp", "layout", "resolved", "resolve_failed", "unreached", "no_draw",
                                    "open_pose", "draws")}
            rec["layouts_run"] = gens[-1]["layouts"] if gens else 0
            t["runs"][f"{method}_s{s}"] = rec
        for method in ("ea_grasp", "ea"):
            rs = [v for k, v in t["runs"].items() if k.startswith(method + "_s")]
            if rs:
                t[f"{method}_median"] = {
                    "final_fit": round(float(np.median([r["final_fit"] for r in rs])), 4),
                    "val": round(float(np.median([r["val"] for r in rs if r["val"] is not None] or [np.nan])), 4),
                    "to_within": [r["to_within"] for r in rs], "rollouts": [r["rollouts"] for r in rs]}
        out["tasks"][task] = t
        print(task, {k: v for k, v in t.items() if k.endswith("_median") or k == "grid"}, flush=True)
    out["compute"] = compute(a.plant)
    out["queue_steps"] = queue_steps()
    p = rows_dir(a.plant) / f"{time.strftime('%Y%m%d')}-ea_grasp.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p.relative_to(ROOT))


LOGS = ROOT / "logs/20261104-hom_grasp_genome_ea"     # queue.sh: one Q_<step>.log per step, /usr/bin/time's last line


def queue_steps():
    """{step: {"wall_s", "peak_rss_gb", "done"}} of the queue's steps (logs/20261104-hom_grasp_genome_ea/Q_*.log)."""
    import re
    out = {}
    for q in sorted(LOGS.glob("Q_*.log")):
        m = re.search(r"peak (\d+) kB, ([\d.]+) s", q.read_text(errors="replace"))
        out[q.stem[2:]] = {"wall_s": float(m.group(2)) if m else None,
                           "peak_rss_gb": round(int(m.group(1)) / 1e6, 2) if m else None,
                           "done": (LOGS / f"{q.stem}.done").exists()}
    return out


def compute(plant):
    """The GPU batches (timing lines) and generations (iteration records) of both searches and the validation."""
    import hom_compute as HC
    import hom_ea as E
    out = {}
    for task in R.TASKS:
        t = {}
        for k, ls in (("ea_grasp", HC.batch_lines(rows_dir(plant) / f"{stem(task)}_timing.jsonl")),
                      ("ea", [x for s in (0, 1, 2) for x in HC.batch_lines(E.OUT / f"{stem(task)}_s{s}_timing.jsonl")]),
                      ("val", HC.batch_lines(rows_dir(plant) / "val" / f"{stem(task)}_timing.jsonl"))):
            if ls:
                t[k] = HC.summarize(ls)
        it = R.read_rows(rows_dir(plant) / f"{stem(task)}_iters.jsonl")
        if it:
            t["ea_grasp_iters"] = {"gens": len(it), "wall_s": round(sum(x["total_s"] for x in it), 1),
                                   "draw_s": round(sum(x["draw_s"] for x in it), 1),
                                   "child_cpu_s": round(sum(x["child_cpu_s"] for x in it), 1),
                                   "open_s": round(sum(x.get("open_s", 0.0) for x in it), 1),
                                   "gpu_s": round(sum(x.get("gpu_s", 0.0) for x in it), 1),
                                   "rollouts": int(sum(x["rollouts"] for x in it)),
                                   "batches": int(sum(x.get("batches", 0) for x in it))}
        out[task] = t
    return out


def main():
    global ROWS_ROOT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "validate", "cpu", "report"):
        p = sub.add_parser(name)
        p.add_argument("--plant", default="sv25", choices=list(R.PLANTS))
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        p.add_argument("--smoke", action="store_true", help="rows under rows_smoke/ea_grasp/")
        if name != "report":
            p.add_argument("--task", choices=list(R.TASKS), required=True)
        if name == "run":
            p.add_argument("--pop", type=int, default=64)
            p.add_argument("--delta", type=float, default=30.0, help="hom_landscape.EA_DELTA (mm)")
            p.add_argument("--budget", type=int, default=6000, help="rollouts after which no generation starts")
            p.add_argument("--gens", type=int, default=GENS)
        if name == "cpu":
            p.add_argument("--seed", type=int, default=R.VAL_SEEDS[0])
    a = ap.parse_args()
    if a.smoke:
        ROWS_ROOT = R.OUT / "rows_smoke" / "ea_grasp"
    return {"run": cmd_run, "validate": cmd_validate, "cpu": cmd_cpu, "report": cmd_report}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
