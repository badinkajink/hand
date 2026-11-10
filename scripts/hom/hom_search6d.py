#!/usr/bin/env python3
"""Six search methods on the SR2 hand's six-coordinate layouts under HOM, run live on the GPU evaluator (post-reset step
4; the search methods page, 20261109-hom_search_methods.html, compared them by replay on the 2-D lattice's stored grids).

Space: the 8,192 Sobol layouts of fp_sr2_toy (rows sobol8k: the six mount coordinates and the standardized ones z in
[-0.5, 0.5]^6), each with the task's foundational pose at the toy's object (rows sobol8kff_<pose><d>_v4b, the plant's
finger clearance). A layout's fitness is that pose's mean score at seeds 0-2 (three rollouts), 0 at no rollout when the
pose is not reached. A search moves in z and evaluates the Sobol layout nearest a point; a layout costs its rollouts once
per search. Every search of a task shares one evaluation of each layout (a layout's score does not depend on the search
that visits it), and the searches advance in lockstep, so one round's new layouts run as full GPU batches. No layout is
evaluated that no search visits.
Methods, each with a budget of 6,000 rollouts and three random starts (seeds 0-2):
  ea64     the EA of Yang et al. (hom_landscape.ea_search's selection, population 64): each generation keeps its best
           half, takes its best eighth as parents and fills the other half with children, each coordinate of a parent
           moved by U(-delta/2, delta/2) with delta 0.25 (the 2-D EA's 30 mm of a 120 mm travel);
  pso16    particle swarm optimization with 16 particles and the global best (hom_search_compare's Clerc and Kennedy
           coefficients, velocities capped at a fifth of the box, a particle stopped at a wall);
  pso64r   64 particles on a ring;
  random   the Sobol layouts in a random order, 64 a round;
  sobol    the Sobol layouts in their draw order, a Sobol sequence, 64 a round (the same for every seed);
  rrt      the FP paper's RRT (fp_sr2_toy.rrt's rule, the 6-D run's step of 0.10): a uniform target, the tree's nearest
           layout, a step toward the target, and the Sobol layout nearest that point among those not yet in the tree
           (the Sobol layouts lie about 0.2 apart, more than the step), which joins the tree when its pose is reached;
           from the layout nearest fp_sr2_toy.SEED6, 32 a round.
Each search's best layout then runs at seeds 20-24, which no search used; a method is scored against the best layout any
method found, by those validated scores. validate --track-top N also reruns the N layouts of best tracking score at seeds
0-2 among every layout the searches ran (the searches' fitness is the earlier score).
Rows: docs/experiments/20261029-foundational_pose_review/rows/search6d<tag>/<task>_d<d>.jsonl (the shared rollouts,
hom_gpu.py's format), <task>_<method>_s<seed>_gens.jsonl (one line a round: rollouts, layouts, incumbent) and
<task>_d<d>_val.jsonl (the validation).

    HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25 \\
        .venv/bin/python scripts/hom/hom_search6d.py run --task twist [--methods ea64 pso16 ...] [--seeds 0 1 2]
    .venv/bin/python scripts/hom/hom_search6d.py validate --task twist [--track-top 5]
    .venv/bin/python scripts/hom/hom_search6d.py report
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_gpu as HG  # noqa: E402
import hom_landscape as L  # noqa: E402

T = HG.T
OUT = HG.ROWS / f"search6d{HG.ROWS_TAG}"
TASKS = {"twist": 40.0, "roll": 40.0, "press": 25.0}
VARIANT = {"twist": "base", "roll": "base", "press": "hold"}
BATCH = {"twist": 2048, "roll": 2048, "press": 1536}
SEEDS, VAL_SEEDS = (0, 1, 2), (20, 21, 22, 23, 24)
BUDGET = 6000
METHODS = ("ea64", "pso16", "pso64r", "random", "sobol", "rrt")
EA_DELTA = 0.25
CHUNK = {"random": 64, "sobol": 64, "rrt": 32}
RRT_STEP = 0.10
MAX_ROUNDS = 400


class Space:
    """The Sobol layouts of one task: standardized coordinates, a k-d tree, which reach the pose, and their poses."""

    def __init__(self, task):
        from scipy.spatial import cKDTree
        self.task, self.d = task, TASKS[task]
        self.dmm, self.Z = L.layouts()
        self.tree = cKDTree(self.Z)
        self.items = {int(key["call"]): (key, mxy, setup)
                      for key, mxy, setup in HG.layout_list(task, "sobol8kff_size", 0, self.d)}
        self.reach = np.zeros(len(self.Z), bool)
        self.reach[list(self.items)] = True

    def nearest(self, z):
        return int(self.tree.query(np.clip(z, -0.5, 0.5))[1])


class Search:
    """One search: propose() gives the layouts of its next round, update() takes the fitness of every layout run."""

    def __init__(self, method, seed, sp):
        self.method, self.seed, self.sp = method, seed, sp
        self.rng = np.random.default_rng([sum(map(ord, sp.task + method)), seed])
        self.seen, self.fit, self.log, self.done = [], {}, [], False
        n = len(sp.Z)
        if method == "ea64":
            self.P = list(self.rng.choice(n, 64, replace=False))
        elif method.startswith("pso"):
            self.n = 16 if method == "pso16" else 64
            self.X = self.rng.uniform(-0.5, 0.5, (self.n, 6))
            self.V = self.rng.uniform(-1.0, 1.0, (self.n, 6)) * 0.2
            self.Pb = self.Pf = None
        elif method == "random":
            self.order, self.k = list(self.rng.permutation(n)), 0
        elif method == "sobol":
            self.order, self.k = list(range(n)), 0
        elif method == "rrt":
            self.order, self.k = self.rrt_order(), 0

    def rrt_order(self):
        """The tree's layouts in the order they join it (no rollout: joining needs only the reached pose)."""
        import fp_sr2_toy as FP
        sp = self.sp
        z0 = FP.standard(np.asarray(FP.SEED6, float), FP.LO6, FP.HI6)
        i0 = sp.nearest(z0)
        tree, order, member = [sp.Z[i0]], [i0], {i0}
        for _ in range(200000):
            u = self.rng.uniform(-0.5, 0.5, 6)
            Za = np.array(tree)
            dist = np.linalg.norm(Za - u, axis=1)
            j = int(dist.argmin())
            z = u if dist[j] <= RRT_STEP else Za[j] + RRT_STEP * (u - Za[j]) / dist[j]
            near = sp.tree.query(np.clip(z, -0.5, 0.5), k=64)[1]
            i = next((int(c) for c in near if int(c) not in member), None)
            if i is not None and sp.reach[i]:
                member.add(i)
                tree.append(sp.Z[i])
                order.append(i)
            if len(order) >= min(int(sp.reach.sum()), BUDGET // 3 + CHUNK["rrt"]):
                break
        return order

    def spent(self):
        return 3 * len({i for i in self.seen if self.sp.reach[i]})

    def propose(self):
        if self.done:
            return []
        if self.method == "ea64":
            return list(self.P)
        if self.method.startswith("pso"):
            return [self.sp.nearest(x) for x in self.X]
        out = self.order[self.k:self.k + CHUNK[self.method]]
        self.k += len(out)
        if not out:
            self.done = True
        return out

    def update(self, idx, fit):
        """idx: the round's proposals; fit: {layout: fitness} of every layout evaluated so far."""
        self.seen += idx
        for i in idx:
            self.fit[i] = fit.get(i, 0.0)
        inc = max(dict.fromkeys(self.seen), key=lambda i: self.fit[i]) if self.seen else None
        self.log.append({"round": len(self.log), "rollouts": self.spent(), "layouts": len(set(self.seen)),
                         "incumbent": inc, "incumbent_fit": self.fit.get(inc, 0.0)})
        if self.spent() >= BUDGET:
            self.done = True
            return
        if self.method == "ea64":
            P = sorted(self.P, key=lambda i: -self.fit[i])
            keep, parents = P[:32], P[:8]
            kids = []
            for k in range(32):
                z = self.sp.Z[parents[k % len(parents)]] + self.rng.uniform(-EA_DELTA / 2, EA_DELTA / 2, 6)
                kids.append(self.sp.nearest(z))
            self.P = keep + kids
        elif self.method.startswith("pso"):
            import hom_search_compare as SC
            f = np.array([self.fit[i] for i in idx])
            if self.Pb is None:
                self.Pb, self.Pf = self.X.copy(), f.copy()
            else:
                up = f > self.Pf
                self.Pb[up], self.Pf[up] = self.X[up], f[up]
            if self.method == "pso16":
                G = np.repeat(self.Pb[[int(np.argmax(self.Pf))]], self.n, axis=0)
            else:
                nb = np.stack([np.roll(np.arange(self.n), s) for s in (-1, 0, 1)], 1)
                G = self.Pb[nb[np.arange(self.n), np.argmax(self.Pf[nb], axis=1)]]
            r1, r2 = self.rng.random((self.n, 6)), self.rng.random((self.n, 6))
            self.V = np.clip(SC.W * self.V + SC.C1 * r1 * (self.Pb - self.X) + SC.C2 * r2 * (G - self.X), -0.2, 0.2)
            self.X = self.X + self.V
            wall = (self.X < -0.5) | (self.X > 0.5)
            self.X, self.V = np.clip(self.X, -0.5, 0.5), np.where(wall, 0.0, self.V)


def fitness(task, path, seeds=SEEDS, score="rollout"):
    """{layout: mean score over the seeds} from the shared rows (the earlier score, as the searches of the search methods
    page; the tracking score, score="track", is reported beside it)."""
    f = L.track_score if score == "track" else L.rollout_score
    acc = {}
    for x in (path.read_text().splitlines() if path.exists() else []):
        if x.strip():
            r = json.loads(x)
            if int(r["seed"]) in seeds:
                acc.setdefault(int(r["call"]), []).append(f(task, r))
    return {c: float(np.mean(v)) for c, v in acc.items() if len(v) >= len(seeds)}


def evaluate(sp, idx, path, seeds):
    """Run every reached layout of idx at the seeds it lacks in the rows, as GPU batches in spawned processes."""
    import multiprocessing as mp
    have = {}
    for x in (path.read_text().splitlines() if path.exists() else []):
        if x.strip():
            r = json.loads(x)
            have.setdefault(int(r["call"]), set()).add(int(r["seed"]))
    scene = HG.Scene(T.SHAPE[sp.task], sp.d)
    worlds = []
    for i in dict.fromkeys(idx):
        if i not in sp.items:
            continue
        todo = [s for s in seeds if s not in have.get(i, set())]
        if not todo:
            continue
        key, mxy, setup = sp.items[i]
        bm, q_open = scene.prepare(mxy, setup)
        worlds += [{"key": key, "seed": s, "setup": setup, "body_mounts": bm, "q_open": q_open,
                    "var": HG.variant_world(VARIANT[sp.task], s)} for s in todo]
    if not worlds:
        return 0
    ctx = mp.get_context("spawn")
    tpath = path.with_name(path.stem + "_timing.jsonl")
    n0 = sum(1 for x in (tpath.read_text().splitlines() if tpath.exists() else []) if x.strip())
    b = BATCH[sp.task]
    for k, b0 in enumerate(range(0, len(worlds), b)):
        spec = {"shape": T.SHAPE[sp.task], "d_mm": sp.d, "task": sp.task, "worlds": worlds[b0:b0 + b],
                "dt": worlds[0]["var"]["dt"], "gen_seed": b0, "batch": n0 + k, "path": str(path),
                "tpath": str(tpath), "layouts": "sobol8kff_size"}
        pr = ctx.Process(target=HG.batch_child, args=(spec,))
        pr.start()
        pr.join()
        if pr.exitcode != 0:
            raise RuntimeError(f"{sp.task}: batch {n0 + k} failed (exit {pr.exitcode})")
    return len(worlds)


def cmd_run(a):
    task = a.task
    sp = Space(task)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{task}_d{sp.d:g}.jsonl"
    runs = [Search(m, s, sp) for m in a.methods for s in a.seeds]
    t0 = time.time()
    for rnd in range(MAX_ROUNDS):
        props = {id(r): r.propose() for r in runs}
        live = [r for r in runs if props[id(r)]]
        if not live:
            break
        n = evaluate(sp, [i for r in live for i in props[id(r)]], path, SEEDS)
        if n or rnd == 0:
            fit = fitness(task, path)
        for r in live:
            r.update(props[id(r)], fit)
        print(f"{time.strftime('%H:%M:%S')} {task} round {rnd}: {len(live)} searches, {n} rollouts run, "
              f"{len(fit)} layouts scored, {time.time() - t0:.0f} s; incumbents "
              + " ".join(f"{r.method}/{r.seed} {r.log[-1]['incumbent_fit']:.2f}@{r.log[-1]['rollouts']}"
                         for r in runs if r.log), flush=True)
    for r in runs:
        (OUT / f"{task}_{r.method}_s{r.seed}_gens.jsonl").write_text("".join(json.dumps(x, default=int) + "\n"
                                                                            for x in r.log))
    return 0


def incumbents(task):
    """{(method, seed): (layout, fitness at seeds 0-2, rollouts)} from the searches' logs."""
    out = {}
    for p in sorted(OUT.glob(f"{task}_*_s*_gens.jsonl")):
        m, s = p.stem[len(task) + 1:].rsplit("_s", 1)
        last = [json.loads(x) for x in p.read_text().splitlines() if x.strip()][-1]
        out[(m, int(s.split("_")[0]))] = (int(last["incumbent"]), float(last["incumbent_fit"]), int(last["rollouts"]))
    return out


def cmd_validate(a):
    task = a.task
    sp = Space(task)
    best = {v[0] for v in incumbents(task).values()}
    if a.track_top:   # also the layouts of best tracking score at seeds 0-2 among every layout the searches ran
        ft = fitness(task, OUT / f"{task}_d{sp.d:g}.jsonl", score="track")
        best |= set(sorted(ft, key=lambda c: -ft[c])[:a.track_top])
    path = OUT / f"{task}_d{sp.d:g}_val.jsonl"
    n = evaluate(sp, sorted(best), path, VAL_SEEDS)
    print(f"{task}: {len(best)} layouts, {n} rollouts run", flush=True)
    return 0


def summary():
    out = {}
    for task, d in TASKS.items():
        inc = incumbents(task)
        if not inc:
            continue
        vp = OUT / f"{task}_d{d:g}_val.jsonl"
        val = fitness(task, vp, VAL_SEEDS)
        rows = [json.loads(x) for x in vp.read_text().splitlines() if x.strip()] if vp.exists() else []
        track = {}
        for r in rows:
            track.setdefault(int(r["call"]), []).append(L.track_score(task, r))
        top = max(val.values()) if val else None
        per = {}
        for (m, s), (i, f, n) in inc.items():
            per.setdefault(m, []).append({"seed": s, "layout": i, "fit": f, "rollouts": n, "val": val.get(i),
                                          "val_track": float(np.mean(track[i])) if i in track else None})
        out[task] = {"best_val": top, "methods": per,
                     "evaluated": len(fitness(task, OUT / f"{task}_d{d:g}.jsonl"))}
    return out


def cmd_report(_a):
    S = summary()
    for task, v in S.items():
        print(f"{task}: best validated {v['best_val']}, {v['evaluated']} layouts evaluated")
        for m, rs in v["methods"].items():
            print(f"  {m:7s} " + "  ".join(f"s{r['seed']} {r['fit']:.2f} -> {r['val'] if r['val'] is None else round(r['val'], 2)}"
                                          f" ({r['rollouts']})" for r in sorted(rs, key=lambda r: r["seed"])))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task", required=True, choices=list(TASKS))
    r.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    v = sub.add_parser("validate")
    v.add_argument("--task", required=True, choices=list(TASKS))
    v.add_argument("--track-top", type=int, default=0, help="also the N layouts of best tracking score at seeds 0-2")
    sub.add_parser("report")
    a = ap.parse_args()
    return {"run": cmd_run, "validate": cmd_validate, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
