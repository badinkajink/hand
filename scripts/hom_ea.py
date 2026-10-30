#!/usr/bin/env python3
"""The evolutionary search of Yang et al. (IEEE TEVC 2026) run live on the HOM GPU evaluator, over the symmetric tripods'
5 mm lattice, for the object-size study of the design page (owner, 2026-10-30: the EA "drastically improves the sample
efficiency of our search ... we can perhaps explore at even more cylinder, sphere sizes for these tasks and measure
trends?").

The search is hom_landscape.ea_search, which was checked against the full grid's stored rollouts: an elitist
mutation-only EA whose generation keeps its best half, takes its best eighth as parents, and fills the other half with
children, each a parent moved by U(-delta/2, delta/2) mm in x_sep and y_sep, rounded to the 5 mm lattice and clipped to
the gantries' travel. A layout's fitness is its best grasp's score: every grasp of its grasp set (fp_sr2_toy.py grid2d,
rows grid2dff_*: up to 16 grasps, the plant's finger clearance) run on seeds 0-2 by scripts/hom_gpu.py, the score of
hom_landscape.rollout_score averaged over the seeds, the best of the set. A layout whose pose is not reached scores 0 at
no rollout. Each generation's new layouts run as GPU batches; the rows are kept, so a stopped search resumes by replaying
its generator against them. The sphere tasks run the variant base, the cylinder tasks hold (the object held at its start
until the grasp closes, hom_tasks.PLANT_VARIANTS).

    .venv/bin/python scripts/hom_ea.py run --task twist --d 45 [--pop 64] [--budget 6000] [--seed 0]
    .venv/bin/python scripts/hom_ea.py sweep --tasks twist roll --d 25 35 45 [--pop 64] [--budget 6000]
    .venv/bin/python scripts/hom_ea.py report      # per task and size: reach, the EA's and the grid's best layouts
Rows: docs/experiments/20261029-foundational_pose_review/rows/ea_live/<task>_d<d>_s<seed>.jsonl (the rollouts, in
hom_gpu.py's format), <task>_d<d>_s<seed>_timing.jsonl (one line per batch) and <task>_d<d>_s<seed>_gens.jsonl (one line
per generation: the population with its fitness, the incumbent, the rollouts so far).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_gpu as HG  # noqa: E402
import hom_landscape as L  # noqa: E402

T = HG.T
OUT = HG.ROWS / "ea_live"
SRC = "grid2dff"                                # the pose toy's grids with the plant's finger clearance
STEP = 5.0
LO, HI = np.array([40.0, 50.0]), np.array([160.0, 170.0])     # fp_sr2_toy.LO2, HI2
SEEDS = (0, 1, 2)
VARIANT = {"twist": "base", "roll": "base", "pinpush": "hold", "press": "hold"}
BATCH = {"twist": 2048, "roll": 2048, "pinpush": 1536, "press": 1536}
SIZES = {"sphere": (25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 56.0, 63.0, 71.0, 80.0),
         "cylinder": (8.0, 10.0, 13.0, 16.0, 20.0, 25.0, 32.0, 40.0, 50.0)}
NEAR = 0.05                                      # a layout counts as near the best within this score


def lattice():
    """{(x_sep, y_sep): grid index} in fp_sr2_toy.grid2d's order (x_sep major)."""
    xs = np.arange(LO[0], HI[0] + 1e-9, STEP)
    ys = np.arange(LO[1], HI[1] + 1e-9, STEP)
    return {(float(x), float(y)): k for k, (x, y) in enumerate((x, y) for x in xs for y in ys)}


def fitness(task, path):
    """{grid index: (best-of-set score, rollouts)} from rows in hom_gpu.py's format."""
    sc = {}
    for x in (path.read_text().splitlines() if path.exists() else []):
        if x.strip():
            r = json.loads(x)
            sc.setdefault(int(r["call"]), {}).setdefault(int(r.get("grasp", 0)), []).append(L.rollout_score(task, r))
    return {c: (max(float(np.mean(v)) for v in g.values()), sum(len(v) for v in g.values())) for c, g in sc.items()}


def wait_rows(task, d, src=SRC):
    """The toy's grid rows for the task's object of diameter d (mm), waiting for the toy's queue to write them."""
    import fp_sr2_toy as FP
    p = HG.ROWS / f"{FP.size_name(src, T.SHAPE[task], d)}.jsonl"
    while not p.exists():
        print(f"waiting for {p.name}", flush=True)
        time.sleep(60)
    return p


class Live:
    """One task on one object: the lattice's grasp sets, the GPU evaluation of a list of layouts, and the fitness of
    every layout run so far (read back from the rows). src: the toy's grids (grid2dgr: grasps clear of the ground)."""

    def __init__(self, task, d, seed, out=OUT, src=SRC):
        wait_rows(task, d, src)
        self.task, self.d, self.src = task, float(d), src
        self.scene = HG.Scene(T.SHAPE[task], d)
        self.cells, self.grasps = lattice(), {}
        for key, mxy, setup in HG.layout_list(task, f"{src}_size", 0, d):
            self.grasps.setdefault(int(key["call"]), []).append((key, mxy, setup))
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{task}_d{d:g}_s{seed}"
        self.path, self.tpath, self.gpath = (out / f"{stem}.jsonl", out / f"{stem}_timing.jsonl",
                                             out / f"{stem}_gens.jsonl")
        self.fit, self.cost = {}, {}
        self._read()

    def _read(self):
        for c, (f, n) in fitness(self.task, self.path).items():
            self.fit[c], self.cost[c] = f, n

    def evaluate(self, cells):
        """Run every layout of cells (lattice points) not run yet; unreached layouts score 0 at no rollout."""
        import multiprocessing as mp
        todo = []
        for xy in cells:
            c = self.cells[xy]
            if c in self.fit:
                continue
            if c not in self.grasps:
                self.fit[c], self.cost[c] = 0.0, 0
                continue
            todo.append(c)
        if not todo:
            return
        var = VARIANT[self.task]
        worlds = []
        for c in todo:
            for key, mxy, setup in self.grasps[c]:
                bm, q_open = self.scene.prepare(mxy, setup)
                for sd in SEEDS:
                    worlds.append({"key": key, "seed": sd, "setup": setup, "body_mounts": bm, "q_open": q_open,
                                   "var": HG.variant_world(var, sd)})
        ctx = mp.get_context("spawn")
        b = BATCH[self.task]
        n0 = sum(1 for x in (self.tpath.read_text().splitlines() if self.tpath.exists() else []) if x.strip())
        for i, b0 in enumerate(range(0, len(worlds), b)):
            spec = {"shape": T.SHAPE[self.task], "d_mm": self.d, "task": self.task, "worlds": worlds[b0:b0 + b],
                    "dt": worlds[0]["var"]["dt"], "gen_seed": b0, "batch": n0 + i, "path": str(self.path),
                    "tpath": str(self.tpath), "layouts": f"{self.src}_size"}
            pr = ctx.Process(target=HG.batch_child, args=(spec,))
            pr.start()
            pr.join()
            if pr.exitcode != 0:
                raise RuntimeError(f"{self.task} {self.d:g} mm: batch {n0 + i} failed (exit {pr.exitcode})")
        self._read()
        for c in todo:                                  # a layout whose every rollout failed to score
            self.fit.setdefault(c, 0.0)
            self.cost.setdefault(c, 0)


def search(live, rng, pop, delta, budget, gens):
    """The EA of hom_landscape.ea_search with live evaluation, one GPU evaluation per generation; returns the
    generations' log lines."""
    keys = list(live.cells)
    P = [keys[i] for i in rng.choice(len(keys), pop, replace=False)]
    seen = []

    def spent():
        return sum(live.cost[live.cells[k]] for k in dict.fromkeys(seen))
    log = []
    for g in range(gens + 1):
        live.evaluate(P)
        seen += P
        P = sorted(P, key=lambda k: -live.fit[live.cells[k]])
        inc = max(dict.fromkeys(seen), key=lambda k: live.fit[live.cells[k]])
        line = {"gen": g, "rollouts": spent(), "layouts": len(set(seen)), "incumbent": list(inc),
                "incumbent_fit": live.fit[live.cells[inc]],
                "pop": [[k[0], k[1], round(live.fit[live.cells[k]], 4)] for k in P]}
        log.append(line)
        print(f"{live.task} {live.d:g} mm gen {g}: {line['rollouts']} rollouts, {line['layouts']} layouts, incumbent "
              f"{inc} {line['incumbent_fit']:.3f}", flush=True)
        if line["rollouts"] >= budget or g == gens:
            break
        keep, parents = P[:pop // 2], P[:max(1, pop // 8)]
        kids = []
        for i in range(pop - len(keep)):
            m = np.asarray(parents[i % len(parents)]) + rng.uniform(-delta / 2, delta / 2, 2)
            m = np.clip(STEP * np.round(m / STEP), LO, HI)
            kids.append((float(m[0]), float(m[1])))
        P = keep + kids
    return log


OUT_SRC = {"grid2dff": OUT, "grid2dgr": HG.ROWS / "ea_live_gr"}


def cmd_run(task, d, seed, pop, delta, budget, gens, src=SRC):
    live = Live(task, d, seed, OUT_SRC[src], src)
    t0 = time.time()
    log = search(live, np.random.default_rng(1000 * seed + 7), pop, delta, budget, gens)
    live.gpath.write_text("".join(json.dumps(x) + "\n" for x in log))
    print(f"{task} {d:g} mm seed {seed}: incumbent {log[-1]['incumbent']} {log[-1]['incumbent_fit']:.3f} after "
          f"{log[-1]['rollouts']} rollouts in {time.time() - t0:.0f} s -> {live.gpath.relative_to(ROOT)}", flush=True)


def reach_share(task, d, src=SRC):
    """Share of the lattice's 625 layouts whose pose for the task is reached on the object of diameter d (the toy's
    rows), or None before the toy has run that size."""
    import fp_sr2_toy as FP
    p = HG.ROWS / f"{FP.size_name(src, T.SHAPE[task], d)}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return float(np.mean([bool((r["fp"].get(HG.POSE[task]) or {}).get("ok")) for r in rows]))


def outcome(fit, inv):
    """The best layout of {grid index: (score, rollouts)} and the layouts within NEAR of it: their count, centroid and
    range (mm)."""
    run = {c: f for c, (f, n) in fit.items() if n > 0}
    if not run:
        return None
    best = max(run, key=run.get)
    near = np.array([inv[c] for c, f in run.items() if f >= run[best] - NEAR])
    return {"best_xy": list(inv[best]), "best": round(run[best], 4), "layouts": len(run),
            "rollouts": int(sum(n for _f, n in fit.values())), "near_n": len(near),
            "near_centroid": near.mean(0).round(1).tolist(), "near_lo": near.min(0).tolist(),
            "near_hi": near.max(0).tolist()}


def cmd_report(seed=0, src=SRC):
    """Per task and size: the share of layouts that reach the pose, the EA's outcome and the full grid's where one ran.
    Writes rows/ea_live[_gr]/summary.json and the figures 20261030-size_trends[_gr].png and size_reach[_gr].png."""
    inv = {c: xy for xy, c in lattice().items()}
    out_dir, grid = OUT_SRC[src], HG.ROWS / f"{src}_v4b"
    out = {}
    for task in HG.TASKS:
        for d in SIZES[T.SHAPE[task]]:
            res = {"task": task, "d_mm": d, "reach": reach_share(task, d, src)}
            if (out_dir / f"{task}_d{d:g}_s{seed}_gens.jsonl").exists():
                res["ea"] = outcome(fitness(task, out_dir / f"{task}_d{d:g}_s{seed}.jsonl"), inv)
            if (grid / f"{task}_d{d:g}.jsonl").exists():
                res["grid"] = outcome(fitness(task, grid / f"{task}_d{d:g}.jsonl"), inv)
            out[f"{task}_d{d:g}"] = res
            if res.get("ea") or res.get("grid"):
                print(task, d, {k: v for k, v in res.items() if k in ("reach", "ea", "grid")}, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"near": NEAR, "seed": seed, "src": src, **out}, indent=1))
    fig_trends(out, "" if src == SRC else "_gr")
    return 0


NAMES = {"twist": "Twist", "roll": "Roll", "pinpush": "Pin-push", "press": "Press"}


def fig_trends(out, suffix=""):
    """Per task (columns): the best score found against the object's diameter, and the best layout's x_sep and y_sep
    with the range of the layouts within NEAR of it; the EA's (filled) and the full grid's (open) where both ran."""
    import fp_review_figures as FR
    plt = FR.plt_setup()
    fig, axs = plt.subplots(3, 4, figsize=(18, 11), sharex="col")
    for j, task in enumerate(("twist", "roll", "pinpush", "press")):
        for kind, mk, fc, dx in (("ea", "o", "k", -0.3), ("grid", "s", "none", 0.3)):
            pts = [(v["d_mm"], v[kind]) for v in out.values() if v["task"] == task and v.get(kind)]
            if not pts:
                continue
            d = np.array([p[0] for p in pts]) + dx
            lab = "EA, population 64" if kind == "ea" else "full grid"
            axs[0, j].plot(d, [p[1]["best"] for p in pts], mk, mfc=fc, mec="k", ms=7, label=lab)
            for i, (ax, k) in enumerate(((axs[1, j], 0), (axs[2, j], 1))):
                y = np.array([p[1]["best_xy"][k] for p in pts])
                lo = np.array([p[1]["near_lo"][k] for p in pts])
                hi = np.array([p[1]["near_hi"][k] for p in pts])
                ax.errorbar(d, y, yerr=[y - lo, hi - y], fmt=mk, mfc=fc, mec="k", ecolor="0.5", ms=7, capsize=3)
        axs[0, j].set_title(NAMES[task] + (", spheres" if T.SHAPE[task] == "sphere" else ", cylinders"))
        axs[0, j].set_ylim(0, 1.05)
        axs[1, j].axhline(LO[0], color="0.6", ls=":", lw=1)
        axs[1, j].axhline(HI[0], color="0.6", ls=":", lw=1)
        axs[2, j].axhline(LO[1], color="0.6", ls=":", lw=1)
        axs[2, j].axhline(HI[1], color="0.6", ls=":", lw=1)
        axs[2, j].set_xlabel("Object diameter (mm)")
    axs[0, 0].set_ylabel("Best layout's score")
    axs[1, 0].set_ylabel("Its thumb-to-pair x_sep (mm)")
    axs[2, 0].set_ylabel("Its index-to-middle y_sep (mm)")
    axs[0, 0].legend(loc="lower left")
    fig.tight_layout()
    FR.save(fig, f"20261030-size_trends{suffix}.png")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, tasks in zip(axs, (("twist",), ("pinpush", "press"))):
        for task, ls in zip(tasks, ("-", "--")):
            pts = [(v["d_mm"], v["reach"]) for v in out.values() if v["task"] == task and v.get("reach") is not None]
            lab = {"twist": "tripod on the sphere", "pinpush": "pinch", "press": "press"}[task]
            ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], "k" + ls, marker="o", ms=5, label=lab)
        ax.set_xlabel("Object diameter (mm)")
        ax.set_ylim(0, 100)
        ax.legend(loc="lower right")
    axs[0].set_ylabel("Layouts that reach the pose (%)")
    fig.tight_layout()
    FR.save(fig, f"20261030-size_reach{suffix}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report")
    p.add_argument("--src", choices=("grid2dff", "grid2dgr"), default=SRC)
    for name in ("run", "sweep"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("--task", choices=HG.TASKS, required=True)
            p.add_argument("--d", type=float, required=True)
        else:
            p.add_argument("--tasks", nargs="+", choices=HG.TASKS, required=True)
            p.add_argument("--d", type=float, nargs="+", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--pop", type=int, default=64)
        p.add_argument("--delta", type=float, default=L.EA_DELTA)
        p.add_argument("--budget", type=int, default=6000, help="rollouts after which no generation starts")
        p.add_argument("--gens", type=int, default=40)
        p.add_argument("--src", choices=tuple(OUT_SRC), default=SRC, help="the toy's grids: grid2dgr, the grasps clear "
                       "of the ground (hom_ground.py)")
    a = ap.parse_args()
    if a.cmd == "report":
        return cmd_report(src=a.src)
    if a.cmd == "run":
        cmd_run(a.task, a.d, a.seed, a.pop, a.delta, a.budget, a.gens, a.src)
        return 0
    for d in a.d:
        for task in a.tasks:
            cmd_run(task, d, a.seed, a.pop, a.delta, a.budget, a.gens, a.src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
