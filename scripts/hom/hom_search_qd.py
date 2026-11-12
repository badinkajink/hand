#!/usr/bin/env python3
"""MAP-Elites, CMA-ME, CMA-ES and Bayesian optimization against the EA of Yang et al. and random search, replayed on the
stored grids of the SR2 layout lattice under the tracking score J (owner, 2026-11-12: "can we test how map-elites and/or
another outer loops optimizer does").

The grids are the hand-design talk's: grid2dff_v4b_sv25r (SCS0009 plant at 0.72 N m/rad, 2.5 MPa pads, MuJoCo 3.6.0;
twist and roll on a 40 mm sphere, pin-push and press on a 25 mm cylinder). As in hom_search_compare.py, every search
runs on the grid as a lookup evaluator: a layout's fitness is its best grasp's mean J over seeds 0-2
(hom_landscape.track_score), a layout costs the rollouts its grid row spent (16 grasps x 3 seeds = 48), a layout whose
pose is not reached scores 0 at no cost, and a layout visited again is free. The methods therefore differ only in which
layouts they visit. Every method moves in the box of the two gantries standardized to [0, 1]^2 and is scored at the
nearest point of the 5 mm lattice. 200 runs per method and task; a run stops after 12,000 rollouts.

Methods:
  ea64, ea16   the EA of Yang et al. (hom_landscape.ea_search), populations 64 and 16;
  random       the lattice's layouts in a random order;
  mapelites    MAP-Elites (Mouret and Clune 2015) in pyribs: a grid archive of 8 x 8 cells over the box (15 mm by 15 mm),
               each cell keeping the best layout that falls in it, and the iso+line operator of Vassiliades and Mouret
               (2018; iso 0.05, line 0.2), 16 children a round from elites drawn at random, after 16 random layouts.
               The archive's measure is the layout itself, so the archive is the best layout of each 15 mm region;
  mapelites4   the same with 4 x 4 cells (30 mm, the EA's mutation width);
  cmame        CMA-ME (Fontaine et al. 2020) in pyribs: the same archive, filled by one CMA-ES emitter ranked by
               improvement of the archive (sigma0 0.2, 16 a round, restart when no cell improves);
  cmaes        CMA-ES (Hansen) in pycma from a random start, sigma0 0.3, its default population of 6, restarted with a
               doubled population (IPOP) when it stops or when three generations bring no new layout;
  bo           Bayesian optimization: a Gaussian process (scikit-learn, Matern 5/2 kernel plus white noise, length
               scales fitted each round) on the layouts run so far, and batch Thompson sampling over the lattice
               (16 posterior draws a round, each draw's best layout not yet run), after 16 random layouts.

    .venv/bin/python scripts/hom/hom_search_qd.py               # rows, then the figures
    .venv/bin/python scripts/hom/hom_search_qd.py --figure      # the figures from the rows
    .venv/bin/python scripts/hom/hom_search_qd.py --methods bo --runs 20     # a subset, rows not written
Writes docs/experiments/20261029-foundational_pose_review/rows/20261112-search_qd.tsv (per task and method),
20261112-search_qd_curves.tsv (share of runs within 0.05 of the grid's best against rollouts) and the figures
media/20261112-search_qd.png and media/20261112-search_qd_archive.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_landscape as L  # noqa: E402

ROWS = L.ROWS
GRID = ROWS / "grid2dff_v4b_sv25r"
OUT = ROWS / "20261112-search_qd.tsv"
CURVES = ROWS / "20261112-search_qd_curves.tsv"
MEDIA = ROWS.parent / "media"
FIG = MEDIA / "20261112-search_qd.png"
FIG_ARCHIVE = MEDIA / "20261112-search_qd_archive.png"
TASKS = L.EA_TASKS                                      # (task, size): twist 40, roll 40, pin-push 25, press 25
RUNS, CAP = 200, 12000
EPS = (0.05, 0.02)
BUDGETS = (2000, 3000, 4000, 6000, 8000, 12000)
CURVE_AT = np.arange(0, CAP + 1, 250)
METHODS = ("ea64", "ea16", "random", "mapelites", "mapelites4", "cmame", "cmaes", "bo")
LABELS = {"ea64": "EA, population 64", "ea16": "EA, population 16", "random": "random search",
          "mapelites": "MAP-Elites, 8 x 8 cells", "mapelites4": "MAP-Elites, 4 x 4 cells", "cmame": "CMA-ME", "cmaes": "CMA-ES (IPOP)", "bo": "Bayesian optimization"}
ARCHIVE_DIMS = {"mapelites": (8, 8), "mapelites4": (4, 4), "cmame": (8, 8)}
BATCH = 16


def lookup(task, size):
    """The grid under J as a lookup: {(x_sep, y_sep): (best grasp's mean J over seeds 0-2, rollouts)}, the lattice's
    coordinates, the grid's best and its total rollouts."""
    import hom_rows
    L.GRID_SRC, L.GRID_ROWS = "grid2dff", GRID
    poses = L.grid_poses(task, size)
    sc = {}
    for x in (GRID / f"{task}_d{size:g}.jsonl").read_text().splitlines():
        if not x.strip():
            continue
        r = hom_rows.rescale_metrics(json.loads(x))
        if r.get("status", "ok") != "ok":
            continue
        sc.setdefault(int(r["call"]), {}).setdefault(int(r.get("grasp", 0)), []).append(L.track_score(task, r))
    look = {}
    for call, (xs, ys, _rec) in poses.items():
        g = sc.get(call, {})
        look[(float(xs), float(ys))] = (max((float(np.mean(v)) for v in g.values()), default=0.0),
                                        sum(len(v) for v in g.values()))
    xs = np.array(sorted({k[0] for k in look}))
    ys = np.array(sorted({k[1] for k in look}))
    return look, xs, ys, max(f for f, _n in look.values()), sum(n for _f, n in look.values())


class Evaluator:
    """The lookup with the replay's accounting: a layout's rollouts are spent on its first visit."""

    def __init__(self, look, xs, ys):
        self.look, self.lo, self.hi = look, np.array([xs[0], ys[0]]), np.array([xs[-1], ys[-1]])
        self.seen, self.hist, self.cost = {}, [], 0

    def cell(self, z):
        """The lattice layout nearest a point of the standardized box."""
        m = np.clip(5.0 * np.round((self.lo + np.clip(z, 0.0, 1.0) * (self.hi - self.lo)) / 5.0), self.lo, self.hi)
        return float(m[0]), float(m[1])

    def z(self, k):
        return (np.asarray(k, float) - self.lo) / (self.hi - self.lo)

    def __call__(self, k):
        if k not in self.seen:
            f, n = self.look[k]
            self.seen[k] = f
            self.cost += n
            self.hist.append((self.cost, k))
        return self.seen[k]

    def done(self):
        return self.cost >= CAP or len(self.seen) >= len(self.look)


def run_ribs(ev, rng, kind):
    """MAP-Elites (iso+line) or CMA-ME on a pyribs grid archive whose measure is the layout's standardized position."""
    from ribs.archives import GridArchive
    from ribs.emitters import EvolutionStrategyEmitter, IsoLineEmitter
    from ribs.schedulers import Scheduler
    seed = int(rng.integers(2 ** 31))
    archive = GridArchive(solution_dim=2, dims=ARCHIVE_DIMS[kind], ranges=[(0.0, 1.0), (0.0, 1.0)], seed=seed)
    init = rng.random((BATCH, 2))
    if kind.startswith("mapelites"):
        em = IsoLineEmitter(archive, iso_sigma=0.05, line_sigma=0.2, initial_solutions=init,
                            bounds=[(0.0, 1.0)] * 2, batch_size=BATCH, seed=seed)
    else:
        em = EvolutionStrategyEmitter(archive, x0=init[0], sigma0=0.2, ranker="2imp", bounds=[(0.0, 1.0)] * 2,
                                      batch_size=BATCH, seed=seed)
    sched = Scheduler(archive, [em])
    stale = 0
    while not ev.done() and stale < 200:
        n0 = len(ev.seen)
        X = sched.ask()
        cells = [ev.cell(x) for x in X]
        obj = np.array([ev(k) for k in cells])
        meas = np.array([ev.z(k) for k in cells])
        sched.tell(obj, meas)
        stale = stale + 1 if len(ev.seen) == n0 else 0
    return archive


def run_cmaes(ev, rng):
    """IPOP-CMA-ES in pycma, maximizing J (pycma minimizes, so -J)."""
    import cma
    pop = None
    while not ev.done():
        opts = {"bounds": [[0.0, 0.0], [1.0, 1.0]], "seed": int(rng.integers(1, 2 ** 31)), "verbose": -9,
                "tolflatfitness": 3}
        if pop:
            opts["popsize"] = pop
        es = cma.CMAEvolutionStrategy(rng.random(2), 0.3, opts)
        idle = 0
        while not es.stop() and not ev.done() and idle < 3:
            n0 = len(ev.seen)
            X = es.ask()
            es.tell(X, [-ev(ev.cell(x)) for x in X])
            idle = idle + 1 if len(ev.seen) == n0 else 0
        pop = 2 * es.popsize


def run_bo(ev, rng, keys):
    """A Gaussian process on the layouts run so far, batch Thompson sampling over the lattice."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    Z = np.array([ev.z(k) for k in keys])
    for i in rng.choice(len(keys), BATCH, replace=False):
        ev(keys[i])
    while not ev.done():
        run = [i for i, k in enumerate(keys) if k in ev.seen]
        y = np.array([ev.seen[keys[i]] for i in run])
        kern = (ConstantKernel(0.1, (1e-3, 10.0)) * Matern(0.15, (0.03, 1.0), nu=2.5)
                + WhiteKernel(1e-3, (1e-6, 0.1)))
        gp = GaussianProcessRegressor(kern, normalize_y=True, random_state=int(rng.integers(2 ** 31)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(Z[run], y)
        free = np.array([i for i, k in enumerate(keys) if k not in ev.seen])
        if not len(free):
            break
        S = gp.sample_y(Z[free], n_samples=BATCH, random_state=int(rng.integers(2 ** 31)))
        pick = []
        for j in range(S.shape[1]):
            for i in np.argsort(-S[:, j]):
                if free[i] not in pick:
                    pick.append(free[i])
                    break
        for i in pick:
            ev(keys[i])
            if ev.done():
                break


def one_run(method, look, xs, ys, run):
    rng = np.random.default_rng(1000 * run + 7)
    if method.startswith("ea") or method == "random":
        hist, _seen = L.ea_search(look, rng, xs, ys, pop=int(method[2:]) if method.startswith("ea") else L.EA_POP,
                                  budget=CAP, random=method == "random")
        return hist, None
    ev = Evaluator(look, xs, ys)
    archive = None
    if method in ("mapelites", "mapelites4", "cmame"):
        archive = run_ribs(ev, rng, method)
    elif method == "cmaes":
        run_cmaes(ev, rng)
    elif method == "bo":
        run_bo(ev, rng, list(look))
    return ev.hist, archive


def task_method(args):
    """Every run of one method on one task: the summary row, the curve rows and one run's archive at 6,000 rollouts."""
    task, size, method, runs = args
    look, xs, ys, best, total = lookup(task, size)
    t_hit = {e: [] for e in EPS}
    at6k, final, near6k, share = [], [], [], np.zeros(len(CURVE_AT))
    near = {k for k, (f, n) in look.items() if n > 0 and f >= best - EPS[0]}
    for run in range(runs):
        hist, _arch = one_run(method, look, xs, ys, run)
        inc, hit, inc_at = None, {e: None for e in EPS}, []
        for cost, k in hist:
            if inc is None or look[k][0] > look[inc][0]:
                inc = k
            inc_at.append((cost, look[inc][0]))
            for e in EPS:
                if hit[e] is None and look[inc][0] >= best - e:
                    hit[e] = cost
        for e in EPS:
            t_hit[e].append(hit[e])
        at6k.append(max([s for c, s in inc_at if c <= 6000] or [0.0]))
        final.append(look[inc][0] if inc else 0.0)
        near6k.append(len({k for c, k in hist if c <= 6000} & near))
        share += np.array([hit[0.05] is not None and hit[0.05] <= b for b in CURVE_AT], float)
    row = {"task": task, "size_mm": size, "method": method, "runs": runs, "grid_best": round(best, 4),
           "grid_rollouts": total, "grid_near_0.05": len(near)}
    for e in EPS:
        v = np.array([x if x is not None else np.inf for x in t_hit[e]], float)
        row[f"reached_{e:g}"] = round(float(np.mean(np.isfinite(v))), 4)
        row[f"median_to_{e:g}"] = int(np.median(v)) if np.isfinite(np.median(v)) else ""
    for b in BUDGETS:
        row[f"within_0.05_by_{b}"] = round(float(np.mean([h is not None and h <= b for h in t_hit[0.05]])), 4)
    row.update(median_at_6000=round(float(np.median(at6k)), 4), final_median=round(float(np.median(final)), 4),
               near_found_by_6000=float(np.median(near6k)))
    curves = [{"task": task, "method": method, "rollouts": int(b), "share_within_0.05": round(float(s), 4)}
              for b, s in zip(CURVE_AT, share / runs)]
    print(f"{task:8s} {method:10s} within 0.05 by 6,000: {row['within_0.05_by_6000']:.2f}, by 12,000: "
          f"{row['within_0.05_by_12000']:.2f}; median to 0.05: {row['median_to_0.05']}; at 6,000: "
          f"{row['median_at_6000']:.3f} (grid {best:.3f}); near layouts by 6,000: {row['near_found_by_6000']:.0f} "
          f"of {len(near)}", flush=True)
    return row, curves


def compare(methods, runs, jobs, write=True):
    todo = [(t, s, m, runs) for t, s in TASKS for m in methods]
    with ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(task_method, todo))
    if not write:
        return
    rows = [r for r, _c in res]
    curves = [c for _r, cs in res for c in cs]
    for path, data in ((OUT, rows), (CURVES, curves)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(data)
        print("wrote", path.relative_to(L.ROOT))


NAMES = {"twist": "twist, 40 mm sphere", "roll": "roll, 40 mm sphere", "pinpush": "pin-push, 25 mm cylinder",
         "press": "press, 25 mm cylinder"}
STYLE = {"ea64": ("#8c510a", "--", 1.8), "ea16": ("#8c510a", "-", 1.4), "random": ("#8a8a8a", ":", 1.8),
         "mapelites": ("#c0392b", "-", 2.0), "mapelites4": ("#c0392b", ":", 2.0), "cmame": ("#e67e22", "-.", 1.8), "cmaes": ("#1f63b4", "-", 1.8),
         "bo": ("#2a9d8f", "-", 2.0)}


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 14,
                         "mathtext.fontset": "stix"})
    return plt


def figure():
    """Share of the 200 runs within 0.05 of the grid's best J against rollouts, one panel per task."""
    plt = plt_setup()
    cur = list(csv.DictReader(open(CURVES), delimiter="\t"))
    fig, axs = plt.subplots(1, 4, figsize=(15.0, 4.4), sharey=True)
    for j, (task, _size) in enumerate(TASKS):
        ax = axs[j]
        for m, (c, ls, lw) in STYLE.items():
            pts = [(int(r["rollouts"]), float(r["share_within_0.05"])) for r in cur
                   if r["task"] == task and r["method"] == m]
            if pts:
                ax.plot([p[0] / 1e3 for p in pts], [p[1] for p in pts], color=c, ls=ls, lw=lw, label=LABELS[m])
        ax.axvline(6.0, color="#bbbbbb", lw=1.0, ls=":")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, CAP / 1e3)
        ax.grid(color="#eeeeee")
        ax.set_title(NAMES[task], fontsize=14)
        ax.set_xlabel("Rollouts (thousands)")
    axs[0].set_ylabel("Share of runs within 0.05 of the best J")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=13, frameon=False)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(FIG, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", FIG.relative_to(L.ROOT))


def figure_archive(task="roll", size=40.0, run=0):
    """The roll's J landscape, and the layouts MAP-Elites and the EA (population 64) ran in one run by 6,000 rollouts,
    each with its incumbent."""
    plt = plt_setup()
    look, xs, ys, best, _total = lookup(task, size)
    Zm = np.full((len(ys), len(xs)), np.nan)
    for (x, y), (f, n) in look.items():
        Zm[int(np.searchsorted(ys, y)), int(np.searchsorted(xs, x))] = f if n > 0 else np.nan
    fig, axs = plt.subplots(1, 3, figsize=(15.6, 5.2))
    for ax, method in zip(axs, ("full", "mapelites", "ea64")):
        im = ax.imshow(Zm, origin="lower", extent=(xs[0] - 2.5, xs[-1] + 2.5, ys[0] - 2.5, ys[-1] + 2.5),
                       cmap="viridis", vmin=0, vmax=max(best, 1e-3), aspect="equal")
        if method == "full":
            k = max(look, key=lambda q: look[q][0])
            ax.plot(*k, "*", color="white", mec="k", ms=16)
            ax.set_title(f"{NAMES[task].capitalize()}:\nJ of every layout", fontsize=14)
        else:
            hist, _arch = one_run(method, look, xs, ys, run)
            pts = np.array([k for c, k in hist if c <= 6000])
            ax.plot(pts[:, 0], pts[:, 1], "o", mfc="none", mec="white", ms=4.5, mew=0.9)
            inc = max((k for c, k in hist if c <= 6000), key=lambda q: look[q][0])
            ax.plot(*inc, "*", color="#c0392b", mec="k", ms=16)
            ax.set_title(f"{LABELS[method]}:\n{len(pts)} layouts run by 6,000 rollouts", fontsize=14)
            if method == "mapelites":
                for v in np.linspace(xs[0], xs[-1], ARCHIVE_DIMS[method][0] + 1)[1:-1]:
                    ax.axvline(v, color="white", lw=0.5, alpha=0.5)
                for v in np.linspace(ys[0], ys[-1], ARCHIVE_DIMS[method][1] + 1)[1:-1]:
                    ax.axhline(v, color="white", lw=0.5, alpha=0.5)
        ax.set_xlabel("$x_{sep}$ (mm)")
        ax.set_ylabel("$y_{sep}$ (mm)")
    fig.subplots_adjust(wspace=0.32)
    fig.colorbar(im, ax=axs, shrink=0.85, label="J of the best grasp")
    fig.savefig(FIG_ARCHIVE, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG_ARCHIVE.relative_to(L.ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--figure", action="store_true", help="only the figures, from the rows")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if not a.figure:
        full = list(a.methods) == list(METHODS) and a.runs == RUNS
        compare(a.methods, a.runs, a.jobs, write=full)
        if not full:
            return 0
    figure()
    figure_archive()
    return 0


if __name__ == "__main__":
    sys.exit(main())
