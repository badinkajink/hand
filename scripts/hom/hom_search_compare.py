#!/usr/bin/env python3
"""Particle swarm optimization against the evolutionary algorithm of Yang et al., random search, a Sobol sequence and the
FP paper's RRT, on the stored full
grids of the SR2 layout study (owner, 2026-11-09: "see if particle swarm optimization, my advisor's preferred genetic
algorithm, improves efficiency or performance on these tasks").

Every search runs on the grid as a lookup evaluator (hom_landscape.ea_landscape), as the EA's reliability test of the
trends page did: a layout's fitness is its best grasp's score (the grasp set's best mean over seeds 0-2), a layout costs
the rollouts its grid row spent, a layout whose pose is not reached scores 0 at no cost, and a layout visited again is
free. So the methods differ only in which layouts they visit. The swarm moves in mm over the gantries' box and is scored
at the nearest point of the 5 mm lattice. Constriction coefficients of Clerc and Kennedy (w 0.7298, c1 = c2 1.49618),
velocities capped at a fifth of the box, a particle stopped at a wall, global-best or ring topology. 200 runs per method,
task and plant; each run stops after the grid's own rollouts or 400 iterations.

Plants: the work plant at 10 MPa (rows grid2dff_v4b) and the SCS0009 plant at 0.72 N m/rad with 2.5 MPa pads (rows
grid2dff_v4b_sv25), the owner's plant for now (11-09: "we can stick to 0.72").

    .venv/bin/python scripts/hom/hom_search_compare.py            # rows, then the figure
    .venv/bin/python scripts/hom/hom_search_compare.py --figure   # the figure from the rows
Writes docs/experiments/20261029-foundational_pose_review/rows/20261109-search_compare.tsv (per plant, task and method)
and 20261109-search_compare_curves.tsv (share of runs within 0.05 of the grid's best against rollouts), and
media/20261109-search_compare.png.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_landscape as L  # noqa: E402

ROWS = L.ROWS
OUT = ROWS / "20261109-search_compare.tsv"
CURVES = ROWS / "20261109-search_compare_curves.tsv"
FIG = L.ROWS.parent / "media" / "20261109-search_compare.png"
PLANTS = {"work": "", "scs0009": "_sv25"}               # rows tag of each plant's grids
PLANT_NAMES = {"work": "work plant, 10 MPa", "scs0009": "SCS0009 plant, 2.5 MPa"}
TASKS = L.EA_TASKS                                      # (task, size) at the default objects
RUNS, ITERS = 200, 400
EPS = (0.05, 0.02)
BUDGETS = (2000, 3000, 4000, 6000, 8000)
CURVE_AT = np.arange(0, 12001, 250)
W, C1, C2, VMAX = 0.7298, 1.49618, 1.49618, 0.2         # constriction coefficients; velocity cap, share of the box
METHODS = {"ea16": ("ea", 16), "ea64": ("ea", 64), "pso_g16": ("gbest", 16), "pso_g64": ("gbest", 64),
           "pso_r64": ("ring", 64), "random": ("random", 0), "sobol": ("sobol", 0), "rrt": ("rrt", 0)}
LABELS = {"ea16": "EA, population 16", "ea64": "EA, population 64", "pso_g16": "PSO, global best, 16",
          "pso_g64": "PSO, global best, 64", "pso_r64": "PSO, ring, 64", "random": "random search",
          "sobol": "Sobol sequence", "rrt": "RRT of the FP paper"}


def pso_search(look, rng, xs, ys, swarm=16, topology="gbest", budget=None):
    """One run of particle swarm optimization on the lattice of the grid: positions in mm, each scored at its nearest
    lattice point; a layout is evaluated once. Returns [(rollouts so far, layout)] for every new evaluation."""
    seen, hist, cost = {}, [], [0]

    def ev(k):
        if k not in seen:
            f, n = look[k]
            seen[k] = f
            cost[0] += n
            hist.append((cost[0], k))
        return seen[k]

    lo, hi = np.array([xs[0], ys[0]], float), np.array([xs[-1], ys[-1]], float)
    vmax = VMAX * (hi - lo)

    def cell(x):
        m = np.clip(5.0 * np.round(x / 5.0), lo, hi)
        return float(m[0]), float(m[1])

    X = lo + rng.random((swarm, 2)) * (hi - lo)
    V = rng.uniform(-1.0, 1.0, (swarm, 2)) * vmax
    f = np.array([ev(cell(x)) for x in X])
    P, Pf = X.copy(), f.copy()
    for _ in range(ITERS):
        if budget and cost[0] >= budget:
            break
        if topology == "gbest":
            G = np.repeat(P[[int(np.argmax(Pf))]], swarm, axis=0)
        else:                                            # ring: the best of each particle and its two neighbours
            nb = np.stack([np.roll(np.arange(swarm), s) for s in (-1, 0, 1)], 1)
            G = P[nb[np.arange(swarm), np.argmax(Pf[nb], axis=1)]]
        r1, r2 = rng.random((swarm, 2)), rng.random((swarm, 2))
        V = np.clip(W * V + C1 * r1 * (P - X) + C2 * r2 * (G - X), -vmax, vmax)
        X = X + V
        wall = (X < lo) | (X > hi)
        X, V = np.clip(X, lo, hi), np.where(wall, 0.0, V)
        C = np.clip(5.0 * np.round(X / 5.0), lo, hi).tolist()
        for i in range(swarm):
            fi = ev((C[i][0], C[i][1]))
            if fi > Pf[i]:
                P[i], Pf[i] = X[i], fi
    return hist, seen


RRT_STEP, RRT_SEED = 0.05, (100.0, 110.0)              # the 2-D run of the design page (rows rrt2d_summary.json)


def order_search(look, cells, budget):
    """Evaluate the lattice cells in the given order, each once, until the budget: the history of new evaluations."""
    seen, hist, cost = set(), [], 0
    for k in cells:
        if k in seen or k not in look:
            continue
        seen.add(k)
        cost += look[k][1]
        hist.append((cost, k))
        if cost >= budget:
            break
    return hist


def sobol_cells(xs, ys, run, n=4096):
    """A scrambled Sobol sequence over the box, each point rounded to the lattice."""
    from scipy.stats import qmc
    u = qmc.Sobol(d=2, scramble=True, seed=run).random(n)
    lo, hi = np.array([xs[0], ys[0]], float), np.array([xs[-1], ys[-1]], float)
    m = np.clip(5.0 * np.round((lo + u * (hi - lo)) / 5.0), lo, hi)
    return [(float(a), float(b)) for a, b in m]


def rrt_cells(look, xs, ys, rng, tries=40000):
    """The FP paper's RRT on the lattice (fp_sr2_toy.rrt): a uniform sample in the box, the nearest tree layout, a step
    of RRT_STEP of the box toward the sample, rounded to the lattice; kept when its pose is reached (a layout of no
    rollouts is not). The tree's layouts in the order they were added."""
    lo, hi = np.array([xs[0], ys[0]], float), np.array([xs[-1], ys[-1]], float)
    span = hi - lo
    tree, order = [np.array(RRT_SEED)], [RRT_SEED]
    have = {RRT_SEED}
    n_reach = sum(1 for f_n in look.values() if f_n[1] > 0)
    for _ in range(tries):
        if len(order) >= n_reach:                       # every reached layout is in the tree
            break
        u = lo + rng.random(2) * span
        T = np.array(tree)
        dz = np.linalg.norm((T - u) / span, axis=1)
        i = int(dz.argmin())
        z = u if dz[i] <= RRT_STEP else T[i] + RRT_STEP * (u - T[i]) / dz[i]
        m = np.clip(5.0 * np.round(z / 5.0), lo, hi)
        k = (float(m[0]), float(m[1]))
        if k in have or k not in look or look[k][1] == 0:
            continue
        have.add(k)
        tree.append(m)
        order.append(k)
    return order


def run_method(method, look, xs, ys, total, rng, run=0):
    kind, n = METHODS[method]
    if kind == "sobol":
        return order_search(look, sobol_cells(xs, ys, run), total)
    if kind == "rrt":
        return order_search(look, rrt_cells(look, xs, ys, rng), total)
    if kind in ("gbest", "ring"):
        return pso_search(look, rng, xs, ys, swarm=n, topology=kind, budget=total)[0]
    return L.ea_search(look, rng, xs, ys, pop=n or L.EA_POP, budget=total, random=kind == "random")[0]


def compare():
    rows, curves = [], []
    for plant, tag in PLANTS.items():
        L.GRID_SRC, L.GRID_ROWS = "grid2dff", ROWS / f"grid2dff_v4b{tag}"
        for task, size in TASKS:
            look, basin, _top, best, total, xs, ys = L.ea_landscape(task, size)
            fit = look["best"]
            best_cell = max(fit, key=lambda k: fit[k][0])
            for method in METHODS:
                t_hit = {e: [] for e in EPS}
                at6k, final, other, share_curve = [], [], [], np.zeros(len(CURVE_AT))
                for run in range(RUNS):
                    rng = np.random.default_rng(1000 * run + 7)
                    hist = run_method(method, fit, xs, ys, total, rng, run)
                    inc, hit, inc_at = None, {e: None for e in EPS}, []
                    for cost, k in hist:
                        if inc is None or fit[k][0] > fit[inc][0]:
                            inc = k
                        inc_at.append((cost, fit[inc][0]))
                        for e in EPS:
                            if hit[e] is None and fit[inc][0] >= best - e:
                                hit[e] = cost
                    for e in EPS:
                        t_hit[e].append(hit[e])
                    at6k.append(max([s for c, s in inc_at if c <= 6000] or [0.0]))
                    final.append(fit[inc][0])
                    other.append(basin[inc] != basin[best_cell])
                    h = hit[0.05]
                    share_curve += np.array([h is not None and h <= b for b in CURVE_AT], float)
                row = {"plant": plant, "task": task, "size_mm": size, "method": method, "grid_rollouts": total,
                       "grid_best": round(best, 4), "runs": RUNS}
                for e in EPS:
                    v = np.array([x if x is not None else np.inf for x in t_hit[e]], float)
                    row[f"reached_{e:g}"] = round(float(np.mean(np.isfinite(v))), 4)
                    row[f"median_to_{e:g}"] = round(float(np.median(v)), 0) if np.isfinite(np.median(v)) else ""
                    row[f"q75_to_{e:g}"] = (round(float(np.percentile(v, 75)), 0)
                                            if np.isfinite(np.percentile(v, 75)) else "")
                for b in BUDGETS:
                    row[f"within_0.05_by_{b}"] = round(float(np.mean([h is not None and h <= b for h in t_hit[0.05]])), 4)
                row.update(median_at_6000=round(float(np.median(at6k)), 4), final_median=round(float(np.median(final)), 4),
                           other_basin=round(float(np.mean(other)), 4))
                rows.append(row)
                for b, s in zip(CURVE_AT, share_curve / RUNS):
                    curves.append({"plant": plant, "task": task, "method": method, "rollouts": int(b),
                                   "share_within_0.05": round(float(s), 4)})
                print(f"{plant:8s} {task:8s} {method:8s} within 0.05 by 6,000: {row['within_0.05_by_6000']:.2f}; "
                      f"median to 0.05: {row['median_to_0.05']}; at 6,000: {row['median_at_6000']:.3f} "
                      f"(grid {best:.3f}, {total} rollouts)", flush=True)
    for path, data in ((OUT, rows), (CURVES, curves)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(data)
        print("wrote", path.relative_to(L.ROOT))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 13,
                         "mathtext.fontset": "stix"})
    cur = list(csv.DictReader(open(CURVES), delimiter="\t"))
    style = {"ea16": ("#8c510a", "-", 1.6), "ea64": ("#8c510a", "--", 1.8), "pso_g16": ("#1f63b4", "-", 1.6),
             "pso_g64": ("#1f63b4", "--", 1.8), "random": ("#8a8a8a", ":", 1.8), "sobol": ("#7b3fa0", ":", 2.0),
             "rrt": ("#2a9d8f", "-.", 1.8)}                # the ring swarm is in the tables only
    names = {"twist": "twist, 40 mm sphere", "roll": "roll, 40 mm sphere", "pinpush": "pin-push, 25 mm cylinder",
             "press": "press, 25 mm cylinder"}
    fig, axs = plt.subplots(2, 4, figsize=(13.2, 6.6), sharex=True, sharey=True)
    for i, plant in enumerate(PLANTS):
        for j, (task, _size) in enumerate(TASKS):
            ax = axs[i, j]
            for m, (c, ls, lw) in style.items():
                pts = [(int(r["rollouts"]), float(r["share_within_0.05"])) for r in cur
                       if r["plant"] == plant and r["task"] == task and r["method"] == m]
                if pts:
                    ax.plot([p[0] / 1e3 for p in pts], [p[1] for p in pts], color=c, ls=ls, lw=lw, label=LABELS[m])
            ax.axvline(6.0, color="#bbbbbb", lw=1.0, ls=":")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(color="#eeeeee")
            if i == 0:
                ax.set_title(names[task], fontsize=13)
            if j == 0:
                ax.set_ylabel(f"{PLANT_NAMES[plant]}\nshare of runs within 0.05")
            if i == 1:
                ax.set_xlabel("rollouts (thousands)")
    axs[0, 0].legend(loc="lower right", fontsize=10.5, frameon=True, framealpha=0.92, edgecolor="none")
    fig.tight_layout()
    fig.savefig(FIG, dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", FIG.relative_to(L.ROOT))


PARETO = ROWS / "20261109-pareto_twist_press.tsv"
PARETO_FIG = L.ROWS.parent / "media" / "20261109-pareto_twist_press.png"


def pareto():
    """The twist (40 mm sphere) and press (25 mm cylinder) best-of-16 scores of every layout that reaches both poses on
    the SCS0009 plant's grids (seeds 0-2), and the Pareto front of the two: the layouts that no other layout beats on
    both scores at once. Writes the rows and the figure of the first-principles section."""
    L.GRID_SRC, L.GRID_ROWS = "grid2dff", ROWS / "grid2dff_v4b_sv25"
    tw = L.ea_landscape("twist", 40.0)[0]["best"]
    pr = L.ea_landscape("press", 25.0)[0]["best"]
    keys = [k for k in tw if k in pr and tw[k][1] > 0 and pr[k][1] > 0]
    pts = np.array([[tw[k][0], pr[k][0]] for k in keys])
    front = [i for i in range(len(pts)) if not np.any(np.all(pts >= pts[i], 1) & np.any(pts > pts[i], 1))]
    with open(PARETO, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["x_sep_mm", "y_sep_mm", "twist", "press", "front"])
        for i, k in enumerate(keys):
            w.writerow([k[0], k[1], round(pts[i, 0], 4), round(pts[i, 1], 4), int(i in front)])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 13})
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.scatter(pts[:, 0], pts[:, 1], s=16, color="#9a9a9a", alpha=0.7, label="layout", zorder=2)
    fr = sorted(front, key=lambda i: pts[i, 0])
    ax.step(pts[fr, 0], pts[fr, 1], where="post", color="#c0392b", lw=1.4, zorder=3)
    ax.plot(pts[fr, 0], pts[fr, 1], "o", color="#c0392b", ms=7, label="Pareto front: no layout scores higher on both",
            zorder=4)
    d = max((i for i in range(len(pts)) if i not in front), key=lambda i: pts[i].min())   # a balanced dominated layout
    win = max((i for i in front if np.all(pts[i] >= pts[d])), key=lambda i: pts[i].sum())
    ax.add_patch(plt.Rectangle(pts[d], 1.05 - pts[d, 0], 1.05 - pts[d, 1], color="#c0392b", alpha=0.08, zorder=1))
    ax.plot(*pts[d], "s", color="#1f63b4", ms=8, zorder=5)
    ax.annotate(f"({keys[d][0]:.0f}, {keys[d][1]:.0f}) mm is dominated: ({keys[win][0]:.0f}, {keys[win][1]:.0f}) mm\n"
                "scores higher on both (shaded: the scores that would)", pts[d], xytext=(0.03, 0.9),
                textcoords="axes fraction", fontsize=12, arrowprops=dict(arrowstyle="-", color="#1f63b4", lw=1.0))
    ax.set_xlabel("twist score, 40 mm sphere")
    ax.set_ylabel("press score, 25 mm cylinder")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(color="#eeeeee")
    ax.legend(loc="lower left", fontsize=11.5, frameon=True, framealpha=0.92, edgecolor="none")
    fig.tight_layout()
    fig.savefig(PARETO_FIG, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"{len(keys)} layouts reach both poses; {len(front)} on the front; wrote", PARETO_FIG.relative_to(L.ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--figure", action="store_true", help="only the figures, from the rows")
    a = ap.parse_args()
    if not a.figure:
        compare()
    figure()
    pareto()
    return 0


if __name__ == "__main__":
    sys.exit(main())
