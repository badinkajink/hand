#!/usr/bin/env python3
"""Tasks whose best finger layouts differ: one layout for all of them (a generalist), or a change of layout between
tasks (owner, 2026-11-03 20:3x: "how can we accommodate tasks with conflicting morphology objectives? can we linearly
interpolate? if not, is there some rule we can establish (pareto front) for when blending, however we do it, is
optimal, vs when we really should switch morphologies"; 22:3x: "we shouldnt consider time, only performance. for
desired performance, we should be able to mathematically formulate the blend vs switch decision, given the current
layout"; "there can be many capable generalists, just like many human hands can all be very capable").

The SR2 hand moves its finger bases on leadscrew gantries, so it can change layout between tasks. Two layout sets:
  lattice  the symmetric tripods (x_sep, y_sep) on the 5 mm lattice of fp_sr2_toy.grid2d, which the HOM grids scored on
           four tasks: the twist and the roll of a 40 mm sphere, the pin-push and the press of a 25 mm cylinder. A
           layout's score f_t(x) is S1: the best grasp of its grasp set at HOM's defaults (--plant picks the grid);
  sizes    the 8,192 Sobol layouts of the six gantry coordinates (rows/sizes_v4bff, work plant at 10 MPa) on 16 tasks:
           twist and roll at 30, 40, 50 and 63 mm, pin-push and press at 10, 16, 25 and 40 mm; one grasp per layout,
           the foundational pose.
Scores are hom_landscape.rollout_score averaged over seeds 0-2; a layout whose pose for a task is not reached scores 0.

The decision, on performance alone. Desired performance is a threshold tau_t per task, here a level alpha of each
task's best: tau_t = alpha f_t*, f_t* = max_x f_t(x). Then
  capable tasks of x    C(x) = {t : f_t(x) >= tau_t};
  generalists           G = {x : C(x) = T}, the layouts that do every task at the desired performance; N_G = |G|;
  decision at x0        stay if C(x0) = T; else move once to a generalist if G is not empty; else the fewest layouts
                        K* (a minimum set cover of the tasks by the sets C(x), exact) of which x0 can remain one: the
                        tasks in C(x0) stay on x0 and the others need min cover of T - C(x0) more layouts;
  alpha*_K              the highest level that K layouts reach on every task; alpha*_1 = max_x min_t f_t(x) / f_t*,
                        the best generalist's worst task. A generalist exists for alpha <= alpha*_1;
  incompatible tasks    pairs with no layout capable of both at alpha; a set of mutually incompatible tasks needs one
                        layout each, so its size bounds K* from below. Pairwise compatible tasks can still lack a
                        common layout when three or more are combined.
Richer adaptation per task (S2's grasp, S3's forces and contact modes, hom_grasp_refine.py) raises f_t(x) and so
enlarges G at a given alpha.

Secondary quantities (lattice), for task weights p (equal unless given):
  specialist   a_t = argmax_x f_t(x) (ties go to the layout with the better mean over the other tasks);
  J_K          the best K layouts with each task run on the better of them: the maximum over partitions of the tasks
               into K groups of sum_g max_x sum_{t in g} p_t f_t(x) (exact); J_T = sum_t p_t f_t*;
  pair curves  for tasks A, B and a weight w on A: the best layout's w f_A + (1 - w) f_B, the better specialist's, and
               w f_A* + (1 - w) f_B*. Blend gain = best minus the better specialist (> 0 when a third layout beats both);
  segment      the lattice layouts nearest to (1 - l) a_A + l a_B, l = 0, 0.05, ..., 1, the straight line between two
               specialists; its dominance margin is the largest amount by which one layout beats a segment layout on
               both tasks (0 on the Pareto front).

    python3 scripts/hom_task_conflict.py scope [--plant sv25]      # <date>-conflict_<plant>.json in OUT
    python3 scripts/hom_task_conflict.py scope --grid sizes        # <date>-conflict_sizes.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/experiments/20261103-hom_task_conflict"
GRID_ROWS = ROOT / "docs/experiments/20261029-foundational_pose_review/rows"
SIZES = GRID_ROWS / "sizes_v4bff"
TASKS = {"twist": 40, "roll": 40, "pinpush": 25, "press": 25}     # object diameter (mm): sphere, sphere, cylinder x2
GRIDS = {"sv25": "_sv25", "k21": "_k21", "k21mu0p4": "_k21mu0p4", "work10": ""}   # hom_grasp_refine.PLANTS' row tags
SEEDS = (0, 1, 2)
LATTICE = (40.0, 50.0, 5.0)                                        # fp_sr2_toy.LO2 and the grid's step (mm)
WEIGHTS = np.linspace(0.0, 1.0, 21)
STEPS = np.linspace(0.0, 1.0, 21)
ALPHAS = np.round(np.arange(0.50, 1.0001, 0.01), 2)
SHOW = (0.6, 0.7, 0.8, 0.9, 0.95)


# ------------------------------------------------------------------------------------------------ the score matrices

def s1(tag, task, seeds=SEEDS):
    """{call: (best grasp, its mean score over seeds, (x_sep, y_sep))} of one grid (hom_grasp_refine.s1_table for any
    grid tag)."""
    import hom_landscape as L
    p = GRID_ROWS / f"grid2dff_v4b{tag}" / f"{task}_d{TASKS[task]:g}.jsonl"
    sc, xy = {}, {}
    for x in p.read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            if int(r["seed"]) in seeds:
                c = int(r["call"])
                sc.setdefault(c, {}).setdefault(int(r.get("grasp", 0)), []).append(L.rollout_score(task, r))
                xy[c] = (r["d"][2] - r["d"][0], r["d"][3] - r["d"][5])
    out = {}
    for c, g in sc.items():
        means = {j: float(np.mean(v)) for j, v in g.items() if len(v) >= len(seeds)}
        if means:
            j = max(sorted(means), key=lambda k: means[k])
            out[c] = (j, means[j], xy[c])
    return out


def matrix(plant):
    """The lattice layouts reached for at least one task: task names, calls, xy (n, 2) in mm, F (n, T) S1 by task (0
    where the task's pose is not reached)."""
    per = {t: s1(GRIDS[plant], t) for t in TASKS}
    calls = sorted(set().union(*per.values()))
    xy = np.array([next(per[t][c][2] for t in TASKS if c in per[t]) for c in calls], float)
    F = np.array([[per[t][c][1] if c in per[t] else 0.0 for t in TASKS] for c in calls])
    return list(TASKS), calls, xy, F


def sizes_matrix():
    """The size grid: task names <task>_d<mm> by task and size, calls, d6 (n, 6) in mm, F (n, T) (0 where not reached)."""
    import hom_landscape as L
    files = [p for p in SIZES.glob("*_d*.jsonl") if not p.name.endswith("_timing.jsonl")]
    order = {t: k for k, t in enumerate(TASKS)}
    files.sort(key=lambda p: (order[p.stem.split("_d")[0]], float(p.stem.split("_d")[1])))
    per, d6 = {}, {}
    for p in files:
        task = p.stem.split("_d")[0]
        sc = {}
        for x in p.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                if int(r["seed"]) in SEEDS:
                    c = int(r["call"])
                    sc.setdefault(c, []).append(L.rollout_score(task, r))
                    d6.setdefault(c, r["d"])
        per[p.stem] = {c: float(np.mean(v)) for c, v in sc.items() if len(v) >= len(SEEDS)}
    names = list(per)
    calls = sorted(set().union(*(set(v) for v in per.values())))
    F = np.array([[per[n].get(c, 0.0) for n in names] for c in calls])
    return names, calls, np.array([d6[c] for c in calls], float), F


# ------------------------------------------------------------------------------------------------ the decision

def capable(F, alpha):
    """(n, T): True where a layout reaches alpha of the task's best."""
    return F >= alpha * F.max(0) - 1e-12


def min_cover(M, among=None):
    """The fewest layouts whose capable tasks together are every task of the mask `among` (all by default): an exact set
    cover by breadth-first search over task masks, on the layouts whose task sets no other layout's contains. Returns
    (K, [layout indices]), or (None, []) if a task has no capable layout."""
    T = M.shape[1]
    full = (1 << T) - 1 if among is None else among
    if full == 0:
        return 0, []
    masks = (M.astype(np.int64) * (1 << np.arange(T, dtype=np.int64))).sum(1) & full
    if int(np.bitwise_or.reduce(masks)) != full:
        return None, []
    vals, first = np.unique(masks, return_index=True)
    order = np.argsort([-bin(int(v)).count("1") for v in vals], kind="stable")
    vals, first = vals[order], first[order]
    keep = [k for k in range(len(vals)) if vals[k] and not np.any((vals[:k] | vals[k]) == vals[:k])]
    A, idx = vals[keep].tolist(), first[keep].tolist()
    seen, frontier = {0: None}, [0]
    for K in range(1, T + 1):
        nxt = []
        for s in frontier:
            for a, i in zip(A, idx):
                t = s | a
                if t not in seen:
                    seen[t] = (s, i)
                    nxt.append(t)
        if full in seen:
            picks, s = [], full
            while seen[s] is not None:
                s, i = seen[s]
                picks.append(i)
            return K, picks[::-1]
        frontier = nxt
    return None, []


def incompatible(M):
    """Pairs of tasks that no layout does both of, and the largest set of mutually incompatible tasks (a lower bound on
    K*)."""
    T = M.shape[1]
    bad = {(i, j) for i, j in itertools.combinations(range(T), 2) if not np.any(M[:, i] & M[:, j])}
    best = [0] if T else []
    for r in range(2, T + 1):
        hit = next((list(c) for c in itertools.combinations(range(T), r)
                    if all(p in bad for p in itertools.combinations(c, 2))), None)
        if hit is None:
            break
        best = hit
    return sorted(bad), best


def decide(F, X, i0, alpha):
    """The decision at layout i0 for desired performance alpha: stay, move once to a generalist, or the fewest layouts
    with i0 kept for the tasks it does."""
    M = capable(F, alpha)
    T = F.shape[1]
    met = [t for t in range(T) if M[i0, t]]
    gen = np.flatnonzero(M.all(1))
    out = {"alpha": float(alpha), "met": met, "n_generalists": int(len(gen))}
    if len(met) == T:
        return out | {"action": "stay"}
    if len(gen):
        k = int(gen[np.argmin(np.linalg.norm(X[gen] - X[i0], axis=1))])
        return out | {"action": "move to a generalist", "nearest": k,
                      "distance_mm": round(float(np.linalg.norm(X[k] - X[i0])), 1)}
    rest = sum(1 << t for t in range(T) if not M[i0, t])
    k_rest, cover = min_cover(M, among=rest)
    return out | {"action": "switch", "extra_layouts": k_rest, "cover": cover, "K_star": min_cover(M)[0]}


def coverage(F, names):
    """Per alpha of ALPHAS: generalists, K*, its cover, incompatible pairs; alpha*_K for K up to K* at alpha = 1."""
    rows, astar = [], {}
    gamma = (F / F.max(0)).min(1)
    for alpha in ALPHAS:
        M = capable(F, alpha)
        K, cover = min_cover(M)
        bad, clique = incompatible(M)
        rows.append({"alpha": float(alpha), "n_generalists": int((gamma >= alpha - 1e-12).sum()), "K_star": K,
                     "cover": cover, "incompatible": [[names[i], names[j]] for i, j in bad],
                     "mutually_incompatible": [names[i] for i in clique]})
        if K is not None:
            astar[K] = max(astar.get(K, 0.0), float(alpha))
    for K in range(1, max(astar) + 1):                     # alpha*_K: the highest alpha with K* <= K
        astar[K] = max([a for k, a in astar.items() if k <= K], default=None)
    return {"alpha_star": {str(k): astar[k] for k in sorted(astar)}, "best_generalist_worst": round(float(gamma.max()), 4),
            "levels": rows}


# ------------------------------------------------------------------------------------------------ secondary (lattice)

def partitions(items, k):
    """Every partition of items into k non-empty groups."""
    if k == 1:
        yield [list(items)]
        return
    if len(items) == k:
        yield [[i] for i in items]
        return
    first, rest = items[0], items[1:]
    for p in partitions(rest, k - 1):
        yield [[first]] + p
    for p in partitions(rest, k):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]


def best_k(F, p, k):
    """J_K and its layouts: (value, [(task group, layout index)])."""
    best = None
    for part in partitions(list(range(F.shape[1])), k):
        val, picks = 0.0, []
        for g in part:
            s = F[:, g] @ p[g]
            i = int(np.argmax(s))
            val += float(s[i])
            picks.append((g, i))
        if best is None or val > best[0] + 1e-12:
            best = (val, picks)
    return best


def specialist(F, t):
    """The best layout of task t; ties go to the better mean over the other tasks."""
    top = np.flatnonzero(F[:, t] >= F[:, t].max() - 1e-9)
    rest = np.delete(F[top], t, axis=1).mean(1)
    return int(top[np.argmax(rest)])


def front(fa, fb):
    """Indices of the Pareto front of two scores (both maximized), by falling fa."""
    order = np.lexsort((-fb, -fa))
    out, top = [], -np.inf
    for i in order:
        if fb[i] > top + 1e-12:
            out.append(int(i))
            top = fb[i]
    return out


def margin(fa, fb, i, idx):
    """The largest amount by which one layout of idx beats layout i on both scores (0 if none beats it on both)."""
    return float(max(0.0, np.max(np.minimum(fa[idx] - fa[i], fb[idx] - fb[i]))))


def nearest(xy, point):
    """Index of the grid layout at the lattice point nearest to point, or None if no task reached it."""
    x0, y0, step = LATTICE
    q = (x0 + step * round((point[0] - x0) / step), y0 + step * round((point[1] - y0) / step))
    hit = np.flatnonzero(np.all(np.abs(xy - q) < 1e-6, axis=1))
    return int(hit[0]) if len(hit) else None


def pair(F, xy, calls, names, i, j):
    """Pair curves, front and segment of tasks i and j."""
    from scipy.stats import spearmanr
    fa, fb = F[:, i], F[:, j]
    a, b = specialist(F, i), specialist(F, j)
    allidx = np.arange(len(F))
    curves = []
    for w in WEIGHTS:
        s = w * fa + (1 - w) * fb
        k = int(np.argmax(s))
        curves.append({"w": round(float(w), 3), "best": round(float(s[k]), 4), "best_call": calls[k],
                       "specialist": round(float(max(s[a], s[b])), 4),
                       "utopia": round(float(w * fa[a] + (1 - w) * fb[b]), 4)})
    seg, seen = [], set()
    for lam in STEPS:
        k = nearest(xy, (1 - lam) * xy[a] + lam * xy[b])
        if k is None:
            seg.append({"l": round(float(lam), 3), "call": None, "fa": 0.0, "fb": 0.0, "margin": None})
            continue
        if k in seen:
            continue
        seen.add(k)
        seg.append({"l": round(float(lam), 3), "call": calls[k], "xy": xy[k].tolist(), "fa": round(float(fa[k]), 4),
                    "fb": round(float(fb[k]), 4), "margin": round(margin(fa, fb, k, allidx), 4)})
    both = (fa > 0) & (fb > 0)
    return {"tasks": [names[i], names[j]],
            "a": {"call": calls[a], "xy": xy[a].tolist(), "fa": round(float(fa[a]), 4), "fb": round(float(fb[a]), 4)},
            "b": {"call": calls[b], "xy": xy[b].tolist(), "fa": round(float(fa[b]), 4), "fb": round(float(fb[b]), 4)},
            "distance_mm": round(float(np.linalg.norm(xy[a] - xy[b])), 1),
            "rho": round(float(spearmanr(fa[both], fb[both]).statistic), 3), "n_both": int(both.sum()),
            "blend_gain_max": round(max(c["best"] - c["specialist"] for c in curves), 4),
            "switch_gain_half": next(round(c["utopia"] - c["best"], 4) for c in curves if abs(c["w"] - 0.5) < 1e-9),
            "segment_margin_max": round(max(s["margin"] for s in seg if s["margin"] is not None), 4),
            "curves": curves, "front": [[calls[k], round(float(fa[k]), 4), round(float(fb[k]), 4)] for k in fr]
            if (fr := front(fa, fb)) else [], "segment": seg}


# ------------------------------------------------------------------------------------------------ scope

def breadth(F, names):
    """alpha*_1 of each task type over windows of consecutive sizes (1 to 4 sizes), and over all tasks."""
    N = F / F.max(0)
    out = {}
    for task in TASKS:
        cols = [k for k, n in enumerate(names) if n.split("_d")[0] == task]
        out[task] = {}
        for L_ in range(1, len(cols) + 1):
            out[task][L_] = [{"sizes": [names[c].split("_d")[1] for c in cols[s:s + L_]],
                              "alpha1": round(float(N[:, cols[s:s + L_]].min(1).max()), 4)}
                             for s in range(len(cols) - L_ + 1)]
    return out


def cmd_scope(a):
    t0 = time.time()
    if a.grid == "sizes":
        names, calls, X, F = sizes_matrix()
        tag = "sizes"
    else:
        names, calls, X, F = matrix(a.plant)
        tag = a.plant
    T = F.shape[1]
    fstar = F.max(0)
    spec = [specialist(F, t) for t in range(T)]
    out = {"grid": "sizes_v4bff" if a.grid == "sizes" else f"grid2dff_v4b{GRIDS[a.plant]}", "seeds": SEEDS,
           "tasks": names, "n_layouts": len(calls), "n_all_tasks": int((F > 0).all(1).sum()),
           "best": [round(float(v), 4) for v in fstar],
           "specialists": [{"task": names[t], "call": calls[k], "x": X[k].tolist(), "scores": F[k].round(4).tolist(),
                            "ties": int((F[:, t] >= fstar[t] - 1e-9).sum())} for t, k in enumerate(spec)],
           "coverage": coverage(F, names)}
    for lev in out["coverage"]["levels"]:
        lev["cover"] = [{"call": calls[i], "x": X[i].tolist(), "tasks": [names[t] for t in range(T)
                                                                           if F[i, t] >= lev["alpha"] * fstar[t] - 1e-12]}
                        for i in lev["cover"]]
    starts = {f"best for {names[t]}": spec[t] for t in range(T)}
    gamma = (F / fstar).min(1)
    starts["best generalist"] = int(np.argmax(gamma))
    out["decisions"] = []
    for alpha in SHOW:
        for label, i0 in starts.items():
            d = decide(F, X, i0, alpha)
            d = d | {"x0": label, "x0_call": calls[i0], "met": [names[t] for t in d["met"]]}
            if "nearest" in d:
                d["nearest"] = calls[d["nearest"]]
            if "cover" in d:
                d["cover"] = [calls[i] for i in d["cover"]]
            out["decisions"].append(d)
    if a.grid == "sizes":
        out["breadth"] = breadth(F, names)
    else:
        p = np.full(T, 1.0 / T) if not a.weights else np.asarray(a.weights, float) / np.sum(a.weights)
        out["weights"] = p.round(4).tolist()
        out["J"] = []
        for k in range(1, T + 1):
            val, picks = best_k(F, p, k)
            out["J"].append({"K": k, "J": round(val, 4), "layouts": [{"tasks": [names[t] for t in g], "call": calls[i],
                                                                      "x": X[i].tolist(), "scores": F[i].round(4).tolist()}
                                                                     for g, i in picks]})
        out["pairs"] = [pair(F, X, calls, names, i, j) for i, j in itertools.combinations(range(T), 2)]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{time.strftime('%Y%m%d')}-conflict_{tag}.json"
    path.write_text(json.dumps(out, indent=1))
    cov = out["coverage"]
    print(f"{tag}: {len(calls)} layouts, {T} tasks ({out['n_all_tasks']} layouts reach every task), "
          f"{time.time() - t0:.0f} s -> {path.relative_to(ROOT)}")
    print(f"  best generalist's worst task {cov['best_generalist_worst']:.3f} of its best; alpha*_K "
          + ", ".join(f"K={k}: {v}" for k, v in cov["alpha_star"].items()))
    for lev in cov["levels"]:
        if lev["alpha"] in SHOW:
            print(f"  alpha {lev['alpha']:.2f}: {lev['n_generalists']} generalists, K* {lev['K_star']}, "
                  f"{len(lev['incompatible'])} incompatible pairs, largest mutually incompatible set "
                  f"{lev['mutually_incompatible']}")
    if "breadth" in out:
        for task, by in out["breadth"].items():
            print(f"  {task}: alpha*_1 over windows of 1-4 sizes: "
                  + "; ".join(f"{L_}: " + " ".join(f"{w['alpha1']:.2f}" for w in ws) for L_, ws in by.items()))
    else:
        print("  J_K " + ", ".join(f"{r['K']}: {r['J']:.3f}" for r in out["J"]))


def main():
    ap = argparse.ArgumentParser(description="One finger layout for several HOM tasks, or a change of layout")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("scope")
    ps.add_argument("--grid", default="lattice", choices=("lattice", "sizes"))
    ps.add_argument("--plant", default="sv25", choices=list(GRIDS), help="the lattice grid's plant")
    ps.add_argument("--weights", type=float, nargs=len(TASKS), default=None, help="task weights in TASKS order")
    a = ap.parse_args()
    {"scope": cmd_scope}[a.cmd](a)


if __name__ == "__main__":
    main()
