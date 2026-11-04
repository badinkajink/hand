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

Validation (2026-11-04): the lattice scores above are means over seeds 0-2, and each task's best is the best of up to
561 such means, so it is biased upward. `validate` runs every layout of a plant's grid on every task it reaches, with
its S1 grasp, at seeds 20-24, which no grid or search used, on the GPU evaluator of the grids (hom_grasp_refine.setenv
and evaluate, one hom_gpu.batch_child process per batch). Rows: OUT/rows/<plant>/val/<task>_d<d>.jsonl in hom_gpu.py's
format (key: call, pick "S1", grasp, v), _timing.jsonl one line per GPU batch, _runs.jsonl one line per run (open-pose
and GPU-batch seconds, batches, rollouts, wall seconds).

    python3 scripts/hom_task_conflict.py scope [--plant sv25]      # <date>-conflict_<plant>.json in OUT
    python3 scripts/hom_task_conflict.py scope --grid sizes        # <date>-conflict_sizes.json
    python3 scripts/hom_task_conflict.py validate --plant sv25 [--tasks twist roll]   # through resguard (queue.sh)
    python3 scripts/hom_task_conflict.py scope --plant sv25 --val   # chosen at seeds 0-2, scored at 20-24; smoothed

Shared layouts refined (2026-11-04): S2 and S3 of hom_grasp_refine.py on at most N_SHARED layouts (shared_layouts():
the specialists, the J_K layouts, the max-min layout, the alpha*_K covers, the smoothed J_1 layout and the score-refine
study's best refined layout of each task), every task each reaches, rows under OUT/rows_refine/<plant>/ in that
script's layout; `refined` recomputes J_K and the coverage quantities at S1, S2 and S3 (seeds 20-24).

    python3 scripts/hom_task_conflict.py refine layouts --plant sv25           # REFINE/<plant>/<task>_layouts.json
    python3 scripts/hom_task_conflict.py refine run --task twist --level grasp # then --level full; through resguard
    python3 scripts/hom_task_conflict.py refine validate --task twist          # S1, S2 and S3 picks at seeds 20-24
    python3 scripts/hom_task_conflict.py refine report                         # hom_grasp_refine's report there
    python3 scripts/hom_task_conflict.py refined --plant sv25                  # <date>-conflict_refined_<plant>.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
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
VAL_SEEDS = (20, 21, 22, 23, 24)                                   # hom_grasp_refine.VAL_SEEDS
ROWS = OUT / "rows"
LATTICE = (40.0, 50.0, 5.0)                                        # fp_sr2_toy.LO2 and the grid's step (mm)
LATTICE_HI = (160.0, 170.0)                                        # fp_sr2_toy.HI2
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
    where the task's pose is not reached), R (n, T) True where it is reached."""
    per = {t: s1(GRIDS[plant], t) for t in TASKS}
    calls = sorted(set().union(*per.values()))
    xy = np.array([next(per[t][c][2] for t in TASKS if c in per[t]) for c in calls], float)
    F = np.array([[per[t][c][1] if c in per[t] else 0.0 for t in TASKS] for c in calls])
    R = np.array([[c in per[t] for t in TASKS] for c in calls])
    return list(TASKS), calls, xy, F, R


def val_matrix(plant, calls, R):
    """(n, T): each layout's S1 grasp at VAL_SEEDS (validate's rows), 0 where the task's pose is not reached."""
    import hom_landscape as L
    F = np.zeros(R.shape)
    row = {c: k for k, c in enumerate(calls)}
    for t, task in enumerate(TASKS):
        acc = {}
        p = val_path(plant, task)
        for x in (p.read_text().splitlines() if p.exists() else []):
            if x.strip():
                r = json.loads(x)
                if int(r["seed"]) in VAL_SEEDS:
                    acc.setdefault(int(r["call"]), []).append(L.rollout_score(task, r))
        missing = [c for c in calls if R[row[c], t] and len(acc.get(c, [])) < len(VAL_SEEDS)]
        if missing:
            raise SystemExit(f"{plant} {task}: {len(missing)} reached layouts lack seeds {VAL_SEEDS} (run validate)")
        for c, v in acc.items():
            F[row[c], t] = float(np.mean(v))
    return F


def smooth(F, xy, R):
    """Each layout's score the mean over its 3 x 3 block of the lattice (clipped at the lattice's edges; a neighbour
    that does not reach the task scores 0), and 0 where the layout itself does not reach the task: a stand-in for
    seed noise and for a gantry that places the fingers within one lattice step (5 mm) of the commanded layout."""
    x0, y0, step = LATTICE
    nx = int(round((LATTICE_HI[0] - x0) / step)) + 1
    ny = int(round((LATTICE_HI[1] - y0) / step)) + 1
    ix = np.rint((xy[:, 0] - x0) / step).astype(int)
    iy = np.rint((xy[:, 1] - y0) / step).astype(int)
    G = np.zeros((nx + 2, ny + 2, F.shape[1]))
    G[ix + 1, iy + 1] = F
    ones = np.zeros((nx + 2, ny + 2))
    ones[1:-1, 1:-1] = 1.0
    S, n = np.zeros_like(G), np.zeros_like(ones)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            S[1:-1, 1:-1] += G[1 + dx:nx + 1 + dx, 1 + dy:ny + 1 + dy]
            n[1:-1, 1:-1] += ones[1 + dx:nx + 1 + dx, 1 + dy:ny + 1 + dy]
    return np.where(R, S[ix + 1, iy + 1] / n[ix + 1, iy + 1, None], 0.0)


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
    """J_K and its layouts: (value, [(task group, layout index)]); a group's ties go to the layout with the better mean
    over all tasks, as the specialists' do."""
    best = None
    for part in partitions(list(range(F.shape[1])), k):
        val, picks = 0.0, []
        for g in part:
            s = F[:, g] @ p[g]
            top = np.flatnonzero(s >= s.max() - 1e-9)
            i = int(top[np.argmax(F[top].mean(1))])
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


# ------------------------------------------------------------------------------------------------ validation of choices

def r4(v):
    return np.round(np.asarray(v, float), 4).tolist()


def chosen(Fc, Fs, X, calls, names, p, pairs=True):
    """The choices made on the score matrix Fc, each also scored on Fs, the same layouts at other seeds or smoothed (Fc
    = Fs: in sample). Each task's best on Fs is its specialist's score there, so a level alpha on Fs is a fraction of
    what the chosen specialist reaches (Fs's own best where the specialist scores 0 on Fs). Choices: the specialists; J_K's layouts on raw scores and on scores normalized by
    each task's best; the max-min layout; the cover of alpha*_K for each K and the covers at the levels SHOW; per pair of
    tasks the best layout of each weighted sum, the front and the straight segment between the specialists. A segment
    layout's margin on Fs is taken against the front chosen on Fc."""
    from scipy.stats import spearmanr
    T = Fc.shape[1]
    spec = [specialist(Fc, t) for t in range(T)]
    bc, bs = Fc.max(0), Fs[spec, np.arange(T)]
    zero = [names[t] for t in range(T) if bs[t] <= 1e-9]      # a specialist that scores 0 on Fs: Fs's own best instead
    ref = np.where(bs > 1e-9, bs, Fs.max(0))
    Nc, Ns = Fc / bc, Fs / np.where(ref > 1e-9, ref, np.inf)

    def lay(i, **kw):
        return {"call": calls[i], "x": X[i].tolist(), "choose": r4(Fc[i]), "score": r4(Fs[i])} | kw

    out = {"best_choose": r4(bc), "best_score": r4(bs), "specialist_zero_on_score": zero,
           "specialists": [lay(k, task=names[t]) for t, k in enumerate(spec)]}
    for key, (A, B) in {"J": (Fc, Fs), "J_norm": (Nc, Ns)}.items():
        out[key] = []
        for k in range(1, T + 1):
            val, picks = best_k(A, p, k)
            out[key].append({"K": k, "choose": round(val, 4),
                             "score": round(float(sum(B[i, g] @ p[g] for g, i in picks)), 4),
                             "layouts": [lay(i, tasks=[names[t] for t in g]) for g, i in picks]})
        out[key + "_T_over_1"] = {s: round(out[key][-1][s] / out[key][0][s], 4) if out[key][0][s] > 0 else None
                                  for s in ("choose", "score")}
    g = int(np.argmax(Nc.min(1)))
    out["max_min"] = lay(g, ratio_choose=r4(Nc[g]), ratio_score=r4(Ns[g]), alpha_choose=round(float(Nc[g].min()), 4),
                         alpha_score=round(float(Ns[g].min()), 4))
    cov = coverage(Fc, names)
    level = {lev["alpha"]: lev for lev in cov["levels"]}

    def reach(cover):
        return round(float(Ns[cover].max(0).min()), 4) if len(cover) else None

    out["alpha_star"] = {}
    for k, a in cov["alpha_star"].items():
        cover = level[a]["cover"]
        out["alpha_star"][k] = {"choose": a, "score": reach(cover), "layouts": [lay(i) for i in cover]}
    gen_c, gen_s = Nc.min(1), Ns.min(1)
    out["levels"] = []
    for alpha in SHOW:
        G = np.flatnonzero(gen_c >= alpha - 1e-12)
        cover = level[alpha]["cover"]
        out["levels"].append({"alpha": alpha, "n_generalists_choose": int(len(G)),
                              "n_generalists_kept": int((gen_s[G] >= alpha - 1e-12).sum()),
                              "K_star_choose": level[alpha]["K_star"], "cover_score": reach(cover),
                              "cover": [lay(i) for i in cover]})
    out["pairs"] = []
    for i, j in itertools.combinations(range(T), 2) if pairs else ():
        a, b = spec[i], spec[j]
        fr = front(Fc[:, i], Fc[:, j])
        curves = []
        for w in WEIGHTS:
            sc, ss = w * Fc[:, i] + (1 - w) * Fc[:, j], w * Fs[:, i] + (1 - w) * Fs[:, j]
            k = int(np.argmax(sc))
            sp = a if sc[a] >= sc[b] else b
            curves.append({"w": round(float(w), 3), "call": calls[k], "best_choose": round(float(sc[k]), 4),
                           "best_score": round(float(ss[k]), 4), "specialist_choose": round(float(sc[sp]), 4),
                           "specialist_score": round(float(ss[sp]), 4),
                           "utopia_choose": round(float(w * bc[i] + (1 - w) * bc[j]), 4),
                           "utopia_score": round(float(w * bs[i] + (1 - w) * bs[j]), 4)})
        seg, seen = [], set()
        for lam in STEPS:
            k = nearest(X, (1 - lam) * X[a] + lam * X[b])
            if k is None or k in seen:
                continue
            seen.add(k)
            seg.append({"l": round(float(lam), 3), "call": calls[k], "x": X[k].tolist(),
                        "choose": r4(Fc[k, [i, j]]), "score": r4(Fs[k, [i, j]]),
                        "margin_choose": round(margin(Fc[:, i], Fc[:, j], k, fr), 4),
                        "margin_score": round(margin(Fs[:, i], Fs[:, j], k, fr), 4)})
        rho = {}
        for s, M in (("choose", Fc), ("score", Fs)):
            both = (M[:, i] > 0) & (M[:, j] > 0)
            rho[s] = round(float(spearmanr(M[both, i], M[both, j]).statistic), 3) if both.sum() > 2 else None
        half = next(c for c in curves if abs(c["w"] - 0.5) < 1e-9)
        out["pairs"].append({
            "tasks": [names[i], names[j]], "a": lay(a), "b": lay(b),
            "distance_mm": round(float(np.linalg.norm(X[a] - X[b])), 1), "rho": rho,
            "blend_gain_max": {s: round(max(c[f"best_{s}"] - c[f"specialist_{s}"] for c in curves), 4)
                               for s in ("choose", "score")},
            "switch_gain_half": {s: round(half[f"utopia_{s}"] - half[f"best_{s}"], 4) for s in ("choose", "score")},
            "segment_margin_max": {s: round(max([x[f"margin_{s}"] for x in seg] or [0.0]), 4)
                                   for s in ("choose", "score")},
            "curves": curves, "front": [lay(k) for k in fr], "segment": seg})
    return out


def noise(F0, F1, R, names):
    """Per task, over the layouts that reach it: Spearman's rho between the two seed sets' scores, the median and the
    95th percentile of their absolute difference, and the regression of the best at seeds 0-2 at the other seeds."""
    from scipy.stats import spearmanr
    out = {}
    for t, n in enumerate(names):
        m = R[:, t]
        d = np.abs(F1[m, t] - F0[m, t])
        k = specialist(F0, t)
        out[n] = {"n": int(m.sum()), "rho": round(float(spearmanr(F0[m, t], F1[m, t]).statistic), 3),
                  "abs_diff_median": round(float(np.median(d)), 4), "abs_diff_p95": round(float(np.percentile(d, 95)), 4),
                  "mean_search": round(float(F0[m, t].mean()), 4), "mean_val": round(float(F1[m, t].mean()), 4),
                  "best_search": round(float(F0[k, t]), 4), "best_search_at_val": round(float(F1[k, t]), 4),
                  "best_val": round(float(F1[:, t].max()), 4)}
    return out


def decisions(F, X, calls, names):
    """The decision at each specialist and at the best generalist for the levels SHOW."""
    T = F.shape[1]
    spec = [specialist(F, t) for t in range(T)]
    starts = {f"best for {names[t]}": spec[t] for t in range(T)}
    starts["best generalist"] = int(np.argmax((F / F.max(0)).min(1)))
    out = []
    for alpha in SHOW:
        for label, i0 in starts.items():
            d = decide(F, X, i0, alpha)
            d = d | {"x0": label, "x0_call": calls[i0], "met": [names[t] for t in d["met"]]}
            if "nearest" in d:
                d["nearest"] = calls[d["nearest"]]
            if "cover" in d:
                d["cover"] = [calls[i] for i in d["cover"]]
            out.append(d)
    return out


def named(cov, F, X, calls, names):
    """coverage()'s covers as calls, positions and the tasks each layout reaches at the level."""
    fstar = F.max(0)
    for lev in cov["levels"]:
        lev["cover"] = [{"call": calls[i], "x": X[i].tolist(),
                         "tasks": [names[t] for t in range(F.shape[1]) if F[i, t] >= lev["alpha"] * fstar[t] - 1e-12]}
                        for i in lev["cover"]]
    return cov


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
        names, calls, X, F, R = matrix(a.plant)
        tag = a.plant
    T = F.shape[1]
    fstar = F.max(0)
    spec = [specialist(F, t) for t in range(T)]
    out = {"grid": "sizes_v4bff" if a.grid == "sizes" else f"grid2dff_v4b{GRIDS[a.plant]}", "seeds": SEEDS,
           "tasks": names, "n_layouts": len(calls), "n_all_tasks": int((F > 0).all(1).sum()),
           "best": [round(float(v), 4) for v in fstar],
           "specialists": [{"task": names[t], "call": calls[k], "x": X[k].tolist(), "scores": F[k].round(4).tolist(),
                            "ties": int((F[:, t] >= fstar[t] - 1e-9).sum())} for t, k in enumerate(spec)],
           "coverage": named(coverage(F, names), F, X, calls, names), "decisions": decisions(F, X, calls, names)}
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
    if a.val:
        F1 = val_matrix(a.plant, calls, R)
        S0, S1 = smooth(F, X, R), smooth(F1, X, R)
        mats = {"search": (F, F), "val": (F, F1), "val_only": (F1, F1), "smooth": (S0, S1)}
        out["val"] = {
            "seeds": VAL_SEEDS, "smooth": "mean over the 3 x 3 lattice block (5 mm steps), 0 where the task is not reached",
            "variants": {"search": "chosen and scored at seeds 0-2", "val": "chosen at seeds 0-2, scored at seeds 20-24",
                         "val_only": "chosen and scored at seeds 20-24",
                         "smooth": "chosen on the smoothed scores at seeds 0-2, scored smoothed at seeds 20-24"},
            "noise": noise(F, F1, R, names),
            "coverage": {k: named(coverage(M, names), M, X, calls, names)
                         for k, M in (("val", F1), ("smooth_search", S0), ("smooth_val", S1))},
            "decisions": {"val": decisions(F1, X, calls, names), "smooth_val": decisions(S1, X, calls, names)},
            "chosen": {k: chosen(Fc, Fs, X, calls, names, p) for k, (Fc, Fs) in mats.items()},
            "matrix": {"calls": calls, "x": X.tolist(), "reach": R.astype(int).tolist(), "search": r4(F), "val": r4(F1)}}
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
    if "val" in out:
        for t, n in out["val"]["noise"].items():
            print(f"  {t}: rho(seeds 0-2, 20-24) {n['rho']}, median |diff| {n['abs_diff_median']}; best at 0-2 "
                  f"{n['best_search']} -> {n['best_search_at_val']} at 20-24 (best at 20-24 {n['best_val']})")
        for k, c in out["val"]["chosen"].items():
            print(f"  {k}: J_K " + ", ".join(f"{r['K']}: {r['choose']:.3f}->{r['score']:.3f}" for r in c["J"])
                  + f"; J_T/J_1 {c['J_T_over_1']['score']}; alpha*_K "
                  + ", ".join(f"{K}: {v['choose']}->{v['score']}" for K, v in c["alpha_star"].items())
                  + "; generalists kept at " + ", ".join(f"{lev['alpha']}: {lev['n_generalists_kept']}/"
                                                         f"{lev['n_generalists_choose']}" for lev in c["levels"])
                  + "; segment margin max " + " ".join(f"{p['segment_margin_max']['score']:.2f}" for p in c["pairs"]))


# ------------------------------------------------------------------------------------------------ validation

def val_path(plant, task):
    return ROWS / plant / "val" / f"{task}_d{TASKS[task]:g}.jsonl"


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def cmd_validate(a):
    """Every layout of the plant's grid with its S1 grasp at VAL_SEEDS, task by task; a stopped run resumes from its
    rows."""
    import hom_grasp_refine as R
    R.setenv(a.plant)
    import fp_sr2_toy as FP
    import hom_ea as E
    import hom_gpu as HG
    for task in a.tasks or TASKS:
        t0 = time.time()
        FP.set_sizes(**({"sph_d_mm": TASKS[task]} if R.POSE[task] == "tripod" else {"cyl_d_mm": TASKS[task]}))
        s1, toy = R.s1_table(task, a.plant), R.toy_rows(task)
        items = []
        for c in sorted(s1):
            if c not in toy:
                continue
            j = s1[c][0]
            v = toy[c]["fp"][R.POSE[task]]["grasps"][j]["v"]
            key = {"d": toy[c]["d"], "call": c, "pick": "S1", "grasp": j, "v": [round(float(y), 7) for y in v], "prm": {}}
            items.append((key, toy[c]["d"], np.asarray(v, float), {}))
        path = val_path(a.plant, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        scene = HG.Scene("sphere" if R.POSE[task] == "tripod" else "cylinder", TASKS[task])
        tm = {}
        n = R.evaluate(task, a.plant, path, items, VAL_SEEDS, scene, HG, E, tm)
        wall = time.time() - t0
        append(path.with_name(path.stem + "_runs.jsonl"),
               {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "task": task, "plant": a.plant, "layouts": len(items),
                "seeds": list(VAL_SEEDS), "rollouts": n, **tm, "wall_s": round(wall, 2)})
        print(f"{time.strftime('%H:%M:%S')} {a.plant} {task}: {len(items)} layouts, {n} rollouts run, open poses "
              f"{tm.get('open_s', 0):.0f} s, GPU batches {tm.get('gpu_s', 0):.0f} s ({tm.get('batches', 0)}), wall "
              f"{wall:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------ shared layouts refined

REFINE = OUT / "rows_refine"
SCORE_REFINE = ROOT / "docs/experiments/20261103-hom_score_refine/rows"
N_SHARED = 10
LEVELS = ("S1", "S2", "S3")


def newest(folder, pattern):
    files = sorted(folder.glob(pattern))
    if not files:
        raise SystemExit(f"no {pattern} in {folder.relative_to(ROOT)}")
    return files[-1]


def shared_layouts(plant):
    """{call: [roles]}, at most N_SHARED layouts in this order: the specialists, the J_1 layout on raw and on normalized
    scores, the max-min layout, the layouts of J_2 and J_3 (raw, then normalized) and the covers of alpha*_2 and
    alpha*_3, all chosen at seeds 0-2 (scope --val, variant val); the J_1 layout of the smoothed scores; and the best
    layout of each task in the score-refine study after refinement (S3e at seeds 20-24, 60 layouts per task)."""
    J = json.loads(newest(OUT, f"*-conflict_{plant}.json").read_text())
    if "val" not in J:
        raise SystemExit(f"run scope --plant {plant} --val first")
    c = J["val"]["chosen"]["val"]
    roles = [(s["call"], f"specialist of {s['task']}") for s in c["specialists"]]
    roles += [(c["J"][0]["layouts"][0]["call"], "J_1"), (c["J_norm"][0]["layouts"][0]["call"], "J_1 normalized"),
              (c["max_min"]["call"], "max-min")]
    for key in ("J", "J_norm"):
        for r in c[key][1:3]:
            roles += [(L_["call"], f"{key} K={r['K']}") for L_ in r["layouts"]]
    for k in ("2", "3"):
        roles += [(L_["call"], f"cover of alpha*_{k}") for L_ in c["alpha_star"][k]["layouts"]]
    roles.append((J["val"]["chosen"]["smooth"]["J"][0]["layouts"][0]["call"], "J_1 smoothed"))
    rep = json.loads(newest(SCORE_REFINE / plant, "*-score_levels.json").read_text())
    roles += [(v["best_s3e_val"]["call"], f"refined best of {t}") for t, v in rep["tasks"].items()]
    out = {}
    for call, role in roles:
        if call in out:
            out[call].append(role)
        elif len(out) < N_SHARED:
            out[call] = [role]
    return out


def write_shared(plant):
    """REFINE/<plant>/<task>_layouts.json in hom_grasp_refine's format: the shared layouts that reach the task, with S1
    and its grasp from the plant's grid (group: the first role). An existing file is kept if it names the same layouts."""
    import hom_grasp_refine as R
    shared = shared_layouts(plant)
    for task in TASKS:
        s1, toy = R.s1_table(task, plant), R.toy_rows(task)
        lays = []
        for c, roles in shared.items():
            if c not in s1 or c not in toy:
                continue
            rec = toy[c]["fp"][R.POSE[task]]
            lays.append({"call": c, "xy": toy[c]["xy"], "d": toy[c]["d"], "group": roles[0], "roles": roles,
                         "s1": round(s1[c][1], 4), "s1_grasp": s1[c][0], "v1": rec["grasps"][s1[c][0]]["v"],
                         "widths": rec["grasps"][0]["widths"]})
        p = R.rows_dir(plant) / f"{task}_layouts.json"
        if p.exists():
            old = [L_["call"] for L_ in json.loads(p.read_text())["layouts"]]
            if old != [L_["call"] for L_ in lays]:
                raise SystemExit(f"{p.relative_to(ROOT)} names other layouts ({old}); move it away to start again")
            print(f"{task}: {len(lays)} layouts, sample kept")
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"task": task, "plant": plant, "n_reached": len(s1), "base": None, "shared": True,
                                 "layouts": lays}, indent=1))
        print(f"{task}: {len(lays)} layouts -> {p.relative_to(ROOT)}: " + ", ".join(
            f"{L_['call']} {tuple(L_['xy'])} ({L_['group']})" for L_ in lays))


def cmd_refine(a):
    """hom_grasp_refine's searches (levels grasp and full), validation and report on the shared layouts, with
    hom_grasp_refine.ROWS_ROOT set to REFINE (its layouts() reads the sample that `refine layouts` writes)."""
    import hom_grasp_refine as R
    R.ROWS_ROOT = REFINE
    if a.what == "layouts":
        write_shared(a.plant)
    elif a.what == "run":
        R.cmd_run(argparse.Namespace(task=a.task, level=a.level, plant=a.plant, n=0, iters=0, pop=0, top=R.N_TOP,
                                     rand=R.N_RAND, layouts_from=None))
    elif a.what == "validate":
        path = R.rows_dir(a.plant) / "val" / f"{a.task}_d{TASKS[a.task]:g}.jsonl"
        tpath = path.with_name(path.stem + "_timing.jsonl")
        n0, b0, t0 = len(R.read_rows(path)), len(R.read_rows(tpath)), time.time()
        R.cmd_validate(argparse.Namespace(task=a.task, plant=a.plant, picks_from=None))
        new = R.read_rows(tpath)[b0:]
        append(path.with_name(path.stem + "_runs.jsonl"),
               {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "task": a.task, "plant": a.plant, "seeds": list(VAL_SEEDS),
                "rollouts": len(R.read_rows(path)) - n0, "batches": len(new),
                "gpu_batch_s": round(sum(r["build_s"] + r["run_s"] + r["score_s"] for r in new), 2),
                "wall_s": round(time.time() - t0, 2)})
    elif a.what == "report":
        R.cmd_report(argparse.Namespace(plant=a.plant))


def refined_matrices(plant):
    """From the newest report of REFINE/<plant>: names, calls, xy, roles, and per level of LEVELS the scores at seeds 0-2
    (S1 the grid's, S2 and S3 the picks') and at seeds 20-24, (n, T) each with 0 where the task is not reached. The
    searches score their start as candidate 0, so the picks are elitist at seeds 0-2."""
    rep = json.loads(newest(REFINE / plant, "*-score_levels.json").read_text())
    shared = shared_layouts(plant)
    calls = list(shared)
    xy, srch, val = {}, {k: np.zeros((len(calls), len(TASKS))) for k in LEVELS}, \
        {k: np.zeros((len(calls), len(TASKS))) for k in LEVELS}
    for t, task in enumerate(TASKS):
        for e in rep["tasks"].get(task, {}).get("layouts", []):
            i = calls.index(e["call"])
            xy[e["call"]] = e["xy"]
            for k in LEVELS:
                s, v = e[k.lower()], e[f"{k.lower()}_val"]
                if s is None or v is None:
                    raise SystemExit(f"{task} call {e['call']}: {k} missing (search {s}, validation {v})")
                srch[k][i, t], val[k][i, t] = s, v
    return list(TASKS), calls, np.array([xy[c] for c in calls], float), shared, srch, val, rep


def cmd_refined(a):
    """J_K, the coverage quantities and the decision on the shared layouts at S1, S2 and S3, each chosen at seeds 0-2
    and scored at seeds 20-24 (chosen() without pairs), against each level's own best among the shared layouts; and the
    generalists at the thresholds of S1, tau_t = alpha f_t* with f_t* the S1 specialist's score at seeds 20-24, at each
    level (does the task's own grasp, forces and modes on one layout enlarge G?)."""
    names, calls, X, shared, srch, val, rep = refined_matrices(a.plant)
    T = len(names)
    p = np.full(T, 1.0 / T)
    out = {"plant": a.plant, "report": str(newest(REFINE / a.plant, "*-score_levels.json").relative_to(ROOT)),
           "tasks": names, "layouts": [{"call": c, "x": X[i].tolist(), "roles": shared[c]} for i, c in enumerate(calls)],
           "levels": {}}
    s1_spec = [specialist(srch["S1"], t) for t in range(T)]
    tau = val["S1"][s1_spec, np.arange(T)]
    for k in LEVELS:
        Fc, Fs = srch[k], val[k]
        c = chosen(Fc, Fs, X, calls, names, p, pairs=False)
        fixed = (Fs / np.where(tau > 1e-9, tau, np.inf)).min(1)
        c["search"], c["val"] = r4(Fc), r4(Fs)
        c["at_s1_thresholds"] = {"tau": r4(tau), "gamma": r4(fixed),
                                 "n_generalists": {str(al): int((fixed >= al - 1e-12).sum()) for al in SHOW}}
        c["coverage_val"] = named(coverage(Fs, names), Fs, X, calls, names)
        out["levels"][k] = c
        print(f"{k}: J_K at seeds 20-24 " + ", ".join(f"{r['K']}: {r['score']:.3f}" for r in c["J"])
              + f"; J_T/J_1 {c['J_T_over_1']['score']}; max-min {c['max_min']['call']} {tuple(c['max_min']['x'])} "
              f"{c['max_min']['alpha_score']}; alpha*_K " + ", ".join(f"{K}: {v['score']}" for K, v in
                                                                     c["alpha_star"].items())
              + "; generalists at S1's thresholds " + ", ".join(f"{al}: {n}" for al, n in
                                                               c["at_s1_thresholds"]["n_generalists"].items()))
    path = OUT / f"{time.strftime('%Y%m%d')}-conflict_refined_{a.plant}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser(description="One finger layout for several HOM tasks, or a change of layout")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("scope")
    ps.add_argument("--grid", default="lattice", choices=("lattice", "sizes"))
    ps.add_argument("--plant", default="sv25", choices=list(GRIDS), help="the lattice grid's plant")
    ps.add_argument("--weights", type=float, nargs=len(TASKS), default=None, help="task weights in TASKS order")
    ps.add_argument("--val", action="store_true", help="also the choices scored at seeds 20-24 and smoothed (lattice)")
    pv = sub.add_parser("validate")
    pv.add_argument("--plant", default="sv25", choices=list(GRIDS))
    pv.add_argument("--tasks", nargs="+", default=None, choices=list(TASKS))
    pr = sub.add_parser("refine")
    pr.add_argument("what", choices=("layouts", "run", "validate", "report"))
    pr.add_argument("--plant", default="sv25", choices=list(GRIDS))
    pr.add_argument("--task", default=None, choices=list(TASKS))
    pr.add_argument("--level", default=None, choices=("grasp", "full"))
    pd = sub.add_parser("refined")
    pd.add_argument("--plant", default="sv25", choices=list(GRIDS))
    a = ap.parse_args()
    {"scope": cmd_scope, "validate": cmd_validate, "refine": cmd_refine, "refined": cmd_refined}[a.cmd](a)


if __name__ == "__main__":
    main()
