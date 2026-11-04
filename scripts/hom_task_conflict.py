#!/usr/bin/env python3
"""Tasks whose best finger layouts differ: one layout for all of them, a layout between their best ones, or a change
of layout between tasks (owner, 2026-11-03 20:3x: "how can we accommodate tasks with conflicting morphology objectives?
can we linearly interpolate? if not, is there some rule we can establish (pareto front) for when blending, however we
do it, is optimal, vs when we really should switch morphologies").

The SR2 hand moves its finger bases on leadscrew gantries, so it can change layout between tasks. The layouts here are
the symmetric tripods (x_sep, y_sep) on the 5 mm lattice of fp_sr2_toy.grid2d, which the HOM grids scored on four
tasks: the twist and the roll of a 40 mm sphere, the pin-push and the press of a 25 mm cylinder. A layout's score on
task t, f_t(x), is S1: the best grasp of its grasp set at HOM's defaults, its task score (hom_landscape.rollout_score)
averaged over seeds 0-2. A layout whose pose for a task is not reached scores 0 on it.

Quantities, for task weights p (each task's share of the episodes; equal unless given):
  specialist   a_t = argmax_x f_t(x), score f_t* (ties go to the layout with the better mean over the other tasks);
  J_1          max_x sum_t p_t f_t(x): the best single layout (the blend);
  J_K          the best K layouts with each task run on the better of them: the maximum over partitions of the tasks
               into K groups of sum_g max_x sum_{t in g} p_t f_t(x) (exact); J_T = sum_t p_t f_t* (one per task);
  normalized   the same with f_t / f_t*, so that each task's best counts 1;
  pair curves  for tasks A, B and a weight w on A: the best layout's w f_A + (1 - w) f_B, the better specialist's, and
               the utopia point w f_A* + (1 - w) f_B* (each task on its own specialist). The blend gain (best minus the
               better specialist) is above 0 when a layout other than the two specialists beats both; the switch gain
               (utopia minus best) is what changing layout between A and B adds;
  segment      the lattice layouts nearest to (1 - l) a_A + l a_B, l = 0, 0.05, ..., 1: the straight line in layout
               space between two specialists. Its dominance margin is the largest amount by which one layout of the
               grid beats a segment layout on both tasks (0 for a layout on the Pareto front).
Rule (score per second): with episodes of T_e s, a change of layout of T_s s (the gantries' travel,
manta_hand.kinematics.move_time_estimate_s) and tasks in runs of N episodes, K layouts give more score per second than
one when J_K / J_1 > 1 + T_s / (N T_e).

    python3 scripts/hom_task_conflict.py scope [--plant sv25]   # <date>-conflict_<plant>.json in OUT, and a summary
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/experiments/20261103-hom_task_conflict"
GRID_ROWS = ROOT / "docs/experiments/20261029-foundational_pose_review/rows"
TASKS = {"twist": 40, "roll": 40, "pinpush": 25, "press": 25}     # object diameter (mm): sphere, sphere, cylinder x2
GRIDS = {"sv25": "_sv25", "k21": "_k21", "k21mu0p4": "_k21mu0p4", "work10": ""}   # hom_grasp_refine.PLANTS' row tags
SEEDS = (0, 1, 2)
LATTICE = (40.0, 50.0, 5.0)                                        # fp_sr2_toy.LO2 and the grid's step (mm)
WEIGHTS = np.linspace(0.0, 1.0, 21)
STEPS = np.linspace(0.0, 1.0, 21)


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
    """The layouts reached for at least one task: calls, xy (n, 2) in mm, F (n, T) S1 by task in TASKS order (0 where
    the task's pose is not reached), the best grasp per task (-1 where not reached)."""
    per = {t: s1(GRIDS[plant], t) for t in TASKS}
    calls = sorted(set().union(*per.values()))
    xy = np.array([next(per[t][c][2] for t in TASKS if c in per[t]) for c in calls], float)
    F = np.array([[per[t][c][1] if c in per[t] else 0.0 for t in TASKS] for c in calls])
    G = np.array([[per[t][c][0] if c in per[t] else -1 for t in TASKS] for c in calls])
    return calls, xy, F, G


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


def pair(F, xy, calls, i, j):
    """Pair curves, front and segment of tasks i and j."""
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
    from scipy.stats import spearmanr
    both = (fa > 0) & (fb > 0)
    fr = front(fa, fb)
    return {"tasks": [list(TASKS)[i], list(TASKS)[j]],
            "a": {"call": calls[a], "xy": xy[a].tolist(), "fa": round(float(fa[a]), 4), "fb": round(float(fb[a]), 4)},
            "b": {"call": calls[b], "xy": xy[b].tolist(), "fa": round(float(fa[b]), 4), "fb": round(float(fb[b]), 4)},
            "distance_mm": round(float(np.linalg.norm(xy[a] - xy[b])), 1),
            "rho": round(float(spearmanr(fa[both], fb[both]).statistic), 3), "n_both": int(both.sum()),
            "blend_gain_max": round(max(c["best"] - c["specialist"] for c in curves), 4),
            "switch_gain_half": next(round(c["utopia"] - c["best"], 4) for c in curves if abs(c["w"] - 0.5) < 1e-9),
            "switch_gain_max": round(max(c["utopia"] - c["best"] for c in curves), 4),
            "segment_margin_max": round(max(s["margin"] for s in seg if s["margin"] is not None), 4),
            "curves": curves, "front": [[calls[k], round(float(fa[k]), 4), round(float(fb[k]), 4)] for k in fr],
            "segment": seg}


def move_time_s(xy_a, xy_b):
    """Gantry time (s) from layout a to layout b, all axes at once: the thumb moves x_sep/2 in x, the index and middle
    x_sep/2 in x and y_sep/2 in y (fp_sr2_toy.layout2); the driver's trapezoid of the longest travel."""
    sys.path.insert(0, str(ROOT / "src/morphohand/driver/manta/host"))
    from manta_hand.kinematics import move_time_estimate_s
    d = np.abs(np.asarray(xy_b, float) - np.asarray(xy_a, float)) / 2.0
    return move_time_estimate_s(1, float(d.max()))


def cmd_scope(a):
    t0 = time.time()
    calls, xy, F, G = matrix(a.plant)
    T, names = F.shape[1], list(TASKS)
    p = np.full(T, 1.0 / T) if not a.weights else np.asarray(a.weights, float) / np.sum(a.weights)
    fstar = F.max(0)
    N = F / fstar
    spec = [specialist(F, t) for t in range(T)]
    out = {"plant": a.plant, "grid": f"grid2dff_v4b{GRIDS[a.plant]}", "seeds": SEEDS, "tasks": TASKS,
           "weights": p.round(4).tolist(), "n_layouts": len(calls), "n_all_tasks": int((F > 0).all(1).sum()),
           "specialists": [{"task": names[t], "call": calls[k], "xy": xy[k].tolist(), "score": round(float(F[k, t]), 4),
                            "scores": F[k].round(4).tolist(), "ties": int((F[:, t] >= fstar[t] - 1e-9).sum())}
                           for t, k in enumerate(spec)],
           "levels": {}, "pairs": []}
    for name, M in (("raw", F), ("normalized", N)):
        rows = []
        for k in range(1, T + 1):
            val, picks = best_k(M, p, k)
            rows.append({"K": k, "J": round(val, 4),
                         "layouts": [{"tasks": [names[t] for t in g], "call": calls[i], "xy": xy[i].tolist(),
                                      "scores": F[i].round(4).tolist()} for g, i in picks]})
        out["levels"][name] = rows
    mm = int(np.argmax(N.min(1)))
    out["maxmin"] = {"call": calls[mm], "xy": xy[mm].tolist(), "scores": F[mm].round(4).tolist(),
                     "worst_fraction": round(float(N[mm].min()), 4)}
    for i, j in itertools.combinations(range(T), 2):
        out["pairs"].append(pair(F, xy, calls, i, j))
    for pr in out["pairs"]:
        pr["move_s"] = round(move_time_s(pr["a"]["xy"], pr["b"]["xy"]), 2)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{time.strftime('%Y%m%d')}-conflict_{a.plant}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"{a.plant}: {len(calls)} layouts ({out['n_all_tasks']} reach every task), {time.time() - t0:.0f} s -> "
          f"{path.relative_to(ROOT)}")
    for s in out["specialists"]:
        print(f"  best for {s['task']:8s} {s['xy']} score {s['score']:.3f} ({s['ties']} tied); on the four tasks "
              f"{s['scores']}")
    for name in ("raw", "normalized"):
        print(f"  {name}: " + ", ".join(f"J_{r['K']} {r['J']:.3f}" for r in out["levels"][name]))
        for r in out["levels"][name][:3]:
            print(f"    K={r['K']}: " + "; ".join(f"{'+'.join(L_['tasks'])} at {L_['xy']}" for L_ in r["layouts"]))
    print(f"  max-min layout {out['maxmin']['xy']}: worst task at {out['maxmin']['worst_fraction']:.3f} of its best")
    for pr in out["pairs"]:
        print(f"  {pr['tasks'][0]}-{pr['tasks'][1]}: specialists {pr['distance_mm']} mm apart ({pr['move_s']} s), "
              f"rho {pr['rho']}, blend gain {pr['blend_gain_max']:.3f}, switch gain at w 0.5 "
              f"{pr['switch_gain_half']:.3f} (max {pr['switch_gain_max']:.3f}), segment margin up to "
              f"{pr['segment_margin_max']:.3f}")


def main():
    ap = argparse.ArgumentParser(description="One finger layout for several HOM tasks, or a change of layout")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("scope")
    ps.add_argument("--plant", default="sv25", choices=list(GRIDS))
    ps.add_argument("--weights", type=float, nargs=len(TASKS), default=None, help="task weights in TASKS order")
    a = ap.parse_args()
    {"scope": cmd_scope}[a.cmd](a)


if __name__ == "__main__":
    main()
