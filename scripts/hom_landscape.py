#!/usr/bin/env python3
"""Landscapes of HOM task scores over the SR2 hand's layouts and object sizes: the size study of the design page
(docs/experiments/20261029-foundational_pose_review), from the rows of scripts/hom_gpu.py run on --layouts sobol8k_size.

Every one of the 8,192 Sobol layouts of scripts/fp_sr2_toy.py is scored for each task and object size. A rollout's score
lies in [0, 1] and is 0 when the object dropped:
  twist    the twist reached on three pads over the 90 deg ramp (30 deg/s for 3 s), divided by 90 deg;
  roll     1 - the angular-velocity RMSE over the 40 deg/s steps, divided by the step (w_rmse_ratio), at least 0;
  pin-push 1 - the pinch-angle RMSE over the 15 deg stroke, at least 0;
  press    1 - the pinch-angle RMSE over the 20 deg stroke, at least 0.
A layout's score is the mean over its seeds; a layout that does not reach the task's pose on that object scores 0.
Neighbourhoods are taken on the standardized six coordinates (each gantry's travel mapped to [-0.5, 0.5], the toy's z):
a layout's smoothed score is the mean over its K nearest layouts, itself included, and a basin is the set of layouts
whose steepest ascent of the smoothed score over the K-nearest-neighbour graph ends at the same peak.

  .venv/bin/python scripts/hom_landscape.py stats            # tables of the size study -> rows/sizes/landscape.json
  .venv/bin/python scripts/hom_landscape.py figures          # the page's figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fp_review_figures as FR  # noqa: E402

ROWS = FR.ROWS
SIZE_ROWS = ROWS / "sizes"
SIZES = {"twist": (30.0, 40.0, 50.0, 63.0), "roll": (30.0, 40.0, 50.0, 63.0), "pinpush": (10.0, 16.0, 25.0, 40.0),
         "press": (10.0, 16.0, 25.0, 40.0)}
TASKS = tuple(SIZES)
POSE_ROWS = {"twist": "tripod", "roll": "tripod", "pinpush": "pinch", "press": "pinch"}
STROKE = {"roll": 1.0, "pinpush": 15.0, "press": 20.0}          # the score's denominators (w_rmse_ratio is already /step)
K_NN = 16
TOP = 0.10                                                     # the best tenth


def rollout_score(task, r):
    """A rollout's score in [0, 1] (module docstring)."""
    if not r.get("held"):
        return 0.0
    if task == "twist":
        return float(np.clip((r.get("ramp_three_pads_max_deg") or 0.0) / 90.0, 0.0, 1.0))
    m = r.get("w_rmse_ratio") if task == "roll" else r.get("theta_rmse_deg")
    return 0.0 if m is None else float(np.clip(1.0 - m / STROKE[task], 0.0, 1.0))


def layouts():
    """The 8,192 Sobol layouts: six coordinates (mm, n x 6) and standardized coordinates (n x 6), by draw index."""
    rows = FR.load("sobol8k")
    order = np.argsort([r["call"] for r in rows])
    d = np.array([rows[i]["d"] for i in order], float)
    z = np.array([rows[i]["z"] for i in order], float)
    return d, z


def reach(task, size):
    """Draw indices whose pose for the task's object of this size was reached."""
    import fp_sr2_toy as FP
    name = FP.size_name("sobol8k", "sphere" if POSE_ROWS[task] == "tripod" else "cylinder", size)
    pose = {"twist": "tripod", "roll": "tripod", "pinpush": "pinch", "press": "press"}[task]
    out = set()
    for r in FR.load(name):
        if (r["fp"].get(pose) or {}).get("ok"):
            out.add(int(r["call"]))
    return out


def scores(task, size, n, rows_dir=SIZE_ROWS):
    """Per draw index: mean score over seeds (0 where the pose is not reached), held in every seed, seeds run, and
    the per-seed scores (n x seeds, NaN where not run)."""
    import hom_rows
    p = rows_dir / f"{task}_d{size:g}.jsonl"
    rows = [hom_rows.rescale_metrics(json.loads(x)) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    seeds = sorted({r["seed"] for r in rows})
    per = np.full((n, max(1, len(seeds))), np.nan)
    held = np.zeros((n, max(1, len(seeds))), bool)
    for r in rows:
        if r.get("status", "ok") == "ok":
            i, j = int(r["call"]), seeds.index(r["seed"])
            per[i, j], held[i, j] = rollout_score(task, r), bool(r.get("held"))
    run = np.isfinite(per).all(1)
    mean = np.zeros(n)
    mean[run] = per[run].mean(1)
    return {"mean": mean, "held_all": run & held.all(1), "run": run, "per": per, "seeds": seeds}


def knn(z, k=K_NN):
    """Indices (n x k) of each layout's k nearest layouts in the standardized coordinates, itself first."""
    from scipy.spatial import cKDTree
    return cKDTree(z).query(z, k=k)[1]


def smooth(s, nb):
    return s[nb].mean(1)


def basins(sm, nb):
    """Steepest ascent of the smoothed score over the neighbour graph: each layout's peak (a draw index)."""
    up = nb[np.arange(len(sm)), np.argmax(sm[nb], 1)]           # the best of each layout's neighbourhood (maybe itself)
    peak = up.copy()
    for _ in range(64):
        nxt = up[peak]
        if np.array_equal(nxt, peak):
            break
        peak = nxt
    return peak


def neighbourhoods(s, nb, q=0.9, min_size=20):
    """High-scoring neighbourhoods: the basins of the smoothed score whose peak lies at or above its q-quantile and
    that hold at least min_size layouts. [(peak, members, smoothed score at the peak)], best first."""
    sm = smooth(s, nb)
    pk = basins(sm, nb)
    thr = np.quantile(sm, q)
    out = []
    for p in np.unique(pk):
        mem = np.flatnonzero(pk == p)
        if sm[p] >= thr and len(mem) >= min_size:
            out.append((int(p), mem, float(sm[p])))
    return sorted(out, key=lambda x: -x[2]), sm


def interaction_share(S, per_seed_var):
    """Two-way decomposition of a layouts x sizes score matrix: the shares of its total sum of squares from the layout,
    the size and their interaction, and the interaction expected from seed noise alone (per_seed_var: each cell's
    variance of the mean over seeds)."""
    mu = S.mean()
    a = S.mean(1, keepdims=True) - mu
    b = S.mean(0, keepdims=True) - mu
    e = S - mu - a - b
    tot = ((S - mu) ** 2).sum()
    n, m = S.shape
    noise = float(np.nansum(per_seed_var)) * (n - 1) * (m - 1) / (n * m)
    return {"layout": float((a ** 2).sum() * m / tot), "size": float((b ** 2).sum() * n / tot),
            "interaction": float((e ** 2).sum() / tot), "interaction_noise": noise / float(tot)}


def top_kept(a, b, frac=TOP):
    """Share of the best `frac` of layouts by score a that are in the best `frac` by score b."""
    k = max(1, int(round(frac * len(a))))
    ta, tb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return len(ta & tb) / k


def cmd_stats(a):
    d, z = layouts()
    n = len(d)
    nb = knn(z)
    out = {"n_layouts": n, "k_nn": K_NN, "tasks": {}}
    for task in TASKS:
        T = {"sizes": {}, "pairs": {}}
        S, V = [], []
        for size in SIZES[task]:
            sc = scores(task, size, n)
            if not sc["run"].any():
                continue
            R = reach(task, size)
            nbh, sm = neighbourhoods(sc["mean"], nb)
            var = np.where(sc["run"], np.nanvar(sc["per"], 1, ddof=1) / max(1, len(sc["seeds"])), 0.0)
            S.append(sc["mean"])
            V.append(np.nan_to_num(var))
            T["sizes"][f"{size:g}"] = {
                "reached": len(R), "run": int(sc["run"].sum()), "held_all": int(sc["held_all"].sum()),
                "score_median_reached": float(np.median(sc["mean"][list(R)])) if R else None,
                "score_p90": float(np.quantile(sc["mean"], 0.9)),
                "neighbourhoods": [{"peak": p, "n": len(m), "score_smoothed": round(v, 4), "d_peak": d[p].round(1).tolist(),
                                    "d_median": np.median(d[m], 0).round(1).tolist()} for p, m, v in nbh[:12]],
                "n_neighbourhoods": len(nbh)}
        if len(S) >= 2:
            S, V = np.array(S).T, np.array(V).T
            szs = [k for k in T["sizes"]]
            for i in range(len(szs)):
                for j in range(i + 1, len(szs)):
                    T["pairs"][f"{szs[i]}-{szs[j]}"] = {"spearman": round(FR.spearman(S[:, i], S[:, j]), 3),
                                                       "top_kept": round(top_kept(S[:, i], S[:, j]), 3)}
            T["variance"] = {k: round(v, 4) for k, v in interaction_share(S, V).items()}
        out["tasks"][task] = T
    SIZE_ROWS.mkdir(parents=True, exist_ok=True)
    (SIZE_ROWS / "landscape.json").write_text(json.dumps(out, indent=1))
    for task, T in out["tasks"].items():
        print(task, {k: (v["reached"], v["held_all"], len(v["neighbourhoods"])) for k, v in T["sizes"].items()},
              T.get("variance"), {k: v["spearman"] for k, v in T["pairs"].items()}, flush=True)
    return 0


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    a = ap.parse_args()
    return {"stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
