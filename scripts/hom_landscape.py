#!/usr/bin/env python3
"""Landscapes of HOM task scores over the SR2 hand's layouts and object sizes: the size study of the design page
(docs/experiments/20261029-foundational_pose_review), from the rows of scripts/hom_gpu.py run on --layouts sobol8k_size.

Every one of the 8,192 Sobol layouts of scripts/fp_sr2_toy.py is scored for each task and object size. A rollout's score
lies in [0, 1] and is 0 when the object dropped:
  twist    the twist reached on three pads over the 90 deg ramp (30 deg/s for 3 s), divided by 90 deg;
  roll     1 / (1 + the angular-velocity RMSE over the 40 deg/s steps divided by the step, w_rmse_ratio): 1 for perfect
           tracking, 0.5 for an error as large as the step;
  pin-push 1 - the pinch-angle RMSE over the 15 deg stroke, at least 0;
  press    1 - the pinch-angle RMSE over the 20 deg stroke, at least 0.
A layout's score is the mean over its seeds; a layout that does not reach the task's pose on that object scores 0.
The hand is the CAD hand with the v4b fingertip (SCALE_HAND=cad_v2; the pose rows' names end in _v4b).
Neighbourhoods are taken on the standardized six coordinates (each gantry's travel mapped to [-0.5, 0.5], the toy's z):
a layout's smoothed score is the mean over its K nearest layouts, itself included, and a basin is the set of layouts
whose steepest ascent of the smoothed score over the K-nearest-neighbour graph ends at the same peak.

  .venv/bin/python scripts/hom_landscape.py stats            # tables of the size study -> rows/sizes/landscape.json
  .venv/bin/python scripts/hom_landscape.py figures          # the page's figures
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fp_review_figures as FR  # noqa: E402

ROWS = FR.ROWS
SIZE_ROWS = ROWS / "sizes_v4b"                 # the v4b fingertip (SCALE_HAND=cad_v2, the default of the toy)
SIZES = {"twist": (30.0, 40.0, 50.0, 63.0), "roll": (30.0, 40.0, 50.0, 63.0), "pinpush": (10.0, 16.0, 25.0, 40.0),
         "press": (10.0, 16.0, 25.0, 40.0)}
TASKS = tuple(SIZES)
POSE_ROWS = {"twist": "tripod", "roll": "tripod", "pinpush": "pinch", "press": "pinch"}
STROKE = {"pinpush": 15.0, "press": 20.0}                     # the pinch tasks' strokes (deg), the score's denominators
K_NN = 16
TOP = 0.10                                                     # the best tenth


def rollout_score(task, r):
    """A rollout's score in [0, 1] (module docstring)."""
    if not r.get("held"):
        return 0.0
    if task == "twist":
        return float(np.clip((r.get("ramp_three_pads_max_deg") or 0.0) / 90.0, 0.0, 1.0))
    if task == "roll":
        m = r.get("w_rmse_ratio")
        return 0.0 if m is None else float(1.0 / (1.0 + m))
    m = r.get("theta_rmse_deg")
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


# ------------------------------------------------------------------------------------------------ value model and gradients

VALUE_LS = (0.1, 0.15, 0.2)            # RBF length scales tried, standardized units (1 = a gantry's travel)
VALUE_LAM = (0.03, 0.1, 0.3)           # ridge, as a share of the score's variance


def rbf(A, B, ls):
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-0.5 * np.maximum(d2, 0.0) / ls ** 2)


class Value:
    """Kernel ridge regression of a task's score over the standardized layout coordinates, with its gradient: the
    counterpart, for a training-free controller, of the frozen critic whose design gradient Bohlinger and Peters
    (arXiv 2606.00702) follow."""

    def __init__(self, Z, y, ls, lam):
        from scipy.linalg import cho_factor, cho_solve
        self.Z, self.ls, self.mu = Z, float(ls), float(y.mean())
        K = rbf(Z, Z, ls)
        K[np.diag_indices_from(K)] += lam * float(y.var())
        self.alpha = cho_solve(cho_factor(K, overwrite_a=True), y - self.mu)

    def __call__(self, X):
        return self.mu + rbf(X, self.Z, self.ls) @ self.alpha

    def grad(self, X):
        Kx = rbf(X, self.Z, self.ls) * self.alpha[None, :]
        return (Kx @ self.Z - Kx.sum(1)[:, None] * X) / self.ls ** 2


def value_cv(Z, y, folds=5, seed=0):
    """Held-out R^2 and Spearman of the value model per (length scale, ridge), by k-fold cross-validation."""
    idx = np.random.default_rng(seed).permutation(len(y)) % folds
    out = {}
    for ls in VALUE_LS:
        for lam in VALUE_LAM:
            pred = np.zeros(len(y))
            for k in range(folds):
                tr = idx != k
                pred[~tr] = Value(Z[tr], y[tr], ls, lam)(Z[~tr])
            out[(ls, lam)] = (1.0 - float(((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                              FR.spearman(pred, y), pred)
    return out


def ascend(V, X, steps=300, step0=0.02, box=0.5, trust=None):
    """Gradient ascent of the value model from the rows of X inside the standardized box, with a per-start step that
    halves when a step does not raise the value (a backtracking line search per start). trust: each coordinate stays
    within this distance of its start (standardized units), as the trust region of Bohlinger and Peters keeps the
    design where the model was fitted."""
    X0, X = X.copy(), X.copy()
    lo, hi = (-box, box) if trust is None else (np.maximum(X0 - trust, -box), np.minimum(X0 + trust, box))
    f = V(X)
    st = np.full(len(X), step0)
    for _ in range(steps):
        g = V.grad(X)
        gn = np.linalg.norm(g, axis=1, keepdims=True)
        Xn = np.clip(X + st[:, None] * g / np.maximum(gn, 1e-12), lo, hi)
        fn = V(Xn)
        up = fn > f
        X[up], f[up] = Xn[up], fn[up]
        st = np.where(up, np.minimum(st * 1.2, step0), st * 0.5)
        if (st < 1e-4).all():
            break
    return X, f


def svgd(V, X, steps=300, eps=0.01, temp=None, box=0.5):
    """Stein variational gradient descent toward p(x) ~ exp(f(x) / temp): particles climb the value model and repel
    each other through an RBF kernel of median bandwidth, so they spread over the high-value region instead of
    collapsing on one peak."""
    X = X.copy()
    f0 = V(X)
    temp = temp or max(float(np.quantile(f0, 0.9) - np.quantile(f0, 0.5)) / 3.0, 1e-3)
    for _ in range(steps):
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        h = max(float(np.median(d2)) / math.log(len(X) + 1.0), 1e-6)
        Kxx = np.exp(-d2 / h)
        drive = Kxx @ (V.grad(X) / temp)
        repel = (Kxx.sum(1)[:, None] * X - Kxx @ X) * (2.0 / h)
        X = np.clip(X + eps * (drive + repel) / len(X), -box, box)
    return X, V(X)


def modes(X, f, radius=0.05):
    """Group end points closer than `radius` (standardized) around the best of each group: [(point, value, count)]."""
    order = np.argsort(-f)
    used = np.zeros(len(X), bool)
    out = []
    for i in order:
        if used[i]:
            continue
        near = (~used) & (np.linalg.norm(X - X[i], axis=1) < radius)
        used |= near
        out.append((X[i], float(f[i]), int(near.sum())))
    return out


VALUE_TRUST = 0.1                      # trust region of the local ascent: 0.1 of each gantry's travel (6-11 mm)


def grid_gradient_check(V, task, size, step_mm=10.0, seed=0):
    """The value model's gradient against the measured 2-D grid (foundational poses, mean over seeds): from every
    reached grid layout, a step of step_mm along the model's gradient within the grid's plane (snapped to the grid)
    against a step of the same length in a random direction. Returns the share of steps that raised the measured score
    and the median change, for both."""
    import fp_sr2_toy as FP
    tab = [r for r in grid_table(task, size) if r.get("n_grasps")]
    meas = {(r["x_sep"], r["y_sep"]): r["fp"] for r in tab if np.isfinite(r["fp"])}
    if not meas:
        return None
    rng = np.random.default_rng(seed)
    rngs = (FP.HI6 - FP.LO6)
    dz_dx = np.array([-0.5, 0.0, 0.5, 0.0, 0.5, 0.0]) / rngs          # d(standardized coordinates)/d(x_sep) per mm
    dz_dy = np.array([0.0, 0.0, 0.0, 0.5, 0.0, -0.5]) / rngs
    keys = list(meas)
    Z = np.array([FP.standard(np.array(FP.layout2(*k)), FP.LO6, FP.HI6) for k in keys])
    G = V.grad(Z)
    g2 = np.stack([G @ dz_dx, G @ dz_dy], 1)                           # d value / d(x_sep, y_sep), per mm
    out = {}
    for name, dirs in (("gradient", g2), ("random", rng.normal(size=g2.shape))):
        u = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
        ch = []
        for k, (x, y_) in enumerate(keys):
            tgt = (x + 5.0 * round(step_mm * u[k, 0] / 5.0), y_ + 5.0 * round(step_mm * u[k, 1] / 5.0))
            if tgt != (x, y_) and tgt in meas:
                ch.append(meas[tgt] - meas[(x, y_)])
        ch = np.array(ch)
        out[name] = {"steps": len(ch), "raised": float(np.mean(ch > 0)), "lowered": float(np.mean(ch < 0)),
                     "median_change": float(np.median(ch))}
    return out


def cmd_value(a):
    """The value model of one task and object size on the Sobol layouts (mean score over seeds; 0 where the pose is not
    reached): cross-validated fit, check on the 2-D grid's foundational poses, gradient ascent from every reached layout,
    and SVGD particles from a random tenth. Writes rows/sizes_v4b/value_<task>_d<size>.json."""
    import fp_sr2_toy as FP
    d, z = layouts()
    sc = scores(a.task, a.d, len(d))
    if not sc["run"].any():
        print("no rows yet", flush=True)
        return 1
    y = sc["mean"]
    cv = value_cv(z, y)
    (ls, lam), (r2, rho, pred) = max(cv.items(), key=lambda kv: kv[1][0])
    V = Value(z, y, ls, lam)
    out = {"task": a.task, "d_mm": a.d, "n": len(y), "ls": ls, "lam": lam, "cv_r2": r2, "cv_spearman": rho,
           "cv": {f"{k[0]}_{k[1]}": [round(v[0], 4), round(v[1], 4)] for k, v in cv.items()}}
    g = {(round(x, 1), round(y_, 1)): v for x, y_, v in
         ((r["x_sep"], r["y_sep"], r.get("fp")) for r in grid_table(a.task, a.d)) if v is not None and np.isfinite(v)}
    if g:
        Zg = np.array([FP.standard(np.array(FP.layout2(*k)), FP.LO6, FP.HI6) for k in g])
        out["grid_check"] = {"n": len(g), "spearman": FR.spearman(V(Zg), np.array(list(g.values())))}
    gc = grid_gradient_check(V, a.task, a.d)
    if gc:
        out["grid_steps"] = gc
    R = sorted(reach(a.task, a.d))
    top = np.array(R)[np.argsort(-y[R])[:200]]
    Xt, ft = ascend(V, z[top], trust=VALUE_TRUST)
    out["trust"] = {"radius": VALUE_TRUST, "starts": len(top), "pred_gain_median": float(np.median(ft - V(z[top]))),
                    "moves_mm": np.median(np.abs((Xt - z[top]) * (FP.HI6 - FP.LO6)), 0).round(2).tolist(),
                    "layouts": [{"call": int(c), "d": d[c].round(1).tolist(),
                                 "d_new": (FP.LO6 + (x + 0.5) * (FP.HI6 - FP.LO6)).round(1).tolist(),
                                 "pred": round(float(V(z[c:c + 1])[0]), 4), "pred_new": round(float(f_), 4)}
                                for c, x, f_ in zip(top, Xt, ft)]}
    Xa, fa = ascend(V, z[R])
    M = modes(Xa, fa)
    rng = np.random.default_rng(1)
    Xs, fs = svgd(V, z[rng.choice(R, size=max(64, len(R) // 10), replace=False)])
    Ms = modes(Xs, fs, radius=0.1)
    nb = knn(z)
    nbh, _sm = neighbourhoods(y, nb)
    out.update(best_sobol={"call": int(np.argmax(y)), "score": float(y.max()), "d": d[int(np.argmax(y))].round(1).tolist()},
               ascent={"starts": len(R), "modes": [{"d": (FP.LO6 + (p + 0.5) * (FP.HI6 - FP.LO6)).round(1).tolist(),
                                                    "value": round(v, 4), "starts": c} for p, v, c in M[:12]],
                       "n_modes": len(M), "gain_median": float(np.median(fa - V(z[R])))},
               svgd={"particles": len(Xs), "value_median": float(np.median(fs)),
                     "modes": [{"d": (FP.LO6 + (p + 0.5) * (FP.HI6 - FP.LO6)).round(1).tolist(), "value": round(v, 4),
                                "particles": c} for p, v, c in Ms[:12]], "n_modes": len(Ms)},
               knn_basins=[{"d_peak": d[p].round(1).tolist(), "n": len(m), "score_smoothed": round(v, 4)}
                           for p, m, v in nbh[:12]])
    path = SIZE_ROWS / f"value_{a.task}_d{a.d:g}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"{a.task} {a.d:g} mm: ls {ls} lam {lam} cv R2 {r2:.3f} Spearman {rho:.3f}; grid check "
          f"{out.get('grid_check')}; ascent {len(R)} starts -> {len(M)} modes, top "
          f"{[(m['d'], m['value'], m['starts']) for m in out['ascent']['modes'][:3]]}; svgd {len(Ms)} modes", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ the 2-D grid with grasp sets

GRID_ROWS = ROWS / "grid2d_v4b"
POSE_KEY = {"twist": "tripod", "roll": "tripod", "pinpush": "pinch", "press": "press"}


def grid_poses(task, size, src="grid2d"):
    """The grid's layouts for the task's object: {call: (x_sep, y_sep, pose record with its grasps or None)}."""
    import fp_sr2_toy as FP
    name = FP.size_name(src, "sphere" if POSE_ROWS[task] == "tripod" else "cylinder", size)
    return {int(r["call"]): (r["xy"][0], r["xy"][1], (r["fp"].get(POSE_KEY[task]) or {}) if r["ok"] else None)
            for r in FR.load(name)}


def grid_scores(task, size, rows_dir=GRID_ROWS, name=None):
    """Per (call, grasp): [score of each seed], and per (call, grasp, seed): held."""
    import hom_rows
    p = rows_dir / f"{name or f'{task}_d{size:g}'}.jsonl"
    sc, held = {}, {}
    for x in (p.read_text().splitlines() if p.exists() else []):
        if not x.strip():
            continue
        r = hom_rows.rescale_metrics(json.loads(x))
        if r.get("status", "ok") != "ok":
            continue
        k = (int(r["call"]), int(r.get("grasp", 0)))
        sc.setdefault(k, []).append(rollout_score(task, r))
        held[k + (r["seed"],)] = bool(r.get("held"))
    return sc, held


def grid_table(task, size):
    """Per grid layout: x_sep, y_sep, reached, grasps run, the foundational pose's score (grasp 0, mean over seeds),
    the best grasp's (max over grasps of the mean), its index, the share of grasps held in every seed, logvol."""
    poses = grid_poses(task, size)
    sc, held = grid_scores(task, size)
    out = []
    for call, (xs, ys, rec) in sorted(poses.items()):
        row = {"call": call, "x_sep": xs, "y_sep": ys, "reached": bool(rec and rec.get("ok"))}
        if row["reached"]:
            g = {j: float(np.mean(v)) for (c, j), v in sc.items() if c == call}
            ok = [all(h for (c, j2, _s), h in held.items() if c == call and j2 == j) for j in g]
            row.update(n_grasps=len(g), fp=g.get(0, np.nan), best=max(g.values()) if g else np.nan,
                       best_grasp=max(g, key=g.get) if g else None, held_share=float(np.mean(ok)) if ok else np.nan,
                       logvol=(rec.get("grasps") or [{}])[0].get("logvol"))
        out.append(row)
    return out


def cmd_grid(a):
    out = {}
    for task in TASKS:
        for size in SIZES[task]:
            tab = grid_table(task, size)
            run = [r for r in tab if r.get("n_grasps")]
            if not run:
                continue
            fp, best = np.array([r["fp"] for r in run]), np.array([r["best"] for r in run])
            out[f"{task}_d{size:g}"] = {
                "layouts": len(tab), "reached": sum(r["reached"] for r in tab), "run": len(run),
                "fp_median": round(float(np.nanmedian(fp)), 4), "best_median": round(float(np.nanmedian(best)), 4),
                "gain_median": round(float(np.nanmedian(best - fp)), 4),
                "best_is_fp": round(float(np.mean([r["best_grasp"] == 0 for r in run])), 3),
                "fp_vs_best_spearman": round(FR.spearman(fp, best), 3),
                "held_share_median": round(float(np.nanmedian([r["held_share"] for r in run])), 3)}
            print(task, size, out[f"{task}_d{size:g}"], flush=True)
    GRID_ROWS.mkdir(parents=True, exist_ok=True)
    (GRID_ROWS / "summary.json").write_text(json.dumps(out, indent=1))
    return 0


def grid_maps(task, size):
    """The grid's layout table as maps over (x_sep, y_sep) (rows y_sep, columns x_sep; NaN where the pose is not
    reached or not run), with the axes."""
    tab = grid_table(task, size)
    xs = np.unique([r["x_sep"] for r in tab])
    ys = np.unique([r["y_sep"] for r in tab])
    maps = {k: np.full((len(ys), len(xs)), np.nan) for k in ("fp", "best", "held_share", "logvol", "reached")}
    for r in tab:
        i, j = int(np.searchsorted(ys, r["y_sep"])), int(np.searchsorted(xs, r["x_sep"]))
        maps["reached"][i, j] = float(r["reached"])
        for k in ("fp", "best", "held_share", "logvol"):
            if r.get(k) is not None:
                maps[k][i, j] = r[k]
    return xs, ys, maps


def grid_ascent(Z):
    """Steepest ascent of a map over its 8-neighbour grid (NaN cells excluded): the peak cell (flat index) of each
    cell, -1 for NaN cells."""
    ny, nx = Z.shape
    up = -np.ones(Z.size, int)
    for i in range(ny):
        for j in range(nx):
            if not np.isfinite(Z[i, j]):
                continue
            best, arg = Z[i, j], i * nx + j
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < ny and 0 <= b < nx and np.isfinite(Z[a, b]) and Z[a, b] > best:
                        best, arg = Z[a, b], a * nx + b
            up[i * nx + j] = arg
    peak = up.copy()
    for _ in range(nx * ny):
        nxt = np.where(peak >= 0, up[np.maximum(peak, 0)], -1)
        if np.array_equal(nxt, peak):
            break
        peak = nxt
    return peak


SMOOTH_CELLS = 1.5                   # Gaussian width of the grid maps' smoothing, cells (7.5 mm)


def smooth_map(Z, sigma=SMOOTH_CELLS):
    """Gaussian smoothing over the finite cells (normalised by the weight of the finite cells in reach)."""
    from scipy.ndimage import gaussian_filter
    ok = np.isfinite(Z)
    num = gaussian_filter(np.where(ok, Z, 0.0), sigma, mode="constant")
    den = gaussian_filter(ok.astype(float), sigma, mode="constant")
    return np.where(ok, num / np.maximum(den, 1e-9), np.nan)


def peaks(sm, min_cells=8, prominence=0.05):
    """Basin peaks of a smoothed map: steepest ascent's fixed points whose basin holds min_cells cells or more and whose
    height stands prominence above the basin's lowest cell."""
    pk = grid_ascent(sm)
    out = []
    for p in np.unique(pk[pk >= 0]):
        mem = pk == p
        v = sm.flat[mem]
        if mem.sum() >= min_cells and sm.flat[p] - np.nanmin(v) >= prominence:
            out.append(int(p))
    return pk, out


def fig_grid(task, size, out=None):
    """Six maps over the grid for one task and object: the foundational pose's score, the best grasp's, their gain, the
    share of grasps held in every seed, the feasible log volume of the grasp set, and the best-grasp score smoothed over
    3 x 3 cells with its gradient and the peaks of its basins."""
    plt = FR.plt_setup()
    xs, ys, M = grid_maps(task, size)
    ext = [xs[0] - 2.5, xs[-1] + 2.5, ys[0] - 2.5, ys[-1] + 2.5]
    sm = smooth_map(M["best"])
    gy, gx = np.gradient(sm, ys, xs)
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 9.6), constrained_layout=True)
    panels = (("fp", "foundational pose: score", "viridis", (0, None)), ("best", "best of the grasp set: score", "viridis",
              (0, None)), ("gain", "best minus foundational pose", "magma", (0, None)),
              ("held_share", "share of grasps held in every seed", "cividis", (0, 1)),
              ("logvol", "feasible log volume of the grasp set", "cividis", (None, None)), ("smooth", "best, smoothed (7.5 mm);"
              " gradient, basin peaks", "viridis", (0, None)))
    vmax = np.nanmax(M["best"]) if np.isfinite(M["best"]).any() else 1.0
    for ax, (k, title, cmap, (lo, hi)) in zip(axs.flat, panels):
        Z = {"gain": M["best"] - M["fp"], "smooth": sm}.get(k, M.get(k))
        if k in ("fp", "best", "smooth"):
            lo, hi = 0.0, vmax
        ax.imshow(np.where(np.isfinite(M["reached"]) & (M["reached"] > 0), np.nan, 0.0), origin="lower", extent=ext,
                  cmap="Greys", vmin=-1, vmax=1, aspect="equal")
        im = ax.imshow(Z, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi, aspect="equal")
        fig.colorbar(im, ax=ax, shrink=0.82)
        if k == "smooth":
            X, Y = np.meshgrid(xs, ys)
            s = (slice(None, None, 2), slice(None, None, 2))
            ax.quiver(X[s], Y[s], gx[s], gy[s], color="white", angles="xy", scale_units="xy",
                      scale=np.nanmax(np.hypot(gx, gy)) / 8.0 if np.isfinite(gx).any() else 1.0, width=0.004)
            _pk, big = peaks(sm)
            ax.plot(X.flat[big], Y.flat[big], "r*", ms=14, mec="k")
        ax.set_title(title)
        ax.set_xlabel("thumb-to-pair distance x_sep (mm)")
        ax.set_ylabel("index-middle spacing y_sep (mm)")
    obj = f"{'sphere' if T_SHAPE[task] == 'sphere' else 'cylinder'} {size:g} mm"
    fig.suptitle(f"{TASK_NAME[task]}, {obj}: symmetric tripods every 5 mm, 16 grasps each (grey: pose not reached)")
    FR.save(fig, out or f"20261029-grid_{task}_d{size:g}.png")


GRASP_VARS = {   # (index in the pose's variables, label, scale to display units)
    "tripod": ((2, "sphere depth below the palm (mm)", 1000.0), (6, "contact latitude (deg)", 180 / np.pi),
               (3, "thumb azimuth offset (deg)", 180 / np.pi), (4, "index azimuth offset (deg)", 180 / np.pi)),
    "pinch": ((2, "cylinder depth below the palm (mm)", 1000.0), (3, "cylinder yaw (deg)", 180 / np.pi),
              (None, "lever, middle to pinch (mm)", 1000.0), (6, "pinch latitude (deg)", 180 / np.pi))}


def fig_grid_grasps(task, size, out=None):
    """The best grasp's variables over the grid (top row) against the foundational pose's (bottom row)."""
    plt = FR.plt_setup()
    poses = grid_poses(task, size)
    tab = {r["call"]: r for r in grid_table(task, size)}
    xs = np.unique([r["x_sep"] for r in tab.values()])
    ys = np.unique([r["y_sep"] for r in tab.values()])
    fam = "tripod" if POSE_ROWS[task] == "tripod" else "pinch"
    spec = GRASP_VARS[fam]
    maps = np.full((2, len(spec), len(ys), len(xs)), np.nan)
    for call, (x, y, rec) in poses.items():
        r = tab.get(call)
        if not rec or not rec.get("ok") or not r or r.get("best_grasp") is None:
            continue
        i, j = int(np.searchsorted(ys, y)), int(np.searchsorted(xs, x))
        for row, g in enumerate((rec["grasps"][r["best_grasp"]], rec["grasps"][0])):
            v = np.asarray(g["v"], float)
            for c, (k, _lab, sc) in enumerate(spec):
                maps[row, c, i, j] = (v[4] - v[5]) * sc if k is None else v[k] * sc
    ext = [xs[0] - 2.5, xs[-1] + 2.5, ys[0] - 2.5, ys[-1] + 2.5]
    fig, axs = plt.subplots(2, len(spec), figsize=(4.4 * len(spec), 8.4), constrained_layout=True)
    for c, (_k, lab, _sc) in enumerate(spec):
        lo, hi = np.nanpercentile(maps[:, c], [2, 98])
        for row, who in enumerate(("best grasp", "foundational pose")):
            ax = axs[row, c]
            im = ax.imshow(maps[row, c], origin="lower", extent=ext, cmap="coolwarm", vmin=lo, vmax=hi, aspect="equal")
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f"{who}: {lab}", fontsize=12)
            ax.set_xlabel("x_sep (mm)")
            ax.set_ylabel("y_sep (mm)")
    fig.suptitle(f"{TASK_NAME[task]}, {T_SHAPE[task]} {size:g} mm: the grasp that scored best on each layout, and the "
                 f"foundational pose")
    FR.save(fig, out or f"20261029-grid_grasps_{task}_d{size:g}.png")


def transfer_grasp(fp, v, M_src, M_tgt):
    """A grasp moved to another layout as a keyframe: the object's pose and the contact points stay fixed in the palm
    frame (the IK keyframe of the task study). The pinch family's variables are already palm-frame; the tripod's contact
    azimuths are offsets from each finger's mount direction, so they are re-expressed for the target's mounts."""
    v = np.asarray(v, float).copy()
    if fp != "tripod":
        return v
    xc, yc = v[0], v[1]
    for k, f in enumerate(("thumb", "index", "middle")):
        az = math.atan2(M_src[f][1] - yc, M_src[f][0] - xc) + v[3 + k]
        v[3 + k] = math.atan2(math.sin(az - math.atan2(M_tgt[f][1] - yc, M_tgt[f][0] - xc)),
                              math.cos(az - math.atan2(M_tgt[f][1] - yc, M_tgt[f][0] - xc)))
    return v


def feasible(fp, d, v):
    """The toy's record of grasp v on layout d (mm) if every scaled slack is >= 0 and the clearances hold, else None."""
    import fp_sr2_toy as FP
    pr = FP.Problem(fp, d)
    pr.pick_branches(v)
    sl = float(pr.slacks(v).min())
    if sl < 0.0:
        return None
    clear = FP.clearances(fp, v, pr.M, pr.branch)
    if clear[0] < FP.FF_CLEAR or clear[1] < FP.OBJ_CLEAR:
        return None
    rec = {"ok": True, "v": [round(float(x), 6) for x in v], "branch": dict(pr.branch), "slack": round(sl, 4)}
    rec.update(FP.metrics(fp, v, pr.M, pr.branch))
    return rec


def cmd_derive(a):
    """Grasp sets for the validation and the transfer runs on the grid, from the grid's GPU rows (seeds 0-2):
      grid2dval_<pose><d>_v4b   each layout's foundational pose and best grasp (to run on fresh seeds);
      grid2dnb_<pose><d>_v4b    each layout's own best grasp, its four grid neighbours' best grasps and the keyframe
                                grasp (the best grasp of the layout at the smoothed map's highest peak), each moved as a
                                palm-frame keyframe (transfer_grasp) and kept when feasible; `src` names each grasp's origin.
    The pinch family's files hold the pin-push's grasps under 'pinch' and the press's under 'press'."""
    import fp_sr2_toy as FP
    fams = {"tripod": ("twist", "roll"), "pinch": ("pinpush", "press")}
    for fam, tasks in fams.items():
        shape = "sphere" if fam == "tripod" else "cylinder"
        size = a.d_sphere if fam == "tripod" else a.d_cylinder
        FP.set_sizes(**({"sph_d_mm": size} if shape == "sphere" else {"cyl_d_mm": size}))
        src_rows = {int(r["call"]): r for r in FR.load(FP.size_name("grid2d", shape, size))}
        val, nb = {}, {}
        for task in tasks:
            pose = POSE_KEY[task]
            tab = {r["call"]: r for r in grid_table(task, size)}
            if not any(r.get("n_grasps") for r in tab.values()):
                print(task, size, "no GPU rows yet", flush=True)
                continue
            xs = np.unique([r["x_sep"] for r in tab.values()])
            ys = np.unique([r["y_sep"] for r in tab.values()])
            ny = len(ys)
            xs_, ys_, M_ = grid_maps(task, size)
            sm = smooth_map(M_["best"])
            top = int(np.nanargmax(sm))
            key_call = int((top % len(xs)) * ny + top // len(xs))      # grid index: x major, y minor
            key_g = src_rows[key_call]["fp"][pose]["grasps"][tab[key_call]["best_grasp"]]
            key_M = FP.mounts6(src_rows[key_call]["d"])
            n_feas = {"nb": 0, "nb_tried": 0, "key": 0}
            for call, r in tab.items():
                row = src_rows[call]
                rec = row["fp"].get(pose) or {}
                if not r.get("n_grasps") or not rec.get("ok"):
                    continue
                gs = rec["grasps"]
                vr = val.setdefault(call, {"d": row["d"], "call": call, "xy": row["xy"], "ok": True, "fp": {}})
                vr["fp"][pose] = dict(gs[0], grasps=[dict(gs[0], src="fp"), dict(gs[r["best_grasp"]], src="best")])
                d = np.asarray(row["d"], float)
                M = FP.mounts6(d)
                out = [dict(gs[r["best_grasp"]], src="own")]
                ix, iy = call // ny, call % ny
                for name, (dx, dy) in (("left", (-1, 0)), ("right", (1, 0)), ("down", (0, -1)), ("up", (0, 1))):
                    jx, jy = ix + dx, iy + dy
                    if not (0 <= jx < len(xs) and 0 <= jy < ny):
                        continue
                    nbr = jx * ny + jy
                    if not tab.get(nbr, {}).get("n_grasps"):
                        continue
                    srow = src_rows[nbr]
                    g = srow["fp"][pose]["grasps"][tab[nbr]["best_grasp"]]
                    n_feas["nb_tried"] += 1
                    f = feasible(pose, d, transfer_grasp(pose, g["v"], FP.mounts6(srow["d"]), M))
                    if f is not None:
                        n_feas["nb"] += 1
                        out.append(dict(f, src=name))
                f = feasible(pose, d, transfer_grasp(pose, key_g["v"], key_M, M))
                if f is not None:
                    n_feas["key"] += 1
                    out.append(dict(f, src="keyframe"))
                nr = nb.setdefault(call, {"d": row["d"], "call": call, "xy": row["xy"], "ok": True, "fp": {}})
                nr["fp"][pose] = dict(gs[0], grasps=out, keyframe_call=key_call)
            print(task, size, "keyframe layout", key_call, src_rows[key_call]["xy"], n_feas, flush=True)
        for name, rows in (("grid2dval", val), ("grid2dnb", nb)):
            p = ROWS / f"{FP.size_name(name, shape, size)}.jsonl"
            p.write_text("".join(json.dumps(rows[c]) + "\n" for c in sorted(rows)))
            print("wrote", p.relative_to(ROOT), len(rows), flush=True)
    return 0


T_SHAPE = {"twist": "sphere", "roll": "sphere", "pinpush": "cylinder", "press": "cylinder"}
TASK_NAME = {"twist": "Twist", "roll": "Roll", "pinpush": "Pin-push", "press": "Press"}


def cmd_figures(a):
    for task in TASKS:
        for size in SIZES[task]:
            if (GRID_ROWS / f"{task}_d{size:g}.jsonl").exists():
                fig_grid(task, size)
    return 0


VIEWS = ((90.0, -12.0), (0.0, -12.0), (45.0, 50.0))          # free camera (azimuth, elevation): side, front, from below


def grasp_views(task, size, picks, out, w=360, h=300):
    """Keyframes of grasps at the touch pose (the TPU block meshes on the object, before the 8 mm opening), each from
    the three VIEWS, one row per (call, grasp, label) of picks; a PNG at media/<out>."""
    import mujoco
    import fp_sr2_toy as FP
    import hom_contact_rig as HR
    import hom_gpu as HG
    T, S = HG.T, HG.S
    shape = T.SHAPE[task]
    FP.set_sizes(**({"sph_d_mm": size} if shape == "sphere" else {"cyl_d_mm": size}))
    scene = HG.Scene(shape, size)
    m = scene.m_pt
    poses = grid_poses(task, size)
    rows = []
    ren = mujoco.Renderer(m, h, w)
    try:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, S.Z_OBJ + 0.01]
        cam.distance = 0.24
        for call, j, label in picks:
            xs, ys, rec = poses[call]
            g = (rec.get("grasps") or [rec])[j]
            mxy = HG.mount_xy(FP.layout2(xs, ys))
            if shape == "sphere":
                obj, _tg, n, _pc, _ax = FP.geometry("tripod", np.asarray(g["v"], float), FP.mounts6(FP.layout2(xs, ys)))
                setup = HG.tripod_setup(task, mxy, (np.asarray(obj["centre"], float),
                                                    {f: np.asarray(n[f], float) for f in HG.FINGERS}), size)
            else:
                setup = HG.pinch_setup(task, g["v"], FP.layout2(xs, ys))
            bm = scene.body_mounts(mxy)
            for f in HG.FINGERS:
                m.body_pos[scene.mount_bid[f]] = bm[f]
            S.set_palm(m, setup["palm"])
            _q_open, q_touch = T.open_pose(setup, m)
            dd = mujoco.MjData(m)
            ob = m.body(S.OBJ).id
            qa = m.jnt_qposadr[m.body_jntadr[ob]]
            dd.qpos[qa:qa + 7] = [0.0, 0.0, S.Z_OBJ, *setup["quat"]]
            for f in HG.FINGERS:
                for k, jn in enumerate(HG.JOINTS):
                    dd.qpos[m.jnt_qposadr[m.joint(f"{f}_{jn}").id]] = q_touch[f][k]
            mujoco.mj_forward(m, dd)
            ims = []
            for az, el in VIEWS:
                cam.azimuth, cam.elevation = az, el
                ren.update_scene(dd, cam)
                ims.append(ren.render().copy())
            row = np.concatenate(ims, 1)
            rows.append(HR.annotate(row, f"x_sep {xs:.0f} mm, y_sep {ys:.0f} mm, grasp {j}: {label}", size=16))
    finally:
        ren.close()
    from PIL import Image
    FR.MEDIA.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, 0)).save(FR.MEDIA / out)
    print("wrote", (FR.MEDIA / out).relative_to(ROOT), flush=True)


def cmd_views(a):
    """Keyframes of the foundational pose, the best and the worst grasp (by the mean score over seeds) on the layouts
    a.calls; without GPU rows yet, grasps 0, 1 and the last."""
    sc, _held = grid_scores(a.task, a.d)
    poses = grid_poses(a.task, a.d)
    picks = []
    for call in a.calls:
        rec = poses[call][2]
        if not rec or not rec.get("ok"):
            print(call, "not reached")
            continue
        g = {j: float(np.mean(v)) for (c, j), v in sc.items() if c == call}
        if g:
            jb, jw = max(g, key=g.get), min(g, key=g.get)
            picks += [(call, 0, f"foundational pose, score {g.get(0, float('nan')):.2f}"),
                      (call, jb, f"best, score {g[jb]:.2f}"), (call, jw, f"worst, score {g[jw]:.2f}")]
        else:
            n = len(rec.get("grasps") or [rec])
            picks += [(call, 0, "foundational pose"), (call, 1, "grasp 1"), (call, n - 1, f"grasp {n - 1}")]
    grasp_views(a.task, a.d, picks, a.out or f"20261029-grid_grasps_{a.task}_d{a.d:g}.png")
    return 0


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    sub.add_parser("grid")
    sub.add_parser("figures")
    p = sub.add_parser("derive")
    p.add_argument("--d-sphere", type=float, default=40.0)
    p.add_argument("--d-cylinder", type=float, default=25.0)
    p = sub.add_parser("value")
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--d", type=float, required=True)
    p = sub.add_parser("views")
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--d", type=float, required=True)
    p.add_argument("--calls", type=int, nargs="+", required=True, help="grid indices (x_sep major, y_sep minor)")
    p.add_argument("--out", default="")
    a = ap.parse_args()
    return {"stats": cmd_stats, "grid": cmd_grid, "figures": cmd_figures, "derive": cmd_derive, "views": cmd_views,
            "value": cmd_value}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
