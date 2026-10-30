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
            rng = np.random.default_rng(0)                    # null: the scores shuffled among the reached layouts
            Ri = np.array(sorted(R))
            null = []
            for _ in range(3):
                sh = sc["mean"].copy()
                sh[Ri] = rng.permutation(sh[Ri])
                null.append(len(neighbourhoods(sh, nb)[0]))
            T["sizes"][f"{size:g}"] = {
                "n_neighbourhoods_null": float(np.mean(null)),
                "neighbourhood_size_median": float(np.median([len(m) for _, m, _ in nbh])) if nbh else None,
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


PEAK_RADII, PEAK_DIRS, PEAK_TOP = (2.0, 4.0, 8.0), 10, 20
LOCAL_RADIUS, LOCAL_N = 2.0, 64
MOVE_ROWS = {"trust": "valtr", "peak": "peak", "local": "local"}


def local_offsets(n, radius, seed):
    """n scrambled-Sobol points spread evenly over the six-dimensional ball of the given radius (mm): six Sobol
    coordinates give a direction through the normal quantile function, the seventh the distance (radius u^(1/6))."""
    from scipy.stats import norm, qmc
    u = qmc.Sobol(7, scramble=True, seed=seed).random(n)
    g = norm.ppf(np.clip(u[:, :6], 1e-9, 1.0 - 1e-9))
    return radius * u[:, 6:] ** (1.0 / 6.0) * g / np.linalg.norm(g, axis=1, keepdims=True)


def cmd_moves(a):
    """Source layouts for fp_sr2_toy.py poses (--src <task>...). kind trust: the value model's trust-region moves of the
    best 200 Sobol layouts and their starts (rows/valtr_<task><d>.jsonl). kind peak: the PEAK_TOP starts that scored
    best on the trust rows' fresh seeds, each with PEAK_DIRS random directions at each of PEAK_RADII mm in the six
    mount coordinates (rows/peak_<task><d>.jsonl). kind local: the same starts, each with LOCAL_N layouts of a local
    Sobol design within LOCAL_RADIUS mm (rows/local_<task><d>.jsonl)."""
    import fp_sr2_toy as FP
    tag = f"{a.task}{a.d:g}"
    rows = []

    def add(d, **kw):
        d = np.asarray(d, float)
        rows.append(dict({"d": d.round(3).tolist(), "call": len(rows),
                          "z": FP.standard(d, FP.LO6, FP.HI6).round(5).tolist()}, **kw))
    if a.kind == "trust":
        v = json.loads((SIZE_ROWS / f"value_{a.task}_d{a.d:g}.json").read_text())
        for L_ in v["trust"]["layouts"]:
            add(L_["d"], src_call=L_["call"], role="start", pred=L_["pred"])
            add(L_["d_new"], src_call=L_["call"], role="moved", pred=L_["pred_new"])
    elif a.kind == "local":
        for k, r in enumerate(x for x in FR.load(f"peak_{tag}") if x["radius_mm"] == 0.0):
            d0 = np.asarray(r["d"], float)
            add(d0, center=r["center"], radius_mm=0.0)
            for off in local_offsets(LOCAL_N, LOCAL_RADIUS, seed=k):
                d1 = np.clip(d0 + off, FP.LO6, FP.HI6)
                add(d1, center=r["center"], radius_mm=round(float(np.linalg.norm(d1 - d0)), 3))
    else:
        src = {r["call"]: r for r in FR.load(f"valtr_{tag}")}
        sc, _ = grid_scores(a.task, a.d, ROWS / "valtr", f"{a.task}_d{a.d:g}_s345")
        best = sorted(((float(np.mean(s)), c) for (c, _g), s in sc.items() if src[c]["role"] == "start" and len(s) == 3),
                      reverse=True)[:PEAK_TOP]
        rng = np.random.default_rng(7)
        for _s, c in best:
            d0 = np.asarray(src[c]["d"], float)
            add(d0, center=c, radius_mm=0.0)
            for rad in PEAK_RADII:
                for _ in range(PEAK_DIRS):
                    u = rng.normal(size=6)
                    add(np.clip(d0 + rad * u / np.linalg.norm(u), FP.LO6, FP.HI6), center=c, radius_mm=rad)
    path = ROWS / f"{MOVE_ROWS[a.kind]}_{tag}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} layouts -> {path.relative_to(ROOT)}", flush=True)
    return 0


EA_POP, EA_DELTA, EA_RUNS, EA_GENS = 16, 30.0, 200, 400     # population, mutation width (mm), runs, generation cap
EA_TASKS = (("twist", 40.0), ("roll", 40.0), ("pinpush", 25.0), ("press", 25.0))
EA_EPS = (0.05, 0.02)                                     # score margins below the grid's best


def ea_landscape(task, size):
    """The grid as a lookup evaluator: {(x_sep, y_sep): (fitness, rollouts)} for the fitness 'best' (the best grasp of
    the set, every grasp's three seeds run) and 'fp' (the foundational pose's three seeds); a layout whose pose is not
    reached scores 0 at no rollout. Also each cell's basin on the smoothed best-of-16 map and the best basin's peak (the
    highest smoothed peak), the grid's best score and its total rollouts."""
    tab = grid_table(task, size)
    sc, _ = grid_scores(task, size)
    n_run, n_fp = {}, {}
    for (c, j), v in sc.items():
        n_run[c] = n_run.get(c, 0) + len(v)
        if j == 0:
            n_fp[c] = len(v)
    look = {"best": {}, "fp": {}}
    for r in tab:
        k = (float(r["x_sep"]), float(r["y_sep"]))
        ok = bool(r.get("n_grasps"))
        look["best"][k] = (float(r["best"]) if ok else 0.0, n_run.get(r["call"], 0) if ok else 0)
        look["fp"][k] = (float(r["fp"]) if ok and np.isfinite(r["fp"]) else 0.0, n_fp.get(r["call"], 0) if ok else 0)
    xs, ys, M = grid_maps(task, size)
    sm = smooth_map(M["best"])
    pk, pks = peaks(sm)
    top = max(pks, key=lambda p: sm.flat[p])
    basin = {(float(x), float(y)): int(pk[i * len(xs) + j]) for i, y in enumerate(ys) for j, x in enumerate(xs)}
    return look, basin, top, float(np.nanmax(M["best"])), sum(n for _f, n in look["best"].values()), xs, ys


def ea_search(look, rng, xs, ys, pop=EA_POP, delta=EA_DELTA, budget=None, random=False):
    """One run of the elitist mutation-only search of Yang et al. (TEVC 2026) on the grid's lattice: the best half of
    each generation is kept, the best eighth are parents, and each child is a parent moved by U(-delta/2, delta/2) mm
    per coordinate, rounded to the 5 mm lattice and clipped to the grid. A layout is evaluated once (later visits reuse
    its score). random=True draws layouts uniformly without replacement instead. Returns [(rollouts so far, layout)]
    for every new evaluation."""
    keys = list(look)
    seen, cost, hist = {}, 0, []

    def ev(k):
        nonlocal cost
        if k not in seen:
            f, n = look[k]
            seen[k] = f
            cost += n
            hist.append((cost, k))
        return seen[k]
    if random:
        for i in rng.permutation(len(keys)):
            ev(keys[i])
            if budget and cost >= budget:
                break
        return hist, seen
    lo, hi = np.array([xs[0], ys[0]], float), np.array([xs[-1], ys[-1]], float)
    P = [keys[i] for i in rng.choice(len(keys), pop, replace=False)]
    for k in P:
        ev(k)
    for _ in range(EA_GENS):
        if budget and cost >= budget:
            break
        P = sorted(P, key=lambda k: -seen[k])
        keep, parents = P[:pop // 2], P[:max(1, pop // 8)]
        kids = []
        for i in range(pop - len(keep)):
            m = np.asarray(parents[i % len(parents)]) + rng.uniform(-delta / 2, delta / 2, 2)
            m = np.clip(5.0 * np.round(m / 5.0), lo, hi)
            kids.append((float(m[0]), float(m[1])))
        for k in kids:
            ev(k)
        P = keep + kids
    return hist, seen


def cmd_ea(a):
    """The evolutionary search of Yang et al. against random search and the full grid, on the grid's stored rollouts
    (seeds 0-2): rollouts until the incumbent (the best layout evaluated so far, by its own fitness) scores within 0.05
    and 0.02 of the grid's best best-of-16 score, for populations of EA_POP and 4 EA_POP. Writes rows/ea/summary.json."""
    out = {}
    for task, size in EA_TASKS:
        look, basin, _top, best, total, xs, ys = ea_landscape(task, size)
        best_cell = max(look["best"], key=lambda k: look["best"][k][0])
        reached = [k for k, (f, n) in look["best"].items() if n > 0]
        res = {"grid_rollouts": total, "grid_best": best, "grid_best_cell": list(best_cell), "layouts_reached": len(reached),
               "layouts_within": {f"{e:g}": int(sum(look["best"][k][0] >= best - e for k in reached)) for e in EA_EPS},
               "runs": {}}
        cases = [("ea", "best", EA_POP), ("ea", "best", 4 * EA_POP), ("random", "best", 0), ("ea", "fp", EA_POP),
                 ("random", "fp", 0)]
        for name, fit, pop in cases:
            hits = {e: [] for e in EA_EPS}
            finals, other, curves = [], [], []
            for run in range(EA_RUNS):
                rng = np.random.default_rng(1000 * run + 7)
                hist, _seen = ea_search(look[fit], rng, xs, ys, pop=pop or EA_POP, budget=total, random=name == "random")
                inc, t_hit, curve = None, {e: None for e in EA_EPS}, []
                for cost, k in hist:
                    if inc is None or look[fit][k][0] > look[fit][inc][0]:
                        inc = k
                    true = look["best"][inc][0]                  # the incumbent's best-of-16 score
                    curve.append((cost, true))
                    for e in EA_EPS:
                        if t_hit[e] is None and true >= best - e:
                            t_hit[e] = cost
                for e in EA_EPS:
                    hits[e].append(t_hit[e])
                finals.append(look["best"][inc][0])
                other.append(basin[inc] != basin[best_cell])
                curves.append(curve)
            pts = np.linspace(0, total, 61)
            med = [float(np.median([max([s_ for c, s_ in cv if c <= g] or [0.0]) for cv in curves])) for g in pts]

            def q(h):
                v = np.sort(np.array([x if x is not None else np.inf for x in h], float))
                pc = (lambda p: float(np.percentile(v, p)) if np.isfinite(np.percentile(v, p)) else None)
                return {"reached": float(np.mean(np.isfinite(v))), "median": pc(50), "q25": pc(25), "q75": pc(75)}
            key = f"{name}_{fit}" + (f"_pop{pop}" if name == "ea" else "")
            res["runs"][key] = {**{f"to_within_{e:g}": q(hits[e]) for e in EA_EPS},
                                "final_median": float(np.median(finals)), "final_other_basin": float(np.mean(other)),
                                "curve": {"rollouts": pts.round(0).tolist(), "median_best": med}}
            r_ = res["runs"][key]
            print(f"{task} {size:g} mm {key}: " + ", ".join(f"within {e:g}: {r_[f'to_within_{e:g}']}" for e in EA_EPS)
                  + f"; final median {r_['final_median']:.3f}, other basin {r_['final_other_basin']:.2f}; grid {total}",
                  flush=True)
        out[f"{task}_d{size:g}"] = res
    (ROWS / "ea").mkdir(parents=True, exist_ok=True)
    (ROWS / "ea" / "summary.json").write_text(json.dumps({"pop": EA_POP, "delta_mm": EA_DELTA, "runs": EA_RUNS, **out},
                                                         indent=1))
    return 0


DRAWS = ROWS / "grid2d_v4b_draws12"
DRAW_SEEDS = tuple(range(6, 18))


def seed_scores(task, rows_dir, name):
    """{(call, grasp): {seed: score}} of a row file."""
    import hom_rows
    out = {}
    p = rows_dir / f"{name}.jsonl"
    for x in (p.read_text().splitlines() if p.exists() else []):
        if x.strip():
            r = hom_rows.rescale_metrics(json.loads(x))
            if r.get("status", "ok") == "ok":
                out.setdefault((int(r["call"]), int(r.get("grasp", 0))), {})[int(r["seed"])] = rollout_score(task, r)
    return out


def cmd_draws(a):
    """The cylinder tasks over twelve object-pose draws (seeds 6-17) of each grid layout's foundational pose and best
    grasp, the best picked on seeds 0-2: the seeds' grid means, the best grasp's gain over the foundational pose against
    their spread, and the layouts' ranking on three draws against twelve. Writes rows/grid2d_v4b_draws12/summary.json."""
    out = {}
    for task in ("pinpush", "press"):
        for size in (25.0, 10.0, 40.0):
            S12 = seed_scores(task, DRAWS, f"{task}_d{size:g}")
            if not S12:
                continue
            S35 = seed_scores(task, GRID_ROWS, f"val_{task}_d{size:g}")
            S02 = seed_scores(task, GRID_ROWS, f"{task}_d{size:g}")
            best_g = {r["call"]: r["best_grasp"] for r in grid_table(task, size) if r.get("n_grasps")}
            calls = sorted(c for c in best_g if all(len(S12.get((c, g), {})) == len(DRAW_SEEDS) for g in (0, 1)))

            def arr(S, g, seeds, picked=False):
                return np.array([[S[(c, best_g[c] if picked else g)][s] for s in seeds] for c in calls])
            fp = {"s012": arr(S02, 0, (0, 1, 2)), "s345": arr(S35, 0, (3, 4, 5)), "s6_17": arr(S12, 0, DRAW_SEEDS)}
            be = {"s012": arr(S02, None, (0, 1, 2), picked=True), "s345": arr(S35, 1, (3, 4, 5)),
                  "s6_17": arr(S12, 1, DRAW_SEEDS)}
            seeds = list(range(18))
            m_fp = np.concatenate([fp["s012"].mean(0), fp["s345"].mean(0), fp["s6_17"].mean(0)])
            m_be = np.concatenate([be["s012"].mean(0), be["s345"].mean(0), be["s6_17"].mean(0)])
            gain12 = be["s6_17"].mean(1) - fp["s6_17"].mean(1)
            g_seed = be["s6_17"].mean(0) - fp["s6_17"].mean(0)
            top = max(1, int(round(TOP * len(calls))))
            t012 = set(np.argsort(-be["s012"].mean(1))[:top])
            t12 = set(np.argsort(-be["s6_17"].mean(1))[:top])
            res = {"layouts": len(calls), "seed_mean_fp": m_fp.round(4).tolist(), "seed_mean_best": m_be.round(4).tolist(),
                   "seeds": seeds, "fp_seed_sd_6_17": float(fp["s6_17"].mean(0).std(ddof=1)),
                   "fp_seed_range_6_17": float(np.ptp(fp["s6_17"].mean(0))),
                   "gain_median_6_17": float(np.median(gain12)), "gain_positive_6_17": float(np.mean(gain12 > 0)),
                   "gain_seed_means_6_17": g_seed.round(4).tolist(),
                   "gain_median_012": float(np.median(be["s012"].mean(1) - fp["s012"].mean(1))),
                   "gain_median_345": float(np.median(be["s345"].mean(1) - fp["s345"].mean(1))),
                   "fp_median_6_17": float(np.median(fp["s6_17"].mean(1))),
                   "best_median_6_17": float(np.median(be["s6_17"].mean(1))),
                   "split_half_best": FR.spearman(be["s6_17"][:, :6].mean(1), be["s6_17"][:, 6:].mean(1)),
                   "split_half_fp": FR.spearman(fp["s6_17"][:, :6].mean(1), fp["s6_17"][:, 6:].mean(1)),
                   "best_012_vs_6_17": FR.spearman(be["s012"].mean(1), be["s6_17"].mean(1)),
                   "best_345_vs_6_17": FR.spearman(be["s345"].mean(1), be["s6_17"].mean(1)),
                   "fp_345_vs_6_17": FR.spearman(fp["s345"].mean(1), fp["s6_17"].mean(1)),
                   "top_tenth_012_kept": float(len(t012 & t12) / top),
                   "held_all12_fp": float(np.mean((fp["s6_17"] > 0).all(1))),
                   "held_all12_best": float(np.mean((be["s6_17"] > 0).all(1)))}
            out[f"{task}_d{size:g}"] = res
            print(task, size, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()
                               if not isinstance(v, list)}, flush=True)
    DRAWS.mkdir(parents=True, exist_ok=True)
    (DRAWS / "summary.json").write_text(json.dumps(out, indent=1))
    return 0


def cmd_local_report(a):
    """The local refinement: per start, the best of its LOCAL_N neighbours on seeds 9-11 against the start, both scored
    again on seeds 12-14 (a pose not reached scores 0). Writes rows/local/summary_<task><d>.json."""
    from scipy.stats import wilcoxon
    tag = f"{a.task}{a.d:g}"
    rows = {r["call"]: r for r in FR.load(f"local_{tag}")}
    pick, _ = grid_scores(a.task, a.d, ROWS / "local", f"{a.task}_d{a.d:g}_s91011")
    check, _ = grid_scores(a.task, a.d, ROWS / "local", f"{a.task}_d{a.d:g}_s121314")

    def mean(sc, c):
        return float(np.mean(sc[(c, 0)])) if (c, 0) in sc else 0.0
    out = []
    for cen in sorted({r["center"] for r in rows.values()}):
        calls = [c for c, r in rows.items() if r["center"] == cen]
        c0 = next(c for c in calls if rows[c]["radius_mm"] == 0.0)
        nb = [c for c in calls if c != c0]
        best = max(nb, key=lambda c: mean(pick, c))
        out.append({"center": cen, "d": rows[c0]["d"], "start_pick": mean(pick, c0), "start_check": mean(check, c0),
                    "best_call": best, "best_d": rows[best]["d"], "best_radius_mm": rows[best]["radius_mm"],
                    "best_pick": mean(pick, best), "best_check": mean(check, best),
                    "reached": float(np.mean([(c, 0) in pick for c in nb])),
                    "nb_pick_median": float(np.median([mean(pick, c) for c in nb])),
                    "better_than_start_pick": float(np.mean([mean(pick, c) > mean(pick, c0) for c in nb]))})
    gain = np.array([o["best_check"] - o["start_check"] for o in out])
    summ = {"starts": len(out), "best_beats_start_check": float(np.mean(gain > 0)), "gain_check_median": float(np.median(gain)),
            "gain_pick_median": float(np.median([o["best_pick"] - o["start_pick"] for o in out])),
            "wilcoxon_p": float(wilcoxon(gain).pvalue) if np.any(gain != 0) else 1.0,
            "start_check_median": float(np.median([o["start_check"] for o in out])),
            "best_check_median": float(np.median([o["best_check"] for o in out])),
            "reached_median": float(np.median([o["reached"] for o in out])), "per_start": out}
    (ROWS / "local").mkdir(parents=True, exist_ok=True)
    (ROWS / "local" / f"summary_{tag}.json").write_text(json.dumps(summ, indent=1))
    print({k: v for k, v in summ.items() if k != "per_start"}, flush=True)
    return 0


def cmd_moves_report(a):
    """The two move experiments on fresh seeds: the trust-region moves against their starts (seeds 3-5), and the score
    change around the best layouts by perturbation radius (seeds 6-8; a pose not reached scores 0). Writes
    rows/peak/summary_<task><d>.json."""
    from scipy.stats import wilcoxon
    tag = f"{a.task}{a.d:g}"
    out = {}
    src = {r["call"]: r for r in FR.load(f"valtr_{tag}")}
    sc, _ = grid_scores(a.task, a.d, ROWS / "valtr", f"{a.task}_d{a.d:g}_s345")
    pairs = {}
    for (c, _g), s in sc.items():
        pairs.setdefault(src[c]["src_call"], {})[src[c]["role"]] = (float(np.mean(s)), src[c]["pred"])
    P = np.array([(v["start"][0], v["moved"][0], v["start"][1], v["moved"][1]) for v in pairs.values()
                  if "start" in v and "moved" in v])
    dm, dp = P[:, 1] - P[:, 0], P[:, 3] - P[:, 2]
    full = scores(a.task, a.d, len(layouts()[0]))["mean"]
    out["trust"] = {"pairs": len(P), "start_median": float(np.median(P[:, 0])), "moved_median": float(np.median(P[:, 1])),
                    "better": float(np.mean(dm > 0)), "worse": float(np.mean(dm < 0)), "change_median": float(np.median(dm)),
                    "pred_change_median": float(np.median(dp)), "pred_vs_measured_r": float(np.corrcoef(dp, dm)[0, 1]),
                    "wilcoxon_p": float(wilcoxon(dm).pvalue),
                    "starts_seeds012_median": float(np.median([full[v["src_call"]] for v in src.values()
                                                               if v["role"] == "start"])),
                    "data": P.round(4).tolist()}
    pk = {r["call"]: r for r in FR.load(f"peak_{tag}")}
    reach_ = {int(r["call"]): bool((r["fp"].get(POSE_KEY[a.task]) or {}).get("ok"))
              for r in FR.load(f"peak_{tag}_{'tripod' if POSE_ROWS[a.task] == 'tripod' else 'pinch'}{a.d:g}_v4b")}
    sp, _ = grid_scores(a.task, a.d, ROWS / "peak", f"{a.task}_d{a.d:g}_s678")
    s_ = {c: float(np.mean(sp[(c, 0)])) if (c, 0) in sp else 0.0 for c in pk}
    cent = {pk[c]["center"]: s_[c] for c in pk if pk[c]["radius_mm"] == 0.0}
    out["peak"] = {"centres": len(cent), "centre_median": float(np.median(list(cent.values()))), "radii": {}}
    for rad in PEAK_RADII:
        ids = [c for c in pk if pk[c]["radius_mm"] == rad]
        ch = np.array([s_[c] - cent[pk[c]["center"]] for c in ids])
        out["peak"]["radii"][f"{rad:g}"] = {"n": len(ch), "pose_lost": float(np.mean([not reach_.get(c, False) for c in ids])),
                                            "change_median": float(np.median(ch)), "better": float(np.mean(ch > 0.02)),
                                            "worse_0.1": float(np.mean(ch < -0.1)), "worse_0.3": float(np.mean(ch < -0.3)),
                                            "changes": ch.round(4).tolist()}
    (ROWS / "peak").mkdir(parents=True, exist_ok=True)
    (ROWS / "peak" / f"summary_{tag}.json").write_text(json.dumps(out, indent=1))
    print({k: {k2: v2 for k2, v2 in v.items() if k2 not in ("data", "radii")} for k, v in out.items()}, flush=True)
    return 0


def fig_value(task, size, out=None):
    """Design gradients on one task and object, three panels: steps of 10 mm on the measured 2-D grid along the value
    model's gradient and in random directions; the trust-region moves of the best Sobol layouts on fresh seeds, measured
    against predicted change; the score change around the best layouts by perturbation radius, with the share of
    perturbations that lose the pose."""
    plt = FR.plt_setup()
    tag = f"{task}{size:g}"
    v = json.loads((SIZE_ROWS / f"value_{task}_d{size:g}.json").read_text())
    pk = json.loads((ROWS / "peak" / f"summary_{tag}.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)
    ax = axs[0]
    gs = v["grid_steps"]
    for k, (name, lab) in enumerate((("gradient", "along the gradient"), ("random", "random direction"))):
        ax.bar([k - 0.18], [gs[name]["raised"]], 0.34, color="#2ca02c", label="raised" if k == 0 else None)
        ax.bar([k + 0.18], [gs[name]["lowered"]], 0.34, color="#d62728", label="lowered" if k == 0 else None)
        ax.text(k, max(gs[name]["raised"], gs[name]["lowered"]) + 0.03, f"median {gs[name]['median_change']:+.3f}\n"
                f"n = {gs[name]['steps']}", ha="center", fontsize=11)
    ax.set_xticks([0, 1], ["along the gradient", "random direction"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of 10 mm steps")
    ax.set_title("(a) grid: measured score after a step")
    ax.legend(loc="upper right")
    ax = axs[1]
    P = np.array(pk["trust"]["data"])
    ax.scatter(P[:, 3] - P[:, 2], P[:, 1] - P[:, 0], s=14, color="k", alpha=0.6, lw=0)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("predicted change of the score")
    ax.set_ylabel("measured change, fresh seeds")
    ax.set_title(f"(b) best Sobol layouts moved 3-5 mm ({pk['trust']['pairs']} pairs)")
    ax = axs[2]
    rad = [0.0] + [float(r) for r in pk["peak"]["radii"]]
    med = [0.0] + [pk["peak"]["radii"][r]["change_median"] for r in pk["peak"]["radii"]]
    q1 = [0.0] + [float(np.percentile(pk["peak"]["radii"][r]["changes"], 25)) for r in pk["peak"]["radii"]]
    q3 = [0.0] + [float(np.percentile(pk["peak"]["radii"][r]["changes"], 75)) for r in pk["peak"]["radii"]]
    ax.fill_between(rad, q1, q3, color="#1f77b4", alpha=0.2, lw=0, label="quartiles")
    ax.plot(rad, med, "o-", color="#1f77b4", label="median")
    ax.set_xlabel("perturbation of the mounts (mm)")
    ax.set_ylabel("score change against the centre")
    ax2 = ax.twinx()
    ax2.bar(rad[1:], [pk["peak"]["radii"][r]["pose_lost"] for r in pk["peak"]["radii"]], 0.8, color="0.6", alpha=0.6)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("share losing the pose (bars)")
    ax.set_title(f"(c) around the best {pk['peak']['centres']} layouts")
    ax.legend(loc="upper right")
    FR.save(fig, out or f"20261030-value_{task}_d{size:g}.png")


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
    """Grasp sets for the validation and the transfer runs on the grid, one file per task, from the grid's GPU rows
    (seeds 0-2):
      grid2dval_<task>_<pose><d>_v4b   each layout's foundational pose and best grasp (to run on fresh seeds);
      grid2dnb_<task>_<pose><d>_v4b    each layout's own best grasp, its four grid neighbours' best grasps and the
                                       keyframe grasp (the best grasp of the layout at the smoothed map's highest peak),
                                       each moved as a palm-frame keyframe (transfer_grasp) and kept when feasible; `src`
                                       names each grasp's origin.
    hom_gpu.py reads them as --layouts grid2dval_<task>_size and grid2dnb_<task>_size."""
    import fp_sr2_toy as FP
    fams = {"tripod": ("twist", "roll"), "pinch": ("pinpush", "press")}
    for fam, tasks in fams.items():
        shape = "sphere" if fam == "tripod" else "cylinder"
        size = a.d_sphere if fam == "tripod" else a.d_cylinder
        FP.set_sizes(**({"sph_d_mm": size} if shape == "sphere" else {"cyl_d_mm": size}))
        src_rows = {int(r["call"]): r for r in FR.load(FP.size_name("grid2d", shape, size))}
        for task in tasks:
            val, nb = {}, {}
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
            fj = GRID_ROWS / "transfer_feasible.json"
            allf = json.loads(fj.read_text()) if fj.exists() else {}
            allf[f"{task}_d{size:g}"] = dict(n_feas, layouts=len(val), keyframe_call=key_call,
                                              keyframe_xy=src_rows[key_call]["xy"])
            GRID_ROWS.mkdir(parents=True, exist_ok=True)
            fj.write_text(json.dumps(allf, indent=1))
            for name, rows in (("grid2dval", val), ("grid2dnb", nb)):
                p = ROWS / f"{FP.size_name(f'{name}_{task}', shape, size)}.jsonl"
                p.write_text("".join(json.dumps(rows[c]) + "\n" for c in sorted(rows)))
                print("wrote", p.relative_to(ROOT), len(rows), flush=True)
    return 0


def cmd_valrep(a):
    """Queue 2's rows on fresh seeds (3-5) for each task and object whose rows exist: (val) each layout's foundational
    pose and best grasp, against the gain the grid's seeds 0-2 gave; (nb) each layout's own best grasp against its
    neighbours' best grasps and the keyframe grasp moved to it. Writes rows/grid2d_v4b/valrep.json."""
    import fp_sr2_toy as FP
    out = {}
    for task in TASKS:
        shape = "sphere" if POSE_ROWS[task] == "tripod" else "cylinder"
        for size in SIZES[task]:
            vp, npth = GRID_ROWS / f"val_{task}_d{size:g}.jsonl", GRID_ROWS / f"nb_{task}_d{size:g}.jsonl"
            if not vp.exists():
                continue
            sc, _ = grid_scores(task, size, GRID_ROWS, f"val_{task}_d{size:g}")
            tab = {r["call"]: r for r in grid_table(task, size)}
            fp_, be, g0 = [], [], []
            for call, r in tab.items():
                if (call, 0) in sc and (call, 1) in sc and r.get("n_grasps"):
                    fp_.append(np.mean(sc[(call, 0)]))
                    be.append(np.mean(sc[(call, 1)]))
                    g0.append((r["fp"], r["best"]))
            fp_, be, g0 = np.array(fp_), np.array(be), np.array(g0)
            R = {"layouts": len(fp_), "fp_fresh_median": float(np.median(fp_)), "best_fresh_median": float(np.median(be)),
                 "best_beats_fp": float(np.mean(be > fp_)), "gain_fresh_median": float(np.median(be - fp_)),
                 "gain_seeds012_median": float(np.median(g0[:, 1] - g0[:, 0])),
                 "best_repeat_spearman": FR.spearman(g0[:, 1], be),
                 "best_seeds012_median": float(np.median(g0[:, 1]))}
            if npth.exists():
                src = {int(r["call"]): r for r in FR.load(FP.size_name(f"grid2dnb_{task}", shape, size))}
                sn, _ = grid_scores(task, size, GRID_ROWS, f"nb_{task}_d{size:g}")
                d_nb, d_key, n_key = [], [], 0
                for call, row in src.items():
                    gl = row["fp"][POSE_KEY[task]]["grasps"]
                    own = sn.get((call, 0))
                    if own is None:
                        continue
                    own = float(np.mean(own))
                    for j, g in enumerate(gl[1:], 1):
                        if (call, j) not in sn:
                            continue
                        v = float(np.mean(sn[(call, j)]))
                        (d_key if g["src"] == "keyframe" else d_nb).append(v - own)
                d_nb, d_key = np.array(d_nb), np.array(d_key)
                R.update(nb_n=len(d_nb), nb_within_0_1=float(np.mean(d_nb >= -0.1)) if len(d_nb) else None,
                         nb_change_median=float(np.median(d_nb)) if len(d_nb) else None,
                         nb_better=float(np.mean(d_nb > 0.02)) if len(d_nb) else None,
                         key_n=len(d_key), key_change_median=float(np.median(d_key)) if len(d_key) else None)
            out[f"{task}_d{size:g}"] = R
            print(task, size, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R.items()}, flush=True)
    fresh = {}                                  # cmd_cross_grid's worst of the four tasks, on seeds 3-5
    for task, size in DEFAULT.items():
        if (GRID_ROWS / f"val_{task}_d{size:g}.jsonl").exists():
            sc, _ = grid_scores(task, size, GRID_ROWS, f"val_{task}_d{size:g}")
            fresh[task] = {c: (np.mean(sc[(c, 0)]), np.mean(sc[(c, 1)])) for c, j in sc if j == 0 and (c, 1) in sc}
    if len(fresh) == 4:
        calls = sorted(set.intersection(*[set(v) for v in fresh.values()]))
        W = np.array([[[fresh[t][c][k] for t in DEFAULT] for c in calls] for k in (0, 1)]).min(2)
        out["cross_fresh"] = {"layouts": len(calls), "share_all_above_0.5_fp": float(np.mean(W[0] > 0.5)),
                              "share_all_above_0.5_best": float(np.mean(W[1] > 0.5)),
                              "maximin_median_fp": float(np.median(W[0])), "maximin_median_best": float(np.median(W[1]))}
        print("four tasks, seeds 3-5:", out["cross_fresh"], flush=True)
    import hom_rows
    seeds = {}                                  # the foundational pose's mean score per seed (a seed's object offset
    for task, size in DEFAULT.items():          # and yaw are the same on every layout; seed 0 has none)
        per = {}
        for name in (f"{task}_d{size:g}", f"val_{task}_d{size:g}"):
            p = GRID_ROWS / f"{name}.jsonl"
            for x in (p.read_text().splitlines() if p.exists() else []):
                r = hom_rows.rescale_metrics(json.loads(x)) if x.strip() else {}
                if r.get("status", "ok") == "ok" and r and int(r.get("grasp", 0)) == 0:
                    per.setdefault(int(r["seed"]), []).append(rollout_score(task, r))
        seeds[task] = {str(sd): round(float(np.mean(v)), 3) for sd, v in sorted(per.items())}
    out["fp_seed_means"] = seeds
    print("foundational pose, mean per seed:", seeds, flush=True)
    (GRID_ROWS / "valrep.json").write_text(json.dumps(out, indent=1))
    return 0


T_SHAPE = {"twist": "sphere", "roll": "sphere", "pinpush": "cylinder", "press": "cylinder"}
TASK_NAME = {"twist": "Twist", "roll": "Roll", "pinpush": "Pin-push", "press": "Press"}


COORD_LABELS = ("thumb x", "thumb y", "index x", "index y", "middle x", "middle y")


def fig_grid_sizes(task, out=None):
    """The grid's best-of-16 score of one task at each object size whose grid ran, smoothed (7.5 mm), with its basin
    peaks (stars; the largest basin's peak circled); one panel per size on a shared colour scale."""
    sizes = [sz for sz in SIZES[task] if (GRID_ROWS / f"{task}_d{sz:g}.jsonl").exists()]
    if len(sizes) < 2:
        return None
    plt = FR.plt_setup()
    fig, axs = plt.subplots(1, len(sizes), figsize=(5.2 * len(sizes), 5.0), constrained_layout=True)
    maps, peaks_out = [], {}
    for sz in sizes:
        xs, ys, M = grid_maps(task, sz)
        maps.append((xs, ys, smooth_map(M["best"])))
    vmax = max(float(np.nanmax(m)) for _, _, m in maps)
    for ax, sz, (xs, ys, sm) in zip(np.atleast_1d(axs), sizes, maps):
        ext = [xs[0] - 2.5, xs[-1] + 2.5, ys[0] - 2.5, ys[-1] + 2.5]
        im = ax.imshow(sm, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        pk, big = peaks(sm)
        X, Y = np.meshgrid(xs, ys)
        ax.plot(X.flat[big], Y.flat[big], "r*", ms=13, mec="k")
        if big:
            main = max(big, key=lambda q: int((pk == q).sum()))
            ax.plot(X.flat[main], Y.flat[main], "o", ms=20, mfc="none", mec="w", mew=2)
            peaks_out[f"{sz:g}"] = [[float(X.flat[q]), float(Y.flat[q]), float(sm.flat[q]), int((pk == q).sum())]
                                    for q in big]
        ax.set_title(f"{sz:g} mm {T_SHAPE[task]}")
        ax.set_xlabel("x_sep (mm)")
        ax.set_ylabel("y_sep (mm)")
    fig.colorbar(im, ax=axs, shrink=0.8, label="best of 16, smoothed")
    fig.suptitle(f"{TASK_NAME[task]}: the grid's best grasp score across object sizes (circle: the largest basin's peak)")
    FR.save(fig, out or f"20261030-grid_sizes_{task}.png")
    return peaks_out
DEFAULT = {"twist": 40.0, "roll": 40.0, "pinpush": 25.0, "press": 25.0}


def pareto(F):
    """Indices of the rows of F (n x m, larger is better) that no other row beats on every column."""
    order = np.argsort(-F.sum(1))
    keep = []
    for i in order:
        if not any(np.all(F[j] >= F[i]) and np.any(F[j] > F[i]) for j in keep):
            keep.append(i)
    return np.array(sorted(keep))


def cmd_cross_grid(a):
    """The four tasks on the 2-D grid at the default objects, each with its best grasp of 16: the worst of the four
    best-of-16 scores per layout (a generalist's map), the Pareto set over the four, and the same with the foundational
    pose. Writes rows/grid2d_v4b/cross_grid.json and media/20261030-cross_grid.png."""
    tabs = {t: {r["call"]: r for r in grid_table(t, sz) if r.get("n_grasps")} for t, sz in DEFAULT.items()}
    calls = sorted(set.intersection(*[set(v) for v in tabs.values()]))
    tasks = list(DEFAULT)
    B = np.array([[tabs[t][c]["best"] for t in tasks] for c in calls])
    Fp = np.array([[tabs[t][c]["fp"] for t in tasks] for c in calls])
    xy = np.array([[tabs[tasks[0]][c]["x_sep"], tabs[tasks[0]][c]["y_sep"]] for c in calls])
    Pb, Pf = pareto(B), pareto(Fp)
    mb, mf = B.min(1), Fp.min(1)
    ib, if_ = int(np.argmax(mb)), int(np.argmax(mf))
    out = {"layouts": len(calls), "pareto_best": len(Pb), "pareto_fp": len(Pf),
           "maximin_best": {"xy": xy[ib].tolist(), "scores": B[ib].round(3).tolist()},
           "maximin_fp": {"xy": xy[if_].tolist(), "scores": Fp[if_].round(3).tolist()},
           "maximin_median_best": float(np.median(mb)), "maximin_median_fp": float(np.median(mf)),
           "share_all_above_0.5_best": float(np.mean(mb > 0.5)), "share_all_above_0.5_fp": float(np.mean(mf > 0.5))}
    (GRID_ROWS / "cross_grid.json").write_text(json.dumps(out, indent=1))
    plt = FR.plt_setup()
    xs, ys = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    ext = [xs[0] - 2.5, xs[-1] + 2.5, ys[0] - 2.5, ys[-1] + 2.5]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    for ax, (M_, P_, lab) in zip(axs, ((mf, Pf, "foundational pose"), (mb, Pb, "best of 16 per task"))):
        Z = np.full((len(ys), len(xs)), np.nan)
        for (x, y_), v in zip(xy, M_):
            Z[int(np.searchsorted(ys, y_)), int(np.searchsorted(xs, x))] = v
        im = ax.imshow(Z, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=max(float(np.nanmax(mb)), 0.01),
                       aspect="equal")
        ax.plot(xy[P_, 0], xy[P_, 1], "o", ms=5, mfc="none", mec="r", mew=1.2, label="Pareto set over the four tasks")
        k = int(np.argmax(M_))
        ax.plot(*xy[k], "w*", ms=16, mec="k", label="best worst-case score")
        ax.set_title(f"worst of the four task scores: {lab}")
        ax.set_xlabel("x_sep (mm)")
        ax.set_ylabel("y_sep (mm)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=10, frameon=False)
    fig.colorbar(im, ax=axs, shrink=0.8, label="worst of the four scores")
    FR.save(fig, "20261030-cross_grid.png")
    print(out, flush=True)
    return 0


def cmd_cross(a):
    """The four tasks on the Sobol layouts at the default objects (DEFAULT): rank correlation between tasks, the
    Pareto front over the four scores, each task's best layout scored on the others, and the layout with the best worst
    score. Writes rows/sizes_v4b/cross.json and media/20261030-cross_tasks.png."""
    d, z = layouts()
    n = len(d)
    S = {}
    for task, size in DEFAULT.items():
        sc = scores(task, size, n)
        R = reach(task, size)
        if not R or sc["run"].sum() < len(R):
            print(f"{task} {size:g} mm not complete", flush=True)
            return 1
        S[task] = sc["mean"]
    tasks = list(S)
    F = np.stack([S[t] for t in tasks], 1)
    rho = {f"{a_}-{b_}": round(FR.spearman(S[a_], S[b_]), 3) for i, a_ in enumerate(tasks) for b_ in tasks[i + 1:]}
    P = pareto(F)
    best = {t: int(np.argmax(S[t])) for t in tasks}
    mm = int(np.argmax(F.min(1)))
    out = {"sizes": DEFAULT, "spearman": rho, "pareto_n": len(P), "pareto": [int(i) for i in P],
           "best": {t: {"call": i, "d": d[i].round(1).tolist(), "scores": F[i].round(3).tolist()} for t, i in best.items()},
           "maximin": {"call": mm, "d": d[mm].round(1).tolist(), "scores": F[mm].round(3).tolist()},
           "best_rank_elsewhere": {t: {u: int((S[u] > S[u][i]).sum()) for u in tasks if u != t} for t, i in best.items()}}
    (SIZE_ROWS / "cross.json").write_text(json.dumps(out, indent=1))
    plt = FR.plt_setup()
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)
    for ax, (u, v_) in zip(axs[:2], (("twist", "roll"), ("pinpush", "press"))):
        ax.scatter(S[u], S[v_], s=3, color="0.6", alpha=0.4, lw=0, label="Sobol layouts")
        ax.scatter(S[u][P], S[v_][P], s=14, color="#d62728", lw=0, label="Pareto set over the four tasks")
        ax.set_xlabel(f"{TASK_NAME[u]} score ({DEFAULT[u]:g} mm)")
        ax.set_ylabel(f"{TASK_NAME[v_]} score ({DEFAULT[v_]:g} mm)")
        ax.set_title(f"Spearman {FR.spearman(S[u], S[v_]):.2f}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=10, frameon=False)
    ax = axs[2]
    for k, t in enumerate(tasks):
        top = np.argsort(-S[t])[:int(TOP * n)]
        ax.errorbar(np.arange(6) + (k - 1.5) * 0.12, np.median(z[top], 0),
                    yerr=[np.median(z[top], 0) - np.percentile(z[top], 25, 0),
                          np.percentile(z[top], 75, 0) - np.median(z[top], 0)], fmt="o-", capsize=3, lw=1.4,
                    label=TASK_NAME[t])
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(6), COORD_LABELS)
    ax.set_ylim(-0.55, 0.55)
    ax.set_ylabel("standardized coordinate")
    ax.set_title("the best tenth of layouts per task")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=10, frameon=False)
    FR.save(fig, "20261030-cross_tasks.png")
    print({k: v for k, v in out.items() if k != "pareto"}, flush=True)
    return 0


def fig_sizes(task, out=None):
    """The size study of one task over the Sobol layouts: (a) rank correlation of the layouts' scores between object
    sizes; (b) where the best tenth of layouts sits at each size, as the median of each of the six mount coordinates
    (standardized; 0 is the centre of a gantry's travel) with its quartiles; (c) the scores of the reached layouts per
    size."""
    plt = FR.plt_setup()
    d, z = layouts()
    n = len(d)
    S, sz = [], []
    for size in SIZES[task]:
        sc = scores(task, size, n)
        if sc["run"].any():
            S.append((sc, sorted(reach(task, size))))
            sz.append(size)
    if len(sz) < 2:
        return
    fig, axs = plt.subplots(1, 3, figsize=(17.0, 5.0), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.6, 1]})
    m = len(sz)
    C = np.ones((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                C[i, j] = FR.spearman(S[i][0]["mean"], S[j][0]["mean"])
    ax = axs[0]
    im = ax.imshow(C, cmap="viridis", vmin=0, vmax=1)
    for i in range(m):
        for j in range(m):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", color="w" if C[i, j] < 0.6 else "k", fontsize=12)
    ax.set_xticks(range(m), [f"{x:g}" for x in sz])
    ax.set_yticks(range(m), [f"{x:g}" for x in sz])
    ax.set_xlabel("object diameter (mm)")
    ax.set_ylabel("object diameter (mm)")
    ax.set_title("(a) Spearman of layout scores")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axs[1]
    cols = FR.plt_setup().cm.plasma(np.linspace(0.05, 0.85, m))
    for k, ((sc, R), size) in enumerate(zip(S, sz)):
        top = np.argsort(-sc["mean"])[:max(1, int(TOP * n))]
        med = np.median(z[top], 0)
        q1, q3 = np.percentile(z[top], [25, 75], 0)
        x = np.arange(6) + (k - (m - 1) / 2) * 0.12
        ax.errorbar(x, med, yerr=[med - q1, q3 - med], fmt="o-", color=cols[k], capsize=3, lw=1.4, label=f"{size:g} mm")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(6), COORD_LABELS)
    ax.set_ylim(-0.55, 0.55)
    ax.set_ylabel("standardized coordinate")
    ax.set_title("(b) the best tenth of layouts: median and quartiles")
    ax.legend(loc="upper right", ncol=2)
    ax = axs[2]
    ax.boxplot([sc["mean"][R] for sc, R in S], showfliers=False)
    ax.set_xticks(range(1, m + 1), [f"{x:g}" for x in sz])
    ax.set_xlabel("object diameter (mm)")
    ax.set_ylabel("score of reached layouts")
    ax.set_title("(c) scores per size")
    fig.suptitle(f"{TASK_NAME[task]}: 8,192 Sobol layouts across object sizes, foundational poses, seeds 0-2")
    FR.save(fig, out or f"20261030-sizes_{task}.png")


def cmd_figures(a):
    for task in TASKS:
        for size in SIZES[task]:
            if (GRID_ROWS / f"{task}_d{size:g}.jsonl").exists():
                fig_grid(task, size)
            if (ROWS / "peak" / f"summary_{task}{size:g}.json").exists():
                fig_value(task, size)
        fig_sizes(task)
        pk_ = fig_grid_sizes(task)
        if pk_:
            fj = GRID_ROWS / "peaks_by_size.json"
            allp = json.loads(fj.read_text()) if fj.exists() else {}
            allp[task] = pk_
            fj.write_text(json.dumps(allp, indent=1))
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
    sub.add_parser("cross")
    sub.add_parser("valrep")
    sub.add_parser("cross-grid")
    sub.add_parser("ea")
    sub.add_parser("draws")
    for name in ("moves", "moves-report", "local-report"):
        p = sub.add_parser(name)
        p.add_argument("--task", choices=TASKS, required=True)
        p.add_argument("--d", type=float, required=True)
        if name == "moves":
            p.add_argument("--kind", choices=tuple(MOVE_ROWS), required=True)
    p = sub.add_parser("views")
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--d", type=float, required=True)
    p.add_argument("--calls", type=int, nargs="+", required=True, help="grid indices (x_sep major, y_sep minor)")
    p.add_argument("--out", default="")
    a = ap.parse_args()
    return {"stats": cmd_stats, "grid": cmd_grid, "figures": cmd_figures, "derive": cmd_derive, "views": cmd_views,
            "value": cmd_value, "moves": cmd_moves, "moves-report": cmd_moves_report, "local-report": cmd_local_report, "ea": cmd_ea, "draws": cmd_draws,
            "cross": cmd_cross,
            "valrep": cmd_valrep, "cross-grid": cmd_cross_grid}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
