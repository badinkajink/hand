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


# ------------------------------------------------------------------------------------------------ the 2-D grid with grasp sets

GRID_ROWS = ROWS / "grid2d"
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


def smooth_map(Z):
    """3 x 3 mean over the finite cells."""
    P = np.pad(Z, 1, constant_values=np.nan)
    st = np.stack([P[1 + di:P.shape[0] - 1 + di, 1 + dj:P.shape[1] - 1 + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)])
    with np.errstate(all="ignore"):
        out = np.nanmean(st, 0)
    return np.where(np.isfinite(Z), out, np.nan)


def fig_grid(task, size, out=None):
    """Six maps over the grid for one task and object: the foundational pose's score, the best grasp's, their gain, the
    share of grasps held in every seed, the feasible log volume of the grasp set, and the best-grasp score smoothed over
    3 x 3 cells with its gradient and the peaks of its basins."""
    plt = FR.plt_setup()
    xs, ys, M = grid_maps(task, size)
    ext = [xs[0] - 2.5, xs[-1] + 2.5, ys[0] - 2.5, ys[-1] + 2.5]
    sm = smooth_map(M["best"])
    gy, gx = np.gradient(sm, ys, xs)
    peak = grid_ascent(sm)
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 9.6), constrained_layout=True)
    panels = (("fp", "foundational pose: score", "viridis", (0, None)), ("best", "best of the grasp set: score", "viridis",
              (0, None)), ("gain", "best minus foundational pose", "magma", (0, None)),
              ("held_share", "share of grasps held in every seed", "cividis", (0, 1)),
              ("logvol", "feasible log volume of the grasp set", "cividis", (None, None)), ("smooth", "best, smoothed;"
              " gradient and basin peaks", "viridis", (0, None)))
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
            pk = np.unique(peak[peak >= 0])
            big = [p for p in pk if (peak == p).sum() >= 8]
            ax.plot(X.flat[big], Y.flat[big], "r*", ms=14, mec="k")
        ax.set_title(title)
        ax.set_xlabel("thumb-to-pair distance x_sep (mm)")
        ax.set_ylabel("index-middle spacing y_sep (mm)")
    obj = f"{'sphere' if T_SHAPE[task] == 'sphere' else 'cylinder'} {size:g} mm"
    fig.suptitle(f"{TASK_NAME[task]}, {obj}: symmetric tripods every 5 mm, 16 grasps each (grey: pose not reached)")
    FR.save(fig, out or f"20261029-grid_{task}_d{size:g}.png")


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
    p = sub.add_parser("views")
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--d", type=float, required=True)
    p.add_argument("--calls", type=int, nargs="+", required=True, help="grid indices (x_sep major, y_sep minor)")
    p.add_argument("--out", default="")
    a = ap.parse_args()
    return {"stats": cmd_stats, "grid": cmd_grid, "figures": cmd_figures, "views": cmd_views}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
