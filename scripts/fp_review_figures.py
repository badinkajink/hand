#!/usr/bin/env python3
"""Figures of the 2026-10-29 foundational-pose review page (scripts/fp_review_page.py), from the rows of
scripts/fp_sr2_toy.py in docs/experiments/20261029-foundational_pose_review/rows.

  landscape  the 2-D case (symmetric tripods, x_sep and y_sep): six score landscapes, Pareto-optimal candidates ringed
             (the layout of Fig. 5 of the foundational-pose paper)
  efficiency candidates per RRT call against calls, 2-D and 6-D, and the share of candidates above a rotation range
  embed      t-SNE and PCA of the 6-D candidates coloured by mean score, with example layouts
  sobol      the six metrics of RRT candidates against Sobol candidates

  .venv/bin/python scripts/fp_review_figures.py [landscape efficiency embed sobol]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
D = ROOT / "docs/experiments/20261029-foundational_pose_review"
ROWS = D / "rows"
MEDIA = D / "media"
FPS = ("pinch", "tripod")
KEYS = (("tmr_deg", +1, "rotation range (deg)"), ("mft_mNm", -1, "largest joint torque (mN m)"),
        ("smin_mm", +1, "smallest pad singular value (mm/rad)"))
FP_NAME = {"pinch": "pinch (pin-push)", "tripod": "tripod (roll, twist)"}
C_THUMB, C_INDEX, C_MIDDLE = "#1f4e9c", "#b03a2e", "#2e7d32"


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                         "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "mathtext.fontset": "dejavuserif"})
    return plt


def save(fig, name):
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / name, dpi=150, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("wrote", (MEDIA / name).relative_to(ROOT), flush=True)


def load(name):
    p = ROWS / f"{name}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def table(rows):
    """Candidates, raw metrics (n, 6), scores 0-100 (n, 6), mean score, Pareto mask."""
    cand = [r for r in rows if r["ok"]]
    raw = np.array([[r["fp"][fp][k] for fp in FPS for k, _, _ in KEYS] for r in cand], float)
    sgn = np.array([s for _ in FPS for _, s, _ in KEYS], float)
    lo, hi = raw.min(0), raw.max(0)
    span = np.maximum(hi - lo, 1e-12)
    sc = np.where(sgn > 0, (raw - lo) / span, (hi - raw) / span) * 100.0
    keep = np.ones(len(sc), bool)
    for i in range(len(sc)):
        dom = np.all(sc >= sc[i], axis=1) & np.any(sc > sc[i], axis=1)
        keep[i] = not dom.any()
    return cand, raw, sc, sc.mean(1), keep


def fig_landscape():
    plt = plt_setup()
    cand, raw, sc, mean, par = table(load("rrt2d"))
    d = np.array([r["d"] for r in cand])
    xs, ys = d[:, 2] - d[:, 0], d[:, 3] - d[:, 5]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.6), sharex=True, sharey=True)
    for k, ax in enumerate(axs.flat):
        fp, (key, s, lab) = FPS[k // 3], KEYS[k % 3]
        im = ax.scatter(xs, ys, c=raw[:, k], cmap="RdYlBu_r" if s > 0 else "RdYlBu", s=16, lw=0)
        ax.scatter(xs[par], ys[par], s=42, facecolors="none", edgecolors="k", lw=1.0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=11)
        ax.set_title(f"{FP_NAME[fp]}:\n{lab}", fontsize=12.5)
        ax.set_aspect("equal")
    for ax in axs[1]:
        ax.set_xlabel("thumb to pair, x_sep (mm)")
    for ax in axs[:, 0]:
        ax.set_ylabel("index to middle, y_sep (mm)")
    fig.tight_layout()
    save(fig, "20261029-landscapes_2d.png")
    return {"n": len(cand), "pareto": int(par.sum()), "x_range": [float(xs.min()), float(xs.max())],
            "y_range": [float(ys.min()), float(ys.max())]}


def moving(x, w):
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        a = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[a]) / (i + 1 - a)
    return out


def fig_efficiency():
    plt = plt_setup()
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.0))
    info = {}
    for name, col, lab in (("rrt2d", C_THUMB, "2-D (symmetric tripods)"), ("rrt6d", C_INDEX, "6-D (all mounts)")):
        rows = load(name)
        summ = json.loads((ROWS / f"{name}_summary.json").read_text()) if (ROWS / f"{name}_summary.json").exists() else {}
        rr = [r for r in rows if r.get("kind") == "rrt"]
        calls = np.array([r["call"] for r in rr])
        okc = np.array([r["ok"] for r in rr], float)
        # efficiency per RRT call: calls rejected for being too close to a candidate count as calls without one
        allc = np.zeros(int(summ.get("calls", calls.max() if len(calls) else 0)) + 1)
        allc[calls[okc > 0].astype(int)] = 1.0
        eff = moving(allc[1:], 400)
        axs[0].plot(np.arange(1, len(eff) + 1), eff, color=col, lw=2, label=lab)
        cand, raw, sc, mean, par = table(rows)
        for k, fp in enumerate(FPS):
            tmr = raw[:, 3 * k]
            th = np.arange(0, 61, 1)
            frac = [(tmr > t).mean() * 100 for t in th]
            axs[1].plot(th, frac, color=col, lw=2, ls="-" if fp == "pinch" else "--",
                        label=f"{lab}, {FP_NAME[fp]}")
        info[name] = {"calls": int(summ.get("calls", 0)), "candidates": len(cand), "solver_calls": len(rows),
                      "eff_last": summ.get("efficiency_last"), "seconds": summ.get("seconds"),
                      "pareto": int(par.sum()), "tmr_pos": {fp: float((raw[:, 3 * k] > 0).mean())
                                                             for k, fp in enumerate(FPS)}}
    axs[0].set_xscale("log")
    axs[0].set_xlabel("RRT calls")
    axs[0].set_ylabel("candidates per call, last 400 calls")
    axs[0].legend(frameon=False)
    axs[1].set_xlabel("rotation range threshold (deg)")
    axs[1].set_ylabel("candidates above it (%)")
    axs[1].legend(frameon=False, fontsize=11)
    fig.tight_layout()
    save(fig, "20261029-sampling_efficiency.png")
    return info


def draw_layout(ax, d, col="k", label=None):
    th, ix, md = (d[0], d[1]), (d[2], d[3]), (d[4], d[5])
    pts = np.array([th, ix, md, th])
    ax.plot(pts[:, 0], pts[:, 1], color=col, lw=1.2)
    for (x, y), c in zip((th, ix, md), (C_THUMB, C_INDEX, C_MIDDLE)):
        ax.plot(x, y, "o", ms=7, color=c)
    if label:
        ax.set_title(label, fontsize=12)
    ax.set_xlim(-85, 85)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def fig_embed():
    plt = plt_setup()
    emb = json.loads((ROWS / "embed6d.json").read_text())
    cand, raw, sc, mean, par = table(load("rrt6d"))
    Y = np.array(emb["tsne_p30"])
    P = np.array(emb["pca"])
    d = np.array([r["d"] for r in cand])
    fig = plt.figure(figsize=(16.5, 9.6))
    W, Hh = 0.225, 0.52
    axs = [fig.add_axes([0.01 + k * 0.237, 0.38, W, Hh]) for k in range(4)]
    cax = fig.add_axes([0.962, 0.40, 0.010, 0.46])
    a, a2, b, c = axs
    im = a.scatter(Y[:, 0], Y[:, 1], c=mean, cmap="RdYlBu_r", s=5, lw=0, vmin=10, vmax=60)
    a.set_title(f"(a) t-SNE, all {len(Y)} RRT candidates", fontsize=12)
    if "rrt_sub_tsne_p30" in emb:
        Ysub = np.array(emb["rrt_sub_tsne_p30"])
        a2.scatter(Ysub[:, 0], Ysub[:, 1], c=mean[np.array(emb["rrt_sub_idx"])], cmap="RdYlBu_r", s=8, lw=0,
                   vmin=10, vmax=60)
        a2.set_title(f"(b) t-SNE, {len(Ysub)} RRT candidates at random", fontsize=12)
    if "sobol_tsne_p30" in emb:
        cs, raws, scs, means, pars = table(load("sobol6d"))
        Ys = np.array(emb["sobol_tsne_p30"])
        # Sobol candidates scored on the RRT candidates' scale, so the colours compare
        sgn = np.array([s_ for _ in FPS for _, s_, _ in KEYS], float)
        lo, hi = raw.min(0), raw.max(0)
        span = np.maximum(hi - lo, 1e-12)
        scs = np.clip(np.where(sgn > 0, (raws - lo) / span, (hi - raws) / span) * 100.0, 0, 100)
        b.scatter(Ys[:, 0], Ys[:, 1], c=scs.mean(1), cmap="RdYlBu_r", s=8, lw=0, vmin=10, vmax=60)
        b.set_title(f"(c) t-SNE, {len(Ys)} Sobol candidates", fontsize=12)
    c.scatter(P[:, 0], P[:, 1], c=mean, cmap="RdYlBu_r", s=6, lw=0, vmin=10, vmax=60)
    v = emb["pca_var"]
    c.set_title(f"(d) PCA of (a): {100 * v[0]:.0f} and {100 * v[1]:.0f} % of variance", fontsize=12)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("mean score (0-100)")
    from scipy.cluster.vq import kmeans2
    _, lab = kmeans2(Y, 6, minit="++", seed=1)
    picks = []
    for k in range(6):
        idx = np.flatnonzero(lab == k)
        if idx.size:
            picks.append(int(idx[np.argmax(mean[idx])]))
    picks = sorted(picks, key=lambda i: -mean[i])
    for j, i in enumerate(picks):
        a.annotate(str(j + 1), Y[i], fontsize=13, weight="bold", xytext=(4, 4), textcoords="offset points")
        a.plot(*Y[i], "o", ms=9, mfc="none", mec="k", mew=1.5)
        ax = fig.add_axes([0.01 + j * 0.158, 0.02, 0.14, 0.27])
        draw_layout(ax, d[i], label=f"{j + 1}: mean score {mean[i]:.0f}")
    save(fig, "20261029-tsne_pca_6d.png")
    return {"n": len(cand), "sobol_n": emb.get("sobol_n"),
            "picks": [{"d": d[i].round(1).tolist(), "mean": round(float(mean[i]), 1),
                       "scores": sc[i].round(0).tolist(), "raw": raw[i].round(2).tolist()} for i in picks],
            "pca_var": v}


def fig_sobol():
    plt = plt_setup()
    rr, ss = load("rrt6d"), load("sobol6d")
    c_r = [r for r in rr if r["ok"]]
    c_s = [r for r in ss if r["ok"]]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4))
    for k, ax in enumerate(axs.flat):
        fp, (key, s, lab) = FPS[k // 3], KEYS[k % 3]
        a = np.array([r["fp"][fp][key] for r in c_r])
        b = np.array([r["fp"][fp][key] for r in c_s])
        bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 30)
        ax.hist(a, bins, density=True, alpha=0.55, color=C_INDEX, label=f"RRT, {len(a)}")
        ax.hist(b, bins, density=True, alpha=0.55, color=C_THUMB, label=f"Sobol, {len(b)}")
        ax.set_title(f"{FP_NAME[fp]}: {lab}", fontsize=12)
        ax.set_yticks([])
    axs[0, 0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "20261029-rrt_vs_sobol.png")
    return {"rrt_ok": len(c_r), "sobol_ok": len(c_s), "sobol_n": len(ss)}


def main():
    which = sys.argv[1:] or ["landscape", "efficiency", "embed", "sobol"]
    out = {}
    for w in which:
        out[w] = {"landscape": fig_landscape, "efficiency": fig_efficiency, "embed": fig_embed, "sobol": fig_sobol,
                  "dyn2d": fig_dyn2d, "dyn6d": fig_dyn6d, "four2d": fig_four2d, "four6d": fig_four6d,
                  "sense6d": fig_sense6d}[w]()
        print(w, json.dumps(out[w])[:600], flush=True)
    p = ROWS / "figures.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(out)
    p.write_text(json.dumps(old, indent=1))



# ------------------------------------------------------------------------------------------------ dynamic (HOM) scores
def lay_key(r):
    """A layout's key in the GPU rows: the six coordinates, or (x_sep, y_sep) for the 2-D equator-grasp rows."""
    return tuple(r["d"]) if "d" in r else (r["x_sep_mm"], r["y_sep_mm"])


def sep2(k):
    """(x_sep, y_sep) of a symmetric layout's six coordinates."""
    return (round(k[2] - k[0], 1), round(k[3] - k[5], 1)) if len(k) == 6 else k


def load_hom(name, variant=None):
    """GPU rows of scripts/hom_gpu.py, per layout (and the given variant): held fraction and medians over seeds. Sphere
    tasks: ramp (deg) and rmse (velocity RMSE per step); cylinder tasks: theta (pinch-angle RMSE, deg) and gain (the
    smallest stroke gain); medians over the held runs."""
    import hom_rows
    rows = [hom_rows.rescale_metrics(r) for r in load(name) if r.get("status", "ok") == "ok"
            and (variant is None or r.get("variant", "base") == variant)]
    by = {}
    for r in rows:
        by.setdefault(lay_key(r), []).append(r)
    out = []
    for k, rr in by.items():
        held = [r for r in rr if r.get("held")]

        def med(m, src):
            v = [r.get(m) for r in src if r.get(m) is not None]
            return float(np.median(v)) if v else float("nan")
        out.append({"key": k, "kin": rr[0].get("kin"), "kin_pinch": rr[0].get("kin_pinch"),
                    "kin_press": rr[0].get("kin_press"), "n": len(rr), "held": len(held) / len(rr),
                    "ramp": float(np.median([r.get("ramp_three_pads_max_deg") or 0.0 for r in rr])),
                    "rmse": med("w_rmse_ratio", held), "theta": med("theta_rmse_deg", held), "gain": med("gain_min", held)})
    return out


TASK_SCORE = {"twist": ("ramp", +1, "twist: ramp on three pads (deg)"),
              "roll": ("rmse", -1, "roll: velocity RMSE per step"),
              "pinpush": ("theta", -1, "pin-push: pinch-angle RMSE (deg)"),
              "press": ("theta", -1, "press: pinch-angle RMSE (deg)")}


def task_table(names, variant=None):
    """{task: {layout key: per-layout dict}} for names = {task: rows name}."""
    return {t: {x["key"]: x for x in load_hom(n, variant)} for t, n in names.items()}


def fig_four2d():
    """The 2-D case from the toy's foundational poses, four tasks: landscapes over (x_sep, y_sep), and the twist and
    roll from the foundational pose against the task study's equator grasp on the same layouts."""
    plt = plt_setup()
    T4 = task_table({t: f"hom2dfp_gpu_{t}" for t in ("twist", "roll", "pinpush", "press")})
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 12.0), sharex=True, sharey=True)
    info = {}
    for ax, t in zip(axs.flat, ("twist", "roll", "pinpush", "press")):
        m, sgn, title = TASK_SCORE[t]
        keys = sorted(T4[t])
        xy = np.array([sep2(k) for k in keys])
        v = np.array([T4[t][k][m] for k in keys])
        held = np.array([T4[t][k]["held"] >= 1.0 for k in keys])
        drop2 = np.array([T4[t][k]["held"] < 0.5 for k in keys])           # dropped in two seeds or more
        ok = np.isfinite(v) & ~drop2
        lo, hi = np.nanpercentile(v[ok], [5, 95])
        cm = "RdYlBu_r" if sgn > 0 else "RdYlBu"
        im = ax.scatter(xy[ok, 0], xy[ok, 1], c=v[ok], cmap=cm, s=30, lw=0, vmin=lo, vmax=hi)
        ax.scatter(xy[drop2, 0], xy[drop2, 1], s=40, marker="x", color="k", lw=1.0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(title, fontsize=12.5)
        ax.set_aspect("equal")
        near = xy[:, 0] <= 60.0                                             # the thumb within 60 mm of the pair
        info[t] = {"n": len(keys), "held_all": float(held.mean()), "dropped2": float(drop2.mean()),
                   "median": float(np.nanmedian(v[np.isfinite(v)])), "n_near": int(near.sum()),
                   "held_near": float(held[near].mean()), "held_far": float(held[~near].mean())}
    for ax in axs[1]:
        ax.set_xlabel("thumb to pair, x_sep (mm)")
    for ax in axs[:, 0]:
        ax.set_ylabel("index to middle, y_sep (mm)")
    fig.tight_layout()
    save(fig, "20261029-hom_four_tasks_2d.png")
    # matched grasps: the equator grasp (hom2d_gpu_*) against the foundational pose on the same layouts
    eq = task_table({"twist": "hom2d_gpu_twist", "roll": "hom2d_gpu_roll"})
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, t in zip(axs, ("twist", "roll")):
        m, sgn, title = TASK_SCORE[t]
        fpk = {sep2(k): x for k, x in T4[t].items()}
        common = sorted(set(eq[t]) & set(fpk))
        a = np.array([eq[t][k][m] for k in common])
        b = np.array([fpk[k][m] for k in common])
        ha = np.array([eq[t][k]["held"] >= 1.0 for k in common])
        hb = np.array([fpk[k]["held"] >= 1.0 for k in common])
        both = ha & hb & np.isfinite(a) & np.isfinite(b)
        ax.scatter(a[~both], b[~both], s=14, color="#c8c8c8", lw=0, label="dropped in a seed from either grasp")
        ax.scatter(a[both], b[both], s=14, color=C_THUMB, lw=0, alpha=0.75, label="held in every seed from both")
        lim = np.nanpercentile(np.r_[a, b], [0, 99]) if sgn > 0 else np.nanpercentile(np.r_[a, b], [0, 98])
        lim = [min(lim[0], float(np.nanmin(np.r_[a, b]))) * (0.95 if sgn < 0 else 1.0), lim[1] * 1.05]
        ax.plot(lim, lim, color="k", lw=1.0, ls=":")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel("from the equator grasp")
        ax.set_ylabel("from the foundational pose")
        ax.set_title(title, fontsize=12.5)
        better = (b > a) if sgn > 0 else (b < a)
        info[f"matched_{t}"] = {"n": len(common), "eq_median": float(np.nanmedian(a)), "fp_median": float(np.nanmedian(b)),
                                "fp_better": float(np.mean(better[np.isfinite(a) & np.isfinite(b)])),
                                "held_eq": float(ha.mean()), "held_fp": float(hb.mean())}
    axs[0].legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=10.5, loc="lower right", markerscale=2)
    fig.tight_layout()
    save(fig, "20261029-hom_matched_grasps_2d.png")
    return info


def rank01(v):
    """Percentile ranks in [0, 1] (ties averaged), NaN kept."""
    v = np.asarray(v, float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    order = np.argsort(np.argsort(v[ok]))
    out[ok] = order / max(ok.sum() - 1, 1)
    return out


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def pareto_mask(sc):
    """Non-dominated rows of a score matrix, every column maximised."""
    n = len(sc)
    keep = np.ones(n, bool)
    for i in range(n):
        dom = np.all(sc >= sc[i], axis=1) & np.any(sc > sc[i], axis=1)
        keep[i] = not dom.any()
    return keep


def fig_four6d(prefix="hom6d_gpu", pop="sobol8k"):
    """The Sobol layouts under the four tasks from their foundational poses: rank correlations between the kinematic
    and dynamic scores, the four-task Pareto front, and the t-SNE map coloured by the mean rank over the tasks."""
    plt = plt_setup()
    T4 = task_table({t: f"{prefix}_{t}" for t in ("twist", "roll", "pinpush", "press")})
    keys = sorted(set.intersection(*[set(v) for v in T4.values()]))
    held = np.array([all(T4[t][k]["held"] >= 1.0 for t in T4) for k in keys])
    sc = {}
    for t in T4:
        m, sgn, _ = TASK_SCORE[t]
        sc[t] = sgn * np.array([T4[t][k][m] for k in keys])
    kin = {"tripod range": np.array([T4["twist"][k]["kin"]["tmr_deg"] for k in keys]),
           "pinch range": np.array([T4["pinpush"][k]["kin_pinch"]["tmr_deg"] for k in keys]),
           "tripod pad s.v.": np.array([T4["twist"][k]["kin"]["smin_mm"] for k in keys]),
           "pinch pad s.v.": np.array([T4["pinpush"][k]["kin_pinch"]["smin_mm"] for k in keys])}
    labels = list(kin) + ["twist ramp", "- roll RMSE", "- pin-push RMSE", "- press RMSE"]
    cols = list(kin.values()) + [sc["twist"], sc["roll"], sc["pinpush"], sc["press"]]
    M = np.array([[spearman(a[held], b[held]) for b in cols] for a in cols])
    S4 = np.stack([sc[t] for t in ("twist", "roll", "pinpush", "press")], 1)
    good = held & np.all(np.isfinite(S4), 1)
    par = np.zeros(len(keys), bool)
    par[np.flatnonzero(good)] = pareto_mask(S4[good])
    R = np.stack([rank01(np.where(good, S4[:, j], np.nan)) for j in range(4)], 1)
    mean_rank = np.nanmean(R, 1)
    fig = plt.figure(figsize=(19, 6.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0])
    a = fig.add_subplot(gs[0])
    im = a.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    a.set_xticks(range(len(labels)))
    a.set_xticklabels(labels, rotation=55, ha="right", fontsize=11)
    a.set_yticks(range(len(labels)))
    a.set_yticklabels(labels, fontsize=11)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                       color="white" if abs(M[i, j]) > 0.6 else "black")
    a.axhline(3.5, color="k", lw=0.8)
    a.axvline(3.5, color="k", lw=0.8)
    a.set_title("Spearman over layouts held in every run", fontsize=12.5)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    b = fig.add_subplot(gs[1])
    xs = np.arange(4)
    for i in np.flatnonzero(good & ~par)[::7]:
        b.plot(xs, R[i], color="#d0d0d0", lw=0.6, alpha=0.5)
    for i in np.flatnonzero(par):
        b.plot(xs, R[i], color=C_INDEX, lw=1.1, alpha=0.8)
    b.set_xticks(xs)
    b.set_xticklabels(["twist\nramp", "roll\nRMSE", "pin-push\nRMSE", "press\nRMSE"])
    b.set_ylabel("percentile among held layouts (1 best)")
    b.set_title(f"four-task Pareto front: {int(par.sum())} of {int(good.sum())} layouts", fontsize=12.5)
    c = fig.add_subplot(gs[2])
    info = {}
    emb_p = ROWS / f"embed_{pop}.json"
    if emb_p.exists():
        emb = json.loads(emb_p.read_text())
        pos = {tuple(np.round(d, 3)): xy for d, xy in zip(emb["d"], emb["tsne"])}
        P = np.array([pos.get(tuple(np.round(k, 3)), (np.nan, np.nan)) for k in keys], float)
        okp = np.isfinite(P[:, 0])
        c.scatter(P[okp & ~good, 0], P[okp & ~good, 1], s=4, color="#d6d6d6", lw=0)
        im = c.scatter(P[okp & good, 0], P[okp & good, 1], c=mean_rank[okp & good], cmap="RdYlBu_r", s=22, lw=0.3,
                       edgecolor="#555555")
        c.scatter(P[okp & par, 0], P[okp & par, 1], s=70, facecolor="none", edgecolor="k", lw=1.1)
        fig.colorbar(im, ax=c, fraction=0.046, pad=0.02, label="mean percentile over the four tasks")
        c.set_title(f"t-SNE of {int(okp.sum())} Sobol layouts: {int(good.sum())} held in all four tasks "
                    f"coloured, front ringed", fontsize=11.5)
        c.set_xticks([])
        c.set_yticks([])
        info["tsne_n"] = int(okp.sum())
    fig.tight_layout()
    save(fig, "20261029-hom_four_tasks_6d.png")
    # are the layouts that hold all four tasks clustered in the standardized six coordinates?
    D = np.array(keys, float)
    lo6 = np.array([-80.0, -55.0, 20.0, 25.0, 20.0, -85.0])
    hi6 = np.array([-20.0, 55.0, 80.0, 85.0, 80.0, -25.0])
    Z = (D - lo6) / (hi6 - lo6)

    def nn(Zs):
        dd = np.linalg.norm(Zs[:, None] - Zs[None], axis=-1)
        np.fill_diagonal(dd, np.inf)
        return float(dd.min(1).mean())
    rng = np.random.default_rng(0)
    rand = [nn(Z[rng.choice(len(Z), int(good.sum()), replace=False)]) for _ in range(50)]
    ti = np.hypot(D[:, 2] - D[:, 0], D[:, 3] - D[:, 1])
    pp_held = np.array([T4["pinpush"][k]["held"] >= 1.0 for k in keys])
    info["cluster"] = {"nn_good": nn(Z[good]), "nn_rand": float(np.mean(rand)), "nn_rand_sd": float(np.std(rand)),
                       "med_all": {c_: float(np.median(D[:, j])) for j, c_ in enumerate(("tx", "ty", "ix", "iy", "mx", "my"))},
                       "med_good": {c_: float(np.median(D[good, j])) for j, c_ in enumerate(("tx", "ty", "ix", "iy", "mx", "my"))},
                       "pinpush_held_ti60": float(pp_held[ti <= 60].mean()), "n_ti60": int((ti <= 60).sum()),
                       "pinpush_held_ti80": float(pp_held[ti > 80].mean())}
    info.update({"n": len(keys), "held_all4": int(held.sum()), "good": int(good.sum()), "pareto": int(par.sum()),
                 "spearman": {f"{labels[i]} | {labels[j]}": round(float(M[i, j]), 3)
                              for i in range(len(labels)) for j in range(i + 1, len(labels))},
                 "held_each": {t: float(np.mean([T4[t][k]["held"] >= 1.0 for k in keys])) for t in T4},
                 "median_each": {t: float(np.nanmedian([T4[t][k][TASK_SCORE[t][0]] for k in keys])) for t in T4},
                 "pareto_layouts": [list(np.round(keys[i], 1)) for i in np.flatnonzero(par)][:20]})
    return info


SENSE_VARS = ("rate50", "cal_notip", "cal_tip", "cal_tipfl")


def fig_sense6d(prefix="hom6d_gpu_sense"):
    """The Sobol layouts under the sensing variants at 50 Hz: each layout's score with exact sensing against its score
    with calibration errors and fingertips, without and with the force loop; and the pinch tasks' force error per
    layout."""
    plt = plt_setup()
    tasks = [t for t in ("twist", "roll", "pinpush", "press") if (ROWS / f"{prefix}_{t}.jsonl").exists()]
    if not tasks:
        return {}
    V = {t: {v: {x["key"]: x for x in load_hom(f"{prefix}_{t}", v)} for v in SENSE_VARS} for t in tasks}
    fig, axs = plt.subplots(2, len(tasks), figsize=(5.0 * len(tasks), 9.6), squeeze=False)
    info = {}
    for j, t in enumerate(tasks):
        m, sgn, title = TASK_SCORE[t]
        ref = V[t]["rate50"]
        info[t] = {}
        # the seed-noise baseline: exact sensing with seeds 3-5 (rows <prefix>_<task>_s345)
        new = {x["key"]: x for x in load_hom(f"{prefix}_{t}_s345", "rate50")}
        if new:
            keys = sorted(set(ref) & set(new))
            a = np.array([ref[k][m] for k in keys])
            b = np.array([new[k][m] for k in keys])
            both = (np.array([ref[k]["held"] >= 1.0 and new[k]["held"] >= 1.0 for k in keys]) & np.isfinite(a)
                    & np.isfinite(b))
            top = both & (rank01(np.where(both, sgn * a, np.nan)) >= 0.9)
            info[t]["seeds"] = {"n": len(keys), "spearman": spearman(a[both], b[both]),
                                "held_same": float(np.mean([(ref[k]["held"] >= 1.0) == (new[k]["held"] >= 1.0)
                                                            for k in keys])),
                                "top10_kept": float(np.mean(rank01(np.where(both, sgn * b, np.nan))[top] >= 0.9))
                                if top.any() else float("nan")}
        for i, var in enumerate(("cal_tip", "cal_tipfl")):
            ax = axs[i, j]
            oth = V[t][var]
            keys = sorted(set(ref) & set(oth))
            a = np.array([ref[k][m] for k in keys])
            b = np.array([oth[k][m] for k in keys])
            ha = np.array([ref[k]["held"] >= 1.0 for k in keys])
            hb = np.array([oth[k]["held"] >= 1.0 for k in keys])
            both = ha & hb & np.isfinite(a) & np.isfinite(b)
            ax.scatter(a[both], b[both], s=4, color=C_THUMB if var == "cal_tip" else C_INDEX, lw=0, alpha=0.5)
            lo, hi = np.nanpercentile(np.r_[a[both], b[both]], [1, 99])
            ax.plot([lo, hi], [lo, hi], color="k", lw=1.0, ls=":")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            rho = spearman(a[both], b[both])
            name = "calibration errors, fingertips" if var == "cal_tip" else "the same with the force loop"
            ax.set_title(f"{title.split(':')[0]}: {name}\nSpearman {rho:.2f}; held {100 * hb.mean():.0f} % "
                         f"(exact {100 * ha.mean():.0f} %)", fontsize=11.5)
            ax.set_xlabel(f"exact sensing: {title.split(': ')[1]}", fontsize=11.5)
            ax.set_ylabel(name.split(",")[0] if var == "cal_tip" else "with the force loop", fontsize=11.5)
            top = both & (rank01(np.where(both, sgn * a, np.nan)) >= 0.9)
            keep_top = float(np.mean(rank01(np.where(both, sgn * b, np.nan))[top] >= 0.9)) if top.any() else float("nan")
            info[t][var] = {"n": len(keys), "spearman": rho, "held_exact": float(ha.mean()), "held": float(hb.mean()),
                            "median_exact": float(np.nanmedian(a[both])), "median": float(np.nanmedian(b[both])),
                            "top10_kept": keep_top}
    fig.tight_layout()
    save(fig, "20261029-hom_sensing_6d.png")
    # the pinch tasks' force error per layout (mean over pads of |mean pad force - the same seed's with exact sensing|)
    out = {}
    for t in [t for t in tasks if t in ("pinpush", "press")]:
        rows = [r for r in load(f"{prefix}_{t}") if r.get("status", "ok") == "ok" and r.get("F_mean")]
        ref = {(lay_key(r), r["seed"]): r for r in rows if r["variant"] == "rate50"}
        per = {}
        for r in rows:
            b0 = ref.get((lay_key(r), r["seed"]))
            if b0 is None or r["variant"] == "rate50":
                continue
            dF = float(np.mean([abs(r["F_mean"][f] - b0["F_mean"][f]) for f in b0["F_mean"]]))
            per.setdefault(r["variant"], {}).setdefault(lay_key(r), []).append(dF)
        out[t] = {v: {"layout_median": float(np.median([np.median(x) for x in d.values()])),
                      "layouts_over_1N": float(np.mean([np.median(x) > 1.0 for x in d.values()])),
                      "run_p90": float(np.percentile([y for x in d.values() for y in x], 90))} for v, d in per.items()}
        # the signed change of the total mean pad force against the same seed under exact sensing
        for v in per:
            ch = [sum(r["F_mean"].values()) - sum(ref[(lay_key(r), r["seed"])]["F_mean"].values()) for r in rows
                  if r["variant"] == v and (lay_key(r), r["seed"]) in ref]
            out[t][v].update(total_change_median=float(np.median(ch)), total_change_positive=float(np.mean(np.array(ch) > 0)))
        out[t]["exact_total_median"] = float(np.median([sum(r["F_mean"].values()) for r in rows if r["variant"] == "rate50"]))
    info["force"] = out
    return info


def fig_dyn2d():
    """The 2-D case under HOM on the GPU: the twist's ramp and the roll's error per layout, beside the kinematic range."""
    plt = plt_setup()
    tw = {x["key"]: x for x in load_hom("hom2d_gpu_twist")}
    ro = {x["key"]: x for x in load_hom("hom2d_gpu_roll")}
    keys = sorted(set(tw) & set(ro))
    xs = np.array([k[0] for k in keys])
    ys = np.array([k[1] for k in keys])
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    panels = [(np.array([tw[k]["kin"]["tmr_deg"] for k in keys]), "tripod: kinematic rotation range (deg)", "RdYlBu_r"),
              (np.array([tw[k]["ramp"] for k in keys]), "twist under HOM: ramp on three pads (deg)", "RdYlBu_r"),
              (np.array([ro[k]["rmse"] for k in keys]), "roll under HOM: velocity RMSE per step", "RdYlBu")]
    for ax, (v, title, cm) in zip(axs, panels):
        lo, hi = np.nanpercentile(v, [5, 95])
        im = ax.scatter(xs, ys, c=v, cmap=cm, s=26, lw=0, vmin=lo, vmax=hi)
        drop = np.array([tw[k]["held"] < 0.5 for k in keys])
        ax.scatter(xs[drop], ys[drop], s=60, marker="x", color="k", lw=1.2)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(title, fontsize=12.5)
        ax.set_xlabel("thumb to pair, x_sep (mm)")
        ax.set_aspect("equal")
    axs[0].set_ylabel("index to middle, y_sep (mm)")
    fig.tight_layout()
    save(fig, "20261029-hom_landscape_2d.png")
    kin = np.array([tw[k]["kin"]["tmr_deg"] for k in keys])
    ramp = np.array([tw[k]["ramp"] for k in keys])
    return {"n": len(keys), "dropped_twist": int(drop.sum()),
            "spearman_kin_ramp": float(np.corrcoef(np.argsort(np.argsort(kin)), np.argsort(np.argsort(ramp)))[0, 1])}


def fig_dyn6d(name_tw="hom6d_gpu_twist", name_ro="hom6d_gpu_roll"):
    """Six-coordinate layouts under HOM from their foundational poses: kinematic against dynamic scores, the dynamic
    Pareto front, and the t-SNE map of the Sobol layouts coloured by the twist's ramp."""
    plt = plt_setup()
    tw = {x["key"]: x for x in load_hom(name_tw)}
    ro = {x["key"]: x for x in load_hom(name_ro)}
    keys = sorted(set(tw) & set(ro))
    kin = np.array([tw[k]["kin"]["tmr_deg"] for k in keys])
    smin = np.array([tw[k]["kin"]["smin_mm"] for k in keys])
    ramp = np.array([tw[k]["ramp"] for k in keys])
    rmse = np.array([ro[k]["rmse"] for k in keys])
    held = np.array([min(tw[k]["held"], ro[k]["held"]) for k in keys])
    ok = held >= 1.0
    sc = np.stack([ramp, -np.nan_to_num(rmse, nan=9.0)], 1)
    par = np.ones(len(keys), bool)
    for i in range(len(keys)):
        if not ok[i]:
            par[i] = False
            continue
        dom = ok & np.all(sc >= sc[i], 1) & np.any(sc > sc[i], 1)
        par[i] = not dom.any()
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.4))
    a, b, c = axs
    tw_ok = np.array([tw[k]["held"] >= 1.0 for k in keys])
    a.scatter(kin[~tw_ok], ramp[~tw_ok], s=5, color="#c8c8c8", lw=0, label="twist dropped in a seed")
    a.scatter(kin[tw_ok], ramp[tw_ok], s=5, alpha=0.6, color=C_THUMB, lw=0, label="twist held in every seed")
    a.set_xlabel("tripod: kinematic rotation range (deg)")
    a.set_ylabel("twist under HOM: ramp on three pads (deg)")
    a.legend(frameon=False, fontsize=11, loc="upper right", markerscale=3)
    b.scatter(ramp[ok], rmse[ok], s=6, alpha=0.4, color="#888888", lw=0)
    b.scatter(ramp[par], rmse[par], s=26, color=C_INDEX, lw=0, label=f"Pareto front ({int(par.sum())})")
    b.set_xlabel("twist: ramp on three pads (deg)")
    b.set_ylabel("roll: velocity RMSE per step")
    b.legend(frameon=False, fontsize=11)
    emb_p = ROWS / "embed_sobol8k.json"
    info = {}
    if emb_p.exists():
        emb = json.loads(emb_p.read_text())
        pos = {tuple(np.round(d, 3)): xy for d, xy in zip(emb["d"], emb["tsne"])}
        pts = [(pos.get(tuple(np.round(k, 3))), tw[k]["ramp"], ok[i]) for i, k in enumerate(keys)]
        pts = [p for p in pts if p[0] is not None]
        P = np.array([p[0] for p in pts])
        im = c.scatter(P[:, 0], P[:, 1], c=[p[1] for p in pts], cmap="RdYlBu_r", s=5, lw=0)
        fig.colorbar(im, ax=c, fraction=0.046, pad=0.02, label="twist ramp (deg)")
        c.set_title(f"t-SNE of {len(pts)} Sobol layouts", fontsize=12.5)
        c.set_xticks([])
        c.set_yticks([])
        info["tsne_n"] = len(pts)
    fig.tight_layout()
    save(fig, "20261029-hom_landscape_6d.png")
    rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    info.update({"n": len(keys), "held_all": int(ok.sum()), "pareto": int(par.sum()),
                 "spearman_kin_ramp": float(np.corrcoef(rank(kin[ok]), rank(ramp[ok]))[0, 1]),
                 "spearman_smin_ramp": float(np.corrcoef(rank(smin[ok]), rank(ramp[ok]))[0, 1]),
                 "spearman_smin_rmse": float(np.corrcoef(rank(smin[ok]), rank(rmse[ok]))[0, 1]),
                 "ramp_median": float(np.median(ramp[ok])), "rmse_median": float(np.nanmedian(rmse[ok])),
                 "pareto_layouts": [list(np.round(keys[i], 1)) for i in np.flatnonzero(par)][:12]})
    return info


if __name__ == "__main__":
    main()
