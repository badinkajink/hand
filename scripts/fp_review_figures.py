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
        out[w] = {"landscape": fig_landscape, "efficiency": fig_efficiency, "embed": fig_embed, "sobol": fig_sobol}[w]()
        print(w, json.dumps(out[w])[:600], flush=True)
    p = ROWS / "figures.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(out)
    p.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
