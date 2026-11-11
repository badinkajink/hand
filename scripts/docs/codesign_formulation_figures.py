#!/usr/bin/env python3
"""Data figures of the co-design formulation for the hand-design talk (owner, 2026-11-10: "beautiful, didactic
diagrams / visual content"). The diagrams and formula images are LaTeX (scripts/docs/codesign_tikz.py); these come from
the stored grid of the press on the 25 mm cylinder (SCS0009 plant, 2.5 MPa pads, rows grid2dff_v4b_sv25, the earlier
press score) and from CPU renders of the SR2 hand. Writes docs/experiments/20261110-codesign_formulation/media/:

  20261110-layout_space.png      what a layout is: the palm, the landscape of hands, three hands rendered
  20261110-ea_anatomy.png        one generation of Yang et al.'s EA, step by step, on the landscape
  20261110-ea_snapshots.png      where the EA has looked after 16, 48, 96, 192 layouts, and its curve against random
  20261110-grasp_candidates.png  the 16 candidate grasps of the best hand, rendered, with their scores
  20261110-figures.json          the numbers the page quotes

    HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25 MUJOCO_GL=egl \\
        .venv/bin/python scripts/docs/codesign_formulation_figures.py
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

OUT = ROOT / "docs/experiments/20261110-codesign_formulation/media"
INK, GREY, PALE, LINE = "#1f2937", "#9ca3af", "#f3f4f6", "#d1d5db"
OUTER, INNER, SIM = "#2563eb", "#059669", "#d97706"
PAR, KEPT, KID, STAR = "#c2410c", "#fdba74", "#ea580c", "#facc15"
CMAP, VMAX = "YlGnBu", 0.85
TASK, SIZE = "press", 25.0
BELOW = "90,62,0.21"                 # the films' camera from below: the three fingertips around the object
plt.rcParams.update({"font.family": "Lato", "font.size": 15, "axes.titlesize": 17, "axes.titleweight": "bold",
                     "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.edgecolor": GREY,
                     "axes.titlepad": 10, "mathtext.fontset": "cm"})


VARIANT = ""      # "track" (--track): hom_landscape ranks by the tracking score on the rerun grids; files *_track.*


def use_track():
    """The v2 deck's variant (owner, 2026-11-11): every grid table, EA lookup and film score of this process takes
    hom_landscape.track_score on the grids rerun with the tracking fields, and every output is written as *_track."""
    global VARIANT
    import hom_landscape as L
    VARIANT = "track"
    L.rollout_score = L.track_score


def grid():
    import hom_landscape as L
    L.GRID_SRC, L.GRID_ROWS = "grid2dff", L.ROWS / ("grid2dff_v4b_sv25r" if VARIANT == "track" else "grid2dff_v4b_sv25")
    xs, ys, M = L.grid_maps(TASK, SIZE)
    look = L.ea_landscape(TASK, SIZE)[0]["best"]
    return L, xs, ys, M, look


def landscape(ax, xs, ys, M, label=True):
    ext = [xs[0] - 2.5, xs[-1] + 2.5, ys[0] - 2.5, ys[-1] + 2.5]
    im = ax.imshow(M["best"], origin="lower", extent=ext, cmap=CMAP, vmin=0, vmax=VMAX, aspect="equal")
    if label:
        ax.set_xlabel("thumb to pair (mm)")
        ax.set_ylabel("index to middle (mm)")
    for s in ax.spines.values():
        s.set_color(LINE)
    return im


def cell(xs, ys, M, x, y):
    return float(M["best"][list(ys).index(y), list(xs).index(x)])


def save(fig, name):
    name = name.replace(".png", "_track.png") if VARIANT == "track" else name
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ renders
def still(xy, grasp=None, t_s=1.9, view=None):
    """One frame of a CPU rollout of the press at layout xy (seed 0), at t_s seconds, without the label bars; view
    "azimuth,elevation,distance" for hom_films.Film (elevation > 0 looks up from below)."""
    import os
    from PIL import Image
    import hom_films as HF
    cache = ROOT / "logs/20261110-codesign_stills" / f"{xy[0]:.0f}_{xy[1]:.0f}_{grasp}_{(view or 'side').replace(',', '_')}.png"
    if VARIANT == "track":                                  # the tracking-score grids' grasp indices are the same sets
        cache = cache.with_name("track_" + cache.name)
    if cache.exists() and grasp is not None:
        return np.asarray(Image.open(cache).convert("RGB")), grasp, None
    if view:
        os.environ["HOM_FILM_VIEW"] = view
    else:
        os.environ.pop("HOM_FILM_VIEW", None)
    L, xs, ys, M, _ = grid()
    call = HF.call_at(TASK, SIZE, *xy)
    g = HF.best_grasp(TASK, SIZE, call) if grasp is None else grasp
    fr, res = HF.clip(TASK, "grid2dff_size", SIZE, call, g, "", seed=0, var="hold", render=True)
    k = min(len(fr) - 1, int(round(t_s / (4 * 0.01))))
    cache.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(fr[k][28:-26]).save(cache.with_name(cache.name.replace("_None_", f"_{g}_")))
    return fr[k][28:-26], g, res


def layout_space():
    """A layout is where the three fingers mount; each point of the landscape is one hand; three of them rendered."""
    L, xs, ys, M, _ = grid()
    best = np.unravel_index(np.nanargmax(M["best"]), M["best"].shape)
    A = (float(xs[best[1]]), float(ys[best[0]]))
    B = (70.0, 110.0)
    reached = [(float(x), float(y)) for j, y in enumerate(ys) for i, x in enumerate(xs)
               if np.isfinite(M["best"][j, i]) and 0.05 < M["best"][j, i] < 0.3]
    C = max(reached, key=lambda p: np.hypot(p[0] - A[0], p[1] - A[1]))
    picks = [("A", A), ("B", B), ("C", C)]
    fig = plt.figure(figsize=(24, 6.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.35, 1.05, 1.0, 1.0, 1.0], wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A layout: where the fingers mount")
    ax.add_patch(FancyBboxPatch((-100, -105), 200, 210, boxstyle="round,pad=0,rounding_size=18", fc=PALE, ec=GREY,
                                lw=1.5))
    ax.add_patch(Rectangle((-82.5, -6), 62.5, 12, fc="#e5e7eb", ec="none"))
    ax.add_patch(Rectangle((20, 25), 62.5, 60, fc="#e5e7eb", ec="none"))
    ax.add_patch(Rectangle((20, -85), 62.5, 60, fc="#e5e7eb", ec="none"))
    x, y = A
    pts = {"thumb": (-x / 2, 0.0), "index": (x / 2, y / 2), "middle": (x / 2, -y / 2)}
    for name, (px, py) in pts.items():
        ax.add_patch(plt.Circle((px, py), 9, fc=INNER, ec="white", lw=2, zorder=3))
        ax.text(px + (-13 if name == "thumb" else 13), py, name, ha="right" if name == "thumb" else "left",
                va="center", fontsize=14, color=INK, zorder=4)
    kw = dict(arrowstyle="<|-|>", mutation_scale=16, lw=1.8, color=OUTER)
    ext = dict(color=OUTER, lw=0.9, ls=":")
    ax.plot([-x / 2, -x / 2], [0, -118], **ext)
    ax.plot([x / 2, x / 2], [-y / 2, -118], **ext)
    ax.add_patch(FancyArrowPatch((-x / 2, -118), (x / 2, -118), **kw))
    ax.text(0, -126, f"thumb to pair: {x:.0f} mm", ha="center", va="top", color=OUTER, fontsize=14)
    ax.plot([x / 2, 118], [y / 2, y / 2], **ext)
    ax.plot([x / 2, 118], [-y / 2, -y / 2], **ext)
    ax.add_patch(FancyArrowPatch((118, -y / 2), (118, y / 2), **kw))
    ax.text(126, 0, f"index to\nmiddle:\n{y:.0f} mm", ha="left", va="center", color=OUTER, fontsize=14)
    ax.text(0, 98, "grey: the gantries' travel", ha="center", va="top", fontsize=12, color=GREY)
    ax.set_xlim(-108, 190)
    ax.set_ylim(-150, 112)
    ax.set_aspect("equal")
    ax.axis("off")
    ax = fig.add_subplot(gs[0, 1])
    im = landscape(ax, xs, ys, M)
    ax.set_title("Each point is one hand")
    for lab, (px, py) in picks:
        ax.scatter([px], [py], s=260, facecolors="white", edgecolors=INK, linewidths=2, zorder=4)
        ax.text(px, py, lab, ha="center", va="center", fontsize=12, weight="bold", color=INK, zorder=5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("press score", fontsize=12)
    info = {}
    for j, (lab, xy) in enumerate(picks):
        img, g, res = still(xy, view=BELOW)
        ax = fig.add_subplot(gs[0, 2 + j])
        ax.imshow(img)
        ax.axis("off")
        sc = cell(xs, ys, M, *xy)
        ax.set_title(f"{lab}: ({xy[0]:.0f}, {xy[1]:.0f}) mm, score {sc:.2f}", fontsize=15)
        info[lab] = dict(layout=list(xy), grid_score=round(sc, 3), grasp=g)
    save(fig, "20261110-layout_space.png")
    return info


def film_three_hands(picks=((70.0, 150.0), (70.0, 110.0), (125.0, 55.0))):
    """Three hands of the press landscape grasp the 25 mm cylinder and press it (owner, 2026-11-10 night: "the reader
    still needs some videos of grasping before hand"): the CPU plant at seed 0, each hand from its best grasp of 16,
    side by side under a label bar with the hand and its GPU score. Writes 20261110-film_three_hands.mp4 (and .jpg)."""
    import hom_films as HF
    L, xs, ys, M, _ = grid()
    clips, info = [], {}
    for lab, xy in zip("ABC", picks):
        call = HF.call_at(TASK, SIZE, *xy)
        g = HF.best_grasp(TASK, SIZE, call)
        sc = cell(xs, ys, M, *xy)
        fr, res = HF.clip(TASK, "grid2dff_size", SIZE, call, g, f"{lab}: ({xy[0]:.0f}, {xy[1]:.0f}) mm, score {sc:.2f}",
                          seed=0, var="hold" if VARIANT == "track" else "base", render=True)
        clips.append(fr)
        info[lab] = dict(layout=list(xy), grasp=int(g), grid_score=round(sc, 4), cpu_score=round(float(HF.score(TASK, res)), 3))
    HF.compose(clips, OUT / ("20261110-film_three_hands_track.mp4" if VARIANT == "track" else
                             "20261110-film_three_hands.mp4"))
    return info


# ------------------------------------------------------------------------------------------------ the EA
def ea_anatomy(seed=7, gen=1):
    """One generation of the EA (population 16): the population, its ranking, the parents' mutations, the next one."""
    L, xs, ys, M, look = grid()
    gens = []
    L.ea_search(look, np.random.default_rng(seed), xs, ys, pop=16, gens=gens)
    G = gens[gen]
    f = {k: look[k][0] for k in G["P"]}
    fig = plt.figure(figsize=(21, 6.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.05, 1, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    landscape(ax, xs, ys, M)
    P = np.array(G["P"], float)
    ax.scatter(P[:, 0], P[:, 1], s=70, c="white", edgecolors=INK, linewidths=1.6, zorder=4)
    ax.set_title("1  A population of 16 hands")
    ax = fig.add_subplot(gs[0, 1])
    order = sorted(G["P"], key=lambda k: -f[k])
    cols = [PAR if i < 2 else KEPT if i < 8 else LINE for i in range(len(order))]
    ax.barh(np.arange(len(order)), [f[k] for k in order], color=cols, height=0.78)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlim(0, 1.32)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_box_aspect(1.0)
    ax.set_xlabel("fitness: press score")
    for yv, txt, c in ((1.5, "top eighth: parents", PAR), (7.5, "top half: kept", "#b45309"),
                       (15.5, "lower half: dropped", GREY)):
        ax.axhline(yv, color=c, lw=1.2, ls="--")
        ax.text(1.31, yv - 0.25, txt, ha="right", va="bottom", fontsize=13, color=c, weight="bold")
    ax.set_title("2  Rank them by fitness")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax = fig.add_subplot(gs[0, 2])
    landscape(ax, xs, ys, M)
    for (a, raw, kid) in G["moves"]:
        ax.add_patch(Rectangle((a[0] - 15, a[1] - 15), 30, 30, fill=False, ec=PAR, lw=1.4, ls="--", zorder=3))
        ax.add_patch(FancyArrowPatch(a, kid, arrowstyle="-|>", mutation_scale=12, lw=1.4, color=INK, zorder=5,
                                     shrinkA=6, shrinkB=5))
    K = np.array(G["kids"], float)
    ax.scatter(K[:, 0], K[:, 1], s=60, c=KID, edgecolors="white", linewidths=1, zorder=6)
    A = np.array(G["parents"], float)
    ax.scatter(A[:, 0], A[:, 1], s=300, marker="*", c=STAR, edgecolors=INK, linewidths=1, zorder=7)
    cx, cy = A[:, 0].mean(), A[:, 1].mean()
    ax.set_xlim(cx - 45, cx + 45)
    ax.set_ylim(cy - 45, cy + 45)
    ax.set_title("3  Move each parent up to 15 mm")
    ax.text(0.03, 0.03, "dashed: where a child can land;\nchildren snap to the 5 mm lattice", transform=ax.transAxes,
            fontsize=12, color=INK, va="bottom", bbox=dict(fc="white", ec="none", alpha=0.8, pad=3))
    ax = fig.add_subplot(gs[0, 3])
    landscape(ax, xs, ys, M)
    Kp = np.array(G["keep"], float)
    ax.scatter(Kp[:, 0], Kp[:, 1], s=70, c=KEPT, edgecolors=INK, linewidths=1.2, zorder=4, label="kept")
    ax.scatter(K[:, 0], K[:, 1], s=70, c=KID, edgecolors=INK, linewidths=1.2, zorder=5, label="children")
    ax.legend(loc="upper right", fontsize=12, frameon=True)
    ax.set_title("4  The next generation")
    save(fig, "20261110-ea_anatomy.png")
    return dict(seed=seed, generation=gen, parents=G["parents"], best=round(max(f.values()), 3))


def ea_snapshots():
    """Where the EA has evaluated after 16, 48, 96 and 192 layouts, and the best score found against random search."""
    L, xs, ys, M, look = grid()
    best_all = max(v for v, _n in look.values())
    hist, seen = L.ea_search(look, np.random.default_rng(3), xs, ys, pop=16)
    order = [k for _c, k in hist]
    fig = plt.figure(figsize=(21, 6.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.35], wspace=0.22)
    for j, n in enumerate((16, 48, 96, 192)):
        ax = fig.add_subplot(gs[0, j])
        landscape(ax, xs, ys, M, label=j == 0)
        pts = np.array(order[:n], float)
        if n > 16:
            ax.scatter(pts[:-16, 0], pts[:-16, 1], s=22, c=KEPT, edgecolors="none", zorder=3)
        ax.scatter(pts[-16:, 0], pts[-16:, 1], s=46, c=KID, edgecolors="white", linewidths=0.8, zorder=4)
        kb = max(order[:n], key=lambda k: seen[k])
        ax.scatter([kb[0]], [kb[1]], s=320, marker="*", c=STAR, edgecolors=INK, linewidths=1, zorder=5)
        ax.set_title(f"after {n} hands: best {seen[kb]:.2f}", fontsize=15)
        if j:
            ax.set_xlabel("thumb to pair (mm)")
    ax = fig.add_subplot(gs[0, 4])
    budgets = np.arange(1, 401)
    med = {}
    for kind, col, lab in ((False, OUTER, "EA, population 16"), (True, GREY, "random search")):
        curves = []
        for r in range(200):
            h, s_ = L.ea_search(look, np.random.default_rng(100 + r), xs, ys, pop=16, random=kind)
            b = np.maximum.accumulate([s_[k] for _c, k in h])
            curves.append(np.interp(budgets, np.arange(1, len(b) + 1), b, right=b[-1]))
        med[lab] = np.median(curves, 0)
        ax.plot(budgets, med[lab], color=col, lw=3.2, label=lab)
        ax.fill_between(budgets, *np.percentile(curves, [25, 75], 0), color=col, alpha=0.16, lw=0)
    ax.axhline(best_all, color=INK, ls=":", lw=1.4)
    ax.text(395, best_all + 0.005, "best hand on the grid", ha="right", va="bottom", fontsize=12)
    ax.set_xlabel("hands evaluated")
    ax.set_ylabel("best score found")
    ax.set_ylim(best_all - 0.37 if VARIANT == "track" else 0.45, best_all + 0.06)
    ax.legend(fontsize=13, loc="lower right", frameon=False)
    ax.set_title("median of 200 searches", fontsize=15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "20261110-ea_snapshots.png")
    hit = {k: (int(np.argmax(v >= best_all - 0.02)) + 1 if (v >= best_all - 0.02).any() else None)
           for k, v in med.items()}
    return dict(best_on_grid=round(best_all, 3), hands_to_within_002=hit)


def grasp_candidates(xy):
    """The 16 candidate grasps of one hand, each rendered after the lift, with the GPU's score (mean of seeds 0-2)."""
    import hom_films as HF
    L, xs, ys, M, _ = grid()
    call = HF.call_at(TASK, SIZE, *xy)
    sc, _ = L.grid_scores(TASK, SIZE)
    g_sc = {j: float(np.mean(v)) for (c, j), v in sc.items() if c == call}
    gbest = max(g_sc, key=g_sc.get)
    fig, axs = plt.subplots(4, 4, figsize=(15, 13.2))
    for j, ax in enumerate(axs.flat):
        ax.axis("off")
        if j not in g_sc:
            continue
        img, _g, _r = still(xy, grasp=j)
        ax.imshow(img)
        best = j == gbest
        ax.set_title(f"grasp {j}" + (" (foundational pose)" if j == 0 else "") + f": {g_sc[j]:.2f}", fontsize=13,
                     color=INNER if best else INK, weight="bold" if best else "normal")
        if best:
            ax.add_patch(Rectangle((0, 0), img.shape[1] - 1, img.shape[0] - 1, fill=False, ec=INNER, lw=5))
    fig.suptitle(f"One hand, ({xy[0]:.0f}, {xy[1]:.0f}) mm: 16 candidate grasps on the 25 mm cylinder; the inner loop "
                 "keeps the best", fontsize=18, weight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, "20261110-grasp_candidates.png")
    order = sorted(g_sc, key=lambda j: -g_sc[j])                  # the slide version: two rows, best first
    fig, axs = plt.subplots(2, 8, figsize=(24, 7.0))
    for ax, j in zip(axs.flat, order):
        img, _g, _r = still(xy, grasp=j)
        ax.imshow(img[:, 40:-40])
        ax.axis("off")
        best = j == gbest
        col = INNER if best else ("#b91c1c" if g_sc[j] < 0.05 else INK)          # track: red is a score below 0.05
        ax.set_title(f"{g_sc[j]:.2f}" + ("  pose solve" if j == 0 else ""), fontsize=17, color=col,
                     weight="bold" if best else "normal")
        if best:
            ax.add_patch(Rectangle((0, 0), img.shape[1] - 81, img.shape[0] - 1, fill=False, ec=INNER, lw=6))
    fig.suptitle("Sixteen grasps of one hand, best to worst: press score of each (red: " + ("score below 0.05)"
                 if VARIANT == "track" else "the cylinder drops)"),
                 fontsize=19, weight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(hspace=0.32)
    save(fig, "20261110-grasp_candidates_wide.png")
    return dict(layout=list(xy), best_grasp=gbest, scores={str(k): round(v, 3) for k, v in sorted(g_sc.items())})


def fp_views(xy=(110.0, 85.0), task="twist", d=40.0):
    """The talk's first "Which grasp?" figure: the foundational pose, the best and the worst grasp of one hand from three
    sides (hom_landscape.cmd_views, as rendered for the talk on 2026-11-02 on hand (110, 85) mm). In the track variant
    the hand is the twist's best under the tracking score, (105, 70) mm, and the grasps are ranked by that score."""
    import argparse
    import hom_films as HF
    import hom_landscape as L
    if VARIANT == "track":
        xy = (105.0, 70.0)
    L.GRID_SRC, L.GRID_ROWS = "grid2dff", L.ROWS / ("grid2dff_v4b_sv25r" if VARIANT == "track" else "grid2dff_v4b_sv25")
    call = HF.call_at(task, d, *xy)
    name = "20261111-twist_grasps" + ("_track" if VARIANT == "track" else "") + ".png"
    L.cmd_views(argparse.Namespace(task=task, d=d, calls=[call], out=str(OUT / name)))
    return dict(layout=list(xy), call=call, file=name)


def tuning(plant="sv25"):
    """The inner loop's tuning: per task, the median over the 30 best SCS0009 layouts of the best candidate so far
    (seeds 0-2) by CEM iteration, tuning the grasp alone (S2) or the grasp, forces and contact modes (S3), against
    HOM's defaults with the best of 16 grasps (S1). Data of the score-levels page (hom_score_refine_page)."""
    import hom_score_refine_page as R
    J = R.report(plant)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.4), sharey=True)
    out = {}
    for ax, task in zip(axs, ("twist", "roll", "press")):
        s1 = R.med(R.vals(J["tasks"][task]["layouts"], "s1"))
        ax.axhline(s1, color=GREY, ls=":", lw=2, label="S1: defaults, best of 16 grasps")
        row = {"s1": round(float(s1), 3)}
        for level, c, lab in (("grasp", "#6ee7b7", "S2: CEM-refined grasp"), ("full", INNER, "S3: CEM-refined grasp and HOM")):
            G = R.gens(plant, level, task)
            its = sorted({g["it"] for g in G})
            m = [R.med([g["best"] for g in G if g["it"] == k]) for k in its]
            q = [np.percentile([g["best"] for g in G if g["it"] == k], [25, 75]) for k in its]
            ax.fill_between(its, [a for a, _b in q], [b for _a, b in q], color=c, alpha=0.18, lw=0)
            ax.plot(its, m, "-o", color=c, lw=3, ms=6, label=lab)
            row[level] = round(float(m[-1]), 3)
        out[task] = row
        ax.set_title({"twist": "Twist", "roll": "Roll", "press": "Press"}[task])
        ax.set_xlabel("CEM iteration")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("best score so far (median of 30 hands)")
    axs[0].legend(loc="lower right", fontsize=12, frameon=False)
    save(fig, "20261110-tuning.png")
    return out


def tuning_track():
    """The inner loop's tuning under the tracking score (owner, 2026-11-11: "hom tuning alone is missing here ... this is
    with the old scoring"): per task, the median over the 30 best SCS0009 layouts (S1 on the tracking score) of the best
    candidate so far (seeds 0-2) by CEM iteration, for three arms from the same start (S1's grasp, HOM's defaults) and
    the same budget (32 candidates x 8 iterations): the grasp alone, HOM's forces and contact modes alone, and both.
    Rows: hom_grasp_refine.py --score track, docs/experiments/20261103-hom_score_refine/rows/sv25_track/."""
    R = ROOT / "docs/experiments/20261103-hom_score_refine/rows/sv25_track"
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.4), sharey=True)
    out = {}
    for ax, (task, d) in zip(axs, (("twist", 40), ("roll", 40), ("press", 25))):
        lays = [L_ for L_ in json.loads((R / f"{task}_layouts.json").read_text())["layouts"] if L_["group"] == "top"]
        top = {L_["call"] for L_ in lays}                 # the 30 best; the sample's random layouts ran level both only
        s1 = float(np.median([L_["s1"] for L_ in lays]))
        ax.axhline(s1, color=GREY, ls=":", lw=2, label="S1: defaults, best of 16 grasps")
        row = {"s1": round(s1, 4)}                       # four decimals: the slide rounds them to two
        for level, c, lab in (("grasp", "#6ee7b7", "CEM-refined grasp"), ("hom", "#7c3aed", "CEM-refined HOM"),
                              ("both", INNER, "CEM-refined grasp and HOM")):
            gp = R / level / f"{task}_d{d}_gens.jsonl"
            G = [json.loads(x) for x in gp.read_text().splitlines() if x.strip()] if gp.exists() else []
            G = [g for g in G if g["call"] in top]
            if not G:
                continue
            its = sorted({g["it"] for g in G})
            m = [float(np.median([g["best"] for g in G if g["it"] == k])) for k in its]
            q = [np.percentile([g["best"] for g in G if g["it"] == k], [25, 75]) for k in its]
            ax.fill_between(its, [a_ for a_, _b in q], [b_ for _a, b_ in q], color=c, alpha=0.16, lw=0)
            ax.plot(its, m, "-o", color=c, lw=3, ms=6, label=lab)
            row[level] = round(m[-1], 4)
            row[level + "_layouts"] = len({g["call"] for g in G})
        out[task] = row
        ax.set_title({"twist": "Twist", "roll": "Roll", "press": "Press"}[task])
        ax.set_xlabel("CEM iteration")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("best score so far (median of 30 hands)")
    axs[0].legend(loc="lower right", fontsize=12, frameon=False)
    save(fig, "20261110-tuning.png")
    reps = sorted(R.glob("*-tuning_levels.json"))
    if reps:            # the arms' picks at the validation seeds 20-24, elitist (hom_grasp_refine.py report_levels)
        rep = json.loads(reps[-1].read_text())["tasks"]
        def arm(x):     # one arm on the 30 best, with its Spearman correlation and best layout over the whole sample
            return {"median": x["median"], "gain": x["gain"]["median"], "rho": x["rho_s1_val"], "best": x["best"]["xy"],
                    **({"n_all": x["all"]["n"], "rho_all": x["all"]["rho_s1_val"], "best_all": x["all"]["best"]["xy"]}
                       if "all" in x else {})}
        out["val"] = {t: {"layouts": r["n"], "layouts_all": r.get("n_all", r["n"]), "s1": r["median_s1_val"],
                          "best_s1": r["best_s1_val"]["xy"],
                          "best_s1_all": r.get("best_s1_val_all", r["best_s1_val"])["xy"],
                          **{lv: arm(r[k]) for lv in ("grasp", "hom", "both") for k in [f"{lv}_elitist_val"] if k in r},
                          **{lv: arm(r[k]) for lv, k in (("shared", "shared_val"), ("shared_s1grasp", "shared_s1grasp_val"))
                             if k in r and "median" in r[k]}}
                      for t, r in rep.items()}
        out["val_report"] = str(reps[-1].relative_to(ROOT))
    return out


FPR = ROOT / "docs/experiments/20261029-foundational_pose_review/rows"
SRR = ROOT / "docs/experiments/20261103-hom_score_refine/rows"
D_MM = {"twist": 40, "roll": 40, "pinpush": 25, "press": 25}


def _timing(path, lo=None, hi=None):
    """Sums over the GPU batch timing lines of a rows file (s): scene build, physics and scoring, and the rollouts. With
    lo and hi, only the batches stamped in [lo, hi)."""
    p = Path(path)
    T = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    T = [x for x in T if (lo is None or x["when"] >= lo) and (hi is None or x["when"] < hi)]
    return {"build": sum(x["build_s"] for x in T), "physics": sum(x["run_s"] for x in T),
            "scoring": sum(x["score_s"] for x in T), "rollouts": sum(x["worlds"] for x in T)}


def _log_lines(path):
    """(seconds of the day, rest of the line) of a queue log's 'HH:MM:SS ...' lines."""
    out = []
    for x in Path(path).read_text().splitlines():
        if re.match(r"\d\d:\d\d:\d\d ", x):
            out.append((int(x[:2]) * 3600 + int(x[3:5]) * 60 + int(x[6:8]), x[9:]))
    return out


TASKS3 = ("twist", "roll", "press")


def _compute_record():
    """The GPU time of one rollout, of the S1 landscapes and of CEM-refining grasp and HOM on every layout, for the
    twist, the roll and the press (owner, 2026-11-11: "1 layout x 1 rollout", "s1 landscapes (scale down to three
    tasks)", "CEM-refinement (remove one level)"). A layout counts when the pose toy finds the task's grasp on it: of the
    lattice's 625, 527 for the 40 mm sphere and 369 for the press's 25 mm cylinder. S1: the grids' batch time (16 grasps
    x 3 seeds per layout). CEM: 768 rollouts per layout (32 candidates x 8 iterations x 3 seeds) at each task's measured
    wall time per rollout of its grasp-and-HOM runs. Deck: S3 of the score-levels study of 11-03 (rows sv25/full,
    queue log logs/20261103-hom_score_refine/queue.log, which marks each step's start). v2: level both of 11-11 on the 30
    best layouts (logs/20261111-hom_tuning_track/queue.log) and on 30 others (logs/20261111-talk_v2/full_set.log), rows
    sv25_track/both. One rollout: the twist's batch time per rollout in its S1 grid."""
    track = VARIANT == "track"
    gdir = FPR / ("grid2dff_v4b_sv25r" if track else "grid2dff_v4b_sv25")
    s1 = {t: _timing(gdir / f"{t}_d{D_MM[t]}_timing.jsonl") for t in TASKS3}
    if not all(v["rollouts"] for v in s1.values()):
        raise FileNotFoundError(f"no timing lines in {gdir.relative_to(ROOT)}")
    comps = ("build", "physics", "scoring")
    if track:
        tune = {}
        for _, x in _log_lines(ROOT / "logs/20261111-hom_tuning_track/queue.log"):
            m = re.match(r"(\w+?)_(grasp|hom|both)(?: resume)? rc \d+, peak \d+ kB, ([\d.]+) s", x)
            if m:
                tune[(m[1], m[2])] = tune.get((m[1], m[2]), 0.0) + float(m[3])
        full = {}
        fs = ROOT / "logs/20261111-talk_v2/full_set.log"
        for _, x in (_log_lines(fs) if fs.exists() else []):
            m = re.match(r"both_(\w+) rc 0, peak \d+ kB, ([\d.]+) s", x)
            if m:
                full[m[1]] = float(m[2])
        wall = {t: tune[(t, "both")] + full.get(t, 0.0) for t in TASKS3}
        hi = {t: "2026-11-11 23:59" if t in full else "2026-11-11 10:10" for t in TASKS3}
        cem = {t: _timing(SRR / "sv25_track/both" / f"{t}_d{D_MM[t]}_timing.jsonl", "2026-11-11 01:00", hi[t])
               for t in TASKS3}
        runs = {f"{t}_{lv}": round(w, 1) for (t, lv), w in tune.items()}
    else:
        L = _log_lines(ROOT / "logs/20261103-hom_score_refine/queue.log")
        step = {x.split()[0]: (s1_ - s0) % 86400 for (s0, x), (s1_, _) in zip(L, L[1:])}
        wall = {t: float(step[f"sv25_full_{t}"]) for t in TASKS3}
        cem = {t: _timing(SRR / "sv25/full" / f"{t}_d{D_MM[t]}_timing.jsonl", "2026-11-03 18:00", "2026-11-03 22:10")
               for t in TASKS3}
        runs = {f"{t}_{lv}": float(step[f"sv25_{lv}_{t}"]) for t in D_MM for lv in ("grasp", "full")}
    if not all(v["rollouts"] for v in cem.values()):
        raise FileNotFoundError("a grasp-and-HOM run has no timing lines")
    layouts = {t: round(s1[t]["rollouts"] / 48) for t in TASKS3}
    rate = {t: wall[t] / cem[t]["rollouts"] for t in TASKS3}
    s1_s = sum(s1[t][k] for t in TASKS3 for k in comps)
    cem_s = sum(layouts[t] * 768 * rate[t] for t in TASKS3)
    phys = sum(cem[t]["physics"] for t in TASKS3) / sum(wall.values())
    one = {k: s1["twist"][k] / s1["twist"]["rollouts"] for k in comps}
    meas_wall = sum(wall.values())
    cem_parts = {k: sum(cem[t][k] for t in TASKS3) / meas_wall * cem_s for k in comps}
    cem_parts["other"] = max(0.0, cem_s - sum(cem_parts.values()))
    parts = {"one": {**{k: round(v, 5) for k, v in one.items()}, "other": 0.0},
             "s1": {**{k: round(sum(s1[t][k] for t in TASKS3), 1) for k in comps}, "other": 0.0},
             "cem": {k: round(v, 1) for k, v in cem_parts.items()}}
    return {"made": time.strftime("%Y-%m-%d %H:%M"), "variant": VARIANT or "deck", "tasks": list(TASKS3),
            "lattice": 625, "layouts": layouts, "parts": parts,
            "one": {"task": "twist", "s": round(sum(s1["twist"][k] for k in comps) / s1["twist"]["rollouts"], 4)},
            "s1": {"s": round(s1_s, 1), "rollouts": sum(s1[t]["rollouts"] for t in TASKS3),
                   "physics_share": round(sum(s1[t]["physics"] for t in TASKS3) / s1_s, 3)},
            "cem": {"s": round(cem_s, 1), "rollouts": sum(layouts.values()) * 768, "physics_share": round(phys, 3),
                    "ms_per_rollout": {t: round(1000 * rate[t], 1) for t in TASKS3},
                    "measured_rollouts": {t: cem[t]["rollouts"] for t in TASKS3},
                    "measured_s": {t: round(wall[t], 1) for t in TASKS3}},
            "runs_s": runs, "gpu": "NVIDIA GeForce RTX 4070 Ti SUPER"}


def _hm(s):
    return f"{s / 60:.0f} min" if s < 3600 else f"{s / 3600:.1f} h"


def compute_cost():
    """The GPU time behind the CEM-refinement slide (owner, 2026-11-11: "a full, simple timing chart"; then "should
    only have 3 bars. might need logarithmic time scale"): one rollout, the S1 landscapes of the twist, the roll and the
    press, and an estimate for CEM-refining grasp and HOM on every layout of them, on a log time axis (_compute_record).
    The numbers are kept in figures.json (compute, compute_track), so the chart redraws without the logs or the
    gitignored grids."""
    key = "compute_track" if VARIANT == "track" else "compute"
    path = OUT / "20261110-figures.json"
    old = (json.loads(path.read_text()) if path.exists() else {}).get(key)
    try:
        rec = _compute_record()
    except (FileNotFoundError, KeyError, OSError) as e:
        if not old:
            raise
        print(f"compute: {e}; drawing the record of {old['made']}", flush=True)
        rec = old
    n = sum(rec["layouts"].values())

    def dur(x):
        return f"{1000 * x:.0f} ms" if x < 1 else (f"{x:.0f} s" if x < 60 else (f"{x / 60:.0f} min" if x < 3600 else
                                                                                f"{x / 3600:.1f} h"))
    rows = [("one", f"One rollout\n1 layout \u00d7 1 grasp \u00d7 1 seed ({rec['one']['task']})", rec["one"]["s"],
             None, "#64748b"),
            ("s1", f"S1 landscapes, three tasks\n{n:,} layouts \u00d7 48 rollouts", rec["s1"]["s"], rec["s1"]["rollouts"],
             "#64748b"),
            ("cem", f"CEM-refined grasp and HOM, three tasks\n{n:,} layouts \u00d7 768 rollouts", rec["cem"]["s"],
             rec["cem"]["rollouts"], "#64748b")]
    fig = plt.figure(figsize=(14.0, 5.0))               # text >= 13 px when the slide shows it 1,150 px wide
    gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1.0], wspace=0.06)
    ax, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    x0 = 0.01
    for i, (k, _lab, x, nr, col) in enumerate(rows):
        y, est = len(rows) - 1 - i, k == "cem"
        ax.barh(y, x - x0, left=x0, height=0.56, color=col, alpha=0.55 if est else 1.0, hatch="///" if est else None,
                edgecolor="white", linewidth=0.8)
        txt = (f"about {dur(x)}, {nr / 1e6:.2f} M rollouts (estimate)" if est else
               dur(x) + (f", {nr:,} rollouts" if nr else ""))
        if nr:                                          # inside the long bars, at their end
            ax.text(x / 1.4, y, txt, va="center", ha="right", fontsize=15, color=INK, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
        else:
            ax.text(x * 1.35, y, txt, va="center", ha="left", fontsize=16, color=INK)
    ax.set_xscale("log")
    ticks = (0.01, 0.1, 1.0, 10.0, 60.0, 600.0, 3600.0, 36000.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["10 ms", "0.1 s", "1 s", "10 s", "1 min", "10 min", "1 h", "10 h"])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(x0, 1.2e5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab, _, _, _ in rows][::-1], fontsize=15)
    ax.set_xlabel(f"GPU time on one {rec['gpu'].replace('NVIDIA GeForce ', '')}, log scale", fontsize=15)
    ax.tick_params(axis="x", labelsize=14)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    parts = (("physics", "GPU physics", OUTER, "white"), ("build", "scene build", "#93c5fd", INK),
             ("scoring", "scoring", SIM, "white"), ("other", "candidate sampling", GREY, INK))
    for i, (k, *_r) in enumerate(rows):
        y, pr = len(rows) - 1 - i, rec["parts"][k]
        tot, left = sum(pr.values()), 0.0
        for c, lab, col, tc in parts:
            w = 100.0 * pr[c] / tot
            ax2.barh(y, w, left=left, height=0.56, color=col, edgecolor="white", linewidth=0.8,
                     label=lab if i == len(rows) - 1 else None)
            if w >= 5:
                ax2.text(left + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=14 if w >= 15 else 12.5,
                         color=tc)
            left += w
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.6, len(rows) - 0.4)
    ax2.set_yticks([])
    ax2.set_xlabel("share of the bar\u2019s time (%)", fontsize=15)
    ax2.tick_params(axis="x", labelsize=14)
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=14, frameon=False)
    save(fig, "20261111-compute.png")
    return rec


ROLES = (("twist", "f_tripod", "Twist: each of three fingers"), ("roll", "f_tripod", "Roll: each of three fingers"),
         ("press", "f_pinch", "Press: the two pinching fingers"), ("press", "f_press", "Press: the pressing finger"),
         ("pinpush", "f_pinch", "Pin-push: the two pinching fingers"), ("pinpush", "f_drive", "Pin-push: the pusher"))
C_FORCE = "#b91c1c"


def _forces_of_code():
    """HOM's default contact forces (hom_tasks.PRM) and the ranges the CEM searches (hom_grasp_refine.FORCES), read from
    the sources so that neither module is imported."""
    import ast
    src = (ROOT / "scripts/hom/hom_tasks.py").read_text()
    prm = src[src.index("\nPRM = {"):]
    prm = prm[:prm.index("}") + 1]
    default = {k: float(v) for k, v in re.findall(r'"(f_\w+)": ([\d.]+)', prm)}
    tree = ast.parse((ROOT / "scripts/hom/hom_grasp_refine.py").read_text())
    rng = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "FORCES" for t in n.targets))
    return default, rng


def contact_force():
    """Where HOM's contact forces act and which values the CEM picked (owner, 2026-11-11: "give the audience an
    intuition for how these selected newtons come from / relate to HOM control"). (a) HOM moves each fingertip on the
    object in its contact frame {L1}: spin, roll and slide, with the separation along x1 held at zero (Eq. 2); the
    contact force F acts along that same x1: each finger in contact aims past the surface and presses with F, and
    friction then allows up to mu F along the surface. (b) The newtons of each role: HOM's default, the range the CEM
    searches and the forces it picked on the 30 best layouts of each task (the picks at the validation seeds; the deck:
    S3 of the score-levels study, rows sv25/val; v2: the levels hom and both, rows sv25_track/val). Drawn in the style
    of the HOM review page's Eq. (1) figure (scripts/hom/hom_eq1_figures.py)."""
    import hom_eq1_figures as E
    from matplotlib.patches import Circle, Polygon
    track = VARIANT == "track"
    default, rng = _forces_of_code()
    mu, weight = 1.0, 0.025 * 9.81                 # the HOM scenes' friction and their objects' weight (N)
    rdir = SRR / ("sv25_track" if track else "sv25")
    levels = (("hom", "CEM-refined HOM", "#7c3aed"), ("both", "CEM-refined grasp and HOM", INNER)) if track else \
        (("S3", "S3: CEM-refined grasp and HOM", INNER),)
    picks = {}
    for task in D_MM:
        p = rdir / "val" / f"{task}_d{D_MM[task]}.jsonl"
        lp = rdir / f"{task}_layouts.json"
        if not p.exists() or not lp.exists():
            continue
        top = {L_["call"] for L_ in json.loads(lp.read_text())["layouts"] if L_.get("group", "top") == "top"}
        for x in p.read_text().splitlines():
            r = json.loads(x) if x.strip() else None
            if r and r["pick"] in [lv for lv, _, _ in levels] and r["call"] in top:
                for k, v in r["prm"].items():
                    if k.startswith("f_"):
                        picks.setdefault((task, k, r["pick"]), {})[r["call"]] = float(v)
    rows = [(t, k, lab) for t, k, lab in ROLES if any((t, k, lv) in picks for lv, _, _ in levels)]
    style = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
             "font.size": 15}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(16.5, 6.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.06)
        S = E.scene(gap=0.0)
        p1, x1, z1 = S["p1"], S["x1"], S["z1"]
        lim = ((p1[0] - 30.0, p1[0] + 33.0), (p1[1] - 24.0, p1[1] + 21.0))
        ax = fig.add_subplot(gs[0, 0])
        E.draw_hand(ax, S, fin_alpha=1.0, obj_alpha=1.0)
        ax.set_aspect("equal")
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
        ax.axis("off")
        ax.set_title("(a) HOM sets the motion, $F$ sets the press", fontsize=17, pad=6)
        ghost = S["pad"] + 3.0 * x1                       # the finger aims past the surface
        ax.add_patch(Circle(tuple(ghost), E.RHO, fc="none", ec=E.C_FIN, lw=1.5, ls=(0, (4, 3)), zorder=6))
        E.frame(ax, p1, x1, None, L=6.0, ms=11, lw=1.8)
        half = math.atan(mu)
        tip = p1 + 15.0 * x1
        for s_ in (1.0, -1.0):                            # the friction cone, half-angle atan(mu)
            ax.plot(*np.c_[p1, p1 + 17.0 * (math.cos(half) * x1 + s_ * math.sin(half) * z1)], color=C_FORCE, lw=1.3,
                    ls=(0, (4, 3)), zorder=7)
        ax.add_patch(Polygon([p1, p1 + 17.0 * (math.cos(half) * x1 + math.sin(half) * z1),
                              p1 + 17.0 * (math.cos(half) * x1 - math.sin(half) * z1)], closed=True, fc=C_FORCE,
                             alpha=0.08, ec="none", zorder=6))
        E.arrow(ax, p1, tip, C_FORCE, lw=3.6, ms=21, z=9)
        E.label(ax, tip + np.array([1.5, 3.5]), "$F$", C_FORCE, fs=20, ha="left")
        E.arrow(ax, tip, tip - 0.85 * mu * 15.0 * z1, C_FORCE, lw=1.8, ms=14, ls=(0, (3, 2)), z=9)
        E.label(ax, tip - 0.85 * mu * 15.0 * z1 + np.array([1.0, 2.5]), r"friction $\leq \mu F$", C_FORCE, fs=15,
                ha="left")
        q = p1 - 5.0 * x1 + 3.0 * z1                      # HOM's motion, drawn on the finger's side
        E.arrow(ax, q, q - 13.0 * z1, E.AX_Z, lw=3.0, ms=18, z=9)
        E.label(ax, q - 13.0 * z1 + np.array([-2.5, 2.5]), "slide", E.AX_Z, fs=16, ha="right")
        E.curved(ax, p1, 6.5, 190, 330, "#111111", lw=2.0, ms=14)
        E.label(ax, p1 + np.array([-13.5, -3.0]), "roll", "#111111", fs=16)
        E.label(ax, p1 + 7.5 * x1 + np.array([0.5, -3.2]), "$x_1$", E.AX_X, fs=15)
        E.label(ax, ghost + np.array([-4.0, -14.0]), "aimed past\nthe surface", E.C_FIN, fs=15.5)
        ax.text(lim[0][0] + 0.8, lim[1][0] + 0.8, "HOM, Eq. (2): the fingertip's spin, roll\nand slide on the object; "
                "separation 0", fontsize=15.5, va="bottom", ha="left", zorder=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9ca3af", lw=0.8))
        ax.text(lim[0][1] - 0.8, lim[1][1] - 0.8, "contact force: $F$ along $x_1$,\nfriction up to $\\mu F$ along "
                "the surface", fontsize=15.5, va="top", ha="right", color=C_FORCE, zorder=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_FORCE, lw=0.8))
        ax = fig.add_subplot(gs[0, 1])
        for i, (task, key, lab) in enumerate(rows):
            y = len(rows) - 1 - i
            lo, hi = rng[task][key]
            ax.plot([lo, hi], [y, y], color=LINE, lw=12, solid_capstyle="butt", zorder=1)
            for j, (lv, _, col) in enumerate(levels):
                v = np.array(sorted(picks.get((task, key, lv), {}).values()))
                if len(v):                     # one dot per layout, spread up and down so that equal picks stay visible
                    off = 0.0 if len(levels) == 1 else (-0.13 if j == 0 else 0.13)
                    yy = y + off + np.random.default_rng(i * 10 + j).uniform(-0.08, 0.08, len(v))
                    ax.scatter(v, yy, s=24, color=col, alpha=0.75, lw=0, zorder=3)
            ax.scatter([default[key]], [y], marker="D", s=90, color=INK, zorder=4)
        ax.axvline(weight, color=GREY, ls=":", lw=1.8, zorder=0)
        ax.text(weight + 0.08, len(rows) - 0.45, f"object weight {weight:.2f} N", color="#6b7280", fontsize=15,
                va="bottom")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([lab for _, _, lab in rows][::-1], fontsize=16)
        ax.yaxis.tick_right()
        ax.set_xlim(0, 6.3)
        ax.set_ylim(-0.7, len(rows) - 0.2)
        ax.set_xlabel("contact force $F$ (N)", fontsize=16)
        ax.tick_params(axis="x", labelsize=15)
        for sp in ("top", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_title("(b) The newtons of each role", fontsize=17, pad=66)
        hs = [plt.Line2D([], [], marker="D", ls="", color=INK, ms=8, label="HOM's default"),
              plt.Line2D([], [], lw=8, color=LINE, label="range the CEM searches")]
        hs += [plt.Line2D([], [], marker="o", ls="", color=c, ms=6, label=f"{n}, 30 layouts") for _, n, c in levels]
        ax.legend(handles=hs, loc="lower center", fontsize=15, frameon=False, bbox_to_anchor=(0.5, 1.0), ncol=2)
        save(fig, "20261111-contact_force.png")
    med = {f"{t}_{k}_{lv}": round(float(np.median(list(v.values()))), 2) for (t, k, lv), v in picks.items()}
    return {"default": default, "range": {t: {k: list(v) for k, v in r.items()} for t, r in rng.items()}, "mu": mu,
            "weight_N": round(weight, 3), "median_pick": med, "rows": [f"{t}_{k}" for t, k, _ in rows]}


def pinch_film():
    """Three panels of the hand-brake study's film (owner, 2026-11-11: "a video ... showing how without condim 4, just
    basic point contacts, we can't stabilize a grasp"): thumb and index of the SR2 hand lower the screwdriver from
    horizontal by relaxing their pinch, in MuJoCo with point contact (condim 3), with condim 4 and with the 1 mm sphere
    pads (docs/experiments/20261001-hom_hand_brake, films *_d15.mp4 of scripts/hom/hom_hand_brake.py), each under a label
    bar. Writes media/20261111-pinch_three_contacts.mp4 and its poster."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    src = ROOT / "docs/experiments/20261001-hom_hand_brake/media"
    panels = (("mp3", "Point contact (condim 3)"), ("p4s", "Condim 4: torsional friction"),
              ("s1", "1 mm sphere pads"))
    clips = [[f for f in imageio.get_reader(str(src / f"20261001-hand_brake_{k}_d15.mp4"))] for k, _ in panels]
    n = min(len(c) for c in clips)
    h, w = clips[0][0].shape[:2]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    bar = Image.new("RGB", (w * len(panels), 40), "white")
    dr = ImageDraw.Draw(bar)
    for i, (_, lab) in enumerate(panels):
        dr.text((i * w + 10, 8), lab, fill="black", font=font)
    bar = np.asarray(bar)
    frames = [np.concatenate([bar, np.concatenate([c[i] for c in clips], axis=1)], axis=0) for i in range(n)]
    out = OUT / "20261111-pinch_three_contacts.mp4"
    imageio.mimwrite(str(out), frames, fps=25, codec="libx264", quality=7, macro_block_size=8)
    Image.fromarray(frames[min(n - 1, 48)]).save(OUT / "20261111-pinch_three_contacts_poster.jpg", quality=88)
    rows = {}
    for x in (ROOT / "docs/experiments/20261001-hom_hand_brake/hand_brake.jsonl").read_text().splitlines():
        r = json.loads(x)
        if r.get("d_mm") == 15.0:
            rows[r["spec"]] = {"N_start": round(r["N_start"], 2), "N_at_45": round(r["N_at_45"], 2),
                               "N_at_80": round(r["N_at_80"], 2), "phi_end": round(r["phi_end"], 1),
                               "phi_max": round(r["phi_max"], 1)}
    print("wrote", out.relative_to(ROOT), f"{out.stat().st_size / 1e6:.2f} MB, {n} frames", flush=True)
    return {"frames": n, "size": [w * len(panels), h + 40], "brake_d15": rows}


def drake_hydro(E=2.5e6, F=3.0, mu=1.0):
    """Drake's hydroelastic contact for the talk (owner, 2026-11-11: "an explanation of drake and its hydroelastic
    model ... zoomed in on drake with some more clarity / visual focus/detail on how it actually works (drake) in hom").
    The 10.55 mm fingertip sphere pressed with F onto the 12.5 mm screwdriver, the tool's axis along x, computed from the
    law Drake uses for a compliant sphere, p = E d / R with d the depth below the sphere's surface: (a) the field in the
    section through the tool's axis, (b) the contact surface (the tool's surface inside the fingertip) with the pressure
    on it, (c) the patch seen through the tool with the friction arm r = int(rho p dA) / int(p dA)."""
    from matplotlib.patches import Circle, Rectangle
    R, rc = 10.55e-3, 6.25e-3
    th = np.linspace(-0.9, 0.9, 721)                       # rad around the tool from its top
    xs = np.linspace(-6e-3, 6e-3, 721)
    TH, X = np.meshgrid(th, xs, indexing="ij")
    Y, Z = rc * np.sin(TH), rc * np.cos(TH)
    dA = rc * (th[1] - th[0]) * (xs[1] - xs[0])

    def field(delta):
        cz = rc + R - delta
        d = R - np.sqrt(X ** 2 + Y ** 2 + (Z - cz) ** 2)
        return np.where(d > 0, E * d / R, 0.0)
    lo, hi = 1e-7, 2e-3
    for _ in range(60):                                     # the approach at which the normal force is F
        mid = 0.5 * (lo + hi)
        Fz = float((field(mid) * np.cos(TH)).sum() * dA)
        lo, hi = (mid, hi) if Fz < F else (lo, mid)
    delta = 0.5 * (lo + hi)
    P = field(delta)
    rho = np.sqrt(X ** 2 + Y ** 2)
    arm = float((P * rho).sum() / P.sum())
    area = float((P > 0).sum() * dA)
    style = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
             "font.size": 15}
    with plt.rc_context(style):
        fig, axs = plt.subplots(1, 3, figsize=(16.0, 5.4), gridspec_kw={"width_ratios": [1.0, 1.0, 1.0], "wspace": 0.22})
        cz = rc + R - delta
        # (a) the field in the fingertip, the section y = 0 through the tool's axis
        ax = axs[0]
        g = np.linspace(-R, R, 400)
        GX, GZ = np.meshgrid(g, g)
        rr = np.sqrt(GX ** 2 + GZ ** 2)
        Pf = np.where(rr <= R, E * (R - rr) / R, np.nan)
        ax.add_patch(Rectangle((-16, -rc * 1e3), 32, 2 * rc * 1e3, fc="#e5e7eb", ec="#4b5563", lw=1.4, zorder=1))
        cs = ax.contourf(GX * 1e3, (GZ + cz) * 1e3, Pf / 1e6, levels=np.linspace(0, E / 1e6, 11), cmap="Blues",
                         zorder=2)
        ax.add_patch(Circle((0, cz * 1e3), R * 1e3, fill=False, ec="#1e3a8a", lw=1.6, zorder=3))
        a0 = float(np.sqrt(max(R ** 2 - (R - delta) ** 2, 0.0)))
        ax.plot([-a0 * 1e3, a0 * 1e3], [rc * 1e3, rc * 1e3], color=C_FORCE, lw=4.0, solid_capstyle="butt", zorder=4)
        ax.text(9.5, -2.5, "tool", fontsize=15, color="#374151", ha="right")
        ax.text(-R * 1e3 - 0.5, cz * 1e3 + R * 1e3 * 0.75, "fingertip", fontsize=15, color="#1e3a8a", ha="right")
        ax.set_xlim(-17, 14)
        ax.set_ylim(-8, cz * 1e3 + R * 1e3 + 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        cb = fig.colorbar(cs, ax=ax, fraction=0.05, pad=0.02, ticks=[0, E / 2e6, E / 1e6])
        cb.set_label("field pressure (MPa)", fontsize=14)
        cb.ax.tick_params(labelsize=13)
        ax.set_title("(a) The fingertip's field:\n$p = E\\,d/R$, zero on its surface", fontsize=16)
        # (b) the contact surface and its pressure, zoomed
        ax = axs[1]
        a = float(np.sqrt(max(R ** 2 - (R - delta) ** 2, 0.0)))
        zz = np.linspace(-1, 1, 7)
        sec = np.linspace(-1.25 * a, 1.25 * a, 400)
        ax.add_patch(Rectangle((-8, -4), 16, 4 + rc * 1e3 - rc * 1e3, fc="#e5e7eb", ec="none"))
        t_ = np.linspace(-np.pi, np.pi, 400)
        ax.plot(R * 1e3 * np.cos(t_), cz * 1e3 + R * 1e3 * np.sin(t_), color="#1e3a8a", lw=1.6)
        ax.axhline(rc * 1e3, color="#4b5563", lw=1.4)
        pin = np.where(np.abs(sec) < a, E * ((R - np.sqrt(sec ** 2 + (rc - cz) ** 2)) / R), 0.0)
        pin = np.maximum(pin, 0.0)
        ax.plot(sec * 1e3, np.full_like(sec, rc * 1e3), color=C_FORCE, lw=4.0, solid_capstyle="butt",
                alpha=np.nan if False else 1.0, zorder=3)
        sc = 2.2 / max(pin.max(), 1.0)                          # mm of arrow per Pa
        for x_ in np.linspace(-0.92 * a, 0.92 * a, 9):
            pv = E * max(R - np.sqrt(x_ ** 2 + (rc - cz) ** 2), 0.0) / R
            if pv > 0:
                ax.annotate("", xy=(x_ * 1e3, rc * 1e3), xytext=(x_ * 1e3, rc * 1e3 + sc * pv * 1.0 + 0.05),
                            arrowprops=dict(arrowstyle="-|>", color=C_FORCE, lw=1.6))
        ax.plot(sec * 1e3, rc * 1e3 + sc * pin, color=C_FORCE, lw=1.2, ls=(0, (3, 2)))
        ax.text(0, rc * 1e3 + sc * pin.max() + 0.35, f"peak {pin.max() / 1e6:.2f} MPa", ha="center", fontsize=14,
                color=C_FORCE)
        ax.text(0, rc * 1e3 - 1.15, "contact surface: the tool's\nsurface inside the fingertip", ha="center",
                va="top", fontsize=14, color=C_FORCE)
        ax.set_xlim(-1.6 * a * 1e3, 1.6 * a * 1e3)
        ax.set_ylim(rc * 1e3 - 3.0, rc * 1e3 + sc * pin.max() + 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"(b) Section through the overlap,\n{delta * 1e6:.0f} \u00b5m deep at {F:g} N", fontsize=16)
        # (c) the patch seen through the tool
        ax = axs[2]
        S_ = rc * TH
        m = P > 0
        ext = np.abs(X[m]).max() * 1.35
        cs2 = ax.contourf(X * 1e3, S_ * 1e3, np.where(m, P / 1e6, np.nan), levels=10, cmap="Reds")
        ax.add_patch(Circle((0, 0), arm * 1e3, fill=False, ec="k", lw=1.6, ls=(0, (4, 3))))
        ax.text(arm * 1e3 * 0.72, arm * 1e3 * 0.72 + 0.25, r"$\bar r$", fontsize=17, ha="left")
        ax.set_xlim(-ext * 1e3, ext * 1e3)
        ax.set_ylim(-ext * 1e3, ext * 1e3)
        ax.set_aspect("equal")
        ax.set_xlabel("along the tool (mm)", fontsize=14)
        ax.set_ylabel("around the tool (mm)", fontsize=14)
        ax.tick_params(labelsize=13)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.set_title("(c) The patch: $F = \\int p\\,dA$,\n$T \\leq \\mu \\int r\\,p\\,dA = \\mu F \\bar r$",
                     fontsize=16)
        ax.text(0.02, 0.02, f"area {area * 1e6:.1f} mm$^2$, $\\bar r$ = {arm * 1e3:.2f} mm", transform=ax.transAxes,
                fontsize=14, va="bottom")
        save(fig, "20261111-drake_hydro.png")
    return {"E_Pa": E, "F_N": F, "mu": mu, "delta_um": round(delta * 1e6, 1), "patch_area_mm2": round(area * 1e6, 2),
            "arm_mm": round(arm * 1e3, 3), "peak_MPa": round(float(P.max()) / 1e6, 3),
            "T_max_mNm": round(mu * F * arm * 1e3, 2), "R_mm": R * 1e3, "tool_d_mm": 2e3 * rc}


FINGER_SCENE = ROOT / ("logs/20261102-hom_servo_plant/scenes/cyl025.0/"
                       "x092.5_y102.5_pads_bv2__kp0.72_kv0_fr0.18_fl0_dp0.0144.xml")   # a scene of the talk's landscapes


def finger_spheres(q=(0.0, 0.55, 0.55)):
    """One finger of the cad_v2 hand with the v4b fingertip for the talk's SR2 slide (owner, 2026-11-11: "update the
    morphit + sphere packed model with the newest fingertip"), in the four views of the cad_v1 page's Figure 4: the CAD
    part meshes; the collision spheres of the rigid parts with the pad's 1,018 sphere pads; both; the spheres from the
    dorsal side. From a scene of the talk's landscapes (cad_v2 hand, replace_tips' tcd_ell7.4x8.5x11x22 pads), thumb at
    yaw, MCP and PIP q (rad), every other geom hidden. Rendered on Mesa's software renderer (MUJOCO_GL=egl with the Mesa
    vendor file), off the GPU queue."""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    m = mujoco.MjModel.from_xml_path(str(FINGER_SCENE))
    d = mujoco.MjData(m)
    f = "thumb"
    for j, v in zip(("yaw", "mcp", "pip"), q):
        d.qpos[m.joint(f"{f}_{j}").qposadr[0]] = v
    mujoco.mj_forward(m, d)
    mine = {i for i in range(m.nbody) if m.body(i).name.startswith(f + "_")}
    kind = []
    for g in range(m.ngeom):
        nm = m.geom(g).name or ""
        if m.geom_bodyid[g] not in mine:
            kind.append(None)
        elif "_pad" in nm:
            kind.append("pad")
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            kind.append("sphere")
        elif m.geom_group[g] == 2:
            kind.append("mesh")
        elif nm.endswith("_tipvis"):
            kind.append("tipmesh")
        else:
            kind.append(None)
    col = {"mesh": (0.42, 0.36, 0.80, 1.0), "tipmesh": (0.10, 0.10, 0.11, 1.0), "sphere": (0.96, 0.55, 0.24, 1.0),
           "pad": (0.13, 0.77, 0.37, 1.0)}
    base = m.geom_rgba.copy()
    for t in range(m.ntex):                               # a white sky, as the cad_v1 page's figures
        if m.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            a, n_ = m.tex_adr[t], m.tex_width[t] * m.tex_height[t] * m.tex_nchannel[t]
            m.tex_data[a:a + n_] = 255
    ids = [m.body(f"{f}_{b}").id for b in ("mount", "mcp_frame", "pip_frame", "tip")]
    pts = np.array([d.geom_xpos[g] for g, k in enumerate(kind) if k in ("sphere", "pad")])
    lo_, hi_ = pts.min(axis=0), pts.max(axis=0)
    look = 0.5 * (lo_ + hi_)
    span = float(np.linalg.norm(hi_ - lo_))
    ax_ = d.xaxis[m.joint(f"{f}_pip").id]
    az0 = math.degrees(math.atan2(ax_[1], ax_[0]))
    m.vis.global_.fovy = 12
    m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 1280
    m.vis.headlight.ambient[:] = (0.45, 0.45, 0.45)
    m.vis.headlight.diffuse[:] = (0.6, 0.6, 0.6)
    W, H = 620, 820
    views = (({"mesh", "tipmesh"}, az0 - 45, "CAD meshes"), ({"sphere", "pad"}, az0 - 45, "Spheres and the pad\u2019s spheres"),
             ({"mesh", "tipmesh", "sphere", "pad"}, az0 - 45, "Both"), ({"sphere", "pad"}, az0 + 135, "Spheres, dorsal side"))
    shots = []
    with mujoco.Renderer(m, H, W) as r:
        opt = mujoco.MjvOption()
        for grp in range(6):
            opt.geomgroup[grp] = 1
            opt.sitegroup[grp] = 0                         # the scene's gantry and frame sites
        for show, az, title in views:
            rgba = base.copy()
            for g, k in enumerate(kind):
                rgba[g] = col[k] if k in show else (0, 0, 0, 0)
            m.geom_rgba[:] = rgba
            cam = mujoco.MjvCamera()
            cam.lookat[:] = look
            cam.distance, cam.azimuth, cam.elevation = 0.62 * span / math.tan(math.radians(6.0)), az, -12.0
            r.update_scene(d, camera=cam, scene_option=opt)
            shots.append((r.render().copy(), title))
    m.geom_rgba[:] = base
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    rows = np.flatnonzero(np.any([(im < 245).any(axis=(1, 2)) for im, _ in shots], axis=0))
    r0, r1 = max(0, rows[0] - 16), min(H, rows[-1] + 17)
    tiles = []
    for im, title in shots:
        crop = Image.fromarray(im[r0:r1])
        bar = Image.new("RGB", (crop.width, 40), "white")
        ImageDraw.Draw(bar).text((10, 7), title, fill="black", font=font)
        t = Image.new("RGB", (crop.width, crop.height + 40), "white")
        t.paste(bar, (0, 0))
        t.paste(crop, (0, 40))
        tiles.append(t)
    w, h = tiles[0].size
    sheet = Image.new("RGB", (2 * w, 2 * h), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % 2) * w, (k // 2) * h))
    out = OUT / "20261111-finger_spheres_cad_v2.png"
    sheet.save(out)
    print("wrote", out.relative_to(ROOT), flush=True)
    return {"scene": str(FINGER_SCENE.relative_to(ROOT)), "finger": f, "q_rad": list(q),
            "spheres": int(sum(k == "sphere" for k in kind)), "pads": int(sum(k == "pad" for k in kind)),
            "meshes": int(sum(k == "mesh" for k in kind))}


def gpu_rate():
    """Rollouts per second of the GPU grid of the press (timing rows)."""
    rows = [json.loads(x) for x in (ROOT / "docs/experiments/20261029-foundational_pose_review/rows/grid2dff_v4b_sv25/"
                                    "press_d25_timing.jsonl").read_text().splitlines() if x.strip()]
    per = [(r["build_s"] + r["run_s"] + r["score_s"]) / r["worlds"] for r in rows]
    return dict(gpu=rows[0]["gpu"], worlds_per_batch=rows[0]["worlds"],
                s_per_rollout_median=round(float(np.median(per)), 4),
                batch_s_median=round(float(np.median([r["build_s"] + r["run_s"] + r["score_s"] for r in rows])), 1))


def main_film():
    """Only the three-hands film (python ... film): the rest of the figures are unchanged."""
    path = OUT / "20261110-figures.json"
    info = json.loads(path.read_text()) if path.exists() else {}
    key = "film_three_hands" + ("_track" if VARIANT == "track" else "")
    info[key] = film_three_hands(((70.0, 65.0), (70.0, 110.0), (125.0, 55.0))) if VARIANT == "track" else film_three_hands()
    info["film_three_hands"] = info[key] if VARIANT != "track" else info.get("film_three_hands")
    path.write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(info[key]), flush=True)
    return 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = [x for x in sys.argv[1:] if x != "--track"]
    path = OUT / "20261110-figures.json"
    info = json.loads(path.read_text()) if path.exists() else {}
    sfx = "_track" if VARIANT == "track" else ""
    jobs = {"ea_anatomy": ea_anatomy, "ea_snapshots": ea_snapshots, "gpu": gpu_rate,
            "tuning": tuning_track if VARIANT == "track" else tuning, "compute": compute_cost,
            "contact_force": contact_force, "pinch_film": pinch_film, "drake_hydro": drake_hydro,
            "finger_spheres": finger_spheres,
            "layout_space": layout_space, "fp_views": fp_views}
    for k, fn in jobs.items():
        if not only or k in only:
            info[k + sfx] = fn()
    if not only or "grasp_candidates" in only:
        a_xy = (70.0, 65.0) if VARIANT == "track" else tuple(info["layout_space"]["A"]["layout"])
        info["grasp_candidates" + sfx] = grasp_candidates(a_xy)
    path.write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(info, default=float)[:900], flush=True)
    return 0


if __name__ == "__main__":
    if "--track" in sys.argv:
        use_track()
    sys.exit(main_film() if [x for x in sys.argv[1:] if x != "--track"] == ["film"] else main())
