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
import sys
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


def grid():
    import hom_landscape as L
    L.GRID_SRC, L.GRID_ROWS = "grid2dff", L.ROWS / "grid2dff_v4b_sv25"
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
    ax.set_ylim(0.45, best_all + 0.06)
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
        col = INNER if best else ("#b91c1c" if g_sc[j] < 0.05 else INK)
        ax.set_title(f"{g_sc[j]:.2f}" + ("  pose solve" if j == 0 else ""), fontsize=17, color=col,
                     weight="bold" if best else "normal")
        if best:
            ax.add_patch(Rectangle((0, 0), img.shape[1] - 81, img.shape[0] - 1, fill=False, ec=INNER, lw=6))
    fig.suptitle("Sixteen grasps of one hand, best to worst: press score of each (red: the cylinder drops)",
                 fontsize=19, weight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(hspace=0.32)
    save(fig, "20261110-grasp_candidates_wide.png")
    return dict(layout=list(xy), best_grasp=gbest, scores={str(k): round(v, 3) for k, v in sorted(g_sc.items())})


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
        for level, c, lab in (("grasp", "#6ee7b7", "S2: tune the grasp"), ("full", INNER, "S3: tune grasp, forces, modes")):
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


def gpu_rate():
    """Rollouts per second of the GPU grid of the press (timing rows)."""
    rows = [json.loads(x) for x in (ROOT / "docs/experiments/20261029-foundational_pose_review/rows/grid2dff_v4b_sv25/"
                                    "press_d25_timing.jsonl").read_text().splitlines() if x.strip()]
    per = [(r["build_s"] + r["run_s"] + r["score_s"]) / r["worlds"] for r in rows]
    return dict(gpu=rows[0]["gpu"], worlds_per_batch=rows[0]["worlds"],
                s_per_rollout_median=round(float(np.median(per)), 4),
                batch_s_median=round(float(np.median([r["build_s"] + r["run_s"] + r["score_s"] for r in rows])), 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:]
    path = OUT / "20261110-figures.json"
    info = json.loads(path.read_text()) if path.exists() else {}
    jobs = {"ea_anatomy": ea_anatomy, "ea_snapshots": ea_snapshots, "gpu": gpu_rate, "tuning": tuning,
            "layout_space": layout_space}
    for k, fn in jobs.items():
        if not only or k in only:
            info[k] = fn()
    if not only or "grasp_candidates" in only:
        info["grasp_candidates"] = grasp_candidates(tuple(info["layout_space"]["A"]["layout"]))
    path.write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(info, default=float)[:900], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
