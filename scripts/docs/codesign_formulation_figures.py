#!/usr/bin/env python3
"""Figures of the co-design formulation for the hand-design talk (owner, 2026-11-10: "beautiful, clear, diagrams and
visual content that is communicable for my audience"; the SR2 part "hierarchical, rather than cyclical"; HOM diagrams
"ideally even simpler/clearer"). Writes docs/experiments/20261110-codesign_formulation/media/:

  20261110-bilevel_{all,codesign,outer,inner,sim}.png   the nested problem, whole and with one section highlighted
  20261110-ea_snapshots.png                             Yang et al.'s EA on the stored 25 mm press landscape (SCS0009)
  20261110-hom_loop.png                                 one control period of HOM, four boxes

    HOM_ROWS_TAG=_sv25 .venv/bin/python scripts/docs/codesign_formulation_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

OUT = ROOT / "docs/experiments/20261110-codesign_formulation/media"
INK, GREY, PALE = "#1f2937", "#9ca3af", "#f3f4f6"
COL = {"outer": "#2563eb", "inner": "#059669", "sim": "#d97706", "codesign": "#7c3aed"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "mathtext.fontset": "cm"})


def box(ax, x, y, w, h, title, lines, color, on):
    fc, ec, tc = (color, color, "white") if on else (PALE, GREY, INK)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec=ec, lw=2.5))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=17.5, weight="bold", color=tc)
    for k, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.85 - 0.48 * k, s, ha="center", va="top", fontsize=15.5, color=tc)


def arrow(ax, a, b, text="", color=INK, rad=0.0, dy=0.12):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=22, lw=2.2, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, text, ha="center", va="bottom", fontsize=14, color=color)


def bilevel(active):
    """The nested problem as three boxes: morphology optimization around multi-hand control around the simulator."""
    fig, ax = plt.subplots(figsize=(16, 5.9), dpi=110)
    ax.set_xlim(0, 16)
    ax.set_ylim(0.4, 6.3)
    ax.axis("off")
    on_all = active == "all"
    hue = COL["codesign"] if active in ("codesign", "all") else INK
    ax.text(8, 5.45, r"$m^\star=\arg\max_{m}\ \mathbb{E}_{T,d}\left[\ \max_{g,\,\phi}\ J(m,g,\phi;\,T,d)\ \right]$",
            ha="center", va="center", fontsize=27, color=hue)
    box(ax, 0.2, 1.5, 4.5, 2.9, "Morphology optimization", ["outer loop: choose the hand $m$",
                                                            "6 mount coordinates", "Yang et al.'s EA"],
        COL["outer"], on_all or active in ("outer", "codesign"))
    box(ax, 5.75, 1.5, 4.5, 2.9, "Multi-hand control", ["inner loop: run the hand $m$", "grasp $g$, HOM parameters $\\phi$",
                                                       "same controller, every hand"],
        COL["inner"], on_all or active in ("inner", "codesign"))
    box(ax, 11.3, 1.5, 4.5, 2.9, "Simulation", ["score the rollout: $J$", "contact $c$, servo $p$, sensing $s$",
                                                "calibrated on hardware"],
        COL["sim"], on_all or active in ("sim", "codesign"))
    arrow(ax, (4.7, 3.4), (5.75, 3.4), "$m$")
    arrow(ax, (5.75, 2.4), (4.7, 2.4), "$J^\\star(m)$", dy=-0.5)
    arrow(ax, (10.25, 3.4), (11.3, 3.4), "$g,\\phi$")
    arrow(ax, (11.3, 2.4), (10.25, 2.4), "$J$", dy=-0.5)
    ax.text(13.55, 0.85, "tasks $T$ and object sizes $d$", ha="center", va="center", fontsize=15, color=INK)
    arrow(ax, (13.55, 1.05), (13.55, 1.5))
    ax.text(5.3, 0.85, "the hierarchy of the talk: outer loop, inner loop, evaluator", ha="center", va="center",
            fontsize=14, color=GREY, style="italic")
    fig.savefig(OUT / f"20261110-bilevel_{active}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ea_snapshots():
    """Where the EA (population 16, delta 30 mm) has evaluated after 16, 48, 96 and 192 layouts, on the stored landscape
    of the press on the 25 mm cylinder (SCS0009 plant, 2.5 MPa, earlier press score), and the best score found against
    random search, median over 200 runs."""
    import hom_landscape as L
    L.GRID_SRC, L.GRID_ROWS = "grid2dff", L.ROWS / "grid2dff_v4b_sv25"
    task, size = "press", 25.0
    xs, ys, M = L.grid_maps(task, size)
    look = L.ea_landscape(task, size)[0]["best"]
    best_all = max(f for f, _n in look.values())
    rng = np.random.default_rng(3)
    hist, seen = L.ea_search(look, rng, xs, ys, pop=16, budget=None)
    order = [k for _c, k in hist]
    fig = plt.figure(figsize=(18, 5.6), dpi=110)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.25], wspace=0.28)
    ext = [xs[0] - 2.5, xs[-1] + 2.5, ys[0] - 2.5, ys[-1] + 2.5]
    for j, n in enumerate((16, 48, 96, 192)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(M["best"], origin="lower", extent=ext, cmap="YlGnBu", vmin=0, vmax=0.85, aspect="auto")
        pts = np.array(order[:n], float)
        ax.scatter(pts[:-16, 0], pts[:-16, 1], s=22, c="#fdba74", edgecolors="none") if len(pts) > 16 else None
        ax.scatter(pts[-16:, 0], pts[-16:, 1], s=40, c="#ea580c", edgecolors="white", linewidths=0.7)
        kb = max(order[:n], key=lambda k: seen[k])
        ax.scatter([kb[0]], [kb[1]], s=260, marker="*", c="#f59e0b", edgecolors=INK, linewidths=0.8, zorder=5)
        ax.set_title(f"after {n} layouts\nbest {seen[kb]:.2f}", fontsize=15)
        ax.set_xlabel("thumb to pair (mm)", fontsize=13)
        if j == 0:
            ax.set_ylabel("index to middle (mm)", fontsize=13)
        ax.tick_params(labelsize=11)
        im_ = ax.images[0]
    cb = fig.colorbar(im_, ax=fig.axes[:4], orientation="horizontal", fraction=0.045, pad=0.2, aspect=60)
    cb.set_label("press score of the layout (best grasp)", fontsize=12)
    ax = fig.add_subplot(gs[0, 4])
    budgets = np.arange(1, 400)
    for kind, col, lab in ((False, COL["outer"], "EA, population 16"), (True, GREY, "random search")):
        curves = []
        for r in range(200):
            h, s_ = L.ea_search(look, np.random.default_rng(100 + r), xs, ys, pop=16, random=kind)
            b = np.maximum.accumulate([s_[k] for _c, k in h])
            curves.append(np.interp(budgets, np.arange(1, len(b) + 1), b, right=b[-1]))
        ax.plot(budgets, np.median(curves, 0), color=col, lw=3, label=lab)
        ax.fill_between(budgets, *np.percentile(curves, [25, 75], 0), color=col, alpha=0.18)
    ax.axhline(best_all, color=INK, ls=":", lw=1.5)
    ax.text(budgets[-1], best_all + 0.01, "best on the grid", ha="right", va="bottom", fontsize=12)
    ax.set_xlabel("layouts evaluated", fontsize=13)
    ax.set_ylabel("best score found", fontsize=13)
    ax.set_ylim(0.3, best_all + 0.08)
    ax.legend(fontsize=12, loc="lower right", frameon=False)
    ax.set_title("median of 200 runs, quartiles shaded", fontsize=14)
    fig.savefig(OUT / "20261110-ea_snapshots.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dict(task=task, size=size, layouts=len(look), best=best_all, run_best=max(seen.values()))


def hom_loop():
    """One HOM control period for one finger, four boxes and the measured state fed back."""
    fig, ax = plt.subplots(figsize=(16, 6.4), dpi=110)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    g = COL["inner"]
    items = [("Object command", ["how the object", "should move: $V$"]),
             ("Contact references", ["per fingertip: stick,", "roll or slide: $v_{ref}$"]),
             ("Least squares", ["joint speeds $u$ that best", "make $v_{ref}$ happen"]),
             ("Servo targets", ["integrate $u$, then press", "with force $F$: $+J^{\\top}Fn/k_p$"])]
    w, h, y0 = 3.35, 2.1, 3.2
    for k, (t, ls) in enumerate(items):
        x = 0.35 + k * 3.95
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=g if k == 2 else "white", ec=g, lw=2.5))
        tc = "white" if k == 2 else INK
        ax.text(x + w / 2, y0 + h - 0.22, t, ha="center", va="top", fontsize=17, weight="bold", color=tc)
        for j, s in enumerate(ls):
            ax.text(x + w / 2, y0 + h - 0.85 - 0.42 * j, s, ha="center", va="top", fontsize=15, color=tc)
        if k < 3:
            arrow(ax, (x + w, y0 + h / 2), (x + 3.95, y0 + h / 2), color=g)
    ax.add_patch(FancyBboxPatch((8.2, 0.35), 7.4, 1.15, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=PALE, ec=GREY, lw=2))
    ax.text(11.9, 0.93, "hand and object: joint angles, object pose, pad forces", ha="center", va="center",
            fontsize=15, color=INK)
    arrow(ax, (13.88, 3.15), (13.88, 1.55), color=INK)
    arrow(ax, (9.93, 1.55), (9.93, 3.15), color=INK)
    ax.text(9.75, 2.35, "measured every 10 ms", ha="right", va="center", fontsize=14, color=INK)
    ax.text(8.0, 6.1, "The same law runs every hand: the hand enters only through the Jacobian $A(m,g)$",
            ha="center", va="center", fontsize=16, color=INK)
    ax.text(8.0, 5.65, "tuned per hand, if at all: forces $F$ and mode weights $w$, together $\\phi$", ha="center",
            va="center", fontsize=14.5, color=GREY, style="italic")
    fig.savefig(OUT / "20261110-hom_loop.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for a in ("all", "codesign", "outer", "inner", "sim"):
        bilevel(a)
    hom_loop()
    print(ea_snapshots(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
