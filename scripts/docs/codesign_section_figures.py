#!/usr/bin/env python3
"""Slide figures for the three level sections of the hand-design talk (owner, 2026-11-10 evening: the morphology,
control and dynamics sections should have the co-design section's "clarity, visual elegance, and incrementalism"). Each
figure makes one point at slide size, in the typeface (Nimbus Sans, the drawings' Helvetica) and colours of the TikZ
drawings. Writes docs/experiments/20261110-codesign_formulation/media/:

  20261110-slide_landscape.png     every hand on the lattice, press on the 25 mm cylinder, the best one starred
  20261110-slide_tasks.png         the four tasks' landscapes, each as a share of its own best, best hands starred
  20261110-slide_one_hand.png      the same four landscapes with the one hand that reaches 0.84 of every task's best
  20261110-slide_three_hands.png   and with the three hands that reach 0.95
  20261110-slide_size_trend.png    the best thumb-to-pair distance against object diameter, with Borras and Dollar's
                                   slope of 1 (the workspace rule) in purple

Data: the 625-hand grids grid2dff_v4b_sv25 (SCS0009 plant, 2.5 MPa pads, best of 16 grasps, mean of seeds 0-2), the
task-conflict study's validated covers (seeds 20-24, docs/experiments/20261103-hom_task_conflict/20261104-conflict_sv25.json)
and the live evolutionary search at 9-10 sizes per task on the same plant (rows/ea_live_sv25/summary.json).

    .venv/bin/python scripts/docs/codesign_section_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401

OUT = ROOT / "docs/experiments/20261110-codesign_formulation/media"
FPR = ROOT / "docs/experiments/20261029-foundational_pose_review/rows"
CONFLICT = ROOT / "docs/experiments/20261103-hom_task_conflict/20261104-conflict_sv25.json"
INK, GREY, LINE, NEW = "#1f2937", "#9ca3af", "#d1d5db", "#7c3aed"
STAR = "#facc15"
CMAP = "YlGnBu"
TASKS = [("twist", 40.0, "Twist", "40 mm sphere", "#b91c1c"), ("roll", 40.0, "Roll", "40 mm sphere", "#0e7490"),
         ("pinpush", 25.0, "Pin-push", "25 mm cylinder", "#a16207"), ("press", 25.0, "Press", "25 mm cylinder", "#4d7c0f")]
COVER_MARKS = ["o", "s", "D"]
plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 17, "axes.titlesize": 19, "axes.labelsize": 17,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 15, "axes.edgecolor": GREY,
                     "axes.titlepad": 10, "mathtext.fontset": "cm"})


def grid(task, size):
    import hom_landscape as L
    L.GRID_SRC, L.GRID_ROWS = "grid2dff", L.ROWS / "grid2dff_v4b_sv25"
    xs, ys, M = L.grid_maps(task, size)
    return np.asarray(xs, float), np.asarray(ys, float), np.asarray(M["best"], float)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    return OUT / name


def plane(ax, xs, ys, B, vmax, ylabel=True):
    ext = [xs[0] - 2.5, xs[-1] + 2.5, ys[0] - 2.5, ys[-1] + 2.5]
    im = ax.imshow(np.ma.masked_invalid(B), origin="lower", extent=ext, cmap=CMAP, vmin=0, vmax=vmax, aspect="equal")
    ax.set_facecolor("#f3f4f6")
    ax.set_xlabel(r"thumb to pair $s_1$ (mm)")
    if ylabel:
        ax.set_ylabel(r"index to middle $s_2$ (mm)")
    ax.set_xticks([50, 100, 150])
    ax.set_yticks([50, 100, 150])
    for s in ax.spines.values():
        s.set_color(LINE)
    return im


def star(ax, x, y, size=520):
    ax.scatter([x], [y], marker="*", s=size, c=STAR, edgecolors=INK, linewidths=1.6, zorder=5)


def landscape():
    """Every hand on the lattice for one task and size; the best starred."""
    xs, ys, B = grid("press", 25.0)
    j, i = np.unravel_index(np.nanargmax(B), B.shape)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    im = plane(ax, xs, ys, B, 0.85)
    star(ax, xs[i], ys[j])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("press score")
    cb.outline.set_edgecolor(LINE)
    ax.set_title("Press, 25 mm cylinder: 625 hands", loc="left")
    return save(fig, "20261110-slide_landscape.png"), dict(best=[float(xs[i]), float(ys[j])], score=float(B[j, i]),
                                                            reached=int(np.isfinite(B).sum()))


def tasks(marks=None, name="20261110-slide_tasks.png"):
    """The four tasks' landscapes, each as a share of its own best; marks = [(xy, tasks served)] adds the chosen hands."""
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.6), gridspec_kw=dict(wspace=0.08))
    info = {}
    for k, (ax, (task, size, name_, obj, _c)) in enumerate(zip(axes, TASKS)):
        xs, ys, B = grid(task, size)
        top = np.nanmax(B)
        im = plane(ax, xs, ys, B / top, 1.0, ylabel=k == 0)
        if k:
            ax.set_yticklabels([])
        j, i = np.unravel_index(np.nanargmax(B), B.shape)
        if marks is None:
            star(ax, xs[i], ys[j])
        else:
            for (xy, served), mk in zip(marks, COVER_MARKS):   # each panel shows the hand that serves its task
                if task in served:
                    ax.scatter([xy[0]], [xy[1]], marker=mk, s=260, facecolors=NEW, edgecolors="white", linewidths=2.0,
                               zorder=5)
        ax.set_title(f"{name_}, {obj}", loc="left", fontsize=17)
        info[task] = dict(best=[float(xs[i]), float(ys[j])], score=round(float(top), 3))
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.012)
    cb.set_label("share of the task's best")
    cb.outline.set_edgecolor(LINE)
    return save(fig, name), info


def covers():
    """The hands of the task-conflict study's validated covers: one hand (0.84 of every task's best), three (0.95)."""
    d = json.loads(CONFLICT.read_text())["val"]["coverage"]["val"]
    lv = {round(x["alpha"], 2): x for x in d["levels"]}
    one = [(c["x"], c["tasks"]) for c in lv[0.84]["cover"]]
    three = [(c["x"], c["tasks"]) for c in lv[0.95]["cover"]]
    p1, _ = tasks(one, "20261110-slide_one_hand.png")
    p3, _ = tasks(three, "20261110-slide_three_hands.png")
    return dict(one=one, three=three, alpha_star=d["alpha_star"])


def size_trend():
    """The best thumb-to-pair distance (centroid of the hands within 0.05 of each size's best, with their range)
    against object diameter, four tasks, least-squares lines; Borras and Dollar's slope of 1 through each panel's
    centre in purple."""
    S = json.loads((FPR / "ea_live_sv25/summary.json").read_text())

    def series(task):
        pts = sorted((v["d_mm"], v["ea"]) for v in S.values() if isinstance(v, dict) and v.get("task") == task
                     and v.get("ea"))
        d = np.array([p[0] for p in pts])
        c = np.array([p[1]["near_centroid"][0] for p in pts])
        lo = np.array([p[1]["near_lo"][0] for p in pts])
        hi = np.array([p[1]["near_hi"][0] for p in pts])
        k, b = np.polyfit(d, c, 1)
        return d, c, lo, hi, k, b

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True, gridspec_kw=dict(wspace=0.06))
    slopes = {}
    for ax, title, group in ((axes[0], "Spheres", TASKS[:2]), (axes[1], "Cylinders", TASKS[2:])):
        cen = []
        for n, (task, _s, name_, _o, col) in enumerate(group):
            d, c, lo, hi, k, b = series(task)
            slopes[task] = round(float(k), 2)
            off = (-0.5, 0.5)[n]
            ax.errorbar(d + off, c, yerr=[c - lo, hi - c], fmt="none", ecolor=col, alpha=0.35, lw=1.6, capsize=0)
            ax.plot(d + off, c, "o", color=col, ms=9, label=f"{name_}: {k:.2f} mm per mm")
            xx = np.linspace(d.min() - 2, d.max() + 2, 50)
            ax.plot(xx, k * xx + b, "-", color=col, lw=2.4)
            cen.append((d.mean(), c.mean()))
        x0, y0 = np.mean(cen, axis=0)
        lo_, hi_ = ax.get_xlim()
        xx = np.linspace(lo_ + 1, hi_ - 1, 20)
        ax.plot(xx, y0 + (xx - x0), ls=(0, (5, 4)), color=NEW, lw=2.4, label="slope 1: Borràs and Dollar")
        ax.set_title(title, loc="left")
        ax.set_xlabel("object diameter $d$ (mm)")
        ax.grid(color="#eef0f3", lw=0.8)
        for s in ax.spines.values():
            s.set_color(LINE)
        ax.legend(loc="upper left", frameon=False, handlelength=1.8)
    axes[0].set_ylabel(r"best thumb to pair $s_1^\star$ (mm)")
    axes[0].set_ylim(30, 180)
    return save(fig, "20261110-slide_size_trend.png"), slopes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = {}
    _, info["landscape"] = landscape()
    _, info["tasks"] = tasks()
    info["covers"] = covers()
    _, info["slopes"] = size_trend()
    path = OUT / "20261110-figures.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old["section_figures"] = info
    path.write_text(json.dumps(old, indent=1, default=float))
    print(json.dumps(info, default=float)[:1200], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
