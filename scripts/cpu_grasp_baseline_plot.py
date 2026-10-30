"""Heatmaps of a grasp baseline over the cylinder grid (diameter x mass).

    .venv-cpu/bin/python scripts/cpu_grasp_baseline_plot.py docs/experiments/<date>-real_v3-p0-baseline/p0_trials.json

Left: contact retention -- fraction of the reorientation (turn + hold) with >= 2 fingers on the
cylinder, mean over trials; each cell also carries how many trials held to the end. Right: the
fraction with all three fingers on. One sequential hue, one shared 0-100% scale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# reference sequential ramp (dataviz palette.md, blue 100 -> 700)
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE, INK, INK2 = "#fcfcfb", "#1f1f1e", "#6b6a66"


def main(path: Path, title: str = "P0 open-loop baseline") -> Path:
    data = json.loads(path.read_text())
    trials = data["trials"]
    D = sorted({t["diameter_mm"] for t in trials})
    M = sorted({t["mass_g"] for t in trials})
    ret = np.full((len(D), len(M)), np.nan)
    all3 = np.full_like(ret, np.nan)
    held = np.zeros_like(ret, dtype=int)
    n = np.zeros_like(held)
    for i, dm in enumerate(D):
        for j, mg in enumerate(M):
            rs = [t for t in trials if t["diameter_mm"] == dm and t["mass_g"] == mg]
            ret[i, j] = np.mean([t["retention"] for t in rs])
            all3[i, j] = np.mean([t["all3"] for t in rs])
            held[i, j] = sum(t["held"] for t in rs)
            n[i, j] = len(rs)
    cmap = LinearSegmentedColormap.from_list("seq_blue", RAMP)
    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), facecolor=SURFACE,
                             gridspec_kw=dict(wspace=0.12, left=0.06, right=0.9, top=0.8, bottom=0.11))
    panels = [(ret, "Contact retention  (≥ 2 fingers on the cylinder)", True),
              (all3, "All three fingers on the cylinder", False)]
    for ax, (Z, sub, show_held) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        im = ax.imshow(Z * 100, cmap=cmap, vmin=0, vmax=100, origin="lower", aspect="auto")
        for i in range(len(D)):                     # 2 px surface gaps between cells
            ax.axhline(i - 0.5, color=SURFACE, lw=2)
        for j in range(len(M)):
            ax.axvline(j - 0.5, color=SURFACE, lw=2)
        for i in range(len(D)):
            for j in range(len(M)):
                v = Z[i, j] * 100
                col = "white" if v >= 55 else INK
                ax.text(j, i + (0.13 if show_held else 0), f"{v:.0f}%", ha="center", va="center",
                        fontsize=11.5, fontweight="semibold", color=col)
                if show_held:
                    ax.text(j, i - 0.24, f"{held[i, j]}/{n[i, j]} held", ha="center", va="center",
                            fontsize=8.5, color=col, alpha=0.9)
        ax.set_xticks(range(len(M)), [f"{m:g}" for m in M])
        ax.set_yticks(range(len(D)), [f"{d:g}" for d in D])
        ax.set_xlabel("Cylinder mass (g)")
        if ax is axes[0]:
            ax.set_ylabel("Cylinder diameter (mm)")
        ax.set_title(sub, loc="left", fontsize=12.5, color=INK, pad=10)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    cax = fig.add_axes([0.915, 0.11, 0.012, 0.69])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("% of the reorientation phase (turn + hold)", color=INK2)
    cb.ax.tick_params(length=0, colors=INK2)
    allt = trials
    fig.text(0.06, 0.93, title, fontsize=17, fontweight="bold", color=INK)
    fig.text(0.06, 0.875,
             f"D1's shipped plan ({data['plan']}) on the real_v3 hand, MorphIt collision model.  "
             f"{len(allt)} trials, {int(n.max())} per cell with ±2 mm / ±2° staging error.  "
             f"Overall retention {np.mean([t['retention'] for t in allt]) * 100:.1f}%,  "
             f"held to the end {np.mean([t['held'] for t in allt]) * 100:.1f}%.",
             fontsize=10.5, color=INK2)
    out = path.with_name(path.stem.replace("_trials", "") + "_retention.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    return out


if __name__ == "__main__":
    print(main(Path(sys.argv[1]), *(sys.argv[2:3])))
