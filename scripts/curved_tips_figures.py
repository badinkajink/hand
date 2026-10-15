#!/usr/bin/env python3
"""Figures of the curved-tip page (docs/experiments/20261015-curved_tips/), drawn from the study's summaries and shapes,
in the style of scripts/tip_codesign_figures.py (9.5 pt at 150 dpi on 7.2 in wide figures, at least 13 px when a figure
is shown 828 px wide; pads red, Drake blue, the law black).

  shapes   media/fig_shapes.png: each tip's profile along the finger (y = 0) and its section across the finger at the
           contact point (z = 0), design surfaces, with the magnet slot of the print files
  bed      media/fig_bed.png: twist-onset arm at 1 N (law, pads, Drake), spin arm at 1 rad/s over the law, and the spin
           arm's growth from 0.3 to 3 rad/s, per tip (bed/summary.json)
  traj     media/fig_traj.png: RMS tool-angle and normal-force differences between arms per tip, with the spread between
           placements (traj/summary.json)
  screen   media/fig_screen.png: the stage score per condition, J2 and the Drake agreement per tip (screen/summary.json)
  stage3   media/fig_stage3.png: J3 and the paired difference against the as-built block (final/summary.json)
  cost     media/fig_cost.png: CPU step at the three-finger grip and GPU world-step per tip (cost/)

    ~/.claude/bin/resguard.sh run --mem 2G -- .venv/bin/python scripts/curved_tips_figures.py shapes bed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import curved_tips_common as C  # noqa: E402

D = C.D
MEDIA = D / "media"
INK, INK2, GRID = "#111111", "#555555", "#d9d9d9"
PADS, DRAKE, THIRD = "#b2182b", "#2166ac", "#e08214"    # validated as a set (dataviz validator, light, all pairs)
ORDER = [C.ASBUILT, C.FINALIST] + [k + C.PRINT for k in C.CAPSULES + C.ELLIPSOIDS]


def short(k):
    """Axis label of a tip."""
    import re
    if k == C.ASBUILT:
        return "As-built\nblock"
    if k == C.FINALIST:
        return "Block\n(3, 6)"
    m = re.match(r"tcd_cap([0-9.]+)", k)
    if m:
        return f"Capsule\nr {m[1]}"
    m = re.match(r"tcd_ell([0-9.]+)x([0-9.]+)x([0-9.]+)(?:x([0-9.]+))?", k)
    if m:
        if m[1] == m[2] == m[3]:
            return f"Hemisphere\nR {m[1]}"
        return f"Ellipsoid\n{m[2]} · {m[3]}" + (f"/{m[4]}" if m[4] else "")
    m = re.match(r"tcd_sph([0-9.]+)", k)
    if m:
        return f"Whole sphere\nR {m[1]}"
    return k


def title2(k):
    """Two-line panel title of a design key."""
    import re
    if k.startswith("tcd_rs3_re3"):
        return "As-built block\nfillets 3 and 3 mm"
    if k.startswith("tcd_rs3_re6"):
        return "Block finalist\nfillets 3 and 6 mm"
    m = re.match(r"tcd_cap([0-9.]+)", k)
    if m:
        return f"Capsule\nradius {m[1]} mm"
    m = re.match(r"tcd_ell([0-9.]+)x([0-9.]+)x([0-9.]+)(?:x([0-9.]+))?", k)
    if m:
        return f"Half-ellipsoid\n{m[1]} × {m[2]} × {m[3]}" + (f"/{m[4]}" if m[4] else "") + " mm"
    return k


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans", "axes.edgecolor": "#888888",
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.04})
    MEDIA.mkdir(parents=True, exist_ok=True)
    return plt


def _outline(sh, plane, n=720):
    """Closed outline of a design shape in a plane through the contact point: 'xz' (y = 0, the profile along the
    finger) or 'xy' (z = 0, the section across it), by radial bisection from an interior point (mm)."""
    import tip_codesign_geometry as TG
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if plane == "xz":
        o = np.array([TG.C0[0], 0.0, TG.C0[2]])
        D2 = np.column_stack([np.cos(th), np.zeros(n), np.sin(th)])
    else:
        o = np.array([TG.C0[0], 0.0, 0.0])
        D2 = np.column_stack([np.cos(th), np.sin(th), np.zeros(n)])
    lo, hi = np.zeros(n), np.full(n, 0.03)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        ins = sh.inside_design(o + mid[:, None] * D2)
        lo, hi = np.where(ins, mid, lo), np.where(ins, hi, mid)
    P = (o + lo[:, None] * D2) * 1e3
    return P


def fig_shapes():
    """Per tip, the profile along the finger (z horizontal, distal end left; x up, palmar on top) and the section
    across the finger at the contact point (y horizontal), drawn side by side on one x axis."""
    plt = setup()
    import tip_codesign_geometry as TG
    tips = [C.ASBUILT.rsplit("_p", 1)[0], C.FINALIST.rsplit("_p", 1)[0]] + C.CAPSULES + C.ELLIPSOIDS
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.0), dpi=150)
    gap = 4.0
    walls = {}
    if (D / "print/walls.jsonl").exists():
        for line in open(D / "print/walls.jsonl"):
            r = json.loads(line)
            walls[r["design"]] = r
    for ax, k in zip(axs.ravel(), tips):
        sh = TG.shape(k)
        Pz = _outline(sh, "xz")
        Py = _outline(sh, "xy")
        zoff = -TG.Z_END * 1e3                                  # profile from 0 (distal end) to 21.41 (mount face)
        ax.fill(Pz[:, 2] + zoff, Pz[:, 0], color="#e6c7a8", ec=INK, lw=0.8)
        yoff = TG.LENGTH * 1e3 + gap + TG.WIDTH / 2 * 1e3
        ax.fill(Py[:, 1] + yoff, Py[:, 0], color="#e6c7a8", ec=INK, lw=0.8)
        # the magnet slot of the print files (curved tips) or the face-up pocket (blocks): the disc 5 mm under the
        # palmar reach at the contact point, 4.76 x 1.59 mm
        xm = (TG.X_FACE * 1e3 - 5.0 - 1.59, TG.X_FACE * 1e3 - 5.0)
        ax.add_patch(plt.Rectangle((zoff - 2.38, xm[0]), 4.76, 1.59, color="#555555", lw=0))
        ax.add_patch(plt.Rectangle((yoff - 2.38, xm[0]), 4.76, 1.59, color="#555555", lw=0))
        ax.plot([zoff], [TG.X_FACE * 1e3], "v", color=PADS, ms=4)
        ax.set_aspect("equal")
        ax.set_xlim(-1, yoff + 9)
        ax.set_ylim(TG.X_BACK * 1e3 - 1, TG.X_FACE * 1e3 + 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title2(k), fontsize=9.0)
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    out = MEDIA / "fig_shapes.png"
    fig.savefig(out)
    print("wrote", out.relative_to(ROOT))


BED_ORDER = [C.ASBUILT, C.FINALIST] + [k + C.PRINT for k in C.CAPSULES + C.ELLIPSOIDS] + [C.LITERAL, C.SPHERE_ENV, C.SPHERE]


def fig_bed():
    """Three panels over the tips: (a) twist-onset arm at 1 N, law, pads and Drake (mm); (b) spin arm at 1 rad/s over
    the law, pads and Drake; (c) the spin arm at 3 rad/s over the arm at 0.3 rad/s, pads and Drake."""
    plt = setup()
    S = json.loads((D / "bed/summary.json").read_text())["tips"]
    tips = [k for k in BED_ORDER if k in S]
    x = np.arange(len(tips))
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.4), dpi=150, sharex=True)
    g = lambda k, *path: _get(S[k], *path)  # noqa: E731
    law = np.array([g(k, "law", "onset_arm_mm", "1") for k in tips], float)
    ax = axs[0]
    ax.plot(x, law, "_", color=INK, ms=14, mew=1.6, label="Law (Drake's field, quasi-static)")
    ax.plot(x - 0.12, [g(k, "pads", "onset_arm_1N_mm") for k in tips], "o", color=PADS, ms=4.5, label="1 mm pads, builder v2")
    ax.plot(x + 0.12, [g(k, "drake", "onset_arm_1N_mm") for k in tips], "s", color=DRAKE, ms=4.2, label="Drake hydroelastic")
    ax.set_ylabel("Twist-onset arm\nat 1 N (mm)")
    ax.set_ylim(0, 3.1)
    ax.legend(fontsize=8.3, loc="upper right", ncol=1)
    ax = axs[1]
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.plot(x - 0.12, [g(k, "ratios", "pads_spin_1_over_law") for k in tips], "o", color=PADS, ms=4.5)
    ax.plot(x + 0.12, [g(k, "ratios", "drake_spin_1_over_law") for k in tips], "s", color=DRAKE, ms=4.2)
    ax.set_ylabel("Spin arm at 1 rad/s\nover the law")
    ax.set_ylim(0.85, 1.3)
    ax = axs[2]
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.plot(x - 0.12, [g(k, "ratios", "pads_spin_3_over_0.3") for k in tips], "o", color=PADS, ms=4.5)
    ax.plot(x + 0.12, [g(k, "ratios", "drake_spin_3_over_0.3") for k in tips], "s", color=DRAKE, ms=4.2)
    ax.set_ylabel("Spin arm at 3 rad/s\nover 0.3 rad/s")
    ax.set_ylim(0.9, 1.7)
    ax.set_xticks(x)
    ax.set_xticklabels([short(k).replace("\n", " ") for k in tips], fontsize=8.5, rotation=32, ha="right",
                       rotation_mode="anchor")
    for a in axs:
        a.axvline(1.5, color=GRID, lw=1.2)
        a.axvline(len(BED_ORDER) - 3.5, color=GRID, lw=1.2)
    fig.tight_layout(h_pad=0.6)
    out = MEDIA / "fig_bed.png"
    fig.savefig(out)
    print("wrote", out.relative_to(ROOT))


def _get(d, *path):
    for p in path:
        if d is None:
            return np.nan
        d = d.get(p) if isinstance(d, dict) else None
    return np.nan if d is None else d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figs", nargs="+", choices=["shapes", "bed", "traj", "screen", "stage3", "cost"])
    a = ap.parse_args()
    for f in a.figs:
        globals()[f"fig_{f}"]()


if __name__ == "__main__":
    main()
