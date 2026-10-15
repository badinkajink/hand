#!/usr/bin/env python3
"""Figures of the fingertip co-design page (docs/experiments/20261014-tip_codesign/), drawn from the study's summaries.
Text is 9.5 pt at 150 dpi on 7.2 in wide figures, at least 13 px when a figure is shown 828 px wide.

  deterioration   media/fig_deterioration.png: pads (1 mm spheres) and Drake against the Drake-field law over the flat
                  family's fillet radius r and the 10.55 mm sphere: twist-onset arm, sliding arm of the driven spin at
                  1 rad/s, braking normal force at 45 deg (deterioration/summary.json)
  rate            media/fig_rate.png: the sliding arm of the driven spin against spin rate on the r 2.7 mm block: pads as
                  built, pads with the friction-row fix (pf0.1:ir1000), Drake, the law and the quasi-static pads
  screen          media/fig_screen.png: J1 and the pads' spin-arm ratio over the r_side x r_end grid
                  (screen/summary_stage1.json, screen/fidelity/law.jsonl)
  stage2          media/fig_stage2.png: the stage score per condition and the agreement with Drake per stage-2 tip
  stage3          media/fig_stage3.png: J3 and the paired difference against the as-built tip with its 90 % interval

    .venv/bin/python scripts/tip_codesign_figures.py deterioration rate
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs/experiments/20261014-tip_codesign"
MEDIA = D / "media"
INK, INK2, GRID = "#111111", "#555555", "#d9d9d9"
PADS, DRAKE, FIX = "#b2182b", "#2166ac", "#e08214"


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


def fig_deterioration():
    plt = setup()
    S = json.loads((D / "deterioration/summary.json").read_text())["members"]
    rs = sorted((float(k), k) for k in S if k.replace(".", "").isdigit())
    r = np.array([x for x, _ in rs])
    get = lambda var, f: np.array([S[k][var].get(f) if isinstance(S[k][var], dict) else None for _, k in rs], float)  # noqa: E731
    sph = S["sphere"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), dpi=150)
    xs = 9.0                                            # the sphere's column, right of the family
    panels = [("Twist-onset arm / law", "onset_over_law", None), ("Spin arm at 1 rad/s / law", "spin_over_law", "1"),
              ("Brake force at 45° / Drake", "N45_over_drake", None)]
    for ax, (title, var, sub) in zip(axs, panels):
        if sub:
            yp = np.array([S[k]["base"][var][sub] for _, k in rs])
            yd = np.array([S[k]["drake"][var].get(sub, np.nan) for _, k in rs])
            sp, sd = sph["base"][var][sub], sph["drake"][var].get(sub)
        elif var == "N45_over_drake":
            yp = np.array([S[k]["base"][var] for _, k in rs])
            yd = None
            sp, sd = sph["base"][var], None
        else:
            yp = np.array([S[k]["base"][var] for _, k in rs])
            yd = np.array([S[k]["drake"][var] for _, k in rs])
            sp, sd = sph["base"][var], sph["drake"][var]
        ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
        ax.plot(r, yp, "o-", color=PADS, ms=3.5, lw=1.3, label="1 mm pads")
        ax.plot([xs], [sp], "o", color=PADS, ms=3.5)
        if yd is not None:
            ax.plot(r, yd, "s--", color=DRAKE, ms=3.2, lw=1.1, label="Drake hydroelastic")
            if sd is not None:
                ax.plot([xs], [sd], "s", color=DRAKE, ms=3.2)
        ax.set_xticks([0, 2, 4, 6, xs])
        ax.set_xticklabels(["0", "2", "4", "6", "sphere"])
        ax.set_xlabel("Fillet radius r (mm)")
        ax.set_title(title, fontsize=9.5)
    axs[0].set_ylim(0.85, 1.2)
    axs[1].set_ylim(0.8, 1.9)
    axs[2].set_ylim(0.4, 1.15)
    axs[0].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout(w_pad=1.2)
    out = MEDIA / "fig_deterioration.png"
    fig.savefig(out)
    print("wrote", out.relative_to(ROOT))


def fig_rate():
    plt = setup()
    m = json.loads((D / "deterioration/summary.json").read_text())["members"]["2.7"]
    fig, ax = plt.subplots(figsize=(4.6, 2.9), dpi=150)
    for key, lab, c, mk, ls in (("base", "1 mm pads as built", PADS, "o", "-"), ("fix", "pads, friction rows fixed", FIX, "^", "-"),
                                ("drake", "Drake hydroelastic", DRAKE, "s", "--")):
        sp = m[key]["spin_mm"]
        w = sorted(float(x) for x in sp)
        ax.plot(w, [sp[f"{x:g}"] for x in w], mk + ls, color=c, ms=3.6, lw=1.3, label=lab)
    ax.axhline(m["law"]["spin_1N_mm"], color=INK, lw=1.0, ls=":", label="law (quasi-static)")
    ax.axhline(m["law"]["pads_quasistatic_spin_1N_mm"], color=PADS, lw=0.9, ls=(0, (1, 2)), label="pads, quasi-static")
    ax.set_xscale("log")
    ax.set_xlabel("Spin rate ω (rad/s)")
    ax.set_ylabel("Sliding arm at 1 N (mm)")
    ax.legend(fontsize=8.2, loc="upper left")
    fig.tight_layout()
    out = MEDIA / "fig_rate.png"
    fig.savefig(out)
    print("wrote", out.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figs", nargs="+", choices=["deterioration", "rate", "screen", "stage2", "stage3"])
    a = ap.parse_args()
    for f in a.figs:
        globals()[f"fig_{f}"]()


if __name__ == "__main__":
    main()
