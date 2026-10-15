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
    drk = {}                                            # Drake's 1 N spins at every rate, sweep A and the fix grid
    for f in ("spin", "fixgrid"):
        for r in map(json.loads, open(D / f"deterioration/{f}.jsonl")):
            if r.get("case", "spin") == "spin" and r.get("model") == "drake_tpu2p7" and r.get("N") == 1.0 and r.get("status") == "complete":
                drk[f"{r['omega']:g}"] = r["rbar_spin_mm"]
    for key, lab, c, mk, ls in (("base", "1 mm pads as built", PADS, "o", "-"), ("fix", "pads, friction rows fixed", FIX, "^", "-"),
                                ("drake", "Drake hydroelastic", DRAKE, "s", "--")):
        sp = drk if key == "drake" else m[key]["spin_mm"]
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


RS = (1.0, 2.0, 3.0, 4.5, 6.0, 7.4)
RE = (1.0, 3.0, 6.0, 10.5)


def _grid_key(rs, re_):
    return f"tcd_rs{rs:g}_re{re_:g}_p0.2"


def fig_screen():
    """Three maps over the as-printed grid (rows r_side, columns r_end): J1, and the 1 mm pads against the law: the
    sliding arm of the 1 rad/s spin and the braking normal force at 45 deg."""
    plt = setup()
    S = json.loads((D / "screen/summary_stage1.json").read_text())["candidates"]
    law = {r["tip"]: r for r in map(json.loads, open(D / "screen/fidelity/law.jsonl")) if r["field"] == "drake"}
    maps = {"J1": np.full((len(RS), len(RE)), np.nan), "spin": np.full((len(RS), len(RE)), np.nan),
            "N45": np.full((len(RS), len(RE)), np.nan)}
    for i, rs in enumerate(RS):
        for j, re_ in enumerate(RE):
            k = _grid_key(rs, re_)
            if k not in S:
                continue
            v, L = S[k], law.get(k)
            maps["J1"][i, j] = v["J1"] if v["J1"] is not None else np.nan
            f = v["fidelity"]
            if L and f.get("pads_spin_1N_1_mm"):
                maps["spin"][i, j] = f["pads_spin_1N_1_mm"] / L["spin_arm_mm"]["1"]
            if L and f.get("pads_N45_N") and L["brake_N"].get("N45"):
                maps["N45"][i, j] = f["pads_N45_N"] / L["brake_N"]["N45"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), dpi=150)
    spec = [("J1 at \u03bc 0.7 (stage 1)", "J1", "Greys", None, None, "{:.2f}"),
            ("Pads spin arm / law, 1 rad/s", "spin", "RdBu_r", 0.0, 2.0, "{:.2f}"),
            ("Pads brake force / law, 45\u00b0", "N45", "RdBu_r", 0.0, 2.0, "{:.2f}")]
    for ax, (title, k, cmap, lo, hi, fmt) in zip(axs, spec):
        A = maps[k]
        if lo is None:
            lo, hi = np.nanmin(A), np.nanmax(A)
        ax.imshow(A, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        for i in range(len(RS)):
            for j in range(len(RE)):
                if np.isfinite(A[i, j]):
                    t = (A[i, j] - lo) / (hi - lo) if hi > lo else 0.5
                    dark = (t > 0.6) if cmap == "Greys" else (t < 0.12 or t > 0.88)
                    ax.text(j, i, fmt.format(A[i, j]), ha="center", va="center", fontsize=8.6,
                            color="white" if dark else INK)
        ax.set_xticks(range(len(RE)))
        ax.set_xticklabels([f"{x:g}" for x in RE])
        ax.set_yticks(range(len(RS)))
        ax.set_yticklabels([f"{x:g}" for x in RS])
        ax.set_xlabel("End fillet r_end (mm)")
        ax.grid(False)
        ax.set_title(title, fontsize=9.3)
    axs[0].set_ylabel("Side fillet r_side (mm)")
    fig.tight_layout(w_pad=0.8)
    out = MEDIA / "fig_screen.png"
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
