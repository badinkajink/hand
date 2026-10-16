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
  stiffness media/fig_stiffness.png: the probe maps of four tips under the finite elements and the models
           (stiffness/summary.json)

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
    m = re.match(r"tcd_qell(c?)([0-9.]+)x([0-9.]+)x([0-9.]+)(?:x([0-9.]+))?", k)
    if m:
        return f"Quarter{' (c)' if m[1] else ''}\n{m[3]} · {m[4]}" + (f"/{m[5]}" if m[5] else "")
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
    m = re.match(r"tcd_qell([0-9.]+)x([0-9.]+)x([0-9.]+)(?:x([0-9.]+))?", k)
    if m:
        return f"Quarter-ellipsoid\n{m[1]} × {m[2]} × {m[3]}" + (f"/{m[4]}" if m[4] else "") + " mm"
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
    c = getattr(sh, "c0", TG.C0)                        # an interior point (a quarter-ellipsoid's lies above its back)
    if plane == "xz":
        o = np.array([c[0], 0.0, 0.0 if getattr(sh, "family", "") == "qell" else TG.C0[2]])
        D2 = np.column_stack([np.cos(th), np.zeros(n), np.sin(th)])
    else:
        o = np.array([c[0], 0.0, 0.0])
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
    tips = [C.ASBUILT.rsplit("_p", 1)[0], C.FINALIST.rsplit("_p", 1)[0]] + C.CAPSULES + C.ELLIPSOIDS + C.QUARTERS
    fig, axs = plt.subplots(3, 4, figsize=(7.2, 4.5), dpi=150)
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
        import curved_tips_fe as CF
        dep = CF.depth_of(sh)
        xm = (TG.X_FACE * 1e3 - dep - 1.59, TG.X_FACE * 1e3 - dep)
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
    for ax in axs.ravel()[len(tips):]:
        ax.axis("off")
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    out = MEDIA / "fig_shapes.png"
    fig.savefig(out)
    print("wrote", out.relative_to(ROOT))


BED_ORDER = ([C.ASBUILT, C.FINALIST] + [k + C.PRINT for k in C.CAPSULES + C.ELLIPSOIDS + C.QUARTERS]
             + [C.QUARTER_C + C.PRINT, C.LITERAL, C.SPHERE_ENV, C.SPHERE])


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
        a.axvline(7.5, color=GRID, lw=1.2)
        a.axvline(len(tips) - 3.5, color=GRID, lw=1.2)
    fig.tight_layout(h_pad=0.6)
    out = MEDIA / "fig_bed.png"
    fig.savefig(out)
    print("wrote", out.relative_to(ROOT))


TRAJ_ORDER = [C.ASBUILT, C.FINALIST] + [k + C.PRINT for k in C.CAPSULES + C.ELLIPSOIDS + C.QUARTERS]


def fig_traj():
    """Per tip, the median over the 24 placements (bar: 25-75 %) of the RMS tool-angle difference and of the RMS
    friction-force difference between the pads at the planner's numerics and Drake, between the RL trainer's numerics and
    the planner's, and of the spread between placements (Drake's rollouts of one hand at two seeds)."""
    plt = setup()
    S = json.loads((D / "traj/summary.json").read_text())["tips"]
    tips = [k for k in TRAJ_ORDER if k in S]
    x = np.arange(len(tips))
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.0), dpi=150, sharex=True)
    series = (("plan-drake", "Pads at the planner's numerics against Drake", PADS, "o", -0.18),
              ("rl-plan", "RL trainer's numerics against the planner's", THIRD, "^", 0.0),
              ("spread", "Spread between placements (Drake, two seeds)", DRAKE, "s", 0.18))
    for ax, key, ylab in ((axs[0], "rms_dang_deg", "RMS tool-angle\ndifference (deg)"),
                          (axs[1], "rms_dT_N", "RMS friction-force\ndifference (N)")):
        for pair, lab, col, mk, dx in series:
            med, lo, hi = [], [], []
            for k in tips:
                v = (S[k]["spread"].get("drake") if pair == "spread" else S[k]["pairs"].get(pair)) or {}
                q = v.get(key) or {}
                med.append(q.get("median", np.nan)), lo.append(q.get("q25", np.nan)), hi.append(q.get("q75", np.nan))
            med, lo, hi = map(np.array, (med, lo, hi))
            ax.errorbar(x + dx, med, yerr=[med - lo, hi - med], fmt=mk, color=col, ms=4.5, lw=1.0, capsize=2,
                        label=lab)
        ax.set_ylabel(ylab)
        ax.set_yscale("log")
    axs[0].set_ylim(0.5, 120)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=8.3, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    axs[1].set_ylim(0.05, 1.0)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([short(k).replace("\n", " ") for k in tips], fontsize=8.5, rotation=32, ha="right",
                           rotation_mode="anchor")
    for a in axs:
        a.axvline(1.5, color=GRID, lw=1.2)
        a.axvline(7.5, color=GRID, lw=1.2)
    fig.tight_layout(h_pad=0.6, rect=(0, 0, 1, 0.92))
    out = MEDIA / "fig_traj.png"
    fig.savefig(out)
    print("wrote", out.relative_to(ROOT))


SCREEN_ORDER = [C.ASBUILT, C.FINALIST] + [k + C.PRINT for k in C.CAPSULES + C.ELLIPSOIDS + C.QUARTERS]
COND_LABEL = {"0.7|kp0.72_kv0_fr0.18_fl0_dp0.0144": ("Friction 0.7", "o"), "1.0|kp0.72_kv0_fr0.18_fl0_dp0.0144": ("Friction 1.0", "^"),
              "0.5|kp0.72_kv0_fr0.18_fl0_dp0.0144": ("Friction 0.5", "v"),
              "0.7|kp0.5_kv0_fr0.18_fl0_dp0.0144": ("Friction 0.7, servo gain 0.5", "D"),
              "0.7|kp0.72_kv0_fr0.12_fl0_dp0.0144": ("Friction 0.7, servo torque 0.12 N m", "s")}


def fig_screen():
    """(a) The stage score J per tip at each of the five conditions, J2 (their minimum) as a bar; (b) the D1-D8 replays
    whose pads end within 3 deg of Drake with its held outcome, of 24, and the agreement gate."""
    plt = setup()
    S = json.loads((D / "screen/summary.json").read_text())["tips"]
    tips = [k for k in SCREEN_ORDER if k in S]
    x = np.arange(len(tips))
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.2), dpi=150, sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]))
    ax = axs[0]
    ax.bar(x, [S[k]["J2"] or 0 for k in tips], width=0.62, color="#e6e6e6", ec="#9a9a9a", lw=0.6, label="J2 (minimum over the conditions)")
    cols = ["#111111", "#2166ac", "#b2182b", "#e08214", "#1b7837"]
    for (c, (lab, mk)), col in zip(COND_LABEL.items(), cols):
        ax.plot(x, [S[k]["conditions"].get(c, {}).get("J") for k in tips], mk, color=col, ms=4.2, label=lab, mfc="none" if mk in "Ds" else col)
    ax.set_ylabel("Stage score J")
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=8.0, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.0)
    ax = axs[1]
    base = S[C.ASBUILT]["agree_drake_3deg"]
    ax.bar(x, [S[k]["agree_drake_3deg"] or 0 for k in tips], width=0.62,
           color=[DRAKE if S[k]["gate_pass"] else "#c9c9c9" for k in tips])
    ax.axhline(base - 2, color=INK2, lw=0.9, ls="--")
    ax.text(len(tips) - 0.6, base - 2 + 0.6, f"Gate: {base - 2} of 24", fontsize=8.3, ha="right", color=INK2)
    ax.set_ylabel("D1-D8 replays within\n3 deg of Drake (of 24)")
    ax.set_ylim(0, 24)
    ax.set_xticks(x)
    ax.set_xticklabels([short(k).replace("\n", " ") for k in tips], fontsize=8.5, rotation=32, ha="right", rotation_mode="anchor")
    for a in axs:
        a.axvline(1.5, color=GRID, lw=1.2)
        a.axvline(7.5, color=GRID, lw=1.2)
    fig.tight_layout(h_pad=0.6)
    out = MEDIA / "fig_screen.png"
    fig.savefig(out)
    print("wrote", out.relative_to(ROOT))


def fig_stage3():
    """J3 per stage-3 tip at its four conditions and their minimum, and the paired difference against the as-built
    block with its 90 % bootstrap interval."""
    plt = setup()
    S = json.loads((D / "final/summary.json").read_text())["candidates"]
    tips = [C.ASBUILT, C.FINALIST] + [k for k in S if k not in (C.ASBUILT, C.FINALIST)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=150, gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(tips))[::-1]
    ax = axs[0]
    cols = ["#111111", "#b2182b", "#e08214", "#1b7837"]
    conds = [c for c in COND_LABEL if c != "1.0|kp0.72_kv0_fr0.18_fl0_dp0.0144"]
    for c, col in zip(conds, cols):
        lab, mk = COND_LABEL[c]
        ax.plot([S[k]["J"].get(c) for k in tips], y, mk, color=col, ms=4.2, label=lab, mfc="none" if mk in "Ds" else col)
    ax.plot([S[k]["J3"] for k in tips], y, "|", color=INK, ms=14, mew=2.0, label="J3 (minimum)")
    ax.set_yticks(y)
    ax.set_yticklabels([short(k).replace("\n", " ") for k in tips], fontsize=8.5)
    ax.set_xlabel("Stage score on fresh placements")
    ax.set_xlim(0, 0.45)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, fontsize=8.0, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.3)
    ax = axs[1]
    ax.axvline(0, color=INK2, lw=0.9)
    for yy, k in zip(y, tips):
        v = S[k]
        if "dJ3_ci90" not in v:
            continue
        lo, hi = v["dJ3_ci90"]
        ax.plot([lo, hi], [yy, yy], color=PADS, lw=1.6)
        ax.plot([v["dJ3_vs_base"]], [yy], "o", color=PADS, ms=4.5)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("J3 minus the as-built block's\n(90 % bootstrap interval)")
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.84))
    out = MEDIA / "fig_stage3.png"
    fig.savefig(out)
    print("wrote", out.relative_to(ROOT))


STIFF_TIPS = ["tcd_rs3_re3", "tcd_cap8.5", "tcd_ell7.4x8.5x16x32", "tcd_qell7.4x8.5x11x22"]
STIFF_PROBES = ["apex", "y2.6", "y4.1", "y5.2", "z+4", "z-4", "z-7", "cyl0", "cyl+4", "cyl-4"]
PROBE_LABEL = {"apex": "Apex", "y2.6": "y 2.6", "y4.1": "y 4.1", "y5.2": "y 5.2", "z+4": "z +4", "z-4": "z -4",
               "z-7": "z -7", "cyl0": "Cyl. 0", "cyl+4": "Cyl. +4", "cyl-4": "Cyl. -4"}
STIFF_MODELS = [("fe", "Finite elements (reference)", INK, "s"), ("drake", "Drake's field", DRAKE, "o"),
                ("v1", "Pads v1", "#8c6bb1", "v"), ("v2", "Pads v2", PADS, "^"), ("v2c", "Pads v2, centroid depth", THIRD, "<"),
                ("fit", "Pads fitted to the finite elements", "#1b7837", "D")]


def fig_stiffness():
    """Per tip, the approach at 1 N at each probe position under each model (top), and the same divided by the model's
    apex sphere approach (bottom: the map's variation alone)."""
    plt = setup()
    S = json.loads((D / "stiffness/summary.json").read_text())["tips"]
    tips = [k for k in STIFF_TIPS if k in S]
    fig, axs = plt.subplots(2, len(tips), figsize=(7.2, 4.6), dpi=150, sharex=True, sharey="row")
    x = np.arange(len(STIFF_PROBES))
    for j, k in enumerate(tips):
        T = S[k]
        for key, lab, col, mk in STIFF_MODELS:
            if key == "fe":
                vals = [(T["fe"].get(p) or {}).get("1") for p in STIFF_PROBES]
            elif key in T["models"]:
                vals = [(T["models"][key]["approach_um"].get(p) or {}).get("1") for p in STIFF_PROBES]
            else:
                continue
            v = np.array([np.nan if a is None else a for a in vals], float)
            for row, yy in ((0, v), (1, v / v[0])):
                axs[row, j].plot(x, yy, mk + "-", color=col, ms=3.6, lw=0.9, label=lab if row == 0 else None,
                                 mfc="none" if key == "fit" else col)
        axs[0, j].set_title(title2(k).replace("Half-ellipsoid", "Half-ellipsoid").replace("\n", "\n"), fontsize=8.6)
        axs[0, j].set_yscale("log")
        axs[1, j].set_xticks(x)
        axs[1, j].set_xticklabels([PROBE_LABEL[p] for p in STIFF_PROBES], rotation=90, fontsize=8.6)
        for r in (0, 1):
            axs[r, j].axvline(6.5, color=GRID, lw=1.2)
    axs[0, 0].set_ylabel("Approach at 1 N (um)".replace("um", "\u00b5m"))
    axs[1, 0].set_ylabel("Over the apex\nsphere approach")
    axs[0, 0].set_ylim(15, 900)
    from matplotlib.ticker import FixedLocator, NullLocator
    axs[0, 0].yaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500]))
    axs[0, 0].yaxis.set_minor_locator(NullLocator())
    axs[0, 0].set_yticklabels(["20", "50", "100", "200", "500"])
    axs[1, 0].set_ylim(0.0, 1.65)
    seen = {}
    for a in axs[0]:
        for h_, l_ in zip(*a.get_legend_handles_labels()):
            seen.setdefault(l_, h_)
    order = [lab for _, lab, _, _ in STIFF_MODELS if lab in seen]
    fig.legend([seen[k] for k in order], order, fontsize=8.6, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(h_pad=0.4, w_pad=0.3, rect=(0, 0, 1, 0.9))
    out = MEDIA / "fig_stiffness.png"
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
    ap.add_argument("figs", nargs="+", choices=["shapes", "bed", "traj", "screen", "stage3", "stiffness"])
    a = ap.parse_args()
    for f in a.figs:
        globals()[f"fig_{f}"]()


if __name__ == "__main__":
    main()
