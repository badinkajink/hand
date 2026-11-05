#!/usr/bin/env python3
"""Figures of the fingertip-shape rerun page (docs/experiments/20261104-fingertip_shape_rerun/), from the run's summaries,
in the style of scripts/curved_tips_figures.py (9.5 pt at 150 dpi on 7.2 in wide figures, at least 13 px when a figure is
shown 828 px wide; pads red, Drake blue, the law black; 10-15's values at 10 MPa on the cad_v1 hand as open markers).

  mount    media/fig_mount.png: each tip's design surface on the v4b finger, the profile along the finger (y = 0) and the
           section across it 6 mm distal of the contact point, with the cassette 3T, the cap TW15H3 and the DIP link
           (mount/sections.npz)
  bed      media/fig_bed.png: twist-onset arm at 1 N and spin arm at 1 rad/s over the law, per tip (bed/summary.json)
  screen   media/fig_screen.png: the stage score per condition and J2, the Drake agreement and the cap's share of the
           cells' normal force, per tip (screen/summary.json)
  stage3   media/fig_stage3.png: J3 and the paired difference against the v4b tip (final/summary.json)
  fe       media/fig_fe.png: approach at 1 N under the 2 mm sphere at the seven map points and under the 12.5 mm rod, and
           the force resolution at the contact point with 05R4 (fe/summary.json)

    ~/.claude/bin/resguard.sh run --mem 2G -- .venv/bin/python scripts/fingertip_shape_rerun_figures.py mount bed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import curved_tips_figures as CF  # noqa: E402
import fingertip_shape_rerun_common as C  # noqa: E402

D = C.D
MEDIA = D / "media"
OLD = ROOT / "docs/experiments/20261015-curved_tips"
INK, INK2, GRID, PADS, DRAKE, THIRD = CF.INK, CF.INK2, CF.GRID, CF.PADS, CF.DRAKE, CF.THIRD
TPU, RIGID = "#e6c7a8", "#9a9a9a"
ORDER = [C.REF] + C.HALF + [C.HEMI] + C.CAPSULES + C.BLOCKS + C.QUARTERS
SEPS = (2.5, 3.5, 6.5, 8.5, 11.5)               # between the half-ellipsoids, hemisphere, capsules, blocks, quarters, sphere


def setup():
    CF.MEDIA = MEDIA
    return CF.setup()


def J(rel, base=D):
    p = base / rel
    return json.loads(p.read_text()) if p.exists() else {}


def short(k):
    return C.short(k).replace("half ", "Half ").replace("quarter ", "Quarter ").replace("capsule", "Capsule").replace(
        "block", "Block").replace("hemisphere", "Hemisphere").replace("sphere 10.55", "Sphere 10.55")


def ticks(ax, tips):
    ax.set_xticks(np.arange(len(tips)))
    ax.set_xticklabels([short(k) for k in tips], fontsize=8.5, rotation=32, ha="right", rotation_mode="anchor")


def seps(axs, tips):
    for a in np.atleast_1d(axs):
        for s in SEPS:
            if s < len(tips):
                a.axvline(s, color=GRID, lw=1.2)


def old_key(k):
    return C.OLD_KEY.get(k, k)


def fig_mount():
    """Per tip two panels: the profile along the finger at y = 0 (z_t horizontal, distal end left; x_t up, palmar on
    top) and the section across it 6 mm distal of the contact point (y horizontal), through the cap's side blocks; the
    tip's pad in TPU colour over the rigid parts, the grooves and pin hole of pad 05R4 darker, TPU inside a rigid part
    where the 11/22 has none red."""
    plt = setup()
    S = np.load(D / "mount" / "sections.npz")
    M = J("mount/mount.json")["tips"]
    xF, yF, zF = S["x_F"], S["y_F"], S["z_F"]
    z_tip = -51.55 - 36.36
    fig, axs = plt.subplots(6, 4, figsize=(7.2, 9.6), dpi=150)
    rig = {p: (S[f"part:{p}|y0"], S[f"part:{p}|zd6"]) for p in ("dip_link", "cassette", "cap")}
    for i, k in enumerate(ORDER):
        r, c = 2 * (i // 4), i % 4
        for row, plane in ((r, "y0"), (r + 1, "zd6")):
            ax = axs[row, c]
            pad = S[f"{k}|{plane}"]
            h = np.zeros(pad.shape + (3,)) + 1.0
            rigid = np.logical_or.reduce([v[0 if plane == "y0" else 1] for v in rig.values()])
            h[rigid] = np.array([0.6, 0.6, 0.6])
            ref = S[f"{C.REF}|{plane}"]
            h[pad] = np.array([0.90, 0.78, 0.66])
            h[pad & rigid & ref] = np.array([0.55, 0.40, 0.28])      # grooves and pin hole, cut as in pad 05R4
            h[pad & rigid & ~ref] = np.array([0.80, 0.10, 0.10])     # TPU inside a rigid part where the 11/22 has none
            if plane == "y0":                                   # rows x_F, columns z_F -> x_t up, z_t right
                ext = (zF[0] - z_tip, zF[-1] - z_tip, -xF[-1], -xF[0])
                ax.imshow(h[::-1], extent=ext, origin="lower", interpolation="nearest", aspect="equal")
                ax.set_xlim(-13.5, 12)
            else:                                               # rows x_F, columns y_F
                ext = (yF[0], yF[-1], -xF[-1], -xF[0])
                ax.imshow(h[::-1], extent=ext, origin="lower", interpolation="nearest", aspect="equal")
                ax.set_xlim(-11.5, 11.5)
            ax.set_ylim(-7.5, 11.2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            if plane == "y0":
                v = M[k]["verdict"]
                ax.set_title(f"{short(k)}\n{'mounts as the 11/22' if v == 'mounts' else 'envelope alone'}", fontsize=8.6)
    fig.tight_layout(h_pad=0.3, w_pad=0.2)
    out = MEDIA / "fig_mount.png"
    fig.savefig(out)
    print("wrote", out.relative_to(ROOT))


def fig_bed():
    """(a) Twist-onset arm at 1 N: the law, the pads and Drake at 2.5 MPa, and 10-15's pads at 10 MPa; (b) spin arm at
    1 rad/s over the law, pads and Drake, with 10-15's pads."""
    plt = setup()
    S = J("bed/summary.json").get("tips", {})
    O = J("bed/summary.json", OLD).get("tips", {})
    tips = [k for k in ORDER + [C.SPHERE] if k in S]
    x = np.arange(len(tips))
    g = lambda d, k, *p: CF._get(d.get(k) or {}, *p)  # noqa: E731
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.8), dpi=150, sharex=True)
    ax = axs[0]
    ax.plot(x, [g(S, k, "law", "onset_arm_mm", "1") for k in tips], "_", color=INK, ms=14, mew=1.6, label="Law, 2.5 MPa")
    ax.plot(x - 0.14, [g(S, k, "pads", "onset_arm_1N_mm") for k in tips], "o", color=PADS, ms=4.5, label="Pads, 2.5 MPa")
    ax.plot(x + 0.14, [g(S, k, "drake", "onset_arm_1N_mm") for k in tips], "s", color=DRAKE, ms=4.2, label="Drake, 2.5 MPa")
    ax.plot(x - 0.14, [g(O, old_key(k), "pads", "onset_arm_1N_mm") for k in tips], "o", color=PADS, mfc="none", ms=4.5,
            label="Pads, 10 MPa (15 October)")
    ax.set_ylabel("Twist-onset arm\nat 1 N (mm)")
    ax.set_ylim(0, 3.6)
    ax.legend(fontsize=8.0, loc="upper left", ncol=2)
    ax = axs[1]
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.plot(x - 0.14, [g(S, k, "ratios", "pads_spin_1_over_law") for k in tips], "o", color=PADS, ms=4.5)
    ax.plot(x + 0.14, [g(S, k, "ratios", "drake_spin_1_over_law") for k in tips], "s", color=DRAKE, ms=4.2)
    ax.plot(x - 0.14, [g(O, old_key(k), "ratios", "pads_spin_1_over_law") for k in tips], "o", color=PADS, mfc="none", ms=4.5)
    ax.set_ylabel("Spin arm at 1 rad/s\nover the law")
    ax.set_ylim(0.8, 1.4)
    ticks(ax, tips)
    seps(axs, tips)
    fig.tight_layout(h_pad=0.6)
    out = MEDIA / "fig_bed.png"
    fig.savefig(out)
    print("wrote", out.relative_to(ROOT))


COND_LABEL = {f"0.7|{C.SERVO}": ("Friction 0.7", "o"), f"1.0|{C.SERVO}": ("Friction 1.0", "^"),
              f"0.5|{C.SERVO}": ("Friction 0.5", "v"), "0.7|kp0.5_kv0_fr0.18_fl0_dp0.0144": ("Friction 0.7, servo gain 0.5", "D"),
              "0.7|kp0.72_kv0_fr0.12_fl0_dp0.0144": ("Friction 0.7, torque limit 0.12 N m", "s")}


def fig_screen():
    """(a) Stage score J per tip at the five conditions and J2 as a bar, 10-15's J2 as a dash; (b) the D1-D8 replays
    within 3 deg of Drake (of 24) and the gate; (c) the share of the fingers' normal force on the cells' cylinder that the
    rigid parts take in the HOM turns (median over the rollouts of all conditions, bar to the 75th percentile)."""
    plt = setup()
    S = J("screen/summary.json").get("tips", {})
    O = J("screen/summary.json", OLD).get("tips", {})
    tips = [k for k in ORDER if k in S]
    x = np.arange(len(tips))
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.8), dpi=150, sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1, 0.9]))
    ax = axs[0]
    ax.bar(x, [S[k]["J2"] or 0 for k in tips], width=0.62, color="#e6e6e6", ec="#9a9a9a", lw=0.6, label="J2 (minimum)")
    cols = ["#111111", "#2166ac", "#b2182b", "#e08214", "#1b7837"]
    for (c, (lab, mk)), col in zip(COND_LABEL.items(), cols):
        ax.plot(x, [S[k]["conditions"].get(c, {}).get("J") for k in tips], mk, color=col, ms=4.2, label=lab,
                mfc="none" if mk in "Ds" else col)
    ax.plot(x, [(O.get(old_key(k)) or {}).get("J2") for k in tips], "_", color=INK2, ms=16, mew=1.4, label="J2, 15 October")
    ax.set_ylabel("Stage score J")
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=7.8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=0.9)
    ax = axs[1]
    base = (S.get(C.REF) or {}).get("agree_drake_3deg") or 0
    ax.bar(x, [S[k]["agree_drake_3deg"] or 0 for k in tips], width=0.62,
           color=[DRAKE if S[k]["gate_pass"] else "#c9c9c9" for k in tips])
    ax.axhline(base - 2, color=INK2, lw=0.9, ls="--")
    ax.text(len(tips) - 0.6, base - 2 + 0.6, f"Gate: {base - 2} of 24", fontsize=8.3, ha="right", color=INK2)
    ax.set_ylabel("Replays within 3 deg\nof Drake (of 24)")
    ax.set_ylim(0, 24)
    ax = axs[2]
    med = np.array([S[k]["rigid_share_all"]["median"] or 0 for k in tips])
    q75 = np.array([S[k]["rigid_share_all"]["q75"] or 0 for k in tips])
    ax.bar(x, med, width=0.62, color=RIGID)
    ax.errorbar(x, med, yerr=[np.zeros_like(med), q75 - med], fmt="none", ecolor=INK2, lw=0.9, capsize=2)
    ax.set_ylabel("Rigid parts' share\nof the normal force")
    ax.set_ylim(0, 0.7)
    ticks(ax, tips)
    seps(axs, tips)
    fig.tight_layout(h_pad=0.6)
    out = MEDIA / "fig_screen.png"
    fig.savefig(out)
    print("wrote", out.relative_to(ROOT))


def fig_stage3():
    plt = setup()
    S = J("final/summary.json").get("candidates", {})
    if not S:
        print("no stage-3 summary")
        return
    tips = [C.REF] + [k for k in S if k != C.REF]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=150, gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(tips))[::-1]
    ax = axs[0]
    cols = ["#111111", "#b2182b", "#e08214", "#1b7837"]
    conds = [c for c in COND_LABEL if not c.startswith("1.0|")]
    ax.barh(y, [S[k]["J3"] for k in tips], height=0.6, color="#e6e6e6", ec="#9a9a9a", lw=0.6, label="J3 (minimum)")
    for c, col in zip(conds, cols):
        lab, mk = COND_LABEL[c]
        ax.plot([S[k]["J"].get(c) for k in tips], y, mk, color=col, ms=4.2, label=lab, mfc="none" if mk in "Ds" else col)
    ax.set_yticks(y)
    ax.set_yticklabels([short(k) for k in tips], fontsize=8.5)
    ax.set_xlabel("Stage score on fresh placements")
    ax.legend(fontsize=7.4, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax = axs[1]
    for yy, k in zip(y, tips):
        v = S[k]
        if "dJ3_ci90" in v:
            lo, hi = v["dJ3_ci90"]
            ax.plot([lo, hi], [yy, yy], color=INK2, lw=1.4)
            ax.plot([v["dJ3_vs_base"]], [yy], "o", color=PADS, ms=4.5)
    ax.axvline(0, color=INK2, lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("J3 minus the v4b tip's (90 % interval)")
    fig.tight_layout(w_pad=0.6)
    out = MEDIA / "fig_stage3.png"
    fig.savefig(out)
    print("wrote", out.relative_to(ROOT))


MAP = ("apex", "y2.6", "y4.1", "y5.2", "z+4", "z-4", "z-7")


def fig_fe():
    """(a) Approach at 1 N under the 2 mm-radius sphere at the seven map points over the apex's, per tip; (b) approach
    at 1 N under the 2 mm sphere at the apex and the 12.5 mm rod across the finger; (c) the force resolution at the
    contact point with 05R4 (one 141 Hz sample, contact position and background unknown)."""
    plt = setup()
    S = J("fe/summary.json").get("tips", {})
    tips = [k for k in ORDER if k in S]
    if not tips:
        print("no FE summary")
        return
    x = np.arange(len(tips))
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.4), dpi=150, sharex=True)
    ax = axs[0]
    for i, k in enumerate(tips):
        r = S[k]["sphere_map_over_apex"]
        v = [r.get(p) for p in MAP if r.get(p) is not None]
        ax.plot([i] * len(v), v, "o", color=PADS, ms=3.6, alpha=0.8)
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.set_ylabel("Sphere approach at 1 N\nover the apex's")
    ax = axs[1]
    ax.plot(x - 0.12, [S[k]["sphere_apex_um"]["1"] for k in tips], "o", color=PADS, ms=4.5, label="2 mm sphere at the apex")
    ax.plot(x + 0.12, [S[k]["rod_across_um"]["1"] for k in tips], "s", color=DRAKE, ms=4.2, label="12.5 mm rod across")
    ax.set_ylabel("Approach at\n1 N (µm)")
    ax.legend(fontsize=8.0, loc="upper left", ncol=2)
    ax = axs[2]
    res = [((S[k].get("sensing") or {}).get("centre_force_N")) for k in tips]
    ax.bar(x, [r if r is not None else 0 for r in res], width=0.62, color=THIRD)
    for i, r in enumerate(res):
        if r is None:
            ax.text(i, 0.002, "no fit", rotation=90, ha="center", va="bottom", fontsize=7.6, color=INK2)
    ax.set_ylabel("Force resolution at\nthe contact point (N)")
    ticks(ax, tips)
    seps(axs, tips)
    fig.tight_layout(h_pad=0.6)
    out = MEDIA / "fig_fe.png"
    fig.savefig(out)
    print("wrote", out.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="+", choices=["mount", "bed", "screen", "stage3", "fe"])
    a = ap.parse_args()
    for w in a.what:
        {"mount": fig_mount, "bed": fig_bed, "screen": fig_screen, "stage3": fig_stage3, "fe": fig_fe}[w]()


if __name__ == "__main__":
    main()
