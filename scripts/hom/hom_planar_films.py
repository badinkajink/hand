#!/usr/bin/env python3
"""Films of Figure 5 of the HOM review page: Eq. (1) and Eq. (2) on a planar two-joint finger with the SR2 link lengths
and the v4b pad beside a fixed 25 mm cylinder, one relative-velocity command at a time (owner, 2026-11-10: "videos that
really explain well the physical, spatial intuition of the GCF twist and hom control"). The motion is the same
computation as the figure (hom_review_figures.planar_run: unit weights, ell = 20 mm, no joint limits).

Each frame: the finger, the pad and its trail; the GCF at the pad's witness point (normal n toward the cylinder,
tangent t, roll about the axis out of the page); the commanded relative velocity (dashed) and the achieved one (solid);
on the right, Eq. (2) and the three components, commanded against achieved. Writes
docs/experiments/20261029-hom_control_review/media/20261110-planar_{separation,slide,roll}.mp4, the three in sequence
(20261110-planar_motions.mp4) and posters.

    MPLBACKEND=Agg .venv/bin/python scripts/hom/hom_planar_films.py
"""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyArrowPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_review_figures as RF  # noqa: E402

MEDIA = ROOT / "docs/experiments/20261029-hom_control_review/media"
INK, GREY, PALE = "#1f2937", "#9ca3af", "#f3f4f6"
FING, PAD, PADL = "#94a3b8", "#059669", "#d1fae5"
CMD, ACH, NRM, TAN = "#7c3aed", "#ea580c", "#dc2626", "#2563eb"
FPS, SLOW, HOLD_S = 25, 4.0, 1.6
plt.rcParams.update({"font.family": "Lato", "font.size": 17, "mathtext.fontset": "cm"})
R = 12.5
TH0 = np.radians([20.0, 40.0])
CASES = [("separation", "Separation", (0.0, 20.0, 0.0), "close the gap along the normal at 20 mm/s"),
         ("slide", "Slide", (0.0, 0.0, 20.0), "slide along the surface at 20 mm/s"),
         ("roll", "Roll", (math.radians(40.0), 0.0, 0.0), "roll the fingertip at 40 deg/s")]
UNITS = [("roll", "deg/s", 180.0 / math.pi, 50.0), ("separation", "mm/s", 1.0, 25.0), ("slide", "mm/s", 1.0, 25.0)]


def setup():
    _, pad0 = RF.planar_fk(TH0)
    al = math.radians(-20.0)
    return pad0 + (RF.RHO + R + 6.0) * np.array([math.cos(al), math.sin(al)])


def eq_png():
    """Eq. (2) for the planar finger, typeset by LaTeX (scripts/docs/codesign_tikz.compile_tex)."""
    import codesign_tikz as CT
    CT.OUT = MEDIA
    tex = (r"\fontsize{20}{25}\selectfont$\displaystyle u=\arg\min_{u}\ \sum_{k\,\in\,\{\mathrm{roll},\,\mathrm{sep},"
           r"\,\mathrm{slide}\}} w_k\Big(v_{\mathrm{ref},k}-\big(A(\theta)\,u\big)_k\Big)^2+w_{\mathrm{d}}\lVert u\rVert^2$")
    return np.asarray(Image.open(CT.compile_tex("20261110-planar_eq2", tex, border="10pt")).convert("RGB"))


def view(runs, C):
    pts = [np.zeros(2), C + R, C - R] + [s[k] for run in runs for s in run for k in ("pip", "pad")]
    P = np.array(pts)
    lo, hi = P.min(0) - 16.0, P.max(0) + 16.0
    return lo, hi


def arc_arrow(ax, c, r, w, color, ls="-"):
    """A curved arrow about c for an angular rate w (rad/s): sweep proportional to |w|, counter-clockwise if w > 0."""
    sweep = max(25.0, min(300.0, abs(math.degrees(w)) * 4.0))
    a0 = 20.0
    t1, t2 = (a0, a0 + sweep) if w > 0 else (a0 + sweep, a0)
    ax.add_patch(Arc(c, 2 * r, 2 * r, theta1=min(t1, t2), theta2=max(t1, t2), color=color, lw=3, ls=ls))
    end = math.radians(t2 if w > 0 else t2)
    e = c + r * np.array([math.cos(end), math.sin(end)])
    tang = np.array([-math.sin(end), math.cos(end)]) * (1 if w > 0 else -1)
    ax.add_patch(FancyArrowPatch(e - 0.1 * tang, e + 1.5 * tang, arrowstyle="-|>", mutation_scale=22, color=color, lw=0))


def frame(run, k, C, lims, title, sub, vref, eqimg, means):
    s = run[k]
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.02, 0.06, 0.56, 0.80])
    ax.set_xlim(lims[0][0], lims[1][0])
    ax.set_ylim(lims[0][1], lims[1][1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle(tuple(C), R, fc="#e5e7eb", ec="#6b7280", lw=2, zorder=1))
    ax.text(C[0], C[1] - R - 3.0, "fixed 25 mm cylinder", ha="center", va="top", fontsize=14, color="#6b7280")
    for j in range(0, k, 10):                      # earlier poses, faint
        g = run[j]
        P = np.array([[0, 0], g["pip"], g["pad"]])
        ax.plot(P[:, 0], P[:, 1], color=FING, lw=7, alpha=0.12, solid_capstyle="round", zorder=2)
    tr = np.array([g["pad"] for g in run[:k + 1]])
    ax.plot(tr[:, 0], tr[:, 1], color=PAD, lw=1.6, ls="--", zorder=3)
    P = np.array([[0, 0], s["pip"], s["pad"]])
    ax.plot(P[:, 0], P[:, 1], color=FING, lw=12, solid_capstyle="round", zorder=4)
    for q in (P[0], P[1]):
        ax.add_patch(Circle(tuple(q), 2.6, fc="white", ec=INK, lw=2, zorder=5))
    ax.add_patch(Circle(tuple(s["pad"]), RF.RHO, fc=PADL, ec=PAD, lw=2.5, zorder=6))
    ax.text(*(P[0] + np.array([-4, 4])), "MCP", ha="right", fontsize=13, color=INK)
    # the GCF at the pad's witness point
    p1 = s["p1"]
    n = (C - s["pad"]) / np.linalg.norm(C - s["pad"])
    t = np.array([-n[1], n[0]])
    L = 13.0
    for vec, side, col, lab in ((n, -t, NRM, "$n$ separation"), (t, n, TAN, "$t$ slide")):
        ax.add_patch(FancyArrowPatch(p1, p1 + L * vec, arrowstyle="-|>", mutation_scale=16, color=col, lw=2.0, zorder=13))
        q = p1 + L * vec + 4.2 * side
        ax.text(*q, lab, color=col, fontsize=15, ha="center", va="center", zorder=13,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.add_patch(Circle(tuple(p1), 1.3, fc="white", ec=INK, lw=1.6, zorder=9))
    ax.plot(*p1, "o", ms=3, color=INK, zorder=10)
    ax.text(*(p1 + np.array([-3.5, -5.5])), "GCF", fontsize=13, color=INK, ha="right", zorder=9)
    # commanded (dashed) and achieved (solid) relative velocity at the GCF origin
    sc = 0.55                                      # mm of arrow per mm/s
    vr, va = np.asarray(s["v_ref"], float), s["v_ach"]      # the command HOM used, after the anti-windup clamp
    if np.abs(vr - np.asarray(vref, float)).max() > 1e-9:
        fig.text(0.60, 0.515, "anti-windup: inside 0.5 mm the closing command is set to 0", fontsize=14,
                 color=CMD, weight="bold")
    lin_c = vr[1] * n + vr[2] * t
    lin_a = va[1] * n + va[2] * t
    if np.linalg.norm(lin_c) > 1e-9:
        ax.add_patch(FancyArrowPatch(p1, p1 + sc * lin_c, arrowstyle="-|>", mutation_scale=20, color=CMD, lw=2.8,
                                     ls="--", zorder=11))
    if np.linalg.norm(lin_a) > 0.3:
        ax.add_patch(FancyArrowPatch(p1, p1 + sc * lin_a, arrowstyle="-|>", mutation_scale=20, color=ACH, lw=3.6,
                                     zorder=12))
    if abs(vr[0]) > 1e-9:
        arc_arrow(ax, s["pad"], RF.RHO + 4.5, vr[0], CMD, ls="--")
    if abs(va[0]) > math.radians(1.0):
        arc_arrow(ax, s["pad"], RF.RHO + 8.0, va[0], ACH)
    fig.text(0.02, 0.93, f"HOM, one fingertip, one command: {title}", fontsize=27, weight="bold", color=INK)
    fig.text(0.02, 0.885, sub + ", zero in the other two", fontsize=18, color="#4b5563")
    fig.text(0.02, 0.03, f"t = {s['t']:.2f} s, shown at 1/{SLOW:g} speed", fontsize=14, color=GREY)
    lg = fig.add_axes([0.60, 0.80, 0.38, 0.05])
    lg.axis("off")
    lg.plot([0.0, 0.08], [0.5, 0.5], color=CMD, lw=3, ls="--")
    lg.text(0.10, 0.5, "commanded", va="center", fontsize=15, color=CMD)
    lg.plot([0.40, 0.48], [0.5, 0.5], color=ACH, lw=4)
    lg.text(0.50, 0.5, "achieved", va="center", fontsize=15, color=ACH)
    lg.set_xlim(0, 1)
    lg.set_ylim(0, 1)
    ea = fig.add_axes([0.60, 0.62, 0.38, 0.15])
    ea.imshow(eqimg)
    ea.axis("off")
    fig.text(0.60, 0.565, "every 10 ms: the joint speeds $u$ that best make\nthe commanded relative motion happen",
             fontsize=14, color="#4b5563")
    for j, (name, unit, f, lim) in enumerate(UNITS):
        b = fig.add_axes([0.66, 0.43 - 0.13 * j, 0.30, 0.08])
        cmd, ach = vr[j] * f, va[j] * f
        b.barh([0], [ach], color=ACH, height=0.6)
        b.axvline(cmd, color=CMD, lw=3, ls="--")
        b.axvline(0, color=INK, lw=1)
        b.set_xlim(-lim, lim)
        b.set_yticks([])
        b.tick_params(labelsize=12)
        for sp in ("top", "right", "left"):
            b.spines[sp].set_visible(False)
        fig.text(0.60, 0.465 - 0.13 * j, name, fontsize=16, color=INK, weight="bold")
        fig.text(0.60, 0.435 - 0.13 * j, f"{ach:+.1f} {unit}", fontsize=15, color=ACH)
    fig.text(0.60, 0.04, means, fontsize=16, color=INK, va="bottom")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def summary(name, run):
    m = np.array([s["v_ach"] for s in run]).mean(0)
    roll, sep, slide = math.degrees(m[0]), m[1], m[2]
    if name == "separation":
        return f"Mean over the second: separation {sep:+.1f} mm/s,\nuntil the clamp stops the pad 0.5 mm from the cylinder."
    if name == "slide":
        return f"Mean over the second: slide {slide:+.1f} mm/s,\nwith {sep:+.1f} mm/s leaking into separation."
    return (f"Mean over the second: roll {roll:+.1f} deg/s of the 40 commanded;\nthe least squares trades the rest against "
            f"{sep:+.1f} mm/s\nof separation and {slide:+.1f} mm/s of slide.")


def main():
    import hom_contact_rig as HR
    C = setup()
    runs = {name: RF.planar_run(vref, TH0, C, R, T=1.0) for name, _t, vref, _s in CASES}
    lims = view(list(runs.values()), C)
    eqimg = eq_png()
    allf = []
    for name, title, vref, sub in CASES:
        run = runs[name]
        msg = summary(name, run)
        n = len(run)
        idx = [min(n - 1, int(round(i * n / (SLOW * FPS)))) for i in range(int(SLOW * FPS))]
        frames = [frame(run, k, C, lims, title, sub.capitalize(), vref, eqimg, msg) for k in idx]
        frames += [frames[-1]] * int(HOLD_S * FPS)
        out = MEDIA / f"20261110-planar_{name}.mp4"
        HR.write_mp4(frames, out, fps=FPS)
        Image.fromarray(frames[-1]).save(out.with_suffix(".jpg"), quality=88)
        allf += frames
        print(name, msg, flush=True)
    HR.write_mp4(allf, MEDIA / "20261110-planar_motions.mp4", fps=FPS)
    Image.fromarray(allf[len(allf) // 2]).save(MEDIA / "20261110-planar_motions.jpg", quality=88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
