#!/usr/bin/env python3
"""Figures and numbers of the page on the fingertip's calibration on the UR5 (docs/experiments/20261026-fingertip_calibration/):
the six motions of a session, the press targets on the pad, the learning set-up, location error by force level, location
error against the distance to the nearest training press, shear on the slides, and transfer and pooling across tips.

Reads the sessions' analysis outputs in docs/experiments/20261022-ur5_tip_calibration/data/ (holds.npz, summary.json,
checks.json, density_curve.json, pooled_checks.json) and the estimator's surface basis; the Gaussian process is the one
of scripts/ur5_tip_calibration_cb1r_checks.py (board change to contact point and 3D force).

    .venv/bin/python scripts/fingertip_calibration_figures.py [--only NAME ...]
writes media/*.png and data/20261026-numbers.json in the page's folder.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration_cb1r_checks as K  # noqa: E402
import ur5_tip_calibration_density_curve as DC  # noqa: E402

D = ROOT / "docs/experiments/20261026-fingertip_calibration"
DATA, MEDIA = D / "data", D / "media"
NUMBERS = DATA / "20261026-numbers.json"
UR5 = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data"
BASIS = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-b_v2_surface_basis.npz"
SESS = [("20261022-210852-sphere2", "Tip 0, sphere2", "#555555", "o"), ("20261023-203938-cb1t", "Tip 1, cb1t", "#2e5fa8", "s"),
        ("20261025-152920-cb2t", "Tip 2, cb2t", "#b06010", "^")]
NUM = {}
PAD_COL, SPH_COL = "#efe6d6", "#d8dde3"


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                         "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12, "mathtext.fontset": "dejavuserif"})
    return plt


def save(fig, name):
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / name, dpi=150, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("wrote", (MEDIA / name).relative_to(ROOT), flush=True)


# ------------------------------------------------------------------------------------------------ figure: motions
def fig_motions():
    """Six panels: the pad (face down, an arc of its section) over the 5 mm sphere, and the motion of each kind of action."""
    plt = plt_setup()
    from matplotlib.patches import Circle, FancyArrowPatch, Arc, Polygon
    R = 5.0
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 7.4))

    def pad(ax, dx=0.0, dy=0.0, ang=0.0, indent=0.0, alpha=1.0):
        """The pad's section as a filled half-ellipse, face down, apex at (dx, dy - indent), turned by ang (deg) about
        its apex."""
        t = np.linspace(0, math.pi, 120)
        a, b = 6.5, 7.0
        x = a * np.cos(t)
        y = b * np.sin(t)
        y = y - 0.0
        P = np.c_[x, y + 0.0]
        th = math.radians(ang)
        Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        P = P @ Rm.T + np.array([dx, dy - indent])
        ax.add_patch(Polygon(P, closed=True, fc=PAD_COL, ec="#8a6d45", lw=1.4, alpha=alpha, zorder=2))
        return P

    def sphere(ax, cx=0.0, cy=-R):
        ax.add_patch(Circle((cx, cy), R, fc=SPH_COL, ec="#555555", lw=1.2, zorder=1))
        ax.plot([cx], [cy], "+", color="#555555", ms=8, zorder=3)

    def arrow(ax, x0, y0, x1, y1, col="black", lw=2.0):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=lw, color=col), zorder=5)

    def curl(ax, cx, cy, r, t0, t1, col="#c0392b"):
        ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=t0, theta2=t1, color=col, lw=2.2, zorder=5))
        t = math.radians(t1)
        x, y = cx + r * math.cos(t), cy + r * math.sin(t)
        ax.annotate("", xy=(x - 0.01 * math.sin(t) * r, y + 0.01 * math.cos(t) * r), xytext=(x + 0.25 * math.sin(t), y - 0.25 * math.cos(t)),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=col), zorder=5)

    # (a) touch-off: straight down along the normal until the push reaches 0.5 N
    ax = axes[0, 0]
    sphere(ax)
    pad(ax, dy=1.5)
    arrow(ax, 7.5, 6.5, 7.5, 2.5)
    ax.text(7.9, 4.5, "0.15 mm/s\nto 0.5 N", fontsize=11.5, va="center")
    ax.set_title("(a) Touch-off on the top", fontsize=12.5)
    ax.text(0.5, -0.02, "Registers the pad and the sphere:\nonset of the push at zero force", transform=ax.transAxes,
            ha="center", va="top", fontsize=11.5)
    # (b) offset touch-off: 3 mm beside the top, the contact normal tilted
    ax = axes[0, 1]
    sphere(ax)
    pad(ax, dx=3.0, dy=0.6)
    th = math.asin(3.0 / 6.0)
    cx, cy = 3.0 * 0.5, 0.0
    arrow(ax, 8.5, 6.0, 8.5, 2.0)
    ax.plot([0, 6 * math.sin(th) * 1.2], [-R, -R + 6 * math.cos(th) * 1.2], ":", color="#c0392b", lw=1.6, zorder=4)
    ax.set_title("(b) Offset touch-off, 3 mm aside", fontsize=12.5)
    ax.text(0.5, -0.02, "Tilted normal: fixes the place\nalong the finger", transform=ax.transAxes, ha="center", va="top",
            fontsize=11.5)
    # (c) press along the surface normal at force levels
    ax = axes[0, 2]
    sphere(ax)
    pad(ax, indent=0.8)
    arrow(ax, 7.5, 6.5, 7.5, 2.5)
    ax.text(7.9, 4.5, "0.25 to 5 N,\n1.5 s holds", fontsize=11.5, va="center")
    ax.set_title("(c) Press along the normal", fontsize=12.5)
    ax.text(0.5, -0.02, "Board change against contact\npoint and normal force", transform=ax.transAxes, ha="center",
            va="top", fontsize=11.5)
    # (d) slide: turn about the sphere's centre, constant indentation
    ax = axes[1, 0]
    sphere(ax)
    pad(ax, ang=-10.0, indent=0.6, dx=0.0)
    pad(ax, indent=0.6, alpha=0.25)
    ax.plot([0], [-R], "o", color="#c0392b", ms=7, zorder=6)
    curl(ax, 0, -R, 8.4, 70, 110)
    ax.set_title("(d) Slide: turn about the centre", fontsize=12.5)
    ax.text(0.5, -0.02, "Contact slides at constant\nindentation: shear, then slip", transform=ax.transAxes, ha="center",
            va="top", fontsize=11.5)
    # (e) roll: turn about the contact point
    ax = axes[1, 1]
    sphere(ax)
    pad(ax, ang=-12.0, indent=0.6)
    pad(ax, indent=0.6, alpha=0.25)
    ax.plot([0], [-0.6], "o", color="#c0392b", ms=7, zorder=6)
    curl(ax, 0, -0.6, 3.0, 40, 140)
    ax.set_title("(e) Roll: turn about the contact", fontsize=12.5)
    ax.text(0.5, -0.02, "Contact moves over both surfaces;\npresses at new tool orientations", transform=ax.transAxes,
            ha="center", va="top", fontsize=11.5)
    # (f) twist: turn about the normal through the contact
    ax = axes[1, 2]
    sphere(ax)
    pad(ax, indent=0.6)
    ax.plot([0, 0], [-0.6, 9.0], "--", color="#c0392b", lw=1.4, zorder=4)
    from matplotlib.patches import Ellipse
    ax.add_patch(Arc((0, 8.0), 6.0, 1.6, theta1=200, theta2=340, color="#c0392b", lw=2.2, zorder=6))
    ax.annotate("", xy=(2.85, 7.75), xytext=(2.4, 7.5), arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#c0392b"), zorder=6)
    ax.set_title("(f) Twist: turn about the normal", fontsize=12.5)
    ax.text(0.5, -0.02, "Friction torque about the normal:\nhalf the out-and-back difference", transform=ax.transAxes,
            ha="center", va="top", fontsize=11.5)
    for ax in axes.ravel():
        ax.set_xlim(-9.5, 12.5)
        ax.set_ylim(-10.5, 9.5)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout(h_pad=3.2)
    save(fig, "20261026-motions.png")


# ------------------------------------------------------------------------------------------------ figure: coverage
KINDS = [("grid", "Grid presses (pass 2b)", "#2e5fa8", "o"), ("twin", "Twins, mirrored in y (pass 3)", "#b06010", "D"),
         ("dense", "New targets (pass 6)", "#2a9d5c", "^"), ("ref", "Reference presses", "#000000", "*")]


def fig_coverage():
    plt = plt_setup()
    H = K.load(UR5 / "20261025-152920-cb2t")
    P = K.sub(H, H["kind"] == "press")
    ids = P["id"].astype(str)
    b = np.load(BASIS)
    pts, nrm = b["pts"], b["normals"]
    d = DC.nearest(pts, P["p"])
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 5.6), gridspec_kw=dict(width_ratios=(1.0, 1.25, 1.25)))
    views = [("(a) Face, seen from the palm", nrm[:, 0] > 0.3, 1, 2, "y (mm)", "z (mm)"),
             ("(b) +y flank", nrm[:, 1] > 0.3, 0, 2, "x (mm)", "z (mm)"),
             ("(c) −y flank", nrm[:, 1] < -0.3, 0, 2, "x (mm)", "z (mm)")]
    for ax, (title, m, i, j, xl, yl) in zip(axes, views):
        far = m & (d > 1.5)
        ax.scatter(pts[m & ~far, i], pts[m & ~far, j], s=9, c="0.82", zorder=1)
        ax.scatter(pts[far, i], pts[far, j], s=9, c="0.45", zorder=1)
        for key, lab, col, mk in KINDS:
            acts = np.unique(ids[np.char.startswith(ids, key)])
            c = np.array([np.median(P["p"][ids == a], axis=0) for a in acts]) if len(acts) else np.zeros((0, 3))
            nc = np.array([np.median(P["n"][ids == a], axis=0) for a in acts]) if len(acts) else np.zeros((0, 3))
            if len(c):
                sel = (nc[:, 0] > 0.3) if i == 1 else ((nc[:, 1] > 0.3) if "+y" in title else (nc[:, 1] < -0.3))
                ax.scatter(c[sel, i], c[sel, j], s=58 if mk == "*" else 34, marker=mk, c=col, ec="black", lw=0.5, zorder=3,
                           label=lab if ax is axes[0] else None)
        ax.set_title(title, fontsize=12.5)
        ax.set_aspect("equal")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axes[0].scatter([], [], s=9, c="0.45", label="FE surface point over 1.5 mm from any press")
    fig.legend(loc="lower center", ncol=3, fontsize=11.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "20261026-coverage.png")
    NUM["coverage_cb2t"] = {k: int(len(np.unique(ids[np.char.startswith(ids, k)]))) for k, _, _, _ in KINDS}
    NUM["coverage_cb2t"]["fast_grid"] = int(len(np.unique(ids[np.char.startswith(ids, "fast")])))
    NUM["coverage_cb2t"]["rate"] = int(len(np.unique(ids[np.char.startswith(ids, "rate")])))
    NUM["coverage_cb2t"]["press_holds"] = int(len(ids))


# ------------------------------------------------------------------------------------------------ figure: learning set-up
def fig_learning():
    plt = plt_setup()
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="black", lw=1.0))
        ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=12.5, weight="bold")
        ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top", fontsize=11, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.4))

    inp, mdl, out = "#eef3fa", "#fbf1e4", "#eaf5ec"
    box(0.6, 27, 24, 20, "Input", "Board change Δb:\n15 numbers, µT,\nhold mean less the\nrests, per-chip gain\ncorrected", inp)
    box(29, 31, 32, 16, "Gaussian process", "squared-exponential kernel on\nstandardized inputs; one length\nand noise by marginal likelihood", mdl)
    box(29, 12, 32, 16, "MLP ensemble", "5 networks, 2 hidden layers of\n64 SiLU units, AdamW; steps\nchosen on the latest fifth", mdl)
    box(65.5, 22, 33.9, 22, "Output", "contact point p (3, pad frame,\nmoved onto the surface)\nforce f (3): normal and shear\nin the pad frame", out)
    arrow(25.4, 39, 28.6, 39)
    arrow(25.4, 33, 28.6, 22)
    arrow(61.4, 39, 65.1, 36)
    arrow(61.4, 20, 65.1, 28)
    box(0.6, 0.0, 60.4, 10.6, "Training and scoring", "holds grouped by press point; a group is never split\n"
        "between training and test; pooling adds another tip's holds", "#f4f4f4")
    box(65.5, 1.2, 33.9, 17.5, "Physical model + GP", "the GP's input is the physical\nestimate (point and normal force)\n"
        "and its output the estimate's\ncorrection", "#f4f4f4")
    save(fig, "20261026-learning.png")


# ------------------------------------------------------------------------------------------------ figure: error by force
def grouped_cv(P, k=5, seed=0):
    """GP predictions (contact point, 3D force) for every press hold, five folds over groups of points within 0.75 mm."""
    g = DC.groups(P)
    ug = np.random.default_rng(seed).permutation(np.unique(g))
    fold = {u: i % k for i, u in enumerate(ug)}
    fk = np.array([fold[x] for x in g])
    Y = np.full((len(g), 6), np.nan)
    for f in range(k):
        m = fk == f
        Y[m] = K.gp(K.sub(P, ~m)).predict(P["d_board"][m])
    return Y


def fig_by_level():
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    nums = {}
    for sess, name, col, mk in SESS:
        H = K.load(UR5 / sess)
        P = K.sub(H, H["kind"] == "press")
        Y = grouped_cv(P)
        e_gp = K.loc_err(P, Y[:, :3])
        e_ph = K.loc_err(P, P["est_p"])
        lv = np.round(P["level_N"], 2)
        L = [l for l in sorted(np.unique(lv)) if (lv == l).sum() >= 10]
        ax.plot(L, [K.med(e_ph[lv == l]) for l in L], "--" + mk, color=col, mfc="white", ms=6)
        ax.plot(L, [K.med(e_gp[lv == l]) for l in L], "-" + mk, color=col, ms=6, label=name)
        fe = np.abs(Y[:, 3:] @ np.zeros(3))
        fn_hat = -np.sum(Y[:, 3:] * P["n"], 1)
        nums[sess] = dict(gp_by_level={f"{l:g}": round(K.med(e_gp[lv == l]), 2) for l in L},
                          physical_by_level={f"{l:g}": round(K.med(e_ph[lv == l]), 2) for l in L},
                          gp_over_1p5N=round(K.med(e_gp[P["f_n"] >= 1.5]), 3), gp_under_1p5N=round(K.med(e_gp[P["f_n"] < 1.5]), 3),
                          gp_fn_err_N=round(K.med(np.abs(fn_hat - P["f_n"])), 3), n_holds=int(len(lv)),
                          n_groups=int(len(np.unique(DC.groups(P)))))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 3, 5])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "3", "5"])
    ax.set_yticks([0.2, 0.5, 1, 2, 5, 10])
    ax.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10"])
    ax.set_xlabel("Force level of the hold (N)")
    ax.set_ylabel("Location error, median (mm)")
    ax.plot([], [], "-", color="0.3", label="GP, held-out points")
    ax.plot([], [], "--", color="0.3", mfc="white", marker="o", label="Physical model, untrained")
    ax.legend(fontsize=11, frameon=False, ncol=2, loc="upper right")
    ax.grid(True, which="major", color="0.9")
    fig.tight_layout()
    save(fig, "20261026-error_by_level.png")
    NUM["by_level"] = nums


# ------------------------------------------------------------------------------------------------ figure: density
def fig_density():
    plt = plt_setup()
    dc = json.loads((UR5 / "20261025-152920-cb2t/density_curve.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.4))
    bins = list(dc["distance"].items())
    mid = []
    for k, v in bins:
        lo, up = (float(x) for x in k.split("-"))
        mid.append(0.5 * (lo + min(up, 5.0)))
    labels = [k.replace("4-99", "4+") for k, _ in bins]
    x = np.arange(len(bins))
    a1.plot(x, [v["over_1p5N"] for _, v in bins], "-o", color="#2e5fa8", label="1.5 N and over")
    a1.plot(x, [v["under_1p5N"] for _, v in bins], "-s", color="#b06010", mfc="white", label="Under 1.5 N")
    a1.axhline(dc["floor"]["over_1p5N"], color="#2e5fa8", ls=":", lw=1.2)
    a1.text(0.0, dc["floor"]["over_1p5N"] * 1.06, "same point, other speed", fontsize=11, color="#2e5fa8", va="bottom")
    a1.set_xticks(x)
    a1.set_xticklabels(labels, rotation=35, fontsize=11)
    a1.set_yscale("log")
    a1.set_yticks([0.2, 0.5, 1, 2, 5])
    a1.set_yticklabels(["0.2", "0.5", "1", "2", "5"])
    a1.set_xlabel("Distance to the nearest training hold (mm)")
    a1.set_ylabel("Location error, median (mm)")
    a1.set_title("(a) Error against distance", fontsize=13)
    a1.legend(fontsize=11, frameon=False)
    fr = dc["fraction"]
    F = sorted(fr, key=float)
    ng = [fr[f]["n_groups"] for f in F]
    a2.plot(ng, [fr[f]["over_1p5N"] for f in F], "-o", color="#2e5fa8")
    lo = [fr[f]["over_1p5N_seed_range"][0] for f in F]
    hi = [fr[f]["over_1p5N_seed_range"][1] for f in F]
    a2.fill_between(ng, lo, hi, color="#2e5fa8", alpha=0.15, lw=0)
    a2.plot(ng, [fr[f]["under_1p5N"] for f in F], "-s", color="#b06010", mfc="white")
    for f, n in zip(F, ng):
        a2.text(n, fr[f]["over_1p5N"] * 0.86, f"{fr[f]['nearest_mm']:.1f}", fontsize=10.5, ha="center", va="top", color="0.25")
    a2.set_xlabel("Press points in training")
    a2.set_ylabel("Location error, median (mm)")
    a2.set_title("(b) Error against the number of points", fontsize=13)
    a2.text(0.98, 0.95, "numbers: median distance to the\nnearest training hold (mm)", transform=a2.transAxes, ha="right",
            va="top", fontsize=10.5, color="0.25")
    fig.tight_layout()
    save(fig, "20261026-density.png")


# ------------------------------------------------------------------------------------------------ figure: shear
def fig_shear():
    plt = plt_setup()
    H = K.load(UR5 / "20261025-152920-cb2t")
    P = K.sub(H, H["kind"] == "press")
    S = K.sub(H, H["kind"] == "slide")
    n = S["n"]
    ft = K.shear(S["f"], n)
    # predictions: presses only; presses and the other slides (five folds over slides)
    Yp = K.gp(P).predict(S["d_board"])[:, 3:]
    ids = S["id"].astype(str)
    u = np.unique(ids)
    fold = {a: i % 5 for i, a in enumerate(u)}
    fk = np.array([fold[a] for a in ids])
    Ys = np.full((len(ids), 3), np.nan)
    for f in range(5):
        m = fk == f
        Ys[m] = K.gp(K.cat(P, K.sub(S, ~m))).predict(S["d_board"][m])[:, 3:]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.9), sharex=True, sharey=True)
    lim = 1.4
    for ax, (F, title) in zip(axes, ((S["est_f"], "(a) Physical model"), (Yp, "(b) GP trained on presses"),
                                    (Ys, "(c) GP with the other slides"))):
        est = K.shear(F, n)
        ax.scatter(ft.ravel(), est.ravel(), s=3, alpha=0.25, color="#2e5fa8", rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], "-", color="black", lw=0.8)
        e = np.linalg.norm(est - ft, axis=1)
        ax.set_title(title, fontsize=12.5)
        ax.text(0.04, 0.96, f"error {K.med(e):.2f} N\nangle {K.med(K.angle(F, S['f'])):.1f}°", transform=ax.transAxes,
                va="top", fontsize=11.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("HEX-E shear component (N)")
    axes[0].set_ylabel("Estimated shear component (N)")
    fig.tight_layout()
    save(fig, "20261026-shear.png")
    NUM["shear_cb2t"] = dict(n_slides=int(len(u)), n_holds=int(len(ids)), hexe_median_N=K.med(np.linalg.norm(ft, axis=1)))


# ------------------------------------------------------------------------------------------------ figure: transfer, pooling
def fig_transfer():
    plt = plt_setup()
    ck = json.loads((UR5 / "20261025-152920-cb2t/checks.json").read_text())
    pc = json.loads((UR5 / "20261025-114653-cb1r/pooled_checks.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.4), gridspec_kw=dict(width_ratios=(1.0, 1.35)))
    t = ck["transfer"]
    names = ["Physical model,\nrest fit only", "GP trained\non tip 1", "GP trained\non tip 2 itself"]
    v = [t["physical_own_rest"]["over_1p5N"], t["gp_id1"]["over_1p5N"], t["gp_own_5fold"]["over_1p5N"]]
    a1.bar(range(3), v, color=["#888888", "#2e5fa8", "#b06010"], width=0.6)
    for i, y in enumerate(v):
        a1.text(i, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=12)
    a1.set_xticks(range(3))
    a1.set_xticklabels(names, fontsize=11)
    a1.set_ylabel("Location error on tip 2, median (mm)")
    a1.set_title("(a) Tip 2's grid, 1.5 N and over", fontsize=13)
    a1.set_ylim(0, 1.75)
    cv = pc["coverage"]
    keys = [("id1", "Tip 1's presses"), ("id1_and_id2_twins", "+ tip 2's twins"), ("id1_and_id2_presses", "+ all tip 2's presses"),
            ("id2_presses", "Tip 2's presses alone")]
    cols = ["#2e5fa8", "#7fa6d9", "#2a9d5c", "#b06010"]
    w = 0.2
    for j, ((k, lab), c) in enumerate(zip(keys, cols)):
        vals = [cv[k]["left/over_1.5N"]["median_mm"], cv[k]["right/over_1.5N"]["median_mm"]]
        a2.bar(np.arange(2) + (j - 1.5) * w, vals, w, color=c, label=lab)
        for i, y in enumerate(vals):
            a2.text(i + (j - 1.5) * w, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=10)
    a2.set_xticks(range(2))
    a2.set_xticklabels(["Twins on the −y side", "Twins on the +y side"])
    a2.set_ylabel("Location error on tip 1's twins (mm)")
    a2.set_title("(b) Pooling: tip 1's twins, 1.5 N and over", fontsize=13)
    a2.legend(fontsize=10.5, frameon=False, loc="upper left")
    a2.set_ylim(0, 3.1)
    fig.tight_layout()
    save(fig, "20261026-transfer_pooling.png")
    NUM["pooling_shear"] = pc["shear"]
    NUM["transfer"] = t


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    figs = dict(motions=fig_motions, coverage=fig_coverage, learning=fig_learning, by_level=fig_by_level,
                density=fig_density, shear=fig_shear, transfer=fig_transfer)
    if NUMBERS.exists():
        NUM.update(json.loads(NUMBERS.read_text()))
    DATA.mkdir(parents=True, exist_ok=True)
    for k, f in figs.items():
        if a.only is None or k in a.only:
            f()
    NUMBERS.write_text(json.dumps(NUM, indent=1, default=float) + "\n")
    print("wrote", NUMBERS.relative_to(ROOT))


if __name__ == "__main__":
    main()
