#!/usr/bin/env python3
"""Page of the second fingertip against the calibration of the UR5 fingertip (2026-10-23): the two pads' pole patterns and
rest fits, the transfer in simulation, the owner's presses on the desk and each predictor's answers. Reads the two rounds'
results in docs/experiments/20261023-second_fingertip_transfer/data/ (scripts/fingertip_second_tip_transfer.py, from the
saved recordings beside them) and the film of scripts/fingertip_second_tip_film.py; writes
docs/experiments/20261023-second_fingertip_transfer/20261023-second_fingertip_transfer.html.

    .venv/bin/python scripts/fingertip_second_tip_transfer_page.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import fingertip_wrench_model as W  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import ur5_tip_calibration_page as P  # noqa: E402   (esc, code, fig, film, table, plt_setup, TEX_RE)

D = ROOT / "docs/experiments/20261023-second_fingertip_transfer"
DATA = D / "data"
MEDIA = D / "media"
PAGE = D / "20261023-second_fingertip_transfer.html"
TEMPLATE = ROOT / "scripts/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
R1_PATH = DATA / "20261023-round1_hand_held.json"
R2_PATH = DATA / "20261023-round2_fixed_tip.json"
FILM = MEDIA / "20261023-viewer_centre_presses.mp4"
ROBOT_CAL = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data/20261022-210852-sphere2/calib_rest.json"
BASIS = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-b_v2_surface_basis.npz"
REF = np.array([8.475, 0.502, 10.360])
STREAM_ORDER = ("0x0C", "0x11", "0x12", "0x13", "0x10")   # chips in the board's records (reskin5x_plot.ADDRS)
MODELS = [("physical", "Physical model", "#304080", "o"), ("physical+gp", "Physical model with GP correction", "#b06010", "^"),
          ("robot cal", "Robot calibration as it stands", "#777777", "s"), ("gp raw", "GP on the board change", "#000000", "x"),
          ("mlp raw", "MLP on the board change", "#7a3d99", "+")]
NAME = {k: n for k, n, _, _ in MODELS}


def link(p, text):
    return f'<a href="{P.esc(os.path.relpath(p, D))}">{text}</a>'


# ------------------------------------------------------------------------------------------------ figures
def face(ax, basis):
    """The pad seen from the palm: its surface points whose normal faces the palm, in light grey."""
    X, N = basis["pts"], basis["normals"]
    m = N[:, 0] > 0.3
    ax.plot(X[m, 1], X[m, 2], ".", ms=2.5, color="0.82", zorder=0)


def fig_poles(R2, robot, basis, out):
    plt = P.plt_setup()
    from matplotlib.patches import Circle, Rectangle
    own = R2["rest"]["own"]
    cals = [("(a) Robot fingertip of sphere2, board 9c:d0", robot.poles, robot.k, robot.rms_mT),
            ("(b) Second fingertip, board d9:94", own["poles"], own["k"], own["rms_mT"])]
    M = W.Model(robot)
    chips = W.chip_points()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.6), sharey=True)
    for ax, (title, poles, k, rms) in zip(axes, cals):
        face(ax, basis)
        # each chip's label on the side of its square away from the magnets
        place = {"U1": (1.9, -1.1, "left", "center"), "U2": (0.0, 1.8, "center", "bottom"),
                 "U3": (1.8, 0.0, "left", "center"), "U4": (0.0, -1.8, "center", "top"),
                 "U5": (-1.8, 0.0, "right", "center")}
        for (y, z), u in zip(chips[:, 1:], ("U1", "U2", "U3", "U4", "U5")):
            ax.add_patch(Rectangle((y - 1.5, z - 1.5), 3.0, 3.0, fc="white", ec="0.35", lw=1.0, zorder=1))
            dy, dz, ha, va = place[u]
            ax.text(y + dy, z + dz, u, fontsize=11, color="0.35", ha=ha, va=va, zorder=1)
        for (y, z), s in zip(M.mc[:, 1:], poles):
            ax.add_patch(Circle((y, z), 1.5, fc="#c0392b" if s > 0 else "#2e5fa8", ec="black", lw=0.8, zorder=2))
            ax.text(y, z, "N" if s > 0 else "S", ha="center", va="center", color="white", fontsize=11, weight="bold", zorder=3)
        ax.plot(REF[1], REF[2], "*", ms=13, mfc="#f2c200", mec="black", zorder=4)
        ax.set_title(title, fontsize=13)
        ax.text(0.5, -0.17, f"Strength {k:.3f} of N52, rest fit {rms:.2f} mT RMS", transform=ax.transAxes, ha="center",
                fontsize=12)
        ax.set_aspect("equal")
        ax.set_xlim(-10.5, 10.5)
        ax.set_xlabel("y, across the finger (mm)")
    axes[0].set_ylabel("z, from the mount end (mm)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_trace(R2, out):
    plt = P.plt_setup()
    rows = R2["trace"]["rows"]
    t = np.array([r["t"] for r in rows])
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9.0, 7.6), sharex=True, gridspec_kw=dict(height_ratios=(1, 1.1, 1.1)))
    a1.plot(t, [r["change_uT"] for r in rows], "-", lw=1.2, color="0.25")
    for h in R2["holds"]:
        for a in (a1, a2, a3):
            a.axvspan(h["t0_s"], h["t1_s"], color="#e8eef8", zorder=0)
    a1.axhline(R2["trace"]["threshold_uT"], color="0.5", lw=1, ls=":")
    a1.set_ylabel("Board change (µT)")
    for key in ("physical", "physical+gp", "robot cal"):
        _, name, col, mk = next(m for m in MODELS if m[0] == key)
        tt = [r["t"] for r in rows if key in r]
        a2.plot(tt, [r[key]["f_n"] for r in rows if key in r], mk, ms=4, color=col, label=name)
    a2.set_ylabel("Normal force (N)")
    for key, ls in (("physical", "o"), ("physical+gp", "^")):
        col = next(m[2] for m in MODELS if m[0] == key)
        tt = [r["t"] for r in rows if key in r]
        a3.plot(tt, [r[key]["p"][1] for r in rows if key in r], ls, ms=4, color=col, mfc="white")
        a3.plot(tt, [r[key]["p"][2] for r in rows if key in r], ls, ms=4, color=col)
    a3.axhline(REF[1], color="0.4", lw=1, ls=":")
    a3.axhline(REF[2], color="0.4", lw=1, ls="--")
    a3.text(t[0] + 3.0, REF[2] + 0.4, "reference point's z", fontsize=11, color="0.3")
    a3.text(t[0] + 3.0, REF[1] + 0.4, "reference point's y", fontsize=11, color="0.3")
    a3.set_ylim(-4.5, 13.0)
    a3.set_ylabel("Contact point (mm)")
    a3.set_xlabel("Time after 12:27:20 (s)")
    a3.set_xlim(t[0], t[-1])
    for a, k in zip((a1, a2, a3), "abc"):
        a.text(0.005, 0.97, f"({k})", transform=a.transAxes, va="top", fontsize=13)
    h, lab = a2.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=11, frameon=False)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_points(R2, basis, out):
    plt = P.plt_setup()
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(6.6, 7.0))
    face(ax, basis)
    for (y, z) in W.chip_points()[:, 1:]:
        ax.add_patch(Rectangle((y - 1.5, z - 1.5), 3.0, 3.0, fc="none", ec="0.65", lw=0.8, ls="--", zorder=1))
    H = R2["holds"]
    for h in H:
        a, b = h["est"]["physical"]["p"], h["est"]["physical+gp"]["p"]
        ax.plot([a[1], b[1]], [a[2], b[2]], "-", lw=0.8, color="#b06010", alpha=0.6, zorder=2)
    for key, name, col, mk in MODELS:
        pts = np.array([h["est"][key]["p"] for h in H])
        ok = np.all(np.isfinite(pts), axis=1)
        kw = dict(mfc="white", mew=1.4) if mk in ("o", "^", "s") else dict(mew=1.6)
        ax.plot(pts[ok, 1], pts[ok, 2], mk, ms=8, color=col, label=name, zorder=3, **kw)
    ax.plot(REF[1], REF[2], "*", ms=16, mfc="#f2c200", mec="black", zorder=4, label="Reference point")
    ax.set_aspect("equal")
    ax.set_xlim(-10.5, 10.5)
    ax.set_xlabel("y, across the finger (mm)")
    ax.set_ylabel("z, from the mount end (mm)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_step(R1, out):
    plt = P.plt_setup()
    st = R1["step"]
    x = np.arange(15)
    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    w = 0.27
    ax.bar(x - w, st["step_uT"], w, color="0.55", label="Measured")
    ax.bar(x, st["uniform"]["fit_uT"], w, color="#304080", label=f"Uniform field of {st['uniform']['g_norm_uT']:.0f} µT")
    ax.bar(x + w, st["offset"]["fit_uT"], w, color="#b03030", label="Rigid offset of the pad")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} {a}" for c in STREAM_ORDER for a in "xyz"], rotation=60, ha="right", fontsize=11)
    ax.set_ylabel("Change of the rest (µT)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=3, fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ page
def build():
    from fingertip_second_tip_transfer_text import sections          # the prose, kept apart from the plotting
    R1, R2 = json.loads(R1_PATH.read_text()), json.loads(R2_PATH.read_text())
    robot = W.Calib.load(ROBOT_CAL)
    basis = W.load_basis(BASIS)
    MEDIA.mkdir(parents=True, exist_ok=True)
    F = dict(poles=MEDIA / "20261023-pole_patterns.png", trace=MEDIA / "20261023-round2_trace.png",
             points=MEDIA / "20261023-round2_points.png", step=MEDIA / "20261023-round1_rest_step.png",
             film=FILM, poster=FILM.with_suffix(".png"))
    fig_poles(R2, robot, basis, F["poles"])
    fig_trace(R2, F["trace"])
    fig_points(R2, basis, F["points"])
    fig_step(R1, F["step"])
    S, lede, summary, film_html = sections(R1, R2, robot, F, P, link, NAME)
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "Second fingertip transfer",
                 "H1": "Transfer of the UR5 fingertip&#8217;s calibration to a second fingertip with two magnets reversed",
                 "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-23", "LEDE": lede,
                 "BYLINE": ("Presses by the owner on the workstation&#8217;s board, 2026-10-23 12:15 to 12:28. Analysis from the "
                            "saved recordings by " + P.code("scripts/fingertip_second_tip_transfer.py") + "; page by "
                            + P.code("scripts/fingertip_second_tip_transfer_page.py") + "."),
                 "FILM": film_html, "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = P.TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary, date="2026-10-23")
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    build()
