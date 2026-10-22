#!/usr/bin/env python3
"""Result page of the model-based contact estimation study (2026-10-20): the measurement model of the five-chip board under
the magnetic pad, its calibration at rest, the FE contact basis over the pad's surface, the estimator's errors on simulated
presses, its behaviour without a press, the owner's first presses by hand, the drift of the rest reading with the chips'
temperature and the pad's seat (2026-10-21), and the live browser viewer.

Reads docs/experiments/20261020-fingertip_contact_estimation/ (results.json and data/20261020-uncertainty_maps.npz from
scripts/fingertip_contact_estimation_study.py; media/ from scripts/fingertip_contact_film.py and the viewer), draws the
figures into media/, renders the mathematics with scripts/texsvg.py, applies scripts/retro_style.py and writes
docs/experiments/20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html.

    .venv/bin/python scripts/fingertip_contact_estimation_page.py
"""
from __future__ import annotations

import base64
import html
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = ROOT / "docs/experiments/20261020-fingertip_contact_estimation"
MEDIA = D / "media"
PAGE = D / "20261020-model_based_contact_estimation.html"
TEMPLATE = ROOT / "scripts/fingertip_contact_estimation_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
FE_PAGE = ROOT / "docs/experiments/20261019-v2_tip_fe_checks/20261019-v2_fingertip_stiffness_checks.html"
STATUS_PAGE = ROOT / "docs/experiments/20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html"
ADDR = ("0x0C", "0x11", "0x12", "0x13", "0x10")
U = ("U1", "U2", "U3", "U4", "U5")


def esc(s):
    return html.escape(str(s), quote=True)


def rel(p):
    import os
    return os.path.relpath(p, D)


def link(p, text):
    return f'<a href="{esc(rel(p))}">{text}</a>'


def code(p):
    return f"<code>{esc(p)}</code>"


def img_uri(png, width=1100, quality=88):
    from PIL import Image
    im = Image.open(png).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def fig(n, png, caption, width=1100):
    return (f'<figure id="fig{n}"><img alt="Figure {n}" src="{img_uri(png, width)}">'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def film(n, mp4, poster, caption):
    v = "data:video/mp4;base64," + base64.b64encode(Path(mp4).read_bytes()).decode()
    return (f'<figure id="fig{n}"><video controls muted loop playsinline preload="metadata" poster="{img_uri(poster, 1100)}" '
            f'src="{v}" style="width:100%"></video><figcaption>Figure {n}. {caption}</figcaption></figure>')


def table(head, body, num_cols=(), caption=None, n=None):
    cap = f"<caption>Table {n}. {caption}</caption>" if caption else ""
    th = "".join(f'<th{" class=n" if k in num_cols else ""}>{h}</th>' for k, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td{" class=n" if k in num_cols else ""}>{c}</td>' for k, c in enumerate(r)) + "</tr>"
                 for r in body)
    return f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


# ------------------------------------------------------------------------------------------------ figures

def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                         "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12})
    return plt


def fig_rest(res, out):
    plt = plt_setup()
    c = res["calibration"]
    meas, mod = np.array(c["measured_uT"]) / 1e3, np.array(c["model_uT"]) / 1e3
    perm = c["best"]["perm"]
    fig, ax = plt.subplots(figsize=(11, 4.0), dpi=100)
    x = np.arange(15)
    ax.bar(x - 0.2, meas.ravel(), 0.4, color="#303030", label="Measured")
    ax.bar(x + 0.2, mod.ravel(), 0.4, color="#d08040", label="Model, best fit")
    ax.set_xticks(x, [a for i in range(5) for a in "xyz"])
    for i in range(5):
        ax.text(3 * i + 1, -0.16, f"{ADDR[i]} at {U[perm[i]]}", transform=ax.get_xaxis_transform(), ha="center", fontsize=13)
        if i:
            ax.axvline(3 * i - 0.5, color="#999", lw=0.8)
    ax.axhline(0, color="#000", lw=0.8)
    ax.set_ylabel("Field at rest (mT), chip axes")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_quiet(res, out):
    plt = plt_setup()
    q = res["no_contact"]
    S, G = np.array(q["series"]), np.array(q["g_series"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.2), dpi=100, sharex=True)
    a1.plot(S[:, 0], S[:, 1], color="#303030", lw=1, label="Explained chi-square of the best contact")
    a1.axhline(q["threshold"], color="#c00000", lw=1.2, ls="--", label=f"Threshold {q['threshold']:.0f}")
    a1.set_ylabel("Chi-square")
    a1.legend(loc="upper left")
    a1.grid(alpha=0.3)
    for k, (lbl, col) in enumerate((("x", "#c00000"), ("y", "#008000"), ("z", "#0040ff"))):
        a2.plot(S[:, 0], G[:, k], color=col, lw=1, label=f"Background change along {lbl}")
    a2.set_ylabel("Background (uT)")
    a2.set_ylim(min(-2.5, float(G.min()) - 0.5), float(G.max()) + 3.0)
    a2.set_xlabel("Time (s)")
    a2.legend(loc="upper left", ncol=3)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_surface_maps(npz, out):
    """Model 1-sigma of the press for 1 N along the normal at every pad point, seen from the palm and from both sides."""
    plt = plt_setup()
    from matplotlib.colors import LogNorm
    z = np.load(npz)
    P, N, tri = z["pts"], z["normals"], z["tris"]
    tn = N[tri].mean(1)
    views = (("Palmar face, seen from the palm", 0, 1, (2, 1), "y (mm)"),
             ("+y flank, seen from +y", 1, 1, (2, 0), "x (mm)"),
             ("-y flank, mirrored", 1, -1, (2, 0), "x (mm)"))
    fig, axs = plt.subplots(2, 3, figsize=(14, 6.4), dpi=100, gridspec_kw=dict(wspace=0.32))
    for r, (key, lbl) in enumerate((("as_found_press", "Magnets as found"), ("n52_press", "N52, poles one way"))):
        vt = np.nanmean(z[key][tri], axis=1)
        lo, hi = (float(np.nanpercentile(vt, q)) for q in (2, 98))
        for c, (title, ax_i, sgn, (iu, iv), yl) in enumerate(views):
            ax = axs[r, c]
            vis = (sgn * tn[:, ax_i] > 0.25) & np.isfinite(vt)
            im = ax.tripcolor(P[:, iu], P[:, iv], tri[vis], facecolors=np.clip(vt[vis], lo, hi), cmap="viridis",
                              norm=LogNorm(vmin=lo, vmax=hi), edgecolors="none")
            if c == 0:
                ax.scatter(z["magnets"][:, 2], z["magnets"][:, 1], marker="o", s=70, facecolor="none", edgecolor="#f04040", lw=1.5)
                ax.scatter(z["chips"][:, 2], z["chips"][:, 1], marker="s", s=70, facecolor="none", edgecolor="#fff", lw=1.5)
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(title, fontsize=13)
            ax.set_xlabel("z, along the finger (mm)")
            ax.set_ylabel(f"{lbl}\n{yl}" if c == 0 else yl)
        cb = fig.colorbar(im, ax=list(axs[r]), shrink=0.9, pad=0.01)
        cb.set_label("Press 1-sigma (N)")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_session(res, out):
    """The owner's presses by hand: reading change, the estimate, and the last rest's offset against the rigid fit."""
    plt = plt_setup()
    s = res["session"]
    T = np.array(s["trace"], float)
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 9.6), dpi=100, gridspec_kw=dict(height_ratios=[1.15, 1.0, 1.1]))
    a1.plot(T[:, 0], T[:, 1], color="#000", lw=1.2)
    for q in s["rests"]:
        a1.axvspan(q["t0"], q["t1"], color="#c8c8c8", alpha=0.6, lw=0)
        if q["offset_uT"] > 5:
            a1.text(0.5 * (q["t0"] + q["t1"]), q["offset_uT"] + 14, f"rest: {q['offset_uT']:.0f} uT", ha="center", fontsize=12)
    for k, q in enumerate(s["peaks"], 1):
        i = int(np.argmin(np.abs(T[:, 0] - q["t"])))
        a1.text(q["t"], T[i, 1] + 10, f"press {k}", ha="center", fontsize=12)
    a1.set_ylabel("Reading change (uT)")
    a1.set_ylim(0, T[:, 1].max() * 1.18)
    a1.grid(alpha=0.3)
    a1.set_title("Norm of the 15-component change from the first rest; grey: rests", fontsize=12)
    a2.plot(T[:, 0], T[:, 2], color="#c00000", lw=1.1, label="Pressing force")
    a2.plot(T[:, 0], T[:, 3], color="#0040ff", lw=1.1, label="Shear magnitude")
    for q in s["rests"]:
        a2.axvspan(q["t0"], q["t1"], color="#c8c8c8", alpha=0.6, lw=0)
    a2.set_ylabel("Estimate (N)")
    a2.set_xlabel("Time (s)")
    a2.legend(loc="upper left")
    a2.grid(alpha=0.3)
    last = s["rests"][-1]
    m, fit = np.array(last["components_uT"]), np.array(last["rigid_fit_uT"])
    x = np.arange(15)
    a3.bar(x - 0.2, m, 0.4, color="#303030", label=f"Offset of the rest at {last['t0']:.0f}-{last['t1']:.0f} s")
    a3.bar(x + 0.2, fit, 0.4, color="#d08040", label="Rigid motion of the whole pad, fitted")
    a3.set_xticks(x, [c for _ in range(5) for c in "xyz"])
    for i in range(5):
        a3.text(3 * i + 1, -0.2, ADDR[i], transform=a3.get_xaxis_transform(), ha="center", fontsize=13)
        if i:
            a3.axvline(3 * i - 0.5, color="#999", lw=0.8)
    a3.axhline(0, color="#000", lw=0.8)
    a3.set_ylabel("Offset (uT), chip axes")
    a3.legend(loc="lower left")
    a3.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _clock_ticks(ax, start_h, t_s, every_h=2):
    """Ticks in clock time on an axis of seconds from a start at start_h hours after midnight."""
    first = int(np.ceil(start_h))
    hours = np.arange(first, start_h + t_s.max() / 3600 + 1e-9, every_h)
    ax.set_xticks((hours - start_h) * 3600, [f"{int(h) % 24:02d}:00" for h in hours])


def fig_drift(res, out):
    """The night and the warm-up: the chips' temperatures and the rest reading's change, uncorrected and corrected."""
    plt = plt_setup()
    import fingertip_wrench_model as W
    th = res["thermal"]
    kap = np.array(th["kappa_per_C"])
    zn, zw = np.load(D / "data/20261021-overnight_drift_60s.npz"), np.load(D / "data/20261021-warmup_1s.npz")
    cols = ("#c00000", "#d08000", "#008000", "#0040ff", "#8000a0")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.6), dpi=100, gridspec_kw=dict(height_ratios=[1, 1.2]))
    q = zn["t"] > 1800
    t, T, B = zn["t"][q], zn["T"][q], zn["B"][q]
    C = W.thermal_correct(B.reshape(-1, 5, 3), T, kap).reshape(-1, 15)
    for i in range(5):
        ax[0, 0].plot(t, T[:, i], color=cols[i], lw=1.2, label=ADDR[i])
    ax[0, 0].set_ylabel("Die temperature (\u00b0C)")
    ax[0, 0].legend(loc="upper right", ncol=3, fontsize=11)
    ax[0, 0].set_title("Night of 20 October, pad untouched", fontsize=13)
    ax[1, 0].plot(t, B[:, 2] - B[:5, 2].mean(), color="#909090", lw=1.6, label="Uncorrected")
    ax[1, 0].plot(t, C[:, 2] - C[:5, 2].mean(), color="#000", lw=1.6, label="Corrected (Eq. 9)")
    ax[1, 0].set_ylabel("Change of 0x0C's z reading (\u00b5T)")
    ax[1, 0].legend(loc="upper right")
    start_h = 20 + 6 / 60 + 47 / 3600
    for a in ax[:, 0]:
        _clock_ticks(a, start_h, zn["t"])
        a.set_xlim(t[0], t[-1])
        a.grid(alpha=0.3)
    ok = np.isfinite(zw["B"]).all(1) & (zw["t"] > 2) & (zw["t"] < 300)
    tw, Tw, Bw = zw["t"][ok], zw["T"][ok], zw["B"][ok]
    Cw = W.thermal_correct(Bw.reshape(-1, 5, 3), Tw, kap).reshape(-1, 15)
    for i in range(5):
        ax[0, 1].plot(tw, Tw[:, i] - Tw[:8, i].mean(), color=cols[i], lw=1.2, label=ADDR[i])
    ax[0, 1].set_ylabel("Temperature rise (\u00b0C)")
    ax[0, 1].set_title("Warm-up after plugging in, 21 October 09:59", fontsize=13)
    ax[1, 1].plot(tw, np.linalg.norm(Bw - Bw[:8].mean(0), axis=1), color="#909090", lw=1.6, label="Uncorrected")
    ax[1, 1].plot(tw, np.linalg.norm(Cw - Cw[:8].mean(0), axis=1), color="#000", lw=1.6, label="Corrected (Eq. 9)")
    ax[1, 1].set_ylabel("Change of the rest reading (\u00b5T)")
    ax[1, 1].set_xlabel("Time from the viewer's start (s)")
    ax[1, 1].set_ylim(0, 150)
    for x0, lbl in ((150, "touch"), (228, "touch")):
        ax[1, 1].axvline(x0, color="#c00000", lw=1, ls="--")
        ax[1, 1].text(x0 + 3, 138, lbl, color="#c00000", fontsize=12)
    ax[1, 1].legend(loc="upper left")
    for a in ax[:, 1]:
        a.set_xlim(0, 300)
        a.grid(alpha=0.3)
    ax[1, 0].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_rezero(res, out):
    """The shown force over the recordings by hand, baseline kept and with the re-zero at rest."""
    plt = plt_setup()
    z = np.load(D / "data/20261021-rezero_replay.npz")
    rp = res["thermal"]["replay"]
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.8), dpi=100)
    for a, (key, name, title) in zip(axs, (("0959_touches", "09:59 touches", "Touches of 21 October from 09:59 (the warm-up of Figure 7)"),
                                           ("2006_presses", "20:06 presses", "Presses of 20 October from 20:06, baseline from the rest at 8-10 s"))):
        K, A, Z = z[key + "_kept_trace"], z[key + "_auto_trace"], z[key + "_auto_zeros"]
        a.plot(K[:, 0], K[:, 2], color="#e08080", lw=1.0,
               label=f"Baseline kept: contact shown {rp[name]['baseline kept']['shown_share_pct']:.0f} % of frames")
        a.plot(A[:, 0], A[:, 2], color="#000", lw=1.0,
               label=f"Re-zero after 3 s at rest: {rp[name]['auto-zero']['shown_share_pct']:.0f} %")
        top = max(float(K[:, 2].max()), float(A[:, 2].max()), 1.0)
        big = Z[:, 1] > 26
        a.plot(Z[~big, 0], np.full((~big).sum(), -0.04 * top), "|", color="#0040ff", ms=7)
        a.plot(Z[big, 0], np.full(big.sum(), -0.04 * top), "|", color="#0040ff", ms=16, mew=2,
               label="Re-zeros (tall: more than 26 \u00b5T, about 1 N on the face)")
        a.set_ylim(-0.1 * top, top * 1.25)
        a.set_xlim(K[0, 0], K[-1, 0])
        a.set_ylabel("Pressing force shown (N)")
        a.set_title(title, fontsize=13)
        a.legend(loc="upper right", fontsize=11)
        a.grid(alpha=0.3)
    axs[-1].set_xlabel("Time from the recording's start (s)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_poles(res, out):
    """The neighbouring fingertip's field at the board, per pole pattern, against distance."""
    plt = plt_setup()
    po = res["poles"]
    cols = {"all one way": "#000", "checkerboard": "#c00000", "centre reversed": "#0040ff", "rows opposite": "#008000"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.9), dpi=100, sharey=True)
    for a, cfg, xl in ((axs[0], "facing", "Gap between the two palmar faces (mm)"),
                       (axs[1], "side", "Spacing of the two fingertips across the finger (mm)")):
        for name, v in po.items():
            c = v["neighbour"][cfg]
            a.semilogy(c["mm"], c["nonuniform_uT"], color=cols[name], lw=1.8, label=f"{name[0].upper() + name[1:]} (net {v['net']:+d})")
            a.semilogy(c["mm"], c["moving5_uT"], color=cols[name], lw=1.2, ls="--")
        lo = min(v["press_uT_per_N"]["face"] for v in po.values())
        hi = max(v["press_uT_per_N"]["face"] for v in po.values())
        a.axhspan(lo, hi, color="#d0d0d0", alpha=0.6, lw=0)
        a.text(c["mm"][-1], hi * 1.12, "1 N press on the face", ha="right", fontsize=12)
        a.set_xlabel(xl)
        a.grid(alpha=0.3, which="both")
        a.set_title("Palmar faces toward each other" if cfg == "facing" else "Side by side, turned the same way", fontsize=13)
    axs[0].set_ylabel("Field at the board not uniform (\u00b5T)")
    axs[0].legend(loc="upper right", fontsize=11)
    axs[1].text(0.98, 0.02, "dashed: change for 5 mm more", transform=axs[1].transAxes, ha="right", va="bottom", fontsize=12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ page

def build():
    res = json.loads((D / "results.json").read_text())
    MEDIA.mkdir(exist_ok=True)
    fig_rest(res, MEDIA / "20261020-rest_fit.png")
    fig_surface_maps(D / "data/20261020-surface_uncertainty.npz", MEDIA / "20261020-surface_uncertainty.png")
    nc = dict(res["surface"]["as found"]["no_contact"], threshold=25.0)
    fig_quiet({"no_contact": nc}, MEDIA / "20261020-no_press.png")
    fig_session(res, MEDIA / "20261020-presses_by_hand.png")
    fig_drift(res, MEDIA / "20261021-rest_drift.png")
    fig_rezero(res, MEDIA / "20261021-rezero_replay.png")
    fig_poles(res, MEDIA / "20261022-pole_patterns.png")
    th = res["thermal"]
    c, fe, fs = res["calibration"], res["fe"], res["fe_surface"]
    b = c["best"]
    syn = {(s["magnets"], s["window"]): s for s in res["synthetic"]}
    n52f = syn[("N52, poles one way", "50 ms")]
    sf = res["surface"]
    s52, saf = sf["N52, poles one way"]["regions"], sf["as found"]["regions"]
    m52, maf = sf["N52, poles one way"]["maps"], sf["as found"]["maps"]
    q = sf["as found"]["no_contact"]
    se = res["session"]
    rests = se["rests"]
    big = [r for r in rests if r["offset_uT"] > 30]
    fmin, fmax = min(c["field_mT"]), max(c["field_mT"])
    perm = b["perm"]
    assign = ", ".join(f"{ADDR[i]} at {U[perm[i]]}" for i in range(5))
    sr = res["synthetic_recording"]
    fa = fs["face_agreement"]

    sh = [np.linalg.norm(r["translation_um"]) for r in big]
    tl = [np.linalg.norm(r["tilt_mrad"]) for r in big]
    sw = th["floor_sweep"]
    old_rest = [r_ for r_ in sw if r_["floor_uT"] == 0][0]
    k_lo, k_hi = -100 * max(th["kappa_per_C"]), -100 * min(th["kappa_per_C"])
    seat = th["seat_0959"]
    moved = [q_ for q_ in seat if q_["offset_uT"] > 20]
    before = th["shown"]
    lede = (f"A model of the five-chip board, calibrated at rest, turns the 15 field components into a contact point and "
            f"force anywhere on the pad's surface through an FE basis at the {fs['pads']} sphere pads of the simulations; on "
            f"simulated presses with N52 magnets the median errors are {s52['all']['pos_median_mm']:.2f}&#8202;mm and "
            f"{s52['all']['press_abs_median_N']:.2f}&#8202;N ({saf['all']['pos_median_mm']:.2f}&#8202;mm and "
            f"{saf['all']['press_abs_median_N']:.2f}&#8202;N with today's magnets). On the board the rest reading drifts by "
            f"{k_lo:.2f}&#8211;{k_hi:.2f}&#8202;% of the field per &#176;C of chip temperature and moves by "
            f"{min(q_['offset_uT'] for q_ in moved):.0f}&#8211;{max(r['offset_uT'] for r in big):.0f}&#8202;&#181;T when a "
            f"touch shifts the pad in its seat. A temperature correction, a {th['floor_uT']:.0f}&#8202;&#181;T noise floor and "
            f"a re-zero after 3&#8202;s at rest stop the viewer showing a contact at rest, which it did "
            f"{before['warm-up, uncorrected, floor 0 uT']['share_pct']:.0f}&#8211;"
            f"{before['night, uncorrected, floor 0 uT']['share_pct']:.0f}&#8202;% of the time. No estimate has been compared "
            f"with a reference force.")

    terms = ('<dl class="terms">'
             "<dt>Tip frame</dt><dd>$x$ along the pad's normal toward the palmar face, $y$ across the finger, $z$ along it "
             "from the mount toward the tip end, in mm (the FE's frame). The palmar face spans about $y=-6$ to $6$ and "
             "$z=1$ to $20$&#8202;mm at $x$ up to 8.5&#8202;mm; the pad's cut face, on the board, lies at $x=-4.95$&#8202;mm.</dd>"
             "<dt>Surface point $\\mathbf p$, normal $\\mathbf n$, tangents $\\mathbf t_1$, $\\mathbf t_2$</dt><dd>A point "
             "of the pad's free surface, its outward unit normal, and two unit tangents: $\\mathbf t_1$ is $\\mathbf e_z$ "
             "projected on the tangent plane ($\\mathbf e_y$ where the normal lies within 25&#176; of $z$), "
             "$\\mathbf t_2=\\mathbf n\\times\\mathbf t_1$.</dd>"
             "<dt>Regions</dt><dd>Face: $n_x>0.7$; edge: $0.3\\le n_x\\le0.7$; sides: $n_x<0.3$ (the two flanks and the "
             "tip end).</dd>"
             "<dt>Reading, reading change</dt><dd>The 15 field components the five chips report, each in its own axes, "
             "in &#181;T; the reading change $\\Delta\\mathbf b$ is the reading minus a baseline taken at rest.</dd>"
             "<dt>Noise per sample</dt><dd>Standard deviation of one conversion with the pad at rest: 1.65&#8202;&#181;T "
             "on a chip's $x$ and $y$ axes and 2.57&#8202;&#181;T on $z$ (GAIN_SEL 1, RES 1, DIG_FILT 5; a conversion "
             "every 7.0&#8211;7.1&#8202;ms). The viewer averages the 50&#8202;ms before each frame, about 7 conversions, "
             "which divides the noise by $\\sqrt 7$.</dd>"
             "<dt>Magnet strength $k$</dt><dd>A magnet's moment relative to a 3&#8202;&#215;&#8202;2&#8202;mm N52 cylinder "
             "(remanence 1.45&#8202;T).</dd>"
             "<dt>Rest residual</dt><dd>RMS over the 15 components of the measured minus the modelled reading at rest, "
             "in mT.</dd>"
             "<dt>Sensitivity $S(\\mathbf p)$</dt><dd>The reading change per newton of a patch load centred at surface "
             "point $\\mathbf p$: three columns, for loads along $x$, $y$ and $z$.</dd>"
             "<dt>Pressing force $f_n$, shear $\\mathbf f_t$</dt><dd>The force $\\mathbf f$ the object applies to the pad, "
             "split into $f_n=-\\mathbf f\\cdot\\mathbf n$ and its components along $\\mathbf t_1$, $\\mathbf t_2$, in N.</dd>"
             "<dt>Explained chi-square $e$</dt><dd>The decrease of the weighted squared residual when the estimated contact "
             "is added (Eq. 5). The viewer shows a contact when $e>25$ and the pressing force is at least 0.25&#8202;N "
             "(Eq. 11; until 2026-10-20, $e>25$ alone).</dd>"
             "<dt>Die temperature $T_c$</dt><dd>The temperature chip $c$ reports with each conversion, in &#176;C (noise "
             "0.10&#8211;0.11&#8202;&#176;C a conversion); the viewer averages it over the last 3&#8202;s.</dd>"
             "<dt>Gain coefficient $\\kappa_c$</dt><dd>The relative change of chip $c$'s reading per &#176;C of its die "
             "temperature, fitted on a night at rest (Eq. 9), in % per &#176;C.</dd>"
             "<dt>Noise floor $\\sigma_{\\mathrm{fl}}$</dt><dd>A standard deviation added to the noise of every component "
             "in the estimator, in &#181;T, which averaging over the 50&#8202;ms window does not reduce (Eq. 10).</dd>"
             "<dt>Smallest press shown</dt><dd>The pressing force along the normal at which a press's explained "
             "chi-square reaches 25 (from its value at 1&#8202;N, since $e$ grows with the square of the force), or "
             "0.25&#8202;N where that is larger; median over the candidate points of a region.</dd>"
             "<dt>Stillness $s$, re-zero</dt><dd>$s$: the norm over the 15 components of the standard deviation of the "
             "reading's ten 0.1&#8202;s means over the last second, in &#181;T (2.2&#8202;&#181;T at rest from the chips' "
             "noise). A re-zero: a new baseline the viewer takes by itself once $s$ has stayed below 3&#8202;&#181;T for "
             "3&#8202;s (Eq. 12).</dd>"
             "<dt>Model 1-sigma</dt><dd>Square root of a diagonal entry of the inverse Fisher information at the estimate "
             "(Eq. 6): the spread the noise alone causes when the model is exact. The position 1-sigma is the geometric "
             "mean of those along $\\mathbf t_1$ and $\\mathbf t_2$.</dd>"
             "<dt>Simulated press</dt><dd>A reading made by the model itself: a point uniform by area over the surface, "
             "$f_n$ uniform in 0.3&#8211;3&#8202;N, shear normal with 0.3&#8202;N standard deviation along each tangent, a "
             "background change of 5&#8202;&#181;T, a strength change of 0.1&#8202;% and the noise of 50&#8202;ms.</dd>"
             "<dt>Placed away</dt><dd>A simulated press whose estimated point lies more than 1.5&#8202;mm from the true one; "
             "the position and force errors in the tables leave these out and the tables give their share.</dd>"
             "<dt>Background change $\\Delta\\mathbf g$, strength change $\\varepsilon$</dt><dd>A uniform field change in "
             "the tip frame (the finger turning in Earth's field), in &#181;T; a relative change of the magnets' strength "
             "(N52 loses about 0.12&#8202;% per &#176;C).</dd>"
             "<dt>Rest offset</dt><dd>The mean reading over a stretch with nothing pressing (a rest) minus the mean "
             "reading of the first rest, in &#181;T (norm over 15 components unless a chip is named).</dd></dl>")

    eq1 = (r"$$\mathbf B(\mathbf p)=\sum_{j=1}^{5}\sum_{q} w_q\,\frac{\mu_0}{4\pi}\,"
           r"\frac{3\,\hat{\mathbf r}_{jq}(\hat{\mathbf r}_{jq}\!\cdot\mathbf m_j)-\mathbf m_j}{\lVert\mathbf r_{jq}\rVert^{3}},\qquad"
           r"\mathbf r_{jq}=\mathbf p-\mathbf c_j-\boldsymbol\rho_q,\qquad \mathbf m_j=k\,s_j\,\frac{B_r V}{\mu_0}\,\mathbf e_x$$")
    eq2 = (r"$$\mathbf b_c=R_c\big(\mathbf B(\mathbf p_c)+\mathbf g\big),\qquad \mathbf z_c=s\,\mathbf e_x,\qquad"
           r"\mathbf x_c=\cos(\theta_0+\phi_c)\,\mathbf e_y+\sin(\theta_0+\phi_c)\,\mathbf e_z,\qquad \mathbf y_c=\mathbf z_c\times\mathbf x_c$$")
    eq3 = (r"$$\min_{s_j,\;\pi,\;s,\;\theta_0,\;k,\;\boldsymbol\delta,\;\mathbf g}\;\sum_{i=1}^{5}\Big\lVert \mathbf b_i^{\,\mathrm{rest}}"
           r"-R_{\pi(i)}\big(\mathbf B(\mathbf p_{\pi(i)}-\boldsymbol\delta)+\mathbf g\big)\Big\rVert^{2},\qquad \lvert g_x\rvert,"
           r"\lvert g_y\rvert,\lvert g_z\rvert\le 100\ \mu\mathrm T$$")
    eq4 = (r"$$\Delta\mathbf b=S(\mathbf p)\,\mathbf f+N\mathbf q+\mathbf n,\qquad S(\mathbf p)=\frac{\partial\mathbf b}{\partial\mathbf m}"
           r"\,M(\mathbf p),\qquad N=\begin{bmatrix}U&\mathbf b^{\mathrm{rest}}\end{bmatrix},\quad U=\begin{bmatrix}R_1\\ \vdots\\ R_5\end{bmatrix},"
           r"\qquad \mathbf q\sim\mathcal N(0,P),\quad \mathbf f\sim\mathcal N(0,\sigma_f^2 I),\quad \mathbf n\sim\mathcal N(0,\Sigma)$$")
    eq5 = (r"$$C=\Sigma+NPN^{\top},\quad G=S^{\top}C^{-1}S,\quad \mathbf t=S^{\top}C^{-1}\Delta\mathbf b,\quad"
           r"\hat{\mathbf f}=\big(G+\sigma_f^{-2}I\big)^{-1}\mathbf t,\quad e=\mathbf t^{\top}\hat{\mathbf f},\quad"
           r"\ell(\mathbf p)=e-\ln\det\big(I+\sigma_f^{2}G\big)$$")
    eq5b = (r"$$\hat{\mathbf p}=\arg\max_{\mathbf p}\ \ell(\mathbf p)\quad\text{subject to}\quad \hat{\mathbf f}\cdot\mathbf n<0,"
            r"\quad \lVert\hat{\mathbf f}_t\rVert\le\mu_{\max}\,\lvert\hat{\mathbf f}\cdot\mathbf n\rvert,\quad"
            r"\lVert\hat{\mathbf f}\rVert\le f_{\max}$$")
    eq6 = (r"$$J=\big[\,S\ \ \ \partial_{t_1}S\,\hat{\mathbf f}\ \ \ \partial_{t_2}S\,\hat{\mathbf f}\,\big],\qquad"
           r"\operatorname{cov}(\hat{\mathbf f},\hat{\mathbf p})\approx\Big(J^{\top}C^{-1}J+\operatorname{diag}\big(\sigma_f^{-2}I_3,\,0,\,0\big)\Big)^{-1}$$")
    eq7 = (r"$$k_i=\frac{E^{*}a_i}{h_i},\quad h_i=\max\big(\mathbf n_i\cdot(\mathbf x_i-\mathbf x_c),\,1\ \mathrm{mm}\big),\qquad"
           r"\sum_i k_i\,[\delta-g_i]_+=f_n,\qquad p_i=\frac{k_i\,[\delta-g_i]_+}{a_i}$$")
    eq8 = (r"$$\Delta\mathbf b^{\mathrm{rest}}\approx A\begin{bmatrix}\boldsymbol\tau\\ \boldsymbol\omega\end{bmatrix}+U\,\Delta\mathbf g,"
           r"\qquad A=\frac{\partial\mathbf b}{\partial(\boldsymbol\tau,\boldsymbol\omega)}\Big|_{\text{all five magnets together}}$$")

    eq9 = (r"$$\tilde{\mathbf b}_c=\frac{\mathbf b_c}{1+\kappa_c\,(\bar T_c-T_{\mathrm{ref}})},\qquad T_{\mathrm{ref}}=25\ "
           r"^{\circ}\mathrm C,\qquad \kappa_c=\arg\min_{\kappa}\sum_{t}\big\lVert\tilde{\mathbf b}_c(t)-"
           r"\langle\tilde{\mathbf b}_c\rangle\big\rVert^{2}$$")
    eq10 = (r"$$\Sigma=\operatorname{diag}\big(\sigma_i^{2}/n+\sigma_{\mathrm{fl}}^{2}\big),\qquad n\approx 7,\qquad"
            r"\sigma_{\mathrm{fl}}=2\ \mu\mathrm T$$")
    eq11 = (r"$$\text{contact shown}\iff e>25\ \text{and}\ f_n\ge f_{\min},\qquad f_{\min}=0.25\ \mathrm N"
            r"\ \ (0.15\ \mathrm N\ \text{once shown})$$")
    eq12 = (r"$$s=\Big\lVert\operatorname{sd}_{k}\big(\bar{\mathbf b}_k\big)\Big\rVert<3\ \mu\mathrm T\ \text{for}\ "
            r"3\ \mathrm s\ \ \text{and}\ \ \lVert\bar{\mathbf b}-\mathbf b_0\rVert>5\ \mu\mathrm T\quad\Rightarrow\quad"
            r"\mathbf b_0\gets\bar{\mathbf b}$$")

    t1 = table(["Assignment of 0x0C 0x11 0x12 0x13 0x10", "Poles (centre third)", "$\\theta_0$ (&#176;)", "$k$", "Rest residual (mT)"],
               [[" ".join(U[u] for u in r["perm"]), " ".join("+" if p > 0 else "&#8722;" for p in r["poles"]),
                 f"{r['theta0']:.1f}", f"{r['k']:.3f}", f"{r['rms_mT']:.3f}"] for r in c["per_assignment"][:6]],
               num_cols=(2, 3, 4), n=1,
               caption=f"The best fit of each assignment of chip addresses to positions, of {c['n_hypotheses']} fitted "
                       f"hypotheses (poles in the FE row's order; a rest residual with $k$ at its bound of 0.1 is a failed fit).")
    keys = ("y_mm", "z_mm", "f_press_N", "f_y_N", "f_z_N")
    t2 = table(["Magnets", "Samples", "$y$ (mm)", "$z$ (mm)", "$f_n$ (N)", "$f_y$ (N)", "$f_z$ (N)", "Shown as contact"],
               [[s["magnets"][0].upper() + s["magnets"][1:], s["window"]] + [f"{s['rms'][k]:.2f} ({s['sigma_median'][k]:.2f})" for k in keys]
                + [f"{s['detected_at_threshold']} of {s['n']}"] for s in res["synthetic"]],
               num_cols=(2, 3, 4, 5, 6, 7), n=2,
               caption="The face-only estimator (the first version, on the 133 face points): RMS error on 300 simulated "
                       "presses on the face, with the median model 1-sigma in brackets, and the presses shown as contacts "
                       "($e>25$).")
    rows3 = []
    for name, R in (("N52, poles one way", s52), ("As found", saf)):
        for reg in ("face", "edge", "sides", "all"):
            v = R[reg]
            rows3.append([name, f"{reg} ({v['n']})", f"{v['far_pct']:.1f}", f"{v['pos_median_mm']:.2f}",
                          f"{v['press_abs_median_N']:.3f}", f"{v['press_rms_N']:.2f}", f"{v['sd_press_median_N']:.3f}",
                          f"{v['sd_pos_median_mm']:.2f}", f"{v['press_z_rms']:.2f}"])
    t3 = table(["Magnets", "Region (presses)", "Placed away (%)", "Position error, median (mm)", "Press error, median (N)",
                "Press error, RMS (N)", "Press 1-sigma, median (N)", "Position 1-sigma, median (mm)",
                "Press error / 1-sigma, RMS"],
               rows3, num_cols=(2, 3, 4, 5, 6, 7, 8), n=3,
               caption="The whole-surface estimator on 600 simulated presses per configuration, 50&#8202;ms of samples, by "
                       "region of the true point. Errors and 1-sigmas leave out the presses placed away.")
    t4 = table(["Press", "$f_n$ true (N)", "$f_n$ estimated (N)", "Shear true (N)", "Shear estimated (N)",
                "Position error (mm)", "Position 1-sigma (mm)"],
               [[q_["name"], f"{q_['f_n']:.2f}", f"{e_['f_n']:.2f}", f"{q_['f_t'][0]:.2f}, {q_['f_t'][1]:.2f}",
                 f"{e_['f_t'][0]:.2f}, {e_['f_t'][1]:.2f}", f"{e_['pos_err_mm']:.2f}", f"{e_['sd_pos_mm']:.2f}"]
                for q_, e_ in zip(sr["presses"], sr["estimates"])],
               num_cols=(1, 2, 3, 4, 5, 6), n=4,
               caption="The five synthetic presses of Figure 1 (magnets as found) through the viewer's pipeline: "
                       "calibration and baseline on the recording's first 2&#8202;s, the 50&#8202;ms window at each press's "
                       "middle. Shear along $\\mathbf t_1$, $\\mathbf t_2$.")
    t5 = table(["Rest (s)", "Offset (&#181;T)", "Shift (&#181;m)", "Tilt (mrad)", "Rigid fit residual (%)", "Estimator reads"],
               [[f"{r['t0']:.1f}&#8211;{r['t1']:.1f}", f"{r['offset_uT']:.1f}", f"{np.linalg.norm(r['translation_um']):.1f}",
                 f"{np.linalg.norm(r['tilt_mrad']):.1f}", f"{r['residual_pct']:.0f}",
                 (f"{r['read_as']['f_n']:.1f}&#8202;N at ({', '.join(f'{v:.1f}' for v in r['read_as']['p'])})"
                  if r["read_as"] else "no contact")] for r in rests[1:]],
               num_cols=(0, 1, 2, 3, 4), n=5,
               caption=f"The rests after the presses of Figure 5 against the first rest ({rests[0]['t0']:.1f}&#8211;"
                       f"{rests[0]['t1']:.1f}&#8202;s): the rigid motion of all five magnets together and a uniform field "
                       f"fitted to each offset (Eq. 8; 9 parameters for 15 components), and the whole-surface "
                       f"estimator's reading of it. Random 15-component patterns leave a median residual of "
                       f"{se['random_residual_pct'][0]:.0f}&#8202;% and 1&#8202;% of them below "
                       f"{se['random_residual_pct'][1]:.0f}&#8202;%.")

    wv = th["warmup"]["variants"]
    t6 = table(["Correction", "Drift after the warm-up, norm (&#181;T)", "Largest component (&#181;T)"],
               [[k[0].upper() + k[1:], f"{v['norm_uT']:.1f}", f"{v['max_uT']:.1f}"] for k, v in wv.items()],
               num_cols=(1, 2), n=6,
               caption=f"The rest reading's change over the warm-up of 21 October ({th['warmup']['span_s']:.0f}&#8202;s from "
                       f"the viewer's start to the first touch; the chips warmed by {min(th['warmup']['T_rise_C']):.1f}&#8211;"
                       f"{max(th['warmup']['T_rise_C']):.1f}&#8202;&#176;C), with the coefficients fitted on the night before "
                       f"(Eq. 9): per chip or one for all, with each chip's own temperature or the five chips' mean.")
    t7 = table(["Noise floor (&#181;T)", "Night, contact shown (% of minutes)", "Night, largest force shown (N)",
                "Warm-up, contact shown (% of seconds)", "Warm-up uncorrected (%)",
                "Smallest press shown: face, edge, sides (N)"],
               [[f"{r_['floor_uT']:.1f}", f"{r_['night_shown_pct']:.1f}", f"{r_['night_max_N']:.2f}",
                 f"{r_['warmup_shown_pct']:.1f}", f"{r_['warmup_uncorrected_shown_pct']:.1f}",
                 ", ".join(f"{r_['smallest_N'][k]:.2f}" for k in ("face", "edge", "sides"))] for r_ in sw],
               num_cols=(0, 1, 2, 3, 4), n=7,
               caption="The estimator at rest as the viewer runs it (Eq. 11, the 50&#8202;ms noise) for each noise floor: the "
                       "share of the night's 60&#8202;s means and of the warm-up's 1&#8202;s means shown as a contact, both "
                       "corrected for temperature unless marked, and the smallest press along the normal that shows, with "
                       "the 0.25&#8202;N threshold in every row. The viewer uses 2&#8202;&#181;T.")
    rp = th["replay"]
    t8 = table(["Recording", "Frames", "Contact shown, baseline kept (%)", "Contact shown, re-zero (%)", "Re-zeros",
                "Of them over 26&#8202;&#181;T", "Re-zeros the reading came back from"],
               [[nm, f"{rp[nm]['auto-zero']['frames']:,}", f"{rp[nm]['baseline kept']['shown_share_pct']:.1f}",
                 f"{rp[nm]['auto-zero']['shown_share_pct']:.1f}", f"{rp[nm]['auto-zero']['rezeros']}",
                 f"{rp[nm]['auto-zero']['rezeros_over_26uT']}", f"{len(rp[nm]['auto-zero']['came_back'])}"]
                for nm in ("09:59 touches", "20:06 presses", "19:39 session")],
               num_cols=(1, 2, 3, 4, 5, 6), n=8,
               caption="The viewer's pipeline at 30&#8202;Hz over the recordings by hand, with the temperature correction, "
                       "the 2&#8202;&#181;T floor and the 0.25&#8202;N threshold, keeping the first baseline or re-zeroing at "
                       "rest (Eq. 12). A re-zero the reading came back from: within 10&#8202;s the reading returned to "
                       "within 30&#8202;% of the re-zero's offset of the earlier baseline. 26&#8202;&#181;T is about 1&#8202;N "
                       "on the face; the 19:39 session lasts 24&#8202;s.")
    z6 = np.load(D / "data/20261020-200647_presses_records.npz")
    tz, Bz = z6["t"], z6["r"][:, :, 1:].reshape(-1, 15).astype(float)
    start_off = float(np.linalg.norm(Bz[tz < 4].mean(0) - Bz[tz > tz[-1] - 100].mean(0)))

    cmd = ("<pre class=\"cmd\">~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/wrench_viewer.py\n"
           "    # open http://localhost:8790; keys b (new baseline), c (recalibrate at rest), r (record a CSV), z (re-zero on/off)\n"
           "    #   --floor 2 --min-force 0.25 --auto-zero 3 (0: off) --no-thermal\n"
           "~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/wrench_viewer.py --replay REC.csv --http 8791\n"
           "~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/pad_shift_check.py REC.csv   # or a stream log, --t0 --t1\n"
           "~/miniconda3/bin/python scripts/fingertip_contact_estimation_study.py      # the numbers of this page\n"
           "~/miniconda3/bin/python scripts/fingertip_contact_estimation_study.py thermal   # Section 9, 3 min\n"
           "~/miniconda3/bin/python scripts/fingertip_drift_film.py                    # Film 9\n"
           "~/miniconda3/envs/femtip/bin/python scripts/fingertip_wrench_fe.py --surface   # the surface basis, 101 s, 2.0 GB</pre>")

    S = []
    S.append(("terms", "Terms and metrics", terms))
    S.append(("model", "Measurement model of the board under the pad",
              f"<p>The board's five MLX90393 chips sit 7&#8202;mm apart in a plus, U1 at the centre and U2 to U5 toward the "
              f"tip end, toward $+y$, toward the mount and toward $-y$, on a sensing plane 0.42&#8202;mm below the pad's back "
              f"face. The five 3&#8202;&#215;&#8202;2&#8202;mm magnets sit at the positions of the FE row {code(fe['case'])} "
              f"of {link(FE_PAGE, 'the v2 fingertip checks')}, {fe['depth_mm']}&#8202;mm below the palmar face. Each magnet "
              f"is a uniformly magnetized cylinder, evaluated by a volume quadrature of 72 point dipoles (Eq. 1). On its "
              f"axis at the chips' distance of 5.37&#8202;mm this gives 19.970&#8202;mT, as the closed form does. One "
              f"point dipole gives 21.068&#8202;mT.</p>{eq1}"
              f"<p>Here $\\mathbf c_j$ is magnet $j$'s centre, $\\boldsymbol\\rho_q$ and $w_q$ the quadrature's points and "
              f"weights, $s_j=\\pm1$ its pole and $V$ its volume. A chip reports the field in its own axes (Eq. 2). Its $z$ "
              f"axis is the pad's normal up to a sign $s$, and its $x$ axis lies at $\\theta_0+\\phi_c$ from $\\mathbf e_y$, "
              f"where $\\phi_c$ is the part's rotation in the board's pick-and-place file (U1 0&#176;, U2 180&#176;, U3 "
              f"0&#176;, U4 90&#176;, U5 270&#176;) and $\\theta_0$, the orientation of the part's axes in its footprint, is "
              f"common to the five chips. A permutation $\\pi$ assigns the stream's chips (the board answers at 0x0C and "
              f"0x10&#8211;0x13) to positions.</p>{eq2}"))
    S.append(("rest", "Calibration from the reading at rest",
              f"<p>With the pad untouched the reading is the magnets' field plus Earth's field: 15 numbers the model must "
              f"reproduce. The fit (Eq. 3) searches the poles of the four corner magnets relative to the centre one (16 "
              f"patterns), the sign $s$, the assignment $\\pi$ (24) and $\\theta_0$ from four quadrants, and fits the strength "
              f"$k$, the chips' offset $\\boldsymbol\\delta$ from the pad and a uniform field $\\mathbf g$ by least squares. A "
              f"first pass on the normal component and the in-plane magnitude of each chip's field, which do not depend on "
              f"the chips' rotations, ranks the discrete hypotheses for the full fit.</p>{eq3}"
              f"<p>The pad as found has magnets of unknown grade, inserted with random poles. Its best fit places "
              f"{assign}. The fit has $\\theta_0={b['theta0']:.1f}$&#176;, two diagonally opposite corner magnets reversed, "
              f"$k={b['k']:.2f}$ and the chips offset by ({b['dx']:+.2f}, {b['dy']:+.2f}, {b['dz']:+.2f})&#8202;mm. It is at "
              f"{b['rms_mT']:.2f}&#8202;mT RMS against fields of {fmin:.1f}&#8211;{fmax:.1f}&#8202;mT per chip (Table 1, "
              f"Figure 2). The board turned by 180&#176;, which this pole pattern alone cannot distinguish, fits at "
              f"{c['turned_rms_mT']:.2f}&#8202;mT: the pick-and-place rotations do not follow the turn. The uniform field "
              f"ends at its bound, so about 0.4&#8202;mT of the reading is model error. The full search takes about a minute; "
              f"with the board's chip frames kept, the viewer refits poles, strength and offset in 6&#8211;9&#8202;s.</p>"
              f"{t1}"
              + fig(2, MEDIA / "20261020-rest_fit.png",
                    "The reading at rest per chip axis, measured with the pad as found and modelled by the best fit (labels: "
                    "address, fitted position, axis).")))
    S.append(("fe", "FE basis over the pad's surface",
              f"<p>A contact at surface point $\\mathbf p$ with force $\\mathbf f$ moves each magnet rigidly by "
              f"$M(\\mathbf p)\\mathbf f$, taken from the FE of concept b's pad: the gyroid infill's homogenized stiffness, "
              f"the cut face and the mount face fixed, $E^*={fe['E_star_MPa']:.1f}$&#8202;MPa from the pressure law and an "
              f"approach of {fe['approach_1N_um']:.0f}&#8202;&#181;m at 1&#8202;N. The first basis loaded {fe['patch_centres']} "
              f"points of a 1&#8202;mm grid on the palmar face. The basis of the estimator now loads the surface points of the "
              f"tip's {fs['pads']} sphere pads in front of the cut face, the contact model of the simulations "
              f"({fs['area_mm2']:.0f}&#8202;mm&#178;: {fs['regions']['face']} on the face, {fs['regions']['edge']} on the "
              f"edge, {fs['regions']['sides']} on the sides). Each takes 1&#8202;N along $x$, $y$ and $z$, spread by a "
              f"Gaussian of $\\sigma={fe['sigma_mm']}$&#8202;mm over the FE's {fs['surface_nodes']:,} free surface nodes "
              f"(each node's area projected on its smoothed normal; nodes facing within 73&#176; of the pad's normal): "
              f"{fs['load_cases']:,} load cases on one factorization of {fs['n_dof']:,} degrees of freedom, 101&#8202;s "
              f"and 2.0&#8202;GB. On the face its press motions agree with the first basis within "
              f"{100 * fa['rel_diff_median']:.1f}&#8202;% (median over {fa['points']} points; "
              f"{100 * fa['rel_diff_p90']:.0f}&#8202;% at the 90th percentile, the paired points "
              f"{fa['match_mm_median']:.1f}&#8202;mm apart). The normal compliance at the loaded patch is "
              f"{fs['normal_compliance_um_per_N'][0]:.0f}, {fs['normal_compliance_um_per_N'][1]:.0f} and "
              f"{fs['normal_compliance_um_per_N'][2]:.0f}&#8202;&#181;m/N at the 5th, 50th and 95th percentiles over the "
              f"surface; the basis also keeps every load's displacement at the pad points for the viewer.</p>"))
    S.append(("estimator", "Estimator over the surface",
              f"<p>The magnet motions are interpolated between the pad points by a thin-plate spline in three dimensions "
              f"and passed through the linear field model (the change of the 15 components per micrometre and milliradian "
              f"of each magnet), which gives $S(\\mathbf p)$ (Eq. 4). A background change and a strength change enter "
              f"through $N$ with a Gaussian prior ($P$: 30&#8202;&#181;T per axis, $3\\times10^{{-3}}$ in strength) so "
              f"that they cannot take up a press; the face-only estimator projected them out.</p>{eq4}"
              f"<p>For every candidate point the force follows in closed form (Eq. 5). The candidate is chosen by the log "
              f"marginal likelihood $\\ell(\\mathbf p)$ with a force prior of $\\sigma_f=3$&#8202;N: without the prior, a "
              f"point where the reading barely responds explains any pattern with a large enough force, and the determinant "
              f"charges each point for the patterns it could explain. The force must press, lie within a cone of "
              f"$\\mu_{{\\max}}=2$ and stay below $f_{{\\max}}=25$&#8202;N. The candidates are the pad points' triangulation with "
              f"every triangle split 16 times, {sf['as found']['candidates']:,} points about 0.2&#8202;mm apart; one estimate "
              f"takes {sf['N52, poles one way']['frame_ms']:.1f}&#8202;ms, and the Fisher information gives its "
              f"uncertainty (Eq. 6).</p>{eq5}{eq5b}{eq6}"))
    S.append(("simulated", "Errors on simulated presses",
              f"<p>The simulated presses are readings made by the model itself with the measured noise, so they measure "
              f"the estimator and the information in the reading. With N52 "
              f"magnets the whole-surface estimator places {s52['all']['far_pct']:.0f}&#8202;% of presses away. For the "
              f"rest the median position error is {s52['face']['pos_median_mm']:.2f}&#8202;mm on the face and "
              f"{s52['sides']['pos_median_mm']:.2f}&#8202;mm on the sides. The median press error is "
              f"{s52['face']['press_abs_median_N']:.3f} and {s52['sides']['press_abs_median_N']:.3f}&#8202;N (Table 3). The "
              f"errors follow the model 1-sigma (ratio {min(v['press_z_rms'] for v in s52.values()):.2f}&#8211;"
              f"{max(v['press_z_rms'] for v in s52.values()):.2f}). The RMS press error on the sides, "
              f"{s52['sides']['press_rms_N']:.2f}&#8202;N, comes from a few presses deep on the flanks, where the 1-sigma "
              f"is largest (Figure 3). With the magnets as found, {saf['all']['far_pct']:.0f}&#8202;% are placed away and "
              f"the rest have {saf['all']['pos_median_mm']:.2f}&#8202;mm and {saf['all']['press_abs_median_N']:.3f}&#8202;N. "
              f"The face-only estimator gave {n52f['rms']['y_mm']:.2f}&#8202;mm and {n52f['rms']['f_press_N']:.3f}&#8202;N "
              f"RMS on the face with N52 (Table 2).</p>{t3}"
              + fig(3, MEDIA / "20261020-surface_uncertainty.png",
                    "Model 1-sigma of the pressing force for 1&#8202;N along the normal at each pad point, 50&#8202;ms of "
                    "samples, seen from the palm and from both sides; colour scales clipped at the 95th percentile. Red "
                    "circles: magnets; white squares: chips.")
              + f"<p>Through the viewer's own pipeline the five synthetic presses of Figure 1 return their pressing force "
                f"within 0.25&#8202;N and their point within 1.2&#8202;mm with the magnets as found (Table 4).</p>{t4}"
              + f"<p>{t2}</p>"))
    S.append(("quiet", "Behaviour without a press",
              f"<p>Over 30&#8202;s of the board with the pad untouched ({q['frames']} frames of 50&#8202;ms), with the chips' "
              f"noise alone in the covariance (Section 9 adds a floor), the reading's "
              f"chi-square against $C$ has a median of {q['chi2_median']:.1f} for 15 components. The best contact the "
              f"whole-surface estimator finds explains a median chi-square of {q['e_median']:.1f} and a 99th percentile of "
              f"{q['e_p99']:.1f}; {q['false_contacts']} frame of {q['frames']} exceeds the threshold of 25 (Figure 4).</p>"
              + fig(4, MEDIA / "20261020-no_press.png",
                    "The whole-surface estimator over 30&#8202;s without a press: the explained chi-square of the best "
                    "contact against the threshold (top) and the background change it assigns (bottom).")))
    pk = se["peaks"]
    S.append(("hand", "Presses by hand with the pad loose in its seat",
              f"<p>The owner pressed the pad by hand five times in {se['span_s']:.0f}&#8202;s at {se['start'][11:16]} with "
              f"the viewer recording; the reading departed {min(p_['dB_uT'] for p_ in pk):.0f}&#8211;"
              f"{max(p_['dB_uT'] for p_ in pk):.0f}&#8202;&#181;T from the preceding rest at the presses (Figure 5). After "
              f"the first press it returned within {rests[1]['offset_uT']:.0f}&#8202;&#181;T of the first rest; after the "
              f"fourth and fifth it stayed {rests[2]['offset_uT']:.0f} and {rests[3]['offset_uT']:.0f}&#8202;&#181;T away "
              f"for the 2&#8211;3&#8202;s of the following rests, without decay. The chips' temperatures, averaged over 1&#8202;s, "
              f"varied by at most {max(se['temperature_C']):.2f}&#8202;&#176;C over the recording. A rigid motion of the whole magnet set and a "
              f"uniform field (Eq. 8) reproduce the two large offsets with {rests[2]['residual_pct']:.0f} and "
              f"{rests[3]['residual_pct']:.0f}&#8202;% residual (Table 5): shifts of "
              f"{np.linalg.norm(rests[2]['translation_um']):.0f} and {np.linalg.norm(rests[3]['translation_um']):.0f}"
              f"&#8202;&#181;m and tilts of {np.linalg.norm(rests[2]['tilt_mrad']):.1f} and "
              f"{np.linalg.norm(rests[3]['tilt_mrad']):.1f}&#8202;mrad. The owner confirmed the pad rocking from side to "
              f"side in its seat as it was pushed. The FE fixes the pad's cut face. The estimator therefore explains the "
              f"offsets as standing contacts of {min(r['read_as']['f_n'] for r in big if r['read_as']):.0f}&#8211;"
              f"{max(r['read_as']['f_n'] for r in big if r['read_as']):.0f}&#8202;N, and the presses themselves as "
              f"{min(p_['f_n'] for p_ in pk):.1f}&#8211;{max(p_['f_n'] for p_ in pk):.1f}&#8202;N with the point jumping "
              f"between the face and the +y flank (Figure 6). The model gives about 26&#8202;&#181;T per newton for a press "
              f"on the face; whether the pad also gave more field per newton than the model while it moved cannot be told "
              f"without a reference force.</p>{eq8}{t5}"
              + fig(5, MEDIA / "20261020-presses_by_hand.png",
                    f"The owner's recording of {se['start'][:16]}: the reading change from the first rest with the rests "
                    f"in grey (top), the whole-surface estimator's pressing force and shear where it shows a contact "
                    f"(middle), and the last rest's offset per chip axis against the rigid motion fitted to it (bottom).")
              + film(6, MEDIA / "20261020-viewer_session_replay.mp4", MEDIA / "20261020-viewer_session_replay.png",
                     "The same recording replayed through the viewer: the presses, then the standing contacts the shifted "
                     "pad leaves (the force arrow is drawn at most 25&#8202;mm long).")))
    nt, wu, an = th["night"], th["warmup"], th["after_nuisances"]
    sh0 = th["shown"]
    f15 = [r_ for r_ in sw if r_["floor_uT"] == 1.5][0]
    f2 = [r_ for r_ in sw if r_["floor_uT"] == 2.0][0]
    lim0 = th["detection_limit_N"]["floor 0 uT"]
    pre = [q_ for q_ in seat if q_["t1"] < 150]
    first_m = [q_ for q_ in moved if q_["t0"] < 400]
    later_m = [q_ for q_ in moved if q_["t0"] >= 400]
    rv = th["rigid_vs_press"]
    j0 = int(np.argmax(nt["raw_ptp_uT"]))
    cb = rp["20:06 presses"]["auto-zero"]["came_back"]
    S.append(("drift", "Drift of the rest reading",
              f"<p>The firmware runs the chips without their on-chip temperature compensation (TCMP_EN 0), and each chip's "
              f"sensitivity falls by {k_lo:.2f}&#8211;{k_hi:.2f}&#8202;% per &#176;C of its die temperature. Over the night "
              f"of 20 October the pad was untouched from 20:18 to 09:15. The chips followed the room through "
              f"{min(nt['T_range_C']):.1f}&#8211;{max(nt['T_range_C']):.1f}&#8202;&#176;C, and the rest reading moved by up to "
              f"{nt['raw_ptp_uT'][j0]:.0f}&#8202;&#181;T peak to peak, on {ADDR[j0 // 3]}'s {'xyz'[j0 % 3]} axis in a field of "
              f"{nt['field_mT'][j0 // 3]:.1f}&#8202;mT (Figure 7). Dividing each chip's reading by its gain at its own "
              f"temperature (Eq. 9), with $\\kappa_c$ chosen to make the night's corrected reading of that chip the most "
              f"constant, leaves {nt['corrected_ptp_uT'][j0]:.0f}&#8202;&#181;T on that axis and {nt['corrected_rms_uT']:.1f}"
              f"&#8202;&#181;T RMS over the 15 components, against {nt['raw_rms_uT']:.1f}&#8202;&#181;T uncorrected. On "
              f"{ADDR[0]} one degree is {-1e3 * th['kappa_per_C'][0] * nt['field_mT'][0]:.0f}&#8202;&#181;T, twice the change "
              f"of a 1&#8202;N press on the face. NdFeB loses about 0.12&#8202;% of its remanence per &#176;C, a sixth of the "
              f"coefficient.</p>{eq9}"
              f"<p>The board warms after it is plugged in. On 21 October it was plugged in again at 09:55; over the "
              f"{wu['span_s']:.0f}&#8202;s from the viewer's start at 09:59:07 to the first touch the chips warmed by "
              f"{min(wu['T_rise_C']):.1f}&#8211;{max(wu['T_rise_C']):.1f}&#8202;&#176;C and the rest reading moved by "
              f"{wv['uncorrected']['norm_uT']:.0f}&#8202;&#181;T. The night's coefficients take that to "
              f"{wv['per chip, own temperature']['norm_uT']:.1f}&#8202;&#181;T (Table 6). One coefficient for all chips does "
              f"as well ({wv['one coefficient, own temperature']['norm_uT']:.1f}&#8202;&#181;T); the five chips' mean "
              f"temperature in place of each chip's own leaves {wv['per chip, mean temperature']['norm_uT']:.0f}&#8202;&#181;T, "
              f"because the chips warm at different rates.</p>{t6}"
              f"<p>Whether the viewer shows a contact at rest depends more on the estimator's noise model than on the "
              f"correction (Table 7). With the chips' noise alone, under 1&#8202;&#181;T a component for 50&#8202;ms, the "
              f"estimator showed a contact in {sh0['night, corrected, floor 0 uT']['share_pct']:.0f}&#8202;% of the night's "
              f"minutes and {sh0['warm-up, corrected, floor 0 uT']['share_pct']:.0f}&#8202;% of the warm-up's seconds after the "
              f"correction, at up to {max(sh0['night, corrected, floor 0 uT']['max_N'], sh0['warm-up, corrected, floor 0 uT']['max_N']):.1f}"
              f"&#8202;N ({sh0['night, uncorrected, floor 0 uT']['share_pct']:.0f} and "
              f"{sh0['warm-up, uncorrected, floor 0 uT']['share_pct']:.0f}&#8202;% uncorrected, the viewer as it ran until "
              f"20 October). Its two nuisances, a uniform field and a uniform gain change, leave up to "
              f"{an['corrected']['night_max_uT']:.0f}&#8202;&#181;T of the corrected night and "
              f"{an['corrected']['warmup_end_uT']:.0f}&#8202;&#181;T of the corrected warm-up, and against that noise the "
              f"remainder is read as a contact. The viewer now adds a noise floor of {th['floor_uT']:.0f}&#8202;&#181;T to every "
              f"component (Eq. 10) and shows a contact only above {th['min_force_N']:.2f}&#8202;N (Eq. 11); with both, no "
              f"contact shows at rest in either stretch. The floor costs sensitivity: the smallest press shown rises from "
              f"{lim0['face']:.2f} to {f2['smallest_N']['face']:.2f}&#8202;N on the face, from {lim0['edge']:.2f} to "
              f"{f2['smallest_N']['edge']:.2f}&#8202;N on the edge and from {lim0['sides']:.2f} to "
              f"{f2['smallest_N']['sides']:.2f}&#8202;N on the sides. A floor of 1.5&#8202;&#181;T shows "
              f"{f15['night_shown_pct']:.0f}&#8202;% of the night as a contact. The correction matters once the temperature "
              f"moves by more than about 1&#8202;&#176;C, where the gain nuisance's prior of 0.3&#8202;% no longer takes up the "
              f"change: uncorrected, the night shows a contact in {sh0['night, uncorrected, floor 2 uT']['share_pct']:.0f}"
              f"&#8202;% of its minutes even with the floor.</p>{eq10}{eq11}{t7}"
              f"<p>A touch also moves the pad in its seat. Before the first touch of 21 October the corrected rests stayed "
              f"within {max(q_['offset_uT'] for q_ in pre):.0f}&#8202;&#181;T of the first; after it they lay "
              f"{min(q_['offset_uT'] for q_ in first_m):.0f}&#8211;{max(q_['offset_uT'] for q_ in first_m):.0f}&#8202;&#181;T "
              f"away and after a later touch {min(q_['offset_uT'] for q_ in later_m):.0f}&#8211;"
              f"{max(q_['offset_uT'] for q_ in later_m):.0f}&#8202;&#181;T, which rigid shifts of "
              f"{min(q_['shift_um'] for q_ in moved):.0f}&#8211;{max(q_['shift_um'] for q_ in moved):.0f}&#8202;&#181;m and "
              f"tilts of {min(q_['tilt_mrad'] for q_ in moved):.1f}&#8211;{max(q_['tilt_mrad'] for q_ in moved):.1f}&#8202;mrad "
              f"reproduce within {max(q_['residual_pct'] for q_ in moved):.0f}&#8202;% (Eq. 8). At the viewer's start on 20 "
              f"October at 20:06 the pad sat still {start_off / 1e3:.1f}&#8202;mT from its later rest for 4&#8202;s, until a "
              f"touch moved it, and the viewer took its baseline there. The reading alone cannot tell a shift in the seat "
              f"from a press: a rigid motion of the magnet set and a uniform field leave {rv['press_median_pct']:.0f}&#8202;% "
              f"of a simulated 1&#8202;N press unexplained (median over the surface; {rv['random_median_pct']:.0f}&#8202;% of a "
              f"random pattern). Their time courses differ: after a touch the pad comes to rest, and a press by hand wavers. "
              f"The viewer therefore re-zeros once the reading has been as still as at rest for 3&#8202;s and has moved more "
              f"than 5&#8202;&#181;T from the baseline (Eq. 12). Over the recording of 21 October that takes the share of "
              f"frames with a contact shown from {rp['09:59 touches']['baseline kept']['shown_share_pct']:.0f} to "
              f"{rp['09:59 touches']['auto-zero']['shown_share_pct']:.0f}&#8202;%, and over the presses of 20 October, from "
              f"the rest at 8&#8211;10&#8202;s, from {rp['20:06 presses']['baseline kept']['shown_share_pct']:.0f} to "
              f"{rp['20:06 presses']['auto-zero']['shown_share_pct']:.0f}&#8202;% (Table 8, Figure 8, Film 9). "
              f"Of that night's {rp['20:06 presses']['auto-zero']['rezeros']} re-zeros, {len(cb)} took in a state the "
              f"reading came back from within 10&#8202;s"
              + (f" ({', '.join(f'{c_[1]:.0f}' for c_ in cb)}&#8202;&#181;T)" if cb else "")
              + ". A press held as still as the pad at rest for 3&#8202;s goes into the baseline the same way.</p>"
              f"{eq12}{t8}"
              + fig(7, MEDIA / "20261021-rest_drift.png",
                    "The rest reading against the chips' die temperatures. Left: the night of 20 October from 20:36, 60&#8202;s "
                    "means of the temperatures (top) and of 0x0C's z reading, uncorrected and corrected (bottom). Right: the "
                    "warm-up after the board was plugged in again on 21 October, 1&#8202;s means of the temperature rise "
                    "(top) and of the norm of the 15-component change (bottom); after the touches the pad's seat moves the "
                    "rest, which the correction leaves.")
              + fig(8, MEDIA / "20261021-rezero_replay.png",
                    "The pressing force the viewer shows over the recordings by hand, with the temperature correction, the "
                    "2&#8202;&#181;T floor and the 0.25&#8202;N threshold, keeping the first baseline (light red) or "
                    "re-zeroing after 3&#8202;s at rest (black; ticks: re-zeros).")
              + film(9, MEDIA / "20261021-viewer_drift_replay.mp4", MEDIA / "20261021-viewer_drift_replay.png",
                     "The viewer replaying the warm-up and the touches of 21 October five times faster than recorded, with "
                     "the settings of 20 October (top) and of 21 October (bottom). The poster is 245&#8202;s into the "
                     "recording, after the second touch: the chips are 0.9&#8211;1.4&#8202;&#176;C warmer than at the "
                     "calibration, and the top viewer shows a standing contact of about 10&#8202;N.")))
    po = res["poles"]
    sd = po["all one way"]["neighbour"]["side"]["mm"]
    i26, i10 = sd.index(26.0), po["all one way"]["neighbour"]["facing"]["mm"].index(10.0)
    t9 = table(["Pattern", "Poles toward the palm", "Net moment (magnets)", "Position error (mm)", "Press error (N)",
                "Placed away (%)", "1 N press on the face (&#181;T)", "Side by side at 26 mm (&#181;T)",
                "Facing, faces 10 mm apart (&#181;T)"],
               [[k[0].upper() + k[1:], " ".join("N" if q_ > 0 else "S" for q_ in v["poles"]), f"{v['net']:+d}",
                 f"{v['simulated']['all']['pos_median_mm']:.2f}", f"{v['simulated']['all']['press_abs_median_N']:.3f}",
                 f"{v['simulated']['all']['far_pct']:.1f}", f"{v['press_uT_per_N']['face']:.0f}",
                 f"{v['neighbour']['side']['nonuniform_uT'][i26]:.0f} ({v['neighbour']['side']['moving5_uT'][i26]:.0f})",
                 f"{v['neighbour']['facing']['nonuniform_uT'][i10]:.0f} ({v['neighbour']['facing']['moving5_uT'][i10]:.0f})"]
                for k, v in po.items()],
               num_cols=(2, 3, 4, 5, 6, 7, 8), n=9,
               caption="Four pole patterns of the five N52 magnets ($k=1$; poles in the FE row's order: the corner at the "
                       "mount end on $-y$, the one on $+y$, the centre, the tip-end corner on $-y$, the one on $+y$): the "
                       "whole-surface estimator's median errors on 600 simulated presses with 50&#8202;ms of samples, the "
                       "reading change of a 1&#8202;N press along the normal on the face (median), and the part of a second "
                       "fingertip's field at the board that a uniform field does not take up, with the change for 5&#8202;mm "
                       "more in brackets.")
    ao, cb_, cr = po["all one way"], po["checkerboard"], po["centre reversed"]
    S.append(("poles", "Pole pattern of the five magnets",
              f"<p>The print procedure seats every magnet north toward the palmar face, and the N52 rows above assume that "
              f"pattern. The eFlesh authors seat neighbouring magnets in alternating polarity, so that the far field of a "
              f"sensor cancels and a second sensor nearby induces less signal (Pattabiraman et al., arXiv:2506.09994, "
              f"Section 2.7.1); their cube has four magnets, whose moments cancel. Five magnets cannot cancel: the "
              f"smallest net moment is one magnet's. The checkerboard, with the centre and the corners on one diagonal "
              f"north toward the palm and the other diagonal south, has that net moment, and its corners' first moments "
              f"cancel as well; the pad as found has this pattern (Section 3). Table 9 and Figure 10 compare four "
              f"patterns: the estimator's errors, and the field a second fingertip with the same pattern puts on the "
              f"board, in the part a uniform field does not take up, which the estimator reads as a contact.</p>"
              f"<p>Against all one way, the checkerboard lowers the neighbour's field by "
              f"{ao['neighbour']['side']['nonuniform_uT'][i26] / cb_['neighbour']['side']['nonuniform_uT'][i26]:.1f} times "
              f"side by side at 26&#8202;mm ({ao['neighbour']['side']['nonuniform_uT'][i26]:.0f} to "
              f"{cb_['neighbour']['side']['nonuniform_uT'][i26]:.0f}&#8202;&#181;T) and "
              f"{ao['neighbour']['facing']['nonuniform_uT'][i10] / cb_['neighbour']['facing']['nonuniform_uT'][i10]:.1f} times "
              f"with the palmar faces 10&#8202;mm apart ({ao['neighbour']['facing']['nonuniform_uT'][i10]:.0f} to "
              f"{cb_['neighbour']['facing']['nonuniform_uT'][i10]:.0f}&#8202;&#181;T), and a 1&#8202;N press on the face "
              f"changes the reading by {cb_['press_uT_per_N']['face']:.0f} against {ao['press_uT_per_N']['face']:.0f}&#8202;"
              f"&#181;T. The estimator's median errors grow from {ao['simulated']['all']['pos_median_mm']:.2f} to "
              f"{cb_['simulated']['all']['pos_median_mm']:.2f}&#8202;mm and from {ao['simulated']['all']['press_abs_median_N']:.3f} "
              f"to {cb_['simulated']['all']['press_abs_median_N']:.3f}&#8202;N. Reversing only the centre magnet gives the "
              f"smallest errors, {cr['simulated']['all']['pos_median_mm']:.2f}&#8202;mm and "
              f"{cr['simulated']['all']['press_abs_median_N']:.3f}&#8202;N, and lowers the neighbour's field by "
              f"{ao['neighbour']['side']['nonuniform_uT'][i26] / cr['neighbour']['side']['nonuniform_uT'][i26]:.1f} times. "
              f"With the checkerboard a fingertip side by side at 26&#8202;mm still changes the reading by "
              f"{cb_['neighbour']['side']['moving5_uT'][i26]:.0f}&#8202;&#181;T when it moves 5&#8202;mm away, about a "
              f"{cb_['neighbour']['side']['moving5_uT'][i26] / cb_['press_uT_per_N']['face']:.1f}&#8202;N press; the other "
              f"fingertips' field, placed by the hand's joint angles, has to be subtracted as well.</p>{t9}"
              + fig(10, MEDIA / "20261022-pole_patterns.png",
                    "The part of a second fingertip's field at the board that a uniform field does not take up, for four "
                    "pole patterns of the five N52 magnets (both tips alike). The left plot is against the gap between the "
                    "palmar faces, with the second tip mirrored. The right plot is against the spacing across the finger. "
                    "Dashed lines show the change when the second tip moves 5&#8202;mm farther. Grey: the reading change of a 1&#8202;N press on the face.")))
    S.append(("viewer", "Live viewer in the browser",
              f"<p>{code('src/morphohand/driver/reskin5x/host/wrench_viewer.py')} reads the board and calibrates at rest; "
              f"its page is at http://localhost:8790. The page draws the pad's whole surface deformed by the FE basis at "
              f"the estimated force (the displacement drawn at most 2&#8202;mm), coloured by the pressure of the sphere pads "
              f"under a sphere pressed at the estimated point along the normal (Eq. 7, the stiffness rule of the pad builder "
              f"v2 with $h_i$ the pad's depth below the FE nodes' centroid $\\mathbf x_c$ along its normal). It also draws "
              f"the force arrow at the contact point, the 1-sigma position ellipse in the tangent plane, the magnets in "
              f"their fitted poles and the chips with their fitted addresses, and traces of force, contact point and "
              f"reading. Every record of the stream goes to {code('logs/20261020-contact_estimation/stream/')} as it "
              f"arrives (88 bytes a record), so a session is kept without pressing r; "
              f"{code('pad_shift_check.py')} reports the rest offsets of a recording or of a stretch of that log. The server "
              f"corrects every reading for its chip's temperature, shows a contact by Eq. 11 and re-zeros at rest by Eq. 12; "
              f"the page has sliders for the threshold force and the noise floor and a switch for the re-zero (key z), and "
              f"shows the chips' temperature change since the calibration. When the board is unplugged the server waits for "
              f"it, opens it again and takes a new baseline 2&#8202;s later.</p>{eq7}"
              + cmd))
    S.append(("limits", "Limitations",
              "<p>No estimate has been compared with a reference force or position; the errors above are the model against "
              "itself. The rest residual of 0.43&#8202;mT, about 10&#8202;% of the field, shows how well the magnets' geometry "
              "is known, and the sensitivity can be wrong by a similar or larger fraction. The FE is linear elastic with the "
              "gyroid's homogenized stiffness, without the stiffening of large presses or the infill's local buckling, and "
              "with the pad's cut face fixed: a pad that moves in its seat breaks the model, as the presses by hand show. "
              "The cover's walls along the flanks are not in the FE; points behind them are still candidates. The estimator "
              "assumes one contact; a press spread over the pad is reported near its centre of load. The background term "
              "covers uniform fields only, so a steel tool near the pad reads as a contact. The pressure patch assumes a "
              "sphere of a chosen radius on the sphere pads, and the deformation drawn is the FE's response to a Gaussian "
              "patch of load. The magnets in the pad today are of unknown grade and their poles are mixed; the N52 "
              "rows assume the same geometry with $k=1$ and all poles one way. The temperature coefficients come from one "
              "night on one board, and the correction uses the chips' die temperature: the magnets' own coefficient follows "
              "the pad's temperature, which lags the chips', and is left to the gain nuisance. The noise floor raises the "
              "smallest press shown on the face to 0.6&#8202;N, and the re-zero takes in a press held still for 3&#8202;s.</p>"))
    S.append(("future", "Future work",
              "<ul>"
              "<li>Seat: with the pad fixed in its seat, record five firm presses with 2&#8202;s rests and run "
              "<code>pad_shift_check.py</code> on it. Rest offsets of a few &#181;T pass; offsets of tens of &#181;T that a "
              "rigid motion fits mean the pad still moves.</li>"
              "<li>N52 magnets: recalibrate (key c). The fit should give $k$ near 1, all poles one way and a rest residual "
              "below 0.43&#8202;mT; a $k$ well below 1 or a larger residual means the magnets' depth or the pad's seat on "
              "the board differs from the FE row.</li>"
              "<li>A force reference without the UR5: a kitchen scale under one of the printed indenters, the finger "
              "pressed onto it at the face centre, a flank and the tip end at about 1, 2 and 4&#8202;N, with the scale's "
              "reading written down for each hold. The ratio of the estimated to the scale's force at each point tests "
              "$E^*$ and $k$; errors beyond 2&#8211;3 model sigmas falsify the model there.</li>"
              "<li>The UR5 session with the HEX-E sensor, on the robot's computer. Take points over the face and both flanks "
              "from the basis's pad points, with three normal forces and two shear directions each. Return to a "
              "reference point every ten presses to measure drift, and rest 30&#8202;s for the pad's recovery. Refit $k$, "
              "$\\boldsymbol\\delta$, $\\theta_0$, $E^*$ and per-magnet strengths on half of the data, and compare the other half's "
              "errors with the model 1-sigma.</li>"
              "<li>Large presses: the FE row's rod responses at 0.5, 1 and 3&#8202;N give the stiffening as a force-magnitude "
              "correction, to be checked against the reference.</li>"
              "<li>Temperature: with the pad at rest, hold a warm hand around the finger without touching the pad for 60&#8202;s, "
              "then take it away, with the viewer logging, and run the study's thermal step on that stretch of the stream "
              "log. The correction passes if no contact shows while the chips warm by 2&#8211;3&#8202;&#176;C. A drift that "
              "grows with the temperature after the correction means that the magnets' coefficient needs a term of its own, "
              "with the pad's temperature lagging the chips'. Firmware with TCMP_EN 1, the chips' own compensation, is the "
              "alternative to test the same way.</li>"
              "<li>Re-zero: press and hold at about 0.5, 1 and 2&#8202;N for 2, 5 and 10&#8202;s each on the kitchen scale "
              "with the viewer logging. The study's replay counts the holds that the re-zero takes in. None of the holds "
              "under 3&#8202;s should be taken in.</li>"
              "<li>Pole pattern: seat the N52 magnets in the checkerboard in two pads, fix them side by side 26, 30 and "
              "40&#8202;mm apart and facing at 10 and 20&#8202;mm, and record one board while the other pad moves in 5&#8202;mm "
              "steps. The model of Figure 10 passes if the measured change, less its uniform part, is within 30&#8202;% of "
              "it; then subtract the modelled field and check that no contact shows.</li>"
              "<li>Floor: after the seat fix and the N52 magnets, repeat the night at rest and the floor sweep (Table 7); "
              "lower the floor to the smallest value that still shows no contact at rest.</li></ul>"
              f"<p>Status of the printed-fingertip work: {link(STATUS_PAGE, 'status page')}.</p>"))

    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "Model-based contact estimation",
                 "H1": "Model-based estimation of contact position and force on the magnetic fingertip",
                 "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-20", "LEDE": lede,
                 "BYLINE": f"Built {time.strftime('%Y-%m-%d %H:%M')} from results.json ({res['when']}; Section 9 of "
                           f"2026-10-21, Section 10 of 2026-10-22); the readings at rest and the recordings of Sections 8 and "
                           f"9 are measured, the other presses and the fields of Section 10 modelled.",
                 "FILM": film(1, MEDIA / "20261020-viewer_surface_presses.mp4", MEDIA / "20261020-viewer_surface_presses.png",
                              "The browser viewer replaying five synthetic presses made by the model with the measured noise "
                              "and the magnets as found (Table 4): the palmar face, the +y flank, the face's edge on the -y "
                              "side (behind the pad in this view), the tip end and a press with shear along the finger. "
                              "Arrow: estimated force; ring: 1-sigma position ellipse; colour: pad pressure on a 300&#8202;kPa "
                              "scale; deformation drawn 20 times, at most 2&#8202;mm."),
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=(
        f"A model of the five-chip board turns its 15 field components into a contact point and force anywhere on the "
        f"pad's surface, from an FE basis at the {fs['pads']} sphere pads; on simulated presses with N52 magnets the "
        f"median errors are {s52['all']['pos_median_mm']:.2f} mm and {s52['all']['press_abs_median_N']:.2f} N. In the "
        f"first presses by hand the pad rocked in its seat, leaving offsets of {min(r['offset_uT'] for r in big):.0f}-"
        f"{max(r['offset_uT'] for r in big):.0f} uT that read as standing contacts. The rest reading drifts "
        f"{k_lo:.2f}-{k_hi:.2f} % of the field per degree C of chip temperature; a correction from each chip's temperature, "
        f"a 2 uT noise floor and a re-zero after 3 s at rest stop the viewer showing contacts at rest."))
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    build()
