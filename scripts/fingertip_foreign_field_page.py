#!/usr/bin/env python3
"""Page of the test for fields from outside the pad in the fingertip's contact estimator (2026-10-23): the owner's touches
of steel on the second fingertip, why the estimator reads them as presses, the test (fingertip_wrench_model.ForeignField),
its scores on the UR5 sessions' holds and streams and on the desk recordings, and the toggle in the viewer. Reads
docs/experiments/20261023-foreign_field_rejection/data/ (scripts/fingertip_foreign_field.py) and the film of
scripts/fingertip_foreign_field_film.py; writes docs/experiments/20261023-foreign_field_rejection/
20261023-foreign_field_rejection.html.

    .venv/bin/python scripts/fingertip_foreign_field_page.py
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
import fingertip_foreign_field as FF  # noqa: E402
import fingertip_wrench_model as W  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import ur5_tip_calibration_page as P  # noqa: E402   (esc, code, fig, film, table, plt_setup, TEX_RE)

D = FF.OUT
DATA = FF.DATA
MEDIA = D / "media"
PAGE = D / "20261023-foreign_field_rejection.html"
TEMPLATE = ROOT / "scripts/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
FILM = MEDIA / "20261023-viewer_steel_touches.mp4"
CHIPS = ("U4", "U1", "U2", "U3", "U5")                    # along the finger from the mount end, then the two sides
OFF, ON = "#9a9a9a", "#000000"


def link(p, text):
    return f'<a href="{P.esc(os.path.relpath(p, D))}">{text}</a>'


def load():
    J = {k: json.loads((DATA / f"{k}.json").read_text())
         for k in ("touches", "holds", "approach", "frames_touches", "frames_round1", "frames_round2", "frames_sphere2",
                   "frames_rod1", "episodes_sphere2", "episodes_rod1")}
    J["cal"] = W.Calib.load(FF.CAL)
    # the touches' and the presses' patterns against the images of the magnets in a steel plane at the face
    img = FF.image(W.Model(J["cal"]), FF.FACE_X)
    six = np.array([p["d_uT"] for p in J["touches"]["plateaus"] if p["group"] == "six"])
    pr = np.array([p["d_uT"] for p in J["touches"]["presses"]])
    cos = lambda a: float(a @ img / np.linalg.norm(a) / np.linalg.norm(img))   # noqa: E731
    J["cos"] = dict(t13=cos(six[:3].mean(0)), t46=cos(six[3:].mean(0)), press=cos(pr.mean(0)))
    # the frames of the six touches still shown with the test on: where in each touch, their force and Lambda
    t, Fr = series("touches")
    rel, f, lam, dd = [], [], [], []
    for p in (p for p in J["touches"]["plateaus"] if p["group"] == "six"):
        a = J["touches"]["window"]["t0"] + p["t0_s"]
        m = (t >= a - 0.3) & (t <= a + p["s"] + 0.3) & (Fr[:, 4] > 0)
        rel += [(x, p["s"]) for x in t[m] - a]
        f += Fr[m, 2].tolist()
        lam += Fr[m, 1].tolist()
        dd += Fr[m, 0].tolist()
    onset = [x for x, s in rel if x < 0.5 * s]
    J["left"] = dict(n=len(rel), onset=len(onset), end=len(rel) - len(onset), onset_s=[min(onset), max(onset)],
                     f=[min(f), max(f)], lam=[min(lam), max(lam)], d=[min(dd), max(dd)])
    return J


# ------------------------------------------------------------------------------------------------ figures
def fig_patterns(J, out):
    """The change per chip in the tip frame: the means of touches 1-3 and 4-6, the images of the magnets in a steel plane
    at the face, and the mean of the six presses."""
    plt = P.plt_setup()
    M = W.Model(J["cal"])
    R = J["touches"]
    six = np.array([p["d_uT"] for p in R["plateaus"] if p["group"] == "six"])
    pr = np.array([p["d_uT"] for p in R["presses"]])
    rows = [("(a) Touches 1 to 3, mean", six[:3].mean(0)), ("(b) Touches 4 to 6, mean", six[3:].mean(0)),
            ("(c) Images of the magnets in a steel plane at the face, computed", FF.image(M, FF.FACE_X)),
            ("(d) Six presses with the printed sphere, mean", pr.mean(0))]
    names = [W.U_ORDER[u] for u in J["cal"].perm]
    order = [names.index(u) for u in CHIPS]
    fig, axes = plt.subplots(4, 1, figsize=(10.0, 10.5), sharex=True, sharey=True)
    w = 0.26
    cols = ("#304080", "#b06010", "#3a8a3a")
    for ax, (title, d) in zip(axes, rows):
        T = np.einsum("cji,cj->ci", M.R, d.reshape(5, 3))[order]
        for k in range(3):
            ax.bar(np.arange(5) + (k - 1) * w, T[:, k], w, color=cols[k], label=f"{'xyz'[k]}")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Change (µT)")
        ax.set_ylim(-120, 210)
        ax.grid(axis="y", alpha=0.3)
    labels = {"U4": "U4\nmount end", "U1": "U1\ncentre", "U2": "U2\ntip end", "U3": "U3\n+y side", "U5": "U5\n−y side"}
    axes[-1].set_xticks(range(5), [labels[u] for u in CHIPS])
    axes[0].legend(title="Tip-frame component", ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def series(name):
    z = np.load(DATA / f"frames_{name}.npz")
    return float(z["t0"]) + z["dt"].astype(float), z["F"].astype(float)


def fig_traces(J, out):
    """The change, Lambda and the force shown over 15:38:30-15:40:30 and over the six touches."""
    plt = P.plt_setup()
    t, F = series("touches")
    day = J["touches"]["window"]["t0"]
    win = [("From 15:38:30 to 15:40:30", FF.T.clock("15:38:30", day), FF.T.clock("15:40:30", day)),
           ("The six touches, from 15:43:12", FF.T.clock("15:43:12", day), FF.T.clock("15:43:46", day))]
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 8.6), sharex="col", gridspec_kw=dict(width_ratios=[2.2, 1.0]))
    for j, (title, a, b) in enumerate(win):
        m = (t >= a) & (t <= b)
        x = t[m] - a
        axes[0, j].plot(x, F[m, 0], color="black", lw=0.9)
        axes[0, j].set_title(f"({'ab'[j]}) {title}", loc="left")
        lam = np.where(np.isfinite(F[m, 1]), F[m, 1], np.nan)
        axes[1, j].plot(x, np.clip(lam, -300, 1000), ".", ms=2.5, color="black")
        axes[1, j].axhline(25, color="#c0392b", lw=1.2, ls="--")
        axes[1, j].set_yscale("symlog", linthresh=30)
        axes[2, j].plot(x, F[m, 2] * F[m, 3], color=OFF, lw=1.4, label="Test off")
        axes[2, j].plot(x, F[m, 2] * F[m, 4], color=ON, lw=1.1, label="Test on")
        axes[2, j].set_xlabel("Time (s)")
        for ax in axes[:, j]:
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Board change (µT)")
    axes[1, 0].set_ylabel("Λ")
    axes[2, 0].set_ylabel("Force shown (N)")
    axes[1, 1].text(0.98, 0.06, "threshold 25", transform=axes[1, 1].transAxes, ha="right", color="#c0392b", fontsize=12)
    axes[2, 1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_lambda(J, out):
    """Lambda of every UR5 hold and every desk plateau, with the threshold."""
    plt = P.plt_setup()
    H, R = J["holds"], J["touches"]
    rows = [("UR5 presses, sphere2 and rod1", np.r_[H["sphere2_press"]["lam"], H["rod1_press"]["lam"]], "#304080"),
            ("UR5 slides, rolls and twists", np.r_[H["sphere2_turn"]["lam"], H["rod1_turn"]["lam"]], "#3a8a3a"),
            ("Presses with the printed sphere, desk", np.array([p["lam"] for p in R["presses"]]), "#b06010"),
            ("Plateaus before the six touches", np.array([p["lam"] for p in R["plateaus"] if p["group"] != "six"]), "#777777"),
            ("The six touches of steel", np.array([p["lam"] for p in R["plateaus"] if p["group"] == "six"]), "#c0392b")]
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    rng = np.random.default_rng(1)
    for k, (name, L, c) in enumerate(rows):
        y = len(rows) - 1 - k + rng.uniform(-0.18, 0.18, len(L))
        ax.plot(L, y, "o", ms=4 if len(L) > 50 else 6, mfc="none" if len(L) > 50 else c, mec=c, alpha=0.8)
        ax.text(1.02, len(rows) - 1 - k, f"n = {len(L)}", transform=ax.get_yaxis_transform(), va="center", fontsize=12)
    ax.axvline(25, color="#c0392b", ls="--", lw=1.2)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(-1000, 1500)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlabel("Λ = χ² of the best contact less χ² of the far field")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_approach(J, out):
    plt = P.plt_setup()
    A = J["approach"]["rows"]
    x = np.array([r["change_uT"] for r in A])
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax[0].plot(x, [r["shown_off"] for r in A], "o-", color=OFF, label="Test off")
    ax[0].plot(x, [r["shown_on"] for r in A], "s-", color=ON, label="Test on")
    ax[0].set_ylabel("Share of frames with a contact shown")
    ax[0].legend()
    ax[1].plot(x, [r["f_mean_shown_off"] if r["shown_off"] > 0 else np.nan for r in A], "o-", color=OFF, label="Test off")
    ax[1].plot(x, [r["f_mean_shown_on"] if r["shown_on"] > 0 else np.nan for r in A], "s-", color=ON, label="Test on")
    ax[1].set_ylabel("Mean force shown (N)")
    for k, a in enumerate(ax):
        a.set_xlabel("Board change of the scaled touch (µT)")
        a.set_title(f"({'ab'[k]})", loc="left")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def build():
    from fingertip_foreign_field_text import sections          # the prose, kept apart from the plotting
    J = load()
    MEDIA.mkdir(parents=True, exist_ok=True)
    F = dict(patterns=MEDIA / "20261023-field_patterns.png", traces=MEDIA / "20261023-touch_traces.png",
             lam=MEDIA / "20261023-lambda.png", approach=MEDIA / "20261023-approach.png",
             film=FILM, poster=FILM.with_suffix(".png"))
    fig_patterns(J, F["patterns"])
    fig_traces(J, F["traces"])
    fig_lambda(J, F["lam"])
    fig_approach(J, F["approach"])
    S, lede, summary, film_html = sections(J, F, P, link)
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "Foreign field rejection",
                 "H1": "Rejection of fields from steel and other sources outside the pad in the fingertip&#8217;s contact estimator",
                 "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-23", "LEDE": lede,
                 "BYLINE": ("Touches by the owner on the workstation&#8217;s board, 2026-10-23 15:38 to 15:44. Analysis by "
                            + P.code("scripts/fingertip_foreign_field.py") + "; page by "
                            + P.code("scripts/fingertip_foreign_field_page.py") + "."),
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
