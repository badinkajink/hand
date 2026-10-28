#!/usr/bin/env python3
"""Page on magnetic crosstalk between neighbouring printed fingertips (hand-off 20261027-docs_update_impratio, step 4;
owner 2026-10-27: "crosstalk is going to be problem, so itd be helpful to present some compelling results on that").
Reads the rows of scripts/fingertip_crosstalk.py (data/budget.json: v2 ID 1's field model under a five- or four-magnet
neighbour; data/sharpie.json: the owner's sharpie pinch with v1 ID 2 and ID 3) and, once, the multi-board viewer's
streams of that pinch for Figure 2's trace (kept in media/20261027-sharpie_trace.json, since logs/ is not tracked).
Writes docs/experiments/20261027-fingertip_crosstalk/20261027-magnetic_crosstalk_between_neighbouring_fingertips.html.

    ~/.claude/bin/resguard.sh run --mem 2G --cpu 100 -- .venv/bin/python scripts/fingertip_crosstalk_page.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import ur5_tip_calibration_page as P  # noqa: E402   (esc, fig, table, TEX_RE, plt_setup)

D = ROOT / "docs/experiments/20261027-fingertip_crosstalk"
DATA, MEDIA = D / "data", D / "media"
PAGE = D / "20261027-magnetic_crosstalk_between_neighbouring_fingertips.html"
TEMPLATE = ROOT / "scripts/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TRACE = MEDIA / "20261027-sharpie_trace.json"
EXP = ROOT / "docs/experiments"
L = dict(
    overview=ROOT / "docs/overviews/printed_fingertips.html",
    cal=EXP / "20261026-fingertip_calibration/20261026-fingertip_calibration_on_the_ur5.html",
    steel=EXP / "20261023-foreign_field_rejection",
    holder=EXP / "20261027-fingertip_holder/README.md",
    v3=EXP / "20261027-v3_tip_calibration_and_crosstalk/README.md",
    gt=EXP / "20261022-ur5_tip_calibration/20261022-ur5_tip_ground_truth_and_predictors.html",
)
BLUE, ORANGE, GREY = "#2e5fa8", "#b06010", "0.45"
FACING_W = (0, 5, 10, 15, 20, 30)
SIDE_G = (0, 2, 5, 10, 20)


def link(p, text):
    return f'<a href="{P.esc(os.path.relpath(p, D))}">{text}</a>'


def sig2(x):
    """Two significant figures with thousands separators."""
    if x == 0:
        return "0"
    d = 10 ** (math.floor(math.log10(abs(x))) - 1)
    v = round(x / d) * d
    return f"{v:,.0f}" if abs(v) >= 10 else f"{v:.1f}"


def rows_by(B):
    out = {}
    for r in B["rows"]:
        kind, val = r["pose"].split(", ")
        out[(r["source"], "facing" if kind == "facing" else "side", float(val.split()[1]))] = r
    return out


# ------------------------------------------------------------------------------------------------ the sharpie trace

def sharpie_trace(dt=0.2):
    """|change| of each board from rest (thermally corrected, as fingertip_crosstalk.window_change) in bins of dt s from
    12:17:50 to 12:19:15, read once from the viewer's streams and kept in media/."""
    if TRACE.exists():
        return json.loads(TRACE.read_text())
    import fingertip_crosstalk as FC
    import fingertip_wrench_model as W
    kappa = W.load_thermal()
    t0, t1 = FC.stamp("12:17:50"), FC.stamp("12:19:15")
    out = dict(t0="12:17:50", dt_s=dt, boards={}, source=[str((FC.STREAM / f).relative_to(ROOT)) for f in FC.SHARPIE["files"].values()])
    for name, fn in FC.SHARPIE["files"].items():
        a = np.memmap(FC.STREAM / fn, np.dtype([("t", "<f8"), ("r", "<f4", (20,))]), mode="r")
        i0, i1 = np.searchsorted(a["t"], FC.stamp(FC.SHARPIE["rest"][0]) - 5), np.searchsorted(a["t"], t1 + 1)
        t, r = np.array(a["t"][i0:i1]), np.array(a["r"][i0:i1]).reshape(-1, 5, 4).astype(float)
        s = (t >= FC.stamp(FC.SHARPIE["rest"][0])) & (t < FC.stamp(FC.SHARPIE["rest"][1]))
        ref = W.thermal_correct(r[s][:, :, 1:], r[s, :, 0].mean(0), kappa).reshape(-1, 15).mean(0)
        vals = []
        for b in np.arange(t0, t1, dt):
            m = (t >= b) & (t < b + dt)
            if m.sum() < 3:
                vals.append(None)
                continue
            d = W.thermal_correct(r[m][:, :, 1:], r[m, :, 0].mean(0), kappa).reshape(-1, 15).mean(0) - ref
            vals.append(round(float(np.linalg.norm(d)), 2))
        out["boards"][name] = vals
    MEDIA.mkdir(parents=True, exist_ok=True)
    TRACE.write_text(json.dumps(out) + "\n")
    return out


def sharpie_numbers(S):
    """Per window: the uniform field's magnitude, the forces without and with the two-board pose fit, and the distance
    between the two contact points in ID 2's frame (ID 3's point placed by the fitted pose)."""
    from scipy.spatial.transform import Rotation
    out = []
    for w in S["windows"]:
        R = Rotation.from_rotvec(w["rotvec"]).as_matrix()
        t = np.asarray(w["t_mm"])
        pA = np.asarray(w["with_crosstalk"]["ID 2"]["p"])
        pB = R @ np.asarray(w["with_crosstalk"]["ID 3"]["p"]) + t
        out.append(dict(
            t=w["window"], d=w["d_uT"], chi0=w["chi2_without"], chi1=w["chi2_with"], x=w["crosstalk_uT"],
            g0=[float(np.linalg.norm(w["without"][k]["g"])) for k in ("ID 2", "ID 3")],
            f0=[w["without"][k]["f_n"] for k in ("ID 2", "ID 3")], f1=[w["with_crosstalk"][k]["f_n"] for k in ("ID 2", "ID 3")],
            apart=float(np.linalg.norm(pA - pB)), apex=w["apex_distance_mm"], angle=w["normal_angle_deg"]))
    return out


# ------------------------------------------------------------------------------------------------ figures

def fig_budget(B):
    plt = P.plt_setup()
    rb = rows_by(B)
    press = B["press_signal_uT"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    for ax, kind, xs, xl, title in ((axs[0], "facing", FACING_W, "Gap W between the facing pads (mm)", "(a) Pads facing"),
                                    (axs[1], "side", SIDE_G, "Gap g between the flanks (mm)", "(b) Side by side")):
        ax.axhspan(press["0.2-0.4 N"], press["1.5-3 N"], color="0.92", zorder=0)
        ax.axhline(press["0.4-0.7 N"], color="black", ls="--", lw=1.1, zorder=1)
        for src, col, mk in (("five N52", BLUE, "o"), ("four N52", ORANGE, "s")):
            y = [rb[(src, kind, float(x))]["norm_uT"] for x in xs]
            yk = [rb[(src, kind, float(x))]["kin"]["0.3 mm, 0.3 deg"]["residual_uT_median"] for x in xs]
            lab = "Five N52 magnets" if src == "five N52" else "Four N52 magnets"
            ax.semilogy(xs, y, "-" + mk, color=col, ms=7, lw=1.6, label=f"{lab}: field at the board")
            ax.semilogy(xs, yk, "--" + mk, color=col, ms=7, lw=1.3, mfc="white",
                        label=f"{lab}: left after subtraction at 0.3 mm, 0.3\N{DEGREE SIGN} off")
        ax.set_xlabel(xl)
        ax.set_xticks(xs)
        ax.set_title(title, loc="left")
        ax.set_ylim(0.1, 2e4)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("Change at v2 ID 1's board (\N{MICRO SIGN}T)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h, l_ = axs[0].get_legend_handles_labels()
    h += [Line2D([], [], color="black", ls="--", lw=1.1), Patch(color="0.92")]
    l_ += ["A 0.5 N face press", "Face presses of 0.2-3 N"]
    fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13), fontsize=11.5)
    fig.tight_layout()
    MEDIA.mkdir(parents=True, exist_ok=True)
    out = MEDIA / "20261027-crosstalk_budget.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_sharpie(S, T, Sn):
    import fingertip_crosstalk as FC
    plt = P.plt_setup()
    fig, (a, b) = plt.subplots(2, 1, figsize=(11.0, 8.2), sharex=True, gridspec_kw=dict(height_ratios=(1.15, 1.0)))
    t0 = FC.stamp(T["t0"])
    ref = FC.stamp("12:18:00")
    for name, col in (("ID 2", BLUE), ("ID 3", ORANGE)):
        v = np.array([np.nan if x is None else x for x in T["boards"][name]], float)
        tt = t0 + T["dt_s"] * np.arange(len(v)) - ref
        a.plot(tt, v, "-", color=col, lw=1.3, label=f"{name}: change at the board")
    for k, w in enumerate(Sn):
        s0, s1 = FC.stamp(w["t"][0]) - ref, FC.stamp(w["t"][1]) - ref
        for ax in (a, b):
            ax.axvspan(s0, s1, color="0.9", zorder=0)
        mid = 0.5 * (s0 + s1)
        for j, col in ((0, BLUE), (1, ORANGE)):
            a.plot([mid], [math.sqrt(5) * w["g0"][j]], "D", color=col, mfc="white", ms=7, mew=1.4,
                   label=("Uniform part, over the five chips" if k == 0 and j == 0 else None))
            b.plot([mid - 0.25], [w["f0"][j]], "o", color=col, mfc="white", ms=8, mew=1.5)
            b.plot([mid + 0.25], [w["f1"][j]], "o", color=col, ms=8)
    a.set_ylabel("Change from rest (\N{MICRO SIGN}T)")
    a.set_ylim(0, 420)
    a.set_title("(a) The boards during the pinch", loc="left")
    b.set_ylabel("Estimated contact force (N)")
    b.set_ylim(0, 13.5)
    b.set_xlabel("Time from 12:18:00 on 2026-10-27 (s)")
    b.set_title("(b) Force of each board's contact, per window", loc="left")
    from matplotlib.lines import Line2D
    hb = [Line2D([], [], color=BLUE, marker="o", ls="", ms=8, label="ID 2"), Line2D([], [], color=ORANGE, marker="o", ls="", ms=8, label="ID 3"),
          Line2D([], [], color="0.3", marker="o", ls="", mfc="white", ms=8, label="Viewer's estimate"),
          Line2D([], [], color="0.3", marker="o", ls="", ms=8, label="With the two-board pose fit")]
    b.legend(handles=hb, loc="upper right", frameon=False, fontsize=12, ncol=2)
    a.legend(loc="upper right", frameon=False, fontsize=12)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    a.set_xlim(-8, 72)
    fig.tight_layout()
    out = MEDIA / "20261027-sharpie_pinch.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ text

def sections(B, S, Sn, f1, f2):
    rb = rows_by(B)
    r5 = lambda kind, x: rb[("five N52", kind, float(x))]  # noqa: E731
    r4 = lambda kind, x: rb[("four N52", kind, float(x))]  # noqa: E731
    k3 = lambda r: r["kin"]["0.3 mm, 0.3 deg"]  # noqa: E731
    k1 = lambda r: r["kin"]["0.1 mm, 0.1 deg"]  # noqa: E731
    press = B["press_signal_uT"]
    p05 = press["0.4-0.7 N"]
    crb2 = lambda kind, xs: [r5(kind, x)["crb"]["1"]["two_boards"] for x in xs]  # noqa: E731
    cf, cs = crb2("facing", FACING_W), crb2("side", SIDE_G)
    known = r5("facing", 0)["crb"]["known"]["one_board"]
    shown = [r for r in B["rows"] if r["shown_N"] > 0]
    gated = [r for r in B["rows"] if r["shown_gate_N"] > 0]
    w0 = Sn[0]
    later = Sn[1:]
    g_med = [float(np.median([w["g0"][j] for w in Sn])) for j in (0, 1)]

    lede = (
        "A neighbouring fingertip&#8217;s magnets change the reading of a magnetic fingertip&#8217;s board far more than a press "
        f"does. In v2 ID 1&#8217;s field model a five-magnet neighbour adds {r5('facing', 0)['norm_uT']:.0f}&#8202;&#181;T with the "
        f"pads facing and touching. Side by side it adds {r5('side', 0)['norm_uT']:,.0f}&#8202;&#181;T. A 0.5&#8202;N press changes "
        f"the board by {p05:.0f}&#8202;&#181;T. Subtracting the field predicted from a pose 0.3&#8202;mm and 0.3&#176; off leaves "
        f"{k3(r5('facing', 0))['residual_uT_median']:.0f} and {k3(r5('side', 0))['residual_uT_median']:.0f}&#8202;&#181;T. The "
        "neighbour&#8217;s pose can instead be fitted from both boards, with the joint angles as a 1&#8202;mm, 1&#176; prior. The "
        f"Cram&#233;r&#8211;Rao bound on a press at the face&#8217;s centre is then {min(cf):.3f}&#8211;{max(cf):.3f}&#8202;N facing and "
        f"{min(cs):.3f}&#8211;{max(cs):.3f}&#8202;N side by side, against {known:.3f}&#8202;N with the pose known. In the owner&#8217;s "
        "pinch of a marker pen between two v1 tips, such a fit explained the first window. It lowered the estimated forces from "
        f"{w0['f0'][0]:.1f} and {w0['f0'][1]:.1f}&#8202;N to {w0['f1'][0]:.1f} and {w0['f1'][1]:.1f}&#8202;N.")
    summary = (
        f"Magnetic crosstalk between neighbouring fingertips (v2 ID 1's field model): a five-N52 neighbour adds "
        f"{r5('facing', 0)['norm_uT']:.0f} uT facing and {r5('side', 0)['norm_uT']:,.0f} uT side by side when touching, against "
        f"{p05:.0f} uT for a 0.5 N press; subtracting the field at a pose 0.3 mm and 0.3 deg off leaves "
        f"{k3(r5('facing', 0))['residual_uT_median']:.0f} and {k3(r5('side', 0))['residual_uT_median']:.0f} uT; fitting the pose from "
        f"both boards with a 1 mm, 1 deg prior keeps the press bound at {min(cf):.3f}-{max(cf):.3f} N facing and "
        f"{min(cs):.3f}-{max(cs):.3f} N side by side ({known:.3f} N known); the sharpie pinch's first window fitted from both "
        f"boards (chi-square {sum(w0['chi0']):,.0f} to {sum(w0['chi1']):.0f}; {w0['f0'][0]:.1f} and {w0['f0'][1]:.1f} N to "
        f"{w0['f1'][0]:.1f} and {w0['f1'][1]:.1f} N).")

    S_ = []
    terms = [
        ("Crosstalk", "The change in one fingertip&#8217;s board reading caused by another fingertip&#8217;s magnets, with nothing "
         "touching the first."),
        ("Board, change", "The five MLX90393 chips under the pad, read as 15 components (five chips, three axes) in &#181;T. The "
         "change is the reading less the reading at rest, corrected for temperature."),
        ("Field at the board", "The norm of the neighbour&#8217;s field over the 15 components, in &#181;T, from the dipole model of "
         "the magnets. The largest chip is the largest single component."),
        ("Press signal", "The norm of the change under a face press of v2 ID 1 on the UR5, median over the presses in a band of "
         f"force. It is {press['0.2-0.4 N']:.0f}&#8202;&#181;T at 0.2&#8211;0.4&#8202;N and {p05:.0f}&#8202;&#181;T at 0.4&#8211;0.7&#8202;N. "
         f"At 0.7&#8211;1.5&#8202;N it is {press['0.7-1.5 N']:.0f}&#8202;&#181;T, and at 1.5&#8211;3&#8202;N {press['1.5-3 N']:.0f}&#8202;&#181;T."),
        ("Receiving tip, neighbour", "The receiving tip is v2 ID 1, five N52 magnets, with its field model fitted at rest in UR5 "
         "session 20261026-192158-v2id1. The neighbour is the same five-magnet tip, or four N52 magnets in an alternating "
         "5.0&#8202;mm square (pad 05R4), both at v2 ID 1&#8217;s magnet strength."),
        ("Facing, W", "The pads face each other across a gap W between their apexes, as in a pinch, in mm. The apex is the "
         "centre of a pad&#8217;s face."),
        ("Side by side, g", "The tips lie side by side with their apexes level and a gap g between their flanks, in mm."),
        ("Viewer, phantom contact", "The multi-board viewer&#8217;s estimator shows a contact when its fit explains a "
         "&#967;&#178; of more than 25 and the force is at least 0.25&#8202;N. It uses a 2&#8202;&#181;T floor and 50&#8202;ms of "
         "samples. A phantom contact is one shown with nothing pressing."),
        ("Steel test", "The viewer&#8217;s test for a foreign field: a field that varies slowly across the board, fitted as a "
         "far source, rejects a contact when its statistic &#923; exceeds 25 (" + link(L["steel"], "foreign-field page")
         + "). The part of the neighbour&#8217;s field that this fit does not absorb is listed as left after the far-field fit."),
        ("Pose error", "An error of the neighbour&#8217;s pose, as from the joint angles. It is drawn with a standard deviation "
         "per axis of 0.1, 0.3 or 1&#8202;mm and as many degrees, 100 draws per pose."),
        ("Residual", "The field left after the field predicted at the erroneous pose is subtracted, median over the draws, in "
         "&#181;T."),
        ("Light and firm presses", "Thirty face presses of v2 ID 1 at 0.3&#8211;0.7&#8202;N and thirty at 1.5&#8211;2.5&#8202;N, "
         "to which the residuals are added."),
        ("Cram&#233;r&#8211;Rao bound", "The smallest standard deviation of an unbiased estimate of a press along the face normal "
         "at the face&#8217;s centre, in N. The board&#8217;s noise is that of the viewer&#8217;s 50&#8202;ms window. The press&#8217;s force, the "
         "uniform field, the magnets&#8217; strength and the neighbour&#8217;s pose are unknown. The joint angles enter as a Gaussian "
         "prior on the pose."),
        ("Two-board pose fit", "The neighbour&#8217;s pose fitted to both boards&#8217; readings at once, each board explained by a "
         "contact of its own and the other tip&#8217;s field."),
        ("Uniform field", "The part of a board&#8217;s change that is the same at every chip, fitted by the estimator. The "
         "Earth&#8217;s field of about 50&#8202;&#181;T gives one when a tip turns."),
        ("&#967;&#178;", "The squared, noise-weighted residual of a board&#8217;s change after the estimator&#8217;s best contact and "
         "uniform field. A smaller value means the change is better explained."),
    ]
    S_.append(("terms", "Terms and metrics", '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"))

    # 2. the field at the board
    def brow(r):
        return [r["source"].capitalize(), r["pose"].replace("side by side", "Side by side").replace("facing", "Facing"),
                f"{r['norm_uT']:,.0f}" if r["norm_uT"] >= 10 else f"{r['norm_uT']:.1f}",
                f"{r['after_far_uT']:,.0f}" if r["after_far_uT"] >= 10 else f"{r['after_far_uT']:.1f}",
                f"{r['shown_N']:.1f}" if r["shown_N"] > 0 else "none", f"{r['shown_gate_N']:.1f}" if r["shown_gate_N"] > 0 else "none",
                f"{k1(r)['residual_uT_median']:.1f}", f"{k3(r)['residual_uT_median']:.1f}",
                f"{r['kin']['1 mm, 1 deg']['residual_uT_median']:.1f}", f"{100 * k3(r)['shown_frac']:.0f}",
                f"{r['pose_gradient_uT_per_mm']:,.0f}"]
    tab1 = P.table(
        ["Neighbour", "Pose", "Field at the board (&#181;T)", "Left after the far-field fit (&#181;T)", "Contact shown (N)",
         "Shown with the steel test (N)", "Residual, 0.1&#8202;mm, 0.1&#176; (&#181;T)", "Residual, 0.3&#8202;mm, 0.3&#176; (&#181;T)",
         "Residual, 1&#8202;mm, 1&#176; (&#181;T)", "Phantoms at 0.3&#8202;mm, 0.3&#176; (%)", "Gradient (&#181;T/mm)"],
        [brow(r) for r in B["rows"]], num_cols=tuple(range(2, 11)), n=1,
        caption=f"The neighbour&#8217;s field at v2 ID 1&#8217;s board by pose, and what is left after subtracting the field "
                f"predicted at an erroneous pose. A 0.5&#8202;N press changes the board by {p05:.0f}&#8202;&#181;T. Gradient: the "
                "largest change of the field per mm of the neighbour&#8217;s position. Script: "
                "<code>scripts/fingertip_crosstalk.py budget</code>; rows in <code>data/budget.json</code>.")
    gate4 = gated[0] if gated else None
    field = (
        f"<p>A five-magnet neighbour changes v2 ID 1&#8217;s reading by {r5('facing', 0)['norm_uT']:.0f}&#8202;&#181;T with the pads "
        f"facing and touching (Figure&#160;1a). Across a 10&#8202;mm gap the change is {r5('facing', 10)['norm_uT']:.0f}&#8202;&#181;T, and "
        f"across 30&#8202;mm it is {r5('facing', 30)['norm_uT']:.0f}&#8202;&#181;T, still twice the {p05:.0f}&#8202;&#181;T of a "
        f"0.5&#8202;N press. Side by side the change at contact is {r5('side', 0)['norm_uT']:,.0f}&#8202;&#181;T, eight times the "
        f"facing value. It falls to {r5('side', 10)['norm_uT']:.0f}&#8202;&#181;T at a 10&#8202;mm gap and "
        f"{r5('side', 20)['norm_uT']:.0f}&#8202;&#181;T at 20&#8202;mm (Figure&#160;1b). The largest single component reaches "
        f"{r5('side', 0)['max_chip_uT']:,.0f}&#8202;&#181;T side by side.</p>"
        f"<p>Four N52 magnets in an alternating square give {r4('facing', 0)['norm_uT']:.0f}&#8202;&#181;T facing and touching, and "
        f"{r4('facing', 5)['norm_uT']:.0f}&#8202;&#181;T at 5&#8202;mm. Their field falls faster with distance, since the "
        "alternating poles cancel far away. Side by side they give "
        f"{r4('side', 0)['norm_uT']:,.0f}&#8202;&#181;T at contact, and {r4('side', 10)['norm_uT']:.0f}&#8202;&#181;T at 10&#8202;mm.</p>"
        "<p>The field changes quickly with the neighbour&#8217;s pose. Facing and touching it changes by "
        f"{r5('facing', 0)['pose_gradient_uT_per_mm']:.0f}&#8202;&#181;T per mm and {r5('facing', 0)['pose_gradient_uT_per_deg']:.0f}"
        f"&#8202;&#181;T per degree. Side by side it changes by {r5('side', 0)['pose_gradient_uT_per_mm']:,.0f}&#8202;&#181;T per mm. "
        "Small motions of the fingers therefore appear as a field that varies by tens to hundreds of &#181;T. Without the "
        f"steel test the viewer would show a phantom contact of {min(r['shown_N'] for r in shown):.1f}&#8211;"
        f"{max(r['shown_N'] for r in shown):.0f}&#8202;N at {len(shown)} of the 22 poses. The steel test rejects all of them "
        + (f"but one, an {gate4['shown_gate_N']:.1f}&#8202;N phantom from four magnets side by side at "
           f"{gate4['pose'].split('g ')[1]}." if gate4 else ".")
        + " Much of the side-by-side field is not a far field: "
        f"{r5('side', 0)['after_far_uT']:,.0f}&#8202;&#181;T of it is left after the steel test&#8217;s far-field fit at contact.</p>")
    S_.append(("field", "Field of a neighbouring fingertip at the board",
               field + P.fig(1, f1, "Change at v2 ID 1&#8217;s board caused by a neighbouring fingertip, in its field model. (a) Pads "
                             "facing across a gap W. (b) Side by side with a gap g between the flanks. Solid: the neighbour&#8217;s "
                             "field. Dashed: the median left after subtracting the field predicted at a pose 0.3&#8202;mm and "
                             "0.3&#176; off. Blue: a five-magnet neighbour like v2 ID 1. Orange: four N52 magnets in an "
                             "alternating 5.0&#8202;mm square. Dashed black line: a 0.5&#8202;N face press; grey band: presses of "
                             "0.2&#8211;3&#8202;N. Script: <code>scripts/fingertip_crosstalk.py budget</code>.") + tab1))

    # 3. subtraction from the joint angles
    sub = (
        "<p>The joint angles give the neighbour&#8217;s pose, so its field can be predicted and subtracted. The SR2 hand&#8217;s "
        "SCS0009 servos read 0.29&#176; per tick, 0.2&#8211;0.35&#8202;mm at the fingertip. With pose errors of 0.1&#8202;mm and "
        f"0.1&#176;, {k1(r5('facing', 0))['residual_uT_median']:.1f}&#8202;&#181;T is left facing and touching, and "
        f"{k1(r5('side', 0))['residual_uT_median']:.0f}&#8202;&#181;T side by side (Table&#160;1). At 0.3&#8202;mm and 0.3&#176; the "
        f"residuals are {k3(r5('facing', 0))['residual_uT_median']:.0f} and {k3(r5('side', 0))['residual_uT_median']:.0f}&#8202;&#181;T. "
        f"Facing, the residual stays below a 0.5&#8202;N press from 5&#8202;mm on ({k3(r5('facing', 5))['residual_uT_median']:.0f}"
        f"&#8202;&#181;T). Side by side it equals that press at 10&#8202;mm ({k3(r5('side', 10))['residual_uT_median']:.0f}&#8202;&#181;T) "
        f"and falls to {k3(r5('side', 20))['residual_uT_median']:.1f}&#8202;&#181;T at 20&#8202;mm.</p>"
        f"<p>At 0.3&#8202;mm and 0.3&#176; the viewer shows a phantom contact in {100 * k3(r5('facing', 0))['shown_frac']:.0f}&#8202;% of "
        f"the draws facing and touching. Side by side within 5&#8202;mm it shows one in "
        f"{100 * min(k3(r5('side', g))['shown_frac'] for g in (0, 2, 5)):.0f}&#8211;"
        f"{100 * max(k3(r5('side', g))['shown_frac'] for g in (0, 2, 5)):.0f}&#8202;%, and the steel test passes "
        f"{100 * min(k3(r5('side', g))['shown_gate_frac'] for g in (0, 2, 5)):.0f}&#8211;"
        f"{100 * max(k3(r5('side', g))['shown_gate_frac'] for g in (0, 2, 5)):.0f}&#8202;% of the draws. Added to the light "
        f"presses, the residual of a touching neighbour moves the estimated force by a median "
        f"{k3(r5('facing', 0))['light_force_added_N_median']:.1f}&#8202;N facing and "
        f"{k3(r5('side', 0))['light_force_added_N_median']:.1f}&#8202;N side by side. The firm presses move by "
        f"{k3(r5('facing', 0))['firm_force_added_N_median']:.2f} and {k3(r5('side', 0))['firm_force_added_N_median']:.1f}&#8202;N. "
        "Subtraction from the joint angles alone therefore works beyond about 5&#8202;mm facing and 10&#8202;mm side by side.</p>")
    S_.append(("subtract", "Subtracting the field predicted from the joint angles", sub))

    # 4. two-board pose fit
    def crow(kind, x):
        r = r5(kind, x)
        c = r["crb"]
        return [("Facing, W " if kind == "facing" else "Side by side, g ") + f"{x:g}&#8202;mm"] + [
            f"{c[k]['one_board']:.3f}" for k in ("known", "0.3", "1", "free")] + [
            f"{c[k]['two_boards']:.3f}" for k in ("0.3", "1", "free")]
    tab2 = P.table(
        ["Pose of a five-magnet neighbour", "Pose known", "One board, 0.3&#8202;mm, 0.3&#176; prior", "One board, 1&#8202;mm, 1&#176; prior",
         "One board, no prior", "Two boards, 0.3&#8202;mm, 0.3&#176; prior", "Two boards, 1&#8202;mm, 1&#176; prior", "Two boards, no prior"],
        [crow("facing", x) for x in FACING_W] + [crow("side", x) for x in SIDE_G], num_cols=tuple(range(1, 8)), n=2,
        caption="Cram&#233;r&#8211;Rao bound on a press along the face normal at v2 ID 1&#8217;s face centre, in N, with the "
                "neighbour&#8217;s pose estimated with the press. One board: v2 ID 1&#8217;s alone. Two boards: the neighbour&#8217;s "
                "board as well, whose reading v2 ID 1&#8217;s magnets change. Prior: the joint angles&#8217; standard deviation per axis.")
    fit = (
        "<p>The pose can also be estimated together with the press, from the boards themselves. The neighbour&#8217;s field "
        "then adds six unknowns, and the joint angles constrain them as a prior. With both boards and a 1&#8202;mm, 1&#176; prior, "
        f"the bound on a press at the face&#8217;s centre is {min(cf):.3f}&#8211;{max(cf):.3f}&#8202;N facing. Side by side it is "
        f"{min(cs):.3f}&#8211;{max(cs):.3f}&#8202;N, against {known:.3f}&#8202;N with the pose known (Table&#160;2). With v2 "
        f"ID 1&#8217;s board alone and the same prior it is {min(r5('facing', x)['crb']['1']['one_board'] for x in FACING_W):.3f}&#8211;"
        f"{max(r5('facing', x)['crb']['1']['one_board'] for x in FACING_W):.3f}&#8202;N facing and "
        f"{min(r5('side', x)['crb']['1']['one_board'] for x in SIDE_G):.3f}&#8211;{max(r5('side', x)['crb']['1']['one_board'] for x in SIDE_G):.3f}"
        "&#8202;N side by side. The first tip&#8217;s magnets also change the second board&#8217;s reading, and that change "
        "constrains the relative pose. Without any prior the bound with two boards is "
        f"{min(r5('facing', x)['crb']['free']['two_boards'] for x in FACING_W):.3f}&#8211;"
        f"{max(r5(k, x)['crb']['free']['two_boards'] for k, xs in (('facing', FACING_W), ('side', SIDE_G)) for x in xs):.3f}&#8202;N.</p>"
        "<p>The bound holds only if the field of the neighbour is modelled well. At contact side by side the field is "
        f"{r5('side', 0)['norm_uT']:,.0f}&#8202;&#181;T, so a model error of 1&#8202;% leaves {r5('side', 0)['norm_uT'] / 100:.0f}"
        "&#8202;&#181;T, close to a 0.2&#8202;N press. The field model of each tip has been fitted at rest only, and its error at a few "
        "millimetres from a second tip has not been measured.</p>")
    S_.append(("fit", "Fitting the neighbour's pose from both boards", fit + tab2))

    # 5. sharpie
    sh = (
        "<p>On 27 October at 12:18 the owner pinched a marker pen between v1 ID 2 and ID 3, with both boards streaming to the "
        "multi-board viewer (Figure&#160;2). The two tips were held by hand. Over the eight windows of the pinch, ID 2&#8217;s "
        f"board changed by {min(w['d'][0] for w in Sn):.0f}&#8211;{max(w['d'][0] for w in Sn):.0f}&#8202;&#181;T from rest, and "
        f"ID 3&#8217;s by {min(w['d'][1] for w in Sn):.0f}&#8211;{max(w['d'][1] for w in Sn):.0f}&#8202;&#181;T. Most of each change is a "
        f"uniform field, a median {g_med[0]:.0f}&#8202;&#181;T at ID 2 and {g_med[1]:.0f}&#8202;&#181;T at ID 3 per chip. The tips "
        "were picked up and turned, and in the Earth&#8217;s field of about 50&#8202;&#181;T a turn changes each chip by up to twice "
        "that.</p>"
        f"<p>In the first window, {w0['t'][0]}&#8211;{w0['t'][1]}, the viewer&#8217;s estimator explains neither board: &#967;&#178; is "
        f"{w0['chi0'][0]:.0f} at ID 2 and {w0['chi0'][1]:,.0f} at ID 3. It shows {w0['f0'][0]:.1f} and {w0['f0'][1]:.1f}&#8202;N. A pose of "
        f"ID 3 fitted to both boards, with a contact on each, lowers &#967;&#178; to {w0['chi1'][0]:.0f} and {w0['chi1'][1]:.0f}. That pose "
        f"places ID 3&#8217;s apex {w0['apex']:.1f}&#8202;mm from ID 2&#8217;s, with the normals {w0['angle']:.0f}&#176; from facing. Its "
        f"field is {w0['x'][0]:.0f}&#8202;&#181;T at ID 2&#8217;s board and {w0['x'][1]:.0f}&#8202;&#181;T at ID 3&#8217;s. The forces fall "
        f"to {w0['f1'][0]:.1f} and {w0['f1'][1]:.1f}&#8202;N, and the two contact points lie {w0['apart']:.1f}&#8202;mm apart.</p>"
        "<p>In the later windows the fitted pose changes from window to window: the apexes lie "
        f"{min(w['apex'] for w in later):.1f}&#8211;{max(w['apex'] for w in later):.0f}&#8202;mm apart, with the normals "
        f"{min(w['angle'] for w in later):.0f}&#8211;{max(w['angle'] for w in later):.0f}&#176; from facing. There &#967;&#178; is "
        f"{min(min(w['chi0']) for w in later):.0f}&#8211;{max(max(w['chi0']) for w in later):.0f} without the neighbour&#8217;s field and "
        f"{min(min(w['chi1']) for w in later):.0f}&#8211;{max(max(w['chi1']) for w in later):.0f} with it. The fitted field is "
        f"{min(min(w['x']) for w in later):.0f}&#8211;{max(max(w['x']) for w in later):.0f}&#8202;&#181;T, and the forces change by up to "
        f"{max(max(abs(a - b) for a, b in zip(w['f0'], w['f1'])) for w in later):.1f}&#8202;N. The pose is not determined in "
        "this steady part of the hold, so its forces are not corrected.</p>")
    S_.append(("sharpie", "The owner's pinch of a marker pen between two v1 tips",
               sh + P.fig(2, f2, "The owner&#8217;s pinch of a marker pen between v1 ID 2 (blue) and ID 3 (orange), 2026-10-27, "
                          "from the multi-board viewer&#8217;s streams. (a) Each board&#8217;s change from rest, the norm over its 15 "
                          "components in bins of 0.2&#8202;s. Diamonds: the estimator&#8217;s uniform field in each window, as its "
                          "norm over the five chips. (b) The force of each board&#8217;s contact per window: the viewer&#8217;s "
                          "estimate (open) and with the two-board pose fit (filled). Grey: the eight windows. Script: "
                          "<code>scripts/fingertip_crosstalk.py sharpie</code>; rows in <code>data/sharpie.json</code>.")))

    # 6. limitations
    S_.append(("limits", "Limitations", (
        "<p>Figure&#160;1, Tables&#160;1 and 2 come from the dipole model of v2 ID 1&#8217;s magnets, fitted at rest on the UR5. No "
        "field of a second tip has been measured at a known pose. The neighbour has v2 ID 1&#8217;s magnet strength. The bound "
        "assumes an exact field model and the noise of a 50&#8202;ms window, and it is the bound for one press at the face&#8217;s "
        "centre. In the pinch the two v1 tips were held by hand, so neither the pose nor the force is known. The pinch&#8217;s "
        "boards use the fits of ID 2 on the UR5 and of ID 3 on the bench.</p>")))

    # 7. future work
    fw = [
        ("The neighbour&#8217;s field on the holder.", "The " + link(L["v3"], "plan for the four-magnet v3 tips")
         + " maps it without contact. The robot tip v3 ID 1 dwells at 45 facing poses (W 0.5&#8211;30&#8202;mm, twists "
         "&#8722;20 to +20&#176;) and 31 side-by-side poses (g 13&#8211;25&#8202;mm). It does so once with both stations of the "
         "T-shaped holder (" + link(L["holder"], "holder README") + ") empty, and once with v3 ID 2 and ID 3 on them. Each held "
         "tip&#8217;s pose and strength are fitted in five folds by pose. A held-out residual above 10&#8202;% of the field at "
         "W 5&#8202;mm and under (3.4&#8202;&#181;T at W 0.5&#8202;mm) means the model cannot be subtracted to the board&#8217;s "
         "noise. At these poses the held tips&#8217; magnets pull on the robot tip with at most 0.69&#8202;mN "
         "(<code>scripts/fingertip_crosstalk.py pull</code>)."),
        ("A pinch between two v3 tips.", "The robot apex presses the held apex on the +y station at 0.25&#8211;3&#8202;N, also "
         "shifted by &#177;2 and &#177;4&#8202;mm and turned by &#177;10 and &#177;20&#176; (16 actions, 80 holds). The robot "
         "tip&#8217;s contact is estimated from the raw change, with the modelled field at the commanded pose subtracted, and with "
         "the held tip&#8217;s pose fitted from the board. With the fitted pose, a force error above 0.2&#8202;N at 0.5&#8202;N "
         "contradicts the one-board bound of Table&#160;2."),
        ("A five-magnet neighbour.", "Decision 5 of the plan, about 30&#8202;min with a five-magnet tip on the +y station. Its "
         "483&#8202;&#181;T at W 0.5&#8202;mm tests the model at 1&#8202;% (4.8&#8202;&#181;T); at v3&#8217;s 34&#8202;&#181;T, "
         "1&#8202;% is about the board&#8217;s noise. A misfit above 1&#8202;% at the closest poses means the bounds of "
         "Table&#160;2 do not apply."),
    ]
    S_.append(("future", "Future work", "<ul class=\"open\">" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in fw) + "</ul>"))
    return S_, lede, summary


def build():
    B = json.loads((DATA / "budget.json").read_text())
    S = json.loads((DATA / "sharpie.json").read_text())
    T = sharpie_trace()
    Sn = sharpie_numbers(S)
    f1 = fig_budget(B)
    f2 = fig_sharpie(S, T, Sn)
    S_, lede, summary = sections(B, S, Sn, f1, f2)
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S_, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S_, 1))
    byline = ("Field model of v2 ID 1 (five N52) fitted at rest in UR5 session 20261026-192158-v2id1; the owner&#8217;s pinch with "
              "v1 ID 2 and ID 3 on 2026-10-27 at 12:18. Analysis <code>scripts/fingertip_crosstalk.py</code> (commit bc508a6e); "
              "page by <code>scripts/fingertip_crosstalk_page.py</code>; rows in " + link(DATA, "data/") + ". The fingertip and its "
              "calibration are described on " + link(L["overview"], "the fingertip overview") + " and "
              + link(L["cal"], "the calibration page") + ".")
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "Magnetic crosstalk between fingertips",
                 "H1": "Magnetic crosstalk between neighbouring fingertips and its removal by the fingers' pose",
                 "EYEBROW": "Fingertip sensing &#183; 2026-10-27", "LEDE": lede, "BYLINE": byline, "FILM": "",
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = P.TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary, date="2026-10-27")
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    build()
