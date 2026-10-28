#!/usr/bin/env python3
"""Page on magnetic crosstalk between neighbouring printed fingertips (hand-off 20261027-docs_update_impratio, step 4;
owner 2026-10-27: "crosstalk is going to be problem, so itd be helpful to present some compelling results on that").
Reads the rows of scripts/fingertip_crosstalk.py (data/budget.json: v2 ID 1's field model under a five- or four-magnet
neighbour; data/sharpie.json: the owner's sharpie pinch with v1 ID 2 and ID 3) and, once, the multi-board viewer's
streams of that pinch for the trace of its figure (kept in media/20261027-sharpie_trace.json, since logs/ is not tracked).
The measured section reads the UR5 coordinator's analysis of the holder sessions (scripts/fingertip_crosstalk_holder.py,
data/20261027-crosstalk_touchoffs.json in the v3 plan folder, as committed at HEAD) and draws its figure from that file's
bins; its numbers follow the coordinator's lines in that folder's feed.jsonl. The last figure is the UR5 cell render of the holder map
(scripts/ur5_cell_render.py, in the v3 plan folder's media/). Writes docs/experiments/20261027-fingertip_crosstalk/20261027-magnetic_crosstalk_between_neighbouring_fingertips.html.

    ~/.claude/bin/resguard.sh run --mem 2G --cpu 100 -- .venv/bin/python scripts/fingertip_crosstalk_page.py
"""
from __future__ import annotations

import io
import json
import math
import os
import subprocess
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
V3 = EXP / "20261027-v3_tip_calibration_and_crosstalk"
HOVER = V3 / "media/20261027-ur5_cell_holder_hover.png"
HOLD = V3 / "data/20261027-crosstalk_touchoffs.json"
SEC = dict(terms=1, field=2, holder=3, subtract=4, fit=5, sharpie=6, limits=7, future=8)
FIG = dict(budget=1, holder=2, hovers=3, sharpie=4, render=5)
TAB = dict(field=1, holder=2, hovers=3, crb=4)
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


def at_head(path):
    """A file of the coordinator's v3 folder as committed at HEAD, LFS pointers resolved. Its working copy can be mid-analysis
    (on 2026-10-27 at 22:13 the holder file was rewritten before its commit and feed line); the page follows the commits."""
    rel = str(Path(path).relative_to(ROOT))
    blob = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"], capture_output=True, check=True).stdout
    if blob.startswith(b"version https://git-lfs.github.com/spec/v1"):
        blob = subprocess.run(["git", "-C", str(ROOT), "lfs", "smudge", "--", rel], input=blob, capture_output=True,
                              check=True).stdout
    return blob


def count(n):
    """Counts under ten in words."""
    return "one two three four five six seven eight nine".split()[n - 1] if 0 < n < 10 else str(n)


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


def fig_holder(H):
    """The touch-off approaches between v3 tips on the holder: each board's change against W, from the bins of the
    coordinator's net-moment fit (measured, the nominal model times its gain, with the source tip's net moment, left)."""
    plt = P.plt_setup()
    from matplotlib.ticker import FuncFormatter
    nm = H["net_moment"]
    panels = (("(a) Held board, +y station", nm["robot_tip"]["bins"]["held0"]),
              ("(b) Held board, \N{MINUS SIGN}y station", nm["robot_tip"]["bins"]["held3"]),
              ("(c) Robot tip\N{RIGHT SINGLE QUOTATION MARK}s board over the +y tip", nm["held_tip_+y"]["bins"]),
              ("(d) Robot tip\N{RIGHT SINGLE QUOTATION MARK}s board over the \N{MINUS SIGN}y tip", nm["held_tip_-y"]["bins"]))
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 8.4), sharex=True, sharey=True)
    for ax, (title, rows) in zip(axs.flat, panels):
        w = [r["W_mm"] for r in rows]
        ax.semilogy(w, [r["meas_uT"] for r in rows], "-o", color="black", ms=4.5, lw=1.6, label="Measured")
        ax.semilogy(w, [r["model_uT"] for r in rows], "--s", color=BLUE, ms=4.5, lw=1.4, label="Nominal dipole model")
        ax.semilogy(w, [r["model_moment_uT"] for r in rows], "-", color=ORANGE, lw=2.0,
                    label="Model with the source tip\N{RIGHT SINGLE QUOTATION MARK}s net moment")
        ax.semilogy(w, [r["left_uT"] for r in rows], ":^", color=GREY, ms=4.5, lw=1.6,
                    label="Left after the model with the net moment")
        ax.set_title(title, loc="left")
        ax.set_xlim(-1.5, 20.5)
        ax.set_xticks((0, 5, 10, 15, 20))
        ax.set_ylim(0.2, 100)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.grid(True, which="major", color="0.88", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs[1]:
        ax.set_xlabel("Gap W between the apexes (mm)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Change at the board (\N{MICRO SIGN}T)")
    h, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=13)
    fig.tight_layout()
    out = MEDIA / "20261027-holder_touchoffs.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ text

HOLD_MADE = "2026-10-27 22:50"   # the holder analysis that the measured section's written sentences quote (feed 21:44 to 22:52)


def holder_numbers(H):
    """The measured section's numbers, read from the coordinator's holder file: each board's change at contact (the bin
    nearest W 0), at W 4.75 and 12.5 mm, the nominal model there, its gain and explained share (the analysis of 21:44),
    the net moments (21:47) and the map's hovers against the model with those moments (22:25)."""
    near = lambda rows, w: min(rows, key=lambda r: abs(r["W_mm"] - w))  # noqa: E731
    b, nm = H["boards"], H["net_moment"]
    at = {k: {w: near(b[k]["bins"], w) for w in (0.0, 4.75, 12.5)} for k in ("held0", "held3", "robot")}
    rt, hp, hm = nm["robot_tip"], nm["held_tip_+y"], nm["held_tip_-y"]
    m_rt = 1e3 * np.asarray(rt["moment_Am2"])
    cross = [1e3 * np.asarray(c["moment_Am2"]) for c in rt["cross"].values()]
    hv = H["hovers"]
    hv_ses = list(dict.fromkeys(h["session"] for h in hv))          # the map first, then the close hovers over -y
    mp = [h for h in hv if h["session"] == hv_ses[0]]
    cl = [h for h in hv if len(hv_ses) > 1 and h["session"] == hv_ses[1]]
    first = {h["id"]: h for h in mp if not h["id"].endswith("-rep")}
    reps = [h for h in mp if h["id"].endswith("-rep")]
    rep_diff = {k: [float(np.linalg.norm(np.subtract(r["boards"][k]["meas"], first[r["id"][:-4]]["boards"][k]["meas"])))
                    for r in reps] for k in ("held0", "held3", "robot")}
    close = [h for h in mp if h["station"] == "+y" and h["W_mm"] <= 1.05]
    far = [h for h in mp if h["W_mm"] > 40]
    under = lambda h: next(k for k, v in h["boards"].items() if v.get("station") == h["station"])  # noqa: E731
    st_w = lambda st: [h["W_mm"] for h in mp if h["station"] == st]  # noqa: E731
    near = lambda w: [h for h in cl if abs(h["W_mm"] - w) < 0.3]  # noqa: E731
    cl_at = {w: {k: dict(meas=[h["boards"][k]["meas_uT"] for h in near(w)], left=[h["boards"][k]["left_uT"] for h in near(w)])
                 for k in ("held3", "robot")} for w in (0.5, 5.0)}
    touch_ses = [x for x in (s["name"] for s in H["sessions"]) if any(t["session"] == x for t in H["touches"])]
    return dict(
        at=at, gain={k: b[k]["gain"] for k in at}, expl={k: 100 * b[k]["explained_near"] for k in at},
        n_touch=len(H["touches"]), sessions=touch_ses,
        m_rt=m_rt, m1=float(np.linalg.norm(m_rt)) / rt["share_of_one_magnet"],
        share=dict(robot=100 * rt["share_of_one_magnet"], plus=100 * hp["share_of_one_magnet"], minus=100 * hm["share_of_one_magnet"]),
        with_m=dict(held0=100 * rt["explained"]["held0"]["model_and_moment"], held3=100 * rt["explained"]["held3"]["model_and_moment"],
                    plus=100 * hp["explained"]["model_and_moment"], minus=100 * hm["explained"]["model_and_moment"]),
        cross_diff=float(np.max(np.abs(cross[0] - cross[1]))),
        cross_pred=sorted(100 * c["explained_with_moment"] for c in rt["cross"].values()),
        n_hover=len(hv), n_map=len(mp), n_close=len(cl), map_session=hv_ses[0], close_session=hv_ses[1] if cl else None,
        n_rep=sum(1 for h in hv if h["id"].endswith("-rep")), n_close_rep=sum(1 for h in cl if h["id"].endswith("-rep")),
        n_st={st: len(st_w(st)) for st in ("+y", "-y")}, w_st={st: (min(st_w(st)), max(st_w(st))) for st in ("+y", "-y")},
        n_st_all={st: sum(1 for h in hv if h["station"] == st) for st in ("+y", "-y")},
        cl_w=(min(h["W_mm"] for h in cl), max(h["W_mm"] for h in cl)) if cl else None, cl_peak=max((h["peak_N"] for h in cl), default=0),
        cl_at=cl_at, twists=sorted({round(h["heading_deg"]) for h in hv}),
        hx={k: {m: 100 * v[m] for m in v} for k, v in H["hover_explained"].items()},
        close_meas=[h["boards"]["held0"]["meas_uT"] for h in close], close_left=[h["boards"]["held0"]["left_uT"] for h in close],
        close_w=(min(h["W_mm"] for h in close), max(h["W_mm"] for h in close)),
        rep_med=[float(np.median(v)) for v in rep_diff.values()], rep_max=max(max(v) for v in rep_diff.values()),
        drift_med=float(np.median([v["ref_drift_uT"] for h in mp for v in h["boards"].values()])),
        far_meas=[h["boards"][under(h)]["meas_uT"] for h in far], far_model=[h["boards"][under(h)]["model_moment_uT"] for h in far])


def holder_section(H, hn, fh, fv):
    """Section 'Neighbour field measured on the fingertip holder', with its tables and figures."""
    at, g, e, sh, wm, hx = hn["at"], hn["gain"], hn["expl"], hn["share"], hn["with_m"], hn["hx"]
    c = lambda k, w: at[k][w]["meas_uT"]  # noqa: E731
    ses = hn["sessions"]
    low = [100 * (1 - at[k][0.0]["model_uT"] / c(k, 0.0)) for k in ("held0", "held3")]
    left12 = [at[k][12.5]["left_uT"] for k in at]
    def rng(xs, f="{:.0f}"):
        a, b = f.format(min(xs)), f.format(max(xs))
        return a if a == b else f"{a} to {b}"
    mx, my, mz = hn["m_rt"]
    tw = hn["twists"]
    sg = lambda x: f"{x:.0f}".replace("-", "&#8722;")  # noqa: E731
    ca = hn["cl_at"]

    def robot_shares(vp, vm):
        """The moments' effect on the robot tip's board over +y and over -y, in one sentence when both go the same way."""
        up = lambda v: v["model_and_moment"] >= v["model"]  # noqa: E731
        fr = lambda v: f"from {sg(v['model'])} to {sg(v['model_and_moment'])}&#8202;%"  # noqa: E731
        verb = lambda v: "raise" if up(v) else "lower"  # noqa: E731
        if up(vp) == up(vm):
            return f"On the robot tip&#8217;s board the moments {verb(vp)} the share {fr(vp)} over +y and {fr(vm)} over &#8722;y."
        return (f"On the robot tip&#8217;s board the moments {verb(vp)} the share {fr(vp)} over +y. Over &#8722;y they "
                f"{verb(vm)} it {fr(vm)}.")
    body = (
        f"<p>On 2026-10-27 the UR5 moved v3 ID 1 through {hn['n_touch']} touch-off approaches onto the two held v3 tips, in "
        f"sessions {', '.join(ses[:-1])} and {ses[-1]}. In session {hn['map_session']} it also made the map&#8217;s "
        f"{hn['n_map']} hovers, and {hn['n_close']} close hovers over the &#8722;y tip in session {hn['close_session']}. All three boards streamed throughout. Both pads were placed by the holder&#8217;s frame "
        "through both apexes, measured by touch-offs. That frame tilts by 0.25&#176;, with a residual of 0.087&#8202;mm rms.</p>"
        f"<p>Under the robot tip a held board changed by {rng([c('held0', 0.0), c('held3', 0.0)])}&#8202;&#181;T at contact "
        f"(Figure&#160;{FIG['holder']}, Table&#160;{TAB['holder']}). At W 4.75&#8202;mm it changed by "
        f"{rng([c('held0', 4.75), c('held3', 4.75)])}&#8202;&#181;T, and at W 12.5&#8202;mm by "
        f"{rng([c('held0', 12.5), c('held3', 12.5)])}&#8202;&#181;T. The robot tip&#8217;s board changed by {c('robot', 0.0):.0f}, "
        f"{c('robot', 4.75):.0f} and {c('robot', 12.5):.0f}&#8202;&#181;T at these gaps.</p>"
        f"<p>The nominal model&#8217;s gains are {g['robot']:.2f} on the robot tip&#8217;s board, and {g['held0']:.2f} and "
        f"{g['held3']:.2f} on the +y and &#8722;y held boards. On the robot tip&#8217;s board {e['robot']:.0f}&#8202;% of the "
        "change&#8217;s energy below W 10&#8202;mm is explained by the model. On the held boards it is "
        f"{low[0]:.0f} and {low[1]:.0f}&#8202;% low at contact, and its explained share is {e['held0']:.0f} and "
        f"{e['held3']:.0f}&#8202;%. Beyond W 5&#8202;mm every board&#8217;s change falls more slowly than the model&#8217;s. At "
        f"W 12.5&#8202;mm, {rng(left12)}&#8202;&#181;T is left after the scaled model.</p>"
        "<p>The four magnets of a v3 tip do not cancel fully. Besides the nominal model, each tip was fitted with a net moment, "
        f"one point dipole at its magnets&#8217; centroid. The net moments are {rng(list(sh.values()))}&#8202;% of one magnet&#8217;s "
        f"moment, {hn['m1']:.1f}&#8202;mA&#8202;m&#178;. The robot tip&#8217;s moment, fitted on both held boards, is "
        f"({mx:.2f}, {my:.2f}, {mz:.2f})&#8202;mA&#8202;m&#178; in its pad frame. With it the explained share on the held "
        f"boards rises to {wm['held0']:.0f} and {wm['held3']:.0f}&#8202;%. Fitted on either held board alone, the two moments "
        f"agree within {hn['cross_diff']:.2f}&#8202;mA&#8202;m&#178; in each component. The moment fitted on one held board gives "
        f"an explained share of {hn['cross_pred'][0]:.0f} and {hn['cross_pred'][1]:.0f}&#8202;% on the other. The held tips&#8217; "
        f"moments, fitted on the robot tip&#8217;s board, are {sh['plus']:.1f} and {sh['minus']:.1f}&#8202;% of a magnet. With "
        f"them its explained share rises from {e['robot']:.0f}&#8202;% to {wm['plus']:.0f} and {wm['minus']:.0f}&#8202;%.</p>"
        "<p>After the net moment, about 10&#8202;&#181;T is left on the held boards at contact and 1 to 2&#8202;&#181;T at W "
        "10&#8202;mm. On the robot tip&#8217;s board 3 to 5&#8202;&#181;T is left beyond W 8&#8202;mm, which is explained by "
        "neither held tip&#8217;s moment. Its own motion through the room&#8217;s field is the likely source. The moments lie "
        "mostly in the face&#8217;s plane, along y and z of the pad frame. A tilt of the magnets in their pockets gives a moment "
        "in that plane; a spread of their strengths gives one along the normal, x.</p>"
        "<p>The moments were then compared with hovers on which they were not fitted "
        f"(Figure&#160;{FIG['hovers']}, Table&#160;{TAB['hovers']}). In the map the robot made {hn['n_st']['+y']} hovers over the "
        f"+y tip at W {hn['w_st']['+y'][0]:.1f} to {hn['w_st']['+y'][1]:.0f}&#8202;mm, at twists of {tw[0]:+d} to {tw[-1]:+d}&#176; "
        f"and with 3&#8202;mm shifts. Over the &#8722;y tip it made {hn['n_st']['-y']}, at W {hn['w_st']['-y'][0]:.0f}&#8202;mm and "
        f"over. The {hn['n_close']} close hovers over the &#8722;y tip span W {hn['cl_w'][0]:.1f} to {hn['cl_w'][1]:.0f}&#8202;mm, "
        f"and all stayed below the hover&#8217;s 0.5&#8202;N contact guard. Of the {hn['n_hover']} hovers, {hn['n_rep']} repeat "
        f"earlier poses. On the held board under the robot tip, {hx['held0 at +y']['model_and_moment']:.1f}&#8202;% of the change "
        f"is explained over +y and {hx['held3 at -y']['model_and_moment']:.1f}&#8202;% over &#8722;y. The nominal model gives "
        f"{sg(hx['held0 at +y']['model'])} and {sg(hx['held3 at -y']['model'])}&#8202;%. "
        f"{robot_shares(hx['robot at +y'], hx['robot at -y'])}</p>"
        f"<p>At W {hn['close_w'][0]:.1f} to {hn['close_w'][1]:.0f}&#8202;mm the +y board read {rng(hn['close_meas'])}&#8202;&#181;T, "
        f"and {rng(hn['close_left'])}&#8202;&#181;T is left after the model with the moment. At W 0.5&#8202;mm the &#8722;y board "
        f"read {rng(ca[0.5]['held3']['meas'])}&#8202;&#181;T and the robot tip&#8217;s board {rng(ca[0.5]['robot']['meas'])}&#8202;&#181;T. "
        f"After the model with the moments, {rng(ca[0.5]['held3']['left'])} and {rng(ca[0.5]['robot']['left'])}&#8202;&#181;T are left. At W 5&#8202;mm they read "
        f"{rng(ca[5.0]['held3']['meas'])} and {rng(ca[5.0]['robot']['meas'])}&#8202;&#181;T, and {rng(ca[5.0]['held3']['left'])} and "
        f"{rng(ca[5.0]['robot']['left'])}&#8202;&#181;T are left. In the map, a repeat about 40&#8202;min after its first run "
        f"differs from it by a median {min(hn['rep_med']):.2f} to {max(hn['rep_med']):.1f}&#8202;&#181;T, and by at most "
        f"{hn['rep_max']:.1f}&#8202;&#181;T. The two reference dwells of a hover differ by a median {hn['drift_med']:.1f}&#8202;&#181;T. "
        f"At W 45&#8202;mm the board under the robot tip changed by {rng(hn['far_meas'], '{:.1f}')}&#8202;&#181;T, and the model "
        f"with the moments gives {rng(hn['far_model'], '{:.1f}')}&#8202;&#181;T.</p>"
        ).replace("(-", "(&#8722;").replace(", -", ", &#8722;").replace("of -", "of &#8722;")
    rows = []
    for k, name, src in (("held0", "Held, +y station", "Robot tip"), ("held3", "Held, &#8722;y station", "Robot tip"),
                         ("robot", "Robot tip&#8217;s", "+y and &#8722;y held tips")):
        ms = f"{sh['robot']:.0f}" if k != "robot" else f"{sh['plus']:.1f} and {sh['minus']:.1f}"
        we = f"{wm[k]:.0f}" if k != "robot" else f"{wm['plus']:.0f} and {wm['minus']:.0f}"
        rows.append([name, src, f"{c(k, 0.0):.1f}", f"{c(k, 4.75):.1f}", f"{c(k, 12.5):.1f}", f"{at[k][0.0]['model_uT']:.1f}",
                     f"{g[k]:.2f}", f"{e[k]:.0f}", ms, we])
    w0 = [at[k][0.0]["W_mm"] for k in at]
    tab = P.table(
        ["Receiving board", "Source tip", "Change at contact (&#181;T)", "Change at W 4.75&#8202;mm (&#181;T)",
         "Change at W 12.5&#8202;mm (&#181;T)", "Nominal model at contact (&#181;T)", "Gain", "Explained, nominal model (%)",
         "Source&#8217;s net moment (% of a magnet)", "Explained with it (%)"],
        rows, num_cols=tuple(range(2, 10)), n=TAB["holder"],
        caption="Change of each board through the touch-off approaches between v3 tips on the holder, and the share of its "
                "energy below W 10&#8202;mm that is explained by the nominal dipole model and with the source tip&#8217;s net "
                f"moment. Contact is the bin nearest W 0 (W {min(w0):.2f} to {max(w0):.2f}&#8202;mm). Rows in "
                f"<code>{HOLD.relative_to(ROOT)}</code>; analysis <code>scripts/fingertip_crosstalk_holder.py</code>.".replace("W -", "W &#8722;").replace("to -", "to &#8722;"))
    fig = P.fig(FIG["holder"], fh,
                "Change of each board through the touch-off approaches between v3 tips on the holder, against the gap W, in "
                "bins of 0.5 to 1&#8202;mm. In (a) and (b) the held boards under the robot tip are shown, in (c) and (d) the robot "
                "tip&#8217;s board over each held tip. Black: measured. Dashed blue: the nominal dipole model, scaled by one gain per board. "
                "Orange: the model with the source tip&#8217;s net moment. Grey dotted: what is left after it. Bins of the "
                f"net-moment fit in the file of Table&#160;{TAB['holder']}.")
    sign = lambda x: f"{x:.1f}".replace("-", "&#8722;")  # noqa: E731
    hrows = []
    for st, n in (("+y", hn["n_st"]["+y"]), ("-y", hn["n_st"]["-y"])):
        other = "held3" if st == "+y" else "held0"
        mine = "held0" if st == "+y" else "held3"
        stn = st.replace("-", "&#8722;")
        for k, name in ((mine, f"Held, {stn} station, under the robot tip"),
                        (other, f"Held, {'&#8722;y' if st == '+y' else '+y'} station, not under the robot tip"),
                        ("robot", "Robot tip&#8217;s")):
            v = hx[f"{k} at {st}"]
            hrows.append([name, stn, str(hn["n_st_all"][st]), sign(v["model"]), sign(v["model_and_moment"])])
    tab_h = P.table(
        ["Board", "Robot tip over", "Hovers", "Explained, nominal model (%)", "Explained, with the net moments (%)"],
        hrows, num_cols=(2, 3, 4), n=TAB["hovers"],
        caption="Share of the hovers&#8217; measured change that is explained by the nominal dipole model and with the net moments "
                "fitted on the touch-off approaches, by board and by the station the robot tip hovers over. Over &#8722;y the map&#8217;s "
                "hovers and the close hovers are pooled. A share below zero "
                "means the scaled model fits worse than a prediction of no change. Rows in the file of "
                f"Table&#160;{TAB['holder']} (<code>hover_explained</code>).")
    fig_h = P.fig(FIG["hovers"], fv,
                  "Change of each board in the map&#8217;s hovers, against the gap W. In (a) and (b) the robot tip hovers over the "
                  "+y tip, in (c) and (d) over the &#8722;y tip. Over &#8722;y the map&#8217;s hovers and the close hovers are drawn together. "
                  "Black circles: measured, at twists of &#8722;20 to +10&#176; "
                  "together. Open squares: the robot apex shifted 3&#8202;mm along or across the tip. Orange dashes: the model "
                  "with the net moments fitted on the touch-off approaches. Blue dashes: the nominal dipole model with the same "
                  f"gains. Rows in the file of Table&#160;{TAB['holder']}.")
    return body + fig + tab + fig_h + tab_h


def fig_hovers(H):
    """The map's hovers: each board's change against W, with the model with the net moments (fitted on the touch-off
    approaches) and the nominal model, scaled by the same gains."""
    plt = P.plt_setup()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullFormatter
    nm = H["net_moment"]
    gain = lambda b, st: (nm["robot_tip"]["gains"][b] if b.startswith("held") else nm[f"held_tip_{st}"]["gain"])  # noqa: E731
    panels = (("(a) Held board at +y, robot tip over +y", "+y", "held0"),
              ("(b) Robot tip\N{RIGHT SINGLE QUOTATION MARK}s board over +y", "+y", "robot"),
              ("(c) Held board at \N{MINUS SIGN}y, robot tip over \N{MINUS SIGN}y", "-y", "held3"),
              ("(d) Robot tip\N{RIGHT SINGLE QUOTATION MARK}s board over \N{MINUS SIGN}y", "-y", "robot"))
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 8.4))
    for ax, (title, st, b) in zip(axs.flat, panels):
        hv = [h for h in H["hovers"] if h["station"] == st]
        for h in hv:
            v = h["boards"][b]
            shift = any(t in h["id"] for t in ("-x+", "-x-", "-y+", "-y-"))
            ax.plot(h["W_mm"], v["meas_uT"], "s" if shift else "o", ms=6.5, color="black", mfc="white" if shift else "black",
                    mew=1.3, zorder=3)
            ax.plot(h["W_mm"], v["model_moment_uT"], "_", ms=17, mew=2.2, color=ORANGE, zorder=2)
            ax.plot(h["W_mm"], gain(b, st) * v["model_uT"], "_", ms=17, mew=1.8, color=BLUE, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.4, 60)
        ax.set_xticks((0.5, 1, 2, 5, 10, 20, 45))
        for a in (ax.xaxis, ax.yaxis):
            a.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            a.set_minor_formatter(NullFormatter())
        ax.set_title(title, loc="left")
        ax.grid(True, which="major", color="0.88", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs[1]:
        ax.set_xlabel("Gap W between the apexes (mm)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Change at the board (\N{MICRO SIGN}T)")
    h = [Line2D([], [], color="black", marker="o", ls="", ms=6.5), Line2D([], [], color="black", marker="s", mfc="white", ls="", ms=6.5),
         Line2D([], [], color=ORANGE, marker="_", ls="", ms=17, mew=2.2), Line2D([], [], color=BLUE, marker="_", ls="", ms=17, mew=1.8)]
    l_ = ["Measured", "Measured, apex shifted 3 mm", "Model with the net moments", "Nominal dipole model"]
    fig.legend(h, l_, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=13)
    fig.tight_layout()
    out = MEDIA / "20261027-holder_hovers.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def sections(B, S, Sn, H, f1, f2, fh, fv):
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

    hn = holder_numbers(H)
    con = [hn["at"][k][0.0]["meas_uT"] for k in hn["at"]]
    far = [hn["at"][k][12.5]["meas_uT"] for k in hn["at"]]
    ex, exm, shr = list(hn["expl"].values()), list(hn["with_m"].values()), list(hn["share"].values())
    bound = cf + cs

    hx = hn["hx"]
    hov_held = [hx["held0 at +y"]["model_and_moment"], hx["held3 at -y"]["model_and_moment"]]
    hov_nom = [hx["held0 at +y"]["model"], hx["held3 at -y"]["model"]]
    neg = lambda x: f"{x:.0f}".replace("-", "&#8722;")  # noqa: E731

    def both(xs, f=lambda x: f"{x:.0f}", pct="&#8202;%"):
        a, b = f(xs[0]), f(xs[1])
        return f"{a}{pct}" if a == b else f"{a} and {b}{pct}"
    board = "either held board" if f"{hov_held[0]:.0f}" == f"{hov_held[1]:.0f}" else "the two held boards"
    lede = (
        "A neighbouring fingertip&#8217;s magnets change the reading of a magnetic fingertip&#8217;s board by more than a press "
        "does. On the UR5, two four-magnet v3 tips with their pads touching changed each other&#8217;s boards by "
        f"{min(con):.0f} to {max(con):.0f}&#8202;&#181;T. A 0.5&#8202;N press changed v2 ID 1&#8217;s board by {p05:.0f}&#8202;&#181;T. "
        f"The four magnets of a tip do not cancel fully: each tip has a net moment of {min(shr):.0f} to {max(shr):.0f}&#8202;% of "
        "one magnet&#8217;s. With these moments, fitted on touch-offs, the dipole model explains "
        f"{both(hov_held)} of the change at {board} in {hn['n_hover']} hovers. Without them it explains "
        f"{both(hov_nom, neg)}. In v2 ID 1&#8217;s field model a "
        f"five-magnet neighbour adds {r5('facing', 0)['norm_uT']:.0f}&#8202;&#181;T facing and "
        f"{r5('side', 0)['norm_uT']:,.0f}&#8202;&#181;T side by side, touching. When the neighbour&#8217;s pose is fitted from both "
        f"boards, the Cram&#233;r&#8211;Rao bound on a press is {min(bound):.3f}&#8211;{max(bound):.3f}&#8202;N, against "
        f"{known:.3f}&#8202;N with the pose known.")
    summary = (
        "Magnetic crosstalk between neighbouring fingertips: on the UR5 two four-magnet v3 tips touching changed each other's "
        f"boards by {min(con):.0f}-{max(con):.0f} uT, against {p05:.0f} uT for a 0.5 N press on v2 ID 1; each tip has a net "
        f"moment of {min(shr):.0f}-{max(shr):.0f} % of one magnet, and with these moments, fitted on touch-offs, the dipole model "
        f"explains {both(hov_held, pct=' %')} of the change at {board} in {hn['n_hover']} hovers "
        f"({both(hov_nom, pct=' %')} without). In v2 ID 1's field model a five-N52 neighbour adds "
        f"{r5('facing', 0)['norm_uT']:.0f} uT facing and {r5('side', 0)['norm_uT']:,.0f} uT side by side when touching; fitting "
        f"its pose from both boards keeps the press bound at {min(bound):.3f}-{max(bound):.3f} N ({known:.3f} N with the pose "
        "known).")

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
        ("Receiving tip, neighbour", f"In the field model of Sections {SEC['field']}, {SEC['subtract']} and {SEC['fit']} the "
         "receiving tip is v2 ID 1, with five N52 magnets. Its field model was fitted at rest in UR5 session "
         "20261026-192158-v2id1. "
         "The neighbour is the same five-magnet tip, or four N52 magnets in an alternating 5.0&#8202;mm square (pad 05R4), both "
         "at v2 ID 1&#8217;s magnet strength."),
        ("Facing, W", "The pads face each other across a gap W between their apexes, as in a pinch, in mm. The apex is the "
         "centre of a pad&#8217;s face. On the holder W is the robot apex&#8217;s height above the held apex; it is negative "
         "when the pads press into each other."),
        ("Side by side, g", "The tips lie side by side with their apexes level and a gap g between their flanks, in mm."),
        ("v3 tips, robot tip, held tips", "Three fingertips with pad 05R4: four N52 magnets of 3 &#215; 2&#8202;mm in an "
         "alternating 5.0&#8202;mm square, 7.5&#8202;mm under the apex. The robot tip, v3 ID 1, is on the UR5. The two held tips "
         "face up on the +y and &#8722;y stations of the T-shaped fingertip holder (" + link(L["holder"], "holder README")
         + "), 57&#8202;mm apart, each with its own board."),
        ("Touch-off, hover", "In a touch-off the robot lowers its apex onto a held apex until the force sensor detects contact. "
         "Its approach runs from the top of the travel to that onset. In a hover the robot apex dwells at a gap W over a held "
         "apex without contact, between two dwells at a reference 40&#8202;mm above."),
        ("Measured change", "The change of a board&#8217;s 15 components caused by the other tip, as a norm in &#181;T. In a "
         "touch-off it is taken from the board&#8217;s reading with the robot far; in a hover, from the mean of the two "
         "reference dwells."),
        ("Nominal dipole model, gain", "The field model of Section " + str(SEC["field"]) + " for four N52 magnets at their "
         "nominal places with alternating poles (<code>scripts/fingertip_crosstalk.py budget</code>). It has v2 ID 1&#8217;s "
         "chip geometry and magnet strength and was fixed before the measurement. The gain is one factor per "
         "receiving board by which the model is scaled to the measured change; 1 means the model&#8217;s strength is right."),
        ("Explained share", "The share of the measured change&#8217;s energy that the scaled model accounts for, "
         "$1 - \\sum_k \\lVert M_k - g\\,P_k \\rVert^2 \\big/ \\sum_k \\lVert M_k \\rVert^2$. For the touch-off "
         "approaches k runs over the samples at W under 10&#8202;mm, and for the map over the hovers above one station. $M_k$ "
         "is the measured change and $P_k$ the model&#8217;s, each with 15 components, and $g$ the gain. A share below zero "
         "means the model fits worse than a prediction of no change."),
        ("Net moment", "The vector sum of a tip&#8217;s four magnetic moments, in mA&#8202;m&#178;, in the tip&#8217;s pad frame: "
         "x along the face&#8217;s normal, y and z in its plane. The nominal alternating square has none. It is given as a share "
         "of one magnet&#8217;s moment."),
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
        [brow(r) for r in B["rows"]], num_cols=tuple(range(2, 11)), n=TAB["field"],
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
               field + P.fig(FIG["budget"], f1, "Change at v2 ID 1&#8217;s board caused by a neighbouring fingertip, in its field model. (a) Pads "
                             "facing across a gap W. (b) Side by side with a gap g between the flanks. Solid: the neighbour&#8217;s "
                             "field. Dashed: the median left after subtracting the field predicted at a pose 0.3&#8202;mm and "
                             "0.3&#176; off. Blue: a five-magnet neighbour like v2 ID 1. Orange: four N52 magnets in an "
                             "alternating 5.0&#8202;mm square. Dashed black line: a 0.5&#8202;N face press; grey band: presses of "
                             "0.2&#8211;3&#8202;N. Script: <code>scripts/fingertip_crosstalk.py budget</code>.") + tab1))

    # 3. measured on the holder
    S_.append(("holder", "Neighbour field measured on the fingertip holder", holder_section(H, hn, fh, fv)))

    # 4. subtraction from the joint angles
    sub = (
        "<p>The joint angles give the neighbour&#8217;s pose, so its field can be predicted and subtracted. The SR2 hand&#8217;s "
        "SCS0009 servos read 0.29&#176; per tick, 0.2&#8211;0.35&#8202;mm at the fingertip. With pose errors of 0.1&#8202;mm and "
        f"0.1&#176;, {k1(r5('facing', 0))['residual_uT_median']:.1f}&#8202;&#181;T is left facing and touching, and "
        f"{k1(r5('side', 0))['residual_uT_median']:.0f}&#8202;&#181;T side by side (Table&#160;{TAB['field']}). At 0.3&#8202;mm and 0.3&#176; the "
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

    # 5. two-board pose fit
    def crow(kind, x):
        r = r5(kind, x)
        c = r["crb"]
        return [("Facing, W " if kind == "facing" else "Side by side, g ") + f"{x:g}&#8202;mm"] + [
            f"{c[k]['one_board']:.3f}" for k in ("known", "0.3", "1", "free")] + [
            f"{c[k]['two_boards']:.3f}" for k in ("0.3", "1", "free")]
    tab2 = P.table(
        ["Pose of a five-magnet neighbour", "Pose known", "One board, 0.3&#8202;mm, 0.3&#176; prior", "One board, 1&#8202;mm, 1&#176; prior",
         "One board, no prior", "Two boards, 0.3&#8202;mm, 0.3&#176; prior", "Two boards, 1&#8202;mm, 1&#176; prior", "Two boards, no prior"],
        [crow("facing", x) for x in FACING_W] + [crow("side", x) for x in SIDE_G], num_cols=tuple(range(1, 8)), n=TAB["crb"],
        caption="Cram&#233;r&#8211;Rao bound on a press along the face normal at v2 ID 1&#8217;s face centre, in N, with the "
                "neighbour&#8217;s pose estimated with the press. One board: v2 ID 1&#8217;s alone. Two boards: the neighbour&#8217;s "
                "board as well, whose reading v2 ID 1&#8217;s magnets change. Prior: the joint angles&#8217; standard deviation per axis.")
    fit = (
        "<p>The pose can also be estimated together with the press, from the boards themselves. The neighbour&#8217;s field "
        "then adds six unknowns, and the joint angles constrain them as a prior. With both boards and a 1&#8202;mm, 1&#176; prior, "
        f"the bound on a press at the face&#8217;s centre is {min(cf):.3f}&#8211;{max(cf):.3f}&#8202;N facing. Side by side it is "
        f"{min(cs):.3f}&#8211;{max(cs):.3f}&#8202;N, against {known:.3f}&#8202;N with the pose known (Table&#160;{TAB['crb']}). With v2 "
        f"ID 1&#8217;s board alone and the same prior it is {min(r5('facing', x)['crb']['1']['one_board'] for x in FACING_W):.3f}&#8211;"
        f"{max(r5('facing', x)['crb']['1']['one_board'] for x in FACING_W):.3f}&#8202;N facing and "
        f"{min(r5('side', x)['crb']['1']['one_board'] for x in SIDE_G):.3f}&#8211;{max(r5('side', x)['crb']['1']['one_board'] for x in SIDE_G):.3f}"
        "&#8202;N side by side. The first tip&#8217;s magnets also change the second board&#8217;s reading, and that change "
        "constrains the relative pose. Without any prior the bound with two boards is "
        f"{min(r5('facing', x)['crb']['free']['two_boards'] for x in FACING_W):.3f}&#8211;"
        f"{max(r5(k, x)['crb']['free']['two_boards'] for k, xs in (('facing', FACING_W), ('side', SIDE_G)) for x in xs):.3f}&#8202;N.</p>"
        "<p>The bound holds only if the field of the neighbour is modelled well. At contact side by side the field is "
        f"{r5('side', 0)['norm_uT']:,.0f}&#8202;&#181;T, so a model error of 1&#8202;% leaves {r5('side', 0)['norm_uT'] / 100:.0f}"
        "&#8202;&#181;T, close to a 0.2&#8202;N press. On the holder, two v3 tips facing each other changed the held boards by 55 "
        "to 60&#8202;&#181;T at contact. The model with each tip&#8217;s net moment left about 10&#8202;&#181;T of that "
        f"(Section&#160;{SEC['holder']}).</p>")
    S_.append(("fit", "Fitting the neighbour's pose from both boards", fit + tab2))

    # 6. sharpie
    sh = (
        "<p>On 27 October at 12:18 the owner pinched a marker pen between v1 ID 2 and ID 3, with both boards streaming to the "
        f"multi-board viewer (Figure&#160;{FIG['sharpie']}). The two tips were held by hand. Over the eight windows of the pinch, ID 2&#8217;s "
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
               sh + P.fig(FIG["sharpie"], f2, "The owner&#8217;s pinch of a marker pen between v1 ID 2 (blue) and ID 3 (orange), 2026-10-27, "
                          "from the multi-board viewer&#8217;s streams. (a) Each board&#8217;s change from rest, the norm over its 15 "
                          "components in bins of 0.2&#8202;s. Diamonds: the estimator&#8217;s uniform field in each window, as its "
                          "norm over the five chips. (b) The force of each board&#8217;s contact per window: the viewer&#8217;s "
                          "estimate (open) and with the two-board pose fit (filled). Grey: the eight windows. Script: "
                          "<code>scripts/fingertip_crosstalk.py sharpie</code>; rows in <code>data/sharpie.json</code>.")))

    # 7. limitations
    S_.append(("limits", "Limitations", (
        f"<p>Figure&#160;{FIG['budget']} and Tables&#160;{TAB['field']} and {TAB['crb']} come from the dipole model of v2 ID 1&#8217;s "
        "magnets, fitted at rest on the UR5. The neighbour has v2 ID 1&#8217;s magnet strength. The measured field of "
        f"Section&#160;{SEC['holder']} is between four-magnet v3 tips facing each other, through {hn['n_touch']} touch-off "
        f"approaches and {count(hn['n_hover'])} hovers; no side-by-side pose has been measured. The bound assumes an exact field model "
        "and the noise of a 50&#8202;ms window, and it is the bound for one press at the face&#8217;s centre. In the pinch the two "
        "v1 tips were held by hand, so neither the pose nor the force is known. The pinch&#8217;s boards use the fits of ID 2 on "
        "the UR5 and of ID 3 on the bench.</p>")))

    # 8. future work
    fw = [
        ("The neighbour&#8217;s field on the holder.", f"The map of the "
         + link(L["v3"], "plan for the four-magnet v3 tips") + " and the close hovers over the &#8722;y tip ran on 2026-10-27 "
         f"(Section&#160;{SEC['holder']}). The map&#8217;s geometry is drawn in Figure&#160;{FIG['render']}. Each held tip&#8217;s "
         "pose and strength are to be fitted in five folds by pose (<code>scripts/fingertip_crosstalk.py</code>). At W 5&#8202;mm "
         "and under, a held-out residual above 10&#8202;% of the field means the model cannot be subtracted to the "
         "board&#8217;s noise. With the net moments fitted on the touch-offs, "
         f"{min(hn['close_left']):.0f} to {max(hn['close_left']):.0f}&#8202;&#181;T of the +y board&#8217;s "
         f"{min(hn['close_meas']):.0f} to {max(hn['close_meas']):.0f}&#8202;&#181;T is left at W 0.5 to 1&#8202;mm. Of the "
         f"&#8722;y board&#8217;s {min(hn['cl_at'][0.5]['held3']['meas']):.0f} to {max(hn['cl_at'][0.5]['held3']['meas']):.0f}"
         f"&#8202;&#181;T at W 0.5&#8202;mm, {min(hn['cl_at'][0.5]['held3']['left']):.0f} to "
         f"{max(hn['cl_at'][0.5]['held3']['left']):.0f}&#8202;&#181;T is left. At these poses the held tips&#8217; magnets pull "
         "on the robot tip with at most 0.69&#8202;mN (<code>scripts/fingertip_crosstalk.py pull</code>)."),
        ("A pinch between two v3 tips.", "A plan of its own follows the map. The robot apex presses the held apex on the +y "
         "station at 0.25&#8211;3&#8202;N. The robot&#8217;s target is also shifted 2 and 4&#8202;mm off its apex, across and "
         "along, and turned by &#8722;20 to +20&#176;. The apex press comes first, in the middle and last. The robot "
         "tip&#8217;s contact is estimated from the raw change, with the modelled field at the commanded pose subtracted, and with "
         "the held tip&#8217;s pose fitted from the board. With the fitted pose, a force error above 0.2&#8202;N at 0.5&#8202;N "
         f"contradicts the one-board bound of Table&#160;{TAB['crb']}."),
        ("A five-magnet neighbour.", "The owner tabled this check on 2026-10-27. It takes about 30&#8202;min with a five-magnet "
         "tip on the +y station. Its 483&#8202;&#181;T at W 0.5&#8202;mm tests the model at 1&#8202;% (4.8&#8202;&#181;T); at "
         "v3&#8217;s 34&#8202;&#181;T, 1&#8202;% is about the board&#8217;s noise. A misfit above 1&#8202;% at the closest poses "
         f"means the bounds of Table&#160;{TAB['crb']} do not apply."),
    ]
    S_.append(("future", "Future work", "<ul class=\"open\">" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in fw) + "</ul>"
               + P.fig(FIG["render"], io.BytesIO(at_head(HOVER)), "Geometry of the crosstalk map on the UR5 cell, rendered in MuJoCo from the fixtures&#8217; "
                       "registrations. The fingertip holder (centre) stands level on the 2x4, with both held tips facing up. The "
                       "pad of v3 ID 1 on the robot hovers 2&#8202;mm over the +y held tip at twist 0, pose "
                       "<code>hov-f-W2-t+0</code> of the plan. The homing seat stands in front of the holder&#8217;s column. "
                       "Image: <code>" + str(HOVER.relative_to(ROOT)) + "</code>, by <code>scripts/ur5_cell_render.py</code>.")))
    return S_, lede, summary


def build():
    B = json.loads((DATA / "budget.json").read_text())
    S = json.loads((DATA / "sharpie.json").read_text())
    T = sharpie_trace()
    Sn = sharpie_numbers(S)
    H = json.loads(at_head(HOLD))
    if H.get("made") != HOLD_MADE:
        print(f"fingertip_crosstalk_page: {HOLD.name} was remade at {H.get('made')} (the page's sentences quote {HOLD_MADE}); "
              "check the measured section against the coordinator's newest feed line", flush=True)
    f1 = fig_budget(B)
    f2 = fig_sharpie(S, T, Sn)
    fh = fig_holder(H)
    fv = fig_hovers(H)
    S_, lede, summary = sections(B, S, Sn, H, f1, f2, fh, fv)
    assert [sid for sid, _, _ in S_] == list(SEC), [sid for sid, _, _ in S_]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S_, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S_, 1))
    byline = ("Field model of v2 ID 1 (five N52) fitted at rest in UR5 session 20261026-192158-v2id1; the owner&#8217;s pinch with "
              "v1 ID 2 and ID 3 on 2026-10-27 at 12:18; three v3 tips on the UR5&#8217;s fingertip holder in sessions 200536-holdv3id1, "
              "202426-holdapexv3id1 and 212131-holdmapv3id1 on 2026-10-27, analysed by <code>scripts/fingertip_crosstalk_holder.py</code> "
              "(commits 53993cbc and 1114cb94). Analysis <code>scripts/fingertip_crosstalk.py</code> (commit bc508a6e); "
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
