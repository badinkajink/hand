#!/usr/bin/env python3
"""Page on the motions of the UR5 calibration for the four-magnet fingertips (owner, 2026-10-29: the indenter plate v2d
waits "until we have 100% finalized all the motions we need for effective calibration"). Reads what
scripts/ur5_tip_calibration_motion_value.py measured on v3 ID 1 and ID 2
(docs/experiments/20261027-v3_tip_calibration_and_crosstalk/data/20261029-motion_value.json) and the twist rows of
scripts/ur5_tip_calibration_four_vs_five.py, draws the figures into media/ and writes
docs/experiments/20261029-calibration_motions/20261029-calibration_motions_v3.html.

    .venv/bin/python scripts/calibration_motions_page.py
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
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import ur5_tip_calibration_page as P  # noqa: E402   (esc, fig, film, table, plt_setup, TEX_RE)

D = ROOT / "docs/experiments/20261029-calibration_motions"
MEDIA = D / "media"
STEM = "20261029-calibration_motions_v3"
PAGE = D / f"{STEM}.html"
TEMPLATE = ROOT / "scripts/calibration_motions_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
EXP = ROOT / "docs/experiments"
V3 = EXP / "20261027-v3_tip_calibration_and_crosstalk"
DATA = V3 / "data/20261029-motion_value.json"
TWISTS = {"v3 ID 1": V3 / "data/20261028-four_vs_five_twists.json", "v3 ID 2": V3 / "data/20261029-four_vs_five_grid_v3id2.json"}
TOP_EDGE = V3 / "data/20261029-top_edge_gp_v3id2.json"
PLATE_FILM = EXP / "20261028-indenter_plate_layout/media/20261028-indenter_plate_layout_v2d_film.mp4"
PLATE_POSTER = EXP / "20261028-indenter_plate_layout/media/20261028-indenter_plate_layout_v2d_film_poster.png"
L = dict(v3=V3 / "README.md", plate=EXP / "20261028-indenter_plate_layout/20261028-indenter_plate_layout.html",
         state=EXP / "20261022-ur5_tip_calibration/lab/STATE.md", queue=EXP / "20261022-ur5_tip_calibration/QUEUE.md",
         tall=EXP / "20261026-tall_indenter/README.md",
         status=EXP / "20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html")
TIPS = ("v3 ID 1", "v3 ID 2")
FX = ("sphere", "tall", "rod")
FX_NAME = dict(sphere="Low sphere", tall="Tall indenter", rod="Rod")
TOUCH_S = dict(sphere=48.0, tall=55.0, rod=55.0)          # a touch-off at v 0.0005 (lab NOTES, "Faster touch-offs")


def link(p, text):
    return f'<a href="{P.esc(os.path.relpath(p, D))}">{text}</a>'


def code(s):
    return f"<code>{P.esc(s)}</code>"


def mm(x, d=2):
    return f"{x:.{d}f}"


def pct(x):
    return f"{100 * x:.0f}"


# ------------------------------------------------------------------------------------------------------------ figures
def fig_fixtures(R, out):
    plt = P.plt_setup()
    sets = (("own_cv", "Own presses, five folds", "#1f5fbf"), ("pooled_cv", "Three indenters pooled, five folds", "#2e8b57"),
            ("without_it", "Trained without its own presses", "#c0392b"), ("sphere_only", "Trained on the low sphere alone", "#e8a35c"))
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=110)
    groups = [(t, f) for t in TIPS for f in FX]
    x = np.arange(len(groups))
    w = 0.2
    for k, (key, lab, col) in enumerate(sets):
        loc, frc = [], []
        for t, f in groups:
            s = R["tips"][t]["fixtures"][key].get(f)
            loc.append(s["loc_mm"] if s else np.nan)
            frc.append(100 * s["force_rel_err"] if s else np.nan)
        ax[0].bar(x + (k - 1.5) * w, loc, w, color=col, label=lab)
        ax[1].bar(x + (k - 1.5) * w, frc, w, color=col, label=lab)
    short = dict(sphere="Sphere", tall="Tall", rod="Rod")
    labels = [f"{short[f]}\n{t.replace('v3 ', '')}" for t, f in groups]
    for a, yl in zip(ax, ("Location error (mm)", "Force error (%)")):
        a.set_xticks(x, labels)
        a.set_ylabel(yl)
        a.grid(alpha=0.3, axis="y")
        a.axvline(2.5, color="k", lw=0.6)
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_time(R, out):
    plt = P.plt_setup()
    T = R["tips"]["v3 ID 2"]["time"]
    fams = (("touch-off", "Touch-offs", "#7f7f7f"), ("press", "Presses", "#1f5fbf"), ("end press", "Presses on the tip's end",
                                                                                        "#17becf"),
            ("slide", "Slides", "#c0392b"), ("twist", "Twists", "#e8a35c"), ("roll", "Rolls", "#9467bd"))
    fig, ax = plt.subplots(figsize=(11.0, 3.4), dpi=110)
    y = np.arange(len(FX))
    left = np.zeros(len(FX))
    for key, lab, col in fams:
        h = np.array([T[f].get(key, {}).get("hours", 0.0) for f in FX])
        ax.barh(y, h, left=left, color=col, label=lab)
        left += h
    ax.set_yticks(y, [FX_NAME[f] for f in FX])
    ax.invert_yaxis()
    ax.set_xlabel("Robot time on v3 ID 2's night suite (h)")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(frameon=False, ncol=3, fontsize=11, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_count(R, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=110, sharey=True)
    cols = dict(all="#000000", sphere="#1f5fbf", tall="#c0392b", rod="#2e8b57")
    names = dict(all="All presses", sphere="Low sphere", tall="Tall indenter", rod="Rod")
    top = 0.0
    for a, t in zip(ax, TIPS):
        c = R["tips"][t]["count"]
        fx = R["tips"][t]["fixtures"]["pooled_cv"]
        n_full = round(c["actions"] * 4 / 5)
        xs = [c[f"{fr:g}"]["training_actions"] for fr in (0.25, 0.5, 0.75)] + [n_full]
        for k in ("all", "sphere", "tall", "rod"):
            ys = [c[f"{fr:g}"][k]["loc_mm"] for fr in (0.25, 0.5, 0.75)] + [fx[k]["loc_mm"]]
            a.plot(xs, ys, "o-", color=cols[k], label=names[k], lw=2 if k == "all" else 1.3)
            top = max(top, max(ys))
        a.set_title(t)
        a.set_xlabel("Press actions in training")
        a.grid(alpha=0.3)
    ax[0].set_ylim(0, 1.08 * top)
    ax[0].set_ylabel("Location error (mm)")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_transfer(R, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=110, sharey=True)
    cols = dict(all="#000000", sphere="#1f5fbf", tall="#c0392b", rod="#2e8b57")
    names = dict(all="All presses", sphere="Low sphere", tall="Tall indenter", rod="Rod")
    top = 0.0
    for a, (key, r) in zip(ax, R["transfer"].items()):
        b = key.split(" -> ")[1]
        own = R["tips"][b]["fixtures"]["pooled_cv"]
        ks = [0] + [int(k) for k in r["with_k"]]
        for f in ("all", "sphere", "tall", "rod"):
            ys = [r["presses"][f]["loc_mm"]] + [r["with_k"][k][f]["loc_mm"] for k in r["with_k"]]
            a.plot(ks, ys, "o-", color=cols[f], label=names[f], lw=2 if f == "all" else 1.3)
            a.axhline(own[f]["loc_mm"], color=cols[f], ls=":", lw=1)
            top = max(top, max(ys))
        a.set_title(f"{key.replace(' -> ', ' calibration on ')}")
        a.set_xlabel(f"Press actions of {b} added to training")
        a.grid(alpha=0.3)
    ax[0].set_ylim(0, 1.08 * top)
    ax[0].set_ylabel("Location error (mm)")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------- sections
def sections(R, figs):
    S = []
    t1, t2 = TIPS
    F = {t: R["tips"][t]["fixtures"] for t in TIPS}
    T2 = R["tips"][t2]["time"]
    hours = lambda t, f, k: R["tips"][t]["time"][f].get(k, {}).get("hours", 0.0)          # noqa: E731
    total2 = sum(v["hours"] for f in FX for k, v in T2[f].items() if k != "press families")
    slide2 = sum(hours(t2, f, "slide") for f in FX)
    turn2 = sum(hours(t2, f, k) for f in FX for k in ("twist", "roll"))
    touch2 = sum(hours(t2, f, "touch-off") for f in FX)
    touch_new = sum(T2[f]["touch-off"]["actions"] * TOUCH_S[f] for f in FX) / 3600
    hard = T2["sphere"]["press families"].get("hard", {})

    # 1 terms
    terms = [
        ("v3 ID 1, v3 ID 2", "The two four-magnet tips that ran the suite. Each has pad 05R4 with four N52 magnets of 3 x 2 "
         "mm in a 5.0 mm square, their centres 8.5 mm under the apex. v3 ID 1 ran it from 2026-10-27 to 10-28, v3 ID 2 in the night "
         "of 10-28 to 10-29."),
        ("Low sphere, tall indenter, rod", "The three indenters on the cell. The low sphere is a 5 mm sphere on a cone, its "
         "top 46 mm above the table. The tall indenter is a 5 mm sphere on a 160 mm column, which reaches the flanks and "
         "edges at steep angles. The rod is a half rod of 12.5 mm radius, the bench screwdriver's, its axis across the "
         "finger."),
        ("Press, hold", "A press moves the pad along its normal onto a target point and stops at force levels from 0.25 to "
         "8 N. A hold is one level's window, the unit of the data. A press action is one target point with all its "
         "levels."),
        ("Slide, twist, roll", "A slide moves the pressed pad along the surface by a set amplitude in a set direction, "
         "at 1 N on most slides. A twist turns the pad about the contact normal, a roll about an axis in the surface."),
        ("Touch-off", "An approach to first contact, stopped at 0.5 N. A pass of 18 to 21 touch-offs registers each "
         "indenter."),
        ("Direction GP", "The estimator scored on this page: a Gaussian process from the direction of the board's change "
         "$d/|d|$ (15 chip components, uT) to the contact point $p$ and to $f/|d|$. The force estimate is that times $|d|$."),
        ("Location error", "Distance between the estimated and the registered contact point, across the contact normal, "
         "as the median over the test holds (mm)."),
        ("Force error, force bias", "Force error is $\\big||\\hat f| - |f|\\big| / |f|$, the force bias is $(|\\hat f| - "
         "|f|)/|f|$; medians in % of the measured force $f$ (HEX-E)."),
        ("Shear error", "Norm of the difference between the estimated and the measured shear, the force's part across the "
         "contact normal (N); median over the slides' holds."),
        ("Five folds by action", "Every action's holds are held out together; the GP is fitted on the other four fifths. "
         "\"Trained on A\" fits once on set A and predicts the holds named."),
        ("Gain", "The board's change per newton, $|d|/f_n$ (uT/N)."),
        ("Robot time", "From each action's start to its end in the session's log, travel included."),
    ]
    S.append(("terms", "Terms and metrics",
              '<dl class="terms">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in terms) + "</dl>"))

    # 2 time
    rows = []
    for f in FX:
        for key, lab in (("touch-off", "Touch-offs"), ("press", "Presses"), ("end press", "Presses on the tip's end"),
                         ("slide", "Slides"), ("twist", "Twists"), ("roll", "Rolls")):
            v = T2[f].get(key)
            if v:
                rows.append([FX_NAME[f], lab, str(v["actions"]), f"{v['median_s']:.0f}", f"{v['hours']:.2f}"])
    rows.append(["All", "", str(sum(int(r[2]) for r in rows)), "", f"{total2:.2f}"])
    body = (f"<p>v3 ID 2's night suite took {total2:.1f} h of robot time. The slides took {slide2:.1f} h of it, the "
            f"twists and rolls {turn2:.2f} h and the touch-offs {touch2:.2f} h. A slide on the low sphere takes "
            f"{T2['sphere']['slide']['median_s']:.0f} s, against {T2['rod']['slide']['median_s']:.0f} s on the rod. Table 1 "
            f"lists every family; Figure 2 draws them.</p>"
            + P.table(["Indenter", "Action", "Actions", "Median (s)", "Hours"], rows, (2, 3, 4),
                      "Robot time of v3 ID 2's night suite by indenter and action (10-28 18:44 to 10-29 08:13).", 1)
            + P.fig(2, figs["time"], "Robot time of v3 ID 2's night suite by indenter and action. The bars sum to "
                    f"{total2:.1f} h."))
    S.append(("time", "Robot time of the suite on v3 ID 2", body))

    # 3 fixtures
    rows = []
    for t in TIPS:
        for f in FX:
            o, p, w = F[t]["own_cv"][f], F[t]["pooled_cv"][f], F[t]["without_it"][f]
            s = F[t]["sphere_only"].get(f)
            rows.append([t, FX_NAME[f], str(F[t]["press_actions"][f]),
                         f"{mm(o['loc_mm'])} / {pct(o['force_rel_err'])}", f"{mm(p['loc_mm'])} / {pct(p['force_rel_err'])}",
                         f"{mm(w['loc_mm'])} / {pct(w['force_rel_err'])} ({w['force_rel_bias'] * 100:+.0f})",
                         f"{mm(s['loc_mm'])} / {pct(s['force_rel_err'])} ({s['force_rel_bias'] * 100:+.0f})" if s else "&#8212;"])
    w_loc = [F[t]["without_it"][f]["loc_mm"] for t in TIPS for f in FX]
    o_loc = [F[t]["own_cv"][f]["loc_mm"] for t in TIPS for f in FX]
    rodb = [F[t]["without_it"]["rod"]["force_rel_bias"] for t in TIPS]
    body = (f"<p>Each indenter's contacts must be in the training data. Trained on the other two indenters, the GP places "
            f"each indenter's presses {min(w_loc):.2f} to {max(w_loc):.2f} mm off, against {min(o_loc):.2f} to "
            f"{max(o_loc):.2f} mm with its own presses in five folds. The rod's forces are then read "
            f"{-100 * max(rodb):.0f} to {-100 * min(rodb):.0f} % low: a line contact gives less board change per newton than "
            f"the 5 mm sphere. The tall indenter's presses are placed {mm(F[t1]['without_it']['tall']['loc_mm'])} and "
            f"{mm(F[t2]['without_it']['tall']['loc_mm'])} mm off without its own presses, with a 90th percentile over 6 mm.</p>"
            f"<p>On the hand a contact's shape is unknown, so one GP must cover all three indenters. Pooling costs the low "
            f"sphere {mm(F[t1]['pooled_cv']['sphere']['loc_mm'] - F[t1]['own_cv']['sphere']['loc_mm'])} "
            f"and {mm(F[t2]['pooled_cv']['sphere']['loc_mm'] - F[t2]['own_cv']['sphere']['loc_mm'])} mm and the rod "
            f"{mm(F[t1]['pooled_cv']['rod']['loc_mm'] - F[t1]['own_cv']['rod']['loc_mm'])} and "
            f"{mm(F[t2]['pooled_cv']['rod']['loc_mm'] - F[t2]['own_cv']['rod']['loc_mm'])} mm against each indenter's own "
            f"GP.</p>"
            + P.table(["Tip", "Indenter", "Press actions", "Own, five folds", "Pooled, five folds", "Without its own (bias)",
                       "Low sphere alone (bias)"], rows, (2,),
                      "Location error (mm) / force error (%) of the direction GP on each indenter's presses, by training "
                      "set; force bias in % in brackets.", 2)
            + P.fig(3, figs["fixtures"], "Location and force error of the direction GP on each indenter's presses, by "
                    "training set, for v3 ID 1 (left three groups) and v3 ID 2 (right three)."))
    S.append(("fixtures", "Presses on each indenter", body))

    # 4 shared points
    rows = []
    for t in TIPS:
        s = R["tips"][t]["shared"]["at_1N_and_over"]
        rows.append([t, str(s["pairs"]), mm(s["point_mm"]), f"{s['tool_angle_deg']:.1f}", f"{s['board_angle_deg']:.1f}",
                     mm(s["gain_ratio"]), f"{mm(s['shear_per_N_tall'])} / {mm(s['shear_per_N_sphere'])}",
                     f"{mm(s['gp_loc_tall_mm'])} / {mm(s['gp_loc_sphere_mm'])}"])
    s1, s2 = (R["tips"][t]["shared"]["at_1N_and_over"] for t in TIPS)
    body = (f"<p>For the same registered contact, the board's change differs between the tall indenter and the low "
            f"sphere. Six "
            f"points, three on the face and three on the edges, were pressed on both at the same levels in the same "
            f"session. At 1 N and over, their registered contact points lie {mm(s1['point_mm'])} and "
            f"{mm(s2['point_mm'])} mm apart on v3 ID 1 and ID 2. The tool's orientation differs by {s1['tool_angle_deg']:.1f} "
            f"and {s2['tool_angle_deg']:.1f} deg, so gravity acts on the finger the same way. The board's changes point "
            f"{s1['board_angle_deg']:.1f} and {s2['board_angle_deg']:.1f} deg apart, and the tall indenter gives "
            f"{s1['gain_ratio']:.2f} and {s2['gain_ratio']:.2f} times the low sphere's change per newton. Its forces have a "
            f"shear of {mm(s1['shear_per_N_tall'])} and {mm(s2['shear_per_N_tall'])} N per newton, against "
            f"{mm(s1['shear_per_N_sphere'])} and {mm(s2['shear_per_N_sphere'])} on the low sphere, mostly along the finger.</p>"
            f"<p>A GP was trained on the low sphere's other presses. It places the six points {mm(s1['gp_loc_sphere_mm'])} "
            f"and {mm(s2['gp_loc_sphere_mm'])} mm off on the low sphere, and {mm(s1['gp_loc_tall_mm'])} and "
            f"{mm(s2['gp_loc_tall_mm'])} mm off on the tall indenter. The column's side compliance, 0.0009 mm/N "
            f"({link(L['tall'], 'tall indenter README')}), moves the sphere 0.005 mm at 5 N, so the column's bending "
            f"does not explain it. The cause is open. Plate v2d has one 5 mm sphere, its top 160 mm above the plate, for "
            f"both the low sphere's suite and the tall indenter's ({link(L['plate'], 'plate layout page')}). Whether a "
            f"sphere on such a column gives the low sphere's readings is untested.</p>"
            + P.table(["Tip", "Pairs", "Registered points apart (mm)", "Tool turned (deg)", "Board changes apart (deg)",
                       "Gain, tall over sphere", "Shear per newton, tall / sphere", "Sphere GP's location error, tall / sphere (mm)"],
                      rows, (1, 2, 3, 4, 5, 6, 7),
                      "The six shared points at 1 N and over: medians over the pairs of one point and level.", 3))
    S.append(("shared", "Tall indenter against the low sphere at six shared points", body))

    # 5 force levels
    rows = []
    for t in TIPS:
        for name in ("0.5 N and under", "over 3.5 N"):
            v = R["tips"][t]["levels"][name]
            rows.append([t, name, str(v["left_out"]["n"]), f"{mm(v['pooled_cv']['loc_mm'])} / {pct(v['pooled_cv']['force_rel_err'])}",
                         f"{mm(v['left_out']['loc_mm'])} / {pct(v['left_out']['force_rel_err'])}"])
    lv = {t: R["tips"][t]["levels"] for t in TIPS}
    body = (f"<p>Holds over 3.5 N add nothing to training. Left out, they are placed {mm(lv[t1]['over 3.5 N']['left_out']['loc_mm'])} "
            f"and {mm(lv[t2]['over 3.5 N']['left_out']['loc_mm'])} mm off with {pct(lv[t1]['over 3.5 N']['left_out']['force_rel_err'])} "
            f"and {pct(lv[t2]['over 3.5 N']['left_out']['force_rel_err'])} % force error, as well as in five folds with "
            f"them. The direction GP scales its force with $|d|$, so it extends to 8 N. The holds at 0.5 N and under are "
            f"placed 1.1 to 1.6 mm off whether or not they are in training. A force of 0.25 N is three to four times the "
            f"board's noise-equivalent force of 62 to 83 mN. The hard presses on the low sphere, which press to 6 to 8 N, "
            f"took {hard.get('hours', 0):.2f} h in {hard.get('actions', 0)} actions on v3 ID 2.</p>"
            + P.table(["Tip", "Holds", "Holds tested", "In training, five folds", "Left out of training"], rows, (2, 3, 4),
                      "Location error (mm) / force error (%) on the holds of a force range, with the range in the pooled "
                      "five folds and left out of training.", 4))
    S.append(("levels", "Force levels", body))

    # 6 count
    c = {t: R["tips"][t]["count"] for t in TIPS}
    body = (f"<p>Halving the press actions raises the pooled location error from {mm(F[t1]['pooled_cv']['all']['loc_mm'])} to "
            f"{mm(c[t1]['0.5']['all']['loc_mm'])} mm on v3 ID 1 and from {mm(F[t2]['pooled_cv']['all']['loc_mm'])} to "
            f"{mm(c[t2]['0.5']['all']['loc_mm'])} mm on v3 ID 2. A quarter of them gives {mm(c[t1]['0.25']['all']['loc_mm'])} "
            f"and {mm(c[t2]['0.25']['all']['loc_mm'])} mm. The curve flattens from three quarters on (Figure 4), so the press "
            f"count stays as it is. Each subset is drawn at random by action, three times, and its median is shown.</p>"
            + P.fig(4, figs["count"], "Location error of the pooled direction GP against the number of press actions in "
                    "training, by indenter; the rightmost point is the full five folds."))
    S.append(("count", "Number of presses", body))

    # 7 tip end
    te = {t: R["tips"][t]["tip_end"] for t in TIPS}
    top = json.loads(TOP_EDGE.read_text())
    body = (f"<p>The tip's end must be pressed. On v3 ID 2, a pooled GP without the {len(te[t2]['actions'])} end presses "
            f"places them {mm(te[t2]['without_end_presses']['loc_mm'])} mm off at the median, with an angle error of "
            f"{te[t2]['without_end_presses']['angle_deg']:.0f} deg. With the other end presses in training, one action left out "
            f"at a time, it places them {mm(te[t2]['with_other_end_presses']['loc_mm'])} mm off. A GP on the low sphere's "
            f"presses alone places them {mm(top['grid_only']['all']['loc_mm'])} and "
            f"{mm(top['with_the_pass']['all']['loc_mm'])} mm off ({link(L['v3'], 'v3 README')}, the top edge). v3 ID 1 has "
            f"only the {len(te[t1]['actions'])} end presses of the tall indenter's first 8 top-edge actions: "
            f"{mm(te[t1]['without_end_presses']['loc_mm'])} mm without them and "
            f"{mm(te[t1]['with_other_end_presses']['loc_mm'])} mm with the other.</p>")
    S.append(("tip_end", "Presses on the tip's end", body))

    # 8 slides
    rows = []
    for t in TIPS:
        for f in FX:
            v = R["tips"][t]["slides"].get(f)
            if not v:
                continue
            rows.append([t, FX_NAME[f], str(v["slide_actions"]), mm(v["hexe_shear_N"]),
                         f"{mm(v['presses']['shear_err_N'])} / {mm(v['presses']['loc_mm'])}",
                         f"{mm(v['presses_other_slides']['shear_err_N'])} / {mm(v['presses_other_slides']['loc_mm'])}",
                         f"{mm(v['all_cv']['shear_err_N'])} / {mm(v['all_cv']['loc_mm'])}"])
    sl = {t: R["tips"][t]["slides"] for t in TIPS}
    body = (f"<p>Slides are required in training, each indenter's own among them. On the low sphere's slides, a GP trained "
            f"on presses alone errs in shear by {mm(sl[t1]['sphere']['presses']['shear_err_N'])} and "
            f"{mm(sl[t2]['sphere']['presses']['shear_err_N'])} N on v3 ID 1 and ID 2. The measured shear is "
            f"{mm(sl[t1]['sphere']['hexe_shear_N'])} and {mm(sl[t2]['sphere']['hexe_shear_N'])} N. The other indenters' slides "
            f"bring the error to {mm(sl[t1]['sphere']['presses_other_slides']['shear_err_N'])} and "
            f"{mm(sl[t2]['sphere']['presses_other_slides']['shear_err_N'])} N, and the sphere's own slides in five folds to "
            f"{mm(sl[t1]['sphere']['all_cv']['shear_err_N'])} and {mm(sl[t2]['sphere']['all_cv']['shear_err_N'])} N.</p>"
            f"<p>The own slides also place the contact while it slides: {mm(sl[t1]['sphere']['all_cv']['loc_mm'])} and "
            f"{mm(sl[t2]['sphere']['all_cv']['loc_mm'])} mm on the low sphere, against "
            f"{mm(sl[t1]['sphere']['presses_other_slides']['loc_mm'])} and "
            f"{mm(sl[t2]['sphere']['presses_other_slides']['loc_mm'])} mm with only the other indenters' slides. The tall "
            f"indenter and the rod follow the same order (Table 5).</p>"
            + P.table(["Tip", "Indenter", "Slides", "Measured shear (N)", "Presses only", "With the other indenters' slides",
                       "With its own slides, five folds"], rows, (2, 3, 4, 5, 6),
                      "Shear error (N) / location error (mm) of the direction GP on each indenter's slides, by training set.", 5))
    S.append(("slides", "Slides and shear", body))

    # 9 twists
    tw = {}
    for t in TIPS:
        d = json.loads(TWISTS[t].read_text())
        key = "four_v3id1" if t == t1 else "four_v3id2"
        tw[t] = (d["twists"] if "twists" in d and key in d["twists"] else d)[key]
    body = (f"<p>No torsion estimate is learned from the twists. The GP's torsion estimate correlates with the HEX-E's at r = "
            f"{tw[t1]['gp_twists']['r']:.2f} and {tw[t2]['gp_twists']['r']:.2f} on v3 ID 1 and ID 2, with an error of "
            f"{tw[t1]['gp_twists']['rms_Nmm']:.1f} and {tw[t2]['gp_twists']['rms_Nmm']:.1f} N mm rms. The measured torsion "
            f"itself is {tw[t1]['gp_twists']['hexe_rms_Nmm']:.1f} and {tw[t2]['gp_twists']['hexe_rms_Nmm']:.1f} N mm rms. "
            f"The twists and rolls took {turn2:.2f} h on v3 ID 2. Their holds were left out of every training set on this "
            f"page, so what they add to location and force is not measured here.</p>")
    S.append(("twists", "Twists and rolls", body))

    # 10 transfer
    tr = R.get("transfer") or {}
    if tr:
        rows = []
        for key, r in tr.items():
            b = key.split(" -> ")[1]
            own = R["tips"][b]["fixtures"]["pooled_cv"]
            for f in ("all",) + FX:
                rows.append([key.replace(" -> ", " on "), FX_NAME.get(f, "All presses"),
                             f"{mm(r['presses'][f]['loc_mm'])} / {pct(r['presses'][f]['force_rel_err'])}"]
                            + [f"{mm(r['with_k'][k][f]['loc_mm'])} / {pct(r['with_k'][k][f]['force_rel_err'])}" for k in r["with_k"]]
                            + [f"{mm(own[f]['loc_mm'])} / {pct(own[f]['force_rel_err'])}"])
        ks = list(next(iter(tr.values()))["with_k"])
        a, b = list(tr)
        ra, rb = tr[a], tr[b]
        c1, c2 = R["tips"][t1]["count"], R["tips"][t2]["count"]
        body = (f"<p>Each tip must run its own suite. Trained on v3 ID 1's presses and slides, the GP places v3 ID 2's "
                f"presses {mm(ra['presses']['all']['loc_mm'])} mm off, with {pct(ra['presses']['all']['force_rel_err'])} % "
                f"force error. In ID 2's own five folds they are placed {mm(F[t2]['pooled_cv']['all']['loc_mm'])} mm off. "
                f"The other way round it is "
                f"{mm(rb['presses']['all']['loc_mm'])} mm against {mm(F[t1]['pooled_cv']['all']['loc_mm'])} mm.</p>"
                f"<p>A tip's own presses added to the other tip's data recover most of the error, but no more than its own "
                f"presses alone. With {ks[-1]} of v3 ID 2's press actions added to ID 1's data, ID 2's presses are placed "
                f"{mm(ra['with_k'][ks[-1]]['all']['loc_mm'])} mm off. From {c2['0.5']['training_actions']} of ID 2's own "
                f"actions alone they are placed {mm(c2['0.5']['all']['loc_mm'])} mm off (section 6). On v3 ID 1 the same "
                f"gives "
                f"{mm(rb['with_k'][ks[-1]]['all']['loc_mm'])} mm against {mm(c1['0.5']['all']['loc_mm'])} mm from "
                f"{c1['0.5']['training_actions']} of its own (Figure 5).</p>"
                + P.table(["Calibration", "Presses", "Alone"] + [f"With {k}" for k in ks] + ["Tip's own, five folds"], rows,
                          tuple(range(2, 4 + len(ks))),
                          "Location error (mm) / force error (%) on the other tip's presses: one tip's presses and slides "
                          "alone, with k of the other tip's press actions added, and the other tip's own five folds.", 6)
                + P.fig(5, figs["transfer"], "Location error on the other tip's presses against the number of its press "
                        "actions added to one tip's calibration; dotted, the tip's own five folds."))
        S.append(("transfer", "One tip's calibration on the other", body))

    # 11 list
    S.append(("list", "Motions for the calibration and for the plate", motion_list(R, T2, touch_new, touch2, hard, turn2)))
    S.append(("future", "Future work", future()))
    return S


def motion_list(R, T2, touch_new, touch2, hard, turn2):
    t1, t2 = TIPS
    rows = [
        ["Touch-offs, pass 1", "Each indenter", "18 to 21", f"{touch_new:.2f}", "Keep, at 0.15 mm/s (owner, 10-29)",
         "Registers each indenter"],
        ["Grid presses, 0.25 to 3.5 N", "Low sphere", f"{T2['sphere']['press']['actions'] - hard.get('actions', 0)}",
         f"{T2['sphere']['press']['hours'] - hard.get('hours', 0):.2f}", "Keep",
         "Without them: 0.90 to 0.96 mm (Table 2)"],
        ["Presses on the tip's end", "Low sphere", "12", f"{T2['sphere'].get('end press', {}).get('hours', 0):.2f}",
         "Keep", "Without them: over 5 mm (section 7)"],
        ["Hard presses, 4 to 8 N", "Low sphere", str(hard.get("actions", 0)), f"{hard.get('hours', 0):.2f}",
         "Cut to 5 as a test", "Extrapolated from 3.5 N (Table 4)"],
        ["Grid presses, 0.25 to 3.5 N", "Tall indenter", str(T2["tall"]["press"]["actions"]), f"{T2['tall']['press']['hours']:.2f}",
         "Keep", "Without them: 1.29 to 1.80 mm (Table 2)"],
        ["Grid presses, 0.25 to 3.5 N", "Rod", str(T2["rod"]["press"]["actions"]), f"{T2['rod']['press']['hours']:.2f}",
         "Keep", "Without them: forces 14 % low (Table 2)"],
        ["Slides", "Each indenter", str(sum(T2[f]["slide"]["actions"] for f in FX)),
         f"{sum(T2[f]['slide']['hours'] for f in FX):.2f}", "Keep, on each indenter",
         "Shear error halves; own slides place sliding contacts (Table 5)"],
        ["Twists and rolls", "Low sphere, tall indenter, rod", str(sum(T2[f].get(k, {}).get("actions", 0) for f in FX
                                                                        for k in ("twist", "roll"))),
         f"{turn2:.2f}", "Cut to 4 twists on the low sphere", "The GP gives no torsion (section 9); 4 remain as test data"],
        ["Facing hovers", "Holder, pad-up tip", "137", "&#8212;", "Done on 10-27", "Crosstalk between facing pads"],
        ["Side-by-side hovers", "Holder, -y tip facing up", "14 and repeats", "0.2", "Run once",
         "Crosstalk beside a neighbour"],
    ]
    keep = (touch_new + sum(T2[f]["press"]["hours"] for f in FX) - hard.get("hours", 0) + 5 * 64 / 3600
            + T2["sphere"].get("end press", {}).get("hours", 0) + sum(T2[f]["slide"]["hours"] for f in FX)
            + 4 * T2["sphere"]["twist"]["median_s"] / 3600)
    full = sum(v["hours"] for f in FX for k, v in T2[f].items() if k != "press families")
    return (f"<p>Table 7 lists the motions with v3 ID 2's robot time. With the cuts in the table, a tip's suite takes "
            f"{keep:.1f} h against v3 ID 2's {full:.1f} h. The kept motions run as they did, with the faster touch-offs, "
            f"4 twists and 5 hard presses. The plate is laid out from the indenters and pad regions in the table. The low "
            f"5 mm sphere is pressed on the face, the edges and the tip's end, and the tall column on the flanks. The rod "
            f"and the holder, with both tips facing up, are the other two.</p>"
            + P.table(["Motion", "Indenter", "Actions", "Hours", "Decision", "Reason"], rows, (2, 3),
                      "Motions of the calibration suite, with v3 ID 2's counts and robot time.", 7))


def future():
    items = [
        "Find why the tall indenter and the low sphere differ at the same registered contact. Press the six shared "
        "points on both again in one session, with the HEX-E's raw wrench logged at 1 kHz. Then compare the force's "
        "direction during the approach. If the board directions agree within 3 deg and the gains within 5 %, one 5 mm "
        "sphere on a column can stand for both.",
        "Measure what a 12.5 mm sphere adds: a pass of 30 face and edge presses on it, scored as the rod is in Table 2. "
        "If the pooled GP without them places them within 0.1 mm of its own five folds, the plate needs no 12.5 mm sphere.",
        "Thin the slides: rerun the slides part of scripts/ur5_tip_calibration_motion_value.py with half the low sphere's "
        "slides, by direction and by amplitude. A rise of under 10 % in shear error would halve their 4.2 h.",
        "Test the twists' value for location and force: add their holds to the pooled training set in "
        "scripts/ur5_tip_calibration_motion_value.py and score the presses and slides again. If the errors move by under "
        "2 %, cutting them to 4 stands.",
        "Score ID 3's suite against ID 1 and ID 2 with --parts transfer once it runs.",
    ]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def build():
    R = json.loads(DATA.read_text())
    MEDIA.mkdir(parents=True, exist_ok=True)
    figs = dict(time=MEDIA / f"{STEM}_time.png", fixtures=MEDIA / f"{STEM}_fixtures.png", count=MEDIA / f"{STEM}_count.png",
                transfer=MEDIA / f"{STEM}_transfer.png")
    fig_time(R, figs["time"])
    fig_fixtures(R, figs["fixtures"])
    fig_count(R, figs["count"])
    if R.get("transfer"):
        fig_transfer(R, figs["transfer"])
    S = sections(R, figs)
    film_html = P.film(1, PLATE_FILM, PLATE_POSTER,
                       "The calibration's motions, one action of each kind, replayed on the proposed indenter plate v2d "
                       f"({link(L['plate'], 'plate layout page')}). Two side-by-side hovers and a facing hover come first. "
                       "A flank slide and a slide on the 5 mm sphere follow, then a press on the 12.5 mm sphere and a "
                       "slide on the rod.")
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {hd}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {hd}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    lede, summary = lede_text(R)
    byline = ("Direction GP on v3 ID 1's and ID 2's sessions on the low sphere, the tall indenter and the rod. Analysis "
              + code("scripts/ur5_tip_calibration_motion_value.py") + ", rows in "
              + link(DATA, "data/20261029-motion_value.json") + "; page by " + code("scripts/calibration_motions_page.py")
              + ". The cell's state: " + link(L["state"], "lab STATE.md") + "; the plate: "
              + link(L["plate"], "indenter plate layout") + "; the tips: " + link(L["v3"], "v3 README") + ".")
    page = TEMPLATE.read_text()
    title = "Contribution of each UR5 calibration motion on the four-magnet fingertips"
    for k, v in {"TITLE": title, "H1": title, "EYEBROW": "Printed fingertips &#183; 2026-10-29", "LEDE": lede,
                 "BYLINE": byline, "FILM": film_html, "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = P.TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary, date="2026-10-29")
    PAGE.write_text(page)
    print(f"{PAGE}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


def lede_text(R):
    t1, t2 = TIPS
    F = {t: R["tips"][t]["fixtures"] for t in TIPS}
    w = [F[t]["without_it"][f]["loc_mm"] for t in TIPS for f in FX]
    o = [F[t]["own_cv"][f]["loc_mm"] for t in TIPS for f in FX]
    hi = [R["tips"][t]["levels"]["over 3.5 N"]["left_out"]["loc_mm"] for t in TIPS]
    s1, s2 = (R["tips"][t]["shared"]["at_1N_and_over"] for t in TIPS)
    tr = R["transfer"]
    tx = [v["presses"]["all"]["loc_mm"] for v in tr.values()]
    lede = (f"Presses on each of the three indenters, presses on the tip's end and slides must all be in a tip's training "
            f"data. Without its own presses each indenter's contacts are placed {min(w):.2f} to {max(w):.2f} mm off, against "
            f"{min(o):.2f} to {max(o):.2f} mm with them. Presses over 3.5 N add nothing: left out of training, they are "
            f"still placed within {min(hi):.2f} to {max(hi):.2f} mm. No torsion estimate is learned from the twists. "
            f"Trained on one tip, the GP places the "
            f"other tip's presses {min(tx):.2f} to {max(tx):.2f} mm off, so each tip must run its own suite. At six points "
            f"pressed on both, the board's changes on the tall indenter and the low sphere point "
            f"{min(s1['board_angle_deg'], s2['board_angle_deg']):.0f} to {max(s1['board_angle_deg'], s2['board_angle_deg']):.0f} "
            f"deg apart for the same contact. That comparison must be repeated on plate v2d's single 5 mm sphere, on a 160 mm "
            f"column, before it replaces the low sphere.")
    summary = (f"On v3 ID 1 and ID 2, each indenter's presses, the tip's end and slides must be in a tip's calibration: an "
               f"indenter left out is placed {min(w):.2f} to {max(w):.2f} mm off against {min(o):.2f} to {max(o):.2f} mm. "
               f"Presses over 3.5 N add nothing, twists give no torsion estimate; one tip's calibration places the other's presses "
               f"{min(tx):.2f} to {max(tx):.2f} mm off; the tall indenter and the low sphere disagree by "
               f"{min(s1['board_angle_deg'], s2['board_angle_deg']):.0f} to {max(s1['board_angle_deg'], s2['board_angle_deg']):.0f} "
               f"deg at the same contact.")
    return lede, summary


if __name__ == "__main__":
    build()
