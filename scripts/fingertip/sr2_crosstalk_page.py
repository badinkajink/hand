#!/usr/bin/env python3
"""Builds docs/experiments/20261104-sr2_hand_crosstalk/20261105-sr2_hand_fingertip_crosstalk.html: magnetic crosstalk
between the three fingertip boards on the SR2 hand, measured on the UR5 at the home layout (night of 2026-11-04) and at
the stepper layouts of 2026-11-05 (owner, 18:2x on 11-05: "this crosstalk analysis is high priority, lets make sure it
gets done and addressed with the appropriate effort").

Data: the analysis of scripts/fingertip/sr2_crosstalk_hand.py (data/<STEM>.json and data/<STEM>_rows.npz), the lab's
runbook numbers in the folder's README.md, and the time-lapse film media/20261105-timelapse.mp4 when the lab has pushed it.
The design page's helpers draw the tables and figures (scripts/hom/fp_review_page.py).

    .venv/bin/python scripts/fingertip/sr2_crosstalk_page.py figs [STEM]     # the drawings
    .venv/bin/python scripts/fingertip/sr2_crosstalk_page.py [STEM]          # the page
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import docs_index  # noqa: E402
import fp_review_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

fill, table, img, film, esc = P.fill, P.table, P.img, P.film, P.esc
D = ROOT / "docs/experiments/20261104-sr2_hand_crosstalk"
DATA = D / "data"
MEDIA = D / "media"
OUT = D / "20261105-sr2_hand_fingertip_crosstalk.html"
TEMPLATE = ROOT / "scripts/fingertip/sr2_crosstalk_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
STEM = "20261105-sr2_crosstalk_hand_20261104_20261105"
F = ("thumb", "index", "middle")
TIP_ID = {"thumb": 1, "index": 3, "middle": 2}
COLOR = {"thumb": "#b03a2e", "index": "#1f4e9c", "middle": "#5b8c3a"}
LAYOUT_LABEL = {"home": "Home", "g12": "g12", "sv1_w6689": "sv1_w6689", "rv04_mid": "rv04_mid", "sv1_w0099": "sv1_w0099",
                "sv1_u0308": "sv1_u0308", "hom_roll_85_170": "HOM roll", "hom_press_70_150": "HOM press",
                "hom_generalist_70_110": "HOM generalist", "hom_twist_100_70": "HOM twist", "hom_cover_50_75": "HOM cover"}
STAGES = (("raw", "No compensation"), ("own_angles", "Own-angle GP, five folds"),
          ("own_angles_left_out_layout", "Own-angle GP, layout left out"),
          ("own_angles_home_only", "Own-angle GP, home layout only"),
          ("all_angles_mounts", "Nine-angle and mount GP, five folds"),
          ("own_angles_and_magnets_driver", "Own-angle GP and magnet model, driver's spacing"),
          ("own_angles_and_magnets_cad_v3", "Own-angle GP and magnet model, cad_v3 spacing"))


def load(stem):
    R = json.loads((DATA / f"{stem}.json").read_text())
    Z = np.load(DATA / f"{stem}_rows.npz", allow_pickle=False)
    return R, Z


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
                         "legend.fontsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 13, "mathtext.fontset": "cm"})
    return plt


def save(fig, name):
    import matplotlib.pyplot as plt
    MEDIA.mkdir(exist_ok=True)
    fig.savefig(MEDIA / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", (MEDIA / name).relative_to(ROOT), flush=True)


def fig_layouts(R, Z):
    """The mounts of every layout in the palm frame: a colour per layout, a marker per finger."""
    plt = plt_setup()
    lay, M = Z["layout"].astype(str), Z["mounts"]
    seen = {}
    for name, m in zip(lay, M):
        if name != "other":
            seen.setdefault(name, m)
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    for f, x0, y0, w, h in (("thumb", -80, -55, 60, 110), ("index", 20, 25, 60, 60), ("middle", 20, -85, 60, 60)):
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ls=":", ec="0.5", lw=1.0))
    cmap = plt.get_cmap("tab10")
    marker = {"thumb": "o", "index": "s", "middle": "^"}
    for k, (name, m) in enumerate(seen.items()):
        xy = m.reshape(3, 2)
        c = cmap(k % 10)
        ax.plot(np.r_[xy[:, 0], xy[0, 0]], np.r_[xy[:, 1], xy[0, 1]], color=c, lw=1.0, alpha=0.6)
        for j, f in enumerate(F):
            ax.plot(*xy[j], marker[f], color=c, ms=8, mec="k", mew=0.5)
        ax.plot([], [], "-", color=c, lw=3, label=LAYOUT_LABEL.get(name, name))
    for f in F:
        ax.plot([], [], marker[f], color="w", mec="k", ms=8, label=f"{f.capitalize()} (v3 ID {TIP_ID[f]})")
    ax.set_xlabel("$x_P$ (mm)")
    ax.set_ylabel("$y_P$ (mm)")
    ax.set_aspect("equal")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(alpha=0.3)
    save(fig, "20261105-layouts.png")


def norms(Z, f, key):
    Y = Z[f"{f}_Y"]
    P_ = Z[f"{f}_P_{key}"] if key != "raw" else np.zeros_like(Y)
    e = np.linalg.norm(Y - P_, axis=1)
    return e[np.isfinite(e)]


def fig_cdf(R, Z):
    """Each board's change and what each model leaves, as cumulative distributions in uT and N."""
    plt = plt_setup()
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.0), sharey=True)
    styles = (("raw", "No compensation", "0.55", "-"), ("own_angles", "Own-angle GP, five folds", "k", "-"),
              ("own_angles_left_out_layout", "Own-angle GP, layout left out", "#b03a2e", "--"),
              ("own_angles_home_only", "Own-angle GP, home only", "#1f4e9c", ":"))
    for ax, f in zip(axs, F):
        if f"{f}_Y" not in Z:
            ax.set_visible(False)
            continue
        upn = R["uT_per_N"][f]
        for key, lab, c, ls in styles:
            if key != "raw" and f"{f}_P_{key}" not in Z:
                continue
            e = np.sort(norms(Z, f, key))
            ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=c, ls=ls, lw=1.8, label=lab if f == F[0] else None)
        nz = [v["median_uT"] for k, v in R["noise"].items() if k.endswith(f) and v["median_uT"]]
        if nz:
            ax.axvline(float(np.median(nz)), color="0.4", lw=1, ls="-.", label="Zero-pose noise, median" if f == F[0] else None)
        ax.set_xscale("log")
        ax.set_xlim(0.8, 300)
        ax.text(0.97, 0.04, f"{f.capitalize()} board\n(v3 ID {TIP_ID[f]})", transform=ax.transAxes, ha="right", va="bottom")
        ax.set_xlabel("|d| (µT)")
        sec = ax.secondary_xaxis("top", functions=(lambda x, u=upn: x / u, lambda x, u=upn: x * u))
        sec.set_xlabel("Force equivalent (N)")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("Share of poses")
    fig.legend(loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.14))
    save(fig, "20261105-residual_cdf.png")


def fig_neighbour(R, Z):
    """What the own-angle GP leaves against the distance to the nearest other pad (cad_v3 spacing), with the magnet
    model's field there."""
    plt = plt_setup()
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)
    for ax, f in zip(axs, F):
        if f"{f}_dist_cad_v3" not in Z:
            ax.set_visible(False)
            continue
        ok = Z[f"{f}_mount_ok"].astype(bool)
        d = Z[f"{f}_dist_cad_v3"][ok]
        e = np.linalg.norm(Z[f"{f}_Y"] - Z[f"{f}_P_own_angles"], axis=1)[ok]
        m = np.linalg.norm(Z[f"{f}_M_cad_v3"].sum(1), axis=1)[ok]
        ax.plot(d, e, ".", color=COLOR[f], ms=4, alpha=0.6, label="Left by the own-angle GP, this board")
        ax.plot(d, m, ".", color="0.2", ms=3, alpha=0.5, label="Magnet model, cad_v3 spacing" if f == F[0] else None)
        ax.set_yscale("log")
        ax.set_ylim(0.01, 300)
        ax.set_title(f"{f.capitalize()} board")
        ax.set_xlabel("Nearest other pad (mm)")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("|d| (µT)")
    fig.legend(loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    save(fig, "20261105-neighbour_distance.png")


def fig_trace(R, Z):
    """The thumb's change through a stretch of poses in force equivalent, with the poses where the viewer shows a contact."""
    plt = plt_setup()
    f = "thumb"
    if f"{f}_Y" not in Z:
        return
    upn = R["uT_per_N"][f]
    fam = Z["family"].astype(str)[Z[f"{f}_rows"]]
    sel = np.flatnonzero(np.char.startswith(fam, "random"))[:220]      # poses where every finger moves
    raw = np.linalg.norm(Z[f"{f}_Y"][sel], axis=1) / upn
    own = np.linalg.norm(Z[f"{f}_Y"][sel] - Z[f"{f}_P_own_angles"][sel], axis=1) / upn
    n = len(sel)
    i = np.arange(n)
    fig, ax = plt.subplots(figsize=(15.5, 4.2))
    ax.plot(i, raw, color="0.55", lw=1.2, label="No compensation")
    ax.plot(i, own, color="k", lw=1.2, label="Own-angle GP, five folds")
    if f"{f}_shown_raw" in Z:
        s = Z[f"{f}_shown_raw"][sel]
        k = np.flatnonzero(np.nan_to_num(s) > 0)
        ax.plot(k, raw[k], "v", color="#b03a2e", ms=6, label="Contact shown by the viewer, uncompensated")
    ax.axhline(0.25, color="0.4", ls="-.", lw=1, label="Viewer's force threshold, 0.25 N")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 60)
    ax.set_xlim(0, n)
    ax.set_xlabel("Random pose, in order of the run")
    ax.set_ylabel("Force equivalent (N)")
    ax.legend(loc="upper right", ncol=2, fontsize=11, frameon=False)
    ax.grid(alpha=0.3, which="both")
    save(fig, "20261105-thumb_trace.png")


def cmd_figs(stem):
    R, Z = load(stem)
    fig_layouts(R, Z)
    fig_cdf(R, Z)
    fig_neighbour(R, Z)
    fig_trace(R, Z)
    return 0


# ------------------------------------------------------------------------------------------------- prose
def n1(v, nd=1):
    return "–" if v is None else f"{v:.{nd}f}"


def stage_rows(R, f):
    return [(k, lab) for k, lab in STAGES if k in R.get("viewer", {}).get(f, {})]


def sec_terms(R):
    upn = R["uT_per_N"]
    terms = [
        ("Board, chip, reading", "five MLX90393 chips, U1 to U5, sit on the ReSkin 5X board under each fingertip. The magnetic "
         "field is measured by each chip along its three axes, in µT. A reading is the 15 components, each divided by its chip's gain "
         "at its die temperature (the UR5 calibration's 0.7 % per °C)."),
        ("Tips and boards", "v3 ID 1 is on the thumb (mux channel 3) and v3 ID 2 on the middle finger (channel 0). v3 ID 3 is "
         "on the index (channel 1), behind a second mux since 11:4x on 11-05. In each pad sit four N52 magnets of 3 by 2 mm on a "
         "5.0 mm square with poles (1, -1, -1, 1)."),
        ("Palm frame $P$, layout", "$x_P$ points from the thumb toward the index and middle pair, $y_P$ from the middle toward "
         "the index. A layout is the three mounts' positions $(x, y)$ in $P$, set by the stepper gantries. Home is thumb "
         "(-80, -55), index (80, 85) and middle (80, -85) mm."),
        ("Zero pose", "every joint at 0 deg: the fingers straight, hanging below the palm."),
        ("Dwell, frame", "a dwell is the last 2.5 s of a pose's hold, and its reading is the mean of its plausible records. A "
         "frame is the mean of 8 consecutive records, 50 ms at 160 records/s. The contact viewer decides on each frame. Records with "
         "|B| on any chip more than 30 % from the board's median, or a component above 2e4 µT, are dropped as misframed."),
        ("Change $d$", "a reading less the zero pose's at the same layout, interpolated in time between the zero dwells "
         "(one every 12 poses). $|d|$ is the norm over the 15 components, in µT."),
        ("Noise", "each zero dwell predicted by linear interpolation between its two neighbours at the same layout; $|d|$ of the "
         "difference."),
        ("Force equivalent", "$|d|$ divided by the tip's change per newton. That is the median $|d|/|f|$ of the tip's UR5 face "
         f"presses at 1 to 3 N: {upn['thumb']:.1f}, {upn['index']:.1f} and {upn['middle']:.1f} µT/N for the thumb's, index's "
         "and middle's tips."),
        ("Own-angle GP", "Gaussian-process regression from the finger's three measured joint angles to its board's change, "
         "$d \\approx \\mu(q_{\\mathrm{own}})$. The kernel is squared-exponential on standardized inputs, and its length and "
         "noise are chosen on a grid by marginal likelihood (<code>ur5_tip_calibration_board_gp.fit</code>). The nine-angle and mount GP "
         "takes all nine angles and the six mount coordinates."),
        ("Five folds, layout left out, home only", "the tests of a GP. Five folds: the poses split at random into five parts, "
         "each predicted by the GP fitted on the other four. Layout left out: every pose of one layout predicted by the GP "
         "fitted on all other layouts. Home only: the poses at every other layout predicted by the GP fitted at home."),
        ("Pad distance", "the distance between the origins of two tips' pad frames in the cad_v2 hand's forward kinematics, at "
         "the measured joints and the layout's mounts. The pad frame is the calibration's: its origin lies on the tip's "
         "midplane, 2.05 mm from the finger's axis toward the face."),
        ("Magnet model", "the field of the other two tips' magnets at the board's five chips. Each magnet is a cylinder of point "
         "dipoles (<code>fingertip_wrench_model.cylinder_quadrature</code>), with each tip's poles and strength from its "
         "calibration of 2026-11-02. It is taken as a change from the layout's zero pose, as $d$ is."),
        ("Viewer's decision", "the contact viewer (<code>wrench_viewer.py</code>, at its defaults) shows a contact on a frame "
         "when its estimator explains the change with a chi-square above 25 and a press of 0.25 N or more. The contact is not "
         "shown when a field from outside the pad explains the change better by more than 25. A contact shown with nothing "
         "touching is a false contact."),
        ("Injected press", "a UR5 press of the same tip, 0.25 to 3 N, whose change is added to a frame of the hand. The force "
         "error is the shown press less the press's normal force, over the normal force. The location error is the distance "
         "along the surface from the readout's contact point to the press's."),
        ("Readout", "the model that turns a board's change into a contact point and force once a contact is shown. The viewer "
         "on the hand uses the board GP, weighted by its $w$ against the physical estimator. A hold is one static press or slide "
         "of the UR5 calibration, and $w$ is 1 less the GP's posterior "
         "variance from its 256 nearest holds. The MLP ensemble (five networks of two hidden layers of 128 units) can take the "
         "GP's place, alone or weighted by the GP's $w$."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in terms) + "</dl>"


def sec_measurement(R, Z):
    lay = Z["layout"].astype(str)
    st = Z["set"].astype(str)
    rows = []
    for s in sorted(set(st)):
        k = st == s
        names = list(dict.fromkeys(lay[k]))
        rows.append((s[:4] + "-" + s[4:6] + "-" + s[6:], len(k.nonzero()[0]), ", ".join(LAYOUT_LABEL.get(n, n) for n in names)))
    out = ("<p>The hand hangs from the UR5's flange with the arm powered off. Each finger's board streams through the C6 "
           "mux at 160 records/s. The lab's runner moves the fingers through the plans in 3 deg waypoints, pauses 3 s at "
           "each pose, and logs the servos' measured angles about 8.6 times a second. A zero pose comes every 12 poses. The "
           "mounts move only between layouts, with the fingers at the zero pose, and each layout opens with a 30 s baseline. "
           "The mounts of every step are the station's stepper counts. From 14:11 on 11-05 the owner looked at each layout's "
           "stepper move before its poses ran. Earlier that day the middle's X stage once skipped a 45 mm move unseen, so that "
           "run (run3_from32) is left out. The middle's mount in run1_attended and run2_layouts is uncertain, and those runs "
           "stay out of the magnet model and the layout table.</p>"
           + table(["Date", "Dwells", "Layouts"], rows, "Data sets and the layouts they cover.", num_from=1))
    out += img(MEDIA / "20261105-layouts.png",
               "Mounts of every layout in the palm frame. Each layout's three mounts are joined in its colour, and the dotted "
               "rectangles are the stages' travel boxes.")
    out += film(MEDIA / "20261105-timelapse.mp4", "The lab's camera on 11-05, one frame every 2 s, played at 20 times real "
                "time.")
    out += film(MEDIA / "20261105-film_cad_sweep.mp4", "The cad_v3 hand at the measured joints of the first random poses of "
                "each layout, and each board's change in force equivalent: uncompensated (grey) and after the own-angle GP "
                "(black). Red text marks a pose whose uncompensated change the contact viewer shows as a contact. The line at "
                "0.25 N is the viewer's force threshold.")
    return out


def sec_noise(R):
    rows = [(k.split()[0][:4] + "-" + k.split()[0][4:6] + "-" + k.split()[0][6:], k.split()[1].capitalize(), v["n"],
             n1(v["median_uT"], 2), n1(v["p95_uT"], 2)) for k, v in sorted(R["noise"].items())]
    return ("<p>At the zero pose the readings drift by 18 to 22 µT over one to two hours (night of 11-04). Every change is "
            "therefore taken against zero dwells a few minutes apart. The noise of that reference is 2.5 to 4.0 µT, about 0.1 N.</p>"
            + table(["Set", "Board", "Zero dwells", "Noise, median (µT)", "Noise, p95 (µT)"], rows,
                    "Noise of the zero-pose reference.", num_from=2))


def sec_own(R):
    ff_ = [R["own"][f]["own_angles"]["five_folds"]["median_uT"] for f in F if f in R["own"]]
    lo_ = [R["own"][f]["own_angles"]["left_out_layout_all"]["median_uT"] for f in F
           if "left_out_layout_all" in R["own"].get(f, {}).get("own_angles", {})] or ff_
    rows = []
    for f in F:
        o = R["own"].get(f)
        if not o:
            continue
        c = o["change_uT"]
        g = o["own_angles"]
        rows.append((f.capitalize(), o["rows"], n1(c["median"]), n1(c["p95"]), n1(g["five_folds"]["median_uT"], 2),
                     n1(g["five_folds"]["p95_uT"], 2), n1(g.get("left_out_layout_all", {}).get("median_uT"), 2),
                     n1(g.get("left_out_layout_all", {}).get("p95_uT"), 2),
                     n1(g.get("home_only_to_layouts", {}).get("median_uT"), 2)))
    out = ("<p>A finger's own pose changes its board by a median of 65 to 67 µT, about 2.3 N in force equivalent. Most of "
           "it is a field that is fixed to the hand, seen from a board that turns with the finger. A uniform field turned with "
           "each tip's frame leaves a median of 12.0 and 13.5 µT of the thumb's and middle's change (night of 11-04, "
           "<code>sr2_crosstalk_night.py</code>). A GP on the finger's three angles leaves "
           + f"{min(ff_):.1f} to {max(ff_):.1f} µT over five folds, and {min(lo_):.1f} to {max(lo_):.1f} µT with each layout left "
           "out of its fit. Poses at which the hand was not still are left out. At these a joint moved more than 2 deg "
           "during the dwell, or a board's reading spread more than 25 µT over it. They are "
           + f"{R['own']['thumb'].get('left_out_moving', 0)} of the thumb's poses, {R['own']['index'].get('left_out_moving', 0)} "
           f"of the index's and {R['own']['middle'].get('left_out_moving', 0)} of the middle's. At one of them, at rv04_mid, the index board's reading stepped by about 180 µT halfway through the "
           "dwell while no joint moved more than 0.9 deg. What changed it is not known.</p>"
           + table(["Board", "Poses", "Change, median (µT)", "Change, p95 (µT)", "Five folds, median (µT)", "Five folds, p95 (µT)",
                    "Layout left out, median (µT)", "Layout left out, p95 (µT)", "Home only, median (µT)"], rows,
                   "Each board's change and what the own-angle GP leaves of it.", num_from=1))
    rows2 = []
    for f in F:
        o = R["own"].get(f)
        if not o:
            continue
        for name, lab in (("own_angles", "Own angles"), ("own_angles_mount", "Own angles and mount"),
                          ("all_angles_mounts", "Nine angles and mounts")):
            v = o[name]["five_folds"]
            rows2.append((f.capitalize(), lab, n1(v["median_uT"], 2), n1(v["p95_uT"], 2), n1(v["median_N"], 3), n1(v["p95_N"], 3)))
    out += table(["Board", "GP inputs", "Median (µT)", "p95 (µT)", "Median (N)", "p95 (N)"], rows2,
                 "What three GPs leave, five folds.", num_from=2)
    out += img(MEDIA / "20261105-residual_cdf.png", "Each board's change over the poses (grey), and what the own-angle GP leaves "
               "of it. The GP is tested by five folds (black), with the layout left out (red, dashed), and fitted at home "
               "only (blue, dotted). The vertical "
               "line is the zero pose's noise.")
    lo = [(f.capitalize(), k.split(" ", 1)[1], v["n"], n1(v["median_uT"], 2), n1(v["p95_uT"], 2))
          for f in F if f in R["own"] for k, v in R["own"][f]["own_angles"].items() if k.startswith("left_out ")]
    if lo:
        out += table(["Board", "Layout left out", "Poses", "Median (µT)", "p95 (µT)"], lo,
                     "The own-angle GP on each layout, fitted on all the others.", num_from=2)
    return out


def sec_layouts(R):
    rows = [(r["set"][:4] + "-" + r["set"][4:6] + "-" + r["set"][6:], r["family"].replace("layout_", ""),
             ", ".join(f"{f} {v:.0f}" for f, v in r["moved_mm"].items() if v > 0.5),
             *(n1(r["change_uT"].get(f)) for f in F)) for r in R["layouts"]]
    if not rows:
        return "<p>No layout moves in these data.</p>"
    ch = [v for r in R["layouts"] for v in r["change_uT"].values()]
    return (f"<p>Moving the mounts changes the boards at the zero pose by {min(ch):.1f} to {max(ch):.1f} µT, little more than "
            "the reference's noise. The change is measured between the zero dwell before a move and the baseline after it, "
            f"25 to 104 s apart. A zero-pose reference taken at one layout is therefore good to within {max(ch):.0f} µT at the others.</p>"
            + table(["Set", "Move", "Mounts moved (mm)", "Thumb (µT)", "Index (µT)", "Middle (µT)"], rows,
                    "Change of each board at the zero pose across each move of the mounts.", num_from=3))


def sec_neighbour(R):
    rows, rows_d = [], []
    for f in F:
        o = R["neighbour"].get(f)
        if not o:
            continue
        for geo, lab in (("driver", "Driver's"), ("cad_v3", "cad_v3")):
            v = o[geo]
            rows.append((f.capitalize(), lab, n1(v["closest_pad_mm"]), o["rows_near"], n1(v["share_nominal"], 2),
                         n1(v["one_gain"], 2), n1(v["share_one_gain"], 2), n1(v["cosine_median"], 2)))
        for b_, v in o["cad_v3"]["by_distance"].items():
            rows_d.append((f.capitalize(), b_, v["n"], n1(v["model_uT_median"]), n1(v["model_uT_max"]), n1(v["left_uT_median"], 1),
                           n1(v["after_model_uT_median"], 1)))
    nb = R["neighbour"]
    g_d = [nb[f]["driver"]["one_gain"] for f in F]
    g_c = [nb[f]["cad_v3"]["one_gain"] for f in F]
    sh = [nb[f][g]["share_one_gain"] for f in F for g in ("driver", "cad_v3")]
    cs = [nb[f][g]["cosine_median"] for f in F for g in ("driver", "cad_v3")]
    near = [nb[f]["rows_near"] for f in F]
    cl = min(nb[f]["driver"]["closest_pad_mm"] for f in F)
    near50 = [nb[f]["driver"]["by_distance"]["35-50 mm"]["left_uT_median"] for f in F if "35-50 mm" in nb[f]["driver"]["by_distance"]]
    far70 = [nb[f]["driver"]["by_distance"]["70-999 mm"]["left_uT_median"] for f in F if "70-999 mm" in nb[f]["driver"]["by_distance"]]
    return (f"<p>Another finger's pad brings four magnets toward the board. The magnet model gives their field at the board's "
            f"chips from the pads' poses, at two finger spacings. On these layouts the pads came no closer than {cl:.0f} mm "
            f"(driver's spacing), and the model gives 5 µT or more on {min(near)} to {max(near)} poses per board. There its "
            f"vector lines up with the leftover at a median cosine of {min(cs):.2f} to {max(cs):.2f}. At the driver's spacing "
            f"the fitted gain is {min(g_d):.2f} to {max(g_d):.2f}: the model's field is two to three times too strong. At "
            f"cad_v3's spacing, with each finger about 6 mm further out, the gain is {min(g_c):.2f} to {max(g_c):.2f}. With its "
            f"gain, either spacing removes {min(sh):.2f} to {max(sh):.2f} of the leftover's energy on those poses. Within 50 mm "
            f"of another pad the own-angle GP leaves a median of {min(near50):.1f} to {max(near50):.1f} µT, against "
            f"{min(far70):.1f} to {max(far70):.1f} µT beyond 70 mm.</p>"
            + table(["Board", "Spacing", "Closest pad (mm)", "Poses near", "Share, nominal", "One gain", "Share, one gain",
                     "Cosine, median"], rows,
                    "The magnet model against what the own-angle GP leaves, on the poses where either spacing's model gives "
                    "5 µT or more (poses near). Share: the part of the leftover's energy removed by the model, nominal or with "
                    "one fitted gain. Cosine: between the model's 15-component vector and the leftover's.", num_from=2)
            + table(["Board", "Pad distance", "Poses", "Magnet model, median (µT)", "Magnet model, max (µT)",
                     "Left by the own-angle GP, median (µT)", "After the magnet model, median (µT)"], rows_d,
                    "The leftover and the magnet model (cad_v3 spacing, one gain) by distance to the nearest other pad.", num_from=2)
            + img(MEDIA / "20261105-neighbour_distance.png", "What the own-angle GP leaves (colour) and the magnet model's "
                  "field at the cad_v3 spacing (black), against the distance to the nearest other pad."))


def sec_viewer(R):
    rows = []
    for f in F:
        for k, lab in stage_rows(R, f):
            v = R["viewer"][f][k]
            rows.append((f.capitalize(), lab, v["poses"], v["frames"],
                         f"{v['frames_shown']} ({100 * (v['frames_shown_share'] or 0):.2f} %)", v["poses_with_a_frame_shown"],
                         n1(v["frame_force_N_median"], 2), v["frames_rejected_as_foreign"]))
    return ("<p>With nothing touching any tip, every frame the viewer shows as a contact is a false contact.</p>"
            + table(["Board", "Compensation", "Poses", "Frames", "Frames shown", "Poses with a frame shown",
                     "Shown press, median (N)", "Frames rejected as a field from outside"], rows,
                    "Contacts shown by the viewer with nothing touching, on the 50 ms frames.", num_from=2)
            + img(MEDIA / "20261105-thumb_trace.png", "The thumb board's change through its first 220 random poses in "
                  "force equivalent, uncompensated (grey) and after the own-angle GP (black). Triangles mark the poses whose dwell "
                  "the viewer shows as a contact uncompensated."))


def sec_presses(R):
    PR_ = R.get("presses_on_the_hand", {})
    if not PR_:
        return "<p>Not run.</p>"
    rd = PR_.get("readout", {})
    frames = (("zero_pose_noise", "Zero-pose noise"), ("raw", "Hand, no compensation"), ("own_angles", "Hand, own-angle GP"))
    reads = (("board_gp", "Board GP (viewer)"), ("mlp", "MLP ensemble"), ("mlp_gp_weight", "MLP ensemble, GP's weight"),
             ("physical", "Physical estimator"))

    def cell(v):
        return f"{n1(v['force_error_abs_median_pct'])} ({n1(v['force_error_abs_p90_pct'])})"
    rows, rows_l = [], []
    for f in F:
        o = PR_.get(f)
        if not o:
            continue
        for k, lab in frames:
            v = o[k]
            rows.append((f.capitalize(), lab, f"{100 * v['detected_share']:.0f} %", *(cell(v[m]) for m, _ in reads)))
            rows_l.append((f.capitalize(), lab, n1(v["board_gp"]["location_error_median_mm"], 2),
                           n1(v["mlp"]["location_error_median_mm"], 2), n1(v["board_gp"]["location_error_p90_mm"], 2),
                           n1(v["mlp"]["location_error_p90_mm"], 2)))
    n_tr = rd.get("training_holds")
    n_gp = rd.get("training_holds_gp")
    return ("<p>UR5 presses of the same tip are added to the hand's frames. Their force and location errors give the "
            "residual's effect on a contact. The "
            "presses come from a fifth of each tip's calibration points, and both readouts were fitted without those points. "
            f"The GP was fitted on {n_gp:,} holds, at most 1,500 of each set, and the MLP ensemble on all {n_tr:,}. The "
            "calibration page compares the two the same way (its Table 33). The physical estimator decides whether a contact is shown, and each readout then gives its press "
            "and point.</p>"
            + table(["Board", "Frames", "Detected", *(lab for _, lab in reads)], rows,
                    "Force error of the presses shown, median of |shown - true| / true in % (p90 in brackets), by readout.",
                    num_from=2)
            + table(["Board", "Frames", "Board GP, median (mm)", "MLP ensemble, median (mm)", "Board GP, p90 (mm)",
                     "MLP ensemble, p90 (mm)"], rows_l,
                    "Location error of the readouts' contact point along the surface, over every injected press.", num_from=2))


def sec_limits():
    items = [
        "Nothing touched the tips in these runs, so contacts are only injected. A press on the hand also moves the finger, "
        "which the injection does not include.",
        "The joint angles are the servos' readings. The SCS0009's readout has 0.29 deg steps.",
        "The pad frames come from the cad_v2 model at the stepper counts. A stage that skipped a move would put a pad "
        "elsewhere. The owner looked at every layout move from 14:11 on 11-05, and the runs before it with an uncertain "
        "mount stay out of the magnet model.",
    ]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def sec_future():
    items = [
        "Run the viewer on the hand with the own-angle GP subtracted live: <code>wrench_viewer.py</code> with a GP of each "
        "finger's three angles from the station's telemetry. Press each tip with a known force at the HEX-E. A false contact "
        "while the fingers move, or a press error above the injected one, falsifies the compensation.",
        "Rerun this analysis with the HOM layouts once the lab has pushed hom1 (<code>sr2_crosstalk_hand.py --sets 20261104 "
        "20261105</code>; the lab's push adds hom1 to <code>data/20261105-day/</code>). The cover and press layouts bring the "
        "pads closer than these did. Where the pads come within 25 mm, a gain of the magnet model near 1 at cad_v3's spacing "
        "supports the CAD's finger positions. A gain near 0.4 at the driver's spacing supports the driver's.",
        "Measure each finger's yaw axis in the palm frame with the UR5 (the cad_v3 page's Future work). It fixes the spacing "
        "the magnet model uses.",
        "Repeat one layout on another day, to measure how far the own-angle GP moves with the boards' drift.",
    ]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def sec_files(stem):
    rows = [("scripts/fingertip/sr2_crosstalk_hand.py", "the analysis"),
            (f"docs/experiments/20261104-sr2_hand_crosstalk/data/{stem}.json", "its results"),
            (f"docs/experiments/20261104-sr2_hand_crosstalk/data/{stem}_rows.npz", "per pose: changes, predictions, distances"),
            ("docs/experiments/20261104-sr2_hand_crosstalk/README.md", "the lab's runbook of both days"),
            ("docs/experiments/20261104-sr2_hand_crosstalk/data/20261104-night/", "the night's board logs, events and telemetry"),
            ("docs/experiments/20261104-sr2_hand_crosstalk/data/20261105-day/", "11-05's board logs, events and telemetry"),
            ("scripts/fingertip/sr2_crosstalk_page.py", "this page and its drawings")]
    return "<ul>" + "".join(f"<li><code>{a}</code>: {b}</li>" for a, b in rows) + "</ul>"


def lede_and_summary(R):
    v = R["viewer"]
    th, mi, ix = v["thumb"]["raw"], v["middle"]["raw"], v["index"]["raw"]
    own = [v[f]["own_angles"]["frames_shown"] for f in F]
    lo = [v[f]["own_angles_left_out_layout"]["frames_shown"] for f in F]
    fr = [v[f]["own_angles"]["frames"] for f in F]
    ch = [R["own"][f]["change_uT"]["median"] for f in F]
    mx = max(R["own"][f]["change_uT"]["max"] for f in F)
    lay = max(x for r in R["layouts"] for x in r["change_uT"].values())
    pr = R["presses_on_the_hand"]
    e_own = [pr[f]["own_angles"][m]["force_error_abs_median_pct"] for f in F for m in ("board_gp", "mlp")]
    e_zero = [pr[f]["zero_pose_noise"][m]["force_error_abs_median_pct"] for f in F for m in ("board_gp", "mlp")]
    e_raw = [pr[f]["raw"][m]["force_error_abs_median_pct"] for f in F for m in ("board_gp", "mlp")]
    n_lay = len({x for x in R["own"]["thumb"]["layouts"]})
    gap = max(abs(pr[f][k]["board_gp"]["force_error_abs_median_pct"] - pr[f][k]["mlp"]["force_error_abs_median_pct"])
              for f in F for k in ("zero_pose_noise", "own_angles"))
    lede = (f"Uncompensated, the contact viewer would show a contact in {100 * th['frames_shown_share']:.1f} % of the thumb's "
            f"50 ms frames with nothing touching, at a median of {th['frame_force_N_median']:.1f} N, over two days and {n_lay} "
            "layouts of the SR2 hand. The change comes mostly from the finger's own pose. A Gaussian-process model predicts each "
            "board's change from its finger's three joint angles. With that prediction subtracted, "
            f"{min(own)} to {max(own)} of {min(fr):,} to {max(fr):,} frames per board still show a false contact, and "
            f"{min(lo)} to {max(lo)} with the layout left out of the model's fit. UR5 presses added to the compensated frames "
            f"read with a median force error of {min(e_own):.0f} to {max(e_own):.0f} %, against {min(e_zero):.0f} to "
            f"{max(e_zero):.0f} % on the zero pose's noise. The two readouts, the board GP and the MLP ensemble, give errors "
            f"within {gap:.0f} percentage points of each other.")
    summary = (f"Uncompensated, the fingertip viewer on the SR2 hand would show a {th['frame_force_N_median']:.1f} N contact in "
               f"{100 * th['frames_shown_share']:.1f} % of the thumb's 50 ms frames with nothing touching. A GP of each board's "
               f"change from its finger's three angles leaves {min(own)} to {max(own)} false frames per board out of "
               f"{min(fr):,} to {max(fr):,} ({min(lo)} to {max(lo)} with the layout left out of its fit), and injected presses read with "
               f"{min(e_own):.0f} to {max(e_own):.0f} % median force error.")
    return lede, summary


def render(page):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = texsvg.render(items, cache_path=TEX_CACHE)
    it = iter(svgs)
    return P.TEX_RE.sub(lambda m: next(it), page), len(items)


def cmd_page(stem):
    R, Z = load(stem)
    sections = [
        ("terms", "Terms and metrics", sec_terms(R)),
        ("measurement", "Measurement on the hand", sec_measurement(R, Z)),
        ("noise", "Noise of the zero-pose reference", sec_noise(R)),
        ("own", "Change of each board with its own finger's pose", sec_own(R)),
        ("layouts", "Change of each board when the mounts move", sec_layouts(R)),
        ("neighbour", "Neighbouring pads and the magnet model", sec_neighbour(R)),
        ("viewer", "False contacts in the contact viewer", sec_viewer(R)),
        ("presses", "Presses added to the hand's frames", sec_presses(R)),
        ("limits", "Limitations", sec_limits()),
        ("future", "Future work", sec_future()),
        ("files", "Files", sec_files(stem)),
    ]
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|µT|N\b|ms\b|s\b)", "\\1&nbsp;\\2", s_))
    body = [(sid, hd, unit(b)) for sid, hd, b in sections]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(body, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{b}</section>' for k, (sid, hd, b) in enumerate(body, 1))
    lede, summary = lede_and_summary(R)
    byline = ("Measured by the UR5 lab session on the hand at the UR5; analysis by the coordinating session. Builder: "
              "<code>scripts/fingertip/sr2_crosstalk_page.py</code>.")
    page = TEMPLATE.read_text()
    for k_, v in {"TITLE": "SR2 fingertip crosstalk", "H1": "Magnetic crosstalk between the fingertip boards on the SR2 hand",
                  "EYEBROW": "Fingertip sensing on the hand, 2026-11-04 and 11-05", "LEDE": unit(lede), "BYLINE": byline,
                  "TOC": toc, "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k_ + "}}", v)
    page, n = render(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary, date="2026-11-05")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT}: {OUT.stat().st_size / 1e6:.2f} MB, {n} formulas", flush=True)
    return 0


def main():
    a = sys.argv[1:]
    if a and a[0] == "figs":
        return cmd_figs(a[1] if len(a) > 1 else STEM)
    return cmd_page(a[0] if a else STEM)


if __name__ == "__main__":
    sys.exit(main())
