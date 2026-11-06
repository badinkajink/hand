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
          ("own_angles_and_magnets", "Own-angle GP and magnet model"))


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
    """The mounts of every layout in the palm frame, one panel."""
    plt = plt_setup()
    lay, M = Z["layout"].astype(str), Z["mounts"]
    seen = {}
    for name, m in zip(lay, M):
        seen.setdefault(name, m)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    for f, x0, y0, w, h in (("thumb", -80, -55, 60, 110), ("index", 20, 25, 60, 60), ("middle", 20, -85, 60, 60)):
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ls=":", ec=COLOR[f], lw=1.2))
    for name, m in seen.items():
        xy = m.reshape(3, 2)
        ax.plot(np.r_[xy[:, 0], xy[0, 0]], np.r_[xy[:, 1], xy[0, 1]], color="0.75", lw=0.8, zorder=1)
        for k, f in enumerate(F):
            ax.plot(*xy[k], "o", color=COLOR[f], ms=6, zorder=2)
        ax.annotate(LAYOUT_LABEL.get(name, name), xy[0], textcoords="offset points", xytext=(-6, 6), ha="right",
                    fontsize=11, color="0.25")
    for f in F:
        ax.plot([], [], "o", color=COLOR[f], label=f"{f.capitalize()} (v3 ID {TIP_ID[f]})")
    ax.set_xlabel("$x_P$ (mm)")
    ax.set_ylabel("$y_P$ (mm)")
    ax.set_aspect("equal")
    ax.legend(loc="center", frameon=False)
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
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
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
            ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=c, ls=ls, lw=1.8, label=lab)
        nz = [v["median_uT"] for k, v in R["noise"].items() if k.endswith(f) and v["median_uT"]]
        if nz:
            ax.axvline(float(np.median(nz)), color="0.4", lw=1, ls="-.", label="Zero-pose noise, median")
        ax.set_xscale("log")
        ax.set_xlim(0.8, 300)
        ax.set_xlabel("Change or residual, |d| (µT)")
        sec = ax.secondary_xaxis("top", functions=(lambda x, u=upn: x / u, lambda x, u=upn: x * u))
        sec.set_xlabel(f"{f.capitalize()} board (v3 ID {TIP_ID[f]}): force equivalent (N)")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("Share of poses")
    axs[0].legend(loc="upper left", fontsize=11, frameon=False)
    save(fig, "20261105-residual_cdf.png")


def fig_neighbour(R, Z):
    """What the own-angle GP leaves against the distance to the nearest other pad, with the magnet model's field."""
    plt = plt_setup()
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)
    for ax, f in zip(axs, F):
        if f"{f}_dist" not in Z:
            ax.set_visible(False)
            continue
        d = Z[f"{f}_dist"]
        e = np.linalg.norm(Z[f"{f}_Y"] - Z[f"{f}_P_own_angles"], axis=1)
        m = np.linalg.norm(Z[f"{f}_M"].sum(1), axis=1)
        ax.plot(d, e, ".", color=COLOR[f], ms=4, alpha=0.6, label="Left by the own-angle GP")
        ax.plot(d, m, ".", color="0.2", ms=3, alpha=0.5, label="Magnet model, nominal")
        ax.set_yscale("log")
        ax.set_ylim(0.1, 300)
        ax.set_title(f"{f.capitalize()} board")
        ax.set_xlabel("Distance to the nearest other pad (mm)")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("|d| (µT)")
    axs[0].legend(loc="upper right", fontsize=11, frameon=False)
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
        ("Board, chip, reading", "the ReSkin 5X board under each fingertip carries five MLX90393 chips, U1 to U5. Each chip "
         "gives the magnetic field along its three axes in µT. A reading is the 15 components, each divided by its chip's gain "
         "at its die temperature (the UR5 calibration's 0.7 % per °C)."),
        ("Tips and boards", "the thumb carries v3 ID 1 (mux channel 3), the index v3 ID 3 (channel 1; through a second mux "
         "since 11:4x on 11-05) and the middle finger v3 ID 2 (channel 0). Each pad holds four N52 magnets of 3 by 2 mm on a "
         "5.0 mm square with poles (1, -1, -1, 1)."),
        ("Palm frame $P$, layout", "$x_P$ points from the thumb toward the index and middle pair, $y_P$ from the middle toward "
         "the index. A layout is the three mounts' positions $(x, y)$ in $P$, set by the stepper gantries. Home is thumb "
         "(-80, -55), index (80, 85) and middle (80, -85) mm."),
        ("Zero pose", "every joint at 0 deg: the fingers straight, hanging below the palm."),
        ("Dwell, frame", "a dwell is the last 2.5 s of a pose's hold, and its reading is the mean of its plausible records. A "
         "frame is the mean of 8 consecutive records, 50 ms at 160 records/s: what the contact viewer reads. Records with "
         "|B| on any chip more than 30 % from the board's median, or a component above 2e4 µT, are dropped as misframed."),
        ("Change $d$", "a reading less the zero pose's at the same layout, interpolated in time between the zero dwells "
         "(one every 12 poses). $|d|$ is the norm over the 15 components, in µT."),
        ("Noise", "each zero dwell predicted by linear interpolation between its two neighbours at the same layout; $|d|$ of the "
         "difference."),
        ("Force equivalent", f"$|d|$ divided by the tip's change per newton: {upn['thumb']:.1f}, {upn['index']:.1f} and "
         f"{upn['middle']:.1f} µT/N for the thumb's, index's and middle's tips, the median $|d|/|f|$ of each tip's UR5 face "
         "presses at 1 to 3 N."),
        ("Own-angle GP", "Gaussian-process regression from the finger's three measured joint angles to its board's change, "
         "$d \\approx \\mu(q_{\\mathrm{own}})$, with a squared-exponential kernel on standardized inputs and the length and noise "
         "chosen on a grid by marginal likelihood (<code>ur5_tip_calibration_board_gp.fit</code>). The nine-angle and mount GP "
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
        ("Viewer's decision", "the contact viewer (<code>wrench_viewer.py</code>, at its defaults) shows a contact when its "
         "estimator explains the frame's change with a chi-square above 25, at a press of 0.25 N or more, and when a field from "
         "outside the pad does not explain it better by more than 25. A contact shown with nothing touching is a false contact."),
        ("Injected press", "a UR5 press of the same tip, 0.25 to 3 N, whose change is added to a frame of the hand. The force "
         "error is the shown press less the press's normal force, over the normal force. The viewer on the hand shows the "
         "press of its board GP (the rig-pooled GP of 2026-11-05, which was trained on these presses)."),
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
           "mux at 160 records/s, and the station logs the servos at 20 Hz. The lab's runner moves the fingers through the "
           "plans in 3 deg waypoints and holds each pose for 3 s. A zero pose comes every 12 poses. The mounts move only between "
           "layouts, with the fingers at the zero pose, and each layout opens with a 30 s baseline. On 11-05 the owner "
           "confirmed each layout's stepper move before its poses ran.</p>"
           + table(["Date", "Dwells", "Layouts"], rows, "Data sets and the layouts they cover.", num_from=1))
    out += img(MEDIA / "20261105-layouts.png",
               "Mounts of every layout in the palm frame. Each layout's three mounts are joined by a grey triangle; the dotted "
               "rectangles are the stages' travel boxes.")
    out += film(MEDIA / "20261105-timelapse.mp4", "The lab's camera on 11-05, one frame every 2 s, played at 20 times real "
                "time.")
    return out


def sec_noise(R):
    rows = [(k.split()[0][:4] + "-" + k.split()[0][4:6] + "-" + k.split()[0][6:], k.split()[1].capitalize(), v["n"],
             n1(v["median_uT"], 2), n1(v["p95_uT"], 2)) for k, v in sorted(R["noise"].items())]
    return ("<p>The zero-pose readings drift by tens of µT over an hour, so every change is taken against zero dwells "
            "minutes apart. The noise of that reference is about 3 µT, 0.1 N.</p>"
            + table(["Set", "Board", "Zero dwells", "Noise, median (µT)", "Noise, p95 (µT)"], rows,
                    "Noise of the zero-pose reference.", num_from=2))


def sec_own(R):
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
    out = ("<p>A finger's own pose changes its board by tens of µT. The board turns in the Earth's field and in the fields of "
           "the hand's servos and steppers, and the change follows the finger's three angles.</p>"
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
    out += img(MEDIA / "20261105-residual_cdf.png", "Each board's change over the poses (grey) and what the own-angle GP leaves "
               "of it: five folds (black), the layout left out (red, dashed) and fitted at home only (blue, dotted). The vertical "
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
    return ("<p>Moving the mounts changes every board, including the boards whose finger stays. The change is measured "
            "between the zero dwell before a move and the baseline after it, about a minute apart.</p>"
            + table(["Set", "Move", "Mounts moved (mm)", "Thumb (µT)", "Index (µT)", "Middle (µT)"], rows,
                    "Change of each board at the zero pose across each move of the mounts.", num_from=3))


def sec_neighbour(R):
    rows = []
    for f in F:
        o = R["neighbour"].get(f)
        if not o:
            continue
        for b, v in o["by_distance"].items():
            rows.append((f.capitalize(), b, v["n"], n1(v["model_uT_median"]), n1(v["model_uT_max"]), n1(v["left_uT_median"], 2),
                         n1(v["after_model_uT_median"], 2)))
    sh = "; ".join(f"{f} {R['neighbour'][f]['explained_share_fitted']:.2f}" for f in F if f in R["neighbour"])
    return ("<p>Another finger's pad brings four magnets near the board. What the own-angle GP leaves is set against the "
            f"distance to the nearest other pad and against the magnet model. The share of the leftover's energy that the "
            f"magnet model explains, with a gain per source tip, is {sh}.</p>"
            + table(["Board", "Pad distance", "Poses", "Magnet model, median (µT)", "Magnet model, max (µT)",
                     "Left by the own-angle GP, median (µT)", "After the magnet model, median (µT)"], rows,
                    "The leftover and the magnet model by distance to the nearest other pad.", num_from=2)
            + img(MEDIA / "20261105-neighbour_distance.png", "What the own-angle GP leaves (colour) and the magnet model's "
                  "field (black), against the distance to the nearest other pad."))


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
    rows = []
    for f in F:
        o = R.get("presses_on_the_hand", {}).get(f)
        if not o:
            continue
        for k, lab in (("zero_pose_noise", "Zero-pose noise"), ("raw", "Hand, no compensation"),
                       ("own_angles", "Hand, own-angle GP")):
            v = o[k]
            rows.append((f.capitalize(), lab, v["presses"], f"{100 * v['detected_share']:.0f} %",
                         n1(v["board_gp"]["force_error_median_pct"]), n1(v["board_gp"]["force_error_abs_median_pct"]),
                         n1(v["board_gp"]["force_error_abs_p90_pct"]), n1(v["physical"]["force_error_abs_median_pct"])))
    return ("<p>Real presses are added to the hand's frames to see what the residual does to a contact.</p>"
            + table(["Board", "Frames", "Presses", "Detected", "Force error, median (%)", "Force error, median |.| (%)",
                     "Force error, p90 |.| (%)", "Physical estimator, median |.| (%)"], rows,
                    "UR5 presses of 0.25 to 3 N added to frames of the zero pose and of the moving hand.", num_from=2))


def sec_limits():
    items = [
        "Nothing touched the tips in these runs, so contacts are only injected. A press on the hand also moves the finger, "
        "which the injection does not include.",
        "The board GP of the viewer was fitted on the same UR5 presses that are injected, so its force errors here are those "
        "of its training data.",
        "The joint angles are the servos' readings. The SCS0009's readout has 0.29 deg steps.",
        "The pad frames come from the cad_v2 model at the commanded mounts. A stepper that skipped a move would put a pad "
        "elsewhere; the owner checked every move on 11-05.",
    ]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def sec_future():
    items = [
        "Run the viewer on the hand with the own-angle GP subtracted live: <code>wrench_viewer.py</code> with a GP of the "
        "finger's three angles from the station's telemetry. Press each tip with a known force on the HEX-E's tool. A false "
        "contact at rest, or a press error above the injected one, falsifies the compensation.",
        "Bring the pads within 20 mm of each other in a plan of their own, the thumb against the index and the middle. The "
        "magnet model predicts the neighbour's field there; a gain far from 1 or a leftover above 10 µT says the model's pad "
        "frames or strengths are off.",
        "Repeat a layout the next day, to measure how far the own-angle GP moves with the drift of the boards' rest.",
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
    v = R.get("viewer", {})
    th = v.get("thumb", {}).get("raw", {})
    own = [v[f]["own_angles"]["frames_shown"] for f in F if f in v]
    lo = [v[f].get("own_angles_left_out_layout", {}).get("frames_shown") for f in F if f in v]
    pr = R.get("presses_on_the_hand", {})
    e_own = [pr[f]["own_angles"]["board_gp"]["force_error_abs_median_pct"] for f in F if f in pr]
    e_raw = [pr[f]["raw"]["board_gp"]["force_error_abs_median_pct"] for f in F if f in pr]
    lede = (f"A finger's own pose changes its fingertip board by up to {max(R['own'][f]['change_uT']['max'] for f in R['own']):.0f} "
            f"µT, several newtons in force equivalent. The other fingers change it by a few µT. Uncompensated, the contact "
            f"viewer would show a contact in {100 * th.get('frames_shown_share', 0):.1f} % of the thumb's 50 ms frames with "
            f"nothing touching, at a median of {n1(th.get('frame_force_N_median'), 1)} N. After a Gaussian-process model "
            f"of the board's change from its finger's three joint angles, it shows {sum(own)} such frames"
            + (f", and {sum(x or 0 for x in lo)} when the layout is left out of the fit" if any(x is not None for x in lo) else "")
            + f". UR5 presses added to the hand's frames then read with a median force error of {min(e_own):.0f} to "
            f"{max(e_own):.0f} %, against {min(e_raw):.0f} to {max(e_raw):.0f} % uncompensated.")
    summary = lede
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
