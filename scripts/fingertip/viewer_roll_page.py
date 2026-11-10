#!/usr/bin/env python3
"""Builds docs/experiments/20261110-fingertip_viewer_roll/20261110-fingertip_viewer_on_a_ur5_roll.html: the fingertip
viewer replayed on a UR5 rod roll beside the roll, with the physical model and the GP of the UR5 calibration, and the FE
and field model of the same roll (owner, 2026-11-10: "a video of the magnetic fingertip viewer in sync with our snapshots
of a roll, to demonstrate the effect of the calibration. ideally updated FEM and magnetic field simulation plots/videos
if we dont arleady have them").

Data in the page's data/ folder (scripts/fingertip/viewer_roll_film.py prepare, frames, table, compose and
scripts/fingertip/fingertip_roll_fe_film.py solve, film); films in media/.

    .venv/bin/python scripts/fingertip/viewer_roll_page.py
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

D = ROOT / "docs/experiments/20261110-fingertip_viewer_roll"
DATA, MEDIA = D / "data", D / "media"
OUT = D / "20261110-fingertip_viewer_on_a_ur5_roll.html"
STEM = "20261110-v3id5_rod_roll"
TEX_CACHE = D / "texsvg_cache.json"
HOMREV = "../20261029-hom_control_review/20261029-hom_control_and_fingertip_sensing.html"
PHYS = "../20261026-fingertip_physical_model/20261026-fingertip_physical_model.html"
CAL = "../20261026-fingertip_calibration/20261026-fingertip_calibration_on_the_ur5.html"
TIPS = "../../overviews/printed_fingertips.html"
EST = ("physical_stored_frames", "physical", "gp")
ENAME = {"physical_stored_frames": "Physical model, the board's stored chip frames (the viewer's rest fit)",
         "physical": "Physical model, full rest calibration of this board",
         "gp": "GP of the UR5 calibration"}
COL = {"physical_stored_frames": "#e39b8f", "physical": "#c0392b", "gp": "#1f4fd8"}
table, esc = P.table, P.esc


def load():
    F = json.loads((DATA / f"{STEM}_frames.json").read_text())
    FE = json.loads((DATA / f"{STEM}_fe_summary.json").read_text())
    GP = json.loads((DATA / "20261110-board_gp_v3id5_no_rolls.json").read_text())
    FS = json.loads((DATA / f"{STEM}_film_summary.json").read_text())
    return F, FE, GP, FS


def stats(F):
    rows = F["rows"]
    out = {}
    hexe = np.array([np.linalg.norm(r["f"]) for r in rows])
    for k in EST:
        sh = np.array([r[k]["contact"] for r in rows])
        d = np.array([np.linalg.norm(np.subtract(r[k]["p"], r["p"])) for r in rows if r[k]["contact"]])
        fn = np.array([r[k]["fn"] for r in rows if r[k]["contact"]])
        face = sum(1 for r in rows if r[k]["contact"] and r[k]["p"][0] > 5.0)
        out[k] = dict(shown=int(sh.sum()), med=float(np.median(d)), p90=float(np.percentile(d, 90)), mx=float(d.max()),
                      within1=float(np.mean(d <= 1.0)), fn=float(np.median(fn)), face=face)
    none = [r for r in rows if not r["physical"]["contact"]]
    out["n"] = len(rows)
    out["hexe"] = float(np.median(hexe))
    out["hexe_lo"], out["hexe_hi"] = float(hexe.min()), float(hexe.max())
    out["none"] = len(none)
    fz = np.array([np.linalg.norm(r["f"]) for r in none]) if none else np.zeros(1)
    out["none_f_lo"], out["none_f_hi"] = float(fz.min()), float(fz.max())
    tz = np.array([r["p"][2] for r in rows])
    out["travel"] = float(tz.max() - tz.min())
    out["t0"], out["t1"] = rows[0]["t"], rows[-1]["t"]
    return out


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.labelsize": 13, "legend.fontsize": 12,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "mathtext.fontset": "dejavuserif"})
    return plt


def fig_distance(F):
    """The distance of each estimate from the robot's contact over the roll, and the turn."""
    plt = plt_setup()
    rows = F["rows"]
    t = np.array([r["t"] for r in rows])
    fig = plt.figure(figsize=(9.0, 5.2))
    ax = fig.add_axes([0.09, 0.33, 0.88, 0.60])
    for k in EST:
        d = np.array([np.linalg.norm(np.subtract(r[k]["p"], r["p"])) if r[k]["contact"] else np.nan for r in rows])
        ax.plot(t, d, ".", ms=4 if k == "gp" else 3, color=COL[k], label=ENAME[k])
    ax.set_yscale("log")
    ax.set_ylim(0.1, 30)
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    ax.set_ylabel("Distance from the\nrobot's contact (mm)")
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", frameon=False, fontsize=11, ncol=1, bbox_to_anchor=(0.0, 1.13))
    ax.grid(True, which="major", axis="y", color="0.9")
    a2 = fig.add_axes([0.09, 0.11, 0.88, 0.17], sharex=ax)
    a2.plot(t, [r["theta"] for r in rows], color="0.4", lw=1.5)
    a2.set_ylim(-8, 8)
    a2.set_yticks([-6, 0, 6])
    a2.set_ylabel("Turn (°)")
    a2.set_xlabel("Time from the press's level hold (s)")
    for a in (ax, a2):
        a.spines[["top", "right"]].set_visible(False)
    path = MEDIA / f"{STEM}_distance.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def fig_holds(FE):
    """Measured and modelled change at the 15 chip axes at three holds of the roll."""
    plt = plt_setup()
    H = FE["holds_U_order"]
    labs = [f"U{u}{c}" for u in range(1, 6) for c in "xyz"]
    fig, axs = plt.subplots(3, 1, figsize=(9.0, 7.6), sharex=True)
    xb = np.arange(15)
    lim = max(max(abs(v) for h in H.values() for v in h["measured"] + h["modelled"]) * 1.1, 40)
    for ax, (name, h) in zip(axs, H.items()):
        ax.bar(xb - 0.2, h["measured"], 0.4, color="#111827", label="measured by the board")
        ax.bar(xb + 0.2, h["modelled"], 0.4, color="#e8590c", label="FE and field model")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylim(-lim, lim)
        me, mo = np.array(h["measured"]), np.array(h["modelled"])
        r = float(np.corrcoef(me, mo)[0, 1])
        ax.set_title(f"Hold at {name.replace('press, 0 deg', 'the press, 0°').replace(' deg', '°')}: correlation "
                     f"{r:.2f}, model/measured {np.linalg.norm(mo) / np.linalg.norm(me):.2f}", fontsize=13, loc="left")
        ax.set_ylabel("Change (µT)")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(loc="upper right", frameon=False, ncol=2, fontsize=11.5)
    axs[-1].set_xticks(xb)
    axs[-1].set_xticklabels(labs, rotation=90, fontsize=11)
    axs[-1].set_xlabel("Chip and axis (chip frames)")
    fig.tight_layout()
    path = MEDIA / f"{STEM}_fe_holds.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def sec_terms(GP):
    return (
        "<p><b>Fingertip.</b> v3 ID 5, a printed TPU pad of layout 05R4 on DIP link b v4. Its four N52 magnets of 3 by "
        "2 mm sit in a 5.0 mm square over the ReSkin 5X board. The board has five three-axis MLX90393 chips, with U1 at "
        "the centre. <b>Reading</b>: the 15 "
        "components of the five chips, µT. <b>Board change</b> $\\Delta b$: the reading less the rest reading, corrected "
        "for each chip's die temperature.</p>"
        "<p><b>Pad frame</b> P, mm: x along the face's normal toward the palm, y across the finger, z along it toward the "
        "tip end. The same frame is used by the FE model and the viewer.</p>"
        "<p><b>Robot's contact point</b> $p$: the point of the fitted pad surface nearest the rod's axis, at every robot "
        "sample (125 Hz). It is computed from the flange pose, the HEX-E's deflection under its measured wrench and the "
        "rod's registration (<code>scripts/hom/hom_filter_real_check.py</code>). <b>HEX-E force</b> $f$: the force of "
        "the six-axis sensor on the robot, turned into the pad frame. <b>Turn</b> $\\theta$: the flange's rotation from "
        "the pressed pose about the roll axis.</p>"
        "<p><b>Viewer</b>: <code>wrench_viewer.py</code>, the fingertip viewer of the bench. Every 1/30 s it estimates "
        "the contact from the mean board change of the last 50 ms. A contact is shown when the physical model explains "
        "more than 25 in chi-square and presses at least 0.25 N; that decision is the physical model's in every row.</p>"
        "<p><b>Physical model</b>: the FE surface basis of pad 05R4 and the magnets' dipole field, with poles, magnet "
        "strength and the board's offsets fitted to the rest reading. The estimator searches the free surface for the "
        "point and force that best explain $\\Delta b$. In the viewer's own rest fit the board's stored chip frames "
        "are kept. In the <b>full rest calibration</b> the chip frames of this board are fitted as well. Its residual "
        "at rest is 0.373 mT, against 0.447 mT.</p>"
        f"<p><b>GP of the UR5 calibration</b>: a Gaussian process from $\\Delta b$ to the contact point and force. It "
        f"was fitted to {GP['n_holds']:,} holds of v3 ID 5 on the inverted rig: presses, sweeps, offsets and slides of "
        f"its three sessions. The {GP['roll_action_holds_left_out']} holds of roll actions were left out. In the viewer "
        "its estimate is weighted by one minus its posterior variance against the physical model's; the weight was "
        "0.97 to 0.99 here.</p>"
        "<p><b>Distance from the robot's contact</b> $\\lVert \\hat{p} - p \\rVert$, mm, over the frames with a contact "
        "shown; $\\hat{p}$ is the estimate's point. <b>Press</b>: the estimate's force along the inward normal, N.</p>"
        "<p><b>Pattern correlation</b> $r$: the Pearson correlation of the 15 modelled and the 15 measured components "
        "of $\\Delta b$. <b>Size ratio</b>: the norm of the modelled change over the norm of the measured one.</p>")


def sec_roll(S):
    return (
        "<p>The roll ran at 04:32 on 4 November, in the night of rod calibration on the inverted rig (HOM review page, "
        f"<a href=\"{HOMREV}\">Figure 8</a>). v3 ID 5 sat pad up in seat 2. The UR5 pressed a rod of radius 12.5 mm on "
        "the face to 4 N, with the rod across the finger. It then turned the rod about its contact line to +6°, to "
        "&minus;6° and back to 0°. The turn ran at about 0.4°/s, with a 0.4 s hold every 3°. The robot's contact moved "
        f"{S['travel']:.1f} mm along the finger. The HEX-E's force was {S['hexe_lo']:.1f} to {S['hexe_hi']:.1f} N.</p>"
        "<p>The cell's overhead camera was off that night, so no picture of the roll exists. The films draw the roll "
        "from the robot's record instead: the rod's pose in the pad frame at every robot sample. The board recorded "
        "152 readings per second through the session.</p>")


def sec_viewer(S, fig1):
    p, q, g = S["physical_stored_frames"], S["physical"], S["gp"]
    body = [[ENAME[k], f"{S[k]['shown']}", f"{S[k]['med']:.2f}", f"{S[k]['p90']:.2f}", f"{S[k]['mx']:.2f}",
             f"{100 * S[k]['within1']:.{1 if 0 < S[k]['within1'] < 1 else 0}f}", f"{S[k]['fn']:.2f}"] for k in EST]
    body.append(["HEX-E", "", "", "", "", "", f"{S['hexe']:.2f}"])
    tab = table(["Estimate", f"Frames shown, of {S['n']}", "Median distance (mm)", "90th percentile (mm)",
                 "Largest (mm)", "Within 1 mm (%)", "Median press (N)"], body,
                "Contact estimates of the viewer during the roll against the robot's contact point and the HEX-E. "
                "Distances over the frames with a contact shown.")
    return (
        P.film(MEDIA / f"{STEM}_viewer_film.mp4",
               "The fingertip viewer (right) replayed on the roll at 4x speed. One estimate is shown per row: the "
               "physical model fitted to the rest reading (top) and the GP of the UR5 calibration (bottom). Green marks "
               "the robot's contact point and the HEX-E's force. On the left, (a) is the roll drawn from the robot's "
               "record, as a section across the rod. Panel (b) gives the three contact points seen from the palm, and "
               "(c) each estimate's distance from the robot's contact. In the top row the physical model has the full "
               "rest calibration of this board.")
        + f"<p>With the GP of the UR5 calibration, the shown contact stayed within {g['mx']:.2f} mm of the robot's "
        f"contact point. Its median distance was {g['med']:.2f} mm. Its press was {g['fn']:.2f} N, against "
        f"{S['hexe']:.2f} N from the HEX-E. With the physical model the contact jumped across the face, onto the &minus;y side "
        f"wall and to the mount end. Its median distance was {q['med']:.2f} mm with the full rest calibration and "
        f"{p['med']:.2f} mm with the viewer's own rest fit. Its press was {q['fn']:.2f} and {p['fn']:.2f} N (Table 1, "
        "Figure 1).</p>"
        f"<p>No contact was shown in {S['none']} of the {S['n']} frames, while the rod pressed "
        f"{S['none_f_lo']:.1f} to {S['none_f_hi']:.1f} N. In those frames less than 25 in chi-square was explained by "
        "the physical model. No contact is shown in the GP's row in the same frames.</p>"
        + tab
        + P.img(fig1, "Distance of each estimate from the robot's contact point over the roll, on a log scale, and the "
                      "rod's turn. One point per film frame with a contact shown, every 0.16 s of the recording."))


def sec_fe(FE, fig2):
    fm = FE["film"]
    tr = FE["magnet_travel_um"]["median"]
    return (
        "<p>At every 0.5 s of the roll the rod was set on the FE model of pad 05R4 at its recorded pose. It is the model "
        f"of the estimator's surface basis, with {FE['n_dof']:,} degrees of freedom. Its walls and skins are solid 85A "
        "TPU, its core a homogenized 30 % gyroid, and its four magnets are bonded in. The cut face and the mount face "
        "are fixed. The "
        "rod was pressed along &minus;x until the frictionless contact on the face's nodes gave the HEX-E's force along "
        "&minus;x. The HEX-E's whole force was then applied on the contact nodes in proportion to their pressure. The "
        "field model, with the full rest calibration of v3 ID 5, turns each magnet's motion into the change at the five "
        "chips.</p>"
        + P.film(MEDIA / f"{STEM}_fe_field_film.mp4",
                 "Finite elements and magnetic field of the same roll, at 4x speed. (a) The section along the finger "
                 "through the contact, coloured by displacement. The deformed face is drawn at true scale and the "
                 "magnets' motion 100x. (b) The contact's nodal forces, the magnets' motion and the chips' change, seen "
                 "from the palm. (c) The 15 components of the board change, measured and modelled.")
        + f"<p>The face sank a median {FE['approach_um']['median']:.0f} µm under the rod, at most "
        f"{FE['approach_um']['max']:.0f} µm. The four magnets moved {min(tr):.0f} to {max(tr):.0f} µm, by their "
        "medians over the roll, mostly toward the board.</p>"
        f"<p>The modelled change at the chips had the pattern of the measured change. Their correlation was a median "
        f"{fm['correlation_median']:.2f} over the {fm['frames']} frames, from {fm['correlation_range'][0]:.2f} to "
        f"{fm['correlation_range'][1]:.2f}. The modelled change was {fm['size_ratio_median']:.2f} times larger, "
        f"{fm['modelled_norm_uT_median']:.0f} against {fm['measured_norm_uT_median']:.0f} µT over the 15 components. "
        "The largest difference was on U1's y axis, the chip under the face centre (Figure 2). There the modelled "
        "change was 28 to 61 µT at the three holds, and the measured change &minus;3 to 22 µT. The measured change per "
        f"newton was {fm['measured_norm_uT_median'] / 4.06:.1f} µT/N. On the <a href=\"{PHYS}\">physical-model "
        "page</a>, 12.6 to 14.0 µT/N were measured at the face centre of three four-magnet tips, against 16.4 to "
        "18.4 µT/N in that page's model.</p>"
        + P.img(fig2, "Board change at the 15 chip axes at three holds of the roll: the press's level hold at 0°, and "
                      "the holds at +6° and &minus;6°. Measured by the board (black) and given by the FE and field model "
                      "for the robot's contact point and the HEX-E's force (orange)."))


PART_TEXT = {   # the synced films, in the order of the table
    "ur5_views": "The UR5 replay: the cell, the probe over seat 2 and the rod along its axis, side by side",
    "ur5_cell": "The UR5 replay: the cell",
    "ur5_rig": "The UR5 replay: the probe and the rod over seat 2",
    "ur5_rod_axis": "The UR5 replay: the rod on v3 ID 5, seen along the rod's axis",
    "viewer_both": "The fingertip viewer, both estimates (Film 1's right side)",
    "viewer_physical": "The fingertip viewer, the physical model's row",
    "viewer_gp": "The fingertip viewer, the GP's row",
    "roll_section": "The roll drawn from the robot's record, a section across the rod",
    "palm_points": "The three contact points seen from the palm, with the distances",
    "distance": "Each estimate's distance from the robot's contact, and the turn",
    "fe_section": "FE section through the contact, magnets' motion and chips' change",
    "fe_palm": "Magnets' motion and chips' change seen from the palm",
    "fe_bars": "The 15 components of the board change, measured and modelled",
}
PARTS = MEDIA / "20261110-synced_parts"


def video_size(path):
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=width,height,nb_read_frames", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip().split(",")
    return int(out[0]), int(out[1]), int(out[2])


def sec_parts():
    rows, frames = [], set()
    for k, txt in PART_TEXT.items():
        f = PARTS / f"{STEM}_{k}.mp4"
        if not f.exists():
            continue
        w, h, n = video_size(f)
        frames.add(n)
        rows.append([f"<code>{f.name}</code>", txt, f"{w} by {h}", f"{f.stat().st_size / 1e6:.1f}"])
    if not rows:
        return ""
    n = frames.pop() if len(frames) == 1 else None
    tab = table(["File", "Shows", "Pixels", "MB"], rows, "The synced films in <code>media/20261110-synced_parts/</code>, "
                f"each with {n} frames at 25 frames per second." if n else "The synced films.", num_from=3)
    return (
        "<p>No camera filmed the roll. Film 3 therefore replays the UR5's own record in a model of the cell: the "
        "robot's recorded joint angles drive the URDF's link meshes, and the HEX-E, mount v7's adapter, the probe "
        "base and the rod tool are mounted on the flange. The seat block, its 2x4 and the three held tips stand at the "
        "session's registered poses. v3 ID 5's tip is placed by the joint rod registration, as in the analysis. "
        "The three views are rendered at 1280 by 720 pixels.</p>"
        + P.film(PARTS / f"{STEM}_ur5_views.mp4",
                 "The UR5's record of the roll replayed in the cell at 4x speed: the cell (left), the probe and the rod "
                 "over seat 2 (top right), and the rod on v3 ID 5 seen along the rod's axis (bottom right).")
        + "<p>Real footage of this rig exists only as the overhead D435's time-lapse: one frame every 1.94 s at 640 by "
        "360 pixels. The camera was off during the night's rolls. Film 4 shows the same rod on the same tip the evening "
        "before, from 18:52 to 19:24 on 3 November, during its presses and slides; it is not in step with the other "
        "films.</p>"
        + P.film(MEDIA / f"20261110-v3id5_rod_real_timelapse.mp4",
                 "The real UR5 pressing and sliding the rod on v3 ID 5 at seat 2, 3 November 18:52 to 19:24, from the "
                 "lab's overhead time-lapse (<code>lab/20261103-tipcal-timelapse_60fps.mp4</code>), at 47x speed.")
        + "<p>Every panel of Films 1 to 3 is also a film of its own, for slides (Table 2). All of them start at the "
        "press's level hold and run at 4x speed. Frame k of each shows the recording 0.16 k s after the level hold, "
        "and the last frame is held for 1.5 s. Films placed side by side with a common start therefore stay in step. "
        "Each frame shows the time from the level hold and the rod's turn in a corner, as does the viewer's status "
        "line. All are H.264 with 4:2:0 colour, the format QuickTime and PowerPoint open.</p>"
        + tab)


def sec_limits():
    return ("<p>The page shows one roll of one tip at 4 N. The GP was fitted to the same tip's rig sessions, the rod's "
            "session of that night among them, so it had seen presses with this rod near this point. A model of one "
            f"tip, used on another tip, misplaced contacts by 1.0 to 1.3 mm on the <a href=\"{CAL}\">calibration "
            "page</a>. The UR5 film is a replay of the robot's recorded joints in a model of the cell; no camera "
            "filmed the roll. The FE contact is frictionless, and the HEX-E's force is spread over it by pressure. The FE "
            "face sank 0.53 mm, while the robot's record puts the rod 0.89 to 1.04 mm into the undeformed pad.</p>")


def sec_future():
    return ("<ul>"
            "<li>Replay the four 1.5 N rolls of the same night: set <code>ROLL</code> in "
            "<code>scripts/fingertip/viewer_roll_film.py</code> and run <code>prepare</code>, <code>frames</code>, "
            "<code>capture</code> and <code>compose</code>. On Figure 8 of the HOM review page the raw GP's error grew "
            "from 0.7 to 1.25 mm at 1.5 N. A median distance over 1 mm there would show that the GP's 0.49 mm holds at "
            "4 N only.</li>"
            "<li>Refit the field model's strength at the face centre to the UR5 presses, as the physical-model page's "
            "press-fitted model does, and rerun <code>viewer_roll_film.py frames --calib</code>. If the physical "
            "model's median distance stays above 2 mm, the face centre's excess signal does not explain its error.</li>"
            "<li>Run the overhead D435 time-lapse (<code>lab/timelapse_d435.py</code> on corvus) during the next roll on "
            "the UR5, so that a film can show the real roll beside the viewer.</li>"
            "</ul>")


def sec_files():
    return ("<ul>"
            "<li><code>scripts/fingertip/viewer_roll_film.py</code>. With <code>prepare</code> the roll's board records, "
            "the robot's contact and the GP without rolls are cut. With <code>frames</code> the viewer's own "
            "<code>App</code> is run on a set clock. With <code>capture</code> the viewer's page is loaded in headless "
            "Chrome, its WebSocket replaced by a feed. <code>compose</code> and <code>table</code> write the film and "
            "the page's table.</li>"
            "<li><code>scripts/fingertip/fingertip_roll_fe_film.py</code>: <code>solve</code> (FE keyframes, femtip "
            "environment) and <code>film</code></li>"
            "<li>Data in <code>data/</code>: <code>20261110-v3id5_rod_roll_board.npz</code> (raw board records), "
            "<code>_truth.npz</code>, <code>_frames.json</code>, <code>_fe_summary.json</code>, "
            "<code>20261110-board_gp_v3id5_no_rolls.npz</code> (the GP in the viewer's <code>--board-gp</code> format) "
            "and <code>20261110-v3id5_full_rest_calib.json</code> (the viewer's <code>--calib</code> format)</li>"
            "<li>Films in <code>media/</code>: <code>20261110-v3id5_rod_roll_viewer_film.mp4</code> and "
            "<code>20261110-v3id5_rod_roll_fe_field_film.mp4</code>; the synced parts and the UR5 replay in "
            "<code>media/20261110-synced_parts/</code> (<code>viewer_roll_film.py parts</code> and <code>ur5</code>, "
            "<code>fingertip_roll_fe_film.py film --parts</code>); all H.264, yuv420p</li>"
            f"<li>Builder: <code>scripts/fingertip/viewer_roll_page.py</code>. Related: <a href=\"{TIPS}\">the printed "
            f"fingertips overview</a>, <a href=\"{CAL}\">the calibration page</a>, <a href=\"{PHYS}\">the physical-model "
            "page</a>.</li></ul>")


def lede(S, FE):
    g, q = S["gp"], S["physical"]
    fm = FE["film"]
    text = (
        "The fingertip viewer was replayed on a rod roll of the UR5 calibration, beside the roll drawn from the "
        "robot's record. With the GP of the UR5 calibration, the shown contact was a median "
        f"{g['med']:.2f} mm from the robot's contact point. Its press was {g['fn']:.2f} N, against "
        f"{S['hexe']:.2f} N from the HEX-E. With the physical model fitted to the rest reading, the median distance was "
        f"{q['med']:.2f} mm and the press {q['fn']:.2f} N. The GP had no hold of a roll in its training. The chips' "
        "change was also computed with a finite-element and field model of the same roll. It correlates with the "
        f"measured change at {fm['correlation_median']:.2f}, at {fm['size_ratio_median']:.2f} times its size.")
    summary = (f"Replayed on a 4 N rod roll of v3 ID 5, the fingertip viewer placed contacts a median {g['med']:.2f} mm "
               f"from the robot's contact with the UR5-trained GP and {q['med']:.2f} mm with the physical model alone; "
               f"an FE and field model of the roll gives the chips' change at correlation "
               f"{fm['correlation_median']:.2f} and {fm['size_ratio_median']:.2f} times its size.")
    return text, summary


def render(page_text, holder):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page_text)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    holder.append(len(items))
    return P.TEX_RE.sub(lambda m: next(svgs), page_text)


def main():
    F, FE, GP, FS = load()
    S = stats(F)
    fig1, fig2 = fig_distance(F), fig_holds(FE)
    sections = [("terms", "Terms and metrics", sec_terms(GP)),
                ("roll", "Rod roll on the UR5 calibration rig", sec_roll(S)),
                ("viewer", "Contact estimates of the viewer during the roll", sec_viewer(S, fig1)),
                ("fe", "Finite elements and magnetic field of the same roll", sec_fe(FE, fig2)),
                ("parts", "UR5 replay of the roll and synced films for slides", sec_parts()),
                ("limits", "Limitations", sec_limits()),
                ("future", "Future work", sec_future()),
                ("files", "Files", sec_files())]
    lede_t, summary = lede(S, FE)
    unit = (lambda s_: re.sub(r"(\d) (%|mm|µm|µT|N\b|s\b|mT)", "\\1&nbsp;\\2", s_))
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{unit(b)}</section>'
                     for k, (sid, hd, b) in enumerate(sections, 1))
    byline = ("v3 ID 5 on seat 2 of the inverted rig, 4 November 2026, 04:32; replays and FE solves 2026-11-10. "
              "Builder: <code>scripts/fingertip/viewer_roll_page.py</code>.")
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": "Fingertip viewer on a roll", "H1": "Fingertip viewer replayed on a UR5 rod roll, with the "
                 "finite-element and field model of the roll", "EYEBROW": "Fingertip sensing, 2026-11-10",
                 "LEDE": unit(lede_t), "BYLINE": byline, "TOC": toc, "SECTIONS": secs,
                 "FOOTER": str(OUT.relative_to(ROOT)), "TEXCSS": texsvg.CSS + texsvg.BLOCK_CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    n = []
    page = render(page, n)
    page = docs_index.tag(page, topic="fingertip", summary=summary, date="2026-11-10")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {n[0]} formulas", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
