#!/usr/bin/env python3
"""Builds docs/experiments/20261105-object_tracking_sim/20261105-object_tracking_sim.html: steps 1 to 3 of the object
seating and tracking plan (docs/notes/20261105-object_seating_and_tracking/) in simulation, with the owner's changes of
2026-11-05 ("the dropping seat ... i think we should just move the ur5 up more"; "what the hell are we using ballast
for?"; "can we take this feedback and execute on these initial steps 1-3 in sim for review?").

Data, all under the folder's rows/: stem.json and rollouts.jsonl (scripts/hardware/object_tracking_rollouts.py, the
repository's .venv), tagsize.json, visibility_flat.json, turntable.json and track_flat.json
(scripts/hardware/object_tracking_sim.py, ~/miniconda3/bin/python); the films in media/.

    .venv/bin/python scripts/hardware/object_tracking_sim_page.py figs   # the figures
    .venv/bin/python scripts/hardware/object_tracking_sim_page.py        # the page
"""
from __future__ import annotations

import collections
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import docs_index  # noqa: E402
import fp_review_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

fill, table, img, film, esc = P.fill, P.table, P.img, P.film, P.esc
D = ROOT / "docs/experiments/20261105-object_tracking_sim"
ROWS = D / "rows"
MEDIA = D / "media"
OUT = D / "20261105-object_tracking_sim.html"
TEMPLATE = ROOT / "scripts/hardware/object_tracking_sim_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
PLAN = "docs/notes/20261105-object_seating_and_tracking/20261105-object_seating_and_tracking_plan.md"
TASKS = ("pinpush", "press", "roll", "twist")
NAME = {"pinpush": "Pin-push", "press": "Press", "roll": "Roll", "twist": "Twist"}
ROLES = ("generalist", "twist", "roll", "pinpush", "press")
XY = {"generalist": (70, 110), "twist": (100, 70), "roll": (85, 170), "pinpush": (75, 55), "press": (70, 150)}
LAYOUT = {r: f"({XY[r][0]}, {XY[r][1]})" for r in ROLES}
CONFIGS = ("base", "flat", "built", "stem")
CNAME = {"base": "study", "flat": "flat seat", "built": "pocket seat", "stem": "held to 1.6 s"}


def load():
    rows = [json.loads(x) for x in (ROWS / "rollouts.jsonl").read_text().splitlines() if x.strip()]
    C = {"stem": json.loads((ROWS / "stem.json").read_text()), "rows": rows,
         "tag": json.loads((ROWS / "tagsize.json").read_text())}
    for k in ("visibility_flat", "turntable", "tilt"):
        p = ROWS / f"{k}.json"
        C[k] = json.loads(p.read_text()) if p.exists() else None
    parts = sorted(ROWS.glob("track_flat_part*of*.json"))
    C["track"] = None
    if parts:
        J = [json.loads(p_.read_text()) for p_ in parts]
        C["track"] = {"pair": J[0]["pair"], "fps": J[0]["fps"], "rows": [r for j in J for r in j["rows"]]}
    return C


def score_of(r):
    return float(r.get("score", 0.0)) if r.get("status") == "ok" else 0.0


def grid(C):
    """{(task, role, config): [rows]} with every rollout, the diverged ones scored 0."""
    call_role = {}
    for r in C["rows"]:
        if r.get("role"):
            call_role[r["call"]] = r["role"]
    out = collections.defaultdict(list)
    for r in C["rows"]:
        out[(r["task"], r.get("role") or call_role[r["call"]], r["config"])].append(r)
    return out


def mean_score(G, task, role, cfg):
    v = [score_of(r) for r in G.get((task, role, cfg), [])]
    return st.mean(v) if v else None


def f2(v, k=2):
    return "&ndash;" if v is None else f"{v:.{k}f}"


def masses():
    sys.path.insert(0, str(ROOT / "scripts/hardware"))
    import object_tracking_mjcf as OM
    out = {}
    for shape, r in (("cylinder", 12.5), ("sphere", 20.0)):
        stem = json.loads((ROWS / "stem.json").read_text())["stem_mm"][shape]
        mb, ms, mk = OM.masses(shape, r, stem)
        yc_stem = -(r - 1.0 + (stem + 1.0) / 2)
        yc_block = OM.block_centre_O(r, stem)[1]
        com = (ms * yc_stem + mk * yc_block) / OM.MASS_G
        out[shape] = {"r_mm": r, "stem_mm": stem, "body_g": mb, "stem_g": ms, "block_g": mk, "com_mm": com,
                      "block_centre_mm": yc_block, "block_bottom_mm": yc_block - OM.BLOCK.height / 2}
    return out


# ------------------------------------------------------------------------------------------------- figures
def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
                         "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "mathtext.fontset": "cm"})
    return plt


def save(fig, name):
    import matplotlib.pyplot as plt
    fig.savefig(MEDIA / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", (MEDIA / name).relative_to(ROOT), flush=True)


SIZE_COL = {10.0: "#9e0142", 12.0: "#f46d43", 13.0: "#000000", 14.0: "#3288bd", 16.0: "#5e4fa2"}


def fig_tagsize(C):
    plt = plt_setup()
    R = C["tag"]["rows"]
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, (res, dist) in zip(axs.ravel(), ((1080, 0.3), (1080, 0.5), (720, 0.3), (720, 0.5))):
        for fam, ls in (("tag36h11", "-"), ("tag25h9", "--")):
            for s, col in SIZE_COL.items():
                rr = sorted([r for r in R if r["res"] == res and r["dist_m"] == dist and r["family"] == fam
                             and r["size_mm"] == s], key=lambda r: r["angle_deg"])
                ax.plot([r["angle_deg"] for r in rr], [r["rate"] for r in rr], ls, color=col, lw=2.2 if s == 13 else 1.4,
                        label=f"{fam}, {s:g} mm" if (res, dist) == (1080, 0.3) else None)
        ax.axhline(0.99, color="0.5", lw=0.8, ls=":")
        ax.axvline(50, color="0.5", lw=0.8, ls=":")
        ax.set_title(f"{res}p, {dist:.1f} m")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    for ax in axs[1]:
        ax.set_xlabel("Angle from the tag's normal (deg)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Frames decoded")
    fig.legend(loc="outside right center", ncols=1)
    save(fig, "20261105-tagsize.png")


def fig_scores(C):
    plt = plt_setup()
    G = grid(C)
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.4), sharey=True, sharex=True, constrained_layout=True)
    mk = {"base": ("o", "#000000"), "flat": ("s", "#1f77b4"), "built": ("v", "#d62728")}
    for ax, task in zip(axs.ravel(), TASKS):
        for k, cfg in enumerate(("base", "flat", "built")):
            xs, ys, lo, hi = [], [], [], []
            for i, role in enumerate(ROLES):
                v = [score_of(r) for r in G.get((task, role, cfg), [])]
                if not v:
                    continue
                xs.append(i + (k - 1) * 0.22)
                ys.append(np.mean(v))
                lo.append(np.mean(v) - min(v))
                hi.append(max(v) - np.mean(v))
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk[cfg][0], color=mk[cfg][1], ms=7, capsize=2, lw=1,
                        label=CNAME[cfg].capitalize() if task == "pinpush" else None)
        ax.set_xticks(range(len(ROLES)), [f"{XY[r][0]},{XY[r][1]}" for r in ROLES], rotation=30)
        ax.set_title(NAME[task])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
    for ax in axs[:, 0]:
        ax.set_ylabel("Task score")
    for ax in axs[1]:
        ax.set_xlabel("Finger layout (mm)")
    fig.legend(loc="outside upper center", ncols=3)
    save(fig, "20261105-scores.png")


def fig_coverage(C):
    V = C["visibility_flat"]
    if V is None:
        return
    plt = plt_setup()
    S = V["single"]
    az = sorted({s["az"] for s in S})
    el = sorted({s["el"] for s in S})
    Z = np.full((len(el), len(az)), np.nan)
    Hh = np.full((len(el), len(az)), np.nan)
    for s in S:
        Z[el.index(s["el"]), az.index(s["az"])] = s["covered"]
        Hh[el.index(s["el"]), az.index(s["az"])] = s["hand"]
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, constrained_layout=True)
    for ax, M, title in ((axs[0], Z, "Two or more block tags seen whole by one camera"),
                         (axs[1], Hh, "The camera's hand tag seen whole")):
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       extent=(az[0] - 5, az[-1] + 5, el[0] - 5, el[-1] + 5))
        for p_ in V["pair"]:
            ax.plot(p_[0], p_[1], "o", ms=14, mfc="none", mec="white", mew=2.5)
        ax.set_title(title + " (share of the task's frames)")
        ax.set_ylabel("Elevation (deg)")
        ax.set_yticks(el)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    axs[1].set_xlabel("Azimuth from the palm's x axis (deg)")
    axs[1].set_xticks(range(0, 360, 30))
    save(fig, "20261105-coverage.png")


def tilt_summary(C):
    """Per pick of the study (config base): the object's tilt at 0.8, 1.6 and 2.0 s and its largest during the task
    (median and largest over seeds), from the trajectories; rows/tilt.json."""
    from sr2hand.geometry.rotations import quat_to_mat
    out = []
    G = grid(C)
    for task in TASKS:
        for role in ROLES:
            vals = []
            for r in G[(task, role, "base")]:
                z = np.load(ROOT / r["traj"], allow_pickle=True)
                t, q = z["t"], z["qpos"]
                Rm = quat_to_mat(q[:, 3:7])
                tilt = np.degrees(np.arccos(np.clip(Rm[:, 2, 1], -1, 1)))
                i = {x: int(np.searchsorted(t, x)) for x in (0.8, 1.6, 2.0)}
                vals.append((tilt[i[0.8]], tilt[i[1.6]], tilt[i[2.0]], tilt[i[2.0]:].max()))
            v = np.array(vals)
            out.append({"task": task, "role": role, "tilt_08_med": float(np.median(v[:, 0])),
                        "tilt_16_med": float(np.median(v[:, 1])), "tilt_20_med": float(np.median(v[:, 2])),
                        "tilt_20_max": float(v[:, 2].max()), "task_max_med": float(np.median(v[:, 3])),
                        "task_max_max": float(v[:, 3].max())})
    (ROWS / "tilt.json").write_text(json.dumps(out, indent=0))
    print("wrote", (ROWS / "tilt.json").relative_to(ROOT))


def cmd_figs():
    C = load()
    MEDIA.mkdir(parents=True, exist_ok=True)
    tilt_summary(C)
    fig_tagsize(C)
    fig_scores(C)
    fig_coverage(C)
    return 0


# ------------------------------------------------------------------------------------------------- sections
def sec_terms():
    terms = [
        ("$P$, $O$, $B$", "$P$ is the palm frame of <code>assets/cad_v3/README.md</code>. Its $x$ axis points from the "
         "thumb toward the index and middle, and its $z$ axis up toward the flange. $O$ is the object frame of the HOM "
         "scenes, at the object's centre. $B$ is the tag block's frame, at the block's centre, with $z_B$ toward the "
         "object. $T_{PO}$ is the pose of $O$ in $P$, in mm."),
        ("Study", "The task-conflict study's protocol. The object rests on a 3 mm post. The cylinder is held at its start "
         "pose until 0.8 s, and the post is removed at 1.6 s."),
        ("Flat seat", "The object as built rests on a flat seat. At 1.6 s the arm lifts the hand 40 mm in 0.4 s."),
        ("Pocket seat", "The flat seat with a 4 mm pocket round the block's lower end, 0.3 mm of clearance per side."),
        ("Held to 1.6 s", "The object as built is held at its start pose until 1.6 s and then released. There is no seat."),
        ("Task score", "<code>hom_landscape.rollout_score</code>, 0 unless the object is held at the end. Pin-push and "
         "press score $1-\\theta_\\mathrm{RMSE}/\\theta_\\mathrm{amp}$, with amplitudes of 15 and 20 deg. The roll scores "
         "$1/(1+\\omega_\\mathrm{RMSE}/40\\,\\mathrm{deg\\,s^{-1}})$. The twist scores the largest twist held on three pads "
         "over 90 deg. A pick's score is the mean over seeds 20 to 24."),
        ("Pick", "One layout's S3 grasp, commanded forces and contact modes for one task "
         "(<code>docs/experiments/20261103-hom_task_conflict/rows_refine/sv25</code>)."),
        ("Finger clearance", "The smallest distance between the stem or block and any finger geom over a rollout. It is "
         "negative when they touch."),
        ("Seat clearance", "The height of the block's lowest corner over the seat after 2.0 s."),
        ("Tilt", "The angle between $O$'s $+y$ axis, which points up the stem, and the palm's $+z$ axis."),
        ("Decode rate", "The share of frames in which the detector returns a tag with its right id."),
        ("Corner error", "The RMS distance in px between detected and true tag corners."),
        ("Corner jitter", "The RMS spread of each corner over 8 noise draws at one pose, in px."),
        ("Coverage", "The share of the task's frames ($t \\geq$ 2.0 s) in which at least two distinct block tags are seen "
         "whole by the two cameras together. Whole means that nine points of the tag are unoccluded and 10 px or more "
         "inside the image, with the tag within 60 deg of facing the camera."),
        ("Rotation and position error", "The RMS over frames of the angle between the solved and the true rotation, and "
         "of the distance between the solved and the true origin."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in terms) + "</dl>"


def sec_changes(C):
    M = masses()
    cy, sp = M["cylinder"], M["sphere"]
    st_rows = C["stem"]["rows"]
    cyl_reach = max(max(r["below_bottom_open_mm"], r["below_bottom_touch_mm"]) for r in st_rows
                    if r["task"] in ("pinpush", "press"))
    sph_reach = max(max(r["below_bottom_open_mm"], r["below_bottom_touch_mm"]) for r in st_rows
                    if r["task"] in ("roll", "twist"))
    v_cyl = 1.24 * math.pi * 12.5 ** 2 * 100 / 1000
    v_sph = 1.24 * 4 / 3 * math.pi * 20 ** 3 / 1000
    return (f'<p>Four changes to the plan (<code>{PLAN}</code>) come from the owner\'s review and from this simulation. '
            f'First, the seat no longer drops away. It is fixed to the table. At 1.6 s, when the simulation removes its '
            f'post, the arm lifts the hand 40 mm in 0.4 s on a minimum-jerk profile. The peak acceleration is '
            f'1.44 m/s&sup2;.</p>'
            f'<p>Second, the objects have no ballast. The earlier recipes put steel at the centre only to reach 25 g with '
            f'thin walls and sparse infill. A denser infill reaches 25 g alone. The cylinder\'s body is then '
            f'{cy["body_g"]:.1f} g, {100 * cy["body_g"] / v_cyl:.0f} % of solid PLA. The sphere\'s body is '
            f'{sp["body_g"]:.1f} g, {100 * sp["body_g"] / v_sph:.0f} %. The stem and the 3 g block lower the centre of mass '
            f'by {-cy["com_mm"]:.1f} mm in the cylinder and {-sp["com_mm"]:.1f} mm in the sphere. Every rollout here '
            f'includes them.</p>'
            f'<p>Third, the stems are longer than the plan\'s 22 mm. The fingers were posed at the open and touch poses of '
            f'the five layouts\' S3 grasps. There the press\'s middle finger reaches {cyl_reach:.1f} mm below the cylinder, '
            f'because it touches 30 deg below the equator. The tripods reach {sph_reach:.1f} mm below the sphere. Each '
            f'stem puts the block\'s top 15 mm under the lowest point: {cy["stem_mm"]:.1f} mm on the cylinder and '
            f'{sp["stem_mm"]:.1f} mm on the sphere. Fourth, the seat is flat, for the reasons of Section SEC[seat].</p>'
            + table(["", "Cylinder", "Sphere"],
                    [["Stem length (mm)", f"{cy['stem_mm']:.1f}", f"{sp['stem_mm']:.1f}"],
                     ["Body, stem, block (g)", f"{cy['body_g']:.1f}, {cy['stem_g']:.2f}, {cy['block_g']:.0f}",
                      f"{sp['body_g']:.1f}, {sp['stem_g']:.2f}, {sp['block_g']:.0f}"],
                     ["Centre of mass below $O$ (mm)", f"{-cy['com_mm']:.1f}", f"{-sp['com_mm']:.1f}"],
                     ["Block's underside below $O$ (mm)", f"{-cy['block_bottom_mm']:.1f}", f"{-sp['block_bottom_mm']:.1f}"]],
                    "The objects as built and simulated: 25 g in all, PLA stems 4 mm across and the plan's 3 g hollow block.",
                    num_from=1))


def tag_reach(R, res, dist, fam, size):
    """The largest view angle with a decode rate of 0.99 or more at it and every smaller angle."""
    rr = sorted([r for r in R if r["res"] == res and r["dist_m"] == dist and r["family"] == fam and r["size_mm"] == size],
                key=lambda r: r["angle_deg"])
    ok = None
    for r in rr:
        if r["rate"] < 0.99:
            break
        ok = r["angle_deg"]
    return ok


def sec_tagsize(C):
    R = C["tag"]["rows"]
    sizes = (10.0, 12.0, 13.0, 14.0, 16.0)
    rows = []
    for res in (1080, 720):
        for dist in (0.25, 0.3, 0.4, 0.5):
            cells = []
            for fam in ("tag36h11", "tag25h9"):
                for s_ in sizes:
                    v = tag_reach(R, res, dist, fam, s_)
                    cells.append("&ndash;" if v is None else f"{v:g}")
            rows.append([f"{res}p", f"{dist:.2f}"] + cells)
    tb = table(["Stream", "Distance (m)"] + [f"36h11 {s_:g}" for s_ in sizes] + [f"25h9 {s_:g}" for s_ in sizes], rows,
               "The largest angle from the tag's normal (deg) at which a tag decodes in at least 99 % of frames, and at "
               "every smaller angle, by family and size (mm). 70 is the largest angle tested. A dash means the tag fails "
               "head-on. Each cell has 48 frames: 6 poses jittered by 2 mm and 1 deg, with 8 noise draws each.", num_from=2)
    r13 = {(r["res"], r["angle_deg"]): r for r in R if r["family"] == "tag36h11" and r["size_mm"] == 13.0
           and r["dist_m"] == 0.3}
    jit = [r["corner_jitter_px"] for r in R if r["dist_m"] == 0.3 and r["corner_jitter_px"] is not None]
    err = [r["corner_err_px"] for r in R if r["dist_m"] == 0.3 and r["corner_err_px"] is not None and r["angle_deg"] <= 60]
    return (f'<p>At 1080p every tag of 10 mm or more decodes in at least 99 % of frames up to 65 deg from its normal at '
            f'0.3 m. The plan\'s 13 mm tag36h11 decodes to {tag_reach(R, 1080, 0.5, "tag36h11", 13.0):g} deg even at 0.5 m. '
            f'At 720p it decodes to {tag_reach(R, 720, 0.3, "tag36h11", 13.0):g} deg at 0.3 m and to '
            f'{tag_reach(R, 720, 0.5, "tag36h11", 13.0):g} deg at 0.5 m. At 0.3 m a cell of the 13 mm tag spans '
            f'{r13[(1080, 0.0)]["px_per_cell"]:.1f} px at 1080p and {r13[(720, 0.0)]["px_per_cell"]:.1f} px at 720p.</p>'
            f'<p>The corner jitter is {min(jit):.2f} to {max(jit):.2f} px at 0.3 m, under the plan\'s 0.15 px. The corner '
            f'error against the true corners is {min(err):.2f} to {max(err):.2f} px RMS up to 60 deg. It grows with the '
            f'angle, so part of it is a bias of the detector\'s edge fit.</p>'
            f'<p>The 13 mm tag36h11 stays in the plan. It is the largest tag36h11 whose one-cell white border fits a '
            f'17.3 mm face of the block. A 14 mm tag would need 17.5 mm. tag25h9 decodes a little further at the same '
            f'size, because its cells are larger. It has 35 ids, against 587 for tag36h11.</p>'
            f'<p>The test led to two corrections. <code>pupil_apriltags</code> places pixel centres at +0.5 px, and OpenCV '
            f'places them at integers. On images of known geometry its corners lay +0.31 to +0.50 px off in both axes. '
            f'<code>sr2hand.tracking.pose.detect</code> therefore subtracts 0.5 px. The second correction concerns the '
            f'renders. Black cells standing 0.1 mm proud of the face showed their side walls at oblique views. That moved '
            f'the corners 0.23 px outward and put the block 1.16 mm too close at 0.3 m. The tags here are flat mosaics '
            f'0.04 mm thick.</p>'
            + img(MEDIA / "20261105-tagsize.png",
                  "Share of frames decoded against the angle from the tag's normal, at 0.3 and 0.5 m and at 1080p and 720p. "
                  "Solid lines are tag36h11 and dashed lines tag25h9. The thick black line is the plan's 13 mm tag36h11. The "
                  "dotted lines mark 99 % and 50 deg, the plan's acceptance.") + tb)


def sec_seat(C):
    G = grid(C)
    rows, cyl_close, cyl_up, worst = [], 0, 0, 0.0
    for task in TASKS:
        for role in ROLES:
            fl = G[(task, role, "flat")]
            b, f = mean_score(G, task, role, "base"), mean_score(G, task, role, "flat")
            k, h = mean_score(G, task, role, "built"), mean_score(G, task, role, "stem")
            cl = min(r["clear_min_mm"] for r in fl)
            n_touch = sum(r["clear_min_mm"] < 0.0 for r in fl)
            se = min(r["seat_clear_min_mm"] for r in fl) + 4.0           # over the flat seat, not the pocket's rim
            if task in ("pinpush", "press"):
                cyl_close += abs(f - b) <= 0.02
                cyl_up += f - b > 0.02
                worst = min(worst, f - b)
            rows.append([NAME[task], LAYOUT[role], f2(b), f2(f), f"{f - b:+.2f}", f2(k), f2(h),
                         ("&gt; 10" if cl >= 9.99 else f"{cl:.1f}"), f"{n_touch}", f"{se:.1f}"])
    tb = table(["Task", "Layout (mm)", "Study", "Flat seat", "Change", "Pocket seat", "Held to 1.6 s",
                "Finger clearance (mm)", "Rollouts touching", "Seat clearance (mm)"], rows,
               "Mean task score of each pick over seeds 20 to 24 in four configurations. For the flat seat: the smallest "
               "finger clearance over the five rollouts, the rollouts in which the stem or block touches a finger, and the "
               "smallest seat clearance after 2.0 s. Clearances are reported up to 10 mm.", num_from=2)
    seat_min = min(min(r["seat_clear_min_mm"] for r in G[(t, ro, "flat")]) for t in TASKS for ro in ROLES) + 4.0
    return (f'<p>On a flat seat the cylinder tasks score as in the study. Of the ten cylinder picks, {cyl_close} score '
            f'within 0.02 of the study. The largest loss is {-worst:.3f}. The other {cyl_up} picks score higher, by 0.09 '
            f'to 0.34; they are the three weakest picks. The stem and block clear every finger by 1.6 mm or more in all 50 '
            f'cylinder rollouts. Except for the pin-push on the press specialist\'s layout, they clear by 5.3 mm or more. '
            f'The object is held at the end of every flat-seat rollout.</p>'
            f'<p>A pocket that locates the block breaks the cylinder tasks. With 4 mm walls and 0.3 mm of clearance, the '
            f'pin-push scores 0 at four of five layouts. The press scores 0.04 to 0.64 lower. The grasp forces come on at 0.8 s '
            f'while the block is caught in the pocket. The pinch then pushes the object 1 to 3 mm against the walls. Once '
            f'the seat leaves, the cylinder swings about the pinch axis and falls (Film 1). Holding the object fixed until '
            f'1.6 s fails the same way.</p>'
            f'<p>On the study\'s post the object is supported but free to slide and tilt. Before 1.6 s it travels 1.9 to '
            f'30.3 mm as the fingers close. On the flat seat it travels within 1 to 5 mm of that.</p>'
            f'<p>The lift\'s acceleration does not matter. Three picks were run with the lift\'s acceleration and no seat. '
            f'They scored 0.87, 0.87 and 0.95, against 0.86, 0.87 and 0.94 without it. The 40 mm lift is more than '
            f'needed. During every task the block stays at least {seat_min:.1f} mm above the seat. It therefore never '
            f'comes down more than {40.0 - seat_min:.1f} mm below its seated height. A 15 mm lift would leave '
            f'{15.0 - (40.0 - seat_min):.1f} mm.</p>'
            + img(MEDIA / "20261105-scores.png",
                  "Task score of each pick in the study, on the flat seat and on the pocket seat. Markers are the means over "
                  "seeds 20 to 24 and bars the range. One pin-push rollout of the generalist held to 1.6 s diverged; it is "
                  "scored 0.")
            + film(MEDIA / "20261105-film_seat_press.mp4",
                   "The press on the press specialist's layout (70, 150), seed 20: the study's post, the pocket seat and the "
                   "flat seat. In the pocket the block is caught while the forces come on. After the lift the cylinder "
                   "swings about the pinch and drops.") + tb)


def sec_sphere(C):
    T = {(r["task"], r["role"]): r for r in C["tilt"]}
    rows = []
    for task in TASKS:
        for role in ROLES:
            r = T[(task, role)]
            rows.append([NAME[task], LAYOUT[role], f"{r['tilt_16_med']:.0f}", f"{r['tilt_20_max']:.0f}",
                         f"{r['task_max_med']:.0f}", f"{r['task_max_max']:.0f}"])
    tb = table(["Task", "Layout (mm)", "Tilt at 1.6 s, median", "Tilt at 2.0 s, largest", "Largest in the task, median",
                "Largest in the task, largest"], rows,
               "Tilt of the object (deg) in the study's own rollouts, without a stem, at seeds 20 to 24: at the post's "
               "removal, at the task's start and during the task. For the cylinder the tilt is mostly the turn about the "
               "pinch axis that the task commands.", num_from=2)
    G = grid(C)
    bad = [(t, ro) for t in ("roll", "twist") for ro in ROLES
           if sum(r["clear_min_mm"] < 0 for r in G[(t, ro, "flat")]) >= 3]
    return (f'<p>A stem on the sphere works at two of the five layouts. In the roll and the twist the sphere may roll on '
            f'every pad (free rolling, $w_\\mathrm{{roll}}$ = 0). In the study\'s own rollouts its vertical axis turns by '
            f'2 to 104 deg (median per pick) before the task starts. A stem turns with it.</p>'
            f'<p>On the flat seat the stem or block touches a finger in 3 to 5 of 5 rollouts in {len(bad)} of the ten '
            f'sphere picks. They are both tasks at the roll and press specialists\' layouts, and the twist at the twist '
            f'specialist\'s. Their scores fall by up to 0.50. At the generalist (70, 110) the stem clears by 7.1 mm or more. At '
            f'the pin-push specialist (75, 55) it never comes within 10 mm. At both layouts the sphere tasks score within '
            f'0.04 of the study (Film 2).</p>'
            f'<p>Tags on the sphere\'s surface remain the option for the other three layouts. The cylinder tilts 3 to 17 '
            f'deg before its task and up to 34 deg in it, mostly about the pinch axis. Its stem clears in every rollout.</p>'
            + film(MEDIA / "20261105-film_sphere_stem.mp4",
                   "The sphere on the flat seat at seed 20. Left: the roll at the roll specialist's layout (85, 170). The "
                   "tripod turns the sphere until the stem lies beside the fingers. Middle and right: the twist and the roll "
                   "at the generalist (70, 110), where the stem stays within 40 deg of vertical.") + tb)


def sec_cameras(C):
    V = C["visibility_flat"]
    S = {(x["az"], x["el"]): x for x in V["single"]}
    a, b = (tuple(x) for x in V["pair"])
    sep = abs((a[0] - b[0] + 180.0) % 360.0 - 180.0)
    per = V["per_rollout"]
    worst = min(per, key=lambda r: r["covered"])
    groups = {(r["task"], r["role"]) for r in per}
    n_full = sum(1 for g in groups if min(x["covered"] for x in per if (x["task"], x["role"]) == g) >= 0.999)
    return (f'<p>Two D435s stand on their sides at azimuths {a[0]:.0f} and {b[0]:.0f} deg from the palm\'s $x$ axis, '
            f'{sep:.0f} deg apart. Both are 0.3 m from the block in the task, at its height. Together they see two or more '
            f'block tags whole in {100 * V["covered"]:.1f} % of the task\'s frames over all 100 flat-seat rollouts. Each '
            f'camera\'s hand tag is in view in {100 * V["hand_tags_both"]:.0f} % of them. In the seated frames, two or '
            f'more block tags are in view in {100 * V["covered_seated"]:.0f} %.</p>'
            f'<p>{n_full} of the 20 picks are covered in every frame. The worst rollout is the '
            f'{NAME[worst["task"]].lower()} at the {worst["role"]} specialist\'s layout, at {100 * worst["covered"]:.1f} %. '
            f'Alone, the cameras cover {100 * S[a]["covered"]:.0f} % and {100 * S[b]["covered"]:.0f} % of the task\'s '
            f'frames.</p>'
            f'<p>On its side a D435\'s colour stream is 42 deg across and 69 deg high. The hand tags sit at the height of '
            f'the tag mount\'s tag 6, 44 mm above $P$, here 110 mm from its axis. Such a tag lies about 170 mm above the '
            f'block in the task, and the seated block is another 40 mm lower. At 0.3 m the two do not fit in the 42 deg '
            f'height of an upright camera.</p>'
            f'<p>The candidates were 36 azimuths and five elevations from 0 to 40 deg. Each was aimed at the middle '
            f'between the seated block and its hand tag in the task. Rays from the camera to nine points of each tag were '
            f'tested against the fingers\' visual meshes, the object, the seat and the palm, at 10 frames/s. A pair had to '
            f'be 60 deg apart or more. From 10 deg or more above the block, a pair keeps the block\'s tags in view in up '
            f'to 99.9 % of frames. Its hand tags are then hidden behind the fingers.</p>'
            + img(MEDIA / "20261105-coverage.png",
                  "One camera's coverage over the task's frames of the 100 flat-seat rollouts, by azimuth and elevation. "
                  "Top: two or more block tags in view and whole. Bottom: the camera's hand tag in view and whole. "
                  "Circles mark the chosen pair."))


def sec_turntable(C):
    J = C["turntable"]
    rows = []
    for r in J["turntable"]:
        rows.append([f"{r['wedge_deg']:.0f}", "CAD" if r["map"] == "cad" else "refined",
                     f"{r['tracked']}/{r['frames']}", f"{r['tags_mean']:.1f}", f"{r['rot_rms_deg']:.2f}",
                     f"{r['pos_rms_mm']:.2f}", f"{r['angle_rms_deg']:.2f}", f"{r['circle_rms_mm']:.2f}"])
    sp = J.get("spin")
    tb = table(["Wedge (deg)", "Tag map", "Frames tracked", "Tags per frame", "Rotation error (deg)",
                "Position error (mm)", "Angle against the encoder (deg)", "Centre off its path (mm)"], rows,
               "The block on the turntable through 360 deg in 5 deg steps, three noise draws per step. One camera on its "
               "side stands 0.3 m away and 20 deg above the block. The last two columns are the bench's own scores, which "
               "need no ground truth. One is the solved angle about the fitted axis against the encoder's 12 bits. The "
               "other is the block centre's distance from its fitted circle, or from its mean on the level table.",
               num_from=2)
    ba = J["ba"]
    ref = [r for r in J["turntable"] if r["map"] == "refined"]
    cad = [r for r in J["turntable"] if r["map"] == "cad"]
    out = (f'<p>With its tag map refined by bundle adjustment, the block is solved in every frame to '
           f'{max(r["rot_rms_deg"] for r in ref):.2f} deg and {max(r["pos_rms_mm"] for r in ref):.2f} mm RMS against the '
           f'truth. With the CAD map the errors are {max(r["rot_rms_deg"] for r in cad):.2f} deg and '
           f'{max(r["pos_rms_mm"] for r in cad):.2f} mm. Both hold level and on the 20 deg wedge.</p>'
           f'<p>The printed block was modelled with errors on each tag (1 sd): 0.10 mm in its face, 0.05 mm off it, 0.30 deg '
           f'of turn and 0.20 deg of tilt. Its corners then lie {J["map_error_cad_mm"][0]:.3f} mm RMS from the CAD map. The '
           f'adjustment used {ba["views"]} views with {ba["tags_per_view"]:.1f} tags each, at {ba["rms_px"]:.2f} px RMS. It '
           f'brings the map error to {ba["map_error_refined_mm"][0]:.3f} mm.</p>')
    if sp:
        out += (f'<p>The block was also spun at {sp["deg_per_s"]:.0f} deg/s on the wedge. The frame was read by a rolling shutter '
                f'in {1000 * sp["readout_s"]:.0f} ms, each row exposed for {1000 * sp["exposure_s"]:.0f} ms. The block was '
                f'tracked in {sp["tracked"]} of {sp["frames"]} frames. Against its pose at the frame\'s middle the errors '
                f'were {sp["rot_mid_rms"]:.2f} deg and {sp["err_mid_rms"]:.2f} mm RMS. Each frame was also timed when the '
                f'block\'s column was read; the rotation error was then {sp["rot_row_rms"]:.2f} deg. The plan\'s estimate of '
                f'a 0.2 deg shear at the roll\'s 40 deg/s is of the right size.</p>')
    out += ('<p>One camera\'s position error lies mostly along its axis. The detector places corners about 0.06 px outside '
            'the true ones. The 13 mm tags then look 0.1 % larger, and the block appears about 0.5 mm closer at 0.3 m. '
            'Across the image the error is 0.01 to 0.02 mm. This bias does not show in the bench\'s own scores, the last '
            'two columns. For a second camera 90 deg away, the same depth is a lateral offset, which is measured well.</p>')
    return out + tb


def sec_tracking(C):
    J = C["track"]
    if J is None:
        return "<p>Not run.</p>"
    rows = []
    R = sorted(J["rows"], key=lambda r: (TASKS.index(r["task"]), ROLES.index(r["role"])))
    for r in R:
        rows.append([NAME[r["task"]], LAYOUT[r["role"]], f"{100 * r['kin_tracked_task']:.0f}",
                     f2(r.get("kin_pos_rms_mm")), f2(r.get("kin_rot_rms_deg")), f2(r.get("kin_pos_p99_mm")),
                     f2(r.get("kin_rot_p99_deg")), f2(r.get("pos_rms_mm")), f2(r.get("rot_rms_deg")),
                     f2(r.get("hand_tag_rot_rms_deg"))])
    tb = table(["Task", "Layout (mm)", "Frames tracked (%)", "Position RMS (mm)", "Rotation RMS (deg)",
                "Position, 99th pct (mm)", "Rotation, 99th pct (deg)", "Hand-tag route, position RMS (mm)",
                "Hand-tag route, rotation RMS (deg)", "Hand tag's rotation error (deg)"], rows,
               "The object's pose in the palm frame over the task's frames of each pick's seed-20 rollout, with both "
               "cameras in one solve. Columns 3 to 7 take each camera's pose in $P$ from the arm, exact here. Columns 8 "
               "and 9 take it from the camera's one 40 mm hand tag. The last column is that tag's own rotation error.",
               num_from=2)
    K = [r for r in R if r.get("kin_pos_rms_mm") is not None]
    H = [r for r in R if r.get("pos_rms_mm") is not None]
    wk = max(K, key=lambda r: r["kin_pos_p99_mm"])
    trk = min(r["kin_tracked_task"] for r in K)
    trk_txt = "in every frame of the task" if trk >= 0.9995 else f"in {100 * trk:.1f} to 100 % of the task's frames"
    return (f'<p>With each camera\'s pose in the palm frame known, the tracker solves $T_{{PO}}$ {trk_txt}. Per pick the '
            f'errors are {min(r["kin_pos_rms_mm"] for r in K):.2f} to {max(r["kin_pos_rms_mm"] for r in K):.2f} mm and '
            f'{min(r["kin_rot_rms_deg"] for r in K):.2f} to {max(r["kin_rot_rms_deg"] for r in K):.2f} deg RMS. The 99th '
            f'percentiles stay under {sorted(r["kin_pos_p99_mm"] for r in K)[-2]:.2f} mm and '
            f'{sorted(r["kin_rot_p99_deg"] for r in K)[-2]:.2f} deg except in one pick. That pick is the '
            f'{NAME[wk["task"]].lower()} at the {wk["role"]} specialist\'s layout, where the stem swings among the fingers: '
            f'{wk["kin_pos_p99_mm"]:.2f} mm and {wk["kin_rot_p99_deg"]:.2f} deg. Each frame was rendered by both cameras at '
            f'30 frames/s from the rollout and passed through the sensor model. It was then detected and solved, and '
            f'compared with the simulator\'s pose.</p>'
            f'<p>The plan\'s first route takes each camera\'s pose from its one 40 mm hand tag. It gives '
            f'{min(r["pos_rms_mm"] for r in H):.1f} to {max(r["pos_rms_mm"] for r in H):.1f} mm. A single tag seen nearly '
            f'head-on fixes its own tilt to only {min(r["hand_tag_rot_rms_deg"] for r in R):.1f} to '
            f'{max(r["hand_tag_rot_rms_deg"] for r in R):.1f} deg. The block lies about 170 mm from the tag, so each degree '
            f'moves the block about 3 mm in $P$. The plan\'s second route calibrates the camera\'s pose in the arm\'s base '
            f'once and takes the hand\'s pose from the joints. It avoids this error. The hand tags then only flag a camera '
            f'that has been bumped.</p>'
            + "".join(film(MEDIA / f"20261105-track_{t}_{c}.mp4", cap) for t, c, cap in (
                ("pinpush", 162, "The two cameras' images of the pin-push at the generalist (70, 110), seed 20, with the "
                 "detected tags outlined and the error of the solved pose."),
                ("twist", 170, "The twist at the press specialist's layout (70, 150), seed 20, the pick with the least "
                 "coverage. The stem swings toward the fingers.")))
            + tb)


def sec_limits():
    items = [
        "The camera and sensor are models. The camera is a pinhole with the plan's focal length and no distortion. The "
        "sensor adds a Gaussian blur of 0.6 px, shot noise of 10 electrons per digital number and 1.5 of read noise. "
        "There is no compression. The intrinsics are exact in every solve here; on the bench they come from a calibration.",
        "The cameras' poses in the palm frame are exact in the kinematic route. On the bench they will be off by the "
        "calibration's error and by the UR5's error over the 40 mm lift.",
        "The tags' print errors of 0.10 mm and 0.30 deg are assumed. The bundle adjustment ran with exact intrinsics.",
        "The rolling shutter's readout of 25 ms and exposure of 5 ms are assumed. The spin ran on the turntable; the HOM "
        "rollouts were rendered with a global shutter.",
        "The scores come from CPU MuJoCo at seeds 20 to 24 on the SCS0009 plant, with pads at 2.5 MPa. The scene friction "
        "is 1.0, and the bare PLA of the stem and block has 0.35.",
        "The hand tags were placed on the hand's outline, 110 mm from $P$'s axis and 44 mm above it, facing their camera. "
        "Their brackets are not in the cad_v3 model. The hand's static parts were not tested as occluders.",
    ]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def sec_future():
    items = [
        "Print the tag plate and record it with the colour stream at 1080p and 720p (plan, step 1). The 13 mm tag36h11 "
        "should decode in 99 % of frames to 50 deg at 0.3 m. A lower rate means the real sensor is noisier or softer than "
        "the model; <code>object_tracking_sim.py tagsize</code> takes <code>Sensor</code>'s blur and noise.",
        "Print the block, refine its map from 40 views and run the turntable (plan, step 3). With the refined map the "
        "simulated errors are 0.08 deg against the encoder and 0.20 mm from the fitted path. More than 0.3 deg or 0.3 mm "
        "falsifies the model of the print errors or of the detector.",
        "Calibrate each camera against the UR5's base with the hand tags over about 15 arm poses "
        "(<code>cv2.calibrateRobotWorldHandEye</code>). Then move the arm by known steps of 10 to 40 mm and compare the "
        "tracked motion of a seated object's block with the arm's. A difference over 0.3 mm puts the kinematic route "
        "outside the target.",
        "Mount the two cameras on their sides at azimuths 0 and 260 deg from $x_P$, 0.3 m from the block in the task and "
        "at its height. Check the coverage of the first trials against <code>rows/visibility_flat.json</code>.",
        "Choose the sphere's tags for the roll, press and twist specialists' layouts: tags on the sphere's surface, or "
        "these layouts left out of the sphere series. Rerun <code>object_tracking_rollouts.py</code> with a design that "
        "keeps the finger clearance at 3 mm in every rollout.",
        "Measure the pad's friction on the objects' contact surface (plan, decision 1), and rerun the rollouts at that "
        "friction.",
    ]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def sec_files():
    rows = [("scripts/hardware/object_tracking_rollouts.py", "the HOM rollouts of the five layouts in four configurations "
             "(the repository's .venv)"),
            ("scripts/hardware/object_tracking_sim.py", "rendering, the sensor model and detection: tagsize, visibility, "
             "pairs, turntable, track, film (~/miniconda3/bin/python)"),
            ("scripts/hardware/object_tracking_mjcf.py", "the stem, block, tags and seat in MJCF"),
            ("src/sr2hand/tracking/", "the tracking package for the bench: tag maps, detection, pose solves, bundle "
             "adjustment"),
            ("docs/experiments/20261105-object_tracking_sim/rows/", "stem.json, rollouts.jsonl, tagsize.json, "
             "visibility_flat.json, turntable.json, tilt.json, track_flat_part*.json"),
            ("logs/20261105-object_tracking_sim/", "the rollouts' trajectories and the per-frame tracking records (outside "
             "git)"),
            ("scripts/hardware/object_tracking_sim_page.py", "this page and its figures"),
            (PLAN, "the plan, updated with these results")]
    return "<ul>" + "".join(f"<li><code>{a}</code>: {b}</li>" for a, b in rows) + "</ul>"


def lede_and_summary(C):
    G = grid(C)
    V, J = C["visibility_flat"], C["track"]
    K = [r for r in J["rows"] if r.get("kin_pos_rms_mm") is not None] if J else []
    cyl = [(mean_score(G, t, ro, "flat") - mean_score(G, t, ro, "base")) for t in ("pinpush", "press") for ro in ROLES]
    if not K:
        return "", ""
    pos, rot = max(r["kin_pos_rms_mm"] for r in K), max(r["kin_rot_rms_deg"] for r in K)
    lede = (f"In simulation, steps 1 to 3 of the object tracking plan pass for the cylinder at all five layouts and for the "
            f"sphere at two. The seat is flat, and the arm lifts the hand 40 mm at 1.6 s. The largest loss of a cylinder "
            f"pick against the task-conflict study is then {-min(cyl):.3f}, and the stem clears every finger by 1.6 mm or "
            f"more. The sphere rolls in the HOM tripod, so its stem touches a finger at three layouts. Two D435s on their "
            f"sides see two or more block tags in {100 * V['covered']:.1f} % of the task's frames. With their poses taken "
            f"from the arm, the object is tracked to {pos:.2f} mm and {rot:.2f} deg RMS or better.")
    summary = (f"In simulation, a flat seat with a 40 mm arm lift keeps the HOM cylinder scores within {-min(cyl):.3f} of "
               f"the study with the stem clear of the fingers; the sphere's stem touches the fingers at three of five "
               f"layouts. Two side-mounted D435s track the object to {pos:.2f} mm and {rot:.2f} deg RMS with their poses "
               f"from the arm.")
    return lede, summary


def render(page):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = texsvg.render(items, cache_path=TEX_CACHE)
    it = iter(svgs)
    return P.TEX_RE.sub(lambda m: next(it), page), len(items)


def cmd_page():
    C = load()
    sections = [
        ("terms", "Terms and metrics", sec_terms()),
        ("changes", "Objects, seat and lift after the owner's review", sec_changes(C)),
        ("tags", "Tag family and size (step 1)", sec_tagsize(C)),
        ("seat", "Seat, stem and task scores in the HOM rollouts (step 2)", sec_seat(C)),
        ("sphere", "Rotation of the sphere in the HOM tripod", sec_sphere(C)),
        ("cameras", "Camera placement and visibility of the tags (step 2)", sec_cameras(C)),
        ("turntable", "Tag block on a turntable and bundle adjustment of its tags (step 3)", sec_turntable(C)),
        ("tracking", "Tracking of the object through the HOM rollouts", sec_tracking(C)),
        ("limits", "Limitations", sec_limits()),
        ("future", "Future work", sec_future()),
        ("files", "Files", sec_files()),
    ]
    num_ = {sid: k for k, (sid, _h, _b) in enumerate(sections, 1)}
    sub = (lambda s_: re.sub(r"SEC\[(\w+)\]", lambda m_: str(num_[m_.group(1)]), s_))
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|kg|px|N\b|ms\b|s\b|g\b|m\b)", "\\1&nbsp;\\2", s_))
    body = [(sid, hd, unit(sub(b))) for sid, hd, b in sections]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(body, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{b}</section>' for k, (sid, hd, b) in enumerate(body, 1))
    lede, summary = lede_and_summary(C)
    byline = ("Simulation of the plan's steps 1 to 3 on 2026-11-05, with the owner's changes of that day; no bench "
              "measurement. Builder: <code>scripts/hardware/object_tracking_sim_page.py</code>.")
    page = TEMPLATE.read_text()
    for k_, v in {"TITLE": "Object tracking simulation",
                  "H1": "Seat, tag block and cameras for HOM experiments with the SR2 hand on the UR5, in simulation",
                  "EYEBROW": "Hand and bench, 2026-11-05", "LEDE": unit(lede), "BYLINE": byline, "TOC": toc,
                  "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k_ + "}}", v)
    page, n = render(page)
    page = docs_index.tag(page, topic="hardware", summary=summary, date="2026-11-05")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT}: {OUT.stat().st_size / 1e6:.2f} MB, {n} formulas", flush=True)
    return 0


def main():
    a = sys.argv[1:]
    if a and a[0] == "figs":
        return cmd_figs()
    return cmd_page()


if __name__ == "__main__":
    sys.exit(main())
