#!/usr/bin/env python3
"""Builds docs/experiments/20261029-foundational_pose_review/20261109-hom_tracking_score.html: the tracking score of the
HOM tasks, its check against the owner's verdicts on the films of 2026-11-09, and the layouts re-picked by it at four
object sizes per task, filmed (owner, 2026-11-09: "you have my go ahead on the rescoring"; "i value
simplicity/interpretability/capturing the essence of the motion/success monotonically and smoothly over the 0-1 range
over overtuning the metric"; "i would prefer getting a usable set of videos over a full run").

Data: rows/20261109-track_rerun_sv25.jsonl and rows/20261109-track_picks_sv25.json (scripts/hom/hom_track_films.py);
films in media/.

    .venv/bin/python scripts/hom/hom_track_score_page.py
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

D, MEDIA, ROWS = P.D, P.MEDIA, P.ROWS
OUT = D / "20261109-hom_tracking_score.html"
TAG = "_sv25"
SEARCH = "20261109-hom_search_methods.html"
CONFLICT = "../20261103-hom_task_conflict/20261104-hom_task_conflict.html"
table, esc = P.table, P.esc
TASKS = ("twist", "roll", "press")
TNAME = {"twist": "Twist", "roll": "Roll", "press": "Press"}
OBJ = {"twist": "sphere", "roll": "sphere", "press": "cylinder"}
VERDICT = {("roll", 25.0, "best"): "loses a finger, oscillates", ("roll", 30.0, "best"): "loses a finger, oscillates",
           ("roll", 80.0, "best"): "loses a finger, oscillates", ("roll", 63.0, "best"): "works",
           ("twist", 25.0, "best"): "loses fingers, oscillates", ("twist", 30.0, "best"): "loses fingers, oscillates",
           ("twist", 25.0, "one"): "looks better than the pick", ("twist", 30.0, "one"): "looks better than the pick",
           ("press", 10.0, "best"): "good", ("press", 13.0, "best"): "good", ("press", 40.0, "best"): "good",
           ("press", 50.0, "best"): "good"}


def load():
    rows = [json.loads(x) for x in (ROWS / f"20261109-track_rerun{TAG}.jsonl").read_text().splitlines() if x.strip()]
    picks = json.loads((ROWS / f"20261109-track_picks{TAG}.json").read_text())
    return rows, picks


def f2(v):
    return "" if v is None else f"{v:.2f}"


def still_pinpush():
    """The pin-push's earlier score for a cylinder held still: 1 - the RMS of its 15 deg profile over the stroke."""
    t = np.linspace(0.0, 4.5, 4501)
    th = np.interp(t, [0.0, 1.0, 3.0, 4.0, 4.5], [0.0, 15.0, -15.0, 0.0, 0.0])
    return 1.0 - float(np.sqrt(np.mean(th ** 2))) / 15.0


def sec_terms():
    return (
        "<p><b>Layout.</b> The placement of the three finger mounts on the palm. On the 2-D lattice used here it is "
        "two distances in mm: thumb to the index-middle pair, $x_{sep}$, and index to middle, $y_{sep}$, in 5 mm "
        "steps. <b>Object size.</b> The diameter of the twist's and the roll's sphere, or of the press's cylinder. "
        "<b>Plant.</b> The CAD-model SR2 hand with the v4b fingertip as 1 mm sphere pads at 2.5 MPa, the SCS0009 "
        "servo model (0.72 N m/rad, at most 0.18 N m) and HOM at a 0.01 s period. Every rollout here ran on the CPU "
        "with every contact, at seeds 0, 1 and 2.</p>"
        "<p><b>Held.</b> The object ended within 15 mm of its start height with at least two fingers loaded.</p>"
        "<p><b>Earlier score.</b> The score of the landscapes up to this page "
        "(<code>hom_landscape.rollout_score</code>), 0 when not held. Twist: the largest yaw of the sphere at any "
        "instant with all three pads loaded, divided by 90 deg and capped at 1. Roll: $1/(1+r)$ with $r$ the RMS "
        "error of the angular velocity divided by the 40 deg/s step. Press: 1 minus the RMS pinch-angle error over "
        "the 20 deg stroke, at least 0.</p>"
        "<p><b>Tracking error</b> $e(t)$, in deg: the angle between the object's orientation and the commanded one. "
        "For a sphere, both orientations are integrated from the measured and the commanded angular velocities in the "
        "world frame. For a cylinder, the angle is taken between its axis and the start axis turned by the commanded "
        "pinch angle. <b>Commanded rotation</b> $\\theta(t)$: the angle of the commanded orientation from the task's "
        "start. Both are taken at every control period of the task.</p>"
        "<p><b>Three-pad share</b> $c_3$: the share of the task's control periods in which all three pad forces "
        "exceed 0.05 N, the twist's contact threshold.</p>"
        "<p><b>Tracking score</b> (<code>hom_landscape.track_score</code>):</p>"
        "$$s = \\mathbb{1}[\\mathrm{held}]\\;\\max\\!\\Big(0,\\; 1 - \\frac{e_{\\mathrm{rms}}}{\\theta_{\\mathrm{rms}}}"
        "\\Big)\\; c_3,$$"
        "<p>with RMS values over the task. The middle factor is the share of the commanded rotation the object "
        "followed. It is 1 when the object follows the command and 0 when it does no better than staying still. The "
        "score falls linearly with the RMS error and with the time lost on two pads. A layout's tracking score is its "
        "mean over the three seeds.</p>")


def sec_floors(rows):
    ref = {r["task"]: r["track_ref_rms_deg"] for r in rows if r.get("track_ref_rms_deg")}
    body = [["Twist", "largest yaw on three pads / 90 deg", "0", f"{ref['twist']:.1f}"],
            ["Roll", "1 / (1 + velocity RMSE / 40 deg/s)", "0.50", f"{ref['roll']:.1f}"],
            ["Pin-push", "1 - angle RMSE / 15 deg", f"{still_pinpush():.2f}", ""],
            ["Press", "1 - angle RMSE / 20 deg", f"{1.0 - ref['press'] / 20.0:.2f}", f"{ref['press']:.1f}"]]
    return (
        "<p>Three of the four earlier scores give a large share of their range to an object that is held but does not "
        "move. A held, motionless sphere scores 0.50 on the roll. A held, motionless cylinder scores "
        f"{still_pinpush():.2f} on the pin-push and {1.0 - ref['press'] / 20.0:.2f} on the press (Table 1). The "
        "roll's per-size picks of the first films scored 0.55 to 0.58, so they rolled barely better than a still "
        "sphere. The twist's score counts one instant: the largest yaw reached while three pads touch. A sphere that "
        "whirls about other axes and passes 90 deg of yaw once scores 1.</p>"
        + table(["Task", "Earlier score", "Held still", "RMS commanded rotation (deg)"], body,
                "The earlier scores and their values for an object held still. The last column is "
                "$\\theta_{\\mathrm{rms}}$ of the tracking score.", num_from=2))


def sec_check(rows):
    lab = [r for r in rows if str(r.get("set", "")).startswith("labels_")]
    body = []
    for r in lab:
        kind = r["set"].split("_")[1]
        body.append([TNAME[r["task"]], f"{r['size']:g}", "per-size pick" if kind == "best" else "(70, 110)",
                     f"({r['layout'][0]:.0f}, {r['layout'][1]:.0f})", f2(r["old"]),
                     f2(max(0.0, 1.0 - r["track_err_rms_deg"] / r["track_ref_rms_deg"])
                        if r.get("track_err_rms_deg") is not None else None),
                     f2(r.get("contact_all_frac")), f2(r["track"]), VERDICT.get((r["task"], float(r["size"]), kind), "")])
    good = [r["track"] for r in lab if VERDICT.get((r["task"], float(r["size"]), r["set"].split("_")[1])) == "works"]
    bad = [r["track"] for r in lab if "loses" in VERDICT.get((r["task"], float(r["size"]), r["set"].split("_")[1]), "")]
    r30, r63 = (next(r for r in lab if r["task"] == "roll" and r["set"] == "labels_best" and float(r["size"]) == s)
                for s in (30.0, 63.0))
    return (
        "<p>The 23 filmed rollouts of the first films were run again at seed 0 and scored both ways (Table 2). The "
        "owner judged the clips on 2026-11-09: \"in sphere roll, only the 63mm clip/layout works, the rest lose a "
        "finger contact and oscillate a bunch\"; \"in twist, the smaller two options lose finger contacts and "
        "oscillate. honestly the generalist looks better\"; \"press is good\". The tracking score orders every one of "
        f"these verdicts. The clips judged to lose fingers score {min(bad):.2f} to {max(bad):.2f}, and the roll that "
        f"works scores {max(good):.2f}. On the earlier score the failing twists scored 0.87 and 1.00.</p>"
        "<p>The three-pad share is needed for the roll. Without it, the 30 mm roll that loses a finger follows "
        f"{r30['track_nc']:.2f} of the commanded rotation and outranks the 63 mm roll that works, at "
        f"{r63['track_nc']:.2f}. With it, the 30 mm roll keeps three pads for {100 * r30['contact_all_frac']:.0f} % of "
        f"the task and scores {r30['track']:.2f}.</p>"
        + table(["Task", "Size (mm)", "Hand", "Layout (mm)", "Earlier score", "Rotation followed", "Three-pad share",
                 "Tracking score", "Owner's verdict"], body,
                "The filmed rollouts of the first films, seed 0, scored both ways. Rotation followed is the middle "
                "factor of the tracking score.", num_from=4))


def sec_picks(picks):
    out = ["<p>Each size's layout was picked again by the tracking score. The stored GPU rows of the live EA at that "
           "size hold no orientations, so a short list was taken from them first. It held the eight candidates "
           "closest to the command by the stored metrics, the first films' pick and layout (70, 110) mm. Each "
           "candidate ran at seeds 0 to 2 on the CPU, and the best mean was filmed at the seed closest to that mean (Table 3).</p>"]
    body = []
    first = json.loads((ROWS / "films_v4b.json").read_text())
    for task in TASKS:
        for s, v in sorted(picks.get(task, {}).items(), key=lambda kv: float(kv[0])):
            f0 = first[f"sizes4_{task}"][s]
            old = next(c for c in v["candidates"] if (c["call"], c["grasp"]) == (f0["call"], f0["grasp"]))
            body.append([TNAME[task], s, f"({v['layout'][0]:.0f}, {v['layout'][1]:.0f})", f2(v["track_mean"]),
                         f2(v["old_mean"]), f2(old["track"]), f2(v["generalist_track_mean"])])
    out.append(table(["Task", "Size (mm)", "Picked layout (mm)", "Tracking score", "Earlier score",
                      "First films' pick, tracking", "(70, 110), tracking"], body,
                     "Layouts picked by the mean tracking score over seeds 0-2 on the CPU, against the first films' "
                     "pick and layout (70, 110) mm on the same rollouts.", num_from=1))
    for task in TASKS:
        out.append(P.film(MEDIA / f"20261109-film_track_{task}{TAG}.mp4",
                          f"{TNAME[task]}: the layout picked by the tracking score at each of four {OBJ[task]} sizes, at the "
                          "seed closest to its mean."))
        out.append(P.film(MEDIA / f"20261109-film_one_{task}{TAG}.mp4",
                          f"{TNAME[task]}: layout (70, 110) mm at the same sizes, seed 0. At 13 mm the press shows the "
                          "layout's second grasp, since the first one's rollout diverged; at 50 mm the toy found no "
                          "press pose." if task == "press" else
                          f"{TNAME[task]}: layout (70, 110) mm at the same sizes, seed 0."))
    return "".join(out)


def sec_limits():
    return ("<p>The short lists came from GPU rows without the GPU's orientations, ranked by the stored end yaw "
            "(twist), the stored per-step angles (roll) and the stored angle RMSE (press). A layout outside the eight "
            "may score higher. Three seeds per candidate were run, on the CPU only. The landscapes, searches and "
            "layout choices of the earlier pages, the task-conflict page's 0.84 among them, still use the earlier "
            "score.</p>")


def sec_future():
    return ("<ul>"
            "<li>New rollouts on the CPU and the GPU record <code>track_err_rms_deg</code>, "
            "<code>track_ref_rms_deg</code> and <code>contact_all_frac</code> (<code>hom_tasks.metrics</code>, about "
            "20 ms per rollout). Rank the post-reset grids by the tracking score and quote the earlier score beside "
            "it. If the grids' Spearman correlation between the two scores is above 0.9 on a task, that task's earlier "
            "landscapes stand.</li>"
            "<li>Rescore the task-conflict page with the tracking score and restate its single-layout level, which is "
            "0.84 on the earlier score.</li>"
            "<li>Run the live EA at these sizes with the tracking score as its fitness. If its best layouts do not "
            "beat the short-list picks by more than the seed spread, the short lists were enough.</li>"
            "</ul>")


def sec_files():
    return ("<ul><li><code>scripts/hom/hom_tasks.py</code> (<code>track_errors</code>, the new fields of "
            "<code>metrics</code>), <code>scripts/hom/hom_landscape.py</code> (<code>track_score</code>)</li>"
            "<li><code>scripts/hom/hom_track_films.py</code>: <code>labels</code>, <code>rerun TASK</code>, "
            "<code>film TASK</code>; rows <code>rows/20261109-track_rerun_sv25.jsonl</code> and "
            "<code>rows/20261109-track_picks_sv25.json</code></li>"
            "<li>Films: <code>media/20261109-film_track_*_sv25.mp4</code>, clips "
            "<code>media/20261109-clip_track_*_sv25.mp4</code>; layout (70, 110) mm: "
            "<code>media/20261109-film_one_*_sv25.mp4</code></li>"
            f"<li>Builder: <code>scripts/hom/hom_track_score_page.py</code>. Related: <a href=\"{SEARCH}\">the search "
            f"methods page</a>, <a href=\"{CONFLICT}\">the task-conflict page</a>.</li></ul>")


def lede(rows, picks):
    rng = {t: [v["track_mean"] for v in picks.get(t, {}).values()] for t in TASKS}
    gen = {t: [v["generalist_track_mean"] for v in picks.get(t, {}).values() if v["generalist_track_mean"] is not None]
           for t in TASKS}
    span = (lambda xs: "" if not xs else f"{min(xs):.2f}" if f"{min(xs):.2f}" == f"{max(xs):.2f}" else
            f"{min(xs):.2f} to {max(xs):.2f}")
    text = (
        "The earlier HOM task scores did not follow the motion seen in the films. A held, motionless sphere scored "
        "0.50 on the roll, and two whirling spheres scored 0.87 and 1.00 on the twist. The tracking score replaces "
        "them: the share of the commanded rotation the object followed, by RMS angle, times the share of the task on "
        "three pads. It agreed with every verdict of the owner on the 23 filmed rollouts. Layouts re-picked by it at "
        f"four object sizes per task score {span(rng['twist'])} on the twist, {span(rng['roll'])} on the roll and "
        f"{span(rng['press'])} on the press. Layout (70, 110) mm scores {span(gen['twist'])}, {span(gen['roll'])} and "
        f"{span(gen['press'])} at the same sizes.")
    summary = (f"A tracking score (share of the commanded rotation followed, times the share of the task on three pads) "
               f"replaces the HOM task scores, which gave 0.50 to a still sphere on the roll and 1.00 to a whirling "
               f"twist; re-picked layouts score {span(rng['twist'])} (twist), {span(rng['roll'])} (roll) and "
               f"{span(rng['press'])} (press) at four sizes each.")
    return text, summary


def main():
    rows, picks = load()
    sections = [("terms", "Terms and metrics", sec_terms()),
                ("floors", "The earlier scores on a still or whirling object", sec_floors(rows)),
                ("check", "The tracking score against the owner's verdicts on the films", sec_check(rows)),
                ("picks", "Layouts re-picked by the tracking score at four object sizes", sec_picks(picks)),
                ("limits", "Limitations", sec_limits()),
                ("future", "Future work", sec_future()),
                ("files", "Files", sec_files())]
    lede_t, summary = lede(rows, picks)
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|N\b|s\b)", "\\1&nbsp;\\2", s_))
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{unit(b)}</section>'
                     for k, (sid, hd, b) in enumerate(sections, 1))
    byline = ("CPU rollouts on the SCS0009 plant at 2.5 MPa, 2026-11-09. Follows <a href=\"" + SEARCH + "\">the search "
              "methods page</a>. Builder: <code>scripts/hom/hom_track_score_page.py</code>.")
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": "HOM tracking score", "H1": "Tracking score of the HOM tasks and films of the SR2 hand across "
                 "object sizes", "EYEBROW": "HOM layout study, 2026-11-09", "LEDE": unit(lede_t), "BYLINE": byline,
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS + texsvg.BLOCK_CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    n = []
    page = P.render(page, n)
    page = docs_index.tag(page, topic="design", summary=summary, date="2026-11-09")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {n[0]} formulas", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
