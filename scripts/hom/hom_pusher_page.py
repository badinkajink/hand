#!/usr/bin/env python3
"""Builds docs/experiments/20261110-hom_pusher_estimate/20261110-hom_unsensed_pusher.html: the pin-push of the SR2 hand
under HOM with the pusher's contact unsensed and estimated (owner, 2026-11-09, an exception to the pin-push rule,
exploratory: "push with the unsensorized back of the pusher's fingertip. Estimate where and how hard the back of the
pusher touches the cylinder from the pusher's joint kinematics and the sensed thumb and index tips").

Data: rows/20261110-pusher_estimate.jsonl (scripts/hom/hom_pusher_estimate.py); the film in media/.

    .venv/bin/python scripts/hom/hom_pusher_page.py
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

D = ROOT / "docs/experiments/20261110-hom_pusher_estimate"
OUT = D / "20261110-hom_unsensed_pusher.html"
ROWS = D / "rows" / "20261110-pusher_estimate.jsonl"
MEDIA = D / "media"
TEX_CACHE = D / "texsvg_cache.json"
FPR = "../20261029-foundational_pose_review/"
HOMREV = "../20261029-hom_control_review/20261029-hom_control_and_fingertip_sensing.html"
TRACK = FPR + "20261109-hom_tracking_score.html"
CAL = "../20261026-fingertip_calibration/20261026-fingertip_calibration_on_the_ur5.html"
VARS = ("hold", "tip0", "tip0_pe", "tip", "tip_pe", "tipfl", "tip_pefl")
VNAME = {"hold": "Exact sensing", "tip0": "Boards' noise", "tip0_pe": "Boards' noise, pusher estimated",
         "tip": "Boards, 50 Hz, late", "tip_pe": "Boards, 50 Hz, late, pusher estimated",
         "tipfl": "Boards, 50 Hz, late, force loop", "tip_pefl": "Boards, 50 Hz, late, force loop on the estimate"}
table, esc = P.table, P.esc


def load():
    return [json.loads(x) for x in ROWS.read_text().splitlines() if x.strip()]


def f2(v):
    return "&#8211;" if v is None else f"{v:.2f}"


def stats(rs):
    held = [r for r in rs if r.get("held")]
    med = (lambda k: float(np.median([r[k] for r in held if r.get(k) is not None]))
           if any(r.get(k) is not None for r in held) else None)
    return {"n": len(rs), "held": len(held), "track": float(np.mean([r["track"] for r in rs])),
            "old": float(np.mean([r["old"] for r in rs])), "theta": med("theta_rmse_deg"),
            "loc": med("pusher_loc_err_mm"), "ferr": med("pusher_force_err_N"), "ftrue": med("pusher_force_true_N")}


def sec_terms():
    terms = [
        ("Pin-push", "the HOM task of the SR2 hand in which the thumb and the index pinch a 25 mm cylinder, 100 mm long. "
         "The middle finger, the pusher, turns it about the pinch axis by friction, 15 deg up and down "
         f"(<a href=\"{TRACK}\">tracking page</a>). The owner set the pin-push aside on 2026-11-09 and asked for this "
         "study as an exception."),
        ("Layouts", "the five best layouts of the pin-push grid on the SCS0009 plant at 2.5 MPa by the earlier score, "
         "and layout (70, 110) mm, each with its best grasp. A layout is written ($x_{sep}$, $y_{sep}$) in mm."),
        ("Plant", "the CAD-model SR2 hand with the v4b fingertip as 1 mm sphere pads at 2.5 MPa. The SCS0009 servo "
         "model has 0.72 N m/rad and 0.18 N m. The rollouts ran on the CPU with every contact, at seeds 0 to 4, and the "
         "cylinder is held at its start until the grasp closes."),
        ("Boards", "the fingertips' magnetic boards as calibrated on the UR5 "
         f"(<a href=\"{CAL}\">calibration page</a>). The contact point has an error of 0.4 mm (sigma) at 1.5 N and "
         "more, 0.7 mm from 0.5 N and 1.3 mm below. The normal force is within 5 % with 0.07 N of noise, and no contact "
         "is shown under 0.25 N (<code>hom_tasks.TIP_SIGMA_MM</code>, <code>TIP_NOISE_N</code>)."),
        ("Sensing variants", "exact sensing: the simulator's contacts at 100 Hz. Boards' noise: every fingertip read by "
         "its board at 100 Hz, with the object pose exact. Boards, 50 Hz, late: the same boards at 50 Hz with every "
         "measurement 40 ms late. The tag tracker's noise of 0.5 mm and 1 deg is then on the object pose, as on the HOM "
         "review page. Pusher estimated: the pusher's board replaced by the estimate of Section SEC[estimate]. Force "
         "loop: the pusher's commanded approach speed corrected by 10 mm/s per N of force error."),
        ("Location error", "the distance between the estimated contact point and the simulator's centre of pressure of "
         "the pusher's pads, in mm. A rollout's value is its median over the control periods in which the pusher's pad "
         "force exceeds 0.05 N."),
        ("Force error", "the absolute difference between the estimated normal force and the simulator's, in N; a "
         "rollout's median over the same periods. True force: the simulator's normal force, the same median."),
        ("Tracking score", "the share of the commanded rotation the cylinder followed, times the share of the task on "
         f"three pads, 0 when it is dropped (<a href=\"{TRACK}\">tracking page</a>). Earlier score: 1 minus the RMS "
         "pinch-angle error over the 15 deg stroke."),
        ("Held", "the cylinder ended within 15 mm of its start height with at least two fingers loaded."),
    ]
    return "<dl>" + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"


def sec_estimate():
    return (
        "<p>The pusher touches the cylinder where no board senses it. Its contact point is estimated from its measured "
        "joint angles and the tracked pose of the cylinder. The estimate is the witness point of the pusher's "
        "fingertip, the point of the tip closest to the cylinder. HOM computes the same point for a finger without "
        "contact (<code>hom.Mirror.gcf_witness</code>). The normal force comes from the cylinder's balance. The normal "
        "forces $F_t$ and $F_i$ and the contact points of the thumb and the index are read from their boards. The "
        "directions $x_t$ and $x_i$ in which these pads push follow from the contact points. With the pusher's "
        "direction $x_p$ at its estimated point and the cylinder's weight $m g$,</p>"
        "$$\\hat F_p = \\max\\big(0,\\; -(F_t\\,x_t + F_i\\,x_i + m\\,g)\\cdot x_p\\big).$$"
        "<p>Friction at the pinch is left out; its components along $x_p$ vanish when the pinch pads' normals lie "
        "along the pinch axis. The estimate replaces the pusher's reading in HOM: its contact frame, its contact test "
        "and, with the force loop, its force error (<code>hom_tasks.pusher_estimate</code>, variants "
        "<code>tip_pe</code> and <code>tip_pefl</code>).</p>"
        "<p>The grasps place the pusher's pad on the cylinder, as in every pin-push grasp of the pose toy. Its reading "
        "is withheld, as the back of the SR2 fingertip has no board. The witness point is therefore taken on the pad's "
        "surface, and the pad's compliance is in the contact.</p>")


def sec_results(rows):
    S = {v: stats([r for r in rows if r["variant"] == v]) for v in VARS}
    S = {v: s for v, s in S.items() if s["n"]}
    body = [[VNAME[v], f"{s['held']} of {s['n']}", f2(s["track"]), f2(s["old"]), f2(s["theta"]), f2(s["loc"]),
             f2(s["ferr"]), f2(s["ftrue"])] for v, s in S.items()]
    ex, b0, be = S.get("hold"), S.get("tip0"), S.get("tip0_pe")
    lt, lte = S.get("tip"), S.get("tip_pe")
    t = ("<p>The contact point was estimated well and the force poorly (Table 1). With the boards' noise alone the "
         f"location error was a median {be['loc']:.2f} mm. The force error was {be['ferr']:.2f} N, against a true "
         f"pusher force of {be['ftrue']:.2f} N. The pinch forces are about 3 N, and each board's gain error of 5 % "
         "(one standard deviation) alone puts about 0.2 N into their difference.</p>"
         f"<p>With exact sensing the pin-push held the cylinder in {ex['held']} of {ex['n']} rollouts and scored "
         f"{ex['track']:.2f}. With the boards' noise it held it in {b0['held']} and scored {b0['track']:.2f}, and with "
         f"the pusher estimated as well in {be['held']} and {be['track']:.2f}. At 50 Hz with the measurements late, the "
         f"cylinder was held in {lt['held']} of {lt['n']} rollouts with boards on all three fingers ({lt['track']:.2f}). "
         f"With the pusher estimated it was held in {lte['held']} ({lte['track']:.2f}). Most cylinders were dropped "
         "with the force loops, on the pusher's board and on the estimate, which scored 0.00.</p>")
    t += table(["Sensing", "Held", "Tracking score", "Earlier score", "Pinch-angle RMSE (deg)", "Location error (mm)",
                "Force error (N)", "True force (N)"], body,
               "The pin-push on six layouts at seeds 0-4 per sensing variant. Scores are means over the 30 rollouts; the "
               "angle and the estimate's errors are medians over the held rollouts.", num_from=1)
    lays = sorted({tuple(r["layout"]) for r in rows})
    body = []
    for lay in lays:
        cells = []
        for v in ("hold", "tip0", "tip0_pe", "tip", "tip_pe"):
            rs = [r for r in rows if tuple(r["layout"]) == lay and r["variant"] == v]
            cells.append(f"{np.mean([r['track'] for r in rs]):.2f} ({sum(bool(r.get('held')) for r in rs)})"
                         if rs else "&#8211;")
        body.append([f"({lay[0]:.0f}, {lay[1]:.0f})", *cells])
    t += table(["Layout (mm)", "Exact sensing", "Boards' noise", "Boards' noise, pusher estimated", "Boards, 50 Hz, late",
                "Boards, 50 Hz, late, pusher estimated"], body,
               "Tracking score per layout, mean over seeds 0-4, with the rollouts held of 5 in brackets.", num_from=1)
    return t


def sec_film():
    return P.film(MEDIA / "20261110-film_pusher_estimate.mp4", "The pin-push on layout (75, 55) mm, seed 0: exact "
                  "sensing, the boards' noise, and the boards' noise with the pusher estimated. "
                  "<code>hom_pusher_estimate.py film</code>.")


def sec_limits():
    return ("<p>The contact is the pusher's pad, read by no board; a contact on the back of the fingertip, a rigid "
            "printed part, was not simulated. The force balance omits the pinch's friction and the cylinder's "
            "acceleration. Six layouts at five seeds were run, on the CPU only.</p>")


def sec_future():
    return ("<ul>"
            "<li>A pin-push grasp with the back of the pusher's fingertip on the cylinder. It needs a pose-toy constraint "
            "that puts the cap's outer surface on the cylinder at the lever; then rerun <code>hom_pusher_estimate.py "
            "run</code> on those grasps. The estimate is good enough for the back if its location error stays under "
            "1 mm.</li>"
            "<li>The pusher's force from its own servo: the deflection of its joints from their targets times the "
            "servo's stiffness, through the pad's Jacobian, beside the balance. It is better if its force error falls "
            "under half the balance's 0.26 N.</li>"
            "<li>The cylinder was dropped on five of the six layouts with the boards at 50 Hz and 40 ms late. Rerun "
            "<code>tip</code> with the delay alone and with the rate alone (variants of <code>hom_tasks.VARIANTS</code>) "
            "to find which of the two drops it.</li>"
            "</ul>")


def sec_files():
    return ("<ul><li><code>scripts/hom/hom_pusher_estimate.py</code>: <code>run</code>, <code>report</code>, "
            "<code>film</code>; rows <code>rows/20261110-pusher_estimate.jsonl</code></li>"
            "<li><code>scripts/hom/hom_tasks.py</code>: <code>pusher_estimate</code>, the variants <code>tip_pe</code> "
            "and <code>tip_pefl</code>, and the rows' <code>pusher_est</code> fields</li>"
            "<li>Builder: <code>scripts/hom/hom_pusher_page.py</code></li></ul>")


def lede(rows):
    S = {v: stats([r for r in rows if r["variant"] == v]) for v in VARS}
    ex, b0, be, lt, lte = S["hold"], S["tip0"], S["tip0_pe"], S["tip"], S["tip_pe"]
    text = (f"The unsensed pusher's contact point in the SR2 pin-push was estimated within a median {be['loc']:.2f} mm "
            f"of the simulator's, and its force within {be['ferr']:.2f} N of {be['ftrue']:.2f} N. The point comes from "
            "the pusher's joint angles and the tracked cylinder, the force from the cylinder's balance with the thumb's "
            f"and the index's boards. On six layouts the pin-push scored {ex['track']:.2f} with exact sensing. With the "
            f"boards' noise it scored {b0['track']:.2f}, and with the pusher estimated as well {be['track']:.2f}. At "
            f"50 Hz with every measurement 40 ms late, the cylinder was held in {lt['held']} of {lt['n']} rollouts, and "
            f"in {lte['held']} with the estimate.")
    summary = (f"The SR2 pin-push with the pusher unsensed: its contact point estimated within {be['loc']:.2f} mm and "
               f"its force within {be['ferr']:.2f} N of {be['ftrue']:.2f} N; tracking score {ex['track']:.2f} with exact "
               f"sensing, {b0['track']:.2f} with the boards' noise and {be['track']:.2f} with the estimate.")
    return text, summary


def render(page_text, holder):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page_text)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    holder.append(len(items))
    return P.TEX_RE.sub(lambda m: next(svgs), page_text)


def main():
    rows = load()
    sections = [("terms", "Terms and metrics", sec_terms()),
                ("estimate", "The pusher's contact from kinematics and the cylinder's balance", sec_estimate()),
                ("results", "The pin-push with the estimate in place of a pusher board", sec_results(rows)),
                ("film", "Film of one layout under three kinds of sensing", sec_film()),
                ("limits", "Limitations", sec_limits()),
                ("future", "Future work", sec_future()),
                ("files", "Files", sec_files())]
    lede_t, summary = lede(rows)
    num = {sid: k for k, (sid, _h, _b) in enumerate(sections, 1)}
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|N\b|s\b)", "\\1&nbsp;\\2", s_))
    sections = [(sid, hd, unit(re.sub(r"SEC\[(\w+)\]", lambda m_: str(num[m_.group(1)]), b))) for sid, hd, b in sections]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{b}</section>' for k, (sid, hd, b) in enumerate(sections, 1))
    byline = ("CPU rollouts with every contact, 2026-11-10, on the SCS0009 plant at 2.5 MPa. Follows <a href=\"" + HOMREV
              + "\">the HOM review page</a>, whose sensing variants are reused. Builder: "
                "<code>scripts/hom/hom_pusher_page.py</code>.")
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": "Unsensed pusher in the pin-push", "H1": "Contact estimation for the unsensed pusher of the "
                 "SR2 hand's pin-push under HOM", "EYEBROW": "HOM control study, 2026-11-10", "LEDE": unit(lede_t),
                 "BYLINE": byline, "TOC": toc, "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    n = []
    page = render(page, n)
    page = docs_index.tag(page, topic="mechanisms", summary=summary, date="2026-11-10")
    page = retro_style.apply(page)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {n[0]} formulas", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
