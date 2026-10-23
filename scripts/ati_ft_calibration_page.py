#!/usr/bin/env python3
"""Page of 2026-10-23: a plan to calibrate the lab's uncalibrated ATI Mini45 on the UR5 with known masses, to replace the
HEX-E whose link fails (owner: "the only way to permanently fix the HEX-E robot is to calibrate one of our uncalibrated ATI
sensors ... we can probably calibrate the ATI with just a 3dp mount to the HEX-E, with the current indenter setup").

Reads docs/experiments/20261023-ati_ft_calibration/ (the excitation simulation of scripts/ati_calibration_sim.py, the fit's
self-test of scripts/ati_calibration_fit.py, the printed parts of scripts/ati_stack_mount.py and their renders) and writes
docs/experiments/20261023-ati_ft_calibration/20261023-ati_ft_calibration_plan.html.

    .venv/bin/python scripts/ati_ft_calibration_page.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import v2_tip_fe_checks_page as VP  # noqa: E402

D = ROOT / "docs/experiments/20261023-ati_ft_calibration"
PAGE = D / "20261023-ati_ft_calibration_plan.html"
TEMPLATE = ROOT / "scripts/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
CAL = ROOT / "docs/experiments/20261022-ur5_tip_calibration"
HANDOFF = Path.home() / ".claude-accounts/colorado/uploads/ec307d8f-8a52-45e1-992b-f83a6be75e2a/d354d3cb-ATI_Mini40_Mini45_Calibration_Handoff.md"
table, img_uri, esc = VP.table, VP.img_uri, VP.esc


def rel(p):
    return os.path.relpath(p, D)


def link(p, text):
    return f'<a href="{rel(p)}">{text}</a>'


def code(p):
    return f"<code>{esc(os.path.relpath(p, ROOT) if isinstance(p, Path) else p)}</code>"


def fig(n, png, caption, width=1100):
    return (f'<figure id="fig{n}"><img src="{img_uri(png, width)}" alt="{esc(caption.split(".")[0])}">'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def f3(v):
    return ", ".join(f"{x:.3f}" for x in v)


def sections():
    sim = json.loads((D / "20261023-ati_calibration_sim.json").read_text())
    st = json.loads((D / "20261023-ati_fit_selftest.json").read_text())
    parts = json.loads((D / "20261023-ati_stack_parts.json").read_text())
    cur = sim["curves"]
    full = st["fits"]["full"]["median"]
    sv = sim["singular_values_scaled"]
    lede = (f"The HEX-E's link to its box failed in every run of 22 October, and its own gains were off (Fz 0.78 "
            f"of the other axes in one fit against gravity). The lab's ATI Mini45 (FT11452) and Mini40 (FT16552) have no factory calibration. "
            f"In a simulation on the presses of the UR5 run sphere2, the presses alone excite three of the six wrench "
            f"directions, since every press acts at nearly the same lever. Fitted to them, the ATI is off on other wrenches by up "
            f"to {max(cur['presses, HEX-E reference'][-1]['general_median_abs'][:3]):.1f} N and "
            f"{max(cur['presses, HEX-E reference'][-1]['general_median_abs'][3:]) * 1e3:.0f} N·mm. Forty poses of a "
            f"weighed 0.5 kg mass at five hook points excite all six. They recover a simulated Mini45 to "
            f"{max(full[:3]):.2f} N and {max(full[3:]) * 1e3:.1f} N·mm, about its resolution. Printed parts put the ATI "
            f"between the HEX-E and mount v3; for the masses a hook cross goes in the fork's place. The Mini45's hole "
            f"clocking, thread depth and height are unmeasured, so the parts are drafts.")
    summary = (f"ATI Mini45 calibration plan: presses alone excite 3 of 6 wrench directions (fit off by up to "
               f"{max(cur['presses, HEX-E reference'][-1]['general_median_abs'][:3]):.1f} N on general wrenches); 40 poses of a weighed 0.5 kg mass at 5 hook points recover a simulated Mini45 to "
               f"{max(full[:3]):.2f} N, {max(full[3:]) * 1e3:.1f} N mm; printed stack and hook cross drafted (ATI hole "
               f"clocking unmeasured); Net F/T logger tested on a simulated box.")
    S = []
    terms = [("Wrench $w$", "Force and torque at the ATI's origin: $F_x, F_y, F_z$ in N and $T_x, T_y, T_z$ in N·m. The "
              "origin is taken at the centre of the ATI's mounting face."),
             ("Gage values $g$", "The six numbers the Net F/T box sends in its raw-gage mode: uncalibrated strain gage "
              "values, minus their value at rest."),
             ("Calibration matrix $C$", "The 6 x 6 matrix of $w = C\\,(g - g_0)$, fitted here. Only its row shapes can be taken "
              "from the factory matrices of other Mini45s."),
             ("Excitation", "How well the reference wrenches span the six directions of $w$, measured by the singular values "
              "of the stacked wrenches with torques divided by 0.1 m. A small value is a direction the data barely test."),
             ("Error on general wrenches", "The median absolute error of a fitted $C$ on wrenches it was not fitted on: a "
              "0.3 kg mass at random points and orientations, and random forces at the origin.")]
    S.append(("terms", "Terms and metrics", '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"))

    rows = [["Mini45, FT11452", "9105-TW-Mini45, US-120-160 as received", "None for this serial; FT31565 (Mini45, TWE) and "
             "FT32816 (Mini45, Net F/T) give row shapes"],
            ["Mini40, FT16552", "9105-TW-Mini40, SI-20-1", "None for this serial; FT21031 (Mini40, MUX) gives row shapes"],
            ["Net F/T box, FT35796", "9105-NETBA, with an analog front end", "Its calibration slots are unread"]]
    body = ("<p>The owner's handoff note (October 2026) lists the hardware and the factory matrices recovered for other "
            "serials of the same models. The note's own warning applies: a borrowed matrix maps another unit's signals "
            "through other electronics. Here those matrices are used to check the gage order and signs, and in the "
            "six- and twelve-parameter models below.</p>"
            + table(["Sensor", "Hardware", "Calibration"], rows, caption="The lab's ATI hardware (from the RMA form in the handoff "
                    "note).", n=1)
            + "<p>Before any load, the box's present state is saved as found: every web page, its netftapi2.xml, the "
            "selected calibration slot and its serial. Then the factory raw-gage slot (#16, FT00000, ATI Quick Start "
            "section 3.4.2) is selected by hand in the web page, so that the box sends gage values.</p>")
    S.append(("sensors", "The lab's ATI sensors and what is known", body))

    p0, ph = cur["presses, HEX-E reference"][-1], cur["presses (HEX-E) and 24 gravity poses (HEX-E)"][-1]
    pm = cur["presses (HEX-E) and 5 hook points x 8 poses (known mass)"][-1]
    mo = cur["5 hook points x 8 poses (known mass) alone"][-1]
    rows = []
    for lab, r in (("Presses, HEX-E as reference", p0), ("Presses and one hook point, HEX-E as reference", ph),
                   ("Presses (HEX-E) and five hook points (known mass)", pm), ("Five hook points (known mass) alone", mo)):
        rows.append([lab, f3(r["press_median_abs"][:3]), ", ".join(f"{x * 1e3:.1f}" for x in r["press_median_abs"][3:]),
                     f3(r["general_median_abs"][:3]), ", ".join(f"{x * 1e3:.1f}" for x in r["general_median_abs"][3:])])
    body = (f"<p>The first idea is to calibrate the ATI against the HEX-E on the sphere presses of the existing plan. "
            f"Sphere2's press holds were moved to an ATI origin in the stack below. A simulated Mini45 was fitted against the "
            f"HEX-E's wrench. Its gages were up to 10 % off the FT31565 shape. The HEX-E had its data-sheet noise and a 3 % "
            f"gain error per axis ({code(ROOT / 'scripts/ati_calibration_sim.py')}).</p>"
            f"<p>Every press acts about 80 mm from the ATI, so each torque is nearly a fixed multiple of its force. The scaled "
            f"singular values are {', '.join(f'{v:.3g}' for v in sv[:3])} for three directions and "
            f"{', '.join(f'{v:.2g}' for v in sv[3:])} for the others. Torsion about the tool axis reaches only {sim['excitation']['range_N_Nm'][5] * 1e3:.0f} N·mm. A fit to the "
            f"presses reproduces held-out presses but misses general wrenches (Table 2). A known mass hung at five points "
            f"60 mm out along x and y and 100 mm up the axis fills the missing directions. With the mass as its own "
            f"reference, the HEX-E's gain errors stay out of the fit.</p>"
            + table(["Data in the fit", "Presses: F (N)", "Presses: T (N·mm)", "General: F (N)", "General: T (N·mm)"], rows,
                    num_cols=(1, 2, 3, 4), caption="Median absolute error of the fitted ATI on held-out presses and on general "
                    "wrenches, 59 training presses, 20 simulated sensors.", n=2))
    S.append(("excitation", "Excitation from the existing presses", body))

    rows = [[m, f3(st["fits"][k]["median"][:3]), ", ".join(f"{x * 1e3:.2f}" for x in st["fits"][k]["median"][3:])]
            for k, m in (("full", "Full 6 x 6, ridge"), ("rows12", "Row shapes, six output and six gage gains"),
                         ("rows6", "Row shapes, six output gains"))]
    body = (f"<p>The fit uses 40 poses: the 0.5 kg mass at each of the five hook points, at eight tool orientations each. The "
            f"design's scaled singular values are {', '.join(f'{v:g}' for v in st['design'])}, so all six directions are "
            f"excited. The simulated Mini45 has its gages up to 10 % off the FT31565 shape and noise at its stated resolution "
            f"(1/16 N, 1/752 N·m). The full matrix recovers held-out wrenches to that resolution (Table 3). The "
            f"twelve-parameter model is nearly as good. The six-parameter model is not, because the gages differ from the "
            f"borrowed shape ({code(ROOT / 'scripts/ati_calibration_fit.py')} --selftest).</p>"
            f"<p>The reference wrench of a pose is $w = [m\\,g_S,\\; r \\times m\\,g_S]$, where $g_S$ is gravity in the ATI's "
            f"frame from the robot's joints and $r$ is the hook point. The tool's own weight and the gages' zero are the same "
            f"at a given orientation with the mass on and off. Taking that difference cancels both, so each orientation is "
            f"recorded twice. The HEX-E logs as well, and the same differences calibrate its gains against the mass, the "
            f"open question of QUEUE.md R5.</p>"
            + table(["Model", "F error (N)", "T error (N·mm)"], rows, num_cols=(1, 2),
                    caption="Fit to 40 known-mass poses, scored on a 0.3 kg mass at other points and orientations (medians).", n=3))
    S.append(("masses", "Calibration with known masses", body))

    pr = parts["parts"]
    rows = [["ATI base", f"{parts['stack_mm']['base']:.0f} mm, Ø63", "Over v3's M6 heads; four M3 x 20 into v3's adapter inserts; "
             "six M3 x 10 up into the ATI, heads 12 mm up in counterbores"],
            ["ATI Mini45", f"{parts['stack_mm']['ati']:.1f} mm, Ø45", "Stand-in: height, hole clocking and thread depth unmeasured"],
            ["ATI top", f"{parts['stack_mm']['top']:.0f} mm, Ø63", "Six M3 x 12 down into the ATI; four M3 x 5.7 brass inserts in "
             "v3's foot pattern (radius 25.5 mm at 0, 90, 198, 270°)"],
            ["Hook cross", "140 x 140 x 105 mm", "In the fork's place for the calibration; 4.5 mm holes for S hooks at the five "
             "points"]]
    body = (f"<p>The ATI goes between mount v3's adapter and its standoff-fork, so the HEX-E, the indenters and the finger stay "
            f"as they are (Fig. 1). v3's adapter is bolted to the HEX-E first, with its M6 x 20 tightened before anything covers "
            f"them. The base then goes over the M6 heads and screws into the adapter's brass inserts outside the ATI's "
            f"footprint. The fork screws into the top as it did into the adapter. The stack makes the tool "
            f"{parts['stack_mm']['total_added']:.1f} mm longer, so the calibration plans must be rebuilt for it. For the "
            f"masses, the hook cross goes in the fork's place (Fig. 2).</p>"
            f"<p>The Mini45's interface comes from ATI's drawing 9230-05-1094 as quoted by a web search. Its mounting side has six "
            f"M3 on a 39.24 mm circle, its tool side six M3 on 28.2 mm, and 3 mm dowel holes. ATI's own site could not be "
            f"reached on 23 October. The two holes of each pair are assumed {parts['unmeasured']['ATI_PAIR_DEG']:.0f}° apart. "
            f"The ATI is clocked at {parts['clock_deg']:.0f}°, which leaves {parts['counterbore_wall_to_base_m3_mm']:.1f} mm of wall "
            f"between its screws' counterbores and the base's M3 holes ({code(ROOT / 'scripts/ati_stack_mount.py')}).</p>"
            + (lambda pl: (f"<p>The base, the top and the hook cross print on one P1S plate, at Bambu's 0.20 mm Standard with "
                           f"six walls. The plate takes {pl['time']} and {pl['filament_g']:.0f} g, most of it the cross. It is "
                           f"{code(D / 'print/20261023-ati_stack_p1s_pla.3mf')}; after a remeasure it is rebuilt by "
                           f"scripts/ati_stack_mount.py and then scripts/v3b_ur5v2_pla_plate.py ati_stack.</p>") if pl else "")(
                json.loads((D / "print/20261023-ati_stack_p1s_pla.json").read_text())
                if (D / "print/20261023-ati_stack_p1s_pla.json").exists() else None)
            + table(["Part", "Size", "Fastening"], rows, caption="Printed parts and the ATI in the stack.", n=4)
            + fig(1, D / "media/20261023-ati_stack_exploded.png", "The stack exploded along the tool axis: v3's adapter, the "
                  "ATI base, the Mini45 (a stand-in cylinder), the ATI top and v3's standoff-fork.")
            + fig(2, D / "media/20261023-ati_hook_cross.png", "The hook cross on the ATI top in the fork's place, with its "
                  "five hook points."))
    S.append(("stack", "The stack on the HEX-E and the hook cross", body))

    steps = ["Measure the Mini45: height, the angle between the holes of a pair, the dowel holes' places and the M3 thread "
             "depth on both faces; set ATI_* in scripts/ati_stack_mount.py and rebuild.",
             "Save the Net F/T box's state as found (web pages, netftapi2.xml, the selected calibration), then select the "
             "raw-gage slot #16 (FT00000) by hand. Connect the box, check that the gages read steady at rest.",
             "Run scripts/ati_netft_logger.py --ip <box> --out <session folder> beside the UR5 program, so that both write "
             "into the same session folder with host time stamps.",
             "Print the base, the top and the hook cross; press the brass inserts into the top.",
             "Assemble the stack with the hook cross. For each of the five hook points, run the plan's gravity stations "
             "once without the mass and once with the weighed mass on its hook. With the cross on, the tool stays "
             "63.7 mm from the robot's links and 67 mm from the cell at all 16 stations of the lab's plan "
             "(scripts/ati_gravity_pose_check.py). A hanging mass must not swing within those distances.",
             "Fit with scripts/ati_calibration_fit.py on the pairs (mass-on minus mass-off at each orientation) and check "
             "the held-out poses; the same pairs give the HEX-E's gains.",
             "Swap the cross for the fork and finger; run the sphere plan rebuilt for the 44.7 mm stack; compare the ATI "
             "with the HEX-E on the presses."]
    body = "<ol>" + "".join(f"<li>{x}</li>" for x in steps) + "</ol>"
    S.append(("procedure", "Procedure", body))

    body = ("<p>In both simulations the sensor is linear with the FT31565 row shape and gage gains within 10 %, and the "
            "mass and hook geometry are known exactly. Hysteresis, temperature drift and a damaged gage are not in them. The stack's "
            "parts are drafts from a quoted drawing; nothing here has been printed or connected. The Mini40 (1 N·m torque "
            "range) suits a 0.2 kg mass on the same hooks; the Mini45 takes the 0.5 kg.</p>")
    S.append(("limits", "Limitations", body))
    fw = ["Fit the 40-pose calibration on the real Mini45 and score it on a held-out hook point. An error above 0.1 N or "
          "5 N·mm means a nonlinearity or a gage problem that a 6 x 6 matrix does not represent.",
          "Repeat the five-hook procedure a day later without remounting: a change of C beyond the fit's spread is drift.",
          "Compare the ATI and the HEX-E on the sphere presses after both are calibrated with the mass. Their difference "
          "is the HEX-E's error, which was in the ground truth of 22 October."]
    S.append(("future", "Future work", "<ol>" + "".join(f"<li>{x}</li>" for x in fw) + "</ol>"))
    return S, lede, summary


def main():
    S, lede, summary = sections()
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    byline = (f"Workstation, 23 October 2026, built from {code(D)} by {code(ROOT / 'scripts/ati_ft_calibration_page.py')}. "
              f"Owner's handoff note on the ATI hardware: {code('ATI_Mini40_Mini45_Calibration_Handoff.md')} (uploaded "
              f"23 October). The calibration runs it builds on: "
              + link(CAL / "20261022-ur5_tip_ground_truth_and_predictors.html", "UR5 ground truth and predictors") + ".")
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "ATI calibration plan", "H1": "Empirical calibration of the lab's ATI Mini45 on the UR5 with known masses",
                 "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-23", "LEDE": lede, "BYLINE": byline,
                 "FILM": "", "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary)
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    main()
