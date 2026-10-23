#!/usr/bin/env python3
"""Page of 2026-10-23: the calibration of the lab's ATI Mini45 on the UR5 with a weighed mass: the coaxial stack on the
HEX-E from ATI's drawing, the hook cross, the robot plan, the session program and its simulated sessions (owner: "the only
way to permanently fix the HEX-E robot is to calibrate one of our uncalibrated ATI sensors"; later the same day: "design a
mount for the ATI for the HEX-E and build up a calibration plan ... the Z-axis is the noisiest axis ... maybe we have to
design an alternate mount that has it rotated?").

Reads docs/experiments/20261023-ati_ft_calibration/ (the stack's parts of scripts/ati_stack_mount.py, the plate of
scripts/v3b_ur5v2_pla_plate.py ati_stack and ati_mass, the mass holder of scripts/ati_stack_mount.py holder, the plan of
scripts/ati_calibration_plan.py, the simulated sessions and the reference comparison of scripts/ati_sim_sessions.py (and its
runs at 250 g, 112 g and 112 g with 12 s rests), the press excitation of scripts/ati_calibration_sim.py, the film of
scripts/ati_calibration_film.py) and writes docs/experiments/20261023-ati_ft_calibration/20261023-ati_ft_calibration_plan.html.

    .venv/bin/python scripts/ati_ft_calibration_page.py
"""
from __future__ import annotations

import base64
import json
import math
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
HANDOFF = D / "20261023-ati_mini40_mini45_calibration_handoff.md"     # the owner's note as uploaded on 10-23, unchanged
DRAWING = "https://www.ati-ia.com/app_content/Documents/9230-05-1094.auto.pdf"
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


def film(n, mp4, poster, caption):
    v = "data:video/mp4;base64," + base64.b64encode(Path(mp4).read_bytes()).decode()
    return (f'<figure id="fig{n}"><video controls muted loop playsinline autoplay preload="metadata" '
            f'poster="{img_uri(poster, 1280)}" src="{v}" style="width:100%"></video>'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def nf(v, k=3):
    return ", ".join(f"{x:.{k}f}" for x in v)


def nmm(v, k=1):
    return ", ".join(f"{x * 1e3:.{k}f}" for x in v)


def load():
    j = lambda p: json.loads(Path(p).read_text())                              # noqa: E731
    return dict(parts=j(D / "20261023-ati_stack_parts.json"), plan=j(D / "plan/20261023-ati_plan_mass.json"),
                sims=j(D / "20261023-ati_sim_sessions.json"), plate=j(D / "print/20261023-ati_stack_p1s_pla.json"),
                press=j(D / "20261023-ati_calibration_sim.json"), holder=j(D / "20261023-ati_mass_holder.json"),
                hplate=j(D / "print/20261023-ati_mass_holder_p1s_pla.json"),
                cplate=j(D / "print/20261023-ati_mass_cap2_p1s_pla.json"),
                by_mass={t: j(D / f"20261023-ati_sim_sessions_{t}.json")
                         for t in ("m250", "m112", "m112r12", "m112stiff", "mini40_m112")},
                parts40=j(D / "20261023-ati_stack_mini40_parts.json"), plan40=j(D / "plan/20261023-ati_plan_mass_mini40.json"),
                plate40=j(D / "print/20261023-ati_stack_mini40_p1s_pla.json"))


def least(plan):
    st = plan["stations"]
    out = {k: min(s["clear"][k] for s in st) for k in ("self_tool", "port", "tool_cell", "links_cell")}
    for kind in ("mass_line", "mass_tool", "mass_links", "mass_cell"):
        out[kind] = min(v for s in st for k, v in s["clear"].items() if k.startswith(kind + "_") and k.split("_")[-1] in s["eyes"])
    q = np.array([s["q_deg"] for s in st])
    out["q_lo"], out["q_hi"] = q.min(0), q.max(0)
    return out


def by_mass_rows(X):
    """Error against the truth by hung mass and rest: the medians' range over five sensors and three axes, and the largest
    90th percentile."""
    out = []
    for sims in [X["sims"]] + [X["by_mass"][t] for t in ("m250", "m112", "m112r12", "m112stiff", "mini40_m112")]:
        md = np.array([x["truth"]["full"]["median"] for x in sims["sessions"]])
        p9 = np.array([x["truth"]["full"]["p90"] for x in sims["sessions"]])
        off = (sims["sim_line_mm"] - sims["line_mm"]) / 2 if sims.get("sim_line_mm") else 0.0
        out.append(dict(sensor="Mini40" if "mini40" in sims["plan"] else "Mini45",
                        mass_g=sims.get("mass_g", 500.3), rest_s=sims.get("rest_s") or 3.0, F=(md[:, :3].min(), md[:, :3].max()),
                        F90=p9[:, :3].max(), T=(md[:, 3:].min() * 1e3, md[:, 3:].max() * 1e3), T90=p9[:, 3:].max() * 1e3,
                        off=off, line=(f"{off:.0f} mm off the model" if off > 0 else f"thread, {sims.get('line_mm', 0.3):.1f} mm")))
    return out


def sections(X):
    parts, plan, sims, plate, press = X["parts"], X["plan"], X["sims"], X["plate"], X["press"]
    hold, hplate, BM = X["holder"], X["hplate"], by_mass_rows(X)
    hg = hold["geometry"]
    m112 = next(b for b in BM if b["mass_g"] < 200 and b["rest_s"] == 3.0 and b["off"] == 0 and b["sensor"] == "Mini45")
    m40 = next(b for b in BM if b["sensor"] == "Mini40")
    p40, pl40, pt40 = X["parts40"], X["plan40"], X["plate40"]
    L40 = least(pl40)
    stiff = next(b for b in BM if b["off"] > 0)
    lo_st = np.array([x["leave_eye_out"]["full"]["median"] for x in X["by_mass"]["m112stiff"]["sessions"]])
    stiff_lo = (lo_st[:, 3:].min() * 1e3, lo_st[:, 3:].max() * 1e3)
    slow = next(b for b in BM if b["rest_s"] > 3.0)
    L = least(plan)
    rc = sims["reference_compare"]
    tr = np.array([s["truth"]["full"]["median"] for s in sims["sessions"]])
    lo = np.array([s["leave_eye_out"]["full"]["median"] for s in sims["sessions"]])
    sv = plan["checks"]["scaled_singular_values"]
    zM = parts["z_M"]
    ev = {e["name"]: e for e in parts["eyes"]}
    lede = (f"The Mini45 goes on the HEX-E coaxially, and a weighed mass is the reference of its calibration. In simulation, "
            f"with the HEX-E as reference the ATI's force is off by {min(rc['hexe_coaxial']['median'][:3]):.2f} to "
            f"{max(rc['hexe_coaxial']['median'][:3]):.2f} N (medians). Turning the ATI by 90° moves the HEX-E's Fz gain error "
            f"into the ATI's Fx. With a 0.5 kg mass, five simulated Mini45s are recovered to {tr[:, :3].min():.3f} to "
            f"{tr[:, :3].max():.3f} N and {tr[:, 3:].min() * 1e3:.1f} to {tr[:, 3:].max() * 1e3:.1f} N·mm. With the lab's "
            f"100 g weight in its printed holder the error is {m112['F'][0]:.3f} to {m112['F'][1]:.3f} N; it falls about "
            f"as the inverse of the mass. The printed stack "
            f"is built to ATI's drawing 9230-05-1094; it prints in {plate['time']} on the P1S. The robot plan has "
            f"{len(plan['stations'])} stations at five eyes of a hook cross and takes about {plan['minutes']:.0f} min. Every "
            f"station and move is checked against the arm, the cell and the hanging mass. For the lab's Mini40 a base and a "
            f"top were drawn from ATI's drawings 9230-05-1278 and 9230-05-1314; the cross and the holder are shared. Its plan has "
            f"{len(pl40['stations'])} stations, and with the 100 g weight five simulated Mini40s are recovered to "
            f"{m40['F'][0]:.4f} to {m40['F'][1]:.4f} N.")
    summary = (f"ATI Mini45 calibration with a weighed mass: coaxial stack from ATI drawing 9230-05-1094 (P1S, {plate['time']}); "
               f"{len(plan['stations'])} checked stations at five eyes, about {plan['minutes']:.0f} min; simulated sessions "
               f"recover the sensor to {tr[:, :3].max():.3f} N and {tr[:, 3:].max() * 1e3:.1f} N mm (medians) with 0.5 kg and "
               f"to {m112['F'][1]:.3f} N with the lab's 100 g weight in a printed holder; with the HEX-E "
               f"as reference the error is up to {max(rc['hexe_coaxial']['median'][:3]):.2f} N, coaxial or turned 90 deg. The Mini40 "
               f"gets its own base and top (ATI drawings 9230-05-1278, 9230-05-1314), {len(pl40['stations'])} stations, and "
               f"{m40['F'][1]:.4f} N with the 100 g weight in simulation.")
    S = []
    terms = [("Wrench $w$", "Force and torque at the ATI's sensing origin, in the ATI's frame: $F_x, F_y, F_z$ in N and "
              "$T_x, T_y, T_z$ in N·mm. The origin is the centre of the tool adapter plate's face; $+z$ points from the "
              "mounting side to the tool side (ATI drawing 9230-05-1094)."),
             ("Gage values $g$", "The six raw strain gage values the Net F/T box sends over RDT when calibration #16 "
              "(FT00000) is selected, in counts."),
             ("Calibration matrix $C$", "The 6 x 6 matrix of $w = C\\,(g - g_0)$, from counts to N and N·m; $g_0$ is the "
              "gage values with no load on the tool side."),
             ("Eye", "A 1.2 mm hole through a 1.6 mm end wall of the hook cross, with a slot behind the wall. The line from "
              "which the mass hangs passes through it, its stopper knot in the slot. Each eye has an exit point $p_e$ and an "
              "outward direction $u$."),
             ("Line", "A thin, flexible line from an eye to the holder's tie hole: polyester thread, dental floss or braided "
              "fishing line, 0.2 to 0.5 mm across. Its diameter $d$ is measured and given to the session."),
             ("Mass $m$", "What hangs from the eye, weighed to 0.1 g: the lab's 100 g weight with its printed holder, the "
              "holder's two screws and the line, or a heavier mass."),
             ("Bend", "The angle between an eye's outward direction and gravity. The line hangs free of its arm when the bend "
              "is at most 75°, that is $g \\cdot u \\geq 0.26$ with $g$ the unit vector of gravity."),
             ("Station", "A joint vector of the UR5 that turns the ATI about a fixed point $P_0$ until gravity has a chosen "
              "direction in the ATI's frame."),
             ("Excitation", "The singular values of the stacked reference wrenches with torques divided by 0.1 m. A small value "
              "marks a direction of $w$ that the data barely test."),
             ("Error against the truth", "For a simulated sensor, the median absolute error per axis of a fitted $C$ on 2000 "
              "wrenches outside the fit. Their forces are 0.2 to 5 N, at points within 150 mm of the ATI's origin."),
             ("Leave-one-eye-out error", "$C$ fitted without one eye's pairs and scored on them, each eye in turn; the "
              "median absolute error per axis.")]
    S.append(("terms", "Terms and metrics", '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"))

    m, co, tu = rc["mass"]["median"], rc["hexe_coaxial"]["median"], rc["hexe_turned"]["median"]
    rows = [["Weighed mass", nf(m[:3]), nmm(m[3:])], ["HEX-E, ATI coaxial", nf(co[:3]), nmm(co[3:])],
            ["HEX-E, ATI turned 90° about y", nf(tu[:3]), nmm(tu[3:])]]
    body = (f"<p>The HEX-E's z axis is its noisiest. In its data sheet the noise per sample is 0.15 N in Fz and 0.035 N in "
            f"Fx and Fy. On 22 October the lab measured 0.21 N in Fz, at 100 Hz with filter 3. Its gains differ more than its "
            f"noise: in the gravity sweep of run first2, Fz came out at 0.78 of the three axes' mean. Fx and Fy came out at "
            f"1.135 and 1.085. The ATI Mini45 has one force resolution for all three axes (ATI's Mini45 page). The resolution is "
            f"1/20 lbf in the US-120-160 calibration that the sensor was received with, and 1/16 N in SI-145-5.</p>"
            f"<p>In simulation the ATI was calibrated against three references on the same 40 pairs of a session (Table 1). "
            f"The HEX-E's reading had the gains above, and its noise was averaged over each 3 s rest. That noise averages "
            f"down to 0.012 N in Fz, while its gain errors pass into $C$ unchanged. With the ATI turned 90°, the ATI's Fx is "
            f"read by the HEX-E's Fz. The error in Fz then falls from {co[2]:.3f} to {tu[2]:.3f} N. The error in Fx rises "
            f"from {co[0]:.3f} to {tu[0]:.3f} N. With the mass as reference, no sensor axis enters the fit: the robot turns "
            f"gravity through every direction of the ATI's frame. So the stack is coaxial. The HEX-E is logged in the "
            f"session only to fit its own gains, R5 of the tip calibration's queue.</p>"
            + table(["Reference", "F error (N)", "T error (N·mm)"], rows, num_cols=(1, 2),
                    caption="Error against the truth of the full fit with each reference, medians; for the HEX-E over 20 draws "
                    f"of its noise and torque gains ({code(ROOT / 'scripts/ati_reference_compare.py')}).", n=1)
            + f"<p>The sphere presses of the tip calibration cannot replace the gravity stations either. Every press acts about "
            f"80 mm from the ATI, so each torque is nearly a fixed multiple of its force. Their scaled singular values are "
            f"{nf(press['singular_values_scaled'][:3], 1)} for three directions and {nf(press['singular_values_scaled'][3:], 2)} "
            f"for the others ({code(ROOT / 'scripts/ati_calibration_sim.py')}).</p>")
    S.append(("reference", "The HEX-E's z axis and the reference of the calibration", body))

    rows = [["Mounting side (robot)", "6 x M3 x 0.5, 5.0 deep", "39.24 mm; pairs 20° apart about 60, 180, 300°",
             "1 x Ø3, 5 deep, at +X on 39.24 mm", "Ø9.5 recess, 3 deep"],
            ["Tool side", "6 x M3 x 0.5, 3.0 deep", "28.2 mm; pairs 40° apart about 60, 180, 300°",
             "2 x Ø3, 3.4 deep, at 0 and 120° on 28.194 mm", "Ø9.5 recess, 3 deep"]]
    body = (f"<p>The Mini45's interface is given in ATI's drawing 9230-05-1094 rev 20 (<a href=\"{DRAWING}\">ATI</a>, "
            f"fetched 23 October) and in Table 2. The transducer is 45 mm across and 15.7 mm high, with a 5.7 mm mounting adapter plate and "
            f"a 3.7 mm tool adapter plate. The cable of the standard Mini45-R leaves the mounting plate radially, 110° from "
            f"+X. The drawing's note 3 forbids screws deeper than the threads. Note 5 forbids any part that touches both "
            f"plates. In the morning's draft, 20° pairs and 4 mm threads had been assumed on both faces. On the tool side the pairs are "
            f"40° apart and the threads 3.0 mm deep.</p>"
            + table(["Face", "Threads", "Bolt circle", "Dowel holes", "Centre"], rows,
                    caption="The Mini45's interface on drawing 9230-05-1094 rev 20; angles from the ATI's +X toward +Y.", n=2))
    S.append(("interface", "The Mini45's interface on ATI's drawing", body))

    sc = parts["screws"]
    rows = [["ATI base", "Ø70, 18 mm and a Ø44 pad 2 mm high", sc[0]["what"].replace("socket head, ", "") + "; " +
             sc[1]["what"].replace("socket head, ", "")],
            ["ATI Mini45", "Ø45, 15.7 mm", f"Turned {parts['clock_deg']:.0f}° about the tool axis from $x_M$; its cable "
             f"toward {parts['cable_deg_M']:.0f}° in M"],
            ["ATI top", "Ø63, 11 mm", sc[2]["what"].replace("socket head, ", "") + "; four M3 x 5.7 brass inserts in mount "
             "v3's foot pattern"],
            ["Hook cross", "160 x 160 x 124 mm", sc[4]["what"].replace("socket head, ", "")]]
    body = (f"<p>The ATI goes between mount v3's adapter and its standoff-fork (Fig. 1). The HEX-E, its adapter and the "
            f"indenters stay as they are. The adapter's M6 x 20 are tightened first; the base fits over their heads and "
            f"screws into the adapter's brass inserts beside the ATI. The fork, or the hook cross for the calibration, screws "
            f"into the top as it did into the adapter. The tool becomes {parts['added_length_mm']:.1f} mm longer, and the "
            f"ATI's origin stands at $z_M$ = {zM['ati_origin']:.1f} mm, $M$ being mount v3's frame on the HEX-E's tool "
            f"face.</p>"
            f"<p>The screws stop short of the ATI's threads: 4.0 mm of the 5.0 mm on the mounting side and 2.5 mm of the "
            f"3.0 mm on the tool side. A pad 44 mm across lifts the ATI 2 mm off the base, so the cable's fitting clears the "
            f"disc. A channel 7 mm wide and a zip-tie tunnel lead the cable to the robot's side. At a clock of 114° the walls "
            f"between the base's own holes are {parts['wall_mm']:.1f} mm, the thickest of any clock. The cable then leaves "
            f"44° from the HEX-E's cable port. Dowel pins are optional: one 3 x 8 in the base and two 3 x 6 in the top locate "
            f"the ATI to its drawing.</p>"
            f"<p>The three parts print on one P1S plate at Bambu's 0.20 mm Standard with six walls. The plate takes "
            f"{plate['time']} and {plate['filament_g']:.0f} g, with no supports "
            f"({code(D / 'print/20261023-ati_stack_p1s_pla.3mf')}). The arms' eyes print as horizontal holes and are opened "
            f"with a 1.0 to 1.2 mm drill.</p>"
            + table(["Part", "Size", "Fastening"], rows, caption="The stack's parts and screws "
                    f"({code(ROOT / 'scripts/ati_stack_mount.py')}).", n=3)
            + fig(1, D / "media/20261023-ati_stack_exploded.png", "The stack exploded along the tool axis: mount v3's adapter, "
                  "the ATI base, the Mini45 with its cable's fitting, the ATI top and mount v3's standoff-fork."))
    S.append(("stack", "The stack on the HEX-E", body))

    T = 0.117 * 9.796                                     # the 100 g weight with its holder and screws, N
    lam = lambda E, d: math.sqrt(E * math.pi * d ** 4 / 64.0 / T)          # noqa: E731  bending length, mm (E in N/mm2)
    body = (f"<p>For the calibration a hook cross is screwed on in the fork's place (Fig. 2). Four arms reach "
            f"{abs(ev['+x']['p_M'][0]):.0f} mm along $\\pm x_M$ and $\\pm y_M$, and a post reaches 118 mm up the tool axis. "
            f"Each of the five ends is an eye: a hole of radius $R$ = {parts['eye']['hole_d'] / 2:.1f} mm through a wall "
            f"$t$ = {parts['eye']['wall']:.1f} mm thick, with a slot behind it. A mass hangs from one eye at a time on a line "
            f"whose stopper knot sits in the slot, and is weighed to 0.1 g. Its wrench at the ATI's origin is the "
            f"reference:</p>"
            f"$$w = \\begin{{bmatrix}} m\\,\\mathbf{{g}}_A \\\\ p_l \\times m\\,\\mathbf{{g}}_A \\end{{bmatrix}}.$$"
            f"<p>Here $m$ is the mass and $\\mathbf{{g}}_A$ local gravity in the ATI's frame, from the robot's joints (9.796 "
            f"m/s² in Boulder). $p_l$ is a point of the line's centre line. With $p_e$ the eye's exit point, $u$ its outward "
            f"axis, $\\alpha$ the angle between gravity and $u$, $n$ the unit direction of gravity across $u$ and $d$ the "
            f"line's diameter,</p>"
            f"$$p_l = \\begin{{cases}} p_e - t\\,u, & \\tan\\alpha \\le (R - d/2)/t, \\\\ p_e + R\\,n + \\tfrac{{d}}{{2}}\\,\\hat m, "
            f"& \\text{{otherwise.}} \\end{{cases}}$$"
            f"<p>Near the eye's axis the line runs straight from the knot. Past "
            f"{math.degrees(math.atan((parts['eye']['hole_d'] / 2 - 0.15) / parts['eye']['wall'])):.0f}° for a line 0.3 mm "
            f"across, it bends over the hole's outer edge and hangs $d/2$ off it. $\\hat m$ is the unit vector across gravity "
            f"in the plane of $u$ and gravity, on the side of $u$. At one station the tool's own weight and the gages' zero "
            f"are the same with and without the mass. Each station is therefore recorded both ways, and their difference is "
            f"the mass's wrench alone.</p>"
            f"<p>The model of $p_l$ is right only for a line that turns at the knot or at the edge. The bending length "
            f"$\\sqrt{{EI/T}}$ is the length over which a line turns toward gravity under its tension $T$. Under the "
            f"{T:.2f} N of the 100 g weight and its holder, it is {lam(117e3, 0.5):.0f} mm for a copper wire 0.5 mm across. "
            f"It is {lam(117e3, 1.0):.0f} mm for one 1 mm across, and {lam(3e3, 0.3):.1f} mm for a nylon monofilament "
            f"0.3 mm across. For a polyester thread, dental floss or a braided fishing line it is about 0.1 to 0.2 mm, since "
            f"their fibres bend apart. A stiff line moves the line of action off $p_l$ by about its bending length. Each "
            f"millimetre adds {T:.2f} N·mm to a torque reference. In simulation at 112 g, the line was hung "
            f"{stiff['off']:.0f} mm farther off the edge than modelled (Table 6). The torque errors rose from "
            f"{m112['T'][0]:.1f} to {m112['T'][1]:.1f} N·mm to {stiff['T'][0]:.1f} to {stiff['T'][1]:.1f} N·mm. The forces stay at "
            f"{stiff['F'][0]:.3f} to {stiff['F'][1]:.3f} N, since the force references do not depend on $p_l$. The "
            f"session's own check, leaving one eye out, gives {stiff_lo[0]:.1f} to {stiff_lo[1]:.1f} N·mm with the stiff line "
            f"as with thread, so the error would not be seen in the lab.</p>"
            f"<p>The lab's mass is a 100 g weight: a body {hold['weight']['d_mm']:.0f} mm across and "
            f"{hold['weight']['h_mm']:.0f} mm tall, with a knob {hold['weight']['knob_mm'][0]:.0f} mm across and "
            f"{hold['weight']['knob_mm'][1]:.0f} mm tall on top. It sits in a printed holder (Fig. 3). The cup has three "
            f"crush ribs and a floor with a hole for pushing the weight out. The cap is screwed on with two M3 x 12 and "
            f"presses on the weight's top face. Its pocket, 0.3 mm wider and 0.5 mm deeper than the knob, centres it. A tab "
            f"on the cap has a tie hole {hold['weight']['tie_hole_mm']:.0f} mm across on the axis. The line is tied through "
            f"it, and the holder hangs from the hole's top, {hg['seat_mm']:.1f} mm above the cup's underside. The holder is "
            f"{2 * hg['radius_mm']:.0f} mm across its ears. The plan models the mass as a line "
            f"{hg['envelope']['line_mm']:.0f} mm long, then a cylinder {2 * hg['envelope']['r_mm']:.0f} mm across and "
            f"{hg['envelope']['h_mm']:.0f} mm long. The holder stays inside that envelope with its tie hole "
            f"{hg['seat_below_eye_mm'][0]:.0f} to {hg['seat_below_eye_mm'][1]:.0f} mm below the eye, so the line is tied "
            f"with 55 to 70 mm between the eye and the tie hole. The cup and cap print upright without supports in "
            f"{hplate['time']} ({code(D / 'print/20261023-ati_mass_holder_p1s_pla.3mf')}); the cap alone prints in "
            f"{X['cplate']['time']} ({code(D / 'print/20261023-ati_mass_cap2_p1s_pla.3mf')}).</p>"
            + fig(2, D / "media/20261023-ati_hook_cross.png", "The hook cross on the stack, the tool 30° from vertical and the "
                  "100 g weight in its holder hanging from the +x eye.")
            + fig(3, D / "media/20261023-ati_mass_holder.png", "The holder of the 100 g weight, exploded and cut through its "
                  f"axis ({code(ROOT / 'scripts/ati_stack_mount.py')} holder). Cap 2 of 15:34 has a pocket for the measured "
                  "knob and a tie hole; the cap of 12:52 had a slot for a pigtail's loop.", width=900))
    S.append(("cross", "The hook cross and the hanging mass", body))

    rows = [["Mounting side (robot)", "3 x M3 x 0.5, 5.0 deep", "33 mm; at 82.5, 202.5, 322.5°",
             "2 x Ø3, 2.5 deep (-E: 5.0), at 37.5 and 277.5° on 33 mm", "Ø6.35 through hole"],
            ["Tool side", "3 x M3 x 0.5, 3.0 deep", "33 mm; at 60, 180, 300°", "2 x Ø3, 2.5 deep, at 0 and 120° on 35 mm",
             "Through hole"]]
    sc40 = p40["screws"]
    body = (f"<p>The lab's second sensor is a Mini40 (FT16552, received for the SI-20-1 calibration: 20 N in $F_x$ and $F_y$, "
            f"60 N in $F_z$, 1 N·m in torque). It is read by the same Net F/T box, one sensor at a time. Its interface is given "
            f"in ATI's drawings 9230-05-1278 rev 11 (Mini40-R and -A) and 9230-05-1314 rev 09 (Mini40-E), fetched 23 October "
            f"(Table 4). Both drawings give the same holes. The transducer is 40.1 mm across and 12.27 mm tall; the -E is "
            f"14.0 mm tall. Its origin is the centre of the tool face, as the Mini45's. The cable leaves the mounting plate "
            f"radially, at 150° from +X on the -R and at 157.5° on the -E with a 17.3 mm strain relief. The -A's cable leaves "
            f"the mounting face's centre and is not provided for.</p>"
            f"<p>The Mini40 takes its own base and top, and the cross, the holder and the line are the Mini45's (Fig. 4). The base "
            f"is the Mini45's with a pad {p40['ati']['d_mm'] - 2.1:.0f} mm across and a channel 10 mm wide that takes either "
            f"cable. Its three M3 x 12 enter 4.0 mm of the 5.0 mm threads, and the top's three M3 x 10 enter 2.5 mm of the 3.0 mm. "
            f"At a clock of {p40['clock_deg']:.0f}° the base's least wall is {p40['wall_mm']:.2f} mm. The cable then leaves "
            f"toward {p40['cable_deg_M']:.0f}° in M, beside the Mini45's {parts['cable_deg_M']:.0f}°, so both cables are routed "
            f"the same way along the arm. The thickest walls, 5.63 mm, would point the cable to 132°. The tool becomes "
            f"{p40['added_length_mm']:.1f} mm longer, and the eyes keep their places in the ATI's frame on the -E as well. The "
            f"two parts print in {pt40['time']} and {pt40['filament_g']:.0f} g "
            f"({code(D / 'print/20261023-ati_stack_mini40_p1s_pla.3mf')}).</p>"
            + table(["Face", "Threads", "Bolt circle", "Dowel holes", "Centre"], rows,
                    caption="The Mini40's interface on drawings 9230-05-1278 rev 11 and 9230-05-1314 rev 09; angles from the "
                    "ATI's +X toward +Y.", n=4)
            + fig(4, D / "media/20261023-ati_stack_mini40_exploded.png", "The Mini40's stack exploded along the tool axis "
                  f"({code(ROOT / 'scripts/ati_stack_mount.py')} --sensor mini40): mount v3's adapter, the Mini40's base, the "
                  "Mini40-R with its cable's fitting, its top, and mount v3's standoff-fork."))
    S.append(("mini40", "The Mini40's interface and stack", body))

    body = (f"<p>At a station the ATI is turned about a fixed point $P_0$ until gravity has a chosen direction in its frame. "
            f"$P_0$ is ({nf(plan['p0_mm'], 0)}) mm in the robot's base frame: 430 mm above the table and 110 mm short of the "
            f"indenters' 2x4. The candidates were 160 gravity directions, each at 24 turns about the vertical. They were "
            f"solved on the branch of the lab's homing pose ({code(ROOT / 'scripts/ati_calibration_plan.py')}), and "
            f"{plan['checks']['n_candidates']} directions passed every check. Eight stations per eye were then chosen greedily "
            f"for D-optimality: each next station raised the log-determinant of the information of the six wrench "
            f"directions the most. The excitation is {nf(sv[:5], 1)} and {sv[5]:.1f}. The random poses of the morning's "
            f"draft gave 21.5 down to 6.3.</p>"
            f"<p>At each station the joints stay on the branch of the homing pose. Joint 5 stays between 30 and 130°, and "
            f"joint 6 within 150° of the homing pose's, for the cables. The tool clears the forearm and wrist links by at "
            f"least {L['self_tool']:.1f} mm, and the HEX-E's cable port clears them by {L['port']:.1f} mm. The tool and the "
            f"links clear the table, the 2x4, the plywood and both indenters by "
            f"{min(L['tool_cell'], L['links_cell']):.0f} mm. The mass is modelled as a cylinder 50 mm across and 80 mm long, "
            f"below 45 mm of line. It stays {L['mass_tool']:.0f} mm from the tool and {L['mass_links']:.0f} mm from the "
            f"robot's links, the shoulder included, and {L['mass_cell']:.0f} mm from the cell. Joint 5 reaches "
            f"{L['q_hi'][4]:.0f}°, past the 100° window of the tip calibration's program. Those stations are checked on the "
            f"robot in the dry run at half speed.</p>"
            f"<p>The session runs one block per eye. A block starts at the reference station without the mass, for the "
            f"drift. The eye's eight stations follow without the mass. At the loading station the operator hangs the mass, "
            f"and the same stations are visited with it. At the loading station again the operator takes it off. Within a "
            f"block the stations are ordered as the cycle of least joint travel. In that cycle every move is clear with "
            f"the mass hanging, and no move bends the line back over its arm. Every move was checked at ten points along "
            f"its joint path. The {len(plan['stations'])} stations and {sum(len(b['steps']) for b in plan['blocks'])} steps "
            f"take about {plan['minutes']:.0f} min at 0.3 rad/s (Fig. 5). Each station has 4 s for the swing to decay and "
            f"3 s of rest. The plan is {code(D / 'plan/20261023-ati_plan_mass.json')}.</p>"
            f"<p>The Mini40's plan was built the same way from its parts "
            f"({code(D / 'plan/20261023-ati_plan_mass_mini40.json')}). It has {len(pl40['stations'])} stations and "
            f"{sum(len(b['steps']) for b in pl40['blocks'])} steps and takes about {pl40['minutes']:.0f} min. Its excitation is "
            f"{nf(pl40['checks']['scaled_singular_values'][:5], 1)} and {pl40['checks']['scaled_singular_values'][5]:.1f}. The "
            f"tool clears the forearm and wrist by {L40['self_tool']:.1f} mm and the HEX-E's cable port by {L40['port']:.1f} mm, "
            f"and the mass's line clears the tool by {L40['mass_line']:.1f} mm. Joint 5 reaches {L40['q_hi'][4]:.0f}°.</p>"
            + film(5, D / "media/20261023-ati_calibration_sim_session.mp4", D / "media/20261023-ati_calibration_sim_session.png",
                   "The plan in a simulated session at 24 times speed. Left, the cell from the front; right, the tool from "
                   "the side away from the robot. The label bar names the block, the eye and the mass's state."))
    S.append(("plan", "Robot plan of the calibration", body))

    rows = []
    for s, t, l in zip(sims["sessions"], tr, lo):
        rows.append([str(s["seed"]), str(s["pairs"]), nf(t[:3]), nmm(t[3:]), nf(l[:3]), nmm(l[3:])])
    r12 = np.array([s["truth"]["rows12"]["median"] for s in sims["sessions"]])
    body = (f"<p>The session program ({code(ROOT / 'scripts/ati_calibration_session.py')}) ran the plan against a "
            f"simulated robot and box in virtual time, five times with five simulated sensors "
            f"({code(ROOT / 'scripts/ati_calibration_sim_session.py')}). Each sensor had FT31565's row shapes at the "
            f"US-120-160 range with gage gains up to 10 % off, and 3 counts of noise per sample at 500 Hz. Its gages drifted "
            f"up to 2 counts a minute with a random walk on top. The ATI sat 1.5° and 0.3 mm off its nominal place, and the "
            f"eyes 0.2 mm off theirs. Their holes were 0.1 mm smaller than drawn, and a line 0.3 mm across was knotted behind each eye. The robot's flange stopped 0.03° off its "
            f"kinematic model, and the mass swung 0.3 to 1.5° after each move. In the analysis "
            f"({code(ROOT / 'scripts/ati_calibration_analysis.py')}) the drift is taken from the reference station's rests, "
            f"and 40 pairs are formed, mass on minus mass off at each station.</p>"
            f"<p>The full 6 x 6 fit recovers each sensor to {tr[:, :3].min():.3f} to {tr[:, :3].max():.3f} N and "
            f"{tr[:, 3:].min() * 1e3:.1f} to {tr[:, 3:].max() * 1e3:.1f} N·mm against the truth (Table 5). Left out one "
            f"eye at a time, it predicts that eye's pairs to {lo[:, :3].min():.3f} to {lo[:, :3].max():.3f} N. The model "
            f"with FT31565's row shapes and twelve gains reaches {r12[:, :3].max():.3f} N. The ATI's 1.5° turn in the stack "
            f"moves its rows away from the borrowed shape. So the full matrix is the calibration.</p>"
            + table(["Sensor", "Pairs", "F error (N)", "T error (N·mm)", "Left-out eye: F (N)", "Left-out eye: T (N·mm)"], rows,
                    num_cols=(1, 2, 3, 4, 5), caption="Simulated sessions on the plan: error against the truth and "
                    "leave-one-eye-out error of the full fit, medians per axis "
                    f"({code(D / '20261023-ati_sim_sessions.json')}).", n=5))
    rows = [[b["sensor"], f"{b['mass_g']:.0f}", f"{b['rest_s']:.0f}", b["line"], f"{b['F'][0]:.4f} to {b['F'][1]:.4f}",
             f"{b['F90']:.4f}", f"{b['T'][0]:.2f} to {b['T'][1]:.2f}", f"{b['T90']:.2f}"] for b in BM]
    body += (f"<p>The sessions above hung 500.3 g. The same five sensors were calibrated with 250 g and with 112 g, the "
             f"100 g weight with its holder and screws (Table 6). The error grows as the mass shrinks. At 112 g the medians "
             f"are {m112['F'][0]:.3f} to {m112['F'][1]:.3f} N, about four times those at 500 g. They remain below the "
             f"HEX-E reference's {min(co[:3]):.2f} to {max(co[:3]):.2f} N. Averaging 12 s at each station in place of 3 s "
             f"leaves them at {slow['F'][0]:.3f} to {slow['F'][1]:.3f} N. So the noise of the rests is not the limit at 112 g. "
             f"The limit is the gages' drift between a station's two rests, which are minutes apart. In the simulator that "
             f"drift is up to 2 counts a minute with a random walk on top; the box check measures the real one.</p>"
             + f"<p>The Mini40 was simulated on its own plan at 112 g, with FT21031's row shapes and 7.25 times the Mini45's "
             f"counts per N (5 times per N·m). That ratio is the one between the two models' largest calibrations, SI-580-20 "
             f"and SI-80-4: their flexures are taken to be strained alike at full range, and the box's raw gain to be the "
             f"same. The 100 g weight then moves the gages 7 times as far as on the Mini45, against the same noise and drift. "
             f"The medians are {m40['F'][0]:.4f} to {m40['F'][1]:.4f} N and {m40['T'][0]:.2f} to {m40['T'][1]:.2f} N·mm "
             f"(Table 6).</p>"
             + table(["Sensor", "Mass (g)", "Rest (s)", "Line", "F error (N), median", "F error (N), 90th percentile",
                      "T error (N·mm), median", "T error (N·mm), 90th percentile"], rows, num_cols=(1, 2, 4, 5, 6, 7),
                     caption="Error against the truth of the full fit by sensor, hung mass, rest time and line: the range of the "
                     "medians over five sensors and three axes, and the largest 90th percentile. In the last row the "
                     "simulated line hangs 2 mm farther off the hole's edge than the analysis takes it to, as a stiff line "
                     f"would ({code(ROOT / 'scripts/ati_sim_sessions.py')} --mass-g, --rest-s, --sim-line-mm, --plan).", n=6))
    S.append(("sim", "Simulated sessions", body))

    pl = plan["payload"]
    steps = [
        "Box: before changing anything, photograph its DIP switches (four screws, the box unpowered) and save every web "
        "page and <code>http://&lt;box&gt;/netftapi2.xml</code>. Power it from 25 V DC at 10 W on the Pwr/CAN connector, or "
        "from a PoE switch. With DIP switch 9 on, its address is 192.168.1.1 (Net F/T manual, section 3.4). The lab computer "
        "is given 192.168.1.100/24 on that network: <code>sudo ip addr add 192.168.1.100/24 dev &lt;interface&gt;</code>.",
        "On the box's Configuration page, select #16-FT00000, lbf and lbf-in, and apply (ATI Quick Start, section 3.4.2). "
        "Counts per Force and Counts per Torque must then be 1. On comm.htm, enable RDT. Check the stream with "
        "<code>python3 scripts/ati_netft_logger.py --ip 192.168.1.1 --out logs/&lt;date&gt;-ati_box_check --seconds 10</code>. "
        "It should show the RDT rate set on comm.htm, status 0 and steady gages at rest.",
        f"Print the plates {code(D / 'print/20261023-ati_stack_p1s_pla.3mf')} and "
        f"{code(D / 'print/20261023-ati_mass_holder_p1s_pla.3mf')} (or {code(D / 'print/20261023-ati_mass_cap2_p1s_pla.3mf')} "
        "alone, if the cup of 12:52 is printed). Open the four arm eyes with a 1.0 to 1.2 mm drill; drill them larger only if "
        "the line does not pass, and give the drill's size to the session as <code>--eye-hole-mm</code>. Press four M3 x 5.7 "
        "brass inserts into the top.",
        "Assemble the stack. Six M3 x 12 go from the base's underside into the ATI's mounting side. The base goes over the "
        "adapter's M6 heads with four M3 x 20. Six M3 x 10 go through the top into the ATI's tool side; no longer screw may "
        "be used there, since its threads are 3.0 mm deep. The cable lies in the base's channel, tied in the tunnel, with a "
        "service loop to the forearm. The hook cross takes four M3 x 12 into the top's inserts. No part may touch both of "
        "the ATI's plates.",
        "Press the weight into the cup, knob up, until it sits on the floor; if the ribs are too tight, scrape them with a "
        "knife. Screw the cap on with two M3 x 12. Tie a line through the cap's tie hole: polyester thread, dental floss or "
        "braided fishing line, 0.2 to 0.5 mm across and rated 2 kg or more, never wire or monofilament. Measure its diameter. "
        "Weigh the weight, holder, screws and line together to 0.1 g. The payload is "
        f"{pl['kg']:.3f} kg, its centre of mass {pl['cog_mm'][2]:.0f} mm from the flange; the program sets it, and adds "
        "the weighed mass while the mass hangs.",
        "To hang the mass at the program's prompt, pass the line's free end through the eye from outside (a drop of "
        "superglue stiffens its tip) and out of the slot. Tie a figure-eight stopper knot so that the tie hole hangs 55 to "
        "70 mm below the eye, and pull the knot back into the slot. To take it off, push the knot out of the slot and untie "
        "it.",
        "Jog the robot to the plan's homing pose (the lab's home of 22 October: joints 48.3, -86.3, 109.5, -24.8, 47.3, "
        "6.4°). A dry run with no mass at half speed, a hand on the emergency stop: <code>python3 scripts/ati_calibration_session.py "
        "--plan docs/experiments/20261023-ati_ft_calibration/plan/20261023-ati_plan_mass.json --ati 192.168.1.1 --dry "
        "--speed 0.5 --name dry</code>.",
        "The run takes about 27 min, with ten mass changes at the program's prompts. Its command is the dry run's with "
        "<code>--mass-g &lt;grams&gt; --line-mm &lt;mm&gt; --hexe 192.168.0.6 --name run1</code> in place of <code>--dry "
        "--speed 0.5 --name dry</code>. The grams are the weighed holder's with its line, and the millimetres the line's "
        "diameter.",
        "The Mini40, after the Mini45's run. Switch the box off and move its transducer cable from the Mini45 to the Mini40. "
        "Unscrew the cross, the top, the Mini45 and its base. Screw the Mini40 to its own base with three M3 x 12 from below, "
        "the base onto the adapter with four M3 x 20, and the Mini40's top onto it with three M3 x 10, never longer. Lay the "
        "cable in the base's channel and screw the cross onto the top. Repeat the box check, then the dry run and the run with "
        "<code>plan/20261023-ati_plan_mass_mini40.json</code> and <code>--name dry40</code>, <code>--name run40</code>. The "
        "Mini40's torque range is 1 N·m: a push of about 12 N at an arm's end saturates it, so the cross is handled gently.",
        "On the workstation: <code>.venv/bin/python scripts/ati_calibration_analysis.py &lt;session folder&gt;</code> writes "
        "the matrix, the leave-one-eye-out errors and the HEX-E's gains into the folder's analysis/.",
        "Afterwards: the fork in the cross's place and the sphere plans rebuilt for the 46.7 mm longer tool, to compare the "
        "calibrated ATI with the HEX-E on the presses."]
    body = "<ol>" + "".join(f"<li>{x}</li>" for x in steps) + "</ol>"
    S.append(("procedure", "Procedure", body))

    body = ("<p>The simulated sensor is linear, with no hysteresis, no temperature drift beyond a linear trend and random "
            "walk, and no damaged gage. The ATI's place in the stack, the eyes' places and the line's departure from each "
            "eye are taken from the drawings. The mass's swing is modelled as a damped pendulum. The plan is checked against "
            "the URDF's collision meshes and the cell of the tip calibration, as the rod plan placed it on 23 October. "
            "Objects outside that model are seen only in the dry run. The Mini40's counts per N in the box's raw mode are "
            "assumed from its calibrations' ranges, and its mass of 50 g from the data sheet. The drift that limits the calibration with "
            "100 g is the simulator's assumption. The line is modelled as turning at its knot or at the hole's edge; its "
            "fibres' bending length of 0.1 to 0.2 mm is not in the model.</p>")
    S.append(("limits", "Limitations", body))
    fw = ["Run the box check and record the box's state as found. A status other than 0 at rest stops the plan before any "
          "print. So does a gage that does not change when the tool side is pressed by hand.",
          "Record 10 min of the box at rest in the box check, and rerun the simulated sessions at 112 g with its noise and "
          "drift. If the median error then exceeds 0.01 N, hang 0.25 to 0.5 kg in place of the 100 g weight; the simulated "
          "error is then a half and a quarter (Table 6).",
          "Run the session on the Mini45 and the analysis. A leave-one-eye-out error above 0.1 N or 5 N·mm is about three "
          "times the simulated. It would mean a nonlinearity, a gage fault or a wrong eye geometry, none of which a 6 x 6 "
          "matrix represents.",
          "Repeat the session a day later without remounting: a change of $C$ beyond the leave-one-eye-out spread is drift "
          "of the sensor or the box.",
          "Compare the ATI and the HEX-E on the sphere presses with the HEX-E's gains from the same session. Their difference "
          "is the HEX-E's error in the ground truth of 22 October."]
    S.append(("future", "Future work", "<ol>" + "".join(f"<li>{x}</li>" for x in fw) + "</ol>"))
    return S, lede, summary


def main():
    S, lede, summary = sections(load())
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    byline = (f"Workstation, 23 October 2026, built from {code(D)} by {code(ROOT / 'scripts/ati_ft_calibration_page.py')}. "
              f"Owner's handoff note on the ATI hardware, uploaded 23 October and copied unchanged into this folder: "
              f"{link(HANDOFF, HANDOFF.name)}. The calibration runs it builds on: "
              + link(CAL / "20261022-ur5_tip_ground_truth_and_predictors.html", "UR5 ground truth and predictors") + ".")
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "ATI calibration plan",
                 "H1": "Calibration of the lab's ATI Mini45 and Mini40 on the UR5 with a weighed mass",
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
