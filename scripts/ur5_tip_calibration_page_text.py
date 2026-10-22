#!/usr/bin/env python3
"""Text of the calibration protocol page (scripts/ur5_tip_calibration_page.py builds it): sections(res, plan, h) returns
the sections as (id, heading, html), the lede and the summary for the docs index; numbers come from results.json."""
from __future__ import annotations

import numpy as np

NB = "&#8202;"          # hair space between a number and its unit


def f(x, n=2):
    return f"{x:.{n}f}"


def u(x, unit, n=2):
    return f"{x:.{n}f}{NB}{unit}"


def film_caption(res):
    return ("The simulated session of Section 7, replayed from its joint record at 40 times its speed. Left: the cell. "
            "Right: the pad over the 5" + NB + "mm sphere, with the stage, the action and the HEX-E's change from the rest "
            "before each press. The film runs through homing, the gravity sweep, the touch-offs, the pivot check, the "
            "calibration presses, the slides, rolls and twists, and the final reference press.")


def byline(res):
    return (f"Built from results.json ({res['when']}) of the simulated session {res['session']}. The HEX-E's numbers come "
            "from its data sheet; every result in Section 7 is simulated. The program is still to run on the robot.")


def sections(res, plan, h):
    fig, table, link, code, esc = h["fig"], h["table"], h["link"], h["code"], h["esc"]
    MEDIA = h["MEDIA"]
    ck = plan["checks"]
    cnt = ck["counts"]
    n_act = len(plan["actions"])
    pl = plan["payload"]
    rv = {r["name"]: r for r in res.get("registration_vs_truth", [])}
    cad, tch, piv = rv.get("CAD and homing"), rv.get("Touch-offs"), rv.get("Touch-offs with the pivot folded in")
    pc = res.get("pivot_check", {})
    gn = res["registration_notes"].get("gravity", {})
    an = res["analysis"]
    gt = an.get("ground_truth_vs_sim", {})
    est = an.get("estimator_vs_ground_truth", {})
    tru = res.get("truth", {})
    dur = res["duration_min"]
    reach = ck["reachable_by_region"]
    allp = ck["points_by_region"]
    pm = ck["placement_mm_deg"]
    S = []

    # ------------------------------------------------------------------------------------------------ lede
    lede = (f"The fingertip is calibrated on the lab's UR5 by pressing its pad on a printed 5{NB}mm sphere. One program "
            f"runs the {n_act} actions and logs the HEX-E force-torque sensor, the board and the robot. The lab's HEX-E is "
            f"35{NB}mm tall, so the pad's face centre lies 128.4{NB}mm from the flange. The session measures the tool's pose "
            f"on the flange itself, from a homing on the indenter's key and 38 touch-offs. In a simulated session with "
            f"injected errors, the sphere's place on the pad was known within {u(tch['tan_median'], 'mm')} (median). The "
            f"normal force was known within {100 * gt['relative_normal_force']['median']:.1f} %.")
    summary = (f"A protocol and program calibrate the magnetic fingertip on the lab's UR5 (CB3) and OptoForce HEX-E. The "
               f"HEX-E is 35 mm tall, so the face centre is 128.4 mm from the flange. Homing on the indenter's key, a gravity "
               f"sweep and 38 touch-offs measure the tool's pose; {cnt['grid']} presses and "
               f"{cnt['shear'] + cnt['roll'] + cnt['twist']} slides, rolls and twists follow. In a simulated session the "
               f"contact point on the pad is known within {f(tch['tan_median'])} mm (median) and the normal force within "
               f"{100 * gt['relative_normal_force']['median']:.1f} %.")

    # ------------------------------------------------------------------------------------------------ terms
    frames = table(["Frame", "Origin and axes"],
                   [["B", "The robot's base, as the controller reports poses."],
                    ["E", "The robot's flange (tool0)."],
                    ["S", "The HEX-E, its moment origin on its tool face."],
                    ["M", "The mount: origin on the HEX-E's tool face, z along the tool axis toward the tip, x toward the "
                          "palmar side and the HEX-E's dowel."],
                    ["P", "The pad: the estimator's tip frame, x along the face's normal, y across the finger, z toward the "
                          "tip end, origin on the pad's mount face."],
                    ["I", "The indenter: z up from its base, the sphere's centre at (0, 0, 41) mm."]],
                   caption="Frames. T<sub>XY</sub> is the pose of frame Y in frame X.", n=1)
    S.append(("terms", "Terms and metrics", (
        '<dl class="terms">'
        "<dt>HEX-E</dt><dd>OptoForce's six-axis force-torque sensor of 2017 (model HEX-70-XE-200N). It sits between the "
        "robot's flange and the mount, and its Ethernet DAQ streams over UDP.</dd>"
        "<dt>CB3</dt><dd>The UR5's controller generation in the lab. Its kinematics are UR's DH parameters for the UR5.</dd>"
        "<dt>TCP</dt><dd>Tool centre point: here the pad's frame on the flange, $T_{EP}$. The program keeps the "
        "controller's TCP at zero and computes every flange pose itself.</dd>"
        "<dt>Homing</dt><dd>Seating the mount's homing socket on the key of the indenter's post, in freedrive. The owner "
        "calls it the indexer.</dd>"
        "<dt>Touch-off</dt><dd>A slow approach along a surface's normal. Its onset is the flange position where the push "
        "starts, fitted to the push against the travel.</dd>"
        "<dt>Registration</dt><dd>A least-squares fit of $T_{EM}$, $T_{MP}$ and the sphere's centre to the homing and "
        "the touch-offs.</dd>"
        "<dt>Pivot check</dt><dd>Presses of the face centre from four turns about its normal. The board's change shows "
        "where each press landed.</dd>"
        "<dt>Press, level, hold, rest</dt><dd>A press goes along the surface's normal from 1.5" + NB + "mm off. At each "
        "force level the robot stops for a 0.6" + NB + "s hold. A rest of 0.5" + NB + "s, off the surface, comes before and "
        "after.</dd>"
        "<dt>Slide, roll, twist</dt><dd>Turns of the pressed pad. A slide turns about the sphere's centre, so the contact "
        "slides at constant indentation. A roll turns about the contact point, so the contact rolls. A twist turns about "
        "the normal through the contact.</dd>"
        "<dt>Push, shear, torsion</dt><dd>The force's component along the inward normal (N), its component in the "
        "tangent plane (N), and the moment about the normal through the contact point (N" + NB + "mm).</dd>"
        "<dt>Contact point error</dt><dd>How far apart two estimates of the sphere's place on the pad are, along the "
        "surface (mm). The ground truth is compared through the sphere's centre in the pad frame. The estimator is "
        "compared through its contact point.</dd>"
        "<dt>Ground truth</dt><dd>The contact point, normal and force of a hold, from the robot, the HEX-E and the "
        "registration (Section 5).</dd>"
        "<dt>Estimator</dt><dd>The board's model-based contact estimator "
        f"({link(h['EST_PAGE'], 'model-based contact estimation')}), run as the live viewer runs it.</dd></dl>" + frames)))

    # ------------------------------------------------------------------------------------------------ HEX-E and stack
    stack = [
        ("UR5 flange face (tool0)", "0", "", "ISO 9409-1-50-4-M6"),
        ("HEX-E tool face (UR adapter built in)", "35.0", "", "Data sheet: 35 mm tall, 70 mm across"),
        ("Mount v3: adapter top, the foot on it", "50.0", "", "Adapter 15 mm"),
        ("Foot top, where the column starts", "63.0", "", "Foot 13 mm"),
        ("Homing socket's centre", "84.0", "17.0", "On the column's palmar face"),
        ("PIP axis of the DIP link", "92.0", "0", "Mount v3 parameters"),
        ("Pad's tip frame origin", "117.50", "2.05", "DIP link b v4 in mount v3"),
        ("Face centre, the design's contact point", "128.36", "10.55", "The pad's apex"),
        ("Tip end", "138.91", "", ""),
    ]
    sec = (
        "<p>The lab's sensor is OptoForce's HEX-E of 2017. It is 35" + NB + "mm tall with the UR adapter built in, 70" + NB +
        "mm across and 260" + NB + "g. The mount was drawn on the later OnRobot HEX-E QC, which is 50" + NB + "mm tall and "
        "weighs 347" + NB + "g. The HEX-E's tool face has four M6 threads 7" + NB + "mm deep on a 50" + NB + "mm circle and a "
        "6" + NB + "mm dowel hole. This matches the owner's finding of \"not much thread on the hex e\". Table 2 gives the data "
        "sheet and Table 3 the stack on the flange.</p>"
        + table(["Quantity", "F<sub>xy</sub>", "F<sub>z</sub>", "T<sub>xy</sub>", "T<sub>z</sub>"],
                [["Nominal capacity", "200 N", "200 N", "10 N m", "6.5 N m"],
                 ["Noise, 1 sigma over one second at no load", "0.035 N", "0.15 N", "0.002 N m", "0.001 N m"],
                 ["Noise-free resolution", "0.2 N", "0.8 N", "0.010 N m", "0.002 N m"],
                 ["Deformation at nominal capacity", "1.7 mm", "0.3 mm", "2.5 deg", "5 deg"],
                 ["Nonlinearity, hysteresis, crosstalk", "Under 2, 2 and 5 % of full scale", "", "", ""],
                 ["Sampling", "Up to 500 Hz", "", "", ""]],
                caption="OptoForce HEX-E, data sheet V1.0 of 2017-07-07 (" + f'<a href="{h["DATASHEET"]}">PDF</a>' +
                        "); the UR mounting is in OptoForce's user guide for UR (" + f'<a href="{h["UR_GUIDE"]}">PDF</a>' +
                        ").", n=2)
        + table(["Feature", "Along the tool axis (mm)", "Toward the palmar side, x<sub>M</sub> (mm)", "Source"],
                [list(r) for r in stack], num_cols=(1, 2),
                caption="The tool stack from the UR flange. The pad frame P is the mount frame M shifted by "
                        "(2.05, 0, 82.50) mm.", n=3)
        + "<p>The face centre therefore lies 128.36" + NB + "mm from the flange along the tool axis. The mount's pages "
        f"list 143.36{NB}mm, which is 15{NB}mm too far. Their payload of 0.471{NB}kg is 87{NB}g too heavy. The payload for "
        f"the pendant is {pl['kg']:.3f}{NB}kg, with its centre of gravity at "
        f"({', '.join(f'{v:.1f}' for v in pl['cog_mm'])}){NB}mm from the flange. The program sets it through RTDE.</p>"
        "<p>The turn of the mount about the tool axis depends on how the HEX-E and the mount's dowel sit on the flange. "
        "The session measures it. Before the session, the stack is measured once with calipers, from the flange face to the "
        "HEX-E's tool face. A value other than 35.0" + NB + "mm goes into the plan's " + code("geometry.T_EM") + ".</p>"
        "<p>The program polls the HEX-E's DAQ one sample at a time at 500" + NB + "Hz, in a thread. magpie_control's reader "
        "has done the same since Niraj Pudasaini's commit 806b94a of 2026-03-06, and so has magpie_force_control's C++ "
        "driver. The reader before it, of 2024-11, slept 20" + NB + "ms after each request and read at most 50" + NB +
        "Hz. Its socket also kept late answers, and each late answer passed for a newer one, so its readings fell further "
        "behind over a run. The program stops the DAQ and empties the socket before it starts. Before each request it "
        "empties the socket again and counts what it drops. It drops repeated samples by their sequence numbers. Against "
        "a fake DAQ on this computer, its samples were 0.0" + NB + "ms old at the median, both after late answers and "
        "with a DAQ left streaming by an earlier program. In the second case magpie_control's current reader returned "
        "samples 64" + NB + "ms old. The default address is 192.168.0.5; the C++ driver uses 192.168.0.3.</p>")
    S.append(("stack", "OptoForce HEX-E and the tool stack on the flange", sec))

    # ------------------------------------------------------------------------------------------------ the cell
    still = MEDIA / "20261022-cell_stills.png"
    sec = (
        f"<p>The indenter goes on a 38{NB}mm block clamped to the plywood; a 2 by 4 laid flat will do. Its base stands "
        f"{-pm[1]:.0f}{NB}mm in front of the robot's base, with its x axis pointing away from the robot. At homing the "
        "finger then points away from the robot.</p>"
        "<p>On the plywood alone, the wrist comes within 5-7" + NB + "mm of the board in the flank presses. 24 of 161 "
        "actions then fail the plan's checks. On the block only two fail: two gravity orientations that the arm cannot "
        "reach.</p>"
        "<p>The checks were run on the CB3's kinematics, on the IK branch of the homing pose with the upper arm above the "
        "table. The forearm and wrists are the URDF's collision meshes. The HEX-E is a 70" + NB + "mm cylinder; the mount "
        "and finger are their meshes. Table 4 lists the clearances kept.</p>"
        + table(["Part", "Least clearance (mm)"],
                [["Robot links", "8"], ["HEX-E", "6"], ["Mount", "3 (2 in the touch-offs on its own column)"],
                 ["DIP link, cover and cap", "0.8"], ["Pad, away from its contact", "0.3"]], num_cols=(),
                caption="Clearances to the indenter, its homing post, the block, the plywood and the table.", n=4)
        + f"<p>The sphere reaches {ck['reachable_points']} of the pad's 812 surface points along their normal. Table 5 "
        "counts them by region. The flanks are reached only above the cover's 3" + NB + "mm walls. Cap TW15H3 covers the "
        "tip end.</p>"
        + table(["Region", "Surface points", "Reached by the sphere"],
                [[r.capitalize(), allp[r], reach[r]] for r in ("face", "edge", "flank", "tip")], num_cols=(1, 2),
                caption="The pad's FE surface points by the direction of their normal: face (x component above 0.9), edge "
                        "(0.5-0.9), flank (y component above 0.7), tip end (z component above 0.7).", n=5)
        + (fig(2, still, "The cell at a touch-off on the face (top) and on the flank (bottom), from the simulated session. "
               "Grey: the UR5 and the HEX-E; blue: mount v3; cream: DIP link b v4; dark grey: its cover and cap; orange: "
               "the pad. The sphere indenter stands on the block and the plywood, its homing post beside it.")
           if still.exists() else "")
        + fig(3, MEDIA / "20261022-targets_on_the_pad.png",
              "The pad's FE surface points and the plan's targets in the pad frame. Coloured points are reached by the sphere "
              "along their normal; grey ones are behind the cover's walls, under the cap or beside the link."))
    S.append(("cell", "Cell: indenter, block and plywood", sec))

    # ------------------------------------------------------------------------------------------------ stages
    sm = res.get("stage_minutes", {})
    rows = [
        ["Preflight", "1", "Stream rates and noise; payload set; TCP zero", sm.get("preflight")],
        ["Homing", "1", "The indenter's pose in B, through the mount's CAD", sm.get("homing")],
        ["Gravity", str(cnt["gravity"]), "The HEX-E's axes, sign and gains; the payload's weight and centre", sm.get("gravity")],
        ["Column touch-offs", str(cnt["mount"]), "The mount's turn and tilt on the flange", sm.get("mount")],
        ["Pad touch-offs", str(cnt["pad"]), "The pad's surface on the flange; the sphere's centre", sm.get("pad")],
        ["Registration", "1 fit", "T<sub>EM</sub>, T<sub>MP</sub> and the sphere's centre", None],
        ["Pivot check", str(cnt["pivot"]), "Where presses from four turns land, by the board", sm.get("pivot")],
        ["Calibration presses", str(cnt["grid"]), "The board against force and contact point", sm.get("grid")],
        ["Slides", str(cnt["shear"]), "Shear in stick and in slip", sm.get("shear")],
        ["Rolls", str(cnt["roll"]), "The contact rolling", sm.get("roll")],
        ["Twists", str(cnt["twist"]), "Torsion", sm.get("twist")],
        ["Final", f"{1 + cnt['final_gravity']}", "Reference press and gravity again: drift",
         (sm.get("final") or 0) + (sm.get("final_gravity") or 0)],
    ]
    sec = (
        table(["Stage", "Actions", "Measures", "Minutes (simulated)"], [[a, b, c, "" if d is None else f"{d:.1f}"] for a, b, c, d in rows],
              num_cols=(1, 3), caption=f"The session's stages in their order. The simulated session took {dur:.0f} min.", n=6)
        + "<h3>Homing on the indexer</h3>"
        "<p>The program puts the robot in freedrive. The operator seats the column's socket on the indenter's key, with "
        "the pad's face down and the finger along the indenter's +x. The key's faces are drafted 20" + NB + "deg. They "
        "centre the socket across the finger and in height, and they fix its turn. Along the finger the key has 0.15" + NB +
        "mm of play each way. The robot records its flange for 1" + NB + "s and lifts 12" + NB + "mm off the key. The "
        "seating is repeated three times. With $T_{IM}^{h}$ the mount's pose at homing from the CAD,</p>"
        "$$T_{BI} = \\bar T_{BE}^{h}\\, T_{EM}\\, (T_{IM}^{h})^{-1},$$"
        "<p>where $\\bar T_{BE}^{h}$ is the mean of the three seatings. At homing the face centre stands 30" + NB + "mm to "
        "+y of the sphere and 5" + NB + "mm above it.</p>"
        "<h3>Gravity sweep</h3>"
        f"<p>The tool is held at {cnt['gravity']} orientations, 150{NB}mm above the indenter. The finger is level and rolled "
        "about itself in 60" + NB + "deg steps, tilted down 45 and 90" + NB + "deg, and tilted up 30" + NB + "deg. At each, "
        "the HEX-E is averaged over 1.5" + NB + "s. Let $\\hat g_E = R_{BE}^{\\top}(0, 0, -1)$ be the direction of gravity "
        "in the flange frame. Two linear least-squares fits give</p>"
        "$$f_S = A\\, \\hat g_E + o_f, \\qquad \\tau_S = B\\, \\hat g_E + o_\\tau .$$"
        "<p>Here $A = m g\\, K$. The matrix $K$ maps a force on the tool in the flange frame to the HEX-E's reading: its "
        "axes, its sign, its gains and its crosstalk. The weight $m g$ is taken as the mean singular value of $A$, which "
        "keeps the factory gain. The HEX-E's turn and sign come from the polar factor of $A$. The payload's centre $c$ "
        "comes from $B = m g\\, K_\\tau [c]_\\times$. The board's rest readings are logged at the same orientations; they "
        "follow Earth's field.</p>"
        "<h3>Touch-offs</h3>"
        "<p>Each touch-off starts 1.5" + NB + "mm off the surface; on the rigid column it starts 2" + NB + "mm off. The "
        "robot approaches along the normal at 0.15" + NB + "mm/s, or 0.1" + NB + "mm/s on the column. It stops when the "
        "push reaches 0.5" + NB + "N, or 0.8" + NB + "N on the column. Two column touch-offs, at $z_M$ = 52 and 68" + NB +
        "mm, push on to 5" + NB + "N. On the rigid column the travel past the onset is the HEX-E's and the robot's "
        "deflection along $x_M$, the direction of the face presses. By the data sheet it is 0.12 and 0.16" + NB + "mm at "
        "5" + NB + "N; the two heights separate the HEX-E's shift from its tilt. The push $y$ against the travel $s$ is "
        "fitted by</p>"
        "$$y = a\\,(s - s_0)_+^{\\,p} + b, \\qquad p \\in \\{1, 1.25, 1.5, 1.75, 2\\}.$$"
        "<p>The onset $s_0$ is searched over the 0.5" + NB + "mm before the push rises. The rise is found on a "
        "50" + NB + "ms moving average, because single samples of the HEX-E's noise cross small thresholds. At the onset "
        "the push is zero, so the compliance of the HEX-E and of the pad does not enter it. The flange's pose at the onset, "
        "$T_{BE}^{(k)}$, is the touch's datum.</p>"
        "<h3>Registration</h3>"
        "<p>One Levenberg-Marquardt fit has 15 parameters. They are small motions of the mount on the flange and of the "
        "pad on the mount (a rotation vector and a translation each), and the shift of the sphere's centre $c_B$. It "
        "minimises</p>"
        "$$\\sum_{\\mathrm{pad}} \\frac{\\big(d_P(T_{PB}^{(k)} c_B) - R\\big)^2}{\\sigma_p^2} + "
        "\\sum_{\\mathrm{column}} \\frac{\\big(n_M^{\\top} T_{MB}^{(k)} c_B - 17 - R\\big)^2}{\\sigma_f^2} + "
        "\\big\\| \\Sigma_h^{-1/2} R_{BI}^{\\top} (c_B - c_B^{h}) \\big\\|^2 + \\|\\delta_{EM}\\|^2_{\\Sigma_{EM}^{-1}} + "
        "\\|\\delta_{MP}\\|^2_{\\Sigma_{MP}^{-1}} .$$"
        "<p>In it, $T_{PB}^{(k)} = (T_{BE}^{(k)} T_{EM} T_{MP})^{-1}$ and $d_P$ is the signed distance to the pad's surface "
        "(its 1556 FE triangles). The sphere's radius is $R = 5$" + NB + "mm, $n_M$ is the column face's normal and "
        "$c_B^{h}$ is the sphere's centre from homing. Table 7 gives the widths.</p>"
        + table(["Term", "Width"],
                [["Pad touch-offs, &#963;<sub>p</sub>", "0.03 mm"], ["Column touch-offs, &#963;<sub>f</sub>", "0.02 mm"],
                 ["Homing, along the key / across / in height", "0.25 / 0.10 / 0.12 mm"],
                 ["Mount on the flange, rotation", "0.5 deg; 2 deg about the tool axis"],
                 ["Mount on the flange, translation", "0.4 mm across; 0.6 mm along the tool axis"],
                 ["Pad on the mount", "0.6 deg; 0.3 mm"]],
                caption="Widths of the registration's terms.", n=7)
        + "<h3>Pivot check</h3>"
        "<p>The face centre is pressed to 1.5 and 3" + NB + "N with 1.5" + NB + "s holds. The presses are at twist 0, at "
        "four offsets of 0.5" + NB + "mm along the tangents $t_1$ and $t_2$, and at four turns of 15 to 60" + NB + "deg "
        "about the normal. The turns go in the direction that keeps the mount clear of the homing post. The offset presses "
        "give the board's change per millimetre, $J$ (15 by 2). A turned press whose change at 3" + NB + "N differs by "
        "$\\Delta b$ from the reference's sits $\\Delta p = J^{+} \\Delta b$ from it.</p>"
        "<p>The program prints these offsets beside the ones the registration predicts. It warns above 0.5" + NB + "mm, "
        "which a wrong TCP or a shifted seat would produce. With " + code("--pivot-fit") + " the offsets also enter the "
        "fit, with a width of 0.10" + NB + "mm. In simulation that made the contact point worse (Section 7), so by default "
        "the pivot is only a check.</p>"
        "<h3>Presses</h3>"
        "<p>Each press starts 1.5" + NB + "mm off and moves at 0.5" + NB + "mm/s. It holds 0.6" + NB + "s at each force "
        "level of Table 8, then returns to the rest at 2" + NB + "mm/s. The reference press comes first and after every "
        "15 presses: the face centre to 0.5, 1 and 2" + NB + "N at twist 0. Each target turns about the vertical through "
        "its contact, its twist. The plan takes the first twist of 0, &#177;30, &#177;60 ... 180" + NB + "deg at which "
        "every pose of the press reaches and clears.</p>"
        + table(["Region", "Force levels (N)"],
                [["Face", "0.25, 0.5, 1, 2, 3.5, 5"], ["Edge", "0.25, 0.5, 1, 2, 3"], ["Flank", "0.25, 0.5, 1, 2"]],
                caption="Force levels of the calibration presses.", n=8)
        + "<h3>Slides, rolls and twists</h3>"
        "<p>The pad is pressed to 1" + NB + "N, and also to 2" + NB + "N at the face centre. It then turns in 0.5" + NB +
        "deg steps along a blended path and holds 0.4" + NB + "s at each angle. A turn by $\\theta$ about the sphere's "
        "centre $c$ slides the pad over the sphere by $R\\theta$, at constant indentation:</p>"
        "$$T_{BE}' = \\mathrm{Rot}(c, a, \\theta)\\, T_{BE}, \\qquad a = R_{BI}\\,(e_z \\times u), \\qquad s = R\\,\\theta .$$"
        "<p>Here $u$ is the sliding direction in the indenter's plane. Table 9 gives the angles.</p>"
        + table(["Move", "About", "Angles or distances"],
                [["Slide in stick", "The sphere's centre", "0.1, 0.2, 0.3 mm in 8 directions"],
                 ["Slide in slip", "The sphere's centre", "0.5 and 1 mm along and across the finger, and back"],
                 ["Roll", "The contact point, axis in the tangent plane", "3 and 6 deg each way, along and across"],
                 ["Twist", "The normal through the contact", "5, 10 and 20 deg each way, at 1 and 2.5 N"]],
                caption="Turns of the pressed pad.", n=9)
        + "<h3>Safety and logging</h3>"
        "<p>The robot stops (stopL) and the action ends on any of these: a HEX-E change past 10" + NB + "N or 1.5" + NB +
        "N" + NB + "m, no HEX-E sample for 100" + NB + "ms, or a protective stop. The robot then lifts 5" + NB + "mm and the "
        "next action starts. Three aborts in a row end the session. A load change past 8" + NB + "N while travelling also "
        "stops it. Targets more than 400" + NB + "mm from the indenter are refused.</p>"
        "<p>Large moves first lift the pad 60" + NB + "mm above the indenter's top. Their joints are solved on the homing "
        "branch, as the plan checked them. " + code("--dry 15") + " adds 15" + NB + "mm to every standoff: the whole session "
        "without contact, for reach and cables. " + code("--step") + " asks before each action.</p>"
        "<p>Each action's start and end go to " + code("events.jsonl") + " with fsync. The streams go to " + code("ft.bin") +
        ", " + code("board.bin") + " (the viewer's format) and " + code("robot.bin") + ". " + code("--resume") + " skips the "
        "actions that ended ok and reloads the registration.</p>")
    S.append(("stages", "Stages of the session", sec))

    # ------------------------------------------------------------------------------------------------ ground truth
    sec = (
        "<p>Each hold gives one ground-truth sample. Its inputs are the flange's mean pose $T_{BE}$ over the hold and the "
        "HEX-E's contact wrench $(f_S, \\tau_S)$. The HEX-E deflects under the wrench by $\\delta_S = (C_r \\tau_S, C_t f_S)$, "
        "with $C_r$ and $C_t$ from its data sheet. The pad's pose and the sphere's centre in the pad frame are then</p>"
        "$$T_{BP} = T_{BE}\\, T_{ES}\\, \\exp(\\delta_S)\\, T_{ES}^{-1}\\, T_{EP}, \\qquad c_P = T_{BP}^{-1} c_B .$$"
        "<p>The contact point $p$ is the pad's surface point whose normal passes through $c_P$, and $n$ is that normal. "
        "The indentation is $R - d_P(c_P)$. The force on the pad is the HEX-E's change from the rest before the press, less "
        "the change of the payload's weight between the rest's orientation and the hold's:</p>"
        "$$f_P = R_{PE}\\, K^{-1} \\big(f_S - f_S^{\\mathrm{rest}} - A(\\hat g_E - \\hat g_E^{\\mathrm{rest}})\\big), \\qquad "
        "f_n = -f_P^{\\top} n .$$"
        "<p>The shear is $f_P + f_n n$. The torsion is $\\tau_n = n^{\\top}\\big(\\tau_E - (p_E - o_E) \\times f_E\\big)$, "
        "with $\\tau_E$ the torque change about the HEX-E's origin $o_E$ in flange axes.</p>"
        "<p>The board's change is the hold's mean reading less the rests before and after the press, interpolated to the "
        "hold's time. Each chip is first divided by its gain at its die temperature, $1 + \\kappa_c (T_c - 25)$, as the viewer "
        "corrects it.</p>")
    S.append(("truth", "Ground truth of each hold", sec))

    # ------------------------------------------------------------------------------------------------ simulator
    t_ = tru
    inj = [
        ["Mount on the flange", "Turned 0.90 deg about the tool axis; shifted 0.25, -0.20, 0.45 mm", "Touch-offs"],
        ["Pad in its seat", "Turned 0.3-0.4 deg; shifted 0.15, -0.12, 0.20 mm", "Pad touch-offs"],
        ["HEX-E height", f"+{t_.get('hex_height_err_mm', 0.3):.1f} mm", "Touch-offs"],
        ["HEX-E axes on the flange", f"Turned {t_.get('hex_turn_deg', 7.3):.1f} deg", "Gravity sweep"],
        ["HEX-E crosstalk", f"{100 * t_.get('crosstalk', 0.012):.1f} % rms", "Gravity sweep"],
        ["HEX-E offsets", "Up to 2.1 N and 0.05 N m, drifting", "Tare before each press"],
        ["Payload past the HEX-E", f"{1e3 * t_.get('payload_kg', 0.184):.0f} g", "Gravity sweep"],
        ["Indenter's place", "Shifted 1.5, -2.0, 0.8 mm; plywood tilted 0.2 deg", "Homing"],
        ["Homing seat", "0.10 mm along the key, 0.03 across, 0.04 in height", "Touch-offs"],
        ["Robot kinematics", f"Each DH parameter off by {t_.get('dh_err', 0.08):.2f} mm or mrad (1 sigma)", "None: left in"],
        ["Robot repeatability", f"{t_.get('repeat_mm', 0.01):.2f} mm per move", "None: left in"],
        ["Pad", f"{t_.get('pad_soft', 1.15):.2f} times the FE's compliance; friction {t_.get('mu', 1.0):.1f}", "Presses"],
    ]
    sec = (
        "<p>The program runs unchanged against a simulator, " + code("ur5_tip_calibration_sim.py") + ". The simulator "
        "answers the same calls as ur_rtde, the HEX-E's stream and the board's reader. Its clock is virtual and advances "
        "with the program's sleeps. It models the following:</p><ul>"
        "<li>The robot follows trapezoidal speed profiles on the CB3's kinematics. Poses are reported from the nominal "
        "parameters, and the tool sits where perturbed ones put it.</li>"
        "<li>The contact is solved at each HEX-E sample on the pad's FE surface: the compliance between its 812 points, "
        "with the HEX-E's compliance in series.</li>"
        "<li>Normal contact with the sphere is a linear complementarity problem. Coulomb friction acts on the indenter's "
        "motion along the pad since the contact began.</li>"
        "<li>The HEX-E's reading includes the payload's weight, its axes, crosstalk, offsets and data-sheet noise.</li>"
        "<li>The board's reading comes from the magnet model of the as-found calibration (magnets at 0.35 of N52, "
        "checkerboard). It includes Earth's field turned into the pad's frame, the chips' warm-up and their noise.</li></ul>"
        + table(["Error", "Injected", "Measured by"], inj, caption="Errors injected into the simulated session.", n=10))
    S.append(("sim", "Simulator of the session", sec))

    # ------------------------------------------------------------------------------------------------ simulated session
    rows = res.get("registration_vs_truth", [])
    ab = res.get("actions_abort", [])
    on = res.get("touch_onsets", [])
    on_pad = [o["s0_minus_expected"] for o in on if o["kind"] == "pad"]
    on_face = [o["s0_minus_expected"] for o in on if o["kind"] == "mount"]
    gtr = res["gravity_truth"]
    lv = an.get("estimator_by_level", {})
    lvm = an.get("estimator_masked_by_level", {})
    sec = (
        f"<p>The whole plan ran in {dur:.0f}{NB}min of the virtual clock. {res['actions_ok']} actions ended ok"
        + (f" and {len(ab)} were aborted." if ab else " and none was aborted.") + " Figure 1 shows the session.</p>"
        "<h3>Gravity sweep</h3>"
        + table(["Quantity", "Fitted", "Simulated"],
                [["Payload past the HEX-E's sensing element", f"{1e3 * gn['mass_kg']:.0f} g", f"{1e3 * gtr['mass_kg']:.0f} g"],
                 ["Its centre from the HEX-E's origin (mm)", ", ".join(f"{v:.1f}" for v in gn["com_mm"]),
                  ", ".join(f"{v:.1f}" for v in gtr["com_mm"])],
                 ["HEX-E's turn about the tool axis", f"{abs(gn['z_turn_deg']):.2f} deg", f"{gtr['turn_deg']:.2f} deg"],
                 ["Residual, force / torque", f"{1e3 * gn['rms_f_N']:.1f} mN / {1e6 * gn['rms_t_Nm']:.0f} uN m", ""]],
                num_cols=(1, 2), caption="The gravity sweep's fit against the simulator.", n=11)
        + "<h3>Touch-offs and registration</h3>"
        f"<p>The pad's onsets fell {min(on_pad):+.2f} to {max(on_pad):+.2f}{NB}mm from where the CAD and homing put the "
        f"surface. The column's fell {min(on_face):+.2f} to {max(on_face):+.2f}{NB}mm from it. The onset fits' residuals "
        f"were {1e3 * np.median([o['rms_N'] for o in on]):.0f}{NB}mN (median), the HEX-E's noise.</p>"
        "<p>Table 12 and Figure 4 compare the tool's pose after each step with the simulator's truth. The comparison is "
        "made at the face centre and at the contact point of every press pose in the plan. The true contact goes through "
        "the perturbed kinematics.</p>"
        + table(["Registration", "Face centre (mm)", "Pad axes (deg)", "Sphere's centre (mm)", "Contact, median (mm)",
                 "Contact, 95th percentile (mm)", "Contact, largest (mm)", "Indentation, median (mm)"],
                [[esc(r["name"]), f(r["face_centre_mm"], 3), f(r["axes_deg"]), f(r["centre_mm"], 3), f(r["tan_median"], 3),
                  f(r["tan_p95"], 3), f(r["tan_max"], 3), f(r["nrm_median"], 3)] for r in rows], num_cols=tuple(range(1, 8)),
                caption=f"Errors against the simulator's truth: after homing alone, after the touch-offs, and with the pivot's "
                        f"offsets folded into the fit (an earlier session). Contact points over the plan's "
                        f"{rows[0]['n'] if rows else 0} press poses.", n=12)
        + (fig(4, MEDIA / "20261022-registration_against_truth.png",
               "Left: error of the face centre's position and of the pad's axes on the flange after each step. Right: "
               "distribution of the contact point's error over the plan's press poses.") if rows else "")
        + (f"<p>The touch-offs bring the contact point from {u(cad['tan_median'], 'mm')} to {u(tch['tan_median'], 'mm')} "
           f"(median), and to {u(tch['tan_max'], 'mm')} at worst. The indentation's error falls from "
           f"{u(cad['nrm_median'], 'mm')} to {u(tch['nrm_median'], 'mm', 3)}. What remains is the robot's kinematic error. It "
           "changes with the joints, while the registration fits one tool pose for every orientation.</p>" if tch and cad else "")
        + (f"<p>The pivot check put the turned presses at most {u(pc['worst_mm'], 'mm')} from where the registration "
           "predicted them. That is the size of the remaining kinematic error.</p>" if pc else "")
        + (f"<p>In an earlier session the pivot's offsets were folded into the fit. The sphere's centre then ended "
           f"{u(piv['centre_mm'], 'mm')} from the truth, and the contact point's median error grew to "
           f"{u(piv['tan_median'], 'mm')}. The board's offsets carried about 0.1{NB}mm of bias with the as-found magnets. "
           "One tool pose also cannot follow the robot's error from one turn to the next.</p>" if piv else "")
        + "<h3>Ground truth against the simulator</h3>"
        + table(["Quantity", "Median", "90th percentile", "Holds"],
                [["Contact point along the surface (mm)", f(gt["location_mm"]["median"], 3), f(gt["location_mm"]["p90"], 3), gt["location_mm"]["n"]],
                 ["Sphere's centre along the normal (mm)", f(gt["depth_mm"]["median"], 3), f(gt["depth_mm"]["p90"], 3), gt["depth_mm"]["n"]],
                 ["Normal force (N)", f(gt["normal_force_N"]["median"], 3), f(gt["normal_force_N"]["p90"], 3), gt["normal_force_N"]["n"]],
                 ["Normal force above 0.4 N, relative", f"{100 * gt['relative_normal_force']['median']:.1f} %",
                  f"{100 * gt['relative_normal_force']['p90']:.1f} %", gt["relative_normal_force"]["n"]],
                 ["Shear (N)", f(gt["shear_N"]["median"], 3), f(gt["shear_N"]["p90"], 3), gt["shear_N"]["n"]],
                 ["Torsion in the twists (N mm)", f(gt["torsion_Nmm"]["median"], 2), f(gt["torsion_Nmm"]["p90"], 2), gt["torsion_Nmm"]["n"]]]
                if gt else [], num_cols=(1, 2, 3),
                caption="Ground truth of every hold (Section 5) against the simulator's own contact.", n=13)
        + fig(5, MEDIA / "20261022-ground_truth_against_simulator.png",
              "Left: the ground truth's normal force against the simulator's. Middle: the error of the sphere's place on the "
              "pad. Right: shear in the slides, rolls and twists, and in the presses.")
        + "<h3>Estimator on the session</h3>"
        + (f"<p>The estimator was calibrated on the rest before the first reference press. Its fit gave the magnets "
           f"{an['estimator_calibration']['k']:.3f} of N52's strength; the simulator used 0.351. Table 14 scores it against "
           "the ground truth by force level. With the as-found magnets it places contacts from 1" + NB + "N up; at 0.25 and "
           "0.5" + NB + "N most are placed in the wrong region. Masking the surface that the cover's walls and the cap cover "
           "helps at 0.5" + NB + "N only.</p>" if est else "")
        + (table(["Force level (N)", "Holds", "Contact, median (mm)", "Contact, 90th percentile (mm)", "Over 2 mm off",
                  "Over 2 mm off, masked", "Normal force, median error"],
                 [[k, v["n"], f(v["loc_median_mm"]), f(v["loc_p90_mm"]), f"{100 * v['far_2mm']:.0f} %",
                   f"{100 * lvm[k]['far_2mm']:.0f} %" if k in lvm else "",
                   f"{100 * v['rel_force_median']:.0f} %" if v["rel_force_median"] is not None else ""] for k, v in lv.items()],
                 num_cols=tuple(range(0, 7)), caption="The estimator on the calibration presses of the simulated session, by "
                 "force level.", n=14) if lv else "")
        + (f"<p>From 1{NB}N up, its normal force reads about 15 % high. On the face's presses it reads "
           f"{an['by_region'].get('face', {}).get('force_scale', float('nan')):.2f} times the ground truth. The simulated pad "
           f"is {t_.get('pad_soft', 1.15):.2f} times softer than its FE model. A session on the robot measures this scale on "
           "the printed pad.</p>")
        + fig(6, MEDIA / "20261022-estimator_on_the_session.png",
              "The estimator on the calibration presses. Left: its normal force against the ground truth. Right: its contact "
              "point's error against the hold's force level, by region.")
        + fig(7, MEDIA / "20261022-press_and_twist_traces.png",
              "One calibration press and one twist. Top: the push along the approach and the HEX-E's torque. Bottom: the norm "
              "of the board's change from the rest. Orange bands mark the holds at the force levels, blue ones the holds at "
              "each angle of the twist."))
    S.append(("simres", "Simulated session", sec))

    # ------------------------------------------------------------------------------------------------ lab computer
    sec = (
        "<p>The robot's computer runs the session from a clone of this repository. The step-by-step runbook for the operator "
        "and for a Claude Code session there is " + code("docs/experiments/20261022-ur5_tip_calibration/README.md") + ". "
        "Its setup script pulls the plan through git-lfs and makes a Python venv with numpy, scipy, pyserial and ur_rtde. "
        "It also lists old copies of magpie_control's F/T reader, probes the robot, the DAQ and the board, and runs three "
        "stages in the simulator:</p>"
        '<pre class="cmd">'
        "git clone --depth 1 git@github.com:badinkajink/hand.git ~/hand &amp;&amp; cd ~/hand\\n"
        "bash scripts/ur5_tip_calibration_lab_setup.sh\\n"
        "P=src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py; PLAN=docs/experiments/20261022-ur5_tip_calibration/plan/20261022-plan_sphere.json\\n"
        "~/.venvs/tipcal/bin/python $P --plan $PLAN --stages preflight                  # streams, payload\\n"
        "~/.venvs/tipcal/bin/python $P --plan $PLAN --stages preflight,homing,gravity,mount   # at 30 % speed\\n"
        "~/.venvs/tipcal/bin/python $P --plan $PLAN --dry 15                            # whole session, 15 mm off\\n"
        "~/.venvs/tipcal/bin/python $P --plan $PLAN --name sphere1                      # the session\\n"
        "~/.venvs/tipcal/bin/python $P --plan $PLAN --resume logs/&lt;date&gt;-tip_calibration/&lt;session&gt;"
        "</pre>"
        "<p>Before the first run:</p><ul>"
        "<li>Measure the stack with calipers, from the flange face to the HEX-E's tool face. The plan has 35.0" + NB +
        "mm.</li>"
        "<li>Check that the HEX-E's DAQ answers on 192.168.0.5, or pass " + code("--ft") + ". Pass " + code("--board") +
        " if the board's port is not found by its id.</li>"
        f"<li>Clamp the indenter on a 38{NB}mm block to the plywood, {-pm[1]:.0f}{NB}mm in front of the robot's base. Its "
        "x axis points away from the robot and its homing post stands as in Figure 2. Homing measures the placement. "
        f"Past 20{NB}mm, 5{NB}deg of yaw or 2{NB}deg of tilt from the plan's, the program stops. The plywood is then moved, "
        "or the plan is rebuilt with " + code("--placement x,y,z,yaw --block h") + " (" +
        code("scripts/ur5_tip_calibration_plan.py") + ", 20" + NB + "s).</li>"
        "<li>Leave slack in the board's USB cable and the HEX-E's cable. The flank presses turn the tool 90" + NB + "deg "
        "about its axis.</li>"
        "<li>Run " + code("--stages preflight,homing,gravity,mount") + " first, at reduced speed on the pendant, and watch "
        "the first touch-off.</li></ul>"
        "<p>Back on the workstation, the session folder goes into " + code("logs/") + ". The analysis then gives the ground "
        "truth and the estimator's scores, and the page collects them:</p>"
        '<pre class="cmd">'
        ".venv/bin/python scripts/ur5_tip_calibration_analysis.py logs/&lt;date&gt;-tip_calibration/&lt;session&gt;\\n"
        ".venv/bin/python scripts/ur5_tip_calibration_page.py collect logs/&lt;date&gt;-tip_calibration/&lt;session&gt;"
        "</pre>")
    S.append(("run", "Running the session on the lab's computer", sec))

    # ------------------------------------------------------------------------------------------------ limitations
    sec = (
        "<p>Every result in Section 7 is simulated; the program is still to run on the robot. The simulator takes the "
        "HEX-E's errors from its data sheet. Its hysteresis, nonlinearity and temperature drift are left out, and the "
        "gravity sweep calibrates only its linear map at the payload's 1.8" + NB + "N.</p>"
        "<p>The robot's error is modelled as a perturbation of its DH parameters (0.08" + NB + "mm and mrad). A real CB3's "
        "error may be larger, and it limits the contact point (Table 12). The pad is the FE model's: linear, and seated as "
        "designed. The printed pad's shape, its seat and its creep enter only a session on the robot; the pad rocked in its "
        "seat under the owner's presses on 10-20. The pivot check relies on the board's pattern with the as-found magnets, "
        "about 15" + NB + "uT/mm at 3" + NB + "N. The tip end cannot be pressed past cap TW15H3.</p>"
        "<p>The simulator's friction acts on the contact as a whole and has no torsional part. Its twists therefore made "
        "almost no torsion: the median was 0.00" + NB + "N" + NB + "mm at 1" + NB + "N and 0.10" + NB + "N" + NB + "mm at 2.5" +
        NB + "N. The torsion error of Table 13 is the measurement's alone. A Hertz estimate for the 5" + NB + "mm sphere, "
        "with a friction coefficient of 1, gives about 0.7" + NB + "N" + NB + "mm at 1" + NB + "N and 2.3" + NB + "N" + NB +
        "mm at 2.5" + NB + "N on the real pad. The face's normal is perpendicular to the tool axis, so a 2.5" + NB + "N push "
        "puts about 230" + NB + "N" + NB + "mm of moment on the HEX-E. The torsion is about 1" + NB + "% of that. The "
        "difference between the turns of +20 and -20" + NB + "deg reverses the friction and leaves the push's moment, so "
        "it gives twice the torsion. A change of the push between the two turns still enters through the crosstalk of up "
        "to 5" + NB + "%, most at 1" + NB + "N.</p>")
    S.append(("limits", "Limitations", sec))

    # ------------------------------------------------------------------------------------------------ future work
    sec = (
        "<ul>"
        "<li>Stack height: calipers on the HEX-E and the adapter. A value more than 0.2" + NB + "mm off 35.0" + NB + "mm goes "
        "into the plan's $T_{EM}$ before the session.</li>"
        "<li>The HEX-E's gain: hang a known mass of 100-200" + NB + "g from the mount at the gravity sweep's orientations. "
        "The fitted $A$ should change by that weight times $K$; a different change rescales the factory gain.</li>"
        "<li>First session on the robot: preflight, homing, gravity and the touch-offs. The pad touch-offs should fit to "
        "about 0.01" + NB + "mm rms, as in simulation. Larger residuals mean that the printed pad or its seat differs from "
        "the CAD.</li>"
        "<li>The rod indenter: a plan with " + code("--indenter rod") + " and homing on the rod's own key, for line "
        "contacts along and across the finger.</li>"
        "<li>The N52 magnets: the same plan after the swap. The simulated estimator reaches 0.18" + NB + "mm and 0.03" + NB +
        "N with them, and the pivot check gains three times the signal.</li>"
        "<li>The neighbour's field: fix a second pad with magnets on the plywood and sweep the board past it without "
        "contact. This checks the field that the joints' kinematics will subtract.</li>"
        "<li>A learned model on the session's holds, which the analysis writes, compared with the model-based estimator on "
        "the same holds.</li>"
        "<li>The HEX-E's deflection from mount-0 and mount-2 (5" + NB + "N at $z_M$ = 52 and 68" + NB + "mm), fitted as a "
        "shift and a tilt, in place of the data sheet's in " + code("scripts/ur5_tip_calibration_analysis.py") + ". If the "
        "depth at 5" + NB + "N changes by more than 0.02" + NB + "mm, the data sheet's compliance was too coarse for the "
        "ground truth of Section 5.</li>"
        "<li>The sphere-pad contact model against the session: replay the measured pose of the indenter in the pad frame "
        "against tip b's 1" + NB + "mm pads (builder v2) in MuJoCo. Then compare push against depth at the 68 grid points, "
        "shear against slide, and the twists' torsion loop. A new script reads the session folder. The pads fail as a "
        "model of the printed tip if their depth at 2 to 5" + NB + "N is off by more than the 20-28" + NB + "% by which "
        "they missed the FE model.</li>"
        "<li>Torsion without the lever arm: the sphere on the HEX-E's axis and the finger clamped to the table. The HEX-E's "
        "$\\tau_z$ then reads the torsion, at a noise of 0.001" + NB + "N" + NB + "m per sample. The twists of this plan "
        "should also go to 5" + NB + "N, about 6" + NB + "N" + NB + "mm by the Hertz estimate.</li></ul>"
        f"<p>Status of the printed-fingertip work: {link(h['STATUS_PAGE'], 'status page')}.</p>")
    S.append(("future", "Future work", sec))
    return S, lede, summary
