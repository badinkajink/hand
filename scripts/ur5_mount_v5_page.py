#!/usr/bin/env python3
"""Result page of 2026-10-23: UR5 calibration mount v5, mount v4's fork as printed with a PLA wedge pushed into the gap
between its stop plate and the DIP link's tongue, in place of v4's three brass set screws (owner, after v4's fork went to
the P1S: "i dont have any of that hardware except for the heat inserts ... brass hardware is not really stocked in the lab").

Reads docs/experiments/20261023-ur5_calibration_mount_v5/ (parts, link and wedge checks, payload, clearances along the lab's
sphere and rod plans, the P1S plate of the wedges, the renders and the film), written by scripts/ur5_calibration_mount.py
with UR5M_REV=v5 and scripts/v3b_ur5v2_pla_plate.py mount_v5_wedge, and v3's and v4's records for the comparison; writes
docs/experiments/20261023-ur5_calibration_mount_v5/20261023-ur5_calibration_mount_v5.html.

    .venv/bin/python scripts/ur5_mount_v5_page.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import ur5_mount_v4_page as V4  # noqa: E402  (fig, film, table, esc)

D = ROOT / "docs/experiments/20261023-ur5_calibration_mount_v5"
PFX = "20261023-ur5_calibration_mount_v5"
PAGE = D / f"{PFX}.html"
TEMPLATE = ROOT / "scripts/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
D3 = ROOT / "docs/experiments/20261021-ur5_calibration_mount_v3"
P3 = "20261021-ur5_calibration_mount_v3"
D4 = ROOT / "docs/experiments/20261023-ur5_calibration_mount_v4"
P4 = "20261023-ur5_calibration_mount_v4"
CAL = ROOT / "docs/experiments/20261022-ur5_tip_calibration"
PAGES = dict(v4=D4 / f"{P4}.html", results=CAL / "20261022-ur5_tip_ground_truth_and_predictors.html",
             queue=CAL / "QUEUE.md",
             status=ROOT / "docs/experiments/20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html")
table, esc, img_uri = V4.table, V4.esc, V4.img_uri


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
    return V4.film(n, mp4, poster, caption)


def load():
    j = lambda p: json.loads(Path(p).read_text())                              # noqa: E731
    return dict(parts=j(D / f"{PFX}_parts.json"), chk=j(D / f"{PFX}_check.json"), chk3=j(D3 / f"{P3}_check.json"),
                chk4=j(D4 / f"{P4}_check.json"), pay=j(D / f"{PFX}_payload.json"), pay3=j(D3 / f"{P3}_payload.json"),
                plate=j(D / f"print/{PFX}_p1s_pla_wedge.json"), plate4=j(D4 / f"print/{P4}_p1s_pla.json"),
                clr=j(D / f"{PFX}_clearance.json"), step=j(D / f"{PFX}_step.json")["parts"])


def deg(v):
    return "over 5" if v is None else f"{v:g}"


def sections(R):
    P = R["parts"]["parameters"]
    W = R["chk"]["wedge"]
    g = W["geometry"]
    c5, c3, c4 = R["chk"]["b_links"]["v4b"], R["chk3"]["b_links"]["v4b"], R["chk4"]["b_links"]["v4b"]
    t3, t4s, t5 = c3["turn_to_contact_deg"], c4["turn_to_contact_deg_screws_set"], c5["turn_to_contact_deg_wedge_set"]
    same = all(R["chk"]["b_links"][k]["turn_to_contact_deg_wedge_set"] == t5 for k in ("v2b", "v3b"))
    seat = {s["case"]: s for s in W["seats"]}
    fr = {f["mu"]: f for f in W["friction"]}
    pl = W["plate"]
    sph, rod = R["clr"]["plans"]["sphere2"]["groups"], R["clr"]["plans"]["rod1"]["groups"]
    w_least = min(sph["stop_wedge"]["registered_mm"], rod["stop_wedge"]["registered_mm"])
    plate = R["plate"]
    sw = next(o for o in plate["objects"] if o["id"] == "SW1")
    it5 = {i["item"]: i for i in R["pay"]["items"]}
    it3 = {i["item"]: i for i in R["pay3"]["items"]}
    dm = it5["standoff_fork"]["mass_g"] - it3["standoff_fork"]["mass_g"]
    m_w, m_ins = it5["stop_wedge"]["mass_g"], it5["slotted_arm_insert"]["mass_g"]
    pay5 = 0.384 + (dm + m_w + m_ins) / 1000
    pay4 = 0.384 + (dm + 3 * 0.9 + 3 * 0.25 + 4 * 0.4) / 1000      # v4's page: set screws, brass nuts, inserts
    slope_v3 = 0.21                                                # mm/N on the face, run sphere2 (results page)
    lede = (f"Mount v5 stops the DIP link's tongue with a printed PLA wedge in place of v4's brass set screws, their nuts and "
            f"three of its inserts. With the wedge seated, the turn to contact under a face press falls from "
            f"{t3['about_y_pip_-']:g}° in mount v3 to {t5['about_y_pip_-']:g}°, the check's smallest step. The wedge is pushed "
            f"by hand into the {g['gap_nominal_mm']:.2f} mm gap between v4's stop plate and the tongue, from the tip end. It "
            f"jams on the plate's lower edge and is locked there by friction. Its {g['taper_deg']:g}° taper closes any gap from "
            f"{g['t0_mm']:.1f} to {g['t1_mm']:.1f} mm. Over the lab's sphere and rod plans, every seat of the wedge stays at "
            f"least {math.floor(w_least * 10) / 10:.1f} mm from the indenter and the cell. The fork is v4's as printed, and two "
            f"wedges print in {plate['time']}.")
    summary = (f"Mount v5 = v4's fork as printed + a PLA wedge pushed into the stop plate's gap in place of three brass set "
               f"screws: face-press turn to contact {t3['about_y_pip_-']:g} -> {t5['about_y_pip_-']:g} deg; the wedge closes "
               f"gaps of {g['t0_mm']:.1f}-{g['t1_mm']:.1f} mm and locks by friction; >= {math.floor(w_least * 10) / 10:.1f} mm "
               f"from the cell on the sphere and rod plans; two wedges in {plate['time']}; not yet measured.")
    S = []
    terms = [
        ("Frames F and M", "F is the DIP link's frame: x_F toward the dorsal side, y_F along the PIP axis, z_F along the "
         "finger, negative toward the tip. M is the mount's frame: its origin at the adapter's face on the HEX-E, z_M along "
         "the tool axis toward the tip, x_M toward the palmar side. Figures are drawn in M."),
        ("Gap at the plate's edge", "The distance along x_F, in mm, from the tongue's back (x_F 8.0) to the stop plate's "
         f"lower edge (x_F {P['stop_x'][0]:.2f}, z_F {P['stop_bot_z']:.1f}). It is {g['gap_nominal_mm']:.2f} mm in the model, "
         "with the link in the middle of its play."),
        ("Seat", "The place where the pushed wedge stops, given by the z_F of its leading end. At its seat the wedge's "
         "thickness at the plate's edge equals the gap."),
        ("Turn to contact", "The least angle, in degrees, through which the link can be turned about an axis through the "
         "bridge's middle before its mesh overlaps the fork's or the wedge's. The clamp plate and the slotted arm's screw are "
         "left off, so the number measures the fork's shape. The check steps through 0.05, 0.1, 0.15 ... 5°."),
        ("Friction coefficient", "μ, the ratio of the friction force to the normal force where PLA slides on PLA, taken as "
         "0.2 to 0.4 at both faces of the wedge."),
        ("Preload", "The normal force, in N, with which the seated wedge presses on the tongue's back."),
        ("Pull-out force", "The force along the tool axis, in N, that draws the seated wedge out again. A positive value "
         "means that the wedge stays in place without a push."),
        ("Clearance", "The least distance, in mm, from a tool part to the indenter, the cell and the other indenter over a "
         "plan's press poses and every degree of its turns. The plans require 3 mm for the mount and 0.8 mm for the finger. "
         "The wedge is checked as the space it can take over all its seats, on the mount's 3 mm."),
    ]
    S.append(("terms", "Terms and metrics",
              '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"))

    body = (f"<p>Mount v4 stops the tongue with three brass M3 set screws in heat-set inserts, locked by brass jam nuts "
            f"({link(PAGES['v4'], 'mount v4')}). The lab stocks the inserts but no brass screws or nuts. Without the screws, "
            f"v4's stop plate stands {g['gap_nominal_mm']:.2f} mm behind the tongue. A press on the face then turns the link "
            f"through its full {t3['about_y_pip_-']:g}° of play, as in v3. Steel set screws would sit 4.5 mm behind the board's "
            f"magnetometer chips. The fork of v4 was already printing on the P1S, so v5 keeps that fork and closes the gap "
            f"with a printed part.</p>")
    S.append(("hardware", "Hardware of mount v4", body))

    rows = [["Width (y_F)", f"{g['width_mm']:.1f} mm", "the plate's inner face is flat over |y_F| ≤ 8.5 mm, rounded to 9.5"],
            ["Thickness", f"{g['t0_mm']:.2f} to {g['t1_mm']:.2f} mm", "at the leading end and at the end of the incline"],
            ["Taper", f"{g['taper_deg']:g}°", "of the inclined face against the flat face"],
            ["Incline", f"{g['incline_mm']:.1f} mm", "along the tool axis"],
            ["Fin", f"{g['fin_len_mm']:.1f} mm long, face at x_F {g['fin_x_F']:.1f}", "inside the plate's outer face at x_F 16.0"],
            ["Gap at the plate's edge", f"{g['gap_nominal_mm']:.2f} mm", f"the wedge closes {g['t0_mm']:.2f} to {g['t1_mm']:.2f} mm"],
            ["Material", "PLA, solid", "no metal near the board's chips"]]
    frows = [[f"{mu:.1f}", f"{fr[mu]['preload_per_push']:.2f}", f"{fr[mu]['pull_out_per_preload']:.2f}"] for mu in sorted(fr)]
    body = (f"<p>The gap between the stop plate and the tongue runs {B_MOUNT_Z - P['stop_bot_z']:.2f} mm along the tool "
            f"axis, from the neck's step at z_F {B_MOUNT_Z:.2f} to the plate's lower edge at z_F {P['stop_bot_z']:.1f}. Its "
            f"faces are flat and parallel over the plate's width: the tongue's back at x_F 8.0 and the plate's inner face at "
            f"x_F {P['stop_x'][0]:.2f}. Toward the tip the gap is open, and the tongue's back runs on to its end at z_F -98.46 "
            f"(Fig. 2).</p>"
            f"<p>The wedge is pushed into this gap from the tip end, its flat face sliding on the tongue's back (Fig. 1). Its "
            f"inclined face rises {g['taper_deg']:g}° toward the tip and meets the plate's lower edge, where the wedge jams "
            f"(Fig. 2, right). The wedge therefore stops at the seat where its thickness at the edge equals the gap. Over the "
            f"{g['incline_mm']:.1f} mm incline the thickness grows from {g['t0_mm']:.2f} to {g['t1_mm']:.2f} mm. The plate's "
            f"print and the link's place in its {t3['about_y_pip_-']:g}° of play move the gap at the edge by about 0.5 mm "
            f"either way, and any gap in that range is closed. A fin at the trailing end is pushed with a finger and gripped "
            f"to pull the wedge out.</p>"
            f"<p>The wedge is locked by friction at its two faces (Table 2). A push of 1 N along the tool axis gives a "
            f"preload of {fr[0.4]['preload_per_push']:.2f} to {fr[0.2]['preload_per_push']:.2f} N on the tongue for μ from 0.4 "
            f"to 0.2. A pull of {fr[0.2]['pull_out_per_preload']:.2f} to {fr[0.4]['pull_out_per_preload']:.2f} N per newton of "
            f"preload draws it out again. The taper's tangent, {math.tan(math.radians(g['taper_deg'])):.2f}, is smaller than "
            f"the sum of the two friction coefficients, so the seated wedge stays in place without a push. A press on the face "
            f"raises the normal force at both faces, and with it the friction on the wedge.</p>"
            f"<p>A face press now loads the plate at its lower edge, {pl['cantilever_mm']:.1f} mm below the dorsal ear. As a "
            f"cantilever of printed PLA ($E$ = 2 GPa, {pl['width_mm']:.0f} mm wide, {pl['thickness_mm']:.2f} mm thick) the plate "
            f"bends {pl['bend_at_edge_mm_per_N']:.3f} mm per newton there. Per newton pressed on the face, the contact point "
            f"then moves about {pl['contact_travel_mm_per_N']:.3f} mm. The run sphere2 measured {slope_v3:.2f} mm per newton "
            f"on v3's face. The set screws of v4 bore on the plate 19.5 mm below the ear, where it bends about half as "
            f"much.</p>"
            + table(["Feature", "Value", "Note"], rows, num_cols=(), caption="The wedge (F frame).", n=1)
            + table(["μ at both faces", "Preload per newton of push", "Pull-out per newton of preload"], frows,
                    num_cols=(0, 1, 2), caption="The seated wedge's friction, from the balance of forces on a 10° wedge.", n=2)
            + fig(1, D / f"media/{PFX}_wedge.png", "The wedge in the stop plate's gap, seen from the dorsal side, at the "
                  "nominal gap. DIP link b v4 with cover 3S, pad and tip cap TW15H3S on v4's fork.")
            + fig(2, D / f"media/{PFX}_wedge_sections.png", "Section through the middle of the link (M frame, the tool axis "
                  "up, the palmar side to the right). Left: the wedge between the tongue's back and the plate, behind the "
                  "contact point of a press on the face (red arrow). Right: the plate's lower edge on the wedge's incline."))
    S.append(("wedge", "Wedge in the stop plate's gap", body))

    trows = []
    axes = (("about_x", "About x (the face's normal)"), ("about_y_pip", "About y (the PIP axis)"), ("about_z_tool", "About z (the tool axis)"))
    for k, lab in axes:
        for sg, nm in (("-", "the sense a face press drives") if k == "about_y_pip" else ("-", "one sense"), ("+", "other sense")):
            trows.append([f"{lab}, {nm}", deg(t3[f'{k}_{sg}']), deg(t4s[f'{k}_{sg}']), deg(t5[f'{k}_{sg}'])])
    srows = [[s["case"].capitalize(), f"{s['gap_at_edge_mm']:.2f}", f"{s['leading_end_z_F']:.2f}", f"{s['fin_end_z_F']:.2f}",
              f"{s['past_plate_end_mm']:.2f}", f"{s['past_finger_end_mm']:.2f}"] for s in W["seats"]]
    ov = W["overlaps_at_seat_mm3"]
    pos = R["chk"]["poses_table"]
    pose_ok = all(all(v == 0.0 for kk, v in pos[f"{kind}_{ps}"].items() if kk != "socket_floor_minus_pin_top_mm")
                  for kind in ("sphere", "rod") for ps in ("touch", "homing"))
    crow = []
    for nm, lab, gr in (("sphere2", "Sphere, 22 October (reg2144)", sph), ("rod1", "Rod, 23 October (rod1)", rod)):
        crow.append([lab] + [f"{gr[k]['registered_mm']:.2f} ({gr[k]['registered_at']})"
                             for k in ("mount", "stop_wedge", "finger", "hex_e")])
    nom, thin = seat["nominal link"], seat["thinnest gap"]
    thick = seat["thickest gap"]
    body = (f"<p>Seated at the nominal gap, the wedge overlaps none of the parts that it touches or passes: DIP link b v4, "
            f"cover 3S, the pad, tip cap TW15H3S, the fork and the clamp plate ({max(ov.values()):.3f} mm³ at most). Its path "
            f"in, from 25 mm beyond the seat in steps of 0.25 mm, is {'clear' if W['path_in_clear'] else 'blocked'} with "
            f"0.03 mm left at the tongue and at the plate's edge. The link still slides onto the fork before the wedge goes in, "
            f"as in v4.</p>"
            f"<p>With the wedge modelled 0.02 mm off the tongue, the turn to contact that a face press drives falls from "
            f"{t3['about_y_pip_-']:g}° to {t5['about_y_pip_-']:g}° (Table 3). The set screws of v4 give the same value. The "
            f"turn about the tool axis falls to {t5['about_z_tool_+']:g}° and {t5['about_z_tool_-']:g}°, against "
            f"{t4s['about_z_tool_+']:g}° and {t4s['about_z_tool_-']:g}° with the screws. The turn about x, which flank presses "
            f"drive, stays at {t5['about_x_+']:g}°."
            + (" DIP links b v2 and v3 give the same turns." if same else "") + "</p>"
            + table(["Turn of the link", "v3", "v4, screws set", "v5, wedge set"], trows, num_cols=(1, 2, 3),
                    caption="Turn to contact in degrees, DIP link b v4.", n=3)
            + f"<p>The seat moves {g['incline_mm']:.1f} mm along the tool axis over the wedge's range of gaps (Table 4). At "
              f"the nominal gap the fin ends {-nom['past_finger_end_mm']:.2f} mm short of the finger's end at z_F -100.06. In "
              f"the thinnest gap the fin reaches {thin['past_finger_end_mm']:.1f} mm beyond it. In the thickest gap the leading "
              f"end stops {thick['leading_end_short_of_neck_step_mm']:.2f} mm short of the neck's step.</p>"
            + table(["Case", "Gap at the edge (mm)", "Leading end (z_F, mm)", "Fin's end (z_F, mm)", "Beyond the plate's end (mm)",
                     "Beyond the finger's end (mm)"], srows, num_cols=(1, 2, 3, 4, 5),
                    caption="Seats of the wedge. A negative distance beyond the finger's end lies inside the finger.", n=4)
            + f"<p>The space that the wedge can take over all its seats was checked along the lab's two plans, with each "
              f"session's registration (Table 5). It stays {sph['stop_wedge']['registered_mm']:.2f} mm from the indenter and "
              f"the cell over the sphere plan and {rod['stop_wedge']['registered_mm']:.2f} mm over the rod plan. The mount's "
              f"other parts are v4's, and their least clearance is unchanged. It is {sph['mount']['registered_mm']:.2f} mm at "
              f"{sph['mount']['registered_at']} of the sphere plan, which the run of 22 October went through, and "
              f"{rod['mount']['registered_mm']:.2f} mm on the rod plan. At homing and at the touching pose of both indenters, "
              f"{'no part overlaps the indenter, the wedge included' if pose_ok else 'a part overlaps the indenter'}.</p>"
            + table(["Plan", "Mount (needs 3.0)", "Wedge's space (needs 3.0)", "Finger (needs 0.8)", "HEX-E (needs 6.0)"], crow,
                    num_cols=(1, 2, 3, 4), caption="Least clearance in mm with the session's registration, and the action "
                    "where it occurs.", n=5))
    S.append(("checks", "Fit, play and clearance checks", body))

    prow = [["Standoff-fork v4", "1", "printed on 2026-10-23 from mount v4's 3MF; its three stop bores stay empty"],
            ["Wedge", "2", "one to fit, one spare; PLA, the 3MF named above"],
            ["M3 x 5.7 brass heat-set insert", "1", "in the inner block, for the slotted arm's M3 x 12"],
            ["Adapter, clamp plate and their screws", "&#8211;", "v3's, as printed"]]
    st = R["step"].get("stop_wedge")
    body = (f"<p>Two wedges print on the P1S in {plate['time']} with {plate['filament_g']:.1f} g of PLA "
            f"({code(D / ('print/' + PFX + '_p1s_pla_wedge.3mf'))}). Each lies on its flat face and is printed solid at "
            f"0.1 mm with six walls. The incline therefore prints as steps 0.1 mm high and 0.57 mm long. By the plate "
            f"builder's count the extruded plastic is {sw['fill_ratio']:.2f} of the wedge's volume, below the 0.85 that it "
            f"expects of a solid part. Six walls of 0.42 mm fill most of the wedge's 1.2 to 5.5 mm section. The fork is v4's "
            f"as printed, and only the slotted arm's insert is pressed into it. The payload is {pay5:.3f} kg, against "
            f"{pay4:.3f} kg for v4 with its screws and 0.384 kg for v3."
            + (f" Read back, the wedge's STEP file has {st['readback_solids']} solid of {st['readback_volume_mm3']:.0f} mm³."
               if st else "") + "</p>"
            + table(["Part", "Quantity", "Note"], prow, num_cols=(1,), caption="Parts for v5.", n=6))
    S.append(("print", "Print, parts and payload", body))

    steps = ["Press one insert into the inner block's bore on the slotted side with a soldering iron, 0.2 mm below the face. "
             "The stop plate's three bores stay empty.",
             "Seat the fork on the adapter as in v3 (M6 x 20 captive, four M3 x 12 into the adapter's inserts).",
             "Slide DIP link b onto the fork along the tool axis as in v3. Put on the clamp plate (two M3) and the slotted "
             "arm's M3 x 12 with its washer, into the insert, and tighten.",
             "Hold the wedge with its fin away from the link and its flat face on the tongue's back. Push it into the gap "
             "from the tip end until it stops under a firm finger.",
             "Push the wedge home again before each session, since printed PLA creeps under a held load.",
             "Re-run the pad touch-offs and the registration.",
             "To take the link off, pull the fin toward the tip with two fingers or pliers first."]
    body = ("<ol>" + "".join(f"<li>{x}</li>" for x in steps) + "</ol>"
            f"<p>A firm push may turn the link within its {t3['about_y_pip_-']:g}° of play until the walls around its legs stop "
            f"it. The walls then fix the link's place, and the wedge closes the gap that is left.</p>")
    S.append(("assembly", "Assembly", body))

    body = ("<p>Before the first run on v5, a dial indicator on the DIP link's dorsal side measures its travel. A rigid part "
            "presses the pad at 1 to 5 N, once on v3 and once on v5. Then the reference press of the calibration program gives "
            "the same comparison on the robot. Its depth to 2 N was 0.71 to 0.76 mm on v3. On v5 it should fall toward the FE "
            "pad's indentation, about 0.27 mm, plus the HEX-E's 0.08 mm. A depth still above 0.5 mm places the rest of the "
            "compliance outside the DIP link's seat: in the arm, the HEX-E or the indenter's block. The fin's distance from "
            "the plate's end, read with a caliper before and after a session, shows whether the wedge creeps out.</p>")
    S.append(("proof", "Proof on the bench", body))

    body = ("<p>The wedge stops presses that push the tongue toward the dorsal side: the face and, in part, the edges. Flank "
            f"presses turn the link about x within the walls' 0.15 mm, and that turn stays at {t5['about_x_+']:g}°. The wedge "
            "is kept in place by friction alone, and its friction coefficients are the range for dry PLA on PLA. The plate's "
            "bending is a beam estimate. The design has been checked in simulation; neither the wedge nor the link's travel "
            "on v5 has been measured.</p>")
    S.append(("limits", "Limitations", body))
    fw = ["Print the two wedges and fit one on v4's fork. Measure the link's travel with a dial indicator at 1, 2 and 5 N on "
          "v3 and on v5. A travel on v5 above a third of v3's at 5 N means that the plate, its edge or the wedge gives.",
          "Run sphere2's grid again on v5 and refit the indentation beyond the FE pad with "
          + code(ROOT / "scripts/ur5_tip_calibration_fe.py") + ". A face slope still above 0.1 mm/N places the compliance "
          "outside the link's seat.",
          "Read the fin's distance from the plate's end with a caliper before and after each session. A drift above 0.5 mm "
          "means that the wedge walks out, and a detent is then added to the fin.",
          "If the flank slope (0.14 mm/N on v3) stays, close the walls around the legs from 0.15 to 0.08 mm on a test print. "
          "Flank presses turn the link about x within those gaps."]
    S.append(("future", "Future work", "<ol>" + "".join(f"<li>{x}</li>" for x in fw) + "</ol>"))
    return S, lede, summary


B_MOUNT_Z = -77.05          # z_F of the neck's step, where the tongue's back begins (B_LINK mount_z)


def main():
    R = load()
    S, lede, summary = sections(R)
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    head_film = film(0, D / f"media/{PFX}_wedge_in.mp4", D / f"media/{PFX}_wedge_in.png",
                     "The wedge slides into the gap between the stop plate and the tongue's back from the tip end, until its "
                     "incline meets the plate's lower edge. Mount v5 with DIP link b v4, cover 3S, pad and tip cap "
                     "TW15H3S.").replace("Figure 0. ", "")
    byline = (f"Workstation, built from {code(D)} by {code(ROOT / 'scripts/ur5_mount_v5_page.py')}; geometry from "
              f"{code(ROOT / 'scripts/ur5_calibration_mount.py')} with UR5M_REV=v5. Mount v4: {link(PAGES['v4'], 'its page')}; "
              f"the measurement that led here: {link(PAGES['results'], 'UR5 ground truth and predictors')}; "
              f"status: {link(PAGES['status'], 'status page')}.")
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "UR5 mount v5", "H1": "Calibration mount v5 with a PLA wedge behind the DIP link's tongue",
                 "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-23", "LEDE": lede, "BYLINE": byline,
                 "FILM": head_film, "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS}.items():
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
