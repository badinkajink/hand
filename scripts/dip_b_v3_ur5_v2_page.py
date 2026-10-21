#!/usr/bin/env python3
"""Result page of 2026-10-20: DIP link concept b v3 and the UR5e calibration mount v2, the owner's notes after printing v2 b
and mount v1 turned into geometry, with the checks that each change keeps the joint's clearance, the walls, the pad's
stiffness and readout, the link's fit in the mount and the homing.

Reads docs/experiments/20261018-dip_link_board_mount/{v2,v3}/b/evaluation.json and params/20261020-v3_concept_b.json
(scripts/dip_link_v2.py b with DLV2_PARAMS_B), docs/experiments/20261020-ur5_calibration_mount_v2/*_check.json, _parts.json,
_step.json and print/*.json (UR5M_REV=v2 scripts/ur5_calibration_mount.py, scripts/v3b_ur5v2_pla_plate.py), the finite
elements of b's pad with and without the walls (docs/experiments/20261019-v2_tip_fe_checks/fe.jsonl, sensing.json), applies
scripts/retro_style.py and writes docs/experiments/20261020-ur5_calibration_mount_v2/20261020-dip_link_b_v3_and_ur5_mount_v2.html.

    .venv/bin/python scripts/dip_b_v3_ur5_v2_page.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import retro_style  # noqa: E402
import v2_tip_fe_checks_page as VP  # noqa: E402  (table, img_uri, video_uri, films_html, rows, esc)

D = ROOT / "docs/experiments/20261020-ur5_calibration_mount_v2"
PAGE = D / "20261020-dip_link_b_v3_and_ur5_mount_v2.html"
TEMPLATE = ROOT / "scripts/dip_b_v3_ur5_v2_page.template.html"
DIP = ROOT / "docs/experiments/20261018-dip_link_board_mount"
FE = ROOT / "docs/experiments/20261019-v2_tip_fe_checks"
UP = "20261020-ur5_calibration_mount_v2"
PAGES = dict(dip=DIP / "20261018-dip_link_magnetometer_board_mount.html",
             ur5=ROOT / "docs/experiments/20261018-ur5_calibration_mount/20261018-ur5_calibration_mount.html",
             fe=FE / "20261019-v2_fingertip_stiffness_checks.html",
             status=ROOT / "docs/experiments/20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html")
table, img_uri, esc, rows = VP.table, VP.img_uri, VP.esc, VP.rows


def rel(p):
    return os.path.relpath(p, D)


def link(p, text):
    return f'<a href="{rel(p)}">{text}</a>'


def code(p):
    return f"<code>{esc(os.path.relpath(p, ROOT))}</code>"


def fig(n, png, caption, width=1100):
    return (f'<figure id="fig{n}"><img src="{img_uri(png, width)}" alt="{esc(caption.split(".")[0])}">'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def load():
    j = lambda p: json.loads(Path(p).read_text())                              # noqa: E731
    ev = {k: j(DIP / f"{k}/b/evaluation.json") for k in ("v2", "v3")}
    par = j(DIP / "params/20261020-v3_concept_b.json")
    pv = {q["key"]: q["value"] for q in par["parameters"]}
    chk = j(D / f"{UP}_check.json")
    parts = j(D / f"{UP}_parts.json")
    step = j(D / f"{UP}_step.json")
    plates = {k: j(D / f"print/20261020-p1s_pla_{k}.json") for k in ("calibration", "link")}
    plates["payload"] = j(D / f"{UP}_payload.json")
    plates["b00"] = j(DIP / "v3/b/print/20261020-dip_link_v3b_print_pad_ell11_5_B00_85a_t2_g30.json")
    plates["b4"] = j(DIP / "v3/b/print/20261020-dip_link_v3b_print_pad_ell11_4_B00_85a_t2_g30.json")
    plates["fork"] = j(D / "print/20261020-p1s_pla_fork_and_indenters.json")
    plates["tipcap"] = j(DIP / "v3/b/20261020-dip_link_v3b_tip_caps.json")
    plates["caps"] = j(DIP / "v3/b/print/20261020-dip_link_v3b_p1s_pla_tip_caps.json")
    plates["b00p"] = j(DIP / "v3/b/print/20261020-dip_link_v3b_print_pad_ell11_5_B00_pin_85a_t2_g30.json")
    plates["rev2"] = j(DIP / "v3/b/20261021-dip_link_v3b_solid_covers_and_caps2.json")
    plates["plate2"] = j(DIP / "v3/b/print/20261021-dip_link_v3b_p1s_pla_solid_covers_caps2.json")
    plates["b00r"] = j(DIP / "v3/b/print/20261021-dip_link_v3b_print_pad_ell11_5_B00_pin_raised_85a_t2_g30.json")
    fe = {}
    for r in rows(FE / "fe.jsonl"):
        if r["case"] in ("b_v2", "b_v2_wall3", "b_v2_wall495"):
            fe.setdefault(r["case"], {})[r["probe"]] = r["per_force"]
    sens = {r["case"]: r for r in j(FE / "sensing.json")["rows"]}
    audit = j(FE / "print_audit.json")
    return ev, pv, chk, parts, step, plates, fe, sens, audit


def fe_change(fe, case, F):
    d = [fe[case][p][F]["approach_um"] / fe["b_v2"][p][F]["approach_um"] - 1 for p in fe[case] if p in fe["b_v2"]]
    return 100 * float(np.mean(d)), 100 * float(min(d)), len(d)


def main():
    ev, pv, chk, parts, step, plates, fe, sens, audit = load()
    c2, c3 = ev["v2"]["clearance"]["per_part"], ev["v3"]["clearance"]["per_part"]
    w2, w3 = ev["v2"]["walls"], ev["v3"]["walls"]
    m2, m3 = ev["v2"]["mass_g"], ev["v3"]["mass_g"]
    g3 = chk["b_links"]["v3b"]["gaps_by_region_mm"]
    hs, hr = chk["poses_table"]["sphere_homing_geometry"], chk["poses_table"]["rod_homing_geometry"]
    pc, pl, pb = plates["calibration"], plates["link"], plates["b00"]
    r4 = ev["v3"]["r4"]["ell11_5"]["groove_to_pocket_mm"]
    b_u1 = ev["v3"]["sensing"]["static_fields"]["ell11_5"]["B_U1_mT"]
    pb_t = pb["summary"]["print_time"].replace("m ", "&#8202;min ").replace("s", "&#8202;s")
    p4 = plates["b4"]
    p4_t = p4["summary"]["print_time"].replace("m ", "&#8202;min ").replace("s", "&#8202;s")
    has_w3 = "b_v2_wall3" in fe and "b_v2_wall3" in sens
    w3_6 = fe_change(fe, "b_v2_wall3", "6") if "b_v2_wall3" in fe else (0, 0, 0)
    w5_6 = fe_change(fe, "b_v2_wall495", "6") if "b_v2_wall495" in fe else (0, 0, 0)
    sb = sens["b_v2"]["gain"]["centre_force_N"]
    sw = sens["b_v2_wall3"]["gain"]["centre_force_N"] if has_w3 else None
    sw5 = sens["b_v2_wall495"]["gain"]["centre_force_N"] if "b_v2_wall495" in sens else None
    meas = audit["measured_20261020"]
    P_DY, P_DZ = -hs["move_to_touch_table"][1], -hs["move_to_touch_table"][2]
    ins_ok = all(v["insertion_clear"] and v["overlap_fork_mm3"] == 0 for v in chk["b_links"].values())
    poses_ok = all(all(x == 0 for k, x in v.items() if k != "socket_floor_minus_pin_top_mm")
                   for k, v in chk["poses_table"].items() if not k.endswith("geometry"))
    POSE_TXT = ("At neither pose does the mount, the link or the pad overlap the indenter, for either indenter." if poses_ok
                else "A pose overlaps; see the check file.")

    lede = (f"Concept b becomes v3 and the UR5e mount v2 after the owner's first prints. The cover gains side walls along the "
            f"pad's flanks, a column {pv['col_margin']:.1f}&#8202;mm inside the 11/22 outline replaces the neck, and the cable's leg "
            f"opens to the top. The walls change the pad's approach by at most {abs(w5_6[1]):.2f}&#8202;% at 6&#8202;N; the owner chose "
            f"them flush with the pad (W00). Walls around both legs of the link clock it on the mount's fork to "
            f"{chk['b_links']['v3b']['turn_to_contact_deg']['about_z_tool_+']:.1f}&#176; about the tool axis without a set screw, "
            f"and a tapered key on each indenter's post homes it. "
            + (f"The readout model resolves {sens['bm_five_d75']['gain']['centre_force_N']:.3f}&#8202;N at the contact point with "
               f"five magnets and {sens['bm_four_d75']['gain']['centre_force_N']:.3f}&#8202;N with four. "
               if "bm_four_d75" in sens and "bm_five_d75" in sens else "")
            + (f"A cap over the tip end, hung on the distal board screws with a pin into the pad, holds the pad's last "
               f"4.9&#8202;mm at a cost of 1&#8211;6&#8202;% of that resolution." if plates.get("tipcap", {}).get("variants") else ""))

    S = []
    S.append(("terms", "Terms and metrics", f"""
<dl class="terms">
<dt>F, M, table frame</dt><dd>F is the DIP study's finger frame (mm; x toward the dorsal side, y along the PIP axis, z up, the
straight finger hanging along &#8722;z, the PIP axis through (0, y, &#8722;51.55)). M is the mount's frame: origin at the centre of the
adapter's face on the HEX-E, z along the tool axis toward the tip, x toward the palmar side, y along the PIP axis. The table frame
of the homing: z up from the table, x along the tool axis toward the tip, the palmar side down.</dd>
<dt>Cover</dt><dd>The PLA plate of concept b between the board's component side and the 85A pad, with windows over the chips and
the two dovetail rails the pad snaps onto (the owner calls it the cassette).</dd>
<dt>Wall clearance</dt><dd>Gap on each side, across the finger (along y), between a cover wall's inner face and the pad's straight
flank as designed, in mm. The walls follow the pad's outline, so the gap is the same along the finger; at the walls' distal end, the
pad's widest section, they stand 14.8&#8202;mm plus twice the clearance apart. The fit covers W00, W10 and W20 have 0.00, 0.10 and
0.20&#8202;mm (14.80, 15.00 and 15.20&#8202;mm apart).</dd>
<dt>Groove fit</dt><dd>How much narrower than the cover's rail the pad's dovetail groove is on each flank, in mm: B00 has 0.00 (the
groove as wide as the rail), B05 0.05. v3 takes B00, the fit chosen on the v2 rails.</dd>
<dt>Clearance to the joint</dt><dd>Least signed distance from a part fixed to the DIP link to the MCP link and the PIP servo over
PIP &#8722;18 to 92&#176; in 0.5&#176; steps, in mm; the specification's R7 asks 0.5&#8202;mm for rigid parts and 1.0&#8202;mm for the cable.</dd>
<dt>Least wall</dt><dd>Thinnest PLA in the checked sections of a printed part, in mm; R11 asks 0.9&#8202;mm (0.8&#8202;mm for two 0.4&#8202;mm
lines).</dd>
<dt>Approach</dt><dd>How far a rigid probe moves into the pad past first touch at a given normal force, from the linear finite
elements of the 10-18 calibration (18 probes on the palmar face, 0.5&#8211;6&#8202;N), in &#181;m.</dd>
<dt>Force resolution</dt><dd>One standard deviation of the normal force estimated from one sample of the five MLX90393 chips at
the contact point, with the contact position and a uniform background field unknown (Cram&#233;r&#8211;Rao bound of the readout model,
<code>tpu_tip_magnet_sensing</code>), at the gain the field needs, in N; smaller is finer.</dd>
<dt>Magnet depth</dt><dd>Distance along the palmar normal from the pad's apex, its most palmar point, to the magnets' palmar
faces, in mm. The magnets are 2&#8202;mm thick and b's pad is 13.45&#8202;mm deep, so 11.45&#8202;mm less the depth of TPU lies behind
them, and the chips' sensing plane is 0.42&#8202;mm behind the pad.</dd>
<dt>Magnet travel</dt><dd>Mean translation of the magnets when the 12.5&#8202;mm rod presses the contact point with 1&#8202;N, from
the finite elements, in &#181;m.</dd>
<dt>Position resolution</dt><dd>One standard deviation of the contact position along y and z, estimated from one sample of the
five chips by the same bound as the force resolution, in mm.</dd>
<dt>Datum</dt><dd>A face of the mount against which a face of the DIP link is pressed, so that the link takes one pose; the gap
given is the nominal one before the link is pressed on.</dd>
<dt>Homing</dt><dd>Seating the key of a post on the indenter's base in a socket of the mount, which fixes where the pad's contact
point is relative to the indenter's top and how the finger is turned about the key's axis.</dd>
<dt>Turn to contact</dt><dd>The smallest turn of the DIP link about an axis through the bridge's middle, from its place on the fork,
at which the link and the fork overlap, with the clamp plate and the screws off, in degrees on a grid of 0.05 to 5&#176;.</dd>
<dt>Tip end</dt><dd>The plane z_F &#8722;98.46 where the DIP link's tongue, the cover and the pad end; the board stops 0.5&#8202;mm short of
it. Seen along the finger it is a T: the tongue and the cover 21 &#215; 5.1&#8202;mm, the pad's flat end 4.4 &#215; 7.5&#8202;mm.</dd>
<dt>Interference</dt><dd>How far a wall's inner face stands inside the pad's flank as designed, on each side, in mm: the negative
of the wall clearance. Covers I05, I10 and I15 have 0.05, 0.10 and 0.15&#8202;mm, so their walls squeeze the 14.8&#8202;mm pad to 14.7, 14.6
and 14.5&#8202;mm.</dd>
<dt>Seat raise</dt><dd>Height of the proximal board screws' seats above the cover's front face, in mm; the wall beneath a seat is solid
but for the screw's 2.4&#8202;mm hole. 0 on the covers W00&#8211;W20, 3.0 on I05&#8211;I15.</dd>
<dt>Tip cap TP, TW</dt><dd>A PLA part over the tip end: an end plate 1.2&#8202;mm thick on the T, two ears 3&#8202;mm thick on the cover's
front face under the two distal board screws, and a 2&#8202;mm pin 2.7&#8202;mm into the pad's flat end. TP's ears stand 0.5&#8202;mm off
the pad; TW adds side walls 6&#8202;mm tall along the pad's distal flanks, on the pad (0.00&#8202;mm, as W00).</dd>
</dl>"""))

    t1 = [["The 30&#8202;mm disc on the UR5 adapter has no bore in the HEX-E; only the 5.5&#8202;mm nub fits",
           "Spigot removed; the printed dowel (5.95&#8202;mm) kept", "Adapter 16.0&#8202;mm tall, one solid"],
          ["Palmar side aligned with the sensor's x-axis, taken from the centre toward the nub",
           "The palmar normal points at the dowel (as in v1); the foot's four M3 now sit at 22.5, 112.5, 202.5 and 280&#176;, so the "
           "standoff-fork goes on one way only", "Hex key reaches all four heads"],
          ["The rod in the horn slot is tight and snapped; clamping can leave the link crooked; an exterior mold on the outer volume",
           "Key removed; v1's ears now fit the bridge's palmar and dorsal faces 0.2&#8202;mm off, an M3 set screw through the dorsal ear "
           "seats the link on the palmar ear, the inner block's arc 0.05&#8202;mm above the bridge stops it along the tool axis",
           "Both b links slide on along the tool axis with no overlap"],
          ["Homing geometry to zero the mount to the table indenters, a slot/rod mating",
           "A 6.3&#8202;mm socket in the column's palmar face; a post with a 6.0&#8202;mm pin on each indenter's base",
           "No overlap at homing or touching, either indenter"],
          ["Mount offset so the finger is in line with the centre", "Tool axis on the finger's mid-plane (y_F 0, 3.85&#8202;mm moved)",
           f"Contact point at M ({hs['contact_M'][0]:.2f}, {hs['contact_M'][1]:.2f}, {hs['contact_M'][2]:.2f})&#8202;mm"],
          ["Concept b chosen; the double-ended Qwiic cable needs a channel to slot into",
           "The leg toward the back open to the top out through the back", "Cable clearance unchanged"],
          ["Rails pop off under a sideways push (B00; B05 worse); beef up the cassette with side volume around the pad",
           f"Two walls {pv['wall_h']:.1f}&#8202;mm tall along the pad's flanks from the mount face to the widest section, counterbored "
           "over the proximal screws, the chip windows cut out", f"Approach change {w5_6[1]:+.2f}&#8202;% at 6&#8202;N (FE)"],
          ["The block under the pad takes side and large contacts on rigid PLA; a column a little inside the 11/22, a fillet or angled "
           "edge to the board mount", f"Column {pv['col_margin']:.1f}&#8202;mm inside the 11/22 mount-face outline; 45&#176; "
           f"chamfers of {pv['col_chamfer']:.1f}&#8202;mm to the board mount", f"Link {m2['dip_link_solid']:.1f} &#8594; "
           f"{m3['dip_link_solid']:.1f}&#8202;g solid"],
          ["The rails slot well at B00; B05 snaps on more easily and pops off sooner",
           "Groove fit 0.00&#8202;mm (B00) in the v3 parameters; pad B00 with its five magnets as an XL project with three pauses",
           f"Grooves {r4:.1f}&#8202;mm from the magnet pockets (R4 asks 2.0)"],
          ["W00 works fine (the fit covers with pad B00)", "Wall clearance 0.00&#8202;mm in the v3 parameters; the link plate takes "
           "cover W00", f"Link plate {pl['time']}, no slice problems"],
          ["Re-prepare the four-magnet configuration; with five, the magnets sit far from the tip and may not move much",
           "Pad B00 with the four magnets of the 10-18 print (ID B4) as an XL project; b's pad in the readout model with four and "
           "five magnets at 6.5&#8211;8.5&#8202;mm", "Table 7, Figure 12"],
          ["Is it possible to run one wall around the magnets instead of two?", "Both pads with one perimeter along each pocket "
           "(IDs 5W1, 4W1) as XL projects; the same in the readout model", "Loops counted in the G-code; Table 7"],
          ["The homing slot and shaft should not be round: they leave the finger's turn unknown; clock it exactly",
           "A rectangular key on each post and a socket in the column, every face at 20&#176;; the key's long faces seat with no "
           "play, its ends 0.15&#8202;mm inside the socket's", "Key and socket meet with no overlap at homing; the key's top 0.8&#8202;mm "
           "short of the floor"],
          ["An exterior mold for the two legs, a perimeter volume that clocks the link; the set screw could fix the link wrongly",
           "Walls around both legs' outlines 0.15&#8202;mm off, the outline swept along the tool axis; no set screw",
           f"Turn to contact {chk['b_links']['v3b']['turn_to_contact_deg']['about_z_tool_+']:.1f}&#176; about the tool axis, "
           f"{chk['b_links']['v3b']['turn_to_contact_deg']['about_y_pip_+']:.1f}&#176; about the PIP axis (ears fork: "
           f"{min(v for k_, v in chk['b_links']['v3b']['turn_to_contact_deg_ears_fork'].items() if k_.startswith('about_z')):.2f} and "
           f"{min(v for k_, v in chk['b_links']['v3b']['turn_to_contact_deg_ears_fork'].items() if k_.startswith('about_y')):.1f}&#176;)"],
          ["A standard sparse 15&#8202;% grid infill on the mount's parts", "Bambu Studio's 0.20&#8202;mm Standard (two walls, 15&#8202;% "
           "grid) for the fork and both indenters", f"Plate {plates['fork']['time']}, {plates['fork']['filament_g']:.0f}&#8202;g"],
          ["The pad gets popped out a lot; a T-shaped plate over the tip end, fastened to the link and the pad and molded around "
           "the fingertip's top (19:48); the recommendation taken (20:36)",
           "Tip caps TP and TW (Section 9): end plate, ears under the distal board screws (M2 &#215; 8 for &#215; 5), a pin into the pad "
           "in place of a thread or insert; TW with side walls; pad B00P with the pin's hole",
           "No overlap with any part; TP's ears 0.50&#8202;mm off the pad; joint clearance over 6.7&#8202;mm; Table 8"],
          ["10-21: the cassette still lets the pad snap out; make its side walls clamp a little (zero clearance or a very marginal "
           "interference that squishes the pad in)", "Covers I05, I10, I15 with the walls 0.05, 0.10 and 0.15&#8202;mm inside the pad's "
           "flank on each side (Section 10)", "Interference " + ", ".join(f"{v['interference_with_B00R_mm3']:.1f}" for v in
                                                                          plates["rev2"]["covers"].values())
           + "&#8202;mm&#179; with pad B00R; no other overlap"],
          ["10-21: the walls' proximal end snapped off beside the screw hole when the pad was pushed off sideways; boost the bottom "
           "screws up; make the structure more solid and better connected",
           "Proximal screw seats 3&#8202;mm up on solid wall (M2 &#215; 8), no wall above them toward the mount face; walls straight up the "
           "21&#8202;mm edge with a 1&#8202;mm chamfer, ending at the U3/U5 windows; pad B00R with notches 3&#8202;mm deeper",
           "Heads 0.20&#8202;mm from B00R's notches; walls 3.1&#8211;4.1&#8202;mm thick"],
          ["10-21: beef up the tip caps, above all the one that curves around the ellipse (a tiny piece with sharp angles)",
           "TP2: 1.6&#8202;mm plate, chamfered ears, filleted pin root; TW2: solid blocks from the pad's flank to the 21&#8202;mm edge, "
           "5&#8202;mm tall, heads on top (M2 &#215; 10)", "No overlap but the designed interference; joint clearance over 6.6&#8202;mm"]]
    S.append(("changes", "Changes from the owner's notes of 2026-10-20", f"""
<p>Table 1 lists each note, the change it made and the check that covers it. The DIP link changes are design switches of the
concept b generator with their own parameters (<code>cover_walls</code>, <code>neck</code>, <code>cable_leg</code> in
{code(DIP / 'params/20261020-v3_concept_b.json')}); a parameter file without them still builds v2. The mount's changes sit behind
<code>UR5M_REV=v2</code> in {code(ROOT / 'scripts/ur5_calibration_mount.py')}, which without it still builds v1.</p>
{table(["Owner's note", "Change", "Check"], t1, caption="The notes of 2026-10-20 and what they changed.", n=1)}"""))

    films = VP.films_html([("DIP link b v3 with its cover, pad and Qwiic cable",
                            DIP / "v3/b/media/20261020-dip_link_v3b_turntable.mp4",
                            DIP / "v3/b/media/20261020-dip_link_v3b_iso.png")],
                          "Turntable of DIP link b v3 with its cover, the 11/22 pad and the Qwiic cable in its groove.", 4)
    t2 = [["Clearance of the new body to the joint (mm)", f"{c2['dip_link_new_body']['min_mm']:.2f}",
           f"{c3['dip_link_new_body']['min_mm']:.2f}"],
          ["Clearance of the Qwiic cable (mm)", f"{c2['qwiic_cable']['min_mm']:.2f}", f"{c3['qwiic_cable']['min_mm']:.2f}"],
          ["Least wall beside J1 (mm)", f"{w2['neck_z-72.30_around_J1']['min_wall_mm']:.2f}",
           f"{w3['neck_z-72.30_around_J1']['min_wall_mm']:.2f}"],
          ["Least wall of the cover (mm)", f"{w2['cover_x3.35']['min_wall_mm']:.2f}", f"{w3['cover_x3.35']['min_wall_mm']:.2f}"],
          ["DIP link, solid PLA (g)", f"{m2['dip_link_solid']:.2f}", f"{m3['dip_link_solid']:.2f}"],
          ["Cover, solid PLA (g)", f"{m2['cover_solid']:.2f}", f"{m3['cover_solid']:.2f}"]]
    S.append(("dip", "DIP link b v3", f"""
<p>Figure 1 shows the neck. In v2 a block as wide as the cassette (21&#8202;mm) and as deep as the tip filled the 9.75&#8202;mm between the
pad's mount face and the bridge, its palmar face level with the pad's apex. In v3 the part palmar of the tongue's front face
(x_F 3.95) is a column of the 11/22 pad's mount-face section shrunk by {pv['col_margin']:.1f}&#8202;mm: 11.7&#8202;mm wide and reaching
x_F &#8722;8.84, 1.71&#8202;mm behind the apex, so a flat object pressed on the palmar side compresses the pad that far before it meets the
column, and every other tip, being larger, covers it too. Dorsal of x_F 3.95 the board mount stays as wide as the cassette up to the
bridge, where it takes the board's tab, J1, the plug and the floor of the cable's groove; {pv['col_chamfer']:.1f}&#8202;mm chamfers at
45&#176; join it to the column. Beside J1 the column leaves {w3['neck_z-72.30_around_J1']['min_wall_mm']:.2f}&#8202;mm of PLA (Table 2). The bridge and the clevis arms, which take the servo's horn and the joint's shaft, are as before.</p>
{fig(1, DIP / 'v3/b/media/20261020-dip_link_b_v2_v3_palmar.png', 'Concept b from the palmar side, finger up: v2 as printed and v3.')}
<p>Figure 2 shows the cover. Its web stays as thick as the chips (0.9&#8202;mm), because the pad's back face rests on the chip tops;
the new volume is two walls {pv['wall_h']:.1f}&#8202;mm tall along the pad's flanks, where the flanks run straight back from the
pad's equator. They run from the mount face toward the pad's widest section, the owner's &#8216;from the screw holes at the bottom until
the ellipse starts to curve back in&#8217;, and have the windows of the two outer chips and their capacitors cut out of them, {pv['wall_window_gap']:.1f}&#8202;mm
around (owner, 15:20 and 15:50: the first draft reached over them); at {pv['wall_h']:.1f}&#8202;mm they cover the flank's whole straight
part and 1.05&#8202;mm past the equator, where it has curved in by 0.06&#8202;mm (owner, 15:50: 'pop up the retaining lip a lot more');
they stand {pv['wall_clear']:.2f}&#8202;mm off the flank, and slope outward to the cover's edge
({pv['wall_top_t']:.1f}&#8202;mm thick at the top); each proximal screw head sits in a counterbore through its wall, and the walls end at
45&#176;. A sideways push on the pad then bears on the wall instead of prying the dovetails apart. The walls also stiffen the 0.9&#8202;mm web
along both long edges, which is where the v2 cover lifted off the bed and curled.</p>
{fig(2, DIP / 'v3/b/media/20261020-dip_link_b_v2_v3_cover.png', 'The cover of v2 and of v3, and v3 with the pad (see-through).')}
<p>Figure 3 shows the cable's first turn from the PIP side. In v2 the leg from the plug toward the back was open to the top only up
to x_F 6.0&#8202;mm and ran under 3.45&#8202;mm of bridge from there to the back face: the bar in the owner's photograph, through which the
cable's far plug cannot pass. In v3 the leg is open to the top out through the back, so the cable comes up the J1 slot, lies into
the leg from the top and presses into the groove on the back from behind. Its path, and so its 1.24&#8202;mm clearance to the MCP link,
are unchanged (Table 2); a strip of Kapton tape over the leg keeps it down.</p>
{fig(3, DIP / 'v3/b/media/20261020-dip_link_b_v2_v3_cable_leg.png', 'The leg toward the back, seen from the PIP side: v2 roofed, v3 open.')}
{films}
{table(["Check", "v2", "v3"], t2, num_cols=(1, 2), caption="Checks of v2 and v3 (evaluation.json of each, scripts/dip_link_v2.py b). "
       "The clevis arms kept from the CAD hand's link reach the MCP link by 0.18&#8202;mm at PIP &#8722;3.5&#176; in both, as on the hand "
       "today.", n=2)}"""))

    fe_rows = []
    for c, lab in (("b_v2", "No walls (v2 cover)"), ("b_v2_wall3", "Walls 3.0&#8202;mm (first draft of the v3 cover)"),
                   ("b_v2_wall495", f"Walls over the whole straight flank, 4.95&#8202;mm (v3 cover, {pv['wall_h']:.1f}&#8202;mm)")):
        if c not in fe:
            continue
        mean6, low6, n_ = fe_change(fe, c, "6") if c != "b_v2" else (0.0, 0.0, len(fe[c]))
        rod = fe[c]["cx12.5_z0"]["1"]["approach_um"]
        s = sens.get(c, {}).get("gain", {})
        fe_rows.append([lab, f"{rod:.1f}", "&#8211;" if c == "b_v2" else f"{mean6:+.2f} / {low6:+.2f}",
                        f"{s['centre_force_N']:.4f}" if s else "&#8211;",
                        f"{s['centre_sig_y_mm']:.2f} &#215; {s['centre_sig_z_mm']:.2f}" if s else "&#8211;"])
    S.append(("walls", "Side walls and the pad's stiffness and readout", f"""
<p>The walls touch the pad, so they could change the stiffness the 10-18 calibration fitted and the motion the magnetometers read.
In the finite elements of b's pad (the 11/22 tip cut at the cover, five magnets, the model of the night of 10-19) the walls fix the
flank's nodes in y over their height and length up to the apex's z, both ways; the printed wall stands 0.10&#8202;mm off, resists
only outward motion and has the chip windows cut out of it, so the change is an upper bound; the v3 walls
({pv['wall_h']:.1f}&#8202;mm) cover the whole straight flank, the last row of Table 3. With {pv['wall_h']:.1f}&#8202;mm walls the approach falls by {abs(w3_6[0]):.2f}&#8202;% on average at 6&#8202;N
(at most {abs(w3_6[1]):.2f}&#8202;% over {w3_6[2]} probes), and with walls over the whole straight flank by {abs(w5_6[1]):.2f}&#8202;% at most
(Table 3), against the calibration's 5&#8202;% bound."""
              + (f" Over the whole straight flank the magnets move {sens['b_v2_wall495']['magnet_travel_um_per_N']:.2f} in place of "
                 f"{sens['b_v2']['magnet_travel_um_per_N']:.2f}&#8202;&#181;m per newton, so the force resolution at the contact point "
                 f"goes from {sb:.4f} to {sw5:.4f}&#8202;N, {100 * (sw5 / sb - 1):.0f}&#8202;% coarser, and the contact position stays "
                 f"within {sens['b_v2_wall495']['gain']['centre_sig_y_mm']:.2f} by {sens['b_v2_wall495']['gain']['centre_sig_z_mm']:.2f}"
                 "&#8202;mm; concept a's tip gives 0.065&#8202;N." if sw else " The readout rows are pending.") + f"""</p>
{table(["Pad", "Rod approach at 1&#8202;N (&#181;m)", "Approach change at 6&#8202;N, mean / largest (%)", "Force resolution (N)",
        "Contact position, 1 sd (mm)"], fe_rows, num_cols=(1, 2, 3, 4),
       caption="b's pad with and without the walls' hold (scripts/v2_tip_fe_checks.py cases b_v2, b_v2_wall3, b_v2_wall495; "
               "fe.jsonl, sensing.json). The rod is the 12.5&#8202;mm one at the contact point.", n=3)}"""))

    gl = chk["b_links"]
    t4 = [[lab, f"{gl['v2b']['gaps_by_region_mm'][k]:.2f}", f"{gl['v3b']['gaps_by_region_mm'][k]:.2f}"]
          for k, lab in (("horn_leg_outline_to_walls", "Horn leg's outline to its walls"),
                         ("slotted_leg_outline_to_walls", "Slotted leg's outline to its walls"),
                         ("bridge_palmar_face_to_palmar_ear", "Bridge's palmar face to the fork"),
                         ("bridge_back_to_ear", "Bridge's back to the dorsal ear"),
                         ("horn_arm_inner_face", "Horn-side arm to the inner block (closed by the clamp plate)"),
                         ("slotted_arm_inner_face", "Slotted arm to the inner block (M3 and washer close it)"))]
    t4.append(["Bridge's lowest point to the inner block's arc (the stop along the tool axis)", "0.05", "0.05"])
    tt, te = gl["v3b"]["turn_to_contact_deg"], gl["v3b"]["turn_to_contact_deg_ears_fork"]
    tmin = lambda d, a: min(v for k_, v in d.items() if k_.startswith(a))          # noqa: E731
    S.append(("mount", "UR5e calibration mount v2", f"""
<p>Mount v2 keeps v1's standoff-fork and its assembly (Figure 5): the DIP link slides on along the tool axis with its clevis arms
either side of the inner block, the clamp plate presses the horn-side arm onto the inner block with two M3 into the ears, and the M3
with its washer goes through the slotted arm's bore into the inner block. The key in the horn slot is gone, and walls around both
legs locate the link in its place (owner, 18:55). In each layer of each leg (the horn leg's outer, slot and inner layers and the
slotted leg) the fork fills the region outside the leg's outline, 0.15&#8202;mm off it, the outline swept along the tool axis so the
link slides in past the walls; the horn slot is filled before, so nothing enters it (Figures 6 to 8). The walls stop 0.2&#8202;mm short
of each leg's outer face, so the clamp plate and the slotted arm's screw press the legs onto the inner block, and the set screw is
gone. With the clamp plate and the screws off, a turn of {tmin(tt, 'about_z'):.1f}&#176; about the tool axis,
{tmin(tt, 'about_y'):.1f}&#176; about the PIP axis or {tmin(tt, 'about_x'):.1f}&#176; about the palmar normal brings the link onto the
fork, against {tmin(te, 'about_z'):.2f}, {tmin(te, 'about_y'):.1f} and {tmin(te, 'about_x'):.1f}&#176; on the ears fork of 15:40. The inner
block's arc stands 0.05&#8202;mm above b's bridge, the stop along the tool axis, so this fork takes concept b's links only (v2 as printed
and v3; the CAD hand's link and a have higher bridges). {'Both b links slide on with no overlap at any of 121 positions over the last 30&#8202;mm.' if ins_ok else
 'An insertion check failed; see the check file.'} Table 4 gives the gaps.</p>
<p>The adapter has only the printed dowel toward the HEX-E, and the palmar normal points at it, the sensor's x-axis as the owner
takes it from the centre toward the nub. v1 had the same orientation, but its foot's four M3 were 90&#176; apart, so the
standoff-fork could go on in four orientations; at 22.5, 112.5, 202.5 and 280&#176; it goes on in one. The tool axis runs through the
finger's mid-plane (y_F 0), 3.85&#8202;mm from v1's, so the contact point lies on the axis in y and 10.55&#8202;mm from it along the
palmar normal (Figure 9).</p>
{fig(5, D / f'media/{UP}_exploded.png', 'Mount v2 exploded along the tool axis; the DIP link slides on as in v1.')}
{fig(6, D / f'media/{UP}_fork_closeup.png', 'DIP link b on the fork, horn and dorsal side, with the clamp plate.')}
{fig(7, D / f'media/{UP}_fork_slotted_side.png', 'DIP link b on the fork, slotted side: the leg in its walls.')}
{fig(8, D / f'media/{UP}_fork_alone.png', 'The standoff-fork alone, its fork end up.')}
{fig(9, D / f'media/{UP}_on_the_hex_e_palmar.png', 'Mount v2 on the HEX-E (context) with DIP link b v3, palmar side, finger down.')}
{table(["Gap", "v2 b as printed (mm)", "v3 b (mm)"], t4, num_cols=(1, 2),
       caption=f"Gaps of the two b links on the fork ({UP}_check.json; nearest surface samples 0.1&#8202;mm apart; the arc's gap by "
               "construction).", n=4)}"""))

    t5 = [[k, f"({h['post_xy'][0]:.2f}, {h['post_xy'][1]:.2f})", f"{h['mouth_z']:.2f}", f"{h['pin_top_z']:.2f}",
           f"{h['indenter_top_z']:.1f}"] for k, h in (("Sphere", hs), ("Rod", hr))]
    S.append(("homing", "Homing to the indenters", f"""
<p>Each indenter's base gains a post with a key (owner, 18:55: the round pin left the finger's turn about it unknown): a block
and on it a frustum {hs['key_at_mouth_mm'][0]:.0f} by {hs['key_at_mouth_mm'][1]:.1f}&#8202;mm at the socket's mouth, every face drafted
{hs['draft_deg']:.0f}&#176;, its long side along the tool axis. The socket in the column's palmar face has the same faces,
{hs['socket_mouth_mm'][0]:.0f} by {hs['socket_mouth_mm'][1]:.0f}&#8202;mm at the mouth and 5.3&#8202;mm deep, centred
{chk['poses_table']['sphere_homing_geometry']['socket_mouth_M'][2]:.0f}&#8202;mm from the flange. Pushed in, the key's long faces meet the
socket's at the mouth's plane: no play across the finger and none in the turn about the key's axis. Its ends stand 0.15&#8202;mm inside
the socket's, so the long faces always seat first and the position along the tool axis stays within &#177;0.15&#8202;mm; its top stays
0.8&#8202;mm short of the floor. With the palmar side down and the tool axis along the base's +x, the key seats when the pad's contact
point stands 30&#8202;mm to +y of the indenter's top and 5&#8202;mm above it (Figure 10); from there the robot moves
(0, &#8722;{P_DY:.0f}, &#8722;{P_DZ:.0f})&#8202;mm in the table frame to touch. {POSE_TXT} Table 5 gives the posts. Bolt the base with
the post toward the robot.</p>
{fig(10, D / f'media/{UP}_homing.png', 'Homing over the rod indenter: the key in the socket, the pad 30 mm to the side and 5 mm up.')}
<p>The cradle of the first draft of 2026-10-20, which took the tongue's back as its datum and let the link in from the palmar
side, stays in the script as <code>fork_style = "cradle"</code>, and the ears fork of 15:40 with its set screw as "ears"; the plates
print the fork with the walls.</p>
{table(["Indenter", "Post in the base frame, x, y (mm)", "Socket mouth above the table at homing (mm)", "Key top (mm)",
        "Indenter top (mm)"], t5, num_cols=(2, 3, 4), caption="Homing posts (homing() in scripts/ur5_calibration_mount.py).", n=5)}"""))

    asm = ["Adapter on the HEX-E: dowel into its hole, four M6 &#215; 12 with DIN 125 washers, 1.5&#8202;N&#8202;m.",
           "Standoff-fork on the adapter: four M3 &#215; 12 through the foot; it fits one way, the palmar ear toward the dowel.",
           "Qwiic cable into J1; pass its far plug up the J1 slot from the neck's underside; slide the board along the tongue's seat, "
           "J1 first.",
           "Lay the cable into the open leg from the top, press it into the groove on the back, Kapton over the leg.",
           "Cover over the board: four M2 &#215; 5 into the tongue (1.9&#8202;mm holes); the two proximal heads through the walls' "
           "counterbores.",
           "Pad onto the rails between the walls (B00).",
           "Slide the link onto the fork along the tool axis, each leg into its walls and the arms either side of the inner block, "
           "until the bridge meets the inner block; the cable into the slot on the dorsal side and up the column's groove.",
           "Clamp plate on the horn-side arm with two M3 &#215; 16 into the ears; M3 &#215; 12 with the DIN 9021 washer through the "
           "slotted arm's bore; all snug. No set screw: the walls locate the link.",
           "Homing post toward the robot; push the socket onto the key in freedrive until it stops, record the pose, then move "
           "(0, &#8722;30, &#8722;5)&#8202;mm.",
           f"UR5e payload {plates['payload']['total_g'] / 1000:.3f}&#8202;kg (the HEX-E's 0.347&#8202;kg included) with its centre of gravity "
           f"{plates['payload']['cog_from_ur_flange_mm'][2]:.1f}&#8202;mm out from the flange and "
           f"{plates['payload']['cog_from_ur_flange_mm'][1]:.1f}&#8202;mm along the PIP axis; TCP at the contact point, "
           f"{plates['payload']['tcp_contact_from_ur_flange_mm'][2]:.1f}&#8202;mm out and 10.55&#8202;mm toward the dowel (M axes; "
           f"{UP}_payload.json)."]
    S.append(("assembly", "Assembly", "<ol>" + "".join(f"<li>{a}</li>" for a in asm) + "</ol>"))

    t6 = [[esc(p["project"].split("/")[-1]), p["time"], f"{p['filament_g']:.1f}" if p["filament_g"] < 10 else f"{p['filament_g']:.0f}",
           ", ".join(o["label"] for o in p["objects"]),
           "none" if not p["problems"] else esc("; ".join(p["problems"]))] for p in (pc, plates["fork"], pl, plates["caps"],
                                                                                    plates["plate2"])]
    S.append(("print", "Print and CAD files", f"""
<p>Five Bambu Lab P1S plates in PLA (Table 6), the parts' settings written into each project so Bambu Studio keeps them when it
slices again, and a 5&#8202;mm brim on every part. The calibration plate of 15:40 (printed: the adapter and the clamp plate on it are
unchanged) has the mount and the indenters at 0.15&#8202;mm with five walls and 40&#8202;% gyroid and the covers solid at 0.1&#8202;mm. The
fork with the walls and both indenters with the keyed posts print on their own plate at Bambu Studio's 0.20&#8202;mm Standard, two
walls and 15&#8202;% grid (owner, 19:10: "a standard sparse 15% grid infill"), in {plates['fork']['time']}; the DIP link keeps
0.1&#8202;mm on tree supports and its cover prints solid, both with walls under 1&#8202;mm. The fit covers W00, W10 and W20 have
one, two and three notches in their distal edge; they fit the v2 link already printed, so the walls can be tried on it before v3
prints. W00 held pad B00 in the owner's test, so the design and the link plate take it. The tip caps TP and TW print solid at
0.1&#8202;mm on their own plate, the end plate on the bed (Section 9). The 85A pad is B00 with its five 3&#8202;&#215;&#8202;2&#8202;mm magnets, its ID engraved in the mount face: it prints on the XL's
tool 2 in {pb_t} with pauses at {', '.join(f"{q['print_z']:.1f}" for q in pb['pauses'])}&#8202;mm, the settings of the owner's good
print of 2026-10-18 ({code(DIP / pb['project'].split('board_mount/')[1])}). The pad with the four magnets of the 10-18 print
(ID B4) prints in {p4_t} with pauses at {', '.join(f"{q['print_z']:.1f}" for q in p4['pauses'])}&#8202;mm
({code(DIP / p4['project'].split('board_mount/')[1])}); its pockets lie {p4['r4']['notch_to_pocket_mm']:.2f}&#8202;mm from the
screw-head notches (R4 asks 2.0). Both pads also print with one perimeter along each pocket, a modifier over the pocket layers
that keeps two at the outer surface (IDs 5W1 and 4W1, files <code>*_pocketwalls1_*</code>; the G-code shows one loop around every
pocket). STEP files for Inventor: the mount's parts as exact solids in M ({code(D / 'cad')}, read back
valid), and v3's new body, cover and cable-groove cutter as exact solids in F ({code(DIP / 'v3/b/cad')}).</p>
{table(["Project", "Time", "PLA (g)", "Parts", "Slice checks failed"], t6, num_cols=(1, 2),
       caption="P1S plates (scripts/v3b_ur5v2_pla_plate.py), print/20261020-p1s_pla_*.json; the tip caps' plate in v3/b/print.", n=6)}
{fig(11, D / 'print/20261020-p1s_pla_fork_and_indenters_map.png', 'The plate of 19:10: the fork with the walls and both indenters with their keyed posts.', 900)}"""))

    MAG = [("Five", "bm_five_d65", 6.5), ("Five", "bm_five_d70", 7.0), ("Five", "bm_five_d75", 7.5), ("Five", "bm_five_d85", 8.5),
           ("Four", "bm_four_d70", 7.0), ("Four", "bm_four_d75", 7.5),
           ("Five, one wall", "bm_five_d75_w1", 7.5), ("Four, one wall", "bm_four_d75_w1", 7.5)]
    BASE = {"bm_five_d65": "b_v2_d65", "bm_five_d70": "b_v2_d70", "bm_five_d75": "b_v2", "bm_five_d85": "b_v2_d85",
            "bm_four_d70": "b_v2_four_d70", "bm_four_d75": "b_v2_four"}
    if all(c in sens for _, c, _ in MAG):
        sm = {c: sens[c] for _, c, _ in MAG}
        rf = {c: r["gain"]["centre_force_N"] for c, r in sm.items() if r["gain_sel"] is not None}
        tr = {c: r["magnet_travel_um_per_N"] for c, r in sm.items()}
        ap1 = {c: r["approach_1N_um"] for c, r in sm.items()}
        gdev = max(100 * abs(sens[BASE[c]]["gain"]["centre_force_N"] / v - 1) for c, v in rf.items() if c in BASE)
        t7 = [[lab, f"{dp:.1f}", f"{11.45 - dp:.2f}", f"{tr[c]:.2f}", f"{sm[c]['B_max_component_mT']:.1f}",
               "none reads it" if sm[c]["gain_sel"] is None else f"{sm[c]['gain_sel']}, {sm[c]['res']}",
               f"{rf[c]:.3f}" if c in rf else "&#8212;",
               f"{sm[c]['gain']['centre_sig_y_mm']:.2f} / {sm[c]['gain']['centre_sig_z_mm']:.2f}" if c in rf else "&#8212;"]
              for lab, c, dp in MAG]
        S.append(("magnets", "Four and five magnets in b's pad", f"""
<p>With the magnets 7.5&#8202;mm under the apex, the readout model resolves {rf['bm_five_d75']:.3f}&#8202;N at the contact point with five
magnets and {rf['bm_four_d75']:.3f}&#8202;N with four (Table 7, Figure 12). The fifth magnet lies over U1, the chip under the contact
point, while the four lie between the chips. The four sit no shallower: on the 11/22 at the 2.9&#8202;mm seat both layouts need
7.5&#8202;mm to keep 1.8&#8202;mm of TPU between the distal slots and the surface, and the four-magnet tip printed on 10-18 has them there
too.</p>
<p>The magnets move little because the pad's back rests on the cover: at 1&#8202;N the five travel {tr['bm_five_d75']:.1f}&#8202;&#181;m, against
9.4&#8202;&#181;m in a's whole tip. The chips' sensing plane lies 4.4&#8202;mm behind them instead of 7.9&#8202;mm, so the same travel changes the
field at the chips more, and b resolves finer than a (0.028 against 0.065&#8202;N on the base grid). Moving the five toward the surface
trades the two: from 7.5 to 6.5&#8202;mm their travel grows from {tr['bm_five_d75']:.1f} to {tr['bm_five_d65']:.1f}&#8202;&#181;m and the force
resolution goes from {rf['bm_five_d75']:.3f} to {rf['bm_five_d65']:.3f}&#8202;N; the four's is {rf['bm_four_d70']:.3f}&#8202;N at 7.0&#8202;mm. At
8.5&#8202;mm the field at a chip reaches {sm['bm_five_d85']['B_max_component_mT']:.0f}&#8202;mT, more than any gain setting reads with the
model's 25&#8202;% margin.</p>
<p>One perimeter along each pocket in place of two (<code>metric_magnet_prints.onewall_modifier</code>, two loops kept at the outer
surface) turns the inner 0.45&#8202;mm of solid TPU around each magnet into gyroid. The five then travel {tr['bm_five_d75_w1']:.1f}&#8202;&#181;m and
the four {tr['bm_four_d75_w1']:.1f}&#8202;&#181;m at 1&#8202;N, and the pad's approach at 1&#8202;N grows by
{100 * (ap1['bm_five_d75_w1'] / ap1['bm_five_d75'] - 1):.1f}&#8202;%. The five's force resolution stays at {rf['bm_five_d75_w1']:.3f}&#8202;N;
the four's improves to {rf['bm_four_d75_w1']:.3f}&#8202;N, {rf['bm_four_d75_w1'] / rf['bm_five_d75']:.1f} times the five's.</p>
{table(["Magnets", "Magnet depth (mm)", "TPU behind the magnets (mm)", "Magnet travel at 1 N (&#181;m)", "Largest field at a chip (mT)",
        "GAIN_SEL, RES", "Force resolution at the contact point (N)", "Position resolution at the contact point, y / z (mm)"], t7,
       num_cols=(1, 2, 3, 4, 6), caption="b's pad in the readout model (scripts/v2_tip_fe_checks.py --sense bm_*, sensing.json); "
       "two perimeters along each pocket unless one is named; no cover walls.", n=7)}
{fig(12, DIP / 'v3/b/media/20261020-dip_link_v3b_magnet_layouts.png', "The two layouts over the board's chips, palmar view, and "
     "the readout model against the magnets' depth; open markers: one wall around the pockets.", 900)}
<p>These cases run on a grid with 0.5&#8202;mm elements through the pad's depth and 0.3&#8202;mm along the finger from 3.5&#8202;mm. On the
base grid of Table 3 the magnets at 7.5&#8202;mm lie in 1.1&#8202;mm elements, and the same cases' force resolution differs by up to
{gdev:.0f}&#8202;%.</p>"""))

    tc, pcap, pbp = plates["tipcap"], plates["caps"], plates["b00p"]
    CAPS = [("No walls (v2's cover)", "bm_five_d75"), ("Cover walls, as printed", "bm_five_d75_walls"),
            ("Walls, cap's end plate and pin", "bm_five_d75_capend"), ("Walls, cap with side walls", "bm_five_d75_cap")]
    if all(c in sens for _, c in CAPS) and tc.get("variants"):
        def at(r, y, dz):
            return next(f for yy, zz, f in r["gain"]["points_y_dz_force_N"] if yy == y and zz == dz)
        ref = sens["bm_five_d75_walls"]
        t8 = [[lab, f"{sens[c]['magnet_travel_um_per_N']:.2f}", f"{sens[c]['approach_1N_um']:.1f}",
               f"{sens[c]['gain']['centre_force_N']:.4f}", f"{at(sens[c], 0.0, 4.0):.4f}",
               f"{100 * (sens[c]['gain']['centre_force_N'] / ref['gain']['centre_force_N'] - 1):+.1f} / "
               f"{100 * (at(sens[c], 0.0, 4.0) / at(ref, 0.0, 4.0) - 1):+.1f}"] for lab, c in CAPS]
        vp, vw = tc["variants"]["TP"], tc["variants"]["TW"]
        fl = tc["finger_length"]
        pbp_t = pbp["summary"]["print_time"].replace("m ", "&#8202;min ").replace("s", "&#8202;s")
        S.append(("caps", "Caps over the tip end", f"""
<p>The owner found the pad popping out of its rails (19:48) and proposed a T-shaped plate over the tip end, fastened to the link and
to the pad and molded around the fingertip's top. Nothing holds the pad's last 4.9&#8202;mm: the rails stop at z_F &#8722;93.55 and the cover
walls at the apex's z, and between them the pad rests on dovetails with a 0.3&#8202;mm undercut per side. Two caps of that idea are built
(Figure 13). Each has an end plate on the T of the tip end, so the pad sits between the column at its mount face and the plate.
Across the finger a 2&#8202;mm pin 2.7&#8202;mm into the pad's flat end holds it, 1.5&#8202;mm in front of the pad's back face and
{pbp['pin_hole']['to_pocket_mm']:.2f}&#8202;mm from the nearest magnet pocket. No thread goes into the TPU: an M2 heat-set insert, 3.2&#8202;mm
across, would leave 0.6&#8202;mm of TPU at the 4.4&#8202;mm end face. The caps hang from the two distal board screws, which stand 2.65&#8202;mm
from the tip end outside the pad; M2 &#215; 8 in place of M2 &#215; 5 pass a 3&#8202;mm ear and end where they end now, so the link takes no
new hole (an axial M2 in the 3.6&#8202;mm tongue would leave 0.85&#8202;mm of PLA beside the board's seat). TP's ears stand
{vp['gaps_mm']['sides_to_pad']:.2f}&#8202;mm off the pad, measured normal to its outline. TW's side walls run along the pad's distal
flanks from 0.2&#8202;mm past the U3/U5 windows (z_F &#8722;{abs(tc['wall_z0_F']):.2f}) to the ears; beside the screw heads the 3&#8202;mm ears take
their place.</p>
<p>In the finite elements (Table 8), against v3 as printed, the end plate and the pin coarsen the force resolution at the contact
point by {100 * (sens['bm_five_d75_capend']['gain']['centre_force_N'] / ref['gain']['centre_force_N'] - 1):.0f}&#8202;% and TW by
{100 * (sens['bm_five_d75_cap']['gain']['centre_force_N'] / ref['gain']['centre_force_N'] - 1):.0f}&#8202;%; 4&#8202;mm toward the tip the
changes are {100 * (at(sens['bm_five_d75_capend'], 0.0, 4.0) / at(ref, 0.0, 4.0) - 1):.0f} and
{100 * (at(sens['bm_five_d75_cap'], 0.0, 4.0) / at(ref, 0.0, 4.0) - 1):.0f}&#8202;%. The magnets' travel falls more than the resolution,
because the plate stops their slide toward the tip; the centre magnet over U1 keeps 96 and 92&#8202;% of its motion along the palmar
normal. The model bonds the plate and the pin to the pad, its pin is 3&#8202;mm long and the side walls hold the flank both ways,
so the table bounds the caps' cost.</p>
<p>Neither cap overlaps the pad with its hole, the cover, the link, the board, the magnets or the screws; the distal screw heads
stand {tc['distal_screws_to_pad_mm']:.2f}&#8202;mm from the pad. Over PIP &#8722;18 to 92&#176; the caps stay more than
{vp['pip_sweep']['min_mm']:.1f}&#8202;mm from the MCP link and the PIP servo, and they are {vp['width_mm']:.1f}&#8202;mm wide, the link's width.
TP weighs {vp['mass_solid_g']:.2f}&#8202;g and TW {vw['mass_solid_g']:.2f}&#8202;g solid. The plate lengthens the finger from
{fl['pip_to_tip_mm']:.2f} to {fl['pip_to_tip_with_cap_mm']:.2f}&#8202;mm, PIP axis to tip, which the hand model takes if a cap stays.
Both print on one P1S plate in {pcap['time']} ({pcap['filament_g']:.1f}&#8202;g, solid at 0.1&#8202;mm, end plate on the bed); pad B00P,
B00 with the pin's hole, prints on the XL in {pbp_t} with the same three pauses. STEP:
{code(DIP / 'v3/b/cad')} (<code>*_tip_cap_TP.step</code>, <code>*_tip_cap_TW.step</code>, single valid solids).</p>
<ol>
<li>Take out the two distal board screws; the proximal pair keeps the board in place.</li>
<li>With the pad on its rails and against the column, slide the cap onto the tip along the finger: the walls (TW) and ears pass
the pad's flanks, the pin enters the pad's hole. For the B00 already printed, melt a 2&#8202;mm hole 3&#8202;mm deep with a hot rod
at the pin's place, or snip the pin off.</li>
<li>Press the plate onto the pad's end and drive the two M2 &#215; 8 (brass or A2 stainless) through the ears; their 2.4&#8202;mm holes take
up &#177;0.2&#8202;mm of the pad's printed length.</li>
</ol>
{fig(13, DIP / 'v3/b/media/20261020-dip_link_v3b_tip_caps.png', "Caps TP and TW on b v3's tip end, TW pulled off along the finger "
     "with the distal screws out, and TW seen from beyond the tip.", 1000)}
{table(["Pad held by", "Magnet travel at 1 N (&#181;m)", "Approach at 1 N (&#181;m)", "Force resolution at the contact point (N)",
        "Force resolution 4 mm toward the tip (N)", "Change from the walls as printed, contact point / 4 mm toward the tip (%)"], t8,
       num_cols=(1, 2, 3, 4),
       caption="b's pad with five magnets 7.5 mm deep in the readout model (scripts/v2_tip_fe_checks.py --sense, sensing.json); "
       "the walls and the caps held bonded to the pad.", n=8)}"""))

    r2, pl2, pbr = plates["rev2"], plates["plate2"], plates["b00r"]
    if r2.get("covers") and r2.get("caps"):
        cv, cp, rh = r2["covers"], r2["caps"], r2["raised_heads"]
        pbr_t = pbr["summary"]["print_time"].replace("m ", "&#8202;min ").replace("s", "&#8202;s")
        t9 = [[k, f"{-v['wall_clear_mm']:.2f}", f"{v['interference_with_B00R_mm3']:.1f}", f"{v['mass_solid_g']:.2f}",
               f"{v['notches']}"] for k, v in cv.items()]
        t10 = [[k, cp[k]["label"].split(": ")[1], f"{cp[k]['side_height_mm']:.1f}", cp[k]["distal_screws"],
                f"{max(cp[k]['overlap_mm3']['pad_B00R'], 0.0):.1f}", f"{cp[k]['gaps_mm']['sides_to_B00R']:.2f}",
                f"{cp[k]['mass_solid_g']:.2f}"] for k in ("TP2", "TW2")]
        S.append(("rev2", "Covers and caps after the 10-21 breakage", f"""
<p>On 10-21 the pad, pushed sideways, came off cover W00 and broke the proximal end of a wall. Between each proximal screw's
counterbore and the mount face the wall was a fin 0.66&#8202;mm thick: the counterbore, 4.2&#8202;mm across for the screw head at
y&#8202;&#177;8.0, cut through the wall's whole height and through its inner face, because the head stands 0.78&#8202;mm inside the pad's flank.
Past the U3/U5 windows the walls ended in stubs 1.5&#8202;mm off the pad. The covers I05, I10 and I15 change three things (Figure 14).
Their walls clamp the pad: the inner faces stand 0.05, 0.10 and 0.15&#8202;mm inside the flank on each side (Table 9). The proximal screws'
heads sit {rh['seat_above_cover_mm']:.1f}&#8202;mm up, on wall that is solid beneath them but for the 2.4&#8202;mm hole, and the wall above the
seats is gone from the counterbore's distal edge to the mount face, which leaves no fin or skin beside the heads. The walls rise straight up the cover's
21&#8202;mm edge to a 1&#8202;mm chamfer, 3.1&#8211;4.1&#8202;mm thick where the old ones tapered to 1.2&#8202;mm, and end at the U3/U5 windows; the
capacitors beside U3 and U5, 0.5&#8202;mm tall, keep their windows in the web alone. The pad's proximal notches are
{pbr['proximal_notch_depth_mm']:.1f}&#8202;mm deep in place of 1.5, so the raised heads stand {rh['to_B00R_mm']:.2f}&#8202;mm from pad B00R; in a pad
cut for the old covers they would overlap {rh['overlap_B00_notch_1p5_mm3']:.1f}&#8202;mm&#179; of TPU at the two proximal corners.</p>
<p>TW's walls were 1.2&#8202;mm thick and stepped down to the ears. TW2 replaces them with solid blocks from the pad's flank, at
0.05&#8202;mm interference, to the cover's edge, 5&#8202;mm tall over the pad's whole straight flank, with the distal screws' heads on
top (M2 &#215; 10). TP2 keeps TP's ears 0.5&#8202;mm off the pad. Both have a 1.6&#8202;mm end plate, 0.5&#8202;mm chamfers along the sides'
outer and proximal top edges and a 0.4&#8202;mm fillet at the pin's root (Table 10). Pad B00R's hole has a 0.5&#8202;mm chamfer at its mouth
for the fillet and stays {pbr['pin_hole']['to_pocket_mm']:.2f}&#8202;mm from the nearest pocket. The plate lengthens the finger to
{r2['finger_length']['pip_to_tip_with_cap_mm']:.2f}&#8202;mm, PIP axis to tip. No cover or cap overlaps the link, the board, the
magnets or the screws; over PIP &#8722;18 to 92&#176; they stay more than {min(cv['I05']['pip_sweep']['min_mm'], cp['TP2']['pip_sweep']['min_mm'], cp['TW2']['pip_sweep']['min_mm']):.1f}&#8202;mm
from the MCP link and the PIP servo, and every part is 21.0&#8202;mm wide. STEP: <code>v3/b/cad/20261021-dip_link_v3b_cover_I*.step</code>,
<code>*_tip_cap_TP2.step</code>, <code>*_tip_cap_TW2.step</code> (single valid solids).</p>
<p>The three covers and both caps print on one P1S plate in {pl2['time']} ({pl2['filament_g']:.1f}&#8202;g, solid at 0.1&#8202;mm, the covers with six
perimeters), and pad B00R on the XL in {pbr_t} with the same three pauses. The covers have one, two and three notches in the distal
edge right of the centre line (the W covers' are left of it).</p>
<ol>
<li>The proximal pair of board screws becomes M2 &#215; 8 (heads on the raised seats); the distal pair M2 &#215; 8 through TP2's ears or
M2 &#215; 10 through TW2's blocks, all ending at x_F 7.90 in the tongue as before.</li>
<li>Pads printed for the old covers (B00, B00P) meet the raised heads at their proximal corners: cut each proximal notch 3&#8202;mm deeper
with a knife (a 4.2&#8202;mm circle around the screw, to 4.5&#8202;mm from the back face), or print B00R.</li>
</ol>
{fig(14, DIP / 'v3/b/media/20261021-dip_link_v3b_solid_covers_and_caps2.png', "Cover I05 with its screws; caps TW2 and TP2 on the "
     "finger with pad B00R; the raised seat beside the pad's deeper notch.", 1000)}
{table(["Cover", "Interference per side (mm)", "Interference with B00R (mm&#179;)", "Mass, solid (g)", "Notches"], t9,
       num_cols=(1, 2, 3, 4), caption="The solid-wall covers (scripts/dip_b_tip_cap.py covers2, "
       "v3/b/20261021-dip_link_v3b_solid_covers_and_caps2.json).", n=9)}
{table(["Cap", "Sides", "Side height (mm)", "Distal screws", "Interference with B00R (mm&#179;)", "Sides to B00R (mm)",
        "Mass, solid (g)"], t10, num_cols=(2, 4, 5, 6), caption="The second caps; TW2's blocks stand 0.05 mm inside the pad's flank, which shows as "
       "interference.", n=10)}"""))

    S.append(("weights", "Weights of the 85A fit parts", f"""
<p>The A tips weigh {meas['A']['mass_g']:.2f}&#8202;g and the B pads {meas['B']['mass_g']:.2f}&#8202;g, 14&#8202;% under the slicer's
{meas['A']['sliced_g_at_profile_1p26']:.2f} and {meas['B']['sliced_g_at_profile_1p26']:.2f}&#8202;g, which take the XL profile's
1.26&#8202;g/cm&#179;; Siraya Tech gives 1.16&#8202;g/cm&#179; for Flex TPU 85A (ISO 1183). At 1.16 the parts contain
{100 * (1 - meas['A']['measured_over_commanded']):.1f} and {100 * (1 - meas['B']['measured_over_commanded']):.1f}&#8202;% less TPU than
the slicer extruded, and {100 * (1 - meas['A']['measured_over_fe']):.1f}&#8202;% less and {100 * (meas['B']['measured_over_fe'] - 1):.1f}&#8202;%
more than the finite elements model; by the under-extrusion runs of {link(PAGES['fe'], '10-19')}, about 2.3&#8211;2.5&#8202;% of approach per
1&#8202;% of TPU, the A tips are about 7&#8202;% softer than modelled and the B pads as modelled. A filament thinner than 1.75&#8202;mm would
explain the gap to the slicer as well as under-extrusion: one metre of the spool weighs {meas['A']['filament_1m_g_expected']:.2f}&#8202;g at
1.75&#8202;mm and 1.16&#8202;g/cm&#179;.</p>"""))

    S.append(("limits", "Limitations", """
<p>The clearances, gaps and overlaps are computed on meshes of the designs, not of prints; the owner's printers have made holes
0.1&#8202;mm small and pins near size. The walls' effect comes from linear finite elements with the flank held both ways; the
pop-off itself, a large deformation of the pad over the dovetails, is not modelled, so the fit covers decide it. The HEX-E's x-axis
is taken as the owner reads it (centre to the nub); the sensor's own marking decides it. The readout model is a dipole model with
the MLX90393's datasheet noise and the board 0.42&#8202;mm behind the pad; nothing in Table 7 is measured, and its depth sweep
ignores the 1.8&#8202;mm pocket-wall rule, which at the 2.9&#8202;mm seat only 7.5&#8202;mm meets for either layout on the 11/22. The homing assumes the tool axis along the
base's x; another heading moves the post's position by the same rotation. The caps' holds are bonded in the model, and whether a cap
keeps the pad on is a test on the bench.</p>"""))

    S.append(("future", "Future work", f"""
<ul>
<li>Read the five chips under pads B00 and B4 at GAIN_SEL&#8202;1 (RES&#8202;1 for B00) before any touch: the model puts
{b_u1:.1f}&#8202;mT on U1 under B00 and at most {sens['bm_four_d75']['B_max_component_mT']:.1f}&#8202;mT on a chip under B4. A field more than
20&#8202;% off (about 0.3&#8202;mm of distance) means the magnets sit at another depth or the board at another gap. Then press each pad
at the contact point with the UR5e at 0.5, 1 and 3&#8202;N, ten times each, and compare the spread of the force estimates: Table 7
predicts B4's {sens['bm_four_d75']['gain']['centre_force_N'] / sens['bm_five_d75']['gain']['centre_force_N']:.1f} times B00's; equal
spreads would mean the readout is limited by something the model leaves out, such as the board's own noise.</li>
<li>On the HEX-E, check that the palmar normal lies along the sensor's +x: press the pad by hand along the palmar normal and read Fx
against Fy; more than 5&#8202;% of the force in Fy means the axis or the dowel's position differs, and <code>dowel_ang</code> turns the
whole mount.</li>
<li>Home on each indenter three times, re-seating the key each time, and record the poses: a spread over 0.1&#8202;mm or 0.2&#176;
about the key's axis means the long faces do not seat (ends binding: raise <code>home_end_relief</code>).</li>
<li>Slide the link into the new fork without the clamp plate and rock it by hand: if it rocks more than Table 4's turns suggest,
the printed walls stand further off than 0.15&#8202;mm and <code>mold_gap</code> 0.10 tightens them; if it does not go in, 0.20.</li>
<li>Weigh one metre of the 85A spool to 0.01&#8202;g. Near {meas['A']['filament_1m_g_expected']:.2f}&#8202;g the A tips are under-extruded
by about 7&#8202;%; near {meas['A']['filament_1m_g_expected'] * meas['A']['measured_over_commanded']:.2f}&#8202;g the filament is thin and the
tips contain what the slicer laid down. The UR5e indentation of one tip then measures the stiffness the weighing predicts.</li>
<li>Print plate <code>20261021-dip_link_v3b_p1s_pla_solid_covers_caps2.3mf</code> and pad B00R. Press B00R into I05, I10 and I15
in turn and push it sideways at the apex as hard as popped it off W00: take the least interference that keeps it on; if I15 does,
and the pad's static field at rest moves more than 10&#8202;% from W00's, the clamp is squeezing the magnets and I10 is the choice.</li>
<li>Print plate <code>20261020-dip_link_v3b_p1s_pla_tip_caps.3mf</code> and pad B00P, fit TP, and push the pad sideways at the
tip and at the apex as hard as popped it off before; if it stays on, keep TP (Table 8's smaller cost), otherwise fit TW. With the
cap on, repeat the presses of the first item: a spread more than Table 8's change above the walls-only pad means the plate preloads
the pad, and a thinner plate or a 0.1&#8202;mm relief over the pad's end removes it.</li>
<li>When the fingers go on the hand, rebuild the v2 finger's scenes with v3's link (<code>scripts/v2_finger_scenes.py</code> with the
v3 meshes) and run the first-bench export and the trajectory-clearance gate: the narrower neck changes the index and middle links'
closest approach, which was 0.10&#8202;mm with v2 a.</li>
</ul>"""))

    files = [DIP / "params/20261020-v3_concept_b.json", DIP / "v3/b/evaluation.json", DIP / "v3/b/cad", DIP / "v3/b/print",
             D / f"{UP}_check.json", D / f"{UP}_payload.json", D / f"{UP}_parts.json", D / f"{UP}_step.json", D / "cad", D / "print", D / "README.md",
             ROOT / "scripts/dip_link_concept_b.py", ROOT / "scripts/dip_link_v2.py", ROOT / "scripts/ur5_calibration_mount.py",
             ROOT / "scripts/v3b_ur5v2_pla_plate.py", ROOT / "scripts/v3b_pad_b00_magnets_xl.py", ROOT / "scripts/v2_tip_fe_checks.py", ROOT / "scripts/dip_b_v3_figures.py",
             ROOT / "scripts/dip_b_magnet_layouts_figure.py", ROOT / "scripts/dip_b_tip_cap.py",
             DIP / "v3/b/20261020-dip_link_v3b_tip_caps.json", DIP / "v3/b/20261021-dip_link_v3b_solid_covers_and_caps2.json",
             ROOT / "scripts/dip_b_v3_ur5_v2_page.py"]
    S.append(("files", "Files", "<ul>" + "".join(f"<li>{code(f)}</li>" for f in files if Path(f).exists()) + "</ul>"
              + f"<p>Related pages: {link(PAGES['dip'], 'DIP-link redesign')}, {link(PAGES['ur5'], 'UR5e mount v1')}, "
                f"{link(PAGES['fe'], 'v2 fingertip checks')}, {link(PAGES['status'], 'printed-fingertip status')}.</p>"))

    toc, body = [], []
    for k, (sid, title, html_) in enumerate(S):
        num = "" if sid in ("terms",) else f"{k}. "
        toc.append(f'<li><a href="#{sid}">{num}{esc(title)}</a></li>')
        body.append(f'<section id="{sid}"><h2>{num}{esc(title)}</h2>{html_}</section>')
    tpl = TEMPLATE.read_text()
    out = (tpl.replace("{{TITLE}}", "DIP link b v3 and UR5e mount v2")
           .replace("{{TEXCSS}}", "").replace("{{EYEBROW}}", "Printed fingertip &#183; 2026-10-20")
           .replace("{{H1}}", "DIP link b v3 and UR5e calibration mount v2").replace("{{LEDE}}", lede)
           .replace("{{BYLINE}}", f"Built {time.strftime('%Y-%m-%d %H:%M')} by scripts/dip_b_v3_ur5_v2_page.py")
           .replace("{{TOC}}", "\n".join(toc)).replace("{{SECTIONS}}", "\n".join(body))
           .replace("{{FOOTER}}", "Local page; data in the folders named under Files."))
    out = retro_style.apply(out)
    PAGE.write_text(out)
    print(f"page -> {PAGE.relative_to(ROOT)} ({PAGE.stat().st_size / 1e6:.2f} MB); lede {len(lede.split())} words")


if __name__ == "__main__":
    main()
