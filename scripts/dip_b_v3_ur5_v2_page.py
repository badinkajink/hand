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
    pc, pl = plates["calibration"], plates["link"]
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

    lede = (f"Concept b becomes v3 and the UR5e mount v2 after the owner's first prints. The cover gains side "
            f"walls along the pad's flanks, the neck under the pad becomes a column {pv['col_margin']:.1f}&#8202;mm inside the "
            f"11/22 outline, and the cable's leg opens to the top, so the double-ended Qwiic cable lies in. The walls change the "
            f"pad's approach by at most {abs(w3_6[1]):.2f}&#8202;% at 6&#8202;N"
            + (f" and its force resolution from {sb:.3f} to {sw:.3f}&#8202;N" if sw else "") +
            f". The mount loses the spigot and the key, puts the finger on the tool axis, takes the link in a cradle on the "
            f"tongue's back, and homes on a post beside each indenter. Two P1S plates print in {pc['time']} and {pl['time']}.")

    S = []
    S.append(("terms", "Terms and metrics", f"""
<dl class="terms">
<dt>F, M, table frame</dt><dd>F is the DIP study's finger frame (mm; x toward the dorsal side, y along the PIP axis, z up, the
straight finger hanging along &#8722;z, the PIP axis through (0, y, &#8722;51.55)). M is the mount's frame: origin at the centre of the
adapter's face on the HEX-E, z along the tool axis toward the tip, x toward the palmar side, y along the PIP axis. The table frame
of the homing: z up from the table, x along the tool axis toward the tip, the palmar side down.</dd>
<dt>Cover</dt><dd>The PLA plate of concept b between the board's component side and the 85A pad, with windows over the chips and
the two dovetail rails the pad snaps onto (the owner calls it the cassette).</dd>
<dt>Wall clearance</dt><dd>Gap between a cover wall's inner face and the pad's flank, in mm; the fit covers W00, W10 and W20 have
0.00, 0.10 and 0.20&#8202;mm.</dd>
<dt>Clearance to the joint</dt><dd>Least signed distance from a part fixed to the DIP link to the MCP link and the PIP servo over
PIP &#8722;18 to 92&#176; in 0.5&#176; steps, in mm; the specification's R7 asks 0.5&#8202;mm for rigid parts and 1.0&#8202;mm for the cable.</dd>
<dt>Least wall</dt><dd>Thinnest PLA in the checked sections of a printed part, in mm; R11 asks 0.9&#8202;mm (0.8&#8202;mm for two 0.4&#8202;mm
lines).</dd>
<dt>Approach</dt><dd>How far a rigid probe moves into the pad past first touch at a given normal force, from the linear finite
elements of the 10-18 calibration (18 probes on the palmar face, 0.5&#8211;6&#8202;N), in &#181;m.</dd>
<dt>Force resolution</dt><dd>One standard deviation of the normal force estimated from one sample of the five MLX90393 chips at
the contact point, with the contact position and a uniform background field unknown (Cram&#233;r&#8211;Rao bound of the readout model,
<code>tpu_tip_magnet_sensing</code>), at the gain the field needs, in N; smaller is finer.</dd>
<dt>Datum</dt><dd>A face of the mount against which a face of the DIP link is pressed, so that the link takes one pose; the gap
given is the nominal one before the link is pressed on.</dd>
<dt>Homing</dt><dd>Seating the pin of a post on the indenter's base in a socket of the mount, which fixes where the pad's contact
point is relative to the indenter's top.</dd>
</dl>"""))

    t1 = [["The 30&#8202;mm disc on the UR5 adapter has no bore in the HEX-E; only the 5.5&#8202;mm nub fits",
           "Spigot removed; the printed dowel (5.95&#8202;mm) kept", "Adapter 16.0&#8202;mm tall, one solid"],
          ["Palmar side aligned with the sensor's x-axis, taken from the centre toward the nub",
           "The palmar normal points at the dowel (as in v1); the foot's four M3 now sit at 22.5, 112.5, 202.5 and 280&#176;, so the "
           "standoff-fork goes on one way only", "Hex key reaches all four heads"],
          ["The rod in the horn slot is tight and snapped; clamping can leave the link crooked",
           "Key removed; a cradle on the tongue: back plate (x datum), rails (y), the bridge on the inner block's flat bottom (z), an M3 "
           "pusher in the lip", "Both b links go in along x with no overlap"],
          ["Homing geometry to zero the mount to the table indenters, a slot/rod mating",
           "A 6.3&#8202;mm socket in the column's palmar face; a post with a 6.0&#8202;mm pin on each indenter's base",
           "No overlap at homing or touching, either indenter"],
          ["Mount offset so the finger is in line with the centre", "Tool axis on the finger's mid-plane (y_F 0, 3.85&#8202;mm moved)",
           f"Contact point at M ({hs['contact_M'][0]:.2f}, {hs['contact_M'][1]:.2f}, {hs['contact_M'][2]:.2f})&#8202;mm"],
          ["Concept b chosen; the double-ended Qwiic cable needs a channel to slot into",
           "The leg toward the back open to the top out through the back", "Cable clearance unchanged"],
          ["Rails pop off under a sideways push (B00; B05 worse); beef up the cassette with side volume around the pad",
           f"Two walls {pv['wall_h']:.1f}&#8202;mm tall along the pad's flanks from the mount face to the widest section, counterbored "
           "over the proximal screws", f"Approach change {w3_6[1]:+.2f}&#8202;% at 6&#8202;N (FE)"],
          ["The block under the pad takes side and large contacts on rigid PLA; a column a little inside the 11/22, a fillet or angled "
           "edge to the board mount", f"Column {pv['col_margin']:.1f}&#8202;mm inside the 11/22 mount-face outline; 45&#176; "
           f"chamfers of {pv['col_chamfer']:.1f}&#8202;mm to the board mount", f"Link {m2['dip_link_solid']:.1f} &#8594; "
           f"{m3['dip_link_solid']:.1f}&#8202;g solid"]]
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
pad's equator. They run from the mount face to the pad's widest section, the owner's &#8216;from the screw holes at the bottom until
the ellipse starts to curve back in&#8217;, stand {pv['wall_clear']:.2f}&#8202;mm off the flank, and slope outward to the cover's edge
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
    for c, lab in (("b_v2", "No walls (v2 cover)"), ("b_v2_wall3", f"Walls {pv['wall_h']:.1f}&#8202;mm (v3 cover)"),
                   ("b_v2_wall495", "Walls over the whole straight flank, 4.95&#8202;mm")):
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
flank's nodes in y over their height and length, both ways; the printed wall stands 0.10&#8202;mm off and resists only outward
motion, so the change is an upper bound. With {pv['wall_h']:.1f}&#8202;mm walls the approach falls by {abs(w3_6[0]):.2f}&#8202;% on average at 6&#8202;N
(at most {abs(w3_6[1]):.2f}&#8202;% over {w3_6[2]} probes), and with walls over the whole straight flank by {abs(w5_6[1]):.2f}&#8202;% at most
(Table 3), against the calibration's 5&#8202;% bound."""
              + (f" The magnets then move {sens['b_v2_wall3']['magnet_travel_um_per_N']:.2f} in place of "
                 f"{sens['b_v2']['magnet_travel_um_per_N']:.2f}&#8202;&#181;m per newton, so the force resolution at the contact point "
                 f"goes from {sb:.4f} to {sw:.4f}&#8202;N, {100 * (sw / sb - 1):.0f}&#8202;% coarser, and the contact position stays "
                 f"within {sens['b_v2_wall3']['gain']['centre_sig_y_mm']:.2f} by {sens['b_v2_wall3']['gain']['centre_sig_z_mm']:.2f}"
                 "&#8202;mm; concept a's tip gives 0.065&#8202;N." if sw else " The readout rows are pending.") + f"""</p>
{table(["Pad", "Rod approach at 1&#8202;N (&#181;m)", "Approach change at 6&#8202;N, mean / largest (%)", "Force resolution (N)",
        "Contact position, 1 sd (mm)"], fe_rows, num_cols=(1, 2, 3, 4),
       caption="b's pad with and without the walls' hold (scripts/v2_tip_fe_checks.py cases b_v2, b_v2_wall3, b_v2_wall495; "
               "fe.jsonl, sensing.json). The rod is the 12.5&#8202;mm one at the contact point.", n=3)}"""))

    gl = chk["b_links"]
    t4 = [[lab, f"{gl['v2b']['gaps_by_region_mm'][k]:.2f}", f"{gl['v3b']['gaps_by_region_mm'][k]:.2f}"]
          for k, lab in (("tongue_back_to_back_plate", "Tongue's back to the back plate (x datum)"),
                         ("bridge_top_to_inner_block", "Bridge to the inner block (z datum, closed by the pusher)"),
                         ("tongue_sides_to_rails", "Tongue's sides to the rails"),
                         ("horn_arm_inner_face", "Horn-side arm to the inner block"),
                         ("slotted_arm_inner_face", "Slotted arm to the inner block (M3 and washer close it)"),
                         ("neck_back_to_back_plate", "Neck's back to the back plate"),
                         ("bridge_back_to_ear", "Bridge's back to the ear"))]
    S.append(("mount", "UR5e calibration mount v2", f"""
<p>Figure 5 shows mount v2 on the HEX-E with link b v3. The adapter has only the printed dowel toward the HEX-E, and the palmar
normal points at it, the sensor's x-axis as the owner takes it from the centre toward the nub. v1 had the same orientation, but its
foot's four M3 were 90&#176; apart, so the standoff-fork could go on in four orientations; at 22.5, 112.5, 202.5 and 280&#176; it goes on in
one. The tool axis now runs through the finger's mid-plane (y_F 0), 3.85&#8202;mm from v1's, so the contact point lies on the axis
in y and 10.55&#8202;mm from it along the palmar normal.</p>
{fig(5, D / f'media/{UP}_on_the_hex_e_palmar.png', 'Mount v2 on the HEX-E (context) with DIP link b v3, palmar side, finger down.')}
<p>The key in the horn slot and the palmar ear are gone. The link now goes in dorsally, along &#8722;x of the mount (Figure 6): the
clevis arms pass either side of the inner block, whose bottom is flat 0.05&#8202;mm above the bridge's highest point, and the tongue
enters a cradle that hangs below the dorsal ear (Figures 7 and 8). The tongue's back face, 21 by 21&#8202;mm right behind the board, rests
on the cradle's back plate, so the normal force of an indentation presses the link onto its datum; rails stand 0.2&#8202;mm beside the
tongue; an M3 in the lip under the tongue's end pushes the link toward the flange until the bridge bears on the inner block; the M3
with its washer through the slotted arm's bore pulls that arm onto the inner block. Table 4 gives the gaps.
{'Both b links, v2 as printed and v3, go in along x with no overlap at any of 121 positions over the last 30&#8202;mm.' if ins_ok else
 'An insertion check failed; see the check file.'}</p>
{fig(6, D / f'media/{UP}_insertion.png', 'The DIP link 16 mm short of its place, going in dorsally.')}
{fig(7, D / f'media/{UP}_fork_alone.png', 'The standoff-fork alone, its fork end up.')}
{fig(8, D / f'media/{UP}_dorsal.png', 'Mount v2 with link b v3 from the dorsal side.')}
{table(["Gap", "v2 b as printed (mm)", "v3 b (mm)"], t4, num_cols=(1, 2),
       caption=f"Gaps of the two b links in mount v2 ({UP}_check.json; nearest surface samples 0.1&#8202;mm apart).", n=4)}"""))

    t5 = [[k, f"({h['post_xy'][0]:.2f}, {h['post_xy'][1]:.2f})", f"{h['pin_top_z']:.2f}", f"{h['indenter_top_z']:.1f}"]
          for k, h in (("Sphere", hs), ("Rod", hr))]
    S.append(("homing", "Homing to the indenters", f"""
<p>Each indenter's base gains a post: a 12&#8202;mm column with a 6.0&#8202;mm pin, 8.5&#8202;mm long, which seats in a 6.3&#8202;mm socket
8.0&#8202;mm deep in the column's palmar face, z_M {chk['poses_table']['sphere_homing_geometry']['socket_floor_M'][2]:.0f}&#8202;mm from the
flange. With the palmar side down and the tool axis along the base's +x, the pin bottoms in the socket when the pad's contact point
stands 30&#8202;mm to +y of the indenter's top and 5&#8202;mm above it (Figure 9); from there the robot moves
(0, &#8722;{P_DY:.0f}, &#8722;{P_DZ:.0f})&#8202;mm in the table frame
to touch. The post sits 40.36&#8202;mm toward the robot from the indenter's top, beside the column at the touching pose.
{POSE_TXT} Table 5 gives the posts. Bolt the base with the post toward the robot.</p>
{fig(9, D / f'media/{UP}_homing.png', 'Homing over the rod indenter: the pin in the socket, the pad 30 mm to the side and 5 mm up.')}
{table(["Indenter", "Post in the base frame, x, y (mm)", "Pin top above the table (mm)", "Indenter top (mm)"], t5,
       num_cols=(2, 3), caption="Homing posts (homing() in scripts/ur5_calibration_mount.py).", n=5)}"""))

    asm = ["Adapter on the HEX-E: dowel into its hole, four M6 &#215; 12 with DIN 125 washers, 1.5&#8202;N&#8202;m.",
           "Standoff-fork on the adapter: four M3 &#215; 12 through the foot; it fits one way, the cradle away from the dowel.",
           "Qwiic cable into J1; pass its far plug up the J1 slot from the neck's underside; slide the board along the tongue's seat, "
           "J1 first.",
           "Lay the cable into the open leg from the top, press it into the groove on the back, Kapton over the leg.",
           "Cover over the board: four M2 &#215; 5 into the tongue (1.9&#8202;mm holes); the two proximal heads through the walls' "
           "counterbores.",
           "Pad onto the rails between the walls (B00).",
           "Link into the mount from the palmar side, arms either side of the inner block, the tongue into the cradle; the cable up "
           "the slot in the inner block and the column's groove.",
           "M3 &#215; 12 with the DIN 9021 washer through the slotted arm's bore, snug; M3 &#215; 8 into the lip until it meets the "
           "tongue's end, then a quarter turn.",
           "Homing post toward the robot; seat the pin in the socket in freedrive, record the pose, then move (0, &#8722;30, &#8722;5)&#8202;mm.",
           f"UR5e payload {plates['payload']['total_g'] / 1000:.3f}&#8202;kg (the HEX-E's 0.347&#8202;kg included) with its centre of gravity "
           f"{plates['payload']['cog_from_ur_flange_mm'][2]:.1f}&#8202;mm out from the flange and "
           f"{plates['payload']['cog_from_ur_flange_mm'][1]:.1f}&#8202;mm along the PIP axis; TCP at the contact point, "
           f"{plates['payload']['tcp_contact_from_ur_flange_mm'][2]:.1f}&#8202;mm out and 10.55&#8202;mm toward the dowel (M axes; "
           f"{UP}_payload.json)."]
    S.append(("assembly", "Assembly", "<ol>" + "".join(f"<li>{a}</li>" for a in asm) + "</ol>"))

    t6 = [[esc(p["project"].split("/")[-1]), p["time"], f"{p['filament_g']:.0f}", ", ".join(o["label"] for o in p["objects"]),
           "none" if not p["problems"] else esc("; ".join(p["problems"]))] for p in (pc, pl)]
    S.append(("print", "Print and CAD files", f"""
<p>Two Bambu Lab P1S plates in PLA (Table 6), the parts' settings written into each project so Bambu Studio keeps them when it slices
again: the mount and the indenters at 0.15&#8202;mm with five walls and 40&#8202;% gyroid, the covers solid at 0.1&#8202;mm, the DIP link at
0.1&#8202;mm on tree supports, and a 5&#8202;mm brim on every part, which also keeps the covers flat. The fit covers W00, W10 and W20 have
one, two and three notches in their distal edge; they fit the v2 link already printed, so the walls can be tried on it before v3
prints. The 85A pad is unchanged (B00). STEP files for Inventor: the mount's parts as exact solids in M ({code(D / 'cad')}, read back
valid), and v3's new body, cover and cable-groove cutter as exact solids in F ({code(DIP / 'v3/b/cad')}).</p>
{table(["Project", "Time", "PLA (g)", "Parts", "Slice checks failed"], t6, num_cols=(1, 2),
       caption="P1S plates (scripts/v3b_ur5v2_pla_plate.py), print/20261020-p1s_pla_*.json.", n=6)}
{fig(10, D / 'print/20261020-p1s_pla_calibration_map.png', 'The calibration plate: mount v2, both indenters, covers W00, W10, W20.', 900)}"""))

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
is taken as the owner reads it (centre to the nub); the sensor's own marking decides it. The homing assumes the tool axis along the
base's x; another heading moves the post's position by the same rotation.</p>"""))

    S.append(("future", "Future work", f"""
<ul>
<li>Print the calibration plate, put cover W10 on the v2 link with pad B00 and push the pad sideways by hand at its widest section;
if it still pops off, try W00, and if W00 binds the pad, W20. The chosen clearance goes into <code>wall_clear</code> and the link plate
prints with it.</li>
<li>On the HEX-E, check that the palmar normal lies along the sensor's +x: press the pad by hand along the palmar normal and read Fx
against Fy; more than 5&#8202;% of the force in Fy means the axis or the dowel's position differs, and <code>dowel_ang</code> turns the
whole mount.</li>
<li>Home on each indenter three times, re-seating the pin each time, and record the poses: a spread over 0.2&#8202;mm means the pin's fit,
and a socket 0.1&#8202;mm smaller tightens it.</li>
<li>Weigh one metre of the 85A spool to 0.01&#8202;g. Near {meas['A']['filament_1m_g_expected']:.2f}&#8202;g the A tips are under-extruded
by about 7&#8202;%; near {meas['A']['filament_1m_g_expected'] * meas['A']['measured_over_commanded']:.2f}&#8202;g the filament is thin and the
tips contain what the slicer laid down. The UR5e indentation of one tip then measures the stiffness the weighing predicts.</li>
<li>When the fingers go on the hand, rebuild the v2 finger's scenes with v3's link (<code>scripts/v2_finger_scenes.py</code> with the
v3 meshes) and run the first-bench export and the trajectory-clearance gate: the narrower neck changes the index and middle links'
closest approach, which was 0.10&#8202;mm with v2 a.</li>
</ul>"""))

    files = [DIP / "params/20261020-v3_concept_b.json", DIP / "v3/b/evaluation.json", DIP / "v3/b/cad", DIP / "v3/b/print",
             D / f"{UP}_check.json", D / f"{UP}_payload.json", D / f"{UP}_parts.json", D / f"{UP}_step.json", D / "cad", D / "print", D / "README.md",
             ROOT / "scripts/dip_link_concept_b.py", ROOT / "scripts/dip_link_v2.py", ROOT / "scripts/ur5_calibration_mount.py",
             ROOT / "scripts/v3b_ur5v2_pla_plate.py", ROOT / "scripts/v2_tip_fe_checks.py", ROOT / "scripts/dip_b_v3_figures.py",
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
