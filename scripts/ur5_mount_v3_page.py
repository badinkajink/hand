#!/usr/bin/env python3
"""Result page of 2026-10-21: UR5e calibration mount v3 for the owner's M6 x 20 screws, tightened from above (owner, 17:54).

Reads docs/experiments/20261021-ur5_calibration_mount_v3/ (parts, STEP read-back, link checks, screw and key checks, payload,
the insert bores of 2026-10-22, the P1S plates' records), written by scripts/ur5_calibration_mount.py with UR5M_REV=v3 and
scripts/v3b_ur5v2_pla_plate.py mount_v3 mount_v3_adapter, and v2's records for the comparison; applies scripts/retro_style.py
and writes docs/experiments/20261021-ur5_calibration_mount_v3/20261021-ur5_calibration_mount_v3.html.

    .venv/bin/python scripts/ur5_mount_v3_page.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import v2_tip_fe_checks_page as VP  # noqa: E402  (table, img_uri, esc)

D = ROOT / "docs/experiments/20261021-ur5_calibration_mount_v3"
PFX = "20261021-ur5_calibration_mount_v3"
PAGE = D / f"{PFX}.html"
TEMPLATE = ROOT / "scripts/ur5_mount_v3_page.template.html"
D2 = ROOT / "docs/experiments/20261020-ur5_calibration_mount_v2"
P2 = "20261020-ur5_calibration_mount_v2"
PAGES = dict(v2=D2 / "20261020-dip_link_b_v3_and_ur5_mount_v2.html",
             v4=ROOT / "docs/experiments/20261021-dip_link_b_v4/20261021-dip_link_b_v4_and_wall_heights.html",
             status=ROOT / "docs/experiments/20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html")
AD_PLATE = "20261022-ur5_calibration_mount_v3_p1s_pla_adapter_inserts"     # the adapter alone, with the insert bores
table, img_uri, esc = VP.table, VP.img_uri, VP.esc
MM, DEG, X = "&#8202;mm", "&#176;", "&#8202;&#215;&#8202;"


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
    return dict(parts=j(D / f"{PFX}_parts.json"), step=j(D / f"{PFX}_step.json")["parts"], chk=j(D / f"{PFX}_check.json"),
                scr=j(D / f"{PFX}_screws.json"), pay=j(D / f"{PFX}_payload.json"), plate=j(D / f"print/{PFX}_p1s_pla.json"),
                ins=j(D / f"{PFX}_inserts.json"), plate_ad=j(D / f"print/{AD_PLATE}.json"),
                parts2=j(D2 / f"{P2}_parts.json"), pay2=j(D2 / f"{P2}_payload.json"), chk2=j(D2 / f"{P2}_check.json"))


def key_cell(keys, tilt):
    k, kp = keys[f"key_tilt{tilt:.0f}"], keys[f"key_plus_0.5_tilt{tilt:.0f}"]
    if kp == "clear":
        return "Clear by 0.5" + MM
    if k == "clear":
        return "Clear"
    return f"Meets the fork ({sum(k.values()):.2f}&#8202;mm&#179;)"


def main():
    d = load()
    P = d["parts"]["parameters"]
    chk, scr, pay, pay2, plate = d["chk"], d["scr"], d["pay"], d["pay2"], d["plate"]
    pip3, pip2 = d["parts"]["frame"]["pip_axis_M"][2], d["parts2"]["frame"]["pip_axis_M"][2]
    dz = pip3 - pip2
    seated = {r["pose"]: r for r in scr["poses"] if r["bolt_deg"] == 45.0}
    prot, prot_w = seated["seated"]["protrusion_mm"], seated["seated_washer"]["protrusion_mm"]
    tilt = P["key_tilt"]
    under = {r["pose"]: r["key_overlaps_mm3"] for r in scr["poses"] if r["bolt_deg"] == 315.0}
    side = {r["pose"]: r["key_overlaps_mm3"] for r in scr["poses"] if r["bolt_deg"] == 45.0}
    ok_tilts = [t for t in (15.0, 20.0, 25.0) if all(under[p][f"key_tilt{t:.0f}"] == "clear" for p in under)]
    capt = {c["bolt_deg"]: c["overlap_above_pocket_mm3"] for c in scr["captive"]}
    objs = {o["id"]: o for o in plate["objects"]}
    tcp3, tcp2 = pay["tcp_contact_from_ur_flange_mm"][2], pay2["tcp_contact_from_ur_flange_mm"][2]
    zero_ov = all(v == 0.0 for r in scr["poses"] for v in r["overlap_mm3"].values())
    links = chk["b_links"]
    ind_same = all(abs(d["parts"]["parts"][k]["volume_mm3"] - d["parts2"]["parts"][k]["volume_mm3"]) < 0.5
                   for k in ("indenter_sphere", "indenter_rod"))
    ins, pad = d["ins"], d["plate_ad"]

    lede = (f"Mount v3 attaches DIP link b to the HEX-E with four M6{X}20 screws, tightened from above after assembly. The "
            f"adapter is {P['ad_t']:.0f}{MM} thick, so a seated screw enters the HEX-E's thread "
            f"{prot:.1f}{MM}. The heads sit captive in the fork's {P['foot_t']:.0f}{MM} foot. The two screws on the slotted-arm side take a straight 5{MM} hex key from above, the two "
            f"under the column's corners a ball-end key at {tilt:.0f}{DEG}. Everything above the foot rose {dz:.0f}{MM}; the links, "
            f"clamp plate and indenters fit as in v2. Since 2026-10-22 the foot's four M3 screws go into brass heat-set inserts in "
            f"{P['ins_hole']:.1f}{MM} bores of the adapter, which reprints alone in {pad['time']}.")

    S = []
    S.append(("terms", "Terms and metrics", f"""
<dl class="terms">
<dt>M</dt><dd>The mount frame, in mm: origin at the centre of the adapter's face on the HEX-E, z_M along the tool axis toward the
fingertip, x_M toward the palmar side (the contact normal, toward the dowel), y_M along the PIP axis toward the slotted arm. The four
M6 screws stand on a 50{MM} circle at 45, 135, 225 and 315{DEG} from +x_M: two at y_M +17.68 (the slotted-arm side) and two at
y_M &#8722;17.68, 0.68{MM} outside the column's faces and under its corners.</dd>
<dt>Seated, lifted</dt><dd>A screw is seated when its head rests on the adapter's top face (without a washer: the head's top at
z_M {P['ad_t'] + P['m6_head_k']:.0f}) and lifted when it has slid up until its tip is flush with the adapter's face on the HEX-E
(the head's top at z_M {P['m6_len'] + P['m6_head_k']:.0f}).</dd>
<dt>Thread engagement</dt><dd>How far a seated screw reaches past the adapter's face into the HEX-E's tapped hole: the screw's
length under the head minus the plastic (and washer) under the head, in mm.</dd>
<dt>Key tilt</dt><dd>The angle between a hex key's axis and the tool axis while its tip is in a screw's socket, in degrees. A key is
modelled as a cylinder of 2.885{MM} radius (a 5{MM} hex key's corners) from 0.5{MM} above the head's top, 129.5{MM} long. It is
clear when its overlap with every part is zero, and clear by 0.5{MM} when a cylinder 0.5{MM} wider is also clear.</dd>
<dt>Overlap</dt><dd>Volume common to two parts' meshes, from manifold3d booleans, in mm&#179;. Values under 0.001 are the
meshes' facets.</dd>
<dt>Turn to contact</dt><dd>The least turn of the DIP link about an axis through the bridge's middle at which it meets the fork
(overlap over 0.001{MM}&#179;), with the clamp plate and the slotted arm's screw off, in degrees; the angles tried were 0.05 to 5{DEG}.</dd>
<dt>Payload</dt><dd>Mass and centre of gravity of everything past the UR5e's flange (the HEX-E, the mount, the DIP link with its
board and pad, the screws), the printed parts at the plastic the slicer lays down; the values for the UR5e's set_payload.</dd>
<dt>Heat-set insert</dt><dd>A knurled brass sleeve with an internal M3 thread, pressed into a printed bore with a soldering iron:
the plastic melts into its knurls and sets, and the screw then turns in brass. Its size is its length and its diameter across
the knurls, in mm.</dd>
<dt>Displaced plastic, room below the insert</dt><dd>The volume between the bore's diameter and the insert's knurl diameter over
the insert's length, an upper bound on the plastic the knurls push aside; and the bore's volume between the insert's end and the
bore's end, in mm&#179;.</dd>
<dt>As sliced</dt><dd>A bore's diameter in the sliced G-code: twice the radius of the extrusion path of the wall next to the bore,
less one line width ({ins['as_sliced']['outer_wall_line_width_mm']:.2f}{MM}), per layer.</dd>
</dl>"""))

    rows = [
        ("M6 screws into the HEX-E", f"M6{X}12, DIN 125 washer", f"M6{X}20, washer optional"),
        ("Plastic under each M6 head", f"3.0{MM}", f"{P['ad_t']:.1f}{MM}"),
        ("Thread engagement", f"7.4{MM}", f"{prot:.1f}{MM}; {prot_w:.1f}{MM} over a washer"),
        ("Adapter, thickness", f"{d['parts2']['parameters']['ad_t']:.0f}{MM}", f"{P['ad_t']:.0f}{MM}"),
        ("Foot of the standoff-fork", f"{d['parts2']['parameters']['foot_t']:.0f}{MM}, 63{MM} across",
         f"{P['foot_t']:.0f}{MM}, {P['foot_d']:.0f}{MM} across"),
        ("M6 heads", "In the adapter's counterbores, covered by the foot", "Captive in pockets of the foot, keys from above"),
        ("Foot screws", f"4 M3{X}12 at 22.5, 112.5, 202.5, 280{DEG}, threads formed in the adapter",
         f"4 M3{X}12 counterbored, at 0, 90, 198, 270{DEG}, into brass heat-set inserts (2026-10-22)"),
        ("PIP axis above the HEX-E's face", f"{pip2:.0f}{MM}", f"{pip3:.0f}{MM}"),
        ("Contact point from the UR5e's flange", f"{tcp2:.1f}{MM}", f"{tcp3:.1f}{MM}"),
        ("Payload: mass, centre of gravity from the flange",
         f"{pay2['total_g']:.0f}&#8202;g, {pay2['cog_from_ur_flange_mm'][2]:.1f}{MM}",
         f"{pay['total_g']:.0f}&#8202;g, {pay['cog_from_ur_flange_mm'][2]:.1f}{MM}"),
    ]
    S.append(("changes", "Request of 2026-10-21 17:54 and the changes", f"""
<p>The owner, at 17:54: "i forgot that i only have 20mm long m6 screws. so we need to pop everything up a little. there is not much
thread on the hex e ft sensor. then, if you could make the holes accessible from the top plate, that would be great. so i dont have
to mount it to the robto first. i can pre-place the screws through the dip link interface. now that we've offset the dip link, two
holes could easily be exposed, and two we might have to finnagle in there. should still be able to tighten with a ball-end allen key."
The thread depth of the owner's HEX-E is unknown. The UR5e's flange allows 8{MM} and the Quick Changer's tool plate 9{MM}, so v3 sets
the engagement to {prot:.1f}{MM}, and a washer under each head brings it to {prot_w:.1f}{MM}.</p>"""
              + table(["Quantity", "Mount v2", "Mount v3"], rows, caption="Mount v2 against v3; Figure 1 shows the screws in their "
                      "pockets.", n=1)))

    S.append(("access", "Screw pockets and key access", fig(1, D / f"media/{PFX}_sections.png",
        f"Sections through the M6 axes in the x_M&#8211;z_M plane: the adapter (tan), the standoff-fork (blue), the clamp plate (dark "
        f"blue), the screws seated (grey) and lifted (dashed heads), the 5{MM} key (red). Left, y_M +17.68: straight keys through "
        f"{P['key_d']:.0f}{MM} holes. Right, y_M &#8722;17.68: keys at {tilt:.0f}{DEG} through channels that leave the column's palmar "
        f"and dorsal faces.", width=1200)
        + f"""
<p>Each M6 head sits in a pocket {P['m6_pocket_d']}{MM} across that reaches from the adapter's top face to z_M
{P['m6_pocket_top']}, so a screw can lift until its tip is flush with the adapter's face and the adapter sits flat on the HEX-E
before any screw is started. A head (10{MM}) cannot pass the {P['key_d']:.0f}{MM} hole or the {P['key_d']:.0f}{MM}-wide channel
above it. A head 0.1{MM} above its pocket meets the foot with {min(capt.values()):.0f}&#8211;{max(capt.values()):.0f}&#8202;mm&#179;
at the four screws. {'The screws and the printed parts have zero overlap in all three positions at all four screws.' if zero_ov else 'The check file lists overlaps of the screws with the printed parts.'}</p>
<p>The two screws on the slotted-arm side lie 4.4{MM} beyond the column's face, so a straight key reaches them. The holes above
them notch the outer corners of the walls around the slotted leg, 3.5{MM} from each screw's axis, away from the faces that locate
the leg. The two screws at y_M &#8722;17.68 sit under the column's corners, where the fork's ears and the clamp plate's M3 screws lie
above them. A channel leans {tilt:.0f}{DEG} out of the column's palmar or dorsal face and is wide enough for the socket's height
anywhere between seated and lifted. Table 2 gives the keys at the tilts tried, within the 25{DEG} a ball end usually allows. At
{', '.join(f'{t:.0f}' for t in ok_tilts)}{DEG} the key clears the fork with the screw seated, on a washer and lifted. At 15{DEG} it
grazes the channel with the screw lifted, and at 25{DEG} with the screw seated without a washer.</p>"""
        + table(["Screws", "Key tilt", "Seated", "Seated on a washer", "Lifted"],
                [("y_M +17.68 (45, 135" + DEG + ")", f"0{DEG}, straight", *(key_cell(side[p], 0.0) for p in ("seated", "seated_washer", "lifted"))),
                 ("y_M +17.68", f"10{DEG}", *(key_cell(side[p], 10.0) for p in ("seated", "seated_washer", "lifted")))]
                + [("y_M &#8722;17.68 (225, 315" + DEG + ")", f"{t:.0f}{DEG}, ball end",
                    *(key_cell(under[p], t) for p in ("seated", "seated_washer", "lifted"))) for t in (15.0, 20.0, 25.0)],
                caption="A 5" + MM + " hex key in each screw's socket against the mount, the clamp plate, DIP links b v3 and v4, "
                        "their covers and the pad; only the fork is ever met.", n=2)))

    lrows = [(k.replace("v2b", "b v2").replace("v3b", "b v3").replace("v4b", "b v4"), f"{v['overlap_fork_mm3']:.4f}",
              "Yes" if v["insertion_clear"] else "No", f"{v['least_gap_mm']:.3f}",
              f"{v['turn_to_contact_deg']['about_x_+']}", f"{v['turn_to_contact_deg']['about_y_pip_+']}",
              f"{v['turn_to_contact_deg']['about_z_tool_+']}") for k, v in links.items()]
    hom = chk["poses_table"]
    hom_ok = all(all(v == 0.0 for kk, v in hom[p].items() if kk != "socket_floor_minus_pin_top_mm")
                 for p in ("sphere_touch", "sphere_homing", "rod_touch", "rod_homing"))
    S.append(("fit", "DIP links, clamp plate and indenters", f"""
<p>Above the foot, v3's fork is v2's moved {dz:.0f}{MM} along the tool axis, with the walls around the legs computed again from DIP
links b v2, v3 and v4 (their union equals v2's, area for area, because v4 lies inside v3). Table 3 gives the three links in the fork:
no overlap, the link slides on along the tool axis, and it turns the same small angles before it meets the fork as in v2. The clamp
plate is v2's. The homing socket moved with the fork, so the contact point's offset from it is unchanged and the indenters of
v2 fit as printed ({'their meshes rebuild to the same volumes' if ind_same else 'see the parts file'}); at homing and at the touch,
neither indenter overlaps the mount, the link or the pad ({'zero in all four poses' if hom_ok else 'see the check file'}).</p>"""
        + table(["Link", "Overlap with the fork (mm&#179;)", "Slides on", "Least gap (mm)", "Turn about x (" + DEG + ")",
                 "About the PIP axis (" + DEG + ")", "About the tool axis (" + DEG + ")"], lrows, num_cols=(1, 3, 4, 5, 6),
                caption="DIP links b in the v3 fork (scripts/ur5_calibration_mount.py check).", n=3)
        + fig(2, D / f"media/{PFX}_keys.png", f"Mount v3 on the HEX-E with DIP link b v4, the tool axis up as when the screws are "
              f"tightened from above: a straight key in a screw on the slotted-arm side and a ball-end key at {tilt:.0f}{DEG} in one "
              f"under the column.")
        + fig(3, D / f"media/{PFX}_exploded.png", "The parts exploded along the tool axis in the order they go together: the M3 "
              "inserts into the adapter, the M6 screws into the adapter, the fork over their heads, then the DIP link and the clamp "
              "plate.")))

    ib, sl = ins["bores"], ins["as_sliced"]["bores"]
    dia = [v for b in ib for k, v in b["diameter_mm_at_depth"].items() if float(k) >= 0.35]
    s_lo, s_hi = min(b["hole_d_range_mm"][0] for b in sl), max(b["hole_d_range_mm"][1] for b in sl)
    scr_ov = max(abs(v) for b in ib for v in b["screw_overlap_mm3"].values())
    sc, bo = ins["screw"], ins["bore"]
    irows = [
        ("Insert", f"M3{X}{ins['insert']['length_mm']:.1f} brass, {ins['insert']['knurl_d_mm']:.1f}{MM} across the knurls",
         f"CNC Kitchen's M3{X}5.7; its M3{X}3 takes the same bore"),
        ("Bore diameter", (f"{min(dia):.2f}" if f"{min(dia):.2f}" == f"{max(dia):.2f}" else f"{min(dia):.2f}&#8211;{max(dia):.2f}")
         + f"{MM} from 0.35{MM} below the face to the end", f"4.0{MM}, straight"),
        ("Bore depth", f"{bo['depth_mm']:.1f}{MM}", f"At least {ins['rule']['min_depth_mm']:.1f}{MM}: the insert's length and 1{MM}"),
        ("Mouth", f"{bo['mouth_chamfer_mm']:.1f}{MM}{X}45{DEG} chamfer, {bo['mouth_d_mm']:.1f}{MM} across at the face",
         "No chamfer needed"),
        ("Least wall", f"{min(b['least_wall_mm'] for b in ib):.1f}{MM} to the disc's rim, "
                       f"{min(b['knurl_wall_mm'] for b in ib):.1f}{MM} from the knurls", f"At least {ins['rule']['min_wall_mm']:.1f}{MM}"),
        ("Displaced plastic; room below the insert", f"{ins['displaced_envelope_mm3']:.0f} and "
                                                     f"{ins['room_below_insert_mm3']:.0f}&#8202;mm&#179;", ""),
        (f"M3{X}12 from the foot", f"Ends {sc['reach_below_face_mm']:.1f}{MM} below the face: {sc['engaged_mm']:.1f}{MM} in the "
                                   f"insert, the tip {sc['tip_to_bore_end_mm']:.1f}{MM} above the bore's end; overlap with the "
                                   f"adapter and the fork {scr_ov:.0f}&#8202;mm&#179;", ""),
        ("As sliced", f"{s_lo:.2f}&#8211;{s_hi:.2f}{MM} from the second layer to {sl[0]['last_layer_z_mm']:.1f}{MM}; the first layer "
                      f"{sl[0]['first_layer_hole_d_mm']:.2f}{MM}", ""),
    ]
    S.append(("inserts", "Brass inserts in the adapter (2026-10-22)", f"""
<p>The owner, on 2026-10-22: "having some trouble with the ur5 mount plate. let's not go for self-threaded m3 holes there and just
put four brass insert bores in there. make suer to size the bore correctly for the brass insert, theres some guidance online on how
to shape the profile for a good press insert." The four 2.75{MM} holes in which the foot's M3{X}12 formed their own thread were
replaced by bores for brass heat-set inserts in the adapter's top face. The fork is unchanged. Its 3.4{MM} clearance holes and
{P['m3_cb_d']:.1f}{MM} counterbores stand over the inserts, so the fork printed on 2026-10-21 is kept and only the adapter is
printed again.</p>
<p>The bore follows CNC Kitchen's design table for its M3{X}5.7 insert, {ins['insert']['knurl_d_mm']:.1f}{MM} across the knurls:
a straight {bo['d_mm']:.1f}{MM} hole, walls of at least {ins['rule']['min_wall_mm']:.1f}{MM}, and a blind hole at least 1{MM}
longer than the insert. Its guide adds that no chamfer is needed and that the insert is pressed flush. The bore is
{bo['depth_mm']:.1f}{MM} deep, {bo['depth_mm'] - ins['insert']['length_mm']:.1f}{MM} longer than the insert, because the screw
passes through the insert and ends {sc['reach_below_face_mm']:.1f}{MM} below the face. A {bo['mouth_chamfer_mm']:.1f}{MM}{X}45{DEG}
chamfer at the mouth departs from the guide: the adapter prints with this face on the bed, where the first layer spreads into the
hole, and the plastic an insert pushes up collects in the chamfer below the face on which the foot sits. Table 4 gives the bores as
exported and as sliced; in the sliced G-code the slicer's elephant-foot compensation widens the first layer to
{sl[0]['first_layer_hole_d_mm']:.2f}{MM}, and the bore's end is bridged.</p>
<p>Each insert is pressed in from the top face before the M6 screws go in: with the soldering iron it is sunk about 90&#8202;% of
the way, then a flat tool presses it flush and holds it for a few seconds while the plastic sets (CNC Kitchen's procedure). The
adapter's six walls put 2.7{MM} of solid plastic around each bore, so the knurls cut into perimeters, not infill.
{link(D / f'print/{AD_PLATE}.3mf', f'print/{AD_PLATE}.3mf')} is the adapter alone, {pad['time']}, {pad['filament_g']:.0f}&#8202;g,
with the settings of the full plate (Section 5).</p>"""
        + table(["Quantity", "Adapter v3, all four bores", "CNC Kitchen's guidance"], irows,
                caption="The adapter's insert bores, measured on the exported mesh and in the sliced G-code of the adapter's plate "
                        "(scripts/ur5_calibration_mount.py inserts), against CNC Kitchen's table for its M3 inserts.", n=4)
        + fig(4, D / f"media/{PFX}_inserts.png", f"Left: the adapter 3{MM} below its top face, seen along the tool axis: the four "
              f"insert bores ({bo['d_mm']:.1f}{MM}; dashed, the inserts' {ins['insert']['knurl_d_mm']:.1f}{MM} knurls) at 0, 90, 198 and "
              f"270{DEG}, and the four M6 holes ({P['m6_hole']:.1f}{MM}). Right: a section along the tool axis through the foot screw at "
              f"90{DEG}: the adapter (tan), the foot (blue), the insert (gold) flush with the face, the M3{X}12 (grey).", width=1200)))

    o1, o2 = objs["U1"], objs["U2"]
    S.append(("printing", "Printing and assembly", f"""
<p>{link(D / f'print/{AD_PLATE}.3mf', f'print/{AD_PLATE}.3mf')}: the adapter (U1) alone, {pad['time']},
{pad['filament_g']:.0f}&#8202;g, for the fork already printed. {link(D / f'print/{PFX}_p1s_pla.3mf', f'print/{PFX}_p1s_pla.3mf')}:
the adapter and the standoff-fork (U2) on one plate, {plate['time']}, {plate['filament_g']:.0f}&#8202;g. Both are PLA on the P1S at
Bambu Studio's 0.20{MM} Standard with 15&#8202;% grid infill, as the fork of 2026-10-20 19:10. The adapter has six walls, so
2.7{MM} of solid plastic rings each M6 hole from the head down to the HEX-E and each insert bore, and the head's bearing ring
(radius 3.2 to 5.0{MM}) never stands on infill; the fork has four. The adapter prints on its top face, the dowel up; the fork on
its foot, the pockets' ceilings bridged over 12.5{MM}. No supports; the slicer's check passed (U1 {o1['layers']} layers, U2
{o2['layers']}; plastic over volume {o1['fill_ratio']:.2f} and {o2['fill_ratio']:.2f}).</p>
<ol>
<li>Four M3 heat-set inserts into the adapter's bores from its top face (the face away from the dowel), each flush with the face:
an insert left standing proud tilts the foot.</li>
<li>Four M6{X}20 socket head screws into the adapter's holes from its top face. A DIN 125 washer under each head shortens the
thread engagement from {prot:.1f} to {prot_w:.1f}{MM}.</li>
<li>The fork over the heads, each into its pocket; four M3{X}12 through the counterbores in the foot into the inserts, snug. The M6
screws are now captive.</li>
<li>The DIP link and the clamp plate as in v2: the link slides on along the tool axis, two M3{X}16 through the clamp plate, the
M3{X}12 with its DIN 9021 washer through the slotted arm.</li>
<li>Onto the HEX-E: the dowel into its hole. With the flange facing down the screws slide out until their heads meet the pockets'
tops and their tips are flush; push each up with the key and turn it into the thread.</li>
<li>The two screws on the slotted-arm side with a straight 5{MM} key from above, the two under the column with a ball-end key at
about {tilt:.0f}{DEG}, leaning out to the palmar and to the dorsal side. Snug, about 1.5&#8202;N&#8202;m: the heads bear on PLA.</li>
<li>UR5e: set_payload {pay['total_g'] / 1000:.3f}&#8202;kg, centre of gravity ({', '.join(f'{v:.1f}' for v in pay['cog_from_ur_flange_mm'])}){MM}
from the flange; the contact point at ({', '.join(f'{v:.2f}' for v in pay['tcp_contact_from_ur_flange_mm'])}){MM}.</li>
</ol>"""))

    S.append(("future", "Future work", f"""
<ul>
<li>Insert size: measure one insert's knurls with calipers before pressing any in. The bore takes {ins['insert']['knurl_d_mm']:.1f}{MM}
knurls; for 4.0{MM} inserts (the generic M3{X}4{X}5) set <code>ins_hole</code> to 3.6, for 5.0{MM} ones (the Voron
M3{X}5{X}4) to 4.4, in the v3 block of <code>P</code> in <code>scripts/ur5_calibration_mount.py</code>, then rebuild with
<code>UR5M_ONLY=adapter</code> and slice the plate <code>mount_v3_adapter</code>. An insert that turns with the screw or rises when
the screw is tightened says the bore is too large for it.</li>
<li>Thread depth of the HEX-E: turn one M6{X}20 in by hand until it stops and count the turns (1{MM} per turn). Under
{prot + 0.5:.1f}{MM}, put a DIN 125 washer under each head ({prot_w:.1f}{MM}) or rebuild with <code>ad_t</code> 1{MM} larger in the v3
block of <code>P</code> in <code>scripts/ur5_calibration_mount.py</code>.</li>
<li>Seat creep: tighten one M6 to 1.5&#8202;N&#8202;m on the printed adapter, mark the head's height and measure it again after a day.
A head that sinks more than 0.1{MM} shows that the six walls under it are too few.</li>
<li>Homing repeatability and the HEX-E's x-axis check, still open from v2, on the raised mount.</li>
</ul>"""))

    S.append(("limits", "Limitations", f"""
<p>The HEX-E's tool-side thread depth comes from no drawing; the data sheets give 8{MM} on the UR5e's flange and 9{MM} on the Quick
Changer's tool plate. The key model is a cylinder over the hex's corners, and the tilt a ball-end key allows depends on its
maker. The torque the heads take on PLA, the print's fit on the dowel and the bridges over the pockets are untested. The insert
bores are CNC Kitchen's for a {ins['insert']['knurl_d_mm']:.1f}{MM} insert; the owner's inserts were not measured, and the printed
bores were not gauged.</p>"""))

    files = [D / f"{PFX}_parts.json", D / f"{PFX}_step.json", D / f"{PFX}_check.json", D / f"{PFX}_screws.json",
             D / f"{PFX}_inserts.json", D / f"{PFX}_payload.json", D / "cad", D / "print", D / "media", D / "README.md",
             ROOT / "scripts/ur5_calibration_mount.py",
             ROOT / "scripts/v3b_ur5v2_pla_plate.py", ROOT / "scripts/ur5_mount_v3_page.py"]
    S.append(("files", "Files", "<ul>" + "".join(f"<li>{code(f)}</li>" for f in files if Path(f).exists()) + "</ul>"
              + f"<p>STEP files (Inventor: File &gt; Open, STEP, Convert Model) are exact solids in M; each read back as one valid solid "
                f"within 0.1&#8202;% of its mesh's volume. Related pages: {link(PAGES['v2'], 'DIP link b v3 and UR5e mount v2')}, "
                f"{link(PAGES['v4'], 'DIP link b v4 and the wall-height sweep')}, {link(PAGES['status'], 'printed-fingertip status')}.</p>"))

    toc, body = [], []
    for k, (sid, title, html_) in enumerate(S):
        num = "" if sid == "terms" else f"{k}. "
        toc.append(f'<li><a href="#{sid}">{num}{esc(title)}</a></li>')
        body.append(f'<section id="{sid}"><h2>{num}{esc(title)}</h2>{html_}</section>')
    out = (TEMPLATE.read_text().replace("{{TITLE}}", "UR5e calibration mount v3")
           .replace("{{TEXCSS}}", "").replace("{{EYEBROW}}", "Printed fingertip &#183; 2026-10-21")
           .replace("{{H1}}", "UR5e calibration mount v3 with captive M6 &#215; 20 screws").replace("{{LEDE}}", lede)
           .replace("{{BYLINE}}", f"Built {time.strftime('%Y-%m-%d %H:%M')} by scripts/ur5_mount_v3_page.py")
           .replace("{{TOC}}", "\n".join(toc)).replace("{{SECTIONS}}", "\n".join(body))
           .replace("{{FOOTER}}", "Local page; data in the folders named under Files."))
    summary = (f"Mount v3 takes M6 x 20 screws: a 15 mm adapter sets {prot:.1f} mm of thread in the HEX-E, the heads sit captive in "
               f"the fork's foot and tighten from above, two with a straight key and two under the column with a ball-end key at "
               f"{tilt:.0f} deg; everything above the foot rose {dz:.0f} mm and the DIP links and indenters fit as in v2. Since "
               f"2026-10-22 the foot's four M3 go into brass heat-set inserts in {P['ins_hole']:.1f} mm bores of the adapter.")
    out = docs_index.tag(out, topic="fingertip", summary=summary, date="2026-10-21")
    out = retro_style.apply(out)
    PAGE.write_text(out)
    print(f"page -> {PAGE.relative_to(ROOT)} ({PAGE.stat().st_size / 1e6:.2f} MB); lede {len(lede.split())} words")


if __name__ == "__main__":
    main()
