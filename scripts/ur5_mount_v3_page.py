#!/usr/bin/env python3
"""Result page of 2026-10-21: UR5e calibration mount v3 for the owner's M6 x 20 screws, tightened from above (owner, 17:54).

Reads docs/experiments/20261021-ur5_calibration_mount_v3/ (parts, STEP read-back, link checks, screw and key checks, payload,
the P1S plate's record), written by scripts/ur5_calibration_mount.py with UR5M_REV=v3 and scripts/v3b_ur5v2_pla_plate.py
mount_v3, and v2's records for the comparison; applies scripts/retro_style.py and writes
docs/experiments/20261021-ur5_calibration_mount_v3/20261021-ur5_calibration_mount_v3.html.

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

    lede = (f"Mount v3 attaches DIP link b to the HEX-E with four M6{X}20 screws, tightened from above after the mount and the "
            f"link are assembled. The adapter is {P['ad_t']:.0f}{MM} thick, so a seated screw enters the HEX-E's thread "
            f"{prot:.1f}{MM}. Each head sits captive in a pocket of the fork's "
            f"{P['foot_t']:.0f}{MM} foot. The two screws on the slotted-arm side take a straight 5{MM} hex key from above, the two "
            f"under the column's corners a ball-end key at {tilt:.0f}{DEG}. Everything above the foot rose {dz:.0f}{MM}; the links, "
            f"clamp plate and indenters fit as in v2. The two printed parts take {plate['time']} on the P1S.")

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
modelled as a cylinder of 2.885{MM} radius (a 5{MM} hex key's corners) from 0.5{MM} above the head's top, 129.5{MM} long; it is
clear when its overlap with every part is zero, clear by 0.5{MM} when a cylinder 0.5{MM} wider is also clear.</dd>
<dt>Overlap</dt><dd>Volume common to two parts' meshes, from manifold3d booleans, in mm&#179;. Values under 0.001 are the
meshes' facets.</dd>
<dt>Turn to contact</dt><dd>The least turn of the DIP link about an axis through the bridge's middle at which it meets the fork
(overlap over 0.001{MM}&#179;), with the clamp plate and the slotted arm's screw off, in degrees; the angles tried were 0.05 to 5{DEG}.</dd>
<dt>Payload</dt><dd>Mass and centre of gravity of everything past the UR5e's flange (the HEX-E, the mount, the DIP link with its
board and pad, the screws), the printed parts at the plastic the slicer lays down; the values for the UR5e's set_payload.</dd>
</dl>"""))

    rows = [
        ("M6 screws into the HEX-E", f"M6{X}12, DIN 125 washer", f"M6{X}20, washer optional"),
        ("Plastic under each M6 head", f"3.0{MM}", f"{P['ad_t']:.1f}{MM}"),
        ("Thread engagement", f"7.4{MM}", f"{prot:.1f}{MM}; {prot_w:.1f}{MM} over a washer"),
        ("Adapter, thickness", f"{d['parts2']['parameters']['ad_t']:.0f}{MM}", f"{P['ad_t']:.0f}{MM}"),
        ("Foot of the standoff-fork", f"{d['parts2']['parameters']['foot_t']:.0f}{MM}, 63{MM} across",
         f"{P['foot_t']:.0f}{MM}, {P['foot_d']:.0f}{MM} across"),
        ("M6 heads", "In the adapter's counterbores, covered by the foot", "Captive in pockets of the foot, keys from above"),
        ("Foot screws", f"4 M3{X}12 at 22.5, 112.5, 202.5, 280{DEG}", f"4 M3{X}12 counterbored, at 0, 90, 198, 270{DEG}"),
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
The thread depth of the owner's HEX-E is unknown; the UR5e's flange allows 8{MM} and the Quick Changer's tool plate 9{MM}, so v3 sets
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
above it: a head 0.1{MM} above its pocket meets the foot with {min(capt.values()):.0f}&#8211;{max(capt.values()):.0f}&#8202;mm&#179;
at the four screws. {'The screws and the printed parts have zero overlap in all three positions at all four screws.' if zero_ov else 'The check file lists overlaps of the screws with the printed parts.'}</p>
<p>The two screws on the slotted-arm side lie 4.4{MM} beyond the column's face, so a straight key reaches them; the holes above
them notch the outer corners of the walls around the slotted leg, 3.5{MM} from each screw's axis, away from the faces that locate
the leg. The two screws at y_M &#8722;17.68 sit under the column's corners, where the fork's ears and the clamp plate's M3 screws lie
above them; a channel leans {tilt:.0f}{DEG} out of the column's palmar or dorsal face and is wide enough for the socket's height
anywhere between seated and lifted. Table 2 gives the keys at the tilts tried, within the 25{DEG} a ball end usually allows: at
{', '.join(f'{t:.0f}' for t in ok_tilts)}{DEG} the key clears the fork with the screw seated, on a washer and lifted; at 15{DEG} it
grazes the channel with the screw lifted, at 25{DEG} with the screw seated without a washer.</p>"""
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
        + fig(3, D / f"media/{PFX}_exploded.png", "The parts exploded along the tool axis in the order they go together: the M6 "
              "screws into the adapter, the fork over their heads, then the DIP link and the clamp plate.")))

    o1, o2 = objs["U1"], objs["U2"]
    S.append(("printing", "Printing and assembly", f"""
<p>{link(D / f'print/{PFX}_p1s_pla.3mf', f'print/{PFX}_p1s_pla.3mf')}: the adapter (U1) and the standoff-fork (U2) on one P1S plate in
PLA, {plate['time']}, {plate['filament_g']:.0f}&#8202;g, Bambu Studio's 0.20{MM} Standard with 15&#8202;% grid infill as the fork of
2026-10-20 19:10. The adapter has six walls, so 2.7{MM} of solid plastic rings each M6 hole from the head down to the HEX-E and
the head's bearing ring (radius 3.2 to 5.0{MM}) never stands on infill; the fork has four. The adapter prints on its top face, the
dowel up; the fork on its foot, the pockets' ceilings bridged over 12.5{MM}. No supports; the slicer's check passed (U1
{o1['layers']} layers, U2 {o2['layers']}; plastic over volume {o1['fill_ratio']:.2f} and {o2['fill_ratio']:.2f}).</p>
<ol>
<li>Four M6{X}20 socket head screws into the adapter's holes from its top face (the face away from the dowel). A DIN 125 washer
under each head shortens the thread engagement from {prot:.1f} to {prot_w:.1f}{MM}.</li>
<li>The fork over the heads, each into its pocket; four M3{X}12 through the counterbores in the foot into the adapter's 2.75{MM}
holes. The M6 screws are now captive.</li>
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
<li>Thread depth of the HEX-E: turn one M6{X}20 in by hand until it stops and count the turns (1{MM} per turn). Under
{prot + 0.5:.1f}{MM}, put a DIN 125 washer under each head ({prot_w:.1f}{MM}) or rebuild with <code>ad_t</code> 1{MM} larger in the v3
block of <code>P</code> in <code>scripts/ur5_calibration_mount.py</code>.</li>
<li>Seat creep: tighten one M6 to 1.5&#8202;N&#8202;m on the printed adapter, mark the head's height and measure it again after a day;
a head that sinks more than 0.1{MM} says the six walls under it are too few.</li>
<li>Homing repeatability and the HEX-E's x-axis check, still open from v2, on the raised mount.</li>
</ul>"""))

    S.append(("limits", "Limitations", f"""
<p>The HEX-E's tool-side thread depth comes from no drawing; the data sheets give 8{MM} on the UR5e's flange and 9{MM} on the Quick
Changer's tool plate. The key model is a cylinder over the hex's corners, and the tilt a ball-end key allows depends on its
maker. The torque the heads take on PLA, the print's fit on the dowel and the bridges over the pockets are untested.</p>"""))

    files = [D / f"{PFX}_parts.json", D / f"{PFX}_step.json", D / f"{PFX}_check.json", D / f"{PFX}_screws.json",
             D / f"{PFX}_payload.json", D / "cad", D / "print", D / "media", D / "README.md", ROOT / "scripts/ur5_calibration_mount.py",
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
               f"{tilt:.0f} deg; everything above the foot rose {dz:.0f} mm and the DIP links and indenters fit as in v2.")
    out = docs_index.tag(out, topic="fingertip", summary=summary, date="2026-10-21")
    out = retro_style.apply(out)
    PAGE.write_text(out)
    print(f"page -> {PAGE.relative_to(ROOT)} ({PAGE.stat().st_size / 1e6:.2f} MB); lede {len(lede.split())} words")


if __name__ == "__main__":
    main()
