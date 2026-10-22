#!/usr/bin/env python3
"""Result page of the UR5e calibration mount (hand-off docs/handoff/20261018-ur5_calibration_mount.md, step 4).

Reads the study's own files in docs/experiments/20261018-ur5_calibration_mount/ (interfaces, parts, STEP, checks,
estimates, print projects, assembly; built by scripts/ur5_calibration_mount.py) and the figures in its media/ and print/,
inlines the figures as data URIs, renders the mathematics with scripts/texsvg.py, applies scripts/retro_style.py and writes
docs/experiments/20261018-ur5_calibration_mount/20261018-ur5_calibration_mount.html.

    .venv/bin/python scripts/ur5_calibration_mount_page.py
"""
from __future__ import annotations

import base64
import html
import io
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = ROOT / "docs/experiments/20261018-ur5_calibration_mount"
PFX = "20261018-ur5_calibration_mount"
PAGE = D / f"{PFX}.html"
TEMPLATE = ROOT / "scripts/ur5_calibration_mount_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
DIP_PAGE = ROOT / "docs/experiments/20261018-dip_link_board_mount/20261018-dip_link_magnetometer_board_mount.html"
PROC_PAGE = ROOT / "docs/experiments/20261018-fingertip_print_procedure/20261018-siraya85a_fingertip_print_procedure.html"
STATUS_PAGE = ROOT / "docs/experiments/20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html"
READY_PAGE = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness/20261016-cad_hand_fingertip_deployment_readiness.html"
MLX_PAGE = ROOT / "docs/experiments/20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html"


def J(name):
    return json.loads((D / f"{PFX}_{name}.json").read_text())


def esc(s):
    return html.escape(str(s), quote=True)


def relp(p):
    import os
    return os.path.relpath(p, D)


def img_uri(png, width=1100, quality=86):
    from PIL import Image
    im = Image.open(png).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def figure(png, caption, n, width=1100):
    alt = re.sub(r"\$[^$]*\$", "", caption.split(".")[0])
    return (f'<figure id="fig{n}"><img src="{img_uri(png, width)}" alt="{esc(alt)}">'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def table(head, rows, num_cols=(), caption=None, n=None):
    cap = f"<caption>Table {n}. {caption}</caption>" if caption else ""
    th = "".join(f'<th{" class=n" if k in num_cols else ""}>{h}</th>' for k, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td{" class=n" if k in num_cols else ""}>{c}</td>' for k, c in enumerate(r)) + "</tr>"
                 for r in rows)
    return f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def f2(v, nd=2):
    return f"{v:.{nd}f}"


def build():
    itf, prt, stp, chk, est, asm = J("interfaces"), J("parts"), J("step"), J("check"), J("estimate"), J("assembly")
    prn = {n: json.loads((D / f"print/{PFX}_{n}_pla_t3.json").read_text()) for n in ("mount", "indenters")}
    P = prt["parameters"]
    M = D / "media"
    da, db, do = chk["dip_links"]["a"], chk["dip_links"]["b"], chk["dip_links"]["orig"]
    ea, eb, eo = est["deflection"]["a"], est["deflection"]["b"], est["deflection"]["orig"]
    pay = est["payload"]
    mass = est["masses_g"]
    fld = est["field"]
    mount_g = round(mass["adapter"] + mass["standoff_fork"] + mass["clamp_plate"])
    link_lo = min(ea["dip_link_parts_bonded"]["normal_10N_mm"], eb["dip_link_parts_bonded"]["normal_10N_mm"])
    link_hi = max(ea["dip_link_parts_separate"]["normal_10N_mm"], eb["dip_link_parts_separate"]["normal_10N_mm"])
    tot_lo = ea["mount_to_clamp"]["normal_10N_mm"] + ea["dip_link_parts_bonded"]["normal_10N_mm"]
    tot_hi = eb["mount_to_clamp"]["normal_10N_mm"] + max(eb["dip_link_parts_separate"]["normal_10N_mm"],
                                                         ea["dip_link_parts_separate"]["normal_10N_mm"])
    pt = {n: v["slice"]["summary"] for n, v in prn.items()}
    figs = {}
    n_fig = [0]

    def fig(stem, caption, width=1100, sub=M):
        n_fig[0] += 1
        figs[stem] = n_fig[0]
        return figure(sub / f"{PFX}_{stem}.png", caption, n_fig[0], width)

    n_tab = [0]

    def tab(head, rows, caption, num_cols=()):
        n_tab[0] += 1
        return table(head, rows, num_cols=num_cols, caption=caption, n=n_tab[0])

    lede = (f"Three PLA parts clamp a DIP link as printed (the CAD hand's, or v2 a or b of the DIP study, which keep its clevis arms "
            f"within 0.01&#8202;mm) to the tool plate of an OnRobot HEX-E QC on the UR5e, so that the five magnetometers can be "
            f"calibrated against the HEX-E. The finger lies along the tool axis, which puts the normal force on the HEX-E's "
            f"lateral axes (noise 0.035&#8202;N against 0.15&#8202;N along the tool axis); the nearest chip sits "
            f"{da['nearest_chip_from_flange_mm']:.0f}&#8202;mm from the flange face. Under 10&#8202;N at the contact point the mount "
            f"deflects {ea['mount_to_clamp']['normal_10N_mm']:.3f}&#8202;mm, mount and link together {tot_lo:.3f}-{tot_hi:.3f}&#8202;mm, "
            f"and the HEX-E {ea['hex_e']['normal_10N_mm']:.2f}&#8202;mm. The parts weigh {mount_g}&#8202;g; with the HEX-E the "
            f"UR5e's payload is {pay['total_kg']:.2f}&#8202;kg.")

    S = []
    # ------------------------------------------------------------------ terms
    S.append(("terms", "Terms and metrics", f"""
<dl class="terms">
<dt>Finger frame F</dt><dd>The DIP study's frame, mm: $x_\\mathrm{{F}}$ toward the dorsal side, $y_\\mathrm{{F}}$ along the PIP axis,
$z_\\mathrm{{F}}$ up; the PIP axis passes through (0,&#8202;$y$,&#8202;&#8722;51.55) and the straight finger hangs along $-z_\\mathrm{{F}}$. The horn-side
clevis arm lies at $y_\\mathrm{{F}}$ from &#8722;23.25 to &#8722;17.05, the slotted arm from 13.40 to 17.40.</dd>
<dt>Mount frame M</dt><dd>Origin at the centre of the flange face (the ISO 9409-1-50-4-M6 face that the adapter bolts to), mm:
$z_\\mathrm{{M}}$ along the tool axis toward the tip, $x_\\mathrm{{M}}$ toward the palmar side (the direction of the contact normal),
$y_\\mathrm{{M}}$ along the PIP axis toward the slotted arm; $x_\\mathrm{{M}} = -x_\\mathrm{{F}}$, $y_\\mathrm{{M}} = y_\\mathrm{{F}} + 3.85$,
$z_\\mathrm{{M}} = -z_\\mathrm{{F}} - 5.55$. The PIP axis lies at $z_\\mathrm{{M}} = 46$. The STEP files are in M.</dd>
<dt>Contact point</dt><dd>The point of the tip's palmar face over the centre chip U1, $x_\\mathrm{{F}} = -10.55$, $y_\\mathrm{{F}} = 0$;
$z_\\mathrm{{M}}$ = {da['contact_point_M'][2]:.1f}&#8202;mm for v2 a, {db['contact_point_M'][2]:.1f} for v2 b and
{do['contact_point_M'][2]:.1f} for the CAD hand's link.</dd>
<dt>Design loads</dt><dd>10&#8202;N normal at the contact point (into the tip, along $-x_\\mathrm{{M}}$) and 5&#8202;N tangential, across the
finger ($y$) or along it ($z$).</dd>
<dt>Deflection</dt><dd>Displacement of the contact point relative to the flange face under a design load, mm, from Euler-Bernoulli
bending and shear over the sections of the meshes (Section 5); the HEX-E's own deflection is given separately from its data
sheet.</dd>
<dt>Force resolution of the tip</dt><dd>Cram&#233;r-Rao bound (one standard deviation, N) of the normal force at the contact point from
one 141&#8202;Hz sample of the five chips, contact position and a uniform background field unknown (DIP study): 0.063&#8202;N with five
magnets to 0.171&#8202;N with four.</dd>
<dt>HEX-E signal noise</dt><dd>Standard deviation of a one-second no-load signal (OnRobot data sheet); noise-free resolution, the
data sheet's second figure, is the step it resolves without averaging.</dd>
<dt>Clearance, overlap</dt><dd>Least distance between a mount part and a DIP link (mm), and the volume they share (mm&#179;), from the
meshes.</dd>
<dt>Payload</dt><dd>Mass (kg) and centre of gravity (mm from the UR5e flange) that the UR5e is told with <code>set_payload</code>.</dd>
</dl>"""))
    # ------------------------------------------------------------------ interfaces
    fl = itf["flange"]
    u, q, h = fl["ur5e"], fl["qc_tool_side"], fl["hex_e_qc"]
    rows = [["Outer diameter", "63&#8202;mm (H8 as drawn)", "63&#8202;mm h8", "63&#8202;mm"],
            ["Bolts", "4 &#215; M6-6H on 50&#8202;mm, 8&#8202;mm deep", "4 &#215; M6 on 50&#8202;mm, 9&#8202;mm deep",
             "4 &#215; M6 &#215; 12 + 1.6&#8202;mm washer through 3.0&#8202;mm of PLA: 7.4&#8202;mm in the flange"],
            ["Centring recess", "31.50 H7, 6.20&#8202;mm deep", "31.50 H7, 7.10&#8202;mm deep", "spigot 31.40 &#215; 5.5&#8202;mm"],
            ["Dowel hole", "6 H7, 6.20&#8202;mm deep, 45&#176; from the bolts", "6 H7, 6.10&#8202;mm deep", "printed dowel 5.95 &#215; 5.0&#8202;mm"],
            ["Height in the stack", "&#8212;", "15.15&#8202;mm (rim to tool face)", "HEX-E QC 50&#8202;mm + tool plate 15.15&#8202;mm"],
            ["Source", "UR5e technical specification, Nov 2023", "OnRobot Quick Changers data sheet v2.0, 1.7",
             "this page"]]
    S.append(("interfaces", "Interfaces of the mount", f"""
<p>The adapter disc fits three faces with one geometry: the UR5e's tool flange, the ISO face of OnRobot's Quick Changer tool plate,
which couples to the HEX-E QC, and the ISO tool flange of a HEX-E without the quick changer. The HEX-E QC adds 50&#8202;mm between the
UR5e flange and the quick changer's coupling plane and the tool plate 15.15&#8202;mm more, so the mount's flange face lies 65.15&#8202;mm
from the UR5e flange. The HEX-E's capacity is 200&#8202;N in force and 10&#8202;N&#8202;m in lateral torque; it weighs 0.347&#8202;kg.</p>
{tab(["Feature", "UR5e tool flange", "Quick Changer tool plate", "Adapter (this mount)"], rows,
     "Flange-side interfaces; UR5e and OnRobot values as drawn in their data sheets.")}
<p>The DIP link's PIP joint, measured from <code>fingertip_mount_DIP_link.stl</code> and identical within 0.01&#8202;mm on v2 a and b:
the horn-side arm is three layers, an outer layer of 2.0&#8202;mm with the 2.3&#8202;mm centre hole on the PIP axis (1.6&#8202;mm thick within
3.2&#8202;mm of the axis), the horn slot of 2.0&#8202;mm, and an inner layer of 2.2&#8202;mm whose top is an arc of radius 3.2&#8202;mm about the
axis, where the servo's output boss enters. The slot has the shape of the SCS0009 horn's arm: open toward the PIP axis above
$z_\\mathrm{{F}}$ = &#8722;53.5, 2.92&#8202;mm half-width at &#8722;54 tapering to 2.50&#8202;mm, and a round end of radius 2.5&#8202;mm with its bottom at
&#8722;68.55. Four 1.6&#8202;mm holes pass through all three layers at 6.5 to 14.5&#8202;mm from the axis. The slotted arm, 4.0&#8202;mm thick, has
a 5.2&#8202;mm bore on the axis with a 2.0&#8202;mm slot running 5.4&#8202;mm toward the tip. The arms are 30.45&#8202;mm apart; the bridge between
them begins at a radius of 12.75&#8202;mm about the PIP axis on the CAD hand's link, 13.35 on v2 a and 13.45 on v2 b. On the hand the
horn's centre screw passes the 2.3&#8202;mm hole, the horn sits in the slot on the servo's boss, and a 5&#8202;mm shaft in the MCP link's
MR105 bearing passes the 5.2&#8202;mm bore. Every value with its source is in <code>{PFX}_interfaces.json</code>.</p>"""))
    # ------------------------------------------------------------------ design
    S.append(("design", "Design", f"""
<p>The mount is three printed parts and eleven screws (Figures 1-5). The <b>adapter disc</b> has the spigot and dowel of the
flange interface on its robot side and four M6 &#215; 12 in counterbores. The <b>standoff-fork</b> is one part: a foot bolted to the
adapter by four M3 &#215; 12, a 34 &#215; 34&#8202;mm column, and a fork in the place of the MCP link and the PIP servo. The fork's inner
block fills the 30.45&#8202;mm between the clevis arms, 0.10&#8202;mm from each, down to a radius of 12.5&#8202;mm about the PIP axis, which
clears the bridge of all three links by 0.25 to 0.95&#8202;mm. Beside the inner block a hub ends 0.3&#8202;mm short of the horn slot's walls,
and a 1.6&#8202;mm key, the horn's thickness, hangs from it into the slot with 0.10&#8202;mm on each flank and a round end of radius
2.4&#8202;mm in the slot's 2.5&#8202;mm end. Two ears beyond the link's width take the clamp screws. The <b>clamp plate</b> presses the
horn-side arm's outer face onto the inner block with two M3 &#215; 16 into the ears, and an M3 &#215; 12 with a DIN 9021 washer
through the slotted arm's 5.2&#8202;mm bore pulls that arm onto the inner block's other face. All M3 screws form their thread in
2.75&#8202;mm holes of the PLA, the owner's size.</p>
{fig("on_the_hex_e", "The mount with DIP link v2 a, its board and the five-magnet 11/22 tip on the HEX-E QC, finger pointing down, seen from the dorsal and horn side. The UR5e flange, HEX-E and tool plate are cylinders of their data-sheet sizes.")}
<p>The finger lies along the tool axis, the MCP link's direction at PIP 0. A normal force on the tip then loads the HEX-E across
its axis, where its signal noise is 0.035&#8202;N, against 0.15&#8202;N along the axis; with the finger across the tool axis the normal
force would fall on the noisier axis. The PIP axis lies 46&#8202;mm from the flange face, which puts v2 a's board 62.0&#8202;mm and its
nearest chip {da['nearest_chip_from_flange_mm']:.1f}&#8202;mm from the flange face (v2 b: 64.3 and {db['nearest_chip_from_flange_mm']:.1f}&#8202;mm),
(127 and 139&#8202;mm from the UR5e flange).</p>
<p>The horn-side arm is the datum in $y$. Its three layers are 6.2&#8202;mm of PLA joined to the bridge 9&#8202;mm below the clamp. They are too
stiff to bend onto the inner block (about 4&#8202;N per micrometre), so the clamp plate pushes the whole link 0.10&#8202;mm toward the slotted
side until the inner layer rests on the inner block. The slotted arm, 4.0&#8202;mm thick and 12.5&#8202;mm long, bends that far for about
24&#8202;N of its screw's pull. The key and the hub sit 0.15&#8202;mm or more from the outer layer after that shift. The link seats along
the tool axis when the slot's round end meets the key's end; the hub then stays 0.2&#8202;mm clear of the slot walls. Tightened with
the tip pushed toward its back, the key bears on the slot's palmar flank, the flank that the normal force loads, so that a normal
press meets no play even if the clamps slip.</p>
{fig("fork_closeup", "The fork with DIP link v2 a, dorsal and horn side: the clamp plate over the horn-side arm, the cable slot and groove, a zip-tie pocket, J1 with the Qwiic plug, and the board under the link's own clamp plate.")}
<p>The Qwiic cable leaves J1 toward the PIP joint at $x_\\mathrm{{F}}$ = 5.5-6.3&#8202;mm on v2 a and 2.0-2.8&#8202;mm on v2 b, both
within $|y_\\mathrm{{F}}| \\le 2$&#8202;mm. A slot 5.5&#8202;mm wide through the inner block, open on the dorsal side, takes it from either
link; the slot continues as a groove down the column's dorsal face and leaves through a notch in the foot. Two cable ties pass
through tunnels under the groove, 1.4&#8202;mm below its floor, and anchor the cable to the column, so a pull on the cable ends at the
column and leaves J1 unloaded.</p>
{fig("fork_alone", "The standoff-fork without the DIP link, its fork end up: the key, the hub, the inner block between the arms, the ears with the clamp-screw holes, the cable slot and the groove with its tie pockets.")}
{fig("sections", "Sections through the clevis of the assembled mount with DIP link v2 a (grey) and the fork (blue). Left, the plane $x_\\mathrm{{F}} = 0$: the hub, the key in the horn slot, the inner block between the arms, the clamp plate on the horn-side arm and the M3 hole for the slotted arm. Right, the plane $y_\\mathrm{{F}} = -20.2$ through the slot: the key 0.10 mm from each flank, its round end in the slot's, and the ears with the clamp-screw holes.", 1200)}
{fig("exploded", "Exploded along the tool axis: the adapter with its spigot and dowel up, the standoff-fork, the DIP link sliding on, and the clamp plate.")}"""))
    # ------------------------------------------------------------------ checks
    rows = []
    for k, lab in (("orig", "CAD hand's link"), ("a", "v2 a"), ("b", "v2 b")):
        c = chk["dip_links"][k]
        rows.append([lab, f2(c["nominal"]["overlap_fork_mm3"], 3), f2(c["assembled"]["overlap_fork_mm3"], 3),
                     f2(c["nominal"]["least_gap_mm"], 3), "clear" if c["insertion_clear"] else
                     f"{len(c['insertion_overlaps'])} positions", f2(c["nearest_chip_from_flange_mm"], 1),
                     f2(c["board_proximal_edge_from_flange_mm"], 1)])
    w = chk["walls_mm"]
    S.append(("checks", "Clearances and walls", f"""
<p>None of the three links overlaps the fork or the clamp plate, at the nominal pose or assembled (shifted 0.10&#8202;mm in $y$ onto the
datum and 0.10&#8202;mm along the tool axis onto the key's end). None touches the fork anywhere on the 30&#8202;mm slide onto it, checked
in 0.25&#8202;mm steps (Table 2). A hex key reaches each of the foot's four M3 straight from above, and the clamp screws pass 0.5&#8202;mm or
more from every link. The thinnest walls are the key at {w['key_thickness']}&#8202;mm, the ears at {w['ear_wall_inner']}&#8202;mm beside
the 2.75&#8202;mm holes, and the floor over the tie tunnels at {w['tie_floor']}&#8202;mm. The board's nearest chip lies 65.6&#8202;mm from the
flange face on the CAD hand's link and its tab 53.5&#8202;mm; on the v2 links, which the owner prints, the board is 62&#8202;mm or more
away.</p>
{tab(["DIP link", "Overlap (mm&#179;)", "Overlap, assembled", "Least gap (mm)", "Slide-on", "Nearest chip (mm)", "Board edge (mm)"], rows,
     "Fit of the three DIP links in the fork, from the meshes (<code>" + PFX + "_check.json</code>): overlap with the fork at the "
     "nominal and the assembled pose, least gap at the nominal pose, the 30 mm slide onto the fork, and the nearest chip and the "
     "board's nearest edge from the flange face.", num_cols=(1, 2, 3, 5, 6))}"""))
    # ------------------------------------------------------------------ deflection and payload
    rows = []
    for k, lab in (("orig", "CAD hand's link"), ("a", "v2 a"), ("b", "v2 b")):
        e = est["deflection"][k]
        rows.append([lab, f2(e["lever_flange_to_contact_mm"], 1), f2(e["mount_to_clamp"]["normal_10N_mm"], 4),
                     f2(e["mount_to_clamp"]["tangential_5N_y_mm"], 4),
                     f"{e['dip_link_parts_bonded']['normal_10N_mm']:.3f}-{e['dip_link_parts_separate']['normal_10N_mm']:.3f}",
                     f2(e["hex_e"]["normal_10N_mm"], 3), f2(e["hex_e"]["tangential_5N_y_mm"], 3)])
    names = {"adapter": "Adapter disc", "standoff_fork": "Standoff-fork", "clamp_plate": "Clamp plate",
             "Quick Changer tool side": "Quick Changer tool plate"}
    prow = [[esc(names.get(i["item"], i["item"])), f2(i["mass_g"], 1), ", ".join(f"{v + 0.0:.1f}" for v in i["cog_M_mm"])]
            for i in pay["items"]]
    cg = pay["cog_from_ur_flange_mm"]
    S.append(("deflection", "Deflection and payload", f"""
<p>The mount moves the contact point {ea['mount_to_clamp']['normal_10N_mm']:.3f}&#8202;mm under the 10&#8202;N normal load and
{ea['mount_to_clamp']['tangential_5N_y_mm']:.3f}&#8202;mm under 5&#8202;N across the finger. The DIP link with its board and clamp plate or
cover adds {link_lo:.3f}-{link_hi:.3f}&#8202;mm, the range between its parts acting as one section and as separate beams (Table 3). Both
lie under the 0.1&#8202;mm the hand-off set. The HEX-E deflects 0.49&#8202;mm under the same normal load, because a force sensor measures through
its own strain. Its data sheet gives 1.7&#8202;mm at 200&#8202;N and 2.5&#176; at 10&#8202;N&#8202;m. The contact point therefore moves about
0.05&#8202;mm per newton toward the robot during a press, 0.15&#8202;mm at the protocol's 3&#8202;N.</p>
<p>The estimate integrates bending and shear along the tool axis over the sections of the meshes, cut every 0.5&#8202;mm from the foot
to the contact point at $z_c$, with PLA's modulus $E$ = 2.5&#8202;GPa (100&#8202;% printed PLA, the low end of 2.5-3.5&#8202;GPa), $G = E/2.7$
and a shear area $A_s = 0.83\\,A$:
$$\\delta = F \\int_{{z_0}}^{{z_c}} \\frac{{(z_c - z)^2}}{{E\\,I(z)}}\\,dz + F \\int_{{z_0}}^{{z_c}} \\frac{{dz}}{{G\\,A_s(z)}}$$
where $I(z)$ is the second moment of the section about the axis across the load. In the clamp zone the fork, the arms and the clamp
plate count as one section; that assumes the clamps do not slip, which needs a friction torque above the
10&#8202;N &#215; {ea['lever_flange_to_contact_mm'] - 46:.1f}&#8202;mm = {(ea['lever_flange_to_contact_mm'] - 46) * 0.01:.2f}&#8202;N&#8202;m
about the PIP axis for v2 a; the key takes it if they do. The HEX-E's part, linear in its data-sheet deformations at capacity:
$$\\delta_\\mathrm{{HEX}} = F\\,(C_f + C_t L^2), \\qquad C_f = \\frac{{1.7\\ \\mathrm{{mm}}}}{{200\\ \\mathrm{{N}}}}, \\quad
C_t = \\frac{{2.5^\\circ}}{{10\\ \\mathrm{{N\\,m}}}}$$
with $L$ the distance from the HEX-E's tool plane to the contact point, {ea['hex_e']['lever_mm']:.1f}&#8202;mm for v2 a.</p>
{tab(["DIP link", "Lever (mm)", "Mount, 10 N (mm)", "Mount, 5 N (mm)", "Link, 10 N (mm)", "HEX-E, 10 N (mm)",
      "HEX-E, 5 N (mm)"], rows,
     "Deflection of the contact point under 10 N normal and 5 N across the finger (<code>" + PFX + "_estimate.json</code>): "
     "lever from the flange face to the contact point; the mount up to its clamps; the DIP link with its board and clamp plate "
     "or cover, as one section and as separate beams; the HEX-E from its data sheet. The tip's own compression is the indentation "
     "the board measures and is not included.",
     num_cols=(1, 2, 3, 4, 5, 6))}
<p>The printed parts weigh {mount_g}&#8202;g solid, the adapter {mass['adapter']:.0f}&#8202;g, the standoff-fork {mass['standoff_fork']:.0f}&#8202;g
and the clamp plate {mass['clamp_plate']:.0f}&#8202;g; the print project prints them at 40&#8202;% gyroid outside the fork and uses
{pt['mount']['filament_g']:.0f}&#8202;g of filament. With the HEX-E QC, the tool plate, screws and DIP link v2 a with its tip, the payload
is {pay['total_kg']:.3f}&#8202;kg. Its centre of gravity is {cg[2]:.0f}&#8202;mm from the flange along the tool axis and
{abs(cg[1]):.1f}&#8202;mm off it (Table 4). The payload is 13&#8202;% of the UR5e's 5&#8202;kg payload.</p>
{tab(["Item", "Mass (g)", "Centre of gravity in M (mm)"], prow, "Payload for <code>set_payload</code>: "
     f"{pay['total_kg']:.3f}&#8202;kg, centre of gravity ({cg[0]:.1f}, {cg[1]:.1f}, {cg[2]:.1f})&#8202;mm from the UR5e flange.",
     num_cols=(1,))}"""))
    # ------------------------------------------------------------------ assembly and print
    pl = [[esc(i["item"]), str(i.get("qty", "")), esc(i.get("material", "")),
           f"<code>{esc(Path(i['file']).name)}</code>" if i.get("file") else ""] for i in asm["parts_list"]]
    steps = "".join(f"<li>{esc(s_)}</li>" for s_ in asm["assembly"])
    S.append(("assembly", "Parts, print and assembly", f"""
{tab(["Item", "Qty", "Material", "STEP"], pl, "Parts list (<code>" + PFX + "_assembly.json</code>); STEP files in <code>cad/</code>.")}
<p>The three parts print on one plate of the Prusa XL in PLA on tool 3 at 0.15&#8202;mm layers, without supports. The adapter lies on its
tool face, so the mating face and the spigot are top surfaces. The standoff-fork stands on its foot with the fork end up. The clamp
plate lies on the face that presses the arm (Figure {n_fig[0] + 1}). PrusaSlicer estimates {pt['mount']['print_time']} and
{pt['mount']['filament_g']:.1f}&#8202;g; the two indenters take {pt['indenters']['print_time']} and
{pt['indenters']['filament_g']:.1f}&#8202;g on a second plate. Every part slices to its full height and no support is generated.</p>
{fig("mount_pla_t3_bed", "The mount's plate on the Prusa XL bed, PLA on tool 3.", 1000, D / "print")}
<ol>{steps}</ol>
{fig("indenters", "The optional table indenters: a 5 mm sphere for point presses and a half rod of radius 12.5 mm, the bench screwdriver's, each on a base with slots for two M6 bolts 50 mm apart.", 900)}"""))
    # ------------------------------------------------------------------ protocol
    fa, fb = fld["a"]["sources"], fld["a"]
    rows_f = [["Brake solenoid of wrist 3", "0.01-0.06", f"{fa['brake_solenoid']['B_nearest_chip_uT'][0]:.2f}-{fa['brake_solenoid']['B_nearest_chip_uT'][1]:.2f}",
               f"{fa['brake_solenoid']['spread_across_chips_uT'][0]:.2f}-{fa['brake_solenoid']['spread_across_chips_uT'][1]:.2f}"],
              ["Residual dipole of wrist 3's rotor", "0.005-0.03", f"{fa['rotor_residual']['B_nearest_chip_uT'][0]:.2f}-{fa['rotor_residual']['B_nearest_chip_uT'][1]:.2f}",
               f"{fa['rotor_residual']['spread_across_chips_uT'][0]:.2f}-{fa['rotor_residual']['spread_across_chips_uT'][1]:.2f}"]]
    rows_r = [["HEX-E, $F_x$ and $F_y$ (the normal and across)", "0.035", "0.2"],
              ["HEX-E, $F_z$ (along the finger)", "0.15", "0.8"],
              ["Tip, five chips, one 141&#8202;Hz sample", "0.063 (five magnets) to 0.171 (four)", "&#8212;"]]
    S.append(("protocol", "Calibration protocol", f"""
<p>The board is calibrated on the robot against the HEX-E, with the robot pressing the tip on a rigid indenter bolted to the table.
Within one calibration set the tool keeps one orientation, so Earth's field, about 50&#8202;&#181;T against the chips' 1.15-1.75&#8202;&#181;T
of noise at 141&#8202;Hz, stays a constant offset; a new orientation starts a new set with its own background.</p>
{tab(["Reference or sensor", "Signal noise, 1&#8202;s at no load (N)", "Noise-free resolution (N)"], rows_r,
     "Force resolution of the HEX-E (OnRobot data sheet v1.5) and of the tip (DIP study).", num_cols=())}
<p>The UR5e's wrist 3 contains the motor and the solenoid that keeps its brake released while the arm is powered, about 60&#8202;mm
behind the flange face, so the chips lie {fb['r_nearest_chip_mm']:.0f}-{fb['r_farthest_chip_mm']:.0f}&#8202;mm from them. As point dipoles
on the tool axis,
$$B = \\frac{{\\mu_0\\, m}}{{2\\pi r^3}}$$
with a brake moment $m$ of 0.01-0.06&#8202;A&#8202;m&#178; (50-120 ampere-turns over about 1&#8202;cm&#178;, its iron raising the air-core moment
two to five times) and a residual rotor moment of 0.005-0.03&#8202;A&#8202;m&#178; (2-5&#8202;% of a multipole ring's magnets), they give at most
{fa['brake_solenoid']['B_nearest_chip_uT'][1]:.1f}&#8202;&#181;T at the nearest chip and at most
{fa['brake_solenoid']['spread_across_chips_uT'][1]:.2f}&#8202;&#181;T of difference across the chips (Table {n_tab[0] + 1}). The readout estimates a
uniform background with the force, so only the difference across the chips enters the force, and it is below the chips' noise.
Without the HEX-E's 65&#8202;mm the brake would give up to {fa['mount_on_the_ur_flange_brake_high']:.1f}&#8202;&#181;T. The rotor turns with
every move of wrist 3, so its part of the field changes with the pose; step 5 measures it.</p>
{tab(["Source", "Dipole moment (A&#8202;m&#178;)", "Field at the nearest chip (&#181;T)", "Difference across the chips (&#181;T)"], rows_f,
     "UR5e wrist field at DIP link v2 a's chips, estimated (assumptions in the text).", num_cols=(1, 2, 3))}
<ol>
<li><b>Mount.</b> HEX-E QC on the UR5e, the mount on its tool plate, the DIP link with its board, tip and Qwiic cable in the fork
(Section 6). Declare the payload of Table 4; set the TCP at the contact point. Keep the arm powered and its brakes released for the
whole set.</li>
<li><b>Orientation.</b> Tool axis horizontal, palmar face down, so the indenter presses along the HEX-E's $x$ or $y$ axis (read them
off the housing's marks and record the angle to the finger's normal).</li>
<li><b>HEX-E.</b> At that orientation, no contact: zero the HEX-E (it removes the tool's weight), then log 60&#8202;s at no load for its
noise and drift. Hang 50, 100, 200 and 300&#8202;g (0.49-2.94&#8202;N) from the contact point to check its gain over the calibration's
range.</li>
<li><b>Board.</b> Five chips at the gain the tip's static field allows (v2 a with the five-magnet 11/22 tip: GAIN_SEL 3, 11.4&#8202;mT at
U1; with the five-magnet block: GAIN_SEL 5, 7.6&#8202;mT), RES 0, OSR 0, DIG_FILT 5: 141&#8202;Hz. Log the chips and the HEX-E's wrench
on one clock with the robot's TCP pose.</li>
<li><b>No-contact sweep.</b> Visit every press pose of steps 7-8 with the tip 5&#8202;mm off the indenter, 2&#8202;s at each: the background per
chip and pose. A pose-dependent change above the chips' noise is the wrist rotor's and is subtracted per pose.</li>
<li><b>Rigid control.</b> A solid PLA dummy of the tip's outline, pressed to 3&#8202;N with a 60&#8202;s hold and released: the hysteresis and
drift of the HEX-E, mount and robot, which the tip's numbers are read against.</li>
<li><b>Normal presses.</b> The 5&#8202;mm sphere presses at 15 contact points, 2&#8202;mm apart along the finger (&#8722;4 to 4&#8202;mm from the contact
point) and 3&#8202;mm across (&#8722;3, 0, 3). Each point is approached along the local normal at 0.5&#8202;mm/s until the HEX-E reads 0.05&#8202;N. The load
then goes to 0.25, 0.5, 1, 2 and 3&#8202;N for 2&#8202;s each and returns in the same steps. There are three repeats.</li>
<li><b>Tangential loads.</b> At the contact point and the four points 3&#8202;mm from it, at 1, 2 and 3&#8202;N normal, move the indenter across
and along the finger until the tangential force reaches 3&#8202;N or stops rising (slip).</li>
<li><b>Hysteresis and creep.</b> At the contact point, three loops each to 0.5, 1 and 3&#8202;N at 0.1&#8202;N/s, holding the robot still for
60&#8202;s at the peak and 60&#8202;s at zero. During a hold the 85A tip relaxes and the HEX-E's force falls; the board's reading against the
HEX-E over the hold gives its creep error in newtons, and the difference between loading and unloading at equal force its
hysteresis. Both are compared with the tip's 0.063-0.171&#8202;N resolution and with the rigid control.</li>
</ol>
<p>A one-second mean of the HEX-E lowers its noise by the square root of its samples if the noise is white; step 3's log shows
whether it is. The HEX-E's deflection moves the contact point 0.05&#8202;mm per newton (Section 5), which the press positions are
corrected by.</p>"""))
    # ------------------------------------------------------------------ limitations, future, files
    S.append(("limits", "Limitations", """
<p>Nothing has been printed or measured. The deflections are beam estimates with one PLA modulus and assume the clamps hold by
friction; the fits (spigot, dowel, key, inner block) are nominal and the printer's tolerance decides them. The HEX-E's deflection is
its data sheet's typical value at capacity taken as linear, and its hysteresis and nonlinearity are given there as under 2&#8202;% of
capacity, which at 200&#8202;N exceeds the calibration's loads, so the rigid control measures them. The wrist's field is a dipole
estimate from assumed moments; the UR5e's motor and brake are not documented. The lab's HEX-E is taken to be the QC version; the
adapter fits a non-QC HEX-E's ISO flange as well.</p>"""))
    S.append(("future", "Future work", f"""
<ul>
<li>Print <code>print/{PFX}_mount_pla_t3.3mf</code> and fit the adapter in the tool plate before anything else: a spigot that does not
slide in by hand is rebuilt with <code>spigot_d</code> 0.05&#8202;mm smaller (<code>scripts/ur5_calibration_mount.py</code>, then
<code>build</code>, the STEP export and <code>print</code>).</li>
<li>Slide a printed v2 a or b onto the fork: if the key does not enter, <code>key_flank</code> goes from 0.10 to 0.15&#8202;mm; if the link
rocks after clamping, the inner block's 0.10&#8202;mm per side is the first dimension to reduce.</li>
<li>Measure the mount's stiffness on the robot: press the rigid dummy to 10&#8202;N and read the contact point's motion (the robot's
position against a dial indicator on the table); more than 0.1&#8202;mm beyond the HEX-E's 0.49&#8202;mm refutes the estimate of Section 5.</li>
<li>Run the no-contact sweep of step 5 with wrist 3 turned through 360&#176;: a field change above 1.5&#8202;&#181;T at a chip refutes the
wrist estimate.</li>
<li>Calibrate v2 a with the five-magnet 11/22 tip (steps 1-9) and fit the readout model's field per newton per chip to the
presses.</li>
</ul>"""))
    files = [f"<code>{esc(Path(v['file']).name)}</code> ({v['bytes'] // 1000}&#8202;kB)" for v in stp["parts"].values()]
    S.append(("files", "Files", f"""
<p>Folder <code>docs/experiments/20261018-ur5_calibration_mount/</code>, built by <code>scripts/ur5_calibration_mount.py</code> (runbook
<a href="README.md">README.md</a>) and this page by <code>scripts/ur5_calibration_mount_page.py</code>. STEP, exact solids in M that open
in Inventor with File &gt; Open, Options, Convert Model: {", ".join(files)}; STL beside each in <code>cad/</code>. The DIP study's STEP
files (frame D of <code>fingertip_mount_DIP_link.stl</code>) enter M by a translation of (0, &#8722;17.40, 46.0)&#8202;mm. Print projects
and their slice checks in <code>print/</code>. Data: <code>{PFX}_interfaces.json</code>, <code>_parts.json</code>, <code>_step.json</code>,
<code>_check.json</code>, <code>_estimate.json</code>, <code>_assembly.json</code>. Related pages: <a href="{relp(DIP_PAGE)}">the DIP link and its
board mount</a>, <a href="{relp(READY_PAGE)}">readiness of the printed fingertips</a>, <a href="{relp(MLX_PAGE)}">the MLX90393
readout</a>, <a href="{relp(PROC_PAGE)}">the 85A print procedure</a>, <a href="{relp(STATUS_PAGE)}">the status of the printed-fingertip
work</a>. Hand-off <code>docs/handoff/20261018-ur5_calibration_mount.md</code>.</p>"""))

    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    title = "Calibration mount for the instrumented DIP link on a UR5e and HEX-E"
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "UR5e calibration mount", "H1": title, "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-18",
                 "LEDE": lede, "BYLINE": f"Built {time.strftime('%Y-%m-%d %H:%M')} from the study's files; nothing printed or measured yet.",
                 "TOC": toc, "SECTIONS": secs,
                 "FOOTER": "docs/experiments/20261018-ur5_calibration_mount/20261018-ur5_calibration_mount.html",
                 "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None) for m in TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas, {n_fig[0]} figures", flush=True)


if __name__ == "__main__":
    build()
