#!/usr/bin/env python3
"""Builds docs/experiments/20261119-v4b_build/20261119-v4b_finger_build.html: the v4b index finger as laser-cut
plates, STS3032 servos, ball bearings and three printed parts, with its clearance check, PIP drive, cut sheet and bill
of materials (owner, 2026-11-19: "can you prepare a dxf of laid out parts to cut on acrylic, and prepare a BOM for this?
we have the sts3032s on hand already"; "yea let's go for v4b then, with emphasis on physical feasibility, minimal
slop/backlash, simplicity ideally, and non exotic/esoteric parts").

Data: the folder's data/20261119-v4b_{check_fine,pip,parts,printed}.json and data/20261119-v4b_bom.csv, written by
scripts/cad/v4b_build.py (check --fine, pip, dxf, printed, bom), and its media/ figures (dxf, pip, render).

    .venv/bin/python scripts/cad/v4b_build_page.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401
import docs_index  # noqa: E402
import fp_review_page as H  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import v4b_build as V  # noqa: E402

D = ROOT / "docs/experiments/20261119-v4b_build"
DATA, MEDIA = D / "data", D / "media"
OUT = D / "20261119-v4b_finger_build.html"
TEMPLATE = ROOT / "scripts/cad/cad_v3_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
LAYOUTS = "../20261118-sr2_actuator_layout/20261118-sr2_finger_actuator_layouts.html"
table, img, esc = H.table, H.img, H.esc
B, T = V.B, V.T


def load():
    J = {k: json.loads((DATA / f"20261119-v4b_{k}.json").read_text()) for k in ("check_fine", "pip", "parts", "printed")}
    with (DATA / "20261119-v4b_bom.csv").open() as fh:
        J["bom"] = list(csv.reader(fh))
    J["g"] = V.geometry()
    J["keepout"] = V.cmd_keepout()  # writes data/20261119-v4b_dip_keepout.stl
    return J


def mn(x, nd=1):
    """A number with a typographic minus."""
    return f"{x:.{nd}f}".replace("-", "&minus;")


NAMES = [("flange_L45", "PIP bearing's flange (L4, L5)"), ("flange_L2", "idler bearing's flange (L2)"),
         ("flange_H1", "pin a's bearing flange (H1)"), ("flange_R1", "pin b's bearing flange (R1)"),
         ("dip_cheek", "DIP link cheek"), ("dip_lugarm", "DIP link lug arm"), ("tip", "fingertip envelope"),
         ("sph_ring", "spherical bearing's ring"), ("pipdisc", "PIP servo's disc horn"), ("mcpdisc", "MCP servo's disc horn"),
         ("yawdisc", "yaw servo's disc horn"), ("idler_dowel", "idler dowel"), ("knuckle_boss", "knuckle boss"),
         ("knuckle", "knuckle"), ("pin_a_nut", "pin a's nut"), ("pin_b_nut", "pin b's nut"), ("pin_a_head", "pin a's head"),
         ("pin_a", "pin a's screw"), ("pin_b", "pin b's screw"), ("beam", "x-stage beam")]


def part_name(raw):
    """A readable name for a primitive of the clearance check."""
    nm = raw.split(":", 1)[1]
    for k, v in NAMES:
        if nm.startswith(k):
            return v
    m = re.match(r"(yaw|mcp|pip)servo_(case|tabs|boss)", nm)
    if m:
        return f"{m.group(1).upper() if m.group(1) != 'yaw' else 'yaw'} servo's {dict(case='case', tabs='tabs', boss='spline boss')[m.group(2)]}"
    m = re.match(r"(\w+) (.*?)( \([+-]y\)| [+-])?$", nm)  # the check's names before 11-20 end in " +" or " -"
    return f"{m.group(2)} {m.group(1)}" if m else nm


def designed(key, gap):
    """Gaps the parts are drawn with: a bearing's flange 0.3 mm from the face beside it, Y3 0.925 mm from L2."""
    return ("flange" in key and ("dip_cheek" in key or "dip_lugarm" in key) and abs(gap - 0.3) < 0.02) or \
           ("Y3" in key and "L2" in key and abs(gap - B["l2_y3_gap"]) < 0.02)


LEVEL = {"mount": 0, "beam": 0, "yoke": 1, "link": 2, "phorn": 3, "knuckle": 3, "rod": 3, "dip": 3}


def pose_cells(key, v):
    """The yaw, MCP and PIP of a pair's least gap; 'any' for a joint that does not move the two parts relative to each
    other (the mount and the beam stay fixed, the yoke turns with the yaw, the link with the MCP, the rest with the PIP)."""
    la, lb = sorted(LEVEL[p.split(":", 1)[0]] for p in key.split(" | "))
    dep = (la == 0 and lb >= 1, la <= 1 and lb >= 2, lb == 3)
    return [mn(x, 0) if d else "any" for x, d in zip(v[1:], dep)]


def gaps(J):
    least = [(k, v) for k, v in J["check_fine"]["least"]]
    beam = min((v for k, v in least if k.startswith("beam:")), key=lambda v: v[0])
    moving = [(k, v) for k, v in least if not k.startswith("beam:") and not designed(k, v[0])]
    return least, beam, moving


def sec_terms(J):
    g = J["g"]
    return f"""<dl class="terms">
<dt>Mount frame, gimbal centre G</dt><dd>The finger's frame on its mount, in mm: $x$ points to the dorsal side, $y$ to the
side and $z$ up toward the palm. Its origin is on v3's yaw axis. The yaw and MCP axes cross at G, {mn(-g['Gz'], 1)} mm
below v3's yaw axis; the PIP axis lies {B['proximal']:.0f} mm below G when the finger is straight.</dd>
<dt>Yaw, MCP and PIP angles</dt><dd>The joint angles in degrees, as in the simulator. The driver's ranges are yaw
&plusmn;85&deg;, MCP &minus;15&deg; to 92&deg; and PIP &minus;18&deg; to 92&deg;.</dd>
<dt>Gap</dt><dd>The least distance in mm between two parts over a grid of 9,828 poses. Yaw runs from &minus;85&deg; to
85&deg; in 15&deg; steps, MCP from &minus;15&deg; to 92&deg; in 3&deg; steps and PIP from &minus;25&deg; to 92&deg; in
6&deg; steps, with 85&deg; and 92&deg; added at the ends. Each plate is cut into 1 mm strips (boxes). Servos, horns, bearings, screws, the
knuckle and the DIP link are boxes or flat cylinders. Pairs that touch by design, such as a pin in its bearing, are left
out.</dd>
<dt>Designed gap</dt><dd>A gap drawn into the parts: 0.3 mm between a bearing's flange and the face beside it, and
{B['l2_y3_gap']:.3g} mm between the plates L2 and Y3.</dd>
<dt>Beam clearance</dt><dd>The least gap between the x-stage beam and the yoke or the MCP link. The beam's underside
lies {B['beam_z']:.1f} mm above v3's yaw axis between $y$ = {mn(B['beam_y'][0])} and {mn(B['beam_y'][1])} mm over the
whole carriage stroke.</dd>
<dt>Module half-width</dt><dd>The largest $|y|$ of the mount and the yoke over all yaw angles. The index and middle
mounts lie 56.8 mm apart, so the limit is {B['half_width']:.1f} mm.</dd>
<dt>PIP torque per SCS0009 limit</dt><dd>The PIP torque at the PIP servo's stall torque, divided by the stall torque of
the SCS0009, the current hand's PIP servo. By virtual work
$\\tau_{{\\mathrm{{PIP}}}} = \\tau_{{\\mathrm{{horn}}}}\\, d\\theta/d\\varphi$, with $\\theta$ the horn angle and
$\\varphi$ the PIP angle. The stall torques at 6 V are 0.441 N&nbsp;m (STS3032) and 0.226 N&nbsp;m (SCS0009).</dd>
<dt>Horn angle $\\theta$</dt><dd>The PIP servo's horn angle in degrees, from the horn arm's direction along $+y$.</dd>
<dt>Bearing tilt</dt><dd>The angle between the spherical bearing's outer ring and its inner ring, which sits on the lug's
dowel. The bearing's limit is 16&deg; before the ring's edge meets the clamping face.</dd>
<dt>Least web</dt><dd>The narrowest strip of plate between two cut contours of one part (outline, holes and slots), in
mm, from the contours drawn at 0.1 mm spacing.</dd>
<dt>Bore recess</dt><dd>The empty end of a bearing's 6 mm hole. The MF63ZZ is 2.5 mm wide with a 0.6 mm flange. With
the flange on one face of a {T:.3f} mm plate, {T - (B['mf63']['B'] - B['mf63']['flange_t']):.3f} mm of the hole on the
other face is empty. An M2 nut in it (4.6 mm across the corners) bears on the inner ring alone.</dd>
<dt>Precision shoulder screw</dt><dd>A screw with a ground, unthreaded 3 mm shoulder, 0 to 0.025 mm undersize. The
McMaster 90323A series used here has a 5 mm by 1.3 mm head and an M2 thread 4 mm long.</dd>
</dl>"""


def sec_layout(J):
    g = J["g"]
    fig = img(MEDIA / "20261119-v4b_assembly.png",
              "The assembly as the boxes and cylinders of the clearance check, at four poses, from the side (left) and from "
              "the palmar side (right). Grey: mount and yaw servo. Blue: yoke and MCP servo. Orange: MCP link, PIP servo and "
              "lower plates. Purple: horn arm H1. Yellow: knuckle. Red: rod R1. Green: DIP link and fingertip envelope.",
              jpeg=True)
    rows = [["Yaw", "the yaw servo's output", "none", "the servo; Y1 is screwed to its disc horn (4 M2&times;5)"],
            ["MCP", "the MCP servo's output; an MF63ZZ in L2, flange toward the servo", "3&times;8 dowel pressed into Y3",
             "the servo; L1 is screwed to its disc horn (4 M2&times;5)"],
            ["PIP", "MF63ZZs in L4 and L5, flanges on the plates' inner faces", "two 3&times;6 dowels pressed into the DIP link's cheeks",
             "4 mm bosses 0.3 mm tall on the cheeks bear on the two inner rings"],
            ["Pin a, horn arm to knuckle", "an MF63ZZ in H1, flange toward the knuckle",
             f"3&times;{J['printed']['pin_a']['shoulder']:.0f} shoulder screw from the knuckle's far face",
             "an M2 nut in H1's bore recess clamps the inner ring against the knuckle's boss"],
            ["Pin b, knuckle to rod", "an MF63ZZ in R1, flange toward the knuckle",
             f"3&times;{J['printed']['pin_b']['shoulder']:.0f} shoulder screw from a counterbore in the knuckle",
             "an M2 nut in R1's bore recess, as at pin a"],
            ["Lug, rod to DIP link", "the spherical plain bearing, bonded into R1", "3&times;12 dowel pressed through both cheeks",
             "4 mm bosses 1.3 mm tall on the cheeks' inner faces clamp the inner ring"]]
    tab = table(["Joint", "Turns in", "Pin", "Located along the pin by"], rows,
                "The joints and their pins. Every pin is steel and turns in a ball bearing or the spherical plain bearing.",
                num_from=9)
    ratio = J["pip"]["torque_scs0009_min_m25_92"] * 0.2256 / 0.4413
    return f"""<p>The finger is v4b of the actuator-layout study (<a href="{LAYOUTS}">its page</a>): a gimbal MCP with the
yaw and MCP axes crossing at G, and the PIP servo off the finger. The yaw servo stands on the mount behind G with its shaft
on the yaw axis. The yoke is screwed to that servo's disc horn: the back plate Y1, the MCP servo plate Y2 and the printed
idler plate Y3. The MCP servo stands in Y2 with its shaft on the MCP axis. The MCP link is screwed to that servo's disc
horn by its plate L1 and turns on an idler dowel in Y3 through a bearing in its plate L2. Below the MCP servo the cross
plate L3 joins L1 and L2 to the lower plates L4 and L5. The PIP servo lies flat between them, screwed by its tabs to the
strips L6a and L6b. Its horn arm H1 drives the rod R1 through the printed knuckle. The rod turns the DIP link at a lug
{B['pip']['lug_r']:.0f} mm from the PIP axis (Section SEC[pip]).</p>
{fig}
<p>Each joint turns in ball bearings on steel pins (Table TAB[joints]). The precision shoulders are 0 to 0.025 mm under
3 mm, and the MF63ZZ's bore is 0 to 0.008 mm under. A shoulder therefore has at most 0.025 mm of radial clearance before
its nut clamps the ring. The dowels are h8 or m6: an h8 dowel slides in a bearing with at most 0.014 mm of clearance, and an m6
dowel is a light press fit. No pin turns in acrylic or in a printed hole. Play at the PIP servo's horn reaches the PIP
divided by $d\\theta/d\\varphi$, which is at least {ratio:.2f} over the PIP's working range. One degree of play at the
horn therefore becomes at most {1 / ratio:.2f}&deg; at the PIP.</p>
{tab}"""


def sec_pip(J):
    P = J["pip"]
    fig = img(MEDIA / "20261119-v4b_pip_drive.png",
              "The PIP drive over its closing range. (a) PIP torque per SCS0009 limit; the dashed line is one SCS0009. (b) The "
              "horn angle and the spherical bearing's tilt; the dotted line is the bearing's 16&deg; limit. The grey band is the "
              "driver's PIP range.")
    lo, hi = P["range_deg"]
    return f"""<p>The PIP drive gives at least {P['torque_scs0009_min_m25_92']:.2f} times the current hand's PIP torque from
&minus;25&deg; to 92&deg;, and {P['torque_scs0009_median_m25_92']:.2f} times at the median. The linkage closes from
{mn(lo, 0)}&deg;, the end of the computed range, to {hi:.0f}&deg;. Past {hi:.0f}&deg; the rod no longer reaches the lug.
The PIP servo lies flat between the lower plates with its shaft along $x$. Its disc horn turns H1, which puts pin a
{B['pip']['r']:.0f} mm from the shaft. The knuckle turns on pin a and on pin b, {B['pip']['knuckle_e']:.0f} mm away and square
to it. The rod hangs from pin b {B['pip']['rod_offset_c']:.1f} mm to the side of the knuckle; it is {P['rod_mm']:.2f} mm long
between its bearings.</p>
{fig}
<p>Over the closing range the horn turns from {P['horn_deg_at_ends'][0]:.1f}&deg; to {mn(P['horn_deg_at_ends'][1])}&deg;,
inside its {B['pip']['horn_limit_deg']:.0f}&deg; limit. The spherical bearing tilts at most {P['bearing_tilt_deg_max']:.1f}&deg;.
The torque is least, {P['torque_scs0009_min_range']:.2f} times, at {P['curve']['pip'][int(V.np.argmin(P['curve']['torque']))]:.0f}&deg; of PIP
and rises toward both ends of the range.
The horn angle that gives each PIP angle is in <code>data/20261119-v4b_pip.json</code> for the servo's command. An earlier
version of the layout page gave 2.06 times for v4c's PIP drive. Its code inverted the ratio $d\\theta/d\\varphi$; the
corrected value for v4c's layout is 1.22 times.</p>"""


def sec_clearance(J):
    least, beam, moving = gaps(J)
    hw = J["check_fine"]["module_half_width_mm"]
    rows = []
    for k, v in least[:14]:
        a, b = k.split(" | ")
        rows.append([f"{part_name(a)} and {part_name(b)}" + (" (designed)" if designed(k, v[0]) else ""),
                     f"{v[0]:.2f}"] + pose_cells(k, v))
    tab = table(["Pair", "Gap (mm)", "Yaw (deg)", "MCP (deg)", "PIP (deg)"], rows,
                "The fourteen closest pairs of parts over the 9,828 poses, with the pose of each pair's least gap. A joint that "
                "does not move the two parts relative to each other is marked any.", num_from=1)
    k0, v0 = moving[0]
    a0, b0 = k0.split(" | ")
    return f"""<p>Apart from the designed gaps, the closest parts are the {part_name(a0)} and the {part_name(b0)},
{v0[0]:.2f} mm apart at a PIP of {mn(v0[3], 0)}&deg; (Table TAB[gaps]). The yoke and the MCP link stay
{beam[0]:.2f} mm below the x-stage beam. The module's half-width is {hw[0]:.1f} mm, at the back plate Y1, against the limit
of {B['half_width']:.1f} mm. The PIP rod closes at every pose of the grid.</p>
{tab}
<p>H1 and R1 come closest at &minus;25&deg;, the end of the checked range. The drive closes down to
{mn(J['pip']['range_deg'][0], 0)}&deg;, but the clearance was checked from &minus;25&deg;, 7&deg; past the driver's
limit.</p>"""


def sec_cut(J):
    pj = J["parts"]
    fig = img(MEDIA / "20261119-v4b_cut_sheet.png",
              f"The cut sheet, {pj['sheet_mm'][0]:.0f} by {pj['sheet_mm'][1]:.1f} mm. Red: cut outlines, holes and slots. Blue: "
              "the engraved part labels. C1 and C2 are the fit coupons.")
    order = ["M1", "M2", "M3", "Y1", "Y2", "L1", "L2", "L3", "L4", "L5", "L6a", "L6b", "H1", "R1", "C1", "C2"]
    seen, rows = set(), []
    for pid in order:
        ps = [p for p in pj["parts"] if p["id"] == pid]
        if not ps or pid in seen:
            continue
        seen.add(pid)
        p = ps[0]
        web = "&ndash;" if p["least_web_mm"] is None else f"{p['least_web_mm']:.2f}"
        name = p["name"].split(" ", 1)[1]
        name = re.sub(r" \([+-]y\)$", "", name) if len(ps) > 1 else name.replace("(-y)", "(&minus;y)")
        rows.append([p["id"], name[0].upper() + name[1:], str(len(ps)), f"{p['size_mm'][0]:.1f} &times; {p['size_mm'][1]:.1f}",
                     str(p["holes"]), str(p["slots"]), web])
    tab = table(["Part", "Name", "Qty", "Size (mm)", "Holes", "Slots", "Least web (mm)"], rows,
                "The laser-cut parts. The size is the part's bounding box on the sheet; the slots include the servo cutouts.",
                num_from=2)
    n_parts = sum(1 for p in pj["parts"] if not p["id"].startswith("C"))
    return f"""<p>The {n_parts} plates and two coupons fit on one sheet of 1/8-inch ({T:.3f} mm) cast acrylic,
{pj['sheet_mm'][0]:.0f} by {pj['sheet_mm'][1]:.1f} mm (<a href="cut/20261119-v4b_acrylic_3mm.dxf"><code>cut/20261119-v4b_acrylic_3mm.dxf</code></a>).
The DXF is R12 in mm: cut outlines are closed polylines and circles on the layer CUT, labels are text on the layer ENGRAVE.
Holes and slots are drawn at nominal size. The bearing holes are 6 mm for the MF63ZZ and 12 mm for the spherical
bearing, M2 holes are 2.2 mm, and slots are 0.1 mm wider than the sheet. Coupon C1 has bearing holes from 5.8 to 6.1 mm and from 11.8 to 12.0 mm,
and slots from 3.0 to 3.3 mm. Coupon C2 is one STS3032 seat, the case cutout and its four tab holes as in M1, Y2 and the
strips L6a and L6b. Cut both first; the hole sizes and the slot width are set in <code>B</code> of
<code>scripts/cad/v4b_build.py</code>, and <code>dxf</code> rewrites the sheet.</p>
{fig}
<p>The servo seats have the thinnest webs, 1.11 mm between the case cutout and a tab hole. The STS3032's tab holes lie
2.31 and 2.49 mm beyond the ends of its case, so any plate that takes the servo by its tabs has webs of this size. L6a's
1.01 mm web runs along its top edge, which is cemented to L3. Every other web is 1.39 mm or more (Table TAB[parts]).</p>
{tab}"""


def sec_bom(J):
    head, *rows = J["bom"]
    body = [[esc(r[0]), r[1], esc(r[2]), esc(r[3]), esc(r[4]), esc(r[5])] for r in rows]
    tab = table(["Item", "Qty", "Specification", "Where", "Source", "Note"], body,
                "Bill of materials for one finger (<code>data/20261119-v4b_bom.csv</code>).", num_from=1)
    return f"""<p>Besides the three servos on hand, one finger is built from five MF63ZZ bearings and one spherical plain
bearing. The pins are two precision shoulder screws and four dowels; the rest is common M2 and M3 hardware. The printed
parts are the knuckle, the yoke idler plate Y3 and the DIP link.</p>
{tab}"""


def sec_printed(J):
    pr, KO = J["printed"], J["keepout"]
    A0 = V.pip_kin(V.np.array([0.0]))[1]["A0"]
    pz = -B["proximal"]
    rows = [["Cheeks", f"two, {B['dip_cheek_y'][1] - B['dip_cheek_y'][0]:.1f} mm thick, at $|y|$ = {B['dip_cheek_y'][0]:.1f} to "
             f"{B['dip_cheek_y'][1]:.1f} mm; inner faces {2 * B['dip_cheek_y'][0]:.1f} mm apart"],
            ["PIP pins", "a press fit for a 3&times;6 dowel through each cheek, on the PIP axis"],
            ["Outer bosses", "4 mm across, 0.3 mm tall, on each cheek's outer face around the PIP pin; they bear on the PIP bearings' inner rings"],
            ["Lug", f"{B['pip']['lug_r']:.0f} mm from the PIP axis: at PIP 0, {A0[0]:.2f} mm dorsal and {pz - A0[2]:.2f} mm "
             "distal of the PIP axis"],
            ["Lug pin", "a press fit for a 3&times;12 dowel through both cheeks; the dowel ends 0.1 mm inside the outer faces"],
            ["Inner bosses", "4 mm across, 1.3 mm tall, on each cheek's inner face around the lug pin; they clamp the spherical bearing's 4.8 mm inner ring"],
            ["Rod's sweep", "no material inside <a href=\"data/20261119-v4b_dip_keepout.stl\"><code>data/20261119-v4b_dip_keepout.stl</code></a>, "
             f"the rod's boxes over the PIP's closing range in the DIP link's frame (origin on the PIP axis, $x$ dorsal, $z$ toward "
             f"the MCP at PIP 0, mm); it spans $y$ = {mn(KO['bounds_mm'][0][1], 2)} to {KO['bounds_mm'][1][1]:.2f} mm"],
            ["Fingertip seat", "b v4's tongue, board seat and column below the PIP axis"]]
    tab = table(["Feature", "Dimension"], rows, "The DIP link's interface with the lower plates and the rod.", num_from=9)
    return f"""<p>The knuckle (<a href="data/20261119-v4b_knuckle.stl"><code>data/20261119-v4b_knuckle.stl</code></a>) is a
{pr['knuckle_mm'][0]:.1f} by {pr['knuckle_mm'][1]:.1f} by {pr['knuckle_mm'][2]:.1f} mm block with two 3.1 mm holes at right
angles, {B['pip']['knuckle_e']:.0f} mm apart. A 4 mm boss 0.5 mm tall around each hole bears on a bearing's inner ring and keeps
the block 0.5 mm off the flange. Pin a's head sits on the block's far face; pin b's head sits in a 5.6 mm counterbore
1.3 mm deep. The yoke idler plate Y3
(<a href="data/20261119-v4b_y3_idler_plate.stl"><code>data/20261119-v4b_y3_idler_plate.stl</code></a>) is the plate's outline
extruded {T:.3f} mm, with a 2.95 mm hole reamed to a press fit for the idler dowel. Both are printed in PETG or PLA at
100&nbsp;% infill.</p>
<p>The DIP link is to be drawn from b v4 in Inventor, with b v4's fingertip seat on new cheeks (Table TAB[dip]). The
clearance check models it as the two cheeks, a lug arm and a box for the tip.</p>
{tab}"""


def sec_assembly():
    return """<ol>
<li>Cut the coupons C1 and C2. Choose the bearing holes that press fit and the slot width that takes a tab snugly; set
them in <code>scripts/cad/v4b_build.py</code> and rerun <code>dxf</code>. Cut the sheet.</li>
<li>Press the MF63ZZs into L2, L4 and L5 with the flange on the face toward the servo or the DIP link. Press those of
H1 and R1 with the flange toward the knuckle. Bond the spherical bearing into R1.</li>
<li>Cement M1 into the top plate M2 with the two gussets M3. Screw the yaw servo into M1 (4 M2&times;8 with nuts).</li>
<li>Fit pin b: the 3&times;7 shoulder screw through the knuckle and R1's bearing, its nut in R1's bore recess. Fit pin a:
the 3&times;12 shoulder screw through the knuckle and H1's bearing, its nut in H1's bore recess. Screw H1 to the PIP
servo's disc horn (2 M2&times;5).</li>
<li>Press the 3&times;12 dowel through one cheek of the DIP link, the spherical bearing and the other cheek. Press the two
3&times;6 dowels into the cheeks on the PIP axis.</li>
<li>Screw the PIP servo to L6a and L6b. Put the DIP link's dowels into the bearings of L4 and L5 and the strips' tabs into
their slots. Cement L1 and L2 into L3, then L4 and L5.</li>
<li>Centre each servo (position 2048) before its disc horn goes on, so that the rest pose lies at mid-travel.</li>
<li>Screw the MCP servo into Y2, cement Y2's tab into Y1 and screw Y1 to the yaw servo's disc horn (4 M2&times;5).</li>
<li>Bring the MCP link up around the MCP servo and screw L1 to its disc horn (4 M2&times;5).</li>
<li>Press the 3&times;8 dowel into Y3, slide it into L2's bearing and glue Y3's tab into Y1 with cyanoacrylate.</li>
</ol>"""


def sec_limits(J):
    return f"""<p>The check models parts as boxes and flat cylinders and takes poses on a grid of 15&deg; in yaw, 3&deg; in MCP
and 6&deg; in PIP. The DIP link is an envelope until it is drawn. It and the fingertip are left out of the beam check:
with the MCP and PIP angles summing to 158&deg; or more, the fingertip envelope reaches the beam. The yoke and the finger
hang on the yaw servo's output shaft
alone, with no idler on the yaw axis. A 3 N force along $x$ at the pad, about 92 mm below that shaft, bends it with
0.28 N&nbsp;m. Feetech gives no load rating for the shaft. The servo seats' 1.1 mm webs are acrylic's weak point. The top
plate M2 is a test-stand plate. On the hand the module needs a new carriage underside: G sits {mn(-B['Gz'], 0)} mm and the
PIP axis {mn(-(B['Gz'] - B['proximal']), 0)} mm below v3's yaw axis. Bearing fits, play and friction are not measured;
the clearances and torques are from the model.</p>"""


def sec_future(J):
    hw = J["check_fine"]["module_half_width_mm"][0]
    return f"""<ul>
<li>Cut C1 and C2 and record the press-fit hole sizes and the slot width; rerun <code>scripts/cad/v4b_build.py dxf</code>.
If C2's webs crack when its four M2 screws are tightened, cut M1, Y2, L6a and L6b from Delrin of the same thickness or
print them.</li>
<li>Draw the DIP link from b v4 with the interface of Table TAB[dip], replace its envelope in <code>solids()</code> with
boxes from its STL and rerun <code>check --fine</code>. A gap under 0.5 mm, other than the designed ones, refutes the
interface.</li>
<li>Assemble the finger and measure its play. Hold each joint at rest, load the pad with 2 N in each direction, and read
the pad's displacement with a dial gauge while the servos' positions are logged. More than 1&deg; at any joint beyond the
servo's own reading refutes the claim that the bearings and pins add no slop.</li>
<li>Draw the carriage underside for the hand, with the module's half-width of {hw:.1f} mm and the beam's underside at
21.9 mm. Then rerun the layout page's index-middle replay on the built dimensions.</li>
</ul>"""


def sec_files():
    rows = [["<code>scripts/cad/v4b_build.py</code>", "parameters, parts, clearance check, PIP drive, cut sheet, STLs, bill of "
             "materials and renders (<code>check --fine</code>, <code>pip</code>, <code>dxf</code>, <code>printed</code>, "
             "<code>bom</code>, <code>render</code>)"],
            ["<code>scripts/cad/v4b_build_page.py</code>", "this page"],
            ["<code>scripts/cad/v4c_hardware.py</code>", "the PIP drive's kinematics (universal joint at the horn, spherical bearing at the lug)"],
            ["<code>cut/20261119-v4b_acrylic_3mm.dxf</code>", "the cut sheet"],
            ["<code>data/20261119-v4b_bom.csv</code>", "the bill of materials"],
            ["<code>data/20261119-v4b_knuckle.stl</code>, <code>data/20261119-v4b_y3_idler_plate.stl</code>", "the printed parts"],
            ["<code>data/20261119-v4b_dip_keepout.stl</code>", "the rod's sweep in the DIP link's frame, for drawing the DIP link"],
            ["<code>data/20261119-v4b_check_fine.json</code>", "the forty closest pairs, the module's half-width and the parameters"],
            ["<code>data/20261119-v4b_pip.json</code>", "the PIP drive: range, torque, horn angle and tilt along the PIP angle"],
            ["<code>data/20261119-v4b_parts.json</code>, <code>data/20261119-v4b_printed.json</code>", "part sizes and webs; the pin stacks"]]
    return table(["File", "Content"], rows, "Files, relative to this page's folder unless they start with <code>scripts/</code>.",
                 num_from=9)


def lede_and_summary(J):
    least, beam, moving = gaps(J)
    P = J["pip"]
    hw = J["check_fine"]["module_half_width_mm"][0]
    n_plates = sum(1 for p in J["parts"]["parts"] if not p["id"].startswith("C"))
    lede = (f"A v4b index finger can be built from {n_plates} plates of 1/8-inch acrylic and three STS3032 servos. Its pins "
            f"are steel and turn in five MF63ZZ ball bearings and a spherical plain bearing; three parts are printed. The least "
            f"gap between moving parts is {moving[0][1][0]:.2f} mm over the driver's yaw and MCP ranges and the PIP from "
            f"&minus;25&deg; to 92&deg;. Only two bearing flanges come closer, at 0.3 mm by design. The module stays "
            f"{beam[0]:.2f} mm below the x-stage beam and within &plusmn;{hw:.1f} mm of its plane. The PIP drive gives at least "
            f"{P['torque_scs0009_min_m25_92']:.2f} times the current hand's PIP torque.")
    summary = (f"A v4b index finger from {n_plates} acrylic plates, three STS3032s, five MF63ZZ bearings, a spherical plain "
               f"bearing and three printed parts keeps its moving parts at least {moving[0][1][0]:.2f} mm apart over the driver's "
               f"ranges, and its PIP drive gives {P['torque_scs0009_min_m25_92']:.2f} times the current hand's PIP torque. "
               f"The cut sheet (DXF) and the bill of materials are in the folder.")
    return lede, summary


def main():
    J = load()
    sections = [("terms", "Terms and metrics", sec_terms(J)),
                ("layout", "Layout of the finger and its joints", sec_layout(J)),
                ("pip", "PIP drive through a knuckle and a spherical plain bearing", sec_pip(J)),
                ("clearance", "Clearance over the joint ranges", sec_clearance(J)),
                ("cut", "Cut sheet and fit coupons", sec_cut(J)),
                ("bom", "Bill of materials", sec_bom(J)),
                ("printed", "Printed parts and the DIP link's interface", sec_printed(J)),
                ("assembly", "Assembly order", sec_assembly()),
                ("limits", "Limitations", sec_limits(J)),
                ("future", "Future work", sec_future(J)),
                ("files", "Files", sec_files())]
    num_ = {sid: k for k, (sid, _h, _b) in enumerate(sections, 1)}
    sub = (lambda s: re.sub(r"SEC\[(\w+)\]", lambda m_: str(num_[m_.group(1)]), s))
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|N\b|g\b)", "\\1&nbsp;\\2", s_))
    body = [(sid, hd, unit(sub(b))) for sid, hd, b in sections]
    html_all = "\n".join(b for _s, _h, b in body)
    caps = dict((cap[:30], n) for n, cap in re.findall(r"<caption>Table (\d+)\. ([^<]{0,40})", html_all))
    ref = {"joints": next(n for c, n in caps.items() if c.startswith("The joints and their pins")),
           "gaps": next(n for c, n in caps.items() if c.startswith("The fourteen closest pairs")),
           "parts": next(n for c, n in caps.items() if c.startswith("The laser-cut parts")),
           "dip": next(n for c, n in caps.items() if c.startswith("The DIP link's interface"))}
    body = [(sid, hd, re.sub(r"TAB\[(\w+)\]", lambda m_: ref[m_.group(1)], b)) for sid, hd, b in body]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(body, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{b}</section>' for k, (sid, hd, b) in enumerate(body, 1))
    lede, summary = lede_and_summary(J)
    byline = ("Design and clearance check in Python; nothing has been cut or assembled. Builders: "
              "<code>scripts/cad/v4b_build.py</code>, <code>scripts/cad/v4b_build_page.py</code>.")
    page = TEMPLATE.read_text().replace("The cad_v3 page (scripts/cad/cad_v3_page.py)", "The v4b build page (scripts/cad/v4b_build_page.py)")
    for k_, v in {"TITLE": "v4b finger build", "H1": "Parts and clearances of a laser-cut v4b index finger",
                  "EYEBROW": "Hand and bench, 2026-11-19", "LEDE": unit(lede), "BYLINE": byline, "TOC": toc, "SECTIONS": secs,
                  "FOOTER": str(OUT.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k_ + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in H.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = H.TEX_RE.sub(lambda m: next(svgs), page)
    page = docs_index.tag(page, topic="hardware", summary=summary, date="2026-11-19")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT}: {OUT.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
