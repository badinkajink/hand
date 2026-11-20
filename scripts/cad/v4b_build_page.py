#!/usr/bin/env python3
"""Builds docs/experiments/20261119-v4b_build/20261119-v4b_finger_build.html: the v4b index finger in mixed manufacture
(printed bodies, a laser-cut PIP linkage), with its yaw stage held at both ends of the yaw servo, its carriage on the
index gantry, the clearance check, the PIP drive, the playable MJCF, the cut sheet, the printed parts and the bill of
materials. Owner, 2026-11-20: "seems like most of these parts can be 3d printed. i just wanted to use the laser cutter
for things that needed more rigidity/strength, like the linkages"; "can we just make the phalanx longer? so we can
minimize the cantilever?"; "you're going to have to design the carriage as well to minimize cantilever off the lead
screw nut". The all-acrylic version of 2026-11-19 is commit 823a33fbb.

Data: the folder's data/20261119-v4b_{check_fine,stroke,pip,loads,parts,printed}.json and data/20261119-v4b_bom.csv,
written by scripts/cad/v4b_build.py (check --fine, stroke, pip, loads, dxf, printed, bom), its media/ figures and film
(render) and assets/mjcf/experimental/20261120-v4b_mixed/ (mjcf).

    .venv/bin/python scripts/cad/v4b_build_page.py
"""
from __future__ import annotations

import base64
import csv
import io
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
CAD_V3 = "../20261104-cad_v3_hand/20261104-cad_v3_hand_on_ur5.html"
FILM = "media/20261119-v4b_mixed_range.mp4"
table, img, esc = H.table, H.img, H.esc
B, T = V.B, V.T


def load():
    J = {k: json.loads((DATA / f"20261119-v4b_{k}.json").read_text())
         for k in ("check_fine", "stroke", "pip", "loads", "parts", "printed", "flush")}
    with (DATA / "20261119-v4b_bom.csv").open() as fh:
        J["bom"] = list(csv.reader(fh))
    J["g"] = V.geometry()
    J["keepout"] = V.cmd_keepout()  # writes data/20261119-v4b_dip_keepout.stl
    return J


def mn(x, nd=1):
    """A number with a typographic minus."""
    return f"{x:.{nd}f}".replace("-", "&minus;")


NAMES = [("flange_PIP", "PIP bearing's flange"), ("flange_L2", "idler bearing's flange"), ("flange_Y4", "rear bearing's flange"),
         ("flange_H1", "pin a's bearing flange (H1)"), ("flange_R1", "pin b's bearing flange (R1)"),
         ("dip_cheek", "DIP link cheek"), ("dip_lugarm", "DIP link lug arm"), ("tip", "fingertip envelope"),
         ("sph_ring", "spherical bearing's ring"), ("pipdisc", "PIP servo's disc horn"), ("mcpdisc", "MCP servo's disc horn"),
         ("yawdisc", "yaw servo's disc horn"), ("idler_dowel", "idler dowel"), ("rear_dowel", "rear dowel"),
         ("knuckle_boss", "knuckle boss"), ("knuckle", "knuckle"), ("pin_a_nut", "pin a's nut"), ("pin_b_nut", "pin b's nut"),
         ("pin_a_head", "pin a's head"), ("pin_a", "pin a's screw"), ("pin_b", "pin b's screw"), ("leadnut", "lead nut"),
         ("H1 PIP crank", "crank H1"), ("R1 PIP rod", "rod R1"),
         ("C column", "carriage column"), ("C block plate", "carriage block plate"), ("C rail", "carriage rail"),
         ("C pocket side", "carriage pocket side"), ("C roof over horn", "carriage roof over the horn"), ("C front roof", "carriage front roof"),
         ("C roof", "carriage roof"), ("C rear wall", "carriage rear wall"),
         ("Y1 front arm", "yoke front arm"), ("Y1 web root", "yoke front arm (web end)"), ("Y1 idler bar", "yoke front arm (idler bar)"),
         ("Y1 collar root", "yoke front arm (collar end)"), ("Y4", "yoke rear arm"), ("Y web", "yoke web"), ("Y2 collar root", "yoke collar root"),
         ("Y2 collar", "yoke collar"), ("Y3 idler arm", "yoke idler arm"), ("L1 horn side", "link horn side"), ("L2 idler side", "link idler side"),
         ("L merge block", "link merge block"), ("L cradle side", "link cradle side"), ("L cheek root", "link cheek"),
         ("L PIP cheek", "link PIP cheek"), ("L tab frame", "link tab frame")]


def part_name(raw):
    """A readable name for a primitive of the clearance check."""
    nm = raw.split(":", 1)[1]
    m = re.match(r"(yaw|mcp|pip)servo_(case|tabs|boss)", nm)
    if m:
        who = {"yaw": "yaw", "mcp": "MCP", "pip": "PIP"}[m.group(1)]
        return f"{who} servo's {dict(case='case', tabs='tabs', boss='spline boss')[m.group(2)]}"
    for k, v in NAMES:
        if nm.startswith(k):
            return v
    return nm


def designed(key, gap):
    """Gaps the parts are drawn with: a bearing's flange 0.3 mm from the DIP link's cheek, or 0.6 mm from the carriage's
    rear wall."""
    return (("flange" in key) and ("dip_cheek" in key or "dip_lugarm" in key) and abs(gap - 0.3) < 0.02) or \
           ("flange_Y4" in key and "rear wall" in key and abs(gap - 0.6) < 0.02)


def boxed_disc(key):
    """The yaw servo's disc horn under the carriage's roof: the check draws the disc as three rectangles whose corners reach
    1.155 times its radius, so this gap reads 1.4 mm short of the round disc's 1.9 mm."""
    return "roof over horn" in key and "yawdisc" in key


LEVEL = {"carriage": 0, "yoke": 1, "link": 2, "phorn": 3, "knuckle": 3, "rod": 3, "dip": 3}


def pose_cells(key, v):
    """The yaw, MCP and PIP of a pair's least gap; 'any' for a joint that does not move the two parts relative to each
    other (the carriage stays fixed, the yoke turns with the yaw, the link with the MCP, the rest with the PIP)."""
    la, lb = sorted(LEVEL[p.split(":", 1)[0]] for p in key.split(" | "))
    dep = (la == 0 and lb >= 1, la <= 1 and lb >= 2, lb == 3)
    return [mn(x, 0) if d else "any" for x, d in zip(v[1:], dep)]


def gaps(J):
    least = [(k, v) for k, v in J["check_fine"]["least"]]
    moving = [(k, v) for k, v in least if not designed(k, v[0]) and not boxed_disc(k)]
    return least, moving


def film(rel, caption):
    """The film as a file beside the page, with a poster frame inlined."""
    poster = (D / rel).with_suffix(".jpg")
    pa = ""
    if poster.exists():
        pa = f' poster="data:image/jpeg;base64,{base64.b64encode(poster.read_bytes()).decode()}"'
    return (f'<figure><video controls muted loop playsinline preload="metadata"{pa} src="{rel}"></video>'
            f'<figcaption>Film 1. {caption}</figcaption></figure>')


def ld(J, design, case):
    return J["loads"]["designs"][design][case]


def sec_terms(J):
    g = J["g"]
    return f"""<dl class="terms">
<dt>Mount frame, gimbal centre G</dt><dd>The finger's frame on its carriage, in mm: $x$ points to the dorsal side, $y$ to
the side and $z$ up toward the palm. Its origin is on v3's yaw axis at the centre of the index travel box. The yaw axis
runs along $x$ and the MCP axis along $y$; they cross at G, {mn(-g['Gz'], 0)} mm below v3's yaw axis. The PIP axis lies
{B['proximal']:.0f} mm below G when the finger is straight.</dd>
<dt>Yaw, MCP and PIP angles</dt><dd>The joint angles in degrees, as in the simulator. The driver's ranges are yaw
&plusmn;85&deg;, MCP &minus;15&deg; to 92&deg; and PIP &minus;18&deg; to 92&deg;.</dd>
<dt>Stroke $s$</dt><dd>The carriage's travel along $x$ from the travel box's centre, in mm. It runs from
{mn(V.STROKE[0], 2)} mm at the far stop to {V.STROKE[1]:.0f} mm at the stop beside the stepper. The firmware homes at that
stop and uses 60 mm of the stroke (<a href="{CAD_V3}">cad_v3 page</a>).</dd>
<dt>Yoke reach</dt><dd>The distance along $x$ from G to the middle of the yaw servo's disc horn, in mm. The finger hangs
from the yaw servo at that distance.</dd>
<dt>Rear dowel, support span $L_s$</dt><dd>The 3 mm dowel behind the yaw servo's base, on the yaw axis, held by the
carriage; the yoke's rear arm turns on it in a ball bearing. The support span is the distance along $x$ from the disc
horn's middle to the rear bearing's middle.</dd>
<dt>Bending moment at the horn, radial force</dt><dd>Take a force $\\mathbf{{F}}$ at the pad and its moment about the
horn's middle. The yaw servo turns against the moment's component along the yaw axis. The component square to that axis,
$M_\\perp$, bends the servo's output. With the rear dowel, $M_\\perp$ is taken as a pair of radial forces
$M_\\perp / L_s$ at the horn and at the dowel; the horn also takes $\\mathbf{{F}}$'s own radial part. The pad lies
7.4 mm palmar of the finger's axis and 36.36 mm below the PIP axis at MCP and PIP 0.</dd>
<dt>Lever from the lead nut</dt><dd>The height of the lead screw's axis ({V.LEAD_NUT['axis'][1]:.2f} mm) above the pad,
in mm. A pinch force along $x$ at the pad tilts the carriage on its MGN7C block with this lever.</dd>
<dt>Gap</dt><dd>The least distance in mm between two parts over a grid of 9,828 poses. Yaw runs from &minus;85&deg; to
85&deg; in 15&deg; steps and MCP from &minus;15&deg; to 92&deg; in 3&deg; steps. PIP runs from &minus;25&deg; to 92&deg; in
6&deg; steps; 85&deg; and 92&deg; are added at the ends. Each printed or acrylic plate is cut into 1 mm strips (boxes);
blocks, servos, horns, bearings, screws, the knuckle and the DIP link are boxes or flat cylinders. Pairs that touch by
design, such as a pin in its bearing, are left out.</dd>
<dt>Designed gap</dt><dd>A gap drawn into the parts. A PIP bearing's flange lies 0.3 mm from the DIP link's cheek, and
the rear bearing's flange 0.6 mm from the carriage's rear wall.</dd>
<dt>Gantry clearance</dt><dd>The least gap in mm between the carriage or the finger and the gantry's parts below the
beam: the beam, its bearing blocks, the coupler, the stepper, the lead screw and the rail. It is taken over 13 stroke
positions and 300 finger poses.</dd>
<dt>Module half-width</dt><dd>The largest $|y|$ of the carriage and the yoke over all yaw angles. The index and middle
mounts lie 56.8 mm apart, so the limit is {B['half_width']:.1f} mm.</dd>
<dt>PIP torque per SCS0009 limit</dt><dd>The PIP torque at the PIP servo's stall torque, divided by the stall torque of
the SCS0009, the current hand's PIP servo. By virtual work
$\\tau_{{\\mathrm{{PIP}}}} = \\tau_{{\\mathrm{{horn}}}}\\, d\\theta/d\\varphi$, with $\\theta$ the horn angle and
$\\varphi$ the PIP angle. The stall torques at 6 V are 0.441 N&nbsp;m (STS3032) and 0.226 N&nbsp;m (SCS0009).</dd>
<dt>Least web</dt><dd>The narrowest strip of plate between two cut contours of one acrylic part, in mm.</dd>
<dt>Precision shoulder screw</dt><dd>A screw with a ground, unthreaded 3 mm shoulder, 0 to 0.025 mm undersize. The
McMaster 90323A series used here has a 5 mm by 1.3 mm head and an M2 thread 4 mm long.</dd>
</dl>"""


def sec_make(J):
    pr = J["printed"]["parts"]
    vol = lambda k: pr[k]["volume_mm3"] / 1000.0  # noqa: E731
    rows = [["P1 carriage", "printed", f"{vol('P1 carriage'):.1f}", "a column around the lead nut, rails beside the stepper and a pocket for the yaw servo in one part"],
            ["P2 yoke", "printed", f"{vol('P2 yoke'):.1f}", "front arm, web, rear arm, the MCP servo's collar and the idler arm in one part, with no glued joints"],
            ["P3 proximal link", "printed", f"{vol('P3 link'):.1f}", "one part from the MCP horn to the PIP bearings; the PIP servo slides into its cradle"],
            ["P4 knuckle", "printed", f"{vol('P4 knuckle'):.2f}", "a small block with two crossed holes"],
            ["H1 crank", "acrylic", "&ndash;", "a thin link loaded in its plane, 37 N from the rod at the PIP servo's stall; a bearing 1.75 mm from its edge"],
            ["R1 rod", "acrylic", "&ndash;", "a thin link in tension and compression between two bearings"],
            ["DIP link", "printed", "&ndash;", "to be drawn from b v4 in Inventor"]]
    tab = table(["Part", "Made by", "Volume (cm<sup>3</sup>)", "Reason"], rows,
                "How each part is made. The volume is the solid's; at 1.24 g/cm<sup>3</sup> (PLA) and 40 % infill the printed "
                "parts weigh about half of it in grams.", num_from=2)
    return f"""<p>The bodies are printed and the PIP linkage is cut from acrylic (Table TAB[make]). The
acrylic build of 2026-11-19 had 15 plates with tab-and-slot joints, and its servo seats left webs of 1.1 mm between a case
cutout and a tab hole. Here each servo sits in a printed body: the yaw servo in a pocket of the carriage, the MCP servo in
a collar of the yoke and the PIP servo in a cradle of the link. Each servo's tabs screw into the print with
self-forming M2 threads in 1.8 mm holes. Ball bearings press into printed bores; the print fit coupon P0 sets their size,
the dowels' press fit and the M2 hole before the parts are printed.</p>
{tab}
<p>The crank H1 and the rod R1 are the two links of the PIP drive. Each is a 1/8-inch plate loaded in its plane, with
bearings near its ends. A laser-cut hole is round to the kerf and the plate is the same in every direction, so these two
stay acrylic. They fit on a sheet of 130 by 58 mm with a spare of each and the fit coupon C1 (Section SEC[parts]).</p>"""


def sec_yaw(J):
    g = J["g"]
    acr, mix = "v4b acrylic (2026-11-19)", "v4b mixed (this build)"
    fig = img(MEDIA / "20261119-v4b_side_compare.png",
              "Side view at yaw 0, MCP &minus;15&deg; and PIP &minus;25&deg;, where the crank rises highest toward the yaw "
              "stage. Each part's boxes are projected onto the x-z plane. The dotted line marks the yaw servo's spline. Left: "
              "the acrylic build, its yaw servo's disc horn 36 mm behind G. Right: this build, with the yaw "
              "servo 20 mm closer to G, the yoke's web under it and the rear dowel behind it.")
    fig2 = img(MEDIA / "20261119-v4b_yaw_stage.png",
               "The yaw stage from behind and below at yaw 25&deg;, MCP 30&deg; and PIP 20&deg;, with the carriage drawn "
               "translucent. The yoke (blue) wraps under the yaw servo from its disc horn to the rear arm, which turns on "
               "the dowel in the carriage's rear wall.", jpeg=True)
    rows = []
    for F in (3.0, 12.1):
        for case, lab in (("pinch", "along $x$ (pinch)"), ("lateral", "along $y$"), ("vertical", "along $z$")):
            a, b = ld(J, acr, f"{case}_{F:g}N"), ld(J, mix, f"{case}_{F:g}N")
            rows.append([f"{F:g} N {lab}", f"{a['bending_Nm_at_horn']:.2f}", f"{a['horn_radial_N']:.1f}", "0",
                         f"{b['horn_radial_N']:.1f}", f"{b['rear_radial_N']:.1f}"])
    tab = table(["Force at the pad", "Acrylic: bending at the horn (N m)", "Acrylic: radial at the horn (N)",
                 "This build: bending (N m)", "This build: radial at the horn (N)", "This build: radial at the dowel (N)"], rows,
                f"The yaw servo's output under a force at the pad, at MCP and PIP 0. The acrylic build hangs the finger on the "
                f"output alone; this build adds the rear dowel {J['loads']['designs'][mix]['pinch_3N']['span_mm']:.1f} mm behind "
                f"the horn. 12.1 N is the pinch at the STS3032's stall torque on the MCP.", num_from=1)
    p3a, p3m = ld(J, acr, "pinch_3N"), ld(J, mix, "pinch_3N")
    return f"""<p>The yaw servo now sits with its spline {g['xY']:.0f} mm behind G, and the yoke wraps around it to a
rear dowel. In the acrylic build the spline was 37 mm behind G. That distance came from the PIP crank: at MCP
&minus;15&deg; the proximal link tips its dorsal side up. The knuckle then came within 1.45 mm of the yoke's back plate
(Figure FIG[side], left). This build makes the proximal link {B['proximal'] - 60:.0f} mm longer, {B['proximal']:.0f} mm
from G to the PIP axis, and lowers the PIP servo with it. At the same pose the knuckle passes 1.48 mm under the yoke's
web, and the yaw servo moves 20 mm forward. The yoke's reach falls from {J['loads']['designs'][acr]['horn_reach_mm']:.0f}
to {J['loads']['designs'][mix]['horn_reach_mm']:.0f} mm. The crank and rod keep their dimensions relative to the PIP
axis, so the PIP drive is unchanged (Section SEC[pip]).</p>
{fig}
<p>The yoke is a U around the yaw servo. Its front arm is screwed to the servo's disc horn, and a web passes
{B['web']['r']:.1f} mm below the yaw axis. Its rear arm turns in an MF63ZZ on a 3 mm dowel. That dowel is
bonded into the carriage's rear wall, {B['yaw_rear_gap'] + B['mf63']['flange_t']:.1f} mm behind the servo's base. The web clears the
servo's lower tabs at every yaw angle. The carriage grips the servo's upper half in a pocket and screws its two upper tabs
to the pocket's front wall. Further forward, the web would meet the link's merge block at MCP &minus;15&deg;; it now
clears it by 0.60 mm.</p>
{fig2}
<p>With the dowel, the yaw servo's output takes no bending moment (Table TAB[loads]). A 3 N pinch at the pad bends the
acrylic build's output with {p3a['bending_Nm_at_horn']:.2f} N&nbsp;m. Here the same pinch puts
{p3m['horn_radial_N']:.1f} N on the horn and {p3m['rear_radial_N']:.1f} N on the dowel. The output's own bearings sit a few
millimetres apart inside the case, so a moment reaches them as a much larger pair of forces than these. Feetech gives no
load rating for the output. The moment comes from the pad's height below the yaw axis, not from the yoke's reach. The
dowel therefore relieves the output more than the shorter reach does: the reach changes only the moments of the lateral
and vertical forces.</p>
{tab}"""


def sec_carriage(J):
    st = J["stroke"]["least"]
    L = J["loads"]["designs"]
    curl = J["stroke"]["curl_reaches_gantry"]
    rows = []
    seen = set()
    for k, v in st:
        a, b = k.split(" | ")
        nm = part_name(b)
        if (a, nm) in seen or b.startswith("dip:"):
            continue
        seen.add((a, nm))
        rows.append([f"{a} and {nm}", f"{v[0]:.2f}", mn(v[1], 1)])
        if len(rows) == 8:
            break
    tab = table(["Pair", "Gap (mm)", "Stroke (mm)"], rows,
                "The closest gantry parts to the carriage over the stroke, with the stroke of each least gap. The finger's "
                "own parts stay further away.", num_from=1)
    return f"""<p>The carriage P1 bolts to the gantry where the v3 carriage did, so the travel box and the
{V.STROKE[1] - V.STROKE[0]:.1f} mm stroke stay as built. The lead nut's flange is screwed to its column with three M3
screws, and the MGN7C block with four M2 screws. The hole patterns are read from the v3 carriage in the owner's CAD. G lies under the
lead nut, {abs(V.LEAD_NUT['axis'][1] - B['Gz']):.0f} mm below the screw's axis. The yaw servo sits behind the column, under
the coupler and the stepper. Above the bearing blocks' underside at 24.4 mm, the column stays within the 20.7 mm of $x$
that the two blocks leave over the stroke. Below the stepper's underside at 21.9 mm, the roof over the yaw servo runs back
to the rear wall.</p>
<p>Two side rails, 4 mm thick at $|y|$ = {B['carriage']['rail_y'][0]:.0f} to {B['carriage']['rail_y'][1]:.0f} mm, join the
column to the roof and the rear wall. They rise to {B['carriage']['rail_top']:.0f} mm beside the coupler and the stepper,
which are at most 20.0 mm wide. The roof alone is thin there: the stepper, the coupler and the bearing blocks hang down to
21.9, 22.4 and 24.4 mm, and the finger's parts sweep up to 19 mm. The rails clear the stepper by 0.9 mm and the beam's other parts by 0.5 mm or
more over the whole stroke (Table TAB[stroke]). The lead screw runs through the column's 8.6 mm bore around the nut's
barrel.</p>
{tab}
<p>The longer finger raises the lever from the lead nut to the pad: {L['v3']['nut_to_pad_mm']:.1f} mm on v3,
{L['v4b acrylic (2026-11-19)']['nut_to_pad_mm']:.1f} mm on the acrylic build and {L['v4b mixed (this build)']['nut_to_pad_mm']:.1f} mm
here. The extra 6 mm of link adds 4 % to the acrylic build's lever. The finger curled back toward the palm reaches the
gantry's lower parts once MCP and PIP sum to {curl[0]:.0f}&deg; (MCP {curl[1]:.0f}&deg;, PIP {curl[2]:.0f}&deg;). The bench
trials so far kept that sum under 72&deg;.</p>"""


def sec_layout(J):
    fig = img(MEDIA / "20261119-v4b_assembly.png",
              "The index module at four poses, from the side (left) and from the palmar side (right), rendered from the MJCF "
              "with the carriage translucent. Grey: carriage and gantry. Blue: yoke. Brown: proximal link. White: crank H1 and "
              "rod R1. Yellow: knuckle. Green: DIP link envelope. Black: servos and the fingertip envelope.", jpeg=True)
    pr = J["printed"]["knuckle"]
    rows = [["Yaw", "the yaw servo's output and an MF63ZZ in the yoke's rear arm", "3&times;8 dowel bonded into the carriage's rear wall",
             "the servo; the yoke's front arm is screwed to its disc horn (4 M2&times;6)"],
            ["MCP", "the MCP servo's output and an MF63ZZ in the link's idler side", "3&times;8 dowel pressed into the yoke's idler arm",
             "the servo; the link's horn side is screwed to its disc horn (4 M2&times;6)"],
            ["PIP", "MF63ZZs in the link's two PIP cheeks, flanges inside", "two 3&times;6 dowels pressed into the DIP link's cheeks",
             "4 mm bosses on the DIP link's cheeks bear on the two inner rings"],
            ["Pin a, crank to knuckle", "an MF63ZZ in H1, flange toward the knuckle",
             f"3&times;{pr['pin_a']['shoulder']:.0f} shoulder screw from the knuckle's far face",
             "an M2 nut in H1's bore recess clamps the inner ring against the knuckle's boss"],
            ["Pin b, knuckle to rod", "an MF63ZZ in R1, flange toward the knuckle",
             f"3&times;{pr['pin_b']['shoulder']:.0f} shoulder screw from a counterbore in the knuckle", "an M2 nut in R1's bore recess, as at pin a"],
            ["Lug, rod to DIP link", "the spherical plain bearing, bonded into R1", "3&times;12 dowel pressed through both cheeks",
             "4 mm bosses on the cheeks' inner faces clamp the inner ring"]]
    tab = table(["Joint", "Turns in", "Pin", "Located along the pin by"], rows,
                "The joints and their pins. Every pin is steel and turns in a ball bearing or the spherical plain bearing.", num_from=9)
    ratio = J["pip"]["torque_scs0009_min_m25_92"] * 0.2256 / 0.4413
    return f"""<p>The finger keeps v4b's kinematics: the yaw and MCP axes cross at G (Figure FIG[assembly]). The PIP servo
lies in the proximal link and drives the DIP link through a crank, a knuckle and a rod. The MCP servo stands in the
yoke's collar with its shaft on the MCP axis. The link is screwed to that servo's disc horn on one side and turns on the
yoke's idler dowel on the other. Below the MCP servo a merge block joins the link's two sides to the cradle, where the PIP
servo lies with its shaft along $x$.</p>
{fig}
<p>Each joint turns in ball bearings on steel pins (Table TAB[joints]). The precision shoulders are 0 to 0.025 mm under
3 mm, and the MF63ZZ's bore is 0 to 0.008 mm under. The dowels are h8 or m6: an h8 dowel slides in a bearing with at most
0.014 mm of clearance, and an m6 dowel is a light press. No pin turns in a printed hole. Play at the PIP servo's horn
reaches the PIP divided by $d\\theta/d\\varphi$, which is at least {ratio:.2f} over the PIP's working range. One degree of
play at the horn therefore becomes at most {1 / ratio:.2f}&deg; at the PIP.</p>
{tab}"""


def sec_pip(J):
    P = J["pip"]
    fig = img(MEDIA / "20261119-v4b_pip_drive.png",
              "The PIP drive over its closing range. (a) PIP torque per SCS0009 limit; the dashed line is one SCS0009. (b) The "
              "horn angle and the spherical bearing's tilt; the dotted line is the bearing's 16&deg; limit. The grey band is the "
              "driver's PIP range.")
    lo, hi = P["range_deg"]
    return f"""<p>The PIP drive gives at least {P['torque_scs0009_min_m25_92']:.2f} times the current hand's PIP torque from
&minus;25&deg; to 92&deg;, and {P['torque_scs0009_median_m25_92']:.2f} times at the median. It closes from {mn(lo, 0)}&deg;
to {hi:.0f}&deg;. The crank was not changed: the longer link let the yaw servo come forward without trading torque for
room. The PIP servo's spline lies {abs(B['pip']['spline_z']):.0f} mm below G and {B['proximal'] + B['pip']['spline_z']:.0f} mm
above the PIP axis, as in the acrylic build relative to its PIP axis. H1 puts pin a {B['pip']['r']:.0f} mm from the servo's
shaft. The knuckle turns on pin a and on pin b, {B['pip']['knuckle_e']:.0f} mm away and square to it, and the rod is
{P['rod_mm']:.2f} mm long between its bearings.</p>
{fig}
<p>Over the closing range the horn turns from {P['horn_deg_at_ends'][0]:.1f}&deg; to {mn(P['horn_deg_at_ends'][1])}&deg;,
inside its {B['pip']['horn_limit_deg']:.0f}&deg; limit, and the spherical bearing tilts at most
{P['bearing_tilt_deg_max']:.1f}&deg;. The horn angle for each PIP angle is in <code>data/20261119-v4b_pip.json</code>.</p>"""



def sec_flush(J):
    F = J["flush"]
    tip = F["tip_palmar_x"]
    rows = []
    for v in F["variants"]:
        v = dict(v)
        v["least"] = next(p for p in [v["least"]] + v["next"] if not boxed_disc(p[0]))
        a, b = v["least"][0].split(" | ")
        rng = v["pip_range_deg"]
        rows.append([v["name"][0].upper() + v["name"][1:], mn(v["servo_base_x"]), f"{mn(rng[0], 0)} to {rng[1]:.0f}",
                     f"{v['torque_min_m18_92']:.2f}" if v["torque_min_m18_92"] is not None else "&ndash;", f"{v['rod_mm']:.1f}",
                     f"{v['least'][1][0]:.2f}", f"{part_name(a)} and {part_name(b)}"])
    tab = table(["Variant", "Servo base $x$ (mm)", "PIP closes (deg)", "Least torque, &minus;18 to 92&deg; (SCS0009)", "Rod (mm)",
                 "Least gap (mm)", "Between"], rows,
                f"The PIP servo's base moved toward the fingertip's palmar face ($x$ = {mn(tip)} mm). A dash marks a drive that does not "
                "close over the driver's range. The gap is the least over the clearance grid with PIP from &minus;18&deg;, without the "
                "designed gaps.", num_from=1)
    S = F["searches"]
    s1, s2 = list(S.values())
    b1 = s1["best"]
    worst = lambda d: part_name(d["least"][0].split(" | ")[0]) + " and " + part_name(d["least"][0].split(" | ")[1])  # noqa: E731
    v0 = F["variants"][0]
    return f"""<p>The PIP servo's base stands {tip - v0['servo_base_x']:.1f} mm palmar of the fingertip's face. The servo lies
along $x$ with its spline dorsal, and the crank, knuckle and rod fill the space behind it. A base flush with the fingertip
needs the servo 8 mm further dorsal, and no layout checked here clears with it (Table TAB[flush]). Even 2 mm costs the PIP
drive its last few degrees: it then closes only to {F['variants'][1]['pip_range_deg'][1]:.0f}&deg;.</p>
{tab}
<p>With the servo flush, the drive works better when its rest pose sits at PIP 20&deg;, the middle of the range: it then
gives {F['variants'][2]['torque_min_m18_92']:.2f} times the current hand's torque. The crank stack, 8 mm further back,
rises into the yoke's web at MCP &minus;15&deg;. A link 3 mm longer clears the web, but the rod's path from the knuckle to
the lug then crosses the end of the PIP servo's case. Turning the servo end for end, with its long end toward the MCP,
lowers the horn by 7 mm and clears the yoke without a longer link. The rod then sweeps through the 18 mm disc horn at the
extended end.</p>
<p>Two searches set the servo flush and varied the crank radius, the lug radius, the rest pose and the servo's direction.
{s1['close']} of {s1['combinations']} and {s2['close']} of {s2['combinations']} proportions close over the driver's range,
{s1['close_long_end_up'] + s2['close_long_end_up']} of them with the servo turned. With the long end toward the PIP, the
strongest drive brings together the {worst(b1['down'])}.{' Turned, it brings together the ' + worst(b1['up']) + '.' if 'up' in b1 else ''}
A flush servo therefore needs a different lower linkage, for example a lug beyond the servo's case on a longer arm of the
DIP link.</p>"""

def sec_clearance(J):
    least, moving = gaps(J)
    hw = J["check_fine"]["module_half_width_mm"]
    rows = []
    for k, v in least[:16]:
        a, b = k.split(" | ")
        note = " (designed)" if designed(k, v[0]) else (" (disc drawn as boxes; 1.9 mm to the round disc)" if boxed_disc(k) else "")
        rows.append([f"{part_name(a)} and {part_name(b)}" + note, f"{v[0]:.2f}"] + pose_cells(k, v))
    tab = table(["Pair", "Gap (mm)", "Yaw (deg)", "MCP (deg)", "PIP (deg)"], rows,
                "The sixteen closest pairs of parts over the 9,828 poses, with the pose of each pair's least gap. A joint that "
                "does not move the two parts relative to each other is marked any.", num_from=1)
    k0, v0 = moving[0]
    a0, b0 = k0.split(" | ")
    return f"""<p>Apart from the designed gaps, the closest parts are the {part_name(a0)} and the {part_name(b0)},
{v0[0]:.2f} mm apart (Table TAB[gaps]). The module's half-width is {hw[0]:.1f} mm, at the {part_name('x:' + hw[1][0])},
against the limit of {B['half_width']:.1f} mm. The PIP rod closes at every pose of the grid. H1 and R1 come closest at
&minus;25&deg; of PIP, 7&deg; past the driver's limit.</p>
{tab}"""


def sec_mjcf(J):
    mj = json.loads((DATA / "20261119-v4b_mjcf.json").read_text())
    mj["closure_mm_max"] = max(v["closure_mm_max"] for v in mj.values() if isinstance(v, dict) and "closure_mm_max" in v)
    fm = film(FILM, "The index and middle modules on their gantries, seen from the thumb's side. The carriages run to +30 and "
                    "&minus;30 mm, the yaw to &plusmn;60&deg;, the MCP from 90&deg; to &minus;15&deg; and the PIP from 92&deg; to "
                    "&minus;25&deg;. Two combined poses follow. Rendered from the MJCF with the joints set from the PIP drive's "
                    "solution; the palm and the thumb's gantry are hidden.")
    return f"""<p>Two scenes in <code>assets/mjcf/experimental/20261120-v4b_mixed/</code> run under physics:
<code>v4b_mixed_index.xml</code> has the index module on its gantry, and <code>v4b_mixed_pair.xml</code> adds the middle
module with the palm and the thumb's gantry. Each finger has four position actuators: stroke, yaw, MCP and the PIP
servo's horn. The PIP joint follows the horn through the knuckle's two hinges and a ball joint at the lug, an equality
constraint. The printed and acrylic parts are their FreeCAD solids; each printed piece is also a convex collision mesh,
and the servos are boxes. Parent and child bodies collide, so the yoke against the link and the carriage against the
yoke show as contacts.</p>
<pre class="cmd">.venv/bin/python -m mujoco.viewer --mjcf=assets/mjcf/experimental/20261120-v4b_mixed/v4b_mixed_pair.xml</pre>
<p>The keys home, flexed, abducted and curled set every joint, the linkage's included; the lug points coincide there to
{mj['closure_mm_max']:.3f} mm. Driven from home to three poses for 3 s each, the actuators came within
{mj['tracking_deg_max']:.1f}&deg; of their targets. The lug points parted by at most {mj['closure_mm_moving']:.1f} mm while
moving, and no two parts interpenetrated.</p>
{fm}"""


def sec_parts(J):
    pj, pr = J["parts"], J["printed"]["parts"]
    fig = img(MEDIA / "20261119-v4b_cut_sheet.png",
              f"The cut sheet, {pj['sheet_mm'][0]:.0f} by {pj['sheet_mm'][1]:.1f} mm: two of each acrylic part and the fit "
              "coupon C1. Red: cut outlines and holes. Blue: the engraved part labels.")
    rows = []
    for nm, rec in pr.items():
        if nm.startswith(("H1", "R1")):
            continue
        bb = rec["bbox_mm"]
        size = f"{bb[3] - bb[0]:.1f} &times; {bb[4] - bb[1]:.1f} &times; {bb[5] - bb[2]:.1f}"
        rows.append([nm.split(" ", 1)[0], nm.split(" ", 1)[1], f'<a href="data/{rec["file"]}.stl"><code>data/{rec["file"]}.stl</code></a>',
                     size, f"{rec['volume_mm3'] / 1000:.2f}"])
    tab = table(["Part", "Name", "STL", "Size (mm)", "Volume (cm<sup>3</sup>)"], rows,
                "The printed parts, each one closed solid in its body's frame at the zero pose (mm). Their STEP files are on the "
                "workstation in <code>assets/cad_v4/v4b_build/step/</code>.", num_from=3)
    rows2 = []
    for p in pj["parts"]:
        rows2.append([p["id"], p["name"].split(" ", 1)[1], str(p["qty"]), f"{p['size_mm'][0]:.1f} &times; {p['size_mm'][1]:.1f}",
                      str(p["holes"]), f"{p['least_web_mm']:.2f}"])
    tab2 = table(["Part", "Name", "Qty", "Size (mm)", "Holes", "Least web (mm)"], rows2,
                 "The laser-cut parts. The size is the part's bounding box on the sheet.", num_from=2)
    KO = J["keepout"]
    A0 = V.pip_kin(V.np.array([0.0]))[1]["A0"]
    rows3 = [["Cheeks", f"two, {B['dip_cheek_y'][1] - B['dip_cheek_y'][0]:.1f} mm thick, at $|y|$ = {B['dip_cheek_y'][0]:.1f} to "
              f"{B['dip_cheek_y'][1]:.1f} mm; inner faces {2 * B['dip_cheek_y'][0]:.1f} mm apart"],
             ["PIP pins", "a press fit for a 3&times;6 dowel through each cheek, on the PIP axis"],
             ["Outer bosses", "4 mm across, 0.3 mm tall, on each cheek's outer face; they bear on the PIP bearings' inner rings"],
             ["Lug", f"{B['pip']['lug_r']:.0f} mm from the PIP axis: at PIP 0, {A0[0]:.2f} mm dorsal and {-B['proximal'] - A0[2]:.2f} mm "
              "distal of the PIP axis"],
             ["Lug pin", "a press fit for a 3&times;12 dowel through both cheeks"],
             ["Inner bosses", "4 mm across, 1.3 mm tall, on each cheek's inner face; they clamp the spherical bearing's inner ring"],
             ["Rod's sweep", "no material inside <a href=\"data/20261119-v4b_dip_keepout.stl\"><code>data/20261119-v4b_dip_keepout.stl</code></a>, "
              f"the rod's boxes over the PIP's closing range in the DIP link's frame; it spans $y$ = {mn(KO['bounds_mm'][0][1], 2)} "
              f"to {KO['bounds_mm'][1][1]:.2f} mm"],
             ["Fingertip seat", "b v4's tongue, board seat and column below the PIP axis"]]
    tab3 = table(["Feature", "Dimension"], rows3, "The DIP link's interface with the link's PIP cheeks and the rod; unchanged "
                 "from the acrylic build relative to the PIP axis.", num_from=9)
    return f"""<p>The four printed parts and the print fit coupon are STL files in the folder (Table TAB[printed]). They come
from FreeCAD through <code>scripts/cad/v4b_freecad.py</code>: each part is the union of extruded outlines and boxes, with
its holes cut. Print the coupon P0 first. Its bores for the MF63ZZ run from 5.9 to 6.2 mm, and its dowel holes from 2.85 to
3.0 mm. Its M2 holes run from 1.7 to 1.9 mm and its bonding holes from 3.3 to 3.5 mm. The sizes chosen go into
<code>scripts/cad/v4b_build.py</code>, and <code>printed</code> rewrites the parts.</p>
{tab}
<p>The acrylic parts fit on one 1/8-inch sheet (<a href="cut/20261119-v4b_acrylic_3mm.dxf"><code>cut/20261119-v4b_acrylic_3mm.dxf</code></a>,
R12 in mm: cut outlines on the layer CUT, labels on ENGRAVE). Holes are drawn at nominal size: 6 mm for the MF63ZZ, 12 mm
for the spherical bearing and 2.2 mm for M2. Coupon C1 has bearing holes from 5.8 to 6.1 mm and from 11.8 to 12.0 mm.</p>
{fig}
{tab2}
<p>The DIP link is to be drawn from b v4 in Inventor, with b v4's fingertip seat on new cheeks (Table TAB[dip]).</p>
{tab3}"""


def sec_bom(J):
    head, *rows = J["bom"]
    body = [[esc(r[0]), r[1], esc(r[2]), esc(r[3]), esc(r[4]), esc(r[5])] for r in rows]
    tab = table(["Item", "Qty", "Specification", "Where", "Source", "Note"], body,
                "Bill of materials for one finger (<code>data/20261119-v4b_bom.csv</code>).", num_from=1)
    return f"""<p>Besides the three servos on hand, one finger is built from six MF63ZZ bearings and one spherical plain
bearing. Its pins are two shoulder screws and six dowels. The rest is common M2 and M3 hardware and epoxy. The tab nuts and the acrylic
cement of the acrylic build are gone.</p>
{tab}"""


def sec_assembly():
    return """<ol>
<li>Print the coupon P0 and cut C1. Choose the bores and holes that fit; set them in <code>scripts/cad/v4b_build.py</code>
and rerun <code>printed</code> and <code>dxf</code>. Print P1 to P4 and cut the sheet.</li>
<li>Press the MF63ZZs into the link's idler side and its two PIP cheeks, flanges toward the servo or the DIP link. Press
one into the yoke's rear arm, flange toward the back, and one each into H1 and R1, flanges toward the knuckle.
Bond the spherical bearing into R1.</li>
<li>Centre each servo (position 2048) before its disc horn goes on, so that the rest pose lies at mid-travel.</li>
<li>Slide the yaw servo up into the carriage's pocket and screw its upper tabs to the pocket's front wall (2 M2&times;6,
the driver through the rear wall's access holes). Fit its disc horn.</li>
<li>Bring the yoke up around the yaw servo and screw its front arm to the disc horn (4 M2&times;6) before the MCP servo
is in place. Push the rear dowel through the rear wall into the rear bearing and bond it in the wall with epoxy, the
servo at its centre.</li>
<li>Screw the MCP servo into the yoke's collar from the horn side (4 M2&times;5) and fit its disc horn.</li>
<li>Slide the PIP servo into the link's cradle from the dorsal side and screw its tabs to the tab frame (4 M2&times;5). Fit
its disc horn and screw H1 to it (2 M2&times;5).</li>
<li>Bring the link up around the MCP servo and screw its horn side to the MCP horn (4 M2&times;6). Press the 3&times;8 idler
dowel through the yoke's idler arm into the link's bearing.</li>
<li>Fit pin b through the knuckle and R1's bearing, then pin a through the knuckle and H1's bearing, each nut in its
bearing's bore recess.</li>
<li>Press the 3&times;12 lug dowel through the DIP link's cheeks and the spherical bearing. Press the two 3&times;6 dowels
through the link's PIP bearings into the cheeks on the PIP axis.</li>
<li>Mount the carriage on the gantry as the v3 carriage: the lead nut's flange on three M3 screws and the MGN7C block on
four M2 screws.</li>
</ol>"""


def sec_limits(J):
    return f"""<p>The check models parts as boxes and flat cylinders on a pose grid of 15&deg; in yaw, 3&deg; in MCP and
6&deg; in PIP. The DIP link is an envelope until it is drawn. The gantry's lower parts are boxes read from the owner's
CAD of 2026-11-04, and the carriage's interfaces copy the v3 carriage's holes; neither was measured on the hand. Where
the STS3032's cable leaves the case is not in Feetech's drawing: the design leaves 2.5 mm behind each servo's base and a
slot in the carriage's roof. The yaw servo's output and the rear dowel share the yoke's radial loads only if the dowel
is coaxial with the output. Bonding the dowel with the yoke in place sets that. Print accuracy, bearing fits, play and
friction are not measured; the clearances, loads and torques are from the model.</p>"""


def sec_future(J):
    return f"""<ul>
<li>Print P0 and cut C1; record the bores, press fits and M2 hole that fit, and rerun <code>printed</code> and
<code>dxf</code>.</li>
<li>Print the carriage first and offer it to the index gantry. Its column must take the lead nut and the MGN7C block
without shims, and the carriage must run the full stroke by hand. A carriage that binds or stops short refutes the
interfaces copied from the v3 carriage.</li>
<li>Draw the DIP link from b v4 with the interface of Table TAB[dip], replace its envelope in <code>solids()</code> with
boxes from its STL and rerun <code>check --fine</code>. A gap under 0.5 mm, other than the designed ones, refutes the
interface.</li>
<li>Assemble the finger and measure its play. Hold each joint at rest and load the pad with 2 N in each direction. Read
the pad's displacement with a dial gauge while the servos' positions are logged. Repeat with the rear dowel removed. A pad
displacement along $x$ no smaller with the dowel than without refutes the claim that the dowel stiffens the yaw stage.</li>
<li>Rerun the layout page's index-middle replay (<code>scripts/cad/cad_v4_layouts.py replay</code>) with this build's
{B['proximal']:.0f} mm link and the carriage, and the hand-object scale tasks with the longer finger.</li>
</ul>"""


def sec_files():
    rows = [["<code>scripts/cad/v4b_build.py</code>", "parameters, parts, clearance and gantry checks, PIP drive, loads, cut sheet, "
             "printed parts, bill of materials, MJCF and renders (<code>check --fine</code>, <code>stroke</code>, <code>pip</code>, "
             "<code>loads</code>, <code>dxf</code>, <code>printed</code>, <code>bom</code>, <code>mjcf</code>, <code>render</code>)"],
            ["<code>scripts/cad/v4b_freecad.py</code>", "the solids through FreeCAD: STEP and STL per part"],
            ["<code>scripts/cad/v4b_build_page.py</code>", "this page"],
            ["<code>scripts/cad/v4c_hardware.py</code>", "the PIP drive's kinematics"],
            ["<code>assets/mjcf/experimental/20261120-v4b_mixed/</code>", "the two playable scenes and their meshes"],
            ["<code>cut/20261119-v4b_acrylic_3mm.dxf</code>", "the cut sheet"],
            ["<code>data/20261119-v4b_bom.csv</code>", "the bill of materials"],
            ["<code>data/20261119-v4b_{carriage,yoke,link,knuckle,print_fit_coupon}.stl</code>", "the printed parts"],
            ["<code>data/20261119-v4b_dip_keepout.stl</code>", "the rod's sweep in the DIP link's frame, for drawing the DIP link"],
            ["<code>data/20261119-v4b_check_fine.json</code>, <code>data/20261119-v4b_stroke.json</code>",
             "the closest pairs, the module's half-width, the gantry clearances and the parameters"],
            ["<code>data/20261119-v4b_loads.json</code>", "the yaw stage's support forces and the levers from the lead nut"],
            ["<code>data/20261119-v4b_pip.json</code>", "the PIP drive along the PIP angle"],
            ["<code>data/20261119-v4b_parts.json</code>, <code>data/20261119-v4b_printed.json</code>", "acrylic part sizes and webs; "
             "printed part volumes and the pin stacks"],
            [f"<code>{FILM}</code>", "Film 1"]]
    return table(["File", "Content"], rows, "Files, relative to this page's folder unless they start with <code>scripts/</code> or "
                 "<code>assets/</code>.", num_from=9)


def lede_and_summary(J):
    _least, moving = gaps(J)
    L = J["loads"]["designs"]
    p3a, p3m = L["v4b acrylic (2026-11-19)"]["pinch_3N"], L["v4b mixed (this build)"]["pinch_3N"]
    st = min(v[0] for k, v in J["stroke"]["least"] if not k.split(" | ")[1].startswith("dip:"))
    lede = (f"The v4b index finger is now printed except for its PIP crank and rod, which stay acrylic. Its yaw servo sits "
            f"{B['yaw_spline_x']:.0f} mm behind the gimbal instead of 37 mm, because a {B['proximal'] - 60:.0f} mm longer proximal "
            f"link lowers the PIP crank under the yoke. The yoke wraps the yaw servo and turns on a rear dowel in the "
            f"carriage. A 3 N pinch then puts {p3m['horn_radial_N']:.1f} N on the servo's output and no bending moment. The "
            f"acrylic build bent that output with {p3a['bending_Nm_at_horn']:.2f} N&nbsp;m. A printed carriage bolts to the "
            f"gantry like v3's and clears it by {st:.1f} mm or more over the full stroke. Moving parts stay "
            f"{moving[0][1][0]:.2f} mm or more apart, and a playable MJCF of the index and middle modules is in the folder.")
    summary = (f"A mixed-manufacture v4b finger (printed bodies, acrylic crank and rod) puts its yaw servo {B['yaw_spline_x']:.0f} mm "
               f"behind the gimbal instead of 37 mm, with a rear dowel that leaves the servo's output without bending; a 3 N "
               f"pinch loads it with {p3m['horn_radial_N']:.1f} N. Its printed carriage clears the index gantry over the full "
               f"stroke, and an MJCF of the index and middle modules is in the folder.")
    return lede, summary


def main():
    J = load()
    sections = [("terms", "Terms and metrics", sec_terms(J)),
                ("make", "Printed bodies and an acrylic PIP linkage", sec_make(J)),
                ("yaw", "Yaw stage held at both ends of the yaw servo", sec_yaw(J)),
                ("carriage", "Carriage on the index gantry", sec_carriage(J)),
                ("layout", "Layout of the finger and its joints", sec_layout(J)),
                ("pip", "PIP drive through a knuckle and a spherical plain bearing", sec_pip(J)),
                ("flush", "PIP servo's base and the fingertip's palmar face", sec_flush(J)),
                ("clearance", "Clearance over the joint ranges", sec_clearance(J)),
                ("mjcf", "Playable MJCF of the index and middle modules", sec_mjcf(J)),
                ("parts", "Printed parts, cut sheet and the DIP link's interface", sec_parts(J)),
                ("bom", "Bill of materials", sec_bom(J)),
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
    figs = dict((cap[:30], n) for n, cap in re.findall(r"<figcaption>Figure (\d+)\. ([^<]{0,40})", html_all))
    ref = {"make": "How each part is made", "loads": "The yaw servo's output under", "stroke": "The closest gantry parts",
           "joints": "The joints and their pins", "gaps": "The sixteen closest pairs", "printed": "The printed parts, each",
           "dip": "The DIP link's interface", "flush": "The PIP servo's base moved toward"}
    ref = {k: next(n for c, n in caps.items() if c.startswith(v[:30])) for k, v in ref.items()}
    fref = {"side": next(n for c, n in figs.items() if c.startswith("Side view at yaw 0")),
            "assembly": next(n for c, n in figs.items() if c.startswith("The index module at four"))}
    body = [(sid, hd, re.sub(r"FIG\[(\w+)\]", lambda m_: fref[m_.group(1)], re.sub(r"TAB\[(\w+)\]", lambda m_: ref[m_.group(1)], b)))
            for sid, hd, b in body]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(body, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{b}</section>' for k, (sid, hd, b) in enumerate(body, 1))
    lede, summary = lede_and_summary(J)
    byline = ("Design, clearance checks and simulation in Python and FreeCAD; nothing has been printed, cut or assembled. "
              "This version replaces the all-acrylic build of 2026-11-19 (commit 823a33fbb). Builders: "
              "<code>scripts/cad/v4b_build.py</code>, <code>scripts/cad/v4b_build_page.py</code>.")
    page = TEMPLATE.read_text().replace("The cad_v3 page (scripts/cad/cad_v3_page.py)", "The v4b build page (scripts/cad/v4b_build_page.py)")
    for k_, v in {"TITLE": "v4b finger build", "H1": "Printed v4b index finger with an acrylic PIP linkage and its gantry carriage",
                  "EYEBROW": "Hand and bench, 2026-11-20", "LEDE": unit(lede), "BYLINE": byline, "TOC": toc, "SECTIONS": secs,
                  "FOOTER": str(OUT.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k_ + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in H.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = H.TEX_RE.sub(lambda m: next(svgs), page)
    page = docs_index.tag(page, topic="hardware", summary=summary, date="2026-11-20")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT}: {OUT.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
