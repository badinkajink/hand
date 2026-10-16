#!/usr/bin/env python3
"""Page: deployment readiness of the CAD hand with printed 85A magnet fingertips (study 20261016-sim2real_fingertip_readiness;
hand-off docs/handoff/20261016-sim2real_fingertip_readiness.md).

Every number is read from the study's rows through summary.json:

    .venv/bin/python scripts/sim2real_fingertip_analysis.py
    .venv/bin/python scripts/sim2real_fingertip_readiness_page.py
Out: docs/experiments/20261016-sim2real_fingertip_readiness/20261016-cad_hand_fingertip_deployment_readiness.html (media
inlined).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261016-sim2real_fingertip_readiness")
OUT = os.path.join(D, "20261016-cad_hand_fingertip_deployment_readiness.html")
TPL = os.path.join(ROOT, "scripts/sim2real_fingertip_readiness_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
FIG, TAB, REFS = [0], [0], {}
EXTRA_CSS = """
.tw th{white-space:normal !important;vertical-align:bottom;text-transform:none}
.tw td:first-child{min-width:9em}
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
figure.films video{width:100%;height:auto}
figure.img img{width:100%;height:auto}
.eq{margin:.8em 0;text-align:center}
""" + texsvg.BLOCK_CSS + "\n"
MM, UM, NN, DEG, MPA = "&#8202;mm", "&#8202;&#181;m", "&#8202;N", "&#176;", "&#8202;MPa"
VARIANTS = ["block_e10", "block4_g30", "block5_g30", "block4w1_g30", "cap_g30", "ell16_g30", "ell11_g30",
            "block4_g50", "block5_g50", "block4w1_g50", "cap_g50", "ell16_g50", "ell11_g50"]


# ------------------------------------------------------------------------------------------ helpers

def J(rel):
    p = os.path.join(D, rel)
    return json.load(open(p)) if os.path.exists(p) else None


def data_uri(path, mime=None):
    if mime is None:
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".mp4": "video/mp4"}[os.path.splitext(path)[1]]
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def jpeg_uri(path, quality=86, width=None):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def figure(body, caption, key=None, cls="img"):
    FIG[0] += 1
    if key:
        REFS[f"fig:{key}"] = FIG[0]
    return f'<figure class="{cls}">{body}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def img(rel, caption, key=None, width=None):
    p = os.path.join(D, rel)
    if not os.path.exists(p):
        raise SystemExit(f"missing figure {rel}")
    src = jpeg_uri(p, width=width) if width else data_uri(p)
    return figure(f'<img alt="" src="{src}">', caption, key)


def film(rel, poster_rel, caption, key=None):
    p = os.path.join(D, rel)
    if not os.path.exists(p):
        raise SystemExit(f"missing film {rel}")
    v = (f'<video controls muted loop playsinline preload="metadata" poster="{jpeg_uri(os.path.join(D, poster_rel))}" '
         f'src="{data_uri(p)}"></video>')
    return figure(v, caption, key, cls="films")


def table(head, body, caption, note=None, text_cols=(0,), key=None):
    TAB[0] += 1
    if key:
        REFS[f"tab:{key}"] = TAB[0]
    cls = lambda i: "" if i in text_cols else " class=num"  # noqa: E731
    out = [f"<p class='tcap'><b>Table&#160;{TAB[0]}.</b> {caption}</p>",
           "<div class='tw'><table><thead><tr>" + "".join(f"<th{cls(i)}>{h}</th>" for i, h in enumerate(head))
           + "</tr></thead><tbody>"]
    for r in body:
        out.append("<tr>" + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table></div>")
    if note:
        out.append(f"<p class='tnote'>{note}</p>")
    return "".join(out)


def render_tex(t):
    maths = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = texsvg.render(maths, cache_path=TEX_CACHE, scale=P.TEX_SCALE)
    it = iter(svgs)
    return P.TEX_RE.sub(lambda m: next(it), t), len(maths)


def f(x, n=2):
    return "&#8211;" if x is None else f"{x:.{n}f}"


def plain(v):
    """Sentence-case name of a variant (as scripts/sim2real_fingertip_cells.plain)."""
    shape = {"cap": "Capsule r 8.5&#8202;mm", "ell16": "Half-ellipsoid 16/32&#8202;mm", "ell11": "Half-ellipsoid 11/22&#8202;mm"}
    if v == "block_e10":
        return "Block, model default"
    base, inf = v.rsplit("_g", 1)
    mags = {"block4": ", four magnets", "block5": ", five magnets", "block4w1": ", four magnets, one wall"}.get(base, "")
    sh = "Block" if base.startswith("block") else shape[base]
    return f"{sh}{mags}, {inf}&#8202;%"


S = J("summary.json") or {}
MAN = J("manifest.json") or {}


# ------------------------------------------------------------------------------------------ terms

GLOSSARY = [
    ("CAD hand", "The SR2 hand assembled from the STLs of its printed parts on the real_v1 mounts "
     "(<code>assets/mjcf/experimental/20261013-cad_v1/</code>): MCP and PIP axes 30.80&#8202;mm apart against real_v1&#8217;s "
     "20.75&#8202;mm, 151 MorphIt collision spheres per finger, printed-part masses. The bench hand is built from these parts."),
    ("Variant", "One printable fingertip on the CAD hand: an outer shape (the block with 3&#8202;mm side and end fillets as printed "
     "in 0.2&#8202;mm layers, a capsule of radius 8.5&#8202;mm, or a half-ellipsoid 7.4&#8202;&#215;&#8202;8.5&#8202;mm in section and "
     "16/32 or 11/22&#8202;mm long distal/proximal of the apex), Siraya Tech Flex TPU 85A at 30 or 50&#8202;% gyroid, and four or "
     "five 3&#8202;&#215;&#8202;2&#8202;mm N52 magnets (block: also one perimeter at each pocket). The baseline is the block at the "
     "modelling default: 10&#8202;MPa pads, solid TPU mass, no magnets or board."),
    ("Solid modulus \\(E_s\\)", "Young&#8217;s modulus of the printed TPU itself in the finite elements; 12&#8202;MPa, the tensile "
     "modulus of NinjaFlex 85A, stands in for Siraya&#8217;s unmeasured value."),
    ("Pad modulus \\(E\\)", "The modulus of the 1&#8202;mm sphere pads (each pad&#8217;s stiffness \\(K_i = E a_i / h_i\\)) that "
     "reproduces the finite elements&#8217; approach of the whole printed tip under the 25&#8202;mm rod at 0.5, 1 and 3&#8202;N; for the "
     "curved tips the block&#8217;s value times the curved tip&#8217;s tool-fitted pad modulus over the block&#8217;s (15 October)."),
    ("Approach", "Travel of a rigid 25&#8202;mm diameter rod (the bench screwdriver&#8217;s shaft) into the tip from first touch at a "
     "given force, in &#181;m."),
    ("Twist-onset arm, spin arm, creep", "On the contact bed, two tips pinch the rod at 1&#8202;N each: the twist-onset arm is the "
     "torque at which the rod starts to turn under a slow torque ramp over \\(2\\mu N\\) (mm); the spin arm the mean torque that "
     "turns it at 1&#8202;rad/s over \\(2\\mu N\\) (mm); twist creep the turn rate at half the onset torque (&#176;/s); pull onset "
     "the force at which the rod starts to slide along its axis over \\(2N\\); rigid Coulomb friction gives zero creep."),
    ("Builders v1 and v2", "v1 gives all pads of a finger one stiffness with \\(h\\) = 8.5&#8202;mm; v2 (the repository default) gives "
     "each pad its own \\(h_i\\), the pad&#8217;s depth below the tip&#8217;s volume centroid."),
    ("Planner and trainer numerics", "Planner (and evaluation): MuJoCo 1&#8202;ms step, impratio 100, Newton solver with 100 "
     "iterations. Trainer: 2&#8202;ms, impratio 10, 10 iterations, the settings of the RL runs."),
    ("Servo plant", "Finger servos as position actuators with gain 0.72&#8202;N&#8202;m/rad, torque limit 0.18&#8202;N&#8202;m (80&#8202;% of "
     "the SCS0009&#8217;s 0.23&#8202;N&#8202;m stall at 6&#8202;V) and joint damping 0.0144&#8202;N&#8202;m&#8202;s/rad."),
    ("Held, held turn, three pads", "Held: at the end the cylinder is less than 20&#8202;mm below its height after the grip and at "
     "least two fingers each press on it with its weight. Held turn: the turn of the cylinder&#8217;s axis at the end when held, "
     "else 0. Three pads: all three fingers&#8217; pads press with at least the cylinder&#8217;s weight."),
    ("Cell, cell score", "Cell: one finger layout and cylinder of the scale study on the CAD hand (diameters 10, 16, 25&#8202;mm). "
     "Cell score: the median held HOM turn over placements 0&#8211;2 at the candidate grasp whose median is largest."),
    ("Board", "The ReSkin 5X magnetometer board: FR-4 20&#8202;&#215;&#8202;20&#8202;&#215;&#8202;0.6&#8202;mm, five MLX90393 chips "
     "(U1 at the centre, U2&#8211;U5 7&#8202;mm from it), a 10&#8202;&#215;&#8202;9&#8202;mm connector tab; in the scenes on the dorsal "
     "face of the DIP link&#8217;s 3.0&#8202;mm plate behind the tip, U1 over the contact point."),
    ("Force bound", "The Cram&#233;r&#8211;Rao bound of the normal force at the contact point from one 141&#8202;Hz sample of the five "
     "chips (GAIN_SEL 7, RES 0, OSR 0, DIG_FILT 5), the contact position and a uniform background field unknown (N)."),
    ("RMS tool-angle difference", "Two rollouts of one placement under two settings, sampled every 20&#8202;ms: the root mean "
     "square of the angle between the tool&#8217;s orientations (&#176;)."),
    ("Held cosine", "For the reorientation policies: the mean over held rollouts of the final cosine between the tool&#8217;s "
     "axis and vertical (+1 tip down, the goal)."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


# ------------------------------------------------------------------------------------------ tables from the summary

def print_table():
    rows = []
    for r in S.get("prints", []):
        if r["tip"] is None:
            lab = "Seat coupon" + (", one wall" if r["walls"] == 1 else "")
        else:
            lab = r["name"].split("siraya85a_")[1]
        rows.append([lab, f"{r['infill']}&#8202;%" if r["infill"] else "30&#8202;%", r["magnets"] or "&#8211;",
                     f(r["depth_mm"], 1) if r["depth_mm"] else "&#8211;", f(r["wall_min_mm"], 2) if r["wall_min_mm"] else "&#8211;",
                     ", ".join(f"{z:g}" for z in r["pauses"]) or "&#8211;", r["print_time"], f(r["flow_max"], 1)])
    return table(["Print", "Gyroid", "Magnets", "Magnet top under the apex (mm)", "Least TPU around a pocket (mm)",
                  "Pauses at (mm)", "Print time", "Largest flow (mm<sup>3</sup>/s)"], rows,
                 "The 85A print projects (print/, one sheet each).", key="prints")


def fe_table():
    rows = []
    for r in S.get("fe", []):
        lay = {"four_d3x2_d5": "Four magnets", "five_d3x2_d5": "Five magnets", "four_d3x2_d5_w1": "Four magnets, one wall"}[r["layout"]]
        a = r["approach_um"]
        rows.append([lay, r["recipe"][-2:] + "&#8202;%", f(r["E_star_MPa"], 2), f(a["0.5"], 1), f(a["1"], 1), f(a["3"], 1),
                     f"{r['face_normal_um_per_N'][0]:.0f}&#8211;{r['face_normal_um_per_N'][2]:.0f}",
                     f(r["magnet_travel_um_per_N"], 1)])
    return table(["Block", "Gyroid", "Pad modulus E (MPa)", "Approach at 0.5 N (&#181;m)", "At 1 N", "At 3 N",
                  "Face compliance over 15 points (&#181;m/N)", "Magnet travel (&#181;m/N)"], rows,
                 "Finite elements of the 85A magnet blocks at solid modulus 12&#8202;MPa (fe/fem.jsonl).", key="fe")


# ------------------------------------------------------------------------------------------ sections

def print_section():
    tc = J("print/template_changes.json") or {}
    ch = {c["key"]: c for c in tc.get("changed", [])}
    pr = [r for r in S.get("prints", []) if r["tip"] is not None]
    cp = [r for r in S.get("prints", []) if r["tip"] is None]
    t_block = [r["print_time"] for r in pr if r["tip"] == "block"]
    t_cur = [r["print_time"] for r in pr if r["tip"] != "block"]
    fl = [r["flow_max"] for r in S.get("prints", [])]
    return (f"<p>Fourteen PrusaSlicer projects print the 2026-10-16 magnet geometry in Siraya Tech Flex TPU 85A on extruder 2 of "
            f"the owner&#8217;s Prusa XL 5T: a seat coupon with two walls and one with one wall at each seat, and twelve tips, "
            f"the four-magnet, five-magnet and one-wall blocks and the three curved four-magnet tips, each at 30 and 50&#8202;% "
            f"gyroid (Table&#160;[[tab:prints]]). The template is the owner&#8217;s v17 project with Siraya&#8217;s Prusa XL "
            f"0.4&#8202;mm profile in filament slot 2: nozzle {ch.get('temperature', {}).get('siraya', '240')}&#8202;&#176;C, "
            f"bed {ch.get('bed_temperature', {}).get('siraya', '35')}&#8202;&#176;C, extrusion multiplier "
            f"{ch.get('extrusion_multiplier', {}).get('siraya', '0.96')}, at most "
            f"{ch.get('filament_max_volumetric_speed', {}).get('siraya', '12')}&#8202;mm<sup>3</sup>/s, 0.6&#8202;mm retraction, "
            f"and Siraya&#8217;s print speeds (perimeters {ch.get('perimeter_speed', {}).get('siraya', '60')}&#8202;mm/s, external "
            f"perimeters {ch.get('external_perimeter_speed', {}).get('siraya', '40')}&#8202;mm/s, bridges "
            f"{ch.get('bridge_speed', {}).get('siraya', '25')}&#8202;mm/s). The projects keep the template&#8217;s two perimeters "
            "and four top and four bottom layers, the recipe of every finite-element model and of the 1.8&#8202;mm pocket-wall "
            "rule, and its thin bridges: Siraya&#8217;s thick bridges lay a 0.4&#8202;mm strand whose underside hangs 0.2&#8202;mm "
            "below a 0.2&#8202;mm layer, onto a magnet that sits 0.1&#8211;0.2&#8202;mm under the bridge. "
            "<code>print/template_changes.json</code> lists every key taken over and every key kept.</p>"
            f"<p>PrusaSlicer 2.9.2 sliced every project. In each G-code the only tool selected is T1 (extruder 2), the part "
            f"prints at 240&#8202;&#176;C on a 35&#8202;&#176;C bed, the largest volumetric flow of any move on a part is "
            f"{max(fl):.1f}&#8202;mm<sup>3</sup>/s, every pocket is empty on every layer it spans, the layer under it covers "
            "its whole footprint, and the first layer over it is bridge infill printed right after its pause (M601). Each "
            "pocket has two perimeter loops in the two-wall prints and one in the one-wall prints. The blocks print in "
            f"{min(t_block)}&#8211;{max(t_block)} and the curved tips in {min(t_cur)}&#8211;{max(t_cur)} with two pauses each. "
            "The seats are 2.7&#8202;mm, 90&#8202;% of the magnet&#8217;s diameter: a provisional value until the coupon shows "
            "which of 2.7, 2.8 and 2.9&#8202;mm takes and keeps a magnet in 85A, and "
            "<code>scripts/sim2real_fingertip_prints.py build --seat 2.8</code> rewrites the tips.</p>" + print_table())


def readout_section():
    sens = (J("sensing/sensing.json") or {}).get("rows", [])

    def row(lay, rec, plane):
        return next(r for r in sens if r["layout"] == lay and r["recipe"] == rec and r["plane"] == plane)
    body = []
    for lay, lab in (("four_d3x2_d5", "Four"), ("five_d3x2_d5", "Five"), ("four_d3x2_d5_w1", "Four, one wall")):
        for rec in ("w85_30", "w85_50"):
            for plane, pl in (("touch", "Against the tip"), ("plate", "Behind the DIP plate")):
                r = row(lay, rec, plane)
                body.append([lab, rec[-2:] + "&#8202;%", pl, f(r["B_max_component_mT"], 2), str(r["gain_sel"]),
                             f(r["centre_force_N"], 3), f(r["median_force_N"], 3), f(sum(r["magnet_travel_1N_um"]) /
                                                                                  len(r["magnet_travel_1N_um"]), 1)])
    f5, f4 = row("five_d3x2_d5", "w85_30", "plate"), row("four_d3x2_d5", "w85_30", "plate")
    t5, t4 = row("five_d3x2_d5", "w85_30", "touch"), row("four_d3x2_d5", "w85_30", "touch")
    g5 = row("five_d3x2_d5", "w85_50", "plate")
    flip = [x for r in (f4, f5) for k in r["polarity_flip_dB_uT"] for x in k]
    return (f"<p>On the CAD finger the five-chip board cannot lie against the tip: the DIP link&#8217;s 3.0&#8202;mm dorsal plate "
            "fills the space behind the tip&#8217;s back face over its whole length and width (finger frame "
            "\\(x\\) = 6.45&#8211;9.45&#8202;mm). On the plate&#8217;s dorsal face, chips toward the plate, the chips&#8217; sensing "
            "points lie 3.42&#8202;mm behind the back face instead of 0.42&#8202;mm. There the force bound at the contact point "
            f"doubles: {f(f4['centre_force_N'], 2)}&#8202;N with four magnets and {f(f5['centre_force_N'], 2)}&#8202;N with five at "
            f"30&#8202;% gyroid, against {f(t4['centre_force_N'], 2)} and {f(t5['centre_force_N'], 2)}&#8202;N with the chips "
            f"against the tip (Table&#160;[[tab:readout]]). The fifth magnet, under the contact point and over U1, lowers the bound "
            f"{f4['centre_force_N'] / f5['centre_force_N']:.1f} times. At 50&#8202;% gyroid the tip is about twice as stiff and "
            f"the bound rises to {f(g5['centre_force_N'], 2)}&#8202;N with five magnets; only the 30&#8202;% prints move the "
            f"magnets 12&#8202;&#181;m/N or more, the travel the readout needs for 0.1&#8202;N per sample. Behind the plate "
            f"the static field is {f(f4['B_max_component_mT'], 1)}&#8202;mT (four) and {f(f5['B_max_component_mT'], 1)}&#8202;mT "
            "(five) on the largest axis, inside GAIN_SEL 7&#8217;s 4.8&#8202;mT range, so the bound uses the datasheet&#8217;s noise "
            "at the gain the board would run. A magnet mounted the wrong way round changes the field at a chip by "
            f"{min(flip) / 1e3:.1f}&#8211;{max(flip) / 1e3:.1f}&#8202;mT, a thousand times the noise, so the zero-load reading "
            "identifies it (procedure, Section&#160;[[sec:procedure]]).</p>"
            + table(["Magnets", "Gyroid", "Chips", "Largest field component (mT)", "GAIN_SEL", "Force bound at the contact "
                     "point (N)", "Median over 15 points (N)", "Magnet travel at 1 N (&#181;m/N)"], body,
                    "Readout of the 85A magnet blocks: point-dipole fields of the 3&#8202;&#215;&#8202;2&#8202;mm N52 magnets at "
                    "the five chips and the Cram&#233;r&#8211;Rao bound from one 141&#8202;Hz sample, the finite elements&#8217; "
                    "magnet motions at solid modulus 12&#8202;MPa (sensing/sensing.json). A softer Siraya print moves the "
                    "magnets further and lowers the bound in proportion: travel and bound scale as 12&#8202;MPa over the "
                    "solid modulus.", key="readout"))


def hand_section():
    sec = MAN.get("sections", {})
    fd, ms = sec.get("finger_dimensions", {}), sec.get("masses", {})
    var = S.get("variants", {})
    tips = [var[v]["tip_g"] for v in var if v != "block_e10"]
    prov = MAN.get("provenance_labels", {})
    n_cited = len(MAN.get("cited_files", {}))
    rows = []
    for v in VARIANTS:
        if v not in var:
            continue
        r = var[v]
        rows.append([plain(v), f(r["E_MPa"], 2), f(r["tpu_g"], 2), f(r["magnets_g"], 2), f(r["tip_g"], 2),
                     "yes" if r["board"] else "no"])
    return (f"<p>The bench hand is the SR2 hand built from its printed parts, whose MCP link spaces the MCP and PIP axes "
            f"{fd.get('mcp_axis_to_pip_axis', {}).get('value', 30.8)}&#8202;mm apart, 10.05&#8202;mm more than real_v1&#8217;s "
            "capsule model; every simulation on this page that names the CAD hand uses that assembly, the real_v1 mounts and "
            "workspace, and the servo plant. The D1&#8211;D8 bench plans and the D6 policies were made on real_v1&#8217;s "
            "shorter fingers, so their sections test the new tips on those hands. "
            f"<code>manifest.json</code> lists the parts, models, plans, policies, recipes and numerics this page rests on: "
            f"{n_cited} cited files, each with its size, SHA-256 and last commit, and every value labelled with its source as "
            + ", ".join(f"<i>{k}</i>" for k in prov) + ". The assembled finger is 85.2&#8202;g in the model against the "
            "owner&#8217;s estimate of 120&#8202;g; no part has been weighed.</p>"
            f"<p>Each variant (Table&#160;[[tab:variants]]) sets the tip&#8217;s mass from its 85A G-code (the filament extruded on "
            f"the part) plus 0.106&#8202;g per magnet, {min(tips):.2f}&#8211;{max(tips):.2f}&#8202;g against the 5.84&#8202;g of the "
            "model&#8217;s solid TPU tip, and adds the five-chip board to each DIP link as a collision box with its 0.64&#8202;g. "
            "The production real_v1 files are unchanged; the variant scenes are new files under "
            "<code>assets/mjcf/experimental/20261016-sim2real_fingertip_readiness/</code>.</p>"
            + table(["Variant", "Pad modulus E (MPa)", "Printed TPU (g)", "Magnets (g)", "Tip (g)", "Board"], rows,
                    "The CAD-hand variants (README.json in each variant&#8217;s folder).", key="variants"))


def procedure_section():
    pr = (J("bench/procedure.json") or {}).get("steps", [])
    body = []
    for st in pr:
        pred = st["predicts"]
        if st["id"] == "indentation":
            b = pred["block, four magnets, 30 % gyroid"]
            p_txt = (f"Four-magnet block, 30&#8202;%: {b['1 N']['E_s 12 MPa']:.0f}&#8202;&#181;m at 1&#8202;N for "
                     f"\\(E_s\\) = 12&#8202;MPa, {b['1 N']['E_s 6 MPa']:.0f} for 6, {b['1 N']['E_s 24 MPa']:.0f} for 24")
        elif st["id"] == "twist":
            p_txt = (f"Arm at 1&#8202;N: block {pred['block4_g30']['1']:.2f}&#8202;mm, capsule {pred['cap_g30']['1']:.2f}, "
                     f"half-ellipsoids {pred['ell16_g30']['1']:.2f} and {pred['ell11_g30']['1']:.2f}")
        elif st["id"] == "board":
            p_txt = (f"U1 {pred['five_magnets_plate_mT'][0][0]:.2f}&#8202;mT (five magnets), "
                     f"{pred['four_magnets_plate_mT'][0][0]:.2f} (four); one reversed magnet shifts a chip by 0.9&#8211;2.2&#8202;mT")
        elif st["id"] == "servo_step":
            p_txt = "Time constant 0.02&#8202;s; gain 0.72&#8202;N&#8202;m/rad; stall above 0.18&#8202;N&#8202;m"
        elif st["id"] == "pull":
            p_txt = "Onset over twice the pinch force equals the friction coefficient; the models use 1.0"
        elif st["id"] == "mass":
            p_txt = "Tips 3.2&#8211;3.7&#8202;g with magnets; finger 85.2&#8202;g in the model"
        elif st["id"] == "calibration":
            p_txt = (f"Magnet travel {pred['travel_um_per_N']['five'][0]:.1f}&#8202;&#181;m/N at the centre magnet; U1 "
                     f"{pred['signal_U1_uT_per_N']['five']:.0f}&#8202;&#181;T/N (five magnets, 30&#8202;%)")
        else:
            p_txt = pred if isinstance(pred, str) else ""
        body.append([st["id"].replace("_", " ").capitalize(), st["what"], p_txt, st["status"]])
    return ("<p>Every quantity that sets a model on this page is still unmeasured on the printed 85A tips. The measurements "
            "below, in order, replace the proxies; each comes with the value the models predict "
            "(<code>bench/procedure.json</code>). The indentation sets Siraya&#8217;s solid modulus: in linear elastic contact "
            "the response at force \\(F\\) and modulus \\(E_s\\) equals the 12&#8202;MPa response at \\(F \\cdot 12\\,\\mathrm{MPa}/E_s\\), "
            "so a measured approach \\(d\\) at force \\(F\\) gives \\(E_s = 12\\,\\mathrm{MPa}\; F / F_{12}(d)\\). The pull sets the "
            "friction coefficient independently of the modulus, and the twist onset then tests the contact patch.</p>"
            + table(["Check", "Procedure", "Model prediction", "Status"], body,
                    "Bench measurements before and after the first actuation.", text_cols=(0, 1, 2, 3), key="procedure"))


def bed_table():
    B = S.get("bed", {})
    fe = {(r["layout"], r["recipe"]): r for r in S.get("fe", [])}
    fe_of = {"block4_g30": ("four_d3x2_d5", "w85_30"), "block4_g50": ("four_d3x2_d5", "w85_50"),
             "block5_g30": ("five_d3x2_d5", "w85_30"), "block5_g50": ("five_d3x2_d5", "w85_50"),
             "block4w1_g30": ("four_d3x2_d5_w1", "w85_30"), "block4w1_g50": ("four_d3x2_d5_w1", "w85_50")}
    rows = []
    for v in VARIANTS:
        for arm, lab in (("v2", "Pads v2"), ("v1", "Pads v1"), ("drake", "Drake")):
            b = B.get(f"{v}|{arm}")
            if not b:
                continue
            a = b.get("approach_um", {})
            fe_a = fe.get(fe_of.get(v, ("", "")), {}).get("approach_um", {}) if arm == "v2" else {}
            rows.append([plain(v) if arm == "v2" else "", lab, f(b["E_MPa"], 2),
                         f(a.get("1"), 0) + (f" ({fe_a['1']:.0f})" if fe_a else ""),
                         f(a.get("3"), 0) + (f" ({fe_a['3']:.0f})" if fe_a else ""),
                         f((b.get("onset_arm_mm") or {}).get("1"), 2), f(b.get("spin_arm_mm"), 2),
                         f((b.get("twist_creep_deg_s") or {}).get("1"), 3), f(b.get("pull_mu_eff"), 3),
                         f(b.get("pull_creep_mm_s"), 4)])
    return table(["Variant", "Model", "E (MPa)", "Approach at 1 N, &#181;m (FE)", "At 3 N (FE)", "Twist-onset arm (mm)",
                  "Spin arm at 1 rad/s (mm)", "Twist creep (&#176;/s)", "Pull onset over 2N", "Pull creep (mm/s)"], rows,
                 "Contact on the bed at 1&#8202;N per tip unless stated, friction 1.0, planner numerics; in brackets the finite "
                 "elements&#8217; approach of the printed block (bed/*.jsonl, fe/fem.jsonl).", text_cols=(0, 1), key="bed")


def stiffness_section():
    B = S.get("bed", {})
    fe = {(r["layout"], r["recipe"]): r for r in S.get("fe", [])}
    e30 = [fe[(l, "w85_30")]["E_star_MPa"] for l in ("four_d3x2_d5", "five_d3x2_d5", "four_d3x2_d5_w1") if (l, "w85_30") in fe]
    e50 = [fe[(l, "w85_50")]["E_star_MPa"] for l in ("four_d3x2_d5", "five_d3x2_d5", "four_d3x2_d5_w1") if (l, "w85_50") in fe]
    g = lambda v, arm, k, n="1": ((B.get(f"{v}|{arm}") or {}).get(k) or {}).get(n)  # noqa: E731
    half = (B.get("block4_g30|E0.5x") or {}).get("approach_um", {})
    dbl = (B.get("block4_g30|E2x") or {}).get("approach_um", {})
    rl = B.get("block4_g30|rl") or {}
    creep_ratio = [g(v, "v2", "twist_creep_deg_s") / g(v, "drake", "twist_creep_deg_s") for v in VARIANTS
                   if g(v, "v2", "twist_creep_deg_s") and g(v, "drake", "twist_creep_deg_s")]
    txt = (f"<p>At a solid modulus of 12&#8202;MPa the 85A blocks take a pad modulus of {min(e30):.2f}&#8211;{max(e30):.2f}&#8202;MPa "
           f"at 30&#8202;% gyroid and {min(e50):.2f}&#8211;{max(e50):.2f}&#8202;MPa at 50&#8202;% (Table&#160;[[tab:fe]]); the fifth "
           "magnet stiffens the face by 6&#8202;% and one wall at the pockets softens it by 1&#8202;%. 30&#8202;% gyroid in 85A is "
           "about half as stiff as the 95A, 20&#8202;% prints of 13 October (16.7&#8202;MPa), and 50&#8202;% about as stiff. Every "
           "displacement scales as 12&#8202;MPa over the solid modulus, so Siraya&#8217;s value moves all of these numbers in "
           "proportion.</p>"
           + fe_table()
           + f"<p>On the contact bed, with each variant&#8217;s pads at its finite-element modulus, builders v1 and v2 and Drake "
           f"agree on the approach within a few percent (block, 30&#8202;%: {f(g('block4_g30', 'v2', 'approach_um'), 0)}, "
           f"{f(g('block4_g30', 'v1', 'approach_um'), 0)} and {f(g('block4_g30', 'drake', 'approach_um'), 0)}&#8202;&#181;m at "
           "1&#8202;N, against 61&#8202;&#181;m in the finite elements) and on the twist-onset arm (2.5&#8211;2.6&#8202;mm on the "
           "blocks, 0.7&#8211;0.9&#8202;mm on the curved tips), and the pull onset equals the friction coefficient in every model "
           "(Table&#160;[[tab:bed]]). They separate on sliding: the pads&#8217; twist creep is "
           f"{min(creep_ratio):.0f}&#8211;{max(creep_ratio):.0f} times Drake&#8217;s, and their spin arm on the blocks 1.2&#8211;1.5 "
           "times, the rate effect of MuJoCo&#8217;s elliptic cone (14 October). At the RL trainer&#8217;s numerics (2&#8202;ms, "
           f"impratio 10) the block&#8217;s pads give the same approach and onset arm but {f((rl.get('twist_creep_deg_s') or {}).get('1'), 2)}"
           f"&#8202;&#176;/s of twist creep and {f(rl.get('pull_creep_mm_s'), 3)}&#8202;mm/s of pull creep, ten times the "
           "planner&#8217;s: impratio and step size change the sliding, not the stiffness.</p>"
           + bed_table()
           + "<p>The approach is the measurement that sets the modulus. Halving the pad modulus of the 30&#8202;% block raises "
           f"the approach at 1&#8202;N from {f(g('block4_g30', 'v2', 'approach_um'), 0)} to {f(half.get('1'), 0)}&#8202;&#181;m"
           + (f" and doubling it lowers it to {f(dbl.get('1'), 0)}&#8202;&#181;m" if dbl else "")
           + ", while the onset arm changes by a few percent and the pull onset not at all. A rod indentation at 0.5, 1 and "
           "3&#8202;N on a printed block therefore fixes Siraya&#8217;s solid modulus, a two-tip pull fixes the friction "
           "coefficient, and the twist onset checks the patch model (Section&#160;[[sec:procedure]]).</p>")
    return txt


def main():
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "GLOSSARY": glossary()}
    raise SystemExit("draft: sections still to be written (see the hand-off note)")


if __name__ == "__main__":
    main()
