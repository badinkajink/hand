#!/usr/bin/env python3
"""Print sheet of a DIP-link fit-test run, read at the printers on a phone: the two plates, every piece with its printed
ID, the checks in order, the parameter each check sets, and the owner's records.

The page is built from the run folder that scripts/dip_fit_print_plates.py writes (pieces.json, the plate maps, the XL
project's report, records.csv) and regenerates when they change; filled rows of records.csv appear in Section 5.

    .venv/bin/python scripts/dip_fit_print_plates_page.py [--run 20261019b]
"""
from __future__ import annotations

import base64
import csv
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

RUN = ROOT / "docs/experiments/20261019-dip_link_fit_prints"
PAGE = RUN / "20261019-dip_link_fit_prints.html"
TEMPLATE = ROOT / "scripts/dip_fit_print_plates_page.template.html"
STUDY = "docs/experiments/20261018-dip_link_board_mount"
PARAMS = f"{STUDY}/params/20261018-v2_concept_a.json"
CONCEPT = "a"
RUNS = {"20261019": ("20261019-dip_link_fit_prints", "a"), "20261019b": ("20261019-dip_link_fit_prints_b", "b")}


def set_run(rid):
    """Point RUN, PAGE, PARAMS and CONCEPT at a run of scripts/dip_fit_print_plates.py."""
    global RUN, PAGE, PARAMS, CONCEPT
    folder, CONCEPT = RUNS[rid]
    RUN = ROOT / "docs/experiments" / folder
    PAGE = RUN / f"{folder}.html"
    PARAMS = f"{STUDY}/params/20261018-v2_concept_{CONCEPT}.json"


def rel(p):
    return str(Path(p).relative_to(ROOT))


def esc(s):
    return html.escape(str(s), quote=True)


def img_uri(png, width=1100):
    from PIL import Image
    im = Image.open(png).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.quantize(colors=64).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), im.width, im.height


FIG = [0]


def figure(png, caption, alt):
    uri, w, h = img_uri(png)
    FIG[0] += 1
    return (f'<figure><img src="{uri}" width="{w}" height="{h}" alt="{esc(alt)}">'
            f"<figcaption>Figure {FIG[0]}. {caption}</figcaption></figure>")


def code(s):
    return f"<code>{esc(s)}</code>"


def hmin(t):
    """PrusaSlicer's '1h 4m 3s' or '14m 3s' as '1 h 4 min' or '14 min'."""
    if " min" in t:
        return t
    m = sum(float(x[:-1]) * {"d": 1440, "h": 60, "m": 1, "s": 1 / 60}[x[-1]] for x in t.split())
    m = round(m)
    return f"{m // 60} h {m % 60} min" if m >= 60 else f"{m} min"


def files_section(R, p1s, xl, n_pla, n_tpu, a1m=None, p85=None):
    ov = p1s["overrides"]
    xs = xl["settings"]
    inside = "tip" if CONCEPT == "a" else "pad"
    if p85:
        fo, sp, rc = p85["filament_settings"], p85["speeds_mm_s"], p85["recipe"]
        n85 = len(p85["objects"])
        tpu_txt = (
            f"The 85A blocks of both runs print on one Bambu Lab P1S plate from its external spool: "
            f"{code(p85['project'])} ({esc(p85['time'])}, {p85['filament_g']:.1f} g, {n85} blocks), run a&#8217;s channel "
            "blocks F00 to F40 in the front row and run b&#8217;s groove blocks G00 to G20 behind them. In Bambu "
            "Studio&#8217;s print dialog switch the AMS off, so that the printer feeds from the external spool; the PLA "
            "plate then prints from AMS slot 1. The A1 mini projects printed badly (owner, 18:05), with Siraya Tech&#8217;s "
            "A1 mini values and with a generic TPU profile alike, and stay in the folder as a record. The P1S project uses "
            f"every filament value of Siraya Tech&#8217;s P1S profile ({code(p85['filament_source'])}): "
            f"{fo['nozzle_temperature'][0]} &#176;C, a {fo['textured_plate_temp'][0]} &#176;C {esc(p85['bed_type'])}, flow "
            f"{fo['filament_flow_ratio'][0]}, {fo['filament_retraction_length'][0]} mm retraction, the fan from layer "
            f"{int(fo['close_fan_the_first_x_layers'][0]) + 1}, a cap of {fo['filament_max_volumetric_speed'][0]} mm&#179;/s "
            "and no slowdown for layer cooling, and the speeds of its process: outer walls "
            f"{sp['outer_wall_speed']:g} mm/s, inner walls {sp['inner_wall_speed']:g}, infill {sp['sparse_infill_speed']:g}, "
            f"top surfaces {sp['top_surface_speed']:g}, the first layer {sp['initial_layer_speed']:g}, and a "
            f"{p85['skirt_loops']}-loop skirt. The structure is the recipe of the finite-element models, the same on every "
            f"printer: {rc['wall_loops']} walls, {rc['sparse_infill_density']} gyroid, {rc['top_shell_layers']} skins of "
            f"{rc['layer_height']} mm, {rc['elefant_foot_compensation']} mm elephant-foot compensation, no brim, a minimum "
            f"sparse infill area of {rc['minimum_sparse_infill_area']} and a minimum top shell of "
            f"{rc['top_shell_thickness']} mm (Siraya&#8217;s process has 4 walls, 20 % gyroid and 5 top skins). The "
            f"interference chosen from these blocks applies to {inside}s printed on the P1S with this profile. When the XL "
            f"is free, its project prints the same blocks on the v17 template: tool 2, {xs['layer_height']} mm layers, "
            f"{xs['perimeters']} perimeters, {xs['fill_density']} gyroid, no pause.")
    elif a1m:
        fo, po = a1m["filament_settings"], a1m["overrides"]
        tpu_txt = (
            f"The 85A blocks print on the {esc(a1m['printer'])}, whose AMS lite feeds the Siraya Tech 85A through slot "
            f"{a1m['slot']}. The TPU is the project&#8217;s only filament: in Bambu Studio&#8217;s print dialog, map it to "
            f"slot {a1m['slot']} (or to the external spool if Bambu Studio refuses TPU in the AMS lite). The filament "
            f"settings are Siraya Tech&#8217;s own A1 mini profile for this TPU ({code(a1m['filament_source'])}): "
            f"{fo['nozzle_temperature'][0]} &#176;C, a {fo['textured_plate_temp'][0]} &#176;C {esc(a1m['bed_type'])}, flow "
            f"{fo['filament_flow_ratio'][0]}, {fo['filament_retraction_length'][0]} mm retraction at "
            f"{fo['filament_retraction_speed'][0]} mm/s, the fan from layer {int(fo['close_fan_the_first_x_layers'][0]) + 1}. "
            f"One value differs: the volumetric cap is {fo['filament_max_volumetric_speed'][0]} mm&#179;/s against "
            "Siraya&#8217;s 8 (about 24 mm/s against 80 to 100), because the 85A reaches the extruder through the AMS lite "
            "tube. The process keeps the recipe of the finite-element models and the XL prints, the same on every printer: "
            f"{po['wall_loops']} walls, {po['sparse_infill_density']} gyroid, {po['top_shell_layers']} skins of "
            f"{po['layer_height']} mm, {po['elefant_foot_compensation']} mm elephant-foot compensation, no brim "
            "(Siraya&#8217;s process has 3 walls and 25 % gyroid). Two Bambu defaults take the XL projects&#8217; values so "
            f"that these small blocks print that recipe: the minimum sparse infill area is {po['minimum_sparse_infill_area']} "
            "mm&#178; against Bambu&#8217;s 15, which made every gyroid island of a block solid zig-zag infill, and the "
            f"minimum top shell is {po['top_shell_thickness']} mm against Bambu&#8217;s 1.0, which added a fifth top skin. "
            "The interference chosen from these blocks applies to "
            f"{inside}s printed on the A1 mini with this profile and the same 0.2 mm elephant-foot compensation; the XL "
            "prints its channels to its own widths. When the XL is free, its project prints the same blocks on the v17 "
            f"template: tool 2, {xs['layer_height']} mm layers, {xs['perimeters']} perimeters, {xs['fill_density']} "
            "gyroid, no pause.")
    else:
        tpu_txt = ("The XL project prints on the v17 template as the 85A fingertips did: tool 2, "
                   f"{xs['layer_height']} mm layers, {xs['perimeters']} perimeters, {xs['fill_density']} gyroid, no pause.")
    rows = (f"<tr><td>Bambu Lab P1S, PLA</td><td>{code(Path(p1s['project']).name)}</td><td>{n_pla} "
            f"({len(p1s['objects'])} objects)</td><td>{esc(p1s['time'])}</td><td>{p1s['filament_g']:.1f} g</td></tr>")
    if p85:
        rows += (f"<tr><td>Bambu Lab P1S, 85A from the external spool, both runs</td><td>{code(Path(p85['project']).name)}"
                 f"</td><td>{len(p85['objects'])}</td><td>{esc(p85['time'])}</td><td>{p85['filament_g']:.1f} g</td></tr>")
    if a1m:
        rows += (f"<tr><td>Bambu Lab A1 mini, 85A in slot {a1m['slot']}{' (printed badly)' if p85 else ''}</td>"
                 f"<td>{code(Path(a1m['project']).name)}</td>"
                 f"<td>{n_tpu}</td><td>{esc(a1m['time'])}</td><td>{a1m['filament_g']:.1f} g</td></tr>")
    rows += (f"<tr><td>Prusa XL 5T, tool 2, 85A{' (when free)' if a1m else ''}</td><td>{code(Path(xl['project']).name)}"
             f"</td><td>{n_tpu}</td><td>{esc(hmin(xl['time']))}</td><td>{xl['filament_g']:.1f} g</td></tr>")
    tmap = (figure(ROOT / p85["map"], "The P1S 85A plate of both runs from above, front edge at the bottom; each "
                   "block&#8217;s ID is engraved in its top face, which lies inside the tip (F) or the pad (G).",
                   "P1S 85A plate map with piece IDs") if p85 else
            figure(ROOT / a1m["map"], f"The A1 mini plate from above; each block&#8217;s ID is engraved in its top face, "
                   f"which lies inside the {inside}.", "A1 mini plate map with piece IDs") if a1m else
            figure(ROOT / xl["map"], f"The XL plate from above; each block&#8217;s ID is engraved in its top face, which "
                   f"lies inside the {inside}.", "XL plate map with piece IDs"))
    return (
        '<section id="files"><h2>1. Files and settings</h2>'
        f"<p>All files are in {code(rel(RUN) + '/')}. Open the P1S project in Bambu Studio and print it as it is: it is "
        f"sliced, with the G-code inside, for the {esc(p1s['printer'])} with {esc(p1s['filament'])} on the "
        f"{esc(p1s['bed_type'])} at {p1s['bed_c']} &#176;C. The process is {esc(p1s['presets']['process'])} changed to "
        f"{ov['layer_height']} mm layers, {ov['wall_loops']} walls, {ov['sparse_infill_density']} rectilinear infill, the "
        "seam at the back, an automatic brim and no supports; Bambu Studio marks these values as changed.</p>"
        f"<p>{tpu_txt}</p>"
        '<div class="tw"><table class="facts"><tr><th>Plate</th><th>File</th><th>Pieces</th><th>Time</th>'
        f"<th>Filament</th></tr>{rows}</table></div>"
        f"<p>Each piece's STL, in print orientation on the bed and named by its ID, is in {code(rel(RUN / 'stl') + '/')}, "
        "for printing one piece again.</p>"
        + figure(ROOT / p1s["map"], "The P1S plate from above, front edge at the bottom. Bold: the ID printed on the "
                 "piece; below: its variant.", "P1S plate map with piece IDs")
        + tmap + "</section>")


def ids_section(pieces):
    rows = "".join(
        f"<tr><td>{esc(s['id'])}</td><td>{esc(s['piece'])}{' (2)' if s.get('copies', 1) > 1 else ''}</td>"
        f"<td>{esc(s['variant'])}</td><td>{esc(s['mark'])}</td>"
        f"<td>{dict(p1s='P1S', a1m='A1 mini', xl='XL').get(s['printer'], s['printer'])}</td></tr>"
        for s in pieces)
    note = ("The digits of K, F and C give the variant in hundredths of a millimetre: K25 is the key with a 0.25 mm "
            "detent, F10 the channel block 0.10 mm narrower per flank, C30 the carrier with 0.30 mm chip windows."
            if CONCEPT == "a" else
            "The digits of G and S give the variant in hundredths of a millimetre: G10 is the groove block 0.10 mm "
            "narrower per flank, S45 the tongue with a 0.45 mm deep seat pocket.")
    return ('<section id="ids"><h2>2. Pieces and their IDs</h2>'
            f"<p>{note}</p>"
            '<div class="tw"><table class="ids"><tr><th>ID</th><th>Piece</th><th>Variant</th><th>How to tell it</th>'
            f"<th>Printer</th></tr>{rows}</table></div></section>")


def terms_section():
    T = [("Test piece", "A small part printed by the same process as the real part, to decide one fit before the whole "
          "DIP link is printed; in materials testing it is called a coupon."),
         ("Key and channel", "The PLA dovetail key on the DIP link&#8217;s back face and the 85A channel in the "
          "tip&#8217;s back that slides over it, from the key&#8217;s free end until the tip stops on the flange."),
         ("Flank interference", "How much narrower the channel is than the key on each flank, in mm; 0 is a "
          "line-to-line fit."),
         ("Detent height", "Height of the bump on the key&#8217;s head that clicks into the groove across the channel, "
          "in mm; the design has 0.25."),
         ("Chip-window clearance", "Gap on each side between an MLX90393 chip (3 mm square) and its window in the "
          "0.85 mm window plate, in mm; the design has 0.20."),
         ("Self-forming hole", "A hole smaller than the screw, in which the screw cuts its own thread in PLA; the "
          "owner&#8217;s rule of 2026-10-18 is 1.8 mm for M2 and 2.75 mm for M3."),
         ("Groove width", "Width of the open groove that guides the Qwiic cable, in mm; the design has 2.4 for a "
          "2.0 mm bundle, with turns of 2.0 mm centre-line radius."),
         ("F*, C*", "The channel block and the carrier chosen in the first round of a check, used in its second.")]
    dl = "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in T)
    return f'<section id="terms"><h2>3. Terms</h2><dl class="glossary">{dl}</dl></section>'


def checks_section():
    return (
        '<section id="checks"><h2>4. Checks, in order</h2>'
        "<h3>4.1 Snap fit: K and F</h3>"
        "<p>Round 1: slide each F block onto K25 by hand, F00 first. Record whether it slides on, whether the detent "
        "clicks, and whether the block stays when pulled along the channel and when twisted. F* is the block with the "
        "largest interference that still slides on by hand and stays when twisted. Round 2: K15, K35 and K45 in F*; the "
        f"smallest detent that clicks and keeps the block in place sets {code('detent_h')}, and F* sets {code('fit_flank')}.</p>"
        "<h3>4.2 Board fit: C, CP and P</h3>"
        "<p>Lay the ReSkin 5X board component side down on the window plate of C15, C20 and C30. C* is the carrier "
        "with the smallest clearance in which all five chips drop into their windows and the board lies flat without "
        f"rocking; it sets {code('clear_chip')}. On C*, mate the Qwiic plug to J1: J1 and the plug sit in their slot "
        "without touching its walls, and the cable leaves through its groove. Put on a clamp plate (CP) so that its two "
        "pins pass the board&#8217;s proximal holes into the window plate, then drive two M2 &#215; 4 countersunk screws "
        "into the clamp plate&#8217;s 1.8 mm holes and two M2 &#215; 5 screws through its ears into the shoulders. On "
        f"P, the largest pin that enters the board&#8217;s 2.40 mm hole without play sets {code('pin_d')} (design "
        "2.35 mm).</p>"
        "<h3>4.3 Screw holes: H</h3>"
        "<p>Drive an M2 &#215; 5 into each M2 hole of the top row and of the front face (the front holes have the sizes "
        "engraved above them) and an M3 &#215; 12 into each M3 hole; the M3 holes go through, so hold the piece over "
        "an edge. The smallest hole that takes the screw without cracking and does not let it pull out sets "
        f"{code('screw_thread_d')} (clamp plate) and {code('m2_pilot')} (shoulders) for prints on the P1S, and "
        "checks the owner&#8217;s 2.75 mm for M3.</p>"
        "<h3>4.4 Cable groove: W</h3>"
        "<p>Press the Qwiic cable into each groove of W and around its turn. The narrowest groove that takes the cable "
        f"below the surface without forcing sets the groove width, the design&#8217;s {code('cable_d')} plus twice "
        f"{code('cable_clear')} (2.0 + 2 &#215; 0.20 = 2.4 mm).</p></section>")


def records_section(rows):
    filled = [r for r in rows if (r.get("result") or "").strip()]
    body = "".join(f"<tr><td>{esc(r['id'])}</td><td>{esc(r['test'])}</td><td>{esc(r['check'])}</td>"
                   f"<td>{esc(r.get('result', ''))}</td><td>{esc(r.get('notes', ''))}</td></tr>" for r in rows)
    counts = {}
    for r in rows:
        k = r["test"].split(",")[0].split(" (")[0]
        counts[k] = counts.get(k, 0) + 1
    return ('<section id="record"><h2>5. Record</h2>'
            f"<p>{code(rel(RUN / 'records.csv'))} has one row per check, keyed by ID, {len(rows)} rows: "
            + ", ".join(f"{esc(k.lower())} {v}" for k, v in counts.items())
            + ". Write y, n or a number in result and anything seen in notes; "
            + ("a pairing is written K25+F10, and F* and C* stand for the choices of round 1. " if CONCEPT == "a" else
               "a pairing is written R+G10 or S45+V. ")
            + f"{len(filled)} of {len(rows)} rows are filled.</p>"
            f"<details><summary>All {len(rows)} rows</summary><div class=\"tw\"><table class=\"rec\"><tr><th>ID</th>"
            f"<th>Test</th><th>Check</th><th>Result</th><th>Notes</th></tr>{body}</table></div></details></section>")


A1M_SOLID = (
    "<li>The A1 mini project of 16:55 printed every 85A block solid: Bambu&#8217;s minimum sparse infill area of "
    "15 mm&#178; turned each gyroid island inside a block&#8217;s walls into zig-zag infill, and its 1.0 mm minimum top "
    "shell added a fifth top skin. Since 17:06 the project sets both as the XL projects do (0 mm&#178; and 0.7 mm), and "
    "the build stops unless every block prints gyroid. Blocks printed from the 16:55 project have the same IDs and are "
    "stiffer behind their walls than the tip or pad they test; record the fits from blocks of the 17:06 project.</li>")


def plug_slot_item():
    return ("<li>The neck carrier&#8217;s J1 slot is 7.8 mm wide since 17:42 (6.6 mm before). The Qwiic cable&#8217;s plug "
            "is JST&#8217;s SHR-04V-S-B, 7.0 mm wide at its protrusions, where the model had the 5.0 mm SHR-04V-S. The plug "
            "did not enter run a&#8217;s 6.6 mm slot (owner). The slot clears the plug by 0.4 mm a side and J1 by "
            "0.9 mm.</li>")


def cable_item():
    return ("<li>Since 17:57 the neck carrier&#8217;s cable groove is 3.0 mm wide (2.4 mm before). The Qwiic cable is a flat "
            "ribbon about 4 mm wide and 1 mm thick, which had to be squished into the 2.2 to 2.6 mm grooves of run "
            "a&#8217;s W (owner). Massaged into a bundle, it is modelled as 2.6 mm (2.0 mm before). To keep "
            "R7&#8217;s 1.0 mm from the MCP link, the first turn, toward the back right out of the plug, has a 1.5 mm radius "
            "(2.0 mm before; the ribbon still lies flat there and bends flatwise): 1.24 mm. W is new: grooves 2.8, 3.0 and "
            "3.4 mm wide, 3.0 mm deep, each with a 2.0 mm turn.</li>")


P1S_85A = (
    "<li>Since 18:13 the 85A blocks of both runs print on one P1S plate from its external spool, run a&#8217;s before "
    "run b&#8217;s PLA plate (owner, 18:05: the A1 mini prints went badly with Siraya Tech&#8217;s A1 mini values and "
    "with a generic TPU profile alike; 18:10: one plate for both, then the PLA). The plate uses Siraya Tech&#8217;s P1S "
    "profile, its filament values and its process speeds (45 to 60 mm/s; the A1 mini projects ran Bambu&#8217;s 200 to "
    "300 mm/s process capped at 2 mm&#179;/s). It also uses the finite-element models&#8217; structure.</li>")


SCREW_HOLES_B = (
    "<li>The tongue&#8217;s four M2 holes in S and N are 1.9 mm since 18:30 (1.8 mm before). In run a&#8217;s clamp plate "
    "the M2 screws did not form their thread in 1.8 mm holes, although 1.8 mm was fine in the hole piece H (owner).</li>")


def changes_section():
    return (
        '<section id="changes"><h2>7. Changes from the pieces of 13:08</h2>'
        f"<p>The set in {code(STUDY + '/v2/coupons/')} was four XL projects with PLA on tool 3. This run differs in "
        "eight ways.</p><ul>"
        "<li>One plate per printer: the PLA pieces on the P1S, the 85A blocks on the A1 mini (from 16:40, while the XL "
        "is in use) or the XL.</li>"
        "<li>Every piece has its ID printed on it; the three window gauges had none.</li>"
        "<li>The detent sweep reaches 0.45 mm and the flank sweep 0.30 and 0.40 mm, so that each choice can fall "
        "inside its sweep.</li>"
        "<li>Three carriers, one per chip-window clearance, replace the window gauges; a carrier contains the "
        "gauge&#8217;s window plate and adds J1, the cable groove, the clamp plate and the screws. The pin, screw-hole "
        "and cable-groove pieces P, H and W join them.</li>"
        "<li>The carrier is cut at the bridge&#8217;s flat, which left it standing on two 0.8 mm strips under the "
        "bridge&#8217;s curved face; that face reaches the full section 1.8 mm above the bed. The plates fill it down to "
        "the bed (199.5 mm&#179;); the J1 plug, the cable and the board lie outside the fill.</li>"
        "<li>The first P1S project of this run started with Bambu Studio&#8217;s generic start G-code: it loaded no "
        "filament from the AMS and left the nozzle at 205 &#176;C. Since 16:40 every Bambu project has the printer&#8217;s "
        "own start, end and filament-change G-code, and the build stops unless the G-code loads from the AMS and heats to "
        "the filament&#8217;s temperatures.</li>" + A1M_SOLID + P1S_85A + "</ul></section>")


def results_section_a():
    return (
        '<section id="results"><h2>6. Results so far</h2>'
        "<p>Results the owner reported by 18:30, and what each changed in v2 a (the record of Section 5 has them):</p><ul>"
        "<li>Board fit: C20 takes the board well; C15 does too once a print defect on one edge is cleaned off. C20 is the "
        "safer choice, so clear_chip stays 0.20 mm.</li>"
        "<li>Pins: the 2.40 mm pin of P fits the board&#8217;s 2.40 mm hole; pin_d is 2.40 mm (2.35 mm before).</li>"
        "<li>J1 slot: the Qwiic cable&#8217;s plug does not enter the carriers&#8217; 6.6 mm slot. The plug is JST&#8217;s "
        "SHR-04V-S-B, 7.0 mm wide at its protrusions, where the model had the 5.0 mm SHR-04V-S. Since 17:42 v2 a&#8217;s "
        "slot is 7.8 mm wide, 0.4 mm clear of the plug on each side.</li>"
        "<li>Cable groove: W&#8217;s turns are fine, but the cable is a flat ribbon about 4 mm wide and 1 mm thick that had "
        "to be squished into the 2.2 to 2.6 mm grooves. Since 17:57 v2 a models the ribbon, massaged into a bundle, as "
        "2.6 mm (2.0 mm before) in a 3.0 mm groove, and the cable turns at the plug&#8217;s rear face (after a 0.5 mm "
        "straight lead before), so that it stays 1.44 mm from the MCP link (R7 asks 1.0 mm). Run b&#8217;s W tests "
        "grooves of 2.8, 3.0 and 3.4 mm.</li>"
        "<li>Screw holes: M2 screws form their thread in H&#8217;s 1.8 mm holes, but not in the clamp plate&#8217;s "
        "1.8 mm holes; since 18:30 v2 a&#8217;s clamp plate and b&#8217;s tongue have 1.9 mm holes (screw_thread_d). The "
        "shoulders&#8217; pilots (m2_pilot) stay 1.8 mm.</li>"
        "<li>Open: the snap fit (K and F), the other screw holes of H, and C30.</li></ul></section>")


def future_section():
    return (
        '<section id="future"><h2>Future work</h2><ul>'
        f"<li>Enter fit_flank and detent_h (from K and F) and screw_thread_d and m2_pilot (from H) in {code(PARAMS)} and "
        f"rebuild v2 a: {code('scripts/dip_link_v2.py a build step check sense render print docs')}; then print the DIP "
        "link and a channelled tip. clear_chip (0.20 mm), pin_d (2.40 mm) and the cable groove (3.0 mm) are set.</li>"
        "<li>If K25 slides onto every block up to F40, the flank sweep stopped short of the limit; the next plate "
        "extends it to 0.60 mm. If K45 still does not hold in F*, the detent groove in the tip is the next variable "
        f"({code('groove_r')}).</li>"
        "<li>If no carrier takes the board flat, measure the chips&#8217; positions on the board and compare them with "
        f"{code(STUDY + '/spec.json')} before widening the windows further.</li>"
        "<li>Concept b&#8217;s rails, pad grooves and board seat join the next run&#8217;s plates (autopilot steps 4 and "
        "9), through the piece list of scripts/dip_fit_print_plates.py.</li></ul></section>")


def terms_section_b():
    T = [("Test piece", "A small part printed by the same process as the real part, to decide one fit before the whole "
          "DIP link is printed; in materials testing it is called a coupon."),
         ("Rail and groove", "Concept b&#8217;s cover has two PLA dovetail rails on its front face; the 85A pad has "
          "two matching grooves in its back face and is pressed onto the cover until both rails snap in."),
         ("Flank interference", "How much narrower the pad&#8217;s groove is than the rail on each flank, in mm; 0 is a "
          "line-to-line fit; the design has 0.05."),
         ("Seat pocket", "The pocket in the DIP link&#8217;s tongue that takes the board&#8217;s FR-4. It is open at the "
          "tongue&#8217;s distal end so that the board slides in along the finger, J1 first. Its depth is the rim height, "
          "0.45 mm in the design for a 0.60 mm board, so that the rim stays 0.15 mm short of the cover and the cover "
          "clamps the board."),
         ("Neck carrier", "The DIP link around the board: the tongue with the seat pocket, the neck with the slot for J1 "
          "and its plug, and the bridge with the Qwiic cable&#8217;s groove, open to the top over the plug and roofed "
          "along the run to the slotted arm."),
         ("Self-forming hole", "A hole smaller than the screw, in which the screw cuts its own thread in PLA; 1.9 mm for "
          "M2 since 2026-10-19 (1.8 mm, the owner&#8217;s rule of 10-18, was too small in run a&#8217;s clamp plate)."),
         ("G*, S*", "The groove block and the seat depth chosen in the first round of a check.")]
    dl = "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in T)
    return f'<section id="terms"><h2>3. Terms</h2><dl class="glossary">{dl}</dl></section>'


def checks_section_b():
    return (
        '<section id="checks"><h2>4. Checks, in order</h2>'
        "<h3>4.1 Snap fit: R and G</h3>"
        "<p>Press each G block onto the rail strip R by hand, G00 first, grooves over the rails. Record whether it goes "
        "on, whether both rails snap in, whether it stays under a 3 N pull away from the strip (a 300 g weight hung from "
        "it) and whether it stays when pushed along the rails. G* is the block with the largest interference that still "
        f"goes on by hand and stays under the pull; it sets {code('groove_fit')} (design 0.05 mm).</p>"
        "<h3>4.2 Board seat: S, V and N</h3>"
        "<p>Slide the ReSkin 5X board, component side up, into the pocket of S35, S45 and S55 from the open end until "
        "its main square meets the proximal wall and its tab passes the slot. Lay a cover V on it, windows over the "
        "parts, and drive four M2 &#215; 5 screws through the cover and the board into the 1.9 mm holes. S* is the "
        "deepest pocket in which the cover still clamps the board without play; it sets "
        f"{code('rim_h')} (design 0.45 mm). Then mate the Qwiic plug to J1 and slide the board into N from the "
        "neck&#8217;s underside, J1 and the plug first into their slot; clamp it with the second cover. Bend the cable "
        "toward the back in the open groove over the plug, lay it along the groove of the run and put a strip of Kapton "
        "tape over the open part.</p></section>")


def changes_section_b():
    return (
        '<section id="changes"><h2>6. Changes from concept b&#8217;s coupons of 18 October</h2>'
        f"<p>The rail coupon in {code(STUDY + '/v2/b/print/')} was a rail strip and three groove blocks (0, 0.05 and "
        "0.10 mm per flank) as XL projects. This run differs in ten ways.</p><ul>"
        "<li>One plate per printer: the PLA pieces on the P1S, the 85A blocks on the A1 mini (from 16:40, while the XL "
        "is in use) or the XL.</li>"
        "<li>Every piece has its ID printed on it.</li>"
        "<li>The groove sweep reaches 0.20 mm per flank.</li>"
        "<li>Three tongues with seat pockets 0.35, 0.45 and 0.55 mm deep, the neck carrier with J1&#8217;s slot and the "
        "cable&#8217;s groove of 19 October, and two covers join the snap fit. The neck carrier&#8217;s tongue is "
        "thickened 1.45 mm to the spine&#8217;s back so that the piece stands on one flat face.</li>"
        "<li>The first P1S project of this run started with Bambu Studio&#8217;s generic start G-code: it loaded no "
        "filament from the AMS and left the nozzle at 205 &#176;C. Since 16:40 every Bambu project has the printer&#8217;s "
        "own start, end and filament-change G-code, and the build stops unless the G-code loads from the AMS and heats to "
        "the filament&#8217;s temperatures.</li>" + A1M_SOLID + plug_slot_item() + cable_item() + P1S_85A + SCREW_HOLES_B + "</ul></section>")


def future_section_b():
    return (
        '<section id="future"><h2>Future work</h2><ul>'
        f"<li>Enter groove_fit and rim_h in {code(PARAMS)} and rebuild v2 b: "
        f"{code('scripts/dip_link_v2.py b build step check sense render print docs')}; then print the DIP link, the cover "
        "and a pad.</li>"
        "<li>If G20 still goes on by hand and lifts off under the pull, the groove sweep stopped short; the next plate "
        "extends it to 0.40 mm.</li>"
        "<li>If the board rocks in every pocket, measure the FR-4&#8217;s thickness and the tallest part under the "
        f"cover and set {code('fr4')} and {code('chip_h')} before changing the pocket.</li></ul></section>")


def build():
    R = json.loads((RUN / "pieces.json").read_text())
    pieces = R["pieces"]
    p1s, xl, a1m, p85 = R["plates"]["p1s"], R["plates"]["xl"], R["plates"].get("a1m"), R["plates"].get("p1s85a")
    n_pla = sum(s.get("copies", 1) for s in pieces if s["material"] == "PLA")
    n_tpu = sum(1 for s in pieces if s["material"] == "85A")
    tpu_where = (f"on one Bambu Lab P1S plate with the other run&#8217;s 85A blocks, from the external spool "
                 f"({esc(p85['time'])}, {p85['filament_g']:.1f} g for all {len(p85['objects'])})" if p85 else
                 f"on the Bambu Lab A1 mini ({esc(a1m['time'])}, {a1m['filament_g']:.1f} g; or on tool 2 of the Prusa XL)"
                 if a1m else f"on tool 2 of the Prusa XL ({esc(hmin(xl['time']))}, {xl['filament_g']:.1f} g)")
    with open(RUN / "records.csv") as f:
        rows = list(csv.DictReader(f))
    lede_b = (f"Two plates print the fit tests of DIP-link concept b, v2 (the link 9.75 mm longer, the Qwiic cable turned "
              f"toward the back at the plug and routed past the slotted arm): {n_pla} PLA pieces on the Bambu Lab P1S as "
              f"one sliced Bambu Studio project ({esc(p1s['time'])}, {p1s['filament_g']:.1f} g) and {n_tpu} Siraya Tech "
              f"85A groove blocks {tpu_where}. Each "
              "piece has its ID printed on it: R for the cover's rail strip, G for the groove blocks, S for the three "
              "tongues with seat pockets, N for the neck carrier with J1's slot and the cable groove, V for the two covers. "
              "The checks of Section 4 fill records.csv, and each answer sets one parameter of the design file.")
    lede = (f"Two plates print every fit test of DIP-link concept a, v2 (the link 9.75 mm longer, with the Qwiic cable "
            f"routed past the slotted arm): {n_pla} PLA pieces on the Bambu Lab P1S as one sliced Bambu Studio project "
            f"({esc(p1s['time'])}, {p1s['filament_g']:.1f} g) and {n_tpu} Siraya Tech 85A channel blocks {tpu_where}. "
            "Each piece has its ID printed on it: K for the four PLA "
            "keys, F for the channel blocks and C for the three board carriers, the digits giving the variant in "
            "hundredths of a millimetre; P, H and W are the pin, screw-hole and cable-groove pieces, with each size "
            "engraved beside it. The checks of Section 4 fill records.csv, and each answer sets one parameter of the "
            "design file.")
    if CONCEPT == "b":
        lede = lede_b
        secs = [files_section(R, p1s, xl, n_pla, n_tpu, a1m, p85), ids_section(pieces), terms_section_b(), checks_section_b(),
                records_section(rows), changes_section_b(), future_section_b()]
    else:
        secs = [files_section(R, p1s, xl, n_pla, n_tpu, a1m, p85), ids_section(pieces), terms_section(), checks_section(),
                records_section(rows), results_section_a(), changes_section(), future_section()]
    toc = ["<li><a href='#files'>1. Files and settings</a></li>", "<li><a href='#ids'>2. Pieces and their IDs</a></li>",
           "<li><a href='#terms'>3. Terms</a></li>", "<li><a href='#checks'>4. Checks, in order</a></li>",
           "<li><a href='#record'>5. Record</a></li>"]
    toc += (["<li><a href='#results'>6. Results so far</a></li>",
             "<li><a href='#changes'>7. Changes from the pieces of 13:08</a></li>"] if CONCEPT == "a" else
            ["<li><a href='#changes'>6. Changes from concept b's coupons of 18 October</a></li>"])
    toc += [
           "<li><a href='#future'>Future work</a></li>"]
    up = "../.."
    eyebrow = (f'<a href="{up}/index.html">SR&#178; Hand documentation</a> &#183; <a href="{up}/pages.html">index of '
               f'result pages</a> &#183; <a href="{up}/overviews/printed_fingertips.html">printed fingertips and force '
               f'sensing</a> &#183; 19 October 2026')
    byline = ("Bambu Lab P1S, 0.4 mm nozzle, Bambu PLA Basic &#183; Prusa XL 5T, tool 2, Siraya Tech TPU 85A &#183; "
              f"design: {code(PARAMS)}; the DIP-link study: <a href=\"../20261018-dip_link_board_mount/"
              "20261018-dip_link_magnetometer_board_mount.html\">magnetometer board mount page</a> &#183; built "
              f"{time.strftime('%Y-%m-%d %H:%M')}")
    rarg = "" if CONCEPT == "a" else " --run 20261019b"
    footer = (f"Rebuild: <code>scripts/dip_fit_print_plates.py{rarg} pieces bambu xl</code> (femtip Python under resguard; "
              "Bambu Studio 2.5 and PrusaSlicer flatpaks), then <code>.venv/bin/python "
              f"scripts/dip_fit_print_plates_page.py{rarg}</code> (template <code>scripts/dip_fit_print_plates_page.template.html"
              f"</code>). Sources: {code(rel(RUN / 'pieces.json'))}, the XL project&#8217;s report "
              f"{code(xl['report'])}, {code(rel(RUN / 'records.csv'))}; concept {CONCEPT}&#8217;s geometry from "
              "<code>scripts/dip_link_v2.py</code> at the design file.")
    page = TEMPLATE.read_text()
    h1 = ("Print sheet for the DIP-link fit tests of 19 October 2026" if CONCEPT == "a" else
          "Print sheet for the fit tests of DIP-link concept b, 19 October 2026")
    for k, v in dict(TITLE="DIP fit print sheet" if CONCEPT == "a" else "DIP b fit print sheet", H1=h1,
                     EYEBROW=eyebrow, LEDE=lede, BYLINE=byline, TOC="\n".join(toc), SECTIONS="\n".join(secs),
                     FOOTER=footer).items():
        page = page.replace("{{" + k + "}}", v)
    if "{{" in page:
        raise SystemExit(f"unfilled placeholder: {re.findall(r'{{[A-Z_]+}}', page)}")
    page = retro_style.apply(page)
    PAGE.write_text(page)
    print(f"wrote {rel(PAGE)}: {len(page) / 1e6:.2f} MB, {FIG[0]} figures, {len(rows)} record rows", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="20261019", choices=list(RUNS))
    set_run(ap.parse_args().run)
    build()
