#!/usr/bin/env python3
"""Print sheet of a DIP-link fit-test run, read at the printers on a phone: the two plates, every piece with its printed
ID, the checks in order, the parameter each check sets, and the owner's records.

The page is built from the run folder that scripts/dip_fit_print_plates.py writes (pieces.json, the plate maps, the XL
project's report, records.csv) and regenerates when they change; filled rows of records.csv appear in Section 5.

    .venv/bin/python scripts/dip_fit_print_plates_page.py
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


def files_section(R, p1s, xl, n_pla, n_tpu):
    ov = p1s["overrides"]
    xs = xl["settings"]
    return (
        '<section id="files"><h2>1. Files and settings</h2>'
        f"<p>All files are in {code(rel(RUN) + '/')}. Open the P1S project in Bambu Studio and print it as it is: it is "
        f"sliced, with the G-code inside, for the {esc(p1s['printer'])} with {esc(p1s['filament'])} on the "
        f"{esc(p1s['bed_type'])} at {p1s['bed_c']} &#176;C. The process is {esc(p1s['presets']['process'])} changed to "
        f"{ov['layer_height']} mm layers, {ov['wall_loops']} walls, {ov['sparse_infill_density']} rectilinear infill, the "
        "seam at the back, an automatic brim and no supports; Bambu Studio marks these values as changed. The XL project "
        f"prints on the v17 template as the 85A fingertips did: tool 2, {xs['layer_height']} mm layers, "
        f"{xs['perimeters']} perimeters, {xs['fill_density']} gyroid, no pause.</p>"
        '<div class="tw"><table class="facts"><tr><th>Plate</th><th>File</th><th>Pieces</th><th>Time</th>'
        "<th>Filament</th></tr>"
        f"<tr><td>Bambu Lab P1S, PLA</td><td>{code(Path(p1s['project']).name)}</td><td>{n_pla} "
        f"({len(p1s['objects'])} objects)</td><td>{esc(p1s['time'])}</td><td>{p1s['filament_g']:.1f} g</td></tr>"
        f"<tr><td>Prusa XL 5T, tool 2, 85A</td><td>{code(Path(xl['project']).name)}</td><td>{n_tpu}</td>"
        f"<td>{esc(hmin(xl['time']))}</td><td>{xl['filament_g']:.1f} g</td></tr></table></div>"
        f"<p>Each piece's STL, in print orientation on the bed and named by its ID, is in {code(rel(RUN / 'stl') + '/')}, "
        "for printing one piece again.</p>"
        + figure(ROOT / p1s["map"], "The P1S plate from above, front edge at the bottom. Bold: the ID printed on the "
                 "piece; below: its variant.", "P1S plate map with piece IDs")
        + figure(ROOT / xl["map"], "The XL plate from above; each block's ID is engraved in its top face, which lies "
                 "inside the tip.", "XL plate map with piece IDs")
        + "</section>")


def ids_section(pieces):
    rows = "".join(
        f"<tr><td>{esc(s['id'])}</td><td>{esc(s['piece'])}{' (2)' if s.get('copies', 1) > 1 else ''}</td>"
        f"<td>{esc(s['variant'])}</td><td>{esc(s['mark'])}</td><td>{'P1S' if s['printer'] == 'p1s' else 'XL'}</td></tr>"
        for s in pieces)
    return ('<section id="ids"><h2>2. Pieces and their IDs</h2>'
            "<p>The digits of K, F and C give the variant in hundredths of a millimetre: K25 is the key with a 0.25 mm "
            "detent, F10 the channel block 0.10 mm narrower per flank, C30 the carrier with 0.30 mm chip windows.</p>"
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
            + ". Write y, n or a number in result and anything seen in notes; a pairing is written K25+F10, and F* and "
              f"C* stand for the choices of round 1. {len(filled)} of {len(rows)} rows are filled.</p>"
            f"<details><summary>All {len(rows)} rows</summary><div class=\"tw\"><table class=\"rec\"><tr><th>ID</th>"
            f"<th>Test</th><th>Check</th><th>Result</th><th>Notes</th></tr>{body}</table></div></details></section>")


def changes_section():
    return (
        '<section id="changes"><h2>6. Changes from the pieces of 13:08</h2>'
        f"<p>The set in {code(STUDY + '/v2/coupons/')} was four XL projects with PLA on tool 3. This run differs in "
        "five ways.</p><ul>"
        "<li>One plate per printer: the PLA pieces on the P1S, the 85A blocks on the XL.</li>"
        "<li>Every piece has its ID printed on it; the three window gauges had none.</li>"
        "<li>The detent sweep reaches 0.45 mm and the flank sweep 0.30 and 0.40 mm, so that each choice can fall "
        "inside its sweep.</li>"
        "<li>Three carriers, one per chip-window clearance, replace the window gauges; a carrier contains the "
        "gauge&#8217;s window plate and adds J1, the cable groove, the clamp plate and the screws. The pin, screw-hole "
        "and cable-groove pieces P, H and W join them.</li>"
        "<li>The carrier is cut at the bridge&#8217;s flat, which left it standing on two 0.8 mm strips under the "
        "bridge&#8217;s curved face; that face reaches the full section 1.8 mm above the bed. The plates fill it down to "
        "the bed (199.5 mm&#179;); the J1 plug, the cable and the board lie outside the fill.</li></ul></section>")


def future_section():
    return (
        '<section id="future"><h2>Future work</h2><ul>'
        f"<li>Enter fit_flank, detent_h, clear_chip, pin_d, screw_thread_d, m2_pilot and the groove width in "
        f"{code(PARAMS)} and rebuild v2 a: {code('scripts/dip_link_v2.py a build step check sense render print docs')}; "
        "then print the DIP link and a channelled tip.</li>"
        "<li>If K25 slides onto every block up to F40, the flank sweep stopped short of the limit; the next plate "
        "extends it to 0.60 mm. If K45 still does not hold in F*, the detent groove in the tip is the next variable "
        f"({code('groove_r')}).</li>"
        "<li>If no carrier takes the board flat, measure the chips&#8217; positions on the board and compare them with "
        f"{code(STUDY + '/spec.json')} before widening the windows further.</li>"
        "<li>Concept b&#8217;s rails, pad grooves and board seat join the next run&#8217;s plates (autopilot steps 4 and "
        "9), through the piece list of scripts/dip_fit_print_plates.py.</li></ul></section>")


def build():
    R = json.loads((RUN / "pieces.json").read_text())
    pieces = R["pieces"]
    p1s, xl = R["plates"]["p1s"], R["plates"]["xl"]
    n_pla = sum(s.get("copies", 1) for s in pieces if s["printer"] == "p1s")
    n_tpu = sum(1 for s in pieces if s["printer"] == "xl")
    with open(RUN / "records.csv") as f:
        rows = list(csv.DictReader(f))
    lede = (f"Two plates print every fit test of DIP-link concept a, v2 (the link 9.75 mm longer, with the Qwiic cable "
            f"routed past the slotted arm): {n_pla} PLA pieces on the Bambu Lab P1S as one sliced Bambu Studio project "
            f"({esc(p1s['time'])}, {p1s['filament_g']:.1f} g) and {n_tpu} Siraya Tech 85A channel blocks on tool 2 of "
            f"the Prusa XL ({esc(hmin(xl['time']))}, {xl['filament_g']:.1f} g). Each piece has its ID printed on it: K for the four PLA "
            "keys, F for the channel blocks and C for the three board carriers, the digits giving the variant in "
            "hundredths of a millimetre; P, H and W are the pin, screw-hole and cable-groove pieces, with each size "
            "engraved beside it. The checks of Section 4 fill records.csv, and each answer sets one parameter of the "
            "design file.")
    secs = [files_section(R, p1s, xl, n_pla, n_tpu), ids_section(pieces), terms_section(), checks_section(),
            records_section(rows), changes_section(), future_section()]
    toc = ["<li><a href='#files'>1. Files and settings</a></li>", "<li><a href='#ids'>2. Pieces and their IDs</a></li>",
           "<li><a href='#terms'>3. Terms</a></li>", "<li><a href='#checks'>4. Checks, in order</a></li>",
           "<li><a href='#record'>5. Record</a></li>", "<li><a href='#changes'>6. Changes from the pieces of 13:08</a></li>",
           "<li><a href='#future'>Future work</a></li>"]
    up = "../.."
    eyebrow = (f'<a href="{up}/index.html">SR&#178; Hand documentation</a> &#183; <a href="{up}/pages.html">index of '
               f'result pages</a> &#183; <a href="{up}/overviews/printed_fingertips.html">printed fingertips and force '
               f'sensing</a> &#183; 19 October 2026')
    byline = ("Bambu Lab P1S, 0.4 mm nozzle, Bambu PLA Basic &#183; Prusa XL 5T, tool 2, Siraya Tech TPU 85A &#183; "
              f"design: {code(PARAMS)}; the DIP-link study: <a href=\"../20261018-dip_link_board_mount/"
              "20261018-dip_link_magnetometer_board_mount.html\">magnetometer board mount page</a> &#183; built "
              f"{time.strftime('%Y-%m-%d %H:%M')}")
    footer = ("Rebuild: <code>scripts/dip_fit_print_plates.py pieces bambu xl</code> (femtip Python under resguard; "
              "Bambu Studio 2.5 and PrusaSlicer flatpaks), then <code>.venv/bin/python "
              "scripts/dip_fit_print_plates_page.py</code> (template <code>scripts/dip_fit_print_plates_page.template.html"
              f"</code>). Sources: {code(rel(RUN / 'pieces.json'))}, the XL project&#8217;s report "
              f"{code(xl['report'])}, {code(rel(RUN / 'records.csv'))}; concept a&#8217;s geometry from "
              "<code>scripts/dip_link_v2.py</code> at the design file.")
    page = TEMPLATE.read_text()
    for k, v in dict(TITLE="DIP fit print sheet", H1="Print sheet for the DIP-link fit tests of 19 October 2026",
                     EYEBROW=eyebrow, LEDE=lede, BYLINE=byline, TOC="\n".join(toc), SECTIONS="\n".join(secs),
                     FOOTER=footer).items():
        page = page.replace("{{" + k + "}}", v)
    if "{{" in page:
        raise SystemExit(f"unfilled placeholder: {re.findall(r'{{[A-Z_]+}}', page)}")
    page = retro_style.apply(page)
    PAGE.write_text(page)
    print(f"wrote {rel(PAGE)}: {len(page) / 1e6:.2f} MB, {FIG[0]} figures, {len(rows)} record rows", flush=True)


if __name__ == "__main__":
    build()
