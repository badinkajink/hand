#!/usr/bin/env python3
"""Page: the magnetometer board mount in the DIP link for the printed 85A fingertips (study 20261018-dip_link_board_mount;
hand-off docs/handoff/20261018-calibration_and_dip_finish.md, step 5).

Every number is read from the study's files: decision.json (the three concepts scored, the Qwiic study, the tip
attachment), final/evaluation.json, final/20261018-dip_link_final_{check,sense,qwiic,print_settings,parts_list,assembly}.json.

    .venv/bin/python scripts/dip_link_board_mount_page.py
Out: docs/experiments/20261018-dip_link_board_mount/20261018-dip_link_magnetometer_board_mount.html (media inlined).
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

D = os.path.join(ROOT, "docs/experiments/20261018-dip_link_board_mount")
FIN = os.path.join(D, "final")
PFX = "20261018-dip_link_final"
OUT = os.path.join(D, "20261018-dip_link_magnetometer_board_mount.html")
TPL = os.path.join(D, "page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
FIG, TAB, REFS = [0], [0], {}
EXTRA_CSS = """
.tw th{white-space:normal !important;vertical-align:bottom;text-transform:none}
.tw td:first-child{min-width:9em}
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
figure.img img{width:100%;height:auto}
figure.row3 .imgs{display:grid;grid-template-columns:1fr;gap:10px}
figure.row3 img{width:100%;height:auto}
figure.row3 .lab{text-align:center;font-size:15px}
ol.steps li{margin:.3em 0}
figure.panels img{display:block;width:auto;max-width:100%;height:auto;margin:0 auto 10px}
figure.films video{width:100%;height:auto;display:block}
""" + texsvg.BLOCK_CSS + "\n"
MM, NN, DEG, MT = "&#8202;mm", "&#8202;N", "&#176;", "&#8202;mT"


# ------------------------------------------------------------------------------------------ helpers

def J(rel):
    return json.load(open(os.path.join(D, rel)))


def data_uri(path):
    mime = {".png": "image/png", ".jpg": "image/jpeg"}[os.path.splitext(path)[1]]
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


def img(rel, caption, key=None, width=None, png=False):
    p = os.path.join(D, rel)
    if not os.path.exists(p):
        raise SystemExit(f"missing figure {rel}")
    src = data_uri(p) if png else jpeg_uri(p, width=width)
    return figure(f'<img alt="" src="{src}">', caption, key)


def film(rel, poster_rel, caption, key=None):
    p = os.path.join(D, rel)
    if not os.path.exists(p):
        raise SystemExit(f"missing film {rel}")
    v = (f'<video controls muted loop playsinline preload="metadata" poster="{jpeg_uri(os.path.join(D, poster_rel))}" '
         f'src="data:video/mp4;base64,{base64.b64encode(open(p, "rb").read()).decode()}"></video>')
    return figure(v, caption, key, cls="films")


def pil_uri(im, quality=88):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def split_panels(path, legend_rows=True, min_gap=18):
    """The panels of a matplotlib figure laid out in one row, cut at the runs of white columns at least min_gap px wide
    (the gaps between panels; a panel's axis label and tick labels lie nearer to it). With legend_rows the legend under the
    panels, below the first white row band of 30 px or more in the lower 40 % of the image, is dropped."""
    import numpy as np
    from PIL import Image
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(int)
    white = a.min(axis=2) > 245
    H, Wd = white.shape

    def runs(mask):
        out, k = [], 0
        while k < len(mask):
            if mask[k]:
                k1 = k
                while k1 < len(mask) and mask[k1]:
                    k1 += 1
                out.append((k, k1))
                k = k1
            else:
                k += 1
        return out
    top = H
    if legend_rows:
        band = [(y0, y1) for y0, y1 in runs(white.all(axis=1)) if y0 >= 0.6 * H and y1 - y0 >= 30]
        if band:
            top = band[0][0]
    cuts = [(x0, x1) for x0, x1 in runs(white[:top].all(axis=0)) if x1 - x0 >= min_gap or x0 == 0 or x1 == Wd]
    edges = [0] + [v for c in cuts for v in c] + [Wd]
    spans = [(edges[k], edges[k + 1]) for k in range(0, len(edges) - 1, 2) if edges[k + 1] - edges[k] > 40]
    return [im.crop((max(0, x0 - 8), 0, min(Wd, x1 + 8), top)) for x0, x1 in spans]


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


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def cap(s):
    s = str(s)
    return s[:1].upper() + s[1:]


# ------------------------------------------------------------------------------------------ data

DEC = J("decision.json")
EV = J("final/evaluation.json")
CHK = J(f"final/{PFX}_check.json")
SEN = J(f"final/{PFX}_sense.json")
QW = J(f"final/{PFX}_qwiic.json")
PS = J(f"final/{PFX}_print_settings.json")
PL = J(f"final/{PFX}_parts_list.json")
AS = J(f"final/{PFX}_assembly.json")
CA = J("concept_a/20261018-concept_a_evaluation.json")
SPEC = J("spec.json")
FR = EV["force_resolution_N"]
REF = DEC["reference_current_plate_N"]
CL = CHK["clearance"]
WORST = EV["clearance"]["worst_rigid"]
QCH = QW["chosen"]
TA = DEC["tip_attachment"]
SF = SEN["static_fields"]
TIPN = {"block4": "Block, four magnets", "block5": "Block, five magnets", "cap8.5": "Capsule r 8.5&#8202;mm",
        "ell16": "Half-ellipsoid 16/32&#8202;mm", "ell11": "Half-ellipsoid 11/22&#8202;mm"}


def srow(layout, recipe, case):
    for r in SEN["rows"]:
        if r["layout"] == layout and r["recipe"] == recipe and r["case"] == case:
            return r
    raise KeyError((layout, recipe, case))


def req(rid):
    return next(r for r in SPEC["requirements"] if r["id"] == rid)


# ------------------------------------------------------------------------------------------ sections

GLOSSARY = [
    ("Frame F", "The finger frame of the specification (<code>spec.json</code>): index finger at zero joint angles, x "
                "dorsal, y along the PIP axis, z up (proximal), in mm. The tip's back face lies at x 6.45&#8202;mm, its "
                "mount face at z &#8722;67.30&#8202;mm and the contact point at (&#8722;10.55, 0, &#8722;78.16)&#8202;mm."),
    ("Gap", "Distance from the tip's undeformed back face to the chips' sensing plane, which lies 0.42&#8202;mm under each "
            "MLX90393 package top, in mm. The current DIP link mounts the board on a 3.0&#8202;mm plate: gap 3.42&#8202;mm."),
    ("Chip-to-magnet distance", "Least distance from a chip's sensing point to a magnet's centre, in mm (to the magnet's "
                                "surface where stated). Magnets are N52 discs 3&#8202;mm across and 2&#8202;mm thick."),
    ("Force resolution", "Cram&#233;r&#8211;Rao bound (one standard deviation, N) of the normal force at the contact point "
                         "from one 141&#8202;Hz sample of the five chips, with the contact position and a uniform "
                         "background field unknown; magnet motions per newton from the readiness study's finite elements of "
                         "the 85A block at 30&#8202;% gyroid unless stated."),
    ("False force", "Normal force (N) the force estimator reads per micrometre of rigid board motion toward the magnets; a "
                    "measure of how much board play the readout tolerates."),
    ("Clearance", "Least signed distance (mm) from a part fixed to the DIP link to the MCP link and the PIP servo over PIP "
                  "&#8722;18 to 92&#176; in 0.5&#176; steps, from surface samples 0.25&#8202;mm apart "
                  "(<code>dip_link_geometry.points_clearance_exact</code>). Requirement R7 asks 0.5&#8202;mm for rigid parts "
                  "and 1.0&#8202;mm for the cable; a negative value is an overlap."),
    ("GAIN_SEL", "The MLX90393's analogue gain setting (0&#8211;7), chosen so the static field at the chips fits the range "
                 "with 1.25 times headroom; lower settings cover larger fields with coarser steps."),
    ("R1&#8211;R12", "The requirements of the specification (<code>spec.json</code>): R1 gap at most 0.52&#8202;mm, R4 no "
                     "retention feature within 2.0&#8202;mm of a magnet pocket, R5 tip surface unchanged and nothing wider "
                     "than 21.0&#8202;mm over the tip's length, R7 clearance, R10 no ferromagnetic fastener within 15&#8202;mm of "
                     "a chip and the board no closer to the PIP servo than now, R11 "
                     "printable walls of 0.9&#8202;mm or more, R12 at most 7.6&#8202;g without the tip."),
    ("Interference per flank", "How much narrower each flank of the tip's channel is than the PLA key it slides on, in mm."),
    ("Detent", "A 0.25&#8202;mm ridge across each key's head face that drops into a groove of the channel floor and keeps "
               "the tip from sliding off along &#8722;z."),
    ("Qwiic, J1", "Qwiic is SparkFun's I2C connection on a 4-pin JST SH connector (1.0&#8202;mm pitch); J1 is the board's "
                  "side-entry Qwiic header on its tab. The cable ends in a JST SHR-04V-S housing."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def lede():
    return (f"Concept a's final DIP link puts the sensing plane of the ReSkin 5X board's chips {f(EV['gap_mm'])}{MM} behind the "
            f"85A tip's back face, against {f(EV['gap_current_plate_mm'])}{MM} on the current plate. At the tips printed with "
            f"2.9{MM} magnet seats the predicted force resolution at the contact point is {f(FR['four_30'], 3)}{NN} with four "
            f"magnets and {f(FR['five_z7p4_30'], 3)}{NN} with five, against "
            f"{f(srow('four_d3x2_d5', 'w85_30', 'plate_now')['centre_force_N'], 3)} and "
            f"{f(srow('five_z7p4', 'w85_30', 'plate_now')['centre_force_N'], 3)}{NN} on the current plate. A 0.85{MM} window "
            f"plate and a 3.0{MM} clamp plate sandwich the board with four brass screws and two pins; the least clearance over "
            f"PIP &#8722;18 to 92{DEG} is {f(WORST['min_mm'], 3)}{MM}. No orientation of the board's Qwiic connector clears, so "
            f"a pigtail runs to a Qwiic header on the clamp plate's back. Concept b, a rigid cassette inside the tip that "
            f"resolves {f(DEC['concepts']['b']['force_resolution_N']['four_30'], 3)}{NN}, also goes to a first print.")


def design_section():
    L = EV["stack_x_F"]
    t = [f"<p>The DIP link's plate behind the tip becomes a window plate {f(L['window_plate'][1] - L['window_plate'][0])}{MM} "
         f"thick, printed at the chips' package height less 0.05{MM}. The five MLX90393 chips reach through its windows "
         f"and stand 0.05{MM} proud of its palmar face, so the tip's back face rests on the window plate and presses on the "
         f"chip tops; the sensing plane lies {f(EV['gap_mm'])}{MM} behind the back face (requirement R1 allows 0.52{MM}). "
         f"Behind the board's bare face a 3.0{MM} PLA clamp plate presses the FR-4 onto the window plate: two M1.6 "
         f"countersunk brass screws pass from the palmar side through the window plate and the board's distal holes into "
         f"brass nuts in the clamp plate, and two M2 brass screws pull the clamp plate's ears toward the shoulders of the "
         f"link, which the ears miss by 0.3{MM} so the preload goes into the board. Two pins of the clamp plate pass "
         f"through the board's proximal holes into reamed holes of the window plate with 0.025{MM} radial clearance and "
         f"locate the board in its plane; the stack has no designed clearance normal to the board "
         f"(Figures&#160;[[fig:iso]]&#8211;[[fig:section]]).</p>",
         f"<p>The tip keeps its outer surface and its stiffness. It slides along +z onto two PLA dovetail keys of the window "
         f"plate until its mount face meets the bridge; a detent on each key keeps it there (Section&#160;5). The "
         f"tip and the board come off separately: the tip by a pull along &#8722;z, the board after four screws. Without "
         f"the tip the assembly weighs {f(CHK['mass_g']['total_without_tip'])}&#8202;g at the links' 15&#8202;% infill "
         f"(R12 limit {f(CHK['mass_g']['limit_R12'], 1)}&#8202;g; the current DIP link with the board weighs "
         f"{f(CHK['mass_g']['current_dip_link_with_board'])}&#8202;g).</p>",
         img(f"final/media/{PFX}_iso.png", "Final DIP link assembled at PIP 0&#176;, seen from the dorsal side: clamp plate "
             "(blue), the Qwiic header in its pocket on the clamp plate's back (white), brass screws and nuts, the 85A "
             "block tip (tan), the PIP servo and the clevis arm of the current link.", "iso", width=1400),
         img(f"final/media/{PFX}_exploded.png", "Exploded view. The Qwiic header and the pigtail (white, not numbered) are "
             "glued to the clamp plate and move with it.", "exploded", width=1400),
         figure("".join(f'<img alt="" src="{pil_uri(p_)}">' for p_ in split_panels(os.path.join(FIN, "media",
                                                                                              f"{PFX}_section.png"))),
                "Sections through the stack at y = 0 (chips U4, U1, U2), at y = 3.0&#8202;mm (magnets and a key) and at z "
                "&#8722;78.16&#8202;mm (chips U5, U1, U3). Colours: 85A block tip tan, DIP link grey, board green, clamp "
                "plate blue, brass screws and nuts ochre, wires orange, magnets red with their north half darker; the cross "
                "marks the contact point.", "section", cls="panels")]
    return "".join(t)


def concepts_section():
    C = DEC["concepts"]
    keys = ("a", "b", "c")
    rowsdef = [
        ("Force resolution, four / five magnets (N)",
         lambda c: f"{f(c['force_resolution_N']['four_30'], 3)} / {f(c['force_resolution_N']['five_30'], 3)}"),
        ("Gap (mm)", lambda c: f(c["gap_mm"])),
        ("Chip to magnet centre (mm)", lambda c: f(c["chip_to_magnet_centre_mm"])),
        ("False force (N per &#181;m)", lambda c: f(c["board_play"]["false_force_x_N_per_um"], 3)),
        ("Board retention", lambda c: esc(cap(c["board_play"]["mechanism"]))),
        ("Contact geometry", lambda c: esc(cap(c["contact_geometry"]))),
        ("Worst rigid clearance (mm)", lambda c: f(c["clearance_mm"]["worst_rigid"], 3)),
        ("Board-edge exposure", lambda c: esc(cap(c["board_edge_exposure"]))),
        ("Printability on the XL", lambda c: esc(cap(c["printability"]))),
        ("Assembly and replacement", lambda c: esc(cap(c["assembly"]))),
        ("Cable past the PIP joint", lambda c: esc(cap(c["cable"]))),
        ("Mass without the tip (g)", lambda c: f(c["mass_g"])),
    ]
    body = [["Design"] + [esc(C[k]["name"]) for k in keys]] + [[name] + [fn(C[k]) for k in keys] for name, fn in rowsdef]
    head = ["Criterion"] + [f"Concept {k}" for k in keys]
    ims = "".join(f'<div><img alt="" src="{jpeg_uri(os.path.join(D, p), width=1400)}"><div class="lab">{lab}</div></div>'
                  for p, lab in (("concept_a/media/20261018-concept_a_iso.png", "a"),
                                 ("concept_b/media/20261018-concept_b_iso.png", "b"),
                                 ("concept_c/media/20261018-concept_c-iso.png", "c")))
    return "".join([
        f"<p>Three concepts were drawn to the same specification, and the owner takes a and b to a first print. Concept a "
        f"is the design of Section&#160;2. "
        f"Concept b replaces the tip's dorsal 6{MM} with a rigid cassette that clamps the board 0.42{MM} behind an 11{MM} "
        f"TPU pad; it resolves the finest forces, {f(C['b']['force_resolution_N']['four_30'], 3)}{NN} with four magnets, "
        f"but the cassette stiffens the contact by about 20&#8202;% and halves the magnet travel per newton, and the "
        f"11/22 half-ellipsoid's groove lies 0.85{MM} from a magnet pocket against R4's 2.0{MM}. Concept c places the "
        f"board at a print pause and prints the cap and the tip over it; it resolves "
        f"{f(C['c']['force_resolution_N']['four_30'], 3)}{NN}, has no screw preload on the board, and a worn tip costs a "
        f"board. Table&#160;[[tab:concepts]] scores the three on eight criteria; the force resolutions compare with "
        f"{f(REF['four_30'], 3)}{NN} (four magnets) and {f(REF['five_30'], 3)}{NN} (five, corner magnets 7.0{MM} apart as "
        f"the concepts were scored) on the current plate.</p>",
        table(head, body, "The three concepts on eight criteria (<code>decision.json</code>; force "
              "resolution at 30&#8202;% gyroid, four-magnet block; concept b on its coarse finite-element grid).",
              text_cols=(0, 1, 2, 3), key="concepts"),
        figure(f'<div class="imgs">{ims}</div>', "Concepts a, b and c assembled, from top to bottom (renders from each concept's folder).",
               "concepts", cls="row3"),
    ])


def qwiic_section():
    rows = []
    for r in QW["rows"]:
        rows.append([("Flipped, " if r["flipped"] else "") + esc(r["orientation"]),
                     f(r["connector_clearance_mm"]), f(r["tab_clearance_mm"]), f(r["beyond_tip_end_mm"], 1),
                     f(r["beyond_band_mm"], 1), "yes" if (r["clears_links"] and r["inside_envelope"]) else "no"])
    hb = QCH["header_box_F"]
    return "".join([
        f"<p>No orientation of the board's own Qwiic connector J1 clears both the links and the space the board already "
        f"occupies, so the final removes J1 and solders a pigtail from its pads to a Qwiic header on the clamp plate's "
        f"back. The study placed J1, its mated SHR-04V-S housing and the cable's first 3{MM} in eight orientations of "
        f"the board (Table&#160;[[tab:qwiic]]). With the tab proximal the connector overlaps the PIP servo by "
        f"{f(-QW['rows'][0]['connector_clearance_mm'])}{MM} at PIP &#8722;18{DEG}, "
        f"{f(-QW['rows'][4]['connector_clearance_mm'])}{MM} with the board flipped. With the tab distal or lateral the "
        f"connector clears the links by 12.2&#8211;15.4{MM} but stands 12.5{MM} beyond the tip's end or 12.6{MM} outside "
        f"the 21{MM} band, where grasped objects meet it: in the readiness study the cylinder touched the board in 9 to 27 "
        f"of 27 runs per cell with the board inside the band.</p>",
        table(["Board orientation", "Connector clearance (mm)", "Tab clearance (mm)", "Beyond the tip's end (mm)",
               "Outside the 21&#8202;mm band (mm)", "Clears"], rows,
              "J1 with its mated plug and the cable's first 3&#8202;mm in eight board orientations: least clearance to the "
              "MCP link and the PIP servo over PIP &#8722;18 to 92&#176;, and how far the connector stands outside the "
              "tip's end and the 21&#8202;mm band (<code>final/20261018-dip_link_final_qwiic.json</code>). Clears: 0.5&#8202;mm "
              "or more from the links and nothing outside the envelope.", text_cols=(0, 5), key="qwiic"),
        f"<p>The header is a JST SM04B-SRSS-TB (side entry, 6.0 by 4.25{MM}, 2.9{MM} tall), glued into a 1.0{MM} pocket "
        f"cut into the clamp plate's back between z {f(hb[0][2])} and {f(hb[1][2])}{MM}, with its mating face toward the "
        f"PIP joint. The pocket keeps the clamp plate printable on its back face and leaves "
        f"{f(QCH['clamp_plate_under_pocket_mm'])}{MM} of plate under the header. Four 20{MM} wires (AWG 32&#8211;36) run "
        f"from J1's pads through a 0.6{MM} groove to the header's leads. Header, plug, cable start and pigtail clear the PIP "
        f"servo by {f(CL['qwiic_connector']['min_mm'], 2)}{MM} at PIP &#8722;18{DEG} (Figure&#160;[[fig:qwiic]]); the "
        f"plug stands {f(QCH['height_above_clamp_plate_mm'])}{MM} above the clamp plate's back, inside the "
        f"21{MM} band.</p>",
        img(f"final/media/{PFX}_qwiic.png", "Qwiic header in its pocket on the clamp plate's back with the mated plug, the "
            "cable's first 3&#8202;mm and the pigtail from J1's pads (PIP 0&#176;).", "qwiic", width=1400),
    ])


def tip_section():
    rows = [[esc(cap(r["load"])), esc(cap(r["path"])), esc(cap(r["numbers"])), esc(cap(r["status"]))]
            for r in TA["load_paths"]]
    tests = "".join(f"<li><b>{t['id']}</b> {esc(t['name'])}: {esc(t['what'])}. It decides {esc(t['decides'])}.</li>"
                    for t in TA["tests"])
    return "".join([
        f"<p>{esc(cap(TA['method']))}. {esc(TA['todays_prints'])} The closest channel lies "
        f"{f(min(v['min_mm'] for v in CHK['channels_to_magnets_R4'].values()))}{MM} from a magnet (11/22 half-ellipsoid; "
        f"R4 asks 2.0{MM}).</p>",
        "<p>Table&#160;[[tab:loads]] follows the specification's design loads, 10&#8202;N normal and 5&#8202;N tangential "
        "at the contact point, and adds a pull-off and a twist about the normal. The normal load and the load toward the "
        "PIP joint bear on rigid faces. A 5&#8202;N lateral load acts 17.0&#8202;mm from the back face, so the tip pivots on "
        "its back edge and pulls the far key with 7.7&#8202;N. The twist cannot exceed the torsional friction of the "
        "contact patch, \\(\\tau \\le \\tfrac{2}{3}\\,\\mu F_n a\\) with friction coefficient \\(\\mu\\), normal force "
        "\\(F_n\\) and patch radius \\(a\\): 20&#8202;N&#8202;mm at \\(\\mu = 1\\), \\(F_n = 10\\)&#8202;N and \\(a = 3\\)&#8202;mm. The force at which "
        "the detents and the dovetail lips let go in 85A TPU is unmeasured; tests T1 and T2 measure it.</p>",
        table(["Load", "Path", "Numbers", "Status"], rows,
              "Load paths of the tip attachment (<code>decision.json</code>, tip_attachment).", text_cols=(0, 1, 2, 3),
              key="loads"),
        f"<ol class='steps'>{tests}</ol>",
    ])


def clearance_section():
    names = {"dip_link_new": "New DIP-link features", "clamp_plate": "Clamp plate", "board": "Board",
             "fasteners": "Screws and nuts", "wires": "Wires on the DIP link", "qwiic_connector": "Qwiic header, plug, pigtail"}
    other = {"mcp_link": "MCP link", "servo_pip": "PIP servo"}
    rows = [[names[k], f(v["min_mm"], 3), f(v["pip_deg"], 1), other.get(v["part"], v["part"]),
             "1.0" if k in ("wires",) else "0.5"]
            for k, v in CL.items() if k in names]
    T = CHK["to_pip_servo"]
    ra = CA["requirements"]
    w = CHK["walls"]
    wname = {"window": "the window plate", "bridge": "the bridge behind the mount face", "clamp": "the clamp plate's rims",
             "keys": "the walls beside the keys"}
    thin = {}
    for k, v in w.items():
        if v["min_wall_mm"] < 0.9:
            nm = wname.get(k.split("_")[0], k)
            thin[nm] = min(thin.get(nm, 9.0), v["min_wall_mm"])
    cv = json.load(open(os.path.join(FIN, f"{PFX}_clearance_curve.json")))
    q = cv["q_deg"]
    lo = [min(r["mcp_link"][i] for r in cv["rows"].values()) for i in range(len(q))]
    los = [min(r["servo_pip"][i] for r in cv["rows"].values()) for i in range(len(q))]
    k_star = min(range(len(q)), key=lambda i: min(lo[i], los[i]))
    dn = cv["rows"]["dip_link_new"]
    others = min(min(min(r["mcp_link"]), min(r["servo_pip"])) for k, r in cv["rows"].items() if k != "dip_link_new")
    k_s = min(range(len(q)), key=lambda i: dn["servo_pip"][i])
    return "".join([
        f"<p>Every new part clears the MCP link and the PIP servo over the joint range (Table&#160;[[tab:clear]], "
        f"Figure&#160;[[fig:sweep]]); the least clearance is {f(WORST['min_mm'], 3)}{MM}, between the new DIP-link "
        f"features and the MCP link at PIP {f(WORST['pip_deg'], 0)}{DEG}. The new DIP-link features stay "
        f"{f(min(dn['mcp_link']), 3)} to {f(max(dn['mcp_link']), 3)}{MM} from the MCP link at every angle and come within "
        f"{f(dn['servo_pip'][k_s], 2)}{MM} of the PIP servo at PIP {f(q[k_s], 0)}{DEG}; the other parts keep "
        f"{f(others, 2)}{MM} or more. The clevis arms are the current link's, unchanged. The chips "
        f"stay {f(T['chips_min_mm'])}{MM} or more from the PIP servo ({f(T['chips_current_plate_min_mm'])}{MM} on the "
        f"current plate).</p>",
        table(["Part", "Least clearance (mm)", "At PIP (deg)", "Against", "Requirement (mm)"], rows,
              "Clearance of the final's parts over PIP &#8722;18 to 92&#176; (<code>final/20261018-dip_link_final_check.json</code>).",
              text_cols=(0, 3), key="clear"),
        film(f"final/media/{PFX}_pip_sweep.mp4", f"final/media/{PFX}_pip_sweep.png",
             "PIP joint from &#8722;18 to 92&#176; and back with concept a's final parts on the DIP link, seen along the PIP "
             "axis from the horn side, the MCP link and the PIP servo translucent; flexion turns the tip to the left. Right: "
             "least clearance of the parts on the DIP link to the MCP link and to the PIP servo at each angle, and the part "
             "that comes closest (<code>final/20261018-dip_link_final_clearance_curve.json</code>, 0.5&#176; steps, values "
             f"capped at 6&#8202;mm). The poster frame is PIP {f(q[k_star], 1)}&#176;, where the clearance is least.", "sweep"),
        f"<p>Against the specification, R1 (gap {f(EV['gap_mm'])} against 0.52{MM}), R5 (21.0{MM} band, FR-4 edge 0.5{MM} "
        f"inside the rim), R6 (clevis arms kept), R7, R9 (tip and board replaceable with hand tools) and R12 are met in the "
        f"model; R2 and R3 wait on a print. R10 asks that the board come no nearer the PIP servo than on the current plate "
        f"({f(T['spec_R10_mm'], 1)}{MM}); the FR-4's proximal corner comes within {f(T['main_square_min_mm'])}{MM} at PIP "
        f"{f(T['main_square_pip_deg'], 1)}{DEG}, as any board placed this near the tip does. R11's 0.9{MM} wall rule is "
        f"missed by "
        + ", ".join(f"{k} ({f(v)}{MM})" for k, v in sorted(thin.items(), key=lambda kv: kv[1])) +
        f": the window plate's thickness follows the chips' height. R4 is met for the board's retention and the tip's "
        f"channels ({f(min(v['min_mm'] for v in CHK['channels_to_magnets_R4'].values()))}{MM} or more from a "
        f"magnet).</p>",
    ])


def sensing_section():
    rows = []
    for k, v in SF.items():
        d, p = v["design"], v["plate_now"]
        rows.append([TIPN[k], f(min(d["chip_to_magnet_centre_mm"])), f(min(d["chip_to_magnet_surface_mm"])),
                     f(d["B_U1_mT"]), str(d["gain_sel"]), f(min(p["chip_to_magnet_centre_mm"])), f(p["B_U1_mT"]),
                     str(p["gain_sel"])])
    fr = []
    for lay, lab in (("four_d3x2_d5", "Block, four magnets"), ("five_z7p4", "Block, five magnets (corners 7.4&#8202;mm apart)")):
        for rec, g in (("w85_30", "30"), ("w85_50", "50")):
            fr.append([f"{lab}, {g}&#8202;% gyroid", f(srow(lay, rec, "design")["centre_force_N"], 3),
                       f(srow(lay, rec, "chips_1.00")["centre_force_N"], 3),
                       f(srow(lay, rec, "chips_0.80")["centre_force_N"], 3),
                       f(srow(lay, rec, "plate_now")["centre_force_N"], 3),
                       f(srow(lay, rec, "design")["false_force_N_per_um"]["x"]["normal"], 3)])
    return "".join([
        f"<p>With the as-printed magnets the final resolves {f(FR['four_30'], 3)}{NN} with four magnets and "
        f"{f(FR['five_z7p4_30'], 3)}{NN} with five at 30&#8202;% gyroid, about half the current plate's values "
        f"(Table&#160;[[tab:force]]). The five-magnet block's corner magnets were printed 7.4{MM} apart instead of "
        f"7.0{MM} (2.9{MM} seats); moving them changes the five-magnet resolution from 0.061 to "
        f"{f(FR['five_z7p4_30'], 3)}{NN}. One micrometre of board motion toward the magnets reads as "
        f"{f(srow('four_d3x2_d5', 'w85_30', 'design')['false_force_N_per_um']['x']['normal'], 3)}{NN} with four "
        f"magnets, which is why the board is clamped. Chips 0.10{MM} taller or shorter than the window plate assumes "
        f"move the gap to 0.27 or 0.47{MM} and the resolution by 2&#8211;3&#8202;%.</p>",
        table(["Tip", "Chip to magnet centre (mm)", "Chip to magnet surface (mm)", "Field at U1 (mT)", "GAIN_SEL",
               "Current plate: centre (mm)", "Current plate: field at U1 (mT)", "Current plate: GAIN_SEL"], rows,
              "Static field and distances at the five tips' printed magnet positions, final and current plate "
              "(<code>final/20261018-dip_link_final_sense.json</code>, static_fields; U1 is the centre chip).",
              key="fields"),
        table(["Tip and infill", "Final (N)", "Chips 1.00&#8202;mm (N)", "Chips 0.80&#8202;mm (N)", "Current plate (N)",
               "False force (N per &#181;m)"], fr,
              "Force resolution at the contact point. Chips 1.00 and 0.80&#8202;mm: the final with the datasheet's "
              "extreme package heights under a window plate printed for 0.90&#8202;mm.", key="force"),
    ])


def drawing_section():
    def pic(tag):
        return f'<img alt="" src="{data_uri(os.path.join(FIN, "media", f"{PFX}_drawing_{tag}.png"))}">'
    fits = [[esc(r["feature"]), esc(r["nominal"]), esc(cap(r["fit"]))] for r in EV["fits"]]
    return ("<p>The dimensioned drawing (<code>final/media/20261018-dip_link_final_drawing.png</code>, one sheet with the "
            "fits) gives the stack at the centre chip with the Qwiic header in its pocket, the sections through the screws "
            "and the pins, and the dovetail key against its channel (Figures&#160;[[fig:drawab]] and [[fig:drawcd]]); "
            "Table&#160;[[tab:fits]] lists the critical fits and their tolerances. The window plate's thickness and the "
            "pin diameter are set from measurements on the owner's board and from the pin-fit coupon before the first "
            "print.</p>"
            "<p>For editing the parts in Onshape, <a href=\"onshape/20261018-dip_link_onshape_parts.step.zip\">"
            "<code>onshape/20261018-dip_link_onshape_parts.step.zip</code></a> contains each part as an exact solid: the board "
            "as supplied, with J1 and the mated Qwiic plug, at the placements of concepts a and b; concept a's final parts; "
            "concept b's new body and cover. <a href=\"onshape/20261018-dip_link_onshape_tips.step.zip\"><code>"
            "onshape/20261018-dip_link_onshape_tips.step.zip</code></a> contains concept a's block tip with its channels and "
            "concept b's pad as faceted solids, and <a href=\"onshape/20261018-dip_link_onshape_context_stl.zip\"><code>"
            "onshape/20261018-dip_link_onshape_context_stl.zip</code></a> the current DIP link and the MCP link and PIP servo "
            "at PIP &#8722;18, 0 and 92&#176; as meshes. The files use the coordinates of the DIP link STL, "
            "\\(x_D = -x_F\\), \\(y_D = y_F + 21.25\\), \\(z_D = -z_F - 51.55\\) (mm), so they line up with the DIP link "
            "in the owner's Part Studio (<code>scripts/dip_link_onshape_export.py</code>).</p>"
            + figure(pic("a") + pic("b"), "Drawing panels A, the stack at y = 0 through U1, and B, the section through the "
                     "M1.6 screws and nuts (mm).", "drawab")
            + figure(pic("c") + pic("d"), "Drawing panels C, the section through the pins and the dovetail keys, and D, a "
                     "key (grey) against its channel in the tip (dashed, tan) (mm).", "drawcd")
            + table(["Feature", "Nominal", "Fit and tolerance"], fits, "Critical fits and tolerances of the final (mm; "
                    "<code>final/evaluation.json</code>, fits).", text_cols=(0, 1, 2), key="fits"))


def print_section():
    rows = []
    for k, v in PS["parts"].items():
        if k == "tips":
            rows.append(["85A tips", esc(v["tool"]), esc(cap(v["material"])), esc(cap(v["orientation"])),
                         "As the 2.9&#8202;mm sheets", esc(cap(v["infill"])), esc(cap(v.get("pauses", "")))])
            continue
        nm = {"dip_link": "DIP link", "clamp_plate": "Clamp plate", "pin_fit_coupon": "Pin-fit coupon"}.get(k, cap(k))
        tool = "PLA tool (3, 4 or 5)" if "PLA tool" in v["tool"] else cap(v["tool"])
        rows.append([esc(nm), esc(tool), esc(cap(v["material"])),
                     esc(cap(v["orientation"])), f"{v.get('layer_mm', '')} mm, {v.get('perimeters', '')} perimeters",
                     esc(cap(v.get("infill", ""))), esc(cap(v.get("supports", v.get("use", "")))) +
                     (f"; {esc(v['after'])}" if v.get("after") else "")])
    pl = [[esc(cap(r["item"])), esc(cap(r.get("material", ""))), esc(str(r.get("qty", "")))] for r in PL]
    seq = "".join(f"<li>{esc(s)}</li>" for s in AS["sequence"])
    return "".join([
        f"<p>All rigid parts print on the Prusa XL with 0.4{MM} nozzles; the PLA parts go on a PLA tool, which is tool 3, "
        f"4 or 5 in the owner's v17 template (tool 1 is loaded with 95A TPU and tool 2 with Siraya 85A). STEP and STL files of every "
        f"part are in <code>final/cad/</code>, print files in <code>final/print/</code>.</p>",
        table(["Part", "Tool", "Material", "Orientation", "Layers", "Infill", "Supports and finishing"], rows,
              "XL print settings (<code>final/20261018-dip_link_final_print_settings.json</code>).",
              text_cols=(0, 1, 2, 3, 4, 5, 6), key="print"),
        table(["Item", "Material", "Quantity"], pl, "Parts list (<code>final/20261018-dip_link_final_parts_list.json</code>).",
              text_cols=(0, 1, 2), key="parts"),
        f"<p>Assembly sequence:</p><ol class='steps'>{seq}</ol>",
    ])


def checks_section():
    items = "".join(f"<li>{esc(s)}</li>" for s in AS["first_print_checks"])
    return ("<p>At the first print the owner checks the board pocket, the clamp preload and the tip fit, in this "
            f"order:</p><ol class='steps'>{items}</ol>")


def limitations():
    return ("<p>Every number on this page comes from models: the force resolution from the readiness study's finite "
            "elements of the 85A block and a dipole field model, the clearance from the CAD meshes, the retention of the "
            "tip from rigid-body levers. The window plate's thickness assumes 0.90&#8202;mm chips; the owner's board sets "
            "it. The board's flatness on a window plate printed as vertical perimeters, the screws' preload in PLA and its "
            "creep, the detents' retention in 85A TPU and the cable's route across the PIP joint are unmeasured. Concept b's "
            "numbers come from a coarser finite-element grid than a's and c's.</p>")


def open_items():
    items = [
        "Measure the five chips' package height and the FR-4 thickness on the owner's board with a micrometer; set "
        "<code>chip_h</code> and <code>fr4</code> in <code>scripts/dip_link_concept_a.py</code>, rerun "
        "<code>scripts/dip_link_board_mount.py build</code>, <code>check</code> and <code>sense</code>. A chip height "
        "outside 0.85&#8211;0.95&#8202;mm changes the window plate.",
        "Print the pin-fit coupon (<code>final/print/20261018-dip_link_final_pin_fit_coupon.stl</code>) and set "
        "<code>pin_d</code> to the pin that enters the board's holes without play.",
        "Print the attachment fit coupon (test T1) and choose the interference per flank; then reprint one "
        "four-magnet block with channels and run test T2. A tip that slides past its detent under 5&#8202;N fails the "
        "design load.",
        "Bench check of the clamp from the specification: zero-load field before and after 50 presses at 3&#8202;N and "
        "after 1&#8202;h at 5&#8202;N; a drift above one force-resolution step means the board moves in its clamp.",
        "Route the Qwiic cable across the PIP joint on the printed finger over &#8722;18 to 92&#176; and measure the "
        "least gap to the PIP servo; below 1.0&#8202;mm the cable needs a guide on the DIP link.",
    ]
    return "<ol class='steps'>" + "".join(f"<li>{i}</li>" for i in items) + "</ol>"


def byline():
    return ("Generator <code>scripts/dip_link_board_mount.py</code> (on <code>scripts/dip_link_concept_a.py</code>) &#183; "
            "builder <code>scripts/dip_link_board_mount_page.py</code> &#183; files "
            "<code>docs/experiments/20261018-dip_link_board_mount/</code> (<code>spec.json</code>, <code>decision.json</code>, "
            "<code>concept_a/</code>, <code>concept_b/</code>, <code>concept_c/</code>, <code>final/</code>)")


def main():
    REFS.update({})
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "LEDE": lede(), "GLOSSARY": glossary(), "DESIGN": design_section(),
         "CONCEPTS": concepts_section(), "QWIIC": qwiic_section(), "TIP": tip_section(), "CLEARANCE": clearance_section(),
         "SENSING": sensing_section(), "DRAWING": drawing_section(), "PRINT": print_section(), "CHECKS": checks_section(),
         "LIMITS": limitations(), "OPEN": open_items()}
    v["FOOTER"] = ("<p>Rebuild: the run order in the docstring of <code>scripts/dip_link_board_mount.py</code> (build, step, "
                   "check, sense, render, drawing, docs, eval, decision), then <code>.venv/bin/python "
                   "scripts/dip_link_board_mount_page.py</code>. Files in <code>docs/experiments/20261018-dip_link_board_mount/"
                   "</code>; hand-off <code>docs/handoff/20261018-calibration_and_dip_finish.md</code>. Related pages: "
                   "<a href=\"../20261016-sim2real_fingertip_readiness/20261016-cad_hand_fingertip_deployment_readiness.html\">"
                   "deployment readiness of the CAD hand with printed 85A fingertips</a>, <a href=\"../20261018-fingertip_print_procedure/"
                   "20261018-siraya85a_fingertip_print_procedure.html\">print procedure for the 85A fingertips</a>. Topic "
                   "overview: <a href=\"../../overviews/printed_fingertips.html\">printed fingertips and force sensing</a>.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    miss = sorted(set(re.findall(r"\[\[((?:fig|tab|sec):[a-z0-9_]+)\]\]", t)) - set(REFS))
    if miss:
        raise SystemExit(f"unresolved references: {miss}")
    t = re.sub(r"\[\[((?:fig|tab|sec):[a-z0-9_]+)\]\]", lambda m: str(REFS[m.group(1)]), t)
    t, n = render_tex(t)
    t = retro_style.apply(t)
    open(OUT, "w").write(t)
    print(f"formulas {n}; figures {FIG[0]}; tables {TAB[0]}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
