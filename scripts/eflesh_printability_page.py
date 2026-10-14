#!/usr/bin/env python3
"""Page: printability, stiffness and a one-command build of eFlesh lattice fingertips (2026-10-13).

Owner, 2026-10-13: "something like eflesh is still going to be most useful for force sensing resolution ... is it
possible to improve the eflesh generation process? ... these micro-variations in generated structures never actually
realized themselves in the real print ... delamination was a big issue with the tighter spacing. a 0.4mm nozzle really
doesn't create that much detail." Past fingers: ~/wxie_workspace/eFlesh/microstructure, ~/wxie_workspace/fingers (v17
printed).

Data (docs/experiments/20261013-eflesh_printability/): cutcell_test.json, strut_map.json, volume_sweep.json,
audit.jsonl (scripts/eflesh_printability.py), build/*.json (scripts/eflesh_build.py), fem.jsonl
(scripts/eflesh_tip_fem.py); the earlier eFlesh tip on its design mesh and the gyroid tip's finite-element rows come from
docs/experiments/20261009-tpu_tip_fem/.

    .venv/bin/python scripts/eflesh_printability_page.py
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261013-eflesh_printability")
TIP = os.path.join(ROOT, "docs/experiments/20261009-tpu_tip_fem")
OUT = os.path.join(D, "20261013-eflesh_lattice_printability.html")
TPL = os.path.join(ROOT, "scripts/eflesh_printability_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
TIP_PAGE = "../20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html"
FIG, TAB, REFS = [0], [0], {}
HS = "&#8202;"
EXTRA_CSS = """
:root{--s-x:#2a78d6;--s-z:#eb6834;--s-sh:#1baf7a;--s-ref:#48545D;--s-pads:#8a4fbf;--s-res:#c2185b}
.tw th{white-space:normal !important;vertical-align:bottom}
.tw td:first-child{min-width:11em}
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
figure.films{display:grid;grid-template-columns:minmax(0,1fr);gap:14px}
figure.films video,figure.pics img{width:100%;height:auto}
ul.refs li{margin-bottom:8px}
"""


def load(name):
    return json.load(open(os.path.join(D, name)))


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def data_uri(path, mime):
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def figure(body, caption, legend=None, key=None, cls="diagram"):
    FIG[0] += 1
    if key:
        REFS[f"fig:{key}"] = FIG[0]
    lg = P._legend_html(legend) if legend else ""
    return f'<figure class="{cls}">{body}{lg}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def table(head, body, caption, note=None, text_cols=(0,), key=None):
    TAB[0] += 1
    if key:
        REFS[f"tab:{key}"] = TAB[0]
    cls = lambda i: "" if i in text_cols else " class=num"  # noqa: E731
    out = [f"<p class='tcap'><b>Table&#160;{TAB[0]}.</b> {caption}</p>",
           "<div class='tw'><table><thead><tr>" + "".join(f"<th{cls(i)}>{h}</th>" for i, h in enumerate(head)) + "</tr></thead><tbody>"]
    for r in body:
        out.append("<tr>" + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table></div>")
    if note:
        out.append(f"<p class='tnote'>{note}</p>")
    return "".join(out)


def film(name, caption, key):
    p = os.path.join(D, "media", f"20261013-{name}.mp4")
    poster = os.path.join(D, "media", f"20261013-{name}_poster.png")
    v = (f'<video controls muted loop playsinline preload="metadata" poster="{data_uri(poster, "image/png")}" '
         f'src="{data_uri(p, "video/mp4")}"></video>')
    return figure(v, caption, key=key, cls="films")


def pc(x, nd=0):
    return f"{100 * x:.{nd}f}{HS}%"


def tmin(s):
    """'1h 12m 51s' -> minutes."""
    h = re.search(r"(\d+)h", s); m = re.search(r"(\d+)m", s); sec = re.search(r"(\d+)s", s)
    return (int(h.group(1)) if h else 0) * 60 + (int(m.group(1)) if m else 0) + (int(sec.group(1)) if sec else 0) / 60


def fmt_time(s):
    m = tmin(s)
    return f"{m:.0f}{HS}min" if m < 90 else f"{m / 60:.1f}{HS}h"


# ------------------------------------------------------------------------------------------ data

class Data:
    def __init__(self):
        self.cut = load("cutcell_test.json")
        self.map = load("strut_map.json")
        self.vol = load("volume_sweep.json")
        self.audit = {r["name"]: r for r in jl(os.path.join(D, "audit.jsonl"))}
        self.builds = {}
        for f in sorted(os.listdir(os.path.join(D, "build"))):
            if f.endswith(".json"):
                r = json.load(open(os.path.join(D, "build", f)))
                self.builds[r["name"]] = r
        self.fem = {r["tip"]: r for r in jl(os.path.join(D, "fem.jsonl"))}
        import tpu_tip_print_plan as PP          # force noise with the current MLX90393 model, from the stored magnet motions
        for r in self.fem.values():
            if "1" in r["per_force"]:
                fr = dict(magnet=r["magnet"], depth_mm=r["depth_mm"], per_force={"1": r["per_force"]["1"]},
                          **{f"shear_{a}_magnet_{q}_per_N": [0, 0, 0] for a in "zy" for q in ("t_um", "w_mrad")})
                r["field"] = {f"{g:g}": {k: v for k, v in PP.field(fr, g).items() if k in ("B0_mT", "S_normal_norm", "sigF_normal_N")}
                              for g in (1.0, 4.0)}
        s = json.load(open(os.path.join(TIP, "summary.json")))
        self.ef_design = s["resolved"]["eflesh_c8"]
        self.ef_gpu = s["gpu"]["eflesh_c8_surf"]
        self.magrows = {(r["recipe"], r["magnet"], r["depth_mm"]): r for r in jl(os.path.join(TIP, "magnet_runs.jsonl"))}
        cells = [json.loads(l) for l in open(os.path.join(TIP, "cells.jsonl"))]
        self.ef_cell = next((c for c in cells if c.get("kind") == "eflesh" and abs(c.get("E_rel_target", 0) - 0.0035) < 1e-9
                             and abs(c.get("nu_s", 0) - 0.45) < 1e-9), None)

    def thick(self, E, c):
        r = min(self.map["rows"], key=lambda r: abs(r["E"] - E))
        return [t * c for t in r["thickness_cell_units"]]


# ------------------------------------------------------------------------------------------ figures

def svg_struts(X: Data):
    W, H = 860, 360
    out = P._svg_open(W, H, "Strut thickness against eFlesh's modulus parameter for three cell sizes")
    fx, fy = P._panel(out, 90, 40, 560, 250, (0.001, 0.02), (0, 1.4), [0.001, 0.002, 0.005, 0.01, 0.02], [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
                      "eFlesh modulus parameter E (log)", "Strut thickness (mm)", logx=True, yfmt="{:.1f}")
    out.append(f'<rect x="{fx(0.001):.1f}" y="{fy(0.45):.1f}" width="{fx(0.02) - fx(0.001):.1f}" height="{fy(0) - fy(0.45):.1f}" '
               f'style="fill:rgba(200,20,40,0.08)"/>')
    out.append(f'<line x1="{fx(0.001):.1f}" x2="{fx(0.02):.1f}" y1="{fy(0.45):.1f}" y2="{fy(0.45):.1f}" style="stroke:#b00020;stroke-dasharray:5 4"/>'
               f'<text x="{fx(0.02) + 8:.1f}" y="{fy(0.45) + 5:.1f}" style="fill:#b00020">One bead, 0.45{HS}mm</text>')
    col = {8: "var(--s-x)", 6: "var(--s-sh)", 4: "var(--s-z)"}
    for c in (8, 6, 4):
        pts_lo = [(r["E"], min(r["thickness_cell_units"]) * c) for r in X.map["rows"]]
        pts_hi = [(r["E"], max(r["thickness_cell_units"]) * c) for r in X.map["rows"]]
        P._path(out, fx, fy, pts_lo, col[c], width=2.4)
        P._path(out, fx, fy, pts_hi, col[c], dashed=True, width=2.0)
        out.append(f'<text x="{fx(0.02) + 8:.1f}" y="{fy(pts_hi[-1][1]) + 5:.1f}" style="fill:{col[c]}">{c}{HS}mm cells</text>')
    out.append("</svg>")
    leg = [("Thinnest strut", "var(--ink2)", False, None), ("Thickest strut", "var(--ink2)", True, None)]
    return "".join(out), leg


def svg_support(X: Data):
    W, H = 860, 360
    out = P._svg_open(W, H, "Islands under half supported and supported area against cell size, two nozzles")
    A = X.audit
    pts = {(0.4, c): A.get(f"cube24_c{c}") for c in (8, 6, 4)}
    pts.update({(0.25, c): A.get(f"cube24_c{c}_n025") for c in (6, 4)})
    fx, fy = P._panel(out, 80, 44, 300, 230, (3.5, 8.5), (0, 80), [4, 6, 8], [0, 20, 40, 60, 80], "Cell size (mm)",
                      "Islands under half supported (%)")
    gx, gy = P._panel(out, 520, 44, 300, 230, (3.5, 8.5), (0, 100), [4, 6, 8], [0, 20, 40, 60, 80, 100], "Cell size (mm)",
                      "Extruded area on the layer below (%)")
    for nz, colr, dash in ((0.4, "var(--s-x)", False), (0.25, "var(--s-z)", True)):
        a = [(c, 100 * pts[(nz, c)]["islands_under_half_supported"]) for c in (4, 6, 8) if pts.get((nz, c))]
        b = [(c, 100 * pts[(nz, c)]["supported_area"]) for c in (4, 6, 8) if pts.get((nz, c))]
        P._path(out, fx, fy, a, colr, dashed=dash); P._path(out, gx, gy, b, colr, dashed=dash)
        for c, v in a:
            P._marker(out, fx(c), fy(v), colr)
        for c, v in b:
            P._marker(out, gx(c), gy(v), colr)
    out.append("</svg>")
    leg = [("0.4" + HS + "mm nozzle, 0.2" + HS + "mm layers", "var(--s-x)", False, "circle"),
           ("0.25" + HS + "mm nozzle, 0.1" + HS + "mm layers", "var(--s-z)", True, "circle")]
    return "".join(out), leg


# ------------------------------------------------------------------------------------------ sections

GLOSSARY = [
    ("eFlesh modulus parameter \\(E\\)", "Target Young&#8217;s modulus of the lattice over the solid&#8217;s, the input of eFlesh&#8217;s map from "
     "material to geometry (pattern 0646, Poisson&#8217;s ratio 0.09); the printed v17 used 0.0035."),
    ("Cell size", "Edge of eFlesh&#8217;s cubic cell, in mm; v17 used 8&#8202;mm."),
    ("Strut thickness", "eFlesh&#8217;s four thickness parameters times the cell size, in mm. Bead: one extrusion line, 0.45&#8202;mm wide on the "
     "0.4&#8202;mm nozzle; PrusaSlicer&#8217;s Arachne generator varies it and prints 0.34&#8202;mm at the least."),
    ("Island", "A connected region of one layer&#8217;s beads. The nozzle prints each island as its own path, with a retraction (2.5&#8202;mm of "
     "TPU pulled back in the owner&#8217;s FLEX profile) on the way to the next."),
    ("Support", "Share of an island&#8217;s area that lies on the beads of the layer below, measured by drawing every bead at its slicer width "
     "on a 0.025&#8202;mm grid. Islands under half supported: the share of all islands of the layers counted whose support is below 50&#8202;%. "
     "Supported area: the share of all extruded area that lies on the layer below."),
    ("Pad modulus \\(E^*\\)", "The modulus of the pads&#8217; pressure law that best reproduces a tip&#8217;s approach under the screwdriver "
     f"shaft (radius 12.5&#8202;mm) at 0.5, 1 and 3&#8202;N, as on the <a href=\"{TIP_PAGE}\">printed fingertip page</a>; the contact models "
     "use 10&#8202;MPa."),
    ("Magnet travel, force noise", "Travel: displacement of the magnet toward the back face per newton on the face, in &#181;m/N. Force noise: "
     "the load whose field change at an MLX90393 1&#8202;mm behind the back face equals one 141&#8202;Hz sample&#8217;s noise, in N (point "
     "dipole of a 4.76 &#215; 1.59&#8202;mm N52 disc whose top is 5&#8202;mm under the face)." + " It combines the finite-element travel, a dipole field and the MLX90393 datasheet noise (GAIN_SEL 7, RES 0, OSR 0, DIG_FILT 5); drift and stray fields are on the <a href=\"../20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html\">readout page</a>."),
    ("Open lattice, skin", "Open: the lattice&#8217;s struts end at the part&#8217;s faces, as eFlesh prints its sensors; the v17 finger is "
     "open except for its solid back. Skin: a closed 0.9&#8202;mm layer (two beads) over the faces, which eFlesh does not have; "
     "<code>scripts/eflesh_build.py</code> adds one only with <code>--skin</code>."),
    ("As-printed tip", "The tip as PrusaSlicer extrudes it: the G-code&#8217;s beads drawn at their widths, averaged onto 0.1&#8202;mm voxels, "
     "each 0.2&#8202;mm layer two voxels tall, solved as one TPU (30&#8202;MPa, Poisson&#8217;s ratio 0.45) on the GPU."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def tools_section(X: Data):
    c = X.cut
    st = {(r["input"][:3], r["cell_mm"]): r for r in c["stitch"]}
    t8, t6 = st[("SR2", 8)], st[("SR2", 6)]
    cubes = [r for r in c["stitch"] if r["input"].startswith("24")]
    txt = (f"<p>eFlesh&#8217;s cut-cell tool, <code>cut_cells_cli</code>, the step that fits the lattice to a shape, ends with a segmentation "
           f"fault after {c['cut_cells_cli']['seconds']:.2f}&#8202;s on every input tried on this workstation: the SR2 fingertip at 8 and "
           f"6&#8202;mm cells and a plain 16&#8202;mm cube. The backtrace stops in OpenVDB&#8217;s <code>meshToLevelSet</code>, called from "
           f"<code>mesh2sdf</code> before any cell is processed, which points at the OpenVDB and TBB build rather than at the shapes. The "
           f"notebook in the owner&#8217;s copy (commit 3f8ffe9, 2025-10-27) assigns <code>cut_cells_cli</code> and overwrites it on the next "
           f"line with <code>stitch_cells_cli</code>, which tiles whole cells over the input&#8217;s bounding box and ignores the surface. "
           f"Given the whole SR2 tip (17.0 &#215; 14.8 &#215; 21.4&#8202;mm), it returns a {t8['extent_mm'][0]:.0f} &#215; "
           f"{t8['extent_mm'][1]:.0f} &#215; {t8['extent_mm'][2]:.0f}&#8202;mm block of 8&#8202;mm cells ({t8['faces'] / 1e6:.2f}&#8202;M "
           f"triangles, {t8['obj_MB']:.0f}&#8202;MB) and a {t6['extent_mm'][0]:.0f} &#215; {t6['extent_mm'][1]:.0f} &#215; "
           f"{t6['extent_mm'][2]:.0f}&#8202;mm block of 6&#8202;mm cells ({t6['faces'] / 1e6:.2f}&#8202;M triangles, {t6['obj_MB']:.0f}&#8202;MB). "
           f"This is why only boxes came out of the owner&#8217;s runs.</p>"
           f"<p>File size follows the number of cells, since the tool meshes every cell at 50 samples per edge: "
           + ", ".join(f"{r['obj_MB']:.0f}&#8202;MB" for r in cubes)
           + " for a 24&#8202;mm cube at 8, 6 and 4&#8202;mm cells. The surface it writes has a median edge of 0.16&#8202;mm, a third of a "
             "bead, so the slicer reduces most of that detail to the same beads.</p>")
    return txt


def struts_section(X: Data):
    svg, leg = svg_struts(X)
    lo8, hi8 = X.thick(0.001, 8), X.thick(0.01, 8)
    v = X.vol
    a, b = v["v15_E0.002_c8"], v["v14_E0.01_c8"]
    txt = (f"<p>eFlesh&#8217;s map from the modulus parameter to geometry moves the struts very little. From \\(E\\) = 0.001 to 0.01, a "
           f"tenfold range of target modulus, two of the four thickness parameters stay at their bounds (1.0 and 1.2&#8202;mm at 8&#8202;mm "
           f"cells) and the other two grow from {lo8[1]:.2f} to {hi8[1]:.2f}&#8202;mm and from {lo8[2]:.2f} to {hi8[2]:.2f}&#8202;mm, about half a "
           f"bead (Figure&#160;[[fig:struts]]). The lattice&#8217;s solid fraction goes from {pc(a['design_frac'], 1)} at \\(E\\) = 0.002 to "
           f"{pc(b['design_frac'], 1)} at 0.01. PrusaSlicer reproduces both within 3&#8202;%: the owner&#8217;s FLEX profile extrudes "
           f"{pc(a['print_frac'], 1)} and {pc(b['print_frac'], 1)} (filament volume over the box). The slicer does print the designed "
           f"difference; the difference is a fraction of a bead per strut, the size of the variation between prints of TPU, so a fivefold "
           f"step in \\(E\\) need not show on the bench.</p>"
           f"<p>The design map also starts from a stiffer point than its label: our homogenization of eFlesh&#8217;s cell at \\(E\\) = 0.0035 "
           f"returns {X.ef_cell['Ex']:.4f} of the solid&#8217;s modulus ({pc(X.ef_cell['rho'], 1)} solid), {X.ef_cell['Ex'] / 0.0035:.1f} times the "
           f"target (<a href=\"{TIP_PAGE}#cells\">printed fingertip page</a>).</p>" if X.ef_cell else "")
    rows = []
    for nm, lab in (("v15_E0.002_c8", "0.002"), ("v17_E0.0035_c8", "0.0035, six pouches"), ("v14_E0.01_c8", "0.01")):
        r = v[nm]
        E = r["E"]
        th = X.thick(E, 8)
        rows.append([lab, f"{min(th):.2f}&#8211;{max(th):.2f}", pc(r["design_frac"], 1), pc(r["print_frac"], 1),
                     f"{r['paths_med']:.0f}", f"{r['retr_per_layer']:.0f}", fmt_time(r["time"])])
    tab = table(["Modulus parameter \\(E\\)", "Struts (mm)", "Designed<br>solid", "Extruded<br>solid", "Paths<br>per layer",
                 "Retractions<br>per layer", "Print<br>time"],
                rows, "The owner&#8217;s three 8&#8202;mm lattices sliced alone with the v17 settings (2 perimeters, no infill, 5 top and 3 bottom "
                "layers). Extruded solid: filament volume over the lattice&#8217;s box; the 0.0035 lattice is a smaller box with six pouches.",
                key="sweep")
    return txt + figure(svg, "Thinnest (solid) and thickest (dashed) strut of eFlesh&#8217;s pattern 0646 against the modulus parameter, "
                             "for 8, 6 and 4&#8202;mm cells (eFlesh&#8217;s <code>Material2Geometry</code>, Poisson&#8217;s ratio 0.09). Red: "
                             "thinner than one bead of the 0.4&#8202;mm nozzle.", leg, key="struts") + tab


def layers_section(X: Data):
    A = X.audit
    svg, leg = svg_support(X)
    c8, c6, c4 = A["cube24_c8"], A["cube24_c6"], A["cube24_c4"]
    f6, f4 = A["cube24_c6_n025"], A["cube24_c4_n025"]
    v17, g20 = A["v17_project"], A["tip_gyroid20"]
    txt = (f"<p>A strut lattice prints as many small islands per layer, and the islands shrink with the cell. In one 24&#8202;mm cube of "
           f"eFlesh&#8217;s lattice at \\(E\\) = 0.0035, the share of islands that rest on the layer below over less than half their area goes "
           f"from {pc(c8['islands_under_half_supported'])} at 8&#8202;mm cells to {pc(c6['islands_under_half_supported'])} at 6&#8202;mm and "
           f"{pc(c4['islands_under_half_supported'])} at 4&#8202;mm, and the share of all extruded area that rests on the layer below from "
           f"{pc(c8['supported_area'])} to {pc(c6['supported_area'])} and {pc(c4['supported_area'])} (Figure&#160;[[fig:support]], "
           f"Table&#160;[[tab:audit]]). The median island shrinks from {c8['island_area_median_mm2']:.2f} to {c4['island_area_median_mm2']:.2f}&#8202;mm&#178;, "
           f"one loop of a bead around a strut, while the layer shift of an inclined strut stays near 0.2&#8202;mm. A bead that lies on "
           f"nothing bonds at its edges only, which is the delamination the owner saw at tight spacing; Visible Touch (UCLA) reports "
           f"&#8220;lattice delamination&#8221; in eFlesh&#8217;s reference design, and Aguilar-Segovia et al. avoid printed truss lattices for "
           f"the same islands (<a href=\"#related\">Related work</a>).</p>"
           f"<p>A 0.25&#8202;mm nozzle with 0.1&#8202;mm layers moves the limit by one step: 6&#8202;mm cells then leave "
           f"{pc(f6['islands_under_half_supported'])} of the islands under half supported, as 8&#8202;mm cells do on the 0.4&#8202;mm nozzle, and "
           f"4&#8202;mm cells {pc(f4['islands_under_half_supported'])}; the cube&#8217;s print time grows from {fmt_time(c6['time'])} to "
           f"{fmt_time(f6['time'])}. This is the slicer&#8217;s geometry only; whether TPU 95A runs through a 0.25&#8202;mm nozzle is a bench "
           f"question.</p>"
           f"<p>The printed v17 finger has a median of {v17['islands_median']:.0f} islands in each lattice layer and {v17['retracts_per_layer']:.0f} "
           f"retractions per layer, against {g20['islands_median']:.0f} island and {g20['retracts_per_layer']:.1f} retractions for the 20&#8202;% "
           f"gyroid tip, whose infill lines join its walls (Figure&#160;[[fig:film_tip]]). The gyroid&#8217;s supported area, "
           f"{pc(g20['supported_area'])}, is close to a lattice&#8217;s at 8&#8202;mm cells; it differs in printing one connected path per layer "
           f"instead of dozens of separate loops.</p>")
    rows = []
    for nm, lab in (("v17_project", "Finger v17 as printed"), ("v15_E0.002", "Lattice, \\(E\\) 0.002, 8&#8202;mm"),
                    ("v17_E0.0035", "Lattice, \\(E\\) 0.0035, 8&#8202;mm, pouches"), ("v14_E0.01", "Lattice, \\(E\\) 0.01, 8&#8202;mm"),
                    ("cube24_c8", "Cube, 8&#8202;mm cells"), ("cube24_c6", "Cube, 6&#8202;mm cells"), ("cube24_c4", "Cube, 4&#8202;mm cells"),
                    ("cube24_c6_n025", "Cube, 6&#8202;mm, 0.25&#8202;mm nozzle"), ("cube24_c4_n025", "Cube, 4&#8202;mm, 0.25&#8202;mm nozzle"),
                    ("tip_gyroid20", "SR2 tip, gyroid 20&#8202;%"), ("tip_gyroid10", "SR2 tip, gyroid 10&#8202;%")):
        r = A.get(nm)
        if not r:
            continue
        rows.append([lab, f"{r['islands_median']:.0f}", f"{r['retracts_per_layer']:.0f}", f"{r['island_area_median_mm2']:.2f}",
                     pc(r["supported_area"]), pc(r["islands_under_half_supported"], 1), fmt_time(r["time"])])
    tab = table(["Print", "Islands<br>per layer", "Retractions<br>per layer", "Median island<br>(mm&#178;)", "Supported<br>area",
                 "Islands under<br>half supported", "Print<br>time"], rows,
                "Toolpaths of each print with the owner&#8217;s PrusaSlicer FLEX profile (Prusa XL, 0.4&#8202;mm nozzle, 0.2&#8202;mm layers "
                "unless stated), lattice layers only. Cube: eFlesh&#8217;s lattice at \\(E\\) = 0.0035 filling a 24&#8202;mm cube.", key="audit")
    films = (film("eflesh_cells_8_4mm", "eFlesh&#8217;s lattice in a 24&#8202;mm cube at 8&#8202;mm (left) and 4&#8202;mm cells (right), layer by "
                  "layer as PrusaSlicer prints it: blue beads rest on the layer below, red beads on nothing, grey is the layer below.", "film_cells")
             + film("eflesh_v17_vs_gyroid_tip", "The printed v17 finger (left) and the 20&#8202;% gyroid SR2 tip (right) layer by layer at one "
                    "scale, coloured as in Figure&#160;[[fig:film_cells]].", "film_tip"))
    return txt + figure(svg, "Support of eFlesh&#8217;s lattice against cell size in a 24&#8202;mm cube for two nozzles.", leg, key="support") + tab + films


def fem_section(X: Data):
    F = X.fem
    ref = X.magrows.get(("w95_20", "d4.76x1.59", 5.0))
    rows = []
    lab = {"gyroid20": "Gyroid 20&#8202;%, walls and skins", "eflesh_open": "Lattice, open as eFlesh prints it",
           "eflesh_top": "Lattice, added skin on the palmar face", "eflesh_gap": "Lattice, added palmar skin and walls stopping 1.5&#8202;mm below it",
           "eflesh_all": "Lattice, added skin on every face", "gyroid10": "Gyroid 10&#8202;%, walls and skins"}
    for k in ("gyroid20", "gyroid10", "eflesh_open", "eflesh_top", "eflesh_gap", "eflesh_all"):
        r = F.get(k)
        if not r:
            continue
        pf = r["per_force"].get("1", {})
        fld = r.get("field", {})
        rows.append([lab[k], pc(r["solid_of_box"]), f"{pf.get('delta_um', float('nan')):.0f}", f"{r['E_best_MPa']:.2f}",
                     f"{-pf.get('magnet_t_um', [float('nan')])[0]:.1f}", f"{fld.get('1', {}).get('sigF_normal_N', float('nan')):.3f}",
                     f"{fld.get('4', {}).get('sigF_normal_N', float('nan')):.3f}"])
    if ref:
        from tpu_tip_print_plan import field as PPfield
        f1, f4 = PPfield(ref, 1.0), PPfield(ref, 4.0)
        rows.append(["Gyroid 20&#8202;%, earlier model (homogenized core, NdFeB magnet)", "&#8211;", f"{ref['per_force']['1']['delta_um']:.0f}",
                     f"{ref['E_best_MPa']:.2f}", f"{-ref['per_force']['1']['magnet_t_um'][0]:.1f}", f"{f1['sigF_normal_N']:.3f}", f"{f4['sigF_normal_N']:.3f}"])
    ed = X.ef_design
    rows.append(["Open lattice, design mesh, no magnet (earlier)", "&#8211;", f"{ed['delta_um']['1.0']:.0f}", f"{ed['E_best']:.2f}",
                 "&#8211;", "&#8211;", "&#8211;"])
    tab = table(["Tip", "Solid", "Approach<br>at 1" + HS + "N (&#181;m)", "Pad modulus<br>(MPa)", "Magnet travel<br>(&#181;m/N)",
                 "Force noise, chip<br>at 1" + HS + "mm (N)", "Chip at<br>4" + HS + "mm (N)"], rows,
                "The SR2 tip printed face up with the magnet 5&#8202;mm under the contact point, solved as printed. Solid: share of the tip&#8217;s "
                "box. The last two rows are the earlier models of the <a href=\"" + TIP_PAGE + "\">printed fingertip page</a>.", key="fem")
    sec = figure(f'<img src="{data_uri(os.path.join(D, "media", "20261013-tip_sections.png"), "image/png")}" alt="Sections of three printed tips">',
                 "Sections of the printed tips through the contact point, palmar face up, as the solver sees them: TPU black, the bonded "
                 "magnet red. The three skinned tips are additions to eFlesh&#8217;s design.", key="sections", cls="pics")
    txt = fem_text(X)
    return txt + tab + sec


def fem_text(X: Data):
    F = X.fem
    g, o, t, a, w = F.get("gyroid20"), F.get("eflesh_open"), F.get("eflesh_top"), F.get("eflesh_all"), F.get("eflesh_gap")
    ref = X.magrows.get(("w95_20", "d4.76x1.59", 5.0))
    if not (g and o and t and a and w and ref):
        return "<p>Runs in progress.</p>"
    d = lambda r: r["per_force"]["1"]["delta_um"]  # noqa: E731
    tr = lambda r: -r["per_force"]["1"]["magnet_t_um"][0]  # noqa: E731
    nz = lambda r, gap="1": r["field"][gap]["sigF_normal_N"]  # noqa: E731
    return (f"<p>Printed as eFlesh prints it, open on every face on a solid base, the lattice tip approaches {d(o):.0f}&#8202;&#181;m at "
            f"1&#8202;N (pad modulus {o['E_best_MPa']:.2f}&#8202;MPa) and its magnet travels {tr(o):.0f}&#8202;&#181;m/N, against "
            f"{d(g):.0f}&#8202;&#181;m ({g['E_best_MPa']:.1f}&#8202;MPa) and {tr(g):.0f}&#8202;&#181;m/N for the 20&#8202;% gyroid tip solved "
            f"the same way. The normal-force noise of one 141&#8202;Hz sample at a chip 1&#8202;mm behind the back face is {nz(o):.3f}&#8202;N "
            f"against {nz(g):.3f}&#8202;N (Table&#160;[[tab:fem]]).</p>"
            f"<p>Three skins, none of which eFlesh prints, close the surface for contact, and each costs magnet travel. A 0.9&#8202;mm skin on "
            f"the palmar face only gives {d(t):.0f}&#8202;&#181;m and {nz(t):.3f}&#8202;N and keeps {100 * tr(t) / tr(o):.0f}&#8202;% of the open "
            f"lattice&#8217;s travel. Walls that rise from the base and stop 1.5&#8202;mm below that skin still stiffen the tip through the "
            f"struts joined to them: {d(w):.0f}&#8202;&#181;m, {nz(w):.3f}&#8202;N and {100 * tr(w) / tr(o):.0f}&#8202;% of the travel. The same "
            f"skin on every face carries the contact to the base: {d(a):.0f}&#8202;&#181;m, {nz(a):.3f}&#8202;N and "
            f"{100 * tr(a) / tr(o):.0f}&#8202;% of the travel, no better than the gyroid tip.</p>"
            f"<p>The as-printed gyroid is softer than the earlier models of the same tip, {d(g):.0f} against {ref['per_force']['1']['delta_um']:.0f}"
            f"&#8202;&#181;m at 1&#8202;N and {tr(g):.0f} against {-ref['per_force']['1']['magnet_t_um'][0]:.0f}&#8202;&#181;m/N, because each layer "
            f"of beads rests on part of the one below and the magnet is TPU here. The open eFlesh tip lies below the pads&#8217; failure of the "
            f"pull test at 1&#8202;MPa, so a simulated pad for it needs the bed&#8217;s low-modulus checks first.</p>")


def build_section(X: Data):
    B = X.builds
    o, a, t, gw = (B.get("sr2tip_c8_E0035_open"), B.get("sr2tip_c8_E0035"), B.get("sr2tip_c8_E0035_topskin"),
                   B.get("sr2tip_c8_E0035_gapwalls"))
    txt = ("<p><code>scripts/eflesh_build.py</code> replaces the notebook, Blender and Onshape with one command that reads a closed shape "
           "and writes a PrusaSlicer project:</p><ol>"
           "<li>Shape: an STL or OBJ in print orientation, or <code>sr2tip</code>, the whole SR2 fingertip printed face up.</li>"
           "<li>Lattice: eFlesh&#8217;s own stitch tool fills the shape&#8217;s bounding box with whole cells at the requested \\(E\\) and cell "
           "size, shifted so that a cell is centred under the contact point.</li>"
           "<li>Cut and base: exact booleans in manifold3d keep the lattice inside the shape, open on its faces as eFlesh prints it, on a "
           "solid base on the bed; <code>--skin</code> adds a closed skin on every face, on the top face only, or on the top face with walls "
           "that rise from the base and stop <code>--wall-gap</code> below it. The result is watertight by construction, so no repair step "
           "follows.</li>"
           "<li>Pouches: eFlesh&#8217;s pouch around each magnet (1&#8202;mm wall and floor, a 1&#8202;mm lip whose hole is 0.8 of the "
           "cavity&#8217;s radius), the lattice cleared around it. Since 2026-10-13 the cavity is 90&#8202;% of the magnet&#8217;s diameter "
           "with its floor and top on the layer grid and a layer of air above the magnet, so the lip prints over air: in the owner&#8217;s "
           "eFlesh prints the lip printed onto the magnet and the nozzle pulled magnets out. The builds on this page predate the "
           "change.</li>"
           "<li>Project: a 3MF with the owner&#8217;s printer, filament and print settings copied from the printed v17 project, the part&#8217;s "
           "own settings (2 perimeters, no infill, solid layers for the skin and base) and an M601 pause before the first layer above the "
           "highest cavity.</li>"
           "<li>Check: PrusaSlicer slices the project; the report gives the pause found in the G-code, islands, support and print time.</li></ol>")
    rows = []
    for r, lab in ((o, "Open, as eFlesh prints it"), (t, "Added skin on the palmar face"), (gw, "Added palmar skin, gapped walls"),
                   (a, "Added skin on every face")):
        if not r:
            continue
        s = r["slice"]
        rows.append([lab, f"{r['project_MB']:.1f}", pc(r["solid_fraction"]),
                     f"{r['pause_print_z']:.1f} / {s['pause_found_before_layer_z']:.1f}", f"{s['islands_median']:.0f}", pc(s["supported_area"]),
                     pc(s["islands_under_half_supported"], 1), fmt_time(s["time"])])
    tab = table(["SR2 tip, lattice \\(E\\) 0.0035, 8" + HS + "mm cells", "Project<br>(MB)", "Solid", "Pause planned<br>/ in G-code (mm)",
                 "Islands<br>per layer", "Supported<br>area", "Islands under<br>half supported", "Print<br>time"], rows,
                "The SR2 tips built with one 4.76 &#215; 1.59&#8202;mm magnet 5&#8202;mm under the contact point. Pause: the print height of "
                "the layer PrusaSlicer holds back for the magnet. The parts have 0.27&#8211;0.36&#8202;M triangles; the v17 project was "
                "22.5&#8202;MB.", key="builds")
    cmd = ("<pre>~/miniconda3/envs/femtip/bin/python \\\n  scripts/eflesh_build.py --shape sr2tip \\\n  --cell 8 --E 0.0035 "
           "--magnet 4.76,1.59,c,c,5 \\\n  --base 1.6 --name sr2tip_c8_E0035_open</pre>")
    files = ("<p>Projects: <code>docs/experiments/20261013-eflesh_printability/build/20261013-sr2tip_c8_E0035_open.3mf</code>, "
             "<code>..._topskin.3mf</code>, <code>..._gapwalls.3mf</code> and <code>20261013-sr2tip_c8_E0035.3mf</code> (skin on every face), "
             "each with its report beside "
             "it. They print on tool 2, as v17 did; PrusaSlicer stops before the layer at 12.2&#8202;mm for the magnet.</p>")
    return txt + tab + cmd + files


RELATED = [
    ("eFlesh itself", "The last commit to github.com/notvenky/eFlesh is the merge of 28 October 2025; arXiv 2506.09994 is still version 1, three "
     "outside pull requests were closed unmerged and issue #23 (cross-talk between two facing eFlesh jaws, March 2026) has no reply. Users report "
     "drift (TacO, arXiv 2605.21976) and lattice delamination (Visible Touch, arXiv 2609.14156).", "https://github.com/notvenky/eFlesh"),
    ("Arslan Waltersson and Karayiannidis, Chalmers and Lund (Sweden), IEEE RA-L 2026",
     "A gyroid TPU 85A pad printed on a Prusa XL with twelve 2 &#215; 1&#8202;mm magnets glued in at a print pause over 13 MLX90393 chips; the "
     "central chip has no magnet and removes the Earth&#8217;s field, which otherwise caused force errors up to 0.7&#8202;N. Six-axis "
     "force and torque error 2&#8211;3&#8202;% of range. STL, PCB and firmware are released.", "https://arxiv.org/abs/2606.11952"),
    ("Jain, Parikh and Jain, Preprints.org, September 2026 (abstract only)",
     "One magnet over one MLX90393: 0.135&#8202;&#181;T/g in cast silicone, 0.084 in 10&#8202;% honeycomb and 0.063 in 10&#8202;% gyroid TPU; "
     "an outer wall almost removed a 5&#8202;% gyroid&#8217;s response.", "https://doi.org/10.20944/preprints202609.2559.v1"),
    ("Kim et al., ETH Zurich (Switzerland), arXiv 2605.01434, May 2026",
     "ORCA-hand fingertips with a printed TPU skin, one 8 &#215; 1&#8202;mm magnet and four single-axis Hall sensors read at 1&#8202;kHz; "
     "0.125&#8202;N force error up to 4&#8202;N.", "https://arxiv.org/abs/2605.01434"),
    ("Yoshimura, Kawaharazuka and Okada, University of Tokyo (Japan), IROS 2025",
     "M3D-skin: slicer infill of TPU 95A between conductive TPU layers, printed on a Prusa XL tool-changer, read resistively. The same "
     "lab (Hiraoka et al., IROS 2024) built a magnetic sensor on frame structures for a humanoid; its full text was not read.",
     "https://arxiv.org/abs/2510.12419"),
    ("Gro&#223; et al., TU Munich (Germany), IEEE Sensors Journal 2025",
     "Nine tactile elements from commercial parts compared; the magnetic ones did best in every loading case. No German printed-lattice "
     "magnetic sensor from 2025&#8211;2026 was found.", "https://doi.org/10.1109/JSEN.2025.3606565"),
    ("Aguilar-Segovia, Grzeskowiak, Lefebvre, Babel and Gu&#233;gan, Inria and INSA Rennes (France), IEEE Sensors Journal 2025",
     "Foam-like capacitive sensors from IceSL&#8217;s gyroid, phasor and polyfoam infill in TPU 60A, responsive from 90&#8202;Pa to "
     "170&#8202;kPa; the authors print continuous foams because a truss lattice prints each strut as its own island.",
     "https://hal.science/hal-04891580/document"),
    ("Tricard et al., Inria Nancy (France), ACM Transactions on Graphics 2020",
     "Phasor-noise microstructures generated as toolpaths without a mesh, with orientation and scale as fields; NinjaFlex 85A cubes "
     "measured 0.08&#8211;2.78&#8202;MPa. Available as an infill in IceSL.", "https://hal.inria.fr/hal-02524371/file/main.pdf"),
    ("Wade, Beck and MacCurdy, CU Boulder (USA), Additive Manufacturing 2025",
     "OpenVCAD slices implicit geometry and material fields straight to toolpaths on Prusa XL and MK4S printers, including varioShore, "
     "a foaming TPU whose stiffness follows the nozzle temperature.", "https://arxiv.org/abs/2505.08093"),
    ("OrcaSlicer 2.3.1 and 2.4.0",
     "TPMS-D and TPMS-FK infill, a gyroid whose vertical wavelength shortens with density (which changes a gyroid tip&#8217;s stiffness at "
     "the same percentage), and adaptive TPMS density, merged 9 October 2026.", "https://github.com/OrcaSlicer/OrcaSlicer/pull/16005"),
    ("Bang et al., DGIST (Korea), ICRA 2026",
     "An SLS-printed TPU 90A lattice skin with magnets over 16 MLX90393 chips, 1.13&#8202;mm mean localization error; the authors call FDM "
     "printing of the lattice nearly impossible.", "https://arxiv.org/abs/2605.28352"),
    ("de la Rosa, Mayuet Ares and Rodr&#237;guez-Parada, C&#225;diz (Spain), Polymers 2025",
     "Open-cell TPU lattices string and scatter more as cells shrink, with stringing growing with the retractions; closed cells print cleanly.",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC12073879/"),
]


def related():
    return ("<p>A literature search of 13 October 2026 for printed, lattice and foam magnetic touch sensors since eFlesh. Entries name the "
            "group and country, then what bears on the SR2 tip.</p><ul class='refs'>"
            + "".join(f"<li><b>{a}.</b> {b} <a href=\"{u}\">{u.split('//')[1][:60]}</a></li>" for a, b, u in RELATED) + "</ul>")


def limitations(X=None):
    t = (X.fem.get("eflesh_open") or X.fem.get("eflesh_top")) if X else None
    big = ""
    if t and "3" in t["per_force"]:
        big = (f"The model is linear. The open eFlesh tip approaches {t['per_force']['1']['delta_um'] / 1e3:.2f}&#8202;mm at 1&#8202;N and "
               f"{t['per_force']['3']['delta_um'] / 1e3:.1f}&#8202;mm at 3&#8202;N, where its struts bend beyond small strain and can buckle; on the "
               f"fingertip page the 5&#8202;% gyroid tip lost 11&#8202;% of its force to large deformation at 0.35&#8202;mm.")
    items = ([big] if big else []) + [
        "The solver joins every bead to its neighbours as one solid; the weaker bond between layers and any delamination are absent from the "
        "stiffness, which therefore bounds a printed tip&#8217;s stiffness from above.",
        "The magnet is TPU bonded into its pocket in the as-printed runs; the earlier runs took it as NdFeB. Only normal load is solved.",
        "Support is geometric: cooling, stringing and the 2.5&#8202;mm retractions are not in it.",
        "The boolean cut keeps whole cells cut by the surface; eFlesh&#8217;s cut-cell method re-optimizes the boundary cells, which are "
        "softer here.",
        "One printer profile: the owner&#8217;s v17 project (Prusa XL, Generic FLEX, 0.2&#8202;mm layers); the 0.25&#8202;mm nozzle was checked "
        "in the slicer only.",
    ]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def open_items():
    items = [
        ("Bench check of the as-printed model.", "Print <code>build/20261013-sr2tip_c8_E0035_open.3mf</code> and the 20&#8202;% gyroid tip, "
         "indent both with the shaft and fit them with <code>scripts/tpu_tip_calibrate.py fit --indent</code>; the as-printed model holds if the "
         "measured approach at 1&#8202;N lies within its approach and twice it (the bond between layers can only soften the print)."),
        ("Coupons for eFlesh&#8217;s modulus parameter.", "Build 24&#8202;mm cubes at \\(E\\) = 0.002, 0.0035 and 0.01 with "
         "<code>scripts/eflesh_build.py --shape cube24.stl --cell 8 --skin 0 --base 0</code>, print three of each and press them with a "
         "plate; the parameter is usable if the stiffness ratio between 0.002 and 0.01 exceeds the scatter between copies."),
        ("Continuous-path soft structures.", "Slice the SR2 tip with IceSL&#8217;s phasor infill and with OrcaSlicer&#8217;s TPMS-D at "
         "10&#8202;% and audit them with <code>scripts/eflesh_printability.py</code>; a structure qualifies with at most 5 islands per layer "
         "and a pad modulus under 2&#8202;MPa in <code>scripts/eflesh_tip_fem.py</code>."),
        ("A wall clear of the lattice.", "Build the SR2 tip with walls joined to the base only, 0.5&#8202;mm clear of every strut "
         "(<code>scripts/eflesh_build.py</code> with the walls subtracted from the lattice&#8217;s offset), and solve it with "
         "<code>scripts/eflesh_tip_fem.py</code>; closed sides are free if it keeps the palmar-skin tip&#8217;s magnet travel within 10&#8202;%."),
        ("Shear in the as-printed model.", "Add a tangential load to <code>scripts/eflesh_tip_fem.py</code> and report the noise across and "
         "along the finger, which the open sides of the lattice should lower most."),
        ("eFlesh&#8217;s cut-cell tool.", "Rebuild <code>cut_cells_cli</code> against one OpenVDB and oneTBB; it works if it returns the SR2 tip "
         "at 8&#8202;mm cells, which then replaces the boolean cut in <code>scripts/eflesh_build.py</code>."),
    ]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def lede(X: Data):
    A = X.audit
    c8, c6, c4 = A["cube24_c8"], A["cube24_c6"], A["cube24_c4"]
    t = X.builds.get("sr2tip_c8_E0035_topskin")
    F = X.fem
    s = (f"On the Prusa XL&#8217;s 0.4&#8202;mm nozzle, eFlesh&#8217;s lattice prints soundly at 8&#8202;mm cells and fails below: "
         f"{pc(c8['islands_under_half_supported'])}, {pc(c6['islands_under_half_supported'])} and {pc(c4['islands_under_half_supported'])} of "
         f"each layer&#8217;s islands rest less than half on the layer below at 8, 6 and 4&#8202;mm cells. eFlesh&#8217;s cut-cell tool crashes "
         f"on every input, so its notebook tiles whole cells and returns boxes; a one-command build now cuts the lattice to any closed shape, adds "
         f"the magnet pouches and writes a PrusaSlicer project with the pause, in {t['seconds']:.0f}&#8202;s for the SR2 tip.")
    if F.get("eflesh_open") and F.get("gyroid20") and F.get("eflesh_all"):
        e, g, a = F["eflesh_open"], F["gyroid20"], F["eflesh_all"]
        s += (f" Solved as printed, the open eFlesh tip moves its magnet {-e['per_force']['1']['magnet_t_um'][0] / -g['per_force']['1']['magnet_t_um'][0]:.0f} "
              f"times as far per newton as the 20&#8202;% gyroid tip, for {e['field']['1']['sigF_normal_N']:.3f}&#8202;N of force noise against "
              f"{g['field']['1']['sigF_normal_N']:.3f}&#8202;N; a skin over its sides removes the difference ({a['field']['1']['sigF_normal_N']:.3f}&#8202;N).")
    return s


def byline():
    return ("Audit <code>scripts/gcode_audit.py</code>, <code>scripts/eflesh_printability.py</code> &#183; build <code>scripts/eflesh_build.py</code> "
            "&#183; as-printed stiffness <code>scripts/eflesh_tip_fem.py</code> &#183; builder <code>scripts/eflesh_printability_page.py</code> "
            "&#183; rows <code>docs/experiments/20261013-eflesh_printability/</code> &#183; owner&#8217;s eFlesh files "
            "<code>~/wxie_workspace/eFlesh/microstructure</code>, <code>~/wxie_workspace/fingers</code>")


def render_tex(t):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None) for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    return P.TEX_RE.sub(lambda m: next(svgs), t), len(items)


def main():
    X = Data()
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "GLOSSARY": glossary()}
    v["LEDE"] = lede(X)
    v["TOOLS"] = tools_section(X)
    v["STRUTS"] = struts_section(X)
    v["LAYERS"] = layers_section(X)
    v["FEM"] = fem_section(X)
    v["BUILD"] = build_section(X)
    v["RELATED"] = related()
    v["LIMITS"] = limitations(X)
    v["OPEN"] = open_items()
    v["FOOTER"] = ("<p>Rebuild: <code>.venv/bin/python scripts/eflesh_printability_page.py</code>. G-code and voxels are regenerated by the "
                   "scripts above into <code>logs/20261013-eflesh_*</code>.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    miss = sorted(set(re.findall(r"\[\[((?:fig|tab):[a-z_]+)\]\]", t)) - set(REFS))
    if miss:
        raise SystemExit(f"unresolved references: {miss}")
    t = re.sub(r"\[\[((?:fig|tab):[a-z_]+)\]\]", lambda m: str(REFS[m.group(1)]), t)
    t, n = render_tex(t)
    t = retro_style.apply(t)
    open(OUT, "w").write(t)
    print(f"formulas {n}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
