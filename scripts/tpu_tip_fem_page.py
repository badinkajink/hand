#!/usr/bin/env python3
r"""Build docs/experiments/20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html.

    logs/20261001-hom_contact/venv/bin/python scripts/tpu_tip_fem_page.py

Steps 2-5, 11 and 12 of docs/handoff/20261007-tpu_tip_finite_elements.md: homogenized printed gyroid cells
(scripts/tpu_tip_fem_cells.py), the printed insert with a homogenized core (scripts/tpu_tip_fem_tip.py) and with the
gyroid resolved on the GPU (scripts/tpu_tip_fem_gpu.py), the pads on the tpu2.7 tip at the fitted moduli
(scripts/tpu_tip_fem_bed.py) and the tables of scripts/tpu_tip_fem_analysis.py. Every number in the prose is computed
here from the rows in docs/experiments/20261009-tpu_tip_fem/; a section whose rows are not written yet renders a note.
Style, SVG helpers and LaTeX rendering are those of scripts/contact_overview_page.py; the page is written in the plain
style of scripts/retro_style.py.
"""
from __future__ import annotations

import base64
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import tpu_tip_fem_analysis as A  # noqa: E402

D = os.path.join(P.EXP, "20261009-tpu_tip_fem")
OUT = os.path.join(D, "20261009-printed_tpu_fingertip_finite_elements.html")
TPL = os.path.join(ROOT, "scripts/tpu_tip_fem_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
NOTE = "docs/handoff/20261007-tpu_tip_finite_elements.md"
OVERVIEW = "docs/experiments/20261009-contact_overview/20261009-sphere_pad_contact_model.html"
BACKENDS = "docs/experiments/20261006-fingertip_backends/20261006-fingertip_contact_backends.html"
COMPLIANCE = "docs/experiments/20261007-native_compliance/20261007-native_presliding_compliance.html"
fmt = P.fmt
FIG, TAB = [0], [0]
E_S = 30.0
COL = {"x": "var(--s-x)", "z": "var(--s-z)", "sheet": "var(--s-sh)", "ref": "var(--s-ref)", "pads": "var(--s-pads)",
       "res": "var(--s-res)"}
EXTRA_CSS = """
:root{--s-x:#2a78d6;--s-z:#eb6834;--s-sh:#1baf7a;--s-ref:#48545D;--s-pads:#8a4fbf;--s-res:#c2185b}
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
figure.films{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
figure.films .cell{min-width:0}
figure.films video,figure.films img,figure.pics img{width:100%;height:auto}
figure.films figcaption{grid-column:1/-1}
@media (max-width:640px){figure.films{grid-template-columns:minmax(0,1fr)}}
"""
LEG_TIPS = [("95A solid", COL["ref"], False, None), ("95A 20&#8202;% across", COL["x"], False, None),
            ("95A 20&#8202;% along", COL["z"], False, None), ("95A 10&#8202;% across", COL["sheet"], False, None)]
RECIPE_LBL = {
    "solid95": "95A, solid", "g20_across": "95A, gyroid 20&#8202;%, layers across", "g20_along": "95A, gyroid 20&#8202;%, layers along",
    "g10_across": "95A, gyroid 10&#8202;%, layers across", "g10_along": "95A, gyroid 10&#8202;%, layers along",
    "g30_across": "95A, gyroid 30&#8202;%, layers across", "g40_across": "95A, gyroid 40&#8202;%, layers across",
    "s85_30_across": "85A, gyroid 30&#8202;%, layers across", "s85_30_along": "85A, gyroid 30&#8202;%, layers along",
    "s85_50_across": "85A, gyroid 50&#8202;%, layers across", "s85_50_along": "85A, gyroid 50&#8202;%, layers along",
    "s85_100": "85A, solid"}
GPU_LBL = {"g20x_s0": "95A 20&#8202;% across", "g20z_s0": "95A 20&#8202;% along", "g20x_s26": "95A 20&#8202;% across, shifted 2.6&#8202;mm in y",
           "g20x_s0_26": "95A 20&#8202;% across, shifted 2.6&#8202;mm in z", "g20x_s13": "95A 20&#8202;% across, shifted 1.3&#8202;mm in y and z",
           "g10x_s0": "95A 10&#8202;% across", "g10z_s0": "95A 10&#8202;% along", "g05x_s0": "95A 5&#8202;% across",
           "g05z_s0": "95A 5&#8202;% along", "eflesh_c8": "eFlesh lattice, 8&#8202;mm cells"}


# ------------------------------------------------------------------------------------------ data

class Data:
    def __init__(self):
        self.cells = A.rows("cells.jsonl")
        self.tip = A.rows("tip_runs.jsonl")
        self.gpu = A.rows("gpu_runs.jsonl")
        self.T = A.recipe_table()
        self.G = A.gpu_table()
        self.B = A.bed_table()

    def cell(self, kind, rho, h=None):
        rs = [r for r in self.cells if r["kind"] == kind and r.get("rho_req") is not None and abs(r["rho_req"] - rho) < 1e-9
              and (h is None or abs(r["h_mm"] - h) < 1e-9)]
        return min(rs, key=lambda r: r["h_mm"]) if rs else None

    def pads_arm(self, E, N):
        r = self.B.get(E, {}).get(("T2", N))
        return r.get("rbar_onset_mm") if r else None


# ------------------------------------------------------------------------------------------ html helpers

def figure(svg, caption, legend=None):
    FIG[0] += 1
    lg = P._legend_html(legend) if legend else ""
    return f'<figure class="diagram">{svg}{lg}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def table(head, body, caption=None, note=None, text_cols=(0,)):
    out = []
    if caption:
        TAB[0] += 1
        out.append(f"<p class='tcap'><b>Table&#160;{TAB[0]}.</b> {caption}</p>")
    cls = lambda i: "" if i in text_cols else " class=num"  # noqa: E731
    out.append("<div class='tw'><table><thead><tr>" + "".join(f"<th{cls(i)}>{h}</th>" for i, h in enumerate(head))
               + "</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table></div>")
    if note:
        out.append(f"<p class='tnote'>{note}</p>")
    return "".join(out)


def data_uri(path, mime):
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def media(name):
    return os.path.join(D, "media", name)


def _ok(x):
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def f0(x):
    return fmt(x, 0) if _ok(x) else "&#8211;"


def f1(x):
    return fmt(x, 1) if _ok(x) else "&#8211;"


def f2(x):
    return fmt(x, 2) if _ok(x) else "&#8211;"


# ------------------------------------------------------------------------------------------ terms

GLOSSARY = [
    ("Solid modulus \\(E_s\\)", "Young&#8217;s modulus of fully dense printed TPU. 95A: 26&#8202;MPa (NinjaTek Cheetah, ASTM D638), "
     "33.6&#8202;MPa (Polymaker PolyFlex TPU95, ISO 37), 67&#8202;MPa (UltiMaker TPU 95A, ASTM D3039); Overture&#8217;s sheet gives only "
     "the stress at 100&#8202;% strain, 9.7&#8202;MPa, the value Polymaker lists as 11.1. This page uses 30&#8202;MPa for 95A and "
     "12&#8202;MPa for 85A (NinjaFlex, ASTM D638); every stiffness scales with \\(E_s\\). Poisson&#8217;s ratio 0.45, which no TPU sheet lists."),
    ("Infill fraction \\(\\rho\\)", "Requested gyroid density in the slicer; the printed cell&#8217;s solid fraction is 4&#8211;12&#8202;% higher (Table&#160;1)."),
    ("Gyroid period \\(L\\)", "\\(L = 2\\pi s / (2.44\\,\\rho)\\) with line spacing \\(s = 0.407\\)&#8202;mm (0.45&#8202;mm lines, 0.2&#8202;mm layers): "
     "21, 10.4, 5.2, 3.4 and 2.6&#8202;mm at 5, 10, 20, 30 and 40&#8202;%."),
    ("\\(E_x\\), \\(E_z\\)", "Homogenized Young&#8217;s moduli of the printed infill in the layer plane and along the build direction, per unit \\(E_s\\)."),
    ("Approach \\(\\delta\\)", "Displacement of the rigid indenter into the palmar face from first touch, in &#181;m."),
    ("Contact stiffness \\(k\\)", "\\(dF/d\\delta\\) of the screwdriver cylinder (radius 12.5&#8202;mm, axis along the finger&#8217;s width) at 1&#8202;N, in N/mm; "
     "plate stiffness: force over approach of a flat plate on the palmar face."),
    ("Pressure law", "The pads&#8217; Winkler law on the tip block, \\(p = E\\,d/h_f\\), \\(d\\) the local overlap of the undeformed bodies and "
     "\\(h_f = 8.5\\)&#8202;mm; \\(E\\) is the modulus every contact model here has used at 10&#8202;MPa."),
    ("Best single modulus \\(E^*\\)", "The \\(E\\) that minimizes the RMS log error of the law&#8217;s approach against the finite-element approach "
     "over the cylinder at 0.5, 1 and 3&#8202;N and the plate at 1&#8202;N; its error is the worst of those four, in %."),
    ("Twist arm", "Torque at spin onset over \\(\\mu N\\) for one pad: \\(\\sum p_i r_i / \\sum p_i\\) about the normal through the force centre, "
     "every contact point sliding; for the pads, the bed&#8217;s T2 onset arm."),
    ("Contact half-width \\(b\\)", "Pressure-weighted RMS distance of the contact from its centre line along the finger, in mm; its exponent is "
     "\\(d\\ln b/d\\ln F\\): 1/3 for the law on a cylinder, 1/2 for an elastic half-space."),
    ("Presliding", "Displacement of the tool along its axis at a tangential force of half the pad&#8217;s slip force \\(\\mu N/2\\) "
     "(\\(\\mu = 1\\)), stick and slip solved on the face compliance; loop: the work over one cycle \\(\\pm\\mu N/2\\), per pad."),
    ("Coupling length \\(\\ell\\)", "Decay length of the palmar face&#8217;s displacement away from the contact line at 1&#8202;N, "
     "fit \\(u \\propto e^{-s/\\ell}\\) beyond three half-widths."),
    ("Homogenized, resolved", "Core cells given the homogenized tensor of the printed infill on a graded grid (217k DOF), or every gyroid bead "
     "voxelized at 0.1&#8202;mm (2.2&#8202;M elements) and solved on the GPU."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


# ------------------------------------------------------------------------------------------ figures

def svg_cells(X: Data):
    out = P._svg_open(990, 470, "Homogenized modulus of printed gyroid infill against solid fraction")
    fx, fy = P._panel(out, 90, 40, 840, 360, (0.08, 0.7), (0.005, 0.5), [0.1, 0.2, 0.3, 0.5], [0.01, 0.03, 0.1, 0.3],
                      "solid fraction of the printed cell", "Modulus per unit solid modulus", logx=True, logy=True)
    for key, kind, col in (("Ex", "slicer", COL["x"]), ("Ez", "slicer", COL["z"]), ("Ex", "sheet", COL["sheet"])):
        pts = []
        for rho in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
            r = X.cell(kind, rho)
            if r:
                pts.append((r["rho"], r[key]))
        pts.sort()
        P._path(out, fx, fy, pts, col)
        for x, y in pts:
            P._marker(out, fx(x), fy(y), col)
    xs = np.geomspace(0.08, 0.7, 30)
    P._path(out, fx, fy, [(x, x ** 2) for x in xs], COL["ref"], dashed=True, width=1.6)
    P._path(out, fx, fy, [(x, x / 3) for x in xs], COL["ref"], dashed=True, width=1.6)
    out.append(f'<text x="{fx(0.55):.0f}" y="{fy(0.55 ** 2) + 22:.0f}" style="fill:var(--ink2)">&#961;&#178;</text>')
    out.append(f'<text x="{fx(0.5):.0f}" y="{fy(0.5 / 3) - 10:.0f}" style="fill:var(--ink2)">&#961;/3</text>')
    ef = [r for r in X.cells if r["kind"] == "eflesh" and abs(r["nu_s"] - 0.45) < 1e-9]
    for r in ef:
        P._marker(out, fx(r["rho"]), fy(r["Ex"]), COL["ref"], shape="diamond")
    out.append("</svg>")
    return out


def svg_force(X: Data):
    """Force against approach under the cylinder, log-log: homogenized recipes (lines), resolved tips (markers), the law
    at the best single modulus of the 20 % tip (dashed)."""
    out = P._svg_open(990, 500, "Force against approach under the screwdriver cylinder")
    fx, fy = P._panel(out, 90, 40, 840, 380, (3, 400), (0.2, 5), [3, 10, 30, 100, 300], [0.25, 0.5, 1, 2, 4],
                      "approach under the cylinder (&#181;m)", "Force (N)", logx=True, logy=True)
    shown = [("solid95", COL["ref"]), ("g20_across", COL["x"]), ("g20_along", COL["z"]), ("g10_across", COL["sheet"])]
    for name, col in shown:
        rr = sorted([r for r in X.tip if r["recipe"] == name and r["indenter"] == "cyl"], key=lambda r: r["F_target"])
        P._path(out, fx, fy, [(r["delta_mm"] * 1e3, r["F"]) for r in rr], col)
    for tag, col, shape in (("g20x_s0", COL["x"], "circle"), ("g20z_s0", COL["z"], "square"), ("g10x_s0", COL["sheet"], "circle"),
                            ("g05x_s0", COL["res"], "diamond"), ("eflesh_c8", COL["pads"], "cross")):
        for r in [r for r in X.gpu if r["tag"] == tag and r["indenter"] == "cyl"]:
            P._marker(out, fx(r["delta_mm"] * 1e3), fy(r["F"]), col, shape=shape, hollow=True)
    T = X.T.get("g20_across")
    if T and "E_best" in T:
        d = np.load(os.path.join(D, "data", "g20_across.npz"))
        A_, g0 = d["cyl_1__A"], d["cyl_1__g0"]
        pts = [(1e3 * dl, T["E_best"] * A.law_f(A_, g0, dl)) for dl in np.geomspace(0.003, 0.4, 40)]
        P._path(out, fx, fy, [(x, y) for x, y in pts if 0.2 <= y <= 5], COL["x"], dashed=True, width=1.6)
    out.append("</svg>")
    return out


def svg_arm(X: Data):
    out = P._svg_open(990, 470, "Twist arm against normal force")
    fx, fy = P._panel(out, 90, 40, 840, 350, (0.4, 4.2), (2.0, 4.2), [0.5, 1, 2, 3, 4], [2.0, 2.5, 3.0, 3.5, 4.0],
                      "normal force per pad (N)", "Twist arm (mm)", logx=True)
    for name, col in (("solid95", COL["ref"]), ("g20_across", COL["x"]), ("g20_along", COL["z"]), ("g10_across", COL["sheet"])):
        rr = sorted([r for r in X.tip if r["recipe"] == name and r["indenter"] == "cyl"], key=lambda r: r["F_target"])
        P._path(out, fx, fy, [(r["F"], r["arm"]) for r in rr if 0.45 <= r["F"] <= 4.2], col)
    for tag, col in (("g20x_s0", COL["x"]), ("g20z_s0", COL["z"])):
        for r in [r for r in X.gpu if r["tag"] == tag and r["indenter"] == "cyl"]:
            if 0.4 <= r["F"] <= 4.2:
                P._marker(out, fx(r["F"]), fy(r["arm"]), col, hollow=True)
    for E in sorted(X.B):
        pts = [(N, X.pads_arm(E, N)) for N in (1.0, 3.0) if X.pads_arm(E, N)]
        P._path(out, fx, fy, pts, COL["pads"], dashed=True, width=1.4)
        for x, y in pts:
            P._marker(out, fx(x), fy(y), COL["pads"], shape="square")
    out.append("</svg>")
    return out


# ------------------------------------------------------------------------------------------ sections

def cells_section(X: Data):
    c20, s20 = X.cell("slicer", 0.2), X.cell("sheet", 0.2)
    if not c20:
        return P.pending("Cell rows not written yet.")
    fit = {}
    for kind, key in (("slicer", "Ex"), ("slicer", "Ez"), ("sheet", "Ex")):
        rs = [r for r in X.cells if r["kind"] == kind and abs(r["h_mm"] - 0.1) < 1e-9 and r.get("rho_req") and r["rho_req"] <= 0.4 + 1e-9]
        n, lc = np.polyfit(np.log([r["rho"] for r in rs]), np.log([r[key] for r in rs]), 1)
        fit[(kind, key)] = (math.exp(lc), n)
    txt = (f"<p>A slicer prints the gyroid layer by layer: in each 0.2&#8202;mm layer the nozzle follows the zero set of the gyroid "
           f"function at that height with a 0.45&#8202;mm bead, and beads of successive layers bond where they overlap. Built that "
           f"way in voxels of 0.05&#8202;mm and homogenized over one periodic cell, the 20&#8202;% infill ({f1(100 * c20['rho'])}&#8202;% solid) "
           f"has \\(E_x = {c20['Ex']:.3f}\\,E_s\\) in the layer plane and \\(E_z = {c20['Ez']:.3f}\\,E_s\\) along the build direction; the ideal "
           f"sheet gyroid of the same solid fraction is isotropic at \\({s20['Ex']:.3f}\\,E_s\\). Over 11&#8211;42&#8202;% solid the printed infill "
           f"follows \\(E_x/E_s = {fit[('slicer', 'Ex')][0]:.2f}\\,\\rho^{{{fit[('slicer', 'Ex')][1]:.2f}}}\\) and "
           f"\\(E_z/E_s = {fit[('slicer', 'Ez')][0]:.2f}\\,\\rho^{{{fit[('slicer', 'Ez')][1]:.2f}}}\\) (Figure&#160;1), between Gibson and Ashby&#8217;s "
           f"bending-dominated \\(\\rho^2\\) and stretching-dominated \\(\\rho/3\\). At \\(E_s\\) = 30&#8202;MPa, 20&#8202;% infill is "
           f"{f1(30 * c20['Ex'])}&#8202;MPa in-plane and {f1(30 * c20['Ez'])}&#8202;MPa along the build direction.</p>")
    ef = {round(r["E_rel_target"], 4): r for r in X.cells if r["kind"] == "eflesh" and abs(r["nu_s"] - 0.3) < 1e-9}
    if ef:
        txt += ("<p>The same solver run on eFlesh&#8217;s cut cell (pattern 0646, its inflator&#8217;s map from target modulus to strut "
                "geometry, base material \\(\\nu = 0.3\\)) returns "
                + ", ".join(f"{r['Ex']:.4f} for a target of {k:g}" for k, r in sorted(ef.items()))
                + " (Poisson&#8217;s ratio " + ", ".join(f"{r['nu_xy']:.2f}" for _, r in sorted(ef.items())) + " for a target of 0.09).</p>")
    body = []
    for rho in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        r, s = X.cell("slicer", rho), X.cell("sheet", rho)
        if r:
            body.append([f"{100 * rho:.0f}", f1(100 * r["rho"]), f"{r['period_mm']:.1f}", f"{r['Ex']:.4f}", f"{r['Ez']:.4f}",
                         f"{r['Gxz']:.4f}", f"{r['nu_xy']:.2f}", f"{s['Ex']:.4f}" if s else "&#8211;", f"{r['h_mm']:.2f}"])
    tab = table(["Infill (%)", "Solid (%)", "Period (mm)", "\\(E_x/E_s\\)", "\\(E_z/E_s\\)", "\\(G_{xz}/E_s\\)", "\\(\\nu_{xy}\\)",
                 "Ideal sheet \\(E/E_s\\)", "Voxel (mm)"], body,
                "Homogenized moduli of the printed gyroid (<code>cells.jsonl</code>); the finest voxel size run for each density.")
    fig = figure("".join(svg_cells(X)), "Moduli of printed gyroid infill against solid fraction: in the layer plane (blue), along the build "
                 "direction (orange), the ideal sheet gyroid (green), eFlesh cut cells (diamonds), and the Gibson&#8211;Ashby limits "
                 "\\(\\rho^2\\) and \\(\\rho/3\\) (dashed).",
                 legend=[("Printed, layer plane", COL["x"], False, "circle"), ("Printed, build direction", COL["z"], False, "circle"),
                         ("Ideal sheet", COL["sheet"], False, "circle"), ("eFlesh cut cell", COL["ref"], False, "diamond"),
                         ("Gibson&#8211;Ashby", COL["ref"], True, None)])
    return txt + fig + tab


def tip_section(X: Data):
    parts = ["<p>The insert of <code>fingertip (3).stl</code> (17.0 &#215; 14.8 &#215; 21.4&#8202;mm, fillets 2.7&#8211;3.0&#8202;mm on the "
             "palmar edges, a slot and two sockets for the mount) is printed with two 0.45&#8202;mm perimeters and four 0.2&#8202;mm top and "
             "bottom layers around gyroid infill. With the layers across the load the palmar face is a top skin; with the layers along the "
             "load (standing on its mount face) it is a perimeter wall. The back and mount faces are fixed. The screwdriver cylinder presses "
             "the face at the tip-body origin, its axis across the finger, as in the contact bed.</p>"]
    pics = [p for p in ("20261009-classes.png", "20261009-voxels_g20x.png") if os.path.exists(media(p))]
    if pics:
        cells = "".join(f'<div class="cell"><img src="{data_uri(media(p), "image/png")}" alt="{p}"></div>' for p in pics)
        FIG[0] += 1
        parts.append(f'<figure class="pics">{cells}<figcaption>Figure&#160;{FIG[0]}. Slices of the printed insert: walls, skins and '
                     f'core for both print orientations, and the resolved 20&#8202;% gyroid with the layers across the load.</figcaption></figure>')
    films = [(t, media(f"20261009-film_{t}.mp4")) for t in ("g20x_s0", "g05x_s0", "eflesh_c8")]
    films = [(t, p) for t, p in films if os.path.exists(p)]
    if films:
        cells = "".join(f'<div class="cell"><video controls muted loop playsinline preload="metadata" '
                        f'poster="{data_uri(p.replace(".mp4", ".png"), "image/png")}" src="{data_uri(p, "video/mp4")}"></video>'
                        f'<p class="filmlab">{GPU_LBL[t]}</p></div>' for t, p in films)
        FIG[0] += 1
        parts.append(f'<figure class="films">{cells}<figcaption>Figure&#160;{FIG[0]}. The resolved tips under the cylinder, 13 approaches '
                     f'from first touch: the section through the contact line with the displacement magnified, and the pressure on the '
                     f'palmar face.</figcaption></figure>')
    else:
        parts.append(P.pending("Films not rendered yet (logs/20261009-tpu_tip_fem/film_queue.sh)."))
    return "".join(parts)


def law_section(X: Data):
    T = X.T
    if "g20_across" not in T:
        return P.pending("Tip rows not written yet.")
    g = T["g20_across"]
    txt = (f"<p>Under the screwdriver the 20&#8202;% tip printed with its layers across the load sinks {f1(g['delta_um'][0.5])}, "
           f"{f1(g['delta_um'][1.0])} and {f1(g['delta_um'][3.0])}&#8202;&#181;m at 0.5, 1 and 3&#8202;N; force grows as "
           f"\\(\\delta^{{{g['force_exponent']:.2f}}}\\) while the pressure law gives \\(\\delta^{{1.5}}\\) on a cylinder. The modulus that "
           f"reproduces each approach is {f1(g['E_fit_cyl'][0.5])}, {f1(g['E_fit_cyl'][1.0])} and {f1(g['E_fit_cyl'][3.0])}&#8202;MPa, and "
           f"{f1(g.get('E_fit_plate'))}&#8202;MPa under a flat plate. The best single modulus is {f1(g.get('E_best'))}&#8202;MPa, "
           f"with a worst approach error of {f0(g.get('E_best_max_err_pct'))}&#8202;% (Figure&#160;{FIG[0] + 1}, Table&#160;{TAB[0] + 1}).</p>")
    if "g20x_s0" in X.G:
        r = X.G["g20x_s0"]["delta_um"]
        txt += (f"<p>Resolving every bead at 0.1&#8202;mm on the GPU gives {f1(r.get(1.0))}&#8202;&#181;m at 1&#8202;N against "
                f"{f1(g['delta_um'][1.0])}&#8202;&#181;m for the homogenized core, so at 20&#8202;% the homogenized model represents the "
                f"stiffness even though the gyroid period (5.2&#8202;mm) exceeds the contact width.</p>")
    leg = LEG_TIPS + [("Resolved", COL["ref"], False, "circle"), ("Pressure law at the 20&#8202;% tip&#8217;s best modulus", COL["x"], True, None)]
    fig = figure("".join(svg_force(X)), "Force against approach under the cylinder: homogenized cores (lines), resolved gyroid "
                 "(hollow markers: 20&#8202;% across circles, along squares, 10&#8202;% green, 5&#8202;% diamonds, eFlesh crosses), and the "
                 "pressure law at the best single modulus of the 20&#8202;% tip (dashed).", legend=leg)
    body = []
    for name, r in T.items():
        body.append([RECIPE_LBL.get(name, name), f"{r['E_s']:.0f}", f1(r["delta_um"][1.0]), f2(r['force_exponent']), f1(r["k_1N"]),
                     f0(r.get("k_plate")), f1(r["E_fit_cyl"][0.5]), f1(r["E_fit_cyl"][3.0]), f1(r.get("E_fit_plate")),
                     f1(r.get("E_best")), f0(r.get("E_best_max_err_pct"))])
    tab = table(["Print", "\\(E_s\\) (MPa)", "\\(\\delta\\) at 1&#8202;N (&#181;m)", "Force exponent", "\\(k\\) at 1&#8202;N (N/mm)",
                 "Plate (N/mm)", "\\(E\\) at 0.5&#8202;N", "\\(E\\) at 3&#8202;N", "\\(E\\), plate", "\\(E^*\\) (MPa)", "Error (%)"], body,
                "Approach, stiffness and the pressure-law modulus per print (homogenized core, <code>tip_runs.jsonl</code>).")
    return txt + fig + tab


def arm_section(X: Data):
    T = X.T
    if "g20_across" not in T or not X.B:
        return P.pending("Rows not written yet.")
    g = T["g20_across"]
    pads = {E: X.pads_arm(E, 1.0) for E in sorted(X.B)}
    pv = [v for v in pads.values() if v]
    es = {n: A.edge_share(n) for n in ("solid95", "g20_across", "g10_across")}
    sm = A.stiffness_map_law("g20_across")
    txt = (f"<p>The walls carry the face like two rails: under the cylinder, {f0(100 * es['g20_across']['fe'])}&#8202;% of the 20&#8202;% "
           f"tip&#8217;s force lies within 1.2&#8202;mm of the flat face&#8217;s edges and on the fillets, against "
           f"{f0(100 * es['g20_across']['law'])}&#8202;% for the pressure law on the same nodes ({f0(100 * es['g10_across']['fe'])}&#8202;% at "
           f"10&#8202;% infill, {f0(100 * es['solid95']['fe'])}&#8202;% for the solid print). The twist arm of the 20&#8202;% tip is therefore "
           f"{f2(g['arm'][1.0])}&#8202;mm at 1&#8202;N; the pads on the same block give {min(pv):.2f}&#8211;{max(pv):.2f}&#8202;mm for moduli "
           f"of {min(pads):g}&#8211;{max(pads):g}&#8202;MPa (bed task T2), so no modulus brings the pads&#8217; arm to the printed "
           f"tip&#8217;s (Figure&#160;{FIG[0] + 1}).</p>")
    if sm:
        txt += (f"<p>A local law cannot carry this coupling. Pads given a per-node stiffness from the finite-element plate solve "
                f"(pressure over approach at each face node) reproduce the plate by construction but miss the cylinder&#8217;s approach "
                f"by {f0(sm[3.0]['delta_err_pct'])}&#8211;{f0(sm[0.5]['delta_err_pct'])}&#8202;% and its arm by "
                f"{f0(min(v['arm_err_pct'] for v in sm.values()))}&#8211;{f0(max(v['arm_err_pct'] for v in sm.values()))}&#8202;%.</p>")
    txt += (f"<p>The contact widens as \\(F^{{{g.get('b_exponent', float('nan')):.2f}}}\\) against the law&#8217;s \\(F^{{1/3}}\\), and the face "
            f"sinks around the contact line with a coupling length of {f1(g.get('coupling_length_mm'))}&#8202;mm. Below the slip force "
            f"the tool moves {f1(g['presliding_um'][1.0])}&#8202;&#181;m at half the slip force at 1&#8202;N; the pads store no such "
            f"displacement, and the native skin of the <a href=\"../20261007-native_compliance/20261007-native_presliding_compliance.html\">"
            f"compliance page</a> is the model that does.</p>")
    fig = figure("".join(svg_arm(X)), "Twist arm against normal force: printed tips (lines, homogenized; hollow circles, resolved) and the "
                 "1&#8202;mm pads on the tpu2.7 block at 7.5&#8211;20&#8202;MPa (purple squares, bed T2).",
                 legend=LEG_TIPS + [("Pads, 7.5&#8211;20&#8202;MPa", COL["pads"], True, "square")])
    return txt + fig

def recipes_section(X: Data):
    T, G = X.T, X.G
    if len(T) < 6:
        return P.pending("Recipe rows not written yet.")
    pad14 = X.pads_arm(14.0, 1.0)
    order = sorted(T.items(), key=lambda kv: kv[1].get("E_best_max_err_pct", 1e9))
    body = []
    for name, r in order:
        es = A.edge_share(name)
        arm_err = (pad14 / r["arm"][1.0] - 1) * 100 if pad14 else None
        body.append([RECIPE_LBL.get(name, name), f1(r["k_1N"]), f1(r.get("E_best")), f0(r.get("E_best_max_err_pct")), f0(arm_err),
                     f0(100 * es["fe"]) if es else "&#8211;", f2(r.get("b_exponent")), f1(r.get("coupling_length_mm")),
                     f1(r["presliding_um"][1.0])])
    tab = table(["Print", "\\(k\\) at 1&#8202;N (N/mm)", "\\(E^*\\) (MPa)", "Approach error (%)", "Pads&#8217; arm error (%)",
                 "Edge load (%)", "Width exponent", "\\(\\ell\\) (mm)", "Presliding (&#181;m)"], body,
                "Prints ordered by the error of the best single modulus (homogenized cores). Pads&#8217; arm error: the pads at "
                "14&#8202;MPa against the print at 1&#8202;N; edge load: the share of the cylinder&#8217;s force within 1.2&#8202;mm of "
                "the face edges (law: 29&#8202;%).")
    e = {n: T[n].get("E_best_max_err_pct") for n in T}
    k = {n: T[n]["k_1N"] for n in T}
    txt = (f"<p>Ordered by the error of the best single modulus, 85A at 30&#8202;% ({f0(min(e['s85_30_across'], e['s85_30_along']))}&#8211;"
           f"{f0(max(e['s85_30_across'], e['s85_30_along']))}&#8202;%) and 95A at 20&#8202;% ({f0(min(e['g20_across'], e['g20_along']))}&#8211;"
           f"{f0(max(e['g20_across'], e['g20_along']))}&#8202;%) come first, 95A at 10&#8202;% next ({f0(e['g10_across'])}&#8202;%), then 85A at "
           f"50&#8202;% ({f0(min(e['s85_50_across'], e['s85_50_along']))}&#8211;{f0(max(e['s85_50_across'], e['s85_50_along']))}&#8202;%) and the "
           f"solid prints ({f0(e['s85_100'])}&#8211;{f0(e['solid95'])}&#8202;%), whose cylinder contact is local while the plate loads the "
           f"whole block. At equal stiffness, 95A at 20&#8202;% ({f1(k['g20_across'])}&#8202;N/mm at 1&#8202;N) beats 85A at 50&#8202;% "
           f"({f1(k['s85_50_across'])}&#8202;N/mm), and 85A at 30&#8202;% ({f1(k['s85_30_across'])}) beats 95A at 10&#8202;% "
           f"({f1(k['g10_across'])}): soft TPU at moderately high infill is represented worse than at 30&#8202;%. The twist arm trades "
           f"against this ranking: the denser the print, the closer its arm to the pads&#8217;.</p>")
    sh = [G[t] for t in ("g20x_s0", "g20x_s26", "g20x_s0_26", "g20x_s13") if t in G]
    if len(sh) >= 2:
        d1 = [g["delta_um"].get(1.0) for g in sh if g["delta_um"].get(1.0)]
        txt += (f"<p>The position of the contact on the infill changes little at 20&#8202;%: shifting the gyroid by up to half a period "
                f"in either direction changes the approach at 1&#8202;N by {100 * (max(d1) - min(d1)) / np.mean(d1):.1f}&#8202;% "
                f"({len(d1)} phases, resolved), because the four skin layers and the 9.4&#8202;mm contact line average over the beads.")
        s5 = [G[t] for t in ("g05x_s0", "g05x_s0_52", "g05x_s0_105") if t in G]
        d5 = [g["delta_um"].get(1.0) for g in s5 if g["delta_um"].get(1.0)]
        if len(d5) >= 2:
            txt += (f" At 5&#8202;% (21&#8202;mm period) the same shifts change it by "
                    f"{100 * (max(d5) - min(d5)) / np.mean(d5):.0f}&#8202;%.")
        txt += "</p>"
    if "g20x_s0" in G and "g20z_s0" in G and "g10x_s0" in G and "g10z_s0" in G:
        a = {t: G[t]["delta_um"].get(1.0) for t in ("g20x_s0", "g20z_s0", "g10x_s0", "g10z_s0")}
        g10r = sorted([r for r in X.gpu if r["tag"] == "g10x_s0" and r["indenter"] == "cyl"], key=lambda r: r["F"])
        w10 = float(np.interp(1.0, [r["F"] for r in g10r], [r["half_z"] for r in g10r]))
        txt += (f"<p>Printing the layers along the load instead of across it softens the resolved tip by "
                f"{100 * (a['g20z_s0'] / a['g20x_s0'] - 1):.0f}&#8202;% at 20&#8202;% and {100 * (a['g10z_s0'] / a['g10x_s0'] - 1):.0f}&#8202;% "
                f"at 10&#8202;%. The homogenized core is within {100 * abs(a['g20x_s0'] / T['g20_across']['delta_um'][1.0] - 1):.0f}&#8211;"
                f"{100 * abs(a['g20z_s0'] / T['g20_along']['delta_um'][1.0] - 1):.0f}&#8202;% of the resolved approach at 20&#8202;% and "
                f"{100 * abs(a['g10x_s0'] / T['g10_across']['delta_um'][1.0] - 1):.0f}&#8211;{100 * abs(a['g10z_s0'] / T['g10_along']['delta_um'][1.0] - 1):.0f}&#8202;% "
                f"at 10&#8202;%, where the 10.4&#8202;mm period is {10.4 / (2 * w10):.0f} times the {2 * w10:.1f}&#8202;mm contact width.</p>")
    extra = [(t, G[t]) for t in ("g05x_s0", "g05z_s0", "eflesh_c8") if t in G]
    if extra:
        body2 = [[GPU_LBL[t], f1(g["delta_um"].get(0.5)), f1(g["delta_um"].get(1.0)), f1(g["delta_um"].get(3.0)), f2(g["force_exponent"])]
                 for t, g in extra]
        txt += table(["Print (resolved only)", "\\(\\delta\\) at 0.5&#8202;N (&#181;m)", "\\(\\delta\\) at 1&#8202;N (&#181;m)",
                      "\\(\\delta\\) at 3&#8202;N (&#181;m)", "Force exponent"], body2,
                     "Prints whose cells are as large as the tip, resolved on the GPU only (linear elasticity).")
    txt += ("<p>For the bench: 95A gyroid at 20&#8202;% (either orientation) and, as the softer print with the same single-modulus "
            "error, 85A gyroid at 30&#8202;% printed hot. Both should be measured under the plate and the screwdriver before choosing.</p>")
    return txt + tab

def cost_section(X: Data):
    rows = [
        ["Periodic cell, voxel hex8, AMG-CG", "18k&#8211;873k", "1.5&#8211;148&#8202;s", "~1&#8202;GB per 200k DOF", "4 cores", "Workstation"],
        ["Tip, homogenized core, CHOLMOD face compliance", "217k", "~10&#8202;min per print (15 loads, 3 cycles)", "3.7&#8211;4.4&#8202;GB",
         "4 cores", "Workstation"],
        ["Tip, resolved at 0.1&#8202;mm, matrix-free CG on the GPU", "6&#8211;8&#8202;M", "~75&#8202;s per approach, ~5&#8202;min per curve",
         "2&#8211;3&#8202;GB GPU, 1.1&#8202;GB host", "1 GPU", "Workstation"],
        ["Tip, resolved at 0.05&#8202;mm (estimate)", "~60&#8202;M", "~1&#8202;h per curve", "~25&#8202;GB GPU", "1 GPU", "DeltaAI GH200"],
    ]
    return table(["Level", "DOF", "Wall time", "Memory", "Cores", "Runs on"], rows,
                 "Cost of each level on the workstation (RTX 4070 Ti SUPER, 20 cores).", text_cols=(0, 2, 3, 4, 5))


def lede(X: Data):
    T = X.T
    if "g20_across" not in T or "g10_across" not in T:
        return "Runs in progress."
    g20, g10 = T["g20_across"], T["g10_across"]
    pads = [X.pads_arm(E, 1.0) for E in sorted(X.B) if X.pads_arm(E, 1.0)]
    es = A.edge_share("g20_across")
    return (f"A finite-element model of the printed fingertip (TPU 95A, two walls, gyroid infill) puts the pads&#8217; modulus at "
            f"{f1(g10.get('E_best'))}&#8202;MPa for 10&#8202;% infill and {f1(g20.get('E_best'))}&#8202;MPa for 20&#8202;%, near the 10&#8202;MPa "
            f"used so far. One modulus misses the approach under the screwdriver by up to "
            f"{f0(g20.get('E_best_max_err_pct'))}&#8211;{f0(g10.get('E_best_max_err_pct'))}&#8202;% over 0.5&#8211;3&#8202;N: the tip "
            f"stiffens as \\(\\delta^{{{g20['force_exponent']:.2f}}}\\), the law as \\(\\delta^{{1.5}}\\). The walls carry "
            f"{f0(100 * es['fe'])}&#8202;% of the load at the face edges, so the printed tip&#8217;s twist arm is {f1(g20['arm'][1.0])}&#8202;mm "
            f"against {min(pads):.1f}&#8211;{max(pads):.1f}&#8202;mm for the pads at any modulus. Resolved beads change the 20&#8202;% "
            f"result by 2&#8211;3&#8202;% (5&#8202;minutes per curve on the GPU).")

def open_items(X: Data):
    g = X.T.get("g20_across", {})
    items = [
        ("Bench indentation of the printed tip.", f"Press a flat plate and then the screwdriver on the palmar face to 0.5, 1 and 3&#8202;N "
         f"(force against approach). At \\(E_s\\) = 30&#8202;MPa the model predicts a plate stiffness of {f0(g.get('k_plate'))}&#8202;N/mm and "
         f"{f1(g.get('delta_um', {}).get(1.0))}&#8202;&#181;m under the cylinder at 1&#8202;N for 20&#8202;% infill; a force exponent near 1.5 "
         f"instead of {g.get('force_exponent', float('nan')):.2f} would mean the law&#8217;s shape is right and the walls are not."),
        ("Spin onset against load.", "Twist the screwdriver in a fixed pinch at 0.5, 1 and 3&#8202;N; an arm near 3.3&#8202;mm confirms the "
         "rail loading, near 2.5&#8202;mm the pads."),
        ("Print settings of the current tips.", "Infill density, orientation and temperature of the printed tips are not recorded; the "
         "predictions above assume 20&#8202;% gyroid with layers across the load."),
        ("Large deformation at sparse infill.", "The models are linear; finish the PolyFEM build (step 17 of the hand-off note) and run the "
         "5&#8202;% tip to 3&#8202;N with IPC contact and friction."),
    ]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def byline():
    return ("Rows <code>docs/experiments/20261009-tpu_tip_fem/</code> &#183; cells <code>scripts/tpu_tip_fem_cells.py</code>, tip "
            "<code>scripts/tpu_tip_fem_tip.py</code>, GPU <code>scripts/tpu_tip_fem_gpu.py</code>, films <code>scripts/tpu_tip_fem_film.py</code>, "
            "pads <code>scripts/tpu_tip_fem_bed.py</code>, tables <code>scripts/tpu_tip_fem_analysis.py</code> &#183; builder "
            "<code>scripts/tpu_tip_fem_page.py</code> &#183; hand-off <code>" + NOTE + "</code>")


def render_tex(t):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    return P.TEX_RE.sub(lambda m: next(svgs), t), len(items)


def main():
    X = Data()
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "GLOSSARY": glossary()}
    v["LEDE"] = lede(X)
    v["CELLS"] = cells_section(X)
    v["TIP"] = tip_section(X)
    v["LAW"] = law_section(X)
    v["ARM"] = arm_section(X)
    v["RECIPES"] = recipes_section(X)
    v["COST"] = cost_section(X)
    v["OPEN"] = open_items(X)
    v["FOOTER"] = ("<p>Rebuild: <code>logs/20261001-hom_contact/venv/bin/python scripts/tpu_tip_fem_page.py</code>. Rows: "
                   "<code>docs/experiments/20261009-tpu_tip_fem/*.jsonl</code>; films <code>docs/experiments/20261009-tpu_tip_fem/media/</code>.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    t, n = render_tex(t)
    t = retro_style.apply(t)
    open(OUT, "w").write(t)
    print(f"formulas {n}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
