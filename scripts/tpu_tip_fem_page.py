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
READOUT_PAGE = "../20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html"
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
.tw th{white-space:normal !important;vertical-align:bottom}
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
           "g05z_s0": "95A 5&#8202;% along", "eflesh_c8": "Lattice of eFlesh cut cells, 8&#8202;mm"}


# ------------------------------------------------------------------------------------------ data

class Data:
    def __init__(self):
        self.cells = A.rows("cells.jsonl")
        self.tip = A.rows(A.TIP_ROWS)                                  # homogenized core, gaps from the STL surface
        self.gpu = [r for r in A.rows("gpu_runs.jsonl") if r["tag"].endswith(A.SUFFIX)]
        self.T = A.recipe_table()
        self.R = A.resolved_table()
        self.B = A.bed_table()
        self.nl = A.rows("gpu_nl_runs.jsonl")
        self.pf = A.rows("polyfem_runs.jsonl")
        self.pc = A.rows("plate_check.jsonl")
        self.val = A.rows("validate.jsonl")
        self.tol = A.rows("gpu_tol.jsonl")
        self.gpu_all = A.rows("gpu_runs.jsonl")

    def cell(self, kind, rho, h=None):
        rs = [r for r in self.cells if r["kind"] == kind and r.get("rho_req") is not None and abs(r["rho_req"] - rho) < 1e-9
              and (h is None or abs(r["h_mm"] - h) < 1e-9)]
        return min(rs, key=lambda r: r["h_mm"]) if rs else None

    def gpu_rows(self, base, ind="cyl"):
        tag = base + A.SUFFIX
        return sorted({r["delta_mm"]: r for r in self.gpu if r["tag"] == tag and r["indenter"] == ind}.values(),
                      key=lambda r: r["delta_mm"])

    def pads_arm(self, E, N=3.0):
        return A.pads_arm_at(self.B, E, N)

    def arm_err(self, E, arm_fe, N=3.0):
        pa = self.pads_arm(E, N)
        return (pa / arm_fe - 1) * 100 if pa and arm_fe else None

    def nl_row(self, tag, material, delta):
        rs = [r for r in self.nl if r.get("tag") == tag and r["material"] == material and abs(r["delta_mm"] - delta) < 1e-9]
        return rs[-1] if rs else None


# ------------------------------------------------------------------------------------------ html helpers

REFS = {}


def figure(svg, caption, legend=None, key=None):
    FIG[0] += 1
    if key:
        REFS[f"fig:{key}"] = FIG[0]
    lg = P._legend_html(legend) if legend else ""
    return f'<figure class="diagram">{svg}{lg}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def table(head, body, caption=None, note=None, text_cols=(0,), key=None):
    out = []
    if caption:
        TAB[0] += 1
        if key:
            REFS[f"tab:{key}"] = TAB[0]
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


def sgn(x, nd=1, unit=True):
    """Signed percentage for the checks table: explicit plus, typographic minus."""
    t = f"{x:+.{nd}f}".replace("-", "&#8722;")
    return t + ("&#8202;%" if unit else "")


def pct(x, nd=0):
    """Signed percentage with a typographic minus; '0' when it rounds to zero."""
    if not _ok(x):
        return "&#8211;"
    t = f"{x:.{nd}f}"
    if float(t) == 0:
        return "0"
    return t.replace("-", "&#8722;")


def f2(x):
    return fmt(x, 2) if _ok(x) else "&#8211;"


# ------------------------------------------------------------------------------------------ terms

GLOSSARY = [
    ("Solid modulus \\(E_s\\)", "Young&#8217;s modulus of fully dense printed TPU. 95A: 26&#8202;MPa (NinjaTek Cheetah, ASTM D638), "
     "33.6&#8202;MPa (Polymaker PolyFlex TPU95, ISO 37), 67&#8202;MPa (UltiMaker TPU 95A, ASTM D3039); Overture&#8217;s sheet gives only "
     "the stress at 100&#8202;% strain, 9.7&#8202;MPa. This page uses 30&#8202;MPa for 95A and 12&#8202;MPa for 85A (NinjaFlex, ASTM D638), "
     "and Poisson&#8217;s ratio 0.45, which no TPU sheet lists. In the linear models every stiffness scales with \\(E_s\\)."),
    ("Infill fraction \\(\\rho\\)", "Requested gyroid density in the slicer; the printed cell&#8217;s solid fraction is 4&#8211;12&#8202;% higher (Table&#160;[[tab:cells]])."),
    ("Gyroid period \\(L\\)", "\\(L = 2\\pi s / (2.44\\,\\rho)\\) with line spacing \\(s = 0.407\\)&#8202;mm (0.45&#8202;mm lines, 0.2&#8202;mm layers): "
     "21, 10.4, 5.2, 3.4 and 2.0&#8202;mm at 5, 10, 20, 30 and 50&#8202;% (Table&#160;[[tab:cells]])."),
    ("\\(E_x\\), \\(E_z\\)", "Homogenized Young&#8217;s moduli of the printed infill in the layer plane and along the build direction, per unit \\(E_s\\)."),
    ("Layers across, along", "Layers across the load: the palmar face is printed as a top skin, the part lying on its back face. Layers along the "
     "load: the face is a perimeter wall, the part standing on its mount face."),
    ("Approach \\(\\delta\\)", "Displacement of the rigid indenter into the palmar face from first touch, in &#181;m. The indenter is the "
     "screwdriver shaft, a cylinder of radius 12.5&#8202;mm with its axis across the finger, at the contact bed&#8217;s position."),
    ("Contact stiffness \\(k\\)", "\\(dF/d\\delta\\) under the cylinder at 1&#8202;N, in N/mm. Plate stiffness: force over approach of a flat plate on "
     "the palmar face."),
    ("Pressure law", "The pads&#8217; Winkler law on the tip block, \\(p = E\\,d/h_f\\), with \\(d\\) the local overlap of the undeformed bodies and "
     "\\(h_f = 8.5\\)&#8202;mm; every contact model of this project has used \\(E\\) = 10&#8202;MPa."),
    ("Best single modulus \\(E^*\\)", "The \\(E\\) that minimizes the RMS log error of the law&#8217;s approach against the finite-element approach "
     "under the cylinder at 0.5, 1 and 3&#8202;N. Approach error: the largest of the three errors, in %."),
    ("Twist arm", "Torque at spin onset over \\(\\mu N\\): \\(\\sum p_i r_i / \\sum p_i\\) about the normal through the force centre, every contact "
     "point sliding. For the pads it is the onset arm of bed task T2 at the same normal force. Arm error: pads&#8217; arm at \\(E^*\\) over "
     "the printed tip&#8217;s arm, minus one, at 3&#8202;N."),
    ("Edge load", "Share of the cylinder&#8217;s force on the face nodes with \\(|y| > 3.5\\)&#8202;mm, the outer 0.9&#8202;mm of the 8.8&#8202;mm "
     "flat face and the fillets beyond it, at 1&#8202;N. The pressure law on the same nodes gives 23&#8202;%."),
    ("Contact half-width \\(b\\)", "Pressure-weighted RMS distance of the contact from its centre line along the finger, in mm. Width exponent: "
     "\\(d\\ln b/d\\ln F\\), 1/3 for the law on a cylinder and 1/2 for an elastic half-space."),
    ("Orientation spread", "Approach at 1&#8202;N with the layers along the load over the approach with the layers across, minus one."),
    ("Location spread", "Range over the mean of the approach (or arm) at 1&#8202;N when the resolved gyroid is shifted under the contact by a "
     "quarter or half period in \\(y\\), \\(z\\) or both."),
    ("Homogenized, resolved", "Core cells given the homogenized tensor of the printed infill on a graded hexahedral grid (217k DOF), or every "
     "bead voxelized at 0.1&#8202;mm (1.7&#8211;2.2&#8202;M elements) and solved on the GPU."),
    ("Neo-Hookean", "Compressible Neo-Hookean solid with the same small-strain \\(E_s\\) and \\(\\nu\\): the large-deformation check of the "
     "5&#8202;% tip."),
    ("Magnet depth", "Distance from the palmar face to the magnet&#8217;s top face, in mm; the magnet lies under the contact point "
     "with its axis normal to the face."),
    ("Magnet travel", "Displacement of the magnet&#8217;s centre toward the back face under 1&#8202;N on the face, in &#181;m; along and "
     "across the finger, under 1&#8202;N of shear on a 1&#8202;N contact, in &#181;m/N."),
    ("Signal, force noise", "Signal: change of the field at the magnetometer per newton of load, in &#181;T/N, from a point dipole of the "
     "magnet&#8217;s moment. Force noise: the load whose signal equals one standard deviation of one MLX90393 sample at "
     "141&#8202;Hz (GAIN_SEL 7, RES 0, OSR 0, DIG_FILT 5: 1.15&#8202;&#181;T on the chip&#8217;s in-plane axes, 1.75&#8202;&#181;T normal "
     "to it, from the datasheet&#8217;s Figures 2 and 3), in N. It combines the finite-element travel, the dipole field and the "
     "datasheet noise; drift, the Earth&#8217;s field and other magnets are on the <a href=\"" + READOUT_PAGE + "\">readout page</a>."),
    ("Chip distance", "Distance of the MLX90393&#8217;s sensing point behind the tip&#8217;s back face, in mm; negative when the chip is "
     "set into the tip."),
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


RES_LBL = {"g05x_s0": "95A, gyroid 5&#8202;%, layers across", "g05z_s0": "95A, gyroid 5&#8202;%, layers along",
           "eflesh_c8": "Lattice of eFlesh cut cells, 8&#8202;mm"}
PAIRS = [("g20_across", COL["x"]), ("s85_50_across", COL["sheet"]), ("g10_across", COL["z"]), ("s85_30_across", COL["res"])]


def svg_force(X: Data):
    """Force against approach under the cylinder, log-log: homogenized prints (lines), resolved tips (hollow markers), the
    law at the best single modulus of the 20 % tip (dashed)."""
    out = P._svg_open(990, 500, "Force against approach under the screwdriver cylinder")
    fx, fy = P._panel(out, 90, 40, 840, 380, (3, 1500), (0.2, 5), [3, 10, 30, 100, 300, 1000], [0.25, 0.5, 1, 2, 4],
                      "approach under the cylinder (&#181;m)", "Force (N)", logx=True, logy=True)
    for name, col in [("solid95", COL["ref"])] + PAIRS:
        rr = sorted([r for r in X.tip if r["recipe"] == name and r["indenter"] == "cyl"], key=lambda r: r["F_target"])
        P._path(out, fx, fy, [(r["delta_mm"] * 1e3, r["F"]) for r in rr], col)
    for base, col, shape in (("g20x_s0", COL["x"], "circle"), ("g10x_s0", COL["z"], "circle"), ("g05x_s0", COL["ref"], "diamond"),
                             ("eflesh_c8", COL["pads"], "cross")):
        for r in X.gpu_rows(base):
            if 0.2 <= r["F"] <= 5:
                P._marker(out, fx(r["delta_mm"] * 1e3), fy(r["F"]), col, shape=shape, hollow=True)
    g = X.T.get("g20_across")
    if g and "E_best" in g:
        d = np.load(os.path.join(A.TIP_DATA, "g20_across.npz"))
        A_, g0 = d["cyl_1__A"], d["cyl_1__g0"]
        pts = [(1e3 * dl, g["E_best"] * A.law_f(A_, g0, dl)) for dl in np.geomspace(0.003, 0.4, 40)]
        P._path(out, fx, fy, [(x, y) for x, y in pts if 0.2 <= y <= 5], COL["x"], dashed=True, width=1.6)
    out.append("</svg>")
    return out


def svg_arm(X: Data):
    """Twist arm against normal force: the printed tips of the two matched-stiffness pairs (lines) and the pads at each
    print's best single modulus (squares, bed T2)."""
    out = P._svg_open(990, 470, "Twist arm against normal force")
    fx, fy = P._panel(out, 90, 40, 840, 350, (0.4, 4.2), (2.2, 3.8), [0.5, 1, 2, 3, 4], [2.2, 2.6, 3.0, 3.4, 3.8],
                      "normal force per pad (N)", "Twist arm (mm)", logx=True)
    for name, col in PAIRS:
        rr = sorted([r for r in X.tip if r["recipe"] == name and r["indenter"] == "cyl"], key=lambda r: r["F_target"])
        P._path(out, fx, fy, [(r["F"], r["arm"]) for r in rr if 0.45 <= r["F"] <= 4.2], col)
        E = X.T[name]["E_best"]
        pts = [(N, X.pads_arm(E, N)) for N in (1.0, 3.0) if X.pads_arm(E, N)]
        P._path(out, fx, fy, pts, col, dashed=True, width=1.4)
        for x, y in pts:
            P._marker(out, fx(x), fy(y), col, shape="square")
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
           f"\\(E_z/E_s = {fit[('slicer', 'Ez')][0]:.2f}\\,\\rho^{{{fit[('slicer', 'Ez')][1]:.2f}}}\\) (Figure&#160;[[fig:cells]]), between Gibson and Ashby&#8217;s "
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
                "Homogenized moduli of the printed gyroid (<code>cells.jsonl</code>); the finest voxel size run for each density.", key="cells")
    fig = figure("".join(svg_cells(X)), "Moduli of printed gyroid infill against solid fraction: in the layer plane (blue), along the build "
                 "direction (orange), the ideal sheet gyroid (green), eFlesh cut cells (diamonds), and the Gibson&#8211;Ashby limits "
                 "\\(\\rho^2\\) and \\(\\rho/3\\) (dashed).",
                 legend=[("Printed, layer plane", COL["x"], False, "circle"), ("Printed, build direction", COL["z"], False, "circle"),
                         ("Ideal sheet", COL["sheet"], False, "circle"), ("eFlesh cut cell", COL["ref"], False, "diamond"),
                         ("Gibson&#8211;Ashby", COL["ref"], True, None)], key="cells")
    return txt + fig + tab


def film_stats(name):
    path = os.path.join(D, "data", f"film_{name}.npz")
    if not os.path.exists(path):
        return None
    S = np.load(path)
    p, X = S["p"][-1] / 0.01, S["X_face"]                           # nodal force over the 0.1 mm voxel face, MPa
    end, mid = np.abs(X[:, 1]) > A.EDGE_Y, np.abs(X[:, 1]) < 2.0
    return dict(F=float(S["F"][-1]), delta_um=float(S["deltas"][-1]) * 1e3, p_end=float(p[end].max()), p_mid=float(p[mid].max()))


def tip_section(X: Data):
    parts = ["<p>The insert of <code>fingertip (3).stl</code> (17.0 &#215; 14.8 &#215; 21.4&#8202;mm, a flat palmar face 8.8&#8202;mm wide between "
             "3&#8202;mm fillets, a slot and two sockets for the mount) is printed with two 0.45&#8202;mm perimeters and four 0.2&#8202;mm top and "
             "bottom layers around gyroid infill. The back and mount faces are fixed. The screwdriver cylinder presses the face 10.8&#8202;mm from the "
             "mount, its axis across the finger, as in the contact bed. Contact gaps are taken from the STL surface, so the fillets keep their "
             "curvature on both grids.</p>"]
    pics = [p for p in ("20261009-classes.png", "20261009-voxels_g20x.png") if os.path.exists(media(p))]
    if pics:
        cells = "".join(f'<div class="cell"><img src="{data_uri(media(p), "image/png")}" alt="{p}"></div>' for p in pics)
        FIG[0] += 1
        parts.append(f'<figure class="pics">{cells}<figcaption>Figure&#160;{FIG[0]}. Slices of the printed insert: walls, skins and '
                     f'core for both print orientations, and the resolved 20&#8202;% gyroid with the layers across the load.</figcaption></figure>')
    films = [(t, media(f"20261009-film_{t}.mp4")) for t in ("g20x_s0", "g05x_s0", "eflesh_c8")]
    films = [(t, p) for t, p in films if os.path.exists(p)]
    st = {t: film_stats(t) for t, _ in films}
    if films and all(st.values()):
        cells = "".join(f'<div class="cell"><video controls muted loop playsinline preload="metadata" '
                        f'poster="{data_uri(p.replace(".mp4", ".png"), "image/png")}" src="{data_uri(p, "video/mp4")}"></video>'
                        f'<p class="filmlab">{GPU_LBL[t]}</p></div>' for t, p in films)
        FIG[0] += 1
        REFS["fig:films"] = FIG[0]
        g, f5, ef = st["g20x_s0"], st["g05x_s0"], st["eflesh_c8"]
        parts.append(f'<figure class="films">{cells}<figcaption>Figure&#160;{FIG[0]}. The resolved tips under the cylinder over 13 approaches from '
                     f'first touch: the section through the contact line, displacement magnified 15, 5 and 2 times, and the pressure on the palmar '
                     f'face. On the 20&#8202;% tip the contact is a line across the face whose ends, over the side walls, reach {f1(g["p_end"])}&#8202;MPa '
                     f'against {f1(g["p_mid"])}&#8202;MPa mid-line at {f1(g["F"])}&#8202;N. On the 5&#8202;% tip the 0.8&#8202;mm skin bends between '
                     f'the walls across a 21&#8202;mm cell, and the walls and one strut support the cylinder ({f1(f5["p_end"])} and {f1(f5["p_mid"])}&#8202;MPa '
                     f'at {f1(f5["F"])}&#8202;N). The eFlesh lattice, clipped to the insert without a skin, touches the cylinder only where struts reach '
                     f'the face; at an approach of {f0(ef["delta_um"])}&#8202;&#181;m the force is {f2(ef["F"])}&#8202;N.</figcaption></figure>')
    else:
        parts.append(P.pending("Films not rendered yet."))
    return "".join(parts)


ORDER = ["solid95", "g40_across", "g30_across", "g20_across", "g20_along", "g10_across", "g10_along", "s85_100", "s85_50_across",
         "s85_50_along", "s85_30_across", "s85_30_along"]


def law_section(X: Data):
    T, R = X.T, X.R
    if "g20_across" not in T:
        return P.pending("Tip rows not written yet.")
    g = T["g20_across"]
    ex = [T[n]["force_exponent"] for n in T]
    er = [T[n]["E_best_max_err_pct"] for n in T]
    txt = (f"<p>Under the screwdriver the 20&#8202;% tip printed with its layers across the load sinks {f1(g['delta_um'][0.5])}, "
           f"{f1(g['delta_um'][1.0])} and {f1(g['delta_um'][3.0])}&#8202;&#181;m at 0.5, 1 and 3&#8202;N (Figure&#160;{FIG[0] + 1}). Its force grows as "
           f"\\(\\delta^{{{g['force_exponent']:.2f}}}\\), while the pressure law on a cylinder gives \\(\\delta^{{1.5}}\\). The modulus that reproduces "
           f"each approach falls from {f1(g['E_fit_cyl'][0.5])} to {f1(g['E_fit_cyl'][3.0])}&#8202;MPa over that range; the best single modulus, "
           f"{f1(g['E_best'])}&#8202;MPa, misses by up to {f0(g['E_best_max_err_pct'])}&#8202;%. Every print shows the mismatch, with force exponents "
           f"of {min(ex):.2f}&#8211;{max(ex):.2f} and approach errors of {f0(min(er))}&#8211;{f0(max(er))}&#8202;% (Table&#160;{TAB[0] + 1}): under a "
           f"cylinder the force on an elastic body grows almost linearly with approach, and the law&#8217;s grows as the overlapping area times "
           f"the overlap.</p>")
    if "g20x_s0" in R and "g10x_s0" in R:
        txt += (f"<p>The resolved tips give \\(E^*\\) = {f1(R['g20x_s0']['E_best'])}&#8202;MPa at 20&#8202;% and {f1(R['g10x_s0']['E_best'])}&#8202;MPa "
                f"at 10&#8202;%, against {f1(g['E_best'])} and {f1(T['g10_across']['E_best'])}&#8202;MPa for the homogenized core, although the "
                f"gyroid period (5.2 and 10.4&#8202;mm) exceeds the contact width. The 5&#8202;% tip, resolved only, gives "
                f"{f1(R['g05x_s0']['E_best'])}&#8202;MPa.</p>")
    leg = [("95A solid", COL["ref"], False, None), ("95A 20&#8202;% across", COL["x"], False, None), ("85A 50&#8202;% across", COL["sheet"], False, None),
           ("95A 10&#8202;% across", COL["z"], False, None), ("85A 30&#8202;% across", COL["res"], False, None),
           ("Resolved 95A 5&#8202;%", COL["ref"], False, "diamond"), ("Resolved eFlesh", COL["pads"], False, "cross"),
           ("Law at the 20&#8202;% tip&#8217;s \\(E^*\\)", COL["x"], True, None)]
    fig = figure("".join(svg_force(X)), "Force against approach under the cylinder: homogenized cores (lines), resolved tips (hollow markers: "
                 "95A 20&#8202;% and 10&#8202;% circles, 5&#8202;% diamonds, eFlesh crosses), and the pressure law at the best single modulus of the "
                 "20&#8202;% tip (dashed).", legend=leg)
    body = []
    for name in [n for n in ORDER if n in T]:
        r = T[name]
        body.append([RECIPE_LBL.get(name, name), f"{r['E_s']:.0f}", f1(r["delta_um"][1.0]), f2(r["force_exponent"]), f1(r["k_1N"]),
                     f1(r["E_fit_cyl"][0.5]), f1(r["E_fit_cyl"][3.0]), f1(r["E_best"]), f0(r["E_best_max_err_pct"]), f2(r.get("b_exponent"))])
    tab = table(["Print", "\\(E_s\\) (MPa)", "Approach at 1&#8202;N (&#181;m)", "Force exponent", "Stiffness at 1&#8202;N (N/mm)",
                 "Law modulus at 0.5&#8202;N (MPa)", "Law modulus at 3&#8202;N (MPa)", "\\(E^*\\) (MPa)", "Approach error (%)", "Width exponent"], body,
                "Approach, stiffness and the pressure-law modulus per print under the cylinder (homogenized core, <code>tip_runs_surf.jsonl</code>).",
                key="law")
    return txt + fig + tab


def arm_section(X: Data):
    T = X.T
    if not all(n in T for n, _ in PAIRS) or not X.B:
        return P.pending("Rows not written yet.")
    es = {n: A.edge_share(n) for n in ("solid95", "g20_across", "g10_across", "s85_50_across")}
    g, s = T["g20_across"], T["s85_50_across"]
    c20, c50 = X.cell("slicer", 0.2), X.cell("slicer", 0.5)
    con = (f" The contrast between the solid walls and the core in the layer plane, \\(E_s/E_x\\), is {1 / c20['Ex']:.0f} at 20&#8202;% infill and "
           f"{1 / c50['Ex']:.0f} at 50&#8202;% (Table&#160;[[tab:cells]]).") if c20 and c50 else ""
    txt = (f"<p>The walls carry the edges of the face like rails (Figure&#160;[[fig:films]]): under the cylinder {f0(100 * es['g20_across']['fe'])}&#8202;% of the "
           f"20&#8202;% tip&#8217;s force lies on the outer 0.9&#8202;mm of the flat face and on the fillets, against {f0(100 * es['g20_across']['law'])}&#8202;% "
           f"for the pressure law on the same nodes ({f0(100 * es['g10_across']['fe'])}&#8202;% at 10&#8202;% infill, {f0(100 * es['solid95']['fe'])}&#8202;% "
           f"for the solid print). Load at the ends of the contact line lengthens the twist arm: at 3&#8202;N it is {f2(g['arm'][3.0])}&#8202;mm on the "
           f"20&#8202;% tip against {f2(X.pads_arm(g['E_best']))}&#8202;mm for the pads at \\(E^*\\) (Figure&#160;{FIG[0] + 1}). The pads approach that "
           f"arm only at {min(X.B):g}&#8202;MPa ({f2(X.pads_arm(min(X.B)))}&#8202;mm), a fifth of the modulus that reproduces the approach.</p>"
           f"<p>A softer TPU at higher infill reaches the same contact stiffness with stiffer infill relative to its walls. 85A gyroid at 50&#8202;% "
           f"({f1(s['k_1N'])}&#8202;N/mm at 1&#8202;N against {f1(g['k_1N'])} for 95A at 20&#8202;%) puts {f0(100 * es['s85_50_across']['fe'])}&#8202;% "
           f"of the load on the edges, and its arm at 3&#8202;N, {f2(s['arm'][3.0])}&#8202;mm, is {f1(abs(X.arm_err(s['E_best'], s['arm'][3.0])))}&#8202;% "
           f"from the pads&#8217; at its \\(E^*\\) of {f1(s['E_best'])}&#8202;MPa.{con} In these linear models the TPU grade scales every stiffness "
           f"by \\(E_s\\); the infill fraction sets the share of the walls.</p>")
    leg = [(lab, col, False, None) for lab, col in (("95A 20&#8202;% across", COL["x"]), ("85A 50&#8202;% across", COL["sheet"]),
                                                     ("95A 10&#8202;% across", COL["z"]), ("85A 30&#8202;% across", COL["res"]))]
    leg.append(("Pads at the print&#8217;s \\(E^*\\)", COL["ref"], True, "square"))
    fig = figure("".join(svg_arm(X)), "Twist arm against normal force: printed tips of the two pairs of equal contact stiffness (lines, homogenized "
                 "core) and the pads at each print&#8217;s best single modulus (squares and dashed lines, bed task T2; the bed gives no onset at "
                 "1&#8202;N below 7.5&#8202;MPa).", legend=leg)
    return txt + fig


def recipes_section(X: Data):
    T, R = X.T, X.R
    if len(T) < 10:
        return P.pending("Recipe rows not written yet.")
    pair = {"g20_across": "g20_along", "g10_across": "g10_along", "s85_30_across": "s85_30_along", "s85_50_across": "s85_50_along"}
    pair.update({v: k for k, v in list(pair.items())})
    rows = []
    for name, r in T.items():
        a = r["arm"][3.0]
        o = pair.get(name)
        across = name if name.endswith("across") else o
        spread = (T[pair[across]]["delta_um"][1.0] / T[across]["delta_um"][1.0] - 1) * 100 if o and across in T and pair[across] in T else None
        rows.append(dict(lbl=RECIPE_LBL.get(name, name), model="Homogenized", full=True, k=r["k_1N"], E=r["E_best"], err=r["E_best_max_err_pct"],
                         arm=a, pads=X.pads_arm(r["E_best"]), aerr=X.arm_err(r["E_best"], a), edge=A.edge_share(name)["fe"], spread=spread))
    for base in ("g05x_s0", "g05z_s0", "eflesh_c8"):
        if base not in R:
            continue
        r = R[base]
        a = r["arm"].get(3.0)
        spread = None
        if base.startswith("g05") and "g05x_s0" in R and "g05z_s0" in R:
            spread = (R["g05z_s0"]["delta_um"][1.0] / R["g05x_s0"]["delta_um"][1.0] - 1) * 100
        rows.append(dict(lbl=RES_LBL[base], model="Resolved", full=len(r["cases"]) == 3, k=r["k_1N"], E=r["E_best"], err=r["E_best_max_err_pct"], arm=a,
                         pads=X.pads_arm(r["E_best"]) if a else None, aerr=X.arm_err(r["E_best"], a) if a else None,
                         edge=r.get("edge_share_1N"), spread=spread))
    rows.sort(key=lambda r: (r["aerr"] is None, abs(r["aerr"]) if r["aerr"] is not None else 0))
    body = [[r["lbl"], r["model"], f1(r["k"]), f1(r["E"]) if r["E"] >= 1 else f2(r["E"]), f0(r["err"]), f2(r["arm"]), f2(r["pads"]),
             pct(r["aerr"], 1), f0(100 * r["edge"]) if _ok(r["edge"]) else "&#8211;", pct(r["spread"], 0)] for r in rows]
    tab = table(["Print", "Model", "Stiffness at 1&#8202;N (N/mm)", "\\(E^*\\) (MPa)", "Approach error (%)", "Arm, tip (mm)", "Arm, pads (mm)",
                 "Arm error (%)", "Edge load (%)", "Orientation spread (%)"], body,
                "Prints ranked by how closely the pads at their best single modulus reproduce the twist arm at 3&#8202;N. A dash: the pads&#8217; "
                "bed covers 2.5&#8211;77&#8202;MPa, and the eFlesh rows stop at 1&#8202;N.", text_cols=(0, 1), key="recipes")
    k = {n: T[n]["k_1N"] for n in T}
    ae = {n: X.arm_err(T[n]["E_best"], T[n]["arm"][3.0]) for n in T}
    sh20 = [R[t]["delta_um"][1.0] for t in ("g20x_s0", "g20x_s26", "g20x_s0_26", "g20x_s13") if t in R and R[t]["delta_um"].get(1.0)]
    sh5 = [R[t]["delta_um"][1.0] for t in ("g05x_s0", "g05x_s0_52", "g05x_s0_105") if t in R and R[t]["delta_um"].get(1.0)]
    arm5 = [R[t]["arm"][1.0] for t in ("g05x_s0", "g05x_s0_52", "g05x_s0_105") if t in R and R[t]["arm"].get(1.0)]
    full = [r["err"] for r in rows if r["full"]]
    txt = (f"<p>The approach error of the best single modulus is {f0(min(full))}&#8211;{f0(max(full))}&#8202;% for "
           f"every print loaded to 3&#8202;N, so Table&#160;{TAB[0]} ranks the prints by the twist arm. At equal contact stiffness the softer TPU at higher infill "
           f"wins both pairs: 85A at 50&#8202;% against 95A at 20&#8202;% ({f1(k['s85_50_across'])} and {f1(k['g20_across'])}&#8202;N/mm; arm error "
           f"{pct(ae['s85_50_across'], 1)} against {pct(ae['g20_across'], 1)}&#8202;%), and 85A at 30&#8202;% against 95A at 10&#8202;% "
           f"({f1(k['s85_30_across'])} and {f1(k['g10_across'])}&#8202;N/mm; {pct(ae['s85_30_across'], 1)} against {pct(ae['g10_across'], 1)}&#8202;%). "
           f"Printing the layers along the load softens the 20&#8202;% tip by {f0((R['g20z_s0']['delta_um'][1.0] / R['g20x_s0']['delta_um'][1.0] - 1) * 100)}&#8202;% "
           f"(resolved) and lengthens its arm further ({pct(ae['g20_along'], 1)}&#8202;%); at 50&#8202;% the orientation changes the approach by "
           f"{f0((T['s85_50_along']['delta_um'][1.0] / T['s85_50_across']['delta_um'][1.0] - 1) * 100)}&#8202;%.</p>")
    if len(sh20) >= 2 and len(sh5) >= 2:
        txt += (f"<p>The position of the contact on the infill matters at sparse infill only: shifting the resolved gyroid under the cylinder changes "
                f"the approach at 1&#8202;N by {100 * (max(sh20) - min(sh20)) / np.mean(sh20):.1f}&#8202;% at 20&#8202;% ({len(sh20)} positions) and "
                f"{100 * (max(sh5) - min(sh5)) / np.mean(sh5):.0f}&#8202;% at 5&#8202;% ({len(sh5)} positions), and the 5&#8202;% tip&#8217;s arm from "
                f"{min(arm5):.2f} to {max(arm5):.2f}&#8202;mm. The 5&#8202;% tip ({f1(R['g05x_s0']['E_best'])}&#8202;MPa, {f1(R['g05x_s0']['k_1N'])}&#8202;N/mm) "
                f"lies near the pads&#8217; lower limit, since the pads fail the pull test of bed task T1 at 1&#8202;MPa, and the eFlesh lattice "
                f"({f2(R['eflesh_c8']['E_best'])}&#8202;MPa) lies below it.</p>")
    txt += (f"<p>For the bench: 85A gyroid at 50&#8202;% with two walls, printed hot, in either orientation (\\(E^*\\) "
            f"{f1(T['s85_50_along']['E_best'])}&#8211;{f1(T['s85_50_across']['E_best'])}&#8202;MPa, as stiff under the screwdriver as the 95A 20&#8202;% "
            f"tip), with the 95A 20&#8202;% tip as the reference; 85A at 30&#8202;% for a softer tip ({f1(T['s85_30_along']['E_best'])}&#8211;"
            f"{f1(T['s85_30_across']['E_best'])}&#8202;MPa, {f1(k['s85_30_across'])}&#8202;N/mm).</p>")
    return txt + tab


def checks_section(X: Data):
    rows = []
    v1 = [r for r in X.val if abs(r["F"] - 1.0) < 0.05]
    if v1:
        v = v1[-1]
        rows.append(["Solid insert under a 40&#8202;mm sphere, approach at 1&#8202;N", f"{v['delta_mm'] * 1e3:.2f}&#8202;&#181;m (graded hex grid)",
                     f"Hertz {v['delta_hertz_mm'] * 1e3:.2f}&#8202;&#181;m", sgn((v['delta_mm'] / v['delta_hertz_mm'] - 1) * 100, 1)])
    T, R = X.T, X.R
    if "g20x_s0" in R:
        rows.append(["95A 20&#8202;% across, approach at 1&#8202;N", f"{f1(R['g20x_s0']['delta_um'][1.0])}&#8202;&#181;m (resolved)",
                     f"{f1(T['g20_across']['delta_um'][1.0])}&#8202;&#181;m (homogenized)",
                     sgn((R['g20x_s0']['delta_um'][1.0] / T['g20_across']['delta_um'][1.0] - 1) * 100, 1)])
    for t in X.tol:
        if t["indenter"] == "cyl":
            b = [g for g in X.gpu_all if g["tag"] == t["tag"].replace("_tol1e-8", "") and g["indenter"] == "cyl" and abs(g["delta_mm"] - t["delta_mm"]) < 1e-9]
            if b:
                rows.append([f"CG tolerance 10<sup>&#8722;5</sup> against 10<sup>&#8722;8</sup>, 20&#8202;% tip at {t['delta_mm'] * 1e3:.0f}&#8202;&#181;m",
                             f"{b[0]['F']:.4f}&#8202;N", f"{t['F']:.4f}&#8202;N", sgn((b[0]['F'] / t['F'] - 1) * 100, 2)])
    pc = sorted(X.pc, key=lambda r: r["n_dof"])
    sp = [r for r in X.gpu_all if r["tag"] == "solid_x" + A.SUFFIX and r["indenter"] == "plate"]
    sh = [r for r in X.tip if r["recipe"] == "solid95" and r["indenter"] == "plate" and abs(r["F_target"] - 1.0) < 1e-9]
    if pc and sp and sh:
        ref = pc[-1]["k_N_per_mm"]
        rows.append(["Solid insert under a flat plate, stiffness", f"{1 / sh[0]['delta_mm']:.0f} (graded hex) and {sp[0]['F'] / sp[0]['delta_mm']:.0f}&#8202;N/mm "
                     f"(voxels)", f"{ref:.0f}&#8202;N/mm (quadratic tetrahedra, {pc[-1]['n_dof'] / 1e3:.0f}k DOF; {pc[0]['k_N_per_mm']:.0f} at "
                     f"{pc[0]['n_dof'] / 1e3:.0f}k)", sgn((1 / sh[0]['delta_mm'] / ref - 1) * 100, 0, unit=False) + " and " + sgn((sp[0]['F'] / sp[0]['delta_mm'] / ref - 1) * 100, 0)])
    gb = {(r["material"], round(r["delta_mm"], 3)): r for r in X.nl if r.get("case") == "block"}
    pf = {r["tag"]: r for r in X.pf}
    if ("neohookean", 1.002) in gb and "plate_h0.5_d1.0_NH" in pf:
        g_ratio = gb[("neohookean", 1.002)]["F"] / gb[("linear", 1.002)]["F"]
        p_ratio = pf["plate_h0.5_d1.0_NH"]["F_contact"] / pf["plate_h0.5_d1.0_LE"]["F_contact"]
        rows.append(["Block 10&#8202;&#215;&#8202;10&#8202;&#215;&#8202;5&#8202;mm at 20&#8202;% strain, Neo-Hookean over linear force",
                     f"{g_ratio:.3f} (GPU solver)", f"{p_ratio:.3f} (PolyFEM)", sgn((g_ratio / p_ratio - 1) * 100, 1)])
    s = pf.get("sphere_20x20x10_h0.08_sub6")
    if s:
        rows.append(["Sphere R&#8202;10&#8202;mm, 0.1&#8202;mm into a bonded 20&#8202;&#215;&#8202;20&#8202;&#215;&#8202;10&#8202;mm block",
                     f"{s['F_contact']:.3f}&#8202;N (PolyFEM, IPC contact)", f"{s['F_bonded_layer']:.3f}&#8202;N (bonded layer; Hertz {s['F_hertz']:.3f})",
                     sgn((s['F_contact'] / s['F_bonded_layer'] - 1) * 100, 0)])
    nl = []
    for d in (0.2, 0.35):
        a, b = X.nl_row("g05x_s0" + A.SUFFIX, "neohookean", d), X.nl_row("g05x_s0" + A.SUFFIX, "linear", d)
        if a and b:
            nl.append((d, a, b))
            rows.append([f"95A 5&#8202;% tip at {d * 1e3:.0f}&#8202;&#181;m, force", f"{a['F']:.2f}&#8202;N (Neo-Hookean)", f"{b['F']:.2f}&#8202;N (linear)",
                         sgn((a['F'] / b['F'] - 1) * 100, 0)])
    for d in (0.4, 0.7, 1.0):
        a, b = X.nl_row("eflesh_c8" + A.SUFFIX, "neohookean", d), X.nl_row("eflesh_c8" + A.SUFFIX, "linear", d)
        if a and b:
            rows.append([f"eFlesh lattice tip at {d * 1e3:.0f}&#8202;&#181;m, force", f"{a['F']:.2f}&#8202;N (Neo-Hookean)",
                         f"{b['F']:.2f}&#8202;N (linear)", sgn((a['F'] / b['F'] - 1) * 100, 0)])
    txt = ""
    if nl:
        d, a, b = nl[-1]
        d0, a0, b0 = nl[0]
        txt += (f"<p>With a compressible Neo-Hookean solid the force on the resolved 5&#8202;% tip is {(1 - a0['F'] / b0['F']) * 100:.0f}&#8202;% "
                f"below the linear model&#8217;s at {d0 * 1e3:.0f}&#8202;&#181;m and {(1 - a['F'] / b['F']) * 100:.0f}&#8202;% below at {d * 1e3:.0f}&#8202;&#181;m "
                f"({a['F']:.2f} against {b['F']:.2f}&#8202;N); its twist arm moves from {b['arm']:.2f} to {a['arm']:.2f}&#8202;mm, and no element loses "
                f"more than {(1 - a['J_min']) * 100:.0f}&#8202;% of its volume. The GPU solver for this check reproduces PolyFEM&#8217;s Neo-Hookean "
                f"block within 0.3&#8202;% (Table&#160;{TAB[0] + 1}).</p>")
        enl = [(X.nl_row("eflesh_c8" + A.SUFFIX, "neohookean", dd), X.nl_row("eflesh_c8" + A.SUFFIX, "linear", dd)) for dd in (0.4, 0.7, 1.0)]
        enl = [(dd, a_, b_) for dd, (a_, b_) in zip((0.4, 0.7, 1.0), enl) if a_ and b_]
        if enl:
            dd, a_, b_ = enl[-1]
            txt = txt.replace("</p>", f" On the eFlesh lattice at {dd * 1e3:.0f}&#8202;&#181;m the change is {sgn((a_['F'] / b_['F'] - 1) * 100, 0)}.</p>", 1)
    if pc and sp and sh:
        txt += (f"<p>The plate stiffness depends on the discretization: for the solid insert the two hexahedral models give "
                f"{1 / sh[0]['delta_mm']:.0f} and {sp[0]['F'] / sp[0]['delta_mm']:.0f}&#8202;N/mm and quadratic tetrahedra "
                f"{pc[0]['k_N_per_mm']:.0f} and {pc[-1]['k_N_per_mm']:.0f}&#8202;N/mm on two meshes, while under the cylinder the hexahedral models "
                f"agree within 3&#8202;%. This page therefore fits the modulus and states the bench prediction under the cylinder.</p>")
    tab = table(["Check", "This work", "Reference", "Difference"], rows,
                "Checks of the finite-element models (<code>validate.jsonl</code>, <code>gpu_tol.jsonl</code>, <code>plate_check.jsonl</code>, "
                "<code>gpu_nl_runs.jsonl</code>, <code>polyfem_runs.jsonl</code>). Bonded layer: Dimitriadis et al. 2002 for a sphere on a layer "
                "bonded to a rigid base.", text_cols=(0, 1, 2))
    return txt + tab


def _span(vals, fmt, unit):
    """'a&#8211;b unit' over vals, or 'a unit' when the ends print alike."""
    lo, hi = fmt.format(min(vals)), fmt.format(max(vals))
    return (lo if lo == hi else f"{lo}&#8211;{hi}") + unit


def cost_section(X: Data):
    pc = sorted(X.pc, key=lambda r: r["n_dof"])
    dof = {r["tag"]: r["n_dof"] for r in X.gpu_all if "n_dof" in r}
    nh = [r for r in X.nl if r.get("case") == "tip" and r["material"] == "neohookean" and r["tag"] in dof]
    nh_row = (["Tip, resolved, Neo-Hookean with contact", _span([dof[r["tag"]] / 1e6 for r in nh], "{:.1f}", "&#8202;M"),
               _span([r["t_s"] / 60 for r in nh], "{:.0f}", "&#8202;min per approach"),
               f"{max(r['gpu_mem_GB'] for r in nh):.0f}&#8202;GB GPU", "1 GPU"] if nh else
              ["Tip, resolved, Neo-Hookean with contact", "&#8211;", "&#8211;", "&#8211;", "1 GPU"])
    f64 = any(r.get("tangent") == "float64" for r in nh)
    rows = [
        ["Periodic cell, voxel hex8, AMG-CG", "18k&#8211;873k", "1.5&#8211;148&#8202;s", "~1&#8202;GB per 200k DOF", "4 cores"],
        ["Tip, homogenized core, CHOLMOD face compliance", "217k", "~10&#8202;min per print (15 loads, 3 cycles)", "3.7&#8211;4.4&#8202;GB", "4 cores"],
        ["Tip, resolved at 0.1&#8202;mm, matrix-free CG", "5&#8211;8&#8202;M", "60&#8211;140&#8202;s per approach", "2&#8211;3&#8202;GB GPU, 1.1&#8202;GB host",
         "1 GPU"],
        nh_row,
        ["Solid insert, quadratic tetrahedra, plate", f"{pc[0]['n_dof'] / 1e3:.0f}k&#8211;{pc[-1]['n_dof'] / 1e3:.0f}k" if pc else "&#8211;",
         f"{pc[0]['t_solve_s']:.0f}&#8211;{pc[-1]['t_solve_s']:.0f}&#8202;s" if pc else "&#8211;",
         f"{pc[0]['rss_GB']:.1f}&#8211;{pc[-1]['rss_GB']:.1f}&#8202;GB" if pc else "&#8211;", "2 cores"],
        ["PolyFEM, IPC contact, sphere on a block", "57k", "7&#8211;14&#8202;min", "1.1&#8202;GB", "2 cores"],
    ]
    return table(["Level", "DOF", "Wall time", "Memory", "Cores"], rows,
                 "Cost of each level on the workstation (RTX 4070 Ti SUPER, 20 cores). A resolved tip at 0.05&#8202;mm (~18&#8202;M elements, "
                 "~25&#8202;GB) would need a larger GPU such as DeltaAI&#8217;s GH200."
                 + (" The eFlesh lattice&#8217;s Neo-Hookean solve stores the element tangents in float64: with float32 tangents Newton "
                    "diverges on its struts." if f64 else ""), text_cols=(0, 2, 3, 4))


def limitations(X: Data):
    ef = X.nl_row("eflesh_c8" + A.SUFFIX, "neohookean", 1.0)
    nl = "the 5&#8202;% tip and the eFlesh lattice were" if ef else "the 5&#8202;% tip was"
    # the float64 attempt of 2026-10-10 wrote only its linear row; its Newton history is in logs/20261009-tpu_tip_fem/gpu_queue11.log
    fail = ("" if ef or not X.nl_row("eflesh_c8" + A.SUFFIX, "linear", 1.0) else
            " The Neo-Hookean solve of the eFlesh lattice at an approach of 1.0&#8202;mm did not converge: in nine Newton steps from the linear "
            "solution the residual force stayed at 1.4&#8211;1.7 times the reaction force (vector norms), and every conjugate-gradient solve "
            "stopped at its limit of 60,000 iterations.")
    return ("<ul class='open'>"
            f"<li>The tip models are linear and small-strain; only {nl} solved with large deformation.{fail}</li>"
            "<li>\\(E_s\\) comes from datasheets (26&#8211;67&#8202;MPa for 95A) and \\(\\nu\\) = 0.45 is assumed; every stiffness and \\(E^*\\) scales "
            "with \\(E_s\\).</li>"
            "<li>The printed geometry (0.45&#8202;mm beads, 0.2&#8202;mm layers, two walls, four skin layers, fused beads) is modelled from the slicer&#8217;s "
            "settings without its toolpaths, and the print settings of the current tips are not recorded.</li>"
            "<li>Contact is frictionless in the normal solves, and no printed tip has been measured.</li>"
            "<li>The magnet is modelled as bonded to its pocket and as a point dipole of remanence 1.45&#8202;T; the dipole is within a "
            "few percent at the 12&#8211;17&#8202;mm from magnet to chip. The noise is the datasheet&#8217;s; the chip&#8217;s alignment, the "
            "earth&#8217;s field and the other fingers&#8217; magnets are left out.</li>"
            "<li>The whole tips are held on their back and mount faces, as on the current mount; another attachment changes the "
            "shear travel along the finger most.</li></ul>")


def lede(X: Data):
    import tpu_tip_print_plan as PP
    T = X.T
    M = PP.rows()
    g0, g3 = M.get(("w95_20", "none", None)), M.get(("w95_20", PP.PLAN_MAG, PP.PLAN_DEPTH))
    sc = [M[(k, PP.PLAN_MAG, PP.PLAN_DEPTH)] for k in PP.PRINT_COL if (k, PP.PLAN_MAG, PP.PLAN_DEPTH) in M]
    if not (g0 and g3 and len(sc) == len(PP.PRINT_COL)) or "g20_across" not in T:
        return "Runs in progress."
    er = [T[n]["E_best_max_err_pct"] for n in T]
    trav = [-r["per_force"]["1"]["magnet_t_um"][0] for r in sc]
    sig = [PP.field(r, PP.GAPS[0])["sigF_normal_N"] for r in sc]
    Es = sorted(r["E_best_MPa"] for r in sc)
    ae = [-X.arm_err(r["E_best_MPa"], r["per_force"]["3"]["arm_mm"]) for r in sc + [g0]
          if X.arm_err(r["E_best_MPa"], r["per_force"]["3"]["arm_mm"]) is not None]
    return ("Five whole tips are ready to print (Table&#160;[[tab:plan]]): 95A at 20&#8202;% gyroid as the reference, 85A at 50&#8202;%, the "
            "print whose twist arm the pads match best, the softer 85A at 30&#8202;% and 95A at 10&#8202;%, each with a 4.76 &#215; 1.59&#8202;mm "
            f"N52 magnet {PP.PLAN_DEPTH:g}&#8202;mm under the contact point, and the reference without a magnet. Each maps to one modulus of the sphere "
            f"pads, {f1(Es[0])}&#8211;{f1(Es[-1])}&#8202;MPa for the magnet tips; the magnet stiffens the face above it, from "
            f"{f1(g0['E_best_MPa'])} to {f1(g3['E_best_MPa'])}&#8202;MPa on the reference. Before measurement a tip&#8217;s stiffness is "
            "known within the factor of 2.6 between 95A datasheets and its friction not at all; an indentation and a pull test fix "
            f"both and leave three model errors: {f0(min(er))}&#8211;{f0(max(er))}&#8202;% in approach over 0.5&#8211;3&#8202;N, a pad twist arm "
            f"{f0(min(ae))}&#8211;{f0(max(ae))}&#8202;% shorter than the tip&#8217;s, and the tip&#8217;s tangential compliance, which no pad model "
            "represents. Under "
            f"1&#8202;N the magnet moves {f0(min(trav))}&#8211;{f0(max(trav))}&#8202;&#181;m, which an MLX90393 1&#8202;mm behind the tip reads "
            f"with {f2(min(sig))}&#8211;{f2(max(sig))}&#8202;N of noise per 141&#8202;Hz sample.")


def open_items(X: Data):
    import tpu_tip_print_plan as PP
    M = PP.rows()
    g = M.get(("w95_20", PP.PLAN_MAG, PP.PLAN_DEPTH)) or {}
    pf = g.get("per_force", {})
    f = PP.field(g, PP.GAPS[0]) if g else {}
    pa = X.pads_arm(g["E_best_MPa"]) if g else None
    items = [
        ("Indentation of each print.", "Press the screwdriver shaft on the palmar face at the contact point to 0.5, 1 and 3&#8202;N and "
         "record force against approach. For the 95A 20&#8202;% magnet tip at \\(E_s\\) = 30&#8202;MPa the model predicts "
         f"{f1(pf.get('0.5', {}).get('delta_um'))}, {f1(pf.get('1', {}).get('delta_um'))} and {f1(pf.get('3', {}).get('delta_um'))}&#8202;&#181;m; "
         "<code>scripts/tpu_tip_calibrate.py fit</code> turns the measured approaches into \\(E_s\\) and \\(E^*\\). A force exponent near "
         "1.5 instead of 1.07 would falsify the line-contact result."),
        ("Pull and twist between two printed tips.", f"The pull gives \\(\\mu\\); an onset arm near {f2(pf.get('3', {}).get('arm_mm'))}&#8202;mm "
         f"at 3&#8202;N confirms the load on the walls, near {f2(pa)}&#8202;mm the pads."),
        ("Magnet signal under weights.", f"Load the face with 0.5&#8211;3&#8202;N at the contact point and record the field: the model "
         f"predicts {f1(f.get('S_normal_norm'))}&#8202;&#181;T/N at a chip 1&#8202;mm behind the 95A 20&#8202;% tip. Less than half of it "
         "means the magnet sits loose in its pocket; the model takes it as bonded."),
        ("Chip set into the back of the tip.", "Seat the MLX90393 in a pocket of the tip&#8217;s back face and repeat the weights: the "
         "noise should fall as Figure&#160;[[fig:magres]]b predicts until the static field reaches 4.8&#8202;mT."),
        ("Plate stiffness.", "Rerun the solid insert under the plate on voxels of 0.2, 0.1 and 0.05&#8202;mm "
         "(<code>scripts/tpu_tip_fem_gpu.py</code>; 0.05&#8202;mm on DeltaAI) and on quadratic tetrahedra "
         "(<code>scripts/tpu_tip_fem_plate_check.py</code>); the plate result is usable once the three agree within 5&#8202;%."),
    ]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def byline():
    return ("Rows <code>docs/experiments/20261009-tpu_tip_fem/</code> &#183; cells <code>scripts/tpu_tip_fem_cells.py</code>, tip "
            "<code>scripts/tpu_tip_fem_tip.py</code>, GPU <code>scripts/tpu_tip_fem_gpu.py</code>, films <code>scripts/tpu_tip_fem_film.py</code>, "
            "pads <code>scripts/tpu_tip_fem_bed.py</code>, tables <code>scripts/tpu_tip_fem_analysis.py</code> &#183; builder "
            "<code>scripts/tpu_tip_fem_page.py</code> &#183; print plan and magnet (2026-10-12) <code>scripts/tpu_tip_design.py</code>, "
            "<code>scripts/tpu_tip_magnet.py</code>, <code>scripts/tpu_tip_print_plan.py</code>, calibration "
            "<code>scripts/tpu_tip_calibrate.py</code>, calculator <code>scripts/tpu_tip_widget.py</code>, print files <code>docs/experiments/20261009-tpu_tip_fem/print/</code> &#183; "
            "hand-off <code>" + NOTE + "</code>")


def render_tex(t):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    return P.TEX_RE.sub(lambda m: next(svgs), t), len(items)


def main():
    X = Data()
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "GLOSSARY": glossary()}
    import tpu_tip_print_plan as PP
    H = sys.modules[__name__]
    v["LEDE"] = lede(X)
    v["PLAN"] = PP.plan_text(H, X) + PP.plan_section(H, X)
    v["MAGNET"] = PP.magnet_section(H, X)
    v["GAP"] = PP.gap_section(H, X)
    v["CALIB"] = PP.calib_section(H, X)
    import tpu_tip_widget as W
    v["CALC"] = W.section(H, X)
    v["CELLS"] = cells_section(X)
    v["TIP"] = tip_section(X)
    v["LAW"] = law_section(X)
    v["ARM"] = arm_section(X)
    v["RECIPES"] = recipes_section(X)
    v["CHECKS"] = checks_section(X)
    v["COST"] = cost_section(X)
    v["LIMITS"] = limitations(X)
    v["OPEN"] = open_items(X)
    v["FOOTER"] = ("<p>Rebuild: <code>.venv/bin/python scripts/tpu_tip_fem_page.py</code>. Rows: "
                   "<code>docs/experiments/20261009-tpu_tip_fem/*.jsonl</code>; films <code>docs/experiments/20261009-tpu_tip_fem/media/</code>.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    import re
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
