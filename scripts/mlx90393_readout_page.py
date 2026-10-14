#!/usr/bin/env python3
"""Page: MLX90393 readout of printed fingertips on the five-chip board, with one or four embedded magnets (2026-10-13).

Owner, 2026-10-13: "can we revisit the mlx driver code and whatever analysis exists and see if the drastically reduced
sensitivity in the gyroid is potentially still usable? i actually recall you provided the estimated noise floor for these
materials earlier. how did you make those computations? is that just the material analysis? please make that clear in the
corresponding docpage, and new analysis that also incorporates the mlx magnetometer sensor"; earlier the same day, four
magnets per finger, Amazon N52 magnets, pockets undersized to about 90 % of the diameter and 95 % of the height; later the
same day, "we are using the 5-magnetometer board from anyskin/visuoskin, reused in eflesh ... i believe the schematics are
available somewhere. this should improve your simulation. ill use the aplysiatech magnets, sure".

Data: scripts/mlx90393_model.py (datasheet tables, eFlesh firmware, the board, each tip's magnet motion),
scripts/tpu_tip_magnet_sensing.py (Cramer-Rao bounds on docs/experiments/20261013-magnet_layouts/fem.jsonl from
scripts/tpu_tip_magnet_layouts.py), scripts/tpu_tip_design.py (press-fit pockets).

    .venv/bin/python scripts/mlx90393_readout_page.py
"""
from __future__ import annotations

import math
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import mlx90393_model as MLX  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import tpu_tip_design as TD  # noqa: E402
import tpu_tip_magnet_sensing as SEN  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261013-magnet_layouts")
OUT = os.path.join(D, "20261013-mlx90393_fingertip_readout.html")
TPL = os.path.join(ROOT, "scripts/mlx90393_readout_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
TIP_PAGE = "../20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html"
EF_PAGE = "../20261013-eflesh_printability/20261013-eflesh_lattice_printability.html"
BOARD_URL = "https://github.com/raunaqbhirangi/reskin_sensor/tree/main/circuits"
FIG, TAB, REFS = [0], [0], {}
HS = "&#8202;"
EXTRA_CSS = """
:root{--s-x:#2a78d6;--s-z:#eb6834;--s-sh:#1baf7a;--s-ref:#48545D;--s-pads:#8a4fbf;--s-res:#c2185b}
.tw th{white-space:normal !important;vertical-align:bottom}
.tw td:first-child{min-width:10em}
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
"""
MAIN = "high_gain_141Hz"
ONE, FOUR = "one_d4.76x1.59_d5", "four_d3.18x1.59_d5"


def figure(body, caption, legend=None, key=None):
    FIG[0] += 1
    if key:
        REFS[f"fig:{key}"] = FIG[0]
    lg = P._legend_html(legend) if legend else ""
    return f'<figure class="diagram">{body}{lg}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


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


def f(x, nd=2):
    return "&#8734;" if not math.isfinite(x) else f"{x:.{nd}f}"


def fn(x):
    """Force with two significant figures."""
    if not math.isfinite(x):
        return "&#8734;"
    return f"{x:.3f}" if x < 0.1 else f"{x:.2f}" if x < 10 else f"{x:.1f}"


class Data:
    def __init__(self):
        self.settings = {k: MLX.setting(k) for k in MLX.SETTINGS}
        self.tips = MLX.table()
        self.layouts = SEN.table()
        self.fem = SEN.rows()
        self.mount = MLX.mount_table()
        self.tip_ref_E = next(r["E_star"] for r in MLX.tips() if r["tip"] == "w95_20_model")

    def tip(self, tip, setting=MAIN, gap="touch"):
        return next(r for r in self.tips if r["tip"] == tip and r["setting"] == setting and r["gap"] == gap)

    def lay(self, name, chips="board", gap="touch"):
        return next((r for r in self.layouts if r["layout"] == name and r["chips"] == chips and r["gap"] == gap), None)

    def mnt(self, tip, depth, gap="touch"):
        return next(r for r in self.mount if r["tip"] == tip and r["depth"] == depth and r["gap"] == gap)


# ------------------------------------------------------------------------------------------ figures

def svg_noise(X: Data):
    W, H = 860, 380
    out = P._svg_open(W, H, "MLX90393 noise against conversion time with the settings marked")
    fx, fy = P._panel(out, 90, 40, 560, 270, (1.0, 210.0), (0.1, 10.0), [1, 2, 5, 10, 20, 50, 100, 200], [0.1, 0.2, 0.5, 1, 2, 5, 10],
                      "Conversion time of X, Y and Z (ms, log)", "RMS noise per sample (&#181;T, log)", logx=True, logy=True)
    for (osr, df), (nxy, nz) in sorted(MLX.NOISE_MG.items()):
        T = MLX.burst_ms(osr, df) - (67 + 192 + 100) / 1000 + 0.36
        P._marker(out, fx(T), fy(nxy / 10), "var(--s-x)", r=3.4)
        P._marker(out, fx(T), fy(nz / 10), "var(--s-z)", shape="square", r=3.2)
    # each label sits one row above the top of its own line, and the lines to its right start below it
    rows = {"page_until_now": 0, "eflesh_firmware": 1, MAIN: 2}
    for name, col in (("eflesh_firmware", "#b00020"), ("page_until_now", "var(--ink3)"), (MAIN, "var(--s-sh)")):
        T = X.settings[name]["burst_ms"]
        y = 56 + 18 * rows[name]
        out.append(f'<line x1="{fx(T):.1f}" x2="{fx(T):.1f}" y1="{y + 6}" y2="310" style="stroke:{col};stroke-dasharray:5 4;stroke-width:1.5"/>')
        lab = {"eflesh_firmware": "eFlesh firmware", "page_until_now": "Pages until now", MAIN: "141 Hz setting"}[name]
        out.append(f'<text x="{fx(T) + 4:.1f}" y="{y:.1f}" style="fill:{col}">{lab}</text>')
    out.append("</svg>")
    return "".join(out), [("X and Y axes", "var(--s-x)", False, "circle"), ("Z axis", "var(--s-z)", False, "square")]


def svg_board(X: Data):
    """(a) Plan of the tip's back face with the board; (b) section along the finger through U1."""
    W, H = 860, 500
    out = P._svg_open(W, H, "Plan of the tip's back face with the five-chip board, its chips, the magnets and each chip's in-plane "
                            "signal; section along the finger through the contact point with the magnet and the board")
    out.append('<defs><marker id="sigar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">'
               '<path d="M0,0 L10,5 L0,10 Z" style="fill:#b00020"/></marker></defs>')
    zc, o = TD.Z_C, MLX.BOARD_OUTLINE
    tip_fill, board_st = "fill:rgba(42,120,214,0.12);stroke:var(--ink);stroke-width:1.8", "fill:#cfe3c9;stroke:#3f7a46;stroke-width:1.4"
    # ---------------- (a) plan, y across (right), z along (up)
    s, cx, top, zmax = 11.5, 255.0, 46.0, 22.5
    px = lambda y: cx + y * s  # noqa: E731
    py = lambda z: top + (zmax - z) * s  # noqa: E731
    out.append(f'<text x="70" y="24" style="fill:var(--ink)">(a) Back face, seen from the board</text>')
    u0, u1 = o["u"]
    v0, v1 = o["v"]
    tu0, tu1 = o["tab_u"]
    tv0, _ = o["tab_v"]
    pts = [(u0, v1), (u1, v1), (u1, v0), (tu1, v0), (tu1, tv0), (tu0, tv0), (tu0, v0), (u0, v0)]
    out.append('<path d="M' + " L".join(f"{px(u):.1f},{py(zc + v):.1f}" for u, v in pts) + f' Z" style="{board_st}"/>')
    hu, hv, hr = MLX.BOARD_HOLE
    out.append(f'<circle cx="{px(hu):.1f}" cy="{py(zc + hv):.1f}" r="{hr * s:.1f}" style="fill:#fff;stroke:#3f7a46;stroke-width:1.2"/>')
    out.append(f'<rect x="{px(-TD.Y_HALF):.1f}" y="{py(TD.Z_END):.1f}" width="{2 * TD.Y_HALF * s:.1f}" height="{TD.Z_END * s:.1f}" '
               f'style="{tip_fill}"/>')
    S = np.array(X.tip("gyroid20")["S_uT_per_N"])
    for (name, (u, v)), Sk in zip(MLX.BOARD_CHIPS.items(), S):
        out.append(f'<rect x="{px(u - 1.5):.1f}" y="{py(zc + v + 1.5):.1f}" width="{3 * s:.1f}" height="{3 * s:.1f}" '
                   f'style="fill:#2b2f33"/>')
        e = np.array([Sk[1], Sk[2]])
        mag = float(np.linalg.norm(e))
        if mag > 1.0:
            e /= mag
            a0 = np.array([u, v]) + 1.8 * e
            a1 = a0 + 0.30 * mag * e
            out.append(f'<line x1="{px(a0[0]):.1f}" y1="{py(zc + a0[1]):.1f}" x2="{px(a1[0]):.1f}" y2="{py(zc + a1[1]):.1f}" '
                       'style="stroke:#b00020;stroke-width:2.4" marker-end="url(#sigar)"/>')
    out.append(f'<circle cx="{px(0):.1f}" cy="{py(zc):.1f}" r="{2.38 * s:.1f}" style="fill:rgba(178,58,58,0.30);stroke:#b23a3a;stroke-width:1.6"/>')
    for sy in (-1, 1):
        for sz in (-1, 1):
            out.append(f'<circle cx="{px(3.0 * sy):.1f}" cy="{py(zc + 3.5 * sz):.1f}" r="{1.59 * s:.1f}" '
                       'style="fill:none;stroke:#b23a3a;stroke-width:1.4;stroke-dasharray:4 3"/>')
    for name, (u, v) in MLX.BOARD_CHIPS.items():
        out.append(f'<text x="{px(u):.1f}" y="{py(zc + v) + 5:.1f}" text-anchor="middle" style="fill:#ffffff">{name}</text>')
    yt = py(zc + tv0 + 4.5)
    out.append(f'<text x="{px(tu1 + 0.6):.1f}" y="{yt:.1f}" style="fill:#3f7a46">Connector tab,</text>'
               f'<text x="{px(tu1 + 0.6):.1f}" y="{yt + 17:.1f}" style="fill:#3f7a46">toward the mount</text>')
    out.append(f'<text x="{px(0):.1f}" y="{py(TD.Z_END) - 8:.1f}" text-anchor="middle" style="fill:var(--ink2)">Tip end</text>')
    zb = -10.6
    for z in (0, 10, 20):
        out.append(f'<line x1="{px(-11.2):.1f}" x2="{px(-10.7):.1f}" y1="{py(z):.1f}" y2="{py(z):.1f}" style="stroke:var(--ink3)"/>'
                   f'<text x="{px(-11.5):.1f}" y="{py(z) + 4:.1f}" text-anchor="end" style="fill:var(--ink3)">{z}</text>')
    for y in (-10, 0, 10):
        out.append(f'<line x1="{px(y):.1f}" x2="{px(y):.1f}" y1="{py(zb):.1f}" y2="{py(zb) + 5:.1f}" style="stroke:var(--ink3)"/>'
                   f'<text x="{px(y):.1f}" y="{py(zb) + 21:.1f}" text-anchor="middle" style="fill:var(--ink3)">{y}</text>')
    out.append(f'<text x="{px(0):.1f}" y="{py(zb) + 41:.1f}" text-anchor="middle" style="fill:var(--ink2)">Across the finger, y (mm)</text>')
    out.append(f'<text transform="translate({px(-13.4):.1f},{py(6.5):.1f}) rotate(-90)" text-anchor="middle" style="fill:var(--ink2)">'
               'Along the finger, z (mm)</text>')
    # ---------------- (b) section through U1 along the finger: z right, x up (palmar face up)
    s2, ox, oy = 9.0, 520.0, 96.0
    qz = lambda z: ox + (z + 9.5) * s2  # noqa: E731
    qx = lambda x: oy + (TD.X_FACE - x) * s2  # noqa: E731
    out.append(f'<text x="{ox - 20:.1f}" y="24" style="fill:var(--ink)">(b) Section along the finger through U1</text>')
    path = [(TD.X_BACK, 0.0), (TD.X_FACE, 0.0), (TD.X_FACE, TD.Z_END - TD.R_END)]
    for i in range(1, 13):
        a = math.pi / 2 * i / 12
        path.append((TD.X_FACE - TD.R_END + TD.R_END * math.cos(a), TD.Z_END - TD.R_END + TD.R_END * math.sin(a)))
    path.append((TD.X_BACK, TD.Z_END))
    out.append('<path d="M' + " L".join(f"{qz(z):.1f},{qx(x):.1f}" for x, z in path) + f' Z" style="{tip_fill}"/>')
    _, mg = TD.pocket("d4.76x1.59", 5.0)
    out.append(f'<rect x="{qz(zc - mg["r"]):.1f}" y="{qx(mg["x_hi"]):.1f}" width="{2 * mg["r"] * s2:.1f}" '
               f'height="{(mg["x_hi"] - mg["x_lo"]) * s2:.1f}" style="fill:rgba(178,58,58,0.75);stroke:#b23a3a"/>')
    xb_top = TD.X_BACK - MLX.PKG_H_MM          # board's top face, the chips' tops against the back face
    out.append('<path d="M' + " L".join(f"{qz(zc + v):.1f},{qx(x):.1f}" for v, x in
                                         ((tv0, xb_top), (v1, xb_top), (v1, xb_top - MLX.BOARD_T_MM), (tv0, xb_top - MLX.BOARD_T_MM)))
               + f' Z" style="{board_st}"/>')
    for name in ("U4", "U1", "U2"):
        v = MLX.BOARD_CHIPS[name][1]
        out.append(f'<rect x="{qz(zc + v - 1.5):.1f}" y="{qx(TD.X_BACK):.1f}" width="{3 * s2:.1f}" height="{MLX.PKG_H_MM * s2:.1f}" '
                   'style="fill:#2b2f33"/>'
                   f'<circle cx="{qz(zc + v):.1f}" cy="{qx(TD.X_BACK - MLX.SPOT_MM):.1f}" r="2.6" style="fill:#f2c94c"/>'
                   f'<text x="{qz(zc + v):.1f}" y="{qx(xb_top - MLX.BOARD_T_MM) + 16:.1f}" text-anchor="middle" '
                   f'style="fill:var(--ink)">{name}</text>')
    out.append(f'<circle cx="{qz(zc):.1f}" cy="{qx(TD.X_FACE):.1f}" r="3.5" style="fill:#b00020"/>')
    xd = qz(-2.2)
    out.append(f'<line x1="{xd:.1f}" x2="{xd:.1f}" y1="{qx(TD.X_FACE):.1f}" y2="{qx(TD.X_BACK):.1f}" style="stroke:var(--ink2)"/>'
               f'<text x="{xd - 5:.1f}" y="{qx(0) + 5:.1f}" text-anchor="end" style="fill:var(--ink2)">17 mm</text>')
    yl = qx(TD.X_FACE) - 12
    out.append(f'<text x="{qz(zc):.1f}" y="{yl:.1f}" text-anchor="middle" style="fill:#b00020">Contact point</text>')
    out.append(f'<text x="{qz(zc):.1f}" y="{qx(mg["x_lo"]) + 20:.1f}" text-anchor="middle" style="fill:#b23a3a">Magnet, 5 mm deep</text>')
    out.append(f'<text x="{qz(-6.0):.1f}" y="{qx(xb_top - MLX.BOARD_T_MM) + 16:.1f}" text-anchor="middle" style="fill:#3f7a46">Board</text>')
    g = X.tip("gyroid20")
    gv = X.tip("gyroid20", gap="v17")
    lines = [f"Yellow: sensing spots, {MLX.SPOT_MM:.2f} mm behind", f"the back face. U1 reads {g['S_norm']:.1f} &#181;T/N;",
             f"five chips resolve {fn(g['sigF_board_bg_N'])} N.", f"Under the v17 plate ({MLX.GAPS['v17']:.2f} mm):",
             f"U1 reads {gv['S_norm']:.1f} &#181;T/N; five chips {fn(gv['sigF_board_bg_N'])} N."]
    for i, t in enumerate(lines):
        out.append(f'<text x="{ox - 20:.1f}" y="{qx(xb_top - MLX.BOARD_T_MM) + 62 + 21 * i:.1f}" style="fill:var(--ink2)">{t}</text>')
    out.append("</svg>")
    legend = [("Tip&#8217;s back face, 14.8 &#215; 21.4&#8202;mm", "rgba(42,120,214,0.6)", False, "square"),
              ("Board, 20 &#215; 29&#8202;mm FR-4", "#3f7a46", False, "square"), ("MLX90393, 3 &#215; 3&#8202;mm", "#2b2f33", False, "square"),
              ("Magnet 4.76 &#215; 1.59&#8202;mm", "#b23a3a", False, "circle"), ("Four 3.18&#8202;mm magnets", "#b23a3a", True, None),
              ("In-plane signal per newton", "#b00020", False, None)]
    return "".join(out), legend


def svg_tips(X: Data):
    W, H = 860, 400
    out = P._svg_open(W, H, "Force noise and force-equivalent disturbances of four tips on the five-chip board")
    tips = [("w95_20_model", ("Gyroid 20 %,", "earlier model")), ("gyroid20", ("Gyroid 20 %,", "as printed")),
            ("eflesh_top", ("eFlesh,", "palmar skin")), ("eflesh_open", ("eFlesh,", "open"))]
    fx = lambda i: 150 + i * 170  # noqa: E731
    lo, hi = 0.003, 30.0
    fy = lambda v: 40 + 290 * (math.log10(hi) - math.log10(v)) / (math.log10(hi) - math.log10(lo))  # noqa: E731
    for v in (0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30):
        out.append(f'<line x1="90" x2="820" y1="{fy(v):.1f}" y2="{fy(v):.1f}" style="stroke:var(--rule2)"/>'
                   f'<text x="82" y="{fy(v) + 4:.1f}" text-anchor="end" style="fill:var(--ink3)">{v:g}</text>')
    out.append('<text x="90" y="26" style="fill:var(--ink2)">Force (N, log)</text>')
    keys = [("sigF_N", "var(--ink3)", "circle", "Noise, U1 alone"), ("sigF_board_bg_N", "var(--s-x)", "circle", "Noise, five chips"),
            ("drift10C_board_N", "var(--s-sh)", "square", "Offset drift, 10&#8202;&#176;C"),
            ("earth_N", "var(--s-z)", "diamond", "Earth&#8217;s field, U1 alone"),
            ("neighbour_board_N", "var(--s-res)", "cross", "Facing magnet, 25&#8594;30&#8202;mm")]
    for i, (k, lbl) in enumerate(tips):
        r = X.tip(k)
        for j, (q, col, shp, _) in enumerate(keys):
            P._marker(out, fx(i) - 48 + 24 * j, fy(max(lo, r[q])), col, shape=shp, r=6)
        out.append(f'<text x="{fx(i):.1f}" y="356" text-anchor="middle" style="fill:var(--ink)">{lbl[0]}</text>'
                   f'<text x="{fx(i):.1f}" y="376" text-anchor="middle" style="fill:var(--ink)">{lbl[1]}</text>')
    out.append("</svg>")
    return "".join(out), [(lbl, col, False, shp) for _, col, shp, lbl in keys]


# ------------------------------------------------------------------------------------------ sections

GLOSSARY = [
    ("Magnet travel", "Displacement of the magnet&#8217;s centre toward the back face per newton of normal load at the contact point, in "
     "&#181;m/N, from finite elements of the tip (material and structure)."),
    ("Five-chip board", "ReSkin&#8217;s 5X board, with which AnySkin and eFlesh read their sensors: five MLX90393 in a plus on rigid FR-4 "
     "0.6&#8202;mm thick, U1 at the centre and U2&#8211;U5 7.00&#8202;mm from it, 20 &#215; 29&#8202;mm with its connector tab."),
    ("Sensing gap", "Distance from the tip&#8217;s back face to the plane of the chips&#8217; sensing spots, in mm: 0.42 with the chips "
     "against the back face (the spot&#8217;s depth in the package), 2.92 under the 2.5&#8202;mm of solid TPU of the v17 finger&#8217;s "
     "plate."),
    ("Signal", "Change of the magnetic field at a chip per newton, in &#181;T/N: the magnet&#8217;s travel and rotation turned into a "
     "field change by a point dipole of its moment (N52, remanence 1.45&#8202;T)."),
    ("Force noise", "Standard deviation of the normal force estimated from one sample of the chips, in N: weighted least squares on the "
     "stacked readings, which reaches the Cram&#233;r&#8211;Rao bound of this linear model. Five chips: all of the board&#8217;s, with a "
     "uniform field (the Earth&#8217;s) estimated alongside."),
    ("Force per count", "The load that changes U1&#8217;s normal-axis reading by one least significant bit, in N."),
    ("Force-equivalent disturbance", "A field change that is not the magnet&#8217;s, read through the same estimate: what the tip would "
     "report as force. Earth&#8217;s field: 50&#8202;&#181;T in its worst direction, the change of a reading when the hand turns over; "
     "the five-chip estimate removes it. Offset drift: the datasheet&#8217;s &#177;1000-count band from 25&#8202;&#176;C over "
     "125&#8202;&#176;C, spread linearly, per 10&#8202;&#176;C (19&#8202;&#181;T on Z, 12&#8202;&#181;T on X and Y at the highest gain), "
     "independent on every axis of every chip, as an RMS. Facing magnet: an identical magnet in an opposing fingertip, on the normal "
     "through U1, moving from 25 to 30&#8202;mm in front of the sensing plane."),
    ("Bias", "The force a one-position estimate reads, over the true force, when the contact lies elsewhere on the face: "
     "\\(\\hat F/F = S(y,z)\\cdot S_0/|S_0|^2\\), with \\(S_0\\) the signal at the contact point."),
    ("Settings", "GAIN_SEL (analog gain, 0&#8211;7), RES (output resolution, 0&#8211;3), OSR and DIG_FILT (oversampling and digital "
     "filter, which set the conversion time) of the MLX90393; HALLCONF 0xC throughout."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def chain_section(X: Data):
    old, new = X.settings["page_until_now"], X.settings[MAIN]
    return ("<p>Every force noise these pages have given comes from three models in a row, with no measurement:</p><ol>"
            "<li>Finite elements of the tip give the magnet&#8217;s travel and rotation per newton. This is the material and structure "
            "analysis: the gyroid&#8217;s or the lattice&#8217;s stiffness, the walls and skins, the magnet as a stiff inclusion "
            f"(<a href=\"{TIP_PAGE}#magnet\">fingertip page</a>, homogenized core; <a href=\"{EF_PAGE}#fem\">eFlesh page</a>, solved from "
            "each tip&#8217;s G-code).</li>"
            "<li>A point dipole of the magnet&#8217;s moment turns that motion into a field change at each chip of the five-chip board, "
            "whose sensing spots lie 0.42&#8202;mm behind the back face (<a href=\"#board\">board section</a>). Until 2026-10-13 this step "
            "used one chip 1&#8202;mm behind the back face, under the magnet.</li>"
            "<li>The MLX90393 datasheet gives each chip&#8217;s noise per sample; the force noise is the standard deviation of the force "
            "estimated from the chips&#8217; readings.</li></ol>"
            "<p>Until 2026-10-13 step 3 took 2.3&#8202;&#181;T on the chip&#8217;s in-plane axes and 3.5&#8202;&#181;T normal to it as the "
            "noise of a 100&#8202;Hz sample. The datasheet&#8217;s Figures 2 and 3 plot the noise against the conversion time of all three "
            f"axes, and those values belong to a 2.6&#8202;ms conversion, a burst rate of about {old['rate_Hz']:.0f}&#8202;Hz. At "
            f"{new['rate_Hz']:.0f}&#8202;Hz (GAIN_SEL 7, RES 0, OSR 0, DIG_FILT 5) the noise is {new['noise_xy_uT']:.2f} and "
            f"{new['noise_z_uT']:.2f}&#8202;&#181;T, so every force noise quoted before is twice the chip&#8217;s; both pages now use the "
            "141&#8202;Hz values. Steps 1&#8211;3 leave out what limits a stiff tip in use: offset drift, the Earth&#8217;s field, other "
            "magnets, and the firmware&#8217;s resolution. The sections below add them.</p>")


def chip_section(X: Data):
    svg, leg = svg_noise(X)
    rows = []
    for k in ("eflesh_firmware", "page_until_now", MAIN, "high_gain_75Hz"):
        s = X.settings[k]
        rows.append([s["label"], f"{s['burst_ms']:.1f}", f"{s['rate_Hz']:.0f}", f"{s['noise_xy_uT']:.2f} / {s['noise_z_uT']:.2f}",
                     f"{s['lsb_xy_uT']:.3f} / {s['lsb_z_uT']:.3f}", f"{s['range_xy_mT']:.1f} / {s['range_z_mT']:.1f}"])
    ef = X.settings["eflesh_firmware"]
    g = X.tip("gyroid20")
    txt = (f"<p>eFlesh&#8217;s firmware (<code>~/wxie_workspace/eFlesh/arduino/5X_eflesh_stream/5X_eflesh_stream.ino</code> with its "
           "arduino-MLX90393 library) sets HALLCONF 0xC, GAIN_SEL 1, RES 2 on all three axes, OSR 0 and DIG_FILT 4, turns temperature "
           "compensation off, and reads the board&#8217;s five chips in burst mode (T, X, Y, Z) over 400&#8202;kHz I&#178;C. That is a "
           f"wide-range setting for eFlesh&#8217;s 9.5&#8202;mm magnets close to the board: one count is {ef['lsb_xy_uT']:.2f}&#8202;&#181;T "
           f"on X and Y and {ef['lsb_z_uT']:.2f}&#8202;&#181;T on Z, coarser than the chip&#8217;s own noise at that conversion time, and a "
           f"burst takes {ef['burst_ms']:.1f}&#8202;ms ({ef['rate_Hz']:.0f}&#8202;Hz). A tip with one 4.76&#8202;mm magnet 5&#8202;mm deep puts "
           f"{g['B0_uT'] / 1e3:.2f}&#8202;mT on U1 with the chips against the back face, inside the &#177;4.8&#8202;mT the datasheet "
           "guarantees at the highest gain (Table&#160;[[tab:chip]]), so GAIN_SEL 7 and RES 0 cost nothing and make one count "
           "0.24&#8202;&#181;T on Z.</p>"
           "<p>The datasheet gives noise at GAIN_SEL 7 only. At GAIN_SEL 1 this model keeps that field noise and adds the counting step "
           "(one count over \\(\\sqrt{12}\\)), so the eFlesh firmware&#8217;s noise here is a lower bound.</p>")
    tab = table(["Setting", "Burst<br>(ms)", "Rate<br>(Hz)", "Noise X&#8202;Y / Z<br>(&#181;T)", "One count X&#8202;Y / Z<br>(&#181;T)",
                 "Range X&#8202;Y / Z<br>(mT)"], rows,
                "MLX90393 settings from the datasheet: burst of T, X, Y and Z without delay (section 12), noise from Figures 2 and 3 with the "
                "counting step added, one count from Table 17, range as 2<sup>15</sup> counts up to the 50&#8202;mT saturation onset.",
                text_cols=(0,), key="chip")
    return txt + tab + figure(svg, "RMS noise of one sample against the conversion time of X, Y and Z, read off the datasheet&#8217;s Figures "
                                   "2 and 3 (GAIN_SEL 7, all OSR and DIG_FILT values), with the three settings of Table&#160;[[tab:chip]].",
                              leg, key="noise")


def board_section(X: Data):
    svg, leg = svg_board(X)
    g, gv = X.tip("gyroid20"), X.tip("gyroid20", gap="v17")
    S = np.array(g["S_uT_per_N"])
    out_n = float(np.mean(S[1:, 0]))
    out_t = float(np.mean(np.hypot(S[1:, 1], S[1:, 2])))
    m15, m5 = X.mnt("w95_20", 1.5), X.mnt("w95_20", 5.0)
    rows = []
    lbl = {("gyroid20", "touch"): "Gyroid 20&#8202;%, as printed; chips against the back face",
           ("gyroid20", "v17"): "Gyroid 20&#8202;%, as printed; under the v17 plate",
           ("w95_20", "touch"): "Gyroid 20&#8202;%, earlier model; chips against the back face",
           ("w95_20", "v17"): "Gyroid 20&#8202;%, earlier model; under the v17 plate"}
    for r in X.mount:
        rows.append([lbl[(r["tip"], r["gap"])], f"{r['depth']:g}", f"{r['gap_mm']:.2f}", f"{r['travel_um']:.1f}", f"{r['S_U1']:.1f}",
                     f"{r['B0_U1_mT']:.2f}", fn(r["sigF_U1"]), fn(r["sigF_board_bg"]), f"{r['E_star']:.1f}"])
    txt = ("<p>eFlesh reads its sensors with ReSkin&#8217;s 5X board, which AnySkin also uses; the ReSkin repository holds its Gerbers, "
           f"bill of materials and pick-and-place file (<a href=\"{BOARD_URL}\">reskin_sensor/circuits</a>, <code>5X_ReSkin.Zip</code>, "
           "exported 2021-03-10). Five MLX90393 (ELW-ABA-011) sit in a plus: U1 at the centre of a 20 &#215; 20&#8202;mm square and "
           "U2&#8211;U5 7.00&#8202;mm from it on the square&#8217;s two axes. A 10 &#215; 9&#8202;mm tab beyond U4 carries the four-pin "
           "connector, which makes the board 20 &#215; 29&#8202;mm; the README specifies rigid FR-4 0.6&#8202;mm thick. Each chip&#8217;s "
           "sensing spot lies at the centre of its 3 &#215; 3&#8202;mm package, 0.42 &#177; 0.10&#8202;mm under the top face (datasheet "
           "Figure 26). eFlesh&#8217;s firmware addresses the chips at 0x0C&#8211;0x10 on one 400&#8202;kHz I&#178;C bus and runs each in "
           "its own burst, so the board delivers every chip&#8217;s samples at the chip&#8217;s own rate.</p>"
           "<p>The model places U1 under the contact point, the plus along and across the finger, and the tab toward the mount "
           "(Figure&#160;[[fig:board]]). The board is 20&#8202;mm wide against the tip&#8217;s 14.8&#8202;mm and overhangs 2.6&#8202;mm on "
           "each side; U3 and U5 lie 0.4&#8202;mm inside the tip&#8217;s long edges. A newton at the contact point moves the magnet "
           f"{g['travel_um']:.1f}&#8202;&#181;m toward the board on the printed gyroid tip: U1 reads {S[0, 0]:.1f}&#8202;&#181;T on its "
           f"normal axis, and the four outer chips read {out_n:.1f}&#8202;&#181;T normal and {out_t:.1f}&#8202;&#181;T in-plane, pointing "
           "at U1.</p>"
           "<p>The owner&#8217;s v17 eFlesh finger (<code>~/wxie_workspace/fingers/merged_finger_v17.3mf</code>) held the board in a "
           "2&#8202;mm recess on the bed side of its plate, <code>finger_back_v5_2.stl</code>, printed solid in the lattice&#8217;s TPU. "
           f"{MLX.V17_COVER_MM:.1f}&#8202;mm of TPU separated the chips from the lattice, and the sensing spots lay "
           f"{MLX.GAPS['v17']:.2f}&#8202;mm behind it. Against the back face they lie {MLX.SPOT_MM:.2f}&#8202;mm behind it, and U1&#8217;s "
           f"signal is {g['S_norm'] / gv['S_norm']:.1f} times larger (Table&#160;[[tab:mount]]). A plate whose recess opens toward the "
           "tip, or the board bonded to the back face, halves every force noise and force-equivalent disturbance on this page.</p>"
           "<p>The earlier model&#8217;s runs at three depths show the board favouring a deeper magnet. Between 1.5 and 5&#8202;mm the "
           f"magnet&#8217;s travel falls from {m15['travel_um']:.1f} to {m5['travel_um']:.1f}&#8202;&#181;m/N, the field&#8217;s change at "
           f"the chips grows faster, and the five-chip noise falls from {fn(m15['sigF_board_bg'])} to {fn(m5['sigF_board_bg'])}&#8202;N. "
           f"At 5&#8202;mm the field at U1 is {m5['B0_U1_mT']:.2f}&#8202;mT, under the 4.8&#8202;mT the datasheet guarantees at the highest "
           "gain; a deeper magnet needs a lower gain, for which the datasheet gives no noise.</p>")
    tab = table(["Tip and mount", "Magnet<br>depth (mm)", "Sensing<br>gap (mm)", "Travel<br>(&#181;m/N)", "Signal at<br>U1 (&#181;T/N)",
                 "Field at U1<br>at rest (mT)", "Noise, U1<br>alone (N)", "Noise, five<br>chips (N)", "Pad modulus<br>(MPa)"], rows,
                "The five-chip board reading the 20&#8202;% gyroid tip with one 4.76 &#215; 1.59&#8202;mm magnet, against the sensing gap and "
                "the magnet&#8217;s depth below the palmar face; one 141&#8202;Hz sample. Five chips: the Earth&#8217;s field estimated "
                "alongside. Earlier model: the homogenized runs of the <a href=\"" + TIP_PAGE + "#magnet\">fingertip page</a>; as printed: the "
                "G-code run of the <a href=\"" + EF_PAGE + "#fem\">eFlesh page</a>.", key="mount")
    fig = figure(svg, "The five-chip board under the SR2 tip. (a) The tip&#8217;s back face seen from the board: U1 under the contact point, "
                      "the one-magnet design (solid) and the four-magnet grid (dashed) above it; arrows give each outer chip&#8217;s "
                      "in-plane signal for 1&#8202;N at the contact point on the printed gyroid tip (0.3&#8202;mm per &#181;T/N). (b) The "
                      "section through U1 with the chips against the back face.", leg, key="board")
    return txt + fig + tab


def tips_section(X: Data):
    svg, leg = svg_tips(X)
    tips = [("w95_20_model", "Gyroid 95A 20&#8202;%, earlier model"), ("w95_10_model", "Gyroid 95A 10&#8202;%, earlier model"),
            ("gyroid20", "Gyroid 95A 20&#8202;%, as printed"), ("eflesh_top", "Lattice of eFlesh with palmar skin, as printed"),
            ("eflesh_open", "Open lattice of eFlesh, as printed")]
    rows = []
    for k, lbl in tips:
        r, e = X.tip(k), X.tip(k, "eflesh_firmware")
        rows.append([lbl, f"{r['travel_um']:.0f}", f"{r['S_norm']:.0f}", fn(r["sigF_N"]), fn(r["sigF_board_bg_N"]), fn(e["lsb_force_N"]),
                     fn(r["drift10C_board_N"]), fn(r["earth_N"]), fn(r["neighbour_board_N"])])
    g, o = X.tip("gyroid20"), X.tip("eflesh_open")
    ge = X.tip("gyroid20", "eflesh_firmware")
    ef_max = max(X.tip(k)[q] for k in ("eflesh_open", "eflesh_top") for q in ("drift10C_board_N", "neighbour_board_N"))
    txt = ("<p>The board removes the Earth&#8217;s field and lowers the noise; drift and the other fingertips&#8217; magnets stay. On the "
           f"printed 20&#8202;% gyroid tip one sample&#8217;s normal-force noise is {fn(g['sigF_N'])}&#8202;N from U1 alone and "
           f"{fn(g['sigF_board_bg_N'])}&#8202;N from the five chips with the Earth&#8217;s field estimated alongside, against "
           f"{fn(o['sigF_board_bg_N'])}&#8202;N on open eFlesh (Table&#160;[[tab:tips]]). Without that estimate the Earth&#8217;s field "
           f"would read as {fn(g['earth_N'])}&#8202;N on U1 alone. Offset drift differs from chip to chip and stays: 10&#8202;&#176;C reads as "
           f"{fn(g['drift10C_board_N'])}&#8202;N on the gyroid against {fn(o['drift10C_board_N'])}&#8202;N on eFlesh. A facing "
           f"fingertip&#8217;s magnet moving from 25 to 30&#8202;mm reads as {fn(g['neighbour_board_N'])} against "
           f"{fn(o['neighbour_board_N'])}&#8202;N, because its field changes across the board mostly as a gradient, which the uniform "
           f"estimate keeps. Estimating the gradient as well brings it to {fn(g['neighbour_board_grad_N'])}&#8202;N and raises the noise to "
           f"{fn(g['sigF_board_grad_N'])}&#8202;N (Figure&#160;[[fig:tips]]). With eFlesh&#8217;s firmware setting one count on U1&#8217;s "
           f"normal axis is {fn(ge['lsb_force_N'])}&#8202;N on the gyroid tip.</p>"
           "<p>The gyroid tip is usable as a force sensor on the board if the readout zeroes whenever the finger is out of contact, "
           "corrects each chip&#8217;s offset with its burst&#8217;s temperature channel, and subtracts the other fingertips&#8217; magnets, "
           f"whose positions follow from the joint angles. eFlesh&#8217;s disturbances stay under {math.ceil(ef_max * 10) / 10:g}&#8202;N "
           "without these.</p>")
    tab = table(["Tip", "Travel<br>(&#181;m/N)", "Signal at<br>U1 (&#181;T/N)", "Noise, U1<br>alone (N)", "Noise, five<br>chips (N)",
                 "One count,<br>eFlesh firmware (N)", "Drift per<br>10&#8202;&#176;C (N)", "Earth&#8217;s field,<br>U1 alone (N)",
                 "Facing magnet<br>25&#8594;30&#8202;mm (N)"], rows,
                "Each tip with one 4.76 &#215; 1.59&#8202;mm N52 magnet 5&#8202;mm under the contact point, read by the five-chip board "
                "with its chips against the back face at the 141&#8202;Hz setting. Five chips: the Earth&#8217;s field estimated alongside; "
                "drift and facing magnet through that estimate. One count: eFlesh&#8217;s firmware setting, U1&#8217;s normal axis. Travel "
                f"from the <a href=\"{TIP_PAGE}#magnet\">fingertip page</a> (earlier model) and the <a href=\"{EF_PAGE}#fem\">eFlesh page</a> "
                "(as printed).", key="tips")
    return txt + tab + figure(svg, "Normal-force noise of one 141&#8202;Hz sample and three force-equivalent disturbances for four tips on the "
                                   "five-chip board, chips against the back face. The five-chip estimate removes the Earth&#8217;s field, so "
                                   "its marker shows U1 alone.", leg, key="tips")


LAY_LBL = {ONE: "One 4.76 &#215; 1.59&#8202;mm magnet", FOUR: "Four 3.18 &#215; 1.59&#8202;mm magnets",
           "four_d3x2_d5": "Four 3 &#215; 2&#8202;mm magnets", "four_d3x1_d5": "Four 3 &#215; 1&#8202;mm magnets",
           "four_d4.76x1.59_d5": "Four 4.76 &#215; 1.59&#8202;mm magnets"}


def layouts_section(X: Data):
    R = X.fem
    none = R.get("none")
    rows = []
    for r in X.layouts:
        c, m = r["centre"], r["median"]
        chips = "U1 alone" if r["chips"] == "centre" else "five chips" + (", v17 plate" if r["gap"] == "v17" else "")
        rows.append([f"{LAY_LBL.get(r['layout'], r['layout'])}, {chips}", fn(c["sigF_n"]), fn(c["sigF_n_bg"]),
                     fn(c["sigF_n_unk_bg"]), f(c["sig_z_mm_bg"], 1), f(c["sig_y_mm_bg"], 1), fn(m["sigF_n_unk_bg"]),
                     f"{r['bias_range'][0]:.2f}&#8211;{r['bias_range'][1]:.2f}"])
    u1, one, four = X.lay(ONE, "centre"), X.lay(ONE), X.lay(FOUR)
    ratio = (3.18 ** 2 * 1.59) / (4.76 ** 2 * 1.59)
    txt = ("<p>The board&#8217;s five chips resolve what one chip cannot. With one 4.76&#8202;mm magnet, U1 alone reads the contact point "
           f"to {fn(u1['centre']['sigF_n'])}&#8202;N with the position known, but its three readings cannot locate the contact or separate "
           f"the Earth&#8217;s field (&#8734; in Table&#160;[[tab:layouts]]), and a contact 3&#8211;4&#8202;mm away reads "
           f"{u1['bias_range'][0]:.2f} of its force. The five chips read {fn(one['centre']['sigF_n'])}&#8202;N with the position known, "
           f"{fn(one['centre']['sigF_n_bg'])}&#8202;N with the Earth&#8217;s field unknown and {fn(one['centre']['sigF_n_unk_bg'])}&#8202;N "
           f"with both unknown, locating the contact to {f(one['centre']['sig_z_mm_bg'], 1)}&#8202;mm along the finger and "
           f"{f(one['centre']['sig_y_mm_bg'], 1)}&#8202;mm across.</p>"
           f"<p>Four 3.18&#8202;mm magnets read the contact point worse, {fn(four['centre']['sigF_n_unk_bg'])}&#8202;N with both unknown, "
           f"because it lies between them and each carries {100 * ratio:.0f}&#8202;% of the large magnet&#8217;s moment. They spread the "
           "signal over the face: over the 15 load points the median noise with both unknown is "
           f"{fn(four['median']['sigF_n_unk_bg'])}&#8202;N against {fn(one['median']['sigF_n_unk_bg'])}&#8202;N for one magnet, whose bound "
           "grows quickly away from the contact point. Read as if the contact were at the contact point, they over-read contacts near a "
           f"magnet by up to {100 * (four['bias_range'][1] - 1):.0f}&#8202;%. Their four fields add to {four['B0_max_mT']:.2f}&#8202;mT at "
           "the chips, at the edge of the highest gain&#8217;s range.</p>")
    txt += table(["Layout and chips", "Force noise,<br>position known (N)", "Earth&#8217;s field<br>unknown (N)",
                 "Position and Earth&#8217;s<br>field unknown (N)", "Contact along<br>finger (mm)", "Across<br>(mm)",
                 "Median over 15<br>points, both unknown (N)", "Bias over<br>the face"], rows,
                "Cram&#233;r&#8211;Rao bounds for 1&#8202;N normal at the contact point, 141&#8202;Hz setting, chips against the back face "
                "except under the v17 plate (2.92&#8202;mm); &#8734;: the readings cannot separate the unknowns. 95A 20&#8202;% gyroid tip, "
                "homogenized core, magnets bonded, tops 5&#8202;mm deep; the four-magnet grid is 6&#8202;mm across and 7&#8202;mm along the "
                "finger. Contact along and across: standard deviation of the contact&#8217;s position with both unknown. Bias: range of the "
                "force a one-position estimate reads over 15 load points from &#8722;3 to 3&#8202;mm across and &#8722;4 to 4&#8202;mm along.",
                key="layouts")
    if none:
        E0 = none["E_best_MPa"]
        txt += (f"<p>The magnets stiffen the tip about equally: the pad modulus at the contact point is {E0:.1f}&#8202;MPa without a magnet, "
                f"{R[ONE]['E_best_MPa']:.1f} with one and {R[FOUR]['E_best_MPa']:.1f} with four (this coarser grid gives "
                f"{100 * (R[ONE]['E_best_MPa'] / X.tip_ref_E - 1):.0f}&#8202;% more than the fingertip page&#8217;s for the same tip). Four "
                "magnets spread the stiffening over the face instead of concentrating it under the contact point "
                "(Table&#160;[[tab:stiff]]).</p>")
        srows = []
        for name, lbl in ((ONE, "One 4.76&#8202;mm magnet"), (FOUR, "Four 3.18&#8202;mm magnets")):
            if name not in R:
                continue
            ratio_c = [p["normal_face_um_per_N"] / q["normal_face_um_per_N"] for p, q in zip(R[name]["patches"], none["patches"])]
            centre = [p["normal_face_um_per_N"] / q["normal_face_um_per_N"] for p, q in zip(R[name]["patches"], none["patches"])
                      if p["y"] == 0.0 and abs(p["z"] - TD.Z_C) < 1e-6][0]
            srows.append([lbl, f"{R[name]['E_best_MPa']:.1f}", f"{100 * (1 - centre):.0f}", f"{100 * (1 - max(ratio_c)):.0f}&#8211;"
                          f"{100 * (1 - min(ratio_c)):.0f}"])
        txt += table(["Layout", "Pad modulus<br>(MPa)", "Face compliance lost<br>at the contact point (%)",
                      "Lost over the 15<br>load points (%)"], srows,
                     "Stiffening by the magnets: the face&#8217;s own compliance under a 1&#8202;N patch (Gaussian, &#963; 0.6&#8202;mm) against "
                     "the tip without magnets, at the contact point and over the 15 points of the load grid.", key="stiff")
    return txt


AMAZON = [
    ("AplysiaTech", "3/16 &#215; 1/16&#8202;in (4.76 &#215; 1.59&#8202;mm)", 4.76, 1.59, "&#177;0.005&#8202;in", "400, $22.49", "https://www.amazon.com/dp/B0DSWGZ921"),
    ("AplysiaTech", "5 &#215; 2&#8202;mm", 5.0, 2.0, "&#177;0.1&#8202;mm", "300, $21.59", "https://www.amazon.com/dp/B0F4CJWFRN"),
    ("AplysiaTech", "5 &#215; 1&#8202;mm", 5.0, 1.0, "&#177;0.1&#8202;mm", "500, $22.79", "https://www.amazon.com/dp/B0F4CVLSYZ"),
    ("AplysiaTech", "4 &#215; 2&#8202;mm", 4.0, 2.0, "&#177;0.1&#8202;mm", "400, $21.59", "https://www.amazon.com/dp/B0F4MYLY85"),
    ("AplysiaTech", "3 &#215; 2&#8202;mm", 3.0, 2.0, "&#177;0.1&#8202;mm", "600, $22.79", "https://www.amazon.com/dp/B0G52HW8SY"),
    ("AplysiaTech", "3 &#215; 1&#8202;mm", 3.0, 1.0, "&#177;0.1&#8202;mm", "1000, $21.59", "https://www.amazon.com/dp/B0F4CD3GC1"),
    ("AplysiaTech", "1/8 &#215; 1/32&#8202;in (3.18 &#215; 0.79&#8202;mm)", 3.175, 0.794, "&#177;0.005&#8202;in", "600, $17.99", "https://www.amazon.com/dp/B0FPCSQJQ5"),
    ("Boom (BOOMStore)", "4 &#215; 1&#8202;mm", 4.0, 1.0, "&#177;0.1&#8202;mm", "100, $19.49", "https://www.amazon.com/dp/B0DM723TXB"),
]
KJ = [("D31-N52", "3/16 &#215; 1/16&#8202;in", 4.76, 1.59), ("D21B-N52", "1/8 &#215; 1/16&#8202;in", 3.175, 1.588),
      ("D201-N52", "1/8 &#215; 1/32&#8202;in", 3.175, 0.794), ("D301-N52", "3/16 &#215; 1/32&#8202;in", 4.76, 0.794)]


def buy_section(X: Data):
    rows = []
    for brand, size, d, t, tol, pack, url in AMAZON:
        V = math.pi * (d / 2) ** 2 * t
        rows.append([f"{brand}, {size}", tol, pack, f"{V / (math.pi * 2.38 ** 2 * 1.59):.2f}",
                     f"{TD.FIT_D * d:.2f} &#215; {TD.FIT_T * t:.2f}", f"<a href=\"{url}\">{url.split('/dp/')[1]}</a>"])
    tab = table(["Amazon listing (N52, axial, Ni-Cu-Ni)", "Tolerance", "Pack, price", "Moment over<br>4.76 &#215; 1.59",
                 "Pocket at 90&#8202;/&#8202;95&#8202;%<br>(mm)", "ASIN"], rows,
                "N52 disc magnets on amazon.com, pages read 2026-10-13 (logged out; prices as shown). AplysiaTech states the grade, axial "
                "magnetization, coating and tolerance for every size. Pocket: 90&#8202;% of the diameter and 95&#8202;% of the height, the "
                "owner&#8217;s press fit.", key="buy")
    kj = ", ".join(f"{p} ({s})" for p, s, d, t in KJ)
    m32 = (3.0 ** 2 * 2.0) / (3.175 ** 2 * 1.588)
    m31 = (3.0 ** 2 * 1.0) / (3.175 ** 2 * 1.588)
    s2, s1 = TD.print_sheet("d4.76x1.59", 5.0), TD.print_sheet("d4.76x1.59", 5.0, layer=0.1)
    proud = (s1["printed_pocket_print_heights_mm"][0] + 1.59) - s1["printed_pocket_print_heights_mm"][1]
    clear = s1["pause_before_layer_at_mm"] - (s1["printed_pocket_print_heights_mm"][0] + 1.59)
    txt = ("<p>The owner chose AplysiaTech (2026-10-13), which states the grade, magnetization, coating and tolerance for every size "
           "(Table&#160;[[tab:buy]]). The one-magnet tip uses its 3/16 &#215; 1/16&#8202;in disc, B0DSWGZ921. No Amazon listing offers "
           "1/8 &#215; 1/16&#8202;in in N52, the size of the four-magnet runs; AplysiaTech&#8217;s 3 &#215; 2&#8202;mm disc has "
           f"{m32:.2f} times its moment and the 3 &#215; 1&#8202;mm disc {m31:.2f} times, and K&amp;J Magnetics sells the size as "
           f"D21B-N52 (K&amp;J references: {kj}, all &#177;0.004&#8202;in).</p>"
           f"<p>The print files of the <a href=\"{TIP_PAGE}#plan\">fingertip page</a> now have the owner&#8217;s press fit: a pocket "
           f"{s2['pocket_diameter_mm']:.2f}&#8202;mm across (90&#8202;% of the diameter) and {s2['pocket_height_mm']:.2f}&#8202;mm deep "
           "(95&#8202;% of the thickness), its floor at the magnet&#8217;s bottom face. The slicer voids whole layers, those whose "
           f"mid-height lies inside the pocket, so at 0.2&#8202;mm layers the pocket prints as {s2['pocket_layers']} layers, "
           f"{s2['printed_pocket_height_mm']:.1f}&#8202;mm deep, the magnet&#8217;s top lies flush with it, and the 90&#8202;% diameter "
           f"holds the magnet. At 0.1&#8202;mm layers it prints as {s1['pocket_layers']} layers, {s1['printed_pocket_height_mm']:.1f}&#8202;mm, "
           f"and the magnet stands {proud:.2f}&#8202;mm proud, {clear:.2f}&#8202;mm under the next layer&#8217;s nozzle height. The owner&#8217;s "
           "eFlesh trials in <code>create_pouch.py</code> (2025-10-26) converged on a cavity of 7.56 &#215; 2.65&#8202;mm, since eFlesh&#8217;s "
           "pouch scales the listed diameter by 1.05 and the thickness by 0.945; <code>scripts/eflesh_build.py</code> now cuts its pouches "
           "to the 90&#8202;/&#8202;95&#8202;% fit as well.</p>")
    return txt + tab


def limitations():
    z = (TD.X_FACE - 5 - 1.59 / 2) - (TD.X_BACK - MLX.GAPS["touch"])
    dip = 2e-7 * 1.45 * math.pi * (2.38e-3) ** 2 * 1.59e-3 / (4e-7 * math.pi) / (z * 1e-3) ** 3 * 1e3
    disc = MLX.disc_axis_mT(z)
    items = [
        "The chip&#8217;s noise at a gain below 7 is not in the datasheet; the model keeps the highest gain&#8217;s field noise and adds the "
        "counting step.",
        "The offset drift is the datasheet&#8217;s band over the full temperature range spread linearly, independent on every axis of every "
        "chip; the chip&#8217;s drift near room temperature can be smaller, and its gradient is largest at 85&#8202;&#176;C.",
        "The board is rigid and fixed behind a fixed back face; the plate, adhesive or recess that holds it is not modelled. The chips "
        "U2&#8211;U5 are turned in 90&#8202;&#176; steps on the board, which relabels their axes without changing the bounds.",
        f"Magnets are point dipoles. U1 lies {z:.1f}&#8202;mm from the magnet&#8217;s centre, where the dipole&#8217;s on-axis field is "
        f"{100 * (dip / disc - 1):.0f}&#8202;% above a uniformly magnetized disc&#8217;s ({dip:.2f} against {disc:.2f}&#8202;mT).",
        "The four-magnet runs use the homogenized gyroid core on a coarser grid, 13&#8202;% stiffer than the fingertip page&#8217;s, with the "
        "contact as a Gaussian patch; the bounds hold for small loads, where the response is linear.",
        "Three four-magnet layouts have not run: 3 &#215; 2 and 3 &#215; 1&#8202;mm magnets (AplysiaTech sizes) and four 4.76&#8202;mm "
        "magnets. A single magnet deeper than 5&#8202;mm needs the layout grid refined below x&#8202;=&#8202;1&#8202;mm.",
    ]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def open_items(X: Data):
    g = X.tip("gyroid20")
    one = X.lay(ONE)
    items = [
        ("Bench noise and drift on the board.", "Stream the five chips at GAIN_SEL 7, RES 0, OSR 0, DIG_FILT 5 under a printed tip with "
         "the chips against its back face, 10&#8202;min unloaded, then turn the hand over. The noise model holds if each chip&#8217;s standard "
         "deviation is within 30&#8202;% of 1.15 and 1.75&#8202;&#181;T; the turn should move all five chips&#8217; readings by one uniform "
         f"field, which the five-chip estimate removes; U1 should read about {g['B0_uT'] / 1e3:.1f}&#8202;mT at rest."),
        ("Firmware setting for a stiff tip.", "Change <code>setGainSel(0x1)</code> and <code>setResolution(0x2, 0x2, 0x2)</code> in "
         "<code>5X_eflesh_stream.ino</code> to <code>0x7</code> and <code>0x0</code> and DIG_FILT to 5 for the gyroid tips; keep "
         "eFlesh&#8217;s setting for its large magnets."),
        ("A plate with the recess toward the tip.", "Draw the SR2 tip&#8217;s mount with the board&#8217;s recess opening toward the "
         "tip&#8217;s back face, so the chips touch it; the v17 plate&#8217;s 2.5&#8202;mm of TPU doubles every force on this page."),
        ("The AplysiaTech four-magnet layouts.", "<code>~/miniconda3/envs/femtip/bin/python scripts/tpu_tip_magnet_layouts.py run --layout L"
         "</code> for <code>four_d3x2_d5</code>, <code>four_d3x1_d5</code> and <code>four_d4.76x1.59_d5</code> (2&#8202;min and "
         "5.5&#8202;GB each), then this page. Four magnets are worth their stiffening if the median over the face with both unknown stays "
         f"under the one-magnet {fn(one['median']['sigF_n_unk_bg'])}&#8202;N while the contact point stays under 0.2&#8202;N."),
        ("A deeper magnet.", "Refine <code>GRID['x']</code> in <code>scripts/tpu_tip_magnet_layouts.py</code> to 0.3&#8202;mm from "
         "x&#8202;=&#8202;&#8722;2 to 7&#8202;mm, rerun <code>none</code> and <code>one_d4.76x1.59_d5</code>, and add magnets 6.5 and "
         "8&#8202;mm deep read at GAIN_SEL 5. Deeper is better if the five-chip noise at the contact point, with both unknown, falls under "
         f"the 5&#8202;mm value of {fn(one['centre']['sigF_n_unk_bg'])}&#8202;N."),
        ("Facing fingertips.", "Compute each chip&#8217;s field from the other two fingertips&#8217; magnets over the grasp-to-turn trajectory "
         "of the deployed hands, and the residual after subtracting it from the joint angles with 0.5&#8202;mm of position error."),
    ]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def lede(X: Data):
    g, gv, o = X.tip("gyroid20"), X.tip("gyroid20", gap="v17"), X.tip("eflesh_open")
    one, four = X.lay(ONE), X.lay(FOUR)
    return (f"eFlesh&#8217;s five-chip board reads one 141&#8202;Hz sample of the printed 20&#8202;% gyroid tip to "
            f"{fn(g['sigF_board_bg_N'])}&#8202;N of normal force with the Earth&#8217;s field estimated and removed, against "
            f"{fn(g['sigF_N'])}&#8202;N from the chip under the magnet alone and {fn(o['sigF_board_bg_N'])}&#8202;N on open eFlesh. Its chips "
            "must sit against the tip&#8217;s back face: under the 2.5&#8202;mm of TPU of the v17 finger&#8217;s plate the noise doubles to "
            f"{fn(gv['sigF_board_bg_N'])}&#8202;N. Offset drift ({fn(g['drift10C_board_N'])}&#8202;N per 10&#8202;&#176;C) and a facing "
            f"fingertip&#8217;s magnet ({fn(g['neighbour_board_N'])}&#8202;N) remain on the gyroid and need temperature correction and "
            "subtraction from the joint angles. One 4.76&#8202;mm magnet reads the contact point to "
            f"{fn(one['centre']['sigF_n_unk_bg'])}&#8202;N with its position and the Earth&#8217;s field unknown; four 3.18&#8202;mm magnets "
            f"read it to {fn(four['centre']['sigF_n_unk_bg'])}&#8202;N and the rest of the face better.")


def byline():
    return ("Chip and board model <code>scripts/mlx90393_model.py</code> (datasheet rev. 012 in <code>logs/20261013-mlx/</code>; board from "
            f"<a href=\"{BOARD_URL}\">reskin_sensor/circuits</a>) &#183; layouts <code>scripts/tpu_tip_magnet_layouts.py</code>, bounds "
            "<code>scripts/tpu_tip_magnet_sensing.py</code> &#183; builder <code>scripts/mlx90393_readout_page.py</code> &#183; rows "
            "<code>docs/experiments/20261013-magnet_layouts/fem.jsonl</code>")


def render_tex(t):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None) for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    return P.TEX_RE.sub(lambda m: next(svgs), t), len(items)


def main():
    X = Data()
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "GLOSSARY": glossary(), "LEDE": lede(X), "CHAIN": chain_section(X), "CHIP": chip_section(X),
         "BOARD": board_section(X), "TIPS": tips_section(X), "LAYOUTS": layouts_section(X), "BUY": buy_section(X),
         "LIMITS": limitations(), "OPEN": open_items(X),
         "FOOTER": "<p>Rebuild: <code>.venv/bin/python scripts/mlx90393_readout_page.py</code>.</p>"}
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
    os.makedirs(D, exist_ok=True)
    open(OUT, "w").write(t)
    print(f"formulas {n}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
