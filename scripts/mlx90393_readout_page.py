#!/usr/bin/env python3
"""Page: MLX90393 readout of printed fingertips with one or four embedded magnets (2026-10-13).

Owner, 2026-10-13: "can we revisit the mlx driver code and whatever analysis exists and see if the drastically reduced
sensitivity in the gyroid is potentially still usable? i actually recall you provided the estimated noise floor for these
materials earlier. how did you make those computations? is that just the material analysis? please make that clear in the
corresponding docpage, and new analysis that also incorporates the mlx magnetometer sensor"; and earlier the same day, four
magnets per finger, Amazon N52 magnets, pockets undersized to about 90 % of the diameter and 95 % of the height.

Data: scripts/mlx90393_model.py (datasheet tables, eFlesh firmware, each tip's magnet motion), scripts/tpu_tip_magnet_sensing.py
(Cramer-Rao bounds on docs/experiments/20261013-magnet_layouts/fem.jsonl from scripts/tpu_tip_magnet_layouts.py).

    .venv/bin/python scripts/mlx90393_readout_page.py
"""
from __future__ import annotations

import json
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
import tpu_tip_magnet_sensing as SEN  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261013-magnet_layouts")
OUT = os.path.join(D, "20261013-mlx90393_fingertip_readout.html")
TPL = os.path.join(ROOT, "scripts/mlx90393_readout_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
TIP_PAGE = "../20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html"
EF_PAGE = "../20261013-eflesh_printability/20261013-eflesh_lattice_printability.html"
FIG, TAB, REFS = [0], [0], {}
HS = "&#8202;"
EXTRA_CSS = """
:root{--s-x:#2a78d6;--s-z:#eb6834;--s-sh:#1baf7a;--s-ref:#48545D;--s-pads:#8a4fbf;--s-res:#c2185b}
.tw th{white-space:normal !important;vertical-align:bottom}
.tw td:first-child{min-width:10em}
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
"""
MAIN = "high_gain_141Hz"


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
        self.tip_ref_E = next(r["E_star"] for r in MLX.tips() if r["tip"] == "w95_20_model")

    def tip(self, tip, setting=MAIN):
        return next(r for r in self.tips if r["tip"] == tip and r["setting"] == setting)

    def lay(self, name, chips):
        return next((r for r in self.layouts if r["layout"] == name and r["chips"] == chips), None)


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
    for name, col in (("eflesh_firmware", "#b00020"), ("page_until_now", "var(--ink3)"), (MAIN, "var(--s-sh)")):
        s = X.settings[name]
        T = s["burst_ms"]
        out.append(f'<line x1="{fx(T):.1f}" x2="{fx(T):.1f}" y1="40" y2="310" style="stroke:{col};stroke-dasharray:5 4;stroke-width:1.5"/>')
        lab = {"eflesh_firmware": "eFlesh firmware", "page_until_now": "Pages until now", MAIN: "141 Hz setting"}[name]
        out.append(f'<text x="{fx(T) + 5:.1f}" y="{56 + 18 * list(X.settings).index(name):.1f}" style="fill:{col}">{lab}</text>')
    out.append("</svg>")
    return "".join(out), [("X and Y axes", "var(--s-x)", False, "circle"), ("Z axis", "var(--s-z)", False, "square")]


def svg_tips(X: Data):
    W, H = 860, 400
    out = P._svg_open(W, H, "Force noise and force-equivalent disturbances of four tips")
    tips = [("w95_20_model", ("Gyroid 20 %,", "earlier model")), ("gyroid20", ("Gyroid 20 %,", "as printed")),
            ("eflesh_top", ("eFlesh,", "palmar skin")), ("eflesh_open", ("eFlesh,", "open"))]
    fx = lambda i: 150 + i * 170  # noqa: E731
    lo, hi = 0.003, 30.0
    fy = lambda v: 40 + 290 * (math.log10(hi) - math.log10(v)) / (math.log10(hi) - math.log10(lo))  # noqa: E731
    for v in (0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30):
        out.append(f'<line x1="90" x2="820" y1="{fy(v):.1f}" y2="{fy(v):.1f}" style="stroke:var(--rule2)"/>'
                   f'<text x="82" y="{fy(v) + 4:.1f}" text-anchor="end" style="fill:var(--ink3)">{v:g}</text>')
    out.append('<text x="90" y="26" style="fill:var(--ink2)">Force (N, log)</text>')
    keys = [("sigF_N", "var(--s-x)", "circle", "Noise, one sample"), ("drift10C_N", "var(--s-sh)", "square", "Offset drift, 10&#8202;&#176;C"),
            ("earth_N", "var(--s-z)", "diamond", "Earth&#8217;s field, 50&#8202;&#181;T"),
            ("neighbour_N", "var(--s-res)", "cross", "Facing magnet, 25&#8594;30&#8202;mm")]
    for i, (k, lbl) in enumerate(tips):
        r = X.tip(k)
        for j, (q, col, shp, _) in enumerate(keys):
            P._marker(out, fx(i) - 36 + 24 * j, fy(max(lo, r[q])), col, shape=shp, r=6)
        out.append(f'<text x="{fx(i):.1f}" y="356" text-anchor="middle" style="fill:var(--ink)">{lbl[0]}</text>'
                   f'<text x="{fx(i):.1f}" y="376" text-anchor="middle" style="fill:var(--ink)">{lbl[1]}</text>')
    out.append("</svg>")
    return "".join(out), [(lbl, col, False, shp) for _, col, shp, lbl in keys]


# ------------------------------------------------------------------------------------------ sections

GLOSSARY = [
    ("Magnet travel", "Displacement of the magnet&#8217;s centre toward the back face per newton of normal load at the contact point, in "
     "&#181;m/N, from finite elements of the tip (material and structure)."),
    ("Signal", "Change of the magnetic field at the MLX90393 per newton, in &#181;T/N: the magnet&#8217;s travel and rotation turned into a "
     "field change by a point dipole of its moment (N52, remanence 1.45&#8202;T)."),
    ("Force noise", "The load whose signal equals one standard deviation of one sample&#8217;s noise on the chip&#8217;s three axes, in N; "
     "with several chips, the Cram&#233;r&#8211;Rao bound, the smallest standard deviation any unbiased estimate from those readings can "
     "reach."),
    ("Force per count", "The load that changes the chip&#8217;s normal-axis reading by one least significant bit, in N."),
    ("Force-equivalent disturbance", "A field change that is not the magnet&#8217;s, divided by the signal: what the tip would report as "
     "force. Earth&#8217;s field: 50&#8202;&#181;T, the change of a reading when the hand turns over. Offset drift: the datasheet&#8217;s "
     "&#177;1000-count band from 25&#8202;&#176;C over 125&#8202;&#176;C, spread linearly, per 10&#8202;&#176;C (19&#8202;&#181;T on Z at the "
     "highest gain). Facing magnet: the change of the on-axis field of an identical magnet in an opposing fingertip moving from 25 to "
     "30&#8202;mm."),
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
            "<li>A point dipole of the magnet&#8217;s moment turns that motion into a field change at the chip, 1&#8202;mm behind the back "
            "face under the magnet.</li>"
            "<li>The MLX90393 datasheet gives the chip&#8217;s noise per sample; the force noise is the load whose field change equals "
            "it.</li></ol>"
            "<p>Until 2026-10-13 step 3 took 2.3&#8202;&#181;T on the chip&#8217;s in-plane axes and 3.5&#8202;&#181;T normal to it as the "
            "noise of a 100&#8202;Hz sample. The datasheet&#8217;s Figures 2 and 3 plot the noise against the conversion time of all three "
            f"axes, and those values belong to a 2.6&#8202;ms conversion, a burst rate of about {old['rate_Hz']:.0f}&#8202;Hz. At {new['rate_Hz']:.0f}&#8202;Hz (GAIN_SEL 7, RES 0, OSR 0, DIG_FILT 5) the noise is "
            f"{new['noise_xy_uT']:.2f} and {new['noise_z_uT']:.2f}&#8202;&#181;T, so every force noise quoted before is twice the "
            "chip&#8217;s; both pages now use the 141&#8202;Hz values. Steps 1&#8211;3 leave out what limits a stiff tip in use: offset drift, "
            "the Earth&#8217;s field, other magnets, and the firmware&#8217;s resolution. The sections below add them.</p>")


def chip_section(X: Data):
    svg, leg = svg_noise(X)
    rows = []
    for k in ("eflesh_firmware", "page_until_now", MAIN, "high_gain_75Hz"):
        s = X.settings[k]
        rows.append([s["label"], f"{s['burst_ms']:.1f}", f"{s['rate_Hz']:.0f}", f"{s['noise_xy_uT']:.2f} / {s['noise_z_uT']:.2f}",
                     f"{s['lsb_xy_uT']:.3f} / {s['lsb_z_uT']:.3f}", f"{s['range_xy_mT']:.1f} / {s['range_z_mT']:.1f}"])
    ef = X.settings["eflesh_firmware"]
    txt = (f"<p>eFlesh&#8217;s firmware (<code>~/wxie_workspace/eFlesh/arduino/5X_eflesh_stream/5X_eflesh_stream.ino</code> with its "
           "arduino-MLX90393 library) sets HALLCONF 0xC, GAIN_SEL 1, RES 2 on all three axes, OSR 0 and DIG_FILT 4, turns temperature "
           "compensation off, and reads five chips in burst mode (T, X, Y, Z) over 400&#8202;kHz I&#178;C. That is a wide-range setting "
           f"for eFlesh&#8217;s 9.5&#8202;mm magnets close to the board: one count is {ef['lsb_xy_uT']:.2f}&#8202;&#181;T on X and Y and "
           f"{ef['lsb_z_uT']:.2f}&#8202;&#181;T on Z, coarser than the chip&#8217;s own noise at that conversion time, and a burst takes "
           f"{ef['burst_ms']:.1f}&#8202;ms ({ef['rate_Hz']:.0f}&#8202;Hz). A tip with one 4.76&#8202;mm magnet 5&#8202;mm deep puts "
           "3.6&#8202;mT on a chip 1&#8202;mm behind it, inside the highest gain&#8217;s range (Table&#160;[[tab:chip]]), so GAIN_SEL 7 and "
           "RES 0 cost nothing and make one count 0.24&#8202;&#181;T on Z.</p>"
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


def tips_section(X: Data):
    svg, leg = svg_tips(X)
    tips = [("w95_20_model", "Gyroid 95A 20&#8202;%, earlier model"), ("w95_10_model", "Gyroid 95A 10&#8202;%, earlier model"),
            ("gyroid20", "Gyroid 95A 20&#8202;%, as printed"), ("eflesh_top", "eFlesh, palmar skin, as printed"),
            ("eflesh_open", "eFlesh, open, as printed")]
    rows = []
    for k, lbl in tips:
        r, e = X.tip(k), X.tip(k, "eflesh_firmware")
        rows.append([lbl, f"{r['travel_um']:.0f}", f"{r['S_norm']:.0f}", fn(r["sigF_N"]), fn(e["lsb_force_N"]), fn(r["drift10C_N"]),
                     fn(r["earth_N"]), fn(r["neighbour_N"])])
    g, gm, o = X.tip("gyroid20"), X.tip("w95_20_model"), X.tip("eflesh_open")
    ge = X.tip("gyroid20", "eflesh_firmware")
    txt = (f"<p>Noise is not what limits the gyroid tip. At 141&#8202;Hz one sample&#8217;s normal-force noise is {fn(g['sigF_N'])}&#8202;N "
           f"for the 20&#8202;% tip solved from its G-code and {fn(gm['sigF_N'])}&#8202;N on the stiffer earlier model, against "
           f"{fn(o['sigF_N'])}&#8202;N for the open eFlesh lattice (Table&#160;[[tab:tips]]). Averaging to 50&#8202;Hz takes the gyroid to "
           f"{fn(g['sigF_50Hz_N'])}&#8202;N. What scales against it is everything else: a field change that is not the magnet&#8217;s reads "
           f"as force in proportion to one over the signal, and the gyroid&#8217;s signal is {o['S_norm'] / g['S_norm']:.0f} times smaller. "
           f"The Earth&#8217;s field over a turn of the hand reads as {fn(g['earth_N'])}&#8202;N on the printed gyroid against "
           f"{fn(o['earth_N'])}&#8202;N on eFlesh, 10&#8202;&#176;C of offset drift as {fn(g['drift10C_N'])} against {fn(o['drift10C_N'])}"
           f"&#8202;N, and a facing fingertip&#8217;s magnet moving from 25 to 30&#8202;mm as {fn(g['neighbour_N'])} against "
           f"{fn(o['neighbour_N'])}&#8202;N (Figure&#160;[[fig:tips]]). With eFlesh&#8217;s firmware setting one count on the normal axis is "
           f"{fn(ge['lsb_force_N'])}&#8202;N on the gyroid tip.</p>"
           "<p>The gyroid tip is usable as a force sensor if the readout removes those fields: a reference reading without the magnet&#8217;s "
           "signal (a second chip, or the four chips of the next section), a zero taken whenever the finger is out of contact, the "
           "burst&#8217;s temperature channel or the chip&#8217;s own compensation for drift, and the other fingertips&#8217; magnets "
           "subtracted from the joint angles, since every magnet&#8217;s position is known from the hand&#8217;s kinematics. Each of these "
           "is a calibration on the bench; none is needed to the same degree on eFlesh, whose disturbances stay under 1&#8202;N "
           "uncorrected.</p>")
    tab = table(["Tip", "Travel<br>(&#181;m/N)", "Signal<br>(&#181;T/N)", "Noise, one<br>sample (N)", "One count,<br>eFlesh firmware (N)",
                 "Drift per<br>10&#8202;&#176;C (N)", "Earth&#8217;s<br>field (N)", "Facing magnet<br>25&#8594;30&#8202;mm (N)"], rows,
                "Each tip with one 4.76 &#215; 1.59&#8202;mm N52 magnet 5&#8202;mm under the contact point and an MLX90393 1&#8202;mm behind "
                "the back face at the 141&#8202;Hz setting; the fifth column at eFlesh&#8217;s firmware setting. Travel from the "
                f"<a href=\"{TIP_PAGE}#magnet\">fingertip page</a> (earlier model) and the <a href=\"{EF_PAGE}#fem\">eFlesh page</a> "
                "(as printed).", key="tips")
    return txt + tab + figure(svg, "Normal-force noise of one 141&#8202;Hz sample and three force-equivalent disturbances for four tips.",
                              leg, key="tips")


def layouts_section(X: Data):
    R = X.fem
    none = R.get("none")
    rows = []
    lab = {("one_d4.76x1.59_d5", "under"): "One 4.76 &#215; 1.59&#8202;mm magnet, one chip",
           ("one_d4.76x1.59_d5", "grid4"): "One 4.76 &#215; 1.59&#8202;mm magnet, four chips",
           ("four_d3.18x1.59_d5", "under"): "Four 3.18 &#215; 1.59&#8202;mm magnets, a chip under each",
           ("four_d3x1_d5", "under"): "Four 3 &#215; 1&#8202;mm magnets, a chip under each",
           ("four_d3.18x1.59_d3", "under"): "Four 3.18 &#215; 1.59&#8202;mm magnets 3&#8202;mm deep",
           ("four_d4.76x1.59_d5", "under"): "Four 4.76 &#215; 1.59&#8202;mm magnets, a chip under each"}
    for r in X.layouts:
        c, m = r["centre"], r["median"]
        rows.append([lab.get((r["layout"], r["chips"]), r["layout"]), f"{r['E_best_MPa']:.1f}", fn(c["sigF_n"]), fn(c["sigF_n_unk"]),
                     f(c["sig_z_mm"], 1), f(c["sig_y_mm"], 1), fn(c["sigF_n_bg"]), fn(m["sigF_n"]),
                     f"{r['bias_range'][0]:.2f}&#8211;{r['bias_range'][1]:.2f}"])
    one, onec, four = X.lay("one_d4.76x1.59_d5", "under"), X.lay("one_d4.76x1.59_d5", "grid4"), X.lay("four_d3.18x1.59_d5", "under")
    txt = (f"<p>The number of chips matters more than the number of magnets. One 4.76&#8202;mm magnet over one chip reads a contact at the "
           f"contact point with {fn(one['centre']['sigF_n'])}&#8202;N of noise, but one chip&#8217;s three readings cannot separate the "
           "force from where it is applied or from the Earth&#8217;s field, and when the contact lies 3&#8211;4&#8202;mm away it reads "
           f"{one['bias_range'][0]:.2f} of the force (Table&#160;[[tab:layouts]]). The same magnet read by four chips on a 6 &#215; 7&#8202;mm "
           f"grid gives {fn(onec['centre']['sigF_n'])}&#8202;N with the position known, {fn(onec['centre']['sigF_n_unk'])}&#8202;N while also "
           f"locating the contact to {f(onec['centre']['sig_z_mm'], 1)}&#8202;mm along the finger, and {fn(onec['centre']['sigF_n_bg'])}&#8202;N "
           "with a uniform background field unknown. Four 3.18&#8202;mm magnets with a chip under each are worse at the contact point, "
           f"{fn(four['centre']['sigF_n'])}&#8202;N, which lies between the magnets: each moves less and carries 45&#8202;% of the large "
           f"magnet&#8217;s moment. They read shear along the finger better ({fn(four['centre']['sigF_z'])} against "
           f"{fn(one['centre']['sigF_z'])}&#8202;N) and over-read contacts near a magnet by up to {100 * (four['bias_range'][1] - 1):.0f}&#8202;%.</p>")
    if none:
        E0 = none["E_best_MPa"]
        txt += (f"<p>The magnets stiffen the tip about equally: the pad modulus at the contact point is {E0:.1f}&#8202;MPa without a magnet, "
                f"{R['one_d4.76x1.59_d5']['E_best_MPa']:.1f} with one and {R['four_d3.18x1.59_d5']['E_best_MPa']:.1f} with four "
                f"(this coarser grid gives {100 * (R['one_d4.76x1.59_d5']['E_best_MPa'] / X.tip_ref_E - 1):.0f}&#8202;% more than the "
                "fingertip page&#8217;s for the same tip). Four magnets spread the stiffening over the face "
                "instead of concentrating it under the contact point (Table&#160;[[tab:stiff]]).</p>")
        srows = []
        for name, lbl in (("one_d4.76x1.59_d5", "One 4.76&#8202;mm magnet"), ("four_d3.18x1.59_d5", "Four 3.18&#8202;mm magnets")):
            if name not in R:
                continue
            ratio = [p["normal_face_um_per_N"] / q["normal_face_um_per_N"] for p, q in zip(R[name]["patches"], none["patches"])]
            centre = [p["normal_face_um_per_N"] / q["normal_face_um_per_N"] for p, q in zip(R[name]["patches"], none["patches"])
                      if p["y"] == 0.0 and abs(p["z"] - 10.86) < 1e-6][0]
            srows.append([lbl, f"{R[name]['E_best_MPa']:.1f}", f"{100 * (1 - centre):.0f}", f"{100 * (1 - max(ratio)):.0f}&#8211;"
                          f"{100 * (1 - min(ratio)):.0f}"])
        txt += table(["Layout", "Pad modulus<br>(MPa)", "Face compliance lost<br>at the contact point (%)",
                      "Lost over the 15<br>load points (%)"], srows,
                     "Stiffening by the magnets: the face&#8217;s own compliance under a 1&#8202;N patch (Gaussian, &#963; 0.6&#8202;mm) against "
                     "the tip without magnets, at the contact point and over the 15 points of the load grid.", key="stiff")
    tab = table(["Layout", "Pad modulus<br>(MPa)", "Force noise,<br>position known (N)", "Position<br>unknown (N)",
                 "Position along<br>finger (mm)", "Across<br>(mm)", "With Earth&#8217;s<br>field unknown (N)",
                 "Median over<br>15 points (N)", "Bias over<br>the face"], rows,
                "Cram&#233;r&#8211;Rao bounds for 1&#8202;N normal at the contact point, 141&#8202;Hz setting, chips 1&#8202;mm behind the back "
                "face; &#8734;: the readings cannot separate the unknowns. 95A 20&#8202;% gyroid tip, homogenized core, magnets bonded, tops "
                "5&#8202;mm deep; the four-magnet grid is 6&#8202;mm across and 7&#8202;mm along the finger. Bias: range of the force a "
                "one-position estimate reads over 15 load points from &#8722;3 to 3&#8202;mm across and &#8722;4 to 4&#8202;mm along.", key="layouts")
    return txt + tab


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
FIT_D, FIT_T = 0.90, 0.95


def buy_section(X: Data):
    rows = []
    for brand, size, d, t, tol, pack, url in AMAZON:
        V = math.pi * (d / 2) ** 2 * t
        rows.append([f"{brand}, {size}", tol, pack, f"{V / (math.pi * 2.38 ** 2 * 1.59):.2f}", f"{FIT_D * d:.2f} &#215; {FIT_T * t:.2f}",
                     f"<a href=\"{url}\">{url.split('/dp/')[1]}</a>"])
    tab = table(["Amazon listing (N52, axial, Ni-Cu-Ni)", "Tolerance", "Pack, price", "Moment over<br>4.76 &#215; 1.59",
                 "Pocket at 90&#8202;/&#8202;95&#8202;%<br>(mm)", "ASIN"], rows,
                "N52 disc magnets on amazon.com, pages read 2026-10-13 (logged out; prices as shown). AplysiaTech states the grade, axial "
                "magnetization, coating and tolerance for every size. Pocket: 90&#8202;% of the diameter and 95&#8202;% of the height, the "
                "owner&#8217;s press fit.", key="buy")
    kj = ", ".join(f"{p} ({s}, pocket {FIT_D * d:.2f} &#215; {FIT_T * t:.2f}&#8202;mm)" for p, s, d, t in KJ)
    txt = ("<p>For a four-magnet tip at the 6 &#215; 7&#8202;mm grid, magnets up to 5&#8202;mm across fit with 1&#8202;mm of printed wall "
           "between pockets; the 1/16&#8202;in (1.59&#8202;mm) thickness keeps the design&#8217;s depth. The listings below with a stated "
           "tolerance of &#177;0.1&#8202;mm suit a press fit; a 3.0&#8202;mm magnet is not 1/8&#8202;in, and a pocket drawn for one loosens "
           "on the other. No Amazon listing offers 1/8 &#215; 1/16&#8202;in in N52, the size of the four-magnet runs; K&amp;J Magnetics "
           f"sells it as D21B-N52, with the references {kj}, all &#177;0.004&#8202;in.</p>"
           "<p>Pockets follow the owner&#8217;s experience that a printed TPU hole must be undersized: 90&#8202;% of the magnet&#8217;s "
           "diameter and 95&#8202;% of its height. The owner&#8217;s eFlesh trials in <code>create_pouch.py</code> (2025-10-26) "
           "converged on a cavity of 7.56 &#215; 2.65&#8202;mm, since eFlesh&#8217;s pouch scales the listed diameter by 1.05 and the "
           "thickness by 0.945. With the height at 95&#8202;% the magnet stands 0.05 of its thickness above the pocket, and the first "
           "layer after the pause covers it.</p>")
    return txt + tab


def limitations():
    items = [
        "The chip&#8217;s noise at a gain below 7 is not in the datasheet; the model keeps the highest gain&#8217;s field noise and adds the "
        "counting step.",
        "The offset drift is the datasheet&#8217;s band over the full temperature range spread linearly; the chip&#8217;s drift near room "
        "temperature can be smaller, and its gradient is largest at 85&#8202;&#176;C.",
        "The four-magnet runs use the homogenized gyroid core on a coarser grid, 13&#8202;% stiffer than the fingertip page&#8217;s, with the "
        "contact as a Gaussian patch; the bounds hold for small loads, where the response is linear.",
        "Magnets are point dipoles; a chip 1&#8202;mm behind the back face is 6&#8211;9&#8202;mm from a magnet, three to four of its "
        "diameters, where the dipole is within a few percent of a disc.",
        "Three of six layouts ran before the session ended: the four-magnet runs with 3 &#215; 1&#8202;mm magnets, with 3.18&#8202;mm magnets "
        "3&#8202;mm deep and with four 4.76&#8202;mm magnets are queued.",
    ]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def open_items():
    items = [
        ("Bench noise and drift of one chip.", "Stream one MLX90393 at GAIN_SEL 7, RES 0, OSR 0, DIG_FILT 5 under a printed tip for 10&#8202;min "
         "with no load, then turn the hand through 180&#8202;&#176;; the noise model holds if the standard deviation is within 30&#8202;% of "
         "1.15 and 1.75&#8202;&#181;T, and the turn shows the Earth&#8217;s field this page assumes."),
        ("Firmware setting for a stiff tip.", "Change <code>setGainSel(0x1)</code> and <code>setResolution(0x2, 0x2, 0x2)</code> in "
         "<code>5X_eflesh_stream.ino</code> to <code>0x7</code> and <code>0x0</code> and DIG_FILT to 5 for the gyroid tips; keep "
         "eFlesh&#8217;s setting for its large magnets."),
        ("The other three layouts.", "<code>~/miniconda3/envs/femtip/bin/python scripts/tpu_tip_magnet_layouts.py run --layout L</code> for "
         "<code>four_d3x1_d5</code>, <code>four_d3.18x1.59_d3</code> and <code>four_d4.76x1.59_d5</code> (2&#8202;min and 5.5&#8202;GB "
         "each), then this page."),
        ("Five chips under four magnets.", "Add eFlesh&#8217;s five-chip cross (a chip at the centre and one under each magnet) to "
         "<code>scripts/tpu_tip_magnet_sensing.py</code>; four magnets are worth it if the centre chip brings the force noise at the contact "
         "point under the one-magnet, four-chip value."),
        ("Facing fingertips.", "Compute each chip&#8217;s field from the other two fingertips&#8217; magnets over the grasp-to-turn trajectory "
         "of the deployed hands, and the residual after subtracting it from the joint angles with 0.5&#8202;mm of position error."),
    ]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def lede(X: Data):
    g, gm, o = X.tip("gyroid20"), X.tip("w95_20_model"), X.tip("eflesh_open")
    onec = X.lay("one_d4.76x1.59_d5", "grid4")
    four = X.lay("four_d3.18x1.59_d5", "under")
    return (f"The gyroid tip&#8217;s smaller signal is fine for noise and costly for disturbances. With the MLX90393 at its highest gain and "
            f"141&#8202;Hz, one sample&#8217;s normal-force noise is {fn(g['sigF_N'])}&#8202;N on the 20&#8202;% gyroid tip as printed "
            f"({fn(gm['sigF_N'])}&#8202;N on the earlier model) against {fn(o['sigF_N'])}&#8202;N on the open eFlesh lattice. The Earth&#8217;s "
            f"field, 10&#8202;&#176;C of drift and a facing magnet moving 5&#8202;mm read as {fn(g['earth_N'])}, {fn(g['drift10C_N'])} and "
            f"{fn(g['neighbour_N'])}&#8202;N on the gyroid and under 1&#8202;N on eFlesh, so the gyroid needs a reference reading and "
            "calibrated subtraction. eFlesh&#8217;s firmware setting must change for it. One magnet read by four chips resolves force to "
            f"{fn(onec['centre']['sigF_n'])}&#8202;N and the contact to {f(onec['centre']['sig_z_mm'], 1)}&#8202;mm along the finger; four "
            f"smaller magnets with a chip each read the contact point worse, {fn(four['centre']['sigF_n'])}&#8202;N.")


def byline():
    return ("Chip model <code>scripts/mlx90393_model.py</code> (datasheet rev. 012 in <code>logs/20261013-mlx/</code>) &#183; layouts "
            "<code>scripts/tpu_tip_magnet_layouts.py</code>, bounds <code>scripts/tpu_tip_magnet_sensing.py</code> &#183; builder "
            "<code>scripts/mlx90393_readout_page.py</code> &#183; rows <code>docs/experiments/20261013-magnet_layouts/fem.jsonl</code>")


def render_tex(t):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None) for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    return P.TEX_RE.sub(lambda m: next(svgs), t), len(items)


def main():
    X = Data()
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "GLOSSARY": glossary(), "LEDE": lede(X), "CHAIN": chain_section(X), "CHIP": chip_section(X),
         "TIPS": tips_section(X), "LAYOUTS": layouts_section(X), "BUY": buy_section(X), "LIMITS": limitations(),
         "OPEN": open_items(),
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
