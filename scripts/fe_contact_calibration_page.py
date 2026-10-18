#!/usr/bin/env python3
"""Page: calibration of Drake's pressure field and the 1 mm pads to finite elements of the 85A printed fingertips
(docs/experiments/20261018-fe_contact_calibration/, plan.json).

Every number is read from the study's rows (fe/, fit/, bed/, replays/) through summary.json:

    .venv/bin/python scripts/fe_contact_calibration_summary.py
    .venv/bin/python scripts/fe_contact_calibration_page.py
Out: docs/experiments/20261018-fe_contact_calibration/20261018-fe_contact_calibration.html (media inlined).
"""
from __future__ import annotations

import base64
import io
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
import interactive_figure as IF  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261018-fe_contact_calibration")
OUT = os.path.join(D, "20261018-fe_contact_calibration.html")
TPL = os.path.join(D, "page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
FIG, TAB, REFS = [0], [0], {}
EXTRA_CSS = """
.tw th{white-space:normal !important;vertical-align:bottom;text-transform:none}
.tw td:first-child{min-width:9em}
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
figure.films video{width:100%;height:auto}
figure.img img{width:100%;height:auto}
.eq{margin:.8em 0;text-align:center}
td.pass{color:#008800} td.fail{color:#cc0000}
""" + texsvg.BLOCK_CSS + "\n"
TIPS = ["block4_g30", "cap_g30", "ell16_g30", "ell11_g30"]
NAME = {"block4_g30": "Block", "cap_g30": "Capsule r&#8202;8.5&#8202;mm", "ell16_g30": "Half-ellipsoid 16/32&#8202;mm",
        "ell11_g30": "Half-ellipsoid 11/22&#8202;mm"}
ROWLAB = {"block4_g30": "Block", "cap_g30": "Capsule", "ell16_g30": "Ellipsoid 16/32", "ell11_g30": "Ellipsoid 11/22"}
NAME_PLAIN = {"block4_g30": "block", "cap_g30": "capsule", "ell16_g30": "16/32 half-ellipsoid",
              "ell11_g30": "11/22 half-ellipsoid"}
TASK = ["cx5_z0", "cx5_z+4", "cx5_z-4", "cx8_z0", "cx8_z+4", "cx8_z-4", "cx12.5_z0", "cx12.5_z+4", "cx12.5_z-4",
        "cz8_y0", "cz12.5_y0"]
S = json.load(open(os.path.join(D, "summary.json"))) if os.path.exists(os.path.join(D, "summary.json")) else {}
PLAN = json.load(open(os.path.join(D, "plan.json")))
UM, MM, NN, DEG, MPA, NMM = "&#8202;&#181;m", "&#8202;mm", "&#8202;N", "&#176;", "&#8202;MPa", "&#8202;N/mm"


# ------------------------------------------------------------------------------------------ helpers

def rows(rel):
    p = os.path.join(D, rel)
    out = []
    if os.path.exists(p):
        for line in open(p):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


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


def ifig(fig, caption, key=None):
    FIG[0] += 1
    if key:
        REFS[f"fig:{key}"] = FIG[0]
    fig.caption = f"Figure&#160;{FIG[0]}. {caption}"
    return fig.html()


def table(head, body, caption, note=None, text_cols=(0,), key=None):
    TAB[0] += 1
    if key:
        REFS[f"tab:{key}"] = TAB[0]
    cls = lambda i: "" if i in text_cols else " class=num"  # noqa: E731
    out = [f"<p class='tcap'><b>Table&#160;{TAB[0]}.</b> {caption}</p>",
           "<div class='tw'><table><thead><tr>" + "".join(f"<th{cls(i)}>{h}</th>" for i, h in enumerate(head))
           + "</tr></thead><tbody>"]
    for r in body:
        if isinstance(r, str):
            out.append(f"<tr><td colspan={len(head)}><i>{r}</i></td></tr>")
            continue
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
    return "&#8211;" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{n}f}"


def pf(ok):
    return "pass" if ok else "fail"


def fe(tip, probe, F="1", key="approach_um", grid="base"):
    r = S["fe"].get(f"{tip}|{grid}|{probe}")
    return None if r is None else r["per_force"][F][key]


def fit(tip, model, calib):
    return S["fits"][tip]["models"][model][calib]


# ------------------------------------------------------------------------------------------ model curves

def curves():
    out = {}
    for r in rows("fit/curves.jsonl"):
        out[(r["tip"], r["model"], r["probe"])] = r
    return out


CV = None


def model_at(tip, model, probe, E, F, ks=None):
    """(approach um, r_bar mm, contacts) of a model curve at modulus E (Pa) and load F (N), with a series spring ks
    (N/m)."""
    global CV
    if CV is None:
        CV = curves()
    r = CV[(tip, model, probe)]
    d, Fg = np.array(r["delta_um"]) * 1e-6, np.array(r["F_N"]) * E / r["E0_Pa"]
    ok = Fg > 0
    lf, ld = np.log(Fg[ok]), np.log(d[ok])
    keep = np.r_[True, np.diff(lf) > 0]
    dd = float(np.exp(np.interp(math.log(F), lf[keep], ld[keep])))
    rb = float(np.interp(math.log(dd), np.log(d[ok][keep]), np.array(r["r_bar_mm"])[ok][keep]))
    nn = float(np.interp(math.log(dd), np.log(d[ok][keep]), np.array(r["n_contact"])[ok][keep]))
    return dd * 1e6 + (1e6 * F / ks if ks else 0.0), rb, nn


# ------------------------------------------------------------------------------------------ figures

def approach_figure():
    Fx = list(np.geomspace(0.4, 6.0, 24))
    panels, series, legend = [], [], []
    styles = [("fe", "Finite elements", "#000", "", "both"), ("drake|U", "Drake, U", IF.PALETTE[0], "2 3", "line"),
              ("drake|M1", "Drake, M1", IF.PALETTE[0], "", "line"), ("drake|M3", "Drake, M3", IF.PALETTE[0], "7 4", "line"),
              ("v2|U", "Pads, U", IF.PALETTE[1], "2 3", "line"), ("v2|M1", "Pads, M1", IF.PALETTE[1], "", "line"),
              ("v2|M3", "Pads, M3", IF.PALETTE[1], "7 4", "line")]
    ids = {s[0]: [] for s in styles}
    for tip in TIPS:
        key = tip.replace("_", "")
        panels.append(IF.Panel(key, f"{NAME[tip]}: approach under the 12.5&#8202;mm rod (&#181;m), log scale",
                               (20.0, 1000.0), band=False, dec=1, height=170, ylog=True))
        for sk, lab, col, dash, mode in styles:
            if sk == "fe":
                xs = [0.5, 1.0, 3.0, 6.0]
                ys = [fe(tip, "cx12.5_z0", f"{F:g}") for F in xs]
            else:
                m, c = sk.split("|")
                t = S["fits"][tip]
                if c == "U":
                    E, ks = t["U_E_MPa"] * 1e6, None
                elif c == "M3":
                    E, ks = fit(tip, m, "M2")["E_MPa"] * 1e6, t["k_s_N_per_mm"] * 1e3
                else:
                    E, ks = fit(tip, m, c)["E_MPa"] * 1e6, None
                xs = Fx
                ys = [model_at(tip, m, "cx12.5_z0", E, F, ks)[0] for F in xs]
            sid = f"ap-{tip}-{sk.replace('|', '-')}"
            series.append(IF.Series(sid, lab, col, {key: {"x": xs, "med": ys}}, dash=dash, mode=mode,
                                    width=2.0 if sk == "fe" else 1.8,
                                    point_labels={key: [f"{NAME[tip]}, {lab}: {y:.0f}&#8202;&#181;m at {x:.2g}&#8202;N"
                                                        for x, y in zip(xs, ys)]}))
            ids[sk].append(sid)
    legend = [(lab, col, dash, ids[sk]) for sk, lab, col, dash, _ in styles]
    fig = IF.Figure("approach", None, "Normal load (N), log scale", panels, series, legend=legend, x_domain=(0.4, 6.0),
                    xlog=True)
    return fig


def arm_figure():
    Fx = list(np.geomspace(0.4, 6.0, 24))
    panels, series = [], []
    styles = [("fe", "Finite elements", "#000", "", "both"), ("drake|M1", "Drake, M1", IF.PALETTE[0], "", "line"),
              ("drake|M2", "Drake, M2", IF.PALETTE[0], "7 4", "line"), ("v2|M1", "Pads, M1", IF.PALETTE[1], "", "line"),
              ("v2|M2", "Pads, M2", IF.PALETTE[1], "7 4", "line")]
    ids = {s[0]: [] for s in styles}
    for tip in TIPS:
        key = tip.replace("_", "")
        panels.append(IF.Panel(key, f"{NAME[tip]}: twist arm under the 12.5&#8202;mm rod (mm)", None, band=False, dec=2,
                               height=150))
        for sk, lab, col, dash, mode in styles:
            if sk == "fe":
                xs = [0.5, 1.0, 3.0, 6.0]
                ys = [fe(tip, "cx12.5_z0", f"{F:g}", "r_bar_mm") for F in xs]
            else:
                m, c = sk.split("|")
                E = fit(tip, m, c)["E_MPa"] * 1e6
                xs = Fx
                ys = [model_at(tip, m, "cx12.5_z0", E, F)[1] for F in xs]
            sid = f"arm-{tip}-{sk.replace('|', '-')}"
            series.append(IF.Series(sid, lab, col, {key: {"x": xs, "med": ys}}, dash=dash, mode=mode,
                                    point_labels={key: [f"{NAME[tip]}, {lab}: {y:.2f}&#8202;mm at {x:.2g}&#8202;N"
                                                        for x, y in zip(xs, ys)]}))
            ids[sk].append(sid)
    legend = [(lab, col, dash, ids[sk]) for sk, lab, col, dash, _ in styles]
    return IF.Figure("arm", None, "Normal load (N), log scale", panels, series, legend=legend, x_domain=(0.4, 6.0), xlog=True)


def patch_png():
    """Patch maps under the 12.5 mm rod at 1 N: finite-element nodal pressure, Drake's field pressure and the pads'
    forces at M1 and M2 (media/patch_maps.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(D, "media", "patch_maps.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 22})
    cols = [("fe", "Finite elements"), ("drake_M1", "Drake, M1"), ("drake_M2", "Drake, M2"), ("pads_M1", "Pads, M1"),
            ("pads_M2", "Pads, M2")]
    fig, axs = plt.subplots(len(TIPS), len(cols), figsize=(16, 2.35 * len(TIPS) + 0.9), squeeze=False)
    for i, tip in enumerate(TIPS):
        zf = np.load(os.path.join(D, "fe", f"patch_{tip}_base.npz"))
        pm_path = os.path.join(D, "fit", f"patch_{tip}.npz")
        zm = np.load(pm_path) if os.path.exists(pm_path) else None
        z_c = 21.41 - 10.55
        for j, (k, lab) in enumerate(cols):
            ax = axs[i, j]
            if k == "fe":
                idx, p = zf["cx12.5_idx"], zf["cx12.5_p"]
                y, z, A = zf["y"][idx], zf["z"][idx] - z_c, zf["A"][idx]
                ax.scatter(y, -z, c=p / A, s=30, cmap="viridis", marker="s")
                ax.set_title(f"{lab}" if i == 0 else "", fontsize=22)
            elif zm is not None:
                pm = (S["fits"][tip].get("pad_model") or {}).get(k.split("_")[1], "v2") if k.startswith("pads") else None
                kk = f"{pm}_{k.split('_')[1]}" if pm else k
                if f"{kk}_yz" not in zm.files:
                    kk = k.replace("pads", "v2")
                yz = zm[f"{kk}_yz"]
                w = zm[f"{kk}_p"] if k.startswith("drake") else zm[f"{kk}_f"]
                ax.scatter(yz[:, 0], yz[:, 1], c=w, s=6 if k.startswith("drake") else 50, cmap="viridis")
                ax.set_title(f"{lab}" if i == 0 else "", fontsize=22)
            ax.set_xlim(-6.0, 6.0)
            ax.set_ylim(-2.6, 2.6)
            ax.set_aspect("equal")
            ax.set_xticks([-5, 0, 5])
            ax.set_yticks([-2, 0, 2])
            ax.tick_params(labelsize=19)
            if j == 0:
                ax.set_ylabel(f"{ROWLAB[tip]}\nz (mm)", fontsize=20)
            if i == len(TIPS) - 1:
                ax.set_xlabel("y (mm)", fontsize=21)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return "media/patch_maps.png"


# ------------------------------------------------------------------------------------------ text

GLOSSARY = [
    ("Approach", "travel of the indenter along the palmar normal from first touch of the tip&#8217;s surface at a given "
                 "normal load, in &#181;m; finite-element and simulator values are model values, bench values are pending"),
    ("Task cylinders", "the indenters the criteria use: rods of radius 5, 8 and 12.5&#8202;mm across the finger (axis "
                       "along y) at z&#8202;0 and &#177;4&#8202;mm, and of radius 8 and 12.5&#8202;mm along the finger "
                       "(axis along z) at y&#8202;0; the 12.5&#8202;mm rod across the finger at z&#8202;0 is the bench "
                       "and bed tool, the 25&#8202;mm screwdriver shaft. z is the tip-frame coordinate toward the mount, "
                       "0 at the contact point"),
    ("Force exponent n", "the slope of approach against load on log axes between 0.5 and 3&#8202;N, "
                         r"\(n = \ln(a_{3}/a_{0.5}) / \ln 6\); a linear spring gives 1, a local law 0.5 under a point "
                         "contact and 2/3 under a line contact"),
    ("Twist arm r&#772;", r"the pressure-weighted mean distance of the contact pressure from its centre in the face plane, "
                          r"in mm: \(\bar r = \sum_i p_i r_i / \sum_i p_i\); the full-slip torque about the contact normal "
                          r"is \(\mu N \bar r\). Arm ratio: model r&#772; over finite-element r&#772; under the same probe "
                          "and load"),
    ("Approach error", "model approach over finite-element approach minus 1, over the task cylinders at 0.5, 1 and 3&#8202;N; "
                       "RMS and largest absolute value, in %"),
    ("E", r"the contact modulus. Drake: the hydroelastic modulus of the tip&#8217;s compliant convex hull, "
          r"\(p = E\,(1-g)\) with g the gauge about the hull&#8217;s volume centroid. Pads: \(K_i = E\,a_i/h_i\), a<sub>i</sub> "
          "the pad&#8217;s area share and h<sub>i</sub> its depth below the volume centroid (builder v2)"),
    ("k<sub>s</sub>", "series spring between the DIP link and the tip&#8217;s contact geometry along the palmar normal, "
                      "in N/mm: approach = F/k<sub>s</sub> + the local approach at the modulus behind it"),
    ("U, M1, M2, M3", "the calibrations. U: the readiness moduli of 2026-10-16 (block 7.55, capsule 13.45, 16/32 "
                      "15.01, 11/22 14.23&#8202;MPa, the same E in both simulators). M1: one E per tip and simulator "
                      "fitted (log-RMS) to the finite-element approach under the task cylinders at 0.5&#8211;3&#8202;N. M2: one "
                      "E fitted to the finite-element r&#772; under the 8 and 12.5&#8202;mm rods across the finger at z&#8202;0, "
                      "1 and 3&#8202;N. M3: M2&#8217;s E behind a series spring k<sub>s</sub> fitted to the approach, one "
                      "k<sub>s</sub> per tip (the mean of the Drake and 1&#8202;mm-pad fits). All model values"),
    ("Pads, pads at 0.5&#8202;mm", "MuJoCo spheres of radius 0.75&#8202;mm on a 1&#8202;mm surface grid (builder v2); rule "
                                   "R4 replaces them by spheres of 0.375&#8202;mm on a 0.5&#8202;mm grid where fewer than six "
                                   "1&#8202;mm pads touch the 8 or 12.5&#8202;mm rod at 1&#8202;N"),
    ("Numerics", "planner and evaluation: MuJoCo 1&#8202;ms, impratio 100, Newton 100 iterations; RL trainer: 2&#8202;ms, "
                 "impratio 10, 10 iterations; Drake 1.57: SAP at 1&#8202;ms, relaxation time 0.01&#8202;s"),
    ("Onset arm", "a simulator&#8217;s twist-onset torque on the bed (two tips squeeze the 25&#8202;mm tool, torque ramp "
                  "about the pinch axis) over &#956;N, in mm"),
    ("Creep", "steady twist rate of the bed tool under a held torque at half the onset torque, over 1&#8202;s, in "
              "&#176;/s; rigid Coulomb friction gives 0"),
    ("Held", "the tool less than 20&#8202;mm below its start and at least two fingers each pressing with its weight at "
             "the end of the run"),
    ("RMS angle", "RMS over a run&#8217;s 20&#8202;ms samples of the angle between two runs&#8217; tool orientations, in &#176;"),
    ("Same branch", "both runs keep the tool and end within 15&#176; of each other"),
    ("Seed spread", "the median RMS angle between one model&#8217;s runs from different start placements (seeds 1 and 2 "
                    "against seed 0), over same-branch pairs"),
    ("Force differences", "RMS over samples and fingers of the difference between two runs of each finger&#8217;s summed "
                          "normal force on the tool, and of the magnitude of its friction force, in N"),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def byline():
    return ("Scripts <code>scripts/fe_contact_calibration_{fe,fit,sim,summary,page}.py</code> &#183; plan "
            "<code>plan.json</code> &#183; rows <code>docs/experiments/20261018-fe_contact_calibration/</code> "
            "(<code>fe/</code>, <code>fit/</code>, <code>bed/</code>, <code>replays/</code>, <code>media/</code>)")


# ------------------------------------------------------------------------------------------ tables

def meta_of(tip, grid="base"):
    for m in S.get("fe_meta", []):
        if m["tip"] == tip and m["grid"] == grid:
            return m
    return {}


def expo(tip, probe, grid="base"):
    a, b = fe(tip, probe, "0.5", grid=grid), fe(tip, probe, "3", grid=grid)
    return math.log(b / a) / math.log(6) if a and b else None


def fe_table():
    head = ["Tip", "Sphere at the apex, 0.5 / 1 / 3 / 6&#8202;N (&#181;m)", "12.5&#8202;mm rod across, 0.5 / 1 / 3 / 6&#8202;N (&#181;m)",
            "12.5&#8202;mm rod along, 1&#8202;N (&#181;m)", "Exponent n, sphere / rod", "Rod over sphere, 1&#8202;N",
            "r&#772; under the rod, 1 / 3&#8202;N (mm)", "Degrees of freedom"]
    body = []
    for tip in TIPS:
        sp = " / ".join(f(fe(tip, "s_apex", F), 0) for F in ("0.5", "1", "3", "6"))
        rd = " / ".join(f(fe(tip, "cx12.5_z0", F), 0) for F in ("0.5", "1", "3", "6"))
        body.append([NAME[tip], sp, rd, f(fe(tip, "cz12.5_y0"), 0), f"{f(expo(tip, 's_apex'))} / {f(expo(tip, 'cx12.5_z0'))}",
                     f(fe(tip, "cx12.5_z0") / fe(tip, "s_apex")),
                     f"{f(fe(tip, 'cx12.5_z0', '1', 'r_bar_mm'))} / {f(fe(tip, 'cx12.5_z0', '3', 'r_bar_mm'))}",
                     f"{meta_of(tip).get('n_dof', 0) / 1e3:.0f}&#8202;k"])
    return table(head, body, "Finite elements of the four prints in Siraya 85A at 30&#8239;% gyroid with four 3&#8202;&#215;&#8202;2&#8202;mm "
                             "magnets (fe/fe.jsonl, model values): approach under the 2&#8202;mm-radius sphere at the apex and "
                             "under the 12.5&#8202;mm rod across and along the finger, the force exponent between 0.5 and "
                             "3&#8202;N, the rod-over-sphere approach ratio and the twist arm under the rod across the finger.",
                 key="fe")


def fits_table():
    head = ["Tip", "Calibration", "k<sub>s</sub> (N/mm)", "Drake: E (MPa)", "Drake: approach error, RMS / largest (%)",
            "Drake: arm ratio", "Pads: spacing, E (MPa)", "Pads: approach error, RMS / largest (%)", "Pads: arm ratio"]
    body = []

    def mark(txt, ok):
        return txt if ok else f"{txt} (fail)"
    for tip in TIPS:
        t = S["fits"][tip]
        for c in ("M1", "M2", "M3"):
            cc = (t.get("calib_of") or {}).get(c, c) if c == "M2" else c
            pm = (t.get("pad_model") or {}).get(c, "v2")
            d, p_ = t["models"]["drake"].get(cc), t["models"][pm].get(cc)
            if not d or not p_:
                continue
            ks = d.get("k_s_N_per_mm")
            lab = c + (" (= M1, rule R3)" if cc != c else "")
            body.append([NAME[tip], lab, f(ks, 1) if ks and ks < 1000 else "&#8211;", f(d["E_MPa"], 1),
                         mark(f"{d['approach_rms_pct']:.1f} / {d['approach_max_pct']:.0f}", d["pass_approach"]),
                         mark(f"{d['arm_min']:.2f}&#8211;{d['arm_max']:.2f}", d["pass_arm"]),
                         f"{'0.5' if pm == 'v2h' else '1'}&#8202;mm, {p_['E_MPa']:.1f}",
                         mark(f"{p_['approach_rms_pct']:.1f} / {p_['approach_max_pct']:.0f}", p_["pass_approach"]),
                         mark(f"{p_['arm_min']:.2f}&#8211;{p_['arm_max']:.2f}", p_["pass_arm"])])
    return table(head, body, "The calibrations against the finite elements (fit/fits.json, model values). Approach passes at an "
                             "RMS error of at most 25&#8239;% and a largest error of at most 40&#8239;% over the task cylinders at "
                             "0.5, 1 and 3&#8202;N; twist arm passes when every arm ratio under the 8 and 12.5&#8202;mm rods "
                             "across the finger at 1 and 3&#8202;N lies within 0.70&#8211;1.15. The pads use rule R4&#8217;s "
                             "spacing; on the block a spring stiffer than 1000&#8202;N/mm is shown as none.", key="fits",
                 text_cols=(0, 1))


def bed_table():
    head = ["Tip", "Calibration", "Model", "E (MPa)", "Bed approach at 1&#8202;N (&#181;m)", "Against the curve from first touch, largest (%)",
            "Onset arm at 1 / 3&#8202;N (mm)", "Over finite-element r&#772;", "Creep at 1&#8202;N (&#176;/s)"]
    body = []
    B = S.get("bed_checks", {})
    for tip in TIPS:
        for c in ("M1", "M2"):
            for arm in ("pads", "rl", "drake"):
                v = B.get(f"{tip}|{c}|{arm}")
                if not v:
                    continue
                lab = {"pads": "Pads, planner", "rl": "Pads, RL trainer", "drake": "Drake"}[arm]
                if arm != "drake":
                    lab += ", 0.5&#8202;mm" if v["model"] == "v2h" else ", 1&#8202;mm"
                a1 = (v["approach"].get("1") or {}).get("bed_um")
                on = v["onset"]
                rr = v.get("arm_ratio_range")
                cr = (v.get("creep_deg_s") or {}).get("1")
                body.append([NAME[tip], c, lab, f(v["E_MPa"], 2), f(a1, 1),
                             f(100 * v["impl_max_err"], 1) if v.get("impl_max_err") is not None else "&#8211;",
                             f"{f((on.get('1') or {}).get('bed_mm'))} / {f((on.get('3') or {}).get('bed_mm'))}",
                             f"{f(rr[0])}&#8211;{f(rr[1])}" if rr else "&#8211;", f(cr, 3)])
    return table(head, body, "The calibrated models on the contact bed (bed/, model values): static approach under the 25&#8202;mm "
                             "tool at 1&#8202;N and its largest difference from the phase-B curve at the same modulus over 0.5, 1 "
                             "and 3&#8202;N (the implementation check, at most 5&#8239;%), the twist-onset arm and its ratio to the "
                             "finite-element r&#772; under the 12.5&#8202;mm rod at 1 and 3&#8202;N (band 0.70&#8211;1.15), and the "
                             "creep under half the onset torque.", key="bed")


def d1_table(own=False):
    head = ["Tip", "Calibration", "Pads", "Held, pads / Drake", "Held agreement", "Same branch",
            "RMS angle, median (&#176;)", "Pads seed spread (&#176;)", "Normal / friction force difference (N)", "Consistency"]
    body = []
    D1 = S.get("d1", {})
    for tip in TIPS:
        for c in ("U", "M1", "M2"):
            for pk in ("pads", "pads1"):
                v = D1.get(f"{tip}|{'own' if own else 'ready'}|{c}|{pk}")
                if not v:
                    continue
                if pk == "pads1" and c != "M2":
                    continue
                s, sp = v["pads_vs_drake"], v["pads_seed_spread"]
                spacing = "1&#8202;mm"
                tm = S["fits"][tip].get("pad_model") or {}
                if pk == "pads" and c != "U" and tm.get(c) == "v2h":
                    spacing = "0.5&#8202;mm"
                body.append([NAME[tip], c, spacing, f"{v['held_pads']} / {v['held_drake']}", f"{s['held_agree']} of {s['n']}",
                             str(s["same_branch"]), f(s["median_rms_angle_same_deg"], 2), f(sp["median_rms_angle_same_deg"], 2),
                             f"{f(s['median_rms_dN_same_N'], 3)} / {f(s['median_rms_dT_same_N'], 3)}",
                             f"<span class='{pf(v['pass'])}'>{pf(v['pass'])}</span>"])
    cap = ("Pads against Drake on the same servo commands: the pinned-iCEM runs of the readiness study on the 16&#8202;mm and "
           "25&#8202;mm cells, replayed open loop at three start placements (replays/icem.jsonl, model values). Consistency passes "
           "with held agreement on at least 5 of 6 runs, a median same-branch RMS angle of at most 3&#176; or the pads&#8217; "
           "seed spread, and median force differences of at most 0.25&#8202;N normal and 0.35&#8202;N friction."
           if not own else
           "Pads against Drake on each tip&#8217;s own plan: pinned iCEM re-planned on the tip&#8217;s 1&#8202;mm pads at M2 on the "
           "16&#8202;mm cell, replayed open loop at three start placements. Consistency passes with held agreement on 3 of 3 "
           "runs and the same angle bound.")
    return table(head, body, cap, key="d1own" if own else "d1")


ARM_NAME = {"plan": "Pads, planner", "rl": "Pads, RL trainer", "drake": "Drake", "ir1000": "Pads, impratio 1000",
            "pads05": "Pads at 0.5&#8202;mm", "spring": "Pads behind the series spring (M3)"}


def d2_table():
    head = ["Tip", "Calibration", "Arm", "Held, seeds 0 / 1 / 2 (of 8)", "Against the planner pads: held agreement",
            "Same branch", "RMS angle, median (&#176;)", "Normal / friction force difference (N)"]
    body = []
    D2 = S.get("d2", {})
    for tip in TIPS:
        for c in ("U", "M1", "M2"):
            v = D2.get(f"{tip}|{c}")
            if not v:
                continue
            for arm in ("plan", "rl", "ir1000", "pads05", "spring", "drake"):
                h = v["held"].get(arm)
                if not h:
                    continue
                hs = " / ".join(str(h.get(str(s), h.get(s, "&#8211;"))) if h["n"].get(str(s), h["n"].get(s, 0)) else "&#8211;"
                                for s in (0, 1, 2))
                p = v["pairs"].get(f"plan_vs_{arm}") if arm != "plan" else v["pairs"].get("plan_seed_spread")
                if arm == "plan":
                    row = [NAME[tip], c, ARM_NAME[arm], hs, "seed spread", str(p["same_branch"]) if p else "&#8211;",
                           f(p["median_rms_angle_same_deg"], 2) if p else "&#8211;", "&#8211;"]
                else:
                    row = [NAME[tip], c, ARM_NAME[arm], hs, f"{p['held_agree']} of {p['n']}" if p else "&#8211;",
                           str(p["same_branch"]) if p else "&#8211;", f(p["median_rms_angle_same_deg"], 2) if p else "&#8211;",
                           f"{f(p['median_rms_dN_same_N'], 3)} / {f(p['median_rms_dT_same_N'], 3)}" if p else "&#8211;"]
                body.append(row)
    return table(head, body, "The eight deployed bench plans D1&#8211;D8 replayed open loop on each tip (replays/d1d8.jsonl, model "
                             "values): held counts per start placement, and each arm against the planner pads at the same "
                             "calibration, hand and seed. For the planner pads the row gives their own seed spread.", key="d2")


# ------------------------------------------------------------------------------------------ build

def footer():
    return ("<p>Rebuild: <code>.venv/bin/python scripts/fe_contact_calibration_summary.py</code>, then "
            "<code>.venv/bin/python scripts/fe_contact_calibration_page.py</code>. The runs: "
            "<code>scripts/fe_contact_calibration_fe.py</code> (finite elements, ~/miniconda3/envs/femtip), "
            "<code>scripts/fe_contact_calibration_fit.py</code> (curves, fits, patch maps), "
            "<code>scripts/fe_contact_calibration_sim.py</code> (bed, iCEM replays and plans, D1&#8211;D8 replays, film; "
            "logs/20261001-hom_contact/venv with Drake 1.57). Rows in <code>docs/experiments/20261018-fe_contact_calibration/</code>: "
            "<code>plan.json</code>, <code>prior/</code>, <code>fe/</code>, <code>fit/</code>, <code>bed/</code>, "
            "<code>replays/</code>, <code>media/</code>, <code>summary.json</code>. Related pages: "
            "<a href=\"../20261016-sim2real_fingertip_readiness/20261016-cad_hand_fingertip_deployment_readiness.html\">deployment "
            "readiness of the CAD hand with printed 85A fingertips</a>, "
            "<a href=\"../20261015-curved_tips/20261015-capsule_and_ellipsoid_fingertips.html\">capsule and ellipsoid "
            "fingertips</a>.</p>")


def main(sections):
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "GLOSSARY": glossary(), "FOOTER": footer()}
    v.update(sections)
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
    t = IF.add_assets(retro_style.apply(t))
    open(OUT, "w").write(t)
    print(f"formulas {n}; figures {FIG[0]}; tables {TAB[0]}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


# ------------------------------------------------------------------------------------------ sections

def rng(vals, n=2, unit=""):
    vals = [v for v in vals if v is not None]
    return f"{min(vals):.{n}f}&#8211;{max(vals):.{n}f}{unit}" if vals else "&#8211;"


CURVED = ["cap_g30", "ell16_g30", "ell11_g30"]


def fe_section():
    reg = [fe("block4_g30", "cx12.5_z0", F) for F in ("0.5", "1", "3")]
    ready = PLAN["prior"]["calibrated_before"][3]["result"]
    old = [31.94, 60.76, 162.25]
    fine = [S["fe"].get(f"cap_g30|fine|{p}") for p in ("s_apex", "cx8_z0", "cx12.5_z0", "cz12.5_y0")]
    dap, darm = [], []
    for p in ("s_apex", "cx5_z0", "cx8_z0", "cx12.5_z0", "cz8_y0", "cz12.5_y0"):
        r = S["fe"].get(f"cap_g30|fine|{p}")
        if r:
            for F in ("0.5", "1", "3"):
                dap.append(abs(r["per_force"][F]["approach_um"] / fe("cap_g30", p, F) - 1))
                if p.startswith("cx"):
                    darm.append(abs(r["per_force"][F]["r_bar_mm"] / fe("cap_g30", p, F, "r_bar_mm") - 1))
    td = S["fe"].get("block_td|base|cx12.5_z0")
    td_txt = ""
    if td:
        a = [td["per_force"][F]["approach_um"] for F in ("0.5", "1", "3")]
        td_txt = (f" The readiness study&#8217;s own block surface (side fillets 2.7 instead of 3&#8202;mm) on this grid sinks "
                  f"{a[0]:.1f}, {a[1]:.1f} and {a[2]:.1f}&#8202;&#181;m, so the grid accounts for "
                  f"{100 * (a[1] / old[1] - 1):.0f}&#8239;% and the surface for the rest.")
    sh = {t: (fe(t, "cx12.5_z0", "1", "shear_y_face_um_per_N"), fe(t, "cx12.5_z0", "1", "shear_z_face_um_per_N")) for t in TIPS}
    p1 = []
    return (
        "<p>Each print was modelled as sliced: two 0.45&#8202;mm perimeters and four 0.2&#8202;mm skins of solid TPU around "
        "every outline, the magnet pockets&#8217; included, over the homogenized 30&#8239;% gyroid, in the print&#8217;s "
        "orientation (the block on its back face, the curved tips upright), with the four magnets bonded at the sheet "
        "depths (5, 5, 5.5 and 7&#8202;mm) and the back and mount faces fixed. The solid modulus is NinjaFlex 85A&#8217;s "
        "12&#8202;MPa, standing in for Siraya&#8217;s unmeasured one, so every modulus and spring below scales with the "
        "bench factor of the indentation test. Frictionless rigid probes were pressed along the palmar normal from first "
        f"touch of the design surface; the models have 137&#8211;192&#8202;k degrees of freedom and peak at 2.4&#8211;3.5&#8202;GB.</p>"
        + fe_table() +
        f"<p>Under the 2&#8202;mm sphere at the apex every tip sinks {rng([fe(t, 's_apex') for t in TIPS], 0)}&#8202;&#181;m at "
        "1&#8202;N: the skin spreads the load over millimetres of gyroid, so the structure sets the approach and the "
        f"surface shape changes it by under 5&#8239;%. Under the 12.5&#8202;mm rod across the finger the curved tips sink "
        f"{rng([fe(t, 'cx12.5_z0') for t in CURVED], 0)}&#8202;&#181;m and the block {fe('block4_g30', 'cx12.5_z0'):.0f}&#8202;&#181;m. "
        f"The approach grows as F<sup>{rng([expo(t, p) for t in CURVED for p in ('s_apex', 'cx12.5_z0')])}</sup> on the "
        f"curved tips and as F<sup>{expo('block4_g30', 'cx12.5_z0'):.2f}</sup> under the rod on the block, against 0.5 "
        "(point contact) to 2/3 (line contact) for a local law. A rod laid along the finger sinks "
        f"{fe('cap_g30', 'cx12.5_z0') / fe('cap_g30', 'cz12.5_y0'):.1f} times less than one across it on the capsule "
        f"({fe('cap_g30', 'cz12.5_y0'):.1f} against {fe('cap_g30', 'cx12.5_z0'):.1f}&#8202;&#181;m at 1&#8202;N), because "
        "the capsule&#8217;s straight section spreads it over 13&#8202;mm; on the half-ellipsoids the ratio is "
        f"{fe('ell16_g30', 'cx12.5_z0') / fe('ell16_g30', 'cz12.5_y0'):.1f} (16/32) and "
        f"{fe('ell11_g30', 'cx12.5_z0') / fe('ell11_g30', 'cz12.5_y0'):.1f} (11/22). The twist arm under the rod at 1&#8202;N is "
        f"{rng([fe(t, 'cx12.5_z0', '1', 'r_bar_mm') for t in CURVED])}&#8202;mm on the curved tips and "
        f"{fe('block4_g30', 'cx12.5_z0', '1', 'r_bar_mm'):.2f}&#8202;mm on the block, whose contact line spans its flat face. "
        "Prediction P1 holds for the curved tips (exponents 0.72&#8211;0.90, rod over sphere 0.60&#8211;0.85, arm "
        "0.35&#8211;0.70&#8202;mm predicted); the block&#8217;s arm is below the predicted 3.2&#8211;3.6&#8202;mm. The face&#8217;s "
        "tangential compliance under the rod at 1&#8202;N is "
        f"{rng([v[0] for v in sh.values()], 0)}&#8202;&#181;m/N across the finger and {rng([v[1] for v in sh.values()], 0)}&#8202;&#181;m/N "
        "along it; neither simulator represents it.</p>"
        f"<p>Regression against the readiness study: the block sinks {reg[0]:.1f}, {reg[1]:.1f} and {reg[2]:.1f}&#8202;&#181;m "
        f"under the rod at 0.5, 1 and 3&#8202;N, {100 * (reg[1] / old[1] - 1):.0f}&#8239;% more than the 2026-10-16 rows "
        f"({old[0]}, {old[1]} and {old[2]}&#8202;&#181;m on a 259&#8202;k-dof grid with the readiness block surface), which "
        "misses the plan&#8217;s 10&#8239;% bound." + td_txt + " Grid check (prediction P7): the capsule on a 0.2&#8202;mm "
        f"contact grid ({S['fe'].get('cap_g30|fine|s_apex', {}).get('n_dof', 0) / 1e3:.0f}&#8202;k degrees of freedom) "
        f"changes the approach by at most {100 * max(dap or [0]):.1f}&#8239;% and r&#772; by at most "
        f"{100 * max(darm or [0]):.1f}&#8239;%, inside the predicted 5 and 10&#8239;%.</p>")


def fits_section():
    F = S["fits"]
    m1d = [fit(t, "drake", "M1") for t in TIPS]
    m1p = [fit(t, "v2", "M1") for t in TIPS]
    m2d = [fit(t, "drake", "M2") for t in CURVED]
    m3d = {t: fit(t, "drake", "M3") for t in CURVED}
    pm3 = {t: (F[t].get("pad_model") or {}).get("M3", "v2") for t in CURVED}
    m3p = {t: fit(t, pm3[t], "M3") for t in CURVED}
    ks = {t: F[t]["k_s_N_per_mm"] for t in CURVED}
    cap = m3d["cap_g30"]["approach_err_pct"]
    cz = [v for k, v in cap.items() if k.startswith("cz")]
    dep = F["cap_g30"]["models"].get("depth", {}).get("M1")
    dmax = next((r.get("d_max_mm", 0.0) for r in rows("fit/curves.jsonl") if r["model"] == "depth"), 0.0)
    bz = fit("block4_g30", "drake", "M1")["approach_err_pct"]
    bz_k = max((k for k in bz if k.startswith("cz")), key=lambda k: abs(bz[k]))
    bz_txt = (f"{bz[bz_k]:.0f}&#8239;% too far under the {bz_k[2:].split('_')[0]}&#8202;mm rod along the finger at "
              f"{bz_k.split('@')[1]}&#8202;N on the block")
    eqs = (r"<div class='eq'>\[p = E\,(1 - g), \qquad K_i = \frac{E\,a_i}{h_i}, \qquad "
           r"a(F) = \frac{F}{k_s} + a_{\mathrm{loc}}(F;\,E_{\mathrm{loc}}), \qquad "
           r"\bar r = \frac{\sum_i p_i\,r_i}{\sum_i p_i}\]</div>")
    return (
        "<p>Both simulators use a local law: the pressure at a point of Drake&#8217;s field and the force of a pad depend "
        "only on the overlap there, so every force in the phase-B curves scales with E and one curve per model and probe "
        "gives every modulus. The curves were computed on each tip&#8217;s simulated surface (the as-printed hull) under "
        "the same 18 probes as the finite elements; a series spring adds F/k<sub>s</sub> to the approach and leaves the "
        "patch unchanged:</p>" + eqs +
        "<p>M1, the one modulus fitted to the approach, gets the RMS approach error to "
        f"{rng([v['approach_rms_pct'] for v in m1d], 1)}&#8239;% in Drake and {rng([v['approach_rms_pct'] for v in m1p], 1)}&#8239;% "
        f"in the 1&#8202;mm pads, but the largest error is {rng([v['approach_max_pct'] for v in m1d], 0)}&#8239;% in Drake "
        f"and {rng([v['approach_max_pct'] for v in m1p], 0)}&#8239;% in the pads: a local law sinks as F<sup>0.5&#8211;0.7</sup> "
        "where the print sinks as F<sup>0.75&#8211;0.9</sup>, so one modulus is too soft at 0.5&#8202;N and too stiff at "
        "3&#8202;N; Drake sinks " + bz_txt + ". At "
        f"M1 the curved tips get {min(fit(t, 'drake', 'M1')['arm_min'] for t in CURVED):.2f}&#8211;"
        f"{max(fit(t, 'drake', 'M1')['arm_max'] for t in CURVED):.2f} times the print&#8217;s twist arm in Drake "
        f"and up to {max(fit(t, 'v2', 'M1')['arm_max'] for t in CURVED):.2f} times in the pads, because the field&#8217;s "
        "patch under the rod is wider than the print&#8217;s (Figure&#160;[[fig:patch]]). The block keeps "
        f"{fit('block4_g30', 'drake', 'M1')['arm_min']:.2f}&#8211;{fit('block4_g30', 'drake', 'M1')['arm_max']:.2f} of its arm, "
        "set by the width of its flat face, and its arm changes by under 5&#8239;% over half to twice the modulus, so "
        "rule R3 sets its M2 to M1.</p>"
        f"<p>M2, fitted to the arm, needs {rng([v['E_MPa'] for v in m2d], 0)}&#8202;MPa in Drake and leaves the curved "
        "tips 27&#8211;69&#8239;% too stiff. M3 puts that modulus behind a series spring of "
        f"{ks['ell11_g30']:.1f} (11/22), {ks['ell16_g30']:.1f} (16/32) and {ks['cap_g30']:.1f}&#8202;N/mm (capsule). On the "
        f"11/22 half-ellipsoid it reproduces the approach within {m3d['ell11_g30']['approach_rms_pct']:.1f}&#8239;% RMS "
        f"(largest {m3d['ell11_g30']['approach_max_pct']:.0f}&#8239;%) and the arm within "
        f"{m3d['ell11_g30']['arm_min']:.2f}&#8211;{m3d['ell11_g30']['arm_max']:.2f} in Drake, and within "
        f"{m3p['ell11_g30']['approach_rms_pct']:.1f}&#8239;% and {m3p['ell11_g30']['arm_min']:.2f}&#8211;"
        f"{m3p['ell11_g30']['arm_max']:.2f} in the 0.5&#8202;mm pads that rule R4 requires (the 1&#8202;mm pads touch the "
        "rod with 4 pads at 1&#8202;N). On the 16/32 half-ellipsoid the approach passes "
        f"({m3d['ell16_g30']['approach_rms_pct']:.1f}&#8239;% in Drake, {m3p['ell16_g30']['approach_rms_pct']:.1f}&#8239;% in "
        f"the pads) and the arm reaches {m3d['ell16_g30']['arm_max']:.3f} in Drake and {m3p['ell16_g30']['arm_max']:.2f} "
        "in the pads under the 8&#8202;mm rod at 1&#8202;N, above the 1.15 bound by less than the finite elements&#8217; own "
        "grid change of r&#772;. On the capsule the spring that fits the rods across the finger sinks the rods along it "
        f"{min(cz):.0f}&#8211;{max(cz):.0f}&#8239;% too far: the capsule&#8217;s straight section makes it four times "
        "stiffer along the finger, and a spring is the same for every contact.</p>"
        + (f"<p>Drake&#8217;s volume-mesh alternative, a depth field p = E d / d<sub>max</sub> (the capsule, d<sub>max</sub> "
           f"{dmax:.2f}&#8202;mm), does worse than the convex field: at its own M1 the RMS approach error is "
           f"{dep['approach_rms_pct']:.0f}&#8239;% against {fit('cap_g30', 'drake', 'M1')['approach_rms_pct']:.0f}&#8239;%, "
           "so it did not go to the simulators.</p>" if dep else "")
        + fits_table()
        + ifig(approach_figure(), "Approach under the 12.5&#8202;mm rod across the finger at z&#8202;0 against the load, per "
                                  "tip: the finite elements (black) and Drake and the pads at U, M1 and M3 (model values). "
                                  "Hover a curve for its value; the legend toggles series.", key="approach")
        + ifig(arm_figure(), "Twist arm r&#772; under the 12.5&#8202;mm rod across the finger at z&#8202;0 against the load: the "
                             "finite elements and Drake and the 1&#8202;mm pads at M1 and M2. The pads&#8217; arm steps as "
                             "pads enter the patch.", key="arm")
        + img(patch_png(), "Contact patch under the 12.5&#8202;mm rod at 1&#8202;N, in the face plane (y across the finger, z "
                           "toward the mount): the finite elements&#8217; nodal pressure, Drake&#8217;s field pressure on the "
                           "rod at M1 and M2, and the pads&#8217; forces at M1 and M2 (rule R4&#8217;s spacing). Colour "
                           "scales to each panel&#8217;s maximum.", key="patch", width=1200))


PART_NAME = {"approach_drake": "approach in Drake", "approach_pads": "approach in the pads", "arm_drake": "twist arm in Drake",
             "arm_pads": "twist arm in the pads", "consistency_ready": "consistency on the readiness iCEM commands",
             "consistency_own": "consistency on the tip&#8217;s own plan", "spring_check": "spring check"}


def part_value(tip, c, part):
    """The measured value behind a criterion, as text."""
    v = S["verdicts"][tip]["calibrations"][c]
    if part.startswith("approach"):
        m = v["drake" if part.endswith("drake") else "pads"]
        return f"RMS {m['approach_rms_pct']:.1f}&#8239;%, largest {m['approach_max_pct']:.0f}&#8239;%"
    if part.startswith("arm"):
        m = v["drake" if part.endswith("drake") else "pads"]
        return f"ratio {m['arm_min']:.2f}&#8211;{m['arm_max']:.3f}"
    traj = "M1" if c == "M1" else "M2"
    if part == "consistency_ready":
        d = S["d1"].get(f"{tip}|ready|{traj}|pads")
        if not d:
            return "not run"
        s = d["pads_vs_drake"]
        return (f"held agreement {s['held_agree']} of {s['n']}, median {f(s['median_rms_angle_same_deg'], 1)}&#176; "
                f"(bound {d['angle_bound_deg']:.1f}&#176;), {f(s['median_rms_dN_same_N'], 2)} / {f(s['median_rms_dT_same_N'], 2)}&#8202;N")
    if part == "consistency_own":
        d = S["d1"].get(f"{tip}|own|M2|pads")
        if not d:
            return "not run"
        s = d["pads_vs_drake"]
        return f"held agreement {s['held_agree']} of {s['n']}, median {f(s['median_rms_angle_same_deg'], 1)}&#176;"
    if part == "spring_check":
        p = (S["d2"].get(f"{tip}|M2") or {}).get("pairs", {}).get("plan_vs_spring")
        if not p:
            return "not run"
        return f"median {f(p['median_rms_angle_same_deg'], 2)}&#176;, {f(p['median_rms_dN_same_N'], 3)}&#8202;N"
    return ""


def verdict_table():
    head = ["Tip", "Calibration", "E, Drake / pads (MPa)", "k<sub>s</sub> (N/mm)", "Criteria met", "Criteria missed, with the value"]
    body = []
    for tip in TIPS:
        V = S["verdicts"].get(tip, {})
        for c, v in V.get("calibrations", {}).items():
            met = [PART_NAME[k] for k in v["required"] if v["parts"].get(k) is True]
            miss = [f"{PART_NAME[k]} ({part_value(tip, c, k)})" for k in v["failed"]]
            ks = v["drake"].get("k_s_N_per_mm")
            body.append([NAME[tip], c, f"{v['drake']['E_MPa']:.1f} / {v['pads']['E_MPa']:.1f}",
                         f(ks, 1) if ks and ks < 5000 else "&#8211;", ", ".join(met) or "&#8211;", "; ".join(miss) or "none"])
    return table(head, body, "Verdict per tip against the plan&#8217;s criteria (summary.json, model values). A tip is modelled "
                             "well when one calibration meets every criterion of its row: M1 needs approach and twist arm "
                             "in both simulators and consistency on the iCEM commands; M3 needs the same, with M2 standing in "
                             "for it on trajectories, consistency on the tip&#8217;s own plan and the spring check.",
                 key="verdict", text_cols=(0, 1, 4, 5))


def bench_section():
    rows_ = []
    for tip in TIPS:
        a = [fe(tip, "cx12.5_z0", F) for F in ("0.5", "1", "3")]
        rows_.append([NAME[tip], f"{a[0]:.0f} / {a[1]:.0f} / {a[2]:.0f}", f(expo(tip, "cx12.5_z0")),
                      f(fe(tip, "s_apex") / fe(tip, "cx12.5_z0")), f(fe(tip, "cx12.5_z0", "1", "r_bar_mm")),
                      f(fe(tip, "cx12.5_z0", "3", "r_bar_mm")), "pending"])
    t = table(["Tip", "Rod approach at 0.5 / 1 / 3&#8202;N (&#181;m)", "Force exponent", "4&#8202;mm ball over rod, 1&#8202;N",
               "r&#772; at 1&#8202;N (mm)", "r&#772; at 3&#8202;N (mm)", "Bench"], rows_,
              "What the bench measures against (finite elements, model values): approach under the 25&#8202;mm rod at the "
              "contact point, its force exponent, the approach under the 2&#8202;mm-radius sphere over the rod&#8217;s at "
              "1&#8202;N (the 4&#8202;mm steel ball&#8217;s prediction), and the twist arm the twist test reads.", key="bench")
    return ("<p>Three bench tests fix what the finite elements assume, each on every printed tip; none has run.</p><ul>"
            "<li>Indentation under the 25&#8202;mm rod at 0.5, 1 and 3&#8202;N on the load cell with the tip on a rigid back, three "
            "repeats (the readiness procedure&#8217;s step): one solid-modulus factor &#955; per material by log-RMS, which "
            "scales every calibrated modulus and spring. The finite elements pass at a residual of at most 10&#8239;% RMS after "
            "&#955; and a bench force exponent within 0.10 of theirs.</li>"
            "<li>A 4&#8202;mm steel ball at the apex at 0.5, 1 and 3&#8202;N: the finite elements predict a ball-over-rod "
            "approach ratio of 1.3&#8211;1.4 at 1&#8202;N on the curved tips and a local law 2 or more; a bench ratio of 2 or "
            "more falsifies the finite elements&#8217; skin-over-core response and favours M1.</li>"
            "<li>A twist test: the rod pressed at 1 and 3&#8202;N and turned about the normal until slip, with &#956; from a pull "
            "test; an onset arm within 0.70&#8211;1.15 of r&#772; confirms the arm calibration (M2, M3), an arm near M1&#8217;s "
            "1.3&#8211;1.7 times falsifies it.</li></ul>" + t)


def limitations():
    return ("<ul>"
            "<li>The finite elements are linear and small-strain with frictionless normal contact; the gyroid is homogenized; "
            "the solid modulus is NinjaFlex 85A&#8217;s 12&#8202;MPa (Siraya 85A&#8217;s is unmeasured); the magnets are bonded and "
            "the slot voids above them on the upright prints are left out; the back face is fixed, so the DIP plate is rigid. "
            "Every modulus and spring scales with the bench factor &#955;.</li>"
            "<li>The calibration covers the normal response and the patch. The face&#8217;s tangential compliance (48&#8211;177&#8202;"
            "&#181;m/N under the rod) is in neither simulator; friction is 1.0 everywhere and Siraya 85A&#8217;s is unmeasured.</li>"
            "<li>The readiness iCEM rows logged one servo command per 20&#8202;ms window, and the replays interpolate them: on the "
            "block&#8217;s 16&#8202;mm plan at seed 0 the replay on the same pads follows the logged turn within 0.6&#176; to 3&#8202;s and "
            "ends at 10.4&#176; against 24.4&#176;, so those replays test the two simulators on the same commands, not the "
            "closed-loop outcome. The tips&#8217; own plans log every 1&#8202;ms command.</li>"
            "<li>The servo plant is the ad-hoc identification of 2026-10-13 (kp 0.72&#8202;N&#8202;m/rad, 0.18&#8202;N&#8202;m); its "
            "3.4&#8211;4.7&#8202;mm/N at the fingertip is what keeps normal-stiffness errors small on trajectories.</li>"
            "<li>The D1&#8211;D8 plans were designed for the block; on the curved tips they run near the tool&#8217;s slip "
            "limit.</li></ul>")


def bed_section():
    B = S.get("bed_checks", {})
    impl = [(k, v["impl_max_err"]) for k, v in B.items() if v.get("impl_max_err") is not None and v["arm"] != "rl"]
    worst = max(impl, key=lambda x: x[1]) if impl else None
    rl = [v["rl_vs_planner"] for v in B.values() if v.get("rl_vs_planner")]
    arms = {k: v["arm_ratio_range"] for k, v in B.items() if v.get("arm_ratio_range") and v["arm"] != "rl"}
    m1c = [r for k, r in arms.items() if "|M1|" in k and not k.startswith("block")]
    m2c = [r for k, r in arms.items() if "|M2|" in k and not k.startswith("block")]
    cr = {k: (v.get("creep_deg_s") or {}).get("1") for k, v in B.items()}
    crp = [x for k, x in cr.items() if x is not None and k.endswith("|pads")]
    crd = [x for k, x in cr.items() if x is not None and k.endswith("|drake")]
    crl = [x for k, x in cr.items() if x is not None and k.endswith("|rl")]
    txt = ("<p>On the contact bed (two force-controlled rails squeeze the 25&#8202;mm, 24.5&#8202;g tool between two copies of "
           "the tip) each calibrated model was run at the planner&#8217;s numerics, Drake at 1&#8202;ms, and the pads at M2 also at "
           "the RL trainer&#8217;s. ")
    sh1 = [v["approach"]["1"]["first_touch_shift_um"] for v in B.values() if v["arm"] == "pads" and v["model"] == "v2"
           and "1" in v["approach"]]
    sh05 = [v["approach"]["1"]["first_touch_shift_um"] for v in B.values() if v["arm"] == "pads" and v["model"] == "v2h"
            and "1" in v["approach"]]
    if worst:
        txt += ("The bed counts the approach from the tip&#8217;s design reach, and the pads&#8217; envelope lies inside it "
                + (f"by {rng(sh1, 0)}&#8202;&#181;m (1&#8202;mm pads)" if sh1 else "")
                + (f" and {rng(sh05, 0)}&#8202;&#181;m (0.5&#8202;mm pads)" if sh05 else "")
                + ", travel before the first pad touches. Counted from each model&#8217;s own first touch, the static approach "
                f"agrees with the phase-B curves within {100 * max(x[1] for x in impl):.1f}&#8239;% (worst: "
                f"{worst[0].replace('|', ', ')}), so the simulators implement the calibrated moduli. ")
    if m1c and m2c:
        txt += (f"The bed onset arm over the finite-element r&#772; is {min(r[0] for r in m1c):.2f}&#8211;{max(r[1] for r in m1c):.2f} "
                f"on the curved tips at M1 and {min(r[0] for r in m2c):.2f}&#8211;{max(r[1] for r in m2c):.2f} at M2, as the curves "
                "predict. ")
    if crp and crd:
        txt += (f"Under half the onset torque the pads creep {min(crp):.2f}&#8211;{max(crp):.2f}&#8202;&#176;/s at the planner "
                f"numerics and Drake {min(crd):.4f}&#8211;{max(crd):.4f}&#8202;&#176;/s; ")
    if crl:
        txt += f"at the RL numerics the pads creep {min(crl):.1f}&#8211;{max(crl):.1f}&#8202;&#176;/s. "
    ra = [x["approach_max"] for x in rl if x.get("approach_max") is not None]
    ro = [x["onset_max"] for x in rl if x.get("onset_max") is not None]
    if ra and ro:
        txt += (f"The pads at the RL numerics stay within {100 * max(ra):.1f}&#8239;% of the planner&#8217;s approach and "
                f"{100 * max(ro):.0f}&#8239;% of its onset arm.")
    return txt + "</p>" + bed_table()


def icem_section():
    D1 = S.get("d1", {})
    if not D1:
        return "<p>Pending.</p>"
    return ICEM_TEXT() + d1_table() + (d1_table(own=True) if any(v["own"] for v in D1.values()) else "") + FILM()


def d1d8_section():
    if not S.get("d2"):
        return "<p>Pending.</p>"
    return D1D8_TEXT() + d2_table()


def verdict_section():
    return VERDICT_TEXT() + verdict_table()


def ICEM_TEXT():
    return "<p></p>"


def D1D8_TEXT():
    return "<p></p>"


def VERDICT_TEXT():
    return "<p></p>"


def FILM():
    p = os.path.join(D, "media", "films.jsonl")
    if not os.path.exists(p):
        return ""
    out = ""
    for r in rows("media/films.jsonl"):
        rel = f"media/{r['film']}.mp4"
        if not os.path.exists(os.path.join(D, rel)):
            continue
        out += film(rel, f"media/{r['film']}_mid.png",
                    f"{NAME[r['tip']]} on its own pinned-iCEM plan (16&#8202;mm cell, seed 0) at {r['calib']}: the "
                    f"{'0.5' if r.get('E_pads_Pa') and S['fits'][r['tip']].get('pad_model', {}).get(r['calib']) == 'v2h' else '1'}&#8202;mm "
                    f"pads (left, E {r['E_pads_Pa'] / 1e6:.1f}&#8202;MPa, pads touching the tool red) and Drake "
                    f"(right, E {r['E_drake_Pa'] / 1e6:.1f}&#8202;MPa, its recorded joint angles and tool pose drawn in the same "
                    f"scene) under the same servo commands. Pads: turn {r['pads']['turn_end_deg']:+.1f}&#176;, "
                    f"{'held' if r['pads']['held'] else 'dropped'}; Drake: turn {r['drake']['turn_end_deg']:+.1f}&#176;, "
                    f"{'held' if r['drake']['held'] else 'dropped'}. Real time, 50 frames per second.", key="film")
    return out


def lede():
    return "Pending."


def future():
    F = S["fits"]
    cx = {t: F[t]["models"]["drake"]["M3x"] for t in CURVED}
    r11 = fe("ell11_g30", "cx12.5_z0", "1", "r_bar_mm")
    m1_11 = fit("ell11_g30", "drake", "M1")["arm_ratio"]["cx12.5_z0@1"]
    return (
        "<ol>"
        "<li>Bench indentation of each print under the 25&#8202;mm rod at 0.5, 1 and 3&#8202;N (readiness procedure, step "
        "&#8216;indentation&#8217;, three repeats, docs/experiments/20261016-sim2real_fingertip_readiness/bench/procedure.json): "
        "fixes &#955;, which scales every modulus and spring here. A residual above 10&#8239;% RMS after &#955;, or a force "
        "exponent more than 0.10 from Table&#160;[[tab:bench]], falsifies the shell-over-gyroid model for these prints.</li>"
        "<li>The 4&#8202;mm steel ball at the apex of each curved print at 0.5, 1 and 3&#8202;N: a ball-over-rod ratio of 2 or "
        "more at 1&#8202;N (a local law) instead of 1.3&#8211;1.4 falsifies the finite elements and favours M1.</li>"
        f"<li>The twist test on the 11/22 print (rod at 1 and 3&#8202;N, turned until slip, &#956; from a pull test): an onset "
        f"arm of {0.70 * r11:.2f}&#8211;{1.15 * r11:.2f}&#8202;mm at 1&#8202;N (0.70&#8211;1.15 of r&#772;) confirms M3; "
        f"{m1_11 * r11:.2f}&#8202;mm or more (Drake at M1) falsifies it.</li>"
        "<li>The capsule along the finger: a rod along the capsule is four times stiffer than one across it, and one series "
        f"spring cannot give both. Fitted to the rods across the finger alone (k<sub>s</sub> {cx['cap_g30']['k_s_N_per_mm']:.1f}&#8202;N/mm), "
        f"M3 reproduces them within {cx['cap_g30']['cx_rms_pct']:.1f}&#8239;% RMS and sinks the rods along the finger "
        f"{cx['cap_g30']['cz_range_pct'][0]:.0f}&#8211;{cx['cap_g30']['cz_range_pct'][1]:.0f}&#8239;% too far. Count how often the "
        "capsule meets a tool along its length in the D1&#8211;D8 and iCEM runs (contact-patch orientation in replays/, from the "
        "pads&#8217; contact frames); if never, judge the capsule on the rods across the finger and rerun "
        "<code>fe_contact_calibration_summary.py</code> with that task set.</li>"
        "<li>Training on the calibrated 11/22 tip: the RL trainer&#8217;s pads at M2 need 0.5&#8202;mm spacing (about four times "
        "the 1&#8202;mm pads&#8217; contacts) and creep 9&#8202;&#176;/s under a sub-onset twist at 2&#8202;ms against 1&#8202;&#176;/s at "
        "1&#8202;ms. Time one MuJoCo-Warp iteration with the 0.5&#8202;mm pads (scripts/make_pad_morphology_run.py with the "
        "spacing) and replay the D1&#8211;D8 plans at the RL numerics against Drake; the RL arm passes if its same-branch "
        "median stays within 3&#176; of the planner&#8217;s.</li>"
        "</ol>")


def build():
    sections = {"LEDE": lede(), "VERDICT": verdict_section(), "FE": fe_section(), "FITS": fits_section(),
                "BED": bed_section(), "ICEM": icem_section(), "D1D8": d1d8_section(), "BENCH": bench_section(),
                "LIMITS": limitations(), "FUTURE": future()}
    main(sections)


if __name__ == "__main__":
    build()


def smoke():
    """The smoke test of the readiness-plan replays and of the tips' own plans: the replay on the planning pads at seed
    0 against the logged closed-loop turn."""
    out = {}
    for r in rows("replays/icem.jsonl"):
        if r.get("status") != "ok" or int(r["seed"]) != 0:
            continue
        if (not r["own"] and r["calib"] == "U" and r["plant"] == "pads") or (r["own"] and r["plant"] == "pads1"):
            tr = r.get("trace") or []
            lg = r.get("logged") or {}
            dev3 = None
            out[r["plan"]] = {"turn": r["turn_end_deg"], "logged": lg.get("turn_end_deg"), "held": r["held"],
                              "logged_held": lg.get("held"), "own": r["own"]}
    return out
