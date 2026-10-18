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
.pass{color:#008800} .fail{color:#cc0000}
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
RB, KS = r"\(\bar r\)", r"\(k_s\)"


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


def PF(ok):
    """A pass/fail cell, capitalized and coloured."""
    return f"<span class='{pf(ok)}'>{pf(ok).capitalize()}</span>"


def cell_of(plan):
    return "16" if ":16:" in plan else "25"


def pairs_by_cell(d):
    """{cell diameter: [pairs]} of a D1 entry."""
    out = {}
    for p_ in (d or {}).get("pairs", []):
        out.setdefault(cell_of(p_["plan"]), []).append(p_)
    return out


def grid_change():
    """(approach, r_bar) relative changes of the capsule on the 0.2 mm contact grid against the base grid."""
    dap, darm = [], []
    for p_ in ("s_apex", "cx5_z0", "cx8_z0", "cx12.5_z0", "cz8_y0", "cz12.5_y0"):
        r = S["fe"].get(f"cap_g30|fine|{p_}")
        if r:
            for F_ in ("0.5", "1", "3"):
                dap.append(abs(r["per_force"][F_]["approach_um"] / fe("cap_g30", p_, F_) - 1))
                if p_.startswith("cx"):
                    darm.append(abs(r["per_force"][F_]["r_bar_mm"] / fe("cap_g30", p_, F_, "r_bar_mm") - 1))
    return dap or [0.0], darm or [0.0]


def expo_task(tips):
    """Finite-element force exponents of the tips under the task cylinders (fit/fits.json)."""
    return [x for t in tips for x in S["fits"][t]["fe"]["exponent_task"].values()]


def local_expo(model, tips=TIPS):
    """Force exponents of a model's phase-B curves under the task cylinders: the curves scale with E, so the exponent
    is the same at every modulus."""
    out = []
    for t in tips:
        for p_ in TASK:
            a, b = model_at(t, model, p_, 1e7, 0.5)[0], model_at(t, model, p_, 1e7, 3.0)[0]
            out.append(math.log(b / a) / math.log(6))
    return out


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
                    E, ks = fit(tip, m, "M3")["E_MPa"] * 1e6, t["k_s_N_per_mm"] * 1e3
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
                cc = (S["fits"][tip].get("calib_of") or {}).get(c, c)
                E = fit(tip, m, cc)["E_MPa"] * 1e6
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
    ("Force exponent n", r"the slope of approach against load on log axes between 0.5 and 3&#8202;N, "
                         r"\(n = \ln(a_{3}/a_{0.5}) / \ln 6\) with \(a_F\) the approach at F newtons; a linear spring "
                         "gives 1, a local law 0.5 under a point contact and 2/3 under a line contact"),
    (r"Twist arm \(\bar r\)", r"the pressure-weighted mean distance of the contact pressure from its centre in the face "
                              r"plane, in mm: \(\bar r = \sum_i p_i r_i / \sum_i p_i\); the full-slip torque about the "
                              r"contact normal is \(\mu N \bar r\). Arm ratio: model \(\bar r\) over finite-element "
                              r"\(\bar r\) under the same probe and load"),
    ("Approach error", "model approach over finite-element approach minus 1, over the task cylinders at 0.5, 1 and 3&#8202;N; "
                       "RMS and largest absolute value, in %; positive when the model sinks too far"),
    ("E", r"the contact modulus. Drake: the hydroelastic modulus of the tip&#8217;s compliant convex hull, "
          r"\(p = E\,(1-g)\) with g the gauge about the hull&#8217;s volume centroid (0 on the surface, 1 at the centroid). "
          r"Pads: \(K_i = E\,a_i/h_i\), \(a_i\) the pad&#8217;s area share and \(h_i\) its depth below the volume "
          "centroid (builder v2)"),
    (r"\(k_s\)", r"series spring between the DIP link and the tip&#8217;s contact geometry along the palmar normal, in N/mm: "
                 r"approach \(= F/k_s\) + the local approach at the modulus behind it"),
    ("U, M1, M2, M3", r"the calibrations. U: the readiness moduli of 2026-10-16 (block 7.55, capsule 13.45, 16/32 "
                      r"15.01, 11/22 14.23&#8202;MPa, the same E in both simulators). M1: one E per tip and simulator "
                      r"fitted (log-RMS) to the finite-element approach under the task cylinders at 0.5&#8211;3&#8202;N. M2: one "
                      r"E fitted to the finite-element \(\bar r\) under the 8 and 12.5&#8202;mm rods across the finger at "
                      r"z&#8202;0, 1 and 3&#8202;N. M3: M2&#8217;s E behind a series spring \(k_s\) fitted to the approach, one "
                      r"\(k_s\) per tip (the mean of the Drake and 1&#8202;mm-pad fits); on trajectories M2 stands in for M3 "
                      "when the spring check passes. All model values"),
    ("Rules R3, R4", r"the plan&#8217;s decision rules. R3: M2 equals M1 on a tip whose \(\bar r\) changes by less than "
                     r"5&#8239;% over half to twice the modulus. R4: spheres of radius 0.375&#8202;mm on a 0.5&#8202;mm grid "
                     "replace the 1&#8202;mm pads where fewer than six 1&#8202;mm pads touch the 8 or 12.5&#8202;mm rod at "
                     "1&#8202;N"),
    ("Pads", "MuJoCo spheres of radius 0.75&#8202;mm on a 1&#8202;mm grid over the tip&#8217;s surface (builder v2), or "
             "the 0.5&#8202;mm pads of rule R4"),
    ("Phase-B curves", r"each model&#8217;s force, \(\bar r\) and pads in contact against approach under the 18 "
                       "finite-element probes, computed once per tip at a reference modulus; every force scales with E"),
    ("Numerics", "planner and evaluation: MuJoCo 1&#8202;ms, impratio 100, Newton 100 iterations; RL trainer: 2&#8202;ms, "
                 "impratio 10, 10 iterations; Drake 1.57: SAP at 1&#8202;ms, relaxation time 0.01&#8202;s"),
    ("Onset arm", r"a simulator&#8217;s twist-onset torque on the bed (two tips squeeze the 25&#8202;mm tool, torque ramp "
                  r"about the pinch axis) over \(\mu N\), in mm"),
    ("Creep", "steady twist rate of the bed tool under a held torque at half the onset torque, over 1&#8202;s, in "
              "&#176;/s; rigid Coulomb friction gives 0"),
    ("Pinned iCEM, cell", "Pinned iCEM: the scale study&#8217;s sampling planner (improved cross-entropy method, 64 rollouts "
                          "of 0.5&#8202;s replanned every 20&#8202;ms, a cost that keeps three pads loaded), whose first knot is "
                          "fixed to the servo command in force. Cell: one finger layout and one cylinder (100&#8202;mm long, "
                          "25&#8202;g, on a 3&#8202;mm post) on the CAD hand; the 16&#8202;mm cell has a cylinder of 16&#8202;mm "
                          "diameter, the 25&#8202;mm cell one of 25&#8202;mm"),
    ("D1&#8211;D8", "the eight real_v1 hands built and run on the bench in September 2026, each with its deployed open-loop "
                    "plan for the 25&#8202;mm screwdriver; the plans were designed for the block"),
    ("Held", "the tool less than 20&#8202;mm below its start and at least two fingers each pressing with its weight at "
             "the end of the run"),
    ("RMS angle", "RMS over a run&#8217;s 20&#8202;ms samples of the angle between two runs&#8217; tool orientations (on the "
                  "CAD-hand cells between the cylinders&#8217; axes, since a cylinder is symmetric about its axis), in &#176;"),
    ("Same branch", "both runs keep the tool and end within 15&#176; of each other: on the CAD-hand cells the angle between "
                    "the two cylinders&#8217; axes at the last sample, on the D1&#8211;D8 replays the difference of the two "
                    "turns"),
    ("Seed spread", "the median RMS angle between one model&#8217;s runs from different start placements (seeds 1 and 2 "
                    "against seed 0), over same-branch pairs"),
    ("Force differences", "RMS over samples and fingers of the difference between two runs of each finger&#8217;s summed "
                          "normal force on the tool, and of the magnitude of its friction force, in N"),
    ("Spring check, check arms", "Spring check: the D1&#8211;D8 replays of the 11/22 tip&#8217;s pads at M2 behind the M3 "
                                 "slide spring against the same pads without it; M2 stands in for M3 when the same-branch "
                                 "median RMS angle is at most 1&#176; and the normal-force difference at most 0.05&#8202;N. "
                                 "Check arms: D1&#8211;D8 replays at U with the pads at impratio 1000 (creep) and at "
                                 "0.5&#8202;mm spacing (discreteness)"),
    ("Bench factor &#955;", "Siraya 85A&#8217;s solid modulus over the finite elements&#8217; 12&#8202;MPa, fixed by the bench "
                            "indentation test; every calibrated modulus and spring scales with it"),
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
            rf"Twist arm {RB} under the rod, 1 / 3&#8202;N (mm)"]
    body = []
    for tip in TIPS:
        sp = " / ".join(f(fe(tip, "s_apex", F), 0) for F in ("0.5", "1", "3", "6"))
        rd = " / ".join(f(fe(tip, "cx12.5_z0", F), 0) for F in ("0.5", "1", "3", "6"))
        body.append([NAME[tip], sp, rd, f(fe(tip, "cz12.5_y0"), 0), f"{f(expo(tip, 's_apex'))} / {f(expo(tip, 'cx12.5_z0'))}",
                     f(fe(tip, "cx12.5_z0") / fe(tip, "s_apex")),
                     f"{f(fe(tip, 'cx12.5_z0', '1', 'r_bar_mm'))} / {f(fe(tip, 'cx12.5_z0', '3', 'r_bar_mm'))}"])
    return table(head, body, "Finite elements of the four prints in Siraya 85A at 30&#8239;% gyroid with four 3&#8202;&#215;&#8202;2&#8202;mm "
                             "magnets (fe/fe.jsonl, model values): approach under the 2&#8202;mm-radius sphere at the apex and "
                             "under the 12.5&#8202;mm rod across and along the finger, the force exponent between 0.5 and "
                             "3&#8202;N, the rod-over-sphere approach ratio and the twist arm under the rod across the finger.",
                 key="fe", text_cols=tuple(range(7)))


def fits_table():
    head = ["Tip", "Calibration", rf"{KS} (N/mm)", "Drake: E (MPa)", "Drake: approach error, RMS / largest (%)",
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
                 text_cols=tuple(range(9)))


def bed_table():
    head = ["Tip", "Calibration", "Model", "E (MPa)", "Bed approach at 1&#8202;N (&#181;m)",
            "Against the curve from first touch, largest (%)", "Onset arm at 1 / 3&#8202;N (mm)",
            rf"Over finite-element {RB}", "Creep at 1&#8202;N (&#176;/s)"]
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
                             rf"finite-element {RB} under the 12.5&#8202;mm rod at 1 and 3&#8202;N (band 0.70&#8211;1.15), and the "
                             "creep under half the onset torque.", key="bed", text_cols=tuple(range(9)))


def d1_table(own=False):
    head = ["Tip", "Calibration", "Pads", "Runs held, pads / Drake", "Held agreement", "Same branch",
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
                np_ = sum(1 for r in v["runs"] if r["plant"] != "drake")
                nd_ = sum(1 for r in v["runs"] if r["plant"] == "drake")
                body.append([NAME[tip], c, spacing, f"{v['held_pads']} of {np_} / {v['held_drake']} of {nd_}",
                             f"{s['held_agree']} of {s['n']}", f"{s['same_branch']} of {s['n']}",
                             f(s["median_rms_angle_same_deg"], 2), f(sp["median_rms_angle_same_deg"], 2),
                             f"{f(s['median_rms_dN_same_N'], 3)} / {f(s['median_rms_dT_same_N'], 3)}", PF(v["pass"])])
    cap = ("Pads against Drake on the same servo commands: the pinned-iCEM runs of the readiness study on the 16&#8202;mm and "
           "25&#8202;mm cells, replayed open loop at three start placements (replays/icem.jsonl, model values). Consistency passes "
           "with held agreement on at least 5 of 6 runs, a median same-branch RMS angle of at most 3&#176; or the pads&#8217; "
           "seed spread, whichever is larger, and median force differences of at most 0.25&#8202;N normal and 0.35&#8202;N "
           "friction. Drake ran on the 16&#8202;mm cell alone at U, and at seed 0 alone on the 16/32 tip."
           if not own else
           "Pads against Drake on each tip&#8217;s own plan: pinned iCEM re-planned on the tip&#8217;s 1&#8202;mm pads at M2 on the "
           "16&#8202;mm cell, replayed open loop at three start placements. Consistency passes with held agreement on 3 of 3 "
           "runs and the same angle bound.")
    return table(head, body, cap, key="d1own" if own else "d1", text_cols=tuple(range(10)))


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
                if p and not p["n"]:
                    p = None
                if arm == "plan":
                    row = [NAME[tip], c, ARM_NAME[arm], hs, "Seed spread" if p else "&#8211;",
                           f"{p['same_branch']} of {p['n']}" if p else "&#8211;",
                           f(p["median_rms_angle_same_deg"], 2) if p else "&#8211;", "&#8211;"]
                else:
                    row = [NAME[tip], c, ARM_NAME[arm], hs, f"{p['held_agree']} of {p['n']}" if p else "&#8211;",
                           f"{p['same_branch']} of {p['n']}" if p else "&#8211;",
                           f(p["median_rms_angle_same_deg"], 2) if p else "&#8211;",
                           f"{f(p['median_rms_dN_same_N'], 3)} / {f(p['median_rms_dT_same_N'], 3)}" if p else "&#8211;"]
                body.append(row)
    return table(head, body, "The eight deployed bench plans D1&#8211;D8 replayed open loop on each tip (replays/d1d8.jsonl, model "
                             "values): held counts per start placement, and each arm against the planner pads at the same "
                             "calibration, hand and seed. For the planner pads the row gives their own seed spread where "
                             "seeds 1 and 2 ran.", key="d2", text_cols=tuple(range(8)))


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
    if not vals:
        return "&#8211;"
    a, b = f"{min(vals):.{n}f}", f"{max(vals):.{n}f}"
    return (a if a == b else f"{a}&#8211;{b}") + unit


CURVED = ["cap_g30", "ell16_g30", "ell11_g30"]


def fe_section():
    reg = [fe("block4_g30", "cx12.5_z0", F) for F in ("0.5", "1", "3")]
    old = [31.94, 60.76, 162.25]
    dap, darm = grid_change()
    td = S["fe"].get("block_td|base|cx12.5_z0")
    td_txt = ""
    if td:
        a = [td["per_force"][F]["approach_um"] for F in ("0.5", "1", "3")]
        td_txt = (f" The readiness study&#8217;s own block surface (side fillets 2.7 instead of 3&#8202;mm) on this grid sinks "
                  f"{a[0]:.1f}, {a[1]:.1f} and {a[2]:.1f}&#8202;&#181;m, so the grid accounts for "
                  f"{100 * (a[1] / old[1] - 1):.0f}&#8239;% and the surface for the rest.")
    sh = {t: (fe(t, "cx12.5_z0", "1", "shear_y_face_um_per_N"), fe(t, "cx12.5_z0", "1", "shear_z_face_um_per_N")) for t in TIPS}
    base = [m for m in S.get("fe_meta", []) if m["grid"] == "base"]
    fine = meta_of("cap_g30", "fine")
    dof = rng([m["n_dof"] / 1e3 for m in base], 0)
    gb = rng([m["peak_rss_GB"] for m in base], 1)
    ec = expo_task(CURVED)
    es = [expo(t, "s_apex") for t in CURVED]
    eb = expo_task(["block4_g30"])
    cap_long = 2 * fe("cap_g30", "cz12.5_y0", "1", "half_z_mm")
    cap_across = 2 * fe("cap_g30", "cx12.5_z0", "1", "half_z_mm")
    return (
        "<p>Each print was modelled as sliced: two 0.45&#8202;mm perimeters and four 0.2&#8202;mm skins of solid TPU around "
        "every outline, the magnet pockets&#8217; included, over the homogenized 30&#8239;% gyroid, in the print&#8217;s "
        "orientation (the block on its back face, the curved tips upright), with the four magnets bonded at the sheet "
        "depths (5, 5, 5.5 and 7&#8202;mm) and the back and mount faces fixed. The solid modulus is NinjaFlex 85A&#8217;s "
        "12&#8202;MPa, standing in for Siraya&#8217;s unmeasured one. Frictionless rigid probes were pressed along the palmar "
        f"normal from first touch of the design surface; the models have {dof}&#8202;k degrees of freedom and peak at "
        f"{gb}&#8202;GB.</p>"
        + fe_table() +
        f"<p>Under the 2&#8202;mm sphere at the apex every tip sinks {rng([fe(t, 's_apex') for t in TIPS], 0)}&#8202;&#181;m at "
        "1&#8202;N: the skin spreads the load over millimetres of gyroid, so the structure sets the approach and the "
        f"surface shape changes it by under 5&#8239;%. Under the 12.5&#8202;mm rod across the finger the curved tips sink "
        f"{rng([fe(t, 'cx12.5_z0') for t in CURVED], 0)}&#8202;&#181;m and the block {fe('block4_g30', 'cx12.5_z0'):.0f}&#8202;&#181;m. "
        rf"The approach grows as \(F^{{n}}\) with n {rng(ec)} under the task cylinders and {rng(es)} under the sphere on the "
        f"curved tips, and {rng(eb)} on the block, against 0.5 (point contact) to 2/3 (line contact) for a local law. "
        f"A rod laid along the finger sinks {fe('cap_g30', 'cx12.5_z0') / fe('cap_g30', 'cz12.5_y0'):.1f} times less than "
        f"one across it on the capsule ({fe('cap_g30', 'cz12.5_y0'):.1f} against {fe('cap_g30', 'cx12.5_z0'):.1f}&#8202;&#181;m at "
        f"1&#8202;N): it meets the capsule&#8217;s straight section over {cap_long:.1f}&#8202;mm, the rod across it over "
        f"{cap_across:.1f}&#8202;mm. On the half-ellipsoids the ratio is "
        f"{fe('ell16_g30', 'cx12.5_z0') / fe('ell16_g30', 'cz12.5_y0'):.1f} (16/32) and "
        f"{fe('ell11_g30', 'cx12.5_z0') / fe('ell11_g30', 'cz12.5_y0'):.1f} (11/22). The twist arm under the rod at 1&#8202;N is "
        f"{rng([fe(t, 'cx12.5_z0', '1', 'r_bar_mm') for t in CURVED])}&#8202;mm on the curved tips and "
        f"{fe('block4_g30', 'cx12.5_z0', '1', 'r_bar_mm'):.2f}&#8202;mm on the block, whose contact line spans its flat face. "
        "The face&#8217;s tangential compliance under the rod at 1&#8202;N is "
        f"{rng([v[0] for v in sh.values()], 0)}&#8202;&#181;m/N across the finger and {rng([v[1] for v in sh.values()], 0)}&#8202;&#181;m/N "
        "along it; neither simulator represents it.</p>"
        f"<p>Regression against the readiness study: the block sinks {reg[0]:.1f}, {reg[1]:.1f} and {reg[2]:.1f}&#8202;&#181;m "
        f"under the rod at 0.5, 1 and 3&#8202;N, {100 * (reg[1] / old[1] - 1):.0f}&#8239;% more than the 2026-10-16 rows "
        f"({old[0]}, {old[1]} and {old[2]}&#8202;&#181;m on a 259&#8202;k-dof grid with the readiness block surface), outside "
        "the plan&#8217;s 10&#8239;% bound." + td_txt + " Grid check (prediction P7): the capsule on a 0.2&#8202;mm "
        f"contact grid ({fine.get('n_dof', 0) / 1e3:.0f}&#8202;k degrees of freedom) changes the approach by at most "
        rf"{100 * max(dap):.1f}&#8239;% and {RB} by at most {100 * max(darm):.1f}&#8239;%.</p>")


def fits_section():
    F = S["fits"]
    m1d = [fit(t, "drake", "M1") for t in TIPS]
    m1p = [fit(t, "v2", "M1") for t in TIPS]
    m2d = [fit(t, "drake", "M2") for t in CURVED]
    m2e = [v for t in CURVED for v in fit(t, "drake", "M2")["approach_err_pct"].values()]
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
    r3 = [F["block4_g30"]["models"][m]["R3_rbar_change"] for m in ("drake", "v2")]
    n11 = F["ell11_g30"]["models"]["v2"]["M2"]["contacts_1N"]
    dark = local_expo("drake")
    eqs = (r"<div class='eq'>\[p = E\,(1 - g), \qquad K_i = \frac{E\,a_i}{h_i}, \qquad "
           r"a(F) = \frac{F}{k_s} + a_{\mathrm{loc}}(F;\,E_{\mathrm{loc}}), \qquad "
           r"\bar r = \frac{\sum_i p_i\,r_i}{\sum_i p_i}\]</div>")
    return (
        "<p>Both simulators use a local law: the pressure at a point of Drake&#8217;s field and the force of a pad depend "
        "only on the overlap there, so every force in the phase-B curves scales with E and one curve per model and probe "
        "gives every modulus. The curves were computed on each tip&#8217;s simulated surface (the as-printed hull) under "
        rf"the same 18 probes as the finite elements; a series spring adds \(F/k_s\) to the approach and leaves the "
        "patch unchanged:</p>" + eqs +
        "<p>M1, the one modulus fitted to the approach, gets the RMS approach error to "
        f"{rng([v['approach_rms_pct'] for v in m1d], 1)}&#8239;% in Drake and {rng([v['approach_rms_pct'] for v in m1p], 1)}&#8239;% "
        f"in the 1&#8202;mm pads, but the largest error is {rng([v['approach_max_pct'] for v in m1d], 0)}&#8239;% in Drake "
        f"and {rng([v['approach_max_pct'] for v in m1p], 0)}&#8239;% in the pads. Drake&#8217;s field sinks with a force "
        f"exponent of {rng(dark)} under the task cylinders, where the prints sink with {rng(expo_task(TIPS))}, so one "
        "modulus is too soft at 0.5&#8202;N and too stiff at 3&#8202;N; the largest error is the block&#8217;s, "
        f"{bz[bz_k]:.0f}&#8239;% too far under the {bz_k[2:].split('_')[0]}&#8202;mm rod along the finger at "
        f"{bz_k.split('@')[1]}&#8202;N. At M1 the curved tips get {min(fit(t, 'drake', 'M1')['arm_min'] for t in CURVED):.2f}&#8211;"
        f"{max(fit(t, 'drake', 'M1')['arm_max'] for t in CURVED):.2f} times the print&#8217;s twist arm in Drake "
        f"and up to {max(fit(t, 'v2', 'M1')['arm_max'] for t in CURVED):.2f} times in the pads, because the field&#8217;s "
        "patch under the rod is wider than the print&#8217;s (Figure&#160;[[fig:patch]]). The block keeps "
        f"{fit('block4_g30', 'drake', 'M1')['arm_min']:.2f}&#8211;{fit('block4_g30', 'drake', 'M1')['arm_max']:.2f} of its arm, "
        f"set by the width of its flat face; its arm changes by {100 * r3[0]:.1f}&#8239;% in Drake and {100 * r3[1]:.1f}&#8239;% "
        "in the pads over half to twice the modulus, so rule R3 sets its M2 to M1.</p>"
        f"<p>M2, fitted to the arm, needs {rng([v['E_MPa'] for v in m2d], 0)}&#8202;MPa in Drake and leaves the curved "
        f"tips {-max(m2e):.0f}&#8211;{-min(m2e):.0f}&#8239;% too stiff. M3 puts that modulus behind a series spring of "
        f"{ks['ell11_g30']:.1f} (11/22), {ks['ell16_g30']:.1f} (16/32) and {ks['cap_g30']:.1f}&#8202;N/mm (capsule). On the "
        f"11/22 half-ellipsoid it reproduces the approach within {m3d['ell11_g30']['approach_rms_pct']:.1f}&#8239;% RMS "
        f"(largest {m3d['ell11_g30']['approach_max_pct']:.0f}&#8239;%) and the arm within "
        f"{m3d['ell11_g30']['arm_min']:.2f}&#8211;{m3d['ell11_g30']['arm_max']:.2f} in Drake, and within "
        f"{m3p['ell11_g30']['approach_rms_pct']:.1f}&#8239;% and {m3p['ell11_g30']['arm_min']:.2f}&#8211;"
        f"{m3p['ell11_g30']['arm_max']:.2f} in the 0.5&#8202;mm pads that rule R4 requires (at M2 the 1&#8202;mm pads touch "
        f"the 8 and 12.5&#8202;mm rods with {n11['cx8_z0']:.0f} pads at 1&#8202;N). On the 16/32 half-ellipsoid the approach "
        f"passes ({m3d['ell16_g30']['approach_rms_pct']:.1f}&#8239;% in Drake, {m3p['ell16_g30']['approach_rms_pct']:.1f}&#8239;% in "
        f"the pads) and the arm reaches {m3d['ell16_g30']['arm_max']:.3f} in Drake and {m3p['ell16_g30']['arm_max']:.2f} "
        "in the pads under the 8&#8202;mm rod at 1&#8202;N, above the 1.15 bound. On the capsule the spring that fits the "
        f"rods across the finger sinks the rods along it {min(cz):.0f}&#8211;{max(cz):.0f}&#8239;% too far: a spring is the "
        f"same for every contact, and the print is "
        f"{fe('cap_g30', 'cx12.5_z0') / fe('cap_g30', 'cz12.5_y0'):.1f} times stiffer along the finger.</p>"
        + (f"<p>Drake&#8217;s volume-mesh alternative, a depth field \\(p = E\\,d/d_{{\\max}}\\) (the capsule, "
           f"\\(d_{{\\max}}\\) {dmax:.2f}&#8202;mm), does worse than the convex field: at its own M1 the RMS approach error is "
           f"{dep['approach_rms_pct']:.0f}&#8239;% against {fit('cap_g30', 'drake', 'M1')['approach_rms_pct']:.0f}&#8239;%, "
           "so it did not go to the simulators.</p>" if dep else "")
        + fits_table()
        + ifig(approach_figure(), "Approach under the 12.5&#8202;mm rod across the finger at z&#8202;0 against the load, per "
                                  "tip: the finite elements (black) and Drake and the 1&#8202;mm pads at U, M1 and M3 (model values). "
                                  "Hover a curve for its value; the legend toggles series.", key="approach")
        + ifig(arm_figure(), rf"Twist arm {RB} under the 12.5&#8202;mm rod across the finger at z&#8202;0 against the load: the "
                             "finite elements and Drake and the 1&#8202;mm pads at M1 and M2. The pads&#8217; arm steps as "
                             "pads enter the patch.", key="arm")
        + img(patch_png(), "Contact patch under the 12.5&#8202;mm rod at 1&#8202;N, in the face plane (y across the finger, z "
                           "toward the mount): the finite elements&#8217; nodal pressure, Drake&#8217;s field pressure on the "
                           "rod at M1 and M2, and the pads&#8217; forces at M1 and M2 (rule R4&#8217;s spacing). Colour "
                           "scales to each panel&#8217;s maximum.", key="patch", width=1200))






def verdict_table():
    head = ["Tip", "Calibration", "E, Drake / pads (MPa)", rf"{KS} (N/mm)", "Approach, Drake / pads",
            "Twist arm, Drake / pads", "Consistency, readiness plans / own plan", "Spring check"]
    body = []

    def pf2(ok, val):
        return ("Pass" if ok else "Fail") + (f" ({val})" if val and not ok else "")

    def cons(x):
        return "Not run" if x is None else ("Pass" if x else "Fail")
    for tip in TIPS:
        V = S["verdicts"].get(tip, {})
        for c, v in V.get("calibrations", {}).items():
            P_ = v["parts"]
            ks = v["drake"].get("k_s_N_per_mm")
            dr, pa = v["drake"], v["pads"]
            app = (pf2(P_["approach_drake"], f"{dr['approach_max_pct']:.0f}&#8239;% largest") + " / "
                   + pf2(P_["approach_pads"], f"{pa['approach_max_pct']:.0f}&#8239;% largest"))
            arm = pf2(P_["arm_drake"], f"{dr['arm_max']:.3f}") + " / " + pf2(P_["arm_pads"], f"{pa['arm_max']:.3f}")
            con = cons(P_.get("consistency_ready")) + (" / " + cons(P_.get("consistency_own")) if c == "M3" else "")
            spr = "&#8211;" if c != "M3" else cons(P_.get("spring_check"))
            body.append([NAME[tip], c, f"{v['drake']['E_MPa']:.1f} / {v['pads']['E_MPa']:.1f}",
                         f(ks, 1) if ks and ks < 1000 else "&#8211;", app, arm, con, spr])
    return table(head, body, "Verdict per tip against the plan&#8217;s criteria (summary.json, model values). A tip is modelled "
                             "well when one calibration passes every column of its row: M1 needs approach and twist arm in "
                             "both simulators and consistency on the readiness iCEM commands; M3 needs the same, with M2 "
                             "standing in for it on trajectories, consistency on the tip&#8217;s own plan and the spring check. "
                             "A failed approach gives the largest error, a failed arm the largest ratio (bound 1.15); Not "
                             "run marks a test without Drake replays.",
                 key="verdict", text_cols=(0, 1, 4, 5, 6, 7))


def bench_section():
    rows_ = []
    for tip in TIPS:
        a = [fe(tip, "cx12.5_z0", F) for F in ("0.5", "1", "3")]
        rows_.append([NAME[tip], f"{a[0]:.0f} / {a[1]:.0f} / {a[2]:.0f}", f(expo(tip, "cx12.5_z0")),
                      f(fe(tip, "s_apex") / fe(tip, "cx12.5_z0")), f(fe(tip, "cx12.5_z0", "1", "r_bar_mm")),
                      f(fe(tip, "cx12.5_z0", "3", "r_bar_mm")), "Pending"])
    m1r = [fit(t, m, "M1")["arm_ratio"][k] for t in CURVED for m in ("drake", "v2") for k in ("cx12.5_z0@1", "cx12.5_z0@3")]
    bo = [fe(t, "s_apex") / fe(t, "cx12.5_z0") for t in CURVED]
    t = table(["Tip", "Rod approach at 0.5 / 1 / 3&#8202;N (&#181;m)", "Force exponent", "4&#8202;mm ball over rod, 1&#8202;N",
               rf"Twist arm {RB} at 1&#8202;N (mm)", rf"Twist arm {RB} at 3&#8202;N (mm)", "Bench"], rows_,
              "What the bench measures against (finite elements, model values): approach under the 25&#8202;mm rod at the "
              "contact point, its force exponent, the approach under the 2&#8202;mm-radius sphere over the rod&#8217;s at "
              "1&#8202;N (the 4&#8202;mm steel ball&#8217;s prediction), and the twist arm the twist test reads.", key="bench")
    return ("<p>Three bench tests fix what the finite elements assume, each on every printed tip; none has run.</p><ul>"
            "<li>Indentation under the 25&#8202;mm rod at 0.5, 1 and 3&#8202;N on the load cell with the tip on a rigid back, three "
            "repeats (the readiness procedure&#8217;s step): one bench factor &#955; per material by log-RMS, which scales every "
            "calibrated modulus and spring. The finite elements pass at a residual of at most 10&#8239;% RMS after &#955; and a "
            "bench force exponent within 0.10 of theirs.</li>"
            f"<li>A 4&#8202;mm steel ball at the apex at 0.5, 1 and 3&#8202;N: the finite elements predict a ball-over-rod "
            f"approach ratio of {rng(bo, 1)} at 1&#8202;N on the curved tips and a local law 2 or more; a bench ratio of 2 or "
            "more falsifies the finite elements&#8217; skin-over-core response and favours M1.</li>"
            "<li>A twist test: the rod pressed at 1 and 3&#8202;N and turned about the normal until slip, with &#956; from a pull "
            rf"test; an onset arm within 0.70&#8211;1.15 of {RB} confirms the arm calibration (M2, M3), an arm of "
            f"{rng(m1r, 1)} times {RB}, the models&#8217; at M1, falsifies it.</li></ul>" + t)


def limitations():
    w = [r["wall_s"] for r in rows("replays/icem.jsonl") if r.get("status") == "ok" and r["plant"] == "drake"]
    return ("<ul>"
            "<li>The finite elements are linear and small-strain with frictionless normal contact; the gyroid is homogenized; "
            "the solid modulus is NinjaFlex 85A&#8217;s 12&#8202;MPa (Siraya 85A&#8217;s is unmeasured); the magnets are bonded and "
            "the slot voids above them on the upright prints are left out; the back face is fixed, so the DIP plate is rigid. "
            "Every modulus and spring scales with the bench factor &#955;.</li>"
            "<li>The calibration covers the normal response and the patch. The face&#8217;s tangential compliance (48&#8211;177&#8202;"
            "&#181;m/N under the rod) is in neither simulator; friction is 1.0 everywhere and Siraya 85A&#8217;s is unmeasured.</li>"
            "<li>The readiness iCEM rows logged one servo command per 20&#8202;ms window; the replays hold each command for "
            "10&#8202;ms at the interpolated value, in both simulators. They test the two simulators on the same commands; where "
            "a replay on the planning pads leaves the logged turn, it no longer reproduces the closed-loop run.</li>"
            f"<li>Runs cut for time: Drake took {min(w):.0f}&#8211;{max(w):.0f}&#8202;s per CAD-hand replay against the "
            "15&#8202;s planned. Drake replayed the readiness commands on the 11/22 tip at U (16&#8202;mm cell, three "
            "placements) and M2 (both cells, three placements) and on its own plan (three placements), and on the 16/32 tip at "
            "U (16&#8202;mm cell) and M2 (both cells) at seed 0 only. The half-ellipsoids have no M1 replays, the 16/32 tip none "
            "on the 25&#8202;mm cell at U, and the block and the capsule have pad replays only. "
            "The D1&#8211;D8 replays at M1 and M2 ran at seed 0, except the 11/22 tip at M2 (seeds 0&#8211;2); the RL arm, the "
            "spring check and the Drake arm ran on the 11/22 tip alone and the impratio-1000 arm at U and seed 0; the bed of "
            "the block and the capsule has the static and twist cases only.</li>"
            "<li>The 0.5&#8202;mm pads at the capsule&#8217;s M2 modulus (36&#8202;MPa) and at the U moduli (13&#8211;14&#8202;MPa) "
            "are too soft per pad for the pad builder (<code>reorient_backends.replace_tips</code>), which maps each pad&#8217;s "
            "stiffness to a MuJoCo impedance d0 and refuses d0 below 0.05 at the 0.02&#8202;s relaxation: the capsule has no "
            "replays at M2 and the 0.5&#8202;mm check arm did not run.</li>"
            "<li>The servo plant is the ad-hoc identification of 2026-10-13 (kp 0.72&#8202;N&#8202;m/rad, 0.18&#8202;N&#8202;m); its "
            "3.4&#8211;4.7&#8202;mm/N at the fingertip is what keeps normal-stiffness errors small on trajectories.</li>"
            "<li>The D1&#8211;D8 plans were designed for the block; on the curved tips they run near the tool&#8217;s slip "
            "limit.</li></ul>")


def bed_section():
    B = S.get("bed_checks", {})
    impl = [(k, v["impl_max_err"]) for k, v in B.items() if v.get("impl_max_err") is not None and v["arm"] != "rl"]
    worst = max(impl, key=lambda x: x[1]) if impl else None
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
                f"agrees with the phase-B curves within {100 * max(x[1] for x in impl):.1f}&#8239;% (worst: the "
                f"{NAME_PLAIN[worst[0].split('|')[0]]} at {worst[0].split('|')[1]} in "
                f"{'Drake' if worst[0].endswith('drake') else 'the pads'}), so the simulators implement the calibrated "
                "moduli. ")
    if m1c and m2c:
        txt += (rf"The bed onset arm over the finite-element {RB} is {min(r[0] for r in m1c):.2f}&#8211;{max(r[1] for r in m1c):.2f} "
                f"on the curved tips at M1 and {min(r[0] for r in m2c):.2f}&#8211;{max(r[1] for r in m2c):.2f} at M2, as the "
                "curves predict. ")
    if crp and crd:
        txt += (f"Under half the onset torque the pads creep {min(crp):.2f}&#8211;{max(crp):.2f}&#8202;&#176;/s at the planner "
                f"numerics and Drake {min(crd):.3f}&#8211;{max(crd):.3f}&#8202;&#176;/s")
        txt += (f"; at the RL numerics the pads creep {min(crl):.1f}&#8211;{max(crl):.1f}&#8202;&#176;/s. " if crl else ". ")
    rl = {v["tip"]: v["rl_vs_planner"] for v in B.values() if v.get("rl_vs_planner")}
    d2rl = d2p("ell11_g30", "M2", "rl")
    unst = sum(v["held"].get("rl", {}).get("unstable", 0) for v in S.get("d2", {}).values())
    if rl:
        parts = []
        for t, x in rl.items():
            pt = (f"the {NAME_PLAIN[t]} differs by {100 * x['approach_max']:.2f}&#8239;% in approach and "
                  f"{100 * x['onset_max']:.1f}&#8239;% in onset arm on the bed (bound 10&#8239;%)")
            if t == "ell11_g30" and d2rl:
                fail = not x["ok"] or d2rl["median_rms_angle_same_deg"] > 3.0
                pt += (f" and by a same-branch median of {d2rl['median_rms_angle_same_deg']:.1f}&#176; on its D1&#8211;D8 "
                       f"replays (bound 3&#176;), so it {'fails' if fail else 'passes'}")
            else:
                pt += (", and its replays at the RL numerics did not run" if x["ok"] else ", so it fails")
            parts.append(pt)
        txt += ("The numerics criterion compares the pads at the RL trainer&#8217;s 2&#8202;ms and impratio 10 with the "
                "planner&#8217;s: " + "; ".join(parts) + ". "
                + ("No replay at the RL numerics ended unstable." if unst == 0 else
                   f"{unst} replays at the RL numerics ended unstable."))
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


def predictions_table():
    F = S["fits"]
    m1_app = {t: (fit(t, "drake", "M1")["pass_approach"], fit(t, (F[t].get("pad_model") or {}).get("M1", "v2"), "M1")["pass_approach"])
              for t in TIPS}
    n_both = sum(1 for t in TIPS if all(m1_app[t]))
    m3 = {t: (fit(t, "drake", "M3")["pass_approach"], fit(t, "drake", "M3")["pass_arm"]) for t in CURVED}
    dap, darm = grid_change()
    sp = d2p("ell11_g30", "M2", "spring")
    ec = expo_task(CURVED) + [expo(t, "s_apex") for t in CURVED]
    p1_ok = min(ec) >= 0.72 and max(ec) <= 0.90
    p5 = []
    for t in ("ell11_g30", "ell16_g30"):
        a, b = d1g(t, "U"), d1g(t, "M2")
        if a and b and a["pads_vs_drake"]["median_rms_angle_same_deg"] is not None \
                and b["pads_vs_drake"]["median_rms_angle_same_deg"] is not None:
            p5.append((t, a["pads_vs_drake"]["median_rms_angle_same_deg"], b["pads_vs_drake"]["median_rms_angle_same_deg"],
                       a["pads_vs_drake"], b["pads_vs_drake"]))
    p5_fals = any(abs(y - x) > 2.0 for _, x, y, _, _ in p5)

    def same_set(t, c):
        return {(p_["plan"], p_["seed"]) for p_ in d1g(t, c)["pairs"] if p_["same_branch"]}
    p5_same = all(same_set(t, "U") == same_set(t, "M2") for t, *_ in p5)
    rows_ = [
        ["P1", rf"Curved tips: exponent 0.72&#8211;0.90, rod over sphere 0.60&#8211;0.85, {RB} 0.35&#8211;0.70&#8202;mm; block "
               rf"rod exponent 0.85&#8211;0.95, {RB} 3.2&#8211;3.6&#8202;mm",
         f"Curved tips: exponents {rng(ec)}"
         + ("" if p1_ok else f" ({sum(1 for x in ec if x < 0.72)} under the predicted 0.72, none under the falsifier&#8217;s 0.65)")
         + f", rod over sphere {rng([fe(t, 'cx12.5_z0') / fe(t, 's_apex') for t in CURVED])}, {RB} "
         f"{rng([fe(t, 'cx12.5_z0', '1', 'r_bar_mm') for t in CURVED])}&#8202;mm. Block: rod exponent "
         f"{expo('block4_g30', 'cx12.5_z0'):.2f} as predicted, {RB} {fe('block4_g30', 'cx12.5_z0', '1', 'r_bar_mm'):.2f}&#8202;mm "
         "under the predicted range"],
        ["P2", "M1 passes approach on at least 3 of 4 tips in both simulators and fails the arm on the curved tips",
         f"Falsified: {n_both} of 4 tips pass approach in both; the curved tips fail the arm"],
        ["P3", "M3 passes approach and arm in Drake on the three curved tips",
         "Falsified: " + "; ".join(f"{NAME_PLAIN[t]} {'passes' if all(m3[t]) else 'fails ' + ' and '.join(x for x, ok in (('approach', m3[t][0]), ('arm', m3[t][1])) if not ok)}"
                                   for t in CURVED)],
        ["P4", "The spring moves the D1&#8211;D8 trajectories by at most 1&#176; and 0.05&#8202;N",
         (f"{'Met' if (sp['median_rms_angle_same_deg'] or 9) <= 1 and (sp['median_rms_dN_same_N'] or 9) <= 0.05 else 'Falsified'} on the 11/22 tip "
          f"({f(sp['median_rms_angle_same_deg'], 2)}&#176;, {f(sp['median_rms_dN_same_N'], 3)}&#8202;N)") if sp else "Not run"],
        ["P5", "Held agreement on at least 5 of 6 runs at U, M1 and M2, and the calibration moves a tip&#8217;s same-branch "
               "median by at most 1&#176;; falsified by a move of more than 2&#176; or of 2 held outcomes",
         (f"{'Falsified' if p5_fals else 'Met'}: the same-branch median moves "
          + " and ".join(f"from {x:.1f}&#176; at U to {y:.1f}&#176; at M2 on the {NAME_PLAIN[t]}" for t, x, y, a, b in p5)
          + (", over the same placements of the 16&#8202;mm cell" if p5_same else "")
          + "; held agreement stays complete") if p5 else "Not run"],
        ["P6", "Impratio 1000 closes the pads&#8217; held-count gap to Drake on the capsule and the 11/22 tip; 0.5&#8202;mm pads do not",
         p6()],
        ["P7", rf"The finer contact grid changes approach by at most 5&#8239;% and {RB} by at most 10&#8239;%",
         f"Met: {100 * max(dap):.1f}&#8239;% and {100 * max(darm):.1f}&#8239;%"],
    ]
    return table(["Prediction", "Claim", "Outcome"], rows_, "The plan&#8217;s predictions against the runs.", key="pred",
                 text_cols=(0, 1, 2))


def p6():
    pr = PLAN["control_relevance"]["pads_vs_drake_D1_D8_U"]
    out, ok_ir = [], []
    for tip in ("cap_g30", "ell11_g30"):
        h = {a: d2held(tip, "U", a) for a in ("plan", "ir1000")}
        if not all(h.values()):
            return "Not run"
        dk = pr[tip]["drake_held_seed0"]
        ok_ir.append(abs(h["ir1000"][0] - dk) <= 1)
        out.append(f"{NAME_PLAIN[tip]}: planner pads {h['plan'][0]}, impratio 1000 {h['ir1000'][0]}, Drake {dk} of 8")
    verdict = "Met" if all(ok_ir) else "Falsified"
    return f"{verdict} (seed 0): " + "; ".join(out) + ". The 0.5&#8202;mm arm could not be built (Limitations)"


def verdict_section():
    return VERDICT_TEXT() + verdict_table() + predictions_table()


def d1g(tip, calib, own=False, pads="pads"):
    return S.get("d1", {}).get(f"{tip}|{'own' if own else 'ready'}|{calib}|{pads}")


def ICEM_TEXT():
    sm = smoke()
    ready = {k: v for k, v in sm.items() if not v["own"]}
    good = [k for k, v in ready.items() if v["logged"] is not None and abs(v["turn"] - v["logged"]) <= 1.0]
    bad = [(k, v) for k, v in ready.items() if v["logged"] is not None and abs(v["turn"] - v["logged"]) > 1.0]
    own = {k: v for k, v in sm.items() if v["own"]}
    D1 = S.get("d1", {})
    first = [p_["angle_first_deg"] for v in D1.values() if v["pads"] == "pads" for p_ in v["pairs"]]
    txt = ("<p>The consistency test gives both simulators the same servo commands and compares the tool&#8217;s axis, which "
           "sets the turn of the symmetric cylinder. Each logged pinned-iCEM run of the readiness study (planned on the U pads, "
           "on the 16 and 25&#8202;mm cells) was replayed open loop from three start placements: reset, the candidate "
           "grasp&#8217;s grip targets for 0.8&#8202;s, then the logged commands, each held for 10&#8202;ms at its value "
           "mid-interval, with the post&#8217;s collisions off from 0.8&#8202;s, to 6.3&#8202;s. ")
    if ready:
        txt += (f"On the planning pads at seed 0, {len(good)} of {len(good) + len(bad)} replays end within 1&#176; of the logged "
                "turn")
        if bad:
            txt += "; " + "; ".join(f"the {NAME_PLAIN[k.split('_cyl')[0]]}&#8217;s {k.split(':')[1]}&#8202;mm plan ends at "
                                    f"{v['turn']:+.1f}&#176; against the logged {v['logged']:+.1f}&#176;" for k, v in bad)
        txt += ". "
    for k, v in own.items():
        txt += (f"The 11/22 tip&#8217;s own plan, logged at every 1&#8202;ms command, replays to {v['turn']:+.1f}&#176; on "
                f"its 1&#8202;mm pads against the planner&#8217;s {v['logged']:+.1f}&#176;. ")
    if first:
        txt += (f"The two simulators leave the grip with the tool axes {min(first):.0f}&#8211;{max(first):.0f}&#176; apart at "
                "the first sample, 20&#8202;ms after it, so each run&#8217;s turn, counted from its own pose there, starts "
                "from a different reference; a pair is on the same branch when the axes end within 15&#176;. ")
    g = d1g("ell11_g30", "M2")
    gu = d1g("ell11_g30", "U")
    go = d1g("ell11_g30", "M2", own=True)
    if g:
        s_ = g["pads_vs_drake"]
        bc = pairs_by_cell(g)
        p16, p25 = bc.get("16", []), bc.get("25", [])
        tp25 = [r["turn"] for r in g["runs"] if r["plant"] == "pads" and ":25:" in r["plan"]]
        td25 = [r["turn"] for r in g["runs"] if r["plant"] == "drake" and ":25:" in r["plan"]]
        txt += (f"On the 11/22 tip at M2 the 0.5&#8202;mm pads and Drake keep the tool in the same {s_['held_agree']} of "
                f"{s_['n']} replays. On the 16&#8202;mm cell the axes end {rng([p_['angle_end_deg'] for p_ in p16], 0)}&#176; "
                f"apart and all {len(p16)} pairs are on the same branch, with a median RMS angle of "
                f"{s_['median_rms_angle_same_deg']:.1f}&#176; against the pads&#8217; seed spread of "
                f"{g['pads_seed_spread']['median_rms_angle_same_deg']:.1f}&#176; and normal and friction forces "
                f"{s_['median_rms_dN_same_N']:.2f} and {s_['median_rms_dT_same_N']:.2f}&#8202;N apart (bounds 0.25 and "
                f"0.35&#8202;N). On the 25&#8202;mm cell Drake turns the tool {rng(td25, 0)}&#176; and the pads "
                f"{rng(tp25, 0)}&#176;, and the axes end {rng([p_['angle_end_deg'] for p_ in p25], 0)}&#176; apart. "
                f"Consistency {'passes' if g['pass'] else 'fails'} on the angle and the forces. ")
        if gu:
            su = gu["pads_vs_drake"]
            txt += (f"At U, where Drake ran on the 16&#8202;mm cell, {su['same_branch']} of {su['n']} pairs end on the same "
                    f"branch, {su['median_rms_angle_same_deg']:.1f}&#176; RMS apart, with the fingers&#8217; normal and "
                    f"friction forces {su['median_rms_dN_same_N']:.2f} and {su['median_rms_dT_same_N']:.2f}&#8202;N RMS apart, "
                    "so U fails on the forces. ")
    if go:
        so = go["pads_vs_drake"]
        txt += (f"On the tip&#8217;s own plan, re-planned on its 1&#8202;mm pads at M2, both simulators keep the tool in "
                f"{so['held_agree']} of {so['n']} runs and the axes end {rng([p_['angle_end_deg'] for p_ in go['pairs']], 0)}&#176; "
                f"apart in every pair, with the normal forces {rng([p_['rms_dN_N'] for p_ in go['pairs']], 1)}&#8202;N RMS apart "
                "(Figure&#160;[[fig:film]]). ")
    gi = d1g("ell11_g30", "M2", pads="pads_ir1000")
    if gi and g:
        si = gi["pads_vs_drake"]
        txt += (f"At impratio 1000, ten times the planner&#8217;s, the pads put {si['same_branch']} of the {si['n']} readiness "
                f"pairs on Drake&#8217;s branch (against {g['pads_vs_drake']['same_branch']} at impratio 100), at a median "
                f"{si['median_rms_angle_same_deg']:.1f}&#176; against a {gi['angle_bound_deg']:.1f}&#176; bound and normal "
                f"forces {si['median_rms_dN_same_N']:.2f}&#8202;N apart: consistency still fails. ")
    return "<p>" + txt.strip() + "</p>"


def d2p(tip, calib, arm):
    v = S.get("d2", {}).get(f"{tip}|{calib}")
    if not v:
        return None
    return v["pairs"].get(f"plan_vs_{arm}") if arm != "plan" else v["pairs"].get("plan_seed_spread")


def d2held(tip, calib, arm, seed=0):
    v = S.get("d2", {}).get(f"{tip}|{calib}")
    h = (v or {}).get("held", {}).get(arm)
    if not h:
        return None
    n = h["n"].get(str(seed), h["n"].get(seed, 0))
    return (h.get(str(seed), h.get(seed)), n) if n else None


def D1D8_TEXT():
    txt = ("<p>The eight deployed bench plans D1&#8211;D8 were designed for the block and are replayed open loop as in the "
           "readiness study (20&#8202;ms samples of tool pose and finger forces). ")
    sp = d2p("ell11_g30", "M2", "spring")
    if sp:
        ok = (sp['median_rms_angle_same_deg'] or 9) <= 1 and (sp['median_rms_dN_same_N'] or 9) <= 0.05
        txt += (f"Spring check on the 11/22 tip: the 0.5&#8202;mm pads at M2 behind the {S['fits']['ell11_g30']['k_s_N_per_mm']:.1f}&#8202;N/mm "
                f"slide spring (M3) against the same pads without it agree on the held outcome in {sp['held_agree']} of "
                f"{sp['n']} runs; the {sp['same_branch']} same-branch pairs differ by a median {f(sp['median_rms_angle_same_deg'], 2)}&#176; "
                f"and {f(sp['median_rms_dN_same_N'], 3)}&#8202;N per finger (bounds 1&#176; and 0.05&#8202;N), so M2 "
                f"{'stands in' if ok else 'does not stand in'} for M3 on trajectories. ")
    rl = d2p("ell11_g30", "M2", "rl")
    if rl:
        txt += (f"At the RL trainer&#8217;s numerics the same pads agree with the planner&#8217;s on {rl['held_agree']} of {rl['n']} runs, "
                f"median {f(rl['median_rms_angle_same_deg'], 1)}&#176; over {rl['same_branch']} same-branch pairs. ")
    dk = d2p("ell11_g30", "M2", "drake")
    if dk:
        hp, hd = d2held("ell11_g30", "M2", "plan"), d2held("ell11_g30", "M2", "drake")
        txt += (f"Drake against the pads on the 11/22 tip at M2 (seed 0): Drake keeps the tool on {hd[0]} hands and the pads on "
                f"{hp[0]} of 8, held agreement {dk['held_agree']} of {dk['n']}, median {f(dk['median_rms_angle_same_deg'], 1)}&#176; "
                f"over {dk['same_branch']} same-branch pairs. ")
    chk = []
    for tip in ("cap_g30", "ell11_g30"):
        h = {a: d2held(tip, "U", a) for a in ("plan", "ir1000")}
        if all(h.values()):
            chk.append(f"{NAME_PLAIN[tip]} {h['plan'][0]} and {h['ir1000'][0]} of 8")
    if chk:
        pr = PLAN["control_relevance"]["pads_vs_drake_D1_D8_U"]
        txt += ("Check arm at U, seed 0, held by the planner pads and by the pads at impratio 1000: " + "; ".join(chk)
                + f". Drake kept the tool on {pr['cap_g30']['drake_held_seed0']} (capsule) and "
                f"{pr['ell11_g30']['drake_held_seed0']} (11/22) of 8 in the readiness rows; the gap stays open at impratio "
                "1000 (prediction P6).")
    return "<p>" + txt.strip() + "</p>"


def VERDICT_TEXT():
    V = S.get("verdicts", {})
    well = [t for t in TIPS if V.get(t, {}).get("modelled_well")]
    e11 = V["ell11_g30"]["calibrations"]["M3"]
    e16 = V["ell16_g30"]["calibrations"]["M3"]
    cap = fit("cap_g30", "drake", "M3")["approach_err_pct"]
    cz = [v for k, v in cap.items() if k.startswith("cz")]
    blp = fit("block4_g30", "v2", "M1")
    bz = fit("block4_g30", "drake", "M1")["approach_err_pct"]
    bk = max(bz, key=lambda k: abs(bz[k]))
    dap, darm = grid_change()
    names = {"block4_g30": "the block", "cap_g30": "the capsule", "ell16_g30": "the 16/32 half-ellipsoid",
             "ell11_g30": "the 11/22 half-ellipsoid"}
    g, go, g16 = d1g("ell11_g30", "M2"), d1g("ell11_g30", "M2", own=True), d1g("ell16_g30", "M2")
    sp = d2p("ell11_g30", "M2", "spring")
    ratio = fe("cap_g30", "cx12.5_z0") / fe("cap_g30", "cz12.5_y0")
    txt = (f"<p>Against the plan&#8217;s criteria {len(well) if well else 'none'} of the four prints "
           f"{'is' if len(well) <= 1 else 'are'} modelled well{': ' + ', '.join(names[t] for t in well) if well else ''} "
           "(Table&#160;[[tab:verdict]]). ")
    txt += ("The 11/22 half-ellipsoid at M3 meets approach and twist arm in Drake "
            f"({e11['drake']['approach_rms_pct']:.1f}&#8239;% RMS, arm {e11['drake']['arm_min']:.2f}&#8211;{e11['drake']['arm_max']:.2f}) "
            f"and in the 0.5&#8202;mm pads ({e11['pads']['approach_rms_pct']:.1f}&#8239;%, {e11['pads']['arm_min']:.2f}&#8211;"
            f"{e11['pads']['arm_max']:.2f}), and misses consistency and the spring check. ")
    if g:
        s_ = g["pads_vs_drake"]
        txt += (f"On the readiness commands both simulators keep the tool in {s_['held_agree']} of {s_['n']} replays, "
                f"{s_['same_branch']} pairs end on the same branch, and their median RMS angle is "
                f"{s_['median_rms_angle_same_deg']:.1f}&#176; against a {g['angle_bound_deg']:.1f}&#176; bound, with forces "
                f"{s_['median_rms_dN_same_N']:.2f} and {s_['median_rms_dT_same_N']:.2f}&#8202;N apart against 0.25 and 0.35&#8202;N. ")
    if go:
        txt += (f"On its own plan the tool axes end {rng([p_['angle_end_deg'] for p_ in go['pairs']], 0)}&#176; apart in all "
                f"{go['pads_vs_drake']['n']} pairs. ")
    if sp:
        txt += (f"Behind the spring the pads&#8217; D1&#8211;D8 replays move by a median {sp['median_rms_angle_same_deg']:.2f}&#176; "
                f"and {sp['median_rms_dN_same_N']:.3f}&#8202;N (bounds 1&#176; and 0.05&#8202;N), so M2 cannot stand in for M3. ")
    txt += (f"The 16/32 half-ellipsoid at M3 meets approach ({e16['drake']['approach_rms_pct']:.1f}&#8239;% in Drake) and misses "
            f"the arm bound by {e16['drake']['arm_max'] - 1.15:.3f} in Drake and {e16['pads']['arm_max'] - 1.15:.3f} in the pads, "
            rf"less than the {100 * max(darm):.0f}&#8239;% by which the capsule&#8217;s {RB} changes on the finer finite-element "
            "grid")
    if g16:
        s16 = g16["pads_vs_drake"]
        txt += (f"; at M2 {s16['same_branch']} of its {s16['n']} pairs {'ends' if s16['same_branch'] == 1 else 'end'} on the "
                "same branch, "
                f"{s16['median_rms_angle_same_deg']:.1f}&#176; RMS apart against a {g16['angle_bound_deg']:.1f}&#176; bound")
    txt += (". The capsule misses approach at every calibration in Drake: at M3 the rods along its straight section sink "
            f"{min(cz):.0f}&#8211;{max(cz):.0f}&#8239;% too far, because the print is {ratio:.1f} times stiffer along the finger "
            "than across it and one spring gives both directions the same compliance. "
            f"The block, the control, meets both criteria in the pads at M1 ({blp['approach_rms_pct']:.1f}&#8239;%, arm "
            f"{blp['arm_min']:.2f}&#8211;{blp['arm_max']:.2f}) and misses approach in Drake by its largest error, "
            f"{bz[bk]:.0f}&#8239;% under the {bk[2:].split('_')[0]}&#8202;mm rod {'along' if bk.startswith('cz') else 'across'} "
            f"the finger at {bk.split('@')[1]}&#8202;N. The block and the capsule have no Drake replays (Limitations).</p>")
    return txt


def FILM():
    p = os.path.join(D, "media", "films.jsonl")
    if not os.path.exists(p):
        return ""
    out = ""
    last = {}
    for r in rows("media/films.jsonl"):
        last[r["film"]] = r
    for r in last.values():
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
    m1 = [fit(t, "drake", "M1") for t in TIPS]
    V = S["verdicts"]
    well = [t for t in TIPS if V.get(t, {}).get("modelled_well")]
    e11 = V["ell11_g30"]["calibrations"]["M3"]
    ks11 = S["fits"]["ell11_g30"]["k_s_N_per_mm"]
    ratio = fe("cap_g30", "cx12.5_z0") / fe("cap_g30", "cz12.5_y0")
    txt = (("None of the four printed 85A fingertips meets every criterion of the plan. " if not well else
            f"{len(well)} of the four printed 85A fingertips meet every criterion of the plan. ")
           + "One modulus fitted to the finite-element approach leaves the largest approach error at "
           f"{min(v['approach_max_pct'] for v in m1):.0f}&#8211;{max(v['approach_max_pct'] for v in m1):.0f}&#8239;% in Drake and "
           f"gives the curved tips {min(fit(t, 'drake', 'M1')['arm_min'] for t in CURVED):.1f}&#8211;"
           f"{max(fit(t, 'drake', 'M1')['arm_max'] for t in CURVED):.1f} times the print&#8217;s twist arm. A stiff modulus behind "
           f"a {ks11:.1f}&#8202;N/mm series spring reproduces the 11/22 half-ellipsoid, the quarter-ellipsoid print, within "
           f"{e11['drake']['approach_rms_pct']:.1f}&#8239;% RMS approach in Drake and {e11['pads']['approach_rms_pct']:.1f}&#8239;% in "
           f"0.5&#8202;mm pads, arm {min(e11['drake']['arm_min'], e11['pads']['arm_min']):.2f}&#8211;"
           f"{max(e11['drake']['arm_max'], e11['pads']['arm_max']):.2f}.")
    g = d1g("ell11_g30", "M2")
    if g:
        bc = pairs_by_cell(g)
        e16 = [p_["angle_end_deg"] for p_ in bc.get("16", [])]
        e25 = [p_["angle_end_deg"] for p_ in bc.get("25", [])]
        s_ = g["pads_vs_drake"]
        txt += (" On the same servo commands the two simulators then part: the tool axes end "
                f"{min(e25):.0f}&#8211;{max(e25):.0f}&#176; apart on the 25&#8202;mm cell, and {min(e16):.0f}&#8211;{max(e16):.0f}&#176; "
                f"apart on the 16&#8202;mm cell with the fingers&#8217; normal forces {s_['median_rms_dN_same_N']:.2f}&#8202;N RMS apart "
                "against a 0.25&#8202;N bound.")
    return txt + (f" One spring cannot fit the capsule, which is {ratio:.1f} times stiffer along the finger than across it.")


def future():
    F = S["fits"]
    cx = {t: F[t]["models"]["drake"]["M3x"] for t in CURVED}
    r11 = fe("ell11_g30", "cx12.5_z0", "1", "r_bar_mm")
    m1_11 = fit("ell11_g30", "drake", "M1")["arm_ratio"]["cx12.5_z0@1"]
    w = [r["wall_s"] for r in rows("replays/icem.jsonl") if r.get("status") == "ok" and r["plant"] == "drake"]
    B = S.get("bed_checks", {})
    crp = [(v.get("creep_deg_s") or {}).get("1") for k, v in B.items() if k.endswith("|M2|pads")]
    crd = [(v.get("creep_deg_s") or {}).get("1") for k, v in B.items() if k.endswith("|M2|drake")]
    crp, crd = [x for x in crp if x is not None], [x for x in crd if x is not None]
    bo = [fe(t, "s_apex") / fe(t, "cx12.5_z0") for t in CURVED]
    return (
        "<ol>"
        "<li>Bench indentation of each print under the 25&#8202;mm rod at 0.5, 1 and 3&#8202;N (readiness procedure, step "
        "&#8216;indentation&#8217;, three repeats, docs/experiments/20261016-sim2real_fingertip_readiness/bench/procedure.json), "
        "entered as a CSV (tip, probe, force_N, approach_um) and refitted with <code>fe_contact_calibration_fit.py fit --bench "
        "CSV --out FOLDER/fit/fits.json</code>: fixes &#955;, which scales every modulus and spring here. A residual above "
        "10&#8239;% RMS after &#955;, or a force exponent more than 0.10 from Table&#160;[[tab:bench]], falsifies the "
        "shell-over-gyroid model for these prints.</li>"
        "<li>The 4&#8202;mm steel ball at the apex of each curved print at 0.5, 1 and 3&#8202;N: a ball-over-rod ratio of 2 or "
        f"more at 1&#8202;N (a local law) instead of {rng(bo, 1)} falsifies the finite elements and favours M1.</li>"
        "<li>Consistency of the block and the capsule in Drake, the replays cut for time: "
        "<code>fe_contact_calibration_sim.py icem --tips block4_g30 cap_g30 --calib M1 --plants drake --seeds 0 1 2</code> "
        f"(12 replays at {min(w):.0f}&#8211;{max(w):.0f}&#8202;s, {12 * min(w) / 60:.0f}&#8211;{12 * max(w) / 60:.0f}&#8202;min), "
        "then the summary. The block is the control: if its pads and Drake also miss the consistency criterion on these "
        "commands (5 of 6 held outcomes, the angle and force bounds), the split appears on every tip shape and a calibration "
        "of the normal response cannot remove it.</li>"
        f"<li>The pads&#8217; creep against Drake&#8217;s: on the bed at M2 the pads creep {rng(crp, 2)}&#8202;&#176;/s under half "
        f"the onset torque and Drake {rng(crd, 3)}&#8202;&#176;/s. Add an impratio-1000 arm to <code>bed_model</code> in "
        "<code>fe_contact_calibration_sim.py</code> and run <code>bed --tips ell11_g30 --calib M2 --arms pads ir1000 drake "
        "--cases T2</code>, and "
        "measure the printed 11/22 tip&#8217;s creep on the bench under a held twist at half its onset torque for 1&#8202;s. "
        "A bench creep near Drake&#8217;s (under 0.05&#8202;&#176;/s) makes the pads&#8217; creep a model error to remove "
        "before the replays can agree; a bench creep near the pads&#8217; makes Drake&#8217;s stick the error.</li>"
        f"<li>The twist test on the 11/22 print (rod at 1 and 3&#8202;N, turned until slip, &#956; from a pull test): an onset "
        rf"arm of {0.70 * r11:.2f}&#8211;{1.15 * r11:.2f}&#8202;mm at 1&#8202;N (0.70&#8211;1.15 of {RB}) confirms M3; "
        f"{m1_11 * r11:.2f}&#8202;mm or more (Drake at M1) falsifies it.</li>"
        "<li>The series spring in both simulators: the spring check failed on the 11/22 tip (Table&#160;[[tab:d2]]), so M2 "
        "cannot stand in for M3 on trajectories. Add the spring to the Drake cell plant as a prismatic joint between the DIP "
        rf"link and the tip with a PD actuator (\(p = k_s\), target 0) beside the pads&#8217; slide joint "
        "(<code>fe_contact_calibration_sim.add_spring</code>), and rerun <code>fe_contact_calibration_sim.py icem --calib M3</code> "
        "on the 11/22 tip; the consistency criterion decides.</li>"
        "<li>The capsule along the finger: one series spring cannot give the capsule both stiffnesses. Fitted to the rods "
        rf"across the finger alone (\(k_s\) {cx['cap_g30']['k_s_N_per_mm']:.1f}&#8202;N/mm), M3 reproduces them within "
        f"{cx['cap_g30']['cx_rms_pct']:.1f}&#8239;% RMS and sinks the rods along the finger "
        f"{cx['cap_g30']['cz_range_pct'][0]:.0f}&#8211;{cx['cap_g30']['cz_range_pct'][1]:.0f}&#8239;% too far. Count how often the "
        "capsule meets a tool along its length in the D1&#8211;D8 and iCEM runs (contact-patch orientation in replays/, from the "
        "pads&#8217; contact frames); if never, judge the capsule on the rods across the finger and rerun "
        "<code>fe_contact_calibration_summary.py</code> with that task set.</li>"
        "</ol>")


def build():
    sections = {"LEDE": lede(), "VERDICT": verdict_section(), "FE": fe_section(), "FITS": fits_section(),
                "BED": bed_section(), "ICEM": icem_section(), "D1D8": d1d8_section(), "BENCH": bench_section(),
                "LIMITS": limitations(), "FUTURE": future()}
    main(sections)



def smoke():
    """The smoke test of the readiness-plan replays (the U pads at seed 0) and of the tips' own plans (the 1 mm pads they
    were planned on, at seed 0) against the logged closed-loop turn."""
    out = {}
    for r in rows("replays/icem.jsonl"):
        if r.get("status") != "ok" or int(r["seed"]) != 0:
            continue
        if (not r["own"] and r["calib"] == "U" and r["plant"] == "pads") or (r["own"] and r["plant"] == "pads1"):
            lg = r.get("logged") or {}
            out[r["plan"]] = {"turn": r["turn_end_deg"], "logged": lg.get("turn_end_deg"), "held": r["held"],
                              "logged_held": lg.get("held"), "own": r["own"]}
    return out


if __name__ == "__main__":
    build()
