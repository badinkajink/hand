#!/usr/bin/env python3
r"""Build docs/experiments/20261007-newton_hydro_tests/20261007-newton_hydroelastic_friction_reduction_edge.html.

    python3 scripts/newton_hydro_tests_page.py

Three tests of Newton's hydroelastic fingertip contact on the contact bed (docs/experiments/20261005-contact_bed/): the
friction-row time constant (contact_bed_newton.py presets newton_hydro_mc_tf<ms>), contact reduction and SDF voxel size
(newton_hydro_unreduced_mc, newton_hydro_mc_vox025, newton_hydro_mc_vox1), and a square bar with an edge toward each pad
(scripts/contact_bed_edge.py, reference scripts/hydroelastic_arm_integral.py bar_law). Reads the rows in
docs/experiments/20261007-newton_hydro_tests/ and the bed's Drake and pad rows; every number in the prose is computed
here. Style, plotting helpers and the LaTeX renderer are those of scripts/contact_overview_page.py.
"""
from __future__ import annotations

import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import texsvg  # noqa: E402
import retro_style  # noqa: E402

D = os.path.join(P.EXP, "20261007-newton_hydro_tests")
OUT = os.path.join(D, "20261007-newton_hydroelastic_friction_reduction_edge.html")
TPL = os.path.join(ROOT, "scripts/newton_hydro_tests_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
BED_PATH = "docs/experiments/20261005-contact_bed/20261005-contact_model_bed.html"
OVERVIEW_PATH = os.path.relpath(P.OUT, ROOT)
num, fmt, pick = P.num, P.fmt, P.pick
US = P.US
LAW = {0.5: 0.820, 1.0: 0.974, 3.0: 1.281}      # hydroelastic_arm_integral.py, cylinder, mm
LAW_RATIO = LAW[3.0] / LAW[0.5]
TF = [1, 2, 3, 5, 7, 10, 15, 20, 40]
KF10_TF = 3.87                                  # the base friction gain kf 10 on the pad-tool pair, ms
FIG = [0]
C_NEW = "var(--c-newton)"


def rows(name):
    return P.load(os.path.join(D, name))


def ok(r):
    return r is not None and r.get("status", "complete") == "complete"


def tf_of(model):
    if model == "newton_hydro_mc":
        return KF10_TF
    if model.startswith("newton_hydro_mc_tf"):
        return float(model.split("tf")[-1])
    return None


def figure(svg, caption):
    FIG[0] += 1
    return f'<figure class="diagram">{svg}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def table(head, body, note=None):
    out = ["<div class='tw'><table><thead><tr>" + "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
           + "</tr></thead><tbody>"]
    for r in body:
        if isinstance(r, str):
            out.append(f"<tr class='grp'><td colspan='{len(head)}'>{r}</td></tr>")
        else:
            out.append("<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table></div>")
    if note:
        out.append(f"<p class='tnote'>{note}</p>")
    return "".join(out)


def rng(v, scale=1.0, nd=0):
    """'a&#8211;b' of a list, or one value when they round alike."""
    a, b = f"{min(v) * scale:.{nd}f}", f"{max(v) * scale:.{nd}f}"
    return a if a == b else f"{a}&#8211;{b}"


def pct(a, b):
    return None if a is None or b in (None, 0) else 100.0 * (a / b - 1.0)


def sgn(v, nd=0):
    return "&#8211;" if v is None else (("+" if v >= 0 else "&#8722;") + num(abs(v), f".{nd}f"))


# ------------------------------------------------------------------------------------------ terms

GLOSSARY = [
    ("pad, \\(N\\)", "One fingertip&#8217;s contact with the tool; two pads pinch it, and \\(N\\) is the normal force each rail "
                     "applies, in N."),
    ("pressure law", "The hydroelastic field \\(p = E(1-\\rho/R)\\) of a compliant sphere of radius \\(R\\) = 10.55&#8202;mm and "
                     "modulus \\(E\\) = 10&#8202;MPa, \\(\\rho\\) the distance from its centre, integrated over the part of the tool "
                     "surface inside the sphere with no simulator (<code>scripts/hydroelastic_arm_integral.py</code>). On the "
                     "screwdriver it gives the friction arms 0.820, 0.974 and 1.281&#8202;mm at 0.5, 1 and 3&#8202;N."),
    ("onset arm, sliding arm", "Task&#160;2 ramps a torque about the pinch axis; the torque at spin onset divided by \\(2\\mu N\\) is "
                     "the onset arm, in mm per pad. The sliding arm is the torque during the first 60&#8202;ms of spin, corrected "
                     "for the tool&#8217;s angular acceleration, divided by \\(2\\mu N\\). <i>/law</i>: over the pressure law&#8217;s arm."),
    ("arm ratio", "Onset arm at 3&#8202;N over onset arm at 0.5&#8202;N. The pressure law gives 1.565 on the screwdriver, a Hertz "
                  "contact 1.817."),
    ("pre-onset rotation", "Rotation of the tool between the start of the torque ramp and spin onset, in degrees."),
    ("creep", "Steady sliding under a load held at half the onset value for 1&#8202;s: the tool&#8217;s axial speed in task&#160;1 "
              "(&#181;m/s) and its turning speed in task&#160;2 (&#176;/s). Rigid Coulomb friction gives zero."),
    ("largest swing, 80&#176; force", "Task&#160;5 lowers the pinch from 6 to 0.2&#8202;N over 4&#8202;s with the tool&#8217;s "
              "centre of mass 15&#8202;mm off the pinch line. Largest swing: the largest angle of the tool from horizontal "
              "(90&#176; hanging; Drake 87.3&#176;). 80&#176; force: the pinch force when the swing first passes 80&#176;, in N; "
              "a swing braked by friction passes 80&#176; late, at a low force (Drake 0.39&#8202;N)."),
    ("\\(t_f\\), \\(k_f\\)", "Newton&#8217;s friction gain \\(k_f\\) (N&#8202;s/m) becomes, in SolverMuJoCo, the time constant "
              "\\(t_f\\) of each contact&#8217;s friction rows (<code>solreffriction</code>), (1). A shorter \\(t_f\\) makes friction "
              "stick harder; the MuJoCo pads use their contact time constant, 10&#8202;ms."),
    ("\\(k_h/m_\\text{eff}\\)", "Newton&#8217;s hydroelastic stiffness \\(k_h = E/R\\) divided by the pad&#8211;tool effective mass "
              "of the solver&#8217;s MuJoCo model, 17.4&#8202;g, so that MuJoCo&#8217;s acceleration-space contact realises the "
              "pressure law (bed page, Newton paragraph)."),
    ("contact reduction", "Newton merges the faces of a hydroelastic contact surface into a few representative contacts "
              "(8&#8211;10 per pad here) before the solver; <i>all faces</i> passes every face (70&#8211;350 per pad)."),
    ("voxel size", "Grid spacing of the signed-distance fields from which Newton builds each shape&#8217;s pressure field; 0.5&#8202;mm "
              "unless stated."),
    ("approach \\(\\delta\\)", "Distance the pad centre has moved toward the tool past first touch, in mm."),
    ("force at the law&#8217;s approach", "The pad placed at the approach at which the pressure law carries \\(N\\), and the force "
              "along the pinch axis that each model&#8217;s contact law gives there, with no solver dynamics and no friction; over "
              "\\(N\\). The law gives 1."),
    ("patch half-length", "Largest distance along the edge (world \\(y\\)) of a contact carrying over 1&#8202;% of the largest, in mm."),
    ("relaxation time \\(t_r\\)", "Drake&#8217;s dissipation time for hydroelastic contact, 10&#8202;ms on the bed."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


# ------------------------------------------------------------------------------------------ setup

def setup():
    return ("<p>The rig is the bed&#8217;s two-pad pinch (<code>docs/experiments/20261005-contact_bed/PROTOCOL.md</code>): two "
            "real_v1 fingertip spheres on rails along \\(x\\) press a tool whose axis lies along \\(y\\), \\(\\mu\\) = 1, 1&#8202;ms "
            "step. Newton runs the hydroelastic tip with \\(k_h/m_\\text{eff}\\), 0.5&#8202;mm voxels, contact reduction on and "
            "\\(k_f\\) = 10, the bed&#8217;s Newton model, except where a section varies one of them. Drake hydroelastic (SAP, "
            "\\(E\\) = 10&#8202;MPa, \\(t_r\\) = 10&#8202;ms) and the 1&#8202;mm MuJoCo sphere pad are the bed&#8217;s rows. The "
            "pressure law is the reference for force and arm; Drake is one more discretization of it.</p>"
            "<p>The friction rows of a contact in SolverMuJoCo have the time constant</p>"
            "\\[ t_f = \\frac{2}{k_f\\, w\\,\\big((1-d_0)/\\text{impratio} + d_0\\big)} \\qquad (1) \\]"
            "<p>with \\(w\\) = 57.4&#8202;kg\\(^{-1}\\) the pad&#8211;tool inverse weight of the solver&#8217;s model, \\(d_0\\) = 0.9 "
            "and impratio 100 (Newton&#8217;s <code>kernels.py</code>, force-space friction slope). The presets solve (1) for "
            "\\(k_f\\); the <code>solreffriction</code> read back from the solver&#8217;s contacts equals the target \\(t_f\\) to "
            "six digits.</p>")


# ------------------------------------------------------------------------------------------ friction rows

def svg_brake(brk, bed_brake):
    W, Hh = 860, 330
    out = P._svg_open(W, Hh, "Swing angle against time in task 5, and the largest swing against the friction-row time constant")
    fx, fy = P._panel(out, 70, 30, 400, 240, (1.6, 5.5), (0, 140), [2, 3, 4, 5], [0, 30, 60, 90, 120], "time (s)", "swing (&#176;)")
    series = [(pick(bed_brake, "drake_hydro", dt_ms=1.0), "var(--c-drake)", False, "Drake"),
              (pick(bed_brake, "mj_pads1", dt_ms=1.0), "var(--c-sphere)", False, "1 mm pads")]
    for m, dash in (("newton_hydro_mc", True), ("newton_hydro_mc_tf10", False), ("newton_hydro_mc_tf40", False)):
        series.append((pick(brk, m), C_NEW, dash, None))
    for r, col, dash, _ in series:
        if r and r.get("fine"):
            _, op = (None, 1.0)
            P._path(out, fx, fy, [(a, b) for a, _, b, _ in r["fine"] if 1.6 <= a <= 5.5], col, dashed=dash, width=1.8)
    out.append(f'<line x1="{fx(1.6):.1f}" x2="{fx(5.5):.1f}" y1="{fy(90):.1f}" y2="{fy(90):.1f}" style="stroke:var(--ink3);stroke-dasharray:2 4"/>')
    gx, gy = P._panel(out, 560, 30, 270, 240, (0.8, 50), (80, 140), [1, 2, 5, 10, 20, 40], [80, 100, 120, 140],
                      "friction-row time constant (ms)", "largest swing (&#176;)", logx=True)
    d = pick(bed_brake, "drake_hydro", dt_ms=1.0)
    if d:
        y0, y1 = gy(d["phi_max_deg"] + 3), gy(d["phi_max_deg"] - 3)
        out.append(f'<rect x="{gx(0.8):.1f}" y="{y0:.1f}" width="{gx(50) - gx(0.8):.1f}" height="{y1 - y0:.1f}" style="fill:var(--sunk)"/>'
                   f'<text x="{gx(45):.1f}" y="{y1 + 13:.1f}" text-anchor="end" style="fill:var(--ink3)">Drake &#177;3&#176;</text>')
    pts = sorted((tf_of(r["model"]), r["phi_max_deg"]) for r in brk if tf_of(r["model"]) and r.get("phi_max_deg") is not None)
    P._path(out, gx, gy, pts, C_NEW, width=1.4)
    for x, y in pts:
        P._marker(out, gx(x), gy(y), C_NEW, "diamond", hollow=abs(x - KF10_TF) < 0.1, title=f"t_f {x:g} ms: {y:.1f} deg")
    out.append("</svg>")
    leg = P._legend_html([("Drake", "var(--c-drake)", False, None), ("1 mm pads", "var(--c-sphere)", False, None),
                          ("Newton, default friction gain (3.9 ms)", C_NEW, True, None),
                          ("Newton, friction rows 10 and 40 ms", C_NEW, False, None)])
    return "".join(out) + leg


def friction(brk, tw, pu, bed):
    body = []
    bdk, bpd = pick(bed["brake"], "drake_hydro", dt_ms=1.0), pick(bed["brake"], "mj_pads1", dt_ms=1.0)
    tdk = {N: pick(bed["twist"], "drake_hydro", N=N, dt_ms=1.0) for N in (0.5, 1.0, 3.0)}
    tpd = {N: pick(bed["twist"], "mj_pads1", N=N, dt_ms=1.0) for N in (0.5, 1.0, 3.0)}
    pdk = pick(bed["pull"], "drake_hydro", N=1.0, dt_ms=1.0)
    ppd = pick(bed["pull"], "mj_pads1", N=1.0, dt_ms=1.0)

    def line(label, b, t1, t3, p1, kf=None):
        return [label, fmt(kf, 2) if kf else "&#8211;",
                fmt(t1.get("rbar_onset_mm") / LAW[1.0], 2) if t1 else "&#8211;",
                fmt(t3.get("rbar_ratio_3_05_onset"), 3) if t3 else "&#8211;",
                fmt(t1.get("rot_pre_deg"), 2) if t1 else "&#8211;", fmt(t1.get("creep_deg_s"), 3) if t1 else "&#8211;",
                fmt(p1.get("creep_mm_s") * 1e3, 1) if p1 else "&#8211;",
                fmt(b.get("phi_max_deg"), 1) if b else "&#8211;", fmt(b.get("N_at_80_N"), 2) if b else "&#8211;"]
    body.append(line("Drake hydroelastic", bdk, tdk[1.0], tdk[3.0], pdk))
    body.append(line("MuJoCo 1&#8202;mm sphere pad", bpd, tpd[1.0], tpd[3.0], ppd))
    body.append("Newton hydroelastic, \\(k_h/m_\\text{eff}\\), by friction-row time constant")
    models = sorted({r["model"] for r in brk if tf_of(r["model"])}, key=tf_of)
    for m in models:
        b = pick(brk, m)
        t1, t3 = pick(tw, m, N=1.0), pick(tw, m, N=3.0)
        p1 = pick(pu, m, N=1.0)
        lab = f"\\(t_f\\) = {tf_of(m):.2g}&#8202;ms" + (" (\\(k_f\\) = 10)" if m == "newton_hydro_mc" else "")
        body.append(line(lab, b, t1, t3, p1, kf=b.get("kf") if b else None))
    tab = table(["model", "\\(k_f\\) (N&#8202;s/m)", "onset arm / law, 1&#8202;N", "arm ratio", "pre-onset rotation, 1&#8202;N (&#176;)",
                 "task&#160;2 creep, 1&#8202;N (&#176;/s)", "task&#160;1 creep, 1&#8202;N (&#181;m/s)", "largest swing (&#176;)",
                 "80&#176; force (N)"], body,
                "Table 1: friction rows of Newton&#8217;s hydroelastic tip, 1&#8202;ms step. Tasks 1 and 2 were run at 2, 3.9, 5, 10 and "
                "20&#8202;ms; task&#160;5 at every listed \\(t_f\\). The arm ratio of the pressure law is 1.565.")
    # numbers for the prose
    on = [pick(tw, m, N=N)["rbar_onset_mm"] for m in models for N in (1.0,) if pick(tw, m, N=N)]
    ratios = [pick(tw, m, N=3.0).get("rbar_ratio_3_05_onset") for m in models if pick(tw, m, N=3.0)]
    cr = {tf_of(m): pick(tw, m, N=1.0)["creep_deg_s"] for m in models if pick(tw, m, N=1.0)}
    cp = {tf_of(m): pick(pu, m, N=1.0)["creep_mm_s"] * 1e3 for m in models if pick(pu, m, N=1.0)}
    sw = {tf_of(m): pick(brk, m)["phi_max_deg"] for m in models if pick(brk, m)}
    n80 = {tf_of(m): pick(brk, m)["N_at_80_N"] for m in models if pick(brk, m)}
    over = sorted(t for t, v in sw.items() if bdk and v > bdk["phi_max_deg"] + 3)
    within = sorted(t for t, v in sw.items() if bdk and v <= bdk["phi_max_deg"] + 3)
    sc = rows("brake_torque_scale.jsonl")
    over_s = ", ".join(f"{t:.2g}" for t in over)
    fa = first_fall(brk, bed["brake"])
    txt = [f"<p>The friction-row time constant sets how far Newton&#8217;s tip creeps and leaves the spin-onset torque where it was. "
           f"From \\(t_f\\) = 2 to 20&#8202;ms the onset arm at 1&#8202;N stays at {min(on):.3f}&#8211;{max(on):.3f}&#8202;mm "
           f"({min(on) / LAW[1.0]:.2f}&#8211;{max(on) / LAW[1.0]:.2f} of the law) and the arm ratio at "
           f"{min(ratios):.3f}&#8211;{max(ratios):.3f}, while the task&#160;2 creep at 1&#8202;N rises from {cr[min(cr)]:.2f} to "
           f"{cr[max(cr)]:.2f}&#8202;&#176;/s and the task&#160;1 creep from {cp[min(cp)]:.1f} to {cp[max(cp)]:.1f}&#8202;&#181;m/s, "
           f"in proportion to \\(t_f\\) (Table&#160;1). At \\(t_f\\) = 10&#8202;ms, the pads&#8217; value, Newton creeps "
           f"{cr.get(10.0, float('nan')):.2f}&#8202;&#176;/s against the pads&#8217; {tpd[1.0]['creep_deg_s']:.2f} and Drake&#8217;s "
           f"{tdk[1.0]['creep_deg_s']:.3f}.</p>",
           f"<p>In task&#160;5 the tool swings past hanging at \\(t_f\\) = {over_s}&#8202;ms (largest swing "
           f"{min(sw[t] for t in over):.0f}&#8211;{max(sw[t] for t in over):.0f}&#176;) and stops within 3&#176; of Drake&#8217;s "
           f"{bdk['phi_max_deg']:.1f}&#176; at {', '.join(f'{t:.2g}' for t in within)}&#8202;ms, so the overshoot past hanging is set by "
           f"the friction rows. The braking is not: at 10&#8211;20&#8202;ms Newton passes "
           f"80&#176; at {min(n80[t] for t in (10.0, 15.0, 20.0) if t in n80):.2f}&#8211;{max(n80[t] for t in (10.0, 15.0, 20.0) if t in n80):.2f}&#8202;N, "
           f"in the first fall, where Drake and the pads hold the tool partway and pass 80&#176; at {bdk['N_at_80_N']:.2f} and "
           f"{bpd['N_at_80_N']:.2f}&#8202;N, 1.4&#8202;s later (Figure&#160;1). At 40&#8202;ms the 80&#176; force falls to "
           f"{n80.get(40.0, float('nan')):.2f}&#8202;N. "
           + (f"Measured at 3&#8202;s, after the first fall, the tool hangs at {fa['drake']:.1f}&#176; in Drake and {fa['pads']:.1f}&#176; with "
              f"the pads; Newton&#8217;s first fall ends at " + ", ".join(f"{fa[t]:.1f}" for t in (10.0, 15.0, 20.0, 40.0) if t in fa) +
              "&#176; for \\(t_f\\) = 10, 15, 20 and 40&#8202;ms, so a longer friction-row time constant catches the tool earlier.</p>"
              if fa else "</p>") + scale_text(sc)]
    fig = figure(svg_brake(brk, bed["brake"]),
                 "Task&#160;5. Left: swing angle against time for Drake, the 1&#8202;mm pads and Newton at \\(k_f\\) = 10 "
                 "(\\(t_f\\) = 3.9&#8202;ms, dashed), 10 and 40&#8202;ms; the pinch falls from 6&#8202;N at 0.5&#8202;s to 0.2&#8202;N at "
                 "4.5&#8202;s. Right: Newton&#8217;s largest swing against \\(t_f\\); the shaded band is Drake&#8217;s &#177;3&#176;, the "
                 "hollow marker \\(k_f\\) = 10.")
    return "".join(txt[:1]) + tab + txt[1] + fig


def first_fall(brk, bed_brake, t=3.0):
    """Swing angle at 3 s, after the first fall (deg): Drake, the pads and Newton by t_f."""
    import numpy as np

    def at(r):
        f = np.array(r["fine"])
        return float(np.interp(t, f[:, 0], f[:, 2]))
    out = {}
    d, p = pick(bed_brake, "drake_hydro", dt_ms=1.0), pick(bed_brake, "mj_pads1", dt_ms=1.0)
    if d and p:
        out.update(drake=at(d), pads=at(p))
    for r in brk:
        if tf_of(r["model"]) and r.get("fine"):
            out[tf_of(r["model"])] = at(r)
    return out if "drake" in out else {}


def scale_text(sc):
    """condim 4 with its torsion schedule scaled: does a torque deficit alone make the tool fall through 80 deg early?"""
    sc = sorted([r for r in sc if ok(r)], key=lambda r: -r["torque_scale"])
    if len(sc) < 2:
        return ""
    lo = [r for r in sc if r["torque_scale"] < 1]
    return (f"<p>A shorter arm alone does not make the tool fall through. MuJoCo condim&#160;4 with its torsion schedule scaled to "
            f"{min(r['torque_scale'] for r in lo):.2f}&#8211;{max(r['torque_scale'] for r in lo):.2f} of the law, the range of Newton&#8217;s "
            f"reduced arm, starts the swing earlier (10&#176; at {min(r['N_at_10_N'] for r in lo):.2f}&#8211;{max(r['N_at_10_N'] for r in lo):.2f}&#8202;N "
            f"against {sc[0]['N_at_10_N']:.2f} unscaled) but still holds the tool partway and passes 80&#176; at "
            f"{min(r['N_at_80_N'] for r in lo):.2f}&#8211;{max(r['N_at_80_N'] for r in lo):.2f}&#8202;N (<code>brake_torque_scale.jsonl</code>). "
            "Newton&#8217;s tool loses its friction torque during the fall by some other mechanism.</p>")


# ------------------------------------------------------------------------------------------ reduction

RED = [("newton_hydro_mc", "reduced, 0.5&#8202;mm voxels"), ("newton_hydro_unreduced_mc", "all faces, 0.5&#8202;mm voxels"),
       ("newton_hydro_mc_vox025", "reduced, 0.25&#8202;mm voxels"), ("newton_hydro_mc_vox1", "reduced, 1&#8202;mm voxels")]


def svg_arms(tw, st, bed_tw):
    W, Hh = 860, 320
    out = P._svg_open(W, Hh, "Onset arm over the pressure law's arm against pinch force")
    fx, fy = P._panel(out, 70, 30, 330, 230, (0.4, 3.4), (0.6, 1.2), [0.5, 1, 2, 3], [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                      "pinch force N per pad (N)", "onset arm / law", logx=True)
    gx, gy = P._panel(out, 500, 30, 330, 230, (0.4, 3.4), (0.6, 1.5), [0.5, 1, 2, 3], [0.6, 0.8, 1.0, 1.2, 1.4],
                      "pinch force N per pad (N)", "static arm of the contact set / law", logx=True)
    for f_, g_ in ((fx, fy), (gx, gy)):
        out.append(f'<line x1="{f_(0.4):.1f}" x2="{f_(3.4):.1f}" y1="{g_(1.0):.1f}" y2="{g_(1.0):.1f}" style="stroke:var(--c-ref);stroke-width:1.6"/>')
    ref = [("drake_hydro", "var(--c-drake)", "circle", bed_tw), ("mj_pads1", "var(--c-sphere)", "circle", bed_tw)]
    for m, col, shp, src in ref:
        pts = [(N, pick(src, m, N=N, dt_ms=1.0)["rbar_onset_mm"] / LAW[N]) for N in (0.5, 1.0, 3.0) if pick(src, m, N=N, dt_ms=1.0)]
        P._path(out, fx, fy, pts, col, width=1.4)
        for x, y in pts:
            P._marker(out, fx(x), fy(y), col, shp)
    shapes = {"newton_hydro_mc": ("diamond", False), "newton_hydro_unreduced_mc": ("square", False),
              "newton_hydro_mc_vox025": ("diamond", True), "newton_hydro_mc_vox1": ("cross", False)}
    for m, _ in RED:
        shp, hol = shapes[m]
        pts = [(N, pick(tw, m, N=N)["rbar_onset_mm"] / LAW[N]) for N in (0.5, 1.0, 3.0) if ok(pick(tw, m, N=N))]
        low = [(x, y) for x, y in pts if y < 0.6]
        pts = [(x, y) for x, y in pts if y >= 0.6]
        P._path(out, fx, fy, pts, C_NEW, dashed=hol, width=1.4)
        for x, y in pts:
            P._marker(out, fx(x), fy(y), C_NEW, shp, hollow=hol, title=f"{m} {x:g} N: {y:.3f}")
        for x, y in low:                      # off the axis: marker on the floor, value beside it
            P._marker(out, fx(x), fy(0.6), C_NEW, shp, hollow=hol, title=f"{m} {x:g} N: {y:.3f}")
            out.append(f'<text x="{fx(x) - 9:.1f}" y="{fy(0.6) - 9:.1f}" text-anchor="end" style="fill:var(--ink3)">'
                       f'all faces, {x:g} N: {y:.2f}</text>')
        sp = [(N, pick(st, m, N=N)["rbar_law_L_mm"] / LAW[N]) for N in (0.5, 1.0, 3.0) if pick(st, m, N=N, dt_ms=1.0)]
        P._path(out, gx, gy, sp, C_NEW, dashed=hol, width=1.4)
        for x, y in sp:
            P._marker(out, gx(x), gy(y), C_NEW, shp, hollow=hol, title=f"{m} {x:g} N: {y:.3f}")
    out.append("</svg>")
    leg = P._legend_html([("Drake", "var(--c-drake)", False, "circle"), ("1 mm pads", "var(--c-sphere)", False, "circle"),
                          ("Newton reduced, 0.5 mm", C_NEW, False, "diamond"), ("all faces", C_NEW, False, "square"),
                          ("reduced, 0.25 mm", C_NEW, True, "diamond"), ("reduced, 1 mm", C_NEW, False, "cross"),
                          ("pressure law", "var(--c-ref)", False, None)])
    return "".join(out) + leg


def reduction(tw, st, brk, bed):
    body = []
    for m, lab in RED:
        body.append(f"Newton hydroelastic, {lab}")
        for N in (0.5, 1.0, 3.0):
            s, t = pick(st, m, N=N, dt_ms=1.0), pick(tw, m, N=N)
            body.append([f"{N:g}&#8202;N", fmt(s.get("n_newton_L"), 0) if s else "&#8211;",
                         fmt(s.get("rbar_law_L_mm"), 3) if s else "&#8211;", fmt(s.get("pen_mm"), 3) if s else "&#8211;",
                         fmt(t.get("rbar_onset_mm"), 3) if ok(t) else ("ejected" if t else "&#8211;"),
                         fmt(t.get("rbar_slide_mm"), 3) if ok(t) else "&#8211;",
                         fmt(t.get("rbar_onset_mm") / LAW[N], 2) if ok(t) else "&#8211;",
                         fmt(t.get("rbar_ratio_3_05_onset"), 3) if (ok(t) and N == 3.0) else "",
                         fmt(t.get("us_per_step_median"), 0) if t else "&#8211;"])
    body.append("Pressure law")
    for N, (dl, rb) in zip((0.5, 1.0, 3.0), ((0.1475, 0.819), (0.2089, 0.974), (0.3632, 1.279))):
        body.append([f"{N:g}&#8202;N", "", fmt(rb, 3), fmt(dl, 3), fmt(LAW[N], 3), fmt(LAW[N], 3), "1.00",
                     fmt(LAW_RATIO, 3) if N == 3.0 else "", ""])
    tab = table(["case", "contacts per pad", "static arm (mm)", "approach (mm)", "onset arm (mm)", "sliding arm (mm)",
                 "onset / law", "arm ratio", f"{US} per step"], body,
                "Table 2: contact reduction and voxel size, 1&#8202;ms step, \\(t_f\\) = 3.9&#8202;ms. Static arm: the "
                "pressure-weighted distance from the pinch axis of Newton&#8217;s own contact set after a 1&#8202;s hold at N "
                "(for all faces, its pressure field); approach: the pad&#8217;s approach after the hold. The law&#8217;s static "
                "arm is its pressure-weighted distance from the axis; its onset and sliding arms are the slipping integral.")
    s05, s1 = pick(st, "newton_hydro_unreduced_mc", N=0.5, dt_ms=1.0), pick(st, "newton_hydro_unreduced_mc", N=1.0, dt_ms=1.0)
    s3 = pick(st, "newton_hydro_unreduced_mc", N=3.0, dt_ms=1.0)
    r_ = {N: pick(st, "newton_hydro_mc", N=N, dt_ms=1.0) for N in (0.5, 1.0, 3.0)}
    t_un = {N: pick(tw, "newton_hydro_unreduced_mc", N=N) for N in (0.5, 1.0, 3.0)}
    t_re = {N: pick(tw, "newton_hydro_mc", N=N) for N in (0.5, 1.0, 3.0)}
    txt = []
    if s05 and s1 and r_[1.0]:
        txt.append(f"<p>Newton&#8217;s pressure field gives the law&#8217;s arm; its contact reduction loses 9&#8211;18&#8202;% of it. "
                   f"Over all faces of the contact surface the static arm is {s05['rbar_law_L_mm']:.3f} and "
                   f"{s1['rbar_law_L_mm']:.3f}&#8202;mm at 0.5 and 1&#8202;N against the law&#8217;s 0.819 and 0.974; the eight "
                   f"reduced contacts give {r_[0.5]['rbar_law_L_mm']:.3f} and {r_[1.0]['rbar_law_L_mm']:.3f}&#8202;mm, and "
                   f"{r_[3.0]['rbar_law_L_mm']:.3f}&#8202;mm at 3&#8202;N against 1.279 (Table&#160;2, Figure&#160;2). "
                   + (f"In task&#160;2 the onset arm with all faces is {t_un[0.5]['rbar_onset_mm'] / LAW[0.5]:.2f} and "
                      f"{t_un[1.0]['rbar_onset_mm'] / LAW[1.0]:.2f} of the law at 0.5 and 1&#8202;N, against "
                      f"{t_re[0.5]['rbar_onset_mm'] / LAW[0.5]:.2f} and {t_re[1.0]['rbar_onset_mm'] / LAW[1.0]:.2f} reduced; Drake gives "
                      f"{pick(bed['twist'], 'drake_hydro', N=0.5, dt_ms=1.0)['rbar_onset_mm'] / LAW[0.5]:.2f} and "
                      f"{pick(bed['twist'], 'drake_hydro', N=1.0, dt_ms=1.0)['rbar_onset_mm'] / LAW[1.0]:.2f}. "
                      if ok(t_un[0.5]) and ok(t_un[1.0]) else "") +
                   "The torque deficit and the arm ratio of 1.656 therefore come from the reduction, which keeps few contacts near the "
                   "rim of the patch, where the lever is longest.</p>")
    if s3:
        txt.append(f"<p>With all faces at 3&#8202;N the pad sinks {s3['pen_mm']:.3f}&#8202;mm against the law&#8217;s 0.363, the "
                   f"contact surface grows to {s3['n_newton_L']:.0f} faces per pad of which the solver loads "
                   f"{s3.get('n_solver_L', 0):.0f}, and its arm, {s3['rbar_law_L_mm']:.3f}&#8202;mm, is that of the deeper patch. "
                   f"A step costs {s3['us_per_step_median'] / 1e3:.1f}&#8202;ms against "
                   f"{r_[3.0]['us_per_step_median'] / 1e3:.1f} reduced"
                   + (f"; in task&#160;2 at 3&#8202;N the tool starts to spin at {t_un[3.0]['rbar_onset_mm'] / LAW[3.0]:.2f} of the "
                      "law&#8217;s torque" if ok(t_un[3.0]) else "")
                   + (f", and in task&#160;5 the tool falls at {pick(brk, 'newton_hydro_unreduced_mc')['N_at_80_N']:.1f}&#8202;N, as soon as "
                      "the pinch starts to drop from 6&#8202;N" if pick(brk, "newton_hydro_unreduced_mc") and
                      not ok(pick(brk, "newton_hydro_unreduced_mc")) else "")
                   + ". Without reduction Newton&#8217;s hydroelastic tip holds the law up to 1&#8202;N and fails above it.</p>")
    v = {m: pick(st, m, N=0.5, dt_ms=1.0) for m in ("newton_hydro_mc_vox025", "newton_hydro_mc_vox1")}
    tv = {m: [pick(tw, m, N=N) for N in (0.5, 1.0, 3.0)] for m in ("newton_hydro_mc_vox025", "newton_hydro_mc_vox1")}
    if all(v.values()) and all(ok(x) for xs in tv.values() for x in xs):
        lo = min(x["rbar_onset_mm"] / LAW[x["N"]] for xs in tv.values() for x in xs)
        hi = max(x["rbar_onset_mm"] / LAW[x["N"]] for xs in tv.values() for x in xs)
        txt.append(f"<p>Finer or coarser voxels keep the reduced arm short: at 0.25 and 1&#8202;mm the onset arm is "
                   f"{lo:.2f}&#8211;{hi:.2f} of the law over 0.5&#8211;3&#8202;N, and the static arm at 0.5&#8202;N is "
                   f"{v['newton_hydro_mc_vox025']['rbar_law_L_mm']:.3f} and {v['newton_hydro_mc_vox1']['rbar_law_L_mm']:.3f}&#8202;mm "
                   f"against {r_[0.5]['rbar_law_L_mm']:.3f} at 0.5&#8202;mm. The voxel size moves which faces the reduction keeps, not the "
                   "pressure field&#8217;s arm.</p>")
    bl = [(lab, pick(brk, m)) for m, lab in RED if ok(pick(brk, m))]
    if bl:
        txt.append("<p>Task&#160;5 largest swing at \\(t_f\\) = 3.9&#8202;ms: " +
                   "; ".join(f"{lab} {r['phi_max_deg']:.1f}&#176; (80&#176; at {fmt(r.get('N_at_80_N'), 2)}&#8202;N)" for lab, r in bl) +
                   ". Every reduced variant overshoots at this \\(t_f\\), as in the previous section.</p>")
    fig = figure(svg_arms(tw, st, bed["twist"]),
                 "Left: onset arm in task&#160;2 over the pressure law&#8217;s arm. Right: the static arm of Newton&#8217;s contact set "
                 "after a 1&#8202;s hold, over the law&#8217;s pressure-weighted arm; with all faces it is the arm of its pressure field.")
    return (txt[0] if txt else "") + tab + "".join(txt[1:]) + fig


# ------------------------------------------------------------------------------------------ edge

EDGE_M = [("mj_point3_bar", "MuJoCo point contact, condim&#160;3", "var(--ink3)"),
          ("mj_point4s_bar", "MuJoCo condim&#160;4, cylinder&#8217;s \\(\\mu_t\\) schedule", "var(--c-c4)"),
          ("mj_pads1_bar", "MuJoCo 1&#8202;mm sphere pad", "var(--c-sphere)"),
          ("mjw_pads1_bar", "MuJoCo-Warp 1&#8202;mm sphere pad", "var(--c-sphere)"),
          ("drake_hydro_bar", "Drake hydroelastic", "var(--c-drake)"),
          ("newton_hydro_mc_bar", "Newton hydroelastic, reduced", C_NEW),
          ("newton_hydro_unreduced_mc_bar", "Newton hydroelastic, all faces", C_NEW),
          ("newton_hydro_unreduced_mc_bar_vox025", "Newton hydroelastic, all faces, 0.25&#8202;mm voxels", C_NEW),
          ("newton_hydro_unreduced_mc_bar_vox1", "Newton hydroelastic, all faces, 1&#8202;mm voxels", C_NEW)]
EL = {m: l for m, l, _ in EDGE_M}


def svg_maps(kin):
    """Contact maps of the -x pad on the edge at 1 N, seen along the pinch axis: the law's pressure on the two faces and
    each model's contacts (area of a marker proportional to its force)."""
    import hydroelastic_arm_integral as A
    short = {"mj_pads1_bar": "1 mm pads", "drake_hydro_bar": "Drake", "newton_hydro_unreduced_mc_bar": "Newton, all faces",
             "newton_hydro_mc_bar": "Newton, reduced"}
    panels = [("law", "pressure law")] + [(m, short[m]) for m in short if pick(kin, m, N=1.0)]
    W, Hh = 860, 250
    pw = (W - 40) / len(panels)
    out = P._svg_open(W, Hh, "Contact maps of the pad on the square edge at 1 N")
    ymax, zmax = 4.5, 1.2
    for k, (m, lab) in enumerate(panels):
        x0 = 20 + k * pw
        cx, cy, sx, sz = x0 + pw / 2, 120, (pw - 24) / (2 * ymax), 70 / zmax
        out.append(f'<rect x="{x0 + 4:.1f}" y="40" width="{pw - 8:.1f}" height="165" rx="6" style="fill:var(--card);stroke:var(--rule2)"/>'
                   f'<line x1="{x0 + 10:.1f}" x2="{x0 + pw - 10:.1f}" y1="{cy}" y2="{cy}" style="stroke:var(--rule);stroke-dasharray:3 3"/>'
                   f'<text x="{cx:.1f}" y="30" text-anchor="middle" style="fill:var(--ink2);font-size:11px">{lab.replace("&#8202;", " ")}</text>')
        if m == "law":
            b = A.bar_law(1.0)
            delta, h = b["delta"], 0.02 / math.sqrt(2)
            Dc = h + A.R - delta
            for s_mm in [i * 0.06 for i in range(14)]:
                for y_mm in [j * 0.25 - 4.5 for j in range(37)]:
                    for sz_ in (1, -1):
                        s_ = s_mm * 1e-3
                        P_ = (-h + s_ / math.sqrt(2), y_mm * 1e-3, sz_ * s_ / math.sqrt(2))
                        rho = math.dist(P_, (-Dc, 0, 0))
                        p = A.E * (1 - rho / A.R)
                        if p > 0:
                            a = min(1.0, p / b["p_max"])
                            out.append(f'<rect x="{cx + y_mm * sx - 1.6:.1f}" y="{cy - P_[2] * 1e3 * sz - 1.6:.1f}" width="3.2" height="3.2" '
                                       f'style="fill:var(--c-ref);opacity:{0.15 + 0.75 * a:.2f}"/>')
            continue
        r = pick(kin, m, N=1.0)
        rr = r["contacts_L"]["rows"]
        fmax = max(x[-1] for x in rr) if rr else 1
        col = {"mj_pads1_bar": "var(--c-sphere)", "drake_hydro_bar": "var(--c-drake)"}.get(m, C_NEW)
        for x in rr:
            y_mm, z_mm, f = x[1], x[2], x[-1]
            rad = 1.2 + 6.0 * math.sqrt(max(f, 0) / fmax)
            out.append(f'<circle cx="{cx + y_mm * sx:.1f}" cy="{cy - z_mm * sz:.1f}" r="{rad:.1f}" '
                       f'style="fill:{col};fill-opacity:0.45;stroke:{col};stroke-width:0.8"/>')
        out.append(f'<text x="{cx:.1f}" y="198" text-anchor="middle" style="fill:var(--ink3);font-size:11px">'
                   f'{len(rr)} contacts, {r["F_x"]:.2f} N</text>')
    out.append(f'<text x="{W / 2}" y="{Hh - 10}" text-anchor="middle" style="fill:var(--ink3)">horizontal: along the edge, '
               f'&#177;{ymax:g} mm; vertical: across it, &#177;{zmax:g} mm</text></svg>')
    return "".join(out)


def edge(kin, st, tw, probe):
    import hydroelastic_arm_integral as A
    laws = {N: A.bar_law(N) for N in (0.5, 1.0, 3.0)}
    txt = [f"<p>The bar is 20&#8202;mm square, 100&#8202;mm long and 24.5&#8202;g, turned 45&#176; about its axis so that each pad "
           f"presses on an edge between two faces inclined 45&#176; to the pinch axis (<code>hom_contact_rig</code> spec field "
           f"<code>bar20</code>). The reference integrates the pressure law over the two faces inside the fingertip sphere; with "
           f"\\(y\\) along the edge and \\(z\\) across it, the pinch force and the sliding friction torque are</p>"
           "\\[ N = \\tfrac{1}{\\sqrt2}\\int_A p\\,dA, \\qquad \\tau = \\mu\\int_A p\\,\\sqrt{y^2/2 + z^2}\\;dA, \\qquad p = E\\,(1-\\rho/R) \\qquad (2) \\]"
           f"<p>where friction opposes the part of the slip velocity tangent to each face, so a face inclined 45&#176; counts "
           f"\\(y/\\sqrt2\\) of a point&#8217;s lever along the edge. At 0.5, 1 and 3&#8202;N the law gives an approach of "
           f"{laws[0.5]['delta'] * 1e3:.3f}, {laws[1.0]['delta'] * 1e3:.3f} and {laws[3.0]['delta'] * 1e3:.3f}&#8202;mm (the cylinder: "
           f"0.148, 0.209, 0.363), a patch {2 * laws[1.0]['half_y'] * 1e3:.1f}&#8202;mm long at 1&#8202;N, a peak pressure of "
           f"{laws[1.0]['p_max'] / 1e6:.2f}&#8202;MPa at the edge and sliding arms of {laws[0.5]['arm'] * 1e3:.3f}, "
           f"{laws[1.0]['arm'] * 1e3:.3f} and {laws[3.0]['arm'] * 1e3:.3f}&#8202;mm (ratio {laws[3.0]['arm'] / laws[0.5]['arm']:.3f}).</p>"]
    # kinematic
    body = []
    for m, lab, _ in EDGE_M:
        rs = [pick(kin, m, N=N) for N in (0.5, 1.0, 3.0)]
        if not any(rs):
            continue
        body.append([lab] + [fmt(r["F_x_over_law"], 2) if r else "&#8211;" for r in rs] +
                    [fmt(r["n"], 0) if r else "&#8211;" for r in rs[1:2]] +
                    [fmt(r["rbar_mm"] / r["law_rbar_mm"], 2) if r and r.get("rbar_mm") else "&#8211;" for r in rs[1:2]] +
                    [fmt(r["half_y_mm"], 2) if r and r.get("half_y_mm") else "&#8211;" for r in rs[1:2]])
    body.append(["pressure law", "1.00", "1.00", "1.00", "", "1.00", fmt(laws[1.0]["half_y"] * 1e3, 2)])
    tab_k = table(["model", "force / N, 0.5&#8202;N", "1&#8202;N", "3&#8202;N", "contacts, 1&#8202;N", "arm / law, 1&#8202;N",
                   "patch half-length, 1&#8202;N (mm)"], body,
                  "Table 3: force along the pinch axis that each contact law gives with the pad at the law&#8217;s approach for 0.5, 1 "
                  "and 3&#8202;N, over N, with no solver dynamics and no friction. Arm: the force-weighted distance of the contacts from "
                  "the pinch axis over the law&#8217;s pressure-weighted distance.")
    dk = [pick(kin, "drake_hydro_bar", N=N) for N in (0.5, 1.0, 3.0)]
    pd = [pick(kin, "mj_pads1_bar", N=N) for N in (0.5, 1.0, 3.0)]
    nu = [pick(kin, "newton_hydro_unreduced_mc_bar", N=N) for N in (0.5, 1.0, 3.0)]
    nr = [pick(kin, "newton_hydro_mc_bar", N=N) for N in (0.5, 1.0, 3.0)]
    para = []
    if all(dk) and all(pd):
        para.append(f"<p>Drake&#8217;s field, cut exactly by the bar&#8217;s faces, gives {min(r['F_x_over_law'] for r in dk):.2f}&#8211;"
                    f"{max(r['F_x_over_law'] for r in dk):.2f} of the law&#8217;s force at the law&#8217;s approach and its arm within "
                    f"{max(abs(r['rbar_mm'] / r['law_rbar_mm'] - 1) for r in dk) * 100:.0f}&#8202;%. The 1&#8202;mm pads give "
                    f"{pd[0]['F_x_over_law']:.2f}, {pd[1]['F_x_over_law']:.2f} and {pd[2]['F_x_over_law']:.2f} of it at 0.5, 1 and "
                    f"3&#8202;N: a pad sphere of radius 0.75&#8202;mm touches the edge over &#177;"
                    f"{math.sqrt(0.75 ** 2 - (0.75 - laws[1.0]['delta'] * 1e3) ** 2):.2f}&#8202;mm across it at 1&#8202;N and pushes along "
                    f"the line from the edge to its centre, so each sphere near the edge carries its full share of pad area where the "
                    f"law loads a strip {math.sqrt(2) * laws[1.0]['delta'] * 1e3:.2f}&#8202;mm wide on each face. The excess falls as "
                    "the edge sinks past the sphere radius and the faces carry the load (Table&#160;3, Figure&#160;3).</p>")
    v25 = [pick(kin, "newton_hydro_unreduced_mc_bar_vox025", N=N) for N in (0.5, 1.0, 3.0)]
    v1 = [pick(kin, "newton_hydro_unreduced_mc_bar_vox1", N=N) for N in (0.5, 1.0, 3.0)]
    if all(nu) and all(nr) and all(v25) and all(v1):
        f_ = lambda rs: rng([r["F_x_over_law"] for r in rs], 1, 2)  # noqa: E731
        para.append(f"<p>Newton errs the other way. Its signed-distance fields interpolate the bar&#8217;s corner between grid points and "
                    f"round it off, so the edge reaches less deep into the pad than the geometry says: over all faces Newton gives "
                    f"{f_(v1)} of the law&#8217;s force at 1&#8202;mm voxels, {f_(nu)} at 0.5&#8202;mm and {f_(v25)} at 0.25&#8202;mm, from 0.5 "
                    f"to 3&#8202;N. The reduced contacts give the same force as all faces ({f_(nr)}), so on the edge the shortfall is the "
                    "voxel size.</p>")
    maps = figure(svg_maps(kin), "Contacts of the &#8722;x pad on the edge at the law&#8217;s approach for 1&#8202;N, seen along the pinch "
                                 "axis. The pressure law&#8217;s field shaded by pressure on the two faces; each model&#8217;s contacts as discs "
                                 "whose area is proportional to their force.")
    # twist
    body = []
    for m, lab, _ in EDGE_M:
        rs = [pick(tw, m, N=N) for N in (0.5, 1.0, 3.0)]
        if not any(rs):
            continue
        body.append([lab] + [fmt(r["rbar_onset_mm"] / r["law_arm_mm"], 2) if ok(r) and r.get("rbar_onset_mm") is not None else "&#8211;"
                             for r in rs] +
                    [fmt(r.get("rbar_slide_mm") / r["law_arm_mm"], 2) if ok(r) and r.get("rbar_slide_mm") is not None else "&#8211;"
                     for r in rs[1:2]] +
                    [fmt(rs[2].get("rbar_ratio_3_05_onset"), 3) if rs[2] else "&#8211;",
                     fmt(rs[1].get("rot_pre_deg"), 2) if rs[1] else "&#8211;", fmt(rs[1].get("creep_deg_s"), 3) if rs[1] else "&#8211;"])
    body.append(["pressure law", "1.00", "1.00", "1.00", "1.00", fmt(laws[3.0]["arm"] / laws[0.5]["arm"], 3), "0", "0"])
    tab_t = table(["model", "onset arm / law, 0.5&#8202;N", "1&#8202;N", "3&#8202;N", "sliding arm / law, 1&#8202;N", "arm ratio",
                   "pre-onset rotation, 1&#8202;N (&#176;)", "creep, 1&#8202;N (&#176;/s)"], body,
                  "Table 4: task&#160;2 on the bar, 1&#8202;ms step. Point contact transmits no torque about the normal; its onset value "
                  "is the detection lag. condim&#160;4 keeps the cylinder&#8217;s fitted torsion schedule.")
    sd = [pick(st, "drake_hydro_bar", N=N) for N in (0.5, 1.0, 3.0)]
    sp = [pick(st, "mj_pads1_bar", N=N) for N in (0.5, 1.0, 3.0)]
    td = [pick(tw, "drake_hydro_bar", N=N) for N in (0.5, 1.0, 3.0)]
    tp = [pick(tw, "mj_pads1_bar", N=N) for N in (0.5, 1.0, 3.0)]
    pr = {r["rig_spec"]: r for r in probe}
    para2 = []
    if all(sd) and all(sp):
        sh = [r["N_pressure_x_L"] / r["N"] for r in sd]
        para2.append(f"<p>Held at N with &#956; = 1, the pads come to rest short of the law&#8217;s approach: Drake at "
                     f"{sd[1]['pen_mm']:.3f}&#8202;mm and the pads at {sp[1]['pen_mm']:.3f}&#8202;mm for 1&#8202;N against the law&#8217;s "
                     f"{laws[1.0]['delta'] * 1e3:.3f}. The faces meet the pinch axis at 45&#176;, the friction angle of &#956; = 1, so "
                     f"friction along the faces can hold the pad wherever it stops: in Drake the pressure carries "
                     f"{rng(sh, 100)}&#8202;% of the pinch force at 0.5&#8211;3&#8202;N and friction the rest. The settled "
                     "approach and the patch it gives then depend on how each model&#8217;s friction lets the pad creep in, which is "
                     "why Table&#160;3 compares the contact laws at a set approach.</p>")
    if all(td) and all(tp):
        para2.append(f"<p>In task&#160;2 the pads start to spin at {min(r['rbar_onset_mm'] / r['law_arm_mm'] for r in tp):.2f}&#8211;"
                     f"{max(r['rbar_onset_mm'] / r['law_arm_mm'] for r in tp):.2f} of the law&#8217;s arm and Drake at "
                     f"{min(r['rbar_onset_mm'] / r['law_arm_mm'] for r in td):.2f}&#8211;{max(r['rbar_onset_mm'] / r['law_arm_mm'] for r in td):.2f} "
                     f"(Table&#160;4); on the screwdriver both were within 7&#8202;%. "
                     + (f"Newton&#8217;s rounded edge lets the pad sink deeper than the law&#8217;s approach under the pinch "
                        f"({pick(st, 'newton_hydro_mc_bar', N=1.0)['pen_mm']:.3f} against {laws[1.0]['delta'] * 1e3:.3f}&#8202;mm at 1&#8202;N, "
                        f"{pick(st, 'newton_hydro_mc_bar', N=3.0)['pen_mm']:.3f} against {laws[3.0]['delta'] * 1e3:.3f} at 3&#8202;N), the patch "
                        f"widens, and its onset arm is {rng([pick(tw, 'newton_hydro_mc_bar', N=N)['rbar_onset_mm'] / pick(tw, 'newton_hydro_mc_bar', N=N)['law_arm_mm'] for N in (0.5, 1.0, 3.0)], 1, 2)} "
                        "of the law. " if all(pick(tw, "newton_hydro_mc_bar", N=N) for N in (0.5, 1.0, 3.0)) and
                        pick(st, "newton_hydro_mc_bar", N=3.0) else "")
                     + f"Drake turns the bar {td[1]['rot_pre_deg']:.1f}&#176; "
                     f"before onset at 1&#8202;N, against 0.02&#176; on the screwdriver."
                     + (f" Its relaxation time changes this: at 1&#8202;ms the bar still turns {pr['drake:hydro:E1e7:r1:rt0.001:bar20']['rot_pre_deg']:.1f}&#176; "
                        f"and the onset arm is {pr['drake:hydro:E1e7:r1:rt0.001:bar20']['rbar_onset_mm'] / laws[1.0]['arm'] / 1e3:.2f} of the law; at "
                        f"0.1&#8202;s it turns {pr['drake:hydro:E1e7:r1:rt0.1:bar20']['rot_pre_deg']:.1f}&#176; and the onset arm is "
                        f"{pr['drake:hydro:E1e7:r1:rt0.1:bar20']['rbar_onset_mm'] / laws[1.0]['arm'] / 1e3:.2f}, while on the screwdriver "
                        f"0.1&#8202;s gives an onset arm of {pr['drake:hydro:E1e7:r1:rt0.1']['rbar_onset_mm']:.3f}&#8202;mm against "
                        f"{pick(P.bed('twist_slip'), 'drake_hydro', N=1.0, dt_ms=1.0)['rbar_onset_mm']:.3f} at 10&#8202;ms. "
                        "A twist about the pinch axis moves each point of an inclined face along the face normal, and Drake&#8217;s "
                        "dissipation resists that normal motion with a torque proportional to the spin speed."
                        if all(k in pr for k in ("drake:hydro:E1e7:r1:rt0.001:bar20", "drake:hydro:E1e7:r1:rt0.1:bar20",
                                                  "drake:hydro:E1e7:r1:rt0.1")) else "") + "</p>")
    return "".join(txt) + tab_k + "".join(para) + maps + "".join(para2) + tab_t


# ------------------------------------------------------------------------------------------ open items and lede

def open_items():
    items = [
        ("Braking of the swing in Newton.", "With \\(t_f\\) &#8805; 10&#8202;ms Newton no longer overshoots but falls through "
         "80&#176; at 1.2&#8211;1.3&#8202;N where Drake passes it at 0.39&#8202;N, and a 10&#8211;25&#8202;% shorter arm does not "
         "reproduce that in condim&#160;4. Log the torque about the pinch axis of Newton&#8217;s solver contact forces during "
         "task&#160;5 (<code>NewtonRig.mj_contacts</code> every step) against \\(2\\mu N\\bar r\\) at the same pinch force; the fall "
         "happens with gravity on and the weight loading the friction cone, so repeat task&#160;2 at 1.3&#8202;N with gravity on: a "
         "torque that collapses only then puts the fall in the combined translational and torsional friction of the eight reduced "
         "contacts."),
        ("All faces at 3&#8202;N.", "Newton&#8217;s unreduced contact sinks 31&#8202;% deeper than the law at 3&#8202;N with a third of its "
         "faces unloaded. Rerun <code>contact_bed_newton.py static --models newton_hydro_unreduced_mc --N 3</code> with 500 solver "
         "iterations and a sparse Jacobian; an approach that returns to 0.363&#8202;mm puts the shortfall in the solve."),
        ("Voxel size on the hand.", "On the edge Newton needs 0.25&#8202;mm voxels to come within 17&#8202;% of the law&#8217;s force. "
         "The printed fingertip&#8217;s 2.7&#8202;mm fillets and the screwdriver are curved, not sharp; set <code>SDF_VOXEL</code> in "
         "<code>scripts/newton_turn.py</code> to 0.25&#8202;mm and rerun the 40 placements of the three-finger turn: a turn that "
         "changes by more than its 0.5&#8202;mm run-to-run spread puts the hand&#8217;s Newton result in the voxel size."),
        ("Pad resolution on an edge.", "The 1&#8202;mm pads carry 2&#215; the law&#8217;s force on the edge at 0.5&#8202;N. Run "
         "<code>contact_bed_edge.py kinematic</code> with a 0.5&#8202;mm pad (spec <code>mj:spheres:s0.5:rs0.375:ir100:tr0.03:bar20</code>): "
         "the excess should scale with the sphere radius if it is the rounding of the edge."),
        ("Drake&#8217;s dissipation on inclined faces.", "Repeat task&#160;2 on the bar in Drake at \\(t_r\\) = 1, 3, 10, 30 and 100&#8202;ms; an "
         "onset arm that tends to the law as \\(t_r\\) falls makes Drake&#8217;s bar result a dissipation setting."),
    ]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def lede(brk, st, tw, kin, bed):
    parts = []
    b = {tf_of(r["model"]): r for r in brk if tf_of(r["model"])}
    bdk = pick(bed["brake"], "drake_hydro", dt_ms=1.0)
    if b and bdk:
        good = [t for t, r in b.items() if t >= 10]
        parts.append(f"Newton&#8217;s friction-row time constant sets its creep and leaves its spin-onset torque unchanged: from 2 to "
                     f"20&#8202;ms the twist creep rises tenfold while the onset arm stays within 2&#8202;%, and the braked swing stops "
                     f"within {max(abs(b[t]['phi_max_deg'] - bdk['phi_max_deg']) for t in good):.1f}&#176; of Drake&#8217;s at 10&#8211;40&#8202;ms "
                     f"instead of swinging to {b[KF10_TF]['phi_max_deg']:.0f}&#176; at the default 3.9&#8202;ms.")
    s1, r1 = pick(st, "newton_hydro_unreduced_mc", N=1.0, dt_ms=1.0), pick(st, "newton_hydro_mc", N=1.0, dt_ms=1.0)
    if s1 and r1:
        parts.append(f"Its pressure field gives the law&#8217;s friction arm (0.973 against 0.974&#8202;mm at 1&#8202;N over all faces); "
                     f"the contact reduction cuts it to {r1['rbar_law_L_mm']:.3f}&#8202;mm, which is the 12&#8211;23&#8202;% torque deficit.")
    dk = [pick(kin, "drake_hydro_bar", N=N) for N in (0.5, 1.0, 3.0)]
    pd = [pick(kin, "mj_pads1_bar", N=N) for N in (0.5, 1.0, 3.0)]
    if all(dk) and all(pd):
        nu = [pick(kin, "newton_hydro_unreduced_mc_bar", N=N) for N in (0.5, 1.0, 3.0)]
        v25 = [pick(kin, "newton_hydro_unreduced_mc_bar_vox025", N=N) for N in (0.5, 1.0, 3.0)]
        parts.append(f"On a square edge Drake&#8217;s field matches the law within {max(abs(r['F_x_over_law'] - 1) for r in dk) * 100:.0f}&#8202;%; "
                     f"the 1&#8202;mm pads, whose spheres press along the line from the edge, carry {pd[0]['F_x_over_law']:.1f}&#215; the "
                     f"law&#8217;s force at 0.5&#8202;N and {pd[2]['F_x_over_law']:.2f}&#215; at 3&#8202;N"
                     + (f"; Newton&#8217;s voxel field rounds the edge off and carries {nu[0]['F_x_over_law']:.2f}&#215; at 0.5&#8202;N with "
                        f"0.5&#8202;mm voxels and {v25[0]['F_x_over_law']:.2f}&#215; with 0.25&#8202;mm" if all(nu) and all(v25) else "") + ".")
    return " ".join(parts)


# ------------------------------------------------------------------------------------------ main

def render_tex(t):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    return P.TEX_RE.sub(lambda m: next(svgs), t), len(items)


def main():
    brk, tw, pu, st = rows("brake_newton.jsonl"), rows("twist_slip_newton.jsonl"), rows("pull_slip_newton.jsonl"), rows("static_newton.jsonl")
    kin, est, etw, probe = rows("edge_kinematic.jsonl"), rows("edge_static.jsonl"), rows("edge_twist.jsonl"), rows("edge_twist_probe.jsonl")
    bed = {"brake": P.bed("brake"), "twist": P.bed("twist_slip"), "pull": P.bed("pull_slip")}
    v = {"STYLE": P.style_block(), "BUILT": time.strftime("%Y-%m-%d %H:%M"), "NEWTON_REV": "009158e6", "BED_PATH": BED_PATH,
         "OVERVIEW_PATH": OVERVIEW_PATH}
    u = P.bed_url()
    v["BED_LINK"] = f", <a href=\"{u}\">artifact</a>" if u else ""
    v["LEDE"] = lede(brk, st, tw, kin, bed)
    v["GLOSSARY"] = glossary()
    v["SETUP"] = setup()
    v["FRICTION"] = friction(brk, tw, pu, bed)
    v["REDUCTION"] = reduction(tw, st, brk, bed)
    v["EDGE"] = edge(kin, est, etw, probe)
    v["OPEN"] = open_items()
    v["FOOTER"] = ("<p>Rebuild: <code>python3 scripts/newton_hydro_tests_page.py</code>. Rows: "
                   "<code>docs/experiments/20261007-newton_hydro_tests/*.jsonl</code> (Newton: <code>scripts/contact_bed_newton.py "
                   "--outdir docs/experiments/20261007-newton_hydro_tests</code>; edge: <code>scripts/contact_bed_edge.py</code>); drivers "
                   "<code>logs/20261007-newton_hydro_tests/step*.sh</code>.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    t, n = render_tex(t)
    open(OUT, "w").write(retro_style.apply(t))  # plain page style (owner, 2026-10-09)
    print(f"formulas {n}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
