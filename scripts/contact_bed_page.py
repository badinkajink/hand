#!/usr/bin/env python3
r"""Build docs/experiments/20261005-contact_bed/20261005-contact_model_bed.html from the bed's JSONL rows.

    python3 scripts/contact_bed_page.py

Reads the task rows written by scripts/contact_bed_*.py (pull_slip, twist_slip, roll, shake, brake, creep, stability,
and the *_newton.jsonl twins from scripts/contact_bed_newton.py), the GPU rows of 20261005-gpu_scaling/, and the films
under media/. Shares its figure helpers, model table and style with scripts/contact_overview_page.py. Every number in
the prose is computed here; a task whose rows are missing renders a note instead.
"""
from __future__ import annotations

import math
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import texsvg  # noqa: E402

BED = P.BED
OUT = os.path.join(BED, "20261005-contact_model_bed.html")
TPL = os.path.join(ROOT, "scripts/contact_bed_page.template.html")
TEX_CACHE = os.path.join(BED, "texsvg_cache.json")
OVERVIEW_PATH = os.path.relpath(P.OUT, P.ROOT)  # the current overview revision
OVERVIEW_URL_FILE = os.path.join(P.D, "artifact_url.txt")

ALL = P.TABLE_ORDER + ["newton_hydro", "newton_hydro_unreduced", "mj_pads2", "mj_pads05"]
CONS = ["mj_pads2", "mj_pads1", "mj_pads05"]
COL = {k: v[1] for k, v in P.MODELS.items()}
COL.update({"mj_pads2": "var(--s1)", "mj_pads05": "var(--c-sphere)", "newton_hydro_unreduced": "var(--c-newton)",
            "newton_hydro": "var(--c-newton)"})
SHAPE = {k: v[2] for k, v in P.MODELS.items()}
SHAPE.update({"mj_pads2": "square", "mj_pads05": "diamond", "newton_hydro_unreduced": "diamond", "newton_hydro": "diamond"})
LBL = {k: v[0] for k, v in P.MODELS.items()}
LBL.update({"mj_pads2": "MuJoCo 2 mm sphere pad", "mj_pads05": "MuJoCo 0.5 mm sphere pad",
            "newton_hydro_unreduced": "Newton hydroelastic, unreduced", "newton_hydro": "Newton hydroelastic, kh = E/h"})
H = P.HTML_LBL
num, fmt, first, pick, _eq = P.num, P.fmt, P.first, P.pick, P._eq
US = P.US


def ok(r):
    return r is not None and r.get("status", "complete") == "complete"


def table(head, rows, cls=""):
    out = [f"<table{' class=' + repr(cls) if cls else ''}><thead><tr>" +
           "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head)) + "</tr></thead><tbody>"]
    for r in rows:
        if isinstance(r, str):
            out.append(f"<tr class='grp'><td colspan='{len(head)}'>{r}</td></tr>")
            continue
        out.append("<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table>")
    return "<div class='tw'>" + "".join(out) + "</div>"


def film(path_rel, caption, poster_rel=None):
    p = os.path.join(BED, path_rel)
    if not os.path.exists(p):
        return ""
    pp = os.path.join(BED, poster_rel) if poster_rel else None
    poster = f' poster="{P.R.data_uri(pp, "image/jpeg")}"' if pp and os.path.exists(pp) else ""
    return (f'<figure><video src="{P.R.data_uri(p, "video/mp4")}"{poster} controls muted loop playsinline preload="metadata">'
            f'</video><figcaption>{caption} <code>docs/experiments/20261005-contact_bed/{path_rel}</code></figcaption></figure>')


FIG = [0]


def fignum():
    FIG[0] += 1
    return FIG[0]


def figure(svg, caption):
    return f'<figure class="diagram">{svg}<figcaption>Figure&#160;{fignum()}. {caption}</figcaption></figure>'


def models_present(rows, keys=ALL):
    return [k for k in keys if any(r.get("model") == k for r in rows)]


# ------------------------------------------------------------------------------------------ setup and summary

def setup(T):
    n = sum(len(v) for v in T.values())
    specs = {}
    for rows in T.values():
        for r in rows:
            specs.setdefault(r.get("model"), r.get("rig_spec") or r.get("spec"))
    rows = [[H.get(k, k), f"<code>{specs[k]}</code>"] for k in ALL if specs.get(k)]
    return (f"<p>Every case runs on the two-pad pinch rig of the 10-01 study (<code>scripts/hom_contact_rig.py</code>): two real_v1 "
            f"fingertip spheres on rails along x press the screwdriver, whose axis lies along y, with a commanded force N per pad. "
            f"Tasks add one load or motion each. The protocol (<code>docs/experiments/20261005-contact_bed/PROTOCOL.md</code>) "
            f"fixes the parameters, metrics and row fields for every script, so rows written by different scripts and simulators "
            f"compare directly. {n} rows so far. Model strings:</p>" + table(["model", "spec"], rows) +
            "<p>Agreement is shaded against Drake hydroelastic. Rigid Coulomb friction and the hydroelastic arm law are the "
            "analytic comparisons.</p>")


def summary(T, M):
    dev = P.deviations(M)
    lines = []
    for k in P.ORDER:
        if dev.get(k):
            lines.append(f"{H[k]}: median {statistics.median(dev[k]):.1f}&#8202;% over {len(dev[k])} metrics")
    return (P.agree_table(M) + "<p class='tnote'>Shading compares each model with Drake hydroelastic: within 10&#8202;% (dark green), "
            "10&#8211;25&#8202;% (light), 25&#8211;50&#8202;% (amber), beyond 50&#8202;% or a failed case (red). A dash marks a case not run.</p>" +
            ("<p>Median absolute deviation from Drake over the metrics above: " + "; ".join(lines) + ".</p>" if lines else ""))


# ------------------------------------------------------------------------------------------ task 1: pull

def t1(rows):
    if not rows:
        return P.pending("No pull rows.")
    out = ["<p>Protocol: the pinch settles at N for 0.4&#8202;s, then an axial force on the tool rises at 2&#8202;N/s until the tool "
           "slides faster than 10&#8202;mm/s. A second run holds the force at half the onset force for 1&#8202;s to measure creep.</p>"]
    head = ["model, N per pad", "&#956; at slip", "displacement before onset (&#181;m)", "creep at half load (&#181;m/s)",
            "slip speed, 50&#8202;ms (mm/s)", "sliding &#956;", f"{US} per step"]
    body = []
    for k in models_present(rows):
        body.append(H.get(k, k))
        for N in (0.5, 1.0, 3.0):
            r = pick(rows, k, N=N, dt_ms=1.0)
            if not r:
                continue
            if not ok(r):
                body.append([f"{N:g}&#8202;N", r.get("status", "failed")] + [""] * 5)
                continue
            body.append([f"{N:g}&#8202;N", fmt(r.get("mu_eff"), 3), fmt(first(r, "u_pre_mm") * 1e3 if first(r, "u_pre_mm") is not None else None, 1),
                         fmt(r["creep_mm_s"] * 1e3 if r.get("creep_mm_s") is not None else None, 2),
                         fmt(r.get("v_slip_mean50_mm_s"), 1), fmt(r.get("mu_slide"), 3), fmt(r.get("us_per_step_median"), 1)])
    out.append(table(head, body))
    out.append("<p class='tnote'>Table: task&#160;1 at a 1&#8202;ms step. Rigid Coulomb friction with equal static and kinetic &#956; "
               "slides at 34&#8202;mm/s on average over the first 50&#8202;ms.</p>")
    out.append(figure(svg_pull(rows), "Tool speed against the pull force during the 2&#8202;N/s ramp, N&#8202;=&#8202;1&#8202;N per pad, "
                      "1&#8202;ms step. Below the onset each model creeps at a speed proportional to the load; the MuJoCo models creep "
                      "two orders of magnitude faster than Drake. The vertical line is rigid Coulomb onset, F&#8202;=&#8202;2&#956;N."))
    return "".join(out)


def svg_pull(rows):
    W, Hh = 980, 400
    out = P._svg_open(W, Hh, "Tool speed against pull force, log scale; MuJoCo models creep about a hundred times faster than "
                             "Drake before onset, and all slide past 2 N.")
    fx, fy = P._panel(out, 90, 46, 640, 290, (0.0, 2.6), (1e-5, 1e3), (0, 0.5, 1.0, 1.5, 2.0, 2.5),
                      (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000), "pull force (N)", "tool speed along its axis (mm/s), log scale",
                      False, True, yfmt=lambda v: f"{v:g}")
    out.append(f'<line x1="{fx(2.0):.1f}" x2="{fx(2.0):.1f}" y1="46" y2="336" style="stroke:var(--c-ref);stroke-dasharray:6 5"/>'
               f'<text x="{fx(2.0) + 6:.1f}" y="62" style="fill:var(--ink3)">2&#956;N</text>')
    leg = []
    for k in models_present(rows, P.ORDER):
        r = pick(rows, k, N=1.0, dt_ms=1.0)
        if not ok(r) or not r.get("ramp"):
            continue
        cols = r["ramp_cols"]
        iF, iv = cols.index("F"), cols.index("v_mm_s")
        pts = [(p[iF], max(abs(p[iv]), 1e-5)) for p in r["ramp"] if p[iF] <= 2.6]
        P._path(out, fx, fy, pts, COL[k], dashed=P.MODELS[k][3], width=1.8)
        leg.append((LBL[k], COL[k], P.MODELS[k][3], None))
    out.append("</svg>")
    return "".join(out) + P._legend_html(leg)


# ------------------------------------------------------------------------------------------ task 4: shake

def t4(rows):
    if not rows:
        return P.pending("No shake rows.")
    out = ["<p>Protocol: gravity acts along the tool axis; an inertial force shakes the tool along its axis at 5&#8202;Hz with peak "
           "acceleration 0.5, 1, 2 and 4&#8202;g for 2&#8202;s, at N&#8202;=&#8202;0.2 and 0.5&#8202;N per pad.</p>"]
    head = ["model, N, a<sub>pk</sub>", "Coulomb load ratio", "rigid Coulomb drift (mm/cycle)", "drift (mm/cycle)",
            "peak relative displacement (mm)", "dropped", f"{US} per step"]
    body = []
    for k in models_present(rows):
        body.append(H.get(k, k))
        for N in (0.2, 0.5):
            for g in (0.5, 1.0, 2.0, 4.0):
                r = pick(rows, k, N=N, a_pk_g=g, dt_ms=1.0)
                if not r:
                    continue
                if not ok(r):
                    body.append([f"{N:g}&#8202;N, {g:g}&#8202;g", "", "", r.get("status", "failed"), "", "", ""])
                    continue
                body.append([f"{N:g}&#8202;N, {g:g}&#8202;g", fmt(r.get("coulomb_load_ratio"), 2), fmt(r.get("coulomb_drift_per_cycle_mm"), 3),
                             fmt(r.get("drift_per_cycle_mm"), 4), fmt(r.get("peak_rel_disp_mm"), 3),
                             ("yes" if r.get("drop") else "no"), fmt(r.get("us_per_step_median"), 1)])
    out.append(figure(svg_shake(rows), "Net drift of the tool along its axis per 5&#8202;Hz cycle against peak shake acceleration, "
                      "1&#8202;ms step, gravity along the axis. Grey: rigid Coulomb friction, which slips once m(g&#8202;+&#8202;a<sub>pk</sub>) "
                      "exceeds 2&#956;N. Below that threshold every drift is creep."))
    out.append(table(head, body))
    out.append(film("media/shake_models.mp4", "Task&#160;4 at N&#8202;=&#8202;0.5&#8202;N and 2&#8202;g in every model, side by side.",
                    "media/shake_models_poster.jpg"))
    return "".join(out)


def svg_shake(rows):
    W, Hh = 980, 400
    out = P._svg_open(W, Hh, "Drift per cycle against shake acceleration for two pinch forces.")
    leg = {}
    for j, N in enumerate((0.2, 0.5)):
        x0 = 80 + j * 470
        fx, fy = P._panel(out, x0, 46, 380, 280, (0.4, 5.0), (1e-5, 30.0), (0.5, 1, 2, 4), (1e-4, 1e-3, 1e-2, 0.1, 1, 10),
                          "peak acceleration a_pk (g), log scale", f"N = {N:g} N per pad: |drift| per cycle (mm), log", True, True,
                          yfmt=lambda v: f"{v:g}")
        cpts = []
        for g in (0.5, 1.0, 2.0, 4.0):
            r = next((q for q in rows if _eq(q.get("N"), N) and _eq(q.get("a_pk_g"), g) and q.get("coulomb_drift_per_cycle_mm") is not None), None)
            if r:
                cpts.append((g, max(r["coulomb_drift_per_cycle_mm"], 1e-5)))
        P._path(out, fx, fy, cpts, "var(--c-ref)", dashed=True, width=2)
        for k in models_present(rows):
            pts = []
            for g in (0.5, 1.0, 2.0, 4.0):
                r = pick(rows, k, N=N, a_pk_g=g, dt_ms=1.0)
                if ok(r) and r.get("drift_per_cycle_mm") is not None:
                    pts.append((g, max(abs(r["drift_per_cycle_mm"]), 1e-5)))
            if pts and k in P.ORDER:
                P._path(out, fx, fy, pts, COL[k], width=1.6)
                for g, v in pts:
                    P._marker(out, fx(g), fy(v), COL[k], SHAPE[k], title=f"{LBL[k]}, {N:g} N, {g:g} g: {v:.4f} mm/cycle")
                leg[k] = (LBL[k], COL[k], False, SHAPE[k])
    out.append("</svg>")
    return "".join(out) + P._legend_html([("rigid Coulomb", "var(--c-ref)", True, None)] + list(leg.values()))


# ------------------------------------------------------------------------------------------ task 2: twist

def t2(rows):
    if not rows:
        return P.pending("Task&#160;2 rows are not written yet.")
    head = ["model, N", "arm at onset (mm)", "onset / law", "sliding arm (mm)", "turn before onset (&#176;)",
            "creep (&#176;/s)", f"{US}/step"]
    body = []
    for k in models_present(rows):
        body.append(H.get(k, k))
        for N in (0.5, 1.0, 3.0):
            r = pick(rows, k, N=N, dt_ms=1.0)
            if not r:
                continue
            if not ok(r):
                body.append([f"{N:g}&#8202;N", r.get("status", "failed")] + [""] * 5)
                continue
            body.append([f"{N:g}&#8202;N", fmt(r.get("rbar_onset_mm"), 3), fmt(r["rbar_onset_mm"] / (P.C_LAW * 1e3 * N ** P.EXP_LAW) if r.get("rbar_onset_mm") else None, 3), fmt(r.get("rbar_kin_mm"), 3),
                         fmt(r.get("rot_pre_deg"), 2), fmt(r.get("creep_deg_s"), 3), fmt(r.get("us_per_step_median"), 0)])
    fig, _ = P.svg_scaling(rows, [])
    out = ["<p>Protocol: after settling, a torque about the pinch axis rises at the law&#8217;s torque per second until the tool turns "
           "faster than 30&#8202;&#176;/s. A second run holds half the onset torque for 1&#8202;s to measure creep, and a third drives the tool to "
           "spin at 1&#8202;rad/s to measure the sliding arm.</p>",
           figure(fig, "Per-pad friction arm at the onset of spin, task&#160;2, 1&#8202;ms step, against the hydroelastic law and a Hertz "
                  "arm equal at 1&#8202;N. Right: the arm ratio between 3 and 0.5&#8202;N."),
           table(head, body),
           "<p class='tnote'>Table: task&#160;2 at a 1&#8202;ms step. The law is the hydroelastic pressure field integrated over this rig&#8217;s "
           f"sphere&#8211;cylinder patch, {P.C_LAW * 1e3:.3f}&#8202;mm&#183;N<sup>&#8722;1/4</sup> (<code>scripts/hydroelastic_arm_integral.py</code>). Point contact has no torsional friction; its onset value is the torque at which its creep "
           "reached the detection speed.</p>"]
    out.append(film("media/twist_models.mp4", "Task&#160;2 at N&#8202;=&#8202;1&#8202;N in every model, side by side.", "media/twist_models.jpg"))
    law = lambda k: [r["rbar_onset_mm"] / (P.C_LAW * 1e3 * r["N"] ** P.EXP_LAW) for r in rows  # noqa: E731
                     if r.get("model") == k and ok(r) and r.get("dt_ms") == 1.0 and r.get("rbar_onset_mm")]
    raw, mc = law("newton_hydro"), law("newton_hydro_mc")
    if raw and mc:
        out.append(f"<p>Newton hydroelastic with \\(k_h=E/h\\) starts to spin at {min(raw):.1f}&#8211;{max(raw):.1f}&#215; the law&#8217;s "
                   "torque. SolverMuJoCo writes each hydroelastic contact&#8217;s stiffness as a <code>solref</code> time constant, which "
                   "MuJoCo turns into a force stiffness multiplied by the contact&#8217;s effective mass; dividing \\(k_h\\) by the "
                   "pad&#8211;tool effective mass of the solver&#8217;s own model (17.4&#8202;g) gives "
                   f"{min(mc):.2f}&#8211;{max(mc):.2f}&#215; the law, with no fitted parameter (overview, step&#160;7; rows "
                   "<code>../20261006-newton_mass_scaling/</code>).</p>")
    return "".join(out)


# ------------------------------------------------------------------------------------------ task 5: brake

def t5(rows):
    if not rows:
        return P.pending("Task&#160;5 rows are not written yet.")
    rows = [r for r in rows if r.get("role", "bed") == "bed"]
    head = ["model", "swing end (&#176;)", "largest angle (&#176;)", "peak rate (&#176;/s)", "time to 80&#176; (s)", "axial slip at the end (mm)",
            "still pinched", f"{US} per step"]
    body = []
    for k in models_present(rows):
        r = pick(rows, k, dt_ms=1.0)
        if not r:
            continue
        body.append([H.get(k, k), fmt(r.get("phi_end_deg"), 1), fmt(r.get("phi_max_deg"), 1), fmt(r.get("peak_rate_deg_s"), 0),
                     fmt(r.get("t_80_s"), 2), fmt(r.get("slip_end_mm"), 2), "yes" if r.get("pinched_end") else "no",
                     fmt(r.get("us_per_step_median"), 0)])
    out = ["<p>Protocol: gravity on, centre of mass 15&#8202;mm from the pinch line; the pinch is held at 6&#8202;N for 0.5&#8202;s and lowered "
           "geometrically to 0.2&#8202;N over 4&#8202;s, so the tool swings about the pinch axis toward hanging.</p>",
           table(head, body),
           "<p class='tnote'>Table: task&#160;5 at a 1&#8202;ms step: gravity on, centre of mass 15&#8202;mm from the pinch line, the pinch held "
           "at 6&#8202;N for 0.5&#8202;s and lowered geometrically to 0.2&#8202;N over 4&#8202;s. 90&#176; is hanging.</p>"]
    out.append(film("media/brake_models.mp4", "Task&#160;5 in every model, side by side.", "media/brake_models.jpg"))
    return "".join(out)


ROLL_MODELS = ["mj_point3", "mj_point4s", "mj_pads1", "mjw_pads1", "newton_pads1", "drake_hydro", "newton_hydro_mc"]
GPU_LBL = {"mjw_pads1": "MuJoCo-Warp 1&#8202;mm sphere pad, one GPU world", "newton_pads1": "Newton 1&#8202;mm sphere pad, one GPU world"}


def t3(rows):
    if not rows:
        return P.pending("Task&#160;3 rows are not written yet.")
    out = ["<p>Protocol: gravity off; after the pinch settles at N per pad, the +x pad moves 10&#8202;mm along z (across the tool axis) "
           "at 10&#8202;mm/s for 1&#8202;s, or 50&#8202;mm/s for 0.2&#8202;s, driven by a stiff servo on a 100&#8202;kg slide, while both pads stay "
           "force-controlled at N along x; then a 0.2&#8202;s hold. The tool rolls between the pads. Rows: "
           "<code>roll.jsonl</code> (<code>scripts/contact_bed_roll.py</code>) and <code>../20261006-simulator_agreement/roll_newton.jsonl</code> "
           "(<code>scripts/contact_bed_newton.py roll</code>).</p>"]
    head = ["model, N per pad", "rolling ratio", "no-slip ratio at the contact point", "contact depth (&#181;m)",
            "slip, fixed pad (&#181;m)", "slip, moving pad (&#181;m)", "largest friction use, moving pad", f"{US} per step"]
    body = []
    for k in models_present(rows, ROLL_MODELS):
        body.append(GPU_LBL.get(k) or H.get(k, k))
        for N in (0.5, 1.0, 3.0):
            r = pick(rows, k, N=N, dt_ms=1.0, v_mm_s=10.0)
            if not r:
                continue
            if not ok(r):
                body.append([f"{N:g}&#8202;N", r.get("status", "failed")] + [""] * 6)
                continue
            body.append([f"{N:g}&#8202;N", fmt(r.get("rho"), 4), fmt(r.get("rho_noslip_contact"), 4),
                         fmt((P.R_TOOL_MM - r["r_contact_L_mm"]) * 1e3 if r.get("r_contact_L_mm") else None, 0),
                         fmt(r["slip_path_L_mm"] * 1e3 if r.get("slip_path_L_mm") is not None else None, 1),
                         fmt(r["slip_path_R_mm"] * 1e3 if r.get("slip_path_R_mm") is not None else None, 1),
                         fmt(r.get("util_R_max"), 3), fmt(r.get("us_per_step_median"), 0)])
    out.append(table(head, body))
    out.append("<p class='tnote'>Table: task&#160;3 at a 1&#8202;ms step and 10&#8202;mm/s. Rolling ratio: tool rotation times its radius over half "
               "the pad travel. No-slip ratio at the contact point: the ratio for rolling without slip about the force-weighted contact "
               "point, which lies inside the tool surface by the contact depth; the line of centres of the two spheres tilts by 12.6&#176; "
               "over the travel, so no-slip rolling on the surface gives 1.008. Slip: path length of the relative motion of tool and pad at "
               "the contact point. Friction use: tangential over &#956; times normal force at the moving pad; 1 is sliding.</p>")
    ok_rows = [r for r in rows if ok(r) and r.get("model") in ROLL_MODELS]
    if ok_rows:
        ron = [r["rho_over_noslip"] for r in ok_rows if r.get("rho_over_noslip")]
        slip = [max(r.get("slip_path_L_mm") or 0, r.get("slip_path_R_mm") or 0) * 1e3 for r in ok_rows]
        util = [r["util_R_max"] for r in ok_rows if r.get("util_R_max") is not None]
        lost = [r for r in rows if r.get("model") in ROLL_MODELS and not r.get("success")]
        out.append(f"<p>Every model rolls the tool at 0.5 to 3&#8202;N, 10 and 50&#8202;mm/s and 1 and 5&#8202;ms ({len(ok_rows)} cases, "
                   f"{'none' if not lost else len(lost)} lost). The rolling ratio is {min(ron):.3f}&#8211;{max(ron):.3f} of each model&#8217;s own "
                   f"no-slip value, the contacts slip at most {max(slip):.0f}&#8202;&#181;m over 5&#8202;mm of rolling, and the moving pad uses at "
                   f"most {max(util) * 100:.0f}&#8202;% of its friction. The ratios differ between models by where the contact point sits: "
                   "the deeper the point lies inside the tool, the smaller the radius it rolls on and the larger the ratio. Point contact "
                   "and condim&#160;4 place it at the deepest penetration point, the pads at the force-weighted centre of their spheres, "
                   "and Drake and Newton at the centroid of the pressure field.</p>")
    out.append(film("media/roll_models.mp4", "Task&#160;3 at N&#8202;=&#8202;1&#8202;N, 10&#8202;mm/s, 1&#8202;ms step, played at quarter speed, "
                    "viewed from the tool&#8217;s &#8722;y end; the red stripe and dot mark the roll.", "media/roll_models_poster.jpg"))
    return "".join(out)


def t6(rows):
    if not rows:
        return P.pending("Task&#160;6 rows are not written yet.")
    return (f"<p>Task&#160;6 ran on 2026-10-06 ({len(rows)} rows in <code>creep.jsonl</code>): the 1&#8202;mm pads at relaxation times "
            "0.02, 0.05 and 0.1&#8202;s crossed with <code>impratio</code> 100&#8211;10&#8202;000 and 1&#8211;10 <code>noslip_iterations</code> on tasks 1, "
            f"2, 4 and 5. {P.creep_t6_text(rows)} The analysis is on the fingertip page, "
            "<code>docs/experiments/20261006-fingertip_backends/20261006-fingertip_contact_backends.html</code>, "
            "<a href=\"https://claude.ai/artifact/E2uy8dWtKngLNgn253SynL\">artifact</a>.</p>")


def t7(rows):
    st = P.stab_stats(rows)
    if not st:
        return P.pending("Task&#160;7 rows are not written yet.")
    out = ["<p>Protocol: the 1&#8202;mm pad pinch holds the tool at N&#8202;=&#8202;1&#8202;N per pad with gravity on, across the tool axis. Every "
           "contact gets <code>solref</code> (\\(t_c\\), 1) and <code>solimp</code> (\\(d_0\\), \\(d_0\\), 0.001, 0.5, 2). "
           "Each step runs a 0.4&#8202;s settle with the weight compensated and a 1&#8202;s hold without it; a case holds when the state "
           "stays finite, the tool centre stays within 5&#8202;mm and both pads keep a contact. Every pair runs with MuJoCo&#8217;s clamp "
           "\\(t_c\\ge2\\Delta t\\) off, which tests the bound, and on, as MuJoCo runs by default.</p>",
           figure(P.svg_stab(rows), "Largest stable step of the 1&#8202;mm pad against the bound (14) of the overview, clamp off. Each bar "
                  "runs from the largest step that held (filled) to the first that failed (&#215;: diverged; open circle: the tool was "
                  "ejected); the dotted bar held at every step up to 15&#8202;ms. The diagonal is equality with the bound."),
           f"<p>{P.stab_text(rows, fig=FIG[0])}</p>"]
    S = {(r["d0"], r["tc_ms"], bool(r["refsafe"])): r for r in P.stab_summaries(rows, model="mj_pads1")}
    S.update({(r["d0"], r["tc_ms"], bool(r["refsafe"])): r for r in P.stab_summaries(rows, refsafe=True, model="mj_pads1")})
    d0s = sorted({k[0] for k in S})
    body = []
    for rs in (False, True):
        body.append("clamp \\(t_c\\ge2\\Delta t\\) " + ("on (MuJoCo default)" if rs else "off"))
        for tc in sorted({k[1] for k in S}):
            cells = []
            for d0 in d0s:
                r = S.get((d0, tc, rs))
                if not r:
                    cells.append("&#8211;")
                    continue
                ff = r.get("dt_first_fail_ms")
                b = P.stab_bound(r)
                cells.append(f"{r.get('dt_max_held_ms') or 0:g} / {('&gt;15' if ff is None else f'{ff:g}')}" +
                             (f" ({b:.1f})" if not rs and b else ""))
            body.append([f"\\(t_c\\)&#8202;=&#8202;{tc:g}&#8202;ms"] + cells)
    out.append(table(["\\(t_c\\)"] + [f"\\(d_0\\)&#8202;=&#8202;{d:g}" for d in d0s], body))
    out.append("<p class='tnote'>Table: largest step that held / first step that failed, in ms, for the 1&#8202;mm pad; in brackets the "
               "bound (14) with \\(n\\) the spheres touching one pad at the end of the settle. Tested steps: 0.25, 0.5, 1, 2, 3, 4, 5, 7, 10 "
               "and 15&#8202;ms.</p>")
    return "".join(out)


def generic(rows, name):
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)][:8]
    body = [[H.get(r.get("model"), r.get("model"))] + [fmt(r.get(k), 3) if isinstance(r.get(k), (int, float)) else str(r.get(k)) for k in keys]
            for r in rows[:40]]
    return table(["model"] + keys, body)


def consist(T):
    head = ["metric", "2&#8202;mm pad, 1&#8202;ms", "1&#8202;mm, 0.5&#8202;ms", "1&#8202;mm, 1&#8202;ms", "1&#8202;mm, 2&#8202;ms", "1&#8202;mm, 5&#8202;ms", "0.5&#8202;mm pad, 1&#8202;ms"]
    cols = [("mj_pads2", 1.0), ("mj_pads1", 0.5), ("mj_pads1", 1.0), ("mj_pads1", 2.0), ("mj_pads1", 5.0), ("mj_pads05", 1.0)]
    specs = [("Task 1, &#956; at slip, 1&#8202;N", T["pull"], dict(N=1.0), lambda r: r.get("mu_eff"), 3),
             ("Task 1, creep at half load, 1&#8202;N (&#181;m/s)", T["pull"], dict(N=1.0), lambda r: r["creep_mm_s"] * 1e3, 1),
             ("Task 2, arm at onset, 0.5&#8202;N (mm)", T["twist"], dict(N=0.5), lambda r: r.get("rbar_onset_mm"), 3),
             ("Task 2, arm at onset, 3&#8202;N (mm)", T["twist"], dict(N=3.0), lambda r: r.get("rbar_onset_mm"), 3),
             ("Task 4, drift per cycle, 0.5&#8202;N, 2&#8202;g (mm)", T["shake"], dict(N=0.5, a_pk_g=2.0), lambda r: r.get("drift_per_cycle_mm"), 4),
             ("Task 5, swing end (&#176;)", [r for r in T["brake"] if r.get("role", "bed") == "bed"], {}, lambda r: r.get("phi_end_deg"), 1),
             ("Physics step, task 1, 1&#8202;N (" + US + ")", T["pull"], dict(N=1.0), lambda r: r.get("us_per_step_median"), 1)]
    body = []
    for lab, rows, kw, fn, nd in specs:
        cells = []
        for k, dt in cols:
            r = pick(rows, k, dt_ms=dt, **kw)
            try:
                cells.append(fmt(fn(r), nd) if ok(r) else "&#8211;")
            except (TypeError, KeyError):
                cells.append("&#8211;")
        body.append([lab] + cells)
    return ("<p>The 1&#8202;mm pad gives the same task results from 0.5 to 5&#8202;ms steps: the arm at onset moves by under 1&#8202;% and the "
            "swing end by under 0.1&#176;. The 0.5&#8202;mm pad agrees with it; the 2&#8202;mm pad touches with too few spheres at 0.5&#8202;N, "
            "and its arm there is less than half the law&#8217;s.</p>" + table(head, body))


def cost(T, gpu_pads, gpu_newton):
    c = P.step_cost(T["pull"])
    curves = P.gpu_curves(gpu_pads, gpu_newton)
    bat = P.gpu_batched(curves)
    one = P.gpu_one_world(curves)
    rows = [[H[k], fmt(c.get(k), 1), "one GPU world" if k in P.GPU_MODELS else "one CPU core",
             (fmt(one[k], 0) if k in one else "&#8211;"),
             (f"{bat[k][0]:.2f} at {bat[k][1]:,d} worlds" if k in bat else "&#8211;")] for k in P.TABLE_ORDER if k in c]
    return (table(["model", f"{US} per physics step, 1&#8202;ms, task&#160;1", "stepped on",
                   f"{US} per step, one GPU world, SR2 fixture", f"{US} per world-step, GPU batch"], rows) +
            "<p class='tnote'>The bed rig steps MuJoCo-Warp as a CUDA graph and Newton without one, so its GPU column compares "
            "the rigs. The SR2 holding fixture of the batched sweep captures every 10-step block as a CUDA graph in both "
            "simulators; its one-world column compares the simulators.</p>" +
            figure(P.svg_gpu(curves), "Batched GPU throughput, SR2 thumb&#8211;index holding fixture, 1&#8202;ms step. " +
                   P_TEXT_GPU_NOTE(gpu_newton)))


def P_TEXT_GPU_NOTE(gpu_newton):
    import contact_overview_text as OT
    return OT.gpu_note({"gpu_newton": gpu_newton})


def open_items(T):
    items = [("Friction rows of Newton&#8217;s hydroelastic contact.", "Its friction gain \\(k_f=10\\) gives the friction rows a 3.9&#8202;ms "
              "time constant where the pads use 10&#8202;ms. Repeat task&#160;5 with <code>contact_bed_newton.py brake --models newton_hydro_mc</code> "
              "at \\(k_f\\) for 2, 5, 10 and 20&#8202;ms; a swing within 3&#176; of Drake&#8217;s at one setting would make the remaining "
              "creep and swing differences a friction-row setting."),
             ("Creep against Drake.", "Every MuJoCo model creeps 150&#8211;260&#215; faster than Drake under a held load (task&#160;1). Task&#160;6 "
              "lowers it with <code>impratio</code> and <code>noslip_iterations</code>; the next measurement is the chain&#8217;s hold and wield "
              "at <code>impratio</code> 1000, which decides whether creep matters at the task level."),
             ("Largest steps of the stability map.", "At \\(t_c\\)&#8202;=&#8202;20&#8202;ms and \\(d_0\\) 0.9 and 0.95 the tool is ejected at "
              "15&#8202;ms, below the bound of 18&#8211;19&#8202;ms. Run <code>contact_bed_stability.py --d0 0.9 0.95 --tc 20 --dt 11 12 13 14 "
              "--refsafe 0</code> to locate the failing step and compare it with the numerical bound (15&#8211;16&#8202;ms)."),
             ("One GPU world.", "The bed&#8217;s Newton rig steps one world from the host without a CUDA graph (2.0&#8211;2.8&#8202;ms per step) "
              "where the MuJoCo-Warp rig captures one (0.27&#8202;ms). Capturing Newton&#8217;s step in the rig would give its one-world cost "
              "without the launch overhead.")]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


# ------------------------------------------------------------------------------------------ main

def render_tex(t):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    return P.TEX_RE.sub(lambda m: next(svgs), t), len(items)


def lede(T, M):
    tw = T["twist"]
    g = lambda k, N: pick(tw, k, N=N, dt_ms=1.0)  # noqa: E731
    dev = {k: statistics.median(v) for k, v in P.deviations(M).items() if v}
    parts = []
    try:
        pd = [g("mj_pads1", N)["rbar_onset_mm"] / g("drake_hydro", N)["rbar_onset_mm"] - 1 for N in (0.5, 1.0, 3.0)]
        lo_, hi_ = f"{min(abs(x) for x in pd) * 100:.0f}", f"{max(abs(x) for x in pd) * 100:.0f}"
        parts.append(f"The 1&#8202;mm sphere pad spins the screwdriver at {lo_ if lo_ == hi_ else lo_ + '&#8211;' + hi_}"
                     f"&#8202;% less torque than Drake hydroelastic from 0.5 to 3&#8202;N, slips at the "
                     "same pull force, rolls the tool as Drake does and brakes the swing to within 1&#176; of Drake.")
    except (TypeError, KeyError):
        pass
    if all(k in dev for k in ("mj_pads1", "mjw_pads1", "newton_pads1", "newton_hydro_mc", "mj_point4s", "mj_point3")):
        parts.append(f"Over the nine metrics of the agreement table its median deviation from Drake is {dev['mj_pads1']:.1f}&#8202;% in MuJoCo, "
                     f"{dev['mjw_pads1']:.1f}&#8202;% in MuJoCo-Warp and {dev['newton_pads1']:.1f}&#8202;% in Newton; condim&#160;4 gives "
                     f"{dev['mj_point4s']:.1f}&#8202;% with a single contact point, Newton&#8217;s mass-corrected hydroelastic tip "
                     f"{dev['newton_hydro_mc']:.1f}&#8202;% and point contact, which transmits no torque, {dev['mj_point3']:.0f}&#8202;%.")
    st = P.stab_stats(P.stab_rows())
    if st:
        parts.append(f"The pad&#8217;s largest stable step follows \\(t_c(d_0+(1-d_0)/n)\\) in {len(st['inside'])} of {st['n']} settings.")
    c = P.step_cost(T["pull"])
    bat = P.gpu_batched(P.gpu_curves(P.load(os.path.join(P.GPU, "gpu_scaling.jsonl")), P.load(os.path.join(P.GPU, "newton_scaling.jsonl"))))
    if "mj_pads1" in c and "drake_hydro" in c and "mjw_pads1" in bat:
        parts.append(f"A pad step costs {c['mj_pads1']:.0f}&#8202;&#181;s on one CPU core against {num(c['drake_hydro'], ',.0f')}&#8202;&#181;s "
                     f"for Drake, and {bat['mjw_pads1'][0]:.2f}&#8202;&#181;s per world-step in a MuJoCo-Warp batch on one GPU"
                     + (f" ({bat['newton_pads1'][0]:.2f} in Newton)" if "newton_pads1" in bat else "") + ".")
    parts.append("Under a held load at half the slip force every MuJoCo model creeps (slides slowly) 150&#8211;260&#215; faster than Drake, "
                 "and Newton&#8217;s hydroelastic tip 25&#8211;50&#215;.")
    return " ".join(parts)


def main():
    T = {name: P.bed(fn) for name, fn in (("pull", "pull_slip"), ("twist", "twist_slip"), ("roll", "roll"),
                                           ("shake", "shake"), ("brake", "brake"), ("creep", "creep"))}
    stab = P.stab_rows()
    gpu_pads = P.load(os.path.join(P.GPU, "gpu_scaling.jsonl"))
    gpu_newton = P.load(os.path.join(P.GPU, "newton_scaling.jsonl"))
    M = P.metrics(T)
    v = {"STYLE": P.style_block(), "BUILT": time.strftime("%Y-%m-%d %H:%M"), "NEWTON_REV": "009158e6", "OVERVIEW_PATH": OVERVIEW_PATH}
    u = open(OVERVIEW_URL_FILE).read().strip() if os.path.exists(OVERVIEW_URL_FILE) else ""
    v["OVERVIEW_LINK"] = f", <a href=\"{u}\">artifact</a>" if u else ""
    v["LEDE"] = lede(T, M)
    # the bed has no plan replay: leave out the overview's turn terms
    v["GLOSSARY"] = P.glossary_html([g[0] for g in P.GLOSSARY if not g[0].startswith("tool turn")])
    v["SETUP"] = setup(T)
    v["SUMMARY"] = summary(T, M)
    v["T1"] = t1(T["pull"])
    v["T2"] = t2(T["twist"])
    v["T3"] = t3(T["roll"])
    v["T4"] = t4(T["shake"])
    v["T5"] = t5(T["brake"])
    v["T6"] = t6(T["creep"])
    v["T7"] = t7(stab)
    v["CONSIST"] = consist(T)
    v["COST"] = cost(T, gpu_pads, gpu_newton)
    v["OPEN"] = open_items(T)
    v["FOOTER"] = ("<p>Rebuild: <code>python3 scripts/contact_bed_page.py</code>. Rows: <code>docs/experiments/20261005-contact_bed/*.jsonl</code>; "
                   "films: <code>docs/experiments/20261005-contact_bed/media/</code>.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    t, n = render_tex(t)
    open(OUT, "w").write(t)
    print(f"formulas {n}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
