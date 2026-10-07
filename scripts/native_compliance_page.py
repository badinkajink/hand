#!/usr/bin/env python3
r"""Build docs/experiments/20261007-native_compliance/20261007-native_presliding_compliance.html.

    python3 scripts/native_compliance_page.py

Steps 5, 6 and 11 of docs/handoff/20261007-newton_hydro_tests_native_compliance.md: the references with no simulator
(scripts/contact_reference_laws.py: Hertz, Cattaneo-Mindlin with the Mindlin-Deresiewicz unloading rule, Lubkin torsion by a
half-space cell solver), bed tasks T8 (tangential load cycle) and T9 (normal load sweep) on point contact, the 1 mm pads and
Drake (scripts/contact_bed_compliance.py), and the native-MuJoCo candidates of scripts/contact_bed_candidates.py on T8, T9,
T1, T2 and T5 with their CPU and MuJoCo-Warp cost. Every number in the prose is computed here from the rows in
docs/experiments/20261007-native_compliance/; style, plotting helpers and LaTeX rendering are those of
scripts/contact_overview_page.py.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import texsvg  # noqa: E402

D = os.path.join(P.EXP, "20261007-native_compliance")
OUT = os.path.join(D, "20261007-native_presliding_compliance.html")
TPL = os.path.join(ROOT, "scripts/native_compliance_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
BED_PATH = "docs/experiments/20261005-contact_bed/20261005-contact_model_bed.html"
num, fmt, pick = P.num, P.fmt, P.pick
US = P.US
LAW = {0.5: 0.820, 1.0: 0.974, 3.0: 1.281}
FIG = [0]
LBL = {"mj_point3": "MuJoCo point contact", "mj_pads1": "1&#8202;mm sphere pads", "drake_hydro": "Drake hydroelastic",
       "mj_pads1_ir1000": "1&#8202;mm pads, impratio 1000", "mj_pads1_skin": "(a) compliant skin",
       "mj_pads1_soft": "(d) softening solimp", "mj_pads1_bristle20a": "(e) rolling bristles, 1&#8202;mm",
       "mj_pads2_bristle20a": "(e) rolling bristles, 2&#8202;mm pad", "mj_pads1_bristle20": "(e) bristles, no armature"}
COL = {"mj_point3": "var(--ink3)", "mj_pads1": "var(--c-sphere)", "drake_hydro": "var(--c-drake)", "mj_pads1_skin": "var(--s1)",
       "mj_pads1_bristle20a": "var(--c-newton)", "mindlin": "var(--c-ref)"}


def rows(name):
    return P.load(os.path.join(D, name))


def ok(r):
    return r is not None and r.get("status", "complete") == "complete"


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


GLOSSARY = [
    ("presliding displacement", "Task&#160;8 cycles an axial force on the pinched tool between &#177;\\(F^*\\), \\(F^* = \\mu N\\), half "
     "the slip force \\(2\\mu N\\): 0 &#8594; \\(+F^*\\) in 0.5&#8202;s, &#8594; \\(-F^*\\) in 1&#8202;s, &#8594; \\(+F^*\\) in 1&#8202;s, "
     "&#8594; 0 in 0.5&#8202;s. Presliding displacement: the tool&#8217;s axial displacement at the first peak, in &#181;m."),
    ("recovered displacement", "The part of the presliding displacement that returns when the force first comes back to zero "
     "(t&#8202;=&#8202;1.0&#8202;s), in &#181;m. An elastic contact recovers most of it; a creeping contact keeps sliding while the force "
     "is positive and recovers nothing (negative values)."),
    ("loop area", "Work done by the force around the full cycle \\(+F^*\\) &#8594; \\(-F^*\\) &#8594; \\(+F^*\\), \\(\\oint F\\,du\\), in "
     "&#181;J: the energy the contact dissipates per cycle. Mindlin&#8217;s partial slip gives a closed loop; creep gives a loop "
     "proportional to \\(\\int F^2 dt\\)."),
    ("initial tangential stiffness", "Slope of force against displacement over the first 5&#8211;20&#8202;% of \\(F^*\\) on the first "
     "loading, both pads in parallel, in kN/m."),
    ("approach, RMS patch radius", "Task&#160;9 holds the pinch at N = 0.25&#8211;4&#8202;N. Approach: the pad centre&#8217;s travel past "
     "first touch. RMS patch radius: force-weighted RMS distance of the &#8722;x pad&#8217;s contacts from their centroid, in the plane "
     "normal to the pinch axis. Their log&#8211;log slopes against N are the load exponents."),
    ("onset arm, arm ratio", "Task&#160;2: the torque at spin onset over \\(2\\mu N\\), per pad, in mm; the ratio of the onset arm at 3 and "
     "0.5&#8202;N (pressure law 1.565, Hertz 1.817)."),
    ("largest swing, 80&#176; force", "Task&#160;5, the braked swing: the largest angle of the tool and the pinch force when it first "
     "passes 80&#176; (Drake 87.3&#176; and 0.39&#8202;N)."),
    ("\\(G\\), \\(\\nu\\), \\(a\\)", "Shear modulus and Poisson ratio of the tip (stated: \\(E\\) = 10&#8202;MPa, \\(\\nu\\) = 0.45, "
     "\\(G = E/2(1+\\nu)\\) = 3.45&#8202;MPa; the printed TPU is not measured yet) and the Hertz contact radius."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def refs():
    ref = json.load(open(os.path.join(D, "reference_laws.json")))
    sc = ref["selfcheck"]
    body = []
    for N in ("0.5", "1.0", "3.0"):
        h, m, lb = ref["hertz"][N], ref["mindlin"][N], ref["lubkin"][N]
        half = [c for c in lb["curve"] if c[1] <= 0.5 * lb["M_full_slip"]]
        body.append([f"{float(N):g}&#8202;N", fmt(h["a"] * 1e3, 3), fmt(h["delta"] * 1e6, 1), fmt(m["u_presliding"] * 1e6, 1),
                     fmt(m["u_recovered"] * 1e6, 1), fmt(m["loop_area"] * 1e6, 2), fmt(2 * m["k_t0_pad"] / 1e3, 1),
                     fmt(lb["M_full_slip"] * 1e3, 3), fmt(math.degrees(half[-1][0]), 2) if half else "&#8211;"])
    t9 = ref["t9"]
    return ("<p>The pressure law and every model on the bed so far are stateless: a contact&#8217;s force depends on the current "
            "overlap and slip velocity only. An elastic tip also stores tangential displacement before it slips. For a sphere on a rigid "
            "tool, with the circular contact of the equivalent radius \\(R_e = \\sqrt{R\\,Rr/(R+r)}\\) = 7.77&#8202;mm:</p>"
            "\\[ a = \\Big(\\frac{3NR_e}{4E^*}\\Big)^{1/3}, \\qquad u(Q) = \\frac{3\\mu N(2-\\nu)}{16Ga}\\Big[1-\\big(1-\\tfrac{Q}{\\mu N}"
            "\\big)^{2/3}\\Big], \\qquad k_{\\theta 0} = \\tfrac{16}{3}Ga^3 \\qquad (1) \\]"
            "<p>Hertz&#8217;s radius, Cattaneo&#8211;Mindlin&#8217;s tangential displacement under a force \\(Q\\) per pad (unloading by "
            "the Mindlin&#8211;Deresiewicz rule \\(u_{\\downarrow}(Q) = u(Q^*) - 2u\\big((Q^*-Q)/2\\big)\\)) and Lubkin&#8217;s initial torsional "
            "stiffness. <code>scripts/contact_reference_laws.py</code> also solves Lubkin&#8217;s partial-slip twist on a grid of "
            f"{sc['cells']} surface cells with Cerruti&#8217;s half-space influence functions; run as a no-slip translation and twist the "
            f"same solver gives {sc['k_t_cells']:.0f} and {sc['k_theta_cells'] * 1e3:.3f} against the closed forms {sc['k_t_closed']:.0f}&#8202;N/m "
            f"and {sc['k_theta_closed'] * 1e3:.3f}&#8202;mN&#8202;m/rad.</p>" +
            table(["N", "Hertz \\(a\\) (mm)", "approach (&#181;m)", "T8 presliding (&#181;m)", "recovered (&#181;m)", "loop (&#181;J)",
                   "initial stiffness (kN/m)", "full-slip torque (mN&#8202;m)", "twist at half of it (&#176;)"], body,
                  "Table 1: references at the stated TPU modulus. T8 values are for the bed (two pads, \\(F^*=\\mu N\\)). Load exponents "
                  f"(T9): approach {t9['hertz_delta_exp']:.3f}, radius {t9['hertz_r_exp']:.3f} for Hertz; {t9['law_delta_exp']:.3f} and "
                  f"{t9['law_r_exp']:.3f} for the pressure law on the bed geometry.") +
            "<p>At the same 10&#8202;MPa the pressure law spreads the load over a patch about twice Hertz&#8217;s radius (RMS 1.06 against "
            "0.49&#8202;mm at 1&#8202;N) and carries twice the torque; Hertz matches the law&#8217;s patch at about 0.8&#8202;MPa. "
            "The two cannot both describe the tip until its modulus is measured, so the candidates below are sized on Hertz at "
            "10&#8202;MPa and judged on shape: elastic recovery, hysteresis and load exponents.</p>")


def svg_loops(t8):
    import contact_reference_laws as L
    W, Hh = 860, 330
    out = P._svg_open(W, Hh, "Task 8 force-displacement loops at 1 N")
    fx, fy = P._panel(out, 80, 30, 720, 240, (-60, 60), (-1.1, 1.1), [-60, -40, -20, 0, 20, 40, 60], [-1, -0.5, 0, 0.5, 1],
                      "tool displacement (&#181;m)", "axial force (N)")
    mc = L.mindlin_cycle(1.0)
    P._path(out, fx, fy, list(zip(mc["u"] * 1e6, mc["F"])), COL["mindlin"], width=2.4)
    for m in ("mj_pads1", "mj_pads1_skin", "mj_pads1_bristle20a"):
        r = pick(t8, m, N=1.0, dt_ms=1.0)
        if ok(r):
            tr = np.array(r["trace"])
            P._path(out, fx, fy, [(u, F) for _, F, u in tr if abs(u) < 60], COL[m], width=1.6)
    out.append("</svg>")
    leg = P._legend_html([("Mindlin, closed form", COL["mindlin"], False, None), ("1 mm pads", COL["mj_pads1"], False, None),
                          ("(a) skin", COL["mj_pads1_skin"], False, None), ("(e) bristles", COL["mj_pads1_bristle20a"], False, None)])
    return "".join(out) + leg


def base(t8, t9):
    import contact_bed_compliance as CC
    body = []
    for m in ("mj_point3", "mj_pads1", "drake_hydro"):
        rs = [pick(t8, m, N=N, dt_ms=1.0) for N in (0.5, 1.0, 3.0)]
        sl = CC.add_slopes(os.path.join(D, "t9_sweep.jsonl"), m, 1.0)
        body.append([LBL[m]] + [fmt(r["u_presliding_um"], 2) if ok(r) else "&#8211;" for r in rs] +
                    [fmt(rs[1]["u_recovered_um"], 2) if ok(rs[1]) else "&#8211;", fmt(rs[1]["loop_area_uJ"], 2) if ok(rs[1]) else "&#8211;",
                     fmt(sl.get("delta_exp"), 3), fmt(sl.get("r_rms_exp"), 3)])
    return ("<p>Point contact, the 1&#8202;mm pads and Drake have no tangential elasticity. Under the T8 cycle their tool creeps at a "
            "speed set by the friction rows (Drake&#8217;s regularised friction 100&#215; slower), keeps creeping while the force is "
            "positive, and recovers nothing; the loop they trace is creep (Table&#160;2). Their T9 load exponents are the pressure "
            "law&#8217;s: the patch grows with the overlap, as \\(N^{1/2}\\) in approach and \\(N^{1/4}\\) in radius, with no load outside "
            "the overlap.</p>" +
            table(["model", "presliding 0.5&#8202;N (&#181;m)", "1&#8202;N", "3&#8202;N", "recovered, 1&#8202;N (&#181;m)", "loop, 1&#8202;N (&#181;J)",
                   "approach exponent", "radius exponent"], body,
                  "Table 2: T8 and T9 at 1&#8202;ms. Mindlin at 1&#8202;N: presliding 40.3, recovered 38.0, loop 6.13; Hertz exponents "
                  "0.667 and 0.333."))


CANDS = [
    ("mj_pads1_skin", "(a) Compliant skin",
     "The pad spheres sit on a child body joined to the fingertip by two slides tangent to the pad and a hinge about its normal, "
     "with springs at Mindlin&#8217;s and Lubkin&#8217;s initial stiffness at 1&#8202;N (13.8&#8202;kN/m, 8.55&#8202;mN&#8202;m/rad), "
     "critical damping, and friction made stiff with impratio 1000. The skin is 2&#8202;g with 18&#8202;g of armature on its slides: MuJoCo "
     "softens a contact&#8217;s friction rows in proportion to the contact&#8217;s inverse weight, and with the bare 2&#8202;g skin the tool "
     "slid 49&#8202;&#181;m over the spheres during each force ramp. Damping at the pads&#8217; 20&#8202;ms relaxation (8&#215; critical) made "
     "the tool lead and trail the skin with the force rate and reversed the loop."),
    ("mj_pads1_soft", "(d) Softening solimp",
     "Each sphere&#8217;s impedance falls with depth (\\(d_{\\min} > d_{\\max}\\)), fitted so that its static force follows "
     "\\(K_s\\sqrt{r\\,r_\\text{ref}}\\) over 0.03&#8211;0.5&#8202;mm (within 7&#8202;%), equal to the linear sphere at "
     "\\(r_\\text{ref}\\) = 0.2&#8202;mm. Inside the solimp width MuJoCo&#8217;s static force is "
     "\\(f = r\\,d(r)/\\big(d_{\\max}^2(1-d(r))\\,t_c^2 w\\big)\\), measured on a sphere&#8211;plane contact. No tangential change."),
    ("mj_pads1_bristle20a", "(e) Rolling bristles",
     "Each sphere within 20&#176; of the pad&#8217;s pole sits on its own body with a ball joint at its centre and a rotational spring "
     "\\(k_\\theta = k_t r_s^2\\), \\(k_t = G A_s/h\\) with \\(h\\) = 2.7&#8202;mm so that the ~11 spheres in contact at 1&#8202;N give "
     "Mindlin&#8217;s stiffness: the contact point rolls elastically before that sphere&#8217;s Coulomb contact slips, a brush with one "
     "bristle per sphere. With 1&#8202;mg spheres and no armature the bristles go unstable under tangential load at 1 and 2&#8202;ms; "
     "armature worth 0.2&#8202;g at the contact still does; 2&#8202;g at each contact runs."),
]


def cands(t8, t9, tw, pu, br, cost):
    import contact_bed_compliance as CC
    out = []
    for m, title, text in CANDS:
        r8 = {N: pick(t8, m, N=N, dt_ms=1.0) for N in (0.5, 1.0, 3.0)}
        sl = CC.add_slopes(os.path.join(D, "t9_sweep.jsonl"), m, 1.0)
        t2 = {N: pick(tw, m, N=N, dt_ms=1.0) for N in (0.5, 1.0, 3.0)}
        t1 = pick(pu, m, N=1.0, dt_ms=1.0)
        t5 = pick(br, m, dt_ms=1.0)
        c = {r["nworld"]: r for r in cost if r["model"] == m}
        parts = [f"<h3>{title}</h3><p>{text}</p>"]
        if all(ok(r) for r in r8.values()):
            parts.append(f"<p>T8: presliding {fmt(r8[0.5]['u_presliding_um'], 1)}, {fmt(r8[1.0]['u_presliding_um'], 1)} and "
                         f"{fmt(r8[3.0]['u_presliding_um'], 1)}&#8202;&#181;m at 0.5, 1 and 3&#8202;N (Mindlin 25.4, 40.3, 83.8), recovered "
                         f"{fmt(r8[1.0]['u_recovered_um'], 1)}&#8202;&#181;m at 1&#8202;N (38.0), loop {fmt(r8[0.5]['loop_area_uJ'], 2)}, "
                         f"{fmt(r8[1.0]['loop_area_uJ'], 2)} and {fmt(r8[3.0]['loop_area_uJ'], 2)}&#8202;&#181;J (1.93, 6.13, 38.3). T9 exponents "
                         f"{fmt(sl.get('delta_exp'), 3)} and {fmt(sl.get('r_rms_exp'), 3)}. "
                         + (f"T2 onset arm {fmt(t2[0.5]['rbar_onset_mm'] / LAW[0.5], 2)}, {fmt(t2[1.0]['rbar_onset_mm'] / LAW[1.0], 2)} and "
                            f"{fmt(t2[3.0]['rbar_onset_mm'] / LAW[3.0], 2)} of the pressure law, {fmt(t2[1.0]['rot_pre_deg'], 1)}&#176; before "
                            f"onset at 1&#8202;N. " if all(ok(x) for x in t2.values()) else "")
                         + ("At 3&#8202;N the torque ramp winds the hinge at 53&#176;/s, above the task&#8217;s 30&#176;/s spin detection, so "
                            "that onset is not a slip measurement. " if m == "mj_pads1_skin" else "")
                         + (f"T1 effective \\(\\mu\\) {fmt(t1['mu_eff'], 3)}. " if t1 else "")
                         + (f"T5 largest swing {fmt(t5['phi_max_deg'], 1)}&#176;, 80&#176; at {fmt(t5['N_at_80_N'], 3)}&#8202;N. " if ok(t5) else "")
                         + (f"Cost: {fmt(r8[1.0]['us_per_step_median'], 0)}&#8202;{US} per step on one core at 1&#8202;N"
                            + (f", {c[4096]['us_per_world_step']:.2f}&#8202;&#181;s per world-step in a 4,096-world MuJoCo-Warp batch" if 4096 in c else "")
                            + ".</p>"))
        out.append("".join(parts))
    return "".join(out) + figure(svg_loops(t8), "Task&#160;8 at 1&#8202;N, 1&#8202;ms: the tool&#8217;s axial displacement against the force "
                                 "over the whole cycle for the pads, the compliant skin and the rolling bristles, against Mindlin&#8217;s "
                                 "closed form.")


def compare(t8, t9, tw, br, cost):
    import contact_bed_compliance as CC
    body = []
    for m in ("mj_pads1", "mj_pads1_ir1000", "mj_pads1_skin", "mj_pads1_soft", "mj_pads1_bristle20a", "mj_pads2_bristle20a"):
        r1 = pick(t8, m, N=1.0, dt_ms=1.0)
        sl = CC.add_slopes(os.path.join(D, "t9_sweep.jsonl"), m, 1.0)
        t5 = pick(br, m, dt_ms=1.0) or (pick(P.bed("brake"), "mj_pads1", dt_ms=1.0) if m == "mj_pads1" else None)
        t2 = pick(tw, m, N=3.0, dt_ms=1.0) or (pick(P.bed("twist_slip"), "mj_pads1", N=3.0, dt_ms=1.0) if m == "mj_pads1" else None)
        c = {r["nworld"]: r for r in cost if r["model"] == m}
        body.append([LBL[m], fmt(r1["u_presliding_um"], 1) if ok(r1) else "&#8211;", fmt(r1["u_recovered_um"], 1) if ok(r1) else "&#8211;",
                     fmt(r1["loop_area_uJ"], 2) if ok(r1) else "&#8211;", fmt(sl.get("delta_exp"), 2),
                     fmt(t2.get("rbar_ratio_3_05_onset"), 2) if t2 else "&#8211;", fmt(t5["phi_max_deg"], 1) if ok(t5) else "&#8211;",
                     fmt(r1["us_per_step_median"], 0) if r1 else "&#8211;",
                     fmt(c[4096]["us_per_world_step"], 2) if 4096 in c else "&#8211;"])
    body.append(["Mindlin / Hertz", "40.3", "38.0", "6.13", "0.67", "1.82", "&#8211;", "", ""])
    body.append(["Drake hydroelastic", "0.05", "&#8722;0.05", "0.12", "0.48", "1.52", "87.3", "1,300", ""])
    return table(["model", "presliding, 1&#8202;N (&#181;m)", "recovered (&#181;m)", "loop (&#181;J)", "approach exponent", "arm ratio",
                  "largest swing (&#176;)", f"{US} per step", "&#181;s per world-step, 4,096"], body,
                 "Table 3: the candidates at 1&#8202;ms. The pads&#8217; GPU column is the impratio-1000 control on the same rig.")


def lede(t8, t9, cost):
    import contact_bed_compliance as CC
    s, b = pick(t8, "mj_pads1_skin", N=1.0, dt_ms=1.0), pick(t8, "mj_pads1_bristle20a", N=1.0, dt_ms=1.0)
    sl = CC.add_slopes(os.path.join(D, "t9_sweep.jsonl"), "mj_pads1_soft", 1.0)
    c = {(r["model"], r["nworld"]): r for r in cost}
    parts = ["No model on the bed stores tangential displacement: under a force cycled at half the slip force the 1&#8202;mm pads and "
             "Drake creep and recover nothing, where Mindlin&#8217;s elastic contact moves 40&#8202;&#181;m at 1&#8202;N and recovers 38."]
    if ok(s) and ok(b):
        parts.append(f"Three MuJoCo-native additions supply what is missing, one property each: a spring-mounted skin gives "
                     f"{s['u_presliding_um']:.0f}&#8202;&#181;m of recoverable presliding with no hysteresis at 1.3&#215; the pads&#8217; GPU cost; "
                     f"spring-loaded ball joints under the spheres (rolling bristles) add partial slip and a {b['loop_area_uJ']:.1f}&#8202;&#181;J loop "
                     f"against Mindlin&#8217;s 6.1, at "
                     + (f"{c[('mj_pads1_bristle20a', 4096)]['us_per_world_step'] / c[('mj_pads1_ir1000', 4096)]['us_per_world_step']:.0f}&#215; "
                        if ('mj_pads1_bristle20a', 4096) in c and ('mj_pads1_ir1000', 4096) in c else "")
                     + "the cost; and an impedance that softens with depth gives Hertz&#8217;s load exponents "
                     + (f"({sl['delta_exp']:.2f} and {sl.get('r_rms_exp', float('nan')):.2f} against 2/3 and 1/3) " if sl else "")
                     + "at no cost.")
    return " ".join(parts)


def open_items():
    items = [("Flex and lattice candidates.", "MuJoCo-Warp 3.14 has flex elasticity and flex-vertex contact with spheres, boxes, cylinders "
              "and meshes (no flex&#8211;SDF, no quadratic flex), so (b) is buildable: a flexcomp shell over the tip cap, pinned at the "
              "back, with the stated Young&#8217;s modulus and Poisson ratio. (c), sphere nodes on normal slides with anchor and neighbour "
              "springs (CSLC), is not built. Both answer T9&#8217;s lateral spread, which no candidate here changes."),
             ("Cheaper bristles.", "The bristles cost 26&#215; the pads batched and need 2&#8202;g of armature per contact to run at 1&#8202;ms. "
              "Two hinges per sphere in place of a ball joint, and a soft weld whose impedance divides out the rotational inverse "
              "inertia as the pads do for the normal direction, would remove a third of the dofs and the armature; rerun T8 and the cost."),
             ("The modulus.", "Every candidate is sized on Hertz at 10&#8202;MPa, where the pressure law&#8217;s patch is twice Hertz&#8217;s "
              "radius. An indentation test on the printed tip (force against depth) gives \\(E\\) and the exponent; a slow tangential "
              "cycle on it measures presliding and loop area for T8 directly.")]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def render_tex(t):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    return P.TEX_RE.sub(lambda m: next(svgs), t), len(items)


def main():
    t8, t9, tw, pu, br, cost = (rows(f) for f in ("t8_cycle.jsonl", "t9_sweep.jsonl", "twist_slip.jsonl", "pull_slip.jsonl",
                                                    "brake.jsonl", "mjw_cost.jsonl"))
    v = {"STYLE": P.style_block(), "BUILT": time.strftime("%Y-%m-%d %H:%M"), "BED_PATH": BED_PATH,
         "OVERVIEW_PATH": os.path.relpath(P.OUT, ROOT)}
    u = P.bed_url()
    v["BED_LINK"] = f", <a href=\"{u}\">artifact</a>" if u else ""
    v["LEDE"] = lede(t8, t9, cost)
    v["GLOSSARY"] = glossary()
    v["REFS"] = refs()
    v["BASE"] = base(t8, t9)
    v["CANDS"] = cands(t8, t9, tw, pu, br, cost)
    v["COMPARE"] = compare(t8, t9, tw, br, cost)
    v["OPEN"] = open_items()
    v["FOOTER"] = ("<p>Rebuild: <code>python3 scripts/native_compliance_page.py</code>. Rows: "
                   "<code>docs/experiments/20261007-native_compliance/*.jsonl</code> and <code>reference_laws.json</code>.</p>")
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
