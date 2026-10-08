#!/usr/bin/env python3
r"""Build docs/experiments/20261007-native_compliance/20261007-native_presliding_compliance.html.

    logs/20261001-hom_contact/venv/bin/python scripts/native_compliance_page.py

Steps 5, 6 and 11 of docs/handoff/20261007-newton_hydro_tests_native_compliance.md: the references with no simulator
(scripts/contact_reference_laws.py: Hertz, Cattaneo-Mindlin with the Mindlin-Deresiewicz unloading rule, Lubkin torsion by a
half-space cell solver), bed tasks T8 (tangential load cycle) and T9 (normal load sweep) on point contact, the 1 mm pads and
Drake (scripts/contact_bed_compliance.py), and the native-MuJoCo candidates of scripts/contact_bed_candidates.py (skin,
rolling bristles, softening impedance, lattice, flex) on T8, T9, T1, T2 and T5 with their CPU and MuJoCo-Warp cost. Every
number in the prose is computed here from the rows in docs/experiments/20261007-native_compliance/ and the bed's rows; style,
plotting helpers and LaTeX rendering are those of scripts/contact_overview_page.py.
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
POLICY_NOTE = "docs/handoff/20261007-contact_model_policy_training.md"
num, fmt, pick = P.num, P.fmt, P.pick
US = P.US
LAW = {0.5: 0.820, 1.0: 0.974, 3.0: 1.281}
FIG = [0]
LBL = {"mj_point3": "MuJoCo point contact", "mj_pads1": "1&#8202;mm sphere pads", "drake_hydro": "Drake hydroelastic",
       "mj_pads1_ir1000": "1&#8202;mm pads, impratio 1000", "mj_pads1_skin": "compliant skin",
       "mj_pads1_soft": "softening impedance", "mj_pads1_bristle20a": "rolling bristles",
       "mj_pads2_bristle20a": "rolling bristles, 2&#8202;mm pad", "mj_pads1_lattice1": "lattice, 1&#8202;mm coupling",
       "mj_pads1_flex20": "flex"}
COL = {"mj_point3": "var(--ink3)", "mj_pads1": "var(--c-sphere)", "drake_hydro": "var(--c-drake)", "mj_pads1_skin": "var(--c-c4)",
       "mj_pads1_bristle20a": "var(--c-newton)", "mj_pads1_soft": "var(--bad)", "mj_pads1_lattice1": "var(--s2)",
       "mj_pads1_flex20": "var(--s3)", "mindlin": "var(--c-ref)"}
CANDS = ("mj_pads1_skin", "mj_pads1_bristle20a", "mj_pads1_soft", "mj_pads1_lattice1", "mj_pads1_flex20")


def rows(name):
    return P.load(os.path.join(D, name))


def ok(r):
    return r is not None and r.get("status", "complete") == "complete"


def figure(svg, caption):
    FIG[0] += 1
    return f'<figure class="diagram">{svg}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def _keep_case(h):
    """Table headers are set in capitals by the house style; keep units (parenthesised, or words starting with the micro
    sign) in their own case, so that µm and mN m are not set as MM and MN M."""
    import re
    h = re.sub(r"(?<!\\)\(([^()\\]*)\)", lambda m: '(<span style="text-transform:none">' + m.group(1) + "</span>)", h)
    return re.sub(r"(?<![>(])(&#181;[A-Za-z]+)", r'<span style="text-transform:none">\1</span>', h)


def table(head, body, note=None):
    out = ["<div class='tw'><table><thead><tr>" + "".join(f"<th{' class=num' if i else ''}>{_keep_case(h)}</th>"
                                                          for i, h in enumerate(head))
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


def slopes(m):
    import contact_bed_compliance as CC
    return CC.add_slopes(os.path.join(D, "t9_sweep.jsonl"), m, 1.0)


class Data:
    """Rows of this study and of the bed (the pads' and Drake's T1, T2 and T5)."""

    def __init__(self):
        self.t8, self.t9, self.tw, self.pu, self.br, self.cost = (rows(f) for f in (
            "t8_cycle.jsonl", "t9_sweep.jsonl", "twist_slip.jsonl", "pull_slip.jsonl", "brake.jsonl", "mjw_cost.jsonl"))
        self.btw, self.bpu, self.bbr = P.bed("twist_slip"), P.bed("pull_slip"), P.bed("brake")

    def t8r(self, m, N=1.0, dt=1.0):
        return pick(self.t8, m, N=N, dt_ms=dt)

    def t2(self, m, N):
        return pick(self.tw, m, N=N, dt_ms=1.0) or pick(self.btw, m, N=N, dt_ms=1.0)

    def t1(self, m, N=1.0):
        return pick(self.pu, m, N=N, dt_ms=1.0) or pick(self.bpu, m, N=N, dt_ms=1.0)

    def t5(self, m):
        return pick(self.br, m, dt_ms=1.0) or pick(self.bbr, m, dt_ms=1.0)

    def gpu(self, m, nw=4096):
        c = [r for r in self.cost if r["model"] == m and r["nworld"] == nw]
        return c[-1]["us_per_world_step"] if c else None

    def cpu(self, m):
        """Median wall time of one 1 ms step on one core at 1 N: T8 if run, else T2's ramp, else T9's settle."""
        for r in (self.t8r(m), self.t2(m, 1.0), pick(self.t9, m, N=1.0, dt_ms=1.0)):
            if r is not None and r.get("us_per_step_median"):
                return r["us_per_step_median"]
        return None


# ------------------------------------------------------------------------------------------ terms

_OV = dict(P.GLOSSARY)
GLOSSARY = [
    ("presliding displacement", "Task&#160;8 holds the pinch at \\(N\\) per pad with gravity off and cycles an axial force on the tool "
     "between &#177;\\(F^*\\), \\(F^* = \\mu N\\), half the slip force \\(2\\mu N\\): 0 &#8594; \\(+F^*\\) in 0.5&#8202;s, &#8594; \\(-F^*\\) in "
     "1&#8202;s, &#8594; \\(+F^*\\) in 1&#8202;s, &#8594; 0 in 0.5&#8202;s. Presliding displacement: the tool&#8217;s axial displacement at the "
     "first peak, in &#181;m."),
    ("recovered displacement", "The part of the presliding displacement that returns when the force first comes back to zero "
     "(t&#8202;=&#8202;1.0&#8202;s), in &#181;m. An elastic contact recovers most of it; a creeping contact keeps sliding while the force "
     "is positive and recovers nothing (negative values)."),
    ("loop area", "Work done by the force around the full cycle \\(+F^*\\) &#8594; \\(-F^*\\) &#8594; \\(+F^*\\), \\(\\oint F\\,du\\), in "
     "&#181;J: the energy the contact dissipates per cycle. Mindlin&#8217;s partial slip gives a closed loop; creep gives a loop "
     "proportional to \\(\\int F^2 dt\\); a linear elastic spring gives none."),
    ("approach, RMS patch radius, load exponents", "Task&#160;9 holds the pinch at \\(N\\) = 0.25&#8211;4&#8202;N per pad for 1&#8202;s with "
     "gravity off. Approach: the pad centre&#8217;s travel past first touch, in mm. RMS patch radius: force-weighted RMS distance of "
     "the &#8722;x pad&#8217;s contacts from their centroid, in the plane normal to the pinch axis, in mm. Load exponents: the slopes of "
     "log approach and log radius against log \\(N\\); the pressure law gives 1/2 and 1/4, Hertz 2/3 and 1/3."),
    ("slip onset, effective \\(\\mu\\) (T1)", _OV["slip onset, effective \\(\\mu\\)"]),
    ("onset arm, arm ratio (T2)", "Task&#160;2 raises a torque about the pinch axis until the tool turns faster than 30&#8202;&#176;/s; the "
     "torque at that onset over \\(2\\mu N\\), in mm per pad, and its ratio to the pressure law&#8217;s arm (0.820, 0.974 and 1.281&#8202;mm "
     "at 0.5, 1 and 3&#8202;N). Arm ratio: the onset arm at 3&#8202;N over that at 0.5&#8202;N (pressure law 1.565, Hertz 1.817). "
     "<i>Twist before onset</i>: the tool&#8217;s rotation between the start of the ramp and onset, in degrees."),
    ("largest swing, 80&#176; force (T5)", "Task&#160;5 lowers the pinch force from 6 to 0.2&#8202;N over 4&#8202;s with the tool&#8217;s "
     "centre of mass 15&#8202;mm off the pinch line, so the tool swings about the pinch axis (90&#176; = hanging). Largest swing: the "
     "largest angle over the run; 80&#176; force: the pinch force when the tool first passes 80&#176; (Drake 87.3&#176; and 0.391&#8202;N)."),
    ("creep", _OV["creep"]),
    ("pressure law", "The stateless contact law the sphere pads discretize and Drake integrates: pressure \\(p = E\\,\\delta/R\\) at "
     "local depth \\(\\delta\\) inside the fingertip sphere of radius \\(R\\) = 10.55&#8202;mm, with \\(E\\) = 10&#8202;MPa. "
     "Integrated over the sphere&#8211;cylinder patch with no simulator (<code>scripts/hydroelastic_arm_integral.py</code>) it gives the "
     "approach and friction arm the tables compare against."),
    ("\\(G\\), \\(\\nu\\), \\(a\\)", "Shear modulus and Poisson ratio of the tip (stated: \\(E\\) = 10&#8202;MPa, \\(\\nu\\) = 0.45, "
     "\\(G = E/2(1+\\nu)\\) = 3.45&#8202;MPa; the printed TPU is not measured yet) and the Hertz contact radius."),
    ("\\(t_c\\), \\(d_0\\), \\(\\hat\\Lambda\\)", "MuJoCo contact parameters. \\(t_c\\), the <code>solref</code> time constant, sets how "
     "fast a penetration is corrected; \\(d_0\\), the <code>solimp</code> impedance, sets what fraction of that correction the solver "
     "enforces; \\(\\hat\\Lambda\\) is MuJoCo&#8217;s estimate of the contact&#8217;s inverse inertia. A contact&#8217;s static stiffness "
     "is \\(1/\\big(t_c^2(1-d_0)\\hat\\Lambda\\big)\\), and its friction rows are softened in the same proportion to "
     "\\(\\hat\\Lambda\\), so contacts on light bodies are soft unless \\(d_0\\) is recalibrated."),
    ("inverse weight \\(w\\)", _OV["inverse weight \\(w\\), effective mass \\(m_\\text{eff}\\)"]),
    ("impratio", "MuJoCo&#8217;s ratio of friction-row to normal-row stiffness; larger values make friction stick harder. The bed&#8217;s "
     "pads use 100."),
    ("armature", "Inertia added to a joint&#8217;s own degree of freedom (kg on a slide, kg&#8202;m&#178; on a hinge or ball joint), which "
     "MuJoCo adds to the diagonal of the mass matrix. It slows that joint&#8217;s motion without adding weight and lowers the inverse "
     "weight of the contacts on the joint&#8217;s body, which stiffens their friction rows."),
    ("cost", "<i>&#181;s per step</i>: median wall time of one 1&#8202;ms physics step on one CPU core at \\(N\\) = 1&#8202;N. <i>&#181;s per "
     "world-step</i>: wall time of one step of a 4,096-world MuJoCo-Warp batch divided by 4,096, in 10-step CUDA graphs, on one "
     "RTX 4070 Ti SUPER. Multiples are against the pads on the same rig (impratio 1000 for the GPU column)."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


# ------------------------------------------------------------------------------------------ references and baselines

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
            "0.49&#8202;mm at 1&#8202;N) and transmits twice the torque; Hertz matches the law&#8217;s patch at about 0.8&#8202;MPa. "
            "The candidates below are sized on Hertz at 10&#8202;MPa and judged on shape: elastic recovery, hysteresis and load exponents.</p>")


def base(X):
    body = []
    for m in ("mj_point3", "mj_pads1", "drake_hydro"):
        rs = [X.t8r(m, N) for N in (0.5, 1.0, 3.0)]
        sl = slopes(m)
        body.append([LBL[m]] + [fmt(r["u_presliding_um"], 2) if ok(r) else "&#8211;" for r in rs] +
                    [fmt(rs[1]["u_recovered_um"], 2) if ok(rs[1]) else "&#8211;", fmt(rs[1]["loop_area_uJ"], 2) if ok(rs[1]) else "&#8211;",
                     fmt(sl.get("delta_exp"), 3), fmt(sl.get("r_rms_exp"), 3)])
    return ("<p>Point contact, the 1&#8202;mm pads and Drake have no tangential elasticity. Under the T8 cycle their tool creeps at a "
            "speed set by the friction rows (Drake&#8217;s regularised friction 100&#215; slower), keeps creeping while the force is "
            "positive, and recovers nothing; the loop they trace is creep (Table&#160;2). Their T9 load exponents are the pressure "
            "law&#8217;s: the patch grows with the overlap, as \\(N^{1/2}\\) in approach and \\(N^{1/4}\\) in radius.</p>" +
            table(["model", "presliding 0.5&#8202;N (&#181;m)", "1&#8202;N", "3&#8202;N", "recovered, 1&#8202;N (&#181;m)", "loop, 1&#8202;N (&#181;J)",
                   "approach exponent", "radius exponent"], body,
                  "Table 2: T8 and T9 at 1&#8202;ms. Mindlin at 1&#8202;N: presliding 40.3, recovered 38.0, loop 6.13; Hertz exponents "
                  "0.667 and 0.333."))


# ------------------------------------------------------------------------------------------ figures

def svg_loops(X):
    import contact_reference_laws as L
    W, Hh = 860, 330
    out = P._svg_open(W, Hh, "Task 8 force-displacement loops at 1 N")
    fx, fy = P._panel(out, 80, 30, 720, 240, (-60, 60), (-1.1, 1.1), [-60, -40, -20, 0, 20, 40, 60], [-1, -0.5, 0, 0.5, 1],
                      "tool displacement (&#181;m)", "axial force (N)")
    mc = L.mindlin_cycle(1.0)
    P._path(out, fx, fy, list(zip(mc["u"] * 1e6, mc["F"])), COL["mindlin"], width=2.4)
    for m in ("mj_pads1", "mj_pads1_skin", "mj_pads1_bristle20a"):
        r = X.t8r(m)
        if ok(r):
            tr = np.array(r["trace"])
            P._path(out, fx, fy, [(u, F) for _, F, u in tr if abs(u) < 60], COL[m], width=1.6)
    out.append("</svg>")
    leg = P._legend_html([("Mindlin, closed form", COL["mindlin"], False, None), ("1 mm pads", COL["mj_pads1"], False, None),
                          ("compliant skin", COL["mj_pads1_skin"], False, None), ("rolling bristles", COL["mj_pads1_bristle20a"], False, None)])
    return "".join(out) + leg


def svg_t9(X):
    """Approach and RMS patch radius against N on log axes: pressure law and Hertz (no simulator), pads and three candidates."""
    ref = json.load(open(os.path.join(D, "reference_laws.json")))["t9"]["rows"]
    W, Hh = 980, 380
    out = P._svg_open(W, Hh, "Task 9: approach and RMS patch radius against pad force on log axes")
    Ns = (0.25, 0.5, 1, 2, 4)
    panels = [("delta_mm", "law_delta", "hertz_delta", (0.02, 0.8), (0.02, 0.05, 0.1, 0.2, 0.5), "(a) approach (mm), log scale"),
              ("r_rms_mm", "law_r_rms", "hertz_r_rms", (0.25, 2.5), (0.3, 0.5, 1.0, 2.0), "(b) RMS patch radius (mm), log scale")]
    models = ("mj_pads1", "mj_pads1_soft", "mj_pads1_lattice1", "mj_pads1_flex20")
    shapes = {"mj_pads1": "circle", "mj_pads1_soft": "square", "mj_pads1_lattice1": "diamond", "mj_pads1_flex20": "circle"}
    for j, (key, lk, hk, ys, yt, ylab) in enumerate(panels):
        x0 = 80 + j * 470
        fx, fy = P._panel(out, x0, 40, 380, 270, (0.2, 5.0), ys, Ns, yt, "pad force N per pad (N), log scale", ylab, True, True)
        P._path(out, fx, fy, sorted((r["N"], r[lk] * 1e3) for r in ref if r.get(lk)), "var(--c-ref)", width=2.0)      # m -> mm
        P._path(out, fx, fy, sorted((r["N"], r[hk] * 1e3) for r in ref if r.get(hk)), "var(--c-ref)", dashed=True, width=2.0)
        for m in models:
            pts = sorted((r["N"], r[key]) for r in X.t9 if r.get("model") == m and r.get("dt_ms") == 1.0 and ok(r) and r.get(key))
            if not pts:
                continue
            P._path(out, fx, fy, pts, COL[m], width=1.4)
            for N, v in pts:
                P._marker(out, fx(N), fy(v), COL[m], shapes[m], hollow=(m == "mj_pads1_flex20"), title=f"{m}: {v:.3f} mm at {N:g} N")
    out.append("</svg>")
    leg = [("pressure law", "var(--c-ref)", False, None), ("Hertz, 10 MPa", "var(--c-ref)", True, None)]
    for m in models:
        sl = slopes(m)
        leg.append((f"{LBL[m]} ({sl.get('delta_exp', float('nan')):.2f}, {sl.get('r_rms_exp', float('nan')):.2f})".replace("&#8202;", " "),
                    COL[m], False, shapes[m]))
    return "".join(out) + P._legend_html(leg)


def film(rel, caption, poster_rel=None):
    p = os.path.join(D, rel)
    if not os.path.exists(p):
        return ""
    pp = os.path.join(D, poster_rel) if poster_rel else None
    poster = f' poster="{P.R.data_uri(pp, "image/jpeg")}"' if pp and os.path.exists(pp) else ""
    FIG[0] += 1
    return (f'<figure><video src="{P.R.data_uri(p, "video/mp4")}"{poster} controls muted loop playsinline preload="metadata"></video>'
            f'<figcaption>Figure&#160;{FIG[0]}. {caption} <code>docs/experiments/20261007-native_compliance/{rel}</code></figcaption></figure>')


# ------------------------------------------------------------------------------------------ candidates

def _t8_sentence(X, m):
    r = {N: X.t8r(m, N) for N in (0.5, 1.0, 3.0)}
    if not all(ok(v) for v in r.values()):
        return ""
    return (f"T8: presliding {fmt(r[0.5]['u_presliding_um'], 1)}, {fmt(r[1.0]['u_presliding_um'], 1)} and {fmt(r[3.0]['u_presliding_um'], 1)}"
            f"&#8202;&#181;m at 0.5, 1 and 3&#8202;N (Mindlin 25.4, 40.3 and 83.8), recovered {fmt(r[1.0]['u_recovered_um'], 1)}&#8202;&#181;m at "
            f"1&#8202;N (38.0), loop {fmt(r[0.5]['loop_area_uJ'], 2)}, {fmt(r[1.0]['loop_area_uJ'], 2)} and {fmt(r[3.0]['loop_area_uJ'], 1)}"
            f"&#8202;&#181;J (1.93, 6.13 and 38.3).")


def _t2_list(X, m, Ns=(0.5, 1.0, 3.0)):
    v = [X.t2(m, N) for N in Ns]
    if not all(ok(x) and x.get("rbar_onset_mm") for x in v):
        return None
    return [x["rbar_onset_mm"] / LAW[N] for x, N in zip(v, Ns)], [x.get("rot_pre_deg") for x in v]


def skin(X):
    m = "mj_pads1_skin"
    t1, t5, sl = X.t1(m), X.t5(m), slopes(m)
    t2, pre = _t2_list(X, m)
    r23 = X.t8r(m, 3.0, 2.0)
    g, gp = X.gpu(m), X.gpu("mj_pads1_ir1000")
    return ("<h3>Compliant skin</h3>"
            "<p>The pad spheres sit on a child body joined to the fingertip by two slides tangent to the pad and a hinge about its "
            "normal. The springs are Mindlin&#8217;s and Lubkin&#8217;s initial stiffness per pad at 1&#8202;N from (1), 13.8&#8202;kN/m and "
            "8.55&#8202;mN&#8202;m/rad; the damping is critical and impratio is 1000. The skin is 2&#8202;g with 18&#8202;g of armature on its "
            "slides: MuJoCo softens a contact&#8217;s friction rows in proportion to the contact&#8217;s inverse weight, and with the bare "
            "2&#8202;g skin the tool slid 49&#8202;&#181;m over the spheres during each force ramp. Damping at the pads&#8217; 20&#8202;ms "
            "relaxation, 8&#215; critical, made the tool lead and trail the skin with the force rate and reversed the loop.</p>"
            f"<p>{_t8_sentence(X, m)} The skin is a linear spring with no partial slip, so the tool returns along its loading path; the "
            f"3&#8202;N loop is negative and grows with the step ({fmt(r23['loop_area_uJ'], 0) if ok(r23) else '&#8211;'}&#8202;&#181;J at "
            f"2&#8202;ms), a discretization error of the stiff spring. T9 is the pads&#8217; (exponents {fmt(sl.get('delta_exp'), 2)} and {fmt(sl.get('r_rms_exp'), 2)}). T1 effective "
            f"\\(\\mu\\) {fmt(t1['mu_eff'], 3)}; T2 onset arm {fmt(t2[0], 2)} and {fmt(t2[1], 2)} of the pressure law at 0.5 and 1&#8202;N, with "
            f"the hinge wound {fmt(pre[0], 1)}&#176; and {fmt(pre[1], 1)}&#176; before onset; T5 largest swing {fmt(t5['phi_max_deg'], 1)}&#176;, "
            f"80&#176; at {fmt(t5['N_at_80_N'], 3)}&#8202;N. At 3&#8202;N the torque ramp winds the hinge at 53&#176;/s, above T2&#8217;s "
            f"30&#176;/s spin detection, so T2 detects the hinge&#8217;s elastic wind-up there before any slip. Cost: {fmt(X.cpu(m), 1)}&#8202;{US} per step against the "
            f"pads&#8217; {fmt(X.cpu('mj_pads1_ir1000'), 1)}, {fmt(g, 2)}&#8202;&#181;s per world-step against {fmt(gp, 2)} "
            f"({g / gp:.1f}&#215;).</p>")


def bristles(X):
    m = "mj_pads1_bristle20a"
    t1, t5 = X.t1(m), X.t5(m)
    t2, pre = _t2_list(X, m)
    g, gp = X.gpu(m), X.gpu("mj_pads1_ir1000")
    c1, c3 = X.cpu(m), X.t8r(m, 3.0)
    p1, p3 = X.cpu("mj_pads1_ir1000"), X.t8r("mj_pads1_ir1000", 3.0)
    g2 = X.gpu("mj_pads2_bristle20a")
    dloop = max(abs(X.t8r(m, N, 2.0)["loop_area_uJ"] / X.t8r(m, N)["loop_area_uJ"] - 1) * 100 for N in (0.5, 1.0, 3.0))
    return ("<h3>Rolling bristles</h3>"
            "<p>Each sphere within 20&#176; of the pad&#8217;s pole sits on its own body with a ball joint at its centre and a rotational "
            "spring \\(k_\\theta = k_t r_s^2\\), \\(k_t = G A_s/h\\), with \\(h\\) = 2.7&#8202;mm so that the ~11 spheres in contact at "
            "1&#8202;N give Mindlin&#8217;s initial stiffness. The contact point of each sphere rolls elastically by \\(r_s\\theta\\) before "
            "that sphere&#8217;s Coulomb contact slips, so the lightly loaded spheres at the patch edge slip first, as in Mindlin&#8217;s "
            "partial slip. With 1&#8202;mg spheres and no armature the bristles go unstable under tangential load at 1 and 2&#8202;ms, and "
            "with 0.2&#8202;g of armature at each contact; 2&#8202;g at each contact runs.</p>"
            f"<p>{_t8_sentence(X, m)} The loops at 2&#8202;ms differ from these by at most {dloop:.0f}&#8202;% (Figure&#160;1 shows 1&#8202;N). T9 is the pads&#8217;. T1 effective "
            f"\\(\\mu\\) {fmt(t1['mu_eff'], 3)}; T2 onset arm {fmt(t2[0], 2)}, {fmt(t2[1], 2)} and {fmt(t2[2], 2)} of the pressure law; T5 largest "
            f"swing {fmt(t5['phi_max_deg'], 1)}&#176;, 80&#176; at {fmt(t5['N_at_80_N'], 3)}&#8202;N. Cost: {fmt(c1, 0)}&#8202;{US} per step "
            f"({c1 / p1:.0f}&#215; the pads; {c3['us_per_step_median'] / p3['us_per_step_median']:.0f}&#215; at 3&#8202;N), "
            f"{fmt(g, 1)}&#8202;&#181;s per world-step ({g / gp:.0f}&#215;). Bristles on the 2&#8202;mm pad cost {g2 / gp:.1f}&#215; batched, "
            "but they eject the tool at 0.5&#8202;N and slip at 3&#8202;N.</p>")


def softening(X):
    m = "mj_pads1_soft"
    sl, t5 = slopes(m), X.t5(m)
    t2, _ = _t2_list(X, m)
    lo, hi = X.t2(m, 0.5), X.t2(m, 3.0)
    p5 = X.t5("mj_pads1")
    sl2 = __import__("contact_bed_compliance").add_slopes(os.path.join(D, "t9_sweep.jsonl"), m, 2.0)
    return ("<h3>Softening impedance</h3>"
            "<p>Each sphere&#8217;s <code>solimp</code> impedance falls with depth (\\(d_{\\min} > d_{\\max}\\)), fitted so that its static "
            "force follows \\(K_s\\sqrt{r\\,r_\\text{ref}}\\) over 0.03&#8211;0.5&#8202;mm of overlap \\(r\\) (within 7&#8202;%), equal to the "
            "linear sphere&#8217;s \\(K_s r\\) at \\(r_\\text{ref}\\) = 0.2&#8202;mm. Pressure proportional to the square root of depth on a "
            "sphere gives Hertz&#8217;s load exponents. Inside the <code>solimp</code> width MuJoCo&#8217;s static force is "
            "\\(f = r\\,d(r)/\\big(d_{\\max}^2(1-d(r))\\,t_c^2 w\\big)\\), measured on a sphere&#8211;plane contact. The tangential behaviour "
            "is the pads&#8217;.</p>"
            f"<p>T9 exponents {fmt(sl.get('delta_exp'), 2)} (approach) and {fmt(sl.get('r_rms_exp'), 2)} (radius) against Hertz&#8217;s 0.67 "
            f"and 0.33 (Figure&#160;3); {fmt(sl2.get('delta_exp'), 2)} and {fmt(sl2.get('r_rms_exp'), 2)} at 2&#8202;ms. T2 arm ratio "
            f"{fmt(hi['rbar_onset_mm'] / lo['rbar_onset_mm'], 2)} (Hertz 1.82, pressure law 1.57), onset arm {fmt(t2[0], 2)}, {fmt(t2[1], 2)} and "
            f"{fmt(t2[2], 2)} of the law. T1 is the pads&#8217;. In T5 the arm shrinks faster as the pinch drops, and the tool passes 80&#176; at "
            f"{fmt(t5['N_at_80_N'], 2)}&#8202;N against the pads&#8217; {fmt(p5['N_at_80_N'], 2)} (largest swing {fmt(t5['phi_max_deg'], 1)}&#176;). "
            f"Cost: the pads&#8217; ({fmt(X.cpu(m), 1)}&#8202;{US} per step).</p>")


def dropped(X):
    la, fl = "mj_pads1_lattice1", "mj_pads1_flex20"
    sla, sfl = slopes(la), slopes(fl)
    r9 = pick(X.t9, la, N=1.0, dt_ms=1.0)
    t1, t5 = X.t1(la), X.t5(la)
    t2, _ = _t2_list(X, la)
    f8 = X.t8r(fl)
    f9 = pick(X.t9, fl, N=1.0, dt_ms=1.0)
    gp = X.gpu("mj_pads1_ir1000")
    return ("<h3>Lattice and flex</h3>"
            "<p>The lattice puts each sphere within 20&#176; of the pole on a slide along its normal with an anchor spring, and couples "
            "neighbouring slides by fixed tendons on the difference of their deflections: the shear layer of a Pasternak foundation, with a "
            "coupling length of 1&#8202;mm. The sphere contact is 10&#215; stiffer so that the lattice springs set the compliance. Its patch radius "
            f"grows with exponent {fmt(sla.get('r_rms_exp'), 2)} (Hertz 0.33), and at the same approach the patch is "
            f"{fmt(r9.get('r_rms_over_law_at_delta'), 2)} of the law&#8217;s (Hertz 0.78). Its 1&#8202;g nodes raise the contacts&#8217; inverse "
            f"weight, which softens their friction rows: effective \\(\\mu\\) {fmt(t1['mu_eff'], 2)} at 1&#8202;N, T2 onset arm {fmt(t2[1], 2)} of the "
            f"law, T5 largest swing {fmt(t5['phi_max_deg'], 0)}&#176;. It costs {fmt(X.gpu(la), 1)}&#8202;&#181;s per world-step "
            f"({X.gpu(la) / gp:.0f}&#215;), and with a 2&#8202;mm coupling length it is unstable at 1&#8202;ms.</p>"
            "<p>The flex replaces the spheres within 20&#176; of the pole by a MuJoCo <code>flexcomp</code>: 165 points and 540 linear "
            "tetrahedra per pad (1&#8202;mm hexagonal columns, two 1.5&#8202;mm layers, the back layer and the rim pinned to the pad), "
            "St&#160;Venant&#8211;Kirchhoff elasticity at \\(E\\) = 10&#8202;MPa and \\(\\nu\\) = 0.45, 0.10&#8202;g of TPU per cap, its surface "
            "tetrahedra in contact with the tool. MuJoCo-Warp 3.14 has flex elasticity and flex&#8211;cylinder collision but only the Euler, "
            "RK4 and implicit integrators, and MuJoCo 3.14 accepts flex elasticity only under its <code>discrete</code> integrator. With "
            "explicit integration (MuJoCo 3.6 <code>implicitfast</code>, 3.14 Euler) the cap diverges at 1&#8202;ms with up to 100&#8202;g of armature on "
            "each vertex degree of freedom, and a 4&#215;4&#215;3&#8202;mm flex block at \\(\\nu\\) = 0.45 still oscillates between 9.5 and "
            "21&#8202;&#181;m about its 14.0&#8202;&#181;m static compression after 6&#8202;s with 3&#8202;kg. Under the discrete integrator on one core it runs at 1&#8202;ms with the TPU&#8217;s mass, and a "
            "4&#215;4&#215;3&#8202;mm flex block under a uniform 1&#8202;N load compresses 19.0&#8202;&#181;m against the exact 18.75 "
            f"(\\(\\nu\\) = 0). Its patch radius grows with exponent {fmt(sfl.get('r_rms_exp'), 2)} and its approach with "
            f"{fmt(sfl.get('delta_exp'), 2)} ({fmt(f9['delta_mm'] * 1e3, 0)}&#8202;&#181;m at 1&#8202;N; Hertz 77). In T8 at 1&#8202;N the tool creeps "
            f"over the light vertices: presliding {fmt(f8['u_presliding_um'], 1)}&#8202;&#181;m, recovered {fmt(f8['u_recovered_um'], 1)}, loop "
            f"{fmt(f8['loop_area_uJ'], 0)}&#8202;&#181;J. It costs {num(X.cpu(fl), ',.0f')}&#8202;{US} per step, "
            f"{X.cpu(fl) / X.cpu('mj_pads1_ir1000'):.0f}&#215; the pads.</p>")


def cands(X):
    return (skin(X) + bristles(X) +
            figure(svg_loops(X), "Task&#160;8 at 1&#8202;N, 1&#8202;ms: the tool&#8217;s axial displacement against the force over the whole "
                                 "cycle for the pads, the compliant skin and the rolling bristles, against Mindlin&#8217;s closed form. The "
                                 "pads creep while the force is positive; the skin returns along its loading line; the bristles trace a loop.") +
            film("media/t8_mj_pads1_bristle20a.mp4", "Task&#160;8 at 1&#8202;N on the rolling bristles, real time: the &#8722;x pad seen "
                 "through the tool, its spheres coloured by contact force (dark to light), the tool&#8217;s displacement traced along the bottom.",
                 "media/t8_mj_pads1_bristle20a.jpg") +
            softening(X) +
            figure(svg_t9(X), "Task&#160;9, 1&#8202;ms: (a) approach and (b) RMS patch radius against the pinch force, log axes, with the "
                              "pressure law and Hertz at 10&#8202;MPa computed without a simulator. Legend values are the fitted approach and "
                              "radius exponents. The pads follow the law; the softening impedance follows Hertz&#8217;s slopes on the law&#8217;s "
                              "patch size; the lattice and the flex follow Hertz&#8217;s radius slope on a smaller patch.") +
            dropped(X))


# ------------------------------------------------------------------------------------------ comparison

def compare(X):
    adds = []
    for m in ("mj_pads1",) + CANDS:
        r1, sl = X.t8r(m), slopes(m)
        adds.append([LBL[m], fmt(r1["u_presliding_um"], 1) if ok(r1) else "&#8211;", fmt(r1["u_recovered_um"], 1) if ok(r1) else "&#8211;",
                     fmt(r1["loop_area_uJ"], 2) if ok(r1) else "&#8211;", fmt(sl.get("delta_exp"), 2), fmt(sl.get("r_rms_exp"), 2)])
    adds.append(["Drake hydroelastic", "0.05", "&#8722;0.05", "0.12", fmt(slopes("drake_hydro").get("delta_exp"), 2),
                 fmt(slopes("drake_hydro").get("r_rms_exp"), 2)])
    adds.append(["Mindlin, Hertz (10&#8202;MPa)", "40.3", "38.0", "6.13", "0.67", "0.33"])
    t3 = table(["model", "presliding, 1&#8202;N (&#181;m)", "recovered (&#181;m)", "loop (&#181;J)", "approach exponent", "radius exponent"],
               adds, "Table 3: what each candidate adds, T8 at 1&#8202;N and T9, 1&#8202;ms step.")
    keeps = []
    gp, cp = X.gpu("mj_pads1_ir1000"), X.cpu("mj_pads1_ir1000")
    run1 = {"mj_pads1": "yes", "mj_pads1_skin": "yes", "mj_pads1_bristle20a": "with 2&#8202;g armature", "mj_pads1_soft": "yes",
            "mj_pads1_lattice1": "1&#8202;mm coupling", "mj_pads1_flex20": "CPU, discrete"}
    for m in ("mj_pads1",) + CANDS:
        t1, t5 = X.t1(m), X.t5(m)
        t2 = _t2_list(X, m, (0.5, 1.0))
        g, c = X.gpu(m) if m != "mj_pads1" else gp, X.cpu(m) if m != "mj_pads1" else cp
        keeps.append([LBL[m], fmt(t1.get("mu_eff"), 3) if t1 else "&#8211;",
                      f"{fmt(t2[0][0], 2)}, {fmt(t2[0][1], 2)}" if t2 else "&#8211;",
                      f"{fmt(t5['phi_max_deg'], 1)}, {fmt(t5['N_at_80_N'], 2)}" if ok(t5) else "&#8211;",
                      num(c, ",.0f") if c else "&#8211;", f"{fmt(g, 2)} ({g / gp:.1f}&#215;)" if g else "&#8211;", run1[m]])
    d1, d5 = pick(X.bpu, "drake_hydro", N=1.0, dt_ms=1.0), X.t5("drake_hydro")
    dt2 = _t2_list(X, "drake_hydro", (0.5, 1.0))
    keeps.append(["Drake hydroelastic", fmt(d1.get("mu_eff"), 3) if d1 else "&#8211;", f"{fmt(dt2[0][0], 2)}, {fmt(dt2[0][1], 2)}" if dt2 else "&#8211;",
                  f"{fmt(d5['phi_max_deg'], 1)}, {fmt(d5['N_at_80_N'], 2)}" if ok(d5) else "&#8211;", num(X.cpu("drake_hydro"), ",.0f"),
                  "&#8211;", "yes"])
    t4 = table(["model", "T1", "T2", "T5", "CPU", "GPU", '<span style="text-transform:none">1&#8202;ms</span>'], keeps,
               "Table 4: what each candidate keeps of the pads&#8217; results and what it costs. T1: effective \\(\\mu\\) at 1&#8202;N. T2: onset "
               "arm over the pressure law&#8217;s at 0.5 and 1&#8202;N. T5: largest swing (&#176;) and 80&#176; force (N). CPU: &#181;s per step "
               "on one core at 1&#8202;N. GPU: &#181;s per world-step at 4,096 worlds and its multiple of the pads&#8217;. 1&#8202;ms: whether "
               "the model runs at a 1&#8202;ms step. The pads&#8217; GPU value is the impratio-1000 "
               "control on the same rig. The softening impedance changes only <code>solimp</code>, so its GPU cost is the pads&#8217;; the flex "
               "has no GPU value because MuJoCo-Warp has no discrete integrator.")
    return (t3 + t4 +
            "<p>The skin is the one candidate that keeps T1, T2 at 0.5 and 1&#8202;N, and T5 within the pads&#8217; agreement at under "
            f"10&#215; their cost, and the contact-model policy training (<code>{POLICY_NOTE}</code>) trains it as its compliant arm "
            f"(arm e). The bristles keep the same tasks at {X.gpu('mj_pads1_bristle20a') / gp:.0f}&#215; the batched cost; the softening impedance "
            "moves T5&#8217;s 80&#176; force.</p>")


def lede(X):
    s, b = X.t8r("mj_pads1_skin"), X.t8r("mj_pads1_bristle20a")
    sl = slopes("mj_pads1_soft")
    gp = X.gpu("mj_pads1_ir1000")
    t5s, t5p = X.t5("mj_pads1_soft"), X.t5("mj_pads1")
    return ("The 1&#8202;mm sphere pads, point contact and Drake store no tangential displacement: under an axial force cycled at half "
            "the slip force they creep and recover nothing, where Mindlin&#8217;s elastic contact moves 40&#8202;&#181;m at 1&#8202;N and recovers "
            f"38. Three MuJoCo-native additions each supply one missing property. A spring-mounted skin gives {s['u_presliding_um']:.0f}&#8202;&#181;m "
            f"of recoverable presliding with no hysteresis at {X.gpu('mj_pads1_skin') / gp:.1f}&#215; the pads&#8217; GPU cost and keeps their pull "
            "and brake results and their twist onset to 1&#8202;N; the contact-model policy training uses it as its compliant arm. Spring-loaded ball joints under the spheres (rolling "
            f"bristles) add partial slip and a {b['loop_area_uJ']:.1f}&#8202;&#181;J loop against Mindlin&#8217;s 6.1, at "
            f"{X.gpu('mj_pads1_bristle20a') / gp:.0f}&#215; the cost. A <code>solimp</code> impedance that softens with depth gives Hertz&#8217;s "
            f"load exponents ({sl['delta_exp']:.2f} and {sl['r_rms_exp']:.2f} against 2/3 and 1/3) at no cost and moves the braked swing&#8217;s "
            f"80&#176; crossing from {t5p['N_at_80_N']:.2f} to {t5s['N_at_80_N']:.2f}&#8202;N. A neighbour-coupled lattice and a MuJoCo flex "
            "are dropped: their light nodes soften friction, and the flex runs at 1&#8202;ms only under MuJoCo 3.14&#8217;s discrete integrator, "
            "which MuJoCo-Warp lacks.")


def open_items():
    items = [("The tip&#8217;s modulus.", "The skin, the bristles and the lattice are sized on Hertz at 10&#8202;MPa, where the pressure "
              "law&#8217;s patch is twice Hertz&#8217;s radius; the two agree near 0.8&#8202;MPa. An indentation test on the printed tip, force "
              "against depth at 0.25&#8211;4&#8202;N, gives \\(E\\) and the approach exponent, which decides between the pads&#8217; 1/2 and "
              "the softening impedance&#8217;s 2/3; T8 on the bench, a slow axial force cycle at a fixed pinch, measures presliding and "
              "loop area."),
             ("T2 on a compliant tip.", "T2 detects spin onset when the tool turns faster than 30&#8202;&#176;/s, and the skin&#8217;s hinge "
              "winds faster than that at 3&#8202;N. Onset taken from the tool&#8217;s rotation relative to the skin body in "
              "<code>scripts/contact_bed_twist.py</code> would measure slip there.")]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def render_tex(t):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    return P.TEX_RE.sub(lambda m: next(svgs), t), len(items)


def main():
    X = Data()
    v = {"STYLE": P.style_block(), "BUILT": time.strftime("%Y-%m-%d %H:%M"), "BED_PATH": BED_PATH,
         "OVERVIEW_PATH": os.path.relpath(P.OUT, ROOT)}
    u = P.bed_url()
    v["BED_LINK"] = f", <a href=\"{u}\">artifact</a>" if u else ""
    v["LEDE"] = lede(X)
    v["GLOSSARY"] = glossary()
    v["REFS"] = refs()
    v["BASE"] = base(X)
    v["CANDS"] = cands(X)
    v["COMPARE"] = compare(X)
    v["OPEN"] = open_items()
    v["FOOTER"] = ("<p>Rebuild: <code>logs/20261001-hom_contact/venv/bin/python scripts/native_compliance_page.py</code>. Rows: "
                   "<code>docs/experiments/20261007-native_compliance/*.jsonl</code> and <code>reference_laws.json</code>; the pads&#8217; "
                   "and Drake&#8217;s T1, T2 and T5 rows are the bed&#8217;s (<code>docs/experiments/20261005-contact_bed/</code>). Rigs: "
                   "<code>scripts/contact_bed_candidates.py</code> (the flex rows need MuJoCo 3.14; every other row is MuJoCo 3.6).</p>")
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
