#!/usr/bin/env python3
r"""Build docs/experiments/20261006-fingertip_backends/20261006-fingertip_contact_backends.html.

    python3 scripts/fingertip_backends_page.py

Reads the bed's creep rows (20261005-contact_bed/creep.jsonl, task T6 of its PROTOCOL.md) and Drake reference rows,
the printed-fingertip rows of scripts/tpu_tip_rig.py (tip_static, tip_T1, tip_T2, tip_T5), the reorientation rows of
scripts/reorient_backends*.py (reorient_mujoco, reorient_drake_*, reorient_gpu) and the films in media/. Every number in
the prose is computed here from those rows. House style, table and chart helpers come from contact_overview_page.py.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import fingertip_geometry as FG  # noqa: E402
import reorient_backends_summary as RS  # noqa: E402
import retro_style  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261006-fingertip_backends")
BED = os.path.join(ROOT, "docs/experiments/20261005-contact_bed")
OUT = os.path.join(D, "20261006-fingertip_contact_backends.html")
TPL = os.path.join(ROOT, "scripts/fingertip_backends_page.template.html")
BED_PATH = "docs/experiments/20261005-contact_bed/20261005-contact_model_bed.html"
OVERVIEW_PATH = "docs/experiments/20261005-contact_overview/20261005-sphere_pad_contact_model.html"
HANDS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
_svg_open, _panel, _path, _marker, _legend = P._svg_open, P._panel, P._path, P._marker, P._legend_html
num = P.num

EXTRA_CSS = """
td.hf{font-variant-numeric:tabular-nums;white-space:nowrap}
td.hf0{background:color-mix(in srgb,var(--bad) 22%,transparent)}
td.hf1{background:color-mix(in srgb,var(--bad) 10%,transparent)}
td.hf2{background:transparent}
td.hf3{background:color-mix(in srgb,var(--good) 12%,transparent)}
td.hf4{background:color-mix(in srgb,var(--good) 26%,transparent)}
tr.bench td{font-weight:600;border-top:2px solid var(--ink3)}
.tw th{text-transform:none;letter-spacing:.01em}
td.lab{white-space:normal;min-width:15em}
.refs li{margin:0 0 8px}
"""


# ------------------------------------------------------------------------------------------ helpers

def jl(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def f(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "&#8211;"
    return num(x, f".{nd}f")


def sig(x, n=2):
    """n significant figures."""
    if x is None or x == 0 or not math.isfinite(x):
        return "&#8211;" if x is None else "0"
    d = max(0, n - 1 - int(math.floor(math.log10(abs(x)))))
    return num(x, f".{d}f")


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


FIG = [0]


def figure(svg, caption):
    FIG[0] += 1
    return f'<figure class="diagram">{svg}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def film(rel, caption, poster=None):
    p = os.path.join(D, rel)
    if not os.path.exists(p):
        return ""
    pp = os.path.join(D, poster) if poster else None
    po = f' poster="{P.R.data_uri(pp, "image/jpeg")}"' if pp and os.path.exists(pp) else ""
    return (f'<figure><video src="{P.R.data_uri(p, "video/mp4")}"{po} controls muted loop playsinline preload="metadata">'
            f'</video><figcaption>{caption} <code>docs/experiments/20261006-fingertip_backends/{rel}</code></figcaption></figure>')


def eq(a, b, tol=1e-9):
    return a is not None and b is not None and abs(float(a) - float(b)) < tol


def one(rows, **kw):
    for r in rows:
        if all(eq(r.get(k), v) if isinstance(v, (int, float)) and not isinstance(v, bool) else r.get(k) == v
               for k, v in kw.items()):
            return r
    return None


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    if len(a) < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num_ = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num_ / den if den > 0 else None


# ------------------------------------------------------------------------------------------ terms

GLOSSARY = [
    ("creep", "Steady sliding of a pinched tool while its load stays below the slip load. <i>Translational creep</i>: axial sliding "
              "speed during a 1&#8202;s hold at half the axial force at which the tool slips (bed task&#160;1), in &#181;m/s. "
              "<i>Twist creep</i>: turning speed about the pinch axis during a 1&#8202;s hold at half the torque at which it spins "
              "(bed task&#160;2), in &#176;/s. Rigid Coulomb friction gives zero."),
    ("pad force N", "Normal force of one fingertip on the tool on the two-pad rig, in N; the rig holds it constant with a force-controlled rail."),
    ("slip onset &#956;<sub>eff</sub>", "Axial force at which the tool starts to slide, divided by 2N; rigid Coulomb friction with &#956;&#8202;=&#8202;1 gives 1.000."),
    ("friction arm r&#772;", "Torque about the pinch axis at which the tool starts to spin, divided by 2&#956;N, in mm per pad (bed task&#160;2). "
                              "A pad transmits at most &#956;N r&#772; about its normal. <i>Static arm</i>: the force-weighted mean distance of the contact "
                              "points (or hydroelastic faces) from their centre, in mm, with the tool pressed at N and no load."),
    ("brake end, N at 45&#176;", "Bed task&#160;5: the tool, its centre of mass 15&#8202;mm from the pinch axis, hangs in the pinch while the pad force falls "
                                  "from 6 to 0.2&#8202;N over 4&#8202;s. Brake end is the final swing angle (90&#176;&#8202;=&#8202;hanging), N at 45&#176; the pad force when "
                                  "the swing passes 45&#176;."),
    ("impratio, noslip, t<sub>r</sub>", "MuJoCo solver settings. <code>impratio</code> scales the friction rows&#8217; stiffness relative to the normal row "
                                         "(elliptic cone). <code>noslip_iterations</code> runs a second solve of the friction rows without softness after the main "
                                         "solve. t<sub>r</sub> is the sphere pads&#8217; relaxation time: their contact damping divided by their stiffness."),
    ("sphere pad", "A fingertip surface packed with 0.75&#8202;mm collision spheres 1&#8202;mm apart, each a spring of stiffness E/h&#8202;&#183;&#8202;A<sub>s</sub> "
                   "(A<sub>s</sub> its share of the surface, h the foundation depth). Contact patch, centre of pressure and torsion follow from which spheres touch."),
    ("contact representation", "<i>Sphere</i>: the 10.55&#8202;mm sphere at the end of real_hand.xml, one MuJoCo contact. <i>Legacy box</i>: the sharp "
                               "box (21.1&#8202;&#215;&#8202;14.8&#8202;mm face) the plans were exported against, one MuJoCo contact. <i>TPU mesh</i>: the printed block as one "
                               "convex mesh, one MuJoCo contact. <i>TPU pads</i>: the block covered with 1&#8202;mm sphere pads. <i>Drake</i>: the same shape as a "
                               "compliant hydroelastic body (E&#8202;10&#8202;MPa, relaxation 0.01&#8202;s) against the rigid tool."),
    ("fillet r<sub>f</sub>", "Radius of the rounded edges of the printed fingertip block, in mm. r<sub>f</sub>&#8202;=&#8202;0 is the sharp box; the "
                             "palmar face is flat over (14.8&#8202;&#8722;&#8202;2r<sub>f</sub>)&#8202;&#215;&#8202;(22&#8202;&#8722;&#8202;2r<sub>f</sub>)&#8202;mm."),
    ("held", "A reorientation seed holds the tool when, at the end of the bench schedule&#8217;s 1.5&#8202;s hold, at least two fingers touch it and its "
             "centre is within 20&#8202;mm of its height at the end of the grip. Reported as held seeds over seeds."),
    ("turn", "Net change of the tool&#8217;s tilt from the end of the grip to the end of the hold, in degrees, positive toward vertical "
             "(the plan&#8217;s direction). Bench values are the net turn on held trials (paper/figures/ranking.json)."),
    ("seed", "One placement of the tool on the post: seed 0 at the plan&#8217;s pose, seeds 1&#8211;8 (and 9&#8211;63 on the GPU) shifted by "
             "normal draws of sd 2&#8202;mm in x and y and 2&#176; in yaw. The same seed is the same placement in every simulator."),
    ("paired agreement", "For two conditions run on the same hands and seeds, the fraction of (hand, seed) pairs with the same held outcome, "
                         "and the median absolute difference in turn on pairs held in both."),
    ("cost", "<i>&#181;s per step</i>: median wall time of one 1&#8202;ms physics step on one CPU core. <i>&#181;s per world-step</i>: GPU wall time "
             "of one step divided by the number of parallel worlds."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in GLOSSARY) + "</dl>"


# ------------------------------------------------------------------------------------------ creep

def creep_data():
    C = [r for r in jl(os.path.join(BED, "creep.jsonl")) if r.get("part") in ("T1", "T2", "T4", "T5")]
    pull = jl(os.path.join(BED, "pull_slip.jsonl"))
    twist = jl(os.path.join(BED, "twist_slip.jsonl"))
    brake = jl(os.path.join(BED, "brake.jsonl"))
    nsi = jl(os.path.join(D, "creep_noslip_iters.jsonl"))
    chain = jl(os.path.join(D, "chain_impratio.jsonl"))
    return C, pull, twist, brake, nsi, chain


def cr(C, part, model, ns, ir, N=None, dt=1.0):
    for r in C:
        if r["part"] == part and r["model"] == model and r["noslip_iterations"] == ns and eq(r["impratio"], ir) \
                and eq(r["dt_ms"], dt) and (N is None or eq(r.get("N"), N)):
            return r
    return None


def drake_ref(pull, twist, brake, N=1.0, dt=1.0):
    p = next((r for r in pull if r.get("chain_spec") == "drake:hydro:rt0.01" and eq(r["N"], N) and eq(r["dt_ms"], dt)), None)
    t = next((r for r in twist if r.get("model") == "drake_hydro" and eq(r["N"], N) and eq(r["dt_ms"], dt)), None)
    b = next((r for r in brake if r.get("model") == "drake_hydro" and eq(r["dt_ms"], dt)), None)
    return p, t, b


def svg_creep(C, pull, twist):
    W, H = 980, 400
    out = _svg_open(W, H, "Creep of the 1 mm sphere pads against impratio, translational and twist, at three pad forces, "
                          "with Drake hydroelastic at 1 and 5 ms steps.")
    irs = [100, 300, 1000, 3000, 10000]
    cols = {0.5: "var(--s3)", 1.0: "var(--s1)", 3.0: "var(--s2)"}
    legend = []
    for k, (part, key, unit, ys, yt, yl) in enumerate((
            ("T1", "creep_mm_s", 1e3, (0.05, 100), (0.1, 1, 10, 100), "translational creep (&#181;m/s), log"),
            ("T2", "creep_deg_s", 1.0, (0.002, 5), (0.01, 0.1, 1), "twist creep (&#176;/s), log"))):
        x0 = 80 + k * 470
        fx, fy = _panel(out, x0, 40, 360, 290, (80, 12500), ys, irs, yt, "impratio, log", yl, True, True,
                        xfmt="{:g}", yfmt=lambda v: f"{v:g}")
        for N, col in cols.items():
            pts = []
            for ir in irs:
                r = cr(C, part, "mj_pads1", 0, ir, N, 1.0)
                if r and r.get(key) and r[key] > 0:
                    pts.append((ir, r[key] * unit))
            _path(out, fx, fy, pts, col, width=1.6)
            for x, y in pts:
                _marker(out, fx(x), fy(y), col, "circle", title=f"{N:g} N, impratio {x:g}: {y:.3g}")
            # Drake at 1 and 5 ms as ticks at the right edge
            src = pull if part == "T1" else twist
            for dt, hollow in ((1.0, False), (5.0, True)):
                if part == "T1":
                    rr = next((r for r in src if r.get("chain_spec") == "drake:hydro:rt0.01" and eq(r["N"], N) and eq(r["dt_ms"], dt)), None)
                else:
                    rr = next((r for r in src if r.get("model") == "drake_hydro" and eq(r["N"], N) and eq(r["dt_ms"], dt)), None)
                if rr and rr.get(key):
                    y = rr[key] * unit
                    if ys[0] <= y <= ys[1]:
                        _marker(out, fx(12500) - (6 if dt == 1.0 else -6) - 4, fy(y), col, "diamond", hollow=hollow,
                                title=f"Drake, {N:g} N, {dt:g} ms: {y:.3g}")
            if k == 0:
                legend.append((f"MuJoCo 1 mm pads, {N:g} N", col, False, "circle"))
    out.append("</svg>")
    legend += [("Drake hydroelastic at 1 ms (filled) and 5 ms (hollow), same force", "var(--ink3)", False, "diamond")]
    return "".join(out) + _legend(legend)


def creep_section():
    C, pull, twist, brake, nsi, chain = creep_data()
    if not C:
        return P.pending("creep rows missing")
    dp, dtw, db = drake_ref(pull, twist, brake, 1.0, 1.0)
    dp5, dtw5, db5 = drake_ref(pull, twist, brake, 1.0, 5.0)
    base1 = cr(C, "T1", "mj_pads1", 0, 100, 1.0)
    base2 = cr(C, "T2", "mj_pads1", 0, 100, 1.0)
    base5 = cr(C, "T5", "mj_pads1", 0, 100)
    lo1, hi1 = cr(C, "T1", "mj_pads1", 0, 100, 0.5), cr(C, "T1", "mj_pads1", 0, 100, 3.0)
    rows = []

    def add(label, model, ns, ir, gpu, note_twist=None):
        t1, t2, t5 = cr(C, "T1", model, ns, ir, 1.0), cr(C, "T2", model, ns, ir, 1.0), cr(C, "T5", model, ns, ir)
        t5b = cr(C, "T5", model, ns, ir, dt=5.0)
        if not t1:
            return
        tw = t2.get("creep_deg_s") if t2 else note_twist
        rows.append([label, sig(t1["creep_mm_s"] * 1e3, 2), sig(tw, 2) if isinstance(tw, float) else (tw or "&#8211;"),
                     f(t1.get("mu_eff"), 3), f(t2.get("rbar_onset_mm"), 3) if t2 else "&#8211;",
                     (f(t5.get("phi_end_deg"), 1) + (" / " + f(t5b.get("phi_end_deg"), 1) if t5b and t5b.get("status") == "complete"
                                                      else (" / ejected" if t5b else ""))) if t5 else "&#8211;",
                     f(t5.get("N_at_45_N"), 3) if t5 else "&#8211;", f(t1.get("us_per_step_median"), 0), gpu])
    add("1 mm pads, impratio 100 (the bed&#8217;s)", "mj_pads1", 0, 100, "yes")
    add("impratio 300", "mj_pads1", 0, 300, "yes")
    add("impratio 1000", "mj_pads1", 0, 1000, "yes")
    add("impratio 3000", "mj_pads1", 0, 3000, "yes")
    add("impratio 10&#8202;000", "mj_pads1", 0, 10000, "yes")
    add("noslip 1, impratio 100", "mj_pads1", 1, 100, "no")
    add("noslip 3, impratio 100", "mj_pads1", 3, 100, "no")
    add("noslip 10, impratio 100", "mj_pads1", 10, 100, "no")
    add("noslip 10, impratio 1000", "mj_pads1", 10, 1000, "no")
    add("t<sub>r</sub> 0.05&#8202;s, impratio 100", "mj_pads1_tr05", 0, 100, "yes")
    add("t<sub>r</sub> 0.1&#8202;s, impratio 100", "mj_pads1_tr10", 0, 100, "yes")
    add("t<sub>r</sub> 0.1&#8202;s, impratio 1000", "mj_pads1_tr10", 0, 1000, "yes")
    if dp and dtw:
        rows.append(["Drake hydroelastic, 1&#8202;ms / 5&#8202;ms", f"{sig(dp['creep_mm_s'] * 1e3)} / {sig(dp5['creep_mm_s'] * 1e3)}",
                     f"{sig(dtw['creep_deg_s'])} / {sig(dtw5['creep_deg_s'])}", f(dp.get("mu_eff"), 3), f(dtw.get("rbar_onset_mm"), 3),
                     f"{f(db.get('phi_end_deg'), 1)} / {f(db5.get('phi_end_deg'), 1)}" if db and db5 else "&#8211;",
                     f(db.get("N_at_45_N"), 3) if db else "&#8211;", f(dp.get("us_per_step_median"), 0), "&#8211;"])
    tab_cap = '<p class="note">Table&#160;1. Creep and its side effects on the 1&#8202;mm sphere pads at 1&#8202;N and a 1&#8202;ms step (brake end also at 5&#8202;ms). MuJoCo-Warp: whether the setting runs in MuJoCo-Warp 3.6.</p>'
    tab = table(["condition (1 N, 1&#8202;ms)", "creep &#181;m/s", "twist creep &#176;/s", "&#956;<sub>eff</sub>", "r&#772; onset mm",
                 "brake end &#176; (1 / 5&#8202;ms)", "N at 45&#176;", "&#181;s/step", "MuJoCo-Warp"], rows)
    ns_rows = sorted([r for r in nsi if r.get("part") == "T2"], key=lambda r: (r["impratio"], r["noslip_iterations"]))
    ns_txt = ", ".join(f"{r['noslip_iterations']} iterations {sig(r['creep_deg_s'])}&#8202;&#176;/s at {f(r['us_per_step_median'], 0)}&#8202;&#181;s"
                       for r in ns_rows if eq(r["impratio"], 100))
    ch = {int(r["impratio"]): r for r in chain}

    def chain_cost(r):
        ph, sm = r["perf"]["phys"], r["perf"]["sim"]
        return 1000 * sum(ph.values()) / sum(sm.values())
    ir1e4_1 = cr(C, "T1", "mj_pads1", 0, 10000, 1.0)
    ir1e4_2 = cr(C, "T2", "mj_pads1", 0, 10000, 1.0)
    ir1e3_1 = cr(C, "T1", "mj_pads1", 0, 1000, 1.0)
    ir1e3_2 = cr(C, "T2", "mj_pads1", 0, 1000, 1.0)
    tr05 = cr(C, "T5", "mj_pads1_tr05", 0, 100)
    tr05b = cr(C, "T5", "mj_pads1_tr05", 0, 100, dt=5.0)
    ns10 = cr(C, "T1", "mj_pads1", 10, 100, 1.0)
    ns10t = cr(C, "T2", "mj_pads1", 10, 100, 1.0)
    out = [f"""<p>MuJoCo&#8217;s friction rows are soft: under a steady sub-threshold load each loaded contact behaves as a damper, so the
tool slides at a constant speed instead of sticking. At steady creep the friction row carries R<sub>t</sub>f<sub>t</sub>&#8202;=&#8202;&#8722;b&#8202;v with
b&#8202;=&#8202;2/(d<sub>0</sub>t<sub>c</sub>) and R<sub>t</sub>&#8202;=&#8202;(1&#8202;&#8722;&#8202;d<sub>0</sub>)&#923;/(d<sub>0</sub>&#8202;impratio). With the pad stiffness calibrated at load,
K&#8202;=&#8202;1/(t<sub>c</sub><sup>2</sup>(1&#8202;&#8722;&#8202;d<sub>0</sub>)&#923;) and t<sub>r</sub>&#8202;=&#8202;2t<sub>c</sub>, the creep speed of n loaded spheres sharing a
tangential load F<sub>t</sub> is</p>
<p class="eqline">v &#8776; F<sub>t</sub> / (impratio &#183; n &#183; K &#183; t<sub>r</sub>),</p>
<p>linear in the load per sphere, inverse in impratio and in the relaxation time, and independent of the step. The 1&#8202;mm pads creep
{sig(base1['creep_mm_s'] * 1e3)}&#8202;&#181;m/s at 1&#8202;N and turn {sig(base2['creep_deg_s'])}&#8202;&#176;/s under half their twist load; Drake creeps
{sig(dp['creep_mm_s'] * 1e3)}&#8202;&#181;m/s and turns {sig(dtw['creep_deg_s'])}&#8202;&#176;/s at a 1&#8202;ms step, about {sig(base1['creep_mm_s'] / dp['creep_mm_s'], 2)} and
{sig(base2['creep_deg_s'] / dtw['creep_deg_s'], 2)} times less. Drake also creeps: its regularised friction gives a speed that grows with the step,
{sig(dp5['creep_mm_s'] * 1e3)}&#8202;&#181;m/s at 5&#8202;ms. Table&#160;1 crosses impratio 100&#8211;10&#8202;000, noslip 1&#8211;10 iterations and t<sub>r</sub>
0.02&#8211;0.1&#8202;s on the bed&#8217;s pull, twist, shake and brake tasks ({len(C)} rows, <code>docs/experiments/20261005-contact_bed/creep.jsonl</code>).</p>
""",
           tab, tab_cap,
           figure(svg_creep(C, pull, twist),
                  "Creep against impratio for the 1&#8202;mm pads at 0.5, 1 and 3&#8202;N (1&#8202;ms step): both speeds fall as 1/impratio. "
                  "Diamonds at the right edge: Drake at the same force, 1&#8202;ms filled and 5&#8202;ms hollow."),
           f"""<p><b>impratio</b> lowers both creeps as 1/impratio. At 10&#8202;000 the pads creep
{sig(ir1e4_1['creep_mm_s'] * 1e3)}&#8202;&#181;m/s and turn {sig(ir1e4_2['creep_deg_s'])}&#8202;&#176;/s, inside Drake&#8217;s own range between its 1 and 5&#8202;ms
steps; slip onset stays at &#956;<sub>eff</sub>&#8202;{f(ir1e4_1['mu_eff'], 3)}, the twist onset arm moves from {f(base2['rbar_onset_mm'], 3)} to
{f(ir1e4_2['rbar_onset_mm'], 3)}&#8202;mm (Drake {f(dtw['rbar_onset_mm'], 3)}), and the brake ends at {f(cr(C, 'T5', 'mj_pads1', 0, 10000)['phi_end_deg'], 1)}&#176;
against Drake&#8217;s {f(db['phi_end_deg'], 1)}&#176;. The rig&#8217;s step cost does not change. On the full hand chain of 2026-10-02 (pick, lift,
braked swing, peg-hole insert) the physics time per simulated second is {f(chain_cost(ch[100]), 0)}, {f(chain_cost(ch[1000]), 0)},
{f(chain_cost(ch[3000]), 0)} and {f(chain_cost(ch[10000]), 0)}&#8202;ms at impratio 100, 1000, 3000 and 10&#8202;000; the chain completes in all four, the
swing ends at {f(ch[100]['phi_brake_end'], 1)}, {f(ch[1000]['phi_brake_end'], 1)}, {f(ch[3000]['phi_brake_end'], 1)} and
{f(ch[10000]['phi_brake_end'], 1)}&#176; (Drake hydroelastic 91.3&#176;) and the lifted tool&#8217;s lean drops from {f(ch[100]['lift_axis_tilt_deg'], 1)} to
{f(ch[10000]['lift_axis_tilt_deg'], 1)}&#176;. MuJoCo-Warp 3.6 accepts any impratio.</p>
<p><b>noslip</b> removes the translational creep and reverses its sign: {sig(ns10['creep_mm_s'] * 1e3)}&#8202;&#181;m/s at 10 iterations, a drift
against the load of about 5&#8202;% of the original speed. It barely touches the pad&#8217;s twist creep, {sig(ns10t['creep_deg_s'])}&#8202;&#176;/s at 10
iterations; its projected Gauss&#8211;Seidel pass converges slowly on the torsion mode shared by ~15 spheres ({ns_txt}). MuJoCo-Warp 3.6 raises
<code>NotImplementedError</code> for any <code>noslip_iterations</code>&#8202;&gt;&#8202;0 (<code>external/mujoco_warp/mujoco_warp/_src/io.py</code>), so it
cannot reach the RL pipeline.</p>
<p><b>A longer relaxation time</b> lowers creep as 1/t<sub>r</sub> but makes the pinch torque depend on spin rate, as Drake&#8217;s relaxation time
does: with t<sub>r</sub>&#8202;0.05&#8202;s the brake swings through to {f(tr05['phi_end_deg'], 1)}&#176; at 1&#8202;ms and stops at
{f(tr05b['phi_end_deg'], 1)}&#176; at 5&#8202;ms, and passes 80&#176; at {f(tr05['N_at_80_N'], 2)}&#8202;N instead of {f(base5['N_at_80_N'], 2)}&#8202;N; with
0.1&#8202;s the tool leaves the pinch at the 5&#8202;ms step.</p>
<p><b>Heavier objects.</b> Mass enters the creep only through the load. With the stiffness calibrated per object at load (the 10-05 pad
calibration), a heavier tool held at the same pad force creeps faster in proportion to the extra tangential load until it slips. Held at the same
fraction of its slip load, it creeps as roughly the square root of the load, because more spheres engage: from 0.5 to 3&#8202;N the pads go from
{sig(lo1['creep_mm_s'] * 1e3)} to {sig(hi1['creep_mm_s'] * 1e3)}&#8202;&#181;m/s, {f(hi1['creep_mm_s'] / lo1['creep_mm_s'], 2)} times for 6 times the load.
At the bed&#8217;s impratio a 10&#8202;s hold at half the slip load moves the tool {f(base1['creep_mm_s'] * 10, 2)}&#8202;mm and turns it
{f(base2['creep_deg_s'] * 10, 1)}&#176;; at impratio 1000 these become {f(ir1e3_1['creep_mm_s'] * 10, 3)}&#8202;mm and {f(ir1e3_2['creep_deg_s'] * 10, 1)}&#176;.</p>
<p>On the deployed three-finger turn (Table&#160;7) impratio 100 agrees with Drake best; each step toward Drake&#8217;s creep loses held seeds on
two of eight hands. Keep impratio 100 for manipulation and raise it only for long static holds, where 10&#8202;000 reproduces Drake&#8217;s creep at
+{f(100 * (chain_cost(ch[10000]) / chain_cost(ch[100]) - 1), 0)}&#8202;% physics time on the hand.</p>"""]
    return "".join(out)


# ------------------------------------------------------------------------------------------ printed fingertip

TIPS = [("sphere", "10.55&#8202;mm sphere", "mj_pads1", "drake_hydro"),
        ("6", "TPU block, r<sub>f</sub> 6&#8202;mm", "mj_tpu6_pads1", "drake_tpu6"),
        ("2.7", "TPU block, r<sub>f</sub> 2.7&#8202;mm (CAD)", "mj_tpu2p7_pads1", "drake_tpu2p7"),
        ("0", "sharp box 14.8&#8202;mm", "mj_tpu0_pads1", "drake_tpu0")]
TIP_COL = {"sphere": "var(--ink3)", "6": "var(--s1)", "2.7": "var(--s2)", "0": "var(--s3)"}


def svg_tip_shapes():
    """End-on (x-y) and side (x-z) sections of the four tips in the tip-body frame, with the tool's circle on the palmar face."""
    W, H = 980, 330
    out = _svg_open(W, H, "Sections of the four fingertip models at the same scale: the sphere of every earlier contact study, the "
                          "printed TPU block with 6 mm and 2.7 mm fillets, and the sharp box.")
    s = 7.0          # px per mm
    shapes = [("sphere", "10.55&#8202;mm sphere"), ("6", "TPU, r<tspan baseline-shift='sub' font-size='9'>f</tspan> 6&#8202;mm"),
              ("2.7", "TPU, r<tspan baseline-shift='sub' font-size='9'>f</tspan> 2.7&#8202;mm"), ("0", "sharp box")]
    for i, (key, lab) in enumerate(shapes):
        cx, cy = 125 + i * 240, 165
        col = TIP_COL[key]
        # end-on section (looking along the finger): x right (palmar), y up (PIP axis)
        if key == "sphere":
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{10.55 * s:.1f}" style="fill:color-mix(in srgb,{col} 18%,transparent);stroke:{col};stroke-width:2"/>')
        else:
            r = float(key)
            x0, x1 = -6.45, 10.55
            y0, y1 = -7.4, 7.4
            rr = r * s
            # front (palmar, +x) edges rounded, back square
            d = (f"M{cx + x0 * s:.1f},{cy - y1 * s:.1f} L{cx + (x1 - r) * s:.1f},{cy - y1 * s:.1f} "
                 f"A{rr:.1f},{rr:.1f} 0 0 1 {cx + x1 * s:.1f},{cy - (y1 - r) * s:.1f} L{cx + x1 * s:.1f},{cy - (y0 + r) * s:.1f} "
                 f"A{rr:.1f},{rr:.1f} 0 0 1 {cx + (x1 - r) * s:.1f},{cy - y0 * s:.1f} L{cx + x0 * s:.1f},{cy - y0 * s:.1f} Z") if r > 0 else \
                (f"M{cx + x0 * s:.1f},{cy - y1 * s:.1f} L{cx + x1 * s:.1f},{cy - y1 * s:.1f} L{cx + x1 * s:.1f},{cy - y0 * s:.1f} "
                 f"L{cx + x0 * s:.1f},{cy - y0 * s:.1f} Z")
            out.append(f'<path d="{d}" style="fill:color-mix(in srgb,{col} 18%,transparent);stroke:{col};stroke-width:2"/>')
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{10.55 * s:.1f}" style="fill:none;stroke:var(--ink3);stroke-width:1;stroke-dasharray:3 4"/>')
            flat = max(0.0, 14.8 - 2 * r)
            out.append(f'<line x1="{cx + 10.55 * s + 5:.1f}" x2="{cx + 10.55 * s + 5:.1f}" y1="{cy - flat / 2 * s:.1f}" y2="{cy + flat / 2 * s:.1f}" '
                       f'style="stroke:var(--ink);stroke-width:3"/>')
            out.append(f'<text x="{cx + 10.55 * s + 10:.1f}" y="{cy + 4:.1f}" style="fill:var(--ink2);font-size:11px">{flat:.1f}</text>')
        # tool cross-section is a line contact here: draw the tool's axis direction (along y) as a band at the palmar face
        out.append(f'<rect x="{cx + 10.55 * s:.1f}" y="{cy - 9 * s:.1f}" width="6" height="{18 * s:.1f}" '
                   f'style="fill:color-mix(in srgb,var(--ink3) 30%,transparent)"/>')
        out.append(f'<line x1="{cx:.1f}" x2="{cx + 4:.1f}" y1="{cy:.1f}" y2="{cy:.1f}" style="stroke:var(--ink3)"/>')
        out.append(f'<text x="{cx:.1f}" y="{cy - 10.55 * s - 18:.1f}" text-anchor="middle" style="fill:var(--ink);font-weight:500">{lab}</text>')
    out.append("</svg>")
    return "".join(out)


def tip_rows():
    st = jl(os.path.join(D, "tip_static.jsonl"))
    T1 = jl(os.path.join(D, "tip_T1.jsonl")) + [dict(r, model={"mj:spheres:s1:rs0.75:tr0.02": "mj_pads1",
                                                                "drake:hydro:rt0.01": "drake_hydro"}.get(r.get("chain_spec")))
                                                for r in jl(os.path.join(BED, "pull_slip.jsonl"))]
    T2 = jl(os.path.join(D, "tip_T2.jsonl")) + jl(os.path.join(BED, "twist_slip.jsonl"))
    T5 = jl(os.path.join(D, "tip_T5.jsonl")) + jl(os.path.join(BED, "brake.jsonl"))
    return st, T1, T2, T5


def svg_arm(T2):
    W, H = 980, 380
    out = _svg_open(W, H, "Friction arm at spin onset against pad force for four fingertips, MuJoCo sphere pads and Drake hydroelastic.")
    fx, fy = _panel(out, 90, 40, 600, 280, (0.4, 3.6), (0.6, 5.0), (0.5, 1, 2, 3), (0.7, 1, 1.5, 2, 3, 4, 5),
                    "pad force N (N), log", "friction arm at spin onset r&#772; (mm), log", True, True)
    leg = []
    for key, lab, mj, dk in TIPS:
        col = TIP_COL[key]
        for model, dashed, shape, hollow in ((dk, False, None, False), (mj, True, "circle", False)):
            pts = sorted((r["N"], r["rbar_onset_mm"]) for r in T2 if r.get("model") == model and eq(r.get("dt_ms"), 1.0)
                         and r.get("rbar_onset_mm") and r.get("status", "complete") == "complete" and 0.5 < r["rbar_onset_mm"] < 5)
            if not pts:
                continue
            _path(out, fx, fy, pts, col, dashed=dashed, width=2.0 if not dashed else 1.4)
            for N, a in pts:
                _marker(out, fx(N), fy(a), col, "diamond" if model == dk else "circle", hollow=(model == dk),
                        title=f"{lab}, {'Drake' if model == dk else 'MuJoCo pads'}: {a:.3f} mm at {N:g} N")
        leg.append((lab, col, False, None))
    out.append("</svg>")
    return "".join(out) + _legend(leg + [("Drake hydroelastic (hollow diamonds, solid line)", "var(--ink2)", False, "diamond"),
                                         ("MuJoCo 1 mm pads (dots, dashed)", "var(--ink2)", True, "circle")])


def tip_section():
    st, T1, T2, T5 = tip_rows()
    if not st:
        return P.pending("printed-tip rows missing")
    n_pads = {}
    for r in st:
        if r.get("info", {}).get("n_spheres"):
            n_pads[r["model"]] = r["info"]["n_spheres"]
    rows = []
    for key, lab, mj, dk in TIPS:
        cells = [lab]
        for N in (0.5, 3.0):
            a = one(st, model=mj, N=N)
            b = one(st, model=dk, N=N)
            if a and b and a.get("L") and b.get("L"):
                cells += [f"{f(a['L'].get('arm_mm'), 2)} / {f(b['L'].get('arm_mm'), 2)}",
                          f"{f(a['L'].get('area_mm2'), 1)} / {f(b['L'].get('area_mm2'), 1)}"]
            else:
                cells += ["&#8211;", "&#8211;"]
        a = one(st, model=mj, N=1.0)
        cells += [str(n_pads.get(mj, "&#8211;")), f"{f(a.get('us_per_step_median'), 0) if a else '&#8211;'}"]
        rows.append(cells)
    t_static = table(["fingertip", "static arm 0.5&#8202;N, mm (MuJoCo / Drake)", "patch area 0.5&#8202;N, mm&#178;",
                      "static arm 3&#8202;N, mm", "patch area 3&#8202;N, mm&#178;", "pad spheres", "&#181;s/step"], rows)

    def g2(model, N, k="rbar_onset_mm", dt=1.0):
        r = next((x for x in T2 if x.get("model") == model and eq(x.get("N"), N) and eq(x.get("dt_ms"), dt)), None)
        return r.get(k) if r else None

    def g5(model, k, dt=1.0):
        r = next((x for x in T5 if x.get("model") == model and eq(x.get("dt_ms"), dt)), None)
        return r.get(k) if r and r.get("status", "complete") == "complete" else None

    def g1(model, N, k, dt=1.0):
        r = next((x for x in T1 if x.get("model") == model and eq(x.get("N"), N) and eq(x.get("dt_ms"), dt)), None)
        return r.get(k) if r else None
    rows2 = []
    for key, lab, mj, dk in TIPS:
        for model, sim in ((mj, "MuJoCo pads"), (mj + "_ir1e4" if key != "sphere" else None, "MuJoCo pads, impratio 10&#8202;000"),
                           (dk, "Drake")):
            if model is None or g5(model, "phi_end_deg") is None and g2(model, 1.0) is None:
                continue
            rows2.append([f"{lab}, {sim}", f(g2(model, 1.0), 3), f(g2(model, 3.0), 3),
                          f"{f(g5(model, 'phi_end_deg'), 1)} / {f(g5(model, 'phi_end_deg', 5.0), 1)}",
                          f(g5(model, "N_at_45_N"), 3), f(g1(model, 1.0, "mu_eff"), 3),
                          sig((g1(model, 1.0, "creep_mm_s") or 0) * 1e3, 2), sig(g2(model, 1.0, "creep_deg_s"), 2)])
    t_tasks = table(["fingertip, model", "r&#772; onset 1&#8202;N mm", "r&#772; onset 3&#8202;N mm", "brake end &#176; (1 / 5&#8202;ms)",
                     "N at 45&#176;", "&#956;<sub>eff</sub> 1&#8202;N", "creep &#181;m/s", "twist creep &#176;/s"], rows2)
    a6, d6 = g2("mj_tpu6_pads1", 1.0), g2("drake_tpu6", 1.0)
    a27, d27 = g2("mj_tpu2p7_pads1", 1.0), g2("drake_tpu2p7", 1.0)
    a6h, d6h = g2("mj_tpu6_pads1", 3.0), g2("drake_tpu6", 3.0)
    a27h, d27h = g2("mj_tpu2p7_pads1", 3.0), g2("drake_tpu2p7", 3.0)
    sph = one(st, model="drake_hydro", N=0.5)
    r6 = one(st, model="drake_tpu6", N=0.5)
    r27 = one(st, model="drake_tpu2p7", N=0.5)
    r0 = one(st, model="drake_tpu0", N=0.5)
    mc = [x for x in T2 if x.get("model") == "mj_tpu6_meshmc" and eq(x.get("N"), 3.0)]
    mc1 = next((x for x in mc if eq(x["dt_ms"], 1.0)), {})
    mc5 = next((x for x in mc if eq(x["dt_ms"], 5.0)), {})
    return f"""<p>Every contact study from 2026-10-01 to 10-05 modelled the SR2 fingertip as the 10.55&#8202;mm sphere at the end of
<code>assets/mjcf/real_v1/real_hand.xml</code>, and the deployed plans were exported against a sharp box 21.1&#8202;mm wide and 14.8&#8202;mm long
(<code>scene_mutate.set_finger_flat_pads</code>). The printed tip is neither. The CAD of 2026-08-27 (<code>~/Downloads/fingertip (3).stl</code>, the TPU
insert on <code>fingertip_mount (2).stl</code>) is a block 17.0&#8202;mm deep, 14.8&#8202;mm wide along the PIP axis and 21.4&#8211;22.0&#8202;mm long; its palmar
face sits 10.55&#8202;mm from the finger axis, flush with the link capsule, its end 37.16&#8202;mm from the PIP axis, and its front and end edges are
rounded with r<sub>f</sub>&#8202;=&#8202;2.7&#8211;3.0&#8202;mm while the back, on the mount&#8217;s 3&#8202;mm plate, is square. On 2026-10-04 the fillet of the
printed tips was given as 6&#8202;mm, so both radii are modelled. <code>scripts/fingertip_geometry.py</code> builds the block for MuJoCo (a convex mesh, or 1&#8202;mm
sphere pads on every face but the back and the mount side, {n_pads.get('mj_tpu6_pads1', '&#8211;')} spheres per tip at r<sub>f</sub>&#8202;6, foundation depth
8.5&#8202;mm) and Drake (a compliant convex, E&#8202;10&#8202;MPa), and <code>scripts/tpu_tip_rig.py</code> puts it on the bed&#8217;s two-pad rig
with the PIP axis along the tool&#8217;s axis, as in the bench grasp.</p>
{figure(svg_tip_shapes(), "The four fingertip models, end-on, at 7&#8202;px per mm: palmar face to the right, the grey band the tool&#8217;s surface "
                       "with its axis vertical (along the PIP axis), dashed the sphere for scale, the black bar and number the flat width across the "
                       "face in mm. The fillet sets how much of the 14.8&#8202;mm face is flat across the tool&#8217;s axis, and so the length of the contact "
                       "strip a cylinder makes on it.")}
<p>The fillet sets the pinch&#8217;s friction arm. In Drake at 0.5&#8202;N the contact strip is {f(r6['L']['extent_y_mm'], 1) if r6 else '&#8211;'}&#8202;mm long for
r<sub>f</sub>&#8202;6, {f(r27['L']['extent_y_mm'], 1) if r27 else '&#8211;'} for 2.7 and {f(r0['L']['extent_y_mm'], 1) if r0 else '&#8211;'} for the sharp box, against a
{f(sph['L']['extent_y_mm'], 1) if sph else '&#8211;'}&#8202;mm disc for the sphere; the static arm is {f(r6['L']['arm_mm'], 2) if r6 else '&#8211;'},
{f(r27['L']['arm_mm'], 2) if r27 else '&#8211;'} and {f(r0['L']['arm_mm'], 2) if r0 else '&#8211;'}&#8202;mm against the sphere&#8217;s {f(sph['L']['arm_mm'], 2) if sph else '&#8211;'}. The
sphere pads reproduce the twist onset of the filleted blocks: {f(a6, 3)} against {f(d6, 3)}&#8202;mm at 1&#8202;N and {f(a6h, 3)} against {f(d6h, 3)} at
3&#8202;N for r<sub>f</sub>&#8202;6, {f(a27, 3)} against {f(d27, 3)} and {f(a27h, 3)} against {f(d27h, 3)} for 2.7 (Table&#160;3, Figure&#160;{FIG[0] + 1}). On the sharp box the
pads put extra spheres along both side edges and overstate the arm by about 8&#8202;%. A single convex mesh gives MuJoCo one contact point and no
torsion; with <code>multiccd</code> it gives five points along the line, and its arm falls with force and changes with step
({f(mc1.get('rbar_onset_mm'), 2)}&#8202;mm at 1&#8202;ms, {f(mc5.get('rbar_onset_mm'), 2)}&#8202;mm at 5&#8202;ms, 3&#8202;N), so it is not a usable torsion model.</p>
{t_static}
<p class="note">Table&#160;2. Static contact on the rig with the tool pressed at the pad force and no other load: force-weighted arm about the pinch normal and patch area (MuJoCo: spheres in contact times their area share; Drake: hydroelastic surface area), MuJoCo 1&#8202;mm pads / Drake hydroelastic. Rows: <code>tip_static.jsonl</code>.</p>
{figure(svg_arm(T2), "Friction arm at spin onset (bed task&#160;2, 1&#8202;ms). The 10.55&#8202;mm sphere grows as N<sup>1/4</sup>; the filleted blocks start "
                     "1.4&#8211;3 times higher and grow less, because the strip length is set by the flat. Rows: <code>tip_T2.jsonl</code>, <code>twist_slip.jsonl</code>.")}
<p>The brake (Table&#160;3) separates the tips. With the sphere the tool swings to 87&#8211;88&#176;. With the 6&#8202;mm block it reaches 45&#176; at the same force within
3&#8202;% (MuJoCo {f(g5('mj_tpu6_pads1', 'N_at_45_N'), 3)}&#8202;N, Drake {f(g5('drake_tpu6', 'N_at_45_N'), 3)}&#8202;N) and stalls at
{f(g5('mj_tpu6_pads1', 'phi_end_deg'), 1)}&#176; in MuJoCo and {f(g5('drake_tpu6', 'phi_end_deg'), 1)}&#176; in Drake at 0.2&#8202;N: as the tool swings, its
contact line turns from across the 2.8&#8202;mm flat to along the 10&#8202;mm flat, the arm grows and the remaining gravity torque cannot overcome it.
The 2.7&#8202;mm block swings at a third of the force ({f(g5('mj_tpu2p7_pads1', 'N_at_45_N'), 2)}&#8211;{f(g5('drake_tpu2p7', 'N_at_45_N'), 2)}&#8202;N at 45&#176;) and
slides 2&#8211;8&#8202;mm along its axis, more at 5&#8202;ms in both simulators. impratio 10&#8202;000 does not change the brake on either block.</p>
{t_tasks}
<p class="note">Table&#160;3. The bed&#8217;s twist, brake and pull tasks on each fingertip (PROTOCOL.md tasks 1, 2 and 5). Rows: <code>tip_T1.jsonl</code>, <code>tip_T2.jsonl</code>, <code>tip_T5.jsonl</code>; sphere rows from the 2026-10-05 bed.</p>"""


# ------------------------------------------------------------------------------------------ the three-finger turn

COND = [  # key, label, family, group
    ("mujoco|legacy|pt|fast|scene|ir1|muscene", "MuJoCo, legacy box, the export&#8217;s 2&#8202;ms pyramidal solver", "mj", "solver"),
    ("mujoco|legacy|pt|fast|bed|ir100|muscene", "MuJoCo, legacy box", "mj", "core"),
    ("mujoco|sphere|pt|fast|bed|ir100|muscene", "MuJoCo, 10.55&#8202;mm sphere", "mj", "core"),
    ("mujoco|tpu6|pt|fast|bed|ir100|muscene", "MuJoCo, TPU r<sub>f</sub>&#8202;6, one mesh", "mj", "core"),
    ("mujoco|tpu6|pads|fast|bed|ir100|muscene", "MuJoCo, TPU r<sub>f</sub>&#8202;6, 1&#8202;mm pads", "mjpad", "core"),
    ("mujoco|tpu2.7|pt|fast|bed|ir100|muscene", "MuJoCo, TPU r<sub>f</sub>&#8202;2.7, one mesh", "mj", "core"),
    ("mujoco|tpu2.7|pads|fast|bed|ir100|muscene", "MuJoCo, TPU r<sub>f</sub>&#8202;2.7, 1&#8202;mm pads", "mjpad", "core"),
    ("drake|legacy|fast|muscene", "Drake, legacy box", "drake", "core"),
    ("drake|sphere|fast|muscene", "Drake, 10.55&#8202;mm sphere", "drake", "core"),
    ("drake|tpu6|fast|muscene", "Drake, TPU r<sub>f</sub>&#8202;6", "drake", "core"),
    ("drake|tpu2.7|fast|muscene", "Drake, TPU r<sub>f</sub>&#8202;2.7", "drake", "core"),
    ("mujoco|tpu6|pads|fast|bed|ir10000|muscene", "MuJoCo, TPU r<sub>f</sub>&#8202;6 pads, impratio 10&#8202;000", "mjpad", "creep"),
    ("mujoco|tpu2.7|pads|fast|bed|ir10000|muscene", "MuJoCo, TPU r<sub>f</sub>&#8202;2.7 pads, impratio 10&#8202;000", "mjpad", "creep"),
    ("mjwarp|tpu6|pads|fast|bed|ir100|muscene", "MuJoCo-Warp, TPU r<sub>f</sub>&#8202;6 pads, 64 seeds", "gpu", "gpu"),
    ("mjwarp|tpu6|pads|fast|bed|ir10000|muscene", "MuJoCo-Warp, TPU r<sub>f</sub>&#8202;6 pads, impratio 10&#8202;000, 64 seeds", "gpu", "gpu"),
    ("mujoco|legacy|pt|fast|bed|ir100|mu1.0", "MuJoCo, legacy box, &#956;&#8202;1", "mj", "mu"),
    ("mujoco|tpu6|pads|fast|bed|ir100|mu1.0", "MuJoCo, TPU r<sub>f</sub>&#8202;6 pads, &#956;&#8202;1", "mjpad", "mu"),
    ("mujoco|legacy|pt|rigid|bed|ir100|muscene", "MuJoCo, legacy box, servo kp&#8202;30", "mj", "plant"),
    ("mujoco|tpu6|pads|rigid|bed|ir100|muscene", "MuJoCo, TPU r<sub>f</sub>&#8202;6 pads, servo kp&#8202;30", "mjpad", "plant"),
    ("mujoco|legacy|pt|fast_fl|bed|ir100|muscene", "MuJoCo, legacy box, servo frictionloss 0.0035", "mj", "plant"),
    ("mujoco|tpu6|pads|fast_fl|bed|ir100|muscene", "MuJoCo, TPU r<sub>f</sub>&#8202;6 pads, servo frictionloss 0.0035", "mjpad", "plant"),
]
FAM_COL = {"mj": "var(--ink3)", "mjpad": "var(--s1)", "gpu": "var(--c-newton)", "drake": "var(--c-drake)"}
GROUP_LBL = {"core": "Fingertip and simulator (servo kp 0.5, &#956; 2.4, 1&#8202;ms)", "solver": "Solver settings",
             "creep": "Creep setting", "gpu": "GPU", "mu": "Friction", "plant": "Servo plant"}
PAIRS = [("mujoco|legacy|pt|fast|bed|ir100|muscene", "drake|legacy|fast|muscene", "legacy box, MuJoCo point vs Drake"),
         ("mujoco|sphere|pt|fast|bed|ir100|muscene", "drake|sphere|fast|muscene", "sphere, MuJoCo point vs Drake"),
         ("mujoco|tpu6|pads|fast|bed|ir100|muscene", "drake|tpu6|fast|muscene", "TPU r<sub>f</sub> 6, MuJoCo pads vs Drake"),
         ("mujoco|tpu2.7|pads|fast|bed|ir100|muscene", "drake|tpu2.7|fast|muscene", "TPU r<sub>f</sub> 2.7, MuJoCo pads vs Drake"),
         ("mujoco|tpu6|pt|fast|bed|ir100|muscene", "drake|tpu6|fast|muscene", "TPU r<sub>f</sub> 6, MuJoCo one mesh vs Drake"),
         ("mujoco|tpu6|pads|fast|bed|ir100|muscene", "mujoco|tpu6|pads|fast|bed|ir10000|muscene", "TPU r<sub>f</sub> 6 pads, impratio 100 vs 10&#8202;000"),
         ("mujoco|tpu6|pads|fast|bed|ir100|muscene", "mjwarp|tpu6|pads|fast|bed|ir100|muscene", "TPU r<sub>f</sub> 6 pads, MuJoCo CPU vs MuJoCo-Warp"),
         ("mujoco|tpu6|pads|fast|bed|ir100|muscene", "mujoco|tpu2.7|pads|fast|bed|ir100|muscene", "MuJoCo pads, r<sub>f</sub> 6 vs 2.7"),
         ("drake|tpu6|fast|muscene", "drake|tpu2.7|fast|muscene", "Drake, r<sub>f</sub> 6 vs 2.7"),
         ("drake|legacy|fast|muscene", "drake|tpu6|fast|muscene", "Drake, legacy box vs TPU r<sub>f</sub> 6"),
         ("mujoco|legacy|pt|fast|scene|ir1|muscene", "mujoco|legacy|pt|fast|bed|ir100|muscene", "MuJoCo legacy box, export solver vs 1&#8202;ms elliptic")]


def by_case(rows):
    out = {}
    for r in rows:
        if r.get("status") in ("complete", "ejected") and r.get("seed") is not None:
            out[(RS.cond_key(r), r["hand"], r["seed"])] = r
    return out


def paired(B, ca, cb):
    same, n, dturn = 0, 0, []
    for (c, h, s), ra in B.items():
        if c != ca:
            continue
        rb = B.get((cb, h, s))
        if rb is None or ra.get("status") != "complete" or rb.get("status") != "complete":
            continue
        n += 1
        if bool(ra.get("held_hold")) == bool(rb.get("held_hold")):
            same += 1
        if ra.get("held_hold") and rb.get("held_hold"):
            dturn.append(abs(ra["turn_hold_deg"] - rb["turn_hold_deg"]))
    return n, same, (statistics.median(dturn) if dturn else None), len(dturn)


def hf_class(fr):
    return "hf" + str(min(4, int(fr * 5 - 1e-9)) if fr > 0 else 0)


def svg_strips(S, bench, key, title, xs, xt, xlab, bench_key):
    """One row per hand: each core condition as a dot on the x scale, the bench as a black diamond."""
    W, H = 980, 70 + 8 * 34 + 40
    out = _svg_open(W, H, title)
    x0, w = 120, 800
    fx = lambda v: x0 + (v - xs[0]) / (xs[1] - xs[0]) * w  # noqa: E731
    for v in xt:
        out.append(f'<line x1="{fx(v):.1f}" x2="{fx(v):.1f}" y1="40" y2="{40 + 8 * 34}" style="stroke:var(--rule2)"/>'
                   f'<text x="{fx(v):.1f}" y="{40 + 8 * 34 + 18}" text-anchor="middle" style="fill:var(--ink3)">{v:g}</text>')
    out.append(f'<text x="{x0 + w / 2}" y="{40 + 8 * 34 + 38}" text-anchor="middle" style="fill:var(--ink2)">{xlab}</text>')
    for i, h in enumerate(HANDS):
        y = 56 + i * 34
        out.append(f'<text x="{x0 - 14}" y="{y + 4}" text-anchor="end" style="fill:var(--ink);font-weight:500">{h}</text>'
                   f'<line x1="{x0}" x2="{x0 + w}" y1="{y}" y2="{y}" style="stroke:var(--rule)"/>')
        k = 0
        for c, lab, fam, grp in COND:
            if grp not in ("core", "gpu"):
                continue
            s = S.get((c, h))
            if not s:
                continue
            v = s[key]
            if v is None:
                continue
            jit = ((k % 5) - 2) * 3.0
            k += 1
            _marker(out, fx(max(xs[0], min(xs[1], v))), y + jit, FAM_COL[fam],
                    {"mj": "circle", "mjpad": "circle", "gpu": "square", "drake": "diamond"}[fam],
                    hollow=(fam == "mj"), r=4.2, title=f"{h}, {lab}: {v:.2f}")
        bv = bench[h][bench_key]
        out.append(f'<path d="M{fx(bv):.1f},{y - 9} L{fx(bv) + 6:.1f},{y} L{fx(bv):.1f},{y + 9} L{fx(bv) - 6:.1f},{y} Z" '
                   f'style="fill:var(--ink);stroke:var(--card);stroke-width:1.5"><title>{h} bench: {bv:.2f}</title></path>')
    out.append("</svg>")
    return "".join(out) + _legend([("MuJoCo, single geom (sphere, legacy box, TPU mesh)", FAM_COL["mj"], False, "circle"),
                                   ("MuJoCo 1&#8202;mm pads on the TPU block", FAM_COL["mjpad"], False, "circle"),
                                   ("MuJoCo-Warp pads, 64 seeds", FAM_COL["gpu"], False, "square"),
                                   ("Drake hydroelastic", FAM_COL["drake"], False, "diamond"),
                                   ("bench (hardware)", "var(--ink)", False, "diamond")])


def turn_section():
    rows = RS.load_rows()
    if not rows:
        return P.pending("reorientation rows missing")
    S = RS.summarize(rows)
    B = RS.bench()
    BC = by_case(rows)
    for (c, h), s in S.items():
        s["held_frac"] = s["held_hold"] / s["n"] if s["n"] else None
    bench = {h: {"hold": B[h]["n_held"] / B[h]["n"], "turn": B[h]["held_deg"]} for h in HANDS}
    # matrix table
    head = ["condition"] + HANDS + ["&#961; held vs bench"]
    trs = []
    grp_done = set()
    present = [c for c in COND if any((c[0], h) in S for h in HANDS)]
    for c, lab, fam, grp in present:
        if grp not in grp_done:
            trs.append(f"<tr class='grp'><td colspan='{len(head)}'>{GROUP_LBL[grp]}</td></tr>")
            grp_done.add(grp)
        cells = [f"<td class='lab'>{lab}</td>"]
        xs, ys = [], []
        for h in HANDS:
            s = S.get((c, h))
            if not s:
                cells.append("<td class='num'>&#8211;</td>")
                continue
            fr = s["held_frac"]
            t = s["turn_held_mean"]
            cells.append(f"<td class='num hf {hf_class(fr)}'>{s['held_hold']}/{s['n']}<br>"
                         f"{(('%+.0f' % t) + '&#176;') if t is not None else '&#8211;'}</td>")
            xs.append(fr)
            ys.append(bench[h]["hold"])
        rho = spearman(xs, ys) if len(xs) == 8 else None
        cells.append(f"<td class='num'>{f(rho, 2) if rho is not None else '&#8211;'}</td>")
        trs.append("<tr>" + "".join(cells) + "</tr>")
    trs.append("<tr class='bench'><td class='lab'>bench, hardware (held trials; net turn on held trials)</td>" +
               "".join(f"<td class='num'>{B[h]['n_held']}/{B[h]['n']}<br>{B[h]['held_deg']:+.0f}&#176;</td>" for h in HANDS) + "<td></td></tr>")
    matrix = ("<div class='tw'><table class='matrix'><thead><tr>" + "".join(
        f"<th{' class=num' if i else ''}>{x}</th>" for i, x in enumerate(head)) + "</tr></thead><tbody>" + "".join(trs) +
              "</tbody></table></div>")
    # scatter across core conditions vs seeds
    core = [c for c, _, _, g in COND if g == "core"]
    spread = []
    for h in HANDS:
        fr = [S[(c, h)]["held_frac"] for c in core if (c, h) in S]
        tm = [S[(c, h)]["turn_held_mean"] for c in core if (c, h) in S and S[(c, h)]["turn_held_mean"] is not None]
        tsd = [S[(c, h)]["turn_held_sd"] for c in core if (c, h) in S and S[(c, h)]["turn_held_sd"] is not None
               and S[(c, h)]["held_hold"] >= 3]
        if len(fr) >= 4:
            spread.append((h, min(fr), max(fr), statistics.pstdev(fr), statistics.pstdev(tm) if len(tm) >= 2 else None,
                           statistics.mean(tsd) if tsd else None, len(fr)))
    sp_rows = [[h, f"{lo:.2f}&#8211;{hi:.2f}", f(sd, 2), f(tsd_c, 1), f(tsd_s, 1), str(n)] for h, lo, hi, sd, tsd_c, tsd_s, n in spread]
    t_spread = table(["hand", "held fraction, range over conditions", "sd over conditions", "turn sd over conditions &#176;",
                      "turn sd over seeds &#176; (mean)", "conditions"], sp_rows)
    pr = []
    for ca, cb, lab in PAIRS:
        n, same, md, nb = paired(BC, ca, cb)
        if n:
            pr.append([lab, str(n), f"{same / n:.2f}", f(md, 1), str(nb)])
    t_pairs = table(["pair (same hands, same seeds)", "pairs", "same held outcome", "median |&#916;turn| &#176;", "held in both"], pr)
    # numbers for the prose
    gl = lambda c, h: S.get((c, h), {})  # noqa: E731
    n_rows = len([r for r in rows if r.get("status") in ("complete", "ejected")])
    sims = sorted({r["sim"] for r in rows})
    films = "".join(film(f"media/{h}_s0_models.mp4",
                         f"{h}, seed 0, in nine contact configurations (MuJoCo panels draw pad spheres touching the tool in red; "
                         f"Drake panels show Drake&#8217;s state drawn in the MuJoCo scene of the same tip).", f"media/{h}_s0_models.jpg")
                    for h in ("D7", "D2", "D5"))
    sd_c = [x[3] for x in spread]
    sp_txt = (f"Over the ten fingertip-and-simulator conditions, a hand&#8217;s held fraction ranges by {f(min(x[2] - x[1] for x in spread), 2)}"
              f"&#8211;{f(max(x[2] - x[1] for x in spread), 2)} (sd {f(min(sd_c), 2)}&#8211;{f(max(sd_c), 2)}); nine seeds resolve a fraction to a "
              f"binomial sd of at most 0.17.") if spread else ""
    return f"""<p>The deployed reorientation is the open-loop three-finger turn of the SR2 paper: the hand closes on the screwdriver lying on a
post with its palm fixed, and a joint-space trajectory moves all three fingertips so that the tool turns toward vertical about the pinch axis.
The bench ran it {sum(B[h]['n'] for h in HANDS)} times on the eight deployed hands (<code>paper/figures/ranking.json</code>). Here each hand&#8217;s plan
replays unchanged ({n_rows} rollouts in MuJoCo, MuJoCo-Warp and Drake): 0.8&#8202;s grip, the plan&#8217;s 4&#8202;s trajectory (1.3&#8202;s more grip, a 1.1&#8202;s turn, hold),
1.5&#8202;s hold, then 4&#8202;s more; the finger servos are the calibrated position servos (kp&#8202;0.5, kv&#8202;0.02, 0.35&#8202;N&#8202;m, joint damping 0.5, measured
masses), friction is the scenes&#8217; 2.4, the step 1&#8202;ms, and nine tool placements per hand are the same seeds in every simulator
(<code>scripts/reorient_backends.py</code>; MuJoCo-Warp runs 64 seeds in one batch). The harness reproduces the 2026-09-16 replay where that
replay&#8217;s outcome is stable (D7 21.9&#8202;&#177;&#8202;4.1&#176; against 21.6&#8202;&#177;&#8202;3.8&#176;, D2, D4, D5 within 3&#176;).</p>
<p><b>Every bench scene starts the tool inside its post.</b> <code>scene_mutate.set_object_platform</code> puts the tool&#8217;s centre and the
post&#8217;s top at the same height, so the 12.5&#8202;mm tool begins 12.5&#8202;mm deep in the post (contact distance &#8722;12.5&#8202;mm in the compiled scene) and
is thrown out in the first 20&#8202;ms. Every replay of a bench plan since 2026-08-29 carries this start, including the plant-correction gate,
Table&#160;IV&#8217;s simulated column and the 2026-09-16 turn studies; the 17&#176; of tilt those replays found &#8220;already in the grip&#8221; is mostly the ejection.
Here the post is shortened so the tool rests on it at the plan&#8217;s pose ({n_rows} rows in <code>reorient_*.jsonl</code>; the rows with the
original post are kept in <code>superseded_tool_in_post/</code>). With the tool resting, D7&#8217;s plan holds the tool on every seed in MuJoCo and Drake
with the legacy box and turns it {f(gl('mujoco|legacy|pt|fast|bed|ir100|muscene', 'D7').get('turn_held_mean'), 0)}&#8211;{f(gl('mujoco|legacy|pt|fast|scene|ir1|muscene', 'D7').get('turn_held_mean'), 0)}&#176;, where the bench turned it 33&#176;.</p>
{plant_paragraph(S)}
{matrix}
<p class="note">Table&#160;4. Held seeds and mean turn on held seeds, per hand and condition. Cell shade: held fraction (red 0, green all).
&#961;: Spearman rank correlation of the held fraction with the bench&#8217;s hold rate over the eight hands.</p>
{figure(svg_strips(S, bench, 'held_frac', 'Held fraction per hand in each simulator and fingertip, and the bench hold rate.', (0, 1), (0, 0.25, 0.5, 0.75, 1), 'fraction of seeds holding the tool at the end of the hold', 'hold'),
        "Held fraction per hand: every dot is one fingertip-and-simulator condition of Table&#160;4&#8217;s first group or a MuJoCo-Warp batch; the black diamond is the bench.")}
<p>{sp_txt} Table&#160;5 sets the spread across conditions against the spread across seeds, and Table&#160;6 pairs conditions seed by seed.</p>
{t_spread}
<p class="note">Table&#160;5. Spread of the outcome across the ten fingertip-and-simulator conditions, and the mean spread of the turn across seeds within one condition (conditions with at least three held seeds).</p>
{t_pairs}
<p class="note">Table&#160;6. Paired agreement on identical hands and tool placements.</p>
{ir_block(S)}
{films}"""


def plant_paragraph(S):
    def hc(c):
        xs = [S[(c, h)] for h in HANDS if (c, h) in S]
        return sum(x["held_hold"] for x in xs), sum(x["n"] for x in xs), [x["turn_held_mean"] for x in xs if x["turn_held_mean"] is not None]
    rk, rn, rt = hc("mujoco|legacy|pt|rigid|bed|ir100|muscene")
    fk, fn, ft = hc("mujoco|legacy|pt|fast|bed|ir100|muscene")
    uk, un, ut = hc("mujoco|legacy|pt|fast|bed|ir100|mu1.0")
    pk, pn, pt = hc("mujoco|tpu6|pads|rigid|bed|ir100|muscene")
    if not rn:
        return ""
    big = sorted(t for t in rt if t is not None)
    return (f"<p>Three things move the outcome more than the choice of simulator. The <b>fingertip</b>: the sphere drops the tool on most hands in "
            f"both simulators, the TPU block holds it on most. The <b>servo</b>: with the shipped gain (kp&#8202;30) the legacy box holds {rk} of {rn} seeds "
            f"and turns the tool {f(big[1] if len(big) > 1 else big[0], 0)}&#8211;{f(big[-1], 0)}&#176; on all but one hand, the bench&#8217;s range "
            f"(the TPU pads: {pk} of {pn}), where the calibrated kp&#8202;0.5 holds {fk} of {fn} and turns at most {f(max(ft), 0)}&#176;. The 2026-09-02 "
            f"calibration that chose kp&#8202;0.5, and its check against the bench (retention 0.94 against 0.92, cosine 0.638 against 0.593 on D7), "
            f"ran on scenes that start the tool inside its post, so both have to be repeated. "
            f"<b>Friction</b>: &#956;&#8202;1 instead of 2.4 takes the legacy box from {fk} to {uk} held seeds of {un}. No condition reproduces the bench "
            f"hand by hand; the TPU models rank the hands against the bench (&#961; down to &#8722;0.8), because D2, which the bench holds 10/10, drops "
            f"in every condition at kp&#8202;0.5 while D5 and D6, held 2&#8211;3 of 10 on the bench, hold in most conditions.</p>")


def ir_block(S):
    """Held fraction of the r_f 6 pads per hand against impratio, CPU (9 seeds) and GPU (64 seeds), with Drake."""
    keys = [("mujoco|tpu6|pads|fast|bed|ir100|muscene", "MuJoCo, impratio 100"),
            ("mujoco|tpu6|pads|fast|bed|ir300|muscene", "MuJoCo, impratio 300"),
            ("mujoco|tpu6|pads|fast|bed|ir1000|muscene", "MuJoCo, impratio 1000"),
            ("mujoco|tpu6|pads|fast|bed|ir10000|muscene", "MuJoCo, impratio 10&#8202;000"),
            ("mjwarp|tpu6|pads|fast|bed|ir100|muscene", "MuJoCo-Warp, impratio 100"),
            ("mjwarp|tpu6|pads|fast|bed|ir10000|muscene", "MuJoCo-Warp, impratio 10&#8202;000"),
            ("drake|tpu6|fast|muscene", "Drake")]
    rows = []
    for k, lab in keys:
        cells = [lab]
        if not any((k, h) in S for h in HANDS):
            continue
        for h in HANDS:
            s_ = S.get((k, h))
            cells.append(f"{s_['held_hold']}/{s_['n']}" if s_ else "&#8211;")
        rows.append(cells)
    if not rows:
        return ""
    d3 = {k: S.get((k, "D3")) for k, _ in keys}
    d4 = {k: S.get((k, "D4")) for k, _ in keys}

    def fr(x):
        return f"{x['held_hold']}/{x['n']}" if x else "&#8211;"
    txt = (f"<p>Matching Drake&#8217;s creep does not bring the turn closer to Drake. With the 6&#8202;mm pads at impratio 10&#8202;000, D3 holds "
           f"{fr(d3['mujoco|tpu6|pads|fast|bed|ir10000|muscene'])} and D4 {fr(d4['mujoco|tpu6|pads|fast|bed|ir10000|muscene'])} seeds against "
           f"{fr(d3['mujoco|tpu6|pads|fast|bed|ir100|muscene'])} and {fr(d4['mujoco|tpu6|pads|fast|bed|ir100|muscene'])} at 100 and "
           f"{fr(d3['drake|tpu6|fast|muscene'])} and {fr(d4['drake|tpu6|fast|muscene'])} in Drake; on the GPU the same change takes D3 from "
           f"{fr(d3['mjwarp|tpu6|pads|fast|bed|ir100|muscene'])} to {fr(d3['mjwarp|tpu6|pads|fast|bed|ir10000|muscene'])} and D4 from "
           f"{fr(d4['mjwarp|tpu6|pads|fast|bed|ir100|muscene'])} to {fr(d4['mjwarp|tpu6|pads|fast|bed|ir10000|muscene'])}. The lost seeds hold "
           f"through the grip exactly as at impratio 100 and drop 0.4&#8211;0.8&#8202;s into the turn, the tool landing on its end. The solver "
           f"converges in both (median 1&#8211;4 Newton iterations, at most 17 of the 100 allowed, on D3 seed 0), so the difference is the friction "
           f"law: soft friction lets the pads slide while the fingers move, and at impratio 10&#8202;000 they hold until the cone limit and then let "
           f"go.</p>")
    return txt + table(["pads on TPU r<sub>f</sub> 6"] + HANDS, rows) + \
        "<p class='note'>Table&#160;7. Held seeds of the 6&#8202;mm-fillet pads per hand against impratio.</p>"


# ------------------------------------------------------------------------------------------ design

def design_section():
    rows = RS.load_rows()
    S = RS.summarize(rows) if rows else {}
    split = sum(1 for h in HANDS if ("mujoco|legacy|pt|fast|bed|ir100|muscene", h) in S and ("drake|legacy|fast|muscene", h) in S
                and abs(S[("mujoco|legacy|pt|fast|bed|ir100|muscene", h)]["held_hold"] - S[("drake|legacy|fast|muscene", h)]["held_hold"]) >= 4)
    return """<p>The fingertip sets what the task does: the friction arm of a pinch differs by 1.4&#8211;4.6 times between the sphere, the two fillets and
the sharp box at 0.5&#8202;N, and the deployed turn holds the tool on 7&#8211;12 of 72 seeds with the sphere and 59&#8211;61 with the 6&#8202;mm block. Whether
two simulators agree about the task depends on how that fingertip&#8217;s contact is represented (Table&#160;6). Finger-shape optimisation has a long record
for the first; choosing a fingertip by how consistently different contact models simulate its task has none that we found.</p>
<h3>Prior work</h3>
<ul class="refs">
<li><b>Fingertip and finger form for a task.</b> Fit2Form (Ha, Agrawal and Song, CoRL 2020) generates parallel-jaw finger pairs per object with a learned
fitness; DGDM (Xu, Ha, Song, CoRL 2024) generates manipulator shapes from task interaction profiles with a dynamics-guided diffusion model;
Meixner, Hazard and Pollard (Humanoids 2019) co-optimise hand morphology and control under explicit placement and model uncertainty and report designs
that use every surface of the hand; Ye et al. (arXiv 2511.13710, 2025) add a planar fingertip modification to a dexterous hand and optimise it
with control through a differentiable neural-physics surrogate, 82.5&#8202;% zero-shot precision grasping; the contact-primitive fingertips of
US&#8202;11&#8202;185&#8202;986 match printed soft tips to clusters of local grasp geometry.</li>
<li><b>Curvature for in-hand rolling.</b> Chen, Lu, Colgate and Lynch (arXiv 2607.12105, 2026) compare flat, spherical and two cylindrical tips of
the same envelope on an Allegro hand rolling objects: the cylinder curved along the rolling direction gives 2.29 turns and 86&#8202;% success,
the sphere 1.39 and 86&#8202;%, the flat 1.46 and 69&#8202;%, the cylinder across it 0.78 and 67&#8202;%. A filleted block is a cylinder in one direction and a
flat in the other, so the same orientation question applies to the SR2 tip.</li>
<li><b>Soft-finger contact.</b> Xydas and Kao (IJRR 1999) give the contact radius a&#8202;&#8733;&#8202;N<sup>&#947;</sup> with 0&#8202;&#8804;&#8202;&#947;&#8202;&#8804;&#8202;1/3 and an
elliptical limit surface in force and torsion; the elastic-foundation law used here is &#947;&#8202;=&#8202;1/4. Spiers, Calli and Dollar (RA-L 2018) switch a
finger between gripping and sliding with a surface that recesses above 1.2&#8211;2.5&#8202;N, a mechanical answer to the hold-versus-slide trade the brake
exposes. Babin and Gosselin (IJRR 2018) design fingers and a fingernail-like edge for picking thin objects off flat surfaces, the case a round or
blunt tip handles worst.</li>
<li><b>Simulators as an uncertainty set.</b> Elandt et al. (IROS 2019) give the pressure-field model, Masterjohn et al. (RA-L 2022) its
velocity-level patch approximation and Castro et al. (T-RO 2023) Drake&#8217;s SAP solver; Horak and Trinkle (RA-L 2019) and Le Lidec et al.
(arXiv 2304.06372) compare contact models on grasping and wedging scenes and find qualitatively different outcomes once several contacts act;
PolySim (arXiv 2510.01708, 2025) trains humanoid policies across several physics engines to cover their differences; Kriegman et al. (RoboSoft 2020)
show that the sim-to-real gap of soft robots depends on morphology and select designs that transfer. None of these uses agreement between contact
models as a criterion for the hardware itself.</li>
</ul>
<h3>Formulation</h3>
<p>Let a fingertip be a parameter vector g: fillet radius across the face and along it, face width, a face curvature (flat to a cylinder of
radius R), an optional rigid nail edge at the end face (protrusion and thickness), and the TPU stiffness E. For a task suite T and an ensemble of
contact models M (MuJoCo point, MuJoCo sphere pads, Drake hydroelastic, Newton hydroelastic), each run on the same seeds, score</p>
<p class="eqline">J(g) = mean<sub>m&#8712;M</sub> success<sub>m</sub>(g) &#8722; &#955;&#8202;&#183;&#8202;disagreement<sub>M</sub>(g) &#8722; &#951;&#8202;&#183;&#8202;cost(g),</p>
<p>with disagreement the fraction of paired seeds on which the models&#8217; outcomes differ (Table&#160;6&#8217;s first column, complemented) and cost the GPU time per
world-step of the pad model. The 2026-09 bench plans held the tool on seven of eight hands in the simulator they were planned in and on
2&#8211;10 of 10 trials on the bench; in Table&#160;4 the held count of that planning model (legacy box, MuJoCo point contact) differs from Drake&#8217;s for
the same plan by four or more seeds of nine on {LEGACY_SPLIT} hands. The quick-release tips of the SR2 hand (paper Sec.&#160;II-B) make each candidate a print and a swap, so the ensemble can be
checked against hardware at every round.</p>
<h3>Tasks</h3>
<ul>
<li>the deployed three-finger turn on D1&#8211;D8 (this page; held fraction and turn);</li>
<li>the pinch brake and swing (bed task&#160;5) and the paper&#8217;s pusher and wield (2026-10-02), where torsion must be low to swing and high to hold;</li>
<li>a thin-object pick: a 20&#8202;mm coin, a 1&#8202;mm card and an M3 washer flat on the table, lifted by a two-finger pinch that has to get an edge
under the object, the task a sphere tip fails by construction;</li>
<li>a held tool under a 2&#8202;g shake (bed task&#160;4), the retention case where creep and torsion both act.</li>
</ul>
<h3>Search</h3>
<p>Each candidate is a mesh from <code>fingertip_geometry.py</code> generalised to these parameters; its static signature on the rig (patch area,
torsion arm against force and against contact-line angle) takes seconds in MuJoCo and Drake and screens out shapes whose two models disagree
statically. The survivors run the task suite on MuJoCo-Warp (64&#8211;512 seeds per design, 20&#8202;&#181;s per world-step with the pads) with Drake on nine
paired seeds for the disagreement term; a cross-entropy or Bayesian search over five or six parameters needs on the order of 100 designs. Three
designs from the front go to the printer and the bench.</p>""".replace("{LEGACY_SPLIT}", str(split))


# ------------------------------------------------------------------------------------------ next

def next_section():
    return """<ol class="open">
<li><b>Confirm the printed fillet.</b> Measure one printed tip with a radius gauge (6&#8202;mm per 2026-10-04, 2.7&#8211;3.0&#8202;mm in the 08-27 CAD). The
two radii give the brake a 3&#215; different swing force on this page; if a third radius is printed, add it with <code>--tips tpu&lt;r&gt;</code>.</li>
<li><b>Fix the post in the scene generator.</b> <code>scene_mutate.set_object_platform</code> should place the tool&#8217;s centre one radius above the
post. Re-run the plant-correction gate (<code>scripts/plant_drop_gate.py</code>) and the eight-hand bench replay
(<code>scripts/real_v1_bench_plants.py</code>) on corrected scenes; if the gate&#8217;s retention match (0.94 against 0.92) does not survive, the plant
calibration of 2026-09-02 rests on the ejection.</li>
<li><b>Bench measurement of the tip&#8217;s torsion.</b> Pinch the screwdriver between two printed tips on the bench at 0.5, 1 and 3&#8202;N and raise a torque
about the pinch axis with a hanging mass on a 50&#8202;mm arm until it turns; the sphere model predicts r&#772;&#8202;0.84&#8211;1.31&#8202;mm, the 6&#8202;mm block 1.13&#8211;1.52,
the 2.7&#8202;mm block about 2.5. This measurement decides which model the brake and the turn should be planned in.</li>
<li><b>Newton on the hand.</b> Port the bench scene to Newton hydroelastic with the TPU mesh (<code>scripts/contact_bed_newton.py</code> has the rig); the
bed found Newton&#8217;s torsion 2.3&#8211;2.9 times the hydroelastic law, so check the rig&#8217;s TPU twist first.</li>
<li><b>First co-design round.</b> Sweep r<sub>f</sub>&#8202;&#8712;&#8202;{0, 1.5, 2.7, 4, 6, 7.4} on the bench turn in MuJoCo pads and Drake (nine seeds each, about
2&#8202;h on four cores) and plot held fraction and paired agreement against r<sub>f</sub>; build the thin-object pick scene and run the sphere, the 6&#8202;mm
block and a block with a 1&#8202;mm nail.</li>
</ol>"""


# ------------------------------------------------------------------------------------------ lede and main

def lede():
    C, pull, twist, brake, nsi, chain = creep_data()
    dp, dtw, db = drake_ref(pull, twist, brake, 1.0, 1.0)
    b1, b2 = cr(C, "T1", "mj_pads1", 0, 100, 1.0), cr(C, "T2", "mj_pads1", 0, 100, 1.0)
    h1, h2 = cr(C, "T1", "mj_pads1", 0, 10000, 1.0), cr(C, "T2", "mj_pads1", 0, 10000, 1.0)
    st, T1, T2, T5 = tip_rows()

    def g2(model, N):
        r = next((x for x in T2 if x.get("model") == model and eq(x.get("N"), N) and eq(x.get("dt_ms"), 1.0)), None)
        return r.get("rbar_onset_mm") if r else None
    errs = [abs(g2(m, N) / g2(d, N) - 1) for m, d in (("mj_tpu6_pads1", "drake_tpu6"), ("mj_tpu2p7_pads1", "drake_tpu2p7"))
            for N in (1.0, 3.0) if g2(m, N) and g2(d, N)]
    rows = RS.load_rows()
    BC = by_case(rows)
    n6, s6, md6, _ = paired(BC, "mujoco|tpu6|pads|fast|bed|ir100|muscene", "drake|tpu6|fast|muscene")
    n27, s27, md27, _ = paired(BC, "mujoco|tpu2.7|pads|fast|bed|ir100|muscene", "drake|tpu2.7|fast|muscene")
    nm, sm_, _, _ = paired(BC, "mujoco|tpu6|pt|fast|bed|ir100|muscene", "drake|tpu6|fast|muscene")
    nl, sl, _, _ = paired(BC, "mujoco|legacy|pt|fast|bed|ir100|muscene", "drake|legacy|fast|muscene")
    S = RS.summarize(rows)
    B = RS.bench()

    def held(c, h):
        x = S.get((c, h))
        return f"{x['held_hold']}/{x['n']}" if x else "&#8211;"
    sph_mj = sum(S[("mujoco|sphere|pt|fast|bed|ir100|muscene", h)]["held_hold"] for h in HANDS if ("mujoco|sphere|pt|fast|bed|ir100|muscene", h) in S)
    sph_dk = sum(S[("drake|sphere|fast|muscene", h)]["held_hold"] for h in HANDS if ("drake|sphere|fast|muscene", h) in S)
    tpu_dk = sum(S[("drake|tpu6|fast|muscene", h)]["held_hold"] for h in HANDS if ("drake|tpu6|fast|muscene", h) in S)
    rig = [S[("mujoco|legacy|pt|rigid|bed|ir100|muscene", h)] for h in HANDS if ("mujoco|legacy|pt|rigid|bed|ir100|muscene", h) in S]
    rig_t = sorted(x["turn_held_mean"] for x in rig if x["turn_held_mean"] is not None)
    rig_k, rig_n = sum(x["held_hold"] for x in rig), sum(x["n"] for x in rig)
    tpu_mj = sum(S[("mujoco|tpu6|pads|fast|bed|ir100|muscene", h)]["held_hold"] for h in HANDS if ("mujoco|tpu6|pads|fast|bed|ir100|muscene", h) in S)
    arm = lambda m, N: (one(st, model=m, N=N) or {}).get("L", {}).get("arm_mm")  # noqa: E731
    ratios = [arm(m, N) / arm("drake_hydro", N) for m in ("drake_tpu6", "drake_tpu2p7") for N in (0.5, 3.0)
              if arm(m, N) and arm("drake_hydro", N)]
    b5, h5 = cr(C, "T5", "mj_pads1", 0, 100), cr(C, "T5", "mj_pads1", 0, 10000)
    return (f"MuJoCo&#8217;s friction creep falls as 1/impratio: at impratio 10&#8202;000 the 1&#8202;mm sphere pads creep "
            f"{sig(h1['creep_mm_s'] * 1e3)}&#8202;&#181;m/s and turn {sig(h2['creep_deg_s'])}&#8202;&#176;/s under half their slip load, against "
            f"{sig(b1['creep_mm_s'] * 1e3)}&#8202;&#181;m/s and {sig(b2['creep_deg_s'])}&#8202;&#176;/s at the bed&#8217;s impratio 100 and "
            f"{sig(dp['creep_mm_s'] * 1e3)}&#8202;&#181;m/s and {sig(dtw['creep_deg_s'])}&#8202;&#176;/s in Drake at 1&#8202;ms, while slip onset, twist onset "
            f"and the brake move by at most {f(100 * abs(h2['rbar_onset_mm'] / b2['rbar_onset_mm'] - 1), 0)}&#8202;% and "
            f"{f(abs(h5['phi_end_deg'] - b5['phi_end_deg']), 0)}&#176;; noslip removes only the translational part and does not run in MuJoCo-Warp. "
            f"The printed SR2 fingertip is a TPU block 14.8&#8202;mm wide with 2.7&#8211;6&#8202;mm fillets; every earlier contact study used a 10.55&#8202;mm sphere. "
            f"Packed with 1&#8202;mm spheres, the block matches Drake&#8217;s hydroelastic torsion onset within {f(100 * max(errs), 0) if errs else '&#8211;'}&#8202;%, "
            f"and its fillet sets the pinch&#8217;s friction arm at {f(min(ratios), 1) if ratios else '&#8211;'}&#8211;{f(max(ratios), 1) if ratios else '&#8211;'} "
            f"times the sphere&#8217;s. On the eight deployed hands&#8217; three-finger turn, the pads and Drake give the same outcome on "
            f"{f(100 * s6 / n6, 0) if n6 else '&#8211;'}&#8202;% of paired seeds with the 6&#8202;mm block and {f(100 * s27 / n27, 0) if n27 else '&#8211;'}&#8202;% "
            f"with the 2.7&#8202;mm block, against {f(100 * sm_ / nm, 0) if nm else '&#8211;'}&#8202;% for MuJoCo&#8217;s single-mesh point contact and "
            f"{f(100 * sl / nl, 0) if nl else '&#8211;'}&#8202;% for the legacy box; the sphere tip holds the tool on {sph_mj} and {sph_dk} of 72 seeds in "
            f"MuJoCo and Drake where the 6&#8202;mm block holds {tpu_mj} and {tpu_dk}. No contact model reproduces the bench hand by hand: D2, held 10/10 "
            f"on the bench, drops in every condition at the calibrated servo gain, and with the tool resting on its post (every bench scene since "
            f"2026-08-29 starts it 12.5&#8202;mm inside the post) only the shipped servo gain kp&#8202;30 turns it by tens of degrees, "
            f"{f(rig_t[1], 0) if len(rig_t) > 1 else '&#8211;'}&#8211;{f(rig_t[-1], 0) if rig_t else '&#8211;'}&#176; on {max(0, len(rig_t) - 1)} hands "
            f"against the bench&#8217;s 33&#8211;70&#176;, while holding {rig_k} of {rig_n} seeds. The co-design section proposes scoring "
            f"candidate fingertips on task success and on this agreement between simulators together.")


def main():
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BED_PATH": BED_PATH, "OVERVIEW_PATH": OVERVIEW_PATH}
    bu = os.path.join(BED, "artifact_url.txt")
    ou = os.path.join(ROOT, "docs/experiments/20261005-contact_overview/artifact_url.txt")
    v["BED_LINK"] = f', <a href="{open(bu).read().strip()}">artifact</a>' if os.path.exists(bu) else ""
    v["OVERVIEW_LINK"] = f', <a href="{open(ou).read().strip()}">artifact</a>' if os.path.exists(ou) else ""
    v["GLOSSARY"] = glossary()
    v["CREEP"] = creep_section()
    v["TIP"] = tip_section()
    v["TURN"] = turn_section()
    v["DESIGN"] = design_section()
    v["NEXT"] = next_section()
    v["LEDE"] = lede()
    v["FOOTER"] = ("<p>Rebuild: <code>python3 scripts/fingertip_backends_page.py</code>. Rows: <code>docs/experiments/20261006-fingertip_backends/*.jsonl</code> "
                   "and <code>docs/experiments/20261005-contact_bed/creep.jsonl</code>; scenes: <code>assets/mjcf/experimental/20261006-reorient_backends/</code>; "
                   "tip meshes: <code>assets/mjcf/experimental/20261006-tpu_tip/</code>; films: <code>docs/experiments/20261006-fingertip_backends/media/</code>.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    open(OUT, "w").write(retro_style.apply(t))  # plain page style (owner, 2026-10-09)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
