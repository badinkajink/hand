#!/usr/bin/env python3
"""Page: accuracy of the 1 mm sphere pads across fingertip geometry, and the co-design of the printed fingertip
(geometry, print recipe, contact model) on the SR2 hand's tasks (study 20261014-tip_codesign; plan
docs/notes/20261014-tip_codesign_plan.md; hand-off docs/handoff/20261014-tip_codesign.md).

Data: docs/experiments/20261014-tip_codesign/ (deterioration/, tolerance/, screen/, final/, print/, media/), every
number read from the study's summaries; scripts/tip_codesign_verify.py recomputes the headline numbers from the rows.

    .venv/bin/python scripts/tip_codesign_figures.py deterioration rate screen stage2 stage3
    .venv/bin/python scripts/tip_codesign_page.py
Out: docs/experiments/20261014-tip_codesign/20261014-sphere_pad_accuracy_and_fingertip_codesign.html (media inlined).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261014-tip_codesign")
OUT = os.path.join(D, "20261014-sphere_pad_accuracy_and_fingertip_codesign.html")
TPL = os.path.join(ROOT, "scripts/tip_codesign_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
FIG, TAB, REFS = [0], [0], {}
SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"
EXTRA_CSS = """
.tw th{white-space:normal !important;vertical-align:bottom;text-transform:none}
.tw td:first-child{min-width:7em}
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
figure.films video{width:100%;height:auto}
figure.img img{width:100%;height:auto}
figure.half img{width:66%;height:auto}
.eq{margin:.8em 0;text-align:center}
""" + texsvg.BLOCK_CSS + "\n"


# ------------------------------------------------------------------------------------------ helpers

def J(rel):
    p = os.path.join(D, rel)
    return json.load(open(p)) if os.path.exists(p) else None


def rows(rel):
    p = os.path.join(D, rel)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def data_uri(path, mime=None):
    if mime is None:
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".mp4": "video/mp4"}[os.path.splitext(path)[1]]
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def jpeg_uri(path, quality=86):
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def figure(body, caption, key=None, cls="img"):
    FIG[0] += 1
    if key:
        REFS[f"fig:{key}"] = FIG[0]
    return f'<figure class="{cls}">{body}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def img(rel, caption, key=None, cls="img"):
    p = os.path.join(D, rel)
    if not os.path.exists(p):
        return f"<p><i>[figure {rel} not drawn yet]</i></p>"
    return figure(f'<img alt="" src="{data_uri(p)}">', caption, key, cls)


def film(rel, poster_rel, caption, key=None):
    p = os.path.join(D, rel)
    if not os.path.exists(p):
        return f"<p><i>[film {rel} not rendered yet]</i></p>"
    v = (f'<video controls muted loop playsinline preload="metadata" poster="{jpeg_uri(os.path.join(D, poster_rel))}" '
         f'src="{data_uri(p)}"></video>')
    return figure(v, caption, key, cls="films")


def table(head, body, caption, note=None, text_cols=(0,), key=None):
    TAB[0] += 1
    if key:
        REFS[f"tab:{key}"] = TAB[0]
    cls = lambda i: "" if i in text_cols else " class=num"  # noqa: E731
    out = [f"<p class='tcap'><b>Table&#160;{TAB[0]}.</b> {caption}</p>",
           "<div class='tw'><table><thead><tr>" + "".join(f"<th{cls(i)}>{h}</th>" for i, h in enumerate(head))
           + "</tr></thead><tbody>"]
    for r in body:
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


def f2(x, n=2):
    return "&#8211;" if x is None else f"{x:.{n}f}"


MM, UM, NN, DEG = "&#8202;mm", "&#8202;&#181;m", "&#8202;N", "&#176;"


def tip_name(k):
    if k == "tpu2.7":
        return "block r&#160;2.7 (simulation default)"
    if k == "tcd_rs3_re3_p0.2":
        return "as-built (3,&#160;3), printed"
    if k == "tcd_rs3_re3":
        return "as-built (3,&#160;3), design surface"
    m = re.match(r"tcd_rs([0-9.]+)_re([0-9.]+)", k)
    return f"({m[1]},&#160;{m[2]})"


# ------------------------------------------------------------------------------------------ sections

GLOSSARY = [
    ("1&#8202;mm pads", "The fingertip&#8217;s palmar region as spheres of radius 0.75&#8202;mm on a 1&#8202;mm lattice (1,160 on the "
     "as-built tip as printed), each a MuJoCo soft contact of stiffness $K_i = E a_i / h$ with $E$ = 10&#8202;MPa, $a_i$ the pad&#8217;s "
     "share of the surface and $h$ = 8.5&#8202;mm; relaxation time 0.02&#8202;s, impratio 100, elliptic friction cone, 1&#8202;ms "
     "step. Builder v1 gives every finger its median $K_i$; builder v2 uses each pad&#8217;s own $K_i$ with $h_i$ the pad&#8217;s "
     "normal distance to the volume centroid. The pads on the printed TPU block are the hand&#8217;s default fingertip."),
    ("Drake hydroelastic", "Drake 1.57&#8217;s compliant convex hull of the tip, modulus 10&#8202;MPa, dissipation time scale "
     "0.01&#8202;s, SAP solver: the dynamic reference."),
    ("Law", "The quasi-static integral of Drake&#8217;s pressure field $p = E\\,(1 - g(\\mathbf{x}))$, $g$ the gauge of the tip&#8217;s "
     "convex hull about its volume centroid, against the bed&#8217;s tool cylinder at angle $\\psi$ to the pinch line "
     "(<code>scripts/tip_codesign_law.py</code>). It gives the approach, the twist-onset arm, the all-slip spin arm and the brake force "
     "with no time stepping."),
    ("Bed", "Two force-controlled rails squeeze a 25&#8202;mm diameter, 24.5&#8202;g screwdriver between two fingertips at 1&#8202;N "
     "per pad, friction coefficient $\\mu$ = 1, 1&#8202;ms step (the contact bed of <code>docs/overviews/contact_model.html</code>)."),
    ("Approach", "Rail travel at 1&#8202;N per pad from first touch, in &#181;m."),
    ("Twist-onset arm", "Bed case T2: the torque at which the tool starts to turn under a slow torque ramp, over $2\\mu N$, in mm; "
     "the law&#8217;s value is its all-slip arm with the tool along the pinch line ($\\psi$ = 0)."),
    ("Spin arm", "The mean torque that drives the tool about the pinch axis at rate $\\omega$ over the turn 0.4&#8211;1.2&#8202;rad, "
     "over $2\\mu N$, in mm; the law&#8217;s value is the quasi-static all-slip arm averaged over $\\psi$ = 23, 46 and 69&#176;."),
    ("Brake force", "Bed case T5: the pinch force per pad at which the tool, released with its centre of mass 15&#8202;mm from the "
     "pinch, stops swinging at 45&#176;, in N."),
    ("Flat family", "<code>tpu&lt;r&gt;</code>: the 17.0&#8202;&#215;&#8202;14.8&#8202;&#215;&#8202;22.0&#8202;mm block with every "
     "edge rounded at fillet radius $r$ = 0&#8211;7.4&#8202;mm; the printed tip in simulation is $r$ = 2.7&#8202;mm."),
    ("Candidate tip (r<sub>side</sub>, r<sub>end</sub>)", "The CAD tip envelope, 17.0&#8202;&#215;&#8202;14.8&#8202;&#215;&#8202;"
     "21.41&#8202;mm, with the side edges rounded at r<sub>side</sub> and the distal end at r<sub>end</sub> (mm), as printed face up "
     "in 0.2&#8202;mm layers, so the curved faces are terraces (keys <code>tcd_rs&lt;r_side&gt;_re&lt;r_end&gt;_p0.2</code>). "
     "As-built is (3,&#160;3); its design surface matches the CAD insert&#8217;s outer palmar half to 1.2&#8202;&#181;m (99th "
     "percentile)."),
    ("Servo-limited plant", "The SR2 servo model with gain kp&#160;0.72, torque limit 0.18&#8202;N&#8202;m and damping 0.0144 "
     "(<code>kp0.72_kv0_fr0.18_fl0_dp0.0144</code>)."),
    ("HOM", "The hand&#8217;s closed-loop in-hand turn controller (<code>hom_turn3</code>, model gain 0.72) on the eight deployed "
     "hands D1&#8211;D8 and on the CAD hand&#8217;s cells."),
    ("Cells", "Four grasps of the CAD hand: cylinders of 25, 16, 10 and 16&#8202;mm diameter on finger layouts with thumb-to-pair "
     "and pair separations of 92.5/102.5, 62.5/72.5, 55/65 and 122.5/132.5&#8202;mm, with 9, 9, 9 and 6 candidate grasps."),
    ("Held turn", "The HOM turn at the end of a rollout when the tool is held, otherwise 0, in degrees."),
    ("Stage score", "The mean of five parts, each a held turn over 90&#176;: for each cell, the mean held turn of the grasp chosen by "
     "its held turn on seed 0, over the scoring seeds; and the mean held turn of HOM on D1&#8211;D8."),
    ("J1, J2, J3", "J1: the stage score at $\\mu$ = 0.7 on the servo-limited plant, scoring seeds 1&#8211;2 (stage 1). J2: the "
     "minimum stage score over five conditions, $\\mu$ = 1.0, 0.7 and 0.5 on the servo-limited plant, gain 0.5 and torque limit "
     "0.12&#8202;N&#8202;m at $\\mu$ = 0.7 (stage 2, HOM seeds 0&#8211;4). J3: the minimum over the four conditions other than "
     "$\\mu$ = 1.0 on fresh placements only, cell seeds 3&#8211;5 and HOM seeds 5&#8211;9 (stage 3)."),
    ("Agreement with Drake", "Plan replays of the eight deployed hands (seeds 0&#8211;2, $\\mu$ = 1.0, 24 runs) whose held outcome "
     "matches Drake&#8217;s and whose turn lies within 3&#176; of Drake&#8217;s. The gate admits a tip with at most two agreements "
     "fewer than the as-built tip."),
    ("Three-finger hold", "Grasps of the four cells (33) that keep the tool through the hold test&#8217;s force and torque ramps at "
     "$\\mu$ = 0.5 with all three fingers loaded."),
    ("Hand tolerance &#964;", "The relative friction change at which the share of hand runs whose outcome changes (held flips or "
     "the turn moves more than 3&#176;) exceeds the chaotic floor by 5 points; 3.65&#8202;% (<code>tolerance/summary.json</code>)."),
    ("E*", "A print recipe&#8217;s effective modulus from the whole-tip finite-element model with a 4.76&#8202;mm magnet 5&#8202;mm "
     "deep."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def question():
    return ("<p>The 1&#8202;mm pads model the printed TPU fingertip in every simulator the hand uses. This study measures how far "
            "they depart from the law and from Drake as the fingertip&#8217;s shape changes, and then uses the pads to choose a "
            "printable fingertip, its print recipe and its pad model for the hand&#8217;s tasks. The deterioration sweep runs the "
            "bed on the flat family ($r$ = 0&#8211;7.4&#8202;mm), the 10.55&#8202;mm sphere tip and five CAD-tip members. The "
            "co-design screens 24 printable candidates of the CAD envelope, the as-built design surface and the simulation "
            "default in three stages: J1 at $\\mu$ = 0.7 (26 tips), J2 over five friction and servo conditions with an agreement "
            "gate against Drake (the eight highest J1, the as-built tip and the default), and J3 on fresh placements for the "
            "finalists against the as-built tip. Finite elements and the slicer then fix the finalists&#8217; print recipe and "
            "files. The plan, its critic&#8217;s revisions and the falsifiers are in "
            "<code>docs/notes/20261014-tip_codesign_plan.md</code>.</p>")


def accuracy_section():
    S = J("deterioration/summary.json")["members"]
    fam = sorted((float(k), k) for k in S if k.replace(".", "").isdigit())
    body = []
    for r, k in fam:
        m = S[k]
        body.append([f"{r:g}", f2(m["base"]["onset_over_law"]), f2(m["drake"]["onset_over_law"]),
                     f2(m["base"]["spin_over_law"]["1"]), f2(m["drake"]["spin_over_law"].get("1")),
                     f2(m["base"]["N45_over_drake"]), f2(m.get("slip_number_1"), 1)])
    sp = S["sphere"]
    body.append(["sphere", f2(sp["base"]["onset_over_law"]), f2(sp["drake"]["onset_over_law"]),
                 f2(sp["base"]["spin_over_law"]["1"]), f2(sp["drake"]["spin_over_law"].get("1")),
                 f2(sp["base"]["N45_over_drake"]), "&#8211;"])
    for k in ("tcd_rs3_re3", "tcd_rs3_re3_p0.2", "tcd_rs5_re1"):
        m = S[k]
        body.append([tip_name(k) if k != "tcd_rs5_re1" else "(5,&#160;1), design surface", f2(m["base"]["onset_over_law"]),
                     f2(m["drake"]["onset_over_law"]), f2(m["base"]["spin_over_law"]["1"]),
                     f2(m["drake"]["spin_over_law"].get("1")), f2(m["base"]["N45_over_drake"]), f2(m.get("slip_number_1"), 1)])
    hr = {r["model"]: r for r in rows("deterioration/hullres.jsonl") if r["case"] == "spin"}
    hl = {r["model"]: r for r in rows("deterioration/hullres_law.jsonl")}
    h5, h25 = hr.get("drake_tcd_rs3_re3_hull0.5mm"), hr.get("drake_tcd_rs3_re3_hull0.25mm")
    l5, l25 = hl.get("drake_tcd_rs3_re3_hull0.5mm"), hl.get("drake_tcd_rs3_re3_hull0.25mm")
    hull = ""
    if h5 and h25 and l5 and l25:
        hull = (f"<p>Drake&#8217;s spin arm on the as-built design surface is {h5['rbar_spin_mm']:.2f}{MM} on the study&#8217;s "
                f"0.5&#8202;mm hull, {h5['rbar_spin_mm'] / l5['law_spin_arm_1N_mm']:.2f} times the law, with "
                f"{h5['n_L']} contact quadrature points; on a 0.25&#8202;mm hull it has {h25['n_L']} points and "
                f"{h25['rbar_spin_mm']:.2f}{MM}, {h25['rbar_spin_mm'] / l25['law_spin_arm_1N_mm']:.2f} times the law "
                f"(<code>scripts/tip_codesign_hullres.py</code>). Drake&#8217;s excess on that member is its contact-surface "
                "resolution.</p>")
    t = (f"<p>The pads give the twist-onset arm within 2&#8202;% of the law for fillet radii of 1.5&#8202;mm and more, and "
         f"7&#8211;14&#8202;% above it on sharper blocks (Figure&#160;[[fig:det]], Table&#160;[[tab:det]]). Their sliding torque "
         f"is too large wherever the patch is long: at 1&#8202;rad/s the spin arm is {S['0']['base']['spin_over_law']['1']:.2f} "
         f"times the law on the sharp block, {S['2.7']['base']['spin_over_law']['1']:.2f} times on the printed tip&#8217;s "
         f"$r$ = 2.7&#8202;mm, {S['7.4']['base']['spin_over_law']['1']:.2f} times on the fully rounded block and "
         f"{sp['base']['spin_over_law']['1']:.2f} times on the sphere, while Drake stays within "
         f"{min(S[k]['drake']['spin_over_law']['1'] for _, k in fam):.2f}&#8211;"
         f"{max(S[k]['drake']['spin_over_law']['1'] for _, k in fam):.2f}. The braked swing needs only "
         f"{min(S[k]['base']['N45_over_drake'] for r, k in fam if r <= 3):.2f}&#8211;"
         f"{max(S[k]['base']['N45_over_drake'] for r, k in fam if r <= 3):.2f} of Drake&#8217;s pinch force for $r$ &#8804; "
         "3&#8202;mm and the same force for $r$ &#8805; 6&#8202;mm: the larger sliding torque brakes the swing earlier. The "
         "as-built design surface and its printed terraces behave like the $r$ = 3&#8202;mm block.</p>")
    t += img("media/fig_deterioration.png", "Pads (1&#8202;mm spheres) and Drake against the law over the flat family&#8217;s fillet "
             "radius and on the 10.55&#8202;mm sphere tip, at 1&#8202;N per pad: twist-onset arm, spin arm at 1&#8202;rad/s, and the "
             "pads&#8217; brake force at 45&#176; over Drake&#8217;s (Drake at 0.5&#8202;ms). Rows: "
             "<code>deterioration/{law,T2,spin,T5}.jsonl</code>.", key="det")
    t += table(["Tip", "Onset / law, pads", "Onset / law, Drake", "Spin / law, pads", "Spin / law, Drake", "Brake force, pads / Drake",
                "Slip number"], body,
               "Bed metrics over the law (Drake for the brake force) at 1&#8202;N per pad; spin at 1&#8202;rad/s. Rows in "
               "<code>deterioration/</code>; summary <code>deterioration/summary.json</code>.", key="det",
               text_cols=(0,))
    t += hull
    t += film("media/bed_spin_tile.mp4", "media/bed_spin_tile_poster.jpg",
              "Driven spin at 1&#8202;N and 1&#8202;rad/s, four tips, one frame per 0.01&#8202;rad (4&#215; slow motion), seen from the "
              "tool through the tool. Pads touching the tool are coloured by normal force over $K$ times penetration: white at 1, "
              "red at 10 or more. The pads at the ends of the contact line carry 10 or more times their spring force. Labels: the "
              "running spin arm from 0.4&#8202;rad and the law&#8217;s. <code>scripts/tip_codesign_film.py bed</code>.",
              key="bedfilm")
    return t


def rate_section():
    S = J("deterioration/summary.json")["members"]["2.7"]
    hf = J("deterioration/handfix_summary.json")
    fixk = "E1e+07_pf0.1_ir1000"
    eq = ("$$f_n = \\frac{D_0}{1+\\mu_r^2}\\left(\\mu\\,\\lVert \\mathbf{a}_t + B\\,\\mathbf{v}_t \\rVert - j_n\\right)$$")
    t = (f"<p>The pads&#8217; sliding torque grows with spin rate (Figure&#160;[[fig:rate]]). On the $r$ = 2.7&#8202;mm block the "
         f"spin arm is {S['base']['spin_mm']['0.03']:.2f}{MM} at 0.03&#8202;rad/s, {S['base']['spin_mm']['1']:.2f}{MM} at "
         f"1&#8202;rad/s and {S['base']['spin_mm']['3']:.2f}{MM} at 3&#8202;rad/s, against the law&#8217;s "
         f"{S['law']['spin_1N_mm']:.2f}{MM}; Drake gives {S['drake']['spin_mm']['1']:.2f}{MM} at 1&#8202;rad/s. MuJoCo&#8217;s "
         "elliptic cone sets the normal force of a sliding contact as</p>"
         f"<div class='eq'>{eq}</div>"
         "<p>with $D_0$ the contact&#8217;s regularisation, $\\mu_r$ its regularised friction, $\\mathbf{a}_t$ and $\\mathbf{v}_t$ "
         "the tangential acceleration and velocity of the contact, $B$ the reference damping and $j_n$ the normal reference "
         "acceleration (<code>engine_core_constraint.c</code>). The slip term $\\mu\\lVert B\\mathbf{v}_t\\rVert$ acts as a phantom "
         "penetration $\\zeta t_r \\mu v_\\mathrm{slip}$, about 120&#8202;&#181;m at 6&#8202;mm from the axis at 1&#8202;rad/s, and "
         f"carries {100 * S['base']['slip_share_1']:.0f}&#8202;% of the sliding pads&#8217; normal force at 1&#8202;rad/s. It does "
         "not depend on impratio. The slip number in Table&#160;[[tab:det]] compares that phantom penetration with the real one; "
         "it falls from 6.2 on the sharp block to 0.3 on the round one, which is why rounder tips deteriorate less.</p>"
         "<p>Explicit pad-tool contact pairs with friction-row damping 0.1&#8202;s and impratio 1000 remove most of the excess "
         f"(spin arm {S['fix']['spin_mm']['1']:.2f}{MM} at 1&#8202;rad/s) but break the hand: HOM turns within 5&#176; of Drake "
         f"fall from 21 to {hf[fixk]['hom']['n_agree']} of 24 and plan replays within 3&#176; from 16 to "
         f"{hf[fixk]['replay']['n_agree']} of 24. The friction rows&#8217; damping that adds normal force to a sliding pad is the "
         "same damping that keeps a stuck pad stuck, so no builder setting removes one without the other, and the pads stay as "
         "built. A planner that relies on torsion should expect 1.9 times the quasi-static sliding arm from the pads at "
         "5&#8202;rad/s (pinch-and-swing peaks reach 306&#176;/s), against 1.6 times from Drake.</p>")
    t += img("media/fig_rate.png", "Spin arm at 1&#8202;N against spin rate on the $r$ = 2.7&#8202;mm block: pads as built, pads with "
             "explicit pairs (friction-row damping 0.1&#8202;s, impratio 1000), Drake, the law and the quasi-static pads. Rows: "
             "<code>deterioration/spin.jsonl</code>, <code>fixgrid.jsonl</code>.", key="rate", cls="half")
    return t


def builder_section():
    B = J("deterioration/summary.json")["builders"]
    v2 = J("deterioration/handv2_summary.json")
    body = []
    for k in ("tpu1", "tpu2.7", "tpu6", "tcd_rs3_re3", "tcd_rs3_re3_p0.2"):
        b = B[k]
        body.append([tip_name(k) if k.startswith("tcd") else f"block r&#160;{k[3:]}"] +
                    [f2(b[v]["arm0_over_law_1N"], 3) for v in ("pads_s0.5", "pads_s1", "pads_s1.5", "pads_hloc", "pads_Keach")]
                    + [f2(100 * b["ripple_1mm"]["1"]["arm0_ripple"], 1)])
    fb = J("final/summary_builders.json") or {}
    t = ("<p>Quasi-static pad models of the same shapes separate the builder&#8217;s geometric error from the rate effect. At 1&#8202;N "
         "the 1&#8202;mm pads give the law&#8217;s twist-onset arm within &#8722;3 to +2&#8202;% on smooth members, and pad spacing, "
         "the local foundation depth and per-pad stiffness move it by less than 1&#8202;% (Table&#160;[[tab:builder]]). The "
         "remaining error is set by where the lattice falls under the contact line: moving the line 1&#8202;mm along the finger "
         "changes the arm by the ripple in the last column.</p>")
    t += table(["Tip", "0.5&#8202;mm pads", "1&#8202;mm pads", "1.5&#8202;mm pads", "1&#8202;mm, local h", "1&#8202;mm, per-pad K",
                "Ripple, %"], body,
               "Quasi-static pads&#8217; twist-onset arm over the law at 1&#8202;N ($\\psi$ = 0) by builder variant, and the ripple of "
               "the 1&#8202;mm pads&#8217; arm as the tool line moves 0&#8211;1&#8202;mm along the finger (max minus min over the "
               "mean). <code>deterioration/builders.jsonl</code>.", key="builder")
    t += (f"<p>On the hand the local foundation depth with per-pad stiffness (builder v2) agrees with Drake on "
          f"{v2['E1e+07_hlocal_Keach']['within3_of_drake']} of 24 plan replays of the as-built tip as printed, against "
          f"{v2['E1e+07']['within3_of_drake']} for builder v1; {v2['v1_vs_v2_changed_replays']} of 40 replays differ by more "
          "than 3&#176; between the two, the same share as any 0.1&#8202;% friction change.</p>")
    if fb:
        t += builders_finalists(fb)
    return t


def builders_finalists(fb):
    body = []
    for k, res in fb.items():
        tags = [x for x in res if isinstance(res[x], dict)]
        v1 = next((res[x] for x in tags if x == "E1e+07"), None)
        v2 = next((res[x] for x in tags if "hlocal" in x), None)
        if v1 and v2:
            body.append([tip_name(k), f"{v1['within3_of_drake']}/{v1['n_vs_drake']}", f"{v2['within3_of_drake']}/{v2['n_vs_drake']}",
                         f"{v1['hom_held']}/{v1['n_hom']}", f"{v2['hom_held']}/{v2['n_hom']}", str(res.get("v1_vs_v2_changed_replays"))])
    if not body:
        return ""
    return table(["Tip", "Agree with Drake, v1", "Agree with Drake, v2", "HOM held, v1", "HOM held, v2", "Replays differing &gt;3&#176;"],
                 body, "Builder v1 against v2 on the finalists: replays D1&#8211;D8 seeds 0&#8211;4 (40), of which seeds 0&#8211;2 "
                 "against Drake (24), and HOM seeds 0&#8211;2 at $\\mu$ = 1.0. <code>final/summary_builders.json</code>.",
                 key="buildfin")


def screen_section():
    S1 = J("screen/summary_stage1.json")
    if not S1:
        return "<p><i>[stage 1 pending]</i></p>"
    C = S1["candidates"]
    rank = sorted(((v["J1"] or -1, k) for k, v in C.items()), reverse=True)
    top = ", ".join(f"{tip_name(k)} {J_:.3f}" for J_, k in rank[:8])
    law = {r["tip"]: r for r in rows("screen/fidelity/law.jsonl") if r["field"] == "drake"}
    sp = [(C[k]["fidelity"]["pads_spin_1N_1_mm"] / law[k]["spin_arm_mm"]["1"], k) for k in C if k in law and C[k]["fidelity"].get("pads_spin_1N_1_mm")]
    on = [C[k]["fidelity"]["pads_onset_1N_mm"] / law[k]["table"]["1|0"]["arm_mm"] for k in C if k in law and C[k]["fidelity"].get("pads_onset_1N_mm")]
    t = (f"<p>The grid crosses six side fillets (1, 2, 3, 4.5, 6 and 7.4&#8202;mm) with four end fillets (1, 3, 6 and "
         f"10.5&#8202;mm), all printed face up at 0.2&#8202;mm; with the as-built design surface and the simulation default it "
         f"has 26 tips. Their J1 ranks {top} first (Figure&#160;[[fig:screen]]); the simulation default scores "
         f"{C['tpu2.7']['J1']:.3f}. Across the 24 printed candidates the pads&#8217; twist-onset arm stays within "
         f"{min(on):.2f}&#8211;{max(on):.2f} of the law, while the 1&#8202;rad/s spin arm reaches {max(sp)[0]:.2f} times the law on "
         f"{tip_name(max(sp)[1])}: the rate effect of the flat family grows on the rounder printed tips, whose short patches make "
         "the law&#8217;s arm small.</p>")
    t += img("media/fig_screen.png", "Stage 1 over the printed grid (rows r<sub>side</sub>, columns r<sub>end</sub>): J1 at "
             "$\\mu$ = 0.7, and the 1&#8202;mm pads against the law: spin arm at 1&#8202;rad/s and brake force at 45&#176;. Rows: "
             "<code>screen/scale/</code>, <code>screen/hom.jsonl</code>, <code>screen/fidelity/</code>.", key="screen")
    return t


def robust_section():
    S2 = J("screen/summary_stage2.json")
    if not S2:
        return "<p><i>[stage 2 pending]</i></p>"
    C, why = S2["candidates"], S2["finalist_rule"]
    base = C["tcd_rs3_re3_p0.2"]
    order = sorted(C, key=lambda k: -(C[k]["J2"] or -1))
    body = [[tip_name(k), f2(C[k]["J2"], 3), f"{C[k]['agree_drake_3deg']}/{C[k]['n_vs_drake']}",
             "yes" if C[k]["gate_pass"] else "no", f2(C[k]["cost_ratio"]), f2(C[k].get("three_finger_frac_mu0.5_cells"))]
            for k in order]
    fin = [k for k in S2["finalists"] if k != "tcd_rs3_re3_p0.2"]
    t = (f"<p>Stage 2 runs the eight highest J1, the as-built tip and the simulation default in five conditions and keeps each "
         f"tip&#8217;s worst condition as J2 (Figure&#160;[[fig:stage2]], Table&#160;[[tab:stage2]]). The as-built tip agrees with "
         f"Drake on {base['agree_drake_3deg']} of 24 plan replays, so the gate needs {base['agree_drake_3deg'] - 2}. The finalists "
         "by the plan&#8217;s rules are " + "; ".join(f"{tip_name(k)}, {why[k][4:] if why[k].startswith('(') else why[k]}" for k in fin)
         + ", and the as-built tip as the baseline.</p>")
    t += img("media/fig_stage2.png", "Stage 2: the stage score in each condition (markers) and J2, the minimum (bar), per tip; right, "
             "plan replays within 3&#176; of Drake with its held outcome, of 24 (dark: passes the gate; dashed: the gate). Rows: "
             "<code>screen/stage2/</code>.", key="stage2")
    t += table(["Tip", "J2", "Agree with Drake", "Gate", "Cost / as-built", "Three-finger share, &#956; 0.5"], body,
               "Stage-2 summary (<code>screen/summary_stage2.json</code>): cost is the median MuJoCo time per replay step over the "
               "as-built tip&#8217;s; three-finger share is the share of cell turn runs ending with all three fingers loaded.",
               key="stage2")
    return t


def confirm_section():
    S3 = J("final/summary_stage3.json")
    if not S3:
        return "<p><i>[stage 3 pending]</i></p>"
    return "<p><i>[stage 3 text after the run]</i></p>" + img("media/fig_stage3.png", "Stage 3.", key="stage3")


def material_section():
    return "<p><i>[FE pending]</i></p>"


def print_section():
    return "<p><i>[print files pending]</i></p>"


def lede():
    return "[lede after the stage-3 results]"


def limitations():
    return ("<p>The references are computed, not measured: the law and Drake share a pressure field and the unmeasured 10&#8202;MPa "
            "modulus, and no printed candidate has been tested on the bench. Task scores come from open-loop plan replays and "
            "closed-loop HOM turns in simulation; the hand tolerance &#964; = 3.65&#8202;% relative friction bounds what any "
            "single run can resolve, and 10&#8202;% of runs change outcome under any friction change of 0.1&#8202;% or more. The "
            "candidate grid varies two fillet radii on one envelope; crowned faces flatten into one terrace when printed face up and "
            "stay outside the grid.</p>")


def open_items():
    items = [
        ("Bench twist of the printed tips", "twist the printed as-built tip and the best finalist on the bench cylinder at 0.1 and "
         "1&#8202;rad/s; a sliding arm near the law (3.7&#8202;mm on the as-built tip) at both rates confirms the pads&#8217; "
         "rate effect as a model artefact, an arm that grows with rate like the pads&#8217; refutes it."),
        ("Friction of the printed tip", "pull test of two printed tips at 0.5, 1 and 2&#8202;N per tip with "
         "<code>scripts/tpu_tip_calibrate.py fit --pull</code>; every task here varies $\\mu$ over 0.5&#8211;1.0."),
        ("Zero-shot policies on the finalist", "evaluate the six servo-plant policies on the best finalist with "
         "<code>contact_feedback_replay.py</code> and the scene from <code>scripts/tip_codesign_rig.py rlscene</code>."),
    ]
    return "<ol>" + "".join(f"<li><b>{a}.</b> {b}</li>" for a, b in items) + "</ol>"


def byline():
    return ("Scripts <code>scripts/tip_codesign_{law,geometry,bed,sweep,rig,analysis,fe,print,film,figures,hullres,verify}.py</code> "
            "&#183; builder <code>scripts/tip_codesign_page.py</code> &#183; rows <code>docs/experiments/20261014-tip_codesign/</code>")


def main():
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "GLOSSARY": glossary(), "QUESTION": question(), "ACCURACY": accuracy_section(),
         "RATE": rate_section(), "BUILDER": builder_section(), "SCREEN": screen_section(), "ROBUST": robust_section(),
         "CONFIRM": confirm_section(), "MATERIAL": material_section(), "PRINT": print_section(), "LIMITS": limitations(),
         "OPEN": open_items(), "LEDE": lede()}
    v["FOOTER"] = ("<p>Rebuild: <code>.venv/bin/python scripts/tip_codesign_page.py</code>. Rows and summaries in "
                   "<code>docs/experiments/20261014-tip_codesign/</code>: <code>deterioration/</code>, <code>tolerance/</code>, "
                   "<code>screen/</code> (stages 1 and 2), <code>final/</code> (stage 3, FE), <code>print/</code>, "
                   "<code>media/</code>; verification <code>verify.json</code> from <code>scripts/tip_codesign_verify.py</code>.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    miss = sorted(set(re.findall(r"\[\[((?:fig|tab):[a-z0-9_]+)\]\]", t)) - set(REFS))
    if miss:
        raise SystemExit(f"unresolved references: {miss}")
    t = re.sub(r"\[\[((?:fig|tab):[a-z0-9_]+)\]\]", lambda m: str(REFS[m.group(1)]), t)
    t, n = render_tex(t)
    t = retro_style.apply(t)
    open(OUT, "w").write(t)
    print(f"formulas {n}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
