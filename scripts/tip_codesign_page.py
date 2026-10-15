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
     "as-built tip as printed), each a MuJoCo soft contact of stiffness \\(K_i = E a_i / h\\) with \\(E\\) = 10&#8202;MPa, \\(a_i\\) the pad&#8217;s "
     "share of the surface and \\(h\\) = 8.5&#8202;mm; relaxation time 0.02&#8202;s, impratio 100, elliptic friction cone, 1&#8202;ms "
     "step. Builder v1 gives every finger its median \\(K_i\\); builder v2 uses each pad&#8217;s own \\(K_i\\) with \\(h_i\\) the pad&#8217;s "
     "normal distance to the volume centroid. The pads on the printed TPU block are the hand&#8217;s default fingertip."),
    ("Drake hydroelastic", "Drake 1.57&#8217;s compliant convex hull of the tip, modulus 10&#8202;MPa, dissipation time scale "
     "0.01&#8202;s, SAP solver: the dynamic reference."),
    ("Law", "The quasi-static integral of Drake&#8217;s pressure field \\(p = E\\,(1 - g(\\mathbf{x}))\\), \\(g\\) the gauge of the tip&#8217;s "
     "convex hull about its volume centroid, against the bed&#8217;s tool cylinder at angle \\(\\psi\\) to the pinch line "
     "(<code>scripts/tip_codesign_law.py</code>). It gives the approach, the twist-onset arm, the all-slip spin arm and the brake force "
     "with no time stepping."),
    ("Bed", "Two force-controlled rails squeeze a 25&#8202;mm diameter, 24.5&#8202;g screwdriver between two fingertips at 1&#8202;N "
     "per pad, friction coefficient \\(\\mu\\) = 1, 1&#8202;ms step (the contact bed of <code>docs/overviews/contact_model.html</code>)."),
    ("Approach", "Rail travel at 1&#8202;N per pad from first touch, in &#181;m."),
    ("Twist-onset arm", "Bed case T2: the torque at which the tool starts to turn under a slow torque ramp, over \\(2\\mu N\\), in mm; "
     "the law&#8217;s value is its all-slip arm with the tool along the pinch line (\\(\\psi\\) = 0)."),
    ("Spin arm", "The mean torque that drives the tool about the pinch axis at rate \\(\\omega\\) over the turn 0.4&#8211;1.2&#8202;rad, "
     "over \\(2\\mu N\\), in mm; the law&#8217;s value is the quasi-static all-slip arm averaged over \\(\\psi\\) = 23, 46 and 69&#176;."),
    ("Brake force", "Bed case T5: the pinch force per pad at which the tool, released with its centre of mass 15&#8202;mm from the "
     "pinch, stops swinging at 45&#176;, in N."),
    ("Flat family", "<code>tpu&lt;r&gt;</code>: the 17.0&#8202;&#215;&#8202;14.8&#8202;&#215;&#8202;22.0&#8202;mm block with every "
     "edge rounded at fillet radius \\(r\\) = 0&#8211;7.4&#8202;mm; the printed tip in simulation is \\(r\\) = 2.7&#8202;mm."),
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
    ("J1, J2, J3", "J1: the stage score at \\(\\mu\\) = 0.7 on the servo-limited plant, scoring seeds 1&#8211;2 (stage 1). J2: the "
     "minimum stage score over five conditions, \\(\\mu\\) = 1.0, 0.7 and 0.5 on the servo-limited plant, gain 0.5 and torque limit "
     "0.12&#8202;N&#8202;m at \\(\\mu\\) = 0.7 (stage 2, HOM seeds 0&#8211;4). J3: the minimum over the four conditions other than "
     "\\(\\mu\\) = 1.0 on fresh placements only, cell seeds 3&#8211;5 and HOM seeds 5&#8211;9 (stage 3)."),
    ("Agreement with Drake", "Plan replays of the eight deployed hands (seeds 0&#8211;2, \\(\\mu\\) = 1.0, 24 runs) whose held outcome "
     "matches Drake&#8217;s and whose turn lies within 3&#176; of Drake&#8217;s. The gate admits a tip with at most two agreements "
     "fewer than the as-built tip."),
    ("Three-finger hold", "Grasps of the four cells (33) that keep the tool through the hold test&#8217;s force and torque ramps at "
     "\\(\\mu\\) = 0.5 with all three fingers loaded."),
    ("Hand tolerance &#964;", "The relative friction change at which the share of hand runs whose outcome changes (held flips or "
     "the turn moves more than 3&#176;) exceeds the chaotic floor by 5 points; 3.65&#8202;% (<code>tolerance/summary.json</code>)."),
    ("E*", "A print recipe&#8217;s effective modulus from the whole-tip finite-element model with a 4.76&#8202;mm magnet 5&#8202;mm "
     "deep."),
    ("RMS difference of two rollouts", "Two rollouts of one placement under two contact models, sampled every 20&#8202;ms: the root "
     "mean square over the samples of the angle between the two tool orientations (&#176;), of the difference of each finger&#8217;s "
     "summed normal force (N, three fingers pooled) and of the length of the difference of each finger&#8217;s friction force on the "
     "tool (N); <code>scripts/pad_builder_check.py</code>."),
    ("Spread between placements", "The same RMS difference between two rollouts of one hand under one contact model at two seeds "
     "(tool placements 2&#8202;mm and 2&#176; apart in standard deviation), median over the hand&#8217;s seed pairs and the eight "
     "hands; the scale below which two contact models cannot be told apart by these placements."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def question():
    return ("<p>The 1&#8202;mm pads model the printed TPU fingertip in every simulator the hand uses. This study measures how far "
            "they depart from the law and from Drake as the fingertip&#8217;s shape changes, and then uses the pads to choose a "
            "printable fingertip, its print recipe and its pad model for the hand&#8217;s tasks. The deterioration sweep runs the "
            "bed on the flat family (\\(r\\) = 0&#8211;7.4&#8202;mm), the 10.55&#8202;mm sphere tip and five CAD-tip members. The "
            "co-design screens 24 printable candidates of the CAD envelope, the as-built design surface and the simulation "
            "default in three stages: J1 at \\(\\mu\\) = 0.7 (26 tips), J2 over five friction and servo conditions with an agreement "
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
         f"\\(r\\) = 2.7&#8202;mm, {S['7.4']['base']['spin_over_law']['1']:.2f} times on the fully rounded block and "
         f"{sp['base']['spin_over_law']['1']:.2f} times on the sphere, while Drake stays within "
         f"{min(S[k]['drake']['spin_over_law']['1'] for _, k in fam):.2f}&#8211;"
         f"{max(S[k]['drake']['spin_over_law']['1'] for _, k in fam):.2f}. The braked swing needs only "
         f"{min(S[k]['base']['N45_over_drake'] for r, k in fam if r <= 3):.2f}&#8211;"
         f"{max(S[k]['base']['N45_over_drake'] for r, k in fam if r <= 3):.2f} of Drake&#8217;s pinch force for \\(r\\) &#8804; "
         "3&#8202;mm and the same force for \\(r\\) &#8805; 6&#8202;mm: the larger sliding torque brakes the swing earlier. The "
         "as-built design surface and its printed terraces behave like the \\(r\\) = 3&#8202;mm block.</p>")
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
              "tool through the tool. Pads touching the tool are coloured by normal force over \\(K\\) times penetration: white at 1, "
              "red at 10 or more. The pads at the ends of the contact line carry 10 or more times their spring force. Labels: the "
              "running spin arm from 0.4&#8202;rad and the law&#8217;s. <code>scripts/tip_codesign_film.py bed</code>.",
              key="bedfilm")
    return t


def rate_section():
    S = J("deterioration/summary.json")["members"]["2.7"]
    hf = J("deterioration/handfix_summary.json")
    fixk = "E1e+07_pf0.1_ir1000"
    eq = ("\\[f_n = \\frac{D_0}{1+\\mu_r^2}\\left(\\mu\\,\\lVert \\mathbf{a}_t + B\\,\\mathbf{v}_t \\rVert - j_n\\right)\\]")
    t = (f"<p>The pads&#8217; sliding torque grows with spin rate (Figure&#160;[[fig:rate]]). On the \\(r\\) = 2.7&#8202;mm block the "
         f"spin arm is {S['base']['spin_mm']['0.03']:.2f}{MM} at 0.03&#8202;rad/s, {S['base']['spin_mm']['1']:.2f}{MM} at "
         f"1&#8202;rad/s and {S['base']['spin_mm']['3']:.2f}{MM} at 3&#8202;rad/s, against the law&#8217;s "
         f"{S['law']['spin_1N_mm']:.2f}{MM}; Drake gives {S['drake']['spin_mm']['1']:.2f}{MM} at 1&#8202;rad/s. MuJoCo&#8217;s "
         "elliptic cone sets the normal force of a sliding contact as</p>"
         f"<div class='eq'>{eq}</div>"
         "<p>with \\(D_0\\) the contact&#8217;s regularisation, \\(\\mu_r\\) its regularised friction, \\(\\mathbf{a}_t\\) and \\(\\mathbf{v}_t\\) "
         "the tangential acceleration and velocity of the contact, \\(B\\) the reference damping and \\(j_n\\) the normal reference "
         "acceleration (<code>engine_core_constraint.c</code>). The slip term \\(\\mu\\lVert B\\mathbf{v}_t\\rVert\\) acts as a phantom "
         "penetration \\(\\zeta t_r \\mu v_\\mathrm{slip}\\), about 120&#8202;&#181;m at 6&#8202;mm from the axis at 1&#8202;rad/s, and "
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
    t += img("media/fig_rate.png", "Spin arm at 1&#8202;N against spin rate on the \\(r\\) = 2.7&#8202;mm block: pads as built, pads with "
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
               "Quasi-static pads&#8217; twist-onset arm over the law at 1&#8202;N (\\(\\psi\\) = 0) by builder variant, and the ripple of "
               "the 1&#8202;mm pads&#8217; arm as the tool line moves 0&#8211;1&#8202;mm along the finger (max minus min over the "
               "mean). <code>deterioration/builders.jsonl</code>.", key="builder")
    t += (f"<p>On the hand the local foundation depth with per-pad stiffness (builder v2) agrees with Drake on "
          f"{v2['E1e+07_hlocal_Keach']['within3_of_drake']} of 24 plan replays of the as-built tip as printed, against "
          f"{v2['E1e+07']['within3_of_drake']} for builder v1; {v2['v1_vs_v2_changed_replays']} of 40 replays differ by more "
          "than 3&#176; between the two, the same share as any 0.1&#8202;% friction change.</p>")
    if fb:
        t += builders_finalists(fb)
    return t + builder_default()


def builder_default():
    """The trajectory-level check of builder v2 as the default (builder_default/summary.json, 2026-10-15)."""
    S = J("builder_default/summary.json")
    if not S or "replay" not in S:
        return ""
    groups = [(w, tip) for tip in ("tpu2.7", "tcd_rs3_re3_p0.2") for w in ("replay", "hom") if tip in S.get(w, {})]
    tname = {"tpu2.7": "Default block", "tcd_rs3_re3_p0.2": "As-built, printed"}
    wname = {"replay": "Replays, D1&#8211;D8 seeds 0&#8211;4", "hom": "HOM, D1&#8211;D8 seeds 0&#8211;2"}
    med = lambda res, pair, k: res["pairs"][pair][k]["median"]  # noqa: E731
    body = []
    for w, tip in groups:
        res = S[w][tip]
        iq = lambda pair: (f"{med(res, pair, 'ang_rms_deg'):.2f} [{res['pairs'][pair]['ang_rms_deg']['q25']:.2f}&#8211;"  # noqa: E731
                           f"{res['pairs'][pair]['ang_rms_deg']['q75']:.2f}]")
        ta = res["tally"]
        body.append([tname[tip], wname[w], iq("v1-drake"), iq("v2-drake"),
                     f"{res['seed_spread']['drake']['ang_rms_deg']['median']:.2f}",
                     f"{med(res, 'v1-drake', 'F_rms_N'):.3f}", f"{med(res, 'v2-drake', 'F_rms_N'):.3f}",
                     f"{med(res, 'v1-drake', 'T_rms_N'):.3f}", f"{med(res, 'v2-drake', 'T_rms_N'):.3f}",
                     f"{ta['v1']['within_of_drake']}/{ta['v1']['n_vs_drake']}", f"{ta['v2']['within_of_drake']}/{ta['v2']['n_vs_drake']}"])
    r0 = S["replay"]["tpu2.7"]
    closer = [(w, tip) for w, tip in groups if med(S[w][tip], "v2-drake", "ang_rms_deg") < med(S[w][tip], "v1-drake", "ang_rms_deg")]
    not_closer = [g for g in groups if g not in closer]
    t = ("<p>Builder v2 is the default of <code>reorient_backends.replace_tips</code> since 15 October for the default block and "
         "the CAD-envelope tips; <code>builder=\"v1\"</code> reproduces every earlier run. Held counts cannot show that two contact "
         "models give the same dynamics (owner, 2026-10-15), so the check before the change compares whole rollouts: the plan "
         "replays and the HOM turn of D1&#8211;D8 on the servo-limited plant under builders v1 and v2 and Drake, with the tool pose and "
         "each finger&#8217;s normal and friction force on the tool every 20&#8202;ms (<code>scripts/pad_builder_check.py</code>). "
         f"On the default block&#8217;s replays the tool orientation differs from Drake&#8217;s by a median RMS of "
         f"{med(r0, 'v2-drake', 'ang_rms_deg'):.2f}&#176; with builder v2 and {med(r0, 'v1-drake', 'ang_rms_deg'):.2f}&#176; with "
         f"v1, against {r0['seed_spread']['drake']['ang_rms_deg']['median']:.2f}&#176; between two placements in Drake; the normal "
         f"force by {med(r0, 'v2-drake', 'F_rms_N'):.3f} and {med(r0, 'v1-drake', 'F_rms_N'):.3f}&#8202;N and the friction force by "
         f"{med(r0, 'v2-drake', 'T_rms_N'):.3f} and {med(r0, 'v1-drake', 'T_rms_N'):.3f}&#8202;N (Table&#160;[[tab:bdefault]], "
         "Figure&#160;[[fig:bdefault]]). Builder v2 is the closer to Drake in tool orientation in "
         f"{len(closer)} of the {len(groups)} tip-and-task groups")
    if not_closer:
        t += " (not in " + ", ".join(f"{tname[tip].lower()} {('replays' if w == 'replay' else 'HOM')}" for w, tip in not_closer) + ")"
    t += ".</p>"
    t += table(["Tip", "Task", "Orientation, v1 (&#176;)", "Orientation, v2 (&#176;)", "Drake between placements (&#176;)",
                "Normal force, v1 (N)", "Normal force, v2 (N)", "Friction force, v1 (N)", "Friction force, v2 (N)",
                "Agree with Drake, v1", "Agree with Drake, v2"], body,
               "RMS difference of each rollout from Drake&#8217;s rollout of the same placement, median [interquartile range] over the "
               "placements, servo-limited plant, \(\mu\) = 1.0. Agree: end turn within 3&#176; (replays) or 5&#176; (HOM) of "
               "Drake&#8217;s with the same held outcome. <code>builder_default/summary.json</code>.", key="bdefault", text_cols=(0, 1))
    pp = r0["per_placement"]
    ex = max((k for k, v in pp.items() if "v1-drake" in v and "v2-drake" in v),
             key=lambda k: pp[k]["v1-drake"]["ang_rms_deg"] - pp[k]["v2-drake"]["ang_rms_deg"])
    t += img("media/fig_builder_default.png",
             "Builder v1, builder v2 and Drake on the servo-limited plant. Top: the default block&#8217;s replay on which v2 gains "
             f"most on v1 ({ex.replace(' s', ', seed ')}): tool rotation from its pose at the end of the grip, and the normal and friction "
             "forces of the three fingers on the tool. Bottom: every placement&#8217;s RMS difference from Drake (dots; bars the "
             "medians) and Drake&#8217;s own spread between placements (dashed). <code>scripts/pad_builder_check.py figure</code>.",
             "bdefault")
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
                 "against Drake (24), and HOM seeds 0&#8211;2 at \\(\\mu\\) = 1.0. <code>final/summary_builders.json</code>.",
                 key="buildfin")


def screen_section():
    S1 = J("screen/summary_stage1.json")
    if not S1:
        return "<p><i>[stage 1 pending]</i></p>"
    C = S1["candidates"]
    rank = sorted(((v["J1"] or -1, k) for k, v in C.items()), reverse=True)
    top = ", ".join(f"{tip_name(k)} at {J_:.3f}" for J_, k in rank[:8])
    law = {r["tip"]: r for r in rows("screen/fidelity/law.jsonl") if r["field"] == "drake"}
    sp = [(C[k]["fidelity"]["pads_spin_1N_1_mm"] / law[k]["spin_arm_mm"]["1"], k) for k in C if k in law and C[k]["fidelity"].get("pads_spin_1N_1_mm")]
    on = [C[k]["fidelity"]["pads_onset_1N_mm"] / law[k]["table"]["1|0"]["arm_mm"] for k in C if k in law and C[k]["fidelity"].get("pads_onset_1N_mm")]
    t = (f"<p>The grid crosses six side fillets (1, 2, 3, 4.5, 6 and 7.4&#8202;mm) with four end fillets (1, 3, 6 and "
         f"10.5&#8202;mm), all printed face up at 0.2&#8202;mm; with the as-built design surface and the simulation default it "
         f"has 26 tips. The eight highest J1 are {top} (Figure&#160;[[fig:screen]]); the simulation default scores "
         f"{C['tpu2.7']['J1']:.3f}. Across the 24 printed candidates the pads&#8217; twist-onset arm stays within "
         f"{min(on):.2f}&#8211;{max(on):.2f} of the law. The 1&#8202;rad/s spin arm runs from {min(sp)[0]:.2f} times the law on "
         f"{tip_name(min(sp)[1])} to {max(sp)[0]:.2f} times on {tip_name(max(sp)[1])}: side fillets of 2&#8202;mm or less raise it "
         "as on the flat family, and the 10.5&#8202;mm end fillet with side fillets of 4.5&#8202;mm or less brings it below the "
         "law.</p>")
    t += img("media/fig_screen.png", "Stage 1 over the printed grid (rows r<sub>side</sub>, columns r<sub>end</sub>): J1 at "
             "\\(\\mu\\) = 0.7, and the 1&#8202;mm pads against the law: spin arm at 1&#8202;rad/s and brake force at 45&#176;. Rows: "
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
         "by the plan&#8217;s rules are " + "; ".join(f"{tip_name(k)}, {(why[k][4:] if why[k].startswith('(') else why[k]).replace('<=', '&#8804;').replace('mu ', '&#956; ')}"
                                                   for k in fin)
         + ", and the as-built tip as the baseline.</p>")
    t += img("media/fig_stage2.png", "Stage 2: the stage score in each condition (markers) and J2, the minimum (bar), per tip; right, "
             "plan replays within 3&#176; of Drake with its held outcome, of 24 (dark: passes the gate; dashed: the gate). Rows: "
             "<code>screen/stage2/</code>.", key="stage2")
    t += film("media/replay_D7_s0_tile.mp4", "media/replay_D7_s0_tile_poster.jpg",
              "Plan replay of the deployed hand D7, seed 0, on the servo-limited plant at \\(\\mu\\) = 1.0 with the four stage-3 tips; "
              "pads touching the tool drawn red; label: hand, time and the tool axis&#8217;s tilt above the horizontal. "
              "<code>scripts/tip_codesign_film.py replay</code>.", key="replayfilm")
    t += table(["Tip", "J2", "Agree with Drake", "Gate", "Cost / as-built", "Three-finger share, &#956; 0.5"], body,
               "Stage-2 summary (<code>screen/summary_stage2.json</code>): cost is the median MuJoCo time per replay step over the "
               "as-built tip&#8217;s; three-finger share is the share of cell turn runs ending with all three fingers loaded.",
               key="stage2")
    return t


def confirm_section():
    S3 = J("final/summary_stage3.json")
    if not S3:
        return "<p><i>[stage 3 pending]</i></p>"
    C = S3["candidates"]
    base = C["tcd_rs3_re3_p0.2"]
    fin = [k for k in S3["finalists"] if k != "tcd_rs3_re3_p0.2"]
    best = max(fin, key=lambda k: C[k]["J3"])
    win = [k for k in fin if C[k].get("beats_base")]
    body = [[tip_name(k), f2(C[k]["J3"], 3)] + [f2(C[k]["J"].get(c), 3) for c in S3["conditions"]]
            + [f2(C[k].get("dJ3_vs_base"), 3), ("[" + f2(C[k]["dJ3_ci90"][0], 3) + ", " + f2(C[k]["dJ3_ci90"][1], 3) + "]")
               if "dJ3_ci90" in C[k] else "&#8211;", f"{C[k]['hold3_mu0.5']}/{C[k]['n_hold_mu0.5']}"] for k in S3["finalists"]]
    t = (f"<p>On fresh placements the as-built tip scores J3 = {base['J3']:.3f}. "
         + (f"{len(win)} finalist{'s' if len(win) > 1 else ''} beat{'' if len(win) > 1 else 's'} it" if win else
            f"No finalist beats it: the best, {tip_name(best)}, scores {C[best]['J3']:.3f}, a paired difference of "
            f"{C[best]['dJ3_vs_base']:+.3f} whose 90&#8202;% interval [{C[best]['dJ3_ci90'][0]:+.3f}, {C[best]['dJ3_ci90'][1]:+.3f}] "
            "includes zero")
         + ", and the two others score lower with intervals below zero (Figure&#160;[[fig:stage3]], Table&#160;[[tab:stage3]]). "
         "Each interval comes from 10,000 resamples that redraw the three seeds within each cell and the five seeds within each "
         "hand, the same draw for both tips, so the difference is paired by placement. Every finalist keeps three-finger holds "
         "at \\(\\mu\\) = 0.5 within two grasps of the as-built tip. The flat-family geometries tested here do not improve on the "
         "as-built fingertip for these tasks; (3,&#160;6), the same side fillet with a rounder end, is the one candidate whose "
         "interval reaches above +0.05.</p>")
    t += img("media/fig_stage3.png", "Stage 3 on fresh placements: the stage score per condition and J3, the minimum (left); the "
             "paired difference from the as-built tip with its 90&#8202;% bootstrap interval (right). <code>final/</code>.",
             key="stage3")
    ic = [r for r in rows("final/icem/turn.jsonl") if r.get("status") == "ok"]
    if ic:
        by = {(r["variant"], r["d_mm"]): r for r in ic}
        b16, b25 = by.get(("tcd_rs3_re3_p0.2", 16.0)), by.get(("tcd_rs3_re3_p0.2", 25.0))
        f16, f25 = by.get((best, 16.0)), by.get((best, 25.0))
        if b16 and b25 and f16 and f25:
            t += (f"<p>Pinned iCEM, the sampling planner that keeps three pads on the cylinder, gives the same order on two cells "
                  f"at seed 0 and \\(\\mu\\) = 1.0: the as-built tip turns the 16&#8202;mm cylinder {b16['turn_end_deg']:.1f}&#176; and "
                  f"the 25&#8202;mm one {b25['turn_end_deg']:.1f}&#176;, {tip_name(best)} {f16['turn_end_deg']:.1f}&#176; and "
                  f"{f25['turn_end_deg']:.1f}&#176;, every run held with all three fingers loaded at the end "
                  "(<code>final/icem/turn.jsonl</code>, <code>scripts/tip_codesign_rig.py icem</code>).</p>")
    t += film("media/hom_cylinder_16_0.625_s3_tile.mp4", "media/hom_cylinder_16_0.625_s3_tile_poster.jpg",
              "HOM on the CAD hand&#8217;s 16&#8202;mm cylinder cell (pair separations 62.5/72.5&#8202;mm), seed 3, \\(\\mu\\) = 0.7, the "
              "grasp each tip chose on seed 0; pads touching the cylinder drawn red. <code>scripts/tip_codesign_film.py hom</code>.",
              key="homfilm")
    cond = {"mu0.7_" + SERVO: "&#956; 0.7", "mu0.5_" + SERVO: "&#956; 0.5", "mu0.7_kp0.5_kv0_fr0.18_fl0_dp0.0144": "kp 0.5",
            "mu0.7_kp0.72_kv0_fr0.12_fl0_dp0.0144": "0.12&#8202;N&#8202;m"}
    t += table(["Tip", "J3"] + [cond.get(c, c) for c in S3["conditions"]] + ["J3 &#8722; as-built", "90&#8202;% interval",
                                                                          "Three-finger hold, &#956; 0.5"], body,
               "Stage-3 scores on fresh placements (<code>final/summary_stage3.json</code>).", key="stage3")
    return t


RECIPES = {"w95_20": ("95A, 20&#8202;% gyroid", 14.6), "w95_10": ("95A, 10&#8202;% gyroid", 7.2),
           "w85_30": ("85A, 30&#8202;% gyroid", 6.9), "w85_50": ("85A, 50&#8202;% gyroid", 16.5)}


def _travel(r, N="1"):
    """Magnet travel along the pinch normal per newton at force N (um/N) of a tpu_tip_magnet row."""
    return -r["per_force"][N]["magnet_t_um"][0] / float(N)


def recipe_choice():
    """Per finalist: the FE rows, the sensing rule (magnet travel >= 12 um/N at 1 N) and the chosen recipe; the
    as-built tip's check uses the 2026-10-09 FE of the CAD insert (docs/experiments/20261009-tpu_tip_fem/magnet_runs.jsonl)."""
    S3 = J("final/stage3_tips.json") or {"finalists": []}
    fe = {}
    for r in rows("final/fe.jsonl"):
        fe[(r.get("tip"), r["recipe"])] = r
    ref = {}
    for l in open(os.path.join(ROOT, "docs/experiments/20261009-tpu_tip_fem/magnet_runs.jsonl")):
        r = json.loads(l)
        if str(r.get("magnet", "")).startswith("d4.76") and abs(float(r.get("depth_mm") or 0) - 5) < 1e-6:
            ref[r["recipe"]] = r
    out = {}
    for k in S3["finalists"]:
        cand = {rc: fe[(k, rc)] for rc in RECIPES if (k, rc) in fe}
        src = "FE of this tip"
        if not cand and k == "tcd_rs3_re3_p0.2":
            cand, src = {rc: ref[rc] for rc in ref}, "FE of the CAD insert (2026-10-09)"
        tr = {rc: _travel(r) for rc, r in cand.items()}
        ok = [rc for rc in ("w95_20", "w85_50", "w95_10", "w85_30") if rc in tr and tr[rc] >= 12.0]
        est = {}
        if "w95_20" in tr and not ok and "w95_10" not in tr:     # the next recipe by the CAD insert's ratio
            est["w95_10"] = tr["w95_20"] * _travel(ref["w95_10"]) / _travel(ref["w95_20"])
        out[k] = {"travel": tr, "E": {rc: r["E_best_MPa"] for rc, r in cand.items()},
                  "approach": {rc: r["per_force"]["1"]["delta_um"] for rc, r in cand.items()}, "estimate": est,
                  "choice": ("w95_20" if "w95_20" in ok else (ok[0] if ok else (next(iter(est)) if est else None))),
                  "src": src}
    return out


def material_section():
    ch = recipe_choice()
    if not ch:
        return "<p><i>[FE pending]</i></p>"
    body = []
    for k, v in ch.items():
        for rc in sorted(v["travel"], key=lambda x: -RECIPES[x][1]):
            body.append([tip_name(k), RECIPES[rc][0], f2(v["E"][rc], 1), f2(v["approach"][rc], 1), f2(v["travel"][rc], 1),
                         "chosen" if rc == v["choice"] else "", v["src"]])
        for rc, x in v["estimate"].items():
            body.append([tip_name(k), RECIPES[rc][0], "&#8211;", "&#8211;", f2(x, 1), "chosen" if rc == v["choice"] else "",
                         "estimate: this tip&#8217;s 95A 20&#8202;% travel times the CAD insert&#8217;s 10&#8202;%/20&#8202;% ratio"])
        if not v["travel"]:
            body.append([tip_name(k), "&#8211;", "&#8211;", "&#8211;", "&#8211;", "", "no FE row (Future work)"])
    t = ("<p>The print recipe sets the tip&#8217;s stiffness, and the stiffness sets how far the embedded magnet moves per "
         "newton, which is what the MLX90393 sensor reads. The sensor resolves about 1.2&#8202;&#181;m of magnet motion, so a "
         "force resolution of 0.1&#8202;N needs at least 12&#8202;&#181;m/N of travel (sensing rule). Each finalist&#8217;s "
         "as-printed body is meshed with the 4.76&#8202;mm magnet 5&#8202;mm under the palmar face and loaded at 0.5, 1 and "
         "3&#8202;N on the pinch line in the whole-tip finite-element model of <code>scripts/tpu_tip_magnet.py</code> "
         "(<code>scripts/tip_codesign_fe.py</code>). The stiffest recipe that meets the rule is chosen "
         "(Table&#160;[[tab:fe]]).</p>")
    t += table(["Tip", "Recipe", "E*, MPa", "Approach at 1&#8202;N, &#181;m", "Magnet travel, &#181;m/N", "Choice", "Source"], body,
               "Finite-element stiffness and magnet travel per recipe (<code>final/fe.jsonl</code>).", key="fe", text_cols=(0, 1, 5, 6))
    soft = [(k, v) for k, v in ch.items() if v["choice"] == "w95_10" and "w95_20" in v["travel"] and "w95_10" in v["travel"]]
    if soft:
        names = " and ".join(tip_name(k) for k, _ in soft)
        t += (f"<p>At 95A 20&#8202;% the {names} tips move the magnet "
              + " and ".join(f"{v['travel']['w95_20']:.1f}" for _, v in soft) + "&#8202;&#181;m/N, below the rule, and at 95A "
              "10&#8202;% " + " and ".join(f"{v['travel']['w95_10']:.1f}" for _, v in soft) + "&#8202;&#181;m/N, so they print "
              "at 10&#8202;% infill.</p>")
    return t


def print_section():
    import glob
    body = []
    for f in sorted(glob.glob(os.path.join(D, "print/*/2026101*_w*.json"))):
        r = json.load(open(f))
        sl = r.get("slice", {})
        body.append([tip_name(r["tip"]) if r["tip"] in ("tcd_rs3_re3", "tcd_rs3_re3_p0.2") else tip_name(r["tip"] + "_p0.2" if "_p" not in r["tip"] else r["tip"]),
                     RECIPES.get(r["recipe"], (r["recipe"],))[0], "yes" if r.get("watertight") else "no", f2(r.get("part_volume_mm3"), 0),
                     f2(sl.get("pause_found_before_layer_z"), 1), str(sl.get("islands_max")), f2(100 * sl.get("supported_area", 0), 0),
                     sl.get("time", "&#8211;"), f"<code>{os.path.relpath(f, D)[:-5]}.3mf</code>"])
    if not body:
        return "<p><i>[print files pending]</i></p>"
    return ("<p>Each finalist is written face up with its magnet pocket as an inner void, as a PrusaSlicer project on the "
            "owner&#8217;s v17 template (TPU on extruder 2, two perimeters, gyroid infill at the recipe&#8217;s density, four top "
            "and four bottom layers) with an M601 pause before the first layer above the pocket, and sliced to check it "
            "(Table&#160;[[tab:print]]). Pad scenes of each finalist for the D6 bench replay and the CAD hand are in "
            "<code>assets/mjcf/experimental/20261014-tip_codesign/&lt;tip&gt;/</code>, and "
            "<code>scripts/tip_codesign_rig.py rlscene --tip &lt;tip&gt; --src &lt;run&gt; --out-dir &lt;new run&gt;</code> writes a "
            "run&#8217;s frozen RL scene with the finalist&#8217;s pads.</p>"
            + table(["Tip", "Recipe", "Watertight", "Part volume, mm&#179;", "Pause at, mm", "Islands per layer, max",
                     "Supported area, %", "Print time", "Project"], body,
                    "Print files and slice audit (<code>print/&lt;design&gt;/</code>; G-code in "
                    "<code>logs/20261014-tip_codesign/print/</code>). Bounds 21.41&#8202;&#215;&#8202;14.8&#8202;&#215;&#8202;17.0&#8202;mm "
                    "for every tip.", key="print", text_cols=(0, 1, 2, 7, 8)))


def rs_re(k):
    m = re.match(r"tcd_rs([0-9.]+)_re([0-9.]+)", k)
    return m[1], m[2]


def lede():
    S = J("deterioration/summary.json")["members"]
    S3 = J("final/summary_stage3.json")["candidates"]
    b = S3["tcd_rs3_re3_p0.2"]
    best = max((k for k in S3 if k != "tcd_rs3_re3_p0.2"), key=lambda k: S3[k]["J3"])
    lo, hi = S3[best]["dJ3_ci90"]
    return (f"The 1&#8202;mm sphere pads reproduce the law&#8217;s twist-onset torque within 2&#8202;% on fillets of 1.5&#8202;mm or more "
            f"and 7&#8211;14&#8202;% high on sharper edges, but their sliding torque grows with spin rate: at 1&#8202;rad/s the spin arm "
            f"is {S['2.7']['base']['spin_over_law']['1']:.2f} times the law on the printed tip&#8217;s 2.7&#8202;mm fillet and "
            f"{S['0']['base']['spin_over_law']['1']:.2f} times on a sharp block, while Drake stays within 0.97&#8211;1.12. The excess "
            "comes from MuJoCo&#8217;s elliptic friction cone, and removing it breaks the hand&#8217;s closed-loop turns. Of 24 "
            "printable fingertip shapes screened on the SR2 hand&#8217;s tasks, none of the three finalists beats the as-built tip "
            f"on fresh placements: the best, with a {rs_re(best)[0]}&#8202;mm side and a {rs_re(best)[1]}&#8202;mm end fillet, scores "
            f"J3 = {S3[best]['J3']:.3f} against {b['J3']:.3f}, a difference whose 90&#8202;% interval runs from {lo:+.3f} to "
            f"{hi:+.3f}. Per-pad stiffness raises the pads&#8217; agreement with Drake on three of the four tips of stage 3.")


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
         "<code>scripts/tpu_tip_calibrate.py fit --pull</code>; every task here varies \\(\\mu\\) over 0.5&#8211;1.0."),
        ("A longer stage 3 for (3,&#160;6)", "rerun <code>scripts/tip_codesign_stage3_queue.py</code> jobs for "
         "<code>tcd_rs3_re6_p0.2</code> and the as-built tip with cell seeds 3&#8211;12 and HOM seeds 5&#8211;19; a 90&#8202;% "
         "interval above zero makes the 6&#8202;mm end fillet the printed default, an interval that still includes zero keeps "
         "the as-built tip."),
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
