#!/usr/bin/env python3
"""Page: capsule, half-ellipsoid and quarter-ellipsoid fingertips against the printed block on the SR2 hand (study
20261015-curved_tips; hand-off docs/handoff/20261015-curved_tips.md; protocol of docs/notes/20261014-tip_codesign_plan.md).

Data: docs/experiments/20261015-curved_tips/ (bed/, cost/, traj/, screen/, final/, cell10/, stiffness/, qell/, fe/,
nail/, print/, media/), every number read from the study's summaries:

    .venv/bin/python scripts/curved_tips_analysis.py bed cost screen stage3 cell10 qell   (one at a time)
    .venv/bin/python scripts/curved_tips_traj_analysis.py
    .venv/bin/python scripts/curved_tips_stiffness.py summary
    .venv/bin/python scripts/curved_tips_figures.py shapes bed traj screen stage3 stiffness
    .venv/bin/python scripts/curved_tips_page.py
Out: docs/experiments/20261015-curved_tips/20261015-capsule_and_ellipsoid_fingertips.html (media inlined).
"""
from __future__ import annotations

import base64
import glob
import io
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import curved_tips_common as C  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261015-curved_tips")
OUT = os.path.join(D, "20261015-capsule_and_ellipsoid_fingertips.html")
TPL = os.path.join(ROOT, "scripts/curved_tips_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
FIG, TAB, REFS = [0], [0], {}
SERVO = C.SERVO
EXTRA_CSS = """
.tw th{white-space:normal !important;vertical-align:bottom;text-transform:none}
.tw td:first-child{min-width:9em}
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
figure.films video{width:100%;height:auto}
figure.img img{width:100%;height:auto}
.eq{margin:.8em 0;text-align:center}
""" + texsvg.BLOCK_CSS + "\n"
MM, UM, NN, DEG = "&#8202;mm", "&#8202;&#181;m", "&#8202;N", "&#176;"
CURVED = C.CURVED                                       # the six printed capsules and half-ellipsoids
QUART = C.CURVED_Q
BLOCKS = C.BLOCKS


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


def img(rel, caption, key=None):
    p = os.path.join(D, rel)
    if not os.path.exists(p):
        return f"<p><i>[figure {rel} not drawn yet]</i></p>"
    return figure(f'<img alt="" src="{data_uri(p)}">', caption, key)


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


def f(x, n=2):
    return "&#8211;" if x is None else f"{x:.{n}f}"


def pct(vals):
    return rng([100 * x for x in vals], 0, "&#8202;%")


def rng(vals, n=2, unit=""):
    v = [x for x in vals if x is not None]
    lo, hi = min(v), max(v)
    s = f"{lo:.{n}f}" if f"{lo:.{n}f}" == f"{hi:.{n}f}" else f"{lo:.{n}f}&#8211;{hi:.{n}f}"
    return s + unit


def name(k):
    """Sentence-case table name of a tip key."""
    s = C.label(k).replace(" mm", "&#8202;mm").replace(" x ", "&#8202;&#215;&#8202;")
    s = s.replace(", printed", "").replace("half-ellipsoid", "Half-ellipsoid").replace("quarter-ellipsoid", "Quarter-ellipsoid")
    s = s.replace("capsule", "Capsule").replace("as-built block", "As-built block").replace("block (", "Block (")
    s = s.replace("sphere R", "Sphere R").replace(", design surface", "")
    return s


# ------------------------------------------------------------------------------------------ data

BED = (J("bed/summary.json") or {}).get("tips", {})
COST = (J("cost/summary.json") or {}).get("tips", {})
TRAJ = (J("traj/summary.json") or {}).get("tips", {})
SCR = J("screen/summary.json") or {}
FIN = J("final/summary.json") or {}
CELL = J("cell10/summary.json") or {}
STF = (J("stiffness/summary.json") or {}).get("tips", {})
QL = J("qell/summary.json") or {}


def bed_v(k, *path):
    d = BED.get(k)
    for p in path:
        if d is None:
            return None
        d = d.get(p)
    return d


# ------------------------------------------------------------------------------------------ sections

GLOSSARY = [
    ("Tips", "Every tip fills the printed block&#8217;s envelope, 17.0&#8202;mm deep along the palmar normal \\(x\\), 14.8&#8202;mm "
     "wide along the PIP axis \\(y\\) and 21.41&#8202;mm long along the finger \\(z\\), with its palmar apex over the bench "
     "grasp&#8217;s contact point \\((y, z) = (0, 0)\\) (Figure&#160;[[fig:shapes]]). Capsule <i>r</i>: a cylinder of radius <i>r</i> "
     "along the finger capped by a hemisphere at the distal end (<code>tcd_cap&lt;r&gt;</code>). Half-ellipsoid "
     "\\(a_y \\times a_x \\times a_z/a_{zp}\\): the palmar half of an ellipsoid with semi-axes \\(a_y\\) across, \\(a_x\\) deep and "
     "\\(a_z\\) distal, \\(a_{zp}\\) proximal of the apex, its section running straight back to the back face behind the equator "
     "(<code>tcd_ell</code>). Quarter-ellipsoid: the same palmar half cut flat at the equator, \\(a_x\\) deep, with a flat back "
     "for the five-chip magnetometer board (<code>tcd_qell</code>). The as-built block has side and end fillets of 3 and "
     "3&#8202;mm, the block finalist of 14 October 3 and 6&#8202;mm (<code>tcd_rs3_re3</code>, <code>tcd_rs3_re6</code>)."),
    ("Printed surface", "Capsules and half-ellipsoids print upright, the mount face on the bed, in 0.1&#8202;mm layers "
     "(<code>_u0.1</code>); quarter-ellipsoids are simulated as printed upright too; blocks print palmar face up in 0.2&#8202;mm "
     "layers (<code>_p0.2</code>). Each simulated tip is the convex hull of its printed layer outlines."),
    ("1&#8202;mm pads", "The tip&#8217;s palmar region as spheres of radius 0.75&#8202;mm on a 1&#8202;mm lattice, each a MuJoCo soft "
     "contact of stiffness \\(K_i = E a_i / h_i\\) with \\(E\\) = 10&#8202;MPa, \\(a_i\\) the pad&#8217;s share of the surface and "
     "\\(h_i\\) its foundation depth; relaxation time 0.02&#8202;s, impratio 100, 1&#8202;ms step. Builder v1: one \\(K\\) per finger, "
     "\\(h\\) = 8.5&#8202;mm. Builder v2 (the repository default since 15 October): \\(h_i\\) the pad&#8217;s normal distance to the "
     "tip&#8217;s volume centroid; on a quarter-ellipsoid the TPU between the pad and the rigid back plane, "
     "\\(h_i = (p_x - x_c)/\\max(n_x, 0.2)\\), at least 1&#8202;mm (<code>tcd_qellc</code>: the centroid depth instead)."),
    ("Drake", "Drake 1.57 hydroelastic contact: the tip&#8217;s compliant convex hull (0.25&#8202;mm mesh), modulus 10&#8202;MPa, "
     "Hunt&#8211;Crossley dissipation 10&#8202;s/m, relaxation time 0.01&#8202;s, SAP solver, 1&#8202;ms step: the dynamic reference."),
    ("Law", "The quasi-static integral of Drake&#8217;s pressure field \\(p = E\\,(1 - g(\\mathbf{x}))\\), \\(g\\) the gauge of the "
     "tip&#8217;s hull about its volume centroid, over the contact with a rigid indenter (<code>scripts/curved_tips_law.py</code>)."),
    ("Bed", "Two force-controlled rails squeeze the bed&#8217;s 25&#8202;mm diameter screwdriver between two tips at 1&#8202;N per "
     "tip, friction coefficient \\(\\mu\\) = 1. Approach: rail travel from first touch (&#181;m). Twist-onset arm: the torque at which "
     "the tool starts to turn under a slow torque ramp, over \\(2\\mu N\\) (mm). Spin arm: the mean torque that drives the tool about "
     "the pinch axis at rate \\(\\omega\\), over \\(2\\mu N\\) (mm). Twist creep: the turn rate under half the onset torque "
     "(&#176;/s); rigid Coulomb friction gives zero."),
    ("Servo-limited plant", "The SR2 servo model, gain 0.72, torque limit 0.18&#8202;N&#8202;m, damping 0.0144 "
     "(<code>kp0.72_kv0_fr0.18_fl0_dp0.0144</code>)."),
    ("D1&#8211;D8 replays", "The eight deployed bench plans replayed open loop on the servo-limited plant at \\(\\mu\\) = 1.0 "
     "with seeds 0&#8211;2 (tool placement drawn with standard deviations 2&#8202;mm and 2&#176;; seed 0 none), 24 per tip."),
    ("RMS tool-angle difference", "Two rollouts of one placement under two contact settings, sampled every 20&#8202;ms up to the "
     "shorter&#8217;s end: the root mean square of the angle between the two tool orientations (&#176;). Pairs: the pads at the "
     "planner&#8217;s numerics (1&#8202;ms, impratio 100) against Drake; the RL trainer&#8217;s numerics (2&#8202;ms, impratio 10, "
     "10 solver iterations) against the planner&#8217;s; and Drake against Drake at two seeds of one hand (the spread between "
     "placements)."),
    ("HOM, cells", "HOM: the hand&#8217;s closed-loop in-hand turn controller (<code>hom_turn3</code>, model gain 0.72). Cells: four "
     "grasps of the CAD hand, cylinders of 25, 16, 10 and 16&#8202;mm diameter on four finger layouts, with 6&#8211;9 candidate "
     "grasps each. Held turn: the turn at the end of a rollout when the tool is held by two or more fingers, otherwise 0 (&#176;)."),
    ("Stage scores", "The stage score \\(J\\) is the mean of five parts, each a held turn over 90&#176;: per cell, the mean held "
     "turn of the grasp chosen by its held turn on one seed, on the scoring seeds; and the mean held turn of HOM on D1&#8211;D8. "
     "J1: at \\(\\mu\\) = 0.7 on the servo-limited plant. J2: the minimum over five conditions (\\(\\mu\\) = 0.7, 1.0, 0.5; servo gain "
     "0.5; torque limit 0.12&#8202;N&#8202;m). J3: the minimum over the four conditions other than \\(\\mu\\) = 1.0 on fresh "
     "placements only (cell seeds 3&#8211;5, HOM seeds 5&#8211;9)."),
    ("Agreement gate", "The D1&#8211;D8 replays whose pads end within 3&#176; of Drake&#8217;s turn with Drake&#8217;s held outcome, "
     "of 24; a tip passes with at most two fewer than the as-built block."),
    ("Probe map", "A rigid frictionless probe moved along \\(-x\\) from first touch of the design surface: a sphere of radius "
     "2&#8202;mm at the apex, across the width at \\(y\\) = 2.6, 4.1 and 5.2&#8202;mm and along the finger at \\(z\\) = +4 "
     "(proximal), &#8722;4 and &#8722;7&#8202;mm (distal); and the bed&#8217;s tool (radius 12.5&#8202;mm, axis along \\(y\\)) at "
     "\\(z\\) = 0, +4 and &#8722;4&#8202;mm. Loads 0.5, 1 and 3&#8202;N. Local stiffness: \\(\\mathrm{d}F/\\mathrm{d}\\delta\\) at the "
     "load (N/mm). Load exponent \\(n\\): \\(\\delta \\propto F^{\\,n}\\) between 0.5 and 3&#8202;N; a Hertz contact on a half-space "
     "gives 2/3."),
    ("Finite elements", "The printed tip as a linear elastic model (<code>scripts/tpu_tip_magnet.py</code>): two 0.45&#8202;mm "
     "perimeters and four 0.2&#8202;mm skins of solid TPU 95A (30&#8202;MPa, Poisson ratio 0.45) over a homogenized 20&#8202;% "
     "gyroid core in the print orientation, the 4.76&#8202;&#215;&#8202;1.59&#8202;mm NdFeB magnet in its pocket, the mount face and "
     "the back face (or a quarter-ellipsoid&#8217;s back plane) fixed; frictionless contact along \\(x\\) on the compliance of the "
     "palmar face nodes."),
    ("Magnet travel, force noise", "The magnet&#8217;s displacement along \\(x\\) per newton of normal load at the apex "
     "(&#181;m/N), and the Cram&#233;r&#8211;Rao bound of the normal force from one 141&#8202;Hz sample of the five MLX90393 chips "
     "with the contact point known (N)."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def question():
    return ("<p>The owner asked on 15 October whether shapes far from the block, a capsule or an anthropomorphic "
            "half-ellipsoid, would grasp and turn as well, cost less to simulate, and keep the contact dynamics the same from "
            "RL training through planning to evaluation. A quarter-ellipsoid with a flat back for the magnetometer board and "
            "a fingernail followed the same afternoon. Three capsules, three half-ellipsoids and three quarter-ellipsoids "
            "inside the block&#8217;s envelope (Figure&#160;[[fig:shapes]]) went through the 14 October protocol: the contact "
            "bed against the law and Drake, the cost of a simulation step on the hand, trajectory agreement of the "
            "D1&#8211;D8 replays under three contact settings, the three-stage task screen with fresh placements in stage 3, and "
            "print files. A probe map against finite elements of the printed tips measures how well each contact model "
            "reproduces the tips&#8217; stiffness. The references are the printed as-built block and the block finalist of "
            "14 October.</p>"
            + img("media/fig_shapes.png", "The tips&#8217; design surfaces: for each, the profile along the finger at \\(y\\) = 0 "
                  "(distal end left, palmar side up) and the section across it at the contact point, with the magnet "
                  "(grey) and the contact point (red). Quarter-ellipsoids end at their flat back, 8.5 or 5&#8202;mm under the apex.",
                  key="shapes"))


def bed_section():
    cur = [k for k in CURVED if k in BED]
    q = [k for k in QUART if k in BED]
    R = lambda ks, key: [bed_v(k, "ratios", key) for k in ks]  # noqa: E731
    law_on = [bed_v(k, "law", "onset_arm_mm", "1") for k in cur]
    blk_on = [bed_v(k, "law", "onset_arm_mm", "1") for k in BLOCKS]
    t = (f"<p>The curved tips turn the tool on a third of the block&#8217;s friction arm. The law&#8217;s twist-onset arm at "
         f"1&#8202;N is {rng(law_on)}&#8202;mm on the capsules and half-ellipsoids against {rng(blk_on)}&#8202;mm on the "
         f"blocks. Builder-v2 pads give {rng(R(cur, 'pads_approach_1N_over_law'))} of the law&#8217;s approach, "
         f"{rng(R(cur, 'pads_onset_1N_over_law'))} of its onset arm and {rng(R(cur, 'pads_spin_1_over_law'))} of its spin arm "
         f"at 1&#8202;rad/s, where the blocks&#8217; pads give {rng(R(BLOCKS, 'pads_spin_1_over_law'))}; their spin arm grows "
         f"by a factor {rng(R(cur, 'pads_spin_3_over_0.3'))} from 0.3 to 3&#8202;rad/s against {rng(R(BLOCKS, 'pads_spin_3_over_0.3'))} "
         f"on the blocks (Figure&#160;[[fig:bed]]). Drake lies within {100 * max(abs(x - 1) for x in R(cur, 'drake_onset_1N_over_law')):.1f}&#8202;% "
         f"of the law&#8217;s onset arm and {rng(R(cur, 'drake_spin_1_over_law'))} of its spin arm.</p>"
         f"<p>Fewer pads touch: {rng([bed_v(k, 'pads', 'touching_1N') for k in cur], 0)} per side at 1&#8202;N against "
         f"{rng([bed_v(k, 'pads', 'touching_1N') for k in BLOCKS], 0)} on the blocks. The pads&#8217; twist creep is "
         f"{rng([bed_v(k, 'pads', 'twist_creep_1N_deg_s') for k in cur])}&#8202;&#176;/s against "
         f"{rng([bed_v(k, 'pads', 'twist_creep_1N_deg_s') for k in BLOCKS])}&#8202;&#176;/s on the blocks and "
         f"{rng([bed_v(k, 'drake', 'twist_creep_1N_deg_s') for k in cur], 3)}&#8202;&#176;/s in Drake.</p>")
    if q:
        qc = C.QUARTER_C + C.PRINT
        t += (f"<p>On the quarter-ellipsoids the back-plane depth makes the pads {rng(R(q, 'pads_approach_1N_over_law'))} of "
              f"the law&#8217;s approach. The law measures depth to the volume centroid, which lies about 3.2&#8202;mm above "
              f"the back plane of an 8.5&#8202;mm quarter-ellipsoid; with the centroid depth the same shape&#8217;s pads give "
              f"{f(bed_v(qc, 'ratios', 'pads_approach_1N_over_law'))} of it and {f(bed_v(qc, 'ratios', 'pads_onset_1N_over_law'))} "
              f"of its onset arm. Drake follows the law here by construction; the finite elements of "
              f"Section&#160;[[sec:stiffness]] measure which depth the printed tip has.</p>")
    t += img("media/fig_bed.png", "Bed per tip: the twist-onset arm at 1&#8202;N (law, pads, Drake), the spin arm at 1&#8202;rad/s "
             "over the law, and the spin arm at 3&#8202;rad/s over the arm at 0.3&#8202;rad/s (<code>bed/summary.json</code>). "
             "Separators: blocks, capsules and half-ellipsoids, quarter-ellipsoids, and the sphere references.", key="bed")
    t += film("media/bed_spin_all_tile.mp4", "media/bed_spin_all_tile_poster.jpg",
              "The bed spin at 1&#8202;N and 1&#8202;rad/s with builder-v2 pads, seen from the tool side through the tool: each "
              "pad touching the tool coloured by its normal force over \\(K_i\\) times its penetration (white 1, red 10 or more); "
              "4&#215; slow motion.", key="bedfilm")
    return t


def cost_section():
    tips = BLOCKS[:1] + CURVED + QUART
    body = []
    for k in tips:
        v = COST.get(k)
        if not v:
            continue
        a, b, g = v["cpu"]["cylinder:16:0.625|pads"], v["cpu"]["cylinder:25:0.125|pads"], v.get("gpu", {})
        body.append([name(k), f(a["step_us"], 0), f(a["ncon"], 0), f(a["solver_niter"], 2), f(b["step_us"], 0),
                     f(a["replan_3thread_s"], 2), f(g.get("us_per_world_step"), 2), f(g.get("contacts_per_world"), 0),
                     f((g.get("gpu_peak_mb") or 0) / 1e3, 1)])
    base = COST[C.ASBUILT]
    cur = [k for k in CURVED + QUART if k in COST]
    gr = [COST[k]["ratios"]["gpu"] for k in cur]
    return (f"<p>The curved tips cost the GPU {rng(gr)} of the block&#8217;s world-step at the held RL state "
            f"({rng([COST[k]['gpu']['us_per_world_step'] for k in cur])} against "
            f"{f(base['gpu']['us_per_world_step'])}&#8202;&#181;s at 2048 worlds), because fewer pads touch the tool. On the CPU "
            f"the three-finger grip takes {rng([COST[k]['cpu']['cylinder:16:0.625|pads']['step_us'] for k in cur], 0)}&#8202;&#181;s "
            f"per step on the 16&#8202;mm cell against {f(base['cpu']['cylinder:16:0.625|pads']['step_us'], 0)}&#8202;&#181;s, since "
            f"fewer contacts need more solver iterations (Table&#160;[[tab:cost]]).</p>"
            + table(["Tip", "CPU step, 16&#8202;mm cell, &#181;s", "Contacts", "Solver iterations", "CPU step, 25&#8202;mm cell, &#181;s",
                     "Replanning 64 plans, 3 threads, s", "GPU world-step, &#181;s", "GPU contacts per world", "GPU memory, GB"],
                    body, "Cost on the hand by the pad-usability protocol: MuJoCo at the servo-limited plant&#8217;s three-finger "
                    "grip on two CAD-hand cells (builder-v2 pads), and MuJoCo-Warp at the held state of an RL run with 2048 "
                    "worlds (<code>cost/summary.json</code>).", key="cost"))


def traj_section():
    def pd(k, pair="plan-drake", key="rms_dang_deg"):
        v = TRAJ.get(k, {}).get("pairs", {}).get(pair, {})
        return (v.get(key) or {}).get("median") if isinstance(v.get(key), dict) else v.get(key)
    best = min((k for k in CURVED if k in TRAJ), key=lambda k: pd(k))
    bad = [k for k in CURVED if k in TRAJ and pd(k) > 6]
    rl = [pd(k, "rl-plan") for k in CURVED + BLOCKS if k in TRAJ and k not in bad]
    spread = [TRAJ[k]["spread"]["drake"]["rms_dang_deg"]["median"] for k in CURVED + BLOCKS if k in TRAJ]
    t = (f"<p>Along the D1&#8211;D8 replays the {name(best).lower()} stays closest to Drake: a median RMS tool-angle difference "
         f"of {f(pd(best))}&#176; with {TRAJ[best]['pairs']['plan-drake']['turn_within_3']} of 24 end turns within 3&#176;, against "
         f"{f(pd(C.ASBUILT))}&#176; and {TRAJ[C.ASBUILT]['pairs']['plan-drake']['turn_within_3']} of 24 for the as-built block "
         f"(Figure&#160;[[fig:traj]]). On the tips whose plans keep the tool under both models, the RL trainer&#8217;s numerics "
         f"differ from the planner&#8217;s by {rng(rl)}&#176;, as much as the contact model does, and two seeds of one hand differ "
         f"by {rng(spread)}&#176; in Drake. On "
         + ", ".join(name(k).lower() for k in bad[:-1]) + " and " + name(bad[-1]).lower() + f" the pads and Drake differ by {rng([pd(k) for k in bad], 1)}&#176;, "
         "because the bench plans drop the tool under one model and keep it under the other.</p>")
    qq = [k for k in QUART if k in TRAJ]
    if qq:
        t += ("<p>The quarter-ellipsoids differ from Drake by " + ", ".join(f"{f(pd(k), 1)}&#176;" for k in qq) + " ("
              + ", ".join(name(k).lower() for k in qq) + "): the bench plans, made for the block, drop the tool from the two "
              "11/22&#8202;mm tips under both models at different times.</p>")
    t += img("media/fig_traj.png", "Trajectory differences per tip, median over the 24 replays with the 25&#8211;75&#8202;% range: "
             "RMS tool-angle and friction-force differences between the pads at the planner&#8217;s numerics and Drake, between "
             "the RL trainer&#8217;s numerics and the planner&#8217;s, and between two seeds of one hand in Drake "
             "(<code>traj/summary.json</code>).", key="traj")
    return t


def screen_section():
    S = SCR.get("tips", {})
    fin = SCR.get("finalists", [])
    why = SCR.get("finalist_rule", {})
    F3 = FIN.get("candidates", {})
    q_gate = [S[k]["agree_drake_3deg"] for k in QUART if k in S]
    rule = {"(1)": "the highest J2 among tips passing the agreement gate",
            "(2)": "the highest J2 at no more than 0.8 of the as-built block&#8217;s replay step",
            "(3)": "the highest three-finger share at \\(\\mu\\) = 0.5"}
    npass = sum(bool(S[k]["gate_pass"]) for k in CURVED if k in S)
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 0: "None"}
    t = ("<p>No tip scores high: the stage scores \\(J\\) lie between 0.04 and 0.44 at every condition, so the held turns "
         f"average under 40&#176; of the 90&#176; goal (Figure&#160;[[fig:screen]]). {words[npass]} of the six capsules and "
         "half-ellipsoids pass the agreement gate with Drake. The finalists are the "
         + "; the ".join(f"{name(k).lower()} ({rule.get(why.get(k, '')[:3], '')})" for k in fin)
         + f". No quarter-ellipsoid passes the gate ({rng(q_gate, 0)} of 24 against "
         f"{S.get(C.ASBUILT, {}).get('agree_drake_3deg')}), so none went to stage 3.</p>")
    t += img("media/fig_screen.png", "Stages 1 and 2 per tip: the stage score at each of the five conditions with J2, their "
             "minimum, as the bar; and the D1&#8211;D8 replays within 3&#176; of Drake with the gate "
             "(<code>screen/summary.json</code>).", key="screen")
    body = []
    for k in [C.ASBUILT, C.FINALIST] + fin:
        v = F3.get(k)
        if not v:
            continue
        ci = v.get("dJ3_ci90")
        body.append([name(k), f(v["J3"], 3), f(v.get("dJ3_vs_base"), 3) if ci else "&#8211;",
                     f"{ci[0]:+.3f} to {ci[1]:+.3f}" if ci else "&#8211;", f"{v['hold3_mu0.5']} of {v['n_hold_mu0.5']}"])
    best = max((k for k in fin if k in F3), key=lambda k: F3[k]["J3"])
    lo, hi = F3[best]["dJ3_ci90"]
    t += (f"<p>On fresh placements no curved finalist beats the as-built block. The best, the {name(best).lower()}, scores "
          f"J3 = {F3[best]['J3']:.3f} against {F3[C.ASBUILT]['J3']:.3f}, and the 90&#8202;% interval of the difference runs "
          f"from {lo:+.3f} to {hi:+.3f} (Table&#160;[[tab:stage3]], Figure&#160;[[fig:stage3]]). Every tip in stage 3 keeps "
          "30 or more of 33 grasps with three loaded fingers through the hold test at \\(\\mu\\) = 0.5.</p>")
    t += table(["Tip", "J3", "J3 minus as-built", "90&#8202;% interval", "Three-finger holds at \\(\\mu\\) = 0.5"], body,
               "Stage 3: the minimum stage score over four conditions on fresh placements and the paired bootstrap "
               "difference against the as-built block (10,000 resamples within hands and cells; "
               "<code>final/summary.json</code>).", key="stage3", text_cols=(0, 3))
    t += img("media/fig_stage3.png", "Stage 3 per tip: the score at each condition and J3 (left), and the difference to the "
             "as-built block with its 90&#8202;% bootstrap interval (right).", key="stage3")
    t += film("media/hom_cylinder_16_0.625_s3_tile.mp4", "media/hom_cylinder_16_0.625_s3_tile_poster.jpg",
              "HOM on the CAD hand&#8217;s 16&#8202;mm cylinder at \\(\\mu\\) = 0.7, seed 3 (a placement no stage used to choose "
              "the grasp), each tip with the grasp it chose on seed 0; pads touching the cylinder red. The half-ellipsoid "
              "16/32&#8202;mm turns the cylinder 49&#176; and drops it; the others hold it.", key="homfilm")
    return t


def cell10_section():
    A = CELL.get("arms", {})
    if not A:
        return "<p><i>[10&#8202;mm cell rows pending]</i></p>"
    cands = CELL["cands"]
    body = [[{"v1": "Pads, builder v1", "v2": "Pads, builder v2", "drake": "Drake"}[a]]
            + [f"{A[a]['held'][str(c)] if str(c) in A[a]['held'] else A[a]['held'].get(c, 0)}" for c in cands]
            + [f"{A[a]['held_total']} of {A[a]['n_runs']}", f"{A[a]['held_seed0']}", f(A[a]["stage_score_deg"], 1)]
            for a in ("drake", "v1", "v2")]
    rv1 = A["v1"]["rms_turn_vs_drake_deg"]
    rv2 = A["v2"]["rms_turn_vs_drake_deg"]
    return (f"<p>Builder v2 lowered the as-built block&#8217;s stage-1 score from 0.428 to 0.219 because no grasp of the "
            f"10&#8202;mm cylinder holds on seed 0. Run with Drake as the plant (the cell&#8217;s scene parsed into Drake 1.57 "
            f"with the tip as a compliant convex hull, <code>scripts/curved_tips_cell_drake.py</code>), the same candidates, seeds "
            f"0&#8211;5 and controller keep the cylinder as builder v1 does: candidates 6 and 7 hold at "
            f"{A['drake']['held']['6'] + A['drake']['held']['7']} of 12 seeds in Drake and "
            f"{A['v1']['held']['6'] + A['v1']['held']['7']} of 12 with v1, and at {A['v2']['held']['6'] + A['v2']['held']['7']} of "
            f"12 with v2 (Table&#160;[[tab:cell10]]). Before a drop the turn traces of both builders follow Drake&#8217;s: the "
            f"median RMS turn-angle difference on candidate 6 is {f(rv2['6'], 1)}&#176; for v2 and {f(rv1['6'], 1)}&#176; for v1. "
            "v2&#8217;s zero for this cell is a builder effect. Whether v2 stays the default waits on the owner.</p>"
            + table(["Plant"] + [f"Grasp {c}" for c in cands] + ["Held, all", "Held at seed 0", "Cell score, &#176;"], body,
                    "The CAD hand&#8217;s 10&#8202;mm cylinder (thumb-to-pair 55&#8202;mm, pair 65&#8202;mm) with the as-built block: "
                    "seeds held of 6 per candidate grasp, \\(\\mu\\) = 0.7, servo-limited plant (<code>cell10/summary.json</code>). "
                    "Cell score: the held turn of seeds 1&#8211;2 of the grasp with the best held turn on seed 0.",
                    key="cell10"))


def stiffness_section():
    if not STF:
        return "<p><i>[stiffness rows pending]</i></p>"
    tips = [k for k in ("tcd_rs3_re3", "tcd_cap8.5", "tcd_ell7.4x8.5x16x32", "tcd_qell7.4x8.5x11x22") if k in STF]
    cur = [k for k in tips if k != "tcd_rs3_re3"]
    sp = [x for k in tips for x in STF[k]["fe_sphere_1N_range_um"]]
    M = lambda m, key, ks=tips: [STF[k]["models"][m][key] for k in ks if m in STF[k]["models"]]  # noqa: E731
    pads = lambda key, ks=tips: M("v1", key, ks) + M("v2", key, ks)  # noqa: E731
    t = (f"<p>The printed tips&#8217; finite elements give a nearly uniform surface: under the 2&#8202;mm sphere the approach at "
         f"1&#8202;N lies between {min(sp):.0f} and {max(sp):.0f}&#8202;&#181;m at all seven points of all four tips, apex, edges "
         f"and ends alike (Figure&#160;[[fig:stiffness]]). The two 0.45&#8202;mm perimeters or four 0.2&#8202;mm skins of solid TPU "
         f"form a shell over the 20&#8202;% gyroid core and spread the load, so the approach grows nearly in proportion to the load "
         f"(\\(n\\) = {rng([STF[k]['fe_exponent_apex_sphere'] for k in tips])} under the sphere) and the 25&#8202;mm tool, which "
         f"loads a far larger area, indents the curved tips {rng([STF[k]['fe_cyl_over_sphere_1N'] for k in cur])} as far as the "
         f"sphere at 1&#8202;N (the flat block {f(STF['tcd_rs3_re3']['fe_cyl_over_sphere_1N'])}). The model of the as-built "
         "block agrees with the 14 October finite elements of the (3,&#160;6) block on a grid twice as fine: 43.4 against "
         "39.1&#8202;&#181;m at 1&#8202;N under the tool, load exponent 0.93 against 0.94.</p>"
         f"<p>Every contact model is a local foundation: the pressure at a point depends on that point&#8217;s overlap alone. At "
         f"10&#8202;MPa Drake&#8217;s field is {rng(M('drake', 'modulus_scale_to_fe'), 1)} times too soft under the sphere and "
         f"{rng(M('drake', 'modulus_scale_cyl0'), 1)} times under the tool, its load exponent is "
         f"{rng(M('drake', 'exponent_apex_sphere'))}, and the tool indents {rng(M('drake', 'cyl_over_sphere_1N', cur))} as far as "
         f"the sphere. The pads behave the same way ({rng(pads('modulus_scale_to_fe'), 1)} and "
         f"{rng(pads('modulus_scale_cyl0'), 1)} times too soft, exponent {rng(pads('exponent_apex_sphere'))}). With each "
         f"model&#8217;s modulus scaled to the finite elements&#8217; apex approach, the RMS error over the sphere positions and the "
         f"three loads is {pct(M('drake', 'rms_err_shape_sphere'))} for Drake and {pct(pads('rms_err_shape_sphere'))} for the "
         f"pads, and over the tool positions {pct(M('drake', 'rms_err_shape_cyl'))} and {pct(pads('rms_err_shape_cyl'))} "
         "(Table&#160;[[tab:stiffness]]). The models put the softest point at the width&#8217;s edge (\\(y\\) = 5.2&#8202;mm) or the "
         "distal end, 25&#8211;55&#8202;% softer than the apex; the printed tips stay within 12&#8202;% of their apex.</p>"
         f"<p>Fitting each pad&#8217;s stiffness to the finite elements does not help. With \\(K_i = c\\,k_i\\), \\(k_i\\) the "
         f"stiffness of the pad&#8217;s 1&#8202;mm patch in the finite elements (a unit force spread over the patch&#8217;s face "
         f"nodes by area, divided into their mean displacement) and \\(c\\) one factor per tip set by the apex sphere at "
         f"1&#8202;N, the error is {pct(M('fit', 'rms_err_shape_sphere'))} over the sphere positions and "
         f"{pct(M('fit', 'rms_err_shape_cyl'))} over the tool&#8217;s, because a patch&#8217;s own stiffness leaves out the load its "
         "neighbours spread to it.</p>"
         "<p>Masterjohn et al. (RA-L 2022) take the hydroelastic field as an approximation of a highly deformable body&#8217;s "
         "response. On these printed tips it misses three properties at once: the modulus by a factor that depends on the "
         "indenter, the load exponent (0.5 against 0.8), and the response&#8217;s dependence on the indenter&#8217;s size. A tip "
         "printed as a shell over a sparse core responds as a plate on a soft foundation, which no local pressure field "
         "reproduces.</p>")
    vs = [(STF[k]["models"]["v1"]["rms_err_shape"], STF[k]["models"]["v2"]["rms_err_shape"]) for k in tips]
    gap = max(abs(a - b) for a, b in vs)
    t += (f"<p>The map does not choose between the pad builders: after the modulus fit v1 and v2 differ by at most "
          f"{100 * gap:.1f} percentage points of RMS error on any tip, and the quarter-ellipsoid&#8217;s centroid-depth pads by "
          f"{100 * abs(STF['tcd_qell7.4x8.5x11x22']['models']['v2c']['rms_err_shape'] - STF['tcd_qell7.4x8.5x11x22']['models']['v2']['rms_err_shape']):.1f}. "
          f"The modulus does: for contact with the bench tool the curved tips correspond to "
          f"{rng(M('v2', 'modulus_scale_cyl0', cur), 1)} times the 10&#8202;MPa that every model uses, the block to "
          f"{f(STF['tcd_rs3_re3']['models']['v2']['modulus_scale_cyl0'], 1)} times. For curved tips the hand should keep builder "
          "v2, the default, and replace the 10&#8202;MPa with \\(E\\) from the tool-probe fit of the tip&#8217;s finite elements "
          "(<code>modulus_scale_cyl0</code> in <code>stiffness/summary.json</code>).</p>")
    body = []
    for k in tips:
        T = STF[k]
        for m, lab in (("drake", "Drake&#8217;s field"), ("v1", "Pads v1"), ("v2", "Pads v2"), ("v2c", "Pads v2, centroid depth"),
                       ("fit", "Pads fitted")):
            v = T["models"].get(m)
            if not v:
                continue
            body.append([name(k if k != "tcd_rs3_re3" else C.ASBUILT) if m == "drake" else "", lab,
                         f(v["modulus_scale_to_fe"], 2), f(v["modulus_scale_cyl0"], 2), f(v["exponent_apex_sphere"], 2),
                         f(v["cyl_over_sphere_1N"], 2), f(100 * v["rms_err_shape_sphere"], 0), f(100 * v["rms_err_shape_cyl"], 0)])
        body.append(["", "<i>Finite elements</i>", "1", "1", f(T["fe_exponent_apex_sphere"], 2), f(T["fe_cyl_over_sphere_1N"], 2),
                     "0", "0"])
    t += img("media/fig_stiffness.png", "Probe maps: the approach at 1&#8202;N at each probe position under the finite elements "
             "and each contact model at 10&#8202;MPa (top), and divided by the model&#8217;s own apex sphere approach (bottom). "
             "Positions: the 2&#8202;mm sphere at the apex, across the width (\\(y\\), mm) and along the finger (\\(z\\), mm), then "
             "the bench tool at three positions along the finger (<code>stiffness/summary.json</code>).", key="stiffness")
    t += table(["Tip", "Model", "Modulus factor, sphere", "Modulus factor, tool", "Load exponent \\(n\\)",
                "Tool over sphere approach", "RMS error, sphere positions, %", "RMS error, tool positions, %"], body,
               "Each model against the finite elements: the factor on its 10&#8202;MPa modulus that reproduces the apex approach "
               "at 1&#8202;N under the sphere and under the tool, the load exponent under the sphere at the apex, the tool&#8217;s "
               "approach over the sphere&#8217;s at the apex at 1&#8202;N, and the RMS approach error over position and the three "
               "loads with the sphere-fitted modulus.", key="stiffness", text_cols=(0, 1))
    return t


def quarter_section():
    T, nail, sens, board = QL.get("tips", {}), QL.get("nail", {}), QL.get("sensing", {}), QL.get("board", {})
    if not T:
        return "<p><i>[quarter-ellipsoid summary pending]</i></p>"
    qk = ["tcd_qell7.4x8.5x11x22", "tcd_qell7.4x5x11x22", "tcd_qell7.4x8.5x16x32"]
    b = T["tcd_rs3_re3"]
    body = []
    for k in qk:
        s = sens.get(k, {}).get("current")
        if not s:
            continue
        body.append([name(k + C.PRINT), f(T[k]["pulp_mm"], 1), f(s["depth_mm"], 2), f(s["approach_1N_um"], 0),
                     f(s["travel_um_per_N"], 2), f(s["shear_y_um_per_N"], 1), f(1e3 * s["noise_N"], 0), f(s["B0_max_mT"], 0)])
    fb = sens.get("tcd_qell7.4x8.5x11x22", {}).get("free_back", {})
    cur = sens.get("tcd_qell7.4x8.5x11x22", {}).get("current", {})
    t = (f"<p>The flat back is {rng([T[k]['back_width_mm']['0'] for k in qk], 1)}&#8202;mm wide at the contact point and "
         f"{rng([T[k]['back_width_mm']['-7'] for k in qk], 1)}&#8202;mm at 7&#8202;mm distal, where the board&#8217;s distal chip "
         f"sits; the five-chip board is {f(board.get('width_mm'), 1)}&#8202;&#215;&#8202;{f(board.get('length_mm'), 1)}&#8202;mm with "
         f"its outer chips 7&#8202;mm from the centre, so it overhangs the back by about "
         f"{(board.get('width_mm', 20) - T[qk[0]]['back_width_mm']['0']) / 2:.1f}&#8202;mm on each side and its side chips sit at "
         f"the tip&#8217;s edges. The mount face shrinks to {rng([T[k]['mount_face_mm2'] for k in qk], 0)}&#8202;mm&#178; from the "
         f"block&#8217;s {b['mount_face_mm2']:.0f}&#8202;mm&#178;.</p>"
         f"<p>With the board on the back plane the magnet barely moves. The finite elements, with every node of the back plane "
         f"fixed, put the magnet&#8217;s travel at {rng([sens[k]['current']['travel_um_per_N'] for k in qk if k in sens], 2)}&#8202;"
         f"&#181;m/N (Table&#160;[[tab:qsense]]): between the magnet and the board lie the pocket&#8217;s solid floor and the "
         f"part&#8217;s bottom skins, 1&#8211;1.9&#8202;mm of nearly solid TPU, and the load compresses the pulp above the magnet "
         f"instead. A first run that left the back plane free (its grid had no node at \\(x\\) = 0) gave "
         f"{f(fb.get('travel_um_per_N'), 0)}&#8202;&#181;m/N and an approach of {f(fb.get('approach_1N_um'), 0)}&#8202;&#181;m against "
         f"{f(cur.get('approach_1N_um'), 0)}; its rows are kept in <code>fe/superseded_free_back.jsonl</code>. The field at the "
         f"nearest chip is {rng([sens[k]['current']['B0_max_mT'] for k in qk if k in sens], 0)}&#8202;mT, above the "
         "MLX90393&#8217;s 50&#8202;mT saturation onset, so the board would saturate before it measured.</p>")
    t += table(["Tip", "Pulp, mm", "Magnet top depth, mm", "Approach at 1&#8202;N, &#181;m", "Magnet travel, &#181;m/N",
                "Magnet travel, shear y, &#181;m/N", "Force noise, mN", "Field at the nearest chip, mT"], body,
               "Sensing of the quarter-ellipsoids with the board on the back plane: finite elements of recipe 95A, 20&#8202;% "
               "gyroid, with the back plane fixed, and the five-chip readout of the 13 October readout page "
               "(<code>qell/summary.json</code>).", key="qsense")
    n1 = nail.get("tcd_qell7.4x8.5x11x22_u0.1", {})
    t += (f"<p>A fingernail, a rigid plate 0.6&#8202;mm thick and 10&#8202;mm wide on the back plane reaching "
          f"{f(n1.get('over_mm'), 1)}&#8202;mm past the distal end, lies behind the rigid board, so it leaves the pads&#8217; "
          f"foundation unchanged. The bench tasks touch it: in the D1&#8211;D8 replays the tool meets the nail in "
          + ", ".join(f"{v['touched']}" for v in nail.values()) + " of 24 runs ("
          + ", ".join(name(k).lower() for k in nail) + f"), at its distal edge, for a median "
          f"{rng([100 * v['step_share_median'] for v in nail.values()], 1)}&#8202;% of the steps and up to "
          f"{rng([v['max_N'] for v in nail.values()], 0)}&#8202;N. Nothing else touches it.</p>")
    tq = T[qk[0]]["terrace_face_up_mm"]
    t += (f"<p>Printed on its back, the quarter-ellipsoid&#8217;s apex is the top of the print and its last layer flattens a "
          f"terrace of {f(tq['0.1']['across'], 1)}&#8202;&#215;&#8202;{f(tq['0.1']['along'], 1)}&#8202;mm at 0.1&#8202;mm layers "
          f"({f(tq['0.2']['across'], 1)}&#8202;&#215;&#8202;{f(tq['0.2']['along'], 1)}&#8202;mm at 0.2&#8202;mm; the 16/32 tip "
          f"{f(T[qk[2]]['terrace_face_up_mm']['0.1']['along'], 1)}&#8202;mm along), larger than the 1&#8202;N contact patch. "
          f"Printed upright, each 0.1&#8202;mm layer&#8217;s outline passes over the apex and the surface departs from the design "
          f"by {T[qk[0]]['terrace_upright_um']:.1f}&#8202;&#181;m there, and the back face for the board becomes a side wall.</p>")
    return t


RECIPES = {"w95_20": "95A, 20&#8202;% gyroid"}


def print_section():
    body = []
    for fp in sorted(glob.glob(os.path.join(D, "print/*/20261015-*_w95_20.json"))):
        r = json.load(open(fp))
        sl = r.get("slice", {})
        body.append([name(r["tip"]), RECIPES.get(r["recipe"], r["recipe"]), "yes" if r.get("watertight") else "no",
                     f(r.get("part_volume_mm3"), 0), f(r["slot"]["wall_min_mm"], 1), f(sl.get("pause_found_before_layer_z"), 1),
                     str(sl.get("islands_max")), f(100 * sl.get("supported_area", 0), 0), sl.get("time", "&#8211;"),
                     f"<code>{os.path.relpath(fp, D)[:-5]}.3mf</code>"])
    if not body:
        return "<p><i>[print files pending]</i></p>"
    return ("<p>Each curved finalist prints upright on its mount face in 0.1&#8202;mm layers after a 0.2&#8202;mm first layer, "
            "as a PrusaSlicer project on the owner&#8217;s v17 template (TPU on extruder 2, two perimeters, gyroid infill, four "
            "top and four bottom layers). The magnet stands on its edge in a slot 90&#8202;% of its diameter wide, the press fit; "
            "the print pauses (M601) before the first layer above the slot, the magnet is pressed down the slot to its seat "
            "with its north pole toward the palmar face, and the next layer bridges the slot. PrusaSlicer 2.9.2 sliced every "
            "project (Table&#160;[[tab:print]]).</p>"
            + table(["Tip", "Recipe", "Watertight", "Part volume, mm&#179;", "TPU around the slot, mm", "Pause at, mm",
                     "Islands per layer, max", "Supported area, %", "Print time", "Project"], body,
                    "Print files and slice audit (<code>print/&lt;design&gt;/</code>, with the upright STL beside each project; "
                    "G-code in <code>logs/20261015-curved_tips/print/</code>).", key="print", text_cols=(0, 1, 2, 8, 9)))


def lede():
    F3 = FIN.get("candidates", {})
    fin = SCR.get("finalists", [])
    best = max((k for k in fin if k in F3), key=lambda k: F3[k]["J3"])
    lo, hi = F3[best]["dJ3_ci90"]
    cur = [k for k in CURVED + QUART if k in COST]
    gr = [COST[k]["ratios"]["gpu"] for k in cur]
    tips = list(STF)
    sp = [x for k in tips for x in STF[k]["fe_sphere_1N_range_um"]]
    return (f"No curved fingertip beats the printed block on the SR2 hand&#8217;s tasks: of three finalists from nine shapes, "
            f"the best, a 7.4&#8202;&#215;&#8202;8.5&#8202;&#215;&#8202;11/22&#8202;mm half-ellipsoid, changes the fresh-placement score by "
            f"{F3[best]['dJ3_vs_base']:+.3f} (90&#8202;% interval {lo:+.3f} to {hi:+.3f}). Curved tips cut the twist-onset arm to a "
            f"third and the GPU step to {rng(gr)} of the block&#8217;s, and the 16/32&#8202;mm half-ellipsoid follows Drake most "
            f"closely along the bench replays. Finite elements of the printed tips show a nearly uniform surface "
            f"({min(sp):.0f}&#8211;{max(sp):.0f}&#8202;&#181;m at 1&#8202;N under a 2&#8202;mm sphere) that neither Drake&#8217;s "
            "field nor any pad builder reproduces. On the 10&#8202;mm cylinder Drake keeps the grasp that builder v2 drops.")


def limitations():
    return ("<p>Every reference is computed. The finite elements use the homogenized 20&#8202;% gyroid of 9 October and a "
            "30&#8202;MPa solid TPU that no bench test of these tips has measured, and Drake&#8217;s field and the pads use "
            "10&#8202;MPa. The task scores come from simulated open-loop replays and HOM turns of the SR2 hand; no curved tip has "
            "been printed. The probe map presses along the palmar normal and keeps only that force component, which on the "
            "tips&#8217; slopes (up to 38&#176; at \\(y\\) = 5.2&#8202;mm) omits the tangential part of the contact force. The "
            "quarter-ellipsoids ran the bench plans made for the block.</p>")


def open_items():
    items = [
        ("Bench probe map", "Press the printed as-built block and the 16/32&#8202;mm half-ellipsoid with a 4&#8202;mm ball at "
         "the seven map points at 0.5, 1 and 3&#8202;N on the bench load cell; an approach within 20&#8202;% of "
         "140&#8211;170&#8202;&#181;m at 1&#8202;N everywhere confirms the shell model, edges 30&#8202;% softer than the apex "
         "refute it in favour of the local models."),
        ("Builder v2 and the 10&#8202;mm cylinder", "If the owner keeps v2, rerun stage 1 of the as-built block with "
         "<code>scripts/curved_tips_rig.py scale --h local --K each</code> at \\(E\\) from "
         "<code>modulus_scale_cyl0</code>; the cylinder held at seed 0 in Drake and v1, so a v2 that still drops it at the "
         "fitted modulus marks the per-pad depth as the cause."),
        ("Tool-fitted modulus in the pipeline", "Set \\(E\\) per tip to the tool-probe fit of builder v2 (13&#8211;31&#8202;MPa) in "
         "<code>reorient_backends.replace_tips</code> and repeat the D1&#8211;D8 trajectory comparison "
         "(<code>scripts/curved_tips_traj.py</code>); a median RMS tool angle below the 10&#8202;MPa value for both pads and "
         "Drake makes the fitted modulus the default."),
        ("Sensing on a quarter-ellipsoid", "Move the magnet to 1.5&#8211;2.5&#8202;mm under the apex with soft gyroid below it "
         "and rerun <code>scripts/curved_tips_fe.py</code>; a design reaches the 2026-10-14 sensing rule if the travel is at "
         "least 12&#8202;&#181;m/N with the field at the nearest chip under 50&#8202;mT."),
        ("Fingernail overhang", "Rerun <code>scripts/curved_tips_nail.py --over 0</code> and <code>--over 0.5</code>; a nail that "
         "the tool no longer touches in any D1&#8211;D8 replay gives the length the bench tasks tolerate."),
    ]
    return "<ol>" + "".join(f"<li><b>{a}.</b> {b}</li>" for a, b in items) + "</ol>"


def byline():
    return ("Scripts <code>scripts/curved_tips_{common,law,bed,cost,traj,traj_analysis,rig,queue,cell_drake,stiffness,fe,nail,"
            "print,film,figures,analysis}.py</code> &#183; builder <code>scripts/curved_tips_page.py</code> &#183; rows "
            "<code>docs/experiments/20261015-curved_tips/</code>")


def main():
    REFS["sec:stiffness"] = 8
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "GLOSSARY": glossary(), "QUESTION": question(), "BED": bed_section(),
         "COST": cost_section(), "TRAJ": traj_section(), "SCREEN": screen_section(), "CELL10": cell10_section(),
         "STIFFNESS": stiffness_section(), "QUARTER": quarter_section(), "PRINT": print_section(),
         "LIMITS": limitations(), "OPEN": open_items(), "LEDE": lede()}
    v["FOOTER"] = ("<p>Rebuild: <code>.venv/bin/python scripts/curved_tips_page.py</code>. Rows and summaries in "
                   "<code>docs/experiments/20261015-curved_tips/</code>: <code>bed/</code>, <code>cost/</code>, <code>traj/</code>, "
                   "<code>screen/</code> (stages 1&#8211;2), <code>final/</code> (stage 3), <code>cell10/</code>, "
                   "<code>stiffness/</code>, <code>qell/</code>, <code>fe/</code>, <code>nail/</code>, <code>print/</code>, "
                   "<code>media/</code>. Protocol: <code>docs/notes/20261014-tip_codesign_plan.md</code>; the block study: "
                   "<a href=\"../20261014-tip_codesign/20261014-sphere_pad_accuracy_and_fingertip_codesign.html\">sphere-pad "
                   "accuracy and fingertip co-design</a>.</p>")
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
    t = retro_style.apply(t)
    open(OUT, "w").write(t)
    print(f"formulas {n}; figures {FIG[0]}; tables {TAB[0]}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
