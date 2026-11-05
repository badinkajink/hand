#!/usr/bin/env python3
"""Page: block, capsule, ellipsoid and hemisphere fingertips on the v4b finger of the SR2 hand, the rerun of the
fingertip-shape screens of 2026-10-14 and 10-15 and the FE checks of 10-18/19 at 2.5 MPa on the current hand
(hand-off docs/handoff/20261103-fingertip_shape_rerun.md).

Data: docs/experiments/20261104-fingertip_shape_rerun/ (mount/, kin/, bed/, cost/, traj/, screen/, final/, fe/, compute/,
media/), every number read from the run's summaries; 10-15's from docs/experiments/20261015-curved_tips/, 10-14's from
docs/experiments/20261014-tip_codesign/ and 10-18's from docs/experiments/20261018-fe_contact_calibration/:

    python3 scripts/fingertip_shape_rerun_analysis.py all
    ~/.claude/bin/resguard.sh run --mem 2G -- .venv/bin/python scripts/fingertip_shape_rerun_figures.py mount bed screen stage3 fe
    .venv/bin/python scripts/fingertip_shape_rerun_page.py
Out: docs/experiments/20261104-fingertip_shape_rerun/20261104-fingertip_shapes_v4b_finger.html (media inlined).
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
import docs_index  # noqa: E402
import fingertip_shape_rerun_common as C  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = str(C.D)
OUT = os.path.join(D, "20261104-fingertip_shapes_v4b_finger.html")
TPL = os.path.join(ROOT, "scripts/fingertip_shape_rerun_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
OLD15 = os.path.join(ROOT, "docs/experiments/20261015-curved_tips")
OLD14 = os.path.join(ROOT, "docs/experiments/20261014-tip_codesign")
OLD18 = os.path.join(ROOT, "docs/experiments/20261018-fe_contact_calibration")
FIG, TAB, FILM, REFS = [0], [0], [0], {}
EXTRA_CSS = """
.tw th{white-space:normal !important;vertical-align:bottom;text-transform:none}
.tw td:first-child{min-width:9em}
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
figure.films video{width:100%;height:auto}
figure.img img{width:100%;height:auto}
""" + texsvg.BLOCK_CSS + "\n"
MM, UM, NN, DEG, MPA = "&#8202;mm", "&#8202;&#181;m", "&#8202;N", "&#176;", "&#8202;MPa"
ORDER = [C.REF] + C.HALF + [C.HEMI] + C.CAPSULES + C.BLOCKS + C.QUARTERS


# ------------------------------------------------------------------------------------------ helpers

def J(rel, base=D):
    p = os.path.join(base, rel)
    return json.load(open(p)) if os.path.exists(p) else {}


def data_uri(path, mime=None):
    if mime is None:
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".mp4": "video/mp4"}[os.path.splitext(path)[1]]
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def jpeg_uri(path, quality=86):
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def figure(body, caption, key=None, cls="img", kind="Figure", counter=FIG):
    counter[0] += 1
    if key:
        REFS[f"{'film' if kind == 'Film' else 'fig'}:{key}"] = counter[0]
    return f'<figure class="{cls}">{body}<figcaption>{kind}&#160;{counter[0]}. {caption}</figcaption></figure>'


def img(rel, caption, key=None):
    p = os.path.join(D, rel)
    if not os.path.exists(p):
        return f"<p><i>[figure {rel} not drawn yet]</i></p>"
    return figure(f'<img alt="" src="{data_uri(p)}">', caption, key)


def film(rel, poster_rel, caption, key=None):
    p = os.path.join(D, rel)
    if not os.path.exists(p):
        return f"<p><i>[film {rel} not rendered yet]</i></p>"
    poster = os.path.join(D, poster_rel)
    pa = f' poster="{jpeg_uri(poster)}"' if os.path.exists(poster) else ""
    v = f'<video controls muted loop playsinline preload="metadata"{pa} src="{data_uri(p)}"></video>'
    return figure(v, caption, key, cls="films", kind="Film", counter=FILM)


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


def rng(vals, n=2, unit=""):
    v = [x for x in vals if x is not None]
    if not v:
        return "&#8211;"
    lo, hi = min(v), max(v)
    s = f"{lo:.{n}f}" if f"{lo:.{n}f}" == f"{hi:.{n}f}" else f"{lo:.{n}f}&#8211;{hi:.{n}f}"
    return s + unit


def name(k):
    """Sentence-case table name of a tip key."""
    s = C.label(k).replace(" mm", "&#8202;mm").replace(" x ", "&#8202;&#215;&#8202;")
    return s[:1].upper() + s[1:]


def get(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


# ------------------------------------------------------------------------------------------ data

MNT = J("mount/mount.json")
BED = J("bed/summary.json").get("tips", {})
COST = J("cost/summary.json").get("tips", {})
TRAJ = J("traj/summary.json").get("tips", {})
SCR = J("screen/summary.json")
FIN = J("final/summary.json")
FE = J("fe/summary.json").get("tips", {})
CMP = J("compute/summary.json")
BED15 = J("bed/summary.json", OLD15).get("tips", {})
COST15 = J("cost/summary.json", OLD15).get("tips", {})
SCR15 = J("screen/summary.json", OLD15)
FIN15 = J("final/summary.json", OLD15)


def old(k):
    return C.OLD_KEY.get(k, k)


# ------------------------------------------------------------------------------------------ sections

def glossary():
    items = [
        ("Tips", "Twelve fingertips inside the printed block&#8217;s envelope, 17.0&#8202;mm deep along the palmar normal "
         "\\(x\\), 14.8&#8202;mm wide along the PIP axis \\(y\\) and 21.41&#8202;mm long along the finger \\(z\\), each with its "
         "palmar apex 10.55&#8202;mm from the finger&#8217;s axis over the contact point \\((y, z) = (0, 0)\\). The v4b tip "
         "(<code>tcd_ell7.4x8.5x11x22</code>) is the 11/22 half-ellipsoid, semi-axes 7.4&#8202;mm across, 8.5&#8202;mm deep, "
         "11&#8202;mm distal and 22&#8202;mm proximal of the apex, the tip calibrated on the UR5 and the reference here. "
         "Half-ellipsoid \\(a_y \\times a_x \\times a_z/a_{zp}\\): the palmar half of such an ellipsoid, its section running "
         "straight back behind the equator. Quarter-ellipsoid: the same half cut flat at its equator. Capsule <i>r</i>: a "
         "cylinder of radius <i>r</i> along the finger capped by a hemisphere at the distal end. Hemisphere: the half-ellipsoid "
         "with every semi-axis 7.4&#8202;mm, the envelope&#8217;s half-width, a dome on a round post. Blocks: the as-built block "
         "with side and end fillets of 3 and 3&#8202;mm and the 14 October finalist with 3 and 6&#8202;mm. The whole real_v1 "
         "sphere tip (<code>tcd_sph10.55</code>) runs on the bed only."),
        ("Surfaces", "The capsules, the half-ellipsoids other than the v4b tip and the quarter-ellipsoids are simulated as "
         "printed upright in 0.1&#8202;mm layers (key suffix <code>_u0.1</code>, within 4.5&#8202;&#181;m of the design over the "
         "contact region), the blocks as printed face up in 0.2&#8202;mm layers (<code>_p0.2</code>), the v4b tip and the "
         "hemisphere on their design surfaces."),
        ("v4b finger", "The CAD model of the SR2 finger with the DIP link b v4, the cassette 3T, the top cover TW15H3 (the cap) "
         "trimmed at \\(x_F = -0.10\\)&#8202;mm with four M2 button heads, and the TPU pad, whose centre lies 36.36&#8202;mm from "
         "the PIP axis (<code>SCALE_HAND=cad_v2</code>, <code>scripts/build_cad_v2_finger.py</code>). The rigid parts are "
         "collision spheres inscribed in each part; every tip replaces the pad and keeps them."),
        ("Mount check", "Each tip&#8217;s design surface placed on the v4b finger and cut where the v4b pad is cut, 13.45&#8202;mm "
         "under the apex, on a 0.2&#8202;mm grid. Extra TPU: the tip&#8217;s volume inside a rigid part where the 11/22 has none "
         "(mm&#179;). Bearing kept: of the v4b pad&#8217;s surface within 0.3&#8202;mm of a part, the share inside the tip. A tip "
         "mounts as the 11/22 does when no part gains more than 2&#8202;mm&#179; and every part keeps 95&#8202;% of its bearing "
         "surface; otherwise it is simulated on the envelope alone, on the 11/22&#8217;s rigid parts. Magnet wall: the least TPU "
         "between one of pad 05R4&#8217;s four 3&#8202;&#215;&#8202;2&#8202;mm magnets (a 5.0&#8202;mm square about the contact "
         "point, tops 7.5&#8202;mm deep) and the tip&#8217;s surface; the print rule is 1.8&#8202;mm."),
        ("1&#8202;mm pads", "The tip&#8217;s palmar region as spheres of radius 0.75&#8202;mm on a 1&#8202;mm lattice, each a MuJoCo "
         "soft contact of stiffness \\(K_i = E a_i / h_i\\), \\(E\\) = 2.5&#8202;MPa, \\(a_i\\) the pad&#8217;s share of the surface "
         "and \\(h_i\\) its depth below the tip&#8217;s volume centroid (builder v2); relaxation time 0.04&#8202;s, impratio 100, "
         "1&#8202;ms step. 2.5&#8202;MPa is the modulus fitted to the UR5 presses of the v3 tips on 30 October."),
        ("Drake", "Drake 1.57 hydroelastic contact of the tip&#8217;s compliant convex hull, modulus 2.5&#8202;MPa, relaxation "
         "time 0.01&#8202;s, SAP solver, 1&#8202;ms step."),
        ("Law", "The quasi-static integral of Drake&#8217;s pressure field \\(p = E\\,(1 - g(\\mathbf{x}))\\), \\(g\\) the gauge of the "
         "tip&#8217;s 0.25&#8202;mm hull about its volume centroid, over the contact with the bed&#8217;s tool, at 2.5&#8202;MPa."),
        ("Bed", "Two force-controlled rails squeeze a 25&#8202;mm diameter, 24.5&#8202;g screwdriver between two copies of a tip at "
         "1&#8202;N per tip, friction coefficient \\(\\mu\\) = 1. Approach: rail travel from first touch (&#181;m). Twist-onset "
         "arm: the torque at which the tool starts to turn under a slow torque ramp, over \\(2\\mu N\\) (mm). Spin arm: the mean "
         "torque that drives the tool about the pinch axis at rate \\(\\omega\\), over \\(2\\mu N\\) (mm). Brake force: the "
         "pinch force at which the tool, released 15&#8202;mm off the pinch, stops swinging at 45&#176; (N)."),
        ("SCS0009 plant", "The finger servo model of the built hand: gain 0.72&#8202;N&#8202;m/rad, torque limit 0.18&#8202;N&#8202;m, "
         "damping 0.0144&#8202;N&#8202;m&#8202;s/rad (<code>kp0.72_kv0_fr0.18_fl0_dp0.0144</code>)."),
        ("HOM, cells", "HOM: the hand&#8217;s closed-loop in-hand turn controller (<code>scripts/hom_turn3.py</code>, model gain 0.72). "
         "Cells: four grasps of cylinders 100&#8202;mm long on the v4b hand, 25, 16, 10 and 16&#8202;mm in diameter, on the "
         "finger layouts of 14 October (thumb-to-pair and pair separations 92.5/102.5, 62.5/72.5, 55/65 and 122.5/132.5&#8202;mm), "
         "with nine candidate grasps each from the scale study&#8217;s kinematics of the v4b finger. Held turn: the turn at the "
         "end of a rollout when two or more fingers hold the tool, otherwise 0 (&#176;)."),
        ("Stage scores", "The mean of five parts, each a held turn over 90&#176;: per cell the mean held turn of the grasp chosen "
         "by its held turn on seed 0, on the scoring seeds; and the mean held turn of HOM on the eight deployed real_v1 hands "
         "D1&#8211;D8. J1: at friction 0.7 on the SCS0009 (cells seeds 1&#8211;2, HOM seeds 0&#8211;2). J2: the minimum over five "
         "conditions (friction 0.7, 1.0 and 0.5; servo gain 0.5&#8202;N&#8202;m/rad; torque limit 0.12&#8202;N&#8202;m; HOM seeds "
         "0&#8211;4). J3: the minimum over the four conditions other than friction 1.0 on fresh placements (cells seeds "
         "3&#8211;5, HOM seeds 5&#8211;9). Placements: seed 0 none, other seeds the tool moved by draws of standard deviation "
         "2&#8202;mm and 2&#176;."),
        ("Agreement gate", "The eight deployed bench plans replayed open loop on the SCS0009 plant at friction 1.0, seeds "
         "0&#8211;2: the replays whose pads end within 3&#176; of Drake&#8217;s turn with Drake&#8217;s held outcome, of 24. A tip "
         "passes with at most two fewer than the v4b tip."),
        ("Rigid share", "Of the normal force that the fingers press on a cell&#8217;s cylinder, summed over a rollout&#8217;s "
         "control samples, the share through finger geoms other than the pads: on the v4b finger the DIP link, the cassette "
         "and the cap."),
        ("RMS tool angle", "Two replays of one placement under two contact settings, sampled every 20&#8202;ms: the root mean "
         "square of the angle between the two tool orientations (&#176;)."),
        ("Finite elements", "The printed tip as voxel finite elements (the model of 18 October): two 0.45&#8202;mm perimeters "
         "and four skins of solid TPU at 12&#8202;MPa over the homogenized 30&#8202;% gyroid of Siraya 85A, in the tip&#8217;s print "
         "orientation, the four magnets of pad 05R4 bonded where they fit, the pad cut where the v4b pad is cut with that face "
         "fixed (a quarter-ellipsoid on its own back plane), frictionless rigid probes. Approach: probe travel from first touch "
         "(&#181;m). Force exponent \\(n\\): the slope of approach against load on log axes between 0.5 and 3&#8202;N. Twist "
         "arm: the pressure-weighted mean distance of the patch from its centre (mm)."),
        ("Force resolution", "The Cram&#233;r&#8211;Rao bound on the normal force at the contact point from one 141&#8202;Hz "
         "sample of the board&#8217;s five MLX90393 chips at the gain the static field allows, with the contact position and a "
         "uniform background field estimated alongside (N; <code>scripts/tpu_tip_magnet_sensing.py</code>)."),
        ("Compute", "Core-hours: the summed wall time of single-core jobs; GPU hours: the wall time of GPU jobs."),
    ]
    return "<dl>" + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in items) + "</dl>"


def question():
    return ("<p>The owner chose the 11/22 half-ellipsoid on 18 October after two screens on the first CAD hand at 10&#8202;MPa "
            "and a calibration of the contact models to the finite elements. Since then the hand has moved to the v4b "
            "finger (DIP link b v4, cassette 3T, cap TW15H3) and the pads&#8217; modulus to 2.5&#8202;MPa; the SCS0009 plant "
            "is the one the screens used. This rerun repeats the 14 and 15 October protocol on the current hand for the "
            "nine curved tips, the two blocks and a hemisphere, with the v4b tip as the reference: the fit of each tip on "
            "the v4b finger, the contact bed against the law and Drake, the cost of a simulation step, the three-stage task "
            "screen with Drake&#8217;s agreement gate, and the finite elements with the force resolution of pad 05R4.</p>"
            "<p>Five rules differ from 15 October. The cells keep their layouts and cylinders, and their candidate grasps "
            "come from the v4b finger&#8217;s kinematics (<code>kin/reach.json</code>). The gate and stage 3 compare each tip "
            "with the v4b tip in place of the as-built block, which runs stage 3 beside it. The cost of finalist rule (2) "
            "is the median, over the placements on which both tips&#8217; pads hold the tool, of the ratio of microseconds "
            "per replay step: the v4b tip&#8217;s pads drop the tool in about half the replays, and a dropped tool makes a "
            "replay cheap. Drake runs the bed&#8217;s four metrics at one load and rate (approach at 1 and 3&#8202;N, onset "
            "arm at 1&#8202;N, spin arm at 1&#8202;rad/s, brake force at a 1&#8202;ms step); its replays of the hand take "
            "130&#8211;350&#8202;s at 2.5&#8202;MPa against 61&#8202;s at 10&#8202;MPa. Every turn row records the rigid "
            "parts&#8217; share of the contact force.</p>")


def mount_section():
    T = MNT.get("tips", {})
    if not T:
        return "<p><i>[mount check not run]</i></p>"
    mounts = [k for k in ORDER if T[k]["verdict"] == "mounts"]
    body = []
    for k in ORDER:
        r = T[k]
        ex, kp = r["extra_mm3"], r["kept"]
        walls = [w for w in r["magnet_walls_mm"] if w is not None]
        body.append([name(k), "Mounts" if r["verdict"] == "mounts" else "Envelope alone", f(r["pad_volume_mm3"], 0),
                     f(ex["cassette"], 1), f(ex["cap"], 1), f(100 * kp["cassette"]["share"], 0), f(100 * kp["cap"]["share"], 0),
                     f(100 * kp["dip_link"]["share"], 0), (f(min(walls), 1) if r["magnets_inside"] else "Outside")])
    t = table(["Tip", "Fit", "Pad volume, mm&#179;", "Extra TPU in the cassette, mm&#179;", "Extra TPU in the cap, mm&#179;",
               "Cassette bearing kept, %", "Cap bearing kept, %", "DIP-link bearing kept, %", "Magnet wall, mm"], body,
              "Each tip on the v4b finger (mount/mount.json; <code>scripts/fingertip_shape_rerun_mount.py</code>).", key="mount")
    fig = img("media/fig_mount.png", "The tips on the v4b finger: for each, the profile along the finger at \\(y = 0\\) (distal "
              "end left, palmar side up) and the section across it 6&#8202;mm distal of the contact point, through the cap&#8217;s "
              "side blocks. TPU of the tip light brown, the rigid parts grey, the 11/22&#8217;s grooves and pin hole dark brown, "
              "TPU inside a rigid part where the 11/22 has none red.", key="mount")
    return (f"<p>Of the twelve tips, {len(mounts)} mount on the v4b finger as the 11/22 does: "
            + ", ".join(name(k) for k in mounts) + " (Table&#160;[[tab:mount]], Figure&#160;[[fig:mount]]). The others are "
            "simulated on the envelope alone, on the 11/22&#8217;s rigid parts.</p>" + fig + t)


def bed_section():
    if not BED:
        return "<p><i>[bed not run]</i></p>"
    body = []
    for k in ORDER + [C.SPHERE]:
        t = BED.get(k)
        if not t:
            continue
        R, lw, Pd = t["ratios"], t["law"] or {}, t["pads"]
        o = BED15.get(old(k)) or {}
        body.append([name(k), f(get(lw, "approach_um", "1"), 0), f(R["pads_approach_1N_over_law"]),
                     f(get(lw, "onset_arm_mm", "1")), f(R["pads_onset_1N_over_law"]), f(R["drake_onset_1N_over_law"]),
                     f(get(o, "pads", "onset_arm_1N_mm")), f(R["pads_spin_1_over_law"]), f(R["drake_spin_1_over_law"]),
                     f(R["pads_spin_3_over_0.3"]), f(Pd.get("N45_1ms_N")), f(get(t, "drake", "N45_1ms_N")),
                     f(Pd.get("touching_1N"), 0), f(get(o, "pads", "touching_1N"), 0)])
    tab = table(["Tip", "Law approach at 1&#8202;N, &#181;m", "Pads over law", "Law onset arm, mm", "Pads over law",
                 "Drake over law", "Pads onset arm, 10&#8202;MPa (15 Oct), mm", "Spin arm at 1&#8202;rad/s, pads over law",
                 "Drake over law", "Pads, 3 over 0.3&#8202;rad/s", "Brake force, pads, N", "Brake force, Drake, N",
                 "Pads touching per side", "Touching, 10&#8202;MPa (15 Oct)"], body,
                "The bed at 1&#8202;N per tip and 2.5&#8202;MPa (bed/summary.json), with the pads&#8217; onset arm and touching "
                "pads of 15 October at 10&#8202;MPa.", key="bed")
    fig = img("media/fig_bed.png", "Bed per tip: the twist-onset arm at 1&#8202;N (law, pads and Drake at 2.5&#8202;MPa; open: the "
              "pads at 10&#8202;MPa on 15 October) and the spin arm at 1&#8202;rad/s over the law.", key="bed")
    curved = C.HALF + [C.REF, C.HEMI] + C.CAPSULES
    g = lambda k, *p: get(BED.get(k) or {}, *p)  # noqa: E731
    onset = [g(k, "law", "onset_arm_mm", "1") for k in curved]
    onset_q = [g(k, "law", "onset_arm_mm", "1") for k in C.QUARTERS]
    onset_b = [g(k, "law", "onset_arm_mm", "1") for k in C.BLOCKS]
    o15 = [get(BED15.get(old(k)) or {}, "law", "onset_arm_mm", "1") for k in curved if k != C.HEMI]
    b15 = [get(BED15.get(k) or {}, "law", "onset_arm_mm", "1") for k in C.BLOCKS]
    R = lambda k, key: g(k, "ratios", key)  # noqa: E731
    text = (
        f"<p>At 2.5&#8202;MPa the curved tips turn the tool on half of the blocks&#8217; friction arm. The law&#8217;s "
        f"twist-onset arm at 1&#8202;N is {rng(onset)}&#8202;mm on the capsules, half-ellipsoids and the hemisphere and "
        f"{rng(onset_q)}&#8202;mm on the quarter-ellipsoids, against {rng(onset_b)}&#8202;mm on the blocks. At 10&#8202;MPa "
        f"on 15 October it was {rng(o15)} against {rng(b15)}&#8202;mm; the softer pads sink deeper, so their patch is wider "
        f"(Table&#160;[[tab:bed]], Figure&#160;[[fig:bed]]). "
        f"{rng([g(k, 'pads', 'touching_1N') for k in curved], 0)} pads touch per side at 1&#8202;N on these tips, against "
        f"{rng([get(BED15.get(old(k)) or {}, 'pads', 'touching_1N') for k in curved if k != C.HEMI], 0)} at 10&#8202;MPa, and "
        f"{rng([g(k, 'pads', 'touching_1N') for k in C.BLOCKS], 0)} on the blocks.</p>"
        f"<p>The pads give {rng([R(k, 'pads_onset_1N_over_law') for k in curved])} of the law&#8217;s onset arm on the "
        f"capsules, half-ellipsoids and hemisphere and {rng([R(k, 'pads_onset_1N_over_law') for k in C.BLOCKS])} on the "
        f"blocks; Drake gives {rng([R(k, 'drake_onset_1N_over_law') for k in ORDER])}. At 1&#8202;rad/s the pads&#8217; spin "
        f"arm is {f(R(C.REF, 'pads_spin_1_over_law'))} of the law&#8217;s on the v4b tip, "
        f"{rng([R(k, 'pads_spin_1_over_law') for k in C.CAPSULES])} on the capsules, "
        f"{rng([R(k, 'pads_spin_1_over_law') for k in C.QUARTERS])} on the quarter-ellipsoids and "
        f"{rng([R(k, 'pads_spin_1_over_law') for k in C.BLOCKS])} on the blocks, and Drake&#8217;s "
        f"{rng([R(k, 'drake_spin_1_over_law') for k in ORDER])}. From 0.3 to 3&#8202;rad/s the pads&#8217; spin arm grows by "
        f"a factor {rng([R(k, 'pads_spin_3_over_0.3') for k in C.BLOCKS])} on the blocks and "
        f"{rng([R(k, 'pads_spin_3_over_0.3') for k in curved + C.QUARTERS])} on the curved tips. The pads&#8217; brake force "
        f"lies within {rng([R(k, 'pads_N45_over_drake') for k in curved + C.QUARTERS])} of Drake&#8217;s on the curved tips "
        f"and {rng([R(k, 'pads_N45_over_drake') for k in C.BLOCKS])} on the blocks.</p>"
        f"<p>On the quarter-ellipsoids each pad&#8217;s depth is taken to the rigid back plane, which makes the pads "
        f"{rng([R(k, 'pads_approach_1N_over_law') for k in C.QUARTERS])} of the law&#8217;s approach; the law measures "
        f"depth to the volume centroid. On 15 October the ratio was 1.26&#8211;1.32.</p>")
    return text + fig + tab


def cost_section():
    if not COST:
        return "<p><i>[cost not run]</i></p>"
    body = []
    for k in ORDER:
        t = COST.get(k)
        if not t:
            continue
        a, b, g = t["cpu"].get("cylinder:16:0.625|pads", {}), t["cpu"].get("cylinder:25:0.125|pads", {}), t.get("gpu", {})
        o = COST15.get(old(k), {}).get("gpu", {})
        body.append([name(k), f(a.get("step_us"), 0), f(a.get("ncon"), 0), f(a.get("solver_niter")), f(b.get("step_us"), 0),
                     f(a.get("replan_3thread_s")), f(g.get("us_per_world_step")), f(g.get("contacts_per_world"), 0),
                     f((g.get("gpu_peak_mb") or 0) / 1000 if g.get("gpu_peak_mb") else None, 1), f(o.get("us_per_world_step")),
                     f((o.get("gpu_peak_mb") or 0) / 1000 if o.get("gpu_peak_mb") else None, 1)])
    tab = table(["Tip", "CPU step, 16&#8202;mm cell, &#181;s", "Contacts", "Solver iterations", "CPU step, 25&#8202;mm cell, &#181;s",
                 "Replanning 64 plans, 3 threads, s", "GPU world-step, &#181;s", "GPU contacts per world", "GPU memory, GB",
                 "GPU world-step at 10&#8202;MPa (15 Oct), &#181;s", "GPU memory (15 Oct), GB"], body,
                "Cost by the pad-usability protocol at 2.5&#8202;MPa (cost/summary.json): MuJoCo at the SCS0009&#8217;s "
                "three-finger grip on two v4b-hand cells, and MuJoCo-Warp at the held state of an RL run with 2048 worlds.",
                key="cost")
    return "<p>[cost prose]</p>" + tab


def screen_section():
    T = SCR.get("tips", {})
    if not T:
        return "<p><i>[screen not run]</i></p>"
    o15 = SCR15.get("tips", {})
    body = []
    for k in ORDER:
        v = T.get(k)
        if not v:
            continue
        ov = o15.get(old(k)) or {}
        body.append([name(k), f(v.get("J1"), 3), f(v.get("J2"), 3), f(ov.get("J2"), 3),
                     f"{v['agree_drake_3deg']}/{v['n_vs_drake']}" if v.get("agree_drake_3deg") is not None else "&#8211;",
                     "Yes" if v.get("gate_pass") else "No", f(v.get("cost_ratio")), f(v.get("three_finger_frac_mu0.5")),
                     f(get(v, "rigid_share_all", "median"))])
    tab = table(["Tip", "J1", "J2", "J2 (15 Oct)", "Agree with Drake", "Gate", "Replay cost over v4b", "Three-finger share, "
                 "friction 0.5", "Rigid share, median"], body,
                "Stages 1 and 2 (screen/summary.json), with J2 of 15 October on the first CAD hand at 10&#8202;MPa.",
                key="screen")
    fig = img("media/fig_screen.png", "Stages 1 and 2 per tip: the stage score at each of the five conditions with J2, their "
              "minimum, as the bar (dash: J2 of 15 October); the D1&#8211;D8 replays within 3&#176; of Drake with the gate; the "
              "rigid parts&#8217; share of the normal force in the cells&#8217; HOM turns.", key="screen")
    out = "<p>[screen prose]</p>" + fig + tab
    C3 = FIN.get("candidates", {})
    if C3:
        b3 = []
        for k in FIN["tips"]:
            v = C3[k]
            ci = v.get("dJ3_ci90")
            b3.append([name(k), f(v["J3"], 3), f(v.get("dJ3_vs_base"), 3) if ci else "&#8211;",
                       f"{ci[0]:+.3f} to {ci[1]:+.3f}" if ci else "&#8211;", f"{v['hold3_mu0.5']} of {v['n_hold_mu0.5']}"])
        out += table(["Tip", "J3", "J3 minus v4b", "90&#8202;% interval", "Three-finger holds at friction 0.5"], b3,
                     "Stage 3 on fresh placements and the paired bootstrap difference against the v4b tip (10,000 resamples "
                     "within hands and cells; final/summary.json).", key="stage3")
        out += img("media/fig_stage3.png", "Stage 3 per tip: the score at each condition and J3 (left), and the difference to "
                   "the v4b tip with its 90&#8202;% bootstrap interval (right).", key="stage3")
    out += film("media/hom_cylinder_16_0.625_s3_tile.mp4", "media/hom_cylinder_16_0.625_s3_tile_poster.jpg",
                "HOM on the v4b hand&#8217;s 16&#8202;mm cylinder at friction 0.7, seed 3 (a placement no stage used to choose the "
                "grasp), each tip with the grasp it chose on seed 0, seen from below the palm; pads touching the cylinder red, "
                "a rigid part touching it yellow.", key="hom")
    return out


def fe_section():
    if not FE:
        return "<p><i>[finite elements not run]</i></p>"
    body = []
    for k in ORDER:
        t = FE.get(k)
        if not t:
            continue
        s = t.get("sensing") or {}
        body.append([name(k), f(get(t, "sphere_apex_um", "1"), 0), f(get(t, "rod_across_um", "1"), 0), f(t.get("rod_along_1N_um"), 0),
                     f(t.get("exponent_sphere")), f(t.get("exponent_rod")), f(get(t, "twist_arm_rod_mm", "1")),
                     f(t.get("sphere_map_spread")),
                     f(s.get("centre_force_N"), 3) if s.get("centre_force_N") is not None else ("No fit" if not t["magnets"] else "&#8211;"),
                     f(s.get("B_max_component_mT"), 1)])
    tab = table(["Tip", "Sphere at the apex, 1&#8202;N, &#181;m", "Rod across, 1&#8202;N, &#181;m", "Rod along, 1&#8202;N, &#181;m",
                 "Exponent, sphere", "Exponent, rod", "Twist arm under the rod, 1&#8202;N, mm", "Sphere map, largest over least",
                 "Force resolution, N", "Largest field component, mT"], body,
                "Finite elements per tip (fe/summary.json; <code>scripts/fingertip_shape_rerun_fe.py</code>).", key="fe")
    return "<p>[FE prose]</p>" + img("media/fig_fe.png", "[FE figure caption]", key="fe") + tab


def compute_section():
    jobs = CMP.get("jobs", {})
    if not jobs:
        return "<p><i>[compute summary not written]</i></p>"
    body = [[k, str(v["n"]), f(v["core_h"]), f(v["wall_h"])] for k, v in sorted(jobs.items())]
    return table(["Jobs", "Count", "Core-hours", "Wall time, h"], body, "Compute of the rerun by kind of job "
                 "(compute/summary.json, from logs/20261104-fingertip_shape_rerun/queue/jobs.jsonl).", key="compute")


def limitations():
    return ("<p>Every reference is computed: the finite elements use NinjaFlex 85A&#8217;s 12&#8202;MPa solid modulus and a "
            "homogenized gyroid, and Drake&#8217;s field and the pads use the 2.5&#8202;MPa fitted to the v3 tips&#8217; UR5 presses. "
            "Ten of the twelve tips would need a cassette and cap of their own; they are simulated with the 11/22&#8217;s. The "
            "D1&#8211;D8 hands are real_v1 hands and their bench plans were made for the block. No tip other than the 11/22 has "
            "been printed on the v4b finger.</p>")


def open_items():
    items = [("[future work]", "[to write from the results]")]
    return "<ol>" + "".join(f"<li><b>{a}.</b> {b}</li>" for a, b in items) + "</ol>"


def lede():
    return "[lede: the outcome with its numbers, written when the stages are done]"


def byline():
    return ("Scripts <code>scripts/fingertip_shape_rerun_{common,mount,rig,traj,bed,cost,fe,queue,analysis,figures,film,page}.py"
            "</code> &#183; rows <code>docs/experiments/20261104-fingertip_shape_rerun/</code> &#183; hand-off "
            "<code>docs/handoff/20261103-fingertip_shape_rerun.md</code>")


def main():
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "GLOSSARY": glossary(), "QUESTION": question(), "MOUNT": mount_section(),
         "BED": bed_section(), "COST": cost_section(), "SCREEN": screen_section(), "FE": fe_section(),
         "COMPUTE": compute_section(), "LIMITS": limitations(), "OPEN": open_items(), "LEDE": lede()}
    v["FOOTER"] = ("<p>Rebuild: <code>python3 scripts/fingertip_shape_rerun_analysis.py all</code>, the figures, then "
                   "<code>.venv/bin/python scripts/fingertip_shape_rerun_page.py</code>. Earlier studies: "
                   "<a href=\"../20261014-tip_codesign/20261014-sphere_pad_accuracy_and_fingertip_codesign.html\">sphere-pad "
                   "accuracy and fingertip co-design</a> (14 October), <a href=\"../20261015-curved_tips/"
                   "20261015-capsule_and_ellipsoid_fingertips.html\">capsule and ellipsoid fingertips</a> (15 October), "
                   "<a href=\"../20261018-fe_contact_calibration/20261018-fe_contact_calibration.html\">fingertip contact "
                   "calibration</a> (18 October).</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    miss = sorted(set(re.findall(r"\[\[((?:fig|tab|sec|film):[a-z0-9_]+)\]\]", t)) - set(REFS))
    if miss:
        raise SystemExit(f"unresolved references: {miss}")
    t = re.sub(r"\[\[((?:fig|tab|sec|film):[a-z0-9_]+)\]\]", lambda m: str(REFS[m.group(1)]), t)
    t, n = render_tex(t)
    t = retro_style.apply(t)
    t = docs_index.tag(t, topic="fingertip", summary=re.sub(r"<[^>]+>", "", lede()), date="2026-11-04")
    open(OUT, "w").write(t)
    print(f"formulas {n}; figures {FIG[0]}; tables {TAB[0]}; films {FILM[0]}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
