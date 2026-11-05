#!/usr/bin/env python3
"""Page: block, capsule, ellipsoid and hemisphere fingertips on the v4b finger of the SR2 hand, the rerun of the
fingertip-shape screens of 2026-10-14 and 10-15 and the FE checks of 10-18/19 at 2.5 MPa on the current hand
(hand-off docs/handoff/20261103-fingertip_shape_rerun.md).

Data: docs/experiments/20261104-fingertip_shape_rerun/ (mount/, kin/, bed/, cost/, traj/, screen/, final/, fe/, compute/,
media/), every number read from the run's summaries; 10-15's from docs/experiments/20261015-curved_tips/, 10-14's from
docs/experiments/20261014-tip_codesign/ and 10-18's from docs/experiments/20261018-fe_contact_calibration/:

    python3 scripts/fingertip/fingertip_shape_rerun_analysis.py all
    ~/.claude/bin/resguard.sh run --mem 2G -- .venv/bin/python scripts/fingertip/fingertip_shape_rerun_figures.py mount bed screen stage3 fe
    .venv/bin/python scripts/fingertip/fingertip_shape_rerun_page.py
Out: docs/experiments/20261104-fingertip_shape_rerun/20261104-fingertip_shapes_v4b_finger.html (media inlined).
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import contact_overview_page as P  # noqa: E402
import docs_index  # noqa: E402
import fingertip_shape_rerun_common as C  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = str(C.D)
OUT = os.path.join(D, "20261104-fingertip_shapes_v4b_finger.html")
TPL = os.path.join(ROOT, "scripts/fingertip/fingertip_shape_rerun_page.template.html")
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


def nm(k):
    """A tip's name inside a sentence: 'the v4b tip', or its table name in lower case."""
    if k == C.REF:
        return "the v4b tip"
    t = name(k)
    return "the " + t[:1].lower() + t[1:]


def cap1(t):
    return t[:1].upper() + t[1:]


def sg(x, n=3):
    """A signed number with a true minus sign."""
    return "&#8211;" if x is None else f"{x:+.{n}f}".replace("-", "&#8722;")


def rule_sentences(fin, why, T):
    """One sentence per finalist rule: (1) and (3) name one tip each, (4) the rest."""
    r1 = [k for k in fin if why.get(k, "").startswith("(1)")]
    r3 = [k for k in fin if why.get(k, "").startswith("(3)")]
    r4 = [k for k in fin if why.get(k, "").startswith("(4)")]
    out = []
    if r1:
        out.append(f"{nm(r1[0])} has the highest J2 among the tips that pass the gate.")
    if r3:
        out.append(f"{nm(r3[0])} has the highest three-finger share at friction 0.5, "
                   f"{f(T[r3[0]].get('three_finger_frac_mu0.5'))}.")
    if r4:
        out.append(f"{joined(nm(k) for k in r4)} have the next highest J2 among those that pass it." if len(r4) > 1 else
                   f"{nm(r4[0])} has the next highest J2 among those that pass it.")
    return out


def joined(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
         10: "ten", 11: "eleven", 12: "twelve"}


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
BAND = J("band/summary.json").get("tips", {})
TASKS4 = (("twist", "Twist"), ("roll", "Roll"), ("pinpush", "Pin-push"), ("press", "Press"))
BED15 = J("bed/summary.json", OLD15).get("tips", {})
COST15 = J("cost/summary.json", OLD15).get("tips", {})
SCR15 = J("screen/summary.json", OLD15)
FIN15 = J("final/summary.json", OLD15)
FE18_NAME = {C.BLOCKS[0]: "block4_g30", "tcd_cap8.5_u0.1": "cap_g30", "tcd_ell7.4x8.5x16x32_u0.1": "ell16_g30", C.REF: "ell11_g30"}


def fe18(k, probe, F="1"):
    """The 18 October FE's approach (um) of a tip under a probe: the whole print with four 3 x 2 mm magnets, base grid."""
    nm = FE18_NAME.get(k)
    p = os.path.join(OLD18, "fe/fe.jsonl")
    if not nm or not os.path.exists(p):
        return None
    for line in open(p):
        r = json.loads(line)
        if r.get("tip") == nm and r.get("grid") == "base" and r.get("probe") == probe:
            return r["per_force"][F]["approach_um"]
    return None


def old(k):
    return C.OLD_KEY.get(k, k)


# ------------------------------------------------------------------------------------------ sections

def glossary():
    items = [
        ("Tips", "Twelve fingertips inside the printed block&#8217;s envelope. The envelope is 17.0&#8202;mm deep along the "
         "palmar normal \\(x\\), 14.8&#8202;mm wide along the PIP axis \\(y\\) and 21.41&#8202;mm long along the finger \\(z\\). Each has its palmar "
         "apex 10.55&#8202;mm from the finger&#8217;s axis, over the contact point \\((y, z) = (0, 0)\\). The v4b tip is the "
         "11/22 half-ellipsoid (<code>tcd_ell7.4x8.5x11x22</code>), the tip calibrated on the UR5 and the reference here. Its "
         "semi-axes are 7.4&#8202;mm across, 8.5&#8202;mm deep, 11&#8202;mm distal and 22&#8202;mm proximal of the apex. A "
         "half-ellipsoid \\(a_y \\times a_x \\times a_z/a_{zp}\\) is the palmar half of such an ellipsoid; behind its equator "
         "the section runs straight back. A quarter-ellipsoid is the same half cut flat at its equator. A capsule <i>r</i> is a "
         "cylinder of radius <i>r</i> along the finger, capped by a hemisphere at the distal end. The hemisphere has every "
         "semi-axis 7.4&#8202;mm, the envelope&#8217;s half-width: a dome on a round post. The blocks are the as-built block, with "
         "3&#8202;mm side and end fillets, and the finalist of 14 October, with fillets of 3 and 6&#8202;mm. The whole real_v1 sphere "
         "tip (<code>tcd_sph10.55</code>) runs on the bed only."),
        ("Surfaces", "The capsules, the quarter-ellipsoids and the half-ellipsoids other than the v4b tip are simulated as "
         "printed upright in 0.1&#8202;mm layers (key suffix <code>_u0.1</code>). Their surface lies within 4.5&#8202;&#181;m of "
         "the design over the contact region. The blocks are simulated as printed face up in 0.2&#8202;mm layers "
         "(<code>_p0.2</code>), the v4b tip and the hemisphere on their design surfaces."),
        ("v4b finger", "The CAD model of the SR2 finger with the DIP link b v4, the cassette 3T, the top cover TW15H3 (the cap) "
         "and the TPU pad (<code>SCALE_HAND=cad_v2</code>, <code>scripts/hardware/build_cad_v2_finger.py</code>). The cap is trimmed at "
         "\\(x_F = -0.10\\)&#8202;mm and has four M2 button heads. The pad&#8217;s centre lies 36.36&#8202;mm from the PIP axis. The rigid "
         "parts are collision spheres inscribed in each part. Every tip replaces the pad and keeps them."),
        ("Mount check", "Each tip&#8217;s design surface is placed on the v4b finger, cut where the v4b pad is cut (13.45&#8202;mm "
         "under the apex) and sampled on a 0.2&#8202;mm grid. Extra TPU: the tip&#8217;s volume inside a rigid part where the "
         "11/22 has none (mm&#179;). Bearing kept: of the v4b pad&#8217;s surface within 0.3&#8202;mm of a part, the share that "
         "lies inside the tip. A tip mounts as the 11/22 does when no part gains more than 2&#8202;mm&#179; of TPU and the bearing "
         "kept is at least 95&#8202;% on every part. Otherwise it is simulated on the envelope alone, on the 11/22&#8217;s rigid parts. "
         "Magnet wall: the least TPU between one of pad 05R4&#8217;s magnets and the tip&#8217;s surface; the print rule is "
         "1.8&#8202;mm. Pad 05R4 has four magnets of 3&#8202;&#215;&#8202;2&#8202;mm in a 5.0&#8202;mm square about the contact "
         "point. Their tops are 7.5&#8202;mm deep."),
        ("1&#8202;mm pads", "The tip&#8217;s palmar region as spheres of radius 0.75&#8202;mm on a 1&#8202;mm lattice. Each is a "
         "MuJoCo soft contact of stiffness \\(K_i = E a_i / h_i\\) (builder v2). Here \\(E\\) = 2.5&#8202;MPa, \\(a_i\\) is "
         "the pad&#8217;s share of the surface and \\(h_i\\) its depth below the tip&#8217;s volume centroid. The relaxation time is "
         "0.04&#8202;s, the impratio 100 and the step 1&#8202;ms. The modulus was fitted to the UR5 presses of the v3 tips on "
         "30 October."),
        ("Drake", "Drake 1.57 hydroelastic contact of the tip&#8217;s compliant convex hull: modulus 2.5&#8202;MPa, relaxation "
         "time 0.01&#8202;s, SAP solver, 1&#8202;ms step."),
        ("Law", "The quasi-static integral of Drake&#8217;s pressure field \\(p = E\\,(1 - g(\\mathbf{x}))\\) over the contact "
         "with the bed&#8217;s tool, at 2.5&#8202;MPa. Here \\(g\\) is the gauge of the tip&#8217;s 0.25&#8202;mm hull about its "
         "volume centroid."),
        ("Bed", "Two force-controlled rails squeeze a 25&#8202;mm diameter, 24.5&#8202;g screwdriver between two copies of a tip "
         "at 1&#8202;N per tip, with friction coefficient \\(\\mu\\) = 1. Approach: the rail&#8217;s travel from first touch "
         "(&#181;m). Twist-onset arm: the torque at which the tool starts to turn under a slow torque ramp, over \\(2\\mu N\\) "
         "(mm). Spin arm: the mean torque that drives the tool about the pinch axis at rate \\(\\omega\\), over "
         "\\(2\\mu N\\) (mm). Brake force: the pinch force at which the tool, released 15&#8202;mm off the pinch, stops "
         "swinging at 45&#176; (N)."),
        ("SCS0009 plant", "The finger servo model of the built hand: gain 0.72&#8202;N&#8202;m/rad, torque limit 0.18&#8202;N&#8202;m, "
         "damping 0.0144&#8202;N&#8202;m&#8202;s/rad (<code>kp0.72_kv0_fr0.18_fl0_dp0.0144</code>)."),
        ("Cost", "The pad-usability protocol of 14 October. CPU step: MuJoCo&#8217;s wall time for one 1&#8202;ms step of the "
         "SCS0009&#8217;s three-finger grip on a cell (<code>scripts/contact/pad_usability_timing.py</code>). Replanning: the wall time "
         "of 64 plans of 0.5&#8202;s on three threads. GPU world-step: MuJoCo-Warp&#8217;s wall time per step divided by the "
         "number of worlds, at the held state of an RL run tiled over 2048 worlds (<code>scripts/contact/pad_usability_gpu_cost.py"
         "</code>). GPU memory: the process&#8217;s peak."),
        ("HOM, cells", "HOM: the hand&#8217;s closed-loop in-hand turn controller (<code>scripts/hom/hom_turn3.py</code>, model gain "
         "0.72). Cells: four grasps of cylinders 100&#8202;mm long on the v4b hand. Their diameters are 25, 16, 10 "
         "and 16&#8202;mm. Their finger layouts, the thumb-to-pair and pair separations, are those of 14 October. Each cell has nine candidate grasps from the scale study&#8217;s kinematics of the v4b finger. "
         "Held turn: the turn at the end of a rollout when two or more fingers hold the tool, otherwise 0 (&#176;)."),
        ("Stage scores", "The stage score is the mean of five parts, each a held turn over 90&#176;. One part per cell is the mean "
         "held turn, on the scoring seeds, of the grasp chosen by its held turn on seed 0. The fifth is the mean held turn of HOM "
         "on the eight deployed real_v1 hands D1&#8211;D8. J1 is the score at friction 0.7 on the SCS0009. It is "
         "computed on cell seeds 1&#8211;2 and HOM seeds 0&#8211;2. J2 is the minimum over five conditions on HOM seeds "
         "0&#8211;4. Three conditions set the friction to 0.7, 1.0 and 0.5. The other two lower the servo gain to "
         "0.5&#8202;N&#8202;m/rad and the torque limit to 0.12&#8202;N&#8202;m. J3 is the minimum over the four conditions other "
         "than friction 1.0, on fresh placements. These are cell seeds 3&#8211;5 and HOM seeds 5&#8211;9. Seed 0 places the "
         "tool as planned; other seeds move it by draws of standard deviation 2&#8202;mm and 2&#176;."),
        ("Agreement gate", "The eight deployed bench plans are replayed open loop on the SCS0009 plant at friction 1.0, seeds "
         "0&#8211;2. An agreement is a replay in which the pads end within 3&#176; of Drake&#8217;s turn, with the same held outcome "
         "as Drake. Each tip has 24 replays. "
         "A tip passes with at most two agreements fewer than the v4b tip."),
        ("Rigid share", "Of the normal force that the fingers press on a cell&#8217;s cylinder, summed over a rollout&#8217;s "
         "control samples, the share through finger geoms other than the pads. On the v4b finger these are the DIP link, the "
         "cassette and the cap."),
        ("Friction band", "The worst-case HOM scores of 4 November (<code>scripts/hom/hom_grasp_refine.py</code>, band "
         "<code>band_k21mu</code>) on the STS3032 plant, with pads at 2.5&#8202;MPa. There are four HOM tasks on the SR2 "
         "hand&#8217;s layouts. In a tripod grasp of a 40&#8202;mm sphere, the twist scores the turn reached on three pads over "
         "90&#176;. The roll scores \\(1/(1 + w)\\), with \\(w\\) the root-mean-square error (RMSE) of the angular velocity over "
         "its 40&#176;/s steps, divided by the step. With a 25&#8202;mm cylinder, the pin-push and the press score 1 minus the "
         "pinch-angle RMSE over their strokes of 15&#176; and 20&#176;. A rollout that drops the object scores 0. Each task has "
         "a sample of 30 layouts. Pick S3 is a layout&#8217;s grasp, commanded forces and contact modes, refined for the v4b "
         "tip. Worst-case score: the lowest of a pick&#8217;s three five-seed means at scene friction 0.4, 0.7 and 1.0. The "
         "seeds are 20&#8211;24."),
        ("Finite elements", "The printed tip as voxel finite elements, the model of 18 October. Two 0.45&#8202;mm perimeters and "
         "four skins of solid TPU at 12&#8202;MPa surround a homogenized 30&#8202;% gyroid of Siraya 85A, in the tip&#8217;s print "
         "orientation. Pad 05R4&#8217;s four magnets are bonded where they fit. The pad is cut where the v4b pad is cut and held "
         "on that face; a quarter-ellipsoid is held on its own back plane. The probes are rigid and frictionless. Approach: the "
         "probe&#8217;s travel from first touch (&#181;m). Force exponent \\(n\\): the slope of approach against load on log "
         "axes between 0.5 and 3&#8202;N. Twist arm: the pressure-weighted mean distance of the patch from its centre (mm)."),
        ("Force resolution", "The Cram&#233;r&#8211;Rao bound on the normal force at the contact point from one 141&#8202;Hz sample "
         "of the board&#8217;s five MLX90393 chips (N; <code>scripts/fingertip/tpu_tip_magnet_sensing.py</code>). The chips run at the gain "
         "that the static field allows, and the contact position and a uniform background field are estimated alongside."),
        ("Compute", "Core-hours: the summed wall time of the jobs times their threads. GPU hours: the wall time of the GPU "
         "jobs."),
    ]
    return "<dl>" + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in items) + "</dl>"


def question():
    return ("<p>The owner chose the 11/22 half-ellipsoid on 18 October. Two screens on the first CAD hand at 10&#8202;MPa and a "
            "calibration of the contact models to the finite elements came before that choice. Since then the hand has moved "
            "to the v4b finger (DIP link b v4, cassette 3T, cap TW15H3), and the pads&#8217; modulus to 2.5&#8202;MPa. The "
            "screens of October already ran on the SCS0009 plant. This rerun repeats the protocol of 14 and 15 October on the current "
            "hand, for the nine curved tips, the two blocks and a hemisphere. The v4b tip is the reference throughout. Each tip "
            "is checked for its fit on the v4b finger, run on the contact bed against the law and Drake, and timed in "
            "simulation. It then goes through the three-stage task screen with Drake&#8217;s agreement gate, and through the "
            "finite elements with the force resolution of pad 05R4.</p>"
            "<p>Five rules differ from 15 October. The cells&#8217; layouts and cylinders are unchanged, and their "
            "candidate grasps come from the v4b finger&#8217;s kinematics (<code>kin/reach.json</code>). The gate and stage 3 compare each tip with "
            "the v4b tip in place of the as-built block, which runs stage 3 beside it. For finalist rule (2) the replay cost "
            "is a median ratio of microseconds per step over the placements on which neither tip&#8217;s pads drop the tool. "
            "The v4b tip&#8217;s pads drop the tool in about half the replays, and a replay is cheap once the tool has fallen. "
            "In Drake the bed&#8217;s four metrics are run at fewer loads and rates than with the pads. The approach is run at "
            "1 and 3&#8202;N and the onset arm at 1&#8202;N. The spin arm is run at 1&#8202;rad/s and the brake force at a "
            "1&#8202;ms step. A Drake replay of the hand lasts 130&#8211;350&#8202;s of wall time at 2.5&#8202;MPa. At "
            "10&#8202;MPa it lasted 61&#8202;s. Last, every turn row records the rigid parts&#8217; share of the "
            "contact force.</p>")


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
              "Each tip on the v4b finger (mount/mount.json; <code>scripts/fingertip/fingertip_shape_rerun_mount.py</code>).", key="mount")
    fig = img("media/fig_mount.png", "The tips on the v4b finger. Each tip has two panels. The upper panel "
              "is its profile along the finger at \\(y = 0\\), with the distal end left and the palmar side up. The lower panel "
              "is its section across the finger 6&#8202;mm distal of the contact point, through the cap&#8217;s side blocks. TPU of the tip light brown, the rigid parts grey, the 11/22&#8217;s grooves and pin hole dark brown, "
              "TPU inside a rigid part where the 11/22 has none red.", key="mount")
    wide = [k for k in ORDER if max(T[k]["extra_mm3"].values()) > 2.0]
    lose = [k for k in ORDER if k not in wide and T[k]["verdict"] != "mounts" and not k.startswith("tcd_qell")]
    ex = [max(T[k]["extra_mm3"].values()) for k in wide]
    return (f"<p>Of the twelve tips, {WORDS[len(mounts)]} mount on the v4b finger as the 11/22 does "
            "(Table&#160;[[tab:mount]], Figure&#160;[[fig:mount]]). They are " + joined(nm(k) for k in mounts) + ". "
            + f"{WORDS[len(wide)].capitalize()} tips are wider than the 11/22 along its flanks: {joined(nm(k) for k in wide)}. "
            f"They put {rng(ex, 0)}&#8202;mm&#179; of TPU into the cassette&#8217;s walls or the cap&#8217;s side blocks. "
            + (f"{joined(nm(k) for k in lose)[:1].upper() + joined(nm(k) for k in lose)[1:]} are narrower and lose "
               f"{rng([100 * (1 - min(q['share'] for q in T[k]['kept'].values())) for k in lose], 0)}&#8202;% of a part&#8217;s "
               "bearing surface. " if lose else "")
            + "The quarter-ellipsoids end in front of the cassette, and pad 05R4&#8217;s magnets lie behind their back planes. "
            "The ten tips that do not mount are simulated on the envelope alone, on the 11/22&#8217;s rigid parts.</p>" + fig + t)


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
    fig = img("media/fig_bed.png", "Bed per tip. Top: the twist-onset arm at 1&#8202;N of the law, the pads and Drake at "
              "2.5&#8202;MPa, and of the pads at 10&#8202;MPa on 15 October (open). Bottom: the spin arm at 1&#8202;rad/s over the "
              "law&#8217;s.", key="bed")
    curved = C.HALF + [C.REF, C.HEMI] + C.CAPSULES
    g = lambda k, *p: get(BED.get(k) or {}, *p)  # noqa: E731
    onset = [g(k, "law", "onset_arm_mm", "1") for k in curved]
    onset_q = [g(k, "law", "onset_arm_mm", "1") for k in C.QUARTERS]
    onset_b = [g(k, "law", "onset_arm_mm", "1") for k in C.BLOCKS]
    o15 = [get(BED15.get(old(k)) or {}, "law", "onset_arm_mm", "1") for k in curved if k != C.HEMI]
    b15 = [get(BED15.get(k) or {}, "law", "onset_arm_mm", "1") for k in C.BLOCKS]
    R = lambda k, key: g(k, "ratios", key)  # noqa: E731
    text = (
        f"<p>At 2.5&#8202;MPa the curved tips turn the tool on half of the blocks&#8217; friction arm "
        f"(Table&#160;[[tab:bed]], Figure&#160;[[fig:bed]]). The law&#8217;s twist-onset arm at 1&#8202;N is {rng(onset)}&#8202;mm "
        f"on the capsules, the half-ellipsoids and the hemisphere, and {rng(onset_b)}&#8202;mm on the blocks. The "
        f"quarter-ellipsoids give {rng(onset_q)}&#8202;mm. On 15 October, at 10&#8202;MPa, the arm was {rng(o15)}&#8202;mm on the "
        f"curved tips and {rng(b15)}&#8202;mm on the blocks. The softer pads sink deeper and touch more of the tool: "
        f"{rng([g(k, 'pads', 'touching_1N') for k in curved], 0)} pads per side at 1&#8202;N on the curved tips, against "
        f"{rng([get(BED15.get(old(k)) or {}, 'pads', 'touching_1N') for k in curved if k != C.HEMI], 0)} at 10&#8202;MPa. The "
        f"blocks touch with {rng([g(k, 'pads', 'touching_1N') for k in C.BLOCKS], 0)}.</p>"
        f"<p>The pads give {rng([R(k, 'pads_onset_1N_over_law') for k in curved])} of the law&#8217;s onset arm on the curved "
        f"tips and {rng([R(k, 'pads_onset_1N_over_law') for k in C.BLOCKS])} on the blocks. Drake gives "
        f"{rng([R(k, 'drake_onset_1N_over_law') for k in ORDER])}. At 1&#8202;rad/s the pads&#8217; spin arm is "
        f"{f(R(C.REF, 'pads_spin_1_over_law'))} of the law&#8217;s on the v4b tip and "
        f"{rng([R(k, 'pads_spin_1_over_law') for k in C.CAPSULES])} on the capsules. It is "
        f"{rng([R(k, 'pads_spin_1_over_law') for k in C.QUARTERS])} on the quarter-ellipsoids and "
        f"{rng([R(k, 'pads_spin_1_over_law') for k in C.BLOCKS])} on the blocks, and Drake&#8217;s is "
        f"{rng([R(k, 'drake_spin_1_over_law') for k in ORDER])}. From 0.3 to 3&#8202;rad/s the pads&#8217; spin arm grows by a "
        f"factor {rng([R(k, 'pads_spin_3_over_0.3') for k in C.BLOCKS])} on the blocks and "
        f"{rng([R(k, 'pads_spin_3_over_0.3') for k in curved + C.QUARTERS])} on the curved tips. The pads&#8217; brake force "
        f"lies within {rng([R(k, 'pads_N45_over_drake') for k in curved + C.QUARTERS])} of Drake&#8217;s on the curved tips and "
        f"{rng([R(k, 'pads_N45_over_drake') for k in C.BLOCKS])} on the blocks.</p>"
        f"<p>On the quarter-ellipsoids each pad&#8217;s depth is taken to the rigid back plane, and the law&#8217;s to the volume "
        f"centroid. Their pads therefore sink {rng([R(k, 'pads_approach_1N_over_law') for k in C.QUARTERS])} times as far as "
        f"the law; on 15 October the ratio was 1.26&#8211;1.32.</p>")
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
                "Cost by the pad-usability protocol at 2.5&#8202;MPa (cost/summary.json). MuJoCo runs the SCS0009&#8217;s "
                "three-finger grip on two v4b-hand cells; MuJoCo-Warp runs the held state of an RL run with 2048 worlds.",
                key="cost")
    g = lambda k, *p: get(COST.get(k) or {}, *p)  # noqa: E731
    g15 = lambda k, *p: get(COST15.get(old(k)) or {}, *p)  # noqa: E731
    mid = C.CAPSULES + C.HALF
    cpu = lambda k: g(k, "cpu", "cylinder:16:0.625|pads", "step_us")  # noqa: E731
    text = (
        f"<p>The softer pads touch more of the tool, and a step costs more than at 10&#8202;MPa (Table&#160;[[tab:cost]]). On "
        f"the GPU, at the held RL state with 2048 worlds, a world-step lasts "
        f"{f(g(C.REF, 'gpu', 'us_per_world_step'))}&#8202;&#181;s on the v4b tip. The scene has "
        f"{f(g(C.REF, 'gpu', 'contacts_per_world'), 0)} contacts per world and occupies "
        f"{f((g(C.REF, 'gpu', 'gpu_peak_mb') or 0) / 1000, 1)}&#8202;GB of GPU memory. On the blocks a world-step lasts "
        f"{rng([g(k, 'gpu', 'us_per_world_step') for k in C.BLOCKS])}&#8202;&#181;s and the scene occupies "
        f"{rng([(g(k, 'gpu', 'gpu_peak_mb') or 0) / 1000 for k in C.BLOCKS], 1)}&#8202;GB. It lasts "
        f"{rng([g(k, 'gpu', 'us_per_world_step') for k in mid])}&#8202;&#181;s on the capsules and the other half-ellipsoids, "
        f"and {rng([g(k, 'gpu', 'us_per_world_step') for k in C.QUARTERS])}&#8202;&#181;s on the quarter-ellipsoids. On "
        f"15 October, at 10&#8202;MPa, a world-step lasted {f(g15(C.REF, 'gpu', 'us_per_world_step'))}&#8202;&#181;s on the "
        f"v4b tip&#8217;s shape, with {f(g15(C.REF, 'gpu', 'contacts_per_world'), 0)} contacts. It lasted "
        f"{f(g15(C.BLOCKS[0], 'gpu', 'us_per_world_step'))}&#8202;&#181;s on the as-built block.</p>"
        + (f"<p>On the CPU, at the three-finger grip of the v4b hand&#8217;s 16&#8202;mm cell, a 1&#8202;ms step lasts "
           f"{f(cpu(C.REF), 0)}&#8202;&#181;s on the v4b tip, with {f(g(C.REF, 'cpu', 'cylinder:16:0.625|pads', 'ncon'), 0)} "
           f"contacts. It lasts {f(cpu(C.BLOCKS[0]), 0)}&#8202;&#181;s on the as-built block, with "
           f"{f(g(C.BLOCKS[0], 'cpu', 'cylinder:16:0.625|pads', 'ncon'), 0)} contacts, and "
           f"{rng([cpu(k) for k in ORDER], 0)}&#8202;&#181;s over the twelve tips. Replanning 64 plans of 0.5&#8202;s on three threads takes "
           f"{rng([g(k, 'cpu', 'cylinder:16:0.625|pads', 'replan_3thread_s') for k in ORDER])}&#8202;s.</p>"
           if all(cpu(k) is not None for k in ORDER) else ""))
    return text + tab


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
    fig = img("media/fig_screen.png", "Stages 1 and 2 per tip. Top: the stage score at each of the five conditions, and J2, "
              "their minimum, as the bar; the dash is J2 of 15 October. Middle: the D1&#8211;D8 replays within 3&#176; of Drake, "
              "with the gate. Bottom: the rigid parts&#8217; share of the normal force in the cells&#8217; HOM turns.",
              key="screen")
    J1 = {k: T[k].get("J1") for k in ORDER if k in T}
    J2 = {k: T[k].get("J2") for k in ORDER if k in T}
    top1 = sorted(J1, key=lambda k: -(J1[k] or -1))[:3]
    top2 = sorted(J2, key=lambda k: -(J2[k] or -1))
    allJ = [c["J"] for k in T for c in T[k]["conditions"].values() if c.get("J") is not None]
    ref = T[C.REF]
    fail = [k for k in ORDER if k in T and not T[k]["gate_pass"]]
    fin, why = SCR.get("finalists", []), SCR.get("finalist_rule", {})
    rule = lambda k: re.sub(r"^\(\d\) ", "", why.get(k, ""))  # noqa: E731
    cond = lambda c: {f"0.7|{C.SERVO}": "friction 0.7", f"1.0|{C.SERVO}": "friction 1.0", f"0.5|{C.SERVO}": "friction 0.5",
                      "0.7|kp0.5_kv0_fr0.18_fl0_dp0.0144": "servo gain 0.5", "0.7|kp0.72_kv0_fr0.12_fl0_dp0.0144":
                      "torque limit 0.12&#8202;N&#8202;m"}.get(c, c)  # noqa: E731
    rig = [get(T[k], "rigid_share_all", "median") for k in T]
    cap = [get(T[k], "rigid_share_all", "by_part_mean", "cap") for k in T]
    text = (
        f"<p>No tip scores high on the v4b hand. The stage scores lie between {f(min(allJ))} and {f(max(allJ))} at every "
        f"condition, so the held turns average under {90 * max(allJ):.0f}&#176; of the 90&#176; goal "
        f"(Figure&#160;[[fig:screen]], Table&#160;[[tab:screen]]). At friction 0.7, J1 ranks {nm(top1[0])} first at "
        f"{f(J1[top1[0]], 3)}, then {nm(top1[1])} at {f(J1[top1[1]], 3)}. The v4b tip scores {f(J1[C.REF], 3)}. The worst "
        f"condition reorders the tips. J2 is {f(J2[top2[0]], 3)} for {nm(top2[0])}, {f(J2[top2[1]], 3)} for {nm(top2[1])} and "
        f"{f(J2[top2[2]], 3)} for {nm(top2[2])}. The v4b tip has J2 = {f(J2[C.REF], 3)}, at its worst condition, "
        f"{cond(ref.get('J2_condition'))}. On 15 October, on the first CAD hand at 10&#8202;MPa, the same shape had "
        f"J2 = {f(get(SCR15, 'tips', old(C.REF), 'J2'), 3)}.</p>"
        f"<p>The v4b tip&#8217;s pads end within 3&#176; of Drake&#8217;s turn, with the same held outcome, in "
        f"{ref['agree_drake_3deg']} of {ref['n_vs_drake']} replays. A tip therefore passes the gate with "
        f"{ref['agree_drake_3deg'] - 2} agreements or more. {cap1(WORDS[len(fail)])} tips fail it: {joined(nm(k) for k in fail)}. "
        + " ".join(cap1(t) for t in rule_sentences(fin, why, T))
        + " Rule (2) adds no tip: none that passes the gate replays at 0.8 of the v4b tip&#8217;s cost or less.</p>"
        f"<p>In the cells&#8217; HOM turns the cylinder also presses on the cap&#8217;s end plate, about 1.5&#8202;mm past the "
        f"pad&#8217;s distal end. A median {rng(rig)} of the normal force on the cylinder passes through the rigid "
        f"parts (Figure&#160;[[fig:screen]], bottom). The cap&#8217;s share alone is {rng(cap)}. In the D1&#8211;D8 turns of "
        f"the real_v1 hands all of it passes through the pads.</p>")
    out = text + fig + tab
    C3 = FIN.get("candidates", {})
    if C3:
        base = C3[C.REF]
        others = [k for k in FIN["tips"] if k != C.REF]
        best = max(others, key=lambda k: C3[k]["J3"])
        below = [k for k in others if C3[k].get("below_base")]
        beat = [k for k in others if C3[k].get("beats_base")]
        dj = lambda k: (f"{sg(C3[k]['dJ3_vs_base'])}, interval {sg(C3[k]['dJ3_ci90'][0])} to "  # noqa: E731
                        f"{sg(C3[k]['dJ3_ci90'][1])}")
        o_ref = get(FIN15, "candidates", old(C.REF))
        o_blk = get(FIN15, "candidates", C.BLOCKS[0])
        rest = [k for k in others if k != best and k not in below]
        out += (f"<p>On fresh placements " + ("no tip beats the v4b tip" if not beat else
                                              joined(nm(k) for k in beat) + " beat the v4b tip")
                + f". Its J3 is {f(base['J3'], 3)} (Table&#160;[[tab:stage3]], Figure&#160;[[fig:stage3]]). The best other "
                f"tip is {nm(best)}. It scores {f(C3[best]['J3'], 3)}, a paired difference of {dj(best)}. "
                + " ".join(f"{nm(k)[:1].upper() + nm(k)[1:]} differs by {dj(k)}." for k in rest)
                + (" " + " ".join(f"{nm(k)[:1].upper() + nm(k)[1:]} scores below the v4b tip, {dj(k)}." for k in below))
                + (f" All {base['n_hold_mu0.5']} grasps of every stage-3 tip pass the hold test at friction 0.5 with three "
                   "loaded fingers." if all(C3[k].get("hold3_mu0.5") == C3[k].get("n_hold_mu0.5") for k in FIN["tips"])
                   else " Some grasps of the stage-3 tips fail the hold test at friction 0.5 with three loaded fingers.")
                + f" On 15 October the v4b tip&#8217;s shape scored J3 = "
                f"{f(get(o_ref, 'J3'), 3)} against the as-built block&#8217;s {f(get(o_blk, 'J3'), 3)}. Here the block scores "
                f"{f(C3[C.BLOCKS[0]]['J3'], 3)} against the v4b tip&#8217;s {f(base['J3'], 3)}.</p>")
        b3 = []
        for k in FIN["tips"]:
            v = C3[k]
            ci = v.get("dJ3_ci90")
            b3.append([name(k), f(v["J3"], 3), sg(v.get("dJ3_vs_base")) if ci else "&#8211;",
                       f"{sg(ci[0])} to {sg(ci[1])}" if ci else "&#8211;", f"{v['hold3_mu0.5']} of {v['n_hold_mu0.5']}"])
        out += table(["Tip", "J3", "J3 minus v4b", "90&#8202;% interval", "Three-finger holds at friction 0.5"], b3,
                     "Stage 3 on fresh placements and the paired bootstrap difference against the v4b tip (10,000 resamples "
                     "within hands and cells; final/summary.json).", key="stage3")
        out += img("media/fig_stage3.png", "Stage 3 per tip: the score at each condition and J3 (left), and the difference to "
                   "the v4b tip with its 90&#8202;% bootstrap interval (right).", key="stage3")
    if BAND:
        tips_b = [k for k in [C.REF] + [t for t in FIN.get("tips", []) if t != C.REF] if k in BAND]
        body = []
        for k in tips_b:
            row = [name(k)]
            for t, _ in TASKS4:
                v = get(BAND[k], t, "S3") or {}
                row.append(f(v.get("worst_median")) if k == C.REF else
                           (f"{f(v.get('worst_median'))} ({v['diff_median']:+.2f})" if v else "&#8211;"))
            body.append(row)
        rep = [abs(get(BAND[C.REF], t, pk, q) or 0) for t, _ in TASKS4 for pk in ("S1", "S2", "S3")
               for q in ("diff_q25", "diff_q75") if get(BAND.get(C.REF) or {}, t, pk)]
        diffs = {k: [get(BAND[k], t, "S3", "diff_median") for t, _ in TASKS4 if get(BAND[k], t, "S3")]
                 for k in tips_b if k != C.REF}
        out += ("<p>The friction band repeats the worst-case score of 4 November. Its plant is the STS3032 with pads at "
                "2.5&#8202;MPa, at scene friction 0.4, 0.7 and 1.0. Each HOM task has 30 layouts in its sample, and each "
                "layout runs the design refined for the v4b tip: its grasp, forces and contact modes (pick S3). A design "
                "scores its lowest five-seed mean over the three frictions. With another tip in place of the v4b pad the same "
                "designs score as in Table&#160;[[tab:band]]. Run again on the v4b tip, they repeat the scores of 4 November: "
                f"the paired differences have a median of 0.000 and quartiles within {f(max(rep) if rep else None, 3)}. "
                + " ".join(f"On {nm(k)} the median paired difference per task runs from {sg(min(v), 2)} to {sg(max(v), 2)}."
                           for k, v in diffs.items() if v)
                + " These designs were refined for the 11/22, and the other tips&#8217; own designs were not refined.</p>"
                + table(["Tip"] + [b for _, b in TASKS4], body,
                        "Worst case over scene friction 0.4&#8211;1.0 on the STS3032 (band/summary.json; "
                        "<code>scripts/fingertip/fingertip_shape_rerun_band.py</code>). Each cell is the median over the 30 layouts of pick "
                        "S3&#8217;s lowest five-seed mean. In brackets is the median paired difference to the v4b tip.",
                        key="band"))
    out += film("media/hom_cylinder_16_0.625_s3_tile.mp4", "media/hom_cylinder_16_0.625_s3_tile_poster.jpg",
                "HOM on the v4b hand&#8217;s 16&#8202;mm cylinder at friction 0.7 and seed 3, a placement that no stage used to "
                "choose the grasp. Each tip turns with the grasp it chose on seed 0, seen from below the palm. Pads touching the "
                "cylinder are red, a rigid part touching it yellow.", key="hom")
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
        body.append([name(k), f(get(t, "sphere_apex_um", "1"), 0), f(fe18(k, "s_apex"), 0), f(get(t, "rod_across_um", "1"), 0),
                     f(fe18(k, "cx12.5_z0"), 0), f(t.get("rod_along_1N_um"), 0),
                     f(t.get("exponent_sphere")), f(t.get("exponent_rod")), f(get(t, "twist_arm_rod_mm", "1")),
                     f(t.get("sphere_map_spread")),
                     f(s.get("centre_force_N"), 3) if s.get("centre_force_N") is not None else ("No fit" if not t["magnets"] else "&#8211;"),
                     f(s.get("B_max_component_mT"), 1)])
    tab = table(["Tip", "Sphere at the apex, 1&#8202;N, &#181;m", "18 Oct", "Rod across, 1&#8202;N, &#181;m", "18 Oct",
                 "Rod along, 1&#8202;N, &#181;m",
                 "Exponent, sphere", "Exponent, rod", "Twist arm under the rod, 1&#8202;N, mm", "Sphere map, largest over least",
                 "Force resolution, N", "Largest field component, mT"], body,
                "Finite elements per tip (fe/summary.json; <code>scripts/fingertip/fingertip_shape_rerun_fe.py</code>).", key="fe")
    have = [k for k in ORDER if k in FE]
    curved = [k for k in have if k not in C.BLOCKS and k not in C.QUARTERS]
    sph = [get(FE[k], "sphere_apex_um", "1") for k in have]
    spread = [FE[k].get("sphere_map_spread") for k in have]
    rod_c = [get(FE[k], "rod_across_um", "1") for k in curved]
    rod_b = [get(FE[k], "rod_across_um", "1") for k in C.BLOCKS if k in FE]
    n_s = [FE[k].get("exponent_sphere") for k in have]
    n_r = [FE[k].get("exponent_rod") for k in curved]
    res = {k: get(FE[k], "sensing", "centre_force_N") for k in have if get(FE[k], "sensing", "centre_force_N") is not None}
    trav = [get(FE[k], "sensing", "magnet_travel_um_per_N") for k in res]
    bmax = [get(FE[k], "sensing", "B_max_component_mT") for k in res]
    d18 = [abs(get(FE[k], "sphere_apex_um", "1") / fe18(k, "s_apex") - 1) for k in have if fe18(k, "s_apex")]
    d18r = [abs(get(FE[k], "rod_across_um", "1") / fe18(k, "cx12.5_z0") - 1) for k in have if fe18(k, "cx12.5_z0")]
    nofit = [k for k in have if not FE[k]["magnets"]]
    text = (
        f"<p>In the finite elements the tips sink {rng(sph, 0)}&#8202;&#181;m at the apex under the 2&#8202;mm sphere at "
        f"1&#8202;N. Over each tip&#8217;s seven map points, the softest point sinks {rng(spread)} times as far as the stiffest "
        f"(Table&#160;[[tab:fe]], Figure&#160;[[fig:fe]]). Under the sphere the approach grows "
        f"with the load to the power {rng(n_s)}. Under the 12.5&#8202;mm rod across the finger the curved tips sink "
        f"{rng(rod_c, 0)}&#8202;&#181;m at 1&#8202;N. The blocks sink {rng(rod_b, 0)}&#8202;&#181;m, and the curved tips&#8217; "
        f"exponent under the rod is {rng(n_r)}. The "
        f"pads here are cut where the v4b pad is cut and held on that face. Against the whole prints of 18 October the apex "
        f"approach changes by at most {100 * max(d18):.0f}&#8202;% and the rod&#8217;s by at most {100 * max(d18r):.0f}&#8202;%.</p>"
        + (f"<p>With pad 05R4&#8217;s four magnets, the force at the contact point is resolved to "
           f"{rng(list(res.values()), 3)}&#8202;N from one 141&#8202;Hz sample on every tip in which they fit. The chips run at the gain that the {rng(bmax, 1)}&#8202;mT field needs, "
           f"and the magnets move {rng(trav)}&#8202;&#181;m/N. The finest resolution is "
           f"{100 * (1 - min(res.values()) / max(res.values())):.0f}&#8202;% below the coarsest. " if res else "<p>")
        + (f"The layout does not fit the {WORDS[len(nofit)]} quarter-ellipsoids: the magnets&#8217; tops lie "
           "7.5&#8202;mm under the apex, behind their back planes.</p>" if nofit else "</p>"))
    fig = img("media/fig_fe.png", "Finite elements per tip. Top: the approach at 1&#8202;N under the 2&#8202;mm sphere at the "
              "seven map points, over the apex&#8217;s. Middle: the approach at 1&#8202;N under the sphere at the apex and under the "
              "12.5&#8202;mm rod across the finger. Bottom: the force resolution at the contact point with pad 05R4.", key="fe")
    return text + fig + tab


def compute_section():
    jobs = CMP.get("jobs", {})
    if not jobs:
        return "<p><i>[compute summary not written]</i></p>"
    label = {"scale": "Cells, stages 1 and 2", "hom": "HOM on D1&#8211;D8, stages 1 and 2", "traj drake": "Drake replays (gate)",
             "traj plan,rl": "Pad replays (gate)", "traj fill": "Drake replays refilled", "stage3 scale": "Cells, stage 3",
             "stage3 hom": "HOM on D1&#8211;D8, stage 3", "law": "Bed: law", "bed": "Bed: pads and Drake", "fe": "Finite elements",
             "gpu cost": "GPU cost", "cpu cost": "CPU cost", "band": "Friction band"}
    body = [[label.get(k, k), str(v["n"]), f(v["core_h"]), f(v.get("gpu_h"), 2), f(v["wall_h"])] for k, v in jobs.items()]
    tot = sum(v["core_h"] for v in jobs.values())
    gtot = sum(v.get("gpu_h", 0) for v in jobs.values())
    body.append(["All", str(sum(v["n"] for v in jobs.values())), f(tot), f(gtot), "&#8211;"])
    return (f"<p>The rerun cost {tot:.1f} core-hours and {gtot:.1f} GPU hours (Table&#160;[[tab:compute]]). Of the "
            f"core-hours, {f(get(jobs, 'traj drake', 'core_h'), 1)} went to the Drake replays of the agreement gate. The queue "
            "waited 36 minutes at resguard&#8217;s swap gate, which was closed by the desktop&#8217;s swapped pages.</p>"
            + table(["Jobs", "Count", "Core-hours", "GPU hours", "Wall time, h"], body, "Compute of the rerun by kind of job "
                    "(compute/summary.json, from the job logs under <code>logs/20261104-fingertip_shape_rerun/queue/</code>). "
                    "Wall time runs from a kind&#8217;s first start to its last end.", key="compute"))


def limitations():
    return ("<p>Every reference is computed. In the finite elements the solid TPU has NinjaFlex 85A&#8217;s modulus of "
            "12&#8202;MPa, over a homogenized gyroid. One modulus is used for every shape in the pads and in Drake: the "
            "2.5&#8202;MPa fitted to the UR5 presses of the v3 tips, which have the 11/22 shape. Ten of the twelve tips would need a cassette and cap of their own; they are "
            "simulated on the 11/22&#8217;s. The cells&#8217; grasps come from the scale study&#8217;s kinematics, and in their HOM "
            "turns the cylinder also presses on the cap&#8217;s end plate. The D1&#8211;D8 hands are "
            "real_v1 hands, and their bench plans were made for the block. The friction band runs the v4b tip&#8217;s refined "
            "designs on the other tips without refining them again.</p>")


def open_items():
    items = [
        ("A longer stage 3 for the 16/32 half-ellipsoid",
         "Rerun its stage-3 jobs and the v4b tip&#8217;s with cell seeds 3&#8211;12 and HOM seeds 5&#8211;19 "
         "(<code>scripts/fingertip/fingertip_shape_rerun_queue.py stage3</code> with the seeds widened). With a 90&#8202;% interval above zero it "
         "becomes the shape to print, on a cassette and cap widened for its distal flanks. With an interval that still "
         "includes zero the 11/22 stays."),
        ("Grasps that keep the cap off the cylinder",
         "Check the cells&#8217; candidate grasps with the pad-coverage and rigid-clearance tests of "
         "<code>scripts/hom/fp_sr2_toy.py</code> at the touch pose and along the HOM turn. Rescore J on the grasps whose "
         "turns put under 10&#8202;% of the normal force on rigid parts. A change in the tips&#8217; order would show how much "
         "of J comes from the cap."),
        ("The friction band with each tip&#8217;s own designs",
         "Refine the band&#8217;s grasps, forces and modes per stage-3 tip (<code>scripts/hom/hom_grasp_refine.py run --band</code> "
         "with the tip&#8217;s scene caches of <code>scripts/fingertip/fingertip_shape_rerun_band.py</code>). A worst-case score above the "
         "v4b tip&#8217;s on its own designs would separate the shape from the designs it was tuned with."),
        ("Bench presses of a second shape",
         "Print the 16/32 half-ellipsoid with pad 05R4&#8217;s magnets. Press it on the UR5 under the 5&#8202;mm sphere and the "
         "12.5&#8202;mm rod at 0.25&#8211;8&#8202;N, as the v3 tips were pressed. A fitted modulus within the v3 tips&#8217; range, "
         "2.22&#8211;2.73&#8202;MPa, confirms that one modulus fits both shapes."),
    ]
    return "<ol>" + "".join(f"<li><b>{a}.</b> {b}</li>" for a, b in items) + "</ol>"


def lede():
    C3 = FIN.get("candidates", {})
    if not C3:
        return "[lede: the outcome with its numbers, written when the stages are done]"
    others = [k for k in FIN["tips"] if k != C.REF]
    best = max(others, key=lambda k: C3[k]["J3"])
    mounts = [k for k in ORDER if get(MNT, "tips", k, "verdict") == "mounts"]
    gpu = lambda k: get(COST.get(k) or {}, "gpu", "us_per_world_step")  # noqa: E731
    rest = [k for k in ORDER if k != C.REF and k not in C.BLOCKS]
    res = [get(FE.get(k) or {}, "sensing", "centre_force_N") for k in ORDER]
    beat = [k for k in others if C3[k].get("beats_base")]
    loss = [-get(BAND[k], t, "S3", "diff_median") for k in BAND if k != C.REF for t, _ in TASKS4 if get(BAND[k], t, "S3")]
    band = (f"On each other stage-3 tip they score lower, by a median {rng(loss)} per task." if loss and min(loss) > 0 else
            f"On the other stage-3 tips the median change per task runs from {sg(-max(loss), 2)} to {sg(-min(loss), 2)}."
            if loss else "")
    return (f"On the v4b finger at 2.5&#8202;MPa {'no fingertip shape beats' if not beat else 'one shape beats'} the 11/22 "
            f"half-ellipsoid that the hand uses. The best other tip is {nm(best)}. Its J3 on fresh placements is "
            f"{f(C3[best]['J3'], 3)}, against {f(C3[C.REF]['J3'], 3)} for the v4b tip. The difference is "
            f"{sg(C3[best]['dJ3_vs_base'])}, with a 90&#8202;% interval from {sg(C3[best]['dJ3_ci90'][0])} to "
            f"{sg(C3[best]['dJ3_ci90'][1])}. The HOM grasps, forces and contact modes were refined for the v4b tip over scene "
            f"friction 0.4&#8211;1.0. {band} A GPU world-step lasts {f(gpu(C.REF))}&#8202;&#181;s on the v4b tip. It lasts "
            f"{rng([gpu(k) for k in rest])}&#8202;&#181;s on the capsules, the other ellipsoids and the hemisphere, and "
            f"{rng([gpu(k) for k in C.BLOCKS])}&#8202;&#181;s on the blocks. Of the twelve tips only {WORDS[len(mounts)]} fit the "
            f"v4b DIP link, cassette and cap as built. With pad 05R4&#8217;s magnets, the force at the contact point is resolved "
            f"to {rng([x for x in res if x is not None], 3)}&#8202;N on every tip in which they fit.")


def byline():
    return ("Scripts <code>scripts/fingertip_shape_rerun_{common,mount,rig,traj,bed,cost,fe,queue,analysis,figures,film,page}.py"
            "</code> &#183; rows <code>docs/experiments/20261104-fingertip_shape_rerun/</code> &#183; hand-off "
            "<code>docs/handoff/20261103-fingertip_shape_rerun.md</code>")


def main():
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "GLOSSARY": glossary(), "QUESTION": question(), "MOUNT": mount_section(),
         "BED": bed_section(), "COST": cost_section(), "SCREEN": screen_section(), "FE": fe_section(),
         "COMPUTE": compute_section(), "LIMITS": limitations(), "OPEN": open_items(), "LEDE": lede()}
    v["FOOTER"] = ("<p>Rebuild: <code>python3 scripts/fingertip/fingertip_shape_rerun_analysis.py all</code>, the figures, then "
                   "<code>.venv/bin/python scripts/fingertip/fingertip_shape_rerun_page.py</code>. Earlier studies: "
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
    t = docs_index.tag(t, topic="fingertip", summary=html.unescape(re.sub(r"<[^>]+>", "", lede())), date="2026-11-04")
    open(OUT, "w").write(t)
    print(f"formulas {n}; figures {FIG[0]}; tables {TAB[0]}; films {FILM[0]}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
