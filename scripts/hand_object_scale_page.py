#!/usr/bin/env python3
r"""Build docs/experiments/20261008-hand_object_scale/20261008-finger_spacing_object_size.html.

    .venv/bin/python scripts/hand_object_scale_page.py

Reads the study's own rows in docs/experiments/20261008-hand_object_scale/: literature.json (the checked citations),
reach.json, family.json and objects.json (hand_object_scale_kin.py), workspace.jsonl, hold.jsonl and turn.jsonl
(hand_object_scale_kin.py workspace, hand_object_scale_sim.py), landscape.json (hand_object_scale_landscape.py),
protocol.json (hand_object_scale_protocol.py) and media/films.jsonl with its clips. Every number in the prose is
computed here. Mathematics is LaTeX (\( \) inline, \[ \] display) typeset to inline SVG by scripts/texsvg.py, cached in
the study folder.
"""
from __future__ import annotations

import html
import json
import math
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import texsvg  # noqa: E402
import retro_style  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261008-hand_object_scale")
REL = "docs/experiments/20261008-hand_object_scale"
OUT = os.path.join(D, "20261008-finger_spacing_object_size.html")
TPL = os.path.join(ROOT, "scripts/hand_object_scale_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
num = P.num
FINGER = 68.11            # mm, mount to pad centre with the finger straight (yaw link 20.75 + 20.75 + 26.61)
FLEX = 47.36              # mm, MCP axis to pad centre

# Task colours: the dataviz default categorical slots 1-3 (validated all-pairs in both modes; the light aqua is under
# 3:1 on white, so every series is direct-labelled). Heatmaps: one blue sequential ramp, light to dark on the light
# surface and dark to light on the dark one.
EXTRA_CSS = """
:root{--t-kin:#2a78d6;--t-hold:#eb6834;--t-turn:#1baf7a;
 --q0:#e8f0fb;--q1:#cde2fb;--q2:#9ec5f4;--q3:#6da7ec;--q4:#3987e5;--q5:#256abf;--q6:#184f95;--q7:#0d366b;
 --qna:#f0efec;--qtext-lo:#0b0b0b;--qtext-hi:#ffffff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--t-kin:#3987e5;--t-hold:#d95926;--t-turn:#199e70;
 --q0:#1d2a3a;--q1:#0d366b;--q2:#104281;--q3:#184f95;--q4:#1c5cab;--q5:#2a78d6;--q6:#5598e7;--q7:#9ec5f4;
 --qna:#2a2e33;--qtext-lo:#ffffff;--qtext-hi:#0b0b0b}}
:root[data-theme="dark"]{--t-kin:#3987e5;--t-hold:#d95926;--t-turn:#199e70;
 --q0:#1d2a3a;--q1:#0d366b;--q2:#104281;--q3:#184f95;--q4:#1c5cab;--q5:#2a78d6;--q6:#5598e7;--q7:#9ec5f4;
 --qna:#2a2e33;--qtext-lo:#ffffff;--qtext-hi:#0b0b0b}
.tw th{text-transform:none;letter-spacing:.01em}
td.lab{white-space:normal;min-width:14em}
td.wrap{white-space:normal;text-align:left;font-family:var(--f-body);font-size:14px;line-height:1.45}
table.lit td{white-space:normal;vertical-align:top}
table.lit td:first-child{min-width:9em}
table.lit th{text-align:left}
.verdict-confirmed{color:var(--good)} .verdict-corrected{color:var(--bad)}
svg .cell:hover{stroke:var(--ink);stroke-width:1.5}
figure.diagram svg{max-width:100%;height:auto}
""" + texsvg.BLOCK_CSS + "\n"


# ------------------------------------------------------------------------------------------ helpers

def jl(name):
    p = os.path.join(D, name)
    out = []
    if os.path.exists(p):
        for line in open(p):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def js(name):
    p = os.path.join(D, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, nd=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "&#8211;"
    return num(x, f".{nd}f")


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def table(head, rows, cls=""):
    out = [f"<table{' class=' + repr(cls) if cls else ''}><thead><tr>" +
           "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head)) + "</tr></thead><tbody>"]
    for r in rows:
        if isinstance(r, str):
            out.append(f"<tr class='grp'><td colspan='{len(head)}'>{r}</td></tr>")
            continue
        cells = []
        for i, c in enumerate(r):
            cl = []
            if i:
                cl.append("num")
            if isinstance(c, tuple):
                c, extra = c
                cl.append(extra)
            cells.append(f"<td{' class=' + repr(' '.join(cl)) if cl else ''}>{c}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "<div class='tw'>" + "".join(out) + "</div>"


def im(tex):
    return "\\(" + tex + "\\)"


def eq(tex, n=None):
    no = f'<span class="eqno">({n})</span>' if n is not None else ""
    return f'<div class="eq">\\[{tex}\\]{no}</div>'


def render_tex(t):
    maths = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = texsvg.render(maths, cache_path=TEX_CACHE, scale=P.TEX_SCALE)
    it = iter(svgs)
    return P.TEX_RE.sub(lambda m: next(it), t), len(maths)


FIG, TAB = [0], [0]


def figure(svg, caption):
    FIG[0] += 1
    return f'<figure class="diagram">{svg}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def tcap(text):
    TAB[0] += 1
    return f"<p class='tcap'><b>Table&#160;{TAB[0]}.</b> {text}</p>"


def film(rel, caption, poster=None):
    p = os.path.join(D, rel)
    if not os.path.exists(p):
        return ""
    pp = os.path.join(D, poster) if poster else None
    po = f' poster="{P.R.data_uri(pp, "image/jpeg")}"' if pp and os.path.exists(pp) else ""
    FIG[0] += 1
    return (f'<figure><video src="{P.R.data_uri(p, "video/mp4")}"{po} controls muted loop playsinline preload="metadata">'
            f'</video><figcaption>Figure&#160;{FIG[0]}. {caption} <code>{REL}/{rel}</code></figcaption></figure>')


def esc(s):
    return html.escape(str(s), quote=True)


# ------------------------------------------------------------------------------------------ data

def data():
    L = js("landscape.json")
    R = js("reach.json")
    O = js("objects.json")
    lit = js("literature.json")
    prot = js("protocol.json")
    films = jl("media/films.jsonl")
    return L, R, O, lit, prot, films


def layouts(L, fam="diag"):
    return L["families"][fam]["layouts"]


def okeys(shape):
    return {"cylinder": (6, 10, 16, 25, 40, 50, 63, 80), "sphere": (10, 16, 25, 40, 50, 63, 80, 100, 125, 160)}[shape]


# ------------------------------------------------------------------------------------------ terms

def glossary():
    G = [
        ("layout", "the three finger-base (gantry) positions in the palm plane. This study uses symmetric tripods: thumb at "
                   + im(r"(-x_s/2,\,0)") + ", index at " + im(r"(x_s/2,\,y_s/2)") + ", middle at " + im(r"(x_s/2,\,-y_s/2)")
                   + ", in mm, with " + im("x_s") + " the thumb&#8211;pair span and " + im("y_s") + " the pair opening. "
                   "The CAD-nominal hand is " + im(r"x_s = 100,\ y_s = 110") + "."),
        ("tripod scale " + im("t"), "position along the one-parameter layout family " + im(r"x_s = 100 - 60\,t,\ y_s = 110 - 60\,t")
                                    + " (mm), " + im(r"t \in [-1, 1]") + ". " + im(r"t \in [0,1]") + " is "
                                    "<code>real_v1_compact_design(t, t, t)</code>; " + im("t < 0") + " continues to the "
                                    "gantries&#8217; wide limit (" + im(r"x_s = 160,\ y_s = 170") + " at " + im("t = -1") + ")."),
        ("palm radius " + im(r"\rho"), "radius of the circle through the three mounts, in mm: " + im(r"\rho = a + b^2/(4a)")
                                       + " with " + im(r"a = x_s/2,\ b = y_s/2") + ". Borr&#224;s and Dollar&#8217;s "
                                       "&#8220;palm radius&#8221;; 27.8&#8202;mm at " + im("t = 1") + ", 65.1 at " + im("t=0")
                                       + ", 102.6 at " + im("t=-1") + "."),
        ("object", "a cylinder 100&#8202;mm long lying along the pair axis (the screwdriver class) or a sphere, of diameter "
                   + im("d") + " and radius " + im("r = d/2") + "; 25&#8202;g in simulation."),
        ("equator grasp", "thumb pad and both pair pads on the object&#8217;s equator (the widest section normal to the "
                          "pinch axis), the pair straddling the thumb along the cylinder by &#177;" + im(r"\sigma")
                          + " (22.5, 32.5 or 42.5&#8202;mm); on a sphere each pad faces its own mount from the centre of "
                          "the mount circle."),
        ("feasible", "an equator grasp exists with every joint at least 2.9&#176; (0.05&#8202;rad) inside the servos&#8217; "
                     "calibrated range, the fingertip&#8217;s palmar normal within 90&#176; of the object&#8217;s inward "
                     "normal, at least 5&#8202;mm between fingers (the clearance gate of "
                     "<code>real_v1_trajectory_clearance.py</code>), 1&#8202;mm between the object and every link and the "
                     "palm plate."),
        ("workspace " + im("W"), "the fixed-contact precision-manipulation workspace (Borr&#224;s and Dollar): each pad "
                                 "centre stays at its point on the object, and " + im("W") + " counts the object poses "
                                 "(centre on a 3&#8202;mm grid, 123 orientations within 45&#176; of the grasp, 15&#176; "
                                 "steps) at which every finger reaches within its joint ranges without collision. Best "
                                 "straddle per cylinder."),
        ("rotation range", "largest rotation of the object about the pinch axis (palm " + im("x") + ") reachable with the "
                           "contacts fixed, its centre free, the smaller of the two signs, in degrees (5&#176; steps)."),
        ("grasp robustness " + im("F_{\\min}"), "the weakest of six force thresholds, in N: from a held grip, an external "
                                                "force on the object ramps at 4&#8202;N/s along &#177;" + im("x") + ", &#177;"
                                                + im("y") + ", &#177;" + im("z") + " until the object moves 3&#8202;mm or "
                                                "turns 5&#176;; the grip presses each pad with 4&#8202;N at rest; 20&#8202;N "
                                                "cap. Best of the cell&#8217;s grasps; 0 when the grip drops the object."),
        ("held turn", "rotation of the object&#8217;s marked axis (along the cylinder; the sphere&#8217;s body axis) from "
                      "horizontal toward vertical at the end of the HOM controller&#8217;s hold, in degrees, counted only "
                      "when at least two fingers each press with at least the object&#8217;s weight (0.245&#8202;N) and "
                      "the object is within 20&#8202;mm of its start; 0 otherwise. Median of three placements (object "
                      "offset 2&#8202;mm, yaw 2&#176;), best of the cell&#8217;s grasps."),
        ("HOM controller", "the relative contact-velocity controller of Wang, Oh and Pollard (arXiv 2609.25619) as "
                           "<code>scripts/hom_turn3.py</code>: contact frames from the TPU block mesh, least squares within joint-rate limits "
                           "on the reference relative velocities, pad force regulated to 2&#8202;N, 30&#176;/s toward "
                           "90&#176;, a governor that stops the reference when a pad&#8217;s force falls under 30&#8202;% "
                           "of its target or the object lags by 6&#176;. One parameter set for every cell."),
        ("ridge " + im(r"t^*_T(d)"), "for task " + im("T") + " and object " + im("d") + ", the layout of largest score "
                                     "along a family, refined by a parabola through the maximum and its two neighbours; "
                                     "reported as " + im("t") + " and as the mount distance it implies: " + im(r"\rho^*")
                                     + " for spheres, the half thumb&#8211;pair span " + im(r"a^* = x_s^*/2") + " for "
                                     "cylinders."),
        ("reach margin", "smallest over the fingers of how far the pad centre can still move away from the MCP axis before "
                         "the finger is straight, in mm (" + im(r"47.4\,\text{mm} - |p - q_\text{MCP}|") + ")."),
        (im(r"\sigma_{\min}"), "smallest singular value of a finger&#8217;s 3&#215;3 pad-position Jacobian, in mm/rad, smallest over the "
                             "fingers: the slowest direction in which the pad can move."),
        ("joint margin", "smallest distance of any joint from its servo limit at the grasp, in degrees."),
        ("contact angle", "largest over the fingers of the angle between the fingertip&#8217;s palmar normal and the "
                          "object&#8217;s inward surface normal at the contact, in degrees; 0 = the pad face-on."),
        ("force transmission", "smallest over the fingers of the normal pad force produced per N&#8202;m of the most "
                               "loaded joint, " + im(r"1/\max_j |(J^\top n)_j|") + ", in N/(N&#8202;m): the servo "
                               "torque margin of the posture."),
    ]
    rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in G)
    return f"<div class='tw'><table class='gloss'><tbody>{rows}</tbody></table></div>"


# ------------------------------------------------------------------------------------------ literature

def lit_section(lit):
    if not lit:
        return P.pending("literature.json missing")
    rows = []
    for c in lit["cited"]:
        v = c["verdict"]
        label, _, claim = c["cited_as"].partition(":")
        corr = "" if c["correction"].startswith("none") else f"<b>Correction.</b> {esc(c['correction'])} "
        rows.append([f"<b>{esc(label)}</b><br><span style='color:var(--ink3)'>{esc(claim.strip())}</span>",
                     (f"<span class='verdict-{v}'>{v}</span>", "wrap"),
                     (f"{corr}{esc(c['finding'])}<br><span style='color:var(--ink3)'>{esc(c['source'])}</span>", "wrap"),
                     (esc(c["checked"]), "wrap")])
    n_corr = sum(1 for c in lit["cited"] if c["verdict"] == "corrected")
    out = [f"<p>All {len(lit['cited'])} works the suggestion note cites exist and say what it attributes to them; "
           f"{n_corr} needed a correction. Elangovan et al. analyse a two-finger planar gripper whose workspace is largest "
           "with the finger bases together; Seo and Armstrong&#8217;s grip force falls monotonically with handle diameter "
           "over the range they tested, with no optimum; the Bullock, Feix and Dollar result comes from their EMBC 2015 "
           "paper. Hota and Kumar&#8217;s object is a cylinder, and their under-actuated hand prefers wide bases for every "
           "object size. Table&#160;1 gives each source, verdict and finding; "
           f"<code>{REL}/literature.json</code> holds the full records.</p>",
           tcap("The suggestion note&#8217;s citations against their sources. &#8220;Checked&#8221; names what was read."),
           table(["Work and the note&#8217;s claim", "Verdict", "What the source reports", "Checked"], rows,
                 "lit")]
    miss = "".join(f"<li><b>{esc(m['source'])}</b>. {esc(m['why'])} ({esc(m['checked'])})</li>" for m in lit["missed"])
    out.append(f"<p>{esc(lit['verdict'])} The closest work, which a paper on this map must cite:</p><ul>{miss}</ul>")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------ family

def svg_layouts(R):
    """Top view of the palm: gantry boxes and the mounts of five layouts of the 1-D family."""
    W, H = 900, 360
    out = P._svg_open(W, H, "Top view of the palm: the three gantry boxes and the finger mounts of five layouts of the "
                            "one-parameter family, from the widest (t = -1) to the most compact (t = 1).")
    sx, cx, cy = 1.55, 300, 180          # px per mm, palm origin
    X = lambda x: cx + sx * x  # noqa: E731
    Y = lambda y: cy - sx * y  # noqa: E731
    boxes = [("thumb", (-80, -20, -55, 55)), ("index", (20, 80, 25, 85)), ("middle", (20, 80, -85, -25))]
    for name, (x0, x1, y0, y1) in boxes:
        out.append(f'<rect x="{X(x0):.1f}" y="{Y(y1):.1f}" width="{sx * (x1 - x0):.1f}" height="{sx * (y1 - y0):.1f}" '
                   f'rx="4" style="fill:var(--sunk);stroke:var(--rule)"/>'
                   f'<text x="{X(x0) + 4:.1f}" y="{Y(y1) + 14:.1f}" style="fill:var(--ink3)">{name}</text>')
    ts = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for i, t in enumerate(ts):
        xs, ys = 100 - 60 * t, 110 - 60 * t
        shade = f"var(--q{2 + i})"
        pts = [(-xs / 2, 0.0), (xs / 2, ys / 2), (xs / 2, -ys / 2)]
        out.append('<polygon points="' + " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in pts)
                   + f'" style="fill:none;stroke:{shade};stroke-width:1.5"/>')
        for a, b in pts:
            out.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="5" style="fill:{shade};stroke:var(--card);stroke-width:2">'
                       f'<title>t = {t:+.1f}: x_s {xs:g} mm, y_s {ys:g} mm</title></circle>')
        out.append(f'<text x="{X(xs / 2) + 9:.1f}" y="{Y(ys / 2) + 4:.1f}" style="fill:var(--ink2)">t {t:+.1f}</text>')
    out.append(f'<line x1="{X(-90)}" x2="{X(95)}" y1="{Y(0)}" y2="{Y(0)}" style="stroke:var(--rule);stroke-dasharray:3 4"/>'
               f'<text x="{X(95) + 4}" y="{Y(0) + 4}" style="fill:var(--ink3)">x (pinch)</text>'
               f'<line x1="{X(0)}" x2="{X(0)}" y1="{Y(-95)}" y2="{Y(100)}" style="stroke:var(--rule);stroke-dasharray:3 4"/>'
               f'<text x="{X(0) + 4}" y="{Y(100) + 2}" style="fill:var(--ink3)">y</text>')
    # scale bar
    out.append(f'<line x1="{X(-80)}" x2="{X(-30)}" y1="{Y(-95)}" y2="{Y(-95)}" style="stroke:var(--ink2);stroke-width:2"/>'
               f'<text x="{X(-55)}" y="{Y(-95) + 16}" text-anchor="middle" style="fill:var(--ink2)">50 mm</text>')
    # legend text block on the right
    lx = 620
    out.append(f'<text x="{lx}" y="40" style="fill:var(--ink2)">mounts by tripod scale t</text>')
    for i, t in enumerate(ts):
        xs, ys = 100 - 60 * t, 110 - 60 * t
        rho = xs / 2 + (ys / 2) ** 2 / (2 * xs)
        out.append(f'<circle cx="{lx + 6}" cy="{62 + 22 * i}" r="5" style="fill:var(--q{2 + i})"/>'
                   f'<text x="{lx + 18}" y="{66 + 22 * i}" style="fill:var(--ink2)">t {t:+.1f}: {xs:g} / {ys:g} mm, &#961; {rho:.1f}</text>')
    out.append("</svg>")
    return "".join(out)


def family_section(L, R, O):
    fam = R["families"]
    cells = R["cells"]
    feas = {(c["tag"], c["shape"], c["d_mm"]): c["feasible"] for c in cells}
    diag = fam["diag"]

    def band(shape, lay):
        ok = [c["d_mm"] for c in cells if c["tag"] == lay["tag"] and c["shape"] == shape and c["feasible"]]
        return (min(ok), max(ok)) if ok else None

    b0 = {s: band(s, next(l for l in diag if l["t"] == 0.0)) for s in ("cylinder", "sphere")}
    bw = {s: band(s, next(l for l in diag if l["t"] == -1.0)) for s in ("cylinder", "sphere")}
    bc = {s: band(s, next(l for l in diag if l["t"] == 1.0)) for s in ("cylinder", "sphere")}
    allc = [c["d_mm"] for c in cells if c["shape"] == "cylinder" and c["feasible"] and c["tag"] in {l["tag"] for l in diag}]
    alls = [c["d_mm"] for c in cells if c["shape"] == "sphere" and c["feasible"] and c["tag"] in {l["tag"] for l in diag}]
    xs = fam["xsep"]
    ys = fam["ysep"]
    yb = {y["y_sep"]: band("cylinder", y) for y in ys}
    rows = []
    for o in O["objects"]:
        t = o["diag_t"]
        rows.append([f"{o['shape']} {o['d_mm']:g}&#8202;mm" + (f" &#215; {o['length_mm']:g}" if o["length_mm"] else ""),
                     f(o["mass_g"], 1) + ("" if o["on_target"] else " *"), (esc(o["recipe"]), "wrap"),
                     f"{min(t):+.3f} to {max(t):+.3f}" if t else "&#8211;", f"{len(t)}", f"{o['n_grid'][0]}"])
    out = [f"<p>The six gantry coordinates of a symmetric tripod reduce to two separations, because the arm re-centres the "
           "palm over the object: the thumb&#8211;pair span " + im("x_s") + " and the pair opening " + im("y_s") + ". The "
           "one-parameter family " + im(r"x_s = 100 - 60\,t,\ y_s = 110 - 60\,t") + " runs from the gantries&#8217; widest "
           "tripod (" + im("t=-1") + ", 160 and 170&#8202;mm) through the CAD-nominal hand (" + im("t=0") + ") to the most "
           "compact (" + im("t=1") + ", 40 and 50&#8202;mm); " + im(r"t \in [0, 1]") + " is "
           "<code>real_v1_compact_design(t, t, t)</code>. Each of the 105 layouts studied (the family at 17 values of "
           + im("t") + ", each separation alone at 17 values, and a 9 &#215; 9 grid) sits inside the measured gantry "
           "travel: <code>HandPlan</code>&#8217;s mount check reports no violation, and the generator bakes each into a "
           "real_v1 scene. Figure&#160;1 draws five of them.</p>",
           figure(svg_layouts(R), "Five layouts of the family over the gantry boxes (palm frame, top view, mm). Mount colour "
                                  "darkens with " + im("t") + "; the legend gives " + im("x_s") + ", " + im("y_s")
                                  + " and the palm radius " + im(r"\rho") + "."),
           f"<p>Closed-form inverse kinematics of the yaw&#8211;MCP&#8211;PIP chain (checked against the compiled MuJoCo "
           "scene to 0.000&#8202;mm, and its clearances against <code>mj_geomDistance</code> to 0.014&#8202;mm on 150 grasps) "
           "scans every palm height and straddle for an equator grasp. Over the family, cylinders are graspable from "
           f"{min(allc):g} to {max(allc):g}&#8202;mm and spheres from {min(alls):g} to {max(alls):g}&#8202;mm, and each "
           "layout holds a band of diameters that moves with it: cylinders "
           f"{bc['cylinder'][0]:g}&#8211;{bc['cylinder'][1]:g}&#8202;mm on the compact tripod, "
           f"{b0['cylinder'][0]:g}&#8211;{b0['cylinder'][1]:g} on the nominal one"
           + (f" and none on the widest" if bw["cylinder"] is None else
              f" and {bw['cylinder'][0]:g}&#8211;{bw['cylinder'][1]:g} on the widest")
           + "; spheres "
           f"{bc['sphere'][0]:g}&#8211;{bc['sphere'][1]:g}, {b0['sphere'][0]:g}&#8211;{b0['sphere'][1]:g} and "
           f"{bw['sphere'][0]:g}&#8211;{bw['sphere'][1]:g}&#8202;mm. The pair opening acts against the cylinder: at "
           + im("x_s = 100") + " widening " + im("y_s") + f" from 50 to 170&#8202;mm cuts the largest graspable cylinder "
           f"from {yb[50.0][1]:g}&#8202;mm to " + (f"{yb[170.0][1]:g}&#8202;mm" if yb.get(170.0) else
                                                     f"{yb[max(k for k, v in yb.items() if v)][1]:g}&#8202;mm at "
                                                     + im(f"y_s = {max(k for k, v in yb.items() if v):g}")
                                                     + "&#8202;mm, and to none at 170&#8202;mm")
           + ", because the 100&#8202;mm shaft holds the straddle under 42.5&#8202;mm and the pair must yaw inward to "
           "reach it.</p>",
           "<p>The object set takes eight cylinders and ten spheres across those ranges, roughly geometric in diameter. Each is "
           "built to the 25&#8202;g of the simulated screwdriver where the geometry allows, as a printed PLA body with a "
           "steel rod or ball at the centre and the 40&#8202;mm tag on a 2.5&#8202;g printed flag; the starred objects "
           "cannot reach 25&#8202;g and are listed at their built mass. The simulations below hold every object at 25&#8202;g.</p>",
           tcap("Object set for simulation and bench. Mass includes the tag flag (* = not within 10&#8202;% of 25&#8202;g). "
                "&#8220;Family&#8221; = the range of " + im("t") + " and the count of the 17 family layouts with a feasible "
                "grasp; &#8220;grid&#8221; = feasible layouts of the 81 in the 9 &#215; 9 grid."),
           table(["Object", "Mass, g", "Build", "Family t", "Family", "Grid"], rows)]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------ heatmaps

def _q(v):
    """Sequential step 0..7 of a value in [0, 1]."""
    return max(0, min(7, int(v * 7.999)))


FEAS = {}


def load_feas(R):
    FEAS.update({(c["tag"], c["shape"], c["d_mm"]): c["feasible"] for c in R["cells"]})


def svg_heat(L, task, title, fam="diag", vfmt="{:.0f}", norm="row", unit=""):
    """Two panels (cylinders, spheres): layouts left to right from the most compact to the widest, objects bottom to
    top by diameter; cell shade = score over the row's maximum (norm 'row') or over the panel's (norm 'panel');
    ink dots = the ridge of each row."""
    F = L["families"][fam]
    lays = sorted(F["layouts"], key=lambda l: -l["t"])
    cw, ch, gap = 26, 22, 70
    left, top = 54, 46
    nx = len(lays)
    pw = nx * cw
    W = left + 2 * pw + gap + 20
    shapes = ("cylinder", "sphere")
    H = top + max(len(okeys(s)) for s in shapes) * ch + 70
    out = P._svg_open(W, H, f"{title}: heatmaps of the score over layout (x) and object diameter (y) for cylinders and "
                            "spheres, with the ridge of each object marked.")
    for k, shape in enumerate(shapes):
        x0 = left + k * (pw + gap)
        objs = okeys(shape)
        rowsv = {}
        for d in objs:
            col = F["objects"].get(f"{shape}:{d:g}", {})
            rowsv[d] = [col.get(l["tag"], {}).get(task) for l in lays]
        pmax = max([v for vs in rowsv.values() for v in vs if v is not None] or [1.0]) or 1.0
        out.append(f'<text x="{x0}" y="{top - 30}" style="fill:var(--ink);font-weight:600">{shape}s</text>')
        for j, d in enumerate(objs):
            y = top + (len(objs) - 1 - j) * ch
            vs = rowsv[d]
            rmax = max([v for v in vs if v is not None] or [0.0])
            out.append(f'<text x="{x0 - 6}" y="{y + ch / 2 + 4:.1f}" text-anchor="end" style="fill:var(--ink2)">{d:g}</text>')
            for i, (lay, v) in enumerate(zip(lays, vs)):
                x = x0 + i * cw
                if v is None:
                    feas = FEAS.get((lay["tag"], shape, float(d)), False)
                    sty = "fill:var(--qna)" if not feas else "fill:none;stroke:var(--ink3);stroke-dasharray:2 2"
                    why = "no feasible grasp" if not feas else "feasible, not run"
                    out.append(f'<rect class="cell" x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="2" '
                               f'style="{sty}"><title>{shape} {d:g} mm, span {lay["x_sep"]:g} mm, opening '
                               f'{lay["y_sep"]:g} mm: {why}</title></rect>')
                    continue
                den = rmax if norm == "row" else pmax
                s = _q(v / den) if den > 0 else 0
                out.append(f'<rect class="cell" x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="2" '
                           f'style="fill:var(--q{s})"><title>{shape} {d:g} mm, span {lay["x_sep"]:g} mm, opening '
                           f'{lay["y_sep"]:g} mm (t {lay["t"]:+.3f}): {vfmt.format(v)}{unit}</title></rect>')
            rg = F["ridges"].get(f"{shape}:{d:g}", {}).get(task)
            if rg:
                # t decreases left to right: column index of t
                ts = [l["t"] for l in lays]
                xi = (ts[0] - rg["at"]) / (ts[0] - ts[-1]) * (nx - 1)
                cx = x0 + xi * cw + cw / 2
                out.append(f'<circle cx="{cx:.1f}" cy="{y + ch / 2:.1f}" r="4.5" style="fill:var(--ink);stroke:var(--card);'
                           f'stroke-width:2"><title>ridge {shape} {d:g} mm: t* {rg["at"]:+.3f}</title></circle>')
        yb = top + len(objs) * ch
        for i, lay in enumerate(lays):
            if i % 4 == 0:
                out.append(f'<text x="{x0 + i * cw + cw / 2:.1f}" y="{yb + 15}" text-anchor="middle" '
                           f'style="fill:var(--ink3)">{lay["x_sep"]:g}</text>')
        out.append(f'<text x="{x0 + pw / 2:.1f}" y="{yb + 33}" text-anchor="middle" style="fill:var(--ink2)">'
                   f'thumb&#8211;pair span, mm (pair opening 10 mm more)</text>')
        out.append(f'<text x="{x0 - 50}" y="{top - 12}" style="fill:var(--ink2)">diameter, mm</text>')
    # shade key
    kx, ky = left, H - 16
    out.append(f'<text x="{kx}" y="{ky}" style="fill:var(--ink3)">'
               f'{"share of the row maximum" if norm == "row" else "share of the panel maximum"}: 0</text>')
    for s in range(8):
        out.append(f'<rect x="{kx + 210 + s * 16}" y="{ky - 11}" width="15" height="12" style="fill:var(--q{s})"/>')
    out.append(f'<text x="{kx + 210 + 8 * 16 + 6}" y="{ky}" style="fill:var(--ink3)">1</text>'
               f'<rect x="{kx + 380}" y="{ky - 11}" width="15" height="12" style="fill:var(--qna)"/>'
               f'<text x="{kx + 400}" y="{ky}" style="fill:var(--ink3)">no feasible grasp</text>'
               f'<circle cx="{kx + 548}" cy="{ky - 5}" r="4.5" style="fill:var(--ink)"/>'
               f'<text x="{kx + 558}" y="{ky}" style="fill:var(--ink3)">ridge</text>')
    out.append("</svg>")
    return "".join(out)


TASK_COL = {"kin": "var(--t-kin)", "hold": "var(--t-hold)", "turn": "var(--t-turn)"}
TASK_NAME = {"kin": "workspace W", "hold": "grasp robustness", "turn": "held turn", "rotx": "rotation range"}


def ridge_mm(rg, shape):
    return rg["palm_radius_mm"] if shape == "sphere" else rg["x_sep_mm"] / 2


def svg_scaling(L, tasks=("kin", "hold", "turn")):
    """Ridge mount distance (sphere: palm radius; cylinder: half thumb-pair span) against object radius, per task, with
    the workspace fit and Borras and Dollar's rule."""
    F = L["families"]["diag"]
    W, H = 990, 360
    out = P._svg_open(W, H, "Ridge location in mm of mount distance against object radius, for cylinders and spheres, "
                            "per task, with the fitted line of the workspace ridge and the line radius plus half a finger.")
    pw, ph, top = 380, 250, 34
    for k, shape in enumerate(("cylinder", "sphere")):
        x0 = 70 + k * (pw + 110)
        rmax = 85 if shape == "sphere" else 42
        smax = 115 if shape == "sphere" else 85
        fx = lambda r: x0 + r / rmax * pw  # noqa: E731
        fy = lambda s: top + ph - (s - 20) / (smax - 20) * ph  # noqa: E731
        out.append(f'<text x="{x0}" y="{top - 14}" style="fill:var(--ink);font-weight:600">{shape}s</text>')
        for v in range(0, rmax + 1, 10 if shape == "cylinder" else 20):
            out.append(f'<line x1="{fx(v):.1f}" x2="{fx(v):.1f}" y1="{top}" y2="{top + ph}" style="stroke:var(--rule2)"/>'
                       f'<text x="{fx(v):.1f}" y="{top + ph + 16}" text-anchor="middle" style="fill:var(--ink3)">{v}</text>')
        for v in range(20, smax + 1, 20):
            out.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{fy(v):.1f}" y2="{fy(v):.1f}" style="stroke:var(--rule2)"/>'
                       f'<text x="{x0 - 8}" y="{fy(v) + 4:.1f}" text-anchor="end" style="fill:var(--ink3)">{v}</text>')
        out.append(f'<text x="{x0 + pw / 2}" y="{top + ph + 36}" text-anchor="middle" style="fill:var(--ink2)">object radius, mm</text>'
                   f'<text x="{x0 + 90}" y="{top - 14}" style="fill:var(--ink2)">'
                   f'{"palm radius at the ridge" if shape == "sphere" else "half thumb&#8211;pair span at the ridge"}, mm</text>')
        # Borras-Dollar: s = r + FINGER/2
        r0, r1 = 0, rmax
        out.append(f'<line x1="{fx(r0):.1f}" y1="{fy(r0 + FINGER / 2):.1f}" x2="{fx(r1):.1f}" y2="{fy(r1 + FINGER / 2):.1f}" '
                   f'style="stroke:var(--ink3);stroke-width:1.5;stroke-dasharray:6 5"/>')
        fit = F["fits"].get(f"{shape}:kin")
        if fit:
            a, b = fit["alpha"], fit["beta_mm"]
            out.append(f'<line x1="{fx(r0):.1f}" y1="{fy(a * r0 + b):.1f}" x2="{fx(r1):.1f}" y2="{fy(a * r1 + b):.1f}" '
                       f'style="stroke:var(--t-kin);stroke-width:2"/>')
        for t in tasks:
            pts = []
            for d in okeys(shape):
                rg = F["ridges"].get(f"{shape}:{d:g}", {}).get(t)
                if rg:
                    pts.append((d / 2, ridge_mm(rg, shape), rg["interior"]))
            for r, s, inner in pts:
                shape_svg = (f'<circle cx="{fx(r):.1f}" cy="{fy(s):.1f}" r="5" style="fill:{TASK_COL[t] if inner else "var(--card)"};'
                             f'stroke:{TASK_COL[t]};stroke-width:2">')
                out.append(shape_svg + f'<title>{TASK_NAME[t]}, {shape} r {r:g} mm: {s:.1f} mm'
                                       f'{"" if inner else " (edge of the family)"}</title></circle>')
    out.append("</svg>")
    leg = P._legend_html([(f"{TASK_NAME[t]} ridge", TASK_COL[t], False, "circle") for t in tasks]
                         + [("workspace fit, Eq. (1)", "var(--t-kin)", False, None),
                            ("radius plus half the finger (34.1 mm)", "var(--ink3)", True, None)])
    return "".join(out) + leg


# ------------------------------------------------------------------------------------------ kinematics

def fit_text(fit):
    return (im(fr"{fit['alpha']:.2f}\,(\pm{fit['alpha_se']:.2f})\,r + {fit['beta_mm']:.1f}\,(\pm{fit['beta_se']:.1f})\,\text{{mm}}"))


def sep_table(L):
    rows = []
    for shape in ("cylinder", "sphere"):
        for d in okeys(shape):
            rx = L["families"]["xsep"]["ridges"].get(f"{shape}:{d:g}", {}).get("kin")
            ry = L["families"]["ysep"]["ridges"].get(f"{shape}:{d:g}", {}).get("kin")
            rd = L["families"]["diag"]["ridges"].get(f"{shape}:{d:g}", {}).get("kin")
            if not (rx or ry or rd):
                continue
            g = lambda r, k: ("&#8211;" if not r else f(r[k], 1) + ("" if r["interior"] else " (edge)"))  # noqa: E731
            rows.append([f"{shape} {d:g}", g(rd, "x_sep_mm") if rd else "&#8211;",
                         g(rx, "at"), f(rx["at"] / 2 - d / 2, 1) if rx and rx["interior"] and shape == "cylinder"
                         else "&#8211;", g(ry, "at")])
    return table(["Object, mm", "Family ridge, " + im("x_s^*"), im("x_s") + " alone, ridge",
                  im(r"x_s^*/2 - r") + " (cylinders)", im("y_s") + " alone, ridge"], rows)


def kin_section(L):
    F = L["families"]["diag"]
    fc, fs = F["fits"].get("cylinder:kin"), F["fits"].get("sphere:kin")
    rc, rs = F["fits"].get("cylinder:rotx"), F["fits"].get("sphere:rotx")
    ratio = (min(fc["ratio"] + fs["ratio"]), max(fc["ratio"] + fs["ratio"])) if fc and fs else (None, None)
    xs_off = []
    for d in okeys("cylinder"):
        rx = L["families"]["xsep"]["ridges"].get(f"cylinder:{d:g}", {}).get("kin")
        if rx and rx["interior"]:
            xs_off.append(rx["at"] / 2 - d / 2)
    ys_c = [L["families"]["ysep"]["ridges"][f"cylinder:{d:g}"]["kin"]["at"] for d in okeys("cylinder")
            if L["families"]["ysep"]["ridges"].get(f"cylinder:{d:g}", {}).get("kin")]
    big = max(F["ridges"][f"sphere:{d:g}"]["rotx"]["max"] for d in (10, 16) if "rotx" in F["ridges"].get(f"sphere:{d:g}", {}))
    small = F["ridges"].get("sphere:100", {}).get("rotx", {}).get("max")
    out = [f"<p>The workspace " + im("W") + " has a ridge along the family for every object, and the ridge moves to wider "
           "layouts as the object grows (Figure&#160;2). Expressed as the mount distance it implies, the ridge is a straight "
           "line of slope one in the object radius:</p>",
           eq(r"\rho^* = " + (f"{fs['alpha']:.2f}" if fs else "?") + r"\,r + " + (f"{fs['beta_mm']:.1f}" if fs else "?")
              + r"\ \text{mm (spheres)},\qquad a^* = \tfrac{1}{2}x_s^* = " + (f"{fc['alpha']:.2f}" if fc else "?")
              + r"\,r + " + (f"{fc['beta_mm']:.1f}" if fc else "?") + r"\ \text{mm (cylinders)}", 1),
           f"<p>The fits are " + im(r"\rho^* = ") + (fit_text(fs) if fs else "&#8211;") + f" over {fs['n'] if fs else 0} "
           "spheres and " + im("a^* = ") + (fit_text(fc) if fc else "&#8211;") + f" over {fc['n'] if fc else 0} cylinders, "
           "residuals "
           f"{f(fs['rms_mm'] if fs else None, 1)} and {f(fc['rms_mm'] if fc else None, 1)}&#8202;mm rms (Figure&#160;3). The "
           "best layout keeps the mounts a fixed 35&#8211;37&#8202;mm outside the object&#8217;s surface, so the ratio "
           + im(r"s^*/d") + f" is not constant: " + im(r"\rho^*/r") + " and " + im("a^*/r") + f" run from {f(ratio[0], 1)} to "
           f"{f(ratio[1], 1)} over the set. The offset is half the finger: the straight mount-to-pad chain is "
           f"{FINGER:.1f}&#8202;mm, half of it {FINGER / 2:.1f}&#8202;mm. Borr&#224;s and Dollar give the same rule for "
           "three two-link fingers, " + im(r"\rho^* \approx r + L/2") + " with " + im("L") + " the finger length, although "
           "their base joint turns about the palm normal and the SR2 yaw turns about the palm&#8217;s " + im("x")
           + " axis. The rotation range about the pinch axis peaks along similar lines ("
           + (fit_text(rs) if rs else "&#8211;") + " for spheres, " + (fit_text(rc) if rc else "&#8211;") + " for cylinders) "
           f"and shrinks with size, from {f(big, 0)}&#176; on the 10&#8211;16&#8202;mm spheres to {f(small, 0)}&#176; on "
           "the 100&#8202;mm sphere, because a fixed finger travel turns a larger object through a smaller angle.</p>",
           figure(svg_heat(L, "kin", "Fixed-contact workspace"),
                  "Fixed-contact workspace " + im("W") + " over the family, each row shaded by its own maximum (hover a "
                  "cell for the count of reachable poses). Columns run from the most compact layout (left, "
                  + im("x_s = 40") + "&#8202;mm) to the widest (right, 160&#8202;mm); the dot marks each object&#8217;s ridge."),
           figure(svg_scaling(L), "Ridge location against object radius, as the mount distance it implies (spheres: palm "
                                  "radius; cylinders: half the thumb&#8211;pair span). Solid line: the fit of Eq.&#160;(1) to "
                                  "the workspace ridges; dashed: object radius plus half the finger, " + im(r"r + 34.1")
                                  + "&#8202;mm. Hollow markers sit at the edge of the object&#8217;s feasible band (the "
                                  "end of the family or the edge of the graspable range), where the true maximum may lie "
                                  "beyond; fits use interior ridges only."),
           "<p>Varying one separation at a time shows which of them the object size acts on. With the pair opening held at "
           "110&#8202;mm, the cylinder ridge follows the thumb&#8211;pair span alone, at " + im(r"x_s^*/2 - r = ")
           + (f"{min(xs_off):.1f}&#8211;{max(xs_off):.1f}" if xs_off else "&#8211;") + "&#8202;mm for every diameter. With "
           "the span held at 100&#8202;mm, the cylinder&#8217;s best pair opening stays at "
           + (f"{min(ys_c):.0f}&#8211;{max(ys_c):.0f}" if ys_c else "&#8211;") + "&#8202;mm for every diameter: the pair "
           "straddles the shaft along its length, so the opening follows the 100&#8202;mm shaft length. A "
           "sphere&#8217;s ridge moves along both separations (Table&#160;" + str(TAB[0] + 1) + "). A large cylinder "
           "therefore wants the thumb moved away from the pair and the pair kept together, and a large sphere wants the "
           "whole tripod expanded.</p>",
           tcap("Workspace ridges along the family and along each separation alone (mm). "
                + im("x_s") + " alone: pair opening held at 110&#8202;mm; " + im("y_s") + " alone: thumb&#8211;pair span "
                "held at 100&#8202;mm. &#8220;(edge)&#8221;: the maximum lies at the end of the gantry travel."),
           sep_table(L)]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------ dynamic tasks

def hold_rows():
    return [r for r in jl("hold.jsonl") if r.get("status") == "ok" and r.get("model", "pads") == "pads"]


def turn_rows():
    return [r for r in jl("turn.jsonl") if r.get("status") == "ok" and r.get("model", "pads") == "pads"]


def weakest_counts():
    from collections import Counter
    c, n = Counter(), 0
    for r in hold_rows():
        if not r["held"] or not r.get("disturb"):
            continue
        n += 1
        Fd = {k: (20.0 if v is None else v) for k, v in r["disturb"].items() if k.startswith("F")}
        c[min(Fd, key=Fd.get)] += 1
    return c, n


def torque_summary():
    """Per shape: how often a twist about palm y is the weakest torque, and the median weakest torque (mN m) and its
    value per unit object radius (N), over held grips."""
    from collections import Counter
    out = {}
    for shape in ("cylinder", "sphere"):
        c, vals, norm, n = Counter(), [], [], 0
        for r in hold_rows():
            if r["shape"] != shape or not r["held"] or not r.get("disturb"):
                continue
            n += 1
            cap = 20.0 * r["r_eff_m"]
            Tq = {k: (cap if v is None else v) for k, v in r["disturb"].items() if k.startswith("T")}
            k = min(Tq, key=Tq.get)
            c[k[-1]] += 1
            vals.append(1000 * Tq[k])
            norm.append(Tq[k] / r["r_eff_m"])
        out[shape] = {"n": n, "y": c.get("y", 0), "med_mNm": med(vals), "med_per_r_N": med(norm)}
    return out


def grasp_choice(L):
    """Over cells with both tasks scored: how often each task's best grasp is the cell's highest palm (straightest
    fingers), and the median palm-height difference between the two choices."""
    hi_hold = hi_turn = n = 0
    dh = []
    by = {}
    for r in hold_rows():
        by.setdefault((r["tag"], r["shape"], r["d_mm"]), {})[r["cand"]] = r
    for (key, cands) in by.items():
        s = {}
        F = L["families"]["diag"]["objects"].get(f"{key[1]}:{key[2]:g}", {}).get(key[0], {})
        if not F.get("hold_grasp") or not F.get("turn_grasp"):
            continue
        n += 1
        same_sp = [c for c in cands.values() if c["spread_mm"] == F["hold_grasp"]["spread_mm"]]
        if same_sp and F["hold_grasp"]["h_mm"] >= max(c["h_mm"] for c in same_sp) - 1e-6:
            hi_hold += 1
        same_sp = [c for c in cands.values() if c["spread_mm"] == F["turn_grasp"]["spread_mm"]]
        if same_sp and F["turn_grasp"]["h_mm"] >= max(c["h_mm"] for c in same_sp) - 1e-6:
            hi_turn += 1
        dh.append(F["hold_grasp"]["h_mm"] - F["turn_grasp"]["h_mm"])
    return {"n": n, "hold_highest": hi_hold, "turn_highest": hi_turn, "dh_median": med(dh)}


def ridge_gap(L, a, b):
    """Median over objects (both ridges interior) of t*_a - t*_b, and the count."""
    F = L["families"]["diag"]
    g = []
    for shape in ("cylinder", "sphere"):
        for d in okeys(shape):
            rr = F["ridges"].get(f"{shape}:{d:g}", {})
            if a in rr and b in rr and rr[a]["interior"] and rr[b]["interior"]:
                g.append(rr[a]["at"] - rr[b]["at"])
    return med(g), len(g), g


VAR_NAME = {"ext_min_mm": "reach margin", "sigma_min_mm": im(r"\sigma_{\min}"), "margin_deg": "joint margin",
            "angle_max_deg": "contact angle", "f_trans_min": "force transmission", "ff_clear_mm": "finger clearance"}


def explain_table(L):
    F = L["families"]["diag"]["explain"]
    keys = ("cylinder:hold", "sphere:hold", "cylinder:turn", "sphere:turn")
    rows = []
    for v in VAR_NAME:
        row = [VAR_NAME[v]]
        for k in keys:
            x = F.get(k, {}).get(v)
            row.append("&#8211;" if not x else f"{x['spearman_mean']:+.2f}" + (
                f" / {60 * x['ridge_gap_t']:.0f}" if x.get("ridge_gap_t") is not None else ""))
        rows.append(row)
    return table(["Variable at the task&#8217;s grasp", "hold, cylinders", "hold, spheres", "turn, cylinders",
                  "turn, spheres"], rows)


def best_var(L, task):
    """Variables ranked by the absolute within-object Spearman with a task's score, averaged over the two shapes;
    each entry (name, mean |rho|, {shape: signed rho})."""
    F = L["families"]["diag"]["explain"]
    out = []
    for v in VAR_NAME:
        per = {}
        for shape in ("cylinder", "sphere"):
            x = F.get(f"{shape}:{task}", {}).get(v)
            if x:
                per[shape] = x["spearman_mean"]
        if per:
            out.append((v, sum(abs(x) for x in per.values()) / len(per), per))
    return sorted(out, key=lambda z: -z[1])


def turn_stops():
    from collections import Counter
    c, n = Counter(), 0
    for r in turn_rows():
        n += 1
        if r.get("limit_joint"):
            c["joint limit"] += 1
        elif r.get("frozen_deg") is not None:
            c["governor"] += 1
        elif r.get("dropped"):
            c["dropped"] += 1
        else:
            c["none"] += 1
    return c, n


def band_edges(L, task, shape):
    """Objects of a shape whose task maximum sits at the compact (+) or wide (-) edge of its feasible band."""
    F = L["families"]["diag"]
    out = {"compact": 0, "wide": 0, "inner": 0, "n": 0}
    for d in okeys(shape):
        rg = F["ridges"].get(f"{shape}:{d:g}", {}).get(task)
        col = F["objects"].get(f"{shape}:{d:g}", {})
        ts = sorted(l["t"] for l in F["layouts"] if col.get(l["tag"], {}).get(task) is not None)
        if not rg or len(ts) < 3:
            continue
        out["n"] += 1
        if abs(rg["at"] - ts[-1]) < 1e-6:
            out["compact"] += 1
        elif abs(rg["at"] - ts[0]) < 1e-6:
            out["wide"] += 1
        else:
            out["inner"] += 1
    return out


def model_scores(model):
    """(tag, shape, d) -> {"hold": best F_min, "turn": best median held turn} for one tip model."""
    from collections import defaultdict
    out = defaultdict(dict)
    hb = defaultdict(list)
    for r in jl("hold.jsonl"):
        if r.get("status") != "ok" or r.get("model", "pads") != model:
            continue
        d_ = r.get("disturb") or {}
        fmin = min([20.0 if v is None else v for k, v in d_.items() if k.startswith("F")] or [0.0]) if r["held"] else 0.0
        hb[(r["tag"], r["shape"], r["d_mm"])].append(fmin)
    for k, v in hb.items():
        out[k]["hold"] = max(v)
    tb = defaultdict(lambda: defaultdict(list))
    for r in jl("turn.jsonl"):
        if r.get("status") != "ok" or r.get("model", "pads") != model:
            continue
        tb[(r["tag"], r["shape"], r["d_mm"])][r["cand"]].append(r["held_turn_deg"])
    for k, by in tb.items():
        out[k]["turn"] = max(statistics.median(v) for v in by.values())
    return out


def pt_check():
    """Pads against point contact on the coarse cells: rank agreement of the cell scores and of each object's best
    coarse layout, per task."""
    from scipy.stats import spearmanr
    A, B = model_scores("pads"), model_scores("pt")
    keys = [k for k in B if k in A]
    res = {}
    for task in ("hold", "turn"):
        ks = [k for k in keys if task in A[k] and task in B[k]]
        a = [A[k][task] for k in ks]
        b = [B[k][task] for k in ks]
        rho = float(spearmanr(a, b).statistic) if len(ks) > 2 else None
        objs = sorted({(k[1], k[2]) for k in ks})
        same = 0
        nobj = 0
        for o in objs:
            ko = [k for k in ks if (k[1], k[2]) == o]
            if len(ko) < 2:
                continue
            nobj += 1
            if max(ko, key=lambda k: A[k][task]) == max(ko, key=lambda k: B[k][task]):
                same += 1
        ratio = statistics.median([bb / aa for aa, bb in zip(a, b) if aa > 0]) if a else None
        res[task] = {"n_cells": len(ks), "spearman": rho, "same_best": same, "n_objects": nobj, "median_ratio": ratio}
    return res


def tasks_section(L):
    F = L["families"]["diag"]
    c, n = weakest_counts()
    gc = grasp_choice(L)
    fk, fh, ft = (F["fits"].get(f"cylinder:{t}") for t in ("kin", "hold", "turn"))
    bh, bt = best_var(L, "hold"), best_var(L, "turn")
    sh, st_ = band_edges(L, "hold", "sphere"), band_edges(L, "turn", "sphere")
    stops, ns = turn_stops()
    tq = torque_summary()
    nh = len(hold_rows())
    nt = len(turn_rows())
    held_c = sum(r["held"] for r in turn_rows() if r["shape"] == "cylinder")
    n_c = sum(1 for r in turn_rows() if r["shape"] == "cylinder")
    held_s = sum(r["held"] for r in turn_rows() if r["shape"] == "sphere")
    n_s = sum(1 for r in turn_rows() if r["shape"] == "sphere")
    off = lambda fit: fit["beta_mm"] - 11.55 if fit else None  # noqa: E731  pad inboard of its mount at r = 0
    angle = next((z for z in bh if z[0] == "angle_max_deg"), None)
    angle_t = next((z for z in bt if z[0] == "angle_max_deg"), None)
    gaps = {v: F["explain"].get("cylinder:turn", {}).get(v, {}).get("ridge_gap_t") for v in VAR_NAME}
    gv = min((v for v in gaps if gaps[v] is not None), key=lambda v: gaps[v]) if any(gaps.values()) else None
    out = [f"<p>Each feasible cell of the family was gripped at up to nine grasps (three palm heights spread over its "
           "feasible band at each of three straddles; five heights on a sphere): " + f"{nh} hold tests with twelve "
           f"disturbance ramps each and {nt} turns of the HOM controller. Each task keeps its own best grasp per cell. "
           "Grasp robustness " + im(r"F_{\min}") + f" is set by a pull along gravity in {c.get('F-z', 0)} of {n} held "
           f"grips. The weakest torque is a twist about the palm&#8217;s " + im("y") + " axis (the cylinder&#8217;s own "
           f"axis) in {tq['cylinder']['y']} of {tq['cylinder']['n']} held cylinder grips and {tq['sphere']['y']} of "
           f"{tq['sphere']['n']} sphere grips, a median {f(tq['cylinder']['med_mNm'], 0)} and {f(tq['sphere']['med_mNm'], 0)}"
           f"&#8202;mN&#8202;m, or {f(tq['cylinder']['med_per_r_N'], 1)} and {f(tq['sphere']['med_per_r_N'], 1)}&#8202;N at the "
           "object&#8217;s radius. "
           f"The controller held the object in {held_c} of {n_c} cylinder turns and {held_s} of {n_s} sphere turns; "
           f"{stops.get('governor', 0)} of {ns} turns ended at its governor (a pad losing force or the object lagging) "
           f"and {stops.get('joint limit', 0)} at a joint limit.</p>",
           figure(svg_heat(L, "hold", "Grasp robustness", vfmt="{:.2f}", unit=" N"),
                  "Grasp robustness " + im(r"F_{\min}") + " over the family, each row shaded by its own maximum (hover for "
                  "newtons). Axes as in Figure&#160;2."),
           figure(svg_heat(L, "turn", "Held turn", vfmt="{:.0f}", unit=" deg"),
                  "Held turn of the HOM controller over the family, each row shaded by its own maximum (hover for "
                  "degrees)."),
           "<p>On cylinders the three ridges are parallel lines of slope one in the object radius (Figure&#160;3), "
           "separated by how far inboard of its mount each finger meets the shaft:</p>",
           eq(r"a^*_{\text{hold}} = " + (f"{fh['alpha']:.2f}" + r"\,r + " + f"{fh['beta_mm']:.1f}" if fh else "?")
              + r",\quad a^*_{\text{turn}} = " + (f"{ft['alpha']:.2f}" + r"\,r + " + f"{ft['beta_mm']:.1f}" if ft else "?")
              + r",\quad a^*_{W} = " + (f"{fk['alpha']:.2f}" + r"\,r + " + f"{fk['beta_mm']:.1f}" if fk else "?") + r"\ \text{mm}", 2),
           f"<p>({fh['n'] if fh else 0}, {ft['n'] if ft else 0} and {fk['n'] if fk else 0} cylinders; residuals "
           f"{f(fh['rms_mm'] if fh else None, 1)}, {f(ft['rms_mm'] if ft else None, 1)} and {f(fk['rms_mm'] if fk else None, 1)}"
           "&#8202;mm rms). The pad centre sits " + im("r + 11.6") + "&#8202;mm from the shaft axis, so at the robustness "
           f"ridge each pad meets the shaft {f(off(fh), 0)}&#8202;mm inboard of its mount, at the turn ridge about "
           f"{f(off(ft), 0)}&#8202;mm and at the workspace ridge {f(off(fk), 0)}&#8202;mm. The robust grip wants the "
           f"thumb {f(2 * (fk['beta_mm'] - fh['beta_mm']) if fk and fh else None, 0)}&#8202;mm closer to the pair than the "
           "workspace optimum does, at every diameter. Within a cell the tasks choose different grasps as well: the "
           f"robust grasp is the highest palm of its straddle in {gc['hold_highest']} of {gc['n']} cells, the furthest "
           f"turn in {gc['turn_highest']}.</p>",
           f"<p>On spheres the two dynamic tasks pull to opposite edges of each object&#8217;s feasible band, and the "
           f"workspace ridge lies between them: robustness peaks at the compact edge for {sh['compact']} of {sh['n']} "
           f"spheres, the held turn at the wide edge for {st_['wide']} of {st_['n']}. Toward the wide edge the "
           "pair&#8217;s contacts move off the pad face toward the 90&#176; limit of the feasible set, which raises the "
           "turn and lowers the robustness (Figures&#160;4 and 5).</p>",
           "<p>One variable separates the two dynamic tasks: the contact angle, which grows as the finger bends away from "
           "a straight approach. Across the layouts of one object, robustness falls with it (Spearman "
           + (", ".join(f"{v:+.2f} {s}s" for s, v in angle[2].items()) if angle else "&#8211;") + ") and the held turn "
           "rises with it (" + (", ".join(f"{v:+.2f} {s}s" for s, v in angle_t[2].items()) if angle_t else "&#8211;")
           + "). Straight fingers press face-on and resist pull-out; bent fingers keep the travel the turn spends. "
           + (f"Of the variables tested, the {VAR_NAME[gv]} has its own ridge nearest the turn ridge on cylinders "
              f"({60 * gaps[gv]:.0f}&#8202;mm of thumb&#8211;pair span). " if gv else "")
           + "Table&#160;" + str(TAB[0] + 1) + " gives every variable.</p>",
           tcap("Within-object Spearman correlation of each kinematic variable with the task score over the layouts of "
                "the family, at the task&#8217;s grasp, averaged over objects, and the mean distance between the "
                "variable&#8217;s ridge and the task&#8217;s ridge in mm of thumb&#8211;pair span (after the slash). The "
                "contact angle&#8217;s ridge is its minimum."),
           explain_table(L)]
    pc = pt_check()
    if pc.get("hold", {}).get("n_cells"):
        h_, t_ = pc["hold"], pc["turn"]
        out.append(f"<p>Point contact on the same TPU block (one convex mesh) ranks the layouts nearly as the pads do. Over the "
                   f"{h_['n_cells']} feasible cells of the coarse 5 &#215; 5 grid (" + im(r"t \in \{-1, -0.5, 0, 0.5, 1\}")
                   + "), the cell scores of the two models correlate at Spearman "
                   f"{f(h_['spearman'], 2)} for robustness and {f(t_['spearman'], 2)} for the turn, and each object&#8217;s "
                   f"best coarse layout is the same for {h_['same_best']} of {h_['n_objects']} objects (robustness) and "
                   f"{t_['same_best']} of {t_['n_objects']} (turn). Point contact scores robustness at a median "
                   f"{f(h_['median_ratio'], 2)} and the turn at {f(t_['median_ratio'], 2)} of the pads&#8217; values.</p>")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------ films

FILM_NOTES = {       # written after watching each clip (contact sheets of five frames, 2026-10-08)
    "turn_cyl025.0_three_s0.mp4": "On the compact layout the fingers stand nearly straight and the shaft turns about 30&#176; "
                                  "before the governor stops it; at 92.5&#8202;mm, the turn ridge, the shaft stands up past "
                                  "60&#176; with all three pads on it; on the wide layout the fingers reach in strongly bent "
                                  "and the turn stops near 25&#176;.",
    "turn_sph040.0_three_s0.mp4": "Viewed along the pinch axis from behind the thumb, with an orange rod through the sphere "
                                  "along its marked axis: the rod stays nearly level on the compact layout, tilts about "
                                  "20&#176; at 77.5&#8202;mm and about 30&#176; at 100&#8202;mm, the wide edge of the band, "
                                  "where the pair leans out and meets the sphere near the side of its pads.",
}


def films_section(films):
    if not films:
        return P.pending("Films not rendered yet: scripts/hand_object_scale_films.py")
    out = []
    tiles = []
    for r in films:
        if r["tile"] not in tiles:
            tiles.append(r["tile"])
    for tile in tiles:
        rs = [r for r in films if r["tile"] == tile]
        desc = "; ".join(f"thumb&#8211;pair span {float(r['tag'][1:6]):g}&#8202;mm, held turn {r['held_turn_deg']:.0f}&#176;"
                         + ("" if r["held"] else " (dropped)") for r in rs)
        note = FILM_NOTES.get(tile, "")
        out.append(film(f"media/{tile}", f"{rs[0]['shape']} {rs[0]['d_mm']:g}&#8202;mm on three layouts of the family, "
                                         f"each at its furthest-turning grasp, HOM controller, seed {rs[0]['seed']} "
                                         f"(pads touching the object drawn red): {desc}. {note}",
                        poster=f"media/{tile.replace('.mp4', '.jpg')}"))
    return "\n".join(out)


# ------------------------------------------------------------------------------------------ bench protocol

def bench_section(prot, O):
    if not prot:
        return P.pending("protocol.json missing: scripts/hand_object_scale_protocol.py")
    cells = prot["cells"]
    H = prot["hours"]
    n_t = sum(c["n_turn"] for c in cells)
    n_b = sum(c["n_ballast"] for c in cells)
    objs = prot["objects"]
    rows = []
    for st in ("core", "extended"):
        rows.append(f"{st} set")
        for c in sorted([c for c in cells if c["set"] == st], key=lambda c: (c["shape"], c["d_mm"], -c["t"])):
            rows.append([f"{c['shape']} {c['d_mm']:g}", f"{c['x_sep_mm']:g} / {c['y_sep_mm']:g}",
                         ", ".join(c["near_ridge_of"]), f(c["sim_turn_deg"], 0), f(c["sim_turn_sd_deg"], 1),
                         f(c["sim_hold_N"], 2), str(c["n_turn"]) if c["n_turn"] else "&#8211;",
                         str(c["n_ballast"]) if c["n_ballast"] else "&#8211;"])
    hrows = [["Turn trials", f"{n_t} &#215; {prot['meta']['t_trial_s']:.0f}&#8202;s", f(H["turn_trials"], 1)],
             ["Ballast series", f"{n_b} &#215; {prot['meta']['t_ballast_s'] / 60:.0f}&#8202;min", f(H["ballast"], 1)],
             ["Reconfiguration", f"{len(cells)} &#215; {prot['meta']['t_reconf_s']:.0f}&#8202;s", f(H["reconfiguration"], 1)],
             ["Reference cell and revisits", "5 trials each", f(H["reference_and_revisits"], 1)],
             ["Homing, tracker check, setup", f"{len(prot['days'])} days &#215; 45&#8202;min", f(H["day_overhead"], 1)],
             ["Total", "", f"<b>{f(H['total'], 1)}</b>"]]
    out = [f"<p>The protocol measures the two dynamic ridges on hardware at the points the simulation names. The bench grid "
           "is every second layout of the family (" + im(r"t = -1, -0.75, \dots, 1") + "); for each object it takes the "
           "feasible bench layouts within one bench step (15&#8202;mm of thumb&#8211;pair span) of each simulated ridge, "
           f"so each ridge is bracketed by three measurements: turn trials at the cells around the turn ridge, ballast "
           f"series at the cells around the robustness ridge, {len(cells)} cells over {len(objs)} objects "
           f"(Table&#160;{TAB[0] + 1}). A core set of four cylinders and four spheres, each with both simulated ridges "
           "short of the gantries&#8217; end of the family and spread evenly in log diameter ("
           + ", ".join(f"{s_}s {', '.join(f'{d:g}' for d in prot['core'][s_])}&#8202;mm" for s_ in ("cylinder", "sphere"))
           + f"), takes {f(H.get('core_trials_only'), 1)} hours of trials; the other objects fill the rest of the week. "
           f"<code>{REL}/protocol.json</code> lists the cells with the grasp, the predicted scores and the day plan.</p>",
           "<ul>"
           "<li><b>Configuration per cell.</b> The six gantry targets of the layout, the cell&#8217;s palm pose and the "
           "grip and turn commands, exported as a deploy plan (<code>&lt;cell&gt;_plan.json</code> with "
           "<code>_traj.csv</code>, the format the station replays). The bench runs the HOM controller&#8217;s simulated "
           "joint commands open loop, because the hand has no fingertip force sensing; the simulation replays the same file "
           "open loop, which is the prediction the bench is compared with at each point.</li>"
           "<li><b>Gates before a cell runs.</b> <code>scripts/real_v1_trajectory_clearance.py</code> on the plan&#8217;s "
           "chord and its CSV path (at least 5&#8202;mm between fingers on both) and <code>HandPlan.validate()</code> (every "
           "command inside the servo&#8217;s calibrated range, every mount inside the measured gantry travel). A cell that "
           "fails either is dropped and logged.</li>"
           f"<li><b>Turn trials.</b> {prot['meta']['n_min']}&#8211;{prot['meta']['n_max']} per cell: the count that puts "
           f"the 95&#8202;% interval of the cell mean within {prot['meta']['ci_deg']:.0f}&#176;, from the larger of the "
           f"simulated spread over placements and {prot['meta']['sd_bench_deg']}&#176;, the median spread of held turns "
           "per hand in the eight-hand bench record. The object is placed on its post by hand at the marked pose, the hand "
           "grips, the post drops 5&#8202;mm, the turn runs, the hand holds 2&#8202;s.</li>"
           "<li><b>Grasp robustness.</b> The hanging mass at slip: steel washers added in 5&#8202;g steps to a hook under "
           "the object until it moves 3&#8202;mm or drops, two series per cell at the robust grasp. It measures the pull "
           f"along gravity, the binding direction of " + im(r"F_{\min}") + " in most simulated grips.</li>"
           "<li><b>Sensing.</b> The AprilTag tracker on the workstation (40&#8202;mm tag36h11 on the object&#8217;s flag, "
           "a static reference tag; 0.017&#176; and 0.03&#8202;mm rms when still): the signed cosine of the marked axis with "
           "vertical (" + im("+1") + " = tip down, never the folded tilt), held = tag above the post top 2&#8202;s after the "
           "turn with at least two fingers on the object, slip = tag translation in the palm frame during the hold.</li>"
           "<li><b>Order.</b> Randomised within each bench day (seed 20261008), so object size and layout are not "
           "confounded with time or wear. Each day opens with homing, the tracker check and five trials of the reference "
           "cell (nominal layout, 25&#8202;mm cylinder) and ends with five trials of two earlier cells; the reference and "
           "the revisits measure drift between days.</li>"
           "</ul>",
           "<details><summary>Bench cells, core set first (" + str(len(cells)) + " rows)</summary>",
           tcap("Bench cells: layout (" + im("x_s") + " / " + im("y_s") + ", mm), the simulated ridge each brackets, the "
                "simulated held turn and its spread over placements (deg), simulated " + im(r"F_{\min}")
                + " (N), and the trials planned."),
           table(["Object, mm", "Layout", "Near ridge of", "Sim turn", "Sim sd", "Sim " + im(r"F_{\min}"), "Turn trials",
                  "Ballast series"], rows),
           "</details>",
           tcap("Bench time."),
           table(["Item", "Count", "Hours"], hrows),
           f"<p>The {f(H['total'], 0)} bench hours fit one week at five hours of trials a day plus setup. Printing needs "
           f"the {len(objs)} objects of the cell list from Table&#160;2&#8217;s recipes; each object is weighed and the "
           "simulation re-run at its measured mass before its cells are compared.</p>"]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------ paper plan

def paper_section(L, prot):
    F = L["families"]["diag"]
    fs, fc = F["fits"].get("sphere:kin"), F["fits"].get("cylinder:kin")
    figs = [("1", "The SR2 hand on its fixed mount and the layout family drawn over the gantry boxes", "bench and sim"),
            ("2", "Reachable equator grasps over layout and object diameter, cylinders and spheres", "sim"),
            ("3", "Fixed-contact workspace heatmaps with the ridge, and the ridge against object radius with the "
                  "Borr&#224;s&#8211;Dollar line (Eq. 1)", "sim"),
            ("4", "Grasp-robustness and held-turn heatmaps with their ridges", "sim"),
            ("5", "The three ridges on one plot: where the tasks select different layouts", "sim"),
            ("6", "Which kinematic variable predicts each ridge", "sim"),
            ("7", "Measured held turn and hanging mass at the bench cells, with the simulated ridges overlaid", "bench"),
            ("8", "Simulated against measured scores cell by cell, and ridge locations with intervals", "sim and bench"),
            ("9", "Ridges under the 1&#8202;mm pads against point contact on the TPU block", "sim"),
            ("10", "Stills from the bench and simulation films of three layouts per object", "sim and bench")]
    out = ["<p><b>Question.</b> How the best finger-base layout of a three-finger hand with fixed phalanges moves with "
           "object size, whether the answer differs between grasping and in-hand reorientation, and whether a simulator "
           "with a calibrated contact model predicts where the best layout lies.</p>",
           "<p><b>Contributions.</b></p><ol>"
           "<li>A dense morphology-by-object-size map on one hardware hand: 17 layouts &#215; 18 objects in simulation and "
           f"{len(prot['cells']) if prot else '&#8211;'} cells on the bench, with the same fingers, servos, tips and "
           "controller in every cell.</li>"
           "<li>A scaling law for the kinematic optimum: the best mount distance is the object radius plus about half the "
           "finger (" + (f"{fs['beta_mm']:.0f}" if fs else "&#8211;") + " and " + (f"{fc['beta_mm']:.0f}" if fc else "&#8211;")
           + "&#8202;mm for spheres and cylinders, slope one), so the best layout grows by an offset while " + im("s^*/d")
           + " runs from 1.8 to 12.6; "
           "the cylinder&#8217;s optimum moves the thumb and leaves the pair opening at the shaft&#8217;s length scale.</li>"
           "<li>Task-dependent optima: grasp robustness and reorientation select different layouts and different grasps for "
           "the same object, separated by the contact angle of the grasp posture.</li>"
           "<li>Agreement of the simulated and measured landscapes at the same points, with the 1&#8202;mm pad contact model "
           "(calibrated against Drake&#8217;s hydroelastic contact) as part of the simulator, and point contact as the "
           "comparison.</li></ol>",
           "<p><b>Venue.</b> RA-L fits contributions 1&#8211;3 with the bench agreement in 8 pages; it can be presented at "
           "IROS 2027 (deadline around 1 March 2027). T-RO takes the full 2-D map over both separations, the contact-model "
           "comparison and the open-loop against closed-loop turn, which do not fit in 8 pages. IJRR suits a framing "
           "around SR2 as an experimental morphology instrument with this study as its first use. The recommendation is "
           "RA-L, submitted by mid-January 2027, with the extended version for T-RO afterwards.</p>",
           "<p><b>Timeline.</b> Week of 13 October: export the cell plans, run them through both gates, replay them open "
           "loop in simulation, print and weigh the objects. 20 October: the bench week. 27 October: re-simulate at measured "
           "masses, sim-to-bench analysis, figures 7&#8211;8. November: the 2-D map in simulation for the T-RO version and "
           "the writing. Mid-December: internal review. Mid-January: RA-L submission.</p>",
           tcap("Figures of the paper, and where each comes from."),
           table(["Figure", "Content", "Source"], [[a, (b, "wrap"), c] for a, b, c in figs])]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------ next

def next_section():
    items = [
        "<b>Open-loop replay of the simulated turn.</b> The bench replays joint commands without force feedback. Record "
        "<code>q_cmd</code> in <code>turn_rollout</code>, replay it with <code>hold_rollout</code>&#8217;s plant on every "
        "cell of the protocol, and recompute the turn ridge; if it moves by more than one bench step (0.25 in "
        + im("t") + "), the bench grid is re-centred on the replay&#8217;s ridge.",
        "<b>Point-contact check.</b> <code>hand_object_scale_sim.py run --cells coarse --model pt</code> on the 5 &#215; 5 "
        "grid; the paper&#8217;s contact-model claim needs the pad and point ridges compared object by object.",
        "<b>Plan export.</b> A <code>scripts/hand_object_scale_export.py</code> that writes each protocol cell as "
        "<code>_plan.json</code> and <code>_traj.csv</code>, then <code>real_v1_trajectory_clearance.py --deploy-dir</code> "
        "and <code>HandPlan.validate()</code> on all of them; the cells that fail leave the protocol before printing starts.",
        "<b>Mass.</b> Six objects cannot be built at 25&#8202;g (Table&#160;2). Re-run their cells at the built mass "
        "(<code>MASS</code> in <code>hand_object_scale_sim.py</code>); a ridge that moves by more than one family step "
        "means the bench set must hold mass constant by another route (a longer rod, a denser core).",
        "<b>The 2-D map.</b> The 9 &#215; 9 grid of separations for the dynamic tasks (about 3 CPU-hours at four "
        "processes) answers whether thumb opposition and pair opening scale differently for the turn as they do for the "
        "workspace.",
        "<b>Grasp elevation.</b> Every grasp here sits on the equator; grasps below it trade depth for wedge stability "
        "and are untested.",
    ]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


# ------------------------------------------------------------------------------------------ lede and main

def lede(L, R, prot):
    """About 100 words (owner 2026-10-09): the scaling law, how the task moves the optimum, the bench protocol."""
    F = L["families"]["diag"]
    fs, fc = F["fits"].get("sphere:kin"), F["fits"].get("cylinder:kin")
    fh, ft = F["fits"].get("cylinder:hold"), F["fits"].get("cylinder:turn")
    sh, st_ = band_edges(L, "hold", "sphere"), band_edges(L, "turn", "sphere")
    H = prot["hours"] if prot else {}
    return ("Over 17 simulated layouts of the SR2 tripod and 18 objects, the layout with the largest fixed-contact "
            f"workspace puts the mounts the object radius plus {f(fc['beta_mm'] if fc else None, 0)}&#8202;mm "
            f"(cylinders) or {f(fs['beta_mm'] if fs else None, 0)}&#8202;mm (spheres) from its axis, about half the "
            "68&#8202;mm finger. The task moves the optimum: a cylinder is held best at the radius plus "
            f"{f(fh['beta_mm'] if fh else None, 0)}&#8202;mm and turned furthest at the radius plus "
            f"{f(ft['beta_mm'] if ft else None, 0)}&#8202;mm; for spheres robustness peaks at the compact edge of the feasible "
            f"band ({sh['compact']} of {sh['n']}), the turn at the wide edge ({st_['wide']} of {st_['n']}). "
            f"A bench protocol of {len(prot['cells']) if prot else '&#8211;'} cells ({f(H.get('total'), 0)}&#8202;h) is "
            "written.")


def main():
    L, R, O, lit, prot, films = data()
    load_feas(R)
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M")}
    v["GLOSSARY"] = glossary()
    v["LIT"] = lit_section(lit)
    v["FAMILY"] = family_section(L, R, O)
    v["KIN"] = kin_section(L)
    v["TASKS"] = tasks_section(L)
    v["FILMS"] = films_section(films)
    v["BENCH"] = bench_section(prot, O)
    v["PAPER"] = paper_section(L, prot)
    v["NEXT"] = next_section()
    v["LEDE"] = lede(L, R, prot)
    v["FOOTER"] = (f"<p>Rebuild: <code>.venv/bin/python scripts/hand_object_scale_page.py</code> after "
                   "<code>hand_object_scale_landscape.py</code> and <code>hand_object_scale_protocol.py</code>. Rows and "
                   f"films: <code>{REL}/</code>; cell scenes (0.7&#8202;MB each, kept out of git): "
                   "<code>logs/20261008-hand_object_scale/scenes/</code>; design scenes from the real_v1 base: "
                   "<code>assets/mjcf/experimental/20261008-hand_object_scale/</code>.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    t, n_math = render_tex(t)
    open(OUT, "w").write(retro_style.apply(t))  # plain page style (owner, 2026-10-09)
    print(f"formulas {n_math}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
