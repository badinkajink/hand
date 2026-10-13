#!/usr/bin/env python3
"""Page: in-hand turn of cylinders on the scale-study layouts under HOM control and sampling MPC (2026-10-13).

Owner, 2026-10-13, on Bayesian optimization of MPC parameters (TurboMPC, arXiv 2606.24039) and success guided sampling
(arXiv 2610.12465): "let's spin out these experiments ... please make sure to include video renders and beautiful
figures to understand these methods and their relative performance". The first experiment is a second controller for
the hand-object scale study, whose bench plan picks its cells from HOM's turn scores.

Data (docs/experiments/20261013-scale_turn_controllers/): hom_postoff/ and hom_postoff_servo/ (HOM without the post,
scripts/hand_object_scale_sim.py run --set post_off=1), mpc_servo/ (scripts/hand_object_scale_mpc.py run),
post_traces.json (scripts/hand_object_scale_post_probe.py), media/ (scripts/hand_object_scale_mpc.py film). HOM with the
post: docs/experiments/20261008-hand_object_scale/turn.jsonl (1 N m plant) and
docs/experiments/20261011-bench_plan/servo_plant/turn.jsonl (servo plant); bench cells from its plan.json.

    .venv/bin/python scripts/scale_turn_controllers_page.py
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import statistics
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261013-scale_turn_controllers")
OUT = os.path.join(D, "20261013-scale_turn_hom_mpc.html")
TPL = os.path.join(ROOT, "scripts/scale_turn_controllers_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
SCALE_DIR = os.path.join(ROOT, "docs/experiments/20261008-hand_object_scale")
BENCH_DIR = os.path.join(ROOT, "docs/experiments/20261011-bench_plan")
SCALE_PAGE = "../20261008-hand_object_scale/20261008-finger_spacing_object_size.html"
BENCH_PAGE = "../20261011-bench_plan/20261011-bench_plan_finger_spacing.html"
SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"
DIAMS = (10.0, 16.0, 25.0)
HS = "&#8202;"
FIG, TAB, REFS, ALG = [0], [0], {}, [0]
C_MPC, C_HOM, C_POST = "var(--s1)", "var(--s2)", "var(--ref)"
SHAPE = {10.0: "circle", 16.0: "square", 25.0: "diamond"}
EXTRA_CSS = """
.tw th{white-space:normal !important;vertical-align:bottom;text-transform:none}
.tw td:first-child{min-width:7em}
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
figure.films{display:grid;grid-template-columns:minmax(0,1fr);gap:14px}
figure.films video{width:100%;height:auto}
figure.alg{margin:1.2em 0;padding:.6em 1em;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
figure.alg figcaption{margin:0 0 .3em}
figure.diagram > svg{width:100%;height:auto}
.eq{margin:.8em 0;text-align:center}
td.hi{font-weight:bold}
""" + texsvg.BLOCK_CSS + "\n"


# ------------------------------------------------------------------------------------------ helpers

def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def data_uri(path, mime):
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def figure(body, caption, legend=None, key=None, cls="diagram"):
    FIG[0] += 1
    if key:
        REFS[f"fig:{key}"] = FIG[0]
    lg = P._legend_html(legend) if legend else ""
    return f'<figure class="{cls}">{body}{lg}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def table(head, body, caption, note=None, text_cols=(0,), key=None):
    TAB[0] += 1
    if key:
        REFS[f"tab:{key}"] = TAB[0]
    cls = lambda i: "" if i in text_cols else " class=num"  # noqa: E731
    out = [f"<p class='tcap'><b>Table&#160;{TAB[0]}.</b> {caption}</p>",
           "<div class='tw'><table><thead><tr>" + "".join(f"<th{cls(i)}>{h}</th>" for i, h in enumerate(head)) + "</tr></thead><tbody>"]
    for r in body:
        out.append("<tr>" + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table></div>")
    if note:
        out.append(f"<p class='tnote'>{note}</p>")
    return "".join(out)


def eq(tex, n=None):
    no = f'<span class="eqno">({n})</span>' if n is not None else ""
    return f'<div class="eq">\\[{tex}\\]{no}</div>'


ALG_RE = re.compile(r"\\begin\{algorithmic\}.*?\\end\{algorithmic\}", re.S)


def alg(caption, body):
    ALG[0] += 1
    return (f'<figure class="alg"><figcaption><b>Algorithm&#160;{ALG[0]}.</b> {caption}</figcaption>'
            f"\\begin{{algorithmic}}[1]\n{body}\n\\end{{algorithmic}}</figure>")


def render_tex(t):
    blocks = ALG_RE.findall(t)
    t = ALG_RE.sub(lambda m: "\x00ALG\x00", t)
    maths = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = texsvg.render(maths + [(b, "block") for b in blocks], cache_path=TEX_CACHE, scale=P.TEX_SCALE)
    it = iter(svgs[:len(maths)])
    t = P.TEX_RE.sub(lambda m: next(it), t)
    ib = iter(svgs[len(maths):])
    return re.sub("\x00ALG\x00", lambda m: next(ib), t), len(maths), len(blocks)


def deg(x, nd=0):
    return f"{x:.{nd}f}&#176;"


def xs_of(t):
    return 100.0 - 60.0 * t


def med(v):
    return statistics.median(v) if v else float("nan")


# ------------------------------------------------------------------------------------------ data

def rows_ok(path, servo_only=False):
    out = []
    for r in jl(path):
        if r.get("status") != "ok" or r.get("model", "pads") != "pads":
            continue
        if servo_only and not (r["shape"] == "cylinder" and r["d_mm"] in DIAMS):
            continue
        out.append(r)
    return out


def cell_scores(rows):
    """(shape, d, t) -> {cand, score, seeds, rows}: the landscape's rule, the candidate with the largest median."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["shape"], float(r["d_mm"]), float(r["t"]))][r["cand"]].append(r)
    out = {}
    for k, cc in by.items():
        ci = max(sorted(cc), key=lambda c: med([x["held_turn_deg"] for x in cc[c]]))
        rr = sorted(cc[ci], key=lambda x: x["seed"])
        out[k] = {"cand": ci, "score": med([x["held_turn_deg"] for x in rr]), "seeds": [x["held_turn_deg"] for x in rr],
                  "rows": rr}
    return out


class Data:
    def __init__(self):
        self.r_post1 = rows_ok(os.path.join(SCALE_DIR, "turn.jsonl"))
        self.r_nopost1 = rows_ok(os.path.join(D, "hom_postoff/turn.jsonl"))
        self.r_post = rows_ok(os.path.join(BENCH_DIR, "servo_plant/turn.jsonl"), servo_only=True)
        self.r_nopost = rows_ok(os.path.join(D, "hom_postoff_servo/turn.jsonl"), servo_only=True)
        self.r_mpc = rows_ok(os.path.join(D, "mpc_servo/turn.jsonl"), servo_only=True)
        self.post1, self.nopost1 = cell_scores(self.r_post1), cell_scores(self.r_nopost1)
        self.post, self.nopost, self.mpc = cell_scores(self.r_post), cell_scores(self.r_nopost), cell_scores(self.r_mpc)
        self.plan = json.load(open(os.path.join(BENCH_DIR, "plan.json")))
        self.traces = json.load(open(os.path.join(D, "post_traces.json"))) if os.path.exists(os.path.join(D, "post_traces.json")) else []
        self.films = jl(os.path.join(D, "media/films.jsonl"))
        tr = os.path.join(D, "media/films_traces.json")
        self.film_traces = json.load(open(tr)) if os.path.exists(tr) else {}
        # cells with all three scores on the servo plant
        self.cells = sorted(k for k in self.mpc if k in self.nopost and k in self.post)

    def swing(self, rows):
        """Held turns of at least 60 deg, and those that went more than 15 deg past HOM's governor freeze."""
        big = [r for r in rows if r["held"] and r["held_turn_deg"] >= 60]
        sw = [r for r in big if r.get("frozen_deg") is not None and r["turn_end_deg"] - r["frozen_deg"] > 15]
        return big, sw


# ------------------------------------------------------------------------------------------ figures

def svg_landscape(X: Data):
    """Cell score against thumb-pair separation, one panel per diameter, three series."""
    W, H = 900, 380
    out = P._svg_open(W, H, "Held turn against thumb-pair separation for three cylinder diameters and three controllers")
    pw, ph, x0, y0, gap = 238, 230, 70, 70, 46
    band = defaultdict(set)
    for c in X.plan["cells"]:
        if c.get("in_band"):
            band[float(c["d_mm"])].add(float(c["t"]))
    for i, d in enumerate(DIAMS):
        px = x0 + i * (pw + gap)
        fx, fy = P._panel(out, px, y0, pw, ph, (35, 130), (0, 120), [40, 60, 80, 100, 120], [0, 30, 60, 90, 120],
                          "Thumb-pair separation (mm)" if i == 1 else "", "Held turn (deg)" if i == 0 else "")
        out.append(f'<text x="{px + pw / 2}" y="{y0 - 40}" text-anchor="middle" style="fill:var(--ink);font-weight:bold">'
                   f'{d:g}{HS}mm cylinder</text>')
        for t in sorted(band[d]):
            xx = fx(xs_of(t))
            out.append(f'<rect x="{xx - 6:.1f}" y="{y0}" width="12" height="{ph}" style="fill:rgba(0,0,0,0.07)">'
                       f'<title>Bench-plan band cell, t = {t:+.3f}</title></rect>')
        out.append(f'<line x1="{px}" x2="{px + pw}" y1="{fy(60):.1f}" y2="{fy(60):.1f}" '
                   f'style="stroke:var(--ink3);stroke-dasharray:2 4"/>')
        for src, col, dash, shape in ((X.post, C_POST, True, "cross"), (X.nopost, C_HOM, False, "square"),
                                      (X.mpc, C_MPC, False, "circle")):
            ks = sorted((k for k in src if k[0] == "cylinder" and k[1] == d), key=lambda k: xs_of(k[2]))
            pts = [(xs_of(k[2]), src[k]["score"]) for k in ks]
            P._path(out, fx, fy, pts, col, dashed=dash, width=2.4 if col != C_POST else 1.8)
            for k in ks:
                v = src[k]
                lab = {C_POST: "HOM, post", C_HOM: "HOM, no post", C_MPC: "MPC"}[col]
                P._marker(out, fx(xs_of(k[2])), fy(v["score"]), col, shape=shape, r=4.4,
                          title=f"{lab}: {d:g} mm, t {k[2]:+.3f}, x_s {xs_of(k[2]):.1f} mm, score {v['score']:.1f} deg, "
                                f"seeds {', '.join(f'{s:.0f}' for s in v['seeds'])}, grasp {v['cand']}")
    out.append("</svg>")
    leg = [("MPC", C_MPC, False, "circle"), ("HOM, post removed at 1.3 s", C_HOM, False, "square"),
           ("HOM, post present (bench plan)", C_POST, True, "cross")]
    return "".join(out), leg


def svg_pairs(X: Data):
    """MPC against HOM (no post), one point per cell, shape by diameter."""
    W, H = 900, 420
    out = P._svg_open(W, H, "MPC cell score against HOM cell score on the servo plant")
    fx, fy = P._panel(out, 300, 40, 320, 320, (0, 120), (0, 120), [0, 30, 60, 90, 120], [0, 30, 60, 90, 120],
                      "HOM held turn, post removed (deg)", "MPC held turn (deg)")
    out.append(f'<path d="M{fx(0):.1f},{fy(0):.1f} L{fx(120):.1f},{fy(120):.1f}" style="stroke:var(--ink3);stroke-dasharray:5 4"/>')
    out.append(f'<rect x="{fx(0):.1f}" y="{fy(120):.1f}" width="{fx(60) - fx(0):.1f}" height="{fy(60) - fy(120):.1f}" '
               f'style="fill:rgba(190,117,20,0.08)"/>')
    out.append(f'<text x="{fx(2):.1f}" y="{fy(116):.1f}" style="fill:var(--ink3)">MPC at least 60&#176;, HOM below</text>')
    for k in X.cells:
        a, b = X.nopost[k]["score"], X.mpc[k]["score"]
        P._marker(out, fx(a), fy(b), C_MPC, shape=SHAPE[k[1]], r=5.2,
                  title=f"{k[1]:g} mm, t {k[2]:+.3f} (x_s {xs_of(k[2]):.1f} mm): HOM {a:.1f} deg, MPC {b:.1f} deg")
    out.append("</svg>")
    leg = [(f"{d:g}{HS}mm cylinder", C_MPC, False, SHAPE[d]) for d in DIAMS]
    return "".join(out), leg


def svg_post_scatter(X: Data):
    """Cell score with the post against without it, both plants."""
    W, H = 900, 400
    out = P._svg_open(W, H, "HOM cell score without the post against with it")
    panels = (("1 N m plant, 8 October cells", X.post1, X.nopost1, 80), ("Servo plant, bench-plan cells", X.post, X.nopost, 520))
    for title, a, b, px in panels:
        fx, fy = P._panel(out, px, 50, 290, 290, (0, 120), (0, 120), [0, 30, 60, 90, 120], [0, 30, 60, 90, 120],
                          "Held turn, post present (deg)", "Held turn, post removed (deg)")
        out.append(f'<text x="{px + 145}" y="22" text-anchor="middle" style="fill:var(--ink);font-weight:bold">{title}</text>')
        out.append(f'<path d="M{fx(0):.1f},{fy(0):.1f} L{fx(120):.1f},{fy(120):.1f}" style="stroke:var(--ink3);stroke-dasharray:5 4"/>')
        for k in sorted(set(a) & set(b)):
            sph = k[0] == "sphere"
            P._marker(out, fx(a[k]["score"]), fy(b[k]["score"]), C_POST if sph else C_HOM, shape="circle", hollow=sph,
                      r=4.0, title=f"{k[0]} {k[1]:g} mm, t {k[2]:+.3f}: post {a[k]['score']:.1f}, no post {b[k]['score']:.1f} deg")
    out.append("</svg>")
    leg = [("Cylinder", C_HOM, False, "circle"), ("Sphere", C_POST, False, "circle")]
    return "".join(out), leg


def svg_post_trace(X: Data, d=16.0, t=0.625, seed=0):
    """Turn angle with and without the post, and the post's force, on one bench-plan cell."""
    ent = {e["post_off"]: e for e in X.traces if e["d_mm"] == d and e["t"] == t and e["seed"] == seed and e["plant"] == SERVO}
    if 0.0 not in ent or 1.0 not in ent:
        return None, None
    W, H = 900, 470
    out = P._svg_open(W, H, f"Turn angle and post force against time, {d:g} mm cylinder")
    fx, fy = P._panel(out, 80, 40, 740, 230, (0.8, 6.3), (-10, 130), [1, 2, 3, 4, 5, 6], [0, 30, 60, 90, 120],
                      "", "Turn angle (deg)")
    gx, gy = P._panel(out, 80, 330, 740, 90, (0.8, 6.3), (0, 2.5), [1, 2, 3, 4, 5, 6], [0, 1, 2],
                      "Time (s)", "Post force on the cylinder (N)", yfmt="{:g}")
    for po, col, dash in ((1.0, C_HOM, False), (0.0, C_POST, True)):
        tr = [x for x in ent[po]["trace"] if x["phase"] != "grip"]
        P._path(out, fx, fy, [(x["t"], x["th"]) for x in tr], col, dashed=dash, width=2.4)
        fr = ent[po]["result"].get("frozen_deg")
        if fr is not None:
            tf = next((x["t"] for x in tr if x["phase"] == "hold"), None)
            if tf is not None:
                P._marker(out, fx(tf), fy(fr), col, shape="square", r=5, title=f"governor freeze at {fr:.1f} deg")
    tr0 = [x for x in ent[0.0]["trace"] if x["phase"] != "grip"]
    P._path(out, gx, gy, [(x["t"], min(2.5, x["post_N"])) for x in tr0], C_POST, width=1.8)
    out.append(f'<line x1="{fx(1.3):.1f}" x2="{fx(1.3):.1f}" y1="40" y2="420" style="stroke:var(--ink3);stroke-dasharray:3 4"/>'
               f'<text x="{fx(1.3) + 6:.1f}" y="{fy(105):.1f}" style="fill:var(--ink3)">Squeeze ends; post removed</text>'
               f'<text x="{fx(1.3) + 6:.1f}" y="{fy(105) + 17:.1f}" style="fill:var(--ink3)">in the no-post run</text>')
    out.append("</svg>")
    r0, r1 = ent[0.0]["result"], ent[1.0]["result"]
    leg = [(f"Post present: held {r0['held']}, end {r0['turn_end_deg']:.0f}&#176;", C_POST, True, None),
           (f"Post removed: held {r1['held']}, end {r1['turn_end_deg']:.0f}&#176;", C_HOM, False, None),
           ("Governor freeze", "var(--ink2)", False, "square")]
    return "".join(out), leg


def svg_film_traces(X: Data):
    """Turn angle against time for HOM and MPC on each filmed cell."""
    cells = [f for f in X.film_traces.values()]
    if not cells:
        return None, None
    n = len(cells)
    W = 900
    pw, ph = (900 - 70 - 30 * (n - 1) - 20) / n, 220
    H = 330
    out = P._svg_open(W, H, "Turn angle against time for HOM and MPC on the filmed cells")
    for i, c in enumerate(cells):
        px = 70 + i * (pw + 30)
        fx, fy = P._panel(out, px, 50, pw, ph, (0.8, 6.3), (-30, 130), [1, 3, 5], [0, 45, 90],
                          "Time (s)" if i == n // 2 else "", "Turn angle (deg)" if i == 0 else "")
        out.append(f'<text x="{px + pw / 2:.1f}" y="24" text-anchor="middle" style="fill:var(--ink);font-weight:bold">'
                   f'{c["d_mm"]:g}{HS}mm, x{HS}<tspan font-style="italic">s</tspan> {c["x_sep_mm"]:g}{HS}mm</text>')
        out.append(f'<line x1="{px}" x2="{px + pw:.1f}" y1="{fy(90):.1f}" y2="{fy(90):.1f}" style="stroke:var(--ink3);stroke-dasharray:2 4"/>')
        for ctrl, col in (("HOM", C_HOM), ("MPC", C_MPC)):
            tr = c.get(ctrl) or []
            P._path(out, fx, fy, [(x["t"], x["th"]) for x in tr if x["t"] >= 0.8], col, width=2.2)
    out.append("</svg>")
    leg = [("MPC", C_MPC, False, None), ("HOM, post removed at 1.3 s", C_HOM, False, None)]
    return "".join(out), leg


def film(stem, caption, key):
    p = os.path.join(D, "media", f"{stem}.mp4")
    poster = os.path.join(D, "media", f"{stem}.jpg")
    if not os.path.exists(p):
        return ""
    v = (f'<video controls muted loop playsinline preload="metadata" poster="{data_uri(poster, "image/jpeg")}" '
         f'src="{data_uri(p, "video/mp4")}"></video>')
    return figure(v, caption, key=key, cls="films")


# ------------------------------------------------------------------------------------------ sections

GLOSSARY = [
    ("Layout \\(t\\), thumb-pair separation \\(x_s\\)", "The scale study&#8217;s diagonal family of symmetric tripods: the thumb sits "
     "opposite the index&#8211;middle pair at separation \\(x_s = 100 - 60\\,t\\) mm, the pair&#8217;s fingers \\(y_s = 110 - 60\\,t\\) mm "
     "apart, \\(t\\) from &#8722;1 (wide) to 1 (compact)."),
    ("Cell", "One layout and one cylinder: 10, 16 or 25&#8202;mm in diameter, 100&#8202;mm long, 25&#8202;g, lying along the pair&#8217;s "
     "axis on a support post."),
    ("Grasp", "One of up to nine palm heights and straddles that the scale study evaluates per cell; the grip command presses each pad "
     "with 4&#8202;N at rest."),
    ("Turn angle \\(\\theta\\)", "Rotation of the cylinder&#8217;s marked axis from its direction after the grip, in the vertical plane "
     "through it, in degrees; 90&#176; stands the cylinder upright."),
    ("Held, held turn", "Held: at the end of the run the cylinder is less than 20&#8202;mm below its height after the grip and at least "
     "two fingers each press on it with at least its weight, 0.245&#8202;N. Held turn: \\(\\theta\\) at the end when held, else 0."),
    ("Cell score", "The median held turn over three placements (the cylinder shifted and yawed by seeds 0, 1 and 2) at the grasp whose "
     "median is largest; the rule of the scale study&#8217;s landscape. MPC runs on the grasp that this rule picks for HOM without the post."),
    ("Servo plant", "The finger servo model of the bench plan: position gain 0.72&#8202;N&#8202;m/rad, torque limit 0.18&#8202;N&#8202;m, "
     "the measured SCS0009 stall torque. The 1&#8202;N&#8202;m plant of the 8 October study has gain 4&#8202;N&#8202;m/rad and limit "
     "1&#8202;N&#8202;m."),
    ("Post", "The 3&#8202;mm-radius cylinder (at most 0.4 of the object radius) the object rests on before the grip. Post present: it "
     "collides for the whole run, as in the rows of 8 and 11 October. Post removed: its collisions end at 1.3&#8202;s, when HOM&#8217;s "
     "squeeze ends and its turn begins."),
    ("Governor freeze, swing past the freeze", "HOM stops advancing its turn reference when a pad&#8217;s force falls below 30&#8202;% of "
     "2&#8202;N or the turn lags the reference by 6&#176; for three ticks, and holds that angle. Swing past the freeze: \\(\\theta\\) at "
     "the end minus \\(\\theta\\) at the freeze."),
    ("Bench-plan band cell", "A cell the bench plan of 11 October predicts to turn: at least 60&#176; in two of three placements on the "
     "servo plant with the post present."),
    ("Wall time per run", "Seconds of wall clock for one run from the grip to the end, on one core for HOM and with six rollout threads "
     "for MPC; the run simulates 6.3&#8202;s at most."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def question(X: Data):
    n_band = sum(1 for c in X.plan["cells"] if c.get("in_band"))
    return (f"<p>The hand-object scale study measures how far the SR2 hand can turn a cylinder in hand as the finger layout and the "
            f"cylinder diameter vary (<a href=\"{SCALE_PAGE}\">8 October page</a>), and the bench plan picks {len(X.plan['cells'])} cells, "
            f"{n_band} of them predicted to turn, from those scores (<a href=\"{BENCH_PAGE}\">11 October page</a>). Every turn score so far "
            "comes from one controller, the relative contact-velocity (HOM) controller of Wang, Oh and Pollard. A ridge in the landscape "
            "describes the hand when a controller of another kind finds it at the same layouts; this page runs sampling model-predictive "
            "control (MPC) on the same cells, grasps, plant, placements and held test.</p>"
            "<p>The servo plant is the comparison plant, as in the bench plan. Each run grips for 0.8&#8202;s with the cell&#8217;s grip "
            "command, then hands control to the controller until 6.3&#8202;s or until HOM&#8217;s hold ends. The support post under the "
            "cylinder touches it during HOM&#8217;s turns (next section), so HOM was rerun on both plants with the post removed at "
            "1.3&#8202;s, and MPC uses that protocol.</p>")


def controllers(X: Data):
    s = ("<p><b>HOM</b> (<code>scripts/hom_turn3.py</code>, at its defaults) commands a rotation of the cylinder about the pinch axis "
         "along a reference that advances at 30&#176;/s toward 90&#176;. Every 10&#8202;ms it computes, finger by finger, the joint "
         "velocities whose contact-point velocities match the commanded object twist plus a normal-force correction toward 2&#8202;N, "
         "by bounded least squares within the joint limits, and integrates them into the servo targets. Its governor stops the reference "
         "when a pad loses force or the turn lags (Terms). One run costs about half a second of one core.</p>"
         "<p><b>MPC</b> (<code>scripts/hand_object_scale_mpc.py</code>) is predictive sampling, the planner of MuJoCo MPC (Howell et al. "
         "2022). Its plan is a piecewise-linear spline of the nine servo targets with three knots over a 0.3&#8202;s horizon. Every "
         "20&#8202;ms it rolls out 32 plans from the plant&#8217;s current state on the plant&#8217;s own model without the post, the "
         "current plan shifted in time and 31 noisy copies, and keeps the cheapest (Algorithm&#160;1). The cost of a rollout, sampled "
         "every 10&#8202;ms at times \\(t_j\\), is Eq.&#160;(1) with the running cost of Eq.&#160;(2):</p>"
         + eq(r"J = \sum_{j=1}^{n-1} c(t_j) + w_{\mathrm{term}}\, c(t_n) + w_{\Delta}\sum_{m=1}^{2}\lVert K_{m+1}-K_m\rVert^2", 1)
         + eq(r"c = w_{\mathrm{tilt}}\,(1-u_z) + w_{\mathrm{pos}}\frac{\lVert p-p_{\mathrm{ref}}\rVert^2}{(10\ \mathrm{mm})^2} + "
              r"w_{\mathrm{drop}}\,\mathbf{1}[\,p_z < z_0 - 15\ \mathrm{mm}\,]", 2)
         + "<p>where \\(u\\) is the cylinder&#8217;s marked axis (\\(u_z = 1\\) upright), \\(p\\) its centre, \\(p_{\\mathrm{ref}}\\) the "
         "centre after the grip raised 5&#8202;mm (HOM&#8217;s lift), \\(z_0\\) the height after the grip and \\(K_m\\) the knots, with "
         "\\(w_{\\mathrm{tilt}} = w_{\\mathrm{pos}} = w_{\\Delta} = 1\\), \\(w_{\\mathrm{drop}} = w_{\\mathrm{term}} = 10\\), noise "
         "\\(\\sigma = 0.08\\) rad. The cost has no contact or finger term; holding comes from the position and drop terms, since a "
         "released cylinder falls the 15&#8202;mm of the drop term in 55&#8202;ms, a fifth of one horizon.</p>")
    s += alg("Predictive sampling, one MPC run",
             r"\Require plant state \(x\) after the grip; knots \(K \in \mathbb{R}^{3\times 9}\) all equal to the grip command" "\n"
             r"\For{\(t_k = 0.8\ \mathrm{s},\ 0.82\ \mathrm{s},\ \dots,\ 6.28\ \mathrm{s}\)}" "\n"
             r"\State shift \(K\) to the knot times \(t_k + \{0,\ 0.15,\ 0.3\}\) s" "\n"
             r"\State \(K_1 \gets K\);\ \ \(K_i \gets \mathrm{clip}(K + \varepsilon_i)\), \(\varepsilon_i \sim \mathcal{N}(0, \sigma^2 I)\), \(i = 2, \dots, 32\)" "\n"
             r"\State roll out every \(K_i\) for 0.3 s from \(x(t_k)\) on the model without the post" "\n"
             r"\State \(K \gets \arg\min_i J(K_i)\) \Comment{Eqs. (1), (2)}" "\n"
             r"\State apply \(K\) to the plant until \(t_k + 0.02\) s" "\n"
             r"\EndFor")
    return s


def post_section(X: Data):
    big, sw = X.swing(X.r_post)
    big0, sw0 = X.swing(X.r_nopost)
    bigw, sww = X.swing([r for r in X.r_post1 if r["shape"] == "cylinder"])
    dif1 = [abs(X.nopost1[k]["score"] - X.post1[k]["score"]) for k in X.post1 if k in X.nopost1 and k[0] == "cylinder"]
    difs = [abs(X.nopost1[k]["score"] - X.post1[k]["score"]) for k in X.post1 if k in X.nopost1 and k[0] == "sphere"]
    difv = [abs(X.nopost[k]["score"] - X.post[k]["score"]) for k in X.post if k in X.nopost]
    n_cyl20 = sum(1 for x in dif1 if x > 20)
    s = (f"<p>The post stands under the cylinder&#8217;s centre and reaches to its lower surface. HOM lifts the cylinder by 5&#8202;mm "
         f"at most and turns it about the contacts&#8217; centroid, near the centre, so a cylinder turned by more than a few tens of "
         f"degrees comes down onto the post. On the servo plant, {len(big)} of the {len(X.r_post)} HOM runs with the post were held at "
         f"60&#176; or more, and in {len(sw)} of them the cylinder had turned more than 15&#176; past the angle where the governor "
         f"stopped the reference: the turn stopped, the cylinder swung on about the pinch under its weight, and the post caught it. "
         f"On the 1&#8202;N&#8202;m plant the same holds for {len(sww)} of {len(bigw)}. Figure&#160;[[fig:posttrace]] shows one such run "
         f"on a bench-plan band cell.</p>")
    svg, leg = svg_post_trace(X)
    if svg:
        s += figure(svg, "HOM on the 16&#8202;mm cylinder at \\(t = 0.625\\) (\\(x_s\\) = 62.5&#8202;mm), the bench plan&#8217;s only "
                    "16&#8202;mm band cell, placement 0, servo plant. Top: turn angle with the post present for the whole run and with it "
                    "removed at 1.3&#8202;s; squares mark the governor&#8217;s freeze. Bottom: the post&#8217;s normal force on the "
                    "cylinder in the run with the post (clipped at 2.5&#8202;N).", leg, key="posttrace")
    s += (f"<p>Removing the post at 1.3&#8202;s leaves the spheres&#8217; scores within {max(difs):.0f}&#176; of their values with it, "
          f"and moves cylinder scores by a median {med(dif1):.0f}&#176; and up to {max(dif1):.0f}&#176; on the 1&#8202;N&#8202;m plant "
          f"({n_cyl20} of {len(dif1)} cells by more than 20&#176;), and by a median {med(difv):.0f}&#176; and up to {max(difv):.0f}&#176; "
          f"on the servo plant (Figure&#160;[[fig:postscatter]]). Scores move both ways: the post blocks some turns and catches others. "
          f"Without the post, {len(sw0)} of the {len(big0)} servo-plant runs held at 60&#176; or more still swung past the freeze, held "
          "by the fingers alone at the end.</p>")
    svg, leg = svg_post_scatter(X)
    s += figure(svg, "HOM cell scores without the post against with it. Left: the 8 October cells on the 1&#8202;N&#8202;m plant, "
                "cylinders and spheres. Right: the bench plan&#8217;s 10, 16 and 25&#8202;mm cylinders on the servo plant. Points on the "
                "dashed line are unchanged.", leg, key="postscatter")
    return s


def landscape_section(X: Data):
    ks = X.cells
    n = len(ks)
    hom = [X.nopost[k]["score"] for k in ks]
    mpc = [X.mpc[k]["score"] for k in ks]
    post = [X.post[k]["score"] for k in ks]
    better = sum(1 for a, b in zip(hom, mpc) if b > a + 10)
    worse = sum(1 for a, b in zip(hom, mpc) if b < a - 10)
    m60 = sum(1 for b in mpc if b >= 60)
    h60 = sum(1 for a in hom if a >= 60)
    p60 = sum(1 for a in post if a >= 60)
    s = (f"<p>Over the {n} servo-plant cells that have all three scores, MPC&#8217;s median cell score is {deg(med(mpc))} against "
         f"{deg(med(hom))} for HOM without the post and {deg(med(post))} with it. MPC scores at least 60&#176; in {m60} cells, HOM in "
         f"{h60} without the post and {p60} with it. MPC exceeds HOM (no post) by more than 10&#176; in {better} cells and falls more "
         f"than 10&#176; below it in {worse} (Figures&#160;[[fig:landscape]] and [[fig:pairs]]).</p>")
    svg, leg = svg_landscape(X)
    s += figure(svg, "Cell score against thumb-pair separation on the servo plant, one panel per cylinder diameter; compact layouts on "
                "the left. Grey columns: the bench plan&#8217;s band cells. Dotted line: 60&#176;. Hover a point for its placements and "
                "grasp.", leg, key="landscape")
    svg, leg = svg_pairs(X)
    s += figure(svg, "MPC cell score against HOM&#8217;s (post removed at 1.3&#8202;s), one point per cell; shape by diameter. "
                "Points above the dashed line turn further under MPC.", leg, key="pairs")
    # bench-plan cells
    body = []
    for c in sorted(X.plan["cells"], key=lambda c: (c["d_mm"], -c["t"])):
        k = ("cylinder", float(c["d_mm"]), float(c["t"]))
        f = lambda src: "&#8211;" if k not in src else f"{src[k]['score']:.0f}"  # noqa: E731
        sd = lambda src: "&#8211;" if k not in src else " / ".join(f"{v:.0f}" for v in src[k]["seeds"])  # noqa: E731
        swn = 0
        if k in X.post:
            swn = sum(1 for r in X.post[k]["rows"] if r.get("frozen_deg") is not None and r["turn_end_deg"] - r["frozen_deg"] > 15)
        body.append([f"{c['d_mm']:g}{HS}mm", f"{c['t']:+.3f}", f"{xs_of(c['t']):.1f}", "yes" if c.get("in_band") else "",
                     f(X.post), str(swn), f(X.nopost), f(X.mpc), sd(X.mpc)])
    s += table(["Cylinder", "Layout \\(t\\)", "\\(x_s\\), mm", "Band cell", "HOM, post, deg", "Placements past the freeze",
                "HOM, no post, deg", "MPC, deg", "MPC placements, deg"], body,
               "The bench plan&#8217;s 15 cells: cell scores under each controller and protocol on the servo plant.",
               note="Placements past the freeze: of the three HOM runs with the post at the scoring grasp, those whose cylinder turned more "
                    "than 15&#176; past the governor&#8217;s freeze. Each column picks its own grasp by the cell-score rule; MPC runs on "
                    "the grasp of the no-post HOM column.", text_cols=(0, 3), key="bench")
    return s


def films_section(X: Data):
    rows = defaultdict(dict)
    for f in X.films:
        rows[(f["d_mm"], f["t"], f["seed"])][f["controller"]] = f
    out = ("<p>Each film shows HOM (left, post removed at 1.3&#8202;s) and MPC (right) on the same grasp and placement, at real "
           "speed; pads touching the cylinder are drawn red and the orange rod marks the cylinder&#8217;s axis. The titles give the "
           "layout; the label gives the time and the turn angle.</p>")
    for (d, t, seed), pair in sorted(rows.items()):
        if "HOM" not in pair or "MPC" not in pair:
            continue
        h, m = pair["HOM"], pair["MPC"]
        stem = f"hom_mpc_cyl{d:05.1f}_{h['tag']}_s{seed}"
        cap = (f"{d:g}{HS}mm cylinder at \\(t = {t:+.3f}\\) (\\(x_s\\) = {xs_of(t):.1f}&#8202;mm), grasp {h['cand']}, placement {seed}. "
               f"HOM: {'held' if h['held'] else 'not held'}, {h['turn_end_deg']:.0f}&#176; at the end. MPC: "
               f"{'held' if m['held'] else 'not held'}, {m['turn_end_deg']:.0f}&#176; at the end, "
               f"{m['fingers_loaded_end']} fingers loaded.")
        out += film(stem, cap, key=f"film{int(d)}{'m' if t < 0 else ''}{abs(int(round(1000 * t)))}")
    svg, leg = svg_film_traces(X)
    if svg:
        out += figure(svg, "Turn angle against time in the filmed runs. Dotted line: 90&#176;.", leg, key="filmtraces")
    return out


def cost_section(X: Data):
    wh = [r["wall_s"] for r in X.r_nopost]
    wm = [r["wall_s"] for r in X.r_mpc]
    body = [["HOM", "100", "&#8211;", "1", f"{med(wh):.2f}", f"{max(wh):.1f}"],
            ["MPC", "50", "32 &#215; 0.3&#8202;s", "6", f"{med(wm):.0f}", f"{max(wm):.0f}"]]
    frac = med([r["plan_s"] / r["wall_s"] for r in X.r_mpc if r.get("wall_s")])
    s = (f"<p>MPC costs about {med(wm) / max(1e-6, med(wh)):.0f} times HOM&#8217;s wall time per run (Table&#160;[[tab:cost]]). Each "
         "replanning rolls out 9.6&#8202;s of simulated time, 32 plans of 0.3&#8202;s, with the 1&#8202;mm sphere pads at 1&#8202;ms steps; "
         f"planning takes {100 * frac:.0f}&#8202;% of MPC&#8217;s wall time. The full servo-plant grid, "
         f"{len(X.r_mpc)} runs, took {sum(wm) / 3600:.1f}&#8202;h of six-thread time.</p>")
    s += table(["Controller", "Control rate, Hz", "Rollouts per replanning", "Threads", "Wall time per run, median, s",
                "Wall time per run, longest, s"], body, "Cost of the two controllers on the servo plant.", key="cost")
    return s


def limitations():
    items = [
        "MPC plans on the plant&#8217;s own model, so its scores are what the hand reaches with an exact model of itself; the bench "
        "hand runs open loop or on a model with error.",
        "MPC runs at one setting (32 plans, 0.3&#8202;s horizon, the weights of Eq.&#160;(2)) and on one grasp per cell, the one HOM "
        "scores best; other grasps can score higher under MPC.",
        "Three placements per cell; a cell score of the median of three changes by tens of degrees when one placement changes.",
        "Only the servo plant and the 10, 16 and 25&#8202;mm cylinders carry MPC runs; the 1&#8202;N&#8202;m plant enters through HOM only.",
        "The no-post protocol removes the post when HOM&#8217;s turn begins; on the bench the post stays where it is unless the arm lifts "
        "the hand.",
    ]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def open_items():
    items = [
        ("Planner with model error.", "Plan on the point-contact block model (<code>pt</code>) while the plant runs the pads, and on the "
         "1&#8202;N&#8202;m model while the plant runs the servo plant, by a planner-model option in "
         "<code>scripts/hand_object_scale_mpc.py</code>. MPC&#8217;s ridge describes the hand if the cells at 60&#176; or more stay "
         "within one layout step of this page&#8217;s."),
        ("MPC on every grasp of the band cells.", "Run <code>hand_object_scale_mpc.py run</code> over all candidates of the 15 bench "
         "cells (about 1,200 runs, 8&#8202;h on 12 threads); the bench plan&#8217;s grasp choice stands if the best MPC grasp is the "
         "grasp the plan exports."),
        ("A learned controller over the cells with success guided sampling.", "Train one policy over the 34 cells in mjlab, the mounts "
         "set per world through <code>body_pos</code> and the cylinder through <code>dr.geom_size</code>, sampling cells by the held "
         "turn fraction \\(q = \\mathbf{1}[\\mathrm{held}]\\,\\mathrm{clip}(\\theta_{\\mathrm{end}}/90^\\circ, 0, 1)\\) with the "
         "success-guided weight of Zhang et al. (target 0.5); compare with uniform sampling at two seeds of 20&#8202;M steps each. The "
         "learned ridge agrees with MPC&#8217;s if their cells at 60&#176; or more coincide in at least two of three diameters."),
        ("Bench plan.", "Replace the band cells chosen from swings caught by the post with the cells MPC turns, and export MPC&#8217;s "
         "servo targets as open-loop plans for <code>scripts/real_v1_trajectory_clearance.py</code>; the replay is worth bench time "
         "if the open-loop replay in simulation keeps at least 60&#176; held in two of three placements."),
        ("Scale-study landscape without the post.", "Rebuild <code>landscape.json</code> of 8 October from "
         "<code>hom_postoff/turn.jsonl</code> and report the moved turn ridges on the scale-study page."),
    ]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def lede(X: Data):
    ks = X.cells
    hom = [X.nopost[k]["score"] for k in ks]
    mpc = [X.mpc[k]["score"] for k in ks]
    big, sw = X.swing(X.r_post)
    m60 = sum(1 for b in mpc if b >= 60)
    h60 = sum(1 for a in hom if a >= 60)
    wm = [r["wall_s"] for r in X.r_mpc]
    return (f"On the servo plant, sampling MPC turned and held the cylinder to a median {deg(med(mpc))} over {len(ks)} layout and "
            f"diameter cells, against {deg(med(hom))} for the HOM controller, and reached 60&#176; in {m60} cells against {h60}. "
            f"HOM&#8217;s high scores are mostly swings after its governor stops: {len(sw)} of its {len(big)} runs held at 60&#176; or more "
            f"turned more than 15&#176; past the stop, and the support post caught the cylinder. Removing the post when the turn begins "
            f"moves cylinder scores by up to {max(abs(X.nopost[k]['score'] - X.post[k]['score']) for k in X.post if k in X.nopost):.0f}&#176;. "
            f"MPC costs {med(wm):.0f}&#8202;s of six-thread time per run, HOM half a second.")


def byline():
    return ("HOM turns <code>scripts/hand_object_scale_sim.py</code> (<code>--set post_off=1</code>) &#183; MPC "
            "<code>scripts/hand_object_scale_mpc.py</code> &#183; post traces <code>scripts/hand_object_scale_post_probe.py</code> "
            "&#183; builder <code>scripts/scale_turn_controllers_page.py</code> &#183; rows "
            "<code>docs/experiments/20261013-scale_turn_controllers/</code>")


def main():
    X = Data()
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "GLOSSARY": glossary()}
    v["QUESTION"] = question(X)
    v["CONTROLLERS"] = controllers(X)
    v["POST"] = post_section(X)
    v["LANDSCAPE"] = landscape_section(X)
    v["FILMS"] = films_section(X)
    v["COST"] = cost_section(X)
    v["LIMITS"] = limitations()
    v["OPEN"] = open_items()
    v["LEDE"] = lede(X)
    v["FOOTER"] = ("<p>Rebuild: <code>.venv/bin/python scripts/scale_turn_controllers_page.py</code>. Rows: "
                   "<code>hom_postoff/</code>, <code>hom_postoff_servo/</code>, <code>mpc_servo/</code> (<code>turn.jsonl</code>, one "
                   "line per run, MPC rows with their 50&#8202;ms traces), <code>post_traces.json</code>, <code>media/films.jsonl</code> "
                   "and <code>media/films_traces.json</code> in <code>docs/experiments/20261013-scale_turn_controllers/</code>. HOM rows "
                   "with the post: <code>docs/experiments/20261008-hand_object_scale/turn.jsonl</code> and "
                   "<code>docs/experiments/20261011-bench_plan/servo_plant/turn.jsonl</code>.</p>")
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
    t, n, nb = render_tex(t)
    t = retro_style.apply(t)
    open(OUT, "w").write(t)
    print(f"formulas {n}, algorithms {nb}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
