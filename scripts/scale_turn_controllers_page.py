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
C_MPC, C_HOM, C_POST, C_MPC3 = "var(--s1)", "var(--s2)", "var(--ref)", "var(--s3)"
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
        self.r_mpc13 = rows_ok(os.path.join(D, "mpc_servo_post13/turn.jsonl"), servo_only=True)
        self.r_mpc3 = rows_ok(os.path.join(D, "mpc3_servo/turn.jsonl"), servo_only=True)
        self.post1, self.nopost1 = cell_scores(self.r_post1), cell_scores(self.r_nopost1)
        self.post, self.nopost, self.mpc = cell_scores(self.r_post), cell_scores(self.r_nopost), cell_scores(self.r_mpc)
        self.mpc13 = cell_scores(self.r_mpc13)
        self.mpc3 = cell_scores(self.r_mpc3)
        self.plan = json.load(open(os.path.join(BENCH_DIR, "plan.json")))
        self.traces = json.load(open(os.path.join(D, "post_traces.json"))) if os.path.exists(os.path.join(D, "post_traces.json")) else []
        self.films = jl(os.path.join(D, "media/films.jsonl"))
        tr = os.path.join(D, "media/films_traces.json")
        self.film_traces = json.load(open(tr)) if os.path.exists(tr) else {}
        # cells with all three scores on the servo plant
        self.cells = sorted(k for k in self.mpc if k in self.nopost and k in self.post)

    @staticmethod
    def peak_rate(r):
        tr = r.get("trace") or []
        rates = [abs(b["th"] - a["th"]) / (b["t"] - a["t"]) for a, b in zip(tr, tr[1:]) if b["t"] > a["t"]]
        return max(rates) if rates else None

    def stats(self, rows):
        held = [r for r in rows if r["held"]]
        pr = [x for x in (self.peak_rate(r) for r in rows) if x is not None]
        return {"n": len(rows), "held": len(held), "h60": sum(1 for r in held if r["held_turn_deg"] >= 60),
                "h60_3": sum(1 for r in held if r["held_turn_deg"] >= 60 and r["fingers_loaded_end"] == 3),
                "two": sum(1 for r in held if r["fingers_loaded_end"] == 2),
                "rate": med(pr) if pr else None, "wall": med([r["wall_s"] for r in rows]) if rows else None}

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
                                      (X.mpc3, C_MPC3, False, "diamond"), (X.mpc, C_MPC, False, "circle")):
            ks = sorted((k for k in src if k[0] == "cylinder" and k[1] == d), key=lambda k: xs_of(k[2]))
            pts = [(xs_of(k[2]), src[k]["score"]) for k in ks]
            P._path(out, fx, fy, pts, col, dashed=dash, width=2.4 if col != C_POST else 1.8)
            for k in ks:
                v = src[k]
                lab = {C_POST: "HOM, post", C_HOM: "HOM, no post", C_MPC: "MPC", C_MPC3: "MPC, three pads"}[col]
                P._marker(out, fx(xs_of(k[2])), fy(v["score"]), col, shape=shape, r=4.4,
                          title=f"{lab}: {d:g} mm, t {k[2]:+.3f}, x_s {xs_of(k[2]):.1f} mm, score {v['score']:.1f} deg, "
                                f"seeds {', '.join(f'{s:.0f}' for s in v['seeds'])}, grasp {v['cand']}")
    out.append("</svg>")
    leg = [("MPC", C_MPC, False, "circle"), ("MPC, three pads, 60&#176;/s (placement 0)", C_MPC3, False, "diamond"),
           ("HOM, post removed", C_HOM, False, "square"), ("HOM, post present (bench plan)", C_POST, True, "cross")]
    return "".join(out), leg


def svg_pairs(X: Data):
    """MPC (two variants) against HOM without the post, one point per cell, shape by diameter."""
    W, H = 900, 420
    out = P._svg_open(W, H, "MPC cell scores against HOM cell scores on the servo plant")
    panels = (("MPC, swing allowed", X.mpc, C_MPC, 90), ("MPC, three pads, 60&#176;/s, placement 0", X.mpc3, C_MPC3, 530))
    for title, src, col, px in panels:
        fx, fy = P._panel(out, px, 50, 300, 300, (0, 120), (0, 120), [0, 30, 60, 90, 120], [0, 30, 60, 90, 120],
                          "HOM held turn, post removed (deg)", "MPC held turn (deg)")
        out.append(f'<text x="{px + 150}" y="22" text-anchor="middle" style="fill:var(--ink);font-weight:bold">{title}</text>')
        out.append(f'<path d="M{fx(0):.1f},{fy(0):.1f} L{fx(120):.1f},{fy(120):.1f}" style="stroke:var(--ink3);stroke-dasharray:5 4"/>')
        out.append(f'<rect x="{fx(0):.1f}" y="{fy(120):.1f}" width="{fx(60) - fx(0):.1f}" height="{fy(60) - fy(120):.1f}" '
                   f'style="fill:rgba(0,0,0,0.05)"/>')
        for k in X.cells:
            if k not in src:
                continue
            a_, b_ = X.nopost[k]["score"], src[k]["score"]
            P._marker(out, fx(a_), fy(b_), col, shape=SHAPE[k[1]], r=5.0,
                      title=f"{k[1]:g} mm, t {k[2]:+.3f} (x_s {xs_of(k[2]):.1f} mm): HOM {a_:.1f} deg, MPC {b_:.1f} deg")
    out.append("</svg>")
    leg = [(f"{d:g}{HS}mm cylinder", "var(--ink2)", False, SHAPE[d]) for d in DIAMS]
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
    """Turn angle against time for each controller on the filmed cells."""
    cells = sorted(X.film_traces.values(), key=lambda c: (c["t"] < 0, c["d_mm"], c["t"]))   # the films' order
    if not cells:
        return None, None
    n = len(cells)
    W, H = 900, 330
    pw, ph = (W - 70 - 26 * (n - 1) - 20) / n, 220
    out = P._svg_open(W, H, "Turn angle against time for HOM and MPC on the filmed cells")
    for i, c in enumerate(cells):
        px = 70 + i * (pw + 26)
        fx, fy = P._panel(out, px, 50, pw, ph, (0.8, 6.3), (-30, 130), [1, 3, 5], [0, 45, 90],
                          "Time (s)" if i == n // 2 else "", "Turn angle (deg)" if i == 0 else "")
        out.append(f'<text x="{px + pw / 2:.1f}" y="24" text-anchor="middle" style="fill:var(--ink);font-weight:bold">'
                   f'{c["d_mm"]:g}{HS}mm, {c["x_sep_mm"]:g}{HS}mm</text>')
        out.append(f'<line x1="{px}" x2="{px + pw:.1f}" y1="{fy(90):.1f}" y2="{fy(90):.1f}" style="stroke:var(--ink3);stroke-dasharray:2 4"/>')
        for ctrl, col in (("HOM", C_HOM), ("MPC3", C_MPC3), ("MPC", C_MPC)):
            tr = c.get(ctrl) or []
            P._path(out, fx, fy, [(x["t"], x["th"]) for x in tr if x["t"] >= 0.8], col, width=2.2)
    out.append("</svg>")
    leg = [("MPC", C_MPC, False, None), ("MPC, three pads, 60&#176;/s", C_MPC3, False, None),
           ("HOM, post removed", C_HOM, False, None)]
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
    ("Three-pad held turn", "A held turn that ends with all three fingers each pressing on the cylinder with at least its weight. "
     "The program&#8217;s criterion for a reorientation (owner, 16 September) is a controlled turn that ends on three pads."),
    ("Pinch-and-swing", "A turn in which one finger leaves the cylinder and the cylinder swings under its weight about the axis "
     "through the two remaining pads, braked by the pads&#8217; friction torque about that axis."),
    ("Peak turn rate", "The largest change of \\(\\theta\\) between the 50&#8202;ms samples of a run&#8217;s trace, in &#176;/s."),
    ("Cell score", "The median held turn over three placements (the cylinder shifted and yawed by seeds 0, 1 and 2) at the grasp whose "
     "median is largest; the rule of the scale study&#8217;s landscape. MPC runs on the grasp that this rule picks for HOM without the post."),
    ("Fingertip", "The printed TPU tip (a block with 2.7&#8202;mm fillets) as 1&#8202;mm sphere pads: 0.75&#8202;mm spheres on a "
     "1&#8202;mm grid, 1,059 per fingertip, each a soft contact with the cylinder; the default contact model of this program "
     "(<a href=\"../../overviews/contact_model.html\">fingertip contact overview</a>). MPC plans on the same model."),
    ("Servo plant", "The finger servo model of the bench plan: position gain 0.72&#8202;N&#8202;m/rad, torque limit 0.18&#8202;N&#8202;m, "
     "the measured SCS0009 stall torque. The 1&#8202;N&#8202;m plant of the 8 October study has gain 4&#8202;N&#8202;m/rad and limit "
     "1&#8202;N&#8202;m."),
    ("Post", "The 3&#8202;mm-radius cylinder (at most 0.4 of the object radius) the object rests on before the grip. Post present: it "
     "collides for the whole run, as in the rows of 8 and 11 October. Post removed: its collisions end when the controller starts "
     "turning, at 1.3&#8202;s for HOM (after its squeeze) and at 0.8&#8202;s for MPC."),
    ("Governor freeze, swing past the freeze", "HOM stops advancing its turn reference when a pad&#8217;s force falls below 30&#8202;% of "
     "2&#8202;N or the turn lags the reference by 6&#176; for three ticks, and holds that angle. Swing past the freeze: \\(\\theta\\) at "
     "the end minus \\(\\theta\\) at the freeze."),
    ("Bench-plan band cell", "A cell the bench plan of 11 October predicts to turn: at least 60&#176; in two of three placements on the "
     "servo plant with the post present."),
    ("Wall time per run", "Seconds of wall clock for one run from the grip to the end, on one core for HOM and with six rollout threads "
     "for MPC; the run simulates 6.3&#8202;s at most."),
    ("Placement 0", "The controlled MPC ran one placement per cell (seed 0); its cell score is that run&#8217;s held turn."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def question(X: Data):
    n_band = sum(1 for c in X.plan["cells"] if c.get("in_band"))
    return (f"<p>The hand-object scale study measures how far the SR2 hand can turn a cylinder in hand as the finger layout and the "
            f"cylinder diameter vary (<a href=\"{SCALE_PAGE}\">8 October page</a>), and the bench plan picks {len(X.plan['cells'])} cells, "
            f"{n_band} of them predicted to turn, from those scores (<a href=\"{BENCH_PAGE}\">11 October page</a>). Every turn score so far "
            "comes from one controller, the relative contact-velocity (HOM) controller of Wang, Oh and Pollard. A ridge in the landscape "
            "describes the hand when a controller of another kind finds it at the same layouts. This page runs sampling model-predictive "
            "control (MPC) on the same cells, grasps, plant, placements and held test, in two forms: with a cost that asks only for an "
            "upright, held cylinder, and with a cost that also keeps all three fingers on the cylinder and the turn below 60&#176;/s, "
            "the program&#8217;s criterion for a reorientation (Terms).</p>"
            "<p>The servo plant is the comparison plant, as in the bench plan. Each run grips for 0.8&#8202;s with the cell&#8217;s grip "
            "command, then hands control to the controller until 6.3&#8202;s or until HOM&#8217;s hold ends. The support post under the "
            "cylinder touches it during HOM&#8217;s turns (next section), so each controller here loses the post when it starts "
            "turning: HOM at 1.3&#8202;s, after its squeeze, and MPC at 0.8&#8202;s, when it takes over.</p>")


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
    s += ("<p><b>MPC with three pads</b> adds two running terms to Eq.&#160;(2) and searches wider: 64 plans over a 0.5&#8202;s "
          "horizon. A MuJoCo contact sensor in the planner&#8217;s model gives each finger&#8217;s net force \\(F_f\\) on the "
          "cylinder, and \\(\\omega\\) is the cylinder&#8217;s angular speed:</p>"
          + eq(r"c_3 = c + w_{\mathrm{c}}\sum_{f}\mathrm{clip}\!\left(\frac{F_{\min}-F_f}{F_{\min}},\,0,\,1\right) + "
               r"w_{\omega}\,\frac{\max(0,\ \omega-\omega_{\max})^2}{\omega_{\max}^2}", 3)
          + "<p>with \\(w_{\\mathrm{c}} = w_{\\omega} = 1\\), \\(F_{\\min} = 0.5\\) N (twice the cylinder&#8217;s weight) and "
          "\\(\\omega_{\\max} = 60\\)&#176;/s, twice HOM&#8217;s reference rate.</p>")
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
    k3 = [k for k in ks if k in X.mpc3]
    mpc3 = [X.mpc3[k]["score"] for k in k3]
    hom3 = [X.nopost[k]["score"] for k in k3]
    m60, h60, p60 = (sum(1 for x in v if x >= 60) for v in (mpc, hom, post))
    st_m, st_h, st_3, st_p = X.stats(X.r_mpc), X.stats(X.r_nopost), X.stats(X.r_mpc3), X.stats(X.r_post)
    s = (f"<p>Over the {n} servo-plant cells, MPC&#8217;s median cell score is {deg(med(mpc))} against {deg(med(hom))} for HOM without "
         f"the post and {deg(med(post))} with it, and MPC scores at least 60&#176; in {m60} cells, HOM in {h60} without the post and "
         f"{p60} with it (Figures&#160;[[fig:landscape]] and [[fig:pairs]]). MPC gets there by pinch-and-swing: it lets one finger "
         f"leave the cylinder, which swings about the remaining pinch at a median peak of {st_m['rate']:.0f}&#176;/s and is held near "
         f"upright; {st_m['two']} of its {st_m['held']} held runs end on two fingers, and {st_m['h60_3']} of its {st_m['n']} runs end "
         f"held at 60&#176; or more on three pads (Table&#160;[[tab:runs]], films in the next section). The hand can swing and hold "
         f"the cylinder at nearly every layout; the narrow bands of the bench plan come from HOM.</p>")
    if k3:
        c60 = sum(1 for x in mpc3 if x >= 60)
        better = sum(1 for a, b in zip(hom3, mpc3) if b > a + 10)
        n3 = sum(1 for r in X.r_mpc3 if r["held"] and r["fingers_loaded_end"] == 3)
        s += (f"<p>With the three-pad and turn-rate terms, MPC holds the cylinder on three pads in {n3} "
              f"of {st_3['n']} runs at placement 0, at a median peak rate of {st_3['rate']:.0f}&#176;/s, and turns it to a median "
              f"{deg(med(mpc3))} over {len(k3)} cells, against {deg(med(hom3))} for HOM on the same cells. It reaches 60&#176; in {c60} "
              f"cells and exceeds HOM by more than 10&#176; in {better}. The 60&#176; cells are "
              + ", ".join(f"the {k[1]:g}{HS}mm cylinder at {xs_of(k[2]):g}{HS}mm ({X.mpc3[k]['score']:.0f}&#176;)"
                          for k in sorted((k for k in k3 if X.mpc3[k]["score"] >= 60), key=lambda k: (k[1], -k[2])))
              + "; every other cell stays between "
              + f"{min(x for x in mpc3 if x < 60):.0f} and {max(x for x in mpc3 if x < 60):.0f}&#176;.</p>")
    svg, leg = svg_landscape(X)
    s += figure(svg, "Cell score against thumb-pair separation on the servo plant, one panel per cylinder diameter; compact layouts on "
                "the left. Grey columns: the bench plan&#8217;s band cells. Dotted line: 60&#176;. Hover a point for its placements and "
                "grasp.", leg, key="landscape")
    svg, leg = svg_pairs(X)
    s += figure(svg, "MPC cell scores against HOM&#8217;s (post removed), one point per cell; shape by diameter. Left: MPC whose cost "
                "asks only for an upright, held cylinder. Right: MPC that also keeps three pads on the cylinder and the turn below "
                "60&#176;/s, at placement 0. Points above the dashed line turn further under MPC; the grey square marks MPC at 60&#176; "
                "or more where HOM stays below.", leg, key="pairs")
    rows = []
    for lab, st_ in (("HOM, post present (bench plan)", st_p), ("HOM, post removed", st_h), ("MPC", st_m),
                     ("MPC, three pads, 60&#176;/s", st_3)):
        if not st_["n"]:
            continue
        rows.append([lab, str(st_["n"]), str(st_["held"]), str(st_["h60"]), str(st_["h60_3"]), str(st_["two"]),
                     "&#8211;" if st_["rate"] is None else f"{st_['rate']:.0f}"])
    s += table(["Controller", "Runs", "Held", "Held at 60&#176; or more", "Of these on three pads", "Held runs ending on two fingers",
                "Median peak turn rate, &#176;/s"], rows,
               "Runs of each controller on the servo plant: all grasps for HOM (three placements each), the scoring grasp for MPC "
               "(three placements; the three-pad variant one).", note="HOM&#8217;s rows store no trace, so their turn rate is not "
               "available; its reference advances at 30&#176;/s.", key="runs")
    both = [k for k in X.mpc13 if k in X.mpc and len(X.mpc13[k]["seeds"]) == 3]
    if both:
        dd = sorted(X.mpc13[k]["score"] - X.mpc[k]["score"] for k in both)
        close = sum(1 for x in dd if abs(x) <= 5)
        far = [x for x in dd if abs(x) > 5]
        s += (f"<p>A first MPC grid kept the post in the plant until 1.3&#8202;s, as for HOM, and MPC turned the cylinder 20&#8211;58&#176; "
              f"before then, in some runs with under 1&#8202;N of finger force. Of the {len(both)} cells it completed, {close} score within "
              f"5&#176; of this page&#8217;s and {len(far)} differ by " + ", ".join(f"{x:+.0f}&#176;".replace("-", "&#8722;") for x in far)
              + " (<code>mpc_servo_post13/</code>).</p>")
    body = []
    for c in sorted(X.plan["cells"], key=lambda c: (c["d_mm"], -c["t"])):
        k = ("cylinder", float(c["d_mm"]), float(c["t"]))
        f = lambda src: "&#8211;" if k not in src else f"{src[k]['score']:.0f}"  # noqa: E731
        swn = 0
        if k in X.post:
            swn = sum(1 for r in X.post[k]["rows"] if r.get("frozen_deg") is not None and r["turn_end_deg"] - r["frozen_deg"] > 15)
        three = "&#8211;"
        if k in X.mpc3:
            r = X.mpc3[k]["rows"][0]
            three = f"{r['held_turn_deg']:.0f} ({r['fingers_loaded_end']})"
        body.append([f"{c['d_mm']:g}{HS}mm", f"{c['t']:+.3f}", f"{xs_of(c['t']):.1f}", "yes" if c.get("in_band") else "",
                     f(X.post), str(swn), f(X.nopost), f(X.mpc), three])
    s += table(["Cylinder", "Layout \\(t\\)", "\\(x_s\\), mm", "Band cell", "HOM, post, deg", "Placements past the freeze",
                "HOM, no post, deg", "MPC, deg", "MPC three pads, deg (fingers)"], body,
               "The bench plan&#8217;s 15 cells: cell scores under each controller and protocol on the servo plant.",
               note="Placements past the freeze: of the three HOM runs with the post at the scoring grasp, those whose cylinder turned more "
                    "than 15&#176; past the governor&#8217;s freeze. Each HOM column picks its own grasp by the cell-score rule; both MPC "
                    "columns run on the grasp of the no-post HOM column; the three-pad column is placement 0, with the fingers loaded "
                    "at the end.", text_cols=(0, 3), key="bench")
    return s


def films_section(X: Data):
    rows = defaultdict(dict)
    for f in X.films:
        rows[(f["d_mm"], f["t"], f["seed"], f["tag"])][f["controller"]] = f
    names = {"HOM": "HOM", "MPC": "MPC", "MPC3": "MPC with three pads"}
    out = ("<p>Each film shows the three controllers on the same grasp and placement, at real speed: HOM (post removed at "
           "1.3&#8202;s), MPC, and MPC with three pads. Pads touching the cylinder are drawn red, the orange rod marks the "
           "cylinder&#8217;s axis, and the post is drawn only while it collides. The label gives the time and the turn angle.</p>")
    order = sorted(rows, key=lambda k: (k[1] < 0, k[0], k[1]))
    for (d, t, seed, tag) in order:
        pair = rows[(d, t, seed, tag)]
        ctrls = [c for c in ("HOM", "MPC", "MPC3") if c in pair]
        if len(ctrls) < 2:
            continue
        stem = f"{'_'.join(c.lower() for c in ctrls)}_cyl{d:05.1f}_{tag}_s{seed}"
        parts = []
        for c in ctrls:
            r = pair[c]
            parts.append(f"{names[c]}: {'held' if r['held'] else 'not held'} at {r['turn_end_deg']:.0f}&#176; on "
                         f"{r['fingers_loaded_end']} fingers")
        cap = (f"{d:g}{HS}mm cylinder at \\(t = {t:+.3f}\\) (\\(x_s\\) = {xs_of(t):.1f}&#8202;mm), grasp {pair[ctrls[0]]['cand']}, "
               f"placement {seed}. " + "; ".join(parts) + ".")
        out += film(stem, cap, key=f"film{int(d)}{'m' if t < 0 else ''}{abs(int(round(1000 * t)))}")
    svg, leg = svg_film_traces(X)
    if svg:
        out += figure(svg, "Turn angle against time in the filmed runs; panels titled by diameter and thumb-pair separation. "
                      "Dotted line: 90&#176;.", leg, key="filmtraces")
    return out


def cost_section(X: Data):
    wh = [r["wall_s"] for r in X.r_nopost]
    wm = [r["wall_s"] for r in X.r_mpc]
    w3 = [r["wall_s"] for r in X.r_mpc3]
    frac = med([r["plan_s"] / r["wall_s"] for r in X.r_mpc if r.get("wall_s")])
    body = [["HOM", "100", "&#8211;", "1", f"{med(wh):.2f}", f"{max(wh):.1f}"],
            ["MPC", "50", "32 &#215; 0.3&#8202;s", "6", f"{med(wm):.0f}", f"{max(wm):.0f}"]]
    if w3:
        body.append(["MPC, three pads", "50", "64 &#215; 0.5&#8202;s", "6", f"{med(w3):.0f}", f"{max(w3):.0f}"])
    s = (f"<p>MPC costs about {med(wm) / max(1e-6, med(wh)):.0f} times HOM&#8217;s wall time per run and the three-pad variant "
         f"{(med(w3) if w3 else float('nan')) / max(1e-6, med(wh)):.0f} times (Table&#160;[[tab:cost]]). Each replanning of the first "
         "rolls out 9.6&#8202;s of simulated time, 32 plans of 0.3&#8202;s, and the second 32&#8202;s, with the 1&#8202;mm sphere pads at "
         f"1&#8202;ms steps; planning takes {100 * frac:.0f}&#8202;% of MPC&#8217;s wall time. The MPC grid, {len(X.r_mpc)} runs, took "
         f"{sum(wm) / 3600:.1f}&#8202;h of six-thread time, and the three-pad grid, {len(X.r_mpc3)} runs, {sum(w3) / 3600:.1f}&#8202;h.</p>")
    s += table(["Controller", "Control rate, Hz", "Rollouts per replanning", "Threads", "Wall time per run, median, s",
                "Wall time per run, longest, s"], body, "Cost of the controllers on the servo plant.", key="cost")
    return s


def limitations(X: Data):
    items = [
        "MPC plans on the plant&#8217;s own model, so its scores are what the hand reaches with an exact model of itself; the bench "
        "hand runs open loop or on a model with error.",
        "MPC runs at two settings and on one grasp per cell, the one HOM scores best without the post; other grasps can score higher "
        "under MPC. The three-pad variant ran one placement per cell.",
        "Three placements per cell; a cell score of the median of three changes by tens of degrees when one placement changes.",
        "Only the servo plant and the 10, 16 and 25&#8202;mm cylinders carry MPC runs; the 1&#8202;N&#8202;m plant enters through HOM only.",
        "The no-post protocol removes the post when each controller starts turning; on the bench the post stays where it is unless the "
        "arm lifts the hand.",
        "The held test asks for two fingers at the cylinder&#8217;s weight and counts turns past 90&#176; at face value, as in the scale "
        f"study; {sum(1 for r in X.r_mpc if r['held'] and r['held_turn_deg'] > 100)} of the {sum(1 for r in X.r_mpc if r['held'])} "
        "held MPC runs end past 100&#176;.",
    ]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def open_items():
    items = [
        ("Three-pad MPC at every placement.", "Run <code>scripts/hand_object_scale_mpc.py run --set w_contact=1 w_rate=1 nsample=64 "
         "horizon=0.5 --seeds 1 2</code> on the 34 cells (68 runs, about 3.6&#8202;h on 12 threads). The controlled landscape of this "
         "page holds if each cell&#8217;s median of three stays within 15&#176; of its placement-0 score."),
        ("Planner with model error.", "Plan on the point-contact block model (<code>pt</code>) while the plant runs the pads, and on the "
         "1&#8202;N&#8202;m model while the plant runs the servo plant, by a planner-model option in "
         "<code>scripts/hand_object_scale_mpc.py</code>. MPC&#8217;s landscape describes the hand if the cells at 60&#176; or more stay "
         "within one layout step of this page&#8217;s."),
        ("A learned controller over the cells with success guided sampling.", "Train one mjlab policy over the 34 cells. The layouts "
         "differ only in the positions of the three mount bodies, which mjlab sets per world through <code>body_pos</code> (with "
         "<code>set_const_0</code>), and the cylinder through <code>geom_size</code>; one reset event draws a cell from the "
         "success-guided weights of Zhang et al. (target 0.5) on the three-pad turn fraction "
         "\\(q = \\mathbf{1}[\\text{held on three pads}]\\,\\mathrm{clip}(\\theta_{\\mathrm{end}}/90^\\circ, 0, 1)\\) and "
         "writes the mounts, the palm offset, the finger pose and per-env grip targets for the closure. The scene must come from the "
         "plant design scenes (the cell scenes have no palm joints), and the post must become a body that moves away per world, since "
         "contact masks are shared across worlds. Compare with uniform sampling at two seeds of 20&#8202;M steps; the learned "
         "landscape agrees with the three-pad MPC if their cells at 60&#176; or more coincide for at least two of the three diameters."),
        ("Bench plan.", "Choose the bench cells from the three-pad MPC scores and export its servo targets as open-loop plans for "
         "<code>scripts/real_v1_trajectory_clearance.py</code>; a cell is worth bench time if the open-loop replay in simulation "
         "keeps at least 60&#176; on three pads in two of three placements."),
        ("Scale-study landscape without the post.", "Rebuild <code>landscape.json</code> of 8 October from "
         "<code>hom_postoff/turn.jsonl</code> and report the moved turn ridges on the scale-study page."),
    ]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def lede(X: Data):
    ks = X.cells
    mpc = [X.mpc[k]["score"] for k in ks]
    big, sw = X.swing(X.r_post)
    st_m = X.stats(X.r_mpc)
    s = (f"On the servo plant, HOM&#8217;s turn scores in the hand-object scale study record its governor and the support post more "
         f"than the hand: in {len(sw)} of its {len(big)} runs held at 60&#176; or more, the cylinder swung on past the governor&#8217;s "
         f"stop and came to rest on the post. Sampling MPC on the same cells holds the cylinder at 60&#176; or more in "
         f"{sum(1 for x in mpc if x >= 60)} of {len(ks)} cells (HOM {sum(1 for k in ks if X.nopost[k]['score'] >= 60)}), by "
         f"pinch-and-swing: {st_m['two']} of its {st_m['held']} held runs end on two fingers, at a median peak rate of "
         f"{st_m['rate']:.0f}&#176;/s.")
    k3 = [k for k in ks if k in X.mpc3]
    if k3:
        m3 = [X.mpc3[k]["score"] for k in k3]
        s += (f" Required to keep three pads on and turn below 60&#176;/s, MPC turns the cylinder to a median {deg(med(m3))} "
              f"(HOM {deg(med([X.nopost[k]['score'] for k in k3]))}) and reaches 60&#176; in {sum(1 for x in m3 if x >= 60)} "
              f"cells.")
    return s


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
    v["LIMITS"] = limitations(X)
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
