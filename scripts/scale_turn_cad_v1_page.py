#!/usr/bin/env python3
"""Page: the scale-turn study of 13-14 October repeated on the CAD-model hand (2026-10-14).

Owner, 2026-10-14: "can you now try to rerun this work, but with the newest CAD-faithful hand:
assets/mjcf/experimental/20261013-cad_v1? don't delete anything, if anything just replicate the page but with the updated
CAD results". The CAD hand (scripts/build_cad_v1_finger.py) keeps real_v1's joints, mounts and actuators; its MCP link
spaces the MCP and PIP axes 30.80 mm apart against 20.75 mm, its links collide through 151 spheres per finger, its masses
are the printed parts', and its TPU block is 21.41 mm long. Every run here sets SCALE_HAND=cad_v1
(scripts/hand_object_scale_kin.py, hand_object_scale_sim.py), which swaps those fingers into the study's scenes.

Data (docs/experiments/20261014-scale_turn_cad_v1/): kin/reach.json, hom_postoff_servo/ and hom_post_servo/
(hand_object_scale_sim.py run), mpc_servo/ and mpc3_servo/ (hand_object_scale_mpc.py run), planners/ (turn.jsonl,
timing.json, gpu_check.jsonl, media/), media/ (three-panel films), post_traces.json. The real_v1 page's rows
(docs/experiments/20261013-scale_turn_controllers/) enter the comparison section.

    .venv/bin/python scripts/scale_turn_cad_v1_page.py
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

D = os.path.join(ROOT, "docs/experiments/20261014-scale_turn_cad_v1")
V1 = os.path.join(ROOT, "docs/experiments/20261013-scale_turn_controllers")    # the real_v1 study, for comparison
OUT = os.path.join(D, "20261014-scale_turn_cad_v1.html")
TPL = os.path.join(ROOT, "scripts/scale_turn_cad_v1_page.template.html")
V1_PAGE = "../20261013-scale_turn_controllers/20261013-scale_turn_hom_mpc.html"
CAD_PAGE = "../20261013-cad_v1_finger/20261013-cad_v1_finger_model.html"
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
PL = os.path.join(D, "planners")
PLANNERS = [  # configuration of scripts/hand_object_scale_planners.py, label, colour, marker
    ("hom", "HOM", "#4A7FC4", "square"),
    ("ps", "Predictive sampling", "#BE7514", "circle"),
    ("ps_pin", "Pinned predictive sampling", "#6B3E0A", "circle"),
    ("mppi_pin", "Pinned MPPI", "#8A8A8A", "cross"),
    ("cem_pin", "Pinned CEM", "#8E44AD", "diamond"),
    ("icem_pin", "Pinned iCEM", "#111111", "diamond"),
    ("anneal_pin", "Pinned annealed sampling", "#C0392B", "square"),
    ("anneal_gpu", "Pinned annealed sampling on the GPU", "#E0789A", "square"),
    ("icem_gpu", "Pinned iCEM on the GPU", "#1C9CB0", "diamond"),
    ("ilqr", "iLQR", "#555555", "cross"),
]
PL_LAB = {c: lab for c, lab, _, _ in PLANNERS}
PL_COL = {c: col for c, _, col, _ in PLANNERS}
PL_SHP = {c: shp for c, _, _, shp in PLANNERS}
FILMED = ("hom", "ps", "ps_pin", "icem_pin")
# the icem_gpu rows of 05:20 overflowed MuJoCo-Warp's 64 contact matches per sensor; the page uses the rerun at 256
PL_VARIANT = {"icem_gpu": "mm256"}
C_ICEM = "#111111"           # pinned iCEM in every figure; black, since var(--s3) green is three-pad MPC
JOINTS = [f"{f} {j}" for f in ("Thumb", "Index-finger", "Middle-finger") for j in ("yaw", "MCP", "PIP")]
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


def lc(label):
    """A label inside a sentence: first letter lower case, the rest kept (iCEM, MPPI)."""
    return label[0].lower() + label[1:]


def signed0(x):
    """Whole degrees with a typographic minus and no negative zero."""
    v = round(x)
    return f"&#8722;{-v}" if v < 0 else f"{v}"


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
        self.r_post1, self.r_nopost1 = [], []          # the 1 N m plant is not rerun on the CAD hand
        self.r_post = rows_ok(os.path.join(D, "hom_post_servo/turn.jsonl"), servo_only=True)
        self.r_nopost = rows_ok(os.path.join(D, "hom_postoff_servo/turn.jsonl"), servo_only=True)
        self.r_mpc = rows_ok(os.path.join(D, "mpc_servo/turn.jsonl"), servo_only=True)
        self.r_mpc13 = []
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
        pl_all = jl(os.path.join(PL, "turn.jsonl"))
        self.r_pl = [r for r in pl_all if r.get("status") == "ok" and (r.get("variant") or "") == PL_VARIANT.get(r["config"], "")]
        self.n_overflow = sum(1 for r in pl_all if r["config"] in PL_VARIANT and not r.get("variant"))
        # pinned iCEM at placement 0 on every reachable cell (planners/landscape_icem_pin.jsonl)
        self.r_land = [r for r in jl(os.path.join(PL, "landscape_icem_pin.jsonl")) if r.get("status") == "ok"]
        self.land = {("cylinder", float(r["d_mm"]), float(r["t"])): {"cand": r["cand"], "score": r["held_turn_deg"],
                                                                       "seeds": [r["held_turn_deg"]], "rows": [r]}
                     for r in self.r_land}
        self.pl = defaultdict(list)
        for r in self.r_pl:
            self.pl[r["config"]].append(r)
        self.pl_cfgs = [c for c, *_ in PLANNERS if self.pl.get(c)]
        self.pl_films = jl(os.path.join(PL, "media/films.jsonl"))
        tp = os.path.join(PL, "timing.json")
        self.timing = json.load(open(tp)) if os.path.exists(tp) else None
        self.gpu_check = jl(os.path.join(PL, "gpu_check.jsonl"))
        # cells with all three scores on the servo plant
        self.cells = sorted(k for k in self.mpc if k in self.nopost and k in self.post)
        # the real_v1 study (20261013-scale_turn_controllers), for the comparison section
        self.r_v1_post = rows_ok(os.path.join(BENCH_DIR, "servo_plant/turn.jsonl"), servo_only=True)
        self.v1_nopost = cell_scores(rows_ok(os.path.join(V1, "hom_postoff_servo/turn.jsonl"), servo_only=True))
        self.v1_post = cell_scores(rows_ok(os.path.join(BENCH_DIR, "servo_plant/turn.jsonl"), servo_only=True))
        self.v1_mpc = cell_scores(rows_ok(os.path.join(V1, "mpc_servo/turn.jsonl"), servo_only=True))
        self.v1_mpc3 = cell_scores(rows_ok(os.path.join(V1, "mpc3_servo/turn.jsonl"), servo_only=True))
        self.v1_pl = defaultdict(list)
        for r in jl(os.path.join(V1, "planners/turn.jsonl")):
            if r.get("status") == "ok" and not r.get("variant"):
                self.v1_pl[r["config"]].append(r)
        tp1 = os.path.join(V1, "planners/timing.json")
        self.v1_timing = json.load(open(tp1)) if os.path.exists(tp1) else None
        rp = os.path.join(D, "kin/reach.json")
        self.reach = json.load(open(rp)) if os.path.exists(rp) else None

    @staticmethod
    def peak_rate(r):
        tr = r.get("trace") or []
        rates = [abs(b["th"] - a["th"]) / (b["t"] - a["t"]) for a, b in zip(tr, tr[1:]) if b["t"] > a["t"]]
        return max(rates) if rates else None

    def stats(self, rows):
        held = [r for r in rows if r["held"]]
        pr = [x for x in (r["peak_rate"] if r.get("peak_rate") is not None else self.peak_rate(r) for r in rows)
              if x is not None]
        return {"n": len(rows), "held": len(held), "h60": sum(1 for r in held if r["held_turn_deg"] >= 60),
                "h60_3": sum(1 for r in held if r["held_turn_deg"] >= 60 and r["fingers_loaded_end"] == 3),
                "two": sum(1 for r in held if r["fingers_loaded_end"] == 2),
                "rate": med(pr) if pr else None, "wall": med([r["wall_s"] for r in rows]) if rows else None}

    def pl_stats(self, cfg):
        rows = self.pl.get(cfg, [])
        held = [r for r in rows if r["held"]]
        f = lambda k: med([r[k] for r in rows if r.get(k) is not None])  # noqa: E731
        return {"n": len(rows), "held": len(held), "three_end": sum(1 for r in held if r["fingers_loaded_end"] == 3),
                "three_frac": statistics.mean([r["three_frac"] for r in rows if r.get("three_frac") is not None] or [float("nan")]),
                "breaks": sum(r.get("contact_breaks") or 0 for r in rows), "turn": med([r["held_turn_deg"] for r in rows]),
                "h60": sum(1 for r in held if r["held_turn_deg"] >= 60 and r["fingers_loaded_end"] == 3),
                "rate": f("cmd_rate_rms"), "rev": f("cmd_reversals"), "jit": f("omega_jitter"), "peak": f("peak_rate"),
                "wall": f("wall_s")}

    def pl_row(self, cfg, d, t, seed):
        return next((r for r in self.pl.get(cfg, []) if r["d_mm"] == d and r["t"] == t and r["seed"] == seed), None)

    def swing(self, rows):
        """Held turns of at least 60 deg, and those that went more than 15 deg past HOM's governor freeze."""
        big = [r for r in rows if r["held"] and r["held_turn_deg"] >= 60]
        sw = [r for r in big if r.get("frozen_deg") is not None and r["turn_end_deg"] - r["frozen_deg"] > 15]
        return big, sw


# ------------------------------------------------------------------------------------------ figures

def svg_landscape(X: Data):
    """Cell score against thumb-pair separation, one panel per diameter, five series."""
    W, H = 900, 380
    out = P._svg_open(W, H, "Held turn against thumb-pair separation for three cylinder diameters and five controllers")
    pw, ph, x0, y0, gap = 238, 230, 70, 70, 46
    band = defaultdict(set)
    for c in X.plan["cells"]:
        if c.get("in_band"):
            band[float(c["d_mm"])].add(float(c["t"]))
    labs = {C_POST: "HOM, post", C_HOM: "HOM, no post", C_MPC: "MPC", C_MPC3: "MPC, three pads", C_ICEM: "Pinned iCEM"}
    for i, d in enumerate(DIAMS):
        px = x0 + i * (pw + gap)
        fx, fy = P._panel(out, px, y0, pw, ph, (35, 165), (0, 120), [40, 80, 120, 160], [0, 30, 60, 90, 120],
                          "Thumb-pair separation (mm)" if i == 1 else "", "Held turn (deg)" if i == 0 else "")
        out.append(f'<text x="{px + pw / 2}" y="{y0 - 40}" text-anchor="middle" style="fill:var(--ink);font-weight:bold">'
                   f'{d:g}{HS}mm cylinder</text>')
        for t in sorted(band[d]):
            xx = fx(xs_of(t))
            out.append(f'<rect x="{xx - 6:.1f}" y="{y0}" width="12" height="{ph}" style="fill:rgba(0,0,0,0.07)">'
                       f'<title>real_v1 bench-plan band cell, t = {t:+.3f}</title></rect>')
        out.append(f'<line x1="{px}" x2="{px + pw}" y1="{fy(60):.1f}" y2="{fy(60):.1f}" '
                   f'style="stroke:var(--ink3);stroke-dasharray:2 4"/>')
        for src, col, dash, shape in ((X.post, C_POST, True, "cross"), (X.nopost, C_HOM, False, "square"),
                                      (X.mpc3, C_MPC3, False, "diamond"), (X.mpc, C_MPC, False, "circle"),
                                      (X.land, C_ICEM, False, "diamond")):
            ks = sorted((k for k in src if k[0] == "cylinder" and k[1] == d), key=lambda k: xs_of(k[2]))
            pts = [(xs_of(k[2]), src[k]["score"]) for k in ks]
            P._path(out, fx, fy, pts, col, dashed=dash, width=2.4 if col != C_POST else 1.8)
            for k in ks:
                v = src[k]
                P._marker(out, fx(xs_of(k[2])), fy(v["score"]), col, shape=shape, r=4.4,
                          title=f"{labs[col]}: {d:g} mm, t {k[2]:+.3f}, x_s {xs_of(k[2]):.1f} mm, score {v['score']:.1f} deg, "
                                f"seeds {', '.join(f'{s:.0f}' for s in v['seeds'])}, grasp {v['cand']}")
    out.append("</svg>")
    leg = [("MPC", C_MPC, False, "circle"), ("MPC, three pads, 60&#176;/s (placement 0)", C_MPC3, False, "diamond"),
           ("Pinned iCEM, three pads (placement 0)", C_ICEM, False, "diamond"),
           ("HOM, post removed", C_HOM, False, "square"), ("HOM, post present (bench plan)", C_POST, True, "cross")]
    return "".join(out), leg


def svg_pairs(X: Data):
    """MPC (two variants) and pinned iCEM against HOM without the post, one point per cell, shape by diameter."""
    W, H = 900, 380
    out = P._svg_open(W, H, "MPC and pinned-iCEM cell scores against HOM cell scores on the servo plant")
    panels = (("MPC, swing allowed", X.mpc, C_MPC), ("MPC, three pads, placement 0", X.mpc3, C_MPC3),
              ("Pinned iCEM, placement 0", X.land, C_ICEM))
    pw, gap = 236, 52
    for i, (title, src, col) in enumerate(panels):
        px = 66 + i * (pw + gap)                      # the last panel's 120 tick ends inside the 900 px box
        fx, fy = P._panel(out, px, 50, pw, 236, (0, 120), (0, 120), [0, 30, 60, 90, 120], [0, 30, 60, 90, 120],
                          "HOM held turn, post removed (deg)" if i == 1 else "", "Held turn (deg)" if i == 0 else "")
        out.append(f'<text x="{px + pw / 2:.1f}" y="22" text-anchor="middle" style="fill:var(--ink);font-weight:bold">{title}</text>')
        out.append(f'<path d="M{fx(0):.1f},{fy(0):.1f} L{fx(120):.1f},{fy(120):.1f}" style="stroke:var(--ink3);stroke-dasharray:5 4"/>')
        out.append(f'<rect x="{fx(0):.1f}" y="{fy(120):.1f}" width="{fx(60) - fx(0):.1f}" height="{fy(60) - fy(120):.1f}" '
                   f'style="fill:rgba(0,0,0,0.05)"/>')
        for k in X.cells:
            if k not in src:
                continue
            a_, b_ = X.nopost[k]["score"], src[k]["score"]
            P._marker(out, fx(a_), fy(b_), col, shape=SHAPE[k[1]], r=4.6,
                      title=f"{k[1]:g} mm, t {k[2]:+.3f} (x_s {xs_of(k[2]):.1f} mm): HOM {a_:.1f} deg, {title.split(',')[0]} {b_:.1f} deg")
    out.append("</svg>")
    leg = [(f"{d:g}{HS}mm cylinder", "var(--ink2)", False, SHAPE[d]) for d in DIAMS]
    return "".join(out), leg


def svg_post_scatter(X: Data):
    """Cell score with the post against without it, both plants."""
    W, H = 900, 400
    out = P._svg_open(W, H, "HOM cell score without the post against with it")
    panels = (("Servo plant, 10, 16 and 25 mm cylinders", X.post, X.nopost, 305),)
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
    hd = lambda r: "held" if r["held"] else "not held"  # noqa: E731
    leg = [(f"Post present: {hd(r0)} at {r0['turn_end_deg']:.0f}&#176;", C_POST, True, None),
           (f"Post removed: {hd(r1)} at {r1['turn_end_deg']:.0f}&#176;", C_HOM, False, None),
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

def _ticks(hi, step):
    return [round(step * i, 6) for i in range(int(math.floor(hi / step + 1e-9)) + 1)]


def svg_pl_scatter(X: Data):
    """Held turn against command rate and against angular-velocity jitter, one point per run, medians large."""
    W, H = 900, 430
    out = P._svg_open(W, H, "Held turn against the servo-command rate and against the cylinder's angular-velocity jitter for "
                      "each planner")
    xr = max(r["cmd_rate_rms"] for r in X.r_pl if r.get("cmd_rate_rms") is not None)
    xj = max(r["omega_jitter"] for r in X.r_pl if r.get("omega_jitter") is not None)
    r_hi = math.ceil(xr / 0.4) * 0.4
    j_hi = math.ceil(xj / 5) * 5
    panels = (("cmd_rate_rms", "Command rate (rad/s)", (0, r_hi), _ticks(r_hi, 0.4), "{:.1f}", 80),
              ("omega_jitter", "Angular-velocity jitter (&#176;/s)", (0, j_hi), _ticks(j_hi, 5), "{:g}", 520))
    for key, xlab, xs, xt, xfmt, px in panels:
        fx, fy = P._panel(out, px, 50, 330, 300, xs, (-5, 100), xt, [0, 30, 60, 90], xlab,
                          "Held turn (deg)" if px == 80 else "", xfmt=xfmt)
        out.append(f'<line x1="{px}" x2="{px + 330}" y1="{fy(60):.1f}" y2="{fy(60):.1f}" style="stroke:var(--ink3);stroke-dasharray:2 4"/>')
        for c in X.pl_cfgs:
            rows = [r for r in X.pl[c] if r.get(key) is not None]
            for r in rows:
                three = r["held"] and r["fingers_loaded_end"] == 3
                P._marker(out, fx(min(xs[1], r[key])), fy(r["held_turn_deg"]), PL_COL[c], shape=PL_SHP[c], hollow=not three,
                          r=3.6, title=f"{PL_LAB[c]}: {r['d_mm']:g} mm, t {r['t']:+.3f}, placement {r['seed']}: "
                                       f"held turn {r['held_turn_deg']:.1f} deg on {r['fingers_loaded_end']} fingers, "
                                       f"command rate {r['cmd_rate_rms']:.2f} rad/s, jitter {r['omega_jitter']:.1f} deg/s")
            if rows:
                mx, my = med([r[key] for r in rows]), med([r["held_turn_deg"] for r in rows])
                P._marker(out, fx(mx), fy(my), PL_COL[c], shape=PL_SHP[c], r=8.0,
                          title=f"{PL_LAB[c]}, median of {len(rows)} runs: {my:.1f} deg, {mx:.2f}")
    out.append("</svg>")
    leg = [(PL_LAB[c], PL_COL[c], False, PL_SHP[c]) for c in X.pl_cfgs]
    return "".join(out), leg


def _pl_cells(X: Data):
    return sorted({(r["d_mm"], r["t"]) for r in X.r_pl}, key=lambda k: (k[1] < 0, -k[0], k[1]))


def svg_pl_traces(X: Data, seed=0):
    """Turn angle against time for the filmed controllers on each planner cell, from the rows' 20 ms logs."""
    cells = _pl_cells(X)
    n = len(cells)
    W, H = 900, 330
    pw, ph = (W - 70 - 26 * (n - 1) - 20) / n, 220
    out = P._svg_open(W, H, "Turn angle against time for HOM and three sampling planners on four cells")
    for i, (d, t) in enumerate(cells):
        px = 70 + i * (pw + 26)
        fx, fy = P._panel(out, px, 50, pw, ph, (0.8, 6.3), (-30, 120), [1, 3, 5], [0, 45, 90],
                          "Time (s)" if i == n // 2 else "", "Turn angle (deg)" if i == 0 else "")
        out.append(f'<text x="{px + pw / 2:.1f}" y="24" text-anchor="middle" style="fill:var(--ink);font-weight:bold">'
                   f'{d:g}{HS}mm, {xs_of(t):g}{HS}mm</text>')
        out.append(f'<line x1="{px}" x2="{px + pw:.1f}" y1="{fy(90):.1f}" y2="{fy(90):.1f}" style="stroke:var(--ink3);stroke-dasharray:2 4"/>')
        for c in FILMED:
            r = X.pl_row(c, d, t, seed)
            if r is None:
                continue
            P._path(out, fx, fy, [(x["t"], max(-30.0, min(120.0, x["th"]))) for x in r["log"] if x["t"] >= 0.8], PL_COL[c],
                    width=2.0)
    out.append("</svg>")
    leg = [(PL_LAB[c], PL_COL[c], False, None) for c in FILMED if c in X.pl_cfgs]
    return "".join(out), leg


def svg_pl_command(X: Data, d=25.0, t=0.125, seed=0, j=5):
    """One servo target against time under each filmed controller, one strip each."""
    runs = [(c, X.pl_row(c, d, t, seed)) for c in FILMED]
    runs = [(c, r) for c, r in runs if r is not None]
    if not runs:
        return None, None
    series = {}
    for c, r in runs:
        L = [x for x in r["log"] if x["t"] >= 0.8 - 1e-9]
        u0 = L[0]["u"][j]
        series[c] = [(x["t"], math.degrees(x["u"][j] - u0)) for x in L]
    sh, gap = 82, 46
    W, H = 900, 40 + len(runs) * (sh + gap) + 20
    out = P._svg_open(W, H, f"{JOINTS[j]} servo target against time under four controllers")
    for i, (c, _) in enumerate(runs):
        y0 = 40 + i * (sh + gap)
        last = i == len(runs) - 1
        lo = min(min(v for _, v in series[c]), 0.0)
        hi = max(max(v for _, v in series[c]), 0.0)
        step = 20 if hi - lo > 50 else (10 if hi - lo > 20 else 5)
        lo, hi = step * math.floor(lo / step), step * math.ceil(hi / step)
        fx, fy = P._panel(out, 90, y0, 760, sh, (0.8, 6.3), (lo, hi), [1, 2, 3, 4, 5, 6] if last else [],
                          [lo, 0, hi] if lo < 0 < hi else [lo, hi], "Time (s)" if last else "", PL_LAB[c])
        if lo < 0 < hi:
            out.append(f'<line x1="90" x2="850" y1="{fy(0):.1f}" y2="{fy(0):.1f}" style="stroke:var(--ink3);stroke-dasharray:2 4"/>')
        P._path(out, fx, fy, series[c], PL_COL[c], width=1.6)
    out.append("</svg>")
    return "".join(out), None


def film(stem, caption, key, media=None):
    media = media or os.path.join(D, "media")
    p = os.path.join(media, f"{stem}.mp4")
    poster = os.path.join(media, f"{stem}.jpg")
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
     "median is largest; the rule of the scale study&#8217;s landscape. MPC and pinned iCEM run on the grasp that this rule picks for HOM "
     "without the post."),
    ("CAD hand, real_v1", "real_v1: the SR2 hand model of <code>assets/mjcf/real_v1/real_hand.xml</code>, links as capsules, "
     "MCP and PIP axes 20.75&#8202;mm apart. CAD hand: the same hand assembled from the STLs of its printed parts "
     "(<code>assets/mjcf/experimental/20261013-cad_v1/</code>), MCP and PIP axes 30.80&#8202;mm apart, 151 collision spheres "
     "per finger, printed-part masses, the TPU block 21.41&#8202;mm long."),
    ("Fingertip", "The printed TPU tip (a block with 2.7&#8202;mm fillets) as 1&#8202;mm sphere pads: 0.75&#8202;mm spheres on a "
     "1&#8202;mm grid, 1,016 per fingertip on the CAD block, each a soft contact with the cylinder; the default contact model of this program "
     "(<a href=\"../../overviews/contact_model.html\">fingertip contact overview</a>). MPC plans on the same model."),
    ("Servo plant", "The finger servo model of the bench plan: position gain 0.72&#8202;N&#8202;m/rad, torque limit 0.18&#8202;N&#8202;m, "
     "the measured SCS0009 stall torque."),
    ("Post", "The 3&#8202;mm-radius cylinder (at most 0.4 of the object radius) the object rests on before the grip. Post present: it "
     "collides for the whole run (<code>hom_post_servo/</code>). Post removed: its collisions end when the controller starts "
     "turning, at 1.3&#8202;s for HOM (after its squeeze) and at 0.8&#8202;s for MPC."),
    ("Governor freeze, swing past the freeze", "HOM stops advancing its turn reference when a pad&#8217;s force falls below 30&#8202;% of "
     "2&#8202;N or the turn lags the reference by 6&#176; for three ticks, and holds that angle. Swing past the freeze: \\(\\theta\\) at "
     "the end minus \\(\\theta\\) at the freeze."),
    ("Bench-plan band cell", "A cell the bench plan of 11 October predicts to turn from real_v1&#8217;s HOM rows: at least "
     "60&#176; in two of three placements on the servo plant with the post present."),
    ("Wall time per run", "Seconds of wall clock for one run from the grip to the end, on one core for HOM and with six rollout threads "
     "for MPC; the run simulates 6.3&#8202;s at most."),
    ("Placement 0", "The three-pad MPC and pinned iCEM ran one placement per cell of the landscape (seed 0); their cell score is that "
     "run&#8217;s held turn."),
    ("Pinned plan", "A plan whose first knot is fixed to the servo command in force at each replanning, so the command changes only "
     "through the later knots. Unpinned, the cheapest sample&#8217;s first knot can move every servo target by about the noise "
     "scale, 0.08&#8202;rad, every 20&#8202;ms."),
    ("Predictive sampling, MPPI, CEM, iCEM, annealed sampling", "Sampling planners that differ in how they turn the rolled-out "
     "samples into the next plan (Algorithm&#160;[[alg:planners]], Eqs.&#160;(4) and (5))."),
    ("Command rate", "RMS over the nine servo targets and the 10&#8202;ms samples after the grip of each target&#8217;s rate of "
     "change, in rad/s."),
    ("Command reversals", "Sign changes of a servo target&#8217;s 10&#8202;ms increments after the grip, increments under "
     "10<sup>&#8722;4</sup>&#8202;rad ignored, per joint per second."),
    ("Angular-velocity jitter", "RMS of the cylinder&#8217;s angular velocity minus its 100&#8202;ms running mean after the grip, in "
     "&#176;/s; zero for a turn at constant rate."),
    ("Three-pad fraction", "Share of the 10&#8202;ms samples after the grip in which all three fingers press on the cylinder with at "
     "least its weight."),
    ("Contact break", "A finger&#8217;s force on the cylinder falling below the cylinder&#8217;s weight for 20&#8202;ms or more after "
     "it had reached it."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def question(X: Data):
    n_cad = len({k for k in X.nopost if k[0] == "cylinder"})
    n_v1 = len({k for k in X.v1_nopost if k[0] == "cylinder"})
    return (f"<p>The <a href=\"{V1_PAGE}\">13 October page</a> measured how far the SR2 hand turns a cylinder in hand on the "
            "scale study&#8217;s layouts under HOM control and sampling model-predictive control (MPC), and compared sampling "
            "planners on four cells. This page repeats its servo-plant measurements on the CAD-model hand "
            f"(<a href=\"{CAD_PAGE}\">finger model from the CAD parts</a>): the finger assembled from the STLs of its printed "
            "parts. Its MCP link spaces the MCP and PIP axes 30.80&#8202;mm apart against real_v1&#8217;s 20.75&#8202;mm, so "
            "each finger reaches 10.05&#8202;mm further. Its links collide through 151 spheres per finger. Its masses are those "
            "of the printed parts and servos, 85&#8202;g per finger. Its TPU fingertip block is 21.41&#8202;mm long against "
            "22.0&#8202;mm, which gives 1,016 pads per fingertip. Joints, mounts, actuators, the servo plant, the cylinders, the "
            "grasp rule and the controllers are the real_v1 page&#8217;s. The scale study&#8217;s closed-form kinematics "
            "(<code>scripts/hand_object_scale_kin.py</code>, <code>SCALE_HAND=cad_v1</code>) use the CAD spacing and agree "
            "with the compiled CAD model to 0.0000&#8202;mm over 600 random poses.</p>"
            f"<p>The longer finger has a feasible grasp in {n_cad} cells of the diagonal layout family with the 10, 16 and "
            f"25&#8202;mm cylinders, against {n_v1} for real_v1 (a grasp is feasible when the scale study&#8217;s clearance "
            "model finds a palm height and straddle at which every pad reaches the cylinder). Each run grips for 0.8&#8202;s with the cell&#8217;s grip command and then "
            "hands control to the controller until 6.3&#8202;s or until HOM&#8217;s hold ends; the support post under the "
            "cylinder is removed when the controller starts turning, at 1.3&#8202;s for HOM and 0.8&#8202;s for MPC, as on the "
            "real_v1 page. Beyond the real_v1 page, pinned iCEM, the planner that turns furthest in the planner comparison, runs on "
            "every cell at placement 0.</p>")


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
    big1, sw1 = X.swing(X.r_v1_post)
    difv = [abs(X.nopost[k]["score"] - X.post[k]["score"]) for k in X.post if k in X.nopost]
    s = (f"<p>With the post present for the whole run, {len(big)} of the {len(X.r_post)} HOM runs on the CAD hand were held "
         f"at 60&#176; or more, and in {len(sw)} of them the cylinder had turned more than 15&#176; past the angle where the "
         f"governor stopped the reference (real_v1: {len(sw1)} of {len(big1)}).")
    svg, leg = svg_post_trace(X)
    s += (" Figure&#160;[[fig:posttrace]] shows one run on the 16&#8202;mm cylinder at the real_v1 bench plan&#8217;s band "
          "cell.</p>" if svg else "</p>")
    if svg:
        s += figure(svg, "HOM on the CAD hand, 16&#8202;mm cylinder at \\(t = 0.625\\) (\\(x_s\\) = 62.5&#8202;mm), placement 0, "
                    "servo plant. Top: turn angle with the post present for the whole run and with it removed at 1.3&#8202;s; "
                    "squares mark the governor&#8217;s freeze. Bottom: the post&#8217;s normal force on the cylinder in the run "
                    "with the post (clipped at 2.5&#8202;N).", leg, key="posttrace")
    if difv:
        s += (f"<p>Removing the post moves the cell scores by a median {med(difv):.0f}&#176; and up to {max(difv):.0f}&#176; "
              f"(Figure&#160;[[fig:postscatter]]). Without the post, {len(sw0)} of the {len(big0)} runs held at 60&#176; or "
              "more still swung past the freeze.</p>")
        svg, leg = svg_post_scatter(X)
        s += figure(svg, "HOM cell scores on the CAD hand without the post against with it, servo plant, 10, 16 and 25&#8202;mm "
                    "cylinders. Points on the dashed line are unchanged.", leg, key="postscatter")
    return s


NUM = {1: "one", 2: "two", 3: "three"}


def cell_list(src, ks):
    """'the 16 mm cylinder at 92.5 and 100 mm (73° and 66°) and the 25 mm cylinder at ...' for single-placement scores."""
    by = defaultdict(list)
    for k in sorted(ks, key=lambda k: (k[1], -k[2])):
        by[k[1]].append(k)

    def val(k):
        n = src[k]["rows"][0]["fingers_loaded_end"]
        return f"{src[k]['score']:.0f}&#176;" + ("" if n == 3 else f" on {NUM.get(n, n)} fingers")

    def join(xs):
        return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    return join([f"the {d:g}{HS}mm cylinder at {join([f'{xs_of(k[2]):g}' for k in kk])}{HS}mm ({join([val(k) for k in kk])})"
                 for d, kk in by.items()])


def land_text(X: Data):
    """Pinned iCEM at placement 0 on every cell against the three-pad MPC and HOM on the same cells."""
    ks = sorted(X.land)
    sc = [X.land[k]["score"] for k in ks]
    rows = [X.land[k]["rows"][0] for k in ks]
    three = sum(1 for r in rows if r["held"] and r["fingers_loaded_end"] == 3)
    d3 = [X.land[k]["score"] - X.mpc3[k]["score"] for k in ks if k in X.mpc3]
    dh = [X.land[k]["score"] - X.nopost[k]["score"] for k in ks if k in X.nopost]
    hi = sorted((k for k in ks if X.land[k]["score"] >= 60), key=lambda k: (k[1], -k[2]))
    f = lambda rr, key: med([r[key] for r in rr if r.get(key) is not None])  # noqa: E731
    m3 = [r for r in X.r_mpc3 if ("cylinder", float(r["d_mm"]), float(r["t"])) in X.land]
    nh = sum(1 for r in rows if r["held"])
    return (f"<p>Pinned iCEM, the planner that turns furthest in the next section, holds the cylinder in "
            + (f"all {len(ks)}" if nh == len(ks) else f"{nh} of the {len(ks)}")
            + f" cells at placement 0 with the same cost and rollout budget as "
            f"the three-pad MPC, on three pads in {three}, and turns it to a median {deg(med(sc))}. Per cell it turns "
            f"{med(d3):.0f}&#176; further than the three-pad MPC and {med(dh):.0f}&#176; further than HOM without the post "
            f"(medians of the differences); it exceeds HOM by more than 10&#176; in {sum(1 for x in dh if x > 10)} cells and falls "
            f"more than 10&#176; below it in {sum(1 for x in dh if x < -10)}. It reaches 60&#176; in {len(hi)} cells: "
            + cell_list(X.land, hi) + ". "
            f"Over these {len(rows)} runs it moves the servo targets at a median {f(rows, 'cmd_rate_rms'):.2f}&#8202;rad/s with "
            f"{f(rows, 'cmd_reversals'):.1f} reversals per joint per second, and the cylinder&#8217;s angular-velocity jitter is "
            f"{f(rows, 'omega_jitter'):.1f}&#8202;&#176;/s. The three-pad MPC&#8217;s runs on the same cells give "
            f"{f(m3, 'cmd_rate_rms'):.2f}&#8202;rad/s, {f(m3, 'cmd_reversals'):.1f} reversals and "
            f"{f(m3, 'omega_jitter'):.1f}&#8202;&#176;/s.</p>")


def landscape_section(X: Data):
    if not X.cells or not X.mpc3:
        return "<p>PENDING: the MPC grids on the CAD hand have not finished (<code>mpc_servo/</code>, <code>mpc3_servo/</code>).</p>"
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
         f"leave the cylinder, which swings about the remaining pinch at a median peak of {st_m['rate']:.0f}&#176;/s. "
         f"{st_m['two']} of its {st_m['held']} held runs end on two fingers, and {st_m['h60_3']} of its {st_m['n']} runs end "
         f"held at 60&#176; or more on three pads (Table&#160;[[tab:runs]]"
         + (", Figure&#160;[[fig:film16625]]" if X.films else "") + "). "
         + (f"HOM with the post, the protocol from which the real_v1 bench plan picked its band cells, reaches 60&#176; in "
            + ("no cell" if p60 == 0 else f"{p60} cells") + " on the CAD hand.</p>"))
    if k3:
        c60 = sum(1 for x in mpc3 if x >= 60)
        better = sum(1 for a, b in zip(hom3, mpc3) if b > a + 10)
        n3 = sum(1 for r in X.r_mpc3 if r["held"] and r["fingers_loaded_end"] == 3)
        s += (f"<p>With the three-pad and turn-rate terms, MPC holds the cylinder on three pads in {n3} "
              f"of {st_3['n']} runs at placement 0, at a median peak rate of {st_3['rate']:.0f}&#176;/s, and turns it to a median "
              f"{deg(med(mpc3))} over {len(k3)} cells, against {deg(med(hom3))} for HOM on the same cells. It reaches 60&#176; in {c60} "
              f"cells and exceeds HOM by more than 10&#176; in {better}. The 60&#176; cells are "
              + cell_list(X.mpc3, [k for k in k3 if X.mpc3[k]["score"] >= 60])
              + "; every other cell stays between "
              + f"{min(x for x in mpc3 if x < 60):.0f} and {max(x for x in mpc3 if x < 60):.0f}&#176;.</p>")
    if X.land:
        s += land_text(X)
    svg, leg = svg_landscape(X)
    s += figure(svg, "Cell score against thumb-pair separation on the servo plant, one panel per cylinder diameter; compact layouts on "
                "the left. Grey columns: the bench plan&#8217;s band cells. Dotted line: 60&#176;. Hover a point for its placements and "
                "grasp.", leg, key="landscape")
    svg, leg = svg_pairs(X)
    s += figure(svg, "Cell scores of MPC and pinned iCEM against HOM&#8217;s (post removed), one point per cell; shape by "
                "diameter. Left: MPC whose cost asks only for an upright, held cylinder. Middle: MPC that also keeps three pads on the "
                "cylinder and the turn below 60&#176;/s, at placement 0. Right: pinned iCEM with the same cost and budget, at placement "
                "0. Points above the dashed line turn further than under HOM; the grey square holds scores of 60&#176; or more where "
                "HOM stays below 60&#176;.", leg, key="pairs")
    rows = []
    for lab, st_ in (("HOM, post present (bench plan)", st_p), ("HOM, post removed", st_h), ("MPC", st_m),
                     ("MPC, three pads, 60&#176;/s", st_3), ("Pinned iCEM, three pads", X.stats(X.r_land))):
        if not st_["n"]:
            continue
        rows.append([lab, str(st_["n"]), str(st_["held"]), str(st_["h60"]), str(st_["h60_3"]), str(st_["two"]),
                     "&#8211;" if st_["rate"] is None else f"{st_['rate']:.0f}"])
    s += table(["Controller", "Runs", "Held", "Held at 60&#176; or more", "Of these on three pads", "Held runs ending on two fingers",
                "Median peak turn rate, &#176;/s"], rows,
               "Runs of each controller on the servo plant: all grasps for HOM (three placements each), the scoring grasp for MPC "
               "and pinned iCEM (three placements for MPC, placement 0 for the three-pad MPC and pinned iCEM).",
               note="HOM&#8217;s rows store no trace, so their turn rate is not available; its reference advances at 30&#176;/s.",
               key="runs")
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
        three = lambda src: ("&#8211;" if k not in src else  # noqa: E731
                             f"{src[k]['rows'][0]['held_turn_deg']:.0f} ({src[k]['rows'][0]['fingers_loaded_end']})")
        body.append([f"{c['d_mm']:g}{HS}mm", f"{c['t']:+.3f}", f"{xs_of(c['t']):.1f}", "yes" if c.get("in_band") else "",
                     f(X.post), str(swn), f(X.nopost), f(X.mpc), three(X.mpc3), three(X.land)])
    s += table(["Cylinder", "Layout \\(t\\)", "\\(x_s\\), mm", "Band cell", "HOM, post, deg", "Placements past the freeze",
                "HOM, no post, deg", "MPC, deg", "MPC three pads, deg (fingers)", "Pinned iCEM, deg (fingers)"], body,
               "The bench plan&#8217;s 15 cells: cell scores under each controller and protocol on the servo plant.",
               note="Placements past the freeze: of the three HOM runs with the post at the scoring grasp, those whose cylinder turned more "
                    "than 15&#176; past the governor&#8217;s freeze. Each HOM column picks its own grasp by the cell-score rule; the MPC "
                    "and pinned-iCEM columns run on the grasp of the no-post HOM column; the last two columns are placement 0, with the "
                    "fingers loaded at the end.", text_cols=(0, 3), key="bench")
    return s


def _hom_jitter_text(X: Data, hm, top):
    rows = [r for r in X.pl.get("hom", []) if r.get("omega_jitter") is not None]
    big = [r["omega_jitter"] for r in rows if r["d_mm"] == 25.0]
    small = [r for r in rows if r["d_mm"] < 25.0]
    if not big or not small:
        return ""
    js = [r["omega_jitter"] for r in small]
    lost = sum(1 for r in small if not r["held"])
    return (f"HOM&#8217;s median angular-velocity jitter, {hm['jit']:.1f}&#8202;&#176;/s, is the "
            + ("largest of the controllers" if top else "HOM median") + f"; it ranges from {min(big):.1f} to {max(big):.1f}&#8202;"
            f"&#176;/s on the 25&#8202;mm cylinder and from {min(js):.0f} to {max(js):.0f}&#8202;&#176;/s on the 16 and "
            f"10&#8202;mm cylinders, where {lost} of its {len(small)} runs end not held.")


def planners_section(X: Data):
    if not X.pl_cfgs:
        return "<p>The planner rows are not yet in <code>planners/turn.jsonl</code>.</p>"
    st = {c: X.pl_stats(c) for c in X.pl_cfgs}
    cells = _pl_cells(X)
    seeds = sorted({r["seed"] for r in X.r_pl})
    band = {(float(c["d_mm"]), float(c["t"])) for c in X.plan["cells"] if c.get("in_band")}
    cell_txt = "; ".join(f"the {d:g}{HS}mm cylinder at \\(t = {t:+.3f}\\) (\\(x_s\\) = {xs_of(t):g}{HS}mm"
                         + (", a bench-plan band cell)" if (d, t) in band else ")") for d, t in cells)
    s = ("<p>The three-pad MPC of the previous section is unpinned predictive sampling: it keeps the cylinder on three pads, but "
         "its servo targets jump at every replanning. This section compares "
         f"sampling planners at the same budget on {len(cells)} cells and {len(seeds)} placements each. The cells are {cell_txt}. Every CPU planner "
         "uses the three-pad cost of Eq.&#160;(3) and spends 64 rollouts of 0.5&#8202;s per 20&#8202;ms replanning on a plan of "
         "three knots with noise 0.08&#8202;rad; HOM runs on the same grasps and placements "
         "(<code>scripts/hand_object_scale_planners.py</code>).</p>"
         "<p>Each planner keeps a nominal plan, shifts it forward at every replanning, and differs in how it turns the rolled-out "
         "samples into the next nominal (Algorithm&#160;[[alg:planners]]). Predictive sampling keeps the cheapest sample. MPPI "
         "averages the samples with the weights of Eq.&#160;(4). CEM refits the mean and spread of the noise to the five cheapest of "
         "32 samples, twice per replanning, and executes the mean. iCEM (Pinneri et al. 2020) draws noise correlated along the "
         "knots, carries the two best samples of the last replanning into the first population and executes the cheapest sample. "
         "Annealed sampling, the sampler of DIAL-MPC (Xue et al. 2024) as implemented on the CPU in our <code>mujoco_mpc</code> "
         "branch, runs four stages of 16 samples whose noise, Eq.&#160;(5), shrinks with the stage and toward the start of the plan, "
         "with an MPPI update after each stage. A pinned planner fixes the first knot of every sample to the servo command in force, "
         "so a replanning moves the command only through the later knots.</p>")
    s += eq(r"w_j = \frac{\exp\!\left(-(J_j - J_{\min})/\lambda\right)}{\sum_l \exp\!\left(-(J_l - J_{\min})/\lambda\right)},"
            r"\qquad \lambda = 0.03\,\operatorname{std}_l J_l", 4)
    s += eq(r"\sigma_{i,h} = \sigma\,\exp\!\left(-\frac{i}{\beta_1 I} - \frac{K-1-h}{\beta_2 K}\right),\qquad "
            r"i = 0,\dots,I-1,\quad h = 0,\dots,K-1", 5)
    s += ("<p>with \\(I = 4\\) stages, \\(K = 3\\) knots, knot \\(h = 0\\) at the start of the plan, and "
          "\\(\\beta_1 = \\beta_2 = 0.5\\).</p>")
    ALG[0] += 1
    REFS["alg:planners"] = ALG[0]
    ALG[0] -= 1
    s += alg("One replanning of the sampling planners at time \\(t_k\\): 64 rollouts in \\(I\\) iterations of \\(n = 64/I\\)",
             r"\Require nominal knots \(\bar K\) shifted to \(t_k\); servo command in force \(u_0\)" "\n"
             r"\For{\(i = 0, \dots, I-1\)}" "\n"
             r"\State \(K_1 \gets \bar K\);\ \ \(K_j \gets \mathrm{clip}(\bar K + \varepsilon_j)\), \(j = 2, \dots, n\) "
             r"\Comment{iCEM: \(\varepsilon\) AR(1) along the knots}" "\n"
             r"\If{pinned} \State first knot of every \(K_j \gets u_0\) \EndIf" "\n"
             r"\State roll out every \(K_j\) for 0.5 s from the plant state; costs \(J_j\) by Eqs. (1)--(3)" "\n"
             r"\If{predictive sampling} \State \(\bar K \gets K_{\arg\min_j J_j}\)" "\n"
             r"\ElsIf{MPPI, annealed sampling} \State \(\bar K \gets \sum_j w_j K_j\) \Comment{Eq. (4); annealed noise Eq. (5)}" "\n"
             r"\ElsIf{CEM, iCEM} \State mean and spread of the noise \(\gets\) those of the 5 cheapest, momentum 0.1" "\n"
             r"\EndIf" "\n"
             r"\EndFor" "\n"
             r"\State apply \(\bar K\) until \(t_k + 0.02\) s \Comment{CEM: the mean; iCEM: the cheapest sample}")

    ps, pp, hm = st.get("ps"), st.get("ps_pin"), st.get("hom")
    pinned = [c for c in X.pl_cfgs if c.endswith("_pin")]
    if ps and pp and hm:
        rates = [st[c]["rate"] for c in pinned]
        revs = [st[c]["rev"] for c in pinned]
        lo_c, hi_c = min(pinned, key=lambda c: st[c]["rev"]), max(pinned, key=lambda c: st[c]["rev"])
        top = max(st, key=lambda c: st[c]["jit"]) == "hom"
        s += (f"<p>Pinning cuts predictive sampling&#8217;s command rate from {ps['rate']:.2f} to {pp['rate']:.2f}&#8202;rad/s, its "
              f"command reversals from {ps['rev']:.1f} to {pp['rev']:.1f} per joint per second and the cylinder&#8217;s "
              f"angular-velocity jitter from {ps['jit']:.1f} to {pp['jit']:.1f}&#8202;&#176;/s (medians over {pp['n']} runs; "
              f"Table&#160;[[tab:plsmooth]], Figure&#160;[[fig:plscatter]]). Every pinned planner moves its servo targets at "
              f"{min(rates):.2f}&#8211;{max(rates):.2f}&#8202;rad/s, at or below HOM&#8217;s {hm['rate']:.2f}, and reverses them "
              f"{min(revs):.1f} ({lc(PL_LAB[lo_c])}) to {max(revs):.1f} ({lc(PL_LAB[hi_c])}) times per joint per second, against "
              f"{hm['rev']:.1f} for HOM (Figure&#160;[[fig:plcommand]]). " + _hom_jitter_text(X, hm, top) + "</p>")
    cpu = [c for c in X.pl_cfgs if c not in ("hom", "anneal_gpu", "icem_gpu")]
    if not cpu or "hom" not in st:
        return s + "<p>PENDING: the planner runs on the CAD hand have not finished (<code>planners/turn.jsonl</code>).</p>"
    best = max(cpu, key=lambda c: (st[c]["turn"], -st[c]["rate"]))
    b = st[best]
    mpc = [c for c in X.pl_cfgs if c != "hom"]
    n_three = sum(st[c]["three_end"] for c in mpc)
    n_mpc = sum(st[c]["n"] for c in mpc)
    br_mpc = sum(st[c]["breaks"] for c in mpc)
    s += (f"<p>{PL_LAB[best]} turns furthest: a median held turn of {deg(b['turn'])} over its {b['n']} runs, {b['h60']} of them "
          f"at 60&#176; or more on three pads, against {deg(hm['turn'])} for HOM ({hm['h60']} of {hm['n']}). "
          + (f"All {n_mpc} MPC runs end held on three pads" if n_three == n_mpc else
             f"The MPC runs end held on three pads in {n_three} of {n_mpc}")
          + (", and none breaks a finger&#8217;s contact; " if br_mpc == 0 else f" and break a finger&#8217;s contact {br_mpc} times in all; ")
          + f"HOM holds {hm['held']} of its {hm['n']} runs and breaks contact {hm['breaks']} times. "
          + (f"Of the pinned planners, MPPI turns least, a median {deg(st['mppi_pin']['turn'])}, at the lowest command rate, "
             f"{st['mppi_pin']['rate']:.2f}&#8202;rad/s." if "mppi_pin" in st and
             min((c for c in pinned if c in st), key=lambda c: st[c]["turn"]) == "mppi_pin" else "") + "</p>")
    if "anneal_gpu" in st:
        g, a_ = st["anneal_gpu"], st.get("anneal_pin")
        s += (f"<p>On the GPU, annealed sampling draws 1,024 rollouts per replanning, 16 times the CPU budget, in four stages of 256 "
              f"(the port is in the cost section). Over {g['n']} runs it turns the cylinder a median {deg(g['turn'])}"
              + (f", against {deg(a_['turn'])} for the CPU version with 64 rollouts" if a_ else "")
              + f", at a command rate of {g['rate']:.2f}&#8202;rad/s and {g['rev']:.1f} reversals per joint per second, and "
              f"costs {g['wall']:,.0f}&#8202;s per run.</p>")
    if "icem_gpu" in st:
        g, a_ = st["icem_gpu"], st.get("icem_pin")
        s += (f"<p>iCEM on the GPU draws the same 1,024 rollouts in two iterations of 512, with the noise and kept elites of the "
              f"CPU version. Over {g['n']} runs it turns the cylinder a median {deg(g['turn'])}"
              + (f", against {deg(a_['turn'])} for iCEM with 64 rollouts" if a_ else "")
              + f", with {g['three_end']} of {g['n']} runs held on three pads, at a command rate of {g['rate']:.2f}&#8202;rad/s, "
              f"and costs {g['wall']:,.0f}&#8202;s per run.</p>")
    rows, rows2 = [], []
    for c in X.pl_cfgs:
        v = st[c]
        rows.append([PL_LAB[c], str(v["n"]), str(v["held"]), str(v["three_end"]), f"{v['three_frac']:.3f}", str(v["breaks"]),
                     f"{v['turn']:.0f}", str(v["h60"])])
        rows2.append([PL_LAB[c], f"{v['rate']:.2f}", f"{v['rev']:.1f}", f"{v['jit']:.1f}", f"{v['peak']:.0f}",
                      f"{v['wall']:.1f}" if v["wall"] < 10 else f"{v['wall']:,.0f}"])
    s += table(["Controller", "Runs", "Held", "Held on three pads at the end", "Three-pad fraction, mean", "Contact breaks, total",
                "Held turn, median, deg", "Held at 60&#176; or more on three pads"], rows,
               f"Holding and turning on the {len(cells)} planner cells at placements {', '.join(str(x) for x in seeds)}, servo "
               "plant. Every MPC planner spends 64 rollouts of 0.5&#8202;s per replanning on six CPU threads, except the two GPU "
               "planners (1,024 on the GPU).", note="The held turn of a run that is not held is 0.", key="planners")
    s += table(["Controller", "Command rate, rad/s", "Command reversals per joint per s", "Angular-velocity jitter, &#176;/s",
                "Peak turn rate, &#176;/s", "Wall time per run, s"], rows2,
               "Smoothness and cost of the same runs, medians over runs.", key="plsmooth")
    runs = [(d, t, sd) for d, t in cells for sd in seeds]
    head = ["Controller"] + [f"{d:g}{HS}mm<br>{xs_of(t):g}{HS}mm<br>{sd}" for d, t, sd in runs]
    body = []
    tops = {}
    for k in runs:
        hv = [r["held_turn_deg"] for c in X.pl_cfgs for r in [X.pl_row(c, *k)] if r is not None and r["held"]]
        tops[k] = max(hv) if hv else None
    for c in X.pl_cfgs:
        line = [PL_LAB[c]]
        for k in runs:
            r = X.pl_row(c, *k)
            if r is None:
                line.append("&#8211;")
            elif not r["held"]:
                line.append("dropped" if r["dropped"] else "not held")
            else:
                txt = signed0(r["held_turn_deg"]) + ("" if r["fingers_loaded_end"] == 3 else f" ({r['fingers_loaded_end']})")
                line.append(f"<b>{txt}</b>" if r["held_turn_deg"] == tops[k] else txt)
        body.append(line)
    s += table(head, body, "Held turn in degrees of every planner run; columns by cylinder diameter, thumb-pair separation "
               "\\(x_s\\) and placement; bold: the largest of the column.",
               note="A number in brackets gives the fingers loaded at the end when fewer than three.", key="plcells")
    svg, leg = svg_pl_scatter(X)
    s += figure(svg, "Held turn against the RMS servo-command rate (left) and against the cylinder&#8217;s angular-velocity jitter "
                "(right), one small marker per run and a large marker at each controller&#8217;s medians. Hollow: not held on three "
                "pads at the end. Dotted line: 60&#176;. Hover a marker for its run.", leg, key="plscatter")
    svg, leg = svg_pl_traces(X)
    s += figure(svg, "Turn angle against time at placement 0 of each planner cell, from the runs&#8217; 20&#8202;ms logs; panels "
                "titled by diameter and thumb-pair separation. Dotted line: 90&#176;.", leg, key="pltraces")
    svg, _ = svg_pl_command(X)
    if svg:
        s += figure(svg, "The index finger&#8217;s PIP servo target, relative to its value at the end of the grip, against time on "
                    "the 25&#8202;mm cylinder at \\(x_s\\) = 92.5&#8202;mm, placement 0, in degrees, one strip per controller, each on its own "
                    "scale (20&#8202;ms samples).",
                    key="plcommand")
    return s


def svg_compare(X: Data):
    """CAD-hand cell scores against real_v1's, one panel per controller, one point per cell both hands reach."""
    W, H = 900, 380
    out = P._svg_open(W, H, "Cell scores on the CAD hand against real_v1 for HOM, MPC and three-pad MPC")
    panels = (("HOM, post removed", X.v1_nopost, X.nopost, C_HOM), ("MPC", X.v1_mpc, X.mpc, C_MPC),
              ("MPC, three pads (placement 0)", X.v1_mpc3, X.mpc3, C_MPC3))
    pw, gap = 236, 52
    for i, (title, a, b, col) in enumerate(panels):
        px = 66 + i * (pw + gap)                      # the last panel's 120 tick ends inside the 900 px box
        fx, fy = P._panel(out, px, 50, pw, 236, (0, 120), (0, 120), [0, 30, 60, 90, 120], [0, 30, 60, 90, 120],
                          "real_v1 cell score (deg)" if i == 1 else "", "CAD-hand cell score (deg)" if i == 0 else "")
        out.append(f'<text x="{px + pw / 2:.1f}" y="22" text-anchor="middle" style="fill:var(--ink);font-weight:bold">{title}</text>')
        out.append(f'<path d="M{fx(0):.1f},{fy(0):.1f} L{fx(120):.1f},{fy(120):.1f}" style="stroke:var(--ink3);stroke-dasharray:5 4"/>')
        for k in sorted(set(a) & set(b)):
            if k[0] != "cylinder":
                continue
            P._marker(out, fx(a[k]["score"]), fy(b[k]["score"]), col, shape=SHAPE[k[1]], r=4.6,
                      title=f"{k[1]:g} mm, t {k[2]:+.3f} (x_s {xs_of(k[2]):.1f} mm): real_v1 {a[k]['score']:.1f} deg, "
                            f"CAD {b[k]['score']:.1f} deg")
    out.append("</svg>")
    leg = [(f"{d:g}{HS}mm cylinder", "var(--ink2)", False, SHAPE[d]) for d in DIAMS]
    return "".join(out), leg


def compare_section(X: Data):
    rows = []
    paras = []
    for lab, a, b in (("HOM, post present", X.v1_post, X.post), ("HOM, post removed", X.v1_nopost, X.nopost),
                      ("MPC", X.v1_mpc, X.mpc), ("MPC, three pads, placement 0", X.v1_mpc3, X.mpc3)):
        ka = [k for k in a if k[0] == "cylinder"]
        kb = [k for k in b if k[0] == "cylinder"]
        both = sorted(set(ka) & set(kb))
        if not kb:
            continue
        va, vb = [a[k]["score"] for k in ka], [b[k]["score"] for k in kb]
        da = [b[k]["score"] - a[k]["score"] for k in both]
        rows.append([lab, str(len(ka)), str(len(kb)), f"{med(va):.0f}", f"{med(vb):.0f}",
                     str(sum(1 for v in va if v >= 60)), str(sum(1 for v in vb if v >= 60)),
                     f"{med(da):+.0f}".replace("-", "&#8722;") if da else "&#8211;"])
    dif = {}
    for lab, a, b in (("hom", X.v1_nopost, X.nopost), ("mpc", X.v1_mpc, X.mpc), ("mpc3", X.v1_mpc3, X.mpc3)):
        both = [k for k in b if k in a and k[0] == "cylinder"]
        dif[lab] = med([b[k]["score"] - a[k]["score"] for k in both]) if both else None
    sg = lambda x: f"{x:+.0f}".replace("-", "&#8722;")  # noqa: E731
    s = ("<p>Table&#160;[[tab:compare]] and Figure&#160;[[fig:compare]] set each controller&#8217;s cell scores on the CAD "
         "hand against real_v1&#8217;s; the cells that only the CAD hand reaches enter the CAD columns of the table but not "
         "the figure. On the cells both hands reach, the median per-cell difference, CAD minus real_v1, is "
         f"{sg(dif['hom'])}&#176; for HOM without the post, {sg(dif['mpc'])}&#176; for MPC and {sg(dif['mpc3'])}&#176; for the "
         "three-pad MPC.")
    if X.pl_cfgs and X.v1_pl:
        hm1 = med([r["held_turn_deg"] for r in X.v1_pl.get("hom", [])])
        ic1 = med([r["held_turn_deg"] for r in X.v1_pl.get("icem_pin", [])])
        s += (f" On the four planner cells HOM turns a median {deg(X.pl_stats('hom')['turn'])} against {deg(hm1)} on real_v1, "
              f"and pinned iCEM {deg(X.pl_stats('icem_pin')['turn'])} against {deg(ic1)} (Table&#160;[[tab:plcompare]]).")
    s += "</p>"
    s += table(["Controller", "Cells, real_v1", "Cells, CAD", "Median cell score, real_v1, deg", "Median cell score, CAD, deg",
                "Cells at 60&#176; or more, real_v1", "Cells at 60&#176; or more, CAD",
                "CAD minus real_v1 on shared cells, median, deg"], rows,
               "Cell scores of each controller on the two hands, servo plant, 10, 16 and 25&#8202;mm cylinders.", key="compare")
    svg, leg = svg_compare(X)
    s += figure(svg, "CAD-hand cell scores against real_v1&#8217;s on the cells both hands reach; shape by diameter. Points "
                "above the dashed line turn further on the CAD hand.", leg, key="compare")
    if X.pl_cfgs and X.v1_pl:
        body = []
        for c in X.pl_cfgs:
            v = X.pl_stats(c)
            r1 = X.v1_pl.get(c, [])
            if r1:
                m1 = med([r["held_turn_deg"] for r in r1])
                t1 = sum(1 for r in r1 if r["held"] and r["fingers_loaded_end"] == 3)
                rate1 = med([r["cmd_rate_rms"] for r in r1 if r.get("cmd_rate_rms") is not None])
                v1txt = [f"{m1:.0f}", f"{t1} of {len(r1)}", f"{rate1:.2f}"]
            else:
                v1txt = ["&#8211;", "&#8211;", "&#8211;"]
            body.append([PL_LAB[c]] + v1txt + [f"{v['turn']:.0f}", f"{v['three_end']} of {v['n']}", f"{v['rate']:.2f}"])
        s += table(["Controller", "Held turn, median, real_v1, deg", "Held on three pads, real_v1", "Command rate, real_v1, rad/s",
                    "Held turn, median, CAD, deg", "Held on three pads, CAD", "Command rate, CAD, rad/s"], body,
                   "The planner comparison on both hands: the same four layouts and cylinders, placements 0 and 1, each "
                   "hand on its own best HOM grasp.", key="plcompare")
    return s


def films_section(X: Data):
    out = ""
    groups = defaultdict(dict)
    for f in X.pl_films:
        groups[(f["d_mm"], f["t"], f["seed"], f["tag"])][f["config"]] = f      # the latest film of a cell wins
    keys = [k for k in sorted(groups, key=lambda k: (k[1] < 0, -k[0], k[1])) if all(c in groups[k] for c in FILMED)]
    if keys:
        out += ("<p>Each film shows four controllers on the same grasp and placement at real speed: HOM top left (post removed at "
                "1.3&#8202;s), unpinned predictive sampling top right, whose servo targets jump at every replanning, pinned "
                "predictive sampling bottom left and pinned iCEM bottom right, all three planners with the three-pad cost. Pads touching the cylinder are drawn red, the orange rod marks the "
                "cylinder&#8217;s axis, and the post is drawn only while it collides. The label gives the time and the turn angle. "
                "Each panel is a fresh run with the row&#8217;s seed; its result is in the caption.</p>")
    for (d, t, seed, tag) in keys:
        g = groups[(d, t, seed, tag)]
        obj = g["hom"]["clip"][len("hom_"):-len(".mp4")]
        parts = [f"{PL_LAB[c]}: {'held' if g[c]['held'] else 'not held'} at {g[c]['turn_end_deg']:.0f}&#176; on "
                 f"{g[c]['fingers_loaded_end']} fingers" for c in FILMED]
        cap = (f"{d:g}{HS}mm cylinder at \\(t = {t:+.3f}\\) (\\(x_s\\) = {xs_of(t):.1f}&#8202;mm), grasp {g['hom']['cand']}, "
               f"placement {seed}. " + "; ".join(parts) + ".")
        out += film(f"planners_{obj}", cap, key=f"plfilm{int(d)}{'m' if t < 0 else ''}{abs(int(round(1000 * t)))}",
                    media=os.path.join(PL, "media"))
    rows = defaultdict(dict)
    for f in X.films:
        rows[(f["d_mm"], f["t"], f["seed"], f["tag"])][f["controller"]] = f
    names = {"HOM": "HOM", "MPC": "MPC", "MPC3": "MPC with three pads"}
    for (d, t, seed, tag) in [k for k in rows if k[0] == 16.0 and k[1] == 0.625]:
        pair = rows[(d, t, seed, tag)]
        ctrls = [c for c in ("HOM", "MPC", "MPC3") if c in pair]
        stem = f"{'_'.join(c.lower() for c in ctrls)}_cyl{d:05.1f}_{tag}_s{seed}"
        parts = [f"{names[c]}: {'held' if pair[c]['held'] else 'not held'} at {pair[c]['turn_end_deg']:.0f}&#176; on "
                 f"{pair[c]['fingers_loaded_end']} fingers" for c in ctrls]
        out += ("<p>The film below shows the three controllers of the landscape on the 16&#8202;mm cylinder at "
                "\\(x_s\\) = 62.5&#8202;mm: HOM, MPC whose cost asks only for an upright, held cylinder, and the three-pad MPC; "
                "both MPC panels are unpinned predictive sampling.</p>")
        out += film(stem, f"{d:g}{HS}mm cylinder at \\(t = {t:+.3f}\\) (\\(x_s\\) = {xs_of(t):.1f}&#8202;mm), grasp "
                    f"{pair[ctrls[0]]['cand']}, placement {seed}, panels left to right. " + "; ".join(parts) + ".",
                    key="film16625")
    return out


def cost_section(X: Data):
    if not X.r_mpc or not X.r_mpc3:
        return "<p>PENDING: the MPC grids on the CAD hand have not finished; their wall times enter this section.</p>"
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
         "rolls out 9.6&#8202;s of simulated time, which is 32 plans of 0.3&#8202;s. Each replanning of the second rolls out 32&#8202;s. Both use the 1&#8202;mm sphere pads at "
         f"1&#8202;ms steps. Planning takes {100 * frac:.0f}&#8202;% of MPC&#8217;s wall time. The MPC grid, {len(X.r_mpc)} runs, took "
         f"{sum(wm) / 3600:.1f}&#8202;h of six-thread time, and the three-pad grid, {len(X.r_mpc3)} runs, {sum(w3) / 3600:.1f}&#8202;h.</p>")
    s += table(["Controller", "Control rate, Hz", "Rollouts per replanning", "Threads", "Wall time per run, median, s",
                "Wall time per run, longest, s"], body, "Cost of the controllers on the servo plant.", key="cost")
    if X.timing:
        pm, pt = X.timing["models"]["pads"], X.timing["models"]["pt"]
        ro = lambda m, n: next(x["wall_s"] for x in m["rollout"] if x["nthread"] == n)  # noqa: E731
        s += (f"<p>The sphere pads make each physics step {pm['grip']['step_us'] / pt['grip']['step_us']:.1f} times as expensive as "
              f"point contact on the TPU block once all three fingers grip the cylinder (Table&#160;[[tab:padcost]], 16&#8202;mm "
              f"cylinder at \\(x_s\\) = 62.5&#8202;mm): {pm['grip']['ncon_mean']:.0f} contacts and {pm['grip']['nefc_mean']:.0f} "
              f"constraint rows against {pt['grip']['ncon_mean']:.0f} and {pt['grip']['nefc_mean']:.0f}, and "
              f"{pm['grip']['step_us']:.0f}&#8202;&#181;s per step against {pt['grip']['step_us']:.1f}, of which "
              f"{pm['grip']['fwd_position_us']:.0f}&#8202;&#181;s is the position stage (geom poses and collision detection) and "
              f"{pm['grip']['constraint_us']:.0f}&#8202;&#181;s the constraint solver. With the hand open the pads still cost "
              f"{pm['open']['step_us']:.0f}&#8202;&#181;s against {pt['open']['step_us']:.1f}, nearly all in the position stage, "
              f"which handles {pm['ngeom']:,} geoms against {pt['ngeom']}. One replanning of 64 plans of 0.5&#8202;s takes "
              f"{ro(pm, 12):.2f}&#8202;s on 12 threads with the pads and {ro(pt, 12):.2f}&#8202;s with point contact. On the six "
              f"threads of the planner runs it takes {ro(pm, 6):.2f}&#8202;s, {ro(pm, 6) / 0.02:.0f} times the 20&#8202;ms it "
              "plans for.</p>")
        body = []
        for lab, m in (("Sphere pads, 1&#8202;mm", pm), ("Point contact, TPU block", pt)):
            body.append([lab, f"{m['ngeom']:,}", f"{m['grip']['ncon_mean']:.0f}", f"{m['grip']['nefc_mean']:.0f}",
                         f"{m['open']['step_us']:.1f}", f"{m['grip']['step_us']:.1f}", f"{m['grip']['constraint_us']:.1f}",
                         f"{ro(m, 1):.2f}", f"{ro(m, 6):.2f}", f"{ro(m, 12):.2f}"])
        s += table(["Fingertip model", "Geoms", "Contacts in the grip", "Constraint rows in the grip", "Step, hand open, &#181;s",
                    "Step, three-finger grip, &#181;s", "Of which the solver, &#181;s", "64 plans of 0.5&#8202;s, 1 thread, s",
                    "6 threads, s", "12 threads, s"], body,
                   "Cost of a physics step and of one replanning&#8217;s rollouts with each fingertip model, CPU MuJoCo, "
                   "<code>scripts/hand_object_scale_mpc_timing.py</code>.", key="padcost")
        if X.v1_timing:
            q1 = X.v1_timing["models"]["pads"]
            s += (f"<p>On real_v1 the same grip costs {q1['grip']['step_us']:.0f}&#8202;&#181;s per step with "
                  f"{q1['grip']['ncon_mean']:.0f} contacts and {q1['ngeom']:,} geoms, and one replanning {ro(q1, 12):.2f}&#8202;s on "
                  f"12 threads, against {pm['grip']['step_us']:.0f}&#8202;&#181;s, {pm['grip']['ncon_mean']:.0f} contacts, "
                  f"{pm['ngeom']:,} geoms and {ro(pm, 12):.2f}&#8202;s on the CAD hand (16&#8202;mm cylinder at \\(x_s\\) = "
                  "62.5&#8202;mm on both hands).</p>")
    if X.gpu_check:
        g = [r for r in X.gpu_check if r.get("pad_filter")]
        big = max(g, key=lambda r: r["nworld"]) if g else None
        if big:
            raw = next((r for r in X.gpu_check if not r.get("pad_filter") and r["nworld"] == big["nworld"]), None)
            s += ("<p>The MuJoCo-Warp port (<code>scripts/hand_object_scale_mpc_gpu.py</code>) rolls out the plans as parallel "
                  "worlds on the GPU from the plant&#8217;s state, one CUDA graph per 10&#8202;ms of simulated time. "
                  + (f"With the planner&#8217;s model as compiled, {raw['nworld']:,} plans of 0.5&#8202;s took {raw['gpu_s']:.1f}&#8202;s "
                     f"against {raw['cpu_s_12threads']:.1f}&#8202;s on 12 CPU threads. " if raw else "")
                  + f"With the pads colliding only with the cylinder and the GPU solver stopped at tolerance "
                  f"10<sup>&#8722;6</sup> or 20 iterations, {big['nworld']:,} plans of 0.5&#8202;s take {big['gpu_s']:.2f}&#8202;s, "
                  f"{1e6 * big['gpu_s'] / (big['nworld'] * 500):.1f}&#8202;&#181;s per world-step and "
                  f"{big['cpu_s_12threads'] / big['gpu_s']:.1f} times the throughput of 12 CPU threads, and the GPU&#8217;s rollout "
                  f"costs agree with the CPU&#8217;s at a Pearson correlation of {big['cost_pearson']:.4f} and a rank correlation of "
                  f"{big['cost_spearman']:.4f} (Table&#160;[[tab:gpu]]). The contact sensors of Eq.&#160;(3) match at most 64 "
                  "contacts each by default in MuJoCo-Warp; the CAD hand&#8217;s link spheres exceed that, and the port sets 256.</p>")
        body = [[f"{r['nworld']:,}", "the cylinder" if r.get("pad_filter") else "every geom", f"{r['cpu_s_12threads']:.2f}",
                 f"{r['gpu_s']:.2f}", f"{r['cost_pearson']:.4f}", f"{r['cost_spearman']:.4f}", "yes" if r["best_same"] else "no"]
                for r in X.gpu_check]
        s += table(["Plans of 0.5&#8202;s", "Pads collide with", "CPU, 12 threads, s", "GPU, s", "Cost correlation, Pearson",
                    "Cost correlation, rank", "Same cheapest plan"], body,
                   "One replanning&#8217;s rollouts on the CPU and on the GPU from the same state, 25&#8202;mm cylinder at "
                   "\\(x_s\\) = 92.5&#8202;mm; GPU solver at tolerance 10<sup>&#8722;6</sup> or 20 iterations.",
                   text_cols=(1, 6), key="gpu")
    return s


def limitations(X: Data):  # noqa: C901
    items = [
        "MPC plans on the plant&#8217;s own model, so its scores are what the hand reaches with an exact model of itself.",
        "Only the servo plant and the 10, 16 and 25&#8202;mm cylinders are rerun on the CAD hand; the real_v1 page&#8217;s "
        "1&#8202;N&#8202;m-plant and sphere rows have no CAD counterpart.",
        "The CAD hand&#8217;s link spheres stand up to 6.7&#8202;mm off the printed parts (MorphIt fit, CAD finger page), so "
        "link contacts with the cylinder start early; the grasp candidates come from the scale study&#8217;s capsule "
        "clearance model with the CAD link lengths.",
        "MPC and pinned iCEM run on one grasp per cell, the one HOM scores best without the post; the three-pad MPC and "
        "pinned iCEM on one placement per cell, the rest on three.",
        "The planner comparison covers four cells at two placements; its medians are over eight runs per controller.",
        f"The first GPU iCEM runs overflowed MuJoCo-Warp&#8217;s 64 contact matches per sensor 51,458 times, which corrupted "
        f"the per-finger forces of the cost, and crashed on the eighth run; their {X.n_overflow} rows are left out and the "
        "planner tables give the rerun at 256 matches. GPU annealed sampling overflowed 40 times, all in its two runs on the 16&#8202;mm "
        "cylinder at \\(x_s\\) = 122.5&#8202;mm; those rows are kept.",
    ]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def open_items():
    items = [
        ("Bench cells from the CAD hand.", "The bench plan of 11 October picks its cells from real_v1&#8217;s HOM scores with the "
         "post, which reach 60&#176; in no cell on the CAD hand. Rebuild it from this page&#8217;s pinned-iCEM landscape "
         "(<code>planners/landscape_icem_pin.jsonl</code>) and export the servo targets of the chosen runs as open-loop plans for "
         "<code>scripts/real_v1_trajectory_clearance.py</code>; a cell is worth bench time if the open-loop replay keeps at least "
         "60&#176; on three pads in two of three placements."),
        ("Pinned iCEM at every placement.", "Run <code>SCALE_HAND=cad_v1 scripts/hand_object_scale_planners.py run --configs "
         "icem_pin --seeds 1 2</code> on the 39 cells (about 3&#8202;h on two six-thread shards; placement 0 took 1.4&#8202;h); "
         "the landscape holds if each cell&#8217;s median over three placements stays within 15&#176; of its placement-0 score."),
        ("Gradient-based MPC.", "Test <code>scripts/hand_object_scale_ilqr.py</code> with <code>hand_object_scale_planners.py run "
         "--configs ilqr --cells cylinder:25:0.125 --seeds 0 --set t_end=1.3</code>, then on the four planner cells; it is "
         "worth keeping if it holds three pads with no contact break at a command rate at or below the pinned samplers&#8217; "
         "and turns within 10&#176; of iCEM."),
        ("Planner with model error.", "Rerun HOM and the pinned planners with the plant&#8217;s sliding friction at 0.7 while "
         "the planner keeps 1.0 (<code>--variant mu0.7 --plant-set mu=0.7</code>), and plan on the real_v1 hand while the "
         "plant runs the CAD hand; a planner transfers if it keeps three pads in at least seven of eight runs."),
        ("Real-time MPC.", "Measure the pinned iCEM with the planner on point contact (the plant keeps the pads), a 0.25&#8202;s "
         "horizon, replanning every 40&#8202;ms and 32 samples, each alone and together, against the speed-up the cost "
         "section predicts. It is usable on the bench if one replanning fits in 20&#8202;ms on 12 threads and the four "
         "planner cells keep three pads."),
    ]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def lede(X: Data):
    n_cad = len({k for k in X.nopost if k[0] == "cylinder"})
    n_v1 = len({k for k in X.v1_nopost if k[0] == "cylinder"})
    s = (f"The CAD-model hand, whose fingers reach 10.05&#8202;mm further than real_v1&#8217;s and collide through their "
         f"printed links, has a feasible grasp in {n_cad} cells of the scale study&#8217;s diagonal layouts with 10, 16 "
         f"and 25&#8202;mm cylinders, against real_v1&#8217;s {n_v1}.")
    hom = [X.nopost[k]["score"] for k in X.nopost if k[0] == "cylinder"]
    hom1 = [X.v1_nopost[k]["score"] for k in X.v1_nopost if k[0] == "cylinder"]
    post = [X.post[k]["score"] for k in X.post if k[0] == "cylinder"]
    if hom:
        p60 = sum(1 for v in post if v >= 60)
        s += (f" HOM without the post turns the cylinder a median {deg(med(hom))} per cell (real_v1 {deg(med(hom1))}); with the "
              f"post, the protocol of the bench plan, it reaches 60&#176; in {'no cell' if p60 == 0 else f'{p60} cells'}.")
    if X.land and X.r_mpc3:
        rows = [X.land[k]["rows"][0] for k in X.land]
        three = sum(1 for r in rows if r["held"] and r["fingers_loaded_end"] == 3)
        m3 = [r for r in X.r_mpc3 if ("cylinder", float(r["d_mm"]), float(r["t"])) in X.land]
        s += (f" Pinned iCEM holds the cylinder on three pads in {three} of the {len(rows)} cells and turns it a median "
              f"{deg(med([r['held_turn_deg'] for r in rows]))}, against {deg(med([r['held_turn_deg'] for r in m3]))} for "
              f"unpinned three-pad MPC at the same budget, at a command rate of "
              f"{med([r['cmd_rate_rms'] for r in rows]):.2f}&#8202;rad/s against {med([r['cmd_rate_rms'] for r in m3]):.2f}.")
    if X.pl_cfgs and "icem_pin" in X.pl_cfgs and X.v1_pl.get("icem_pin"):
        s += (f" On the four planner cells it turns {deg(X.pl_stats('icem_pin')['turn'])} against "
              f"{deg(med([r['held_turn_deg'] for r in X.v1_pl['icem_pin']]))} on real_v1, and 1,024 rollouts on the GPU give "
              f"{deg(X.pl_stats('icem_gpu')['turn'])}." if "icem_gpu" in X.pl_cfgs else ".")
    return s


def byline():
    return ("CAD hand <code>assets/mjcf/experimental/20261013-cad_v1/</code> &#183; <code>SCALE_HAND=cad_v1</code> with "
            "<code>scripts/hand_object_scale_sim.py</code>, <code>hand_object_scale_mpc.py</code>, "
            "<code>hand_object_scale_planners.py</code>, <code>hand_object_scale_mpc_gpu.py</code> &#183; builder "
            "<code>scripts/scale_turn_cad_v1_page.py</code> &#183; rows <code>docs/experiments/20261014-scale_turn_cad_v1/</code>")


def main():
    X = Data()
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline(), "GLOSSARY": glossary()}
    v["QUESTION"] = question(X)
    v["CONTROLLERS"] = controllers(X)
    v["POST"] = post_section(X)
    v["LANDSCAPE"] = landscape_section(X)
    v["PLANNERS"] = planners_section(X)
    v["COMPARE"] = compare_section(X)
    v["FILMS"] = films_section(X)
    v["COST"] = cost_section(X)
    v["LIMITS"] = limitations(X)
    v["OPEN"] = open_items()
    v["LEDE"] = lede(X)
    v["FOOTER"] = ("<p>Rebuild: <code>.venv/bin/python scripts/scale_turn_cad_v1_page.py</code>. Rows in "
                   "<code>docs/experiments/20261014-scale_turn_cad_v1/</code>: <code>kin/reach.json</code>, "
                   "<code>hom_post_servo/</code>, <code>hom_postoff_servo/</code>, <code>mpc_servo/</code>, <code>mpc3_servo/</code> "
                   "(<code>turn.jsonl</code>, one line per run), <code>planners/turn.jsonl</code> (one line per run with its "
                   "20&#8202;ms log), <code>planners/timing.json</code>, <code>planners/gpu_check.jsonl</code>, "
                   "<code>planners/media/films.jsonl</code>, <code>media/films.jsonl</code> and <code>post_traces.json</code>; "
                   "<code>planners/landscape_icem_pin.jsonl</code> holds the pinned-iCEM landscape. Run recipe in "
                   "<code>run/</code>: <code>hom_grids.sh</code>, then <code>queue_cpu.sh</code> and <code>queue_gpu.sh</code> "
                   "side by side, then <code>reruns.sh</code> (GPU iCEM at 256 contact matches, pinned-iCEM landscape); logs in "
                   "<code>logs/20261014-scale_turn_cad_v1/</code>. The real_v1 rows of the comparison are in "
                   "<code>docs/experiments/20261013-scale_turn_controllers/</code>.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    miss = sorted(set(re.findall(r"\[\[((?:fig|tab|alg):[a-z0-9_]+)\]\]", t)) - set(REFS))
    if miss:
        raise SystemExit(f"unresolved references: {miss}")
    t = re.sub(r"\[\[((?:fig|tab|alg):[a-z0-9_]+)\]\]", lambda m: str(REFS[m.group(1)]), t)
    t, n, nb = render_tex(t)
    t = retro_style.apply(t)
    open(OUT, "w").write(t)
    print(f"formulas {n}, algorithms {nb}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
