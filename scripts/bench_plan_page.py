#!/usr/bin/env python3
r"""Build docs/experiments/20261011-bench_plan/20261011-bench_plan_finger_spacing.html and plan.json beside it.

    .venv/bin/python scripts/bench_plan_page.py

The bench plan of the hand-object scale study. It reads the simulation rows of 2026-10-08 in
docs/experiments/20261008-hand_object_scale/ (landscape.json, turn.jsonl, hold.jsonl, protocol.json, objects.json),
picks the bench cells by the rule stated on the page, and writes them with their grasps, simulated outcomes and bench
hours to plan.json, the input of the plan export. Every number in the prose is computed here. Mathematics is LaTeX
(\( \) inline) typeset to inline SVG by scripts/texsvg.py, cached beside the page.
"""
from __future__ import annotations

import collections
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import texsvg  # noqa: E402
import retro_style  # noqa: E402
from hand_object_scale_page import table, im, esc  # noqa: E402

SRC = os.path.join(ROOT, "docs/experiments/20261008-hand_object_scale")
D = os.path.join(ROOT, "docs/experiments/20261011-bench_plan")
REL = "docs/experiments/20261011-bench_plan"
OUT = os.path.join(D, "20261011-bench_plan_finger_spacing.html")
TPL = os.path.join(ROOT, "scripts/bench_plan_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
STUDY = "../20261008-hand_object_scale/20261008-finger_spacing_object_size.html"

BAND_DEG = 60.0           # a trial succeeds, and a layout is in the band, at a held turn of 60 deg or more
STEP_T = 0.125            # the simulation's layout step (3.75 mm of mount distance)
N_CELLS = 5               # layouts per cylinder: one wider than the band, the band, the rest on the compact side
N_TURN = 12               # turn trials per cell
N_BALLAST = 2             # hanging-mass series per cell
REF_TRIALS = 5            # trials of the previous day's band cell that open each bench day (drift)
STAGE1 = (10,)            # cylinder diameters, mm
STAGE2 = (16, 25)
CONTEXT = (6,)            # drawn in Figure 1, not planned
F_CAP = 20.0              # N, the disturbance ramp's cap (landscape.json meta)
COL = {6: "var(--ink3)", 10: "var(--s1)", 16: "var(--s2)", 25: "var(--s3)"}

EXTRA_CSS = """
.tw th{text-transform:none;letter-spacing:.01em}
table.gloss th{text-align:left;vertical-align:top;white-space:nowrap;padding-right:12px}
table.gloss td{white-space:normal}
td.wrap{white-space:normal;text-align:left}
figure.diagram svg{max-width:100%;height:auto}
ol.pred li, ol.steps li{margin:0 0 6px}
""" + texsvg.BLOCK_CSS + "\n"

FIG, TAB = [0], [0]


def figure(svg, caption):
    FIG[0] += 1
    return f'<figure class="diagram">{svg}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def tcap(text):
    TAB[0] += 1
    return f"<p class='tcap'><b>Table&#160;{TAB[0]}.</b> {text}</p>"


def f(x, nd=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "&#8211;"
    return P.num(x, f".{nd}f")


def rng(lo, hi, nd=0, unit=""):
    a, b = f(lo, nd), f(hi, nd)
    return (a if a == b else f"{a}&#8211;{b}") + unit


def render_tex(t):
    maths = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = texsvg.render(maths, cache_path=TEX_CACHE, scale=P.TEX_SCALE)
    it = iter(svgs)
    return P.TEX_RE.sub(lambda m: next(it), t), len(maths)


# ------------------------------------------------------------------------------------------ data

def jl(name):
    return [json.loads(x) for x in open(os.path.join(SRC, name)) if x.strip()]


def load():
    L = json.load(open(os.path.join(SRC, "landscape.json")))
    prot = json.load(open(os.path.join(SRC, "protocol.json")))
    objs = json.load(open(os.path.join(SRC, "objects.json")))
    objs = objs if isinstance(objs, list) else objs.get("objects", [])
    return L, prot, objs, jl("turn.jsonl"), jl("hold.jsonl")


def cylinder_cells(turn_rows, hold_rows):
    """(d, t) -> per-layout record: the turn grasp (largest median held turn over its placements), the hold grasp
    (largest mean pull-out along gravity), the spread of grasps, and the geometry."""
    turns = collections.defaultdict(lambda: collections.defaultdict(list))
    geo = {}
    for r in turn_rows:
        if r["shape"] != "cylinder" or r.get("model") != "pads" or r.get("status") != "ok":
            continue
        k = (r["d_mm"], r["t"])
        turns[k][r["cand"]].append(r["held_turn_deg"] if r.get("held") else 0.0)
        geo[k] = (r["x_sep_mm"], r["y_sep_mm"])
        geo[(k, r["cand"])] = (r["h_mm"], r["spread_mm"])
    pulls = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in hold_rows:
        if r["shape"] != "cylinder" or r.get("model") != "pads" or r.get("status") != "ok":
            continue
        k = (r["d_mm"], r["t"])
        if not r.get("held"):
            v = 0.0
        else:
            v = r["disturb"].get("F-z")
            v = F_CAP if v is None else v
        pulls[k][r["cand"]].append(v)
        geo[(k, r["cand"])] = (r["h_mm"], r["spread_mm"])
    out = {}
    for k, cands in turns.items():
        d, t = k
        meds = {c: statistics.median(v) for c, v in cands.items()}
        ct = max(meds, key=meds.get)
        pm = {c: statistics.fmean(v) for c, v in pulls.get(k, {}).items()}
        ch = max(pm, key=pm.get) if pm else None
        x_s, y_s = geo[k]
        s = x_s / 2
        out[k] = dict(
            d=d, t=t, x_s=x_s, y_s=y_s, s=s, standoff=s - d / 2,
            turn_cand=ct, turn_med=meds[ct], turn_pl=cands[ct], turn_succ=sum(x >= BAND_DEG for x in cands[ct]),
            n_cand=len(meds), n_cand_band=sum(m >= BAND_DEG for m in meds.values()),
            other_meds=sorted(m for c, m in meds.items() if m < BAND_DEG),
            turn_grasp=dict(zip(("h_mm", "spread_mm"), geo[(k, ct)])),
            hold_cand=ch, pull=pm.get(ch) if ch is not None else None,
            pull_turn_grasp=pm.get(ct),
            hold_grasp=dict(zip(("h_mm", "spread_mm"), geo[(k, ch)])) if ch is not None else None,
        )
    return out


def check_against_landscape(C, L):
    """The turn grasp's median must be the landscape's turn score of the same cell."""
    F = L["families"]["diag"]
    tag = {l["t"]: l["tag"] for l in F["layouts"]}
    worst = 0.0
    for (d, t), c in C.items():
        v = F["objects"].get(f"cylinder:{d:g}", {}).get(tag[t], {}).get("turn")
        if v is not None:
            worst = max(worst, abs(v - c["turn_med"]))
    if worst > 0.05:
        raise SystemExit(f"turn scores differ from landscape.json by {worst:.2f} deg")
    return worst


def per_object(C, d):
    return [C[k] for k in sorted(k for k in C if k[0] == d)]


def band(C, d):
    return [c for c in per_object(C, d) if c["turn_med"] >= BAND_DEG]


def plan_cells(C, d):
    """One layout wider than the band, the band's layouts, then the compact side, N_CELLS in all, at the simulation's
    own step."""
    b = band(C, d)
    lo, hi = min(c["t"] for c in b), max(c["t"] for c in b)
    ts = [round(lo - STEP_T, 3)] + [round(lo + i * STEP_T, 3) for i in range(int(round((hi - lo) / STEP_T)) + 1)]
    while len(ts) < N_CELLS:
        ts.append(round(ts[-1] + STEP_T, 3))
    return [C[(d, t)] for t in ts]


def fisher_two_sided(a, n1, b, n2):
    """Two-sided Fisher exact p for a successes of n1 against b of n2."""
    k, N = a + b, n1 + n2

    def pmf(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(N, k)
    p0 = pmf(a)
    return sum(pmf(x) for x in range(max(0, k - n2), min(k, n1) + 1) if pmf(x) <= p0 * (1 + 1e-9))


def hours(n_cells, n_days, meta):
    trials = n_cells * N_TURN * meta["t_trial_s"] / 3600
    ballast = n_cells * N_BALLAST * meta["t_ballast_s"] / 3600
    reconf = n_cells * meta["t_reconf_s"] / 3600
    day = n_days * (meta["t_day_overhead_s"] + REF_TRIALS * meta["t_trial_s"]) / 3600
    return dict(cells=n_cells, turn_trials=n_cells * N_TURN, ballast_series=n_cells * N_BALLAST, h_turn=trials,
                h_ballast=ballast, h_reconf=reconf, days=n_days, h_day=day, total=trials + ballast + reconf + day)


def old_protocol_stats(prot, L):
    """How the 8 October protocol brackets each simulated ridge, and where the cylinder ridges fall on its grid."""
    F = L["families"]["diag"]
    bench_t = prot["meta"]["bench_t"]
    n_two = n_all = off_grid = 0
    for ob, rg in F["ridges"].items():
        for task in ("hold", "turn"):
            if task not in rg:
                continue
            shape, dd = ob.split(":")
            cells = [c for c in prot["cells"] if c["shape"] == shape and f"{c['d_mm']:g}" == dd
                     and task in c["near_ridge_of"]]
            if not cells:
                continue
            n_all += 1
            n_two += len(cells) <= 2
            off_grid += min(abs(rg[task]["at"] - b) for b in bench_t) > 0.06
    return dict(n_all=n_all, n_two=n_two, off_grid=off_grid, cells=len(prot["cells"]), hours=prot["hours"]["total"],
                objects=len({(c["shape"], c["d_mm"]) for c in prot["cells"]}),
                turn_trials=sum(c.get("n_turn", 0) for c in prot["cells"]),
                ballast=sum(c.get("n_ballast", 0) for c in prot["cells"]), days=len(prot["days"]))


# ------------------------------------------------------------------------------------------ figure

def pull_spread(C, ds=(6, 10, 16, 25), lo=15.0, hi=45.0):
    """Largest difference between the cylinders' pull-out curves (linear interpolation against the standoff) on a
    1 mm grid of the shared standoff range."""
    curves = {}
    for d in ds:
        pts = sorted((c["standoff"], c["pull"]) for c in per_object(C, d) if c["pull"] is not None)
        curves[d] = pts

    def at(pts, x):
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return None
    worst = 0.0
    for i in range(int(lo), int(hi) + 1):
        v = [at(p, float(i)) for p in curves.values()]
        v = [x for x in v if x is not None]
        if len(v) == len(curves):
            worst = max(worst, max(v) - min(v))
    return worst


def svg_landscape(C):
    W, H = 990, 380
    out = P._svg_open(W, H, "Held turn of the turn grasp and pull-out force of the hold grasp against the standoff of the "
                            "mounts, for cylinders of 6, 10, 16 and 25 mm; the planned bench layouts are filled.")
    pw, ph, top = 390, 260, 40
    xs, xt = (10, 56), list(range(10, 56, 10))
    panels = (("turn_med", "Held turn of the turn grasp, deg", (0, 100), list(range(0, 101, 20))),
              ("pull", "Pull-out force of the hold grasp, N", (0, 12), list(range(0, 13, 2))))
    planned = {(c["d"], c["t"]) for d in STAGE1 + STAGE2 for c in plan_cells(C, d)}
    for k, (key, ylab, ys, yt) in enumerate(panels):
        x0 = 70 + k * (pw + 100)
        fx, fy = P._panel(out, x0, top, pw, ph, xs, ys, xt, yt, "Standoff of the mounts, mm", ylab)
        if key == "turn_med":
            out.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{fy(BAND_DEG):.1f}" y2="{fy(BAND_DEG):.1f}" '
                       f'style="stroke:var(--ink3);stroke-width:1.5;stroke-dasharray:6 5"/>'
                       f'<text x="{x0 + pw - 4}" y="{fy(BAND_DEG) - 6:.1f}" text-anchor="end" '
                       f'style="fill:var(--ink3)">band, {BAND_DEG:.0f}&#176;</text>')
        for d in CONTEXT + STAGE1 + STAGE2:
            cs = [c for c in per_object(C, d) if c[key] is not None and xs[0] <= c["standoff"] <= xs[1]]
            P._path(out, fx, fy, [(c["standoff"], c[key]) for c in cs], COL[d], dashed=d in CONTEXT,
                    width=2.4 if d not in CONTEXT else 1.6)
            for c in cs:
                on = (d, c["t"]) in planned
                tip = (f"{d} mm cylinder, t = {c['t']:g}, standoff {c['standoff']:.2f} mm: "
                       + (f"{c['turn_med']:.1f} deg" if key == "turn_med" else f"{c['pull']:.2f} N")
                       + (" (planned)" if on else ""))
                P._marker(out, fx(c["standoff"]), fy(c[key]), COL[d], hollow=not on, r=5.0 if on else 3.4, title=tip)
    out.append("</svg>")
    leg = P._legend_html([(f"{d} mm cylinder{' (not planned)' if d in CONTEXT else ''}", COL[d], d in CONTEXT, "circle")
                          for d in CONTEXT + STAGE1 + STAGE2])
    return "".join(out) + leg


# ------------------------------------------------------------------------------------------ sections

def glossary():
    G = [
        ("Layout " + im("t"), "the finger-base positions of the symmetric tripod in the palm frame, in mm: thumb at "
         + im(r"(-x_s/2,\,0)") + ", index and middle at " + im(r"(x_s/2,\,\pm y_s/2)") + ", with "
         + im(r"x_s = 100 - 60\,t") + " and " + im(r"y_s = 110 - 60\,t") + ". Larger " + im("t") + " is more compact; the "
         "simulation steps " + im("t") + " by 0.125."),
        ("Mount distance " + im("s"), im("s = x_s/2") + ": distance from the cylinder&#8217;s axis, which lies along the "
         "pair axis, to the thumb mount and to the line of the pair mounts, in mm. One step of " + im("t") + " is "
         "3.75&#8202;mm."),
        ("Standoff " + im("s - r"), "mount distance minus the cylinder&#8217;s radius " + im("r") + ", in mm."),
        ("Turn grasp, hold grasp", "of the study&#8217;s 8&#8211;9 candidate grasps per layout (palm height and straddle), "
         "the one with the largest median held turn over three placements, and the one with the largest pull-out "
         "force."),
        ("HOM controller", "the relative contact-velocity controller of Wang, Oh and Pollard (arXiv 2609.25619) in "
         "<code>scripts/hom_turn3.py</code>, turning at 30&#176;/s toward 90&#176; with pad force regulated to 2&#8202;N."),
        ("Held turn", "rotation of the cylinder&#8217;s axis from horizontal toward vertical after a 2&#8202;s hold, in "
         "degrees; 0 unless two fingers still press with at least the object&#8217;s weight. The bench measures it with "
         "the AprilTag tracker."),
        ("Success, band", f"a trial succeeds at a held turn of {BAND_DEG:.0f}&#176; or more. A cylinder&#8217;s band is "
         f"the set of layouts whose turn grasp has a median held turn of {BAND_DEG:.0f}&#176; or more."),
        ("Pull-out force", "the smallest load along gravity that moves the held cylinder 3&#8202;mm, in N: a force ramped "
         "at 4&#8202;N/s in simulation, steel washers in 5&#8202;g steps on a hook under the cylinder on the bench."),
        ("Open-loop replay", "the bench sends a simulated turn&#8217;s joint commands without force feedback; the "
         "fingertips have no force sensors."),
    ]
    rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in G)
    return f"<div class='tw'><table class='gloss'><tbody>{rows}</tbody></table></div>"


def science(C, L):
    F = L["families"]["diag"]
    fk, ft = F["fits"]["cylinder:kin"], F["fits"]["cylinder:turn"]
    small = [d for d in (6, 10, 16, 25) if band(C, d)]
    inb = [c for d in small for c in band(C, d)]
    edge = [c for d in STAGE1 + STAGE2 for c in plan_cells(C, d) if c["turn_med"] < BAND_DEG]
    big = max(c["turn_med"] for d in (40, 50, 63, 80) for c in per_object(C, d))
    centres = {d: statistics.fmean(c["standoff"] for c in band(C, d)) for d in small}
    p10 = plan_cells(C, 10)
    peaks = [max((c for c in per_object(C, d) if c["pull"] is not None), key=lambda c: c["pull"]) for d in small]
    b10 = band(C, 10)
    e10 = [c for c in p10 if c["turn_med"] < BAND_DEG]
    others = [m for c in inb for m in c["other_meds"]]
    worst = min(inb, key=lambda c: min(c["turn_pl"]))
    nb = [c["n_cand_band"] for c in inb]
    nc = [c["n_cand"] for c in inb]
    return (
        "<p>The study asks how the finger spacing that holds an object best, and the spacing that turns it furthest, "
        "move with the object&#8217;s size, and whether the simulator predicts where they lie. For a task " + im("T")
        + " with score " + im("J_T") + " the quantity is " + im(r"s^*_T(r) = \arg\max_s J_T(s, r)") + ". Borr&#224;s and "
        "Dollar (2015) and Hota and Kumar (2019) answer the kinematic part, and the simulation of 8 October reproduces "
        "their result on SR2: the fixed-contact workspace is largest at "
        + im(fr"s = {fk['alpha']:.2f}\,r + {fk['beta_mm']:.1f}\,\text{{mm}}") + ", about half a finger outside the "
        "object. The bench can answer what kinematics cannot: whether holding and turning select different spacings on "
        f"a physical hand, and whether the simulated landscape agrees with it at the same layouts (<a href='{STUDY}'>study "
        "page</a>). The simulation makes three predictions for cylinders (Figure&#160;1).</p>"
        "<ol class='pred'>"
        f"<li><b>Turn band.</b> On the 10&#8202;mm cylinder the turn grasp turns "
        f"{rng(min(c['turn_med'] for c in b10), max(c['turn_med'] for c in b10), unit='&#176;')} at standoffs of "
        f"{rng(min(c['standoff'] for c in b10), max(c['standoff'] for c in b10), unit='&#8202;mm')} and "
        f"{rng(min(c['turn_med'] for c in e10), max(c['turn_med'] for c in e10), unit='&#176;')} one layout outside them. "
        f"Cylinders of {small[0]}&#8211;{small[-1]}&#8202;mm have a band; cylinders of 40&#8202;mm and more turn at most "
        f"{big:.0f}&#176; at any layout.</li>"
        f"<li><b>Scaling.</b> The band&#8217;s centre stays at a standoff of "
        f"{rng(min(centres.values()), max(centres.values()), unit='&#8202;mm')} from the {small[0]} to the "
        f"{small[-1]}&#8202;mm cylinder, so its mount distance grows one millimetre per millimetre of radius (fit of "
        "8 October: " + im(fr"s^*_\text{{turn}} = {ft['alpha']:.2f}\,r + {ft['beta_mm']:.1f}\,\text{{mm}}") + ").</li>"
        f"<li><b>Pull-out.</b> The hold grasp&#8217;s pull-out force rises as the mounts close in, on the 10&#8202;mm "
        f"cylinder from {f(p10[0]['pull'])}&#8202;N at a {f(p10[0]['standoff'], 0)}&#8202;mm standoff to "
        f"{f(p10[-1]['pull'])}&#8202;N at {f(p10[-1]['standoff'], 0)}&#8202;mm, and peaks at "
        f"{rng(min(c['pull'] for c in peaks), max(c['pull'] for c in peaks), nd=1, unit='&#8202;N')} near a standoff of "
        f"{rng(min(c['standoff'] for c in peaks), max(c['standoff'] for c in peaks), unit='&#8202;mm')} on all four "
        "cylinders. Holding along gravity therefore favours standoffs well inside the band, and the two tasks select "
        f"different spacings. Against the standoff the four pull-out curves lie within {f(pull_spread(C))}&#8202;N of "
        "each other.</li></ol>"
        f"<p>The band rests on one or two grasps per layout: at each band layout {rng(min(nb), max(nb))} of the "
        f"{rng(min(nc), max(nc))} candidates turn {BAND_DEG:.0f}&#176; or more, and the median of the others is "
        f"{statistics.median(others):.0f}&#176;. The turn grasp&#8217;s three placements also differ, from "
        f"{min(worst['turn_pl']):.0f}&#176; to {max(worst['turn_pl']):.0f}&#176; in the widest case ({worst['d']:g}&#8202;mm, "
        + im(f"t = {worst['t']:g}") + "). The bench outcome is therefore expected to be bimodal, and the plan counts each "
        "trial as a success or a stall.</p>"
    ), dict(centres=centres, inb=inb, edge=edge, big=big, b10=b10, e10=e10)


def old_section(S, L):
    F = L["families"]["diag"]
    tag = {l["t"]: l["tag"] for l in F["layouts"]}
    rg = F["ridges"]["cylinder:25"]["hold"]
    t0 = min(tag, key=lambda t: abs(t - rg["at"]))
    hold = {t: F["objects"]["cylinder:25"][tag[t]]["hold"] for t in (t0 - 0.125, t0, t0 + 0.125)}
    every = "each" if S["n_two"] == S["n_all"] else f"{S['n_two']}"
    return (
        f"<p>The protocol of 8 October measures {S['cells']} cells over {S['objects']} objects in "
        f"{f(S['hours'])}&#8202;h. Three properties keep it from testing these predictions.</p><ol class='steps'>"
        f"<li>Its bench grid takes every second simulated layout, so {every} of its {S['n_all']} ridges is bracketed by at "
        "most two bench layouts, and two values cannot locate a peak. The hold ridge of the 25&#8202;mm cylinder lies at "
        + im(f"t = {t0:g}") + f" with a simulated score of {f(hold[t0])}&#8202;N; the bench layouts beside it score "
        f"{f(hold[t0 - 0.125])} and {f(hold[t0 + 0.125])}&#8202;N.</li>"
        "<li>Its hold score is the weakest of six pull directions, but the hanging mass loads only along gravity, where "
        "the simulated pull-out force rises steadily toward the compact side (prediction&#160;3).</li>"
        "<li>The bench replays the simulated joint commands open loop, but every simulated turn ran the closed-loop "
        "HOM controller, and no open-loop replay has been simulated.</li></ol>"
    )


def stage0_section(objs):
    rec = {o["d_mm"]: o for o in objs if o["shape"] == "cylinder"}
    recipes = "; ".join(f"{d}&#8202;mm: {esc(rec[d]['recipe'])}, {rec[d]['mass_g']:.1f}&#8202;g"
                        for d in STAGE1 + STAGE2 if d in rec)
    n_plans = 2 * N_CELLS * len(STAGE1 + STAGE2)
    return (
        "<p>Four checks need the CPU only, about a day of work, and decide whether a bench turn can test the "
        "simulation.</p><ol class='steps'>"
        "<li><b>Plan export.</b> A new <code>scripts/hand_object_scale_export.py</code> writes the turn and hold grasps "
        f"of every cell in <code>{REL}/plan.json</code> as deploy plans, {n_plans} in all, and runs the clearance gate "
        "(<code>real_v1_trajectory_clearance.py</code>, chord and CSV path) and <code>HandPlan.validate()</code> on each. "
        "A cell whose plan fails moves to the neighbouring layout.</li>"
        "<li><b>Open-loop replay.</b> Replay each simulated turn&#8217;s joint commands on the same plant without force "
        "feedback. If the band disappears, Stage&#160;1 measures pull-out only.</li>"
        "<li><b>Placement tolerance.</b> Repeat the replay at object offsets of &#177;2&#8202;mm and yaws of "
        "&#177;3&#176;, the error of placing the cylinder by hand; the success fraction per cell is the number the bench "
        "rate is compared with.</li>"
        f"<li><b>Mass.</b> Print and weigh the three cylinders ({recipes}) and re-run the "
        f"{N_CELLS * len(STAGE1 + STAGE2)} cells at the measured masses.</li></ol>"
    )


def cell_rows(cells):
    rows = []
    for c in cells:
        pl = ", ".join(f"{x:.0f}" for x in c["turn_pl"])
        rows.append([f"{c['t']:g}", f"{c['x_s']:.1f} / {c['y_s']:.1f}", f(c["standoff"], 2),
                     f"{c['turn_med']:.0f} ({pl})", f"{c['turn_succ']} of {len(c['turn_pl'])}",
                     f(c["pull"]), str(N_TURN), str(N_BALLAST)])
    return rows


HEAD = ["Layout " + im("t"), im("x_s") + " / " + im("y_s") + ", mm", "Standoff, mm",
        "Sim held turn, deg (placements)", "Sim successes", "Sim pull-out, N", "Turn trials", "Mass series"]


def stage1_section(C, H1):
    cells = plan_cells(C, 10)
    nb = sum(c["turn_med"] >= BAND_DEG for c in cells)
    p = fisher_two_sided(8, N_TURN, 1, N_TURN)
    return (
        f"<p>Stage&#160;1 runs the 10&#8202;mm cylinder at {N_CELLS} layouts spaced by the simulation&#8217;s own step "
        "(3.75&#8202;mm of mount distance): one wider than the band, the band&#8217;s "
        f"{nb} layouts and one more compact (Table&#160;1). Each cell gets {N_TURN} turn trials with its turn grasp and "
        f"{N_BALLAST} hanging-mass series with its hold grasp, in random order within the day. With {N_TURN} trials per "
        f"cell, 8 successes against 1 differ at " + im(f"p = {p:.3f}") + " (Fisher&#8217;s exact test, two-sided). It "
        f"takes {f(H1['total'])}&#8202;h on one bench day.</p>"
        + tcap(f"Stage&#160;1 cells on the 10&#8202;mm cylinder. Sim held turn: the turn grasp&#8217;s median over three "
               "placements (object offset 2&#8202;mm, yaw 2&#176;), with the three values; sim successes: placements at "
               f"{BAND_DEG:.0f}&#176; or more; sim pull-out: the hold grasp.")
        + table(HEAD, cell_rows(cells))
        + "<p><b>Decision.</b> Stage&#160;2 starts if a band cell succeeds in 8 or more of its 12 trials, both edge cells "
        "in 2 or fewer, and the mean pull-out force rises across the five cells (rank correlation with the standoff of "
        "&#8722;0.8 or lower). Otherwise the turn trials stop; the bench joint readbacks and tracker traces are compared "
        "with the open-loop replay of Stage&#160;0, and the pull-out trend is reported alone.</p>"
    )


def stage2_section(C, H2):
    c10 = statistics.fmean(c["s"] for c in band(C, 10))
    c25 = statistics.fmean(c["s"] for c in band(C, 25))
    rows = []
    for d in STAGE2:
        rows.append(f"{d}&#8202;mm cylinder")
        rows += cell_rows(plan_cells(C, d))
    return (
        f"<p>Stage&#160;2 runs the 16 and 25&#8202;mm cylinders at five layouts each by the same rule (Table&#160;2) and "
        f"tests prediction&#160;2. The band of the 25&#8202;mm cylinder should sit {f(c25 - c10)}&#8202;mm of mount "
        "distance wider than that of the 10&#8202;mm cylinder, at the same standoff. A band at the same mount distance "
        "for all three diameters would mean that the hand, and not the object, sets the spacing of the turn. "
        f"Stage&#160;2 takes {f(H2['total'])}&#8202;h on two bench days.</p>"
        + tcap("Stage&#160;2 cells, columns as in Table&#160;1.")
        + table(HEAD, rows)
    )


def time_section(H1, H2, S):
    def row(name, h):
        return [name, str(h["cells"]), f"{h['turn_trials']} ({f(h['h_turn'])}&#8202;h)",
                f"{h['ballast_series']} ({f(h['h_ballast'])}&#8202;h)", f"{h['days']} ({f(h['h_day'])}&#8202;h)",
                f(h["total"])]
    tot = {k: H1[k] + H2[k] for k in H1}
    return (
        tcap("Bench time. A day opens with homing, the tracker check and setup (45&#8202;min) and five trials of the "
             "previous day&#8217;s band cell, which measure drift between days; turn trials take 90&#8202;s, mass series "
             "6&#8202;min, a layout change 30&#8202;s (protocol.json of 8 October).")
        + table(["Stage", "Cells", "Turn trials", "Mass series", "Days (overhead)", "Hours"],
                [row("1: 10&#8202;mm cylinder", H1), row("2: 16 and 25&#8202;mm cylinders", H2), row("Total", tot),
                 ["Protocol of 8 October", str(S["cells"]), str(S["turn_trials"]), str(S["ballast"]), str(S["days"]),
                  f(S["hours"])]])
    )


def outcomes_section():
    rows = [
        ["Band and rising pull-out on all three cylinders, bands at one standoff",
         "Holding and turning select different spacings on the hand, and the turn&#8217;s spacing scales with the object",
         "Paper contributions 3 and 4 as planned; spheres next (Future work)"],
        ["Band on the 10&#8202;mm cylinder only", "The band exists, but its location does not follow the radius on the hand",
         "Compare bench and simulated pad positions (tracker) at the 16 and 25&#8202;mm band cells before more objects"],
        ["No band, rising pull-out", "The open-loop turn does not transfer at these grasps; the hold trend stands",
         "The paper reports the simulated turn landscape with the bench pull-out; bench turns wait for closed-loop control"],
        ["Neither", "The bench grasp differs from the simulated one",
         "Measure grip (servo load readbacks) and pad placement before more bench time"],
    ]
    return (tcap("Outcomes after Stage&#160;2 and the step each one leads to.")
            + table(["Outcome", "What it shows", "Next"],
                    [[(a, "wrap"), (b, "wrap"), (c, "wrap")] for a, b, c in rows]))


def next_section(C):
    big = [d for d in (40, 50) if not band(C, d)]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in [
        "<b>Spheres.</b> The simulation turns spheres best at the wide edge of each feasible band and holds them best at "
        "the compact edge; a bench test takes both edges for the 16&#8211;50&#8202;mm spheres (cells in "
        "<code>protocol.json</code> of 8 October).",
        f"<b>Large cylinders.</b> The {' and '.join(str(d) for d in big)}&#8202;mm cylinders have no band in simulation; "
        "three layouts each test that prediction once Stage&#160;2 has shown the band.",
        "<b>Closed-loop turn.</b> Pad force estimated from the servo load readbacks (0.0186&#176; per load unit) would "
        "let the HOM controller run on the hand instead of an open-loop replay.",
        "<b>Two-dimensional map.</b> The 9&#8202;&#215;&#8202;9 grid of " + im("x_s") + " and " + im("y_s") + " (about "
        "3 CPU-hours) tests whether the band depends on " + im("x_s") + " alone.",
    ]) + "</ul>"


def lede(C, H1, H2, sci):
    b10, e10 = sci["b10"], sci["e10"]
    p10 = plan_cells(C, 10)
    return ("The bench week should test three predictions of the 8 October simulation. On a 10&#8202;mm cylinder the "
            f"HOM controller turns it {rng(min(c['turn_med'] for c in b10), max(c['turn_med'] for c in b10), unit='&#176;')} "
            f"toward vertical when the finger mounts stand {rng(min(c['standoff'] for c in b10), max(c['standoff'] for c in b10), unit='&#8202;mm')} "
            f"off its surface, and {rng(min(c['turn_med'] for c in e10), max(c['turn_med'] for c in e10), unit='&#176;')} "
            "one layout outside that band. On the 16 and 25&#8202;mm cylinders the band keeps its standoff, so it moves "
            f"outward with the radius. The pull-out force along gravity rises from {f(p10[0]['pull'])} to "
            f"{f(p10[-1]['pull'])}&#8202;N as the mounts close in. The plan spends {f(H1['total'])}&#8202;h on the "
            f"10&#8202;mm cylinder at five spacings, and {f(H2['total'])}&#8202;h on the other two only if the band "
            "appears.")


def write_plan(C, H1, H2):
    cells = []
    for stage, ds in ((1, STAGE1), (2, STAGE2)):
        for d in ds:
            for c in plan_cells(C, d):
                cells.append(dict(stage=stage, shape="cylinder", d_mm=d, t=c["t"], x_sep_mm=c["x_s"], y_sep_mm=c["y_s"],
                                  mount_distance_mm=c["s"], standoff_mm=round(c["standoff"], 3),
                                  in_band=c["turn_med"] >= BAND_DEG,
                                  turn_grasp=dict(cand=c["turn_cand"], **c["turn_grasp"]),
                                  hold_grasp=dict(cand=c["hold_cand"], **(c["hold_grasp"] or {})),
                                  sim_turn_deg=round(c["turn_med"], 2), sim_turn_placements_deg=c["turn_pl"],
                                  sim_successes=c["turn_succ"], sim_pull_N=round(c["pull"], 3),
                                  n_turn=N_TURN, n_ballast=N_BALLAST))
    plan = dict(meta=dict(date="2026-10-11", source="docs/experiments/20261008-hand_object_scale", band_deg=BAND_DEG,
                          rule="one layout wider than the band, the band, then the compact side; five layouts at the "
                               "simulation's step of 0.125 in t",
                          decision="stage 2 if a band cell succeeds in >= 8 of 12, both edge cells in <= 2 of 12, and "
                                   "pull-out rank correlation with standoff <= -0.8"),
                hours=dict(stage1=H1, stage2=H2), cells=cells)
    with open(os.path.join(D, "plan.json"), "w") as fh:
        json.dump(plan, fh, indent=1)
    return len(cells)


def main():
    os.makedirs(D, exist_ok=True)
    L, prot, objs, turn_rows, hold_rows = load()
    C = cylinder_cells(turn_rows, hold_rows)
    worst = check_against_landscape(C, L)
    meta = prot["meta"]
    H1 = hours(N_CELLS * len(STAGE1), 1, meta)
    H2 = hours(N_CELLS * len(STAGE2), 2, meta)
    S = old_protocol_stats(prot, L)
    sci, sci_d = science(C, L)
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": "2026-10-11"}
    v["GLOSSARY"] = glossary()
    v["SCIENCE"] = sci + figure(svg_landscape(C),
                                "Simulated held turn of each layout&#8217;s turn grasp (left) and pull-out force of its "
                                "hold grasp (right) against the standoff of the mounts, for four cylinders. Filled "
                                "markers are the planned bench layouts. The dashed line marks the band threshold. Hover "
                                "a marker for its layout and value.")
    v["OLD"] = old_section(S, L)
    v["STAGE0"] = stage0_section(objs)
    v["STAGE1"] = stage1_section(C, H1)
    v["STAGE2"] = stage2_section(C, H2)
    v["TIME"] = time_section(H1, H2, S)
    v["OUTCOMES"] = outcomes_section()
    v["NEXT"] = next_section(C)
    v["LEDE"] = lede(C, H1, H2, sci_d)
    n = write_plan(C, H1, H2)
    v["FOOTER"] = (f"<p>Rebuild: <code>.venv/bin/python scripts/bench_plan_page.py</code>; it writes "
                   f"<code>{REL}/plan.json</code> ({n} cells). Simulation rows: "
                   "<code>docs/experiments/20261008-hand_object_scale/</code> (<code>landscape.json</code>, "
                   "<code>turn.jsonl</code>, <code>hold.jsonl</code>, <code>protocol.json</code>, "
                   f"<code>objects.json</code>); turn scores agree with <code>landscape.json</code> to {worst:.2f}&#176;.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    t, n_math = render_tex(t)
    open(OUT, "w").write(retro_style.apply(t))
    print(f"formulas {n_math}; plan cells {n}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
