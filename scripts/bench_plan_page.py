#!/usr/bin/env python3
r"""Build docs/experiments/20261011-bench_plan/20261011-bench_plan_finger_spacing.html and plan.json beside it.

    .venv/bin/python scripts/bench_plan_page.py

The bench plan of the hand-object scale study, built on the simulation rows in ROWS: the servo-limited plant
(scripts/hand_object_scale_sim.py run --cells diag --what hold,turn --seeds 0 1 2 --plant PLANT --set kp_model=0.72
--out-dir ROWS --only cylinder:10 cylinder:16 cylinder:25). The 1 N m plant of the plan's first version is the
comparison: its turn rows of 8 October in SRC, and its hold rows rerun with the drop load in WORK (the same command
with --what hold, the default plant and --out-dir WORK). landscape.json, protocol.json and objects.json come from SRC.
The builder picks the bench cells by the rules stated on the page and writes them with their grasps, simulated outcomes
and bench hours to plan.json, the input of the plan export. Every number in the prose is computed here. Mathematics is
LaTeX (\( \) inline) typeset to inline SVG by scripts/texsvg.py, cached beside the page.
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
import statistics
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import texsvg  # noqa: E402
import retro_style  # noqa: E402
import hand_object_scale_kin as K  # noqa: E402
from hand_object_scale_page import table as _table, im, esc  # noqa: E402

SRC = os.path.join(ROOT, "docs/experiments/20261008-hand_object_scale")
D = os.path.join(ROOT, "docs/experiments/20261011-bench_plan")
ROWS = os.path.join(D, "servo_plant")      # the simulation rows the plan is built on
WORK = os.path.join(D, "work_plant")       # hold rows of the 1 N m plant with the drop load, for the comparison
POLICIES = os.path.join(ROOT, "docs/experiments/20261008-contact_model_policies/final_eval.jsonl")
REPLAYS = os.path.join(ROOT, "docs/experiments/20261006-servo_recalibration/replays.jsonl")
REL = "docs/experiments/20261011-bench_plan"
OUT = os.path.join(D, "20261011-bench_plan_finger_spacing.html")
TPL = os.path.join(ROOT, "scripts/bench_plan_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
STUDY = "../20261008-hand_object_scale/20261008-finger_spacing_object_size.html"
SYSID = "../20260902-servo-sysid/20260902-servo_sysid.html"
POLICY_PAGE = "../20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html"

PLANT = "kp0.72_kv0_fr0.18_fl0_dp0.0144"     # the servo plant of ROWS
PLANT_1NM = "kp4_kv0_fr1_fl0_dp0.08"         # the 1 N m plant of SRC and WORK
STALL_NM = 0.23           # SCS0009 stall torque at 6 V (2.3 kg cm, seller datasheet), N m
TRIP_FRAC = 0.8           # the overload protection cuts the drive to 20 % after a sustained load above 80 % of stall
DEG_PER_LOAD = 0.0186     # position error per unit of the load register (servo study of 2 September, 236 runs), deg
LOAD_FULL = 1000          # the load register at full drive
N_SYSID_RUNS = 236

BAND_DEG = 60.0           # a trial succeeds, and a layout is in the band, at a held turn of 60 deg or more
STEP_T = 0.125            # the simulation's layout step (3.75 mm of mount distance)
N_CELLS = 5               # layouts per cylinder: one wider than the band, the band, the rest on the compact side
N_TURN = 12               # turn trials per cell
N_BALLAST = 2             # hanging-mass series per cell
REF_TRIALS = 5            # trials of the previous day's band cell that open each bench day (drift)
PASS_BAND, PASS_EDGE = 8, 2   # Stage 2 starts if a band cell succeeds in >= 8 of 12 and each edge cell in <= 2
RHO_MAX = -0.8            # ... and the drop load's rank correlation with the standoff is at most this
STAGE1 = (10,)            # cylinder diameters, mm
STAGE2 = (16, 25)
CYL = STAGE1 + STAGE2
F_CAP = 20.0              # N, the disturbance ramp's cap (hand_object_scale_sim.F_CAP)
PUSH_STEP_N, PUSH_HOLD_S = 0.25, 3       # push test: force step, hold per step
PUSH_SERIES, T_PUSH_SERIES_S, T_PUSH_SETUP_S = 6, 240, 900   # three series per finger on two fingers
COL = {10: "var(--s1)", 16: "var(--s2)", 25: "var(--s3)"}
JNAMES = ("yaw", "MCP", "PIP")

EXTRA_CSS = """
.tw th{text-transform:none;letter-spacing:.01em}
table.gloss th{text-align:left;vertical-align:top;white-space:nowrap;padding-right:12px}
table.gloss td{white-space:normal}
td.wrap{white-space:normal;text-align:left}
figure.diagram svg{max-width:100%;height:auto}
ol.pred li, ol.steps li{margin:0 0 6px}
table.cells th, table.cells td{padding:8px 9px}
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


WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()


def word(n, cap=False):
    w = WORDS[n] if 0 <= n < len(WORDS) else str(n)
    return w.capitalize() if cap else w


def table(head, rows, cls=""):
    return _table(head, rows, cls)


def render_tex(t):
    maths = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = texsvg.render(maths, cache_path=TEX_CACHE, scale=P.TEX_SCALE)
    it = iter(svgs)
    return P.TEX_RE.sub(lambda m: next(it), t), len(maths)


def plant(spec):
    """Stiffness (N m/rad), torque limit (N m) and time constant (s) of a plant spec such as PLANT."""
    v = {}
    for tok in spec.split("_"):
        k = re.match(r"[a-z]+", tok).group()
        v[k] = float(tok[len(k):])
    return dict(kp=v["kp"], fr=v["fr"], tau=v["dp"] / v["kp"])


# ------------------------------------------------------------------------------------------ data

def jl(path):
    return [json.loads(x) for x in open(path) if x.strip()] if os.path.exists(path) else []


def load():
    L = json.load(open(os.path.join(SRC, "landscape.json")))
    prot = json.load(open(os.path.join(SRC, "protocol.json")))
    objs = json.load(open(os.path.join(SRC, "objects.json")))
    objs = objs if isinstance(objs, list) else objs.get("objects", [])
    servo = (jl(os.path.join(ROWS, "turn.jsonl")), jl(os.path.join(ROWS, "hold.jsonl")))
    work = (jl(os.path.join(SRC, "turn.jsonl")), jl(os.path.join(WORK, "hold.jsonl")))
    return L, prot, objs, servo, work, jl(os.path.join(SRC, "hold.jsonl"))


def gravity(r):
    """(3 mm threshold, drop load) along gravity of one hold row, N: 0 when the grasp did not hold, F_CAP when the ramp
    reached its cap; the drop load is None in rows without it."""
    if not r.get("held"):
        return 0.0, 0.0
    p = r["disturb"].get("F-z")
    p = F_CAP if p is None else p
    if "lost" not in r:
        return p, None
    q = r["lost"].get("F-z")
    return p, F_CAP if q is None else q


def ok_cyl(r, ds=CYL):
    return (r["shape"] == "cylinder" and r["d_mm"] in ds and r.get("model") == "pads" and r.get("status") == "ok")


def large_without_band(turn_rows, ds=(40, 50)):
    """The cylinders of `ds` whose every layout's best grasp turns less than BAND_DEG in median."""
    med = collections.defaultdict(list)
    for r in turn_rows:
        if ok_cyl(r, ds):
            med[(r["d_mm"], r["t"], r["cand"])].append(r["held_turn_deg"] if r.get("held") else 0.0)
    best = collections.defaultdict(float)
    for (d, t, c), v in med.items():
        best[d] = max(best[d], statistics.median(v))
    return [d for d in ds if d in best and best[d] < BAND_DEG]


def cylinder_cells(turn_rows, hold_rows):
    """(d, t) -> per-layout record: the turn grasp (largest median held turn over its placements), the hold grasp
    (largest drop load along gravity), the spread of grasps, and the geometry."""
    turns = collections.defaultdict(lambda: collections.defaultdict(list))
    kin, geo, grip = {}, {}, {}
    wall = collections.defaultdict(float)
    for r in turn_rows:
        if not ok_cyl(r):
            continue
        k = (r["d_mm"], r["t"])
        turns[k][r["cand"]].append(r["held_turn_deg"] if r.get("held") else 0.0)
        geo[k] = (r["x_sep_mm"], r["y_sep_mm"])
        geo[(k, r["cand"])] = (r["h_mm"], r["spread_mm"])
        kin[(k, r["cand"])] = r.get("kin")
        wall[k] += r.get("wall_s", 0.0)
    holds = collections.defaultdict(dict)
    for r in hold_rows:
        if not ok_cyl(r):
            continue
        k = (r["d_mm"], r["t"])
        holds[k][r["cand"]] = gravity(r)
        grip[(k, r["cand"])] = sum(r.get("F_hold", {}).values())
        geo[(k, r["cand"])] = (r["h_mm"], r["spread_mm"])
        wall[k] += r.get("wall_s", 0.0)
    out = {}
    for k, cands in turns.items():
        d, t = k
        meds = {c: statistics.median(v) for c, v in cands.items()}
        ct = max(meds, key=meds.get)
        loads = holds.get(k, {})
        drops = {c: v[1] for c, v in loads.items() if v[1] is not None}
        ch = max(drops, key=drops.get) if drops else None
        x_s, y_s = geo[k]
        s = x_s / 2
        out[k] = dict(
            d=d, t=t, x_s=x_s, y_s=y_s, s=s, standoff=s - d / 2,
            turn_cand=ct, turn_med=meds[ct], turn_pl=cands[ct], turn_succ=sum(x >= BAND_DEG for x in cands[ct]),
            n_cand=len(meds), n_cand_band=sum(m >= BAND_DEG for m in meds.values()),
            other_meds=sorted(m for c, m in meds.items() if m < BAND_DEG),
            turn_grasp=dict(zip(("h_mm", "spread_mm"), geo[(k, ct)])), turn_kin=kin[(k, ct)],
            hold_cand=ch, drop=drops.get(ch), pull3=loads[ch][0] if ch is not None else None,
            pull3_all=sorted(v[0] for v in loads.values()),
            hold_grasp=dict(zip(("h_mm", "spread_mm"), geo[(k, ch)])) if ch is not None else None,
            grip=grip.get((k, ch)),
            wall_s=wall[k],
        )
    return out


def check_against_landscape(C, L):
    """The 1 N m plant's turn grasp medians must be the landscape's turn scores of the same cells."""
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


def check_work_rows(work_hold, src_hold):
    """The 1 N m hold rows of 11 October must reproduce the 3 mm thresholds of 8 October: (rows compared, differing)."""
    old = {(r["tag"], r["d_mm"], r["cand"]): r.get("disturb") for r in src_hold if ok_cyl(r)}
    pairs = [(old[(r["tag"], r["d_mm"], r["cand"])], r.get("disturb")) for r in work_hold
             if ok_cyl(r) and (r["tag"], r["d_mm"], r["cand"]) in old]
    diff = sum(a != b for a, b in pairs)
    if diff:
        raise SystemExit(f"{diff} of {len(pairs)} 1 N m hold rows differ from 8 October")
    return len(pairs)


def per_object(C, d):
    return [C[k] for k in sorted(k for k in C if k[0] == d)]


def band(C, d):
    return [c for c in per_object(C, d) if c["turn_med"] >= BAND_DEG]


def plan_cells(C, d):
    """Five layouts centred on the band at the simulation's own step: the band's middle layout and two on each side.
    A cylinder without a band is centred on its layout of largest median held turn."""
    b = band(C, d) or [max(per_object(C, d), key=lambda c: c["turn_med"])]
    mid = sorted(c["t"] for c in b)[len(b) // 2]
    ts = [round(mid + i * STEP_T, 3) for i in range(-(N_CELLS // 2), N_CELLS - N_CELLS // 2)]
    return [C[(d, t)] for t in ts if (d, t) in C]


def old_rule_cells(C, d):
    """The first version's rule: one layout wider than the band, the band, then the compact side, N_CELLS in all."""
    b = band(C, d)
    if not b:
        return None
    lo, hi = min(c["t"] for c in b), max(c["t"] for c in b)
    ts = [round(lo - STEP_T, 3)] + [round(lo + i * STEP_T, 3) for i in range(int(round((hi - lo) / STEP_T)) + 1)]
    while len(ts) < N_CELLS:
        ts.append(round(ts[-1] + STEP_T, 3))
    return [C[(d, t)] for t in ts if (d, t) in C]


def fisher_two_sided(a, n1, b, n2):
    """Two-sided Fisher exact p for a successes of n1 against b of n2."""
    k, N = a + b, n1 + n2

    def pmf(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(N, k)
    p0 = pmf(a)
    return sum(pmf(x) for x in range(max(0, k - n2), min(k, n1) + 1) if pmf(x) <= p0 * (1 + 1e-9))


def binom_at_least(n, k, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for m in range(i, j + 1):
                r[order[m]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def hours(n_cells, n_days, meta, push=False):
    trials = n_cells * N_TURN * meta["t_trial_s"] / 3600
    ballast = n_cells * N_BALLAST * meta["t_ballast_s"] / 3600
    reconf = n_cells * meta["t_reconf_s"] / 3600
    day = n_days * (meta["t_day_overhead_s"] + REF_TRIALS * meta["t_trial_s"]) / 3600
    h_push = (T_PUSH_SETUP_S + PUSH_SERIES * T_PUSH_SERIES_S) / 3600 if push else 0.0
    return dict(cells=n_cells, turn_trials=n_cells * N_TURN, ballast_series=n_cells * N_BALLAST, h_turn=trials,
                h_ballast=ballast, h_reconf=reconf, days=n_days, h_day=day, h_push=h_push,
                total=trials + ballast + reconf + day + h_push)


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


def push_prediction(c, finger, P_):
    """Fingertip stiffness along the contact normal (N/mm) and the fingertip force at which the most loaded joint
    reaches the plant's torque limit (N), at the cell's turn grasp, from the finger kinematics."""
    q = np.radians(c["turn_kin"]["q_deg"][finger])
    J = K.jac_pad(finger, q)
    n = np.array([-1.0 if finger == "thumb" else 1.0, 0.0, 0.0])    # outward normal of a cylinder along palm y
    k = 1.0 / float(n @ (J @ J.T / P_["kp"]) @ n) / 1000.0
    g = np.abs(J.T @ n)
    return dict(k=k, F=P_["fr"] / g.max(), joint=JNAMES[int(np.argmax(g))], q_deg=c["turn_kin"]["q_deg"][finger])


def replay_fit():
    """Mean joint error of the 6 October bench replays per plant over the runs every plant replayed (time constant
    0.02 s, gains up to 8 N m/rad): its range, and the largest change across torque limits at one gain."""
    by = collections.defaultdict(dict)
    for r in jl(REPLAYS):
        if r.get("dxy_mm", 0) == 0 and r.get("model") == "pads" and r.get("mu") == "1.0":
            by[(r["kp"], r["tau"], r["fr"], r["fl"])][r["run"]] = r["joint_mae"]
    common = set.intersection(*[set(v) for v in by.values() if len(v) >= 8])
    g = collections.defaultdict(dict)
    for (kp, tau, fr, fl), v in by.items():
        if common <= set(v) and tau == 0.02 and kp <= 8:
            g[(kp, fl)][fr] = statistics.fmean(v[x] for x in common)
    allm = [m for d_ in g.values() for m in d_.values()]
    frs = sorted({fr for d_ in g.values() for fr in d_})
    return dict(lo=min(allm), hi=max(allm), spread=max(max(d_.values()) - min(d_.values()) for d_ in g.values()),
                fr_lo=frs[0], fr_hi=frs[-1], n_runs=len(common))


def policy_grips():
    g = [r["grip_N"] for r in jl(POLICIES) if r.get("status", "ok") == "ok" and r.get("grip_N") is not None]
    return min(g), max(g), len(g)


# ------------------------------------------------------------------------------------------ figure

def curve_spread(C, key, ds=CYL, lo=15.0, hi=45.0):
    """Largest difference between the cylinders' curves of `key` (linear interpolation against the standoff) on a
    1 mm grid of the shared standoff range."""
    curves = {d: sorted((c["standoff"], c[key]) for c in per_object(C, d) if c[key] is not None) for d in ds}

    def at(pts, x):
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return None
    worst = 0.0
    for i in range(int(lo), int(hi) + 1):
        v = [at(p, float(i)) for p in curves.values()]
        if all(x is not None for x in v):
            worst = max(worst, max(v) - min(v))
    return worst


def svg_landscape(C, C1, planned):
    W, H = 990, 380
    out = P._svg_open(W, H, "Held turn of the turn grasp and drop load of the hold grasp against the standoff of the "
                            "mounts, for cylinders of 10, 16 and 25 mm, on the servo plant (solid) and the 1 N m plant "
                            "(dashed); the planned bench layouts are filled.")
    pw, ph, top = 390, 260, 40
    xs, xt = (10, 56), list(range(10, 56, 10))
    ymax = max(c["drop"] for CC in (C, C1) for c in CC.values() if c["drop"] is not None)
    ytop = 2 * math.ceil(ymax / 2)
    panels = (("turn_med", "Held turn of the turn grasp, deg", (0, 120), list(range(0, 121, 20))),
              ("drop", "Drop load of the hold grasp, N", (0, ytop), list(range(0, ytop + 1, 2 if ytop <= 12 else 4))))
    for k, (key, ylab, ys, yt) in enumerate(panels):
        x0 = 70 + k * (pw + 100)
        fx, fy = P._panel(out, x0, top, pw, ph, xs, ys, xt, yt, "Standoff of the mounts, mm", ylab)
        if key == "turn_med":
            out.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{fy(BAND_DEG):.1f}" y2="{fy(BAND_DEG):.1f}" '
                       f'style="stroke:var(--ink3);stroke-width:1.5;stroke-dasharray:2 4"/>'
                       f'<text x="{x0 + pw - 4}" y="{fy(BAND_DEG) - 6:.1f}" text-anchor="end" '
                       f'style="fill:var(--ink3)">band, {BAND_DEG:.0f}&#176;</text>')
        for d in CYL:
            ref = [c for c in per_object(C1, d) if c[key] is not None and xs[0] <= c["standoff"] <= xs[1]]
            P._path(out, fx, fy, [(c["standoff"], c[key]) for c in ref], COL[d], dashed=True, width=1.4)
            cs = [c for c in per_object(C, d) if c[key] is not None and xs[0] <= c["standoff"] <= xs[1]]
            P._path(out, fx, fy, [(c["standoff"], c[key]) for c in cs], COL[d], width=2.4)
            for c in cs:
                on = (d, c["t"]) in planned
                tip = (f"{d} mm cylinder, t = {c['t']:g}, standoff {c['standoff']:.2f} mm: "
                       + (f"{c['turn_med']:.1f} deg" if key == "turn_med" else f"{c['drop']:.2f} N")
                       + (" (planned)" if on else ""))
                P._marker(out, fx(c["standoff"]), fy(c[key]), COL[d], hollow=not on, r=5.0 if on else 3.4, title=tip)
    out.append("</svg>")
    leg = P._legend_html([(f"{d} mm cylinder", COL[d], False, "circle") for d in CYL]
                         + [("1 N m plant", "var(--ink3)", True, None)])
    return "".join(out) + leg


# ------------------------------------------------------------------------------------------ sections

def glossary(C):
    nc = [c["n_cand"] for d in CYL for c in plan_cells(C, d)]
    PS, P1 = plant(PLANT), plant(PLANT_1NM)
    G = [
        ("Layout " + im("t"), "the finger-base positions of the symmetric tripod in the palm frame, in mm: thumb at "
         + im(r"(-x_s/2,\,0)") + ", index and middle at " + im(r"(x_s/2,\,\pm y_s/2)") + ", with "
         + im(r"x_s = 100 - 60\,t") + " and " + im(r"y_s = 110 - 60\,t") + ". Larger " + im("t") + " is more compact; the "
         "simulation steps " + im("t") + " by 0.125."),
        ("Mount distance " + im("s"), im("s = x_s/2") + ": distance from the cylinder&#8217;s axis, which lies along the "
         "pair axis, to the thumb mount and to the line of the pair mounts, in mm. One step of " + im("t") + " is "
         "3.75&#8202;mm."),
        ("Standoff " + im("s - r"), "mount distance minus the cylinder&#8217;s radius " + im("r") + ", in mm."),
        ("Plant", "the simulated finger servo: a position actuator on each joint with a stiffness " + im("k_p") + " in "
         f"N&#8202;m/rad, a torque limit in N&#8202;m and a time constant of {PS['tau']:.2f}&#8202;s. The servo plant "
         f"({PS['kp']:g}&#8202;N&#8202;m/rad, {PS['fr']:g}&#8202;N&#8202;m) gives every prediction in this plan; the "
         f"1&#8202;N&#8202;m plant ({P1['kp']:g}&#8202;N&#8202;m/rad, {P1['fr']:g}&#8202;N&#8202;m) gave its first "
         "version and is shown for comparison."),
        ("Turn grasp, hold grasp", f"of the {rng(min(nc), max(nc))} candidate grasps of a planned layout (palm "
         "height and straddle), the one with the largest median held turn over three placements, and the one with the "
         "largest drop load."),
        ("HOM controller", "the relative contact-velocity controller of Wang, Oh and Pollard (arXiv 2609.25619) in "
         "<code>scripts/hom_turn3.py</code>, turning at 30&#176;/s toward 90&#176; with pad force regulated to 2&#8202;N; "
         "its model of the servo uses the plant&#8217;s stiffness."),
        ("Held turn", "rotation of the cylinder&#8217;s axis from horizontal toward vertical after a 2&#8202;s hold, in "
         "degrees, above 90&#176; when it turns past vertical; 0 unless two fingers still press with at least the "
         "object&#8217;s weight. The bench measures it with the AprilTag tracker."),
        ("Success, band", f"a trial succeeds at a held turn of {BAND_DEG:.0f}&#176; or more. A cylinder&#8217;s band is "
         f"the set of layouts whose turn grasp has a median held turn of {BAND_DEG:.0f}&#176; or more."),
        ("Drop load", "the smallest load along gravity at which the held cylinder leaves the grasp, in N: it moves "
         "15&#8202;mm, or fewer than two fingers press with its weight. The simulation ramps a force at 4&#8202;N/s; the "
         "bench hangs steel washers in 5&#8202;g steps on a hook under the cylinder until it falls."),
        ("3&#8202;mm pull-out", "the smallest load along gravity that moves the held cylinder 3&#8202;mm or turns it "
         "5&#176;, in N; the hold score of the 8 October study and of this plan&#8217;s first version."),
        ("Fingertip stiffness, trip force", "force on a fingertip along the contact normal per millimetre of fingertip "
         "displacement while the servos hold a commanded pose, in N/mm; and the fingertip force at which the most loaded "
         "joint reaches the plant&#8217;s torque limit, in N."),
        ("Open-loop replay", "the bench sends a simulated turn&#8217;s joint commands without force feedback; the "
         "fingertips have no force sensors."),
    ]
    rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in G)
    return f"<div class='tw'><table class='gloss'><tbody>{rows}</tbody></table></div>"


def plant_section(C, C1, R, grips):
    PS, P1 = plant(PLANT), plant(PLANT_1NM)
    err = math.radians(DEG_PER_LOAD * LOAD_FULL)
    s1, w1 = plan_cells(C, 10), [C1.get((10, c["t"])) for c in plan_cells(C, 10)]
    dw = [c["drop"] for c in w1 if c is not None and c["drop"] is not None]
    gw = [c["grip"] for c in w1 if c is not None and c["grip"] is not None]
    ds_, gs = [c["drop"] for c in s1], [c["grip"] for c in s1]
    return (
        f"<p>The SCS0009 finger servo stalls at {STALL_NM:g}&#8202;N&#8202;m at 6&#8202;V (datasheet), and its overload "
        f"protection cuts the drive to 20&#8202;% after a sustained load above {TRIP_FRAC * 100:.0f}&#8202;% of stall "
        f"(<a href='{SYSID}'>servo study</a>). The servo plant therefore limits each joint to {PS['fr']:g}&#8202;N&#8202;m. The "
        f"load register reaches full drive at {DEG_PER_LOAD * LOAD_FULL:.1f}&#176; of position error "
        f"({DEG_PER_LOAD:g}&#176; per load unit over {N_SYSID_RUNS} bench runs). This gives a stiffness of "
        f"{STALL_NM:g}&#8202;N&#8202;m&#8202;/&#8202;{err:.3f}&#8202;rad = {STALL_NM / err:.2f}&#8202;N&#8202;m/rad. The "
        f"plant uses {PS['kp']:g}&#8202;N&#8202;m/rad and a {PS['tau']:.2f}&#8202;s time constant.</p>"
        f"<p>The plan&#8217;s first version used {P1['kp']:g}&#8202;N&#8202;m/rad and a {P1['fr']:g}&#8202;N&#8202;m limit, "
        f"{P1['fr'] / STALL_NM:.1f} times the stall torque. It came from a fit of bench replays that reproduced the joint "
        f"angles to {R['lo']:.1f}&#8211;{R['hi']:.1f}&#176; at every limit from {R['fr_lo']:g} to {R['fr_hi']:g}&#8202;N&#8202;m. "
        f"Policies trained on that plant gripped the screwdriver with {grips[0]:.0f}&#8211;{grips[1]:.0f}&#8202;N "
        f"(<a href='{POLICY_PAGE}'>contact-model policies of 8 October</a>). In this study both plants command "
        "4&#8202;N per pad. The hold grasps of the Stage&#160;1 cells press the cylinder with "
        f"{rng(min(gs), max(gs), 1)}&#8202;N in total on the servo plant"
        + (f" and {rng(min(gw), max(gw), 1)}&#8202;N on the 1&#8202;N&#8202;m plant" if gw else "")
        + f". Under a load along gravity they hold {rng(min(ds_), max(ds_), 1)}&#8202;N"
        + (f" and {rng(min(dw), max(dw), 1)}&#8202;N" if dw else "") + " (Table&#160;1).</p>"
    )


def science(C, C1, L):
    F = L["families"]["diag"]
    fk, ft = F["fits"]["cylinder:kin"], F["fits"]["cylinder:turn"]
    banded = [d for d in CYL if band(C, d)]
    unbanded = [d for d in CYL if not band(C, d)]
    b10, p10 = band(C, 10), plan_cells(C, 10)
    out10 = [C[(10, round(t, 3))] for t in (min(c["t"] for c in b10) - STEP_T, max(c["t"] for c in b10) + STEP_T)
             if (10, round(t, 3)) in C]
    w10 = band(C1, 10)
    centre_s = {d: statistics.fmean(c["s"] for c in band(C, d)) for d in banded}
    centre1 = {d: statistics.fmean(c["standoff"] for c in band(C1, d)) for d in CYL if band(C1, d)}
    peaks = {d: max(per_object(C, d), key=lambda c: c["drop"] or 0.0) for d in CYL}
    inb = [c for d in banded for c in band(C, d)]
    others = [m for c in inb for m in c["other_meds"]]
    worst = min(inb, key=lambda c: min(c["turn_pl"]))
    nb, nc = [c["n_cand_band"] for c in inb], [c["n_cand"] for c in inb]
    rest = []
    for d in banded[1:]:
        b = band(C, d)
        rest.append(f"the {d}&#8202;mm cylinder turns {rng(min(c['turn_med'] for c in b), max(c['turn_med'] for c in b), unit='&#176;')} "
                    + (f"at one layout, a standoff of {f(b[0]['standoff'], 0)}&#8202;mm" if len(b) == 1 else
                       f"at standoffs of {rng(min(c['standoff'] for c in b), max(c['standoff'] for c in b), unit='&#8202;mm')}"))
    for d in unbanded:
        rest.append(f"the {d}&#8202;mm cylinder turns at most {max(c['turn_med'] for c in per_object(C, d)):.0f}&#176; at "
                    "any layout")
    same_s = max(centre_s.values()) - min(centre_s.values()) < STEP_T * 30 / 2
    lost_s = {d: round(statistics.fmean(centre_s.values()) - d / 2, 1) for d in unbanded}
    # the 3 mm pull-out against the drop load at the Stage 1 hold grasps, on both plants
    gap = max(p10, key=lambda c: c["drop"] - c["pull3"])
    gap1 = [C1[(10, c["t"])]["drop"] - C1[(10, c["t"])]["pull3"] for c in p10
            if (10, c["t"]) in C1 and C1[(10, c["t"])]["drop"] is not None]
    txt = (
        "<p>The study asks how the finger spacing that holds an object best, and the spacing that turns it furthest, "
        "move with the object&#8217;s size, and whether the simulator predicts where they lie. For a task " + im("T")
        + " with score " + im("J_T") + " the quantity is " + im(r"s^*_T(r) = \arg\max_s J_T(s, r)") + ". Borr&#224;s and "
        "Dollar (2015) and Hota and Kumar (2019) answer the kinematic part, and the simulation of 8 October reproduces "
        "their result on SR2: the fixed-contact workspace is largest at "
        + im(fr"s = {fk['alpha']:.2f}\,r + {fk['beta_mm']:.1f}\,\text{{mm}}") + ", about half a finger outside the "
        "object. The bench can answer what kinematics cannot: whether holding and turning select different spacings on "
        f"a physical hand, and whether the simulated landscape agrees with it at the same layouts (<a href='{STUDY}'>study "
        "page</a>). On the servo plant the simulation makes three predictions for cylinders (Figure&#160;1).</p>"
        "<ol class='pred'>"
        f"<li><b>Turn band.</b> On the 10&#8202;mm cylinder the turn grasp turns "
        f"{rng(min(c['turn_med'] for c in b10), max(c['turn_med'] for c in b10), unit='&#176;')} at standoffs of "
        f"{rng(min(c['standoff'] for c in b10), max(c['standoff'] for c in b10), unit='&#8202;mm')} and "
        f"{rng(min(c['turn_med'] for c in out10), max(c['turn_med'] for c in out10), unit='&#176;')} one layout outside "
        f"them; on the 1&#8202;N&#8202;m plant this band lay at "
        f"{rng(min(c['standoff'] for c in w10), max(c['standoff'] for c in w10), unit='&#8202;mm')}. "
        + "; ".join(rest)[:1].upper() + "; ".join(rest)[1:] + ".</li>"
        "<li><b>Scaling.</b> "
        + (f"The bands of the {' and '.join(str(d) for d in banded)}&#8202;mm cylinders are both centred on a mount "
           f"distance of {f(statistics.fmean(centre_s.values()))}&#8202;mm, so the band keeps its mount distance and its "
           "standoff shrinks by one millimetre per millimetre of radius"
           + (f"; at that mount distance the standoff of the {' and '.join(str(d) for d in unbanded)}&#8202;mm cylinder "
              f"would be {' and '.join(f(lost_s[d], 0) for d in unbanded)}&#8202;mm, and it has no band"
              if unbanded else "") + ". " if same_s else
           "The band centres lie at mount distances of " + ", ".join(f"{f(centre_s[d])}&#8202;mm ({d}&#8202;mm)"
                                                                       for d in banded) + ". ")
        + f"On the 1&#8202;N&#8202;m plant the band kept its standoff "
        f"({rng(min(centre1.values()), max(centre1.values()), unit='&#8202;mm')} from the {min(centre1)} to the "
        f"{max(centre1)}&#8202;mm cylinder) and moved outward with the radius (fit of 8 October: "
        + im(fr"s^*_\text{{turn}} = {ft['alpha']:.2f}\,r + {ft['beta_mm']:.1f}\,\text{{mm}}") + ").</li>"
        f"<li><b>Drop load.</b> The hold grasp&#8217;s drop load rises as the mounts close in, on the 10&#8202;mm "
        f"cylinder from {f(p10[0]['drop'])}&#8202;N at a {f(p10[0]['standoff'], 0)}&#8202;mm standoff to "
        f"{f(p10[-1]['drop'])}&#8202;N at {f(p10[-1]['standoff'], 0)}&#8202;mm, and peaks at "
        f"{rng(min(c['drop'] for c in peaks.values()), max(c['drop'] for c in peaks.values()), nd=1, unit='&#8202;N')} "
        f"near standoffs of {rng(min(c['standoff'] for c in peaks.values()), max(c['standoff'] for c in peaks.values()), unit='&#8202;mm')} "
        "on all three cylinders. Holding along gravity favours layouts more compact than the band, so the two tasks "
        f"select different spacings. Against the standoff the three drop-load curves lie within "
        f"{f(curve_spread(C, 'drop'))}&#8202;N of each other.</li></ol>"
        "<p>The first version scored holding by the 3&#8202;mm pull-out. On the 1&#8202;N&#8202;m plant the drop load "
        f"of the Stage&#160;1 hold grasps exceeds it by {rng(min(gap1), max(gap1), 1) if gap1 else '&#8211;'}&#8202;N. On "
        "the servo plant the fingers yield before the pads slip, and the cylinder tilts 5&#176; or sags 3&#8202;mm while "
        "still held, so the gap reaches "
        f"{f(gap['drop'] - gap['pull3'])}&#8202;N (" + im(f"t = {gap['t']:g}") + f": {f(gap['pull3'])} against "
        f"{f(gap['drop'])}&#8202;N). The hanging-mass series measure the drop load, and the plan predicts them with "
        "it.</p>"
        f"<p>The band rests on one or two grasps per layout: at each band layout {rng(min(nb), max(nb))} of the "
        f"{rng(min(nc), max(nc))} candidates turn {BAND_DEG:.0f}&#176; or more, and the others turn at most "
        f"{max(others):.0f}&#176;. The turn grasp&#8217;s three placements also differ, from "
        f"{min(worst['turn_pl']):.0f}&#176; to {max(worst['turn_pl']):.0f}&#176; in the widest case ({worst['d']:g}&#8202;mm, "
        + im(f"t = {worst['t']:g}") + "). The bench outcome is therefore expected to be bimodal, and the plan counts each "
        "trial as a success or a stall.</p>"
    )
    return txt, dict(b10=b10, out10=out10, w10=w10, centre_s=centre_s, banded=banded, unbanded=unbanded, peaks=peaks)


def old_section(S, L, C, C1):
    F = L["families"]["diag"]
    tag = {l["t"]: l["tag"] for l in F["layouts"]}
    rg = F["ridges"]["cylinder:25"]["hold"]
    t0 = min(tag, key=lambda t: abs(t - rg["at"]))
    hold = {t: F["objects"]["cylinder:25"][tag[t]]["hold"] for t in (t0 - 0.125, t0, t0 + 0.125)}
    every = "each" if S["n_two"] == S["n_all"] else f"{S['n_two']}"
    b10, w10 = band(C, 10), band(C1, 10)
    none = [d for d in CYL if not band(C, d)]
    return (
        f"<p>The protocol of 8 October measures {S['cells']} cells over {S['objects']} objects in "
        f"{f(S['hours'])}&#8202;h. Four properties keep it from testing these predictions.</p><ol class='steps'>"
        f"<li>Its bench grid takes every second simulated layout, so {every} of its {S['n_all']} ridges is bracketed by at "
        "most two bench layouts, and two values cannot locate a peak. In the 8 October simulation the hold ridge of the "
        "25&#8202;mm cylinder lies at " + im(f"t = {t0:g}") + f" with a score of {f(hold[t0])}&#8202;N; the bench layouts "
        f"beside it score {f(hold[t0 - 0.125])} and {f(hold[t0 + 0.125])}&#8202;N.</li>"
        "<li>Its hold score is the weakest of six pull directions, but the hanging mass loads only along gravity, where "
        "the simulated drop load rises steadily toward the compact side (prediction&#160;3).</li>"
        "<li>The bench replays the simulated joint commands open loop, but every simulated turn ran the closed-loop "
        "HOM controller, and no open-loop replay has been simulated.</li>"
        "<li>Its scores come from the 1&#8202;N&#8202;m plant, on which the 10&#8202;mm cylinder&#8217;s band lies at "
        f"standoffs of {rng(min(c['standoff'] for c in w10), max(c['standoff'] for c in w10), unit='&#8202;mm')}; on the "
        f"servo plant it lies at {rng(min(c['standoff'] for c in b10), max(c['standoff'] for c in b10), unit='&#8202;mm')}"
        + (f", and the {' and '.join(str(d) for d in none)}&#8202;mm cylinder has none" if none else "") + ".</li></ol>"
    )


def stage0_section(objs):
    rec = {o["d_mm"]: o for o in objs if o["shape"] == "cylinder"}
    recipes = "; ".join(f"{d}&#8202;mm: {esc(rec[d]['recipe'])}, {rec[d]['mass_g']:.1f}&#8202;g"
                        for d in CYL if d in rec)
    n_plans = 2 * N_CELLS * len(CYL)
    return (
        "<p>Four checks need the CPU only, about a day of work, and decide whether a bench turn can test the "
        "simulation.</p><ol class='steps'>"
        "<li><b>Plan export.</b> A new <code>scripts/hand_object_scale_export.py</code> writes the turn and hold grasps "
        f"of every cell in <code>{REL}/plan.json</code> as deploy plans, {n_plans} in all, and runs the clearance gate "
        "(<code>real_v1_trajectory_clearance.py</code>, chord and CSV path) and <code>HandPlan.validate()</code> on each. "
        "A cell whose plan fails moves to the neighbouring layout.</li>"
        "<li><b>Open-loop replay.</b> Replay each simulated turn&#8217;s joint commands on the servo plant without force "
        "feedback. If the band disappears, Stage&#160;1 measures the drop load only.</li>"
        "<li><b>Placement tolerance.</b> Repeat the replay at object offsets of &#177;2&#8202;mm and yaws of "
        "&#177;3&#176;, the error of placing the cylinder by hand; the success fraction per cell is the number the bench "
        "rate is compared with.</li>"
        f"<li><b>Mass.</b> Print and weigh the three cylinders ({recipes}) and re-run the "
        f"{N_CELLS * len(CYL)} cells at the measured masses.</li></ol>"
    )


def push_section(cell, preds, cpu_min):
    si, st, wi = preds[(PLANT, "index")], preds[(PLANT, "thumb")], preds[(PLANT_1NM, "index")]
    h = (T_PUSH_SETUP_S + PUSH_SERIES * T_PUSH_SERIES_S) / 60
    mirror = all(abs(a - b) < 0.5 for a, b in zip(si["q_deg"], st["q_deg"]))
    pose = ("the thumb stands in the mirror pose" if mirror else
            "the thumb at " + ", ".join(f(v, 0) for v in st["q_deg"]) + "&#176;")
    return (
        "<p>Stage&#160;1 opens with a push test that calibrates the plant. The hand holds the turn grasp of the band "
        "cell " + im(f"t = {cell['t']:g}") + " (Table&#160;1), index yaw, MCP and PIP at "
        + ", ".join(f(v, 0) for v in si["q_deg"]) + f"&#176;; {pose}. A push-pull force gauge on a manual "
        "linear stage presses the index fingertip along the contact normal, away from where the cylinder would sit. The "
        f"steps are {PUSH_STEP_N:g}&#8202;N, held {PUSH_HOLD_S}&#8202;s each. The push continues until the most loaded servo "
        "trips its overload protection (load readback fixed at 200). The gauge then backs off in the same steps. The fingertip "
        "displacement follows from the joint position readbacks through the finger kinematics "
        "(<code>hand_object_scale_kin.fk</code>). The slope of force against displacement gives the fingertip "
        "stiffness, the force at the trip gives the trip force, and the gap between the loading and unloading branches "
        f"gives the friction band. Three series on the index finger and three on the thumb take {h:.0f}&#8202;min.</p>"
        f"<p>At this pose the servo plant predicts a fingertip stiffness of {f(si['k'], 2)}&#8202;N/mm and a trip force "
        f"of {f(si['F'])}&#8202;N, set by the {si['joint']} joint; the 1&#8202;N&#8202;m plant predicts "
        f"{f(wi['k'], 2)}&#8202;N/mm and {f(wi['F'])}&#8202;N. The plant&#8217;s stiffness then scales by the ratio of "
        "measured to predicted fingertip stiffness, and its torque limit scales by the ratio of trip forces. The 15 cells "
        f"are re-simulated on the refitted plant before the first turn trial ({cpu_min:.0f} CPU-minutes of rollouts). "
        "If the band moves, the Stage&#160;1 cells move with it.</p>"
    )


def cell_rows(cells, C1):
    rows = []
    for c in cells:
        pl = ", ".join(f"{x:.0f}" for x in c["turn_pl"])
        w = C1.get((c["d"], c["t"]))
        rows.append([f"{c['t']:g}", f(c["standoff"], 2), f"{c['turn_med']:.0f} ({pl})", f"{c['turn_succ']} of {len(c['turn_pl'])}", f(c["drop"]),
                     f"{w['turn_med']:.0f}" if w else "&#8211;", f(w["drop"]) if w else "&#8211;"])
    return rows


HEAD = ["Layout " + im("t"), "Standoff,<br>mm", "Held turn, deg<br>(placements)", "Successes", "Drop load,<br>N",
        "1&#8202;N&#8202;m plant:<br>held turn, deg", "1&#8202;N&#8202;m plant:<br>drop load, N"]


def stage1_section(C, C1, H1):
    cells = plan_cells(C, 10)
    bc = [c for c in cells if c["turn_med"] >= BAND_DEG]
    side = (len(cells) - len(bc)) // 2
    rho = spearman([c["standoff"] for c in cells], [c["drop"] for c in cells])
    p = fisher_two_sided(PASS_BAND, N_TURN, PASS_EDGE, N_TURN)
    k, n = sum(c["turn_succ"] for c in bc), sum(len(c["turn_pl"]) for c in bc)
    return (
        f"<p>Stage&#160;1 runs the 10&#8202;mm cylinder at {word(len(cells))} layouts spaced by the simulation&#8217;s own step "
        f"(3.75&#8202;mm of mount distance) and centred on the band: its {word(len(bc))} layouts and {word(side)} on "
        f"each side (Table&#160;1). Each cell gets {N_TURN} turn trials with its turn grasp and {N_BALLAST} hanging-mass series with "
        f"its hold grasp, in random order within the day. With the push test it takes {f(H1['total'])}&#8202;h on one "
        "bench day.</p>"
        + tcap("Stage&#160;1 cells on the 10&#8202;mm cylinder, servo plant, with the 1&#8202;N&#8202;m plant at the "
               "same layouts. Held turn: the turn grasp&#8217;s median over three placements (object offset 2&#8202;mm, "
               f"yaw 2&#176;), with the three values; successes: placements at {BAND_DEG:.0f}&#176; or more; drop load: "
               "the hold grasp.")
        + table(HEAD, cell_rows(cells, C1), "cells")
        + f"<p><b>Decision.</b> Stage&#160;2 starts if a band cell succeeds in {PASS_BAND} or more of its {N_TURN} "
        f"trials, both edge cells in {PASS_EDGE} or fewer, and the mean drop load rises across the five cells (rank "
        f"correlation with the standoff of {P.num(RHO_MAX, '.1f')} or lower; {P.num(rho, '.1f')} in simulation). "
        "Otherwise the turn trials stop; the bench joint readbacks and tracker traces are compared with the open-loop "
        "replay of Stage&#160;0, and the drop-load trend is reported alone. "
        f"{word(PASS_BAND, True)} successes against {word(PASS_EDGE)}, the weakest contrast the rule accepts, differ at "
        + im(f"p = {p:.3f}") + " (Fisher&#8217;s exact test, two-sided). A band cell that succeeds on the bench as "
        f"often as the simulated placements of the band cells ({k} of {n}) reaches {PASS_BAND} of {N_TURN} with "
        f"probability {binom_at_least(N_TURN, PASS_BAND, k / n):.2f}.</p>"
    )


def stage2_section(C, C1, H2):
    rows = []
    for d in STAGE2:
        rows.append(f"{d}&#8202;mm cylinder")
        rows += cell_rows(plan_cells(C, d), C1)
    say = []
    for d in STAGE2:
        ts = [c["t"] for c in plan_cells(C, d)]
        s_ = [f"{t:g}" for t in ts if C[(d, t)]["turn_med"] >= BAND_DEG]
        w_ = [f"{t:g}" for t in ts if (d, t) in C1 and C1[(d, t)]["turn_med"] >= BAND_DEG]
        say.append((d, s_, w_))

    def at(v, d):
        return (f"a band at " + im("t = " + ", ".join(v)) + f" on the {d}&#8202;mm cylinder") if v else \
            f"none on the {d}&#8202;mm cylinder"
    old16 = old_rule_cells(C, 16)
    rho_old = spearman([c["standoff"] for c in old16], [c["drop"] for c in old16]) if old16 else None
    return (
        f"<p>Stage&#160;2 runs the 16 and 25&#8202;mm cylinders at five layouts each by the same rule (Table&#160;2) "
        "and tests prediction&#160;2. In these cells the servo plant predicts "
        + " and ".join(at(s_, d) for d, s_, _ in say)
        + ". The 1&#8202;N&#8202;m plant predicts " + " and ".join(at(w_, d) for d, _, w_ in say)
        + ". The bench successes therefore show which plant places the band.</p>"
        "<p>The cells follow the rule of Stage&#160;1: five layouts centred on the band. A cylinder without a band is "
        "centred on its layout of largest median held turn. The rule of the first version put one layout wider than the "
        "band and the rest on the compact side. It gives the same Stage&#160;1 cells but none for a cylinder without a band"
        + (f". On the 16&#8202;mm cylinder, whose band is one layout wide, it puts three cells past the band, where "
           f"the drop load passes its peak (rank correlation {P.num(rho_old, '.1f')}, against "
           f"{P.num(RHO_MAX, '.1f')} in the decision)" if rho_old is not None else "")
        + f". Stage&#160;2 takes {f(H2['total'])}&#8202;h on two bench days.</p>"
        + tcap("Stage&#160;2 cells, columns as in Table&#160;1.")
        + table(HEAD, rows, "cells")
    )


def time_section(H1, H2, S):
    def row(name, h):
        return [name, str(h["cells"]), f"{h['turn_trials']} ({f(h['h_turn'])}&#8202;h)",
                f"{h['ballast_series']} ({f(h['h_ballast'])}&#8202;h)", f"{h['days']} ({f(h['h_day'])}&#8202;h)",
                f(h["h_push"]) if h["h_push"] else "&#8211;", f(h["total"])]
    tot = {k: H1[k] + H2[k] for k in H1}
    return (
        tcap("Bench time. A day opens with homing, the tracker check and setup (45&#8202;min) and five trials of the "
             "previous day&#8217;s band cell, which measure drift between days. Turn trials take 90&#8202;s, mass series "
             "6&#8202;min and a layout change 30&#8202;s (protocol.json of 8 October). The push test takes "
             f"{T_PUSH_SETUP_S / 60:.0f}&#8202;min to mount the gauge and {PUSH_SERIES} series of "
             f"{T_PUSH_SERIES_S / 60:.0f}&#8202;min.")
        + table(["Stage", "Cells", "Turn trials", "Mass series", "Days (overhead)", "Push test, h", "Hours"],
                [row("1: 10&#8202;mm cylinder", H1), row("2: 16 and 25&#8202;mm cylinders", H2), row("Total", tot),
                 ["Protocol of 8 October", str(S["cells"]), str(S["turn_trials"]), str(S["ballast"]), str(S["days"]),
                  "&#8211;", f(S["hours"])]])
    )


def outcomes_section(C, C1):
    def where(CC, d):
        ts = [c["t"] for c in plan_cells(C, d) if (d, c["t"]) in CC and CC[(d, c["t"])]["turn_med"] >= BAND_DEG]
        if not ts:
            return "none"
        return im("t = " + (f"{ts[0]:g}" if len(ts) == 1 else f"{min(ts):g}\u2013{max(ts):g}"))
    sv = ", ".join(f"{d}&#8202;mm {where(C, d)}" for d in CYL)
    wk = ", ".join(f"{d}&#8202;mm {where(C1, d)}" for d in CYL)
    rows = [
        [f"Bands where the servo plant puts them ({sv}) and a rising drop load",
         "The servo plant predicts the hand: holding and turning select different spacings, and the turn&#8217;s "
         "spacing is a fixed mount distance",
         "Paper contributions 3 and 4 on the servo plant; plant sensitivity, large cylinders and spheres next (Future "
         "work)"],
        [f"Bands where the 1&#8202;N&#8202;m plant puts them ({wk})",
         "The turn&#8217;s spacing scales with the object, and the servo plant misplaces the band",
         "Compare the push test with both plants; re-simulate at the measured strength before more objects"],
        ["No band, rising drop load", "The open-loop turn does not transfer at these grasps; the hold trend stands",
         "The paper reports the simulated turn landscape with the bench drop loads; bench turns wait for closed-loop "
         "control"],
        ["Neither", "The bench grasp differs from the simulated one",
         "Measure grip (servo load readbacks) and pad placement before more bench time"],
    ]
    t = table(["Outcome", "What it shows", "Next"], [[(a, "wrap"), (b, "wrap"), (c, "wrap")] for a, b, c in rows])
    t = t.replace("<th class=num>", "<th>").replace("class='num wrap'", "class='wrap'")
    return tcap("Outcomes after Stage&#160;2 and the step each one leads to.") + t


def next_section(C, big):
    per_cell_s = statistics.fmean(c["wall_s"] for c in C.values())
    return "<ul>" + "".join(f"<li>{x}</li>" for x in [
        "<b>Plant sensitivity.</b> Re-run the 10 and 16&#8202;mm cylinders at stiffnesses of 0.5 and "
        "1.0&#8202;N&#8202;m/rad and torque limits of 0.14 and 0.23&#8202;N&#8202;m "
        "(<code>hand_object_scale_sim.py run --plant kp0.5_kv0_fr0.18_fl0_dp0.01 ...</code>). If the band moves by a "
        "layout or more within that range, the bench locates the band only as well as the push test measures the "
        "plant.",
        "<b>Large cylinders and spheres.</b> Their landscape of 8 October is on the 1&#8202;N&#8202;m plant: the "
        f"{' and '.join(str(d) for d in big)}&#8202;mm cylinders have no band, and spheres turn best at the wide edge of each feasible band and hold best "
        "at the compact edge. Re-run them on the servo plant (<code>--only cylinder:40 cylinder:50 sphere:16 sphere:25 "
        "sphere:40 sphere:50</code>) before planning bench cells.",
        "<b>Closed-loop turn.</b> Pad force estimated from the servo load readbacks (0.0186&#176; per load unit) would "
        "let the HOM controller run on the hand instead of an open-loop replay.",
        "<b>Two-dimensional map.</b> The 9&#8202;&#215;&#8202;9 grid of " + im("x_s") + " and " + im("y_s") + " for the "
        f"three cylinders on the servo plant (about {81 * len(CYL) * per_cell_s / 3600:.0f} CPU-hours at "
        f"{per_cell_s:.0f}&#8202;s per cell) tests whether the band depends on " + im("x_s") + " alone.",
    ]) + "</ul>"


def lede(C, C1, H1, H2, sci):
    b10, out10, w10 = sci["b10"], sci["out10"], sci["w10"]
    p10 = plan_cells(C, 10)
    shift = statistics.fmean(c["standoff"] for c in w10) - statistics.fmean(c["standoff"] for c in b10)
    PS = plant(PLANT)
    rest = [f"a {d}&#8202;mm cylinder turns at the same mount distance" if d in sci["banded"] else
            f"a {d}&#8202;mm cylinder at no layout" for d in CYL[1:]]
    return (f"The bench week tests three predictions of a simulation whose finger servos have the SCS0009&#8217;s "
            f"strength ({PS['kp']:g}&#8202;N&#8202;m/rad, {PS['fr']:g}&#8202;N&#8202;m); the plan&#8217;s first version "
            "used a 1&#8202;N&#8202;m plant. The HOM controller turns a 10&#8202;mm cylinder "
            f"{rng(min(c['turn_med'] for c in b10), max(c['turn_med'] for c in b10), unit='&#176;')} toward vertical "
            "when the finger mounts stand "
            f"{rng(min(c['standoff'] for c in b10), max(c['standoff'] for c in b10), unit='&#8202;mm')} off its surface, "
            f"{f(shift)}&#8202;mm closer than on the 1&#8202;N&#8202;m plant, and "
            f"{rng(min(c['turn_med'] for c in out10), max(c['turn_med'] for c in out10), unit='&#176;')} one layout "
            "outside; " + " and ".join(rest) + ". The drop load along gravity rises from "
            f"{f(p10[0]['drop'])} to {f(p10[-1]['drop'])}&#8202;N as the mounts close in. After a push test that "
            f"calibrates the plant, the plan spends {f(H1['total'])}&#8202;h on the 10&#8202;mm cylinder and "
            f"{f(H2['total'])}&#8202;h on the other two if the band appears.")


def write_plan(C, C1, H1, H2, push_cell, preds):
    cells = []
    for stage, ds in ((1, STAGE1), (2, STAGE2)):
        for d in ds:
            for c in plan_cells(C, d):
                w = C1.get((d, c["t"]))
                cells.append(dict(stage=stage, shape="cylinder", d_mm=d, t=c["t"], x_sep_mm=c["x_s"], y_sep_mm=c["y_s"],
                                  mount_distance_mm=c["s"], standoff_mm=round(c["standoff"], 3),
                                  in_band=c["turn_med"] >= BAND_DEG,
                                  turn_grasp=dict(cand=c["turn_cand"], **c["turn_grasp"]),
                                  hold_grasp=dict(cand=c["hold_cand"], **(c["hold_grasp"] or {})),
                                  sim_turn_deg=round(c["turn_med"], 2), sim_turn_placements_deg=c["turn_pl"],
                                  sim_successes=c["turn_succ"], sim_drop_N=round(c["drop"], 3),
                                  sim_pull3_N=round(c["pull3"], 3),
                                  sim_1Nm=dict(turn_deg=round(w["turn_med"], 2) if w else None,
                                               drop_N=round(w["drop"], 3) if w and w["drop"] is not None else None),
                                  n_turn=N_TURN, n_ballast=N_BALLAST))
    plan = dict(meta=dict(date="2026-10-11", revised="2026-10-11, servo plant", plant=PLANT,
                          rows=os.path.relpath(ROWS, ROOT), comparison=dict(plant=PLANT_1NM,
                                                                           turn=os.path.relpath(os.path.join(SRC, "turn.jsonl"), ROOT),
                                                                           hold=os.path.relpath(os.path.join(WORK, "hold.jsonl"), ROOT)),
                          band_deg=BAND_DEG,
                          rule="five layouts centred on the band at the simulation's step of 0.125 in t (the band's "
                               "middle layout and two on each side); a cylinder without a band is centred on its "
                               "layout of largest median held turn",
                          decision=f"stage 2 if a band cell succeeds in >= {PASS_BAND} of {N_TURN}, both edge cells in "
                                   f"<= {PASS_EDGE} of {N_TURN}, and the drop load's rank correlation with standoff "
                                   f"<= {RHO_MAX}",
                          push_test=dict(cell=dict(d_mm=10, t=push_cell["t"], turn_cand=push_cell["turn_cand"]),
                                         step_N=PUSH_STEP_N, hold_s=PUSH_HOLD_S,
                                         predictions={f"{fi}@{pl}": {k: (round(v, 4) if isinstance(v, float) else v)
                                                                     for k, v in p.items()}
                                                      for (pl, fi), p in preds.items()})),
                hours=dict(stage1=H1, stage2=H2), cells=cells)
    with open(os.path.join(D, "plan.json"), "w") as fh:
        json.dump(plan, fh, indent=1)
    return len(cells)


def main():
    os.makedirs(D, exist_ok=True)
    L, prot, objs, servo, work, src_hold = load()
    C = cylinder_cells(*servo)
    C1 = cylinder_cells(*work)
    worst = check_against_landscape(C1, L)
    n_rep = check_work_rows(work[1], src_hold)
    meta = prot["meta"]
    H1 = hours(N_CELLS * len(STAGE1), 1, meta, push=True)
    H2 = hours(N_CELLS * len(STAGE2), 2, meta)
    S = old_protocol_stats(prot, L)
    p10 = plan_cells(C, 10)
    push_cell = max((c for c in p10 if c["turn_med"] >= BAND_DEG), key=lambda c: (c["turn_succ"], c["turn_med"]))
    preds = {(pl, fi): push_prediction(push_cell, fi, plant(pl)) for pl in (PLANT, PLANT_1NM) for fi in ("index", "thumb")}
    cpu_min = sum(c["wall_s"] for d in CYL for c in plan_cells(C, d)) / 60
    planned = {(c["d"], c["t"]) for d in CYL for c in plan_cells(C, d)}
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": "2026-10-11"}
    v["GLOSSARY"] = glossary(C)
    v["PLANT"] = plant_section(C, C1, replay_fit(), policy_grips())
    sci, sci_d = science(C, C1, L)
    v["SCIENCE"] = sci + figure(svg_landscape(C, C1, planned),
                                "Simulated held turn of each layout&#8217;s turn grasp (left) and drop load of its hold "
                                "grasp (right) against the standoff of the mounts, for three cylinders, on the servo "
                                "plant (solid) and the 1&#8202;N&#8202;m plant (dashed). Filled markers are the planned "
                                "bench layouts. The dotted line marks the band threshold. Hover a marker for its layout "
                                "and value.")
    v["OLD"] = old_section(S, L, C, C1)
    v["STAGE0"] = stage0_section(objs)
    v["PUSH"] = push_section(push_cell, preds, cpu_min)
    v["STAGE1"] = stage1_section(C, C1, H1)
    v["STAGE2"] = stage2_section(C, C1, H2)
    v["TIME"] = time_section(H1, H2, S)
    v["OUTCOMES"] = outcomes_section(C, C1)
    v["NEXT"] = next_section(C, large_without_band(work[0]))
    v["LEDE"] = lede(C, C1, H1, H2, sci_d)
    if TAB[0] != 4 or FIG[0] != 1:
        raise SystemExit(f"tables {TAB[0]}, figures {FIG[0]}: the cross-references in the prose assume 4 and 1")
    n = write_plan(C, C1, H1, H2, push_cell, preds)
    v["FOOTER"] = (f"<p>Rebuild: <code>.venv/bin/python scripts/bench_plan_page.py</code>; it writes "
                   f"<code>{REL}/plan.json</code> ({n} cells). Servo-plant rows: <code>{REL}/servo_plant/</code> "
                   f"(<code>hold.jsonl</code>, <code>turn.jsonl</code>; <code>hand_object_scale_sim.py run --plant "
                   f"{PLANT} --set kp_model=0.72</code>). 1&#8202;N&#8202;m plant: <code>turn.jsonl</code> of "
                   "<code>docs/experiments/20261008-hand_object_scale/</code>, whose turn scores agree with "
                   f"<code>landscape.json</code> to {worst:.2f}&#176;, and <code>{REL}/work_plant/hold.jsonl</code>, "
                   f"whose 3&#8202;mm pull-outs equal those of 8 October in all {n_rep} grasps. Study files: "
                   "<code>landscape.json</code>, <code>protocol.json</code>, <code>objects.json</code> of 8 October.</p>")
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
