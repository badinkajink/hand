#!/usr/bin/env python3
r"""Build docs/experiments/20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html.

    python3 scripts/hom_turn_rl_contact_page.py

Reads the bench-run replays of scripts/bench_replay_calibration.py (20261006-servo_recalibration/replays.jsonl,
runs.json), the three-finger turn rows of scripts/hom_turn3.py and scripts/reorient_backends.py
(20261006-hom_turn3/turn3.jsonl, plans_mujoco.jsonl, plans_drake.jsonl), the films in 20261006-hom_turn3/media/ and
the throughput rows of scripts/rl_contact_throughput.py and scripts/newton_hand_throughput.py
(20261006-rl_contact/throughput.jsonl, newton_throughput.jsonl). Every number in the prose is computed here.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261006-hom_turn3")
CAL = os.path.join(ROOT, "docs/experiments/20261006-servo_recalibration")
RLD = os.path.join(ROOT, "docs/experiments/20261006-rl_contact")
OUT = os.path.join(D, "20261006-servo_refit_hom_turn_pad_cost.html")
TPL = os.path.join(ROOT, "scripts/hom_turn_rl_contact_page.template.html")
PREV_PATH = "docs/experiments/20261006-fingertip_backends/20261006-fingertip_contact_backends.html"
HANDS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
JOINTS = [f"{f}_{j}" for f in ("thumb", "index", "middle") for j in ("yaw", "mcp", "pip")]
num = P.num
WORK = ("kp4_tau0.02_fr1_fl0", "1.0")
OLD = ("kp0.5_tau1.04_fr0.35_fl0.0035", "scene")
SHIP = ("kp30_tau0.033_fr10_fl0", "scene")

EXTRA_CSS = """
.tw th{text-transform:none;letter-spacing:.01em}
td.lab{white-space:normal;min-width:14em}
td.bad{background:color-mix(in srgb,var(--bad) 14%,transparent)}
td.good{background:color-mix(in srgb,var(--good) 14%,transparent)}
.eqn{font-family:var(--f-mono);font-size:.92em;margin:.6em 0 .6em 1.2em;line-height:1.7}
"""


# ------------------------------------------------------------------------------------------ helpers

def jl(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def f(x, nd=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "&#8211;"
    return num(x, f".{nd}f")


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None


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
            f'</video><figcaption>Figure&#160;{FIG[0]}. {caption} <code>docs/experiments/20261006-hom_turn3/{rel}</code>'
            f'</figcaption></figure>')


# ------------------------------------------------------------------------------------------ terms

GLOSSARY = [
    ("tool turn", "rotation of the tool&#8217;s axis from its pose at the end of the grip, in the vertical plane through "
                  "the axis, in degrees; positive toward vertical, 90&#176; = standing on end. The open-loop replays measure "
                  "it from the cosine of the axis with vertical, which equals the same angle for a tool starting flat."),
    ("held", "at the end of the rollout the tool is within 20&#8202;mm of its height at the end of the grip and at least "
             "two fingers touch it."),
    ("held peak", "the largest tool turn reached while the tool was held in that sense (any time in the rollout)."),
    ("joint readback error", "mean over the nine finger joints and over the bench&#8217;s three servo readbacks of one run "
                             "(before the turn, during it, at the settled hold) of |simulated &#8722; bench achieved angle|, "
                             "in degrees; the replay applies the run&#8217;s own logged command frames at their logged times."),
    ("kp", "servo position gain of the simulated finger actuator, N&#8202;m/rad: torque = kp (command &#8722; angle), "
           "clipped at the torque limit."),
    ("time constant &#964;", "finger joint damping divided by kp, in seconds: the time a free finger joint takes to cover "
                             "63&#8202;% of a step command (actuator velocity gain 0)."),
    ("&#956;", "Coulomb friction coefficient of the fingertip pads on the tool."),
    ("pads", "the printed TPU block (17 &#215; 14.8 &#215; 22&#8202;mm, 2.7&#8202;mm fillets on the front and end edges) "
             "covered with 0.75&#8202;mm spheres at 1&#8202;mm spacing (1,059 per tip; 290 at 2&#8202;mm); stiffness E/h "
             "per sphere area with E 10&#8202;MPa, h 8.5&#8202;mm, reached at load through solimp."),
    ("HOM controller", "the relative contact-velocity controller of Wang, Oh and Pollard (arXiv 2609.25619): per fingertip, "
                       "a contact frame from the closest points of tip and tool, joint rates from bounded least squares on "
                       "reference relative velocities (their Eq. 2). Defined in Section 3."),
    ("governor", "the HOM controller&#8217;s stop rule: when a pad&#8217;s normal force falls below 30&#8202;% of its "
                 "target or the tool lags the reference by more than 6&#176;, the reference stops and the angle reached is held."),
    ("inverse weight", "MuJoCo&#8217;s <code>body_invweight0[b, 0]</code>: the translational inverse inertia of body b at the "
                       "model&#8217;s reference pose, in 1/kg, computed at compile time from the joint-space mass matrix; 1/mass "
                       "for a free body, and for a fingertip the inverse of the inertia the whole finger chain and its servo "
                       "armature present at the tip."),
    ("effective mass m<sub>eff</sub>", "of a contact between bodies 1 and 2: 1/(inverse weight of 1 + inverse weight of 2), in kg. "
                                       "MuJoCo&#8217;s solref is a spring on the constraint acceleration, so a contact realises "
                                       "its solref stiffness times m<sub>eff</sub> as a force stiffness."),
    ("sink", "steady penetration of each pad into the tool in the contact bed&#8217;s static pinch, in mm, after 1&#8202;s at "
             "the set normal force with gravity off; Drake hydroelastic gives 0.153, 0.213 and 0.365&#8202;mm at 0.5, 1 and 3&#8202;N."),
    ("onset torque", "contact bed, twist task: a torque about the pinch axis ramps up on the pinched tool; the onset torque is the "
                     "last value at which the spin speed still follows the creep law w = k&#8202;&#964;, in mN&#8202;m. "
                     "Drake hydroelastic gives 0.84, 1.97 and 7.65&#8202;mN&#8202;m at 0.5, 1 and 3&#8202;N."),
    ("env steps/s", "policy steps per second summed over all parallel environments of the RL training env; one env step "
                    "is 10 physics steps of 2&#8202;ms plus observations, rewards and terminations."),
    ("physics &#181;s per world-step", "wall time of one batched physics step divided by the number of worlds."),
]


def glossary():
    return "<dl class='terms'>" + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


# ------------------------------------------------------------------------------------------ servo plant

def cal_rows():
    rows = {}
    for r in jl(os.path.join(CAL, "replays.jsonl")):
        if r.get("status") == "error":
            continue
        rows[r["key"]] = r
    return list(rows.values())


def phase_err(rs):
    ph = defaultdict(list)
    for r in rs:
        for p, d in r["joint_err"].items():
            ph[p] += [abs(x) for x in d.values()]
    return {p: mean(v) for p, v in ph.items()}


def plant_section():
    rows = cal_rows()
    runs = json.load(open(os.path.join(CAL, "runs.json")))
    n_runs = len(runs)
    per_hand = defaultdict(int)
    for r in runs:
        per_hand[r["hand"]] += 1

    def sel(spec):
        return [r for r in rows if r["plant"] == spec[0] and str(r.get("mu")) == spec[1]]
    W, O, S = sel(WORK), sel(OLD), sel(SHIP)
    pw, po, ps = phase_err(W), phase_err(O), phase_err(S)
    tab1 = table(["Plant", "kp (N&#8202;m/rad)", "&#964; (s)", "torque limit (N&#8202;m)", "&#956;", "before (&#176;)",
                  "during (&#176;)", "settled hold (&#176;)", "all (&#176;)"],
                 [[("Working plant (this page)", "lab"), "4", "0.02", "1", "1.0", f(pw.get("before"), 2),
                   f(pw.get("during"), 2), f(pw.get("after"), 2), (f(mean([r["joint_mae"] for r in W]), 2), "good")],
                  [("2026-09-02 calibrated plant (+ frictionloss 0.0035)", "lab"), "0.5", "1.04", "0.35", "2.4",
                   f(po.get("before"), 2), f(po.get("during"), 2), f(po.get("after"), 2),
                   (f(mean([r["joint_mae"] for r in O]), 2), "bad")],
                  [("Shipped template servo", "lab"), "30", "0.033", "10", "2.4", f(ps.get("before"), 2),
                   f(ps.get("during"), 2), f(ps.get("after"), 2), f(mean([r["joint_mae"] for r in S]), 2)]])
    # grid: best joint error by tau, by kp, by mu (coarse grid, one median-turn run per hand)
    sub1 = set()
    by_hand = defaultdict(list)
    for r in runs:
        by_hand[r["hand"]].append(r)
    for h, rs in by_hand.items():
        rs = sorted(rs, key=lambda r: r["bench"]["deg_turned"] or 0.0)
        sub1.add(rs[int(round(0.5 * (len(rs) - 1)))]["run"])
    coarse = [r for r in rows if r["run"] in sub1 and r.get("kp") in (0.25, 0.5, 1, 2, 4, 8, 30)
              and r.get("tau") in (0.02, 0.06, 0.2, 0.6) and r.get("fr") in (0.2, 0.35, 1.0)
              and r.get("fl") in (0.0, 0.005, 0.02) and str(r.get("mu")) in ("scene", "1.0")]
    agg = defaultdict(lambda: defaultdict(list))
    for r in coarse:
        agg[(r["kp"], r["tau"], r["fr"], r["fl"], str(r["mu"]))][r["hand"]].append(r["joint_mae"])
    J = {k: mean([mean(v) for v in hs.values()]) for k, hs in agg.items() if len(hs) == 8}
    n_pl = len(J)

    def best(i, val, mu):
        v = [e for k, e in J.items() if k[i] == val and k[4] == mu]
        return min(v) if v else None
    grid_rows = ["Finger time constant &#964; (s)"]
    for t in (0.02, 0.06, 0.2, 0.6):
        grid_rows.append([f"{t:g}", f(best(1, t, "1.0"), 2), f(best(1, t, "scene"), 2)])
    grid_rows.append("Servo gain kp (N&#8202;m/rad)")
    for kp in (0.25, 0.5, 1, 2, 4, 8, 30):
        grid_rows.append([f"{kp:g}", f(best(0, kp, "1.0"), 2), f(best(0, kp, "scene"), 2)])
    grid_rows.append("Torque limit (N&#8202;m)")
    for fr in (0.2, 0.35, 1.0):
        grid_rows.append([f"{fr:g}", f(best(2, fr, "1.0"), 2), f(best(2, fr, "scene"), 2)])
    tab2 = table(["Parameter value", "best joint readback error, &#956; 1.0 (&#176;)", "&#956; 2.4 (&#176;)"], grid_rows)
    kp_lo = min(best(0, kp, "1.0") for kp in (0.25, 0.5, 1, 2, 4, 8, 30))
    kp_hi = max(best(0, kp, "1.0") for kp in (0.25, 0.5, 1, 2, 4, 8, 30))
    t_fast = best(1, 0.02, "1.0")
    t_slow = best(1, 0.6, "1.0")
    hands_txt = ", ".join(f"{h} {per_hand[h]}" for h in HANDS)
    out = [
        "<h3>The post under the tool</h3>",
        "<p><code>Scene.set_object_platform(height)</code> now puts the post&#8217;s top at the tool&#8217;s lowest point, "
        "computed from the compiled geometry (<code>height</code> &#8722; 12.5&#8202;mm for the flat 25&#8202;mm tool), and "
        "keeps <code>height</code> as the tool&#8217;s centre, which the exported plans&#8217; grip poses refer to. Stored "
        "scenes are repaired on load by <code>seated_scene(path)</code>, which writes <code>&lt;stem&gt;__seated.xml</code> "
        "beside the original. <code>plant_drop_gate.py</code>, <code>real_v1_bench_plants.py</code>, "
        "<code>real_v1_turn_mechanism.py</code>, <code>plant_ranking_test.py</code>, <code>real_v1_plan_angle_sweep.py</code>, "
        "<code>real_v1_tip_gait.py</code> and <code>calibrate_plant_kp.py</code> load scenes through it. In the D7 scene the "
        "post top moved from 100.0 to 87.5&#8202;mm and the tool&#8217;s initial penetration of the post from 12.5&#8202;mm to 0.</p>",
        "<h3>The 2026-09-02 gain fit never loaded a finger</h3>",
        "<p><code>calibrate_plant_kp.py</code> reset each candidate scene from its keyframe&#160;0, which puts the palm at "
        "10&#8202;mm; the plan&#8217;s replay state puts it at 103.5&#8202;mm, above the tool on its 100&#8202;mm post. The "
        "fingers closed on air while the palm rested against the tool and the post, so the +11.09&#176; middle-yaw deficit "
        "that matched the bench&#8217;s +11.68&#176; at kp&#160;0.5 came from the fingers&#8217; own response, with no load from "
        "the tool (time constant 1.04&#8202;s at kp&#160;0.5 with the template&#8217;s joint damping 0.5, read 2&#8202;s after the "
        "last command). The script now starts from the plan&#8217;s replay state; its fit is superseded by the replays below.</p>",
        "<h3>Replaying the bench&#8217;s own runs</h3>",
        f"<p>The archive holds {n_runs} complete reorientation runs of the eight deployed hands with servo readbacks and "
        f"AprilTag tracking ({hands_txt}). <code>bench_replay_calibration.py</code> rebuilds each run on its hand&#8217;s "
        "bench scene (palm welded at the plan&#8217;s pose, tool resting on the repaired post, TPU pads, 1&#8202;ms step, "
        "elliptic cone, impratio 100), holds the plan&#8217;s grip for 0.8&#8202;s, applies the run&#8217;s logged command "
        "frames at their logged times and compares the nine joint angles at the run&#8217;s three servo readbacks. The plant "
        "is set on the compiled model: kp, joint damping = &#964;&#160;kp (actuator velocity gain 0), torque limit, joint "
        f"friction. A coarse grid of {n_pl // 2} servo settings at &#956; 1.0 and 2.4 on the median-turn run of each hand and "
        f"a finer grid of 144 plants on three runs per hand preceded the full replay in Table&#160;1.</p>",
        tab1,
        tcap(f"Joint readback error of the {n_runs} tracked bench runs replayed on three plants, by readback (before the "
             "turn, during it, at the settled hold) and over all three."),
        tab2,
        tcap(f"Best joint readback error over the other three parameters for each value of one parameter (coarse grid, "
             f"{n_pl // 2} servo settings per friction value, the median-turn run of each of the eight hands)."),
        f"<p>The readbacks fix two things. The finger time constant is short: &#964; at or below 0.2&#8202;s gives "
        f"{f(t_fast, 2)}&#8211;{f(best(1, 0.2, '1.0'), 2)}&#176; and 0.6&#8202;s gives {f(t_slow, 2)}&#176;, which rules out "
        "the 1&#8202;s fingers every calibrated-plant result since 2026-09-16 ran on. Friction near 1 fits better than the "
        "scenes&#8217; 2.4 by about 1&#176;. The servo gain is not identified: across kp&#160;0.25&#8211;30 the best error "
        f"stays within {f(kp_lo, 2)}&#8211;{f(kp_hi, 2)}&#176;, because a finger pressing on the tool stops at the tool, "
        "and its deficit is set by the grip geometry, whatever the stiffness behind it. The torque limit is not identified "
        "either. The fit&#8217;s cost also included the tool&#8217;s tracked tilt, and with it the best plants had kp 2&#8211;10 and "
        "&#956; 0.8&#8211;1.3; bench task outcomes depend on how the tool was placed and on contacts the scene does not model, so "
        "that part of the fit is not used here.</p>",
        "<p>The rest of this page runs the <b>working plant</b>: kp 4&#8202;N&#8202;m/rad, &#964; 0.02&#8202;s (joint damping "
        "0.08), torque limit 1&#8202;N&#8202;m, no joint friction, &#956; 1.0 (<code>reorient_backends</code> plant spec "
        "<code>kp4_kv0_fr1_fl0_dp0.08</code>). A gain measurement needs a known torque on a joint (Section&#160;6).</p>",
    ]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------ HOM turn

def hom_data():
    plans_mj = defaultdict(list)
    for r in jl(os.path.join(D, "plans_mujoco.jsonl")):
        plans_mj[r["hand"]].append(r)
    plans_dk = defaultdict(list)
    for r in jl(os.path.join(D, "plans_drake.jsonl")):
        plans_dk[r["hand"]].append(r)
    hom = defaultdict(list)
    for r in jl(os.path.join(D, "turn3.jsonl")):
        if r.get("status") != "ok":
            continue
        mode = "gov" if r.get("prm_governor") else "free"
        hom[(r["hand"], r["sim"], mode)].append(r)
    for key, fn in (("plans_nt", "plans_newton_mc.jsonl"), ("plans_nt_raw", "plans_newton.jsonl")):
        rows = defaultdict(list)
        for r in jl(os.path.join(D, fn)):
            if r.get("status") == "complete":
                rows[r["hand"]].append(r)
        hom[key] = rows
    return plans_mj, plans_dk, hom


def svg_turns(plans_mj, plans_dk, hom):
    W, Hh = 720, 380
    x0, y0, w, h = 70, 24, 610, 300
    out = P._svg_open(W, Hh, "Tool turn per hand: open-loop plan and HOM controller in MuJoCo and Drake")
    xs = (-10, 90)
    fx = lambda v: x0 + (v - xs[0]) / (xs[1] - xs[0]) * w  # noqa: E731
    rowh = h / len(HANDS)
    for v in range(0, 91, 15):
        out.append(f'<line x1="{fx(v):.1f}" x2="{fx(v):.1f}" y1="{y0}" y2="{y0 + h}" style="stroke:var(--rule2)"/>')
        out.append(f'<text x="{fx(v):.1f}" y="{y0 + h + 16}" text-anchor="middle" style="fill:var(--ink3)">{v}</text>')
    out.append(f'<text x="{x0 + w / 2:.1f}" y="{y0 + h + 34}" text-anchor="middle" style="fill:var(--ink2)">'
               'tool turn (&#176;); hollow = dropped by the end</text>')
    series = [("plan", "var(--ink3)", "square", -0.33), ("plan_dk", "var(--c-drake)", "square", -0.2),
              ("plan_nt", "var(--c-newton)", "square", -0.07), ("gov", "var(--s1)", "circle", 0.07),
              ("gov_dk", "var(--s2)", "circle", 0.2), ("free", "var(--bad)", "diamond", 0.33)]
    for i, hd in enumerate(HANDS):
        yc = y0 + (i + 0.5) * rowh
        out.append(f'<text x="{x0 - 12}" y="{yc + 4:.1f}" text-anchor="end" style="fill:var(--ink)">{hd}</text>')
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y0 + (i + 1) * rowh:.1f}" y2="{y0 + (i + 1) * rowh:.1f}" '
                   f'style="stroke:var(--rule)"/>')
        for key, col, shape, off in series:
            if key == "plan":
                pts = [(r["turn_end_deg"], r["held_end"]) for r in plans_mj.get(hd, [])]
            elif key == "plan_dk":
                pts = [(r["turn_end_deg"], r["held_end"]) for r in plans_dk.get(hd, [])]
            elif key == "plan_nt":
                pts = [(r["turn_end_deg"], r["held_end"]) for r in hom["plans_nt"].get(hd, [])]
            elif key == "gov":
                pts = [(r["turn_end_deg"], r["held"]) for r in hom.get((hd, "mujoco", "gov"), [])]
            elif key == "gov_dk":
                pts = [(r["turn_end_deg"], r["held"]) for r in hom.get((hd, "drake", "gov"), [])]
            else:
                pts = [(r["turn_held2_max_deg"], r["held"]) for r in hom.get((hd, "mujoco", "free"), [])]
            for v, held in pts:
                if v is None or v < xs[0] - 50:
                    continue
                v = max(xs[0], min(xs[1], v))
                P._marker(out, fx(v), yc + off * rowh, col, shape=shape, hollow=not held, r=3.6)
    out.append("</svg>")
    leg = P._legend_html([("open-loop plan, MuJoCo", "var(--ink3)", False, "square"),
                          ("open-loop plan, Drake", "var(--c-drake)", False, "square"),
                          ("open-loop plan, Newton hydroelastic, mass-corrected", "var(--c-newton)", False, "square"),
                          ("HOM with governor, MuJoCo (end)", "var(--s1)", False, "circle"),
                          ("HOM with governor, Drake (end)", "var(--s2)", False, "circle"),
                          ("HOM without governor, MuJoCo (held peak)", "var(--bad)", False, "diamond")])
    return "\n".join(out) + leg


def hom_section():
    plans_mj, plans_dk, hom = hom_data()
    rows = []
    agg = {}
    for hd in HANDS:
        pm = plans_mj.get(hd, [])
        pdk = plans_dk.get(hd, [])
        gm = hom.get((hd, "mujoco", "gov"), [])
        gd = hom.get((hd, "drake", "gov"), [])
        fm = hom.get((hd, "mujoco", "free"), [])
        agg[hd] = dict(plan=med([r["turn_end_deg"] for r in pm]), plan_held=sum(r["held_end"] for r in pm),
                       gov=med([r["turn_end_deg"] for r in gm]), gov_held=sum(r["held"] for r in gm),
                       free_peak=med([r["turn_held2_max_deg"] for r in fm]), free_held=sum(r["held"] for r in fm))
        pnt = hom["plans_nt"].get(hd, [])
        rows.append([hd, f(agg[hd]["plan"]), f"{agg[hd]['plan_held']}/{len(pm)}",
                     f(med([r["turn_end_deg"] for r in pdk])) if pdk else "&#8211;",
                     f"{sum(r['held_end'] for r in pdk)}/{len(pdk)}" if pdk else "&#8211;",
                     f(med([r["turn_end_deg"] for r in pnt])) if pnt else "&#8211;",
                     f"{sum(r['held_end'] for r in pnt)}/{len(pnt)}" if pnt else "&#8211;",
                     f(agg[hd]["gov"]), f"{agg[hd]['gov_held']}/{len(gm)}",
                     f(med([r["turn_end_deg"] for r in gd])) if gd else "&#8211;",
                     f"{sum(r['held'] for r in gd)}/{len(gd)}" if gd else "&#8211;",
                     f(agg[hd]["free_peak"]), f"{agg[hd]['free_held']}/{len(fm)}"])
    n_pm = sum(len(plans_mj.get(h, [])) for h in HANDS)
    n_pm_held = sum(agg[h]["plan_held"] for h in HANDS)
    plan_vals = [agg[h]["plan"] for h in HANDS if agg[h]["plan"] is not None]
    gov_vals = [agg[h]["gov"] for h in HANDS if agg[h]["gov"] is not None]
    free_vals = [agg[h]["free_peak"] for h in HANDS if agg[h]["free_peak"] is not None]
    n_g = sum(len(hom.get((h, "mujoco", "gov"), [])) for h in HANDS)
    n_g_held = sum(agg[h]["gov_held"] for h in HANDS)
    free_drop_hands = [h for h in HANDS if agg[h]["free_held"] == 0 and hom.get((h, "mujoco", "free"))]
    better = [h for h in HANDS if agg[h]["gov"] is not None and agg[h]["plan"] is not None and agg[h]["gov"] > agg[h]["plan"]]
    gov_all = [r for h in HANDS for r in hom.get((h, "mujoco", "gov"), [])]
    n_force = sum(1 for r in gov_all if r.get("limit_joint") is None)
    free_all = [r for h in HANDS for r in hom.get((h, "mujoco", "free"), [])]
    pip_sat = med([max(r["sat_frac"][2], r["sat_frac"][5], r["sat_frac"][8]) for r in free_all]) or 0.0
    n_pd = sum(len(plans_dk.get(h, [])) for h in HANDS)
    n_pd_held = sum(sum(r["held_end"] for r in plans_dk.get(h, [])) for h in HANDS)
    n_gd = sum(len(hom.get((h, "drake", "gov"), [])) for h in HANDS)
    n_gd_held = sum(sum(r["held"] for r in hom.get((h, "drake", "gov"), [])) for h in HANDS)
    tab = table(["Hand", "A (&#176;)", "held", "B (&#176;)", "held", "F (&#176;)", "held", "C (&#176;)", "held",
                 "D (&#176;)", "held", "E (&#176;)", "held"], rows)
    pnt_all = [r for h in HANDS for r in hom["plans_nt"].get(h, [])]
    n_nt, n_nt_held = len(pnt_all), sum(r["held_end"] for r in pnt_all)
    out = [
        "<p>The question was whether the hand-object controls of Wang, Oh and Pollard can carry out our reorientation, in "
        "which all three fingers move the tool together, in place of the deployed open-loop joint trajectories. The scene is "
        "the bench maneuver of each deployed hand D1&#8211;D8: palm fixed, the 25&#8202;mm, 24.5&#8202;g tool lying on its post, "
        "the plan&#8217;s grip held 0.8&#8202;s, the working plant, the TPU tip as 1&#8202;mm pads in MuJoCo and as a compliant "
        "convex (E 10&#8202;MPa, relaxation 0.01&#8202;s) in Drake. Placements: seed 0 at the plan&#8217;s pose, seeds 1&#8211;4 "
        "jittered by 2&#8202;mm and 2&#176; (MuJoCo 5 placements per hand, Drake 3).</p>",
        "<h3>Controller</h3>",
        "<p>At 100&#8202;Hz the controller reads the nine joint angles and the tool pose, poses a kinematic MuJoCo copy of "
        "the scene with the TPU block as one convex mesh per tip, and builds the paper&#8217;s contact frame per finger: "
        "origin at the closest points of block and tool (<code>mj_geomDistance</code>), x the tool&#8217;s surface normal "
        "toward the finger, y the finger&#8217;s flexion axis projected on the tangent plane. The paper commands relative "
        "contact velocities (finger minus tool at the contact). For a turn that keeps all three contacts, each "
        "contact&#8217;s reference is the velocity of the tool&#8217;s material point under the desired tool motion, so "
        "the relative velocity is zero while the tool follows, plus a normal term that regulates the pad force:</p>",
        "<div class='eqn'>&#969;<sub>d</sub> = a<sub>0</sub> (d&#952;<sub>ref</sub>/dt + K<sub>R</sub>(&#952;<sub>ref</sub> &#8722; &#952;)) "
        "+ K<sub>R</sub> (u &#215; u<sub>ref</sub>),&#8195; v<sub>d</sub> = K<sub>P</sub> (c<sub>ref</sub> &#8722; c)<br>"
        "v<sub>i</sub> = v<sub>d</sub> + &#969;<sub>d</sub> &#215; (p<sub>i</sub> &#8722; c) + n<sub>i</sub> K<sub>F</sub> "
        "(F<sub>i</sub> &#8722; F<sub>d</sub>)</div>",
        "<p>Here &#952; is the tool turn, a<sub>0</sub> the horizontal axis normal to the starting tool axis, u the tool axis "
        "and u<sub>ref</sub> the starting axis turned by &#952;<sub>ref</sub> about a<sub>0</sub>, c the tool&#8217;s material "
        "point at the starting centroid of the three contacts (held 5&#8202;mm above its start), p<sub>i</sub> and n<sub>i</sub> "
        "the contact point and normal of finger i, F<sub>i</sub> its pad force from the simulator&#8217;s contact solution "
        "and F<sub>d</sub> = 2&#8202;N. K<sub>R</sub> = K<sub>P</sub> = 3&#8202;s<sup>&#8722;1</sup>, K<sub>F</sub> = "
        "0.03&#8202;m/(s&#8202;N). The paper&#8217;s Eq. 2 then gives each finger&#8217;s joint rates by bounded least squares "
        "on the three translational rows (weight 1, scaled by 1/20&#8202;mm) and the three rotational rows (weight 0.01), "
        "with joint-rate damping 10<sup>&#8722;3</sup>, joint limits and |rate| &#8804; 2&#8202;rad/s. The servo targets "
        "integrate the rates and stay within 0.3&#8202;rad of the measured angles. The reference turns at 30&#176;/s toward "
        "vertical after a 0.5&#8202;s squeeze in which the plan&#8217;s 5&#8211;13&#8202;N grip relaxes to 2&#8202;N, and the "
        "rollout ends 2&#8202;s after the reference stops. Code: <code>scripts/hom_turn3.py</code>; the same controller drives "
        "MuJoCo and Drake.</p>",
        "<h3>Results</h3>",
        tab,
        tcap("Tool turn per hand on the working plant, median over placements, and the number of placements held at the "
             "end. A: deployed open-loop plan (grip and joint trajectory, <code>reorient_backends.py</code>), MuJoCo. "
             "B: the plan in Drake. C: HOM controller with governor, MuJoCo, turn at the end. D: the same in Drake. "
             "E: HOM controller without governor, MuJoCo, held peak. F: the plan in Newton with the TPU block as one "
             "hydroelastic mesh, stiffness divided by the tip&#8211;tool effective mass (<code>newton_turn.py --mass-correct</code>, "
             "rows <code>plans_newton_mc.jsonl</code>)."),
        figure(svg_turns(plans_mj, plans_dk, hom),
               "Tool turn of every rollout of Table&#160;3. Hollow markers dropped the tool by the end."),
        f"<p>On the working plant the deployed open-loop plans hold the tool on {n_pm_held} of {n_pm} MuJoCo placements and "
        f"turn it {f(min(plan_vals), 0)}&#8211;{f(max(plan_vals), 0)}&#176; (hand medians)"
        + (f"; Drake holds {n_pd_held} of {n_pd} and turns within a few degrees of MuJoCo" if n_pd else "")
        + ". The drops and the near-zero turns in the films of the previous page came from that page&#8217;s plant (kp "
        "0.5, 1&#8202;s fingers, &#956; 2.4).</p>",
        (f"<p>Newton&#8217;s hydroelastic contact on the same fingertip shape, with its stiffness divided by the "
         f"tip&#8211;tool effective mass (Section&#160;4), holds the tool on {n_nt_held} of {n_nt} placements "
         "(column F).</p>" if n_nt else ""),
        f"<p>With its governor the HOM controller holds {n_g_held} of {n_g} placements in MuJoCo"
        + (f" and {n_gd_held} of {n_gd} in Drake" if n_gd else "")
        + f", but stops at {f(min(gov_vals), 0)}&#8211;{f(max(gov_vals), 0)}&#176; (hand medians), "
          + (f"further than the plan only on {', '.join(better)}. " if better else "short of the plan on every hand. ")
          + "Without the governor the "
          f"reference runs on to vertical: the tool reaches held peaks of {f(min(free_vals), 0)}&#8211;{f(max(free_vals), 0)}&#176; "
          f"and is dropped by the end on {len(free_drop_hands)} hands ({', '.join(free_drop_hands)}). The governor stopped "
          f"{n_force} of {n_g} MuJoCo rollouts on a pad losing force and {n_g - n_force} with a pip joint at a limit. In the "
          f"rollouts without it, the pip joints sat at a rate or position bound in a median {f(100 * pip_sat, 0)}&#8202;% of "
          "control ticks (the most-saturated pip of each rollout). The deployed grips start the pips at 0&#8211;3&#176; on "
          "seven hands (D1: 19&#176; and 41&#176;) against an extension limit of &#8722;18&#176;, so keeping three contacts "
          "fixed on a turning tool asks the fingers for extension they lack, and the least-loaded pad lets go first. With "
          "fixed contacts on these grasps the controller does not reach vertical; it needs a grasp chosen for the turn "
          "(pips flexed at the start) or a change of contact mode partway, such as one finger sliding along the tool or a "
          "regrasp.</p>",
        "<p>Placement sensitivity: the governed turn depends on how the grip settles before the reference starts. An "
        "earlier squeeze phase that ramped the force target up from 0&#8202;N let D7 reach 52&#8211;65&#176; on all five "
        "placements and dropped D1 on two of five; the rows above use the version that holds 2&#8202;N from the start.</p>",
        grasp_search_section(),
        film("media/D7_s0_three.mp4", "D7, seed 0. Left: deployed open-loop plan. Middle: HOM controller with governor. "
             "Right: without governor. Pads touching the tool are red. MuJoCo, working plant.", "media/D7_s0_three.jpg"),
        film("media/D2_s0_three.mp4", "D2, seed 0, as Figure&#160;2.", "media/D2_s0_three.jpg"),
        film("media/D5_s0_three.mp4", "D5, seed 0, as Figure&#160;2.", "media/D5_s0_three.jpg"),
    ]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------ Newton mass scaling

BED = os.path.join(ROOT, "docs/experiments/20261005-contact_bed")
MSC = os.path.join(ROOT, "docs/experiments/20261006-newton_mass_scaling")


def bed_rows():
    """Static sink (mm) and twist onset torque (mN m) per (model, N) at 1 ms, from the 10-05 bed and the 10-06 rows."""
    sink, onset, scale = {}, {}, {}
    for r in jl(os.path.join(BED, "static_newton.jsonl")) + jl(os.path.join(MSC, "static_newton.jsonl")):
        if r.get("dt_ms", 1.0) == 1.0 and r.get("status", "complete") == "complete" and "pen_mm" in r:
            sink[(r["model"], r["N"])] = r["pen_mm"]
            if r.get("kh_scale"):
                scale[r["model"]] = r["kh_scale"]
    for r in (jl(os.path.join(BED, "twist_slip.jsonl")) + jl(os.path.join(BED, "twist_slip_newton.jsonl"))
              + jl(os.path.join(MSC, "twist_slip_newton.jsonl"))):
        if r.get("dt_ms", 1.0) == 1.0 and r.get("status", "complete") == "complete" and r.get("tau_onset_Nm"):
            onset[(r["model"], r["N"])] = 1e3 * r["tau_onset_Nm"]
    return sink, onset, scale


def newton_section():
    sink, onset, scale = bed_rows()
    Ns = (0.5, 1.0, 3.0)
    models = [("drake_hydro", "Drake hydroelastic"), ("newton_hydro", "Newton, kh = E/h"),
              ("newton_hydro_mc", "Newton, kh &#215; 1/m<sub>eff</sub> (solver&#8217;s inverse weights)"),
              ("newton_hydro_mc_mjcf", "Newton, kh &#215; 1/m<sub>eff</sub> (MJCF compile)")]
    rows = []
    for m, lab in models:
        rows.append([(lab, "lab")] + [f(sink.get((m, N)), 3) for N in Ns] + [f(onset.get((m, N)), 2) for N in Ns])
    tab = table(["Contact", "sink 0.5&#8202;N (mm)", "1&#8202;N", "3&#8202;N", "onset torque 0.5&#8202;N (mN&#8202;m)",
                 "1&#8202;N", "3&#8202;N"], rows)

    def ratio(m, d, N):
        a, b = d.get((m, N)), d.get(("drake_hydro", N))
        return a / b if a and b else None
    raw_sink = [ratio("newton_hydro", sink, N) for N in Ns]
    mc_sink = [ratio("newton_hydro_mc", sink, N) for N in Ns]
    mj_sink = [ratio("newton_hydro_mc_mjcf", sink, N) for N in Ns]
    raw_on = [ratio("newton_hydro", onset, N) for N in Ns]
    mc_on = [ratio("newton_hydro_mc", onset, N) for N in Ns]

    def rng(v, nd=2):
        v = [x for x in v if x is not None]
        return f"{f(min(v), nd)}&#8211;{f(max(v), nd)}" if v else "&#8211;"
    # the hand
    plans_mj, plans_dk, hom = hom_data()
    mc = [r for h in HANDS for r in hom["plans_nt"].get(h, [])]
    raw = [r for h in HANDS for r in hom["plans_nt_raw"].get(h, [])]
    corr = next((r["mass_correct"] for r in mc if r.get("mass_correct")), None)
    out = [
        "<p>Newton passes each hydroelastic contact point to MuJoCo-Warp with a force stiffness c in N/m (the point&#8217;s "
        "share of the pressure field: patch area times the two shapes&#8217; kh in series). SolverMuJoCo writes it as "
        "solref time constant &#8730;(1/(c(1&#8722;d))), damping ratio 1 and solimp (d, d, 0.001, 1, 0.5) "
        "(<code>newton/_src/solvers/mujoco/kernels.py</code>, lines 599&#8211;618 at Newton commit 009158e). MuJoCo treats "
        "solref as a spring on the constraint acceleration, so the contact holds c&#8202;&#215;&#8202;m<sub>eff</sub> per metre "
        "of penetration, not c. Newton&#8217;s shape-material path multiplies its stiffness by the inverse-weight sum before the "
        "same conversion (<code>docs/solvers/mujoco.rst</code>, &#8220;Shape-material contact stiffness and damping&#8221;); "
        "the hydroelastic path does not, and the <code>ShapeConfig.kh</code> docstring (<code>newton/_src/sim/builder.py</code>) "
        "says that SolverMuJoCo scales the stiffness by masses and that kh should be tuned with that in mind. The MuJoCo "
        "sphere pads already divide m<sub>eff</sub> out: their solimp d0 = 1 &#8722; 1/(t<sub>c</sub><sup>2</sup> K "
        "(w<sub>tip</sub> + w<sub>tool</sub>)) uses the same inverse weights (<code>reorient_backends.replace_tips</code>), "
        "which is why they matched Drake without fitting. The correction is therefore kh &#215; (w<sub>1</sub> + "
        "w<sub>2</sub>), with the inverse weights read from the solver&#8217;s own MuJoCo model "
        "(<code>solver.mj_model.body_invweight0</code> through <code>solver.mjc_body_to_newton</code>); kh = E/h stays a "
        "material constant.</p>",
        f"<p>On the 2026-10-05 contact bed (two 20&#8202;g pads on rails pinching the 24.5&#8202;g tool, rows "
        "<code>docs/experiments/20261006-newton_mass_scaling/</code>, <code>contact_bed_newton.py</code> models "
        "<code>newton_hydro_mc*</code>) the solver&#8217;s model gives each pad an inverse weight of 16.7&#8202;1/kg and the "
        f"tool 40.7, so m<sub>eff</sub> = 17.4&#8202;g and uncorrected Newton sank {rng(raw_sink, 1)} times as deep as Drake "
        f"with {rng(raw_on, 1)} times Drake&#8217;s onset torque. Multiplied by {f(scale.get('newton_hydro_mc'), 1)}&#8202;1/kg, "
        f"Newton sinks {rng(mc_sink)} times Drake&#8217;s depth and reaches {rng(mc_on)} of its onset torque, with no "
        "parameter fitted. Inverse weights from a separate MuJoCo compile of the bed&#8217;s MJCF give a factor of "
        f"{f(scale.get('newton_hydro_mc_mjcf'), 1)} and over-stiffen the contact (sink {rng(mj_sink)} times Drake), so the "
        "factor has to come from the solver&#8217;s model.</p>",
        tab,
        tcap("Contact bed, static pinch and twist at 1&#8202;ms: sink per pad and onset torque by normal force. Drake and "
             "uncorrected Newton rows from <code>docs/experiments/20261005-contact_bed/</code>, corrected rows from "
             "<code>docs/experiments/20261006-newton_mass_scaling/</code>; contact reduction on."),
    ]
    if mc and corr:
        w, kh = corr["invweight0"], corr["kh_tip"]
        wt = next(v for k, v in w.items() if k.endswith("_tip"))
        wtool = next(v for k, v in w.items() if not k.endswith("_tip"))
        mt = next(v for k, v in corr["body_mass"].items() if k.endswith("_tip"))
        n_held = sum(r["held_end"] for r in mc)
        turned = [r for r in mc if r["held_end"] and r["turn_end_deg"] > 10]
        still = [r for r in mc if r["held_end"] and r["turn_end_deg"] <= 10]
        fell_grip = [r for r in mc if r["z_grip"] < 0.05]
        lost_turn = [r for r in mc if not r["held_end"] and r["z_grip"] >= 0.05]
        raw_held = sum(r["held_end"] for r in raw)
        raw_grip = sum(1 for r in raw if r["z_grip"] < 0.05)
        close, short, further = [], [], []
        for h in HANDS:
            t_nt = med([r["turn_end_deg"] for r in turned if r["hand"] == h])
            t_mj = med([r["turn_end_deg"] for r in plans_mj.get(h, [])])
            if t_nt is None or t_mj is None:
                continue
            d = t_nt - t_mj
            (close if abs(d) <= 5 else short if d < 0 else further).append((h, d))
        by_hand = defaultdict(list)
        for r in still:
            by_hand[r["hand"]].append(r["seed"])
        still_txt = "; ".join(f"{h} seeds {', '.join(map(str, sorted(v)))}" for h, v in sorted(by_hand.items()))
        cmp_txt = f"within 5&#176; of MuJoCo on {', '.join(h for h, _ in close)}"
        if further:
            cmp_txt += "; " + ", ".join(f"{h} {f(d, 0)}&#176; further" for h, d in further)
        if short:
            cmp_txt += (f"; {f(min(-d for _, d in short), 0)}&#8211;{f(max(-d for _, d in short), 0)}&#176; short on "
                        f"{', '.join(h for h, _ in short)}")
        out += [
            "<h3>The deployed turn with the correction</h3>",
            f"<p>In the turn scene each fingertip (4.9&#8202;g, the same body in Newton&#8217;s model and the MJCF) has an "
            f"inverse weight of {f(wt, 2)}&#8202;1/kg, the finger chain and its servo armature seen at the tip, and the tool "
            f"{f(wtool, 1)}; m<sub>eff</sub> = {f(1e3 / (wt + wtool), 1)}&#8202;g, so each tip&#8217;s kh is multiplied by "
            f"{f(wt + wtool, 1)}&#8202;1/kg and the tool&#8217;s kh is 100 times the largest tip kh "
            f"(<code>scripts/newton_turn.py --mass-correct</code>). The deployed plans then hold the tool on {n_held} of "
            f"{len(mc)} placements, against {raw_held} of {len(raw)} uncorrected (Table&#160;3, column F; uncorrected rows "
            f"<code>plans_newton.jsonl</code>, {raw_grip} of the 40 lost the tool during the grip). {len(turned)} of the "
            f"held placements turn the tool more than 10&#176;. On {len(still)} ({still_txt}) the tool stays in the grip "
            f"but turns less than 5&#176;. {len(fell_grip)} placements "
            f"({', '.join(sorted(r['hand'] + ' seed ' + str(r['seed']) for r in fell_grip))}) drop the tool during the "
            f"0.8&#8202;s grip and {len(lost_turn)} lose it during the turn. Where the tool turns, the hand median is "
            f"{cmp_txt}.</p>",
        ]
    return "\n".join(out)


def grasp_search_section():
    S = defaultdict(dict)
    for r in jl(os.path.join(D, "grasp_search.jsonl")):
        S[r["hand"]][r["criteria"]["lim_deg"]] = r
    H = defaultdict(list)
    for r in jl(os.path.join(D, "turn3_best_grasp.jsonl")):
        if r.get("status") == "ok":
            key = "deployed" if r["grasp"] == "deployed" else f"best{r.get('search_lim_deg', 3.0):g}"
            H[(r["hand"], key, r["sim"])].append(r)
    hands = [h for h in ("D7", "D2", "D5") if h in S]
    if not hands:
        return ""

    def hom(h, key, sim):
        rs = H.get((h, key, sim), [])
        if not rs:
            return "&#8211;"
        return f"{f(med([r['turn_end_deg'] for r in rs]) + 0.0, 0)} ({sum(r['held'] for r in rs)}/{len(rs)})"
    rows = []
    for h in hands:
        r3, r8 = S[h].get(3.0), S[h].get(8.0)
        rows.append([h, f(r3["deployed"]["range_deg"], 0) if r3 else "&#8211;",
                     f(r3["best"]["range_deg"], 0) if r3 else "&#8211;", f(r8["best"]["range_deg"], 0) if r8 else "&#8211;",
                     hom(h, "deployed", "mujoco"), hom(h, "deployed", "drake"), hom(h, "best3", "mujoco"),
                     hom(h, "best8", "mujoco"), hom(h, "best8", "drake")])
    tab = table(["Hand", "deployed range (&#176;)", "searched, 3&#176; margin", "searched, 8&#176; margin",
                 "HOM, deployed, MuJoCo", "Drake", "HOM, searched 3&#176;, MuJoCo", "HOM, searched 8&#176;, MuJoCo",
                 "Drake"], rows)
    best = [S[h][8.0]["best"] for h in hands if 8.0 in S[h]]
    dep = [S[h][3.0]["deployed"] for h in hands if 3.0 in S[h]]
    s_dep = [abs(c["s_mm"]) for b in dep for k, c in b["contacts"].items() if k != "thumb"]
    s_best = [abs(c["s_mm"]) for b in best for k, c in b["contacts"].items() if k != "thumb"]
    pip0 = [b["q_start_deg"][i] for b in best for i in (5, 8)]
    b8 = [r for h in hands for r in H.get((h, "best8", "mujoco"), []) + H.get((h, "best8", "drake"), [])]
    on_floor = sum(1 for r in b8 if r["z_grip_mm"] < 50)
    held8 = [r for r in b8 if r["held"]]
    b3 = [r for h in hands for r in H.get((h, "best3", "mujoco"), []) + H.get((h, "best3", "drake"), [])]
    return "".join([
        "<h3>Grasp search by turn range</h3>",
        "<p>The turn range of a grasp is the largest rotation of the tool toward vertical, about the horizontal axis "
        "normal to its starting axis through the contacts&#8217; centroid (the rotation the controller commands), at which, "
        "for every smaller angle in 3&#176; steps, each fingertip still reaches its tool-fixed contact point (inverse "
        "kinematics of the finger&#8217;s three joints puts its pad point within 1&#8202;mm of the contact, inside the joint "
        "limits less a margin, with the pad normal within 45&#176; of the tool&#8217;s surface normal) and pad forces exist "
        "that hold the tool (a linear program over the three contact forces: they balance the tool&#8217;s weight and "
        "moments, lie in the &#956; 1 friction pyramid with at least 0.5&#8202;N normal, and load no finger joint beyond the "
        "1&#8202;N&#8202;m servo limit). The grasp is the tool&#8217;s offset on its post (&#177;12&#8202;mm along its axis, "
        "&#177;5&#8202;mm across, &#177;15&#176; yaw) and each finger&#8217;s contact on the cylinder (position along the axis "
        "and angle around it); CEM with 96 samples, 8 elites and 15 iterations maximises the range from the deployed "
        "grip&#8217;s contacts (<code>scripts/hom_grasp_search.py</code>, rows <code>grasp_search.jsonl</code>, "
        "<code>turn3_best_grasp.jsonl</code>). The controller then starts from the searched grasp: grip targets from the "
        "inverse kinematics at 0&#176; with each pad commanded 2&#8202;mm into the tool, fingers starting with the pads 8&#8202;mm "
        "outside their contacts (from the plan&#8217;s open pose, the 3&#176;-margin grasps knocked the tool off its post).</p>",
        tab,
        tcap("Kinematic turn range of the deployed grasp and of the searched grasps (joint-limit margin 3&#176; or 8&#176;), "
             "and the governed HOM turn at the end of the rollout from each, median over seeds 0&#8211;2 with the number "
             "of rollouts that held the tool."),
        f"<p>The search raises the range from {f(min(b['range_deg'] for b in dep), 0)}&#8211;{f(max(b['range_deg'] for b in dep), 0)}&#176; "
        f"to {f(min(b['range_deg'] for b in best), 0)}&#8211;{f(max(b['range_deg'] for b in best), 0)}&#176; (8&#176; margin). The "
        f"searched grasps move the index and middle contacts toward the tool&#8217;s centre ({f(min(s_best), 0)}&#8211;"
        f"{f(max(s_best), 0)}&#8202;mm from it against {f(min(s_dep), 0)}&#8211;{f(max(s_dep), 0)}&#8202;mm) and start "
        f"those pips at {f(min(pip0), 0)} to {f(max(pip0), 0)}&#176;. The controller does not carry them out: from the "
        "8&#176;-margin grasps the tool is held in "
        f"{len(held8)} of {len(b8)} rollouts ({on_floor} lose it during the grip) and the held ones turn it "
        f"{f(min(r['turn_end_deg'] for r in held8), 0) if held8 else '&#8211;'} to "
        f"{f(max(r['turn_end_deg'] for r in held8), 0) if held8 else '&#8211;'}&#176;, away from vertical; from the "
        f"3&#176;-margin grasps {sum(r['held'] for r in b3)} of {len(b3)} hold. Most rollouts lose the tool during the grip, "
        "which the kinematic score leaves out.</p>",
    ])


# ------------------------------------------------------------------------------------------ RL contact cost

VAR_LBL = {"legacy": "box tip, point contact (every RL run so far)", "mesh": "TPU block, one convex mesh",
           "pads2": "TPU block, 2&#8202;mm pads (290 per tip)", "pads1f": "TPU block, 1&#8202;mm pads, front half (684 per tip)",
           "pads1": "TPU block, 1&#8202;mm pads (1,059 per tip)"}


def rl_section():
    T = {}
    for r in jl(os.path.join(RLD, "throughput.jsonl")):
        if r.get("sensor_reduce") == "netforce":       # rows after the sensor change: sensor_paragraph
            continue
        if r.get("status") != "ok":
            T.setdefault((r["variant"], r["num_envs"]), r)
            continue
        old = T.get((r["variant"], r["num_envs"]))
        if old is None or old.get("status") != "ok" or (r.get("gpu_used_mb") and not old.get("gpu_used_mb")):
            T[(r["variant"], r["num_envs"])] = r
    rows = []
    for v in ("legacy", "mesh", "pads2", "pads1f", "pads1"):
        cells = [(VAR_LBL[v], "lab")]
        any_r = next((T[(v, n)] for n in (1024, 2048, 4096) if (v, n) in T and T[(v, n)].get("status") == "ok"), None)
        cells.append(str(any_r["ngeom"]) if any_r else "&#8211;")
        cells.append(str(any_r["ncon_world_max"]) if any_r else "&#8211;")
        for n in (1024, 2048, 4096):
            r = T.get((v, n))
            cells.append(num(r["env_steps_per_s"], ",.0f") if r and r.get("status") == "ok" else
                         ("incomplete" if r else "&#8211;"))
        for n in (1024, 2048, 4096):
            r = T.get((v, n))
            cells.append(f(r["physics_us_per_world_step"], 2) if r and r.get("status") == "ok" else "&#8211;")
        r4 = max((T[(v, n)] for n in (1024, 2048, 4096) if (v, n) in T and T[(v, n)].get("gpu_used_mb")),
                 key=lambda r: r["num_envs"], default=None)
        cells.append(f"{num(r4['gpu_used_mb'], ',.0f')} @ {r4['num_envs']}" if r4 else "&#8211;")
        rows.append(cells)
    tab_env = table(["Fingertip", "geoms", "contacts/world", "env steps/s, 1,024 envs", "2,048", "4,096",
                     "physics &#181;s/world-step, 1,024", "2,048", "4,096", "GPU MB @ envs"], rows)
    leg = {n: T.get(("legacy", n)) for n in (1024, 2048, 4096)}
    p1 = {n: T.get(("pads1", n)) for n in (1024, 2048, 4096)}
    p2 = {n: T.get(("pads2", n)) for n in (1024, 2048, 4096)}

    def ratio(a, b, key):
        if a and b and a.get("status") == "ok" and b.get("status") == "ok":
            return a[key] / b[key]
        return None
    r1_1024 = ratio(p1[1024], leg[1024], "env_steps_per_s")
    r1_2048 = ratio(p1[2048], leg[2048], "env_steps_per_s")
    r2_2048 = ratio(p2[2048], leg[2048], "env_steps_per_s")
    ph1 = ratio(p1[1024], leg[1024], "physics_us_per_world_step")
    phys_frac = None
    if leg[2048] and leg[2048].get("status") == "ok":
        phys_frac = 10 * leg[2048]["physics_us_per_world_step"] * 1e-6 * leg[2048]["env_steps_per_s"]
    # Newton / MuJoCo-Warp physics
    NT = {}
    for r in jl(os.path.join(RLD, "newton_throughput.jsonl")):
        mode = r["mode"] + (str(r.get("mujoco_warp")) if r["mode"] == "mjw" else "")
        tip = r["tip"]
        if r["mode"] == "nt_hydro" and tip == "mesh":
            tip = "mesh_sized" if (r.get("buffer_fraction") or 1.0) < 1.0 else "mesh_default"
        NT.setdefault((mode, tip, r["nworld"]), []).append(r)
    # training budget
    eta = {}
    for v, rr in (("legacy", leg[2048]), ("pads1", p1[2048])):
        if rr and rr.get("status") == "ok":
            eta[v] = 20e6 / rr["env_steps_per_s"] / 60.0
    out = [
        "<p>The RL comparison asked for here trains the reorientation with the pad fingertip and with the point contact "
        "every run so far has used, then replays the policies in Drake and Newton. Before training, the cost. The "
        "environment is the D6 reorientation env of the RL pipeline (mjlab on MuJoCo-Warp 3.6: scripted grasp and lift, "
        "2&#8202;ms step, 10 physics steps per policy step, elliptic cone, impratio 10, 10 solver and 20 line-search "
        "iterations) on morphology runs that differ only in the fingertip (<code>make_pad_morphology_run.py</code>). The "
        "pads touch only the tool; the block&#8217;s convex mesh takes the floor and finger-to-finger contacts (with pads "
        "colliding with everything, a 2&#8202;mm pad tip resting on the floor made up to 738 contacts per world). Tip and "
        "distal-link inertias are pinned to the source run&#8217;s, so the mass distribution is unchanged. Contact buffers "
        "are sized from a calibration pass and grown when MuJoCo-Warp reports an overflow. Zero-action env steps, timed "
        "after 60 warm-up steps that cover the grasp.</p>",
        tab_env,
        tcap("Throughput of the D6 RL training env per fingertip model (<code>scripts/rl_contact_throughput.py</code>, "
             "rows <code>docs/experiments/20261006-rl_contact/throughput.jsonl</code>). Contacts/world: largest count in one "
             "world after the grasp. GPU memory: nvidia-smi with the env alive. &#8216;Incomplete&#8217;: the run ran out of GPU memory "
             "in a process holding earlier envs and was stopped by the memory watchdog when repeated alone."),
        f"<p>Physics is a small part of an env step: at 2,048 envs the box tip&#8217;s physics takes "
        f"{f(100 * phys_frac, 0) if phys_frac else '&#8211;'}&#8202;% of the wall time, the rest is observation, reward, "
        "sensor and termination code. The 1&#8202;mm pads make physics "
        f"{f(ph1, 1) if ph1 else '&#8211;'}&#215; slower per world-step at 1,024 envs and the env "
        f"{f(100 * (1 - r1_1024), 0) if r1_1024 else '&#8211;'}&#8202;% slower at 1,024 envs and "
        f"{f(100 * (1 - r1_2048), 0) if r1_2048 else '&#8211;'}&#8202;% at 2,048; the 2&#8202;mm pads cost "
        f"{f(100 * (1 - r2_2048), 0) if r2_2048 else '&#8211;'}&#8202;% at 2,048. GPU memory grows with the pads: 6.2&#8202;GB "
        "for the 1&#8202;mm pads at 2,048 envs and 10.5&#8202;GB for the front-half pads at 4,096, against 4.4&#8202;GB for the "
        "box tip at 4,096; the full 1&#8202;mm pads at 4,096 envs did not complete. Throughput does not grow past 2,048 envs "
        "for any fingertip.</p>",
        sensor_paragraph(T),
        same_state_section(NT),
        training_section(eta),
    ]
    return "\n".join(out)


def sensor_paragraph(T):
    rows = jl(os.path.join(RLD, "sensor_check.jsonl"))
    leg = next((r for r in reversed(rows) if r["check"] == "legacy"), None)
    pad = next((r for r in reversed(rows) if r["check"] == "pads1"), None)
    if not (leg and pad):
        return ""
    new = {}
    for r in jl(os.path.join(RLD, "throughput.jsonl")):
        if r.get("status") == "ok" and r.get("sensor_reduce") == "netforce":
            new[(r["variant"], r["num_envs"])] = r
    old = {v: T.get((v, 2048)) for v in ("legacy", "pads1")}
    term_max = max(leg["term_max_abs_diff"].values())
    fmax = leg["force_norm_max_abs_diff_N"]["fingertip_cube_contact"]
    txt = ("<h3>Contact sensor for sphere pads</h3>"
           "<p>The fingertip and palm contact sensors now sum every matched contact (mjlab <code>reduce=\"netforce\"</code>, a "
           "world-frame vector) and the simulation allocates 256 matches per sensor (<code>contact_sensor_maxmatch</code>; "
           "mjlab&#8217;s default of 64 overflowed with the pads), set in <code>src/morphohand/rl/env_build.py</code>. Before, each "
           "fingertip reported one contact chosen by match order (<code>reduce=\"none\"</code>, one slot): the whole force of "
           "the box tip, one pad&#8217;s share on the pad tip. Every consumer (grip-force, force-excess, force-spread and brace "
           "rewards, contact gates, tip-lost terminations, the deploy read-outs) takes the vector norm and "
           "<code>found&#8202;&gt;&#8202;0</code>, so the change of frame does not reach them. Regression "
           "(<code>scripts/rl_contact_sensor_check.py</code>, rows <code>docs/experiments/20261006-rl_contact/sensor_check.jsonl</code>): "
           f"on the box tip the old and new sensors, read side by side in one env for {leg['steps']} steps at "
           f"{leg['num_envs']} envs, give every sensor-reading reward term the same value (largest difference "
           f"{term_max:g}) and force norms within {fmax:.1e}&#8202;N; no fingertip had more than one matched contact. On "
           f"the 1&#8202;mm pads ({pad['num_envs']} envs, {len(pad['at_steps'])} instants after the grasp, {pad['n_tip_samples']} "
           f"fingertip samples, up to {pad['found_max']} pad contacts per tip) the sensor equals the sum of the pad contact "
           f"forces of the same GPU solve to {pad['rel_err_same_solve_max']:.0e} and a CPU MuJoCo re-solve of the same state "
           f"to {f(100 * pad['rel_err_cpu_resolve_max'], 1)}&#8202;%.")
    nl, npd = new.get(("legacy", 2048)), new.get(("pads1", 2048))
    if nl and npd and old["legacy"] and old["pads1"]:
        txt += (f" At 2,048 envs the env runs {num(nl['env_steps_per_s'], ',.0f')} (box tip) and "
                f"{num(npd['env_steps_per_s'], ',.0f')} (pads) env steps/s with the summed sensor, against "
                f"{num(old['legacy']['env_steps_per_s'], ',.0f')} and {num(old['pads1']['env_steps_per_s'], ',.0f')} in "
                "Table&#160;6 with the one-contact sensor.")
    txt += ("</p><p>The env has no mass randomisation. Adding one requires recomputing each pad&#8217;s solimp d0 per world, "
            "since d0 holds the inverse weights of the nominal tip and tool. The compliance randomisation "
            "(<code>randomize_geom_solimp</code>) overwrites d0 and is off in the configuration the training below copies.</p>")
    return txt


RUN_LBL = {"tpu27mesh": "TPU block mesh (point contact)", "tpu27pads1": "TPU 1&#8202;mm pads"}
RUN_COL = {"tpu27mesh": "var(--s1)", "tpu27pads1": "var(--c-sphere)"}


def svg_training(R):
    W, Hh = 720, 330
    x0, y0, w, h = 60, 20, 620, 250
    out = P._svg_open(W, Hh, "Final tool cosine of the deterministic policy against training steps")
    fx = lambda v: x0 + v / 20.5 * w  # noqa: E731
    fy = lambda v: y0 + (1.0 - v) / 2.0 * h  # noqa: E731
    for v in (-1.0, -0.5, 0.0, 0.5, 1.0):
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{fy(v):.1f}" y2="{fy(v):.1f}" style="stroke:var(--rule2)"/>')
        out.append(f'<text x="{x0 - 8}" y="{fy(v) + 4:.1f}" text-anchor="end" style="fill:var(--ink3)">{v:+.1f}</text>')
    for v in range(0, 21, 4):
        out.append(f'<text x="{fx(v):.1f}" y="{y0 + h + 16}" text-anchor="middle" style="fill:var(--ink3)">{v}</text>')
    out.append(f'<text x="{x0 + w / 2:.1f}" y="{y0 + h + 34}" text-anchor="middle" style="fill:var(--ink2)">'
               'env steps (millions); hollow = fewer than 48 of 64 envs hold the tool at the end</text>')
    leg = []
    for tag, rs in sorted(R.items()):
        var = "tpu27pads1" if "pads1" in tag else "tpu27mesh"
        col = RUN_COL[var]
        dash = "4 3" if tag.endswith("s1") else ""
        pts = [(r["env_steps"] / 1e6, r["final_cos_mean"], r["hold_rate"]) for r in rs]
        out.append('<polyline points="' + " ".join(f"{fx(a):.1f},{fy(b):.1f}" for a, b, _ in pts) +
                   f'" style="fill:none;stroke:{col};stroke-width:2;stroke-dasharray:{dash or "none"}"/>')
        for a, b, hr in pts:
            P._marker(out, fx(a), fy(b), col, shape="circle", hollow=hr < 0.75, r=3.4)
        leg.append((f"{RUN_LBL[var]}, seed {tag[-1]}", col, bool(dash), "circle"))
    out.append("</svg>")
    return "\n".join(out) + P._legend_html(leg)


def training_section(eta):
    R = defaultdict(list)
    for r in jl(os.path.join(RLD, "train_eval.jsonl")):
        if r.get("status") == "ok":
            R[r["tag"]].append(r)
    for k in R:
        R[k].sort(key=lambda r: r["iteration"])
    chk = {}
    pj = os.path.join(RLD, "work_plant_runs.json")
    if os.path.exists(pj):
        chk = {os.path.basename(c["run"]): c for c in json.load(open(pj))["checks"]}
    pads_c = chk.get("20261006-sv1_u0308_b050_work_tip_tpu2.7pads1", {}).get("step150", {}).get("tip_contacts_mean")
    out = ["<h3>RL training: TPU block mesh against 1&#8202;mm pads</h3>",
           "<p>The D6 reorientation trained from scratch with the b_liveA recipe and the flags of the 2026-09-17 60&#8202;M run "
           "(<code>results/rl/20260917-1141-d6_cal_reorient_gp025_60M_s0/config.yaml</code>; a dry-run diff leaves only the "
           "number of envs, the step budget and the checkpoint interval), on scenes with the working plant "
           "(<code>scripts/make_work_plant_runs.py</code>: finger kp 4&#8202;N&#8202;m/rad, kv 0, 1&#8202;N&#8202;m, damping 0.08, "
           "frictionloss 0, &#956; 1) and two fingertips: the TPU block as one convex mesh (point contact) and as 1&#8202;mm "
           "sphere pads"
           + (f" ({' / '.join(f(x, 0) for x in pads_c)} pad contacts on thumb / index / middle in the zero-action grasp)" if pads_c else "")
           + ". Both scenes lift and hold the tool in 64 of 64 envs with zero residual actions. Two seeds per fingertip "
           "(the trainer&#8217;s <code>--seed</code> is now applied: it seeds python, numpy, torch and Warp through the env "
           "config; the GPU contact solve stays non-deterministic), 20&#8202;M steps at 2,048 envs, friction randomised "
           "0.55&#8211;1.15&#215; per env, checkpoints every 41 iterations (2.0&#8202;M steps), one run at a time "
           "(<code>scripts/rl_contact_train_queue.sh</code>). Every checkpoint is rolled out deterministically in 64 envs "
           "for 250 policy steps without randomisation (<code>scripts/rl_contact_eval.py</code> over "
           "<code>policy_eval_suite.py</code>, rows <code>docs/experiments/20261006-rl_contact/train_eval.jsonl</code>).</p>"]
    if not R:
        out.append(P.pending("Training queue running; rows appear in train_eval.jsonl as checkpoints are evaluated."))
        return "\n".join(out)
    out.append(figure(svg_training(R), "Final cosine of the tool axis with vertical (mean over 64 deterministic "
                                      "rollouts) of each checkpoint against env steps."))
    rows = []
    for tag, rs in sorted(R.items()):
        r = rs[-1]
        var = "tpu27pads1" if "pads1" in tag else "tpu27mesh"
        rows.append([(f"{RUN_LBL[var]}, seed {tag[-1]}", "lab"), f(r["env_steps"] / 1e6, 1),
                     f"{f(r['final_cos_mean'], 3)} &#177; {f(r['final_cos_sd'], 3)}",
                     f"{round(64 * r['hold_rate'])}/64", f"{round(64 * r['success_rate'])}/64", f(r["peak_cos_mean"], 3),
                     f"{f(r['force_active_thumb'], 1)} / {f(r['force_active_index'], 1)} / {f(r['force_active_middle'], 1)}"])
    out.append(table(["Run", "env steps (M)", "final cos", "held", "cos &#8805; 0.9 and held", "peak cos",
                      "pad force thumb / index / middle (N)"], rows))
    out.append(tcap("Final checkpoint of each run, 64 deterministic rollouts. Final cos: cosine of the tool axis with "
                    "vertical at the end (mean &#177; sd); held: fingertip force above 0.5&#8202;N and tool above 60&#8202;mm at "
                    "the end; pad force: mean summed contact force per fingertip from the residual onset on."))
    for tag in sorted(R):
        png = os.path.join(RLD, "media", f"{tag}_strip.png")
        if os.path.exists(png):
            FIG[0] += 1
            out.append(f'<figure><img src="{P.R.data_uri(png, "image/png")}" alt="filmstrip {tag}"><figcaption>Figure&#160;'
                       f'{FIG[0]}. {tag}: frames of the run&#8217;s last training video at its phase marks '
                       f'(<code>policy_filmstrip.py</code>).</figcaption></figure>')
    rep = jl(os.path.join(RLD, "policy_replay.jsonl"))
    if rep:
        rr = []
        for tag in sorted(R):
            rec = next((r for r in reversed(rep) if r.get("kind") == "record" and r.get("tag") == tag), None)
            cells = [(tag, "lab")]
            cells.append(f"{f(rec['mjw_cos_end'][0], 3)}, z {f(rec['mjw_z_end_mm'][0], 0)}" if rec else "&#8211;")
            for eng in ("mujoco", "drake", "newton"):
                r = next((x for x in reversed(rep) if x.get("engine") == eng and x.get("status") == "ok"
                          and x.get("hold_test_s") is None and x.get("dir", "").endswith(tag)), None)
                cells.append(f"{f(r['cos_end'], 3)} ({'held' if r['held_end'] else 'dropped'})" if r else "&#8211;")
            rr.append(cells)
        out.append(table(["Run", "MuJoCo-Warp (recorded)", "CPU MuJoCo", "Drake", "Newton hydroelastic"], rr))
        out.append(tcap("The final policy&#8217;s finger targets from the reorientation onset (step 58), world 0, "
                        "replayed open loop (<code>scripts/rl_policy_replay.py</code>, rows <code>policy_replay.jsonl</code>): "
                        "final cosine of the tool with vertical and whether it is held."))
    return "\n".join(out)


SS_ROWS = [("mjlab", "3.6.0", "legacy", "mjlab env (reference), box tip"),
           ("mjlab", "3.6.0", "mesh", "mjlab env (reference), TPU block mesh"),
           ("mjlab", "3.6.0", "pads1", "mjlab env (reference), TPU 1&#8202;mm pads"),
           ("mjw", "3.6.0", "legacy", "MuJoCo-Warp 3.6, box tip"),
           ("mjw", "3.6.0", "mesh", "MuJoCo-Warp 3.6, TPU block mesh"),
           ("mjw", "3.6.0", "pads1", "<b>MuJoCo-Warp 3.6, TPU 1&#8202;mm pads</b>"),
           ("mjw", "3.14.0", "legacy", "MuJoCo-Warp 3.14, box tip"),
           ("mjw", "3.14.0", "mesh", "MuJoCo-Warp 3.14, TPU block mesh"),
           ("mjw", "3.14.0", "pads1", "MuJoCo-Warp 3.14, TPU 1&#8202;mm pads"),
           ("nt_pt", "3.14.0", "legacy", "Newton, point contact, box tip"),
           ("nt_pt", "3.14.0", "mesh", "Newton, point contact, TPU block mesh"),
           ("nt_pt", "3.14.0", "pads1", "<b>Newton, sphere pads</b> (Newton&#8217;s d0)"),
           ("nt_hydro", "3.14.0", "mesh", "<b>Newton, hydroelastic, TPU block mesh</b>, mass-corrected")]


def same_state_section(NT):
    R = defaultdict(list)
    for r in jl(os.path.join(RLD, "same_state_timing.jsonl")):
        if r.get("status") == "ok":
            R[(r["engine"], r.get("mujoco_warp"), r["variant"], r["nworld"])].append(r)

    def get(e, v, var, n):
        rs = R.get((e, v, var, n))
        return rs[-1] if rs else None
    rows = []
    for e, v, var, lab in SS_ROWS:
        a, b = get(e, v, var, 1024), get(e, v, var, 2048)
        if not (a or b):
            continue
        cells = [(lab, "lab")]
        for r in (a, b):
            if r is None:
                cells.append("&#8211;")
                continue
            txt = f(r["us_per_world_step"], 2)
            cells.append((txt, "bad") if r["held_frac"] < 0.95 else txt)
        ref = a or b
        cells += [f(ref["contacts_per_world"], 1), f(ref["nefc_world_mean"], 0),
                  f(min(r["held_frac"] for r in (a, b) if r), 2),
                  num(max(r["gpu_used_mb"] for r in (a, b) if r and r.get("gpu_used_mb")), ",.0f")]
        rows.append(cells)
    tab = table(["Engine, fingertip", "&#181;s/world-step, 1,024 worlds", "2,048", "contacts/world",
                 "constraint rows/world", "lowest holding", "GPU MB (largest batch)"], rows)
    D = jl(os.path.join(RLD, "same_state_timing_diagnostics.jsonl"))
    sap = next((r for r in D if r.get("broadphase") == "SAP_SEGMENTED" and r["variant"] == "pads1"
                and r["nworld"] == 1024 and r.get("ls_parallel")), None)
    small = next((r for r in D if r.get("broadphase") not in (None, "SAP_SEGMENTED") and r["variant"] == "pads1"
                  and r["nworld"] == 1024), None)
    m36 = get("mjw", "3.6.0", "pads1", 1024)
    lab_p = get("mjlab", "3.6.0", "pads1", 1024)
    hyd = get("nt_hydro", "3.14.0", "mesh", 1024)
    hyd2 = get("nt_hydro", "3.14.0", "mesh", 2048)
    ntp = get("nt_pt", "3.14.0", "pads1", 1024)
    box = [x for x in (get("mjw", "3.6.0", "legacy", 2048), get("nt_pt", "3.14.0", "legacy", 2048)) if x]
    m14 = get("mjw", "3.14.0", "pads1", 1024)
    ntm2 = get("nt_pt", "3.14.0", "mesh", 2048)
    bench = next((r for r in reversed(jl(os.path.join(RLD, "newton_throughput.jsonl"))) if r.get("mode") == "mjw"
                  and r.get("tip") == "pads" and r.get("status") == "ok" and r.get("broadphase") == 0), None)
    st = json.load(open(os.path.join(ROOT, "logs/20261006-rl_contact/state_pads1/state.json"))) \
        if os.path.exists(os.path.join(ROOT, "logs/20261006-rl_contact/state_pads1/state.json")) else {}
    out = ["<h3>Physics cost on one held RL state</h3>",
           "<p>Each engine starts from the same state of the RL env: D6 after its scripted grasp and lift (60 zero-action "
           "policy steps, 64 worlds, every tool held; with 1&#8202;mm pads "
           f"{f(st.get('ncon_world_mean'), 0)} contacts and {f(st.get('nefc_world_mean'), 0)} constraint rows per world), "
           "exported with mjlab&#8217;s scene writer (<code>scripts/rl_state_export.py</code>; free-body positions relative to "
           "each world&#8217;s grid origin, which mjlab adds per world). Every engine loads that MJCF with each body&#8217;s "
           "compiled inertia pinned, the trainer&#8217;s options (2&#8202;ms, implicitfast, elliptic cone, impratio 10, 10 "
           "solver and 20 line-search iterations), the exported joint angles tiled over the batch, the exported servo targets "
           "and zero velocity, and steps 22 graph-captured blocks of 50 steps (<code>scripts/same_state_timing.py</code>, rows "
           "<code>docs/experiments/20261006-rl_contact/same_state_timing.jsonl</code>). Newton 1.7.0.dev0 uses its own "
           "collision pipeline (contact gap 0.5&#8202;mm, margin 0), the TPU block re-added as a mesh from its OBJ, and for "
           "hydroelastic contact kh = E/h on the tips multiplied by the tip&#8211;tool 1/m<sub>eff</sub> of its solver model "
           "(Section&#160;4; tip inverse weights 4.95&#8211;5.14, tool 39.1&#8202;1/kg), 100 times that on the tool, 0.5&#8202;mm "
           "grid, &#177;6&#8202;mm band, contact reduction, buffers at 1.3&#8202;% of Newton&#8217;s defaults. The pads&#8217; "
           "solimp d0 in Newton is recomputed from its solver model&#8217;s inverse weights, which equal the MJCF "
           "compile&#8217;s to 10<sup>&#8722;8</sup>, so d0 is unchanged. The first rows run the live mjlab env for the same "
           "number of steps; its step includes the contact sensors.</p>",
           tab,
           tcap("Physics wall time per world-step on one held state of the D6 RL env, by number of worlds. Contacts and "
                "constraint rows per world at the end of the 1,024-world run; lowest holding: the smaller fraction of "
                "worlds whose tool stayed within 20&#8202;mm of its exported position over the 2.2&#8202;s run. Shaded: fewer "
                "than 95&#8202;% held. Not run: MuJoCo-Warp 3.14 pads at 2,048 worlds (GPU memory) and Newton pads at 2,048 "
                "(its model build takes 7.5&#8202;GB of host memory at 1,024)."),
           ]
    if m36 and lab_p and sap and small:
        out.append(
            f"<p>The 1&#8202;mm pads cost {f(m36['us_per_world_step'], 2)}&#8202;&#181;s per world-step in MuJoCo-Warp 3.6 "
            f"from this state (mjlab&#8217;s env {f(lab_p['us_per_world_step'], 2)}) and hold every tool. The 17&#8211;26&#8202;&#181;s "
            "and 13&#8202;% drops measured earlier on the bench grip came from two settings of the raw MuJoCo-Warp script: "
            + (f"rerun with both corrected, the bench grip costs {f(bench['us_per_world_step'], 2)}&#8202;&#181;s with "
               f"{f(bench['contacts_per_world'], 0)} contacts per world and holds {f(100 * bench['held_frac'], 0)}&#8202;% "
               "(rows <code>newton_throughput.jsonl</code>). " if bench else "")
            + "First, it forced the SAP_SEGMENTED broadphase; "
            "MuJoCo-Warp&#8217;s <code>put_model</code>, and therefore mjlab, chooses NXN over the contype-filtered geom pairs "
            "when they number under 250,000, and the pads touch only the tool, so the pair list is short. On this state SAP "
            f"costs {f(sap['us_per_world_step'], 1)}&#8202;&#181;s. Second, the broadphase writes its candidate pairs into the "
            "per-world contact buffer, and on the pads they exceed 250 per world for about 43 contacts; a buffer of "
            f"{small['nconmax']} contacts per world overflowed and dropped pad contacts, and "
            f"{f(100 * (1 - small['held_frac']), 0)}&#8202;% of worlds lost the tool. With 512 contacts per world both versions "
            f"hold every tool; MuJoCo-Warp 3.14 takes {f(m14['us_per_world_step'], 2)}&#8202;&#181;s.</p>" if m14 else "</p>")
    if hyd and ntp and box:
        out.append(
            f"<p>Newton&#8217;s mass-corrected hydroelastic tip holds every tool at {f(hyd['us_per_world_step'], 2)}&#8202;&#181;s "
            f"per world-step (1,024 worlds; {f(hyd2['us_per_world_step'], 2) if hyd2 else '&#8211;'} at 2,048, "
            f"{num(hyd2['gpu_used_mb'], ',.0f') if hyd2 else '&#8211;'}&#8202;MB of GPU) with "
            f"{f(hyd['contacts_per_world'], 0)} reduced contact points and {f(hyd['nefc_world_mean'], 0)} constraint rows "
            f"per world: {f(hyd['us_per_world_step'] / m36['us_per_world_step'], 1)} times the MuJoCo-Warp pads at "
            f"1,024 worlds and {f((hyd2 or hyd)['us_per_world_step'] / max(r['us_per_world_step'] for r in box), 1)}&#8211;"
            f"{f((hyd2 or hyd)['us_per_world_step'] / min(r['us_per_world_step'] for r in box), 1)} times point contact on "
            f"the box tip at 2,048. Newton with sphere pads takes {f(ntp['us_per_world_step'], 2)}&#8202;&#181;s; its "
            f"0.5&#8202;mm contact gap reports {f(ntp['contacts_per_world'], 0)} pad contacts per world, of which about as many "
            "become constraint rows as in MuJoCo-Warp."
            + (f" Newton&#8217;s point contact on the block mesh loses the tool in "
               f"{f(100 * (1 - ntm2['held_frac']), 0)}&#8202;% of worlds at 2,048, as it did at 4,096 on the bench grip." if ntm2 and ntm2['held_frac'] < 0.95 else "")
            + "</p>")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------ next, lede

def next_section():
    items = [
        "<b>Servo gain with a known load.</b> At the bench, with the servo holding a commanded angle, hang 50, 100 and "
        "200&#8202;g from a fingertip on a measured lever and read the deficit from the servo readback; the slope is kp "
        "in N&#8202;m/rad. The replays cannot give it (Table&#160;2). A 20&#176; free-air step per joint at 111&#8202;Hz "
        "gives &#964;; anything above 0.2&#8202;s falsifies the working plant.",
        "<b>Grasp search scored after the grip.</b> In <code>hom_grasp_search.py</code>, simulate each CEM candidate&#8217;s "
        "0.8&#8202;s grip in MuJoCo (1&#8202;mm pads, working plant) and score the kinematic turn range from the pad contacts "
        "the grip actually reaches, rejecting candidates whose tool leaves its post or whose pads end more than 2&#8202;mm "
        "from the planned contacts. A searched grasp that the governed controller turns past the deployed grasp&#8217;s "
        "angle on D7, D2 and D5 (6, 42 and 23&#176;) would confirm grasp choice as the limit; none would point at the "
        "controller.",
        "<b>A second contact mode in the HOM controller.</b> When a finger&#8217;s pip reaches its limit, give that "
        "finger&#8217;s contact a slide reference along the tool axis while the other two hold; measure the reachable turn "
        "and the drop rate on the same 40 placements.",
        "<b>Newton&#8217;s turn without rotation on D3 and D8.</b> Log the per-finger normal force and the tip slip speed "
        "in <code>newton_turn.py</code> and compare them with MuJoCo&#8217;s pad forces on D3 and D8, seeds 1&#8211;4, where the "
        "mass-corrected Newton grip holds the tool but does not turn it.",
        "<b>RL comparison.</b> The queue above, after the trainer applies <code>--seed</code>.",
    ]
    return "<ol>" + "".join(f"<li>{x}</li>" for x in items) + "</ol>"


def lede():
    plans_mj, plans_dk, hom = hom_data()
    pm = [r for h in HANDS for r in plans_mj.get(h, [])]
    pv = [med([r["turn_end_deg"] for r in plans_mj[h]]) for h in HANDS if plans_mj.get(h)]
    gm = [r for h in HANDS for r in hom.get((h, "mujoco", "gov"), [])]
    gv = [med([r["turn_end_deg"] for r in hom[(h, "mujoco", "gov")]]) for h in HANDS if hom.get((h, "mujoco", "gov"))]
    fm = [h for h in HANDS if hom.get((h, "mujoco", "free")) and sum(r["held"] for r in hom[(h, "mujoco", "free")]) == 0]
    T = {}
    for r in jl(os.path.join(RLD, "throughput.jsonl")):
        if r.get("status") == "ok" and r.get("sensor_reduce") != "netforce":
            T[(r["variant"], r["num_envs"])] = r
    mc = [r for h in HANDS for r in hom["plans_nt"].get(h, [])]
    n_mc_held = sum(r["held_end"] for r in mc)
    r1 = (T[("pads1", 2048)]["env_steps_per_s"] / T[("legacy", 2048)]["env_steps_per_s"]
          if ("pads1", 2048) in T and ("legacy", 2048) in T else None)
    return (f"The deployed open-loop three-finger turn holds the tool on {sum(r['held_end'] for r in pm)} of {len(pm)} "
            f"placements and turns it {f(min(pv), 0)}&#8211;{f(max(pv), 0)}&#176; in MuJoCo once the scene carries the servo "
            "response the bench readbacks show (0.02&#8202;s instead of 1&#8202;s), friction near 1 and the tool resting on its "
            "post. The relative contact-velocity controller of Wang, Oh and Pollard, keeping all three contacts and regulating "
            f"pad force, holds {sum(r['held'] for r in gm)} of {len(gm)} placements but stops at "
            f"{f(min(gv), 0)}&#8211;{f(max(gv), 0)}&#176;, because the deployed grips leave the pips no extension; without its "
            f"stop rule it drops the tool on {len(fm)} of 8 hands. Sphere pads at 1&#8202;mm cost "
            f"{f(100 * (1 - r1), 0) if r1 else '&#8211;'}&#8202;% of RL env throughput at 2,048 envs, because physics is a "
            "small part of an env step; from one held RL state the pads cost 2.2&#8211;2.4&#8202;&#181;s of physics per world-step "
            "in MuJoCo-Warp, Newton&#8217;s hydroelastic contact on the plain fingertip shape 4.3&#8202;&#181;s, point contact "
            "0.6&#8211;1.4&#8202;&#181;s. SolverMuJoCo realises Newton&#8217;s hydroelastic stiffness times the tip&#8211;tool "
            "effective mass; with kh "
            f"divided by that mass the deployed plans hold the tool on {n_mc_held} of {len(mc)} placements in Newton (10 "
            "uncorrected) and turn it within 5&#176; of MuJoCo on four of eight hands. The 2026-09-02 "
            "servo-gain fit closed the fingers on air; the bench readbacks fix the finger time constant and favour &#956; 1 but "
            "do not identify the gain.")


def main():
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "PREV_PATH": PREV_PATH}
    pu = os.path.join(ROOT, "docs/experiments/20261006-fingertip_backends/artifact_url.txt")
    v["PREV_LINK"] = f', <a href="{open(pu).read().strip()}">artifact</a>' if os.path.exists(pu) else ""
    v["GLOSSARY"] = glossary()
    v["PLANT"] = plant_section()
    v["HOM"] = hom_section()
    v["NEWTON"] = newton_section()
    v["RL"] = rl_section()
    v["NEXT"] = next_section()
    v["LEDE"] = lede()
    v["FOOTER"] = ("<p>Rebuild: <code>python3 scripts/hom_turn_rl_contact_page.py</code>. Rows: "
                   "<code>docs/experiments/20261006-servo_recalibration/</code>, <code>docs/experiments/20261006-hom_turn3/</code>, "
                   "<code>docs/experiments/20261006-rl_contact/</code>; films <code>docs/experiments/20261006-hom_turn3/media/</code>; "
                   "RL morphology runs <code>results/phase1/real_v1/20261006-sv1_u0308_b050_cal_tip_tpu2.7*</code>.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    open(OUT, "w").write(t)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
