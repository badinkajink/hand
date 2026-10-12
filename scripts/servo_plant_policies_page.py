#!/usr/bin/env python3
r"""Build docs/experiments/20261011-servo_plant_policies/20261011-servo_plant_contact_model_policies.html.

    logs/20261001-hom_contact/venv/bin/python scripts/servo_plant_policies_page.py

Steps 2-6 of docs/handoff/20261011-servo_plant_policies.md: the D6 contact-model comparison of 2026-10-08 (the TPU block
as one convex mesh with point contact, against 1 mm sphere pads) retrained on the servo-limited plant (kp 0.72 N m/rad,
torque limit 0.18 N m) with a penalty on fingertip force above 4 N, two seeds per arm, 40 M steps
(scripts/rl_contact_train_queue_20261011.sh). Rows from scripts/rl_contact_eval.py --study 20261011 (final, transfer, tb,
costs, and mechanism for these and the 2026-10-08 final policies) and scripts/rl_policy_replay.py --study 20261011, in
docs/experiments/20261011-servo_plant_policies/; the 2026-10-08 rows are read from that study's folder. Every number in
the prose is computed here from those rows and the saved per-step traces. Helpers, template and style are those of
scripts/contact_model_policies_page.py; figures with data are interactive (scripts/interactive_figure.py) over a static
SVG default view.

Colours: the dataviz reference slots of the 2026-10-08 page (TPU mesh blue, pads orange, skin aqua, box tip ink); the
plant is the line style, solid for the servo-limited plant and dashed for the working plant.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import interactive_figure as IF  # noqa: E402
import retro_style  # noqa: E402
import rl_contact_eval as E  # noqa: E402
import texsvg  # noqa: E402

D = os.path.join(P.EXP, "20261011-servo_plant_policies")
D8 = os.path.join(P.EXP, "20261008-contact_model_policies")
OUT = os.path.join(D, "20261011-servo_plant_contact_model_policies.html")
TPL = os.path.join(ROOT, "scripts/servo_plant_policies_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
NOTE = "docs/handoff/20261011-servo_plant_policies.md"
PAGE8 = "../20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html"
PAGE8_PATH = "docs/experiments/20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html"
fmt = P.fmt
ARMS = ("tpu27mesh", "tpu27pads1", "tpu27skin", "box")
LBL = {"box": "Box tip, point contact", "tpu27mesh": "TPU block, point contact", "tpu27pads1": "TPU block, 1&#8202;mm pads",
       "tpu27skin": "TPU block, pads on a skin"}
SHORT = {"box": "box tip", "tpu27mesh": "TPU mesh", "tpu27pads1": "pads", "tpu27skin": "skin"}
CAP = {"box": "Box tip", "tpu27mesh": "TPU mesh", "tpu27pads1": "Pads", "tpu27skin": "Skin"}
COL = {"box": "#48545D", "tpu27mesh": "#2a78d6", "tpu27pads1": "#eb6834", "tpu27skin": "#1baf7a"}
PLANT = {"11": "servo-limited plant", "08": "working plant"}
PLANT_CAP = {"11": "Servo-limited plant, 2026-10-11", "08": "Working plant, 2026-10-08"}
DASH = {"11": "", "08": "6 4"}
ENGINES = ("mujoco", "drake", "newton")
ENG_LBL = {"mujoco": "CPU MuJoCo", "drake": "Drake", "newton": "Newton"}
SCENES11 = ("tpu27mesh", "tpu27pads1", "box")
ONSET, TURN_END, T_STEP = 58, E.TURN_END, 0.02
R_ARM_MM, MU = 1.00, 1.0           # pad friction arm 1.00 mm (N/1 N)^(1/4) on the screwdriver (2026-10-01), mu of the scenes
FIG, TAB = [0], [0]
# figure and table numbers by name, in the order the page builds them (set in main(); the replay films are optional)
FIGNO, TABNO = {}, {}
EXTRA_CSS = """
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
figure.films{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
figure.films .cell,figure.replays .cell{min-width:0}
figure.replays .cell{margin-bottom:14px}
figure.films video,figure.films img,figure.replays video{width:100%;height:auto}
figure .cell p.filmlab{font-size:16px !important;line-height:1.35 !important;text-align:left !important;margin:4px 0 0 0}
figure .cell p.filmlab code{overflow-wrap:anywhere}
figure.films figcaption{grid-column:1/-1}
@media (max-width:640px){figure.films{grid-template-columns:minmax(0,1fr)}}
"""


# ------------------------------------------------------------------------------------------ rows

def rows(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("status", "ok") in ("ok", "complete"):
                out.append(r)
    return out


def last_by(rs, keys):
    """The newest row per key (rows are appended in time order)."""
    d = {}
    for r in rs:
        d[tuple(r.get(k) for k in keys)] = r
    return list(d.values())


def arm_of(tag):
    for a in ARMS:
        if f"_{a}_40M_" in tag:
            return a
    return None


class Data:
    def __init__(self):
        self.fin = {"11": last_by(rows(D, "final_eval.jsonl"), ("tag",)), "08": last_by(rows(D8, "final_eval.jsonl"), ("tag",))}
        self.mech = {(r["tag"], r.get("perturb") or "nominal"): r for r in last_by(rows(D, "mechanism.jsonl"), ("tag", "perturb"))}
        self.rep = {s: last_by([dict(r, dir=os.path.basename(r["dir"])) for r in rows(d, "policy_replay.jsonl")
                                if r.get("engine") and r.get("kind") != "record" and r.get("hold_test_s") is None],
                               ("dir", "engine", "world"))
                    for s, d in (("11", D), ("08", D8))}
        self.rec = {s: [r for r in rows(d, "policy_replay.jsonl") if r.get("kind") == "record"] for s, d in (("11", D), ("08", D8))}
        self.tr = {"11": last_by(rows(D, "transfer.jsonl"), ("tag", "scene")), "08": last_by(rows(D8, "transfer.jsonl"), ("tag", "scene"))}
        self.tb = {"11": rows(D, "tb_dynamics.jsonl"), "08": rows(D8, "tb_dynamics.jsonl")}
        self.cost = {"11": {r["tag"]: r for r in rows(D, "run_costs.jsonl")}, "08": {r["tag"]: r for r in rows(D8, "run_costs.jsonl")}}
        p = os.path.join(D, "films.json")
        self.films = json.load(open(p)) if os.path.exists(p) else []
        p = os.path.join(D, "plant_check.json")
        self.check = json.load(open(p)) if os.path.exists(p) else {}
        self._tr = {}

    def finals(self, study, arm=None):
        return sorted((r for r in self.fin[study] if arm is None or r["arm"] == arm),
                      key=lambda r: (ARMS.index(r["arm"]), r["seed"]))

    def policies(self, arms=ARMS):
        """(study, final row) of every final policy, servo-limited plant first."""
        return [(s, r) for s in ("11", "08") for a in arms for r in self.finals(s, a)]

    def m(self, tag, perturb="nominal"):
        return self.mech.get((tag, perturb))

    def trace(self, tag):
        """The mech record of a final policy (rl_contact_eval.load_trace): the 2026-10-11 final evaluation's traces,
        the mechanism evaluation's for the 2026-10-08 policies."""
        if tag not in self._tr:
            r = self.m(tag)
            p = os.path.join(D, r["traces"]) if r else None
            self._tr[tag] = E.load_trace(p) if p and os.path.exists(p) else None
        return self._tr[tag]

    def replays(self, study, tag, engine):
        return [r for r in self.rep[study] if r["dir"] == tag and r["engine"] == engine]


# ------------------------------------------------------------------------------------------ html helpers

def figure(svg, caption):
    FIG[0] += 1
    return f'<figure class="diagram">{svg}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def ifig(fig: IF.Figure, caption: str) -> str:
    FIG[0] += 1
    fig.caption = f"Figure&#160;{FIG[0]}. {caption}"
    return fig.html()


def table(head, body, caption=None, note=None, text_cols=(0,)):
    out = []
    if caption:
        TAB[0] += 1
        out.append(f"<p class='tcap'><b>Table&#160;{TAB[0]}.</b> {caption}</p>")
    cls = lambda i: "" if i in text_cols else " class=num"  # noqa: E731
    out.append("<div class='tw'><table><thead><tr>" + "".join(f"<th{cls(i)}>{h}</th>" for i, h in enumerate(head))
               + "</tr></thead><tbody>")
    for r in body:
        if isinstance(r, str):
            out.append(f"<tr class='grp'><td colspan='{len(head)}'>{r}</td></tr>")
        else:
            out.append("<tr>" + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table></div>")
    if note:
        out.append(f"<p class='tnote'>{note}</p>")
    return "".join(out)


def swatch(arm, study="11"):
    dash = "border-top:2px dashed" if study == "08" else "border-top:3px solid"
    return f'<span style="display:inline-block;width:18px;vertical-align:middle;{dash} {COL[arm]};margin-right:6px"></span>'


def f1(x, nd=1):
    return fmt(x, nd)


def rng(v, nd=2, unit=""):
    v = [x for x in v if x is not None]
    if not v:
        return "&#8211;"
    lo, hi = min(v), max(v)
    a, b = f"{lo:.{nd}f}", f"{hi:.{nd}f}"
    return (a if a == b else f"{a}&#8211;{b}") + unit


def sgn_rng(v, nd=2):
    """A range of signed values with explicit signs, e.g. +2.11 to +3.71."""
    v = [x for x in v if x is not None]
    if not v:
        return "&#8211;"
    f = lambda x: (f"+{x:.{nd}f}" if x >= 0 else f"&#8722;{-x:.{nd}f}")  # noqa: E731
    return f(min(v)) if f(min(v)) == f(max(v)) else f"{f(min(v))} to {f(max(v))}"


def tb_last(X: "Data") -> dict:
    """The last event-file row of every run with a final policy, keyed by tag, with its study."""
    last = {}
    for s in ("11", "08"):
        fins = {r["tag"] for r in X.finals(s)}
        for q in X.tb[s]:
            if q["tag"] in fins and (q["tag"] not in last or q["iteration"] > last[q["tag"]]["iteration"]):
                last[q["tag"]] = dict(q, study=s)
    return last


def video(rel):
    p = os.path.join(D, rel)
    if not os.path.exists(p):
        return ""
    jpg = p[:-4] + ".jpg"
    po = f' poster="{P.R.data_uri(jpg, "image/jpeg")}"' if os.path.exists(jpg) else ""
    return f'<video src="{P.R.data_uri(p, "video/mp4")}"{po} controls muted loop playsinline preload="metadata"></video>'


def plabel(study, r):
    return f"{CAP[r['arm']]} s{r['seed']}, {'servo' if study == '11' else 'working'} plant"


# ------------------------------------------------------------------------------------------ trace quantities

def held_mask(tr):
    f = tr["force"]
    return (tr["z"] > E.FLOOR_Z) & ((f >= E.HELD_N).sum(-1) >= 2)


def pinch_series(tr, c):
    """Per policy step: the tool's rotation about the pinch axis since the onset, the mean fingertip rotation and the
    contact spin (deg), each a (T, N) array, signed so that the tool's turn over steps ONSET-TURN_END is positive."""
    a0 = int(c["residual_from"])
    T = tr["cos"].shape[0]
    pt = E.pinch_turn(tr, a0, TURN_END)
    sgn = np.where(pt["tool"][a0:TURN_END].sum(0) < 0, -1.0, 1.0)
    on = (np.arange(T) >= a0)[:, None]
    tool = np.degrees(np.cumsum(np.where(on, pt["tool"], 0.0) * sgn, 0))
    tips = np.degrees(np.cumsum(np.where(on[..., None], pt["tips"], 0.0) * sgn[None, :, None], 0))
    return tool, tips.mean(-1), (tool[..., None] - tips).mean(-1)


def brake_mNm(forces):
    """Summed pad brake torque mu N rbar(N) of the three fingertips at forces N (N), in mN m."""
    f = np.maximum(np.asarray(forces, float), 0.0)
    return float(MU * R_ARM_MM * np.sum(f ** 1.25))


def grip_range(tr):
    """10th and 90th percentile of the summed fingertip force over the held steps of the hold (steps 150-249), N."""
    h = held_mask(tr)
    g = tr["force"].sum(-1)[150:][h[150:]]
    return (float(np.percentile(g, 10)), float(np.percentile(g, 90))) if g.size else None


def turn_time(tr):
    """Seconds after the onset at which the median cos of the rollouts held at 5 s first reaches 90 % of its final value."""
    held = held_mask(tr)[-1]
    if not held.any():
        return None
    m = np.median(tr["cos"][:, held], axis=1)
    k = int(np.argmax(m[ONSET:] >= 0.9 * m[-1]))
    return (k + 1) * T_STEP


# ------------------------------------------------------------------------------------------ glossary

GLOSSARY = [
    ("working plant", "The finger servo model of the 2026-10-06 and 10-08 runs: a position servo with gain \\(k_p\\) = "
     "4&#8202;N&#8202;m/rad, its output limited to \\(\\tau_\\text{max}\\) = 1&#8202;N&#8202;m, and joint damping \\(b\\) = "
     "0.08&#8202;N&#8202;m&#8202;s/rad."),
    ("servo-limited plant", "The finger servo model of this page: \\(k_p\\) = 0.72&#8202;N&#8202;m/rad, \\(\\tau_\\text{max}\\) "
     "= 0.18&#8202;N&#8202;m, \\(b\\) = 0.0144&#8202;N&#8202;m&#8202;s/rad. The SCS0009 servo stalls at 0.23&#8202;N&#8202;m at "
     "6&#8202;V and its overload protection trips above 80&#8202;% of that (0.18&#8202;N&#8202;m); its drive reaches full "
     "duty at 18&#176; of position error, so \\(k_p\\) = 0.23&#8202;N&#8202;m / 0.314&#8202;rad; \\(b/k_p\\) keeps the "
     "fitted 20&#8202;ms time constant."),
    ("contact model", "The fingertip&#8217;s collision geometry. The printed TPU block (17&#215;14.8&#215;22&#8202;mm, 2.7&#8202;mm "
     "fillets) is one convex mesh with MuJoCo point contact (<b>TPU mesh</b>; condim 3, no torsional friction), or 1,060 "
     "spheres of 1&#8202;mm per fingertip (<b>pads</b>), whose distributed contacts give a torsional friction. The "
     "2026-10-08 study also trained the legacy <b>box tip</b> (point contact) and the pads on a sprung mount "
     "(<b>skin</b>). In the tables a policy is its contact model and seed, e.g. Pads s0."),
    ("env step, iteration", "One 20&#8202;ms policy step of one of 2,048 parallel envs; a PPO iteration is 24 steps per "
     "env, 49,152 env steps, and 40&#8202;M env steps are 813 iterations."),
    ("held (load test)", "At least two fingertips press on the tool with 0.24&#8202;N (its weight) or more and the tool is "
     "above 60&#8202;mm; a rollout is held when this is true at 5&#8202;s."),
    ("cos", "Cosine of the tool axis with world vertical, \\(R_{zz}\\): +1 is tip down, the goal; the tool starts "
     "horizontal, at 0."),
    ("held cosine", "Mean final cos over the held rollouts of one evaluation."),
    ("evaluation, closed loop", "64 rollouts of the deterministic policy (the action is the policy mean) in parallel "
     "MuJoCo-Warp envs over 250 policy steps (5&#8202;s), with the training&#8217;s timing (policy from step 58, after "
     "the scripted grasp and lift), no early termination and no randomisation; 256 rollouts per transfer cell."),
    ("grip, grip per finger", "Net force of each fingertip on the tool (the env&#8217;s fingertip contact sensor), mean "
     "over the held steps from step 58, in N; the grip is the sum over the three fingertips."),
    ("grip term", "The reward term on fingertip force above 4&#8202;N: its weight times the mean over the three fingertips "
     "of \\(((N - 4\\,\\text{N})/4\\,\\text{N})^2\\). The working-plant runs used weight +0.25, a bonus for squeezing; the "
     "servo-plant runs use &#8722;2, a penalty."),
    ("pinch axis", "Unit vector from the thumb&#8217;s contact centroid on the tool to the midpoint of the index and middle "
     "fingers&#8217; centroids, per policy step; a finger without contact keeps its last centroid."),
    ("turn about the pinch axis", "Rotation of the tool about the pinch axis from the onset (step 58) to step 150: the sum "
     "over policy steps of the tool&#8217;s rotation increment projected on that step&#8217;s axis, in &#176;, median over "
     "the held rollouts."),
    ("fingertip rotation", "The same sum for a fingertip body, mean over the three fingertips."),
    ("contact spin", "Turn about the pinch axis minus a fingertip&#8217;s rotation about it: how far the tool rotates relative "
     "to the fingertip during the turn, in &#176;, mean over the three fingertips, median over the held rollouts. A grip "
     "that carries the tool rigidly has zero spin."),
    ("pad brake torque", "Torsional friction of a pad patch pressed with normal force \\(N\\) against the tool, \\(T = \\mu N "
     "\\bar r(N)\\), with the friction arm \\(\\bar r(N) = 1.00\\,\\text{mm}\\,(N/1\\,\\text{N})^{1/4}\\) measured on the "
     "real_v1 fingertip and the 24.5&#8202;g screwdriver on 2026-10-01 (Drake hydroelastic; the pads within 3&#8211;7&#8202;%), "
     "so \\(T \\propto N^{5/4}\\); \\(\\mu\\) = 1 in these scenes. Summed over the three fingertips at their grip, in "
     "mN&#8202;m. A point contact has none."),
    ("at the torque limit", "A finger actuator&#8217;s output within 2&#8202;% of \\(\\tau_\\text{max}\\); the share of held "
     "steps from step 58 in which this holds."),
    ("at a range limit", "A finger joint within 1&#176; of a bound of its range, or past it; the extension limit is the "
     "lower bound of the mcp and pip joints (&#8722;0.262 and &#8722;0.314&#8202;rad), the yaw range is &#177;1.484&#8202;rad."),
    ("torque-limit swap", "A final policy evaluated with the other plant&#8217;s torque limit and nothing else changed: "
     "1&#8202;N&#8202;m under a servo-plant policy, 0.18&#8202;N&#8202;m under a working-plant policy (mechanism evaluation, 64 "
     "rollouts)."),
    ("open-loop replay", "The recorded finger targets of a MuJoCo-Warp rollout from step 58, played at 50&#8202;Hz into "
     "another simulator with the palm welded at its lifted pose (<code>scripts/rl_policy_replay.py</code>), on the "
     "plant the policy trained on; Drake and Newton model the TPU block as a hydroelastic tip."),
    ("action noise", "Standard deviation of the policy&#8217;s Gaussian action distribution in training, in action units "
     "(one unit is 0.5&#8202;rad of servo target); 0.30 at the start."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


# ------------------------------------------------------------------------------------------ plant and cost

def hours(study, tag):
    c = X_.cost[study].get(tag, {})
    return c.get("wall_h")


def plant_section(X: Data):
    chk = X.check.get("checks", [])
    ok = [c for c in chk if all(c[k]["held_worlds"] == 64 for k in ("step60", "step150"))]
    head = ["", "Working plant<br>2026-10-08", "Servo-limited plant<br>2026-10-11"]
    body = [["Position gain \\(k_p\\) (N&#8202;m/rad)", "4", "0.72"],
            ["Torque limit \\(\\tau_\\text{max}\\) (N&#8202;m)", "1", "0.18"],
            ["Joint damping \\(b\\) (N&#8202;m&#8202;s/rad)", "0.08", "0.0144"],
            ["Grip term weight", "+0.25 (bonus)", "&#8722;2 (penalty)"],
            ["Contact models trained", "box tip, TPU mesh, pads, skin", "TPU mesh, pads"],
            ["Seeds with a final policy", f"{len(X.finals('08', 'tpu27mesh'))} TPU mesh, {len(X.finals('08', 'tpu27pads1'))} pads",
             f"{len(X.finals('11', 'tpu27mesh'))} TPU mesh, {len(X.finals('11', 'tpu27pads1'))} pads"]]
    out = ("<p>The four runs repeat the 2026-10-08 TPU-mesh and pad runs with two changes, the plant and the sign of the "
           "grip term (Table&#160;1). Everything else is as on 2026-10-08: hand D6, the 24.5&#8202;g screwdriver, "
           "MuJoCo-Warp at 2&#8202;ms with impratio 10, recipe <code>b_liveA</code>, 2,048 envs, 40&#8202;M steps, friction "
           "randomisation, seeds 0 and 1. Before training, the scripted grasp and lift held the tool in 64 of 64 worlds with "
           f"all three fingertips in contact at policy steps 60 and 150 on {len(ok)} of {len(chk)} servo-plant scenes (box tip, "
           "TPU mesh and pads; <code>plant_check.json</code>).</p>")
    out += table(head, body, "Plant and grip term of the two studies (scenes by <code>scripts/make_work_plant_runs.py "
                             "--plant</code>, runs by <code>scripts/rl_contact_train_queue_20261008.sh</code> and "
                             "<code>_20261011.sh</code>).")
    head2 = ["Run", "Iteration (s)", "Wall time (h)", "GPU memory,<br>peak (GB)", "Host memory,<br>peak (GB)", "Failed<br>starts"]
    b2 = []
    for r in X.finals("11"):
        c = X.cost["11"].get(r["tag"], {})
        b2.append([f"{swatch(r['arm'])}{LBL[r['arm']]}, seed {r['seed']}", f1(c.get("s_per_it_median"), 2),
                   f1(c.get("wall_h"), 2), f1((c.get("gpu_mem_peak_mb") or 0) / 1e3, 1), f1(c.get("host_rss_peak_gb"), 1),
                   str(c.get("failed_attempts", 0))])
    sp = {a: [X.cost["11"][r["tag"]]["s_per_it_median"] for r in X.finals("11", a) if r["tag"] in X.cost["11"]]
          for a in ("tpu27mesh", "tpu27pads1")}
    sp8 = {a: [c["s_per_it_median"] for t, c in X.cost["08"].items() if arm_of(t) == a and c.get("s_per_it_median")]
           for a in ("tpu27mesh", "tpu27pads1")}
    tot = sum(X.cost["11"][r["tag"]].get("wall_h") or 0 for r in X.finals("11") if r["tag"] in X.cost["11"])
    if all(sp.values()) and all(sp8.values()):
        out += (f"<p>The four runs took {tot:.1f}&#8202;GPU-h on the RTX 4070 Ti SUPER, one at a time (Table&#160;2). A pad "
                f"iteration cost {np.median(sp['tpu27pads1']) / np.median(sp['tpu27mesh']):.2f}&#215; a mesh iteration "
                f"({np.median(sp['tpu27pads1']):.2f} against {np.median(sp['tpu27mesh']):.2f}&#8202;s); in the 2026-10-08 runs, "
                f"which shared the workstation with other jobs, it cost {np.median(sp8['tpu27pads1']) / np.median(sp8['tpu27mesh']):.2f}&#215; "
                f"({np.median(sp8['tpu27pads1']):.2f} against {np.median(sp8['tpu27mesh']):.2f}&#8202;s).</p>")
    out += table(head2, b2, "Training cost of the servo-plant runs: median wall time per PPO iteration and in total "
                            "(trainer log), peak memory of the whole GPU (30&#8202;s samples) and of the host process "
                            "(<code>/usr/bin/time</code>).")
    return out


# ------------------------------------------------------------------------------------------ films

def ranked(X: Data, study, arm):
    """Final rows of an arm ranked best first (held fraction, then held cosine)."""
    return sorted(X.finals(study, arm), key=lambda r: (r["held_frac"], r["held_cos_mean"] or -2), reverse=True)


def films(X: Data):
    cells = []
    for f in X.films:
        r = next((r for r in X.finals("11") if r["tag"] == f["tag"]), None)
        rel = f"media/{f['tag']}_{f['role']}.mp4"
        if r is None or not os.path.exists(os.path.join(D, rel)):
            continue
        tr = X.trace(r["tag"])
        gr = grip_range(tr[0]) if tr else None
        tt = turn_time(tr[0]) if tr else None
        c70 = float(np.median(tr[0]["cos"][ONSET + 11][held_mask(tr[0])[-1]])) if tr else None
        cells.append(f'<div class="cell">{video(rel)}<p class="filmlab">{swatch(r["arm"])}<b>{LBL[r["arm"]]}</b>, seed '
                     f'{r["seed"]} ({f["role"]} seed): held {r["n_held"]}/64, held cos {r["held_cos_mean"]:.2f}; cos {c70:.2f} '
                     f'at 0.24&#8202;s after the onset, 90&#8202;% of the final cosine at {tt:.2f}&#8202;s; summed fingertip force '
                     f'in the hold {gr[0]:.0f}&#8211;{gr[1]:.0f}&#8202;N. <code>{rel}</code></p></div>')
    if not cells:
        return P.pending("Films not rendered yet (logs/20261011-servo_plant_policies/films.sh).")
    FIG[0] += 1
    return ("<p>All four final policies turn the tool the same way. From the onset the middle finger lifts the handle end "
            "and the tool pivots about the thumb and index contacts; within a quarter second it stands about 25&#8211;40&#176; "
            "from vertical. The middle finger ends horizontal at the limit of its yaw range, pressing on the upper end of the "
            "tool with its pip joint at the extension limit, and the thumb rotates against the turn until it lies along the "
            "shaft.</p>"
            f'<figure class="films">{"".join(cells)}<figcaption>Figure&#160;{FIG[0]}. The final policy of both seeds of each '
            "contact model on the servo-limited plant (best: more held rollouts, then the higher held cosine), one "
            "deterministic rollout rendered close up by <code>scripts/rl_render_reorient.py</code> at 960&#215;720: the scripted "
            "grasp and lift (0&#8211;1.16&#8202;s), then the policy&#8217;s turn and hold to 5&#8202;s; camera 0.24&#8202;m from "
            "the hand, the turn in the image plane (thumb in front, index left, middle right). Cosines are medians over the held "
            "rollouts of the final evaluation, force ranges the 10th&#8211;90th percentiles over the held steps of steps "
            "150&#8211;250. Paths relative to <code>docs/experiments/20261011-servo_plant_policies/</code>.</figcaption></figure>")


# ------------------------------------------------------------------------------------------ final policies

def replay_cell(X: Data, study, tag):
    out = []
    for e in ENGINES:
        rs = X.replays(study, tag, e)
        out.append(f"{sum(1 for r in rs if r.get('held_end'))}/{len(rs)}" if rs else "&#8211;")
    return " &#183; ".join(out)


def final_table(X: Data):
    head = ["Policy", "Held", "Held<br>cos", "Grip, thumb &#183; index &#183;<br>middle (N)",
            "Replays held, CPU MuJoCo<br>&#183; Drake &#183; Newton", "Contact<br>spin (&#176;)"]
    body = []
    for s in ("11", "08"):
        body.append(PLANT_CAP[s])
        for r in X.finals(s):
            m = X.m(r["tag"])
            body.append([f"{swatch(r['arm'], s)}{CAP[r['arm']]} s{r['seed']}", f"{r['n_held']}/{r['n']}",
                         f1(r.get("held_cos_mean"), 3),
                         " &#183; ".join(f1(r.get(k), 1) for k in ("grip_thumb_N", "grip_index_N", "grip_middle_N")),
                         replay_cell(X, s, r["tag"]), f1(m.get("spin_deg") if m else None, 0)])
    dh = [abs(X.m(r["tag"])["n_held"] - r["n_held"]) for r in X.finals("08") if X.m(r["tag"])]
    dc = [abs(X.m(r["tag"])["held_cos_mean"] - r["held_cos_mean"]) for r in X.finals("08") if X.m(r["tag"])]
    note = ("Held, held cosine and grip from each study&#8217;s final evaluation; contact spin from the mechanism "
            "evaluation, which for the 2026-10-08 policies is a second evaluation of the same deterministic policy (held count "
            f"within {max(dh) if dh else 0} rollout{'s' if dh and max(dh) != 1 else ''}, held cosine within "
            f"{max(dc) if dc else 0:.3f} of the first). Replays: worlds whose tool is held at the end over worlds replayed.")
    return table(head, body, "Final policies of both studies, 64 rollouts each.", note=note)


def ifig_traces(X: Data):
    idx = IF.sample_index(250, 200)
    t = (idx + 1) * T_STEP
    series, ids = [], {}
    for s, r in X.policies():
        tr = X.trace(r["tag"])
        if tr is None:
            continue
        tr = tr[0]
        h = held_mask(tr)[-1]
        if not h.any():
            continue
        cos = tr["cos"][:, h][idx]
        g = tr["force"].sum(-1)[:, h][idx]
        sid = f"{s}-{r['arm']}-s{r['seed']}"
        data = {"cos": {"med": np.median(cos, 1), "lo": np.percentile(cos, 10, 1), "hi": np.percentile(cos, 90, 1)},
                "grip": {"med": np.maximum(np.median(g, 1), 0.5), "lo": np.maximum(np.percentile(g, 10, 1), 0.5),
                         "hi": np.maximum(np.percentile(g, 90, 1), 0.5)}}
        series.append(IF.Series(sid, plabel(s, r), COL[r["arm"]], data, dash=DASH[s], width=1.8,
                                facets={"plant": s, "arm": r["arm"]}))
        ids.setdefault((s, r["arm"]), []).append(sid)
    legend = [(f"{CAP[a]}, {PLANT[s]}", COL[a], DASH[s], v) for (s, a), v in ids.items()]
    panels = [IF.Panel("cos", "Cosine of the tool axis with vertical, held rollouts", (-0.2, 1.0), ((0.9, "0.9"),), dec=3, height=170),
              IF.Panel("grip", "Summed fingertip force, held rollouts (N), log scale", (2.0, 200.0), dec=2, ylog=True, height=150)]
    facets = [{"key": "plant", "label": "Plant", "values": [("11", "Servo-limited"), ("08", "Working")]},
              {"key": "arm", "label": "Contact model", "values": [(a, CAP[a]) for a in ARMS], "on": ["tpu27mesh", "tpu27pads1"]}]
    fig = IF.Figure("final-traces", t, "Time (s)", panels, series, facets=facets, legend=legend,
                    vlines=[(ONSET * T_STEP, "Residual policy on (step 58)")], show_band=False, x_dec=2,
                    note="Medians over the held rollouts; hover for a policy&#8217;s 10&#8211;90&#8202;% band.")
    return ifig(fig, "Final policies of both studies: median over the held rollouts of the tool&#8217;s cosine with vertical "
                     "(top) and of the summed fingertip force (bottom, log scale) against time; solid the servo-limited "
                     "plant, dashed the working plant. The default view shows the TPU-mesh and pad policies; the boxes add the "
                     "working plant&#8217;s skin and box-tip policies. Drag to zoom, double-click to reset.")


def final_section(X: Data):
    if not X.finals("11"):
        return P.pending("Final evaluations not written yet (final_eval.jsonl).")
    f11, f08 = X.finals("11"), X.finals("08")
    hc = lambda s, a: [r["held_cos_mean"] for r in X.finals(s, a)]  # noqa: E731
    grip = lambda rs: [r["grip_N"] for r in rs]  # noqa: E731
    thumb = lambda rs: [r["grip_thumb_N"] / r["grip_N"] for r in rs]  # noqa: E731
    n11, nh11 = sum(r["n"] for r in f11), sum(r["n_held"] for r in f11)
    tps = [r for r in f08 if r["arm"] != "box"]
    box = next((r for r in f08 if r["arm"] == "box"), None)
    tt = [turn_time(X.trace(r["tag"])[0]) for r in f11 if X.trace(r["tag"])]
    txt = (f"<p>On the servo-limited plant every final policy holds the tool, {nh11} of {n11} rollouts, as the TPU-block "
           f"policies of the working plant did ({sum(r['n_held'] for r in tps)} of {sum(r['n'] for r in tps)}; the box tip "
           f"{box['n_held'] if box else '&#8211;'} of 64). The held cosine of the TPU-mesh policies is "
           f"{rng(hc('11', 'tpu27mesh'), 3)} (working plant {rng(hc('08', 'tpu27mesh'), 3)}), that of the pad policies "
           f"{rng(hc('11', 'tpu27pads1'), 3)} (working plant {rng(hc('08', 'tpu27pads1'), 3)}). The summed fingertip force falls "
           f"from {rng(grip(tps), 0)}&#8202;N to {rng(grip(f11), 0)}&#8202;N, and the thumb carries "
           f"{rng([100 * v for v in thumb(f11)], 0)}&#8202;% of it ({rng([100 * v for v in thumb(tps)], 0)}&#8202;% on the working "
           f"plant). Every servo-plant policy brings the tool to 90&#8202;% of its final cosine within "
           f"{max(tt):.2f}&#8202;s of the onset (Figure&#160;{FIGNO['traces']}).</p>")
    out = txt + final_table(X)
    out += ifig_traces(X)
    return out


# ------------------------------------------------------------------------------------------ mechanism

def ifig_mech(X: Data):
    idx = IF.sample_index(250, 200)
    t = (idx + 1) * T_STEP
    series, ids = [], {}
    for s, r in X.policies():
        tr = X.trace(r["tag"])
        if tr is None:
            continue
        tr, c = tr
        h = held_mask(tr)[-1]
        if not h.any():
            continue
        tool, tips, spin = pinch_series(tr, c)
        th = tr["force"][:, :, 0]
        sid = f"m{s}-{r['arm']}-s{r['seed']}"
        data = {"turn": {"med": np.median(tool[:, h], 1)[idx]}, "spin": {"med": np.median(spin[:, h], 1)[idx]},
                "thumb": {"med": np.maximum(np.median(th[:, h], 1)[idx], 0.2)}}
        series.append(IF.Series(sid, plabel(s, r), COL[r["arm"]], data, dash=DASH[s], width=1.8,
                                facets={"plant": s, "arm": r["arm"]}))
        ids.setdefault((s, r["arm"]), []).append(sid)
    legend = [(f"{CAP[a]}, {PLANT[s]}", COL[a], DASH[s], v) for (s, a), v in ids.items()]
    panels = [IF.Panel("turn", "Turn of the tool about the pinch axis since the onset (&#176;)", (-20.0, 160.0), band=False, dec=1, height=150),
              IF.Panel("spin", "Contact spin, mean over the three fingertips (&#176;)", (-20.0, 160.0), band=False, dec=1, height=150),
              IF.Panel("thumb", "Thumb fingertip force (N), log scale", (0.2, 100.0), band=False, dec=2, ylog=True, height=130)]
    facets = [{"key": "plant", "label": "Plant", "values": [("11", "Servo-limited"), ("08", "Working")]},
              {"key": "arm", "label": "Contact model", "values": [(a, CAP[a]) for a in ARMS], "on": ["tpu27mesh", "tpu27pads1"]}]
    fig = IF.Figure("mech-traces", t, "Time (s)", panels, series, facets=facets, legend=legend,
                    vlines=[(ONSET * T_STEP, "Residual policy on (step 58)"), (TURN_END * T_STEP, "End of the turn window (step 150)")],
                    show_band=False, x_dec=2)
    spread = {r["tag"]: X.m(r["tag"])["axis_spread_deg"] for _, r in X.policies() if X.m(r["tag"])}
    box = [v for t, v in spread.items() if arm_of(t) == "box"]
    tpu = [v for t, v in spread.items() if arm_of(t) != "box"]
    note = (f" The box-tip policy shakes the tool and its pinch axis moves by a median {box[0]:.0f}&#176; from the "
            f"rollout&#8217;s mean axis during the turn ({rng(tpu, 0)}&#176; for the TPU-block policies), so its curves are "
            "noisy." if box and tpu else "")
    return ifig(fig, "Mechanism of the turn, medians over the held rollouts of each final policy: the tool&#8217;s turn about "
                     "the pinch axis since the onset, the contact spin (the turn minus the fingertips&#8217; own rotation "
                     "about the axis), and the thumb&#8217;s force; solid the servo-limited plant, dashed the working plant." + note)


def ifig_brake(X: Data):
    """Contact spin and held cosine against the summed fingertip force, nominal and with the torque limit swapped; a thin
    line joins each policy's two evaluations."""
    groups, lines, ids = {}, [], {}
    for s, r in X.policies(("tpu27mesh", "tpu27pads1", "tpu27skin")):
        m0 = X.m(r["tag"])
        sw = X.m(r["tag"], "fr=1" if s == "11" else "fr=0.18")
        pts = [(m0, "nominal")] + ([(sw, "swap")] if sw else [])
        xy = []
        for m, cond in pts:
            if m is None or not m.get("n_held"):
                continue
            g = float(sum(m["grip_by_finger_N"]))
            k = (s, r["arm"], cond)
            d = groups.setdefault(k, {"x": [], "spin": [], "cos": [], "lab": []})
            d["x"].append(g)
            d["spin"].append(m["spin_deg"])
            d["cos"].append(m["held_cos_mean"])
            lim = "1 N m" if (s == "11") == (cond == "swap") else "0.18 N m"
            d["lab"].append(f"{plabel(s, r)}, torque limit {lim}: held {m['n_held']}/{m['n']}")
            xy.append((g, m["spin_deg"], m["held_cos_mean"]))
        if len(xy) == 2:
            lines.append((s, r, xy))
    series = []
    for (s, a, cond), d in groups.items():
        sid = f"b{s}-{a}-{cond}"
        series.append(IF.Series(sid, f"{CAP[a]}, {PLANT[s]}{', torque limit swapped' if cond == 'swap' else ''}", COL[a],
                                {"spin": {"x": d["x"], "med": d["spin"]}, "cos": {"x": d["x"], "med": d["cos"]}},
                                mode="markers", open=(cond == "swap"), facets={"plant": s, "arm": a},
                                point_labels={"spin": d["lab"], "cos": d["lab"]}))
        ids.setdefault((s, a, cond), []).append(sid)
    for s, r, xy in lines:
        series.append(IF.Series(f"l{s}-{r['arm']}-s{r['seed']}", "", COL[r["arm"]],
                                {"spin": {"x": [p[0] for p in xy], "med": [p[1] for p in xy]},
                                 "cos": {"x": [p[0] for p in xy], "med": [p[2] for p in xy]}},
                                dash=DASH[s], width=1.0, facets={"plant": s, "arm": r["arm"]}))
    legend = [(f"{CAP[a]}, {PLANT[s]}{', limit swapped' if c == 'swap' else ''}", COL[a], "", v) for (s, a, c), v in ids.items()]
    panels = [IF.Panel("spin", "Contact spin during the turn (&#176;)", (0.0, 160.0), band=False, dec=1, height=200),
              IF.Panel("cos", "Held cosine", (0.0, 1.0), band=False, dec=3, height=170)]
    facets = [{"key": "plant", "label": "Policy trained on", "values": [("11", "Servo-limited plant"), ("08", "Working plant")]},
              {"key": "arm", "label": "Contact model", "values": [(a, CAP[a]) for a in ("tpu27mesh", "tpu27pads1", "tpu27skin")]}]
    fig = IF.Figure("brake", None, "Summed fingertip force (N), log scale", panels, series, facets=facets, legend=legend,
                    x_domain=(5.0, 120.0), xlog=True)
    return ifig(fig, "The pad-brake test. Contact spin and held cosine of each final policy against its summed fingertip "
                     "force, at the torque limit it trained with (filled) and with the other plant&#8217;s limit (hollow, "
                     "joined to its policy&#8217;s filled marker); the pad brake torque grows as the 5/4 power of the force "
                     "at each fingertip. Hover a marker for the policy and its held count.")


def mech_table(X: Data):
    head = ["Policy", "Grip<br>(N)", "Pad brake<br>(mN&#8202;m)", "Turn<br>(&#176;)", "Fingertip<br>rotation (&#176;)",
            "Contact<br>spin (&#176;)", "Thumb mcp at<br>torque limit (%)", "Extension<br>limit (%)"]
    body = []
    for s in ("11", "08"):
        body.append(PLANT_CAP[s])
        for r in X.finals(s):
            m = X.m(r["tag"])
            if not m:
                continue
            body.append([f"{swatch(r['arm'], s)}{CAP[r['arm']]} s{r['seed']}", f1(sum(m["grip_by_finger_N"]), 1),
                         f1(brake_mNm(m["grip_by_finger_N"]), 0) if r["arm"] in ("tpu27pads1", "tpu27skin") else "&#8211;",
                         f1(m["turn_pinch_deg"], 0), f1(m["tip_turn_deg"], 0), f1(m["spin_deg"], 0),
                         f1(100 * m["torque_limit_by_joint"]["thumb_mcp"], 0), f1(100 * m["ext_limit_frac"], 0)])
    return table(head, body, "Mechanism of the turn in each final policy (mechanism evaluation, 64 rollouts; medians over "
                             "the held rollouts, shares over their held steps from step 58): grip, pad brake torque summed "
                             "over the three fingertips at their mean force (the point-contact tips have none), turn about "
                             "the pinch axis, fingertip rotation and contact spin over steps 58&#8211;150, and the shares "
                             "of held steps with the thumb mcp at its torque limit and with a finger joint at its extension "
                             "limit.")


def swap_table(X: Data):
    head = ["Policy", "Torque limit<br>(N&#8202;m)", "Held", "Held cos", "Grip (N)", "Contact spin (&#176;)"]
    body = []
    for s in ("11", "08"):
        body.append(f"{PLANT_CAP[s]} policies")
        pk = "fr=1" if s == "11" else "fr=0.18"
        for r in X.finals(s):
            if r["arm"] == "box":
                continue
            m0, m1 = X.m(r["tag"]), X.m(r["tag"], pk)
            if not m0 or not m1:
                continue
            lim0, lim1 = ("0.18", "1") if s == "11" else ("1", "0.18")
            fmt2 = lambda a, b, nd=3: f"{f1(a, nd)} &#8594; {f1(b, nd)}"  # noqa: E731
            body.append([f"{swatch(r['arm'], s)}{CAP[r['arm']]} s{r['seed']}", f"{lim0} &#8594; {lim1}",
                         f"{m0['n_held']} &#8594; {m1['n_held']}", fmt2(m0["held_cos_mean"], m1["held_cos_mean"]),
                         fmt2(sum(m0["grip_by_finger_N"]), sum(m1["grip_by_finger_N"]) if m1["n_held"] else None, 1),
                         fmt2(m0["spin_deg"], m1["spin_deg"], 0)])
    return table(head, body, "Torque-limit swap: each final policy at its own torque limit and at the other plant&#8217;s, "
                             "nothing else changed (64 rollouts each).")


def mech_text(X: Data):
    f11, f08 = X.finals("11"), X.finals("08")
    sp = lambda s, a: [X.m(r["tag"])["spin_deg"] for r in X.finals(s, a) if X.m(r["tag"])]  # noqa: E731
    tn = lambda s, a: [X.m(r["tag"])["turn_pinch_deg"] for r in X.finals(s, a) if X.m(r["tag"])]  # noqa: E731
    br = lambda s, a: [brake_mNm(X.m(r["tag"])["grip_by_finger_N"]) for r in X.finals(s, a) if X.m(r["tag"])]  # noqa: E731
    tl = [X.m(r["tag"])["torque_limit_by_joint"]["thumb_mcp"] for r in f11 + f08 if X.m(r["tag"]) and r["arm"] != "box"]
    tbl = tb_last(X)
    tbl = {t: q for t, q in tbl.items() if arm_of(t) != "box"}
    gt = {s: [q["Episode_Reward/grip_force_excess"] for q in tbl.values() if q["study"] == s] for s in ("11", "08")}
    al = [q["Episode_Reward/target_axis_alignment"] for q in tbl.values()]
    ext11 = [X.m(r["tag"])["ext_limit_frac"] for r in f11 if X.m(r["tag"])]
    yaw11 = [X.m(r["tag"])["range_limit_by_joint"]["middle_yaw"] for r in f11 if X.m(r["tag"])]
    p1 = (f"<p>Contact spin separates the pads of the two plants. On the working plant the pad policies spun the tool "
          f"{rng(sp('08', 'tpu27pads1'), 0)}&#176; relative to their fingertips during the turn, at a summed pad brake of "
          f"{rng(br('08', 'tpu27pads1'), 0)}&#8202;mN&#8202;m; on the servo-limited plant they spin it "
          f"{rng(sp('11', 'tpu27pads1'), 0)}&#176; at {rng(br('11', 'tpu27pads1'), 0)}&#8202;mN&#8202;m, as far as the "
          f"point-contact TPU mesh ({rng(sp('11', 'tpu27mesh'), 0)}&#176;; {rng(sp('08', 'tpu27mesh'), 0)}&#176; on the working "
          f"plant). The turn about the pinch axis follows: pads {rng(tn('08', 'tpu27pads1'), 0)}&#176; on the working plant and "
          f"{rng(tn('11', 'tpu27pads1'), 0)}&#176; on the servo-limited plant, TPU mesh {rng(tn('08', 'tpu27mesh'), 0)}&#176; "
          f"and {rng(tn('11', 'tpu27mesh'), 0)}&#176;. The skin policies of the working plant spun the tool "
          f"{rng(sp('08', 'tpu27skin'), 0)}&#176; at a pad brake of {rng(br('08', 'tpu27skin'), 0)}&#8202;mN&#8202;m, as far as "
          f"the working plant&#8217;s TPU-mesh policies.</p>")
    p2 = (f"<p>The thumb&#8217;s mcp actuator sits at its torque limit in {rng([100 * v for v in tl], 0)}&#8202;% of the held "
          f"steps of every TPU-block policy on both plants, so the actuator limit sets the grip; the grip term was "
          f"{sgn_rng(gt['08'])} per episode at the end of the working-plant runs and is {sgn_rng(gt['11'])} in the "
          f"servo-plant runs, against an alignment term of {rng(al, 0)} (Figure&#160;{FIGNO['curves']}). On the servo-limited "
          f"plant a finger joint sits at its extension limit in {rng([100 * v for v in ext11], 0)}&#8202;% of the held steps, "
          f"and the middle finger&#8217;s yaw at the end of its range in {rng([100 * v for v in yaw11], 0)}&#8202;% "
          f"(Table&#160;{TABNO['mech']}).</p>")
    return p1 + p2


def middle_yaw(X: Data, tag, perturb="nominal"):
    """Median middle-finger yaw (rad) over the held steps of the hold (steps 150-249) of an evaluation's held rollouts."""
    m = X.m(tag, perturb)
    p = os.path.join(D, m["traces"]) if m else None
    if not p or not os.path.exists(p):
        return None
    tr, c = E.load_trace(p)
    h = held_mask(tr)
    k = list(c["joints"]).index("middle_yaw")
    q = tr["q9"][150:, :, k][h[150:]]
    return float(np.median(q)) if q.size else None


def swap_text(X: Data):
    pk = {"11": "fr=1", "08": "fr=0.18"}
    pairs = {(s, a): [(r, X.m(r["tag"]), X.m(r["tag"], pk[s])) for r in X.finals(s, a)
                      if X.m(r["tag"]) and X.m(r["tag"], pk[s])]
             for s in ("11", "08") for a in ARMS}
    if not any(pairs.values()):
        return P.pending("Torque-limit swap not evaluated yet (rl_contact_eval.py mechanism --perturb fr=...).")
    gr = lambda m: sum(m["grip_by_finger_N"])  # noqa: E731
    mesh, pads = pairs[("11", "tpu27mesh")], pairs[("11", "tpu27pads1")]
    ratio = [gr(m1) / gr(m0) for r, m0, m1 in mesh + pads if m1["n_held"]]
    out = [f"<p>Given 1&#8202;N&#8202;m, the servo-plant policies grip {rng(ratio, 1)}&#215; harder "
           f"(Table&#160;{TABNO['swap']}, Figure&#160;{FIGNO['brake']}). The TPU-mesh policies hold "
           + " and ".join(str(m1["n_held"]) for _, _, m1 in mesh) + " of 64 rollouts and turn the tool as before (held cosine "
           + " and ".join(f"{m0['held_cos_mean']:.3f} &#8594; {m1['held_cos_mean']:.3f}" for _, m0, m1 in mesh) + ")."]
    for r, m0, m1 in pads:
        if m1["n_held"] == m1["n"] and m1["held_cos_mean"] < 0.5:
            y0, y1 = middle_yaw(X, r["tag"]), middle_yaw(X, r["tag"], pk["11"])
            out.append(f" Pad seed {r['seed']} holds all {m1['n']} but no longer turns the tool: its held cosine falls from "
                       f"{m0['held_cos_mean']:.3f} to {m1['held_cos_mean']:.3f} and its contact spin from {m0['spin_deg']:.0f} to "
                       f"{m1['spin_deg']:.0f}&#176;"
                       + (f", and the middle finger&#8217;s yaw stops at {y1:.2f}&#8202;rad instead of {y0:.2f}" if y0 and y1 else "")
                       + ".")
        else:
            out.append(f" Pad seed {r['seed']} drops {m1['n'] - m1['n_held']} of {m1['n']} rollouts and turns the held ones "
                       f"as before (held cosine {m0['held_cos_mean']:.3f} &#8594; {m1['held_cos_mean']:.3f}).")
    w = [(a, r, m1) for a in ("tpu27pads1", "tpu27mesh", "tpu27skin", "box") for r, m0, m1 in pairs[("08", a)]]
    drop = [f"{CAP[a]} s{r['seed']}" for a, r, m1 in w if m1["n_held"] == 0]
    kept = [(a, r, m1) for a, r, m1 in w if m1["n_held"] > 0]
    if w:
        held_txt = "; " + ", ".join(f"{CAP[a]} s{r['seed']} holds {m1['n_held']}" for a, r, m1 in kept) + (
            f" of 64, at held cosines of {rng([m1['held_cos_mean'] for _, _, m1 in kept], 2)}") if kept else ""
        who = ", ".join(drop[:-1]) + " and " + drop[-1] if len(drop) > 1 else "".join(drop)
        out.append(f" Given 0.18&#8202;N&#8202;m, the working-plant policies {who} drop every rollout, so the swap tests the "
                   f"brake in one direction only{held_txt}.")
    return "".join(out) + "</p>"


def mech_section(X: Data):
    if not X.mech:
        return P.pending("Mechanism rows not written yet (mechanism.jsonl).")
    out = mech_text(X)
    out += mech_table(X)
    out += ifig_mech(X)
    out += ("<p>The torque-limit swap changes the grip with the policy held fixed. The pad brake predicts that a harder "
            "grip stops the turn of a pad policy and leaves that of a TPU-mesh policy, which has no brake.</p>")
    out += swap_text(X)
    out += swap_table(X)
    out += ifig_brake(X)
    return out


# ------------------------------------------------------------------------------------------ transfer

def transfer_table(X: Data):
    head = ["Policy", "TPU mesh,<br>closed loop", "Pads,<br>closed loop", "Box tip,<br>closed loop", "CPU MuJoCo,<br>replay",
            "Drake,<br>replay", "Newton,<br>replay"]
    body = []
    for r in X.finals("11"):
        cells = []
        for sc in SCENES11:
            rr = [t for t in X.tr["11"] if t["tag"] == r["tag"] and t["scene"] == sc]
            if rr:
                t = rr[0]
                own = "<b>" if sc == r["arm"] else ""
                cells.append(f"{own}{t['n_held']} &#183; {f1(t['held_cos_mean'], 2)}{'</b>' if own else ''}")
            else:
                cells.append("&#8211;")
        for e in ENGINES:
            rs = X.replays("11", r["tag"], e)
            hc = [x["cos_end"] for x in rs if x.get("held_end")]
            cells.append(f"{sum(1 for x in rs if x.get('held_end'))}/{len(rs)} &#183; {f1(float(np.mean(hc)), 2) if hc else '&#8211;'}"
                         if rs else "&#8211;")
        body.append([f"{swatch(r['arm'])}{CAP[r['arm']]} s{r['seed']}"] + cells)
    return table(head, body, "Transfer of the servo-plant final policies. Closed loop in MuJoCo-Warp on each servo-plant "
                             "scene: rollouts held of 256 &#183; held cosine (bold: the policy&#8217;s own contact model). "
                             "Open-loop replays: worlds held of worlds replayed &#183; mean final cosine of the held ones.")


def transfer_text(X: Data):
    tot = lambda s, arms: replay_totals(X, s, arms)  # noqa: E731
    pads11, mesh11 = tot("11", ("tpu27pads1",)), tot("11", ("tpu27mesh",))
    comp08, point08 = tot("08", ("tpu27pads1", "tpu27skin")), tot("08", ("tpu27mesh", "box"))
    gap = []
    for r in X.finals("11"):
        for e in ENGINES:
            for x in X.replays("11", r["tag"], e):
                rec = next((q for q in X.rec["11"] if q["tag"] == r["tag"]), None)
                if rec and x.get("held_end") and x.get("world") is not None and x["world"] < len(rec["mjw_cos_end"]):
                    gap.append((r["arm"], e, x["cos_end"] - rec["mjw_cos_end"][x["world"]]))
    g = {(a, e): [v for aa, ee, v in gap if aa == a and ee == e] for a in ("tpu27mesh", "tpu27pads1") for e in ENGINES}
    cross = [t for t in X.tr["11"] if t["scene"] in ("tpu27mesh", "tpu27pads1") and t["scene"] != t["arm"]]
    box = [t for t in X.tr["11"] if t["scene"] == "box"]
    out = (f"<p>Replayed open loop on the servo-limited plant, the finger targets of the pad policies kept the tool in "
           f"{pads11[0]:,} of {pads11[1]:,} replays and those of the TPU-mesh policies in {mesh11[0]:,} of {mesh11[1]:,}; on the "
           f"working plant the pad and skin targets kept it in {comp08[0]:,} of {comp08[1]:,} and the point-contact targets in "
           f"{point08[0]:,} of {point08[1]:,} (<a href=\"{PAGE8}\">2026-10-08 page</a>). ")
    if cross:
        out += (f"Closed loop on the other TPU-block contact model, the four policies held "
                f"{sum(t['n_held'] for t in cross):,} of {sum(t['n'] for t in cross):,} rollouts")
        out += (f", and on the box tip {sum(t['n_held'] for t in box):,} of {sum(t['n'] for t in box):,}. " if box else ". ")
    falls, flips = [], 0
    for r in X.finals("11"):
        for x in X.replays("11", r["tag"], "newton"):
            if not x.get("held_end"):
                t = next((tt for tt, c, z in (x.get("trace") or []) if z < E.FLOOR_Z), None)
                if t is not None:
                    falls.append(t)
            elif x.get("cos_end", 1.0) < 0:
                flips += 1
    nt = {a: replay_totals_engine(X, "11", a, "newton") for a in ("tpu27pads1", "tpu27mesh")}
    nt8 = {a: replay_totals_engine(X, "08", a, "newton") for a in ("tpu27pads1",)}
    med = lambda v: (f"{np.median(v):+.3f}".replace("-", "&#8722;") if v else "&#8211;")  # noqa: E731
    out += ("The final cosine of a held replay minus that of the MuJoCo-Warp rollout it replays is a median "
            + ", ".join(f"{med(g[('tpu27pads1', e)])} (pads) and {med(g[('tpu27mesh', e)])} (TPU mesh) in {ENG_LBL[e]}"
                        for e in ENGINES) + ".</p>")
    if falls:
        early = sum(1 for t in falls if t <= 1.0)
        out += (f"<p>Newton loses the most replays: the pad targets keep the tool in {nt['tpu27pads1'][0]} of "
                f"{nt['tpu27pads1'][1]} worlds there ({nt8['tpu27pads1'][0]} of {nt8['tpu27pads1'][1]} at the working "
                f"plant&#8217;s grip) and the TPU-mesh targets in {nt['tpu27mesh'][0]} of {nt['tpu27mesh'][1]}. The tool falls "
                f"{min(falls):.1f}&#8211;{max(falls):.1f}&#8202;s after the onset, in {early} of {len(falls)} worlds within the "
                "first second, after the turn has ended"
                + (f"; one held pad replay ends turned to the wrong pole (cos &#8722;0.80)" if flips == 1 else
                   (f"; {flips} held replays end turned to the wrong pole" if flips else "")) + ".</p>")
    return out


def replay_totals_engine(X: Data, study, arm, engine):
    h = n = 0
    for r in X.finals(study, arm):
        rs = X.replays(study, r["tag"], engine)
        h += sum(1 for x in rs if x.get("held_end"))
        n += len(rs)
    return h, n


def replay_outcome(X: Data, tag, world=0):
    """What happens to the tool in world `world` of a policy's replays: MuJoCo-Warp's final cosine, then Drake's and
    Newton's (held at the end with its cosine, or the time after the onset at which the tool falls below 60 mm)."""
    rec = next((q for q in X.rec["11"] if q["tag"] == tag), None)
    parts = [f"MuJoCo-Warp ends at cos {rec['mjw_cos_end'][world]:.2f}"] if rec else []
    for e in ("drake", "newton"):
        x = next((q for q in X.replays("11", tag, e) if q.get("world") == world), None)
        if x is None:
            continue
        if x.get("held_end"):
            parts.append(f"{ENG_LBL[e]} holds it at cos {x['cos_end']:.2f}")
        else:
            t = next((tt for tt, c, z in (x.get("trace") or []) if z < E.FLOOR_Z), None)
            parts.append(f"in {ENG_LBL[e]} it falls {t:.1f}&#8202;s after the onset" if t is not None else f"{ENG_LBL[e]} drops it")
    return ("; ".join(parts) + ".") if parts else ""


def transfer_section(X: Data):
    if not X.rep["11"] and not X.tr["11"]:
        return P.pending("Replays and transfer cells not written yet (policy_replay.jsonl, transfer.jsonl).")
    out = transfer_text(X) + transfer_table(X)
    rep = []
    for f in X.films:
        if f["role"] != "median":
            continue
        rel = f"media/{f['tag']}_replay_three.mp4"
        if os.path.exists(os.path.join(D, rel)):
            rep.append(f'<div class="cell">{video(rel)}<p class="filmlab">{swatch(f["arm"])}<b>{LBL[f["arm"]]}</b>, '
                       f'seed {f["tag"][-1]}, world 0. {replay_outcome(X, f["tag"])} <code>{rel}</code></p></div>')
    if rep:
        FIG[0] += 1
        out += (f'<figure class="replays">{"".join(rep)}<figcaption>Figure&#160;{FIG[0]}. The median seed of each contact '
                "model, world 0: the MuJoCo-Warp rollout it trained on (left), and its finger targets from the onset of the "
                "turn replayed open loop in Drake (centre) and Newton (right) with the TPU block as a hydroelastic tip and the "
                "palm welded at its lifted pose, on the servo-limited plant. Paths relative to "
                "<code>docs/experiments/20261011-servo_plant_policies/</code>.</figcaption></figure>")
    return out


# ------------------------------------------------------------------------------------------ training curves

TB = [("Train/mean_reward", "Mean episode return", None),
      ("Episode_Reward/target_axis_alignment", "Alignment term of the return", None),
      ("Episode_Reward/grip_force_excess", "Grip term of the return", None),
      ("Policy/mean_std", "Action noise (action units)", (0.0, 0.4))]


def _smooth(y, win=9):
    y = np.asarray(y, float)
    k = np.ones(win) / win
    pad = np.r_[np.full(win // 2, y[0]), y, np.full(win // 2, y[-1])]
    return np.convolve(pad, k, mode="valid")


def ifig_curves(X: Data):
    series, ids = [], {}
    for s in ("11", "08"):
        for r in X.finals(s):
            if r["arm"] not in ("tpu27mesh", "tpu27pads1"):
                continue
            rs = sorted((q for q in X.tb[s] if q["tag"] == r["tag"]), key=lambda q: q["iteration"])
            if not rs:
                continue
            x = np.array([(q["iteration"] + 1) * E.STEPS_PER_IT / 1e6 for q in rs])
            sel = np.unique(np.round(np.linspace(0, len(x) - 1, min(160, len(x)))).astype(int))
            data = {}
            for key, _, _ in TB:
                y = np.array([q.get(key, np.nan) for q in rs], float)
                if np.isfinite(y).any():
                    y = np.where(np.isfinite(y), y, np.nanmean(y))
                    data[key] = {"x": x[sel], "med": _smooth(y)[sel]}
            sid = f"tb{s}-{r['arm']}-s{r['seed']}"
            series.append(IF.Series(sid, plabel(s, r), COL[r["arm"]], data, dash=DASH[s], width=1.6,
                                    facets={"plant": s, "arm": r["arm"]}))
            ids.setdefault((s, r["arm"]), []).append(sid)
    if not series:
        return P.pending("Training curves not written yet (tb_dynamics.jsonl).")
    legend = [(f"{CAP[a]}, {PLANT[s]}", COL[a], DASH[s], v) for (s, a), v in ids.items()]
    panels = [IF.Panel(k, t, d, band=False, height=120) for k, t, d in TB]
    facets = [{"key": "plant", "label": "Plant", "values": [("11", "Servo-limited"), ("08", "Working")]},
              {"key": "arm", "label": "Contact model", "values": [("tpu27mesh", "TPU mesh"), ("tpu27pads1", "Pads")]}]
    fig = IF.Figure("tb-curves", None, "Environment steps (millions)", panels, series, facets=facets, legend=legend,
                    x_domain=(0.0, 40.0), x_dec=1)
    return ifig(fig, "Training curves from the event files of the runs with a final policy, smoothed over nine iterations: "
                     "the stochastic policy&#8217;s mean episode return, its alignment term, the grip term (the working plant&#8217;s "
                     "bonus is positive, the servo plant&#8217;s penalty negative) and the action noise; solid the servo-limited "
                     "plant, dashed the working plant.")


def curves_text(X: Data):
    last = tb_last(X)
    ns = lambda s, a: [q.get("Policy/mean_std") for q in last.values() if q["study"] == s and q["arm"] == a]  # noqa: E731
    return (f"<p>The servo-plant runs end training with an action noise of {rng(ns('11', 'tpu27mesh'), 2)} (TPU mesh) and "
            f"{rng(ns('11', 'tpu27pads1'), 2)} (pads), against {rng(ns('08', 'tpu27mesh'), 2)} and "
            f"{rng(ns('08', 'tpu27pads1'), 2)} on the working plant; every run starts at 0.30.</p>")


def curves_section(X: Data):
    return curves_text(X) + ifig_curves(X)


# ------------------------------------------------------------------------------------------ lede, future work

def replay_totals(X: Data, study, arms):
    h = n = 0
    for r in X.finals(study):
        if r["arm"] in arms:
            for e in ENGINES:
                rs = X.replays(study, r["tag"], e)
                h += sum(1 for x in rs if x.get("held_end"))
                n += len(rs)
    return h, n


def lede(X: Data):
    """About 120 words: the outcome, the turn of the pads, the swap, the replays."""
    f11 = X.finals("11")
    if len(f11) < 4:
        return f"Evaluation in progress: {len(f11)} of 4 final policies evaluated."
    hc = lambda s, a: [r["held_cos_mean"] for r in X.finals(s, a)]  # noqa: E731
    grip = lambda rs: [r["grip_N"] for r in rs]  # noqa: E731
    tps = [r for r in X.finals("08") if r["arm"] in ("tpu27mesh", "tpu27pads1")]
    p = [f"Retrained with the finger torque limited to the SCS0009 servo&#8217;s (0.18 instead of 1&#8202;N&#8202;m) and a "
         f"penalty instead of a bonus on fingertip force, the four D6 policies hold the screwdriver in "
         f"{sum(r['n_held'] for r in f11)} of {sum(r['n'] for r in f11)} rollouts at {rng(grip(f11), 0)}&#8202;N of summed "
         f"fingertip force ({rng(grip(tps), 0)}&#8202;N before). The 1&#8202;mm pad policies now turn it nearly as "
         f"far as the point-contact ones: held cosine {rng(hc('11', 'tpu27pads1'), 2)} against "
         f"{rng(hc('11', 'tpu27mesh'), 2)} ({rng(hc('08', 'tpu27pads1'), 2)} against {rng(hc('08', 'tpu27mesh'), 2)} before)."]
    sw = [(r, X.m(r["tag"]), X.m(r["tag"], "fr=1")) for r in f11]
    sw = [x for x in sw if x[1] and x[2]]
    pad_stop = [x for x in sw if x[0]["arm"] == "tpu27pads1" and x[2]["n_held"] == x[2]["n"] and x[2]["held_cos_mean"] < 0.5]
    mesh = [x for x in sw if x[0]["arm"] == "tpu27mesh"]
    if pad_stop and mesh:
        r, m0, m1 = pad_stop[0]
        p.append(f"Given 1&#8202;N&#8202;m, pad seed {r['seed']} grips with {sum(m1['grip_by_finger_N']):.0f}&#8202;N and stops "
                 f"turning the tool (held cosine {m1['held_cos_mean']:.2f}), while the point-contact policies grip with "
                 f"{rng([sum(m['grip_by_finger_N']) for _, _, m in mesh], 0)}&#8202;N and turn it as before.")
    rp, rm = replay_totals(X, "11", ("tpu27pads1",)), replay_totals(X, "11", ("tpu27mesh",))
    if rp[1] and rm[1]:
        p.append(f"In open-loop replays in CPU MuJoCo, Drake and Newton, pad targets kept the tool in {rp[0]} of {rp[1]} and "
                 f"point-contact targets in {rm[0]} of {rm[1]}.")
    return " ".join(p)


def open_items(X: Data):
    tbl = tb_last(X)
    gt11 = [q["Episode_Reward/grip_force_excess"] for q in tbl.values() if q["study"] == "11"]
    al11 = [q["Episode_Reward/target_axis_alignment"] for q in tbl.values() if q["study"] == "11"]
    th11 = [r["grip_thumb_N"] for r in X.finals("11")]
    wh = {a: [X.cost["11"][r["tag"]]["wall_h"] for r in X.finals("11", a) if r["tag"] in X.cost["11"]] for a in ("tpu27mesh", "tpu27pads1")}
    sk = [X.m(r["tag"])["spin_deg"] for r in X.finals("08", "tpu27skin") if X.m(r["tag"])]
    skg = [r["grip_N"] for r in X.finals("08", "tpu27skin")]
    neg = lambda v: f"{-max(v):.2f}&#8211;{-min(v):.2f}"  # noqa: E731
    items = [
        ("Grip penalty weight.", f"The &#8722;2 penalty costs {neg(gt11)} per episode against an alignment term of "
         f"{rng(al11, 0)}, and the thumb presses {rng(th11, 1)}&#8202;N, above the 4&#8202;N threshold, with its mcp at the "
         "torque limit in every held step. Retrain one TPU-mesh and one pad seed with "
         "<code>--grip-force-penalty-weight -20</code> in <code>scripts/rl_contact_train_queue_20261011.sh</code> "
         f"({rng(wh['tpu27mesh'], 1)} and {rng(wh['tpu27pads1'], 1)}&#8202;GPU-h) and evaluate them with "
         "<code>rl_contact_eval.py final --study 20261011</code>. A policy that holds 64 of 64 with the thumb under 4&#8202;N "
         "and a held cosine within 0.05 of these shows that the turn does not need the saturated thumb."),
        ("Torque limit.", "0.18&#8202;N&#8202;m is 80&#8202;% of the datasheet stall torque; the hand&#8217;s servos have not "
         "been measured at stall, and pad seed 0 stops turning the tool at 1&#8202;N&#8202;m. Evaluate the four finals at 0.12 and 0.23&#8202;N&#8202;m "
         "(<code>rl_contact_eval.py robust --study 20261011 --perturb fr=0.12 fr=0.23</code>); a pad policy that stops "
         "turning at 0.23&#8202;N&#8202;m would make the torque limit a parameter to identify on the bench before a "
         "deployment."),
        ("Bench replay of the finger targets.", "Play the recorded targets of the median rollout of each servo-plant final "
         "policy on the hand at 50&#8202;Hz from the lifted grasp (<code>logs/20261011-servo_plant_policies/replay/"
         "&lt;tag&gt;/rec.npz</code> exported through the real_v1 deploy format, after "
         "<code>scripts/real_v1_trajectory_clearance.py</code>; 10 trials each, tracked). Every policy drives the middle "
         "finger to its yaw range limit; a plan the clearance gate rejects for the index&#8211;middle pair ends the item "
         "for that policy. Targets that drop the tool on the hand in most trials, where the three simulators keep it, put "
         "the gap between simulation and hand in the plant or the fingertip model."),
        ("Skin arm.", f"The pads on a sprung skin spun the tool {rng(sk, 0)}&#176; at a grip of {rng(skg, 0)}&#8202;N on the "
         "working plant. Two skin seeds on the servo-limited plant (6.8&#8202;GPU-h) wait on the owner&#8217;s decision."),
    ]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def byline():
    return ("D6 (<code>results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip*</code>, scenes by "
            "<code>scripts/make_work_plant_runs.py --plant kp0.72_kv0_fr0.18_fl0_dp0.0144</code>) &#183; PPO, 2,048 envs, "
            "40&#8202;M steps, seeds 0&#8211;1 (<code>scripts/rl_contact_train_queue_20261011.sh</code>, runs "
            "<code>results/rl/20261011-d6_servo_*_40M_s*</code>) &#183; evaluation <code>scripts/rl_contact_eval.py --study "
            "20261011</code>, replays <code>scripts/rl_policy_replay.py --study 20261011</code> &#183; rows "
            "<code>docs/experiments/20261011-servo_plant_policies/</code> &#183; builder "
            "<code>scripts/servo_plant_policies_page.py</code> &#183; hand-off <code>" + NOTE + "</code> &#183; the study "
            f"on the working plant: <a href=\"{PAGE8}\"><code>{PAGE8_PATH}</code></a>")


def render_tex(t):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    return P.TEX_RE.sub(lambda m: next(svgs), t), len(items)


X_ = None


def main():
    global X_
    X = X_ = Data()
    replay_films = any(f["role"] == "median" and os.path.exists(os.path.join(D, f"media/{f['tag']}_replay_three.mp4"))
                       for f in X.films)
    for k, name in enumerate(["films", "traces", "mech", "brake"] + (["replays"] if replay_films else []) + ["curves"]):
        FIGNO[name] = k + 1
    for k, name in enumerate(["plant", "cost", "final", "mech", "swap", "transfer"]):
        TABNO[name] = k + 1
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "BYLINE": byline()}
    v["LEDE"] = lede(X)
    v["GLOSSARY"] = glossary()
    v["PLANT"] = plant_section(X)
    v["FILMS"] = films(X)
    v["FINAL"] = final_section(X)
    v["MECH"] = mech_section(X)
    v["TRANSFER"] = transfer_section(X)
    v["CURVES"] = curves_section(X)
    v["OPEN"] = open_items(X)
    v["FOOTER"] = ("<p>Rebuild: <code>logs/20261001-hom_contact/venv/bin/python scripts/servo_plant_policies_page.py</code>. "
                   "Rows: <code>docs/experiments/20261011-servo_plant_policies/*.jsonl</code>; per-step traces "
                   "<code>final_traces/</code> (servo-plant policies) and <code>mech_traces/</code> (working-plant policies and "
                   "the torque-limit swap); films <code>media/</code>.</p>")
    if (FIG[0], TAB[0]) != (len(FIGNO), len(TABNO)):
        raise SystemExit(f"figure/table numbering: built {FIG[0]} figures and {TAB[0]} tables, expected {len(FIGNO)} and {len(TABNO)}")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    t, n = render_tex(t)
    t = IF.add_assets(retro_style.apply(t))  # plain page style (owner, 2026-10-09); interactive figures (2026-10-11)
    open(OUT, "w").write(t)
    print(f"formulas {n}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB, figure data {IF.data_bytes(t) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
