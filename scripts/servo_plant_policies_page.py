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
NOTE_SKIN = "docs/handoff/20261012-servo_skin_policies.md"
PAGE8 = "../20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html"
PAGE8_PATH = "docs/experiments/20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html"
SKIN_PAGE = "../20261012-sprung_skin/20261012-sprung_skin_presliding.html"
SKIN_PAGE_PATH = "docs/experiments/20261012-sprung_skin/20261012-sprung_skin_presliding.html"
CONTACT_OV = "../../overviews/contact_model.html"
fmt = P.fmt
ARMS = ("tpu27mesh", "tpu27pads1", "tpu27skin", "box")
LBL = {"box": "Box tip, point contact", "tpu27mesh": "TPU block, point contact", "tpu27pads1": "TPU block, 1&#8202;mm pads",
       "tpu27skin": "TPU block, 1&#8202;mm pads on a sprung skin"}
SHORT = {"box": "box tip", "tpu27mesh": "TPU mesh", "tpu27pads1": "pads", "tpu27skin": "skin"}
CAP = {"box": "Box tip", "tpu27mesh": "TPU mesh", "tpu27pads1": "Pads", "tpu27skin": "Skin"}
COL = {"box": "#48545D", "tpu27mesh": "#2a78d6", "tpu27pads1": "#eb6834", "tpu27skin": "#1baf7a"}
PLANT = {"11": "servo-limited plant", "08": "working plant"}
PLANT_CAP = {"11": "Servo-limited plant, 2026-10-11", "08": "Working plant, 2026-10-08"}
DASH = {"11": "", "08": "6 4"}
ENGINES = ("mujoco", "drake", "newton")
ENG_LBL = {"mujoco": "CPU MuJoCo", "drake": "Drake", "newton": "Newton"}
SCENES11 = ("tpu27mesh", "tpu27pads1", "tpu27skin", "box")
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
        # a second pass of the Newton replays, same worlds, targets and state (logs/20261011-servo_plant_policies/newton_repeat.sh)
        self.rep2 = last_by([dict(r, dir=os.path.basename(r["dir"])) for r in rows(D, "policy_replay_newton_repeat.jsonl")
                             if r.get("engine") == "newton"], ("dir", "world"))
        self.tr = {"11": last_by(rows(D, "transfer.jsonl"), ("tag", "scene")), "08": last_by(rows(D8, "transfer.jsonl"), ("tag", "scene"))}
        self.tb = {"11": rows(D, "tb_dynamics.jsonl"), "08": rows(D8, "tb_dynamics.jsonl")}
        self.cost = {"11": {r["tag"]: r for r in rows(D, "run_costs.jsonl")}, "08": {r["tag"]: r for r in rows(D8, "run_costs.jsonl")}}
        self.rob = {(r["tag"], r["perturb"]): r for r in last_by(rows(D, "robust.jsonl"), ("tag", "perturb"))}
        self.creep = {r["tag"]: r for r in last_by(rows(D, "creep.jsonl"), ("tag",))}
        p = os.path.join(D, "films.json")
        self.films = json.load(open(p)) if os.path.exists(p) else []
        p = os.path.join(D, "plant_check.json")
        self.check = json.load(open(p)) if os.path.exists(p) else {}
        p = os.path.join(D, "plant_check_skin.json")        # the skin scene's check (2026-10-12) and its spring sizing
        self.check_skin = json.load(open(p)) if os.path.exists(p) else {}
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
     "spheres of 1&#8202;mm per fingertip (<b>pads</b>), whose distributed contacts give a torsional friction, or the same "
     "pads on a sprung skin (<b>skin</b>). The 2026-10-08 study also trained the legacy <b>box tip</b> (point contact). In "
     "the tables a policy is its contact model and seed, e.g. Pads s0."),
    ("sprung skin", "Each fingertip&#8217;s pad spheres on a 2&#8202;g body held to the fingertip by two slides tangent to the "
     "palmar face and a hinge about its normal, with springs at the elastic stiffness of the pads&#8217; contact patch at "
     "5&#8202;N per fingertip, the servo-plant policies&#8217; mean hold force (Mindlin&#8217;s \\(k_t = 8Ga/(2-\\nu)\\) = "
     "52.3&#8202;kN/m per slide, Lubkin&#8217;s \\(k_\\theta = 16Ga^3/3\\) = 0.466&#8202;N&#8202;m/rad; patch radius \\(a\\) = "
     "2.94&#8202;mm, \\(G\\) = 3.45&#8202;MPa, \\(\\nu\\) = 0.45), 0.26&#8202;kg of armature on each slide and "
     "3.6&#215;10<sup>&#8722;6</sup>&#8202;kg&#8202;m&#178; on the hinge so that each spring&#8217;s period stays long against the "
     "2&#8202;ms step, and critical damping (<code>scripts/make_work_plant_runs.py --skin</code>; "
     f"<a href=\"{SKIN_PAGE}\">sprung-skin page</a>). The working-plant skin was sized at 12&#8202;N (65&#8202;kN/m)."),
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
    ("perturbation", "One quantity changed in the evaluation of a final policy on its own scene, everything else as trained "
     "(128 rollouts): the sliding friction of every geom &#215;0.7 or &#215;1.3; the tool&#8217;s mass and inertia &#215;0.8 or "
     "&#215;1.2; the finger servos&#8217; gain \\(k_p\\) set to 0.36, 1.08 or 1.8&#8202;N&#8202;m/rad (0.5, 1.5 and 2.5 times "
     "0.72); their torque limit set to 0.12 or 0.23&#8202;N&#8202;m (1&#8202;N&#8202;m in the torque-limit swap); a 1&#8202;ms "
     "simulation step with the same seconds of grasp and lift; <b>grasp jitter</b>, the tool spawned up to 2&#8202;mm off "
     "its trained position and up to 5&#176; turned about vertical, uniform per rollout. The 2026-10-08 page used the same "
     "set with the gains 2, 6 and 10&#8202;N&#8202;m/rad of its 4&#8202;N&#8202;m/rad plant."),
    ("seed spread, Fisher p, bootstrap interval", "The seed spread is the difference between the two seeds of one contact "
     "model. Fisher p: two-sided Fisher exact test on the held counts of two contact models, both seeds pooled. Bootstrap "
     "interval: the 95&#8202;% interval of the difference in held cosine between two contact models over 20,000 resamples, "
     "each drawing two seeds per model with replacement and then the held rollouts of each drawn seed; a model&#8217;s held "
     "cosine is the mean of its seeds&#8217;. A difference is beyond the seed spread when both seeds of one model lie outside "
     "the range of the other model&#8217;s two seeds, on the same side; for the held cosine that is when the bootstrap interval "
     "excludes zero. Fisher&#8217;s test counts every rollout as independent and can separate two models whose seeds "
     "overlap."),
    ("slide, spring deflection", "The slip record follows the tool&#8217;s material point under each fingertip&#8217;s contact "
     "centroid from one policy step to the next. Its displacement tangential to the palmar face, against the body that "
     "carries the pads (the skin body, else the fingertip), is the <b>slide</b> over the fingertip: the friction rows&#8217; "
     "creep and any slip; net over the hold (steps 150&#8211;250, 2&#8202;s) per second, in &#181;m/s. The <b>spring "
     "deflection</b> is the skin&#8217;s own displacement against its springs: the slides&#8217; tangential displacement in "
     "&#181;m and the hinge&#8217;s angle in mrad. The <b>spring force over the friction limit</b> \\(k_t|q|/\\mu N\\), with "
     "\\(N\\) the fingertip&#8217;s contact force, is the share of the presliding range in use; at 1 the pads slide at the "
     "friction limit."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


# ------------------------------------------------------------------------------------------ plant and cost

def hours(study, tag):
    c = X_.cost[study].get(tag, {})
    return c.get("wall_h")


def plant_section(X: Data):
    chk = X.check.get("checks", []) + X.check_skin.get("checks", [])
    ok = [c for c in chk if all(c[k]["held_worlds"] == 64 and min(c[k]["tip_contacts_mean"]) > 0 for k in ("step60", "step150"))]
    head = ["", "Working plant<br>2026-10-08", "Servo-limited plant<br>2026-10-11"]
    body = [["Position gain \\(k_p\\) (N&#8202;m/rad)", "4", "0.72"],
            ["Torque limit \\(\\tau_\\text{max}\\) (N&#8202;m)", "1", "0.18"],
            ["Joint damping \\(b\\) (N&#8202;m&#8202;s/rad)", "0.08", "0.0144"],
            ["Grip term weight", "+0.25 (bonus)", "&#8722;2 (penalty)"],
            ["Contact models trained", "box tip, TPU mesh, pads, skin", "TPU mesh, pads, skin"],
            ["Skin springs, slide (kN/m) &#183; hinge (N&#8202;m/rad)", "65 &#183; 0.90, sized at 12&#8202;N",
             (f"{X.check_skin['skin']['kt'] / 1e3:.1f} &#183; {X.check_skin['skin']['kth']:.3f}, sized at "
              f"{X.check_skin['skin']['N_ref']:.0f}&#8202;N") if X.check_skin.get("skin") else "&#8211;"],
            ["Seeds with a final policy", f"{len(X.finals('08', 'tpu27mesh'))} TPU mesh, {len(X.finals('08', 'tpu27pads1'))} pads, "
             f"{len(X.finals('08', 'tpu27skin'))} skin",
             f"{len(X.finals('11', 'tpu27mesh'))} TPU mesh, {len(X.finals('11', 'tpu27pads1'))} pads, "
             f"{len(X.finals('11', 'tpu27skin'))} skin"]]
    out = ("<p>The four TPU-mesh and pad runs repeat the 2026-10-08 runs with two changes, the plant and the sign of the "
           "grip term (Table&#160;1); two runs on the sprung skin followed on 2026-10-12 with the same flags, its springs sized "
           "at the 5&#8202;N per fingertip with which the servo-plant policies hold the tool. Everything else is as on "
           "2026-10-08: hand D6, the 24.5&#8202;g screwdriver, MuJoCo-Warp at 2&#8202;ms with impratio 10, recipe "
           "<code>b_liveA</code>, 2,048 envs, 40&#8202;M steps, friction randomisation, seeds 0 and 1. Before training, the "
           "scripted grasp and lift held the tool in 64 of 64 worlds with all three fingertips in contact at policy steps 60 "
           f"and 150 on {len(ok)} of {len(chk)} servo-plant scenes (box tip, TPU mesh, pads and skin; "
           "<code>plant_check.json</code>, <code>plant_check_skin.json</code>).</p>")
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
          for a in ("tpu27mesh", "tpu27pads1", "tpu27skin")}
    sp8 = {a: [c["s_per_it_median"] for t, c in X.cost["08"].items() if arm_of(t) == a and c.get("s_per_it_median")]
           for a in ("tpu27mesh", "tpu27pads1")}
    tot = sum(X.cost["11"][r["tag"]].get("wall_h") or 0 for r in X.finals("11") if r["tag"] in X.cost["11"])
    if all(sp.values()) and all(sp8.values()):
        nrun = sum(1 for r in X.finals("11") if r["tag"] in X.cost["11"])
        out += (f"<p>The {WORDS.get(nrun, nrun)} runs took {tot:.1f}&#8202;GPU-h on the RTX 4070 Ti SUPER, one at a time (Table&#160;2). A pad "
                f"iteration cost {np.median(sp['tpu27pads1']) / np.median(sp['tpu27mesh']):.2f}&#215; a TPU-mesh iteration "
                f"({np.median(sp['tpu27pads1']):.2f} against {np.median(sp['tpu27mesh']):.2f}&#8202;s) and a skin iteration "
                f"{np.median(sp['tpu27skin']) / np.median(sp['tpu27mesh']):.2f}&#215; ({np.median(sp['tpu27skin']):.2f}&#8202;s); "
                f"in the 2026-10-08 runs, "
                f"which shared the workstation with other jobs, a pad iteration cost {np.median(sp8['tpu27pads1']) / np.median(sp8['tpu27mesh']):.2f}&#215; "
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
    return ("<p>The four TPU-mesh and pad final policies turn the tool the same way; the skin policies are in "
            f"Figure&#160;{FIGNO['skin_films']}. From the onset the middle finger lifts the handle end "
            "and the tool pivots about the thumb and index contacts; within a quarter second it stands about 25&#8211;40&#176; "
            "from vertical. The middle finger ends horizontal at the limit of its yaw range, pressing on the upper end of the "
            "tool with its pip joint at the extension limit, and the thumb rotates against the turn until it lies along the "
            "shaft.</p>"
            f'<figure class="films">{"".join(cells)}<figcaption>Figure&#160;{FIG[0]}. The final policy of both seeds of the '
            "TPU mesh and the pads on the servo-limited plant (best: more held rollouts, then the higher held cosine), one "
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
                     "skin policies of both plants and the working plant&#8217;s box-tip policy. Drag to zoom, double-click to "
                     "reset.")


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
           f"{rng(hc('11', 'tpu27pads1'), 3)} (working plant {rng(hc('08', 'tpu27pads1'), 3)}) and that of the skin policies "
           f"{rng(hc('11', 'tpu27skin'), 3)} (working plant {rng(hc('08', 'tpu27skin'), 3)}). The summed fingertip force falls "
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
          f"and {rng(tn('11', 'tpu27mesh'), 0)}&#176;. The skin policies spun the tool {rng(sp('08', 'tpu27skin'), 0)}&#176; at a "
          f"pad brake of {rng(br('08', 'tpu27skin'), 0)}&#8202;mN&#8202;m on the working plant, as far as its TPU-mesh policies, "
          f"and spin it {rng(sp('11', 'tpu27skin'), 0)}&#176; at {rng(br('11', 'tpu27skin'), 0)}&#8202;mN&#8202;m on the "
          "servo-limited plant.</p>")
    p2 = (f"<p>The thumb&#8217;s mcp actuator sits at its torque limit in {rng([100 * v for v in tl], 0)}&#8202;% of the held "
          f"steps of every TPU-block policy on both plants, so the actuator limit sets the grip; the grip term was "
          f"{sgn_rng(gt['08'])} per episode at the end of the working-plant runs and is {sgn_rng(gt['11'])} in the "
          f"servo-plant runs, against an alignment term of {rng(al, 0)} (Figure&#160;{FIGNO['curves']}). On the servo-limited "
          f"plant a finger joint sits at its extension limit in {rng([100 * v for v in ext11], 0)}&#8202;% of the held steps, "
          f"and the middle finger&#8217;s yaw at the end of its range in {rng([100 * v for v in yaw11 if v > 0.5], 0)}&#8202;% "
          + (f"in {WORDS.get(sum(v > 0.5 for v in yaw11), sum(v > 0.5 for v in yaw11))} of the {WORDS.get(len(yaw11), len(yaw11))} "
             "finals; " + " and ".join(f"{CAP[r['arm']].lower()} seed {r['seed']}" for r in f11 if X.m(r["tag"])
                                      and X.m(r["tag"])["range_limit_by_joint"]["middle_yaw"] <= 0.5)
             + (" never brings it there " if max(v for v in yaw11 if v <= 0.5) < 0.01 else
                f" brings it there in {rng([100 * v for v in yaw11 if v <= 0.5], 0)}&#8202;% ") if min(yaw11) <= 0.5 else "")
          + f"(Table&#160;{TABNO['mech']}).</p>")
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
    out = [f"<p>Given 1&#8202;N&#8202;m, the TPU-mesh and pad policies of the servo-limited plant grip {rng(ratio, 1)}&#215; harder "
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
    sk = pairs[("11", "tpu27skin")]
    if sk:
        out.append(" The skin policies grip " + " and ".join(f"{gr(m1) / gr(m0):.1f}" for _, m0, m1 in sk) + "&#215; harder ("
                   + " and ".join(f"{gr(m1):.0f}" for _, _, m1 in sk) + "&#8202;N); "
                   + "; ".join(f"seed {r['seed']} holds {m1['n_held']} of {m1['n']} rollouts (held cosine "
                               f"{m0['held_cos_mean']:.3f} &#8594; {m1['held_cos_mean']:.3f})" for r, m0, m1 in sk) + ".")
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
    head = ["Policy", "TPU mesh,<br>closed loop", "Pads,<br>closed loop", "Skin,<br>closed loop", "Box tip,<br>closed loop",
            "CPU MuJoCo,<br>replay", "Drake,<br>replay", "Newton,<br>replay"]
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
                             "Open-loop replays: worlds held of worlds replayed &#183; mean final cosine of the held ones; Drake "
                             "and Newton replace the skin by the hydroelastic TPU block.")


def transfer_text(X: Data):
    tot = lambda s, arms: replay_totals(X, s, arms)  # noqa: E731
    pads11, mesh11, skin11 = tot("11", ("tpu27pads1",)), tot("11", ("tpu27mesh",)), tot("11", ("tpu27skin",))
    comp08, point08 = tot("08", ("tpu27pads1", "tpu27skin")), tot("08", ("tpu27mesh", "box"))
    gap = []
    for r in X.finals("11"):
        for e in ENGINES:
            for x in X.replays("11", r["tag"], e):
                rec = next((q for q in X.rec["11"] if q["tag"] == r["tag"]), None)
                if rec and x.get("held_end") and x.get("world") is not None and x["world"] < len(rec["mjw_cos_end"]):
                    gap.append((r["arm"], e, x["cos_end"] - rec["mjw_cos_end"][x["world"]]))
    g = {(a, e): [v for aa, ee, v in gap if aa == a and ee == e] for a in SKIN_ARMS for e in ENGINES}
    cross = [t for t in X.tr["11"] if t["scene"] in SKIN_ARMS and t["scene"] != t["arm"]]
    box = [t for t in X.tr["11"] if t["scene"] == "box"]
    out = (f"<p>Replayed open loop on the servo-limited plant, the finger targets of the pad policies kept the tool in "
           f"{pads11[0]:,} of {pads11[1]:,} replays, those of the TPU-mesh policies in {mesh11[0]:,} of {mesh11[1]:,} and those "
           f"of the skin policies in {skin11[0]:,} of {skin11[1]:,}; on the working plant the pad and skin targets kept it in "
           f"{comp08[0]:,} of {comp08[1]:,} and the point-contact targets in {point08[0]:,} of {point08[1]:,} "
           f"(<a href=\"{PAGE8}\">2026-10-08 page</a>). ")
    if cross:
        out += (f"Closed loop on the other two TPU-block contact models, the {WORDS.get(len(X.finals('11')), len(X.finals('11')))} policies held "
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
    nt = {a: replay_totals_engine(X, "11", a, "newton") for a in ("tpu27pads1", "tpu27mesh", "tpu27skin")}
    nt8 = {a: replay_totals_engine(X, "08", a, "newton") for a in ("tpu27pads1",)}
    med = lambda v: (f"{np.median(v):+.3f}".replace("-", "&#8722;") if v else "&#8211;")  # noqa: E731
    out += ("The final cosine of a held replay minus that of the MuJoCo-Warp rollout it replays is a median "
            + ", ".join(f"{med(g[('tpu27pads1', e)])} (pads), {med(g[('tpu27skin', e)])} (skin) and {med(g[('tpu27mesh', e)])} "
                        f"(TPU mesh) in {ENG_LBL[e]}" for e in ENGINES) + ".</p>")
    if falls:
        early = sum(1 for t in falls if t <= 1.0)
        out += (f"<p>Newton loses the most replays: the pad targets keep the tool in {nt['tpu27pads1'][0]} of "
                f"{nt['tpu27pads1'][1]} worlds there ({nt8['tpu27pads1'][0]} of {nt8['tpu27pads1'][1]} at the working "
                f"plant&#8217;s grip), the TPU-mesh targets in {nt['tpu27mesh'][0]} of {nt['tpu27mesh'][1]} and the skin targets "
                f"in {nt['tpu27skin'][0]} of {nt['tpu27skin'][1]}. The tool falls "
                f"{min(falls):.1f}&#8211;{max(falls):.1f}&#8202;s after the onset, in {early} of {len(falls)} worlds within the "
                "first second, after the turn has ended"
                + (f"; one held pad replay ends turned to the wrong pole (cos &#8722;0.80)" if flips == 1 else
                   (f"; {flips} held replays end turned to the wrong pole" if flips else "")) + ".")
        out += newton_repeat_text(X) + "</p>"
    return out


def newton_repeat_text(X: Data):
    """The second Newton pass against the first, world by world."""
    if not X.rep2:
        return ""
    first = {(r["dir"], r["world"]): r for r in X.rep["11"] if r["engine"] == "newton"}
    pairs = [(first[(r["dir"], r["world"])], r) for r in X.rep2 if (r["dir"], r["world"]) in first]
    if not pairs:
        return ""
    same = sum(1 for a, b in pairs if bool(a.get("held_end")) == bool(b.get("held_end")))
    h2 = {a: (sum(1 for _, b in pairs if arm_of(b["dir"]) == a and b.get("held_end")),
              sum(1 for _, b in pairs if arm_of(b["dir"]) == a)) for a in ("tpu27pads1", "tpu27mesh")}
    head = (f" A second Newton pass over the same worlds, targets and onset states kept the tool in {h2['tpu27pads1'][0]} of "
            f"{h2['tpu27pads1'][1]} pad and {h2['tpu27mesh'][0]} of {h2['tpu27mesh'][1]} TPU-mesh replays")
    if same == len(pairs):
        return head + f", and all {len(pairs)} worlds ended the same way in both passes."
    return (head + f"; {same} of {len(pairs)} worlds ended the same way in both passes, so Newton&#8217;s GPU contact solve "
            "does not repeat a replay of this grip exactly.")


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
        p = os.path.join(D, f"media/{tag}_replay_{e}.json")       # the filmed run's own end state
        x = json.load(open(p)) if os.path.exists(p) else None
        if x is None:
            continue
        parts.append(f"{ENG_LBL[e]} holds it at cos {x['cos_end']:.2f}" if x.get("held_end") else
                     f"{ENG_LBL[e]} drops it (tool at {x['z_end_mm']:.0f}&#8202;mm at the end)")
    rows_ = [q for q in X.replays("11", tag, "newton") if q.get("world") == world] + \
            [q for q in X.rep2 if q["dir"] == tag and q.get("world") == world]
    if rows_:
        parts.append(("the two Newton replay passes of this world " if len(rows_) == 2 else "the Newton replay of this world ")
                     + " and then ".join("held" if q.get("held_end") else "dropped" for q in rows_) + " it")
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


# ------------------------------------------------------------------------------------------ the sprung skin (2026-10-12)

SKIN_ARMS = ("tpu27mesh", "tpu27pads1", "tpu27skin")
# conditions of the comparison, in groups: (key, label); keys: an evaluation of rl_contact_eval (nominal: the final
# evaluation; robust perturbations by name; swap:<perturb> the mechanism evaluation; tr:<scene> closed-loop transfer) or
# rp:<engine>, the open-loop replays
SKIN_COND = [
    ("Nominal and grasp jitter", [("nominal", "Nominal, 64 rollouts"), ("noise=2,5", "Grasp jitter, 2&#8202;mm and 5&#176;")]),
    ("Perturbations, 128 rollouts", [
        ("friction=0.7", "Friction &#215;0.7"), ("friction=1.3", "Friction &#215;1.3"),
        ("mass=0.8", "Tool mass &#215;0.8"), ("mass=1.2", "Tool mass &#215;1.2"),
        ("kp=0.36", "Servo gain 0.36&#8202;N&#8202;m/rad"), ("kp=1.08", "Servo gain 1.08&#8202;N&#8202;m/rad"),
        ("kp=1.8", "Servo gain 1.8&#8202;N&#8202;m/rad"), ("fr=0.12", "Torque limit 0.12&#8202;N&#8202;m"),
        ("fr=0.23", "Torque limit 0.23&#8202;N&#8202;m"), ("swap:fr=1", "Torque limit 1&#8202;N&#8202;m, 64 rollouts"),
        ("dt=0.001", "Simulation step 1&#8202;ms")]),
    ("Closed loop on a tip model, 256 rollouts", [("tr:tpu27mesh", "TPU mesh"), ("tr:tpu27pads1", "Pads"),
                                                  ("tr:tpu27skin", "Skin"), ("tr:box", "Box tip")]),
    ("Open-loop replays of the finger targets", [("rp:mujoco", "CPU MuJoCo, 16 worlds"), ("rp:drake", "Drake, 16 worlds"),
                                                 ("rp:newton", "Newton, 12 worlds")]),
]
PERTURB_KEYS = [k for g, cs in SKIN_COND if g.startswith("Perturbations") for k, _ in cs]
# each condition as a phrase in running text
PROSE = {"nominal": "nominal", "noise=2,5": "under grasp jitter", "friction=0.7": "at 0.7 times the friction",
         "friction=1.3": "at 1.3 times the friction", "mass=0.8": "at 0.8 times the tool mass",
         "mass=1.2": "at 1.2 times the tool mass", "kp=0.36": "at a servo gain of 0.36&#8202;N&#8202;m/rad",
         "kp=1.08": "at a servo gain of 1.08&#8202;N&#8202;m/rad", "kp=1.8": "at a servo gain of 1.8&#8202;N&#8202;m/rad",
         "fr=0.12": "at a torque limit of 0.12&#8202;N&#8202;m", "fr=0.23": "at a torque limit of 0.23&#8202;N&#8202;m",
         "swap:fr=1": "at a torque limit of 1&#8202;N&#8202;m", "dt=0.001": "at a 1&#8202;ms step",
         "tr:tpu27mesh": "closed loop on the TPU mesh", "tr:tpu27pads1": "closed loop on the pads",
         "tr:tpu27skin": "closed loop on the skin", "tr:box": "closed loop on the box tip",
         "rp:mujoco": "replayed in CPU MuJoCo", "rp:drake": "replayed in Drake", "rp:newton": "replayed in Newton"}
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
ROBUST_KEYS = ["noise=2,5"] + [k for k in PERTURB_KEYS if not k.startswith("swap:")]    # jitter and ten perturbations
_CMP = {}


def swap_eval(X: Data, tag, perturb):
    """Held count and final cosines of a mechanism evaluation under `perturb`, from its saved traces."""
    m = X.m(tag, perturb)
    p = os.path.join(D, m["traces"]) if m else None
    if not p or not os.path.exists(p):
        return None
    tr, _ = E.load_trace(p)
    h = held_mask(tr)[-1]
    return dict(n=int(h.size), n_held=int(h.sum()), final_cos=tr["cos"][-1].tolist(), held_final=h.tolist())


def cond_eval(X: Data, tag, key):
    """{n, h, arr}: rollouts (or replay worlds), held ones and the final cosines of the held ones, of one policy under one
    condition of SKIN_COND; None where it was not evaluated."""
    if key.startswith("rp:"):
        rs = X.replays("11", tag, key[3:])
        if not rs:
            return None
        return dict(n=len(rs), h=sum(1 for x in rs if x.get("held_end")),
                    arr=np.array([x["cos_end"] for x in rs if x.get("held_end")], float))
    if key == "nominal":
        r = next((q for q in X.fin["11"] if q["tag"] == tag), None)
    elif key.startswith("swap:"):
        r = swap_eval(X, tag, key[5:])
    elif key.startswith("tr:"):
        r = next((t for t in X.tr["11"] if t["tag"] == tag and t["scene"] == key[3:]), None)
    else:
        r = X.rob.get((tag, key))
    if not r:
        return None
    fc, hf = np.asarray(r["final_cos"], float), np.asarray(r["held_final"], bool)
    return dict(n=int(r["n"]), h=int(r["n_held"]), arr=fc[hf])


def fisher_p(a, b):
    """Two-sided Fisher exact test on two (held, n) counts."""
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]], alternative="two-sided")[1])


def boot_delta(a, b, n=20000, seed=0):
    """Difference of two contact models' held cosine (a minus b; each a list of per-seed arrays of the held rollouts'
    final cosines; a model's held cosine is the mean of its seeds' means) and its 95 % interval from n resamples, each
    drawing as many seeds as the model has, with replacement, and then the rollouts of each drawn seed with replacement."""
    if not a or not b or any(len(x) == 0 for x in a + b):
        return None
    rng = np.random.default_rng(seed)

    def draw(arms):
        k = len(arms)
        pick = rng.integers(0, k, size=(n, k))
        out = np.zeros(n)
        for j in range(k):
            bm = np.stack([x[rng.integers(0, len(x), size=(n, len(x)))].mean(1) for x in arms], 1)
            out += bm[np.arange(n), pick[:, j]]
        return out / k

    d = draw(a) - draw(b)
    return (float(np.mean([x.mean() for x in a]) - np.mean([x.mean() for x in b])),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def boot_ci(x, n=20000, seed=0):
    """95 % interval of the mean of x over n resamples of its elements."""
    if len(x) < 2:
        return None
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def skin_compare(X: Data):
    """Per condition of SKIN_COND: each seed's evaluation per contact model, the held counts pooled over seeds, Fisher's
    p of the skin against the pads and the TPU mesh, and the bootstrap difference of held cosine (boot_delta)."""
    if _CMP.get("rows") is not None:
        return _CMP["rows"]
    out = []
    for group, conds in SKIN_COND:
        for key, label in conds:
            seeds = {a: [(r["seed"], cond_eval(X, r["tag"], key)) for r in X.finals("11", a)] for a in SKIN_ARMS}
            ok = all(seeds[a] and all(v is not None for _, v in seeds[a]) for a in SKIN_ARMS)
            row = dict(group=group, key=key, label=label, seeds=seeds, ok=ok)
            if ok:
                pooled = {a: (sum(v["h"] for _, v in seeds[a]), sum(v["n"] for _, v in seeds[a])) for a in SKIN_ARMS}
                arr = {a: [v["arr"] for _, v in seeds[a]] for a in SKIN_ARMS}
                row.update(pooled=pooled,
                           p={b: fisher_p(pooled["tpu27skin"], pooled[b]) for b in ("tpu27pads1", "tpu27mesh")},
                           d={b: boot_delta(arr["tpu27skin"], arr[b]) for b in ("tpu27pads1", "tpu27mesh")},
                           hc={a: [(float(x.mean()) if len(x) else None) for x in arr[a]] for a in SKIN_ARMS})
            out.append(row)
    _CMP["rows"] = out
    return out


def seed_vals(row, a, what):
    """The two seeds' held shares ('held') or held cosines ('cos') of contact model a in one comparison row."""
    if what == "held":
        return [v["h"] / v["n"] for _, v in row["seeds"][a]]
    return [x for x in row["hc"][a] if x is not None]


def beyond(row, b, what):
    """Whether the skin differs from contact model b beyond the seed spread in one condition: both skin seeds outside
    the range of b's two seeds, on the same side."""
    sk, ot = seed_vals(row, "tpu27skin", what), seed_vals(row, b, what)
    if len(sk) < 2 or len(ot) < 2:
        return False
    return max(sk) < min(ot) or min(sk) > max(ot)


def fmt_p(p):
    if p is None:
        return "&#8211;"
    if p >= 0.995:
        return "1"
    if p >= 0.01:
        return f"{p:.2f}"
    m, e = f"{p:.0e}".split("e")
    return f"{m}&#215;10<sup>&#8722;{int(e[1:])}</sup>"


def fmt_d(d):
    if d is None:
        return "&#8211;"
    s = lambda v: (f"+{v:.2f}" if v >= 0.005 else ("0.00" if v > -0.005 else f"&#8722;{-v:.2f}"))  # noqa: E731
    txt = f"{s(d[0])} [{s(d[1])}, {s(d[2])}]"
    return f"<b>{txt}</b>" if d[1] > 0 or d[2] < 0 else txt


def skin_tables(X: Data):
    """Two tables of the comparison: held counts with Fisher's p, and held cosines with the bootstrap differences."""
    cmp_ = skin_compare(X)
    hd = lambda last: ["Condition", "TPU mesh<br>s0, s1", "Pads<br>s0, s1", "Skin<br>s0, s1"] + last  # noqa: E731

    def body(cells):
        out, grp = [], None
        for row in cmp_:
            if row["group"] != grp:
                grp = row["group"]
                out.append(grp)
            out.append([row["label"]] + (cells(row) if row["ok"] else ["&#8211;"] * 5))
        return out

    def held(row):
        c = []
        for a in SKIN_ARMS:
            vs = [v for _, v in row["seeds"][a]]
            n = {v["n"] for v in vs}
            c.append(", ".join(str(v["h"]) for v in vs) + (f" of {n.pop()}" if len(n) == 1 else ""))
        return c + [(f"<b>{fmt_p(row['p'][b])}</b>" if row["p"][b] < 0.05 else fmt_p(row["p"][b]))
                    for b in ("tpu27pads1", "tpu27mesh")]

    def cos(row):
        return ([", ".join(f1(v, 2) for v in row["hc"][a]) for a in SKIN_ARMS]
                + [fmt_d(row["d"]["tpu27pads1"]), fmt_d(row["d"]["tpu27mesh"])])

    t1 = table(hd(["Fisher p, skin<br>against pads", "Fisher p, skin<br>against TPU mesh"]), body(held),
               "Held rollouts of the TPU-mesh, pad and skin policies on the servo-limited plant, seed 0 and seed 1 (replays: "
               "worlds held). Fisher p: two-sided Fisher exact test on the held counts of the skin and of the other contact "
               "model, both seeds pooled; bold under 0.05.")
    t2 = table(hd(["Skin &#8722; pads<br>[95&#8202;% interval]", "Skin &#8722; TPU mesh<br>[95&#8202;% interval]"]), body(cos),
               "Held cosine of the same evaluations, seed 0 and seed 1: the mean final cosine over the held rollouts (replays: "
               "the held worlds). The differences are of the two seeds&#8217; mean, with the 95&#8202;% interval of a bootstrap "
               "that resamples seeds and then rollouts; bold where it excludes zero.")
    return t1, t2


# the comparison figure: one row per condition, a lane per contact model, a dot per seed (seed 0 filled, seed 1 a ring),
# joined by the seed spread; held share on the left, held cosine on the right
def svg_skin_compare(X: Data):
    rows_ = skin_compare(X)
    W_, FS_ = 828, 14
    LAB, PA0, PA1, PB0, PB1 = 8, 262, 470, 510, 812           # label x; held-share and held-cosine panel extents (px)
    LANE, ROWGAP, GRP_H, TOP = 8, 10, 30, 74
    CLO, CHI = 0.4, 1.0
    xa = lambda v: PA0 + (PA1 - PA0) * v  # noqa: E731
    xb = lambda v: PB0 + (PB1 - PB0) * (min(max(v, CLO), CHI) - CLO) / (CHI - CLO)  # noqa: E731
    out, y, grp = [], TOP, None
    marks = []
    for row in rows_:
        if row["group"] != grp:
            grp = row["group"]
            y += 8
            out.append(f'<text x="{LAB}" y="{y + 16}" font-size="{FS_}" font-weight="bold" fill="#000">{grp}</text>')
            y += GRP_H - 4
        h = LANE * 3
        out.append(f'<text x="{LAB + 10}" y="{y + h / 2 + 5:.1f}" font-size="{FS_}" fill="#000">'
                   f'{row["label"].replace("&#8202;", " ")}</text>')
        for gx0, gx1 in ((PA0, PA1), (PB0, PB1)):
            out.append(f'<rect x="{gx0}" y="{y - 3}" width="{gx1 - gx0}" height="{h + 6}" fill="#f4f4f2"/>')
        for k, a in enumerate(SKIN_ARMS):
            ly = y + LANE * k + LANE / 2
            pts = []
            for s, v in row["seeds"][a]:
                if v is None:
                    continue
                hs = v["h"] / v["n"]
                hcv = float(v["arr"].mean()) if len(v["arr"]) else None
                ci = boot_ci(v["arr"]) if len(v["arr"]) > 1 else None
                tip = (f"{CAP[a]} s{s}, {row['label']}: held {v['h']}/{v['n']}"
                       + (f", held cosine {hcv:.3f}" + (f" (95&#8202;% interval {ci[0]:.3f}&#8211;{ci[1]:.3f})" if ci else "")
                          if hcv is not None else ""))
                pts.append((s, hs, hcv, tip))
            for panel, fx, idx in (("a", xa, 1), ("b", xb, 2)):
                vals = [(p[0], p[idx], p[3]) for p in pts if p[idx] is not None]
                if len(vals) == 2:
                    x0, x1 = sorted(fx(v[1]) for v in vals)
                    out.append(f'<line x1="{x0:.1f}" x2="{x1:.1f}" y1="{ly:.1f}" y2="{ly:.1f}" stroke="{COL[a]}" '
                               f'stroke-width="2.5" stroke-opacity="0.55" stroke-linecap="round"/>')
                for s, v, tip in vals:
                    marks.append((s, fx(v), ly, COL[a], tip, v if panel == "b" else None))
        y += h + ROWGAP
    for s, x, ly, c, tip, v in sorted(marks, key=lambda m: -m[0]):    # seed 1 rings first, seed 0 dots on top
        if v is not None and v < CLO:                  # a held cosine below the axis: a triangle at its edge, with the value
            out.append(f'<path d="M{PB0 - 1:.1f},{ly:.1f} l7,-5 l0,10 z" fill="{c if s == 0 else "#fff"}" stroke="{c}" '
                       f'stroke-width="1.5"><title>{tip}</title></path>'
                       f'<text x="{PB0 + 10:.1f}" y="{ly + 4.5:.1f}" font-size="13" fill="#000">{v:.2f}</text>')
            continue
        if s == 0:
            out.append(f'<circle cx="{x:.1f}" cy="{ly:.1f}" r="4.5" fill="{c}" stroke="#fff" stroke-width="1.5">'
                       f'<title>{tip}</title></circle>')
        else:
            out.append(f'<circle cx="{x:.1f}" cy="{ly:.1f}" r="4" fill="#fff" stroke="{c}" stroke-width="2">'
                       f'<title>{tip}</title></circle>')
    bottom = y + 4
    # axes: gridlines, ticks and titles
    ax = []
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        ax.append(f'<line x1="{xa(v):.1f}" x2="{xa(v):.1f}" y1="{TOP - 6}" y2="{bottom}" stroke="#d9d9d9" stroke-width="1"/>')
        ax.append(f'<text x="{xa(v):.1f}" y="{bottom + 18}" font-size="{FS_}" text-anchor="middle" fill="#000">{v:g}</text>')
    for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        ax.append(f'<line x1="{xb(v):.1f}" x2="{xb(v):.1f}" y1="{TOP - 6}" y2="{bottom}" stroke="#d9d9d9" stroke-width="1"/>')
        ax.append(f'<text x="{xb(v):.1f}" y="{bottom + 18}" font-size="{FS_}" text-anchor="middle" fill="#000">{v:g}</text>')
    ax.append(f'<text x="{(PA0 + PA1) / 2}" y="{TOP - 14}" font-size="{FS_}" font-weight="bold" text-anchor="middle" '
              f'fill="#000">Held share</text>')
    ax.append(f'<text x="{(PB0 + PB1) / 2}" y="{TOP - 14}" font-size="{FS_}" font-weight="bold" text-anchor="middle" '
              f'fill="#000">Held cosine</text>')
    ax.append(f'<text x="{(PA0 + PA1) / 2}" y="{bottom + 38}" font-size="{FS_}" text-anchor="middle" fill="#444">'
              f'Share of rollouts holding the tool at 5 s</text>')
    ax.append(f'<text x="{(PB0 + PB1) / 2}" y="{bottom + 38}" font-size="{FS_}" text-anchor="middle" fill="#444">'
              f'Mean final cosine of the held rollouts</text>')
    # legend: contact models, then the seed glyphs
    lg, lx = [], LAB
    for a in SKIN_ARMS:
        lg.append(f'<line x1="{lx}" x2="{lx + 26}" y1="16" y2="16" stroke="{COL[a]}" stroke-width="2.5" stroke-opacity="0.55"/>'
                  f'<circle cx="{lx + 4}" cy="16" r="4.5" fill="{COL[a]}" stroke="#fff" stroke-width="1.5"/>'
                  f'<circle cx="{lx + 22}" cy="16" r="4" fill="#fff" stroke="{COL[a]}" stroke-width="2"/>'
                  f'<text x="{lx + 34}" y="21" font-size="{FS_}" fill="#000">{ {"tpu27mesh": "TPU mesh", "tpu27pads1": "1 mm pads", "tpu27skin": "Pads on the sprung skin"}[a]}</text>')
        lx += {"tpu27mesh": 118, "tpu27pads1": 120, "tpu27skin": 214}[a]
    lg.append(f'<circle cx="{lx + 6}" cy="16" r="4.5" fill="#666" stroke="#fff" stroke-width="1.5"/>'
              f'<text x="{lx + 16}" y="21" font-size="{FS_}" fill="#000">seed 0</text>'
              f'<circle cx="{lx + 80}" cy="16" r="4" fill="#fff" stroke="#666" stroke-width="2"/>'
              f'<text x="{lx + 90}" y="21" font-size="{FS_}" fill="#000">seed 1; the line spans the two seeds</text>')
    H_ = bottom + 50
    return (f'<svg viewBox="0 0 {W_} {H_}" width="100%" role="img" aria-label="Held share and held cosine of the TPU-mesh, '
            f'pad and skin policies, two seeds each, under every evaluation condition" style="max-width:{W_}px;'
            f'font-family:inherit">' + "".join(lg + ax + out) + "</svg>")


def skin_films(X: Data):
    """Close-up films of the two skin finals (best and median seed), the pad and skin finals of seed 0 side by side, and
    the three-engine replay of the median skin seed."""
    out = []
    rk = ranked(X, "11", "tpu27skin")
    cells = []
    for role, r in zip(("best", "median"), rk):
        rel = f"media/{r['tag']}_{role}.mp4"
        if not os.path.exists(os.path.join(D, rel)):
            continue
        tr = X.trace(r["tag"])
        gr = grip_range(tr[0]) if tr else None
        tt = turn_time(tr[0]) if tr else None
        c70 = float(np.median(tr[0]["cos"][ONSET + 11][held_mask(tr[0])[-1]])) if tr else None
        cells.append(f'<div class="cell">{video(rel)}<p class="filmlab">{swatch(r["arm"])}<b>{LBL[r["arm"]]}</b>, seed '
                     f'{r["seed"]} ({role} seed): held {r["n_held"]}/64, held cos {r["held_cos_mean"]:.2f}; cos {c70:.2f} '
                     f'at 0.24&#8202;s after the onset, 90&#8202;% of the final cosine at {tt:.2f}&#8202;s; summed fingertip force '
                     f'in the hold {gr[0]:.0f}&#8211;{gr[1]:.0f}&#8202;N. <code>{rel}</code></p></div>')
    if cells:
        FIG[0] += 1
        out.append(f'<figure class="films">{"".join(cells)}<figcaption>Figure&#160;{FIG[0]}. The two skin final policies, one '
                   "deterministic rollout each, rendered as in Figure&#160;1: the scripted grasp and lift (0&#8211;1.16&#8202;s), "
                   "then the policy&#8217;s turn and hold to 5&#8202;s, camera 0.24&#8202;m from the hand. Paths relative to "
                   "<code>docs/experiments/20261011-servo_plant_policies/</code>.</figcaption></figure>")
    side = "media/20261012_pads_skin_s0_side.mp4"
    if os.path.exists(os.path.join(D, side)):
        pr = next((r for r in X.finals("11", "tpu27pads1") if r["seed"] == 0), None)
        sr = next((r for r in X.finals("11", "tpu27skin") if r["seed"] == 0), None)
        FIG[0] += 1
        out.append(f'<figure class="replays"><div class="cell">{video(side)}</div><figcaption>Figure&#160;{FIG[0]}. Seed 0 of '
                   f"the pad policy (left) and of the skin policy (right), the films of Figures&#160;1 and {FIG[0] - 1} side by side "
                   f"at real speed. Both lift the handle end with the middle finger and pivot the tool about the thumb and index "
                   f"within a quarter second of the onset; the pads hold it at a held cosine of {pr['held_cos_mean']:.2f} with "
                   f"{pr['grip_N']:.0f}&#8202;N and the skin at {sr['held_cos_mean']:.2f} with {sr['grip_N']:.0f}&#8202;N. "
                   f"<code>{side}</code></figcaption></figure>")
    for f_role, r in zip(("best", "median"), rk):
        if f_role != "median":
            continue
        rel = f"media/{r['tag']}_replay_three.mp4"
        if os.path.exists(os.path.join(D, rel)):
            FIG[0] += 1
            out.append(f'<figure class="replays"><div class="cell">{video(rel)}<p class="filmlab">{swatch(r["arm"])}'
                       f'<b>{LBL[r["arm"]]}</b>, seed {r["seed"]}, world 0. {replay_outcome(X, r["tag"])} <code>{rel}</code>'
                       f'</p></div><figcaption>Figure&#160;{FIG[0]}. The median skin seed, world 0, as in Figure&#160;'
                       f'{FIGNO["replays"]}: the MuJoCo-Warp rollout (left) and its finger targets replayed open loop in Drake '
                       "(centre) and Newton (right), where the TPU block is a hydroelastic tip without the skin.</figcaption>"
                       "</figure>")
    return "".join(out)


def skin_mech_table(X: Data):
    head = ["Policy", "Grip<br>(N)", "Slide over the fingertip in the hold,<br>thumb &#183; index &#183; middle (&#181;m/s)",
            "Spring deflection,<br>slides (&#181;m)", "Hinges<br>(mrad)", "Spring force over<br>friction limit",
            "Deflection change<br>over the hold (&#181;m)"]
    body = []
    trip = lambda v, nd=0: " &#183; ".join(f1(x, nd) for x in v) if v else "&#8211;"  # noqa: E731
    for s, label in (("11", PLANT_CAP["11"]), ("08", PLANT_CAP["08"])):
        rs = [r for r in X.finals(s) if r["arm"] in ("tpu27mesh", "tpu27pads1", "tpu27skin") and X.creep.get(r["tag"])]
        if not rs:
            continue
        body.append(label)
        for r in rs:
            c = X.creep[r["tag"]]
            sk = r["arm"] == "tpu27skin"
            body.append([f"{swatch(r['arm'], s)}{CAP[r['arm']]} s{r['seed']}", f1(sum(c["grip_by_finger_N"]), 1),
                         trip(c.get("slide_hold_net_um_s")),
                         trip(c.get("skin_slide_um_hold_median")) if sk else "&#8211;",
                         trip(c.get("skin_hinge_mrad_hold_median")) if sk else "&#8211;",
                         trip(c.get("skin_presliding_used_hold_median"), 2) if sk else "&#8211;",
                         trip(c.get("skin_slide_hold_change_um")) if sk else "&#8211;"])
    return table(head, body, "The tool&#8217;s motion against each fingertip in the hold (steps 150&#8211;250), from a "
                             "second evaluation of each final policy with the slip record (64 rollouts; medians over the held "
                             "rollouts). Slide: net tangential displacement of the tool&#8217;s material point at the contact "
                             "centroid against the body that carries the pads or the mesh, per second. Spring deflection: the "
                             "skin&#8217;s slides and hinge against their springs, median over the hold; spring force over "
                             "friction limit: \\(k_t|q|/\\mu N\\); deflection change: the slides&#8217; displacement between steps "
                             "150 and 250. The working-plant skin&#8217;s springs are 65&#8202;kN/m, sized at 12&#8202;N per "
                             "fingertip.")


def _sg(v, nd=2):
    """A signed number with a typographic minus."""
    return f"+{v:.{nd}f}" if v >= 0.5 * 10 ** -nd else (f"{0:.{nd}f}" if v > -0.5 * 10 ** -nd else f"&#8722;{-v:.{nd}f}")


def fmt_iv(d):
    """The 95 % interval of a boot_delta result as text."""
    s = lambda v: (f"+{v:.2f}" if v >= 0.005 else ("0.00" if v > -0.005 else f"&#8722;{-v:.2f}"))  # noqa: E731
    return f"{s(d[1])} to {s(d[2])}"


def _two(v, nd=2):
    return " and ".join(f"{x:.{nd}f}" for x in v)


def skin_numbers(X: Data) -> dict:
    """The quantities the skin section, the lede and the topic overviews quote, computed once from the rows."""
    if _CMP.get("num") is not None:
        return _CMP["num"]
    f = {a: X.finals("11", a) for a in SKIN_ARMS}
    cmp_ = [r for r in skin_compare(X) if r["ok"]]
    n = dict(rows=cmp_, f=f)
    n["hc"] = {a: [r["held_cos_mean"] for r in f[a]] for a in SKIN_ARMS}
    n["grip"] = {a: [r["grip_N"] for r in f[a]] for a in SKIN_ARMS}
    n["held"] = {a: (sum(r["n_held"] for r in f[a]), sum(r["n"] for r in f[a])) for a in SKIN_ARMS}
    for b in ("tpu27pads1", "tpu27mesh"):
        n[f"bh_{b}"] = [r for r in cmp_ if beyond(r, b, "held")]
        n[f"bc_{b}"] = [r for r in cmp_ if beyond(r, b, "cos")]
        n[f"fp_{b}"] = [r for r in cmp_ if r["p"][b] < 0.05]
        n[f"bi_{b}"] = [r for r in cmp_ if r["d"][b] and (r["d"][b][1] > 0 or r["d"][b][2] < 0)]
    nom = next((r for r in cmp_ if r["key"] == "nominal"), None)
    n["nom"] = nom
    rp = [r for r in cmp_ if r["key"].startswith("rp:")]
    n["rp"] = {a: (sum(r["pooled"][a][0] for r in rp), sum(r["pooled"][a][1] for r in rp)) for a in SKIN_ARMS}
    n["rp_p"] = {b: fisher_p(n["rp"]["tpu27skin"], n["rp"][b]) for b in ("tpu27pads1", "tpu27mesh")} if rp else {}
    pt = [r for r in cmp_ if r["key"] in ROBUST_KEYS]
    n["pt"] = {a: (sum(r["pooled"][a][0] for r in pt), sum(r["pooled"][a][1] for r in pt)) for a in SKIN_ARMS}
    n["pt_p"] = {b: fisher_p(n["pt"]["tpu27skin"], n["pt"][b]) for b in ("tpu27pads1", "tpu27mesh")} if pt else {}
    n["n_cond"] = len(cmp_)
    # drops under jitter and the ten perturbations, per seed
    n["pt_seed"] = {a: [sum(v["n"] - v["h"] for r in pt for s_, v in r["seeds"][a] if s_ == sd)
                        for sd in sorted({s_ for r in pt for s_, _ in r["seeds"][a]})] for a in SKIN_ARMS}
    # the slip record of the skin finals: spring deflection and slide in the hold (creep.jsonl)
    cs = [X.creep[r["tag"]] for r in f["tpu27skin"] if r["tag"] in X.creep]
    flat = lambda k: [v for c in cs for v in (c.get(k) or []) if v is not None]  # noqa: E731
    n["sk_defl"] = flat("skin_slide_um_hold_median")
    n["sk_hinge"] = flat("skin_hinge_mrad_hold_median")
    n["sk_use"] = flat("skin_presliding_used_hold_median")
    n["sk_change"] = flat("skin_slide_hold_change_um")
    n["sk_turn_max"] = flat("skin_slide_um_turn_max")
    n["sk_hinge_turn_max"] = flat("skin_hinge_mrad_turn_max")
    n["sk_slide"] = flat("slide_hold_net_um_s")
    n["sk_slide_um"] = flat("slide_hold_net_um")
    n["sk_ratio"] = [c["slide_hold_net_um"][k] / c["skin_slide_hold_change_um"][k] for c in cs for k in range(3)
                     if c.get("slide_hold_net_um") and c.get("skin_slide_hold_change_um")
                     and c["skin_slide_hold_change_um"][k]]
    other = [X.creep[r["tag"]] for a in ("tpu27mesh", "tpu27pads1") for r in f[a] if r["tag"] in X.creep]
    n["ot_slide"] = [v for c in other for v in c.get("slide_hold_net_um_s") or []]
    w = [X.creep[r["tag"]] for r in X.finals("08", "tpu27skin") if r["tag"] in X.creep]
    n["w_defl"] = [v for c in w for v in c.get("skin_slide_um_hold_median") or []]
    n["w_slide"] = [v for c in w for v in c.get("slide_hold_net_um_s") or []]
    n["w_change"] = [v for c in w for v in c.get("skin_slide_hold_change_um") or []]
    _CMP["num"] = n
    return n


def skin_answer(X: Data) -> dict:
    """The skin result in the order the owner asked for on 2026-10-12 (every lede and summary that reports it: this page,
    the sprung-skin page, the contact and RL overviews): 'verdict', whether the skin changed the held share, the replays or
    the held cosine beyond the seed spread, and 'detail', the pooled counts behind it; 'impratio', that it trained at
    impratio 10 without its presliding; 'test', the test that remains; 'short', all three in one sentence."""
    n = skin_numbers(X)
    if n["nom"] is None:
        return {}
    hc, rp, pt = n["hc"], n["rp"], n["pt"]
    d = n["nom"]["d"]["tpu27pads1"]
    bh = n["bh_tpu27pads1"]
    bcos = n["bc_tpu27pads1"]
    nc = len(n["rows"])
    low = [r for r in bcos if max(seed_vals(r, "tpu27skin", "cos")) < min(seed_vals(r, "tpu27pads1", "cos"))]
    held_part = ("keep the tool as often as the pads, within the pads&#8217; seed spread in every condition, the replays included" if not bh
                 else "keep the tool as often as the pads except " + " and ".join(PROSE[r["key"]] for r in bh)
                 + ", where both skin seeds drop more than both pad seeds")
    cos_part = (f"their held cosine, {_two(hc['tpu27skin'])} against the pads&#8217; {_two(hc['tpu27pads1'])}, lies within "
                f"the pads&#8217; seed spread in {nc - len(bcos)} of {nc} evaluation conditions"
                + (f" and below it in {WORDS.get(len(low), len(low))} ({'; '.join(PROSE[r['key']] for r in low)})" if low else ""))
    verdict = f"Two seeds trained on the sprung skin {held_part}, and {cos_part}."
    dr = n["pt_seed"]["tpu27skin"]
    detail = (f"Pooled over seeds, the skin drops {pt['tpu27skin'][1] - pt['tpu27skin'][0]} of {pt['tpu27skin'][1]:,} "
              f"rollouts under grasp jitter and the ten perturbations against the pads&#8217; "
              f"{pt['tpu27pads1'][1] - pt['tpu27pads1'][0]} (Fisher p = {fmt_p(n['pt_p']['tpu27pads1'])}), "
              f"{max(dr)} of them from its seed {dr.index(max(dr))}, and its finger targets keep the tool in "
              f"{rp['tpu27skin'][0]} of {rp['tpu27skin'][1]} open-loop replays against the pads&#8217; "
              f"{rp['tpu27pads1'][0]} of {rp['tpu27pads1'][1]}; the mean held cosine of the two skin seeds is "
              f"{abs(d[0]):.2f} {'below' if d[0] < 0 else 'above'} the pads&#8217; (95&#8202;% interval {fmt_iv(d)}).")
    sl = n["sk_slide_um"]
    impr = (f"They trained at impratio 10, where in the hold the tool slides {min(sl) / 1e3:.1f}&#8211;{max(sl) / 1e3:.1f}&#8202;mm "
            f"over the spheres while the springs&#8217; deflection changes by {min(n['sk_change']):.0f}&#8211;"
            f"{max(n['sk_change']):.0f}&#8202;&#181;m, so they used the skin&#8217;s compliance and armature without its presliding.")
    test = ("The presliding remains to be tested by a skin policy trained where its friction rows hold, at a higher impratio "
            "with more solver iterations or a 1&#8202;ms step.")
    short = (f"Two skin seeds trained on the servo-limited plant change no held count beyond the pads&#8217; seed spread in "
             f"{nc} evaluation conditions, replays and perturbations included"
             f"{'' if not bh else ' (except ' + ' and '.join(PROSE[r['key']] for r in bh) + ')'}, and end at held cosines of "
             f"{_two(hc['tpu27skin'])} against the pads&#8217; {_two(hc['tpu27pads1'])}, inside that spread in "
             f"{nc - len(bcos)} of the {nc}; they too trained at impratio 10 without the skin&#8217;s presliding, which remains "
             "to be tested with friction rows that hold.")
    lede = ((f"Pads on a sprung skin change no held count beyond the bare pads&#8217; seed spread in {nc} evaluation "
             "conditions, replays and perturbations included, " if not bh else
             "Pads on a sprung skin change the held count beyond the bare pads&#8217; seed spread only "
             + " and ".join(PROSE[r["key"]] for r in bh) + ", ")
            + f"and end at held cosines of {_two(hc['tpu27skin'])} against {_two(hc['tpu27pads1'])}, inside that spread in "
            f"{nc - len(bcos)} of the {nc}. The skin trained at impratio 10, where the tool slides "
            f"{min(sl) / 1e3:.1f}&#8211;{max(sl) / 1e3:.1f}&#8202;mm over the spheres during the hold while the springs&#8217; "
            f"deflection changes by {min(n['sk_change']):.0f}&#8211;{max(n['sk_change']):.0f}&#8202;&#181;m, so its presliding "
            "remains to be tested with friction rows that hold.")
    return dict(verdict=verdict, detail=detail, impratio=impr, test=test, short=short, lede=lede, n=n)


def skin_text(X: Data) -> str:
    a = skin_answer(X)
    if not a:
        return P.pending("The skin finals are not evaluated yet.")
    n = a["n"]
    rows_, nc = n["rows"], len(n["rows"])
    mean = lambda r, arm: float(np.mean(seed_vals(r, arm, "cos")))  # noqa: E731
    lower_p = [r for r in rows_ if mean(r, "tpu27skin") < mean(r, "tpu27pads1")]
    low_p = [r for r in n["bc_tpu27pads1"] if max(seed_vals(r, "tpu27skin", "cos")) < min(seed_vals(r, "tpu27pads1", "cos"))]
    low_m = [r for r in n["bc_tpu27mesh"] if max(seed_vals(r, "tpu27skin", "cos")) < min(seed_vals(r, "tpu27mesh", "cos"))]
    hm = [r for r in rows_ if beyond(r, "tpu27mesh", "held") and min(seed_vals(r, "tpu27skin", "held"))
          > max(seed_vals(r, "tpu27mesh", "held"))]
    pt, dr = n["pt"], n["pt_seed"]["tpu27skin"]
    worst = dr.index(max(dr))
    rob = [r for r in rows_ if r["key"] in ROBUST_KEYS]
    by = sorted(((v["n"] - v["h"], r) for r in rob for sd, v in r["seeds"]["tpu27skin"] if sd == worst), key=lambda x: -x[0])
    big = [r for k, r in by if k >= 5]
    d = n["nom"]["d"]["tpu27pads1"]
    held_ok = not n["bh_tpu27pads1"]
    p1 = ((f"<p>The skin&#8217;s held count stays within the pads&#8217; seed spread in all {nc} conditions" if held_ok else
           f"<p>The skin&#8217;s held count leaves the pads&#8217; seed spread {' and '.join(PROSE[r['key']] for r in n['bh_tpu27pads1'])}")
          + f" (Table&#160;{TABNO['skin']}, Figure&#160;{FIGNO['skin_cmp']}). Fisher&#8217;s test on the counts pooled over seeds "
          f"separates the skin from the pads under the perturbations, where the skin drops {pt['tpu27skin'][1] - pt['tpu27skin'][0]} "
          f"of {pt['tpu27skin'][1]:,} rollouts and the pads {pt['tpu27pads1'][1] - pt['tpu27pads1'][0]} (p = "
          f"{fmt_p(n['pt_p']['tpu27pads1'])}); {max(dr)} of the skin&#8217;s drops come from its seed {worst}"
          + (f", {' and '.join(PROSE[r['key']] for r in big)}" if big else "") + f", and seed {1 - worst} drops {min(dr)}. "
          f"The skin&#8217;s held cosine lies below the pads&#8217; in {len(lower_p)} of {nc} conditions, by {abs(d[0]):.2f} at "
          f"nominal (95&#8202;% interval {fmt_iv(d)})"
          + (f", and below both pad seeds {' and '.join(PROSE[r['key']] for r in low_p)}" if low_p else "")
          + f" (Table&#160;{TABNO['skin_cos']}). It lies below both TPU-mesh seeds in {len(low_m)} of {nc} conditions"
          + (f"; the TPU-mesh policies hold fewer rollouts than both skin seeds {' and '.join(PROSE[r['key']] for r in hm)}"
             if hm else "") + ".</p>")
    p2 = (f"<p>{a['impratio'].replace('They trained', 'The skin policies trained', 1)} {a['test']} The contact overview "
          "lists that run as future work "
          f"(<a href=\"{CONTACT_OV}#open\">the skin at the trainer&#8217;s settings</a>); the skin&#8217;s construction and its "
          f"presliding on the bed are on the <a href=\"{SKIN_PAGE}\">sprung-skin page</a>.</p>")
    f = {r["seed"]: r for r in X.finals("11", "tpu27skin")}
    tt = {sd: turn_time(X.trace(r["tag"])[0]) for sd, r in f.items() if X.trace(r["tag"])}
    p3 = ""
    if tt and "skin_films" in FIGNO and 0 in f and 1 in f:
        yaw = {sd: X.m(r["tag"])["range_limit_by_joint"]["middle_yaw"] for sd, r in f.items() if X.m(r["tag"])}
        p3 = (f"<p>Both skin seeds turn the tool the way the pad policies do and reach 90&#8202;% of their final cosine "
              f"{min(tt.values()):.2f}&#8211;{max(tt.values()):.2f}&#8202;s after the onset (Figures&#160;{FIGNO['skin_films']} and "
              f"{FIGNO.get('skin_side', FIGNO['skin_films'])}). Seed 1 presses the tool with its index at "
              f"{f[1]['grip_index_N']:.1f}&#8202;N, against {f[0]['grip_index_N']:.1f}&#8202;N in seed 0, keeps the middle "
              f"finger&#8217;s yaw off its range limit ({100 * yaw.get(1, 0):.0f}&#8202;% of the held steps at the limit, "
              f"{100 * yaw.get(0, 0):.0f}&#8202;% in seed 0) and leaves the tool "
              f"{np.degrees(np.arccos(f[1]['held_cos_mean'])):.0f}&#176; from vertical.</p>")
    return p1 + p2 + p3


def skin_mech_text(X: Data) -> str:
    n = skin_numbers(X)
    if not n["sk_defl"]:
        return ""
    return (f"<p>Table&#160;{TABNO['skin_mech']} splits the tool&#8217;s motion against each fingertip in the hold. On the skin "
            f"the springs stand deflected by {min(n['sk_defl']):.0f}&#8211;{max(n['sk_defl']):.0f}&#8202;&#181;m and "
            f"{min(n['sk_hinge']):.0f}&#8211;{max(n['sk_hinge']):.0f}&#8202;mrad, carrying {100 * min(n['sk_use']):.0f}&#8211;"
            f"{100 * max(n['sk_use']):.0f}&#8202;% of the friction limit, and their deflection changes by "
            f"{min(n['sk_change']):.0f}&#8211;{max(n['sk_change']):.0f}&#8202;&#181;m over the 2&#8202;s, while the tool slides "
            f"{min(n['sk_slide_um']) / 1e3:.1f}&#8211;{max(n['sk_slide_um']) / 1e3:.1f}&#8202;mm over the spheres: the slide moves "
            f"the tool {min(n['sk_ratio']):.0f}&#8211;{max(n['sk_ratio']):.0f} times as far as the springs. The pad and TPU-mesh "
            f"policies slide the tool over their fingertips at {min(n['ot_slide']):,.0f}&#8211;{max(n['ot_slide']):,.0f}&#8202;"
            f"&#181;m/s, the skin policies at {min(n['sk_slide']):,.0f}&#8211;{max(n['sk_slide']):,.0f}. During the turn the "
            f"springs reach {max(n['sk_turn_max']):.0f}&#8202;&#181;m and {max(n['sk_hinge_turn_max']):.0f}&#8202;mrad. On the "
            f"working plant, at 44&#8211;56&#8202;N, the skin&#8217;s springs stood at {min(n['w_defl']):.0f}&#8211;"
            f"{max(n['w_defl']):.0f}&#8202;&#181;m, changed by {min(n['w_change']):.0f}&#8211;{max(n['w_change']):.0f}&#8202;&#181;m "
            f"over the hold, and the tool slid at {min(n['w_slide']):,.0f}&#8211;{max(n['w_slide']):,.0f}&#8202;&#181;m/s.</p>")


def skin_section(X: Data) -> str:
    if not X.finals("11", "tpu27skin"):
        return P.pending("Skin final policies not evaluated yet (final_eval.jsonl).")
    t1, t2 = skin_tables(X)
    out = skin_text(X) + t1 + t2
    out += figure(svg_skin_compare(X),
                  "The skin against the pads and the TPU mesh under every evaluation of Tables&#160;"
                  f"{TABNO['skin']} and {TABNO['skin_cos']}: held share (left) and held cosine (right) of each seed, seed 0 a "
                  "filled dot and seed 1 a ring, the line between them the seed spread. A skin line that overlaps the pads&#8217; "
                  "line is within their seed spread. Hover a dot for its counts and the 95&#8202;% interval of its held cosine.")
    out += skin_films(X)
    out += skin_mech_text(X) + skin_mech_table(X)
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
            if r["arm"] not in SKIN_ARMS:
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
              {"key": "arm", "label": "Contact model", "values": [("tpu27mesh", "TPU mesh"), ("tpu27pads1", "Pads"),
                                                                    ("tpu27skin", "Skin")], "on": ["tpu27mesh", "tpu27pads1"]}]
    fig = IF.Figure("tb-curves", None, "Environment steps (millions)", panels, series, facets=facets, legend=legend,
                    x_domain=(0.0, 40.0), x_dec=1)
    return ifig(fig, "Training curves from the event files of the runs with a final policy, smoothed over nine iterations: "
                     "the stochastic policy&#8217;s mean episode return, its alignment term, the grip term (the working plant&#8217;s "
                     "bonus is positive, the servo plant&#8217;s penalty negative) and the action noise; solid the servo-limited "
                     "plant, dashed the working plant. The default view shows the TPU-mesh and pad runs; the box adds the skin "
                     "runs.")


def curves_text(X: Data):
    last = tb_last(X)
    ns = lambda s, a: [q.get("Policy/mean_std") for q in last.values() if q["study"] == s and q["arm"] == a]  # noqa: E731
    return (f"<p>The servo-plant runs end training with an action noise of {rng(ns('11', 'tpu27mesh'), 2)} (TPU mesh), "
            f"{rng(ns('11', 'tpu27pads1'), 2)} (pads) and {rng(ns('11', 'tpu27skin'), 2)} (skin), against "
            f"{rng(ns('08', 'tpu27mesh'), 2)}, {rng(ns('08', 'tpu27pads1'), 2)} and {rng(ns('08', 'tpu27skin'), 2)} on the "
            "working plant; every run starts at 0.30.</p>")


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
    """About 130 words: the outcome on the servo-limited plant and the pad brake, then the skin in the owner's order
    (whether it changed anything beyond the seed spread, that it trained at impratio 10 without its presliding, the test
    left; skin_answer)."""
    f11 = [r for r in X.finals("11") if r["arm"] in ("tpu27mesh", "tpu27pads1")]
    if len(f11) < 4:
        return f"Evaluation in progress: {len(f11)} of 4 final policies evaluated."
    hc = lambda s, a: [r["held_cos_mean"] for r in X.finals(s, a)]  # noqa: E731
    grip = lambda rs: [r["grip_N"] for r in rs]  # noqa: E731
    tps = [r for r in X.finals("08") if r["arm"] in ("tpu27mesh", "tpu27pads1")]
    sw = [(r, X.m(r["tag"]), X.m(r["tag"], "fr=1")) for r in f11]
    stop = [x[0] for x in sw if x[1] and x[2] and x[0]["arm"] == "tpu27pads1" and x[2]["n_held"] == x[2]["n"]
            and x[2]["held_cos_mean"] < 0.5]
    p = [f"With the finger torque limited to the SCS0009 servo&#8217;s 0.18&#8202;N&#8202;m and a penalty on fingertip force, "
         f"the D6 policies with TPU-mesh and 1&#8202;mm pad fingertips hold the screwdriver in "
         f"{sum(r['n_held'] for r in f11)} of {sum(r['n'] for r in f11)} rollouts at {rng(grip(f11), 0)}&#8202;N "
         f"({rng(grip(tps), 0)}&#8202;N at 1&#8202;N&#8202;m), and the pads turn it nearly as far as the TPU mesh (held cosine "
         f"{rng(hc('11', 'tpu27pads1'), 2)} against {rng(hc('11', 'tpu27mesh'), 2)})"
         + (f"; at 1&#8202;N&#8202;m pad seed {stop[0]['seed']} stops turning it." if stop else ".")]
    a = skin_answer(X)
    if a:
        p.append(a["lede"])
    return " ".join(p)


def _fr_held(X: Data):
    """Held counts of the pad and skin finals at torque limits 0.12 and 0.23 N m (robust rows)."""
    v = [X.rob[(r["tag"], k)] for a in ("tpu27pads1", "tpu27skin") for r in X.finals("11", a) for k in ("fr=0.12", "fr=0.23")
         if (r["tag"], k) in X.rob]
    if not v:
        return "the tool"
    return f"{min(q['n_held'] for q in v)}&#8211;{max(q['n_held'] for q in v)} of 128 rollouts"


def _slide_mm(X: Data):
    n = skin_numbers(X)
    v = n["sk_slide_um"]
    return f"{min(v) / 1e3:.1f}&#8211;{max(v) / 1e3:.1f}&#8202;mm" if v else "&#8211;"


def _change_um(X: Data):
    n = skin_numbers(X)
    v = n["sk_change"]
    return f"{min(v):.0f}&#8211;{max(v):.0f}&#8202;&#181;m" if v else "&#8211;"


def open_items(X: Data):
    tbl = tb_last(X)
    gt11 = [q["Episode_Reward/grip_force_excess"] for q in tbl.values() if q["study"] == "11"]
    al11 = [q["Episode_Reward/target_axis_alignment"] for q in tbl.values() if q["study"] == "11"]
    th11 = [r["grip_thumb_N"] for r in X.finals("11")]
    wh = {a: [X.cost["11"][r["tag"]]["wall_h"] for r in X.finals("11", a) if r["tag"] in X.cost["11"]] for a in ("tpu27mesh", "tpu27pads1")}
    neg = lambda v: f"{-max(v):.2f}&#8211;{-min(v):.2f}"  # noqa: E731
    items = [
        ("Grip penalty weight.", f"The &#8722;2 penalty costs {neg(gt11)} per episode against an alignment term of "
         f"{rng(al11, 0)}, and the thumb presses {rng(th11, 1)}&#8202;N, above the 4&#8202;N threshold, with its mcp at the "
         "torque limit in every held step. Retrain one TPU-mesh and one pad seed with "
         "<code>--grip-force-penalty-weight -20</code> in <code>scripts/rl_contact_train_queue_20261011.sh</code> "
         f"({rng(wh['tpu27mesh'], 1)} and {rng(wh['tpu27pads1'], 1)}&#8202;GPU-h) and evaluate them with "
         "<code>rl_contact_eval.py final --study 20261011</code>. A policy that holds 64 of 64 with the thumb under 4&#8202;N "
         "and a held cosine within 0.05 of these shows that the turn does not need the saturated thumb."),
        ("Stall torque on the bench.", "0.18&#8202;N&#8202;m is 80&#8202;% of the datasheet stall torque, and the hand&#8217;s "
         "servos have not been measured at stall. Between 0.12 and 0.23&#8202;N&#8202;m every pad and skin final holds "
         f"{_fr_held(X)}; at 1&#8202;N&#8202;m pad seed 0 stops turning the tool. Measure the stall torque of the thumb mcp "
         "servo against a load cell at 6&#8202;V (the push test of the bench plan); a stall above 0.3&#8202;N&#8202;m puts the "
         "hand between the plants of this page, where the torque-limit swap has to be repeated at the measured value."),
        ("Bench replay of the finger targets.", "Play the recorded targets of the median rollout of each servo-plant final "
         "policy on the hand at 50&#8202;Hz from the lifted grasp (<code>logs/20261011-servo_plant_policies/replay/"
         "&lt;tag&gt;/rec.npz</code> exported through the real_v1 deploy format, after "
         "<code>scripts/real_v1_trajectory_clearance.py</code>; 10 trials each, tracked). Every policy drives the middle "
         "finger to its yaw range limit; a plan the clearance gate rejects for the index&#8211;middle pair ends the item "
         "for that policy. Targets that drop the tool on the hand in most trials, where the three simulators keep it, put "
         "the gap between simulation and hand in the plant or the fingertip model."),
        ("The skin with stiff friction rows.", "The skin policies trained at impratio 10, where the tool slides "
         f"{_slide_mm(X)} over the spheres during the hold while the springs&#8217; deflection changes by "
         f"{_change_um(X)}. Rerun the scripted-grasp check of <code>scripts/make_work_plant_runs.py --plant "
         "kp0.72_kv0_fr0.18_fl0_dp0.0144 --skin</code> at impratio 100, 300 and 1000 with 20 solver iterations or a 1&#8202;ms "
         "step, train two skin seeds at the first setting that holds 64 of 64, and evaluate them with <code>rl_contact_eval.py "
         "final creep robust --study 20261011</code>. Springs that carry most of the tool&#8217;s motion against the fingertips "
         "in the hold, with the held cosine, jittered held share and replay counts within the pads&#8217; seed spread, retire "
         f"the skin (<a href=\"{CONTACT_OV}#open\">contact overview</a>, future work)."),
    ]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def byline():
    return ("D6 (<code>results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip*</code>, scenes by "
            "<code>scripts/make_work_plant_runs.py --plant kp0.72_kv0_fr0.18_fl0_dp0.0144</code>) &#183; PPO, 2,048 envs, "
            "40&#8202;M steps, seeds 0&#8211;1 (<code>scripts/rl_contact_train_queue_20261011.sh</code>, runs "
            "<code>results/rl/20261011-d6_servo_*_40M_s*</code>; skin runs <code>results/rl/20261012-d6_servo_tpu27skin_40M_s*"
            "</code>) &#183; evaluation <code>scripts/rl_contact_eval.py --study 20261011</code>, replays "
            "<code>scripts/rl_policy_replay.py --study 20261011</code> &#183; rows "
            "<code>docs/experiments/20261011-servo_plant_policies/</code> &#183; builder "
            "<code>scripts/servo_plant_policies_page.py</code> &#183; hand-offs <code>" + NOTE + "</code> and <code>"
            + NOTE_SKIN + "</code> &#183; the sprung skin: "
            f"<a href=\"{SKIN_PAGE}\"><code>{SKIN_PAGE_PATH}</code></a> &#183; the study "
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
    rk = ranked(X, "11", "tpu27skin")
    skin = bool(rk)
    skin_figs = (["skin_cmp"] if skin else []) + (
        ["skin_films"] if any(os.path.exists(os.path.join(D, f"media/{r['tag']}_{role}.mp4"))
                              for role, r in zip(("best", "median"), rk)) else []) + (
        ["skin_side"] if os.path.exists(os.path.join(D, "media/20261012_pads_skin_s0_side.mp4")) else []) + (
        ["skin_replay"] if len(rk) > 1 and os.path.exists(os.path.join(D, f"media/{rk[1]['tag']}_replay_three.mp4")) else [])
    for k, name in enumerate(["films", "traces", "mech", "brake"] + (["replays"] if replay_films else []) + skin_figs
                             + ["curves"]):
        FIGNO[name] = k + 1
    for k, name in enumerate(["plant", "cost", "final", "mech", "swap", "transfer"] + (["skin", "skin_cos", "skin_mech"] if skin else [])):
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
    v["SKIN"] = skin_section(X)
    v["CURVES"] = curves_section(X)
    v["OPEN"] = open_items(X)
    v["FOOTER"] = ("<p>Rebuild: <code>logs/20261001-hom_contact/venv/bin/python scripts/servo_plant_policies_page.py</code>. "
                   "Rows: <code>docs/experiments/20261011-servo_plant_policies/*.jsonl</code>; per-step traces "
                   "<code>final_traces/</code> (servo-plant policies), <code>mech_traces/</code> (working-plant policies and "
                   "the torque-limit swap) and <code>creep_traces/</code> (the slip record of both studies&#8217; finals); films "
                   "<code>media/</code>. Evaluation scripts of the skin arm: <code>logs/20261011-servo_plant_policies/skin_*.sh"
                   "</code>.</p>")
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
