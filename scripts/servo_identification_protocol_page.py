#!/usr/bin/env python3
"""Page: identification protocol for the SR2 hand's serial-bus servos (draft, 2026-10-13).

Owner, 2026-10-13: "not sure i would rely too much on the servo plant sysid. our past attempts have been fairly ad-hoc. it may
be beneficial to write up a more thorough protocol that we can walk through together (i setup and measure as needed),
leveraging the better actuator models paper as needed (https://arxiv.org/abs/2410.08650v1). we may update our servos to the
feetech sts3032, which are not modeled in any case, by any project".

The page is a plan: no measurement yet. Its facts about the paper, the STS3032 and the past fits are collected in
FACTS below with their sources.

    .venv/bin/python scripts/servo_identification_protocol_page.py
"""
from __future__ import annotations

import math
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import contact_overview_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20261013-servo_identification")
OUT = os.path.join(D, "20261013-servo_identification_protocol.html")
TPL = os.path.join(ROOT, "scripts/servo_identification_protocol_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
BAM = "https://arxiv.org/abs/2410.08650v1"
DEFORM = "../20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html"
FIG, TAB, REFS = [0], [0], {}
EXTRA_CSS = """
.tw th{white-space:normal !important;vertical-align:bottom}
.tw td:first-child{min-width:9em}
.col > figure,.col > .tw{width:auto !important;max-width:100% !important}
ol.steps > li{margin-bottom:10px}
"""


def figure(body, caption, legend=None, key=None):
    FIG[0] += 1
    if key:
        REFS[f"fig:{key}"] = FIG[0]
    lg = P._legend_html(legend) if legend else ""
    return f'<figure class="diagram">{body}{lg}<figcaption>Figure&#160;{FIG[0]}. {caption}</figcaption></figure>'


def table(head, body, caption, text_cols=(0,), key=None, note=None):
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


# ------------------------------------------------------------------------------------------ facts (sources in the strings)

FACTS = dict(
    plant_kp=0.72, plant_tau=0.18, scs_stall=0.23,
    tip_compliance_mm_per_N=(3.4, 4.7), gyroid_mm_per_N=0.063, eflesh_mm_per_N=(0.38, 0.61),
)


# ------------------------------------------------------------------------------------------ figure

def svg_rig():
    """Pendulum rig (left) and the in-hand press (right)."""
    W, H = 860, 360
    out = P._svg_open(W, H, "Two rigs: a servo turning a pendulum arm with a mass at a set distance, and a finger pressing a scale "
                            "through its fingertip")
    out.append('<text x="40" y="26" style="fill:var(--ink)">(a) Pendulum, one servo on the bench</text>')
    # servo body and horn
    cx, cy = 200.0, 120.0
    out.append(f'<rect x="{cx - 46}" y="{cy - 30}" width="92" height="60" rx="4" style="fill:#2b2f33"/>'
               f'<circle cx="{cx}" cy="{cy}" r="9" style="fill:#d0d4d8;stroke:#000"/>'
               f'<text x="{cx}" y="{cy - 40}" text-anchor="middle" style="fill:var(--ink)">Servo clamped to the bench</text>')
    for ang, op in ((math.radians(60), 1.0), (math.radians(20), 0.35)):
        L = 150.0
        x2, y2 = cx + L * math.sin(ang), cy + L * math.cos(ang)
        out.append(f'<g style="opacity:{op}"><line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:#3a5fb2;stroke-width:6"/>'
                   f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="13" style="fill:#b23a3a"/></g>')
    ang = math.radians(60)
    x2, y2 = cx + 150 * math.sin(ang), cy + 150 * math.cos(ang)
    out.append(f'<text x="{x2 + 18:.1f}" y="{y2 + 5:.1f}" style="fill:var(--ink)">Mass m at length l</text>'
               f'<path d="M{cx},{cy + 60} A60,60 0 0 0 {cx + 60 * math.sin(ang):.1f},{cy + 60 * math.cos(ang):.1f}" '
               'style="fill:none;stroke:var(--ink2)"/>'
               f'<text x="{cx + 8}" y="{cy + 84}" style="fill:var(--ink2)">&#952;</text>'
               f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy + 190}" style="stroke:var(--ink3);stroke-dasharray:3 3"/>'
               f'<text x="{cx - 8}" y="{cy + 180}" text-anchor="end" style="fill:var(--ink3)">Gravity</text>'
               f'<text x="40" y="352" style="fill:var(--ink2)">Bus log: position, speed, load, voltage, temperature</text>')
    # (b) finger on a scale
    ox = 500.0
    out.append(f'<text x="{ox}" y="26" style="fill:var(--ink)">(b) One finger of the hand pressing a scale</text>')
    out.append(f'<rect x="{ox + 10}" y="70" width="70" height="44" rx="3" style="fill:#555"/>'
               f'<text x="{ox + 45}" y="62" text-anchor="middle" style="fill:var(--ink)">Palm</text>'
               f'<line x1="{ox + 80}" y1="92" x2="{ox + 190}" y2="150" style="stroke:#3a5fb2;stroke-width:9"/>'
               f'<line x1="{ox + 190}" y1="150" x2="{ox + 240}" y2="240" style="stroke:#3a5fb2;stroke-width:9"/>'
               f'<circle cx="{ox + 80}" cy="92" r="7" style="fill:#2b2f33"/><circle cx="{ox + 190}" cy="150" r="7" style="fill:#2b2f33"/>'
               f'<rect x="{ox + 226}" y="232" width="34" height="22" rx="4" style="fill:rgba(42,120,214,0.35);stroke:var(--ink)"/>'
               f'<rect x="{ox + 180}" y="254" width="140" height="18" style="fill:#cfe3c9;stroke:#3f7a46"/>'
               f'<rect x="{ox + 160}" y="272" width="180" height="16" style="fill:#e8e8e8;stroke:#555"/>'
               f'<text x="{ox + 250}" y="306" text-anchor="middle" style="fill:#3f7a46">Load cell on the palm plate</text>'
               f'<text x="{ox + 218}" y="226" text-anchor="end" style="fill:var(--ink)">Printed or rigid tip</text>'
               f'<text x="{ox}" y="332" style="fill:var(--ink2)">Command a press past first touch;</text>'
               f'<text x="{ox}" y="352" style="fill:var(--ink2)">read force, joint angles and load</text>')
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------------------------------ sections

GLOSSARY = [
    ("Servo", "A serial-bus hobby servo: DC motor, gearbox, output-shaft position sensor and a firmware position loop that sets "
     "the motor&#8217;s PWM duty from the position error. SCS0009 on the hand now; STS3032 a candidate."),
    ("Position-loop stiffness \\(k_\\theta\\)", "Torque per radian of steady position error below saturation, in N&#8202;m/rad: "
     "the servo seen as a torsional spring about its commanded angle. The plant uses "
     f"{FACTS['plant_kp']}&#8202;N&#8202;m/rad."),
    ("Stall torque \\(\\tau_{max}\\)", "Torque at full duty and zero speed at a stated supply voltage, in N&#8202;m."),
    ("Friction", "Torque opposing motion inside the gearbox: Coulomb (constant), viscous (proportional to speed), Stribeck (a "
     "higher breakaway torque at low speed) and load-dependent (growing with the torque the gears carry), in N&#8202;m."),
    ("Armature \\(J_a\\)", "Rotor and gear inertia reflected to the output shaft, in kg&#8202;m&#178;."),
    ("Latency", "Time from a goal sent on the bus to the first change of duty, plus the age of a reading when it arrives, in ms."),
    ("Fingertip compliance", "Displacement of the fingertip along a direction per newton applied there, in mm/N, from the "
     "joints&#8217; stiffness through the finger&#8217;s Jacobian, \\(C = J K_\\theta^{-1} J^T\\), plus the tip&#8217;s own."),
    ("Trajectory error", "RMS difference between the logged and the simulated joint angle over a held-out trajectory, in degrees."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def why_section():
    lo, hi = FACTS["tip_compliance_mm_per_N"]
    elo, ehi = FACTS["eflesh_mm_per_N"]
    return ("<p>The servo plant of every result since 2026-10-11 gives each finger joint a position gain of "
            f"{FACTS['plant_kp']}&#8202;N&#8202;m/rad and a torque limit of {FACTS['plant_tau']}&#8202;N&#8202;m. The gain comes from "
            "the load register&#8217;s slope, 0.018&#8202;degree per load unit, read as full duty at 18&#8202;degrees of error and "
            f"scaled by the seller&#8217;s {FACTS['scs_stall']}&#8202;N&#8202;m stall torque; no measurement fixed the strength, and the "
            "replay fits of 2026-10-06 scored the same joint error for every gain from 0.25 to 8. These numbers decide what the "
            f"simulator must represent. Through the finger, the plant&#8217;s gain makes the fingertip yield {lo}&#8211;{hi}&#8202;mm/N "
            f"toward the tool, against {FACTS['gyroid_mm_per_N']}&#8202;mm/N for the printed gyroid tip and {elo}&#8211;{ehi}&#8202;mm/N "
            f"for eFlesh (<a href=\"{DEFORM}\">fingertip page</a>): if the gain is right, the servo dominates the finger&#8217;s "
            "compliance and the tip&#8217;s deformation is a correction; if the servo is ten times stiffer than the plant, the tip "
            "dominates.</p>"
            "<p>The protocol identifies each servo model on one bench rig, checks it inside the hand against a load cell, and writes "
            "the result as a MuJoCo actuator. It is written for either servo, so a change to the STS3032 repeats the same steps.</p>")


def model_section():
    return ("<p>The model follows Duclusaud, Passault, Padois and Ly, <i>Extended friction models for the physics simulation of "
            f"servo actuators</i> (<a href=\"{BAM}\">arXiv 2410.08650</a>, 2024), who identify servo models from pendulum "
            "trajectories on four servo actuators and check them on 2R arms, with friction models beyond Coulomb and viscous: the "
            "firmware&#8217;s duty, an electrical motor with back-EMF, and friction that depends on speed and on the load the gears "
            "carry. For a joint at angle \\(\\theta\\) commanded to \\(\\theta^*\\), the terms this protocol fits are:</p>"
            "\\[u = \\mathrm{clip}\\left(K_p(\\theta^* - \\theta),\\, -1,\\, 1\\right)\\]"
            "\\[\\tau_m = \\frac{k_t}{R}\\left(u\\,V - k_e\\,\\dot\\theta\\right)\\]"
            "\\[\\tau_f = \\left(\\tau_c + \\mu_L|\\tau_m| + (\\tau_s - \\tau_c)\\,e^{-(\\dot\\theta/\\omega_s)^2}\\right)\\mathrm{sign}(\\dot\\theta) "
            "+ b\\,\\dot\\theta\\]"
            "\\[(J_a + J_\\ell)\\,\\ddot\\theta = \\tau_m - \\tau_f + \\tau_{ext}\\]"
            "<p>Here \\(u\\) is the duty, \\(K_p\\) the firmware&#8217;s gain in duty per radian, \\(V\\) the supply voltage, \\(k_t/R\\) "
            "and \\(k_e\\) the motor&#8217;s torque per volt and back-EMF constant at the output shaft, \\(\\tau_c\\), \\(\\tau_s\\), "
            "\\(\\omega_s\\), \\(\\mu_L\\) and \\(b\\) the friction terms, \\(J_\\ell\\) the load&#8217;s inertia and \\(\\tau_{ext}\\) "
            "gravity and contact. Below saturation the loop is a spring of stiffness \\(k_\\theta = K_p V k_t/R\\); at full duty the "
            "torque is \\(\\tau_{max} = V k_t/R\\). The fit also returns the latency and any dead band around the goal.</p>")


STEPS = [
    ("Registers and limits", "15 min", "Bus free (control station stopped). Dump every control-table register of each servo: "
     "firmware position gains, torque limit, dead band, protection thresholds (overload torque and time, temperature, voltage), "
     "return delay and baud rate. Repeat after power cycling. Output: one JSON per servo; the gains enter \\(K_p\\)."),
    ("Bus timing", "15 min", "Read position, speed, load, voltage and temperature from one servo, then from all nine in turn, for "
     "60&#8202;s each; send step goals at 2&#8202;Hz and timestamp the first reading that moves. Output: sample period, its jitter, "
     "and the goal-to-motion latency."),
    ("Static stiffness and strength, pendulum", "45 min", "Servo clamped, arm horizontal. For masses giving 10, 25, 50, 75 and "
     "100&#8202;% of the seller&#8217;s stall torque at the arm length, command the horizontal angle and hold 5&#8202;s; log the "
     "steady angle error and the load register; repeat approaching from above and below. Output: \\(k_\\theta\\), the dead band, "
     "the hysteresis between the two approaches and \\(\\tau_{max}\\) at the supply voltage; repeat at a second voltage."),
    ("Free swings, torque off", "20 min", "Torque disabled, arm released from 30, 60 and 90&#8202;degrees with two masses and two "
     "lengths, logged until it stops. Output: the gearbox&#8217;s friction and the armature without the firmware in the loop."),
    ("Driven trajectories, pendulum", "60 min", "Torque on. Steps of 10&#8211;90&#8202;degrees, sine sweeps 0.2&#8211;5&#8202;Hz, "
     "and random goal sequences at the bus rate, for three loads (no mass, middle, heavy) and two arm lengths; every run twice. "
     "Two thirds fit the model, one third is held out."),
    ("Each servo of the hand", "30 min", "Steps 3 and 5 shortened (two masses, one sine sweep) on the nine servos, mounted or as "
     "spares of the same batch: the spread between units sets the randomization range."),
    ("Finger presses, in the hand", "60 min", "Finger pressing a load cell fixed in the palm frame (Figure&#160;[[fig:rig]]b), "
     "first with a rigid tip, then the printed tip. Command a press 0&#8211;20&#8202;degrees past first touch in 1&#8202;degree "
     "steps, hold 3&#8202;s each, and log force, joint angles and load. Output: force against commanded penetration, the "
     "fingertip compliance with and without the tip, and the force at which the torque limit or the protection trips."),
    ("Thermal check", "20 min", "Hold 75&#8202;% of the stall torque for 2&#8202;min, then repeat one sine sweep; log temperature. "
     "Output: how far stiffness and friction move with temperature, and the protection trip time."),
]


def steps_section():
    rows = [[f"{i + 1}. {a}", b, c] for i, (a, b, c) in enumerate(STEPS)]
    total = sum(int(t.split()[0]) for _, t, _ in STEPS)
    tab = table(["Step", "Time", "What to do and what it gives"], rows,
                f"Bench steps, about {total / 60:.1f}&#8202;h with the owner setting up and swapping loads; the workstation logs over the "
                "bus and fits between steps.", text_cols=(0, 1, 2), key="steps")
    return ("<p>Two rigs carry the measurements (Figure&#160;[[fig:rig]]): a pendulum with one servo, which isolates the actuator, "
            "and a finger of the hand pressing a load cell, which checks the model where the policy uses it. The serial port is "
            "exclusive and telemetry pauses while a writer owns the bus, so every step runs with the control station stopped and one "
            "logger owning the bus.</p>" + tab)


def rig_section():
    return (figure(svg_rig(), "The two rigs. (a) The pendulum: a servo clamped to the bench turns an arm with a mass at a set "
                              "distance; gravity loads the joint by \\(m g l \\sin\\theta\\). (b) A finger of the hand presses a load "
                              "cell through a rigid or a printed tip.", key="rig")
            + "<p>The pendulum needs a rigid arm with holes every 20&#8202;mm, a set of masses from 10 to 200&#8202;g, a clamp and a "
            "bench supply that reads its current. The finger press needs a load cell of 0&#8211;20&#8202;N (a bar cell and an HX711 "
            "amplifier, or a kitchen scale read by camera) fixed to the palm plate, and a rigid tip of the printed tip&#8217;s shape.</p>")


def fit_section():
    return ("<p>The fit minimizes the trajectory error of simulated against logged angles over the training runs, all parameters "
            "of the model at once, for each servo type; the per-unit runs then refit only \\(k_t/R\\), \\(\\tau_c\\) and \\(K_p\\). "
            "The model is accepted if its trajectory error on the held-out runs is under half of the present plant&#8217;s on the "
            "same runs, and if the finger press predicts the measured force against penetration within 20&#8202;% over the "
            "unsaturated range.</p>"
            "<p>MuJoCo takes the model as a general actuator whose force the duty, the back-EMF and the saturation set, with the "
            "joint&#8217;s armature and damping and a friction loss for the Coulomb term; the load-dependent and Stribeck terms need a "
            "user-written actuator in MuJoCo-Warp, so the first version keeps them out and the held-out error measures what that "
            "costs.</p>")


def owner_section():
    rows = [
        ["Which servo first", "Identify the SCS0009 now, since every bench result used it, and repeat on the STS3032 when it is "
         "ordered; or wait for the STS3032 and identify only it."],
        ["Load cell", "Bar load cell with an HX711 (logged with the bus) or a kitchen scale read by camera (no wiring, slower)."],
        ["Spare servos", "Pendulum work on a spare of each type, so the hand&#8217;s nine stay mounted."],
        ["Supply", "The hand&#8217;s supply voltage during runs, and whether the bench supply can hold it at stall current."],
    ]
    return table(["Decision", "Options"], rows, "Choices that set up the session.", text_cols=(0, 1), key="owner")


def open_items():
    items = [
        ("Logger and fitter.", "<code>scripts/servo_id_log.py</code> (bus logger with step and sweep generators, one JSONL row per "
         "sample, fsynced) and <code>scripts/servo_id_fit.py</code> (model, fit, held-out error, MuJoCo actuator parameters); "
         "written before the session and dry-run on one spare servo."),
        ("Compliance budget with measured gains.", "Recompute the fingertip compliance toward the tool with the identified "
         "\\(k_\\theta\\) and compare it with the tips&#8217; 0.063 and 0.38&#8211;0.61&#8202;mm/N: the tip&#8217;s deformation needs "
         "its own model in the simulator if it exceeds a third of the finger&#8217;s total."),
        ("Policies on the identified plant.", "Retrain the servo-plant reorientation of 2026-10-11 on the identified actuator with "
         "unit-to-unit randomization, and compare its bench transfer with the present plant&#8217;s."),
    ]
    return "<ul class='open'>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def lede():
    total = sum(int(t.split()[0]) for _, t, _ in STEPS)
    return ("The servo plant behind every result since 2026-10-11 has an unmeasured strength: its 0.72&#8202;N&#8202;m/rad gain is "
            "the load register&#8217;s slope scaled by a seller&#8217;s stall torque, and the replay fits could not tell gains of 0.25 "
            "and 8 apart. That gain decides whether the fingertip&#8217;s own deformation matters: with it the finger yields "
            "3.4&#8211;4.7&#8202;mm/N, fifty times the printed gyroid tip. This protocol identifies a servo model with the extended "
            f"friction models of Duclusaud et al. (2024), on a pendulum rig as they do and then in the hand against a load cell, in about "
            f"{total / 60:.1f}&#8202;h of bench time, for the SCS0009 now and the STS3032 if the hand moves to it.")


def byline():
    return ("Draft for review with the owner &#183; builder <code>scripts/servo_identification_protocol_page.py</code> &#183; "
            f"model terms from <a href=\"{BAM}\">arXiv 2410.08650</a>")


def render_tex(t):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None) for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    return P.TEX_RE.sub(lambda m: next(svgs), t), len(items)


def main():
    os.makedirs(D, exist_ok=True)
    v = {"STYLE": P.style_block().replace("</style>", EXTRA_CSS + "</style>"), "BUILT": time.strftime("%Y-%m-%d %H:%M"),
         "LEDE": lede(), "BYLINE": byline(), "GLOSSARY": glossary(), "WHY": why_section(), "MODEL": model_section(),
         "RIG": rig_section(), "STEPS": steps_section(), "FIT": fit_section(), "OWNER": owner_section(), "OPEN": open_items(),
         "FOOTER": "<p>Rebuild: <code>.venv/bin/python scripts/servo_identification_protocol_page.py</code>.</p>"}
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    miss = sorted(set(re.findall(r"\[\[((?:fig|tab):[a-z_]+)\]\]", t)) - set(REFS))
    if miss:
        raise SystemExit(f"unresolved references: {miss}")
    t = re.sub(r"\[\[((?:fig|tab):[a-z_]+)\]\]", lambda m: str(REFS[m.group(1)]), t)
    t, n = render_tex(t)
    t = retro_style.apply(t)
    os.makedirs(D, exist_ok=True)
    open(OUT, "w").write(t)
    print(f"formulas {n}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
