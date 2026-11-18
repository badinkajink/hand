#!/usr/bin/env python3
"""Result page of the SR2 hand bench session of 2026-11-17: the servos moved from 5 to 7.5 V, HOM's twist and press on
the hand, the fingertip contacts in HOM's loop, the pick-up of a lying sphere by controlled slip, and the identification
queue started that night (owner, 19:1x: "make sure that our documentation / summary of the things we worked on, with a
clear motivation for evreything described, gets worked on in a cohesive, readable fashion").

    /usr/bin/python3 scripts/hardware/sr2_bench_20261117_page.py          (DOCS_REFRESH=0 on corvus)

Data: docs/experiments/20261117-sr2_bench_session/data/20261117-figdata.json and 20261117-trial_tracking.json, both written
by scripts/hardware/sr2_bench_20261117_figdata.py from the committed records (docs/experiments/20261115-hom_bench_trials/
runs/ and runs_corvus/, data/2026111{6,7}-pinch_pickup/); the 7.5 V free-air steps from
docs/experiments/20261117-sr2_servo_7v5/data/20261117-step_responses.json; the night's series from
docs/experiments/20261117-sr2_system_identification/data/ when present.
"""
from __future__ import annotations

import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401
import docs_index  # noqa: E402
import retro_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "docs/experiments/20261117-sr2_bench_session"
DATA = D / "data"
PAGE = D / "20261117-sr2_hand_bench_session_7v5.html"
TEMPLATE = ROOT / "scripts/ur5/ur5_tip_calibration_page.template.html"
ID = ROOT / "docs/experiments/20261117-sr2_system_identification"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]     # the reference palette's categorical slots 1-5

# the conditions of each trial of the day (the records' headers do not all name the supply, budget or tips)
COND = {
    "20261117-091523": ("g13 (105, 70)", "5 V", "0.84", "off", "the morning comparison"),
    "20261117-091752": ("g13 (105, 70)", "5 V", "0.84", "on", "stopped by the estimator's stall"),
    "20261117-092112": ("g13 (105, 70)", "5 V", "0.84", "on", "the morning comparison"),
    "20261117-164911": ("g13 (105, 70)", "7.5 V", "0.84", "off", "no target reached corvus"),
    "20261117-165559": ("g13 (105, 70)", "7.5 V", "0.84", "off", "all nine servos reset"),
    "20261117-170919": ("g13 (105, 70)", "7.5 V", "0.56", "off", ""),
    "20261117-171157": ("g13 (105, 70)", "7.5 V", "0.56", "on", ""),
    "20261117-181336": ("g13 (105, 70)", "7.5 V", "0.56", "off", "index and middle met"),
    "20261117-182123": ("g0 (70, 110)", "7.5 V", "0.56", "off", ""),
    "20261117-182443": ("g0 (70, 110)", "7.5 V", "0.56", "on", "the sphere fell at the putdown"),
    "20261117-183037": ("press lat0 (70, 110)", "7.5 V", "0.56", "off", "plan's open pose"),
    "20261117-184442": ("press lat0 (70, 110)", "7.5 V", "0.56", "off", "re-planned open pose, HOM gain 0.825"),
    "20261117-184917": ("press lat0 (70, 110)", "7.5 V", "0.56", "off", "re-planned open pose"),
}
ENDED = {"end of schedule": "full run", "drift guard": "drift guard", "corvus stop": "stopped on corvus"}


def j(p):
    return json.loads(Path(p).read_text())


def esc(s):
    return html.escape(str(s))


def table(head, rows, caption, n):
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    tb = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<table><caption>Table {n}. {caption}</caption><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table>')


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def figure(src, n, caption, width=900):
    return (f'<figure><img src="{src}" alt="Figure {n}" style="max-width:{width}px;width:100%"><figcaption>Figure {n}. '
            f"{caption}</figcaption></figure>")


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 13, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.6})
    return plt


def fig_gaps(fd):
    plt = mpl()
    pick = [("20261116-130318", C[0]), ("20261117-170910", C[1]), ("20261117-181327", C[4]), ("20261117-182114", C[2])]
    by = {g["record"]: g for g in fd["gaps"]}
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for rec, col in pick:
        g = by[rec]
        ax.plot(g["t"], g["measured_mm"], color=col, lw=2.0, label=g["label"])
    g = by["20261117-181327"]
    ax.plot(g["t"], g["target_mm"], color=C[4], lw=1.2, ls="--", label="11-17 18:13, HOM's targets")
    ax.axhline(0.0, color="#555555", lw=1.0)
    ax.text(0.08, 0.8, "pads touch on the model", ha="left", va="bottom", fontsize=11, color="#555555")
    ax.set_xlabel("Time since the trial's start (s)")
    ax.set_ylabel("Index-middle gap on the model (mm)")
    ax.set_xlim(0, 8.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=2)
    return png(fig)


def fig_press(fd):
    plt = mpl()
    pick = ["20261116-131927", "20261117-183028", "20261117-184432"]
    by = {p["record"]: p for p in fd["press"]}
    fig, axes = plt.subplots(len(pick), 1, figsize=(9.0, 7.0), sharex=True)
    for ax, rec in zip(axes, pick):
        p = by[rec]
        ax.plot(p["t"], p["mcp_target"], color=C[1], lw=1.0, label="HOM's target")
        ax.plot(p["t"], p["mcp_measured"], color=C[0], lw=2.0, label="measured")
        c = p["mcp_chatter"]
        ax.set_title(f"{p['label']}: target {c['target_rms_deg']:.1f}° rms at {c['target_hz']:.1f} Hz, "
                     f"joint {c['measured_rms_deg']:.1f}° rms", loc="left", fontsize=12)
        ax.set_ylabel("Middle MCP (°)")
    axes[0].legend(loc="lower right", frameon=False, ncol=2)
    axes[-1].set_xlabel("Time since the trial's start (s)")
    axes[-1].set_xlim(0, 6.0)
    fig.tight_layout()
    return png(fig)


def fig_slip(fd):
    plt = mpl()
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for s, col in zip(fd["slip"], (C[0], C[1], C[2])):
        x = [st["squeeze_5v_deg"] for st in s["steps"]]
        y = [st["tilt_end_deg"] for st in s["steps"]]
        ax.plot(x, y, color=col, lw=2.0, marker="o", ms=6, label=s["label"])
    ax.invert_xaxis()
    ax.axhline(6.0, color="#555555", lw=1.0, ls=":")
    ax.text(0.3, 7.0, "goal: 6°", ha="right", fontsize=11, color="#555555")
    ax.set_xlabel("Squeeze past touch, in 5 V terms (°; at 7.5 V the command × 1.5)")
    ax.set_ylabel("Object's tilt at the step's end (°)")
    ax.legend(loc="upper right", frameon=False)
    return png(fig)


def f(x, n=1):
    return "–" if x is None else f"{x:.{n}f}"


def page():
    fd = j(DATA / "20261117-figdata.json")
    tt = j(DATA / "20261117-trial_tracking.json")
    steps = j(ROOT / "docs/experiments/20261117-sr2_servo_7v5/data/20261117-step_responses.json")
    gaps = {g["record"]: g for g in fd["gaps"]}
    prs = {p["record"]: p for p in fd["press"]}
    slips = fd["slip"]

    # Table 1: every trial of the day
    rows1 = []
    for r in tt["trials"]:
        st = r["stamp"]
        if st not in COND:
            continue
        grasp, volt, budget, tips, note = COND[st]
        tr = r["track"]
        trk = max(0.0, 1.0 - tr["err_rms_deg"] / tr["ref_rms_deg"]) if tr.get("ref_rms_deg") else None
        en = r["ended"]
        ended = ENDED.get(en.get("class"), en.get("class", ""))
        if ended != "full run":
            ended += f" at {en.get('t_s', 0):.1f} s"
        mo = r["motion"]
        rot = (f"{f(tr.get('twist_task_end_deg'))} of {f(tr.get('twist_ref_end_deg'))}" if r["task"] == "twist"
               else f"{f(mo.get('twist_end_deg'))} about z")
        rows1.append([f"{st[9:11]}:{st[11:13]}", esc(grasp), volt, budget, tips, esc(ended), f(trk, 3), rot,
                      f(mo.get("lat_max_mm")), esc(note)])
    t1 = table(["Time", "Grasp and layout", "Supply", "Budget (N m)", "Tips", "Ended", "Tracking", "Rotation (°)",
                "Drift (mm)", "Note"], rows1,
               "Every HOM trial of 2026-11-17 on the SR2 hand, scored by scripts/hardware/hom_trial_tracking.py. "
               "Tracking is 1 − (RMS rotation error ÷ RMS commanded rotation) from 2 s to the schedule's end. For the "
               "twist, rotation is the sphere's turn at the end against the commanded turn; for the press, the "
               "cylinder's turn about the palm's z axis. Drift is the object's largest sideways motion in the hand.", 1)

    # Table 2: index-middle gaps
    rows2 = [[esc(g["label"]), f(g["least_measured_mm"]), f(g["least_target_mm"])] for g in fd["gaps"]]
    t2 = table(["Trial", "Least gap, measured joints (mm)", "Least gap, HOM's targets (mm)"], rows2,
               "The least distance between the index's and the middle's parts on the hand's model over each twist. "
               "Each finger is three balls on its kinematic chain: the tip pad (radius 8 mm), the tip's end (6 mm) and "
               "the PIP joint (8 mm). A negative gap means the balls overlap.", 2)

    # Table 3: press chatter
    rows3 = []
    for p in fd["press"]:
        m, q = p["mcp_chatter"], p["pip_chatter"]
        rows3.append([esc(p["label"]), f"{m['target_rms_deg']:.1f} at {m['target_hz']:.1f} Hz",
                      f"{m['measured_rms_deg']:.1f}", f"{q['target_rms_deg']:.1f} at {q['target_hz']:.1f} Hz",
                      f"{q['measured_rms_deg']:.1f}"])
    t3 = table(["Press", "MCP target (° rms)", "MCP measured (° rms)", "PIP target (° rms)", "PIP measured (° rms)"],
               rows3, "Chatter of the middle finger, the press's pushing finger. Each value is the RMS of the angle "
                      "about its own 0.5 s moving mean over the trial; the frequency is half the residual's zero "
                      "crossings per second.", 3)

    # Table 4: pick-ups
    def turn_at(s):
        for a, b in zip(s["steps"], s["steps"][1:]):
            if b["tilt_end_deg"] < a["tilt_end_deg"] - 5.0:
                return b["squeeze_deg"], b["squeeze_5v_deg"]
        return None, None
    lifts = {"11-16 15:45": 42.0, "11-17 18:07": 44.5, "11-17 18:40": 25.9}
    rows4 = []
    for s in slips:
        sq, sq5 = turn_at(s)
        key = s["label"][:11]
        rows4.append([esc(s["label"]), f(lifts.get(key)), f"{f(sq, 2)} ({f(sq5, 2)})",
                      f(s["steps"][-1]["tilt_end_deg"])])
    t4 = table(["Pick-up", "Tilt after the lift (°)", "First turn at squeeze (°, in 5 V terms)", "Tilt after the slip (°)"],
               rows4, "The three pick-ups of the lying sphere by controlled slip. The squeeze is the PIPs' command past "
                      "touch; at 7.5 V the same command gives 1.5 times the torque, so the 5 V terms are the command "
                      "times 1.5. The putdown then released each sphere upright (0.4°, 0.5° and 0.45° from vertical).", 4)

    # the free-air steps
    st75 = [s for s in steps.get("v7_5", []) if s.get("t90") is not None]
    med = {sp: float(np.median([s["t90"] for s in st75 if s["speed"] == sp])) for sp in sorted({s["speed"] for s in st75})}
    t90_txt = " and ".join(f"{v:.2f} s at speed {sp}" for sp, v in sorted(med.items(), reverse=True))
    slews = [s["slew"] for s in st75 if s.get("slew")]
    slew_txt = f"{min(slews):.0f}-{max(slews):.0f} °/s" if slews else "230-510 °/s"

    g = gaps
    S = []
    lede = (
        "On 2026-11-17 the SR2 hand's servos moved from a 5 V to a 7.5 V supply. At 7.5 V they follow their targets "
        "more closely, and three faults showed that the 5 V servos had hidden. HOM's twist targets drive the index and "
        "middle fingers into each other, because the GPU scenes have no contact between fingers. HOM's press targets "
        "chatter at about 5 Hz on the pushing finger, and the servos now follow most of it. The plans' open poses send "
        "two MCPs past their mechanical stops, so a pad touches the object. Fingertip contacts in HOM's loop did not "
        "improve tracking in three pairs of trials. The pick-up of a lying sphere by controlled slip now runs as one "
        "command.")
    summary = ("SR2 hand at 7.5 V on 2026-11-17: HOM's twist targets drive the index and middle together (least gap "
               f"{g['20261117-181327']['least_measured_mm']:.1f} mm at 7.5 V, {g['20261116-130318']['least_measured_mm']:.1f}"
               " mm at 5 V); the press's pushing finger follows HOM's 5 Hz chatter; fingertip contacts did not improve "
               "tracking; the pick-up by controlled slip runs as one command.")

    S.append(("terms", "Terms and metrics", """
<ul>
<li><b>SR2 hand</b>: the three-finger hand on the UR5, each finger on a gantry in the palm (thumb, index, middle), each
with a yaw, an MCP and a PIP joint driven by an SCS0009 position servo. Angles are in the simulator's convention, in
degrees.</li>
<li><b>Layout (x, y)</b>: the gantries' placement, named by the plan; (105, 70) puts the index and middle mounts 63 mm
apart, (70, 110) 103 mm.</li>
<li><b>HOM</b>: the hand-object mechanism controller of Wang, Oh and Pollard (arXiv 2609.25619), run on the workstation
at 50 Hz. Each period it reads the joints and the tracked object, and sends nine servo targets.</li>
<li><b>Grasp g13, g0, lat0</b>: the plans' grasps. g13 and g0 are ground-clear sphere twists at (105, 70) and
(70, 110). lat0 is the cylinder press with the thumb-index pinch at the axis's height.</li>
<li><b>Budget</b>: the torque HOM may command, summed over the nine servos, in the server's 5 V units (N m).</li>
<li><b>Tracking</b>: 1 − (RMS error of the object's rotation ÷ RMS of the commanded rotation), from 2 s to the
schedule's end. A score of 1 is perfect, and 0 is no better than staying still.</li>
<li><b>Drift</b>: the object's largest sideways motion in the hand during a trial (mm). A trial stops at 20-25 mm.</li>
<li><b>Tips off, tips on</b>: HOM without and with the fingertip boards' contact estimates. With tips on, a finger whose
board reads 0.25 N or more takes its contact frame from the board's centre of pressure.</li>
<li><b>Open pose</b>: the finger pose from which a grasp closes on the object.</li>
<li><b>Squeeze</b>: in the pick-up, the PIPs' command past the angle at which the pads touch the ball.</li>
</ul>"""))

    S.append(("why", "Why these experiments", """
<p>The simulator does not predict HOM's trials on the hand. On the blind twists of 2026-11-16 the simulation's
tracking score differed from the hand's by 0.248 on average. The difference fell to 0.152 with an identified servo
model, and to 0.060 with soft tangential contact as well. The owner's plan is to understand this gap and identify the hand before a large GPU study of
morphology, control and contact models. Each experiment of the day answers one question on that path.</p>
<ul>
<li>Do the fingertip boards' contact estimates improve HOM's tracking when HOM uses them (the morning)?</li>
<li>Does a 7.5 V supply give the servos more stiffness, speed and force (the afternoon; owner: "see if we can improve
stiffness/speed/force with these 50% higher voltage")?</li>
<li>Why do the fingers jam and the presses fail on the hand (the evening)?</li>
<li>Can the pick-up of a lying sphere by controlled slip run without a person choosing each step? The owner: "hopefully
this task can be automated for next runs, without losing this interpretability".</li>
<li>Which measurements identify the hand as it stands (the night's queue)?</li>
</ul>"""))

    S.append(("trials", "The day's HOM trials", f"""
<p>Thirteen HOM trials ran (Table 1). Four of them are the day's controlled comparisons, all with the fingertip contacts
off. They are the twist g13 at 5 V and at 7.5 V, and the twist g0 at (70, 110). At 7.5 V and budget 0.56, g13 tracked
0.439 and 0.404, against 0.322 at 5 V that morning and 0.378 at 5 V on 2026-11-16. Each condition has one or two
trials, so the change with the supply is a hint. The press tracked 0.511, 0.000 and 0.391, for the reasons given in the press section
below.</p>
{t1}"""))

    S.append(("supply", "The servos at 7.5 V", f"""
<p>The SCS0009 is listed for 4-7.4 V, and its input limit register reads 9.0 V. A stalled motor's torque at a given PWM
duty scales with the supply. At 7.5 V the same position error therefore gives 1.5 times the torque and the current. It
gives 2.25 times the heat. In free air a 20° step reached 90 % of its size in {t90_txt} (medians). The peak speeds were {slew_txt}
(docs/experiments/20261117-sr2_servo_7v5/).</p>
<p>The servos protect themselves on PWM duty: over 80 % for about 1 s cuts the output to 20 %. At HOM's 5 V budget
(0.84 N m) all nine servos reset together 1.45 s into a grasp at 16:55, and the sphere slid out at the lift. A supply
dip resets every servo at once; the 5 V gr_g4 failures of 2026-11-16 looked the same. At budget 0.56, which keeps the
summed current at the 5 V level, no servo reset in eight trials. Holding a tilted sphere through a long putdown warmed
the thumb's yaw servo to 50 °C, and three presses warmed it to 56 °C (limit 70 °C). The night's runs pause at 50 °C
until every servo is under 40 °C.</p>"""))

    S.append(("collide", "Index and middle fingers collide in HOM's twist", f"""
<p>At 18:13 the owner saw the index and middle fingers press on each other during g13's twist. The records agree
(Figure 1, Table 2). One second into the twist, HOM's targets put the two tip pads 10.8 mm into each other on the
hand's model. On the hand the index's yaw stopped at −15.5° from 1.6 s, short of its −23.8° target, and the pads came
within {g['20261117-181327']['least_measured_mm']:.1f} mm at 6.4 s. Four servos showed the overload flag at the
putdown.</p>
<p>The GPU scenes leave out contact between fingers, because it slows MuJoCo-Warp (owner: "we actually do let the
fingers collide into each other because that greatly speeds up mjwarp"). HOM's targets therefore pass through the
other finger in simulation. At 5 V the servos lagged their targets by up to 19° and the fingers stayed
{g['20261116-130318']['least_measured_mm']:.1f} mm apart. At 7.5 V and budget 0.56 the lag fell to 13°, and the
fingers met. At (70, 110) the g0 twist kept the fingers {g['20261117-182434']['least_measured_mm']:.1f} to
{g['20261117-182114']['least_measured_mm']:.1f} mm apart, although its targets overlap by up to 14.9 mm. The
workstation added a finger-gap measure to the GPU study (W4): the least index-middle and thumb-finger gaps of the
states and of HOM's targets, computed without a contact solve.</p>
{figure(fig_gaps(fd), 1, "The least distance between the index's and the middle's parts on the hand's model during "
        "four twists. Solid lines use the measured joints, the dashed line HOM's targets of the 18:13 trial. Below "
        "zero the parts overlap.")}
{t2}"""))

    S.append(("tips", "HOM with the fingertip contacts", f"""
<p>The fingertip boards estimate each pad's contact point and force from the magnets' field. With tips on, HOM takes a
finger's contact frame from that estimate whenever the board reads 0.25 N or more. Below 0.25 N the frame comes from
the closest points of the finger and object models. In three pairs of trials the tips-on trial tracked better once and drifted out twice
(Table 1). At 5 V it tracked 0.394 against 0.322 (09:21 and 09:15). At 7.5 V the drift guard stopped it at 5.3 s and at
4.9 s (17:11, 18:24). At 18:24 the sphere fell out of the hand during the putdown.</p>
<p>The owner asked whether this is HOM as published ("this point jitter with updating directly from the contact sensing
is the whole point of the complementary filter in the original paper"). The paper's Eq. (1) uses the measured contact
frames, and in contact these come from the centre of pressure; that part matches. The paper's complementary filter,
Eqs. (4)-(6), smooths the impedance controller's action point toward the measured contact. Each period it moves the
point by k·dt of the way to the measurement. Our gain k reaches 100 per second; the paper prints no values. At the
paper's 500 Hz that is 0.2 of the way. The bench runs at 50 Hz with the same gain. There k·dt reaches 2.0 when the
contact's reference speed is high, and the filtered point then overshoots the measurement; at rest k is 5 per second
and k·dt is 0.1. The workstation added the paper's rate as sub-steps of 2 ms (--filter-step-s 0.002). It also added a
hysteresis on the 0.25 N contact switch (--contact-off). On the calibrated simulation scene neither change alters
tracking, and no 4-5 Hz chatter appears with or without them. On the server's own scene the g0 twist held the sphere in
1 of 6 seeds with the 2 ms sub-steps, against 4 of 6 without, three of the drops at a failed SVD. The night's twist
repeats test the changes on the hand (the identification section).</p>"""))

    p30, p44, p49 = prs["20261117-183028"], prs["20261117-184432"], prs["20261117-184907"]
    S.append(("press", "The cylinder press", f"""
<p>The press pinches the cylinder between the thumb and the index and pushes it with the middle finger. Three things
went wrong on the hand.</p>
<ul>
<li>The open pose brushed the cylinder. The plan sends the index MCP to −14.8°, past its mechanical stop at −11.4°.
Commanded at the stop, the index pad sits on the cylinder on the model, and opening the fingers turned the cylinder
3.6°. The workstation re-planned the open pose inside the stops; then opening turned it 0.24°.</li>
<li>The cylinder turned during the close: 10-14° by 0.8 s and 21-27° by the lift. The turn starts 0.3-0.5 s into the
close, before any board reads 0.3 N, and the thumb's board reaches 0.3 N 0.14-0.40 s before the index's. In
simulation the planned grasp has force closure, and the simulated cylinder turns less than 1°. The thumb's first light
contacts turn the free-standing cylinder before the pinch closes. The owner saw the same at the bench: "the grasp ...
does not have force closure at the very onset, produces some skew, and then messes everything up".</li>
<li>The pushing finger chattered (Figure 2, Table 3). HOM's middle MCP target swung {p30['mcp_chatter']['target_rms_deg']:.1f}°
rms at about {p30['mcp_chatter']['target_hz']:.0f} Hz, and the joint followed with
{p30['mcp_chatter']['measured_rms_deg']:.1f}°. At 5 V on 2026-11-16 the target swung 8.3° at 9 Hz, but the joint moved
only 0.9°. In simulation with the exact object pose, HOM's middle targets on this press swing only 0.13-0.22° rms at
about 0.7 Hz. On the hand the chatter runs through the tracked pose: with trends removed, the middle MCP's target moves
with the tracked cylinder's height (correlation −0.66). The finger pushes the cylinder, the cameras see it move, and
HOM's next target follows; the 7.5 V servos now follow the chatter. Raising HOM's servo gain to 0.825, the expected
7.5 V gain, doubled the chatter ({p44['mcp_chatter']['target_rms_deg']:.1f}° target, {p44['mcp_chatter']['measured_rms_deg']:.1f}°
joint) and tracked 0.000. With the original gain and the re-planned open pose, the chatter was as at 18:30
({p49['mcp_chatter']['target_rms_deg']:.1f}°, {p49['mcp_chatter']['measured_rms_deg']:.1f}°).</li>
</ul>
{figure(fig_press(fd), 2, "The middle finger's MCP during three presses: HOM's target (thin) and the measured angle "
        "(thick). The 5 V servos barely followed the chatter; at 7.5 V the joint follows most of it, and HOM's gain of "
        "0.825 doubles it.")}
{t3}"""))

    S.append(("pickup", "Pick-up of a lying sphere by controlled slip", f"""
<p>A sphere on a stem and block lies on its side; the hand picks it up, turns it upright and puts it down. The thumb and
one partner finger pinch the ball across the stem, 3° above the ball's equator, with about 1.3 N on each pad. The pinch
works as a hinge about the line between the two pads, resisted only by the pads' twisting friction. As the hand rises
at 3 mm/s, the table drags the block, and the torque about the hinge turns the object toward upright. The turning stops
when the block leaves the table, at 26-45°. In the air the squeeze is lowered in small steps. The servos' gears do not
backdrive, so the grip force falls only when a servo reverses. The object swings toward vertical in jumps whenever
gravity's torque exceeds the twisting friction (Figure 3, Table 4). The squeeze is raised again once the tilt is within
6°. The putdown moves the sphere to the table's centre and sets it down, watched by the tracker and the force sensor.
Then it releases the sphere.</p>
<p>At 7.5 V the squeeze was scaled from 8° to 5.3°, so that the grip force matches the 5 V runs. The first turn came
at 2.5 and 2.85° of squeeze, 3.75 and 4.3° in 5 V terms, against 4° at 5 V. The run of 18:38 is the first by one
command, lab/pickup_cycle.sh. It ran plan, grasp, lift, slip, putdown and the wrist's return in 3 minutes, printing each
step's key lines and keeping each step's whole output. It stops at the first step that fails.</p>
<figure><video controls muted playsinline preload="metadata" style="max-width:960px;width:100%"
src="media/20261117-pickup_one_command.mp4"></video><figcaption>Film 1. The pick-up of 18:38 by one command
(lab/pickup_cycle.sh), both cameras: the grasp at 6x, the lift at 3x, the slip at 2x with its waits kept, the putdown at
4x. People at the left of the right-hand camera are blurred.</figcaption></figure>
{figure(fig_slip(fd), 3, "The slip of the three pick-ups: the object's tilt at the end of each squeeze step, against "
        "the squeeze in 5 V terms. The squeeze falls from left to right.", width=800)}
{t4}"""))

    S.append(("tools", "Tools for repeatable runs", """
<ul>
<li>HOM's targets stay 5° inside the ends of the servos' potentiometer track (sr2hand.control.servo_limits). At 09:21
the index PIP's target crossed the end and its reading wrapped.</li>
<li>The fingertip estimator runs in its own process (tip_contacts.TipProcess). An estimate took 16.8-18.8 ms there,
against 68-73 ms in the trial loop's thread, which had stalled the loop.</li>
<li>A trial stops when no target arrives within 0.5 s (hom_trial.py --first-target-s). At 16:49 no target reached
corvus and the hand lifted empty.</li>
<li>The release skill. When the fingers' straight path to zero would push into the object, letgo flexes the MCPs and
PIPs back with the yaws kept. Then the hand rises, and the yaws turn to zero. Each step is checked on the hand's model first, and
the extension is ramped over 3 s (owner: "a slowed/damped ... extension").</li>
<li>The pick-up as one command (lab/pickup_cycle.sh) and the twist repeats as one command (lab/repeat_cycle.sh).</li>
<li>A map of the bench's data paths, rates and clocks with each recorded failure placed on it:
<a href="../../overviews/bench_data_paths.html">docs/overviews/bench_data_paths.html</a>.</li>
</ul>"""))

    night = ""
    nf = ID / "data/20261117-night_summary.json"
    if nf.exists():
        night = j(nf).get("html", "")
    S.append(("ident", "Identification of the hand before the GPU study", f"""
<p>The workstation and the lab agreed on a queue of measurements
(docs/experiments/20261117-sr2_system_identification/QUEUE.md). Geometry comes first, because every later fit reads
the joint angles and the fingertip positions. The servos follow, in free air and then against the force sensor.
Contact and blind validation need the owner at the bench.</p>
<ul>
<li>S1, the geometry after the owner's re-zeroing: each joint's reachable range, and the fingertip tags at 8-10 poses
per finger. The workstation fits the joint zeros and the mount positions from them.</li>
<li>S2, the free-air servo identification of 2026-11-06 repeated at 7.5 V: staircases, timed dwells and steps of every joint.
It ran at the (85, 170) layout; at (70, 110) one block collided in the hand's CAD model.</li>
<li>S3, the torque scale: a fingertip stepped past contact against the table until 3 N.</li>
<li>S4 and S5, contact and blind validation, with the owner.</li>
<li>The fingertip boards recorded through every free-air series, for a new fit of their response to the hand's own
pose. The owner asked for it: "good idea to get the magnetic fields response recalibrated too with this new
sysid".</li>
</ul>
{night}"""))

    S.append(("next", "Next steps", """
<ul>
<li>Run the GPU study with the finger-gap measure. Choose the twist layouts and grasps again among those whose fingers
keep a margin; on the hand g0 at (70, 110) kept 17 mm.</li>
<li>Re-plan every open pose inside the measured MCP stops (−11.4° index, −10.0° middle), as done for g13 and lat0.</li>
<li>Close the press grasp in stages, the index to near contact first and then the thumb, and measure the block's
resistance to turning on the table.</li>
<li>Find what drives the press chatter: test the paper-rate filter and a filtered object pose on the press, with the
chatter of Table 3 as the measure.</li>
<li>Model the pick-up's lift and slip in simulation with the sphere pads and with point contact. Compare the tilt after
the lift and the squeeze at the first turn with Table 4.</li>
</ul>"""))

    S.append(("records", "Records", """
<p>Trial records: docs/experiments/20261115-hom_bench_trials/runs_corvus/ (corvus) and runs/ (the HOM server), stamps
20261117-*. The figure data and the trial table: docs/experiments/20261117-sr2_bench_session/data/, written by
scripts/hardware/sr2_bench_20261117_figdata.py. The pick-ups' telemetry: data/20261116-pinch_pickup/ and
data/20261117-pinch_pickup/ in the bench study. Plans: plans/20261117-*. The runbook:
docs/experiments/20261115-hom_bench_trials/README.md. The identification queue and its notes:
docs/experiments/20261117-sr2_system_identification/. Videos of every step: logs/20261116-hom_trials/ on corvus.</p>"""))

    toc = "".join(f'<li><a href="#{k}">{t}</a></li>' for k, t, _ in S)
    body = "".join(f'<section id="{k}"><h2>{t}</h2>{s}</section>' for k, t, s in S)
    title = "SR2 hand bench session of 2026-11-17 at a 7.5 V servo supply"
    out = (TEMPLATE.read_text()
           .replace("{{TITLE}}", title).replace("{{TEXCSS}}", "")
           .replace("{{EYEBROW}}", "SR2 hand on the UR5 · 2026-11-17")
           .replace("{{H1}}", title).replace("{{LEDE}}", lede)
           .replace("{{BYLINE}}", "Corvus lab session (hand-18) with the workstation's session (hand-4a); the owner at "
                                  "the bench in the morning, 16:5x-17:15 and 18:00-19:10")
           .replace("{{FILM}}", "").replace("{{TOC}}", toc).replace("{{SECTIONS}}", body)
           .replace("{{FOOTER}}", "Built by scripts/hardware/sr2_bench_20261117_page.py from "
                                  "docs/experiments/20261117-sr2_bench_session/data/."))
    out = retro_style.apply(out)
    out = docs_index.tag(out, topic="hardware", summary=summary)
    PAGE.write_text(out)
    print(PAGE, f"{PAGE.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    page()
