#!/usr/bin/env python3
"""Builds docs/experiments/20261110-hom_torque_control/20261110-hom_inverse_dynamics_control.html: inverse-dynamics
torque control of the SR2 hand's fingers under HOM against the SCS0009 position servos, on the layouts and object sizes
of the tracking page's films (owner, 2026-11-09: "implement, test and analyse inverse-dynamics torque control of the
fingers in place of MuJoCo's position actuators"; "that might help things?").

Data: rows/20261110-torque_compare.jsonl (scripts/hom/hom_torque.py compare); films in media/ (hom_torque.py film).

    .venv/bin/python scripts/hom/hom_torque_page.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import docs_index  # noqa: E402
import fp_review_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = ROOT / "docs/experiments/20261110-hom_torque_control"
OUT = D / "20261110-hom_inverse_dynamics_control.html"
ROWS = D / "rows" / "20261110-torque_compare.jsonl"
MEDIA = D / "media"
TEX_CACHE = D / "texsvg_cache.json"
FPR = "../20261029-foundational_pose_review/"
TRACK = FPR + "20261109-hom_tracking_score.html"
SYSID = "../20261106-sr2_servo_sysid/20261106-sr2_servo_identification.html"
HOMREV = "../20261029-hom_control_review/"
TASKS = ("twist", "roll", "press")
TNAME = {"twist": "Twist", "roll": "Roll", "press": "Press"}
ARMS = ("position", "torque 25.6", "torque 63")
ANAME = {"position": "Position servos", "torque 25.6": "Torque, 25.6 rad/s", "torque 63": "Torque, 63 rad/s"}
FINGERS = ("thumb", "index", "middle")
CMD_F = {"twist": (2.5, 2.5, 2.5), "roll": (2.5, 2.5, 2.5), "press": (3.0, 3.0, 0.5)}   # hom_tasks.PRM, N per pad
FILMS = (("twist", 30.0, "pick"), ("roll", 63.0, "pick"), ("press", 13.0, "generalist"))
table, esc = P.table, P.esc
ALGS: list[str] = []


def alg(caption, body):
    """An algorithm as a LaTeX algorithmic block (rendered by texsvg after the page's inline mathematics)."""
    ALGS.append(f"\\begin{{algorithmic}}[1]\n{body}\n\\end{{algorithmic}}")
    return (f'<figure class="alg"><figcaption><b>Algorithm&#160;{len(ALGS)}.</b> {caption}</figcaption>'
            f"\x00ALG{len(ALGS) - 1}\x00</figure>")


def arm(r):
    return "position" if r["plant"] == "position" else f"torque {r['wn']:g}"


def load():
    return [json.loads(x) for x in ROWS.read_text().splitlines() if x.strip()]


def followed(r):
    return max(0.0, 1.0 - r["track_err_rms_deg"] / r["track_ref_rms_deg"]) if r.get("held") else 0.0


def stats(rs):
    """Means over rollouts: scores over all of them (a diverged or dropped rollout scores 0), the rest over the rollouts
    that ran to the end."""
    ok = [r for r in rs if "sat_share" in r]
    held = [r for r in ok if r.get("held")]
    return {"n": len(rs), "held": len(held), "diverged": len(rs) - len(ok),
            "track": float(np.mean([r["track"] for r in rs])), "old": float(np.mean([r["old"] for r in rs])),
            "followed": float(np.mean([followed(r) for r in rs])),
            "three": float(np.mean([r.get("contact_all_frac") or 0.0 for r in rs])),
            "off": float(np.mean([r["off_axis_rms_dps"] for r in held])) if held else None,
            "sat": float(np.mean([r["sat_share"] for r in ok])) if ok else None,
            "contact": {f: float(np.mean([r["contact_frac"][f] for r in ok])) for f in FINGERS} if ok else {},
            "force": {f: float(np.median([r["force_med_N"][f] for r in held if r["force_med_N"][f] is not None]))
                      for f in FINGERS} if held else {}}


def numbers(rows):
    N = {"task": {}, "cfg": {}, "which": {}}
    for task in TASKS:
        for a in ARMS:
            N["task"][(task, a)] = stats([r for r in rows if r["task"] == task and arm(r) == a])
            for which in ("pick", "generalist"):
                rs = [r for r in rows if r["task"] == task and arm(r) == a and r["which"] == which]
                if rs:
                    N["which"][(task, which, a)] = stats(rs)
    for r in rows:
        N["cfg"].setdefault((r["task"], r["size"], r["which"], tuple(r["layout"])), {}).setdefault(arm(r), []).append(r)
    N["cfg"] = {k: {a: stats(rs) for a, rs in v.items()} for k, v in N["cfg"].items()}
    N["seed_sd"] = float(np.median([np.std([r["track"] for r in rs]) for rs in _groups(rows).values()]))
    N["seed_sd_mean"] = float(np.mean([np.std([r["track"] for r in rs]) for rs in _groups(rows).values()]))
    N["small_press"] = {a: sum(bool(r.get("held")) for r in rows if r["task"] == "press" and r["which"] == "generalist"
                               and r["size"] in (10.0, 13.0) and arm(r) == a) for a in ARMS}
    return N


def _groups(rows):
    g = {}
    for r in rows:
        g.setdefault((r["task"], r["size"], r["which"], arm(r)), []).append(r)
    return g


def count(n):
    return "none" if n == 0 else str(n)


def f2(v):
    return "&#8211;" if v is None else f"{v:.2f}"


def f1(v):
    return "&#8211;" if v is None else f"{v:.1f}"


# ------------------------------------------------------------------------------------------------ sections
def sec_terms():
    terms = [
        ("Layout", "the placement of the three finger mounts on the palm. Here it is a symmetric tripod in two "
         "distances, in mm: $x_{sep}$ from the thumb to the index-middle pair and $y_{sep}$ from the index to the "
         "middle."),
        ("Joints", "each finger has three servo joints: the yaw at its mount and the two flexion joints mcp and pip."),
        ("Tasks, object sizes", "the twist and the roll of a sphere and the press of a cylinder under HOM [1], at the "
         f"sizes of the <a href=\"{TRACK}\">tracking page's</a> films. The spheres are 25, 30, 63 and 80 mm across. "
         "The cylinders are 10, 13, 40 and 50 mm across. The roll's command is the 20 deg one of that page."),
        ("Layouts compared", "at each size, the layout picked there by the tracking score (its Table 3) with its grasp, "
         "and layout (70, 110) mm with its first grasp, the foundational pose. Layout (70, 110) mm has no press pose on "
         "the 50 mm cylinder."),
        ("Position servos", "the SCS0009 model of the HOM scripts, a MuJoCo position actuator per joint. Its gain "
         "$k_p$ is 0.72 N m/rad and its force limit $\\tau_{max}$ 0.18 N m; the joint damping $b$ is 0.0144 N m s/rad."),
        ("Torque plant", "the same joints with motors in place of the position actuators. The motors are driven by "
         "the law of Section SEC[plant] at every 1 ms physics step, with the same $\\tau_{max}$, $b$ and armature."),
        ("Loop frequency $\\omega_n$", "the natural frequency of the torque plant's closed loop on the joint, in rad/s. "
         "At 25.6 rad/s its stiffness on the joints' inertia equals the servos' 0.72 N m/rad; 63 rad/s is a 10 Hz loop."),
        ("Held", "the object ended within 15 mm of its start height with at least two fingers loaded."),
        ("Tracking score", "the score of the tracking page (<code>hom_landscape.track_score</code>): the share of the "
         "commanded rotation that the object followed, $1 - e_{rms}/\\theta_{rms}$ at least 0, times the three-pad "
         "share. It is 0 when the object is dropped or the rollout diverged. $e_{rms}$ is the RMS angle between the "
         "object's orientation and the commanded one over the task, $\\theta_{rms}$ the RMS commanded rotation."),
        ("Earlier score", "the landscapes' score before 2026-11-09 (<code>hom_landscape.rollout_score</code>), quoted "
         "beside the tracking score."),
        ("Rotation followed", "the middle factor of the tracking score, $\\max(0, 1 - e_{rms}/\\theta_{rms})$."),
        ("Three-pad share", "the share of the task's 10 ms control periods in which all three pad forces exceed "
         "0.05 N."),
        ("Contact share of a finger", "the share of the task's control periods in which that finger's pad force "
         "exceeds 0.05 N."),
        ("Pad force", "a finger's median pad force over the periods of its contact share, in N. The commanded forces "
         "are 2.5 N per pad in the twist and the roll. In the press they are 3 N for the thumb and the index and 0.5 N "
         "for the middle."),
        ("Off-axis angular velocity", "the RMS over the task of the object's angular velocity about the two axes "
         "normal to the commanded one, in deg/s. These are palm x and y in the twist, x and z in the roll and y and z "
         "in the press."),
        ("Torque-limit share", "the share of physics steps on which at least one of the nine finger joints is at "
         "$\\tau_{max}$."),
        ("Diverged", "a rollout whose state became non-finite, so that HOM's least squares failed; it scores 0."),
    ]
    return "<dl>" + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"


def sec_plant():
    t = (
        "<p>HOM's planner is the same in both plants. In each 10 ms control period, HOM's equation (2) is solved for "
        "each finger's joint velocities $u$, which are integrated into the joint reference $\\theta_{ref}$. The "
        "impedance step then moves the reference's action point onto the measured surface by the joint offset "
        "$\\Delta\\theta_{gap}$. The commanded pad force $F$ acts along the action direction $n$ at the action point "
        "$p$ (<a href=\"" + HOMREV + "20261030-hom_controller_code_walkthrough.html\">HOM code walkthrough</a>). The "
        "position servos receive the target</p>"
        "$$\\theta_{cmd} = \\theta_{ref} + \\Delta\\theta_{gap} + \\frac{J(p)^\\top F n + g(\\theta)}{k_p},$$"
        "<p>where $J(p)$ is the pad body's Jacobian at $p$ and $g(\\theta)$ the joints' gravity torque. Each joint "
        "then exerts $\\tau = \\mathrm{clip}(k_p(\\theta_{cmd} - \\theta) - b\\dot\\theta, \\pm\\tau_{max})$, so the "
        "force reaches the object through the servo's spring.</p>"
        "<p>In the torque plant the actuators are motors. At every 1 ms physics step the joints of each finger get</p>"
        "$$\\tau = \\mathrm{clip}\\big(M(\\theta)\\,\\ddot\\theta_{ref} + c(\\theta, \\dot\\theta) + g(\\theta) + "
        "b\\dot\\theta + J(p)^\\top F n,\\ \\pm\\tau_{max}\\big),$$"
        "$$\\ddot\\theta_{ref} = \\omega_n^2(\\theta_t - \\theta) + 2\\zeta\\omega_n(\\dot\\theta_t - \\dot\\theta).$$"
        "<p>$M$ is the finger's joint-space inertia, which includes the servos' armature of 0.001 kg m$^2$ per joint, "
        "and $c + g$ the Coriolis and gravity torques (MuJoCo's <code>qfrc_bias</code>). The term $b\\dot\\theta$ "
        "cancels the joint damping. $F n$ is the force applied by the pad to the object. Written with the force on the "
        "finger, $f_{cmd} = -F n$, the term is $-J^\\top f_{cmd}$. Within a period the reference $\\theta_t$ runs from the "
        "period's previous reference to its new one at Eq. (2)'s rate:</p>"
        "$$\\theta_t = \\theta_{ref,k-1} + \\frac{t - t_k}{\\Delta t}\\,(\\theta_{ref,k} - \\theta_{ref,k-1}) + "
        "\\Delta\\theta_{gap}, \\qquad \\dot\\theta_t = \\frac{\\theta_{ref,k} - \\theta_{ref,k-1}}{\\Delta t},$$"
        "<p>with $\\Delta t$ = 10 ms. The reference moves at $u$ except where HOM clips it to the joint limits or to "
        "0.3 rad from the measured angle. $J(p)^\\top F n$ is computed once per period, at its start.</p>"
        "<p>The joints' inertia is 1.01 to 1.15 g m$^2$ at the open pose, nearly all armature. With $\\zeta$ = 1 and "
        "$\\omega_n$ = 25.6 rad/s the loop's stiffness $M\\omega_n^2$ is the servos' 0.72 N m/rad, and it is "
        "critically damped. The servos' own damping ratio is $b/(2\\sqrt{k_p M})$ = 0.26. With $\\omega_n$ = 63 "
        "rad/s the stiffness is 4.4 N m/rad, and an error of 2.4 deg reaches $\\tau_{max}$.</p>")
    t += alg("One 1 ms physics step of the torque plant (<code>scripts/hom/hom_torque.py</code>, "
             "<code>TorquePlant.advance</code>).", r"""\Require the period's references \(\theta_{ref,k-1}, \theta_{ref,k}\), offset \(\Delta\theta_{gap}\) and \(J(p)^\top F n\) of each finger
\State compute positions and velocities, \(M(\theta)\), \(c + g\) and the joint damping \Comment{\texttt{mj\_step1}}
\State \(s \gets \min(1, (t - t_k)/\Delta t)\)
\For{each finger}
  \State \(\theta_t \gets \theta_{ref,k-1} + s\,(\theta_{ref,k} - \theta_{ref,k-1}) + \Delta\theta_{gap}\)
  \State \(\ddot\theta_{ref} \gets \omega_n^2(\theta_t - \theta) + 2\zeta\omega_n\big((\theta_{ref,k} - \theta_{ref,k-1})/\Delta t - \dot\theta\big)\)
  \State \(\tau \gets \mathrm{clip}(M\ddot\theta_{ref} + c + g + b\dot\theta + J(p)^\top F n, \pm\tau_{max})\)
\EndFor
\State integrate one step with \(\tau\) on the motors \Comment{\texttt{mj\_step2}}""")
    return t


def sec_compare(N):
    T_ = N["task"]
    body = []
    for task in TASKS:
        for a in ARMS:
            v = T_[(task, a)]
            body.append([TNAME[task] if a == ARMS[0] else "", ANAME[a], f2(v["track"]), f2(v["old"]), f2(v["followed"]),
                         f2(v["three"]), f1(v["off"]), f2(v["sat"]), f"{v['held']} of {v['n']}"])
    tw, ro, pr = (lambda a: T_[("twist", a)]), (lambda a: T_[("roll", a)]), (lambda a: T_[("press", a)])
    W = N["which"]
    t = (
        f"<p>Each task ran at four object sizes on two layouts, at seeds 0, 1 and 2, on the CPU with every contact. The "
        f"torque plant did not raise the tracking score on the twist or the roll (Table 1). At 63 rad/s the twist "
        f"scored {tw('torque 63')['track']:.2f} against {tw('position')['track']:.2f} with the position servos. The "
        f"roll scored {ro('torque 63')['track']:.2f} against {ro('position')['track']:.2f}, and the press "
        f"{pr('torque 63')['track']:.2f} against {pr('position')['track']:.2f}. At 25.6 rad/s, the servos' stiffness, "
        f"the three tasks scored within 0.03 of the position servos.</p>"
        f"<p>The object turned more steadily under the torque plant. Its off-axis angular velocity fell from "
        f"{tw('position')['off']:.1f} to {tw('torque 63')['off']:.1f} deg/s in the twist and from "
        f"{ro('position')['off']:.1f} to {ro('torque 63')['off']:.1f} deg/s in the roll. In the twist the object "
        f"followed {tw('torque 63')['followed']:.2f} of the commanded rotation, against {tw('position')['followed']:.2f} "
        f"with the position servos. Pad contact was lost more often, and the tracking score did not rise. The "
        f"three-pad share fell from "
        f"{tw('position')['three']:.2f} to {tw('torque 63')['three']:.2f} in the twist and from "
        f"{ro('position')['three']:.2f} to {ro('torque 63')['three']:.2f} in the roll.</p>"
        f"<p>The layouts picked by the tracking score were chosen on the position servos at these seeds. Layout "
        f"(70, 110) mm was chosen on neither plant. On it the twist scored "
        f"{W[('twist', 'generalist', 'position')]['track']:.2f} with the position servos, "
        f"{W[('twist', 'generalist', 'torque 25.6')]['track']:.2f} at 25.6 rad/s and "
        f"{W[('twist', 'generalist', 'torque 63')]['track']:.2f} at 63 rad/s. In the press it held the 10 and 13 mm "
        f"cylinders in {count(N['small_press']['position'])} of 6 rollouts with the position servos. At 63 rad/s it "
        f"held them in {count(N['small_press']['torque 63'])} of 6. On the 40 mm cylinder it scored "
        f"{N['cfg'][('press', 40.0, 'generalist', (70.0, 110.0))]['position']['track']:.2f} and "
        f"{N['cfg'][('press', 40.0, 'generalist', (70.0, 110.0))]['torque 63']['track']:.2f}. In the roll its sphere "
        f"followed none of the commanded rotation in any plant. The standard deviation of a configuration's tracking "
        f"score over the three seeds has a median of {N['seed_sd']:.3f} and a mean of {N['seed_sd_mean']:.3f}.</p>")
    t += table(["Task", "Plant", "Tracking score", "Earlier score", "Rotation followed", "Three-pad share",
                "Off-axis angular velocity (deg/s)", "Torque-limit share", "Held"], body,
               "The three plants on the films' layouts and sizes: means over 8 configurations of the twist and the roll "
               "and 7 of the press, seeds 0-2. Scores count a dropped or diverged rollout as 0; the off-axis angular "
               "velocity is over the held rollouts.", num_from=2)
    body = []
    for (task, size, which, lay), v in sorted(N["cfg"].items(), key=lambda kv: (TASKS.index(kv[0][0]), kv[0][1],
                                                                               kv[0][2] != "pick")):
        cells = [f2(v[a]["track"]) + (f" ({v[a]['diverged']} diverged)" if v[a]["diverged"] else "") for a in ARMS]
        body.append([TNAME[task], f"{size:g}", f"({lay[0]:.0f}, {lay[1]:.0f})" + (" pick" if which == "pick" else ""),
                     *cells, *[f1(v[a]["off"]) for a in ARMS]])
    t += table(["Task", "Size (mm)", "Layout (mm)", "Tracking, position servos", "Tracking, 25.6 rad/s",
                "Tracking, 63 rad/s", "Off-axis, position servos (deg/s)", "Off-axis, 25.6 rad/s (deg/s)",
                "Off-axis, 63 rad/s (deg/s)"], body,
               "Each configuration's tracking score and off-axis angular velocity, means over seeds 0-2. Pick: the layout "
               "picked at that size by the tracking score on the position servos.", num_from=3)
    return t


def sec_contact(N):
    T_ = N["task"]
    body = []
    for task in TASKS:
        for a in ARMS:
            v = T_[(task, a)]
            body.append([TNAME[task] if a == ARMS[0] else "", ANAME[a],
                         *[f2(v["contact"].get(f)) for f in FINGERS],
                         *[f"{v['force'].get(f, float('nan')):.2f} ({c:g})" for f, c in zip(FINGERS, CMD_F[task])],
                         f2(v["sat"])])
    tw = (lambda a: T_[("twist", a)])
    pr = (lambda a: T_[("press", a)])
    t = (
        f"<p>Where a pad touched, it pressed with about the commanded force in both plants (Table 3). In the twist the median "
        f"pad force was 2.0 to 2.8 N against the commanded 2.5 N. In the press the pinching pads pressed with 2.7 to "
        f"3.1 N against 3 N. The press's middle finger pressed with 0.21 to 0.23 N against its commanded 0.5 N in every "
        f"plant.</p>"
        f"<p>The index lost the sphere most often. In the twist it touched for {tw('position')['contact']['index']:.2f} "
        f"of the task with the position servos and {tw('torque 63')['contact']['index']:.2f} at 63 rad/s, against "
        f"0.86 to 0.99 for the thumb and the middle.</p>"
        f"<p>The fingers' torque was at its limit most of the time. With the position servos at least one joint was "
        f"at 0.18 N m on {tw('position')['sat']:.2f} of the twist's physics steps and {pr('position')['sat']:.2f} of "
        f"the press's. At 63 rad/s the shares were {tw('torque 63')['sat']:.2f} and {pr('torque 63')['sat']:.2f}. At "
        f"the touch poses of four of the configurations, 30 to 72 mN m per newton of pad force is required at a finger's "
        f"yaw or mcp joint. Up to 0.18 N m is then required at one joint for the commanded 2.5 N. For the press's 3 N "
        f"pinch, 0.19 N m is required at the mcp joints of the thumb and the index, more than the limit. Neither plant "
        f"can exceed it, so the commanded forces leave the joints little torque for motion.</p>")
    t += table(["Task", "Plant", "Contact share, thumb", "Contact share, index", "Contact share, middle",
                "Pad force, thumb (N)", "Pad force, index (N)", "Pad force, middle (N)", "Torque-limit share"], body,
               "Contact and torque per plant: contact shares are means over the rollouts that ran to the end; pad "
               "forces are medians over the held rollouts, with the commanded force in brackets.", num_from=2)
    return t


def sec_films():
    caps = {("twist", 30.0, "pick"): "Twist of the 30 mm sphere on layout (95, 90) mm, picked by the tracking score.",
            ("roll", 63.0, "pick"): "Roll of the 63 mm sphere on layout (155, 85) mm, picked by the tracking score, "
                                    "viewed from below.",
            ("press", 13.0, "generalist"): "Press of the 13 mm cylinder on layout (70, 110) mm, its first grasp."}
    t = ("<p>Each film shows one configuration on the three plants: from left to right the position servos and the "
         "torque plant at 25.6 and 63 rad/s. Each clip runs at the seed whose tracking score is closest to that plant's "
         "mean over seeds 0-2. The rods through the object show its axes.</p>")
    for k in FILMS:
        p = MEDIA / f"20261110-film_torque_{k[0]}_d{k[1]:g}_{k[2]}.mp4"
        t += P.film(p, caps[k] + " Left to right: position servos, torque at 25.6 rad/s, torque at 63 rad/s. "
                    "<code>hom_torque.py film</code>.")
    return t


def sec_hardware():
    return (
        "<p>The SCS0009 servos of the built hand are commanded by position only. A torque $\\tau$ reaches a joint as the "
        "position offset $\\tau/k_p$ through the servo's spring. The spring measured on 2026-11-06 was 0.55 N m/rad, "
        f"at the datasheet's torque and the measured 5.1 V supply (<a href=\"{SYSID}\">servo identification page</a>). "
        "The model has 0.72 N m/rad. Inside a dead band of 0.59 deg the servo gives no drive, and its first drive "
        "outside the band is 13.7 mN m. A torque under $k_p \\cdot$ 0.59 deg, 5.7 to 7.4 mN m, therefore moves nothing, "
        "and the index and middle yaws stick within 23 to 27 mN m of static friction. The servo's position loop runs in "
        "its firmware. The bus delivers a new target to each servo at 50 Hz, against the 1 ms loop of this page.</p>")


def sec_limits(N):
    return (
        "<p>The comparison covers 23 configurations at three seeds, on the CPU only. The tracking-score picks were "
        "chosen on the position servos. The torque plant's model of the hand is exact: $M$, $c$, $g$ and $b$ are the "
        "simulator's own, and no torque sensing or delay was modelled. Two loop frequencies and one damping ratio were "
        f"run. {sum(v['diverged'] for v in N['task'].values())} rollouts of layout (70, 110) mm on the 10 and 13 mm "
        "cylinders diverged and score 0.</p>")


def sec_future():
    return (
        "<ul>"
        "<li>Torque limit. Rerun <code>hom_torque.py compare</code> with <code>TAU_MAX</code> 0.515 N m, the STS3032's, "
        "on both plants. If the torque plant then scores more than 0.03 above the position servos on the twist or the "
        "roll, the torque control was held back by the 0.18 N m limit.</li>"
        "<li>Layouts picked under torque control. Re-pick each size's layout by the torque plant at 63 rad/s from the "
        "tracking page's short lists (<code>hom_track_films.py</code> with <code>hom_torque.TorquePlant</code>). If "
        "the picks score within 0.03 of the position servos' picks, the plant does not change which layout is best.</li>"
        "<li>Contact under torque control. Add a force loop on each pad's measured force to the torque plant's "
        "$J^\\top F n$ term and rerun the twist. The force loop works if the twist's three-pad share returns to the "
        "position servos' 0.40 or above.</li>"
        "<li>Emulation on the built hand. Run the torque plant's law at 50 Hz as position offsets $\\tau/k_p$ with the "
        "0.59 deg dead band in the CPU plant before any bench run. If the tracking scores fall below the position "
        "servos', the law does not transfer to the SCS0009.</li>"
        "</ul>")


def sec_files():
    return ("<ul><li><code>scripts/hom/hom_torque.py</code>: <code>TorquePlant</code>, <code>PositionPlant</code>, "
            "<code>compare</code>, <code>report</code>, <code>film</code>; rows "
            "<code>rows/20261110-torque_compare.jsonl</code></li>"
            "<li><code>scripts/hom/hom_tasks.py</code>: <code>rollout</code> calls <code>plant.set_hom</code> with the "
            "controller's terms of each period</li>"
            f"<li>Layouts and sizes: <a href=\"{TRACK}\">the tracking page</a>, Table 3 "
            "(<code>rows/20261109-track_picks_sv25.json</code>)</li>"
            "<li>Films: <code>media/20261110-film_torque_*.mp4</code></li>"
            "<li>Builder: <code>scripts/hom/hom_torque_page.py</code></li></ul>"
            "<p>[1] S. Wang, J. Oh and N. S. Pollard, \"Relative Contact Velocity-Controlled Hand-Object Mechanism for "
            "Dexterous Tool Manipulation\", arXiv 2609.25619v2, 2026.</p>")


def lede(N):
    T_ = N["task"]
    g = (lambda task, a, k: T_[(task, a)][k])
    text = (
        f"With inverse-dynamics torque control of its fingers in place of the SCS0009 position servos, the SR2 hand "
        f"tracked the HOM twist and roll no better. With a 63 rad/s loop the twist's tracking score on the films' "
        f"layouts was {g('twist', 'torque 63', 'track'):.2f} against {g('twist', 'position', 'track'):.2f}, and the "
        f"roll's {g('roll', 'torque 63', 'track'):.2f} against {g('roll', 'position', 'track'):.2f}. The press's rose "
        f"from {g('press', 'position', 'track'):.2f} to {g('press', 'torque 63', 'track'):.2f}. The object turned more "
        f"steadily: its angular velocity off the commanded axis fell from {g('twist', 'position', 'off'):.0f} to "
        f"{g('twist', 'torque 63', 'off'):.0f} deg/s in the twist. Pad contact was lost more often. In the twist a "
        f"joint was at the servos' 0.18 N m on {100 * g('twist', 'torque 63', 'sat'):.0f} % of physics steps, and on "
        f"{100 * g('twist', 'position', 'sat'):.0f} % with the position servos. On the built hand a torque reaches the "
        f"position-only servo as an offset through its spring, and its 0.59 deg dead band removes torques under 6 to "
        f"7 mN m.")
    summary = (f"Inverse-dynamics torque control of the SR2 fingers under HOM, against the SCS0009 position servos on the "
               f"films' layouts: tracking scores went from {g('twist', 'position', 'track'):.2f} to "
               f"{g('twist', 'torque 63', 'track'):.2f} (twist), {g('roll', 'position', 'track'):.2f} to "
               f"{g('roll', 'torque 63', 'track'):.2f} (roll) and {g('press', 'position', 'track'):.2f} to "
               f"{g('press', 'torque 63', 'track'):.2f} (press) with a 63 rad/s loop; off-axis angular velocity "
               f"{g('twist', 'position', 'off'):.0f} to {g('twist', 'torque 63', 'off'):.0f} deg/s in the twist; joints "
               f"at the 0.18 N m limit on most physics steps.")
    return text, summary


def render(page_text, holder):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page_text)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    holder.append(len(items))
    return P.TEX_RE.sub(lambda m: next(svgs), page_text)


def main():
    rows = load()
    N = numbers(rows)
    sections = [("terms", "Terms and metrics", sec_terms()),
                ("plant", "Position servos and the inverse-dynamics torque plant", sec_plant()),
                ("compare", "Tracking on the films' layouts and object sizes", sec_compare(N)),
                ("contact", "Pad contact, pad force and the torque limit", sec_contact(N)),
                ("films", "Films of three configurations on the three plants", sec_films()),
                ("hardware", "A joint torque on the position servos of the built hand", sec_hardware()),
                ("limits", "Limitations", sec_limits(N)),
                ("future", "Future work", sec_future()),
                ("files", "References and files", sec_files())]
    lede_t, summary = lede(N)
    num = {sid: k for k, (sid, _h, _b) in enumerate(sections, 1)}
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|N\b|s\b)", "\\1&nbsp;\\2", s_))
    sections = [(sid, hd, unit(re.sub(r"SEC\[(\w+)\]", lambda m_: str(num[m_.group(1)]), b))) for sid, hd, b in sections]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{b}</section>' for k, (sid, hd, b) in enumerate(sections, 1))
    byline = ("CPU rollouts with every contact, 2026-11-10: the CAD-model SR2 hand with the v4b fingertip as 1 mm sphere "
              "pads at 2.5 MPa, HOM at a 10 ms period, seeds 0-2. Follows <a href=\"" + TRACK + "\">the tracking "
              "page</a>. Builder: <code>scripts/hom/hom_torque_page.py</code>.")
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": "SR2 finger torque control", "H1": "Inverse-dynamics torque control of the SR2 hand's fingers "
                 "under HOM", "EYEBROW": "HOM control study, 2026-11-10", "LEDE": unit(lede_t), "BYLINE": byline,
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS + texsvg.BLOCK_CSS + ".alg{margin:1em auto;max-width:100%}"}.items():
        page = page.replace("{{" + k + "}}", v)
    n = []
    page = render(page, n)
    svgs = texsvg.render([(b, "block") for b in ALGS], cache_path=TEX_CACHE)
    page = re.sub(r"\x00ALG(\d+)\x00", lambda m_: svgs[int(m_.group(1))], page)
    page = docs_index.tag(page, topic="mechanisms", summary=summary, date="2026-11-10")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {n[0]} formulas, {len(ALGS)} algorithms", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
