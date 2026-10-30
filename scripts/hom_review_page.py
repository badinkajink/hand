#!/usr/bin/env python3
"""Review page of the HOM paper (arXiv 2609.25619v2), of its MuJoCo implementation on the SR2 hand
(src/morphohand/control/), and of the magnetic fingertips as the controller's contact sensor, 2026-10-29.

Owner, 2026-10-29: "create/update a docpage html that more or less covers the original hom-sim paper; how we have
interpreted the paper for control in mujoco with our hands and contact sensing formulation; evaluate how usable our
current real magnetic field sensing fingertips are for transferring sim2real these controls"; "provide illustrative
examples / figures ... be pedagogical". The page names the paper as [1] and nobody else (owner, 2026-10-21).

Reads docs/experiments/20261029-hom_control_review/{rows/sensing.jsonl, media/*.png} (scripts/hom_tasks.py run
--variants ..., scripts/hom_review_figures.py), the films of docs/experiments/20261021-hom_framework/media and a figure of
docs/experiments/20261026-fingertip_calibration/media. Mathematics through scripts/texsvg.py.

    .venv/bin/python scripts/hom_review_page.py
"""
from __future__ import annotations

import base64
import html
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import hom_rows  # noqa: E402
import hom_review_figures as HF  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = ROOT / "docs/experiments/20261029-hom_control_review"
MEDIA = D / "media"
ROWS = D / "rows"
OUT = D / "20261029-hom_control_and_fingertip_sensing.html"
TEMPLATE = ROOT / "scripts/hom_review_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
FW = ROOT / "docs/experiments/20261021-hom_framework"
T_TASK0 = 2.0                                    # s: the task phase starts after the lift (hom_tasks)
CAL = ROOT / "docs/experiments/20261026-fingertip_calibration"
NB = "&#8202;"
REFS = [
    '[1] S. Wang, J. Oh and N. S. Pollard, "Relative Contact Velocity-Controlled Hand-Object Mechanism for Dexterous '
    'Tool Manipulation", arXiv 2609.25619v2, 26 Sep 2026. <a href="https://arxiv.org/abs/2609.25619">arxiv.org/abs/2609.25619</a>; '
    'videos: <a href="https://sunyuw.github.io/hom-sim/">sunyuw.github.io/hom-sim</a>.',
    '[2] S. Wang, J. Oh and N. S. Pollard, "The Foundational Pose as a Selection Mechanism for the Design of Tool-Wielding '
    'Multi-Finger Robotic Hands", IEEE-RAS Humanoids 2025, arXiv 2409.14158.',
    '[3] S. Wang, A. S. Lakshmipathy, J. Oh and N. S. Pollard, "Geodesic tracing-based kinematic integration of rolling and '
    'sliding contact on manifold meshes for dexterous in-hand manipulation", IEEE-RAS Humanoids 2025.',
    '[4] K. M. Lynch and F. C. Park, <i>Modern Robotics: Mechanics, Planning, and Control</i>, Cambridge University Press, '
    '2017, ch. 3 and 5 (twists, the adjoint, Jacobians).',
]
FIG = [0]
FILM = [0]
TAB = [0]


def esc(s):
    return html.escape(str(s), quote=True)


def rel(p):
    import os
    return os.path.relpath(p, D)


def fill(s, **kw):
    """Replace <<key>> tokens (the prose keeps LaTeX braces, so str.format is out)."""
    for k, v in kw.items():
        s = s.replace(f"<<{k}>>", str(v))
    if "<<" in s and ">>" in s:
        left = re.findall(r"<<(\w+)>>", s)
        if left:
            raise KeyError(f"unfilled tokens {left}")
    return s


def data_uri(p, mime):
    return f"data:{mime};base64,{base64.b64encode(Path(p).read_bytes()).decode()}"


def img(path, caption, width=None):
    FIG[0] += 1
    w = f' style="max-width:{width}px"' if width else ""
    return (f'<figure><img src="{data_uri(path, "image/png")}" alt=""{w}>'
            f'<figcaption>Figure {FIG[0]}. {caption}</figcaption></figure>')


def film(path, caption, poster=None):
    if not Path(path).exists():
        return ""
    FILM[0] += 1
    pa = f' poster="{data_uri(poster, "image/jpeg")}"' if poster and Path(poster).exists() else ""
    return (f'<figure><video controls muted loop playsinline preload="metadata"{pa} '
            f'src="{data_uri(path, "video/mp4")}"></video><figcaption>Film {FILM[0]}. {caption}</figcaption></figure>')


def table(head, body, caption, num_from=1):
    TAB[0] += 1
    h = "".join(f'<th class="{"n" if i >= num_from else ""}">{c}</th>' for i, c in enumerate(head))
    b = "".join("<tr>" + "".join(f'<td class="{"n" if i >= num_from else ""}">{c}</td>' for i, c in enumerate(r))
                + "</tr>" for r in body)
    return (f'<div class="tw"><table><caption>Table {TAB[0]}. {caption}</caption><thead><tr>{h}</tr></thead>'
            f'<tbody>{b}</tbody></table></div>')


def render(page_text, holder):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(page_text)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    holder.append(len(items))
    return TEX_RE.sub(lambda m: next(svgs), page_text)


# ------------------------------------------------------------------------------------------------ data
def sensing_rows(name="sensing"):
    """Rows of rows/<name>.jsonl by (task, size, seed, variant): sensing (the CPU runner, seeds 0-2), sensing_gpu (the
    GPU controller, seeds 0-29) or sensing_gpu_val (the GPU controller on the CPU's runs)."""
    p = ROWS / f"{name}.jsonl"
    if not p.exists():
        return []
    out = {}
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[(r["task"], r["d_mm"], r["seed"], r["variant"])] = hom_rows.rescale_metrics(r)
    return list(out.values())


TASKS = ("pinpush", "press", "roll", "twist")
TASK_NAME = {"pinpush": "pin-push", "press": "press", "roll": "roll", "twist": "twist"}
VARIANTS = ("rate50", "notip0", "notip", "tip", "tipbias", "tip3", "tipfl", "tipflb", "cal_notip", "cal_tip", "cal_tipfl")
VAR_TEXT = {
    "rate50": "exact contact point, force and pose",
    "notip0": "no fingertips; exact object pose",
    "notip": "no fingertips; tag tracker",
    "tip": "fingertips as measured; tag tracker",
    "tipbias": "fingertips with a 0.3 N force residual and a 1.5 mm location offset",
    "tip3": "fingertips with three times the measured errors",
    "tipfl": "fingertips as measured, closing a force loop",
    "tipflb": "force loop on fingertips with a 0.3 N force residual",
    "cal_notip": "calibration errors; no fingertips",
    "cal_tip": "calibration errors; fingertips as measured",
    "cal_tipfl": "calibration errors; fingertips closing a force loop",
}


def table_rows():
    """The rows of Tables 4 and 5: the GPU controller's 30 seeds when they exist, else the CPU runner's three."""
    return sensing_rows("sensing_gpu") or sensing_rows()


def gpu_check():
    """The GPU controller (scripts/hom_gpu.py) against the CPU runner (scripts/hom_tasks.py) on the same runs: the
    528 runs of the sensing variants at 50 Hz (rows/sensing.jsonl, rows/sensing_gpu_val.jsonl), and the pinch tasks at
    100 Hz with ten seeds per size (rows/cpu_val_base.jsonl, rows/gpu_val_base.jsonl). Per task: runs, the share with
    the same held outcome, and the median score over held runs on each."""
    out = []
    for label, cn, gn in (("50 Hz, eleven variants, seeds 0-2", "sensing", "sensing_gpu_val"),
                          ("100 Hz, exact sensing, seeds 0-9", "cpu_val_base", "gpu_val_base")):
        c = {(r["task"], r["d_mm"], r["seed"], r.get("variant", "base")): r for r in sensing_rows(cn)
             if r.get("status") == "ok"}
        g = {(r["task"], r["d_mm"], r["seed"], r.get("variant", "base")): r for r in sensing_rows(gn)
             if r.get("status") == "ok"}
        for t in TASKS:
            ks = [k for k in c if k[0] == t and k in g]
            if not ks:
                continue
            same = float(np.mean([bool(c[k].get("held")) == bool(g[k].get("held")) for k in ks]))
            mc = [score(c[k]) for k in ks if c[k].get("held") and score(c[k]) is not None]
            mg = [score(g[k]) for k in ks if g[k].get("held") and score(g[k]) is not None]
            out.append({"set": label, "task": t, "n": len(ks), "same": same,
                        "held_c": sum(bool(c[k].get("held")) for k in ks), "held_g": sum(bool(g[k].get("held")) for k in ks),
                        "med_c": st.median(mc) if mc else None, "med_g": st.median(mg) if mg else None})
    return out


def score(r):
    if r["task"] in ("pinpush", "press"):
        return r.get("theta_rmse_deg")
    return r.get("w_rmse_ratio")


def force_errors(rows, task, var, seeds=None):
    """Per run: the mean over the three pads of |mean pad force - the same run's with exact sensing| (N)."""
    ref = {(r["task"], r["d_mm"], r["seed"]): r for r in rows if r["variant"] == "rate50" and r.get("status") == "ok"}
    out = []
    for r in rows:
        if r["task"] != task or r["variant"] != var or r.get("status") != "ok" or (seeds and r["seed"] not in seeds):
            continue
        b = ref.get((task, r["d_mm"], r["seed"]))
        if b is not None and r.get("F_mean") and b.get("F_mean"):
            out.append(float(np.mean([abs(r["F_mean"][f] - b["F_mean"][f]) for f in b["F_mean"]])))
    return out


def block_medians(rows, task, var):
    """The smallest and largest median force error over blocks of three consecutive seeds."""
    seeds = sorted({r["seed"] for r in rows})
    meds = [st.median(x) for k in range(0, len(seeds) - 2, 3)
            if (x := force_errors(rows, task, var, set(seeds[k:k + 3])))]
    return (min(meds), max(meds)) if meds else (float("nan"), float("nan"))


def summary(rows):
    """Per task and variant: runs, held, median tracking error over held runs, the twist's median ramp, and for the
    two pinch tasks the median force deviation: the mean over the three pads of |mean pad force - the same run's under
    exact sensing| (N), matched by size and seed."""
    out = {}
    ref = {(r["task"], r["d_mm"], r["seed"]): r for r in rows if r["variant"] == "rate50" and r.get("status") == "ok"}
    for t in TASKS:
        for v in VARIANTS:
            rr = [r for r in rows if r["task"] == t and r["variant"] == v and r.get("status") == "ok"]
            if not rr:
                continue
            held = [r for r in rr if r.get("held")]
            sc = [score(r) for r in held if score(r) is not None]
            ramp = [r.get("ramp_three_pads_max_deg") for r in held if r.get("ramp_three_pads_max_deg") is not None]
            dF = []
            for r in rr:
                b = ref.get((t, r["d_mm"], r["seed"]))
                if b is not None and r.get("F_mean") and b.get("F_mean"):
                    dF.append(float(np.mean([abs(r["F_mean"][f] - b["F_mean"][f]) for f in b["F_mean"]])))
            out[(t, v)] = {"n": len(rr), "held": len(held), "med": st.median(sc) if sc else None,
                           "ramp": st.median(ramp) if ramp else None, "dF": st.median(dF) if dF else None,
                           "dF_all": dF}
    return out


# ------------------------------------------------------------------------------------------------ sections
def sec_terms():
    terms = [
        ("Hand-object mechanism (HOM)", "the hand, the object and the generalized contact frames, treated as one "
         "parallel mechanism whose contacts act as joints [1]."),
        ("Twist", r"the angular velocity $\omega$ and linear velocity $v$ of a rigid body, six numbers. The adjoint "
         r"$\mathrm{Ad}_T$ of a pose $T$ is the 6 by 6 matrix that re-expresses a twist in another frame [4]."),
        ("Generalized contact frame (GCF)", "a pair of frames per finger, one fixed to the finger's distal link and "
         "one to the object. When the two bodies are apart, the frames sit at their closest points; when they touch, "
         "at the contact. Axis $x$ is the contact normal and axis $y$ is the finger's joint axis laid into the tangent "
         "plane."),
        ("Relative contact velocities", "the six components of the twist of the finger's GCF relative to the "
         "object's. Spin is the rotation about $x$, and roll the rotations about $y$ and $z$. Separation is the motion "
         "along $x$, positive when the finger closes on the object; slide is the motion along $y$ and $z$."),
        ("Sub-assembly", "a constraint that couples particular fingers' relative velocities or contact forces. The "
         "pinch is two fingers on opposite sides of the object with equal and opposite forces. The pusher is one "
         "finger that turns the object about the pinch axis or supports it."),
        ("Centre of pressure (CoP)", "the force-weighted mean of the points where a fingertip touches the object."),
        ("Servo plant", r"the simulated finger servos. The task study ran on the 1 N m plant, with stiffness $k_p$ = "
         "4 N m/rad and a 1 N m torque limit. The SCS0009 bench servo is 0.72 N m/rad, with a 0.18 N m stall torque."),
        ("Sphere pads", "the printed TPU fingertip block modelled as 1 mm spheres in MuJoCo (pad builder v2), the "
         "project's default fingertip."),
        ("Held", "the object ends no more than 15 mm below its start, and at least two fingers bear load."),
        ("Pinch-angle RMSE", "root-mean-square difference between the cylinder's angle about the pinch axis and the "
         "command (deg), for the pin-push and the press."),
        ("Velocity RMSE per step", "root-mean-square error of the object's angular velocity divided by the 40 deg/s "
         "step it was commanded, the paper's tracking metric; for the roll and the twist."),
        ("Ramp on three pads", "the largest twist (deg) reached at 30 deg/s while all three pads still touch."),
        ("Location error, force error", "for a fingertip on the UR5 bench. The location error is the distance (mm) "
         "between the estimated contact point and the point registered from the robot's pose. The force error is the "
         "error of the estimated normal force, as a percentage of the force-torque sensor's reading."),
        ("Noise-equivalent force", "the force whose signal equals the magnetometers' noise in one record."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"


def sec_paper():
    t = """<p>The HOM paper [1] makes five simulated hands pick up, load and wield a precision screwdriver. Every hand
uses the same controller and the same simple slider commands. Loading means turning the tool from the pose it was picked
up in to the pose it is used in. The hands differ in finger count and kinematics: three commercial hands and two custom
designs, one with three fingers.</p>
<p>The idea comes from machine design. A tool is easy to use when it is mounted on a mechanism that allows the desired
motion. The paper treats the hand and the tool together as such a mechanism, with the contacts as its joints. Two
recurring pieces of that mechanism are named <i>sub-assemblies</i>, after the parts of a machine:</p>
<ul>
<li>The <b>pinch</b>: two fingers press on opposite sides of the tool with equal and opposite forces. A large force fixes
the tool. A force just large enough to hold it lets the tool turn about the line between the contacts, a revolute joint,
or slide along it. Contacts offset from each other turn the tool like the racks of a double rack and pinion.</li>
<li>The <b>pusher</b>: one finger touches the tool and drives it about the pinch axis, or braces it.</li>
</ul>
<p>Every finger is commanded in the same six numbers: how it should roll, spin, slide and separate relative to the
object at their nearest points. A least-squares problem per finger turns those numbers into joint velocities, and an
impedance controller pushes with the commanded force. Because the commands are expressed at the contact, they do not
depend on the hand's kinematics. One slider command produced similar motions on all five hands.</p>
<p>The experiments run in Drake with hydroelastic contact: the simulation at 1 kHz, the controller at 500 Hz, under
teleoperation by sliders.</p>
<ul>
<li>Exp. A moves one finger around a standing cylinder without touching it, one velocity component at a time.</li>
<li>Exp. B picks the cylinder up in a pinch and moves it as the mechanism's output link.</li>
<li>Exp. C turns the pinched cylinder about the pinch axis with a third finger as the pusher, then forms a tripod.</li>
<li>Exp. D picks, loads and wields the screwdriver. Relaxing the pinch lets the tool swing to vertical under gravity.
The pusher aligns it, and repeated twist-and-regrip cycles turn it in a hole.</li>
</ul>
<p>The results are tracking traces with their RMSE. The paper reports no success rates and no hardware runs. It names
contact sensing as the expected difficulty on hardware. It proposes sensors that estimate the centre of pressure at a
high rate, such as intrinsic tactile sensing, fused with motion capture of the object. That proposal is what Section 9
measures against our magnetic fingertips.</p>"""
    t += img(MEDIA / "20261029-pinpush_mechanism.png",
             "The pin-push of our task study as a hand-object mechanism on a 25 mm cylinder (grasp of "
             "<code>design.json</code>, key <code>pinpush_cylinder25</code>). (a) Thumb and index pinch the cylinder; the "
             "dashed line through their contacts is the pinch axis. The middle finger touches the cylinder 52.5 mm along "
             "it. Arrows mark each finger's GCF axes: red the contact normal $x$, green the joint axis $y$. (b) Seen along "
             "the pinch axis, the pinch acts as a revolute joint and the middle finger as its actuator. Eq. (3) commands "
             "the turn through one speed $s$. The pinching pads are drawn at the equator; in the task they sit 15 deg "
             "below it.")
    return t


def sec_twist():
    t = r"""<p>Eq. (1) of the paper describes how a fingertip moves relative to the object. It applies whether or
not the two touch. Place a frame $\{L_1\}$ on the finger's distal link at the point nearest the object. Place a frame
$\{L_0\}$ on the object at the point nearest the finger. When the hand moves, $\{L_1\}$ moves relative to $\{L_0\}$ with
a twist of six numbers. They describe how the finger moves at the contact. It can turn about the normal (spin) or about the two tangent axes
(roll). It can move along the normal (separation) or within the tangent plane (slide).</p>
<p>A wheel on a road is the reference case. Rolling without slipping is a nonzero roll with zero slide and zero
separation: the contact point is the wheel's instantaneous centre of rotation. Skidding is a slide. Lifting off is a
negative separation.</p>
<p>The twist follows from the two bodies' own twists relative to the palm $\{P\}$:</p>
$$\mathcal{V}^{b}_{L_0L_1} = \mathrm{Ad}_{T_{L_1B_1}}\,\mathcal{V}^{b}_{PB_1} - \mathrm{Ad}_{T_{L_1B_0}}\,\mathcal{V}^{b}_{PB_0}. \qquad (1)$$
<p>$\mathcal{V}^{b}_{PB_1}$ is the distal link's twist, expressed in the link's frame $\{B_1\}$, and
$\mathcal{V}^{b}_{PB_0}$ is the object's twist in the object's frame $\{B_0\}$. The adjoint moves a twist to another
frame. For a pose with rotation $R$ and position $p$,</p>
$$\mathrm{Ad}_T = \begin{bmatrix} R & 0 \\ [p]\,R & R \end{bmatrix}, \qquad \mathcal{V} = \begin{bmatrix}\omega \\ v\end{bmatrix},$$
<p>where $[p]$ is the matrix of the cross product with $p$. The lower-left block adds the lever-arm velocity
$\omega \times r$: a body that turns also moves the points away from its axis. The finger's twist comes from its joint
rates $u$ through the link Jacobian, $\mathcal{V}^{b}_{PB_1} = J^{b}_{PB_1}(\theta)\,u$, so Eq. (1) is linear in $u$:</p>
$$\mathcal{V}_{L_0L_1} = A\,u + b, \qquad A = \mathrm{Ad}_{T_{L_1B_1}} J^{b}_{PB_1}, \qquad b = -\mathrm{Ad}_{T_{L_1B_0}}\,\mathcal{V}^{b}_{PB_0}.$$
<p>In our library $A$ and $b$ are computed by <code>HomController.eq1_rows</code>. The object twist in $b$ is the
commanded one, so each finger moves as if the object already followed the command. That makes the object the
mechanism's output link: command its twist, and every finger works out how to move along with it.</p>
<p>The frames are defined apart as well as in contact. Apart, the origins are the witness points, the two bodies'
closest points, and the normal is the direction between them. A finger can therefore be told to approach the object,
travel round it at a constant gap, or roll its fingertip round it, all with the same six numbers. The paper's Exp. A
does exactly that.</p>"""
    return t


def sec_lsq(planar):
    (n1, m1, g1), (n2, m2, g2), (n3, m3, g3) = planar
    t = r"""<p>A finger has three or four joints, and Eq. (1) asks for six velocities. Usually no joint rates satisfy all
six, so the paper takes the joint rates that come closest in a weighted least-squares sense:</p>
$$u_{\mathrm{ref}} = \arg\min_u \sum_i w_i \lVert v_{i,\mathrm{ref}} - v_i(u)\rVert^2 + w_{\mathrm{damp}}\lVert u\rVert^2 + w_{\mathrm{healthy}}\,\lVert k_p(\theta_{\mathrm{healthy}} - \theta) - u\rVert^2 \qquad (2)$$
$$\text{subject to}\quad \theta_{\min} \le \theta + \Delta t\,u \le \theta_{\max},\qquad u_{\min} \le u \le u_{\max}.$$
<p>The sum runs over roll, spin, slide and separation, and $v_i(u)$ is the matching part of $A u + b$. Each term has a
plain job:</p>
<ul>
<li>The weights $w_i$ say which motions matter. Setting $w_{\mathrm{roll}} = 0$ frees the finger to roll, as in the
paper's Exp. B.</li>
<li>The damping term keeps $u$ finite where the Jacobian loses rank, as in damped least squares.</li>
<li>The last term pulls a redundant finger gently toward a recorded comfortable posture $\theta_{\mathrm{healthy}}$, with
$w_{\mathrm{healthy}} \ll w_{\mathrm{damp}} \ll w_i$.</li>
<li>The bounds keep the next joint reference inside the joint limits.</li>
</ul>
<p>The result is integrated into a joint reference, $\theta_{\mathrm{ref}} \leftarrow \theta_{\mathrm{ref}} + \Delta t\,u_{\mathrm{ref}}$.</p>
<p>Angular and linear velocities have different units, so the weights also set an exchange rate. The paper does not
print its weights. Our library divides the linear rows by a length $\ell$ = 20 mm. A slide error of 20 mm/s therefore
costs as much as a roll error of 1 rad/s.</p>
<p>Figure 2 shows Eq. (2) on the smallest case that still has its trade-offs: a planar two-joint finger beside a fixed 25 mm
cylinder. Its links are the SR2 finger's, 30.8 and 26.6 mm long, with a 10.55 mm fingertip radius. In the plane only three
components exist (roll about the joint axis, separation and slide), and the finger has two joint rates. Each panel
commands one component for one second and asks for zero in the other two:</p>
<ul>
<li>Separation at 20 mm/s closes the 6 mm gap. The anti-windup clamp then stops the finger <<g1>> mm from the
cylinder; the clamp zeroes any closing command inside 0.5 mm.</li>
<li>A slide of 20 mm/s yields <<s2>> mm/s of slide, with <<p2>> mm/s leaking into separation: the finger travels round
the cylinder.</li>
<li>A roll of 40 deg/s yields only <<r3>> deg/s. Rolling the fingertip about a point 6 mm away needs the whole finger
to swing, and Eq. (2) trades the remaining roll error against <<p3>> mm/s of separation and <<s3>> mm/s of slide.</li>
</ul>
<p>The paper's Exp. A shows the same coupling on 3-D fingers. Rotation about the GCF's $y$ axis and translation along
its $x$ axis track best, because they map onto flexion. Rotation about $x$ tracks worst, because it needs abduction,
which disturbs every other component.</p>"""
    t = fill(t, g1=f"{g1:.1f}", s2=f"{m2[2]:.1f}", p2=f"{m2[1]:.1f}", r3=f"{math.degrees(m3[0]):.1f}",
             p3=f"{abs(m3[1]):.1f}", s3=f"{abs(m3[2]):.1f}")
    t += img(MEDIA / "20261029-planar_eq1_eq2.png",
             "Eq. (1) and Eq. (2) on a planar two-joint finger with the SR2 link lengths beside a fixed 25 mm "
             "cylinder (grey). Each panel commands one relative velocity for 1 s from the same start, 6 mm from the "
             "cylinder, with zero requested in the other two. Fainter drawings are earlier; dots mark the finger's GCF "
             "origin; dashes trace the fingertip's centre. The box gives the components achieved, averaged over the "
             "second. Computed by <code>scripts/hom_review_figures.py planar</code> (unit weights, $\\ell$ = 20 mm, "
             "no joint limits).")
    return t


def sec_sub():
    m, g, d, mu = 0.0245, 9.81, 0.015, 1.0
    n45 = None
    for N in np.linspace(0.2, 3.0, 28001):
        if 2 * mu * N * 1e-3 * N ** 0.25 / (m * g * d) >= math.cos(math.radians(45)):
            n45 = N
            break
    t = r"""<p>Sub-assemblies add constraints between fingers on top of Eq. (1). The two pinching fingers share one
separation command, so they close on the object together. In contact they press with equal and opposite forces, which
leaves no net force on the object. If the line between the contacts lies inside both friction cones, the pinch holds.</p>
<p>The pusher's command is a rotation of the object about the pinch axis. Let $\{M\}$ be a frame fixed to the object at
the midpoint of the two pinch contacts, with its $x$ axis along the line between them. The pusher finger is given the
object twist</p>
$$\mathcal{V}^{b}_{PB_0} = \mathrm{Ad}_{T_{B_0M}}\begin{bmatrix} s & 0 & 0 & 0 & 0 & 0\end{bmatrix}^{\mathsf T}. \qquad (3)$$
<p>The pinching fingers keep $\mathcal{V}^{b}_{PB_0} = 0$. One number, the speed $s$, now turns the tool about the pinch
axis, as the input of a revolute joint would. The pusher works out its own finger motion from Eq. (1) and Eq. (2).</p>
<p>A pinch acts as a revolute joint, and stays one, only if each pad resists twisting about its own normal. A point
contact has no such torque. A soft pad touches over a patch, and friction across the patch resists the twist with a
torque $\mu N \bar r$, where $\bar r$ is the mean lever arm of the patch. On the real_v1 fingertip, Drake's
hydroelastic model gives $\bar r = 1.00\,\mathrm{mm}\,(N/1\,\mathrm{N})^{1/4}$; the 1 mm sphere pads reproduce it
within 3 to 7 %, as the contact overview derives.</p>
<p>That torque is what makes the paper's loading step work. A tool of mass $m$, whose centre of mass sits $d$ from the
pinch axis, swings down until gravity's torque about the axis equals the two pads' friction torque:</p>
$$m g d \cos\varphi = 2\mu N\,\bar r(N) \quad\Longrightarrow\quad \cos\varphi = \frac{2\mu N\,\bar r(N)}{m g d}.$$
<p>Here $\varphi$ is the angle below horizontal. The friction torque grows as $N^{5/4}$, faster than the force, because
the patch grows with it. Lowering the pinch force therefore lowers the tool smoothly, like a brake. For the 24.5 g tool
at $d$ = 15 mm the law gives 45 deg at <<n45>> N per pad. Figure 3 compares it with two earlier studies. On the two-pad
rig, Drake, the sphere pads and condim 4 passed 45 deg at 1.20, 1.23 and 1.20 N. On the SR2 hand they passed 45 deg at 1.26, 1.28 and 1.33 N. They passed 80 deg at 0.40, 0.41 and 0.45 N, where the
law gives 0.41 N.</p>
<p>Our pin-push uses a different pusher from the paper's demonstration. The demonstration pushes with the back of a
finger, where the SR2 hand has no fingertip. So the middle finger's pad touches the cylinder with its normal along the
pinch axis and sticks. Its tangential motion drags the cylinder round by friction. A drive force of 0.75 N tracked
best; at 0.5 N the friction drive slipped.</p>"""
    t = fill(t, n45=f"{n45:.2f}")
    t += img(MEDIA / "20261029-pinch_brake_law.png",
             "The pinch as a revolute joint with friction: the angle at which a pinched 24.5 g tool hangs (centre of "
             "mass 15 mm from the pinch axis, friction coefficient 1) against the pinch force per pad. The line is the "
             "brake law. Squares are the two-pad rig and circles the SR2 hand, each with Drake hydroelastic, 1 mm "
             "sphere pads and condim 4 (pages of 2026-10-01: <code>20261001-hom_contact_patch</code> and "
             "<code>20261001-hom_hand_brake</code>). Below 0.12 N the pinch cannot support the tool's weight.")
    return t


def sec_filter(fres):
    t = r"""<p>The impedance controller presses at a point and along a direction. Using the GCF origin itself would
be unstable, because the instantaneous contact point jumps with tiny motions and, on hardware, with sensor noise. The
paper therefore treats the measured origin as a noisy reading and filters it. A frame $\{C_j\}$ moves along each body's
surface: $j$ = 1 on the finger, 0 on the object. Each step it moves by the velocity that rolling and sliding kinematics
predict, plus a correction toward the measurement:</p>
$$p_{B_jC_j}[t+1] = \mathrm{trace\_geodesic}\!\left(p_{B_jC_j}[t],\; \dot g_j\,\Delta t\right) \qquad (4)$$
$$\dot g_j = \dot g_{j,\mathrm{ref}} + k_j\,E\left(p_{B_jL_j}[t] - p_{B_jC_j}[t]\right) \qquad (5)$$
$$k_j = k_{\min} + (k_{\max} - k_{\min})\left(1 - e^{-3\lVert \dot g_{j,\mathrm{ref}}\rVert/\alpha}\right) \qquad (6)$$
<p>$\dot g_{j,\mathrm{ref}}$ is the velocity along the surface that the commanded roll and slide imply. It follows from the surfaces' curvatures through the roll-slide kinematics of [3]. $E$ keeps the two tangent components, and the
geodesic trace moves the point along the surface by $\dot g_j \Delta t$.</p>
<p>The gain sets the filter's character. With no commanded rolling, $k_j = k_{\min}$ and the frame drifts slowly
toward the measurement, which smooths it. Our library uses $k_{\min}$ = 5 s$^{-1}$, a time constant of 0.2 s. With fast
commanded rolling, $k_j$ rises toward $k_{\max}$ (100 s$^{-1}$, 10 ms) and the frame follows the measurement closely.
$\alpha$ (20 mm/s) is the speed at which the gain passes 95 % of its range.</p>
<p>In discrete time each step closes the fraction $k\,\Delta t$ of the gap to the measurement. At $k_{\max}$ that
fraction is 1 at 100 Hz and 2 at 50 Hz. Above 1, every step overshoots the measurement and amplifies its noise; at 2
the filter no longer converges. Figure 4 shows both rates on a contact point read with σ = 0.4 mm, the fingertips'
location error at 1.5 N and above:</p>
<ul>
<li>When the 1 s roll is commanded (a), the filter follows within <<a100>> mm RMS at 100 Hz and <<a50>> mm at 50 Hz.</li>
<li>When the same roll happens without a command (b), the gain stays at $k_{\min}$. The filter then lags the contact by
about 0.2 s, an error of <<b100>> mm RMS at either rate.</li>
</ul>
<p>In the four tasks of our study the commanded roll and slide are always zero. The filter therefore runs at
$k_{\min}$ throughout, as a 0.2 s low-pass on the contact point. On the 50 Hz bench, $k_{\max}$ should stay at or below
50 s$^{-1}$.</p>"""
    t = fill(t, a100=f"{fres[(True, 0.01)]:.2f}", a50=f"{fres[(True, 0.02)]:.2f}", b100=f"{fres[(False, 0.01)]:.2f}")
    t += img(MEDIA / "20261029-contact_filter.png",
             "The complementary filter of Eqs. (4) to (6) in one dimension, with our library's gains (5 and "
             "100 s$^{-1}$, $\\alpha$ = 20 mm/s). The contact point rolls 7.4 mm along the pad in one second, a 40 deg/s "
             "roll on a 10.55 mm fingertip, and is measured with 0.4 mm noise (grey). (a) The roll is commanded, so the "
             "gain rises. (b) The roll is not commanded, so the gain stays at its minimum. "
             "<code>scripts/hom_review_figures.py filter</code>.")
    return t


def sec_ours():
    diffs = [
        ("Joint actuation", "torque control; inverse dynamics tracks the joint reference",
         "position servos ($k_p$ 4 N m/rad, 1 N m); the reference is the servo target",
         "the hand's Feetech servos accept position targets only"),
        ("Applied force", "impedance controller; force set as a penetration depth",
         "servo target offsets (Eq. 7); force set in newtons", "a fingertip sensor reports newtons"),
        ("Contact model", "Drake hydroelastic (TetGen meshes)", "MuJoCo, the TPU block as 1 mm sphere pads (builder v2)",
         "the project's default fingertip; it gives the pinch's friction torque"),
        ("Contact point", "centre of pressure of the hydroelastic surface", "force-weighted mean of the pads' contact points",
         "the pads' counterpart; a fingertip sensor's on the bench"),
        ("Closest points", "python-fcl and fcpw on convex decompositions", "<code>mj_geomDistance</code>; analytic block, cylinder and sphere",
         "MuJoCo's own query; analytic bodies"),
        ("Curvature, geodesics", "libigl curvatures; an arc of the osculating circle",
         "exact shape operators; the same arc, projected back onto the surface", "closed form available"),
        ("Weights, units, gains", "not printed", r"$\ell$ = 20 mm, $w_{\mathrm{damp}}$ = $10^{-3}$, $w_{\mathrm{healthy}}$ = $10^{-5}$; "
         r"$k_{\min}$ 5, $k_{\max}$ 100 s$^{-1}$, $\alpha$ 20 mm/s", r"chosen; $\ell$ = 100 mm reproduces Exp. A's ranking"),
        ("Rates", "simulation 1 kHz, controller 500 Hz", "physics 1 kHz, controller 100 Hz (50 Hz checked)",
         "the servo bus closes its loop at about 50 Hz"),
        ("Windup", "anti-windup on separation", r"also $|\theta_{\mathrm{ref}} - \theta| \le$ 0.3 rad",
         "a stopped finger would keep its lag as force"),
        ("Pusher", "normal across the pinch axis; the pusher rolls freely",
         "pin-push: normal along the pinch axis, the pad sticks and drags", "the demonstration pushes with the back of a finger"),
        ("Pinch forces", "equal", "the thumb adds the pusher's force along the pinch axis", "the pusher presses along the axis"),
        ("Palm and object pose", "floating palm; object pose from the simulator", "palm fixed on the gantry; tag tracker on the bench", ""),
    ]
    t = """<p>Our library, <code>src/morphohand/control/hom.py</code> with <code>surfaces.py</code>, implements Eqs. (1) to
(6) as printed and passes 24 tests (<code>tests/test_hom_control.py</code>). The method page lists every difference
from the paper with its reason; Table 1 condenses them. Most follow from two facts about our hand. Its servos accept
position targets only, and its fingertips are modelled as sphere pads in MuJoCo.</p>
<p>A subpage follows the code substep by substep, with the numbers of a twist and a pin-push on the v4b fingertip:
<a href="20261030-hom_controller_code_walkthrough.html">Implementation of the HOM controller on the SR2 hand with the v4b
fingertip</a> (<code>20261030-hom_controller_code_walkthrough.html</code>).</p>"""
    t += table(["Item", "Paper [1]", "Our library", "Reason"],
               [[a, b, c, d_] for a, b, c, d_ in diffs],
               "Differences between the paper's implementation and ours (condensed from the method page, "
               "<code>20261021-hom_controller_method_and_implementation.html</code>, Section 10).", num_from=9)
    t += r"""<p>The largest change is how force is applied. A position servo is a torsional spring of stiffness $k_p$
between its target and its shaft. To press with force $F$ along the contact normal $x_1$, the joints need the torques
$J^{\mathsf T}(F x_1)$, where $J$ is the Jacobian of the contact point. Placing each servo target that many springs'
worth beyond the joint's present reference produces them:</p>
$$\theta_{\mathrm{target}} = \theta_{\mathrm{ref}} + J^{+}(d\,x_1) + \frac{J^{\mathsf T}(F\,x_1)}{k_p} + \frac{\tau_g}{k_p}. \qquad (7)$$
<p>The term $J^{+}(d\,x_1)$ closes a remaining gap $d$ between the reference and the surface; $J^{+}$ is a damped
pseudo-inverse. The last term compensates for the finger's own weight. The force term is small. A 2 N press on a 50 mm lever needs
0.1 N m of joint torque. At $k_p$ = 4 N m/rad, that is a target offset of 0.025 rad, or 1.4 deg.</p>
<p>The same arithmetic makes the force sensitive to geometry. Suppose the model places the surface $\delta$ further away
than it is. The target then sits $\delta$ deeper inside the object, and the spring adds a force of about
$k_p\,\delta/\ell_a^2$ for a lever $\ell_a$. That is 1.6 N per millimetre on the 1 N m plant and 0.29 N per millimetre
on the SCS0009, at a 50 mm lever. A joint off by 1 deg moves a 50 mm finger's tip by 0.9 mm. The force is open loop in
the paper and in our library. On hardware, joint and object-pose errors of this size therefore become force errors of
the same order as the commanded forces. Section 9 simulates that case.</p>
<p>Our task study maps four demonstrated tasks onto the hand (pages of 2026-10-21). With 25 g objects and three seeds
per size, the results were:</p>
<ul>
<li>In the pin-push, the hand held the cylinder in 11 of 12 runs. The pinch-angle RMSE was 2.5 to 4.4 deg on the 16 to
40 mm cylinders.</li>
<li>In the press, the hand held it in 7 of 12 runs.</li>
<li>In the roll, the hand held the sphere in 12 of 12 runs, at 0.50 to 0.69 velocity RMSE per step.</li>
<li>In the twist, the hand held it in 12 of 12 runs. A finger link pressed the sphere with up to 14.6 N during the
ramp.</li>
</ul>
<p>A sampling planner (pinned iCEM) tracked the pinch angle to 0.5 deg. It did so by squeezing with 14 to 22 N per pad,
against 4 N under HOM.</p>
<p>A batched version of this controller runs the four tasks and the sensing variants on MuJoCo-Warp, thousands of
runs at once (<code>scripts/hom_gpu.py</code>). Tables 3 and 4 come from it; Section 9 compares it with this library on
the same runs. The foundational-pose review page uses it to score thousands of hand layouts
(<code>docs/experiments/20261029-foundational_pose_review/</code>).</p>"""
    fl = []
    for stem, cap in (("20261021-hom_pinpush_cylinder25_base", "Pin-push of a 25 mm cylinder under HOM: thumb and index "
                       "pinch, the middle finger drags the cylinder about the pinch axis (seed 0)."),
                      ("20261021-hom_press_cylinder25_base", "Press: the middle finger's pad, 30 deg below the equator, "
                       "lifts its side of the pinched 25 mm cylinder (seed 0)."),
                      ("20261021-hom_roll_sphere40_base", "Roll of a 40 mm sphere along the thumb's pad, steps of "
                       "40 deg/s (seed 0)."),
                      ("20261021-hom_twist_sphere40_base", "Twist of a 40 mm sphere about the vertical, steps and a "
                       "30 deg/s ramp (seed 0).")):
        p = FW / "media" / f"{stem}.mp4"
        fl.append(film(p, cap + " From the task study, <code>20261021-hom_framework/media</code>.",
                       poster=MEDIA / f"{stem}.jpg"))
    t += '<div class="films">' + "".join(fl) + "</div>"
    return t


def sec_sensing():
    t = r"""<p>Per finger and per control period, the controller's input is one contact point and one normal-force
magnitude. The other inputs are the joint angles and the object's pose (<code>hom.Measurement</code>). In simulation
these come from the sphere pads. Let the pads of one fingertip touch the object at points $p_k$ with normal forces
$f_k$; then</p>
$$p_{\mathrm{CoP}} = \frac{\sum_k f_k\, p_k}{\sum_k f_k}, \qquad F_n = \sum_k f_k .$$
<p>A finger whose $F_n$ exceeds 0.02 N counts as touching. Its GCF then comes from the centre of pressure, projected
onto the analytic fingertip and object surfaces (<code>Mirror.gcf_from_cop</code>). The normal is the mean of the
fingertip's outward normal and the reversed object normal. Below 0.02 N, a finger's GCF comes from the witness points of the controller's own model
(<code>Mirror.gcf_witness</code>). That model is a kinematic copy of the hand, posed with the measured joint angles and
object pose. The force reading also decides whether the anti-windup clamp applies. It enters the force
command only through an optional force loop, which adds a closing speed proportional to the force error and is off by
default.</p>
<p>On the bench the same inputs have physical sources. The deployment adapter <code>hom_bench.py</code> runs at 50 Hz.
It takes the servo angles from the bus and the object's pose from the AprilTag tracker (30 frames/s; 0.03 mm and
0.017 deg RMS on a static target). Optionally it takes a centre of pressure and a normal force per finger. Without them
every GCF comes from witness points, and the force is set by Eq. (7) alone.</p>
<p>The magnetic fingertip is the candidate for that optional input. It is a printed TPU pad with four N52 magnets above
five three-axis magnetometers. A press at point $p$ on the pad with force $f$ moves the magnets. At a given point, the change
$b$ in the fifteen field readings is close to linear in the force: $b \approx G(p)\,f$. The estimator picks
the most likely contact point over the pad and solves for the force there; the fingertip overview derives it. The
point and force come out in the fingertip's frame. The finger's forward kinematics transform them into the palm frame, where they become $p_{\mathrm{CoP}}$ and $F_n$.</p>"""
    t += img(CAL / "media/20261026-error_by_level.png",
             "Median location error of the magnetic fingertip against the force of the press, on three tips calibrated "
             "on the UR5 between 22 and 25 October (sessions sphere2, cb1t and cb2t). Solid lines are the trained "
             "Gaussian-process model on held-out presses; dashed lines are the untrained physical model. From the "
             "fingertip calibration page, <code>20261026-fingertip_calibration</code>; the four-magnet v3 tips of 28 and "
             "29 October reach 0.26 and 0.29 mm on the same sphere indenter.")
    return t


# ---------------------------------------------------------------------------------------------- evaluation
def req_table():
    rows = [
        ["Contact point", "GCF origin; the impedance's action point",
         "0.25-0.53 mm at 1.5 N and above; 0.59-0.75 mm under 1 N; 1.10-1.60 mm at 0.5 N and under (each tip trained on "
         "its own presses)", "sufficient (Table 4)"],
        ["Normal force", "touching (0.02 N in simulation); the optional force loop",
         "5-6 %; noise-equivalent force 62-83 mN; display floor 0.25 N", "sufficient for touch; force loop untested on hardware"],
        ["Normal direction", "not used: the GCF normal comes from the surfaces", "2.2 deg", "not needed"],
        ["Shear, slip", "not used; the pusher's drag assumes sticking", "0.12-0.20 N when trained with slides; slip not "
         "evaluated", "not used"],
        ["Torsion", "not used; the pinch relies on the pads' torsional friction", "not recovered", "not used"],
        ["Rate, latency", "50 Hz on the bench", "375 records/s on one board; 152-163 records/s on three boards through a "
         "multiplexer; latency not measured", "rate sufficient; latency unknown"],
        ["Drift", "a phantom force reads as contact", "a press leaves about 0.3 N for minutes; warm-up reads 1-2 N; the "
         "Earth's field turning with the finger moved the zero up to 71 &micro;T", "open on a moving finger"],
        ["Neighbouring fingertips", "every fingertip reads independently", "facing tips at 7-11 mm read 0.31-0.64 N, "
         "0.12-0.23 N after the model; side by side not measured", "open"],
        ["Coverage", "the GCF of the distal link", "face 53-89 % of presses within 1 mm; flanks 28 % within 1.5 mm; no "
         "sensing on links", "links press objects in the twist"],
        ["Calibration", "one model per fingertip", "10.7-12.6 h of UR5 time per tip; another tip's model is 1.0-2.5 mm "
         "off", "costly; three tips needed"],
        ["Integration", "an input of <code>hom_bench.py</code>", "no tip on the hand; the estimator is not connected "
         "to the adapter", "not started"],
    ]
    return table(["Quantity", "Use in the controller", "Fingertips, measured on the UR5", "Status for HOM"], rows,
                 "What the controller uses against what the fingertips measured. Fingertip numbers from the calibration, "
                 "drift, crosstalk and overnight pages of 22 to 29 October and the printed-fingertip overview.",
                 num_from=9)


VAR_SHORT = {"rate50": "Exact sensing", "notip0": "No tips, exact pose", "notip": "No tips", "tip": "Tips",
             "tipbias": "Tips, 0.3 N and 1.5 mm bias", "tip3": "Tips, 3x errors", "tipfl": "Tips, force loop",
             "tipflb": "Tips, force loop, 0.3 N bias", "cal_notip": "No tips", "cal_tip": "Tips",
             "cal_tipfl": "Tips, force loop"}


def sim_table(S, variants, caption):
    nr = max((x["n"] for x in S.values()), default=12)
    head = ["Variant", f"Held, of {nr}:<br>pin-push, press,<br>roll, twist", "Pin-push<br>RMSE<br>(deg)",
            "Press<br>RMSE<br>(deg)", "Roll<br>RMSE<br>/step", "Twist<br>RMSE<br>/step", "Twist<br>ramp<br>(deg)",
            "Pin-push<br>force<br>error (N)", "Press<br>force<br>error (N)"]
    body = []
    for v in variants:
        if not any((t, v) in S for t in TASKS):
            continue
        row = [VAR_SHORT[v], ", ".join("" if S.get((t, v)) is None else str(S[(t, v)]["held"]) for t in TASKS)]
        for t in TASKS:
            x = S.get((t, v))
            row.append("" if x is None or x["med"] is None else (f"{x['med']:.2f}" if t in ("roll", "twist")
                                                                 else f"{x['med']:.1f}"))
        x = S.get(("twist", v))
        row.append("" if x is None or x["ramp"] is None else f"{x['ramp']:.0f}")
        for t in ("pinpush", "press"):
            x = S.get((t, v))
            row.append("" if x is None or x["dF"] is None else ("0" if v == "rate50" else f"{x['dF']:.2f}"))
        body.append(row)
    return table(head, body, caption)


def sec_eval(S):
    def g(t, v, k="med", f="{:.1f}"):
        x = S.get((t, v), {}).get(k)
        return "?" if x is None else f.format(x)

    def h(t, v):
        x = S.get((t, v))
        return "?" if x is None else f"{x['held']} of {x['n']}"

    rows = table_rows()
    nrun = S.get(("pinpush", "rate50"), {}).get("n", 12)
    nseed = len({r["seed"] for r in rows})
    pin_cal = [r for r in rows if r["task"] == "pinpush" and r["variant"] == "cal_notip" and r.get("F_mean")]
    pin_ref = [r for r in rows if r["task"] == "pinpush" and r["variant"] == "rate50" and r.get("F_mean")]
    p90 = lambda v, t: float(np.percentile(S[(t, v)]["dF_all"], 90)) if S.get((t, v), {}).get("dF_all") else float("nan")
    G = gpu_check()
    gsame = (sum(x["same"] * x["n"] for x in G if x["set"].startswith("50")) /
             max(sum(x["n"] for x in G if x["set"].startswith("50")), 1))
    cpu_s = summary(sensing_rows())
    cpu3 = (cpu_s.get(("pinpush", "cal_notip")) or {}).get("dF") or float("nan")
    blk = block_medians(table_rows(), "pinpush", "cal_notip")
    t = fill(r"""<p>For this controller the fingertips' value is force, the one input that its model cannot supply. In
simulation, the contact point they would add changes little: with the joint angles and the object's pose, the
controller's own witness points track as well. The force is different. On position servos it is set open loop by Eq.
(7), so calibration errors of 1 deg per joint and 1 mm in the object's pose make the pinch force wrong by a median
<<dfp>> N on the pin-push. A loop closed on the fingertips' measured force brings that to <<dfpl>> N. Table 2 sets the
controller's needs beside what the fingertips measured on the UR5.</p>""",
             dfp=g("pinpush", "cal_notip", "dF", "{:.2f}"), dfpl=g("pinpush", "cal_tipfl", "dF", "{:.2f}"))
    t += req_table()
    t += r"""<p>The simulation separates what each source of sensing contributes. It runs the four tasks of the task
study at 50 Hz, the bench's rate, with the same sizes and seeds. The fingertip model (<code>hom_tasks.tip_reading</code>)
corrupts each tip contact's true centre of pressure in three ways:</p>
<ul>
<li>The point gets a fixed offset of 0.4 mm and random noise. The noise's σ is 0.4 mm at 1.5 N and above, 0.7 mm from
0.5 N and 1.3 mm below.</li>
<li>The force gets a fixed gain error (σ 5 %) and 0.07 N of noise.</li>
<li>A contact that reads under 0.25 N is dropped, the floor of the fingertip viewer.</li>
</ul>
<p>Every measurement arrives 40 ms late, and the tag tracker adds σ 0.5 mm and 1 deg to the object's pose.</p>"""
    t += fill(r"""<p>The tables come from the GPU controller (<code>scripts/hom_gpu.py</code>) with <<ns>> seeds per size,
<<nr>> runs per task and variant. It draws each run's calibration offsets and fingertip errors as the CPU runner does,
from the same seeds; only the noise of each control period differs. On the CPU runner's 528 runs it gives the same held
outcome in <<sa>> % of runs (Table 3). Its medians match the CPU's within the spread between seeds. That spread is large
for the force error under calibration errors. Its median over the CPU's three seeds is <<c3>> N on the pin-push; over 30
seeds it is <<c30>> N, and blocks of three seeds give medians from <<blo>> to <<bhi>> N.</p>""",
              ns=nseed, nr=nrun, sa=f"{100 * gsame:.0f}", c3=f"{cpu3:.2f}", c30=g("pinpush", "cal_notip", "dF", "{:.2f}"),
              blo=f"{blk[0]:.2f}", bhi=f"{blk[1]:.2f}")
    vb = []
    for x in G:
        nd = 1 if x["task"] in ("pinpush", "press") else 2
        vb.append([x["set"], TASK_NAME[x["task"]].capitalize(), str(x["n"]), f"{x['held_c']} / {x['held_g']}", f"{100 * x['same']:.0f}",
                   "" if x["med_c"] is None else f"{x['med_c']:.{nd}f}", "" if x["med_g"] is None else f"{x['med_g']:.{nd}f}"])
    t += table(["Comparison", "Task", "Runs", "Held,<br>CPU / GPU", "Same<br>held (%)", "Score,<br>CPU", "Score,<br>GPU"], vb,
               "The GPU controller against the CPU runner on the same runs: the task study's grasps, four sizes per "
               "task. At 50 Hz the eleven sensing variants with seeds 0 to 2; at 100 Hz exact sensing with seeds 0 to 9. "
               "Score: the median over held runs of the pinch-angle RMSE (deg) or the velocity RMSE per step. Rows "
               "<code>rows/sensing.jsonl</code>, <code>sensing_gpu_val.jsonl</code>, <code>cpu_val_base.jsonl</code>, "
               "<code>gpu_val_base.jsonl</code>.")
    t += "<p>Table 4 gives the outcomes of the sensing variants.</p>"
    t += sim_table(S, VARIANTS[:8], f"The four tasks under simulated sensing at 50 Hz on the GPU controller: {nrun} "
                   f"runs per task (four sizes, {nseed} seeds). Held runs out of all runs; medians over held runs of the "
                   "tracking errors and of the twist's ramp on three pads, integrated over the 20 ms control period. The "
                   "force error is the mean over the three pads of the difference in mean pad force from the same run "
                   "with exact sensing, median over runs. Rows in <code>rows/sensing_gpu.jsonl</code>.")
    t += fill(r"""<p>Three results stand out.</p>
<ul>
<li>Without fingertips and with the exact object pose, tracking stays close to exact sensing. The pin-push error is
<<a1>> deg against <<a0>>, the press <<b1>> deg against <<b0>>, and the roll and the twist are unchanged (<<c1>> and
<<d1>>). When the model is right, contact frames from its witness points work as well as those from the pads' centre of
pressure.</li>
<li>The tag tracker's noise and delay raise the roll and twist errors to <<c2>> and <<d2>> without fingertips, and to
<<c3>> and <<d3>> with them. The fingertips do not change this.</li>
<li>The measured fingertip errors leave margin. Three times those errors raise the pin-push error from <<a3>> to <<a4>>
deg. A 0.3 N force residual with a 1.5 mm offset raises it to <<a5>> deg.</li>
</ul>
<p>Closing the force loop (variant <code>tipfl</code>: 10 mm/s of closing speed per newton of force error) widens the
twist's ramp on three pads from <<r3>> to <<r6>> deg. It costs the press: <<hp6>> runs held against <<hp3>> without the
loop.</p>""",
              a0=g("pinpush", "rate50"), a1=g("pinpush", "notip0"), b0=g("press", "rate50"), b1=g("press", "notip0"),
              c1=g("roll", "notip0", f="{:.2f}"), d1=g("twist", "notip0", f="{:.2f}"),
              c2=g("roll", "notip", f="{:.2f}"), d2=g("twist", "notip", f="{:.2f}"),
              c3=g("roll", "tip", f="{:.2f}"), d3=g("twist", "tip", f="{:.2f}"),
              a3=g("pinpush", "tip"), a4=g("pinpush", "tip3"), a5=g("pinpush", "tipbias"),
              r3=g("twist", "tip", "ramp", "{:.0f}"), r6=g("twist", "tipfl", "ramp", "{:.0f}"),
              hp6=h("press", "tipfl"), hp3=h("press", "tip"))
    t += img(MEDIA / "20261029-sensing_variants.png",
             f"Every run of the GPU controller under the sensing variants, {nrun} per column (circles, spread "
             "sideways; bars mark medians). Top four panels: tracking errors of the held runs, with the held count above "
             "each column. Bottom two: the pinch tasks' force error per pad against the same run with exact sensing. Red "
             "columns, right of the dotted line, add calibration errors. <code>scripts/hom_review_figures.py sensing</code>.")
    t += fill(r"""<p>Table 5 adds calibration errors that stay constant within a run. Each joint reports and executes
an angle offset by σ 1 deg, and the tracker's object frame is offset by σ 1 mm per axis. The fingertips' location
estimate cannot see such errors, but their force reading can.</p>
<ul>
<li>Without fingertips, the pin-push held in <<h1>> runs, against <<h1r>> with exact sensing, and the press in <<h2>>
against <<h2r>>. Their pad forces are wrong by a median <<f1>> N on the pin-push and <<f2>> N on the press, and in the
worst tenth of runs by more than <<f1p>> and <<f2p>> N. The thumb pressed with <<th1>> N on average, against <<th0>> N
with exact sensing.</li>
<li>With fingertip contact points but no force loop, the errors stay at <<f3>> and <<f4>> N.</li>
<li>The force loop on the fingertips' force brings the errors to <<f5>> and <<f6>> N. That is close to the runs without
calibration errors (<<f7>> and <<f8>> N).</li>
<li>The loop restores the pin-push's hold: <<hp1>> runs held, against <<hp0>> with the fingertips' contact points alone.
Its pinch-angle error is <<a6>> deg, against <<a7>>.</li>
<li>The same loop drops the sphere in the twist: <<ht>> runs held against <<ht0>> without it. It also costs the press:
<<hq1>> runs held against <<hq0>>. Its gain and authority need tuning per task before it runs on the bench.</li>
</ul>""",
              h1=h("pinpush", "cal_notip"), h2=h("press", "cal_notip"),
              f1=g("pinpush", "cal_notip", "dF", "{:.2f}"), f2=g("press", "cal_notip", "dF", "{:.2f}"),
              f1p=f"{p90('cal_notip', 'pinpush'):.1f}", f2p=f"{p90('cal_notip', 'press'):.1f}",
              th0=f"{np.mean([r['F_mean']['thumb'] for r in pin_ref]):.1f}",
              th1=f"{np.mean([r['F_mean']['thumb'] for r in pin_cal]):.1f}",
              f3=g("pinpush", "cal_tip", "dF", "{:.2f}"), f4=g("press", "cal_tip", "dF", "{:.2f}"),
              f5=g("pinpush", "cal_tipfl", "dF", "{:.2f}"), f6=g("press", "cal_tipfl", "dF", "{:.2f}"),
              f7=g("pinpush", "tipfl", "dF", "{:.2f}"), f8=g("press", "tipfl", "dF", "{:.2f}"),
              ht=h("twist", "cal_tipfl"), ht0=h("twist", "cal_tip"), a6=g("pinpush", "cal_tipfl"),
              a7=g("pinpush", "cal_tip"), h1r=h("pinpush", "rate50"), h2r=h("press", "rate50"),
              hp1=h("pinpush", "cal_tipfl"), hp0=h("pinpush", "cal_tip"), hq1=h("press", "cal_tipfl"),
              hq0=h("press", "cal_tip"))
    t += sim_table(S, VARIANTS[8:], "The four tasks with calibration errors (each joint off by σ 1 deg, the object "
                   "frame off by σ 1 mm per axis, both constant within a run), as in Table 4.")
    t += r"""<p>In simulation, these four tasks tolerate about three times the fingertips' measured static errors. The force reading
corrects the calibration errors of a real hand. Four gaps remain before a transfer:</p>
<ul>
<li>No tip is mounted on the hand, and the estimator is not connected to <code>hom_bench.py</code>.</li>
<li>Three tips need 32 to 38 h of UR5 calibration, and another tip's model is 1.0 to 2.5 mm off.</li>
<li>The latency from press to estimate has not been measured; the variants assume 40 ms.</li>
<li>The zero of a turning fingertip moves with the Earth's field: by up to 71 &micro;T over the UR5 calibration
motions. On the robot's tip, a drift of 53 &micro;T read as a 1.0 N press. That is the size of the force errors the loop corrects. A force loop on a moving finger therefore needs the zero
corrected for the finger's orientation first (Future work, item 2).</li>
</ul>"""
    return t


def sec_limits():
    return """<ul>
<li>The sensing variants model the fingertips by their measured static errors: Gaussian location noise that grows at
light force, a 5 % gain error, 0.07 N noise, a 0.25 N detection floor and a 40 ms delay. Errors on a moving contact, the
drift of a turning finger and neighbouring fingertips' fields were not measured and are not modelled.</li>
<li>Every run uses the tasks' 1 N m plant, with $k_p$ = 4 N m/rad. The SCS0009 is 5.6 times softer, so each millimetre
of geometric error gives a smaller force error (Eq. 7).</li>
<li>The controller's model is the convex fingertip block while the plant uses sphere pads; no variant changes the
friction, mass or servo model.</li>
<li>The tasks are the four demonstrations with 25 g objects; the paper's screwdriver pick, load and wield is not part of
this grid.</li>
<li>Tables 4 and 5 come from the GPU controller: float32 physics, a solver tolerance of 10<sup>-6</sup> with 20
iterations, and pads that meet only the object. Single runs of the pinch tasks differ between it and the CPU runner more
than their medians do (Table 3).</li>
</ul>"""


def sec_mpc():
    """The sampling MPC over HOM's commands (scripts/hom_mpc_gpu.py) against the joint-space MPC of 2026-10-21."""
    mp = ROOT / "docs/experiments/20261029-foundational_pose_review/rows/mpc_v4b"
    run = json.loads((mp / "twist_call280_g0_k64.json").read_text())
    run_e = json.loads((mp / "twist_call280_g0_k64_eager.json").read_text())
    tim = json.loads((mp / "timing.json").read_text())
    tim_e = json.loads((mp / "timing_eager.json").read_text())
    tim_o = json.loads((mp / "timing_eager_preport.json").read_text())
    sh_t = json.loads((mp / "shared_gpu" / "timing_eager_0250.json").read_text())
    sh_r = json.loads((mp / "shared_gpu" / "twist_call280_g0_k64_eager.json").read_text())
    R = run["runs"]
    ramp, mpc = R["hom_ramp"], R["mpc_k64"]
    tr = mpc["trace"]
    t1 = next(x["yaw_deg"] for x in tr if x["t"] >= T_TASK0 + 1.0 - 1e-9)
    pw = float(np.median(mpc["plan_wall_s"]))
    pw_e = float(np.median(run_e["runs"]["mpc_k64"]["plan_wall_s"]))
    gr = {r["k"]: r for r in tim["rows"]}
    ge = {r["k"]: r for r in tim_e["rows"]}
    go = {r["k"]: r for r in tim_o["rows"]}
    fd = json.loads((mp / "twist_call280_g0_fd.json").read_text())["runs"]["mpc_fd"]
    fd_lost = next(x["t"] for x in fd["trace"] if min(x["F"]) <= 0.05) - T_TASK0
    fd_dw = max(abs(p_["knots"][0][0] - 30.0) for p_ in fd["plans"])
    js = []
    for task in ("pinpush", "press", "roll", "twist"):
        rows = [json.loads(x) for x in (FW / "rows" / f"mpc_{task}.jsonl").read_text().splitlines() if x.strip()]
        w = [r["wall_s"] for r in rows if r.get("controller") == "mpc" and r.get("wall_s")]
        if w:
            js.append(float(np.median(w)))
    t = fill(r"""<p>The task study of 2026-10-21 also ran a sampling MPC beside HOM: pinned iCEM over the nine servo targets
at three knots, 64 rollouts of 0.5 s on the CPU every 10 ms (<code>scripts/hom_tasks_mpc.py</code>). Per task it took a median
<<j0>> to <<j1>> s of computing per rollout of 6.5 to 7.8 s, about <<rt0>> to <<rt1>> times real time, and squeezed with
14 to 22 N per pad. <code>scripts/hom_mpc_gpu.py</code> samples HOM's commands instead. Per knot of 0.1 s a plan sets the
sphere's angular velocity about the vertical and a scale of the commanded pad forces: two numbers, against nine servo
targets. The horizon is three knots. A cross-entropy method with two iterations refits the best 12 of the samples, and
the previous plan, shifted by one knot, is kept as a sample. Every sample runs in its own MuJoCo-Warp world under the
batched HOM controller, which turns the commands into servo targets every 10 ms. The plant is a one-world batch of the
same scene, and before each replanning its full state is copied into every sample's world: MuJoCo-Warp data, contact
forces and the controller's own state.</p>
<p>Figure <<f>> shows grid layout 280 with the v4b fingertip. A constant 30 deg/s ramp from the task's start turned
the 40 mm sphere <<r_end>> deg in 3 s and lost a pad at <<r_lost>> deg. The MPC with 64 samples commanded up to 90 deg/s
and lowered the force scale to 0.6. It reached <<m1>> deg in 1 s and then held the sphere at <<m_end>> deg with all
three pads on. On this grasp about 52 deg is the most the sphere turns on three pads, and the MPC reached it in half the
time.</p>
<p>A replanning cost <<pw>> s per 0.1 s of control with the GPU to itself, <<rt>> times real time (Table <<tb>>). At 64
worlds a control period takes <<gp64>> ms: <<p64>> ms for the ten physics steps and <<g64>> ms for the controller. From
the second period of the task phase the controller runs as one captured CUDA graph
(<code>hom_gpu.Batch.control_period</code>). Four changes made it capturable. The fingers' roles enter as float masks
instead of boolean indices, and the 3 x 3 systems are solved in closed form. The pad's closest point takes six steps of
Newton's method on its reformulated equation, and the controller's state is updated in place. The same code run
eagerly takes <<e64>> ms a period and plans at <<rte>> times real time. Before the changes a period took <<o64>> ms. That
controller synchronised with the host ten times a period, so it waited for every other job on the GPU. While the size
sweep shared the GPU, its period took <<sh32>> ms at 32 worlds and it planned at <<rsh>> times real time.</p>
<p>A gradient planner on the same commands did worse. Each step took central differences of the cost over the six plan
numbers (12 perturbed worlds and the nominal one in one batch) and moved the plan against the gradient. It turned the
sphere <<fd_end>> deg and lost a pad <<fd_lost>> s into the task, as the ramp did, at <<fd_rt>> times real time. Its
planned rate stayed within <<fd_dw>> deg/s of 30 deg/s. The cost is the number of periods with a pad lost, a step function of
the plan. Perturbations of 5 deg/s left it unchanged until the planned turn was already past the limit.</p>""",
             j0=f"{min(js):,.0f}", j1=f"{max(js):,.0f}", rt0=f"{6.5 / max(js):.3f}", rt1=f"{7.8 / min(js):.3f}",
             f=FIG[0] + 1, r_end=f"{ramp['yaw_end_deg']:.0f}", r_lost=f"{ramp['yaw_three_pads_max_deg']:.0f}",
             m1=f"{t1:.0f}", m_end=f"{mpc['yaw_end_deg']:.0f}", pw=f"{pw:.2f}", rt=f"{0.1 / pw:.3f}",
             rte=f"{0.1 / pw_e:.3f}", tb=TAB[0] + 1, gp64=f"{gr[64]['ms_per_period']:.1f}",
             g64=f"{gr[64]['ms_parts']['controller_graph']:.1f}", p64=f"{gr[64]['ms_parts']['physics']:.1f}",
             e64=f"{ge[64]['ms_per_period']:.1f}", o64=f"{go[64]['ms_per_period']:.1f}",
             sh32=f"{sh_t['rows'][0]['ms_per_period']:.0f}",
             rsh=f"{0.1 / float(np.median(sh_r['runs']['mpc_k64']['plan_wall_s'])):.3f}", fd_end=f"{fd['yaw_end_deg']:.0f}",
             fd_lost=f"{fd_lost:.2f}", fd_rt=f"{0.1 / float(np.median(fd['plan_wall_s'])):.3f}", fd_dw=f"{fd_dw:.1f}")
    t += img(ROOT / "docs/experiments/20261029-hom_control_review/media/20261030-mpc_twist_call280.png",
             "Twist of the 40 mm sphere on grid layout 280, v4b fingertip: a constant 30 deg/s ramp against the MPC over "
             "HOM's commands (64 samples). Top: the sphere's twist. Middle: the MPC's commanded rate and force scale. "
             "Bottom: pad forces.")
    rows = [[str(k), f"{go[k]['ms_per_period']:.1f}", f"{ge[k]['ms_per_period']:.1f}", f"{r['ms_per_period']:.1f}",
             f"{r['ms_parts']['controller_graph']:.1f}", f"{r['ms_parts']['physics']:.1f}"] for k, r in sorted(gr.items())]
    t += table(["Worlds", "Period, before the changes (ms)", "Period, eager (ms)", "Period, graph (ms)",
                "Controller as a graph (ms)", "Physics, 10 steps (ms)"],
               rows, f"Wall time of one control period on the {tim['gpu'].replace('NVIDIA GeForce ', '')} with no other "
               "job: the batched controller before the changes of this section, the changed one run eagerly, and the "
               "changed one captured as a CUDA graph.", num_from=0)
    p2 = mp / "timing_p2ms.json"
    if p2.exists():                                  # the planner's worlds at a 2 ms physics step (2026-10-30, 12:29)
        g2 = {r["k"]: r for r in json.loads(p2.read_text())["rows"]}
        runs = [("1 ms", 2, 0.1, run), ("2 ms", 2, 0.1, json.loads((mp / "twist_call280_g0_k64_p2ms.json").read_text())),
                ("2 ms", 1, 0.2, json.loads((mp / "twist_call280_g0_k64_p2ms_i1_r2.json").read_text()))]

        def stats(r):
            tr_ = r["trace"]
            t0_ = tr_[0]["t"] - 0.01
            lost = next((x["t"] - t0_ for x in tr_ if min(x["F"]) <= 0.05), None)
            t45 = next(x["t"] - t0_ for x in tr_ if x["yaw_deg"] >= 45.0)
            pw_ = float(np.median(r["plan_wall_s"])) if r.get("plan_wall_s") else None
            return pw_, t45, r["yaw_three_pads_max_deg"], lost, r["yaw_end_deg"]
        S_ = [(ph, it, iv) + stats(d_["runs"]["mpc_k64"]) for ph, it, iv, d_ in runs]
        rmp = stats(ramp)
        a, b, c = S_
        t += fill(r"""<p>A 2 ms physics step in the planner's worlds, five steps per period, cut a 64-world period from
<<p1>> to <<p2>> ms (Table <<tb>>). The physics fell from <<f1>> to <<f2>> ms, and the controller's graph stayed at
<<c2>> ms. The plant kept its 1 ms step. With two iterations every 0.1 s a replanning took <<w2>> s, <<x2>> times real
time. The plans kept their quality: the sphere reached 45 deg in <<h2>> s, against <<h1>> s with the 1 ms planner, and
stopped at <<e2>> deg on three pads. With one iteration every 0.2 s a replanning took <<w3>> s, <<x3>> times real time.
With that slower replanning the plant turned past the three-pad limit. It lost a pad <<l3>> s into the task, at
<<e3>> deg, and turned on to <<z3>> deg.</p>""",
                  p1=f"{gr[64]['ms_per_period']:.1f}", p2=f"{g2[64]['ms_per_period']:.1f}",
                  f1=f"{gr[64]['ms_parts']['physics']:.1f}", f2=f"{g2[64]['ms_parts']['physics']:.1f}",
                  c2=f"{g2[64]['ms_parts']['controller_graph']:.1f}", tb=TAB[0] + 1, w2=f"{b[3]:.2f}",
                  x2=f"{b[2] / b[3]:.2f}", h2=f"{b[4]:.2f}", h1=f"{a[4]:.2f}", e2=f"{b[5]:.1f}", w3=f"{c[3]:.2f}",
                  x3=f"{c[2] / c[3]:.2f}", l3=f"{c[6]:.2f}", e3=f"{c[5]:.1f}", z3=f"{c[7]:.0f}")
        rows2 = [[ph, str(it), f"{iv:.1f}", f"{pw_:.2f}", f"{iv / pw_:.2f}", f"{t45:.2f}", f"{e3:.1f}",
                  "&ndash;" if lost is None else f"{lost:.2f}"] for ph, it, iv, pw_, t45, e3, lost, _z in S_]
        rows2.append(["HOM ramp, 30 deg/s", "&ndash;", "&ndash;", "&ndash;", "&ndash;", f"{rmp[1]:.2f}", f"{rmp[2]:.1f}",
                      f"{rmp[3]:.2f}"])
        t += table(["Planner's physics step", "Iterations", "Replanning interval (s)", "Replanning (s)",
                    "Times real time", "Time to 45 deg (s)", "Turn on three pads (deg)", "First pad lost (s)"], rows2,
                   "The MPC over HOM's commands on grid layout 280 with 64 samples and a horizon of 0.3 s: the planner's "
                   "physics step, its iterations and the interval between replannings. Times from the task's start; the "
                   "replanning is the median wall time.", num_from=1)
    return t


def sec_future():
    return """<ol>
<li>Measure the fingertip's latency from a UR5 press to the estimate (timestamped RTDE pose against the board's record
time). Rerun Table 5's force-loop variants with the measured value in place of the assumed 40 ms.</li>
<li>Calibrate the zero of a moving fingertip. Record the board while the UR5 turns an untouched tip through the finger's
range, and fit the field change as a function of orientation. The fit passes if a 90 deg turn then leaves less than
0.25 N.</li>
<li>Connect the estimator to <code>hom_bench.py</code>'s contacts argument and replay a recorded press session through
<code>hom_bench.simulate</code>. The replay must give the same contact frames as the plant's own contacts within
0.5 mm.</li>
<li>Rerun the calibration-error variants on the SCS0009 plant (0.72 N m/rad, 0.18 N m). Then run them on the chain
study's screwdriver load step, where the brake's angle depends on the pinch force (Figure 3).</li>
<li>Tune the force loop per task on the GPU controller: its gain from 0.0025 to 0.02 m/s per newton and a cap on its
closing speed, 30 seeds per point, with calibration errors (<code>scripts/hom_gpu.py run --layouts design</code>). The
loop is ready for the bench if one setting holds the twist in 95 % of runs while keeping the pin-push's force error
under 0.5 N.</li>
<li>Replan every 0.1 s with one iteration, 32 samples, a 2 ms physics step and a horizon of 0.2 s
(<code>hom_mpc_gpu.py run --k 32 --phys-ms 2 --iters 1</code> with <code>H_KNOTS</code> 2). By the periods of Table 7 a
replanning would take about 0.11 s. The setting is good enough if it runs in real time and still stops at the three-pad
limit of layout 280.</li>
<li>Give the gradient planner a cost it can follow: a penalty that rises smoothly as a pad's force falls below 0.5 N,
in place of the count of periods with a pad lost. It is worth keeping if it then stops at the three-pad limit of layout
280, as the cross-entropy method does, with its 13 worlds per step.</li>
<li>Mount one tip on the index finger and run the pin-push on the bench with and without the
force loop, scoring the
pinch angle from the tags. The claim of Table 5 fails if the force loop does not lower the force spread across runs.</li>
</ol>"""


def sec_files():
    rows = "".join(f"<li>{r}</li>" for r in REFS)
    return f"""<p>References:</p><ul>{rows}</ul>
<p>Files:</p><ul>
<li>This page: <code>{OUT.relative_to(ROOT)}</code>, built by <code>scripts/hom_review_page.py</code> with
<code>scripts/hom_review_page.template.html</code>; figures by <code>scripts/hom_review_figures.py</code>.</li>
<li>Controller: <code>src/morphohand/control/hom.py</code>, <code>surfaces.py</code>, <code>hom_bench.py</code>; tests
<code>tests/test_hom_control.py</code>.</li>
<li>Tasks and the sensing variants: <code>scripts/hom_tasks.py</code> (<code>VARIANTS</code>, <code>tip_reading</code>);
rows <code>{(ROWS / 'sensing.jsonl').relative_to(ROOT)}</code>; run with
<code>.venv/bin/python scripts/hom_tasks.py run --tasks pinpush press roll twist --seeds 0 1 2 --variants rate50
notip0 notip tip tip3 tipbias tipfl tipflb cal_notip cal_tip cal_tipfl --rows-file
../20261029-hom_control_review/rows/sensing</code>.</li>
<li>The GPU controller: <code>scripts/hom_gpu.py</code>; Tables 4 and 5 from <code>.venv/bin/python scripts/hom_gpu.py
run --task pinpush press roll twist --layouts design --variants rate50 notip0 notip tip tipbias tip3 tipfl tipflb
cal_notip cal_tip cal_tipfl --seeds 0 1 ... 29 --out sensing_gpu --rows-dir
docs/experiments/20261029-hom_control_review/rows</code> (rows <code>sensing_gpu.jsonl</code>); Table 3 from the same
command with seeds 0 to 2 (<code>sensing_gpu_val.jsonl</code>) and <code>hom_gpu.py compare</code>. Angles of rows
scored before the evening of 2026-10-29 are rescaled to their controller period on loading
(<code>scripts/hom_rows.py</code>).</li>
<li>Method and task pages of 2026-10-21: <code>docs/experiments/20261021-hom_framework/</code>; contact overview
<code>docs/overviews/contact_model.html</code>; printed-fingertip overview <code>docs/overviews/printed_fingertips.html</code>;
calibration page <code>docs/experiments/20261026-fingertip_calibration/</code>.</li>
</ul>"""


# ------------------------------------------------------------------------------------------------ assemble
def assemble(sections, title, h1, eyebrow, lede, byline, summary):
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>'
                     for k, (sid, hd, body) in enumerate(sections, 1))
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": title, "H1": h1, "EYEBROW": eyebrow, "LEDE": lede, "BYLINE": byline, "TOC": toc,
                 "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    n = []
    page = render(page, n)
    page = docs_index.tag(page, topic="mechanisms", summary=summary, date="2026-10-29")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {n[0]} formulas", flush=True)


def main():
    th0 = np.radians([20.0, 40.0])
    _, pad0 = HF.planar_fk(th0)
    al = math.radians(-20.0)
    R = 12.5
    C = pad0 + (HF.RHO + R + 6.0) * np.array([math.cos(al), math.sin(al)])
    planar = []
    for vref in ((0.0, 20.0, 0.0), (0.0, 0.0, 20.0), (math.radians(40.0), 0.0, 0.0)):
        run = HF.planar_run(vref, th0, C, R, T=1.0)
        planar.append((vref, np.array([s["v_ach"] for s in run]).mean(0), run[-1]["gap"]))
    rng = np.random.default_rng(4)
    fres = {}
    speed = math.radians(40.0) * HF.RHO
    for commanded in (True, False):
        for dt in (0.01, 0.02):
            tt = np.arange(0.0, 3.0, dt)
            truth = np.clip(tt - 1.0, 0, 1.0) * speed
            gref = np.where((tt >= 1.0) & (tt < 2.0), speed, 0.0) if commanded else np.zeros_like(tt)
            meas = truth + rng.normal(size=tt.size) * 0.4
            est = HF.filt(meas, gref, dt)
            fres[(commanded, dt)] = float(np.sqrt(np.mean((est - truth) ** 2)))
    S = summary(table_rows())
    sections = [
        ("terms", "Terms and metrics", sec_terms()),
        ("paper", "The HOM paper: a hand and a tool as one mechanism", sec_paper()),
        ("eq1", "Relative contact velocities, Eq. (1)", sec_twist()),
        ("eq2", "Joint velocities by least squares, Eq. (2)", sec_lsq(planar)),
        ("eq3", "Pinch and pusher, Eq. (3), and the pinch's friction torque", sec_sub()),
        ("filter", "Contact estimation for the impedance controller, Eqs. (4) to (6)", sec_filter(fres)),
        ("ours", "The MuJoCo implementation on the SR2 hand", sec_ours()),
        ("sensing", "Contact sensing in simulation and on the bench", sec_sensing()),
        ("eval", "The magnetic fingertips as the controller's contact sensor", sec_eval(S)),
        ("mpc", "Model-predictive control over HOM's commands", sec_mpc()),
        ("limits", "Limitations", sec_limits()),
        ("future", "Future work", sec_future()),
        ("files", "References and files", sec_files()),
    ]
    def g(t, v, k="dF"):
        return S[(t, v)][k]
    lede = fill("""The HOM paper [1] commands each finger by how it should roll, spin, slide and separate relative to the
object, and finds the joint velocities by a small least-squares problem. Our MuJoCo version drives the SR2 hand's
position servos and held the object in 11, 7, 12 and 12 of 12 runs of four demonstrated tasks. Per finger, its
fingertip input is one contact point and one normal force. In simulation the contact point matters little: without
fingertips, the controller's own geometry tracks as well. The force matters. Calibration errors of 1 deg and 1 mm put
the open-loop pinch force off by a median <<f1>> N. A loop on the magnetic fingertips' measured force cuts that error to
<<f2>> N. No fingertip is on the hand yet.""", f1=f"{g('pinpush', 'cal_notip'):.2f}", f2=f"{g('pinpush', 'cal_tipfl'):.2f}")
    summary_text = fill("""The HOM controller's MuJoCo version on the SR2 hand tracks as well without fingertip contact points
as with them. Joint and object calibration errors of 1 deg and 1 mm make its open-loop pinch force wrong by a median
<<f1>> N; a loop on the magnetic fingertips' measured force (located to 0.25-0.53 mm, force within about 5 %) brings it
to <<f2>> N, but the same loop dropped the sphere in <<tw>> twist runs.""", f1=f"{g('pinpush', 'cal_notip'):.2f}",
                        f2=f"{g('pinpush', 'cal_tipfl'):.2f}",
                        tw=f"{S[('twist', 'cal_tipfl')]['n'] - S[('twist', 'cal_tipfl')]['held']} of "
                           f"{S[('twist', 'cal_tipfl')]['n']}")
    byline = ("Review and simulation, 2026-10-29. Paper: arXiv 2609.25619v2 [1]. Controller: "
              "<code>src/morphohand/control/hom.py</code>. Hand: the CAD-model SR2 hand on the 1 N m plant with 1 mm "
              "sphere pads. Fingertips: the four-magnet v3 tips calibrated on the UR5. Builder: "
              "<code>scripts/hom_review_page.py</code>. Code walkthrough: "
              '<a href="20261030-hom_controller_code_walkthrough.html">20261030-hom_controller_code_walkthrough.html</a>.')
    assemble(sections, "HOM control and fingertip sensing",
             "Hand-object mechanism control on the SR2 hand and the fingertip sensing it needs",
             "Reorientation mechanisms and control", lede, byline, summary_text)


if __name__ == "__main__":
    main()
