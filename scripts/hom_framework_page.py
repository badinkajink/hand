#!/usr/bin/env python3
"""Pages of the 2026-10-21 HOM controller study: the method page (the HOM paper's controller and its MuJoCo
implementation in src/morphohand/control/) and the results page (four tasks across object sizes, scripts/hom_tasks.py).

Owner, 2026-10-21: "make sure our implementation really matches [the paper's] (with whatever improvements / mujoco
compatibility changes weve made) and has good documentation for me to understand both his work and what we've changed
(try to avoid personal references ...)". The pages name the paper as [1] and nobody else.

Reads docs/experiments/20261021-hom_framework/{design.json, rows/*.jsonl, rl_assessment.json, media/*.mp4}, renders the
mathematics with scripts/texsvg.py, tags both pages for scripts/docs_index.py and applies scripts/retro_style.py.

    .venv/bin/python scripts/hom_framework_page.py
"""
from __future__ import annotations

import base64
import html
import json
import os
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = ROOT / "docs/experiments/20261021-hom_framework"
ROWS = D / "rows"
MEDIA = D / "media"
METHOD = D / "20261021-hom_controller_method_and_implementation.html"
RESULTS = D / "20261021-hom_tasks_across_object_sizes.html"
TEMPLATE = ROOT / "scripts/hom_framework_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
NB = "&#8202;"
PAPER = ('[1] S. Wang, J. Oh and N. S. Pollard, "Relative Contact Velocity-Controlled Hand-Object Mechanism for Dexterous '
         'Tool Manipulation", arXiv 2609.25619v2, 26 Sep 2026. <a href="https://arxiv.org/abs/2609.25619">arxiv.org/abs/2609.25619</a>; '
         'videos: <a href="https://sunyuw.github.io/hom-sim/">sunyuw.github.io/hom-sim</a>.')
FP = ('[2] S. Wang, J. Oh and N. S. Pollard, "The Foundational Pose as a Selection Mechanism for the Design of Tool-Wielding '
      'Multi-Finger Robotic Hands", IEEE-RAS Humanoids 2025 (relevant_papers/pollard_foundational_pose.pdf).')
GEO = ('[3] S. Wang, A. S. Lakshmipathy, J. Oh and N. S. Pollard, "Geodesic tracing-based kinematic integration of rolling and '
       'sliding contact on manifold meshes for dexterous in-hand manipulation", IEEE-RAS Humanoids 2025.')


def esc(s):
    return html.escape(str(s), quote=True)


def relp(p):
    return os.path.relpath(p, D)


def git_short():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def rows(name):
    p = ROWS / f"{name}.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def rows_file(p):
    p = Path(p)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def video(path, caption, n):
    if not Path(path).exists():
        return ""
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return (f'<figure><video controls muted loop playsinline preload="metadata" src="data:video/mp4;base64,{b64}"></video>'
            f'<figcaption>Film {n}. {caption}</figcaption></figure>')


def table(head, body, num_from=1, caption=None, n=None):
    h = "".join(f'<th class="{"n" if i >= num_from else ""}">{c}</th>' for i, c in enumerate(head))
    b = "".join("<tr>" + "".join(f'<td class="{"n" if i >= num_from else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
                for r in body)
    cap = f"<caption>Table {n}. {caption}</caption>" if caption else ""
    return f'<div class="tw"><table>{cap}<thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def render(page_text, items_holder):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(page_text)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    items_holder.append(len(items))
    return TEX_RE.sub(lambda m: next(svgs), page_text)


def assemble(sections, title, h1, eyebrow, lede, byline, out, summary):
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>'
                     for k, (sid, hd, body) in enumerate(sections, 1))
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": title, "H1": h1, "EYEBROW": eyebrow, "LEDE": lede, "BYLINE": byline, "TOC": toc,
                 "SECTIONS": secs, "FOOTER": str(out.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    n = []
    page = render(page, n)
    page = docs_index.tag(page, topic="mechanisms", summary=summary, date="2026-10-21")
    page = retro_style.apply(page)
    out.write_text(page)
    print(f"{out.relative_to(ROOT)}: {out.stat().st_size / 1e6:.2f} MB, {n[0]} formulas", flush=True)


# ================================================================================================ the method page

def method_page():
    S = []
    S.append(("terms", "Terms and symbols", r"""
<dl class="terms">
<dt>Hand-object mechanism (HOM)</dt><dd>The hand, the object and the generalized contact frames treated as one parallel
mechanism [1]. A finger forms a closed chain with the object whether or not it touches it.</dd>
<dt>Sub-assembly</dt><dd>A constraint beyond Eq. (1) that couples the relative contact velocities and contact forces of particular
fingers [1]: the pinch (two fingers with equal separation velocity, and in contact equal and opposite forces) and the pusher (one
finger whose reference object twist turns or moves the object about or along a pinch).</dd>
<dt>Generalized contact frames (GCFs) $\{L_0\}$, $\{L_1\}$</dt><dd>Two frames per finger, $\{L_1\}$ fixed to the finger body
that touches the object (the distal link) and $\{L_0\}$ fixed to the object. Apart, their origins are the bodies' witness points
(the closest points); in contact, the points of each body's surface closest to the contact point. The $x$ axis is the normal.</dd>
<dt>Relative contact twist $\mathcal V_{L_0L_1}$</dt><dd>The twist of $\{L_1\}$ relative to the object, at $\{L_1\}$'s origin
and in its axes, ordered $(\omega_x,\omega_y,\omega_z,v_x,v_y,v_z)$. With $x$ the normal its elements are the spin $v_{\rm spin}$
(rad/s, about the normal), the roll $v_{\rm roll}\in\mathbb R^2$ (rad/s, about the tangent axes), the separation $v_{\rm sep}$
(m/s, along the normal; positive closes the gap) and the slide $v_{\rm slide}\in\mathbb R^2$ (m/s, along the tangent axes).</dd>
<dt>Object body twist $\mathcal V^b_{PB_0}$</dt><dd>The object's twist relative to the palm frame $\{P\}$, in the object's own
frame $\{B_0\}$: $(\omega_b, v_b)$ with $v_b$ the velocity of the frame's origin. In the paper it is a commanded input of the
planner, not a measurement.</dd>
<dt>Reduced-order speed $s$</dt><dd>The rate (rad/s) at which a pusher turns the object about the axis of a pinch, Eq. (3).</dd>
<dt>$u$, $\theta$, $\theta_{\rm ref}$</dt><dd>A finger's joint rates (rad/s), measured joint angles and integrated joint reference
(rad). Here $\theta_{\rm ref}$ is the servo target before the force offsets.</dd>
<dt>$k_p$</dt><dd>The servo's proportional gain, 4 N&#8202;m/rad on the bench-identified plant (torque limit 1 N&#8202;m, joint damping
0.08 N&#8202;m&#8202;s/rad). The paper's healthy-pose gain, also written $k_p$ there, is $k_h$ on this page.</dd>
<dt>$\{C_0\}$, $\{C_1\}$, $\dot g_j$</dt><dd>The contact frames of the estimator, moving on the object's and the finger's surfaces;
$\dot g_j\in\mathbb R^2$ is the velocity of $\{C_j\}$'s origin along its surface (m/s), in its tangent axes.</dd>
<dt>Centre of pressure (CoP)</dt><dd>The force-weighted mean of a finger's contact points on the object. The paper takes it from
Drake's hydroelastic contact surfaces; here from the sphere pads' MuJoCo contacts.</dd>
<dt>RMSE / step</dt><dd>The paper's tracking metric: root-mean-square error of a tracked velocity divided by the commanded step
magnitude (40&#8202;deg/s or 20&#8202;mm/s).</dd>
</dl>"""))

    S.append(("subassemblies", "Sub-assemblies of a hand-object mechanism", r"""
<p>The paper reads human tool use as a sequence of mechanisms that the hand forms with the tool, and names their recurring parts
sub-assemblies [1]. A two-finger antipodal grasp, the <b>pinch</b>, changes function with its contact forces and positions. With
large forces it is a fixed joint. With forces just large enough to hold the tool it is a revolute or a prismatic joint, so the tool
can turn about the pinch axis or slide through the pinch. With the two contacts offset from each other the fingers act as the two
racks of a double rack-and-pinion and turn the tool about its own axis. A single contact, the <b>pusher</b>, acts as an actuator or a
support: it often turns or moves the tool about or along a pinch, and it can press the tool on the environment, which then acts as
the second finger of a pinch.</p>
<p>The screwdriver sequence of the paper uses all of these: a pinch of fingers 1 and 3 picks the tool up as a fixed joint, relaxes to
a revolute joint so gravity swings the tool down, finger 2 pushes it to vertical about the pinch axis, presses it on the table while
fingers 1 and 3 turn it about its axis, and fingers 1 and 3 release, move round and turn again.</p>"""))

    S.append(("gcf", "Generalized contact frames", r"""
<p>Each finger has two GCFs, $\{L_1\}$ on the finger body expected to touch the object and $\{L_0\}$ on the object [1]. When the
bodies are apart the origins are the witness points and the normal axes follow the gradient of the bodies' signed distance; in
contact, the origins are the points of each surface closest to the contact point, and the normals point from the contact point to
the origins, outward from each body. The finger's $x_1$ therefore points from the finger toward the object and $x_0=-x_1$. The
$y$ axes are aligned as closely as the tangent plane allows with the finger body's joint axis, so that a rotation about $y$ is
the finger's own flexion, and $z=x\times y$.</p>
<p>In the library, <code>Mirror.gcf_witness</code> takes the witness points from <code>mj_geomDistance</code> between the TPU
block's convex mesh and the object on a kinematic copy of the scene, and <code>Mirror.gcf_from_cop</code> projects the pads'
centre of pressure onto analytic surfaces of the block (a box with 2.7&#8202;mm rounded edges) and of the object.</p>"""))

    S.append(("eq1", "Relative contact twist, Eq. (1)", r"""
<p>For every finger the paper writes the relative contact twist from the body twists of the finger body $\{B_1\}$ and the object
$\{B_0\}$, both relative to the palm:</p>
$$\mathcal V^b_{L_0L_1}=\mathrm{Ad}_{T_{L_1B_1}}\mathcal V^b_{PB_1}-\mathrm{Ad}_{T_{L_1B_0}}\mathcal V^b_{PB_0}\qquad(1)$$
<p>Both terms are expressed at $\{L_1\}$'s origin and in its axes, so the second term is the velocity of the object's material point
that coincides with $\{L_1\}$'s origin. With $\mathcal V^b_{PB_1}=J^b_{PB_1}u$ the twist is affine in the finger's joint rates. In
world coordinates, with $R_{L_1}$ the GCF's axes, $J_\omega$ and $J_v(p_{L_1})$ the finger's rotational and translational
Jacobians at $\{L_1\}$'s origin, and $\omega_0$, $v_0$ the object's angular velocity and its origin's velocity under the
reference twist,</p>
$$\mathcal V_{L_0L_1}=\underbrace{\begin{bmatrix}R_{L_1}^{\top}J_\omega\\R_{L_1}^{\top}J_v(p_{L_1})\end{bmatrix}}_{A}u
-\underbrace{\begin{bmatrix}R_{L_1}^{\top}\omega_0\\R_{L_1}^{\top}\bigl(v_0+\omega_0\times(p_{L_1}-p_{B_0})\bigr)\end{bmatrix}}_{-b}.$$
<p><code>HomController.eq1_rows</code> returns $A$ and $b$. A test moves the joints and the object by a small step and differences
the material points; Eq. (1) agrees to 1e-4 relative on five random states.</p>"""))

    S.append(("eq2", "Joint velocities by least squares, Eq. (2)", r"""
<p>The planner solves each finger's joint rates from the reference relative contact velocities $v_{i,\rm ref}$ [1]:</p>
$$u_{\rm ref}=\arg\min_u\sum_{i}w_i\lVert v_{i,\rm ref}-v_i(u)\rVert^2+w_{\rm damp}\lVert u\rVert^2
+w_{\rm healthy}\lVert k_h(\theta_{\rm healthy}-\theta)-u\rVert^2\qquad(2)$$
$$\text{subject to}\quad\theta_{\min}\le\theta+\Delta t\,u\le\theta_{\max},\qquad u_{\min}\le u\le u_{\max},$$
<p>with $i$ over roll, spin, slide and separation and $w_{\rm healthy}\ll w_{\rm damp}\ll w_i$. The damping term keeps $u$ finite
near singularities; the healthy-pose term pulls a redundant finger gently toward a recorded posture. A weight set to zero frees that
relative motion: the paper's Exp. B sets $w_{\rm roll}=0$ so the fingers roll on the object, and Exp. C sets $w_{\rm spin}=0$ for the
two pinching fingers so the object can turn between them. The reference $u_{\rm ref}$ is integrated into $\theta_{\rm ref}$, which an
inverse-dynamics controller tracks.</p>
<p><code>solve_eq2</code> stacks the weighted rows and solves the bounded problem with scipy's BVLS. The paper gives neither its
weights nor the units in which angular and linear rows are compared; here every linear row is divided by $\ell=20$&#8202;mm, so
20&#8202;mm/s weighs as much as 1&#8202;rad/s, and $w_{\rm damp}=10^{-3}$, $w_{\rm healthy}=10^{-5}$, $k_h=1$&#8202;s$^{-1}$,
$|u|\le2$&#8202;rad/s. The joint-limit bound is applied to $\theta_{\rm ref}$, the servo target.</p>
<p><b>Anti-windup.</b> When a finger is within a small distance of the object, a positive $v_{\rm sep,ref}$ is clamped to zero, so
an approach stops just before contact and the force comes from the impedance controller [1]; here the distance is 0.5&#8202;mm.</p>"""))

    S.append(("commands", "Commands: relative velocities, object twist, pinch and pusher, Eq. (3)", r"""
<p>Each finger takes either its four relative contact velocities (the paper's Exp. A moves a finger round a cylinder this
way, one element at a time) or the object's reference body twist $\mathcal V^b_{PB_0}$ through Eq. (1) with $v_{i,\rm ref}=0$, so
that the finger moves with the object's material point and the object becomes the mechanism's effector (Exp. B). The two fingers of
a pinch take the same separation velocity and the same applied force. When a third finger turns the object about the pinch axis as
a pusher, the pinching fingers keep $\mathcal V^b_{PB_0}=0$ and the pusher takes</p>
$$\mathcal V^b_{PB_0}=\mathrm{Ad}_{T_{B_0M}}\begin{bmatrix}s&0&0&0_{1\times3}\end{bmatrix}^{\top},\qquad(3)$$
<p>where $\{M\}$ is the pinch frame fixed to the object, with its origin at the midpoint of the two pinch contacts and its $x$ axis
along the line joining them. One scalar $s$ then drives the revolute joint that the pinch forms, as the operator's slider did in the
paper's Exp. C.</p>
<p><code>HandCommand</code> contains these sliders: per finger a mode (free, pinch or pusher), $v_{\rm ref}$, the weights and the applied
force; for the hand the object twist, $s$ and the pinch's fingers. <code>pusher_body_twist</code> is Eq. (3); a test checks that
points on the pinch axis do not move under it. A pinch used as a prismatic joint takes an explicit pusher twist, a translation along
the tool, in place of Eq. (3) (<code>HandCommand.pusher_twist</code>); the pin-push and press tasks use it.</p>"""))

    S.append(("estimation", "Contact estimation and impedance control, Eqs. (4)-(6)", r"""
<p>Each finger applies its contact force through an impedance controller, which needs an action point and a direction. The GCF's
origin jumps with small contact changes, so the paper filters it: frames $\{C_0\}$ and $\{C_1\}$ move on the object's and the
finger's surfaces, driven by the roll-slide contact kinematics of the reference motion and corrected toward the measured GCF origins
[1, 3]:</p>
$$p_{B_jC_j}[t+1]=\mathrm{trace\_geodesic}\bigl(p_{B_jC_j}[t],\ \dot g_j\Delta t\bigr)\qquad(4)$$
$$\dot g_j=\dot g_{j,\rm ref}+k_j\,E\bigl(p_{B_jL_j}[t]-p_{B_jC_j}[t]\bigr)\qquad(5)$$
$$k_j=k_{\min}+(k_{\max}-k_{\min})\bigl(1-e^{-3\alpha^{-1}\lVert\dot g_{j,\rm ref}\rVert}\bigr)\qquad(6)$$
<p>$E$ keeps the tangential elements of a displacement. The gain is small when the reference geodesic speed is small, so $\{C_j\}$
moves smoothly, and above the speed $\alpha$ it exceeds 95&#8202;% of $k_{\max}$, so $\{C_j\}$ stays near the measured point. The
reference geodesic velocities solve the roll-slide kinematics without the normal rows:</p>
$$\begin{bmatrix}S\,\mathrm{Ad}_{T_{C_1C_0}}G_0(K_0)&-S\,G_1(K_1)\end{bmatrix}
\begin{bmatrix}\dot g_{0,\rm ref}\\ \dot g_{1,\rm ref}\end{bmatrix}=S\,\mathcal V_{L_0L_1,\rm ref},$$
<p>with $S$ selecting the roll and slide elements and $G_j(K_j)$ the surface Jacobian at curvature tensor $K_j$. The paper does not
print $G$. With $x$ the outward normal and $(y,z)$ the tangent axes, a frame moving along its surface at $\dot g$ has linear velocity
$(0,\dot g)$, and its normal turns by $dn=K\dot g$, so</p>
$$G(K)=\begin{bmatrix}0&0\\-K_{21}&-K_{22}\\K_{11}&K_{12}\\0&0\\1&0\\0&1\end{bmatrix}$$
<p>(the spin row is set to zero; $S$ drops it). Two tests check the solve against a sphere rolling and sliding on a sphere in closed
form: rolling moves the contact point at $R_0V/(R_0+R_1)$ on both bodies, sliding at $R_0V/(R_0+R_1)$ on the object and
$-R_1V/(R_0+R_1)$ on the finger. The geodesic is traced, as in the paper, along the arc of the osculating circle of the normal
section; the library projects the arc's end back onto the surface, which is exact for a sphere's great circle and a cylinder's
circumference (both tested).</p>
<p><b>Impedance through the servos.</b> The bench's joints are position servos, so the library writes the impedance as offsets of the
servo target. For a commanded force $F$ along $\{C_1\}$'s normal $x_{C_1}$, with $J$ the translational Jacobian at $\{C_1\}$'s
origin and $d$ the remaining gap,</p>
$$\theta_{\rm cmd}=\theta_{\rm ref}+J^{+}\bigl(\max(d,0)\,x_{C_1}\bigr)+\frac{1}{k_p}J^{\top}F\,x_{C_1}+\frac{\tau_g}{k_p}.$$
<p>The second term moves the action point onto the object's surface, as the paper's reference penetration is measured from the
surface; the third presses with $F$ through the servo spring when the object blocks the finger; the last cancels the finger's own
weight. $J^{+}$ is a damped pseudo-inverse (damping 2&#8202;mm/rad).</p>
<p>The paper's filter gains are not printed. Here $k_{\min}=5$&#8202;s$^{-1}$, $k_{\max}=100$&#8202;s$^{-1}$ and
$\alpha=20$&#8202;mm/s; on a static contact with 1&#8202;mm measurement noise the filtered point stays within 0.35 of the raw
error (test). <code>FilterParams.source</code> selects the paper's $S\,\mathcal V_{L_0L_1,\rm ref}$ or, as an option, the twist that
Eq. (2) plans. The choice matters when a task commands the object twist with zero reference relative velocities, as Exp. B does: the
reference geodesic speed is then zero and the gain never leaves $k_{\min}$ (results page, contact estimator settings). The discrete
update moves $k\,\Delta t$ of the way to the measurement each period, so $k_{\max}\Delta t$ must stay well below 1; the paper's
500&#8202;Hz allows $k_{\max}=100$&#8202;s$^{-1}$, the bench's 50&#8202;Hz about a quarter of it.</p>"""))

    S.append(("paper_exps", "The paper's experiments and implementation", r"""
<p>The paper teleoperates five simulated hands in Drake with hydroelastic contact, chosen because a pinch needs friction torque about
its axis, which point contact lacks: Orca V2, Wuji, Sharpa Wave, a three-finger power-grasp hand (Custom A) and a five-finger
anthropomorphic hand (Custom B). The simulation steps at 1&#8202;kHz and the framework at 500&#8202;Hz. Each finger has eight sliders
(the sub-assembly, the applied force as a reference penetration, and the six relative velocities); the hand has six for the object
twist, one for $s$ and six for the palm, a floating base. The weights of Eq. (2) are the same on every hand. There are four
experiments:</p>
<ul>
<li><b>Exp. A</b>, a finger moved round a standing cylinder without contact, one relative-velocity element at a time (40&#8202;deg/s,
20&#8202;mm/s). The roll about $y$ and the separation track best, because they map onto flexion; the spin about $x$ worst, because it
needs ab- or adduction that also moves the other elements.</li>
<li><b>Exp. B</b>, the cylinder picked in a pinch (approach by $v_{\rm sep}$, anti-windup stop, constant force, palm up) and moved as
the effector by object-twist steps with $w_{\rm roll}=0$. Rotation about the pinch axis cannot be produced by the contacts; the fingers
translate the object instead.</li>
<li><b>Exp. C</b>, the pinch as a revolute joint, finger 3 as the pusher driven by $s$: push away from the palm, gravity return, switch
sides, pull, switch sides, tripod, with $w_{\rm spin}=0$ for the pinching fingers.</li>
<li><b>Exp. D</b>, a precision screwdriver picked, loaded by the gravity swing and the pusher, pressed on the table by finger 2 and
turned by fingers 1 and 3 in release-and-turn cycles.</li>
</ul>
<p>The paper reports RMSE traces, no success rates, and names hardware contact sensing as the open problem; its future work proposes
intrinsic tactile sensing for the centre of pressure and fusion with motion capture.</p>"""))

    S.append(("codemap", "Code map", table(
        ["Paper", "Library (src/morphohand/control/)", "Test (tests/test_hom_control.py)"],
        [["GCFs, Sec. III-B", "<code>Mirror.gcf_witness</code>, <code>Mirror.gcf_from_cop</code>, <code>Gcf</code>",
          "<code>test_gcf_convention</code>"],
         ["Eq. (1)", "<code>HomController.eq1_rows</code>", "<code>test_eq1_rows_match_finite_differences</code>"],
         ["Eq. (2)", "<code>solve_eq2</code>", "<code>test_eq2_unbounded_matches_closed_form</code>, <code>test_eq2_respects_bounds</code>"],
         ["Eq. (3), pinch frame", "<code>pusher_body_twist</code>, <code>pinch_frame</code>", "<code>test_pusher_twist_is_rotation_about_pinch_axis</code>"],
         ["Anti-windup", "<code>HomController.step</code>, <code>HomParams.aw_dist</code>", "the tasks' pick (Section 3 of the results page)"],
         ["Roll-slide kinematics", "<code>surface_jacobian</code>, <code>roll_slide_rates</code>", "<code>test_roll_slide_*</code> (three)"],
         ["Eqs. (4)-(6)", "<code>ContactFilter</code>, <code>FilterParams</code>", "<code>test_filter_smooths_static_noise</code>"],
         ["Geodesic tracer", "<code>Surface.geodesic_step</code> (surfaces.py)", "<code>test_geodesic_step_*</code> (two)"],
         ["Closest points, curvature", "<code>Sphere</code>, <code>Cylinder</code>, <code>RoundedBox</code>, <code>Plane</code>, <code>Posed</code>",
          "<code>test_*_queries</code>, <code>test_rounded_box_*</code>, <code>test_plane</code>"],
         ["Impedance", "servo offsets in <code>HomController.step</code>; <code>reanchor</code>", "the press task"]],
        num_from=9, caption="Where each part of the paper's method is implemented and checked.", n=1)))

    diffs = [
        ["Joint actuation", "Torque-controlled joints; inverse dynamics tracks $\\theta_{\\rm ref}$ and $u_{\\rm ref}$",
         "Position servos, $k_p$ 4&#8202;N&#8202;m/rad, 1&#8202;N&#8202;m limit; $\\theta_{\\rm ref}$ is the target",
         "The bench hand's Feetech servos accept position targets only"],
        ["Applied force", "Impedance controller, force set as a reference penetration distance",
         "Servo offsets $J^{+}(d\\,x)+J^{\\top}F\\,x/k_p+\\tau_g/k_p$, force set in newtons",
         "Same function on position servos; newtons are what a fingertip sensor reports"],
        ["Contact model", "Drake hydroelastic, meshes tetrahedralized with TetGen",
         "MuJoCo, the printed TPU block as 1&#8202;mm sphere pads (builder v2)",
         "The project's default fingertip; pads give the pinch's friction torque in MuJoCo"],
        ["Contact point", "Centre of pressure of the hydroelastic surface",
         "Force-weighted mean of the pads' contact points", "The pads' MuJoCo counterpart; a fingertip sensor's on the bench"],
        ["Closest points", "python-fcl and fcpw on CoACD convex decompositions",
         "<code>mj_geomDistance</code> on the block's convex mesh; analytic block, cylinder, sphere",
         "MuJoCo's own query; the project's bodies are analytic"],
        ["Curvature", "libigl vertex-interpolated principal curvatures", "Exact shape operators of the analytic surfaces",
         "Closed form available"],
        ["Geodesic step", "Arc of the osculating circle", "The same arc, then projected onto the surface", "No drift off the surface"],
        ["Weights and units", "Not printed", "Linear rows / $\\ell$, $\\ell=20$&#8202;mm; $w_{\\rm damp}=10^{-3}$, $w_{\\rm healthy}=10^{-5}$",
         "Chosen; $\\ell=100$&#8202;mm reproduces the paper's Exp. A ranking (Section 13)"],
        ["Filter gains", "Not printed", "$k_{\\min}=5$, $k_{\\max}=100$&#8202;s$^{-1}$, $\\alpha=20$&#8202;mm/s", "Chosen"],
        ["Rates", "Simulation 1&#8202;kHz, framework 500&#8202;Hz", "Physics 1&#8202;kHz, controller 100&#8202;Hz (50&#8202;Hz checked)",
         "The servo bus closes its loop at about 50&#8202;Hz"],
        ["Reference windup", "Anti-windup on $v_{\\rm sep}$ only",
         "Also $|\\theta_{\\rm ref}-\\theta|\\le0.3$&#8202;rad, and <code>reanchor</code> when a pusher switches to force",
         "A stopped pusher otherwise keeps its servo-lag offset as force (1.2&#8202;N on the press task)"],
        ["Prismatic pinch", "Revolute pinch with Eq. (3)", "An explicit pusher translation in place of Eq. (3)",
         "The pin-push and press tasks on SR2"],
        ["Palm", "Floating base, six twist sliders", "Fixed on the gantry (on the bench: the UR5e)",
         "Twists relative to the palm equal world twists"],
        ["Object pose", "From the simulator", "From the simulator; on the bench from the tag tracker", "&#8212;"],
    ]
    S.append(("differences", "Differences from the paper", r"""
<p>The library follows the paper's equations and conventions. Table 2 lists every place where it differs, why, and the
setting a comparison with the reference implementation should check first.</p>""" + table(
        ["Item", "Paper [1]", "This library", "Reason"], diffs, num_from=9,
        caption="Differences between the paper's implementation and the library.", n=2)))

    S.append(("earlier", "Earlier HOM code in this repository", r"""
<p>Three earlier controllers in <code>scripts/</code> were written from the paper's first version and are the ones behind the pages of
October 2, 6, 8, 13 and 14. They differ from the paper more than the library does (Table 3). Their results stand as results of those
controllers; new work uses the library.</p>""" + table(
        ["Item", "hom_control.py, hom_chain.py, hom_paper_tasks.py (10-01 to 10-04)", "hom_turn3.py (10-06)",
         "hand_object_scale_sim.py turn (10-08 to 10-14)"],
        [["Normal axis", "Object's outward normal: $v_{\\rm sep}>0$ opens the gap (the docstring says the paper writes object minus "
          "finger; Eq. (1) is finger minus object)", "Same as hom_control", "Same"],
         ["Object term of Eq. (1)", "Object velocity at the object's witness point (differs from Eq. (1) while apart and turning)",
          "Not used: contacts follow the desired tool twist", "Same as hom_turn3"],
         ["Sub-assemblies", "Pinch hinge and pusher scripted per task", "None: all three contacts follow one tool twist",
          "None"],
         ["Roll and spin", "Free in Exp. B (weights 0)", "Weighted at 0.01, not free", "Same as hom_turn3"],
         ["Force", "Servo offset $J^{\\top}F/k_p+\\tau_g/k_p$ at the touch pose", "Velocity loop on the simulator's measured pad "
          "force, $v_{\\rm sep}=K_F(F_i-F_d)$", "Same as hom_turn3"],
         ["Estimator", "None (sphere-cylinder witness points in closed form)", "None (mesh witness points)", "None"],
         ["Additions", "Gravity feedforward, LEAD clamp", "A governor that freezes the reference on force loss or lag",
          "The same governor; post removal"]],
        num_from=9, caption="The earlier HOM controllers and the library.", n=3) + r"""
<p>The velocity loop on measured force is the largest departure: the bench has no calibrated fingertip force, so a controller that
needs it cannot run there yet. The library keeps it as an option (<code>HomParams.force_loop_gain</code>, off by default).</p>"""))

    S.append(("verification", "Verification", r"""
<p><code>tests/test_hom_control.py</code> has 19 tests, all passing at commit @@COMMIT@@:</p>
<ul>
<li>Surfaces: closest points of the rounded block against 60,000 sampled surface points, the block's shape operator on a face (0),
an edge ($1/r$ across, 0 along) and a corner ($1/r$), cylinder side, cap and interior queries, sphere and plane queries, a posed
surface.</li>
<li>Geodesic steps: 10&#8202;mm along a 20&#8202;mm sphere ends on the great circle within 1e-12&#8202;m; round and along a cylinder.</li>
<li>Roll-slide kinematics: rolling (three radius pairs) and sliding of a sphere on a sphere against the closed forms, and rolling
against the contact point found from the moved bodies (0.1&#8202;%).</li>
<li>Eq. (1) against finite differences of a two-joint finger and a free sphere at five random states and object twists (1e-4
relative); the sign of $v_{\rm sep}$.</li>
<li>Eq. (2) against the unbounded weighted least squares, and its joint-limit bound.</li>
<li>Eq. (3): points on the pinch axis do not move.</li>
<li>The filter at rest: gains equal $k_{\min}$, and the filtered contact point's error is below 0.35 of the raw error.</li>
</ul>
<p>Run them with <code>PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_hom_control.py -q</code>.</p>"""))

    S.append(("expa", "Exp. A on the library and the units of Eq. (2)", expa_section()))

    S.append(("tasks", "Tasks run on the library", r"""
<p>The companion page, <a href="@@RESULTS@@">HOM control of four tasks across object sizes</a>, runs the library on the CAD-model SR2
hand: a pin-push and a press on cylinders of 10&#8211;25&#8202;mm (an index-middle pinch with the thumb as pusher on the end cap) and
rolling and twisting of spheres of 30&#8211;63&#8202;mm (the paper's Exp. B on an equator tripod), with films of every size.</p>"""))

    S.append(("limits", "Limitations", r"""
<p>The library has not been run against the reference implementation, so the agreement is with the paper's text, not its code; the
weights, filter gains and units of Table 2 are this library's choices, and Exp. A shows that the units change which elements track.
The filter has been tested on spheres and in the tasks, not against hydroelastic centres of pressure. Exp. D has not been rerun on the
library (the October 2 runs used the earlier code). No part has run on hardware.</p>"""))

    S.append(("future", "Future work", r"""
<ul>
<li>The length scale of Eq. (2): at $\\ell=100$&#8202;mm, which reproduces Exp. A's ranking, every cylinder pick of the tasks fails
(0 of 24 rollouts hold) and rolling degrades (the results page's section on the length scale), so the tasks keep 20&#8202;mm. The scale or weights of the
reference implementation are the first item of the comparison below; a scale that passes both Exp. A and the picks would replace
20&#8202;mm.</li>
<li>Comparison with the reference implementation: the same scripted sliders on one hand in both codes, compared element by element on
the relative contact velocities of Exp. A and the object twist of Exp. B; first the weights and units of Table 2, then which twist
feeds the filter's process model in Exp. B (the reference relative velocities are zero there) and its gains.</li>
<li>A Drake plant behind the same <code>Mirror</code> (as <code>hom_chain.DrakeChainPlant</code> did for the earlier code) to run the
tasks on hydroelastic contact, the paper's model.</li>
<li>The centre of pressure from the magnet fingertips once calibrated, as the paper's future work proposes; until then the bench runs
with witness points from the tracked tool pose.</li>
</ul>"""))

    S.append(("refs", "References and files", f"""
<p>{PAPER}<br>{FP}<br>{GEO}</p>
<p>Code: <code>src/morphohand/control/hom.py</code>, <code>src/morphohand/control/surfaces.py</code>,
<code>tests/test_hom_control.py</code>, <code>scripts/hom_tasks.py</code>; this page: <code>scripts/hom_framework_page.py</code>.
Earlier controllers: <code>scripts/hom_control.py</code>, <code>hom_chain.py</code>, <code>hom_paper_tasks.py</code>,
<code>hom_turn3.py</code>, <code>hand_object_scale_sim.py</code>.</p>"""))

    commit = git_short()
    S = [(sid, hd, body.replace("@@COMMIT@@", commit).replace("@@RESULTS@@", relp(RESULTS))) for sid, hd, body in S]
    lede = (r"The relative contact-velocity controller of the HOM paper [1] is implemented in <code>src/morphohand/control/</code> "
            r"as one library for MuJoCo hands with position servos, in the paper's conventions: generalized contact frames, the "
            r"relative contact twist of Eq. (1), the least squares of Eq. (2), the pinch and pusher sub-assemblies with Eq. (3), "
            r"the anti-windup of the separation velocity and the complementary filter of Eqs. (4)&#8211;(6). Nineteen tests check it "
            r"against closed forms and finite differences (Eq. (1) to 1e-4 relative), and the paper's Exp. A on the CAD-model hand "
            r"gives the paper's ranking (roll about $y$ and separation best) when Eq. (2) weighs 100&#8202;mm/s like 1&#8202;rad/s. It differs from the paper in the 14 places of "
            r"Table 2, mostly where position servos replace torque control and sphere pads replace hydroelastic contact; the three "
            r"earlier HOM controllers in this repository differ more (Table 3).")
    summary = ("The HOM paper's controller (GCFs, Eqs. 1-6, pinch and pusher) implemented as one MuJoCo library for position-servo "
               "hands; 19 tests pass; 14 differences from the paper listed, mostly servos for torque control and pads for "
               "hydroelastic contact; the three earlier controllers differ more.")
    assemble(S, "HOM controller implementation", "Hand-object mechanism control: the method and its MuJoCo implementation",
             "Reorientation mechanisms and control &#183; 2026-10-21", lede,
             f"Built {time.strftime('%Y-%m-%d %H:%M')} at commit {commit}.", METHOD, summary)


def expa_section():
    R = {r["variant"]: r for r in rows("expa") if r.get("status") == "ok"}
    if not R:
        return "<p>Rows pending.</p>"
    names = {"ell5": "5&#8202;mm", "base": "20&#8202;mm (default)", "ell100": "100&#8202;mm", "si": "1&#8202;m (SI units)"}
    order = [v for v in ("ell5", "base", "ell100", "si") if v in R]
    comps = ["spin", "roll_y", "roll_z", "sep", "slide_y", "slide_z"]
    head = ["Length scale $\\ell$"] + ["Spin", "Roll $y$", "Roll $z$", "Separation", "Slide $y$", "Slide $z$"]
    body = [[names[v]] + [f"{R[v]['elements'][c]['gain']:.2f} ({R[v]['elements'][c]['rmse_ratio']:.2f})" for c in comps]
            for v in order]
    film = video(MEDIA / "20261021-hom_expa_cylinder25_base.mp4",
                 "Exp. A on the CAD-model hand, length scale 20&#8202;mm: the thumb moves round the 25&#8202;mm cylinder "
                 "without touching it, one relative-velocity element at a time (the label names the element).", 1)
    b, h = R.get("base"), R.get("ell100")
    return (r"""
<p>The paper's Exp. A moves one finger round a cylinder without contact and steps each element of its reference relative contact
velocity, the others held at zero. On the CAD-model hand the thumb of the scale study's equator grasp of a 25&#8202;mm cylinder plays
that finger; its palmar side faces the cylinder, so its flexion axis lies in the GCF's tangent plane as in the paper. Each element is
stepped +40&#8202;deg/s or +20&#8202;mm/s for 0.3&#8202;s, then the negative step, then 0.3&#8202;s of rest. Table 4 gives the
tracking gain (median of achieved over commanded during the step, from Eq. (1) at the measured joint rates) and, in brackets, the
paper's RMSE / step.</p>""" + table(head, body, num_from=1, n=4,
                                    caption=f"Exp. A on the library: tracking gain (RMSE / step) per element for {len(order)} length scales of Eq. (2).")
            + r"""
<p>The paper reports the roll about $y$ and the separation tracked best and the spin worst. Which elements the least squares tracks
depends on how a linear error is weighed against an angular one, which the paper does not state. At $\ell=100$&#8202;mm the library
gives the paper's ranking at the top: roll about $y$ """ + f"{h['elements']['roll_y']['gain']:.2f} and separation {h['elements']['sep']['gain']:.2f}" + r""".
At the 20&#8202;mm used in the tasks the separation and slides track (""" + f"{b['elements']['sep']['gain']:.2f}, {b['elements']['slide_y']['gain']:.2f}, {b['elements']['slide_z']['gain']:.2f}" + r""")
and the rotations barely move, because flexing the finger about its joint also translates the contact point by tens of millimetres
per radian, which the 20&#8202;mm scale penalises heavily. In SI units the rotations track and the slides do not. The roll about $z$
needs ab- or adduction in this posture and stays below 0.1 at every scale. The finger never touched the cylinder (closest approach
at least 0.1&#8202;mm in every run).</p>
<p>In the pinch-x grasp of the pin-push and press tasks the index and middle touch the cylinder with the sides of their blocks, so
their flexion axes lie along the contact normal and flexion appears as spin; Exp. A run with the index there tracks spin, not the
roll about $y$ (rows in <code>logs/20261021-hom_tasks/old_rows/</code>).</p>""" + film)


# ================================================================================================ the results page

def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def fmt(x, nd=1):
    if x is None:
        return "&#8212;"
    return f"{(x if abs(x) >= 0.5 * 10 ** -nd else 0.0):.{nd}f}"


def results_page():
    des = json.loads((D / "design.json").read_text()) if (D / "design.json").exists() else {}
    R = {t: [r for r in rows(t) if r.get("status") == "ok"] for t in ("pinpush", "press", "roll", "twist")}
    S = []
    n_tab = [0]
    n_film = [0]

    def T(head, body, caption, num_from=1):
        n_tab[0] += 1
        return table(head, body, num_from=num_from, caption=caption, n=n_tab[0])

    def F(path, caption):
        n_film[0] += 1
        return video(path, caption, n_film[0])

    S.append(("terms", "Terms and metrics", r"""
<dl class="terms">
<dt>Held</dt><dd>At the end of the rollout the object is less than 15&#8202;mm below its start and at least two fingers press on it
with at least half its weight each (0.12&#8202;N; the objects weigh 25&#8202;g).</dd>
<dt>Axial displacement (pin-push)</dt><dd>Movement of the cylinder along its own initial axis from the start of the push to the end
of the rollout, in mm; the command is 10&#8202;mm (10&#8202;mm/s for 1&#8202;s).</dd>
<dt>Axial RMSE / step (pin-push)</dt><dd>Root-mean-square difference between the cylinder's axial velocity and the commanded
10&#8202;mm/s during the push, divided by 10&#8202;mm/s.</dd>
<dt>Tilt</dt><dd>Largest angle between the cylinder's axis and its initial axis during the task, in degrees.</dd>
<dt>Wall force (press)</dt><dd>Sum of the normal forces between the cylinder and the wall, averaged over the last 0.5&#8202;s of each
1&#8202;s force step, in N; the commands are 0.5, 1.0, 2.0 and 0.5&#8202;N.</dd>
<dt>Angular RMSE / step (roll, twist)</dt><dd>The paper's metric: the root-mean-square norm of the difference between the object's
angular velocity and the reference while a reference is commanded, divided by the step of 40&#8202;deg/s.</dd>
<dt>Gain of a step</dt><dd>Angle turned during a 0.5&#8202;s step divided by the commanded 20&#8202;deg (1 tracks exactly).</dd>
<dt>Drift</dt><dd>Largest distance of the sphere's centre from its position at the start of the task, in mm.</dd>
<dt>Twist ramp</dt><dd>After the steps, the twist reference runs at 30&#8202;deg/s for 3&#8202;s; the score is the largest angle
turned while all three pads press with more than 0.05&#8202;N.</dd>
<dt>Command rate</dt><dd>Root-mean-square rate of change of the nine servo targets, sampled every 10&#8202;ms from the start of the task,
in rad/s.</dd>
<dt>Command reversals</dt><dd>Sign changes of a servo target's rate per joint per second over the same samples (changes under
$10^{-4}$&#8202;rad ignored); a target that moves steadily has none, one that dithers has many.</dd>
<dt>Angular jitter</dt><dd>Root-mean-square difference between the object's angular velocity and its 100&#8202;ms running mean,
in deg/s.</dd>
<dt>Wall time per run</dt><dd>Seconds of computation for one rollout: one CPU core for HOM, six rollout threads for MPC.</dd>
</dl>"""))

    S.append(("setup", "Hand, plant and tasks", r"""
<p>Every rollout uses the CAD-model SR2 hand (the printed links of 2026-10-13 on the real_v1 base), the bench-identified servo plant
(4&#8202;N&#8202;m/rad, 1&#8202;N&#8202;m, damping 0.08&#8202;N&#8202;m&#8202;s/rad), the printed TPU block as 1&#8202;mm sphere pads
(builder v2), friction 1.0, 1&#8202;ms steps with the elliptic cone at impratio 100, and the library's controller at 100&#8202;Hz with
the filter on. Objects weigh 25&#8202;g. The palm is welded above the object, which rests on a 3&#8202;mm post until the grasp
closes.</p>
<p>The four tasks map the HOM paper's sub-assemblies onto a three-finger hand whose fingers hang from a palm plate. The task names
"cylinder press" and "cylinder pin-push" are read here as a pusher pressing a tool on the environment and a pusher moving a tool
through a pinch used as a prismatic joint; other readings, such as the foundational-pose paper's press and poke [2], need a fourth
finger or a pusher above the tool.</p>
<ul>
<li><b>Pin-push.</b> A 70&#8202;mm cylinder lies along the palm's $x$ axis. Index and middle pinch it across its middle with
1.5&#8202;N each: a prismatic joint whose torsional friction resists the gravity torque as the centre of mass leaves the pinch. The thumb,
a pusher on the end cap, takes a translation of 10&#8202;mm/s along the axis for 1&#8202;s while the pinching fingers keep
$\mathcal V^b_{PB_0}=0$; then the pinch locks at 3&#8202;N.</li>
<li><b>Press.</b> The same grasp, a wall 4&#8202;mm beyond the far end cap. The thumb pushes the cylinder to the wall at 8&#8202;mm/s,
its reference is re-anchored, and it presses through the servo spring with 0.5, 1.0, 2.0 and 0.5&#8202;N for 1&#8202;s each while the
pinch relaxes to 0.6&#8202;N.</li>
<li><b>Roll.</b> A sphere at the circumcentre of the three mounts, an equator tripod with 2.5&#8202;N per pad, all fingers following
the object twist with $w_{\rm roll}=0$ (Exp. B): steps of &#177;40&#8202;deg/s about the two horizontal axes, 0.5&#8202;s each.</li>
<li><b>Twist.</b> The roll grasp; steps of &#177;40&#8202;deg/s about the vertical axis, then the 30&#8202;deg/s ramp.</li>
</ul>
<p>Every task begins with the paper's pick: fingers 8&#8202;mm off the object approach at $v_{\rm sep}=20$&#8202;mm/s until the
anti-windup stops them 0.5&#8202;mm away, the forces turn on, and the post is removed 0.4&#8202;s before the task.</p>"""))

    # grasps
    gb = []
    for key in ("cylinder10", "cylinder16", "cylinder20", "cylinder25", "cylinder32", "cylinder40",
                "sphere30", "sphere40", "sphere50", "sphere63"):
        g = des.get(key)
        if key not in des:
            continue
        if g is None:
            gb.append([key.replace("cylinder", "Cylinder ").replace("sphere", "Sphere ") + "&#8202;mm",
                       "no grasp with every link clear", "", "", ""])
            continue
        clr = g.get("link_clearance_mm", {})
        gb.append([key.replace("cylinder", "Cylinder ").replace("sphere", "Sphere ") + "&#8202;mm",
                   f"{g['x_sep_mm']:g}, {g['y_sep_mm']:g}",
                   f"{1e3 * g['h']:.1f}" if g["h"] < 1 else f"{g['h']:.1f}",
                   f"{min(g['sigma_min_mm'].values()):.1f}",
                   f"{clr.get('touch', float('nan')):.2f} / {clr.get('open', float('nan')):.2f}" if clr else "&#8212;"])
    if gb:
        S.append(("grasps", "Grasps per object size", r"""
<p>Cylinder grasps come from a search over the scale study's tripod layouts, the pinch position and the palm height: the pad targets
must be reachable inside the servos' calibrated ranges with 3&#8202;deg to spare, and on the MuJoCo model every finger geometry but
the tip must stay 0.5&#8202;mm from the cylinder when the block touches and 1.5&#8202;mm when open; among those, the grasp with the
largest smallest pad-Jacobian singular value. No such grasp exists for 32 or 40&#8202;mm cylinders: the index and middle reach across
the cylinder's top with their PIP links. Sphere grasps are the scale study's equator tripods on the layout with the largest smallest
singular value.</p>""" + T(["Object", "Mount spacing x, y (mm)", "Palm height (mm)", "Smallest singular value (mm/rad)",
                            "Clearance touch / open (mm)"], gb, "The grasp of each object size: mount spacings of the "
                            "tripod layout, palm height above the object centre, the least-conditioned finger's smallest pad-Jacobian "
                            "singular value, and the smallest distance of a non-tip link to the object.")))

    # pinpush / press tables
    if R["pinpush"]:
        body = []
        for d in sorted({r["d_mm"] for r in R["pinpush"]}):
            rs = [r for r in R["pinpush"] if r["d_mm"] == d and r.get("variant", "base") == "base"]
            if not rs:
                continue
            body.append([f"{d:g}", f"{sum(r['held'] for r in rs)}/{len(rs)}",
                         " / ".join(fmt(r.get("disp_end_mm")) for r in rs),
                         " / ".join(fmt(r.get("v_rmse_ratio"), 2) for r in rs),
                         " / ".join(fmt(r.get("tilt_max_deg")) for r in rs)])
        films = "".join(F(MEDIA / f"20261021-hom_pinpush_cylinder{d:g}_base.mp4",
                          f"Pin-push, {d:g}&#8202;mm cylinder, seed 0. The thumb (left) pushes the end cap; index and middle pinch.")
                        for d in sorted({r['d_mm'] for r in R['pinpush']}))
        S.append(("pinpush", "Pin-push", T(["Diameter (mm)", "Held", "Axial displacement (mm), seeds 0/1/2",
                                            "Axial RMSE / step", "Tilt (deg)"], body,
                                           "Pin-push across cylinder sizes, base variant, three seeds (object pose jitter of 1&#8202;mm and 2&#8202;deg).")
                  + f'<div class="films">{films}</div>'))
    if R["press"]:
        body = []
        for d in sorted({r["d_mm"] for r in R["press"]}):
            rs = [r for r in R["press"] if r["d_mm"] == d and r.get("variant", "base") == "base"]
            if not rs:
                continue
            per = []
            for k, Fr in enumerate((0.5, 1.0, 2.0)):
                vals = [r["press_steps"][k]["wall_mean"] for r in rs if r.get("press_steps") and len(r["press_steps"]) > k]
                per.append(" / ".join(f"{v:.2f}" for v in vals) if vals else "&#8212;")
            body.append([f"{d:g}", f"{sum(r['held'] for r in rs)}/{len(rs)}",
                         f"{sum(bool(r.get('press_steps')) and any(s['wall_mean'] > 0.05 for s in r['press_steps']) for r in rs)}/{len(rs)}",
                         *per])
        films = "".join(F(MEDIA / f"20261021-hom_press_cylinder{d:g}_base.mp4",
                          f"Press, {d:g}&#8202;mm cylinder, seed 0. The wall is the plate on the right.")
                        for d in sorted({r['d_mm'] for r in R['press']}))
        S.append(("press", "Press", T(["Diameter (mm)", "Held", "Reached the wall", "Wall force at 0.5&#8202;N (N)",
                                       "at 1.0&#8202;N", "at 2.0&#8202;N"], body,
                                      "Press: wall force against the commanded thumb force, seeds 0/1/2.")
                  + f'<div class="films">{films}</div>'))
    for task, name in (("roll", "Roll"), ("twist", "Twist")):
        if not R[task]:
            continue
        body = []
        for d in sorted({r["d_mm"] for r in R[task]}):
            rs = [r for r in R[task] if r["d_mm"] == d and r.get("variant", "base") == "base"]
            if not rs:
                continue
            gains = {}
            for r in rs:
                for s_ in r.get("steps", []):
                    gains.setdefault(s_["step"], []).append(s_["gain"])
            row = [f"{d:g}", f"{sum(r['held'] for r in rs)}/{len(rs)}",
                   " / ".join(fmt(r.get("w_rmse_ratio"), 2) for r in rs)]
            row += [" / ".join(fmt(g, 2) for g in gains.get(k, [])) for k in (["wx +", "wx -", "wy +", "wy -"] if task == "roll" else ["wz +", "wz -"])]
            row.append(" / ".join(fmt(r.get("drift_max_mm")) for r in rs))
            if task == "twist":
                row.append(" / ".join(fmt(r.get("ramp_three_pads_max_deg")) for r in rs))
            body.append(row)
        head = ["Diameter (mm)", "Held", "Angular RMSE / step"] + (
            ["Gain +x", "Gain &#8722;x", "Gain +y", "Gain &#8722;y"] if task == "roll" else ["Gain +z", "Gain &#8722;z"]) + ["Drift (mm)"]
        if task == "twist":
            head.append("Ramp on three pads (deg)")
        films = "".join(F(MEDIA / f"20261021-hom_{task}_sphere{d:g}_base.mp4", f"{name}, {d:g}&#8202;mm sphere, seed 0.")
                        for d in sorted({r['d_mm'] for r in R[task]}))
        S.append((task, name, T(head, body, f"{name} across sphere sizes, seeds 0/1/2.") + f'<div class="films">{films}</div>'))

    # the length scale of Eq. (2)
    ELLV = {"base": "20&#8202;mm", "ell100": "100&#8202;mm", "si": "1&#8202;m"}
    if any(r.get("variant") in ("ell100", "si") for t in R for r in R[t]):
        body = []
        for t in ("pinpush", "press", "roll", "twist"):
            key = {"pinpush": "disp_end_mm", "press": "wall_err_rms", "roll": "w_rmse_ratio", "twist": "w_rmse_ratio"}[t]
            for d in sorted({r["d_mm"] for r in R[t]}):
                row = [t, f"{d:g}"]
                for v in ("base", "ell100", "si"):
                    rs = [r for r in R[t] if r["d_mm"] == d and r.get("variant", "base") == v]
                    row.append(f"{sum(r['held'] for r in rs)}/{len(rs)}, {fmt(med([r.get(key) for r in rs]), 2)}" if rs else "&#8212;")
                body.append(row)
        S.append(("units", "Length scale of Eq. (2) in the tasks", r"""
<p>Eq. (2) compares angular and linear errors through the length scale $\ell$ (linear rows divided by $\ell$), which the HOM paper
does not state. The method page's Exp. A gives the paper's ranking (roll about $y$ and separation best) at $\ell=100$&#8202;mm. In the
tasks that scale fails: with rotation errors weighed more, the pinching fingers no longer close on the cylinder during the pick, and
no cylinder is held; rolling degrades and twisting holds with smaller ramps. The tasks therefore keep $\ell=20$&#8202;mm.</p>"""
                  + T(["Task", "Size (mm)", "$\\ell=20$&#8202;mm: held, score", "100&#8202;mm", "1&#8202;m (SI)"], body,
                      "Held count and median score at three length scales (score: pin-push displacement in mm, press wall-force RMS "
                      "error in N, roll and twist angular RMSE / step).", num_from=2)))

    # the estimator: source of the reference geodesic velocity, k_max and the controller period
    FS = [r for r in rows_file(D / "filter_sweep.jsonl") if r.get("status") == "ok"]
    if FS:
        body = []
        for t in ("roll", "twist"):
            for d in sorted({r["d_mm"] for r in FS if r["task"] == t}):
                row = [t, f"{d:g}"]
                for src, kmax, dt in (("reference", 100.0, 0.01), ("planned", 100.0, 0.01), ("planned", 25.0, 0.01),
                                      ("reference", 100.0, 0.02), ("planned", 100.0, 0.02), ("planned", 25.0, 0.02)):
                    rs = [r for r in FS if r["task"] == t and r["d_mm"] == d and r["source"] == src and r["k_max"] == kmax
                          and r["dt"] == dt]
                    row.append(f"{sum(r['held'] for r in rs)}/{len(rs)}, {fmt(med([r.get('w_rmse_ratio') for r in rs]), 2)}"
                               if rs else "&#8212;")
                body.append(row)
        S.append(("estimator", "Contact estimator settings", r"""
<p>The filter of Eqs. (4)&#8211;(6) predicts the contact frames' motion from $S\,\mathcal V_{L_0L_1,\rm ref}$, the reference
relative contact velocities. In the roll and twist tasks, as in the paper's Exp. B, those references are zero and the motion comes
from the object twist, so the reference geodesic speed is zero, the gain stays at $k_{\min}$ and the filtered action point trails
the rolling contact with a time constant of $1/k_{\min}=0.2$&#8202;s. Fed instead with the relative twist that Eq. (2) plans
(<code>FilterParams.source="planned"</code>), the filter follows the rolling contact and its gain reaches $k_{\max}$; the discrete
update then moves $k\,\Delta t$ of the way to the measurement each period, which overshoots above 1 and is unstable near 2. The
paper's 500&#8202;Hz gives $k_{\max}\Delta t=0.2$ at $k_{\max}=100$&#8202;s$^{-1}$; at the bench's 50&#8202;Hz the same gain gives 2.</p>"""
                  + T(["Task", "Size (mm)", "Reference, 100&#8202;Hz", "Planned, $k_{\\max}$ 100", "Planned, 25",
                       "Reference, 50&#8202;Hz", "Planned, 100", "Planned, 25"], body,
                      "Held count and median angular RMSE / step for the filter's source, $k_{\\max}$ (s$^{-1}$) and "
                      "controller rate; seeds 0&#8211;2.", num_from=2)))

    # deployment variants
    var_rows = {t: [r for r in R[t] if r.get("variant", "base") in ("nofilter", "rate50", "sense")] for t in R}
    if any(var_rows.values()):
        body = []
        for t in ("pinpush", "press", "roll", "twist"):
            for v in sorted({r["variant"] for r in var_rows[t]}):
                rs = [r for r in var_rows[t] if r["variant"] == v]
                key = {"pinpush": "disp_end_mm", "press": "wall_err_rms", "roll": "w_rmse_ratio", "twist": "w_rmse_ratio"}[t]
                body.append([t, v, f"{sum(r['held'] for r in rs)}/{len(rs)}", fmt(med([r.get(key) for r in rs]), 2)])
        S.append(("variants", "Deployment variants", r"""
<p>Variants of the base runs that move toward the bench: <b>nofilter</b> takes the impedance's action point from the GCF instead of
the filter; <b>rate50</b> runs the controller at 50&#8202;Hz, the servo bus's closed-loop rate; <b>sense</b> runs at 50&#8202;Hz with
the object pose read 40&#8202;ms late with 0.5&#8202;mm and 1&#8202;deg noise, centre-of-pressure noise of 1&#8202;mm and force noise of
0.03&#8202;N.</p>""" + T(["Task", "Variant", "Held", "Median score"], body,
                         "Deployment variants (score: pin-push displacement in mm, press wall-force RMS error in N, roll and twist "
                         "angular RMSE / step).", num_from=2)))

    # pinned iCEM against HOM in the same loop (scripts/hom_tasks_mpc.py); the last row per size, seed and controller
    SCORE = {"pinpush": ("disp_end_mm", 1), "press": ("wall_err_rms", 2), "roll": ("w_rmse_ratio", 2),
             "twist": ("ramp_three_pads_max_deg", 0)}
    MP = {}
    for t in ("pinpush", "press", "roll", "twist"):
        last = {}
        for r in rows(f"mpc_{t}"):
            if r.get("status") == "ok":
                last[(r["d_mm"], r["seed"], r["controller"])] = r
        MP[t] = list(last.values())
    if any(MP.values()):
        body, smooth = [], []
        for t in ("pinpush", "press", "roll", "twist"):
            key, nd = SCORE[t]
            for d in sorted({r["d_mm"] for r in MP[t]}):
                c = {k: [r for r in MP[t] if r["d_mm"] == d and r["controller"] == k] for k in ("hom", "mpc")}
                body.append([t, f"{d:g}"]
                            + [f"{sum(r['held'] for r in c[k])}/{len(c[k])}" if c[k] else "&#8212;" for k in ("hom", "mpc")]
                            + [fmt(med([r.get(key) for r in c[k] if r["held"]]), nd) if c[k] else "&#8212;"
                               for k in ("hom", "mpc")])
            c = {k: [r for r in MP[t] if r["controller"] == k] for k in ("hom", "mpc")}
            if not (c["hom"] or c["mpc"]):
                continue
            smooth.append([t] + [fmt(med([r.get(m) for r in c[k]]), nd2) if c[k] else "&#8212;"
                                 for m, nd2 in (("cmd_rate_rms", 2), ("cmd_reversals", 1), ("omega_jitter", 1), ("wall_s", 0))
                                 for k in ("hom", "mpc")])
        thumb = {k: med([s_["thumb_mean"] for r in MP["press"] if r["controller"] == k for s_ in r.get("press_steps", [])])
                 for k in ("hom", "mpc")}
        films = "".join(F(MEDIA / f"20261021-hom_vs_mpc_{t}_{o}.mp4", f"{t.capitalize()}, {o.replace('cylinder', 'cylinder ').replace('sphere', 'sphere ')}"
                          f"&#8202;mm, seed 0: HOM (left) and pinned iCEM (right) from the same pick.")
                        for t, o in (("pinpush", "cylinder25"), ("press", "cylinder25"), ("roll", "sphere40"), ("twist", "sphere40")))
        S.append(("mpc", "MPC baseline", r"""
<p>From the start of each task (2.0&#8202;s, after the HOM pick and lift) the planner of the 2026-10-14 scale study, pinned iCEM,
replaces the HOM controller. Every 10&#8202;ms it samples 64 plans of the nine servo targets (two iterations of 32, three knots over
0.5&#8202;s, knot noise 0.08&#8202;rad, the first knot fixed to the applied target), rolls them out on the plant's own pads model with
<code>mujoco.rollout</code> and applies the best; between replannings the targets follow that plan at the 1&#8202;ms physics step.
A plan costs, over its 50 samples at 10&#8202;ms,
$$J=\sum_{k=1}^{49}c_k+10\,c_{50}+\sum_{i=1}^{2}\lVert\mathbf u_{i+1}-\mathbf u_i\rVert^2,$$
with $\mathbf u_i$ the knots (rad) and the sample cost $c_k$ of the task:
$$\begin{aligned}
c^{\rm pin\text{-}push}&=\Big(\tfrac{d-d_{\rm ref}}{1\,{\rm mm}}\Big)^2+\tfrac{1-\mathbf a\cdot\mathbf a_0}{1-\cos 5^\circ}
+\textstyle\sum_f{\rm clip}\big(\tfrac{0.2\,{\rm N}-F_f}{0.2\,{\rm N}},0,1\big)+10\,[z<z_0-15\,{\rm mm}],\\
c^{\rm press}&=\Big(\tfrac{F_{\rm wall}-F_{\rm ref}}{0.25\,{\rm N}}\Big)^2+\tfrac{1-\mathbf a\cdot\mathbf a_0}{1-\cos 5^\circ}
+10\,[z<z_0-15\,{\rm mm}],\\
c^{\rm roll,\,twist}&=\tfrac{\lVert\bm\omega-\bm\omega_{\rm ref}\rVert^2}{(10^\circ/{\rm s})^2}+\tfrac{\lVert\mathbf p-\mathbf p_0\rVert^2}{(5\,{\rm mm})^2}
+\textstyle\sum_f{\rm clip}\big(\tfrac{0.5\,{\rm N}-F_f}{0.5\,{\rm N}},0,1\big)+10\,[z<z_0-15\,{\rm mm}],
\end{aligned}$$
where $d$ is the cylinder's displacement along its axis $\mathbf a$ since the task began ($\mathbf a_0$ the axis then), $F_f$ a
finger tip's net force on the object from a contact sensor of the planner's model, $F_{\rm wall}$ the wall's force on the cylinder,
$\bm\omega$ the object's angular velocity, $\mathbf p$ its centre and $z$ its height. The references are the HOM schedule's; the
press steps start at the wall contact, predicted at 0.5&#8202;s until it happens. The HOM arm runs in the same loop and its rows carry
the same command log.</p>"""
                  + T(["Task", "Size (mm)", "Held, HOM", "Held, MPC", "Score, HOM", "Score, MPC"], body,
                      "Pinned iCEM (MPC) against HOM, seeds 0&#8211;2: held count and median score over held runs (pin-push axial "
                      "displacement in mm, press wall-force RMS error in N, roll angular RMSE / step, twist ramp on three pads in deg).",
                      num_from=2)
                  + T(["Task", "Command rate<br>HOM", "<br>MPC", "Reversals<br>HOM", "<br>MPC", "Angular jitter<br>HOM",
                       "<br>MPC", "Time per run<br>HOM", "<br>MPC"], smooth,
                      "Command smoothness and computation per task, medians over sizes and seeds: command rate in rad/s, command "
                      "reversals per joint per second, angular jitter in deg/s, wall time per run in s.")
                  + (f"<p>In the press steps the thumb presses on the end cap with a median {fmt(thumb['hom'], 2)}&#8202;N under HOM "
                     f"and {fmt(thumb['mpc'], 2)}&#8202;N under MPC.</p>" if thumb["hom"] is not None and thumb["mpc"] is not None else "")
                  + films))

    # the window's probes: the impedance from the measured surface and a force loop (scripts/hom_tasks.py --rows-file probes)
    PR = [r for r in rows_file(D / "probes.jsonl") if r.get("status") == "ok"]
    if PR:
        body = []
        for t in ("pinpush", "press", "roll", "twist"):
            key, nd = SCORE[t]
            for d in sorted({r["d_mm"] for r in PR if r["task"] == t}):
                row = [t, f"{d:g}"]
                for v, src in (("base", R[t]), ("surface", PR), ("floop", PR)):
                    rs = [r for r in src if r.get("task", t) == t and r["d_mm"] == d and r.get("variant", "base") == v]
                    row.append(f"{sum(r['held'] for r in rs)}/{len(rs)}, {fmt(med([r.get(key) for r in rs if r['held']]), nd)}"
                               if rs else "&#8212;")
                body.append(row)
        S.append(("probes", "Impedance from the contact surface and a force loop", r"""
<p>Two variants of the base controller: <b>surface</b> keeps the servo target's impedance offset measured from the object's surface at
the measured contact also while the finger touches (<code>HomParams.impedance="surface"</code>), and <b>floop</b> adds
10&#8202;mm/s of $v_{\rm sep}$ per newton of force error, a force loop that needs a fingertip force measurement.</p>"""
                  + T(["Task", "Size (mm)", "Base: held, score", "Surface", "Force loop"], body,
                      "Held count and median score over held runs, seeds 0&#8211;2 (scores as in the MPC table).", num_from=2)))

    rl = D / "rl_assessment.json"
    A = None
    if rl.exists():
        try:
            A = json.loads(rl.read_text())
        except json.JSONDecodeError:
            A = None
    if A:
        NAMES = {"residual_ppo_on_hom": "Residual PPO on HOM", "hom_teacher_imitation": "HOM as teacher (cloning, then DAgger)",
                 "ppo_from_scratch": "PPO from scratch"}
        C = A.get("cost", {})
        body = []
        for o in A.get("options", []):
            c = C.get(o["name"], {})
            gpu = c.get("gpu_hours")
            body.append([NAMES.get(o["name"], esc(o["name"])), esc(o.get("action", "")), esc(o.get("expected_gain", "")),
                         f"{c.get('engineering_days', '&#8212;')}", fmt(gpu, 1) if isinstance(gpu, (int, float)) else "&#8212;"])
        S.append(("rl", "Reinforcement learning: options and costs", r"""
<p>The bench measures the nine joint angles on the servo bus (closed loop at 50&#8202;Hz) and the object pose from two AprilTags at
30&#8202;frames/s; it has no calibrated fingertip or wall force. The HOM controller costs 1.1&#8202;ms per period on one core
(0.97&#8211;1.36&#8202;ms over the 16 cells), against 0.26&#8202;s for one step of 2,048 GPU environments with pads
(7,800 environment steps/s, 43&#8202;min per 20&#8202;M-step seed). The options, their costs for four tasks at three seeds, and what each
could gain over the measured HOM gaps (<code>rl_assessment.json</code>):</p>"""
                  + T(["Option", "Action", "What it could gain", "Engineering (days)", "GPU hours"], body,
                      "Learning options for the four tasks. GPU hours: 20&#8202;M steps per seed, three seeds per task, one policy per "
                      "task over its four sizes.", num_from=3)
                  + "<p>" + esc(A.get("recommendation", "")) + "</p>"))

    S.append(("limits", "Limitations", r"""
<p>All results are simulation on one contact model and one plant. The objects weigh 25&#8202;g. The pin-push and press grasps are one
layout family (the scale study's tripods); a layout searched for these tasks could change the 10 and 32&#8211;40&#8202;mm results.
The task schedules are fixed slider sequences, as in the paper's teleoperation, with no feedback from the task score.</p>"""))
    fw_mpc = (r"""<li>Pinned iCEM on the same tasks and scenes (<code>scripts/hom_tasks_mpc.py</code>): if MPC holds where
HOM drops (50&#8211;63&#8202;mm spheres on the $y$ steps), the failure is the HOM schedule's, not the grasp's.</li>
""" if not any(MP.values()) else "")
    fw_rl = (r"""<li>HOM at 50&#8202;Hz without contact estimates (witness points only, the signals the bench has) on the four tasks,
seeds 0&#8211;2, as a variant of <code>scripts/hom_tasks.py</code>: it measures how much of HOM the bench's signals support and fixes
the base of a residual policy. Then a residual-PPO pilot on twist with the 40&#8202;mm sphere: 5&#8202;M steps, two seeds, a CPU pool
of HOM controllers. The pilot fails if the ramp held on three pads stays at or below HOM's 50&#8202;Hz median of 33&#8202;deg, or if the
policy reverses its servo targets more than twice as often as HOM.</li>
""" if A else "")
    S.append(("future", "Future work", r"""
<ul>
""" + fw_mpc + fw_rl + r"""<li>Press with a thumb posture square to the end cap: a pusher-specific layout search that scores the angle between the block's face and
the cap; wall forces within 0.1&#8202;N of the command would confirm that the edge contact causes the shortfall.</li>
<li>Bench dry run with <code>src/morphohand/control/hom_bench.py</code> at 50&#8202;Hz on recorded tag poses, then the twist task on a
40&#8202;mm sphere on the hand.</li>
</ul>"""))
    S.append(("files", "Files", f"""
<p>Folder <code>docs/experiments/20261021-hom_framework/</code>: <code>design.json</code> (grasps), <code>rows/&lt;task&gt;.jsonl</code>
(one line per rollout with its trace), <code>rows/mpc_&lt;task&gt;.jsonl</code> (the MPC comparison, both arms, with the command log),
<code>rl_assessment.json</code> (learning options), <code>media/</code> (films; <code>20261021-hom_vs_mpc_*.mp4</code> side by side at
every size). Scripts: <code>scripts/hom_tasks.py</code> (design, run), <code>scripts/hom_tasks_mpc.py</code> (MPC),
<code>src/morphohand/control/hom_bench.py</code> (the controller between the bench's signals and the servo targets, with a 50&#8202;Hz
dry run), <code>scripts/hom_framework_page.py</code>. The method page: <a href="{relp(METHOD)}">Hand-object mechanism control: the method and its
MuJoCo implementation</a>.</p>"""))

    # lede from the numbers
    def held(t, d):
        rs = [r for r in R[t] if r["d_mm"] == d and r.get("variant", "base") == "base"]
        return f"{sum(r['held'] for r in rs)}/{len(rs)}" if rs else "&#8212;"
    pp = [r for r in R["pinpush"] if r.get("variant", "base") == "base"]
    tw = [r for r in R["twist"] if r.get("variant", "base") == "base"]
    ro = [r for r in R["roll"] if r.get("variant", "base") == "base"]
    pin = [r["disp_end_mm"] for r in pp if r["held"]] or [0.0]
    wl = [s_["wall_mean"] for r in R["press"] if r.get("variant", "base") == "base" for s_ in r.get("press_steps", [])
          if s_["F_ref"] >= 1.0] or [0.0]
    lede = ("The library's HOM controller runs four tasks on the CAD-model SR2 hand in MuJoCo. Twisting a sphere holds at every size "
            f"from 30 to 63{NB}mm ({sum(r['held'] for r in tw)}/{len(tw)} rollouts) and turns it "
            f"{fmt(min(r.get('ramp_three_pads_max_deg') or 0 for r in tw), 0)}&#8211;{fmt(max(r.get('ramp_three_pads_max_deg') or 0 for r in tw), 0)}{NB}deg on three pads. "
            f"Rolling holds on 30 and 40{NB}mm spheres (angular RMSE / step {fmt(min(r['w_rmse_ratio'] for r in ro if r['d_mm'] <= 40), 2)}&#8211;"
            f"{fmt(max(r['w_rmse_ratio'] for r in ro if r['d_mm'] <= 40), 2)}) and loses larger spheres on the steps about the $y$ axis "
            f"({held('roll', 50.0)} at 50, {held('roll', 63.0)} at 63{NB}mm). The pin-push holds {sum(r['held'] for r in pp)}/{len(pp)} "
            f"cylinders of 10&#8211;25{NB}mm and moves them {fmt(min(pin))}&#8211;{fmt(max(pin))}{NB}mm of the commanded 10; the press "
            f"reaches the wall but delivers {fmt(min(wl))}&#8211;{fmt(max(wl))}{NB}N for commands of 1&#8211;2{NB}N, because "
            "the thumb's block meets the end cap on its edge.") if (tw and ro and pp) else "Rows pending."
    if lede != "Rows pending." and any(MP.values()):
        cnt = {k: [r for t in MP for r in MP[t] if r["controller"] == k] for k in ("hom", "mpc")}
        lede += (f" Pinned iCEM, replanning every 10{NB}ms on the same plant from the task phase on, holds "
                 f"{sum(r['held'] for r in cnt['mpc'])}/{len(cnt['mpc'])} runs against HOM's "
                 f"{sum(r['held'] for r in cnt['hom'])}/{len(cnt['hom'])} in the same loop, at a median "
                 f"{fmt(med([r.get('wall_s') for r in cnt['mpc']]), 0)}{NB}s of computation per run against "
                 f"{fmt(med([r.get('wall_s') for r in cnt['hom']]), 1)}{NB}s, and reverses its servo targets "
                 f"{fmt(med([r.get('cmd_reversals') for r in cnt['mpc']]), 1)} times per second against HOM's "
                 f"{fmt(med([r.get('cmd_reversals') for r in cnt['hom']]), 1)}.")
    if lede != "Rows pending." and PR:
        sr = {d: [r for r in PR if r["task"] == "roll" and r.get("variant") == "surface" and r["d_mm"] == d] for d in (50.0, 63.0)}
        if all(sr.values()):
            lede += (f" With the impedance kept on the measured surface in contact, the roll holds "
                     f"{sum(r['held'] for r in sr[50.0])}/{len(sr[50.0])} at 50 and {sum(r['held'] for r in sr[63.0])}/{len(sr[63.0])} "
                     f"at 63{NB}mm (base {held('roll', 50.0)} and {held('roll', 63.0)}).")
    summary = re.sub(r"<[^>]+>|\$", "", lede.replace("&#8202;", " ").replace("&#8211;", "-"))
    summary = ". ".join(summary.split(". ")[:2]).rstrip(".") + "."
    assemble(S, "HOM tasks across object sizes", "HOM control of four tasks across object sizes on the CAD-model SR2 hand",
             "Reorientation mechanisms and control &#183; 2026-10-21", lede,
             f"Built {time.strftime('%Y-%m-%d %H:%M')} from rows/*.jsonl at commit {git_short()}.", RESULTS, summary)


if __name__ == "__main__":
    which = sys.argv[1:] or ["method", "results"]
    if "method" in which:
        method_page()
    if "results" in which:
        results_page()
