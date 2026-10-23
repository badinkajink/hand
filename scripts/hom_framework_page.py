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
    """The rows of rows/<name>.jsonl, the last one per task, size, seed, variant and controller (a filmed rerun of a seed
    replaces its row)."""
    p = ROWS / f"{name}.jsonl"
    if not p.exists():
        return []
    out = {}
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[(r.get("task"), r.get("d_mm"), r.get("seed"), r.get("variant", "base"), r.get("controller"))] = r
    return list(out.values())


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


def video(path, caption, n, poster=None):
    if not Path(path).exists():
        return ""
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    pa = (f' poster="data:image/jpeg;base64,{base64.b64encode(Path(poster).read_bytes()).decode()}"'
          if poster is not None and Path(poster).exists() else "")
    return (f'<figure><video controls muted loop playsinline preload="metadata"{pa} src="data:video/mp4;base64,{b64}"></video>'
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
<p>The screwdriver sequence of the paper uses all of these. A pinch of fingers 1 and 3 picks the tool up as a fixed joint. The pinch
relaxes to a revolute joint, so gravity swings the tool down. Finger 2 pushes the tool to vertical about the pinch axis and presses it
on the table while fingers 1 and 3 turn it about its axis. Fingers 1 and 3 then release, move round and turn again.</p>"""))

    S.append(("gcf", "Generalized contact frames", r"""
<p>Each finger has two GCFs, $\{L_1\}$ on the finger body expected to touch the object and $\{L_0\}$ on the object [1]. When the
bodies are apart the origins are the witness points, and the normal axes follow the gradient of the bodies' signed distance. In
contact, the origins are the points of each surface closest to the contact point, and the normals point from the contact point to
the origins, outward from each body. The finger's $x_1$ therefore points from the finger toward the object and $x_0=-x_1$. The
$y$ axes are aligned as closely as the tangent plane allows with the finger body's joint axis, so that a rotation about $y$ is
the finger's own flexion, and $z=x\times y$.</p>
<p>In the library, <code>Mirror.gcf_witness</code> takes the witness points from <code>mj_geomDistance</code> between the TPU
block's convex mesh and the object on a kinematic copy of the scene. <code>Mirror.gcf_from_cop</code> projects the pads'
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
weights nor the units in which angular and linear rows are compared. Here every linear row is divided by $\ell=20$&#8202;mm, so
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
origin, $d$ the measured gap at the joint angles $\theta$ and $d_{\rm ref}$ the gap of the action point at the reference
configuration,</p>
$$\theta_{\rm cmd}=\theta_{\rm ref}+J^{+}\bigl(d_{\rm ref}\,x_{C_1}\bigr)+\frac{1}{k_p}J^{\top}F\,x_{C_1}+\frac{\tau_g}{k_p},
\qquad d_{\rm ref}=\min\bigl(\max\bigl(d-x_{C_1}^{\top}J(\theta_{\rm ref}-\theta),\,0\bigr),\,6\,{\rm mm}\bigr).$$
<p>The second term moves the reference's action point onto the object's measured surface, as the paper's reference penetration is
measured from the surface; the third presses with $F$ through the servo spring; the last cancels the finger's own weight. $J^{+}$ is a
damped pseudo-inverse (damping 2&#8202;mm/rad). A reference ahead of the surface ($d_{\rm ref}=0$) is kept, so a pusher's planned
motion still pushes and an object pushed into a finger meets its servo spring. Before 2026-10-22 the gap was $\max(d,0)$, measured at
$\theta$: when an object moved away from a finger's reference while the pad still touched, $d$ was about zero, the target stayed
outside the surface and the force decayed (the results page compares the two). A test checks that the target's action point sits
$F\sum_j (J^{\top}x)_j^2/k_{p,j}$ inside the surface from a reference that lags 1.6&#8202;mm behind the finger.</p>
<p>The paper's filter gains are not printed. Here $k_{\min}=5$&#8202;s$^{-1}$, $k_{\max}=100$&#8202;s$^{-1}$ and
$\alpha=20$&#8202;mm/s; on a static contact with 1&#8202;mm measurement noise the filtered point stays within 0.35 of the raw
error (test). <code>FilterParams.source</code> selects the paper's $S\,\mathcal V_{L_0L_1,\rm ref}$ or, as an option, the twist that
Eq. (2) plans. The choice matters when a task commands the object twist with zero reference relative velocities, as Exp. B does: the
reference geodesic speed is then zero and the gain never leaves $k_{\min}$ (results page, contact estimator settings). The discrete
update moves $k\,\Delta t$ of the way to the measurement each period, so $k_{\max}\Delta t$ must stay well below 1. The paper's
500&#8202;Hz allows $k_{\max}=100$&#8202;s$^{-1}$, and the bench's 50&#8202;Hz allows about a quarter of that.</p>"""))

    S.append(("paper_exps", "The paper's experiments and implementation", r"""
<p>The paper teleoperates five simulated hands in Drake with hydroelastic contact, which is used because a pinch needs friction torque about
its axis, and point contact has none. The hands are Orca V2, Wuji, Sharpa Wave, a three-finger power-grasp hand (Custom A) and a five-finger
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
<p>The paper reports RMSE traces and no success rates, and it names hardware contact sensing as the open problem. Its future work proposes
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
         ["Impedance", "servo offsets in <code>HomController.step</code>; <code>reanchor</code>",
          "<code>test_impedance_target_on_the_surface_from_a_lagging_reference</code>"]],
        num_from=9, caption="Where each part of the paper's method is implemented and checked.", n=1)))

    diffs = [
        ["Joint actuation", "Torque-controlled joints; inverse dynamics tracks $\\theta_{\\rm ref}$ and $u_{\\rm ref}$",
         "Position servos, $k_p$ 4&#8202;N&#8202;m/rad, 1&#8202;N&#8202;m limit; $\\theta_{\\rm ref}$ is the target",
         "The bench hand's Feetech servos accept position targets only"],
        ["Applied force", "Impedance controller, force set as a reference penetration distance",
         "Servo offsets $J^{+}(d_{\\rm ref}\\,x)+J^{\\top}F\\,x/k_p+\\tau_g/k_p$, the gap $d_{\\rm ref}$ measured at the "
         "joint reference; force set in newtons",
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
         "A stopped pusher otherwise keeps its servo-lag offset as force"],
        ["Pusher contact", "Normal across the pinch axis; the pusher rolls freely ($w_{\\rm roll}=0$, Exp. C)",
         "In the pin-push the middle finger's normal is within 15&#8202;deg of the pinch axis and its pad neither rolls nor "
         "slides, so friction turns the cylinder",
         "The demonstration pushes with the back of a finger, which has no tactile sensing on SR2; a freely rolling pusher "
         "lost the drive"],
        ["Pinch forces", "Equal on the two pinching fingers", "The thumb adds the pusher's force along the pinch axis",
         "The pusher presses along the pinch axis on the index's side"],
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
<p><code>tests/test_hom_control.py</code> has 21 tests, all passing at commit @@COMMIT@@:</p>
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
<li>The servo impedance: from a reference 1.6&#8202;mm behind the finger the target's action point sits on the surface plus the
spring's press; the earlier gap stops short by the lag.</li>
<li>The bench adapter: 2&#8202;s of the roll task at 50&#8202;Hz through the bench's units give the controller&#39;s own targets
without adjustment, a replay of the stream gives them again, and a plan with other mounts is refused.</li>
</ul>
<p>Run them with <code>PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_hom_control.py -q</code>.</p>"""))

    S.append(("expa", "Exp. A on the library and the units of Eq. (2)", expa_section()))

    S.append(("tasks", "Tasks run on the library", r"""
<p>The companion page, <a href="@@RESULTS@@">HOM control of four demonstrated tasks across object sizes</a>, runs the library on the
CAD-model SR2 hand: a pin-push and a press on cylinders of 10&#8211;40&#8202;mm (a thumb-index pinch as the revolute joint of Exp. C,
the middle finger as the pusher of Eq. (3), turning the cylinder by friction in the pin-push and by its normal force in the press),
and spheres of 30&#8211;63&#8202;mm rolled along the thumb and twisted (the paper's Exp. B on an equator tripod), with films of every
size. It also varies the pin-push's drive force and lever, and compares HOM with a sampling MPC that plans on the plant's own model
from the same pick.</p>"""))

    S.append(("limits", "Limitations", r"""
<p>The library has not been run against the reference implementation, so the agreement is with the paper's text and not its code. The
weights, filter gains and units of Table 2 are this library's choices, and Exp. A shows that the units change which elements track.
The filter has been tested on spheres and in the tasks, not against hydroelastic centres of pressure. Exp. D has not been rerun on the
library (the October 2 runs used the earlier code). No part has run on hardware.</p>"""))

    S.append(("future", "Future work", r"""
<ul>
<li>The length scale of Eq. (2): $\\ell=100$&#8202;mm reproduces Exp. A's ranking; the results page's section on the length scale
runs the tasks at 20&#8202;mm, 100&#8202;mm and 1&#8202;m. The scale or weights of the reference implementation are the first item of
the comparison below; a scale that passes both Exp. A and the tasks would replace 20&#8202;mm.</li>
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
            r"the anti-windup of the separation velocity and the complementary filter of Eqs. (4)&#8211;(6). Twenty-one tests check it "
            r"against closed forms and finite differences (Eq. (1) to 1e-4 relative), and the paper's Exp. A on the CAD-model hand "
            r"gives the paper's ranking (roll about $y$ and separation best) when Eq. (2) weighs 100&#8202;mm/s like 1&#8202;rad/s. It differs from the paper in the " + str(len(diffs)) + r" places of "
            r"Table 2, mostly where position servos replace torque control and sphere pads replace hydroelastic contact; the three "
            r"earlier HOM controllers in this repository differ more (Table 3).")
    summary = ("The HOM paper's controller (GCFs, Eqs. 1-6, pinch and pusher) implemented as one MuJoCo library for position-servo "
               "hands; 21 tests pass; " + str(len(diffs)) + " differences from the paper listed, mostly servos for torque control and pads for "
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


SIZES_CYL = (10, 16, 25, 40)


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
TASK_LABEL = {"pinpush": "Pin-push", "press": "Press", "roll": "Roll", "twist": "Twist"}


def nf(f):
    """A force in N as written on the page: 0.5, 0.75, 1.0, 1.5, 2.0."""
    return f"{f:.1f}" if abs(f * 10 - round(f * 10)) < 1e-9 else f"{f:.2f}"


def span(xs, nd=1, scale=1.0, dash="&#8211;"):
    """'lo-hi' of the values (one value when both ends print alike), '&#8212;' when there are none."""
    xs = [scale * x for x in xs if x is not None]
    if not xs:
        return "&#8212;"
    lo, hi = fmt(min(xs), nd), fmt(max(xs), nd)
    return lo if lo == hi else f"{lo}{dash}{hi}"


def mpc_findings(MP, arm, pad_force):
    """The MPC section's findings paragraph, from the rows."""
    rng = span

    def top(xs, nd=0):
        xs = [x for x in xs if x is not None]
        return fmt(max(xs), nd) if xs else "&#8212;"

    A = {k: [r for t in MP for r in arm(t, k)] for k in ("hom", "mpc")}
    if not A["mpc"]:
        return ""
    H = {(t, k): [r for r in arm(t, k) if r["held"]] for t in MP for k in ("hom", "mpc")}
    cyl = {k: H[("pinpush", k)] + H[("press", k)] for k in ("hom", "mpc")}
    sph = {k: H[("roll", k)] + H[("twist", k)] for k in ("hom", "mpc")}
    # rollouts whose object fell before the task began (in the shared HOM pick): z 15 mm below its start at t < 2.0 s
    early = [r for r in A["mpc"] if not r["held"] and any(x["t"] < 2.0 and x["pos_mm"][2] < r["trace"][0]["pos_mm"][2] - 15.0
                                                          for x in r.get("trace", []))]
    out = [f"MPC held {sum(r['held'] for r in A['mpc'])} of {len(A['mpc'])} rollouts and HOM "
           f"{sum(r['held'] for r in A['hom'])} of {len(A['hom'])}"
           + (f"; the {WORDS.get(len(early), len(early))} lost by both fell off the post while the fingers closed, before the task "
              "began." if early and
              len(early) == len(A["mpc"]) - sum(r["held"] for r in A["mpc"]) else ".")]
    if cyl["mpc"]:
        th = {(t, k): [r.get("theta_rmse_deg") for r in H[(t, k)]] for t in ("pinpush", "press") for k in ("hom", "mpc")}
        out.append("On the cylinders MPC tracks the pinch angle with an RMSE of " + " and ".join(
            f"{rng(th[(t, 'mpc')])}{NB}deg in the {t.replace('pinpush', 'pin-push')} (HOM {rng(th[(t, 'hom')])}{NB}deg)"
            for t in ("pinpush", "press") if th[(t, "mpc")]) + ". It squeezes to do so: its "
                   f"pads press {rng([pad_force(r) for r in cyl['mpc']], 0)}{NB}N on average against HOM's "
                   f"{rng([pad_force(r) for r in cyl['hom']], 0)}{NB}N, finger links press the cylinder with up to "
                   f"{top([r.get('link_force_max_N') for r in cyl['mpc']])}{NB}N (HOM {top([r.get('link_force_max_N') for r in cyl['hom']])}"
                   f"{NB}N), and the middle finger's pad slides up to {top([r.get('pusher_slip_max_mm') for r in cyl['mpc']])}{NB}mm "
                   f"over the cylinder (HOM {top([r.get('pusher_slip_max_mm') for r in cyl['hom']])}{NB}mm).")
    if sph["mpc"]:
        w = {(t, k): [r.get("w_rmse_ratio") for r in H[(t, k)]] for t in ("roll", "twist") for k in ("hom", "mpc")}
        ce = {k: [r.get("centre_err_max_mm") for r in H[("roll", k)]] for k in ("hom", "mpc")}
        out.append("On the spheres it tracks no better than HOM: its angular RMSE / step is " + " and ".join(
            f"{rng(w[(t, 'mpc')], 2)} in the {t} (HOM {rng(w[(t, 'hom')], 2)})" for t in ("roll", "twist") if w[(t, "mpc")])
                   + (f", and in the roll the sphere's centre leaves its path by {rng(ce['mpc'])}{NB}mm against HOM's "
                      f"{rng(ce['hom'])}{NB}mm" if ce["mpc"] else "")
                   + f", with pads pressing {rng([pad_force(r) for r in sph['mpc']], 0)}{NB}N against "
                     f"{rng([pad_force(r) for r in sph['hom']], 0)}{NB}N.")
    out.append(f"Over held runs its servo targets reverse {rng([r.get('cmd_reversals') for r in A['mpc'] if r['held']])} times "
               f"per second against HOM's {rng([r.get('cmd_reversals') for r in A['hom'] if r['held']])}, and one rollout costs "
               f"{rng([r.get('wall_s') for r in A['mpc'] if r['held']], 0, 1 / 60)}{NB}min of computation against "
               f"{rng([r.get('wall_s') for r in A['hom'] if r['held']], 0)}{NB}s.")
    return "<p>" + " ".join(out) + "</p>"


def results_page():
    des = json.loads((D / "design.json").read_text()) if (D / "design.json").exists() else {}
    R = {t: [r for r in rows(t) if r.get("status") == "ok"] for t in ("pinpush", "press", "roll", "twist")}
    S = []
    n_tab = [0]
    n_film = [0]

    def T(head, body, caption, num_from=1):
        n_tab[0] += 1
        return table(head, body, num_from=num_from, caption=caption, n=n_tab[0])

    def F(path, caption, poster=None):
        n_film[0] += 1
        return video(path, caption, n_film[0], poster)

    def base(t, d=None, v="base"):
        return [r for r in R[t] if r.get("variant", "base") == v and (d is None or r["d_mm"] == d)]

    def held(rs):
        return f"{sum(r['held'] for r in rs)}/{len(rs)}" if rs else "&#8212;"

    def seeds(rs, k, nd=1):
        return " / ".join(fmt(r.get(k), nd) for r in rs)

    def stroke(rs, name):
        return med([s_["gain"] for r in rs if r["held"] for s_ in r.get("strokes", []) if s_["stroke"] == name])

    SCORE = {"pinpush": ("theta_rmse_deg", 1), "press": ("theta_rmse_deg", 1), "roll": ("w_rmse_ratio", 2),
             "twist": ("ramp_three_pads_max_deg", 0)}
    SCORE_TXT = ("score: pin-push and press angle RMSE in deg, roll angular RMSE / step, twist ramp on three pads in deg; "
                 "medians over held runs")

    S.append(("terms", "Terms and metrics", r"""
<dl class="terms">
<dt>Held</dt><dd>At the end of the rollout the object is less than 15&#8202;mm below its start and at least two fingers press on it
with at least half its weight each (0.12&#8202;N; the objects weigh 25&#8202;g).</dd>
<dt>Pinch angle</dt><dd>Rotation of the cylinder's axis about the pinch axis (the palm's $x$ axis, through the thumb's and the
index's contacts) since the start of the task, in deg; positive when the middle finger's side of the cylinder rises.</dd>
<dt>Angle RMSE</dt><dd>Root-mean-square difference between the pinch angle and the commanded angle over the task, in deg.</dd>
<dt>Stroke gain</dt><dd>Change of the pinch angle during one stroke of the command divided by the commanded change; 1 tracks
exactly. Pin-push strokes: up (+15&#8202;deg), down (&#8722;30&#8202;deg) and back (+15&#8202;deg) at 15&#8202;deg/s; press
strokes: press (+20&#8202;deg) and release (&#8722;20&#8202;deg) at 20&#8202;deg/s with 1&#8202;s held between them.</dd>
<dt>Pusher slip</dt><dd>Largest displacement of the middle finger's centre of pressure on the cylinder (in the cylinder's frame)
from its position at the start of the task, in mm; zero when the pad sticks.</dd>
<dt>Off-axis rotation</dt><dd>Largest change of the angle between the cylinder's axis and the plane normal to the pinch axis, in deg:
the cylinder turning in the pinch about any axis but the pinch axis.</dd>
<dt>Angular RMSE / step (roll, twist)</dt><dd>The HOM paper's metric: the root-mean-square norm of the difference between the
object's angular velocity and the reference while a reference is commanded, divided by the step of 40&#8202;deg/s.</dd>
<dt>Step gain (roll, twist)</dt><dd>Angle turned during a 0.5&#8202;s step divided by the commanded 20&#8202;deg.</dd>
<dt>Centre path error (roll)</dt><dd>Largest distance between the sphere's centre and its commanded path, in mm. Rolling on the
thumb moves the centre along the thumb at $r\omega$: 7 to 11&#8202;mm per 20&#8202;deg step for spheres of 40 to 63&#8202;mm.</dd>
<dt>Drift (twist)</dt><dd>Largest distance of the sphere's centre from its position at the start of the task, in mm.</dd>
<dt>Twist ramp on three pads</dt><dd>After the steps the twist reference runs at 30&#8202;deg/s for 3&#8202;s; the largest angle
turned while all three pads press with more than 0.05&#8202;N.</dd>
<dt>Link force</dt><dd>Largest normal force between a finger link other than the tip and the object during a rollout, in N.</dd>
<dt>Pad force</dt><dd>A finger's pads' normal force on the object averaged over the task; the largest of the three fingers'
averages, in N. HOM commands 3&#8202;N to the pinching fingers (the thumb adds the middle finger's share along the pinch axis),
1.5&#8202;N to the pin-push's middle finger, 0.5&#8202;N to the press's and 2.5&#8202;N per pad on the sphere.</dd>
<dt>Command reversals</dt><dd>Sign changes of a servo target's rate between 10&#8202;ms samples, per joint and per second of the task
(changes under $10^{-4}$&#8202;rad ignored). The commanded strokes themselves reverse 0.3 to 0.7 times per second.</dd>
<dt>Command rate</dt><dd>Root-mean-square rate of the nine servo targets over the task's 10&#8202;ms samples, in rad/s.</dd>
<dt>Time per run</dt><dd>Wall-clock seconds to compute one rollout of 4.9 to 7.8&#8202;s simulated time: one core for HOM, five
rollout threads for MPC (six for the 25&#8202;mm pin-push at seed 0).</dd>
</dl>"""))

    lf = {t: [r.get("link_force_max_N") for r in base(t) if r["held"] and r.get("link_force_max_N") is not None] for t in R}
    S.append(("setup", "Hand, plant and tasks", r"""
<p>Every rollout uses the CAD-model SR2 hand (the printed links of 2026-10-13 on the real_v1 base), the bench-identified servo plant
(4&#8202;N&#8202;m/rad, 1&#8202;N&#8202;m, damping 0.08&#8202;N&#8202;m&#8202;s/rad), the printed TPU block as 1&#8202;mm sphere pads
(builder v2), friction 1.0, 1&#8202;ms steps with the elliptic cone at impratio 100, and the library's controller at 100&#8202;Hz with
the filter on. Objects weigh 25&#8202;g. The palm is welded above the object, which rests on a 3&#8202;mm post until the grasp
closes.</p>
<p>The tasks follow four demonstration clips (<code>cylinder_push.mov</code>, <code>cylinder_press.mov</code>,
<code>roll_sphere.mov</code>, <code>twist_sphere.mov</code> in the folder): a cylinder turned about a pinch by a third finger, and a
sphere rolled and twisted in a three-finger grip. On SR2 the fingers hang from the palm plate with their pads facing the opposite
side, so the pinch is thumb against index across the palm's $x$ axis on a 100&#8202;mm cylinder lying along $y$, and the middle
finger, beside the index, is the pusher.</p>
<ul>
<li><b>Pin-push.</b> The pinch is the revolute joint of the paper's Exp. C: both pinching fingers keep
$\mathcal V^b_{PB_0}=0$ and leave spinning and rolling free. The middle finger's pad touches the cylinder 15&#8202;deg below its equator,
40&#8211;50&#8202;mm from the pinch, so its normal is within 15&#8202;deg of the pinch axis. Its reference object twist is Eq. (3)'s
rotation $s$ about the pinch axis, its pad sticks ($w_{\rm slide}$, $w_{\rm roll}$, $w_{\rm spin}$ kept) and it presses with
1.5&#8202;N, so friction transmits its motion along the cylinder's circumference to the cylinder, in both directions. The
demonstration pushes with the back of a finger, which has no tactile sensing on SR2. Command: +15, &#8722;30 and +15&#8202;deg at
15&#8202;deg/s.</li>
<li><b>Press.</b> The same pinch; the middle finger's pad touches 30&#8202;deg below the equator, so its normal force lifts the
cylinder's side about the pinch axis, as the paper's pusher does. Command: up 20&#8202;deg, held 1&#8202;s, down 20&#8202;deg, at
20&#8202;deg/s.</li>
<li><b>Roll.</b> A sphere at the circumcentre of the three mounts in an equator tripod with 2.5&#8202;N per pad. As in the
demonstration it rolls along the thumb's pad: the reference twist is a rotation about the palm's $y$ axis through the thumb's
contact, steps of &#177;40&#8202;deg/s for 0.5&#8202;s in both orders; every finger rolls freely ($w_{\rm roll}=0$, Exp. B).</li>
<li><b>Twist.</b> The roll grasp; steps of &#177;40&#8202;deg/s about the vertical through the centre, then the 30&#8202;deg/s ramp.</li>
</ul>
<p>The thumb opposes both the index and the middle finger along the pinch axis, so it presses with the pinch force (3&#8202;N) plus
the middle finger's force along that axis; with equal forces on the three pads the cylinder slid toward the thumb. The servo
impedance places each target on the object's measured surface, then $F/k$ inside it, whenever a finger's reference falls behind
the surface (the method page's servo impedance, changed on 2026-10-22). Every task begins with the paper's pick: fingers 8&#8202;mm
off the object approach at $v_{\rm sep}=20$&#8202;mm/s until the anti-windup stops them 0.5&#8202;mm away, the forces turn on, and the
post is removed 0.4&#8202;s before the task. The controller's contact measurement comes from the tip's pads only; the largest
force of another link on the object in a held base run is """
              + ", ".join(f"{fmt(max(v), 1)}{NB}N in the {t}" for t, v in lf.items() if v) + ".</p>"))

    # grasps
    gc, gs, clr_min = [], [], []
    for t in ("pinpush", "press"):
        for d in SIZES_CYL:
            g = des.get(f"{t}_cylinder{d:g}")
            if g is None:
                continue
            clr = g.get("link_clearance_mm", {})
            clr_min.append((clr.get("touch", float("nan")), clr.get("touch_ff", float("nan"))))
            gc.append([t.replace("pinpush", "Pin-push").replace("press", "Press"), f"{d:g}",
                       f"{g['x_sep_mm']:g}, {g['y_sep_mm']:g}", f"{1e3 * g['h']:.1f}", f"{g['lever_mm']:.1f}",
                       f"{g['phi_p']:g} / {g['phi_m']:g}", f"{1e3 * g['score']:.1f}"])
    for d in (30, 40, 50, 63):
        g = des.get(f"sphere{d}")
        if g:
            gs.append([f"{d}", f"{g['x_sep_mm']:g}, {g['y_sep_mm']:g}", f"{1e3 * g['h']:.1f}" if g["h"] < 1 else f"{g['h']:.1f}",
                       f"{min(g['sigma_min_mm'].values()):.1f}"])
    if gc:
        S.append(("grasps", "Grasps per object size", r"""
<p>The pinch grasps come from a search over the scale study's tripod layouts, the cylinder's position and depth below the palm, the
pinch's and the middle finger's positions along the cylinder and the latitudes of their contacts below the equator (the pinch at 0
or 15&#8202;deg, the middle finger at 0 or 15&#8202;deg for the pin-push and 30 or 45&#8202;deg for the press). A grasp must be reachable
inside the servos' calibrated ranges with 3&#8202;deg to spare, with every distal link's palmar face within 25&#8202;deg of its contact
normal and the middle finger 40&#8202;mm or more from the pinch. On the MuJoCo model every link but the tips stays 0.5&#8202;mm from
the cylinder at touch and 1.5&#8202;mm when open, and the fingers stay 1&#8202;mm from each other. The grasp with the largest smallest
pad-Jacobian singular value of the three fingers is kept; at touch its links stay """ + fmt(min(c[0] for c in clr_min), 1)
                  + r"""&#8202;mm or more from the cylinder and its fingers """ + fmt(min(c[1] for c in clr_min), 1)
                  + r"""&#8202;mm or more from each other. Pinching contacts 30&#8202;deg below the equator squeezed the cylinder up and
out of the grasp, so they are excluded. The sphere grasps are the scale study's equator tripods.</p>"""
                  + T(["Task", "Diameter (mm)", "Mounts x, y (mm)", "Depth (mm)", "Lever (mm)", "Latitudes (deg)",
                       "Singular value (mm/rad)"], gc,
                      "Pinch grasps: mount spacings of the tripod layout; depth of the cylinder's axis below the mounts; lever, "
                      "the distance of the middle finger's contact from the pinch along the cylinder; contact latitudes below the "
                      "equator, pinch / middle finger; the least-conditioned finger's smallest pad-Jacobian singular value.")
                  + (T(["Sphere (mm)", "Mount spacing x, y (mm)", "Palm height (mm)", "Smallest singular value (mm/rad)"], gs,
                       "Equator tripods of the roll and twist tasks.") if gs else "")))

    for t, name, strokes, cap in (
            ("pinpush", "Pin-push with a friction drive", ("up", "down", "back"),
             "Pin-push: the middle finger drags the cylinder about the thumb-index pinch by friction."),
            ("press", "Press", ("press", "release"),
             "Press: the middle finger's normal force lifts the cylinder about the pinch.")):
        if not R[t]:
            continue
        body = []
        for d in sorted({r["d_mm"] for r in R[t]}):
            rs = base(t, d)
            if not rs:
                continue
            body.append([f"{d:g}", held(rs), seeds(rs, "theta_rmse_deg"), " / ".join(fmt(stroke(rs, k), 2) for k in strokes),
                         seeds(rs, "pusher_slip_max_mm"), seeds(rs, "off_axis_max_deg")])
        films = "".join(F(MEDIA / f"20261021-hom_{t}_cylinder{d:g}_base.mp4",
                          f"{name}, {d:g}&#8202;mm cylinder, seed 0, from the thumb's side: thumb in front, index behind it, the "
                          "middle finger on the right.")
                        for d in sorted({r['d_mm'] for r in R[t]}) if (MEDIA / f"20261021-hom_{t}_cylinder{d:g}_base.mp4").exists())
        S.append((t, name, T(["Diameter (mm)", "Held", "Angle RMSE (deg)", "Gains " + " / ".join(strokes), "Pusher slip (mm)",
                              "Off-axis rotation (deg)"], body,
                             cap + " Seeds 0&#8211;2 (object pose jitter of 1&#8202;mm and 2&#8202;deg), scores per seed; stroke gains are "
                             "medians over held runs.")
                  + f'<div class="films">{films}</div>'))

    for task, name in (("roll", "Roll along the thumb"), ("twist", "Twist")):
        if not R[task]:
            continue
        body = []
        steps = ["wy +", "wy -", "wy - 2", "wy + 2"] if task == "roll" else ["wz +", "wz -"]
        for d in sorted({r["d_mm"] for r in R[task]}):
            rs = base(task, d)
            if not rs:
                continue
            gains = {}
            for r in rs:
                for s_ in r.get("steps", []):
                    gains.setdefault(s_["step"], []).append(s_["gain"])
            row = [f"{d:g}", held(rs), seeds(rs, "w_rmse_ratio", 2), " / ".join(fmt(med(gains.get(k, [])), 2) for k in steps)]
            row.append(seeds(rs, "centre_err_max_mm" if task == "roll" else "drift_max_mm"))
            if task == "twist":
                row.append(seeds(rs, "ramp_three_pads_max_deg", 0))
            row.append(fmt(max((r.get("link_force_max_N") or 0.0) for r in rs), 1))
            body.append(row)
        head = (["Diameter (mm)", "Held", "Angular RMSE / step"]
                + (["Gains down / up / up / down", "Centre path error (mm)"] if task == "roll"
                   else ["Gains +z / &#8722;z", "Drift (mm)", "Ramp on three pads (deg)"]) + ["Link force (N)"])
        films = "".join(F(MEDIA / f"20261021-hom_{task}_sphere{d:g}_base.mp4", f"{name}, {d:g}&#8202;mm sphere, seed 0.")
                        for d in sorted({r['d_mm'] for r in R[task]}) if (MEDIA / f"20261021-hom_{task}_sphere{d:g}_base.mp4").exists())
        S.append((task, name, T(head, body, f"{name} across sphere sizes, seeds 0&#8211;2 (scores per seed); gains are medians"
                                + (" of the four steps in their order." if task == "roll" else ".") + " Link force: the largest over the "
                                "three seeds.")
                  + f'<div class="films">{films}</div>'))

    # pin-push drive force and lever (scripts/hom_tasks.py --set f_drive=... --tag fd05/fd075/fd1/fd2; design --lever, variants
    # l30 l60)
    FD = (("base_fd05", 0.5), ("base_fd075", 0.75), ("base_fd1", 1.0), ("base", 1.5), ("base_fd2", 2.0))
    LV = (("l30", "l30_"), ("base", ""), ("l60", "l60_"))
    if any(base("pinpush", v=v) for v, _ in FD if v != "base"):
        def outcome(rs):
            hh = [r for r in rs if r["held"]]
            return [held(rs), fmt(med([r.get("theta_rmse_deg") for r in hh]), 1),
                    " / ".join(fmt(stroke(rs, k), 2) for k in ("up", "down", "back")),
                    fmt(med([r.get("off_axis_max_deg") for r in hh]), 1), fmt(med([r.get("pusher_slip_max_mm") for r in hh]), 1)]

        def rmse(v, d):
            return med([r.get("theta_rmse_deg") for r in base("pinpush", d, v) if r["held"]])

        sizes = sorted({r["d_mm"] for r in R["pinpush"]})
        body_f, body_l = [], []
        for d in sizes:
            for v, f in FD:
                rs = base("pinpush", d, v)
                if rs:
                    body_f.append([f"{d:g}", nf(f) + (" (base)" if v == "base" else "")] + outcome(rs))
            for v, pre in LV:
                rs, g = base("pinpush", d, v), des.get(f"pinpush_{pre}cylinder{d:g}")
                if rs or g is not None or v != "base":
                    body_l.append([f"{d:g}", f"{g['lever_mm']:g}" + (" (base)" if v == "base" else "") if g else
                                   ("35" if v == "l30" else "60") + ", no grasp", fmt(1e3 * g["score"], 1) if g else "&#8212;"]
                                  + (outcome(rs) if rs else ["&#8212;"] * 5))
        big = (16.0, 25.0, 40.0)
        ran = [(v, f) for v, f in FD if base("pinpush", v=v)]
        ranked = sorted(ran, key=lambda vf: med([rmse(vf[0], d) for d in big]))
        fd05 = {d: [r for r in base("pinpush", d, "base_fd05") if r["held"]] for d in (16.0, 25.0)}
        g_base = [1e3 * des[f"pinpush_cylinder{d:g}"]["score"] for d in sizes if f"pinpush_cylinder{d:g}" in des]
        g_lv = {v: [1e3 * des[f"pinpush_{pre}cylinder{d:g}"]["score"] for d in sizes if des.get(f"pinpush_{pre}cylinder{d:g}")]
                for v, pre in LV[::2]}
        r_lv = {v: [rmse(v, d) for d in big if rmse(v, d) is not None] for v in ("l30", "base", "l60")}
        fd2_40 = base("pinpush", 40.0, "base_fd2")

        films = "".join(F(MEDIA / f"20261021-hom_pinpush_cylinder25_{v}.mp4",
                          f"Pin-push on the 25{NB}mm cylinder, seed 0, {'base grasp, ' if v == 'base' else ''}middle finger "
                          f"{des[f'pinpush_{pre}cylinder25']['lever_mm']:g}{NB}mm from the pinch.")
                        for v, pre in LV
                        if (MEDIA / f"20261021-hom_pinpush_cylinder25_{v}.mp4").exists() and f"pinpush_{pre}cylinder25" in des)
        S.append(("drive", "Pin-push drive force and lever", r"""
<p>The middle finger's commanded force $f_{\rm drive}$ sets the friction available to drag the cylinder round the pinch, and the
lever, the distance of its contact from the pinch along the cylinder, sets the torque that friction exerts about the pinch axis.
The base runs press with 1.5&#8202;N on the best-conditioned grasp with a lever of 40&#8202;mm or more (42.5&#8211;52.5&#8202;mm).
Four more drive forces ran on the base grasps, and two grasp searches placed the middle finger 30&#8211;35&#8202;mm and
60&#8211;65&#8202;mm from the pinch with the rest of the search unchanged.</p>
<p>Over 0.5&#8211;2.0&#8202;N the median angle RMSE on the 16, 25 and 40&#8202;mm cylinders is lowest at """
                  + " and ".join(f"{nf(f)}{NB}N (" + ", ".join(fmt(rmse(v, d), 2) for d in big) + f"{NB}deg)" for v, f in ranked[:2])
                  + ", against " + ", ".join(fmt(rmse("base", d), 2) for d in big) + r"""&#8202;deg at 1.5&#8202;N. """
                  + (r"""At 0.5&#8202;N friction no longer drags the 16 and 25&#8202;mm cylinders round: median angle RMSE """
                     + " and ".join(fmt(med([r["theta_rmse_deg"] for r in fd05[d]]), 1) for d in fd05) + "&#8202;deg, the middle "
                     "finger's pad sliding " + " and ".join(fmt(med([r.get("pusher_slip_max_mm") for r in fd05[d]]), 1) for d in fd05)
                     + "&#8202;mm. " if all(fd05.values()) else "")
                  + r"""At 2.0&#8202;N the 40&#8202;mm cylinder stops following the back stroke (gain """
                  + fmt(stroke(fd2_40, "back"), 2) + r""") while the middle finger's pad slides """
                  + fmt(med([r.get("pusher_slip_max_mm") for r in fd2_40 if r["held"]]), 0) + r"""&#8202;mm.</p>"""
                  + T(["Size<br>(mm)", "Drive force<br>(N)", "Held", "Angle RMSE<br>(deg)", "Gains up /<br>down / back",
                       "Off-axis<br>rotation (deg)", "Pusher<br>slip (mm)"], body_f,
                      "Pin-push drive force on the base grasps, seeds 0&#8211;2: medians over held runs.", num_from=1)
                  + r"""<p>Both lever searches return grasps whose least-conditioned finger has a smaller singular value than the base
grasp's (""" + span(g_lv["l30"]) + " and " + span(g_lv["l60"]) + "&#8202;mm/rad against " + span(g_base) + r"""), and on the
16&#8211;40&#8202;mm cylinders both track worse: an angle RMSE of """ + span(r_lv["l30"]) + "&#8202;deg with the middle finger 35&#8202;mm "
                  "from the pinch and " + span(r_lv["l60"]) + "&#8202;deg at 60&#8202;mm, against " + span(r_lv["base"])
                  + r"""&#8202;deg for the base grasps. On the 10&#8202;mm cylinder the 60&#8202;mm lever gives """ + fmt(rmse("l60", 10.0), 1)
                  + "&#8202;deg against the base grasp's " + fmt(rmse("base", 10.0), 1) + r"""&#8202;deg. With the 35&#8202;mm
lever the cylinder follows the down and back strokes with gains of """
                  + span([stroke(base("pinpush", d, "l30"), k) for d in sizes for k in ("down", "back")], 2)
                  + r"""; no 35&#8202;mm grasp keeps the links clear of the 10&#8202;mm cylinder.</p>"""
                  + T(["Size<br>(mm)", "Lever<br>(mm)", "Singular value<br>(mm/rad)", "Held", "Angle RMSE<br>(deg)",
                       "Gains up /<br>down / back", "Off-axis<br>rotation (deg)", "Pusher<br>slip (mm)"], body_l,
                      "Pin-push lever at 1.5&#8202;N, seeds 0&#8211;2: medians over held runs. Singular value: the grasp's "
                      "least-conditioned finger's smallest pad-Jacobian singular value.", num_from=1)
                  + f'<div class="films">{films}</div>'))

    def compare(variants, key_note):
        body = []
        for t in ("pinpush", "press", "roll", "twist"):
            key, nd = SCORE[t]
            for d in sorted({r["d_mm"] for r in R[t]}):
                row = [TASK_LABEL[t], f"{d:g}"]
                for v in variants:
                    rs = base(t, d, v)
                    row.append(f"{held(rs)}, {fmt(med([r.get(key) for r in rs if r['held']]), nd)}" if rs else "&#8212;")
                body.append(row)
        return body

    if any(base(t, v="gapq") for t in R):
        S.append(("impedance", "Servo impedance measured at the reference", r"""
<p>The servo target of a finger with a commanded force is its joint reference plus two offsets: one that moves the impedance's
action point across a gap onto the object's surface, and one that presses with $F$ through the servo spring. Until 2026-10-22 the
gap was measured at the measured joint angles and closed only while the finger was apart (<b>gapq</b>). A finger whose reference
had fallen behind a moving object while its pad still touched then kept a target outside the surface, and its force decayed.
The pin-push's middle finger stopped 0.6&#8202;mm short of a cylinder its own push had turned, and the 63&#8202;mm sphere peeled off the
pads in the rolling steps. The base runs measure the gap at the reference configuration and close it whenever the reference is
behind the surface; a reference ahead of the surface is kept, so a pusher's motion still pushes.</p>"""
                  + T(["Task", "Size (mm)", "Reference: held, score", "Earlier (gapq)"], compare(("base", "gapq"), ""),
                      f"Servo impedance, seeds 0&#8211;2 ({SCORE_TXT}).", num_from=2)))

    if any(base(t, v="ell100") for t in R):
        S.append(("units", "Length scale of Eq. (2) in the tasks", r"""
<p>Eq. (2) compares angular and linear errors through the length scale $\ell$ (linear rows divided by $\ell$), which the HOM paper
does not state. The method page's Exp. A gives the paper's ranking at $\ell=100$&#8202;mm; the tasks use $\ell=20$&#8202;mm.</p>"""
                  + T(["Task", "Size (mm)", "$\\ell=20$&#8202;mm: held, score", "100&#8202;mm", "1&#8202;m (SI)"],
                      compare(("base", "ell100", "si"), ""), f"Length scale of Eq. (2), seeds 0&#8211;2 ({SCORE_TXT}).", num_from=2)))

    FS = [r for r in rows_file(D / "filter_sweep.jsonl") if r.get("status") == "ok"]
    if FS:
        body = []
        for t in ("roll", "twist"):
            for d in sorted({r["d_mm"] for r in FS if r["task"] == t}):
                row = [TASK_LABEL[t], f"{d:g}"]
                for src, kmax, dt in (("reference", 100.0, 0.01), ("planned", 100.0, 0.01), ("planned", 25.0, 0.01),
                                      ("reference", 100.0, 0.02), ("planned", 100.0, 0.02), ("planned", 25.0, 0.02)):
                    rs = [r for r in FS if r["task"] == t and r["d_mm"] == d and r["source"] == src and r["k_max"] == kmax
                          and r["dt"] == dt]
                    row.append(f"{held(rs)}, {fmt(med([r.get('w_rmse_ratio') for r in rs if r['held']]), 2)}" if rs else "&#8212;")
                body.append(row)
        S.append(("estimator", "Contact estimator settings", r"""
<p>The filter of Eqs. (4)&#8211;(6) predicts the contact frames' motion from $S\,\mathcal V_{L_0L_1,\rm ref}$, the reference
relative contact velocities. In the roll and twist tasks, as in the paper's Exp. B, those references are zero and the motion comes
from the object twist, so the gain stays at $k_{\min}$. Fed instead with the relative twist that Eq. (2) plans
(<code>FilterParams.source="planned"</code>), the filter follows the rolling contact and its gain reaches $k_{\max}$; the discrete
update moves $k\,\Delta t$ of the way to the measurement each period, which overshoots above 1. The paper's 500&#8202;Hz gives
$k_{\max}\Delta t=0.2$ at $k_{\max}=100$&#8202;s$^{-1}$; the bench's 50&#8202;Hz gives 2.</p>"""
                  + T(["Task", "Size (mm)", "Reference, 100&#8202;Hz", "Planned, $k_{\\max}$ 100", "Planned, 25",
                       "Reference, 50&#8202;Hz", "Planned, 100", "Planned, 25"], body,
                      "Held count and median angular RMSE / step for the filter's source, $k_{\\max}$ (s$^{-1}$) and controller "
                      "rate; seeds 0&#8211;2.", num_from=2)))

    VAR = ("nofilter", "rate50", "sense", "floop")
    if any(base(t, v=v) for t in R for v in VAR):
        S.append(("variants", "Deployment variants", r"""
<p>Four variants of the base runs move toward the bench. The <b>nofilter</b> variant takes the impedance's action point from the GCF instead of
the filter. The <b>rate50</b> variant runs the controller at 50&#8202;Hz, the servo bus's closed-loop rate. The <b>sense</b> variant runs at 50&#8202;Hz with
the object pose read 40&#8202;ms late with 0.5&#8202;mm and 1&#8202;deg noise, centre-of-pressure noise of 1&#8202;mm and force noise of
0.03&#8202;N. The <b>floop</b> variant adds 10&#8202;mm/s of $v_{\rm sep}$ per newton of force error, a force loop that needs a fingertip force
measurement.</p>""" + T(["Task", "Size (mm)", "Base: held, score", "No filter", "50&#8202;Hz", "50&#8202;Hz, noisy sensing",
                         "Force loop"], compare(("base",) + VAR, ""), f"Deployment variants, seeds 0&#8211;2 ({SCORE_TXT}).",
                        num_from=2)))

    # pinned iCEM against HOM in the same loop (scripts/hom_tasks_mpc.py); the last row per size, seed and controller
    MP = {t: [r for r in rows(f"mpc_{t}") if r.get("status") == "ok"] for t in ("pinpush", "press", "roll", "twist")}

    def pad_force(r):
        tr = [x for x in r.get("trace", []) if x["t"] >= 2.0 - 1e-9]
        return max(st.mean(x["F"][f] for x in tr) for f in ("thumb", "index", "middle")) if tr else None

    def arm(t, k, d=None):
        return [r for r in MP[t] if r["controller"] == k and (d is None or r["d_mm"] == d)]

    if any(MP.values()):
        body, smooth = [], []
        for t in ("pinpush", "press", "roll", "twist"):
            key, nd = SCORE[t]
            if t == "twist":
                key, nd = "w_rmse_ratio", 2
            for d in sorted({r["d_mm"] for r in MP[t]}):
                c = {k: arm(t, k, d) for k in ("hom", "mpc")}
                body.append([TASK_LABEL[t], f"{d:g}"]
                            + [held(c[k]) for k in ("hom", "mpc")]
                            + [fmt(med([r.get(key) for r in c[k] if r["held"]]), nd) for k in ("hom", "mpc")]
                            + [fmt(med([pad_force(r) for r in c[k] if r["held"]]), 1) for k in ("hom", "mpc")]
                            + [fmt(med([r.get("link_force_max_N") for r in c[k] if r["held"]]), 1) for k in ("hom", "mpc")])
            c = {k: arm(t, k) for k in ("hom", "mpc")}
            if c["hom"] or c["mpc"]:
                smooth.append([TASK_LABEL[t]]
                              + [fmt(med([r.get(m) for r in c[k] if r["held"]]), nd2) for m, nd2 in
                                 (("cmd_reversals", 1), ("cmd_rate_rms", 2), ("wall_s", 0)) for k in ("hom", "mpc")])
        films = "".join(F(MEDIA / f"20261021-hom_vs_mpc_{t}_{o}{d}.mp4",
                          f"{name}, {d}{NB}mm {o}, seed 0: HOM (left) and pinned iCEM (right) from the same pick.",
                          MEDIA / f"20261021-hom_vs_mpc_{t}_{o}{d}.jpg")
                        for t, name, o, ds in (("pinpush", "Pin-push", "cylinder", (25, 40)), ("press", "Press", "cylinder", (25, 40)),
                                               ("roll", "Roll", "sphere", (40, 63)), ("twist", "Twist", "sphere", (40, 63)))
                        for d in ds if (MEDIA / f"20261021-hom_vs_mpc_{t}_{o}{d}.mp4").exists())
        S.append(("mpc", "MPC baseline", r"""
<p>From the start of each task (2.0&#8202;s, after the HOM pick and lift) the planner of the 2026-10-14 scale study, pinned iCEM,
replaces the HOM controller. Every 10&#8202;ms it samples 64 plans of the nine servo targets (two iterations of 32, three knots over
0.5&#8202;s, knot noise 0.08&#8202;rad, the first knot fixed to the applied target). It rolls them out on the plant's own pads model with
<code>mujoco.rollout</code> and applies the best. Between replannings the targets follow that plan at the 1&#8202;ms physics step.
A plan costs, over its 50 samples at 10&#8202;ms,
$$J=\sum_{k=1}^{49}c_k+10\,c_{50}+\sum_{i=1}^{2}\lVert\mathbf u_{i+1}-\mathbf u_i\rVert^2,$$
with $\mathbf u_i$ the knots (rad) and the sample cost $c_k$ of the task:
$$\begin{aligned}
c^{\rm pin\text{-}push,\,press}&=\Big(\tfrac{\theta-\theta_{\rm ref}}{2^\circ}\Big)^2+\Big(\tfrac{\psi-\psi_0}{5^\circ}\Big)^2
+\textstyle\sum_f{\rm clip}\big(\tfrac{0.2\,{\rm N}-F_f}{0.2\,{\rm N}},0,1\big)+10\,[z<z_0-15\,{\rm mm}],\\
c^{\rm roll,\,twist}&=\tfrac{\lVert\bm\omega-\bm\omega_{\rm ref}\rVert^2}{(10^\circ/{\rm s})^2}+\tfrac{\lVert\mathbf p-\mathbf p_{\rm ref}\rVert^2}{(5\,{\rm mm})^2}
+\textstyle\sum_f{\rm clip}\big(\tfrac{0.5\,{\rm N}-F_f}{0.5\,{\rm N}},0,1\big)+10\,[z<z_0-15\,{\rm mm}],
\end{aligned}$$
where $\theta$ is the pinch angle and $\theta_{\rm ref}$ the commanded angle, $\psi$ the angle between the cylinder's axis and the
plane normal to the pinch axis ($\psi_0$ at the task start), $F_f$ a finger tip's net force on the object from a contact sensor of
the planner's model, $\bm\omega$ the object's angular velocity, $\mathbf p$ its centre and $z$ its height. The references are the
HOM schedule's: $\mathbf p_{\rm ref}$ is the centre at the task start plus, in the roll, the integral of the velocity of rolling on
the thumb. The HOM arm runs in the same loop and logs the same commands.</p>"""
                  + mpc_findings(MP, arm, pad_force)
                  + T(["Task", "Size (mm)", "Held<br>HOM", "<br>MPC", "Score<br>HOM", "<br>MPC", "Pad force (N)<br>HOM", "<br>MPC",
                       "Link force (N)<br>HOM", "<br>MPC"],
                      body, "Pinned iCEM (MPC) against HOM, seeds 0&#8211;2: held count, and medians over held runs of the score (angle "
                      "RMSE in deg for pin-push and press, angular RMSE / step for roll and twist), the pad force and the link force.",
                      num_from=2)
                  + T(["Task", "Reversals (1/s)<br>HOM", "<br>MPC", "Command rate (rad/s)<br>HOM", "<br>MPC",
                       "Time per run (s)<br>HOM", "<br>MPC"], smooth,
                      "Command smoothness and computation per task, medians over the held runs of both sizes.")
                  + films))

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
            body.append([NAMES.get(o["name"], esc(o["name"])), esc(o.get("action", "")),
                         f"{c.get('engineering_days', '&#8212;')}", fmt(gpu, 1) if isinstance(gpu, (int, float)) else "&#8212;"])
        S.append(("rl", "Reinforcement learning: options and costs", r"""
<p>The bench measures the nine joint angles on the servo bus (closed loop at 50&#8202;Hz) and the object pose from two AprilTags at
30&#8202;frames/s. The HOM controller costs 1.1&#8202;ms per period on one core, against 0.26&#8202;s for one step of 2,048 GPU
environments with pads (7,800 environment steps/s, 43&#8202;min per 20&#8202;M-step seed). The options and their costs for four tasks
at three seeds (<code>rl_assessment.json</code>, written on 2026-10-21 against the earlier task definitions):</p>"""
                  + T(["Option", "Action", "Engineering (days)", "GPU hours"], body,
                      "Learning options. GPU hours: 20&#8202;M steps per seed, three seeds per task, one policy per task over its "
                      "four sizes.", num_from=2)
                  + "<p>" + esc(A.get("recommendation", "")) + "</p>"))

    S.append(("limits", "Limitations", r"""
<p>All results are simulation on one contact model and one plant, with 25&#8202;g objects. The pinch grasps are one layout family (the
scale study's tripods). The task schedules are fixed slider sequences, as in the paper's teleoperation, with no feedback from the
task score. The MPC plans on the plant's own model from its full state, which a controller on the bench does not have. The lever
searches change the grasp's conditioning together with the lever. In the twist ramp a finger link presses the sphere with up to """
              + fmt(max(lf.get("twist") or [0.0]), 0) + r"""&#8202;N, outside the tips' sensing.</p>"""))
    S.append(("future", "Future work", r"""
<ul>
<li>MPC within HOM's force budget: in <code>scripts/hom_tasks_mpc.py</code>, a cost on each pad's force above 5&#8202;N and on
the force of contact sensors on every finger link, then pin-push and press at 25 and 40&#8202;mm, seeds 0&#8211;2. An angle RMSE
still under 1&#8202;deg with pads under 5&#8202;N and no link contact would show that planning closes HOM's tracking error; an RMSE
near HOM's 2.5&#8202;deg would show that the squeeze does.</li>
<li>Lever at a fixed layout: a <code>--layout</code> option of <code>scripts/hom_tasks.py design --lever</code> that keeps the base
grasp's tripod layout, so the 35 and 60&#8202;mm levers keep its conditioning. A 35&#8202;mm lever within 0.5&#8202;deg of the base
grasp's angle RMSE would put the lever table's losses on the grasps' conditioning.</li>
<li>Drive force: ten seeds at 0.75 and 1.0&#8202;N on the 16 and 25&#8202;mm cylinders (<code>--set f_drive=0.75 --seeds 0 1 2 3 4 5
6 7 8 9</code>) before 0.75&#8202;N replaces 1.5&#8202;N as the pin-push's default. One of three 16&#8202;mm seeds slipped at
0.75&#8202;N (8.8&#8202;deg).</li>
<li>Twist without link contact: a layout or palm height that keeps every link 1&#8202;mm off the sphere through a 90&#8202;deg ramp; a
larger ramp on three pads with link force under 0.5&#8202;N would show that the link's contact limits the present ramp.</li>
<li>Bench dry run with <code>src/morphohand/control/hom_bench.py</code> at 50&#8202;Hz on recorded tag poses, then the pin-push on a
25&#8202;mm cylinder on the hand, with the middle finger's tip sensor as the force signal.</li>
</ul>"""))
    S.append(("files", "Files", f"""
<p>Folder <code>docs/experiments/20261021-hom_framework/</code>: the demonstration clips (<code>*.mov</code>), <code>design.json</code>
(grasps), <code>rows/&lt;task&gt;.jsonl</code> (one line per rollout with its trace), <code>filter_sweep.jsonl</code>,
<code>rows/mpc_&lt;task&gt;.jsonl</code> (the MPC comparison, both arms, with the command log), <code>rl_assessment.json</code>,
<code>media/</code> (films; <code>20261021-hom_vs_mpc_*.mp4</code> side by side). Rows of the earlier task definitions are in commit
8f8fa3ed. Scripts: <code>scripts/hom_tasks.py</code> (design, design <code>--lever</code>, run),
<code>scripts/hom_tasks_mpc.py</code> (the MPC baseline), <code>src/morphohand/control/hom_bench.py</code> (the controller between
the bench's signals and the servo targets, with a 50&#8202;Hz dry run), <code>scripts/hom_framework_page.py</code>. The method page:
<a href="{relp(METHOD)}">Hand-object mechanism control: the method and its MuJoCo implementation</a>.</p>"""))

    pp, pr, ro, tw = base("pinpush"), base("press"), base("roll"), base("twist")
    if pp and pr and ro and tw:
        ppl = [r["theta_rmse_deg"] for r in pp if r["d_mm"] >= 16 and r["held"]]
        lede = (f"On the CAD-model SR2 hand in MuJoCo the library's HOM controller held {held(pp)} pin-push rollouts (a middle "
                f"finger dragging a cylinder round a thumb-index pinch by friction) and tracked the &#177;15{NB}deg command with an "
                f"RMSE of {span(ppl)}{NB}deg on 16&#8211;40{NB}mm cylinders; the press held {held(pr)}, the roll {held(ro)} and the "
                f"twist {held(tw)}.")
        summary = (f"Under the HOM controller the pin-push held {held(pp)} rollouts (angle RMSE {span(ppl, dash='-')} deg on 16-40 mm "
                   f"cylinders), the press {held(pr)}, rolling {held(ro)} and twisting {held(tw)}.")
        fd = [(v, f) for v, f in (("base_fd05", 0.5), ("base_fd075", 0.75), ("base_fd1", 1.0), ("base", 1.5), ("base_fd2", 2.0))
              if base("pinpush", v=v)]

        def fd_rmse(v):
            return [med([r["theta_rmse_deg"] for r in base("pinpush", d, v) if r["held"]]) for d in (16.0, 25.0, 40.0)]
        if len(fd) > 1:
            bv, bf = min(fd, key=lambda vf: med(fd_rmse(vf[0])))
            lede += f" A drive force of {nf(bf)}{NB}N tracks best ({span(fd_rmse(bv))}{NB}deg)."
        mc = [r for t in ("pinpush", "press") for r in MP[t] if r["controller"] == "mpc" and r["held"]]
        hh = {t: [r["theta_rmse_deg"] for r in MP[t] if r["controller"] == "hom" and r["held"]] for t in ("pinpush", "press")}
        hc = [r for t in ("pinpush", "press") for r in MP[t] if r["controller"] == "hom" and r["held"]]
        if mc and hc:
            th_m = [r["theta_rmse_deg"] for r in mc]
            pf_m, pf_h = [pad_force(r) for r in mc], [pad_force(r) for r in hc]
            vs = ", ".join(f"{span(hh[t])}{NB}deg in the {t.replace('pinpush', 'pin-push')}" for t in ("pinpush", "press") if hh[t])
            allm = [r for t in MP for r in MP[t] if r["controller"] == "mpc" and r["held"]]
            lede += (f" Pinned iCEM on the plant's own model tracks the pinch angle to {span(th_m)}{NB}deg (HOM {vs}) by squeezing "
                     f"with {span(pf_m, 0)}{NB}N per pad against {span(pf_h, 0)}{NB}N, tracks the spheres no better than HOM, and costs "
                     f"{span([r['wall_s'] for r in allm], 0, 1 / 60)}{NB}min of computation per rollout.")
            summary += (f" Pinned iCEM tracks the pinch angle to {span(th_m, dash='-')} deg against HOM's "
                        + " and ".join(f"{span(hh[t], dash='-')} deg in the {t.replace('pinpush', 'pin-push')}"
                                       for t in ("pinpush", "press") if hh[t])
                        + f" but squeezes with {span(pf_m, 0, dash='-')} N per pad against {span(pf_h, 0, dash='-')} N.")
    else:
        lede = summary = "Rows pending."
    assemble(S, "HOM tasks across object sizes", "HOM control of four demonstrated tasks across object sizes on the CAD-model SR2 hand",
             "Reorientation mechanisms and control &#183; 2026-10-22", lede,
             f"Built {time.strftime('%Y-%m-%d %H:%M')} from rows/*.jsonl at commit {git_short()}.", RESULTS, summary)


if __name__ == "__main__":
    which = sys.argv[1:] or ["method", "results"]
    if "method" in which:
        method_page()
    if "results" in which:
        results_page()
