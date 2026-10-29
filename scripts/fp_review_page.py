#!/usr/bin/env python3
"""Review page of the foundational-pose paper (arXiv 2409.14158), of how this project samples hand layouts, and of a
plan to reproduce the paper's Pareto front and t-SNE analysis on the SR2 hand with HOM control, 2026-10-29.

Owner, 2026-10-29: "create/update a docpage html that more or less covers the original foundational pose paper; how we
have interpreted that paper, very loosely, for morphology sampling (we do sobol sampling + CEM sampling from a keyframe
for raw coverage and grasping, [a] proposal [is] RRT + a SQP); evaluate the feasibility / propose a plan of reproducing
the dense pareto frontier / T-SNE analysis provided in the FP paper, but updated with our hardware, sensing, and
hom-sim controls ... you cannot separate the quality of the grasp from the dexterous in-hand controls that follow it,
but there is naturally some separation in making the problem more tractable. or does there have to be?" The page names
the papers as [1] and [2] and nobody else (owner, 2026-10-21).

Reads docs/experiments/20261029-foundational_pose_review/rows/ (scripts/fp_sr2_toy.py, scripts/fp_hom_check.py) and
media/ (scripts/fp_review_figures.py). Mathematics through scripts/texsvg.py.

    .venv/bin/python scripts/fp_review_page.py
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
import fp_review_figures as FF  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = ROOT / "docs/experiments/20261029-foundational_pose_review"
MEDIA = D / "media"
ROWS = D / "rows"
OUT = D / "20261029-foundational_pose_hand_design.html"
TEMPLATE = ROOT / "scripts/fp_review_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
NB = "&#8202;"
REFS = [
    '[1] S. Wang, J. Oh and N. S. Pollard, "The Foundational Pose as a Selection Mechanism for the Design of '
    'Tool-Wielding Multi-Finger Robotic Hands", IEEE-RAS Humanoids 2025, arXiv 2409.14158v2 '
    '(<code>relevant_papers/pollard_foundational_pose.pdf</code>).',
    '[2] S. Wang, J. Oh and N. S. Pollard, "Relative Contact Velocity-Controlled Hand-Object Mechanism for Dexterous '
    'Tool Manipulation", arXiv 2609.25619v2, 2026; reviewed on the companion page '
    '<code>docs/experiments/20261029-hom_control_review/</code>.',
    '[3] D. J. Montana, "The kinematics of contact and grasp", IJRR 7(3), 1988.',
    '[4] L. van der Maaten and G. Hinton, "Visualizing data using t-SNE", JMLR 9, 2008.',
]
FIG, TAB = [0], [0]


def esc(s):
    return html.escape(str(s), quote=True)


def fill(s, **kw):
    for k, v in kw.items():
        s = s.replace(f"<<{k}>>", str(v))
    left = re.findall(r"<<(\w+)>>", s)
    if left:
        raise KeyError(f"unfilled tokens {left}")
    return s


def data_uri(p, mime):
    return f"data:{mime};base64,{base64.b64encode(Path(p).read_bytes()).decode()}"


def img(path, caption):
    FIG[0] += 1
    return (f'<figure><img src="{data_uri(path, "image/png")}" alt="">'
            f'<figcaption>Figure {FIG[0]}. {caption}</figcaption></figure>')


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


def load(name):
    return FF.load(name)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------------------------------------ numbers
def numbers():
    N = {}
    fig = json.loads((ROWS / "figures.json").read_text()) if (ROWS / "figures.json").exists() else {}
    N["fig"] = fig
    for name in ("rrt2d", "rrt6d", "sobol6d"):
        p = ROWS / f"{name}_summary.json"
        N[name] = json.loads(p.read_text()) if p.exists() else {}
        rows = load(name)
        N[name]["solver_calls"] = len(rows)
        ms = [r["ms"] for r in rows if "ms" in r]
        N[name]["ms_median"] = st.median(ms) if ms else None
        if rows:
            cand, raw, sc, mean, par = FF.table(rows)
            N[name]["cand"] = len(cand)
            N[name]["pareto"] = int(par.sum())
            N[name]["raw"] = raw
            N[name]["mean"] = mean
            N[name]["d"] = np.array([r["d"] for r in cand])
            N[name]["fail_fp"] = {fp: sum(1 for r in rows if not r["fp"].get(fp, {"ok": True}).get("ok", True))
                                  for fp in FF.FPS}
    chk = [r for r in load("hom_twist_check") if r.get("status") == "ok"]
    N["check"] = chk
    return N


def check_stats(chk):
    """Per layout: the kinematic tripod rotation range, the median twist ramp on three pads and velocity RMSE, held."""
    by = {}
    for r in chk:
        by.setdefault((r["x_sep_mm"], r["y_sep_mm"]), []).append(r)
    lay = []
    for (xs, ys), rr in sorted(by.items()):
        held = [r for r in rr if r.get("held")]
        ramp = [r.get("ramp_three_pads_max_deg") or 0.0 for r in rr]
        lay.append({"x": xs, "y": ys, "tmr": rr[0]["kin"]["tmr_deg"], "mft": rr[0]["kin"]["mft_mNm"],
                    "smin": rr[0]["kin"]["smin_mm"], "ramp": st.median(ramp),
                    "rmse": st.median([r["w_rmse_ratio"] for r in held]) if held else None,
                    "held": len(held), "n": len(rr)})
    return lay


# ------------------------------------------------------------------------------------------------ sections
def sec_terms():
    terms = [
        ("Foundational pose (FP)", "a pose of hand and tool from which the tool can turn in both directions. Every "
         "contact is kept, and the tool is in static equilibrium under its working force [1]. It is a snapshot of a "
         "tool-hand parallel mechanism."),
        ("Parallel mechanism", "a mechanism in which several kinematic chains connect a base to one output link; here "
         "the fingers are the chains, the palm the base and the tool the output link."),
        ("Design space, standardized", "the vector of design parameters $d$, each rescaled to the interval "
         "$[-0.5, 0.5]$ by its bounds, so that distances between designs are comparable across parameters."),
        ("Candidate", "a sampled design that reaches every FP; the paper's sampling efficiency is candidates per RRT "
         "call."),
        ("RRT", "rapidly exploring random tree: each step draws a random point, finds the nearest existing sample and "
         "steps from it toward the point, so the samples grow outward from a seed."),
        ("SQP", "sequential quadratic programming: a constrained nonlinear program solved as a series of quadratic "
         "programs, each on a linearization of the constraints (MATLAB's fmincon in its active-set mode, SciPy's "
         "SLSQP, SNOPT)."),
        ("Sobol sequence", "a low-discrepancy sequence that covers a box more evenly than independent random draws."),
        ("CEM", "cross-entropy method: sample around a mean, keep the best fraction, refit the mean and spread, repeat."),
        ("Pareto front", "the candidates that no other candidate beats on every score at once."),
        ("t-SNE", "t-distributed stochastic neighbour embedding [4]: a map of high-dimensional points into a plane "
         "that keeps each point's nearest neighbours near it. Distances between distant groups, and the groups' "
         "sizes, are not kept."),
        ("PCA", "principal component analysis: the directions of largest variance; a linear map that keeps "
         "distances along them."),
        ("TMR, MSS, MFT", "the paper's metrics per FP. Tool motion range (deg) is better when larger. Mean contact sliding "
         "speed (mm/s) and the maximum finger torque to hold a 10 N tool force (N m) are better when smaller. "
         "Each becomes a score from 0 (worst candidate) to 100 (best)."),
        ("Toy metrics", "this page's counterparts on the SR2 hand. They are the rotation range with the contacts fixed on "
         "the object (deg) and the largest joint torque under the task's grasp forces along that rotation (mN m). The "
         "third is the smallest singular value of the three pad Jacobians (mm/rad): how far the weakest joint direction "
         "moves the pad per radian."),
        ("Layout", "the six palm-frame mount coordinates (mm) of the SR2 hand's three gantries; the symmetric tripod is "
         "the 2-D family set by the thumb-to-pair distance $x_{sep}$ and the index-to-middle distance $y_{sep}$."),
        ("HOM", "the hand-object mechanism controller of [2], which commands each finger's rolling, spinning, sliding "
         "and separation relative to the object."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"


def sec_paper():
    t = r"""<p>The question of the paper [1] is how to design a multi-finger hand that can wield a tool. Its hypothesis reduces the
question to a pose. If a hand can reach a <i>foundational pose</i> with the tool, it can wield the tool from that pose.
Over 100,480 sampled hands, 10,785 reached all three of its poses, and 99.42 % of the 10,000 in its six-dimensional
run then moved the tool in a kinematic planner.</p>
<p>The intuition comes from watching sculptors. A sculptor first sets a tool in the hand, then moves the fingers in a
synchronized, repeatable way while the tool cuts. In the paper this is read as a parallel mechanism. Each finger is a chain
from the palm to the tool, and the contacts are its joints. Once the tool's motion is chosen, the closed chains fix
how every finger must move. Three such mechanisms are identified:</p>
<ul>
<li><b>Carve</b>: three fingers in a triangle round the tool's cross-section.</li>
<li><b>Poke</b>: two pinches at right angles along the tool's two short axes, four fingers in all.</li>
<li><b>Press</b>: a pinch at the middle and one contact at each end, four fingers.</li>
</ul>
<p>In all three the tool turns in a plane. The foundational pose is a pose that the mechanism passes through in its full range of motion. In it the tool can turn
either way, every contact is kept, and the tool is in static equilibrium under a cutting force at its tip.</p>
<p><b>The design space.</b> A template of four identical three-joint fingers has six parameters. They are the distal link's length (each finger
is 100 mm long), the palm's length, half-width and angle, the finger's diameter, and the thumb's rotation about its
axis. The links are cylinders and capsules, simple shapes for rolling and sliding contact. The
hardware is an aluminium skeleton under a printed elastomer skin.</p>
<p><b>Reaching a pose is a set of equations.</b> For each contact, a point on the tool and a point on the finger must
coincide, and their surface normals must point against each other:</p>
$$p_{OC} = p_{OP}, \qquad z_{OC} + z_{OP} = 0. \qquad (1)$$
<p>$p$ and $z$ are a frame's origin and unit normal, $C$ the frame on the tool's surface, $P$ the frame on the finger's
surface, and $O$ the tool's frame. Eq. (1) gives six scalar equations per contact. Each contact has eight unknowns: three place the contact on the tool, and five set the finger's three
joints and its contact point. The hand's pose relative to the tool adds six more. For $n$ contacts that is $6n$ equations in $8n + 6$ unknowns:</p>
$$\mathrm{reach}_{FP}(\theta, d) = 0, \qquad \theta \in \mathbb{R}^{8n+6}. \qquad (2)$$
<p>The system has $2n + 6$ free directions, 12 for carve and 14 for poke and press. A reachable pose is therefore not unique, and one is picked by optimization:</p>
$$\min_{\Delta\theta}\ \mathrm{sampling\_cost}(\Delta\theta) \quad \text{s.t.}\quad \mathrm{reach}_{FPs} = 0,\ \ \mathrm{collision} \ge 0,\ \ \theta_{\min} \le \theta + \Delta\theta \le \theta_{\max}.$$
<p>This problem is solved by MATLAB's fmincon in its active-set mode, an SQP method. The cost is specific to each
pose: for carve, the share of the contact force that drives the cut; for poke and press, how closely the two pinching normals oppose. A small term discourages the finger's side-to-side joint.</p>
<p><b>Sampling grows from a hand that works.</b> The paper's analogy is gas pumped into a sealed chamber of unknown
shape. The gas fills the chamber without passing its walls, and its spread maps the interior. An RRT plays the gas. It
starts from a hand designed by hand that reaches all poses. Each call draws a uniform point in the standardized design
space and steps from the nearest candidate toward it. A new design becomes a candidate under three conditions. It must differ enough from every candidate. The SQP must bring
it to every pose, starting from that nearest candidate's solution. Statics must hold as well. Sampling stops at a target count or when candidates per call stay too low.</p>
<p><b>In the evaluation the tool is moved and the fingers follow.</b> From each pose the tool is turned at a constant rate.
Each step, the finger joint rates are chosen by a quadratic program: every contact stays closed and the sliding
is least.</p>
$$\min_{\dot u_f}\ \lVert E_{xy}\, v_{CP}\rVert^2 \quad \text{s.t.} \quad E_z\, v_{CP} = 0,\ \ \text{joint ranges},\ \ \text{linearized collision} \ge 0.$$
<p>$v_{CP}$ is the velocity of the finger's contact point relative to the tool's, $E_{xy}$ keeps its tangential part
and $E_z$ its normal part. Rolling-contact kinematics [3] then move the contact points over both surfaces, and the path
ends when the program fails or statics break. This is backdriving the mechanism from its output link, like turning an
engine by its shaft. The three metrics are the tool's rotation range (TMR), the mean sliding speed (MSS) and the
largest joint torque to resist a 10 N cutting force (MFT). Three poses and three metrics make nine scores. They are summarized by the Pareto front over the nine and by each
candidate's mean score.</p>
<p><b>Results.</b> In a two-parameter slice (palm length and width), 785 candidates came from 78,511 RRT calls before
the efficiency fell to 1 %. Every one turned the tool from every pose. The landscapes showed clusters, contours and
gradients, and the non-optimal candidates clustered too. In the full six dimensions, 10,000 candidates came from
21,969 calls, and the efficiency was still 45.52 % at the end. As balls of half their least spacing, the 2-D candidates
fill 68.50 % of the standardized square; the 6-D candidates fill $1.11\times10^{-4}$ % of the cube. A t-SNE map of the
six parameters, coloured by mean score, falls into isolated islands, some with gradients. A prototype built from a
Pareto-optimal 2-D candidate cut Play-Doh under open-loop joint control. A fourth contact, the tool braced on the index
finger's proximal link, cut much better, though no step of the method had considered it.</p>
<p>The paper lists its limits plainly. Only three poses were tried. The pose solve does not look at the metrics, so
some hands reach their poses and then wield poorly. Many samples are needed to cover a high-dimensional space, and it is hard to visualize. Its
future work names <i>tool loading</i>: picking a tool up and bringing it into a foundational pose. That is the
subject of the HOM paper [2].</p>"""
    return t


def sec_two_papers():
    t = r"""<p>Read together, the two papers are two halves of one method. The FP paper chooses hands by a pose, and
the HOM paper controls a hand from such a pose. Two details tie them together.</p>
<p>First, the FP paper's evaluation program is a special case of the HOM paper's least squares. HOM's Eq. (2) picks
joint rates $u$ that bring each finger's relative velocities $v_i(u)$ close to references $v_{i,\mathrm{ref}}$:</p>
$$u_{\mathrm{ref}} = \arg\min_u \sum_{i} w_i \lVert v_{i,\mathrm{ref}} - v_i(u)\rVert^2 + w_{\mathrm{damp}}\lVert u\rVert^2 + \dots$$
<p>Set the references to zero and turn the tool through the object's reference twist. Weight the slide heavily, hold
the separation at zero as a constraint, and leave roll and spin free. What remains is the FP paper's program: least
sliding, closed contacts, the tool's motion prescribed. The FP paper's
planner is a kinematic HOM controller with every finger held in contact.</p>
<p>Second, each paper's open problem is the other's subject. The HOM paper says that its controller needs a good
starting pose and that reaching one is hard. The FP paper supplies poses by design and lists loading a tool into its
pose as future work, which is what HOM's pinch brake and pusher do.</p>
<p>For design, this means a foundational pose can be the starting state of a HOM-controlled task, and the HOM
controller can be the evaluation that follows it. Section 7 asks whether the two halves must be scored together or can be scored one after the other.</p>"""
    return t


def sec_ours(N):
    rows = [
        ["Design space", "6: link length, palm length, width and angle, finger diameter, thumb rotation",
         "6 gantry mount coordinates (thumb x ±30 and y ±55 mm, others ±30 mm)",
         "2 separations, $x_{sep}$ and $y_{sep}$ (symmetric tripods)",
         "2 separations, then all 6 coordinates"],
        ["Sampler", "RRT from a seed hand, standardized space", "scrambled Sobol, 8,198 layouts",
         "a 9 by 9 grid and three 17-layout families", "RRT from the CAD-nominal layout; Sobol for comparison"],
        ["Pose per design", "SQP (fmincon) onto three FPs, with collision and statics",
         "geometric grasp fitter at 4 cells (straddle 32 or 40 mm, thumb offset 10 or 20 mm)",
         "closed-form inverse kinematics on a grid of palm heights", "SQP (SLSQP) onto two HOM-task poses"],
        ["Evaluation", "kinematic path: least sliding at closed contacts", "dynamic: retention of a turned "
         "screwdriver at 5 clip levels, then repeated draws with errors", "dynamic: hold under disturbances; HOM turn",
         "kinematic: rotation with fixed contacts, torque, conditioning"],
        ["Selection", "Pareto front of 9 scores; t-SNE", "pass or fail funnel: 6,629, 535, 227",
         "ridge fits against object size", "Pareto front of 6 scores; t-SNE, PCA"],
    ]
    t = r"""<p>The project samples SR2 layouts over the six gantry coordinates and screens them by dynamic simulation.
The foundational-pose idea enters loosely: a grasp found once is replayed or refit per design, and a design survives
if the grasp holds through a task. Table 1 sets three of our studies beside the paper.</p>"""
    t += table(["Stage", "FP paper [1]", "Sobol screen, Aug.-Sep.", "Scale study, Oct.", "Toy, this page"], rows,
               "The foundational-pose pipeline and the project's layout studies.", num_from=9)
    t += r"""<p><b>The Sobol screen</b> (<code>docs/experiments/20260831-real_v1-sobol8192/</code>,
<code>scripts/real_v1_design_search.py</code>) drew 8,198 layouts. A scrambled Sobol sequence (seed 20260830) gave
6,144 uniform layouts and 2,048 biased toward wide ones, and six hand-placed anchors were added. Four gates
followed:</p>
<ul>
<li>6,629 layouts lie inside the gantries' measured travel. Every rejection is on an $x$ rail.</li>
<li>A geometric fitter placed a grasp of the screwdriver at four cells per layout. 3,298 layouts grasp it at one cell
or more.</li>
<li>A turn of the screwdriver followed by a hold and a proof lift, at five clip levels of the servo budget, made 57,220
rollouts. 535 layouts keep the tool at some level.</li>
<li>Repeated draws with modelling errors confirmed 227.</li>
</ul>
<p>Six were exported to the bench. Five passed the deployment gates, and two produced a working band.</p>
<p><b>CEM from a keyframe.</b> In the per-design pipeline and the 108-hand search, the grasp is refined by CEM. It
samples nine constant finger targets around the fitted <code>open_ik</code> keyframe (population 80, 120 to 200
iterations, about 83 s per design on real_v1) and scores a lift and hold. That score optimizes the hold, and on
real_v1 the hold trades against the turn. CEM moved the hand rv04_mid's held turn from cos 0.997 to 0.625.</p>
<p><b>What the population looks like.</b> A uniform draw over a box spreads its variance evenly. PCA of the 8,198
layouts gives 40 % to the thumb's $y$, whose rail is longest, and about 12 % to each other coordinate. Selection breaks
the symmetry: the 535 survivors give 47, 21 and 17 % to their first three components. A check of 2026-09-03 found
the t-SNE of the population statistically indistinguishable from the t-SNE of a uniform box.</p>
<p><b>The scale study</b> (<code>docs/experiments/20261008-hand_object_scale/</code>) is the counterpart of the
paper's 2-D slice. Its symmetric tripods reduce the layout to the two separations. Holding and turning favour
different layouts. For cylinders, the best hold sits at an axis-to-mount distance of 1.01 r + 25.4 mm, and the
furthest turn at 1.11 r + 31.3 mm, where r is the cylinder's radius.</p>
<p><b>RRT with SQP.</b> The alternative raised in discussion uses the paper's own machinery in place of the Sobol draw and the
grasp search. The layouts are explored by an RRT, and each layout is brought into the task's pose by an SQP. The project has
one SQP piece already. In the Drake port a joint layout and posture problem was solved with SNOPT, an SQP solver. Six gantry
offsets and nine finger angles reached three fingertip targets in 1.15 ms, with 3.66 mm of clearance (page
<code>20261001-drake_sr2_planning</code>). Section 5 runs the whole pipeline on the SR2 hand.</p>"""
    return t


def sec_toy(N):
    r2, r6, sb = N["rrt2d"], N["rrt6d"], N["sobol6d"]
    t = r"""<p>The toy reproduces the paper's pipeline on the SR2 hand with two foundational poses taken from the HOM
tasks (<code>scripts/fp_sr2_toy.py</code>). Each pose is a grasp in which a HOM sub-assembly can move the object both
ways:</p>
<ul>
<li><b>Pinch</b> (the pin-push): thumb and index pinch a 25 mm cylinder lying flat. The middle finger touches the
cylinder on the index's side, at least 40 mm along it. The pinch is a revolute joint and the middle finger its
actuator.</li>
<li><b>Tripod</b> (the roll and twist): three pads on the equator of a 40 mm sphere.</li>
</ul>
<p>The unknowns per pose are the object's position and heading and the contact coordinates on its surface. The
joints follow from closed-form inverse kinematics, since each fingertip pad is a sphere. Eq. (1) then reduces to
placing each pad's centre one pad radius off the object along its normal. The constraints are those of our task
design. The servo limits apply less 3 deg. The distal link must face the object: within 25 deg for the pinch, within the
pad's front half for the tripod. Fingers keep 1 mm apart, and other links keep 0.5 mm from the object.</p>
<p>The SQP (SciPy's SLSQP) finds the pose's most interior point: it maximizes a number $t$ subject to every scaled
constraint slack being at least $t$. A pose is reached when $t \ge 0$. A coarse grid over the same unknowns gives the
start, or the nearest candidate's solution does, as in the paper. Each pose has three metrics after the paper's. The first is the object's rotation with the contacts fixed on it, in
both directions, each capped at 60 deg. The second is the largest joint torque under the task's grasp forces along that
rotation. The third is the pads' smallest Jacobian singular value. The paper's sliding
speed is zero here by construction. Two poses and three metrics give six scores.</p>"""
    eff2 = r2.get("candidates", 0) / max(r2.get("calls", 1), 1)
    t += fill(r"""<p><b>The 2-D case.</b> The RRT over the symmetric tripods ran <<c2>> calls in <<s2>> s and stopped at
<<n2>> candidates. By then only <<e2>> % of the last 400 calls had yielded a candidate. Most calls late in the
run fell too close to an existing candidate (closer than 0.03 in standardized units) and were never solved: the
candidates had filled the feasible region. That region spans $x_{sep}$ = <<x0>> to <<x1>> mm. Wider thumbs cannot
reach the pinch with the distal link facing the cylinder. Figure 1 shows the six landscapes.</p>""",
              c2=f"{r2.get('calls', 0):,}", s2=f"{r2.get('seconds', 0):.0f}", n2=r2.get("candidates", 0),
              e2=f"{100 * r2.get('efficiency_last', 0):.2f}", x0=f"{N['d2_x'][0]:.0f}", x1=f"{N['d2_x'][1]:.0f}")
    t += N["land_text"]
    t += img(MEDIA / "20261029-landscapes_2d.png",
             "The 2-D case: every candidate layout in the plane of the thumb-to-pair distance $x_{sep}$ and the "
             "index-to-middle distance $y_{sep}$, coloured by one metric per panel. Red is better, as in Fig. 5 of "
             "[1]. Top row: the pinch; bottom row: the tripod. Rings mark the Pareto-optimal candidates on all six "
             "scores. <code>scripts/fp_review_figures.py landscape</code>.")
    t += fill(r"""<p><b>The 6-D case.</b> From the CAD-nominal layout the RRT reached <<n6>> candidates in <<c6>> calls and
<<s6>> s, <<pc6>> s per call. Over the whole run <<rr6>> % of calls yielded a candidate, and <<e6>> % of the last 400. The
paper's 6-D run ended at 45.52 %. Rotation ranges above zero, the paper's success criterion, held for <<p6>> % of
candidates on the pinch and <<t6>> % on the tripod.</p>""",
              n6=f"{r6.get('candidates', 0):,}", c6=f"{r6.get('calls', 0):,}", s6=f"{r6.get('seconds', 0):.0f}",
              pc6=f"{r6.get('seconds', 0) / max(r6.get('calls', 1), 1):.2f}",
              rr6=f"{100 * r6.get('candidates', 0) / max(r6.get('calls', 1), 1):.0f}",
              e6=f"{100 * r6.get('efficiency_last', 0):.0f}",
              p6=f"{100 * N['tmr_pos6']['pinch']:.1f}", t6=f"{100 * N['tmr_pos6']['tripod']:.1f}")
    t += N["sobol_text"]
    t += img(MEDIA / "20261029-sampling_efficiency.png",
             "Left: candidates per RRT call over the last 400 calls. The 2-D run fills its region and stops at 1 %; the "
             "6-D run keeps finding candidates. Right: the share of candidates whose rotation range exceeds a threshold "
             "(the paper's Fig. 4, top). The pinch's range is small on every layout; the tripod's spreads to 60 deg. "
             "<code>scripts/fp_review_figures.py efficiency</code>.")
    t += N["embed_text"]
    t += img(MEDIA / "20261029-tsne_pca_6d.png",
             "The 6-D candidates in two dimensions, coloured by mean score. (a) t-SNE of all RRT candidates; (b) "
             "t-SNE of as many RRT candidates, drawn at random, as there are Sobol candidates; (c) t-SNE of the Sobol "
             "candidates, scored on (a)'s scale; (d) PCA of (a). Bottom: the best layout of six "
             "regions of (a) (k-means on the map), top view of the mounts: thumb blue, index red, middle green; each "
             "box spans the gantries' travel. <code>scripts/fp_review_figures.py embed</code>.")
    return t


def sec_check(N):
    lay = check_stats(N["check"])
    if not lay:
        return "<p>CHECK_PENDING</p>"
    tmr = [x["tmr"] for x in lay]
    ramp = [x["ramp"] for x in lay]
    t = N["check_text"]
    body = [[f"{x['x']:.1f}", f"{x['y']:.1f}", f"{x['tmr']:.0f}", f"{x['mft']:.0f}", f"{x['smin']:.1f}",
             f"{x['held']}/{x['n']}", f"{x['ramp']:.0f}", "" if x["rmse"] is None else f"{x['rmse']:.2f}"]
            for x in sorted(lay, key=lambda x: x["tmr"])]
    t += table(["$x_{sep}$<br>(mm)", "$y_{sep}$<br>(mm)", "Range<br>(deg)", "Torque<br>(mN m)", "Pad s.v.<br>(mm/rad)",
                "Held", "Ramp, 3 pads<br>(deg)", "RMSE<br>/step"], body,
               "The tripod's kinematic metrics (rotation range, joint torque, smallest pad singular value) against the "
               "HOM twist of a 40 mm sphere on the same layouts: held runs of three seeds, and the medians of the ramp "
               "on three pads and of the velocity RMSE per step. Rows in <code>rows/hom_twist_check.jsonl</code>.", num_from=0)
    return t


def sec_separate(N):
    t = r"""<p>Choosing a hand with its controller in the loop is a nested problem. For each design $d$, find the best
controller $\pi$; then pick the design whose best controller scores highest:</p>
$$d^\ast = \arg\max_d\ J^\ast(d), \qquad J^\ast(d) = \max_\pi J(d, \pi).$$
<p>The inner maximum is what makes the problem expensive. When $\pi$ is a policy trained by RL for each design, the
inner maximum is itself a noisy search. In the July sweep the spread between training seeds on one design (cos sd
0.38) was larger than the differences between designs. Two ways around the inner maximum are available, and they
differ in what they assume.</p>
<p><b>Replace the controller by a pose and a planner.</b> The FP approach keeps a design if it reaches the task's pose
and lets a kinematic planner move it. This assumes the task's mechanism is known: which contacts, which sub-assembly,
which object motion. With that assumption the grasp is no longer a generic grasp-quality optimum. It is the starting
state of the task's mechanism, so its quality is measured by the motion that follows. Our own studies show what happens
without that assumption. CEM's hold score cut rv04_mid's turn from cos 0.997 to 0.625, and the scale study's hold and
turn optima differ by 6 mm of layout per object.</p>
<p><b>Fix the controller.</b> A design-agnostic controller with few parameters, such as HOM, turns the inner maximum
into one evaluation, $J(d) = J(d, \pi_{\mathrm{HOM}})$. HOM's commands are written at the contacts, so the same
command applies to every layout. The design enters only through the finger Jacobians, the joint limits and the
starting pose. That starting pose is a foundational pose. HOM is deterministic given the plant, so a design's score
does not depend on a training seed. Seeds only move the object's start by up to 1 mm and 2 deg.</p>
<p>So the answer to whether there must be a separation is no, provided the controller is fixed. The problem becomes
single-level: sample a design, solve its foundational pose, run the controller, score the run. What remains of the
coupling is the controller's few parameters: grasp forces, rates, the pusher's lever. The pin-push, for instance,
tracked best at 0.75 N of drive. A small grid over these per design, three to five values each, is a light inner
search and keeps the design scores comparable.</p>
<p>A cascade remains useful for cost alone: kinematic poses in milliseconds, dynamic rollouts in seconds, hardware in
hours. A cascade is safe when each early test is a necessary condition for the later ones. Reaching the foundational
pose is necessary by construction, because the mechanism must pass through it. A kinematic score is not necessary
for a good dynamic score; it can only rank. On the twist, the rotation range did not rank (Spearman <<s1>>, Section 6).
The dynamic rollout therefore has to stay in the loop, and at about 2 s per run it can.</p>"""
    return fill(t, s1=f"{N['sp_tmr']:.2f}")


def sec_plan(N):
    r6 = N["rrt6d"]
    pc = r6.get("seconds", 582.0) / max(r6.get("calls", 1), 1)
    rows = [
        ["0. Poses", "Write the HOM tasks as foundational poses: pin-push and press on 16 and 25 mm cylinders, roll and "
         "twist on a 40 mm sphere, the screwdriver's pinch brake and wield", "pose definitions in "
         "<code>fp_sr2_toy.py</code>", "minutes", "a pose no layout reaches"],
        ["1. Sample", "Sobol over the six gantry coordinates (the finger variants of the 10-28 redesign as a seventh, "
         "discrete coordinate); SLSQP or SNOPT onto every pose", f"10,000 layouts at {pc:.2f} s each for two poses",
         f"about {10000 * pc / 3600:.1f} h on one core", "fewer than 10 % reach all poses"],
        ["2. Kinematic scores", "rotation range, joint torque against the SCS0009's 0.18 N m stall, pad conditioning; "
         "HOM's Eq. (2) along each task's reference twist without dynamics, which gives the paper's sliding speed",
         "the rows of stage 1", "1-2 h on one core", ""],
        ["3. HOM rollouts", "the four tasks in MuJoCo on the servo plant with sphere pads, three seeds, two drive "
         "forces", "the 2,500 Sobol layouts that reach every pose", "about 2 s per rollout: "
         "2,500 x 4 x 3 x 2 = 60,000 runs, 33 CPU h, 3 h on 12 cores", "scores that do not repeat across seeds: rank "
         "correlation between seed halves below 0.8"],
        ["4. Sensing", "rerun the dynamic Pareto set with the measured fingertip errors and calibration errors "
         "(variants <code>tip</code>, <code>cal_tipfl</code> of the HOM review)", "about 500 layouts",
         "6,000 runs, 3.3 CPU h", "the Pareto set changes when sensing is added"],
        ["5. Map", "Pareto front over the dynamic scores; t-SNE and PCA of the Sobol-sampled designs, beside a "
         "uniform null", "the scores of stages 3-4", "minutes", "clusters that the uniform null also shows"],
        ["6. Bench", "three to five layouts from different clusters and the front, set on the gantries; the pin-push "
         "and twist with tags and, once mounted, fingertips", "the bench hand", "one or two days", "bench ranking "
         "against simulated ranking, Spearman below 0.5"],
    ]
    t = r"""<p>The paper's dense front and t-SNE map can be rebuilt for the SR2 hand at moderate cost, because the
expensive part, the controller, is fixed and runs in seconds. Table 3 lists the stages. Each has its input, its cost on
this workstation and the result that would stop the plan.</p>"""
    t += table(["Stage", "Work", "Input", "Cost", "Stops the plan"], rows,
               "A plan for a Pareto front and t-SNE map of SR2 layouts under HOM control. Costs from this page's timings "
               "(fp_sr2_toy.py, fp_hom_check.py) and the HOM review's sensing grid.", num_from=9)
    t += r"""<p>Three choices differ from the paper and follow from the toy. Sobol replaces the RRT, since on our
gantry box most layouts reach the poses and the RRT's advantage disappears (Section 5). A Sobol sample is also the
unbiased population that a t-SNE map needs; an RRT's branches print their own structure. The scores come from dynamic rollouts of the controller that will run on the bench. The
kinematic stage only gates reachability, since its rotation range did not rank the twist (Section 6). Finally,
sensing enters as a robustness score on the front, the variants that the HOM review page defines.</p>"""
    return t


def sec_limits():
    return """<ul>
<li>The toy's two poses are grasps of two HOM tasks. The screwdriver's load and wield, the tasks of both papers, are
not posed yet.</li>
<li>Its metrics are kinematic: rotation with fixed contacts, static joint torque, conditioning. Friction cones, servo dynamics and the pads' compliance enter only in Section 6's dynamic
check.</li>
<li>The layout space is the gantries' six coordinates. The paper's template also varies link lengths and the thumb's
rotation, which the SR2 hand cannot change without new parts.</li>
<li>The dynamic check covers one task on symmetric tripods only.</li>
</ul>"""


def sec_future():
    return """<ol>
<li>Pose the screwdriver's pinch brake and wield (the tasks of both papers) as foundational poses in
<code>fp_sr2_toy.py</code>, and check that the brake's angle law holds on the toy's pinch candidates.</li>
<li>Run the dynamic check of Section 6 on the pin-push over the 6-D candidates. The plan's premise fails if the
Spearman correlation between kinematic and dynamic scores falls below 0.5.</li>
<li>Stage 1 of the plan with the 10-28 finger variants as a seventh coordinate, once their CAD models exist in the
kinematics module.</li>
<li>Compare the t-SNE maps of Sobol and RRT candidates against a uniform null with the neighbour-preservation test of
<code>scripts/paper_morphology_diversity.py</code>.</li>
</ol>"""


def sec_files():
    refs = "".join(f"<li>{r}</li>" for r in REFS)
    return f"""<p>References:</p><ul>{refs}</ul>
<p>Files:</p><ul>
<li>This page: <code>{OUT.relative_to(ROOT)}</code>, built by <code>scripts/fp_review_page.py</code> with
<code>scripts/fp_review_page.template.html</code>; figures by <code>scripts/fp_review_figures.py</code>.</li>
<li>Toy: <code>scripts/fp_sr2_toy.py</code> (<code>run2d</code>, <code>run6d</code>, <code>sobol</code>,
<code>embed</code>); dynamic check: <code>scripts/fp_hom_check.py</code>; rows in
<code>{ROWS.relative_to(ROOT)}/</code>.</li>
<li>Earlier studies: <code>docs/experiments/20260831-real_v1-sobol8192/</code>,
<code>docs/experiments/20260828-real_v1_search/</code>, <code>docs/experiments/20261008-hand_object_scale/</code>,
<code>docs/experiments/20261001-drake-port/</code>; the morphology overview <code>docs/overviews/morphology_design.html</code>.</li>
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
    page = docs_index.tag(page, topic="design", summary=summary, date="2026-10-29")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {n[0]} formulas", flush=True)


def main():
    N = numbers()
    r2 = N["rrt2d"]
    d2 = r2["d"]
    xs = d2[:, 2] - d2[:, 0]
    N["d2_x"] = (float(xs.min()), float(xs.max()))
    N["tmr_pos6"] = {fp: float((N["rrt6d"]["raw"][:, 3 * k] > 0).mean()) for k, fp in enumerate(FF.FPS)}
    # the 2-D landscapes: where each metric's best tenth of candidates sits, and the extremes
    raw2 = r2["raw"]
    ys2 = d2[:, 3] - d2[:, 5]

    def top(k, sign, frac=0.1):
        v = raw2[:, k] * sign
        idx = np.argsort(-v)[:max(1, int(round(frac * len(v))))]
        return float(np.median(xs[idx])), float(np.median(ys2[idx]))
    pin_tmr, pin_sm, tri_tmr, tri_mft = top(0, 1), top(2, 1), top(3, 1), top(4, -1)
    N["land_text"] = fill(r"""<p>The two poses pull the layout in different directions. The pinch's rotation range is <<p0>> to <<p1>>
deg. Its best tenth of layouts sits at a median $x_{sep}$ of <<pt>> mm, and its pads are best conditioned at <<ps>> mm.
The pinch's joint torque barely changes, <<m0>> to <<m1>> mN m. The tripod's range reaches <<t1>> deg, best at a wide
thumb and a close pair (median $x_{sep}$ <<tx>> mm, $y_{sep}$ <<ty>> mm). Its torque falls from <<q1>> to <<q0>> mN m as
the thumb moves away (best tenth at <<qx>> mm). With six scores most layouts beat every other on at least one, so
<<np>> of the <<nc>> candidates are Pareto-optimal. The front alone selects little; the landscapes show the
gradients.</p>""",
        p0=f"{raw2[:, 0].min():.0f}", p1=f"{raw2[:, 0].max():.0f}", pt=f"{pin_tmr[0]:.0f}", ps=f"{pin_sm[0]:.0f}",
        m0=f"{raw2[:, 1].min():.0f}", m1=f"{raw2[:, 1].max():.0f}", t1=f"{raw2[:, 3].max():.0f}",
        tx=f"{tri_tmr[0]:.0f}", ty=f"{tri_tmr[1]:.0f}", q0=f"{raw2[:, 4].min():.0f}", q1=f"{raw2[:, 4].max():.0f}",
        qx=f"{tri_mft[0]:.0f}", np=r2["pareto"], nc=r2["cand"])
    sb, r6 = N["sobol6d"], N["rrt6d"]
    N["sobol_text"] = fill(r"""<p>The same solver on <<sn>> Sobol layouts, each started from the grid search alone, reached both
poses on <<sc>> (<<sp>> %) in <<ss>> s. The pinch failed on <<sf1>> layouts and the tripod on <<sf2>>. The RRT's warm
starts raised the yield to <<rp>> %, a factor of <<fx>>. The paper's argument for the RRT holds where the feasible
set is a thin part of the space. On our gantry box most layouts reach both poses, so the advantage is small.</p>""",
        sn=f"{sb.get('n', 0):,}", sc=sb.get("cand", 0), sp=f"{100 * sb.get('cand', 0) / max(sb.get('n', 1), 1):.0f}",
        ss=f"{sb.get('seconds', 0):.0f}", sf1=sb["fail_fp"]["pinch"], sf2=sb["fail_fp"]["tripod"],
        rp=f"{100 * r6['candidates'] / r6['calls']:.0f}",
        fx=f"{(r6['candidates'] / r6['calls']) / (sb.get('cand', 0) / max(sb.get('n', 1), 1)):.2f}")
    emb = json.loads((ROWS / "embed6d.json").read_text())
    pv = emb["pca_var"]
    picks = N["fig"].get("embed", {}).get("picks", [])
    N["embed_text"] = fill(r"""<p>Figure 3 maps the 6-D candidates to the plane. The t-SNE of the RRT candidates falls into islands
of filaments, like the paper's Fig. 6. The same embedding of <<ns>> RRT candidates drawn at random still clumps.
<<ns>> Sobol candidates spread evenly, with no islands. PCA of the RRT candidates is close to isotropic: <<pv>> % of the
variance over the six components, so the candidates fill the box. The islands therefore trace the tree's branches,
each grown from its neighbour in steps of 0.1, and are not separate regions of good layouts. The paper's 6-D
candidates all came from an RRT, so the same test applies to its Fig. 6. The best layouts of six regions, drawn under
the maps, have mean scores of <<m0>> to <<m1>> and differ visibly.</p>""",
        ns=emb.get("sobol_n", 0), pv=", ".join(f"{100 * x:.0f}" for x in pv),
        m0=f"{min(x['mean'] for x in picks):.0f}" if picks else "?",
        m1=f"{max(x['mean'] for x in picks):.0f}" if picks else "?")
    lay = check_stats(N["check"])
    allchk = load("hom_twist_check")
    nog = sum(1 for r in allchk if r.get("status") == "no_grasp")
    dropped = [x for x in lay if x["held"] == 0]
    held = [x for x in lay if x["rmse"] is not None]
    N["sp_tmr"] = spearman([x["tmr"] for x in lay], [x["ramp"] for x in lay])
    N["sp_smin_rmse"] = spearman([x["smin"] for x in held], [x["rmse"] for x in held])
    N["sp_smin_ramp"] = spearman([x["smin"] for x in lay], [x["ramp"] for x in lay])
    N["check_text"] = fill(r"""<p>The toy's scores are kinematic, and the HOM tasks are dynamic. Table 2 runs the HOM twist of a
40 mm sphere on <<nl>> symmetric tripods from the 2-D case, chosen evenly by rank of the tripod's kinematic rotation
range (<code>scripts/fp_hom_check.py</code>). The twist is the task study's: steps of 40 deg/s, then a 30 deg/s ramp,
three seeds each. Each layout gets the task study's own equator grasp. Under that grasp rule <<ng>> of the <<nl>>
layouts have no grasp, though the SQP found a tripod pose on each. The rule fixes the sphere under the mounts'
circumcentre and the contacts facing the mounts, with a 3 deg joint margin; the SQP may move both.</p>
<p>On the <<nr>> layouts that ran, the kinematic rotation range does not rank the ramp held on three pads (Spearman
<<s1>>). <<nd>> layouts dropped the sphere in every seed. Their $x_{sep}$ is <<dx0>> to <<dx1>> mm and their $y_{sep}$
<<dy0>> to <<dy1>> mm. Their kinematic ranges were <<dk0>> to <<dk1>> deg. The pads' smallest singular value ranks the
tracking error of the <<nh>> layouts that held (Spearman <<s2>>), and the ramp weakly (<<s3>>). The rotation range
counts how far the fingers can follow the object. It omits whether friction and the servo forces keep the object in
the grasp, and for the twist that decides the outcome.</p>""",
        nl=len({(r["x_sep_mm"], r["y_sep_mm"]) for r in allchk}), ng=nog, nr=len(lay), s1=f"{N['sp_tmr']:.2f}",
        nd={1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}.get(len(dropped), str(len(dropped))), dx0=f"{min(x['x'] for x in dropped):.0f}" if dropped else "?",
        dx1=f"{max(x['x'] for x in dropped):.0f}" if dropped else "?",
        dy0=f"{min(x['y'] for x in dropped):.0f}" if dropped else "?",
        dy1=f"{max(x['y'] for x in dropped):.0f}" if dropped else "?",
        dk0=f"{min(x['tmr'] for x in dropped):.0f}" if dropped else "?",
        dk1=f"{max(x['tmr'] for x in dropped):.0f}" if dropped else "?",
        nh=len(held), s2=f"{N['sp_smin_rmse']:.2f}", s3=f"{N['sp_smin_ramp']:.2f}")
    sections = [
        ("terms", "Terms and metrics", sec_terms()),
        ("paper", "The foundational-pose paper: choosing hands by a pose", sec_paper()),
        ("two", "The foundational pose and the HOM controller as one method", sec_two_papers()),
        ("ours", "How the project samples SR2 layouts", sec_ours(N)),
        ("toy", "The paper's pipeline on the SR2 hand: RRT, SQP, Pareto front and t-SNE", sec_toy(N)),
        ("check", "Kinematic scores against HOM rollouts on the same layouts", sec_check(N)),
        ("separate", "Grasp and in-hand control: one problem or two", sec_separate(N)),
        ("plan", "A plan for a Pareto front of SR2 layouts under HOM control", sec_plan(N)),
        ("limits", "Limitations", sec_limits()),
        ("future", "Future work", sec_future()),
        ("files", "References and files", sec_files()),
    ]
    byline = ("Review and simulation, 2026-10-29. Paper: arXiv 2409.14158v2 [1]. Hand: the CAD-model SR2 hand "
              "(closed-form kinematics of <code>scripts/hand_object_scale_kin.py</code>; servo plant and 1 mm sphere "
              "pads in the dynamic check). Builder: <code>scripts/fp_review_page.py</code>.")
    rate6 = N["rrt6d"]["candidates"] / N["rrt6d"]["calls"]
    rates = N["sobol6d"]["cand"] / max(N["sobol6d"]["n"], 1)
    sec6 = N["rrt6d"]["seconds"] / N["rrt6d"]["calls"]
    lede = fill("""The foundational-pose paper [1] keeps a hand design if it can reach a few poses that tools are wielded
from. It then scores how far a kinematic planner moves the tool from each pose; 10,785 of 100,480 sampled hands reached
its three poses. The project samples SR2 layouts by a Sobol draw and screens grasps in dynamic simulation. With two HOM-task
grasps as the poses, <<n6>> SR2 layouts were found by the paper's RRT and SQP at <<sc>> s each. Of the calls, <<r6>> %
yielded a layout, against <<rs>> % for Sobol. In a t-SNE map these layouts form islands that follow the tree's
branches; Sobol candidates form none.
On <<nr>> layouts the kinematic rotation range did not rank the HOM twist (Spearman <<sp>>), so the controller has to
stay in the loop. With HOM fixed, a Pareto front over dynamic rollouts of 2,500 layouts costs about 33 CPU-hours.""",
                n6=f"{N['rrt6d']['candidates']:,}", sc=f"{sec6:.2f}", r6=f"{100 * rate6:.0f}", rs=f"{100 * rates:.0f}",
                nr=len(check_stats(N["check"])), sp=f"{N['sp_tmr']:.2f}")
    summary = fill("""On the SR2 hand the foundational-pose pipeline (RRT, SQP, Pareto front, t-SNE) found <<n6>> layouts at
<<sc>> s each, but its t-SNE islands are the RRT's branches (Sobol candidates show none), and the kinematic rotation
range did not rank the HOM twist on <<nr>> layouts (Spearman <<sp>>). With HOM fixed, a Pareto front over dynamic rollouts
of 2,500 layouts costs about 33 CPU-hours.""", n6=f"{N['rrt6d']['candidates']:,}", sc=f"{sec6:.2f}",
                   nr=len(check_stats(N["check"])), sp=f"{N['sp_tmr']:.2f}")
    assemble(sections, "Foundational-pose hand design",
             "Foundational-pose hand design, the project's layout sampling, and a Pareto front for the SR2 hand",
             "Morphology and design search", lede, byline, summary)


if __name__ == "__main__":
    main()
