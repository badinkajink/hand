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
    '[5] N. Bohlinger and J. Peters, "Shape Your Body: Value Gradients for Multi-Embodiment Robot Design", CoRL 2026, '
    '<a href="https://arxiv.org/abs/2606.00702">arXiv 2606.00702</a>.',
    '[6] B. Yang, L. Jiang, W. Wu, M. Cheng and K. Gong, "Evolving Robot Morphology with Manipulation Learning Inspired '
    'by Human Hand Evolution", IEEE Transactions on Evolutionary Computation, 2026, DOI 10.1109/TEVC.2026.3696420.',
    '[7] B. Yang, L. Jiang, W. Wu and R. Zhen, "Evolving Robotic Hand Morphology Through Grasping and Learning", IEEE '
    'Robotics and Automation Letters, 2024, DOI 10.1109/LRA.2024.3440707.',
    '[8] Z. Xie, Y. Xiang, M. Posa and W. Jin, "Where to Touch, How to Contact: A Hierarchical RL-MPC Framework for '
    'Geometry-Aware Sim-to-Real Manipulation", <a href="https://arxiv.org/abs/2601.10930">arXiv 2601.10930</a>, 2026.',
]
GRID_TASKS = (("twist", 40.0, "Twist", "40 mm sphere"), ("roll", 40.0, "Roll", "40 mm sphere"),
              ("pinpush", 25.0, "Pin-push", "25 mm cylinder"), ("press", 25.0, "Press", "25 mm cylinder"))
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


def img(path, caption, jpeg=False):
    """A figure; jpeg=True embeds a render as a JPEG of quality 85 (a third to a fifth of the PNG's size)."""
    FIG[0] += 1
    if jpeg:
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.open(path).convert("RGB").save(buf, "JPEG", quality=85)
        uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    else:
        uri = data_uri(path, "image/png")
    return f'<figure><img src="{uri}" alt=""><figcaption>Figure {FIG[0]}. {caption}</figcaption></figure>'


FILM = [0]


def film(path, caption):
    if not Path(path).exists():
        return ""
    FILM[0] += 1
    poster = Path(path).with_suffix(".jpg")
    pa = f' poster="{data_uri(poster, "image/jpeg")}"' if poster.exists() else ""
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
        ("HOM tasks", "the four tasks of the HOM task study. In the pin-push, thumb and index pinch a 25 mm cylinder and "
         "the middle finger turns it about the pinch axis by friction, 15 deg each way at 15 deg/s. The press uses the "
         "same pinch; the middle finger, below the cylinder's equator, lifts its side by 20 deg and lowers it. The roll "
         "rolls a 40 mm sphere along the thumb's pad. The twist turns the sphere about the vertical, then ramps it at "
         "30 deg/s."),
        ("Press pose", "this page's third pose: the pinch pose with the middle finger 30 to 45 deg below the cylinder's "
         "equator, solved for each layout from its pinch pose."),
        ("Held", "the object ends no more than 15 mm below its start, with at least two fingers bearing load. A layout "
         "counts as held when every seed holds."),
        ("Ramp on three pads", "the largest twist (deg) reached at 30 deg/s while all three pads still touch the sphere."),
        ("Velocity RMSE per step", "the root-mean-square error of the sphere's angular velocity over the commanded "
         "steps, divided by the 40 deg/s step."),
        ("Pinch-angle RMSE", "the root-mean-square difference (deg) between the cylinder's angle about the pinch axis "
         "and the command."),
        ("Sensing variants", "the simulated sensing of the HOM review at the bench's 50 Hz. The four used here are exact "
         "sensing and three with calibration errors of σ 1 deg per joint and σ 1 mm in the object's pose, constant within "
         "a run. Of those three, one has no fingertips, one has the fingertips' measured errors, and one has the "
         "fingertips closing a force loop on the pad force."),
        ("Force error", "the mean over the three pads of the difference between a run's mean pad force and the same "
         "seed's under exact sensing (N)."),
        ("v4b fingertip", "the printed fingertip calibrated on the UR5. Its pad is half an ellipsoid (semi-axes 8.5 mm "
         "deep, 7.4 mm across, 11 mm toward the tip and 22 mm toward the PIP) in a cassette under a trimmed cap. "
         "Sections SEC[v4b] and SEC[grid] use it; the earlier sections use the TPU block."),
        ("Task score", "a rollout's score in [0, 1], 0 when the object is dropped. Twist: the ramp on three pads "
         "divided by 90 deg. Roll: 1/(1 + velocity RMSE per step). Pin-push and press: 1 minus the pinch-angle RMSE "
         "over the stroke (15 and 20 deg), at least 0. A grasp's score is the mean over its seeds."),
        ("Grasp set", "up to 16 grasps of one layout and object: the foundational pose and up to 15 others spread over "
         "its feasible set (Section SEC[grid])."),
        ("Best of 16", "a layout's highest grasp score in its grasp set."),
        ("Held share", "the share of a layout's grasps held in every seed."),
        ("Feasible log volume", r"$\sum_j \log(w_j / r_j)$ over a pose's variables. Here $w_j$ is the feasible width "
         r"of variable $j$ on a line through the foundational pose and $r_j$ its bound range. It measures the size of the "
         r"layout's feasible grasp set and is 0 when every variable is free over its bounds."),
        ("Grasp-matrix singular value", "the smallest singular value of the grasp matrix at the three contacts, the "
         "grasp metric of [6], recorded for every grasp of a set."),
        ("Smoothed map, basin", "the best-of-16 map smoothed by a Gaussian of 7.5 mm. A basin is the set of grid cells "
         "whose steepest ascent on the smoothed map ends at the same peak. It is kept when it has 8 cells or more and its "
         "peak rises 0.05 above its lowest cell."),
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
controller can be the evaluation that follows it. Section SEC[separate] asks whether the two halves must be scored together or can be scored one after the other.</p>"""
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
<code>20261001-drake_sr2_planning</code>). Section SEC[toy] runs the whole pipeline on the SR2 hand.</p>"""
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


def rank(v):
    return np.argsort(np.argsort(np.asarray(v, float)))


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(rank(a[ok]), rank(b[ok]))[0, 1]) if ok.sum() > 2 else float("nan")


def gpu_validation():
    """CPU (fp_hom_check.py, rows/hom2d_shard*) against GPU (hom_gpu.py, rows/hom2d_gpu_*) on the 2-D layouts."""
    cpu = {}
    for q in sorted(ROWS.glob("hom2d_shard*.jsonl")):
        for r in FF.load(q.stem):
            if r.get("status") == "ok":
                cpu[(r["x_sep_mm"], r["y_sep_mm"], r["task"], r["seed"])] = r
    out = {}
    for task, key in (("twist", "ramp_three_pads_max_deg"), ("roll", "w_rmse_ratio")):
        gpu = load(f"hom2d_gpu_{task}")
        pairs = [(cpu.get((g["x_sep_mm"], g["y_sep_mm"], task, g["seed"])), g) for g in gpu]
        pairs = [(c, g) for c, g in pairs if c is not None]
        if not pairs:
            continue
        agree = float(np.mean([bool(c["held"]) == bool(g["held"]) for c, g in pairs]))
        both = [(c, g) for c, g in pairs if c["held"] and g["held"]]
        a = [c.get(key) or 0.0 for c, g in both]
        b = [g.get(key) or 0.0 for c, g in both]
        lay = {}
        for c, g in both:
            lay.setdefault((g["x_sep_mm"], g["y_sep_mm"]), []).append((c.get(key) or 0.0, g.get(key) or 0.0))
        la = [np.mean([x for x, _ in v]) for v in lay.values()]
        lb = [np.mean([y for _, y in v]) for v in lay.values()]
        tim = load(f"hom2d_gpu_{task}_timing")
        out[task] = {"n": len(pairs), "held_cpu": sum(bool(c["held"]) for c, _ in pairs),
                     "held_gpu": sum(bool(g["held"]) for _, g in pairs), "agree": agree, "key": key,
                     "sp_roll": sp(a, b), "sp_lay": sp(la, lb), "mad": float(np.median(np.abs(np.subtract(a, b)))),
                     "ms": (1000 * sum(t["run_s"] for t in tim) / sum(t["worlds"] for t in tim)) if tim else None,
                     "build_s": sum(t["build_s"] for t in tim) if tim else None,
                     "run_s": sum(t["run_s"] for t in tim) if tim else None,
                     "worlds": sum(t["worlds"] for t in tim) if tim else None}
    cpu_wall = [r["wall_s"] for r in cpu.values() if r.get("wall_s")]
    out["cpu_ms"] = 1000 * float(np.median(cpu_wall)) if cpu_wall else None
    return out


def dyn_set(name_tw, name_ro):
    """Per layout: kinematic tripod metrics, the twist's ramp and the roll's error (medians over seeds), held in every
    seed of both tasks."""
    tw = {x["key"]: x for x in FF.load_hom(name_tw)}
    ro = {x["key"]: x for x in FF.load_hom(name_ro)}
    keys = sorted(set(tw) & set(ro))
    if not keys:
        return None
    kin = np.array([tw[k]["kin"]["tmr_deg"] for k in keys])
    smin = np.array([tw[k]["kin"]["smin_mm"] for k in keys])
    ramp = np.array([tw[k]["ramp"] for k in keys])
    rmse = np.array([ro[k]["rmse"] for k in keys])
    ok = np.array([tw[k]["held"] >= 1.0 and ro[k]["held"] >= 1.0 for k in keys])
    rows_tw = load(name_tw)
    return {"n": len(keys), "runs": len(rows_tw) + len(load(name_ro)), "ok": int(ok.sum()),
            "held_tw": float(np.mean([tw[k]["held"] >= 1.0 for k in keys])),
            "held_ro": float(np.mean([ro[k]["held"] >= 1.0 for k in keys])),
            "ramp_all": float(np.median([tw[k]["ramp"] for k in keys])),
            "rmse_tw": float(np.nanmedian([tw[k]["rmse"] for k in keys])),
            "ramp_med": float(np.median(ramp[ok])) if ok.any() else float("nan"),
            "rmse_med": float(np.nanmedian(rmse[ok])) if ok.any() else float("nan"),
            "sp_kin_ramp": sp(kin, ramp), "sp_smin_rmse": sp(smin[ok], rmse[ok]),
            "drop_tw": int(sum(tw[k]["held"] < 1.0 for k in keys))}


HR_ROWS = ROOT / "docs/experiments/20261029-hom_control_review/rows"


def pinch_validation():
    """The pin-push and the press on the GPU (hom_gpu.py) against the CPU runner (hom_tasks.py) on the task study's
    grasps, ten seeds per size (the HOM review's rows cpu_val_base and gpu_val_base): per task, runs, the share with the
    same held outcome, and the largest difference between the two runners' per-size median pinch-angle RMSE."""
    def rows(name):
        p = HR_ROWS / f"{name}.jsonl"
        return {(r["task"], r["d_mm"], r["seed"]): r for r in (json.loads(x) for x in p.read_text().splitlines()
                                                                 if x.strip()) if r.get("status") == "ok"} if p.exists() else {}
    c, g = rows("cpu_val_base"), rows("gpu_val_base")
    out = {}
    for t in ("pinpush", "press"):
        ks = [k for k in c if k[0] == t and k in g]
        if not ks:
            continue
        dmed = []
        for d in sorted({k[1] for k in ks}):
            mc = [c[k]["theta_rmse_deg"] for k in ks if k[1] == d and c[k].get("held")]
            mg = [g[k]["theta_rmse_deg"] for k in ks if k[1] == d and g[k].get("held")]
            if mc and mg:
                dmed.append(abs(st.median(mc) - st.median(mg)))
        out[t] = {"n": len(ks), "same": float(np.mean([bool(c[k]["held"]) == bool(g[k]["held"]) for k in ks])),
                  "dmed": max(dmed) if dmed else float("nan")}
    return out


def gpu_ms(name):
    tim = [t_ for t_ in load(f"{name}_timing") if t_["worlds"] >= 1000]
    return 1000 * sum(t_["run_s"] for t_ in tim) / sum(t_["worlds"] for t_ in tim) if tim else None


def seed_check():
    """The pin-push on 40 layouts of the 2-D case from the pinch pose (rows/cpu_check_2dfp_pinpush, CPU runner, seeds 0
    and 1) against the GPU rows (hom2dfp_gpu_pinpush): the share of layouts with the same held outcome, between engines
    and between seeds of one engine."""
    cpu = {(tuple(r["d"]), r["seed"]): r for r in load("cpu_check_2dfp_pinpush")}
    gpu = {(tuple(r["d"]), r["seed"]): r for r in load("hom2dfp_gpu_pinpush")}
    lays = sorted({k[0] for k in cpu})
    if not lays:
        return None

    def same(A, sa, B, sb):
        pr = [(A[(d, sa)], B[(d, sb)]) for d in lays if (d, sa) in A and (d, sb) in B]
        return float(np.mean([bool(a["held"]) == bool(b["held"]) for a, b in pr])) if pr else float("nan")
    return {"n": len(lays), "cg0": same(cpu, 0, gpu, 0), "cg1": same(cpu, 1, gpu, 1), "cc": same(cpu, 0, cpu, 1),
            "gg": same(gpu, 0, gpu, 1)}


def sphere_check():
    """The twist and the roll from the foundational poses of 40 Sobol layouts, seed 0: the CPU runner
    (rows/cpu_check_sobol_*) against the GPU rows (hom6d_gpu_*)."""
    out = {}
    for task, m in (("twist", "ramp_three_pads_max_deg"), ("roll", "w_rmse_ratio")):
        cpu = load(f"cpu_check_sobol_{task}")
        gpu = {(tuple(r["d"]), r["seed"]): r for r in load(f"hom6d_gpu_{task}")}
        pr = [(c, gpu[(tuple(c["d"]), c["seed"])]) for c in cpu if (tuple(c["d"]), c["seed"]) in gpu]
        if not pr:
            continue
        both = [(c.get(m) or 0.0, g.get(m) or 0.0) for c, g in pr if c["held"] and g["held"]]
        out[task] = {"n": len(pr), "same": float(np.mean([bool(c["held"]) == bool(g["held"]) for c, g in pr])),
                     "sp": sp([x for x, _ in both], [y for _, y in both])}
    return out


def pinch_text():
    v = pinch_validation()
    if not v:
        return ""
    mp, mq = gpu_ms("hom6d_gpu_pinpush"), gpu_ms("hom6d_gpu_press")
    t = fill(r""" A pin-push rollout takes <<mp>> ms and a press <<mq>> ms. On the task study's eight cylinder grasps, ten
seeds each, the GPU and CPU runners agree on the held outcome in <<sp>> % of pin-push runs and <<sq>> % of press runs.
Their median pinch-angle RMSEs per grasp differ by at most <<dm>> deg (HOM review page, Table 3).""",
             mp=f"{mp:.0f}" if mp else "?", mq=f"{mq:.0f}" if mq else "?", sp=f"{100 * v['pinpush']['same']:.0f}",
             sq=f"{100 * v['press']['same']:.0f}", dm=f"{max(v['pinpush']['dmed'], v['press']['dmed']):.2f}")
    c = seed_check()
    if c:
        t += fill(r"""</p>
<p>On the toy's poses the pin-push is less repeatable than on the task study's grasps. On <<n>> layouts of the 2-D case,
the CPU runner and the GPU agree on the held outcome in <<a>> % of layouts with seed 0 and <<b>> % with seed 1. Two
seeds on the same runner agree in <<c>> % (CPU) and <<d>> % (GPU). The start's jitter of up to 1 mm and 2 deg changes
whether the friction drive holds. The two runners therefore differ by no more than two seeds of one runner
do.""", n=c["n"], a=f"{100 * c['cg0']:.0f}", b=f"{100 * c['cg1']:.0f}", c=f"{100 * c['cc']:.0f}", d=f"{100 * c['gg']:.0f}")
    q = sphere_check()
    if q.get("twist") and q.get("roll"):
        t += fill(r""" From the foundational poses of <<n>> Sobol layouts (seed 0), the runners agree on the held outcome in
<<at>> % of twists and <<ar>> % of rolls. Over the layouts both held, the twist's ramp ranks at Spearman <<st>> and the
roll's error at <<sr>>.""", n=q["twist"]["n"], at=f"{100 * q['twist']['same']:.0f}", ar=f"{100 * q['roll']['same']:.0f}",
                  st=f"{q['twist']['sp']:.2f}", sr=f"{q['roll']['sp']:.2f}")
    return t


KIN_LABELS = ("tripod range", "pinch range", "tripod pad s.v.", "pinch pad s.v.")
KIN_PHRASE = {"tripod range": "the tripod's rotation range", "pinch range": "the pinch's rotation range",
              "tripod pad s.v.": "the tripod's smallest pad singular value",
              "pinch pad s.v.": "the pinch's smallest pad singular value"}
DYN_PHRASE = {"twist ramp": "the twist's ramp", "- roll RMSE": "the roll's error", "- pin-push RMSE": "the pin-push's error",
              "- press RMSE": "the press's error"}


def cyl_set(name_pp, name_pr):
    """Per population: the pin-push from the pinch pose and the press from the press pose. Shares of layouts held in
    every seed, median pinch-angle RMSE over layouts with a held run, and how the pinch pose's kinematic range and pad
    singular value rank the pin-push's error."""
    pp = {x["key"]: x for x in FF.load_hom(name_pp)}
    pr = {x["key"]: x for x in FF.load_hom(name_pr)}
    if not pp or not pr:
        return None
    th = [x for x in pp.values() if np.isfinite(x["theta"])]
    return {"n_pp": len(pp), "n_pr": len(pr), "runs": len(load(name_pp)) + len(load(name_pr)),
            "held_pp": float(np.mean([x["held"] >= 1.0 for x in pp.values()])),
            "held_pr": float(np.mean([x["held"] >= 1.0 for x in pr.values()])),
            "th_pp": float(np.median([x["theta"] for x in th])),
            "th_pr": float(np.nanmedian([x["theta"] for x in pr.values()])),
            "sp_range": sp([x["kin_pinch"]["tmr_deg"] for x in th], [x["theta"] for x in th]),
            "sp_smin": sp([x["kin_pinch"]["smin_mm"] for x in th], [x["theta"] for x in th])}


def sec_gpu(N):
    V = N["val"]
    tw, ro = V.get("twist"), V.get("roll")
    if not tw or not ro:
        return "<p>GPU_PENDING</p>"
    big = {"wb": "?", "mtb": "?", "mrb": "?", "gpu": "GPU"}
    for task, key in (("twist", "mtb"), ("roll", "mrb")):
        tim = [t_ for t_ in load(f"hom6d_gpu_{task}_timing") if t_["worlds"] >= 1000]
        if tim:
            big[key] = f"{1000 * sum(t_['run_s'] for t_ in tim) / sum(t_['worlds'] for t_ in tim):.0f}"
            big["wb"] = f"{max(t_['worlds'] for t_ in tim):,}"
            big["gpu"] = tim[0].get("gpu", "GPU").replace("NVIDIA ", "")
    t = fill(r"""<p>A batched version of the HOM controller runs the four tasks on MuJoCo-Warp, at <<mt>> ms per
twist rollout and <<mr>> ms per roll rollout, against about <<mc>> s on one CPU core (<code>scripts/hom_gpu.py</code>). On
the <<nt>> rollouts of each sphere task in the 2-D case it agrees with the CPU runner. The held outcome matches in <<at>> % of
twist rollouts and <<ar>> % of roll rollouts. The twist's ramp on three pads ranks the layouts the same (Spearman
<<st>>), within a median <<dt>> deg per rollout.</p>
<p>At <<wb>> worlds per batch, on the Sobol layouts below, a twist rollout takes <<mtb>> ms and a roll rollout
<<mrb>> ms on one <<gpu>>.<<pinch>></p>
<p>Four facts make the batch possible:</p>
<ul>
<li>The compiled models of two layouts differ only in the positions of the three mount bodies; every other field
matches. One model is therefore used for every layout, with one row of body positions per world. A world is one layout and
one seed.</li>
<li>MuJoCo-Warp computes the poses of geoms welded to the world once, when the data is made. The palm and the support
post are therefore mocap bodies, whose positions are set per world; the post is moved away at the lift. A first
version without this kept every world's fingers among the reference layout's mount blocks, and the joints sagged by up
to 17 deg under a constant command.</li>
<li>The controller is HOM's per-finger step written for all worlds at once in torch. Contact frames come from the centre
of pressure, which a Warp kernel gathers from the pads' contacts, or from the witness points of the rounded fingertip
block and the object. Eq. (1) uses the joint axes and anchors. Eq. (2)'s bounded least squares is solved exactly by
trying all 27 patterns of active joint-rate bounds. The cylinder tasks add the pinch frame of Eq. (3), fixed to the
cylinder when the pusher starts, and the pusher's twist. The filter runs at its minimum gain, as it does on the CPU in
these tasks, and the servo offsets follow Eq. (7) of the HOM review. A second MuJoCo-Warp state holds what the
controller measures, so the sensing variants of Section SEC[sense] act through the kinematics, as on the CPU.</li>
<li>Each batch of up to 2,048 worlds runs in its own process. Within one process, a finished batch's captured CUDA
graphs kept its memory, and the next cylinder batch could not allocate.</li>
</ul>
<p>The physics differs from the CPU runner in float32 arithmetic, a solver tolerance of $10^{-6}$ with 20 iterations,
and pads that collide only with the object.</p>""",
             mt=f"{tw['ms']:.0f}" if tw["ms"] else "?", mr=f"{ro['ms']:.0f}" if ro["ms"] else "?",
             mc=f"{V['cpu_ms'] / 1000:.1f}" if V.get("cpu_ms") else "?", nt=tw["n"],
             at=f"{100 * tw['agree']:.1f}", ar=f"{100 * ro['agree']:.1f}", st=f"{tw['sp_roll']:.3f}",
             dt=f"{tw['mad']:.2f}", pinch=pinch_text(), **big)
    body = []
    for task, x in (("Twist: ramp (deg)", tw), ("Roll: RMSE/step", ro)):
        body.append([task, f"{x['held_cpu']} / {x['held_gpu']}", f"{100 * x['agree']:.1f}", f"{x['sp_roll']:.3f}",
                     f"{x['sp_lay']:.3f}", f"{x['mad']:.3g}", f"{x['ms']:.0f}" if x["ms"] else ""])
    t += table(["Task: score", "Held,<br>CPU / GPU", "Same<br>outcome (%)", "Spearman,<br>rollouts",
                "Spearman,<br>layouts", "Median<br>difference", "GPU ms<br>per rollout"], body,
               f"The GPU evaluator against the CPU runner on the same 212 symmetric tripods, three seeds per task "
               f"({tw['n']} rollouts each). Spearman correlations of the score over the rollouts that both held and over "
               "the layouts' means. Rows <code>hom2d_shard*.jsonl</code> (CPU) and <code>hom2d_gpu_*.jsonl</code> (GPU).",
               num_from=1)
    return t


def sec_check(N):
    d2, r6, s6 = N.get("dyn2d"), N.get("dynrrt"), N.get("dynsob")
    t = fill(r"""<p>The toy's scores are kinematic, and the HOM tasks are dynamic. Run on every layout, HOM ranks them
differently from the kinematic rotation range. The first check, on the CPU, ran the twist on <<nc>> symmetric tripods
from the 2-D case (<code>scripts/fp_hom_check.py</code>), <<nr>> of which had a grasp. It gave a Spearman correlation
of <<s1>> between the kinematic range and the twist's ramp on three pads. The GPU evaluator then ran the four tasks on
every 2-D layout and on both six-coordinate populations.</p>""",
             nc=len({(r["x_sep_mm"], r["y_sep_mm"]) for r in load("hom_twist_check")}),
             nr=len(check_stats(N["check"])), s1=f"{N['sp_tmr']:.2f}")
    if d2:
        t += fill(r"""<p><b>The 2-D case</b> (Figure <<fg>>). <<n>> of the 364 layouts have an equator grasp under the
task study's rule. On them the kinematic rotation range and the twist's ramp are correlated at Spearman <<sp>>, weakly and
the wrong way round. The layouts with the largest kinematic range, a wide thumb and a close pair, are where the
twist drops the sphere. The twist's ramp peaks on a band of wider pairs instead. The roll is held almost everywhere.</p>""",
                  n=d2["n"], sp=f"{d2['sp_kin_ramp']:.2f}", fg=FIG[0] + 1)
        t += img(MEDIA / "20261029-hom_landscape_2d.png",
                 "The 2-D case under HOM on the GPU, every symmetric tripod with an equator grasp, three seeds each. "
                 "Left: the tripod's kinematic rotation range from the toy. Middle: the twist's ramp on three pads "
                 "(median over seeds). Right: the roll's velocity RMSE per step. Crosses mark layouts where the twist "
                 "dropped the sphere in two seeds or more. Colours clipped at the 5th and 95th percentiles. "
                 "<code>scripts/fp_review_figures.py dyn2d</code>.")
        fj = ROWS / "film_two_layouts.json"
        if fj.exists():
            fi = json.loads(fj.read_text())
            b, w = fi["best"], fi["dropped"]
            t += film(MEDIA / "20261029-hom_twist_two_layouts.mp4", fill(
                "The HOM twist of the 40 mm sphere on two layouts of the 2-D case, CPU runner, seed 0. Left: the layout "
                "whose twist went furthest on three pads in every seed (x_sep <<bx>> mm, y_sep <<by>> mm; kinematic "
                "range <<bk>> deg). Right: the layout with the largest kinematic range among those that dropped the "
                "sphere (x_sep <<wx>> mm, y_sep <<wy>> mm; kinematic range <<wk>> deg). <code>scripts/fp_hom_film.py</code>.",
                bx=f"{b['x_sep_mm']:.0f}", by=f"{b['y_sep_mm']:.0f}", bk=f"{b['kin_tmr_deg']:.0f}",
                wx=f"{w['x_sep_mm']:.0f}", wy=f"{w['y_sep_mm']:.0f}", wk=f"{w['kin_tmr_deg']:.0f}"))
    f2 = N["fig"].get("four2d", {})
    mt, mr = f2.get("matched_twist"), f2.get("matched_roll")
    if mt and mr and (MEDIA / "20261029-hom_matched_grasps_2d.png").exists():
        t += fill(r"""<p><b>Two grasps on the same layouts.</b> The toy solved a foundational pose for each 2-D layout,
the most interior tripod pose; the task study's rule gives the equator grasp. On the <<n>> layouts that have both, the
twist from the foundational pose reaches a median ramp of <<f1>> deg, against <<e1>> deg from the equator grasp. The roll
tracks better too, <<f2>> against <<e2>> per step. Both drop the sphere more often from the foundational pose. The twist
held in every seed on <<hf1>> % of these layouts, against <<he1>> %, and the roll on <<hf2>> %, against <<he2>> %. The
most interior pose leaves the fingers room to follow the object, and the equator grasp holds it more often (Figure
<<fg>>).</p>""",
                  n=mt["n"], f1=f"{mt['fp_median']:.0f}", e1=f"{mt['eq_median']:.0f}", f2=f"{mr['fp_median']:.2f}",
                  e2=f"{mr['eq_median']:.2f}", hf1=f"{100 * mt['held_fp']:.0f}", he1=f"{100 * mt['held_eq']:.0f}",
                  hf2=f"{100 * mr['held_fp']:.0f}", he2=f"{100 * mr['held_eq']:.0f}", fg=FIG[0] + 1)
        t += img(MEDIA / "20261029-hom_matched_grasps_2d.png",
                 "The twist and the roll on the same 2-D layouts from the task study's equator grasp (horizontal) and "
                 "from the toy's foundational pose (vertical); medians over three seeds. Grey: the sphere dropped in a "
                 "seed from either grasp. <code>scripts/fp_review_figures.py four2d</code>.")
    if f2.get("pinpush") and (MEDIA / "20261029-hom_four_tasks_2d.png").exists():
        c2 = N.get("cyl2d") or {}
        t += fill(r"""<p><b>Four tasks on the 2-D layouts</b> (Figure <<fg>>), each from its foundational pose. The
twist goes furthest on a diagonal band of layouts, from close pairs with a close thumb to wide pairs with a wide one, and
the roll tracks best on the same band. The cylinder tasks need the thumb close to the pair. With the thumb within 60 mm
of the pair, the pin-push held in every seed on <<pn>> % of layouts and the press on <<qn>> %. Beyond 60 mm, the shares
fall to <<pf>> % and <<qf>> %. Over all 2-D layouts the pinch pose's kinematic rotation range ranks the pin-push's
pinch-angle RMSE at Spearman <<sr>>.</p>""",
                  fg=FIG[0] + 1, pn=f"{100 * f2['pinpush']['held_near']:.0f}", qn=f"{100 * f2['press']['held_near']:.0f}",
                  pf=f"{100 * f2['pinpush']['held_far']:.0f}", qf=f"{100 * f2['press']['held_far']:.0f}",
                  sr=f"{c2.get('sp_range', float('nan')):.2f}")
        t += img(MEDIA / "20261029-hom_four_tasks_2d.png",
                 "The four HOM tasks on the 2-D layouts from the toy's foundational poses (tripod pose for the twist "
                 "and the roll, pinch pose for the pin-push, press pose for the press), medians over three seeds. "
                 "Crosses mark layouts where two seeds or more dropped the object. Colours clipped at the 5th and 95th "
                 "percentiles; red is better in every panel. <code>scripts/fp_review_figures.py four2d</code>.")
        fj2 = ROWS / "film_pinpush_two_layouts.json"
        if fj2.exists() and (MEDIA / "20261029-hom_pinpush_two_layouts.mp4").exists():
            fi = json.loads(fj2.read_text().replace("NaN", "null"))
            h_, d_ = fi["held"], fi["dropped"]
            t += film(MEDIA / "20261029-hom_pinpush_two_layouts.mp4", fill(
                "The pin-push of the 25 mm cylinder from the toy's pinch pose on two 2-D layouts, CPU runner, seed 0. "
                "Left: the thumb close to the pair (x_sep <<hx>> mm, y_sep <<hy>> mm), held in every GPU seed and here, "
                "pinch-angle RMSE <<hr>> deg. Right: a wide thumb (x_sep <<dx>> mm, y_sep <<dy>> mm), dropped in every "
                "GPU seed and here. <code>scripts/fp_hom_film.py pinpush</code>.",
                hx=f"{h_['x_sep_mm']:.0f}", hy=f"{h_['y_sep_mm']:.0f}", hr=f"{h_['cpu_seed0']['theta_rmse_deg']:.1f}",
                dx=f"{d_['x_sep_mm']:.0f}", dy=f"{d_['y_sep_mm']:.0f}"))
    rows = []
    for name, x in (("2-D, equator grasp", d2), ("2-D, foundational pose", N.get("dyn2dfp")),
                    ("6-D RRT, foundational pose", r6), ("6-D Sobol, foundational pose", s6)):
        if x:
            rows.append([name, f"{x['n']:,}", f"{100 * x['held_tw']:.0f}", f"{100 * x['held_ro']:.0f}",
                         f"{x['ramp_all']:.0f}", f"{x['rmse_tw']:.2f}", f"{x['rmse_med']:.2f}",
                         f"{x['sp_kin_ramp']:.2f}", f"{x['sp_smin_rmse']:.2f}"])
    if rows:
        nr = sum(x["runs"] for x in (d2, N.get("dyn2dfp"), r6, s6) if x)
        t += table(["Layouts and<br>starting grasp", "Layouts", "Twist<br>held (%)", "Roll<br>held (%)",
                    "Twist<br>ramp (deg)", "Twist<br>RMSE", "Roll<br>RMSE", "Range vs<br>ramp", "Pad s.v. vs<br>roll RMSE"],
                   rows,
                   f"HOM's twist and roll on the GPU, three seeds per layout and task ({nr:,} rollouts), against the "
                   "toy's kinematic tripod metrics. Held: the share of layouts held in all three seeds. Ramp, twist "
                   "RMSE and the range-ramp correlation: over all layouts. Roll RMSE and its correlation with the pads' "
                   "smallest singular value: over the layouts held in every run of both tasks.", num_from=1)
    if s6 and r6:
        t += fill(r"""<p><b>The six-coordinate layouts</b> start from their foundational poses. On the Sobol layouts the
twist reaches a median ramp of <<rs>> deg, and the roll held in every seed on <<hs>> % of them. From this pose the
kinematic range barely ranks the twist: Spearman <<ss>> on the Sobol layouts and <<s2>> on the 2-D layouts. The pads'
smallest singular value ranks the roll's error at <<sm>>.</p>""",
                  rs=f"{s6['ramp_all']:.0f}", hs=f"{100 * s6['held_ro']:.0f}", ss=f"{s6['sp_kin_ramp']:.2f}",
                  s2=f"{(N.get('dyn2dfp') or {}).get('sp_kin_ramp', float('nan')):.2f}", sm=f"{s6['sp_smin_rmse']:.2f}")
        if (MEDIA / "20261029-hom_landscape_6d.png").exists():
            t += img(MEDIA / "20261029-hom_landscape_6d.png",
                     "The Sobol layouts under HOM from their foundational poses. Left: the twist's ramp on three pads "
                     "against the kinematic rotation range; grey where the twist dropped the sphere in a seed. Middle: the dynamic Pareto front of the twist's ramp and the "
                     "roll's error, over the layouts held in every run. Right: the t-SNE map of the Sobol layouts that "
                     "reach both poses, coloured by the twist's ramp. <code>scripts/fp_review_figures.py dyn6d</code>.")
    cyl = [(name, x) for name, x in (("2-D", N.get("cyl2d")), ("6-D RRT", N.get("cylrrt")), ("6-D Sobol", N.get("cylsob")))
           if x]
    if cyl:
        t += fill(r"""<p><b>The cylinder tasks</b> (Table <<tb>>) hold less often than the sphere tasks on every
population. The pinch pose's kinematic range does not rank the pin-push's error either; on the <<pop>> layouts the
Spearman correlation is <<sr>>. With three seeds, a single layout's pin-push score is rough, since two seeds of the
same runner disagree on the held outcome on a quarter of the layouts (Section SEC[gpu]). Shares over many layouts, such
as the thumb's distance above, are firmer than any one layout's rank.</p>""", tb=TAB[0] + 1, pop=cyl[-1][0],
                  sr=f"{cyl[-1][1]['sp_range']:.2f}")
        t += table(["Layouts", "Pin-push<br>layouts", "Press<br>layouts", "Pin-push<br>held (%)", "Press<br>held (%)",
                    "Pin-push<br>RMSE (deg)", "Press<br>RMSE (deg)", "Range vs<br>pin-push", "Pad s.v. vs<br>pin-push"],
                   [[name, f"{x['n_pp']:,}", f"{x['n_pr']:,}", f"{100 * x['held_pp']:.0f}", f"{100 * x['held_pr']:.0f}",
                     f"{x['th_pp']:.1f}", f"{x['th_pr']:.1f}", f"{x['sp_range']:.2f}", f"{x['sp_smin']:.2f}"]
                    for name, x in cyl],
                   f"HOM's pin-push from the pinch pose and press from the press pose on the GPU, three seeds per layout "
                   f"({sum(x['runs'] for _n, x in cyl):,} rollouts). Held: the share of layouts held in all three seeds. "
                   "RMSE: the median over layouts of the median pinch-angle RMSE over held runs. Correlations (Spearman) "
                   "of the pinch pose's kinematic range and pad singular value with the pin-push's RMSE.", num_from=1)
    f6 = N["fig"].get("four6d", {})
    if f6.get("pareto") and (MEDIA / "20261029-hom_four_tasks_6d.png").exists():
        spm = f6["spearman"]
        cl = f6.get("cluster") or {}
        mg, ma = cl.get("med_good", {}), cl.get("med_all", {})
        best_kin = max(((k, v) for k, v in spm.items() if k.split(" | ")[0] in KIN_LABELS
                        and k.split(" | ")[1] not in KIN_LABELS), key=lambda kv: abs(kv[1]))
        t += fill(r"""<p><b>Four tasks on the Sobol layouts</b> (Figure <<fg>>). Each cylinder task holds in every seed
on <<hc>> % of the layouts, and <<ng>> of the <<n>> layouts hold every run of all four tasks. The pin-push needs the
thumb near the index. It held in every seed on <<p60>> % of the <<n60>> layouts with the two mounts within 60 mm. With
the mounts more than 80 mm apart it held on <<p80>> %.</p>
<p>The <<ng>> layouts that hold all four tasks lie close together. In the standardized six coordinates their mean
distance to the nearest other one is <<nn>>, against <<nr>> ± <<ns>> for random sets of the same size. They bring the
thumb and index mounts toward the palm's centre in $x$: medians <<gtx>> and <<gix>> mm, against <<atx>> and <<aix>> mm
over all layouts. They also shift the thumb toward the middle finger, to $y$ = <<gty>> mm against <<aty>> mm. In the
t-SNE map most
of them fall in three neighbourhoods, and so do most of the <<np>> layouts of the four-task Pareto front.</p>
<p>Over these layouts the four tasks rank the layouts differently. With every score signed so that larger is better,
twist and roll correlate at Spearman <<tr>>, roll and pin-push at <<rp>>, pin-push and press at <<pq>>. The best
kinematic predictor of a dynamic score is <<bk>> (Spearman <<bv>>).</p>""",
                  fg=FIG[0] + 1, ng=f"{f6['good']:,}", n=f"{f6['n']:,}", np=f6["pareto"],
                  hc=f"{100 * min(f6['held_each']['pinpush'], f6['held_each']['press']):.0f}" if
                  abs(f6['held_each']['pinpush'] - f6['held_each']['press']) < 0.01 else
                  f"{100 * f6['held_each']['pinpush']:.0f} and {100 * f6['held_each']['press']:.0f}",
                  p60=f"{100 * cl.get('pinpush_held_ti60', float('nan')):.0f}", n60=cl.get("n_ti60", "?"),
                  p80=f"{100 * cl.get('pinpush_held_ti80', float('nan')):.0f}",
                  nn=f"{cl.get('nn_good', float('nan')):.3f}", nr=f"{cl.get('nn_rand', float('nan')):.3f}",
                  ns=f"{cl.get('nn_rand_sd', float('nan')):.3f}",
                  gtx=f"{mg.get('tx', float('nan')):.0f}", gix=f"{mg.get('ix', float('nan')):.0f}",
                  atx=f"{ma.get('tx', float('nan')):.0f}", aix=f"{ma.get('ix', float('nan')):.0f}",
                  gty=f"{mg.get('ty', float('nan')):.0f}", aty=f"{ma.get('ty', float('nan')):.0f}",
                  tr=f"{spm['twist ramp | - roll RMSE']:.2f}", rp=f"{spm['- roll RMSE | - pin-push RMSE']:.2f}",
                  pq=f"{spm['- pin-push RMSE | - press RMSE']:.2f}",
                  bk=f"{KIN_PHRASE[best_kin[0].split(' | ')[0]]} for {DYN_PHRASE[best_kin[0].split(' | ')[1]]}",
                  bv=f"{best_kin[1]:.2f}")
        t += img(MEDIA / "20261029-hom_four_tasks_6d.png",
                 "The Sobol layouts under the four HOM tasks from their foundational poses, three seeds each. Left: "
                 "Spearman correlations between the toy's kinematic scores (upper left block) and the four dynamic "
                 "scores, signed so that larger is better, over the layouts held in every run. Middle: the four-task "
                 "Pareto front, each layout's percentile per task (red), with every seventh other layout in grey. Right: "
                 "the t-SNE map of the Sobol layouts; those held in every run of all four tasks are coloured by their "
                 "mean percentile, the front ringed. "
                 "<code>scripts/fp_review_figures.py four6d</code>.")
    return t


TASK_LABEL = {"twist": "twist", "roll": "roll", "pinpush": "pin-push", "press": "press"}


def roll_check():
    """The roll on Sobol layouts held in every seed whose exact-sensing error exceeds 2 per step: GPU medians under
    exact sensing and calibration errors without fingertips, and the CPU runner's medians on twelve of them under four
    variants (scripts/fp_roll_sensing_check.py, rows/cpu_check_roll_poor)."""
    V = {v: {x["key"]: x for x in FF.load_hom("hom6d_gpu_sense_roll", v)} for v in ("rate50", "cal_notip")}
    if not V["rate50"]:
        return None
    poor = [k for k in V["rate50"] if V["rate50"][k]["held"] >= 1 and k in V["cal_notip"]
            and V["cal_notip"][k]["held"] >= 1 and V["rate50"][k]["rmse"] > 2.0]
    cpu = load("cpu_check_roll_poor")
    out = {"n": len(poor), "gpu_exact": float(np.median([V["rate50"][k]["rmse"] for k in poor])),
           "gpu_cal": float(np.median([V["cal_notip"][k]["rmse"] for k in poor])), "cpu": {}}
    for v in ("rate50", "notip0", "notip", "cal_notip"):
        rr = [r for r in cpu if r["variant"] == v]
        x = [r["w_rmse_ratio"] for r in rr if r["held"]]
        if rr:
            out["cpu"][v] = {"n": len(rr), "held": sum(bool(r["held"]) for r in rr), "med": float(np.median(x))}
    return out


def sec_sense(N):
    """The Sobol layouts under the HOM review's sensing variants at 50 Hz (rows/hom6d_gpu_sense_*)."""
    f = N["fig"].get("sense6d") or {}
    tasks = [t for t in ("twist", "roll", "pinpush", "press") if t in f]
    if not tasks:
        return "<p>SENSE_PENDING</p>"
    runs = sum(len(load(f"hom6d_gpu_sense_{t}")) for t in tasks)
    nl = max(f[t]["cal_tip"]["n"] for t in tasks)
    t = fill(r"""<p>A layout chosen in simulation must keep its rank on a hand whose joints and object pose are
calibrated only to about a degree and a millimetre. The GPU evaluator therefore reran the four tasks on <<nl>> Sobol
layouts under four of the HOM review's sensing variants. They are exact sensing, and calibration errors without
fingertips, with the measured fingertips, and with the fingertips closing the force loop. All run at the bench's 50 Hz, three seeds each
(<<nr>> rollouts). Each layout's score under exact sensing is set against its score under the calibration errors in
Figure <<fg>>.</p>""", nl=f"{nl:,}", nr=f"{runs:,}", fg=FIG[0] + 1)
    body = []
    for task in tasks:
        a, b = f[task]["cal_tip"], f[task]["cal_tipfl"]
        nd = 1 if task in ("pinpush", "press") else 2
        sd = f[task].get("seeds") or {}
        body.append([TASK_LABEL[task].capitalize(), f"{100 * a['held_exact']:.0f}", f"{100 * a['held']:.0f}",
                     f"{100 * b['held']:.0f}", f"{a['median_exact']:.{nd}f}", f"{a['median']:.{nd}f}",
                     f"{b['median']:.{nd}f}", f"{sd.get('spearman', float('nan')):.2f}", f"{a['spearman']:.2f}",
                     f"{b['spearman']:.2f}", f"{100 * sd.get('top10_kept', float('nan')):.0f}",
                     f"{100 * a['top10_kept']:.0f}", f"{100 * b['top10_kept']:.0f}"])
    def lst(vals, fmt):
        words = [fmt(v) for v in vals]
        return ", ".join(words[:-1]) + " and " + words[-1] if len(words) > 1 else words[0]

    names = lst([TASK_LABEL[x] for x in tasks], str)
    pct = lambda v: f"{100 * v:.0f}"  # noqa: E731
    two = lambda v: f"{v:.2f}"  # noqa: E731
    t += fill(r"""<p>Calibration errors change which layouts hold (Table <<tab>>). With exact sensing at 50 Hz, the
<<names>> hold in every seed on <<h0>> % of the layouts. With calibration errors and the fingertips the shares become
<<h1>> %: the twist loses half its layouts, and the other three tasks gain.</p>""", tab=TAB[0] + 1, names=names,
              h0=lst([f[x]["cal_tip"]["held_exact"] for x in tasks], pct),
              h1=lst([f[x]["cal_tip"]["held"] for x in tasks], pct))
    if all(f[x].get("seeds") for x in tasks):
        t += fill(r"""<p>They also reorder the layouts that hold, more than a new set of seeds does. Among the layouts
held under both, the scores keep their order at Spearman <<s1>> for the <<names>>. Exact sensing with new seeds gives
<<sb>>. Of the best tenth of layouts under exact sensing, <<k1>> % stay in the best tenth with calibration errors.
With new seeds, <<kb>> % stay. A front chosen under exact sensing would therefore change on a hand with these errors, so the
scores that select layouts should be computed under the sensing the hand will have.</p>""",
                  names=names, s1=lst([f[x]["cal_tip"]["spearman"] for x in tasks], two),
                  sb=lst([f[x]["seeds"]["spearman"] for x in tasks], two),
                  k1=lst([f[x]["cal_tip"]["top10_kept"] for x in tasks], pct),
                  kb=lst([f[x]["seeds"]["top10_kept"] for x in tasks], pct))
    t += fill(r"""<p>With the force loop the shares become <<h2>> %, and the scores keep their order at Spearman
<<s2>>.</p>""", h2=lst([f[x]["cal_tipfl"]["held"] for x in tasks], pct),
              s2=lst([f[x]["cal_tipfl"]["spearman"] for x in tasks], two))
    t += table(["Task", "Held (%):<br>exact", "Cal.,<br>tips", "Cal., tips,<br>loop", "Score:<br>exact", "Cal.,<br>tips",
                "Cal., tips,<br>loop", "Spearman:<br>new seeds", "Cal.,<br>tips", "Cal., tips,<br>loop",
                "Best tenth<br>kept (%):<br>new seeds", "Cal.,<br>tips", "Cal., tips,<br>loop"], body,
               "The Sobol layouts under sensing variants at 50 Hz, three seeds each. Held: the share of layouts held in "
               "every seed. Score: the median over layouts held in every seed under both variants of the ramp on three "
               "pads (twist, deg), the velocity RMSE per step (roll) or the pinch-angle RMSE (pin-push, press, deg). "
               "Spearman: each layout's score against its score under exact sensing with seeds 0 to 2; new seeds is "
               "exact sensing with seeds 3 to 5, the spread between seed sets alone. Best tenth kept: the share of the "
               "best tenth under exact sensing that stays in the best tenth.", num_from=1)
    rc = roll_check()
    if rc and "roll" in f and len(rc["cpu"]) == 4:
        a = f["roll"]["cal_tip"]
        t += fill(r"""<p>The roll holds more often under calibration errors too: on <<h1>> % of the layouts against <<h0>> %
with exact sensing. The layouts that track it poorly with exact sensing also track better. On the <<n>> layouts held in both with an exact-sensing error above 2 per step, the median falls from <<g0>> to
<<g1>> per step with calibration errors and no fingertips. On the CPU runner, twelve of these layouts in seed 0 give
<<c0>> with exact sensing and <<c1>> with the exact pose but no fingertips. The tag tracker's noise and 40 ms delay
alone give <<c2>>, and adding the calibration offsets gives <<c3>> (<code>scripts/fp_roll_sensing_check.py</code>). From
these foundational poses the roll therefore tracks better with a noisy, late object pose than with the exact one;
neither the fingertips nor the calibration offsets cause the gain.</p>""",
                  h1=f"{100 * a['held']:.0f}", h0=f"{100 * a['held_exact']:.0f}", n=rc["n"], g0=f"{rc['gpu_exact']:.2f}",
                  g1=f"{rc['gpu_cal']:.2f}", c0=f"{rc['cpu']['rate50']['med']:.2f}", c1=f"{rc['cpu']['notip0']['med']:.2f}",
                  c2=f"{rc['cpu']['notip']['med']:.2f}", c3=f"{rc['cpu']['cal_notip']['med']:.2f}")
    fo = f.get("force") or {}
    if fo.get("pinpush") and fo.get("press"):
        pc = fo["pinpush"]["cal_notip"]
        t += fill(r"""<p>On the pinch tasks the calibration errors make the pads press harder. In the pin-push the total
of the three pads' mean forces rose by a median <<dc>> N over the <<f0>> N of exact sensing, and it rose in <<pp>> % of
runs. The force loop removes that rise (median <<dl>> N).</p>""", dc=f"{pc['total_change_median']:.1f}",
                  f0=f"{fo['pinpush']['exact_total_median']:.1f}", pp=f"{100 * pc['total_change_positive']:.0f}",
                  dl=f"{fo['pinpush']['cal_tipfl']['total_change_median']:.1f}")
        t += fill(r"""<p>The force error per layout is larger on these poses than on the task study's grasps (HOM
review, Table 5). Without fingertips, the pinch force is off by a median <<a0>> N on a pin-push layout and <<b0>> N on
a press layout. It is off by more than 1 N on <<a1>> % and <<b1>> % of layouts. The force loop on the fingertips' force brings
the medians to <<a2>> and <<b2>> N, and the share of layouts over 1 N to <<a3>> % and <<b3>> %.</p>""",
                  a0=f"{fo['pinpush']['cal_notip']['layout_median']:.2f}", b0=f"{fo['press']['cal_notip']['layout_median']:.2f}",
                  a1=f"{100 * fo['pinpush']['cal_notip']['layouts_over_1N']:.0f}",
                  b1=f"{100 * fo['press']['cal_notip']['layouts_over_1N']:.0f}",
                  a2=f"{fo['pinpush']['cal_tipfl']['layout_median']:.2f}", b2=f"{fo['press']['cal_tipfl']['layout_median']:.2f}",
                  a3=f"{100 * fo['pinpush']['cal_tipfl']['layouts_over_1N']:.0f}",
                  b3=f"{100 * fo['press']['cal_tipfl']['layouts_over_1N']:.0f}")
    if (MEDIA / "20261029-hom_sensing_6d.png").exists():
        t += img(MEDIA / "20261029-hom_sensing_6d.png",
                 "Each Sobol layout's score under exact sensing (horizontal) against its score with calibration errors "
                 "and the measured fingertips (top) and with the fingertips closing the force loop (bottom). Layouts held "
                 "in every seed under both; medians over three seeds at 50 Hz. Dotted: equal scores. "
                 "<code>scripts/fp_review_figures.py sense6d</code>.")
    return t


def v4b_numbers():
    """Pose yields of the v4b tip on the 8,192 Sobol layouts per object size, and the load on the rigid parts in the
    held rollouts of the v4b check (GPU, seed 0)."""
    out = {"yield": {}}
    for shape, sizes in (("tripod", (30, 40, 50, 63)), ("pinch", (10, 16, 25, 40))):
        for d in sizes:
            rows = load(f"sobol8k_{shape}{d}_v4b")
            out["yield"][(shape, d)] = {k: sum(1 for r in rows if (r["fp"].get(k) or {}).get("ok"))
                                        for k in (("tripod",) if shape == "tripod" else ("pinch", "press"))}
    for t in ("twist_d40_s0", "pinpush_d25_s0"):
        rr = [json.loads(x) for x in (ROWS / "v4b_check" / f"{t}.jsonl").read_text().splitlines() if x.strip()]
        lf = np.array([r.get("link_force_max_N", np.nan) for r in rr if r.get("held")], float)
        out[t] = (len(rr), int(np.isfinite(lf).sum()), float(np.nanmedian(lf)), float(np.nanpercentile(lf, 90)))
    return out


def grid_numbers():
    """The grid at the default objects: per task the summary of hom_landscape.py grid, the basin peaks of the smoothed
    best-of-16 map, and Spearman correlations of the feasible log volume and of the grasp-matrix singular value with
    the scores."""
    import hom_landscape as L
    summ = json.loads((L.GRID_ROWS / "summary.json").read_text())
    out = {}
    for task, d, _name, _obj in GRID_TASKS:
        xs, ys, M = L.grid_maps(task, d)
        sm = L.smooth_map(M["best"])
        pk, pks = L.peaks(sm)
        nx = len(xs)
        peaks = sorted(((float(xs[p % nx]), float(ys[p // nx]), float(sm.flat[p]), int((pk == p).sum())) for p in pks),
                       key=lambda x: -x[3])
        tab = [r for r in L.grid_table(task, d) if r.get("n_grasps")]
        best = np.array([r["best"] for r in tab])
        lv = np.array([np.nan if r["logvol"] is None else r["logvol"] for r in tab], float)
        ok = np.isfinite(lv)
        sc, _held = L.grid_scores(task, d)
        pairs, within = [], []
        for call, (_x, _y, rec) in L.grid_poses(task, d).items():
            if not rec or not rec.get("ok"):
                continue
            pr = [(g["gsv"], float(np.mean(sc[(call, j)]))) for j, g in enumerate(rec.get("grasps") or [rec])
                  if (call, j) in sc and g.get("gsv") is not None]
            pairs += pr
            a = np.array(pr)
            if len(pr) >= 5 and a[:, 0].std() > 0 and a[:, 1].std() > 0:
                within.append(spearman(a[:, 0], a[:, 1]))
        a = np.array(pairs)
        out[task] = dict(summ[f"{task}_d{d:g}"], peaks=peaks, lv_best=spearman(lv[ok], best[ok]),
                         gsv_all=spearman(a[:, 0], a[:, 1]), gsv_within=float(np.median(within)), n_grasps=len(a))
        if task == "twist":                     # how the winning grasp differs from the foundational pose
            byc = {r["call"]: r for r in tab}
            dv = []
            for call, (_x, _y, rec) in L.grid_poses(task, d).items():
                r = byc.get(call)
                if rec and rec.get("ok") and r and r.get("best_grasp") is not None:
                    vb, v0 = (np.asarray(rec["grasps"][j]["v"], float) for j in (r["best_grasp"], 0))
                    dv.append((1e3 * abs(vb[2] - v0[2]), math.degrees(abs(vb[4] - v0[4])), math.degrees(abs(vb[3])),
                               math.degrees(vb[4]), math.degrees(v0[4])))
            dv = np.array(dv)
            out[task]["vars"] = {"depth_mm": float(np.median(dv[:, 0])), "index_turn": np.percentile(dv[:, 1], [25, 75]),
                                 "thumb_off": float(np.mean(dv[:, 2] > 2.0)), "index_best": np.percentile(dv[:, 3], [25, 75]),
                                 "index_fp": np.percentile(dv[:, 4], [5, 95])}
    return out


def valrep_numbers():
    """The grid's fresh-seed validation and grasp transfer (hom_landscape.py valrep and derive): per task and size,
    valrep.json's row with the feasibility of the moved grasps from transfer_feasible.json; empty before they ran."""
    pv, pf = ROWS / "grid2d_v4b" / "valrep.json", ROWS / "grid2d_v4b" / "transfer_feasible.json"
    if not pv.exists():
        return {}
    f = json.loads(pf.read_text())
    return {k: dict(r, feas=f.get(k, {})) for k, r in json.loads(pv.read_text()).items()
            if isinstance(r, dict) and r.get("nb_n")}


def sec_v4b(N):
    V = N["v4b"]
    y = V["yield"]
    t = fill(r"""<p>Sections SEC[gpu] to SEC[sense] used the printed TPU block as the fingertip. The hand calibrated on the UR5
has the v4b fingertip. Its pad, half an ellipsoid, sits in a cassette under a cap trimmed at the pad's top edge, on DIP
link b v4. The pad's semi-axes are 8.5 mm deep, 7.4 mm across, 11 mm toward the tip and 22 mm toward the PIP. Its apex
lies 10.55 mm from the finger's axis. The tip body sits 36.36 mm below the PIP axis, 9.75 mm lower than the block's. From
this section on, every pose and rollout uses the v4b fingertip (<code>SCALE_HAND=cad_v2</code>).</p>
<p><code>scripts/build_cad_v2_finger.py</code> places the cell's v4b parts on the CAD finger. For the plant it packs the
rigid parts with spheres: 160 on the DIP link, 85 on the cassette and 70 on the cap with its four M2 button heads.
Spheres were added greedily until 97 % of 60,000 surface samples lay within 1 mm of one; the link reached 73 % at its
budget of 160. Each pad is 1,018 spheres of 0.75 mm radius, with stiffnesses from pad builder v2 (Figure <<fa>>). In
the controller the pad is an analytic half-ellipsoid; the code walkthrough of the HOM review page derives its
closest-point solver.</p>
<p>The pad covers only part of its ellipsoid, and it is flat toward the PIP. Tilting a contact normal 10 deg toward the
PIP moves the pad's support point to within 1.5 mm of the pad's end; toward the tip, the same takes 48 deg. The toy's
poses therefore have two more constraints. The pad-coverage slack keeps the support point along each contact normal at
least 1.5 mm inside the pad's ends (<code>fp_sr2_toy.pad_slack</code>). The rigid-clearance slack keeps the spheres of
the link, cassette and cap clear of the object, with the finger placed where the plant's open pose puts it
(<code>rigid_clearance</code>). Without the two constraints the twist's touch poses put the caps 1.2 to 4.0 mm inside the
40 mm sphere (Figure <<fb>>). With them the rigid parts clear the sphere by 1.4 mm or more (Figure <<fc>>).</p>""",
             fa=FIG[0] + 1, fb=FIG[0] + 2, fc=FIG[0] + 3)
    t += img(MEDIA / "20261030-cad_v2_finger_check.png",
             "The v4b fingertip on the CAD finger from five sides: top row with the part meshes, bottom row with the "
             "sphere packs of the rigid parts and the pad's 0.75 mm spheres only.", jpeg=True)
    t += img(MEDIA / "20261030-v4b_touch_check_unconstrained.png",
             "Touch poses of the twist grasp on the 40 mm sphere for four Sobol layouts, solved without the pad-coverage "
             "and rigid-clearance constraints. The rigid parts' spheres are drawn; the caps sit inside the sphere.", jpeg=True)
    t += img(MEDIA / "20261030-v4b_touch_twist.png",
             "Touch poses of the twist grasp for three Sobol layouts with both constraints, from the side, the front "
             "and below.", jpeg=True)
    rows = []
    for (shape, d), v in y.items():
        rows.append([("sphere" if shape == "tripod" else "cylinder") + f" {d} mm", shape if shape == "tripod" else
                     "pinch", f"{v.get('tripod', v.get('pinch')):,}", f"{v['press']:,}" if "press" in v else ""])
    nt, nh_t, lt, lt9 = V["twist_d40_s0"]
    npp, nh_p, lp, lp9 = V["pinpush_d25_s0"]
    t += fill(r"""<p>Table SEC_TAB_Y gives the layouts that reach each pose with the v4b fingertip. Larger objects are
reached by more layouts: the tripod pose by <<t30>> of 8,192 on the 30 mm sphere and <<t63>> on the 63 mm sphere. In the
<<nhp>> held pin-pushes of seed 0, the largest force on the rigid parts had a median of <<lp>> N and a 90th percentile of
<<lp9>> N. In the twist they touch the sphere at the end of the ramp, when a pad leaves
it: median <<lt>> N over <<nht>> held twists. The CPU and GPU controllers agree on the held outcome as often as two seeds
of the GPU controller do (the code walkthrough, Section 13).</p>""",
              t30=f"{y[('tripod', 30)]['tripod']:,}", t63=f"{y[('tripod', 63)]['tripod']:,}", nhp=f"{nh_p:,}",
              lp=f"{lp:.1f}", lp9=f"{lp9:.1f}", lt=f"{lt:.1f}", nht=f"{nh_t:,}").replace("SEC_TAB_Y", str(TAB[0] + 1))
    t += table(["Object", "Pose", "Layouts reaching the pose", "Reaching the press pose"], rows,
               "Sobol layouts (of 8,192) that reach each pose with the v4b fingertip, per object size.", num_from=2)
    return t


def sec_grid(N):
    G = N["grid"]
    tw, ro, pp, pr = (G[t] for t in ("twist", "roll", "pinpush", "press"))
    t = r"""<p>Until this section a layout's score came from one grasp, its foundational pose. That pose maximizes the
margin to the kinematic constraints. Whether a grasp survives the task is decided by friction and the servo forces,
which the margin does not include. Each layout of a grid of symmetric tripods therefore gets a set of up to 16 grasps
(<code>fp_sr2_toy.py grid2d</code>). The grid spans $x_{sep}$ from 40 to 160 mm and $y_{sep}$ from 50 to 170 mm in steps
of 5 mm, 625 layouts. Two parameters keep the maps readable; the six coordinates follow in the size
study.</p>
<p>The feasible set of a pose is thin. The object's position is free by a few millimetres and the contact coordinates
by tens of degrees, and uniform perturbations of 35 % of the bounds around the foundational pose were all infeasible.
The sampler first scans each variable along a line through the foundational pose, at 41 points over its bounds and
then over a fortieth of them, to find its feasible width. It then walks for 600 steps: each step moves one or two
variables by up to half their width and is kept when every constraint is met. From the visited grasps that pass the
clearance checks, 15 are chosen in farthest-point order, in coordinates divided by the widths. The widths also give the
feasible log volume, the size of the layout's grasp set.</p>"""
    rows = []
    for task, d, name, obj in GRID_TASKS:
        g = G[task]
        rows.append([name, obj, str(g["run"]), f"{g['fp_median']:.2f}", f"{g['best_median']:.2f}",
                     f"{100 * g['best_is_fp']:.0f}", f"{g['fp_vs_best_spearman']:.2f}", f"{g['held_share_median']:.2f}",
                     f"{g['lv_best']:.2f}", f"{g['gsv_within']:.2f}"])
    t += fill(r"""<p>Every grasp of every reached layout ran the four HOM tasks on the GPU with three seeds (Table
SEC_TAB_G). The best of 16 raised the median score over the foundational pose in every task: from <<a0>> to <<a1>> in the
twist and from <<c0>> to <<c1>> in the pin-push. The roll rose from <<b0>> to <<b1>> and the press from <<d0>> to <<d1>>. The foundational
pose was the best grasp on <<e0>> to <<e1>> % of the layouts. Its score ranks the layouts' best scores only roughly, at
Spearman <<f0>> to <<f1>>. The grasp therefore has to be searched per layout and per task.</p>""",
              a0=f"{tw['fp_median']:.2f}", a1=f"{tw['best_median']:.2f}", b0=f"{ro['fp_median']:.2f}",
              b1=f"{ro['best_median']:.2f}", c0=f"{pp['fp_median']:.2f}", c1=f"{pp['best_median']:.2f}",
              d0=f"{pr['fp_median']:.2f}", d1=f"{pr['best_median']:.2f}",
              e0=f"{100 * min(g['best_is_fp'] for g in G.values()):.0f}",
              e1=f"{100 * max(g['best_is_fp'] for g in G.values()):.0f}",
              f0=f"{min(g['fp_vs_best_spearman'] for g in G.values()):.2f}",
              f1=f"{max(g['fp_vs_best_spearman'] for g in G.values()):.2f}").replace("SEC_TAB_G", str(TAB[0] + 1))
    t += table(["Task", "Object", "Layouts", "FP median", "Best-of-16 median", "FP best (%)", "Spearman, FP and best",
                "Held share, median", "Spearman, log volume and best", "Spearman, grasp-matrix value and score"], rows,
               "Grasp sets on the 2-D grid at the default objects, v4b fingertip, seeds 0 to 2. The last column is the "
               "median over layouts of the rank correlation between a grasp's grasp-matrix singular value and its "
               "score, within each layout's set.", num_from=2)

    t += fill(r"""<p>The four tasks peak at different layouts (Figures <<f1>> to <<f4>>, bottom right). The twist's largest
basin peaks at $x_{sep}$ = <<tx>> mm and $y_{sep}$ = <<ty>> mm. The roll's peaks at (<<rx>>, <<ry>>) mm, with the index
and middle closer together. The pin-push has one basin, with its peak at (<<px>>, <<py>>) mm on the grid's edge of the
smallest index-to-middle spacing. The press has <<npr>> basins; the largest peaks at (<<p1x>>, <<p1y>>) mm and the highest
at (<<p2x>>, <<p2y>>) mm. The twist and the roll share a
sphere and a grasp family, yet their best layouts lie <<dy>> mm apart in $y_{sep}$.</p>
<p>Two kinematic quantities were tested as predictors. The feasible log volume of a layout's pose ranks its best-of-16
score at Spearman <<l0>> to <<l1>>: layouts with more room around the foundational pose offer better grasps. The
earlier kinematic scores, the rotation range among them, did not rank the rollouts (Section SEC[check]). The grasp
metric of [6] is the smallest singular value of the grasp matrix. Within a layout it ranks the roll's grasps at a median
Spearman of <<g_ro>>, and the twist's, pin-push's and press's at <<g_tw>>, <<g_pp>> and <<g_pr>>.</p>
<p>Figure <<f5>> shows which grasp won the twist. The best grasp keeps the sphere's depth within a median <<dd>> mm
of the foundational pose's. Its index contact sits between <<ib0>> and <<ib1>> deg from the symmetric azimuth on the
middle half of the layouts, while the foundational pose's spans <<if0>> to <<if1>> deg over 90 % of them. Its thumb contact leaves the
symmetric position, which every foundational pose keeps, on <<th>> % of the layouts.</p>""",
              f1=FIG[0] + 1, f4=FIG[0] + 4, f5=FIG[0] + 5,
              tx=f"{tw['peaks'][0][0]:.0f}", ty=f"{tw['peaks'][0][1]:.0f}", rx=f"{ro['peaks'][0][0]:.0f}",
              ry=f"{ro['peaks'][0][1]:.0f}", px=f"{pp['peaks'][0][0]:.0f}", py=f"{pp['peaks'][0][1]:.0f}",
              npr={1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(len(pr["peaks"]), str(len(pr["peaks"]))),
              p1x=f"{pr['peaks'][0][0]:.0f}", p1y=f"{pr['peaks'][0][1]:.0f}",
              p2x=f"{max(pr['peaks'], key=lambda q: q[2])[0]:.0f}", p2y=f"{max(pr['peaks'], key=lambda q: q[2])[1]:.0f}",
              l0=f"{min(g['lv_best'] for g in G.values()):.2f}", l1=f"{max(g['lv_best'] for g in G.values()):.2f}",
              g_ro=f"{ro['gsv_within']:.2f}", g_tw=f"{tw['gsv_within']:.2f}", g_pp=f"{pp['gsv_within']:.2f}",
              g_pr=f"{pr['gsv_within']:.2f}", dy=f"{abs(tw['peaks'][0][1] - ro['peaks'][0][1]):.0f}",
              dd=f"{tw['vars']['depth_mm']:.1f}", ib0=f"{tw['vars']['index_best'][0]:.0f}",
              ib1=f"{tw['vars']['index_best'][1]:.0f}", if0=f"{tw['vars']['index_fp'][0]:.0f}",
              if1=f"{tw['vars']['index_fp'][1]:.0f}", th=f"{100 * tw['vars']['thumb_off']:.0f}")
    cap = ("{name}, {obj}: grid of symmetric tripods every 5 mm, 16 grasps per layout, three seeds (grey: pose not "
           "reached). Top: the foundational pose's score, the best of 16 and their difference. Bottom: the held share, "
           "the feasible log volume and the smoothed best of 16 with its gradient and basin peaks (stars).")
    for task, d, name, obj in GRID_TASKS:
        t += img(MEDIA / f"20261029-grid_{task}_d{d:g}.png", cap.format(name=name, obj=obj))
    t += img(MEDIA / "20261029-grid_grasps_twist_d40.png",
             "Twist on the 40 mm sphere: the variables of each layout's best grasp (top) and of its foundational pose "
             "(bottom), over the grid.")
    t += fill(r"""<p>Figures <<f1>> and <<f2>> show the foundational pose, the best and the worst grasp of two layouts per
task at the touch pose, from the side, the front and below. In every keyframe the pads touch the object and the rigid
parts clear it. The losing grasps put the contacts high on the sphere or yaw the cylinder away from the pinch.</p>""",
              f1=FIG[0] + 1, f2=FIG[0] + 2)
    t += img(MEDIA / "20261030-grid_grasps_twist_peak_d40.png",
             "Twist on the 40 mm sphere: the foundational pose, the best and the worst grasp of the layouts at (105, 100) "
             "mm, the twist's peak, and (95, 75) mm, with their scores; touch pose, v4b fingertip.", jpeg=True)
    t += img(MEDIA / "20261030-grid_grasps_pinpush_d25.png",
             "Pin-push on the 25 mm cylinder: the foundational pose, the best and the worst grasp of the layouts at "
             "(90, 50) mm, the pin-push's peak on the grid's edge, and (90, 100) mm.", jpeg=True)
    VR = N["valrep"]
    if VR:
        names = {"twist": "Twist", "roll": "Roll", "pinpush": "Pin-push", "press": "Press"}
        tw, ro, pp, pr = (VR.get(f"{k}_d{d:g}") for k, d in (("twist", 40), ("roll", 40), ("pinpush", 25), ("press", 25)))
        sd = N["valrep_seeds"]
        f2 = lambda x: f"{x:.2f}"
        rng_ = lambda v: max(v.values()) - min(v.values())
        t += fill(r"""<p>The best of 16 is picked on seeds 0 to 2, so its score there is biased upward. Each layout's
foundational pose and best grasp therefore ran again on seeds 3 to 5 (Table <<tb>>). On the spheres the gain repeats.
The best grasp beat the foundational pose on <<tb0>> % of the layouts in the twist and on <<rb0>> % in the roll. Its
median gain was <<tg>> in the twist and <<rg>> in the roll, against <<tg0>> and <<rg0>> on the seeds it was picked on.
Its fresh scores rank the layouts against the first seeds' scores at Spearman <<tr>>.</p>
<p>On the 25 mm cylinder most of the gain came from picking on three seeds. The median gain fell to <<pg>> in the pin-push
and <<qg>> in the press, from <<pg0>> and <<qg0>>, and the fresh scores rank the layouts at Spearman <<pr_>> and
<<qr>>. A seed sets the object's initial offset, up to 1 mm, and its yaw, up to 2 deg; every layout gets the same draw,
and seed 0 has none. The cylinder tasks depend on that draw. The foundational pose's mean pin-push score over the grid
was <<s0>>, <<s1>> and <<s2>> on seeds 0, 1 and 2, and <<s3>> to <<s5>> on seeds 3 to 5. The press scored <<q2>> on
seed 2. The seeds' means differ by at most <<ts>> in the twist and <<rs_>> in the roll.<<big>></p>""",
                  big=(fill(r""" On the <<d>> mm cylinder, which every seed keeps in the hand, the gains hold better:
<<p1>> and <<q1>> against <<p0>> and <<q0>>, at Spearman <<pr1>> and <<qr1>>.""", d="40",
                            p1=f2(VR["pinpush_d40"]["gain_fresh_median"]), q1=f2(VR["press_d40"]["gain_fresh_median"]),
                            p0=f2(VR["pinpush_d40"]["gain_seeds012_median"]),
                            q0=f2(VR["press_d40"]["gain_seeds012_median"]),
                            pr1=f2(VR["pinpush_d40"]["best_repeat_spearman"]),
                            qr1=f2(VR["press_d40"]["best_repeat_spearman"]))
                       if "pinpush_d40" in VR and "press_d40" in VR else ""),
                  tb=TAB[0] + 1, tb0=f"{100 * tw['best_beats_fp']:.0f}", rb0=f"{100 * ro['best_beats_fp']:.0f}",
                  tg=f2(tw["gain_fresh_median"]), rg=f2(ro["gain_fresh_median"]), tg0=f2(tw["gain_seeds012_median"]),
                  rg0=f2(ro["gain_seeds012_median"]), tr=(f"{tw['best_repeat_spearman']:.2f} in both" if f2(tw["best_repeat_spearman"]) == f2(ro["best_repeat_spearman"])
                      else f"{tw['best_repeat_spearman']:.2f} and {ro['best_repeat_spearman']:.2f}"), 
                  pg=f2(pp["gain_fresh_median"]), qg=f2(pr["gain_fresh_median"]), pg0=f2(pp["gain_seeds012_median"]),
                  qg0=f2(pr["gain_seeds012_median"]), pr_=f2(pp["best_repeat_spearman"]),
                  qr=f2(pr["best_repeat_spearman"]), s0=f2(sd["pinpush"]["0"]), s1=f2(sd["pinpush"]["1"]),
                  s2=f2(sd["pinpush"]["2"]), s3=f2(min(sd["pinpush"][k] for k in "345")),
                  s5=f2(max(sd["pinpush"][k] for k in "345")), q2=f2(sd["press"]["2"]), ts=f2(rng_(sd["twist"])),
                  rs_=f2(rng_(sd["roll"])))
        D4 = [tw, ro, pp, pr]
        span4 = lambda f, scale=100.0, fmt="{:.0f}": (fmt.format(min(scale * f(r) for r in D4)) + " to "
                                                      + fmt.format(max(scale * f(r) for r in D4)))
        t += fill(r"""<p>Grasps were also moved between layouts as palm-frame keyframes. Of the best grasps of each layout's
four grid neighbours, <<nf>> % were feasible on the layout. On fresh seeds <<nw>> % of the feasible ones scored within
0.1 of the layout's own best grasp, at a median change of <<nc>>. The best grasp of each task's peak layout was feasible
on <<kf>> % of the other layouts.</p>""",
                  nf=span4(lambda r: r["feas"]["nb"] / r["feas"]["nb_tried"]), nw=span4(lambda r: r["nb_within_0_1"]),
                  nc=span4(lambda r: r["nb_change_median"], 1.0, "{:+.2f}"),
                  kf=span4(lambda r: r["feas"]["key"] / r["feas"]["layouts"]))
        order = {k: i for i, k in enumerate(names)}
        rows = []
        for k in sorted(VR, key=lambda k: (order[k.split("_d")[0]], float(k.split("_d")[1]))):
            r, task = VR[k], k.split("_d")[0]
            rows.append([f"{names[task]}, {L_SHAPE[task]} {k.split('_d')[1]} mm", str(r["layouts"]),
                         f"{100 * r['best_beats_fp']:.0f}", f"{r['gain_fresh_median']:+.2f}",
                         f"{r['gain_seeds012_median']:+.2f}", f"{r['best_repeat_spearman']:.2f}",
                         f"{100 * r['feas']['nb'] / r['feas']['nb_tried']:.0f}", f"{100 * r['nb_within_0_1']:.0f}",
                         f"{100 * r['feas']['key'] / r['feas']['layouts']:.0f}",
                         f"{r['key_change_median']:+.2f}".replace("-0.00", "0.00").replace("+0.00", "0.00")])
        t += table(["Task and object", "Layouts", "Best beats FP, seeds 3-5 (%)", "Median gain, seeds 3-5",
                    "Median gain, seeds 0-2", "Spearman of the best, seeds 0-2 and 3-5", "Neighbours' grasps feasible (%)",
                    "Within 0.1 of own best (%)", "Peak layout's grasp feasible (%)", "Peak layout's grasp, median change"],
                   rows, "The grid's best grasps on fresh seeds and moved between layouts. Gains are over the "
                   "foundational pose; changes are against the layout's own best grasp on seeds 3 to 5.", num_from=1)
    return t


def gradient_numbers(task="twist", d=40.0):
    """The value model, its gradient check on the grid, the trust-region moves and the peak widths (hom_landscape.py
    value, moves, moves-report), and the median nearest-neighbour spacing of the Sobol layouts."""
    import hom_landscape as L
    from scipy.spatial import cKDTree
    v = json.loads((L.SIZE_ROWS / f"value_{task}_d{d:g}.json").read_text())
    pk = json.loads((ROWS / "peak" / f"summary_{task}{d:g}.json").read_text())
    _d, z = L.layouts()
    nn = float(np.median(cKDTree(z).query(z, k=2)[0][:, 1]))
    more = {}                                          # the other tasks and sizes whose experiments ran
    for t_, d_ in (("twist", 40.0), ("roll", 40.0), ("pinpush", 25.0), ("press", 25.0)):
        pv, pp_ = L.SIZE_ROWS / f"value_{t_}_d{d_:g}.json", ROWS / "peak" / f"summary_{t_}{d_:g}.json"
        if pv.exists() and pp_.exists():
            more[(t_, d_)] = (json.loads(pv.read_text()), json.loads(pp_.read_text()))
    return {"v": v, "pk": pk, "nn": nn, "more": more}


def sec_gradients(N):
    import fp_sr2_toy as FP
    g = N["grad"]
    v, pk = g["v"], g["pk"]
    tr, pe = pk["trust"], pk["peak"]["radii"]
    span = FP.HI6 - FP.LO6
    t = fill(r"""<p>Bohlinger and Peters [5] follow the gradient of a learned critic with respect to the design. Without
RL training the same gradient comes from a smooth model of the HOM scores over the six layout coordinates.
<code>hom_landscape.py value</code> fits kernel ridge regression with a Gaussian kernel to the twist scores of the
8,192 Sobol layouts on the 40 mm sphere. The scores are the foundational pose's over three seeds, 0 where the pose is not
reached. By
five-fold cross-validation it predicts held-out layouts at R<sup>2</sup> <<r2>> and Spearman <<cvs>>, with a length
scale of <<ls>> of each gantry's travel (<<ls0>> to <<ls1>> mm). On the 506 layouts of the grid, which it never saw,
its predictions rank the measured scores at Spearman <<gs>>.</p>
<p>At the scale of the grid the gradient points uphill (Figure <<fa>>a). From each grid layout, a 10 mm step along the
gradient, projected onto the grid's plane, raised the measured score on <<up>> % of <<ns>> steps (median change
<<um>>). A step of the same length in a random direction raised it on <<ru>> % (median <<rm>>).</p>
<p>Near the best layouts the gradient fails. The best 200 Sobol layouts were moved along it inside a trust region of 0.1
of each travel, by a median of <<m0>> to <<m1>> mm per coordinate. The moved layouts and their starts then ran again on
fresh seeds (Figure <<fa>>b). The moves lowered the score in <<wo>> % of the <<np>> pairs that reached their pose (median
change <<cm>>, Wilcoxon p = <<wp>>). The model predicted a median gain of <<pg>>, and predicted and measured changes
correlate at <<pr>>. The starts themselves fell from <<s0>> on the seeds they were chosen on to <<s1>> on fresh
ones.</p>
<p>The best layouts sit on peaks a few millimetres wide (Figure <<fa>>c). Each of the <<nc>> best starts was perturbed in
ten random directions by 2, 4 and 8 mm of the six mount coordinates, its foundational pose solved again, and run on three
more seeds. At 2 mm, <<l2>> % of the perturbed layouts lost the pose and the median score fell by <<c2>>. At 4 mm the
figures were <<l4>> % and <<c4>>, and at 8 mm <<l8>> % and <<c8>>. Neighbouring Sobol layouts lie a median <<nn>> of each
travel apart, <<nn0>> to <<nn1>> mm, so a model fitted to them cannot resolve such peaks. Refining a layout therefore
needs rollouts in its own neighbourhood, at steps of 1 to 2 mm.</p>""",
             r2=f"{v['cv_r2']:.2f}", cvs=f"{v['cv_spearman']:.2f}", ls=f"{v['ls']:g}", ls0=f"{v['ls'] * span.min():.0f}",
             ls1=f"{v['ls'] * span.max():.0f}", gs=f"{v['grid_check']['spearman']:.2f}", fa=FIG[0] + 1,
             up=f"{100 * v['grid_steps']['gradient']['raised']:.0f}", ns=v["grid_steps"]["gradient"]["steps"],
             um=f"{v['grid_steps']['gradient']['median_change']:+.3f}",
             ru=f"{100 * v['grid_steps']['random']['raised']:.0f}", rm=f"{v['grid_steps']['random']['median_change']:+.3f}".replace("+0.000", "0"),
             m0=f"{min(v['trust']['moves_mm']):.1f}", m1=f"{max(v['trust']['moves_mm']):.1f}",
             wo=f"{100 * tr['worse']:.0f}", np=tr["pairs"], cm=f"{tr['change_median']:+.3f}",
             wp=f"{tr['wilcoxon_p']:.0e}", pg=f"{tr['pred_change_median']:+.2f}", pr=f"{tr['pred_vs_measured_r']:.2f}",
             s0=f"{tr['starts_seeds012_median']:.2f}", s1=f"{tr['start_median']:.2f}", nc=pk["peak"]["centres"],
             l2=f"{100 * pe['2']['pose_lost']:.0f}", c2=f"{-pe['2']['change_median']:.2f}",
             l4=f"{100 * pe['4']['pose_lost']:.0f}", c4=f"{-pe['4']['change_median']:.2f}",
             l8=f"{100 * pe['8']['pose_lost']:.0f}", c8=f"{-pe['8']['change_median']:.2f}",
             nn=f"{g['nn']:.2f}", nn0=f"{g['nn'] * span.min():.0f}", nn1=f"{g['nn'] * span.max():.0f}")
    names = {"twist": "Twist", "roll": "Roll", "pinpush": "Pin-push", "press": "Press"}
    rows = []
    for (t_, d_), (vv, kk) in g["more"].items():
        gs_, tr_, r4 = vv.get("grid_steps") or {}, kk["trust"], kk["peak"]["radii"]["4"]
        rows.append([f"{names[t_]}, {d_:g} mm", f"{vv['cv_spearman']:.2f}",
                     f"{vv['grid_check']['spearman']:.2f}" if vv.get("grid_check") else "",
                     f"{100 * gs_['gradient']['raised']:.0f} ({100 * gs_['random']['raised']:.0f})" if gs_ else "",
                     f"{100 * tr_['worse']:.0f}", f"{tr_['change_median']:+.3f}", f"{tr_['pred_change_median']:+.2f}",
                     f"{100 * r4['pose_lost']:.0f}", f"{r4['change_median']:+.2f}"])
    if len(rows) > 1:
        M_ = g["more"]
        sph = [k for k in M_ if k[0] in ("twist", "roll")]
        cyl = [k for k in M_ if k[0] in ("pinpush", "press")]

        def rng_(keys, f, scale=100.0, fmt="{:.0f}"):
            v = [f(*M_[k]) for k in keys]
            return fmt.format(scale * min(v)) + " to " + fmt.format(scale * max(v))
        t += fill(r"""<p>Table <<tb>> repeats the tests for each task where they ran. On the sphere tasks the gradient
raised more grid layouts than random steps did; on the cylinder tasks it raised <<cg>> % against <<cr>> % at random. On
every task, moving the best layouts along the gradient lowered their fresh-seed scores more often than it raised them,
in <<w>> % of pairs. The predicted gains did not correlate with the measured changes (<<r>>). The cylinder tasks'
peaks are wider: a 4 mm perturbation costs a median of <<c4>> there, against <<s4>> on the spheres.</p>""",
                  tb=TAB[0] + 1,
                  cg=rng_(cyl, lambda v, k: v["grid_steps"]["gradient"]["raised"]),
                  cr=rng_(cyl, lambda v, k: v["grid_steps"]["random"]["raised"]),
                  w=rng_(list(M_), lambda v, k: k["trust"]["worse"]),
                  r=rng_(list(M_), lambda v, k: k["trust"]["pred_vs_measured_r"], 1.0, "{:.2f}"),
                  c4=rng_(cyl, lambda v, k: -k["peak"]["radii"]["4"]["change_median"], 1.0, "{:.2f}"),
                  s4=rng_(sph, lambda v, k: -k["peak"]["radii"]["4"]["change_median"], 1.0, "{:.2f}")) if cyl and sph else \
            fill(r"""<p>Table <<tb>> repeats the tests for each task where they ran.</p>""", tb=TAB[0] + 1)
        t += table(["Task", "Spearman, cross-validated", "Spearman, grid", "Grid steps raised, gradient (random) (%)",
                    "Moved best layouts worse (%)", "Measured change", "Predicted change", "4 mm: pose lost (%)",
                    "4 mm: median change"], rows,
                   "The value model and its gradient per task: cross-validated and grid rank correlations, 10 mm steps on "
                   "the grid, the trust-region moves of the best 200 Sobol layouts on fresh seeds (median changes), and 4 mm "
                   "perturbations around the 20 best layouts.", num_from=1)
    t += img(MEDIA / "20261030-value_twist_d40.png",
             "Design gradients on the twist of the 40 mm sphere. (a) Steps of 10 mm on the measured grid along the value "
             "model's gradient and in random directions: the share that raised and lowered the score. (b) The best Sobol "
             "layouts moved inside the trust region, on fresh seeds: measured against predicted change. (c) Score change "
             "around the best layouts by perturbation radius (median and quartiles; a pose not reached scores 0), with "
             "the share of perturbations that lose the foundational pose.")
    return t


def size_numbers():
    """The size study per task whose sizes are all complete (every reached layout run with three seeds): reached
    layouts, Spearman between sizes, the variance shares (hom_landscape.py stats), and the best tenth's median mount
    coordinates in mm per size."""
    import fp_sr2_toy as FP
    import hom_landscape as L
    d, z = L.layouts()
    n = len(d)
    land = json.loads((L.SIZE_ROWS / "landscape.json").read_text())
    out = {}
    for task in L.TASKS:
        per = {}
        for size in L.SIZES[task]:
            sc = L.scores(task, size, n)
            R = L.reach(task, size)
            if not R or sc["run"].sum() < len(R):
                per = None
                break
            top = np.argsort(-sc["mean"])[:int(L.TOP * n)]
            ls_ = land["tasks"][task]["sizes"].get(f"{size:g}", {})
            per[size] = {"reached": len(R), "median": float(np.median(sc["mean"][sorted(R)])),
                         "held_all": int(sc["held_all"][sorted(R)].sum()),
                         "top_mm": np.median(d[top], 0), "nbh": ls_.get("n_neighbourhoods"),
                         "nbh_null": ls_.get("n_neighbourhoods_null"), "nbh_size": ls_.get("neighbourhood_size_median")}
        if per:
            T_ = land["tasks"][task]
            out[task] = {"sizes": per, "pairs": {k: v["spearman"] for k, v in T_["pairs"].items()},
                         "kept": {k: v["top_kept"] for k, v in T_["pairs"].items()}, "variance": T_["variance"]}
    return out


def sec_sizes(N):
    Z = N["sizes"]
    names = {"twist": "twist", "roll": "roll", "pinpush": "pin-push", "press": "press"}
    rows = []
    for task, v in Z.items():
        szs = list(v["sizes"])
        adj = [v["pairs"][f"{a:g}-{b:g}"] for a, b in zip(szs[:-1], szs[1:])]
        rows.append([names[task].capitalize(), ", ".join(f"{s:g}" for s in szs), f"{min(adj):.2f} to {max(adj):.2f}",
                     f"{v['pairs'][f'{szs[0]:g}-{szs[-1]:g}']:.2f}", f"{100 * v['variance']['layout']:.0f}",
                     f"{100 * v['variance']['size']:.0f}", f"{100 * v['variance']['interaction']:.0f}",
                     f"{100 * v['variance']['interaction_noise']:.0f}"])
    obj = {"twist": "spheres of 30, 40, 50 and 63 mm", "roll": "spheres of 30, 40, 50 and 63 mm",
           "pinpush": "cylinders of 10, 16, 25 and 40 mm", "press": "cylinders of 10, 16, 25 and 40 mm"}
    groups = {}
    for k in Z:
        groups.setdefault(obj[k], []).append(f"the {names[k]}")
    ran = " ".join(" and ".join(v).capitalize() + f" ran on {o}." for o, v in groups.items())
    t = fill(r"""<p>Every Sobol layout ran each task on four object sizes with the v4b fingertip, with foundational poses
and three seeds (<code>rows/sizes_v4b</code>). <<ran>> Table <<tb>> splits the variance of the layouts' scores over the sizes
into a part from the layout, a part from the size and their interaction. The interaction is the part of a layout's score
that depends on the object. Seed noise alone would give the interaction in the last column.</p>""", ran=ran,
             tb=TAB[0] + 1)
    t += table(["Task", "Diameters (mm)", "Spearman, neighbouring sizes", "Spearman, smallest and largest",
                "Layout (%)", "Size (%)", "Interaction (%)", "Interaction from seed noise (%)"], rows,
               "How the layouts' scores change with the object's size, over the 8,192 Sobol layouts (a layout that does "
               "not reach the pose scores 0).", num_from=2)
    if "roll" in Z:
        rs = Z["roll"]["sizes"]
        roll_kept = Z["roll"]["kept"][f"{min(rs):g}-{max(rs):g}"]
    if "twist" in Z:
        tw = Z["twist"]["sizes"]
        s0, s1 = min(tw), max(tw)
        xs0 = tw[s0]["top_mm"][2] - tw[s0]["top_mm"][0]
        xs1 = tw[s1]["top_mm"][2] - tw[s1]["top_mm"][0]
        t += fill(r"""<p>The object's size changes which layouts are best, and it does so by more than seed noise in
every task listed. In the twist the interaction is <<ti>> % of the variance against <<tn>> % from seed noise. The
ranking of layouts on the <<s0>> mm sphere keeps only Spearman <<se>> on the <<s1>> mm sphere. The best tenth of twist
layouts spreads its thumb and index mounts as the sphere grows. Their median distance along palm $x$ is
<<x0>> mm at <<s0>> mm and <<x1>> mm at <<s1>> mm (Figure <<f>>). Of the best tenth of twist layouts on the
<<s0>> mm sphere, <<k0>> % are in the best tenth on the <<s1>> mm sphere; a random tenth would give 10 %. Between
neighbouring sizes <<kn0>> to <<kn1>> % are.<<roll>></p>""",
                  ti=f"{100 * Z['twist']['variance']['interaction']:.0f}",
                  tn=f"{100 * Z['twist']['variance']['interaction_noise']:.0f}", s0=f"{s0:g}", s1=f"{s1:g}",
                  se=f"{Z['twist']['pairs'][f'{s0:g}-{s1:g}']:.2f}", x0=f"{xs0:.0f}", x1=f"{xs1:.0f}", f=FIG[0] + 1,
                  k0=f"{100 * Z['twist']['kept'][f'{s0:g}-{s1:g}']:.0f}",
                  kn0=f"{100 * min(Z['twist']['kept'][f'{a_:g}-{b_:g}'] for a_, b_ in zip(sorted(tw)[:-1], sorted(tw)[1:])):.0f}",
                  kn1=f"{100 * max(Z['twist']['kept'][f'{a_:g}-{b_:g}'] for a_, b_ in zip(sorted(tw)[:-1], sorted(tw)[1:])):.0f}",
                  roll=(fill(r""" The roll's ranking changes less with size: its interaction is <<ri>> % of the
variance, and <<rk>> % of its best tenth on the smallest sphere stay in the best tenth on the largest. Its scores rise
with the sphere's size, from a median of <<r0>> over the reached layouts at <<s0>> mm to <<r1>> at <<s1>> mm.""",
                              ri=f"{100 * Z['roll']['variance']['interaction']:.0f}",
                              rk=f"{100 * roll_kept:.0f}",
                              r0=f"{Z['roll']['sizes'][min(Z['roll']['sizes'])]['median']:.2f}",
                              r1=f"{Z['roll']['sizes'][max(Z['roll']['sizes'])]['median']:.2f}",
                              s0=f"{min(Z['roll']['sizes']):g}", s1=f"{max(Z['roll']['sizes']):g}") if "roll" in Z else ""))
    if "pinpush" in Z and "press" in Z:
        P_, R_ = Z["pinpush"], Z["press"]
        a_, b_ = min(P_["sizes"]), max(P_["sizes"])
        key = f"{a_:g}-{b_:g}"
        t += fill(r"""<p>On the cylinders the interaction is <<pi>> % of the pin-push's variance and <<ri>> % of the press's,
against <<pn>> and <<rn>> % from seed noise. Between the <<a>> and <<b>> mm cylinders the layouts' scores correlate at
Spearman <<ps>> in the pin-push and <<rs>> in the press. Of the best tenth on the <<a>> mm cylinder, <<pk>> and <<rk>> %
are in the best tenth on the <<b>> mm cylinder. Small cylinders are hard to keep in the hand. With the foundational
pose, <<h0>> of the <<n0>> layouts that reach the pin-push pose keep the <<a>> mm cylinder in all three seeds, against
<<h1>> of <<n1>> with the <<b>> mm cylinder.</p>""",
                  pi=f"{100 * P_['variance']['interaction']:.0f}", ri=f"{100 * R_['variance']['interaction']:.0f}",
                  pn=f"{100 * P_['variance']['interaction_noise']:.0f}",
                  rn=f"{100 * R_['variance']['interaction_noise']:.0f}", a=f"{a_:g}", b=f"{b_:g}",
                  ps=f"{P_['pairs'][key]:.2f}", rs=f"{R_['pairs'][key]:.2f}", pk=f"{100 * P_['kept'][key]:.0f}",
                  rk=f"{100 * R_['kept'][key]:.0f}", h0=f"{P_['sizes'][a_]['held_all']:,}",
                  n0=f"{P_['sizes'][a_]['reached']:,}", h1=f"{P_['sizes'][b_]['held_all']:,}",
                  n1=f"{P_['sizes'][b_]['reached']:,}")
    nb = [(v["nbh"], v["nbh_null"], v["nbh_size"]) for T_ in Z.values() for v in T_["sizes"].values()
          if v.get("nbh_null") is not None]
    if nb:
        t += fill(r"""<p>The good layouts are grouped. At each size a task's landscape has <<n0>> to <<n1>> high
neighbourhoods. A high neighbourhood is a basin of the score smoothed over each layout's 16 nearest layouts, with its
peak in the best tenth and 20 layouts or more. They hold a median of <<m0>> to <<m1>> layouts each. The same scores shuffled among the reached
layouts scatter into <<u0>> to <<u1>> smaller ones. Each task and size therefore has a few dozen separate regions of
good layouts.</p>""", n0=min(x[0] for x in nb), n1=max(x[0] for x in nb),
                  m0=f"{min(x[2] for x in nb):.0f}", m1=f"{max(x[2] for x in nb):.0f}",
                  u0=f"{min(x[1] for x in nb):.0f}", u1=f"{max(x[1] for x in nb):.0f}")
    GS = N["grid_sizes"]
    by = {}
    for (task, sz), g in GS.items():
        by.setdefault(task, {})[sz] = g
    gp = [task for task in by if len(by[task]) >= 2 and (MEDIA / f"20261030-grid_sizes_{task}.png").exists()]
    if len(gp) == 4:
        def shift(task):
            s = by[task]
            a, b = min(s), max(s)
            return s[a]["peak"][0], s[b]["peak"][0], (s[b]["peak"][0] - s[a]["peak"][0]) / (b - a), a, b
        tw_, pp_, pr_ = shift("twist"), shift("pinpush"), shift("press")
        k_ = [tw_[2], pp_[2], pr_[2]]
        fl_r = [g["flat"] for g in by["roll"].values()]
        fl_t = [g["flat"] for g in by["twist"].values()]
        f0 = FIG[0] + 1 + len(Z)
        cyl_ = (f"The pin-push's and the press's both move from {pp_[0]:.0f} to {pp_[1]:.0f} mm"
                if pp_[:2] == pr_[:2] else f"The pin-push's moves from {pp_[0]:.0f} to {pp_[1]:.0f} mm and the press's "
                f"from {pr_[0]:.0f} to {pr_[1]:.0f} mm")
        t += fill(r"""<p>Table <<tb>> repeats the grid of Section SEC[grid] at three sizes per object (Figures <<f0>> to
<<f1>>). In the twist, the pin-push and the press, the peak of the largest basin moves to a larger $x_{sep}$ as the
object grows. The twist's moves from <<t0>> to <<t1>> mm between the <<ts0>> and <<ts1>> mm spheres. <<cyl>> between the
<<cs0>> and <<cs1>> mm cylinders. That is
<<k0>> to <<k1>> mm of $x_{sep}$ per millimetre of diameter. The thumb and the index-middle pair touch opposite sides
of the object, so their contacts move apart by the change in diameter.</p>
<p>The roll's map is flat. On it <<rf0>> to <<rf1>> % of the layouts score within 0.1 of the map's maximum, against
<<tf0>> to <<tf1>> % in the twist, so its peak has no clear position. On the grid's <<cs0>> mm cylinder the median layout keeps none of its 16
grasps in all three seeds, and on the <<cs1>> mm cylinder it keeps all of them.</p>""",
                  tb=TAB[0] + 1, f0=f0, f1=f0 + len(gp) - 1, cyl=cyl_, t0=f"{tw_[0]:.0f}", t1=f"{tw_[1]:.0f}",
                  ts0=f"{tw_[3]:g}", ts1=f"{tw_[4]:g}", cs0=f"{pp_[3]:g}", cs1=f"{pp_[4]:g}", k0=f"{min(k_):.1f}", k1=f"{max(k_):.1f}",
                  rf0=f"{100 * min(fl_r):.0f}", rf1=f"{100 * max(fl_r):.0f}", tf0=f"{100 * min(fl_t):.0f}",
                  tf1=f"{100 * max(fl_t):.0f}")
        rows = [[names[task].capitalize(), f"{L_SHAPE[task]} {sz:g} mm", str(g["run"]), f"{g['fp_median']:.2f}",
                 f"{g['best_median']:.2f}", f"{g['held_share_median']:.2f}", f"{g['peak'][0]:.0f}, {g['peak'][1]:.0f}",
                 f"{g['peak'][2]:.2f}", f"{100 * g['flat']:.0f}"] for (task, sz), g in GS.items()]
        t += table(["Task", "Object", "Layouts", "FP median", "Best-of-16 median", "Held share, median",
                    "Largest basin's peak, x_sep and y_sep (mm)", "Its smoothed score", "Within 0.1 of the maximum (%)"],
                   rows, "The grid of 625 symmetric tripods at each object size: 16 grasps per layout, seeds 0 to 2, v4b "
                   "fingertip. The peak and the last column use the best-of-16 map smoothed over 7.5 mm.", num_from=2)
    for task in Z:
        t += img(MEDIA / f"20261030-sizes_{task}.png",
                 f"The {names[task]} across object sizes on the Sobol layouts: (a) rank correlation of the layouts' scores "
                 "between sizes; (b) the median and quartiles of the six mount coordinates of the best tenth of layouts per "
                 "size (standardized: each gantry's travel maps to -0.5 to 0.5); (c) the scores of the layouts that reach "
                 "the pose.")
    for task in gp:
        t += img(MEDIA / f"20261030-grid_sizes_{task}.png",
                 f"The {names[task]} on the grid at each object size: the best grasp of 16 per layout, smoothed over "
                 "7.5 mm, with the basin peaks (stars) and the largest basin's peak (circle).")
    return t


L_SHAPE = {"twist": "sphere", "roll": "sphere", "pinpush": "cylinder", "press": "cylinder"}


def grid_size_numbers():
    """The grid at every object size whose rows exist (hom_landscape.py grid): the summary, the peak of the largest
    basin of the smoothed best-of-16 map, and the share of layouts whose smoothed score lies within 0.1 of the map's
    maximum."""
    import hom_landscape as L
    summ = json.loads((L.GRID_ROWS / "summary.json").read_text())
    out = {}
    for task in L.TASKS:
        for sz in L.SIZES[task]:
            if f"{task}_d{sz:g}" not in summ:
                continue
            xs, ys, M = L.grid_maps(task, sz)
            sm = L.smooth_map(M["best"])
            pk, big = L.peaks(sm)
            q = max(big, key=lambda p: int((pk == p).sum()))
            v = sm[np.isfinite(sm)]
            out[(task, sz)] = dict(summ[f"{task}_d{sz:g}"], flat=float(np.mean(v >= v.max() - 0.1)),
                                   peak=(float(xs[q % len(xs)]), float(ys[q // len(xs)]), float(sm.flat[q])))
    return out


def sec_cross(N):
    import hom_landscape as L
    CF = N.get("cross_fresh")
    c = json.loads((L.SIZE_ROWS / "cross.json").read_text())
    rho, b, mm = c["spearman"], c["best"], c["maximin"]
    cross_rho = [rho[k] for k in ("twist-pinpush", "twist-press", "roll-pinpush", "roll-press")]
    rk = c["best_rank_elsewhere"]
    t = fill(r"""<p>The four tasks ran on all 8,192 Sobol layouts at their default objects, with foundational poses and
three seeds. The twist and the roll used the 40 mm sphere, the pin-push and the press the 25 mm cylinder. Tasks on the same object
and grasp family rank the layouts alike: Spearman <<tr>> between the twist and the roll, <<pp>> between the pin-push and
the press. Between a sphere task and a cylinder task the rank correlation is <<c0>> to <<c1>>.</p>
<p>Each task's best layout is a poor choice for the others (Figure <<f>>). The twist's best layout, with a twist score
of <<btw>>, ranks <<rk1>> of 8,192 on the roll. The pin-push's best drops the sphere in both sphere tasks. Of the 8,192 layouts, <<npar>>
(<<ppar>> %) are Pareto-optimal on the four scores. The layout whose worst score is highest reaches <<mm>> on the
twist, roll, pin-push and press. The best tenth of each task sits in a different part of the mount space (Figure <<f>>,
right). The roll's has the thumb and the index nearer each other along palm $x$; the twist's has the index and the
middle nearer the centre line.</p>""",
             tr=f"{rho['twist-roll']:.2f}", pp=f"{rho['pinpush-press']:.2f}", c0=f"{min(cross_rho):.2f}",
             c1=f"{max(cross_rho):.2f}", f=FIG[0] + 1, btw=f"{b['twist']['scores'][0]:.2f}",
             rk1=f"{rk['twist']['roll'] + 1:,}", npar=c["pareto_n"], ppar=f"{100 * c['pareto_n'] / 8192:.1f}",
             mm=", ".join(f"{x:.2f}" for x in mm["scores"][:-1]) + f" and {mm['scores'][-1]:.2f}")
    t += img(MEDIA / "20261030-cross_tasks.png",
             "The four tasks on the 8,192 Sobol layouts at the default objects: the twist against the roll and the "
             "pin-push against the press, with the Pareto set over all four scores (red); and the median and quartiles "
             "of the six mount coordinates of each task's best tenth (standardized).")
    cg = json.loads((L.GRID_ROWS / "cross_grid.json").read_text())
    t += fill(r"""<p>Grasp synthesis changes this picture. On the <<n>> grid layouts that reach all four poses, the worst of a
layout's four foundational-pose scores has a median of <<mf>>, and no layout scores above 0.5 on all four tasks. With
each task's best grasp of 16 the median worst score is <<mb>>, and <<sb>> % of the layouts score above 0.5 on all four
(Figure <<f>>). The best of them, at $x_{sep}$ = <<bx>> mm and $y_{sep}$ = <<by>> mm, scores <<bs>>.<<fresh>></p>""",
              n=cg["layouts"], mf=f"{cg['maximin_median_fp']:.2f}",
              fresh=(fill(r""" These scores are on the seeds the grasps were picked on. On seeds 3 to 5, <<fb>> % of
the layouts score above 0.5 on all four tasks with their best grasps and <<ff>> % with their foundational poses, and
the median worst score is <<mb2>> and <<mf2>> (Section SEC[grid]).""", fb=f"{100 * CF['share_all_above_0.5_best']:.0f}",
                          ff=f"{100 * CF['share_all_above_0.5_fp']:.0f}", mb2=f"{CF['maximin_median_best']:.2f}",
                          mf2=f"{CF['maximin_median_fp']:.2f}") if CF else
                     " The best of 16 is an upper estimate until its fresh-seed runs."),
              mb=f"{cg['maximin_median_best']:.2f}", sb=f"{100 * cg['share_all_above_0.5_best']:.0f}", f=FIG[0] + 1,
              bx=f"{cg['maximin_best']['xy'][0]:.0f}", by=f"{cg['maximin_best']['xy'][1]:.0f}",
              bs=", ".join(f"{x:.2f}" for x in cg["maximin_best"]["scores"]))
    t += img(MEDIA / "20261030-cross_grid.png",
             "The worst of the four task scores on each grid layout, with the foundational pose for every task (left) and "
             "with each task's best grasp of 16 (right); circles: the Pareto set over the four scores; star: the best "
             "worst-case layout.")
    return t


def sec_related():
    return r"""<p>Four lines of work bear on this design study. Each answers part of the question, with a different
controller in the loop.</p>
<ul>
<li>The foundational-pose paper [1] scores hands by kinematic poses and a kinematic planner. Here the poses keep their
role as the starting states of the tasks, the controller is HOM [2], and the scores come from dynamic rollouts.</li>
<li>Bohlinger and Peters [5] train one morphology-conditioned policy and critic with PPO over 50 robots, in 7 to 9 h.
The frozen critic is then a differentiable model of return over up to 1,177 design parameters. Designs improve by
gradient ascent with a trust-region penalty, 1,000 designs per batch in 1 to 2 min. HOM needs no
training per design, so a smooth model fitted to this page's rollouts over the layout coordinates can play the
critic's role. Its gradient is the design gradient of [5]. On the 2-D grid the smoothed maps' arrows are
that gradient.</li>
<li>Yang et al. [6, 7] evolve hand morphologies with an elitist, mutation-only evolutionary algorithm. One
morphology-aware Transformer policy, trained with PPO, controls the whole population: 22,528 hands, scored by the grasp
matrix's smallest singular value and by successful reorientations. Their fitness and their outer loop can be run on this page's
evaluator. Their grasp metric is recorded for every grasp in Section SEC[grid], where it ranked only the roll's grasps.
The maps of Section SEC[grid] show every basin of a task over the grid.</li>
<li>In Xie, Xiang, Posa and Jin [8] a policy chooses where to touch: a contact point on the object and an object
subgoal. A complementarity-free contact-implicit MPC at about 100 Hz then moves a Franka arm's stick through contact.
The actions are written at the contacts, as HOM's commands are. An MPC over HOM's commands for the SR2 hand is listed
in Section SEC[future].</li>
</ul>"""


def sec_history():
    rows = [
        ["2026-10-29, afternoon", "The paper's pipeline on the SR2 toy: RRT and Sobol layouts, SQP onto two poses, Pareto "
         "front, t-SNE and PCA", "reproduce [1] on our hand", "the t-SNE islands come from the RRT; Sobol reaches the "
         "poses on most layouts (Section SEC[toy])"],
        ["2026-10-29, evening", "HOM on the GPU on every layout's foundational pose; four tasks, sensing variants",
         "score layouts by the controller that runs on the bench", "kinematic scores do not rank the rollouts "
         "(Sections SEC[gpu] to SEC[sense])"],
        ["2026-10-29, 23:00", "Grasps synthesized per layout instead of one keyframe or one pose per layout; the 2-D "
         "grid of symmetric tripods before the six coordinates", "the in-hand task depends on the grasp it starts "
         "from; two parameters give readable maps", "Section SEC[grid]"],
        ["2026-10-29, 23:10", "Width scans and a coordinate walk instead of uniform perturbations", "uniform "
         "perturbations of 35 % of the bounds were all infeasible: the feasible set is thin", "16 grasps on every "
         "reached layout"],
        ["2026-10-29, 23:20", "The v4b fingertip instead of the TPU block, with sphere packs of the rigid parts, the "
         "trimmed cap and two pose constraints", "the fingertip calibrated on the UR5 is the one the controller must "
         "match", "every pose and rollout rerun (Section SEC[v4b])"],
        ["2026-10-30, 00:00 to 02:00", "Checks of the v4b hand: touch keyframes from three sides, rigid-part forces, CPU "
         "against GPU; faster GPU geometry for the pad", "caps inside the sphere before the constraints; a batch ten "
         "times slower with the rigid spheres colliding with each other", "36 to 41 ms per rollout"],
        ["2026-10-30, 02:00", "Feasible log volume and the grasp-matrix value as predictors", "a layout's room for "
         "grasps as a sample-efficiency score", "log volume ranks the best score at Spearman 0.39 to 0.68"],
        ["2026-10-30, 02:30 to 05:30", "A smooth model of the scores over the six coordinates and its gradient, in place "
         "of a learned critic's [5]; moves of the best layouts on fresh seeds; perturbations of 2 to 8 mm", "design "
         "gradients without RL", "the gradient climbs at the grid's scale but lowers the best layouts; their peaks are "
         "a few millimetres wide (Section SEC[gradients])"],
        ["2026-10-30, 06:50 to 07:45", "The grid at three sizes per object, beside the Sobol size sweep", "whether the "
         "best layouts move with the object, with grasp synthesis", "the peak's thumb-to-pair distance grows 1.0 to "
         "1.2 mm per millimetre of diameter in three tasks (Section SEC[sizes])"],
        ["2026-10-30, 08:05", "Each layout's best grasp and foundational pose again on seeds 3 to 5; neighbours' best "
         "grasps moved to each layout", "the best of 16 is picked on seeds 0 to 2", "the spheres keep the gain; on the "
         "cylinders most of it came from three object-pose draws shared by every layout (Section SEC[grid])"],
    ]
    return ("<p>Table SEC_TAB_H lists the changes to the method in order, with their reasons and what they showed."
            "</p>").replace("SEC_TAB_H", str(TAB[0] + 1)) + table(
        ["When", "Change", "Reason", "Result"], rows, "Changes to the method on 2026-10-29 and 10-30.", num_from=9)


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
for a good dynamic score; it can only rank. On the twist, the rotation range ranked the layouts weakly the wrong
way round (Spearman <<s1>> over the 2-D case, Section SEC[check]). The dynamic rollout therefore has to stay in the loop, and at
about <<ms>> ms per run on the GPU it can.</p>"""
    d2 = N.get("dyn2d") or {}
    ms = (N.get("val", {}).get("twist") or {}).get("ms")
    return fill(t, s1=f"{d2.get('sp_kin_ramp', float('nan')):.2f}", ms=f"{ms:.0f}" if ms else "?")


def gpu_cost(names):
    """Rollouts and GPU minutes (build, run and scoring) over the timing rows of the given row files."""
    tim = [t_ for n in names for t_ in load(f"{n}_timing")]
    return sum(t_["worlds"] for t_ in tim), sum(t_["build_s"] + t_["run_s"] + t_.get("score_s", 0.0) for t_ in tim) / 60


def press_latitudes():
    """The share of press poses whose middle finger sits at the 30 deg bound, over the three populations."""
    v = [r["fp"]["press"]["v"][7] for n in ("rrt2d_press", "rrt6d_press", "sobol8k_press") for r in load(n)
         if r["fp"]["press"].get("ok")]
    return len(v), float(np.mean(np.degrees(v) < 30.5)) if v else float("nan")


def sec_plan(N):
    r6 = N["rrt6d"]
    pc = r6.get("seconds", 582.0) / max(r6.get("calls", 1), 1)
    n3, m3 = gpu_cost([f"hom6d_gpu_{t}" for t in ("twist", "roll", "pinpush", "press")])
    n4, m4 = gpu_cost([f"hom6d_gpu_sense_{t}" for t in ("twist", "roll", "pinpush", "press")])
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
        ["3. HOM rollouts", "the four tasks on the GPU evaluator, three seeds (run on this page; a second drive "
         "force would double it)", "the Sobol layouts that reach every pose (4,680 of 8,192; 4,358 reach the press "
         "pose)", f"measured: {n3:,} rollouts in {m3:.0f} min of GPU", "scores that do not repeat across seeds: rank "
         "correlation between seed halves below 0.8"],
        ["4. Sensing", "the four tasks with calibration errors, without fingertips, with them and with their force "
         "loop, at 50 Hz (run on this page, Section SEC[sense])", "the same layouts",
         f"measured: {n4:,} rollouts in {m4:.0f} min of GPU", "the Pareto set changes when sensing is added"],
        ["5. Map", "Pareto front over the dynamic scores; t-SNE and PCA of the Sobol-sampled designs, beside a "
         "uniform null", "the scores of stages 3-4", "minutes", "clusters that the uniform null also shows"],
        ["6. Bench", "three to five layouts from different clusters and the front, set on the gantries; the pin-push "
         "and twist with tags and, once mounted, fingertips", "the bench hand", "one or two days", "bench ranking "
         "against simulated ranking, Spearman below 0.5"],
    ]
    t = r"""<p>The paper's dense front and t-SNE map can be rebuilt for the SR2 hand at moderate cost, because the
expensive part, the controller, is fixed and runs in tens of milliseconds on the GPU. Table SEC_TAB_PLAN lists the
stages. Each has its input, its cost on this workstation and the result that would stop the plan. Stages 1 to 5 ran on
this page with the toy's poses. Stage 0 still lacks the screwdriver's poses, and stage 6 needs the bench.</p>"""
    t = t.replace("SEC_TAB_PLAN", str(TAB[0] + 1))
    t += table(["Stage", "Work", "Input", "Cost", "Stops the plan"], rows,
               "A plan for a Pareto front and t-SNE map of SR2 layouts under HOM control. Costs from this page's timings "
               "(fp_sr2_toy.py on the CPU, hom_gpu.py on the GPU).", num_from=9)
    t += r"""<p>Three choices differ from the paper and follow from the toy. Sobol replaces the RRT, since on our
gantry box most layouts reach the poses and the RRT's advantage disappears (Section SEC[toy]). A Sobol sample is also the
unbiased population that a t-SNE map needs; an RRT's branches print their own structure. The scores come from dynamic rollouts of the controller that will run on the bench. The
kinematic stage only gates reachability, since its rotation range did not rank the twist (Section SEC[check]). Finally,
sensing enters as a second run of every layout under calibration errors (Section SEC[sense]).</p>"""
    return t


def sec_limits(N):
    V = N.get("val", {})
    ag = min((V.get(k) or {}).get("agree", float("nan")) for k in ("twist", "roll"))
    return fill("""<ul>
<li>The toy's two poses are grasps of two HOM tasks. The screwdriver's load and wield, the tasks of both papers, are
not posed yet.</li>
<li>The GPU physics uses float32, a looser solver tolerance and pads that meet only the object. Against the CPU runner it
gives the same held outcome in <<ag>> % or more of the 2-D sphere rollouts and in 90 % or more of the cylinder
rollouts. A single rollout's score can differ, most in the cylinder tasks.</li>
<li>The pin-push and the press are seed-sensitive on the toy's poses: two seeds of one runner disagree on the held
outcome on about a quarter of the 2-D layouts. Three seeds per layout rank single layouts roughly, and the four-task
front inherits that noise.</li>
<li>The press pose puts the middle finger at the 30 deg bound, the latitude nearest the pinch, on <<pl>> % of the
<<pn>> layouts that reach it. At steeper latitudes more of the middle finger's force lifts the cylinder; those were
not run.</li>
<li>The layout space is the gantries' six coordinates. The paper's template also varies link lengths and the thumb's
rotation, which the SR2 hand cannot change without new parts.</li>
<<bestlim>>
<li>The cylinder tasks keep few grasps held in every seed (median held share <<hp>> in the pin-push and <<hr>> in the
press), so their maps rank single layouts roughly. The pin-push's peak lies on the grid's edge of the smallest
index-to-middle spacing, 50 mm; smaller spacings are outside the grid.</li>
<li>In the pin-push the middle finger presses with less than its commanded 1.5 N. In the code walkthrough's example it
pressed a median 0.27 N, while its joint reference stayed more than the 6 mm gap cut behind the cylinder. The pin-push
scores include that.</li>
</ul>""", hp=f"{N['grid']['pinpush']['held_share_median']:.2f}", hr=f"{N['grid']['press']['held_share_median']:.2f}",
        bestlim=("<li>Grasps were moved only between grid neighbours 5 mm apart and from each task's peak layout at one "
                 "object size; moves across object sizes were not run.</li>" if N.get("valrep") else
                 "<li>The best of 16 is the largest of 16 grasp scores, each a mean over three seeds, so it is biased "
                 "upward. The gain over the foundational pose has not been repeated on fresh seeds.</li>"),
        ag=f"{100 * ag:.1f}", pn=f"{press_latitudes()[0]:,}", pl=f"{100 * press_latitudes()[1]:.0f}")


def sec_future(N):
    done = bool(N.get("valrep"))
    return """<ol>
<li>Rerun the four-task front's layouts with 30 seeds per task on the GPU (<code>scripts/hom_gpu.py run</code> on a
row file of the front). Keep the layouts whose rank agrees between seed halves at Spearman above 0.8. The front is too
noisy to act on if fewer than half keep their rank.</li>
<li>Run the press from latitudes of 37.5 and 45 deg (the toy's <code>press</code> command with the middle's latitude
fixed) on the Sobol layouts. The 30 deg pose is the right choice if neither raises the share of layouts held in every
seed.</li>
<li>Find why the roll from the foundational poses tracks worse with the exact object pose than with the tracker's
noise and delay (Section SEC[sense]). Trace one layout's commanded and achieved twist at 50 Hz under exact sensing and
the <code>notip</code> variant, then rerun it with the delay alone and the noise alone. The account holds if one of the
two alone gives the gain.</li>
<li>Set three layouts from different parts of the four-task front on the gantries and run the pin-push and the twist
with tags. The simulated ranking fails if the bench ranks the three differently.</li>
<li>Pose the screwdriver's pinch brake and wield as foundational poses in <code>fp_sr2_toy.py</code>.</li>
<li>Compare the t-SNE map of the Sobol layouts' dynamic scores against a uniform null with the neighbour-preservation
test of <code>scripts/paper_morphology_diversity.py</code>.</li>
""" + ("""<li>Move each grid layout's best grasps across object sizes, the 40 mm sphere's to the 30 and 63 mm
spheres and the 25 mm cylinder's to the 10 and 40 mm cylinders. Keep each grasp's variables, check it with
<code>hom_landscape.feasible</code> after <code>fp_sr2_toy.set_sizes</code>, and run the feasible ones with
<code>hom_gpu.py run</code> on seeds 3 to 5. A grasp set transfers across sizes if most moved grasps score within 0.1 of
the size's own best.</li>""" if done else """<li>Rerun each grid layout's foundational
pose and best grasp on seeds 3 to 5. Add the best grasps of its four neighbours, moved to it as palm-frame keyframes
(<code>logs/20261029-hom_sizes/gpu_queue_v4b_2.sh</code>: <code>hom_landscape.py derive</code>, then <code>hom_gpu.py run
--layouts grid2dval_size grid2dnb_size</code>). The gain of Section SEC[grid] repeats if the best grasp beats the
foundational pose on fresh seeds on most layouts. A grasp transfers to a neighbouring layout if its score there comes
within 0.1 of the layout's own best.</li>""") + """
<li>Score the pin-push and the press over twelve object-pose draws: the grid's best grasps and foundational poses
(<code>--layouts grid2dval_pinpush_size grid2dval_press_size</code>) on seeds 6 to 17 with <code>hom_gpu.py
run</code>. The best grasp's gain over the foundational pose is real if its median stays above the spread of the
seeds' means; a ranking of layouts on three draws is then replaced by one on twelve.</li>
<li>Refine the best layouts by rollouts in their own neighbourhood. Around each of the 20 best twist layouts, run a
local Sobol design of 64 layouts within 2 mm, three seeds each (<code>hom_landscape.py moves</code> with a smaller
radius, then <code>fp_sr2_toy.py poses</code> and <code>hom_gpu.py run</code>). The refinement works if the best of
the 64 beats its centre on three further seeds for most of the 20.</li>
<li>Run the evolutionary search of [6] on the same evaluator as a baseline, counting rollouts to reach the best basin of
the grid.</li>
<li>Build a sampling MPC over HOM's commands on MuJoCo-Warp (<code>scripts/hom_mpc_gpu.py</code>) and compare it with
HOM's fixed schedules on the grid's best layouts. Measure its control rate against real time.</li>
</ol>"""


def sec_files():
    refs = "".join(f"<li>{r}</li>" for r in REFS)
    return f"""<p>References:</p><ul>{refs}</ul>
<p>Files:</p><ul>
<li>This page: <code>{OUT.relative_to(ROOT)}</code>, built by <code>scripts/fp_review_page.py</code> with
<code>scripts/fp_review_page.template.html</code>; figures by <code>scripts/fp_review_figures.py</code>.</li>
<li>Toy: <code>scripts/fp_sr2_toy.py</code> (<code>run2d</code>, <code>run6d</code>, <code>sobol</code>,
<code>embed</code>, <code>press --name rrt2d|rrt6d|sobol8k</code>); dynamic check on the CPU:
<code>scripts/fp_hom_check.py</code>; rows in <code>{ROWS.relative_to(ROOT)}/</code>.</li>
<li>GPU rollouts: <code>scripts/hom_gpu.py run --task twist roll pinpush press --layouts 2d|2dfp|rrt6d|sobol8k --seeds 0 1
2 --batch 2048 --out NAME</code>, with <code>--variants rate50 cal_notip cal_tip cal_tipfl</code> for Section SEC[sense];
rows <code>hom2d_gpu_*</code>, <code>hom2dfp_gpu_*</code>, <code>hom6d_rrt_gpu_*</code>, <code>hom6d_gpu_*</code> and
<code>hom6d_gpu_sense_*</code>, each with a <code>_timing</code> file.</li>
<li>The v4b fingertip: <code>scripts/build_cad_v2_finger.py</code>, <code>assets/mjcf/experimental/20261030-cad_v2</code>;
poses <code>fp_sr2_toy.py poses</code> and <code>grid2d</code> (rows <code>sobol8k_*_v4b.jsonl</code>,
<code>grid2d_*_v4b.jsonl</code>); the check <code>logs/20261029-hom_sizes/v4b_check.sh</code> (rows
<code>v4b_check/</code>).</li>
<li>Grid of grasp sets: <code>hom_gpu.py run --layouts grid2d_size</code> (rows <code>grid2d_v4b/</code>); maps and
tables <code>scripts/hom_landscape.py grid</code> and <code>figures</code>.</li>
<li>Earlier studies: <code>docs/experiments/20260831-real_v1-sobol8192/</code>,
<code>docs/experiments/20260828-real_v1_search/</code>, <code>docs/experiments/20261008-hand_object_scale/</code>,
<code>docs/experiments/20261001-drake-port/</code>; the morphology overview <code>docs/overviews/morphology_design.html</code>.</li>
</ul>"""


# ------------------------------------------------------------------------------------------------ assemble
def assemble(sections, title, h1, eyebrow, lede, byline, summary):
    num = {sid: k for k, (sid, _hd, _b) in enumerate(sections, 1)}
    sections = [(sid, hd, re.sub(r"SEC\[(\w+)\]", lambda m_: str(num[m_.group(1)]), body)) for sid, hd, body in sections]
    lede = re.sub(r"SEC\[(\w+)\]", lambda m_: str(num[m_.group(1)]), lede)
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
    N["val"] = gpu_validation()
    N["dyn2d"] = dyn_set("hom2d_gpu_twist", "hom2d_gpu_roll")
    N["dynrrt"] = dyn_set("hom6d_rrt_gpu_twist", "hom6d_rrt_gpu_roll")
    N["dynsob"] = dyn_set("hom6d_gpu_twist", "hom6d_gpu_roll")
    N["dyn2dfp"] = dyn_set("hom2dfp_gpu_twist", "hom2dfp_gpu_roll")
    N["cyl2d"] = cyl_set("hom2dfp_gpu_pinpush", "hom2dfp_gpu_press")
    N["cylrrt"] = cyl_set("hom6d_rrt_gpu_pinpush", "hom6d_rrt_gpu_press")
    N["cylsob"] = cyl_set("hom6d_gpu_pinpush", "hom6d_gpu_press")
    N["v4b"] = v4b_numbers()
    N["grid"] = grid_numbers()
    N["grad"] = gradient_numbers()
    N["sizes"] = size_numbers()
    N["grid_sizes"] = grid_size_numbers()
    N["valrep"] = valrep_numbers()
    vj = ROWS / "grid2d_v4b" / "valrep.json"
    vfull = json.loads(vj.read_text()) if vj.exists() else {}
    N["valrep_seeds"], N["cross_fresh"] = vfull.get("fp_seed_means"), vfull.get("cross_fresh")
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
        ("gpu", "HOM on the GPU: many layouts in one batch", sec_gpu(N)),
        ("check", "Kinematic scores against HOM rollouts on the same layouts", sec_check(N)),
        ("sense", "Sensing errors across the layouts", sec_sense(N)),
        ("v4b", "The calibrated v4b fingertip in the layout search", sec_v4b(N)),
        ("grid", "Grasps synthesized per layout on a grid of symmetric tripods", sec_grid(N)),
        ("gradients", "Design gradients from a model of the HOM scores", sec_gradients(N)),
        ("sizes", "Layouts across object sizes", sec_sizes(N)),
        ("cross", "The four tasks at their default objects: rank correlation and Pareto set", sec_cross(N)),
        ("separate", "Grasp and in-hand control: one problem or two", sec_separate(N)),
        ("plan", "A plan for a Pareto front of SR2 layouts under HOM control", sec_plan(N)),
        ("related", "Related design studies and how this one builds on them", sec_related()),
        ("history", "Changes to the method on 2026-10-29 and 10-30", sec_history()),
        ("limits", "Limitations", sec_limits(N)),
        ("future", "Future work", sec_future(N)),
        ("files", "References and files", sec_files()),
    ]
    byline = ("Review and simulation, 2026-10-29 and 10-30. Paper: arXiv 2409.14158v2 [1]. Hand: the CAD-model SR2 hand "
              "(closed-form kinematics of <code>scripts/hand_object_scale_kin.py</code>; the 1 N m servo plant in the HOM "
              "rollouts, on the CPU and on MuJoCo-Warp, <code>scripts/hom_gpu.py</code>). Fingertip: the TPU block as "
              "1 mm sphere pads up to Section 8, the calibrated v4b fingertip from Section 9. "
              "Builder: <code>scripts/fp_review_page.py</code>.")
    G = N["grid"]
    lv = [g["lv_best"] for g in G.values()]
    fpb = [g["best_is_fp"] for g in G.values()]
    TV, CF = N["valrep"]["twist_d40"], N["cross_fresh"]
    pe = N["grad"]["pk"]["peak"]["radii"]["4"]
    GS = N["grid_sizes"]
    sl = []
    for task in ("twist", "pinpush", "press"):
        szs = sorted(sz for t_, sz in GS if t_ == task)
        if len(szs) >= 2:
            sl.append((GS[(task, szs[-1])]["peak"][0] - GS[(task, szs[0])]["peak"][0]) / (szs[-1] - szs[0]))
    lede = fill("""The foundational-pose paper [1] keeps a hand design if it reaches a few poses that tools are wielded
from. On the SR2 hand with the calibrated v4b fingertip, a GPU version of the HOM controller scored 8,192 layouts on four
tasks and four object sizes. With the best of 16 grasps per layout and task, the median twist score of a 625-layout grid
rose from <<a0>> to <<a1>> on fresh seeds. On <<sb>> % of the grid's layouts all four tasks then scored above 0.5,
against <<sf>> % with the foundational pose. The best layouts move with the object.<<sh>> They sit on peaks a few millimetres wide: moving the
mounts 4 mm cost the best twist layouts a median <<c4>>.""",
                a0=f"{TV['fp_fresh_median']:.2f}", a1=f"{TV['best_fresh_median']:.2f}",
                sb=f"{100 * CF['share_all_above_0.5_best']:.0f}", sf=f"{100 * CF['share_all_above_0.5_fp']:.0f}",
                c4=f"{-pe['change_median']:.2f}",
                sh=(f" In three tasks the best distance from the thumb's mount to the other two grows by {min(sl):.1f} to "
                    f"{max(sl):.1f} mm per millimetre of diameter." if len(sl) == 3 else ""))
    summary = fill("""HOM-controller scores of 8,192 SR2 layouts on four tasks and object sizes, v4b fingertip: synthesizing
the grasp per layout and task raised the grid's median twist score from <<a0>> to <<a1>> on fresh seeds and made <<sb>> %
of layouts score above 0.5 on all four tasks (<<sf>> % with the foundational pose); the best layouts move with object
size and sit on peaks a few millimetres wide, where a smooth score model's gradient fails.""",
                   a0=f"{TV['fp_fresh_median']:.2f}", a1=f"{TV['best_fresh_median']:.2f}",
                   sb=f"{100 * CF['share_all_above_0.5_best']:.0f}", sf=f"{100 * CF['share_all_above_0.5_fp']:.0f}")
    assemble(sections, "Foundational-pose hand design",
             "Foundational-pose hand design, the project's layout sampling, and a Pareto front for the SR2 hand",
             "Morphology and design search", lede, byline, summary)


if __name__ == "__main__":
    main()
