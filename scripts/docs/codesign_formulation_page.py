#!/usr/bin/env python3
"""Builds docs/experiments/20261110-codesign_formulation/20261110-sr2_codesign_formulation.html: the co-design problem of the
hand-design talk as one nested formula, built up term by term (the owner's slides 21-31), then each level in turn: the
outer loop (Yang et al.'s EA, slides 32-33), the inner loop (grasp synthesis, HOM and its tuning, slides 34-35) and the
evaluator (simulation built for co-design, slides 36-38). Owner, 2026-11-10: "an evolving latex formula/equation that
captures the co-design problem (bi-level optimization), that also endures past these slides"; "a similarly really
intuitive, first-principles latex formulation for this outer-loop optimization"; "explaining how it can be decomposed
into HOM and grasp synthesis". FORMULAS holds the LaTeX that the talk's build_talk.py copies.

Figures: scripts/docs/codesign_formulation_figures.py.

    .venv/bin/python scripts/docs/codesign_formulation_page.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import docs_index  # noqa: E402
import fp_review_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = ROOT / "docs/experiments/20261110-codesign_formulation"
MEDIA = D / "media"
OUT = D / "20261110-sr2_codesign_formulation.html"
FPR = "../20261029-foundational_pose_review/"
SEARCH, TRACK, DESIGN = (FPR + "20261109-hom_search_methods.html", FPR + "20261109-hom_tracking_score.html",
                         FPR + "20261029-foundational_pose_hand_design.html")
REFINE = "../20261103-hom_score_refine/20261103-hom_score_refine.html"
table, esc = P.table, P.esc

# (owner's slide, what the step adds, LaTeX). Each step repeats the whole formula with the new term in colour.
NEW = "\\color{#7c3aed}"
FORMULAS = [
    ("21-23", "A hand for a task: the best hand is the one with the highest task score.",
     r"m^\star_T = \arg\max_{m}\; J_T(m)"),
    ("23", "A hand can only be scored through a controller, so the hand and its controller are chosen together. "
           "The outer max is over bodies, the inner one over controllers: a bi-level problem.",
     r"m^\star_T = \arg\max_{m}\; {" + NEW + r"\max_{\pi}}\; J_T(m, {" + NEW + r"\pi})"),
    ("24-27", "The object changes. Either a hand per object size, or one hand for a distribution of sizes.",
     r"m^\star_T = \arg\max_{m}\; {" + NEW + r"\mathbb{E}_{d\sim\mathcal{D}}}\Big[\max_{\pi}\; J_T(m,\pi;\,{" + NEW
     + r"d})\Big]"),
    ("28", "The spaces: six mount coordinates for the hand, nine joints for the controller at each step.",
     r"m\in{" + NEW + r"\mathcal{M}\subset\mathbb{R}^{6}},\qquad \pi:\ x_t\mapsto {" + NEW + r"u_t\in\mathbb{R}^{9}}"),
    ("29", "The task changes too. The three toy tasks are three samples of a distribution over tasks.",
     r"m^\star = \arg\max_{m}\; \mathbb{E}_{({" + NEW + r"T},d)\sim\mathcal{D}}\Big[\max_{\pi}\; J_{" + NEW
     + r"T}(m,\pi;\,d)\Big]"),
    ("30", "The score is a choice. Several metrics are weighted into one, or kept apart on a Pareto front. Ours "
           "rewards following the commanded rotation on three pads.",
     r"J_T = {" + NEW + r"\textstyle\sum_k \lambda_k J_T^{(k)}},\qquad J = \mathbf{1}[\mathrm{held}]\,"
     r"\max\!\Big(0,\,1-\frac{e_{\mathrm{rms}}}{\theta_{\mathrm{rms}}}\Big)\,c_3"),
    ("31", "The score comes from a simulator, with a contact model, a servo model and a sensing model, each "
           "calibrated on hardware.",
     r"J_T(m,\pi;\,d)\;\approx\;J^{\,{" + NEW + r"\mathrm{sim}}}_T(m,\pi;\,d\mid {" + NEW + r"c,\,p,\,s})"),
    ("32 on", "The whole problem, kept for the rest of the talk. The controller is HOM with a grasp $g$ and parameters "
              "$\\phi$, and each brace is one section of the talk.",
     r"m^\star=\underbrace{\arg\max_{m\in\mathcal{M}}}_{\text{morphology optimization}}\ "
     r"\mathbb{E}_{(T,d)\sim\mathcal{D}}\Big[\underbrace{\max_{g\in\mathcal{G}(m,T,d)}\ \max_{\phi}}_{\text{multi-hand "
     r"control}}\ \underbrace{J^{\,\mathrm{sim}}_T(m,g,\phi;\,d\mid c,p,s)}_{\text{sim-to-real}}\Big]"),
]
OUTER = [
    ("Fitness", "A layout's fitness is the inner value: its best grasp's score, averaged over three seeds, with HOM's "
                "default parameters.",
     r"F(m)=\frac{1}{3}\sum_{s=1}^{3}\ \max_{g\in\mathcal{G}(m)}\ J^{\,\mathrm{sim}}(m,g,\phi_0;\,s)"),
    ("Selection", "Each generation keeps its best half; its best eighth are the parents.",
     r"P_t=\{m_1,\dots,m_N\},\qquad E_t=\operatorname{top}_{N/2}(P_t;F),\qquad A_t=\operatorname{top}_{N/8}(P_t;F)"),
    ("Mutation", "A child is a parent moved by up to 15 mm in each coordinate, then rounded to the 5 mm lattice and "
                 "kept inside the workspace.",
     r"m'=\Pi_{\Lambda}\big(a+\delta\,\epsilon\big),\qquad a\in A_t,\ \ \epsilon\sim\mathcal{U}\big([-\tfrac12,"
     r"\tfrac12]^2\big),\ \ \delta=30\ \mathrm{mm}"),
    ("Next generation", "The kept half and the children form the next population. A layout seen before is not "
                        "simulated again.",
     r"P_{t+1}=E_t\,\cup\,\{m'_1,\dots,m'_{N/2}\}"),
]
INNER = [
    ("Two inner choices", "For a given hand, the controller has two parts: a grasp $g$ and HOM's parameters $\\phi$.",
     r"J^\star(m)=\max_{g\in\mathcal{G}(m)}\ \max_{\phi}\ J^{\,\mathrm{sim}}(m,g,\phi)"),
    ("Grasp synthesis", "The foundational pose is the grasp farthest from every constraint: joint limits, pad facing, "
                        "and clearance between fingers and from the object. $v$ holds the object pose and the contact "
                        "points; the toy adds 15 variants.",
     r"g_0=\arg\max_{v,\,\sigma}\ \sigma\quad\text{s.t.}\quad c_j(v;\,m)\ge\sigma\ \ \forall j,\qquad "
     r"\mathcal{G}(m)=\{g_0,\dots,g_{15}\}"),
    ("HOM, the same law for every hand", "Each finger's joint speeds best produce its reference contact motion. The "
                                         "hand enters only through $A_i$ and $b_i$.",
     r"u_i=\arg\min_{u}\ \sum_k w_k\big(v_{\mathrm{ref},k}-[A_i(m,g)\,u+b_i]_k\big)^2+w_{\mathrm{d}}\lVert u\rVert^2"
     r"\quad\text{s.t. joint and speed limits}"),
    ("Pressing through the servo", "The servo target adds the joint torque of the commanded pad force, divided by "
                                   "the servo stiffness.",
     r"q_{\mathrm{cmd}}=q_{\mathrm{ref}}+\frac{1}{k_p}\,J_i^{\top}F_i\,n_i"),
    ("Tuning HOM", "$\\phi$ holds the commanded forces $F$ and the contact-mode weights $w$. The cross-entropy method samples "
                   "them, keeps the elite and refits.",
     r"\phi=(F,\,w),\qquad \phi^{(k)}\sim\mathcal{N}(\mu,\Sigma),\qquad (\mu,\Sigma)\leftarrow"
     r"\operatorname{fit}\big(\operatorname{elite}\{\phi^{(k)}\}\big)"),
]
SIM = [
    ("Rollouts per search", "Every fitness value is a batch of simulations. With a population of 64, each "
                            "generation simulates 32 new layouts, 16 grasps each, at three seeds.",
     r"\underbrace{32}_{\text{new layouts}}\times\underbrace{16}_{\text{grasps}}\times\underbrace{3}_{\text{seeds}}"
     r"=1536\ \text{rollouts per generation}"),
    ("Sensing depends on the hand", "The fingertips' magnets disturb each other's field, and how much depends on the "
                                    "layout. Sensing is then a function of the hand.",
     r"J^{\,\mathrm{sim}}(m,g,\phi\mid c,\,p,\,{\color{#7c3aed}s(m)})"),
]


def eq(tex):
    """A display formula; KaTeX's \\color{#hex} becomes xcolor's \\color[HTML]{HEX} for texsvg."""
    tex = re.sub(r"\\color\{#([0-9a-fA-F]{6})\}", lambda m_: "\\color[HTML]{" + m_.group(1).upper() + "}", tex)
    return f"<p>$${tex}$$</p>"


def rows(items):
    return "".join(f"<p><b>{esc(t)}.</b> {esc(w)}</p>{eq(x)}" for t, w, x in items)


def fig(name, caption):
    return P.img(MEDIA / name, caption)


def sec_terms():
    return (
        "<p><b>Hand</b> $m$: the layout of the SR2 hand, six palm-frame mount coordinates of the three fingers. The "
        "searches ran on its two-coordinate symmetric family: thumb to the index-middle pair and index to middle, in "
        "mm, on a 5 mm lattice. <b>Task</b> $T$: twist, roll, press or pin-push. <b>Object size</b> $d$: the sphere's "
        "or the cylinder's diameter in mm. <b>Controller</b> $\\pi$: a map from the state $x_t$ to the commands of the "
        "nine joints. Here it is HOM (hand-object manipulation control) with a grasp $g$ and parameters $\\phi$.</p>"
        "<p><b>Score</b> $J$: the tracking score of a rollout, in $[0,1]$. It is the share of the commanded rotation "
        f"the object followed, times the share of the task with all three pads loaded (<a href=\"{TRACK}\">tracking-"
        "score page</a>). $e_{\\mathrm{rms}}$ is the RMS angle between the object's orientation and the commanded one, "
        "$\\theta_{\\mathrm{rms}}$ the RMS commanded rotation, $c_3$ the three-pad share. <b>Simulator</b>: $c$ is the "
        "contact model (1 mm sphere pads at 2.5 MPa), $p$ the servo model (0.72 N m/rad), $s$ the sensing model of "
        "the magnetic fingertips.</p>")


def figinfo():
    import json
    f = MEDIA / "20261110-figures.json"
    return json.loads(f.read_text()) if f.exists() else {}


def sec_build():
    body = ["<p>The problem is stated once and grows a term per slide. The new term of each step is in colour, with a "
            "brace that names it. The slide images <code>media/20261110-eq_step1.png</code> to <code>eq_step8.png</code> "
            "share one canvas and one baseline, so the formula grows in place when the slides advance. The last form "
            "stays for the rest of the talk. Each of its braces is one SR2 section: the outer max is morphology "
            "optimization, the inner max is multi-hand control, and the score is the simulation that sim-to-real makes "
            "trustworthy.</p>"]
    for k, (sl, what, tex) in enumerate(FORMULAS, 1):
        body.append(f"<p><b>Step {k}, slide {sl}.</b> {esc(what)}</p>{eq(tex)}")
    body.append(fig("20261110-nested_all.png", "The nested problem drawn as nested boxes: the search over hands holds "
                    "the choice of grasp and parameters, which holds the simulation. Versions with one level highlighted "
                    "(nested_outer, nested_inner, nested_sim) open each section."))
    return "".join(body)


def sec_outer():
    fi = figinfo()
    sn = fi.get("ea_snapshots", {})
    hit = sn.get("hands_to_within_002", {})
    ls = fi.get("layout_space", {})
    lay = "; ".join(f"{k} at ({v['layout'][0]:.0f}, {v['layout'][1]:.0f}) mm scores {v['grid_score']:.2f}"
                    for k, v in ls.items())
    return ("<p>The outer loop searches hands. A hand here is a layout of the three finger mounts, a point on a plane of "
            "two distances (Figure 2). Every point of the plane was simulated for the press on the 25 mm cylinder, so "
            "the search can be replayed and watched. " + (f"On that landscape, {lay}. " if lay else "")
            + "The evolutionary algorithm of Yang et al. keeps the best half of each generation and mutates its best "
            "eighth (Figure 3; <a href=\"" + SEARCH + "\">search-methods page</a>).</p>"
            + rows(OUTER)
            + fig("20261110-layout_space.png", "What a layout is. Left: the palm with the three finger mounts and the two "
                  "distances that define a symmetric layout; grey, the travel of the gantries. Middle: the press score of "
                  "every layout on the 25 mm cylinder (SCS0009 plant, best of 16 grasps, mean of three seeds). Right: "
                  "three of those hands, rendered on the CPU plant at seed 0 after the lift.")
            + fig("20261110-ea_anatomy.png", "One generation of the EA with a population of 16, replayed on the stored "
                  "landscape. The 16 hands are ranked by fitness; the best two are parents, the best eight are kept. Each "
                  "parent makes four children within 15 mm in each coordinate, rounded to the 5 mm lattice.")
            + fig("20261110-ea_snapshots.png", "Where the EA has evaluated after 16, 48, 96 and 192 hands (orange; the last "
                  "16 darker; star: the best so far), and the best score found against the number of hands evaluated, "
                  "for the EA and for random search, median of 200 searches with quartiles."
                  + (f" The median EA search comes within 0.02 of the grid's best ({sn.get('best_on_grid', 0):.2f}) after "
                     f"{hit.get('EA, population 16')} hands; random search after {hit.get('random search')}." if hit else "")))


def sec_inner():
    fi = figinfo()
    gc = fi.get("grasp_candidates", {})
    sc = gc.get("scores", {})
    span = (f" Its 16 grasps score {min(sc.values()):.2f} to {max(sc.values()):.2f}; the best is grasp "
            f"{gc.get('best_grasp')}." if sc else "")
    return ("<p>The inner loop runs one hand. HOM is the same law on every hand, so only the grasp and a few of HOM's "
            "parameters are chosen per hand (<a href=\"" + REFINE + "\">score-levels page</a>). The grasp matters: the "
            "foundational-pose toy proposes 16 grasps per hand, and they score differently (Figure 5)." + span
            + " With HOM's defaults and the best of 16 grasps (level S1), no controller is trained per hand. Tuning the "
            "grasp, the forces and the contact modes per layout (level S3) raised each task's 30 best layouts by a "
            "median of 0.05 to 0.20, and it moved the best layout on three of the four tasks.</p>"
            + rows(INNER)
            + fig("20261110-grasp_candidates.png", "The 16 candidate grasps of hand A, each rendered after the lift (CPU "
                  "plant, seed 0), with the GPU's press score (mean of three seeds); green: the best.")
            + fig("20261110-contact_modes.png", "What HOM commands at each fingertip: a motion relative to the object at "
                  "the contact (stick, roll or slide), and a press along the normal with force F. The contact-mode weights "
                  "w choose among them.")
            + fig("20261110-tuning.png", "Tuning inside the inner loop: the median over the 30 best SCS0009 layouts per "
                  "task of the best score so far by CEM iteration, tuning the grasp alone (S2) or the grasp, forces and "
                  "contact modes (S3), against the defaults with the best of 16 grasps (S1). Earlier task scores: the twist's "
                  "counts its peak yaw on three pads (tracking-score page).")
            + fig("20261110-hom_period.png", "One HOM control period. The least squares is the only step where the hand's "
                  "geometry appears, through the Jacobian A(m, g); the forces F and the weights w are what tuning "
                  "changes."))


def sec_sim():
    fi = figinfo()
    g = fi.get("gpu", {})
    rate = (f"A batch of {g.get('worlds_per_batch')} press rollouts took a median {g.get('batch_s_median')} s on the "
            f"{g.get('gpu', 'GPU')}, {1000 * g.get('s_per_rollout_median', 0):.1f} ms per rollout, against 5.0 s for one "
            "rollout on the CPU. " if g else "")
    return ("<p>The outer loop asks the simulator for thousands of rollouts, so each sim-to-real component was chosen "
            "to keep the score fast, calibrated and observable (Figure 8). " + rate + "Fast: the fingertip is 1 mm "
            "sphere pads, which run in batches on the GPU and agree across MuJoCo, MuJoCo-Warp, Newton and Drake. "
            "Calibrated: the pad modulus, 2.47 MPa, was fitted to presses of three printed tips on the UR5, and the "
            "servo stiffness was identified on the hand. Observable: the magnetic fingertips measure the contact "
            "location and the normal force, the quantities the model and HOM use.</p>"
            + rows(SIM)
            + "<p>The crosstalk study on 11 layouts found that the layout changes how much one fingertip's magnets "
              "disturb another's reading. A co-design that includes sensing would put the sensors' placement into "
              "$m$.</p>"
            + fig("20261110-evaluator.png", "The simulator as an evaluator built for co-design: fast enough for a "
                  "search, calibrated on hardware, and observable through the fingertips."))


SLIDES = [
    ("21", "20261110-eq_step1.png", "a hand for a task"),
    ("23", "20261110-eq_step2.png", "the controller: bi-level"),
    ("24-27", "20261110-eq_step3.png", "object sizes"),
    ("28", "20261110-eq_step4.png", "the spaces, R^6 and R^9"),
    ("29", "20261110-eq_step5.png", "tasks"),
    ("30", "20261110-eq_step6.png", "the metric"),
    ("31", "20261110-eq_step7.png", "simulation"),
    ("31-32", "20261110-eq_step8.png, 20261110-nested_all.png", "the whole problem; its braces are the sections"),
    ("32", "20261110-eq_final_outer.png, 20261110-nested_outer.png", "section opener: choose the hand"),
    ("32-33", "20261110-layout_space.png, 20261110-eq_outer.png, 20261110-ea_anatomy.png, 20261110-ea_snapshots.png",
     "morphology optimization"),
    ("34", "20261110-eq_final_inner.png, 20261110-nested_inner.png", "section opener: run the hand"),
    ("34-35", "20261110-eq_inner_split.png, 20261110-grasp_candidates.png, 20261110-eq_inner_grasp.png, "
              "20261110-contact_modes.png, 20261110-hom_period.png, 20261110-eq_inner_hom.png, "
              "20261110-eq_inner_servo.png, 20261110-eq_inner_tune.png", "multi-hand control"),
    ("36", "20261110-eq_final_sim.png, 20261110-nested_sim.png", "section opener: score the hand"),
    ("36-38", "20261110-evaluator.png, 20261110-eq_sim_budget.png, 20261110-eq_sim_sensing.png", "sim-to-real"),
]


def sec_slides():
    body = [[sl, esc(what), "<br>".join(f"<code>{x.strip()}</code>" for x in files.split(","))] for sl, files, what in SLIDES]
    return ("<p>Every figure and formula image is a PNG in <code>media/</code>, at 400 dpi for the LaTeX images and 160 "
            "dpi for the plots, sized for a 16:9 slide (Table 1). The owner's slide numbers are those of the deck of "
            "2026-11-10.</p>" + table(["Slide", "Content", "Images"], body, "Images for the owner's slides.", num_from=9))


def sec_files():
    return ("<ul><li>LaTeX and TikZ images: <code>scripts/docs/codesign_tikz.py</code>; plots and renders: "
            "<code>scripts/docs/codesign_formulation_figures.py</code> (numbers in "
            "<code>media/20261110-figures.json</code>)</li>"
            "<li>Formulas: <code>FORMULAS</code>, <code>OUTER</code>, <code>INNER</code> and <code>SIM</code> in "
            "<code>scripts/docs/codesign_formulation_page.py</code>, used by the talk's <code>build_talk.py</code></li>"
            f"<li>Related: <a href=\"{DESIGN}\">the design page</a>, <a href=\"{SEARCH}\">the search-methods page</a>, "
            f"<a href=\"{REFINE}\">the score-levels page</a>, <a href=\"{TRACK}\">the tracking-score page</a></li></ul>")


def main():
    sections = [("terms", "Terms and symbols", sec_terms()),
                ("build", "The co-design problem, built up over slides 21 to 31", sec_build()),
                ("outer", "Morphology optimization: the outer loop", sec_outer()),
                ("inner", "Multi-hand control: the inner loop", sec_inner()),
                ("sim", "Sim-to-real: an evaluator built for co-design", sec_sim()),
                ("slides", "Images for the slides", sec_slides()),
                ("files", "Files", sec_files())]
    lede = ("The SR2 part of the hand-design talk follows one nested problem. The hand is chosen in an outer loop, "
            "the grasp and HOM's parameters in an inner loop, and every candidate is scored in simulation. The "
            "problem is built up over slides 21 to 31, one term per slide, and kept on screen afterwards. Each later "
            "section zooms into one level: the evolutionary algorithm of Yang et al. for the hand, grasp synthesis and "
            "HOM for the controller, and a calibrated GPU simulator for the score.")
    summary = ("The talk's co-design problem as one nested formula built up over slides 21-31 (hand, controller, object "
               "size, spaces, tasks, metric, simulation), then zoomed: Yang et al.'s EA (outer), grasp synthesis and "
               "HOM with CEM tuning (inner), and a fast, calibrated, observable simulator (evaluator); with diagrams.")
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{b}</section>' for k, (sid, hd, b) in
                     enumerate(sections, 1))
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": "SR2 co-design formulation", "H1": "The co-design problem of the SR2 hand talk as one nested "
                 "formula", "EYEBROW": "Hand-design talk, 2026-11-10", "LEDE": lede,
                 "BYLINE": "Formulas and diagrams for the talk of 2026-11-17; data from the pages linked in Section 7. "
                           "Builder: <code>scripts/docs/codesign_formulation_page.py</code>.",
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS + texsvg.BLOCK_CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    n = []
    page = P.render(page, n)
    page = docs_index.tag(page, topic="design", summary=summary, date="2026-11-10")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {n[0]} formulas", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
