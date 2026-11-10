#!/usr/bin/env python3
"""Builds docs/experiments/20261110-codesign_formulation/20261110-sr2_codesign_formulation.html: the co-design problem of the
hand-design talk built up one idea per slide (the owner's slides 21-31), then each level in turn: the outer loop (Yang et
al.'s EA, slides 32-33), the inner loop (grasp synthesis, HOM and its tuning, slides 34-35) and the evaluator (simulation
built for co-design, slides 36-38). Owner, 2026-11-10: "an evolving latex formula/equation that captures the co-design
problem (bi-level optimization), that also endures past these slides"; "make sure that we are incrementally, atomically
building up our mathematical/conceptual formalisms. less is more".

The formula images and their one-line descriptions come from media/20261110-formulas.json (scripts/docs/codesign_tikz.py);
the plots and renders from media/20261110-figures.json (scripts/docs/codesign_formulation_figures.py). The talk's
build_talk.py reads the same manifest.

    .venv/bin/python scripts/docs/codesign_formulation_page.py
"""
from __future__ import annotations

import json
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


def manifest():
    return json.loads((MEDIA / "20261110-formulas.json").read_text())


def figinfo():
    f = MEDIA / "20261110-figures.json"
    return json.loads(f.read_text()) if f.exists() else {}


def series(name):
    """A series of formula images, each with its step number and one-line description."""
    out = []
    for k, r in enumerate([r for r in manifest() if r["series"] == name], 1):
        uri = P.data_uri(MEDIA / f"{r['name']}.png", "image/png")
        out.append(f'<p><b>Step {k}.</b> {esc(r["what"])} <span class="small">(<code>media/{r["name"]}.png</code>)'
                   f'</span></p><p style="text-align:center"><img src="{uri}" alt="{esc(r["what"])}" '
                   f'style="max-width:88%"></p>')
    return "".join(out)


def fig(name, caption):
    return P.img(MEDIA / name, caption)


def sec_terms():
    return (
        "<p><b>Hand</b> $m$: the layout of the SR2 hand, six palm-frame mount coordinates of the three fingers. The "
        "searches ran on its two-coordinate symmetric family: thumb to the index-middle pair and index to middle, in "
        "mm, on a 5 mm lattice. <b>Task</b> $T$: twist, roll or press. <b>Object size</b> $d$: the sphere's or the "
        "cylinder's diameter in mm. <b>Controller</b> $\\pi$: a map from the state $x$ to the commands $u$ of the nine "
        "joints. Here it is HOM (hand-object manipulation control) with a grasp $g$ and parameters $\\phi$.</p>"
        "<p><b>Score</b> $J$: the tracking score of a rollout, in $[0,1]$. It is the share of the commanded rotation "
        f"the object followed, times the share of the task with all three pads loaded (<a href=\"{TRACK}\">tracking-score "
        "page</a>). <b>Simulator</b>: $c$ is the contact model (1 mm sphere pads at 2.5 MPa), $p$ the servo model "
        "(0.72 N m/rad), $s$ the sensing model of the magnetic fingertips.</p>"
        "<p>Every formula image adds one idea and carries only the formula and the brace that names the new term. "
        "The images of a series share one canvas and one baseline, so pasted at the same place on successive slides "
        "the formula grows in place.</p>")


def sec_build():
    return ("<p>The problem is stated once and grows one term per slide; the new term is in colour. After step 7 two "
            "images define the metric. The last step names the three levels, and each later section zooms into "
            "one of them.</p>" + series("codesign")
            + "<p><b>The metric, after step 7.</b></p>" + series("metric")
            + "<p><b>The section openers.</b> The final form with one level in colour.</p>" + series("zoom")
            + fig("20261110-nested_all.png", "The nested problem drawn as nested boxes: the search over hands holds "
                  "the choice of grasp and parameters, which holds the simulation. Versions with one level highlighted "
                  "(nested_outer, nested_inner, nested_sim) open each section."))


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
            + "The evolutionary algorithm of Yang et al. is built up below one line at a time (<a href=\"" + SEARCH
            + "\">search-methods page</a>).</p>"
            + fig("20261110-layout_space.png", "What a layout is. Left: the palm with the three finger mounts and the two "
                  "distances that define a symmetric layout; grey, the travel of the gantries. Middle: the press score of "
                  "every layout on the 25 mm cylinder (SCS0009 plant, best of 16 grasps, mean of three seeds). Right: "
                  "three of those hands, rendered from below on the CPU plant at seed 0 after the lift.")
            + series("outer")
            + fig("20261110-ea_anatomy.png", "One generation of the EA with a population of 16, replayed on the stored "
                  "landscape. The 16 hands are ranked by fitness; the best two are parents, the best eight are kept. Each "
                  "parent makes four children within 15 mm in each coordinate, rounded to the 5 mm lattice.")
            + fig("20261110-ea_snapshots.png", "Where the EA has evaluated after 16, 48, 96 and 192 hands: orange, the last 16 "
                  "darker; star, the best so far. Right: the best score found against the number of hands evaluated, for "
                  "the EA and for random search. Median of 200 searches, with quartiles."
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
            "median of 0.05 to 0.20. It also moved the best layout on three of the four tasks.</p>"
            + series("inner")
            + fig("20261110-grasp_candidates.png", "The 16 candidate grasps of hand A, rendered after the lift on the CPU plant "
                  "at seed 0. Each shows the GPU's press score, a mean of three seeds; green marks the best. The slide version, "
                  "20261110-grasp_candidates_wide.png, orders them from best to worst.")
            + fig("20261110-contact_modes.png", "What HOM commands at each fingertip: a motion relative to the object at "
                  "the contact (stick, roll or slide), and a press along the normal with force F. The contact-mode weights "
                  "w choose among them.")
            + fig("20261110-hom_period.png", "One HOM control period. The least squares is the only step where the hand's "
                  "geometry appears, through the Jacobian A(m, g); the forces F and the weights w are what tuning "
                  "changes.")
            + fig("20261110-tuning.png", "Tuning inside the inner loop, by CEM iteration. Each curve is the median over the "
                  "30 best SCS0009 layouts per task of the best score so far. S2 tunes the grasp alone, S3 the grasp, "
                  "forces and contact modes; S1 keeps the defaults with the best of 16 grasps. Earlier task scores: the "
                  "twist's counts its peak yaw on three pads (tracking-score page)."))


def sec_sim():
    g = figinfo().get("gpu", {})
    rate = (f"A batch of {g.get('worlds_per_batch')} press rollouts took a median {g.get('batch_s_median')} s on the "
            f"{g.get('gpu', 'GPU')}, {1000 * g.get('s_per_rollout_median', 0):.1f} ms per rollout. One rollout on the CPU "
            "took 5.0 s. " if g else "")
    return ("<p>The outer loop asks the simulator for thousands of rollouts, so each sim-to-real component was chosen "
            "to keep the score fast, calibrated and observable (Figure 9). " + rate + "Fast: the fingertip is 1 mm "
            "sphere pads, which run in batches on the GPU and agree across MuJoCo, MuJoCo-Warp, Newton and Drake. "
            "Calibrated: the pad modulus, 2.47 MPa, was fitted to presses of three printed tips on the UR5, and the "
            "servo stiffness was identified on the hand. Observable: the magnetic fingertips measure the contact "
            "location and the normal force, the quantities the model and HOM use. The crosstalk study on 11 layouts "
            "found that the layout changes how much one fingertip's magnets disturb another's reading, so sensing "
            "depends on the hand.</p>"
            + series("sim")
            + fig("20261110-evaluator.png", "The simulator as an evaluator built for co-design: fast enough for a "
                  "search, calibrated on hardware, and observable through the fingertips."))


def sec_slides():
    rows = [["21-32", "the co-design build, one term per slide", "eq_cd01 to eq_cd10"],
            ["30", "the metric", "eq_metric1, eq_metric2"],
            ["32", "the levels named; section opener", "eq_cd10, nested_all; eq_final_outer and nested_outer"],
            ["32-33", "morphology optimization", "layout_space; eq_outer1 to eq_outer6; ea_anatomy; ea_snapshots"],
            ["34", "section opener", "eq_final_inner, nested_inner"],
            ["34-35", "multi-hand control", "eq_inner1 to eq_inner4; grasp_candidates_wide; contact_modes; eq_inner5 "
                                           "with hom_period; eq_inner6; eq_inner7, eq_inner8 with tuning"],
            ["36", "section opener", "eq_final_sim, nested_sim"],
            ["36-38", "sim-to-real", "eq_sim1 to eq_sim3; evaluator"]]
    return ("<p>Every image is a PNG in <code>media/</code> named <code>20261110-&lt;name&gt;.png</code>, at 400 dpi "
            "for the LaTeX images and 160 dpi for the plots, sized for a 16:9 slide (Table 1). The slide numbers are those "
            "of the owner's deck of 2026-11-10.</p>"
            + table(["Slide", "Content", "Images, in order"], rows, "Images for the owner's slides.", num_from=9))


def sec_files():
    return ("<ul><li>Formula images and their descriptions: <code>scripts/docs/codesign_tikz.py</code> "
            "(<code>media/20261110-formulas.json</code>), with the TikZ diagrams</li>"
            "<li>Plots and renders: <code>scripts/docs/codesign_formulation_figures.py</code> (numbers in "
            "<code>media/20261110-figures.json</code>)</li>"
            f"<li>Related: <a href=\"{DESIGN}\">the design page</a>, <a href=\"{SEARCH}\">the search-methods page</a>, "
            f"<a href=\"{REFINE}\">the score-levels page</a>, <a href=\"{TRACK}\">the tracking-score page</a></li></ul>")


def main():
    sections = [("terms", "Terms and symbols", sec_terms()),
                ("build", "The co-design problem, built up over slides 21 to 32", sec_build()),
                ("outer", "Morphology optimization: the outer loop", sec_outer()),
                ("inner", "Multi-hand control: the inner loop", sec_inner()),
                ("sim", "Sim-to-real: an evaluator built for co-design", sec_sim()),
                ("slides", "Images for the slides", sec_slides()),
                ("files", "Files", sec_files())]
    lede = ("The SR2 part of the hand-design talk follows one nested problem. The hand is chosen in an outer loop, "
            "the grasp and HOM's parameters in an inner loop, and every candidate is scored in simulation. The "
            "problem is built up over slides 21 to 32, one term per slide, and kept on screen afterwards. Each later "
            "section zooms into one level: the evolutionary algorithm of Yang et al. for the hand, grasp synthesis and "
            "HOM for the controller, and a calibrated GPU simulator for the score.")
    summary = ("The talk's co-design problem built up one term per slide (hand, controller, object size, spaces, tasks, "
               "metric, simulation, HOM), then zoomed level by level: Yang et al.'s EA (outer), grasp synthesis, HOM and "
               "CEM tuning (inner), and a fast, calibrated, observable simulator (evaluator); 32 atomic LaTeX images and "
               "diagrams for the slides.")
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
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {n[0]} inline formulas", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
