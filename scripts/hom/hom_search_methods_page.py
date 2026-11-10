#!/usr/bin/env python3
"""Builds docs/experiments/20261029-foundational_pose_review/20261109-hom_search_methods.html: the search methods of the
SR2 layout study explained from first principles (the evolutionary algorithm of Yang et al., particle swarm optimization,
random search, a Sobol sequence, the FP paper's RRT, the cross-entropy refinement, Pareto fronts and t-SNE), with the six
search methods compared on the stored grids of both plants (owner, 2026-11-09: "complete analysis and explanation of the
evolutionary algorithm we used"; "see if particle swarm optimization ... improves efficiency or performance"; "a pretty
interesting comparison between these methods: sobol, rrt, yang ga, pso ga"; "greater explanation of the pareto front /
t-sne, from first principles").

Data: rows/20261109-search_compare.tsv and rows/20261109-search_compare_curves.tsv, rows/20261109-pareto_twist_press.tsv
(scripts/hom/hom_search_compare.py), with its figures in media/.

    .venv/bin/python scripts/hom/hom_search_methods_page.py
"""
from __future__ import annotations

import csv
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

D, MEDIA, ROWS = P.D, P.MEDIA, P.ROWS
OUT = D / "20261109-hom_search_methods.html"
DESIGN = "20261029-foundational_pose_hand_design.html"
TRENDS = "20261030-hom_layout_trends.html"
REFINE = "../20261103-hom_score_refine/20261103-hom_score_refine.html"
CONFLICT = "../20261103-hom_task_conflict/20261104-hom_task_conflict.html"
fill, table, img, esc = P.fill, P.table, P.img, P.esc
TASKS = ("twist", "roll", "pinpush", "press")
TNAME = {"twist": "Twist", "roll": "Roll", "pinpush": "Pin-push", "press": "Press"}
PLANTS = (("work", "work plant, 10 MPa"), ("scs0009", "SCS0009 plant, 2.5 MPa"))
METHODS = ("ea64", "ea16", "pso_g16", "pso_g64", "pso_r64", "random", "sobol", "rrt")
PROSE = {"ea64": "the EA with 64", "ea16": "the EA with 16", "pso_g16": "the 16-particle swarm",
         "pso_g64": "the 64-particle swarm", "pso_r64": "the ring swarm", "random": "random search", "sobol": "Sobol",
         "rrt": "RRT"}
MHEAD = {"ea64": "EA 64", "ea16": "EA 16", "pso_g16": "PSO 16", "pso_g64": "PSO 64", "pso_r64": "PSO ring 64",
         "random": "Random", "sobol": "Sobol", "rrt": "RRT"}
REFS = [
    '[1] S. Wang, J. Oh and N. S. Pollard, "The Foundational Pose as a Selection Mechanism for the Design of '
    'Tool-Wielding Multi-Finger Robotic Hands", IEEE-RAS Humanoids 2025, arXiv 2409.14158.',
    '[2] B. Yang, L. Jiang, W. Wu, M. Cheng and K. Gong, "Evolving Robot Morphology with Manipulation Learning Inspired '
    'by Human Hand Evolution", IEEE Transactions on Evolutionary Computation, 2026.',
    '[3] J. Kennedy and R. Eberhart, "Particle swarm optimization", Proc. IEEE International Conference on Neural '
    'Networks, 1942-1948, 1995.',
    '[4] M. Clerc and J. Kennedy, "The particle swarm: explosion, stability, and convergence in a multidimensional '
    'complex space", IEEE Transactions on Evolutionary Computation 6(1), 58-73, 2002.',
    '[5] I. M. Sobol\', "On the distribution of points in a cube and the approximate evaluation of integrals", USSR '
    'Computational Mathematics and Mathematical Physics 7(4), 86-112, 1967.',
    '[6] P.-T. de Boer, D. P. Kroese, S. Mannor and R. Y. Rubinstein, "A tutorial on the cross-entropy method", Annals of '
    'Operations Research 134, 19-67, 2005.',
    '[7] S. M. LaValle, "Rapidly-exploring random trees: a new tool for path planning", TR 98-11, Iowa State University, '
    '1998.',
    '[8] O. Shoval, H. Sheftel, G. Shinar, Y. Hart, O. Ramote, A. Mayo, E. Dekel, K. Kavanagh and U. Alon, "Evolutionary '
    'trade-offs, Pareto optimality, and the geometry of phenotype space", Science 336, 1157-1160, 2012.',
    '[9] L. van der Maaten and G. Hinton, "Visualizing data using t-SNE", Journal of Machine Learning Research 9, '
    '2579-2605, 2008.',
    '[10] M. Wattenberg, F. Viégas and I. Johnson, "How to use t-SNE effectively", Distill, 2016.',
]
ALGS: list[str] = []


def alg(caption, body):
    """An algorithm as a LaTeX algorithmic block (rendered by texsvg after the page's inline mathematics)."""
    ALGS.append(f"\\begin{{algorithmic}}[1]\n{body}\n\\end{{algorithmic}}")
    return (f'<figure class="alg"><figcaption><b>Algorithm&#160;{len(ALGS)}.</b> {caption}</figcaption>'
            f"\x00ALG{len(ALGS) - 1}\x00</figure>")


def load():
    S = {(r["plant"], r["task"], r["method"]): r for r in csv.DictReader(open(ROWS / "20261109-search_compare.tsv"),
                                                                          delimiter="\t")}
    F = list(csv.DictReader(open(ROWS / "20261109-pareto_twist_press.tsv"), delimiter="\t"))
    return S, F


def pc(S, plant, task, m, col="within_0.05_by_6000"):
    return 100 * float(S[(plant, task, m)][col])


def pct(v):
    return f"{v:.0f}"


def span_pc(vals):
    lo, hi = min(vals), max(vals)
    return pct(lo) if round(lo) == round(hi) else f"{pct(lo)} to {pct(hi)}"


def med(S, plant, task, m):
    v = S[(plant, task, m)]["median_to_0.05"]
    return f"{float(v):,.0f}" if v else "&#8211;"


# ------------------------------------------------------------------------------------------------ sections
def sec_terms():
    terms = [
        ("Layout, lattice", "the SR2 hand's symmetric tripod in two coordinates: $x_{sep}$, from the thumb's mount to the "
         "pair's, and $y_{sep}$, from the index's mount to the middle's. The lattice has 625 layouts in steps of 5 mm, "
         "$x_{sep}$ from 40 to 160 mm and $y_{sep}$ from 50 to 170 mm."),
        ("HOM tasks", "the twist and the roll of a 40 mm sphere, and the pin-push and the press of a cylinder 25 mm "
         "across, under the HOM controller (trends page)."),
        ("Fitness", "a layout's best-of-16 score: the best mean over seeds 0-2 among the grasps of its grasp set, up to 16 "
         "grasps solved by the pose toy. A score lies in [0, 1] and is 0 when the object drops."),
        ("Rollout", "one simulated run of one grasp at one seed, the unit of cost. A layout costs three rollouts per grasp, "
         "up to 48."),
        ("Full grid", "every layout of the lattice with every grasp at seeds 0-2: 17,712 to 25,296 rollouts per task."),
        ("Replay", "a search run on a full grid's stored scores instead of new rollouts. A layout counts its rollouts the "
         "first time a search visits it, a second visit is free, and a layout whose pose is not reached scores 0 at no "
         "rollouts."),
        ("Incumbent", "the best layout a run has evaluated so far."),
        ("Earlier score, tracking score", "the earlier score is the landscapes' score before 2026-11-09 "
         "(<code>hom_landscape.rollout_score</code>); the stored grids of Section SEC[results] have only it. The tracking score of "
         "the <a href=\"20261109-hom_tracking_score.html\">tracking page</a> is the share of the commanded rotation "
         "that the object followed, times the share of the task on three loaded pads, and 0 when the object is "
         "dropped. Layouts are ranked by it since 2026-11-09 (owner), with the earlier score beside it."),
        ("Within 0.05", "the incumbent's fitness is within 0.05 of the grid's best. The share within 0.05 by N is the share "
         "of 200 runs whose incumbent is within 0.05 after N rollouts. The rollouts to 0.05 are their median over the "
         "runs that got there."),
        ("Basin", "the layouts that drain to one peak of the smoothed best-of-16 map (design page, Section 10). A run ends "
         "in another basin when its last incumbent lies in a basin other than the grid's best layout's."),
        ("Work plant, SCS0009 plant", "the finger servos' model in the scenes. The work plant has 4 N&#8201;m/rad and "
         "1 N&#8201;m, with pads at 10 MPa. The SCS0009 plant has 0.72 N&#8201;m/rad and 0.18 N&#8201;m, the hand's "
         "servos at the datasheet's stiffness, with pads at 2.5 MPa."),
        ("EA, PSO, CEM, Sobol, RRT", "the evolutionary algorithm of Yang et al. (Section 3), particle swarm optimization "
         "(Section 4), the cross-entropy method (Section 7), a Sobol sequence and a rapidly-exploring random tree "
         "(Section 5)."),
        ("Pareto front, t-SNE", "Sections 8 and 9."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"


def sec_problem():
    return fill(r"""<p>A full grid of one task costs 17,712 to 25,296 rollouts. The size study of the
<a href="<<trends>>">trends page</a> therefore searched each new object size with a budget of 6,000 rollouts. A search is
an order of visits to the lattice, set in advance or from the scores seen so far. Each visit returns only the layout's
fitness, at up to 48 rollouts. On the lattice the full grid gives the true best layout, so every search can be scored
against it.</p>
<p>Six methods are compared. In two of them, the EA and PSO, the next layouts follow from the scores. In three, random
search, a Sobol sequence and the RRT of the FP paper, the order of visits is set before any score is known. The same evaluator, a replay
of the stored grid, scores every run, so the methods differ only in the layouts visited. The grids are those of
the trends page at the default objects, on two plants. No new rollout was run for this page.</p>""", trends=TRENDS)


def sec_ea():
    t = fill(r"""<p>An evolutionary algorithm keeps a population of candidate designs. Each generation it scores
them, keeps the better ones and makes new ones from them. Selection passes the information of the scores to the next generation,
and variation explores near what scored well. In an elitist EA the best candidates pass to the next generation unchanged,
so the best score found never falls. A mutation-only EA makes a child by perturbing one parent. It has no
recombination, the mixing of two parents' parameters that a genetic algorithm adds.</p>
<p>Yang et al. [2] evolved 19 parameters of a five-finger hand: finger locations and phalanx lengths. Each generation
of 4 &#215; 512 hands kept its best 50 % and took its best 12.5 % as parents. A child was a parent with every parameter
moved by a uniform draw in $\pm\delta/2$. Each hand was scored with one reinforcement-learning controller, shared by
the population, on 64 grasp and reorientation scenarios, over 11 generations.</p>
<p>Our EA keeps their selection and runs it on the lattice (<code>hom_landscape.ea_search</code>). Each generation keeps
its best half, takes its best eighth as parents, and fills the other half with children. A child moves its parent by a
uniform draw in $\pm 15$ mm per coordinate ($\delta$ = 30 mm), rounded to the lattice and clipped to the gantries'
travel. A layout is evaluated once. HOM has no learned parameters, so a layout's fitness does not depend on the search
that visits it. The live searches of the trends page ran 64 individuals and 6,000 rollouts per object size.</p>""")
    t += alg("The elitist, mutation-only EA of Yang et al. on the layout lattice (<code>hom_landscape.ea_search</code>).",
             r"""\Require population \(N\) (16 or 64), mutation width \(\delta = 30\) mm, budget \(B\) rollouts
\State draw \(N\) layouts at random; evaluate each \Comment{fitness: best-of-16 score}
\While{rollouts spent \(< B\)}
\State sort the population by fitness
\State keep the best \(N/2\); the best \(N/8\) are the parents
\For{each of the other \(N/2\) places}
\State child \(\gets\) next parent \(+\,\mathcal{U}(-\delta/2,\,\delta/2)\) per coordinate
\State round the child to the 5 mm lattice and clip it to the gantries' travel
\State evaluate the child, unless it was evaluated before
\EndFor
\EndWhile
\State \Return the incumbent""")
    return t


def sec_pso():
    t = fill(r"""<p>PSO [3] keeps a swarm of particles. A particle is a point $x_i$ of the search space with a
velocity $v_i$. Each particle's best point so far is kept as $p_i$, and the best point of the whole swarm as
$g$. Each step pulls a particle toward both with random weights and keeps part of its last
velocity:</p>
$$v_i \leftarrow w\,v_i + c_1\,r_1 \odot (p_i - x_i) + c_2\,r_2 \odot (g - x_i), \qquad x_i \leftarrow x_i + v_i,$$
<p>with $r_1$ and $r_2$ uniform in [0, 1] for each coordinate. The inertia $w$ keeps the particles moving and spreads
the search, and the two pulls concentrate it. Clerc and Kennedy [4] showed that $w$ = 0.7298 and $c_1 = c_2$ = 1.49618
make the swarm converge. In a ring the particles see the best of themselves and their two neighbours instead of $g$,
which slows the swarm's collapse onto one point. PSO keeps every particle and moves it. It has no selection,
recombination or mutation, the operators that define a genetic algorithm.</p>
<p>Here the particles move in mm over the gantries' box and are scored at the nearest lattice layout. Velocities are
capped at a fifth of the box, and a particle that reaches a wall stops there. The swarms are 16 and 64 particles with
the global best, and 64 on a ring (<code>scripts/hom/hom_search_compare.py</code>).</p>""")
    t += alg("Particle swarm optimization on the layout lattice (<code>hom_search_compare.pso_search</code>).",
             r"""\Require swarm size \(M\), \(w = 0.7298\), \(c_1 = c_2 = 1.49618\), velocity cap \(v_{\max}\) = a fifth of the box
\State draw positions \(x_i\) in the box and velocities \(v_i\) in \([-v_{\max}, v_{\max}]\); evaluate each \(x_i\) at its nearest layout
\State \(p_i \gets x_i\); \(g \gets\) the best \(p_i\)
\While{rollouts spent \(< B\)}
\For{each particle \(i\)}
\State \(v_i \gets w\,v_i + c_1 r_1 \odot (p_i - x_i) + c_2 r_2 \odot (g - x_i)\), capped at \(\pm v_{\max}\)
\State \(x_i \gets x_i + v_i\), stopped at the box's walls
\State evaluate the layout nearest \(x_i\), unless it was evaluated before
\State if it beats \(p_i\): \(p_i \gets x_i\)
\EndFor
\State \(g \gets\) the best \(p_i\) \Comment{ring: the best of each particle's two neighbours}
\EndWhile
\State \Return the incumbent""")
    return t


def sec_sampling():
    return r"""<p>Random search visits the lattice in a random order, each layout once. A Sobol sequence [5] has low
discrepancy: its first $n$ points cover the square more evenly than $n$ random points, so no region stays unvisited for
long. Here a scrambled sequence of 4,096 points is rounded to the lattice and visited in order.</p>
<p>The FP paper [1] samples its designs with a rapidly-exploring random tree (RRT) [7]. It draws a uniform point in the
standardized design space, finds the nearest design in its tree, and steps toward the point. The new design joins the
tree when its foundational poses are solved. The tree grows outward from a seed through the designs whose poses are
reachable, at a spacing set by the step. It was run on the SR2 hand from the layout (100, 110) mm, with a step of 0.05
of the box, 6 mm (design page, Section 5). The same tree grows here on the lattice: a layout that reaches
the pose joins it, at its rollouts' cost.</p>"""


def sec_results(S):
    t = fill(r"""<p>On the work plant every method came within 0.05 by 6,000 rollouts in <<wmin>> % of runs or more
(Figure <<f1>>, Table <<t1>>).</p>
<p>The methods differ on the roll of the SCS0009 plant. By 6,000 rollouts the EA with 64 individuals came within 0.05 in
<<e64>> % of runs, and with 16 in <<e16>> %. The swarms did so in <<pso>> %, random search in <<rnd>> %, Sobol in <<sob>>
% and RRT in <<rrt>> %. The roll's map has peaks of nearly equal height in separate basins (design page, Section 14). A
global-best swarm gathers on the first good region: <<po>> % of the 16-particle swarm's runs ended in another basin than
the grid's best, against <<eo>> % for the EA with 64.</p>
<p>On the other three tasks of the SCS0009 plant a 16-particle swarm did as well as the EA. It came within 0.05 in <<gt>> % of
twist runs by 6,000 rollouts, against <<et>> % for the EA with 64 individuals, and in <<gp>> % of press runs against
<<ep>> %. Larger swarms were slower on the press, at <<g64>> % for 64 particles and <<r64>> % on the ring.</p>""",
             f1=P.FIG[0] + 1, t1=P.TAB[0] + 1,
             wmin=pct(min(pc(S, "work", tk, m) for tk in TASKS for m in METHODS)),
             e64=pct(pc(S, "scs0009", "roll", "ea64")), e16=pct(pc(S, "scs0009", "roll", "ea16")),
             pso=span_pc([pc(S, "scs0009", "roll", m) for m in ("pso_g16", "pso_g64", "pso_r64")]),
             rnd=pct(pc(S, "scs0009", "roll", "random")), sob=pct(pc(S, "scs0009", "roll", "sobol")),
             rrt=pct(pc(S, "scs0009", "roll", "rrt")),
             po=pct(pc(S, "scs0009", "roll", "pso_g16", "other_basin")),
             eo=pct(pc(S, "scs0009", "roll", "ea64", "other_basin")),
             gt=pct(pc(S, "scs0009", "twist", "pso_g16")), et=pct(pc(S, "scs0009", "twist", "ea64")),
             gp=pct(pc(S, "scs0009", "press", "pso_g16")), ep=pct(pc(S, "scs0009", "press", "ea64")),
             g64=pct(pc(S, "scs0009", "press", "pso_g64")), r64=pct(pc(S, "scs0009", "press", "pso_r64")))
    t += img(MEDIA / "20261109-search_compare.png",
             "Share of 200 runs whose incumbent is within 0.05 of the grid's best, against the rollouts spent, for each "
             "task at its default object on the two plants. Replays of the stored grids; the dotted line marks the live "
             "searches' 6,000 rollouts. The ring swarm is in Tables 1 and 2 only. "
             "<code>scripts/hom/hom_search_compare.py</code>.")
    rows = []
    for plant, pname in PLANTS:
        for tk in TASKS:
            r0 = S[(plant, tk, "ea64")]
            rows.append([f"{pname}, {TNAME[tk].lower()}", f"{float(r0['grid_best']):.2f}", f"{int(r0['grid_rollouts']):,}"]
                        + [pct(pc(S, plant, tk, m)) for m in METHODS])
    t += table(["Plant and task", "Grid's best", "Grid rollouts"] + [MHEAD[m] for m in METHODS], rows,
               "Share of 200 runs (%) within 0.05 of the grid's best by 6,000 rollouts. EA 64 and EA 16: the EA with 64 "
               "and 16 individuals; PSO 16 and 64: global-best swarms; PSO ring 64: 64 particles on a ring.", num_from=1)
    rows = [[f"{pname}, {TNAME[tk].lower()}"] + [med(S, plant, tk, m) for m in METHODS]
            for plant, pname in PLANTS for tk in TASKS]
    t += table(["Plant and task"] + [MHEAD[m] for m in METHODS], rows,
               "Median rollouts to the first incumbent within 0.05, over the runs that got there before the grid's own "
               "rollouts ran out; &#8211;: fewer than half did.", num_from=1)
    t += fill(r"""<p>Where every method succeeds, the speeds differ by about a factor of two (Table <<t2>>). The EA
with 16 individuals and the 16-particle swarm were often the fastest, because a small population converges quickly.
That convergence costs them the roll. Once the population sits on one peak, children and particles revisit evaluated
layouts, which costs nothing and finds nothing new, and their curves in Figure <<f1>> stop rising.</p>
<p>RRT reaches first the layouts near its seed. The seed (100, 110) mm is within 0.05 of the best twist on the SCS0009
plant, and of the best roll on the work plant. There RRT needs only the seed's <<seed>> rollouts. The best roll on the
SCS0009 plant lies far from the seed, and RRT reached it after a median of <<rr>> of the grid's <<rt>> rollouts. Sobol
behaves like random search: by 6,000 rollouts it differs from it by <<sd>> percentage points across the eight
cases.</p>
<p>On this lattice the EA with 64 individuals, the live searches' setting, is the most reliable method. A small swarm or
a small EA reaches near-best layouts sooner on smooth maps and misses the flat roll more often. The lattice has only
625 layouts, so random search is a strong baseline. It came within 0.05 by 6,000 rollouts in <<rmin>> % or more of runs,
except on the SCS0009 roll.</p>""",
              t2=P.TAB[0], f1=P.FIG[0], seed=med(S, "scs0009", "twist", "rrt"), rr=med(S, "scs0009", "roll", "rrt"),
              rt=f"{int(S[('scs0009', 'roll', 'rrt')]['grid_rollouts']):,}",
              sd=_sobol_diff(S), rmin=pct(min(pc(S, p, tk, "random") for p, _ in PLANTS for tk in TASKS
                                             if (p, tk) != ("scs0009", "roll"))))
    return t


def _sobol_diff(S):
    d = [pc(S, p, tk, "sobol") - pc(S, p, tk, "random") for p, _ in PLANTS for tk in TASKS]
    lo, hi = min(d), max(d)
    return f"&#8722;{abs(lo):.0f} to +{hi:.0f}"


def sec_refine():
    t = fill(r"""<p>A layout's grid score uses defaults. Its grasps are the up to 16 that the pose toy solves for
the layout and the object, the foundational pose first, and HOM's commanded forces and contact modes are the task's
defaults. They were tuned per layout on the <a href="<<refine>>">score refinement page</a>, by the
cross-entropy method (CEM) [6]. CEM draws candidates from a Gaussian, keeps the best quarter, and moves the Gaussian
toward them (Algorithm <<a>>). S2 tunes the grasp alone, with 24 candidates per iteration for 8 iterations. S3 tunes the
grasp, the commanded forces and the contact modes' weights together, with 32 candidates per iteration. Every candidate
runs at seeds 0-2. Each pick runs again at seeds 20-24, which no search used, because the best of about 192 noisy means
is biased upward.</p>
<p>On the 30 best layouts per task on the SCS0009 plant, tuning raised the median validated score by 0.05 (roll) to 0.20
(press). It also reordered the layouts (Spearman 0.56 to 0.75). One set of forces and modes shared by all layouts gained
0.19 of the press's 0.20. On the other three tasks a shared set gained 0.005 or less over HOM's defaults. One grasp per layout was
also put in the EA's genome. The incumbents' median validated score then changed by 0.00, &#8722;0.07, +0.06 and +0.17 on
the twist, roll, pin-push and press (refinement page, Section 8).</p>""", refine=REFINE, a=len(ALGS) + 1)
    import hom_grasp_refine as R
    Q = R.report_pso("sv25")
    if "press" in Q:
        rest = [k for k in ("twist", "roll") if k in Q]
        dmax = max(abs(Q[k]["PSO"]["gain_old_median"] - Q[k]["CEM"]["gain_old_median"]) for k in rest)
        t += fill(r"""<p>A swarm of 32 particles with the update of Section SEC[pso] also replaced CEM in S3, with CEM's
rollouts on the same 30 layouts per task (refinement page, Section 9). It raised the median validated score of the press's layouts
by <<pg>> over S1, against <<cg>> for CEM. Its pick scored more than 0.02 above CEM's on <<b>> of the 30 press layouts and
more than 0.02 below it on <<w>>. On <<rest>> the two gains differ by <<dmax>> or less.</p>""",
                  pg=f"{Q['press']['PSO']['gain_old_median']:.2f}", cg=f"{Q['press']['CEM']['gain_old_median']:.2f}",
                  b=Q["press"]["paired_old"]["better"],
                  w={0: "none", 1: "one", 2: "two", 3: "three"}.get(Q["press"]["paired_old"]["worse"],
                                                                   Q["press"]["paired_old"]["worse"]),
                  rest=" and ".join(f"the {k}" for k in rest), dmax=f"{dmax:.2f}")
    t += alg("The cross-entropy method of the score refinement (<code>scripts/hom/hom_grasp_refine.py</code>).",
             r"""\Require start \(\mu_0\) (the grasp, forces and modes to tune), spread \(\sigma_0\), floor \(\sigma_{\min}\), \(n\) candidates per iteration
\For{\(k = 0, 1, \ldots\)}
\State draw \(n\) candidates from \(\mathcal{N}(\mu_k, \mathrm{diag}\,\sigma_k^2)\) inside the variables' ranges \Comment{a grasp the pose toy rejects is drawn again}
\State run each candidate at seeds 0-2; the best quarter by mean score are the elites \(E\)
\State \(\mu_{k+1} \gets 0.8\,\bar{x}_E + 0.2\,\mu_k\); \(\sigma_{k+1} \gets \max(0.8\,s_E + 0.2\,\sigma_k,\ \sigma_{\min})\)
\EndFor
\State \Return the best candidate seen, then run it again at seeds 20-24""")
    return t


def sec_pareto(F):
    keys = [(float(r["x_sep_mm"]), float(r["y_sep_mm"])) for r in F]
    front = [r for r in F if r["front"] == "1"]
    g = next((r for r in front if float(r["x_sep_mm"]) == 70 and float(r["y_sep_mm"]) == 110), None)
    t = fill(r"""<p>With one task the best layout is the one with the highest score. With two tasks a layout has two
scores, and two layouts can each beat the other on one task. Layout A dominates layout B when A scores at least as high
as B on every task and higher on at least one. The Pareto front is the set of layouts that no other layout dominates.
Every layout off the front is beaten on all tasks at once by some layout on it. So for any weights a designer gives the
tasks, the best choice lies on the front.</p>
<p>On the SCS0009 plant's grids <<n>> layouts reach both the twist's and the press's pose, and <<k>> of them form the
front of the two scores (Figure <<f>>).<<gen>> The front is a staircase: between two of its layouts no layout reaches
the scores of the line joining them. A weighted sum of the scores picks one front layout for each choice of weights, but
never one in a dent of the front, which the front itself keeps.</p>
<p>The front selects less as tasks are added, because a layout stays on it when every other layout falls short of it on
at least one score. With the
six kinematic scores of the FP paper's pipeline, 193 of the SR2 hand's 364 RRT candidates are Pareto-optimal (design
page, Section 5). A narrower question is therefore asked on the <a href="<<conflict>>">task-conflict page</a>: how few
layouts reach a given share of every task's best. Shoval et al. [8] assumed that each task has one optimum and that the score
falls with the distance from it. Under those assumptions the Pareto-optimal designs lie on the segment, or in the
polygon, between the single-task optima. The HOM landscapes have many peaks at the 5 mm scale. On the SCS0009 plant a layout on the straight
segment between two specialists is beaten on both tasks by up to 0.61 (task-conflict page, Section 3).</p>""",
             n=len(keys), k=len(front), f=P.FIG[0] + 1, conflict=CONFLICT,
             gen=(f" The layout (70, 110) mm scores {float(g['twist']):.2f} on the twist and {float(g['press']):.2f} on the "
                  "press at seeds 0-2. It lies on the front. On the task-conflict page the same layout reaches 0.84 of "
                  "each task's best over all four tasks." if g else ""))
    t += img(MEDIA / "20261109-pareto_twist_press.png",
             "Twist score on the 40 mm sphere against press score on the 25 mm cylinder of every layout that reaches both "
             "poses, SCS0009 plant, best-of-16 at seeds 0-2. Red: the Pareto front, joined as the staircase of the "
             "scores it dominates. Blue: a balanced layout off the front, with the region of scores that would dominate "
             "it shaded. <code>scripts/hom/hom_search_compare.py</code>.")
    return t


def sec_tsne():
    return fill(r"""<p>t-SNE [9] draws the points of a high-dimensional space in a plane so that each point's near
neighbours stay near it. It works in three steps. First, for each point $i$ the distances to the others become
probabilities with a Gaussian kernel,</p>
$$p_{j\mid i} = \frac{\exp\!\left(-\lVert x_i - x_j\rVert^2 / 2\sigma_i^2\right)}{\sum_{k\neq i}\exp\!\left(-\lVert x_i - x_k\rVert^2 / 2\sigma_i^2\right)}, \qquad p_{ij} = \frac{p_{j\mid i} + p_{i\mid j}}{2n}.$$
<p>Each $\sigma_i$ is set so that the perplexity of point $i$'s distribution, 2 to the power of its entropy, equals a
chosen number, typically 30. The perplexity is about the number of neighbours each point attends to. Second, points
$y_i$ in the plane get similarities from a Student-t kernel with one degree of freedom:</p>
$$q_{ij} = \frac{\left(1 + \lVert y_i - y_j\rVert^2\right)^{-1}}{\sum_{k\neq l}\left(1 + \lVert y_k - y_l\rVert^2\right)^{-1}}.$$
<p>Third, the $y_i$ move by gradient descent to lower the Kullback-Leibler divergence
$\mathrm{KL}(P\Vert Q) = \sum_{i\neq j} p_{ij}\log(p_{ij}/q_{ij})$. In the divergence, near neighbours drawn far
apart cost much more than distant points drawn near. The t kernel's heavy tail lets moderately distant points sit far apart
in the plane, and that separates groups into islands.</p>
<p>In a t-SNE map the neighbours and the groups are kept. The distances between groups, the groups' sizes and densities,
and any axis are not, and runs at other perplexities or seeds give other pictures [10]. Points
coloured by a score show whether good designs gather in groups.</p>
<p>The FP paper's t-SNE map of its RRT designs fell into isolated islands [1]. What makes them was tested on the SR2 hand
(design page, Section 5). A t-SNE of the six-coordinate RRT candidates falls into islands of filaments, like the
paper's map, and so does a random subset of them. As many Sobol candidates spread evenly, with no islands. The islands
therefore come from the tree's growth along filaments, before any score is computed. On the 2-D lattice the scores' own structure shows as basins. A task's map has a few dozen
separate regions of good layouts at each size (design page, Section 12).</p>""")


# ------------------------------------------------------------------------------------------------ live searches
S6DIR = ROWS / "search6d_sv25"
S6SUM = ROWS / "20261110-search6d_sv25_summary.json"
S6FIG = "20261110-search6d_curves.png"
S6_TASKS = {"twist": 40.0, "roll": 40.0, "press": 25.0}
S6_METHODS = ("ea64", "pso16", "pso64r", "random", "sobol", "rrt")
S6_HEAD = {"ea64": "EA 64", "pso16": "PSO 16", "pso64r": "PSO ring 64", "random": "Random", "sobol": "Sobol",
           "rrt": "RRT"}
OBJ = {"twist": "40 mm sphere", "roll": "40 mm sphere", "press": "25 mm cylinder"}


def curve(log):
    """A search's (rollouts, incumbent fitness) per round, without the rounds that changed neither."""
    out = []
    for g in log:
        pt = [int(g["rollouts"]), round(float(g["incumbent_fit"]), 4)]
        if not out or pt != out[-1]:
            out.append(pt)
    return out


def s6_summary():
    """Post-reset step 4 (scripts/hom/hom_search6d.py): per task, each search's incumbent curve and last incumbent, and
    each incumbent's earlier and tracking scores at seeds 20-24; the layouts that reach the pose, those the searches ran,
    Spearman's rho between the two scores over them, and the five best by the tracking score at seeds 0-2 with their
    reruns at seeds 20-24 (post-reset step 18). Written to S6SUM from rows/search6d_sv25/ where those rows are, read
    from S6SUM elsewhere (the shared rollouts stay on the workstation)."""
    if not S6DIR.exists() or not (S6DIR / "twist_d40.jsonl").exists():
        return json.loads(S6SUM.read_text()) if S6SUM.exists() else {}
    import hom_landscape as L
    from sr2hand.numerics.stats import spearman
    from scipy.spatial import cKDTree
    dmm, Z = L.layouts()
    tree = cKDTree(Z)
    spacing = float(np.median(tree.query(Z, k=2)[0][:, 1]))
    rng = np.random.default_rng(0)
    par = rng.choice(len(Z), 20000)   # an EA child of a random parent: each coordinate moved by U(-0.125, 0.125)
    kid = np.clip(Z[par] + rng.uniform(-0.125, 0.125, (20000, 6)), -0.5, 0.5)
    snap = float(np.mean(tree.query(kid)[1] == par))
    out = {}
    for task, d in S6_TASKS.items():
        runs = {}
        for p in sorted(S6DIR.glob(f"{task}_*_s[0-9]_gens.jsonl")):
            m, s = p.stem[len(task) + 1:-len("_gens")].rsplit("_s", 1)
            log = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            runs.setdefault(m, []).append({"seed": int(s), "layout": int(log[-1]["incumbent"]),
                                           "fit": float(log[-1]["incumbent_fit"]),
                                           "rollouts": int(log[-1]["rollouts"]), "layouts": int(log[-1]["layouts"]),
                                           "curve": curve(log)})
        if not runs:
            continue
        val, trk, rms = {}, {}, {}
        vp = S6DIR / f"{task}_d{d:g}_val.jsonl"
        for x in (vp.read_text().splitlines() if vp.exists() else []):
            if x.strip():
                r = json.loads(x)
                val.setdefault(int(r["call"]), []).append(L.rollout_score(task, r))
                trk.setdefault(int(r["call"]), []).append(L.track_score(task, r))
                rms.setdefault(int(r["call"]), []).append((r.get("track_err_rms_deg") if r.get("held") else None,
                                                           r.get("track_ref_rms_deg"), r.get("contact_all_frac", 1.0)))

        def detail(c):
            """A layout's RMS angle error (min, max over the seeds held), commanded RMS rotation and mean share of the
            task with all three pads loaded, at seeds 20-24."""
            a = rms.get(c, [])
            e = [x[0] for x in a if x[0] is not None]
            if len(a) < 5 or not e:
                return None
            return {"err_deg": [round(min(e), 1), round(max(e), 1)], "ref_deg": round(float(np.mean([x[1] for x in a])), 1),
                    "contact_all": round(float(np.mean([x[2] for x in a])), 3), "held": len(e)}
        for rs in runs.values():
            rs.sort(key=lambda r: r["seed"])
            for r in rs:
                c = r["layout"]
                r["val"] = float(np.mean(val[c])) if len(val.get(c, ())) >= 5 else None
                r["val_track"] = float(np.mean(trk[c])) if len(trk.get(c, ())) >= 5 else None
                r["mm"] = [round(float(v), 2) for v in dmm[c]]
                r["rms"] = detail(c)
        acc, acc_t = {}, {}
        sp = S6DIR / f"{task}_d{d:g}.jsonl"
        for x in (sp.read_text().splitlines() if sp.exists() else []):
            if x.strip():
                r = json.loads(x)
                acc.setdefault(int(r["call"]), []).append(L.rollout_score(task, r))
                acc_t.setdefault(int(r["call"]), []).append(L.track_score(task, r))
        ev = sorted(c for c, v in acc.items() if len(v) >= 3)
        fe = {c: float(np.mean(acc[c])) for c in ev}
        ft = {c: float(np.mean(acc_t[c])) for c in ev}
        reach = len(L.reach(task, d, src="sobol8kff"))
        top = [{"layout": c, "fit": round(fe[c], 4), "fit_track": round(ft[c], 4),
                "mm": [round(float(v), 2) for v in dmm[c]],
                "val": float(np.mean(val[c])) if len(val.get(c, ())) >= 5 else None,
                "val_track": float(np.mean(trk[c])) if len(trk.get(c, ())) >= 5 else None, "rms": detail(c)}
               for c in sorted(ev, key=lambda c: -ft[c])[:5]]
        out[task] = {"d": d, "runs": runs, "evaluated": len(ev), "reached": reach, "spacing": round(spacing, 3),
                     "snap": round(snap, 3),
                     "rho": round(float(spearman(np.array([fe[c] for c in ev]), np.array([ft[c] for c in ev]))), 3),
                     "track_top": top}
    if out:
        S6SUM.write_text(json.dumps(out, indent=1) + "\n")
    return out


def fig_search6d(S6):
    """Incumbent fitness against the rollouts a search spent: a panel per task, a colour per method, a line per start."""
    import fp_review_figures as FR
    plt = FR.plt_setup()
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 13})
    tasks = [t for t in S6_TASKS if t in S6]
    fig, axs = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks) + 0.6, 4.5), constrained_layout=True,
                            squeeze=False)
    col = dict(zip(S6_METHODS, ("#1f77b4", "#d62728", "#ff7f0e", "#7f7f7f", "#2ca02c", "#9467bd")))
    for ax, task in zip(axs[0], tasks):
        for m in S6_METHODS:
            for k, r in enumerate(S6[task]["runs"].get(m, [])):
                x, y = np.array(r["curve"], float).T
                ax.step(x, y, where="post", color=col[m], lw=1.5, alpha=0.85, label=S6_HEAD[m] if k == 0 else None)
                ax.plot(x[-1], y[-1], "o", color=col[m], ms=5, alpha=0.9)   # where the search ended
        ax.set_title(f"{TNAME[task]}, {OBJ[task]}")
        ax.set_xlabel("Rollouts spent")
        ax.set_xscale("log")
        ax.set_xlim(40, 7500)
        ax.set_xticks([100, 300, 1000, 6000])
        ax.set_xticklabels(["100", "300", "1,000", "6,000"])
        ax.minorticks_off()
        ax.grid(alpha=0.3)
    axs[0][0].set_ylabel("Incumbent's fitness, seeds 0-2")
    fig.legend(*axs[0][0].get_legend_handles_labels(), loc="outside lower center", frameon=False, ncol=6,
               handlelength=1.4, columnspacing=1.2)
    FR.save(fig, S6FIG)


def s6_stats(S6):
    """Per task: the best validated score of any search and, per method, its starts within 0.05 of it."""
    out = {}
    for task in S6_TASKS:
        v = S6.get(task)
        vals = [r["val"] for rs in v["runs"].values() for r in rs if r["val"] is not None] if v else []
        if not vals:
            continue
        top = max(vals)
        win = next(r for rs in v["runs"].values() for r in rs if r["val"] == top)
        per = {m: {"within": sum(1 for r in rs if r["val"] is not None and r["val"] >= top - 0.05), "n": len(rs),
                   "med_val": float(np.median([r["val"] for r in rs if r["val"] is not None])),
                   "med_track": float(np.median([r["val_track"] for r in rs if r["val_track"] is not None]))}
               for m, rs in v["runs"].items() if any(r["val"] is not None for r in rs)}
        out[task] = {"top": top, "win": win, "per": per}
    return out


def mount(mm, k):
    """Gantry k's (x, y) in mm: 0 the thumb, 1 the index, 2 the middle."""
    return f"({mm[2 * k]:.0f}, {mm[2 * k + 1]:.0f})".replace("-", "&#8722;")


def s6_who(S6, task, layout):
    """The starts whose last incumbent is the layout: 'Random (start 0), RRT (start 2)'; Sobol's starts as one."""
    out = []
    for m in S6_METHODS:
        seeds = [r["seed"] for r in S6[task]["runs"].get(m, []) if r["layout"] == layout]
        if seeds:
            out.append(S6_HEAD[m] if m == "sobol" else
                       f"{S6_HEAD[m]} (start{'s' if len(seeds) > 1 else ''} {', '.join(map(str, seeds))})")
    return ", ".join(out)


def s6_also(S6, St):
    """The other methods whose incumbent was each task's best validated layout, as a sentence."""
    names = {"ea64": "the EA", "pso16": "the 16-particle swarm", "pso64r": "the ring swarm", "sobol": "Sobol",
             "rrt": "RRT"}
    parts = []
    for k in St:
        ms = [names[m] for m in names if any(r["layout"] == St[k]["win"]["layout"] for r in S6[k]["runs"].get(m, []))]
        if ms:
            parts.append((" and ".join(ms), k))
    if not parts:
        return ""
    tail = [f"of {m} on the {k}" for m, k in parts]
    text = "It was also an incumbent " + "".join(x + ", " for x in tail[:-2]) + \
        (f"{tail[-2]} and " if len(tail) > 1 else "") + tail[-1]
    return text + "."


def s6_span(S6, m, key):
    v = [r[key] for t in S6 for r in S6[t]["runs"].get(m, [])]
    return min(v), max(v)


S6_OWNER = {"ea64": "the EA's", "pso16": "the 16-particle swarm's", "pso64r": "the ring swarm's",
            "random": "random search's", "sobol": "Sobol's", "rrt": "RRT's"}
WORD = {1: "one", 2: "two", 3: "three"}


def s6_at(S6, task, m, spend):
    """Each start's incumbent fitness when the search had spent at most `spend` rollouts."""
    out = []
    for r in S6[task]["runs"].get(m, []):
        f = [y for x, y in r["curve"] if x <= spend]
        out.append(f[-1] if f else 0.0)
    return out


def rng2(lo, hi, fmt="{:.2f}"):
    a, b = fmt.format(lo), fmt.format(hi)
    return a if a == b else f"{a} to {b}"


def sec_search6d(S6):
    """Post-reset step 4: the six methods live on the six-coordinate Sobol layouts."""
    St = s6_stats(S6)
    if not St:
        return ""
    tb1, tb2, f1 = P.TAB[0] + 1, P.TAB[0] + 2, P.FIG[0] + 1
    t = fill(r"""<p>On the 2-D lattice a full grid was cheap enough to replay every search on stored scores. The
design page's six-coordinate layouts are too many for that, so the six methods ran live on the GPU evaluator (Figure
<<f>>, Table <<tb>>). The space is the design page's 8,192 Sobol layouts: each gantry's $x$ and $y$ for the thumb, the
index and the middle. Each layout has one grasp, the task's foundational pose solved by the pose toy at the default
object. A layout's fitness is that grasp's mean earlier score at seeds 0-2. A layout whose pose is not reached scores 0
at no rollouts. A search moves in coordinates scaled to $[-0.5, 0.5]$ and runs the Sobol layout nearest its point.</p>
<p>Each method ran from three random starts with 6,000 rollouts each, on the SCS0009 plant at 0.72 N&#8201;m/rad and
2.5 MPa with MuJoCo 3.6. The EA kept its lattice settings, with a mutation of a quarter of each coordinate's range. The
swarms' velocities were capped at a fifth of the range, and RRT grew from the CAD-nominal hand with steps of 0.10. The
Sobol order is the same for every start, so its three starts are one search. The searches of a task shared one
evaluation of each layout, and each counted the rollouts of the layouts it visited. A round is one generation of the EA,
one move of a swarm, or the next 64 layouts of random search and Sobol and the next 32 of RRT. A search ended when it had
spent 6,000 rollouts or after 400 rounds. Each search's last incumbent ran again at seeds 20-24, which no search used.
A method is scored against the best of these validated scores.</p>""", f=f1, tb=tb1)
    # outcome
    n9 = {m: sum(St[k]["per"][m]["within"] for k in St if m in St[k]["per"]) for m in S6_METHODS}
    top = {k: f"{St[k]['top']:.2f}" for k in St}
    t += fill(r"""<p>The best validated layout of every task was among random search's incumbents (Table <<tb2>>).
<<also>> These layouts scored <<tw>> on the twist, <<ro>> on the roll and <<pr>> on the press at seeds 20-24. Over the
three tasks, <<rrt>> of RRT's 9 starts came within 0.05 of the best, and <<rnd>> of random search's. Of the EA's starts
<<ea>> did, and <<g16>> of the 16-particle swarm's. On the 2-D lattice the replays had ranked the EA first (Section
SEC[results]).</p>""", tb2=tb2, tw=top["twist"], ro=top["roll"], pr=top["press"], rrt=n9["rrt"], rnd=n9["random"],
              ea=n9["ea64"], g16=n9["pso16"], also=s6_also(S6, St))
    # the stall
    any_task = next(iter(S6.values()))
    eal, ear = s6_span(S6, "ea64", "layouts"), s6_span(S6, "ea64", "rollouts")
    gl, gr = s6_span(S6, "pso16", "layouts"), s6_span(S6, "pso16", "rollouts")
    rr = s6_span(S6, "pso64r", "rollouts")
    tw_ea = S6["twist"]["runs"]["ea64"]
    t += fill(r"""<p>The EA and the 16-particle swarm stalled (Figure <<f>>). The EA's starts ended after 400 rounds
with <<eal0>> to <<eal1>> layouts visited and <<ear0>> to <<ear1>> rollouts spent. The 16-particle swarm's ended with
<<gl0>> to <<gl1>> layouts and <<gr0>> to <<gr1>> rollouts. In the scaled coordinates the median distance from a Sobol
layout to its nearest neighbour is <<nn>>. An EA child differs from its parent by at most 0.125 in each coordinate. For a
random parent, the layout nearest the child was the parent itself in <<snap>> % of 20,000 draws. A layout visited before
costs nothing, so these searches reached the round limit with most of their budget unspent. The ring of 64 particles
ended at <<rr0>> to <<rr1>> rollouts. Random search, Sobol and RRT ran until the 6,000 were spent.</p>
<p>Before the stall, the EA's incumbents were as good as the other methods' at the same rollouts. On the twist the EA's
starts stopped at <<tr0>> to <<tr1>> rollouts, with incumbents of <<tf>> at seeds 0-2. At 600 rollouts the incumbents of
random search scored <<rnd600>>, and those of RRT <<rrt600>>.</p>""", f=f1, eal0=eal[0], eal1=eal[1], ear0=ear[0], ear1=ear[1], gl0=gl[0], gl1=gl[1], gr0=gr[0],
              gr1=gr[1], nn=f"{any_task['spacing']:.2f}", snap=f"{100 * any_task['snap']:.0f}", rr0=f"{rr[0]:,}",
              rr1=f"{rr[1]:,}", tr0=min(r["rollouts"] for r in tw_ea), tr1=max(r["rollouts"] for r in tw_ea),
              tf=rng2(min(r["fit"] for r in tw_ea), max(r["fit"] for r in tw_ea)),
              rnd600=rng2(min(s6_at(S6, "twist", "random", 600)), max(s6_at(S6, "twist", "random", 600))),
              rrt600=rng2(min(s6_at(S6, "twist", "rrt", 600)), max(s6_at(S6, "twist", "rrt", 600))))
    t += img(MEDIA / S6FIG, "Incumbent's fitness against the rollouts each live search spent on the six-coordinate "
             "Sobol layouts: the mean earlier score at seeds 0-2 of the best layout found so far. One line per random "
             "start, on a logarithmic axis; a dot marks where the search ended. Sobol's three starts coincide. "
             "SCS0009 plant at "
             "0.72 N&#8201;m/rad, 2.5 MPa, MuJoCo 3.6. <code>scripts/hom/hom_search6d.py</code>.")
    rows = []
    for task in S6_TASKS:
        if task not in St:
            continue
        for m in S6_METHODS:
            rs = S6[task]["runs"].get(m)
            if not rs:
                continue
            pm = St[task]["per"].get(m)
            rows.append([TNAME[task] if m == S6_METHODS[0] else "", S6_HEAD[m],
                         ", ".join(f"{r['fit']:.2f}" for r in rs),
                         ", ".join("&#8211;" if r["val"] is None else f"{r['val']:.2f}" for r in rs),
                         f"{pm['within']} of {pm['n']}" if pm else "&#8211;",
                         ", ".join("&#8211;" if r["val_track"] is None else f"{r['val_track']:.2f}" for r in rs),
                         f"{int(np.median([r['layouts'] for r in rs])):,}",
                         f"{int(np.median([r['rollouts'] for r in rs])):,}"])
    t += table(["Task", "Method", "Incumbent's fitness, seeds 0-2, starts 0, 1, 2",
                "Earlier score at seeds 20-24", "Starts within 0.05 of the best validated",
                "Tracking score at seeds 20-24", "Layouts visited (median)", "Rollouts spent (median)"], rows,
               "The live searches on the six-coordinate Sobol layouts: each start's last incumbent, its scores at the "
               "validation seeds 20-24, and the starts whose validated earlier score is within 0.05 of the best of any "
               "search. A search counts the rollouts of each layout it visited once.", num_from=2)
    # per task, noise, coverage
    per = {k: St[k]["per"] for k in St}
    tw_n = sum(v["within"] for v in per["twist"].values())
    tw_who = [(per["twist"][m]["within"], m) for m in S6_METHODS if per["twist"].get(m, {}).get("within")]
    tw_who.sort(key=lambda x: -x[0])
    tw_list = ", ".join(f"{WORD[n]} of {S6_OWNER[m]}" for n, m in tw_who[:-1]) + \
        (" and " if len(tw_who) > 1 else "") + f"{WORD[tw_who[-1][0]]} of {S6_OWNER[tw_who[-1][1]]}"
    ro_v = [r["val"] for rs in S6["roll"]["runs"].values() for r in rs]
    allr = [(t_, m, r) for t_ in St for m, rs in S6[t_]["runs"].items() for r in rs]
    drops = sorted(allr, key=lambda x: -(x[2]["fit"] - x[2]["val"]))
    d1, d2 = drops[0], next(x for x in drops[1:] if x[2]["layout"] != drops[0][2]["layout"])
    wins = [St[k]["win"] for k in St]
    dmax = max(abs(w["fit"] - w["val"]) for w in wins)
    ev = {k: S6[k]["evaluated"] for k in St}
    re_ = {k: S6[k]["reached"] for k in St}
    cov = [100 * ev[k] / re_[k] for k in St]
    t += fill(r"""<p>The twist separated the methods most, with <<twn>> of its 18 starts within 0.05 of <<tw>>:
<<twl>>. On the roll the incumbents scored <<ro0>> to <<ro1>> at the validation seeds, and <<ron>> of 18 starts came
within 0.05. On the press <<prn>> of 18 did.</p>
<p>A mean over three seeds favours lucky layouts. The largest drop from seeds 0-2 to seeds 20-24 was <<d1>>, for
<<d1who>> on the <<d1t>>: from <<d1f>> to <<d1v>>. The next was <<d2>>, for <<d2who>> on the <<d2t>>, from <<d2f>> to
<<d2v>>. The best validated layouts of the three tasks changed by at most <<dmax>> between the two seed sets.</p>
<p>Together the searches ran <<cov>> % of the layouts that reach each pose: <<twe>> of <<twr>> on the twist, <<roe>> of
<<ror>> on the roll and <<pre>> of <<prr>> on the press. Their union is close to a full grid of the Sobol layouts at
seeds 0-2.</p>""", twn=tw_n, tw=top["twist"], twl=tw_list,
              ro0=f"{min(ro_v):.2f}", ro1=f"{max(ro_v):.2f}", ron=sum(v["within"] for v in per["roll"].values()),
              prn=sum(v["within"] for v in per["press"].values()),
              d1=f"{d1[2]['fit'] - d1[2]['val']:.2f}", d1who=f"start {d1[2]['seed']} of {S6_OWNER[d1[1]][:-2]}",
              d1t=d1[0], d1f=f"{d1[2]['fit']:.2f}", d1v=f"{d1[2]['val']:.2f}",
              d2=f"{d2[2]['fit'] - d2[2]['val']:.2f}", d2who=f"start {d2[2]['seed']} of {S6_OWNER[d2[1]][:-2]}",
              d2t=d2[0], d2f=f"{d2[2]['fit']:.2f}", d2v=f"{d2[2]['val']:.2f}", dmax=f"{dmax:.2f}",
              cov=rng2(min(cov), max(cov), "{:.0f}"), twe=f"{ev['twist']:,}", twr=f"{re_['twist']:,}",
              roe=f"{ev['roll']:,}", ror=f"{re_['roll']:,}", pre=f"{ev['press']:,}", prr=f"{re_['press']:,}")
    # the tracking score
    ro_t = [r["val_track"] for rs in S6["roll"]["runs"].values() for r in rs]
    pr_t = [r["val_track"] for rs in S6["press"]["runs"].values() for r in rs]
    tt = {k: S6[k]["track_top"][0] for k in St}
    ro_inc = {r["layout"] for rs in S6["roll"]["runs"].values() for r in rs}
    bt = {k: max((r for rs in S6[k]["runs"].values() for r in rs), key=lambda r: r["val_track"]) for k in St}
    tw_w = St["twist"]["win"]
    t += fill(r"""<p>By the tracking score no roll incumbent rolls the sphere: each tracked at <<rt0>> to <<rt1>> at
seeds 20-24. Over the <<roe>> roll layouts that ran, Spearman's rank correlation between the two scores at seeds 0-2 is
<<rrho>>. On the twist it is <<trho>>, and on the press <<prho>>. The best roll layout by the tracking score at seeds 0-2,
layout <<rtl>>, tracked at <<rtt>> there with an earlier score of <<rtte>>. It was <<rinc>>. On the twist the best validated
layout tracked at <<twt>>. Layout <<c2>>, <<gap>> below it by the earlier score, tracked at <<c2t>>. It is also the best
of the <<twe>> twist layouts by the tracking score at seeds 0-2. On the press the incumbents tracked at <<pt0>> to
<<pt1>>.</p>""", rt0=f"{min(ro_t):.2f}", rt1=f"{max(ro_t):.2f}", roe=f"{ev['roll']:,}", rrho=f"{S6['roll']['rho']:.2f}",
              trho=f"{S6['twist']['rho']:.2f}", prho=f"{S6['press']['rho']:.2f}", rtl=tt["roll"]["layout"],
              rtt=f"{tt['roll']['fit_track']:.2f}",
              rtte=f"{tt['roll']['fit']:.2f}",
              rinc="no search's incumbent" if tt["roll"]["layout"] not in ro_inc else "a search's incumbent",
              twt=f"{tw_w['val_track']:.2f}", c2=bt["twist"]["layout"],
              gap=f"{St['twist']['top'] - bt['twist']['val']:.2f}", c2t=f"{bt['twist']['val_track']:.2f}",
              twe=f"{ev['twist']:,}", pt0=f"{min(pr_t):.2f}", pt1=f"{max(pr_t):.2f}")
    rows = []
    for k in St:
        picks = [("Best earlier score", St[k]["win"]), ("Best tracking score", bt[k])]
        for j, (what, r) in enumerate(picks):
            rows.append([TNAME[k] if j == 0 else "", what, str(r["layout"]), s6_who(S6, k, r["layout"]),
                         mount(r["mm"], 0), mount(r["mm"], 1), mount(r["mm"], 2), f"{r['val']:.2f}",
                         f"{r['val_track']:.2f}"])
    t += table(["Task", "Incumbent", "Layout", "Found by", "Thumb mount (mm)", "Index mount (mm)",
                "Middle mount (mm)", "Earlier score at seeds 20-24", "Tracking score at seeds 20-24"], rows,
               "Each task's incumbent with the best validated earlier score and the one with the best validated "
               "tracking score. Layout: its index in the Sobol draw order. Mount: the gantry's (<i>x</i>, <i>y</i>) "
               "on the palm; the CAD-nominal hand has the thumb at (&#8722;50, 0), the index at (50, 55) and the middle "
               "at (50, &#8722;55).", num_from=4)
    return t + sec_track_top(S6, St)


def sec_track_top(S6, St):
    """Post-reset step 18: the five layouts of best tracking score at seeds 0-2 per task, rerun at seeds 20-24."""
    tops = {k: S6[k]["track_top"] for k in St}
    if not all(e["val_track"] is not None and e.get("rms") for k in tops for e in tops[k]):
        return ""
    tb = P.TAB[0] + 1
    ro = tops["roll"]
    b = max(ro, key=lambda e: e["val_track"])
    w = St["roll"]["win"]
    vt = {k: [e["val_track"] for e in tops[k]] for k in tops}
    whole = b["rms"]["held"] == 5 and b["rms"]["contact_all"] == 1.0
    inc = [(k, e["layout"]) for k in tops for e in tops[k] if s6_who(S6, k, e["layout"])]
    t = fill(r"""<p>The five layouts of best tracking score at seeds 0-2 of each task ran again at seeds 20-24 (Table
<<tb>>). On the roll they tracked at <<r0>> to <<r1>> there. Layout <<c>> tracked best<<again>>. <<held>> Its RMS angle error was
<<e0>> to <<e1>>&#176; against a commanded RMS rotation of <<ref>>&#176;. Its earlier score was <<cv>>, <<gap>> below that
of the best validated roll layout, <<w>>. Layout <<w>> followed the rotation more closely, with an RMS error of <<we0>> to
<<we1>>&#176;. <<wc>> On the twist the five tracked at <<t0>> to <<t1>>, and on the press at <<p0>> to <<p1>>.
<<inc>></p>""", tb=tb, r0=f"{min(vt['roll']):.2f}", r1=f"{max(vt['roll']):.2f}", c=b["layout"],
             again=", as at seeds 0-2" if b["layout"] == ro[0]["layout"] else "",
             held=("It held the sphere at all five seeds with all three pads loaded throughout." if whole else
                   f"It held the sphere at {b['rms']['held']} of five seeds, with all three pads loaded for "
                   f"{100 * b['rms']['contact_all']:.0f} % of the task."),
             e0=f"{b['rms']['err_deg'][0]:.1f}", e1=f"{b['rms']['err_deg'][1]:.1f}", ref=f"{b['rms']['ref_deg']:.1f}",
             cv=f"{b['val']:.2f}", gap=f"{St['roll']['top'] - b['val']:.2f}", w=w["layout"],
             we0=f"{w['rms']['err_deg'][0]:.1f}", we1=f"{w['rms']['err_deg'][1]:.1f}",
             wc=("At no time were all three of its pads loaded, so it tracked at 0." if w["rms"]["contact_all"] == 0
                 else f"All three of its pads were loaded for {100 * w['rms']['contact_all']:.0f} % of the task, and it "
                 f"tracked at {w['val_track']:.2f}."),
             t0=f"{min(vt['twist']):.2f}", t1=f"{max(vt['twist']):.2f}", p0=f"{min(vt['press']):.2f}",
             p1=f"{max(vt['press']):.2f}",
             inc=(f"{WORD.get(len(inc), len(inc)).capitalize()} of the {3 * 5} had been a search's incumbent: "
                  + " and ".join(f"layout {c} on the {k}" for k, c in inc) + "." if inc else
                  "None of them had been a search's incumbent."))
    rows = []
    for k in tops:
        for j, e in enumerate(tops[k]):
            rows.append([TNAME[k] if j == 0 else "", str(e["layout"]), s6_who(S6, k, e["layout"]) or "&#8211;",
                         mount(e["mm"], 0), mount(e["mm"], 1), mount(e["mm"], 2), f"{e['fit_track']:.2f}",
                         f"{e['val_track']:.2f}", f"{e['fit']:.2f}", f"{e['val']:.2f}"])
    t += table(["Task", "Layout", "Incumbent of", "Thumb mount (mm)", "Index mount (mm)", "Middle mount (mm)",
                "Tracking score at seeds 0-2", "Tracking score at seeds 20-24", "Earlier score at seeds 0-2",
                "Earlier score at seeds 20-24"], rows,
               "Per task, the five layouts of best tracking score at seeds 0-2 among all layouts the live searches ran on "
               "the six-coordinate Sobol layouts, rerun at seeds 20-24 (<code>hom_search6d.py validate --track-top "
               "5</code>). Incumbent of: the starts whose last incumbent the layout was. Mount: the gantry's "
               "(<i>x</i>, <i>y</i>) on the palm.", num_from=6)
    return t


# ------------------------------------------------------------------------------------------------ live searches, new sizes
S5_SIZES = {"twist": (28.0, 67.0), "roll": (28.0, 67.0), "press": (18.0, 45.0)}
S5_DIRS = {"ea": ROWS / "ea_live_sv25", "pso": ROWS / "pso16_live_sv25"}
S5_HEAD = {"ea": "EA 64", "pso": "PSO 16"}
S5SUM = ROWS / "20261110-live_sizes_sv25_summary.json"
S5FIG = "20261110-live_sizes_curves.png"


def lattice_call(xy):
    """The 2-D lattice's index of (x_sep, y_sep) in mm (hom_plant_transfer.xy inverted)."""
    return int(round((xy[0] - 40) / 5)) * 25 + int(round((xy[1] - 50) / 5))


def s5_summary():
    """Post-reset step 5 (hom_ea.py run --method ea|pso, three starts each, at sizes the size study did not run): per
    task and size, each search's curve, last incumbent and the rollouts it had spent when that layout first ran; the
    union of the layouts the six searches ran; the spread of a shared layout's score between searches (hom_gpu seeds the
    per-period noise per batch); and the validation at seeds 20-24 (hom_ea_validate.py): each layout's best grasp's
    mean earlier and tracking score. Written to S5SUM where the rows are, read from it elsewhere."""
    import itertools

    import hom_plant_transfer as TR
    if not any((S5_DIRS["ea"] / f"{t}_d{d:g}_s0.jsonl").exists() for t, ds in S5_SIZES.items() for d in ds):
        return json.loads(S5SUM.read_text()) if S5SUM.exists() else {}
    out = {}
    for task, ds in S5_SIZES.items():
        for d in ds:
            runs, own = {}, {}
            for m, dr in S5_DIRS.items():
                for seed in (0, 1, 2):
                    gp = dr / f"{task}_d{d:g}_s{seed}_gens.jsonl"
                    if not gp.exists() or not gp.stat().st_size:
                        continue
                    log = [json.loads(x) for x in gp.read_text().splitlines() if x.strip()]
                    rp = dr / f"{task}_d{d:g}_s{seed}.jsonl"
                    ge, gt = TR.grasp_means(task, rp, "rollout"), TR.grasp_means(task, rp, "track")
                    own[(m, seed)] = {c: max(v.values()) for c, v in ge.items()}
                    inc = log[-1]["incumbent"]
                    ci = lattice_call(inc)
                    gi = max(ge[ci], key=ge[ci].get)   # the grasp behind the search's fitness
                    runs.setdefault(m, []).append({
                        "seed": seed, "xy": [float(v) for v in inc], "fit": float(log[-1]["incumbent_fit"]),
                        "fit_track": gt[ci][gi],
                        "rollouts": int(log[-1]["rollouts"]), "layouts": int(log[-1]["layouts"]), "gens": len(log) - 1,
                        "first": next(int(g["rollouts"]) for g in log if g["incumbent"] == inc), "curve": curve(log)})
            if not runs:
                continue
            diffs = [abs(own[a][c] - own[b][c]) for a, b in itertools.combinations(own, 2)
                     for c in set(own[a]) & set(own[b])]
            union = set().union(*own.values())
            vp = S5_DIRS["ea"] / f"{task}_d{d:g}_val.jsonl"
            ve = {c: max(v.values()) for c, v in TR.grasp_means(task, vp, "rollout").items()}
            vt = {c: max(v.values()) for c, v in TR.grasp_means(task, vp, "track").items()}
            best = max(ve.values()) if ve else None
            for rs in runs.values():
                for r in rs:
                    c = lattice_call(r["xy"])
                    r["val"], r["val_track"] = ve.get(c), vt.get(c)
                    r["within"] = None if best is None or c not in ve else bool(ve[c] >= best - 0.05)
            bc = max(ve, key=ve.get) if ve else None
            bt = max(vt, key=vt.get) if vt else None
            out[f"{task}_{d:g}"] = {
                "task": task, "d": d, "runs": runs, "layouts": len(union),
                "shared": {"pairs": len(diffs), "over_0.02": int(sum(x > 0.02 for x in diffs)),
                           "median": float(np.median(diffs)) if diffs else None,
                           "max": float(max(diffs)) if diffs else None},
                "validated": len(ve), "best": best, "best_xy": list(TR.xy(bc)) if bc is not None else None,
                "best_track_of_best": vt.get(bc) if bc is not None else None,
                "best_track": vt[bt] if bt is not None else None,
                "best_track_xy": list(TR.xy(bt)) if bt is not None else None,
                "best_track_fit": ve.get(bt) if bt is not None else None}
    if out:
        S5SUM.write_text(json.dumps(out, indent=1) + "\n")
    return out


def s5_keys():
    return [f"{t}_{d:g}" for t, ds in S5_SIZES.items() for d in ds]


def s5_ok(S5):
    """Every task and size has its six searches and the validation of their incumbents."""
    return all(k in S5 and S5[k].get("best") is not None and all(len(S5[k]["runs"].get(m, [])) == 3 for m in S5_DIRS)
               and all(r["val"] is not None for rs in S5[k]["runs"].values() for r in rs) for k in s5_keys())


def s5_obj(v):
    return f"{v['d']:g} mm {'cylinder' if v['task'] == 'press' else 'sphere'}"


def fig_newsizes(S5):
    """Incumbent fitness against rollouts for the EA and the 16-particle swarm, a panel per task and size, with each
    incumbent's validated score at the search's end."""
    import fp_review_figures as FR
    plt = FR.plt_setup()
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 14})
    fig, axs = plt.subplots(2, 3, figsize=(11.4, 7.6), constrained_layout=True, squeeze=False)
    col = {"ea": "#1f77b4", "pso": "#d62728"}
    for j, task in enumerate(S5_SIZES):
        for i, d in enumerate(S5_SIZES[task]):
            ax, v = axs[i][j], S5.get(f"{task}_{d:g}")
            ax.set_title(f"{TNAME[task]}, {d:g} mm {'cylinder' if task == 'press' else 'sphere'}")
            ax.set_xlim(0, 6800)
            ax.set_xticks([0, 2000, 4000, 6000])
            ax.set_xticklabels(["0", "2,000", "4,000", "6,000"])
            ax.grid(alpha=0.3)
            if not v:
                continue
            for m in S5_DIRS:
                for r in v["runs"].get(m, []):
                    x, y = np.array(r["curve"] + [[r["rollouts"], r["curve"][-1][1]]], float).T
                    ax.step(x, y, where="post", color=col[m], lw=1.6, alpha=0.85)
                    if r.get("val") is not None:
                        ax.plot(r["rollouts"], r["val"], "o", mfc="white", mec=col[m], mew=1.6, ms=7)
            if v.get("best") is not None:
                ax.axhline(v["best"] - 0.05, color="k", lw=1, ls=":")
    for ax in axs[-1]:
        ax.set_xlabel("Rollouts spent")
    for row in axs:
        row[0].set_ylabel("Incumbent's fitness")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=col[m], lw=1.6) for m in S5_DIRS] + [
        Line2D([], [], ls="none", marker="o", mfc="white", mec="k", mew=1.6, ms=7),
        Line2D([], [], color="k", lw=1, ls=":")]
    fig.legend(hs, [S5_HEAD[m] for m in S5_DIRS] + ["Last incumbent at seeds 20-24", "0.05 below the best validated"],
               loc="outside lower center", frameon=False, ncol=4)
    FR.save(fig, S5FIG)


def sec_newsizes(S, S5):
    """Post-reset step 5: the EA and the 16-particle swarm live at object sizes without a stored grid."""
    if not s5_ok(S5):
        return ""
    keys = s5_keys()
    tb1, tb2, f1 = P.TAB[0] + 1, P.TAB[0] + 2, P.FIG[0] + 1
    allr = {m: [(k, r) for k in keys for r in S5[k]["runs"][m]] for m in S5_DIRS}
    nw = {m: sum(bool(r["within"]) for _k, r in allr[m]) for m in S5_DIRS}
    n02 = sum(S5[k]["shared"]["over_0.02"] for k in keys)
    npairs = sum(S5[k]["shared"]["pairs"] for k in keys)
    dmax = max(S5[k]["shared"]["max"] or 0.0 for k in keys)
    per = {t_: {m: sum(bool(r["within"]) for k, r in allr[m] if S5[k]["task"] == t_) for m in S5_DIRS} for t_ in S5_SIZES}

    def did(t_):
        e, p = per[t_]["ea"], per[t_]["pso"]
        return f"each did so from {e} of 6" if e == p else f"the EA did so from {e} of 6 starts and the swarm from {p}"
    split = [t_ for t_ in S5_SIZES if per[t_]["ea"] != per[t_]["pso"]]
    same = [t_ for t_ in S5_SIZES if t_ not in split]
    rep = {t_: (pc(S, "scs0009", t_, "ea64"), pc(S, "scs0009", t_, "pso_g16")) for t_ in S5_SIZES}
    rest = [v for t_ in S5_SIZES if t_ != "roll" for v in rep[t_]]
    t = fill(r"""<p>At four object sizes without a stored grid, the EA came within 0.05 of the best validated layout
from <<ea>> of its 18 starts. The 16-particle swarm did so from <<pso>> (Figure <<f>>, Table <<tb>>). <<split>>
<<same>> In the replays at the default objects the roll separated the two most: the EA came within 0.05 by 6,000
rollouts in <<rea>> % of runs and the swarm in <<rpso>> %. On the twist and the press both did in <<rrest>> % of
runs.</p>""", ea=nw["ea"], pso=nw["pso"], f=f1, tb=tb1,
              split=" ".join(f"On the {t_} {did(t_)}." for t_ in split),
              same=("On the " + " and the ".join(same) + f" each did so from {per[same[0]]['ea']} of 6."
                    if same and len({per[t_]["ea"] for t_ in same}) == 1 else
                    " ".join(f"On the {t_} {did(t_)}." for t_ in same)),
              rea=pct(rep["roll"][0]), rpso=pct(rep["roll"][1]), rrest=span_pc(rest))
    t += fill(r"""
<p>The replays of Section SEC[results] ran on stored grids at the default objects. At a new object size no grid exists,
so a search runs its layouts as it visits them. The EA with 64 individuals and the 16-particle swarm with the global best
ran live on the GPU evaluator. The objects were of sizes that the size study of the trends page did not run. The twist
and the roll ran on spheres of 28 and 67 mm, and the press on cylinders of 18 and 45 mm. Each method ran from three
random starts on the 2-D lattice, with each layout's grasp set solved by the pose toy at that size. A layout's fitness was its best grasp's mean
earlier score at seeds 0-2, as in the replays. A search stopped after the generation, or the swarm's iteration, in which
its rollouts reached 6,000, or after 40 of them. The plant was the SCS0009 plant at 0.72 N&#8201;m/rad with pads at
2.5 MPa, on MuJoCo 3.6 (<code>hom_ea.py run --method ea|pso</code>).</p>
<p>The evaluator draws each control period's measurement noise from a generator seeded per GPU batch, so a layout's
score depends on its place in the batch. Two searches that ran the same layout scored it differently. At the six tasks
and sizes, <<n02>> of <<np>> pairs of scores for a shared layout differed by more than 0.02. The largest difference was
<<dmax>>. Each search's last incumbent therefore ran again at seeds 20-24, which no search used, with its whole
grasp set (<code>hom_ea_validate.py</code>). So did the five layouts of best earlier score and the five of best tracking
score at seeds 0-2 among all layouts the six searches ran. A start counts as within 0.05 when its incumbent's validated
earlier score is within 0.05 of the best validated score at its task and size.</p>""",
              n02=f"{n02:,}", np=f"{npairs:,}", dmax=f"{dmax:.2f}")
    t += img(MEDIA / S5FIG, "Incumbent's fitness against the rollouts each live search spent at the four new object "
             "sizes. The fitness is the best-of-16 mean earlier score at seeds 0-2 of the best layout found so far, one "
             "line per random start. An open circle marks each search's last incumbent at its earlier score at seeds 20-24; the dotted "
             "line lies 0.05 below the best validated score. SCS0009 plant at 0.72 N&#8201;m/rad, 2.5 MPa, MuJoCo 3.6. "
             "<code>scripts/hom/hom_ea.py</code>.")
    # the largest drop from seeds 0-2 to seeds 20-24, and the searches that stopped at the round limit
    who = {"ea": "the EA", "pso": "the swarm"}
    dk, dm, dr = max(((k, m, r) for m in S5_DIRS for k, r in allr[m]), key=lambda x: x[2]["fit"] - x[2]["val"])
    same_xy = {m: [r for r in S5[dk]["runs"][m] if r["xy"] == dr["xy"]] for m in S5_DIRS}
    dwho = " and ".join((f"{WORD[len(rs)]} of {who[m]}'s starts" if len(rs) > 1 else f"start {rs[0]['seed']} of {who[m]}")
                        for m, rs in same_xy.items() if rs)
    dft = [r["fit_track"] for rs in same_xy.values() for r in rs]
    stop = {m: [r for _k, r in allr[m] if r["gens"] >= 40 and r["rollouts"] < 6000] for m in S5_DIRS}
    sp, se = stop["pso"], stop["ea"]
    t += fill(r"""<p>As on the six-coordinate layouts, a mean over seeds 0-2 favoured lucky layouts. The largest drop
was at the <<dk>>, where <<dwho>> ended on (<<dx>>, <<dy>>). That layout scored <<df>> at seeds 0-2 and <<dv>> at
seeds 20-24. <<twdef>> The grasp behind its <<df>> tracked at <<dft>> at seeds 0-2.</p>
<p>The swarm stopped after 40 iterations in <<nsp>> of its 18 starts, with <<sp0>> to <<sp1>> rollouts spent. In those
starts it ran <<lp0>> to <<lp1>> layouts. An iteration costs only the layouts that the particles had not run before.
<<se>></p>""", dk=f"{S5[dk]['d']:g} mm {S5[dk]['task']}", dwho=dwho, dx=f"{dr['xy'][0]:.0f}", dy=f"{dr['xy'][1]:.0f}",
              df=f"{dr['fit']:.2f}", dv=f"{dr['val']:.2f}",
              twdef=("The twist's earlier score is the largest rotation reached on three loaded pads during the ramp, as "
                     "a share of 90&#176;." if S5[dk]["task"] == "twist" else ""),
              dft=rng2(min(dft), max(dft)), nsp=len(sp), sp0=f"{min(r['rollouts'] for r in sp):,}",
              sp1=f"{max(r['rollouts'] for r in sp):,}", lp0=min(r["layouts"] for r in sp),
              lp1=max(r["layouts"] for r in sp),
              se=("The EA stopped after 40 generations in one start, with " + f"{se[0]['rollouts']:,} rollouts."
                  if len(se) == 1 else f"The EA stopped after 40 generations in {len(se)} starts." if se else
                  "No EA start reached 40 generations."))
    rows = []
    for k in keys:
        v = S5[k]
        for m in S5_DIRS:
            rs = sorted(v["runs"][m], key=lambda r: r["seed"])
            rows.append([TNAME[v["task"]] + ", " + s5_obj(v) if m == "ea" else "", S5_HEAD[m],
                         ", ".join(f"{r['fit']:.2f}" for r in rs), ", ".join(f"{r['val']:.2f}" for r in rs),
                         f"{sum(bool(r['within']) for r in rs)} of 3",
                         ", ".join(f"{r['val_track']:.2f}" for r in rs),
                         f"{int(np.median([r['rollouts'] for r in rs])):,}",
                         f"{int(np.median([r['first'] for r in rs])):,}"])
    t += table(["Task and object", "Method", "Incumbent's fitness, seeds 0-2, starts 0, 1, 2",
                "Earlier score at seeds 20-24", "Starts within 0.05 of the best validated",
                "Tracking score at seeds 20-24", "Rollouts spent (median)",
                "Rollouts when the last incumbent ran (median)"], rows,
               "The live searches at the new object sizes: each start's last incumbent, its scores at the validation "
               "seeds 20-24, and the starts whose validated earlier score is within 0.05 of the best at that task and "
               "size. A search counts the rollouts of each layout it ran once.", num_from=2)
    # the tracking score of the incumbents against the best validated tracking layout
    gap = {k: S5[k]["best_track"] - max(r["val_track"] for rs in S5[k]["runs"].values() for r in rs) for k in keys}
    tr_keys = [k for k in keys if S5[k]["task"] != "press"]
    pr_keys = [k for k in keys if S5[k]["task"] == "press"]
    inc_any = [k for k in tr_keys if any(r["xy"] == S5[k]["best_track_xy"] for rs in S5[k]["runs"].values() for r in rs)]
    t += fill(r"""<p>The searches ranked layouts by the earlier score. By the tracking score the best incumbent at each
twist and roll size tracked <<g0>> to <<g1>> below the best validated layout there (Table <<tb2>>). <<inc>> On the
press the best incumbent tracked within <<gp>> of the best validated layout.</p>""",
              g0=f"{min(gap[k] for k in tr_keys):.2f}", g1=f"{max(gap[k] for k in tr_keys):.2f}", tb2=tb2,
              inc=("None of those four layouts was a search's incumbent." if not inc_any else
                   f"{len(inc_any)} of those four layouts were a search's incumbent."),
              gp=f"{max(gap[k] for k in pr_keys):.2f}")
    rows = []
    for k in keys:
        v = S5[k]
        rows.append([TNAME[v["task"]] + ", " + s5_obj(v), f"{v['layouts']:,}",
                     "({:.0f}, {:.0f})".format(*v["best_xy"]), f"{v['best']:.2f}", f"{v['best_track_of_best']:.2f}",
                     "({:.0f}, {:.0f})".format(*v["best_track_xy"]), f"{v['best_track']:.2f}",
                     f"{v['best_track_fit']:.2f}"])
    t += table(["Task and object", "Layouts run", "Best validated layout (mm)", "Its earlier score",
                "Its tracking score", "Best validated tracking layout (mm)", "Its tracking score", "Its earlier score"],
               rows, "Per task and size: the layouts the six searches ran, and the validated layouts of best earlier and "
               "of best tracking score at seeds 20-24. A layout is ($x_{sep}$, $y_{sep}$).", num_from=1)
    return t


def sec_limits(S5):
    nv = [S5[k]["validated"] for k in s5_keys()] if s5_ok(S5) else []
    return r"""<ul>
<li>The comparison runs on one 2-D lattice of 625 layouts, at the default objects, on stored scores at seeds 0-2. A
replay draws no new noise, so a layout keeps its grid score however often it is visited.</li>
<li>Each method ran with standard settings: the EA's of the trends page, Clerc and Kennedy's coefficients for PSO, and
the design page's RRT step. None was tuned on these grids, and on the six-coordinate layouts the EA's mutation and the
swarms' velocity cap were not scaled to the layouts' spacing.</li>
<li>The live searches on the six-coordinate layouts ran once per method and start, with one grasp per layout, at the
default objects. Their fitness was the earlier score.</li>
""" + (fill(r"""<li>The live searches at new object sizes ran three starts per method on the 2-D lattice, with the earlier
score as fitness. No grid exists at those sizes, so a start was scored against the best of the <<v0>> to <<v1>> layouts
rerun at seeds 20-24 at its task and size.</li>
""", v0=min(nv), v1=max(nv)) if nv else "") + r"""<li>The plants are the work plant and the SCS0009 plant at 0.72 N&#8201;m/rad. The servos measured on the hand on
2026-11-06 give 0.55 N&#8201;m/rad at the datasheet's torque. Grids on that plant, with and without its dead band, and
on MuJoCo 4.0 are on the <a href="20261110-hom_servo_stiffness_mujoco4.html">stiffness page</a>; the methods were not
compared on them.</li>
<li>All grids ran on the MuJoCo and MuJoCo-Warp versions of 2026-10-30 to 11-02, and the live searches on MuJoCo 3.6.</li>
</ul>"""


def sec_future(S5):
    return r"""<ul>
<li>The EA and the swarms on the six-coordinate layouts with RRT's rule, a move to the nearest layout not yet visited, so
that each spends its 6,000 rollouts (<code>hom_search6d.Search</code>). The replays' ranking of the EA fails on these
layouts if it then comes within 0.05 from fewer starts than random search.</li>
""" + (r"""<li>The live searches at the four new object sizes with the tracking score as fitness (<code>hom_ea.py run
--score track</code>), three starts per method and validated as in Section SEC[newsizes]. The EA's lead fails there if
the swarm comes within 0.05 of the best validated tracking score from more starts than the EA.</li>
<li>The evaluator's per-period measurement noise seeded by layout, grasp and seed in place of the batch's start
(<code>gen_seed</code> of <code>hom_gpu.Batch</code>, set to the index of the batch's first world by
<code>hom_ea.py</code>), so that a layout scores the same in every batch. If two searches then still score a shared
layout more than 0.02 apart, the differences have another source.</li>
""" if s5_ok(S5) else r"""<li>Live EA and PSO searches at new object sizes, three random starts each, to test whether the
replay's ranking holds with fresh rollout noise.</li>
""") + r"""<li>The replays on the grids of the measured servo (<code>rows/grid2dff_v4b_sv55db</code>, stiffness page) with
<code>hom_search_compare.py</code>. The EA's ranking there fails if another method comes within 0.05 in 10 percentage
points more runs on any task.</li>
</ul>"""


def sec_files(S5):
    return r"""<ul>
<li>Comparison and figures: <code>scripts/hom/hom_search_compare.py</code>; rows
<code>rows/20261109-search_compare.tsv</code>, <code>rows/20261109-search_compare_curves.tsv</code>,
<code>rows/20261109-pareto_twist_press.tsv</code>.</li>
<li>Grids: <code>rows/grid2dff_v4b/</code> (work plant) and <code>rows/grid2dff_v4b_sv25/</code> (SCS0009 plant);
the EA: <code>hom_landscape.ea_search</code>; the live searches: <code>scripts/hom/hom_ea.py</code>; the RRT:
<code>fp_sr2_toy.rrt</code>; the CEM: <code>scripts/hom/hom_grasp_refine.py</code>.</li>
<li>Live searches on the six-coordinate layouts: <code>scripts/hom/hom_search6d.py</code>; rows
<code>rows/search6d_sv25/</code>, each search's rounds (<code>*_gens.jsonl</code>), the validation
(<code>*_val.jsonl</code>) and the timings in git, the shared rollouts (<code>twist_d40.jsonl</code>,
<code>roll_d40.jsonl</code>, <code>press_d25.jsonl</code>, 17-22 MB) on the workstation; summary
<code>rows/20261110-search6d_sv25_summary.json</code>; queues
<code>logs/20261110-hom_post_reset/gpu_queue_step4.sh</code> and <code>gpu_queue_step4b.sh</code>.</li>
""" + (r"""<li>Live searches at new object sizes: <code>scripts/hom/hom_ea.py</code> and the validation
<code>scripts/hom/hom_ea_validate.py</code>; rows <code>rows/ea_live_sv25/</code> and
<code>rows/pso16_live_sv25/</code>, each search's generations (<code>*_gens.jsonl</code>), the validation
(<code>ea_live_sv25/*_val.jsonl</code>) and the timings in git, each search's rollouts (5-6 MB a search) on the
workstation; summary <code>rows/20261110-live_sizes_sv25_summary.json</code>; queues
<code>logs/20261110-hom_post_reset/gpu_queue_step5.sh</code> and <code>gpu_queue_step5b.sh</code>.</li>
""" if s5_ok(S5) else "") + r"""<li>This page: <code>scripts/hom/hom_search_methods_page.py</code>.</li>
</ul>"""


# ------------------------------------------------------------------------------------------------ assemble
def lede(S, S6, S5):
    e = pct(pc(S, "scs0009", "roll", "ea64"))
    pso = span_pc([pc(S, "scs0009", "roll", m) for m in ("pso_g16", "pso_g64", "pso_r64")])
    g16 = span_pc([pc(S, "scs0009", t, "pso_g16") for t in ("twist", "pinpush", "press")])
    text = fill("""Of six ways to search the SR2 hand's 625-layout lattice, the evolutionary algorithm (EA) of Yang et al.
with 64 individuals, which our size study used, was the most reliable. Replayed 200 times on the stored grids of the
SCS0009 plant, it came within 0.05 of the roll's best score by 6,000 rollouts in <<e>> % of runs. Particle swarm
optimization (PSO) did so in <<pso>> % and random search in <<rnd>> %. A Sobol sequence did so in <<sob>> % and the FP
paper's RRT in <<rrt>> %. On the other three tasks a 16-particle swarm came within 0.05 in <<g16>> % of runs, as the EA
did.""", e=e, pso=pso,
                rnd=pct(pc(S, "scs0009", "roll", "random")), sob=pct(pc(S, "scs0009", "roll", "sobol")),
                rrt=pct(pc(S, "scs0009", "roll", "rrt")), g16=g16)
    St = s6_stats(S6)
    if St:
        n9 = {m: sum(St[k]["per"][m]["within"] for k in St if m in St[k]["per"]) for m in S6_METHODS}
        r0 = min(s6_span(S6, m, "rollouts")[0] for m in ("ea64", "pso16"))
        r1 = max(s6_span(S6, m, "rollouts")[1] for m in ("ea64", "pso16"))
        text += fill(""" On 8,192 Sobol layouts in six mount coordinates, searched live on the GPU, the EA and the 16-particle
swarm stalled after <<r0>> to <<r1>> of their 6,000 rollouts. Their children and particles fell mostly on layouts already run.
RRT came within 0.05 of each task's best validated layout from <<rrt>> of 9 starts, random search from <<rnd>> and the
EA from <<ea>>.""", r0=r0, r1=r1, rrt=n9["rrt"], rnd=n9["random"], ea=n9["ea64"])
    live = ""
    if s5_ok(S5):
        nw = {m: sum(bool(r["within"]) for k in s5_keys() for r in S5[k]["runs"][m]) for m in S5_DIRS}
        tw = {m: sum(bool(r["within"]) for k in s5_keys() if S5[k]["task"] == "twist" for r in S5[k]["runs"][m])
              for m in S5_DIRS}
        text += fill(""" In live searches at four object sizes without a stored grid, the EA came within 0.05 of the best
validated layout from <<ea>> of 18 starts. The 16-particle swarm did so from <<pso>>, and the two differed on the twist
alone, with <<tea>> and <<tpso>> of 6 starts.""", ea=nw["ea"], pso=nw["pso"], tea=tw["ea"], tpso=tw["pso"])
        live = (f" Live at four new object sizes the EA came within 0.05 of the best validated layout from {nw['ea']} of "
                f"18 starts and the 16-particle swarm from {nw['pso']}")
    summary = (f"On the SR2 hand's 625-layout lattice the EA of Yang et al. with 64 individuals was the most reliable of "
               f"six search methods: on the SCS0009 roll it came within 0.05 of the grid's best by 6,000 rollouts in {e} % "
               f"of 200 replays, against {pso} % for particle swarms, {pct(pc(S, 'scs0009', 'roll', 'random'))} % for "
               f"random search and {pct(pc(S, 'scs0009', 'roll', 'rrt'))} % for the FP paper's RRT. On the "
               "six-coordinate Sobol layouts, searched live, the EA and the 16-particle swarm stalled on layouts already "
               "run, and RRT and random search found the best ones" + ("." + live if live else "") +
               "; first-principles notes on the EA, PSO, CEM, Pareto fronts and t-SNE.")
    return text, summary


def main():
    S, F = load()
    S6 = s6_summary()
    if s6_stats(S6):
        fig_search6d(S6)
    S5 = s5_summary()
    if s5_ok(S5):
        fig_newsizes(S5)
    sections = [("terms", "Terms and metrics", sec_terms()),
                ("problem", "Searching the layout lattice", sec_problem()),
                ("ea", "The evolutionary algorithm of Yang et al.", sec_ea()),
                ("pso", "Particle swarm optimization", sec_pso()),
                ("sampling", "Random search, the Sobol sequence and the RRT of the FP paper", sec_sampling()),
                ("results", "The six methods on the stored grids", sec_results(S)),
                ("refine", "Tuning the grasp and the controller per layout", sec_refine()),
                ("pareto", "Pareto fronts", sec_pareto(F)),
                ("tsne", "t-SNE maps", sec_tsne()),
                ("search6d", "Live searches over the six-coordinate layouts", sec_search6d(S6)),
                ("newsizes", "Live searches at object sizes without a stored grid", sec_newsizes(S, S5)),
                ("limits", "Limitations", sec_limits(S5)),
                ("future", "Future work", sec_future(S5)),
                ("refs", "References and files", '<ol class="refs">' + "".join(f"<li>{r[r.index(']') + 2:]}</li>"
                                                                            for r in REFS) + "</ol>" + sec_files(S5))]
    sections = [x for x in sections if x[2]]
    lede_t, summary = lede(S, S6, S5)
    num = {sid: k for k, (sid, _h, _b) in enumerate(sections, 1)}
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|N\b|s\b)", "\\1&nbsp;\\2", s_))
    sections = [(sid, hd, unit(re.sub(r"SEC\[(\w+)\]", lambda m_: str(num[m_.group(1)]), b))) for sid, hd, b in sections]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{b}</section>' for k, (sid, hd, b) in enumerate(sections, 1))
    byline = ("Replay of the stored grids, 2026-11-09, and live searches on the six-coordinate layouts, 2026-11-10 "
              "(Section SEC[search6d])" + (" and at four new object sizes (Section SEC[newsizes])" if "newsizes" in num
                                           else "") + ". The grids: the CAD-model SR2 hand with the v4b "
              "fingertip (<code>SCALE_HAND=cad_v2</code>), HOM on MuJoCo-Warp, the work plant at 10 MPa (2026-10-30) and "
              "the SCS0009 plant at 2.5 MPa (2026-11-02). Follows <a href=\"" + TRENDS + "\">the trends page</a>, Section "
              "3, and <a href=\"" + DESIGN + "\">the design page</a>. Builder: "
              "<code>scripts/hom/hom_search_methods_page.py</code>.")
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": "SR2 layout search methods", "H1": "Layout search methods for the SR2 hand under HOM control",
                 "EYEBROW": "HOM layout study, 2026-11-09", "LEDE": unit(lede_t),
                 "BYLINE": re.sub(r"SEC\[(\w+)\]", lambda m_: str(num[m_.group(1)]), byline), "TOC": toc,
                 "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS + texsvg.BLOCK_CSS + ".alg{margin:1em auto;max-width:100%}"}.items():
        page = page.replace("{{" + k + "}}", v)
    n = []
    page = P.render(page, n)
    svgs = texsvg.render([(b, "block") for b in ALGS], cache_path=P.TEX_CACHE)
    page = re.sub(r"\x00ALG(\d+)\x00", lambda m_: svgs[int(m_.group(1))], page)
    page = docs_index.tag(page, topic="design", summary=summary, date="2026-11-09")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {n[0]} formulas, {len(ALGS)} algorithms", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
