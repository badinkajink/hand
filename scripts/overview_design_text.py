"""Text of the topic overview docs/overviews/morphology_design.html (builder: scripts/topic_overview_page.py).

Reader: a collaborator new to the project. Takeaways: (1) how the eight built hands were chosen and how well the
simulated ranking held on the bench; (2) what predicts whether a hand turns the tool (the grasp) and what predicts only
retention (mount layout, kinematic scores); (3) how the best finger spacing scales with object size. Budget: 1,000 words of
prose. Sources: the pages in PAGES (digest logs/20261009-docs_reorg/design_digest.md, 2026-10-09).
"""

KEY = "design"
SHORT_TITLE = "Morphology and design search"
TITLE = "Finger-layout search for the SR&#178; hand and the transfer of its ranking"

LEDE = (
    "The program sampled 8,198 finger layouts with a Sobol sequence, screened them in simulation for a grasp that keeps "
    "the tool through a perturbed open-loop turn (227 passed), and built eight. On the bench the eight held the tool in "
    "48 of 77 trials, and simulated and measured alignment rank them at \\(\\rho_s = 0.50\\) (\\(p = 0.21\\)); they lie close "
    "together, a median 43.6&#8202;mm apart. Where the thumb grasps the tool predicts whether a hand turns it; the mount "
    "layout does not. For a cylinder or sphere the best layout puts the mounts 35&#8211;37&#8202;mm outside its surface.")

TERMS = [
    ("Layout", "The six gantry coordinates of the three finger bases (the hand&#8217;s morphology)."),
    ("Sobol sample", "A low-discrepancy sequence over the six-dimensional box of gantry travel."),
    ("Retention screen", "A simulated grasp, 60&#8202;mm lift and 5&#8202;s hold with no more than 10&#8202;mm of slip, repeated "
     "under perturbations."),
    ("Alignment", "Cosine of the tool axis with vertical at the end of the turn; +1 is upright."),
    ("\\(\\rho_s\\)", "Spearman rank correlation."),
    ("AUC", "Area under the curve of a score used as a classifier of an outcome; 0.5 is chance, 1 perfect."),
    ("Thumb moment arm", "Moment arm of the thumb&#8217;s contact about the pinch axis of the other two fingers "
     "(<code>tau_thumb</code>)."),
    ("KaRMA", "A kinematic score of the object poses a hand can reach (KaRMA-T: translations, KaRMA-R: rotations)."),
]

SECTIONS = [
    ("funnel", "Search funnel and bench ranking",
     """<p>The population covers the six gantry coordinates; all 8,198 layouts fit the gantry travel. Half fail the
grasp fit, 535 pass the retention screen, 227 are confirmed under perturbations, 19 were exported as plans and eight
built as D1&#8211;D8, numbered by simulated rank. On the bench the eight held the tool in 48 of 77 scored trials, and the
simulated and measured alignment rank them at \\(\\rho_s = 0.50\\) (\\(p = 0.21\\), bootstrap interval &#8722;0.29 to 1.00)
([program record](exp:20260921-workshop_briefing/20260921-sr2_hand_program_record.html)). The eight are concentrated:
their median pairwise separation is 43.6&#8202;mm against 66.9&#8202;mm for eight layouts drawn at random, and seven of
them sit in the compact quadrant.</p>
<p>A design is evaluated in a frozen scene (<code>freeze_scene_for_eval</code>): its mounts become fixed body transforms
and the morphology joints, which drift under contact, are removed. The funnel and the deployed plans take their grasp from
a geometric fitter (<code>fit_real_v1_pose.py</code>) that solves for a palm height and nine finger angles with three pads
at or below the shaft&#8217;s equator. The older CEM search (<code>phase1_optimize_grasp.py</code>) optimises the hold, which on
real_v1 trades against the turn: rv04_mid with the thumb offset turned the tool to +0.997 from its fitted grasp and to
+0.625 after CEM ([design search](exp:20260828-real_v1_search/20260828-real_v1_design_search.html)).</p>
{{FIG funnel}}"""),
    ("predictors", "What predicts a turning hand",
     """<p>Among 108 designs searched with a fixed-contact turn, 49 reorient the tool, all by a two-pad pinch-roll that
sheds the descending pad. The grasp decides more than the layout: sliding the thumb 20&#8202;mm along the shaft takes
rv04_mid from 0 of 8 to 8 of 8, and the thumb&#8217;s moment arm about the pinch axis separates turning from non-turning
designs at \\(\\rho = +0.443\\), AUC 0.821, over 80 graspable designs. Above 55&#8202;mm of thumb-to-pair separation the
mount coordinates stop ordering the designs (\\(\\rho\\) &#8722;0.015 and &#8722;0.092)
([design search](exp:20260828-real_v1_search/20260828-real_v1_design_search.html)).</p>
<p>Kinematic scores predict which designs keep the tool and barely which turn it. KaRMA-T separates retained from
discarded designs at AUC 0.802 but correlates with the screened turn at only \\(\\rho = +0.105\\)
([kinematic rolling-pinch metric](exp:20260910-karma_metric/20260910-karma_metric_evaluation.html)), and an earlier dexterity score ranked the grip and missed the turn.</p>
{{FIG predictors}}"""),
    ("scale", "Finger spacing against object size",
     """<p>On a family of symmetric tripods with the bench servo model and 1&#8202;mm pad fingertips, the layout with the
largest fixed-contact workspace grows one for one with the object: the mounts sit 37.2&#8202;mm outside a sphere&#8217;s
surface and 34.7&#8202;mm outside a cylinder&#8217;s, about half the 68.1&#8202;mm finger. The task moves the optimum: a cylinder
is held most robustly 25.4&#8202;mm out and turned furthest 31.3&#8202;mm out, and for spheres robustness peaks at the compact
edge of the feasible band for 8 of 10 objects and the turn at the wide edge for 10 of 10. A bench protocol of 47 cells
(17.3&#8202;h) is written and has not been run
([finger spacing and object size](exp:20261008-hand_object_scale/20261008-finger_spacing_object_size.html)). A size-7
basketball needs the widest layout, friction of at least 1.0 and MuJoCo&#8217;s elliptic friction cone to be held
([basketball grasp](exp:20260916-basketball/20260916-basketball_grasp.html)).</p>
{{FIG scale}}"""),
    ("earlier", "Earlier design studies in simulation",
     """<p>Before the hand was built, designs had nine parameters: the x and y of each finger base and its proximal length.
<code>morph_pipeline_sweep.py</code> scored a design by baking it into fixed geometry, retargeting the grasp keyframe by
inverse kinematics, searching the grasp by CEM, training a lift policy from
scratch and a turn policy on the lift&#8217;s deliveries, and grading the hand-off with the trajectory-health scorecard. The
policy draw dominated: the held cosine of one design varied with a standard deviation of 0.38 between from-scratch seeds;
a fixed lift policy, a shared warm start and fingertip imitation in the object frame cut it to
0.02, and one design win (H06_04, 0.748 over four draws) replicated ([program record](exp:20260921-workshop_briefing/20260921-sr2_hand_program_record.html)). Of two finger arrangements, the
opposed pair turned the tool to 0.995 and dropped it in every rollout, while the inline pair held at 0.884
([two finger arrangements](exp:20260818-perp_review_page/20260818-perp_reorientation.html)).</p>"""),
]

FIGURES = {
    "funnel": dict(page="20260921-workshop_briefing/20260921-sr2_hand_program_record.html", n=1,
                   source="program record", alt="Bar chart of the number of layouts at each stage of the funnel.",
                   caption="Layouts at each stage of the search, on one scale: the grasp fit removes half the population, "
                           "the retention screen most of the rest."),
    "predictors": dict(page="20260828-real_v1_search/20260828-real_v1_design_search.html", n=4, fn="design_scores",
                       source="design search", alt="Two scatter plots of the best held cosine against two scores.",
                       caption="Best held cosine of each design against its two best-correlated scores, the thumb&#8217;s "
                               "moment arm among them; blue designs turn and hold the tool."),
    "scale": dict(page="20261008-hand_object_scale/20261008-finger_spacing_object_size.html", n=3, fn="scale_ridges",
                  source="finger spacing and object size", alt="Ridge location against object radius for cylinders "
                                                                 "and spheres.",
                  caption="Best mount distance against object radius for spheres and cylinders: workspace, hold and turn "
                          "ridges."),
}

FUTURE = [
    "Score a fresh Sobol sample by the thumb&#8217;s moment arm and run the open-loop turn on it; the claim fails if the AUC on "
    "new designs falls toward 0.5.",
    "Bench the within-family contrast already on the control station (g12 at four clips, u0060 and u0100 at two), twenty "
    "trials in one sitting. If the bench does not reproduce an ordering within one family, an ordering across families "
    "means little.",
    "Choose the next hands to build where the retention screen and the bench-schedule replay disagree most, and keep the "
    "eight built hands as the held-out set.",
    "Train one policy over randomised mounts that observes the six mount coordinates, and test first whether its "
    "behaviour changes with the design vector alone; if it does not, the policy ignores the design and cannot rank it.",
    "Run the 47-cell bench protocol of the object-size study (17.3&#8202;h; core set of 8 objects in 7.1&#8202;h) after "
    "printing the objects and exporting the plans with <code>hand_object_scale_export.py</code>.",
]

PAGES = [
    ("Morphology and design search", "20260818-perp_review_page/20260818-perp_reorientation.html",
     "Opposed and inline pairs"),
    ("Morphology and design search", "20260828-real_v1_search/20260828-real_v1_design_search.html",
     "108 designs; the thumb&#8217;s moment arm"),
    ("Morphology and design search", "20260910-karma_metric/20260910-karma_metric_evaluation.html",
     "KaRMA predicts retention"),
    ("Morphology and design search", "20260916-basketball/20260916-basketball_grasp.html", "Basketball cap grasp"),
    ("Morphology and design search", "20261001-drake-port/20261001-drake_sr2_planning.html", "Drake model and planning"),
    ("Morphology and design search", "20261008-hand_object_scale/20261008-finger_spacing_object_size.html",
     "Spacing against object size"),
    ("Morphology and design search", "20261011-bench_plan/20261011-bench_plan_finger_spacing.html",
     "Bench plan for the spacing study"),
    ("Morphology and design search", "20261013-scale_turn_controllers/20261013-scale_turn_hom_mpc.html",
     "Cylinder turns under HOM and sampling MPC"),
    ("Related pages", "20260921-workshop_briefing/20260921-sr2_hand_program_record.html", "Funnel, bench counts"),
]
