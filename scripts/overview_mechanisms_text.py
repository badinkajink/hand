"""Text of the topic overview docs/overviews/reorientation.html (builder: scripts/topic_overview_page.py).

Reader: a collaborator new to the project. Takeaways: (1) the fixed-contact turn and what limits it, (2) what it does on
the bench-fitted servo model and under hand-object control, (3) the alternatives and what each needs. Budget: 1,100
words of prose. Sources: the pages in PAGES; numbers as those pages give them (digest logs/20261009-docs_reorg/
mechanisms_a_digest.md and mechanisms_b_digest.md, 2026-10-09).
"""

KEY = "mechanisms"
SHORT_TITLE = "Reorientation mechanisms"
TITLE = "Mechanisms and controllers for turning the screwdriver to vertical"

LEDE = (
    "The hand stands the screwdriver up by rotating its three fingertip contacts as one rigid body about an axis above "
    "them, an open-loop plan of two keyframes. In simulation this turns the tool to a cosine of +0.991 &#177; 0.002 with "
    "vertical in 8 of 8 rollouts, and every bench result in the paper used this plan. On the servo model fitted to the "
    "bench, the deployed plans keep the tool in 40 of 40 placements but turn it only 27&#8211;57&#176;; the hand-object "
    "controller of Wang, Oh and Pollard stops at 6&#8211;42&#176; because the grasps leave the fingers no extension.")

TERMS = [
    ("Cosine", "Cosine of the tool axis with vertical, signed: +1 tip down (the goal), 0 horizontal, &#8722;1 handle down."),
    ("Held", "At least two fingertip pads press on the tool with its weight, 0.240&#8202;N, or more, with no floor under it."),
    ("Fixed-contact turn", "The pads stay where they grip the tool and rotate with it as one rigid body."),
    ("Extension left", "Distance the closed grip leaves before the 68.11&#8202;mm finger chain is straight; a fixed-contact "
     "turn uses it up."),
    ("Pivot height", "Height of the rotation axis above the contacts, in half the index&#8211;middle straddle "
     "(<code>axis_k</code>)."),
    ("Plant", "The simulated servo model: shipped (gain 30&#8202;N&#183;m/rad), calibrated (0.5, with 1&#8202;s finger lag), "
     "working (4&#8202;N&#183;m/rad, 0.02&#8202;s lag, fitted to 177 tracked bench runs)."),
    ("HOM controller", "The hand-object controller of arXiv 2609.25619: joint rates by least squares on reference "
     "velocities of each contact frame, with a pinch-force brake."),
    ("D1&#8211;D8", "The eight built hands, numbered by simulated rank."),
]

SECTIONS = [
    ("turn", "Fixed-contact turn",
     """<p>With the pads fixed on the shaft, the turn uses up finger extension. The fitted grasps of the first four
real_v1 hands left 1.3&#8211;7.1&#8202;mm of the 68.11&#8202;mm chain, which bounds a fixed-contact turn at
2.5&#8211;10.2&#176;; residual policies on those grasps turned the tool 0.9&#8211;4.0&#176;
([rotational lock](exp:20260827-real_v1/20260827-real_v1_rotational_lock.html)). Rotating the contacts about an axis
above them moves the cost from extension to retraction and turns the tool open loop without the floor: cosine
+0.991 &#177; 0.002, 8 of 8, on rv05_manual
([design search](exp:20260828-real_v1_search/20260828-real_v1_design_search.html)). This two-keyframe plan is the
controller behind every hardware number in the paper.</p>
<p>The pivot height selects the pole the tool turns to. On the deployed hands a pivot of 0.05 turned it handle down
(cosine &#8722;0.988) and 0.25 tip down (+0.994), and the deployment fitter left 0.44&#8211;1.26&#8202;mm of extension where
the CEM grasp leaves 2.74&#8202;mm ([control diagnosis](exp:20260906-control_diagnosis/20260906-control_diagnosis.html)).
At pivot 0.15, D6 completed the whole chain open loop on 3 of 4 seeds with three pads at all thirteen seams; its fingers
turned the tool 39.5&#176; and the arm the rest
([eight deployed hands](exp:20260908-reorientation_journey/20260908-reorientation_journey.html)).</p>
{{FIG turn}}"""),
    ("scoring", "Scoring a turn",
     """<p>A turn is scored by the signed cosine and a load test. Three scoring errors each produced a result that did not
exist: an unsigned tilt counted 952 of 967 handle-down stands as upright, 116 of 177 table stands had no pad on the tool,
and a success read at the wrong seam reported +0.568 where the chain reached +0.995 two seams later
([eight deployed hands](exp:20260908-reorientation_journey/20260908-reorientation_journey.html),
[table stand](exp:20260904-real_v1_held/20260904-real_v1_table_stand.html)). The pad-elevation result of 2026-09-06
was retracted for the first of these
([pad elevation](exp:20260906-pad_elevation/20260906-pad_elevation.html)).</p>"""),
    ("plants", "Servo plants and the bench",
     """<p>The results depend on the plant they ran on. Every result before 2026-09-16 used the shipped plant,
on which the fingers roll a gripped tool through 90&#176;. On the calibrated plant of 2026-09-16 they turned it
1&#8211;8&#176;, and the bench maneuver became a tip of 20&#8211;40&#176; whose pad force fell to 0.1&#8211;0.5&#8202;N in the first
quarter of the turn ([three plants](exp:20260916-turn_mechanism/20260916-turn_mechanism.html)). That plant was
withdrawn: its fingers lagged by 1.04&#8202;s, its gain fit never loaded a finger, and the bench scenes started the tool
12.5&#8202;mm inside its post. On the working plant fitted to the bench readbacks, the deployed open-loop plans keep the
tool in 40 of 40 placements and turn it 27&#8211;57&#176;
([servo refit](exp:20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html#plant)).</p>
<p>On the bench the rv05 plan came to rest 50&#176; from vertical because the tool wedged against the palm plate, modelled
25&#8202;mm too low; with the built plate height the same plan ends 24&#8211;34&#176; from vertical in simulation
([bench partial reorientation](exp:20260916-tip_gait/20260916-bench_partial_reorientation.html)).</p>"""),
    ("hom", "Hand-object control",
     """<p>The HOM controller tracks a reference velocity for each fingertip&#8217;s contact frame and brakes a pinched tool
by setting the pinch force. A brake needs friction torque about the contact normal, which point contact lacks: with
point contact the tool swings free at the lift in Drake and MuJoCo. With Drake hydroelastic contact, sphere-packed
pads or condim&#160;4 the closed-loop brake ends the swing at 88.2&#8211;91.4&#176; and the tool is inserted 22.0&#8202;mm into its
hole; the pick fails on 8 of 10 perturbed seeds
([chain control](exp:20261002-hom_chain/20261002-hom_screwdriver_chain.html#nominal),
[pinch brake](exp:20261001-hom_hand_brake/20261001-hom_hand_brake.html)).</p>
<p>Applied to the three-finger turn on D1&#8211;D8 with the working plant, the controller keeps every placement (40 of 40 in
MuJoCo, 24 of 24 in Drake) but stops at 6&#8211;42&#176;: the proximal-interphalangeal joints start 0&#8211;3&#176; from their
&#8722;18&#176; limit and sit on a bound in a median 85&#8202;% of control ticks. Grasps searched for a turn range of
18&#8211;81&#176; lost the tool in the grip (3 of 18 held)
([servo refit and hand-object turn](exp:20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html#hom)).</p>
{{FIG hom}}"""),
    ("support", "Turns supported by the floor or by gravity",
     """<p>With the tool standing on the floor, an open-loop gait of four phases turns it 4.99 times in 40 cycles at
0.5&#176; of tilt; without the floor 0 of 6 rollouts complete. The fingertip radius sets the gear ratio, 1.684 for the
10.55&#8202;mm pad ([ground-supported gaiting](exp:20260902-real_v1_gait/20260902-real_v1_gaiting.html)). Releasing the
middle finger lets gravity swing the tool about the thumb&#8211;index pinch to a cosine of +0.92&#8211;0.98 on D8 (6 of 6);
torsional friction of 0.003 halves the swing and 0.01 stops it. The swing is ballistic and was set aside
([pinch and swing](exp:20260916-swing_reorient/20260916-swing_reorient.html)).</p>
{{FIG gait}}"""),
]

FIGURES = {
    "turn": dict(page="20260908-reorientation_journey/20260908-reorientation_journey.html", n=22, photo=True,
                 crop=(0, 0, 1, 0.6), source="eight deployed hands",
                 caption="D6 in the simulated chain at pivot 0.15, close on the hand, left to right and top to bottom: from "
                         "horizontal to vertical, seated in the countersink, and the gait starting, with three pads on the "
                         "shaft throughout (shipped plant)."),
    "hom": dict(page="20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html", n=4, photo=True,
                source="servo refit and hand-object turn",
                caption="D7 on the working plant: the deployed open-loop plan (left), the HOM controller with its governor "
                        "(middle) and without it (right); pads touching the tool are red."),
    "gait": dict(page="20260902-real_v1_gait/20260902-real_v1_gaiting.html", n=1,
                 source="ground-supported gaiting",
                 caption="Accumulated rotation of the standing tool over 40 gait cycles, three repeats per hand, 30&#176; "
                         "stroke (shipped plant)."),
}

FUTURE = [
    "Score grasps after the grip in <code>hom_grasp_search.py</code> (0.8&#8202;s grip on the working plant; reject a "
    "grasp that leaves the post or ends 2&#8202;mm from its planned contacts). A searched grasp that the HOM controller turns "
    "past 6, 42 and 23&#176; on D7, D2 and D5 confirms the grasp as the limit.",
    "Give a finger a second contact mode: when a proximal-interphalangeal joint reaches its limit, slide that pad along "
    "the tool while the other two hold, and measure the turn and the drops on the same 40 placements.",
    "Replay the HOM joint commands open loop on the bench (the hand has no force sensing), D5 or D6 first, ten trials "
    "against the hand&#8217;s own plan, read by the AprilTag tracker.",
    "Measure the servo time constant with a 20&#176; free-air step per joint at 111&#8202;Hz; a settling time above "
    "0.2&#8202;s contradicts the working plant.",
]

PAGES = [  # (group, page under docs/experiments, contents)
    ("Fixed-contact turn and its control", "20260827-real_v1/20260827-real_v1_rotational_lock.html",
     "Extension budget; raised pivot"),
    ("Fixed-contact turn and its control", "20260827-real_v1/20260827-real_v1_routes_to_vertical.html",
     "Residual RL against the open-loop plan"),
    ("Fixed-contact turn and its control", "20260906-control_diagnosis/20260906-control_diagnosis.html",
     "Pivot height and pole"),
    ("Fixed-contact turn and its control", "20260908-reorientation_journey/20260908-reorientation_journey.html",
     "Scoring errors; D6 chain"),
    ("Fixed-contact turn and its control", "20260904-real_v1_held/20260904-real_v1_table_stand.html",
     "Load test"),
    ("Fixed-contact turn and its control", "20260906-pad_elevation/20260906-pad_elevation.html",
     "Retracted"),
    ("Plants and the bench", "20260916-turn_mechanism/20260916-turn_mechanism.html",
     "Plant dependence"),
    ("Plants and the bench", "20260916-tip_gait/20260916-bench_partial_reorientation.html",
     "Palm plate as the stop"),
    ("Hand-object control", "20261001-hom_hand_brake/20261001-hom_hand_brake.html",
     "Brake loading"),
    ("Hand-object control", "20261002-hom_chain/20261002-hom_screwdriver_chain.html",
     "Pick, swing, insertion"),
    ("Hand-object control", "20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html",
     "Working plant; HOM turn"),
    ("Hand-object control", "20261007-contact-gait/20261007-contact_release_real_v1.html",
     "Release and recontact"),
    ("Floor and gravity", "20260902-real_v1_gait/20260902-real_v1_gaiting.html",
     "Gait on the floor"),
    ("Floor and gravity", "20260902-real_v1_gait/20260902-real_v1_gaiting_open_problems.html",
     "Gait open problems"),
    ("Floor and gravity", "20260916-swing_reorient/20260916-swing_reorient.html",
     "Gravity swing"),
]
