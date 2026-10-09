"""Text of the topic overview docs/overviews/hand_plant_bench.html (builder: scripts/topic_overview_page.py).

Reader: a collaborator new to the project. Takeaways: (1) what the hardware is and how its servos behave (a spring with a
protective-torque cutoff; commanded is not achieved); (2) which simulated servo model is current and what it does and does
not identify; (3) what the bench measures and the corrections a simulation needs to match it. Budget: 1,000 words of
prose. Sources: the pages in PAGES (digest logs/20261009-docs_reorg/hardware_digest.md, 2026-10-09).
"""

KEY = "hardware"
SHORT_TITLE = "Hand, plant and bench"
TITLE = "The SR&#178; hand, its servo model and the bench instruments"

LEDE = (
    "The hand has six stepper-gantry coordinates that place the finger bases and nine SCS0009 hobby servos that move the "
    "fingers. Each servo behaves as a spring, lagging its command by 0.0186&#176; per unit of load, with a protective-torque "
    "cutoff; under a grasp the yaw joints reach 0.44&#8211;0.90 of the commanded angle. The simulated servo model in use "
    "since 6 October replays 177 tracked bench runs within 2.89&#176; per joint but does not identify the servo gain. Two "
    "AprilTags give the tool&#8217;s pose to 0.017&#176; and 0.030&#8202;mm.")

TERMS = [
    ("Load", "The servo&#8217;s own load reading, 0&#8211;1000 (a PWM-duty proxy for torque)."),
    ("Deflection", "Commanded minus achieved joint angle at a settled hold, in degrees."),
    ("Protective torque", "The servo register at which it cuts its drive; the shipped value trips at a load of exactly 200."),
    ("Plant", "The simulated servo model: position gain (N&#183;m/rad), joint damping, torque limit and friction."),
    ("Readback error", "Mean absolute difference between simulated and bench joint angles over the nine joints of a run, "
     "in degrees."),
    ("Time constant", "Joint damping over position gain: the time a free finger takes to cover 63&#8202;% of a step."),
    ("CB1", "The board that runs the servo bus and the control station; the workstation runs the camera and the tracker."),
]

SECTIONS = [
    ("hand", "The hand",
     """<p>Each finger base rides on an XY gantry driven by stepper motors: the thumb&#8217;s spans 110 by 60&#8202;mm and the
index and middle bases 60 by 60&#8202;mm, all at full travel since 2026-09-01. Each finger has three servo joints, a yaw at
its base and two flexion joints. The links are printed; the printed TPU fingertip is a 17 by 14.8 by 22&#8202;mm block
with 2.7&#8202;mm fillets. The model differs from the built hand in three measured ways: its link masses are about a
quarter of the real ones, the palm plate sits 25&#8202;mm higher on the hand than it did in the model before 2026-09-16,
and the yaw servos under the palm have no collision geometry
([servo identification](exp:20260902-servo-sysid/20260902-servo_sysid.html),
[bench partial reorientation](exp:20260916-tip_gait/20260916-bench_partial_reorientation.html)).</p>
{{FIG photo}}"""),
    ("servo", "The servo: a spring and a cutoff",
     """<p>Over 236 logged bench runs on 11 hands, the deflection at a settled hold is 0.0186&#176; per unit of load, one
constant for every loaded joint (0.0177&#8211;0.0198, \\(R^2\\) 0.94&#8211;0.99), with no stiction offset. A separate regime
sits at a load of exactly 200, where the protective torque trips; it does so on 42&#8202;% of middle-yaw runs. The
registers read P&#8202;15, I&#8202;0, D&#8202;15: with no integral term a standing load leaves a proportional error
([servo identification](exp:20260902-servo-sysid/20260902-servo_sysid.html)).</p>
<p>The achieved angles therefore differ from the commanded ones. Across 20 runs the yaw joints arrived at
0.44&#8211;0.90 of their commanded travel and the distal flexion joints at 1.00 &#177; 0.05, for commands at 0.1 and 1.5
times the normal speed alike; the simulated rankings use commanded angles. Open-loop plans keep the tool only inside a band of residual clip per plan, and the
largest clip the servos can execute is 1.345&#8202;rad for most of the family, limited by the middle finger&#8217;s distal
joint ([clip, cap and stall](exp:20260830-real_v1-budget-rescreen/20260830-real_v1_budget_rescreen.html)).</p>"""),
    ("plant", "The simulated servo model",
     """<p>Three plants have been used. The shipped model (gain 30&#8202;N&#183;m/rad) held every result before 2026-09-16. The
calibrated model of 2026-09-02 (gain 0.5) came from a fit that closed the fingers on air, and its joint damping gave the
fingers a 1.04&#8202;s lag. The working plant, fitted on 2026-10-06 by replaying 177 tracked bench runs, has gain
4&#8202;N&#183;m/rad, a 0.02&#8202;s time constant, a 1&#8202;N&#183;m torque limit and friction 1.0, and reproduces the joint
readbacks within 2.89&#176;, against 6.25&#176; for the calibrated and 3.29&#176; for the shipped model. The readbacks fix a
time constant of 0.2&#8202;s or less and friction near 1, but not the gain: gains from 0.25 to 30 all fit within
2.53&#8211;2.75&#176;. The same refit found that every bench scene since 2026-08-29 had started the tool 12.5&#8202;mm inside
its post
([servo refit](exp:20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html#plant)).</p>"""),
    ("bench", "Bench instruments",
     """<p>Two AprilTags, one on the palm and one on a vane on the tool, give the tool&#8217;s pose at 30 frames per second with
0.017&#176; and 0.030&#8202;mm rms noise; the bench height equals the simulated height plus 44.0&#8202;mm
([object pose from two tags](exp:20260831-real_v1-object-tracking/20260831-real_v1_object_tracking.html)). The servo bus
reads the nine positions at 111&#8202;Hz once the USB adapter&#8217;s latency timer is set to 1&#8202;ms (7&#8202;Hz at its
16&#8202;ms default) and writes and reads at 75&#8202;Hz through the driver.</p>
<p>On a UR5e the hand must be declared as the arm&#8217;s payload. Declared, wrist stacks of 0&#8211;150&#8202;mm complete 8 of 8
chains and the palm error at the top of the lift falls from 0.97 to 0.01&#8202;mm; undeclared, a 150&#8202;mm stack completes
0 of 8, and 23&#176; of the turn credited to the fingers was the wrist sagging
([wrist geometry](exp:20260904-real_v1_bench/20260904-real_v1_bench_geometry.html)).</p>
{{FIG wrist}}"""),
]

FIGURES = {
    "photo": dict(image="docs/overviews/media/sr2_hand_bench.jpg", photo=True, width=620,
                  alt="The SR2 hand on the bench holding the screwdriver, with AprilTags on palm and tool.",
                  caption="The hand on the bench holding the screwdriver, with the AprilTags of the tracker on the palm and "
                          "the tool (Fig.&#160;1 of the paper, <a href='https://sr2-hand.github.io'>sr2-hand.github.io</a>)."),
    "wrist": dict(page="20260904-real_v1_bench/20260904-real_v1_bench_geometry.html", n=2,
                  source="wrist geometry", alt="Bar chart of chains completed against wrist stack length, payload "
                                               "declared and not declared.",
                  caption="Chains completed against wrist-stack length on a UR5e, with the hand declared as payload and "
                          "without; the stack costs nothing once the arm is told its payload (shipped plant)."),
}

FUTURE = [
    "Measure the servo gain with a known load: hang 50, 100 and 200&#8202;g from a fingertip on a measured lever while the "
    "servo holds, and read the deflection; the slope is the gain in N&#183;m/rad, which the readbacks cannot give.",
    "Measure the time constant with a 20&#176; free-air step per joint read at 111&#8202;Hz; a value above 0.2&#8202;s contradicts "
    "the working plant.",
    "Weigh the printed links and put the real masses and centres of mass into "
    "<code>assets/mjcf/real_v1/real_hand.xml</code>; the model&#8217;s links are a quarter of the real mass.",
    "Look at what the tool&#8217;s handle rests on at the end of one rv05 bench run; if it is a yaw servo, add the servo bank "
    "to the model and regenerate the scenes with <code>build_real_v1_scenes.py --keep-keyframes</code>.",
]

PAGES = [
    ("Hand, plant and bench", "20260827-real_v1/20260827-real_v1_first_night.html", "First run of the CAD hand in simulation"),
    ("Hand, plant and bench", "20260830-real_v1-budget-rescreen/20260830-real_v1_budget_rescreen.html",
     "Clip band, servo range, stall"),
    ("Hand, plant and bench", "20260831-real_v1-object-tracking/20260831-real_v1_object_tracking.html", "AprilTag tracker"),
    ("Hand, plant and bench", "20260902-servo-sysid/20260902-servo_sysid.html", "Spring constant and cutoff"),
    ("Hand, plant and bench", "20260904-real_v1_bench/20260904-real_v1_bench_geometry.html", "UR5e payload and wrist"),
    ("Related pages", "20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html", "Working plant from 177 runs"),
    ("Related pages", "20260916-turn_mechanism/20260916-turn_mechanism.html", "The turn on three plants"),
    ("Related pages", "20260916-tip_gait/20260916-bench_partial_reorientation.html", "Palm plate height"),
]
