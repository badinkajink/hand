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
    "fingers. Each servo behaves as a spring that lags its command by 0.0186&#176; per unit of load, and it has a "
    "protective-torque cutoff. Under a grasp the yaw joints reach 0.44&#8211;0.90 of the commanded angle. The simulated "
    "servo model in use since 6 October replays 177 tracked bench runs within 2.89&#176; per joint. It does not identify "
    "the servo gain. Two AprilTags give the tool&#8217;s pose to 0.017&#176; and 0.030&#8202;mm.")

TERMS = [
    ("Load", "The servo&#8217;s own load reading, 0&#8211;1000. It is a PWM-duty proxy for torque."),
    ("Deflection", "Commanded minus achieved joint angle at a steady hold, in degrees."),
    ("Protective torque", "The servo register at which it cuts its drive; the shipped value trips at a load of exactly 200."),
    ("Plant", "The simulated servo model: position gain (N&#183;m/rad), joint damping, torque limit and friction."),
    ("Readback error", "Mean absolute difference between simulated and bench joint angles over the nine joints of a run, "
     "in degrees."),
    ("Time constant", "Joint damping divided by position gain. A free finger covers 63&#8202;% of a step in this time."),
    ("CB1", "The board that runs the servo bus and the control station; the workstation runs the camera and the tracker."),
]

SECTIONS = [
    ("hand", "The hand",
     """<p>Each finger base rides on an XY gantry driven by stepper motors. The thumb&#8217;s gantry spans 110 by 60&#8202;mm and the
index and middle bases 60 by 60&#8202;mm, all at full travel since 2026-09-01. Each axis is a T5&#8202;&#215;&#8202;1 lead screw at
3,216 steps/mm on a TMC5160 driver. A change of layout takes about 95&#8202;s. Sensorless homing (StallGuard2 against a
hard stop) takes about 2&#8202;min. Each finger has three servo joints: a yaw at its base and two flexion joints. The yaw
spans &#177;85&#176; (commands capped at &#177;70&#176;), the MCP spans &#8722;15 to 92&#176; and the PIP &#8722;18 to 92&#176;.
Seven tracked, re-staged trials of D1&#8217;s plan gave a cosine of +0.881 with a standard deviation of 0.011. The plan
drifts by &#177;5&#176; between sessions ([program record](exp:20260921-workshop_briefing/20260921-sr2_hand_program_record.html)). The links are printed. The printed TPU fingertip is a 17 by 14.8 by 22&#8202;mm block
with 2.7&#8202;mm fillets. The model differs from the built hand in three measured ways. Its link masses are about a
quarter of the real ones. The palm plate sits 25&#8202;mm higher on the hand than it did in the model before 2026-09-16.
The yaw servos under the palm have no collision geometry
([servo identification](exp:20260902-servo-sysid/20260902-servo_sysid.html),
[bench partial reorientation](exp:20260916-tip_gait/20260916-bench_partial_reorientation.html)).</p>
{{FIG photo}}"""),
    ("station", "Control station and the path of a plan",
     """<p>The CB1 board runs the control station (<code>manta_hand.web</code>, port 8765). The station is the only owner of
both serial links. One link goes to the Manta M8P board, which steps the six gantry axes over USB. The other is the
half-duplex bus of the nine servos at 1&#8202;Mbaud. <code>real_v1_export_plan.py</code> writes a plan as three
set-points, a 50&#8202;Hz per-joint trajectory, a build sheet and a pose sheet. A plan that passes the gates is copied to
the CB1. The gates are the servo range with the &#177;70&#176; yaw cap, the gantry travel, and 5&#8202;mm of index&#8211;middle
clearance along both the set-point chord and the trajectory.
During a policy stream the station writes the servos without reading them, and the tracker sends the tool pose over the
network. <code>python -m manta_hand.web --fake</code> runs the station against simulated hardware; the operator&#8217;s guide
is <a href="../hardware_control_station.md">docs/hardware_control_station.md</a>
([program record](exp:20260921-workshop_briefing/20260921-sr2_hand_program_record.html), [residual gates](exp:20260903-sim2real-gates/20260903-sim2real_gates.html)).</p>"""),
    ("servo", "The servo: a spring and a cutoff",
     """<p>Over 236 logged bench runs on 11 hands, the deflection at a steady hold is 0.0186&#176; per unit of load. This is one
constant for every loaded joint (0.0177&#8211;0.0198, \\(R^2\\) 0.94&#8211;0.99), with no stiction offset. A separate regime
sits at a load of exactly 200, where the protective torque trips. It trips on 42&#8202;% of middle-yaw runs. The
registers read P&#8202;15, I&#8202;0, D&#8202;15. With no integral term, a standing load leaves a proportional error
([servo identification](exp:20260902-servo-sysid/20260902-servo_sysid.html)).</p>
<p>The achieved angles therefore differ from the commanded ones. Across 20 runs the yaw joints reached
0.44&#8211;0.90 of their commanded travel and the distal flexion joints reached 1.00 &#177; 0.05. This held for commands at
0.1 and 1.5 times the normal speed alike. The simulated rankings use commanded angles. An open-loop plan keeps the tool
only inside a band of residual clip that differs per plan. The largest clip the servos can execute is 1.345&#8202;rad for
most of the family, and the middle finger&#8217;s distal joint limits it ([clip, cap and stall](exp:20260830-real_v1-budget-rescreen/20260830-real_v1_budget_rescreen.html)).</p>"""),
    ("plant", "The simulated servo model",
     """<p>Three plants have been used. The shipped model (gain 30&#8202;N&#183;m/rad) was used for every result before
2026-09-16. The calibrated model of 2026-09-02 (gain 0.5) came from a fit that closed the fingers on air. Its joint
damping gave the fingers a 1.04&#8202;s lag. The working plant was fitted on 2026-10-06 by replaying 177 tracked bench
runs. It has gain 4&#8202;N&#183;m/rad, a 0.02&#8202;s time constant, a 1&#8202;N&#183;m torque limit and friction 1.0. It
reproduces the joint readbacks within 2.89&#176;, against 6.25&#176; for the calibrated model and 3.29&#176; for the shipped
model. The readbacks fix a time constant of 0.2&#8202;s or less and friction near 1, but they do not fix the gain: gains
from 0.25 to 30 all fit within 2.53&#8211;2.75&#176;. The same refit showed that every bench scene since 2026-08-29 had
started the tool 12.5&#8202;mm inside its post
([servo refit](exp:20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html#plant)).</p>"""),
    ("bench", "Bench instruments",
     """<p>Two AprilTags, one on the palm and one on a vane on the tool, give the tool&#8217;s pose at 30 frames per second. The rms
noise is 0.017&#176; and 0.030&#8202;mm. The bench height equals the simulated height plus 44.0&#8202;mm
([object pose from two tags](exp:20260831-real_v1-object-tracking/20260831-real_v1_object_tracking.html)). The servo bus
reads the nine positions at 111&#8202;Hz once the USB adapter&#8217;s latency timer is set to 1&#8202;ms (7&#8202;Hz at its
16&#8202;ms default). Through the driver it writes and reads at 75&#8202;Hz. The station reads the servos one at a time,
because SYNC READ times out on every register. The servo has no current register.</p>
<p>On a UR5e the hand must be declared as the arm&#8217;s payload. With the payload declared, wrist stacks of 0&#8211;150&#8202;mm complete 8 of 8
chains. The palm error at the top of the lift falls from 0.97 to 0.01&#8202;mm. Without it, a 150&#8202;mm stack completes
0 of 8, and 23&#176; of the turn credited to the fingers was the wrist sagging
([wrist geometry](exp:20260904-real_v1_bench/20260904-real_v1_bench_geometry.html)).</p>
{{FIG wrist}}"""),
]

FIGURES = {
    "photo": dict(image="docs/overviews/media/sr2_hand_bench.jpg", photo=True, width=620,
                  alt="The SR2 hand on the bench holding the screwdriver, with AprilTags on palm and tool.",
                  caption="The hand on the bench holding the screwdriver, with the AprilTags of the tracker on the palm and "
                          "the tool (Fig.&#160;1 of the paper, <a href='https://sr2-hand.github.io'>sr2-hand.github.io</a>)."),
    "wrist": dict(page="20260904-real_v1_bench/20260904-real_v1_bench_geometry.html", n=2, fn="wrist_payload",
                  source="wrist geometry", alt="Bar chart of chains completed against wrist stack length, payload "
                                               "declared and not declared.",
                  caption="Chains completed against wrist-stack length on a UR5e, with the hand declared as payload and "
                          "without. The stack costs nothing once the arm is told its payload (shipped plant)."),
}

FUTURE = [
    "Measure the servo gain with a known load. Hang 50, 100 and 200&#8202;g from a fingertip on a measured lever while the "
    "servo holds, and read the deflection. The slope is the gain in N&#183;m/rad, which the readbacks cannot give.",
    "Measure the time constant with a 20&#176; free-air step per joint read at 111&#8202;Hz; a value above 0.2&#8202;s contradicts "
    "the working plant.",
    "Weigh the printed links and put the real masses and centres of mass into "
    "<code>assets/mjcf/real_v1/real_hand.xml</code>; the model&#8217;s links are a quarter of the real mass.",
    "Look at what the tool&#8217;s handle rests on at the end of one rv05 bench run. If it is a yaw servo, add the servo "
    "bank to the model and regenerate the scenes with <code>build_real_v1_scenes.py --keep-keyframes</code>.",
]

PAGES = [
    ("Hand, plant and bench", "20260827-real_v1/20260827-real_v1_first_night.html", "First run of the CAD hand in simulation"),
    ("Hand, plant and bench", "20260830-real_v1-budget-rescreen/20260830-real_v1_budget_rescreen.html",
     "Clip band, servo range, stall"),
    ("Hand, plant and bench", "20260831-real_v1-object-tracking/20260831-real_v1_object_tracking.html", "AprilTag tracker"),
    ("Hand, plant and bench", "20260902-servo-sysid/20260902-servo_sysid.html", "Spring constant and cutoff"),
    ("Hand, plant and bench", "20260904-real_v1_bench/20260904-real_v1_bench_geometry.html", "UR5e payload and wrist"),
    ("Hand, plant and bench", "20261013-cad_v1_finger/20261013-cad_v1_finger_model.html", "Finger model from the CAD parts"),
    ("Hand, plant and bench", "20261013-servo_identification/20261013-servo_identification_protocol.html",
     "Servo identification protocol, draft"),
    ("Related pages", "20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html", "Working plant from 177 runs"),
    ("Related pages", "20260916-turn_mechanism/20260916-turn_mechanism.html", "The turn on three plants"),
    ("Related pages", "20260916-tip_gait/20260916-bench_partial_reorientation.html", "Palm plate height"),
]
