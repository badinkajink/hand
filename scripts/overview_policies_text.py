"""Text of the topic overview docs/overviews/rl_policies.html (builder: scripts/topic_overview_page.py).

Reader: a collaborator new to the project. Takeaways: (1) no learned policy has run on the hand; a policy is a residual
on a scripted lift and turn, trained with PPO in MuJoCo-Warp; (2) whether a policy would transfer is decided by the servo
plant and the fingertip contact model it trained on; (3) policy quality is a seed draw, so a policy is judged over
perturbed rollouts with a load test. Budget: 1,000 words of prose. Sources: the pages in PAGES (digest
logs/20261009-docs_reorg/policies_digest.md, 2026-10-09).
"""

KEY = "policies"
SHORT_TITLE = "Reinforcement-learning policies"
TITLE = "Reinforcement-learning policies for the screwdriver turn"

LEDE = (
    "No learned policy has run on the hand; the bench results use the open-loop plan. A policy adds a learned "
    "correction to a scripted lift and turn. It is trained with PPO in MuJoCo-Warp at about 9,400 environment steps per "
    "second. The servo plant is limited to the SCS0009&#8217;s torque (0.18&#8202;N&#183;m). On it, D6 policies grip the "
    "tool with 13&#8211;18&#8202;N. Policies trained with compliant sphere-pad fingertips turn the tool nearly as far as "
    "policies trained with point contact (held cosine 0.79&#8211;0.88 against 0.85&#8211;0.90). Open-loop replays ran in "
    "CPU MuJoCo, Drake and Newton. The pad commands keep the tool in 78 of 88 replays, and the point-contact commands "
    "keep it in 74 of 88. The earlier working plant has a torque limit of 1&#8202;N&#183;m, which is 4.4 times the "
    "servo&#8217;s stall torque. On it the pad policies gripped with 44&#8211;55&#8202;N and turned the tool less "
    "(0.61&#8211;0.69). Policies trained on fingers that lagged by 1&#8202;s drop the tool at bench finger speed.")

TERMS = [
    ("Cosine", "Cosine of the tool axis with vertical: +1 tip down (the goal), 0 horizontal; held cosine is its mean over "
     "the rollouts that hold the tool."),
    ("Held", "At the end of the episode at least two fingertips press on the tool with its weight, 0.240&#8202;N, or more, "
     "and the tool is above 60&#8202;mm."),
    ("Residual", "The policy&#8217;s nine finger actions, 0.5&#8202;rad per unit, added to set-points that ease from the open "
     "keyframe to the grip; the palm motion is scripted."),
    ("Plant", "The simulated servo model, one of four. Shipped: gain 30&#8202;N&#183;m/rad. Calibrated: gain 0.5, finger lag "
     "1.04&#8202;s. Working: gain 4&#8202;N&#183;m/rad, lag 0.02&#8202;s, fitted to 177 bench runs, torque limit "
     "1&#8202;N&#183;m. Servo-limited: gain 0.72&#8202;N&#183;m/rad, lag 0.02&#8202;s, torque limit 0.18&#8202;N&#183;m, which "
     "is 80&#8202;% of the SCS0009&#8217;s stall torque."),
    ("Jittered", "The same 64 evaluation rollouts with the tool placed &#177;3&#8202;mm and &#177;10&#176; off the trained pose and "
     "the friction randomised."),
    ("Open-loop replay", "The finger targets recorded from a MuJoCo-Warp rollout, played at 50&#8202;Hz into another "
     "simulator with the palm held at its lifted pose."),
    ("Environment step", "One 20&#8202;ms policy step of one simulated hand: ten 2&#8202;ms physics steps and the reward and "
     "observation code."),
]

SECTIONS = [
    ("structure", "Policy structure and training",
     """<p>A policy outputs nine finger corrections every 20&#8202;ms on top of a scripted episode. In the script the fingers
ease from the open keyframe to the grip and the palm lifts the tool. The policy acts from step 58 for the turn and the
hold. It reads 66 numbers. These are the positions and velocities of the 15 hand joints and the tool&#8217;s position
and pose in the palm frame. They also include the finger joints and tool pose of the open-loop reference rollout at the
same step, the previous action and \\(1-c\\), where \\(c\\) is the cosine. Actor and critic are separate
512-256-128 ELU networks. A warm start copies both, because actor-only warm starts pushed converged policies off their
optimum. From step 58 the reward pays \\(100\exp(-4(1-c)^2)\\) for alignment and \\(300\,\Delta c\\) per step
for progress. Tracking and contact terms apply throughout and a lift term applies during the lift. Penalties apply to
drift, action rate and pad force above a threshold. An episode ends when the tool falls 20&#8202;mm below its spawn
height. It also ends when fewer than three pads touch the tool for 15 consecutive steps
([policies continued on the chain plant](exp:20260920-robust_tranche/20260920-robust_reorient_policies.html)).
PPO trains 2,048&#8211;3,072 hands in parallel at about 9,400 environment steps per second on the workstation&#8217;s
GPU, independent of the batch width. A DeltaAI GH200 runs at 0.74 of that speed. A run of 40&#8202;M steps costs
1.2&#8211;3.4 GPU-hours ([training throughput](exp:20260908-compute_budget/20260908-compute_budget.html),
[contact-model policies](exp:20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html)).</p>
{{FIG timeline}}
{{FIG episode}}
<p>The scripted lift replaced a learned one. On m05, the earlier simulation-only hand, a lift policy
(A) and a turn policy (B) were trained separately, because the two rewards conflicted. Every B trained from a
teleported, pre-lifted tool dropped it at the hand-off. In the live-A reset
(<code>src/morphohand/rl/live_a_runner.py</code>) a frozen A drives the first 40 steps of each B episode, and the
advantages of those steps are set to zero. B then held the tool at 0.110&#8202;m, and the reference turn policy b33
reached a cosine of 0.891. Recipes in <code>configs/recipes/</code> fix the residual scale, easing and contact gate for
training and evaluation alike. On real_v1 a script performs the lift, and the recipe keeps the name b_liveA
([program record](exp:20260921-workshop_briefing/20260921-sr2_hand_program_record.html)).</p>"""),
    ("plants", "Servo plants",
     """<p>Four servo plants have been used. Policies up to 2026-09-16
trained on the shipped plant. The D6 and D1&#8211;D7 policies of 2026-09-17 to 09-20 trained on the calibrated plant. Its fingers lagged by
1.04&#8202;s, because a template joint damping stayed in the model. D6 reached a cosine of 0.969 in 64 of 64 rollouts at a
grip of 10.8&#8202;N ([D6 at 60&#8202;M steps](exp:20260917-d6_cal_60M/20260917-d6_reorient_policy_60M.html)). At bench
finger speed none of ten policies holds the tool through its turn
([chain with the learned turn](exp:20260923-chain_handover_gait/20260923-chain_handover_gait.html)). The policies of
2026-10-08 train on the working plant (4&#8202;N&#183;m/rad, 0.02&#8202;s), which was fitted to the finger angles of bench
runs. Its torque limit of 1&#8202;N&#183;m is 4.4 times the SCS0009&#8217;s stall torque of 0.23&#8202;N&#183;m. Its policies
grip the 24.5&#8202;g tool with 44&#8211;96&#8202;N. The policies of 2026-10-11 train on the servo-limited plant. Its torque
limit is 0.18&#8202;N&#183;m, the torque at which the servo&#8217;s overload protection trips. Its gain is
0.72&#8202;N&#183;m/rad, the gain at which the servo drive saturates at 18&#176; of error ([servo-plant policies](exp:20261011-servo_plant_policies/20261011-servo_plant_contact_model_policies.html)).</p>
{{FIG plants}}"""),
    ("seeds", "Seeds and evaluation",
     """<p>Policy quality depends on the training draw more than on the recipe. Five from-scratch draws on m05, the earlier
simulation-only hand, spread their held cosine with a standard deviation of 0.38. Sixteen draws there, warm-started from
one policy, spread it by 0.032. Two from-scratch D6 runs of 60&#8202;M steps ended at 0.933 and 0.640
([training throughput](exp:20260908-compute_budget/20260908-compute_budget.html),
[D6 at 60&#8202;M steps](exp:20260917-d6_cal_60M/20260917-d6_reorient_policy_60M.html)). A comparison of designs therefore
needs several seeds or a shared warm start.</p>
{{FIG seeds}}
<p>The trajectory-health scorecard (<code>src/morphohand/rl/trajectory_health.py</code>) grades a rollout on six checks
(late finger, idle finger, drop, jitter, de-centring, over-clamp) and reports the worst. In the runs of 2026-10-08 a
checkpoint watch evaluates a run every 4&#8202;M steps. It stops the run after 10&#8202;M steps of degenerate checkpoints.
It also stops the run when the held cosine gains less than 0.02 over 10&#8202;M steps
([contact-model policies](exp:20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html)).</p>
<p>A policy is judged on jittered rollouts with a load test, under its own run&#8217;s timing. Evaluated with the residual
switched on too early, the D6 policy held 12 of 64 rollouts instead of 64. The policies of 2026-09-19 hold 44&#8211;64 of 64 jittered
rollouts on the plant they trained on. With the tool on the scene&#8217;s contact class, 11 of 16 drop it even at the
trained pose (D7 0 of 64). Continued on that class with jitter, D5 holds 55 and D7 28 of 64 jittered rollouts.
Evaluations store their per-step traces beside their pages (<code>traces/</code>). Without a bound the residual
commanded up to 384&#176; past a joint&#8217;s range. Clipped at &#177;1&#8202;rad, the policies still hold 64 of 64 at the
trained pose
([policies across the hands](exp:20260919-hands_tranche/20260919-reorient_policies_across_hands.html),
[policies continued](exp:20260920-robust_tranche/20260920-robust_reorient_policies.html)).</p>
<p>The engineering log is markdown in <code>docs/rl/</code> (<a href="../rl/reorientation.md">reorientation.md</a> lists the
runs in order); the autopilot hand-off notes are in <code>docs/handoff/</code>.</p>"""),
    ("contact", "Fingertip contact model and transfer",
     """<p>Twelve D6 runs on the working plant compared four fingertip models, three seeds each, 40&#8202;M steps under a
checkpoint watch that stops degenerate runs. Point-contact policies turn the tool further where they hold (cosine
0.84&#8211;0.94 against 0.61&#8211;0.69 for the pads). They shake it at 302 and 126&#8202;rad/s&#178;, against 18 for the
pads, and five of their six runs degenerated. Their finger commands keep the tool in 3 of 132 replays in other
simulators. The pad commands keep it in 85 of 88. In closed loop under the other contact models the pad policies hold
76&#8202;% of rollouts. Every final policy grips with 40&#8211;100&#8202;N. All but one lose the tool when the finger servo
gain rises to 10&#8202;N&#183;m/rad ([contact-model policies](exp:20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html)). The fourth
arm was the same pads mounted on a sprung skin. It held 99&#8202;% in closed loop at 44&#8211;56&#8202;N grips on the
1&#8202;N&#183;m plant. It trained at impratio 10, where the skin&#8217;s springs store half of what its friction rows
creep. Its policies therefore did not use the presliding that the skin models. That arm is reported with the skin
([sprung skin](exp:20261012-sprung_skin/20261012-sprung_skin_presliding.html)).</p>
{{FIG replays_working}}
{{FIG transfer}}
<p>On the servo-limited plant a penalty replaces the bonus on fingertip force above 4&#8202;N. Two pad seeds and two
TPU-mesh seeds then hold the tool in 256 of 256 rollouts at a grip of 13&#8211;18&#8202;N. The pad policies turn the tool
nearly as far as the TPU-mesh policies (held cosine 0.79&#8211;0.88 against 0.85&#8211;0.90). During the turn the tool
spins 97&#8211;98&#176; relative to the pads, against 32&#8211;52&#176; at the working plant&#8217;s grip. Given
1&#8202;N&#183;m, pad seed 0 grips with 44&#8202;N and stops turning the tool (held cosine 0.02). Pad seed 1 drops 28 of 64
rollouts, while the TPU-mesh policies turn the tool as before. A pad&#8217;s torsional friction grows as the 5/4 power
of its force. Both kinds of finger commands transfer. The pad commands keep the tool in 78 of 88 open-loop replays and
the TPU-mesh commands keep it in 74 of 88. Newton loses the most ([servo-plant policies](exp:20261011-servo_plant_policies/20261011-servo_plant_contact_model_policies.html)).
Two seeds trained on the same pads mounted on a sprung skin change no held count beyond the pads&#8217; seed spread in
20 evaluation conditions, replays and perturbations included. They end at held cosines of 0.81 and 0.65, against the
pads&#8217; 0.88 and 0.79. The skin policies lie inside the pads&#8217; spread in 18 of the 20 conditions. They trained at
impratio 10 without the skin&#8217;s presliding. That remains to be tested with friction rows that hold
([policies on the sprung skin](exp:20261011-servo_plant_policies/20261011-servo_plant_contact_model_policies.html#skin),
[sprung skin](exp:20261012-sprung_skin/20261012-sprung_skin_presliding.html)).</p>
{{FIG replays_servo}}"""),
]

_SERVO = "docs/experiments/20261011-servo_plant_policies/media/"
_WORK = "docs/experiments/20261008-contact_model_policies/media/"
_PAGE_SERVO = ("20261011-servo_plant_policies/20261011-servo_plant_contact_model_policies.html", "servo-plant policies")
_PAGE_WORK = ("20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html", "contact-model policies")
# the three-simulator replay films are 1920 x 480: MuJoCo-Warp, Drake and Newton panels of 640 px under a 56 px text overlay
_PANELS = (("MuJoCo-Warp", 0), ("Drake", 640), ("Newton", 1280))


def _replay_rows(prefix, runs):
    return [[(prefix + f"{run}_replay_three.mp4", f"{tip}, {sim}", f"640:420:{x}:56") for sim, x in _PANELS]
            for run, tip in runs]


FIGURES = {
    "episode": dict(film=[[(_SERVO + "20261011-d6_servo_tpu27pads1_40M_s0_best.mp4", None)]], width=640,
                    page=_PAGE_SERVO[0], source=_PAGE_SERVO[1],
                    caption="One evaluation rollout of the D6 policy with 1&#8202;mm pad fingertips on the servo-limited "
                            "plant (seed 0, its best of 64 rollouts), at real speed. The fingers close and the palm lifts "
                            "the tool on the script. From step 58 the policy&#8217;s corrections turn it. The middle "
                            "finger lifts the handle end, and the tool pivots about the thumb and index contacts to a held "
                            "cosine of 0.88. The fingertips press on it with 18&#8211;20&#8202;N in sum."),
    "plants": dict(film=[[(_WORK + "20261008-d6_work_tpu27pads1_40M_s0_best.mp4", "Working plant: 1 N·m, 42–44 N grip"),
                          (_SERVO + "20261011-d6_servo_tpu27pads1_40M_s0_best.mp4",
                           "Servo-limited plant: 0.18 N·m, 18–20 N grip")]],
                   pages=[_PAGE_WORK, _PAGE_SERVO],
                   caption="The same D6 recipe with 1&#8202;mm pad fingertips on two servo plants, best seed of each, at "
                           "real speed. With a 1&#8202;N&#183;m torque limit and a bonus on fingertip force (left) the fingers "
                           "squeeze the tool with 42&#8211;44&#8202;N, and it stops at a held cosine of 0.69. At "
                           "0.18&#8202;N&#183;m with a penalty above 4&#8202;N (right) the fingers hold it with "
                           "18&#8211;20&#8202;N and turn it to 0.88."),
    "seeds": dict(film=[[("docs/experiments/20260917-d6_cal_60M/videos/d6_cal_60M_m812.mp4", "Run 1: held cosine 0.93"),
                         ("docs/experiments/20260917-d6_cal_60M/videos/d6_cal_60M_gpm025_m812.mp4",
                          "Run 2: held cosine 0.64, no fingertip force")]],
                  page="20260917-d6_cal_60M/20260917-d6_reorient_policy_60M.html", source="D6 at 60&#8202;M steps",
                  caption="Two from-scratch D6 runs of 60&#8202;M steps on the real_v1 model with its box fingertips "
                          "(scene <code>20260916-sv1_u0308_b050_cal</code>; calibrated plant, kp 0.5&#8202;N&#183;m/rad, "
                          "torque limit 0.35&#8202;N&#183;m), final checkpoints, one deterministic rollout each at half speed. "
                          "Their recipes differ only in the sign of a grip term worth 0.3&#8202;% of the return. Both roll the "
                          "tool the same way through step 110. From step 140 the second run leaves the tool resting at "
                          "50&#176; on the fingers, with zero force at all three fingertips in each of its 64 evaluation "
                          "rollouts. The held cosine is 0.93 for the first run and 0.64 for the second."),
    "replays_working": dict(film=_replay_rows(_WORK + "20261008-d6_work_tpu27", [("mesh_40M_s1", "Point contact"),
                                                                                 ("pads1_40M_s2", "Pads")]),
                            page=_PAGE_WORK[0], source=_PAGE_WORK[1],
                            caption="Open-loop replays on the working plant at real speed, for the median seed of each of two fingertip "
                                    "models, world 0. The left panel is the MuJoCo-Warp rollout the policy trained on. The "
                                    "centre and right panels play its finger targets from the onset of the turn into Drake "
                                    "and Newton. The point-contact targets lose the tool in Drake at 3.4&#8202;s and in Newton "
                                    "within a second. The pad targets keep it at the same angle in all three simulators "
                                    "(cosine 0.58&#8211;0.61)."),
    "replays_servo": dict(film=_replay_rows(_SERVO + "20261011-d6_servo_tpu27", [("mesh_40M_s1", "Point contact"),
                                                                                ("pads1_40M_s1", "Pads")]),
                          page=_PAGE_SERVO[0], source=_PAGE_SERVO[1],
                          caption="The same replays for the median seeds trained on the servo-limited plant. In Drake "
                                  "the tool stays near the MuJoCo-Warp angle for both kinds of targets (cosine 0.79 and 0.77 "
                                  "against 0.83 and 0.80). In this world Newton loses the tool for both. A Newton world can "
                                  "hold the tool in one pass and drop it in the next, while its totals repeat. Over all "
                                  "replays the pad targets keep the tool in 78 of 88 and the point-contact targets keep it "
                                  "in 74 of 88."),
    "timeline": dict(page="20260920-robust_tranche/20260920-robust_reorient_policies.html", n=17,
                     alt="Timeline of one episode: grasp, lift, then the policy's turn and hold.",
                     source="policies continued on the chain plant",
                     caption="One episode: the scripted grasp and lift in simulation steps of 2&#8202;ms (above), and the "
                             "policy steps of 20&#8202;ms in which the residual acts (below)."),
    "transfer": dict(page="20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html", n=4,
                     crop=(0, 0, 1, 0.668),          # the box, TPU-mesh and pad rows; the skin rows are on the skin page
                     alt="Grid of held shares and cosines per contact model and seed under each simulator.",
                     source="contact-model policies",
                     caption="Transfer of the final D6 point-contact and pad policies on the working plant, one row per "
                             "contact model and seed. The left panel gives the held share and cosine in closed loop under "
                             "each fingertip model in MuJoCo-Warp; its skin column runs each policy on the pads mounted "
                             "on a sprung skin. The right panel gives open-loop replays of the finger commands in CPU "
                             "MuJoCo, Drake and Newton. The rows of the policies trained on the skin are on the "
                             "sprung-skin page."),
}

FUTURE = [
    "Replay the finger commands of the servo-plant pad and TPU-mesh final policies on the bench at 50&#8202;Hz from the "
    "lifted grasp, after the trajectory-clearance gate, ten tracked trials each "
    "(<code>logs/20261011-servo_plant_policies/replay/</code>). Commands that drop the tool on the hand in most trials, "
    "where the three simulators keep it, put the gap in the plant or the fingertip model.",
    "Retrain one pad and one TPU-mesh seed with the grip penalty at &#8722;20. At &#8722;2 the term costs under 1&#8202;% of "
    "the return, and the thumb presses 6&#8211;8&#8202;N with its mcp at the torque limit. A policy that turns the tool as "
    "far with the thumb under 4&#8202;N does not need the saturated thumb.",
    "Measure the stall torque of a finger servo on the bench against a load cell at 6&#8202;V. In simulation the pad and skin "
    "policies hold 112&#8211;128 of 128 rollouts at torque limits of 0.12 and 0.23&#8202;N&#183;m. The TPU-mesh policies "
    "drop up to 20 at 0.12. At 1&#8202;N&#183;m pad seed 0 stops turning the tool. A stall well above 0.23&#8202;N&#183;m "
    "calls for a swap of the torque limit to the measured value "
    "(<code>rl_contact_eval.py mechanism --study 20261011 --all --perturb fr=&lt;stall&gt;</code>).",
    "Clip the residual (<code>clip_actions 1.0</code>) for one pad and one TPU-mesh seed and compare the held cosine.",
]

PAGES = [
    ("Policies", "20260908-compute_budget/20260908-compute_budget.html", "Throughput, duty cycle, GH200"),
    ("Policies", "20260917-d6_cal_60M/20260917-d6_reorient_policy_60M.html", "D6 at 60 M steps; grip finetune"),
    ("Policies", "20260919-hands_tranche/20260919-reorient_policies_across_hands.html", "Clipped residual on D1&#8211;D7"),
    ("Policies", "20260920-robust_tranche/20260920-robust_reorient_policies.html", "Jittered continuation"),
    ("Policies", "20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html",
     "Four fingertip models; transfer"),
    ("Policies", "20261011-servo_plant_policies/20261011-servo_plant_contact_model_policies.html",
     "Servo-limited plant; the pad brake; the sprung skin"),
    ("Related pages", "20260827-real_v1/20260827-real_v1_routes_to_vertical.html", "Residual RL against the open-loop plan"),
    ("Related pages", "20260923-chain_handover_gait/20260923-chain_handover_gait.html", "The 1 s finger lag"),
    ("Related pages", "20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html", "Working plant; pad cost in RL"),
]
