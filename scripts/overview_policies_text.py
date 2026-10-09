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
    "correction to a scripted lift and turn and is trained with PPO in MuJoCo-Warp at about 9,400 environment steps per "
    "second. On the servo model fitted to the bench, D6 policies trained with compliant sphere-pad fingertips keep the "
    "tool in 216 of 220 open-loop replays in CPU MuJoCo, Drake and Newton, and policies trained with point contact in 3 "
    "of 132. Earlier policies, trained on fingers that lagged by 1&#8202;s, drop the tool at bench finger speed.")

TERMS = [
    ("Cosine", "Cosine of the tool axis with vertical: +1 tip down (the goal), 0 horizontal; held cosine is its mean over "
     "the rollouts that hold the tool."),
    ("Held", "At the end of the episode at least two fingertips press on the tool with its weight, 0.240&#8202;N, or more, "
     "and the tool is above 60&#8202;mm."),
    ("Residual", "The policy&#8217;s nine finger actions, 0.5&#8202;rad per unit, added to set-points that ease from the open "
     "keyframe to the grip; the palm motion is scripted."),
    ("Plant", "The simulated servo model: shipped (gain 30&#8202;N&#183;m/rad), calibrated (0.5, 1.04&#8202;s finger lag), "
     "working (4&#8202;N&#183;m/rad, 0.02&#8202;s lag, fitted to 177 bench runs)."),
    ("Jittered", "The same 64 evaluation rollouts with the tool placed &#177;3&#8202;mm and &#177;10&#176; off the trained pose and "
     "the friction randomised."),
    ("Open-loop replay", "The finger targets recorded from a MuJoCo-Warp rollout, played at 50&#8202;Hz into another "
     "simulator with the palm held at its lifted pose."),
    ("Environment step", "One 20&#8202;ms policy step of one simulated hand: ten 2&#8202;ms physics steps and the reward and "
     "observation code."),
]

SECTIONS = [
    ("structure", "Policy structure and training",
     """<p>A policy outputs nine finger corrections every 20&#8202;ms on top of a scripted episode: the fingers ease from the
open keyframe to the grip, the palm lifts the tool, and the policy acts from step 58 for the turn and the hold. The
reward grows with the tool axis&#8217;s alignment with vertical, its progress toward it and pad contact, and falls with
lateral drift and grip force above a threshold
([policies continued on the chain plant](exp:20260920-robust_tranche/20260920-robust_reorient_policies.html)).
PPO trains 2,048&#8211;3,072 hands in parallel at about 9,400 environment steps per second on the workstation&#8217;s
GPU, independent of the batch width; a DeltaAI GH200 runs at 0.74 of that speed, and 40&#8202;M steps cost
1.2&#8211;3.4 GPU-hours ([training throughput](exp:20260908-compute_budget/20260908-compute_budget.html),
[contact-model policies](exp:20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html)).</p>
{{FIG timeline}}"""),
    ("plants", "Servo plants",
     """<p>Three servo plants have been used, and only the last is fitted to bench readings. Policies up to 2026-09-16
trained on the shipped plant. The D6 and D1&#8211;D7 policies of 2026-09-17 to 09-20 trained on the calibrated plant,
whose fingers lagged by 1.04&#8202;s because a template joint damping stayed in the model: D6 reached a cosine of 0.969 in
64 of 64 rollouts at a grip of 10.8&#8202;N ([D6 at 60&#8202;M steps](exp:20260917-d6_cal_60M/20260917-d6_reorient_policy_60M.html)),
but at bench finger speed none of ten policies holds the tool through its turn
([chain with the learned turn](exp:20260923-chain_handover_gait/20260923-chain_handover_gait.html)). The policies of
2026-10-08 train on the working plant (4&#8202;N&#183;m/rad, 0.02&#8202;s).</p>"""),
    ("seeds", "Seeds and evaluation",
     """<p>Policy quality depends on the training draw more than on the recipe. Five from-scratch draws on m05, the earlier
simulation-only hand, spread their held cosine with a standard deviation of 0.38; sixteen draws there warm-started from
one policy spread by 0.032;
two from-scratch D6 runs of 60&#8202;M steps ended at 0.933 and 0.640
([training throughput](exp:20260908-compute_budget/20260908-compute_budget.html),
[D6 at 60&#8202;M steps](exp:20260917-d6_cal_60M/20260917-d6_reorient_policy_60M.html)). A design comparison therefore
needs several seeds or a shared warm start.</p>
<p>A policy is judged on jittered rollouts with a load test, under its own run&#8217;s timing. Evaluated with the residual
switched on too early, the D6 policy held 12 of 64 rollouts instead of 64. Trained at one pose, D7 held 64 of 64 at
that pose and 0 of 64 jittered; continuing training under jitter raised D5 from 23 to 57 and D7 from 1 to 35 of 64.
Without a bound the residual commanded up to 384&#176; past a joint&#8217;s range; clipped at &#177;1&#8202;rad the policies still
hold 64 of 64 at the trained pose
([policies across the hands](exp:20260919-hands_tranche/20260919-reorient_policies_across_hands.html),
[policies continued](exp:20260920-robust_tranche/20260920-robust_reorient_policies.html)).</p>"""),
    ("contact", "Fingertip contact model and transfer",
     """<p>Twelve D6 runs on the working plant compared four fingertip models, three seeds each, 40&#8202;M steps under a
checkpoint watch that stops degenerate runs. Point-contact policies turn the tool further where they hold (cosine
0.84&#8211;0.94 against 0.61&#8211;0.79 for pads and skin) but shake it at 302 and 126&#8202;rad/s&#178; against 18 and 19, and
five of their six runs degenerated. Their finger commands keep the tool in 3 of 132 replays in other simulators; the
pad and skin commands keep it in 216 of 220, and in closed loop under the other contact models the skin policies hold
99&#8202;% of rollouts and the pad policies 76&#8202;%. Every final policy grips with 40&#8211;100&#8202;N, and all but one lose the tool
when the finger servo gain rises to 10&#8202;N&#183;m/rad
([contact-model policies](exp:20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html)).</p>
{{FIG transfer}}"""),
]

FIGURES = {
    "timeline": dict(page="20260920-robust_tranche/20260920-robust_reorient_policies.html", n=16,
                     alt="Timeline of one episode: grasp, lift, then the policy's turn and hold.",
                     source="policies continued on the chain plant",
                     caption="One episode: the scripted grasp and lift in simulation steps of 2&#8202;ms (above), and the "
                             "policy steps of 20&#8202;ms in which the residual acts (below)."),
    "transfer": dict(page="20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html", n=4,
                     alt="Grid of held shares and cosines per contact model and seed under each simulator.",
                     source="contact-model policies",
                     caption="Transfer of the final D6 policies, one row per contact model and seed: held share and "
                             "cosine in closed loop under each fingertip model in MuJoCo-Warp (left), and open-loop "
                             "replays of the finger commands in CPU MuJoCo, Drake and Newton (right)."),
}

FUTURE = [
    "Replay the finger commands of the pad, skin and TPU-mesh final policies on the bench at 50&#8202;Hz from the lifted "
    "grasp, ten tracked trials each (<code>logs/20261008-contact_model_policies/replay/</code>). The claim fails if the "
    "TPU-mesh commands hold the tool as often as the pad or skin commands.",
    "Finetune the pad and skin policies with the grip penalty at &#8722;5 (the 2026-09-17 setting) and replay again; the "
    "replays should still hold at a grip near 10&#8202;N.",
    "Randomise the finger servo gain between 2 and 10&#8202;N&#183;m/rad per episode in <code>env_build.py</code>, then evaluate "
    "at 2, 6 and 10 with <code>rl_contact_eval.py robust</code>. A finetuned policy that still drops the tool at 10 "
    "depends on the soft servo for its turn.",
    "Clip the residual (<code>clip_actions 1.0</code>) for one pad and one skin seed and compare the held cosine.",
]

PAGES = [
    ("Policies", "20260908-compute_budget/20260908-compute_budget.html", "Throughput, duty cycle, GH200"),
    ("Policies", "20260917-d6_cal_60M/20260917-d6_reorient_policy_60M.html", "D6 at 60 M steps; grip finetune"),
    ("Policies", "20260919-hands_tranche/20260919-reorient_policies_across_hands.html", "Clipped residual on D1&#8211;D7"),
    ("Policies", "20260920-robust_tranche/20260920-robust_reorient_policies.html", "Jittered continuation"),
    ("Policies", "20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html",
     "Four fingertip models; transfer"),
    ("Related pages", "20260827-real_v1/20260827-real_v1_routes_to_vertical.html", "Residual RL against the open-loop plan"),
    ("Related pages", "20260923-chain_handover_gait/20260923-chain_handover_gait.html", "The 1 s finger lag"),
    ("Related pages", "20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html", "Working plant; pad cost in RL"),
]
