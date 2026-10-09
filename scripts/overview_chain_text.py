"""Text of the topic overview docs/overviews/grasp_to_gait_chain.html (builder: scripts/topic_overview_page.py).

Reader: a collaborator new to the project. Takeaways: (1) on the bench only the open-loop grasp and turn have run, and
the eight hands' simulated and measured rankings agree at rho_s 0.50; (2) in simulation the whole task runs on a UR5e and
loses the tool mostly at the hand-over between the turn and the gait; (3) what has to happen before any of the chain
reaches the hand. Budget: 1,000 words of prose. Sources: the pages in PAGES (digest
logs/20261009-docs_reorg/chain_digest.md, 2026-10-09).
"""

KEY = "chain"
SHORT_TITLE = "Grasp-to-gait chain"
TITLE = "Grasp-to-gait chain on a UR5e and transfer of the open-loop turn to the bench"

LEDE = (
    "On the bench only the open-loop grasp and turn have run: the eight built hands held the tool in 48 of 77 scored "
    "trials, and their simulated and measured alignment rank at \\(\\rho_s = 0.50\\) (\\(p = 0.21\\)). In simulation the "
    "whole task, from grasping the lying tool to a gait that drives a screw, runs as one rollout with the hand on a UR5e. "
    "It completes 6 of 6 times on rv05_manual on the shipped servo model and 12 of 20 times on D5 and D6 with the "
    "learned turn; most losses come at the hand-over from the turn to the gait.")

TERMS = [
    ("Chain", "Grasp, lift, turn to vertical, set-down in a countersink, release or relay hand-over, re-grip in a ring, "
     "and a gait that turns the standing tool about its axis; one simulated rollout."),
    ("Seam", "The boundary between two stages of the chain, where the outcome is read."),
    ("Held", "At least two pads press on the tool with its weight, 0.240&#8202;N, or more."),
    ("Alignment", "On the bench, the cosine of the tool axis with vertical at the end of a trial, from the AprilTag tracker."),
    ("\\(\\rho_s\\)", "Spearman rank correlation over the eight built hands."),
    ("Relay hand-over", "Changing grasp one finger at a time, so that two fingers always hold the tool; a release "
     "hand-over opens all three."),
    ("Plant", "The simulated servo model: shipped (gain 30&#8202;N&#183;m/rad), calibrated (1.04&#8202;s finger lag), working "
     "(fitted to 177 bench runs)."),
]

SECTIONS = [
    ("bench", "Transfer of the open-loop turn",
     """<p>The bench study ran the CEM grasp and the open-loop turn on eight hands, D1&#8211;D8, chosen from the 227 layouts
that passed the simulated screens. Of 77 scored trials, 48 held the tool, 28 dropped it and one is unresolved; the
README and the paper count 80 and 51 because they include three later D1 trials. The simulated and measured alignment
rank the hands at \\(\\rho_s = 0.50\\) (\\(p = 0.21\\), bootstrap interval &#8722;0.29 to 1.00). Hold rate is a separate outcome
(\\(\\rho_s = -0.08\\) against alignment): D2 and D7 held 10 of 10, while D5 turned furthest (0.938) and held 3 of 10
([program record](exp:20260921-workshop_briefing/20260921-sr2_hand_program_record.html)). The servo correction of
2026-09-02, which passed a gate against these bench trials, was later found void
([gates on a residual](exp:20260903-sim2real-gates/20260903-sim2real_gates.html)).</p>"""),
    ("sim", "The chain in simulation",
     """<p>On rv05_manual and the shipped plant the chain completes 6 of 6 times, first with a floating palm and then
on a UR5e, and seats and turns a screw in a 45&#176; countersink 5 of 6 times. The hold after the turn leaks: the tool
creeps down through the pads at about 1.5&#8202;mm/s and the pad force falls from 12.0 to 0.4&#8202;N before the press, so
the set-down has to be one continuous move (one height correction completes 6 of 6, eight corrections 1 of 6). The
countersink costs rotation (49.0 against 40.3&#176; per gait cycle) and cuts the lateral walk from 1.28 to
0.08&#8202;mm ([seam and countersink](exp:20260903-real_v1_chain/20260903-real_v1_chain_and_countersink.html)). The UR5e
must be told the hand&#8217;s payload: declared, the palm error at the top of the lift falls from 0.97 to 0.01&#8202;mm, and
23&#176; of the alignment that had been credited to the turn was the wrist sagging
([wrist geometry](exp:20260904-real_v1_bench/20260904-real_v1_bench_geometry.html)).</p>
{{FIG arm}}"""),
    ("handover", "Hand-over from the turn to the gait",
     """<p>The hand-over loses the most rollouts. On the shipped plant with the table-supported turn, 122 of 128 rollouts
lift the tool, 102 hold the turn, 91 stand it up, 43 reach the gait grip and 41 complete
([screened plans in the chain](exp:20260906-screen_vs_chain/20260906-screen_vs_chain.html)). Changing grasp one finger at a
time keeps the tool touched for all but 0.4&#8202;% of the task, against 49.2&#8202;% for a release, at 23.0 against
44.3&#176; per gait cycle; in the countersink the relay holds a 30&#176; lateral load in 5 of 6 rollouts, where a release
fails at 16&#176; ([hand-over without release](exp:20260903-real_v1_handover/20260903-real_v1_relay_handover.html)).</p>
{{FIG relay}}"""),
    ("learned", "The chain with the learned turn",
     """<p>With the learned turn on the calibrated plant, 12 of 20 rollouts of the D5 and D6 policies complete the chain
and turn the screw 0.14&#8211;0.38 revolutions in eight gait cycles, a median 11.8&#176; per cycle against 34&#176; if the pads
did not slip; the hand-over that works is a release, a lift and a level re-pose of the palm, then a ring grasp fitted per hand.
D1 and D2 complete 0 of 5. That plant&#8217;s fingers lagged by 1.04&#8202;s; with the lag removed the chain loses the tool
before the seat on every hand, so none of these policies is a hardware candidate
([chain with the learned turn](exp:20260923-chain_handover_gait/20260923-chain_handover_gait.html),
[chain on the calibrated plant](exp:20260916-calibrated_plant_chain/20260916-calibrated_plant_chain.html)).</p>"""),
    ("gates", "Gates before the bench",
     """<p>A trajectory reaches the hand only after three checks: finger clearance along both the chord of the plan and
its 50&#8202;Hz trajectory file, the servo limits, and, for a learned residual, an envelope over the commands it can add,
which is not built yet. None of the chained trajectories (gait ring, re-index, relay legs, re-pose) has been exported
or gated ([gates on a residual](exp:20260903-sim2real-gates/20260903-sim2real_gates.html)).</p>"""),
]

FIGURES = {
    "arm": dict(page="20260903-real_v1_chain/20260903-real_v1_chain_and_countersink.html", n=8, photo=True,
                source="seam and countersink", alt="Filmstrip of the hand on a UR5e through the chain.",
                caption="The chain on a UR5e at its seams, rv05_manual on the shipped plant: the wrist rotation through the "
                        "turn is the motion the floating palm made for free."),
    "relay": dict(page="20260903-real_v1_handover/20260903-real_v1_relay_handover.html", n=3,
                  source="hand-over without release", alt="Bar chart of the share of the task with no pad on the tool.",
                  caption="Share of the task with no pad on the tool, for release and relay hand-overs and gait schedules."),
}

FUTURE = [
    "Re-run the chain on the working plant (gain 4&#8202;N&#183;m/rad, 0.02&#8202;s lag) with <code>real_v1_chain_policy.py</code>, "
    "re-choosing the gait timing and the boost gain for the faster fingers.",
    "Continue the 2026-09-20 policies for 20&#8202;M steps on the working plant and run each through the chain from 20 "
    "spawns; five spawns cannot separate a weak policy from an unlucky pose.",
    "Measure the finger dynamics on the bench: a 20&#176; free-air step per joint read at 111&#8202;Hz and a force-gauge push "
    "against a holding servo. The working plant is wrong if the step settles in under 0.2&#8202;s.",
    "Export the gait ring, re-index and relay legs as plans and pass them through "
    "<code>real_v1_trajectory_clearance.py</code> and the servo-limit check before any bench chain.",
]

PAGES = [
    ("Chain and transfer", "20260903-real_v1_chain/20260903-real_v1_chain_and_countersink.html",
     "Floating palm, UR5e, countersink"),
    ("Chain and transfer", "20260903-real_v1_handover/20260903-real_v1_relay_handover.html", "Relay hand-over and gait"),
    ("Chain and transfer", "20260903-sim2real-gates/20260903-sim2real_gates.html", "Checks before a residual"),
    ("Chain and transfer", "20260906-screen_vs_chain/20260906-screen_vs_chain.html", "Losses stage by stage"),
    ("Chain and transfer", "20260916-calibrated_plant_chain/20260916-calibrated_plant_chain.html",
     "Chain on the calibrated plant"),
    ("Chain and transfer", "20260923-chain_handover_gait/20260923-chain_handover_gait.html", "Learned turn in the chain"),
    ("Related pages", "20260921-workshop_briefing/20260921-sr2_hand_program_record.html", "Bench counts and ranking"),
    ("Related pages", "20260904-real_v1_bench/20260904-real_v1_bench_geometry.html", "UR5e payload and wrist"),
]
