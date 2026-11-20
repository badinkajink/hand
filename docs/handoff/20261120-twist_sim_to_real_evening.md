Status: active
Task: 2026-11-20, owner 11:5x: "we have ~9 hours left of light time ... you deliver me a wonderful chart showing whatever
sim2real landscape/ranking correspondence we've determined"; owner 12:0x: "this jitter just seems highly problematic.
smooth control enables more predictable control, im highly confident about this". The owner is writing the paper and
wants quiet: one message tonight with the page's local path and link.
Read first: docs/notes/20261120-twist_ranking_decisions.md (every decision of the day, newest last);
docs/experiments/20261120-twist_sim_to_real/20261120-twist_ranking_sim_to_real.html (builder scripts/hom/twist_sim_to_real_page.py);
docs/experiments/20261120-controller_rematch/20261120-controllers_tuned_forces_calibrated_plant.html (scripts/hom/controller_rematch_page.py);
docs/experiments/20261115-hom_bench_trials/analysis/20261120-command_jitter_*.py; memory project_controller_rematch_2026_11_20.md.
State (12:4x):
- Sim-to-real page v1 built and pushed (61fe8ebd0): 5 layouts, 11 clean trials, rho -0.10; anchor 0.367 vs 0.376.
- Jitter traced (af94c6028, 40268f626): budget passes joint flicker into targets (45 % reversals vs 17 %). Server flags
  --budget-joint-tau 0.1 --target-tau 0.04 exist, off; the owner's yes or no is pending (asked 12:2x). Smoothed
  predictions: data/20261120-smoothed_predictions.json (twist ranking), from 20261120-command_jitter_sim.py --predict.
- Bench (corvus, lab session "Physical model vs GP calibration"): day order of 170 trials, 60 fps since 11:5x, rows carry
  "camera"; the lab will not switch the smoothing without the owner's yes.
- Running: lane_f1 (feasible landscape at 1.0 N, logs/20261119-twist_layout_ranking/lane_f1.log, ~13:30, then summary
  data/20261120-feasible_ranking_f1.json); lane_gpu3 (UHAS published then limits at 1024 worlds, scored on _cal45,
  logs/20261120-controller_rematch/lane_gpu.log, ~16:00); the video pull (logs/20261120-video_pull/, until 23:30).
Next:
1. When LANE_F1_DONE exists: rebuild the sim-to-real page (Figure 2 gains the 1.0 N panel); commit by path.
2. When lane_gpu.log shows the UHAS scores: controller_rematch.py summary, rebuild the rematch page, add the UHAS lines to the decision log.
3. Every 2 h until the light goes (~21:00): rsync corvus's 2026112*_corvus.jsonl and the series rows into runs_corvus/ (see the builder's docstring), run sim/20261120-trials_vs_prediction.py, rebuild the page, commit.
4. Tonight: final rescore and rebuild; publish both pages from the personal login (pinned); send the owner one message with both local paths and links.
5. If the owner says yes to the smoothing: relay the words to the lab with the time; have the scorer report smoothed trials apart (the server header's budget_joint_tau).
Claimed: docs/experiments/20261120-twist_sim_to_real/, scripts/hom/twist_sim_to_real_page.py, docs/experiments/20261119-twist_layout_ranking/sim/20261120-trials_vs_prediction.py, docs/notes/20261120-twist_ranking_decisions.md
Waiting on owner: the smoothed budget on the bench, yes or no (recommend yes: 45 % -> 11-12 % reversals at the same simulated tracking).
Do not: start or stop anything on corvus or the hand; switch the server flags; push data files over 2 MB; republish from the Max seat.
