Status: active
Task: (owner, 2026-11-19 17:2x) "re-running the full task space search, let's say for just the sphere twist, with HOM
... deploying as many morphologies (and top-N controls) on M trials, and seeing how well our real results match with
sim"; the search "definitely get this done overnight, no matter what", and be "ready to fully evaluate starting tomorrow,
in real, how well simulation rankings hold up". Also: "one final valiant pass at improving simulation, improving (closed
loop) controls & stability/smoothness/accuracy of contact sensing" (per-control-step gap over contact model, pose filter
and servo model).
Read first: docs/experiments/20261119-twist_layout_ranking/sim/20261119-twist_ranking_search.py (docstring: model,
stages, rows, heartbeat); logs/20261119-twist_layout_ranking/lane.sh and lane.log; the 11-17 model it reuses,
docs/experiments/20261117-sr2_system_identification/QUEUE.md "W3, A/B" (blind predictions 0.437 vs the hand's 0.447);
docs/experiments/20261115-hom_bench_trials/README.md (how trials run); memory project_sr2_identification_night_2026_11_17.md.
State: 22:1x (hand-4a). Page docs/experiments/20261119-twist_layout_ranking/20261119-sphere_twist_layout_ranking.html,
artifact https://claude.ai/artifact/XiBYaBDci8QPCjNzyZqSue v2 (28e798b5e). Lights went off at 21:10: no twist trial
tonight; the lab's morning series (66 trials, seed 20261120, from 07:00 when the lights are steady; lab commit 5707381ce,
lab/series_morning_order.json) runs block 1 = 11 layouts clear of finger contact (incl. 4 replacements and the anchor) +
the anchor's stage E plan, block 2 = the four layouts whose index and middle links touch on the CAD collision spheres in
the simulated twist ((135,65) g0, (115,70) g7, (105,60) g9, (85,60) g14; found 21:5x, 86bc736b6), block 3 = six
non-touching stage E plans; block 4 = the CEM set when sent. Over the 60 best deployable layouts 16 touch (y_sep <= 75,
x_sep >= 85); the hand's joints on 11-17 read -2 to -5 mm at (105,70) g13 (met), 4-9 mm at (70,110) g0 (sim 17 mm).
Fingertip boards vs simulation at (70,110) g0 (c27f052f8): thumb 2.5 vs 2.1 N steady; index 0.62 vs 1.82 N, ~40-50 %
of the task; middle in contact 62 % vs 100 %. Stage P (plants on all 15 proposed grasps) done. Stage F (lane4) at
iteration 3 of 6 at 22:04, ends ~23:00, then cem_plans and the page (LANE4_DONE). Trial scorer:
sim/20261120-trials_vs_prediction.py (rho per model and per plant).
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Check the CPU lane: lane.log, heartbeat.json (rewritten after every run), LANE_DONE / LANE_STOPPED / FAILED_<… Done 11-19 21:11: hand-4a 20:52: stages A-D and the report done (15424+5784+720+3408 runs, 148 errors); report rerun 20:5x with exclusion counts and candidates (picks unchanged)
- [x] 2. When LANE_DONE exists: run the report (sim/20261119-twist_ranking_report.py) if lane.log says it was skipped,… Done 11-19 21:11: hand-4a: summary, proposal, stage E data and plans pushed by lane5 (6e925ca45, 21:06:59)
- [x] 3. When logs/20261119-twist_layout_ranking/LANE3_DONE exists (or LANE2_DONE with lane3 dead: run lane3.sh's two … Done 11-19 21:11: hand-4a: page rebuilt with the film, prose pass (mean 18.2 words, 0 % over 30), committed 6bf99032b; plan list relabelled 60dceff1c
- [x] 4. Publish the page as an artifact (Artifact tool, icon chart; films as files via python3 scripts/docs/page_film… Done 11-19 21:11: hand-4a 21:2x: artifact https://claude.ai/artifact/XiBYaBDci8QPCjNzyZqSue (07a902cd5); lab told 21:1x; first pass of the 11 default plans started 21:08 (seed 210748)
- [ ] 5. When logs/20261119-twist_layout_ranking/LANE4_DONE exists: read data/20261119-cem_refine.json (stage F picks: start vs pick at seeds 0-2, fresh seeds, friction 0.6; the latitude must stay near the start's) and the CEM plans in data/20261119-trial_plans.json (role 'cem ...'); rebuild the page if lane4's page step failed; commit by path (rows_F stays in logs/; the plans 20261119-rank_*_cem_*; data/; the page), push, republish the page to its artifact URL; SendMessage the lab the CEM plans as a second set to run after the default plans of the same layouts (same layouts, refined grasp and forces; their predictions in meta.ranking).
- [ ] 7. When the lab's first pass is done (33 trials of the 11 default plans in docs/experiments/20261115-hom_bench_trials/runs/ from 20261119-2108, the lab's commit of its corvus records): run .venv/bin/python docs/experiments/20261119-twist_layout_ranking/sim/20261120-trials_vs_prediction.py (writes data/20261120-trials_vs_prediction.json: Spearman rho of measured vs predicted tracking over the layouts), rebuild the page (its section 'Trials on the hand' appears with that file), commit by path, push, republish the artifact XiBYaBDci8QPCjNzyZqSue from docs/experiments/20261119-twist_layout_ranking/20261119-sphere_twist_layout_ranking.html, and write rho with its interval into this note's State for the owner.
- [x] 8. After step 5 (stage F's plans on GitHub): send the lab (SendMessage 'Physical model vs GP calibration') the m… Done 11-19 22:05: hand-4a 22:1x: the morning order was set with the lab (5707381ce, lab/series_morning_order.json): block 1 = 10 clear layouts (7 proposed + 4 replacements minus none; 11 incl. anchor) + the anchor's stage E plan, block 2 = the four touching layouts (b2touch), block 3 = the six non-touching stage E plans; the CEM set goes as block 4 once its plans pass the CAD check
- [ ] 9. When stage F's CEM plans exist (after step 5's cem_plans.py): run HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 CUDA_VISIBLE_DEVICES= .venv/bin/python docs/experiments/20261119-twist_layout_ranking/sim/20261119-cad_finger_gaps.py --plans <the 20261119-rank_*_cem_*_plan.json names> --no-hand --merge (through resguard, taskset on a free core); in data/20261119-trial_plans.json set cad_least_mm and cad_touch (< 2 mm) on the CEM entries; rebuild the page, commit, push, republish; send the lab the CEM plans that do not touch as block 4 (plan names, layouts, predictions, CAD gaps), and say which touch.
Standing directions (for every step):
- The hand, the UR5 and the HOM servers are the lab's (corvus session 'Physical model vs GP calibration'): never start a server, a trial or any motion from here; send plans and predictions only.
<!-- /ap:steps -->
Claimed: docs/experiments/20261119-twist_layout_ranking/, logs/20261119-twist_layout_ranking/, scripts/hom/twist_ranking*
Waiting on owner: the trial count for 11-20 (proposed: about 10 layouts spread over the predicted range, 2 grasps each,
3 trials per pair); switching tip_contacts.OWN_POSE to the (70, 110) GP of 11-19.
Do not: run anything on the hand or the UR5; start a GPU job while a GPU lane runs; change the trials' defaults.
