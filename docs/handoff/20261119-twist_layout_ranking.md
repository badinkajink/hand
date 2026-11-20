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
State: 11-20 08:5x (hand-4a; autopilot disarmed 08:0x, the task is back in this window). lane6 ended 01:38: 214 of the
404 fitting layouts have a feasible grasp (data/20261120-feasible_ranking.json; its seeds 1-2 were scored 0 until the fix
of 08:1x, 0ec04cd54). Owner 08:0x, "Sure on all three decisions": block A4feas (20 predicted-good feasible layouts, 2
trials each; plans pushed 0ec04cd54) after A1, the 19 infeasible layouts last at one trial each; the lab session has not
taken it up (corvus entries unchanged at 08:4x; the lights came on at 08:37 only briefly). Controller rematch
(docs/experiments/20261120-controller_rematch/, dc5852e71): the GPU cal75 plant in the stem scenes matches the CPU plant
(HOM 0.390 against 0.395); HOM's best shared force 1.0 N, spin weight 0.25: 0.660 at seeds 10-12 against 0.386 at the
defaults. Queues in logs/20261120-controller_rematch/: lane_sense.sh (HOM with the measured tracker at 30 and 60 fps,
pose filter) -> lane_gpu.sh (E2 grid, planners) -> lane_uhas.sh (two UHAS trainings); the clearance term on the CPU plant
(sim/20261120-clearance_hom.py). Decision log docs/notes/20261120-twist_ranking_decisions.md is current.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Check the CPU lane: lane.log, heartbeat.json (rewritten after every run), LANE_DONE / LANE_STOPPED / FAILED_<… Done 11-19 21:11: hand-4a 20:52: stages A-D and the report done (15424+5784+720+3408 runs, 148 errors); report rerun 20:5x with exclusion counts and candidates (picks unchanged)
- [x] 2. When LANE_DONE exists: run the report (sim/20261119-twist_ranking_report.py) if lane.log says it was skipped,… Done 11-19 21:11: hand-4a: summary, proposal, stage E data and plans pushed by lane5 (6e925ca45, 21:06:59)
- [x] 3. When logs/20261119-twist_layout_ranking/LANE3_DONE exists (or LANE2_DONE with lane3 dead: run lane3.sh's two … Done 11-19 21:11: hand-4a: page rebuilt with the film, prose pass (mean 18.2 words, 0 % over 30), committed 6bf99032b; plan list relabelled 60dceff1c
- [x] 4. Publish the page as an artifact (Artifact tool, icon chart; films as files via python3 scripts/docs/page_film… Done 11-19 21:11: hand-4a 21:2x: artifact https://claude.ai/artifact/XiBYaBDci8QPCjNzyZqSue (07a902cd5); lab told 21:1x; first pass of the 11 default plans started 21:08 (seed 210748)
- [x] 5. When logs/20261119-twist_layout_ranking/LANE4_DONE exists: read data/20261119-cem_refine.json (stage F picks:… Done 11-19 22:58: hand-4a 22:5x: stage F done 22:50 (CEM picks 0.602 full / 0.604 knots at fresh seeds vs starts 0.441; stage E's grid 0.606); 19 CEM plans, all load (knots reach task_prm); committed 463cc988f; page republished (v3)
- [ ] 7. When the lab's first pass is done (33 trials of the 11 default plans in docs/experiments/20261115-hom_bench_trials/runs/ from 20261119-2108, the lab's commit of its corvus records): run .venv/bin/python docs/experiments/20261119-twist_layout_ranking/sim/20261120-trials_vs_prediction.py (writes data/20261120-trials_vs_prediction.json: Spearman rho of measured vs predicted tracking over the layouts), rebuild the page (its section 'Trials on the hand' appears with that file), commit by path, push, republish the artifact XiBYaBDci8QPCjNzyZqSue from docs/experiments/20261119-twist_layout_ranking/20261119-sphere_twist_layout_ranking.html, and write rho with its interval into this note's State for the owner.
- [x] 8. After step 5 (stage F's plans on GitHub): send the lab (SendMessage 'Physical model vs GP calibration') the m… Done 11-19 22:05: hand-4a 22:1x: the morning order was set with the lab (5707381ce, lab/series_morning_order.json): block 1 = 10 clear layouts (7 proposed + 4 replacements minus none; 11 incl. anchor) + the anchor's stage E plan, block 2 = the four touching layouts (b2touch), block 3 = the six non-touching stage E plans; the CEM set goes as block 4 once its plans pass the CAD check
- [x] 9. When stage F's CEM plans exist (after step 5's cem_plans.py): run HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 CUD… Done 11-19 22:58: hand-4a 22:5x: CAD check of the 19 CEM plans: 6 at the four touching layouts touch (0-1.3 mm), 13 clear (7.5-19.5 mm); block 4 sent to the lab (the anchor's two CEM plans first)
- [ ] 12. When logs/20261119-twist_layout_ranking/LANE6_DONE exists: read data/20261120-feasible_ranking.json (each fitting layout's best grasp whose simulated twist keeps the fingers 10 mm apart on the CAD spheres, at seeds 0-2; layouts with none score 0). Add a page section 'Layouts judged by their feasible twists' to scripts/hom/twist_ranking_page.py: the feasible landscape figure (as Figure 1, feasible score), how many layouts have a feasible grasp, the top feasible layouts, and rho between the feasible ranking and the ranking with collisions allowed. Rebuild, commit, push, republish the artifact. Owner 23:5x: 'we should not be filtering bad hands because of collisions from infeasible trajectories, we should be judging them by their feasible trajectories'.
Standing directions (for every step):
- The hand, the UR5 and the HOM servers are the lab's (corvus session 'Physical model vs GP calibration'): never start a server, a trial or any motion from here; send plans and predictions only.
- The lab's morning series (from 07:00 when the lights are steady; lab/series_morning_order.json): block 1 = 11 layouts clear of finger contact x 3 trials + the anchor's stage E plan; block 2 = the four touching layouts (b2touch); block 3 = six stage E plans; block 4 = 13 CEM plans. Score with sim/20261120-trials_vs_prediction.py (--since 20261120-060000): rho over block 1's 11 default-force layouts is the ranking test; blocks 2-4 are compared with their layout's default plan. Report each touching layout's trials separately (does the CAD check's collision show on the hand?).
- Timing (lab, 23:1x, a248c314d): 105 trials at about 4.5 min each from 07:00; block 1's 33 default-force trials end near 09:30, the anchor's set (default, stage E, CEM knots, CEM full at (70, 110)) near 10:10, then block 2, block 3, block 4's other eleven CEM plans; blocks 3 and 4 are cut first if the day runs short. Score block 1 once its 33 trials are in runs/; rescore as later blocks land.
- Owner 23:5x asked whether the project has run away from them. Keep docs/notes/20261120-twist_ranking_decisions.md current (one line per decision: when, who, what, why; and 'Waiting on owner'), start no new thread without the owner's go except fixes that protect the trials, and lead every report with the two or three results that matter.
<!-- /ap:steps -->
Claimed: docs/experiments/20261119-twist_layout_ranking/, logs/20261119-twist_layout_ranking/, scripts/hom/twist_ranking*
Waiting on owner: the trial count for 11-20 (proposed: about 10 layouts spread over the predicted range, 2 grasps each,
3 trials per pair); switching tip_contacts.OWN_POSE to the (70, 110) GP of 11-19.
Do not: run anything on the hand or the UR5; start a GPU job while a GPU lane runs; change the trials' defaults.
