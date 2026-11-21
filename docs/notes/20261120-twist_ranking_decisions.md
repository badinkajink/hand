# Decisions of the twist layout ranking, 2026-11-19 to 11-20

One line per decision: when, who decided, what, why, and where the detail is. "Waiting on owner" lists the open choices.
Detail: page `docs/experiments/20261119-twist_layout_ranking/20261119-sphere_twist_layout_ranking.html`.

## Decided

- 17:2x, owner: rank all sphere-twist layouts on the calibrated simulation overnight and test the ranking on the hand.
- 17:4x, hand-4a: the model is the one of the 11-17 blind predictions (0.437 against the hand's 0.447). Layouts are
  ranked over 482 lattice layouts x 16 ground-clear grasps, with point contact and the pads (stages A-D).
- 18:1x, owner: refine HOM's forces and grasp too (stages E, F); go ahead with trials once the analysis is done.
- 20:52, hand-4a: 10 layouts spread over the predicted range plus the 11-17 layout (70, 110). These were deployable by
  the three-ball finger test.
- 21:06, hand-4a: stage E's lower forces (1.0-1.25 N in place of 2.5 N) as a second plan per layout. In simulation they
  raise tracking from 0.44 to 0.61 at fresh seeds.
- 21:10, lab: the lights went off, so no trial ran; the series runs from 07:00.
- 21:5x, hand-4a (after the lab's finding at (70, 90)): the CAD collision spheres along the simulated twist replace the
  three-ball test. Four proposed layouts touched index to middle, ranks 1, 2, 5 and 144. Four clear layouts replaced
  them in the first block, and the touching four ran later as block 2.
- 22:5x, hand-4a: the CEM plans that keep the fingers apart became block 4. The CEM gains no more than stage E (0.60
  against 0.61).
- 23:2x, owner: a denser spread over the predicted range, the bad end included. 22 layouts and 3 droppers were added; the
  morning list is 155 trials in about 12 h, block A (the correspondence test) first.
- 23:3x, owner: a protocol for deployment. Written as `.claude/skills/hand-deploy/SKILL.md` and
  `scripts/hom/hand_deploy.py`.
- 23:5x, owner: judge layouts by their feasible trajectories; do not filter them for collisions of one trajectory. Each
  layout's best collision-free grasp is searched (`sim/20261120-feasible_grasps.py`). Result for the 19 touching or
  dropping layouts of the morning list: none has a grasp (of 16) whose simulated twist keeps the fingers 10 mm apart.
  The best reaches -0.0 to 9.2 mm, and the hand runs 8-13 mm closer. Under HOM they have no feasible twist; their
  feasible prediction is 0. The search over all 404 layouts runs overnight (lane6, done near 02:40).
- 01:38, lane6: 214 of the 404 fitting layouts have a grasp whose simulated twist keeps the fingers 10 mm apart under
  both models with the sphere held, and 94 keep 15 mm (`data/20261120-feasible_ranking.json`). The best feasible layout
  is predicted 0.536, (105, 150) g7; the best layout overall, which touches, 0.637.
- 08:0x, owner: "Sure on all three decisions. We don't have to run the full infeasible set, we should get more coverage
  among the probably good and feasible set of hands first." Also: "Wish you would've run the control rematch while I
  was sleeping."
- 08:0x, hand-4a, relayed to the lab with the owner's words (plans pushed 08:07, 0ec04cd54): a block A4feas of 20 layouts after A1, 2 trials each (40
  trials). They are predicted 0.42-0.54 (median 0.463), keep 15 mm or more under both models at their best feasible
  grasp, hold the sphere under all six plants, and lie between the morning list's layouts on the lattice
  (`sim/20261120-feasible_proposal.py`, `data/20261120-vet_feasible.json`). The 19 infeasible layouts run last at one
  trial each (19 trials in place of 42), block 2 first; they are the first to cut. The (70, 110) anchor set, b3E and
  b4cem keep their places after A4feas.
- 08:0x, owner's go: the controller rematch and the collision-aware HOM start today, in simulation on the
  workstation.
- 08:1x-08:3x, hand-4a: the rematch's plant is the GPU port of the calibrated plant (`scripts/hom/hom_gpu.py` with
  HOM_GPU_PLANT=cal75 and HOM_GPU_SCENE=stem; `scripts/hom/controller_rematch.py`; study folder
  `docs/experiments/20261120-controller_rematch/`). Over the 45 morning layouts with a simulated gap of 2 mm or more, its
  HOM scored 0.390 against the CPU plant's 0.395 with the pads (rho 0.71), 0.389 against 0.400 with point contact
  (rho 0.64). Over stage E's 24 force settings on 7 shared layouts, rho 0.99 against the CPU, mean difference 0.011-0.014.
  The cell's post scene had dropped the sphere before the lift in 38 of 135 runs.
- 08:2x, owner: "Why don't we use fingertip sensing in these rollouts?" With the exact object pose, exact fingertip
  contacts change nothing: 0.388 with and without at 2.5 N, 0.649 with and without at 1.0 N (45 layouts, seeds 0-2).
  Under the tracker model, the simulated boards add 0.017 at 1.0 N; HOM's force loop on them lowers the score to 0.413
  with 46 of 135 drops.
- 08:3x, hand-4a: the tracker model of the 11-13 MPC page (0.5 mm and 1 deg fresh noise per frame, 40 ms late) is about
  20 times the jitter the lab measured on 11-13. The 0.538 quoted for stage E under it (08:2x) is withdrawn until the runs
  with the measured tracker are in (`logs/20261120-controller_rematch/lane_sense.sh`).
- 08:4x, owner: "why don't we run tracking at 60hz? Isn't the verdict still out on whether the complementary pose filter
  on the contacts helps too? I guess geometric contact from the tag is ultimately better." The lab measured 960 x 540 at
  60 fps on 11-13 (22 ms frame age against 42 ms, every tag decoded) and left the switch until a test with the block
  moving. lane_sense runs HOM with the measured tracker at 30 and 60 fps, with and without the pose filter.
- 08:4x, hand-4a: the lab has not taken up A4feas or the new order: corvus's entries file is unchanged since 23:30 and its clone is
  at fdd0f621e. The lights came on at 08:37, so the series starts near 08:47 with A1 as before. A4feas and the new order
  must be in before A1 ends, near 13:00.
- 08:5x, hand-4a: a clearance term in HOM's Eq. (2) on the CPU plant (`sim/20261120-clearance_hom.py` of the rematch;
  rows `logs/20261120-controller_rematch/rows_clear.jsonl`): rows that open a finger-to-finger gap below d_s on HOM's own
  link spheres, at seed 0 under both models with the CAD check. At d_s 20 mm and weight 10, 5 of the 17 near-touching
  layouts of the morning list keep 10 mm (none without it); the 10 clear layouts kept their gaps, and 9 of them their
  scores. The simulation's best layouts owe their scores to the close fingers: (135, 65) g0 fell from 0.610 to 0.039 and
  still came within 9.3 mm; (115, 70) g7 kept 13.8 mm at 0.317 (0.552 without).
- 08:5x, hand-4a: the queues asked resguard for 12 GB per run where a run holds 2 GB, so a CPU job beside them closed the
  gate (two refusals at 08:50 and 08:52); they were stopped between runs and restarted at 4-8 GB.
- 09:17-09:21, lab: the series started with the original order and ran (90, 135) g3 t1 and t2 (twist -11 to 53 and -6
  to 55 deg, rise 10.2 mm). It stopped in t2's putdown: the levelling turn stopped when the sphere moved 2.18 mm against
  the hand (limit 2 mm). The sphere hung 49 mm above the table until 10:38; servo 0 reached 52 C, and servos 1, 3 and 6
  flagged overload. A putdown without levelling set it down at 10:39. The lab session's login had reached its four-week
  limit before 08:04, so it could not act on the messages of 08:0x until the owner logged in at 10:33.
- 09:36, hand-4a: HOM with the tracker as measured on 11-13 (45 layouts, seeds 10-12, pads; `logs/20261120-controller_
  rematch/lane_sense.sh`). At 2.5 N: exact pose 0.386 (3 of 135 dropped), 30 fps 0.362 (11), 60 fps 0.382 (5). At 1.0 N
  and spin weight 0.5: exact 0.649 (0), 30 fps 0.539 (23), 60 fps 0.583 (13). The pose filter changed no score by more
  than 0.013; at 60 fps and 1.0 N it cut target reversals from 2.18 to 0.96 per joint per second. The 0.5 mm calibration
  offset changed scores by 0.01 or less. The 0.538 quoted at 08:2x stands: with the measured jitter, 30 fps gives
  0.539-0.552 at 1.0 N, so the frame's delay costs the score, not the old model's noise.
- 09:37, hand-4a: predictions under the 30 fps tracker for every morning plan with one force setting
  (`docs/experiments/20261120-controller_rematch/data/20261120-tracker_predictions.json`), written after the first A1
  trials (none read) and before any stage E trial. Most stage E plans keep their exact-pose prediction; (115, 160) g8 at
  1.0 N drops the sphere at all three seeds under the tracker (0.000 against 0.691), and (110, 155) g3 falls to 0.521
  (0.739).
- 09:36, hand-4a: the grasp search of the 17 near-touching layouts (all grasps, seed 0, both models, CAD check). With the
  clearance term at 2.5 N, 7 have a feasible grasp, the best scoring 0.37. At 1.0 N without the term, 4 have one, scoring
  0.19-0.70: (70, 155) 0.699, (65, 145) 0.685, (55, 130) 0.664. The simulation's top layouts, (135, 65), (115, 70) and
  (105, 60), have none at 1.0 N. Feasibility depends on HOM's force: the landscape of 01:38 holds for 2.5 N only.
- 10:43, lab (the owner's order of 08:0x): the series restarted with A4feas after A1, 170 trials: the rest of A1 (51),
  A4feas (40, t1 and t2 back to back), the (70, 110) anchors (9), b3E (18), b4cem (33), then b2touch, A2bad and A3drop at
  t1 only (19). Order `lab/series_20261120_day_order.json` on corvus; rows still go to
  `20261120-series_morning_seed20261120.jsonl`. hom_series now runs the putdown without levelling when the levelling stops.
- 10:45, lab: stopped after trial 1 of 170, (90, 135) g3 t3 (twist -4.3 to 58.1 deg, rise 8.4 mm, drift 22.1 mm). After
  the twist the sphere was 20.9 mm off its nominal position and tilted 49.6 deg; the levelling turn stopped at its 12 s
  limit. The putdown without levelling released the sphere 18.1 mm off and tilted 34.9 deg, then refused both paths back
  to zero: the index tip end would come 4.0 and 4.9 mm into the block. The hand held its open pose over the sphere.
- 10:4x, hand-4a: the planners on the GPU plant with the exact pose (45 layouts, pads, seeds 0-2 / 10-12; the planners
  roll out the plant itself at a 2 ms step from its exact state). Predictive sampling (32 samples) 0.641 / 0.637 at 2.5 N (7 and 6 of 135 dropped), 0.636 /
  0.637 at 1.0 N. Annealed sampling (16 samples, 2 iterations) 0.582 / 0.578 at 2.5 N and 0.610 / 0.583 at 1.0 N, with 9-15
  drops. Residual annealed sampling on HOM 0.776 / 0.768 on HOM at 2.5 N (no drops) and 0.829 / 0.832 on HOM at 1.0 N (2 and
  1 of 135 dropped; 0.841 / 0.835 over the first 81 and 79 runs). Tuned HOM alone: 0.681 / 0.682 at 0.75 N. The force cost (weight 0.1, target 1 N) changed the
  samplers by 0.03 or less (`data/20261120-rematch_summary.json`).
- 10:50, hand-4a: the GPU queue reordered (`logs/20261120-controller_rematch/lane_gpu2.sh`). After residual annealing on
  HOM at 1.0 N: predictive and annealed sampling with the tracker's 40 ms delay (about 15 min), then the two UHAS trainings
  with their scoring (lane_uhas, until about 15:45), then the four annealing runs at 64 samples and 4 iterations (about
  1 h each), the first to cut. Residual annealing does not take the planner delay yet (`--plan-delay` is implemented for the
  joint-space planners only).
- 10:59, lab: the series restarted from A1_p35_x105y100g15_t1 in the same order. The owner had reset the arm; a letgo,
  a clear and a lateral move over the sphere put the hand back in position. All three trials on (90, 135) g3 moved the
  sphere in the grasp: rise 8-10 mm, drift 10-22 mm. t3's row is in the rows file with a "note" field (runbook ce38b11c5).
- 11:12, lab: three of the day's first five trials started 12-25 deg off the plan's grasp (the sphere's axis against z_P
  at ready): (90, 135) g3 t2 12.2 deg and t3 13.0 deg, (105, 100) g15 t2 25.5 deg; the good starts were 0.6 and 1.2 deg,
  and the 11-19 trials started within 0.3-1.5 deg. A putdown had stood a tilted sphere up by tilting the hand, and letgo
  re-levelled the hand only within its --max-level. Since 11:12 hom_series levels the hand before every recentre and writes
  ready_tilt_deg (9a98f3cad). The series restarted at 11:12 at A1_anchor_x70y110g0_t1, order unchanged.
- 11:2x, hand-4a (the lab's request): `sim/20261120-trials_vs_prediction.py` keeps trials with a start tilted more than 5 deg
  out of the plan means and rho and lists them under "tilted_starts"; today's corvus records and series rows were copied to
  `docs/experiments/20261115-hom_bench_trials/runs_corvus/`.
- 11:4x-11:5x, owner: "id like to prioritize (within reason) the 60fps tracking"; videos to the workstation; reset
  automation on the UR5 ("we really want to maximize real world throughput"); "yea, im down to recompute feasible layouts";
  tip_contacts.OWN_POSE stays. Relayed to the lab with the owner's words. Owner, leaving: "smooth control enables more
  predictable control, im highly confident about this."
- 11:4x, hand-4a: the feasible landscape at 1.0 N (spin weight 0.25) runs (`sim/20261120-feasible_grasps.py --prm ...
  --tag _f1`, rows `logs/20261119-twist_layout_ranking/rows_FG_f1.jsonl`, 10 workers from 12:00). The UHAS trainings ran
  out of GPU memory at 2048 worlds on the calibrated plant (11:07); they run at 1024 worlds from 11:48 (`lane_gpu3.sh`).
  The full annealed-sampling runs are cut.
- 11:5x, lab: 60 fps (960 x 540, 5 ms exposure) since 11:5x, HOM unchanged; every row carries "camera". Static check:
  30 fps sd 0.020-0.026 mm and 0.036 deg, frame age 39 ms; 60 fps sd 0.040-0.057 mm and 0.089 deg, frame age 22 ms.
- 12:1x, hand-4a: the jitter (`docs/experiments/20261115-hom_bench_trials/analysis/20261120-command_jitter_replay.py`,
  `20261120-command_jitter_sim.py`). The hand's joint readings flicker by 1-2 encoder steps: 0.4-0.5 deg RMS about a
  7-period mean, reversing in 62-70 % of periods, before and after the budget. HOM's own targets on the same readings
  (the server records replayed offline without the budget) reversed in 7 % of periods on 11-16 and 17 % under 0.56 N m.
  The budget sets every target as the measured joint plus an offset, so the flicker passes into the targets: 32 % of
  periods reversed at 0.84 N m and 45 % at 0.56 N m (39 records; the replay reproduces the recorded rate within 1 point).
  The simulation's joint readings are exact: the same plans reverse in 0.5-2.3 % of periods with the budget, and in
  0.4-5.3 % with the encoder step or one period of delay added. On the bad layouts HOM's own targets also swing (2-6 deg
  RMS about the 7-period mean on (115, 160) g8 and its neighbours).
- 12:1x, hand-4a: smoothing options in the HOM server, off by default (`hom_bench.HomBench`, `hom_server --budget-joint-tau`
  and `--target-tau`; written atomically, the default path replays the records unchanged). Replayed open loop on the
  39 records: the budget's offsets from joints low-passed at 0.1 s cut the reversals from 45 % to 21 % (median); with HOM's
  targets low-passed at 0.04 s as well, to 11 %.
- 12:14, lab (33ea534d0): the series restarted at A1_r109_x70y90g1_t1. letgo's partial release had left the fingers 35 deg
  from zero, and layout refuses a pose that is neither zero nor open; the series now straightens them with a checked zero
  and lays out again. 12:0x (36adb1cfd): after letgo --tilt-ok the series had read the sphere's tilt in a still-tilted
  hand (34 deg) as toppled; the sphere stood 1.7 deg off vertical. A1_r3_x115y160g8_t3's putdown field ("settled at 34.27
  deg") is that misreading. The day's first 89.5 min: stops 78 %, a clean cycle 142 s
  (`docs/experiments/20261115-hom_bench_trials/analysis/20261120-series_time_table.py`).
- 12:1x, hand-4a: the smoothing flags wait for the owner's yes (asked 12:1x); the lab will not switch without it.
- 12:2x, hand-4a: the sim-to-real page, first version
  (`docs/experiments/20261120-twist_sim_to_real/20261120-twist_ranking_sim_to_real.html`, builder
  `scripts/hom/twist_sim_to_real_page.py`): 5 layouts at the default force so far, rho -0.10; the anchor (70, 110) 0.367
  against 0.376; (115, 160) g8 0.082 against 0.543. Rebuilt through the evening; published from the personal login
  tonight (hand-off note `docs/handoff/20261120-twist_sim_to_real_evening.md`, not armed).
- 12:2x, hand-4a (owner, 11:5x: "what if we let the residual annealed sampling also modulate the initial force"): a
  force channel for the residual planners (`inner_loop_mpc.py --res-force K`: a tenth planned number per knot scales
  HOM's commanded finger forces by 1 + K u, clipped to 0.5-1.5). Queued after UHAS with a smoke run
  (`logs/20261120-controller_rematch/force_res_arms.sh`); the full annealed-sampling runs are cut.
- 12:28, hand-4a: predictions of the 70 morning plans under the smoothed budget, written before any smoothed trial
  (`docs/experiments/20261119-twist_layout_ranking/data/20261120-smoothed_predictions.json`, c5844e356; both models,
  seeds 0-2): mean 0.393 with exact joint readings and 0.384 with the hand's flicker, against 0.373 for the plans'
  predictions; rho 0.89 and 0.87 with them.
- 12:36, lab: stopped, needs a person. A1_p25_x85y170g2_t1 dropped the sphere in its twist (tracked twist -144 to 5.5
  deg, drift 93 mm); it lies on its side. The putdown took it for held and started down until its guards stopped it
  (HEX-E at most 4.5 N); fixed in 252be514a (the putdown refuses an object more than 45 mm from the grasp). The pick-up
  failed: no pinch at (85, 170); at the (105, 70) pick-up layout a fingertip came down on the ball twice (1-2 tags see the
  lying sphere, 6.5 mm between pad and ball). The hand is safe (fingers straight, level, high). Relayed to the owner by
  push at 12:5x. Restart at A1_p25_x85y170g2_t2 once the sphere stands near the table's centre.
- 13:19, lab: colour snapshots of both cameras (`logs/20261116-hom_trials/20261120-snapshot/` on corvus), sent to the
  owner. The ball lies on its side among the cables, centre about (-139, -318) mm in B and 23 mm up, 118 mm from the
  table's centre toward the arm's base.
- 13:2x, owner (relayed to the lab at 13:25 as the go for this recovery): "looks pretty salvageable to me. rotate the wrist
  to align with the stem (parallel), and set up for a two finger pinch. alternative: instead of pinch, slowly push, with
  one of the index or middle finger, the sphere. either along the stem's axis or perpendicular to it."
- 13:28-13:34, lab: the owner's recovery worked. push_sphere.py pushed the lying sphere 50 mm along its stem with the
  middle fingertip, off the cables (13:28); the pinch picked it up (13:31); the series restarted at 13:34 at
  A1_p25_x85y170g2_t2. That trial dropped the sphere again; the pick-up planner found no plan (the stem 47.5 deg off
  across x_P, the wrist's turn limit 45 deg). (85, 170) g2 dropped in both of its trials; hand-4a proposed to the owner to
  skip its third (13:4x).
- 13:54, hand-4a: UHAS's published-settings training on the calibrated plant finished at 1024 worlds (lane_gpu3); its
  scoring on the 45 layouts runs.
- 14:0x, hand-4a: UHAS with its published settings on the calibrated plant (1,024 worlds, 120 min): 0.659 / 0.632 at seeds
  0-2 / 10-12, 20 and 24 of 135 runs dropped, 24.1 command reversals per joint per second (HOM and the planners 1.9-2.3).
  Annealed sampling with 64 samples x 4 iterations at 1.0 N, its state 40 ms late (the run lane_gpu2 had started at
  11:09): 0.832 / 0.837, 3 and 2 drops, 2.2 s per replanning for a chunk of 58 worlds; with 16 x 2 samples 0.584 / 0.555.
  The rematch page carries them (Table 3, sorted, plain names).
- 14:13, lane_f1: the feasible landscape at 1.0 N (spin weight 0.25; `data/20261120-feasible_ranking_f1.json`). 196 of 404
  layouts have a grasp whose simulated twist keeps the fingers 10 mm apart under both models with the sphere held, against
  214 at 2.5 N (182 at both). Over those 182 the mean score rises from 0.407 to 0.644, and the two forces rank them with
  rho 0.20. The best feasible layouts move from (115, 155), (115, 160), (95, 105), (110, 155), (105, 150) at 2.5 N (0.54-
  0.55) to (85, 170), (90, 165), (80, 150), (85, 115), (85, 155) at 1.0 N (0.74-0.77). The sim-to-real page's Figure 3
  shows both.
- 15:01, lab: the series restarted at A1_r130_x95y90g14_t1 (entry 34 of 170); since 14:04, 23 trials ran and five stops
  were recovered on the bench. A1_r4_x110y155g3_t3 set the sphere down seated but tilted 22.7 deg; released, it toppled;
  a 5 N push freed it from the cables and the pinch stood it up (0.6 deg). Its row's putdown field ("settled at ?") is
  wrong: the sphere toppled. The lab's reading: the wide layouts lose the sphere on the hand, (115, 160), (110, 155),
  (85, 170) and (85, 160) (rise 12-29 mm, drift guard or drops, twists of 25-81 deg); the narrow and mid layouts hold it,
  (65, 110), (70, 90), (55, 90), (55, 75) and (100, 80) (rise 2-7 mm, twists of 35-83 deg). Drop recovery: on bare table
  the pinch pick-up worked first time all three times (13:31, 13:56, 14:58); among the cables a push or a pinch descent
  hit them five times (3.2-3.6 N). Request relayed to the owner at 15:0x: route the blue and red cable bundle away from
  the table's centre.
- 15:1x, hand-4a (refresh of 15:15, 36 clean trials on 17 layouts at the default force): rho 0.33 (interval 0.17-0.42),
  measured 0.075 below predicted on average. Split by the index-to-middle spacing y: under 150 mm (12 layouts) Pearson's r
  0.83, rho 0.56, mean absolute error 0.080; at 150 mm or more (5 layouts) the hand lost the sphere and the scores fell
  0.337 below the predictions. The measured score falls with y (rho -0.64), the predicted does not (0.03). The page's
  Figure 2 shows it; the feasible landscapes are now Figure 3.
- 16:1x, owner: "we should get some metrics on the pickup, drop rate incidence, pick up success, push->pickup
  mitigation" and, on smoothness, "doing so would kind of ruin this data, so i expect not". The smoothing stays off for
  today's series (the HOM server record of 16:12: budget 0.56 N m, budget_joint_tau and target_tau None).
- 16:2x, hand-4a (`docs/experiments/20261115-hom_bench_trials/analysis/20261120-recovery_metrics.py`, 68 trials to 16:12):
  the sphere left the grasp in 9 trials (13 %, corvus's drift guard). Putdowns: 52 clean (76 %), 9 stopped and finished by
  the lab session (13 %), 5 tilted releases (7 %), 2 lost in the trial (3 %). Pick-up grasps of a lying sphere: 4 of 9
  succeeded; 4 of 6 with a push first, 0 of 3 without; the 2 failures with a push were pushes that met the cables (14:49,
  14:53); 2 more pick-ups found no plan.
- 16:2x, hand-4a: the residual's force channel (--res-force 2.5) on HOM at 1.0 N: 0.832 / 0.848 at seeds 0-2 / 10-12 (1
  drop each) against 0.829 / 0.832 without; on HOM at 2.5 N it runs (0.850 / 0.841 over its first 9 and 7 runs, against
  0.776 / 0.768 without).
- 17:43, lab: held, needs a person (entry 90 of 170). A4feas_f76_x105y160g0_t2 toppled the sphere; the automated
  pick-up failed four times (17:24-17:41): its pose of the lying sphere jumped 11-13 mm and once flipped 110 deg behind
  the hand turned 62 deg, so every grasp descent stopped. The series had stood up two topples by itself (16:44, 17:00) and
  the lab session a third (17:24-17:30). Relayed to the owner by push with a camera frame at 17:4x; the cables still lie
  across the table. wait_upright.py restarts at entry 91 (A4feas_f17_x95y140g0) once the sphere stands still.
- 17:4x, hand-4a: the GPU queue finished at 16:40. The force channel over HOM at 2.5 N: 0.781 / 0.780 (3 and 1 drops)
  against 0.776 / 0.768 without; the 0.841 told the owner at 16:2x came from its first 7 runs. UHAS with limit and rate
  penalties: 0.709 / 0.724, 17 and 14 drops, 29.1 reversals per joint per second (published settings 0.659 / 0.632 and
  24.1). The hand's joint flicker added to the simulation does not reproduce the wide layouts' losses: (115, 160) g8 0.574
  -> 0.541, (110, 155) g3 0.571 -> 0.558, (85, 170) g2 0.275 -> 0.248, (85, 160) g1 0.334 -> 0.320, all held (seed 0,
  pads). Their cause on the hand is open.
- 17:5x, owner at the bench (by the lab session): "lets just fly blind here and go for the pinch". The lab's
  `pinch_pickup.py grasp --blind` measured the lying sphere once with the hand high and unturned, turned the wrist 62.4 deg,
  descended 124 mm (at most 1.6 N on the HEX-E) and closed without the tracker; the lift went 65.2 mm, the slip took the
  tilt from 59 to 10.7 deg, and the putdown left the sphere 0.7 deg from vertical at the table's centre (17:54, 6fb83e9c8).
  The 17:43 request for a person is withdrawn; the series restarts from entry 90 (A4feas_f17_x95y140g0_t1) near 17:58.
- 19:49, lab: held, needs a person to look at the hand (field 137 of 170, b4cem_x85y120_full_t1; the sphere upright,
  the hand clear). 19:22: the release after b4cem_x115y160_knots_t2 lifted the sphere 6.6 mm and it toppled; the watched
  pick-up lost it 72.5 deg into an 84.9 deg wrist turn and the blind pinch stood it up (0.7 deg). At 19:37, 19:42 and
  19:47 the open pose pushed the centred, upright sphere 8-15 mm toward -x/+y in P and tipped it 6-13 deg; the same plans
  had opened clear since 15:02. The middle finger's yaw stopped 3.2-3.8 deg short of the open pose at (115, 160): a
  finger, likely the middle, or its gantry carriage may not be where the station thinks (the blind pinch's squeeze, or a
  skipped step in the layout changes of 19:26-19:46). Relayed to the owner by push at 19:5x.
- 19:49, lab (114b50e74): the trials' wrist angle moved twice today, -69 deg until 13:35, -22 deg from 14:01 and -37 deg
  from 15:02, because the unturns used an overwritten turn_deg; the pick-up now measures the turn against the newest trial
  record. The trials' scoring should carry the wrist angle (camera views and cables differ with it).
- 20:0x, owner: "we finished for the day. time to wrap up, fill outa anlysis, queue up simulation studies?"
- 20:1x, hand-4a: friction sweep of the 48 plans tried (all but the CEM plans; both models, seeds 0-2;
  `data/20261120-friction_predictions.json`): against the hand, friction 0.4 gives rho 0.72, Pearson's r 0.72, mean absolute
  error 0.105 and the wide layouts 0.008 below their predictions; 0.5: 0.63 / 0.71 / 0.109; 0.6: 0.67 / 0.72 / 0.114; 0.8:
  0.55 / 0.63 / 0.128; 1.0: 0.48 / 0.57 / 0.136. The UR5 rig measured 0.37-0.44 for TPU on bare PLA; the scenes' 1.0 stands
  for grip tape, which the sphere does not have [wrong: the sphere is covered in grip tape, owner 20:2x; see 20:4x below].
  The sim-to-real page carries it (Table 2, the lede) and two films (the same layouts at 2.5 N and 1.0 N).
- 20:1x, hand-4a, queued overnight: lane_mu04 (GPU, `logs/20261120-controller_rematch/lane_mu04.sh`): HOM's force grid at
  friction 0.4 (0.75-2.5 N x spin weight 0.25 and 1.0) and two arms at 0.6, about 30 min; lane_f1mu04 (CPU, 12 workers,
  `logs/20261119-twist_layout_ranking/lane_f1mu04.sh`): the 404 layouts' feasible grasp search at 1.0 N and friction 0.4,
  about 2 h 10 min, summary `data/20261120-feasible_ranking_f1mu0.4.json`. The publish job of both pages re-armed on the
  personal login.
- 20:2x, owner: "the sphere does have grip tape on it though. so thats just the friction? is this with point contact or
  sphere packed contact model? is there a difference or are they still the same? let's get ready to trt smoothing i guess.
  i imagine there was much less jitter on the 1.0 N trials."
- 20:4x, hand-4a: the publish job held at 20:26 (the page said the sphere has no tape) and re-armed after the fixes below.
  The friction fit is an effective value. The friction moved only the wide layouts' predictions: for y under 150 mm the
  mean absolute error was 0.097 at friction 1.0 and 0.101 at 0.4; the wide layouts' measured less predicted went from
  -0.228 to -0.008. The two contact models fit alike at 0.4 (Pearson's r 0.70 point, 0.71 pads; mean absolute error 0.112
  each); a plan's two predictions differ by 0.023 on average at 1.0 and 0.051 at 0.4 (pads lower). The fingertip boards
  of 11-19 at (70, 110), 2.5 N (`data/20261119-tip_forces_vs_sim.json`): index median 0.62 N against 1.74 N simulated
  (0.36), middle 1.24 against 1.54 N and in contact 62 % of the periods against 100 %, thumb 2.50 against 2.12 N. Friction
  limits the tangential force at mu N, so the fitted 0.4 can stand for the index's and middle's lower forces; the tests are
  a drag of a pad on the taped sphere (UR5) and the boards' forces on a wide layout.
  Jitter by force (`analysis/20261120-command_reversals.py`, the commands sent, 132 trials of the day): reversals 27 % at
  1.0 N (21 trials) against 45 % at 2.5 N (88), the budget cutting HOM's targets in 11 % and 95 % of the periods; lower at
  1.0 N on each of the 6 layouts run at both forces at 60 fps (18-33 % against 39-55 %). The high-frequency amplitude
  depends on the layout: 0.23-0.29 deg at (70, 90) and (65, 110), 2.03-2.56 deg at the wide (110, 155) and (115, 160).
  Smoothing series proposed for the next bench day: the default and 1.0 N plans of (70, 90), (70, 110), (85, 120),
  (110, 155), (115, 160), one trial per plan per block, blocks smoothed / plain / plain / smoothed (40 trials; the first
  two blocks give a first read), server `--budget-joint-tau 0.1 --target-tau 0.04` for the smoothed blocks. Predictions
  written before the series (`data/20261121-smoothing_series_predictions.json`, both models, seeds 0-2, with the hand's
  joint flicker): the smoothing changes the predicted score by -0.013 on average at friction 1.0 (-0.068 to +0.001) and
  -0.014 at 0.4 (-0.120 to +0.049). The scorer now keeps smoothed trials apart (record header) and lists
  smoothed_vs_plain; `20261120-command_reversals.py` gives each trial's reversal share. Sent to the lab session.
- 20:4x, lab (c0abdb6a2): the series is ready on corvus. series_entries_20261121_smoothing.txt and
  series_20261121_smoothing_order.json hold 10 plans x 4 blocks (sm, pl, pl, sm), shuffled within each block (seed
  20261121), on ports 47201-47210. hom_series now restarts a server whose plan or flags differ from the entry
  (--serve-mode sm='--budget-joint-tau 0.1 --target-tau 0.04'), checks the flags in the server's command line, and stops
  when the wrist yaw is more than 3 deg from its first trial's (-36.5 deg now). Rows carry serve_flags and wrist_yaw_deg.
  The camera stays 960x540@60. hand-4a, 20:44: hom_serve_budget.sh on the workstation passes the arguments after the
  seventh to hom_server ("${@:8}"), at the lab's request. Before any trial: the middle finger needs a physical check (the
  owner saw it off zero while the station reads it within 0.3 deg). The lights went off about 19:5x.

## Waiting on owner

- The hand (since 19:49): someone checks the middle finger and its gantry carriage; opening the fingers pushes the sphere.

- The cables (lab, 15:01): someone routes the blue and red cable bundle away from the table's centre, so a dropped
  sphere lies on bare table, where the pinch pick-up works.
- The smoothing series (owner 20:2x: "let's get ready to trt smoothing"): proposed at 20:4x, ready to run once the
  middle finger is checked; the lab runs it with the owner's go on the day.


- A feasible landscape at HOM's tuned force (1.0 N or 0.75 N): the 404 layouts' grasp search again, on six CPU cores,
  about 4 h. The landscape of 01:38 holds for 2.5 N only (09:36).
- The tracker's camera mode: 60 fps at 960 x 540 once the moving-block test is done (the lab's), with the simulated
  effect from lane_sense (09:36).
- tip_contacts.OWN_POSE: switch to the (70, 110) GP of 11-19 or not. The lab found no layout dependence (5.9-8.0 uT).
