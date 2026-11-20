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

## Waiting on owner

- The bench (since 12:36): someone at the bench stands the sphere upright near the table's centre; the series restarts
  by itself.

- The smoothed budget on the bench (`hom_server --budget-joint-tau 0.1 --target-tau 0.04`) from the next block boundary,
  yes or no (asked 12:1x). Without it HOM stays as it was.

- A feasible landscape at HOM's tuned force (1.0 N or 0.75 N): the 404 layouts' grasp search again, on six CPU cores,
  about 4 h. The landscape of 01:38 holds for 2.5 N only (09:36).
- The tracker's camera mode: 60 fps at 960 x 540 once the moving-block test is done (the lab's), with the simulated
  effect from lane_sense (09:36).
- tip_contacts.OWN_POSE: switch to the (70, 110) GP of 11-19 or not. The lab found no layout dependence (5.9-8.0 uT).
