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
- 08:4x: the lab has not taken up A4feas or the new order: corvus's entries file is unchanged since 23:30 and its clone is
  at fdd0f621e. The lights came on at 08:37, so the series starts near 08:47 with A1 as before. A4feas and the new order
  must be in before A1 ends, near 13:00.

## Waiting on owner

- The lab session (Remote Control) has not answered the two messages of 08:0x. Open it or tell it to splice A4feas after A1
  and move the infeasible blocks to the end at one trial each; or allow hand-4a to edit
  `series_entries_morning.txt` on corvus between trials.
- The tracker's camera mode: 60 fps at 960 x 540 once the moving-block test is done (the lab's), with the simulated
  effect from lane_sense.

- tip_contacts.OWN_POSE: switch to the (70, 110) GP of 11-19 or not. The lab found no layout dependence (5.9-8.0 uT).
