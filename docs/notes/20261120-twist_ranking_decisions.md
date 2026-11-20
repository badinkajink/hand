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
  layout's best collision-free grasp is being searched (`sim/20261120-feasible_grasps.py`). The touching layouts of the
  morning list get those grasps where one exists.

## Waiting on owner

- Whether to swap the touching layouts' plans (block 2, A2) for their feasible grasps before 11:00 (proposed: yes).
- A controller rematch on the calibrated plant, with each method's force knob tuned: HOM, annealed and predictive
  sampling, and UHAS with a cost on target rate and the torque budget. This would run after block A's results.
- tip_contacts.OWN_POSE: switch to the (70, 110) GP of 11-19 or not. The lab found no layout dependence (5.9-8.0 uT).
