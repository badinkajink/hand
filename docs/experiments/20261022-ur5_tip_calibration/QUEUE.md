# Requests between the workstation and the lab computer for the UR5 calibration of the fingertip

The owner set up two Claude Code sessions on 2026-10-22. The session on the lab computer runs the calibration of
`README.md` and pushes each run's data to GitHub. The session on the workstation polls GitHub every 30 s, analyses each
push and asks for further runs in this file. Only the workstation writes this file. The lab session answers in its
commit messages and in `lab/NOTES.md`, which only the lab writes, so no file is edited on both computers.

## What the lab computer pushes

1. Pull before each commit: `git pull --rebase origin main`. The pull brings this file, new plans and the
   `.gitattributes` rule that stores `docs/**/*.bin` in Git LFS. Without the rule, the `ft.bin` of a full session
   (about 130 MB) exceeds GitHub's limit of 100 MB a file, and the push is refused.
2. After every run of the program (`preflight`, `first`, `dry`, `sphere1` and each resume), copy its session folder
   whole to `lab/<YYYYMMDD>-<time>-<name>/` in this folder. The folder has `events.jsonl`, `registration.json`,
   `touches.json`, `pivots.json`, `session.json`, `plan.json` and the three streams `ft.bin`, `robot.bin` and
   `board.bin`. Add the program's printout: `tmux capture-pane -p -t tipcal -S -5000 > printout.txt`.
3. During `sphere1`, push the small files once more when the register and pivot stages have printed (all but the
   three streams). The registration is then checked here while the grid runs.
4. Write in `lab/NOTES.md` what the program does not log: the caliper reading of the stack from the flange face to the
   HEX-E's tool face (35.0 mm in the plan), the placement as set, the cable routing, what the operator saw or heard,
   and the start time of each run.
5. Give the commit message the run's name and outcome in one line, for example "UR5 lab: first (preflight, homing,
   gravity, mount): gravity 1.74 N, onsets within 0.4 mm".

## Plans written on the workstation

A new plan goes into `plan/` with its date and purpose in the file name. Plans are JSON, which this repository stores in
LFS, and the lab clone skips LFS content (`.lfsconfig`). Fetch one with
`git -c lfs.fetchexclude= lfs pull --include=docs/experiments/20261022-ur5_tip_calibration/plan/<file>`. A new plan runs
in the simulator before it runs on the robot (README rule 2): `$PY $P --plan <file> --sim --name sim-<name>`.

## Requests

Each request names what to run, why, and the result that would change the next step. Its status changes here when the
push that answers it arrives.

- R1 (standing, 2026-10-22 19:30). Push the session folder of every run as above, beginning with `first`. The workstation
  compares each with Table 2 of the README and with the simulated session 151114-sim, and answers here within about
  10 minutes of the push. First answered by `lab/20261022-192715-first/` (ffc6ed71).

- R2 (done by the lab in 20bfc6e7, 19:40, by another route). A stream gap over 0.5 s aborts the action it falls in
  and is logged as `ft_gap`. An orientation read after a lost tare makes the sweep's residual exceed 0.1 N, which stops
  the session before any contact, and the resume re-tares the HEX-E and sweeps again whole. The offsets per restart
  proposed below are therefore not needed. The request as written at 19:38: make the HEX-E's restarts visible, and keep
  the gravity fit through one. In `first` the stream stopped for 7.22 s (19:27:40.36 to 19:27:47.59) while joint 6 turned from 69.0 deg
  at 5.1 deg/s. The DAQ's sample counter then went from 13702 to 102, and the readings came back 645.2 N higher in Fz
  with status 0x40000. grav-02 and grav-03 ended ok with that offset, so the sweep mixed two zeros. The change proposed:
  1. `OptoForceStream` counts a restart when a sequence number is lower than the one before, and the session logs an
     event `ft_restart` with its time, the sequence numbers before and after, and the status after.
  2. `check_ft` aborts an action during which the restart count changed, with the reason "HEX-E restarted".
  3. `stage_gravity` stores the restart count in each record, and `fit_gravity` fits one offset per count. A restart
     then costs only the orientation it interrupts.

  Test it in the simulator with a restart injected (a 7 s gap, then an offset step) once during gravity and once
  during a press. The presses already abort on such a step, through the 10 N limit, so their data are safe without
  the change.

- R3 (implemented 19:59 in the builder, see the end of this item; the plan, before the next run). Order each stage's
  actions by joint 6. Solved as the
  program solves them (IK nearest the homing joints, joint 6 at 1.8 deg), this plan turns joint 6 through about
  4,300 deg in all, and every turn moves the HEX-E's cable:

  | Stage | Actions | Joint 6 span (deg) | Travel in plan order (deg) | Travel sorted (deg) | Steps over 90 deg |
  |---|---|---|---|---|---|
  | Gravity | 12 | -178 to 122 | 989 | 300 | 2 |
  | Pad | 32 | -65 to 72 | 1425 | 137 | 6 |
  | Grid | 74 | -65 to 72 | 1062 | 137 | 1 |
  | Final gravity | 4 | -178 to 62 | 460 | 240 | 2 |
  | Mount, pivot, shear, roll, twist, final | 27 | -62 to 26 | 367 | 367 | 0 |

  Sorted by joint 6, the same actions need about 1,300 deg. The flank touch-offs alternate sides, so the pad stage
  turns 130 deg six times. The robot's own IK put grav-03 at +166.6 deg, 22.8 deg off its target orientation (the fit
  uses the reached orientation), so grav-03 to grav-04 turned 285 deg. The builder's clearance checks must cover each
  action's new neighbours, so the order belongs in `scripts/ur5_tip_calibration_plan.py` (an option such as
  `--order joint6`), with the grid's reference presses kept at their places in time. A new plan then runs in the
  simulator.

  Amended 19:44: the gravity sweeps go out and back, and only the pad and grid stages are sorted. Out and back means
  every other station, in order of joint 6, on the way out and the rest on the way back. For the 12 stations of
  `gravity` that is 559 deg of travel, against 959 deg in plan order and 285 deg sorted. When the stations are
  sorted, the direction of gravity changes in step with time, and a drift term (R6) is then poorly separated from the
  gravity map. With 5 mN of noise, the 1-sigma of the map's columns is 2.66 mN per g in plan order, 3.68 sorted and
  2.72 out and back.

  Implemented at 19:59 in `scripts/ur5_tip_calibration_plan.py`: `--order joint6`, and `--gravity-roll-offset 30`, which
  turns every gravity station's roll by 30 deg. Without the offset, two stations sit half a turn from the homing pose
  (grav-03 at +182 deg and grav-13 at -178 deg), where the robot's IK may take either side; the first run's 285 deg
  turn came from there. Built here with this cell's options from `lab/README.md` plus both new ones:

  | Stage | Travel in plan order (deg) | Travel ordered (deg) |
  |---|---|---|
  | Gravity, 13 stations | 1319 | 600 |
  | Pad, 32 | 1335 | 123 |
  | Grid, 74 | 995 | 499 |
  | Final gravity, 5 stations | 658 | 480 |
  | Shear, roll, twist | 340 | 340 |

  The plan has 157 actions, 4 dropped and no travel path short of clearance. With the offset, grav-08 and fgrav-04
  reach, and grav-11 and fgrav-05 do not. The grid's reference presses stay at places 0, 16, 32, 48, 64 and 73 of 74.
  The final sweep has 5 stations, one short of the program's fit, so it is checked here against the first fit. In the
  default order, the plan's list now follows the session's order of execution, all rolls before all twists. The
  builder used to interleave them, so its travel checks did not cover the transitions that the session makes. The
  gravity check now solves each station nearest the homing branch, as the program does. A full simulated session on
  this plan runs here now.

- R4 (in progress at 19:52: the owner found the connector itself loose and is tightening its coupling ring; the owner,
  before the next run). Test the HEX-E's cable on the robot. The dropouts began at joint 6
  = 69.0 deg turning at +5.1 deg/s and at 23.5 deg turning at -6.9 deg/s. Both angles lie inside the span of the
  presses, and both came mid-turn. That fits a connection that opens when the cable moves, more than a cable drawn
  tight at one end of the span. Reseat both connectors of the cable and relieve its strain at the sensor. Then, with
  the program stopped, turn joint 6 from -180 to +170 deg and back three times at 10 deg/s, 150 mm above the
  indenter, while one reader logs the stream. It passes with no gap over 100 ms and no counter restart.

- R5 (open, 19:38; the owner, when at the robot, not blocking). Check the HEX-E's force gain with a known mass. Its
  raw zero is at about 645 N on a 200 N sensor, and the analysis takes every force from its gain. Run the gravity
  stage twice, with a mass of 100 to 200 g (weighed to 0.1 g) taped to the mount and without it:
  `tipcal_run.sh gainA --stages preflight,homing,gravity`, then `gainB`. The change of the fitted weight against
  m g gives the gain to about 1 %.

- R6 (done by the lab in c26c65f9, 19:52; it had been asked at 19:44). Fit the HEX-E's drift during the sweep. With the
  arm still, `lab/sensor_watch.py` measured the untared Fz creeping 0.4 N in 5 min, 0.08 N/min. That drift was added
  along the sensor's z to the gravity records of the simulated session 151114-sim, and the plain fit was biased:

  | Sweep | Residual, plain fit (mN) | Map A off, plain fit | Weight off, plain fit | Residual, with drift term (mN) | Map A off, with drift term |
  |---|---|---|---|---|---|
  | Full speed, 94 s | 19.1 | 1.3 % | 0.6 % | 4.0 | 0.3 % |
  | 30 % speed, 243 s | 50.3 | 3.3 % | 1.6 % | 4.0 | 0.3 % |

  A is the HEX-E's map from the flange frame, and every force of the analysis goes through it. The drift term is one
  more column in the fit: the force reads A g_E + o_f + v (t - t_mean), with t each record's time (`rec["t"]`, already
  stored) and v the drift per axis, reported in N/min. The torques take the same column. With the term, the residual
  is again a test for cables pulling on the tool, and Table 2's 0.02 N applies to it. The final sweep has 4
  orientations in this plan and gets no fit, because the program needs 6. Its changes between orientations are
  compared with the first fit's prediction on the workstation instead.

- R7 (withdrawn 20:28; kept as an option, not recommended). Turn the flank and edge touch-offs with
  `--touch-twist 30`. Over five simulator truths on the robot's branch, the option helped in two and hurt in two (the
  second table below). The twisted touches stand at other arm configurations. Their kinematic error, about 0.1 mm in
  the simulator, is seen at half the sensitivity and biases the fit. A real CB3 is less accurate than the simulator's.
  The request as written at 20:19: Every touch-off of the canonical plan and of the lab's plan was at twist 0, where the finger lies
  along x_I. Sliding the sphere along x_I then changes no gap on the face or the flanks, so the sphere's place along
  the finger came from the homing's prior alone (0.25 mm). A flank's or an edge's touch at twist t sees that shift by
  its normal's y_P part times sin t. The option tries +-t, then +-2t, then 0. Contact error of the presses against the
  simulator's truth, with the same truth and seed in each pair:

  | Branch and stages | Touch twists | Contact error median (mm) | p95 (mm) | Max (mm) |
  |---|---|---|---|---|
  | Builder's own branch, full session | all 0 | 0.287 | 0.323 | 0.396 |
  | Builder's own branch, through pivot | 20 of 20 twisted | 0.052 | 0.087 | 0.158 |
  | Robot's branch (--q-seed), tonight's stages | all 0 | 0.089 | 0.131 | 0.155 |
  | Robot's branch (--q-seed), tonight's stages | 8 of 20 twisted | 0.077 | 0.098 | 0.108 |

  On the robot's branch, the wrist limit of 93e9c7db leaves 8 of the 20 a twist with every travel path clear. Trying
  60 deg first, or allowing 90, put travel paths into the wrist's limit. Tonight's homing is a single hand seating,
  and the error along the finger depends on its prior.

  Five truths (20:28), each drawn with the mount's and the pad's errors, the indenter's placement, a homing spread of
  0.25 mm along the key, and the seed. Contact error of the presses (median, mm):

  | Truth | All touches at twist 0 | --touch-twist 30 | Error at twist 0, mean vector in P (mm) |
  |---|---|---|---|
  | k1 | 0.299 | 0.114 | (-0.01, 0.13, 0.25) |
  | k3 | 0.324 | 0.020 | (-0.01, 0.16, 0.26) |
  | k4 | 0.377 | 0.526 | (-0.02, 0.13, 0.33) |
  | k5 | 0.473 | 0.605 | (-0.07, -0.10, 0.43) |

  k2 drew a 2 deg turn of the mount, and its touches missed the pad in both runs. The recommended build for tonight is
  the lab's options with `--q-seed=59.48,-75.93,120.3,-44.61,62.11,1.83 --order joint6 --gravity-roll-offset 30`.

- R9 (open, 20:30; the analysis and the owner). With every touch at twist 0, the contact point along the finger (z_P)
  rests on the homing's seating. With a homing spread of 0.25 mm along the key, the four truths put the presses' contact
  point 0.25 to 0.43 mm off along z_P, the same at every press. That offset does not show in the touch residuals (12
  um rms), and with the pivot out tonight nothing in the run measures it. The analysis will estimate it from the board,
  as the mean of the estimator's location error along z_P over the grid, against the estimator's own bias of about
  0.1 mm. On the robot, three freedrive seatings (the runbook's homing) in place of the one recorded seating shrink the
  random part of the seating error. A seating pushed the same way each time keeps its bias.

  Tested at 20:36 on truth k4 through the grid, with tonight's plan and no twists. The registration's offset against
  the simulator's truth is (-0.025, 0.131, 0.323) mm in P. At the 82 holds of 3 N and over, the estimator's mean
  location error against the registration's ground truth is (-0.069, 0.070, 0.306) mm, with standard errors of 0.010,
  0.014 and 0.011 mm. So the board recovers the offset along z_P to 0.017 mm. Its own bias there is -0.019 mm; at 1 N
  it is -0.71 mm. In the simulator the board follows the estimator's own model, so the real bias will be larger. The
  analysis's summary reports it as `board_minus_registration_mm`, a check that leaves the ground truth free of the
  board.

- R8 (open, 20:23; the owner and the lab, before the grid). The gravity sweep of `first2` (19:54, before the e-stop)
  does not give a map of the HEX-E that can be trusted. With the drift term, its residual is 65 mN (Table 2: under
  20). The gains of the three axes are 1.135, 1.085 and 0.78 of their mean. The sensor's axes come out tilted 2.4 deg
  from the tool axis, and the centre of mass 6.7 mm off it. The residual is largest in Fz: +162 mN at grav-04, after
  the 285 deg turn of joint 6, -202 mN at grav-09, after a 178 deg turn, and +160 mN at grav-10. A 1.5 s mean of Fz
  has about 20 mN of noise. With equal gains forced, the residual is 144 mN. The pattern fits a cable twisted by
  joint 6; a low gain of the Fz channel, whose raw zero is near 645 N, would also give it. Before the grid:
  1. route both cables with a service loop tied to the forearm, so that neither drags nor winds (R4), and sweep
     gravity again on the plan of R7;
  2. run R5 as two gravity runs back to back, from the same start pose with the same plan, the second with a weighed
     100 to 200 g mass taped to the mount. The cable forces repeat with the path and cancel in the difference, which
     gives the HEX-E's 3 x 3 map with the Fz gain to about 1 %.

  The grid's data stay usable if the residual remains over 30 mN: each press has its own rests, and the map can be
  refitted afterwards from R5.

The Writing section of `~/.claude/CLAUDE.md` and `~/.claude/writing_reference.md` went to the lab session in three
messages at 19:45; the lab confirmed them at 19:58.
