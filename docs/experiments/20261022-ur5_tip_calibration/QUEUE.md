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
  10 minutes of the push. First answered by `lab/20261022-192715-first/` (ffc6ed71). `first2`, `sphere1` and `tcp1`
  followed; R10 and R11 answer the last two.

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

- R4 (reopened 21:00 as R10: the tightened connector failed again at 20:27 and at 20:46; at 19:52 the owner had found
  the connector loose and tightened its coupling ring). Test the HEX-E's cable on the robot. The dropouts began at joint 6
  = 69.0 deg turning at +5.1 deg/s and at 23.5 deg turning at -6.9 deg/s. Both angles lie inside the span of the
  presses, and both came mid-turn. That fits a connection that opens when the cable moves, more than a cable drawn
  tight at one end of the span. Reseat both connectors of the cable and relieve its strain at the sensor. Then, with
  the program stopped, turn joint 6 from -180 to +170 deg and back three times at 10 deg/s, 150 mm above the
  indenter, while one reader logs the stream. It passes with no gap over 100 ms and no counter restart.

- R5 (changed 21:00: no sweep tonight, a test at rest on a later day, see R11; asked 19:38). Check the HEX-E's force
  gain with a known mass. Its
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

  Amended 21:00: the lab's segment plan of 20:50 has the six pivots again. Their check measures the sphere's place
  along the finger with no use of the board, so the offset along z_P then has two estimates.

- R8 (closed 21:00: the owner left the sweep out of tonight's runs, R11; the cables went into R10; asked 20:23). The gravity sweep of `first2` (19:54, before the e-stop)
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

- R10 (open, 21:00; the owner, before the next run). The HEX-E's link to its box failed in every run tonight, once with
  the arm still. No plan or program change avoids it. The losses, from the streams and `lab/NOTES.md`:

  | Time | Run | The HEX-E's stream | The arm |
  |---|---|---|---|
  | 19:27:40 | first | 7.2 s gap; counter restarted, 13702 to 102 | joint 6 at 69 deg, turning 6.5 deg/s |
  | 19:28:24 | first | stopped for about 2 min | joint 6 at 23.5 deg, turning -7 deg/s |
  | 19:43:27 | none | no sensor in the box for 4 min | still; the owner touched the connector |
  | 19:58:45 | first2 | 0.42 s with no answer, counter running on | joints 5 and 6 turning |
  | 19:59:25 | first2 | no sensor in the box | moved by hand after the e-stop |
  | 20:27:40 | sphere1 | counter restarted, 18754 to 2; 14 samples, then none | joint 4 turning 6.7 deg/s, joint 6 2.8 deg/s |
  | 20:46:58 | tcp1 | no sensor in the box | still since 20:28:20 |

  At 21:21 the lab found that sphere2's short gaps, with the counter running on, came from the program's own fsync of
  `ft.bin` on a slow disk; d1ef3216 moves the fsync to a thread of its own. The 0.42 s gap at 19:58:45 has the same
  signature. The other six losses are the sensor's.

  After each restart the first samples read Fz 1863 N, falling to 655 N within 0.1 s, as the sensor starts up. So the
  sensor lost its power or its link, at the connector or in the cable. Tightening the coupling ring and tying the cable
  down at 19:48 did not stop it. At 19:59 the wrist pressed the cable port against the forearm, which may have damaged
  the port or the cable at its strain relief. Before the next run:
  1. Look at the connector and the port on the sensor: bent or pushed-back pins, a cracked housing, the ring threaded
     fully home. Look along the cable for a kink, mostly near the port.
  2. Fit a spare cable if there is one.
  3. Clamp the cable to the robot's side of the sensor (the flange or the UR adapter), 3 to 5 cm from the connector,
     with that length slack. The wrist's turns then pull on the clamp.
  4. Flex the cable with the arm still while `lab/sensor_watch.py` logs the stream: 10 s at the connector, 10 s at its
     middle and 10 s at the box. It passes with no gap over 100 ms and no counter restart. The place where it fails is
     the part to replace.

  The program's gap guard (20bfc6e7) keeps the data safe: an action with a gap aborts, and `--resume` continues the
  session. A link that fails every few minutes still ends each run within its first stage.

- R11 (open, 21:00; the lab, the next runs). Leave the gravity stage out until the cables are routed. The five stations
  that sphere1 finished before the restart test first2's fit. With only the offset free, first2's map predicts them to
  63 mN rms, and first2's own sweep fitted to 65 mN. Per axis the misfit is 37, 36 and 96 mN, largest in Fz as in
  first2. So first2's map predicts a later session as closely as either sweep fits its own stations, and another sweep
  with the same cables would not improve it. The lab's segment plan of 20:50 has no gravity stage already.

  In a session without the stage, the program sets the press levels with the HEX-E's factory axes and gains (K the
  identity). The analysis takes first2's fit for such a session:
  `.venv/bin/python scripts/ur5_tip_calibration_analysis.py <session> --gravity-from
  docs/experiments/20261022-ur5_tip_calibration/lab/20261022-195420-first2`. It was tested on a simulated session with
  its own fit removed and borrowed back: the 32 arrays of `holds.npz` came out identical.

  The force gain stays open (R5). Both fits give the Fz channel a lower gain than Fx and Fy: 0.78 of their mean in
  first2 and 0.87 over sphere1's five stations. Cables pulling on the tool would give that, and so would a low Fz gain.
  Through first2's map, the analysis reads every force along the sensor's z 28 % larger than the factory gains do. The
  runs tonight do not depend on it, since each press has its own rests and the analysis can apply another map
  afterwards. The test needs no sweep. At the home pose gravity lies along the sensor's -y, and with the tool pointing
  down along its z. At each pose, read 10 s, hang a weighed 100 to 200 g mass from the mount, read 10 s, take it off
  and read 10 s. The change against m g gives that axis's gain to about 1 %.

- R12 (done 21:30 by the workstation, asked by the lab at 21:09; simulated, see the end of this item). The
  builder now checks every 0.5 deg step of each slide, roll and twist as `run_turns` moves them, with all parts against
  the cell (`turn_poses`). In tcp2, twist-0-1.0N's turn put the mount's fork on the indenter's homing key. On tcp2's own
  poses and homing, the cell model puts the fork 5.6 mm from the key at the pressed pose, 3.0 mm at -3.3 deg and 0 at
  -7.1 deg; on the robot the load began at -8.6 deg. Turning the other way moves the fork off (10.7 mm at +10 deg).
  Built with the lab's options, the old twists of +-20 deg all drop at -3 to -4 deg. Each twist now turns one way, out
  and back: 5, 10, 15, 10 and 5 deg, or the mirror of that when the first way does not clear. The friction's torsion
  changes sign with the direction of turning, and the push's moment follows the pose. Half the difference between the
  way out and the way back at 5 and 10 deg is then the torsion. At 15 deg joint 5 stays at 36.6 deg or more. At 20 deg it
  came to 31.2 deg on the robot, 1.2 deg inside the window; the builder's IK gave 31.0.

  The plan is `plan/20261022-plan_sphere_lab_turns_checked.json`, built with the options of `lab/README.md` plus
  `--q-seed=59.48,-75.93,120.3,-44.61,62.11,1.83 --order joint6 --gravity-roll-offset 30`. It has 143 actions, and
  every action outside the turn stages is the same as in the lab's `20261022-plan_sphere_lab_window_untwisted.json`,
  with the same twist and order. Of the turn stages, 4 slides of 7, 2 rolls of 4 and all 8 twists clear. shear-1, -2
  and -4 and roll-1 and -2 come to 2.87 to 2.99 mm of the 3 mm that the mount needs.

  Simulated at 21:18 (`211803-sim-turns`: preflight, homing, shear, roll and twist): all 14 turn actions ended ok.
  `lab/check_arm_samples.py --cell --every 25` finds no sample outside homing that breaks a rule. The least clearances
  of the mount and the finger are 3.0 and 3.0 mm in the slides, 4.7 and 4.7 in the rolls, and 3.7 and 4.0 in the twists.
  Joint 5 came to 30.9 deg in shear-5-1N, against the program's limit of 30. The builder now keeps joint 5 1 deg inside
  the program's window (`WRIST_MARGIN`). That drops shear-5-1N alone, and the plan has 142 actions; every other action
  is unchanged, so the simulated run covers them.

- R13 (21:20; the workstation's analysis after sphere2's register). tcp2's three presses met the sphere 0.72, 0.83 and
  0.76 mm before the planned contact: ref-00 on the face, grid-052 and grid-053 on the flank. Each onset is the start of
  the push along the approach, fitted as for the touch-offs. Read as a shift of the pad, the three give (0.72, 0.10,
  1.07) mm in P: 0.7 mm along the face normal and 1.1 mm along the finger. The two flank points see the finger's
  direction through their normals, 0.44 and 0.30 along z_P. A uniform offset of the pad's surface by 0.77 mm fits the
  three as well. Every touch and press approaches the sphere's top vertically, so a uniform offset changes every gap
  alike, as a higher sphere would. The registration then takes it as the sphere's centre above the homing's, against
  that prior's 0.12 mm. A shift of the pad shows as a change of the TCP instead. sphere2's first 12 onsets (lab, 21:21)
  are 0.64 to 1.19 mm early, close to uniform.

  Tested at 21:35 on the 32 touch-offs of the simulated session 202109-sim-real-t0, with every onset pose raised 0.9 mm
  along z_I and the registration fitted again. The sphere's centre rose 0.92 mm and moved 0.41 mm across the finger.
  The pad frame moved 0.05 mm, the TCP's turn changed 0.4 deg, and the touch residual went from 10 to 12 um rms. At
  the touches, the sphere's centre in P moved 0.12 mm across the finger on average, and at most 0.17 mm on any axis. A
  uniform early contact therefore leaves the contact point within about 0.15 mm.

  The twist's turn added 0.29, 0.47 and 0.63 N along the finger at 5, 10 and 20 deg, after the HEX-E's creep of
  0.08 N/min is removed. Its axis therefore passed off the contact across the finger, by 0.25 mm at the FE's stuck-contact
  stiffness (13 to 15 N/mm at 1.3 N) and by about 1 mm at the stiffness of the presses (about 3 N/mm at 1.2 N). The flank
  touches measure that direction. The 1.1 N in Fy that the lab read at the first step is the twist's own press to 1 N:
  the mount's -92 deg turn puts the pad's normal along the sensor's y.

The Writing section of `~/.claude/CLAUDE.md` and `~/.claude/writing_reference.md` went to the lab session in three
messages at 19:45; the lab confirmed them at 19:58.
