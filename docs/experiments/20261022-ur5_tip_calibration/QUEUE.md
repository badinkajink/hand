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

- R10 (answered by the owner just after midnight on 10-23: the HEX-E cannot be fixed now, and a new cable may be
  ordered but will take a while, so runs keep the intermittent link and its guards; opened 21:00). The HEX-E's link to
  its box failed in every run tonight, once with
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

  Added 21:54: in sphere2r the box stopped answering for 1.4, 2.8 and 1.7 s, with 118 timed-out reads and no counter
  restart, and pad-flank-09, -08 and -04 aborted (lab, 21:52). The reader was polling, so the pause was the link's.

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

- R12 (done 21:24 by the workstation, asked by the lab at 21:09; simulated, see the end of this item). The
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

- R13 (answered 21:48 by sphere2r's registration, see the end of this item; asked 21:19). tcp2's three presses met the sphere 0.72, 0.83 and
  0.76 mm before the planned contact: ref-00 on the face, grid-052 and grid-053 on the flank. Each onset is the start of
  the push along the approach, fitted as for the touch-offs. Read as a shift of the pad, the three give (0.72, 0.10,
  1.07) mm in P: 0.7 mm along the face normal and 1.1 mm along the finger. The two flank points see the finger's
  direction through their normals, 0.44 and 0.30 along z_P. A uniform offset of the pad's surface by 0.77 mm fits the
  three as well. Every touch and press approaches the sphere's top vertically, so a uniform offset changes every gap
  alike, as a higher sphere would. The registration then takes it as the sphere's centre above the homing's, against
  that prior's 0.12 mm. A shift of the pad shows as a change of the TCP instead. sphere2's first 12 onsets (lab, 21:21)
  are 0.64 to 1.19 mm early, close to uniform.

  Tested at 21:25 on the 32 touch-offs of the simulated session 202109-sim-real-t0, with every onset pose raised 0.9 mm
  along z_I and the registration fitted again. The sphere's centre rose 0.92 mm and moved 0.41 mm across the finger.
  The pad frame moved 0.05 mm, the TCP's turn changed 0.4 deg, and the touch residual went from 10 to 12 um rms. At
  the touches, the sphere's centre in P moved 0.12 mm across the finger on average, and at most 0.17 mm on any axis. A
  uniform early contact therefore leaves the contact point within about 0.15 mm.

  The real touches are not uniform, so the guess above does not hold. sphere2's onsets run from 1.19 mm early to 0.89 mm
  late, and sphere2r's registration (21:44, 29 touches) fits them at 57 um rms (Table 2: under 50). Its residuals show no
  trend over the 36 minutes of touches; the largest are about 0.1 mm, on flank-05, edge-08, edge-09, flank-03 and
  flank-01. The TCP moved 2.01 mm and 2.81 deg, and the sphere's centre 1.22 mm. Several parameters lie 3 to 5 sigma past
  their priors:

  | Parameter | Fitted | Posterior 1-sigma | Prior 1-sigma |
  |---|---|---|---|
  | Pad in its seat, about x_M (deg) | -1.81 | 0.33 | 0.6 |
  | Pad in its seat, about y_M (deg) | -1.98 | 0.39 | 0.6 |
  | Mount on the flange, along z (mm) | +0.94 | 0.26 | 0.6 |
  | Sphere's centre along x_I, the finger (mm) | +0.98 | 0.33 | 0.25 |
  | Sphere's centre along z_I (mm) | -0.64 | 0.03 | 0.12 |

  The contact points of the grid therefore carry about 0.3 mm of uncertainty along the finger (R9). The six pivots and
  the board's location error (`board_minus_registration_mm`) check that direction without the touches.

- R14 (done 21:48 by the lab and the workstation). `scripts/ur5_tip_calibration_check.py --clearance [PLAN]` (753ccc2d)
  gives the plan's least clearance of the mount, the finger and the HEX-E with the session's registration, against the
  nominal geometry that the builder checks. Once registered, the program puts the real pad on its targets, and the mount
  stands where the real geometry puts it. With sphere2r's registration the finger comes to 0.57 mm at grid-050 (needed
  0.8) and the mount to 2.27 mm at shear-0-1N (needed 3.0). The lab dropped the actions more than 0.3 mm under a margin
  and every action under the finger's: grid-050, shear-0-1N, shear-0-2N and shear-3-1N. That leaves
  `lab/20261022-plan_sphere_lab_turns_checked_reg2144.json` with 138 actions. grid-025, -027 and -028 stay at 2.75 to
  2.98 mm of the mount's 3 mm. On tcp2's real poses the model was about 1.2 mm conservative, and the 10 N abort stops a
  press that touches.

- R15 (open, 21:54; the lab, before the next pivot stage). All six pivots of sphere2r held 1.5 and 3.0 N, and then
  `stage_pivot` stopped on a TypeError: `board_level` returned None for the reference press. The cause is the 180 s
  ring buffer (`Ring(keep_s=180)`). The stage reads every press's board samples only after the sixth press, and at 30 %
  speed pivot-0's samples have left the buffer by then. A check for None would skip the pivot check on every real run.
  `board_level` should run inside each press's action, while its samples are in the buffer, or read `board.bin` and
  `ft.bin`. The analysis's `pivot_check` (8e9c6f59) does the same computation from the files. On the simulated session
  195934-sim-joint6r30 it gives pivot-5 (0.013, 0.065) mm against the program's (0.009, 0.066).

  The twist's turn added 0.29, 0.47 and 0.63 N along the finger at 5, 10 and 20 deg, after the HEX-E's creep of
  0.08 N/min is removed. Its axis therefore passed off the contact across the finger, by 0.25 mm at the FE's stuck-contact
  stiffness (13 to 15 N/mm at 1.3 N) and by about 1 mm at the stiffness of the presses (about 3 N/mm at 1.2 N). The flank
  touches measure that direction. The 1.1 N in Fy that the lab read at the first step is the twist's own press to 1 N:
  the mount's -92 deg turn puts the pad's normal along the sensor's y.

- R16 (confirmed by the owner's hand check just after midnight on 10-23: the DIP link moves a little in the fork
  when the pad is pressed hard; the fix is R18; opened 22:05). The first analysis of sphere2 (pad, pivots and 12 grid presses as of
  22:02; `data/20261022-210852-sphere2/`, `--gravity-from` first2) finds the chain from the board to the HEX-E softer than
  the model. At the sphere's positions from the registration, the measured normal force is 0.25 of the FE pad's at every
  level from 0.5 to 5 N and in every region. The compliance scale that fits would be about 4, outside the fit's grid
  of 0.6 to 1.8. tcp2's presses, timed from their own onsets, give the same without the registration: 0.83 mm of
  travel at 2 N on the face, where the FE pad indents about 0.27 mm. The board's force, read through the FE model,
  matches the HEX-E: its slope against the measured force is 1.11 on the face, 0.92 on the edges and 1.00 on the flanks.
  The pad itself therefore deforms about as the FE says, and its magnets do not move against the chips by the extra
  0.5 mm. The extra travel lies between the board and the HEX-E: the DIP link's clamp in the fork, the printed mount
  with its inserts, or the arm. The ground truth takes that chain as rigid apart from the HEX-E, so the depth of a press
  and the place of its contact carry the error. Check by hand whether the DIP link moves in the fork when the pad is
  pressed, and with a dial indicator on the link if one is at hand. The grid's reference presses give the trend
  through the night.

  In the same analysis, the board's estimator places contacts within 1.2 to 1.6 mm (median) from 1 N up, and at 9.6 mm
  under 1 N (0.05 mm from 2 N in the simulator). Its force is within 10 to 20 % of the HEX-E's (median) from 1 N up. At
  3 N and over, its mean location error against the registration is (0.10, -0.31, 0.42) mm in P, with standard errors
  of 0.05, 0.16 and 0.26 mm. The offline pivot check (`pivot_check`) places the four offset presses about two centres
  0.42 mm apart, and the press turned 15 deg 2.0 mm off with a fit residual of 12 uT, against 3 to 4.5 uT for the offsets.
  The board's linear map from four offsets does not carry over to a turned press, so that turn does not test the
  registration.

  Update 22:51, same data (`scripts/ur5_tip_calibration_fe.py` of 027f7c6a). With the scale's grid widened to 8, the
  best scale is 4.0, with a median force error of 12 % (302 % at the FE's own compliance). Kept as computed, the FE pad
  gives the measured force once the sphere moves back along the normal by 0.21 mm + 0.19 mm/N times the force (rms
  108 um over 69 holds). The slope is 0.22 mm/N on the face, 0.16 on the edges and 0.09 on the flanks. Every press
  pushes the sphere straight down, so the 2x4, its clamps and the table would give one slope in every region. Slopes
  that change with the load's direction in the tool's frame point to the tool's side. The 0.21 mm at zero force fits
  play that closes at the first touch. A compliance at the printed surface that the 1 mm FE mesh does not resolve
  would also leave the magnets still. It would be largest where the contact is smallest, on the edges and flanks, and
  the face has the largest slope. Hand check, in this order: the DIP link in the fork (palmar and sideways), the mount
  on the HEX-E, the sphere's block on the 2x4. Both fits are in Fig. 1 of
  `20261022-ur5_tip_ground_truth_and_predictors.html` (b34784f6), which is rebuilt on the whole run.

  Update 23:38, the whole session (bf3aa417; 373 press holds over 0.2 N, d42db5bf). The best scale is 4.75 (median
  force error 20 %; 412 % at the FE's own compliance), and the measured force is 0.19 to 0.21 of the FE pad's by region.
  The indentation beyond the FE pad is 0.30 mm + 0.21 mm/N (rms 170 um). By region the slope is 0.212 +- 0.006 mm/N on
  the face, 0.168 +- 0.014 on the edges and 0.23 +- 0.05 on the flanks. The 22:02 slopes (0.22, 0.16, 0.09) came from
  8 flank holds; with 64 the flanks' slope matches the face's, and the conclusion of 22:51 that the regions point to
  the tool's side is withdrawn. The FE model's own error may differ by region too. The arm's joints are in the chain
  as well: a deflection under load that the encoders do not report adds to the depth. The board's force is now 0.83,
  0.78 and 0.96 of the HEX-E's on face, edges and flanks, so the pad is not 4.75 times softer. The hand check stays as
  above; a dial indicator on the sphere's block while a rigid part of the tool presses it would split the indenter's
  side from the tool's. At four flank holds of 0.25 N (grid-061, -065, -066, -067) the registered sphere stands 0.47 to
  0.78 mm clear of the pad while the HEX-E measured contact. At 3 N and over, the estimator's mean location error is
  (-0.33, -1.21, 0.72) mm in P, standard errors 0.07, 0.16 and 0.14 mm over 85 holds: a systematic offset of 1.4 mm,
  of the registered points or of the model's magnets. Between the rests around a press the board's rest field changes
  by 7.1 uT (median); a 1 N reference press changes it by 16 to 24 uT. Below 1 N the two are of the same size, and the
  estimator fails there (9.7 to 9.9 mm).

  Update 23:44. The lab's message stamped 23:50 (the lab's clock runs ahead of this one) traces those four flank
  presses to 0.3 to 0.4 N on the HEX-E before the pad meets the sphere. The load rises with travel and does not show
  on the board; joint 6 stands at 70 to 72 deg, so the cable or a graze. The registration there is not in question.
  The four presses are left out of the ground truth since the rerun of 23:40 (`EXCLUDE` in
  `scripts/ur5_tip_calibration_analysis.py`, recorded in summary.json). With 357 press holds the best scale is still
  4.75 (median force error 19 %), and the indentation beyond the FE pad is 0.33 mm + 0.20 mm/N (rms 133 um, from
  170). By region the slope is 0.212 +- 0.006 mm/N on the face, 0.168 +- 0.014 on the edges and 0.141 +- 0.027 on the
  flanks. The flanks' 0.23 of 23:38 came from the four presses, so the withdrawal at 23:38 rested on them. The slopes
  again fall from the face to the flanks, as at 22:02. A compliance on the tool's side fits that, and so does a regional
  error of the FE model. The board's force is 0.83, 0.78 and 1.05 of the HEX-E's on face, edges and flanks. The
  estimator's numbers above do not change: the four presses stopped at 2 N.

  Update 01:06 on 10-23. The lab's drift study (`20261023-ur5_tip_board_drift.html`) finds that each press leaves a
  6.7 uT step along its own change, so the two-rest zero takes part of the press out. `--board-zero before` in the
  analysis zeroes each hold on the rest before alone (`data/20261022-210852-sphere2-zbefore/`). The board's force then
  reads 0.87, 0.80 and 1.15 of the HEX-E's on face, edges and flanks (two-rest zero 0.83, 0.78, 1.05). The estimator's
  median location error moves by up to 0.5 mm either way by force level, and the 1.4 mm offset at 3 N and over stays.
  The GP gives 0.82 mm on held-out presses (0.84) and 1.98 mm on turns (2.05), with 0.235 N on turns (0.189). The
  default stays the two-rest zero; the option is kept.

- R17 (answered 23:38 and 23:44 on the whole session; begun 22:21 on the workstation; the owner asked at 22:10 through the lab). Predictors of the contact point
  and the normal force from the board, trained on a session's presses in time order:
  `scripts/ur5_tip_calibration_predict.py <analysis output folder>`, outputs in `data/<session>/predict/predict.json`.
  Every fourth grid press is held out. The predictors are the viewer's estimator with its compliance scale fitted (the
  magnets' strength and the chips' offsets held near the rest's fit), a Gaussian process from the inputs (ridge as the
  linear baseline), and the two together. The inputs go in one at a time: the board, the chips' temperature change
  since the press's rest, then the time since that rest. The tool's orientation is tried apart. In this protocol each
  target has a planned orientation, so it identifies the target: on the simulated session it took the GP to 0.24 mm.
  Medians on the held-out presses:

  | Session, predictor | Location (mm) | Normal force (N) |
  |---|---|---|
  | Simulated 195934, viewer's estimator | 0.38 | 0.324 |
  | Simulated 195934, physical model fitted on 3 presses | 0.44 | 0.084 |
  | Simulated 195934, GP from the board on 3 / 65 presses | 4.55 / 0.45 | 1.29 / 0.022 |
  | Simulated 195934, physical model plus a GP on 65 presses | 0.33 | 0.058 |
  | sphere2 as of 22:02, viewer's estimator | 1.97 | 0.376 |
  | sphere2 as of 22:02, physical model fitted on 16 presses | 3.58 | 0.325 |
  | sphere2 as of 22:02, GP from the board on 15 presses | 1.05 | 0.139 |

  sphere2 has two held-out presses so far (10 holds). Its physical fit wants a compliance scale of 0.31 to 0.54 (1.20 in
  the simulator, whose truth is 1.15). At the registered contact points the board changes by a third to a half of the
  model's prediction, while the free estimator matches the force within 10 %. Either the registered points are off by
  about 2 mm, or the model's response over the pad differs from the real pad's; the full grid separates the two. The
  study runs again on the whole folder after the run, and on the turn holds once the turns have run.

  Update 23:44, the whole session without the four presses of R16 (59 training presses, 15 held out with 80 holds,
  and 78 turn holds of the two rolls and eight twists; every press trains the models scored on turns). Medians:

  | sphere2 whole session, predictor | Held-out presses: location (mm) | force (N) | Turns: location (mm) | force (N) |
  |---|---|---|---|---|
  | Viewer's estimator, untrained | 3.79 | 0.382 | 2.44 | 0.430 |
  | Physical model, compliance fitted (c 0.56) | 3.05 | 0.314 | 2.52 | 0.289 |
  | GP, board | 0.84 | 0.086 | 2.05 | 0.189 |
  | GP, board and temperature change | 1.25 | 0.116 | 2.99 | 0.151 |
  | GP, board, temperature change and time since rest | 1.21 | 0.082 | 2.28 | 0.428 |
  | GP, the above and orientation (leaks the target) | 0.35 | 0.064 | 2.12 | 1.385 |
  | Ridge, board | 2.38 | 0.092 | 7.58 | 0.474 |
  | Physical model and residual GP, board | 2.94 | 0.170 | 2.32 | 0.248 |

  The board alone is the best input on this night: the temperature change and the time since the rest raise the GP's
  location error from 0.84 to 1.25 and 1.21 mm. Below 1 N the GP places held-out contacts within 2.5 mm at 0.25 N and
  1.1 mm at 0.5 N, where the untrained estimator is 9.9 mm off. Its error is still falling at 59 presses. On the turns
  it carries over only in part: 2.05 mm against 2.44 mm for the untrained estimator (rolls 1.92 against 5.36 mm, twists
  2.19 against 2.34 mm). The physical model's compliance scale stays at 0.56, below the board's force scale of 0.78 to
  1.05. In its fit the contact points are the registered ones, a median of 2.0 mm from the estimator's from 1 N up (see
  R16's offset). Page: `20261022-ur5_tip_ground_truth_and_predictors.html`, Section 6.

- R18 (moved out of this channel at 10:51 on 10-23 at the owner's request; nothing in it is for the lab session. It continues
  in its own conversation from `docs/handoff/20261023-ur5_mount_v4.md`. When the owner puts v4 on the robot, a new request
  here gives the lab the steps at the end of this item. Designed on the workstation at 00:38, not printed; the owner asked
  through the lab at 00:16). A mount that fastens the
  DIP link rigidly, since the link moves in mount v3's fork (R16). It has to beat the extra indentation of 0.33 mm +
  0.20 mm/N (0.21, 0.17 and 0.14 mm/N on face, edges and flanks). It keeps the HEX-E stack and mount v3's captive
  M6 x 20, the mount's turn of -92.15 deg on the flange (or a plan rebuilt for a new turn), at least 15 mm from the
  HEX-E's cable port to the forearm and wrist in the plan's poses (`ur5_cb3_cell.self_clearance`), 3 mm from the mount
  to the cell and the sphere's homing key along every checked turn, and the pad's seat as it is, so tonight's
  registration and FE model still apply. Proof on the bench: a dial indicator on the link while a rigid part presses
  the pad, with v3 and with the new mount; then the reference press's depth to 2 N (0.71 to 0.76 mm tonight) should
  fall toward the FE pad's indentation, about 0.27 mm.

  Mount v4 (0c2ea786; page `docs/experiments/20261023-ur5_calibration_mount_v4/20261023-ur5_calibration_mount_v4.html`).
  In v3 the legs alone hold the link: with the clamp loose it turns 1.0 deg about the PIP axis before it meets the
  fork's walls, 0.63 mm at the contact point, and a face press drives that turn. v4 is v3 with a stop plate hanging
  from the dorsal ear behind the tongue. Three brass M3 set screws in heat-set inserts are turned onto the tongue's back
  (x_F 8.0) once the link is clamped, then locked with jam nuts. The slotted arm's M3 goes into an insert. With the
  screws set, the turn a face press drives is 0.05 deg (the check's step); the turn about x, which flank presses
  drive, stays 0.3 deg. The link slides on clear. Along plan reg2144 with sphere2's registration the plate stays
  12.0 mm from the cell; the mount's least clearance is v3's (2.75 to 2.84 mm at grid-027). Only the standoff-fork is
  reprinted (P1S, 1 h 46 min, 64.9 g, six walls). The owner decides on printing it. When v4 goes on the robot:
  - payload 0.403 kg in place of 0.384 (the fork +14 g, screws, nuts and inserts +5 g), its centre about 2 mm further
    along the tool axis;
  - pad touch-offs and registration again (the link may sit up to 1 deg differently);
  - plans checked with `UR5_MOUNT=v4` for `scripts/ur5_cb3_cell.py` (the builder and `ur5_tip_calibration_check.py
    --clearance`), which needs `docs/experiments/20261023-ur5_calibration_mount_v4/cad/*.stl` from LFS on the lab
    computer;
  - then the reference press and a short grid, to refit the indentation beyond the FE pad.

- R19 (ready on the workstation at 08:10 on 10-23; the owner asked at about 07:50: "did we also try cylinder presses? can
  we do that if we are feeling good about it?"). Every press so far was on the 5 mm sphere; the rod (12.5 mm half
  cylinder, the screwdriver's radius) stood in every plan as an obstacle only. A rod session presses the pad's face and
  edges against a line, which tests the pad model at the screwdriver's curvature. It also splits R16's extra travel
  (0.33 mm + 0.20 mm/N): a contact this wide leaves the FE mesh little local error, so travel that stays the same on the
  rod lies in series with the pad.
  - Plan `plan/20261023-plan_rod_lab_reg.json`, 99 actions: 22 touch-offs on the rod, 5 pivot presses, 53 grid presses
    (face 49 and edge 23 points reachable; no flank reaches the rod), 1 slide, 2 rolls, plus gravity stations that runs
    skip. Built with `--indenter rod --placement=-289.5,-585.5,70.3,-94.07 --block 38 --block-half 55,500
    --mount-turn=-92.15 --other=sphere,-138.5,-595.6,70.8,-92.3 --q-seed=42.91,-69.80,111.48,-40.17,47.33,1.83
    --order joint6 --gravity-roll-offset 30` (the seed is the rod's hand seating of 18:40, "right_indenter_seated" in
    `lab/poses.jsonl`). The builder now accepts the pad's line contact with the rod; before that it dropped every press.
    Six actions came under 3 mm of mount clearance with the tool as registered last night (2.44 to 2.98 mm) and are out:
    grid-005, -024, -025, -050 and shear-0-1N, -2N. With that tool the finger keeps at least 1.37 mm.
  - Simulator, the whole plan without gravity: 22 touch-offs at 0.015 mm rms, all 57 grid presses, 3 slides and 2 rolls
    ran (`logs/20261023-rodsim/`, 4 min). The analysis matches the simulator's contact to 0.095 mm and its normal force
    to 0.009 N; the estimator places contacts within 0.18 to 0.46 mm from 1 N up.
  - Run, on mount v3 as it stands, at 30 %: `--plan docs/experiments/20261022-ur5_tip_calibration/plan/20261023-plan_rod_lab_reg.json
    --home-from docs/experiments/20261022-ur5_tip_calibration/lab/poses.jsonl:right_indenter_seated --stages
    preflight,homing,pad,register,pivot,grid,shear,roll`, about 1.5 h. The analysis borrows first2's gravity fit as for
    sphere2. The FE comparison takes the rod since 08:21 (the analysis stores the rod's axis per hold): on the simulated
    rod session it fits the pad's compliance at 1.20 times the FE's, where the simulator's is 1.15.
  - 11:00, after rod1 (lab e4fcb876): push rod1's small files now (all but `ft.bin`, `robot.bin` and `board.bin`), so
    that its registration is checked here while rod1g runs. The program's `fit_registration` treats the rod as a
    sphere: each pad touch's residual is the pad's distance to one point, minus R, and homing fixes the rod's axis. All
    22 touch-offs press the same point on the axis. The pad's offset in its seat then moves each real contact along the
    rod, by up to about 2 mm. A cylinder allows that motion; a sphere penalizes it by about 0.16 mm at 2 mm, the size of
    the worst residuals (0.17 to 0.29 mm). The workstation refits with the distance to the axis line. The run goes on
    unchanged unless the refit turns the TCP by more than a degree; then the mount clearance of the remaining actions
    is recomputed here.
  - 11:05: answered by the lab's 292b84b1. The cylinder refit (`scripts/ur5_tip_calibration_rod_registration.py`) does
    not change the registration: 121.7 to 122.1 µm rms with the axis fixed, turning with the mount or free, the TCP
    0.08 to 0.12 mm and 0.19 to 0.36 deg from the program's. The sphere model was not the cause.
  - 11:16, after rod1g ends and before anything is moved: two touch-off sessions, both from the stored seatings. In 19
    of rod1's 22 touches the onset lies within 0.1 mm of the approach's start; the pad already pressed the rod at the
    tared rest pose, so the registration rests on the start of each motion. sphere2's 29 touches were clean (onsets
    0.31 to 2.39 mm into the approach). In the simulator, with sphere2's registered tool as the truth, the rod's
    touch-offs alone leave the pad 0.25 mm off the truth (median), and a joint fit with touch-offs on the sphere
    0.12 mm.
    1. `rod1t`: `plan/20261023-plan_rod_lab_touch35.json` (rod1g's plan, the 22 pad touch-offs at a 3.5 mm standoff),
       `--stages preflight,homing,pad,register --home-from docs/experiments/20261022-ur5_tip_calibration/lab/poses.jsonl:right_indenter_seated`.
    2. `sph3t`: `plan/20261023-plan_sphere_lab_touch35.json` (sphere2's first plan, its 32 pad touch-offs at 3.5 mm),
       `--stages preflight,homing,pad,register --home-from docs/experiments/20261022-ur5_tip_calibration/lab/poses.jsonl:left_indenter_seated`.
    Push each session's small files when it ends. A placement check that stops either one ends the request. rod1g's
    presses are analysed with the joint registration (`ur5_tip_calibration_analysis.py --registration-from`).
  - 11:27, the lab (72b41230, `lab/descent_check.py`): at rest the pad already pressed the rod with 1.85 N (median;
    3.80 N at most), a true onset 1.6 to 2.7 mm early, so both sessions take the 5.0 mm plans (2fd80414).
  - 11:31, how each registration places the contacts. Two whole sessions were simulated with sphere2's registered tool
    as the truth: the rod (3.5 mm touch-offs, pivot and grid) and the sphere (sphere2's first plan). Each registration
    then ran through the analysis, and the table gives the median distance from the analysis's contact to the
    simulator's, along the pad. "×4" means the mount and pad priors of `fit_registration` four times looser.

    | Registration | Rod session (mm) | Sphere session (mm) |
    |---|---|---|
    | The program's own | 0.603 | 0.565 |
    | Its own touch-offs, ×4 | 0.623 | 0.189 |
    | Joint fit of rod and sphere touch-offs | 0.302 | 0.258 |
    | Joint fit, ×4 | 0.091 | 0.167 |

    The depth along the normal stays within 0.013 to 0.024 mm in every row, because each press's own onset sets it.
    So R16's extra indentation stands, and the contact locations behind R17 may carry a bias of about 0.5 mm. rod1g is
    analysed with the joint fit ×4 of `rod1t` and `sph3t`. If `sph3t`'s own registration matches sphere2's (the same
    tool since last night), sphere2 is analysed again with that joint fit, and R17's predictors rerun.
  - 12:04, asked by the lab at 11:57: `ur5_tip_calibration_check.py --clearance` covers the touch-offs as a fresh
    session runs them (9a5ce959). The rest pose is planned with the nominal tool; the tool and the indenter sit where
    the registration puts them; the approach is stepped until the registered pad meets the indenter. `--home-from
    FILE:LABEL` places the indenter from a recorded seating with the session's registered mount.
  - 12:06, the lab's question: `sph3t` at 3.5 mm (`plan/20261023-plan_sphere_lab_touch35.json`, about 41 min at the
    30 % slider, against 56 min at 5.0 mm), if the check with rod1t's registration and `--home-from
    lab/poses.jsonl:left_indenter_seated` predicts every onset at −2.5 mm or later (1 mm of free approach);
    otherwise 5.0 mm. The touch-offs that the check lists under the margins are dropped.
  - 12:09, a second simulated truth changes the choice of 11:31. Its tool error is rod1's fit with the mount's turn
    set to 1.43 deg (the homing tilt measured in rod1), with other robot kinematic errors and no plywood tilt. Median
    contact-location error through the analysis, rod and sphere sessions:

    | Registration | Rod, truth 1 (mm) | Rod, truth 2 (mm) | Sphere, truth 1 (mm) | Sphere, truth 2 (mm) |
    |---|---|---|---|---|
    | The program's own | 0.603 | 0.277 | 0.565 | 0.161 |
    | Joint fit | 0.302 | 0.167 | 0.258 | 0.118 |
    | Joint fit, ×4 | 0.091 | 0.292 | 0.167 | 0.261 |

    The joint fit at the program's priors beats the program's own fit under both truths. The ×4 priors help only under
    truth 1, whose tool turns 2.8 deg. The pivot presses cannot pick between them: the board's offsets between pivots
    stay the same to 0.01 mm under every registration. So rod1g is analysed with the joint fit at the program's
    priors, and the ×4 fit is run beside it as the spread.
  - 12:39, after rod1t (lab 0bc6e0f7; my two peer messages before this were stamped 12:41 and 12:47, ahead of the
    clock). rod1t's 22 onsets are clean (−1.34 to −2.48 mm at a 5.0 mm standoff), but the program's fit is 0.201 mm
    rms, with the rod's centre 4.05 mm from homing. One touch-off causes it: pad-edge-01, +0.66 mm. Without it the
    cylinder fit is 40.4 µm rms over 21, the rod's centre moves 1.17 mm and the TCP 1.60 mm and 4.66 deg from
    nominal. That fit's TCP lies 2.67 mm and 3.68 deg from the program's. At edge-01's onset the pad is 1.38 mm from
    the rod and no modelled part is nearer than the fork's 1.40 mm. Its joint 6 stands at +45.5 deg, the most of the 22
    (next +31.7). A load through the HEX-E's cable, as on last night's flank presses at 70 to 72 deg, would explain
    it; rod1t's streams will show the force's direction. In rod1g, grid-039 and grid-040 ran above 35 deg (42.9 and
    45.5) and are checked against the board's reading. The joint fit leaves edge-01 out
    (`ur5_tip_calibration_rod_registration.py --exclude`).
  - 13:08, after the lab's 12:55 note that sph3t's first five onsets repeat sphere2's within 0.06 mm. The tool is
    then last night's, and the rod is fitted with the spheres' tool and its own centre. A joint fit of rod1t (without
    edge-01) and sphere2 leaves 52.6 µm rms on the rod over 21 and 57.4 µm on sphere2 over 29 (sphere2's own fit:
    57.0). The TCP moves 1.77 mm and 2.04 deg from nominal. Two more angles for the rod's axis turn it 6.6 deg and
    lower the rod's residual by 1.7 µm only, so the axis stays the homing's. The lab confirmed that no one measured the
    rod's level along its length; the ±7 mm touch-offs that would measure it are a later item, not tonight's. When
    sph3t's files land, the same fit takes sph3t and sphere2 together (`--joint sph3t,sphere2`), and rod1g is analysed
    with that registration.
  - 13:58, after the lab's b7a89430 (rod1, rod1t and sph3t whole). R19 is done; the robot stays at home, and any
    further run waits for the owner's go. The joint registration of rod1t (without edge-01), sph3t and sphere2 leaves
    55.6, 45.7 and 55.5 µm rms. Under the rod (radius 12.5 mm, 279 holds) the presses travel 0.069 mm + 0.100 mm/N
    beyond the FE pad, with 58 µm rms; rod1t's own fit gives 0.076 + 0.100. Under the sphere they travel 0.33 + 0.20
    (sphere2's own registration) or 0.17 + 0.22 (joint). A spring in series adds the same travel per newton under both
    indenters, so at most 0.10 mm/N is in the mount, the arm or the supports; the rest arises at the sphere's contact.
    The joint registration moves sphere2's contacts by 1.09 mm (median), 0.94 mm along the finger toward the mount.
    Neither indenter's touch-offs fix that direction well, so the results page keeps sphere2's own registration and
    reports the joint one beside it. Rod predictors after 45 presses: MLP 0.57 mm, GP 1.05 mm. grid-039 and grid-040
    (joint 6 above 40 deg) show no cable load: force 3 to 15 deg off the normal, board over HEX-E 0.55 to 0.66.
    Page: `20261022-ur5_tip_ground_truth_and_predictors.html`, Section 8 (84ddf755).

- R20 (moved out of this channel at 10:51 on 10-23 at the owner's request; nothing in it is for the lab session. It continues
  in its own conversation from `docs/handoff/20261023-ati_ft_calibration.md`, and its requests to the lab will go in
  `docs/experiments/20261023-ati_ft_calibration/QUEUE.md`. Planned on the workstation at 08:18; the owner, 10-23 morning: "the only way to permanently fix the
  HEX-E robot is to calibrate one of our uncalibrated ATI sensors"). Plan and parts:
  `docs/experiments/20261023-ati_ft_calibration/20261023-ati_ft_calibration_plan.html`. The calibration needs a weighed
  mass at five hook points of a printed cross, at eight tool orientations each, with and without the mass. Presses on
  the sphere alone excite three of six wrench directions and are the cross-check only. Nothing for the lab before the
  owner has measured the Mini45 and printed the parts. Then the steps are:
  - the ATI between v3's adapter and the fork (the tool 44.7 mm longer: plans rebuilt for it);
  - `scripts/ati_netft_logger.py --ip <box> --out <session folder>` beside the program;
  - eight checked poses per hook point (`move_guarded.py`), the cross's 70 mm arms checked against the robot's links.

- R21 (15:18 on 10-23; the owner's go, in the workstation chat: "we can run the 40 minute test then"). A day repeat of
  half of last night's grid on the sphere, about 40 min, to score last night's predictors on presses made a day later.
  - Resume sph3t (`lab/20261023-124052-sph3t/`), so its registration (0.043 mm rms) places the presses and no new
    touch-offs are made.
  - Presses: the even-numbered grid presses of sphere2's `plan_resume_2152.json` (grid-000, grid-002, ... grid-072),
    with the same force levels, in last night's order by joint 6. Leave out grid-061, -065, -066 and -067 and any
    press with joint 6 above 60 deg. The reference press at the start, in the middle and at the end. No rolls, twists
    or slides.
  - Before the run, `ur5_tip_calibration_check.py --clearance` with sph3t's registration, as for sph3t.
  - When it ends: the small files first, then the streams. The robot must be free for the owner's ATI work, so stop
    after the presses that fit if the owner needs it sooner.
  - 16:10, done (lab d69efe2d, 599364a4; sph3g 15:36 to 16:03, 27 grid presses and three reference presses, no abort).
    Analysed with sphere2's own registration (`data/20261023-124052-sph3t-s2reg/`, results page Section 9). Trained on all
    74 of sphere2's presses, the GP places the contacts of the presses made a day later within 0.80 mm and the MLP within
    0.71 mm. On the night's own held-out presses they reached 0.84 and 0.74 mm. At 1 and 2 N the board change at the same
    target differs from the night's by 38 and 20 %, against 35 and 23 % between the night's own repeats of the reference
    press. The pad met the sphere earlier: at the same force the registered indentation is 0.15 mm smaller (median), and
    the FE comparison gives 0.19 mm + 0.20 mm/N against the night's 0.33 + 0.20.

- R22 (draft at 16:00 on 10-23; nothing to run before the owner's go and answers. The owner, 10-23: three new tips with the
  checkerboard of poles, "which i will let you calibrate overnight"). Calibration of each new tip on the HEX-E stack as it
  stands. With the ATI in the stack the tool is 44.7 mm longer and every plan is rebuilt, so that plan waits for the
  owner's choice.
  - Plan: `plan/20261023-plan_sphere_lab_offset_touches.json`. It is last night's full plan
    (`lab/20261022-plan_sphere_lab_turns_checked_reg2144.json`) with 12 offset touch-offs after its 32 touch-offs. Every
    touch-off starts 5.0 mm above its predicted onset, as in sph3t.
  - Clearance. The first version of this plan (43566570, 15:54) failed `ur5_tip_calibration_check.py --clearance` with
    sphere2's registration. Three of its offset touch-offs put the finger 0.8 to 1.2 mm into the sphere near the tip end,
    and two brought the mount within 0.1 to 1.7 mm of it. The plan of 80f1f1ea (16:05) replaces it. Its offset touch-offs
    are chosen among those that clear the checker's margins plus 0.3 mm, both as a fresh session runs them and as planned
    with the registration. All 12 clear. Eight of last night's plain touch-offs stay under the margins with sphere2's
    registration (pad-edge-01, -02, pad-face-02, -04, -05, pad-flank-01, -02, -03). Drop them as for sph3t, which leaves
    24 plain and 12 offset touch-offs.
  - Why offset touch-offs. Every touch-off so far met the sphere at its top, with a vertical contact normal. Such a
    touch-off sees no horizontal shift of the sphere's centre. The place along the finger then rests on the homing
    seating, which the fit takes with a prior of 0.25 mm. Sphere2's two registrations put its contacts 0.94 mm apart along
    the finger for this reason. An offset touch-off aims its target 3 mm beside the sphere's top, along the finger, and
    descends vertically. A curved part of the pad then meets the sphere off its top, where the contact normal is tilted.
    The program already applies an action's `offset` in `target_pose`, so nothing in it changes.
  - Expected effect. In the covariance of the registration fit on sphere2's touch-offs, the 12 lower the standard
    deviation of the presses' contact points along the finger from 0.27 to 0.08 mm
    (`scripts/ur5_tip_calibration_touch_information.py --offset-mm 3`). In the simulator every homing seating was put
    0.6 mm off along the finger (`seat_bias_mm`, new in its Truth). The contact error at the presses then fell from 0.23 to
    0.17 mm under the default tool errors. Under sphere2's fitted tool errors it fell from 1.36 to 0.04 mm
    (`data/20261023-offset_touch_sim.json`). The first version's twelve, which failed the clearance check, reached 0.07 mm
    under the default errors.
  - An offset touch-off's onset comes after the same free travel as a plain one's. The program prints it against the
    target's standoff, so it reads about 0.1 to 0.3 mm later than a plain one (the action's `onset_shift_mm`), plus the
    registration's own offset.
  - One tip per session, in two passes. The owner seats the tip in DIP link b and homes the mount by hand on the key
    (`--home-from`, as on 10-22). Every registration of 10-22 and 10-23 lapsed when the finger came off at 16:30 (lab
    f05c1678). The offset touch-offs were cleared against sphere2's tool, so on the new stack they run only after its own
    registration:
    1. `plan/20261023-plan_sphere_lab_plain_touches.json` (the same plan without the offset touch-offs), stages preflight,
       homing, pad, register;
    2. `ur5_tip_calibration_check.py --clearance plan/20261023-plan_sphere_lab_offset_touches.json` with that registration,
       stages pad and grid; drop what falls under the margins;
    3. resume on the offset plan, stages pad and register: the offset touch-offs, now placed with the registered tool, then
       one fit of all touch-offs;
    4. grid and final. The grid is sphere2's, in its order by joint 6, without grid-061, -065, -066 and -067 and without any
       press with joint 6 above 60 deg. No rolls, twists or slides unless the owner asks.
    With the ATI in the stack both plans are rebuilt first.
  - Time per tip, from sph3t and sphere2g: 36 touch-offs at 2.3 min each take about 1 h 25 min, and 73 presses at
    0.9 min each about 1 h 05 min. About 2 h 40 min in all. A 3.5 mm start saves about 20 min if the first onsets come
    within 2.5 mm of the prediction. Three tips need someone to swap them, so one tip can run unattended per night.
  - Analysis: the rest fit chooses the poles (the checkerboard or its mirror). Each tip gets the physical model, the GP
    and the MLP, and sphere2's correction is scored on it (the transfer of Table 8 on the results page).
  - Questions for the owner: (1) is each new tip today's pad with only the magnets changed; (2) one tip tonight, or
    sessions with a swap in between; (3) is the ATI in the stack tonight?
  - The owner's answers (16:42, workstation chat): (1) "same pad, magnet polarities changed to checker board", bottom row
    N S, middle N, top row S N (bottom at the mount end: the centre and one diagonal north, the pattern simulated as set
    D); (2) "only one tip. we will swap tips in the evening"; (3) the ATI stays in the stack "only if the calibration goes
    well". The workstation's advice is to leave it out tonight (one change at a time; the plans are built and checked
    for the HEX-E stack; the ATI's force scale is better measured in a daytime run with both sensors in series).
  - The rest fit searches all 32 pole patterns (`fingertip_wrench_model.calibrate_frames_fixed`), so the analysis finds
    the checkerboard or its mirror by itself. Simulated on sphere2's mechanics: with today's magnets in this checkerboard
    the board changes by 51 uT (median) against 36 uT for today's pad; sphere2's correction gives 1.26 mm and a GP
    retrained on the new tip 0.98 mm. With N52 magnets they give 0.79 and 0.58 mm
    (`data/20261022-210852-sphere2/predict/magnet_transfer_sim_ADE.json`).
  - 19:47, the owner (workstation chat): mount v5 goes on the robot with the first checkerboard tip ("the new dip link
    mount and alternating polarities fingertip"). v5 is v4's fork with the PLA wedge behind the link's tongue
    (`docs/experiments/20261023-ur5_calibration_mount_v5/`). Its frames are v3's, so the plans keep their poses.
    - Pull first. Both plans now set v5's payload at preflight: 0.3993 kg with its centre 34.3 mm from the flange (v3's
      entry was 0.3839 kg at 32.5 mm).
    - Pass 1 runs the stages preflight, homing, gravity, pad and register. The gravity sweep measures the new payload,
      and the analysis then needs no other session's fit. Expect about 1.72 N: first2's 1.567 N plus v5's 15.4 g. Pass
      `--gravity-expect-N 1.72` or leave it out. First2's 1.57 N would stop the session at the 5 % check.
    - Run every clearance check with `UR5_MOUNT=v5`, after
      `git lfs pull --include "docs/experiments/20261023-ur5_calibration_mount_v5/cad/*"`.
    - Checked with v5 and sphere2's registration over pad, grid and final (19:46). The wedge's space stays at least
      8.72 mm from the sphere and the cell as a fresh session runs the touch-offs, and 9.45 mm as planned with the
      registration. The mount's least clearance is 2.50 mm as run and 2.84 mm registered; with v3 it is 2.52 and 2.75 mm.
      The same eight touch-offs fall under the margins as with v3 and are dropped. grid-025, -027 and -028 come 0.01 to
      0.16 mm inside the mount's 3.0 mm with sphere2's registration; pass 2's check with the new registration decides them.
    - The owner notes the wedge fin's distance from the plate's end before and after the session (Section 7 of the v5
      page).
    - The magnet grade does not change the plan. At rest, N52 magnets in this checkerboard put at most 19.7 mT on a chip
      and today's magnets 6.7 mT. The board's range is 38.4 mT. The rest fit gives the strength and the poles.
  - Lab, 19:59 by its clock: pass 1 built as `lab/20261023-plan_sphere_lab_r22_pass1.json` and simulated. It has sph3t's
    19 touch-offs: the eight under the margins and five flanks with joint 6 above +60 deg are dropped. Nine of the 12
    gravity orientations remain, joint 6 from -147.9 to +48.8 deg; the design's condition is 2.9 (2.2 with all 12).
    pad-face-11 is 2.95 mm from the mount with sph3t's tool, as with v3, and stays. Pass 2 also drops fgrav-01 and
    grid-060, -062, -063 and -064 (joint 6 +64.0 to +66.2 deg).
  - Workstation, 19:59: agreed. At sph3t's registration, over the 65 grid presses left, the 19 touch-offs leave the
    contact points' standard deviation along the finger at 0.348 mm (median over the presses). With the 12 offset
    touch-offs it falls to 0.105 mm; sphere2's 29 touch-offs give 0.268 and 0.083 mm. In the simulation of 16:05 the
    offset touch-offs end at joint 6 from -57.2 to +51.6 deg, so the +60 deg rule drops none of them.
  - The owner's go, relayed by the lab at 20:23 by the workstation's clock. The session is named cb1t. The mount is v4's
    fork without the wedge ("for various reasons the v5 with the wedge does not work"), or v3; the lab has asked which.
    The clearance checks cover both, because v5's geometry contains v4's fork. Pass 1 has no gravity sweep. The owner:
    "that's how we knocked out the hex e sensor last time". It runs preflight, homing, pad and register, as sph3t did.
    Blockers at 20:23: the HEX-E's DAQ answers ping but sends no samples, and the tip's board is not on corvus's USB.
  - Workstation, 20:23: cb1t is analysed with `--gravity-from lab/20261022-195420-first2`, as sphere2 and sph3t were. R22
    has no turns, so each hold and its rests share one orientation. The gravity term then vanishes whatever the payload.
    For v4 the plan's v5 payload entry is within 1 g; for v3 it is set to 0.3839 kg at 32.5 mm. Without the wedge or
    screws, v4's fork lets the link turn as in v3. cb1t's travel beyond the FE pad should then compare with sphere2's.
  - Done (lab, 10-23 20:46 to 23:58; pushed 19ed9e25 and a1ad8d74). Workstation, 10-24 00:29: analysed. Page
    `20261024-checkerboard_fingertip_on_the_ur5.html`, analysis in `data/20261023-203938-cb1t/`. The turns added at the
    owner's request use first2's gravity fit as well.
    - The lab's question, pass 2's registration: kept. Three measurements that neither fit used agree better with it.
      With pass 1's registration (`data/20261023-203938-cb1t-pass1reg/`), the grid's indentation beyond the FE pad
      leaves 142 um rms, and with pass 2's 108 um. The FE force error falls from 23.5 to 16.0 %. The board's point lies a
      mean 1.54 mm from the registered one with pass 1's and 0.72 mm with pass 2's. The 1.96 mm move of the sphere's
      centre is 5.4 times pass 1's sideways standard deviation, so plain touch-offs alone leave that much loose. Every
      session should run the offset touch-offs.
    - Poles: (+1, -1, -1, -1, +1) fits the rest at 0.24 mT, the next pattern 0.87 mT, k 0.568. That is the plan if its
      poles were named from the board's side. Read from the palm, the plan is (+1, -1, +1, -1, +1), which leaves 1.81 mT:
      the centre magnet is reversed. The owner is asked which side the plan used.
    - Physical model from the rest fit: 2.06 mm and 0.279 N on the held-out presses (sphere2's tip 3.79 mm), 1.1 to 1.2
      mm at 2 N and over. Sphere2's predictors miss by 2.5 to 8.6 mm; 47 of cb1t's presses give 0.71 mm (GP) and 0.42
      mm (MLP). Series compliance 0.188 mm/N on v3's fork (sphere2 0.203).
    - The HEX-E's zero moved at most 0.254 N between a press's rests, so the taped cable pulled through no press.
    - Owner, 10-25: "ok, it was probably S N S N S then", so the plan's poles were named from the board's side and the
      tip follows the plan. Its magnets are sphere2's ("the exact same as the past tip"). The rest field fixes their
      strength and depth only together: k 0.336 with the magnets 0.71 mm off the model's depth leaves 0.33 mT, against
      0.24 mT at k 0.568 and 1.87 mm (`data/20261023-203938-cb1t/k_check.json`, `scripts/ur5_tip_calibration_k_check.py`).
      Labels: cb1t is v1 ID 1.

- R24 (10:51 on 10-25; owner: "are we prepared for the next fingertip? this will be the same polarities, labeled v1 ID
  2 ... i am printing out the new ur5 mount and will try with that to reduce the bending the effect"). The second
  checkerboard tip, v1 ID 2, on mount v6, as R22 ran on cb1t.
  - Mount v6 (`docs/experiments/20261023-ur5_calibration_mount_v6/`): `UR5_MOUNT=v6` in every clearance check. Its parts
    are the adapter, the fork and the clamp plate, which `check_arm_samples.py` already groups.
  - Payload: about 0.399 kg at 34.3 mm from the flange (v5's record without the wedge and the slotted arm's insert; v6's
    fork slices at 63.7 g).
  - Pass 1: `lab/20261023-plan_sphere_lab_r22_pass1.json` (19 plain touch-offs, then register). Pass 2: the offset plan
    with the rolls and twists, as `lab/20261023-plan_sphere_lab_r22_pass2.json`. Put back the actions dropped for cb1t's
    registration, and check them all again with ID 2's registration.
  - Margins on v6 (owner, 10-25, about the rod plan's pad-face-05 at 2.57 mm: "i think we can allow this tolerance"):
    2.5 mm at the mount for touch-offs; 3.0 mm kept for presses at 3.5 and 5 N. At 5 N cb1t's presses went 1.89 mm into
    the pad, and the check looks 1 mm past contact. With cb1t's registration as the stand-in, v6's closest actions
    are touch-offs: pad-face-11 2.66, pad-face-07 2.70 and pad-off-04 2.86 mm (on v3 2.70, 2.70 and 2.96 mm).
  - No gravity sweep. Tape the board's USB cable as in cb1t's resume before the first touch-off.
  - The session's name is the lab's choice; its notes record v1 ID 2. The workstation analyses it as cb1t, scores
    cb1t's predictors on it, and compares the series compliance with cb1t's 0.188 mm/N to measure v6's effect on the
    bending.
  - The owner's go, 10-25 by 11:41 (workstation chat): "i give my approval for all the current and subsequent
    passes". Passes 1, 2a, 2b and 3 run one after another, each after its clearance check and simulation, with no
    further go. Actions that a check drops need no go either.
  - Pass 3, after pass 2b. The owner asked for "a very comprehensive, dialed in test ... registration, touchoffs, grid,
    roll/twists, and more wrench calibration at the end of all this". On the viewer, the owner sees contact detection
    "pretty good on the fingertip's right side, but pretty bad on the left", and "the contact direction is in general
    pretty noisy". The plan is `lab/20261025-plan_sphere_lab_r24_pass3.json`, written by
    `scripts/ur5_tip_calibration_r24_pass3.py`. Its stages are grid, shear, twist and final, with 31 actions:
    - 24 twin presses, `twin-NNN-tT`. Each is the mirror about y = 0 of a builder grid target whose mirror had no
      target within 1 mm. A twin keeps the original's levels and takes the first turn about the vertical that clears
      (0, +30, -30, +60, -60 deg). The builder's grid has 30 targets at y < -1.5 mm and 19 at y > 1.5 mm; with the
      twins both sides are pressed alike.
    - 3 slides: shear-0-1N, shear-0-2N and shear-3-1N. Each slides the pad over the sphere in the builder's eight
      directions at 0.1, 0.2 and 0.3 mm, without the slip steps of 0.5 and 1 mm. With the slip steps, six of the seven
      slides fell under the margins on v6; without them, four still do. No slide has run on the robot yet, so simulate
      them, run shear-0-1N first, and look at its HEX-E tangential force before running the other two. The presses
      carry 0.1 to 0.45 N of shear, so the slides give the shear calibration that the force direction needs.
    - 3 twists at 5 N at pass 2's face points 0, 2 and 3: one side, out and back at 5, 10 and 15 deg, with 1 s holds.
      On cb1t the torsion at 2.5 N, half the difference between the way out and the way back, was 3 to 10 N mm.
    - final-ref.
  - Workstation check with `UR5_MOUNT=v6` and cb1t's registration (`scripts/ur5_tip_calibration_check.py --clearance`):
    every kept action clears the mount by 3.0 mm, the finger by 0.8 mm and the HEX-E by 6 mm. Joint 6 was not checked.
    Drop the actions above +60 deg, as in R22, and redo the clearance check with ID 2's registration. With cb1t's
    timings (70 s a grid press, 2.8 min a twist, about 25 s a turn hold), pass 3 takes about 75 minutes.
  - The workstation compares each twin's errors with its original's at the same levels, and trains the force estimate
    on the presses and the slides.
  - 10-25 by 13:2x (the lab's messages and the workstation's replies): R24's passes run first on v1 ID 1 on v3 as
    session cb1r (the owner's choice). The owner wants ID 2 on v6 by 14:30, so cb1r runs pass 3 after 2a, then pass
    3b, then pass 2b's grid until 14:20; pass 2b's rolls and twists are skipped. On the robot, the +30 deg twins put
    joint 5 at 17.7 to 27.6 deg, outside the 30 to 100 deg wrist window: pass 3 kept 7 twins. Pass 3b,
    `lab/20261025-plan_sphere_lab_r24_pass3b_cb1r.json`, has 12 of the other 17 turned +10 or +15 deg.
  - Speed for cb2t (owner, 10-25: "it would be nice to run them faster ... speed up motion in between trials"). The
    program at cea8b477 reads these keys of the plan's travel block. Without them a plan runs as before.
    - `speed` scales the speeds and accelerations of the moves between actions (joints at most 1.5 times).
    - `slow_mm_touch` and `slow_mm_press`: an approach crosses its standoff at `fast_v` (m/s) and starts its slow speed
      that far before the expected surface. A push over 0.3 N on the way sends the tool back to the rest pose, and the
      approach then runs slowly from there, as before.
    - `turn_v` sets the turns' speed in contact (default 0.004, as before).
    - Put `"speed": 2.0, "fast_v": 0.005, "slow_mm_touch": 1.5, "slow_mm_press": 0.5, "turn_v": 0.016` in every cb2t
      plan. The owner, 13:2x: "we should add the rolls, twists, slides speedup in addition to the current speed. we can
      save that for id2". On the CB3 the turns' speed acts as rad/s, so rolls, twists and slides turn at about 0.9 deg/s
      instead of 0.23 deg/s, and a turn hold takes about 7 s instead of 25 s. Presses and holds do not change. Pass 1's
      onsets came within 0.84 mm of the expected on cb1r, so 1.5 mm leaves at least 0.66 mm of slow travel before
      contact.
  - The rate set in cb2t's pass 3 (owner, 13:3x: "id2 should have fast and slow turns (ideally more than two
    configurations), isn't this what we need to do to have robust contact poses and ideally wrenches?"). An action's own
    `turn_v` now overrides the plan's. The set has 14 actions:
    - twist-0 at 2.5 and 5 N, roll-0 and shear-0-1N, each at `turn_v` 0.004 (0.23 deg/s) and 0.04 (2.3 deg/s), beside
      their copies at the plan's 0.016;
    - six grid points (grid-000, -009, -015, -023, -032, -043) pressed to 2 N at 0.15 and 2 mm/s, beside the grid's 0.5
      mm/s.
    Rate then varies at a fixed point, force and angle. In the simulator the press at 2 mm/s peaked at 2.46 N against
    2.20 N at 0.15 mm/s, and every action ended ok. On the robot the set takes about 22 min, most of it the four turns
    at 0.23 deg/s.
  - Mount v7 for ID 2, if the owner chooses it over v6 (13:4x: "i might also be able to get id 2 on the v7 dip link
    fork"). The cell model takes `UR5_MOUNT=v7`. With cb1t's registration as the stand-in, pass 1's and pass 2's
    closest actions are pad-face-07 at 2.68 mm and pad-face-11 at 2.70 mm (v6: 2.66 and 2.70). Both stay under the
    touch-off margin of 2.5 mm, and every press and turn of passes 2 and 3 clears 3.0 mm (pass 3 at least 3.39 mm). v7's
    fork is 5.9 % larger by volume than v6's (97,621 against 92,179 mm3), about 3.8 g at v6's 63.7 g, so set the payload
    to 0.403 kg at 34.3 mm, or weigh the fork. The link seats where v6's does, so the frames do not change.
  - Tonight's suite on cb2t, unattended (owner, 15:0x: "we can just do speed settings now, cant we? im about to put on
    id 2 and go home anyway, so we have time to run a big suite"; the owner never set the pendant's slider).
    - Program: a plan's `speed_slider` is set on the controller (RTDE IO) and read back at the start of every run. A
      difference over 0.02 stops the run before any motion; if the controller refuses the input, a warning is printed
      and the value is checked. cb2t's plans from 2b on carry `speed_slider` 0.30, the value every session so far ran
      at, so cb2t's rates match ID 1's. Simulated: "speed slider 0.30 (the plan's 0.30)", then a touch-off pass, a
      press and two slides, no abort.
    - Order: 1 and 2a (prepared, 1ed52366); 2b, R22's grid, rolls and twists with the travel block and `speed_slider`;
      3, `lab/20261025-plan_sphere_lab_r24_pass3_cb2t.json`; 5, `lab/20261025-plan_sphere_lab_r24_pass5_cb2t.json`.
    - Pass 3 has 56 actions: 19 twins, the 3 slides, the 3 twists at 5 N, the rate set at four speeds, and final-ref-p3.
      Pass 5 has 103: the grid again at v 0.002 (fast-grid-*), shear-0 at 0.5, 1.5 and 3 N and shear-3 at 2 N at
      `turn_v` 0.016, 46 slides at 23 points on both sides at 1 and 2 N with 0.1 and 0.2 mm steps at `turn_v` 0.04, and
      final-ref-p5.
    - At the slider's 0.30 the actual rates are 0.045, 0.15, 0.6 and 2.0 mm/s for presses and 0.07, 0.28, 0.69 and 2.3
      deg/s for turns. Every kept action clears the margins on v7 with cb1t's registration, and the joint windows with
      cb1r's. At cb1t's pace the suite takes about 6 h.
    - Why pass 5 has so many slides: on cb1r a GP trained on the presses and the other slides put the slides' shear
      within 0.10 N and the force's direction within 5.7 deg. Trained on the presses alone, it was 0.24 N and 13.2 deg;
      the physical model gives 0.98 N and 57 deg (`data/20261025-114653-cb1r/checks.json`).
  - Pass 6 for cb2t, after pass 5 (owner, 16:3x: "improvisationally devise more tests ... (more speeds, more variations,
    more locations/location density), torsion, contact wrenches"): `lab/20261025-plan_sphere_lab_r24_pass6_cb2t.json`,
    written by `scripts/ur5_tip_calibration_pass6.py`. It has 77 actions:
    - 38 new press targets, by farthest-point sampling at least 1.5 mm from every target of passes 2b, 3 and 5: 9 at y
      < -1.5 mm, 7 between, 22 at y > 1.5 mm; 11 on the face, 15 on edges, 12 on flanks. Each is pressed once at one of
      four contact speeds in turn: 0.15, 0.6, 0.045 and 2.0 mm/s actual.
    - 36 of them slid at 1 N, 0.1 and 0.2 mm in eight directions (`turn_v` 0.04).
    - twist-3 at 2.5 and 5 N out to 20 deg and back. twist-0 at 20 deg and the 25 deg twists bring joint 5 under 34 deg.
    - final-ref-p6.
    For an unattended run, joint 5 keeps 4 deg above the window for presses and twists and 8 deg for slides, on cb2t's
    registration. The first build lost a slide and twist-0 at 20 deg to the window in the simulator. The mount clears
    by at least 3.03 mm on v7 (cb1t's registration). The whole pass ran in the simulator with no abort. It takes about
    85 min at cb1t's pace.
  - The owner, 13:2x: "we can afford to drop pass 2 right? it was just a repeat of friday's run?". cb1r drops pass 2b
    (its grid, rolls and twists repeat cb1t's). Pass 2a's offset touch-offs and registration stay, because pass 3 runs on
    that registration. After pass 3 and 3b the lab homes the robot for the swap.
    - At cb1t's timings a touch-off takes about 65 s instead of 142 s, and a press about 39 s instead of 54 s. The 31
      touch-offs of passes 1 and 2a then take about 35 min instead of 73.
    - In the simulator, 19 touch-offs took 18.7 s each against 43.1 s (median). The registration fitted at 0.013 against
      0.016 mm rms, and every press reached its levels.
  - Pass 3 for cb2t is `lab/20261025-plan_sphere_lab_r24_pass3_cb2t.json`, built by the pass 3 builder that now checks
    the joints. It has 20 twins, 3 slides, 3 twists at 5 N and final-ref-p3. By the simulator's IK on cb1r's
    registration, joint 5 runs from 35.7 to 75.8 deg and joint 6 from -64.3 to +31.5 deg. On v6 with cb1t's
    registration the mount clears by at least 3.6 mm. Recheck it with cb2t's registration.

- R25 (12:24 on 10-26; owner: "we ready to put on id 3?"). The third checkerboard tip, v1 ID 3, goes on mount v7 in
  place of ID 2, as session cb3t. It runs cb2t's suite unchanged, so that the three tips are compared on the same actions.
  - Plans: `lab/20261026-plan_sphere_lab_r25_<pass>_cb3t.json` for passes 1, 2a, 2b, 3, 5 and 6, with 28, 31, 67, 55,
    101 and 73 actions. Each is cb2t's plan as it ran, after the lab's drops. Passes 1 and 2a now set `speed_slider` to
    0.30, as the later passes do. One simulator run of pass 1 checks that key (README rule 2).
  - Payload: 0.403 kg at 34.3 mm, as for ID 2. If ID 3's finger differs from ID 2's in its board or cable, weigh it.
  - Order: 1, 2a, 2b, 3, 5, 6, then home the robot. After 2a, check passes 2b to 6 with cb3t's registration on
    `UR5_MOUNT=v7` and drop what fails, as on cb2t. The margins stay at 2.5 mm for touch-offs and 3.0 mm for presses
    and turns.
  - On cb2t the suite ran from 15:29 to 23:42 with no abort. Push after each pass, and the streams at the end.
  - The workstation then analyses cb3t with cb2t's checks and compares the three tips, including ID 1's and ID 2's
    models applied to ID 3.

- R26 (14:1x on 10-26; owner: "maybe we can add harder (much harder?) presses on id 3 ... we can build up to it
  gradually ... ill be here all day ... we have the 10N limit anyway"). Hard and fast contacts on cb3t, with the owner
  present, after pass 2a and before pass 2b. The viewer rejects presses past 3.5 to 4 N, because the calibration stops at
  5 N on the face, 3 N on the edges and 2 N on the flanks. Built by `scripts/ur5_tip_calibration_hard_pass.py`.
  - hard1, `lab/20261026-plan_sphere_lab_r26_hard1_cb3t.json`: the reference press (0.5, 1 and 2 N), a press at the
    reference point to 1, 2, 3.5, 5, 6, 7 and 8 N at 0.15 mm/s, and the reference press again. Compare the two reference
    presses' board changes at each level with the spread of ref-00 to ref-05 in cb2t's pass 2b. A difference several
    times that spread means the pad moved in its seat: stop and tell the owner.
  - hard2, `lab/20261026-plan_sphere_lab_r26_hard2_cb3t.json`, 57 actions in order of force:
    - 28 presses at 0.15 mm/s: 6 on the flanks to 4 N, 10 on the edges to 6 N, 12 on the face to 8 N;
    - 6 presses at the rate set's points to 1 and 3 N at v 0.0133. The run-up of 0.5 mm limits the speed at contact to
      about 4 mm/s, twice cb2t's fastest;
    - slides at 4 N at 6 points, and at 2 N with turns at 6.9 deg/s (`turn_v` 0.4) at the same points;
    - shear-0 at 4.5 N, shear-3 at 4 N, roll-0 at 4 N, twist-0 to 15 deg and twist-3 to 20 deg at 6 N;
    - roll-0 at 1.5 N, and twist-0 at 2.5 and 5 N, with turns at 6.9 deg/s;
    - the reference press after the presses and at the end.
  - Checks, on 134522-cb3t's registration and v7. On cb2t a press went 0.461 F^0.782 mm past contact (2.34 mm at 8 N by
    the fit), and the clearance check looks 1 mm past contact. Each action's margins therefore exceed the usual ones
    (mount 3.0, finger 0.8, HEX-E 6.0 mm) by its depth at its largest force less 1 mm. The closest mount margin is 4.06
    mm. Presses may go 4 mm past the expected surface (`max_depth_mm` 4.0). Joint 5 stays at 34 deg or more, joint 6
    within -65 and +50 deg. The abort stays at 10 N. Run both plans in the simulator first (README rule 2).
  - At 2 mm/s cb2t's presses overshot their level by 0.65 N. At 4 mm/s expect up to about 3 N, so the 3 N level may
    peak near 6 N. Faster presses need a stronger stop in the program first (`stopL` decelerates at 0.05 m/s^2).
  - About 1.5 h in all. Passes 2b to 6 follow as planned.

- R27 (15:05 on 10-26; owner: "i think it can't hurt to get higher resolution + some data parity on the rod ... might
  as well design and tack it on. we can do rolls, twists, shear, variable speeds/forces on it too. if it truly is
  redundant, we'll find out!"). A rod session on v1 ID 3, with the owner present, after R26 and before cb3t's pass 2b.
  On the first tip on 10-23, a GP trained on sphere presses placed rod contacts within 0.50 mm at 1.5 N and over, and
  within 1.61 mm under 1.5 N. With rod presses in training the errors were 0.41 and 1.14 mm. The rod's give beyond the
  FE model was 0.10 mm/N, against the sphere's 0.20. The lab builds both plans, because the workstation's weekly usage
  is above 90 %.
  - Rod pass 1, the registration: R19's `plan/20261023-plan_rod_lab_touch35.json`, with homing on the rod's key, 22 pad
    touch-offs at a 3.5 mm standoff, 5 pivot presses and the gravity sweeps. Take v7's `T_EM`, `T_MP`, payload, travel
    block and `speed_slider` 0.30 from cb3t's pass 1; keep R19's indenter block, key and placement. Check it on v7 with
    134522-cb3t's registration as the stand-in, simulate it, and run it as a new session.
  - Rod pass 2, on rod pass 1's registration, every ID starting with `rod-`:
    - R19's 53 grid targets, and up to 30 more by farthest-point sampling of the FE surface away from them, at 0.15
      mm/s to 8 N on the face, 6 N on the edges and 4 N on the flanks, with the standard levels below those;
    - at six targets spread over the face, presses to 2 N at 0.045, 0.6 and 2.0 mm/s (v 0.00015, 0.002, 0.0067), and to
      1 and 3 N at about 4 mm/s (v 0.0133);
    - slides around the rod at `dir_deg` 0 and 180 only, with `degs` 0.458 and 0.917 (0.1 and 0.2 mm on R 12.5 mm), at
      1, 2 and 4 N at eight targets with `turn_v` 0.04, and at 2 N with `turn_v` 0.4. The rod is straight along its axis,
      so a turn at `dir_deg` 90 would tilt the pad over the contact line;
    - twists about the vertical at 2.5 and 5 N to 15 deg with `turn_v` 0.016, and at 5 N with `turn_v` 0.4;
    - rolls at `dir_deg` 0 (about the contact line), 3 and 6 deg each way, at 1.5 and 4 N, and at 1.5 N with `turn_v`
      0.4;
    - the rod's reference press at the start and at the end.
  - Checks as in R26: margins raised by each action's depth at its largest force less 1 mm, with the sphere's fit as the
    bound (the rod went in less); joint 5 at 34 deg or more; joint 6 within -65 and +50 deg. Simulate the whole pass.
    `scripts/ur5_tip_calibration_hard_pass.py` shows the target selection, the margins and the joint checks; its plan
    head and turn moves need the rod's.
  - About 30 min for pass 1 and 2.5 h for pass 2. cb3t then resumes at pass 2b.

- R28 (18:47 on 10-26; changed at 18:57, owner: "change of plans, we're doing v2 id 1 (five n52 magnets). so same
  calibration"). The tip v2 ID 1 has five N52 magnets in ID 1 to 3's checkerboard, on mount v7, as session `v2id1`. The
  four-magnet tip v3 ID 1 waits.
  - Stop cb3t at the end of its current action or plan, home the robot and push its files.
  - Run cb3t's plan files in cb3t's order: pass 1, pass 2a, then passes 2b, 3, 5 and 6 with joint 6 capped at +45 deg.
    While the owner is present, run the 24 actions above the cap as on cb3t (lowest joint 6 first). Otherwise they wait
    for the owner. After pass 6, run R26 (hard1, then hard2 if the two reference presses agree). Run R27's rod passes
    only if they are built and checked without long work; otherwise they wait for the reset.
  - Before pass 1: confirm with the owner that the pad's outer shape is B00's, as on ID 1 to 3, since the plans' targets
    come from its FE surface. In pass 1, compare each chip's field at rest and at 5 N with the chips' range at the
    board's gain. N52 raises the field, and a clipped chip would spoil the run.
  - After pass 1, fit the poles at rest (`fingertip_wrench_model.calibrate_frames_fixed` on the board's earlier frames):
    ID 1 and ID 2 gave (+1, -1, -1, -1, +1). The exact reverse means every magnet is turned, which the analysis can undo
    in software. Any other pattern means one or two magnets are turned: stop and tell the owner.
  - Until the reset at 04:00, both sessions keep their reasoning short. Planning or debugging that needs more goes to
    an autopilot job on the personal account (owner).

- R29 (13:23 on 10-27; owner, 10-26: "seems like contact density is a big one. i will mount the taller indenter tmw
  too"; the lab asked the workstation for the tall indenter's next plan, the README's draft request). v2 ID 1 on the
  tall indenter, after its pass 1 (three seatings, pass 1 with `--indenter tall --home-from` the first, as the lab planned).
  - The workstation builds the plan on pass 1's session: `scripts/ur5_tip_calibration_tall_pass.py --session
    lab/<tall pass 1> --base-plan <its plan.json>`. Targets are the points with no sphere press of v2 ID 1 within 1 mm
    that the tall indenter reaches, 1 mm apart. Each gets a press at the grid's levels (face 0.25 to 5 N, edge to 3 N,
    flank to 2 N) and a slide at 1 N (eight directions, 0.1 and 0.2 mm). Twists to 20 deg at 2.5 N and 0.28 deg/s run
    at six face and six edge points. Reference presses go first, after the presses and last.
  - Built on v2 ID 1's sphere registration as a check, it has 137 actions with joint 6 capped at +45 deg. They are 68
    presses (18 face, 26 edge, 24 flank), 53 slides and 12 twists; the least mount margin is 3.01 mm. With joint 6 to
    +60 deg (owner present) it has 161 actions, 13 more flank presses, most of them on the -y side.
  - Push pass 1's session when it is done; the plan follows within minutes and is simulated on that registration
    before it runs. Pass 1 is admitted by the README's marks: within 0.05 mm rms, and a series compliance within
    0.01 mm/N of cb3t's.
  - 14:26 on 10-27, owner: "we can allow the +60 j6, i am here for the next ~5 hours so can debug as needed". The plan
    is built with joint 6 up to +60 deg and runs while the owner is present (until about 19:00); the actions over +45
    deg go first among the presses, so that what is left after 19:00 stays under the cap.
  - 15:29 on 10-27: built on pass 1 (`lab/20261027-143529-tallv2id1`, its registration and plan header with the sphere
    and the rod as obstacles): `lab/20261027-plan_tall_lab_r29_v2id1.json`, 122 actions. They are 56 presses (15 face,
    19 edge, 22 flank; 20 at y < -1.5 mm, 24 at y > 1.5 mm), 50 slides, 12 twists and 4 reference presses. Built with
    joint 6 within -62 and +58 deg (the lab's sim overshot a nominal -64.8 to -66.3), joint 5 from 37.7 deg. The
    finger keeps 1.5 mm on flank actions and twists (the lab's cap at 0.17 mm on pad-flank-02), 0.8 mm elsewhere. By
    `--clearance` on the session the least margins are mount 3.25, finger 1.54 and HEX-E 22.1 mm. The 15 actions past
    +45 deg (8 presses, 7 slides, up to +55.1 deg) all lie within the first 65; no twist passes +45.
  - Simulate on the session's registration, then run with the owner's go.
  - 17:50 on 10-27, the workstation on the 45 presses of a6fe16be (`scripts/ur5_tip_calibration_tall_r29.py`,
    `data/20261027-143529-tallv2id1/r29_tall_vs_sphere.json`). Flank FE points within 1.5 mm of a press go from 23 to
    29 %, edges from 83 to 93 %, the face from 89 to 99 %. A GP of the sphere presses places the new points at 0.98, 1.04
    and 1.21 mm (face, edge, flank, at 1.5 N and over). With the other tall presses in training these are 0.65, 0.94 and
    0.67 mm, and the flank's angle error falls from 8.4 to 4.9 deg. At 1.0 to 1.25 mm from the nearest training press the
    tall presses err by 0.82 mm and held-out sphere presses by 0.36 mm. The tall presses' forces have a shear part along
    the finger (z_P) of 0.08 to 0.13 N per newton, against -0.02 to +0.02 in the sphere session, which the sphere GP does
    not predict (0.02); its angle error grows from 3.2 to 13.2 deg with that shear. The magnets' rest fit is the same in
    the sphere, rod and tall sessions within 0.01 mm. Resuming is fine from the data side. Proposed while the owner is
    present: three reference presses at index 788 on the sphere fixture, with its centre from 192158-v2id1 and the pad's
    pose from 143529. Shear there means the finger or mount changed; none means the tall indenter or the arm's pose.
  - 18:01 on 10-27, built after the owner's yes (about 17:55, through the lab: "we can build it and try to run it after r29 finishes ...
    whatever we find can be used to back-calibrate the data"), with the lab's two points: the tall indenter as an obstacle
    (kind "tall", 143529's T_BI) and a set on 192158's own registration. `scripts/ur5_tip_calibration_sphere_check.py`
    writes `lab/20261027-plan_sphere_lab_check_shear_v2id1.json` (three presses as ref-00 at index 788, 0.5, 1 and 2 N;
    travel lift 30 mm) and two session folders: `lab/20261027-sphchk-a-v2id1reg` (192158's registration) and
    `lab/20261027-sphchk-b-hybrid` (T_EM and T_MP from 143529). Each is resumed with stages preflight and grid, a first,
    then b; no homing or touch-offs. By `--clearance` the least margins are mount 4.00 and 5.03, finger 4.17 and 5.00,
    HEX-E 35.5 and 36.5 mm. Shear along z_P in both sets means the finger or mount changed; in b alone, the tall
    registration's orientation; in neither, the tall indenter's flex or the arm's pose.

- R31 (22:11 on 10-27; the owner's go to the lab at 21:5x: "sure, we can run the v3 id 1 registration and suite ... the rod
  has changed position, yes, but everything else is the same"; joint 6 at +45 deg or less without the owner). v3 ID 1's
  suite, built by `scripts/ur5_tip_calibration_v3_suite.py` from v2 ID 1's sessions as they ran, with
  `suite_value.SUITE`'s cuts, in this order; each plan's checks.other holds the tall indenter, the rod at its three
  seatings and the holder with both held tips (kind tiphold). Times are v2 ID 1's at the program's present speeds.
  1. `lab/20261027-plan_sphere_lab_pass1_v3id1.json`: 18 touch-offs, stages preflight, homing, pad and register, about
     0.5 h. Its 9 gravity poses swing joint 6 from -148.4 to +48.3 deg in the lab's simulator and moved to the morning.
  2. `lab/20261027-plan_sphere_lab_stage2_v3id1.json` (c60d6458): 264 actions, about 6.8 h, resumed on pass 1's
     registration. 19 presses keep the mount 2.38 to 2.92 mm from the sphere's own fixture at 2.5 mm, as on v2 ID 1.
  3. `lab/20261027-plan_tall_lab_pass1_v3id1.json`: v2 ID 1's 21 touch-offs on the tall indenter (to 2.0 mm past); all
     ran there on v2 ID 1 within the program's windows (joint 5 to 98.4 deg, joint 6 to -64 deg).
  4. `lab/20261027-plan_tall_lab_stage2_v3id1.json`: R29 with 4 of its 12 twists and six shared points after tall-ref-a
     (tall-shared-728, -730, -719 on the face, -581, -576, -640 on the edges, at the levels of sphere grid-029, -009,
     -013, -032, -051 and -034), 103 actions. Path check with presses at their largest force's depth: finger 0.94, mount
     2.80, HEX-E 17.7 mm at least; 4 actions reach joint 6 -61 to -63 deg.
  The owner's morning: `lab/20261027-plan_sphere_lab_morning_v3id1.json` (pass 1's 9 gravity poses, then 55 actions, 1.1 h: joint 6 past +44 deg in
  v2 ID 1's run, and grid-055, fast-grid-055 and hard-grid-055, the finger within 0.36 mm of the sphere),
  `lab/20261027-plan_tall_lab_morning_v3id1.json` (17, joint 6 to +55 deg), pad-edge-07 of sphere pass 1, and the rod:
  - `lab/20261027-plan_rod_lab_pass1_search_v3id1.json`: 21 touch-offs from 8 mm above the expected contact to 5 mm past
    it, slow all the way, and 5 pivots. With the program homing on rod_indenter_seated_3 and the rod at each of its three
    seatings, every touch-off meets the rod 7.6 to 9.5 mm into its approach, but at contact 8 to 13 of them keep the
    mount only 1.25 to 2.85 mm from the rod's fixture.
  - `lab/20261027-plan_rod_lab_stage2_v3id1.json` (98 actions, 1.2 h; `lab/20261027-plan_rod_lab_morning_v3id1.json` 2
    more): on each seating's frame 34 to 37 actions keep the mount 1.5 to 3 mm from the rod; 17 presses bring an M2
    button head to the rod at the depth of their largest force (1 without the heads), so their forces would be shared
    with the head.
  The rod waits for the owner: its margins are about the size of its seatings' spread. The top-edge pass is still to be
  built. Path checks: `lab/20261027-paths_*_v3id1*.json` (`scripts/ur5_tip_calibration_holder_paths.py`: touch-offs to
  first contact, presses to their largest force's depth, turns left to the lab's simulator).

- R30 (draft, 18:01 on 10-27; owner: "the 4 n52 magnet fingertips are about ready ... the fingertip indenter (for
  testing crosstalk) is also ready, and i could put the other two 4 n52 magnet fingertips on it (from now on ill just call
  them v3, no ids yet). then we could run a super complete calibration suite + crosstalk analysis"). The plan is
  `docs/experiments/20261027-v3_tip_calibration_and_crosstalk/README.md`. One v3 tip runs v2 ID 1's sphere, rod and tall
  plans on its own registration (stages 1 to 4, about 15.6 h). The holder then gets two more v3 tips for a no-contact
  field map, with the stations empty and then mounted, and for pinch presses (stage 5, about 2 h). Nothing runs before
  the owner answers the README's six decisions. The workstation writes the four-magnet rest fit, the holder's plan
  builder and the `hover` action and `tip` indenter kind for the lab's simulator first.
  - 19:20 on 10-27: the owner stopped R29 at 19:00 for v3 ID 1, and the holder comes first. Done since 18:01: the
    four-magnet rest fit (fe04ef46), the hover action and held boards in the program (85ddc425, the lab), the cell with
    the held boards, the trimmed cap and the M2 heads (5049f2c5), the map's plan (d86bdebd: 3 touch-offs on the +y held
    apex, 58 facing and 31 side-by-side hovers, 20 repeats, about 44 min). Next: the lab homes on the holder with the
    owner's seatings and runs stages preflight, homing and pad in a folder seeded with the hybrid registration; the
    workstation rebuilds the hovers with --z-offset from the touch-offs and the homed registration. The suite after the
    map is cut to 9.1 h (README, "The suite, cut to 9.1 h"), with program changes in the lab first.
  - 19:23 on 10-27: d86bdebd is held. Its checks had the holder where the sphere fixture stood and no low sphere. The
    seatings put the holder behind the low sphere and the rod, and the rod blocks 64 of 89 hovers (wrist 3 0.7 to
    2.6 mm from it at the touch-offs). Without the rod: `lab/20261027-plan_holder_lab_map_norod_v3id1.json` (178b27b5,
    on holder_seated_2), 49 facing and 32 side-by-side hovers, 41 min, wrist 2 20.95 and HEX-E 21.6 mm at least. Open:
    the owner's choice (rod off, move the holder, or the 26 hovers that clear), and the seatings' 4.3 to 4.9 deg tilt.
  - 19:25 on 10-27: the owner reads the tilt as the 2x4's bow, so the lab homes on holder_seated_2 and simulates the
    no-rod plan. Fallback if the rod stays (78d65621): `lab/20261027-plan_holder_lab_map_rodin_v3id1.json`, no
    touch-offs, 21 facing hovers at W 3 mm and over and the far poses W 45 and g 40 mm, 20 repeats, about 18 min, wrist 3
    8.11 mm at least; no side-by-side map with the rod in.
  - 20:04 on 10-27: the owner moved the rod to a new home (rod_indenter_seated to _3, about 96 mm along -x_B), both
    held tips are on their blocks' top faces, and the +y tip is turned 180 deg about the vertical (connector toward +x_I):
    all in ur5_cb3_cell. The holder was reseated (holder_seated_4 to _6; homing on _5). Plan d527eb39,
    `lab/20261027-plan_holder_lab_map_seat5_v3id1.json`: 3 touch-offs on the +y apex, 49 facing hovers on +y and 37 on -y
    (W 2 mm and over), 20 repeats, about 43 min; no side-by-side map (no tip underneath). Wrist 3 17.2 mm at least; the
    paths (`scripts/ur5_tip_calibration_holder_paths.py`, the rod's three seatings) keep |sin q5| 0.63, q3 81 deg, the
    wrist 518 mm from joint 1's axis and 261 deg to any limit. Next: the touch-offs with the owner watching, then the
    hovers rebuilt with --z-offset.
  - 20:14 on 10-27: hold-touch-a and -b (session 200536-holdv3id1, held boards on d9:94) found no contact 1.2 mm past
    the expected +y apex. Likely cause: the seatings' frame (base z_B 76.3, 4.3 deg tilt; the tall indenter on the same
    2x4 at z_B 71.2) with the fork on the seat's walls; a level holder puts the +y apex 2.8 to 3.5 mm lower and 3 to 5 mm
    off across the tip, the -y apex about 7 mm lower. Pushed cabe4a31: `lab/20261027-plan_holder_lab_apexsearch_seat5_v3id1.json`,
    five touch-offs at the +y apex and 3 mm along and across, 6 mm past. Next: `scripts/ur5_tip_calibration_holder_apex_fit.py
    <session>` on their onsets gives the apex's true place; then -y touch-offs with a short reach, and the map rebuilt
    with --z-offset and the six watched hovers first (--first hov-f-W1-x+3,hov-f-W0.5-t+10,hov-f-W0.5-t-20,
    hov-fm-W2-t-20,hov-fm-W2-t+10,hov-fm-W15-t+10).
  - 20:45 on 10-27: the apex search (202426-holdapexv3id1, five contacts) put the +y apex 2.86 mm lower and 2.50 mm
    toward -y_I than the plan, in the program's frame (onsets +2.87 to +4.04 mm, fit rms 0.024 mm). The touches 3 mm along
    x came 0.04 and 0.07 mm deeper than the apex, so the fit is flat along x (`--flat-x`). A level frame through the
    apex, `lab/20261027-holder-apex-plusy/registration.json`, puts the holder's base at z_B 71.35 (the tall indenter's on
    the same 2x4: 71.2). It puts the -y apex 7.13 mm below the plan's; a holder tilted as seated would put it 4.27 mm
    higher. Pushed `lab/20261027-plan_holder_lab_map_apex_v3id1.json`, run like the apex search (--home-from
    lab/poses.jsonl:holder_seated_5, --held-board): five touch-offs on the -y apex, starting 9.5 mm above the level
    prediction and reaching 3 mm past it; 49 facing hovers over +y, hov-f-W1-x+3, hov-f-W0.5-t+10 and hov-f-W0.5-t-20
    first, then hov-fm-W15-t+10; 21 over -y at W 10 mm and over; 20 repeats; about 45 min. Paths
    (`lab/20261027-holder_paths_apex_level.json`, `_tilted.json`; each touch-off stops at its first contact, as in the
    program): 0 of 95 flagged with the holder level or tilted as seated; joint 5 at least 9.0 deg inside its window, the
    wrist 519 mm from joint 1's axis, mount 5.3, finger 4.1, HEX-E 24.5 mm. The -y onsets should read about 0 mm past
    the expected contact if the holder is level, about -4.3 mm (early) if tilted as seated. Next, on the -y onsets:
    `scripts/ur5_tip_calibration_holder_apex_fit.py <session> --flat-x --frame lab/20261027-holder-apex-plusy/registration.json
    --out lab/20261027-holder-apex-both`, then a second plan with the close -y hovers (W 0.5 to 5 mm, hov-fm-W2-t-20 and
    hov-fm-W2-t+10 first) on that frame.

- R23 (lapsed at 19:47 on 10-23: sph3t's registration was made on mount v3, and v5 replaces it. Today's finger, when it
  returns, gets its own touch-offs and registration on v5.) (16:34 on 10-23; the lab's request of 16:35, and the owner: no rehoming). The check of today's finger when it goes
  back on after the ATI work: `plan/20261023-plan_sphere_lab_check6_sph3t.json`, six of sph3t's 19 touch-offs at its
  5.0 mm standoff (pad-face-00, -08, pad-edge-00, -04, -06, pad-flank-07), renamed `chk-*` so that a resume on sph3t
  runs them. They were chosen for residuals within 40 um of sph3t's registration, joint 6 within 52 deg, and both sides
  of the pad. With sph3t's registration they clear the mount by 4.2 mm and the finger by 2.4 mm at least.
  - Resumed on sph3t's registration, each touch-off's printed onset should come within 0.1 mm of its `expect_onset_mm`
    (-0.038 to +0.028 mm: minus sph3t's residual at that point). Then sessions resume on sph3t's registration;
    otherwise the touch-offs and the registration are redone.
  - Along the finger the six see a shift of the link at about 0.3 of its size (their normals' n_z is 0.0 to 0.33). A
    0.3 mm slide of the link along the fork would move their onsets by about 0.1 mm.

The Writing section of `~/.claude/CLAUDE.md` and `~/.claude/writing_reference.md` went to the lab session in three
messages at 19:45; the lab confirmed them at 19:58.
