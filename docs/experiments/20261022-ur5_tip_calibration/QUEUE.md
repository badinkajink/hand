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

The Writing section of `~/.claude/CLAUDE.md` and `~/.claude/writing_reference.md` went to the lab session in three
messages at 19:45; the lab confirmed them at 19:58.
