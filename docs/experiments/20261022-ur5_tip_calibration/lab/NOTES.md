# Notes from the lab computer (only the lab computer writes this file; requests come in `../QUEUE.md`)

The cell as measured, the plan for it and the commands are in `README.md` beside this file. This file adds what the
program does not log, run by run.

## The cell on 2026-10-22, before the first run

- **Stack height** (flange face to the HEX-E's tool face). Not measured with calipers tonight. The plan keeps the data
  sheet's 35.0 mm. The owner's 2 mm checks bound the error to well under 1 mm: at the poses this height gives, the pad's
  contact point stood "About 2 mm, centred" above both the sphere and the rod.
- **Placement as set.** The 2x4 was turned end for end at 18:27, so the keys stand toward the robot. It is clamped at
  both ends on the cutting-mat table. The sphere's base is at (−138.5, −595.6, 70.8) mm, yaw −92.3 deg, and the rod's
  at (−289.5, −585.5, 70.3) mm, yaw −94.1 deg, from hand seatings on the keys at 18:33 and 18:40. The plan's placement
  is the sphere's.
- **The mount's turn** about the tool axis is −92.15 deg from the flange's x (palmar along the flange's −y), so the
  plan's T_EM carries it. The workstation's plans of the 20th to 22nd assumed 0 deg (x_M on tool0's x). A plan built
  without `--mount-turn=-92.15` fails homing's tilt check by about 90 deg on this robot.
- **Cables**, from the owner's photos of 18:2x. The HEX-E's cable leaves the sensor's top and hangs to the table
  beside the wrist. The board's USB cable runs from the QT Py on the dorsal side, down past the mount, with a loose
  loop. Neither was re-routed tonight; the dry run will show whether the turns pull on them.
- **The operator saw** (owner, by eye): both 2 mm gaps right and centred. While the robot stood still, nothing touched
  and no load came on: no HEX-E change over 1 N.
- **HEX-E.** It answered at 192.168.0.6. Its untared Fz drifted from 627.7 N (18:10) to 645.1 N (18:55) in one pose,
  likely warm-up after the box's power cycle at about 17:50. Every datagram's status word is 0x40000, the box's
  "overload warning".
- **Board.** Five chips stream at 369 records/s. The browser viewer's calibration at 18:13 found the poles + + / + / + +
  (tip end first), 0.34 of N52, the pad 0.24 mm off its seat, rms 0.48 mT (the next pattern 0.79).
- **Speed slider** at 30 % from 19:0x, set over RTDE.

## Runs

### 19:27, first (preflight, homing, gravity, mount) at 30 %: stopped in gravity when the HEX-E dropped out

Folder `20261022-192715-first/`: the session folder, the printout and two overhead frames.

- Preflight passed. The HEX-E answered at 100 Hz with filter 3. Its noise was 0.044, 0.016 and 0.230 N, and 0.0029,
  0.0038 and 0.0011 N m, with no timed-out or late reads. The board streamed 360 records/s and the robot 124 Hz.
- Homing from the sphere seating put the indenter at (−139.0, −595.6, 71.1) mm, yaw −92.3 deg, tilt 0.71 deg.
- Gravity: grav-00 to grav-03 ended ok. At 19:27:40.4 the stream stopped for 7.2 s while joint 6 turned through 69 deg
  at 6.5 deg/s. The DAQ's sample counter then restarted (13702, then 102), and the sensor came back untared at Fz
  645.5 N. grav-02 and grav-03 were read with that offset, so this sweep gives no fit.
- At 19:28:24.6 the stream stopped again, with joint 6 at 23.5 deg turning at −6.9 deg/s, for about 2 min. grav-04 to
  grav-06 aborted with no sample for 100 ms, and the program stopped after three aborts in a row (19:29:21).
- The DAQ box answered ping throughout, and the board's stream had no gap over 100 ms. The DAQ's status word read 0
  while the sensor was tared and 0x40000 after the restart, so the "overload warning" bit follows the untared offset.
- The robot was left at grav-06's orientation, the flange at (−133, −512, 365) mm, in normal mode. Joint 6 runs from
  −118 to +167 deg in this sweep. The owner was asked to check the HEX-E's cable, its two connectors and its slack for
  those turns.

### 19:43 to 19:48, the HEX-E's connector

With the arm still at grav-06's orientation, the stream stopped at 19:43:27.8, just after Fz stepped up 3.7 N. The owner
had touched the connector ("seems like its super sensitive!"). The box then listed no sensor (status 131072, a blank
serial number) until 19:47:21. For a few seconds the readings came back wrong (Fz 697 N, then 806 N) while the contact
was making. From 19:47:43 they held the old zero, 651.0 to 652.1 N, with no gaps. The owner tightened the connector and
tied the cable down "as tight and tied down as its gonna get".

### 19:54, first2 (preflight, homing, gravity, mount) at 30 %: gravity done, stopped by the owner in the mount stage

Folder `20261022-195420-first2/`. The arm was first taken home with `move_guarded.py` (0.02 mm from the target).

- Preflight passed: noise 0.039, 0.019 and 0.235 N; no timed-out reads; board 368 records/s.
- Gravity: all 12 orientations, no dropout, 236 s. Weight 1.567 N (160 g), centre of mass (−6.2, −2.6, 13.3) mm from
  the HEX-E's origin, the HEX-E's axes turned −0.61 deg about the tool axis (2.51 deg in all). The force residual is
  0.065 N against Table 2's 0.02 N, mostly along the sensor's z: ±0.10 to 0.20 N at grav-04, -05, -09 and -10. Leaving
  out any one orientation changes it by at most 17 mN. The fitted drift of Fz is 0.086 N/min.
- mount-0 aborted at 19:58:45: the answers stopped for 0.42 s. The sample counter advanced 410 over the gap, so the
  sensor kept sampling.
- mount-1's travel turned joint 5 from 140 to 173.6 deg, 6 deg from the wrist's singularity. The owner saw the wrist
  pressing the HEX-E's cable port ("the HEXE ft sensor gets squished by the wrist pitch joint") and pressed the
  emergency stop at 19:59:01.4 (safety mode 7 until 19:59:12). The HEX-E read at most 1.5 N up to the stop. The owner
  then moved the arm by hand. At 19:59:24.9 the HEX-E read (−14.3, −42.6, −5.8) N, and from 19:59:25.4 the box listed
  no sensor. The program, waiting at its prompt, was interrupted at 20:01.
- After the owner reseated the connector, the HEX-E read again at about 20:06. Its untared zero was 651.25 N, as before.
  Its noise over 15 s was 0.042, 0.041 and 0.242 N, and 0.0029, 0.0039 and 0.0011 N m, as in the table for filter 3.
- The arm went home with `move_guarded.py` at 20:10. The first try crashed in the robot interface (a core dump, nothing
  moved), probably on the control script that the interrupted program had left; the second reached home within 0.04 mm.

What let the travel through, found afterwards:
1. The plan builder solved its poses on its own IK branch (joint 1 −92 deg, joint 5 −90 deg). The robot sat on another
   (joint 1 59 deg, joint 5 62 deg at the seating), and the program solves its moves nearest the robot's joints. So the
   plan's checks described another arm. On the robot's branch, the mount stage's poses have joint 5 at −61 deg, on the
   flipped wrist, so reaching them crosses the singularity.
2. The builder's wrist limit was 7 deg from the singularity.
3. No check put the tool against the robot itself, and the cable port was not in the model. With the port as the owner
   measured it (25.4 mm by 7.9 mm, radial from the sensor's top at homing, then 19 mm toward the flange's −x), the model
   puts it 0.5 mm from the forearm at the pose of the stop.

The plan builder now takes the seating's joints (`--q-seed`), keeps joint 5 at least 30 deg from the singularity and
needs 15 mm between the tool (port included) and the forearm and wrist links (93e9c7db). The mount stage stays out of
tonight's runs (owner: "that straight pointing down orientation---not a fan of it").

### 20:25, sphere1 (preflight, homing, gravity, pad, register, grid; a segment of 12 grid presses) at 30 %: stopped in gravity when the HEX-E restarted

Folder `20261022-202551-sphere1/`. Plan `20261022-plan_sphere_lab_wrist_twist_segment1.json`: gravity, the 32 pad
touch-offs and the first 12 grid presses of the plan with the wrist limit, the joint 6 order and twisted touch-offs.
The owner asked for a short segment first ("just collect some initial data that we think is going to be good/safe, and
see if the whole pipeline from that point on works"). Options `--gravity-expect-N 1.567 --gravity-max-rms 0.2`.

- Preflight passed: noise 0.034, 0.016 and 0.245 N, no timed-out reads.
- Gravity: grav-00 to grav-12 in the new order ended ok, until the stream stopped at 20:27:40.1. The sample counter went
  from 18754 to 2 with a reading of +1862 N in Fz, then nothing. grav-07, grav-02 and grav-01 aborted with no sample, and
  the program stopped at 20:28:20. The arm stood near home with joint 6 at 92 deg.
- The same plan in the simulator (sim-final) stopped at pad-edge-06. Its touch-off twisted by −60 deg needs joint 5 at
  149 deg, which the program's window (30 to 100 deg) refused. The builder now keeps every pose inside the same window.
- Two simulator runs went on during this run, at up to 11 cores. The restart at 20:27:40 was the sensor's own (its
  counter went to 2 and the box lost it), but from now on no simulator or analysis runs during a robot run.

The sensor was back at 20:29:11. The workstation then advised dropping the twisted touch-offs (in five simulated
truths they helped in two and hurt in two), and the plan was rebuilt without them:
`20261022-plan_sphere_lab_window_untwisted.json` (148 actions, the mount stage dropped by the window, 0 travel paths
short). Its segment of 32 touch-offs, 6 pivots and 12 grid presses passed the simulator (sim-seg1: no aborts; pad touches
0.012 mm rms; joint 5 from 36.8 to 85.3 deg and the port at least 48.5 mm from the arm in every tenth sample).

### 20:46, tcp1 (preflight, homing, 3 grid presses, 8 twists) at 30 %: stopped at preflight, no HEX-E

Folder `20261022-204657-tcp1/`. The owner asked for presses and twists to judge the TCP by eye ("can you just do some
presses and twists so i can tell you whether the TCP is correct or not"), with no gravity sweep ("the utility we got from
these big joint movements in this gravity tests are SO MARGINAL"). The HEX-E gave no samples (42 timed-out reads), and
the box listed no sensor. The arm had not moved since 20:28:20, so the connection failed with nothing turning. Nothing
moved in this run.

### 20:51, tcp2 (preflight, homing, 3 grid presses, 8 twists) at 30 %: three presses, then the first twist hit the fixture's key with the mount's fork

Folder `20261022-205059-tcp2/`. Plan `20261022-plan_sphere_lab_window_untwisted_segment_twist.json`, the nominal TCP, the
HEX-E's factory calibration (no gravity stage). The HEX-E had come back at 20:48:19 and held through the run (0 timed-out
reads).

- ref-00, grid-052 and grid-053 pressed to 2 N and back, 1.16 to 1.61 mm past their rest poses, 11.5 to 14.3 s each:
  the first presses of the pad on the sphere.
- twist-0-1.0N turns the pressed pad about the vertical through the contact by +5, +10, +20, then −5, −10 and −20 deg.
  The +20 deg half ran with the load near its press value. On the way to −20 deg, at about −13 deg (joint 5 at 73 deg),
  Fx went from +1.7 to −8.3 N in 4 s and Ty to −0.31 N m; the action aborted at 10 N (20:57:19). In the cell model the
  mount's fork came within 0.1 mm of the indenter at that pose (8.1 mm at t+200 s, 1.0 at t+236, 0.1 at t+241.5). The
  builder checks a twist's press pose only, and the simulator models only the pad's contact, so neither saw it.
- twist-0-2.5N had begun its turn when the program was interrupted (20:58:37). The pad was lifted 10 mm straight off the
  sphere with `move_guarded.py` (HEX-E change at most 0.83 N) and the arm went home. The overhead frame at 21:01 shows
  both fixtures as before.
- The turn's own load, measured from the pressed state by the workstation (1 s means), lies mostly along the sensor's z,
  which at twist 0 runs along the finger: Fz rose 0.32, 0.52 and 0.74 N at +5, +10 and +20 deg (0.29, 0.47 and 0.63 N
  less the HEX-E's creep). The Fy of 1.1 N and Tx of −0.08 N m seen at the first step belong to the press to 1 N itself.
  The robot turned exactly about a line within 0.1 to 0.3 mm of the homed sphere's top, so the pad's contact lay off the
  turn's axis across the finger: 0.25 mm by the FE model's contact stiffness, about 1 mm by the stiffness the presses
  show. That is the TCP check the owner asked for.
- The three presses met the sphere 0.72, 0.83 and 0.76 mm before the planned contact.

No twists or rolls run again until the builder checks every pose along their turns.

### 21:08 to 23:27, sphere2 (pad, register, pivot, grid, then the checked turns) at 30 %, in three parts

Folder `20261022-210852-sphere2/` (one session, resumed twice; `printout.txt`, `printout_resume.txt`,
`printout_grid.txt`; the plans of the resumes as `plan_resume_2131.json` and `plan_resume_2152.json`). No gravity
stage: the HEX-E's factory calibration, first2's fit for the analysis (QUEUE R11). The owner left at about 20:50
("i prioritize getting a segment of fingertip calibration data").

- 21:08, `sphere2` on `20261022-plan_sphere_lab_window_untwisted_segment1.json`: 20 touch-offs with onsets −1.19 to
  +0.17 mm from the expected. Eight aborted with no HEX-E sample for 100 ms; three in a row stopped it at 21:30. The
  pauses (0.05 to 0.36 s, the sample counter running on) came from the reader's own fsync of ft.bin on the WD Green
  SSD, with the overhead time-lapse writing JPEGs to the same disk (stopped at 21:19). Fixed in d1ef3216.
- 21:32, `sphere2r` (resume) on `plan/20261022-plan_sphere_lab_turns_checked.json`, the fixed program: 9 more
  touch-offs; three aborted at 21:42 to 21:44 on the box's own pauses (1.4, 2.8 and 1.7 s, 118 timed-out reads, no
  counter restart: the link of R10). The registration at 21:44 fitted 29 touches at 0.057 mm rms (Table 2: under 0.05),
  moved the TCP 2.01 mm and 2.81 deg and the sphere's centre 1.22 mm. All six pivots held 1.5 and 3.0 N; then the
  program crashed in the pivot check (the reference pivot's samples had left the 180 s rings; fixed in 8be552e1).
- With this registration the plan's least clearances fell under their margins (`scripts/ur5_tip_calibration_check.py
  --clearance`): the finger 0.57 mm at grid-050 (needs 0.8) and the mount 2.27 to 2.62 mm in three slides (needs 3).
  Those four were dropped (`lab/20261022-plan_sphere_lab_turns_checked_reg2144.json`, 138 actions); the workstation
  agreed. grid-025, -027 and -028 stay at 2.75 to 2.98 mm for the mount.
- 21:51, `sphere2g` (resume) on that plan, stages grid, shear, roll, twist: by 22:02, 12 of 73 grid presses, no abort,
  and no HEX-E gap over 31 ms. At 22:02:19 the reader paused 0.36 s (counter running on) and grid-007 aborted: that was
  the copy and LFS upload of the 160 MB session folder at the same moment. Its dirty pages throttle every writer, the
  reader's own write of ft.bin included. During a run only the small files are copied and pushed now; the reader should
  hand its rows to a writer thread and never write a file itself.
- sphere2g went on to exit 0 at 23:27:03. The grid ended 72 of 73 (66 grid presses and 6 reference presses; grid-007
  the one abort). roll-0 and roll-3 at 1.5 N and all eight twists at 1 and 2.5 N ended ok. The HEX-E's force moved at
  most 2.8 N from its tare, the press included, and joint 5 kept between 38 and 65 deg. The HEX-E's stream in sphere2g:
  573,387 samples, no timed-out read, no gap over 0.5 s, and none over 50 ms after 22:02:32.
- The six reference presses (`lab/track_refs.py`), from ref-00 (21:52, onset 1.709 mm, depth 0.709 mm): onset +0.066,
  +0.047, +0.016, +0.035 and +0.048 mm; depth +0.013, +0.052, +0.021, +0.026 and +0.025 mm (ref-05 at 22:56). All well
  inside the stop rule (0.3 and 0.1 mm), so neither the pad nor its clamp crept over 65 min.
- Four flank presses, grid-061, -065, -066 and -067 (22:53 to 22:55, y_P about −6.7 mm, joint 6 at 70 to 72 deg, the
  most turned wrist of the grid), read 0.34 to 0.42 N on the HEX-E before the pad met the sphere. The load rose with
  the travel from the rest pose. The board moved only 2.1 to 4.7 µT there, as in flank presses without that load
  (1.7 to 3.3 µT), against 9 to 12 µT for 0.4 N on the pad. So the load went through the HEX-E's cable or another part of
  the tool. The workstation's FE comparison had flagged the same four, and its analysis now leaves them out (67264280).
- The arm went home at 23:29 with `move_guarded.py` (each waypoint within 0.10 mm; the HEX-E changed at most 0.83 N).
  The whole session folder is pushed as bf3aa417 (442 MB of LFS). The board's host times step back 42 times in 3.0 M
  records, at most 28 ms; the workstation's analysis selects by time masks, which these steps do not upset.
- The page of the night: `docs/experiments/20261022-ur5_tip_calibration/20261022-ur5_tip_calibration_lab_runs.html`
  (97caa02a), https://claude.ai/artifact/1yHnNnGGE7Aq7VMZQmLvdw. This clone has no TeX, so the workstation rendered the
  page's formulas into `texsvg_cache.json` (b404e9b8). About 8,000 docs files here are still LFS pointers, so
  `docs_index` must not run here: its pages.json came out 13 kB short. The workstation runs it.

## State at 23:58 on 2026-10-22 (for a session after a compaction)

- Nothing runs on the robot. The arm is home, and the program and `move_guarded.py` are not running.
  `log_joint_temps.py` still writes `logs/20261022-ur5_joint_temps.jsonl` every 30 s until about 02:10 (reads only).
- No further robot run tonight unless the owner asks. The segment the owner asked for is in. What remains needs someone
  at the bench: R10 (the HEX-E's connector, flex test), R16 (whether the DIP link moves in the fork), and a slow repeat
  of grid-066's approach while someone watches the cable and the tool's parts near the sphere.
- A later resume of the session could retry grid-007 and pad-flank-04, -08 and -09 with the current program (queued
  writer 96abeec0, pivot fix 8be552e1). The gravity, mount and final stages stay out until the owner decides.
- The owner's answers, just after midnight on 2026-10-23. R10: the HEX-E can't be fixed ("it is what it is"). A new
  cable may be ordered but will take a while, so runs live with the intermittent link and its guards. R16: the DIP link
  moves a little in the fork when the pad is pressed hard. A mount that fastens the link more rigidly was requested from
  the workstation. The owner called the day done.
- 00:39, the workstation's answer (R18, 0c2ea786): mount v4, designed and not printed. It is v3 plus a stop plate
  behind the DIP link's tongue: three brass M3 set screws in heat-set inserts, turned onto the tongue's back after the
  link is clamped and locked with jam nuts. A face press now turns the link 0.05 deg instead of 1.0 deg. Only the fork is
  reprinted (1 h 46 min, 64.9 g); the owner decides when. Page:
  `docs/experiments/20261023-ur5_calibration_mount_v4/20261023-ur5_calibration_mount_v4.html`. Before v4 runs here: the
  payload goes to 0.403 kg, its centre about 2 mm further out. Pull the v4 CAD STLs from LFS. Check every plan with
  `UR5_MOUNT=v4` for `ur5_cb3_cell.py`. Then repeat the pad touch-offs and the registration, a reference press and a
  short grid.
- 00:2x, the owner asked for a study of the board's drift. It is `scripts/ur5_tip_calibration_drift.py` on this
  session, run here.

## 2026-10-23 morning: the rod plan (R19), checked here before any run

- The owner asked through the workstation at about 07:50: "did we also try cylinder presses? can we do that if we are
  feeling good about it?". The workstation's plan is `plan/20261023-plan_rod_lab_reg.json`. It has 99 actions on the
  rod with the sphere as an obstacle, seeded from the rod's seating of 18:40 (`right_indenter_seated`).
- In the lab's simulator the whole plan except gravity ran with no abort in 15 min
  (`logs/20261023-tip_calibration/081321-simrod`, printout `logs/20261023-sim-rod.log`). The registration fitted 22
  touches at 0.018 mm rms.
- `lab/check_arm_samples.py --cell` now models the session's own indenter, not always a sphere. Joint 5 stayed between
  40.3 and 57.6 deg and the port at least 48.5 mm from the arm. In the cell the HEX-E kept 31.7 mm, the mount 3.3 mm
  and the finger 2.0 mm. No sample broke a rule.
- At 08:13 the robot was running in normal safety mode at home, with the slider at 0.30. The HEX-E answered at 100 Hz
  with no gap in 20 s and still held last night's tare (Fz −1.1 N), so it had not restarted overnight.
- `lab/tipcal_run.sh` takes the seating from `SEAT=...` (the rod: `poses.jsonl:right_indenter_seated`).
- The run waits for the owner's direct go. It is staged: preflight, homing, pad and register first. Then
  `scripts/ur5_tip_calibration_check.py SESSION --clearance PLAN` with the new registration, dropping what falls under
  the margins. Then the pivot, grid, shear and roll stages.

### 10:45 to about 12:15 on 2026-10-23, rod1 (the rod, R19) at 30 %, in two parts, with the owner's go at 10:4x

The owner, at the desk at about 10:40: "if things are looking good, then yea, go ahead with the rod tests". At 10:45
the robot was at home in normal safety mode with the slider at 0.30. The HEX-E answered with no gap, and nothing else
held the robot or the DAQ.

- 10:45, `rod1` on `plan/20261023-plan_rod_lab_reg.json`, stages preflight, homing, pad, register (`SEAT=...:
  right_indenter_seated`). Homing from the rod's seating of 18:40 put the rod at (−289.0, −584.6, 69.0) mm, yaw
  −94.1 deg, tilt 1.43 deg. All 22 touch-offs ended ok, with onsets −1.09 to −2.02 mm from the expected (median
  −1.52, spread 0.21 mm). The plan has the nominal tool (T_MP translation (2.05, 0, 82.5) mm), not last night's
  registered one, which put the pad 2.45 mm and 2.7 deg off in its seat. So the pad met the rod early, and at the
  touch-offs near −2 mm it touched already at the rest pose (an onset extrapolated before the start). Every approach
  stayed under 1 N. The HEX-E had 60,509 samples, no timeout and no gap. Exit 0 at 10:55.
- The registration fitted the 22 touches at 0.122 mm rms, against 0.05 mm asked and 0.057 mm on the sphere last
  night. The largest residuals were pad-face-07 −292, pad-edge-07 +248, pad-face-01 +199 and pad-edge-03 +174 µm.
  It moved the TCP 1.74 mm and 3.88 deg (last night 2.01 mm and 2.81 deg) and the rod 1.22 mm. Likely causes: the
  rod's seated tilt of 1.43 deg, which a centre-only fit absorbs into the TCP, and the touches that began in contact.
  For the analysis, a residual of a tenth of a millimetre is small against the estimator's 2 mm.
- `scripts/ur5_tip_calibration_check.py --clearance` (it now also lists every action under a margin) with this
  registration: mount least 2.81 mm. Three actions fell under 3 mm: grid-006 2.94, grid-029 2.85 and shear-3-1N
  2.81 mm. Finger least 1.69 mm (needs 0.8), HEX-E 32.9 mm (needs 6). The three were dropped
  (`lab/20261023-plan_rod_lab_reg_rod1.json`, 96 actions); the next closest are grid-051 3.18 and grid-023 3.19 mm.
- 10:57, `rod1g` (resume) on that plan, stages preflight, pivot, grid, roll; printout
  `logs/20261023-105751-tipcal-rod1g.log`. Preflight passed (noise 0.022, 0.033, 0.235 N, no timeout). The pivot
  stage has a reference and four offsets but no twisted press, so `pivots.json` is empty and the fit is in the events.
  The reference presses held: ref-01 (11:11) moved the onset +0.009 mm and the depth +0.002 mm from ref-00 (0.1 N
  at 1.152 mm, depth 0.429 mm); ref-02 (11:23) −0.002 and +0.018 mm.
- 11:16, the workstation (R19): rod1's touch-offs began in contact, so two touch-off sessions after rod1g, `rod1t`
  on the rod and `sph3t` on the sphere, for a joint registration. `lab/descent_check.py` measured the contact from the
  last vertical descent onto each rest pose, which keeps the orientation. The preload is the force at rest against the
  force 6 to 10 mm higher. On sphere2's 29 clean touches it was 0.056 N median (0.169 N at most). On rod1's 20
  touches with a descent it was 1.85 N median and 3.80 N at most; 16 had over 0.3 N, with contact 0.09 to 1.16 mm
  above the rest. That is a true onset of −1.59 to −2.66 mm from the expected (face-02 −2.66, face-07 −2.62). So the
  pad bore up to about 4.3 N at a touch's stop, not 0.5 N. Past the workstation's 2.5 mm, both sessions use its
  5.0 mm plans (`plan/20261023-plan_rod_lab_touch50.json`, `plan/20261023-plan_sphere_lab_touch50.json`, 2fd80414),
  which differ from their sources only in the touch-offs' standoff.
- In the simulator the rod at 3.5 mm ended 22 touches with no abort (0.018 mm rms). The sphere at 3.5 mm aborted
  three in a row with no contact onset. The default truth tilts the plywood 0.23 and 0.17 deg about the base and
  shifts the indenter by (1.5, −2.0, 0.8) mm. That puts the sphere's top 2.33 mm below where the program's homing has
  it and 4.4 mm to the side, which on a sphere of 5 mm radius makes every contact come several millimetres late, past
  the 1.2 mm a touch may go beyond the expected contact. (The rod's top line comes out 0.05 mm high: its plan's
  `sim.T_BI` sits 2.45 mm above the seating and cancels the tilt.) Last night's real onsets on the sphere were −1.19 to
  +0.89 mm. The 5.0 mm plans were simulated with the indenters where the
  stored seatings put them (sim-only copies with `sim.T_BI` from the seating, `--sim-truth` with no tilt or indenter
  error), in `/dev/shm/tipcal-sim`. Both ended with no abort (rod 22 touches, 0.012 mm rms; sphere 32, 0.020 mm).
  `check_arm_samples.py --cell` passed the rod (mount ≥ 3.4, finger ≥ 2.6 mm). On the sphere it found the fork
  2.55 mm from the sphere at face-02 with the simulator's tool; on last night's real motions, with sphere2's
  registered tool, the fork kept ≥ 3.31 mm.
- rod1g ended at 11:47:53 with exit 0: the 51 grid actions and roll-0 and roll-3, no abort. The HEX-E had 299,532
  samples, no timeout and no gap. The reference presses, as onset and depth change from ref-00: ref-01 +0.009/+0.002,
  ref-02 −0.002/+0.018, ref-03 −0.003/+0.023, ref-04 (11:40) −0.046/+0.049 mm. All are inside the stop rule, but the
  depth grew steadily, to half the rule's 0.1 mm. Home through `move_guarded.py` at 11:49. The whole session folder
  is in `lab/20261023-104526-rod1/` (190 MB, with `printout_resume.txt` and `plan_resume_1057.json`).
- 11:52, `rod1t` on `plan/20261023-plan_rod_lab_touch50.json`, stages preflight, homing, pad, register. The 30 %
  slider scales the approach too: 0.045 mm/s, so each touch-off took 81 to 107 s. All 22 ended ok, with clean onsets
  of −1.34 to −2.48 mm from the expected; the descent check had predicted rod1's to within 0.05 mm on most. The
  registration fitted them at 0.201 mm rms (rod1: 0.122 mm), moved the TCP 1.28 mm and 4.95 deg and the rod's centre
  4.05 mm. The program fits the rod as a sphere at one point; the workstation's cylinder fit should judge this
  registration. The HEX-E had one gap of 0.6 s (Fz +0.1 N, not a restart). Exit 0 at 12:30; home at 12:32.
- The workstation chose 3.5 mm for the sphere if every touch-off's predicted onset was −2.5 mm or later. Its check
  (`scripts/ur5_tip_calibration_check.py --clearance PLAN --clearance-stages pad --home-from SEAT`, 9a5ce959) plans each
  touch-off with the nominal tool and finds where the registered pad meets the indenter. With rod1t's registration the
  onsets came out −3.20 to −1.55 mm, so sph3t takes the 5.0 mm plan. The three tools disagree. Onset medians are −1.87
  with rod1t's, −1.40 with rod1's and −0.40 with sphere2's, and sphere2's −1.10 to +0.90 matches last night's real
  onsets. The touch-offs under the margins 1 mm past the contact differ with each tool. All eight in the union are
  dropped: edge-01, face-02, face-04, face-05 (mount 2.43 to 2.99 mm) and flank-03, edge-02, flank-01, flank-02
  (finger 0.57 to 0.77 mm). The result is `lab/20261023-plan_sphere_lab_touch50_sph3t.json`, with 24 touch-offs.
  The workstation (12:41) traced rod1t's poor fit to edge-01 alone (+659 µm). Without it the fit is 40.4 µm rms over
  21, with the rod's centre 1.17 mm from homing. No modelled part was near the rod at edge-01's onset, and its joint 6
  was the largest, +45.5 deg, so the HEX-E's cable is the likely cause.
- So the five flanks with joint 6 above +60 deg were dropped as well: flank-04 (+72.1), flank-08 (+71.8), flank-06
  (+66.2), flank-09 (+65.0) and flank-00 (+64.0). Last night flank-04, -08 and -09 were the pad stage's three aborts
  with no HEX-E sample, and flank-06 and -00 gave the latest onsets. They were the plan's last five, so the 19 left
  are the first 19 of the 24 simulated: no abort, 0.011 mm rms, and the cell check passed (joint 5 47.5 to 80.0 deg,
  mount ≥ 3.2, finger ≥ 1.9 mm).
- 12:40, `sph3t` on that plan, stages preflight, homing, pad, register (`SEAT=...:left_indenter_seated`). All 19
  ended ok with no abort, about 2.1 min each. The onsets repeat last night's sphere2 to within 0.06 mm on most: flank-05
  −0.760 (−0.814), edge-03 −1.184 (−1.186), face-00 −0.682 (−0.677), edge-07 +0.559 (+0.498); the largest change was
  edge-05's 0.157 mm. On the sphere, then, the tool has not moved since last night. The registration fitted the 19 at
  0.043 mm rms (sphere2: 0.057 mm over 29), moved the TCP 1.62 mm and 1.93 deg, and the sphere's centre 1.43 mm. The
  HEX-E had 259,372 samples, no timeout and no gap. Exit 0 at 13:24; home at 13:26 (the HEX-E changed 1.83 N on the
  way, the payload turning with joint 6 from 55 to 6 deg). The session folder is `lab/20261023-124052-sph3t/`.
- The rod's level along its length was never checked. Its pose comes from one hand seating of the fork in its key
  (18:40, tilt 1.43 deg), and all touch-offs lie within 0.9 to 2.6 mm along the 20 mm rod. The workstation fits the
  tool to sph3t and sphere2 together and gives the rod its own position and two axis angles.

### 15:36 to 16:04 on 2026-10-23, sph3g (R21): half of last night's grid again, a day later, on sph3t's registration

The owner's go came at 15:18 in the workstation's chat ("we can run the 40 minute test then"). They confirmed it in
the lab session ("go") at about 15:36.

- The plan is `lab/20261023-plan_sphere_lab_r21_dayrepeat.json`. It holds the even-numbered grid presses of sphere2's
  `plan_resume_2152.json`, in last night's order and at their force levels, with joint 6 from −64.8 to +57.4 deg. Left
  out are grid-060, -062 and -064 (joint 6 above +60 deg) and grid-066. The plan has no grid-050.
  `scripts/ur5_tip_calibration_check.py --clearance` with sph3t's registration then dropped grid-028 (mount 2.34 mm) and
  grid-002 (2.95 mm). That leaves 27 presses, with ref-00, ref-01 and ref-02 at the start, the middle and the end.
- The simulator resumed a copy of sph3t's small files, with its truth set to sph3t's registered tool and sphere. All 30
  actions ran with no abort, and `check_arm_samples.py --cell` passed (joint 5 48.3 to 85.1 deg, mount ≥ 3.3, finger
  ≥ 2.1, HEX-E ≥ 31.6 mm).
- 15:36:30, `sph3g`: `--resume logs/20261023-tip_calibration/124052-sph3t --stages preflight,grid`. All 30 ended with
  no abort, and the HEX-E had 161,858 samples, no timeout and no gap. ref-00 (15:37) passed 0.1 N at 1.545 mm and
  2 N at 2.196 mm, a depth of 0.651 mm. ref-01 (15:51) moved the onset +0.077 mm and the depth −0.004 mm; ref-02
  (16:03) moved them +0.037 and −0.040 mm. Exit 0 at 16:03:34; home at 16:05. The data are in
  `lab/20261023-124052-sph3t/` with `printout_resume.txt` and `plan_resume_1536.json`.
- 16:30, the owner: "its time to calibrate the ati can i take off the finger". The robot stood at home in normal
  safety mode, with nothing from the lab session connected to it or to the DAQ, so yes. The owner (16:3x): the
  geometry does not change, and rehoming on the keys might do more harm than good. Agreed: the indenters and the robot
  stay, and the registration's sphere is better placed than a hand seating, which it moved by 1.43 mm. What can shift
  is the seat of each part taken off and put back: the link in the fork, and the mount on the HEX-E if it comes off.
  So when the finger is back, about six sphere touch-offs at 5.0 mm (some 12 min) are compared with sph3t's onsets,
  which repeated last night's to within 0.06 mm. Within about 0.1 mm, the sessions resume on sph3t's registration;
  otherwise the touch-offs are redone. No rehoming unless an indenter moves.
- 16:5x to 18:00, the ATI Mini45's Net box (NETBA). It is on PoE from a Netgear GS305PP switch, which plugs into port 4
  of the lab's router. The box never appeared on the network: 192.168.0.x showed only the router, UR5, corvus and the
  HEX-E box; nothing answered at 192.168.1.1 or anywhere in 192.168.1.x, 169.254.x.x or IPv6; and the owner's sudo
  tcpdump saw no frame from ATI's MAC prefix 00:16:BD. I read the switch's yellow light as a PoE fault, which was
  wrong. At 18:1x the workstation found the box, powered, at a static 192.168.123.50/24 (MAC 00:16:BD:00:2D:E4,
  firmware F/T BNI v2.2.59). It used ATI's RTA discovery: a broadcast to UDP 255.255.255.255:51000, with the reply on
  multicast 224.0.5.128:28250 (`~/Programs/ati_ft_sensor/demos/ati_gui/netft_gui.py` on the workstation). The yellow
  light was link traffic. None of the scans had covered 192.168.123.x. For a box at an unknown address, use the
  discovery broadcast. ATI's manual: DHCP on by default, with 192.168.1.1 as the fallback after 30 s; PoE IEEE 802.3af
  class 0, Mode A only, 6 W at most; or 11 to 25 V DC on the M12 Pwr/CAN connector. To reach 192.168.1.x without
  losing the internet, corvus takes a live extra address, `nmcli device modify enp3s0 +ipv4.addresses
  192.168.1.100/24` (no sudo; undone with `-ipv4.addresses`). Never set the connection's IPv4 method to Manual: that
  drops the DHCP address and the internet with it.
- 18:29, after the owner put the box back on the lab router, `scripts/ati_netft_discover.py` found it at
  192.168.123.50. corvus held 192.168.123.100/24 live. Ping took 0.4 ms, the web page answered, and one RDT sample
  came back. But the box reports status 0x80020000: an error, with bit 17 set ("Transducer Saturation or A/D operation
  error"). Strain gauge 5 reads 32767, the top of the A/D range, in every sample; the other five read 354, −5035,
  1823, 1917 and 701. With no tare, the forces read 13.4, 26.4 and 273.5 N and the torques −2.12, 1.39 and 0.16 N m.
  The box's calibration is FT35796, SI-125-3, rated 125, 125 and 500 N and 3, 3 and 3 N m. The owner was asked to check
  the transducer cable and the mount for a load. Nothing was tared or changed on the box.
- 18:3x to 18:4x, gauge 5. The owner had seen the same error on the workstation, so it predates the remount.
  - Loosening the mount's screws moved gauges 2, 4 and 6 by 270 to 400 counts but not gauge 5. A power cycle with the
    cables reseated changed nothing; during its start-up the box briefly showed 0x88008010 with every gauge at 65535.
  - Hand presses from 18:37 to 18:40 (519 reads) moved gauge 1 over 1935 counts and gauge 3 over 1235, while gauge 5
    stayed at 32767.
  - The owner: the box's calibration is an Axia80's, and the Mini45's own (US-120-160) is not available. ATI's
    calibration is too expensive.
  - The box holds one calibration (slot #1, FT35796 SI-125-3; #2 to #16 empty). Its per-gauge gains `calggn` (0x5B,
    0x5B, 0x55, 0x5D, 0x57, 0x59) and hardware offsets `calgof` (0x7D46, 0x8088, 0x7B69, 0x7CBC, 0x7E11, 0x81E8) are
    readable in `netftcalapi.xml`. None of the box's interfaces writes them (setting, config, comm, moncon, rundata
    .cgi; ATI's Java tool only reads them). `setbias<n>` is a software bias applied after the A/D, so it cannot recover
    a clipped gauge.
  - Asked of the owner: a cable wiggle test while corvus watches gauge 5, a resistance check of gauge 5's bridge at
    the connector, and July's logs, whose RDT status words would show whether gauge 5 was already stuck. The Fz row of
    the box's matrix weights gauges 1, 3 and 5 equally, so Fz follows a Z press at about 2/3 scale without gauge 5.
  - Fallback put to the workstation: with the load point known (the indenter), a press has 3 unknowns. Five gauges
    fitted to the weighed masses can solve them.

## State at 19:00 on 2026-10-23 (for a session after a compaction)

- The robot stands at home, idle. All of today's sessions are pushed whole: rod1 (and rod1g), rod1t, and sph3t (with
  R21's sph3g appended).
- The ATI work has stopped. Its Net box has a dead gauge-5 input: both the Mini45 and the Mini40 read gauge 5 at +32767
  under the default settings. The notes are in `../../20261023-ati_ft_calibration/lab/NOTES.md`. corvus still holds
  the live 192.168.123.100/24, which is harmless and goes on a reboot.
- Next is R22, the first checkerboard tip. The owner takes the ATI out, puts the mount back on the HEX-E as before
  (−92.15 deg), fits the tip and plugs in its board. Nothing runs before the owner's go in the lab session.
- The lab's steps then:
  1. Pull R22's plans, and build the run plan with sph3t's drops (8 under the margins as run, and the 5 flanks with
     joint 6 above +60 deg).
  2. Simulate it with the seated truth, and check the samples in the cell.
  3. Check the robot, the HEX-E and the board.
  4. Run the plain touch-offs at 5.0 mm, with no rehoming.
  5. Then the as-run clearance check of the offset plan, the offset touch-offs, and the grid.

## 2026-10-23 evening: R22 on mount v5, the first checkerboard tip

### 19:48 to 20:10, pass 1 prepared (nothing moved)

- 19:48, pulled 907edf13. The owner puts mount v5 on the robot with the first checkerboard tip. Both R22 plans set v5's
  payload, 0.3993 kg with its centre 34.3 mm from the flange. Pass 1 adds the gravity sweep, which should weigh about
  1.72 N. v5's CAD came with `git lfs pull --include "docs/experiments/20261023-ur5_calibration_mount_v5/cad/*"`.
- Pass 1's plan is `lab/20261023-plan_sphere_lab_r22_pass1.json`, for the stages preflight, homing, gravity, pad and
  register.
  - Touch-offs: sph3t's 19. The plain plan's touch-offs are identical to sph3t's source, so the same 13 are dropped: 8
    under the margins with sphere2's registration, and 5 flanks with joint 6 above +60 deg.
  - Gravity: 9 of the 12 orientations. grav-07 (+100.8 deg), grav-02 (+152.2) and grav-01 (+92.2) are out by the same
    joint 6 rule. On 10-22 the HEX-E dropped out in two of three sweeps, both at large positive joint 6. The 9 run
    joint 6 from −147.9 to +48.8 deg. The fit stays well posed: its design [g, 1] has a condition number of 2.9, against
    2.2 with all 12. The workstation agreed at 20:03; its analysis needs only the weight and the 3 × 3 map.
- Clearance of the touch-offs with `UR5_MOUNT=v5`, placed from the stored seating as a fresh session runs them
  (`--clearance-stages pad --home-from ...:left_indenter_seated`):
  - with sphere2's tool, the mount keeps at least 3.12 mm, the finger 1.75 mm and the wedge 11.26 mm;
  - with sph3t's tool, pad-face-11 comes 2.95 mm from the mount, 0.05 mm inside the margin. It is the same with v3, and
    face-11 ran clean in sph3t, so it stays.
- The simulator ran pass 1 from 19:55 to 19:59 with no abort. It had the sphere where the stored seating puts it, the
  payload at 1.72 N, and `--gravity-expect-N 1.72`. The sweep weighed 1.725 N with a residual of 0.0064 N, and the 19
  touch-offs fitted at 0.012 mm rms. The cell check passed:
  - joint 5 from 57.5 to 90.5 deg in the sweep and from 47.5 to 80.0 deg at the touch-offs;
  - joint 6 from −148.4 to +48.3 deg in the sweep, and at most +54.8 deg at the touch-offs;
  - the mount at least 3.3 mm from the cell, the finger 2.3 mm and the wedge 11.7 mm; the wrist links at least 11 mm.
- `check_arm_samples.py` had left the wedge out: its list of parts predates v5. The wedge now counts on the mount's
  margin, and the script stops on any part of the tool it does not place in a group.
- Pass 2's drops, agreed with the workstation: fgrav-01 (+92.2 deg) from the final stage, and grid-060 to -067 (joint 6
  from +64.0 to +72.1 deg, the workstation's four among them). In the workstation's simulation the offset touch-offs
  end with joint 6 from −57.2 to +51.6 deg. Pass 2's simulation with the new registration gives the real values.
- Still to come: the owner's go and the tip's name. The board was not on USB at 19:5x.

### 20:20 to 21:28, cb1t: pass 1 on mount v3 after a cable fix

- 20:2x, the owner's go. The owner put back the fork that ran on 10-22 and in sph3t (mount v3; v5's wedge did not work)
  and fitted the first checkerboard tip. The gravity sweep was dropped at the owner's request ("that's how we knocked
  out the hex e sensor last time"). The workstation (20:23) analyses cb1t with first2's gravity fit: with no rolls or
  twists, every hold is compared with a rest at the same orientation. The plan got v3's payload, 0.3839 kg at 32.5 mm.
- The HEX-E sent no samples (120 timed-out reads in 6 s) until the owner replugged it at 20:25:40; its untared Fz was
  then 645 N, as always. The tip's board, a new QT Py (b4:3a:45:b0:d9:94), came on USB at 20:25:50. Its first 5 s read
  had corrupt records just after it was plugged in; the next read, 1867 records, had none.
- 20:31, the first attempt (folder `20261023-203123-cb1t/`): pad-flank-05 and pad-edge-06 aborted with no contact onset.
  - In flank-05 the force rose 0.36 N over 3 mm of free approach, then jumped at 4.2 mm.
  - In edge-06 it jumped 0.5 N after 0.54 mm, with the sphere still about 4 mm away.
  - In sph3t the same approaches stayed within 0.15 N until contact.
  - I stopped the program at 20:34:35, before a third. The owner found the board's USB cable catching on something.
- 20:39, a diagnostic of the same two touch-offs (`lab/20261023-plan_sphere_lab_r22_diag2.json`) started the session
  `20261023-203938-cb1t/`. Both were clean: flank-05 −0.366 mm (fit 82 mN) and edge-06 −0.411 mm (67 mN), against
  sph3t's −0.760 and −0.910 mm.
- 20:45, the guarded move home stopped at the HEX-E's 3 N limit (3.11 N) while the owner taped the cable. My command
  started the resume regardless, and I stopped it in preflight before it moved; its HEX-E noise was 20 times normal
  while the owner worked. From now on the resume starts only after the move prints "home: reached".
- 20:46:47 to 21:26:56, the resume on the pass 1 plan ran the other 17 touch-offs with no abort.
  - The onsets ran from −0.758 to +0.475 mm, 0.0 to 0.5 mm later than sph3t's, from the new tip and the refitted fork.
  - The fits were 30 to 129 mN.
  - The registration fitted the 19 at 0.052 mm rms. It moved the TCP 1.568 mm and 1.318 deg, and the sphere's centre
    0.951 mm (sph3t: 1.62 mm, 1.93 deg and 1.43 mm).
  - The HEX-E had 239,951 samples, no timeout, and one gap of 1.0 s. Exit 0; home at 21:28.
- Pass 2's clearance with cb1t's registration on v3 put no action under the margins over pad, grid and final (mount
  ≥ 3.20 mm, finger ≥ 0.90 mm). The plan is `lab/20261023-plan_sphere_lab_r22_pass2.json`: the 12 offset touch-offs, 65
  grid presses with ref-00 to -05, and final-ref. It has no gravity orientations, pivots, rolls or twists.

### 21:3x to 23:58, cb1t: pass 2 (the offset touch-offs, the grid, rolls and twists)

- 21:3x, the owner asked for the turns: "are we also still doing twists and rolls? the contact pose (direction of
  contact) is pretty important". roll-0, roll-3 and the 8 twists of last night's checked turns went back into the pass 2
  plan. Their joint 6 runs from −23.3 to +26.5 deg, and with cb1t's registration the mount keeps 3.78 mm and the
  finger 3.68 mm.
- The simulator resumed a copy of cb1t, with its truth at cb1t's registered tool and sphere
  (`sim_from_reg.py` in the lab session's scratchpad). Pass 2 ran with no abort. The 31 touch-offs fitted at 0.040 mm
  rms. The cell check flagged grid-027 and -028: the mount came 2.7 to 3.0 mm from the cell at their deepest, so both
  were dropped. The turns alone passed the cell check (mount ≥ 3.4 mm, finger ≥ 3.4 mm).
- 21:56:11 to 22:27:34, pass 2a, stages pad and register: the 12 offset touch-offs with no abort.
  - The onsets ran from −0.002 to +0.468 mm. The simulator gave +0.08 to +0.32 mm.
  - The registration fitted all 31 at 0.054 mm rms. Against pass 1's fit it turned the TCP 1.934 deg (moving it 0.081
    mm) and moved the sphere's centre 1.958 mm. The simulator had predicted 0.894 deg and 0.268 mm. The plain touch-offs
    leave this combination loose, and resolving it is what the offset touch-offs are for.
  - The HEX-E had 187,505 samples, no timeout and no gap.
- The clearance check with that registration put six actions under the mount's 3.0 mm, and they were dropped: grid-025
  (2.56 mm), grid-002 and -007 (2.70), twist-1-1.0N and -2.5N (2.85), and grid-024 (2.99). The simulator then ran the
  grid, the turns and final-ref with its truth at the new registration: no abort, and the cell check passed (mount
  ≥ 3.2 mm, finger ≥ 2.0 mm, joint 6 from −65.8 to +56.7 deg).
- 22:40:05 to 23:57:43, pass 2b, stages grid, roll, twist and final. It ran the 59 grid presses (with ref-00 to -05),
  roll-0 and roll-3, the 6 twists left and final-ref, with no abort, warning or error. The plan of this run is
  `plan_resume_2156_2240.json`; the run at 21:56 had the same plan with the six dropped actions still in it, outside its
  stages.
  - The reference presses, as onset and depth change from ref-00 (1.654 mm at 0.1 N, depth 0.619 mm): ref-01
    +0.044/−0.009, ref-02 +0.050/+0.048, ref-03 +0.037/−0.002, ref-04 +0.004/+0.030, and ref-05 (23:32)
    +0.024/+0.005 mm. All are inside the stop rule (0.3 mm and 0.1 mm), with no trend.
  - The HEX-E had 464,911 samples, no timeout, and one gap of 0.6 s. Home at 23:59.
- The session folder is `20261023-203938-cb1t/`: the session's files, the streams, every printout, and the plans of
  each resume. The first attempt is `20261023-203123-cb1t/`.

## 2026-10-25: R24, the second checkerboard tip (v1 ID 2) on mount v6

### 10:5x, pass 1 prepared (nothing moved)

- Pulled 88f961d9. v6's CAD is in its folder. The cell model takes `UR5_MOUNT=v6` with the plain
  `standoff_fork.stl`; the `_i10` to `_i55` forks beside it are not used.
- Pass 1's plan, `lab/20261025-plan_sphere_lab_r24_pass1.json`, is cb1t's pass 1 with v6's payload (0.399 kg at
  34.3 mm). Its stages are preflight, homing, pad and register, with no gravity sweep. The session will be `cb2t`.
- Clearance with `UR5_MOUNT=v6`, cb1t's last registration and the stored seating: pad-face-11 at 2.66 mm and pad-face-07
  at 2.70 mm from the mount. Both clear the touch-off margin of 2.5 mm (owner, 10-25), so all 19 stay.
- `check_arm_samples.py --cell` with v6 over cb1t's whole recorded motion (pass 1, 2a and 2b) breaks no rule. The mount
  stays at least 4.9 mm from the cell at the touch-offs, 4.5 mm in the grid, and 5.6 to 6.0 mm in the turns; the finger
  stays at least 1.8 mm.
- Waiting for the owner's go. The board's USB cable is to be taped before the first touch-off.

### 11:46 to 14:1x, cb1r: R24's tests on v1 ID 1, still on mount v3

- The owner (lab session, 11:3x): the workstation's tasks are for the v1 ID 1 tip still on the UR5, with "full approval
  ... once we've done the typical checks". Then: "let's make sure we do all the tests intended for v1 id 2 on the v6
  mount on this actual current v1 id1 on the v3 mount. the information we get should all be usable". The fork was
  untouched since cb1t, and the board on USB was ID 1's (QT Py b4:3a:45:b0:d9:94). The robot, the HEX-E (0.047, 0.024
  and 0.229 N) and the board passed their checks.
- 11:46:53 to 12:31:18, pass 1 (`lab/20261023-plan_sphere_lab_r22_pass1.json`, no gravity sweep): 19 touch-offs with no
  abort.
  - The onsets repeat cb1t's to within 0.18 mm, and most to within 0.08 mm.
  - The registration fitted them at 0.046 mm rms (cb1t: 0.052). It moved the TCP 1.509 mm and 1.316 deg, and the
    sphere's centre 0.902 mm (cb1t: 1.568 mm, 1.318 deg and 0.951 mm).
- Pass 2's plan, `lab/20261025-plan_sphere_lab_r24_pass2_cb1r.json`, is cb1t's with the actions dropped for cb1t's
  registration put back. All of them cleared pass 1's registration.
  - The simulator's cell check again found grid-027 and -028 with the mount 2.8 to 3.0 mm from the cell at full depth
    (153 samples, as for cb1t), so both were dropped.
- 12:49:27 to 13:20:49, pass 2a: the 12 offset touch-offs with no abort, then the registration of all 31 at 0.048 mm
  rms. It turned the TCP 1.620 deg and moved the sphere's centre 1.574 mm from pass 1's fit (cb1t: 1.934 deg and
  1.958 mm), so the offset touch-offs move the fit the same way on both days.
- With that registration grid-025 (2.37 mm), -002 (2.53), -007 (2.65) and -024 (2.84) fell under the mount's 3.0 mm, as
  for cb1t. Pass 2b was then dropped altogether: the owner (relayed 13:2x) wants ID 2 on v6 by 14:30, and 2b repeats
  cb1t's.
- Pass 3 (`lab/20261025-plan_sphere_lab_r24_pass3_cb1r.json`, v3's payload, final-ref renamed final-ref-p3):
  - In the simulator, 17 of its 18 twins turned +30 deg put joint 5 at 17.7 to 27.6 deg, outside the wrist window
    (30 to 100 deg); the first three aborted.
  - twin-057-t+30 cleared the window, but its joint 6 reached −84.7 deg, past the −65 deg of cb1t's run.
  - All 18 were dropped, leaving the 6 unturned twins. The workstation rebuilt 12 of them at +10 or +15 deg as pass 3b
    (`..._pass3b_cb1r.json`, 92438a5a): joint 5 35.7 to 47.0 deg.
  - A helper (`joints_check.py` in the lab session's scratchpad) now checks joints 5 and 6 at each rest pose by the
    simulator's IK before a run.
- 13:22:31 to 13:27:39, shear-0-1N alone, the first slide on the robot, with no abort. It pressed to 1 N, then slid
  0.1, 0.2 and 0.3 mm in eight directions.
  - At the holds the normal force stayed between 0.61 and 1.24 N and the tangential force between 0.07 and 0.70 N
    (ratio at most about 0.6).
  - The peak was 1.94 N in one sample (1.60 N over 0.1 s), and the torque at most 113 N mm.
  - The tangential force does not grow steadily from 0.1 to 0.3 mm in every direction, and it carries an offset of
    about +0.3 N along x_I. That is for the analysis.
- 13:31:50, passes 3 and 3b in one run (`..._pass3_3b_cb1r.json`): 18 twins, the other two slides, the 3 twists at 5 N
  and final-ref-p3. Checked with pass 2a's registration (joints, clearance on v3: mount ≥ 3.53 mm) and simulated with it
  (no abort, cell check clean).
  - Its preflight printed a noise of 0.000 N on Fy. The HEX-E reports Fx and Fy in steps of 0.1 N; Fz is finer. In a
    quiet 2 s window every Fy sample sat on one level. sph3t and cb1t show the same steps, so this is the sensor's
    resolution and not a fault. The tangential forces of the slides come from 0.1 N steps; only means over the holds,
    smoothed by the noise, resolve finer.
- 14:07:48, passes 3 and 3b ended (exit 0): every action ok, no abort. The HEX-E gave 215,289 samples, none timed out,
  with no gap.

### 14:06 to 15:0x, cb1r: pass 4, the rate test, on the new program

- Pass 4 (`lab/20261025-plan_sphere_lab_r24_pass4_cb1r.json`) is the first run of the new program (cea8b477, ba5de833).
  It was simulated with pass 2a's registration twice: once in the plan's stage order, then in the run's order (twist,
  roll, shear, grid, final). No action aborted. The cell check on v3 was clean: mount ≥ 3.5 mm, finger ≥ 2.7 mm,
  HEX-E ≥ 34.1 mm, joint 5 39.9 to 70.8 deg. The 2 mm/s presses peaked at 2.51 to 2.71 N in the simulator.
- 14:17:58 to 14:52:13, pass 4 on the robot: 28 of 29 actions ok. The HEX-E gave 204,892 samples, none timed out, with
  no gap.
  - Every fast approach crossed its 1.0 mm of air without meeting the pad early.
  - The presses to 2 N peaked at 2.20 to 2.46 N at the 0.15 mm/s setting, 2.22 to 2.34 N at 0.5 mm/s and 2.47 to
    2.57 N at 2 mm/s.
  - roll-0-tv004 aborted at 14:33:52 without contact: "the fast approach did not end". The arm had reached its
    target. The loop waited for a pose within 0.02 mm, but the CB3 settles about 0.03 mm away and its reported pose
    jitters by about 0.025 mm. The 8 approaches before it had waited 0.4 to 1.4 s for a lucky sample.
  - The fix (207bd7c8): the approach ends when the arm is steady after 0.2 s, as the travel and the turns already do.
    The simulator reports the commanded pose exactly, so it could not show the failure.
- **The pendant's speed slider is at 30 %.** move_guarded read 0.30 from the controller at 14:13, and the arm's own
  data agree.
  - The presses' air crossings ran at 0.30 of their set speed in sphere2 (10-22), rod1 (10-23) and cb1r. Pass 4's
    fast approach, set to 5 mm/s, peaked at 1.6 mm/s.
  - So every speed in every session so far ran at 0.3 times the plan's value. The grid presses ran at 0.15 mm/s, the
    touch-offs at 0.045 mm/s, and the turns at 0.07 deg/s for turn_v 0.004.
  - Pass 4's rates are therefore 0.045, 0.15 and 0.6 mm/s for the presses, and 0.07, 0.28 and 0.69 deg/s for the
    turns.
  - The analysis should take speeds from robot.bin, not from the plans.
- 14:53:10 to 14:57:05, roll-0-tv004 again on the fixed program (`--resume`, stages preflight and roll): ok, its fast
  approach 1.42 s. The HEX-E's one gap (0.6 s) came at the start, before the preflight, as at 13:22. All 29 actions of
  pass 4 are now ok, and cb1r is done.
- The session is in `lab/20261025-114653-cb1r/`: the small files, each run's printout and plan (`printout_resume_HHMM`,
  `plan_resume_HHMM`), then the streams in a separate commit.

## 2026-10-25 evening: R24 on v1 ID 2 (cb2t), mount v7, unattended

### 15:0x to 15:29, prepared

- The owner mounted ID 2 on mount v7. Its board, a QT Py ESP32-S3 (MAC b4:3a:45:b0:9c:d0), still ran the mux firmware
  and sent only "[ERROR] no I2C mux at 0x70 to 0x77 on the STEMMA QT port".
- Reflashed here with `reskin5x_stream` through `host/flash.sh`: arduino-cli 1.1.1 in `~/.local/bin`, the esp32 core
  3.3.2 as on the workstation, and the eFlesh submodule initialised. The board then streamed 376 records/s of 80 bytes,
  with no zero records.
- Plans:
  - passes 1 and 2a from R22's (1ed52366); pass 2b from R22's pass 2 (d97834a3);
  - passes 3 and 5 from the workstation (b43e752d);
  - all with v7's payload (0.403 kg at 34.3 mm) and the travel block; 2b, 3 and 5 with `speed_slider` 0.30.
- Checked on v7 with cb1t's registration:
  - passes 1 and 2a clear the mount by 3.48 mm and the finger by 1.15 mm;
  - passes 2b, 3 and 5 clear the mount by 3.14 mm and the finger by 1.42 mm;
  - three actions reach joint 6 at −65.4 to −65.8 deg: grid-052, and twin-067-t+15 with its slides.
- Passes 1 and 2a, simulated from cb1t's registration on 207bd7c8: no abort, and the cell check on v7 clean.
- The owner's go came at 15:2x: "be thorough, get these variable speeds on the presses and rotational movements, and
  lets get as good a sensor as possible!" The owner left at 15:3x, after the first two touch-offs.

### 15:29:20 to 15:48:19, cb2t pass 1

- 19 touch-offs, no abort. Onsets came −0.77 to +0.49 mm from the expected (median −0.39 mm); the fast part of every
  approach ended before contact.
- The registration fit at 0.051 mm rms. The TCP moved 1.43 mm and 1.93 deg; the sphere's centre moved 0.67 mm.
- The HEX-E gave 113,303 samples, none timed out, with no gap.
- Pass 2a, rechecked with this registration, clears the mount by 3.16 mm and the finger by 1.20 mm, with every joint
  inside its window. In the simulator nothing aborted, and the cell check was clean: mount ≥ 4.1 mm, finger ≥ 2.0 mm.

### 15:52:48 to 16:08:02, cb2t pass 2a (the 12 offset touch-offs)

- No abort. The registration over all 31 touch-offs fit at 0.046 mm rms. The TCP turned 1.88 deg and moved 0.28 mm, and
  the sphere's centre moved 0.80 mm. cb1t (1.93 deg) and cb1r (1.62 deg) turned the same way.
- The HEX-E gave 90,794 samples, none timed out, with no gap.
- Passes 2b, 3 and 5, rechecked on v7 with this registration, have no action under the margins. The mount clears by at
  least 3.74 mm, the finger by 1.07 mm and the HEX-E by 30.9 mm.
- Four actions go past the −65 deg of joint 6 that cb1t's run reached. They were dropped, since the suite runs
  unattended:
  - grid-052 in pass 2b, at −65.1 deg;
  - twin-067-t+15 in pass 3, at −66.1 deg;
  - its two slides in pass 5, also at −66.1 deg.
- That leaves 67 actions in pass 2b, 55 in pass 3 and 101 in pass 5.

### 16:23:58 to 17:23:54, cb2t pass 2b (R22's grid, rolls and twists)

- Simulated first on cb2t's registration: no abort. The cell check on v7 was clean: mount ≥ 3.7 mm, finger ≥ 1.5 mm.
- The run printed "speed slider 0.30 (the plan's 0.30)" with no warning: the controller took the input and read the same
  value back.
- All 67 actions were ok: 58 grid presses, 2 rolls, 6 twists and the final reference press.
- The HEX-E gave 358,770 samples, none timed out. Its one gap (0.7 s) came at the start, before the preflight, as in
  earlier runs.
- Passes 3 and 5, simulated on cb2t's registration: no abort, and the cell checks on v7 were clean.
  - Pass 3: mount ≥ 4.0 mm, finger ≥ 1.4 mm.
  - Pass 5: mount ≥ 3.7 mm, finger ≥ 1.5 mm.

### 17:24:50 to 18:39:44, cb2t pass 3 (twins, slides, twists at 5 N, the rate set at four speeds)

- "speed slider 0.30 (the plan's 0.30)" again. All 55 actions were ok: 36 grid actions (the twins and the rate presses),
  6 slides, 9 twists, 3 rolls and the final reference press.
- The HEX-E gave 448,540 samples, none timed out. Its one gap (0.7 s) came at the start, before the preflight.
- The rate presses to 2 N peaked higher the faster they went:
  - 2.11 to 2.36 N at v 0.00015 (0.045 mm/s at the slider's 0.30);
  - 2.43 to 2.56 N at v 0.002 (0.6 mm/s);
  - 2.95 to 3.28 N at v 0.00667 (2.0 mm/s).

### 18:40:37 to 21:16:54, cb2t pass 5 (the grid at 4 times the contact speed, slides at more forces and points)

- "speed slider 0.30 (the plan's 0.30)". All 101 actions were ok: 52 fast-grid presses, 48 slides and the final
  reference press. A slide took about 3.2 min.
- The HEX-E gave 936,524 samples, none timed out. Its one gap (0.7 s) came at the start, before the preflight.

### Pass 6, prepared during pass 5

- The workstation added pass 6 (776b0f88) under the owner's request of 16:3x: "improvisationally devise more tests ...
  (more speeds, more variations, more locations/location density), torsion, contact wrenches". It has 77 actions:
  - 38 new press targets, each at one of four contact speeds;
  - 36 slides at 1 N on those targets;
  - twist-3 at 2.5 and 5 N out to 20 deg;
  - final-ref-p6.
- Rechecked on v7 with cb2t's registration, four actions were under the margins:
  - dense-574-t+10 and its slide, with the finger at 0.79 mm;
  - slide-dense-464-t-5-1N, with the mount at 2.87 mm;
  - slide-dense-597-t+10-1N, with the mount at 2.82 mm. In the simulator 23 of its samples came within 2.9 to
    3.0 mm, the only samples of the pass that broke a rule.
- The four were dropped (a991ac00). The 73 left clear the mount by 3.25 mm and the finger by 1.13 mm, with every joint
  inside its window.
- Simulated again on cb2t's registration: no abort. The cell check on v7 was clean: mount ≥ 3.4 mm, finger ≥ 1.4 mm,
  joint 5 from 36.2 to 95.6 deg.

### 21:17:47 to 23:42:10, cb2t pass 6

- "speed slider 0.30 (the plan's 0.30)". All 73 actions were ok: 37 presses, 33 slides, 2 twists and the final
  reference press. A slide took 3.2 to 3.75 min.
- The HEX-E gave 865,212 samples, none timed out. Its one gap (0.7 s) came at the start, before the preflight.
- The robot was homed afterwards (23:4x), and it stands there.

### cb2t in all

- 329 actions over passes 1, 2a, 2b, 3, 5 and 6, from 15:29 to 23:42, and not one aborted. Every run's HEX-E stream had
  no timeout; the only gaps came at program start.
- 8 actions were dropped before their runs, past joint 6's −65 deg or under the margins on v7.
- The session is in `lab/20261025-152920-cb2t/`: the small files, each run's printout and plan (`printout_resume_HHMM`,
  `plan_resume_HHMM`), then the streams in a separate commit.

## 2026-10-26: R25 on v1 ID 3 (cb3t), mount v7, cb2t's suite unchanged

### 12:3x to 13:05, prepared

- R25's plans (2abab229) are cb2t's as they ran after the lab's drops. Passes 1 and 2a now carry `speed_slider` 0.30 too.
- Passes 1 and 2a, simulated from cb2t's registration: no abort, "speed slider 0.30 (the plan's 0.30)". On v7 they
  clear the mount by 3.10 mm and the finger by 1.16 mm, and the cell check was clean.
- The owner mounted ID 3 at 13:0x and gave the go. ID 3 reads through the same QT Py as ID 2 (b4:3a:45:b0:9c:d0), still
  on the non-mux firmware: 376 records/s, noise 1.5 to 2.8 µT on all five chips. Chip 3's Z sits near 0 µT at rest,
  so a third of its readings are exactly 0.0; that is the reading's step, not a fault.

### 13:05:46 to 13:25:02, cb3t pass 1

- Onsets came −0.64 to +0.87 mm from the expected (median −0.26 mm).
- pad-edge-07 aborted at 13:24:50: "no HEX-E sample for 100 ms".
  - The HEX-E sent nothing for 2.8 s (55 timed-out reads) and then resumed. Fz was the same on both sides of the gap,
    so the sensor did not restart and lose its tare as on 10-22.
  - The machine was idle (load 0.8). Joint 6 stood at +55.0 deg, the highest of the pass and 5 deg below the cable's
    +60, with the arm creeping in at 0.15 mm/s.
  - The same touch-off ran clean on cb2t. The HEX-E's cable may sit tighter since the swap.
  - Polled alone afterwards for 40 s, the HEX-E gave 4,002 samples with no timeout.
- The registration over the other 18 fit at 0.045 mm rms. The TCP moved 1.19 mm and 1.17 deg; the sphere's centre moved
  0.81 mm.
- Pass 2a re-runs pad-edge-07, since a resume repeats every action that did not end ok. Rechecked with this
  registration, it clears the mount by 3.35 mm and the finger by 1.64 mm, with every joint inside its window.
- Joint 6 goes above +50 deg at 2 rest poses in pass 2a, 7 in 2b, none in 3, 7 in 5 and 6 in 6 (up to 59.3 deg). All
  of them ran clean on cb2t.

### 13:31:28 to 13:31:56, pass 2a stopped: the HEX-E fell off its DAQ

- pad-edge-07 ran first, at joint 6 +54.5 deg. The HEX-E's stream stopped at 13:31:45 and stayed down.
- pad-off-00 and pad-off-01 aborted at once on "no HEX-E sample for 100 ms". After three aborts in a row the program
  stopped the session (exit 1).
- The DAQ box still answered ping, but its status showed no sensor: empty serial number, sample counter 0, status
  131072. This is the failure of 10-22, the HEX-E's connector.
- The arm stood in free air about 15 mm off the pad, lifted after each abort. The guarded move needs the HEX-E, so it
  stayed there.
- The workstation proposed capping joint 6 at +50 deg for the rest of cb3t. The owner chose to try the pose again
  first: "i see that the motion does slightly jostle the DAQ cable, which is unfortunate, but i dont want to lose data
  parity".

### 13:35 to 13:44, the owner's cable work and a retry

- At 13:35:48 the DAQ showed the sensor again (HEXEA478, its sample counter running). Polled for 20 s, the HEX-E gave
  2,001 samples with no timeout.
- The retry (`lab/20261026-plan_sphere_lab_r25_retry_cb3t.json`, on pass 1's registration) ran three repeats of pass 1's
  touch-offs, then pad-edge-07, with the owner watching the cable:
  - the repeats came +0.127, +0.061 and +0.110 mm from the expected, where pass 1's residuals there were under
    0.043 mm;
  - pad-edge-07 ran clean at joint 6 +54 deg. The HEX-E gave 28,347 samples, none timed out.
- The repeats, all late by about 0.1 mm, say the pad moved slightly during the cable work. Pass 1 was redone in a fresh
  session.

### 13:45:22 to 14:05:43, cb3t pass 1 again (session 134522-cb3t)

- All 19 touch-offs were ok, pad-edge-07 among them at joint 6 +54 deg. The HEX-E gave 121,423 samples, none timed
  out.
- The registration fit at 0.034 mm rms. Against the first registration, the pad on the flange moved 0.077 mm and
  turned 0.21 deg, and the sphere moved 0.107 mm: the shift that the repeats showed.
- The suite continues in this session. The first session (`130546-cb3t`) keeps its touch-offs, the two dropouts and
  the retry, as data.
- Pass 2a, rechecked with this registration, clears the mount by 3.49 mm and the finger by 1.69 mm. In the simulator
  nothing aborted, the cell check on v7 was clean, and joint 6 peaked at 51.2 deg.
- The actions above joint 6's +50 deg stay in, for parity with cb2t (the owner).

### 14:10:11 to 14:22:56, cb3t pass 2a

- Preflight passed: noise 0.042, 0.022, 0.230 N, no timeout.
- Of the ten offset touch-offs, eight ended ok, 0.057 to 0.412 mm from the expected. At 14:22:36 the HEX-E fell off its
  DAQ again, at joint 6 +43 deg at the end of a swing from −41 deg. pad-off-10 and pad-off-11 aborted with no HEX-E
  sample for 100 ms. The DAQ answered ping but showed no sensor (serial "", sample counter 0, status 131072).
- The registration fit the 29 touches at 0.042 mm rms. It moved the TCP 0.123 mm and 1.25 deg, and the sphere
  0.567 mm.
- The HEX-E gave 74,235 samples; 331 timed out.

### 14:2x to 14:43, the owner's second cable fix

- The owner fixed the cable a second time. The DAQ showed HEXEA478 again; polled for 20 s, the HEX-E gave 2,002 samples
  with no timeout.

### 14:43:09 to 14:50:56, the check and the swing test (`lab/20261026-plan_sphere_lab_r25_check2_cb3t.json`)

- Three repeats of pass 1's touch-offs came +0.041, +0.066 and +0.050 mm from the expected: the pad had not moved.
- pad-off-10 and pad-off-11 ended ok (+0.200 and +0.138 mm). Pass 2a is complete.
- swing-flank-05-a ended ok at joint 6 −62 deg. On the swing to swing-edge-07-a, the HEX-E fell off its DAQ at
  14:50:45, at joint 6 +53.1 deg. The two next swings aborted too, and the session stopped after three aborts in a row
  (exit 1). The HEX-E gave 45,294 samples; 156 timed out.
- The owner traced the dropouts to the cable striking the table during the swings.

### 15:1x, the third cable fix: a slack chain at the wrist

- The owner reseated the HEX-E connector and fastened a slack chain to the wrist, so the cable cannot reach the table.
- The DAQ showed HEXEA478 with its counter running; polled for 15 s, the HEX-E gave 1,502 samples with no timeout.

### 15:24:11, five swings of joint 6 (`lab/20261026-plan_sphere_lab_r25_check3_cb3t.json`)

- Three repeats of pass 1's touch-offs came +0.060, +0.037 and +0.038 mm from the expected.
- Five swings followed, each from pad-flank-05 at joint 6 −62.5 deg to pad-edge-07 at +53.1 deg. All ten touch-offs
  ended ok: flank-05 +0.020 to +0.152 mm from the expected, edge-07 +0.013 to +0.046 mm. The HEX-E gave 99,479
  samples, none timed out. Its two gaps of 0.7 s, with no timeout, appear in clean runs too (cb2t's passes). Exit 0
  at 15:40:52.

### 15:45, the cap on joint 6 for the rest of the day: +45 deg

- The rule agreed with the workstation before the swings: five clean swings cap joint 6 at +45 deg for R26, R27 and
  cb3t's passes 2b, 3, 5 and 6; any dropout caps it at +35 deg.
- Each action's joint 6 was taken at its own poses, from the arrival at its rest pose to its end (`rec.t_rest0`). For
  passes 2b to 6 this came from cb2t's run of the same actions on v7; for R26, from its simulation on cb3t's pass 1
  registration. No action falls between +43 and +45 deg, so ID 3's small difference from ID 2 cannot carry one over.
- 27 actions over +45 deg were dropped:
  - pass 2b (67 to 60): grid-044, -045, -046, -048, -057, -058 and -059, at +49.5 to +57.1 deg;
  - pass 5 (101 to 94): the same seven as fast-grid-*;
  - pass 6 (73 to 62): dense-554-t+15, -555-t+15, -613-t+10, -614-t+10 and -615-t+10, and the slides
    slide-dense-554-t+15-1N, -555-t+15-1N, -611-t+5-1N, -613-t+10-1N, -614-t+10-1N and -615-t+10-1N, at +45.5 to
    +61.0 deg;
  - R26 hard2 (57 to 55): hard-grid-049 and hard-dense-614-t+10, at +48.4 deg;
  - pass 3 keeps all 55; its highest is +38.0 deg.
- Each plan's `notes` lists its drops and the reason.
- R26 on the final registration (pass 2a's): hard1 clears the mount by 5.34 mm, hard2 by 4.26 mm, the finger by 1.82
  mm and the HEX-E by 28.2 mm.

### 16:00:54 to 16:26:19, the parity actions past the cap, with the owner present

- The owner, through the workstation at about 15:50: "im still in person though. can we just try those 27 dropped
  actions first, now, so that we can get the risky data while im in person?" The workstation asked for the 25 parity
  actions in one plan in session 134522-cb3t, with their original IDs, lowest joint 6 first. R26's two actions go back
  into hard2, which runs after hard1. The cap at +45 deg stays for the unattended passes.
- `lab/20261026-plan_sphere_lab_r25_parity45_cb3t.json` holds pass 2b's 7, pass 5's 7 and pass 6's 11 from before
  the cap. Within each stage they are ordered by joint 6 at their own poses in cb2t's run.
- On the final registration, slide-dense-555-t+15-1N cleared the mount by 2.84 mm (needs 3.0). In the simulator its
  least was 2.87 mm. It is also the highest of the set (joint 6 +61.0 deg), so it was left out. The other 24 clear the
  mount by 3.18 mm or more and the finger by 2.25 mm. Simulated on the final registration: no abort.
- All 24 ended ok: 19 presses at joint 6 +45.7 to +57.6 deg, then 5 slides up to +60.3 deg. The HEX-E gave 151,887
  samples, none timed out. The slack chain held the cable past the +53 deg of the test swings. Exit 0 at 16:26:19.
- hard2 is back to 57 actions. It passes the static check on the final registration (mount 4.26 mm, finger 1.82 mm).
  R26's simulation on that registration (hard2 without the two) had no abort. Its cell check was clean (mount 4.2 mm,
  finger 1.9 mm), and its presses peaked at 8.8 N. hard-ref-c's apparent 8.3 N is the ramp of hard1: the two runs
  share one session copy, and their simulated clocks overlap.

### 16:28:43 to 16:30:32, R26 hard1, the canary

- The owner, at 16:27: "go ahead".
- hard-ref-a pressed the reference point to 2 N. hard-ramp-ref pressed the same point to 8 N at 0.15 mm/s. hard-ref-b
  repeated the 2 N press. Nothing aborted.
- The board's change at each level was taken as its hold mean less its rest mean, over the 20 channels. At 2 N,
  hard-ref-b differed from hard-ref-a by 9.4 % of the change. cb2t's ten reference presses differ by 8.2 % (median)
  and 13.8 % at most.
- Right after the ramp, the pad reached 2 N 72 µm deeper. By hard-ref-c, 1.8 min later, it was within 11 µm. The
  difference was the pad recovering from 8 N.

### 16:31:38 to 18:03:14, R26 hard2

- All 57 actions ended ok, with no abort. The HEX-E gave 548,674 samples; none timed out.
- The presses to 8 N peaked at 8.4 N at most. The fast presses (to 3 N at 4 mm/s) peaked at 4.03 to 4.65 N; the
  simulator had 6.7 to 7.7 N.
- hard-ref-d came after all the presses. Its board change differed from hard-ref-a's by 7.3 %, and the pad reached
  2 N within 25 µm.
- hard-final-ref came after the slides, rolls and twists (6 N, up to 20 deg). Its board change differed by 15.7 %,
  past cb2t's 13.8 %, and the pad reached 2 N 0.22 mm deeper.
- hard2 took 1.53 h on the robot, against 0.30 h in the simulator.

### 18:04:28 to 18:08:42, the repeat check after R26 (`lab/20261026-plan_sphere_lab_r26_check4_cb3t.json`)

- The three repeats of pass 1's touch-offs came +0.216 (face-00), +0.055 (face-03) and +0.104 mm (edge-00) from the
  expected. At 15:24 they came +0.060, +0.037 and +0.038 mm.
- hard-ref-e, 5 min after hard-final-ref, reached 2 N 95 µm deeper than hard-ref-a. Its board change differed by
  10.0 %, within cb2t's spread.
- The hard contacts left a dent near the reference point that is still recovering: face-00 lies nearest that point,
  and face-03 did not move. The pad as a whole did not shift. Before passes 2b to 6 resume on this registration, the
  repeats must come back within 0.1 mm; otherwise pass 1 is redone.
- The owner, at 18:0x: the tip with four N52 magnets will be ready in about 40 min. Until then, rod tests on ID 3;
  then the new tip's calibration.

### R27 pass 1 (the rod), checked here

- `lab/20261026-plan_rod_lab_r27_pass1_rod3t.json` is the workstation's `plan/20261023-plan_rod_lab_touch35.json`
  with the payload, travel block and `speed_slider` of cb3t's pass 1. It runs stages preflight, homing, pad, pivot and
  register in a fresh session, from the rod's seating (`poses.jsonl:right_indenter_seated`).
- With cb3t's registered tool and the rod at its seating, the static check predicts onsets −0.45 to +0.70 mm from
  the expected, and no rest pose in contact. The mount clears the rod by 2.79 mm on the touch-offs (needs 2.5).
- The first simulation (15:08) took the rod from the plan's `sim.T_BI`, which the workstation had set on 10-23 to
  reproduce ID 1's tool error. Its onsets came 1.4 to 2.2 mm early and its clearances down to 1.6 mm. Those reflect
  the stand-in rod, not ID 3.
- Two more simulations used cb3t's registered tool and a fitted rod:
  - the rod where the workstation's joint fit of R19 put it (0.77 mm above the seating): no abort; the onsets came
    0.87 to 2.09 mm early, 12 of 22 in the fast approach, which falls back to the slow one. The mount cleared the
    rod by 3.28 mm or more (pad-edge-01), the finger by 2.76 mm;
  - the rod where rod1's own fit put it (1.1 mm further along its axis): pad-edge-01, pad-face-04, pad-edge-04 and
    pad-edge-06 brought the mount to 1.47 to 2.10 mm of the rod. Nobody has measured the rod's position along its
    axis; touches on its curved face leave that unconstrained.
- pad-edge-01 was dropped (22 to 21 touch-offs). It sits at joint 6 +44.7 deg, against the cap of +45. It was R19's
  outlier (rod1t, +0.66 mm), which the joint fit leaves out. It is also the mount's closest approach in both
  simulations. The other touch-offs clear by 3.30 mm or more with the joint fit's rod.

### 18:09:37 to 18:27:30, R27 rod pass 1 (session `180938-rod3t`)

- The owner, at 18:0x: "run as many rod tests as we can before then", meaning before the tip with four N52 magnets
  is ready (about 40 min).
- `lab/20261026-plan_rod_lab_r27_pass1_rod3t.json`, stages preflight, homing, pad, pivot and register, from the
  rod's seating. Homing put the rod at (−289.0, −584.6, 69.0) mm, yaw −94.1 deg, tilt 1.43 deg.
- All 21 touch-offs ended ok. The onsets came −0.97 to −1.95 mm from the expected (median −1.48). Nine met the rod
  in the fast approach and fell back to the slow one, as the simulation with the workstation's joint-fit rod
  predicted (−0.87 to −2.09 mm). The five pivot presses ended ok.
- The registration fit the 21 touches at 0.051 mm rms (R19's rod1: 0.122 mm). It moved the TCP 1.10 mm and
  2.75 deg, and the rod's centre 1.46 mm from the seating, to (−289.31, −585.17, 70.33) mm. The program fits the rod
  as a sphere at one point, so this TCP lies 0.85 mm and 3.83 deg from cb3t's sphere registration; the cylinder fit
  of `scripts/ur5_tip_calibration_rod_registration.py` should judge it.
- The HEX-E gave 106,641 samples, none timed out. Exit 0 at 18:27:30.

### 18:31:35, R27 rod pass 2a: the rate set alone

- `scripts/ur5_tip_calibration_rod_pass2.py` built rod pass 2 on a simulated stand-in of rod pass 1 (the tool
  where cb3t's registration put it, the rod where the joint fit put it): 132 actions. That covers 63 presses (47 of
  R19's 48 targets, 16 more), 24 rate presses at six face targets, 32 slides at eight, 6 rolls, 3 twists and four
  reference presses. It simulated with no abort and the mount at 3.1 mm or more. Its rod pass 2 needs about 1.7 h.
- Twists go to +15 deg at rod-grid-019. At the reference point a turn about the vertical moves joint 5: +15 deg takes
  it to 26 deg. A negative turn swings the mount into the rod (−5.9 mm at −15 deg).
- In the 15 min left, the rate set ran alone: `lab/20261026-plan_rod_lab_r27_pass2a_rod3t.json`, 27 actions on
  rod pass 1's registration. The mount clears by 5.16 mm and the finger by 4.17 mm. In the simulator there was no
  abort; joint 5 stayed within 42 to 49 deg and joint 6 within −29 to +15 deg. The simulator's fast presses peaked at
  7.9 to 9.0 N. On the sphere it gave 6.7 to 7.7 N where the robot measured 4.0 to 4.7 N.
- All 27 ended ok, 18:31:35 to 18:42:55. The fast presses (to 3 N at 4 mm/s) peaked at 4.65 to 5.13 N, the 2 mm/s
  presses at 3.51 N at most. At 2 N, the reference presses after the set reached within 54 µm of rod-ref-a's flange
  position. The HEX-E gave 67,504 samples, none timed out. The arm went home for the tip swap.
- The rest of rod pass 2 (presses to 8 N, slides, rolls, twists) has not run on ID 3. It needs a rebuild on
  rod3t's registration (`scripts/ur5_tip_calibration_rod_pass2.py logs/20261026-tip_calibration/180938-rod3t`).

### 19:2x, how well the three v1 tips sense (owner: "run a thorough set of analysis on the v1 id 3 data that we did get ... and see how good (ideally) the sensing for the three v1 fingertips are")

- ID 3's sessions were analysed from their lab folders, where the plans of their resumes are
  (`scripts/ur5_tip_calibration_analysis.py lab/20261026-134522-cb3t --gravity-from logs/20261022-tip_calibration/
  195420-first2`, as the workstation analysed ID 1 and ID 2). That gives 813 holds on the sphere (270 presses up to
  7.8 N, 491 slide, 30 twist and 22 roll holds) and 103 on the rod: `data/20261026-134522-cb3t/`,
  `data/20261026-180938-rod3t/`.
- `scripts/ur5_tip_calibration_v1_tips.py` (`data/20261026-v1_tips.json`) scores each tip's sphere presses. ID 1 has
  98 press actions, ID 2 187 and ID 3 60. Median location error over 1.5 N, then under 1.5 N (mm):

  | estimator | ID 1 | ID 2 | ID 3 |
  |---|---|---|---|
  | physical (viewer) | 1.29, 9.64 | 1.35, 8.31 | 1.99, 8.23 |
  | GP, own presses (5 folds) | 0.40, 0.88 | 0.25, 0.72 | 0.46, 1.48 |
  | GP, own, thinned to ID 3's 48 training actions | 0.54, 0.97 | 0.48, 1.09 | 0.51, 1.44 |
  | GP, the other two tips only | 0.74, 1.19 | 0.96, 1.42 | 1.44, 1.79 |
  | GP, the other tips and own (pooled) | 0.29, 0.99 | 0.24, 0.75 | 0.48, 1.10 |

- The GP's force error over 1.5 N on a tip's own presses is 0.20, 0.16 and 0.25 N.
- By region, over 1.5 N (the GP from the other two tips, then the physical estimator):
  - face: 0.65 and 1.32 mm on ID 1, 0.81 and 1.08 mm on ID 2, 0.43 and 1.37 mm on ID 3;
  - edges: 0.83 and 1.45 mm on ID 1, 1.12 and 1.41 mm on ID 2, 1.68 and 2.81 mm on ID 3;
  - flanks: 0.94 and 0.84 mm on ID 1, 1.43 and 2.88 mm on ID 2, 1.72 and 3.50 mm on ID 3.
- On its own presses, ID 3's edges come to 0.55 mm (ID 1 0.57, ID 2 0.32).
- So ID 3's face senses like ID 1 and ID 2. Its edges and flanks respond differently from theirs: a model of ID 1
  and ID 2 misses them by about twice its error between those two. Calibrated on its own presses at the same density,
  ID 3 is as good as the others over 1.5 N (0.51 against 0.54 and 0.48 mm).
- Under 1.5 N, ID 3 is worse (1.44 mm thinned). It has few light presses (76 holds, against 285 and 548), because
  passes 2b to 6 did not run.
- Two cautions. ID 3's presses are mostly on edges and flanks: 37 % of its holds over 1.5 N lie on the face, against
  61 % and 56 %. And 50 of its 194 holds over 1.5 N are R26's 5 to 8 N, beyond the 1 and 3 holds over 5 N of ID 1 and
  ID 2. Over 5 N, the GP from the other tips misses the force by 3.9 N (extrapolation); ID 3's own GP misses it by
  0.36 N.

## 2026-10-26 evening: R28 on v2 ID 1 (five N52 magnets in the checkerboard, mount v7), session `v2id1`

- The owner swapped tips at about 19:00. The workstation's R28 (QUEUE.md): cb3t's plan files in cb3t's order, the cap at
  +45 deg on passes 2b to 6, the 24 actions above it only with the owner present, R26 after pass 6.
- The board check at rest (19:2x, `reskin5x_check.py`): 375 records/s from all five chips, noise 1.6 to 3.0 µT. The
  largest component is chip 0's Bz, −13.4 mT, 34.9 % of the 38.4 mT range. ID 3's chip 0 read 3.6 mT, and ID 3's
  presses to 8 N changed a chip's field by 0.14 mT at most.

### 19:21:57 to 19:42:50, pass 1

- All 19 touch-offs ended ok. The onsets came −0.69 to +0.87 mm from the expected: the nominal tool against a pad
  that sits tilted in its seat (ID 3's first pass 1 had +0.85 mm at pad-edge-07). pad-edge-07 ran at joint 6 +54 deg
  with no gap.
- The registration fit the 19 touches at 0.050 mm rms (ID 3: 0.034, ID 2: 0.046). It moved the TCP 1.04 mm and
  1.71 deg, and the sphere 0.83 mm.
- The HEX-E gave 124,596 samples, none timed out.
- The poles fitted at rest (`fingertip_wrench_model.calibrate_frames_fixed`, with the board's earlier chip frames,
  from the rest before pad-flank-05) are (+1, −1, −1, −1, +1), as on ID 1, ID 2 and ID 3. That fit leaves 0.36 mT
  rms; the next pattern leaves 2.64 mT. The strength k is 0.96 (ID 2: 0.47, ID 3: 0.20). The largest components are
  0.349, 0.096, 0.089, 0.217 and 0.221 of the range.

### 19:44:04 to 19:59:06, pass 2a; 20:00:15 to 20:57:07, pass 2b (capped, 60 actions)

- Pass 2a: all touch-offs ok, pad-off-10 and -11 after their swing among them. The registration fit 31 touches at
  0.047 mm rms; it moved the TCP 0.324 mm and 1.40 deg, the sphere 0.657 mm.
- The owner did not answer whether they would be at the robot for the 24 parity actions, so those wait.
- Pass 2b: all 60 actions ended ok. The HEX-E gave 340,474 samples, none timed out.
- The chips under load: over pass 2b's first presses (up to 5 N), chip 0's largest component reached 0.358 of the
  range (0.349 at rest). The largest change on a chip was 191 µT. N52 leaves the chips far from clipping.

### 20:57:54 to 22:12:57, pass 3 (55 actions)

- All ended ok. The HEX-E gave 449,475 samples, none timed out. Pass 5 started at 22:13:42.

### 22:13:42 to 00:46:50, pass 5 (94 actions)

- All ended ok. The HEX-E gave 917,547 samples, none timed out. Pass 6 started at 00:47:33.

### R26 for v2 ID 1, prepared during pass 5

- The owner, 23:5x: "sure you can run the harder presses. just, as always, be careful." R26 runs unattended after
  pass 6.
- On v2 ID 1's registration, R26's margins tighten. The pad sits about 1 mm differently on the mount: the reference
  point clears the mount by 4.02 mm, against 5.34 mm on ID 3.
- With R26's margins (each raised by the press's depth at its peak, less 1 mm), 14 of its actions fail. hard1's 8 N
  ramp at the reference point needs 4.49 mm.
- `lab/superseded/20261027-plan_sphere_lab_r26_hard1_v2id1.json`: the reference presses stay at the reference point. The
  8 N ramp moves to hard-grid-020's target, where 8 N passes.
- `lab/superseded/20261027-plan_sphere_lab_r26_hard2_v2id1.json` has 44 actions. It is hard2 under the cap of +45 deg
  (no hard-grid-049 or hard-dense-614-t+10), with hard-grid-053 lowered to 3 N and hard-grid-030 to 5 N.
- Dropped from it: hard-grid-000, -003, -005 and -008, whose passing level is no higher than pass 2b's own 5 N on the
  face. Also fast4-grid-000, -009, -023 and -032, hard-shear-0-4.5N, hard-roll-0-4N and hard-twist-0-6N.
- Simulated on v2 ID 1's registration: no abort. The cell check was clean (mount 3.4 mm, finger 1.9 mm, HEX-E 27 mm).
  Joint 5 stayed within 37 to 95 deg and joint 6 within −62.5 to +42.2 deg. Peaks reached 8.8 N.
- hard2 runs only if hard1's reference presses agree: the board's change at 2 N within cb2t's repeat spread
  (13.8 %), and the flange's position within about 0.1 mm once the pad has recovered.

### 00:47:33 to 02:44:22, pass 6 (62 actions); 02:45:07 to 02:47:26, R26 hard1

- Pass 6: all ended ok. The HEX-E gave 699,846 samples, none timed out. v2 ID 1's suite under the cap is complete;
  the 24 parity actions wait for the owner.
- hard1: the 8 N ramp at hard-grid-020's target peaked at 8.31 N. Nothing aborted.
- At 2 N, hard-ref-b's board change differed from hard-ref-a's by 8.8 % of the change (94.9 µT; N52 roughly doubles
  cb2t's 44.4). That is within cb2t's repeat spread of 8.2 % (median) and 13.8 % (max); ID 3 gave 9.4 %. v2 ID 1's own
  ref-00 and ref-05 of pass 2b, an hour apart, differ by 3.6 %.
- The flange at 2 N moved 25 µm. The ramp pressed another point, and the reference point shows no dent.
- hard2 started at 02:48:32.

### 02:48:32 to 04:03:20, R26 hard2 (44 actions)

- All ended ok. The HEX-E gave 447,943 samples, none timed out. The presses peaked at 8.41 N and the twists at 6.17 N.
- hard-ref-c and hard-ref-d bracketed the presses. Their board change at 2 N differed from hard-ref-a's by 4.3 % and
  3.6 % (v2 ID 1's own repeat: 3.6 %). The flange at 2 N moved under 30 µm. The hard presses left the pad unchanged.
- hard-final-ref came after the slides, the roll and the twists, the last two at the reference point at 5 N with fast
  turns. Its board change differed by 4.6 % at 2 N, but the pad reached 2 N 0.24 mm deeper. ID 3 showed the same
  after its twists (0.22 mm, 0.095 mm after 5 min).
- Before the 24 parity actions run on this registration, the repeats of pass 1's touch-offs should come within
  0.1 mm.
- The arm went home. v2 ID 1's suite under the cap (passes 1 to 6) and R26 are complete. The 24 parity actions wait
  for the owner.

### 04:2x, v2 ID 1 against the v1 tips (the owner's worry, 19:2x: the larger seam "has caused some atypical loading patterns")

- v2 ID 1's session was analysed as ID 3's (`lab/20261026-192158-v2id1`, first2's gravity fit): 3,620 holds, in
  `data/20261026-192158-v2id1/`.
- `scripts/ur5_tip_calibration_v1_tips.py --v2` (`data/20261027-v1_v2_tips.json`); v2 ID 1 has 198 press actions.
  Median location error over 1.5 N, then under 1.5 N (mm):

  | estimator | v1 tips (ID 1, ID 2, ID 3) | v2 ID 1 |
  |---|---|---|
  | GP, own presses (5 folds) | 0.40, 0.25, 0.46; 0.88, 0.72, 1.48 | 0.25; 0.66 |
  | GP, own, thinned to 48 training actions | 0.54, 0.48, 0.51; 0.97, 1.09, 1.44 | 0.51; 0.98 |
  | physical (viewer) | 1.29, 1.35, 1.99; 9.64, 8.31, 8.23 | 1.57; 2.02 |
  | GP trained on the v1 tips | – | 2.52; 0.90 |
  | pooled | 0.29, 0.24, 0.48; 0.99, 0.75, 1.10 | 0.29; 0.69 |

- Calibrated on its own presses, v2 ID 1 is as good as the best v1 tip (ID 2), and at one density it equals all
  three. Its force error over 1.5 N is 0.19 N.
- The physical estimator reads its light presses within 2.0 mm, against 8 to 10 mm on the v1 tips. The N52 field
  roughly doubles the board's change at a given press.
- A GP trained on the v1 tips misses v2 ID 1's contacts by 2.5 mm and its force by 1.9 N over 1.5 N. The stronger
  magnets change the board's response, so a v2 tip needs its own calibration.
- No sign of atypical loading from the seam. By region, v2 ID 1's own errors match ID 2's: face 0.23 and 0.68 mm (ID 2
  0.21 and 0.67), edges 0.30 and 0.67 (0.32 and 0.80), flanks 0.45 and 0.58 (0.53 and 0.79). Over 1.5 N they are
  symmetric about the middle: 0.28, 0.23 and 0.29 mm at y < −1.5, the middle and y > 1.5 (ID 2: 0.28, 0.22 and 0.28).

## 2026-10-27 morning: v2 ID 1's parity actions and the rod, with the owner present

- The owner, at 09:1x: "yes run the rod pass and the parity actions". Asked which tip the rod pass runs on: v2 ID 1 as
  mounted.

### 09:17:39 to 09:21:04, the repeat check (`lab/superseded/20261027-plan_sphere_lab_check_v2id1.json`)

- Three repeats of pass 1's touch-offs (chk-face-00, chk-face-03, chk-edge-00) came +0.074, −0.041 and −0.028 mm from
  the expected: the pad had not moved since pass 1. The HEX-E gave 20,026 samples, none timed out.

### 09:23:08 to 09:51:40, the parity actions past the cap (`lab/superseded/20261027-plan_sphere_lab_r25_parity45_v2id1.json`)

- The 24 of ID 3's parity run, and slide-dense-555-t+15-1N, which ID 3 left out (mount 2.84 mm on its
  registration). On v2 ID 1's registration it clears the mount by 3.35 mm; the 24 others clear it by 3.56 mm or more,
  the finger by 2.14 mm. These 25 are the set ID 2 ran before the cap.
- In the simulator on v2 ID 1's registration, none aborted. The mount stayed 3.4 mm or more from the sphere, the finger
  2.3 mm or more. Joint 5 stayed at 41.9 deg or more, and joint 6 reached at most +59.8 deg.
- All 25 ended ok, with joint 6 from +39.7 to +60.0 deg. The HEX-E gave 170,603 samples, none timed out. v2 ID 1
  now has the full sphere suite that ID 2 ran.

### Rod pass 1 for v2 ID 1, checked here (`lab/superseded/20261027-plan_rod_lab_r27_pass1_v2id1.json`)

- R27's rod pass 1, with the touch-offs' slow approach starting 2.5 mm before the expected surface instead of
  1.5 mm. Simulated with v2 ID 1's sphere registration and the rod where the workstation's joint fit of R19 put it,
  pad-face-01 met the rod at the end of the fast approach. Its slow approach then had too few samples and it aborted,
  twice. On ID 3 the real onset of pad-face-01 came at −1.44 mm, at that hand-over.
- With 2.5 mm the simulation had no abort and no contact in the fast approach. The onsets came −1.02 to −1.96 mm from
  the expected, and the registration fit all 21. The mount stayed 3.38 mm or more from the rod (pad-face-04), the
  finger 3.00 mm, the HEX-E 31.9 mm. Joint 6 stayed within −46 and +31 deg.
- The static check, with the rod at its seating, puts the mount 2.88 mm from it at pad-face-04 and 2.91 mm at
  pad-edge-06. ID 3's least there was 2.79 mm, and its run was clean.

### 09:53:03 to 10:11:06, rod pass 1 on v2 ID 1 (session `095303-rodv2id1`)

- Stages preflight, homing, pad, pivot and register in a fresh session, from the rod's seating. Homing put the rod
  where it put it for ID 3: (−289.0, −584.6, 69.0) mm, yaw −94.1 deg, tilt 1.43 deg.
- All 21 touch-offs ended ok, none in the fast approach. The onsets came −1.07 to −1.93 mm from the expected (ID 3:
  −0.97 to −1.95 mm); pad-face-01 came at −1.52 mm. The five pivot presses ended ok.
- The registration fit the 21 touches at 0.030 mm rms (ID 3: 0.051 mm). It moved the TCP 1.04 mm and 2.55 deg, and
  the rod's centre 1.30 mm from the seating, to (−289.39, −585.09, 70.16) mm: 0.20 mm from where ID 3's put it.
- This TCP lies 0.83 mm and 4.22 deg from v2 ID 1's sphere registration; ID 3's rod registration lay 0.85 mm and
  3.83 deg from its own. The program fits the rod as a sphere at one point, and both tips show the same offset.
- The HEX-E gave 107,643 samples, none timed out. Exit 0.

### Rod pass 2 for v2 ID 1, checked here (`lab/superseded/20261027-plan_rod_lab_r27_pass2_v2id1.json`)

- `scripts/ur5_tip_calibration_rod_pass2.py logs/20261027-tip_calibration/095303-rodv2id1 --j6-max 45` built it on
  rod pass 1's registration: 135 actions. That covers 66 presses (47 of R19's 48 targets, 19 more; 37 on the face, 29
  on the edges), rates at six face targets, slides at eight, 6 rolls, the twists at rod-grid-019 and the reference
  presses. rod-grid-040 was dropped: joint 6 reaches +46.7 deg there.
- The builder's least clearance of the mount is 3.05 mm (rod-more-686), with the margins raised by each press's depth.
- In the simulator (the tool where v2 ID 1's sphere registration put it, the rod at the joint fit of R19) all 135
  ended ok. The mount stayed 3.1 mm or more from the rod, the finger 1.5 mm, the HEX-E 31 mm. Joint 5 stayed at
  35.7 deg or more, and joint 6 within −44 and +44 deg. The simulated time was 0.42 h.
- The simulator's fast presses (to 3 N at about 4 mm/s) peaked at 8.9 to 9.4 N. On ID 3's rod it gave 7.9 to 9.0 N
  where the robot measured 4.65 to 5.13 N.

### 10:36:36 to 12:10:46, rod pass 2 on v2 ID 1

- Stages preflight, grid, shear, roll, twist and final, resumed on rod pass 1's session. All 135 ended ok in 1.57 h.
  Joint 6 stayed within −43.6 and +44.3 deg.
- The presses reached their planned tops: 23 at 8 N, 3 at 6.5 N, 19 at 6 N, the rest at 2 to 5 N. No slow press
  passed its top by more than 0.76 N. The fast presses to 3 N at about 4 mm/s peaked at 5.02 to 5.53 N (ID 3's
  rate set: 4.65 to 5.13 N).
- The reference presses rod-ref-b, rod-ref-c and rod-final-ref differed from rod-ref-a's board change at 2 N by 6.5,
  6.9 and 9.1 % (ID 2's sphere repeats: median 8.2 %, at most 13.8 %). At 2 N the flange stopped within 46 µm of
  rod-ref-a's, after the twists too: the rod's twists did not leave the pad deeper, as the sphere's did.
- The HEX-E gave 564,108 samples, none timed out. Exit 0. The arm went home.
- v2 ID 1 now has the rod set ID 3 lacks: rod pass 1, the rate set, and the presses to 8 N, slides, rolls and twists.

## 2026-10-27 afternoon: v2 ID 1 on the tall indenter

- The owner, 12:2x: "im going to get the taller indenter mounted first and we can calibrate v2 id 1 on that in the
  interim". The workstation took R29, the pass after pass 1 (QUEUE.md).

### 13:37 to 13:51, the tall indenter and three seatings on its key

- The owner put the tall indenter on a second 2x4, clamped, about 104 mm further from the robot than the sphere
  fixture. The sphere fixture and the rod stay where they were, and so do their seatings. The second 2x4's top is about
  2 mm higher, since the board bows a little under the clamps.
- Three seatings by hand in teach mode (`lab/teach_hold.py`), recorded with `lab/record_pose.py` as
  `tall_indenter_seated`, `_2` and `_3`. By the plan's homing geometry the first puts the base at (−34.8, −699.4,
  73.0) mm, yaw −93.7 deg, tilt 1.75 deg.
- The spheres the seatings imply lie 0.94 mm (first and second), 0.55 mm (first and third) and 1.47 mm (second and
  third) apart. The second sat 1.0 mm higher at the flange than the others. The flange turned 0.39 to 0.80 deg between
  seatings; the README's seat check asks for 0.15 deg in roll and 0.05 mm along the tool axis.

### 13:52 to 13:58, the mount's fork on the sphere

- After the third seating, `lab/move_guarded.py go home` lifted off the key, then went home in a straight line. Its
  cell model has no tall indenter. 19 mm along the line the HEX-E guard stopped the arm at 3.2 N. The owner: "the top
  right side of the dip link / fork is touching the left side of the sphere". With the tall indenter in the model,
  the fork stood 0.5 mm from the sphere's +y_I side at the stop.
- At rest the fork pushed the sphere with about 5.5 N. The neck should take 35 to 70 N.
- A guarded move back along the same line released the contact. The first try stopped after 0.3 mm, because the
  push falling from 5.5 to 3.2 N passed the 3 N change rule. The second let the push fall and would have stopped on a
  rise of 1 N. Then the arm went 80 mm toward the robot along −x_I, up 40 mm, and home; no move changed the HEX-E's
  force by more than 1 N.
- `lab/leave_tall_key.py` now takes that route from the key: up 10 mm and 15 mm, 80 mm along −x_I, up 40 mm, then
  home. It checks every segment against the cell with the tall indenter, and refuses unless the arm is at a seating.

### Pass 1 on the tall indenter, checked here (`lab/superseded/20261027-plan_tall_lab_pass1_v2id1.json`)

- `scripts/ur5_tip_calibration_plan.py --indenter tall` built it with the first seating's placement, and the sphere
  fixture and the rod as obstacles (the plan's notes give the command). Of the 159 actions built, the 32 touch-offs
  were kept, and 9 dropped for joint 6 outside −65 to +45 deg.
- Every touch-off starts 5.0 mm off the surface (the builder's 1.5 mm; cb3t's pass 1: 5.0 mm), with the slow approach
  from 2.5 mm before it. Payload, travel speeds and speed slider are cb3t's pass 1's.
- On v2 ID 1's registration, with the indenter from the first seating, the static check puts the mount 3.52 mm from
  it, the finger 1.56 mm and the HEX-E 21.4 mm. It predicts onsets of −0.45 to +0.90 mm.
- Simulated with the tool of v2 ID 1's sphere registration and the sphere where each seating puts it:
  - pad-flank-02 brought the finger's cap within 0.17 mm of the sphere at contact (0.00 mm with the second seating's
    sphere), and pad-flank-06 took joint 6 to −66.3 deg. Both were dropped, leaving 21.
  - With the second seating's sphere, pad-edge-02 and pad-flank-02 found no contact within 1.2 mm past the expected
    surface. Every touch-off now runs to 2.0 mm past it.
- The three simulations then had no abort. The registration fit 0.020 to 0.023 mm rms, the mount cleared by 3.11 mm
  or more and the finger by 0.94 mm or more, and joint 6 stayed within −64.3 and +35.0 deg.

### 14:35:29 to 15:10:13, pass 1 on the tall indenter (session `143529-tallv2id1`)

- Stages preflight, homing, pad and register from the first seating, with the owner's go. Homing put the base at
  (−33.3, −699.2, 72.4) mm, yaw −93.4 deg, tilt 1.83 deg.
- All 21 touch-offs ended ok, none in the fast approach. The onsets came −0.24 to +1.25 mm from the expected (median
  +0.35), and grew through the run, which goes in order of joint 6. pad-edge-02 came at +1.25 mm, past the 1.2 mm
  limit of the first build.
- The registration fit the 21 touches at 0.054 mm rms. The README's mark is 0.05 mm; v2 ID 1 on the sphere fit
  0.047 mm over 31 touches. It moved the TCP 1.50 mm and 2.59 deg, and the sphere's centre 1.25 mm from the seating.
- The HEX-E gave 207,467 samples, none timed out. Exit 0.

### R29 on the tall indenter (`lab/superseded/20261027-plan_tall_lab_r29_v2id1.json`, the workstation's, ec08a7aa)

- The owner confirmed joint 6 to +60 deg for R29 (16:3x: "yea thats good with me").
- Simulated on tall pass 1's registration: all 122 ended ok. The mount stayed 3.06 mm or more from the indenter, the
  finger 2.03 mm, the HEX-E 22.5 mm. Joint 6 stayed within −61.5 and +57.5 deg. The simulated time was 0.66 h. The
  static check on the session agrees: mount 3.25 mm, finger 1.54 mm, HEX-E 22.1 mm.
- 16:38:01 to 17:06:25: 45 of the 59 presses ended ok, the 8 past +45 deg first (joint 6 +48.4 to +55.4 deg). Then
  the HEX-E fell off its DAQ: tall-face-702, tall-face-690 and tall-edge-627 aborted with no HEX-E sample, at joint 6
  +18.5 deg, and the session stopped after three aborts in a row. The DAQ showed no sensor (serial "", counter 0,
  status 131072). The HEX-E gave 169,072 samples; 158 timed out. The arm was stopped with the pad 67 mm from anything.
- By 17:2x the sensor was back by itself (serial HEXEA478, its counter restarted; 1001 samples in 10 s, none timed out).
  The owner had not reseated it; the slack loop at the wrist was not tight. An intermittent contact in the connector
  or the cable is the likely cause.
- 17:23:43 to 17:28:17, three repeats of pass 1's touch-offs (`lab/superseded/20261027-plan_tall_lab_check_v2id1.json`):
  chk-face-03 −0.081 mm and chk-edge-00 −0.055 mm from the expected, chk-face-00 −0.166 mm, outside the 0.1 mm mark.
  pad-face-00 is the face's centre (index 788), where R29's reference presses go.
- The owner asked for lower lifts between actions ("maybe dont go up quite as much on these lift offs"):
  `lab/superseded/20261027-plan_tall_lab_r29_lift30_v2id1.json` is R29 with the travel's lift at 30 mm instead of 60,
  being simulated on the session's registration.
- 17:44:27 to 19:00:24, the resume on the lift-30 plan (`printout_resume_1744.txt`; `plan_resume_1744.json` is that
  plan): the preflight, 12 presses, 2 reference presses and slides 1 to 26 of 50, all ok, no aborts.
- The owner stopped R29 there to put on the next tip ("i would like to go home and swap now please"). SIGINT came as
  slide 27 started, before it moved. Not done: 24 slides, the 12 twists and the final reference, 37 of 122 actions.
- The sphere check (`lab/superseded/20261027-plan_sphere_lab_check_shear_v2id1.json`, sets a and b) was not run.
- Home at 19:01:24, on a path checked against the tall indenter. The pad started 1.8 mm from the sphere; no part came
  closer.

## 2026-10-27 evening: v3 ID 1 on the robot, the tip holder

The owner put v3 ID 1 on the robot at about 19:05. The robot's board is still the S3 `b4:3a:45:b0:9c:d0` (five chips
at about 340 records/s). The holder's two v3 tips read through the mux board `b4:3a:45:b0:d9:94`, channels 0 and 3, at
206 records/s each. Runs pass both ports by id (`--board`, `--held-board`).

### Seatings by hand

- The holder: `holder_seated` to `_3` (19:13 to 19:19) and, after the owner doubted the tilt, `_4` to `_6` (19:51 to
  19:54). All six tilt the holder 3.2 to 4.9 deg and differ by up to 4.9 mm and 3.5 deg at the flange. At the seat
  the HEX-E passes about 2 mm from the low sphere; the owner saw the wrist "just barely" clear it.
- The rod blocked most of the holder's map, so the owner moved it about 97 mm along -x_B and re-homed it:
  `rod_indenter_seated` to `_3` (19:36 to 19:42). Its seatings differ by up to 4 deg, mostly a roll about the tool axis.
- The runs home on `holder_seated_5`, the middle of the last three.

### Touch-offs on the held tips

- 20:05:36, `20261027-200536-holdv3id1`, three touch-offs on the +y tip's planned apex: all three found nothing. The
  robot's apex went 1.24 mm below the planned apex, within 0.1 mm of it sideways.
- 20:24:26, `20261027-202426-holdapexv3id1`, the workstation's apex search (five touch-offs reaching 6 mm past): all
  found contact, 2.87 to 4.04 mm below the plan. The workstation's fit puts the +y apex 2.86 mm low and 2.50 mm toward
  -y_I, flat along x (rms 0.024 mm). A level holder through that apex sits at z_B 71.35, beside the tall indenter's
  71.2 on the same 2x4.
- 21:21:31, `20261027-212131-holdmapv3id1`, the five -y touch-offs of the level frame's plan: onsets +0.15 to +0.62 mm
  from its prediction. The holder is level; the hand seatings sat about 4 deg off in the seat.

### The crosstalk map

- The first four hovers ran at 21:28 to 21:30 in the same session (the watched cut of
  `lab/superseded/20261027-plan_holder_lab_map_apex_v3id1.json`): all held, the HEX-E's change at most 0.24 to 0.33 N.
- The owner left at about 21:00 with the go for the rest unattended ("i told you to just send it ... be careful").
- 21:39:37, the rest of the map, resumed in the same folder (`logs/20261027-213937-tipcal-holdmapv3id1.log`): 85
  hovers held, the HEX-E's |F| at most 0.46 N; exit 0 at 22:22:55.
- One false abort: `hov-fm-W10-x+3`, 10 mm from the -y tip, stopped at 0.52 N on the HEX-E against its contact_N of 0.5
  N. The far hovers run 0.45 to 0.52 N from the HEX-E's noise and the arm's inertia alone. Retried alone at 22:24:33
  with contact_N 1.0 (`lab/superseded/20261027-plan_holder_lab_map_apex_retry_v3id1.json`): held, at most 0.38 N. The
  map is complete: 90/90 hovers and the five -y touch-offs.

### The close hovers over the -y tip

- 22:26:40, `20261027-222640-holdclosev3id1`, the workstation's
  `lab/superseded/20261027-plan_holder_lab_map_close_my_v3id1.json` (48 hovers, down to 0.5 mm from the tip): 48/48
  held, |F| at most 0.45 N; exit 0 at 22:50:00. Checked in the cell model first with the level frame
  `lab/20261027-holder-apex-both` (base z_B 71.23).

### v3 ID 1's registration on the sphere

- 22:52:13, `20261027-225213-spherev3id1`, the workstation's `lab/superseded/20261027-plan_sphere_lab_pass1_v3id1.json`:
  18/18 touch-offs, onsets -0.56 to +0.82 mm from the expected, 0.049 mm rms; exit 0 at 23:12:09.
- Against v2 ID 1's registration (192158): the sphere 0.51 mm and 0.68 deg apart, the tool 0.38 mm and 0.52 deg, the
  pad frame 0.47 mm and 1.55 deg; the pad points within 0.45 mm.
- Stage 2 (`lab/superseded/20261027-plan_sphere_lab_stage2_v3id1.json`, 264 actions) resumed the same folder at 23:14:27
  (`logs/20261027-231427-tipcal-spherev3id1.log`). The simulator ran it first in four chunks on v2 ID 1's registration,
  then again on this pass 1's: 0 aborts, joint 6 -61.7 to +43.9 deg, joint 5 at least 5.1 deg inside its window. Two
  actions came under the mount's 3.0 mm margin on v3 ID 1's registration: `dense-700-t+10` (2.97 mm in the workstation's
  path check, 99a8132e; 3.05 mm in the lab's simulation) and `slide-dense-700-t+10-1N` (2.87 mm in the lab's
  simulation). A SIGINT right after `dense-657-t+15` ended (grid 133/161, 0 aborts) stopped the run at 00:35:21 with the
  robot at that press's rest pose. It resumed at 00:35:50 without both
  (`lab/superseded/20261027-plan_sphere_lab_stage2_resume_v3id1.json`, 262 actions;
  `logs/20261028-003550-tipcal-spherev3id1.log`). The simulator ran both new transitions first: mount 5.43 mm in the
  grid, 5.30 mm in the shear stage.
- The grid ended with all 160 actions ok, and 40 of the 84 slides after it. At 02:54:23.9 the HEX-E fell off its DAQ
  in the joint-space travel into `slide-twin-051-t-20-2N`, while joint 6 swung from -35.5 to +13 deg with the tool at
  z 235 mm and nothing in contact. That slide and both `slide-twin-063-t+15` slides aborted on "no HEX-E sample for
  100 ms", and the session stopped after three aborts in a row (exit 1 at 02:54:35). The DAQ answered ping but showed
  no sensor (serial "", counter 0, status 131072), as on 10-26 and in R29. The arm stood in free air, its safety mode
  normal.

### The owner's decisions at about 01:05 and 01:20

- The rod runs without the owner, through the usual process ("can the rod plans run without my checking? i trust the
  registration process"): pass 1 simulated on the rod's three seatings from home, stage 2 re-checked on pass 1's own
  registration, actions under the margins left out.
- The +45 deg cap on joint 6 is lifted for the two morning plans tonight ("yea, let's lift the cap. ill be in at ~9am
  anyway"); the lab holds them to +60 deg. The sphere's
  (`lab/superseded/20261028-plan_sphere_lab_morning_unattended_v3id1.json`) keeps 43 of its 55 actions after simulating
  it from home on v3 ID 1's registration: the three `*grid-055` (finger 0.36 mm), `dense-419-t+15`, `dense-464-t-5` and
  `hard-dense-464-t-5` (finger 0.49 to 0.78 mm), two slides past +60 deg and four past -65 deg are left out. The 43:
  mount 3.22 mm, finger 1.66 mm, joint 6 -63.3 to +59.0 deg.

### Rod pass 1 for v3 ID 1, checked here (`lab/superseded/20261027-plan_rod_lab_pass1_search_v3id1.json`)

- The simulator ran pass 1 (21 touch-offs, 5 pivots) from the home pose on each of the rod's three seatings. It used
  v3 ID 1's tool frames from 225213's registration. All three ended with 0 aborts and registered to 0.012 to 0.015 mm
  rms.
- The mount cleared by at least 3.31, 2.67 and 3.31 mm on seatings 1, 2 and 3; the touch-off margin is 2.5 mm. The
  finger cleared by at least 1.77 mm, the HEX-E by 32 mm and the links by 40 mm.
- On all three seatings joint 6 ran from -51.1 to +29.3 deg and joint 5 from 33.2 to 49.1 deg, 3.2 deg inside its
  window. The elbow stayed at least 73.1 deg from straight.
- These checks took every fifth sample. On seating 2, the closest, the 11 nearest touch-offs were checked again at
  every sample. They gave the same 2.67 mm, at `pad-edge-06`.
- The workstation's path check first gave 1.25 to 2.85 mm, because it used v2 ID 1's tool frames. With v3 ID 1's it
  gives 2.68 mm on seating 2.
- The rod fixture comes off for the side-by-side hovers in the morning. After it goes back on, the owner records
  `rod_indenter_seated_4`. If that seating lies outside seatings 1 to 3, pass 1 is simulated again before it runs.

### The morning's other plans, simulated here from the home pose

- The side-by-side hovers (`lab/20261028-plan_holder_lab_side_norod_v3id1.json`, 45 hovers, with the -y tip
  pad-down): mount 4.14 mm, finger 2.70 mm, joint 6 -6.5 to +6.4 deg, the elbow at least 82.5 deg from straight.
- The sphere's twists, rolls and final references (`lab/superseded/20261028-plan_sphere_lab_stage2_twists_v3id1.json`,
  18 actions): mount 4.06 mm, finger 4.70 mm, joint 6 -17.0 to +25.2 deg, the elbow at least 102.9 deg from straight.

The clearances in this section and the one before it are on mount v3. The next section has them on v7.

## 2026-10-28: v3 ID 1 after the HEX-E's reseat

### The cell model's mount

- The lab's cell checks from the evening of 10-27 to 07:40 on 10-28 placed mount v3's parts. The robot has carried v7
  since R25. The workstation found it at about 07:40; its own path checks had the same default.
- The cell model (`scripts/ur5_cb3_cell.py`) now loads v7 when `UR5_MOUNT` is unset (d79eeca4). v7's standoff fork
  reaches about 5.8 mm further toward a neighbour side by side. Elsewhere the two mounts give nearly the same clearances.
- The simulated paths don't depend on the mount, so only the checks ran again, on v7:
  - Rod pass 1: 3.35, 2.70 and 3.28 mm on seatings 1, 2 and 3.
  - The twists plan: 4.28 mm. The check plan: 5.37 mm. The sphere morning plan: 3.13 mm.
  - The sphere's remaining 62: at least 3.57 mm, except `slide-dense-795-t+15-1N` at 2.88 mm. It is left out of
    `lab/20261028-plan_sphere_lab_stage2_resume3_v3id1.json`, which holds the 43 slides still to do.
- Tall pass 1 homed on seating 2 brought `pad-face-01` and `pad-face-04` to 2.34 and 2.35 mm, under 2.5 mm. Without them
  (`lab/superseded/20261028-plan_tall_lab_pass1_19_v3id1.json`, 19 touch-offs) the mount clears by 3.35, 2.63 and 3.76
  mm.
- The workstation rebuilt the side map on v7 (85c0eab6, 38 hovers). In the lab's simulation 8 of them came under
  3.0 mm: g 14 mm at twist 0, and g 15 mm with s 5 at twist +10, each with its repeat (2.76 to 2.90 mm). Without them
  (`lab/20261028-plan_holder_lab_side_norod30_v3id1.json`, 30 hovers) the mount clears by 3.75 mm. The map needs the
  rod fixture off and the -y held tip pad-down; with the rod on, only 4 side poses clear it.

### 09:33 to 12:23

- The HEX-E came back at 09:33:31, after the owner reseated its connector. A 20 s stream gave 2001 samples, no
  time-outs and no gaps. Untared Fz read 645 N, the sensor's offset after a restart.
- 09:36:12, the check: `chk-face-00`, `chk-face-03` and `chk-edge-00` met the sphere +0.031, +0.025 and +0.034 mm from
  the registration's prediction.
- 09:41:50 to 10:25:26, the twists plan: 2 rolls, 12 twists and 4 final references, all ok.
- 10:27:01 to 11:46:46, rod pass 1 (`20261028-102701-rodv3id1`): 21 search touch-offs, onsets -0.44 to +0.24 mm,
  0.042 mm rms, and 5 pivots. The search approaches slowly over 13 mm, about 3.7 min a touch-off.
- 11:49:08 to 12:21:39, tall pass 1 (`20261028-114908-tallv3id1`): 19 touch-offs, onsets -0.41 to +0.96 mm,
  0.045 mm rms.
- 12:23:03, rod stage 2 resumed on rod pass 1's session: 94 actions
  (`lab/superseded/20261028-plan_rod_lab_stage2_v7_v3id1.json`). The rod morning plan follows it with 6 actions
  (`lab/superseded/20261028-plan_rod_lab_morning6_v3id1.json`).
- Each run's HEX-E summary shows one 0.7 s gap. It falls at the stream's start, while the program configures and
  tares the DAQ, before preflight.
- Every plan of the day travels at speed 1.0, as the sphere has since the dropout at 02:54. The paths are the same,
  so the simulations stand.

### Rod stage 2's four presses

- On rod pass 1's registration the workstation's path check put an M2 head or the cover inside the finger's 0.8 mm
  margin at `rod-grid-049` (0.48 mm), `rod-more-606` (0.52), `rod-grid-035` (0.61) and `rod-grid-048` (0.67). That
  check presses to a model depth: the sphere's, scaled as F^(2/3), plus 0.3 mm.
- At the depths v2 ID 1 reached on the rod, plus 0.3 mm, the four clear by 1.11 to 1.33 mm. v3 ID 1's pad presses
  within 0.05 mm of v2 ID 1's on the sphere (818 holds; 0.315 against 0.331 mm/N). So the four left stage 2 and joined
  the morning plan, with `rod-grid-039` (1.36 mm at the measured depth).
- The lab's simulation of the 94 from home: mount at least 3.52 mm, finger 1.58 mm, joint 6 -56.2 to +30.6 deg. The
  morning plan's 6: mount 4.63 mm, finger 1.57 mm, joint 6 -53.2 to +38.3 deg, under the +45 deg cap. Joint 5 comes
  0.3 deg inside its window at `rod-more-606`, now in the morning plan; in the 94 it keeps 0.82 deg (the workstation's
  check).

### 12:23 to 17:41

- 12:23:03 to 13:45:44, rod stage 2: 94 of 94 ok. 13:47:10 to 13:52:47, the rod morning plan: 6 of 6 ok, the four
  presses back from stage 2 among them.
- 13:54:15, tall stage 2 at travel.speed 1.0 (`lab/superseded/20261028-plan_tall_lab_stage2_ts1_v3id1.json`). 44 actions
  ended ok. At 14:32:30.25 the HEX-E fell off its DAQ again, in the travel into `tall-edge-627`. `tall-edge-627`, `-694`
  and `-626` aborted, and the session stopped at 14:32:40 with the arm in free air at z 373 mm.
- At the last sample, wrist 1 was swinging from -28.5 to +24.7 deg, wrist 2 from 65.8 to 91.9 and the elbow from 98.4
  to 70.7. Joint 6 moved only from 23.1 to 27.1 deg. At 02:54 wrist 1 had swung from -45 to +3 deg, wrist 2 from 46 to
  67 and joint 6 from -35 to +13. So both losses came as wrist 1 swung from negative to positive with wrist 2 rising: a
  bend of the cable across the wrist, not a turn of joint 6. Before the loss the session had swung wrist 1 by more than
  45 deg in 15 travels, up to 68 deg, without a loss. The connector is marginal, and a wrist bend unseats it.
- The owner reseated the connector and the sensor came back at 14:39:22. A 20 s stream was clean, and the check
  (`lab/20261028-plan_sphere_lab_check2_v3id1.json`, the same touch-offs under new names) gave +0.045, +0.008 and
  +0.066 mm.
- The owner: "maybe dont go up so high on these tall indenter moves". The remaining 58 actions and the tall morning
  plan ran with travel.lift_mm 20 (from 60): `lab/superseded/20261028-plan_tall_lab_stage2_l20_v3id1.json` and
  `..._morning_l20_v3id1.json`. Simulated from home first: mount at least 3.78 mm, finger 2.56 mm, the elbow 70.6 deg
  from straight. The TCP stays at z 333 mm at most, against 373. Wrist 1's largest swing drops only from 69 to 64 deg,
  because the tool must turn between actions at any height.
- 14:56:22 to 17:08:55, the 58: all ok, no HEX-E time-out. `tall-edge-626` met 0.31 N in its fast approach. The
  program backed off and finished it at its slow speed.
- 17:09:42 to 17:41:06, the tall morning plan: 18 of 18 ok, joint 6 up to +57 deg with the owner present.
- Between runs, each next plan started only after an exit 0 with no ABORT line, a clean home path check and a reached
  home. A small script did these steps from 13:45 on.

### The owner's decisions on 10-28

- "ok reseated can we gogogogo?", and through the workstation "sure the remaining sphere twists sound good": the
  check, then the twists.
- "let's run the rod block now yea": rod pass 1 and stage 2 now.
- Through the workstation at 13:25: the rod fixture stays on and both held tips stay up, so the side map is off for
  now. The tall plans come next, and the top edge may run with the owner present. The lab asked for the owner's go in
  its own session for the top edge.
- Through the workstation at 13:42 and 14:49: today's runs end before 19:00. Then the owner swaps v3 ID 1 for the next
  four-magnet tip, and the night suite starts on it.

## 2026-10-28 evening: the top edge, the swap and v3 ID 2

### The tall top edge's first 8 (18:03 to 18:11)

- 18:03:40, `lab/20261028-plan_tall_lab_top_edge_first8_v3id1.json` resumed on 114908 with the owner watching:
  top-ref-a, top-tip-401 and top-tip-459 ran ok, 33 to 50 s each. The owner stopped it there: "i think you should do
  the top edge calbiration on the lower sphere, we are close to joint limits on the tall indenter"; "ohterwise it looks
  good".
- From the last pose the home path check refused, with the cap 1.6 mm from the tall. The owner moved the arm in teach
  mode; the check then passed and the arm went home at 18:11.

### The sphere's top edge, built on v3 ID 1 (18:21 to 18:49)

- The workstation's recipe (d096883d) on 225213's registration: `scripts/tall_indenter.py reach --kind sphere` found
  113 reachable points on the tip's end and edges, 29 after the cover and the thinning.
- At the builder's 20 mm lift, the path check flagged the travel from the face-centre reference into the first top
  press. The joint move swept the fork 3.88 mm into the sphere's model, at a flange height of 158 mm. The 20 mm lift
  was the owner's request for the tall. At the sphere plans' 60 mm lift the mount keeps 5.36 mm.
- Three presses (tip-398, -399, -401) put the cap 0.60 to 0.78 mm from the sphere at their depth, under the finger's
  0.8 mm. A path check of all 29 candidates left the flagged ones out (scratchpad `select_te.py`); 10 were kept, nearest
  a limit first.
- The proxy plan on 225213: 12 actions, nothing flagged. Simulated from home: no abort, the mount 5.36 mm and the
  finger 2.41 mm from the cell, joint 6 from -51.5 to +6.4 deg, the elbow at least 68.7 deg from straight. It is
  `lab/20261028-plan_sphere_lab_top_edge_v3id1.json`, with its reach and path check.
- The owner asked whether the checks use MuJoCo. They do not: `scripts/ur5_cb3_cell.py` samples points on the UR5's
  URDF collision meshes and the tool's CAD meshes, and measures them against signed distances of the fixtures at their
  registered poses, the plywood and the table.

### The swap

- At about 18:40 the owner put v3 ID 2 on the robot, from the holder's right-side station, and v3 ID 1 into that
  station pad-up ("yes it went pad up"). The mount stayed on the flange. The robot's board is still
  `b4:3a:45:b0:9c:d0`.
- The owner's go in the lab's session, 18:3x: "you have the go ahead on the tip top edge plan at the end of the runs
  if we make it there".

### Block 1: ID 2's sphere pass 1 from a seed (18:44:15 to 19:08:55)

- `logs/20261028-tip_calibration/184415-spherev3id2` (copy: `lab/20261028-184415-spherev3id2/`): a seed of 225213's
  registration, tool and fixture with the homing pose, from `scripts/ur5_tip_calibration_seed_registration.py`.
  Resumed with stages preflight,pad,register on `lab/20261028-plan_sphere_lab_pass1_night_v3id2.json`.
- 18 of 18 touch-offs, no abort, no early contact. Every onset came 0.18 to 0.50 mm later than 225213's registration
  expected. Pad touches 0.044 mm rms (225213: 0.049). The TCP moved 0.373 mm and 1.27 deg, the sphere's centre
  0.69 mm. The seeded simulations had recovered moves of 0.6 mm and 1.5 deg.
- A touch-off took 81 s on average, 47 s of it in the slow approach: 0.15 mm/s in the plan, 0.045 mm/s at the plan's
  speed slider of 0.30. The owner: "can you go faster on the approaches. this is excruciating". The lab offered to
  simulate a faster touch-off approach or a higher slider; nothing has changed yet.

### The sphere top edge on ID 2 (19:21:11 to 19:30:14)

- The owner, 18:4x: "if the top edge is ready to run on the lower sphere, then let's do that while im still here".
- The same build on 184415's registration (scratchpad `te_after_b1.sh`): 4 of the 29 candidates flagged. tip-398 and
  tip-399 put the cap 0.79 and 0.61 mm from the sphere. edge-520 and edge-643 took joint 6 to -63 deg, past the path
  check's -62. 10 kept. Path check of the plan: nothing flagged, the cap 0.84 mm from the sphere at tip-401, the mount
  5.36 mm. Simulated from home: no abort and one early contact, which the program handled. The mount 5.39 mm and the
  finger 1.50 mm from the cell, joint 6 from -52.5 to +6.4 deg, joint 5 11.1 deg inside its window, the elbow 68.3 deg
  from straight.
- `lab/20261028-plan_sphere_lab_top_edge_v3id2.json` resumed on 184415, stages preflight,grid: 12 of 12 ok (the two
  references and nine tip presses to 2 N, edge-508 to 3 N), no abort, no early contact. The HEX-E's two gaps over 0.5 s both came at the
  start, before the first action.
- The owner watched it: "ok its looking good. godspeed". They left at about 19:25: "worst case i can come back at
  ~9:30 to reseat".

### Block 3: ID 2's tall pass 1 (20:05:08 to 20:39:27)

- Simulated first from each of the tall's three seatings on ID 2's tool (block 1's registration), the tall where 114908
  put it. The pattern repeated v3 ID 1's. Seat 2 brings the mount 2.13 mm from the tall at pad-face-05 and -08, under
  the touch-offs' 2.5 mm. Seats 1 and 3 give 2.88 and 3.25 mm. In all three, joint 6 reaches -64.3 deg.
- `logs/20261028-tip_calibration/200508-tallv3id2` (copy: `lab/20261028-200508-tallv3id2/`): a new session homed on
  `tall_indenter_seated`, as 114908 was, stages preflight,homing,pad,register. The homing put the indenter at (-33.3,
  -699.2, 72.4) mm, yaw -93.4 deg.
- 19 of 19 touch-offs, no abort, no early contact. Onsets -0.11 to +1.18 mm, rising through the pass as 114908's did
  (-0.41 to +0.96). Pad touches 0.073 mm rms (114908: 0.045). The TCP moved 1.35 mm and 2.91 deg from the plan's
  nominal tool, and the indenter's centre 1.85 mm.
- This tool and block 1's sphere tool differ by 0.34 mm and 1.13 deg (v3 ID 1's two: 0.24 mm and 2.12 deg).
- The fitted tall stands 1.85 mm along -y_B from 114908's. The pass does not fix y well: the simulations from seats 1
  and 3, with the tall exactly at 114908's, fitted it 1.54 and 1.30 mm along -y. So the tall need not have moved.
- The robot stood still from 19:30 to 20:05 while the last chunk of block 2's simulation was checked. Block 3 then ran
  before block 2, and the order became 3, 2, 5, 4, 6.

### Block 2: ID 2's sphere stage 2 (20:40:16 to 03:14:45)

- `lab/20261028-plan_sphere_lab_stage2_night_v3id2.json`, resumed on 184415 (printout: `printout_stage2_2040.txt` in
  its copy). Its four chunks were simulated from home on block 1's registration. Least clearances: the mount 3.19 mm
  and the finger 1.46 mm. Joint 6 ran from -63.3 to +42.5 deg, joint 5 stayed 5.3 deg inside its window, and the elbow
  at least 90 deg from straight.
- 211 of 211 ok, no abort, no early contact. Grid 116 (20:40 to 22:18), shear 77 (to 02:28), roll 2, twist 12 and
  final 4. It took 6.6 h, against 6.1 h in the suite's estimate.
- The HEX-E: 2.36 million samples, no time-out. Its two gaps over 0.5 s both came at the start, before the first
  action.
- At 2 N, the face-centre reference stopped 2.23 mm along its approach at 20:41 and 20:49. From 21:01 to 21:22 it
  stopped at 2.32 to 2.35 mm, and at 03:13 at 2.35 to 2.38 mm. So the stop moved 0.10 mm deeper in the first 20 minutes,
  then held within 0.05 mm for six hours. Sideways it moved 0.03 mm at most.

### Block 5: ID 2's rod pass 1 from a seed (03:15:59 to 03:57:13)

- Simulated from home on a seed of block 1's tool and 102701's rod: no abort, the mount 3.37 mm and the finger 2.54 mm
  from the cell, joint 6 from -52.4 to +25.8 deg, joint 5 3.6 deg inside its window, the elbow at least 72 deg from
  straight.
- After block 2's exit 0, the scratchpad script `b5_after_b2.sh` checked the home path from 184415's last pose, went
  home (0.04 mm from the target) and made the seed `logs/20261029-tip_calibration/031559-rodv3id2` (copy:
  `lab/20261029-031559-rodv3id2/`). Resumed there with stages preflight,pad,pivot,register on
  `lab/20261028-plan_rod_lab_pass1_night_v3id2.json`, seat `rod_indenter_seated_3`.
- 21 of 21 touch-offs and 5 pivot presses ok, no abort, no early contact. Every onset came 0.51 to 0.69 mm later than
  the seed expected.
- The touch-off searches 1.2 mm past the expected surface (`beyond_mm`). From the onset to its 0.5 N stop it pressed
  0.14 to 0.34 mm more, so the stops came 0.27 to 0.37 mm short of that limit.
- Pad touches 0.023 mm rms (v3 ID 1 on the rod: 0.042). The fit moved the tool 0.57 mm and 0.86 deg from block 1's
  sphere tool, and the rod 0.16 mm from 102701's. So the late onsets came from the tool as this rod sees it, and the
  rod did not move.
- The pivot stage kept no turn presses, so `pivots.json` is empty, as in 102701.
- The HEX-E: no time-out, and one gap over 0.5 s at 03:16:00, before the first action.

### Block 4: ID 2's tall stage 2 (03:58:45 to 06:44:53)

- `lab/20261028-plan_tall_lab_stage2_night_v3id2.json`, resumed on 200508 with stages preflight,grid,shear,twist,final
  (printout: `printout_stage2_0358.txt` in its copy). Its two chunks were simulated from home on block 3's
  registration. The mount stayed at least 3.18 mm from the cell. Joint 5 came to 99.2 deg at tall-shared-576 and -640,
  and the program aborts an action past 100.
- After block 5's exit 0, `night_rest.sh` started block 6's simulations on block 5's registration (two chunks, nice
  19). It then checked the home path from 031559's last pose and went home (0.04 mm from the target), and block 4
  started at 03:58:45.
- Block 6's two chunks passed the gate at 04:14:53: no abort, the mount at least 3.18 mm and the finger 1.42 mm from the
  cell, joint 6 from -56.1 to +37.3 deg, the elbow at least 66.6 deg from straight. Joint 5 came 0.6 deg inside its
  window, at 30.6 deg.
- 102 of 102 ok, no abort, no early contact. Grid 56 (03:58:49 to 04:40:06), shear 41 slides (to 06:33:51), twist 4
  (to 06:44:13) and final 1. It took 2.77 h, against 2.8 h in the suite's estimate.
- On the robot, joint 5 reached 99.1 deg at tall-shared-576 and -640, as the simulation had it.
- The HEX-E: 995,507 samples, no time-out. Its two gaps over 0.5 s (1.2 and 0.7 s) came at 03:58:45 to 03:58:47,
  before the first action.
- At 2 N, the reference press stopped 2.31 mm along its approach at 03:59, 2.21 and 2.23 mm at 04:39 and 04:40, and
  2.23 mm at 06:44. So the stop came 0.08 to 0.10 mm earlier after the grid's 40 minutes, then held for two hours.
  Sideways it moved 0.05 mm at most. Block 2's sphere reference moved 0.10 mm the other way, deeper, in its first 20
  minutes.

### Block 6: ID 2's rod stage 2 (06:45:40 to 08:13:17)

- `lab/20261028-plan_rod_lab_stage2_night_v3id2.json`, 100 actions, resumed on 031559 with stages
  preflight,grid,shear,roll,twist,final (printout: `printout_stage2_0645.txt` in its copy). Its simulations are in block
  4's section.
- After block 4's exit 0, `night_rest.sh` checked the home path from 200508's last pose and went home (0.02 mm, 0.01
  deg from the target). Block 6 started at 06:45:40, and its preflight read the HEX-E with no time-out.
- 100 of 100 ok, no abort, no early contact. Grid 69 (06:45:44 to 07:42:22), shear 24 slides (to 08:01:47), roll 4 (to
  08:08:26), twist 2 (to 08:12:25) and final 1. It took 1.46 h, against 1.4 h in the suite's estimate.
- On the robot, joint 5 came down to 30.5 deg at rod-twist-019-2.5N and -5N, 0.5 deg inside its window; the simulation
  had 30.6. Joint 6 ran from -55.4 to +37.8 deg, and the elbow stayed at least 66.5 deg from straight.
- The HEX-E: 524,846 samples, no time-out. Its one gap over 0.5 s (0.7 s) came at 06:45:41, before the first action.
- At 2 N, the reference press stopped 2.11 mm along its approach at 06:46, 2.09 and 2.11 mm at 07:36 and 07:37, and 2.14
  mm at 08:13. So on the rod the stop held within 0.03 mm, and sideways within 0.02 mm.

### The night's suite on ID 2: done

- Blocks 1 to 6 and the sphere's top edge: 488 actions, all ok, no abort, no early contact and no HEX-E time-out.
- They took 12.6 h of robot time between 18:44:15 and 08:13:17. The suite estimated 11.7 h; block 2 alone ran 0.5 h
  over.
- The arm stays at the rest pose of block 6's last press, above the rod. It does not move until the owner says so.

### ID 2's analyses, run here for the workstation (08:19 to 10:28)

- The workstation's memory gate was closed (swap over 15 GB), and the session streams are here. So the lab ran its
  commands on corvus while the robot stood idle: `scripts/ur5_tip_calibration_analysis.py --case four_05R4
  --gravity-from lab/20261022-195420-first2` on each session, and `scripts/ur5_tip_calibration_four_vs_five.py
  --v3-key four_v3id2` against v2 ID 1. Each ran alone at nice 19 with OMP_NUM_THREADS=2 (scratchpad
  `id2_analyses.sh`, `id2_queue.sh` and `id2_queue2.sh`).
- Sphere (184415): 30 min, 5.1 GB peak, 2769 holds. Its four vs five took 53 min, mostly refitting the 77 slides.
- At 09:47 the workstation asked for the new fixtures first. The `--ref-drift` run had been going 6 min; it was stopped
  by PID and run again after the rod.
- Tall (200508): 7.8 min, 2.4 GB, 1343 holds. Rod (031559): 2.9 min, 1.6 GB, 668 holds. Their four vs five took 1 s
  and 30 s.
- Sphere with `--ref-drift`: 28 min. It applied 49 reference presses, and the summary did not change past the third
  decimal.
- The outputs are in `logs/20261029-{spherev3id2,spherev3id2_refdrift,tallv3id2,rodv3id2}/` and
  `logs/20261029-four_vs_five_{grid,tall,rod}_v3id2.json`, with the printouts in `logs/20261029-*-analysis-*.log`.
  The workstation copied them at 10:29. Its results are in the v3 feed at 09:46, 10:00 and 10:28, and in the README's
  section "v3 ID 2 against v3 ID 1" (`docs/experiments/20261027-v3_tip_calibration_and_crosstalk/`).
- In brief, from those: on each fixture's shared presses the direction GP gives 0.29 / 0.38 mm for ID 2 / ID 1 on the
  sphere, 0.36 / 0.36 on the rod and 1.06 / 0.95 on the tall. On ID 2, the five-magnet gain is 1.83 to 1.87 times the
  four's. Both tips' boards sit 0.75 and 0.98 mm along -y of the tall's registration at 3 N and over, so the tall's
  larger errors come from the tall indenter, not from ID 2.

### Faster touch-offs: where the time goes and what speed costs

- The owner, 10-28 at about 19:10: "can you go faster on the approaches. this is excruciating".
- Where the time goes (`touch_times.py` here): a touch-off took 81 s on the sphere (block 1), 108 s on the tall
  (block 3) and 106 s on the rod (block 5). The slow approach took 47, 75 and 75 s of that. It runs at the plans' v
  0.00015 m/s, which is 0.045 mm/s at the slider's 0.30. It covers `slow_mm_touch` (1.5 mm on the sphere, 2.5 mm on
  the tall and the rod), the late onset and 0.1 to 0.3 mm more to the 0.5 N stop. The travel took about 20 s and the
  return 8 to 11 s.
- What speed costs (`onset_vs_speed.py` here, the workstation's design; rows in `20261029-onset_vs_speed_184415.json`).
  Block 2's 116 grid presses ran at five plan speeds, measured at 0.042 to 1.5 mm/s. Each press's onset was fitted as
  the touch-offs' are, up to 0.25 N, and compared with 184415's registration (later contact positive). The touch-offs
  through the same residual give back the fit's 0.044 mm rms.
- The reference press stopped 0.10 mm deeper over block 2's first 20 minutes, and the rate presses all ran from 21:30
  to 22:18. So each was compared with the 0.144 mm/s presses run within 10 minutes of it.
- The 5 presses at 0.042 mm/s against those neighbours: -0.000 mm (SE 0.006). A fit of lag times speed plus a drift
  over the 108 presses up to 0.58 mm/s gives a lag of -47 ms (SE 40), so -0.005 mm (SE 0.004) for the step from 0.045
  to 0.15 mm/s. The registration's own rms is 0.044 mm.
- The fit's window matters more: refitted up to 0.25 N instead of 0.4 N, the touch-offs moved 0.015 mm earlier.
- At 0.58 mm/s the presses came 0.026 mm earlier (SE 0.017). At 1.2 and 1.5 mm/s they came 0.29 and 0.16 mm earlier.
  There fit_onset's 25-sample average spans 0.3 to 0.4 mm of travel, so the fitted onset lands before the rise: the
  push passed 0.15 N 0.29 to 0.34 mm after it, against 0.08 mm at the slow speeds. That comes from the fit, not from
  the HEX-E.
- So the touch-offs' slow approach could run at v 0.0005 (0.15 mm/s at the slider), as the presses already do. A
  touch-off would take about 48 s on the sphere and 55 s on the tall and the rod; rod pass 1's 21 would take 19 min
  instead of 37. The choice is the owner's, and the plans are unchanged.

### The owner's answers (11:37, passed on by the workstation at 11:49)

- Touch-offs: "we can approach at 0.15mm/s sure, or faster if you want". The next tip's plans run the touch-offs' slow
  approach at v 0.0005. The workstation advised against going faster: the 0.58 mm/s presses' onsets came 0.026 mm
  early (SE 0.017).
- The owner asked "do you think i should put on v3 id 3 first?", and the workstation recommended it, on ID 2's suite.
  It is not decided, and the go comes in the lab's session.
- The rod stays: "i am not going to move the rod. figure something else out." So the side map without the rod
  (`20261028-plan_holder_lab_side_norod30_v3id1.json`) is off. The workstation is building a side map with both held
  tips pad-up.
- Indenter plate v2d waits until the motions calibration needs are final. The workstation is writing that list.
- `STATE.md` carries the details.

### ID 3's suite and the pad-up side map, checked here (12:05 to 13:36)

- The workstation's plans of 12:00 (62d93424, d6795072): ID 3's six night plans are ID 2's 476 actions, action by
  action, with the 58 touch-offs at v 0.0005. Block 1's seed, `lab/20261029-sphere-seed-v3id3`, is 184415's
  registration exactly. The side map, `lab/20261029-plan_holder_lab_side_up_v3id2.json`, keeps the rod on and both
  held tips pad-up. It runs on the holder seed `lab/20261029-holder-seed-seat5-v3id2`: 184415's tool with
  holder_seated_5's frame and homing.
- Simulated from home at nice 19 (scratchpad `i3_sims.sh`, 12:05 to 12:38). No run aborted or made an early contact.
  - Block 1 ran four times, resumed on the seed. The first run kept the true pad where the seed puts it. The other
    three moved it by 0.026 rad and 0.6 mm, as for ID 2's block 1. The fits recovered the moves: the TCP moved 0.65 to
    0.77 mm, with pad rms 0.013 to 0.029 mm. The fork stayed at least 3.63 mm from anything and the finger 2.50 mm.
    Joint 6 went from -63.0 to 40.7 deg, and joint 5 from 47.1 to 82.2 deg.
  - Block 3 ran fresh from each of the tall's three seatings, with 184415's tool and 200508's tall. The fork stayed
    4.74, 4.00 and 5.12 mm away, and the finger at least 1.50 mm. Joint 6 went down to -64.4 deg and joint 5 up to
    98.4 deg.
  - Block 5 ran on a seed of 184415's tool and 031559's rod. The fork stayed 3.37 mm away and the finger 2.57 mm. Joint
    6 went from -52.4 to 25.8 deg.
  - Rechecked on every 2nd sample (12:47 to 13:35, `fine_checks.sh`): no clearance changed by more than 0.04 mm.
- Blocks 2, 4 and 6 are ID 2's stage 2 plans action by action, and those ran 413 of 413. As for ID 2, each is
  simulated on ID 3's own pass 1 registration on the night. Their cell check now takes every 5th sample, not every
  20th; every 20th spans 5 to 10 mm of fast travel.
- The side map's first version had 28 hovers. Four of them came under the mount's 3.0 mm: g14-s2 at 2.41 mm and g14-s0
  at 2.73 mm, with their repeats. There the fork passes the holder 5 to 7 mm above the hover pose, on the straight
  descent from the reference.
- The workstation's path check had seen 3.19 mm there. It sampled straight moves every 2 mm but kept every 5th sample,
  so it saw them every 10 mm. Its own moves at 0.5 mm steps give 2.42 and 2.73 mm (`hover_descent.py` here). The
  workstation dropped the four hovers and now checks every sample of a straight move (bb7c5578).
- The 24-hover version, simulated from home: no abort. The fork stayed at least 3.01 mm away (g20-s5), the finger 3.00
  mm and the links 10.13 mm. Joint 6 went from -3.5 to 6.4 deg.
- The workstation's memory gate was closed, so its seven path checks ran here at bb7c5578, two at a time at nice 19
  (`path_rechecks.sh`). The outputs are in `logs/20261029-recheck_v3id3/`.
  - Side map: 0 flagged, the fork at 3.11 mm.
  - Sphere pass 1: 2 flagged, with joint 6 1.0 deg past -62.
  - Tall pass 1: 5 flagged on the joint windows, with joint 6 3.7 deg past -62 and joint 5 1.7 deg past 100. In the
    simulations joint 6 stays above -64.4 deg and joint 5 below 98.4 deg, and ID 2's runs of the same actions passed.
  - Rod pass 1: 0 flagged, the fork at 3.77 mm.
  - Rod stage 2 with measured depths: rod-ref-a at 2.96 mm. This is new with the finer sampling. The workstation (13:37)
    traced it to the checker's travel from its own start pose: the press itself clears 4.98 to 5.51 mm, as on rod-ref-b,
    -c and rod-final-ref. On the night, block 6 starts from block 5's last pose. If block 6's simulation then puts the
    fork under 3.0 mm on rod-ref-a, rod-ref-a is dropped and rod-ref-b becomes the first reference.
  - The tall's and the sphere's stage 2 are still running.
- At 12:07, a read-only check of the arm's home path from block 6's last rest pose: the fork stays at least 10.7 mm
  from anything. At 12:31 the joint temperatures read 29.0 to 40.9 C.
- Before a power-off or a swap, the arm goes home, on the owner's go here. The scripts for the next steps wait in the
  scratchpad: `go_home_now.sh`, `block1_id3.sh` and `te_after_b1_id3.sh`. The first two stop unless the speed slider
  reads 0.30 (`slider_check.py`). move_guarded.py does not set the slider, and a power cycle may reset it.

## 2026-10-29 afternoon: v3 ID 3

### The swap

- At about 13:3x the owner put v3 ID 3 on the robot and ID 2 into the station ID 3 left, pad-up. The fork stayed on
  the flange ("yes the fork stayed on"). The robot's board is still `b4:3a:45:b0:9c:d0`.
- Before the swap, the owner moved the arm by hand from block 6's last rest pose above the rod. It ended at TCP
  (-114.6, -355.5, 220.9) mm, in free air, and the slider still read 0.30.
- The owner's go in the lab's session, for block 1: "lets a go mario". Then, for the rest of the day: "1. top edge --
  lets go for it. add more density if anything 2. side map -- sure 3. yea send it".

### Block 1: ID 3's sphere pass 1 from a seed (13:45:08 to 13:59:02)

- The home path from the owner's pose was clear: the fork stayed at least 126 mm from anything. Go home reached its
  target within 0.04 mm.
- `logs/20261029-tip_calibration/134508-spherev3id3` (copy: `lab/20261029-134508-spherev3id3/`) is a copy of the seed
  `lab/20261029-sphere-seed-v3id3`: 184415's registration with its homing pose. It was resumed with stages
  preflight,pad,register on `lab/20261029-plan_sphere_lab_pass1_night_v3id3.json`.
- Preflight: HEX-E 100 Hz, ID 3's board 340 records/s, robot 124 Hz, slider 0.30, payload 0.403 kg.
- 18 of 18 touch-offs, with no abort and no early contact. The onsets came -0.143 to +0.120 mm from where 184415's
  registration expected them. Pad touches 0.044 mm rms, as on ID 2's block 1. The TCP moved 0.392 mm and 0.957 deg
  from ID 2's tool, and the sphere's centre 0.444 mm.
- A touch-off took 45.6 s on average (35 to 58 s) at v 0.0005, against 81 s on ID 2's block 1 at v 0.00015.
- The HEX-E gave 82,737 samples, none timed out, with two gaps of 0.7 s.

## The lab clone and git-lfs

The rebase at 19:21 replaced the plan, `poses.jsonl`, `moves.jsonl` and the images of this folder with git-lfs pointers,
because `.lfsconfig` skips LFS content. The program then failed to read the plan (19:26, nothing moved). The files came
back from the local LFS store, identical to the originals in `logs/20261022-ur5_homing_by_hand/`. This clone now has
`lfs.fetchexclude` cleared in `.git/config`, as the workstation's has, and `tipcal_run.sh` refuses to start on a pointer.
