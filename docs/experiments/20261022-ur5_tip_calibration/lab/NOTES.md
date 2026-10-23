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

## The lab clone and git-lfs

The rebase at 19:21 replaced the plan, `poses.jsonl`, `moves.jsonl` and the images of this folder with git-lfs pointers,
because `.lfsconfig` skips LFS content. The program then failed to read the plan (19:26, nothing moved). The files came
back from the local LFS store, identical to the originals in `logs/20261022-ur5_homing_by_hand/`. This clone now has
`lfs.fetchexclude` cleared in `.git/config`, as the workstation's has, and `tipcal_run.sh` refuses to start on a pointer.
