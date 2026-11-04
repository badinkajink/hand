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
  0.14 to 0.36 mm more, so the stops came 0.15 to 0.44 mm short of that limit. (Corrected at 23:12 on 10-29; this line
  said 0.27 to 0.37 mm.)
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

### The sphere's top edge on ID 3 (14:49:59 to 15:11:45)

- Built on block 1's registration (scratchpad `te_after_b1_id3.sh`). The path check passed 24 of v3 ID 1's 29
  candidates; it flagged sph-top-tip-398, -399 and -401 (finger 0.54 to 0.69 mm) and sph-top-edge-520 and -643
  (joint 6 0.5 deg past -62). The owner asked for density ("add more density if anything"), so all 24 were kept, where
  ID 2 kept 10.
- A mistake caught before anything moved: the first build raised the builder's `--max` from 10 to 29, and the builder
  then filled the plan up to 29 with the 5 flagged candidates. Its path check flagged them (31 actions, 5 flagged). The
  build was stopped at 14:20 and rebuilt with `--max 24`, the number kept (`te_rebuild_id3.sh`). The script now sets
  `--max` from the kept count.
- `lab/20261029-plan_sphere_lab_top_edge_v3id3.json` (527d2390): 24 presses with a reference before and after. Its
  path check: 26 actions, 0 flagged, the mount at 4.91 mm and the finger at 1.69 mm at least. Simulated from home in 3
  chunks: no abort, no early contact, the mount 5.11 mm and the finger 2.26 mm at least, joint 6 from -53.3 to 38.5 deg.
- Run on the owner's go, resumed on block 1's folder with stages preflight,grid (copy:
  `lab/20261029-134508-spherev3id3/plan_top_edge_1449.json`, `printout_top_edge_1449.txt`). 26 of 26, all ok, no abort,
  exit 0 at 15:11:45. A press took 49.9 s on average (33 to 62 s). The HEX-E gave 129,962 samples, none timed out,
  with gaps of 1.0 and 0.7 s.

### The side map on ID 3 (from 15:13:34)

- The workstation's commands of 13:49, run here after block 1 (`side_id3.sh`): the holder seed
  `lab/20261029-holder-seed-seat5-v3id3` (block 1's tool, seat 5's holder frame and homing), the plan
  `lab/20261029-plan_holder_lab_side_up_v3id3.json` (12 poses and their repeats, without g13 and g14) and its path
  check at every 2 mm sample: 24 actions, 0 flagged, the mount at 3.05 mm and the finger at 3.02 mm at least.
- Simulated from home on the seed: no abort, the mount 3.05 mm at least on every 10th sample (g20-s5), joint 6 from
  -3.7 to 6.4 deg. Every 10th sample spans under 1 mm of the hovers' slow descents. The run started without waiting for
  the recheck on every 2nd sample, which had run 32 min by then; on ID 2's side map that recheck changed nothing.
- Session `logs/20261029-tip_calibration/151334-sidev3id3` (copy: `lab/20261029-151334-sidev3id3/`; the streams,
  `held.bin` among them, stay on corvus), stages preflight,hover, with the held tips' mux board (`--held-board`):
  channels 0 and 3 at 206 records/s each.
- 24 of 24 hovers, no abort, exit 0 at 15:29:14. Each held 6 s; the HEX-E's largest change in a hover was 0.23 to
  0.44 N, the magnets' pull. The HEX-E gave 93,426 samples, none timed out, with two gaps of 0.7 s.

### The night's chain (from 15:29)

- `id3_chain.sh` took over at the side map's exit. Block 3's simulation from seat 1 on block 1's tool had passed: the
  mount 4.77 mm and the finger 1.59 mm at least, joint 6 down to -64.3 deg, joint 5 up to 98.3 deg. Block 5's on its
  seed: the mount 3.16 mm, joint 6 from -52.2 to 24.7 deg.
- Block 3 (tall pass 1) started at 15:30:34 after the home path check and go home (0.05 mm), a new session homed on
  `tall_indenter_seated`. The homing put the tall at (-33.3, -699.2, 72.4) mm, yaw -93.4 deg, as for ID 2.
- The side map's recheck on every 2nd sample ended at 15:44: 0 under a margin, the mount 3.05 mm at least, the same as
  on every 10th.

### Block 3: ID 3's tall pass 1 (15:30:34 to 15:48:00)

- `logs/20261029-tip_calibration/153034-tallv3id3` (copy: `lab/20261029-153034-tallv3id3/`), stages
  preflight,homing,pad,register.
- 19 of 19 touch-offs, no abort, no early contact. The onsets came -0.253 to +1.288 mm from the homed tall and rose
  through the pass, as ID 2's did (-0.11 to +1.18 mm); the touch-offs search 2.0 mm past. A touch-off took 54.5 s on
  average (47 to 71 s), against about 108 s on ID 2's block 3.
- Pad touches 0.055 mm rms (ID 2: 0.073). The fitted tall stands 1.46 mm along +y_B from 200508's, so 0.39 mm along -y
  from 114908's; the pass does not fix y well.
- The HEX-E gave 103,927 samples, none timed out, with gaps of 0.8 and 0.7 s.
- At its exit the chain started block 4's simulations on this registration (2 chunks, every 5th sample) and waited for
  block 2's chunks.

### Block 2: ID 3's sphere stage 2 (16:34:16 to 23:07:05)

- `lab/20261029-plan_sphere_lab_stage2_night_v3id3.json`, resumed on 134508 with stages
  preflight,grid,shear,roll,twist,final (printout: `printout_stage2_1634.txt` in its copy). The chain checked the home
  path and went home first (0.02 mm from the target).
- Its 4 chunks passed at 16:33:28, simulated on block 1's registration and checked on every 5th sample: the mount
  3.56 mm and the finger 1.37 mm at least, joint 6 from -63.1 to 42.0 deg, joint 5 6.0 deg inside its window at least.
  Each chunk's check took about 1.5 to 2 h on every 5th sample, so the arm waited 46 min after block 3.
- 211 of 211 ok, no abort, no early contact, and every press reached its force. Grid 116 (16:34 to 18:11), shear 77
  (to 22:23), roll 2, twist 12 and final 4. It took 6.55 h; ID 2's took 6.6 h.
- On the robot, joint 6 ran from -62.8 to +42.3 deg, and joint 5 came 6.1 deg inside its window, as the simulation
  had it. The elbow stayed at least 89 deg from straight.
- The HEX-E: 2.35 million samples, no time-out. Its one gap over 0.5 s (0.7 s) came at 16:34:17, before the first
  action.
- At 2 N, the face-centre reference stopped 2.19 to 2.32 mm along its approach in the grid's ten reference presses
  (16:35 to 18:12, mean 2.26 mm). At 23:05 to 23:07 it stopped at 2.29 to 2.35 mm (mean 2.31 mm). Sideways it moved
  0.05 mm at most. ID 2's reference stepped 0.10 mm deeper in its first 20 minutes. ID 3's scattered over 0.12 mm
  with no step, and ended 0.05 mm deeper after the shear's four hours.

### Block 5: ID 3's rod pass 1 from a seed (23:08:21 to 23:31:41)

- Simulated from home on a seed of block 1's tool and 031559's rod: no abort, the mount 3.16 mm and the finger 2.30 mm
  from the cell, joint 6 from -52.2 to +24.7 deg, joint 5 3.8 deg inside its window, the elbow at least 71 deg from
  straight.
- After block 2's exit 0, the chain checked the home path from 134508's last pose and went home (0.05 mm from the
  target). It made the seed `logs/20261029-tip_calibration/230820-rodv3id3` (copy: `lab/20261029-230820-rodv3id3/`)
  and resumed there with stages preflight,pad,pivot,register on `lab/20261029-plan_rod_lab_pass1_night_v3id3.json`,
  seat `rod_indenter_seated_3`.
- 21 of 21 touch-offs and 5 pivot presses ok, no abort, no early contact. Every onset came 0.58 to 0.79 mm later than
  the seed expected (ID 2: 0.51 to 0.69 mm).
- From the onset to the 0.5 N stop the touch-offs pressed 0.19 to 0.39 mm more. So the stops came 0.11 to 0.39 mm
  short of the search's limit, 1.2 mm past the expected surface (`beyond_mm`), and none reached it. ID 2's came 0.15
  to 0.44 mm short, the closest at pad-face-02.
- A touch-off that reaches the limit before 0.5 N does not abort. The program records the end of its travel and fits
  the onset to the force it has, up to 0.4 N. Only a touch with no contact onset aborts, after it backs off to its
  rest pose.
- Pad touches 0.037 mm rms (ID 2 on the rod: 0.023; ID 1: 0.042). The pivot stage kept no turn presses, so
  `pivots.json` is empty, as for ID 1 and ID 2.
- The fit moved the tool 0.20 mm and 0.57 deg from block 1's sphere tool. It moved the rod 0.59 mm from 031559's, 0.53
  mm of it down. This rod now sits 0.67 mm from 102701's, 0.54 mm lower; ID 2's pass kept it within 0.16 mm.
- ID 2's pass put its late onsets in the tool (0.57 mm), and this pass puts them in the rod. Either the rod sat 0.5 mm
  lower tonight, or the pass hardly tells the two apart. Its 21 touch-offs all approach within 10 deg of straight down,
  and along that line a lower rod and a longer tool look alike.
- Block 6 runs on this registration, and so do its simulations. If the other split is the true one, a press tilted by
  an angle from straight down lands 0.55 mm times its sine sideways of its target. That is 0.10 mm at 10 deg, the
  pass's widest tilt, and 0.15 mm with the rolls' 6 deg more. The mount's 3.0 mm margin covers that.
- The HEX-E: 139,361 samples, no time-out. Its two gaps over 0.5 s (0.8 and 0.7 s) came at 23:08:21 and 23:08:22,
  before the first action.

### Block 4: ID 3's tall stage 2 (23:33:18 to 00:08:00, stopped by a HEX-E loss)

- `lab/20261029-plan_tall_lab_stage2_night_v3id3.json`, 101 actions, resumed on 153034 with stages
  preflight,grid,shear,twist,final. Printout `logs/20261029-233318-tipcal-tallv3id3.log`.
- Its two chunks were simulated from home on block 3's registration and checked on every 5th sample. The mount stayed
  at least 3.73 mm and the finger 1.61 mm from the cell. Joint 6 ran from -63.6 to +44.1 deg, and the elbow stayed at
  least 67 deg from straight. Joint 5 came to 99.4 deg, 0.6 deg inside its window; ID 2's block 4 had 99.2 deg in the
  simulation and 99.1 on the robot.
- After block 5's exit 0, the chain started block 6's simulations on block 5's registration (two chunks, every 5th
  sample). It checked the home path from 230820's last pose and went home (0.02 mm, 0.01 deg from the target), and
  block 4 started at 23:33:18.
- 46 of 101 ok, all in the grid (tall-ref-a to tall-edge-694, 23:33:22 to 00:07:37), no early contact.
- At 00:07:49.67 the travel into tall-edge-626 ended, and 30 ms later the program had no HEX-E sample for 100 ms.
  tall-edge-626, -668 and -670 aborted, and the session stopped after three aborts in a row (exit 1 at 00:08:00). The
  HEX-E gave 206,752 samples, 158 of them timed out. Its two gaps over 0.5 s came at 23:33:19 and 23:33:20, before the
  first action.
- In that travel wrist 1 swung from -36.6 to +19.3 deg, wrist 2 rose from 54.2 to 90.5 deg and the elbow came from 98.9
  to 74.0 deg; joint 6 moved only from 30.1 to 31.4. The losses at 02:54 and 14:32 on 10-28 had the same signature:
  wrist 1 from negative to positive with wrist 2 rising, the loss as the robot came to rest.
- Three of the HEX-E's losses now fall within five travels of tall stage 2's grid, whose order is the same in every
  tip's plan. They came in the travels into tall-face-702 (R29, 17:06 on 10-27), tall-edge-627 (14:32 on 10-28) and
  tall-edge-626 (tonight). ID 2's block 4 passed through the stretch without a loss.
- After the first two aborts the program lifted the pad 5 mm each time. The arm stands with the pad about 12 mm above
  tall-edge-626's point (its 1.5 mm standoff and the two lifts), joint 5 at 90.5 deg and joint 6 at 31.4 deg.
- At 00:10 the DAQ answered a ping, but a 10 s read gave no samples, as after the earlier losses. The chain stopped at
  00:08:09 and moves nothing more. The scratchpad script `hexe_poll.sh` reads the stream for 10 s every 5 min.
- Block 6's two chunks simulated clean on block 5's registration at 23:38 (no abort, no early contact). Their cell
  checks on every 5th sample finished at 00:17 and 00:20. The mount stayed at least 3.40 mm (rod-more-743) and
  3.06 mm (rod-more-686) from the cell, and the finger 1.63 and 1.26 mm. Joint 6 ran from -55.2 to +36.0 deg, joint 5
  came down to 31.7 deg (1.7 deg inside its window), and the elbow stayed at least 66 deg from straight. rod-ref-a is
  over 3.06 mm, so it stays in the plan.
- Ready for the owner's go: `lab/20261030-plan_sphere_lab_check_v3id3.json` holds pass 1's touch-offs at sphere nodes
  788, 728 and 729, renamed chk-*, as the check of 10-28 was for v3 ID 1. Each onset against block 1's registration
  shows whether the reseat moved the tool. The check and block 4's remaining 55 actions are simulated from home at
  nice 19 (the remainder's first travel, from home into tall-edge-626, is new). Scratchpad `resume_id3.sh` gates on
  those simulations and on block 6's chunks, then runs the check, block 4's remainder and block 6.
- The check simulated clean from home (00:17). The mount stayed at least 5.18 mm from the cell and the finger 5.10 mm,
  and joint 6 ran from -28.8 to +6.4 deg. Block 4's remainder, 55 actions and the preflight, simulated clean, and its
  cell check finished at 01:38. The mount stayed at least 4.09 mm from the cell (tall-slide-face-738-1N), the finger
  2.48 mm, the HEX-E 22.87 mm and the links 29.82 mm. Joint 6 ran from -57.0 to +44.1 deg, joint 5 from 37.3 to
  97.6 deg (2.4 deg inside its window), and the elbow stayed at least 70.6 deg from straight. A dry run of
  `resume_id3.sh`'s gate passed all four simulations at 01:39.
- The HEX-E came back by itself at 01:37:18, 89 min after the loss. Every read from 00:12 to 01:32 gave no samples.
  At 01:37:18 a 10 s read gave 1001 samples and a 20 s read 2001, with no time-out and no gap. The force read
  [-0.7, 0.5, -0.2] N with the arm still. After R29's loss on 10-27, the sensor came back by itself in about 15 min.
- From 01:40 the lab reads the stream for 10 s every 10 min (scratchpad `hexe_stay.sh`) and logs each read. It stops at
  a loss (two reads 30 s apart, each under half the samples), when a program starts or after 10 h. `resume_id3.sh`
  now refuses to start while either HEX-E watch runs. The arm waits for the owner's go.
- All 37 reads from 01:39 to 07:40 were clean, 1001 samples each with no time-out. The untared Fz settled from
  -0.1 N at 01:39 to about -1.5 N by 02:19 and has stayed between -1.3 and -2.2 N; Fx and Fy held at -0.7 and 0.4 N.
  The program tares each action at rest before its approach, so this drift does not enter its forces.
- At 07:45 the workstation offered a wrist order of the remaining 55: `lab/20261030-plan_tall_lab_stage2_night_wrist_v3id3.json`
  (c5c34f60, from `scripts/ur5_tip_calibration_wrist_order.py`). It holds the same 101 actions with identical
  definitions and top-level keys. The 46 done keep their places, and none of the 55 moves more than 4 places. A resumed
  session runs the plan it is given, in that plan's order, and skips the ids already done. So `--resume 153034` with
  this plan runs the 55 in the new order.
- The scratchpad script `wrist_sig.py` compares the joints at each action's start with those at the next action's
  start. In the night order's 55, simulated from home, it finds 7 travels where wrist 1 passes from below 0 to above
  it while wrist 2 rises by more than 10 deg. The first runs from home into tall-edge-626 (wrist 1 -24.8 to +24.6 deg,
  wrist 2 47.3 to 96.0 deg). The other six are the slides at flank-580, edge-637, -718, -696, -627 and -626. The
  workstation's replay lists the same seven. Session 153034 had seven such travels on the robot: pad-edge-05 in
  block 3, then tall-flank-580, edge-637, -718, -696, -627 and -626. Only the last lost the sensor. The remainder's
  first travel is a new path, as noted at 00:15, but it has the losses' signature.
- The wrist order's 55 have been simulating from home on 153034's registration since 07:45 (n3b4w, every 5th sample,
  nice 19). `resume_id3.sh` takes a block 4 plan and the simulation that gates it as B4PLAN and B4SIM. The night's
  plan stays the default. The choice is the owner's.
- The wrist order's 55 simulated clean from home at 07:56 (n3b4w: exit 0, no abort, no early contact). `wrist_sig.py`
  finds none of the upward travels in it, against 7 in the night order. It has one swing of wrist 1 over 40 deg,
  against 12. That one is downward, into tall-slide-edge-643-1N: wrist 1 from +23.0 to -32.3 deg, wrist 2 from 88 to
  65 deg. R29's loss, into tall-face-702 (+16.3 to -37.7 deg), came at the end of a swing of that kind. Five of the
  night order's 12 are downward, into the slides at flank-532, edge-645, face-725, face-702 and edge-694.
- The workstation then added the downward crossings to its penalty and searched the orders exactly (21ee5fa8). The
  result is `lab/20261030-plan_tall_lab_stage2_night_wrist2_v3id3.json`. Checked here like the first: the same 101
  actions with identical definitions and top-level keys, the 46 done and the references in place, and no action more
  than 4 places from its own. Its 55 start at tall-edge-625, -622, -610, -604 and -626. They have been simulating
  from home on 153034's registration since 08:18 (n3b4w2, every 5th sample, nice 19), beside n3b4w's cell check.
- wrist2's 55 simulated clean from home at 08:29 (n3b4w2: exit 0, no abort, no early contact). They have no upward
  travel with the signature and no swing of wrist 1 over 40 deg. Wrist 1 crosses 0 in 10 travels, against 12 in wrist
  and 20 in the night order. The wrists travel 1568 deg in all, against 1746 and 2779 deg.
- wrist2's largest swing is into tall-slide-edge-718-1N: wrist 1 from -24.0 to +15.9 deg (+39.9), wrist 2 up only
  5.4 deg. In the travel that lost the sensor at 00:07, wrist 1 swung +56.6 deg and wrist 2 rose 36.3 deg. R29's
  loss came after a swing of -54.0 deg. In 153034 the travel into tall-edge-627 swung +45.3 deg with wrist 2 up
  26 deg, and the sensor held. The cell check of n3b4w2 runs and should end about 09:35.
- n3b4w's cell check finished at 09:09, and `resume_id3.sh`'s gate passed it. The mount stayed at least 4.09 mm from
  the cell (tall-slide-edge-645-1N) and the finger 2.50 mm. Joint 6 ran from -57.0 to +44.1 deg, joint 5 from 37.3
  to 97.6 deg (2.4 deg inside its window), and the elbow stayed at least 70.8 deg from straight. These are the night
  order's margins, as the workstation's path check found.
- n3b4w2's cell check finished at 09:43 and passed the gate with the same margins. The mount stayed at least 4.09 mm
  from the cell (tall-slide-edge-647-1N) and the finger 2.48 mm. Joint 6 ran from -57.0 to +44.1 deg, joint 5 to
  97.6 deg (2.4 deg inside), and the elbow stayed at least 70.8 deg from straight. The gate passed the whole set the
  resume would run with it: the check, wrist2's 55 and block 6's two chunks. The HEX-E's reads stayed clean through
  09:40. The lab recommends wrist2 to the owner.

### The resume on 10-30 (from 10:54)

- At 10:54 the owner gave the go: "sure, just finish now with the original order while im here". The lab stopped the
  HEX-E watch and started `resume_id3.sh` with the night order. The 20 s read was clean, the four simulations passed,
  the home path check passed from the arm's pose, and the arm went home (0.03 mm, 0.01 deg from the target).
- The check, `lab/20261030-105538-chkv3id3` (10:55:38 to 10:58:12, a seed of block 1's registration), gave onsets of
  +0.012 mm (chk-face-00), +0.116 mm (chk-face-03) and +0.037 mm (chk-edge-00). The script stopped on the 0.116 mm.
- That limit was set on the wrong quantity. Block 1's fit left a residual at each node, and the expected onset carries
  it: at chk-face-03's node it was -0.068 mm (the residual's sign is opposite to the onset's). Against block 1's
  residuals at the same nodes, the three changed by -0.023, -0.048 and -0.056 mm. The fit's rms is 0.044 mm, and block
  2's reference stops scattered 0.12 mm. The tool did not move. Scratchpad `chk_change.py` now makes this comparison,
  and `resume_id3.sh` gates on it.
- The HEX-E's noise in the check's preflight, 0.23 N in Fz, matches every session since 10-27 (0.22 to 0.25 N).
- Under the owner's go the lab went on without repeating the check: `resume_id3.sh` takes a finished check as CHKLOG
  and CHK. It started again at 11:00:40. The read was clean, the gate passed, and the home path check passed from the
  check's last pose. The arm went home (0.03 mm), and block 4 resumed at 11:02:12 in the night order
  (`logs/20261030-110212-tipcal-tallv3id3.log`).
- **The HEX-E dropped on block 4's first travel, from home into tall-edge-626.** Wrist 1 went from -24.8 to about
  +25 deg, wrist 2 from 47.3 to 95.9 deg, and the elbow from 109.5 to 70 deg. At 11:02:49, as the travel ended, the
  program had no sample for 100 ms. tall-edge-626, -668 and -670 aborted, and the session stopped at 11:03:04 (exit 1).
  The arm stands in free air, its pad about 12 mm above tall-edge-626's point (joints 80.2, -68.0, 71.7, 24.8, 95.7,
  31.7 deg). At 11:03:35 a 10 s read gave no samples.
- This is the fourth loss after a swing of this kind, and the second on this travel into tall-edge-626. The simulation
  from home (n3b4r) had flagged it as the night order's first such travel. The lab asked the owner to reseat the
  connector and look at the cable's run past the wrist. `hexe_poll.sh` reads every 5 min. After a reseat the lab
  proposes the check again, with the corrected comparison, then block 4 in wrist2's order, which has no such swing.
- At 11:09 the owner reseated the connector and gave the cable more slack: "you have the go ahead to run more". The
  DAQ had restarted and lost its tare (raw Fz 653 N, as on 10-22). The program tares at its start and before each
  action, and `move_guarded.py` tares at rest, so this does not matter. With the arm still at the 11:03 pose, a 10 s
  read gave 1001 samples at 11:09:58.
- `resume_id3.sh` started at 11:10:25 with wrist2's plan (B4SIM n3b4w2). The 20 s read was clean, the gate passed, and
  the arm went home from the 11:03 pose (0.06 mm). The check, `lab/20261030-111133-chkv3id3` (11:11:33 to
  11:14:04), passed on the corrected comparison: +0.013, +0.078 and -0.030 mm from block 1 at the same nodes.
  chk-face-03's node keeps the largest scatter: -0.068, -0.117 and +0.010 mm over block 1 and the two checks.
- **Block 4 finished at 13:19:51: 101 of 101 ok.** The last 55 ran in wrist2's order from 11:15:24
  (`logs/20261030-111524-tipcal-tallv3id3.log`), with no abort and no early contact. The HEX-E gave 745,632 samples
  with no time-out, and its one gap over 0.5 s came at the start. tall-edge-626 passed at 11:18:58 and
  tall-slide-edge-626-1N at 12:42:24. These were the arm's first two arrivals at the pose of the five tall losses
  since the reseat and slack.
- The workstation's count over every log since 10-25 (QUEUE R31, 11:2x) found 4 losses within 10 s after 138 travels
  where wrist 1 crosses zero while wrist 2 rises, and 4 after 1,830 other travels. All five tall losses came on
  arriving at one pose: base 78.5 to 80.5 deg, wrist 1 +18 to +26, wrist 2 90 to 96 and wrist 3 18 to 32. Standing
  at that pose, the link held: from 01:37 to 10:55, and over 11 other visits.
- The chain checked the home path from block 4's last pose, and the arm went home (0.04 mm, 0.01 deg from the
  target). Block 6 started at 13:20:39 (`logs/20261030-132039-tipcal-rodv3id3.log`, `--resume` on 230820, rod seat 3).
- The owner told the workstation at 12:5x that no calibration runs tonight. The next runs use the inverted rig, on
  the low seat strip (ac021c3dd). Nothing follows block 6 in the queue.

### Block 6: ID 3's rod stage 2 (13:20:39 to 14:49:02)

- `lab/20261029-plan_rod_lab_stage2_night_v3id3.json`, 100 actions, resumed on 230820 (rod seat 3) after block 4's home
  path check and go home (0.04 mm, 0.01 deg). Printout `logs/20261030-132039-tipcal-rodv3id3.log`.
- 100 of 100 ok in 1 h 28 min, with no abort and no early contact. The HEX-E gave 529,491 samples with no time-out, and
  its one gap over 0.5 s came at the start. Its two chunks had simulated clean on block 5's registration, with the mount
  at least 3.06 mm and the finger 1.26 mm from the cell.
- The arm stands at rod-final-ref's rest over the rod (joints 32.9, -61.1, 97.5, -33.1, 36.4, -6.8 deg).

### ID 3's suite: done (14:49 on 10-30)

- Blocks 1 to 6, the top edge and the side map all ended ok. Block 4 ran across two HEX-E losses, at 00:08 and 11:03.
  The owner then reseated the connector and gave the cable more slack, and its last 55 actions ran in wrist2's order.
- The owner runs no calibration tonight (via the workstation, 12:5x). The next runs use the inverted rig on the low seat
  strip. Everything comes off the table after this suite.

### The night's plans, after the finer path checks (14:4x)

- The workstation's path checks at every 2 mm sample (bb7c5578) finished here at 14:41. The tall's stage 2 brings the
  mount to 2.97 mm in tall-slide-flank-585-1N's own 2.3 deg turns at depth; the press at the same point,
  tall-flank-585, clears 3.20 mm. At the workstation's ask (14:43), the slide is left out of
  `lab/20261029-plan_tall_lab_stage2_night_v3id3.json`, which now holds 101 actions.
- rod-ref-a stays (its 2.96 mm is on the checker's travel from its own start pose). The other flags are the checker's
  joint windows of -62 and 100 deg; the lab's simulations keep joint 6 inside -65 deg and joint 5 inside 100 deg.

## Probe mode for the inverted rig: what the program needs (estimate, 10-29 16:1x)

The owner approved building the inverted rig at 15:2x on 10-29, in the workstation's session ("let's shore up the plan
and get it ready to print tomorrow"). Its page and runbook are in `docs/experiments/20261029-inverted_calibration_rig/`.
The workstation asked the lab what the probe mode takes in `ur5_tip_calibration.py` (1926 lines) and its simulator
`ur5_tip_calibration_sim.py` (662 lines). This estimate comes from reading both. Nothing has been written yet, and no
robot run comes without the owner's go.

The roles swap. Today the indenter is fixed (T_BI) and the pad moves with the flange (T_EP = T_EM T_MP). In probe mode
the pad is fixed on its seat (T_BP), and the probe's sphere moves with the flange (its centre c_M in M). Most of the
mathematics carries over, because the program already works in P: a touch gives the sphere's centre in P, and the
contact and the board are computed in P.

- **The plan and the registration** (about 40 lines). The plan gains a probe geometry: the sphere's radius, its centre
  and stem in M, and a seat frame per seat on the plate. Registration gains T_BP and c_M, saved and loaded like T_BI.
- **The target pose** (about 40 lines). A probe action names a pad point and its normal in P, a push direction, the
  stem's direction and turn, and a twist about the normal through the sphere's centre. The flange pose puts T_EM c_M at
  T_BP (p + n (R + standoff)). The push direction in B is the action's direction turned by T_BP.
- **Pass 1 with the roles swapped** (about 70 lines). The touch-off is today's press primitive and onset fit, unchanged,
  with the new target pose. A new fit takes T_BP (6) and c_M (3), with priors from the plate's homing, the seat's CAD
  and the probe's CAD. Its residual is today's, the sphere's centre in P against the pad's surface, with q = T_BP^-1
  T_BE T_EM c_M. The fork's homing on the plate's keys is unchanged.
- **The force and the board** (about 50 lines). The push along d_B keeps today's formula. The force on the pad is minus
  the HEX-E's change, turned into P, recorded with each hold. `board_level` reads the seat's mux channel from
  HeldBoardStream, which already keeps a ring per channel in the robot board's format (5 chips x [T, x, y, z]). The
  preflight checks that channel's rate.
- **Gravity from the presses' rests** (about 50 lines; no sweep, owner's rule). `fit_gravity` takes the rest windows
  before and after each press as its poses, which the twists turn. The payload is set at about 0.11 kg plus the
  adapter disc.
- **Safety** (about 40 lines). `in_box` is drawn around the seat (T_BP), not the indenter. `lifted` lifts until the
  probe's sphere and stem clear the seat: 30 mm, or 150 mm where the workstation's replay found the lower lift sweeps
  the probe through a seat. The travel abort and the joint 5 and 6 windows are unchanged.
- **Stages** (about 40 lines): probe touch-offs, probe register, probe presses with levels (then slides and twists),
  and `--dry` keeping the probe 15 mm off.
- **The simulator** (about 150 to 250 lines, the largest part). `World.contact` computes in P from T_BP = flange T_EP. In
  probe mode it takes T_BP fixed and the sphere's centre from the flange, and the HEX-E's compliance moves the sphere,
  not the pad. The board becomes a held board on a fixed pad, so Earth's field stays constant. The HEX-E's payload
  becomes 0.11 kg. Truth gains errors on T_BP and c_M, so the fit can be tested as `sim_seed_perturb.sh` tests today's.
- **Checks** (the lab's scripts). The cell check needs the probe's parts in place of the fork and the tip. The
  workstation's `scripts/ur5_inverted_rig.py` replay already models those clearances, and the lab's simulations would
  check the program's real moves against them.

In all, about 330 lines in the program and 150 to 250 in the simulator, plus the lab's check scripts. That is a day of
work, then 2 to 3 h of simulations: the fit recovering a perturbed T_BP and c_M, and each seat's replay plan (271 to
279 actions). On the robot, once the parts are printed and mounted, a `--dry` run over the plate comes first (about
0.5 h, the owner present), then pass 1's touch-offs on one seat with the owner watching. Each of these needs the owner's
go in the lab's session.

## Probe mode for the inverted rig: written and simulated (10-30, from 13:4x)

The owner said yes in the workstation's session (13:2x: "sure, run the probe mode"), for code and simulations. On
leaving at 13:4x they added: "even if this aborts ... you can run the sim anyway". The dry run and the first touch-offs on
the robot each need the owner's go in the lab's session. The program was edited only after block 6 had exited (14:49);
until then the work ran in a scratchpad copy.

- **The program** (`ur5_tip_calibration.py`, probe mode when the plan's geometry has `probe`):
  - Frame I is the strip's A. Homing seats the probe's socket on a key of the strip, with T_IM_home that key's
    T_AM_home, and gives T_BI = T_BA.
  - Registration gains T_AP (the held pad on its seat) and c_M (the sphere's centre). `fit_probe_registration` fits
    both from pad touches, with priors of 1, 1 and 0.5 deg (the roll about z_P), 0.5 mm and 0.3 mm. Its residual is
    the tip mode's: the sphere's centre in P against the pad's surface.
  - A target gives p, n, d and the mount's rotation R_PM in P. The sphere's centre goes to T_BP (p + R n - standoff d)
    from the registered T_BP and c_M.
  - Travel follows the workstation's replay: straight (back 12 mm along the stem), joint (back 45 mm, a joint move) or
    by way of the hub. From the homing it first lifts 30 or 150 mm out of the key. A straight move only follows the
    action the replay had before it.
  - A twist turns about the pad's normal through the sphere's centre.
  - The board is the seat's, its mux channel on --held-board. Without --board no robot board is opened.
  - The push's running mean takes each of the last 20 HEX-E samples once, and the press tests it only when it is full.
    The tip mode's mean is over the last 8 passes of its 250 Hz loop, about 3 samples. A probe pushes along the HEX-E's
    z, and in the simulator that shorter mean crossed 0.3 N in free air: a dry run's fast approaches stopped as if in
    contact, and once a press counted 0.5 N in air.
  - Gravity, mount, pivot and hover stay the tip mode's.
- **The simulator** (`ur5_tip_calibration_sim.py`): the pad stands still at T_BI T_AP, and the sphere rides on the flange.
  The HEX-E's compliance acts on the probe (`hex_in_P`), and the HEX-E reads the push back. The held board's Earth field
  stays constant. The truth gains `probe_err_mm` and `seat_pose_err`.
- **The tip mode is unchanged.** HEAD's program and simulator and the new ones print the same session line for line on
  v3 ID 3's sphere pass 1 (seed 0, stages preflight, homing, mount, pad, register).
- **Plans**: `scripts/ur5_probe_plan.py` turns the workstation's seat plans (1912b479, with deep_clear_mm) into program
  plans:
  - Each press may go deep_clear_mm past contact.
  - Each twist presses to 3 N, turns with 1 s holds, then repeats at 1 N at the same rest.
  - Travels come from the _paths_ rows.
  - The mux channels are the owner's: seat 1 on 3, seat 2 on 1, seat 3 on 0.
  - Pass 1 is 20 of the presses as touch-offs, chosen greedily on the expected miss of every press.
  - The rest poses come out within 1.9 um of the replay's.
- **Touch-offs at 1 N.** At 0.5 N and 0.4 N, 2 of 20 touch-offs found no onset or too few samples. A probe pushes
  along the HEX-E's z, whose noise (0.21 N a sample) is five times x's and y's, along which the tip mode's touches
  push. At 1.0 N and 0.8 N, 20 of 20 found their onsets.
- **Pass 1 needs the mount seeded, or a larger standoff.** Unseeded, the simulator's 0.9 deg turn of the mount about the
  tool axis tilts the homed strip. Seats 2 and 3 then stood 1.4 mm rms (1.9 mm at most) off their CAD along the normal.
  Rests at 1.5 mm were inside the pad, and two travels met it at 8.2 and 8.9 N. Both fixes worked:
  - A seed with the mount as a tip-mode registration gives it: 20 of 20 and 19 of 20 at 1.5 mm.
  - A standoff of 3.5 mm: 20 of 20 on both seats, unseeded.
  The plans use both. The 3.5 mm rests stand 2 mm back along the push from the replay's checked rest, so they need the
  workstation's check.
- **Accuracy.** After pass 1 the fits leave 0.010 to 0.021 mm rms. Against the simulator's truth, the presses miss by
  0.08 to 0.12 mm rms along the normal and 0.13 to 0.27 mm across. The across part is the roll about z_P, which the
  pad's shape barely shows. All 280 presses as touches would still leave 0.55 deg at a 1 deg prior. At the
  workstation's 0.5 deg the expected miss is 0.13 mm rms. Touch-offs on the cassette's flat side walls would measure the
  roll; the workstation offered to add them.
- **The program's own moves**, checked with the workstation's clearance and joint checks at the simulator's true cell
  (scratchpad `probe_cell_check.py`), are clean in pass 1 apart from the homing. Seated on a key, wrist_3_link stands
  4.03 mm from the low sphere's 2x4 (need 8). The workstation confirmed it (d52b1998b): about 5.5 mm in the real cell,
  and about 40 mm with that 2x4 off the table.

## Probe mode: the tip's side walls (10-30, from 15:1x)

The workstation pushed d37468c18 at 15:13. It holds a check of pass 1 as the converter writes it, and touch-offs on the
tips' side walls. The lab's answers went to the workstation at 15:37.

- **Pass 1 is clean at 3.5 mm.** `ur5_inverted_rig.py --plate block pass1-check` replays the pad stage in the
  converter's order. It flies the lift from the homing, each touch-off from 3.5 mm to its beyond_mm and back, the joint
  travels between them and the travel into stage 2. All three seats pass. The least stem clearances are 3.0, 2.33 and
  2.41 mm, the tool end's 0.56, 0.54 and 0.47 mm (`data/20261030-seat_block_pass1_check.json`).
- **probe-ref-00 keeps homing_lift_mm 30 in the plan.** The program takes that lift only straight after the homing:
  the homing sets `from_homing`, and the first travel clears it. After pass 1, probe-ref-00 flies the joint travel from
  the last touch-off's rest, as the check flew it. The lift serves a session resumed straight into stage 2.
- **The walls** (`data/20261030-seat_block_walls.json`). Each seat gets four touch-offs on the cap's distal strip, where
  the side planes y_F = +-10.5 are flat from x_F 0.5 to 8.0. They sit at x_F 1.0 and 7.0, so the two on a wall stand
  6 mm apart. Each rests 3.5 mm back and goes at most 0.3 mm past the wall. The walls come after the pad's fit,
  because the low touch's strip is narrow. They first sat at z_F -99.4 and moved to -99.75 at 15:40 (8ee8556e, below).
  With the walls after the pad's touch-offs, all three seats run 25 moves each with none flagged (least stem 1.67 mm,
  tool end 0.36 mm). On seat 1 the move from py-lo to ny-hi goes by the hub.
- **The program.**
  - Stage `wall` runs the walls' touch-offs, with a linear onset fit as on the mount's face. A probe session runs
    `--stages preflight,homing,pad,register,wall,register,grid`. The program refuses a wall stage with no pad fit
    before it.
  - `fit_probe_registration` takes a residual per wall touch: the sphere centre's height over the wall's plane, less R,
    less the wall's offset. Each wall has one offset, with a 0.1 mm prior for the print, and a touch counts 0.02 mm.
    So the walls' tilt enters the fit and their width mostly does not. A cassette printed wide would otherwise pull the
    sphere's centre along the stems.
  - The priors now stand at the plan's T_AP and c_M, not at the last fit, so register can run twice. The first fit is
    unchanged: a refit of s1full3's touches gives its numbers to the digit.
  - Register prints the turn about z_P that each wall gives alone, and warns when the two differ by more than 0.8 deg.
- **The simulator** makes the side walls stiff planes, as it makes the mount's face in the tip mode. Their faces come
  from the workstation's rig model (`wall_faces`), moved into P by the converter (`sim.walls`). The truth puts the +y
  wall 0.05 mm out and the -y wall 0.03 mm in. The first run (s1w) stopped on three protective stops: the contact took
  the far wall, across the tip, as touched. A wall now counts only while the sphere's centre is within R of its plane.
- **The converter** adds the walls after pass 1 with `--walls` and `--walls-check`. It takes each wall's travel from the
  check's rows. It stops unless those rows list its touch-offs in order at their standoffs; all three seats match. At
  the CAD, the walls take the turn's sd from 0.49 to 0.17 or 0.18 deg. A press's expected miss falls from 0.133, 0.127
  and 0.129 mm rms to 0.094, 0.089 and 0.090 (seats 1, 2 and 3).
- **Simulated, seat 1, seeded, 3.5 mm rests:**
  - The dry run of 28c2892c (`--dry 15`, s1dry3) had no abort and no early contact in 24 min.
  - Without walls (s1full3): 296 of 296, no abort. From the registration (scratchpad `probe_aim.py`) the presses miss
    by 0.092 mm rms along the normal and 0.270 across. At their true rests, with the robot's kinematic errors kept
    (scratchpad `probe_land.py`), they miss by 0.018 and 0.165 mm. The fit absorbs part of those errors, so
    `probe_aim.py` overstates the miss.
  - With walls at z_F -99.4, through the second register (s1w2): the onsets came -0.027, +0.020, -0.028 and
    -0.111 mm from the expected, with peaks of 1.9 to 2.6 N. The second fit leaves 0.011 mm rms on the walls, with
    offsets 0.055 and 0.009 mm (truth +0.05 and -0.03). It turns the seat +0.625 deg about z_P (sd 0.17), against the
    truth's +0.72 deg. The pad alone gave +0.11 (sd 0.48). From the registration, the across miss falls from 0.270 to
    0.177 mm rms.
  - Seats 2 and 3 the same way (s2w, s3w): turns +0.635 and +0.599 deg (pad alone +0.15 and +0.14), across misses
    0.273 to 0.181 and 0.230 to 0.155 mm rms. No abort.
  - The walls' offsets (scratchpad `probe_wall_variants.py`, the same touches): pinned at zero, the across misses are
    0.164, 0.170 and 0.145 mm; free, 0.179, 0.183 and 0.157; with the 0.1 mm prior, 0.177, 0.181 and 0.155. The prior
    costs little when the print is right.
- **Simulated on the final code, walls at z_F -99.75, whole sessions** (s1wf2, s2wf2, s3wf2: seeded, 3.5 mm rests,
  `--stages preflight,homing,pad,register,wall,register,grid`):
  - 296 of 296, 297 of 297 and 295 of 295 actions, with no abort, no early contact and no warning.
  - The wall onsets came -0.130 to +0.068 mm from the expected. The second fits leave 0.009, 0.016 and 0.006 mm rms
    on the walls. They turn the seats +0.542, +0.645 and +0.620 deg (sd 0.17 to 0.18), against the truth's +0.72;
    each wall alone gives +0.46 to +1.03 deg.
  - At their true rests (`probe_land.py`), the presses miss by 0.018, 0.023 and 0.027 mm rms along the normal. They
    miss by 0.072, 0.047 and 0.057 mm rms across (at most 0.132 mm), where seat 1 without walls missed by 0.165.
  - From the registration alone (`probe_aim.py`) the across misses read 0.191, 0.180 and 0.157 mm. The robot's
    kinematic errors, which the fit absorbs, make most of the difference.
  - The walls' moves and the travel into stage 2, every sample, in the cell (`probe_cell_check.py --ids
    'wall-|probe-ref-00$' --tip-true`): nothing flagged on any seat. The least clearances are 0.38 to 0.40 mm for the
    tool end, 1.59 to 2.61 mm for the stem, 53 mm for the HEX-E and 59 mm for the arm; joint 6 stays within -17 to
    +15 deg. `--tip-true` stands the tip where the simulator had it. With the tip at its CAD pose instead, wall-py-lo
    reads the tool end 0.24 to 0.43 mm inside the tip body, the truth's 0.25 mm and 0.7 deg off the seat's CAD.
  - Seat 1's dry run with walls (`--dry 15`, s1dryw): no abort and no early contact.
- **The rig model's tip-body distance can take the wrong sign.** In seat 1's run without walls (s1full3), the cell
  check flagged probe-flank-476: the tool end 6.44 mm inside seat 1's tip body at 2 N. The workstation's replay has
  that press clear (tool end 1.04 mm, stem 2.87 mm, both to the pad), and the sim's press stood within 0.9 mm of its
  deep pose. An exact test of the flagged point puts it 6.34 mm outside the cover, 6.52 mm outside the cap and
  0.28 mm inside the pad, where the press indents it. `_tree_sdf` takes the sign from its 8 nearest surface samples'
  normals, and 6 mm from any body surface it read inside. The workstation was told: the opposite error, a point inside
  the body read as clear, would be the dangerous one. Over the whole session (every 5th sample) nothing else was
  flagged: 177 samples in the homing (the seated pose, 4.25 mm from the 2x4s, and the socket on the key) and 73 in
  probe-flank-476.
- **The station block's face, a risk the simulator cannot show.** At z_F -99.4 the low touch (x_F 7.0) was 1.04 mm
  from where the station block's coplanar face starts (z_F -98.36). The probe's sphere has a 5 mm radius (`sphere5`,
  the plan's radius 5.0), so there its surface stood only 0.11 mm off the wall's plane. A block face that much proud of
  the tip's wall would have met the low touch first. That happens when the tip sits off across y in its seat. A +y
  low touch 0.12 mm early takes the fitted turn from +0.63 to +0.15 deg: that wall alone gives -0.59 deg, the other
  +0.90. The workstation answered at 15:40 (8ee8556e):
  - The cap's strip runs z_F -100.04 to -98.46 there; its coarse map had shown -99.75 to -99.0. Both touches moved
    to z_F -99.75, where the margin is 0.20 mm: a block face more than 0.20 mm proud still meets the sphere first.
    The lab's message and the workstation's answer had both taken a 2.5 mm radius (0.23 and 0.42 mm); the lab
    corrected it at 16:2x. The cap's distal edge is 0.29 mm from the contact, about 3 sd of the pad fit's z.
  - The seat does not fix the tip across y to 0.1 mm. The M2 sit in 2.4 mm bores, so the tip can shift about 0.2 mm,
    and both 21 mm widths carry +-0.05 to 0.1 mm of print. At worst the block stands about 0.3 mm proud on one side.
  - The strip's README now holds a steel rule flat along the block's side and the tip's wall while the M2 are
    tightened, which makes that side flush to about 0.05 mm.
  - pass1-check now reads a plan's own wall stage (stage 2's first action is the first outside pad and wall).
- **The tip mode is unchanged.** HEAD's program and simulator and the new ones print the same session line for line,
  apart from the session folder's name: 34 lines on v3 ID 3's sphere pass 1 (tipold4 against tipnew6).

## Probe mode: deeper margins, and the dry run along the stem (10-30, from 16:4x)

- **The workstation's exact body check** (bbe38fc93, `ur5_inverted_rig.py body-check`) answers the sign question. It
  takes the nearest point on the cover's and the cap's meshes, with inside decided by the generalized winding number,
  at every action's deep pose, every twist step and every wall touch: 3,135 poses. No tool point was inside the body;
  the exact least was 0.259 mm. Near the surface the tree read up to 0.105 mm long, so deep_clear now keeps 0.1 mm
  more on the tool's end. The seat plans changed in deep_clear_mm only: 56, 71 and 64 actions, each 0.25 or 0.5 mm
  shallower.
- **The plans, rebuilt.** The three lab plans differ from 349ec386's in max_depth_mm on those actions, in one seat-2
  touch-off's beyond_mm (1.2 to 1.0) and in their notes. Pass 1 and the walls are unchanged, and the workstation's new
  pass1-check rows match.
- **The dry run backed off into key 1.** `--dry 15` moved every target 15 mm back along its push. In the tip mode
  that was straight up. In the probe mode a wall's push is horizontal at the seats' height, and a flank's mostly
  sideways. Seat 1's dry run with walls (s1dryw) put the stem 4.58 mm and the tool end 5.70 mm inside key 1 at
  wall-py-lo. The dry rest of pad-flank-449-o35-a180 brought the elbow within 20 deg of straight (the checker's q3).
  No check had covered the dry poses, and the dry run was to be the first step on the robot.
- **The program now backs a probe-mode dry run off along the stem,** the way every travel retreats: target_pose
  moves the rest pose back along z_M by --dry. A non-dry run is unchanged, since a zero move returns the pose as it
  was. Seat 1's non-dry run through the second register (s1wchk) printed the same lines as s1wf3 on the code before
  the change, and the tip mode the same 34 lines as HEAD.
- **Simulated on the rebuilt plans.** The three seats' dry runs with walls (s1dry4, s2dry4, s3dry4: `--dry 15`, the
  full stages) had no abort and no early contact. Over their whole sessions (every 5th sample, the tip where the
  simulator had it), nothing is flagged outside the homing. The q3 flag is gone, the stem stays 2.56 mm or more from
  anything, and joint 6 stays within -16 and +16 deg. The homing flags 176 to 183 samples on each seat: the socket on
  its key, and the seated pose 3.94 to 4.90 mm from the 2x4s (STATE, the owner's call).
- **The workstation's replay of the dry run** (f8f4159e, `ur5_inverted_rig.py --plate block dry-check`, 18:0x) follows
  `travel_probe` move by move, with no pad left out. It runs from 12 mm out of the key through every dry travel, press
  and twist. Seats 1 to 3 give 320, 321 and 319 actions, none flagged; joint 6 stays within -16.5 and +15.8 deg and the
  stem 5.56 mm or more from anything. Four travels turn from straight to joint in the dry run (probe-edge-569 on each
  seat, probe-flank-593 on seat 1), because their dry rests stand over 30 mm apart; both ways are clean.
- **The tool end's 3.0 mm is a floor.** For a point more than 3 mm outside the tip's bounding box, `seat_sdfs`
  returns the box's distance. Measured exactly against the cover, cap and pad meshes (`dry-exact`), the sphere stays
  8.82, 7.80 and 8.17 mm or more from its tip on seats 1, 2 and 3.
- **The old dry run, replayed** (`--along-push`), confirms the lab's finding on seat 1: at wall-py-lo the stem stood
  4.75 mm and the tool end 6.25 mm inside key 1 (the lab had 4.58 and 5.70). It flags three more: wall-py-hi and
  wall-ny-hi also ran into key 1, and pad-flank-591 also took the elbow to q3.
- The rebuilt plans' whole sessions (s1wf3, s2wf3, s3wf3) print what s1wf2 to s3wf2 printed: every action done, no
  abort, the same fits and the same true misses. Their walls' moves are clean in the cell as before.

## The strip on the robot: homing, and the owner's press configuration (11-02, from 12:5x)

- The owner mounted the probe base with the 5 mm sphere, the strip with three new tips on one 2x4 clamped to the
  table, and the RealSense; the robot was on. The tips first went on with their Qwiic plugs toward the robot; the owner
  turned them, since the plans need the plug at the back. Seat 1 (y_A -80) holds v3 ID 6 on mux channel 3, seat 2
  v3 ID 5 on channel 2, seat 3 v3 ID 4 on channel 0.
- Checks at 13:03: robot in normal mode; the slider at 1.0 after the restart (the program sets the plan's 0.30 before any
  motion); payload 0.403 kg, the finger mount's (the probe's is not measured). The HEX-E at 100 Hz with no time-outs,
  untared Fz -33 N. The held boards on channels 0, 2 and 3 at 143 records/s each. A time-lapse runs from 13:05
  (`timelapse_d435.py`, one frame per 2 s, into `/dev/shm/20261102-tipcal-timelapse`, nice 19).
- Homing by hand on key 1 (`teach_hold.py`, `record_pose.py`): 13:11 and 13:35, the joints still to 0.0012 deg, within
  0.39 mm and 0.26 deg of their mean. The strip's A is at (-202.7, -417.0, 39.2) mm in B, yaw -92.07 deg, tilt 1.85 deg:
  192.3 mm from the plans' (-138.8, -595.7, 70.2), or -176.0, -71.0 and -31.0 mm along A's axes. The program refuses
  past 20 mm, and no check covers this place; the owner chose new plans (13:3x).
- The owner's press configuration: the probe straight down over seat 1's pad, recorded at 13:19:50
  (`probe_primed_seat1_v3id6`: joints 46.5, -76.65, 104.73, -120.06, -86.17, 178.27 deg; the sphere's centre 8.1 mm over
  the apex and 2.6 mm toward the robot, the stem 6 deg from vertical). Over seat 2 the owner gave (50, -83, 111, -122,
  -80, 186), and the hand path from there to the homing: joints 1 to 3 to (48, -100, 128); joint 4 from -122 to -51;
  joint 5 from -80 to 71; joint 6 from 186 to -19; joints 1 to 3 to (37, -72, 123); joint 5 from 71 to 41; seat.
- The analysis (scratchpad `ideal_pose_analysis.py`, `ideal_pose_compare.py`), at the strip's place. Seat 2's reference
  press straight down: joints (56.3, -79.2, 108.6, -120.2, -91.8, 180.5); the velocity ellipsoid at the sphere has a
  condition number of 2.36 and a Yoshikawa measure of 146 (1e6 mm^3); the joints push 1.46 N along the press per N m.
  With the old plans' stem in the homing branch: (52.7, -68.8, 91.5, 45.8, 83.8, -1.3), 1.92, 145 and 1.61 N. Over
  seat 2's 297 grid rests the wrist joints turn 45, 1 and 49 deg in all straight down, against 626, 458 and 315 deg with
  the old stems; the largest joint change between two presses is 3.8 deg against 58.6. Every target of the three seats
  reaches straight down; with the stem along the push, 26 to 28 a seat fail |sin q5| >= 0.5. The port keeps 48.5 mm
  from the arm with joint 6 near 0, 90, -90 or 180 deg. Not yet checked: the tool against the tip's body, and the new
  table.
- 13:20:36: the HEX-E gave no samples (120 time-outs in 6 s) after the owner's hand move to the press pose. By 13:27:21
  it streamed at 100 Hz again, untared at Fz 629 N with status 0x40000: a restart, as on 10-22.
- 13:21:06: `teach_hold.py`'s 15 min cap ended teach mode and stopped its script while the owner was moving the arm by
  hand, and the robot e-stopped ("you just moved the robot very, very quickly"). Nothing was damaged. A renewal queued to
  start after the cap never started; it was killed by PID. `teach_hold.py` now takes `--minutes` and an extend file
  (`logs/ur5_lab_teach_extend`), and freedrive is extended or stopped only on the owner's word.

## The probe on the table: plans, three program faults, and seat 1 (11-02, from 13:5x)

- Homing: four seatings on key 1 and three on key 2. Each seating's own frame tilts 1.6-3.3 deg with the probe's rock
  on the key (the workstation: the probe base sits about 2.75 deg about its axis off the fork's T_EM), and key 2's frame
  put the pads 3.3-4.3 mm above key 1's. The socket's point repeats within 0.24 mm (key 1) and 0.55 mm (key 2), and the
  keys stand 80.04 mm apart, 0.42 mm apart in height: the strip is level within 0.33 deg. Its frame through both keys'
  socket points is `20261102-strip_frame_two_keys.json`; the program takes it by the new `--frame`.
- The owner's hand path from key 1 to the press pose, joint 6 near 0 (`poses.jsonl` path_home_to_press_1..3): the
  shoulder lifts 19 deg; then the wrist flips (joint 5 +38 to -90, with joint 3 -20 and joint 4 -31); then over seat 1.
  The HEX-E's cable dragged on the probe during the flip; the owner rerouted it under the probe. The HEX-E dropped at
  14:20 and was back at 14:34 after the owner reseated it.
- The workstation replayed the three seats in the owner's branch (4b7999e2: 0 of 840 actions flagged, stems vertical but
  for walls and flanks, joint 5 -97 to -80, joint 6 within +-16). Program changes (3c51b552, simulated t1a):
  `travel.probe.home_path_q` after the lift, a travel wrist window, joint moves near each action's q_rest, a `return`
  stage, the lift along the key's pin, `check_placement` against the plan's `checks.T_BA`.
- p1dry (14:38, --dry 15): three aborts, "no IK solution". The program's rotation-matrix-to-vector conversion
  (log_so3) turned the plan's 7-decimal rounding (7e-6 off orthonormal) into 1.0 deg near 180 deg, where the flange
  turns with the probe straight down, so every IK target missed its pose by 1.0 deg and the check refused it. Fixed by
  projecting onto the nearest rotation first: the round trip of all 1256 poses of seat 1's plan within 2.3e-9 deg.
  The simulator never meets this: it does not use the robot's rotation vectors.
- s1v6 (14:46): three aborts, "target outside the box": each of p1dry's failed actions had backed off 45 mm along the
  stem and lifted 5 mm, so the robot stood at 425 mm, and the next back-off left the 400 mm box. Fixed (e26f29ad, 14:50):
  the back-off and the abort's lift only near the pad. Simulated from the robot's own 425 mm start (t1high, exit 0).
- Pass 1 of seat 1 starts 5 mm out and runs 2.5 mm past the expected contact (were 3.5 and 1.2): in the simulator a
  1.4 mm, 2 deg error of the frame made 4 of 20 flank and tip touches miss with 1.2.
- s1v6b from 14:50:09: pad-ref-00's onset 1.38 mm before the two-key frame's expectation.

## Seat 1's pass 1, and seats 2 and 3 made ready (11-02, from 15:0x)

- The owner left at about 15:00 with the go for the remaining runs: "you have my go ahead to run the remaining runs".
  On the v3 IDs 1-3 in his viewer, the flank presses are still the hardest and least accurate ("that's where the
  embedded magnets stiffen/alter the typical gyroid infill the most"). "contact direction! super smooth, super accurate,
  super important!" And: "if it's possible to get twists/wrenches, e.g. we see some kind of predictive performance scaling
  with data collection, then you have my go ahead to send off more runs on that end too. while increased force may not be
  necessary, as you measured, varying speed of contact might matter more."
- Seat 1's pass 1 (s1v6b): 20 of 20 landed, every one early, by 0.12 to 2.48 mm on the two-key frame. The fit gives
  0.055 mm rms: the pad stands 1.68 mm and 4.12 deg from its CAD. The turn is -4.07 deg about y_P, +0.56 about x_P and
  -0.26 about z_P (sd 0.20, 0.13, 0.47); the move is (1.61, 0.28, -0.40) mm in P. The tip droops about 4 deg toward the
  robot and stands 1.6 mm high along its face normal. The sphere's centre fits 0.58 mm from its CAD, along the stem.
- The grid started at 15:06 at about 25 s a press. At the fitted pose its rests, deep poses and twists keep joint 5
  within -92 to -75 deg and joint 6 within -18 to +13 deg; the rig model flags none, and the arm keeps 182 mm from the
  table and the 2x4 (scratchpad `check_fitted.py`).
- Seats 2 and 3's plans (44509f1e) come from 4b7999e2's seat plans, aimed by `--frame` as seat 1's. The workstation
  (15:1x) kept these: its rebuild on the two-key frame has the same actions. Pass 1 starts 5.0 mm out and runs up to
  5.0 mm past. In the simulator, with the truth at the two-key frame plus its default errors (about 2.6 mm in plane),
  2.5 mm past left both seats' first three flank touches without contact. The three-aborts rule then stopped the
  sessions. With 5.0 mm all 20 land, 2.2 to 3.2 mm late, and the fits give 0.043 and 0.026 mm rms. A touch stops at 1 N
  on whatever it meets, so the longer reach matters only when the pad stands farther than expected.
- In the two-key frame (rig model, scratchpad `check_seats_twokey.py`), the rests and deep poses of all three seats keep
  joint 5 within -92.3 to -89.7 deg and joint 6 within -1.3 to -0.5 deg, with no flags. The seams were checked as the
  program travels: from one seat's last rest, back 45 mm, to the next seat's hub, then in to its first touch-off. They
  are clean, and the sphere stays 46 to 128 mm over the strip.
- The time-lapse stopped at 15:06:25 on the D435's "Frame didn't arrive within 5000". `timelapse_d435.py` now restarts
  the camera after a stall. It runs again from 15:13 into the same folder, until 08:13 on 11-03.

- The ellipsoids in MuJoCo (`20261102-ellipsoids_owner_vs_old.png`, numbers in `20261102-ellipsoids.json`, script
  `ellipsoid_render.py`): both configurations press seat 2's reference point at its rest, with the strip at the two-key
  frame. Each panel draws the velocity ellipsoid at the sphere's centre, sqrt(eig(J J^T)), and the force ellipsoid,
  1000/sqrt(eig), at one scale in both. The owner's configuration: condition 2.34, Yoshikawa 146 (1e6 mm^3), 525 mm/s
  and 1.48 N per N m along the press. The 10-30 plans' configuration (the homing's branch, joint 5 +82, the 10-30
  stem): 1.92, 143, 539 mm/s and 1.63 N per N m. They are about equally good; the owner's configuration gains its wrist
  travel over a seat's grid (45/1/49 deg against 626/458/315).

## Seats 2 and 3 checked and armed, lower travels, the overnight plans (11-02, 15:2x to 16:1x)

- The workstation replayed seats 2 and 3 and both seams as they run, in the two-key frame (ee772007; `--real --chain`,
  seat 1 from over the seats, pass 1 at 5/5 mm, the grid). It found no joint or travel flags, and the seams' least stem
  is 9.35 mm. Two kinds of flag, neither a collision:
  - 86 travels follow a twist whose press ran its full 2.0 mm. The program then returns st + 0.5 mm from the press's
    end, which leaves the sphere on the pad's surface.
  - 2 presses meet their own pad 0.33 and 0.40 mm outside 4 mm of the target at full depth.
  With seat 1's fit applied to every seat, the flags are the same.
- Lower travels (owner, 14:4x): a 15 mm back-off and hubs 40 mm over each apex replay as clean as 45 mm and 90 mm (the
  workstation, 15:3x). Each travel's closest point is one of its end rests. They are now in seats 2 and 3
  (`probe_low_travel.py`) and in the overnight plans. Between seats the sphere stays 46 to 80 mm over the strip, where
  it reached 128 mm before. Simulated chained, each seat from the previous seat's probe-ref-final rest, with the truth at
  the two-key frame:
  - seat 2: 294 of 294, pass 1 20 of 20 at 0.028 mm rms;
  - seat 3: 295 of 295 at 0.019 mm rms;
  - no abort on either seat.
- The chain (`20261102-chain_seats23.sh`) was armed at 16:10. When seat 1 ends cleanly it starts seat 2, then seat 3. A
  run stopped by a HEX-E loss is resumed on its session once `hexe_probe.py` reads the sensor again, every 60 s for up
  to 3 h and at most 3 times. After seat 3 come the overnight runs in `20261102-overnight.txt`, each gated on a marker.
- Whether more data pays (owner, 15:0x):
  - The workstation's 10-29 `motion_value.json` and 11-02 `v3_data_curve.json`: the direction error falls about 13 to
    17 % per doubling of training data on IDs 1 and 2, the contact point 11 to 19 %.
  - The direction errors at 0.5 N and under are 14 to 18 deg; over 3.5 N they are about 2 deg.
  - Torsion has no skill on IDs 2 and 3, so tonight's extra runs leave twists out.
  - So: more coverage, low forces, and contact speed.
- The overnight plans:
  - The workstation's coverage set: flank obliques at 10/20/30 deg and 8 azimuths, the face and edge points without
    obliques, levels from 0.25 N. Built from 17:00 if its login lasts.
  - The speed set (`probe_speed_plan.py speed`): 12 existing points a seat (the reference, 4 face, 2 + 2 flank,
    3 edge) at 0.05, 0.15, 0.6 and 1.2 mm/s, one ramp to 3 N (2 N on flanks) held 1 s.
  - The fallback (`... fast`): the grid's 264 presses at 0.6 mm/s.
  Rests, push lines and depths are the checked ones.
- The simulator moves at the plan's speeds whatever the slider (`ur5_tip_calibration_sim.py`: "the simulator moves at
  the plan's speeds whatever the slider"). So every simulation so far ran 3.3 times faster than the robot, which is the
  harsher case for a press's overshoot. Simulated at the real speeds (v and fast_v times 0.3), the 2.0 mm/s presses
  overshot 3 N to 5.2 N on average and 8.2 N at most; at 0.6 mm/s, 3.7 and 4.6 N. With the push's 20-sample mean the
  stop comes late, and the owner said more force isn't needed, so the top speed is 1.2 mm/s.
- Seat 1's reference press, 15:06 to 15:55: the depth at 0.5 N held at 1.75 to 1.79 mm, and at 1 N at 1.89 to 1.94 mm.
  The depth at 3 N went from 2.53-2.57 to 2.40-2.44 mm. The pad stiffens at the higher force; the tip does not move.
  The HEX-E's raw Fz at rest climbs about 2.8 N/h since its restart at 14:34. Each action tares, so a press sees about
  0.01 N of it.

## Seat 1 done; the chain on seats 2 and 3, and the overnight list (11-02, from 16:2x)

- Seat 1 (v3 ID 6, s1v6b) ended at 18:27:09 with exit 0. Pass 1 landed 20 of 20, the grid 294 of 294, with no abort.
  The HEX-E gave 1,300,309 samples, none timed out, with two gaps of 0.6 s outside any action. Copy:
  `20261102-145010-s1v6b/` (the small files and the printout); the streams stay on corvus.
- A press took 28.9 s on average and a twist 145 s, so a seat's main run takes about 3.6 h.
- Seat 1's reference press over the run, its stop at 3 N against the first: -0.165 to +0.057 mm along the push (the
  low between 15:43 and 16:11, back by 16:32), at most 0.031 mm across. At the end it stood +0.012 mm.
- The workstation built the overnight coverage set (17:4x, 5a6ff549). The full set reaches 1,285 to 1,289 obliques a
  seat, about 9 h a seat, and stays for later nights. Its night subset runs 221 to 222 obliques plus 10 references a
  seat:
  - flank points: theta 10, 20 and 30 at two opposite azimuths;
  - face and edge points: theta 20 at two;
  - the pair turns 45 deg from point to point.
  The lab's plans (`20261102-plan_probe_table_night_seat*`) drop pass 1, and every id ends in -n: 20 a seat were the
  main plan's ids, which a resumed session skips as done.
- The workstation replayed the whole overnight list chained from seat 3's end: 0 of 1,521 travels flagged. Three night
  flank obliques came 0.33 to 0.37 mm from the rigid tip body at their 1.0 mm depth limit (the margin is 0.4 mm); they
  are capped at 0.75 mm (83150a9a).
- All nine overnight runs passed their chained simulations at the robot's real speeds, with no abort. Each has its
  marker. The order: speed3, speed2, speed1, night1, night2, night3, fast3, fast2, fast1.
- The chain started seat 2 (s2v5) at 18:27:09. It went by seat 2's hub from seat 1's end, and pad-ref-00's onset came
  0.96 mm early (seat 1's: 1.38).

- Seat 2 (v3 ID 5, s2v5) ended at 22:03:30 with exit 0. Pass 1 landed 20 of 20 (0.051 mm rms) and the grid 294 of
  294, with no abort. The HEX-E gave 1,296,427 samples, none timed out. Copy: `20261102-182710-s2v5/`. Its reference
  press moved +0.13 mm at 3 N by 19:17 and was back within 0.01 mm by 20:08; its 0.5 N depth held at 1.76 to 1.83 mm.
- The chain started seat 3 (s3v4) at 22:03:37. pad-ref-00 came 0.15 mm early, and pass 1 landed 20 of 20 at 0.051 mm
  rms.
- The pads against the seat CAD in the two-key frame, seats 1, 2 and 3 (y_A -80, 0, +80):
  - turn about y_P: -4.07, -3.41 and -2.82 deg;
  - move along x_P: +1.61, +1.11 and +0.54 mm;
  - turn about x_P: +0.56, +0.26 and -0.50 deg; about z_P: -0.26, -0.74 and +0.04 deg.
  That is about -3.4 deg common to all three, plus +0.6 deg and -0.53 mm per 80 mm along the strip. It reads as the
  frame's roll about y_A, which the rocking key seatings cannot fix, plus a slight twist along the strip. Each seat's grid
  aims by its own fit, and at the fitted poses the grids flag nothing (joint 5 -94 to -76, joint 6 -18 to +14).

- Seat 3 (v3 ID 4, s3v4) ended at 01:43:27 on 11-03 with exit 0: pass 1 20 of 20 (0.051 mm rms), the grid 295 of
  295, no abort. The HEX-E gave 1,317,359 samples, none timed out. Copy: `20261102-220337-s3v4/`. Across the three
  seats' main runs, 60 touch-offs and 883 grid actions ran, and none aborted.
- The chain began the overnight list at 01:43:35 with speed3 on seat 3's session (resumed: 317 actions done, 20
  touches). The first eight presses, the peak load against the rest (3 N target) and the press's time:

  | Speed | Peak | Time |
  |---|---|---|
  | 0.05 mm/s | 3.7-3.8 N | 30 s |
  | 0.15 mm/s | 3.7-3.9 N | 10 s |
  | 0.6 mm/s | 4.2-4.6 N | 3.3 s |
  | 1.2 mm/s | 5.0-5.4 N | 2.2 s |

  At 1.2 mm/s the stop came 0.4 mm deeper than at 0.05 mm/s, 0.3 mm short of the press's depth limit. The simulator at
  the real speeds had given 3.6, 3.7, 4.1 and 4.7 N.

- The overnight list, from 01:43 on 11-03 (each run on its seat's session; printouts and plans in each session's copy):
  - speed3, speed2 and speed1: 01:43 to 02:47, 48 of 48 each;
  - night1, night2 and night3: 02:47 to 08:31, 232, 231 and 232, about 2 h each;
  - none aborted, and the HEX-E never timed out. Seat 1's reference press stood within 0.03 mm of 15:06's at 03:13.
- fast3 (the fallback) started at 08:31. fast2 and fast1 follow, about 1.8 h each.

- fast3 (the fallback on seat 3) ran 08:31 to 10:07:55, 263 of 263, no abort. The chain started fast2 at 10:07:57.
- 11-03, 10:1x, the owner (relayed by the workstation): "yea, we can pop it on after the fast3 pas" (the 2 mm sphere).
  So fast2 and fast1 came off the list, and fast2 was stopped at 10:16:40 after 25 of 264. The stop was a SIGINT to its
  PID, the program's own Ctrl-C (stopL, files closed), sent 1.5 s after its next action began, while the robot backed
  away from the pad. The chain then stopped (not clean, as designed). The arm holds over seat 2, the flange at z_B 237
  mm.
- The 2 mm sphere's plans (the workstation's 35bd1937, pad-fitted frame, no twists) are converted, simulated chained at
  the real speeds and replayed clean. The rod's plans (a4b4f679: safe_mm 40, the hub 90 mm over the apex, the rod across
  and along the tip) are converted. The program's rod support (geometry kind rod, the line distance, the rod's fit
  across its axis with the pads seeded from the 2 mm sphere's session) is written in `logs/rodwork`. A sphere session
  prints the same with it as before. In the simulator the rod's fit gives 0.12 mm rms against the spheres' 0.03, which
  is not good enough yet.
- A mistake at 10:10: a `kill` of PIDs from `pgrep -f` on a session name matched its own shell, as CLAUDE.md lesson 9
  warns. It stopped only that command and a simulation. Since then sessions are stopped by explicit PIDs and relaunched
  from script files.

- The 2 mm sphere's plans simulated chained at the real speeds, seats 1, 2 and 3 from fast1's end: pass 1 20 of 20 on
  each (0.035, 0.027 and 0.037 mm rms) and 260, 269 and 271 of 260, 269 and 271 presses, no abort. The workstation's
  replay as pushed: 0 of 860 travels flagged.
- The rod in the program (installed 10:4x; gated on geometry.probe.kind == "rod"):
  - Geometry reads axis_M and half_len, and samples the pad's surface densely (vertices, centroids, edge midpoints).
    line_distance gives the surface's least distance from the rod's axis within half_len.
  - register fits the rod with fit_rod_registration. That fits the pad's T_AP about the seed (prior as the spheres'),
    the rod's axis point across the axis (2, prior 1 mm) and the axis' turn about z_M and across (priors 4 and 2 deg).
  - The simulator's probe contact uses the distance to the rod's axis within half_len, and the walls are skipped for a
    rod.
  - `probe_seed_session.py` seeds the rod's session from the 2 mm sphere's session of the same seat: its registration,
    c_M set to the rod's, the homing marked done. The run resumes the seed with stages preflight,pad,register,grid.
- A seed alone does not fit the rod. With the pad held at the 2 mm sphere's fit and only the rod's place fitted, the
  simulator's rod touches gave 0.12 mm rms. Under the simulator's true geometry the same touches agree to 0.05 mm. A
  sphere's touches leave the mount's error on the flange (0.8 mm and 0.9 deg in the simulator) in T_AP, which suits the
  sphere's centre but not the rod's line: the seeded pad stood 1.6 mm and 0.7 deg off. With the pad fitted again with the
  rod, the same touches give 0.015 mm rms, and the pad comes within 0.3 mm and 0.5 deg.
- Seat 1's rod session in the simulator: pass 1 20 of 20 (0.015 mm rms), the grid 177 of 177, no abort. A 2 mm sphere
  session prints line for line the same with the rod code as without it. The converter takes tool "rod": the pass 1
  from the rod-across actions, safe_mm and retract_mm from the seat plan (the rod's 40 and 12; the spheres keep 15).
- A correction to fast3: it ran 252 presses, not 263. Its other 11 ids (`probe-flank-413-v060` and ten more) were the
  ids of speed3's 0.6 mm/s presses on the same session, so `--resume` skipped them as done. Speed3 has those 11 at the
  same speed. The printout's counter numbers the plan's actions, so it ended at (263/263) all the same.

### s1sph2: a touch on the tip's body and a bad fit

- 10:5x, the owner, in the lab's session: "ok the 2mm radius sphere is on there. the arm is oevr the seats. you can go
  ahead". The arm stood over seat 2, moved a little by hand during the swap (joints 54.2, -79.1, 105.9, -114.1, -86.6,
  -2.0 deg). Seat 1's first travel from there, replayed with the workstation's dry-check --real, flagged nothing.
- 11:00:37: the chain `logs/20261103-chain_sphere2.sh` started seat 1 (v3 ID 6) as s1sph2. Preflight: HEX-E 100 Hz, held
  channels 0, 2 and 3 at 137 records/s.
- Pass 1: 19 of the 20 touches came -0.92 to +0.08 mm from the expected, most near -0.7 mm. So the pad stood about 0.7 mm
  nearer than the pad-fitted frame put it.
  - The 17th, pad-tip-349-o35-a270, saw 0.54 N in its fast approach, then an onset 4.81 mm early.
  - The workstation's replay had given that touch's own moves 0.76 mm to the tip's body, the least of the pass.
  - The tool met the tip's body, not the pad: under the fit of the other 19 it stands 4.66 mm off the pad.
- register took all 20 in: 0.758 mm rms, the pad 5.71 mm and 8.65 deg from its CAD, and c_M (2.31, -5.22, 122.11) mm.
  Without that touch, the 19 fit to 0.051 mm rms, with the pad 1.53 mm and 1.34 deg from its CAD and c_M (-0.10, -0.20,
  117.35) mm.
- The program had no gate on the fit, so the grid ran on it:
  - 23 actions, with 10 contacts before the slow approach and 8 presses that reached no level;
  - two travels aborted on the load, at 8.8 N (probe-flank-445) and 8.4 N (probe-flank-430-o35-a180).
  - None of the 2,400 actions of 11-02 and 11-03 had an early contact or an abort.
- The lab's monitor of the run printed nothing, because `cut` at the end of its pipe held the lines. The lab saw the fit
  at 11:24 by reading the printout.
  - The chain was stopped by its PID, and the program by SIGINT to its PID (its Ctrl-C), at 11:25:04.
  - The arm stopped over seat 1 (joints 38.3, -70.1, 97.3, -103.1, -67.0, -7.6 deg), at rest before probe-flank-440's
    approach.
  - The HEX-E read (0.62, -0.30, -0.03) N, its rest reading, so the tool was clear of the pad.
- The fix in the program:
  - register in probe mode leaves out a pad touch standing over REG_DROP_MM = 0.5 mm off the pad, the worst first and at
    most 3, and prints each one;
  - it stops the session before the grid when the rest stand over REG_RMS_MAX_MM = 0.15 mm rms (the registration kept
    as it was);
  - a run stops at its 3rd contact before the slow approach (EARLY_STOP) or its 3rd press with no level (MISS_STOP).
- In the simulator:
  - s1sph2's own touches leave out pad-tip-349-o35-a270 (+2.74 mm under the fit that held it) and fit to 0.051 mm rms;
  - s1sph2's bad fit resumed in the grid stops at the third early contact (without the miss stop: 80 presses reached no
    level first). With the miss stop it stops at the third press with no level (probe-flank-445, probe-flank-430-o35-a180
    and probe-flank-440), exit 1.
- The plans: `20261103-plan_probe_table_sphere2_seat{1,2}_v3id{6,5}_no349.json`, the plans of 8c3ac200 less
  pad-tip-349-o35-a270. Seat 2's pass 1 has the same touch, also 0.76 mm from the body. Seat 3's has none (its least is
  1.41 mm), so its plan stays as it was.
- The workstation's exact meshes (11:5x) agree. At its rest the touch stood 0.71 mm from the body, and 0.69 mm where it
  met it, so the body sits where the model has it relative to the pad, within 0.1 to 0.2 mm. The frame's 0.7 mm error
  before the fit is the whole cause.
  - The other pass 1 touches stand 1.45 mm or more from the body, and at least 0.55 mm with the frame 0.9 mm off.
  - The grid's presses come at least 0.89 mm from the body at their depth, aiming by a fit of about 0.05 mm. They stay.
- The replays of the restart chain from the robot's joints (the lab's wrapper of dry-check --real, and the workstation's
  `--start-q`) flag no travel among 858. The only flags are this morning's three presses at their margin against their
  own pad.
- The fixed program in the simulator (ada1ee4b), at the real speeds:
  - seats 1 and 2 (_no349) 260 of 260 and 269 of 269, pass 1 at 0.037 and 0.028 mm rms (19 touches);
  - seat 3 271 of 271, printing line for line as before the fix;
  - no early contact, miss, abort or touch left out;
  - the rod's register on the fixed program fits as before (f_rod2's touches 0.013 mm).
- The rod's plans, converted again at 12:0x on the workstation's d8cecd9e (fe4e1914):
  - pass 1 takes only presses whose touch-off keeps 2.0 mm from the tip's body (touch_body_mm), giving 20 a seat, all
    3.14 mm or more;
  - ce3a3f0b's plans had five at 1.21 to 1.48 mm (pad-flank-553-r90, pad-flank-563-r90, pad-flank-582-o20-a0-r90,
    pad-edge-573-r90 and pad-edge-569-r90), and are in `superseded/`;
  - the depth is lowered on 11 presses a seat, where the rod came within 0.4 mm of the body;
  - the workstation's replay (bb79d68e) flags none of 587 travels.

### The restart, s1sph2b, and quicker travels

- 12:1x, the owner, in the lab's session: "currently, the 2mm radius sphere is not damaged or contacting anything. it
  is almost grazing the flank of the the fingertip. you have the goahead to contnue, although ive asked the lab session
  to see if we can speed up these fingertip calibrations on these passes".
  - Under s1sph2's good 19-touch fit, the sphere stood 0.6 mm from the pad at the stop pose.
  - Backing along the stem opens the gap steadily: 1.18 mm after 1 mm back, 12.4 mm after 15 mm.
- 12:20:20: `logs/20261103-chain_sphere2b.sh` started seat 1 as s1sph2b, a new session on the _no349 plan.
  - The first travel backed off 15 mm along the stem, went by the hub and came in to pad-ref-00 (-0.588 mm).
  - Pass 1: 19 of 19, 0.060 mm rms, none left out. The pad stands 1.806 mm and 1.960 deg from its CAD, and c_M is at
    (-0.15, 0.03, 117.21) mm.
  - The 19 touches stand within 0.135 mm of s1sph2's (mean -0.014 mm, rms 0.068 mm). Both edge-639 obliques came
    0.13 mm earlier and edge-666 0.11 mm later.
  - The pad's fit moved 0.27 mm and 0.6 deg from the morning's 19-touch refit. The tip may have turned slightly in its
    seat, in the incident or in the owner's look.
- Two flank obliques with max_depth_mm 1.0 (probe-flank-430-o35-a0, probe-flank-438-o35-a90) stop at 1 N: the 2 mm
  sphere reaches its depth there before 2 N (reached_max). The simulator and the 5 mm sphere reached 2 N on them.
- The speed-up (the workstation's timing at the 0.30 slider, 12:19): a straight travel moves at 6 mm/s back and across
  and 3 mm/s in, and the return to the rest at 0.6 mm/s.
  - The quick plans set travel.probe.retract_mm 6 (was 12) and travel.probe.in_v 0.02 (the in-move at 6 mm/s), for
    about 4 s a press, or 20 min a seat. They are `20261103-plan_probe_table_sphere2_seat{2,3}_v3id{5,4}_quick.json` and
    `20261103-plan_probe_table_rod_seat{1,2,3}_v3id{6,5,4}_quick.json` (4f54fc80).
  - The program reads in_v and travel.back_v, with defaults as before (6d0a05ef).
  - The return to the rest stays at 0.6 mm/s. The analysis interpolates the HEX-E's and the board's zeros between
    t_rest0 and t_rest1, and a quicker return would start t_rest1 sooner after release (the workstation, 12:21). Before
    any change, a check of the board across t_rest1 in the 11-02 streams.
  - The workstation's --retract 6 replays of today's exact plans (6e53dc57) flag none of 858 and 587 travels, with the
    same own-press flags. The quick plans differ from them only in retract_mm and in_v.
  - Simulated at the real speeds: seats 2 and 3 269 of 269 and 271 of 271 (0.033 and 0.032 mm rms), and the rod's
    three seats 177, 176 and 173 of the same (0.017 to 0.021 mm rms). No early contact, miss or abort.
- 12:58:17: `logs/20261103-chain_sphere2c.sh` took over from chain_sphere2b, which was stopped by its PID while s1sph2b
  ran on. It waits for s1sph2b's end, then runs seats 2 and 3 on the quick plans.
- The lab's second monitor of the run also printed nothing at first. corvus's `awk` is mawk, which reads a pipe in
  blocks, so fflush alone did not help; `awk -W interactive` does. The lab now tries each monitor's pipe on the log for
  5 s before arming it.

### Seat 1 done; the deeper and faster plans for seats 2 and 3

- s1sph2b ended at 14:48:02 with exit 0. Pass 1 ran 19 of 19 (0.060 mm rms) and the grid 260 of 260, with no early
  contact, no press without a level, and no abort. Copy: `20261103-122020-s1sph2b/`.
  - The HEX-E gave 884,840 samples with none timed out. Its two gaps (1.0 and 0.7 s) came at the program's start, 12:20:20,
    before any action, as at s1sph2's start.
  - Grid presses took 30.5 s on average, and pass 1's touches 37.3 s.
- 40 of the 260 grid presses stopped below their top level, at their depth (reached_max). Of these, 28 were flanks, 11
  edges and 1 a tip end; 26 were o35, 10 o20 and 4 normal.
  - 26 stopped at 1 N of 2 N with max_depth_mm 1.0, and 8 at 2 N of 3 N with 1.25.
  - The workstation's neck rule (840f40da) deepens these presses on seats 2 and 3. The stem's points within 4 mm of the
    sphere's centre keep 0.5 mm from their own pad, and the rest of the stem 1.0 mm from everything.
- The owner's go in the lab's session: "yes please go" (13:5x) to the deeper flank presses, and "Yes go on the faster
  presses too" (13:5x).
  - The plans: `20261103-plan_probe_table_sphere2_seat{2,3}_v3id{5,4}_quick_neck_fast.json`. They are the quick plans
    with 170 and 169 presses deeper, and the grid presses at v 0.002 (0.6 mm/s).
  - The deeper plans' sims passed (l_sph2_2 and l_sph2_3, 269 and 271 of the same, clean).
  - The fast sims (n_sph2_2 and n_sph2_3) had run 96 and 88 grid presses clean when /dev/shm filled at 14:02. The
    owner, relayed at 14:23, said a change of speed alone needs no sim, so the gate (the marker
    `logs/20261103-ok-fast`) was lifted at 14:24:43.
  - The workstation's diff: the fast plans differ from the replayed _quick_neck plans only in v.
- /dev/shm, 16 GB of corvus's 31 GB of RAM, filled at 14:02 with the lab's sim sessions and the time-lapse.
  - The sims and the time-lapse died with ENOSPC. s1sph2b writes to the disk and ran on.
  - The lab deleted the sim sessions of 10-23 to 11-02, whose results are above, freeing 10 GB. It restarted the
    time-lapse at 14:24.
- 14:48:07: chain_sphere2e started seat 2 (v3 ID 5) as s2sph2 on the fast plan. Its first touch, pad-ref-00, came
  -0.543 mm from the expected.
- The speed sets of 11-03 (3c849789, `speed_set_summary.py`, read on corvus at nice 19 beside s1sph2b):
  - holds land +0.10, +0.24, +0.85 and +1.73 N over their level at 0.05, 0.15, 0.6 and 1.2 mm/s;
  - the board's change follows the realized push, |board| ~ push^1.03 per point, every speed within about 2 % and
    with a direction cosine of 0.999 median;
  - t_rest1 starts 2.85 s after the release, and the board still drifts 1.6x its noise within it, so the return to
    the rest stays at 0.6 mm/s.
- The owner's cone sweep (QUEUE R32): the lab wrote the sweep action in `logs/sweepwork`, not yet in src.
  - At each level stop the sphere steps to offsets on rings in the pad's tangent plane via the stopped pose, and slides
    1 mm out and back at 2 N.
  - A sideways push over ft_t_cap_N aborts. Over 1.2 x the push along + 0.5 N, it skips the rest of that level's
    offsets or slides.
  - The test plan, 10 face sweeps on seat 3, is `20261103-plan_probe_table_sphere2_seat3_v3id4_sweep_test.json`
    (4d21a74c). It replays clean.
  - In the simulator at mu 0.45, all 10 sweeps ran whole (56 offsets and 4 slides each, about 118 s). At mu 1.0, all 10
    ran, with face-717's slides skipped.
  - The simulator's probe contact has friction (Truth.mu, 1.0 by default), not normal-only as the lab had told the
    workstation at 12:53.

### The sweep action in the program, and the sweep test on seat 2 (from 15:3x)

- 15:3x, the owner, relayed by the workstation: "sure, yes to all these suggestions. we can put the rod on hopefully
  soon". The suggestions were:
  - the sweep test after seat 3;
  - sweeps in place of grids for the rod and later;
  - the slider at 1.0, with the in-contact speeds rescaled;
  - rolls with the rod or a flat tool;
  - forces to 8 N.
- 15:4x, the owner, in the lab's session: "relays count yes, we can go on the sweep test after 16:37 since i doubt we're
  getting very good info right now, also im gonna compact".
  - From now on, a go that the workstation relays with the owner's words and time counts as the owner's go.
  - The sweep test runs on seat 2 after s2sph2, in place of seat 3's 2 mm grid. Seat 3 gets no 2 mm sphere session
    today. Its rod session will be seeded from s3v4 (the 5 mm sphere, two-key frame).
- 15:42: chain_sphere2e was stopped by its PID. s2sph2 ran on.
- The sweep action is in the program (a2f44c8f, from `logs/sweepwork`):
  - press() takes on_level. It is called after each level's hold, and at the press's depth for a level not reached.
  - At each level, the sphere steps to that level's offsets on rings in the pad's tangent plane (t1 from target.t1),
    each from and back to the stopped pose. At at_N it slides out and back.
  - Every move and hold watches the load as a press does, and the push across the pad's normal. Over ft_t_cap_N the
    sweep aborts. Over ft_t_ratio x the push along + ft_t_floor_N, the rest of that level's offsets or slides is
    skipped.
  - In the simulator, seat 3's quick plan printed line for line as before (o_regress3 against k_sph2_3).
- The test plan is `20261103-plan_probe_table_sphere2_seat2_v3id5_sweep_test.json`, the workstation's 8189c5ff less
  pass 1.
  - It has 12 grid actions: two reference presses (probe-ref-00 and probe-ref-final) around 10 face sweeps.
  - Each sweep has levels 0.5, 1, 2 and 3 N, 56 offsets on rings of 0.06 to 0.24 mm, and four 1 mm slides at 2 N.
  - ft_t_cap_N is 3.0 and max_depth_mm 2.0. The press moves at v 0.0005, the offsets at 0.002 and the slides at 0.0005:
    0.15, 0.6 and 0.15 mm/s at the 0.30 slider.
- 15:43:48: `logs/20261103-chain_sweep2.sh` started. It waits for s2sph2's end, resuming it after a HEX-E loss. Then
  it seeds s2sweep from s2sph2's session with `probe_seed_session.py --keep-probe` (cf1e200f: the source's registration
  and c_M, for the same tool), and runs s2sweep with stages preflight,grid.
- s2sph2 at 15:48, 152 of 269:
  - pass 1 19 of 19 at 0.053 mm rms, the pad 1.497 mm and 1.378 deg from its CAD, c_M (-0.45, -0.44, 117.07) mm;
  - no early contact, no press without a level, no abort.
  - 8 of the 152 stopped below their top level. Of the 146 ids pressed on both seats so far, 32 stopped short on seat 1
    and 8 on seat 2. So the neck rule's deeper presses reached their top on 27 of seat 1's 32.
  - The edge-654 family stops at 2 N of 3 N at 1.75 mm, and probe-flank-582-o35-a90 at 1 N of 2 N at 1.5 mm, as on
    seat 1.
- s2sph2's pass 1 files (touches, registration, session and plan) are in `20261103-144808-s2sph2/`, for the
  workstation's joint fit of c_M.

### The speed slider at 15:42, the sweep test stopped, and the night's rod run (from 16:00)

- s2sph2 ended at 16:11:13 with exit 0: the grid 269 of 269, no early contact, miss or abort. 29 presses stopped short
  of their top level, against seat 1's 40 of 260. The HEX-E gave 497,626 samples with none timed out. Copy, with
  events and printout: `20261103-144808-s2sph2/`.
- **The robot's speed slider went from 0.30 to about 0.50 between 15:42:11 and 15:42:25.** From
  probe-edge-639-o20-a0 on, every move ran 1.67 times faster: the grid's 1 mm fast approach took 0.85 s, not 1.42 s,
  and the press's slow approach ran about 1.0 mm/s, not 0.6.
  - No program set it: the program sets the slider only at its start. None started then, and no one logged in. It was
    most likely moved at the pendant (the owner asked).
  - The sweep test's start set it back to 0.30 at 16:11:15. Its touches ran 2.5 mm in 2.41 s, as at 14:48.
- How it showed, and how the lab first misread it:
  - The program stops for a level about 0.9 s after the force crosses it, as it did all session. At the faster speed
    each stop came about 0.45 mm deeper. The reference press's 0.5 N hold stood at 2.60 to 2.64 mm of travel from
    ref-05 on, against 2.13 to 2.19 mm before.
  - The 3 N presses peaked at 5 to 7.8 N (face-717 7.8 N), against 3.7 to 4.6 N before. The edge-639 family, the
    first at the faster speed, peaked at 5.7 to 7.5 N.
  - The lab read this at 16:0x as the sphere or the tip moving: a "step" of the reference press, with a stiffer end.
    It changed the sweep test to a new session with its own pass 1 (chain_sweep2b,
    `20261103-plan_probe_table_sphere2_seat2_v3id5_sweep_test_pass1.json`) and told the workstation so.
  - The test's touches then found the pad where it was at 14:48: onsets within 0.07 mm on 10 touches, and the travel
    from onset to stop unchanged (-0.017 +- 0.077 mm, as seat 1's two sessions).
  - The force against travel at the reference point, smoothed over the HEX-E's 0.24 N noise, did not change either: 0.5
    N at 1.93 to 1.97 mm of travel in every reference press. The tips and the sphere did not move.
  - The HEX-E streamed at 100 Hz all session and the held board at 455 records/s, so neither caused it.
- 16:1x, the owner, in the lab's session: "if you pause now i can check the seating".
  - chain_sweep2b was stopped by its PID. The program was stopped by SIGINT to its PID at 16:17:26, right after
    pad-edge-639-o35-a0's onset line (its return to the rest done): "interrupted: stopping the robot", exit 0 at
    16:17:29.
  - 10 of the 19 pass 1 touches had run. The arm stands over seat 2 at joints (55.4, -80.1, 109.2, -121.5, -87.9,
    0.2) deg, TCP (-198.0, -487.0, 238.4) mm, with the HEX-E at its rest reading.
- The owner's words, relayed by the workstation:
  - 16:1x: "ok i just checked and confirmed that all were seated. it could be the deflectin of the small spehre
    indenter. also, this is probably going to be the last calibration we do for a while, we're mounting the sr2 hand
    after this. so thsi rod experiment better be good."
  - 16:2x: "to be clear we can still do the rod experiments, just tmw the hand is going on. so i pray to god you get
    as good information as we did ont he v3 ids 1-3 fingertips + the better coverage + angles i was hoping for with
    the inverted rig".
  - 16:3x: "ok i reenabled remote control. itll go on early tomorrow morning, but seems like you have time budget for
    it + additional plans (that i preemptively give yo uthe go-ahead now on since ill be asleep) to improve
    calibration".
- The night's run, as the workstation proposed at 16:22: rod sweeps across and along the tip (rod_deg 90 and 0) on
  all three seats, levels 0.5, 1, 2, 4 and 8 N, slides at 2 and 4 N, then twists. v3 ID 3's rod block 6 is the
  standard: 70 presses (to 8 N), 24 slides, 4 rolls and 2 twists.
- The program (d1b5ea6f):
  - the speed slider is read before every action, and the run stops if it is off the plan's by over 0.02;
  - a probe twist backs off to 0.5 mm behind its rest. It used to back off st + 0.5 mm from its stop, which left a
    press that ran to its full depth on the pad (the workstation's finding of 15:3x).
- In the simulator, on the rod seeded from seat 2's simulated rod session, at the real speeds:
  - the rod sweep plans of c867f7d4 (3 N, rod across; three of seat 2's sweeps): whole at mu 0.45. At mu 1.0 the
    ratio check skipped some 2 N slides.
  - two 8 N sweeps (abort_N 12, ft_t_cap_N 6): whole at mu 0.45, with 64 of 64 offsets and 8 of 8 slides each, and
    0.8 to 2.7 N across at 8 N. At mu 1.0 one was whole and one aborted on the 6 N cap (6.00 N across at 6.04 N
    along). So the 8 N plans take ft_t_cap_N 10.
  - a face and an edge twist (2.5 N; 5, 10, 15, 10 and 5 deg; turn_v 0.016): every turn ran, with peaks of 3.9 N
    and 0.1 N m, and each ended 0.500 mm behind its rest.
  - the 3 N rod sweep simulation of 16:08 on the new program prints line for line as before.
- v3 ID 3's twists on the robot (rod-twist-019, at 2.5 and 5 N) took 86 s each at turn_v 0.016, and its rolls 69 s.
- Corrections: the commit 05a5ea2e is stamped 16:06 but was made at 16:04, and the lab's message to the workstation
  stamped 16:36 was sent at 16:34. The lab's message of 16:27 went to the wrong session (the workstation's had
  restarted); the answers went into STATE (3efa4802) and again to the workstation's new session.

### The rod: a quarter turn, the fit, the trims and the HEX-E (from 16:5x)

- 16:5x, the owner, in the lab's session: "ok it's on there godspeed go". The arm moved in the swap, to joints (49.72,
  -83.46, 116.43, -141.09, -95.65, 0.13) deg. The workstation's checker, run from there, flagged 0 of seat 2's first
  travel and 20 touches.
- 16:58:23: `logs/20261103-chain_rod_night.sh` (PID 2859347) seeded seat 2's session from s2sph2 with the rod's c_M
  (0, 0, 107.5) and started rod21.
  - Pass 1 met the pad from -3.29 to +1.33 mm of where it was expected.
  - register left out pad-edge-628-r90 (+1.85 mm off the pad) and pad-face-717-r90 (+0.51 mm). The other 18 stood
    0.229 mm rms, over the 0.15 mm gate, so the session stopped before the grid (exit 1 at 17:15:54), as designed.
  - The rod tool sat a quarter turn off on its square boss. The workstation's CAD model, the lab's forward model and
    the camera agree. The forward model fits the touches at 1.59 mm rms with the tool as planned, and at 0.31 mm turned
    about 112 deg. Copy: `20261103-165823-rod21-turned/`.
- 17:2x, the owner: "ok it's rotated please go". At 17:26:47 the chain (PID 2862548) started again from a fresh seed of
  s2sph2.
  - Pass 1: 20 touches at +0.13 to +1.33 mm. The fit stands 0.054 mm rms, none left out. The pad moved 0.648 mm and
    0.868 deg from the seed.
  - The rod's axis moved 1.168 mm across itself: c_M (-0.838, 0, 106.687) mm in M. The rod turned -5.40 deg about z_M
    (sd 2.5) and +0.77 deg across. Copy: `20261103-172647-rod21/`.
  - From 17:46 the grid ran probe-ref-00 and two sweeps to 8 N, the first on the robot (sweep-edge-573-r0 and
    sweep-edge-634-r0). On 634 the friction check stopped the 0.5 N offsets at r 0.06 az 135 and the 2 N slides at az
    270.
- With the fitted turn, the workstation's exact margins put a few slides within 0.2 mm of a tip body.
  - The lab stopped the chain by its PID at 17:55:11. It stopped the program by SIGINT after grid 3's finished line
    (exit 0 at 17:55:49).
  - It took those slides and sweeps out of all three seats: the `_sweep8_fit` plans (STATE lists them).
- 18:01:10: the chain started again (PID 2867047, `logs/20261103-rod_night_steps2.txt`), resuming seat 2's session.
  - The HEX-E dropped on the resume's first travel, a large joint 6 turn. Three aborts ("no HEX-E sample for 100 ms")
    stopped the run (exit 1 at 18:01:33). Nothing was pressing.
  - The chain probed the sensor every minute: no samples from 18:01:38 to 18:49:16, 199 of 300 at 18:50:19, and
    streaming at 18:51:23. The owner had reseated the connector (relayed at 18:5x: "i already reseated it...").
  - The first resume (rod21r1, 18:51:28) lost the HEX-E again on its first travel: 155 reads timed out, exit 1 at
    18:52:00. The second (rod21r2, 18:52:17) ran on, and sweep 4 of 85 finished at 18:58.
- The hard stop at 05:50 held the first chain's PID: the chain restarted at 18:01 under a new one. At 05:50 it would
  still have stopped the program, and the chain with it. But a chain waiting out a HEX-E loss would have run the robot
  again once the sensor came back, maybe while the owner mounts the hand.
  - `logs/20261103-deadline_stop.sh` now finds every running chain at 05:50 and stops each by its PID (dc183040).
  - It was rearmed at 18:58 (PID 2874853). The old stop was killed by its PID, with its sleep.

### The night in halves (from 19:0x)

- Seat 2's sweeps take 258 s each on the robot (573, 634 and 638), against the plans' estimate. Roughly:
  - the travel in and the return out, about 31 s;
  - the press and its five level holds, about 31 s;
  - 64 offsets, about 140 s: 1.8 s each at 0.5 N and 2.6 s at 8 N;
  - 8 slides, about 60 s: each 1 mm out at 0.3 mm/s and back.
- At that rate seat 2's _fit plan ends about 23:40 and seat 3's about 04:45. Seat 1 would get its pass 1 and about 10
  sweeps before 05:50, and no seat its extras.
- The lab proposed lighter sweeps (19:0x). The workstation's pick, under the owner's go of 16:3x: lighter sweeps in
  halves, so that a HEX-E loss after the owner is asleep leaves every tip half its coverage.
  - Offsets at azimuths 0, 90, 180 and 270, every ring and level kept: 32 a sweep.
  - Slides at az 0 and 90 at 2 and 4 N: 4 a sweep. Each slide returns the opposite way, so ±along and ±across stay at
    both forces.
  - Twists at 5 N only, after the second half's sweeps, so that a cut takes twists before locations.
  - The 4 N rolls of each seat's extras (3 or 4, about 70 s each) at the very end.
  - The obliques of the extras are dropped for tonight.
- A lighter sweep should take about 157 s. First halves: seat 2 to about 21:00, seat 3 with its pass 1 to about 22:40,
  seat 1 to about 00:30. Second halves: 1, 3 and 2, to about 05:45.
- `20261103-rod_halves.py` writes the six plans (`..._half_a.json`, `..._half_b.json`). Half a holds every other
  sweep in the _fit plan's order, which is sorted by the stem's direction, so each half spans the pad and both rod
  turns. Half b holds the rest, then the twists, the last reference press and the rolls. The reference presses take new
  ids (probe-ref-a1, -b0), so that a resumed session runs each.
- The switch:
  - The chain was killed by its PID at 19:09:41. Sweep 713 had ended at 19:09:19, so the program ran 618 too and got
    its SIGINT right after 618's line (8/85), at 19:13:35; exit 0 at 19:13:38. The HEX-E gave 128,089 samples with
    none timed out.
  - Seat 2 so has 7 sweeps in full and probe-ref-00: 573, 638, 630 and 618 in half a; 634, 569 and 713 in half b.
    (This line said 8 sweeps, 4 in each half, until 19:3x.)
  - The arm stood at 618's rest, joints (35.700, -78.985, 115.334, -113.396, -47.707, 41.349) deg, within 1.2 deg of
    its q_rest. The workstation's checker from there flagged 0 of half a's 27 actions: joint 6 -56.4 to 42.2 deg, the
    rod at least 0.50 mm from the tip body in its own moves.
  - At 19:15:01 the chain (PID 2881408) started on `logs/20261103-rod_night_steps3.txt`. The first light sweep,
    sweep-face-697-r0, ended at 19:17:52 with its hub travel; the HEX-E held.
- 19:1x, the owner, relayed by the workstation: "i mean i wont be in until 9". The SR2 mounting starts about 09:00.
  - The workstation proposed no step after 08:00 and the hard stop at 08:30, and each seat's extras after the second
    halves: 2, 3, then 1.
  - The hard stop was rearmed for 08:30 at 19:10:56 (PID 2877930). LAST_START and STOP_AT in the chain script became
    08:00 and 08:30, edited while no chain ran.
  - The extras (`..._extras_night.json`, also from `20261103-rod_halves.py`) are fca3d1cb's: probe-ref-x0, the rolls
    at 4 and 1.5 N, the obliques at azimuths 0 and 180, those at 90 and 270, and probe-ref-x1. A 4 N roll that half b
    ran is skipped as done.
- The workstation's checker on the whole chain in run order (from 618's rest; 3a and 1a with their pass 1; the extras
  less the 4 N rolls) flagged none of the 589 actions. It ran from 19:11:55 to 19:29:02 (`20261103-rod_night_halves_run_check.json`).

  | Plan | Actions | Joint 6, deg | Rod to the tip body in its own moves, mm |
  |---|---|---|---|
  | 2a | 27 | -56.4 to 42.2 | 0.50 |
  | 3a | 50 | -51.1 to 42.0 | 0.52 |
  | 1a | 54 | -61.0 to 42.2 | 0.52 |
  | 1b | 50 | -61.3 to 42.2 | 0.56 |
  | 3b | 46 | -51.0 to 41.8 | 0.56 |
  | 2b | 44 | -58.1 to 42.3 | 0.56 |
  | 2x | 108 | -59.0 to 0.0 | 0.48 |
  | 3x | 109 | -54.2 to 0.0 | 0.48 |
  | 1x | 101 | -61.1 to 0.0 | 0.49 |

  The checker needs 0.3 mm from the rod's end to a tip and 1.0 mm from the stem. The closest passes:
  - probe-edge-654-o20-a90-r90 in 1x: its joint travel from probe-face-790-o35-a270-r90 passes 0.36 mm from seat 1's
    pad. The workstation's own check of the extras (run_check_rod_extras_seat1) has the same 0.36 mm.
  - Seat 3's edge-639: the stem passes 1.02 to 1.18 mm from seat 3's pad in the eight obliques of 3x, and 1.24 mm in
    sweep-edge-639-r90 of 3b.
  - The rod's end passes 0.48 to 0.59 mm from the tip body in the deep presses near the pad's edge (edge-573, -615,
    -618 and flank-553, -563).
  - Least to the table's 2x4: 17.1 mm in 3a (the checker needs 8 mm).
- 19:43, the workstation (3ebf9b39, dry-check --tool-turn): it reran the whole chain with the rod turned as seat 2's
  fit found it, -5.40 deg about z_M and ±0.77 across.
  - One real flag: 1x's probe-edge-654-o20-a90-r90. Its joint travel from probe-face-790-o35-a270-r90 takes the rod
    0.31 mm into seat 1's pad at -0.77 across, and leaves 0.08 mm at +0.77 (0.36 mm unturned).
  - The exact rod solid (`ur5_rod_fitted_margins.py`, with the turn and the twists) keeps every action of the nine plans
    0.21 mm or more from the tip body. The closest is seat 3's edge-573-o35-a90 at full depth.
  - The k-d tree also flags the edge-654 obliques and face-731, -733 and -735 at 4 to 6.6 mm inside the tip body. The
    exact solid puts them at 0.51 mm or more (edge-654) and 2.99 mm or more (face): the sign error seen before.
  - Seat 3's edge-639 obliques bring the stem 0.95 to 0.97 mm from seat 3's pad at full depth, against a need of
    1.0 mm. The workstation keeps them.
- 19:4x: probe-edge-654-o20-a90-r90 in 1x travels by the hub. The plan was edited in place. `20261103-rod_halves.py`
  (HUB) now writes the same change; its output equals the edited plan less the notes, and the other eight plans are
  unchanged. The workstation's checker on 3x then 1x from 2x's end (`20261103-rod_night_1x_hub_run_check*.json`):
  - Unturned: 0 of 109 and 0 of 101 flagged.
  - The hub travel keeps the rod 1.23 mm from seat 1's pad unturned and at +0.77, and 1.25 mm at -0.77. No travel of
    3x or 1x comes under 0.3 mm with either turn.
  - With the turn, the flags left are the presses' own moves above: the edge-654 obliques, the edge-639 stems at
    0.95 to 0.97 mm, and deep presses at 0.31 to 0.40 mm from the tip body.
- 20:20:11, seat 2's half a (rod21, from 19:15:06) ended cleanly: 26 light sweeps and probe-ref-a1, all ok.
  - The sweeps took 103 to 165 s, median 150 s. 25 reached 8 N; sweep-edge-573-r90 stopped at 2 N (its max_depth
    0.75 mm near the edge).
  - 809 of 832 offsets and 102 of 104 slides were held.
  - The friction check skipped seven times: five at 0.5 N's offsets (r 0.06 mm, the push along the normal 0.16 N or
    less), one at 1 N's offsets (edge-668-r90) and one at 4 N's slides (edge-573-r90).
  - The HEX-E gave 390,162 samples, none timed out.
- 20:20:17, step 2: seat 3's half a (rod32) on `202017-rod32_seed`, seeded from s3v4 (c_M from the plan, (0, 0,
  107.5) mm), with its pass 1 and fit first.
- 20:36:49, seat 3's fit: pad touches 0.041 mm rms (20), chi2 52.3 on 20 dof.
  - c_M (-0.803, 0, 107.078) mm, sd (0.096, 0, 0.162).
  - The rod turned +10.75 deg about z_M (sd 1.46) and +1.05 across (sd 1.66).
  - The pad 1.027 mm and 1.234 deg from the seed (s3v4's T_AP); the rod's axis moved 0.907 mm across itself.
  - Pass 1's onsets ran from +0.38 to -1.35 mm against the expected; the last nine came 0.7 to 1.35 mm early.
  - Seat 2's fit of the same rod, not reseated in between, turned it -5.40 deg (sd 2.49). That is 16 deg apart, about
    5.6 combined sd. c_M agrees within 0.04 mm in x and 0.39 mm in z.
  - The workstation reads the turn as pass 1 trading the rod's turn against the pad's yaw: a square boss can't turn
    the tool 10 deg.
- 20:38, the workstation (exact rod solid at seat 3's turn): 3a keeps 0.47 mm (sweep-edge-615-r90, grid 16) and 3b
  0.25 mm, so 3a was not stopped. The lab's dry-check of 3a from the arm's joints and the exact check agreed (0.47 mm).
- 20:42, the turn envelope (the workstation): every remaining plan against the exact rod solid at turns of -11 to
  +11 deg about z_M, ±1.05 across. What goes under 0.2 mm from the tip body anywhere in that range is left out, so that
  seat 1's fit can't catch the night out. 3a, then running, stays as it was: its three actions under 0.2 mm (grids 2,
  3 and 27) are under only at -11 deg.
- A drop changes the travel into the action after it. The lab re-checked after every change (the workstation's
  dry-check, unturned and at --tool-turn ±11 with ±1.05), and that found:
  - After the five flank-563 drops, probe-flank-563-o35-a270-r90's joint travel took the rod 9.3 mm into seat 3's
    block and 3.6 mm into seat 2's pad, unturned. It now travels by the hub on both seats.
  - At -11 deg the rod reached 0.22 mm into seat 2's pad on the hub travel to edge-654-o20-a0. Each drop of a
    seat 2 edge-654 oblique left the next one a joint travel into the pad, and a hub approach 0.17 to 0.27 mm from it.
    Seat 2's edge-654 obliques all went.
  - 3b's probe-face-692-r90-t-15: its joint travel from sweep-face-735-r90 came 0.19 mm from seat 3's pad at -11,
    -1.05. It now travels by the hub.
- The exact check on the plans as they will run found two more. The envelope list had named one azimuth per sweep,
  and its lists keep only the four closest moves under 1 mm.
  - Seat 1's sweep-edge-569-r0 in 1b: its az 180 offsets came 0.09 mm and its deep pose 0.12 mm from the tip body at
    -11, +1.05. The whole sweep went.
  - Seat 3's sweep-edge-638-r90 in 3b: its az 0 offsets came 0.18 mm at -11, -1.05, and were trimmed with its az 0
    slides.
- What the night gives up (installed 20:53 and 21:42; `20261103-rod_halves.py` TRIM, DROP and HUB):

  | Plan | Left out |
  |---|---|
  | 1a | sweep-edge-615-r90's az 180 offsets, sweep-edge-618-r90's az 0 offsets |
  | 1b | sweep-edge-569-r0 |
  | 3b | sweep-edge-618-r90's az 0 offsets, sweep-edge-638-r90's az 0 offsets and slides |
  | 2b | sweep-edge-618-r90's az 0 offsets (sweep-edge-569-r0 ran in full before the switch) |
  | 2x | flank-563 o20-a90, -a180, -a270, o35-a90, -a180; the eight edge-654 obliques |
  | 3x | the same five flank-563 obliques; edge-573-o35-a90 |
  | 1x | edge-615-o35-a0 and -a180 |

- As installed, every action still to run keeps 0.21 mm or more from the tip body over the whole range (exact). The
  workstation's checker flags no travel, unturned or at ±11 with ±1.05 (`20261103-rod_night_envelope_checks.json`).
  Unturned, chained: 1a 0 of 54, 1b 0 of 49, 3b 0 of 46, 2b 0 of 44; 2x 0 of 97, 3x 0 of 103, 1x 0 of 99.
- Correction: the lab's message to the workstation stamped 20:49 was sent at about 20:44, with the time typed without
  running `date`.
- 21:47:50, seat 3's half a (rod32) ended cleanly: 51 actions ok (pass 1's 20 touches, 28 sweeps, two reference
  presses).
  - Sweeps took 126 to 163 s, median 150 s. 26 reached 8 N; sweep-edge-573-r90 stopped at 2 N, as on seat 2, and
    sweep-face-733-r90 at 4 N.
  - 879 of 896 offsets and 112 of 112 slides were held. Three friction skips, all at 0.5 N's offsets.
  - The HEX-E gave 524,234 samples, none timed out.
- 21:47:53, step 3: seat 1's half a (rod13) on `214753-rod13_seed`, seeded from s1sph2b. Pass 1's onsets came 0.05
  to 1.34 mm late, as seat 2's did (both seeded from a 2 mm sphere session).
- 22:07:10, seat 1's fit: pad touches 0.056 mm rms (20), chi2 76.5 on 20 dof.
  - c_M (-0.884, 0, 106.815) mm, sd (0.111, 0, 0.138).
  - The rod turned -2.90 deg about z_M (sd 3.63) and -0.25 across (sd 1.84). That is inside the envelope, so the
    plans stand.
  - The pad 0.992 mm and 1.318 deg from the seed; the rod's axis moved 1.119 mm across itself.
  - The three fits of the rod:

    | Seat | c_M x, mm | c_M z, mm | Turn about z_M, deg |
    |---|---|---|---|
    | 1 | -0.884 | 106.815 | -2.90 (sd 3.63) |
    | 2 | -0.838 | 106.687 | -5.40 (sd 2.49) |
    | 3 | -0.803 | 107.078 | +10.75 (sd 1.46) |

    Seats 1 and 2 agree on the turn (2.5 deg apart, combined sd 4.4). Seat 3's is the odd one; its pass 1 was the one
    whose onsets ran early. c_M agrees within 0.08 mm in x and 0.39 mm in z.
- 23:29:10, seat 1's half a (rod13) ended cleanly, and with it the first halves: every tip has half its sweep
  locations. It ran 56 actions, all ok.
  - The 32 sweeps took 125 to 161 s, median 152 s. 31 reached 8 N; sweep-edge-618-r90 stopped at 4 N.
  - 988 of the 1008 planned offsets (two sweeps trimmed by 8 each) and 126 of 128 slides were held. The friction
    check skipped seven times, all at 0.5 N's offsets.
  - The HEX-E gave 606,523 samples, none timed out.
- 23:29:10, step 4: seat 1's half b (rod14, 49 grid actions) on `214753-rod13_seed`.
- 23:35, the rod sessions' small files went to the workstation by git (c5a25696): registration, touches, plan,
  session, events snapshots and printouts in `20261103-202017-rod32/`, `20261103-214753-rod13/` and
  `20261103-172647-rod21/`.
- 23:48, the workstation's joint rod fit (`scripts/ur5_rod_joint_registration.py`, c2378efe): one rod over the three
  sessions.
  - Pass 1 alone does not fix the turn: a shared turn fits its 60 touches at +9.29 or -10.88 deg about as well.
  - With each sweep's 0.5 N offset poses from 2a, 3a and 1a as contacts (94 sweeps, both rod orientations), the rod
    is turned -0.22 deg about z_M (sd 1.16) and -1.90 across (sd 0.50), with c_M (-0.993, 0, 106.721).
  - The across tilt is outside the ±1.05 deg checked before.
- The margins with the joint fit, at the fit and at the corners of its 2 sd (the workstation): the second halves keep
  0.28 mm or more and stand. Eight edge-573 obliques of the extras come 0.20 mm or less from the tip body at their deep
  poses and were left out (23:55; `20261103-rod_halves.py` EDGE573):
  - seat 2's o35-a90, o20-a270, o35-a270, o20-a0 and o20-a180;
  - seat 3's o20-a270 and o35-a270;
  - seat 1's o35-a90.
- The workstation's checker on 2x, 3x and 1x chained from 2b's end, unturned (0 of 92, 101 and 98), at the fit and
  at its four corners: no travel flagged. The flags left are the presses' own moves: the k-d tree's deep values, and
  the edge-573 r90 obliques at 0.25 to 0.40 mm.
- 01:19:52 on 11-04, seat 1's half b (rod14) ended cleanly, so seat 1 has all its sweep locations, twists and 4 N
  rolls. It ran 50 actions, all ok.

  | Kind | Count | Median |
  |---|---|---|
  | Sweeps | 30 | 151 s |
  | Twists at 5 N | 14 | 121 s |
  | Rolls at 4 N | 3 | 103 s |
  | Reference presses | 3 | 38 s |

  - These were the night's first rod twists and rolls on the robot, and none aborted. The lab's estimates had
    assumed about 100 s a twist with its travel and 70 s a roll.
  - 946 of 960 offsets and 116 of 120 slides were held; six friction skips.
  - 27 sweeps reached 8 N. sweep-edge-651-r0 stopped at 2 N; sweep-edge-573-r90 and sweep-face-733-r90 at 4 N.
  - The HEX-E gave 663,183 samples, none timed out.
- 01:19:56, step 5: seat 3's half b (rod35, 46 grid actions) on `202017-rod32_seed`.
- Correction: the lab's message to the workstation stamped 19:1x (the plans not fitting) was sent between 18:59 and
  19:05. The workstation's relays reach corvus stamped about 10 min ahead of its clock.

## The lab clone and git-lfs

The rebase at 19:21 replaced the plan, `poses.jsonl`, `moves.jsonl` and the images of this folder with git-lfs pointers,
because `.lfsconfig` skips LFS content. The program then failed to read the plan (19:26, nothing moved). The files came
back from the local LFS store, identical to the originals in `logs/20261022-ur5_homing_by_hand/`. This clone now has
`lfs.fetchexclude` cleared in `.git/config`, as the workstation's has, and `tipcal_run.sh` refuses to start on a pointer.
