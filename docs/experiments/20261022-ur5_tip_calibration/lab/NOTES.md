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

### 14:10:11, cb3t pass 2a

## The lab clone and git-lfs

The rebase at 19:21 replaced the plan, `poses.jsonl`, `moves.jsonl` and the images of this folder with git-lfs pointers,
because `.lfsconfig` skips LFS content. The program then failed to read the plan (19:26, nothing moved). The files came
back from the local LFS store, identical to the originals in `logs/20261022-ur5_homing_by_hand/`. This clone now has
`lfs.fetchexclude` cleared in `.git/config`, as the workstation's has, and `tipcal_run.sh` refuses to start on a pointer.
