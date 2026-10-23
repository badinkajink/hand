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

## The lab clone and git-lfs

The rebase at 19:21 replaced the plan, `poses.jsonl`, `moves.jsonl` and the images of this folder with git-lfs pointers,
because `.lfsconfig` skips LFS content. The program then failed to read the plan (19:26, nothing moved). The files came
back from the local LFS store, identical to the originals in `logs/20261022-ur5_homing_by_hand/`. This clone now has
`lfs.fetchexclude` cleared in `.git/config`, as the workstation's has, and `tipcal_run.sh` refuses to start on a pointer.
