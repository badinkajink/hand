# Crosstalk sweep on the SR2 hand, night of 2026-11-04

This folder holds the runbook, code and plans of the first sweep of the SR2 hand's fingertips through finger motions.
The owner asked for it at 18:0x on 11-04: "see if you can read from everything on this ..., then see if you can run
the homing sequence. in an ideal world, we can measure crosstalk subtraction tonight while im gone". The plan follows
the Future work item "Fingertips beside each other on the hand" of `docs/overviews/printed_fingertips.html`. It records
every board through a wide range of finger motions, with the joint angles. A fit then gives each tip's field at its
neighbours' chips as a function of those angles.

## The setup

- The hand hangs on the UR5's flange. **Never turn on freedrive or teach mode while the SR2 is on the UR5** (owner,
  11-04): the arm drops under the hand's weight. The sweep does not move the arm. `lab/ur5_pose_log.py` logs its pose
  once a second, read-only.
- Fingertips: v3 ID 1 on the thumb (mux channel 3), v3 ID 3 on the index (channel 1), v3 ID 2 on the middle finger
  (channel 0). The owner confirmed channel 1 at 18:4x on 11-04, and each channel's rest field matches its tip's UR5
  sessions: 86, 114 and 57 uT rms against 342 uT or more for the other tips. The mux (0x71) sits on the SparkFun Thing
  Plus ESP32-C6 (40:4C:CA:4C:49:C4), plugged into corvus. The index and middle boards were unplugged until 18:53.
  Since 11:4x on 11-05 the index's cable runs through a second mux, for slack. Its board sits on channel 0 of a mux at
  0x70, which is cabled to channel 1 of the mux at 0x71. The firmware (7e5859883) streams it as channel 1, so the logs
  and `--board-channels 0,1,3` are as before.
- Steppers and servos: the M8P and the CB1 on the hand. From corvus the CB1 answers at 10.99.99.2 once corvus has a
  live second address: `nmcli device modify enp3s0 +ipv4.addresses 10.99.99.77/24`. That command changes no profile.
  Corvus's key `~/.ssh/id_wxie` is in irlab's authorized keys since 18:2x on 11-04.
- The CB1's clock is about 18 hours off, and it has no NTP. All times are taken on corvus.

## Bring-up

1. M8P: hold BOOT0, press and release RESET, release BOOT0, press and release RESET. `/dev/ttyACM0` (the "Manta Hand
   Controller") then appears on the CB1.
2. Station: `ssh -i ~/.ssh/id_wxie irlab@10.99.99.2 '~/run_control_station.sh start'`. It turns servo torque on at
   connect. After a run has disabled the motors, POST `/api/v1/servos/torque` with `{"state": "on"}`: the servos then
   hold where they hang. The station refuses joint commands until it is homed again.
3. Home: POST `/api/v1/home` with `{"confirmation": "HOME ALL AXES"}` and the station's token. It takes about 4 min.
   On 11-04 and 11-05 every axis homed by its timeout; StallGuard2 fired on none.
4. Fingertip boards: `lab/board_log.py PORT logs/20261104-sr2_crosstalk/boards`. Its log has the format of
   `src/sr2hand/driver/reskin5x/host/reskin5x_mux.py`. The records of one read are spaced evenly between that read
   and the one before. After a fingertip is plugged in, `lab/mux_boot_text.py PORT` resets the C6 and prints its
   channel scan. The logger reconnects by itself.
5. Sweep: `lab/run_plan.py PLAN OUTDIR --token-file F [--no-disable]`, under `/usr/bin/python3` on corvus. It drives
   the station over HTTP and writes `events.jsonl` (one line a step) and `telemetry.jsonl` (the station's samples).
   `touch OUTDIR/STOP` ends the run after its current step.

## Plans

`lab/make_plan.py` and `lab/make_random.py` write the plans in `plan/`. Both check finger-to-finger clearance in the CAD
hand (`assets/mjcf/experimental/20261030-cad_v2`, the collision spheres of every part), at each pose and along each
move. All plans keep the gantries at their home layout: thumb (-80, -55), index (80, 85) and middle (80, -85) mm in the
palm frame. At that layout only the thumb and the middle finger can come near each other.

- `20261104-plan_home.json`, 378 steps, about 39 min. It holds the zero pose for a baseline, then turns servo torque
  off and on. Each stepper axis moves 5 mm and back. Each finger then sweeps yaw, mcp and pip alone and through two
  grids, with the other two straight. Last come the thumb and the middle finger toward each other. Every pose clears
  10 mm, and every 12 poses the hand returns to the zero pose.
- `20261104-plan_random_s1.json` and `_s2.json`: 600 random poses of all three fingers each, at least 12 mm apart,
  about 51 min. They are the test set of the subtraction.
- `20261104-plan_test.json`: a short attended test of the runner.

The closest pose is the thumb at mcp 80 and pip 20 with the middle finger at mcp 65 and pip 40: 11.6 mm in the model.
The owner looked at it on the hand at 18:3x: "the gap is about 1cm yea".

## The night of 11-04

The sweep ran from 18:57 to 20:34 and then stopped, with the hand at zero and the motors disabled. The files are in
`data/20261104-night/`: the board logs in `boards/`, and one folder a run with `events.jsonl` (a line per step, on
corvus's clock) and `telemetry.jsonl` (the station's samples, 1.7 Hz until 19:09, then 20 Hz). It also holds
`ur5_pose.jsonl` (the arm, still and powered off from 18:57:27) and `chain.out`.

| Run | Time | Steps | Families | Boards |
|---|---|---|---|---|
| `run0_test` | 18:55:48 to 18:56:42 | 10 | the attended test | all three |
| `run1_home` | 18:57:47 to 19:31:31 | 126 | baseline, torque, every stepper step, all the thumb's families, index yaw and mcp | all three |
| `run1b_home_rest` | 19:40:27 to 19:44:06 | 38 | index pip, 24 of its mcp-pip grid | all three until 19:43:52 |
| `run1c_home_rest2` | 19:52:36 to 20:08:18 | 209 | index yaw-mcp, all the middle's families, every thumb-middle family | thumb and middle |
| `run2_random_s1` | 20:08:19 to 20:34:09 | 355 | 327 random poses (one blocked) | thumb and middle |

The owner paused the hand from 19:05:12 to 19:21:36 to colour the finger links. During that pause the board logger
lost the port for 2 s at 19:14:05.

**The index's board stopped answering at 19:43:52,** when the index stalled at mcp 50 + pip 90. Its curl most likely
pulled its fingertip cable. Until the reset at 20:34, the C6 still read channel 1, which failed every transfer. Those
records have zeros for the index, and the sweep slowed to about 10 records a second. After a reset the C6 finds only
channels 0 and 3. In `boards/20261104-191407-mux.bin` after 19:43:52, half the records are misaligned and hold garbage.
Keep the records whose index board is all zeros and whose middle and thumb boards stay within 30 % of their usual
|B| on every chip: 29,277 records at 9.7 a second. That stretch covers `run1c_home_rest2` and `run2_random_s1`.

The runs stopped three times:
- 19:31:23, the index's mcp drooped under gravity;
- 19:43:59, the index's curl was blocked;
- 20:34:01, the lab's STOP after it found the bad stream.

The motion was not restarted because the index's cable may hang loose across the fingers. Since 20:35:46 the thumb's
and middle's boards log the hand at rest overnight (`logs/20261104-sr2_crosstalk/boards/20261104-203546-mux.bin` on
corvus, channels 0 and 3).

## Analysis

The coordinator analysed the night on the workstation: `scripts/fingertip/sr2_crosstalk_night.py` writes
`data/20261104-night_crosstalk_analysis.json` (9a0e05a53, feed line 55).
- **Neighbours and actuators.** At the home layout, a finger moving alone changed the other boards by 1.2 to 5.4 uT.
  That is near the 3.4 uT between two rest readings. The steppers and servo torque changed them by 1.2 to 5.9 uT.
- **The tip's own pose.** A tip's own pose changed its board by up to 166 uT, about 5.8 N at 28 uT/N. A GP on the
  finger's own three angles, trained with random poses, left 4.4 and 4.5 uT at the median and 7.5 and 9.0 uT at the
  95th percentile on the 327 random poses. Trained on all 318 structured steps, it left 20 and 21 uT. Trained on the
  finger's own sweeps alone (79 and 77 steps), it left 12 to 15 uT (the coordinator's correction at 22:1x; feed line 55
  conflated the two). A uniform field turned with the tip left 12 to 14 uT. So the own-pose term may change with the
  mount's place on the palm.
- **Close pairs.** With the thumb and the middle finger under 20 mm apart in the CAD model, the remainder rose to 6 to
  15 uT. Only 14 poses came that close.
- **Clearance as run.** Replayed with each servo at its own rate (0.6 to 1.4 times), the moves that ran came no closer
  than 10.7 mm in the CAD model, thumb to middle.

For the next sweep, more random poses train the own-pose term. Closer thumb-to-middle poses would test the neighbour
term.

## Next sweep

The owner, 22:10 on 11-04, relayed by the workstation: "ahh yea the index cable was a little tight, i shouldve
prepared better for that. i hoipe the cables not damaged. we could try more thumb-middle poses yea, we can use the
steppers to move them to more realistic poses too". That is a direction, not a go: the hand stays stopped until the
owner re-routes the index's cable with slack. The owner lengthened the cable through a second mux at 11:4x on 11-05
and gave the go (section "11-05").

1. Check the cable. Run `firmware/bench_mux_scan` on the C6. Channel 1 should answer with five chips at every clock
   from 400 down to 20 kHz. Nothing at all, or answers only at the low clocks, means a damaged conductor. Then compare
   channel 1's rest field with v3 ID 3's UR5 sessions, as at 18:53.
2. Plan at several layouts. Move the mounts to grasp-like places with the steppers, and take random poses at each
   layout, with closer thumb-middle poses among them. The GP takes the mount coordinates as inputs, so the own-pose term
   learned at one layout is tested at the others. Check every pose and move, gantry moves included, at the
   rate-spread margin.
3. Have the runner check the boards' stream (record rate, zeroed chips) as well as the servos.

Ready since 22:4x on 11-04, and not run on the hand:
- **`lab/make_layouts.py` and `plan/20261105-plan_layouts_s1.json`.** The plan takes 120 random poses at each of five
  grasp layouts from the transfer study: g12, sv1_w6689, rv04_mid, sv1_w0099 and sv1_u0308. rv05_manual's straight
  fingers clear only 12.3 mm, so it is left out. Half the poses put the thumb and the middle finger within 30 mm. Every
  pose clears 12 mm in the CAD model. So does every move, and every pose's way back to zero, along the straight
  joint-space line. Each gantry move, one axis at a time with straight fingers, clears 25.7 mm or more. There are 672
  steps, about 75 min with the ramps. The index's limits still apply.
- **`lab/run_plan.py`.**
  - It walks every move, back-out and safety return along that line in waypoints 3 deg apart (`--ramp-deg`). Servos
    at different rates then stay within about 2.6 mm of the checked path.
  - `--boards DIR --board-channels 0,1,3` stops a run on a fault in the boards' stream.
  - The safety path waits for the station without the checks, so a standing fault cannot keep the hand from zero.
  - A gantry move may take 300 s.

  On 11-04 the station's fake-hardware mode (`manta_hand.web --fake`) ran the plan's first and last layout steps:
  22 steps through layout g12 and back to home. A board fault, raised by asking for channel 1, stopped a run in
  1.6 s. The hand still reached zero, and the motors were disabled.
- **The workstation's replay** (23:0x on 11-04). It walked every move and safety return in 3 deg waypoints and put
  the closest pair's six joints at the corners of each waypoint's box. No move came under 8.5 mm, which is a bound
  with all six at their extremes at once. Per layout, as line minimum, box minimum, and the safety returns' box
  minimum:

  | Layout | Line min | Box min | Safety-return box min |
  |---|---|---|---|
  | g12 | 12.1 mm | 10.6 mm | 11.3 mm |
  | sv1_w6689 | 12.0 mm | 10.3 mm | 11.6 mm |
  | rv04_mid | 12.1 mm | 8.5 mm (thumb-index, step 299) | 10.8 mm |
  | sv1_w0099 | 11.8 mm | 9.8 mm | 11.3 mm |
  | sv1_u0308 | 12.0 mm | 8.6 mm (thumb-index, step 610) | 11.5 mm |

  On three layouts the thumb comes closest to the index, whose cable is not in the model. The owner's eye check of
  1 cm against 11.6 mm was made at the home layout. The cad_v3 hand-off provisionally finds the CAD's travel boxes
  about 6 mm off the driver's nominal mounts.
- **An attended pass comes first.** `plan/20261105-plan_layouts_attended.json` holds 35 steps, about 13 min. It moves
  through each layout and holds that layout's box-closest and line-closest poses (steps 21, 39, 138, 189, 299, 428,
  496, 595 and 610) for 20 s each. Each pose is reached from straight along its checked way to zero. The owner looks
  at the gaps, and at where the index's cable runs, before the unattended plan.
- **The command,** once the owner has re-routed and checked the index's cable and the attended pass looks right:

      /usr/bin/python3 docs/experiments/20261104-sr2_hand_crosstalk/lab/run_plan.py \
          docs/experiments/20261104-sr2_hand_crosstalk/plan/20261105-plan_layouts_s1.json logs/<date>-sr2_crosstalk/run_layouts \
          --token-file ~/.sr2_station_token --boards logs/<date>-sr2_crosstalk/boards --board-channels 0,1,3

## 11-05

**The index's cable.** At 11:3x the owner lengthened the index's cable with a second mux: "ive plugged the index cable
bck into a separate mux (on port 0 on that mux), with a cable from that mux back to the original mux port. this is to
extend the cable without losing signal, i hope". The new mux answers at 0x70. A mux of this kind switches the bus
through and does not buffer it, so the scan, not the mux, shows that the longer run works.
- `firmware/bench_mux_scan` now looks behind a second mux. In six rounds from 11:40 it found the index's five chips
  (0C, 10 to 13) behind 0x70's channel 0 at 400, 100, 50 and 20 kHz. The middle's and thumb's chips answered on
  channels 0 and 3, as before.
- `firmware/reskin5x_mux_stream` (7e5859883) then streamed all three boards at 161 records/s, with no zeroed chip.
- Channel 1's rest field is 126 uT rms from v3 ID 3's UR5 session of 10-29, against 357 and 474 uT from IDs 2 and 1.
  At 18:53 on 11-04 it was 114 uT. Since then the index's rest field has moved 23 uT rms, and the untouched middle and
  thumb 29 and 34 uT.

The overnight log of the middle and thumb ends at 11:39:56 (`logs/20261104-sr2_crosstalk/boards/20261104-203546-mux.bin`,
2.2 GB at 243 records/s). Since 11:41:16 all three boards log into `logs/20261105-sr2_crosstalk/boards/`.

**The go.** The owner at 11:41, relayed by the workstation: "the index is now plugged in, and i added extra cable length
via another mux in the chain. you have the go ahead to run further calibration now. i dont need to run the viewer at
this moment." Asked in the lab's window whether they would watch the attended pass, the owner answered at 12:0x: "send
it on the attended plan, unattended, and the other plan. i trust you". At 12:04, relayed: "ok i have to leave and cant
supervise, but the ur5 computer has the go ahead to execute the readme refresh, script rebuilt, driver rebuild, shared
math rebuilt, and then the attended runs, unattended. be careful". The CB1's manta driver stays as it is (QUEUE R35),
so the hand ran on the driver of 11-04's 838 steps.

**Without the owner's eye.** The camera has been moved since 11-04 and sees the hand from one side. The time-lapse runs
at 1280 x 720 every 2 s (`/dev/shm/20261105-sr2_timelapse_720`). The mux boards sit strapped to the thumb's housing,
where the main mux sat on 11-04, with cables looping between the fingers. In place of the owner's look, the lab looked
at the frame of each hold before it started the unattended plan. The runner kept its guards and one more: `--max-blocked 1`, so the first pose a joint
cannot reach ends the run and the chain.

**The runs.** Servo torque came on at 12:15, and the hand homed from 12:16:09 to 12:20:04, every axis by its timeout.

| Run | Time | Steps | Result |
|---|---|---|---|
| `run1_attended` | 12:20:15 to 12:34:28 | 35 of 35 | plan done; no blocked pose, no board fault |
| `run2_layouts` | from 12:35:08 | 672 | running |

In `run1_attended` the nine holds settled within 4.2 deg of their commands. The middle's yaw lagged most: 3.9 and
4.2 deg at -37 and -22 deg. The boards gave 160.5 records/s, never under 152 in a second, with no zeroed chip. The
servos stayed at 24 C or under. The frames of the nine holds (`data/20261105-run1_attended_holds.jpg`) show the
cables looping around the thumb and between the fingers. None is caught where the camera can see. The logs are in
`logs/20261105-sr2_crosstalk/` on corvus: `run1_attended/` and `run2_layouts/`, each with `events.jsonl`, `telemetry.jsonl` and `progress.json`, and
the boards in `boards/`.

## Known limits

- The CB1's FTDI latency timer was 16 ms until 19:09 on 11-04, and the station's telemetry, with load and temperature,
  ran at 1.7 Hz. A udev rule has set it to 1 ms since then (`docs/hardware_control_station.md`). The telemetry now runs
  at the station's `--telemetry-hz` of 20 Hz.
- The middle finger's mcp droops about 3.5 deg short of its command when the finger is near horizontal. The runner
  treats a joint as settled once it stops moving, and it stops at an error of 6 deg or more.
- With the hand on the arm as it hangs, the index's mcp servo reaches its protective torque near 55 deg, with its load
  at -195 to -210. Its droop grew from 1.5 deg at 39 deg to 3.4 deg at 56 deg. Commanded to 64 deg at step 126 of
  `run1_home`, it stopped at 55.4 deg, and the runner stopped the chain at 19:31:23. That run had done the baseline,
  torque, every stepper step, the thumb's families and the index's yaw and mcp. Since then the plans cap the index's mcp
  at 50 deg and the middle's at 62 deg (`plan_home_rest`, the `_v2` plans, `lab/night_chain2.sh`). The thumb reached
  88 deg with 1.6 deg of droop.
- At step 38 of `run1b_home_rest` (19:43:59), the index was sent from zero to mcp 50 and pip 90. It stalled at mcp
  45 and pip 80.3, with both servos pushing: pip loads of -525 to -600 for 4 s, then -200 under the servo's protection.
  This was not droop: pip 90 had passed at mcp 0, 25 and 37.5, and mcp 50 + pip 67.5 had passed. The CAD model shows no
  contact there (27.6 mm to its palm box, 12.2 mm to the finger's own base). So something outside the model stopped the
  curl, such as the base's structure or a fingertip cable to the mux on the thumb's housing. Since then the plans keep
  the index's and middle's mcp + pip at 120 deg at most (`plan_home_rest2`, the `_v3` plans, `lab/night_chain3.sh`).
  `run_plan.py` backs out of a pose a joint cannot reach to the last pose reached, logs a `blocked` event, and stops
  the run at the third.
- The plans check each move as a straight line in joint space, as if every servo moved at the same rate. The
  workstation replayed the v2 plans in the cad_v2 model with each servo at 0.6 to 1.4 times the others' rate (11-04,
  evening). Its closest approach was thumb to middle at 8.8 mm on step 440 of `plan_random_s1_v2`, a plan that never
  ran. On the home plans it was 11.9 mm. Future plans should take their move margin from such a replay.
- On 11-04 the runner checked only the servos and the station. It did not see the index's board fail, and the stream
  stayed half garbled for 50 min. Since 22:2x on 11-04 it also watches the boards' stream (`--boards`,
  `--board-channels`).
