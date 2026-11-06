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
| `run2_layouts` | 12:35:08 to 12:43:50 | 31 of 672 | stopped at step 32, the middle's yaw blocked |
| `run3_from32` | 13:32:40 to 13:36:24 | 8 of 643 | the lab's STOP: the middle's X stage had stayed at home |
| `run4_from32` | 13:49:27 to 14:10:03 | 105 of 643 | g12 done; the lab's STOP, so the rest could run gated (below) |
| `chain5` | 14:11:39 to 17:31:24 | 538 of 538 | chain done: four layouts and home, one stop (rv04_mid step 44) |
| `hom1` | 18:01:08 to 20:58:11 | 607 of 607 | chain done: five HOM layouts and home; two stops from the lab's disk writes |

In `run1_attended` the nine holds settled within 4.2 deg of their commands. The middle's yaw lagged most: 3.9 and
4.2 deg at -37 and -22 deg. The boards gave 160.5 records/s, never under 152 in a second, with no zeroed chip. The
servos stayed at 24 C or under. The frames of the nine holds (`data/20261105-run1_attended_holds.jpg`) show the
cables looping around the thumb and between the fingers. None is caught where the camera can see. The logs are in
`logs/20261105-sr2_crosstalk/` on corvus: `run1_attended/` and `run2_layouts/`, each with `events.jsonl`,
`telemetry.jsonl` and `progress.json`, and the boards in `boards/`.

**The stop at step 32.** At 12:43:37 `run2_layouts` stopped on its first blocked pose. Step 32 (random_g12) sent the
middle finger to yaw -34, mcp 43.5 and pip -6.4 deg. Its yaw servo stayed at 27.8 deg, pushing at a load of -285 to
-345, and fell 6.2 deg short. The runner backed out to step 31's pose, took the hand to zero and disabled the motors at
12:43:48.
- The middle's yaw falls short only toward negative yaw. Below -25 deg it fell short by 4.1 deg on average today (at
  most 5.6 deg before step 32), and by 2.6 to 3.0 deg on 11-04 (at most 5.3 deg in 79 poses).
- Its load builds with the angle, and today it builds sooner. Between -20 and -10 deg the median load was -210 at g12
  today, against +75 at the home layout on 11-04. Between -40 and -30 deg it was -225 against -193.
- Not a cable. At 13:0x the owner swept the middle's yaw through its range by hand and found nothing restricting it.
  The frames of the middle's carriage at each layout are in `data/20261105-middle_carriage_by_layout.jpg`.
- Gravity loads every yaw joint. With the fingers hanging, the yaw axis is horizontal, and the cad_v2 model gives a
  pull back toward zero of 0.016 N m at 40 deg, like a pendulum's. (The lab first wrote that gravity does not load the
  yaw; that was wrong.)
- All three yaw servos fall short the same way in the servo's own frame (the thumb's yaw sign is opposite). On 11-04,
  past 25 deg on the servo-positive side, the index fell short by 2.7 deg, the middle by 2.7 and the thumb by 2.4 on
  average. On the other side they fell short by 0.3 to 1.1 deg. The middle crossed 6 deg at -34 deg.

**Not resumed.** The lab had set the rule that a blocked pose ends the chain until the owner has looked. The owner
looked at 13:0x. The hand stays stopped at the home layout: motors disabled at 12:57, servo torque off, unhomed. The board logger keeps logging
all three boards at rest. `lab/resume_plan.py` wrote
`plan/20261105-plan_layouts_s1_resume32.json`, which is ready and not run.
- It homes, returns to g12, takes a baseline, and runs s1 from step 32. It leaves out the 166 random poses whose middle
  yaw is below -25 deg. That leaves 407 random poses, 492 steps, about 46 min.
- It re-checked the 110 moves whose neighbours changed. Fifteen would cross on the straight line (down to -1.6 mm), and
  they go through zero instead. Every step clears: poses 12.0 mm, moves 12.0 mm, ways to zero 12.1 mm, gantry moves
  25.7 mm.
- The shortfall is the yaw servos holding against gravity, not a cable. So s1 could also resume from step 32 as it
  stands (`resume_plan.py PLAN OUT 32` without the cut), with an error limit of its own for the yaw joints. Either
  plan runs after torque on and homing:

      /usr/bin/python3 docs/experiments/20261104-sr2_hand_crosstalk/lab/run_plan.py \
          docs/experiments/20261104-sr2_hand_crosstalk/plan/20261105-plan_layouts_s1_resume32.json logs/<date>-sr2_crosstalk/run3_resume32 \
          --token-file ~/.sr2_station_token --boards logs/<date>-sr2_crosstalk/boards --board-channels 0,1,3

**The afternoon.** The owner came back at 13:0x and found the middle's yaw free through its range, so the
shortfall is gravity against the yaw servos (above). `run_plan.py --max-err-yaw` (24468c353) gives the yaw joints a
limit of their own, 8 deg, while mcp and pip keep 6. `plan/20261105-plan_layouts_s1_from32.json` (00ce73764) runs s1
from step 32 as it stands.
- `run3_from32` homed from 13:28 to 13:32. At its g12 move the station counted joint 4, the middle's X stage (M8P
  socket M5), through 45 mm from 13:33:59 to 13:34:17 with no error. The owner saw the stage still at home, and the lab
  stopped the run at 13:35:57. The steppers are open loop, so nothing records where a stage is.
- With the station stopped, the firmware moved joint 4 10 mm, then 30 mm at the full 12,000 steps/s, and back. The
  owner saw it move fine. GSTAT reads 0x05 on all six TMC5160s: reset and charge-pump undervoltage from power-up,
  never read since, and drv_err 0. So no driver shut down. The likeliest cause, not confirmed: homing by timeout grinds
  each stage into its hardstop for up to 27 s, and the next move did not break joint 4 loose. (The firmware's UART
  register reads fail on every axis because these drivers are TMC5160s on SPI; `RREG5160` reads them.)
- The middle's mount in `run3_from32` was 45 mm off its log. Whether joint 4 also skipped in `run1_attended` or
  `run2_layouts` is unknown, so their middle mounts are uncertain.
- The owner asked for no more homing. `lab/rearm_home.py` re-armed the home instead: joint 4's count zeroed to match
  its stage, then a 1 ms firmware HOME on every axis, which sets homing_result 3 and moves nothing. The station then
  adopted the home (`/api/v1/home/adopt`) with the mounts as the stages stood. `run_plan.py --keep-home` (5c3aeffad)
  leaves the motors on at any stop, so an attended run's next start needs no homing either.
- `run4_from32` started at 13:49:27, and joint 4 moved at its g12 step (13:49:58).

**The gated chain.** The coordinator found at 13:5x that a stepper skipping its layout move brings fingers into each
other in cad_v2 at four of the five remaining layouts, and nothing on the hand can see a skip. So the lab stopped
`run4_from32` after g12 (14:09:56), and the rest of s1 ran as `lab/gated_chain.sh` (`chain5`). After each layout move
the chain waited until the owner had looked at the stages.
- The owner confirmed every move: sv1_w6689 at 14:36, rv04_mid at 15:41, sv1_w0099 at 16:31 ("yea the middle Y-axis
  moved, lets keep going") and sv1_u0308 at 17:06 ("yea it moved its looking good"). The move home at 17:31 had no
  gate. The owner checked it after the chain ended.
- One stop. At 15:50 the middle's yaw stayed 8.1 deg short at step 44 of rv04_mid. The owner watched it stick at
  29.6 deg for 5 s and free itself with nothing touching. The chain went on from step 44 at 15:54, with the yaw limit
  at 10 deg (`MAX_ERR_YAW`). No other pose was blocked. The largest error elsewhere was 6.1 deg, a yaw.
- The measured poses kept clear of each other. At the station's mounts in cad_v2, the servos' mean over each dwell
  came no closer than 11.3 mm (sv1_w6689, step 107, thumb to middle). The other layouts' closest were 12.1, 12.3 and
  12.5 mm. Over every 20 Hz sample of the first two layouts, moves included, the closest was 10.6 mm.
  `lab/measured_clearance.py` computes this. Each chunk's `events.jsonl` and the 523 dwells are in `data/20261105-chain5/`.
- The boards logged at 160.5 records/s throughout.

**The measured-pose gate.** The coordinator's tolerance check (bd20939cc, `data/20261105-plan_*_tolerance_check.txt`)
put the limits' worst case in numbers. At the 10 deg yaw limit, two neighbours both short toward each other would touch
on 11 to 35 steps of each remaining layout and 17 to 22 of each HOM layout, up to 3.8 mm deep. `run_plan.py
--min-clearance MM` closes that gap.
- After a pose passes its joint limits, the runner computes handgeo's clearance of the measured joints at the
  station's `mount_positions`. Under MM it backs out, as from a pose not reached.
- Every step's event records `clear_mm` and `clear_pair`.
- `lab/clearance_helper.py` runs under uv, since the runner's python has no mujoco. A helper that cannot start, or a
  query unanswered for 5 s, is a safety stop.
- `gated_chain.sh` passes `MIN_CLEARANCE` on.
- Dry check on a stand-in station on localhost:
  - 12 roll poses: the gate's clearance matched each step's plan clearance within 0.05 mm.
  - A pose within every joint limit read back 1.8 mm into thumb and middle (the thumb's yaw 9.5 deg off, the others
    5.5): it was backed out, the run stopped and went to zero.
  - A helper killed mid-run stopped the run at the next pose. A helper that could not start stopped it before the
    first.

**The HOM layouts.** `hom1` runs `plan/20261105-plan_layouts_hom_s1.json` in the chunks of
`plan/20261105-hom_s1_chunks`, with `MAX_ERR_YAW=10 MIN_CLEARANCE=5`. The owner chose it at 17:5x, "Yes, with the gate".
It started at 18:01:08 from home, after corvus pulled the merges of QUEUE R35's parts B and C with the hand idle. The
owner confirmed the roll move at 18:04: "yes, only the thumb x has moved, looks centered. the Y-axis on all of them
have moved inward about the same".
- Every pose ran, and the gate backed out none. The closest measured poses per layout: roll 14.6 mm, press 13.0,
  generalist 10.8, twist 11.0, cover 11.9. The largest errors were 7.1 deg on a yaw and 5.6 deg on an mcp or pip.
- Two stops, both the lab's doing. At 18:07 a `git worktree add` wrote GBs to corvus's disk, with memory tight
  (`/dev/shm` held 9 GB). At 18:28 the first try of the data push did the same: its rate cap counted from the script's
  start, so 240 MB went out at once. Both times the board logger fell behind, and the runner's board watch stopped roll
  (18:09:10 at step 25, 18:29:02 at step 70). Roll resumed each time from the stopped step. The board log thinned to
  120 to 150 records a second for about 3 min and 20 s, with no gap over 0.5 s. Since then nothing writes more than a
  few MB a second on corvus while the hand runs.
- The owner confirmed press at 19:17. Then, at 19:1x: "you can go ahead on this with my validation, and run the
  remaining configurations with my go-ahead, but i wont be able to verify personally. maybe you can do some
  intelligent geometry processing with the d435 to roughly estimate travel of each finger".
- `lab/d435_rgbd.py` records the D435's colour and depth every 2 s. `lab/layout_travel.py` registers each finger's depth
  patch from before a move on the frame after it, and converts the shift into palm x and y through the camera's 8 deg
  turn and its perspective. Generalist read -19.9 and +19.6 mm (plan -20, +20). Twist read the carriages -16.0, +16.3
  and +17.6 (-15, +15, +15) and the X stages -21.4 and +17.6 (-20, +20). Cover read the carriages +26.1, -26.5 and
  -26.7 (+25, -25, -25). On frame pairs where nothing moved, every axis read 0 to 4 mm. The lab released those three
  gates on these readings. The lab's lights went out at 20:12; depth, lit by the camera's own projector, read on.
- The move home (20:55 to 20:57) read right for the thumb and the index. The middle stood hidden behind the thumb and is
  unconfirmed: before the next run, the owner looks at the middle's two stages.
- The data from 12:20 to 21:00 is in `data/20261105-day/` (branch `sr2-crosstalk-data-20261105`, 7f04aa8d1 and
  097cdc0e3; merged at ddcb02497 for the first part). It holds the board log cut by t, every run's events, telemetry
  and progress, and manifests that list the uncertain stretches.

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
