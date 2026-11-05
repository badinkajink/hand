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
- Steppers and servos: the M8P and the CB1 on the hand. From corvus the CB1 answers at 10.99.99.2 once corvus has a
  live second address: `nmcli device modify enp3s0 +ipv4.addresses 10.99.99.77/24`. That command changes no profile.
  Corvus's key `~/.ssh/id_wxie` is in irlab's authorized keys since 18:2x on 11-04.
- The CB1's clock is about 18 hours off, and it has no NTP. All times are taken on corvus.

## Bring-up

1. M8P: hold BOOT0, press and release RESET, release BOOT0, press and release RESET. `/dev/ttyACM0` (the "Manta Hand
   Controller") then appears on the CB1.
2. Station: `ssh -i ~/.ssh/id_wxie irlab@10.99.99.2 '~/run_control_station.sh start'`. It turns servo torque on at
   connect.
3. Home: POST `/api/v1/home` with `{"confirmation": "HOME ALL AXES"}` and the station's token. It takes about 4 min.
   On 11-04 every axis homed by its timeout; StallGuard2 fired on none.
4. Fingertip boards: `lab/board_log.py PORT logs/20261104-sr2_crosstalk/boards`. Its log has the format of
   `src/morphohand/driver/reskin5x/host/reskin5x_mux.py`. The records of one read are spaced evenly between that read
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

## Known limits

- The CB1's FTDI latency timer is 16 ms, and no udev rule sets it. The station's telemetry, with load and temperature,
  then runs at 1.7 Hz instead of tens of hertz. `docs/hardware_control_station.md` gives the fix, which needs sudo on
  the CB1.
- The middle finger's mcp droops about 3.5 deg short of its command when the finger is near horizontal. The runner
  treats a joint as settled once it stops moving, and it stops at an error of 6 deg or more.
