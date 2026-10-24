# The calibration cell on the UR5's computer, as measured on 2026-10-22

This folder records the cell as it stood on the evening of 2026-10-22, before the unattended runs. It covers the lab
computer, the measured geometry of the tool and the two indenters, the plan built for them, the changes to the program,
and the steps of the runs. A session started after a compaction, or the owner's work computer, can carry on from this
file alone. The runbook of the session is `../README.md`. Its protocol page is
`../20261022-ur5_tip_calibration_protocol.html`.

## State at the hand-off (2026-10-22, about 19:20)

- The robot (UR5 CB3) stands at home, powered, brakes released, in safety mode normal, with no program running. Teach
  mode is off.
- The speed slider is at 30 %, set over RTDE (`rtde_io.RTDEIOInterface.setSpeedSlider(0.3)`). The value stays when the
  connection closes.
- The HEX-E answers at 192.168.0.6, and the fingertip board streams on `/dev/ttyACM0`.
- The browser viewer (`wrench_viewer.py`, http://localhost:8790) still holds the board's port. **Stop it before the
  program runs**: two readers on one port each get torn records.
- The owner checked by eye that the pad stood 2 mm above each indenter, centred, at the poses the measured geometry
  gives (section "How it was measured").
- The owner's decisions for the night:
  - the rod fixture stays on the 2x4 and is an obstacle in the plan;
  - the runs reuse the hand seating on the sphere (`--home-from`);
  - the first runs go at 30 % speed, raised to 100 % over RTDE once their printouts match Table 2 of `../README.md`;
  - this session commits and pushes, and the work computer's session pulls and analyses (owner, 2026-10-22: "set you up to
    iteratively commit & push ... so that i can set up my work computer claude code to monitor the repo remote and run
    analysis there as needed"; "try to do as much as reasonable here, but ... a lot of our simulation is very compute
    intensive and might need to be offloaded").

## Hardware and addresses

| Item | Value |
|---|---|
| Lab computer | `corvus`, user `will`, 192.168.0.5 by DHCP from the Netgear router 192.168.0.2; clone `/home/will/workspace/hand`; no passwordless sudo |
| Robot | UR5 CB3 at 192.168.0.4, PolyScope 3.15.8; RTDE at 125 Hz; the payload on the pendant was 0.41 kg (preflight sets the plan's 0.384 kg) |
| HEX-E DAQ | OnRobot Compute Box at **192.168.0.6**, DHCP on (`GET http://192.168.0.6/api/cfg/network/ip`), MAC b8:27:eb:25:4a:6e. The program's default `--ft 192.168.0.5` is this computer, so every run passes `--ft 192.168.0.6` |
| HEX-E | serial HEXEA478, firmware 208. After a power cycle of the box re-find it: ping-sweep 192.168.0.0/24, then send the 0x1234/0x0002 request to UDP 49152 |
| ATI Net box | NETBA for the Mini45, MAC 00:16:BD:00:2D:E4, firmware F/T BNI v2.2.59, **static 192.168.123.50/24** (set by an earlier user, not ATI's default). PoE from the Netgear GS305PP switch, whose uplink goes to the router's port 4. corvus reaches it through a live second address; see *Network on corvus* below |
| Board | QT Py ESP32-S3, `/dev/serial/by-id/usb-Adafruit_QT_Py_ESP32-S3__4MB_Flash_2MB_PSRAM__b4:3a:45:b0:9c:d0-if00` (`/dev/ttyACM0`); 5 MLX90393 at 0xC, 0x11, 0x12, 0x13, 0x10; 369 records/s (Table 2 of the runbook says 140: this firmware streams faster; preflight needs 50) |
| Overhead camera | RealSense D435, serial 836612071918, on the USB hub; views the indenters when the arm is clear. Its rough hand-eye calibration (z the best part, owner) is named in `~/workspace/rh56_controller/configs/tabletop_vision_lab.yaml` |
| Python | `/usr/bin/python3` (3.10). Its numpy 1.26.4, scipy 1.12.0, pyserial, ur_rtde 1.5.7, aiohttp, pyrealsense2 and trimesh live in `~/.local/lib/python3.10`: never delete that folder wholesale |

## Network on corvus, and reaching the ATI Net box

corvus has one wired port, `enp3s0`, which takes 192.168.0.5 from the lab router by DHCP (NetworkManager profile
"ASUS Router"). That port carries the internet, and with it every Claude session on this computer.

- **Never set that profile's IPv4 method to Manual.** ATI's guide and similar instructions say to, but they assume a
  computer wired only to the box. Manual drops the DHCP address, the default route and the DNS, and the internet with
  them.
- **To reach a device on another subnet, add a live second address** (no sudo: NetworkManager allows this user
  network control):

      nmcli device modify enp3s0 +ipv4.addresses 192.168.123.100/24     # undo: -ipv4.addresses 192.168.123.100/24
      ip route get 1.1.1.1                                              # must still say src 192.168.0.5

  The profile does not keep it, so a reboot or a reconnect drops it. On 2026-10-23 the internet stayed up throughout.
- **Find an ATI box at an unknown address with ATI's discovery**, not with sweeps:

      /usr/bin/python3 scripts/ati_netft_discover.py --iface-ip 192.168.0.5

  It broadcasts to UDP 51000, and each box answers on the multicast group 224.0.5.128:28250 with its address, mask and
  MAC. On 2026-10-23, sweeps of 192.168.0.x, 192.168.1.x, 169.254.x.x and IPv6 all missed the box at 192.168.123.50,
  and the workstation found it this way.
- **Then check the box:** `ping 192.168.123.50` and `curl http://192.168.123.50/`. Its web pages hold the settings;
  `comm.htm` has DHCP and RDT. Pass the address to both ATI tools, which default to 192.168.1.1:
  `scripts/ati_netft_logger.py --ip 192.168.123.50` and `scripts/ati_calibration_session.py --ati 192.168.123.50`.
- **ATI's defaults** (manual 9620-05-NET FT-23): DHCP on, falling back to the static 192.168.1.1 after 30 s with no
  DHCP answer. DIP switch 9 on forces 192.168.1.1. PoE is IEEE 802.3af class 0, Mode A only (power on the data pairs),
  6 W at most; the alternative is 11 to 25 V DC on the M12 Pwr/CAN connector. The box is up when its MS and Xdcr
  lights turn green, about 20 s after power-up. A flickering yellow light on the switch's port was traffic, not a
  fault.

## The HEX-E: rate, filter, drift

The DAQ answers each poll with its latest single sample, so a slower poll alone leaves the noise as it is. Its low-pass
filter (command 0x0081) lowers it. Fz noise per sample (1 sigma), polled at 100 Hz with the arm off, 2026-10-22:

| Filter code | Cutoff | Fx, Fy (N) | Fz (N) | Tx, Ty, Tz (N m) |
|---|---|---|---|---|
| 0 | none (also 0.53 N at 250 and 500 Hz) | 0.086, 0.059 | 0.534 | 0.0072, 0.0095, 0.0023 |
| 2 | 150 Hz | 0.060, 0.050 | 0.352 | 0.0044, 0.0055, 0.0017 |
| 3 | 50 Hz (the program's default now) | 0.043, 0.050 | 0.213 | 0.0028, 0.0038, 0.0010 |
| 4 | 15 Hz (the data sheet's figures) | 0.034, 0.044 | 0.131 | 0.0016, 0.0021, 0.0006 |

The owner asked for 100 to 200 Hz ("at 500 things are really noise"). The program offers 50, 100, 250 and 500 Hz.

Two open questions about the sensor:

- **Untared Fz.** It reads about +611 to +645 N on this 200 N sensor, and every datagram's status word is 0x40000. The
  box's own texts call that bit "Sensor overload warning reached". The program tares at its start, so the runs go on.
  The cause is not known. One test tells the two candidates apart: read Fz with nothing bolted to the HEX-E's tool side.
  Near zero means the mount's screws preload the sensor; about 600 N means the sensor's own zero has drifted.
- **Drift.** In the same pose at home, the untared Fz read 627.7 N at 18:10 and 645.1 N at 18:55. The box had been
  power-cycled at about 17:50 (its sensor cable came loose when the box moved under the table), so this is likely
  warm-up. Watch the drift before trusting the low force levels (0.25 and 0.5 N).

## The geometry as measured

Frames as in the program: B the robot's base, E the flange (tool0), M the mount, I an indenter (z up from its base,
x_I along the tool axis toward the tip at homing).

- **The mount's turn about the tool axis is −92.15 deg** from the flange's x. The palmar side (x_M) lies along the
  flange's −y, not on its +x as the plan of the workstation assumed. With the palmar side down at homing, the flange's x
  stays horizontal along base +x. The −90 deg part is the mount's clocking. The −2.15 deg is the mean of the two
  seatings' rolls, which were otherwise ±0.58 deg of seating noise. Homing, the plan and the guarded moves all use
  −92.15. The session's register stage refines it.
- **TCP.** The pad's contact point (the homing's contact point, a_M = (10.55, 0, 93.36) mm in M) sits at
  (−0.40, −10.54, 128.36) mm in the flange frame.
- **Indenters.** Both are on one 2x4. The table below gives each base's pose, levelled about its key (tilt set to 0).
  The 2x4's top is level along its length: the two bases differ by 0.4 mm in z over 151 mm. "Left" and "right" are as the
  owner faces the bench, which is mirrored from the camera's view.

  | | origin (mm) | yaw | top z (mm) | key at the socket's mouth (mm) | as seated: roll / pitch |
  |---|---|---|---|---|---|
  | left: 5 mm sphere | (−138.5, −595.6, 70.8) | −92.3 deg | 116.8 | (−106.8, −552.4, 115.3) | −0.58 / −0.08 deg |
  | right: 25 mm rod | (−289.5, −585.5, 70.3) | −94.1 deg | 112.8 | (−256.4, −543.4, 111.4) | +0.58 / +1.30 deg |

- **The 2x4** runs along base x, about 1 m long, clamped at both ends with C-clamps. It stands on the cutting-mat table,
  whose top is at z_B about 32.8 mm (70.8 − 38). Each fixture's key overhangs its edge toward the robot.
- **Home** (recorded at 18:28) has joints (48.29, −86.25, 109.55, −24.82, 47.34, 6.43) deg and the flange at
  (−175.8, −446.6, 264.3) mm, with the tool axis along base −y. It stands above and between the indenters.

## How it was measured

1. The owner turned the 2x4 end for end, so that each key stands between its post and the robot. That is where the
   CAD's homing puts the key: 44.36 mm behind the post along the tool axis and 30 mm to its +y side.
2. In teach mode (`teach_hold.py`), the owner seated the mount's socket on each key with the palmar side down. This
   session then ended teach mode, so the arm locked, and read the flange pose for 1 s over RTDE receive
   (`record_pose.py`, into `poses.jsonl`).
3. `indenters.py` turns a seating into the indenter's pose: T_BI = T_BE T_EM T_IM_home^-1, with the plan's homing
   geometry. It then levels the pose about the key, because the seating fixes the key's place better than its angle.
4. `move_guarded.py` checked the result. It moved the pad's contact point to 2 mm above each indenter's top, in the
   homing's orientation, and the owner checked the gaps by eye (logged in `moves.jsonl`):
   - sphere: reached within 0.08 mm; the force changed at most 0.98 N; the owner saw "About 2 mm, centred";
   - rod: reached within 0.05 mm; the force changed at most 0.66 N; the owner saw "About 2 mm, centred".

`move_guarded.py` is the only tool here that moves the robot. Every move is straight-line and slow:
- it rises 10 mm at 3 mm/s, then 15 mm at 10 mm/s, before travelling;
- it travels at 25 mm/s;
- it makes the last 30 mm at 5 mm/s.

It tares the HEX-E at rest before each move, then calls `stopL` if the force changes past the move's limit. The limits
are 3 N for travel, 2 N for the final approach, and 25 N for lifting off a key: a hand seating leaves the hand's push on
the drafted key, and its release first tripped a 3 N limit. Before any motion it checks the clearances of every robot
link and tool part against both indenters, the 2x4 and the table (scripts/ur5_cb3_cell.py). The cell model draws each
key as a solid block, so the fork overlaps it in the model until 10 mm of lift. That check is waived for the lift-off
alone.

## The plan for this cell

`20261022-plan_sphere_measured_rod.json`, built here in 61 s:

```
/usr/bin/python3 scripts/ur5_tip_calibration_plan.py --indenter sphere --placement=-138.5,-595.6,70.8,-92.3 \
    --block 38 --block-half 55,500 --mount-turn=-92.15 --other=rod,-289.5,-585.5,70.3,-94.07 \
    --out docs/experiments/20261022-ur5_tip_calibration/lab/20261022-plan_sphere_measured_rod.json
```

It has 155 actions:
- gravity 12, mount 6, pad 32, pivot 7, grid 74, shear 7, roll 4, twist 8, final 1, final gravity 4;
- 6 dropped: grav-08, grav-11, fgrav-04 and fgrav-05 have no IK at this placement, and pivot-7 and pivot-8 came too near
  the rod fixture (pivot-8 put the HEX-E 7.3 mm into it);
- no travel path is short of clearance.

As a check, the builder reproduces the workstation's plan here (159 actions; `build_default.log`).

Obstacles the model leaves out:
- the C-clamps at the 2x4's ends;
- the mat table beyond 200 mm from the sphere (the model has its plane at z_B 0 there, while the real top is at 32.8).

The dry run, 15 mm off every surface, exercises the whole plan with no contact. Watch it from the overhead camera.

## Changes to the code (2026-10-22)

- `ur5_tip_calibration.py`:
  - `--ft-rate` now defaults to 100 Hz;
  - a new `--ft-filter` (default 3, a 50 Hz cutoff) sets the DAQ's low-pass, which configure() used to fix at 0;
  - preflight's rate check now compares with the requested rate (`< 0.8 * ft_rate`, in place of `< 100`, which a
    100 Hz poll fails on about half its runs);
  - a new `--home-from FILE:LABEL` takes a recorded seating in place of the three freedrive seatings: nothing moves and
    nobody is prompted, and a failed placement check stops the run instead of waiting for an answer.
- `ur5_tip_calibration_sim.py`: the simulated HEX-E runs at the program's rate, and its noise is the measured noise for
  the program's filter (`HEX_NOISE_MEASURED`).
- `ur5_tip_calibration_plan.py`: three new options, `--mount-turn`, `--block-half` and `--other kind,x,y,z,yaw`.
- `ur5_cb3_cell.py`: other indenters (`OTHER`) count as obstacles, and the UR5 URDF is found under
  `~/workspace/rh56_controller` on this computer.

Simulator checks (stages preflight, homing, gravity, mount):
- canonical plan: mount onsets +0.14 to +0.53 mm;
- this plan with the simulator's own homing: +0.17 to +0.55 mm;
- this plan with `--home-from` fed a seating recorded in the simulator: +0.16 to +0.52 mm.

All are inside Table 2's 1 mm. Feeding the real seating into the simulator fails the mount touch-offs, as it should:
the simulator moves its indenter by a few millimetres on purpose, so a real pose misses its world.

## The unattended runs

Run each from the clone's root in tmux. `tipcal_run.sh NAME [options]` starts one in the tmux session `tipcal` with the
variables below, copies its printout to `logs/<date>-<time>-tipcal-NAME.log`, and refuses while the program runs or when
the plan or the seating is a git-lfs pointer. Written out:

```
PY=/usr/bin/python3
P=src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py
PLAN=docs/experiments/20261022-ur5_tip_calibration/lab/20261022-plan_sphere_measured_rod.json
SEAT=docs/experiments/20261022-ur5_tip_calibration/lab/poses.jsonl:left_indenter_seated   # not HOME: that would change $HOME
FT="--ft 192.168.0.6"          # --ft-rate 100 --ft-filter 3 are the defaults now
```

1. Stop the viewer (`pkill -f wrench_viewer.py` from a script, never in a command line that also names it). Check
   that the robot is at home and the slider is at 30 %.
2. First run, at 30 %: `$PY $P --plan $PLAN $FT --home-from $SEAT --stages preflight,homing,gravity,mount --name first`.
   Compare its printout with Table 2 of `../README.md`. Expect the HEX-E noise of the table above, not the data sheet's.
3. Dry run: `$PY $P --plan $PLAN $FT --home-from $SEAT --dry 15 --name dry`. Start it at 30 %. Raise the slider to
   100 % (`rtde_io.RTDEIOInterface('192.168.0.4').setSpeedSlider(1.0)`) once the first actions pass.
4. Session: `$PY $P --plan $PLAN $FT --home-from $SEAT --name sphere1`. Resume with `--resume logs/<date>-tip_calibration/<time>-sphere1`.
5. Light analysis runs here (`scripts/ur5_tip_calibration_analysis.py`). The session folder and its results go to the
   repository. The heavy simulation goes to the work computer.

The program aborts an action on any of these: a load over 10 N or 1.5 N m in a press, over 8 N in travel, no HEX-E
sample for 100 ms, or a protective stop. After a protective stop it waits for Enter. Unattended, that wait is a safe
stop: notify the owner (PushNotification reaches the phone through Remote Control) and push the state. Before the
runs, and between them, check the overhead camera's view.

## Files in this folder

| File | What it holds |
|---|---|
| `poses.jsonl` | the recorded poses: the home before and after the 2x4 turned, and the seatings on both keys |
| `moves.jsonl` | every guarded move: target, reached pose, error, the force's largest change, why it stopped |
| `record_pose.py`, `indenters.py`, `move_guarded.py`, `teach_hold.py`, `grab_d435.py` | the tools above (`grab_d435.py NAME DIR` saves elsewhere) |
| `tipcal_run.sh` | starts one run of the program in tmux, with its printout copied to `logs/` |
| `<date>-<time>-<name>/` | each run's session folder, its printout and overhead frames (`NOTES.md` has the runs) |
| `20261022-plan_sphere_measured_rod.json` | the plan for this cell (155 actions) |
| `build_measured_rod.log`, `build_default.log` | the builder's output: this plan, and the reproduction of the workstation's |
| `sim_canonical.log`, `sim_measured_simhome.log`, `sim_seating.jsonl` | the simulator checks |
| `1830_after_rotation_*`, `1830_fixtures_zoom.png` | the overhead D435 frame after the 2x4 turned (colour, depth in metres, intrinsics) |
| `teach.log` | the teach-mode switches |

## Outside this folder

The lab computer's disk was 95 % full. The cleanup's root-only part waits for the owner to run
`sudo bash /media/ExpData1/20261022-corvus_cleanup/cleanup_root.sh`. It frees about 90 GB, and its log goes next to it.
Backups of MAGPIE's local-only content and of Humanoid_Simulation's local edit are in the same folder. The clone of
Humanoid_Simulation now tracks correlllab/golem; its images are not built (owner: "dont need to build golem until i
ask"). The unmounted Samsung SSD (/dev/sdc) is a coworker's: leave it alone.
