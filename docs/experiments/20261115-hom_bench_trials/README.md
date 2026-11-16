# HOM trials on the SR2 hand: runbook

How to run a HOM trial on the SR2 hand on the UR5, record it, and use it to calibrate the simulation (owner,
2026-11-15: "make sure we are ready to start collecting data and calibrating from real trials of HOM"). The trial
sequence and the series are those of `docs/notes/20261105-object_seating_and_tracking/20261105-object_seating_and_tracking_plan.md`.

## State on 2026-11-15 17:41

Built and checked without hardware:

- HOM controller server `src/sr2hand/control/hom_server.py` (`serve`). It answers corvus's state messages with
  HomBench's targets at 50 Hz and records every period. Over UDP against a simulated hand (`sil`), its targets equal an
  in-process loop through the same messages on all four tasks.
- Trial script for corvus `lab/hom_trial.py`: `check`, `trial` (link loop, the arm's lift at 1.6 s, a per-period record
  with the arm pose and the HEX-E), `return` (guarded move down). Its defaults move nothing. A dry run against a local
  server ran 302 periods and stopped at the schedule's end.
- The tracker places each frame with the arm's pose at the frame's capture time (`sr2hand.tracking.live.ArmPose.at`).
  Before, a frame 42 ms old was placed with the newest pose, up to about 8 mm off during the lift.
- The station's servo readings are stamped with their own age (`servo_age_s`), so stale readings during a stream show
  as stale.
- Plans of the first series (`plans/`, 14 files; `scripts/hardware/hom_bench_plans.py`): the five layouts of the
  task-conflict study with each task's S3 grasp, forces and contact-mode weights, mounts in the driver's coordinates
  (`sr2hand.robot.sr2_layouts`: driver = P - o, o = (-6.18, -2.40) mm for the thumb, (6.18, 3.40) for the index and
  (6.18, -3.40) for the middle). The sphere tasks run only at the generalist and pin-push layouts (the seating plan's
  decision 2). HomBench builds each plan's GPU world on the CPU (`HomBench.from_plan`, `scripts/hom/hom_bench_world.py`).
- `hom_server nominal --plan PLAN --task T --d D --out NOMINAL.json` gives the object's start pose in P for that
  grasp, and `hom_trial.py check --nominal NOMINAL.json` the tracked pose's distance from it.
- `scripts/hardware/object_tag_map.py` moves a block's fitted tag map into the object's frame (the 11-07 test block:
  0.33 mm RMS against the CAD).

Measured on corvus on 2026-11-13 (`docs/experiments/20261113-tag_tracking_throughput/`): tracker 29.9 Hz with 42 ms frame
age; link round trip 3.4 ms median to the workstation; HOM 4.0 ms per step on corvus's CPU.

Done since 14:15:

- The CB1 runs the read-back patch (owner's go, 2026-11-15: "go ahead on the CB1 patch and restart"). It was applied with
  `git apply -p0` (the CB1 has no `patch`), the originals kept in `~/backups/20261115-stream_read_back/` on the CB1, and
  the station restarted at about 16:12. On the CB1's own patched files with fake hardware
  (`scripts/hardware/station_read_back_check.py`, now without numpy): 50 frames, read-back equal to the written pose,
  2.96 ms per frame. The restarted station reads the nine servos (servo age 2 ms); it is unhomed with no plan loaded.
- The pin-push specialist runs at (75, 56.8), the owner's choice ("sure we can use that layout"): its four plans were
  rewritten there at about 16:18 with the study's pick (`hom_bench_plans.py write`; meta.hom keeps the study's layout).
  Its index and middle gantries sit at the end of their 60 mm travel (margin 0.00 mm). The twist grasp fails the pose
  toy's constraints there (meta.hom.toy_feasible false) and scores 0.619 against 0.641 at the study's layout (CPU, seeds
  20-24).
- From 14:15 to 16:22 `hom_bench_world.make_bench` ran without the plant's hold of the object (HomBench's own hold, 0,
  replaced the world's). The server and the SIL runs build HomBench from the plan and were not affected; the plans'
  check runs on make_bench again from 16:22 (the generalist pin-push plan gives 0.7273 again).
- Stickers for the backs of the DIP links (`assets/tracking/20261115-dip_link_tags/`) and the gantry travel measurement
  (`docs/experiments/20261115-gantry_travel/`): the stage scale, axis direction, backlash and travel ends from the tags.

Blocking a real trial:

0. The gantries (resolved at 17:38; gantry page `docs/experiments/20261115-gantry_travel/
   20261115-gantry_travel_dip_tags.html`). In the owner's DIP-tag session (16:53-17:13) the middle's palm-y stage J4
   moved nothing: its motor was unplugged at the CB1's stepper board, and a home that ends on its timeout (result 3)
   reports homed whether the axis moved or not. Plugged back in at 17:38, J4 moved 4.93 mm toward +y_P for a 5 mm
   command, the station's direction, and homed to its -85 end (tag at y_P -86.93, 0.04 mm from its earlier home). All
   six stages are homed at their stops (counters 0, checked against the tags); the station is stopped, waiting for the
   owner, and a restart turns the servos on before the home is adopted. By the owner's hand check, the index's palm-y
   stage J2 reaches about 1.2 mm past the station's limit at y 25 (to about 23.8 mm), and the other stages end at their
   commanded limits. With both cameras the stages travelled 0.99 to 1.00 times their command (index x 59.50 of 60 mm,
   thumb y 109.92 and 109.85 of 110, thumb x 60.07 of 60), and every jointly solved axis turned 1.3 to 2.3 deg
   clockwise about z_P: the hand sits about 2 deg in yaw off the tool chain's P, or the hand-eye does; the hand's tag 6
   can tell which.
1. Joint angles during a stream (resolved at 16:12: the CB1 is patched). The station does not read the servos while a policy stream runs
   (`manta_hand/runtime.py`, `_telemetry_loop`). Without measured angles HOM fails: in the simulated loop on the cad_v1
   hand it dropped the sphere on the twist and the roll (J 0 against 0.352 and 0.315 with the angles), whether its
   reference or its targets stood in for them. Change on the CB1 (needs the owner's go; the CB1's working copy runs
   uncommitted edits, so the change is made on that file alone): `stream_frame(..., read_back=True)` reads the nine
   positions after each write (`servos.sync_read_joint_positions`, nine reads in about 9 ms at a 1 ms FTDI latency
   timer) and returns them; `/api/v1/stream/frame` passes `read_back` and answers with them; `hom_trial.py` posts frames
   from a worker thread and takes them as the next state's joints. Made in the repository's copy and checked on the
   station's fake hardware (`scripts/hardware/station_read_back_check.py`: 50 frames, read-back equal to the written
   servo pose, 2.1 ms per frame without the real bus). The same change against the CB1's own files, whose `runtime.py`
   differs from the repository's only in comments and whose `web.py` keeps its own defaults, is
   `cb1/20261115-stream_read_back.patch`. To apply, with the owner's go and the station stopped:

       scp -o ProxyJump=corvus docs/experiments/20261115-hom_bench_trials/cb1/20261115-stream_read_back.patch irlab@10.99.99.2:/tmp/
       ssh -J corvus irlab@10.99.99.2 'cd ~/hand && D=src/morphohand/driver/manta/host/manta_hand && \
           cp $D/runtime.py /tmp/runtime.py.bak-20261115 && cp $D/web.py /tmp/web.py.bak-20261115 && \
           patch -p0 --dry-run < /tmp/20261115-stream_read_back.patch && patch -p0 < /tmp/20261115-stream_read_back.patch && \
           python3 -m py_compile $D/runtime.py $D/web.py'

   then `~/run_control_station.sh start` on the CB1. An unpatched station ignores `read_back` and answers without the
   positions.
2. The pin-push specialist layout (resolved: (75, 56.8)). (75, 55) does not fit the hand: its index and middle sit
   0.90 mm past their travel boxes (60.90 mm of the steppers' 60 mm). At (75, 56.8) the S3 picks score 0.916, 0.875,
   0.583 and 0.619 (pin-push, press, roll, twist; CPU, seeds 20-24) against 0.917, 0.874, 0.590 and 0.641; its twist
   grasp fails the pose toy's constraints.
3. Tag maps of the printed objects' blocks. Each block needs about 40 two-camera views and a bundle adjustment
   (`sr2hand.tracking.pose.bundle_adjust_multi`, as for the test block on 2026-11-13), then the map moved into the
   object's frame against the CAD map (`object_tracking_mjcf.object_map`).
4. The seat: the object rests on its block on a flat surface, and the arm brings the hand to it (`check --nominal`).

The plans in the simulated loop (`hom_server sil --plan`, the cad_v2 hand, the measured SCS0009 at 0.55 N m/rad, seed
20; UDP run equal to the in-process loop on each). J is the score of `hom_landscape.track_score`:

| Layout (x_sep, y_sep mm) | Pin-push | Press | Roll | Twist |
|---|---|---|---|---|
| Generalist (70, 110) | 0.277 | 0.935 | 0.404 | 0.439 |
| Pin-push (75, 56.8) | 0.373 | 0.764 | 0.364 | 0.421 |
| Press (70, 150) | 0.008 | 0.566 | | |
| Twist (100, 70) | 0.286 | 0.487 | | |
| Roll (85, 170) | failed | 0 (dropped) | | |

The pin-push row is at (75, 56.8) since 16:29 (at the study's (75, 55): 0.372, 0.764, 0.393 and 0.408;
`logs/20261115-hom_server_sil/sil_v3_pinpush_*.log`). At the roll layout HOM's Eq. (2) least-squares solve failed on the cylinder's diverged state (pin-push at 6.34 s); the
server then holds its last targets and stops the trial.

## Before a session

- The station turns the servos on when it starts. A finger posed by hand past a joint's goal limit snaps back to it
  (on 11-15 the index's MCP servo sat at 149.7 deg against its 143.6 deg limit, a 6 deg snap). Pose the fingers inside
  their limits, or keep hands clear, before `~/run_control_station.sh start`.
- Weigh each object with its stem and block; the simulation takes the measured mass.
- Grip tape on the surfaces HOM grips (the HOM results use friction 1.0; bare PLA gives 0.37-0.44, under which the
  twist fell from 0.91 to 0.08 in simulation).
- Sync corvus (its clone cannot fetch from GitHub): on the workstation
  `git push --no-verify corvus:workspace/hand main:refs/remotes/origin/main`, then on corvus
  `GIT_LFS_SKIP_SMUDGE=1 git merge --ff-only origin/main`.

## A trial

1. Workstation, the HOM server for the task and layout (plant `measured`: the SCS0009 at 0.55 N m/rad; `servo`: 0.72):

       .venv/bin/python -m sr2hand.control.hom_server serve --task press --d 25 --plant measured \
           --plan docs/experiments/20261115-hom_bench_trials/plans/20261115-generalist_press_plan.json \
           --listen 47002 --reply CORVUS_TAILSCALE_IP:47001 \
           --log-dir docs/experiments/20261115-hom_bench_trials/runs

   The station loads the same plan file (its mounts are the driver's), so the gantries and the controller agree.

2. corvus, the object on the seat and the hand over it:

       .venv/bin/python -m sr2hand.control.hom_server nominal --task press --d 25 --plant measured \
           --plan .../20261115-generalist_press_plan.json --out NOMINAL.json        # on the workstation, once per plan
       /usr/bin/python3 docs/experiments/20261115-hom_bench_trials/lab/hom_trial.py check --object cylinder \
           --map MAP.json --nominal NOMINAL.json

   The object's pose in P should be within 1 mm and 2 deg of the grasp's nominal pose.
3. corvus, a dry run through the real tracker (nothing moves; the server answers, the record is written):

       /usr/bin/python3 .../hom_trial.py trial --task press --controller WORKSTATION_IP:47002 --object cylinder \
           --map MAP.json --station http://10.99.99.2:8765 --out-dir OUT

4. With the owner present, the station homed with torque on and the layout's plan loaded: add `--live` (the fingers
   move), then `--arm --ft` (the lift at 1.6 s; the HEX-E's force change past 5 N or a protective stop ends it).
5. After the trial the hand holds the object in the air. The operator looks, then `hom_trial.py return` lowers the
   hand 40 mm at 10 mm/s (stopped by a 3 N force change).

## Records

Each trial gives two JSON-lines files with one line per 20 ms period, joined by the state's sequence number:

- corvus (`OUT/<stamp>_<task>_corvus.jsonl`): the state sent (joints and their time, the object's pose in P with its
  capture time, tracker flags and tag count), the arm's TCP pose, the HEX-E's wrench, the target applied with its age
  and round trip, the lift's phase.
- workstation (`runs/<stamp>_<task>_d<d>_trial<n>.jsonl`): the inputs HOM used (and where its joint angles came from),
  its targets in servo and sim degrees, the phase, the schedule's references, the compute time, any error.

Raw records stay out of git when over 2 MB; they move to the workstation by rsync.

## Scoring and calibration

    .venv/bin/python -m sr2hand.control.trial_score merge --server runs/SERVER.jsonl --corvus OUT/CORVUS.jsonl \
        --mass-g 25 --out TRIAL.jsonl                       # one trial file: header, one line per period
    .venv/bin/python -m sr2hand.control.trial_score score TRIAL.jsonl          # J and its inputs
    .venv/bin/python scripts/hardware/sr2_hom_replay.py replay --truth TRIAL.jsonl   # open loop at the header's plant
    .venv/bin/python scripts/hardware/sr2_hom_replay.py fit TRIAL_*.jsonl            # the plant from the trials

The merge joins the two records by the state's sequence number. Gravity in P comes from the arm's pose and the tool
chain, the target delay from corvus's round trips, and the lift's start from corvus's end line. The score follows
`hom_tasks.metrics` from the tracked quaternions; `J_025` counts a pad in contact from 0.25 N, the fingertip boards'
floor. The replay applies the recorded targets open loop from the first sample. The fit minimizes the joint, position
and orientation errors of the replays over kp, damping, torque limit, dead band, joint friction, contact friction and
pad modulus, with a held-out split. Fitting is to trajectories, never to task outcomes (owner, 2026-10-06).

Checked in simulation: a SIL trial on the generalist twist plan, both records written (`hom_server sil --corvus-log`),
merged and scored at J 0.4397 against the simulation's 0.4386, replayed at its own plant to 0.0013 deg RMS in the
joints, 0.0024 mm in the object's position and 0.0079 deg in its orientation. On six synthetic trials at kp 0.55 and
friction 0.8 the fit recovered kp 0.552, friction 0.809, damping 0.0146 and pad modulus 2.58 MPa (truth 0.0144 and 2.5),
in 2.5 to 13 min on 7 CPU workers (`logs/20261115-sr2_hom_replay/`). The torque limit was not reached in those trials
and stayed undetermined.

Not yet in the replay's model: the stem and tag block (the scene's object is the bare 25 g sphere or cylinder) and the
arm's lift (the replay removes the post at the lift's start). A record without fingertip boards has no contacts, which
the score cannot tell from no contact; with the boards, the press's pusher (median 0.18 N in simulation) lies below their
0.25 N floor. The fit's error weights (1 deg, 1 mm, 1 deg) are to be set from the measured noise of the servos and the
tracker.
