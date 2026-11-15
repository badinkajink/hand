# HOM trials on the SR2 hand: runbook

How to run a HOM trial on the SR2 hand on the UR5, record it, and use it to calibrate the simulation (owner,
2026-11-15: "make sure we are ready to start collecting data and calibrating from real trials of HOM"). The trial
sequence and the series are those of `docs/notes/20261105-object_seating_and_tracking/20261105-object_seating_and_tracking_plan.md`.

## State on 2026-11-15 14:00

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

Measured on corvus on 2026-11-13 (`docs/experiments/20261113-tag_tracking_throughput/`): tracker 29.9 Hz with 42 ms frame
age; link round trip 3.4 ms median to the workstation; HOM 4.0 ms per step on corvus's CPU.

Blocking a real trial:

1. Joint angles during a stream. The station does not read the servos while a policy stream runs
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
2. The hand model for the five layouts. HomBench's CPU path (`hom_tasks.build`) has the cad_v1 hand and the 10-22
   grasps only; the five layouts of the series and their grasps are in the GPU study's worlds. A builder that makes
   HomBench from such a world is being written (`plans/`, see below).
3. Tag maps of the printed objects' blocks. Each block needs about 40 two-camera views and a bundle adjustment
   (`sr2hand.tracking.pose.bundle_adjust_multi`, as for the test block on 2026-11-13), then the map moved into the
   object's frame against the CAD map (`object_tracking_mjcf.object_map`).
4. The seat and the open pose: the flat seat, its height for the arm's pose, and the station's plan for the layout with
   its open pose.

## Before a session

- Weigh each object with its stem and block; the simulation takes the measured mass.
- Grip tape on the surfaces HOM grips (the HOM results use friction 1.0; bare PLA gives 0.37-0.44, under which the
  twist fell from 0.91 to 0.08 in simulation).
- Sync corvus (its clone cannot fetch from GitHub): on the workstation
  `git push --no-verify corvus:workspace/hand main:refs/remotes/origin/main`, then on corvus
  `GIT_LFS_SKIP_SMUDGE=1 git merge --ff-only origin/main`.

## A trial

1. Workstation, the HOM server for the task (plant `measured`: the SCS0009 at 0.55 N m/rad; `servo`: 0.72):

       .venv/bin/python -m sr2hand.control.hom_server serve --task press --d 25 --plant measured \
           --listen 47002 --reply CORVUS_TAILSCALE_IP:47001 \
           --log-dir docs/experiments/20261115-hom_bench_trials/runs

2. corvus, the object on the seat and the hand over it:

       /usr/bin/python3 docs/experiments/20261115-hom_bench_trials/lab/hom_trial.py check --object cylinder --map MAP.json

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
