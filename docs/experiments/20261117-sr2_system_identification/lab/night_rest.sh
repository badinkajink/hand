#!/bin/bash
# The rest of the night of 2026-11-17 on the SR2 hand (corvus), after S2 (s2_chain.sh): the camera tasks first, while
# the lab lights are on (they go off at some time in the night; luma 135-142 at 19:12), then the board sweep.
#   1. S1(b): the DIP tags of the thumb and the middle at the poses of logs/20261117-sr2_s1b_tags/poses.json (chosen for
#      the arm's pose after S2 and the (85, 170) layout; CAD clearance >= 14.1 mm); the index's tag faced away from
#      both cameras at every candidate pose.
#   2. The HOM twist repeats (owner, 18:0x), as an A/B of the paper-rate filter (owner, 19:0x: "you gotta do it right";
#      hand-4a's flags, 7cce889e8): lab/repeat_cycle.sh of the bench study, g0 at (70, 110), twelve cycles over four arms
#      in turn, three each: tips off as before (47051), tips off with --filter-step-s 0.002 (47057), tips on as before
#      (47052), tips on with --filter-step-s 0.002 --contact-off 0.10 (47058).
#   3. Only when the repeats end cleanly: the random-pose sweep of the fingertip boards (owner, 19:1x: "good idea to get
#      the magnetic fields response recalibrated too with this new sysid"), at (85, 170) with the arm 70 mm higher,
#      logs/20261117-sr2_boards_sweep/20261117-plan_random_s3.json (300 poses, CAD clearance >= 12.2 mm).
# It waits for s2_chain.sh to end and goes on only when chain.out says "S2 done". STOP in this folder's log dir ends it
# between steps.
#     nohup setsid bash docs/experiments/20261117-sr2_system_identification/lab/night_rest.sh >> logs/20261117-night_rest.out 2>&1 &
set -u
cd /home/will/workspace/hand || exit 1
ID=docs/experiments/20261117-sr2_system_identification
H=docs/experiments/20261115-hom_bench_trials
XT=docs/experiments/20261104-sr2_hand_crosstalk
ROLL=plans/20261115-roll_press_plan.json
PY=/usr/bin/python3
L=logs/20261117-night_rest
mkdir -p $L
log() { echo "[$(date +%H:%M:%S)] $*"; }
stop_check() { [ -e $L/STOP ] && { log "STOP before $1"; exit 0; }; }
log "waiting for S2 to end"
while pgrep -f "[s]2_chain.sh" > /dev/null; do sleep 20; done
grep -q "S2 done" logs/20261117-sr2_sysid_7v5/chain.out || { log "S2 did not end with 'S2 done': stopping here"; exit 1; }
log "S2 done; lights: $(cd /tmp && timeout 60 $PY - <<'EOF' 2>/dev/null
import numpy as np, pyrealsense2 as rs
out = []
for sn in [d.get_info(rs.camera_info.serial_number) for d in rs.context().devices]:
    p, c = rs.pipeline(), rs.config(); c.enable_device(sn); c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    p.start(c)
    for _ in range(30): f = p.wait_for_frames(5000)
    out.append(f"{sn} {np.asanyarray(f.get_color_frame().get_data()).mean():.0f}"); p.stop()
print(", ".join(out))
EOF
)"
stop_check S1b
log "S1(b): DIP tags"
(cd $ID/lab && $PY s1b_tags.py run --plan ../../20261115-hom_bench_trials/$ROLL --go 2>&1 | grep -v -i token | tail -30)
stop_check repeats
log "HOM twist repeats"
(cd $H && ARMS="47051:off:base 47057:off:fstep 47052:on:base 47058:on:fstep_hyst" \
    bash lab/repeat_cycle.sh ../../../logs/20261116-hom_trials/20261117-night_repeats 12)
rc=$?
log "repeats ended (exit $rc)"
[ $rc -eq 0 ] || { log "the repeats did not end cleanly: no board sweep"; exit 2; }
stop_check sweep
log "board sweep: layout (85, 170), arm 70 mm up"
(cd $H && $PY lab/hom_trial.py return --lift-mm -70 2>&1 | tail -1)
(cd $H && $PY lab/hom_setup.py layout --plan $ROLL --go 2>&1 | grep -v -i token | tail -1)
$PY -u $XT/lab/run_plan.py logs/20261117-sr2_boards_sweep/20261117-plan_random_s3.json logs/20261117-sr2_boards_sweep/s3 \
    --token-file ~/.sr2_station_token --min-clearance 5 --max-blocked 3 --keep-home --max-temp 55
log "sweep ended (exit $?): $(tail -1 logs/20261117-sr2_boards_sweep/s3/events.jsonl | cut -c1-200)"
(cd $H && $PY lab/hom_setup.py diag 2>&1 | grep -E "volts|temperature|status" | cut -c1-200)
log "night done"
