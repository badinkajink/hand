#!/bin/bash
# The second half of the night of 2026-11-17 after night_rest.sh stopped at 19:49:58 (cycle 4's putdown: its 28.6 deg
# levelling turn outran the 12 s move limit; the sphere was put down by hand-run putdown at 19:50 and released upright).
# The remaining eight g0 twist cycles over the same four arms in the same order (two each; repeat_cycle.sh now levels at
# most 20 deg), then, when they end cleanly, the boards' random-pose sweep as in night_rest.sh.
#     nohup setsid bash docs/experiments/20261117-sr2_system_identification/lab/night_rest2.sh >> logs/20261117-night_rest.out 2>&1 &
set -u
cd /home/will/workspace/hand || exit 1
H=docs/experiments/20261115-hom_bench_trials
XT=docs/experiments/20261104-sr2_hand_crosstalk
ROLL=plans/20261115-roll_press_plan.json
PY=/usr/bin/python3
L=logs/20261117-night_rest
mkdir -p $L
log() { echo "[$(date +%H:%M:%S)] $*"; }
stop_check() { [ -e $L/STOP ] && { log "STOP before $1"; exit 0; }; }
log "second half: lights: $(cd /tmp && timeout 60 $PY - <<'EOF' 2>/dev/null
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
stop_check repeats
log "HOM twist repeats, cycles 5-12"
(cd $H && ARMS="47051:off:base 47057:off:fstep 47052:on:base 47058:on:fstep_hyst" \
    bash lab/repeat_cycle.sh ../../../logs/20261116-hom_trials/20261117-night_repeats 8)
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
