#!/bin/bash
# The second A/B of the night of 2026-11-17, after night_rest5.sh (the boards' sweep and S1(a)). The owner put the sphere
# back at the table's centre at about 20:3x ("i placed it back at the center,. leaving now. good luck"). The first A/B's
# 2 ms filter sub-steps cost the server about 10 ms a period (hand-4a, 20:2x), so this one uses the workstation's
# --filter-exact (the paper's continuous filter integrated over each period at one step's cost): twelve g0 cycles at
# (70, 110), four arms in turn, three each: tips off as before (47051), tips off with --filter-exact (47059), tips on as
# before (47052), tips on with --filter-exact --contact-off 0.10 (47060). It waits for night_rest5.sh and starts only
# when its part of the log says "night done" and both cameras' frames are bright enough (mean luma 60 or more).
#     nohup setsid bash docs/experiments/20261117-sr2_system_identification/lab/night_rest6.sh >> logs/20261117-night_rest.out 2>&1 &
set -u
cd /home/will/workspace/hand || exit 1
H=docs/experiments/20261115-hom_bench_trials
PY=/usr/bin/python3
L=logs/20261117-night_rest
mkdir -p $L
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "second A/B waiting for the sweep and S1(a) to end"
while pgrep -f "[n]ight_rest5.sh" > /dev/null; do sleep 20; done
sed -n '/fourth part/,$p' logs/20261117-night_rest.out | grep -q "night done" || { log "the sweep and S1(a) did not end with 'night done': no second A/B"; exit 1; }
[ -e $L/STOP ] && { log "STOP before the second A/B"; exit 0; }
LUMA=$(cd /tmp && timeout 60 $PY - <<'EOF' 2>/dev/null
import numpy as np, pyrealsense2 as rs
out = []
for sn in [d.get_info(rs.camera_info.serial_number) for d in rs.context().devices]:
    p, c = rs.pipeline(), rs.config(); c.enable_device(sn); c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    p.start(c)
    for _ in range(30): f = p.wait_for_frames(5000)
    out.append(np.asanyarray(f.get_color_frame().get_data()).mean()); p.stop()
print(" ".join(f"{x:.0f}" for x in out))
EOF
)
log "second A/B: lights (mean luma) $LUMA"
for x in $LUMA; do awk "BEGIN{exit !($x < 60)}" && { log "too dark for the tracker: no second A/B"; exit 2; }; done
[ -n "$LUMA" ] || { log "no camera frames: no second A/B"; exit 2; }
(cd $H && ARMS="47051:off:base 47059:off:exact 47052:on:base 47060:on:exact_hyst" \
    bash lab/repeat_cycle.sh ../../../logs/20261116-hom_trials/20261117-night_repeats2 12)
log "second A/B ended (exit $?)"
(cd $H && $PY lab/hom_setup.py diag 2>&1 | grep -E "volts|temperature|status" | cut -c1-200)
log "second A/B done"
