#!/bin/bash
# The second A/B's remaining eight cycles (2026-11-17, after night_rest6.sh stopped at 21:13: cycle 4's putdown exited 0
# with the block unseated at 14.5 deg, so the cycle's retry did not run; the sphere was released by hand with
# letgo --tilt-ok and stood at 0.4 deg). repeat_cycle.sh now reads the putdown's text for the unseated case and, when
# a second putdown still leaves it at 17 deg or less, releases with --tilt-ok. Same four arms in the same order, two
# rounds. Then, if the cycles end cleanly, S1(a) again for the five ends its first rule stopped early (night_rest7.sh's
# steps: the arm 70 mm up, the roll layout (85, 170), the ends checked at the station's mounts, the five sweeps).
#     READY_LAT=90 nohup setsid bash docs/experiments/20261117-sr2_system_identification/lab/night_rest8.sh >> logs/20261117-night_rest.out 2>&1 &
set -u
cd /home/will/workspace/hand || exit 1
ID=docs/experiments/20261117-sr2_system_identification
H=docs/experiments/20261115-hom_bench_trials
ROLL=plans/20261115-roll_press_plan.json
PY=/usr/bin/python3
L=logs/20261117-night_rest
O=logs/20261117-sr2_s1a_ranges
log() { echo "[$(date +%H:%M:%S)] $*"; }
[ -e $L/STOP ] && { log "STOP before the second A/B's second part"; exit 0; }
log "second A/B, cycles 5-12"
(cd $H && ARMS="47051:off:base 47059:off:exact 47052:on:base 47060:on:exact_hyst" \
    bash lab/repeat_cycle.sh ../../../logs/20261116-hom_trials/20261117-night_repeats2 8)
rc=$?
log "second A/B part 2 ended (exit $rc)"
[ $rc -eq 0 ] || { log "not clean: no S1(a) redo"; exit 2; }
[ -e $L/STOP ] && { log "STOP before the S1(a) redo"; exit 0; }
(cd $H && $PY lab/hom_trial.py return --lift-mm -70 2>&1 | tail -1)
(cd $H && $PY lab/hom_setup.py layout --plan $ROLL --go 2>&1 | grep -v -i token | tail -1)
(cd $H/lab && $PY - <<'EOF' > /home/will/workspace/hand/$O/mounts.json
import json, sys
from pathlib import Path
sys.argv = ["x"]
import hom_setup as HS
print(json.dumps(HS.Station("http://10.99.99.2:8765", str(Path.home() / ".sr2_station_token")).state()["mount_positions"]))
EOF
)
log "S1(a) redo: mounts $(cat $O/mounts.json)"
(cd $ID/lab && nice -n 10 uv run --no-project --python 3.10 --with mujoco --with numpy --with scipy python s1a_ranges.py check \
    --mounts-json /home/will/workspace/hand/$O/mounts.json 2>&1 | tail -2)
(cd $ID/lab && $PY s1a_ranges.py run --go --only index.yaw.low,middle.yaw.low,middle.mcp.high,index.mcp.low,middle.mcp.low \
    2>&1 | grep -v -i token | tail -8)
(cd $H && $PY lab/hom_setup.py diag 2>&1 | grep -E "volts|temperature|status" | cut -c1-200)
log "S1(a) redo done"
