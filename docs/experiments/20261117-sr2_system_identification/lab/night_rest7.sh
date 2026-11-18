#!/bin/bash
# S1(a) again for the five ends that its first rule stopped early (2026-11-17 21:0x), after the second A/B
# (night_rest6.sh). The first rule (2.5 deg of lag on two steps of 0.35 s) stopped the index's and the middle's yaw at
# -35.2 and -27.5 deg and the middle's MCP at +22.6 deg, where S2's dwells at the same layout show them at -40 and 61 deg
# within 0.9 deg: friction lag, not stops. s1a_ranges.py now stops a sweep when the reading stalls (under 0.6 deg over six
# steps while 2 deg behind) or lags 6 deg. Runs only when the second A/B ended cleanly (exit 0): the arm 70 mm up, the
# roll layout (85, 170), the ends checked at the station's mounts, then the five sweeps.
#     nohup setsid bash docs/experiments/20261117-sr2_system_identification/lab/night_rest7.sh >> logs/20261117-night_rest.out 2>&1 &
set -u
cd /home/will/workspace/hand || exit 1
ID=docs/experiments/20261117-sr2_system_identification
H=docs/experiments/20261115-hom_bench_trials
ROLL=plans/20261115-roll_press_plan.json
PY=/usr/bin/python3
L=logs/20261117-night_rest
O=logs/20261117-sr2_s1a_ranges
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "S1(a) redo waiting for the second A/B to end"
while pgrep -f "[n]ight_rest6.sh" > /dev/null; do sleep 20; done
sed -n '/second A\/B: lights/,$p' logs/20261117-night_rest.out | grep -q "second A/B ended (exit 0)" || { log "the second A/B did not end cleanly: no S1(a) redo"; exit 1; }
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
