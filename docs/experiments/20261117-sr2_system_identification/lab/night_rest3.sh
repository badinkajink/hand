#!/bin/bash
# S1(a) of the identification queue after night_rest2.sh (2026-11-17): each joint's reachable range at the roll layout
# (85, 170) where the boards' sweep ended, the arm 70 mm above the trials' height and the fingers straight. Waits for
# night_rest2.sh, goes on only when its log says "night done" after the second half's start, checks the ends in the CAD
# hand at the station's mounts (s1a_ranges.py check, under uv), then runs them (s1a_ranges.py run --go).
#     nohup setsid bash docs/experiments/20261117-sr2_system_identification/lab/night_rest3.sh >> logs/20261117-night_rest.out 2>&1 &
set -u
cd /home/will/workspace/hand || exit 1
ID=docs/experiments/20261117-sr2_system_identification
H=docs/experiments/20261115-hom_bench_trials
PY=/usr/bin/python3
O=logs/20261117-sr2_s1a_ranges
L=logs/20261117-night_rest
mkdir -p $O $L
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "S1(a) waiting for the second half to end"
while pgrep -f "[n]ight_rest2.sh" > /dev/null; do sleep 20; done
sed -n '/second half/,$p' logs/20261117-night_rest.out | grep -q "night done" || { log "the second half did not end with 'night done': no S1(a)"; exit 1; }
[ -e $L/STOP ] && { log "STOP before S1(a)"; exit 0; }
(cd $H/lab && $PY - <<'EOF' > /home/will/workspace/hand/$O/mounts.json
import json, sys
from pathlib import Path
sys.argv = ["x"]
import hom_setup as HS
print(json.dumps(HS.Station("http://10.99.99.2:8765", str(Path.home() / ".sr2_station_token")).state()["mount_positions"]))
EOF
)
log "S1(a): mounts $(cat $O/mounts.json)"
(cd $ID/lab && nice -n 10 uv run --no-project --python 3.10 --with mujoco --with numpy --with scipy python s1a_ranges.py check \
    --mounts-json /home/will/workspace/hand/$O/mounts.json 2>&1 | tail -20)
[ -f $O/ends.json ] || { log "S1(a): no checked ends"; exit 2; }
log "S1(a): sweeps"
(cd $ID/lab && $PY s1a_ranges.py run --go 2>&1 | grep -v -i token | tail -24)
(cd $H && $PY lab/hom_setup.py diag 2>&1 | grep -E "volts|temperature|status" | cut -c1-200)
log "S1(a) done"
