#!/bin/bash
# The end of the night of 2026-11-17: the twist repeats stopped at 11 of 12 (20:08, cycle 11, tips on as before: the drift
# guard at 27.4 mm, then the sphere fell out of the hand and lies on its side 227 mm from the table's centre; the
# fingers were straightened clear of it). Neither of the last two series needs the sphere or the cameras: the boards'
# random-pose sweep at (85, 170) with the arm 70 mm higher, then S1(a), each after the step before ends cleanly.
#     nohup setsid bash docs/experiments/20261117-sr2_system_identification/lab/night_rest5.sh >> logs/20261117-night_rest.out 2>&1 &
set -u
cd /home/will/workspace/hand || exit 1
ID=docs/experiments/20261117-sr2_system_identification
H=docs/experiments/20261115-hom_bench_trials
XT=docs/experiments/20261104-sr2_hand_crosstalk
ROLL=plans/20261115-roll_press_plan.json
PY=/usr/bin/python3
L=logs/20261117-night_rest
O=logs/20261117-sr2_s1a_ranges
mkdir -p $L $O
log() { echo "[$(date +%H:%M:%S)] $*"; }
stop_check() { [ -e $L/STOP ] && { log "STOP before $1"; exit 0; }; }
stop_check sweep
log "fourth part: board sweep, layout (85, 170), arm 70 mm up"
(cd $H && $PY lab/hom_trial.py return --lift-mm -70 2>&1 | tail -1)
(cd $H && $PY lab/hom_setup.py layout --plan $ROLL --go 2>&1 | grep -v -i token | tail -1)
(cd $H && $PY lab/hom_setup.py status 2>&1 | grep -v -i token | tail -2 | cut -c1-200)
$PY -u $XT/lab/run_plan.py logs/20261117-sr2_boards_sweep/20261117-plan_random_s3.json logs/20261117-sr2_boards_sweep/s3 \
    --token-file ~/.sr2_station_token --min-clearance 5 --max-blocked 3 --keep-home --max-temp 55
src=$?
log "sweep ended (exit $src): $(tail -1 logs/20261117-sr2_boards_sweep/s3/events.jsonl | cut -c1-200)"
(cd $H && $PY lab/hom_setup.py diag 2>&1 | grep -E "volts|temperature|status" | cut -c1-200)
[ $src -eq 0 ] || { log "the sweep did not end cleanly: no S1(a)"; exit 3; }
stop_check S1a
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
[ -f $O/ends.json ] || { log "S1(a): no checked ends"; exit 4; }
log "S1(a): sweeps"
(cd $ID/lab && $PY s1a_ranges.py run --go 2>&1 | grep -v -i token | tail -24)
(cd $H && $PY lab/hom_setup.py diag 2>&1 | grep -E "volts|temperature|status" | cut -c1-200)
log "night done"
