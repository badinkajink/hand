#!/bin/bash
# 2026-11-19 (owner's go, 14:0x-14:19, relayed by hand-4a, covering "the 20 mm/s legs"): the index pad on the taped
# plate at phase A's unturned pose (flex, gap 6, dz 5), dragged 5 mm up and down at 2 and 20 mm/s with the arm at
# 0.5 m/s^2 (20 mm/s from 0.4 to 4.6 mm of each leg), three pairs per speed after 1 s rests, at holds of 1 and 2 N.
# A free-air control at the same speeds and acceleration before and after. The hand's slips in the twists run near
# 40 mm/s; the pads' friction rose from 0.31 to 0.40 (index) between 0.2 and 2 mm/s. Stops at the first failure.
cd /home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab
PY=/usr/bin/python3
L=/home/will/workspace/hand/logs/20261119-sr2_s3_press
run() { "$@" 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -v "^HEX-E before\|diagnostics\|^   drag"; return ${PIPESTATUS[0]}; }
echo "$(date +%H:%M:%S) fast legs start"
run $PY day/wait_cool.py index || exit 1
run $PY s3_press.py pose --finger index --pip 0 --go || exit 1
run $PY s3_approach.py plan --finger index --gap 6 --dz 5 || { echo "plan not clear"; exit 2; }
P=$(ls -t $L/*_approach_index.json | head -1); echo "plan $P"
run $PY s3_approach.py go --plan $P --stop-n 5 --stop-down 3 --go || { echo "approach failed or stopped"; exit 3; }
C="--finger index --protocol --speeds 2,20 --rests 1,1,1 --acc 0.5 --max-temp 40"
run $PY s4b_drag.py $C --normals 0 --surface "free air (control)" \
  --note "free-air control before the index pad's 20 mm/s legs, phase A's pose ($(basename $P))" --go || exit 4
run $PY s4b_drag.py $C --normals 1,2 --surface "pad on tape" \
  --note "index pad on the tape at 2 and 20 mm/s, arm at 0.5 m/s^2, phase A's pose ($(basename $P))" --go || exit 5
run $PY s4b_drag.py $C --normals 0 --surface "free air (control)" \
  --note "free-air control after the index pad's 20 mm/s legs, same pose ($(basename $P))" --go || exit 6
run $PY s3_retreat.py --side -1 --away 10 --up 80 --go || exit 7
echo "$(date +%H:%M:%S) fast legs done"
