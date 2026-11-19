#!/bin/bash
# 2026-11-19 ~14:57: more 20 mm/s legs on the index pad (hand-4a: a 20 mm/s plateau lasts about 0.1 s, so more cycles
# tighten it). The 2 N level of 20261119-145241 stopped on its first 2 mm/s down leg: the normal force rose from 2.2 to
# 3.27 N and the 3 N guard caught it, before any 20 mm/s leg. Here: a hold of 1.5 N, five up and down pairs at 20 mm/s
# after 1 s rests, the arm at 0.5 m/s^2, then a free-air control of the same legs. Same pose (phase A's, re-planned).
cd /home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab
PY=/usr/bin/python3
L=/home/will/workspace/hand/logs/20261119-sr2_s3_press
run() { "$@" 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -v "^HEX-E before\|diagnostics\|^   drag"; return ${PIPESTATUS[0]}; }
echo "$(date +%H:%M:%S) more fast legs start"
run $PY day/wait_cool.py index || exit 1
run $PY s3_approach.py plan --finger index --gap 6 --dz 5 || { echo "plan not clear"; exit 2; }
P=$(ls -t $L/*_approach_index.json | head -1); echo "plan $P"
run $PY s3_approach.py go --plan $P --stop-n 5 --stop-down 3 --go || { echo "approach failed or stopped"; exit 3; }
C="--finger index --protocol --speeds 20 --rests 1,1,1,1,1 --acc 0.5 --max-temp 40"
run $PY s4b_drag.py $C --normals 1.5 --surface "pad on tape" \
  --note "index pad on the tape at 20 mm/s, five pairs, hold 1.5 N, arm at 0.5 m/s^2, phase A's pose ($(basename $P))" --go || exit 4
run $PY s4b_drag.py $C --normals 0 --surface "free air (control)" \
  --note "free-air control after the index pad's extra 20 mm/s legs, same pose ($(basename $P))" --go || exit 5
run $PY s3_retreat.py --side -1 --away 10 --up 80 --go || exit 6
echo "$(date +%H:%M:%S) more fast legs done"
