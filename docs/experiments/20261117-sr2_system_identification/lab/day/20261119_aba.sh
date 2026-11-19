#!/bin/bash
# 2026-11-19 ~15:05 (owner's go for runs, 14:0x-14:19; hand-4a's A-B-A): is the middle pad's higher friction the finger
# or drift over a session? At phase A's pose today the index pad slid at 0.63 (2 mm/s, 1 N) against 0.40 last night;
# the middle pad read 0.62-0.74 last night, after the index. Index pad, middle pad, index pad, each at its own pose of
# phases A and B (flex, gaps 6 and 10, dz 5): a free-air control, then a hold of 1 N with three up and down pairs at
# 2 mm/s after 1 s rests. Stops at the first failure.
cd /home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab
PY=/usr/bin/python3
L=/home/will/workspace/hand/logs/20261119-sr2_s3_press
run() { "$@" 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -v "^HEX-E before\|diagnostics\|^   drag"; return ${PIPESTATUS[0]}; }
echo "$(date +%H:%M:%S) A-B-A start"
k=0
for spec in index:6 middle:10 index:6; do
  IFS=: read f gap <<< "$spec"; k=$((k+1))
  run $PY day/wait_cool.py $f || exit 1
  run $PY s3_approach.py plan --finger $f --gap $gap --dz 5 || { echo "$f plan not clear"; exit 2; }
  P=$(ls -t $L/*_approach_$f.json | head -1); echo "plan $P"
  run $PY s3_approach.py go --plan $P --stop-n 5 --stop-down 3 --go || { echo "$f approach failed or stopped"; exit 3; }
  C="--finger $f --protocol --speeds 2 --rests 1,1,1 --max-temp 40"
  run $PY s4b_drag.py $C --normals 0 --surface "free air (control)" \
    --note "A-B-A step $k of 3: free-air control, $f at its pose of phase $([ $f = index ] && echo A || echo B) ($(basename $P))" --go || exit 4
  run $PY s4b_drag.py $C --normals 1 --surface "pad on tape" \
    --note "A-B-A step $k of 3: $f pad on the tape, 1 N, 2 mm/s, three pairs ($(basename $P))" --go || exit 5
  run $PY s3_retreat.py --side -1 --away 10 --up 80 --go || exit 6
done
bash night/park.sh || exit 7
echo "$(date +%H:%M:%S) A-B-A done"
