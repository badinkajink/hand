#!/bin/bash
# Night of 2026-11-18, phase B (middle, taped plate): back off the index's pose, plan (10 mm gap: the middle sits 5 mm
# closer than its model at PIP 0), approach, S4(b) friction protocol, three PIP presses (S3). Stops at the first failure.
cd /home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab
PY=/usr/bin/python3
L=/home/will/workspace/hand/logs/20261118-sr2_s3_press
run() { "$@" 2>&1 | grep -v Warning | grep -v "^HEX-E before\|diagnostics"; return ${PIPESTATUS[0]}; }
echo "$(date +%H:%M:%S) phase B start"
run $PY s3_retreat.py --side -1 --away 10 --up 80 --go || { echo "retreat failed"; exit 1; }
run $PY s3_approach.py plan --finger middle --gap 10 --dz 5 || { echo "middle plan not clear"; exit 1; }
P=$(ls -t $L/*_approach_middle.json | head -1); echo "plan $P"
run $PY s3_approach.py go --plan $P --stop-n 5 --stop-down 3 --go || { echo "approach failed or stopped"; exit 1; }
run $PY s4b_drag.py --finger middle --protocol --note "night protocol, middle pad on the taped plate ($(basename $P))" --go || exit 2
for i in 1 2 3; do
  sleep 60
  run $PY s3_press.py run --finger middle --joint pip --sign 1 --max-travel 45 --track-block --note "night: middle PIP press $i of 3 into the taped plate" --go || exit 3
done
echo "$(date +%H:%M:%S) phase B done"
