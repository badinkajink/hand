#!/bin/bash
# Night of 2026-11-18, phase A (index, taped plate): approach, S4(b) friction protocol, three PIP presses (S3).
# Stops at the first failure; the arm then holds where it is.
cd /home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab
PY=/usr/bin/python3
echo "$(date +%H:%M:%S) phase A start"
$PY s3_approach.py go --plan /home/will/workspace/hand/logs/20261118-sr2_s3_press/20261118-191629_approach_index.json --stop-n 5 --stop-down 3 --go 2>&1 | grep -v Warning
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "approach failed or stopped"; exit 1; }
$PY s4b_drag.py --finger index --protocol --note "night protocol, index pad on the taped plate (approach 20261118-191629)" --go 2>&1 | grep -v Warning
[ ${PIPESTATUS[0]} -eq 0 ] || exit 2
for i in 1 2 3; do
  sleep 60
  $PY s3_press.py run --finger index --joint pip --sign 1 --max-travel 35 --track-block --note "night: index PIP press $i of 3 into the taped plate" --go 2>&1 | grep -v Warning | grep -v "^HEX-E before\|diagnostics"
  [ ${PIPESTATUS[0]} -eq 0 ] || exit 3
done
echo "$(date +%H:%M:%S) phase A done"
