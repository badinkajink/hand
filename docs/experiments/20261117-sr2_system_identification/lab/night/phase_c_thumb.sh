#!/bin/bash
# Night of 2026-11-18, phase C (thumb, tag plate: its pad faces the tag's side; the taped plate needs a 180 deg turn):
# back off the middle's pose, plan (8 mm gap), approach, S4(b) friction protocol, three MCP and three PIP presses
# (S3; no block tracking, the thumb hides the tag), then the thumb's board angle presses at PIP 15 and 30 deg.
cd /home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab
PY=/usr/bin/python3
L=/home/will/workspace/hand/logs/20261118-sr2_s3_press
run() { "$@" 2>&1 | grep -v Warning | grep -v "^HEX-E before\|diagnostics"; return ${PIPESTATUS[0]}; }
echo "$(date +%H:%M:%S) phase C start"
run $PY s3_retreat.py --side -1 --away 10 --up 80 --go || { echo "retreat failed"; exit 1; }
run $PY s3_approach.py plan --finger thumb --gap 8 --dz 5 || { echo "thumb plan not clear"; exit 1; }
P=$(ls -t $L/*_approach_thumb.json | head -1); echo "plan $P"
run $PY s3_approach.py go --plan $P --stop-n 5 --stop-down 3 --go || { echo "approach failed or stopped"; exit 1; }
run $PY s4b_drag.py --finger thumb --protocol --note "night protocol, thumb pad on the tag plate ($(basename $P))" --go || exit 2
for j in mcp pip; do
  for i in 1 2 3; do
    sleep 60
    run $PY s3_press.py run --finger thumb --joint $j --sign 1 --max-travel 45 --note "night: thumb $j press $i of 3 into the tag plate" --go || exit 3
  done
done
for pip in 15 30; do
  run $PY s3_retreat.py --side 1 --away 10 --up 80 --go || { echo "retreat failed"; exit 4; }
  run $PY s3_press.py pose --finger thumb --pip $pip --go || exit 4
  run $PY s3_approach.py plan --finger thumb --lead ball --gap 10 --dz 5 || { echo "thumb pip $pip plan not clear"; exit 4; }
  P=$(ls -t $L/*_approach_thumb.json | head -1); echo "plan $P"
  run $PY s3_approach.py go --plan $P --stop-n 5 --stop-down 3 --go || { echo "approach failed or stopped"; exit 4; }
  run $PY s3_press.py run --finger thumb --joint mcp --sign 1 --step 1.0 --dwell 1.0 --max-n 2.0 --note "night: thumb board angle press, PIP pre-bent $pip deg ($(basename $P))" --go || exit 4
done
run $PY s3_retreat.py --side 1 --away 10 --up 80 --go || exit 5
run $PY s3_press.py pose --finger thumb --pip 0 --go || exit 5
echo "$(date +%H:%M:%S) phase C done"
