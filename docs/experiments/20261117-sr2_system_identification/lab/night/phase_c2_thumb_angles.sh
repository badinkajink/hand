#!/bin/bash
# Night of 2026-11-18, phase C (rest): the thumb's board angle presses at PIP 15 and 30 deg. Phase C stopped at 21:39
# because the hand on the tag's side kept the tag out of both cameras after the 80 mm rise; each plan now starts from
# the park over the session's start, where the cameras see the tag.
cd /home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab
PY=/usr/bin/python3
L=/home/will/workspace/hand/logs/20261118-sr2_s3_press
run() { "$@" 2>&1 | grep -v Warning | grep -v "^HEX-E before\|diagnostics"; return ${PIPESTATUS[0]}; }
echo "$(date +%H:%M:%S) phase C2 start"
for pip in 15 30; do
  bash /home/will/workspace/hand/logs/20261118-sr2_night/park.sh || { echo "park failed"; exit 1; }
  run $PY s3_press.py pose --finger thumb --pip $pip --go || exit 1
  run $PY s3_approach.py plan --finger thumb --lead ball --gap 10 --dz 5 || { echo "thumb pip $pip plan not clear"; exit 2; }
  P=$(ls -t $L/*_approach_thumb.json | head -1); echo "plan $P"
  run $PY s3_approach.py go --plan $P --stop-n 5 --stop-down 3 --go || { echo "approach failed or stopped"; exit 3; }
  run $PY s3_press.py run --finger thumb --joint mcp --sign 1 --step 1.0 --dwell 1.0 --max-n 2.0 --note "night: thumb board angle press, PIP pre-bent $pip deg ($(basename $P))" --go || exit 4
  run $PY s3_retreat.py --side 1 --away 10 --up 80 --go || { echo "retreat failed"; exit 5; }
done
bash /home/will/workspace/hand/logs/20261118-sr2_night/park.sh || exit 6
run $PY s3_press.py pose --finger thumb --pip 0 --go || exit 6
echo "$(date +%H:%M:%S) phase C2 done"
