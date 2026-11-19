#!/bin/bash
# 2026-11-19 14:3x (owner, 14:0x-14:19, relayed by hand-4a: "can we also run some magnetic fingertip calibrations? what
# else can we do with this before we take out this mount?", "you have my go ahead on whatever runs"): the boards' angle
# presses against the HEX-E. Each pad face on, the PIP pre-bent 15 and then 30 deg so the contact moves along the pad,
# the MCP pressing at 1 deg per s to the queue's 3 N stop; each with its own approach (--lead ball). The index and
# middle press the taped plate from its side; the thumb presses the tag plate, planned from the park over the start
# (the hand on the tag's side hides the tag). Waits under 40 C before each press; stops at the first failure.
# 14:28: the index's PIP 15 press (1 deg steps to 3 N) stuck at 1.95 N for six steps while the MCP's load built to 585,
# then slipped to 3.48 N, over the 3 N rule. The rest stop at 2.6 N or at load 450, in 0.5 deg steps every 0.5 s.
# SPECS (finger:pip:gap ...) picks the index and middle presses to run.
cd /home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab
PY=/usr/bin/python3
L=/home/will/workspace/hand/logs/20261119-sr2_s3_press
PARK=/home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab/night/park.sh
run() { "$@" 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -v "^HEX-E before\|diagnostics"; return ${PIPESTATUS[0]}; }
echo "$(date +%H:%M:%S) board presses start"
for spec in ${SPECS:-index:15:10 index:30:10 middle:15:10 middle:30:14}; do
  IFS=: read f pip gap <<< "$spec"
  run $PY day/wait_cool.py $f || { echo "flags before $f $pip"; exit 1; }
  run $PY s3_press.py pose --finger $f --pip $pip --go || { echo "pose $f $pip failed"; exit 1; }
  run $PY s3_approach.py plan --finger $f --lead ball --gap $gap --dz 5 || { echo "$f pip $pip plan not clear"; exit 2; }
  P=$(ls -t $L/*_approach_$f.json | head -1); echo "plan $P"
  run $PY s3_approach.py go --plan $P --stop-n 5 --stop-down 3 --go || { echo "$f pip $pip approach failed or stopped"; exit 3; }
  run $PY s3_press.py run --finger $f --joint mcp --sign 1 --step 0.5 --dwell 0.5 --max-n 2.6 --max-load 450 \
    --note "board angle press, $f pad face on the taped plate, PIP pre-bent $pip deg, to the 3 N stop ($(basename $P))" --go || exit 4
  run $PY s3_retreat.py --side -1 --away 10 --up 80 --go || { echo "retreat failed"; exit 5; }
  [ "$spec" = "index:30:10" ] && { run $PY s3_press.py pose --finger index --pip 0 --go || exit 6; }
done
run $PY s3_press.py pose --finger middle --pip 0 --go || exit 6
for pip in 15 30; do
  bash $PARK || { echo "park failed"; exit 7; }
  run $PY day/wait_cool.py thumb || { echo "flags before thumb $pip"; exit 1; }
  run $PY s3_press.py pose --finger thumb --pip $pip --go || exit 7
  run $PY s3_approach.py plan --finger thumb --lead ball --gap 10 --dz 5 || { echo "thumb pip $pip plan not clear"; exit 8; }
  P=$(ls -t $L/*_approach_thumb.json | head -1); echo "plan $P"
  run $PY s3_approach.py go --plan $P --stop-n 5 --stop-down 3 --go || { echo "thumb pip $pip approach failed or stopped"; exit 9; }
  run $PY s3_press.py run --finger thumb --joint mcp --sign 1 --step 0.5 --dwell 0.5 --max-n 2.6 --max-load 450 \
    --note "board angle press, thumb pad face on the tag plate, PIP pre-bent $pip deg, to the 3 N stop ($(basename $P))" --go || exit 10
  run $PY s3_retreat.py --side 1 --away 10 --up 80 --go || { echo "retreat failed"; exit 11; }
done
bash $PARK || exit 12
run $PY s3_press.py pose --finger thumb --pip 0 --go || exit 12
echo "$(date +%H:%M:%S) board presses done"
