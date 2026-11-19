#!/bin/bash
# 2026-11-19 (owner's go, 14:0x-14:19, relayed by hand-4a: "you have my go ahead on whatever runs"): shear across the
# finger for the boards. Each pad head on at PIP 0, its MCP holding the pad on the plate; the arm drags the hand +5 mm,
# back, -5 mm, back along the plate's width (w) at 1 mm/s (s4b_drag.py cycles mode). A free-air control first (N 0),
# then 1 N, then 2 N. The index and middle on the taped plate (approach gaps 6 and 10, as phases A and B), the thumb on
# the tag plate (gap 8, as phase C, planned from the park). Waits under 40 C before each finger; stops at the first
# failure. Last night's z drags gave the shear along the finger.
cd /home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab
PY=/usr/bin/python3
L=/home/will/workspace/hand/logs/20261119-sr2_s3_press
PARK=/home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab/night/park.sh
run() { "$@" 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -v "^HEX-E before\|diagnostics\|^   drag"; return ${PIPESTATUS[0]}; }
echo "$(date +%H:%M:%S) shear across start"
for spec in ${SPECS:-index:6:-1 middle:10:-1 thumb:8:1}; do
  IFS=: read f gap side <<< "$spec"
  if [ "$f" = "thumb" ]; then bash $PARK || { echo "park failed"; exit 1; }; fi
  run $PY day/wait_cool.py $f || { echo "flags before $f"; exit 1; }
  run $PY s3_press.py pose --finger $f --pip 0 --go || exit 1
  run $PY s3_approach.py plan --finger $f --gap $gap --dz 5 || { echo "$f plan not clear"; exit 2; }
  P=$(ls -t $L/*_approach_$f.json | head -1); echo "plan $P"
  run $PY s3_approach.py go --plan $P --stop-n 5 --stop-down 3 --go || { echo "$f approach failed or stopped"; exit 3; }
  for n in 0 1 2; do
    run $PY s4b_drag.py --finger $f --normal $n --axis w --dist 5 --speed 1 --cycles 1 --max-temp 40 \
      --surface "$([ $f = thumb ] && echo 'pad on the tag plate' || echo 'pad on tape')$([ $n = 0 ] && echo ' (free-air control)')" \
      --note "shear across the finger for the boards: $f pad head on, MCP hold $n N, +-5 mm along the plate's width at 1 mm/s ($(basename $P))" \
      --go || { echo "$f drag $n failed"; exit 4; }
  done
  run $PY s3_retreat.py --side $side --away 10 --up 80 --go || { echo "retreat failed"; exit 5; }
done
bash $PARK || exit 6
echo "$(date +%H:%M:%S) shear across done"
