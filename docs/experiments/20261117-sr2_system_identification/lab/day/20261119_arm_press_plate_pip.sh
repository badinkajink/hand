#!/bin/bash
# 2026-11-19 17:2x, the PIP pre-bent (17:1x: at PIP 0 the plate meets the pad's flat proximal end, where tip 3 has no sphere presses; owner, 16:3x: "the contact area vs force sensing is really disconcerting ... you have my go ahead to
# execute plans to further our understanding of the existing models"; 16:4x: "you can calibrate more on the post"): the
# arm, not a finger joint, presses the index pad straight into the taped plate with its PIP pre-bent 10, 20 and 30 deg (the
# contact moves distally onto the calibrated part of the pad; --lead ball approach, gap 10) while the servos hold:
# 1 mm/s to 2.5 mm short of the plan's gap, then 0.2 mm/s to 2.5 N along the plate's normal, 2 s still, back; three
# times. The index and middle on the taped plate (approach gaps 6 and 10, as phases A and B), the thumb on the tag plate
# (gap 8, planned from the park). With the joints still, the own-pose term the trials subtract stays near zero, so the
# boards' field per newton is the pad's alone (the MCP presses of 11-18 read about half the calibration's). Waits under
# 40 C before each finger; stops at the first failure. 17:03: the first index press reached 0.97 N in 3 mm past the plan's gap: the
# MCP and PIP gave 0.6 and 0.9 deg (their servos pushing back at loads 60-75), so the press may run 8 mm past the gap.
cd /home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab
PY=/usr/bin/python3
L=/home/will/workspace/hand/logs/20261119-sr2_s3_press
PARK=/home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab/night/park.sh
run() { "$@" 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -v "^HEX-E before\|diagnostics"; return ${PIPESTATUS[0]}; }
echo "$(date +%H:%M:%S) arm presses on the plates start"
for spec in ${SPECS:-index:10:-1:10 index:10:-1:20 index:10:-1:30}; do
  IFS=: read f gap side pip <<< "$spec"
  if [ "$f" = "thumb" ]; then bash $PARK || { echo "park failed"; exit 1; }; fi
  run $PY day/wait_cool.py $f || { echo "flags before $f"; exit 1; }
  run $PY s3_press.py pose --finger $f --pip $pip --go || exit 1
  run $PY s3_approach.py plan --finger $f --lead ball --gap $gap --dz 5 || { echo "$f plan not clear"; exit 2; }
  P=$(ls -t $L/*_approach_$f.json | head -1); echo "plan $P"
  run $PY s3_approach.py go --plan $P --stop-n 5 --stop-down 3 --go || { echo "$f approach failed or stopped"; exit 3; }
  run $PY arm_press.py press --plan $P --target 2.5 --reps 3 --travel ${TRAVEL:-8} --gap-air ${GAPAIR:-1.5} \
    --indenter "$([ $f = thumb ] && echo 'tag plate face' || echo 'taped plate face'), PIP $pip" \
    --note "arm press: $f pad straight into the plate at PIP $pip, the finger's servos holding ($(basename $P))" --go \
    || { echo "$f press ended early"; run $PY s3_retreat.py --side $side --away 10 --up 80 --go; exit 4; }
  run $PY s3_retreat.py --side $side --away 10 --up 80 --go || { echo "retreat failed"; exit 5; }
done
bash $PARK || exit 6
echo "$(date +%H:%M:%S) arm presses on the plates done"
