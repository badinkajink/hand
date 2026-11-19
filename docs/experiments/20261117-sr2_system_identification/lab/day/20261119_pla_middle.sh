#!/bin/bash
# 2026-11-19 (owner's go relayed by hand-4a, 12:0x: "we can go for the pla on tape tests"): the middle's PLA side on the
# taped plate, after its approach (APPROACH, the plan's stem, as $1): a free-air control at the approach's gap, the S4(b)
# protocol with the yaw holding the normal force and stepping back to the clutch, and a second free-air control.
cd /home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab
A=${1:?approach plan stem}
run() { "$@" 2>&1 | grep -v Warning; return ${PIPESTATUS[0]}; }
echo "$(date +%H:%M:%S) middle: control before"
run /usr/bin/python3 s4b_drag.py --finger middle --joint yaw --sign -1 --hold-load 500 --max-temp 40 --protocol --normals 0 \
  --speeds 0.2,2 --rests 1,10 --surface "free air (control)" \
  --note "free-air control before the middle's PLA-side protocol, the middle off the taped plate at the approach's gap, hand turned -92 deg (approach $A)" \
  --go || { echo "control before failed"; exit 1; }
echo "$(date +%H:%M:%S) middle PLA protocol start"
run /usr/bin/python3 s4b_drag.py --finger middle --joint yaw --sign -1 --hold-load 500 --hold-travel 35 --clutch 2 --max-temp 40 \
  --protocol --normals 0.5,1,2 --speeds 0.2,2 --rests 1,10,60 --surface "PLA side on tape" \
  --note "PLA-side drag: the middle's DIP link side on the taped plate, the yaw holding (sign -1), hand turned -92 deg (approach $A)" \
  --go || { echo "protocol failed"; exit 2; }
echo "$(date +%H:%M:%S) middle: control after"
run /usr/bin/python3 s4b_drag.py --finger middle --joint yaw --sign -1 --hold-load 500 --max-temp 40 --protocol --normals 0 \
  --speeds 0.2,2 --rests 1,10 --surface "free air (control)" \
  --note "free-air control after the middle's PLA-side protocol, same pose (approach $A)" \
  --go || { echo "control after failed"; exit 3; }
echo "$(date +%H:%M:%S) middle done"
