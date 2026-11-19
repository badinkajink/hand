#!/bin/bash
# 2026-11-19 12:3x (owner's go relayed by hand-4a, 12:0x: "we can go for the pla on tape tests"): the index's PLA side
# (the DIP link's flat side) on the taped plate. The hand is already at the approach's end (20261119-122743, turned -92
# deg, gap 14 mm, dz 15). S4(b) protocol with the yaw holding the normal force and stepping back to the clutch.
cd /home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab
echo "$(date +%H:%M:%S) index PLA protocol start"
/usr/bin/python3 s4b_drag.py --finger index --joint yaw --sign -1 --hold-load 500 --hold-travel 35 --clutch 2 --max-temp 40 \
  --protocol --normals 0.5,1,2 --speeds 0.2,2 --rests 1,10,60 --surface "PLA side on tape" \
  --note "PLA-side drag: the index's DIP link side on the taped plate, the yaw holding (sign -1), hand turned -92 deg (approach 20261119-122743, gap 14, dz 15)" \
  --go 2>&1 | grep -v Warning
echo "$(date +%H:%M:%S) index PLA protocol exit ${PIPESTATUS[0]}"
