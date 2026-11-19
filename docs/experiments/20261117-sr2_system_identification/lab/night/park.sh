#!/bin/bash
# Park: (already backed off and raised by phase C) up to z_B 357 and level over the session's start, unturned.
cd /home/will/workspace/hand/docs/experiments/20261117-sr2_system_identification/lab
/usr/bin/python3 - <<'PY' 2>&1 | grep -v Warning
import sys, json
import numpy as np
sys.argv = ["x"]
import s3_press as S3
import s3_approach as SA
FT, HS, SL = S3.imports()
start = np.array(json.load(open("/home/will/workspace/hand/logs/20261118-sr2_s3_press/20261118-155238_approach_index.json"))["T_BP0"])
arm = HS.Arm(False, 0.4)
try:
    arm.connect(True)
    T_BP = arm.T_BP()
    if T_BP[2, 3] < 356.0:
        why, _ = arm.move(SA.shifted(T_BP, [0, 0, 356.7 - T_BP[2, 3]]) @ np.linalg.inv(HS.T_EP), 0.012, 0.05, 5.0, 20.0, "park: up")
        if why: raise SystemExit(why)
    T_BP = arm.T_BP()
    d = start[:3, 3] - T_BP[:3, 3]; d[2] = 0.0
    why, _ = arm.move(SA.shifted(T_BP, d) @ np.linalg.inv(HS.T_EP), 0.015, 0.05, 5.0, float(np.linalg.norm(d)) / 15.0 + 6.0, "park: level over the start")
    print("parked: palm origin B %s; stopped: %s" % (np.round(arm.T_BP()[:3, 3], 1).tolist(), why), flush=True)
finally:
    arm.close()
PY
