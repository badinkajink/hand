"""Read-only: the UR5's joints and TCP pose once a second to OUT (JSONL, host time), so the hand's pose in the lab is
known through the night and any motion of the arm shows. rtde_receive only; it sends the robot nothing.

    /usr/bin/python3 ur5_pose_log.py OUT [--period 1]
"""
import json
import sys
import time

import rtde_receive

out = open(sys.argv[1], "a", buffering=1)
period = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
r = None
while True:
    try:
        if r is None:
            r = rtde_receive.RTDEReceiveInterface("192.168.0.4", 10.0)
        out.write(json.dumps({"t": time.time(), "q": r.getActualQ(), "tcp": r.getActualTCPPose(),
                              "robot_mode": r.getRobotMode(), "safety_mode": r.getSafetyMode()}) + "\n")
    except Exception as e:
        out.write(json.dumps({"t": time.time(), "error": str(e)}) + "\n")
        try:
            r.disconnect()
        except Exception:
            pass
        r = None
        time.sleep(5.0)
    time.sleep(period)
