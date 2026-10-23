"""Logs the UR5's joint temperatures (deg C) and joints every PERIOD s over RTDE receive, read-only, as JSON lines into
OUT, for SECONDS: to put the reference presses' drift against the arm's warm-up (workstation, 2026-10-22: "UR arms grow a
few tenths of a mm in their first hour"). Moves nothing.

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/log_joint_temps.py OUT [--period 30] [--seconds 14400]
"""
import argparse
import json
import time

import rtde_receive

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--period", type=float, default=30.0)
ap.add_argument("--seconds", type=float, default=14400.0)
a = ap.parse_args()
r = rtde_receive.RTDEReceiveInterface("192.168.0.4", frequency=10.0,
                                      variables=["timestamp", "actual_q", "joint_temperatures", "robot_mode"])
t0 = time.time()
with open(a.out, "a") as f:
    while time.time() - t0 < a.seconds:
        try:
            rec = dict(t=time.time(), temps_C=list(r.getJointTemperatures()), q=list(r.getActualQ()), mode=r.getRobotMode())
            f.write(json.dumps(rec) + "\n")
            f.flush()
        except Exception as e:                    # noqa: BLE001
            f.write(json.dumps(dict(t=time.time(), error=str(e))) + "\n")
            f.flush()
        time.sleep(a.period)
