"""Teach mode (freedrive) for the hand homing, at the owner's request (2026-10-22: "can you toggle teach mode on so i can
do this with both hands"). The same ur_rtde teachMode() the calibration program uses in its homing. Holds it until the
file logs/ur5_lab_stop_teach appears, or 15 min have passed, then ends teach mode and stops the ur_rtde script; the
arm then holds where it is. Nothing else is sent: no motion, payload or TCP.

    nohup setsid /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/teach_hold.py >> logs/ur5_lab_teach.log 2>&1 &
    touch logs/ur5_lab_stop_teach               # ends it
"""
import time
from pathlib import Path

import rtde_control

HERE = Path(__file__).resolve().parent
STOP = HERE.parents[3] / "logs/ur5_lab_stop_teach"          # logs/ is gitignored
STOP.unlink(missing_ok=True)
c = rtde_control.RTDEControlInterface("192.168.0.4")
c.teachMode()
print(time.strftime("%H:%M:%S"), "teach mode on", flush=True)
t_end = time.time() + 15 * 60
try:
    while time.time() < t_end and not STOP.exists():
        time.sleep(0.2)
finally:
    c.endTeachMode()
    c.stopScript()
    c.disconnect()
    STOP.unlink(missing_ok=True)
    print(time.strftime("%H:%M:%S"), "teach mode off", "(15 min cap)" if time.time() >= t_end else "", flush=True)
