"""Teach mode (freedrive) for the hand homing, at the owner's request (2026-10-22: "can you toggle teach mode on so i can
do this with both hands"). The same ur_rtde teachMode() the calibration program uses in its homing. Holds it until the
file logs/ur5_lab_stop_teach appears, or --minutes have passed (15 by default), then ends teach mode and stops the ur_rtde
script; the arm then holds where it is. Nothing else is sent: no motion, payload or TCP.

Touching logs/ur5_lab_teach_extend restarts the --minutes from that moment without leaving teach mode. Extend it, or stop
it on the owner's word; never let the cap run out, or restart it, while the owner may hold the arm: on 2026-11-02 the
15 min cap ended teach mode under the owner's hands and the robot e-stopped.

    nohup setsid /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/teach_hold.py [--minutes 15] >> logs/ur5_lab_teach.log 2>&1 &
    touch logs/ur5_lab_teach_extend             # another --minutes from now
    touch logs/ur5_lab_stop_teach               # ends it
"""
import argparse
import time
from pathlib import Path

import rtde_control

HERE = Path(__file__).resolve().parent
STOP = HERE.parents[3] / "logs/ur5_lab_stop_teach"          # logs/ is gitignored
EXTEND = HERE.parents[3] / "logs/ur5_lab_teach_extend"
ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=15.0)
a = ap.parse_args()
STOP.unlink(missing_ok=True)
EXTEND.unlink(missing_ok=True)
c = rtde_control.RTDEControlInterface("192.168.0.4")
c.teachMode()
t_end = time.time() + a.minutes * 60
print(time.strftime("%H:%M:%S"), "teach mode on until", time.strftime("%H:%M:%S", time.localtime(t_end)), flush=True)
try:
    while time.time() < t_end and not STOP.exists():
        if EXTEND.exists():
            EXTEND.unlink(missing_ok=True)
            t_end = time.time() + a.minutes * 60
            print(time.strftime("%H:%M:%S"), "extended until", time.strftime("%H:%M:%S", time.localtime(t_end)), flush=True)
        time.sleep(0.2)
finally:
    c.endTeachMode()
    c.stopScript()
    c.disconnect()
    STOP.unlink(missing_ok=True)
    print(time.strftime("%H:%M:%S"), "teach mode off", f"({a.minutes:g} min cap)" if time.time() >= t_end else "", flush=True)
