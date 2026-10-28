"""Cable test of QUEUE.md R4: turns joint 6 alone from LO to HI deg and back CYCLES times, the other joints still, while
the HEX-E and the board are logged with the program's readers. Run it with the owner watching the cables.

The arm stops (stopJ) at the first HEX-E gap over 0.1 s, a force change over --stop-N from the start (the payload's
weight turning gives up to about 3.6 N), a protective or emergency stop, or a board gap over 1 s. Before moving it
checks that the flange is at least 300 mm above the base and the pad's contact point at least 200 mm, so the turning
tool (about 50 mm in radius) clears the indenters (tops at z 113 to 117 mm). The joint speed is set from the speed
slider so that the joint turns at --deg-s. Writes ft.bin and board.bin (the program's formats), robot.npy (t, q0..q5
rad) and summary.json into OUT. Stop the sensor recorder first: one reader on the DAQ and on the board's port.

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/turn_joint6.py OUT [--lo -180] [--hi 170]
        [--cycles 3] [--deg-s 10] [--stop-N 8]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration as U  # noqa: E402

ROBOT, FT = "192.168.0.4", "192.168.0.6"
BOARD = "/dev/serial/by-id/usb-Adafruit_QT_Py_ESP32-S3__4MB_Flash_2MB_PSRAM__b4:3a:45:b0:9c:d0-if00"
TCP_E = np.array([-0.40, -10.54, 128.36])          # the pad's contact point in the flange frame, mm (lab/README.md)

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--lo", type=float, default=-180.0)
ap.add_argument("--hi", type=float, default=170.0)
ap.add_argument("--cycles", type=int, default=3)
ap.add_argument("--deg-s", type=float, default=10.0)
ap.add_argument("--stop-N", type=float, default=8.0)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

import rtde_control  # noqa: E402
import rtde_receive  # noqa: E402

rr = rtde_receive.RTDEReceiveInterface(ROBOT)
if rr.getRobotMode() != 7 or rr.getSafetyMode() != 1:
    raise SystemExit(f"robot mode {rr.getRobotMode()}, safety mode {rr.getSafetyMode()}: not moving")
q0 = np.array(rr.getActualQ())
p = rr.getActualTCPPose()                           # the RTDE TCP is the flange (the program sets it so)
R = U.rotvec_to_R(p[3:]) if hasattr(U, "rotvec_to_R") else None
T = U.ur_to_T(p)
pad = T[:3, :3] @ TCP_E + T[:3, 3]
print(f"flange at {np.round(T[:3, 3], 1)} mm, pad point at {np.round(pad, 1)} mm, joint 6 at {math.degrees(q0[5]):.1f} deg")
if T[2, 3] < 300.0 or pad[2] < 200.0:
    raise SystemExit("too low for a free turn of the tool: move up first")
slider = rr.getTargetSpeedFraction()
speed = min(0.6, math.radians(a.deg_s) / max(slider, 0.05))
print(f"speed slider {slider:.2f}: joint speed {speed:.3f} rad/s commanded, {math.degrees(speed * slider):.1f} deg/s")

ft = U.OptoForceStream(FT, rate=100, appender=U.Appender(out / "ft.bin"), bias=False, filt=3)
ft.configure()
ft.start()
board = U.BoardStream(BOARD, appender=U.Appender(out / "board.bin", U.BoardStream.DTYPE))
board.start()
time.sleep(1.5)
t_ft, w0 = ft.ring.latest()
if t_ft is None or time.time() - t_ft > 0.2:
    raise SystemExit("no HEX-E stream: not moving")

targets = [math.radians(a.lo)] + [math.radians(x) for _ in range(a.cycles) for x in (a.hi, a.lo)] + [q0[5]]
rc = rtde_control.RTDEControlInterface(ROBOT)
rows, why, t_start = [], None, time.time()
try:
    for k, q6 in enumerate(targets):
        q = q0.copy()
        q[5] = q6
        rc.moveJ(q.tolist(), speed, 0.5, True)
        time.sleep(0.05)
        while True:
            t = time.time()
            q_now = np.array(rr.getActualQ())
            rows.append([t, *q_now])
            t_f, w = ft.ring.latest()
            tb, _ = board.ring.latest()
            if t_f is None or t - t_f > 0.1:
                why = f"HEX-E gap at joint 6 {math.degrees(q_now[5]):.1f} deg"
            elif np.linalg.norm(w[:3] - w0[:3]) > a.stop_N:
                why = f"force change {np.linalg.norm(w[:3] - w0[:3]):.1f} N at joint 6 {math.degrees(q_now[5]):.1f} deg"
            elif rr.isProtectiveStopped() or rr.isEmergencyStopped():
                why = "protective or emergency stop"
            elif tb is None or t - tb > 1.0:
                why = f"board gap at joint 6 {math.degrees(q_now[5]):.1f} deg"
            if why:
                rc.stopJ(2.0)
                break
            if rc.getAsyncOperationProgress() < 0 and abs(q_now[5] - q6) < math.radians(0.5):
                break
            if t - t_start > 900:
                why = "over 15 min"
                rc.stopJ(2.0)
                break
            time.sleep(0.008)
        if why:
            break
        print(f"  reached {math.degrees(q6):+.1f} deg ({k + 1}/{len(targets)}), {time.time() - t_start:.0f} s", flush=True)
finally:
    try:
        rc.stopScript()
    except Exception:                                # noqa: BLE001
        pass
    ft.stop = True
    board.stop = True
    time.sleep(0.5)
    np.save(out / "robot.npy", np.array(rows))
    gaps = [list(g) for g in ft.gaps]
    summary = dict(stopped=why, lo=a.lo, hi=a.hi, cycles=a.cycles, deg_s=a.deg_s, q0_deg=np.degrees(q0).tolist(),
                   ft_samples=ft.n, ft_timeouts=ft.n_timeout, ft_gaps=gaps, seconds=time.time() - t_start)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("stopped: " + why if why else "passed: no HEX-E gap, no board gap", flush=True)
    print(f"HEX-E {ft.n} samples, {ft.n_timeout} timed out, {len(gaps)} gaps over 0.5 s", flush=True)
