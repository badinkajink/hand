#!/usr/bin/env python3
"""Joint zero check: move all 9 joints to one pose, then torque off (the gearbox holds the pose by
friction) and print the encoder angles, so the pose can be photographed and compared with the
simulation's geometry (sim zero = every finger straight, perpendicular to the palm, bending in the
pinch direction).

    POSE="0,0,0, 0,0,0, 0,0,0" SERVO_PORT=/dev/ttyUSB1 python3 zero_check.py   (on the CB1, host dir)

POSE = thumb yaw,mcp,pip, index yaw,mcp,pip, middle yaw,mcp,pip (deg, driver frame = raw - zero).
Safety: goals = current positions before torque on; targets bounds-checked by sync_set_joints;
|yaw target| <= 10; abort on |load| > 500, a joint not reaching its target in TRACK_TIMEOUT_S, or the
wall-clock backstop; torque off on all 9 in finally.
"""
from __future__ import annotations

import math
import os
import threading
import time

from manta_hand.servos import FINGER_JOINTS, TORQUE_OFF, ServoBus

POSE = [float(x) for x in os.environ["POSE"].replace(" ", "").split(",")]
SPEED = int(os.environ.get("SPEED", 40))
PORT = os.environ.get("SERVO_PORT", "/dev/ttyUSB1")
BACKSTOP_S = 20.0
TRACK_TIMEOUT_S = 8.0
REACHED_DEG = float(os.environ.get("REACHED", 1.5))
LOAD_ABORT = 500
RAMP_DPS = 10.0      # deg/s
RAMP_DT = 0.05

assert len(POSE) == 9, "POSE needs 9 values"
JOINTS = [(f, j) for f in (0, 1, 2) for j in ("aa", "fe1", "fe2")]
TARGET = dict(zip(JOINTS, POSE))
for (f, j), v in TARGET.items():
    lo, hi = FINGER_JOINTS[f][j][2]
    if not lo <= v <= hi:
        raise SystemExit(f"finger {f} {j}: {v} outside [{lo}, {hi}] -- nothing moved")
    if j == "aa" and abs(v) > 10:
        raise SystemExit(f"finger {f} aa: {v} beyond the +-10 deg adduction rule -- nothing moved")

bus = ServoBus(PORT)
c = bus.controller
SERVO = {k: bus.servo(FINGER_JOINTS[k[0]][k[1]][0]) for k in JOINTS}
ZERO = {k: FINGER_JOINTS[k[0]][k[1]][1] for k in JOINTS}
done: list[str] = []


def torque_off(reason: str) -> None:
    if done:
        return
    done.append(reason)
    for s in SERVO.values():
        try:
            s._call(c.write_torque_enable, TORQUE_OFF)
        except Exception:  # noqa: BLE001
            s.disable()
    print(f"TORQUE OFF (all 9): {reason}", flush=True)


def rel(k) -> float:
    return math.degrees(SERVO[k]._call(c.read_present_position)[0]) - ZERO[k]


def main() -> int:
    start = {k: rel(k) for k in JOINTS}
    moving = [k for k in JOINTS if abs(TARGET[k] - start[k]) > 0.3]
    print("start -> target (deg):", ", ".join(f"{f}{j} {start[(f, j)]:.1f}->{TARGET[(f, j)]:.1f}" for f, j in JOINTS),
          flush=True)
    for k in moving:
        s = SERVO[k]
        s._call(c.write_goal_position, s._call(c.read_present_position)[0])
        s.set_speed(SPEED)
    backstop = threading.Timer(BACKSTOP_S, torque_off, args=("wall-clock backstop",))
    backstop.daemon = True
    backstop.start()
    try:
        for k in moving:
            SERVO[k].enable()
        # The firmware steps straight to a new goal (a 10 deg step reads load ~520), so ramp the
        # goals in software at RAMP_DPS to keep the tracking error, and the load, small.
        span = max((abs(TARGET[k] - start[k]) for k in moving), default=0.0)
        n = max(1, int(math.ceil(span / RAMP_DPS / RAMP_DT)))
        for i in range(1, n + 1):
            if done:
                raise RuntimeError("backstop fired")
            pose: dict = {}
            for f, j in moving:
                pose.setdefault(f, {})[j] = start[(f, j)] + (TARGET[(f, j)] - start[(f, j)]) * i / n
            bus.sync_set_joints(pose, speed=SPEED)
            for k in moving:
                ld = SERVO[k]._call(c.read_present_load)[0]
                if abs(ld) > LOAD_ABORT:
                    raise RuntimeError(f"load {ld} on {k} during ramp")
            time.sleep(RAMP_DT)
        t = time.monotonic()
        left = list(moving)
        while left:
            if done:
                raise RuntimeError("backstop fired")
            for k in list(left):
                ld = SERVO[k]._call(c.read_present_load)[0]
                if abs(ld) > LOAD_ABORT:
                    raise RuntimeError(f"load {ld} on {k}")
                if abs(rel(k) - TARGET[k]) <= REACHED_DEG:
                    left.remove(k)
            if time.monotonic() - t > TRACK_TIMEOUT_S:
                raise RuntimeError(f"not reached in {TRACK_TIMEOUT_S}s: " +
                                   ", ".join(f"{f}{j} at {rel((f, j)):.1f}" for f, j in left))
        time.sleep(0.6)
        print(f"reached in {time.monotonic() - t:.2f}s", flush=True)
    finally:
        backstop.cancel()
        torque_off("end of move")
    time.sleep(1.0)
    print("encoder after release (deg):")
    for f, name in ((0, "thumb"), (1, "index"), (2, "middle")):
        print(f"  {name:6s} yaw {rel((f, 'aa')):6.2f}  mcp {rel((f, 'fe1')):6.2f}  pip {rel((f, 'fe2')):6.2f}")
    print(f"bus timeouts {bus.timeouts}/{bus.transactions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
