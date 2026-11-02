#!/usr/bin/env python3
"""Move one joint to an absolute (zero-relative) angle, then turn torque off and check
whether friction holds it there.

    FINGER=1 JOINT=fe1 TARGET=50 python3 move_joint_and_release.py   (on the CB1, in the host dir)

Safety: goal = current position before torque on; target bounds-checked by sync_set_joints;
wall-clock backstop; abort if the target isn't reached in TRACK_TIMEOUT_S; torque off in finally.
"""
from __future__ import annotations

import glob
import math
import os
import threading
import time

from manta_hand.servos import FINGER_JOINTS, TORQUE_OFF, ServoBus

FINGER = int(os.environ.get("FINGER", 1))
JOINT = os.environ.get("JOINT", "fe1")
TARGET = float(os.environ["TARGET"])
SPEED = int(os.environ.get("SPEED", 40))
BACKSTOP_S = 15.0
TRACK_TIMEOUT_S = 8.0
REACHED_DEG = 1.5

_ports = sorted(glob.glob("/dev/ttyUSB*"))
PORT = os.environ.get("SERVO_PORT") or (_ports[0] if len(_ports) == 1 else "/dev/ttyUSB0")
sid, zero, (lo, hi) = FINGER_JOINTS[FINGER][JOINT]
bus = ServoBus(PORT)
s, c = bus.servo(sid), bus.controller
done: list[str] = []


def torque_off(reason: str) -> None:
    if done:
        return
    done.append(reason)
    try:
        s._call(c.write_torque_enable, TORQUE_OFF)
    except Exception:  # noqa: BLE001
        s.disable()
    print(f"TORQUE OFF: {reason}", flush=True)


def rel() -> float:
    return math.degrees(s._call(c.read_present_position)[0]) - zero


def main() -> int:
    if not lo <= TARGET <= hi:
        raise ValueError(f"target {TARGET} outside [{lo}, {hi}] -- nothing moved")
    pos0 = s._call(c.read_present_position)[0]
    print(f"servo {sid} finger {FINGER} {JOINT}: {math.degrees(pos0) - zero:.2f} -> {TARGET:.2f} deg "
          f"(speed {SPEED}), temp {s._call(c.read_present_temperature)[0]} C", flush=True)
    s._call(c.write_goal_position, pos0)
    s.set_speed(SPEED)
    backstop = threading.Timer(BACKSTOP_S, torque_off, args=("wall-clock backstop",))
    backstop.daemon = True
    backstop.start()
    try:
        s.enable()
        bus.sync_set_joints({FINGER: {JOINT: TARGET}}, speed=SPEED)
        t = time.monotonic()
        while abs((p := rel()) - TARGET) > REACHED_DEG:
            if done:
                raise RuntimeError("backstop fired")
            if time.monotonic() - t > TRACK_TIMEOUT_S:
                raise RuntimeError(f"did not reach {TARGET} (at {p:.2f}) in {TRACK_TIMEOUT_S}s")
        time.sleep(0.5)
        print(f"reached {rel():.2f} deg in {time.monotonic() - t:.2f}s, "
              f"load {s._call(c.read_present_load)[0]}", flush=True)
    finally:
        backstop.cancel()
        torque_off("end of move")
    t_rel = time.monotonic()
    for _ in range(4):
        time.sleep(1.0)
        print(f"  {time.monotonic() - t_rel:.1f}s after release: {rel():.2f} deg", flush=True)
    print(f"bus timeouts {bus.timeouts}/{bus.transactions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
