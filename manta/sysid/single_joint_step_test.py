#!/usr/bin/env python3
"""First motion test: one servo, small slow step out and back, logging position + load.

Default: index fe1 (servo 4), +0.9 -> +10 -> back to start (relative deg), speed 80.

Safety:
  * goal_position is set to the CURRENT position before torque is enabled (no jump)
  * target is checked against FINGER_JOINTS' software limits before anything moves
  * a wall-clock backstop turns torque OFF after BACKSTOP_S no matter what
  * abort (torque off) if the joint fails to reach its target within TRACK_TIMEOUT_S
    or temperature exceeds TEMP_ABORT_C
  * torque OFF in `finally`

Runs on the CB1:  cd ~/hand/src/morphohand/driver/manta/host && python3 - < this_file
Prints CSV between CSV_BEGIN / CSV_END.
"""
from __future__ import annotations

import math
import threading
import time

from manta_hand.servos import FINGER_JOINTS, TORQUE_OFF, ServoBus

FINGER, JOINT = 1, "fe1"
STEP_REL_DEG = 10.0
SPEED = 80
BACKSTOP_S = 10.0
TRACK_TIMEOUT_S = 3.0
REACHED_DEG = 1.0
HOLD_S = 1.0
TEMP_ABORT_C = 45.0

sid, zero_deg, (lo, hi) = FINGER_JOINTS[FINGER][JOINT]
bus = ServoBus("/dev/ttyUSB0")
s, c = bus.servo(sid), bus.controller
rows: list[tuple] = []
t0 = time.monotonic()
torque_off_reason: list[str] = []


def torque_off(reason: str) -> None:
    if torque_off_reason:
        return
    torque_off_reason.append(reason)
    try:
        s._call(c.write_torque_enable, TORQUE_OFF)
    except Exception as exc:  # noqa: BLE001
        print(f"!! torque-off write failed ({exc}); retrying via disable()", flush=True)
        s.disable()
    print(f"TORQUE OFF: {reason}", flush=True)


def rel(pos_rad: float) -> float:
    return math.degrees(pos_rad) - zero_deg


def sample(phase: str, target_rel: float) -> float:
    pos = s._call(c.read_present_position)[0]
    load = s._call(c.read_present_load)[0]
    rows.append((round(time.monotonic() - t0, 4), phase, round(target_rel, 3),
                 round(rel(pos), 3), load))
    return rel(pos)


def go_to(target_rel: float, phase: str) -> None:
    if not lo <= target_rel <= hi:
        raise ValueError(f"target {target_rel} outside software limits [{lo}, {hi}]")
    s._call(c.write_goal_position, math.radians(zero_deg + target_rel))
    t_start = time.monotonic()
    while True:
        p = sample(phase, target_rel)
        if abs(p - target_rel) < REACHED_DEG:
            break
        if time.monotonic() - t_start > TRACK_TIMEOUT_S:
            raise RuntimeError(f"{phase}: did not reach {target_rel:.1f} deg "
                               f"(at {p:.1f}) within {TRACK_TIMEOUT_S}s")
    t_hold = time.monotonic()
    while time.monotonic() - t_hold < HOLD_S:
        sample(phase + "_hold", target_rel)


def main() -> int:
    temp0 = s._call(c.read_present_temperature)[0]
    pos0 = s._call(c.read_present_position)[0]
    start_rel = rel(pos0)
    target = start_rel + STEP_REL_DEG
    print(f"servo {sid} ({['thumb','index','middle'][FINGER]} {JOINT}): start {start_rel:.2f} deg, "
          f"target {target:.2f} deg, limits [{lo}, {hi}], temp {temp0} C", flush=True)
    if not lo <= target <= hi:
        raise ValueError("target outside software limits -- nothing moved")

    s._call(c.write_goal_position, pos0)      # hold-in-place goal BEFORE torque on
    s.set_speed(SPEED)
    backstop = threading.Timer(BACKSTOP_S, torque_off, args=("wall-clock backstop",))
    backstop.daemon = True
    backstop.start()
    try:
        s.enable()
        t_hold = time.monotonic()
        while time.monotonic() - t_hold < 0.5:
            sample("pre_hold", start_rel)
        go_to(target, "step_out")
        go_to(start_rel, "step_back")
        temp1 = s._call(c.read_present_temperature)[0]
        print(f"done; temp {temp0} -> {temp1} C", flush=True)
        if temp1 > TEMP_ABORT_C:
            print("!! temperature above abort threshold", flush=True)
    finally:
        backstop.cancel()
        torque_off("end of test")
        print(f"bus timeouts {bus.timeouts}/{bus.transactions}", flush=True)
        print("CSV_BEGIN")
        print("t_s,phase,target_rel_deg,pos_rel_deg,load")
        for r in rows:
            print(",".join(str(x) for x in r))
        print("CSV_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
