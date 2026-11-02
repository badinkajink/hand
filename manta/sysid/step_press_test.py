#!/usr/bin/env python3
"""Second motion test: index fe1 step out and back, sampled THROUGH the motion, plus a
hand-press check on present_load.

Differences from single_joint_step_test.py:
  * goals go out via ServoBus.sync_set_joints (one sync write, 2 ms read gap) instead of
    the individual write path, whose 0.3 s quiet period hid the whole move last time
  * logs present_load AND read_raw_present_load, to tell "load is really 0" from
    "the conversion is broken"
  * a PRESS_HOLD_S hold at +10 deg: the operator presses the fingertip during it

Safety (same rules as before):
  * goal = current position before torque is enabled (no jump)
  * targets bounds-checked by sync_set_joints against FINGER_JOINTS before any write
  * wall-clock backstop turns torque OFF after BACKSTOP_S no matter what
  * abort if a target isn't reached within TRACK_TIMEOUT_S, or temp > TEMP_ABORT_C
  * torque OFF in `finally`

Runs on the CB1:  cd ~/hand/src/morphohand/driver/manta/host && python3 - < this_file
"""
from __future__ import annotations

import math
import threading
import time

from manta_hand.servos import FINGER_JOINTS, TORQUE_OFF, ServoBus

FINGER, JOINT = 1, "fe1"
STEP_REL_DEG = 10.0
SPEED = 80
BACKSTOP_S = 15.0
TRACK_TIMEOUT_S = 3.0
REACHED_DEG = 1.0
PRESS_HOLD_S = 6.0
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
    raw_load = s._call(c.read_raw_present_load)[0]
    rows.append((round(time.monotonic() - t0, 4), phase, round(target_rel, 3),
                 round(rel(pos), 3), load, raw_load))
    return rel(pos)


def hold(phase: str, target_rel: float, seconds: float) -> None:
    t = time.monotonic()
    while time.monotonic() - t < seconds:
        sample(phase, target_rel)


def go_to(target_rel: float, phase: str) -> None:
    bus.sync_set_joints({FINGER: {JOINT: target_rel}}, speed=SPEED)  # bounds-checked
    t_start = time.monotonic()
    while True:
        p = sample(phase, target_rel)
        if abs(p - target_rel) < REACHED_DEG:
            break
        if time.monotonic() - t_start > TRACK_TIMEOUT_S:
            raise RuntimeError(f"{phase}: did not reach {target_rel:.1f} deg "
                               f"(at {p:.1f}) within {TRACK_TIMEOUT_S}s")
    hold(phase + "_settle", target_rel, 0.3)   # keep logging the settle / overshoot


def main() -> int:
    temp0 = s._call(c.read_present_temperature)[0]
    pos0 = s._call(c.read_present_position)[0]
    start_rel = rel(pos0)
    target = start_rel + STEP_REL_DEG
    print(f"servo {sid} (index {JOINT}): start {start_rel:.2f} deg, target {target:.2f} deg, "
          f"limits [{lo}, {hi}], temp {temp0} C", flush=True)
    if not lo <= target <= hi:
        raise ValueError("target outside software limits -- nothing moved")

    s._call(c.write_goal_position, pos0)      # hold-in-place goal BEFORE torque on
    s.set_speed(SPEED)
    backstop = threading.Timer(BACKSTOP_S, torque_off, args=("wall-clock backstop",))
    backstop.daemon = True
    backstop.start()
    try:
        s.enable()
        hold("pre_hold", start_rel, 0.5)
        go_to(target, "step_out")
        print(">>> PRESS NOW (6 s hold)", flush=True)
        hold("press_hold", target, PRESS_HOLD_S)
        go_to(start_rel, "step_back")
        hold("end_hold", start_rel, HOLD_S)
        temp1 = s._call(c.read_present_temperature)[0]
        print(f"done; temp {temp0} -> {temp1} C", flush=True)
        if temp1 > TEMP_ABORT_C:
            print("!! temperature above abort threshold", flush=True)
    finally:
        backstop.cancel()
        torque_off("end of test")
        print(f"bus timeouts {bus.timeouts}/{bus.transactions}", flush=True)
        print("CSV_BEGIN")
        print("t_s,phase,target_rel_deg,pos_rel_deg,load,raw_load")
        for r in rows:
            print(",".join(str(x) for x in r))
        print("CSV_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
