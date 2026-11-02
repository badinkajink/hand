#!/usr/bin/env python3
"""Full-range friction sweep: each joint in turn, one at a time, ramps slowly from near its low
software limit to near its high limit and back, while position and load are logged.

Comparing load on the way up and on the way down at the same angle separates
  friction ~ (load_up - load_down) / 2      and      gravity / bias ~ (load_up + load_down) / 2.

Only ONE joint has torque on at a time; every other servo is left as it is (torque off).

    JOINTS="1:fe1" RATE=2 python3 full_range_sweep.py        (on the CB1, in the host dir)
    default JOINTS = all nine

Range per joint: [lo + MARGIN, hi - MARGIN] from the software limits, except adduction (aa)
joints, which are clamped to +-AA_MAX (default 10 deg) until their real limits are confirmed.

Legs: start -> low end (to_low), low -> high (up), high -> low (down), low -> start (back).

Safety, per joint:
  * goal = current position before torque on (no jump)
  * every target is inside the software limits by MARGIN; sync_set_joints also bounds-checks
  * collision / jam guard: abort if |target - position| > JAM_DEG for JAM_S
  * abort if |load| > LOAD_ABORT for ABORT_HOLD_S, or temperature > TEMP_ABORT_C
  * wall-clock backstop per joint turns that servo's torque OFF after BACKSTOP_S
  * torque OFF for that servo in `finally`, before the next joint starts
Prints one CSV block per joint between CSV_BEGIN <finger>:<joint> / CSV_END.
"""
from __future__ import annotations

import glob
import math
import os
import threading
import time

from manta_hand.servos import FINGER_JOINTS, TORQUE_OFF, ServoBus

RATE = float(os.environ.get("RATE", 2.0))      # deg/s target ramp rate
MARGIN = float(os.environ.get("MARGIN", 3.0))  # deg kept inside each software limit
AA_MAX = float(os.environ.get("AA_MAX", 10.0))  # deg, adduction clamp
CMD_DT = 0.05                                  # target update period (20 Hz)
SPEED = 80
JAM_DEG = float(os.environ.get("JAM_DEG", 5.0))  # middle fe1's tight spot lags up to ~6 deg
JAM_S = 0.6
LOAD_ABORT, ABORT_HOLD_S = 500, 0.3
TEMP_ABORT_C = 45.0
SETTLE_S = 1.0

_ports = sorted(glob.glob("/dev/ttyUSB*"))
PORT = os.environ.get("SERVO_PORT") or (_ports[0] if len(_ports) == 1 else "/dev/ttyUSB0")
ALL = [(f, j) for f, js in FINGER_JOINTS.items() for j in js]
JOINTS = ([(int(x.split(":")[0]), x.split(":")[1]) for x in os.environ["JOINTS"].split(",")]
          if os.environ.get("JOINTS") else ALL)
bus = ServoBus(PORT)
c = bus.controller


def sweep(finger: int, joint: str) -> None:
    sid, zero, (lo, hi) = FINGER_JOINTS[finger][joint]
    s = bus.servo(sid)
    rel = lambda: math.degrees(s._call(c.read_present_position)[0]) - zero  # noqa: E731
    off: list[str] = []

    def torque_off(reason: str) -> None:
        if off:
            return
        off.append(reason)
        try:
            s._call(c.write_torque_enable, TORQUE_OFF)
        except Exception:  # noqa: BLE001
            s.disable()
        print(f"  TORQUE OFF ({reason})", flush=True)

    pos0 = s._call(c.read_present_position)[0]
    start = math.degrees(pos0) - zero
    low, high = lo + MARGIN, hi - MARGIN
    if joint == "aa":
        low, high = max(low, -AA_MAX), min(high, AA_MAX)
    if not low < high:
        print(f"finger {finger} {joint}: empty range [{low}, {high}], skipped", flush=True)
        return
    start_c = min(max(start, low), high)          # where we return to at the end
    legs = [start, low, high, low, start_c]
    path_len = sum(abs(b - a) for a, b in zip(legs, legs[1:]))
    backstop_s = path_len / RATE + 4 * SETTLE_S + 10.0
    print(f"finger {finger} {joint} (servo {sid}): start {start:.2f}, sweep [{low:.2f}, {high:.2f}], "
          f"limits [{lo}, {hi}], ~{path_len / RATE:.0f}s, backstop {backstop_s:.0f}s", flush=True)
    rows: list[tuple] = []
    t0 = time.monotonic()
    jam_since = over_since = None

    def sample(phase: str, target: float) -> None:
        nonlocal jam_since, over_since
        if off:
            raise RuntimeError(f"torque already off ({off[0]})")
        p = rel()
        load = s._call(c.read_present_load)[0]
        now = time.monotonic()
        rows.append((round(now - t0, 4), phase, round(target, 3), round(p, 3), load))
        if abs(target - p) > JAM_DEG:
            jam_since = jam_since or now
            if now - jam_since > JAM_S:
                raise RuntimeError(f"jam/collision guard: target {target:.2f}, at {p:.2f}")
        else:
            jam_since = None
        if abs(load) > LOAD_ABORT:
            over_since = over_since or now
            if now - over_since > ABORT_HOLD_S:
                raise RuntimeError(f"|load| {load} > {LOAD_ABORT}")
        else:
            over_since = None

    def ramp(a: float, b: float, phase: str) -> None:
        n = max(1, int(abs(b - a) / RATE / CMD_DT))
        for k in range(1, n + 1):
            target = a + (b - a) * k / n
            bus.sync_set_joints({finger: {joint: target}})
            t_next = time.monotonic() + CMD_DT
            while time.monotonic() < t_next:
                sample(phase, target)
        t_end = time.monotonic() + SETTLE_S
        while time.monotonic() < t_end:
            sample(phase + "_settle", b)

    s._call(c.write_goal_position, pos0)
    s.set_speed(SPEED)
    backstop = threading.Timer(backstop_s, torque_off, args=("wall-clock backstop",))
    backstop.daemon = True
    backstop.start()
    try:
        s.enable()
        bus.sync_set_joints({finger: {joint: start}}, speed=SPEED)   # make sure goal_speed is set
        for (a, b), name in zip(zip(legs, legs[1:]), ("to_low", "up", "down", "back")):
            if abs(b - a) > 1e-6:
                ramp(a, b, name)
            temp = s._call(c.read_present_temperature)[0]
            if temp > TEMP_ABORT_C:
                raise RuntimeError(f"temperature {temp} C")
    except RuntimeError as exc:
        print(f"  !! ABORT: {exc}", flush=True)
    finally:
        backstop.cancel()
        torque_off("end of joint")
    print(f"CSV_BEGIN {finger}:{joint}")
    print("t_s,phase,target_deg,pos_deg,load")
    for r in rows:
        print(",".join(str(x) for x in r))
    print("CSV_END", flush=True)


def main() -> int:
    for finger, joint in JOINTS:
        sweep(finger, joint)
        time.sleep(0.5)
    print(f"bus timeouts {bus.timeouts}/{bus.transactions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
