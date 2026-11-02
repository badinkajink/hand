#!/usr/bin/env python3
"""Move the rig's SCS0009s to mid position (0 rad = raw 512) and hold them there while the horns are
fitted, so each link hangs plumb at the center of the servo's 300° readable window.

    python3 center_servos.py [--port /dev/cu.usbmodem...] [--ids 20 21] [--hold-s 300] [--stop-file PATH]

Per servo: goal = current position before torque on, then a software ramp to 0 rad at RATE deg/s.
Hold ends when the stop file appears or after --hold-s (wall-clock backstop timer, independent of the
loop). Aborts with torque off on temperature > 45 °C or |load| > LOAD_ABORT for LOAD_HOLD_S.
Torque off for every servo in `finally`.
"""
from __future__ import annotations

import argparse
import math
import os
import threading
import time

from rustypot import Scs0009PyController

TORQUE_ON, TORQUE_OFF = 1, 2          # manta_hand/servos.py
RATE_DEG_S = 30.0
STEP_S = 0.05
TEMP_ABORT_C = 45
LOAD_ABORT, LOAD_HOLD_S = 500, 3.0    # sustained, so pressing a horn on doesn't trip it


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/cu.usbmodem5B8E1150821")
    ap.add_argument("--ids", type=int, nargs="+", default=[20, 21])
    ap.add_argument("--hold-s", type=float, default=300.0)
    ap.add_argument("--stop-file", default="/tmp/center_servos.stop")
    a = ap.parse_args()
    if os.path.exists(a.stop_file):
        os.remove(a.stop_file)

    c = Scs0009PyController(serial_port=a.port, baudrate=1_000_000, timeout=0.2)
    time.sleep(0.5)
    lock = threading.Lock()
    off_reason: list[str] = []
    timeouts = [0]

    def call(f, *args):
        """Bus call with retries: the bus drops an occasional packet (the hand driver retries too)."""
        for attempt in range(4):
            try:
                with lock:
                    return f(*args)
            except Exception:  # noqa: BLE001
                timeouts[0] += 1
                if attempt == 3:
                    raise
                time.sleep(0.01)

    def all_off(reason: str) -> None:
        with lock:
            if not off_reason:
                off_reason.append(reason)
            for i in a.ids:
                for _ in range(3):
                    try:
                        c.write_torque_enable(i, TORQUE_OFF)
                        break
                    except Exception:  # noqa: BLE001
                        time.sleep(0.02)
        print(f"TORQUE OFF ({reason})", flush=True)

    def rd(f, i):
        return call(f, i)[0]

    ramp_s = max(abs(math.degrees(rd(c.read_present_position, i))) for i in a.ids) / RATE_DEG_S
    backstop = threading.Timer(ramp_s + a.hold_s + 10, all_off, args=("wall-clock backstop",))
    backstop.daemon = True
    backstop.start()
    try:
        start = {}
        for i in a.ids:
            p = rd(c.read_present_position, i)
            lo, hi = rd(c.read_min_angle_limit, i), rd(c.read_max_angle_limit, i)
            if not lo < 0.0 < hi:
                raise RuntimeError(f"id {i}: 0 rad outside its stored limits [{lo:.2f}, {hi:.2f}]")
            start[i] = p
            call(c.write_goal_position, i, p)
            call(c.write_torque_enable, i, TORQUE_ON)
            print(f"id {i}: torque on at {math.degrees(p):+.1f} deg, ramping to 0", flush=True)
        n = max(1, int(ramp_s / STEP_S))
        for k in range(1, n + 1):
            if off_reason:
                raise RuntimeError(off_reason[0])
            for i in a.ids:
                call(c.write_goal_position, i, start[i] * (1 - k / n))
            time.sleep(STEP_S)
        time.sleep(0.5)
        for i in a.ids:
            print(f"id {i}: at {math.degrees(rd(c.read_present_position, i)):+.2f} deg", flush=True)
        print(f"HOLDING at center. Fit the horns, then create {a.stop_file} (backstop {a.hold_s:.0f} s).",
              flush=True)
        t_end = time.monotonic() + a.hold_s
        over_since: dict[int, float | None] = {i: None for i in a.ids}
        last_print = 0.0
        while time.monotonic() < t_end and not os.path.exists(a.stop_file):
            if off_reason:
                raise RuntimeError(off_reason[0])
            now = time.monotonic()
            for i in a.ids:
                load, temp = rd(c.read_present_load, i), rd(c.read_present_temperature, i)
                if temp > TEMP_ABORT_C:
                    raise RuntimeError(f"id {i}: temperature {temp} C")
                if abs(load) > LOAD_ABORT:
                    over_since[i] = over_since[i] or now
                    if now - over_since[i] > LOAD_HOLD_S:
                        raise RuntimeError(f"id {i}: |load| {load} > {LOAD_ABORT} for {LOAD_HOLD_S} s")
                else:
                    over_since[i] = None
                if now - last_print > 15:
                    print(f"  id {i}: pos {math.degrees(rd(c.read_present_position, i)):+.2f} deg  "
                          f"load {load}  temp {temp} C", flush=True)
            if now - last_print > 15:
                last_print = now
            time.sleep(0.2)
    except Exception as exc:  # noqa: BLE001  (includes a bus timeout that survived the retries)
        print(f"!! ABORT: {exc}", flush=True)
    finally:
        backstop.cancel()
        all_off(off_reason[0] if off_reason else "end of hold")
        time.sleep(0.3)
        for i in a.ids:
            print(f"id {i}: final {math.degrees(rd(c.read_present_position, i)):+.2f} deg, "
                  f"torque_enable {rd(c.read_torque_enable, i)}", flush=True)
        print(f"bus retries: {timeouts[0]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
