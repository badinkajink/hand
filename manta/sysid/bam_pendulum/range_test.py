#!/usr/bin/env python3
"""Powered range-of-motion sweep on the double pendulum: joint 1 (servo 20) follows
0 -> +HI -> 0 -> -LO -> 0 deg at RATE deg/s while joint 2 (servo 21) holds 0 (link 2 in line with link 1).
0 deg = hanging plumb (both horns centered 2026-10-29, hanging reads within ~1 deg of 0).

    python3 range_test.py [--hi 74] [--lo 145] [--rate 20]

Safety: goal = current position before torque on; targets kept inside each servo's stored limits;
abort -> torque OFF on |target - position| > JAM_DEG for JAM_S, |load| > LOAD_ABORT for LOAD_S,
temperature > 45 C, or a position jump > JUMP_DEG between samples (sensor blind zone);
wall-clock backstop timer; torque off in `finally`. Prints a CSV log.
"""
from __future__ import annotations

import argparse
import math
import threading
import time

from rustypot import Scs0009PyController

PORT = "/dev/cu.usbmodem5B8E1150821"
J1, J2 = 20, 21
TORQUE_ON, TORQUE_OFF = 1, 2
STEP_S = 0.05
JAM_DEG, JAM_S = 8.0, 0.5
LOAD_ABORT, LOAD_S = 900, 0.5
TEMP_ABORT_C = 45
JUMP_DEG = 25.0
END_HOLD_S = 1.0
MARGIN_DEG = 2.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hi", type=float, default=74.0, help="joint 1 positive end, deg")
    ap.add_argument("--lo", type=float, default=145.0, help="joint 1 negative end (magnitude), deg")
    ap.add_argument("--rate", type=float, default=20.0, help="deg/s")
    ap.add_argument("--single", action="store_true", help="single pendulum: only servo 20, servo 21 untouched")
    ap.add_argument("--zero-deg", type=float, default=0.0, help="servo 20 reading when hanging plumb")
    a = ap.parse_args()

    c = Scs0009PyController(serial_port=PORT, baudrate=1_000_000, timeout=0.2)
    time.sleep(0.5)
    lock = threading.Lock()
    off: list[str] = []
    retries = [0]

    def call(f, *args):
        for attempt in range(4):
            try:
                with lock:
                    return f(*args)
            except Exception:  # noqa: BLE001
                retries[0] += 1
                if attempt == 3:
                    raise
                time.sleep(0.01)

    Z = {J1: a.zero_deg, J2: 0.0}
    JOINTS = (J1,) if a.single else (J1, J2)

    def deg(i):
        return math.degrees(call(c.read_present_position, i)[0]) - Z[i]

    def all_off(reason: str) -> None:
        if not off:
            off.append(reason)
        for i in JOINTS:
            for _ in range(4):
                try:
                    with lock:
                        c.write_torque_enable(i, TORQUE_OFF)
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(0.01)
        print(f"TORQUE OFF ({reason})", flush=True)

    lim = {i: [math.degrees(call(f, i)[0]) for f in (c.read_min_angle_limit, c.read_max_angle_limit)]
           for i in (J1, J2)}
    hi = min(a.hi, lim[J1][1] - a.zero_deg - MARGIN_DEG)
    lo = -min(a.lo, -(lim[J1][0] - a.zero_deg + MARGIN_DEG))
    if not (lim[J2][0] < 0 < lim[J2][1]):
        raise SystemExit("joint 2: 0 outside its stored limits")
    p1, p2 = deg(J1), (deg(J2) if not a.single else 0.0)
    legs = [p1, 0.0, hi, 0.0, lo, 0.0]
    path = sum(abs(y - x) for x, y in zip(legs, legs[1:]))
    backstop_s = path / a.rate + 4 * END_HOLD_S + 15
    print(f"joint 1 limits {lim[J1][0]:+.1f}/{lim[J1][1]:+.1f}; sweep {legs[1:]} deg; "
          f"start j1 {p1:+.2f} j2 {p2:+.2f}; ~{path / a.rate:.0f} s, backstop {backstop_s:.0f} s", flush=True)

    rows: list[tuple] = []
    t0 = time.monotonic()
    state = {"jam": None, "over": {J1: None, J2: None}, "last": {J1: p1, J2: p2}}

    def sample(phase: str, tgt1: float, tgt2: float) -> None:
        if off:
            raise RuntimeError(off[0])
        now = time.monotonic()
        q = {J1: deg(J1), J2: deg(J2) if not a.single else 0.0}
        load = {i: (call(c.read_present_load, i)[0] if i in JOINTS else 0) for i in (J1, J2)}
        rows.append((round(now - t0, 3), phase, round(tgt1, 2), round(q[J1], 2), load[J1],
                     round(tgt2, 2), round(q[J2], 2), load[J2]))
        for i in JOINTS:
            if abs(q[i] - state["last"][i]) > JUMP_DEG:
                raise RuntimeError(f"id {i}: position jumped {state['last'][i]:+.1f} -> {q[i]:+.1f} (blind zone?)")
            state["last"][i] = q[i]
            if abs(load[i]) > LOAD_ABORT:
                state["over"][i] = state["over"][i] or now
                if now - state["over"][i] > LOAD_S:
                    raise RuntimeError(f"id {i}: |load| {load[i]} > {LOAD_ABORT}")
            else:
                state["over"][i] = None
        err = max(abs(tgt1 - q[J1]), abs(tgt2 - q[J2]))
        if err > JAM_DEG:
            state["jam"] = state["jam"] or now
            if now - state["jam"] > JAM_S:
                raise RuntimeError(f"tracking error {err:.1f} deg for {JAM_S} s (jam/collision?)")
        else:
            state["jam"] = None

    def ramp(x: float, y: float, phase: str) -> None:
        n = max(1, int(abs(y - x) / a.rate / STEP_S))
        for k in range(1, n + 1):
            tgt = x + (y - x) * k / n
            call(c.write_goal_position, J1, math.radians(tgt + a.zero_deg))
            t_next = time.monotonic() + STEP_S
            while time.monotonic() < t_next:
                sample(phase, tgt, 0.0)
        t_end = time.monotonic() + END_HOLD_S
        while time.monotonic() < t_end:
            sample(phase + "_hold", y, 0.0)
        temps = [call(c.read_present_temperature, i)[0] for i in JOINTS]
        if max(temps) > TEMP_ABORT_C:
            raise RuntimeError(f"temperature {temps} C")

    backstop = threading.Timer(backstop_s, all_off, args=("wall-clock backstop",))
    backstop.daemon = True
    backstop.start()
    try:
        for i, p in ((J1, p1), (J2, p2)):
            if i in JOINTS:
                call(c.write_goal_position, i, math.radians(p + Z[i]))
                call(c.write_torque_enable, i, TORQUE_ON)
        if not a.single:
            call(c.write_goal_position, J2, 0.0)
        for (x, y), name in zip(zip(legs, legs[1:]), ("to_zero", "up_pos", "back", "up_neg", "home")):
            ramp(x, y, name)
        print("sweep complete", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"!! ABORT: {exc}", flush=True)
    finally:
        backstop.cancel()
        all_off(off[0] if off else "end of test")

    print("CSV_BEGIN\nt_s,phase,tgt1,q1,load1,tgt2,q2,load2")
    for r in rows:
        print(",".join(map(str, r)))
    print("CSV_END")
    if rows:
        q1 = [r[3] for r in rows]
        print(f"joint 1 reached {min(q1):+.1f} .. {max(q1):+.1f} deg; "
              f"max |load| j1 {max(abs(r[4]) for r in rows)}, j2 {max(abs(r[7]) for r in rows)}; "
              f"max |q2| {max(abs(r[6]) for r in rows):.1f} deg; bus retries {retries[0]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
