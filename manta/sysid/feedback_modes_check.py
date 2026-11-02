#!/usr/bin/env python3
"""Hardware check of ServoBus.read_feedback (hand-main branch manta-fast-feedback-read).

All nine servos torque ON, each following a +-AMP deg 0.5 Hz sine around its pose. Per cycle:
one sync_set_joints write, then bus.read_feedback(mode). For each mode, free-running and paced at
50 Hz, DURATION s each. Also checks the speed sign: decoded speed should agree in sign with the
position change across moving samples.

    PYTHONPATH=/tmp/mh_fb python3 feedback_modes_check.py     (on the CB1)

Safety: goals start at the current pose; clamped >= MARGIN inside software limits; abort on
|load| > 500 for 0.3 s, > 5 deg tracking error for 0.6 s, or > 45 C; wall-clock backstop
disables all torque; disable_all in finally.
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import threading
import time

from manta_hand import servos
from manta_hand.servos import FINGER_JOINTS, ServoBus

AMP = float(os.environ.get("AMP", 2.0))
DURATION = float(os.environ.get("DURATION", 8.0))
SPEED, MARGIN = 80, 1.0
print("driver:", servos.__file__, flush=True)
port = sorted(glob.glob("/dev/ttyUSB*"))[0]
bus = ServoBus(port)
JOINTS = [(f, j) for f, js in FINGER_JOINTS.items() for j in js]


def main() -> int:
    off: list[str] = []

    def all_off(reason: str) -> None:
        if not off:
            off.append(reason)
            bus.disable_all()
            print(f"  ALL TORQUE OFF ({reason})", flush=True)

    start = bus.read_feedback("position")["position"]
    center = {(f, j): min(max(start[f][j], FINGER_JOINTS[f][j][2][0] + AMP + MARGIN),
                          FINGER_JOINTS[f][j][2][1] - AMP - MARGIN) for f, j in JOINTS}
    runs = [(m, hz) for m in ("position", "position_load", "all") for hz in (0.0, 50.0)]
    bt = threading.Timer(len(runs) * (DURATION + 1) + 10, all_off, args=("wall-clock backstop",))
    bt.daemon = True
    summary: dict = {"port": port, "runs": {}}
    try:
        bus.sync_set_joints({f: {j: center[(f, j)] for ff, j in JOINTS if ff == f}
                             for f in FINGER_JOINTS}, speed=SPEED)
        bt.start()
        bus.enable_all()
        time.sleep(0.5)
        for mode, hz in runs:
            t_tx = (bus.transactions, bus.timeouts)
            t0 = time.monotonic()
            nxt, misses, stamps = t0, 0, []
            prev = None
            sign_ok = sign_n = 0
            over = jam = None
            while time.monotonic() - t0 < DURATION:
                if off:
                    raise RuntimeError(f"torque already off ({off[0]})")
                if hz:
                    now = time.monotonic()
                    if now < nxt:
                        time.sleep(nxt - now)
                    elif now - nxt > 0.5 / hz:
                        misses += 1
                    nxt += 1 / hz
                t = time.monotonic() - t0
                goal = {}
                for k, (f, j) in enumerate(JOINTS):
                    goal.setdefault(f, {})[j] = center[(f, j)] + AMP * math.sin(
                        2 * math.pi * 0.5 * t + k * 2 * math.pi / len(JOINTS))
                bus.sync_set_joints(goal)
                fb = bus.read_feedback(mode)
                stamps.append(time.monotonic())
                pos = fb["position"]
                err = max(abs(pos[f][j] - goal[f][j]) for f, j in JOINTS)
                now = time.monotonic()
                if err > 5:
                    jam = jam or now
                    if now - jam > 0.6:
                        raise RuntimeError(f"tracking error {err:.1f} deg")
                else:
                    jam = None
                if "load" in fb and max(abs(v) for v in fb["load"].values()) > 500:
                    over = over or now
                    if now - over > 0.3:
                        raise RuntimeError("|load| > 500")
                else:
                    over = None
                if "speed" in fb and prev is not None:
                    for f, j in JOINTS:
                        dp, v = pos[f][j] - prev[f][j], fb["speed"][f][j]
                        if abs(dp) >= 0.29 and v != 0:     # moved >= 1 count, speed non-zero
                            sign_n += 1
                            sign_ok += (dp > 0) == (v > 0)
                prev = pos
            temps = bus.read_field("temperature") or {}
            if temps and max(temps.values()) > 45:
                raise RuntimeError(f"temperature {max(temps.values())} C")
            periods = [b - a for a, b in zip(stamps, stamps[1:])]
            name = f"{mode}@{'free' if not hz else f'{hz:g}Hz'}"
            r = {"rate_hz": round(len(stamps) / DURATION, 1),
                 "period_ms_median": round(st.median(periods) * 1e3, 2),
                 "period_ms_max": round(max(periods) * 1e3, 2),
                 "deadline_misses": misses,
                 "transactions": bus.transactions - t_tx[0], "timeouts": bus.timeouts - t_tx[1],
                 "speed_sign_agree": f"{sign_ok}/{sign_n}" if sign_n else None,
                 "max_temp_c": max(temps.values()) if temps else None}
            summary["runs"][name] = r
            print(f"{name:22} {r}", flush=True)
    except RuntimeError as exc:
        print(f"  !! ABORT: {exc}", flush=True)
        summary["abort"] = str(exc)
    finally:
        bt.cancel()
        all_off("end of test")
    print("JSON_BEGIN\n" + json.dumps(summary, indent=1) + "\nJSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
