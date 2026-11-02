#!/usr/bin/env python3
"""Read-only health check of all nine SCS0009 finger servos.

Runs on the CB1 against the hand repo's driver (~/hand/src/morphohand/driver/manta/host).
Every bus transaction is a READ, issued through Servo._call so the driver's timing and
retry rules apply. It never enables torque, writes a register, or commands motion.

    ssh ... 'cd ~/hand/src/morphohand/driver/manta/host && python3 -' < servo_health_check.py

Prints a JSON blob (between markers) plus a human-readable table.
"""
from __future__ import annotations

import json
import math
import time

from manta_hand.servos import FINGER_JOINTS, ServoBus

PORT = "/dev/ttyUSB0"
FINGER_NAMES = {0: "thumb", 1: "index", 2: "middle"}

# live state
LIVE = ["present_position", "present_load", "present_speed", "present_voltage",
        "present_temperature", "status", "torque_enable", "moving"]
# the servo's own controller + protection configuration (what a BAM model needs)
CONFIG = ["p_coefficient", "i_coefficient", "d_coefficient", "max_torque_limit",
          "overload_torque", "protection_time", "protective_torque", "unloading_condition",
          "led_alarm_condition", "min_angle_limit", "max_angle_limit",
          "max_temperature_limit", "min_voltage_limit", "max_voltage_limit",
          "minimum_startup_force", "cw_dead_zone", "ccw_dead_zone", "return_delay_time",
          "model", "id"]


def read(bus, sid, field):
    fn = getattr(bus.controller, f"read_{field}", None)
    if fn is None:
        return "n/a"
    try:
        v = bus.servo(sid)._call(fn)
        v = v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v
        return float(v) if isinstance(v, (int, float)) else v
    except Exception as exc:  # noqa: BLE001 - report, don't crash the survey
        return f"ERR {type(exc).__name__}: {exc}"[:60]


def main() -> int:
    t0 = time.time()
    bus = ServoBus(PORT)
    out = {"latency_tuning": getattr(bus, "latency_tuning", None), "servos": {}}
    for fid, joints in FINGER_JOINTS.items():
        for jname, (sid, zero_deg, limits) in joints.items():
            rec = {"finger": FINGER_NAMES.get(fid, fid), "joint": jname, "zero_deg": zero_deg,
                   "limits_rel_deg": limits}
            for f in LIVE + CONFIG:
                rec[f] = read(bus, sid, f)
            pos = rec["present_position"]
            if isinstance(pos, float):
                rec["position_rel_deg"] = math.degrees(pos) - zero_deg
            out["servos"][sid] = rec
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["bus_timeouts"] = bus.timeouts
    out["bus_transactions"] = bus.transactions
    print("JSON_BEGIN")
    print(json.dumps(out, default=str))
    print("JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
