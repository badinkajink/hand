#!/usr/bin/env python3
"""Full-hand control-loop rate (test 7): what a policy loop actually gets on this bus.

All nine servos torque ON, holding their current pose plus a small slow sine (+-AMP deg,
0.5 Hz, phase-shifted per joint) so the servos are really working. Each cycle:
  1. ONE sync write of all nine goals (bus.sync_set_joints)
  2. per-servo reads of the feedback fields (position, or position + load)
Variants (each DURATION s):
  free_pos       free-running, position only
  free_pos_load  free-running, position + load
  paced_50       paced to 50 Hz, position + load (deadline misses counted)
  paced_<HZ2>    paced to HZ2 Hz (default 30), position + load

    python3 loop_rate_test.py       (on the CB1, in the host dir; port auto-detected)

Safety: goals start at the current pose (no jump) and are clamped >= MARGIN inside the software
limits; abort if any |load| > LOAD_ABORT for ABORT_HOLD_S, any joint is > JAM_DEG off its goal for
JAM_S, or any servo > TEMP_ABORT_C; wall-clock backstop disables all torque; disable_all in
`finally`. Prints one CSV block (per-cycle timings) per variant and a JSON summary.
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import threading
import time

from manta_hand.servos import FINGER_JOINTS, TORQUE_OFF, ServoBus

AMP = float(os.environ.get("AMP", 1.0))
DURATION = float(os.environ.get("DURATION", 10.0))
HZ2 = float(os.environ.get("HZ2", 30.0))
SPEED = 80
MARGIN = 1.0
LOAD_ABORT, ABORT_HOLD_S = 500, 0.3
JAM_DEG, JAM_S = 5.0, 0.6
TEMP_ABORT_C = 45.0

_ports = sorted(glob.glob("/dev/ttyUSB*"))
PORT = os.environ.get("SERVO_PORT") or (_ports[0] if len(_ports) == 1 else "/dev/ttyUSB0")
bus = ServoBus(PORT)
c = bus.controller
JOINTS = [(f, j, sid, zero, lims) for f, js in FINGER_JOINTS.items()
          for j, (sid, zero, lims) in js.items()]


def latency_ms() -> str:
    name = os.path.basename(PORT)
    try:
        return open(f"/sys/bus/usb-serial/devices/{name}/latency_timer").read().strip()
    except OSError:
        return "?"


def main() -> int:
    off: list[str] = []

    def all_off(reason: str) -> None:
        if off:
            return
        off.append(reason)
        try:
            bus.disable_all()
        except Exception:  # noqa: BLE001
            for _f, _j, sid, _z, _l in JOINTS:
                try:
                    bus.servo(sid)._call(c.write_torque_enable, TORQUE_OFF)
                except Exception:  # noqa: BLE001
                    pass
        print(f"  ALL TORQUE OFF ({reason})", flush=True)

    # start pose, clamped so pose +- AMP stays MARGIN inside the limits
    start = bus.sync_read_joint_positions()
    center = {}
    for f, j, sid, zero, (lo, hi) in JOINTS:
        center[(f, j)] = min(max(start[f][j], lo + AMP + MARGIN), hi - AMP - MARGIN)
    variants = [("free_pos", 0.0, ("position",)), ("free_pos_load", 0.0, ("position", "load")),
                ("paced_50", 50.0, ("position", "load")),
                (f"paced_{HZ2:g}", HZ2, ("position", "load"))]
    total_s = len(variants) * (DURATION + 1.5) + 10
    print(f"port {PORT}, ftdi latency {latency_ms()} ms, {len(JOINTS)} servos, AMP {AMP} deg, "
          f"{DURATION}s per variant, backstop {total_s:.0f}s", flush=True)
    backstop = threading.Timer(total_s, all_off, args=("wall-clock backstop",))
    backstop.daemon = True
    summary: dict = {"port": PORT, "latency_ms": latency_ms(), "variants": {}}
    over_since = jam_since = None

    def goals(t: float) -> dict:
        pose: dict = {}
        for k, (f, j, *_r) in enumerate(JOINTS):
            pose.setdefault(f, {})[j] = center[(f, j)] + AMP * math.sin(
                2 * math.pi * 0.5 * t + k * 2 * math.pi / len(JOINTS))
        return pose

    try:
        # hold the current pose first, with goal_speed set, then torque on
        bus.sync_set_joints({f: {j: center[(f, j)] for (ff, j, *_r) in JOINTS if ff == f}
                             for f in FINGER_JOINTS}, speed=SPEED)
        backstop.start()
        bus.enable_all()
        time.sleep(0.5)
        for name, hz, fields in variants:
            t_before = (bus.transactions, bus.timeouts)
            rows = []
            t0 = time.monotonic()
            nxt = t0
            misses = 0
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
                c0 = time.monotonic()
                g = goals(c0 - t0)
                bus.sync_set_joints(g)
                c1 = time.monotonic()
                pos = {sid: bus.servo(sid)._call(c.read_present_position)[0]
                       for _f, _j, sid, _z, _l in JOINTS}
                c2 = time.monotonic()
                load = ({sid: bus.servo(sid)._call(c.read_present_load)[0]
                         for _f, _j, sid, _z, _l in JOINTS} if "load" in fields else {})
                c3 = time.monotonic()
                rows.append((round(c0 - t0, 5), round((c1 - c0) * 1e3, 3),
                             round((c2 - c1) * 1e3, 3), round((c3 - c2) * 1e3, 3)))
                # safety checks
                worst_err = max(abs(math.degrees(pos[sid]) - zero - g[f][j])
                                for f, j, sid, zero, _l in JOINTS)
                now = time.monotonic()
                if worst_err > JAM_DEG:
                    jam_since = jam_since or now
                    if now - jam_since > JAM_S:
                        raise RuntimeError(f"jam guard: a joint {worst_err:.1f} deg off its goal")
                else:
                    jam_since = None
                if load and max(abs(v) for v in load.values()) > LOAD_ABORT:
                    over_since = over_since or now
                    if now - over_since > ABORT_HOLD_S:
                        raise RuntimeError(f"|load| > {LOAD_ABORT}")
                else:
                    over_since = None
            temps = bus.read_field("temperature") or {}
            if temps and max(temps.values()) > TEMP_ABORT_C:
                raise RuntimeError(f"temperature {max(temps.values())} C")
            periods = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
            cyc = sorted(r[1] + r[2] + r[3] for r in rows)
            s = {"cycles": len(rows), "rate_hz": round(len(rows) / DURATION, 1),
                 "period_ms_median": round(st.median(periods) * 1e3, 2),
                 "period_ms_p99": round(sorted(periods)[int(0.99 * (len(periods) - 1))] * 1e3, 2),
                 "period_ms_max": round(max(periods) * 1e3, 2),
                 "work_ms_median": round(st.median(cyc), 2),
                 "write_ms_median": round(st.median(r[1] for r in rows), 3),
                 "pos_read_ms_median": round(st.median(r[2] for r in rows), 2),
                 "load_read_ms_median": round(st.median(r[3] for r in rows), 2),
                 "deadline_misses": misses,
                 "transactions": bus.transactions - t_before[0],
                 "timeouts": bus.timeouts - t_before[1],
                 "max_temp_c": max(temps.values()) if temps else None}
            summary["variants"][name] = s
            print(f"{name}: {s}", flush=True)
            print(f"CSV_BEGIN {name}\nt_s,write_ms,pos_read_ms,load_read_ms")
            for r in rows:
                print(",".join(str(x) for x in r))
            print("CSV_END", flush=True)
    except RuntimeError as exc:
        print(f"  !! ABORT: {exc}", flush=True)
        summary["abort"] = str(exc)
    finally:
        backstop.cancel()
        all_off("end of test")
    summary["bus_timeouts_total"] = f"{bus.timeouts}/{bus.transactions}"
    print("JSON_BEGIN\n" + json.dumps(summary, indent=1) + "\nJSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
