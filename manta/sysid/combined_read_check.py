#!/usr/bin/env python3
"""Validate the one-transaction feedback read (read_raw_data at 0x38, 6 bytes: position, speed,
load) against rustypot's separate read_present_* calls, on ONE joint under torque.

Decoding (big-endian on the wire):
  position counts = b0<<8 | b1,          deg = (counts - 511) * 300/1024   (same as the driver)
  speed           = b2<<8 | b3,          sign-magnitude, sign = bit 15
  load            = b4<<8 | b5,          sign-magnitude, sign = bit 10, magnitude = low 10 bits
Signs are reported as they come; the check is that both paths agree.

    JOINT=1:fe1 python3 combined_read_check.py      (on the CB1, in the host dir)

Safety: goal = current position first; +-AMP sine clamped inside software limits; abort on
|load| > 500 for 0.3 s or > 5 deg tracking error for 0.6 s; wall-clock backstop; torque off in finally.
"""
from __future__ import annotations

import glob
import math
import os
import threading
import time

from manta_hand.servos import FINGER_JOINTS, TORQUE_OFF, ServoBus

AMP, DUR, SPEED = 3.0, 6.0, 80
f_s, joint = os.environ.get("JOINT", "1:fe1").split(":")
finger = int(f_s)
_ports = sorted(glob.glob("/dev/ttyUSB*"))
bus = ServoBus(os.environ.get("SERVO_PORT") or _ports[0])
c = bus.controller
sid, zero, (lo, hi) = FINGER_JOINTS[finger][joint]
s = bus.servo(sid)


def sm(v: int, sign_bit: int, mag_bits: int) -> int:
    mag = v & ((1 << mag_bits) - 1)
    return -mag if v & (1 << sign_bit) else mag


def decode(raw: list[int]) -> tuple[float, int, int]:
    pos = ((raw[0] << 8 | raw[1]) - 511) * 300 / 1024
    spd = sm(raw[2] << 8 | raw[3], 15, 15)
    load = sm(raw[4] << 8 | raw[5], 10, 10)
    return pos, spd, load


def main() -> int:
    off: list[str] = []

    def torque_off(reason: str) -> None:
        if not off:
            off.append(reason)
            try:
                s._call(c.write_torque_enable, TORQUE_OFF)
            except Exception:  # noqa: BLE001
                s.disable()
            print(f"  TORQUE OFF ({reason})", flush=True)

    pos0 = s._call(c.read_present_position)[0]
    start = math.degrees(pos0) - zero
    center = min(max(start, lo + AMP + 1), hi - AMP - 1)
    bt = threading.Timer(DUR + 6, torque_off, args=("wall-clock backstop",))
    bt.daemon = True
    n = agree_pos = agree_load = agree_load_mag = agree_spd = 0
    mism: list[tuple] = []
    over = jam = None
    try:
        s._call(c.write_goal_position, pos0)
        bus.sync_set_joints({finger: {joint: start}}, speed=SPEED)
        bt.start()
        s.enable()
        t0 = time.monotonic()
        while time.monotonic() - t0 < DUR + 1:
            t = time.monotonic() - t0
            ramp = min(t, 1.0)                       # ease from start to the sine
            tgt = start + (center - start) * ramp + AMP * math.sin(2 * math.pi * 0.5 * max(0, t - 1))
            bus.sync_set_joints({finger: {joint: tgt}})
            raw = list(s._call(c.read_raw_data, 56, 6))
            p_ref = math.degrees(s._call(c.read_present_position)[0])
            l_ref = s._call(c.read_present_load)[0]
            v_ref = s._call(c.read_present_speed)[0]
            p, v, ld = decode(raw)
            n += 1
            agree_pos += abs(p - p_ref) <= 0.6          # 2 counts: reads are ~1 ms apart
            agree_load += ld == l_ref
            agree_load_mag += abs(ld) == abs(l_ref)
            agree_spd += (v == 0) == (v_ref == 0)
            if ld != l_ref and len(mism) < 8:
                mism.append((round(t, 2), raw, ld, l_ref, v, v_ref))
            now = time.monotonic()
            if abs(p - zero - tgt) > 5:
                jam = jam or now
                if now - jam > 0.6:
                    raise RuntimeError("tracking error > 5 deg")
            else:
                jam = None
            if abs(ld) > 500:
                over = over or now
                if now - over > 0.3:
                    raise RuntimeError("|load| > 500")
            else:
                over = None
    except RuntimeError as exc:
        print(f"  !! ABORT: {exc}", flush=True)
    finally:
        bt.cancel()
        torque_off("end")
    print(f"samples {n}: position agrees {agree_pos}, load exact {agree_load}, "
          f"load |magnitude| {agree_load_mag}, speed zero/non-zero {agree_spd}; "
          f"timeouts {bus.timeouts}/{bus.transactions}")
    print("load mismatches (t, raw, decoded, rustypot, speed dec, speed rustypot):")
    for m in mism:
        print("  ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
