#!/usr/bin/env python3
"""Servo gain sweep on ONE joint: does a stiffer internal controller cut stick-slip jitter?

For each (P, I, D) setting: write the gains (lock flag left untouched; every value read back),
then
  sine  : +-AMP deg at 0.5 Hz for SINE_S (the motion that looked jittery), goals at 50 Hz
  hold  : HOLD_S at the sine's center, to see steady-state error, holding load and buzzing
Logged at full bus rate with one 6-byte read (position, speed, load) per sample.
The factory gains (15, 0, 15) run first and last, so drift and heating show up.

    JOINT=1:fe1 python3 gain_sweep.py        (on the CB1, in the host dir)

Safety: goal = current position before torque on; targets inside software limits; abort if
|load| > 500 for 0.3 s, tracking error > 5 deg for 0.6 s, temperature > 45 C, or the position
reverses direction more than BUZZ_MAX times per second during a hold (oscillation). Wall-clock
backstop turns torque off. ALWAYS restores the original P/I/D in `finally`, then reads them back.
"""
from __future__ import annotations

import glob
import math
import os
import threading
import time

from manta_hand.servos import FINGER_JOINTS, TORQUE_OFF, ServoBus

AMP = float(os.environ.get("AMP", 2.0))
SINE_S = 10.0
HOLD_S = float(os.environ.get("HOLD_S", 4.0))
GOAL_HZ = 50.0
SPEED = 80
LEAD = float(os.environ.get("LEAD", 0.0))       # friction feed-forward: lead the goal by LEAD deg in the direction of motion
LEAD_V = float(os.environ.get("LEAD_V", 10.0))  # deg/s; lead = LEAD * tanh(target velocity / LEAD_V)
BUZZ_MAX = 6.0       # direction reversals per second during a hold
SETTINGS = [(15, 0, 15), (30, 0, 15), (45, 0, 15), (15, 2, 15), (15, 5, 15), (30, 2, 15),
            (15, 0, 15)]
if os.environ.get("SETTINGS"):
    SETTINGS = [tuple(int(v) for v in s.split("/")) for s in os.environ["SETTINGS"].split(",")]

f_s, joint = os.environ.get("JOINT", "1:fe1").split(":")
finger = int(f_s)
bus = ServoBus(os.environ.get("SERVO_PORT") or sorted(glob.glob("/dev/ttyUSB*"))[0])
c = bus.controller
sid, zero, (lo, hi) = FINGER_JOINTS[finger][joint]
s = bus.servo(sid)


def sm(v: int, bit: int) -> int:
    m = v & ((1 << bit) - 1)
    return -m if v & (1 << bit) else m


def feedback() -> tuple[float, int]:
    b = list(s._call(c.read_raw_data, 0x38, 6))
    return ((b[0] << 8 | b[1]) - 511) * 300 / 1024 - zero, sm(b[4] << 8 | b[5], 10)


def gains() -> tuple[int, int, int]:
    return (s._call(c.read_p_coefficient)[0], s._call(c.read_i_coefficient)[0],
            s._call(c.read_d_coefficient)[0])


def set_gains(p: int, i: int, d: int) -> None:
    s._call(c.write_p_coefficient, p)
    s._call(c.write_i_coefficient, i)
    s._call(c.write_d_coefficient, d)
    got = gains()
    if got != (p, i, d):
        raise RuntimeError(f"gains read back {got}, wanted {(p, i, d)}")


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

    original = gains()
    lock = s._call(c.read_raw_lock)[0]
    pos0_raw = s._call(c.read_present_position)[0]
    start = math.degrees(pos0_raw) - zero
    want = float(os.environ["CENTER"]) if os.environ.get("CENTER") else start
    center = min(max(want, lo + AMP + 1.5), hi - AMP - 1.5)
    per = SINE_S + HOLD_S + 2.0
    print(f"finger {finger} {joint} (servo {sid}): gains {original}, lock {lock}, start {start:.2f}, "
          f"center {center:.2f}, settings {SETTINGS}, lead {LEAD} deg", flush=True)
    bt = threading.Timer(len(SETTINGS) * (per + 3) + 15 + abs(center - start) / 2.0, torque_off, args=("wall-clock backstop",))
    bt.daemon = True
    rows: list[tuple] = []
    t0 = time.monotonic()
    try:
        s._call(c.write_goal_position, pos0_raw)
        bus.sync_set_joints({finger: {joint: start}}, speed=SPEED)
        bt.start()
        s.enable()
        # ease to center under factory gains
        n_ease = max(25, int(abs(center - start) / 2.0 / 0.04))     # <= 2 deg/s
        for k in range(1, n_ease + 1):
            bus.sync_set_joints({finger: {joint: start + (center - start) * k / n_ease}})
            time.sleep(0.04)
        time.sleep(0.5)
        for p, i, d in SETTINGS:
            set_gains(p, i, d)
            tag = f"P{p}I{i}D{d}"
            print(f"  {tag}: running", flush=True)
            over = jam = None
            for phase, dur in (("sine", SINE_S), ("hold", HOLD_S)):
                ts = time.monotonic()
                nxt = ts
                tgt = center
                reversals, last_dir, last_p = 0, 0, None
                while time.monotonic() - ts < dur:
                    if off:
                        raise RuntimeError(f"torque already off ({off[0]})")
                    now = time.monotonic()
                    if now >= nxt:
                        tgt = center + (AMP * math.sin(2 * math.pi * 0.5 * (now - ts))
                                        if phase == "sine" else 0.0)
                        vel = AMP * math.pi * math.cos(math.pi * (now - ts)) if phase == "sine" else 0.0
                        cmd = tgt + LEAD * math.tanh(vel / LEAD_V)
                        bus.sync_set_joints({finger: {joint: min(max(cmd, lo), hi)}})
                        nxt += 1 / GOAL_HZ
                    pos, load = feedback()
                    now = time.monotonic()
                    rows.append((round(now - t0, 4), tag, phase, round(tgt, 3), round(pos, 3), load))
                    if abs(pos - tgt) > 5:
                        jam = jam or now
                        if now - jam > 0.6:
                            raise RuntimeError(f"{tag}: tracking error {pos - tgt:.1f} deg")
                    else:
                        jam = None
                    if abs(load) > 500:
                        over = over or now
                        if now - over > 0.3:
                            raise RuntimeError(f"{tag}: |load| {load}")
                    else:
                        over = None
                    if phase == "hold" and last_p is not None and pos != last_p:
                        dr = 1 if pos > last_p else -1
                        if last_dir and dr != last_dir:
                            reversals += 1
                            if reversals > BUZZ_MAX * max(1.0, now - ts):
                                raise RuntimeError(f"{tag}: oscillating ({reversals} reversals)")
                        last_dir = dr
                    last_p = pos
            temp = s._call(c.read_present_temperature)[0]
            rows.append((round(time.monotonic() - t0, 4), tag, "temp", 0, 0, temp))
            print(f"  {tag}: done, {temp} C", flush=True)
            if temp > 45:
                raise RuntimeError(f"temperature {temp} C")
    except RuntimeError as exc:
        print(f"  !! ABORT: {exc}", flush=True)
    finally:
        bt.cancel()
        torque_off("end")
        try:
            set_gains(*original)
            print(f"  gains restored to {gains()}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! GAIN RESTORE FAILED: {exc} -- restore P/I/D {original} by hand", flush=True)
    print("CSV_BEGIN gains")
    print("t_s,setting,phase,target_deg,pos_deg,load_or_temp")
    for r in rows:
        print(",".join(str(x) for x in r))
    print("CSV_END")
    print(f"bus timeouts {bus.timeouts}/{bus.transactions}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
