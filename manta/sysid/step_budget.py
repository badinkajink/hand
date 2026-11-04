#!/usr/bin/env python3
"""Measure how many steps/s the firmware can generate with N axes moving at once.

    python3 /tmp/step_budget.py <span_mm> <speed_sps> J2 [J3 ...] [--accel SPS2]

All listed axes start together (back to back, at standstill) from their current
count, go out by span_mm and come back. While they move the board gets no
traffic except one STATALL probe at PROBE_S after the start (often unanswered
under load -- reported, not fatal). Completion is detected by polling STATALL
every 0.25 s once the expected time has passed. Prints expected vs actual time
per leg and the implied total step rate. Axes are disabled at the end.
"""

import sys
import time

sys.path.insert(1, "/tmp")
import gantry_rom_sweep as g  # noqa: E402
from manta_hand import MantaHandDriver  # noqa: E402
from manta_hand.protocol import MantaHandError  # noqa: E402

ACCEL = 8000


def wait_done(d, joints, t0, expected):
    time.sleep(max(0.0, expected - (time.monotonic() - t0)))
    fails = 0
    while True:
        if time.monotonic() - t0 > 3 * expected + 15:
            g.disable_all()
            raise RuntimeError("legs exceeded 3x expected time -- disabled")
        try:
            st = d.get_all_status()
            if not any(st[j].moving for j in joints):
                return time.monotonic() - t0, fails, st
        except (MantaHandError, ValueError):
            fails += 1
        time.sleep(0.25)


def main():
    global ACCEL
    if "--accel" in sys.argv:
        i = sys.argv.index("--accel")
        ACCEL = int(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    span, speed = float(sys.argv[1]), int(sys.argv[2])
    joints = [int(a.upper().lstrip("J")) for a in sys.argv[3:]]
    g.active_joints.extend(joints)
    g.driver = d = MantaHandDriver(g.PORT)
    try:
        st0 = d.get_all_status()
        start = {j: st0[j].position / g.STEPS_PER_MM for j in joints}
        for j in joints:
            d.joints[j].set_scale(g.STEPS_PER_MM)
            near, far = g.travel(j)
            if not near <= start[j] + span <= far:
                sys.exit(f"J{j}: {start[j]:.1f}+{span} mm leaves travel -- nothing moved")
            d.joints[j].enable()
        steps = span * abs(g.STEPS_PER_MM)
        expected = steps / speed + speed / ACCEL
        out = []
        for leg, off in (("out", span), ("back", 0.0)):
            t0 = time.monotonic()
            for j in joints:
                d.joints[j].move_to_mm(start[j] + off, speed, ACCEL)
            t, fails, st = wait_done(d, joints, t0, expected)
            ok = all(abs(st[j].position / g.STEPS_PER_MM - (start[j] + off)) < 0.01 for j in joints)
            out.append(f"{leg} {t:5.1f}s ({fails} unanswered polls{'' if ok else ', POSITION MISMATCH'})")
        rate = len(joints) * steps / ((float(out[0].split()[1][:-1]) + float(out[1].split()[1][:-1])) / 2 - speed / ACCEL)
        print(f"  accel {ACCEL:6d} | {len(joints)} axes @ {speed:5d} sps (asked {len(joints) * speed:6d} total): expected {expected:4.1f}s/leg | "
              + " | ".join(out) + f" | achieved ~{rate:6.0f} steps/s total", flush=True)
    finally:
        g.disable_all()
        d.close()


if __name__ == "__main__":
    main()
