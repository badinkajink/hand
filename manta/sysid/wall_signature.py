#!/usr/bin/env python3
"""Does StallGuard see the home wall at a given speed/current/SGT? Needs a
correct zero (at the wall). Logs sg + stall flag vs position on the approach.

    python3 ~/manta-hand/sysid/wall_signature.py <joint> <speed_sps> <irun> <sgt> [backoff_mm=10] [overtravel_mm=1.5]

Backs off to backoff_mm, approaches the wall at speed_sps (accel 8000, so most of
the approach is at cruise), and STOP+DIS once the count passes -overtravel_mm
(the move is aimed past that so it is still at cruise there). Prints sg
median/p10 and flag fraction for free running (> 2 mm) vs the last 1 mm before
the wall and past it. Run current applies only while approaching.
"""

import os
import sys
import time

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))  # sibling sysid scripts
import gantry_rom_sweep as g  # noqa: E402
from host_home_test import DEFAULT_IHOLD_IRUN, set_run_current  # noqa: E402
from manta_hand import MantaHandDriver  # noqa: E402

ACCEL = 8000


def stats(rows):
    if not rows:
        return "        -"
    sg = sorted(r[1] for r in rows)
    n = len(sg)
    return f"sg med {sg[n // 2]:4d} p10 {sg[n // 10]:4d} flag {sum(r[2] for r in rows) / n:4.0%} (n {n})"


def main():
    j, speed, irun, sgt = (int(a) for a in sys.argv[1:5])
    backoff = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
    over = float(sys.argv[6]) if len(sys.argv) > 6 else 1.5
    if not 1 <= irun <= 3 or over > 6:
        sys.exit("irun 1-3, overtravel <= 6")
    g.active_joints.append(j)
    g.driver = MantaHandDriver(g.PORT)
    jt = g.driver.joints[j]
    try:
        jt.set_scale(g.STEPS_PER_MM)
        jt.enable()
        g.STALL_CONFIRM_SAMPLES = 8
        g.move_and_log(j, backoff, None, {})
        jt.write_reg5160(0x6D, sgt << 16)
        set_run_current(j, (DEFAULT_IHOLD_IRUN & ~0x1F00) | (irun << 8))
        decel_mm = speed ** 2 / (2 * ACCEL) / abs(g.STEPS_PER_MM)
        jt.move_to_mm(-(over + decel_mm + 1.0), speed, ACCEL)
        ramp_s = speed / ACCEL
        t0 = time.monotonic()
        cap = (backoff + over) * abs(g.STEPS_PER_MM) / speed + ramp_s + 3
        rows = []
        while True:
            now = time.monotonic() - t0
            sg, stall = g.drv_status(j)
            mm = jt.status.position / g.STEPS_PER_MM
            if now > ramp_s + 0.2:
                rows.append((mm, sg, int(stall), time.time()))
            if mm <= -over or now > cap:
                break
        g.disable_all()
        set_run_current(j, DEFAULT_IHOLD_IRUN)
        # The approach ended past a correct zero, so the carriage is against the wall
        # (the motor skipped the rest): re-zero so the next run starts exact.
        if mm <= -over:
            jt.zero()
        with open(f"/tmp/wallsig_j{j}_{speed}.csv", "w") as f:
            f.write("t,mm,sg,stall\n")
            for r in rows:
                f.write(f"{r[3]:.4f},{r[0]:.3f},{r[1]},{r[2]}\n")
        free = [r for r in rows if r[0] > 2]
        near = [r for r in rows if 0 < r[0] <= 1]
        past = [r for r in rows if r[0] <= 0]
        print(f"J{j} {speed:5d} sps IRUN {irun} SGT {sgt:2d} | free {stats(free)} | last 1 mm {stats(near)} | past wall {stats(past)}", flush=True)
    finally:
        g.disable_all()
        set_run_current(j, DEFAULT_IHOLD_IRUN)
        g.driver.close()


if __name__ == "__main__":
    main()
