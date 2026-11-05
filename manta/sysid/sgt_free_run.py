#!/usr/bin/env python3
"""Free-running StallGuard reading vs SGT at a chosen run current, no wall contact.

    python3 ~/manta-hand/sysid/sgt_free_run.py <joint> <irun> <span_mm> <sgt> [<sgt> ...]

For each SGT: out by span_mm from the current count and back, at homing speed,
logging DRV_STATUS. Reports the cruise sg median/p10 and stall-flag fraction each
way. A usable homing SGT keeps free-running sg well above 0 with few flags.
The run current applies only during these moves; the firmware default is
restored on exit. No stall abort (a too-sensitive SGT flags constantly), so the
caller must make sure count+span_mm is inside travel. Wall-clock capped.
"""

import signal
import os
import sys
import time

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))  # sibling sysid scripts
import gantry_rom_sweep as g  # noqa: E402
from host_home_test import DEFAULT_IHOLD_IRUN, set_run_current  # noqa: E402
from manta_hand import MantaHandDriver  # noqa: E402


def leg(jt, j, target):
    jt.move_to_mm(target, g.HOME_VELOCITY, g.SWEEP_ACCEL)
    t0, rows, prev = time.monotonic(), [], None
    cap = g.move_cap_s(20 + abs(target)) + 5
    while True:
        now = time.monotonic()
        st = jt.status
        sg, stall = g.drv_status(j)
        if prev:
            vel = abs(st.position - prev[0]) / (now - prev[1])
            if vel >= g.CRUISE_FRACTION * g.HOME_VELOCITY:
                rows.append((sg, stall))
        prev = (st.position, now)
        if not st.moving:
            return rows
        if now - t0 > cap:
            g.disable_all()
            raise RuntimeError("move exceeded cap -- disabled")
        time.sleep(0.004)


def main():
    j, irun, span = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    sgts = [int(a) for a in sys.argv[4:]]
    if not 1 <= irun <= 3:
        sys.exit("irun must be 1-3")
    g.active_joints.append(j)
    signal.signal(signal.SIGHUP, g.on_signal)
    signal.signal(signal.SIGTERM, g.on_signal)
    g.driver = MantaHandDriver(g.PORT)
    jt = g.driver.joints[j]
    try:
        jt.set_scale(g.STEPS_PER_MM)
        start = jt.status.position / g.STEPS_PER_MM
        set_run_current(j, (DEFAULT_IHOLD_IRUN & ~0x1F00) | (irun << 8))
        for sgt in sgts:
            jt.write_reg5160(0x6D, sgt << 16)
            jt.enable()
            out = []
            for target in (start + span, start):
                rows = leg(jt, j, target)
                sg = sorted(r[0] for r in rows)
                n = len(sg) or 1
                out.append(f"{'out' if target > start else 'back'}: sg median {sg[n // 2] if sg else '-':>4} "
                           f"p10 {sg[n // 10] if sg else '-':>4} flagged {sum(r[1] for r in rows) / n:4.0%}")
            jt.disable()
            print(f"  SGT {sgt:3d} @ IRUN {irun}: " + " | ".join(out), flush=True)
    finally:
        g.disable_all()
        set_run_current(j, DEFAULT_IHOLD_IRUN)
        print(jt.status)
        g.driver.close()


if __name__ == "__main__":
    main()
