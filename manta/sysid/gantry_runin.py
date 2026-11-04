#!/usr/bin/env python3
"""Run one gantry axis back and forth over its full range N times (break-in
after a mechanical fix), logging StallGuard sg like gantry_rom_sweep.py.

    python3 /tmp/gantry_runin.py <finger> <joint> <cycles> [--no-home]

--no-home: trust the current step-0 (e.g. carriage pushed to the home
hardstop by hand and ZEROed) instead of StallGuard homing -- for J1, whose
home false-stalls early right after the grace window.

Homes only the cycled axis; the finger's other axis stays disabled and
untouched. Then cycles. Same safety as the sweep: aborts on a stall held 3 samples,
per-move wall-clock caps, STOP+DIS on exit/signal. Ends with the lost-step
check. Writes /tmp/gantry_runin_j<joint>_<timestamp>.csv.
"""

import csv
import signal
import sys
import time

sys.path.insert(1, "/tmp")
import gantry_rom_sweep as g  # noqa: E402
from manta_hand import MantaHandDriver  # noqa: E402


NO_HOME = "--no-home" in sys.argv


def run_cycles(joint, cycles, out):
    near, far = g.travel(joint)
    jt = g.driver.joints[joint]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cycle", "joint", "dir", "t", "mm", "sps", "cruise", "sg", "stall"])
        w.writeheader()
        jt.enable()
        g.move_and_log(joint, near, None, {})
        for c in range(1, cycles + 1):
            for d, target in (("out", far), ("back", near)):
                g.move_and_log(joint, target, w, {"cycle": c, "joint": joint, "dir": d})
            f.flush()
            print(f"  cycle {c}/{cycles} done", flush=True)
        jt.disable()


def main():
    finger, joint, cycles = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    jx, jy = g.STEPPER_JOINTS[finger]
    if joint not in (jx, jy):
        sys.exit(f"J{joint} is not on finger {finger}")
    if any(v <= 0 for v in g.FULL_EXTENSION_MM.values()):
        sys.exit("stale driver: non-positive FULL_EXTENSION_MM -- refusing to move")
    g.active_joints.append(joint)
    signal.signal(signal.SIGHUP, g.on_signal)
    signal.signal(signal.SIGTERM, g.on_signal)
    g.driver = MantaHandDriver(g.PORT)
    out = f"/tmp/gantry_runin_j{joint}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    try:
        # Only the cycled axis is homed; the other axis stays disabled and
        # untouched (2026-11-02: homing J0 here missed its stall and ground
        # into the hardstop, and J1's run-in never needed it).
        g.driver.joints[joint].set_scale(g.STEPS_PER_MM)
        if NO_HOME:
            run_cycles(joint, cycles, out)
            print(f"wrote {out} (no home / lost-step check: zero was set by hand)")
            return
        g.driver.joints[joint].enable()
        # Back off past HOME's ~14 mm unchecked grace window first; the
        # library's own 10 mm back-off still pushes ~4 mm into the wall.
        if abs(g.driver.joints[joint].status.position / g.STEPS_PER_MM) < g.LOST_STEP_START_MM:
            g.move_and_log(joint, g.LOST_STEP_START_MM, None, {})
        g._home_one_axis(g.driver, joint)
        g.driver.joints[joint].disable()
        run_cycles(joint, cycles, out)
        offset, result = g.lost_steps_check(joint)
        print(f"  J{joint} lost-step check: home found at {offset:+.3f} mm vs original zero (result {result})")
        print(f"wrote {out}")
    finally:
        g.disable_all()
        g.driver.close()


if __name__ == "__main__":
    main()
