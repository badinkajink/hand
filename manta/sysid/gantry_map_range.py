#!/usr/bin/env python3
"""Map StallGuard load over a short stretch of one axis with repeated passes.

    python3 /tmp/gantry_map_range.py <joint> <lo_mm> <hi_mm> <cycles> [accel] [--seat]

--seat: start, and end every back pass, by driving slowly 0.5 mm past the
home hardstop (a short deliberate push) and re-zeroing there.

Trusts the current step-0 (no HOME). Higher accel (default 24000 sps^2 ->
~0.9 mm ramps) keeps almost the whole stretch at cruise speed, where sg is
meaningful. Same abort rules as gantry_rom_sweep.py (stall held 3 samples,
per-move cap, STOP+DIS on exit). Writes /tmp/gantry_map_j<joint>_<ts>.csv.
"""

import csv
import signal
import sys
import time

sys.path.insert(1, "/tmp")
import gantry_rom_sweep as g  # noqa: E402
from manta_hand import MantaHandDriver  # noqa: E402

SEAT = "--seat" in sys.argv


SEAT_VELOCITY = 4000  # sps, ~1.2 mm/s into the wall
SEAT_ACCEL = 8000
SEAT_OVERTRAVEL_MM = 0.5  # command this far past the wall; the motor skips the rest


def seat(joint):
    """Drive slowly past the home hardstop by SEAT_OVERTRAVEL_MM (open loop, no
    stall check -- the wall is the point), then ZERO there. Wall-clock capped."""
    jt = g.driver.joints[joint]
    start = jt.status.position / g.STEPS_PER_MM
    target = -SEAT_OVERTRAVEL_MM
    jt.move_to_mm(target, SEAT_VELOCITY, SEAT_ACCEL)
    cap = abs(start - target) * abs(g.STEPS_PER_MM) / SEAT_VELOCITY + 3.0
    t0 = time.monotonic()
    while jt.status.moving:
        if time.monotonic() - t0 > cap:
            g.disable_all()
            raise RuntimeError(f"J{joint}: seat move exceeded {cap:.1f}s cap -- disabled")
        time.sleep(0.05)
    jt.zero()


def main():
    joint, lo, hi, cycles = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    g.SWEEP_ACCEL = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5].isdigit() else 24000
    full = g.FULL_EXTENSION_MM[joint]
    if not (0 < lo < hi < full):
        sys.exit(f"range must be inside (0, {full}) mm")
    g.active_joints.append(joint)
    signal.signal(signal.SIGHUP, g.on_signal)
    signal.signal(signal.SIGTERM, g.on_signal)
    g.driver = MantaHandDriver(g.PORT)
    out = f"/tmp/gantry_map_j{joint}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    jt = g.driver.joints[joint]
    try:
        jt.set_scale(g.STEPS_PER_MM)
        jt.enable()
        if SEAT:
            seat(joint)
        g.move_and_log(joint, lo, None, {})
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["cycle", "joint", "dir", "t", "mm", "sps", "cruise", "sg", "stall"])
            w.writeheader()
            for c in range(1, cycles + 1):
                for d, target in (("out", hi), ("back", lo)):
                    g.move_and_log(joint, target, w, {"cycle": c, "joint": joint, "dir": d})
                if SEAT:
                    seat(joint)
                f.flush()
                print(f"  cycle {c}/{cycles} done" + (" (seated at wall, re-zeroed)" if SEAT else ""), flush=True)
        jt.disable()
        print(f"wrote {out}")
    finally:
        g.disable_all()
        g.driver.close()


if __name__ == "__main__":
    main()
