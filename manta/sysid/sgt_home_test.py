#!/usr/bin/env python3
"""Validate a StallGuard SGT value for one axis's HOME with repeated
back-off-and-rehome cycles (docs/bringup.md step 5).

    python3 /tmp/sgt_home_test.py <joint> <sgt> [cycles=3] [backoff_mm=20]

Each cycle: MOVEMM out to backoff_mm, HOME with COOLCONF = sgt<<16, report
travel and time-to-stall, ZERO. A good SGT stalls well after the grace window
(~6.8 s at 12000 sps / 2000 sps^2) and travels ~backoff_mm every cycle after
the first. Home stall time ~= grace end means a false early stall.

Backstop for a too-insensitive SGT: after the grace window, if sg stays <= 5
for 0.5 s without the firmware finishing, STOP+DIS (the axis is grinding at
the wall). Always disables on exit.
"""

import sys
import time

sys.path.insert(1, "/tmp")
import gantry_rom_sweep as g  # noqa: E402
from manta_hand import MantaHandDriver  # noqa: E402

GRACE_S = g.HOME_VELOCITY / g.HOME_ACCEL + 0.8
GRIND_SG = 5
GRIND_HOLD_S = 0.5
MAX_OVERTRAVEL_MM = 3.0  # assumes the current zero is within a few mm of the real wall


def main():
    joint, sgt = int(sys.argv[1]), int(sys.argv[2])
    cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    backoff = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
    g.active_joints.append(joint)
    g.driver = MantaHandDriver(g.PORT)
    j = g.driver.joints[joint]
    try:
        j.set_scale(g.STEPS_PER_MM)
        for c in range(1, cycles + 1):
            j.enable()
            g.move_and_log(joint, backoff, None, {})
            p0 = j.status.position / g.STEPS_PER_MM
            j.write_reg5160(0x6D, sgt << 16)
            tmo = g._home_timeout_ms(joint)
            j.home(g.HOME_DIRECTION, g.HOME_VELOCITY, g.HOME_ACCEL, tmo)
            t0 = time.monotonic()
            low_since = None
            st = j.status
            while st.homing_result == 1:
                now = time.monotonic() - t0
                if now > tmo / 1000 + 5:
                    g.disable_all()
                    raise RuntimeError("home exceeded cap -- disabled")
                # Travel backstop: the wall should be ~backoff_mm away. The sg-based
                # check below did NOT catch a 52 mm grind at SGT 13 (sg stays noisy
                # while grinding), so cap travel directly.
                p = j.status.position / g.STEPS_PER_MM
                if p0 - p > backoff + MAX_OVERTRAVEL_MM:
                    g.disable_all()
                    raise RuntimeError(f"J{joint} SGT {sgt}: travelled {p0 - p:.1f} mm, past the expected wall "
                                       f"(+{MAX_OVERTRAVEL_MM} mm) without a stall -- disabled")
                sg, _ = g.drv_status(joint)
                if now > GRACE_S + 0.3 and sg <= GRIND_SG:
                    low_since = low_since or now
                    if now - low_since >= GRIND_HOLD_S:
                        g.disable_all()
                        p = j.status.position / g.STEPS_PER_MM
                        raise RuntimeError(f"J{joint} SGT {sgt}: sg<={GRIND_SG} for {GRIND_HOLD_S}s at "
                                           f"{p:.1f} mm without a firmware stall -- grinding, disabled")
                else:
                    low_since = None
                time.sleep(0.05)
                st = j.status
            t = time.monotonic() - t0
            p1 = st.position / g.STEPS_PER_MM
            j.disable()
            j.zero()
            flag = "  <-- at grace end: likely FALSE stall" if t < GRACE_S + 0.6 else ""
            print(f"  cycle {c}: J{joint} SGT {sgt}: from {p0:.1f} mm, stall at {p1:+.2f} mm, "
                  f"travelled {p0 - p1:.1f} mm in {t:.1f}s (result {st.homing_result}){flag}", flush=True)
    finally:
        g.disable_all()
        print(j.status)
        g.driver.close()


if __name__ == "__main__":
    main()
