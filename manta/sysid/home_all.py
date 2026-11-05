#!/usr/bin/env python3
"""Re-home gantry axes one at a time.

    python3 ~/manta-hand/sysid/home_all.py [J0 J1 ...] [--j1-near]     (default: all six)

J0, J2-J5: firmware HOME via the library's _home_one_axis (its own back-off,
COOLCONF, timeout covering full travel). Flags firmware stops that land at the
end of the unchecked window (~6.9 s), the false-stall signature.

J1: wall-push home, no StallGuard. J1's StallGuard reading does not change when
its rotor is fully stopped at the wall (camera-verified 2026-11-03), but J1
stalls cleanly there (coupler stops, nothing slips) and lands in the same place
every time (6/6 within 0.4 mm on camera). So: at IRUN 2 (~287 mA), drive toward
home far enough to be sure it reached the wall, then ZERO.
  default:    from wherever it is, by full travel + 2 mm, no back-off first
              (safe after a board restart; grinds up to ~16 s if already home)
  --j1-near:  current zero trusted: back off to 20 mm, overrun 3 mm (<1 s grind)
Each axis is disabled straight after homing.
"""

import signal
import os
import sys
import time

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))  # sibling sysid scripts
import gantry_rom_sweep as g  # noqa: E402
import host_home_test as hh  # noqa: E402
from manta_hand import MantaHandDriver  # noqa: E402

J1_IRUN = 2  # ~287 mA RMS, only while homing; IRUN 1 can stall in J1's tight zone near home
J1_NEAR_BACKOFF_MM = 20.0
J1_NEAR_OVERTRAVEL_MM = 3.0
J1_SGT_UNUSED_HOLD_S = 1e9  # never confirm a StallGuard stall on J1


def home_firmware(j):
    jt = g.driver.joints[j]
    p0 = jt.status.position / g.STEPS_PER_MM
    t0 = time.monotonic()
    jt.enable()
    g._home_one_axis(g.driver, j)
    jt.disable()
    t = time.monotonic() - t0
    st = jt.status
    flag = "  <-- stopped at end of unchecked window: possible FALSE stall" if abs(t - hh.GRACE_S) < 0.6 else ""
    print(f"  J{j}: firmware home, result {st.homing_result}, {t:.1f}s, started at count {p0:.2f} mm{flag}", flush=True)


def home_j1(near=False):
    jt = g.driver.joints[1]
    jt.enable()
    if near:
        g.STALL_CONFIRM_SAMPLES = hh.BACKOFF_CONFIRM_SAMPLES
        g.move_and_log(1, J1_NEAR_BACKOFF_MM, None, {})
        overtravel = J1_NEAR_OVERTRAVEL_MM
    else:
        # Position unknown: drive from here toward home by more than the whole travel.
        overtravel = g.FULL_EXTENSION_MM[1] + 2.0 - jt.status.position / g.STEPS_PER_MM

    class _Null:
        def writerow(self, _):
            pass
    hh.set_run_current(1, (hh.DEFAULT_IHOLD_IRUN & ~0x1F00) | (J1_IRUN << 8))
    try:
        result, p0, p1, t, _ = hh.home(1, J1_SGT_UNUSED_HOLD_S, _Null(), 1, overtravel)
    finally:
        hh.set_run_current(1, hh.DEFAULT_IHOLD_IRUN)
    jt.disable()
    if result != "overtravel":
        print(f"  J1: wall-push home ended by {result} -- NOT zeroed", flush=True)
        return result
    jt.zero()
    print(f"  J1: wall-push home ({'near' if near else 'full travel'}): count {p0:.2f} -> {p1:.2f} mm "
          f"in {t:.1f}s, zeroed at the wall", flush=True)
    return result


def main():
    near = "--j1-near" in sys.argv
    joints = [int(a.upper().lstrip("J")) for a in sys.argv[1:] if not a.startswith("--")] or [0, 1, 2, 3, 4, 5]
    if any(v <= 0 for v in g.FULL_EXTENSION_MM.values()):
        sys.exit("stale driver: non-positive FULL_EXTENSION_MM -- refusing to move")
    g.active_joints.extend(joints)
    signal.signal(signal.SIGHUP, g.on_signal)
    signal.signal(signal.SIGTERM, g.on_signal)
    g.driver = MantaHandDriver(g.PORT)
    hh.g = g
    try:
        for j in joints:
            g.driver.joints[j].set_scale(g.STEPS_PER_MM)
            if j == 1:
                home_j1(near)
            else:
                home_firmware(j)
    finally:
        g.disable_all()
        for j in joints:
            print(f"J{j}", g.driver.joints[j].status)
        g.driver.close()


if __name__ == "__main__":
    main()
