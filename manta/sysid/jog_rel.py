#!/usr/bin/env python3
"""Move one axis by a relative distance from wherever its count is, then disable.

    python3 /tmp/jog_rel.py <joint> <delta_mm> [velocity_sps=4000] [--irun N]

For probing when the zero is unknown (no bounds check against travel -- keep
delta small). Positive = away from home. Wall-clock capped; STOP+DIS on exit.
--irun N sets the run current for this move only (default restored after).
"""

import signal
import sys
import time

sys.path.insert(1, "/tmp")
import gantry_rom_sweep as g  # noqa: E402
from host_home_test import DEFAULT_IHOLD_IRUN, set_run_current  # noqa: E402
from manta_hand import MantaHandDriver  # noqa: E402

MAX_DELTA_MM = 20.0


def main():
    args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--irun"]
    j, delta = int(args[0]), float(args[1])
    vel = int(args[2]) if len(args) > 2 else 4000
    irun = int(sys.argv[sys.argv.index("--irun") + 1]) if "--irun" in sys.argv else None
    if abs(delta) > MAX_DELTA_MM:
        sys.exit(f"|delta| > {MAX_DELTA_MM} mm -- refusing")
    if irun is not None and not 1 <= irun <= 3:
        sys.exit("--irun must be 1-3")
    g.active_joints.append(j)
    signal.signal(signal.SIGHUP, g.on_signal)
    signal.signal(signal.SIGTERM, g.on_signal)
    g.driver = MantaHandDriver(g.PORT)
    jt = g.driver.joints[j]
    try:
        jt.set_scale(g.STEPS_PER_MM)
        p0 = jt.status.position / g.STEPS_PER_MM
        if irun is not None:
            set_run_current(j, (DEFAULT_IHOLD_IRUN & ~0x1F00) | (irun << 8))
        jt.enable()
        jt.move_to_mm(p0 + delta, vel, 8000)
        cap = abs(delta) * abs(g.STEPS_PER_MM) / vel + 3.0
        t0 = time.monotonic()
        while jt.status.moving:
            if time.monotonic() - t0 > cap:
                raise RuntimeError("move exceeded cap -- disabled")
            time.sleep(0.02)
        jt.disable()
        print(f"J{j}: count {p0:.2f} -> {jt.status.position / g.STEPS_PER_MM:.2f} mm", flush=True)
    finally:
        g.disable_all()
        if irun is not None:
            set_run_current(j, DEFAULT_IHOLD_IRUN)
        g.driver.close()


if __name__ == "__main__":
    main()
