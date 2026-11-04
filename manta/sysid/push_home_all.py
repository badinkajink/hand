#!/usr/bin/env python3
"""Home all listed gantry axes AT ONCE by pushing them into their home walls.

    python3 /tmp/push_home_all.py [J0 J1 ...] [--approach MM] [--overtravel MM]
                                  [--speed SPS] [--no-zero]
    (default: all six, approach 3 mm, overtravel 1.5 mm, 3500 sps, zero at the end)

Why not StallGuard: with several axes stepping, the firmware's step budget
(~24k steps/s total) makes step timing uneven, and the StallGuard reading gets
too noisy to see the ~20 % wall drop at 24 V (2026-11-04: J0/J2/J3 grouped at
8000 and 6000 sps gave 4/9 and 6/12 false or late stops). A wall push needs no
reading: low-current steppers stall cleanly against the wall and skip steps,
so after driving past the wall every carriage sits AT its wall.

Needs the current zeros to be right to within --overtravel (true unless the
stepper board was power-cycled -- then use home_all.py / sg_home.py per axis).
  1. all axes together to --approach mm (no traffic while moving)
  2. all axes together to -overtravel mm: each grinds at the wall for
     overtravel / speed seconds (1.5 mm at 3500 sps = 1.4 s) at low current;
     J1 at IRUN 2 (its tight zone near home can stall IRUN 1 short of the wall)
  3. disable all, ZERO all.
Speed per axis <= 24000 / number of axes. Run under `timeout` as a backstop.
"""

import signal
import sys

sys.path.insert(1, "/tmp")
import gantry_goto as gg  # noqa: E402
import gantry_rom_sweep as g  # noqa: E402
import host_home_test as hh  # noqa: E402
from manta_hand import MantaHandDriver  # noqa: E402

STEP_BUDGET = 24000
IRUN = {1: 2}  # J1 pushes at IRUN 2 (~287 mA); others stay at the default IRUN 1


def arg(name, default, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    flags_with_values = {"--approach", "--overtravel", "--speed"}
    joints = [int(a.upper().lstrip("J")) for i, a in enumerate(sys.argv[1:], 1)
              if a.upper().startswith("J") and sys.argv[i - 1] not in flags_with_values] or list(range(6))
    approach = arg("--approach", 3.0)
    overtravel = arg("--overtravel", 1.5)
    speed = arg("--speed", 3500, int)
    if speed * len(joints) > STEP_BUDGET:
        sys.exit(f"{len(joints)} axes x {speed} sps exceeds the ~{STEP_BUDGET} steps/s budget -- nothing moved")
    if not 0 < overtravel <= 3.0:
        sys.exit("--overtravel must be in (0, 3] mm -- nothing moved")
    if any(v <= 0 for v in g.FULL_EXTENSION_MM.values()):
        sys.exit("stale driver: non-positive FULL_EXTENSION_MM -- refusing to move")
    g.active_joints.extend(joints)
    signal.signal(signal.SIGHUP, g.on_signal)
    signal.signal(signal.SIGTERM, g.on_signal)
    g.driver = d = MantaHandDriver(g.PORT)
    hh.g = g
    gg.g = g
    try:
        for j in joints:
            d.joints[j].set_scale(g.STEPS_PER_MM)
        print(f"approach: all to {approach} mm at {speed} sps", flush=True)
        gg.move_together({j: approach for j in joints}, speed)
        for j, irun in IRUN.items():
            if j in joints:
                hh.set_run_current(j, (hh.DEFAULT_IHOLD_IRUN & ~0x1F00) | (irun << 8))
        try:
            print(f"push: all to -{overtravel} mm (grind ~{overtravel * abs(g.STEPS_PER_MM) / speed:.1f}s at the wall)",
                  flush=True)
            gg.move_together({j: -overtravel for j in joints}, speed)
        finally:
            for j in IRUN:
                if j in joints:
                    hh.set_run_current(j, hh.DEFAULT_IHOLD_IRUN)
        g.disable_all()
        if "--no-zero" not in sys.argv:
            for j in joints:
                d.joints[j].zero()
            print("all zeroed at the wall", flush=True)
        gg.check_drivers(joints)
    finally:
        g.disable_all()
        d.close()


if __name__ == "__main__":
    main()
