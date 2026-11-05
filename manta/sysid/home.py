#!/usr/bin/env python3
"""Home the six gantry axes.

    home              the fast home (~10 s). Refuses, moving nothing, if the stepper
                      board was reset since the last home -- then run `home --full`
    home --full       the full home (~4 min), safe from any position
    home --check      only say whether the fast home is allowed (moves nothing)
    home --auto       fast if allowed, otherwise full (what hand_control.py runs at startup)
(`home` is ~/.local/bin/home on the CB1, which runs this file under a 330 s timeout.)

Full home: the original firmware home that hand_control.py runs at startup
(Gantry.home_all): one axis at a time, out to 10 mm, then HOME into the wall.
At 24 V StallGuard doesn't trigger, so each axis grinds at the wall until its
full-travel timeout and is zeroed there (~221 s in all; zeros agree with the
StallGuard walls to ~0.15 mm, 2026-11-04). Works from any position.

Fast home: push_home_all.py. All six to 3 mm together, then all pushed 1.5 mm
past their walls at low current, zeroed (~9 s, +-0.15 mm). Only valid while the
step counts are right to within 1.5 mm, i.e. the STM32 hasn't reset since the
last home.

How it knows: after every successful home this stores a RAM-only marker on J7,
which has no driver chip (SETSCALE J7 <MARKER>). The STM32 loses it on any reset
(power cycle, DFU reset, crash) -- exactly when it also loses the step counts.
The probe `MOVEMM J7 0 ...` answers OK if the marker is there and
ERR UNCALIBRATED if not; J7 sits at step 0, so the probe moves nothing.
The marker is cleared before homing starts, so an interrupted home forces a
full home next time. Moving a carriage by hand while disabled is NOT detected:
run `home --full` after that.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, HERE)
from manta_hand import MantaHandDriver  # noqa: E402
from manta_hand.protocol import MantaHandError  # noqa: E402

PORT = "/dev/ttyACM0"
MARKER_JOINT = "J7"
MARKER = "7.0"  # any nonzero steps/mm; nothing else uses J7
JOINTS = range(6)


def marker_present(d) -> bool:
    try:
        d.send("MOVEMM", MARKER_JOINT, "0", "1000", "1000")
        return True
    except MantaHandError as e:
        if "UNCALIBRATED" in str(e):
            return False
        raise


def set_marker(d, on: bool):
    d.send("SETSCALE", MARKER_JOINT, MARKER if on else "0")


def disable_all(d):
    for j in JOINTS:
        try:
            d.joints[j].disable()
        except Exception as e:  # keep going: disabling the rest matters more
            print(f"  could not disable J{j}: {e}", flush=True)


def full_home(d):
    from manta_hand.kinematics import Gantry
    print("FULL home: one axis at a time into its wall (~4 min, grinding at the wall is expected)", flush=True)
    for o in Gantry(d).home_all():
        how = "stall" if o["stalled"] else "timeout"
        print(f"  J{o['joint']}: {how}, {o['elapsed_s']:.0f} s", flush=True)


def fast_home():
    print("FAST home: all six pushed into their walls together (~10 s)", flush=True)
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "push_home_all.py")],
                       capture_output=True, text=True)
    sys.stdout.write("".join("  " + line + "\n" for line in r.stdout.splitlines()))
    if r.returncode != 0 or "driver check: all OK" not in r.stdout:
        sys.stderr.write(r.stderr)
        raise RuntimeError("push home failed -- run `home --full`")


def main():
    args = set(sys.argv[1:])
    if args - {"--full", "--check", "--auto"} or len(args) > 1:
        sys.exit(__doc__)
    d = MantaHandDriver(PORT)
    try:
        st = d.get_all_status()
        if any(st[j].moving for j in JOINTS):
            sys.exit("an axis is moving -- nothing done")
        trusted = marker_present(d)
        print(f"step counts {'kept since the last home' if trusted else 'NOT trusted (board reset since the last home)'}",
              flush=True)
        if "--check" in args:
            print("`home` (fast) is OK" if trusted else "`home` will refuse: run `home --full`")
            return
        mode = "full" if ("--full" in args or ("--auto" in args and not trusted)) else "fast"
        if mode == "fast" and not trusted:
            sys.exit("nothing moved: the fast home needs trusted step counts -- run `home --full` "
                     "(~4 min, grinds each axis at its wall)")
        t0 = time.monotonic()
        moved = True
        set_marker(d, False)  # an interrupted home must not look trusted
        if mode == "full":
            full_home(d)
            disable_all(d)
        else:
            d.close()  # push_home_all opens the port itself
            try:
                fast_home()
            finally:
                d = MantaHandDriver(PORT)
        set_marker(d, True)
        print(f"homed ({mode}) in {time.monotonic() - t0:.0f} s -- all axes at 0 mm, disabled", flush=True)
    finally:
        if "moved" in locals():
            disable_all(d)
        d.close()


if __name__ == "__main__":
    main()
