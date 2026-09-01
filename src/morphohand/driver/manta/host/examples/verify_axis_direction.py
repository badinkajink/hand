#!/usr/bin/env python3
"""Which sign of MOVEMM moves an axis AWAY from its home hardstop?

Written to settle the J4 sign question, which hand_control.py's REPL could
not answer: its `<finger>_x <mm>` segment is bounds-checked against
kinematics.axis_stepper_range(), and while FULL_EXTENSION_MM[4] was -62.2
that returned (-62.2, 0.0) for J4 -- so the REPL refused every positive
value and accepted only the negative ones that drive into the home
hardstop. Both are fixed now (positive mm is away from home on every
axis), so this is kept as the regression check: re-run it after any
change to STEPS_PER_MM, FULL_EXTENSION_MM, HOME_DIRECTION, or the wiring
of an axis's direction pin. It talks to the Joint directly and so is not
subject to the bounds check it was written to bypass.

Safe by construction: it homes the axis first, so the axis starts pressed
against its home hardstop. From there a command in the "into home"
direction has nowhere to travel and merely buzzes against the stop, while
a command in the "away" direction moves freely into 50+mm of open travel.
Either outcome is harmless -- which is the whole point of testing from
home rather than from mid-travel.

Run the known-good control axis first, then the one in question:

    python3 verify_axis_direction.py --joint 0     # J0, thumb x, known good
    python3 verify_axis_direction.py --joint 4     # J4, middle x, suspect

Both should report the SAME answer. If they disagree, that difference is
the bug.
"""

import argparse
import sys
import time

sys.path.insert(0, "..")  # allow running without installing the package
from manta_hand import MantaHandDriver
from manta_hand.kinematics import STEPS_PER_MM, _home_one_axis

STEPPER_PORT = "/dev/ttyACM0"
PROBE_VELOCITY = 4000  # steps/s -- ~1.2mm/s, slow enough to watch and abort
PROBE_ACCEL = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", type=int, required=True, choices=range(6))
    ap.add_argument("--mm", type=float, default=3.0,
                    help="probe distance, positive (default 3.0)")
    ap.add_argument("--port", default=STEPPER_PORT)
    args = ap.parse_args()

    j_idx = args.joint
    spm = STEPS_PER_MM
    expected_steps = round(args.mm * spm)

    print(f"J{j_idx}: steps_per_mm = {spm}")
    print(f"commanding MOVEMM J{j_idx} {args.mm:+g}  ->  absolute step target "
          f"{expected_steps:+d}")
    print(f"({'negative' if expected_steps < 0 else 'positive'} step target; homing runs the "
          f"count UP into the home stop, so\n negative = away from home, positive = back into it)\n")

    with MantaHandDriver(args.port) as driver:
        j = driver.joints[j_idx]
        j.enable()
        j.set_scale(spm)

        print(f"homing J{j_idx} first (this is what makes the probe safe)...")
        _home_one_axis(driver, j_idx)
        settled = j.status
        print(f"  homed. position = {settled.position} steps (0 = at the home hardstop)\n")

        input(f"About to command {args.mm:+g}mm on J{j_idx}. WATCH THE AXIS, then press Enter "
              f"(Ctrl-C to abort): ")

        j.move_to_mm(args.mm, PROBE_VELOCITY, PROBE_ACCEL)
        start = time.monotonic()
        while time.monotonic() - start < 30.0:
            st = j.status
            if not st.moving:
                break
            time.sleep(0.2)
        else:
            j.stop()
            j.disable()
            raise SystemExit(f"J{j_idx}: probe move didn't finish in 30s -- disabled")

        final = j.status
        print(f"\n  final position = {final.position} steps "
              f"({final.position / spm:+.3f}mm), target was {final.target}")
        j.disable()

    print("\n" + "=" * 68)
    print("Read the RESULT off the hardware, not off the step counter -- the")
    print("counter is open-loop and will report the target either way:")
    print(f"  moved freely away from the stop  -> {args.mm:+g}mm is AWAY from home.")
    print(f"                                      Positive mm is the safe direction.")
    print(f"  sat still / buzzed at the stop   -> {args.mm:+g}mm is INTO home.")
    print("=" * 68)


if __name__ == "__main__":
    main()
