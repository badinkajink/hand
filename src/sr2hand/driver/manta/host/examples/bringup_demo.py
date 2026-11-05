#!/usr/bin/env python3
"""First-contact bring-up script: one joint at a time, at low current, small
moves, so you can watch each motor and confirm wiring/direction before
trusting anything else. See docs/bringup.md for the full checklist this is
part of.

Run from the CB1: python3 bringup_demo.py
"""

import sys
import time

sys.path.insert(0, "..")  # allow running without installing the package
from manta_hand import MantaHandDriver

PORT = "/dev/ttyACM0"

# Conservative starting point -- REPLACE with your NEMA8's actual rated
# current before doing anything beyond this small sanity-check move.
RUN_MA = 300
HOLD_MA = 150

STEP_TEST = 200      # small, ~1/16 turn at 16 microsteps on a typical NEMA8
VELOCITY = 400
ACCEL = 2000


def main():
    with MantaHandDriver(PORT) as hand:
        print("STATALL before touching anything:")
        for i, s in enumerate(hand.get_all_status()):
            print(f"  J{i}: {s}")

        for joint in hand.joints:
            print(f"\n--- Joint {joint.index} ---")
            joint.set_current(RUN_MA, HOLD_MA)
            joint.set_microsteps(16)
            joint.enable()
            joint.zero()

            print("  moving +200 steps...")
            joint.move_to(STEP_TEST, VELOCITY, ACCEL)
            _wait_until_stopped(joint)

            print("  moving back to 0...")
            joint.move_to(0, VELOCITY, ACCEL)
            _wait_until_stopped(joint)

            joint.disable()
            print(f"  J{joint.index} done -- confirm the motion direction matched what you expected.")


def _wait_until_stopped(joint, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not joint.status.moving:
            return
        time.sleep(0.05)
    raise TimeoutError(f"J{joint.index} never reported stopped")


if __name__ == "__main__":
    main()
