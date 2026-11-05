#!/usr/bin/env python3
"""J0 demo -- the first joint with a confirmed-working end-to-end path
(firmware, UART link, step generation). Sweeps back and forth a few times
at increasing speed, easy to point a camera at.

Only J0 is included here: J1-J7 still have unresolved or known-wrong pin
mappings (see docs/pinout.md's "Per-joint TMC UART status" table) and would
raise MantaHandError the moment a CUR/USTEP/MOVE command came back ERR. Add
a joint to JOINTS below once its entry in that table is fixed and verified.

Run from the CB1: python3 demo.py
"""

import sys
import time

sys.path.insert(0, "..")  # allow running without installing the package
from manta_hand import MantaHandDriver

PORT = "/dev/ttyACM0"
JOINTS = [0]

# Conservative starting point -- replace with your NEMA8's real datasheet
# current rating once you've confirmed basic operation. See docs/bringup.md.
RUN_MA = 300
HOLD_MA = 150
MICROSTEPS = 16

SWEEP_STEPS = 400
ACCEL = 3000
SPEEDS = [300, 600, 1000]  # steps/sec, one full back-and-forth sweep per speed


def wait_until_stopped(joint, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = joint.status
        if not s.moving:
            return s
        time.sleep(0.02)
    raise TimeoutError(f"J{joint.index} never reported stopped")


def sweep(joint, velocity):
    joint.move_to(SWEEP_STEPS, velocity, ACCEL)
    wait_until_stopped(joint)
    joint.move_to(0, velocity, ACCEL)
    wait_until_stopped(joint)


def main():
    with MantaHandDriver(PORT) as hand:
        joints = [hand.joints[i] for i in JOINTS]

        for j in joints:
            print(f"J{j.index}: current={RUN_MA}/{HOLD_MA}mA, microsteps={MICROSTEPS}")
            j.set_current(RUN_MA, HOLD_MA)
            j.set_microsteps(MICROSTEPS)
            j.zero()
            j.enable()

        for velocity in SPEEDS:
            print(f"sweep at {velocity} steps/sec")
            for j in joints:
                sweep(j, velocity)

        for j in joints:
            j.disable()
        print("done")


if __name__ == "__main__":
    main()
