#!/usr/bin/env python3
"""Single-joint bring-up test -- mirrors the manual `screen` sequence from
docs/bringup.md step 6, for one joint at a time. Edit JOINT/RUN_MA/HOLD_MA
below and re-run as needed.

Run from the CB1: python3 single_joint_test.py
"""

import sys
import time

sys.path.insert(0, "..")  # allow running without installing the package
from manta_hand import MantaHandDriver

PORT = "/dev/ttyACM0"
JOINT = 0  # 0 = J0/M1 -- PC13 (J0's UART pin) has no hardware UART capability
           # on this chip at all, so it's now driven by the software
           # bit-banged UART path in tmc2209_uart.c; see docs/pinout.md's
           # "Per-joint TMC UART status" section.

# Conservative starting point -- replace with your NEMA8's real datasheet
# current rating once basic operation (direction, movement) is confirmed.
RUN_MA = 300
HOLD_MA = 150

MICROSTEPS = 16
STEP_TARGET = 200
VELOCITY = 400
ACCEL = 2000


def wait_until_stopped(joint, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = joint.status
        if not s.moving:
            return s
        time.sleep(0.05)
    raise TimeoutError(f"J{joint.index} never reported stopped")


def main():
    with MantaHandDriver(PORT) as hand:
        j = hand.joints[JOINT]

        print(f"Setting current: run={RUN_MA}mA hold={HOLD_MA}mA")
        j.set_current(RUN_MA, HOLD_MA)

        print(f"Setting microsteps: {MICROSTEPS}")
        j.set_microsteps(MICROSTEPS)

        print("Enabling driver")
        j.enable()

        print(f"Moving to {STEP_TARGET} steps...")
        j.move_to(STEP_TARGET, VELOCITY, ACCEL)
        s = wait_until_stopped(j)
        print(f"  stopped at: {s}")

        print("Moving back to 0...")
        j.move_to(0, VELOCITY, ACCEL)
        s = wait_until_stopped(j)
        print(f"  stopped at: {s}")

        j.disable()
        print("Done -- disabled. Confirm direction/behavior matched expectations.")


if __name__ == "__main__":
    main()
