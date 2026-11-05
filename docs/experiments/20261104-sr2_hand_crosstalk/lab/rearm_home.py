"""Give the station back a home without homing. Runs on the CB1 with the station stopped (it holds /dev/ttyACM0).

The station moves nothing until it is homed, and its /api/v1/home/adopt takes over a home only when the board vouches
for one: every axis enabled, stopped, homing_result 2 or 3. A disable clears homing_result, so after every stop that
disabled the motors the station homed again: about 4 min, each axis grinding into its hardstop until its timeout
(StallGuard2 fires on none of them). When every axis's step count still matches where its stage is, this script
re-arms instead. It prints each axis's TMC5160 GSTAT (latched driver shutdowns, cleared by the read) and DRV_STATUS,
enables every axis, zeroes the counts of the axes named with --zero (stages known to sit at home with a wrong count),
and starts a firmware HOME of 1 ms on every axis. That sets homing_result 3 and moves nothing. Then start the station
and POST /api/v1/home/adopt; the station takes the mounts from the counts.

    python3 rearm_home.py [--zero 4] [--read-only]

Used at 13:4x on 11-05 (the owner: "please dont home ... everything is already good"): joint 4's stage sat at home while
its count said 45 mm, after a layout move it did not follow. Valid only while no axis was moved by hand or lost steps.
"""
import argparse
import time

import serial

ap = argparse.ArgumentParser()
ap.add_argument("--zero", type=int, nargs="*", default=[], help="axes whose stage sits at home")
ap.add_argument("--read-only", action="store_true")
a = ap.parse_args()
s = serial.Serial("/dev/ttyACM0", 115200, timeout=2)
time.sleep(0.3)


def cmd(line):
    s.reset_input_buffer()
    s.write((line + "\n").encode())
    return s.readline().decode(errors="replace").strip()


for j in range(6):
    print(f"J{j} GSTAT {cmd(f'RREG5160 J{j} 01')}  DRV_STATUS {cmd(f'RREG5160 J{j} 6F')}  STAT {cmd(f'STAT J{j}')}")
if a.read_only:
    raise SystemExit(0)
before = {j: cmd(f"STAT J{j}") for j in range(6)}
for j in range(6):
    print(f"EN J{j}", cmd(f"EN J{j}"))
for j in a.zero:
    print(f"ZERO J{j}", cmd(f"ZERO J{j}"))
for j in range(6):
    print(f"HOME J{j} (1 ms)", cmd(f"HOME J{j} 1 1000 2000 1"))
time.sleep(0.5)
for j in range(6):
    print(f"J{j} before {before[j]:30s} after {cmd(f'STAT J{j}')}")
