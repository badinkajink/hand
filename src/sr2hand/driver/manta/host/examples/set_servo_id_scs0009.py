#!/usr/bin/env python3
"""Assign a Feetech SCS0009 servo's bus ID over the U2D2 -- same purpose as
set_servo_id.py, but for SCS0009 (Scs0009PyController) instead of ST3215
(Sts3215PyController). Don't use set_servo_id.py for SCS0009 hardware or
this script for ST3215 -- different register maps, per manta_hand/servos.py's
own comment on why ServoBus doesn't mix the two families.

Only run this with a bus state you've confirmed by pinging first (e.g. a
scan across ids 0-10): if more than one servo currently shares OLD_ID, this
retargets all of them at once, since the Feetech protocol addresses by ID,
not physical position.

Explicitly unlocks EEPROM (Lock register) before writing the id and
re-locks after -- skipping this lets the write appear to succeed in the
current session while silently reverting on the next power cycle, per
set_servo_id.py's own docstring and manta_hand/servos.py's comments on
EEPROM writes in general.

Run from wherever the U2D2 is plugged in (Mac or CB1):
    python3 set_servo_id_scs0009.py --port /dev/ttyUSB0 --old-id 3 --new-id 0
"""

import argparse
import sys
import time

sys.path.insert(0, "..")  # allow running without installing the package

try:
    from rustypot import Scs0009PyController
except ImportError as exc:
    raise ImportError("set_servo_id_scs0009.py requires rustypot: pip install rustypot") from exc

SETTLE_DELAY_S = 1.0  # matches ServoBus.__init__'s post-open settle delay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--old-id", type=int, required=True)
    parser.add_argument("--new-id", type=int, required=True)
    args = parser.parse_args()

    print(f"opening {args.port} @ {args.baudrate}...")
    c = Scs0009PyController(serial_port=args.port, baudrate=args.baudrate, timeout=0.5)
    time.sleep(SETTLE_DELAY_S)

    print(f"pinging id {args.old_id}...")
    if not c.ping(args.old_id):
        raise RuntimeError(
            f"no servo responded at id {args.old_id} -- check wiring/power "
            f"and that nothing else on the bus already holds this id"
        )
    time.sleep(SETTLE_DELAY_S)

    print("unlocking EEPROM...")
    c.write_lock(args.old_id, False)
    time.sleep(SETTLE_DELAY_S)

    print(f"writing id {args.old_id} -> {args.new_id}...")
    c.write_id(args.old_id, args.new_id)
    time.sleep(SETTLE_DELAY_S)

    readback = c.read_id(args.new_id)
    if readback[0] != args.new_id:
        raise RuntimeError(f"wrote id {args.new_id} but read back {readback!r}")

    print("re-locking EEPROM...")
    c.write_lock(args.new_id, True)
    time.sleep(SETTLE_DELAY_S)

    print(f"done -- servo now responds at id {args.new_id}")
    print("power-cycle the servo (unplug/replug) and re-run with --old-id "
          f"{args.new_id} --new-id {args.new_id} to confirm it actually persisted, "
          "not just that this session's readback matched.")


if __name__ == "__main__":
    main()
