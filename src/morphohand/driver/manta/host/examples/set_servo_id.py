#!/usr/bin/env python3
"""Assign a Feetech ST3215 servo's bus ID over the U2D2 -- a manual pre-step
before wiring up the hand: each servo ships at ID 1 by default and must get
its real ID assigned one servo at a time, alone on an otherwise-empty bus.
If more than one servo on the bus currently shares
OLD_ID, this writes to all of them at once (the Feetech protocol addresses
by ID, not physical position) -- e.g. running this while daisy-chained to
the already-calibrated hand, whose finger-0 servo is already ID 1, would
retarget that servo's ID too, not just the new one.

Explicitly unlocks EEPROM (Lock register, addr 55) before writing the id
and re-locks after -- confirmed on real hardware that skipping this lets
the write appear to succeed (immediate readback matches the new id) while
never actually committing to non-volatile flash: the id silently reverts
to its old value the next time the servo is power-cycled, even though
everything looked fine in the same powered-on session. Same applies to any
other EEPROM-region write (e.g. min_angle_limit/max_angle_limit) -- always
unlock first if you want it to survive a real power cycle, not just a
fresh serial connection while still powered.

Uses rustypot's register-level write_id/read_id directly (Sts3215PyController,
not the manta_hand.servos.ServoBus wrapper -- ServoBus doesn't expose ID
assignment since normal operation never needs it, and it's built for a
different Feetech servo family (SCS0009) with a different register map --
don't reuse it for ST3215 hardware). Same call pattern as AmazingHand's own
reference tool for this (Demo/AHControl/src/bin/change_id.rs):
write_id(old_id, new_id), then a settle delay, then read_id(new_id) to
confirm the write actually stuck -- see manta_hand/servos.py's own comments
on why every write here gets verified by reading it back rather than
trusted on a clean return.

Run from wherever the U2D2 is plugged in (Mac or CB1):
    python3 set_servo_id.py --port /dev/tty.usbserial-XXXX --old-id 1 --new-id 2
"""

import argparse
import sys
import time

sys.path.insert(0, "..")  # allow running without installing the package

try:
    from rustypot import Sts3215PyController
except ImportError as exc:
    raise ImportError("set_servo_id.py requires rustypot: pip install rustypot") from exc

SETTLE_DELAY_S = 1.0  # matches ServoBus.__init__'s post-open settle delay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--old-id", type=int, default=1, help="factory default is 1")
    parser.add_argument("--new-id", type=int, required=True)
    args = parser.parse_args()

    print(f"opening {args.port} @ {args.baudrate}...")
    c = Sts3215PyController(serial_port=args.port, baudrate=args.baudrate, timeout=0.5)
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
