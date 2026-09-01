#!/usr/bin/env python3
"""Re-measure a servo joint's true center and write it into servos.py's FINGER_JOINTS.

    python3 rezero_finger_joint.py --finger 2 --joint aa   # just aa
    python3 rezero_finger_joint.py --finger 2              # all three joints on finger 2

For when a joint's commanded zero (0.0 relative degrees) no longer sits at the joint's
real physical center -- e.g. mechanical play, a re-seated linkage, or a replaced servo
(see servo_calibration_notes.md's "Servo 6 REPLACED" section for the last time this
happened). Frees the target servo(s) so you can center them by hand, reads back the raw
position, and rewrites that joint's zero_deg in servos.py's FINGER_JOINTS IN PLACE --
that dict is still the single source of truth for every joint's zero and range, this
just edits it directly instead of asking a human to copy a printed value in by hand. The
range half of each entry is left untouched; only the zero_deg number changes. This is a
real, tracked source edit -- review the resulting diff and commit it like any other
calibration change (see servo_calibration_notes.md for the measurement history behind
every current value), it just no longer requires manually retyping the number.

Uses Servo.free_raw(), not Servo.free(): servo_calibration_notes.md found TORQUE_FREE=3
does not reliably produce genuine backdrivable behavior on this SCS0009 firmware (a
servo can read back torque_enable=3, present_load=0 -- both looking free -- while still
visibly springing back to a held position). free_raw() writes the raw value 0 instead,
confirmed backdrivable by hand.

After measuring, re-enables torque and holds the joint at the NEWLY measured center (not
the old one) so you can see nothing sprang back -- this uses the value just read, not
whatever was in FINGER_JOINTS before this run, since the whole point is that value may
now be wrong.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from manta_hand import ServoBus
from manta_hand.servos import DEFAULT_JOINT_SPEED, FINGER_JOINTS

SERVOS_PY = Path(__file__).resolve().parent.parent / "manta_hand" / "servos.py"


def ask(prompt: str, yes: bool) -> str:
    if yes:
        print(f"  {prompt} [auto-yes]")
        return ""
    return input(f"  {prompt} ").strip()


def write_zero_deg(joint: str, servo_id: int, new_zero_deg: float) -> None:
    """Rewrite this one joint's zero_deg in servos.py's FINGER_JOINTS in place.

    Matches on the joint name + servo id (unique per servo, 0-8) rather than the old
    zero_deg's exact text, since a float's on-disk formatting (2 decimals for one entry,
    8 for another) doesn't necessarily match what an f-string would produce for that
    same value -- matching structurally instead of on the old number's text is what
    makes this reliable regardless of how the current value happens to be written."""
    text = SERVOS_PY.read_text()
    pattern = re.compile(rf'("{joint}":\s*\(\s*{servo_id}\s*,\s*)[-+]?\d+\.?\d*(\s*,)')
    new_text, n = pattern.subn(rf'\g<1>{new_zero_deg:.4f}\g<2>', text, count=1)
    if n != 1:
        raise RuntimeError(
            f"expected exactly one match for servo {servo_id} ({joint!r}) in {SERVOS_PY}, "
            f"found {n} -- FINGER_JOINTS may have been reformatted; edit it by hand "
            f"instead of trusting this"
        )
    SERVOS_PY.write_text(new_text)


def rezero_one(bus: ServoBus, finger_id: int, joint: str, yes: bool) -> tuple[int, float, float]:
    """Measure one joint AND write it to servos.py before returning -- deliberately not
    batched across joints. A multi-joint run (--finger 2 with no --joint) stops for a
    human between each one, and a human can stop it early (Ctrl-C, or just deciding
    they're done) after any joint they actually wanted -- a batched write would lose
    that joint's already-good measurement along with the ones never reached. Confirmed
    the hard way: a --finger 2 run stopped after aa alone left aa correctly re-centered
    on the servo but NOT yet in FINGER_JOINTS, because the write used to happen only
    after every requested joint finished."""
    servo_id, old_zero_deg, (lo, hi) = FINGER_JOINTS[finger_id][joint]
    servo = bus.servo(servo_id)
    print(f"\nfinger {finger_id} {joint} (servo {servo_id}): recorded zero_deg={old_zero_deg:.4f}, "
          f"range=[{lo:.2f}, {hi:.2f}]")
    servo.free_raw()
    ask(f"servo {servo_id} is backdrivable now -- move finger {finger_id}'s {joint} to its "
        f"true center by hand, then press Enter:", yes)
    new_zero_deg = servo.status.position_deg
    delta = new_zero_deg - old_zero_deg
    print(f"  read back {new_zero_deg:.4f} deg -- delta from recorded zero: {delta:+.4f} deg")
    servo.enable()
    servo.move_to_deg(new_zero_deg, speed=DEFAULT_JOINT_SPEED)
    print(f"  holding at the newly measured center -- confirm by eye nothing sprang back")
    write_zero_deg(joint, servo_id, new_zero_deg)
    print(f"  written to {SERVOS_PY}: zero_deg {old_zero_deg:.4f} -> {new_zero_deg:.4f}")
    return servo_id, old_zero_deg, new_zero_deg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finger", type=int, required=True, choices=sorted(FINGER_JOINTS))
    ap.add_argument("--joint", choices=("aa", "fe1", "fe2"),
                    help="omit to re-zero all three of this finger's joints, in order")
    ap.add_argument("--servo-port", default="/dev/ttyUSB0")
    ap.add_argument("--yes", action="store_true",
                    help="skip the pause and read whatever position the joint is already "
                         "at -- for confirming a value someone already centered by hand, "
                         "not a substitute for actually centering it")
    args = ap.parse_args()

    joints = [args.joint] if args.joint else list(FINGER_JOINTS[args.finger])
    with ServoBus(args.servo_port) as bus:
        results = [(j, *rezero_one(bus, args.finger, j, args.yes)) for j in joints]

    print(f"\n{len(results)}/{len(joints)} joint(s) written to {SERVOS_PY}.")
    print("This is a real source edit -- review the diff and commit it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
