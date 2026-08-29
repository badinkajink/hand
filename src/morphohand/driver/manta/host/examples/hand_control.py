#!/usr/bin/env python3
"""Simple text commands for controlling the hand -- both the stepper joints
(Manta M8P, via MantaHandDriver) and the servo fingers (U2D2, via ServoBus)
from one interactive shell.

Commands, one per comma-separated segment on a line -- stepper moves are
issued one at a time; any servo joint segments (aa/fe1/fe2) on the same
line land as a single synchronized move via ServoBus.sync_set_joints:

    <finger>_x <mm>    first stepper, absolute position in mm (0 = home)
    <finger>_y <mm>    carriage stepper, absolute position in mm (0 = home)
    <finger>_aa <deg>  adduction/abduction servo, degrees relative to this
                       joint's calibrated zero
    <finger>_fe1 <deg> proximal phalanx flexion servo, degrees relative to zero
    <finger>_fe2 <deg> distal phalanx flexion servo, degrees relative to zero

aa/fe1/fe2 are each bounds-checked against this finger's calibrated range
(see FINGER_JOINTS in manta_hand/servos.py) before anything moves.

<finger> is 0, 1, or 2 -- or its name alias, interchangeably: thumb=0,
index=1, middle=2. Stepper joint mapping (J0-J5 on the Manta M8P):
    finger 0 / thumb:  x=J0, y=J1
    finger 1 / index:  x=J2, y=J3
    finger 2 / middle: x=J4, y=J5

Examples:
    0_x 20, 0_y 15
    thumb_x 20, thumb_y 15
    thumb_fe1 40, thumb_fe2 30
    0_x 30, 0_y 40, 0_fe1 20, 0_aa 5

Two ways to run it:

  Interactive (REPL) -- type commands, see them happen immediately, 'quit'
  or Ctrl-D to exit:
      python3 hand_control.py

  One-shot -- pass the command line as arguments, it runs once and exits
  (useful for scripting):
      python3 hand_control.py "thumb_fe1 40"

STEPPER_PORT and SERVO_PORT are two completely separate physical USB
connections (Manta M8P's USB-CDC vs. the U2D2's USB-serial) -- nothing
about one goes through the other.

Note on coordinate systems: the `<finger>_x`/`<finger>_y` commands here are
firmware-space (raw stepper mm, 0 = wherever that axis last homed to) --
a completely different coordinate system from `manta_hand.Hand`'s
`move_to_local`/`move_to_global` (a finger's own local frame, or the
shared palm-fixed global frame {P}). Both address the same hardware; don't
mix mm values between them without going through the transform in
`manta_hand.kinematics`.
"""

import re
import sys

sys.path.insert(0, "..")  # allow running without installing the package
from manta_hand import MantaHandDriver, ServoBus, Hand
from manta_hand.servos import FINGER_JOINTS
from manta_hand.kinematics import (
    STEPPER_ACCEL,
    STEPPER_JOINTS,
    STEPPER_VELOCITY,
    axis_stepper_range,
)

STEPPER_PORT = "/dev/ttyACM0"
SERVO_PORT = "/dev/ttyUSB0"

# finger 0/1/2 can also be addressed by these names, e.g. "thumb_x 30" or
# "thumb_fe1 40" -- interchangeable with the numeric id everywhere.
FINGER_NAMES = {"thumb": 0, "index": 1, "middle": 2}
_FINGER_TOKEN = r"(\d+|thumb|index|middle)"

_AXIS_CMD_RE = re.compile(rf"{_FINGER_TOKEN}_(x|y|aa|fe1|fe2)\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def _resolve_finger_id(token: str) -> int:
    """Accepts either a numeric finger id or one of FINGER_NAMES' aliases
    (case-insensitive), always returning the numeric id -- the rest of the
    parser and everything downstream only ever deals in numeric ids."""
    if token.isdigit():
        finger_id = int(token)
    else:
        finger_id = FINGER_NAMES[token.lower()]
    _check_finger(finger_id)
    return finger_id


class Command:
    """Base for the different things a parsed segment can turn into. Kept
    as small data (not executed during parsing) so a whole line is fully
    parsed -- and can fail loudly on a bad segment -- before anything
    physically moves."""


class StepperCmd(Command):
    def __init__(self, finger_id: int, axis: str, mm: float):
        self.finger_id = finger_id
        self.axis = axis  # "x" or "y"
        self.mm = mm


class ServoJointCmd(Command):
    def __init__(self, finger_id: int, joint: str, deg: float):
        self.finger_id = finger_id
        self.joint = joint  # "aa", "fe1", or "fe2"
        self.deg = deg  # relative to this joint's calibrated zero


def parse_line(line: str) -> list[Command]:
    commands: list[Command] = []
    for segment in line.split(","):
        segment = segment.strip()
        if not segment:
            continue

        m = _AXIS_CMD_RE.match(segment)
        if m:
            finger_id = _resolve_finger_id(m.group(1))
            axis = m.group(2).lower()
            value = float(m.group(3))
            if axis in ("x", "y"):
                joint_index = STEPPER_JOINTS[finger_id][0 if axis == "x" else 1]
                lo, hi = axis_stepper_range(joint_index)
                if not lo <= value <= hi:
                    raise ValueError(
                        f"{finger_id}_{axis} = {value}mm is outside J{joint_index}'s "
                        f"measured range [{lo}, {hi}]mm"
                    )
                commands.append(StepperCmd(finger_id, axis, value))
            else:  # aa, fe1, fe2
                _servo_id, _zero_deg, (lo, hi) = FINGER_JOINTS[finger_id][axis]
                if not lo <= value <= hi:
                    raise ValueError(
                        f"{finger_id}_{axis} = {value} is outside this joint's "
                        f"calibrated range [{lo}, {hi}]deg"
                    )
                commands.append(ServoJointCmd(finger_id, axis, value))
            continue

        raise ValueError(f"couldn't parse {segment!r}")
    return commands


def _check_finger(finger_id: int):
    if finger_id not in FINGER_JOINTS:
        raise ValueError(f"no finger {finger_id} (have {list(FINGER_JOINTS)})")


def run_line(hand: MantaHandDriver, bus: ServoBus, line: str):
    line = line.strip()
    if not line:
        return
    commands = parse_line(line)  # parse everything before moving anything
    if not commands:
        return

    # finger_id -> {joint_name: deg}, combined into one sync_set_joints call
    # -- any number of aa/fe1/fe2 segments across any number of fingers on
    # the same line land as a single synchronized move, not separate ones.
    servo_pose: dict[int, dict[str, float]] = {}
    for cmd in commands:
        if isinstance(cmd, StepperCmd):
            joint_index = STEPPER_JOINTS[cmd.finger_id][0 if cmd.axis == "x" else 1]
            hand.joints[joint_index].move_to_mm(cmd.mm, STEPPER_VELOCITY, STEPPER_ACCEL)
            print(f"  finger {cmd.finger_id}_{cmd.axis} (J{joint_index}) -> {cmd.mm}mm")
        elif isinstance(cmd, ServoJointCmd):
            servo_pose.setdefault(cmd.finger_id, {})[cmd.joint] = cmd.deg

    if servo_pose:
        bus.sync_set_joints(servo_pose)
        print("  " + ", ".join(
            f"finger {fid} " + " ".join(f"{joint}={deg}" for joint, deg in joints.items())
            for fid, joints in servo_pose.items()
        ))


def main():
    with MantaHandDriver(STEPPER_PORT) as stepper_driver, ServoBus(SERVO_PORT) as bus:
        for servo_id in range(9):  # all 9 real servos (aa/fe1/fe2 x 3 fingers)
            bus.servo(servo_id).enable()

        hand = Hand(stepper_driver, bus)  # enables + scales all 6 steppers internally

        try:
            hand.home_all()  # homes all 6 steppers, then zeros all 9 servos
            if len(sys.argv) > 1:
                run_line(stepper_driver, bus, " ".join(sys.argv[1:]))
            else:
                print(__doc__)
                while True:
                    try:
                        line = input("> ")
                    except EOFError:
                        print()
                        break
                    if line.strip().lower() in ("quit", "exit"):
                        break
                    try:
                        run_line(stepper_driver, bus, line)
                    except ValueError as e:
                        print(f"  error: {e}")
        finally:
            for servo_id in range(9):
                bus.servo(servo_id).disable()
            for x_joint, y_joint in STEPPER_JOINTS.values():
                stepper_driver.joints[x_joint].disable()
                stepper_driver.joints[y_joint].disable()


if __name__ == "__main__":
    main()
