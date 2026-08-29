#!/usr/bin/env python3
"""One example per movement option the manta_hand API supports, each a
standalone, copy-pasteable function. Unlike verify_api_demo.py (which runs
a fixed sequence end-to-end as a bring-up smoke test), this file is a
reference: pick exactly one movement to try, by name.

    python3 movement_examples.py --list
    python3 movement_examples.py home_all
    python3 movement_examples.py move_gantry_local

Needs both the Manta M8P (steppers, USB-CDC, STEPPER_PORT) and the U2D2
(servos, USB-serial, SERVO_PORT) connected -- edit the two constants below
to match your machine. See host/README.md and docs/bringup.md for wiring.

Every demo below calls Hand.home_all() first unless noted otherwise --
move_to_mm/move_to_local/move_to_global are only meaningful once an axis's
step-0 has a real physical reference (see manta_hand.joint.Joint.move_to_mm).
"""

import sys
import time

sys.path.insert(0, "..")  # allow running without installing the package
from manta_hand import Hand, MantaHandDriver, ServoBus

STEPPER_PORT = "/dev/ttyACM0"
SERVO_PORT = "/dev/ttyUSB0"

DEMO_FINGER = 0  # thumb -- origin (-50, 0) in {P}, local box roughly x:[-30,+26.2] y:[-55,+55]


# --------------------------------------------------------------------------
# Homing
# --------------------------------------------------------------------------

def home_all(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """Home every stepper axis (all 6, sequentially) then zero every
    finger's servos -- the one call to run at the start of a session."""
    hand.home_all()


def home_one_finger(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """Home just one finger's two stepper axes + zero its 3 servos,
    leaving the other two fingers untouched."""
    hand.finger(DEMO_FINGER).home()


# --------------------------------------------------------------------------
# Gantry (stepper) moves -- high-level, mm, two coordinate frames
# --------------------------------------------------------------------------

def move_gantry_local(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """Position a finger's gantry in ITS OWN local (q_Fx, q_Fy) frame,
    origin at that finger's box center. Raises ValueError (no hardware
    command sent) if the target is outside this finger's real measured
    range -- see kinematics.py's FULL_EXTENSION_MM."""
    hand.home_all()
    hand.finger(DEMO_FINGER).move_to_local(x_mm=10.0, y_mm=-20.0)


def move_gantry_global(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """Position a finger's gantry in the shared palm-fixed {P} frame
    instead -- converted to that finger's local frame internally, same
    bounds check applies. For the thumb (origin (-50, 0) in {P}), global
    (-40, -20) is the same physical point as local (10, -20) above."""
    hand.home_all()
    hand.finger(DEMO_FINGER).move_to_global(x_mm=-40.0, y_mm=-20.0)


def move_gantry_raw_steps(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """Bypass mm/kinematics entirely: absolute step position on one raw
    Joint, trapezoidal profile. Useful for debugging the wire protocol
    itself, not for normal finger positioning."""
    joint = stepper_driver.joints[0]  # J0 == finger 0 (thumb) x-axis
    joint.enable()
    joint.zero()
    joint.move_to(steps=2000, velocity=6000, accel=2000)


def move_gantry_raw_mm(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """One raw Joint's move_to_mm -- what move_to_local/global build on
    top of, before the per-finger local<->global transform is applied.
    Needs set_scale() first (Gantry.__init__ already does this for every
    joint when you construct a Hand, so it's redundant here -- shown for
    when you're using MantaHandDriver directly, without Hand)."""
    joint = stepper_driver.joints[0]
    joint.set_scale(-3216.0)  # STEPS_PER_MM[0] -- see kinematics.py
    joint.move_to_mm(mm=20.0, velocity=12000, accel=2000)


def jog_gantry_axis(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """Continuous open-loop velocity move (no target position) -- jog
    until stopped. Mainly for manual jogging during bring-up, not normal
    operation. jog(0, accel) ramps smoothly to a stop."""
    joint = stepper_driver.joints[0]
    joint.jog(velocity=2000, accel=2000)
    time.sleep(1.0)
    joint.jog(velocity=0, accel=2000)  # ramp to stop


def stop_and_disable_gantry_axis(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """Emergency-style stop of one axis: abort any in-progress move, then
    cut motor current. driver.stop_all() (see stop_all_and_disable below)
    does this for all 8 joints at once."""
    joint = stepper_driver.joints[0]
    joint.stop()
    joint.disable()


# --------------------------------------------------------------------------
# Servo joints -- one dedicated servo per DOF (aa/fe1/fe2)
# --------------------------------------------------------------------------

def move_servo_joints_via_hand(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """Servo-only move through the same Hand/HandFinger call used for
    gantry moves -- x_mm/y_mm are simply omitted, so only aa/fe1/fe2 move.
    aa/fe1/fe2 are each optional and independent; pass any subset."""
    hand.finger(DEMO_FINGER).move_to_local(aa=10.0, fe1=45.0, fe2=30.0)


def move_single_servo_joint(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """Lower-level: move exactly one joint (aa, fe1, or fe2) on one
    finger, in plain degrees relative to that joint's own calibrated zero
    (see FINGER_JOINTS in servos.py for each joint's real range)."""
    servo_bus.finger(DEMO_FINGER).set_joint("fe1", 45.0, speed=80)


def zero_finger_servo_joints(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """Return one finger's aa/fe1/fe2 servos to their zero references
    without touching its gantry -- what HandFinger.home() does after
    homing the steppers, callable on its own."""
    servo_bus.finger(DEMO_FINGER).zero_joints()


def move_gantry_and_servos_together(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """One call, one finger, both subsystems: gantry position AND joint
    angles in a single move_to_local/move_to_global. Still two separate
    serial transactions under the hood (steppers on the M8P, servos on
    the U2D2) -- not a synchronized single move, just one convenient call."""
    hand.home_all()
    hand.finger(DEMO_FINGER).move_to_global(x_mm=-40.0, y_mm=-20.0, aa=0.0, fe1=20.0, fe2=15.0)


def move_all_joints_realtime(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """The control-loop path: every finger's aa/fe1/fe2 in ONE bus
    transaction (~0.4ms for all 9 servos), for anything that needs to send
    a new pose every frame (e.g. a grasping policy). NOT write-verified and
    NOT what move_to_local/global use (those are ~0.9s/joint, fine for
    one-off setup moves, unusable in a loop). Steppers are assumed already
    positioned and are untouched here.

    Passes speed on this first call -- confirmed on real hardware that a
    joint whose goal_speed was never set (or was left at some earlier
    value) can silently fail to move, or move only partway, on a plain
    sync position write with no error. home_all()/HandFinger.home()
    already set every joint's speed via zero_joints(), so speed is only
    needed here if you're driving a joint this way before that's run."""
    pose = {
        0: {"aa": 0.0, "fe1": 10.0, "fe2": 5.0},
        1: {"fe1": 20.0},
        2: {"aa": -5.0, "fe2": 10.0},
    }
    hand.set_joints_fast(pose, speed=80)


# --------------------------------------------------------------------------
# Direct single-servo access (below the named-joint scheme)
# --------------------------------------------------------------------------

def move_raw_servo_by_id(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """Bypass Finger/FINGER_JOINTS entirely: one servo, absolute degrees.
    Useful for calibration or anything a named joint doesn't cover."""
    servo_bus.servo(0).enable()
    servo_bus.servo(0).move_to_deg(15.0, speed=200)


def free_servo_for_manual_posing(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """Torque off but still readable -- backdrivable by hand. Used to
    manually pose a finger while reading back .status to find calibration
    endpoints (see servo_calibration_notes.md)."""
    servo_bus.servo(0).free()
    time.sleep(2.0)
    print("servo 0 position:", servo_bus.servo(0).status)


# --------------------------------------------------------------------------
# Status / stop
# --------------------------------------------------------------------------

def read_joint_status(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """Per-joint stepper status: position/target (steps), moving, enabled,
    homing_result. driver.get_all_status() returns this for all 8 joints
    in one STATALL round trip instead of 8 separate STAT calls."""
    print(stepper_driver.joints[0].status)
    print(stepper_driver.get_all_status())


def read_servo_status(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """Raw servo angle (calibration offset NOT subtracted)."""
    print(servo_bus.servo(0).status)


def stop_all_and_disable(hand: Hand, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
    """Abort every in-progress stepper move and cut all 8 joints' motor
    current in one command -- the panic button."""
    stepper_driver.stop_all()


# --------------------------------------------------------------------------

DEMOS = {
    "home_all": home_all,
    "home_one_finger": home_one_finger,
    "move_gantry_local": move_gantry_local,
    "move_gantry_global": move_gantry_global,
    "move_gantry_raw_steps": move_gantry_raw_steps,
    "move_gantry_raw_mm": move_gantry_raw_mm,
    "jog_gantry_axis": jog_gantry_axis,
    "stop_and_disable_gantry_axis": stop_and_disable_gantry_axis,
    "move_servo_joints_via_hand": move_servo_joints_via_hand,
    "move_single_servo_joint": move_single_servo_joint,
    "zero_finger_servo_joints": zero_finger_servo_joints,
    "move_gantry_and_servos_together": move_gantry_and_servos_together,
    "move_all_joints_realtime": move_all_joints_realtime,
    "move_raw_servo_by_id": move_raw_servo_by_id,
    "free_servo_for_manual_posing": free_servo_for_manual_posing,
    "read_joint_status": read_joint_status,
    "read_servo_status": read_servo_status,
    "stop_all_and_disable": stop_all_and_disable,
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    if sys.argv[1] == "--list":
        for name, fn in DEMOS.items():
            summary = (fn.__doc__ or "").strip().splitlines()[0]
            print(f"  {name:<36} {summary}")
        sys.exit(0)
    name = sys.argv[1]
    if name not in DEMOS:
        print(f"unknown demo {name!r}. See --list.", file=sys.stderr)
        sys.exit(1)

    with MantaHandDriver(STEPPER_PORT) as stepper_driver, ServoBus(SERVO_PORT) as servo_bus:
        for sid in range(9):
            servo_bus.servo(sid).enable()
        hand = Hand(stepper_driver, servo_bus)  # enables + sets stepper scale internally

        print(f"=== {name} ===")
        DEMOS[name](hand, stepper_driver, servo_bus)
        print("done")


if __name__ == "__main__":
    main()
