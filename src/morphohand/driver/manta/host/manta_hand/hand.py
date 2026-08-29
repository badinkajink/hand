"""Unified per-finger API combining the stepper gantry (Gantry/GantryFinger,
kinematics.py) and the servo aa/fe1/fe2 joints (ServoBus/Finger,
servos.py) behind three calls: move_to_local, move_to_global, and home.

x/y (stepper gantry position) and aa/fe1/fe2 (servo joint angles) are all
optional and independent -- a call can move just the gantry, just one or
more joints, or both together. Only x/y differs between move_to_local and
move_to_global (the local<->global {P} translation, see kinematics.py);
aa/fe1/fe2 behave identically in both since they have no position frame of
their own.

The stepper move and servo move(s) are issued back-to-back on their own
separate serial links (steppers on the Manta M8P's USB-CDC, servos on the
U2D2's USB-serial) -- this is NOT a single synchronized transaction across
both buses, since they're physically different links.
"""

from __future__ import annotations

from .driver import MantaHandDriver
from .kinematics import STEPPER_ACCEL, STEPPER_VELOCITY, Gantry, GantryFinger
from .servos import DEFAULT_JOINT_SPEED, Finger, ServoBus


class HandFinger:
    """One finger's combined stepper gantry + servo joints. Built by
    Hand.finger(), not constructed directly."""

    def __init__(self, gantry_finger: GantryFinger, servo_finger: Finger):
        self._gantry = gantry_finger
        self._servo = servo_finger

    def _apply_joints(self, aa, fe1, fe2, speed):
        for name, val in (("aa", aa), ("fe1", fe1), ("fe2", fe2)):
            if val is not None:
                self._servo.set_joint(name, val, speed=speed)

    def move_to_local(self, x_mm: float | None = None, y_mm: float | None = None,
                       aa: float | None = None, fe1: float | None = None, fe2: float | None = None,
                       velocity: int = STEPPER_VELOCITY, accel: int = STEPPER_ACCEL,
                       speed: int = DEFAULT_JOINT_SPEED):
        """x_mm/y_mm: this finger's own local (q_Fx, q_Fy) frame, origin at
        its box center. aa/fe1/fe2: plain degrees relative to that servo's
        own zero. Every parameter is optional and independent -- pass only
        what you want to move. velocity/accel/speed all have working
        defaults; override only when you need to. Assumes this finger has
        already been homed this session (see .home())."""
        if x_mm is not None or y_mm is not None:
            if x_mm is None or y_mm is None:
                raise ValueError("move_to_local: x_mm and y_mm must both be given together")
            self._gantry.move_to_local(x_mm, y_mm, velocity=velocity, accel=accel)
        self._apply_joints(aa, fe1, fe2, speed)

    def move_to_global(self, x_mm: float | None = None, y_mm: float | None = None,
                        aa: float | None = None, fe1: float | None = None, fe2: float | None = None,
                        velocity: int = STEPPER_VELOCITY, accel: int = STEPPER_ACCEL,
                        speed: int = DEFAULT_JOINT_SPEED):
        """Same as move_to_local, but x_mm/y_mm are in the shared
        palm-fixed global frame {P} -- converted to this finger's local
        frame (simple translation by its origin) before delegating."""
        if x_mm is not None or y_mm is not None:
            if x_mm is None or y_mm is None:
                raise ValueError("move_to_global: x_mm and y_mm must both be given together")
            self._gantry.move_to_global(x_mm, y_mm, velocity=velocity, accel=accel)
        self._apply_joints(aa, fe1, fe2, speed)

    def home(self, velocity: int = STEPPER_VELOCITY, accel: int = STEPPER_ACCEL,
             speed: int = DEFAULT_JOINT_SPEED):
        """Home this finger's two stepper axes (StallGuard2, sequential x
        then y), then return its three servos to their zero references.
        Steppers first, then servos."""
        self._gantry.home(velocity=velocity, accel=accel)
        self._servo.zero_joints(speed=speed)


class Hand:
    """The whole hand: 3 fingers, each with a stepper gantry + 3 servo
    joints. Gantry setup (enable + scale) happens internally in __init__ --
    the caller never touches that."""

    def __init__(self, stepper_driver: MantaHandDriver, servo_bus: ServoBus):
        self._gantry = Gantry(stepper_driver)  # enables + sets scale internally
        self._servo_bus = servo_bus

    def finger(self, finger_id: int) -> HandFinger:
        return HandFinger(self._gantry.finger(finger_id), self._servo_bus.finger(finger_id))

    def home_all(self, velocity: int = STEPPER_VELOCITY, accel: int = STEPPER_ACCEL,
                 speed: int = DEFAULT_JOINT_SPEED):
        """Home every stepper axis (sequentially, all 6) then zero every
        finger's servos."""
        self._gantry.home_all(velocity=velocity, accel=accel)
        for finger_id in range(3):
            self._servo_bus.finger(finger_id).zero_joints(speed=speed)

    def set_joints_fast(self, pose: dict[int, dict[str, float]], speed: int | None = None):
        """Real-time-capable aa/fe1/fe2 update across any number of
        fingers in ONE bus transaction (~0.4ms for all 9 servos) -- the
        control-loop path (e.g. a grasping policy), NOT HandFinger's
        move_to_local/move_to_global (~0.9s per joint via their
        write-verify path, fine for one-off setup moves, unusable in a
        loop). Only touches servos -- steppers are assumed already
        positioned (e.g. once per session via move_to_local/global) and
        untouched here. See ServoBus.sync_set_joints for the exact
        contract (bounds-checked, all-or-nothing, not write-verified) --
        including why `speed` isn't optional the first time you drive a
        given joint through this path (home_all()/HandFinger.home() already
        set it for every joint via zero_joints(), so you only need to pass
        it here for a joint you're driving before that's run this session)."""
        self._servo_bus.sync_set_joints(pose, speed=speed)
