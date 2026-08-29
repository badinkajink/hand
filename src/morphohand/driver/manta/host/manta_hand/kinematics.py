"""Stepper gantry coordinate transform -- per-finger local frame and the
shared palm-fixed global frame {P}, both confirmed against a hand-drawn
diagram and live hardware this session (see
host/examples/servo_calibration_notes.md's "Global coordinate system {P}"
section for the original derivation).

{P} is a top-down, axis-aligned (no rotation) frame at the palm: +x right,
+y up. Each finger has its own axis-aligned local frame (q_Fx, q_Fy),
centered on that finger's own origin relative to {P}. The firmware's
stepper "x"/"y" joint labels are SWAPPED relative to the local q_Fx/q_Fy
axes -- firmware "x" tracks local q_Fy (vertical), firmware "y" tracks
local q_Fx (horizontal) -- confirmed via the thumb's asymmetric 60x110mm
box matching the given firmware stepper ranges (the 110mm clearly belongs
to the taller, q_Ty, dimension). The sign of each relationship additionally
differs per finger, based on which corner of that finger's box its stepper
home position sits at (home = stepper 0mm, and a stepper's mm value always
increases moving away from wherever it homed).

None of this touches servos.py/rustypot -- this module is stepper-only and
importable unconditionally, same tier as driver.py/joint.py.
"""

from __future__ import annotations

import time

from .driver import MantaHandDriver

# finger_id -> (x_joint_index, y_joint_index) on the Manta M8P (J0-J5).
STEPPER_JOINTS = {0: (0, 1), 1: (2, 3), 2: (4, 5)}

# joint index -> steps/mm (SETSCALE calibration, RAM-only on the firmware
# side -- re-sent every run). Negative: HOME's dir=+1 increases the step
# count as it travels TOWARD the near/home hardstop, so 0mm (the post-home
# zero) needs a *negative* scale for positive MOVEMM values to mean "away
# from home" -- getting this backwards drives an axis further into its
# home hardstop with no StallGuard protection at all (MOVEMM doesn't check
# stall, unlike HOME). J0's magnitude is ruler-verified; the rest are
# back-calculated from a known-good 10mm move and haven't been
# individually ruler-checked.
STEPS_PER_MM = {0: -3216.0, 1: -2770.0, 2: -3260.0, 3: -3181.5, 4: -3472.0, 5: -3197.0}

# joint index -> measured home-to-far-hardstop travel (mm). ALWAYS POSITIVE,
# on every axis: HOME_DIRECTION=+1 runs an axis's step count UP into its home
# hardstop and ZERO then pins that stop at step 0, so with all six
# STEPS_PER_MM negative, positive mm is the only direction that travels away
# from home. A negative value here aims the axis back into its own home
# hardstop -- which MOVEMM does NOT stall-check (only HOME does), so it
# grinds unprotected.
#
# J4 was originally -62.2, on the theory that middle's x rail is "mirrored".
# It IS mirrored, but geometrically: middle homes at the opposite corner
# from index (local q_My min rather than max) and still travels positive-mm
# away from it. That mirroring belongs in _TRANSFORM's sign, and is already
# there -- encoding it here as well flipped the sign twice and pointed the
# whole axis backwards. Confirmed on hardware 2026-08-29: J0 (known-good)
# and J4 both move AWAY from home on a +3mm MOVEMM, via
# host/examples/verify_axis_direction.py.
#
# This is the single source of truth for stepper bounds-checking throughout
# this module; do not use the nominal {P}-diagram box dimensions for safety
# checks, since they disagree with real measured travel on J1/J3/J5 by
# several mm (nominal is optimistic there -- see FINGER_GEOMETRY's own
# comment for the derivation).
FULL_EXTENSION_MM = {0: 112.4, 1: 56.2, 2: 62.5, 3: 56.0, 4: 62.2, 5: 54.1}

STEPPER_VELOCITY = 12000
STEPPER_ACCEL = 2000

# StallGuard2 homing, validated across all 6 axes (2026-08-25/26, re-tuned
# 2026-08-28) at this speed/current/accel. COOLCONF (reg 0x6D) = SGT<<16
# with CoolStep disabled (semin=0). SGT=8 on J0/J2/J4. J1/J3/J5 were bumped
# to SGT=10 (2026-08-28) after J1 showed persistent early false stalls at
# lower SGT across multiple live runs -- J1 had previously failed BOTH ways
# (SGT=8 false-early, SGT=11 missed a real stall), so SGT=10 is a deliberate
# choice to accept the timeout-guarantee homing path (_home_timeout_ms
# mathematically covers full travel, so it's still a trustworthy home even
# when StallGuard doesn't fire) rather than keep chasing a perfect
# threshold on this persistently borderline axis. J3/J5 moved to the same
# SGT=10 preemptively ("just to be safe") without their own documented
# false-stall history -- re-tune any of these three individually if they
# turn out to need a different value than J1. Re-tune per-axis for every
# axis if current, speed, or any mechanism changes.
HOME_COOLCONF = {0: 0x00080000, 1: 0x000A0000, 2: 0x00080000, 3: 0x000A0000, 4: 0x00080000, 5: 0x000A0000}
HOME_DIRECTION = 1
HOME_VELOCITY = 12000
HOME_ACCEL = 2000
HOME_TIMEOUT_MARGIN = 1.4  # multiplier over the theoretical minimum travel time
HOME_POLL_PERIOD_S = 0.3
PRE_HOME_BACKOFF_MM = 10.0  # if already closer than this to the hardstop, back off
# first -- re-homing an axis that's already at/near the wall presses into it
# for HOME's whole grace period (~ramp time + 800ms) with StallGuard not even
# checked yet, since the grace period exists to skip the noisy low-speed
# ramp-up, not to detect an already-established stall. See project memory
# project_home_grace_period_push_at_hardstop.md -- rare in practice (usually
# an axis is homed once and then moved elsewhere), but cheap to guard against.

# finger_id -> (origin_x, origin_y) relative to {P}, in mm, and (box_width,
# box_height) i.e. the (q_Fx, q_Fy) span of that finger's local workspace
# box, both from the confirmed diagram. Box is centered on the origin.
#
# The affine transform below (stepper_mm = offset + sign*local_mm) was
# solved per finger/axis from two constraints: (1) firmware x tracks local
# q_Fy and firmware y tracks local q_Fx (the confirmed swap), and (2) each
# finger's stepper-homed corner (thumb: bottom-left / both local min; index:
# top-right / both local max; middle: bottom-right / q_Mx max, q_My min)
# maps to stepper 0mm, with that axis's (always positive) far-end stepper
# range matching the opposite corner. Nominal box half-extents (30/30,
# 30/30, 55/30 for thumb/index/middle) were used to derive these constants; do NOT reuse
# them for bounds-checking -- see FULL_EXTENSION_MM above and
# _LOCAL_BOUNDS below for why real measured travel differs.
FINGER_GEOMETRY = {
    0: {"origin": (-50.0, 0.0), "box": (60.0, 110.0)},  # thumb
    1: {"origin": (50.0, 55.0), "box": (60.0, 60.0)},  # index
    2: {"origin": (50.0, -55.0), "box": (60.0, 60.0)},  # middle
}

# finger_id -> (x_offset, x_sign, y_offset, y_sign) such that:
#   stepper_x_mm = x_offset + x_sign * local_y_mm
#   stepper_y_mm = y_offset + y_sign * local_x_mm
# (note the swap: stepper x <-> local y, stepper y <-> local x)
_TRANSFORM = {
    0: (55.0, 1.0, 30.0, 1.0),  # thumb: stepper_x=local_y+55, stepper_y=local_x+30
    1: (30.0, -1.0, 30.0, -1.0),  # index: stepper_x=30-local_y, stepper_y=30-local_x
    # middle's x_sign is +1 where index's is -1: they are mirror images about
    # {P}'s x axis, so middle homes at local q_My MIN and travels up from it
    # while index homes at q_My MAX and travels down. Both travel positive mm.
    2: (30.0, 1.0, 30.0, -1.0),  # middle: stepper_x=30+local_y, stepper_y=30-local_x
}


def axis_stepper_range(joint_index: int) -> tuple[float, float]:
    """Valid firmware-mm range for one stepper joint: (0.0, limit). Every
    axis homes to 0mm and travels positive away from there -- including J4,
    which used to be special-cased as "mirrored" here (see
    FULL_EXTENSION_MM's comment for why that was wrong). This is the single
    place that understands FULL_EXTENSION_MM's sign convention; anything
    checking a raw firmware-mm command against real hardware range (e.g.
    hand_control.py's REPL) should call this rather than re-deriving it."""
    limit = FULL_EXTENSION_MM[joint_index]
    if limit <= 0.0:
        # Guard the exact regression this function used to encode: with a
        # negative limit, the only values a caller could pass this check are
        # ones that drive the axis into its home hardstop.
        raise ValueError(
            f"FULL_EXTENSION_MM[{joint_index}]={limit} is not positive -- every axis "
            "travels positive-mm away from home; see that dict's comment"
        )
    return (0.0, limit)


def _local_bounds(finger_id: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """Invert this finger's transform against the REAL measured
    FULL_EXTENSION_MM (not the nominal box) to get the valid local (x, y)
    interval, computed once and cached by Gantry/GantryFinger at init."""
    x_joint, y_joint = STEPPER_JOINTS[finger_id]
    x_offset, x_sign, y_offset, y_sign = _TRANSFORM[finger_id]
    # stepper_x = x_offset + x_sign*local_y ranges over [0, FULL_EXTENSION_MM[x_joint]]
    sx0, sx1 = 0.0, FULL_EXTENSION_MM[x_joint]
    y_at_sx0 = (sx0 - x_offset) / x_sign
    y_at_sx1 = (sx1 - x_offset) / x_sign
    y_bounds = (min(y_at_sx0, y_at_sx1), max(y_at_sx0, y_at_sx1))
    sy0, sy1 = 0.0, FULL_EXTENSION_MM[y_joint]
    x_at_sy0 = (sy0 - y_offset) / y_sign
    x_at_sy1 = (sy1 - y_offset) / y_sign
    x_bounds = (min(x_at_sy0, x_at_sy1), max(x_at_sy0, x_at_sy1))
    return x_bounds, y_bounds


def _home_timeout_ms(joint_index: int) -> int:
    """Timeout that's mathematically guaranteed to cover this axis's full
    measured travel range (FULL_EXTENSION_MM) at HOME_VELOCITY/HOME_ACCEL,
    times HOME_TIMEOUT_MARGIN for real-world slop -- computed per axis
    instead of one shared constant, since axes have very different real
    lengths (J0's 112.4mm vs J5's 54.1mm) and a timeout sized for the
    longest axis wastes a lot of grind time on the shorter ones.

    This is what makes a StallGuard miss (homing_result=3) trustworthy
    rather than just a failure: because the timeout can't fire before this
    much distance has actually been traveled, hitting it means the axis has
    covered its ENTIRE real range and therefore must physically be at the
    far hardstop by now, whether or not StallGuard noticed. See
    feedback_homing_timeout_must_cover_full_travel.md for why the timeout
    itself must never be shortened below this regardless."""
    distance_steps = abs(FULL_EXTENSION_MM[joint_index]) * abs(STEPS_PER_MM[joint_index])
    ramp_distance_steps = HOME_VELOCITY**2 / (2 * HOME_ACCEL)
    if distance_steps <= ramp_distance_steps:
        travel_s = (2 * distance_steps / HOME_ACCEL) ** 0.5  # never reaches cruise speed
    else:
        travel_s = HOME_VELOCITY / HOME_ACCEL + (distance_steps - ramp_distance_steps) / HOME_VELOCITY
    return int(travel_s * HOME_TIMEOUT_MARGIN * 1000)


def _home_one_axis(driver: MantaHandDriver, joint_index: int):
    """Sequentially home one axis via StallGuard2, then ZERO it so 0mm
    becomes the home reference for move_to_mm. Never call this
    concurrently with another axis's homing -- see this project's
    sequential-only rule. Relocated verbatim from hand_control.py's
    home_axis (same behavior, validated live this session)."""
    j = driver.joints[joint_index]
    current_mm = j.status.position / STEPS_PER_MM[joint_index]
    if abs(current_mm) < PRE_HOME_BACKOFF_MM:
        # HOME_VELOCITY/HOME_ACCEL, not STEPPER_VELOCITY -- at 400sps a 10mm
        # move takes ~80s, far longer than this poll loop waits, so the
        # first cut of this backoff silently moved on to HOME while the
        # backoff move was still barely underway (~1mm covered, not 10).
        j.move_to_mm(PRE_HOME_BACKOFF_MM, HOME_VELOCITY, HOME_ACCEL)
        start = time.monotonic()
        while time.monotonic() - start < 15.0:
            if not j.status.moving:
                break
            time.sleep(0.2)
        else:
            j.stop()
            j.disable()
            raise RuntimeError(f"J{joint_index}: pre-home backoff didn't finish in 15s -- disabled")
        print(f"  J{joint_index} was within {PRE_HOME_BACKOFF_MM}mm of home -- backed off first")
    j.write_reg5160(0x6D, HOME_COOLCONF[joint_index])
    timeout_ms = _home_timeout_ms(joint_index)
    j.home(HOME_DIRECTION, HOME_VELOCITY, HOME_ACCEL, timeout_ms)
    hard_cap_s = timeout_ms / 1000 + 5.0  # must stay above the firmware timeout --
    # a client-side cap shorter than the firmware's own timeout would fire
    # first and disable the axis before HOME could reach its guaranteed-safe
    # timeout outcome, wrongly treating a still-legitimate approach as a failure.
    start = time.monotonic()
    status = j.status
    while status.homing_result == 1:
        if time.monotonic() - start > hard_cap_s:
            j.stop()
            j.disable()
            raise RuntimeError(f"J{joint_index}: homing exceeded {hard_cap_s:.0f}s hard cap -- disabled")
        time.sleep(HOME_POLL_PERIOD_S)
        status = j.status
    if status.homing_result == 2:
        j.zero()
        print(f"  J{joint_index} homed")
    elif status.homing_result == 3:
        # No stall seen, but timeout_ms guarantees the full measured range
        # was covered -- the axis must be at the far hardstop regardless.
        j.zero()
        print(f"  J{joint_index} homed via timeout guarantee (StallGuard didn't trigger)")
    else:
        j.disable()
        raise RuntimeError(f"J{joint_index}: homing failed (homing_result={status.homing_result})")


class GantryFinger:
    """One finger's stepper gantry: local-frame position control + homing.
    Built by Gantry.finger(), not constructed directly."""

    def __init__(self, driver: MantaHandDriver, finger_id: int):
        self._driver = driver
        self._finger_id = finger_id
        self._x_joint, self._y_joint = STEPPER_JOINTS[finger_id]
        self._x_offset, self._x_sign, self._y_offset, self._y_sign = _TRANSFORM[finger_id]
        self._origin = FINGER_GEOMETRY[finger_id]["origin"]
        self._x_bounds, self._y_bounds = _local_bounds(finger_id)

    def move_to_local(self, x_mm: float, y_mm: float,
                       velocity: int = STEPPER_VELOCITY, accel: int = STEPPER_ACCEL):
        """Move to (x_mm, y_mm) in this finger's own local (q_Fx, q_Fy)
        frame, origin at the box center per the confirmed diagram. Raises
        ValueError (no hardware command issued) if the target is outside
        this finger's real measured range. Assumes this finger's two
        stepper axes have already been homed this session -- see
        GantryFinger.home()/Gantry.home_all()."""
        lo, hi = self._x_bounds
        if not lo <= x_mm <= hi:
            raise ValueError(f"finger {self._finger_id}: local x={x_mm}mm outside real range [{lo:.2f}, {hi:.2f}]mm")
        lo, hi = self._y_bounds
        if not lo <= y_mm <= hi:
            raise ValueError(f"finger {self._finger_id}: local y={y_mm}mm outside real range [{lo:.2f}, {hi:.2f}]mm")
        stepper_x = self._x_offset + self._x_sign * y_mm
        stepper_y = self._y_offset + self._y_sign * x_mm
        self._driver.joints[self._x_joint].move_to_mm(stepper_x, velocity, accel)
        self._driver.joints[self._y_joint].move_to_mm(stepper_y, velocity, accel)

    def move_to_global(self, x_mm: float, y_mm: float,
                        velocity: int = STEPPER_VELOCITY, accel: int = STEPPER_ACCEL):
        """Move to (x_mm, y_mm) in the shared palm-fixed global frame {P}.
        Converts to this finger's local frame (simple translation by its
        origin -- local frames are axis-aligned, no rotation) then
        delegates to move_to_local, which performs the bounds check."""
        origin_x, origin_y = self._origin
        self.move_to_local(x_mm - origin_x, y_mm - origin_y, velocity=velocity, accel=accel)

    def home(self, velocity: int = STEPPER_VELOCITY, accel: int = STEPPER_ACCEL):
        """Home this finger's two stepper axes (x then y, sequentially --
        never concurrently, per this project's homing rule). velocity/accel
        are unused here (homing always uses HOME_VELOCITY/HOME_ACCEL, a
        deliberately different, validated-safe pair) but accepted for
        signature symmetry with move_to_local/move_to_global."""
        _home_one_axis(self._driver, self._x_joint)
        _home_one_axis(self._driver, self._y_joint)


class Gantry:
    """All 3 fingers' stepper gantries on one MantaHandDriver. Enables
    every joint and sets its scale in __init__ -- a caller never has to
    do that manually."""

    def __init__(self, driver: MantaHandDriver):
        self._driver = driver
        self._fingers: dict[int, GantryFinger] = {}
        for finger_id, (x_joint, y_joint) in STEPPER_JOINTS.items():
            driver.joints[x_joint].enable()
            driver.joints[y_joint].enable()
            driver.joints[x_joint].set_scale(STEPS_PER_MM[x_joint])
            driver.joints[y_joint].set_scale(STEPS_PER_MM[y_joint])
            self._fingers[finger_id] = GantryFinger(driver, finger_id)

    def finger(self, finger_id: int) -> GantryFinger:
        return self._fingers[finger_id]

    def home_all(self, velocity: int = STEPPER_VELOCITY, accel: int = STEPPER_ACCEL):
        """Home every stepper axis, sequentially, all 3 fingers in order --
        matches hand_control.py's original home_all_axes iteration order."""
        print("homing all axes sequentially...")
        for finger_id in STEPPER_JOINTS:
            self._fingers[finger_id].home(velocity=velocity, accel=accel)
        print("all axes homed -- 0mm is now each axis's home reference")
