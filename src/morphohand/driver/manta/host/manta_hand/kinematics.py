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
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pyserial is a runtime-only dependency of .driver, and every
    # annotation here is a string (`from __future__ import annotations`), so the
    # import is not needed to load this module. Keeping it lazy is what makes the
    # docstring's "importable unconditionally" claim actually true -- an offline
    # planner on a workstation with no pyserial can read the calibration tables
    # (STEPS_PER_MM / FULL_EXTENSION_MM / _TRANSFORM) as the single source of truth
    # instead of copying them.
    from .driver import MantaHandDriver

# finger_id -> (x_joint_index, y_joint_index) on the Manta M8P (J0-J5).
STEPPER_JOINTS = {0: (0, 1), 1: (2, 3), 2: (4, 5)}

# steps/mm (SETSCALE calibration, RAM-only on the firmware side -- re-sent
# every run). Negative: HOME's dir=+1 increases the step count as it
# travels TOWARD the near/home hardstop, so 0mm (the post-home zero) needs
# a *negative* scale for positive MOVEMM values to mean "away from home"
# -- getting this backwards drives an axis further into its home hardstop
# with no StallGuard protection at all (MOVEMM doesn't check stall, unlike
# HOME).
#
# ONE shared scalar, not six independently-calibrated values, as of
# 2026-08-29. All six axes use the identical motor, leadscrew, and nut --
# steps/mm is a function of motor step angle x microstepping / screw
# pitch, which doesn't depend on rail length or which axis it is, so
# physically there can only be one true value here. The old per-joint dict
# ({0: -3216.0, 1: -2770.0, 2: -3260.0, 3: -3181.5, 4: -3472.0, 5:
# -3197.0}) let J1 drift 13.9% and J4 drift 8.0% from J0's value with
# nothing to flag either as wrong -- both were within a couple percent of
# each other and of the theoretical 3200 (200 full steps/rev x 16
# microsteps / 1mm pitch) EXCEPT those two, which is a calibration error
# in their individual "known-good 10mm move" check, not a real mechanical
# difference. Using J0's magnitude here since it's the only one that was
# ever independently ruler-verified rather than back-calculated from an
# eyeballed move.
STEPS_PER_MM = -3216.0

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
# this module.
#
# 2026-09-01: corrected -- all six axes reach their full nominal travel
# (110mm on J0, 60mm on the rest); there is no wall short of that on
# J1/J3/J5. The previous entries here (56.2/56.0/54.1, a 4-6mm "shortfall"
# against the 60mm nominal box) were wrong, not a real mechanical limit.
# This is a direct correction, not derived from the 45mm
# verify_frame_mapping.py --travel probe: that probe (commanded 45,
# measured 45 on all three, 0.0% scale error) only checked scale linearity
# up to 45mm -- PROBE_MM was deliberately kept short of the old 56ish-mm
# hypothesis, so on its own it cannot distinguish "wall at 56" from "goes
# to 60."
FULL_EXTENSION_MM = {0: 110.0, 1: 60.0, 2: 60.0, 3: 60.0, 4: 60.0, 5: 60.0}

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

class HomingAborted(RuntimeError):
    """Raised when a caller's `cancel` predicate stops a home or gantry move part-way.

    Distinct from RuntimeError (a real failure) on purpose: an aborted axis has NOT
    been zeroed and its step-0 reference is unknown, so a caller has to invalidate the
    whole session's home rather than treat this as a soft skip. See `_home_one_axis`'s
    `cancel` parameter for why an abort can never zero.
    """


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
    """Valid local (x, y) interval a caller can command via
    move_to_local/move_to_global -- the NOMINAL box from FINGER_GEOMETRY
    (e.g. thumb's full 60x110, index/middle's full 60x60), centered on the
    origin.

    2026-09-01: identical to _local_bounds_real_measured now that
    FULL_EXTENSION_MM was corrected to match this nominal box exactly on
    every axis (see that dict's comment) -- the "real measured travel is a
    few mm short of nominal" tradeoff this docstring used to describe was
    based on wrong FULL_EXTENSION_MM entries and no longer applies. Kept as
    a separate function from _local_bounds_real_measured in case
    FULL_EXTENSION_MM needs tightening below nominal again later."""
    box_w, box_h = FINGER_GEOMETRY[finger_id]["box"]
    return (-box_w / 2.0, box_w / 2.0), (-box_h / 2.0, box_h / 2.0)


def _local_bounds_real_measured(finger_id: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """Invert this finger's transform against FULL_EXTENSION_MM to get the
    valid local (x, y) interval. Identical to _local_bounds as of
    2026-09-01, since FULL_EXTENSION_MM now equals the nominal box on every
    axis (see _local_bounds's docstring) -- kept separate in case
    FULL_EXTENSION_MM needs tightening below nominal again later."""
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
    lengths (J0's 110mm vs J1/J3/J5's 60mm) and a timeout sized for the
    longest axis wastes a lot of grind time on the shorter ones.

    This is what makes a StallGuard miss (homing_result=3) trustworthy
    rather than just a failure: because the timeout can't fire before this
    much distance has actually been traveled, hitting it means the axis has
    covered its ENTIRE real range and therefore must physically be at the
    far hardstop by now, whether or not StallGuard noticed. See
    feedback_homing_timeout_must_cover_full_travel.md for why the timeout
    itself must never be shortened below this regardless."""
    distance_steps = abs(FULL_EXTENSION_MM[joint_index]) * abs(STEPS_PER_MM)
    ramp_distance_steps = HOME_VELOCITY**2 / (2 * HOME_ACCEL)
    if distance_steps <= ramp_distance_steps:
        travel_s = (2 * distance_steps / HOME_ACCEL) ** 0.5  # never reaches cruise speed
    else:
        travel_s = HOME_VELOCITY / HOME_ACCEL + (distance_steps - ramp_distance_steps) / HOME_VELOCITY
    return int(travel_s * HOME_TIMEOUT_MARGIN * 1000)


def _abort_axis(driver: MantaHandDriver, joint_index: int, *, settle_s: float = 8.0):
    """Bring one axis to a halt and hand it back to the firmware in a clean state.

    STOP alone is not enough to cancel a HOME. The firmware's Stepper_Stop() only sets
    `target = position` (firmware/Core/Src/stepper.c) -- it leaves `homing_result` at 1,
    so the axis stays logically "homing" and the supervisor tick eventually flips it to
    3 (timed out) on its own. A caller polling for the outcome would then read 3,
    believe the timeout guarantee (see _home_timeout_ms) and ZERO an axis that never
    reached its hardstop, leaving a bogus step-0 reference that the next MOVEMM drives
    away from with no stall protection at all. Stepper_Disable() DOES clear
    homing_result, so the disable here is what actually cancels the home.

    STOP first so the axis decelerates under control rather than dropping torque at
    speed; disable once it has stopped, or once `settle_s` has passed, since an
    unresponsive axis still has to end up de-energised."""
    j = driver.joints[joint_index]
    try:
        j.stop()
        deadline = time.monotonic() + settle_s
        while time.monotonic() < deadline:
            if not j.status.moving:
                break
            time.sleep(0.1)
    finally:
        j.disable()  # the part that actually clears the firmware's homing_result


def _cancelled(cancel: Callable[[], bool] | None) -> bool:
    return bool(cancel and cancel())


def wait_for_axis_idle(driver: MantaHandDriver, joint_index: int, timeout_s: float,
                        *, poll_s: float = 0.15,
                        cancel: Callable[[], bool] | None = None) -> None:
    """Block until one axis reports `moving == False`, or raise.

    MOVEMM is non-blocking (see Joint.move_to_mm) and has no stall protection, so
    anything that issues one and then wants to know it finished has to poll. Polls ONE
    axis with STAT rather than the whole board with STATALL: STATALL is nine USB-CDC
    packets sent from the firmware's main loop, in competition with the step ISRs that
    are running at that very moment at a higher interrupt priority than USB.

    Raises HomingAborted if `cancel` fires (the axis is stopped and disabled first), or
    TimeoutError if it is still moving after `timeout_s`."""
    j = driver.joints[joint_index]
    deadline = time.monotonic() + timeout_s
    while True:
        if _cancelled(cancel):
            _abort_axis(driver, joint_index)
            raise HomingAborted(f"J{joint_index}: motion cancelled")
        if not j.status.moving:
            return
        if time.monotonic() > deadline:
            _abort_axis(driver, joint_index)
            raise TimeoutError(
                f"J{joint_index}: still moving after {timeout_s:.1f}s -- stopped and disabled"
            )
        time.sleep(poll_s)


def move_time_estimate_s(joint_index: int, distance_mm: float,
                          velocity: int = STEPPER_VELOCITY,
                          accel: int = STEPPER_ACCEL) -> float:
    """Trapezoidal-profile travel time for `distance_mm` on this axis, in seconds.

    Same arithmetic as _home_timeout_ms, but for an arbitrary distance, so a settle
    timeout can be sized from the move it is actually waiting on instead of one
    constant that is either far too generous for a 3mm nudge or too tight for a
    full-length one."""
    steps = abs(distance_mm) * abs(STEPS_PER_MM)
    ramp_steps = velocity**2 / (2 * accel)
    if steps <= 0:
        return 0.0
    if steps <= ramp_steps:
        return (2 * steps / accel) ** 0.5
    return velocity / accel + (steps - ramp_steps) / velocity


def _home_one_axis(driver: MantaHandDriver, joint_index: int,
                    *, cancel: Callable[[], bool] | None = None,
                    report: Callable[[str, dict], None] | None = None) -> dict:
    """Sequentially home one axis via StallGuard2, then ZERO it so 0mm
    becomes the home reference for move_to_mm. Never call this
    concurrently with another axis's homing -- see this project's
    sequential-only rule. Relocated verbatim from hand_control.py's
    home_axis (same behavior, validated live this session).

    `cancel`: polled at every wait point. When it fires, the axis is stopped and
    disabled (see _abort_axis) and HomingAborted is raised WITHOUT zeroing -- an
    interrupted home leaves the step-0 reference unknown, and zeroing anyway is exactly
    how a cancelled home becomes a gantry that later drives itself into a hardstop.

    `report`: optional callback(event_name, payload) so a caller can show per-axis
    progress and the stall-vs-timeout outcome live, rather than only in this process's
    stdout.

    Returns {joint, homing_result, stalled, elapsed_s}."""
    def emit(name: str, **payload):
        payload["joint"] = joint_index
        if report is not None:
            report(name, payload)

    j = driver.joints[joint_index]
    if _cancelled(cancel):
        raise HomingAborted(f"J{joint_index}: homing cancelled before it started")
    timeout_ms = _home_timeout_ms(joint_index)
    emit("home_axis_start", timeout_s=timeout_ms / 1000.0,
         travel_mm=FULL_EXTENSION_MM[joint_index])
    started = time.monotonic()

    current_mm = j.status.position / STEPS_PER_MM
    if abs(current_mm) < PRE_HOME_BACKOFF_MM:
        # HOME_VELOCITY/HOME_ACCEL, not STEPPER_VELOCITY -- at 400sps a 10mm
        # move takes ~80s, far longer than this poll loop waits, so the
        # first cut of this backoff silently moved on to HOME while the
        # backoff move was still barely underway (~1mm covered, not 10).
        emit("home_axis_backoff", from_mm=current_mm, to_mm=PRE_HOME_BACKOFF_MM)
        j.move_to_mm(PRE_HOME_BACKOFF_MM, HOME_VELOCITY, HOME_ACCEL)
        try:
            wait_for_axis_idle(driver, joint_index, 15.0, poll_s=0.2, cancel=cancel)
        except TimeoutError as exc:
            raise RuntimeError(
                f"J{joint_index}: pre-home backoff didn't finish in 15s -- disabled") from exc
        print(f"  J{joint_index} was within {PRE_HOME_BACKOFF_MM}mm of home -- backed off first")
    j.write_reg5160(0x6D, HOME_COOLCONF[joint_index])
    j.home(HOME_DIRECTION, HOME_VELOCITY, HOME_ACCEL, timeout_ms)
    hard_cap_s = timeout_ms / 1000 + 5.0  # must stay above the firmware timeout --
    # a client-side cap shorter than the firmware's own timeout would fire
    # first and disable the axis before HOME could reach its guaranteed-safe
    # timeout outcome, wrongly treating a still-legitimate approach as a failure.
    start = time.monotonic()
    status = j.status
    while status.homing_result == 1:
        if _cancelled(cancel):
            _abort_axis(driver, joint_index)
            raise HomingAborted(
                f"J{joint_index}: homing cancelled -- axis stopped and disabled, NOT zeroed")
        if time.monotonic() - start > hard_cap_s:
            j.stop()
            j.disable()
            raise RuntimeError(f"J{joint_index}: homing exceeded {hard_cap_s:.0f}s hard cap -- disabled")
        time.sleep(HOME_POLL_PERIOD_S)
        status = j.status
    elapsed = time.monotonic() - started
    if status.homing_result == 2:
        j.zero()
        print(f"  J{joint_index} homed")
        emit("home_axis_done", homing_result=2, stalled=True, elapsed_s=elapsed)
    elif status.homing_result == 3:
        # No stall seen, but timeout_ms guarantees the full measured range
        # was covered -- the axis must be at the far hardstop regardless.
        j.zero()
        print(f"  J{joint_index} homed via timeout guarantee (StallGuard didn't trigger)")
        emit("home_axis_done", homing_result=3, stalled=False, elapsed_s=elapsed)
    else:
        j.disable()
        emit("home_axis_failed", homing_result=status.homing_result, elapsed_s=elapsed)
        raise RuntimeError(f"J{joint_index}: homing failed (homing_result={status.homing_result})")
    return {"joint": joint_index, "homing_result": status.homing_result,
            "stalled": status.homing_result == 2, "elapsed_s": elapsed}


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

    def home(self, velocity: int = STEPPER_VELOCITY, accel: int = STEPPER_ACCEL,
             *, cancel: Callable[[], bool] | None = None,
             report: Callable[[str, dict], None] | None = None) -> list[dict]:
        """Home this finger's two stepper axes (x then y, sequentially --
        never concurrently, per this project's homing rule). velocity/accel
        are unused here (homing always uses HOME_VELOCITY/HOME_ACCEL, a
        deliberately different, validated-safe pair) but accepted for
        signature symmetry with move_to_local/move_to_global.

        `cancel`/`report` are forwarded to _home_one_axis; see there. Returns one
        outcome dict per axis."""
        return [_home_one_axis(self._driver, self._x_joint, cancel=cancel, report=report),
                _home_one_axis(self._driver, self._y_joint, cancel=cancel, report=report)]

    def stepper_targets(self, x_mm: float, y_mm: float, *, frame: str = "local"
                         ) -> dict[int, float]:
        """Bounds-checked {joint index: firmware mm} for a local- or global-frame
        target, WITHOUT commanding anything. Splitting the arithmetic out from the
        command is what lets a caller move the two axes one at a time (see
        Gantry.move_sequential) while keeping this module the only place that knows
        the transform and the limits."""
        if frame == "global":
            origin_x, origin_y = self._origin
            x_mm, y_mm = x_mm - origin_x, y_mm - origin_y
        elif frame != "local":
            raise ValueError(f"frame must be 'local' or 'global', got {frame!r}")
        lo, hi = self._x_bounds
        if not lo <= x_mm <= hi:
            raise ValueError(f"finger {self._finger_id}: local x={x_mm}mm outside real range "
                             f"[{lo:.2f}, {hi:.2f}]mm")
        lo, hi = self._y_bounds
        if not lo <= y_mm <= hi:
            raise ValueError(f"finger {self._finger_id}: local y={y_mm}mm outside real range "
                             f"[{lo:.2f}, {hi:.2f}]mm")
        return {self._x_joint: self._x_offset + self._x_sign * y_mm,
                self._y_joint: self._y_offset + self._y_sign * x_mm}


class Gantry:
    """All 3 fingers' stepper gantries on one MantaHandDriver. Enables
    every joint and sets its scale in __init__ -- a caller never has to
    do that manually."""

    def __init__(self, driver: MantaHandDriver):
        self._driver = driver
        self._fingers: dict[int, GantryFinger] = {
            finger_id: GantryFinger(driver, finger_id) for finger_id in STEPPER_JOINTS}
        self.prepare()

    def prepare(self) -> None:
        """Enable every axis and re-send its mm calibration.

        Run at construction and again before each home or gantry move, not once at
        startup, because both bits of state are lost more often than they look. EN is
        cleared by any DIS -- which is how a cancelled home is cancelled (see
        _abort_axis), and what the disable-motors control does -- and Stepper_Home
        refuses with ERR NODIAG on an axis that is not enabled, so a session that
        aborted one home could not start another. SETSCALE is RAM-only firmware side
        and does not survive a board reset. Both are single cheap commands; re-asserting
        them is what makes the session recoverable instead of one-shot."""
        for x_joint, y_joint in STEPPER_JOINTS.values():
            for joint_index in (x_joint, y_joint):
                self._driver.joints[joint_index].enable()
                self._driver.joints[joint_index].set_scale(STEPS_PER_MM)

    def finger(self, finger_id: int) -> GantryFinger:
        return self._fingers[finger_id]

    def home_all(self, velocity: int = STEPPER_VELOCITY, accel: int = STEPPER_ACCEL,
                 *, cancel: Callable[[], bool] | None = None,
                 report: Callable[[str, dict], None] | None = None) -> list[dict]:
        """Home every stepper axis, sequentially, all 3 fingers in order --
        matches hand_control.py's original home_all_axes iteration order.

        Returns one outcome dict per axis (see _home_one_axis), so a caller can tell
        which axes found their hardstop via StallGuard2 and which fell through to the
        timeout guarantee. On this hardware as of 2026-08-29 that is not academic:
        J3 and J5 routinely reach `homing_result == 3`, i.e. they grind against the
        hardstop for the axis's full computed timeout (25s and 24s) before the home is
        accepted. That is by design (see HOME_COOLCONF), but it looks exactly like a
        hang to anyone watching, so surface it rather than only print()ing it."""
        print("homing all axes sequentially...")
        self.prepare()  # a previous abort or disable-motors left axes de-energised
        outcomes: list[dict] = []
        for finger_id in STEPPER_JOINTS:
            outcomes += self._fingers[finger_id].home(velocity=velocity, accel=accel,
                                                       cancel=cancel, report=report)
        print("all axes homed -- 0mm is now each axis's home reference")
        return outcomes

    def move_sequential(self, targets: dict[int, tuple[float, float]], *,
                        frame: str = "global",
                        velocity: int = STEPPER_VELOCITY, accel: int = STEPPER_ACCEL,
                        cancel: Callable[[], bool] | None = None,
                        report: Callable[[str, dict], None] | None = None) -> None:
        """Move several fingers to their targets ONE AXIS AT A TIME, waiting for each
        to stop before starting the next.

        This exists because issuing six MOVEMMs back to back is not a motion profile
        this hardware has ever been validated at. Everything that has run on this hand
        -- hand_control.py's REPL, movement_examples.py, the homing sequence -- moves a
        single axis at a time. Six simultaneous starts means six simultaneous
        TMC5160_StartMotionKick current kicks (IRUN 1 -> 7 for 500ms each, see
        firmware/Core/Inc/tmc5160_spi.h) on the 19V rail that also backfeeds the CB1,
        and six axes' worth of step ISRs at NVIC priority 2 preempting the USB
        interrupt at priority 3 while the host is polling the very link those ISRs are
        starving.

        Every target is computed and bounds-checked BEFORE the first command goes out,
        so a plan that does not fit moves nothing rather than stranding the hand
        half-configured. Each axis then gets a settle timeout sized from its own
        travel distance, and `cancel` is honoured between and during axes."""
        self.prepare()
        commands: list[tuple[int, float]] = []
        for finger_id, (x_mm, y_mm) in sorted(targets.items()):
            per_axis = self._fingers[finger_id].stepper_targets(x_mm, y_mm, frame=frame)
            commands += sorted(per_axis.items())

        for joint_index, mm in commands:
            if _cancelled(cancel):
                raise HomingAborted(f"gantry move cancelled before J{joint_index}")
            joint = self._driver.joints[joint_index]
            here = joint.status.position / STEPS_PER_MM
            distance = abs(mm - here)
            if report is not None:
                report("gantry_axis_start", {"joint": joint_index, "target_mm": mm,
                                              "from_mm": here, "distance_mm": distance})
            joint.move_to_mm(mm, velocity, accel)
            # +3s covers the command round trip and the firmware's own start latency;
            # the 1.5x is the same kind of real-world margin HOME_TIMEOUT_MARGIN applies.
            budget = move_time_estimate_s(joint_index, distance, velocity, accel) * 1.5 + 3.0
            wait_for_axis_idle(self._driver, joint_index, budget, cancel=cancel)
            if report is not None:
                report("gantry_axis_done", {"joint": joint_index, "target_mm": mm})
