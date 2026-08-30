"""Feetech SCS0009 servo control over a Waveshare/Robotis U2D2 USB-TTL
adapter -- a completely separate physical link from the Manta M8P's
USB-CDC connection (MantaHandDriver/Joint in driver.py/joint.py). The U2D2
plugs into its own CB1 USB port (typically /dev/ttyUSB0, not
/dev/ttyACM0) and talks directly to the servo bus; nothing about this goes
through the STM32 firmware.

Built on `rustypot` (PyPI: `rustypot`, source at
https://github.com/pollen-robotics/rustypot), the same library used by
Pollen Robotics' own AmazingHand reference control code -- not Feetech's own
scservo_sdk. Install with:

    pip install rustypot

Hand layout (as actually wired and calibrated on real hardware, not a
placeholder): 3 fingers, 3 independently-driven servos each -- one servo per
degree of freedom (aa = adduction/abduction, fe1 = proximal phalanx flexion,
fe2 = distal phalanx flexion), IDs 0/1/2 (finger 0), 3/4/5 (finger 1),
6/7/8 (finger 2). See FINGER_JOINTS below for each joint's servo id, zero
reference, and calibrated range. This replaced an earlier 2-servo
differential-linkage design (shared flexion pair + common-mode adduction
hack); that scheme is gone, not just superseded -- there's no coexisting
"legacy" path here to fall back to.

Finger.servo()/ServoBus.servo() give direct per-servo access for anything
FINGER_JOINTS' named joints don't cover.
"""

from __future__ import annotations

from dataclasses import dataclass

import math
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from rustypot import Scs0009PyController


def _controller_cls():
    """rustypot, imported at the moment a bus is actually opened rather than at
    module load. This module's constant tables -- FINGER_JOINTS above all, the
    authoritative per-servo zero + calibrated range -- are the contract an offline
    planner has to validate a trajectory against, and it should not need the
    hardware's serial stack installed to read them. Opening a real ServoBus still
    fails with the same message it always did."""
    try:
        from rustypot import Scs0009PyController
    except ImportError as exc:  # pragma: no cover
        raise ImportError("manta_hand.servos requires rustypot: pip install rustypot") from exc
    return Scs0009PyController

# Torque-enable states, per AmazingHand's reference code.
TORQUE_ON = 1
TORQUE_OFF = 2
TORQUE_FREE = 3  # backdrivable by hand, no holding torque

# SCS0009 has a 10-bit position sensor across ~300 degrees, i.e. one tick is
# ~300/1024 ~= 0.293 degrees ~= 0.00511 radians. A goal_position write that
# genuinely succeeded can still read back slightly different from the exact
# float we sent due to that quantization -- this tolerance (~3 ticks) is
# wide enough to accept that rounding while still catching a write that
# plain didn't take effect at all (which, on real hardware, showed up as a
# readback wildly different from the target, not just off by a tick or two).
POSITION_TOLERANCE_RAD = 0.02

# Gap enforced after every single INDIVIDUAL controller call (write or
# read), before the next one is allowed to go out. Confirmed empirically
# (not from any datasheet) that calls issued back-to-back with no gap --
# e.g. write_torque_enable, then immediately write_goal_speed, then
# immediately write_goal_position, then immediately read_present_position --
# reproducibly time out, on both the CB1 and a Mac with the same U2D2/servo,
# while the identical sequence with ~0.3s between each call works reliably
# every time. Exactly the same class of bug as the TMC2209 bit-bang link's
# "insufficient turnaround time between transactions" earlier in this
# project -- this is a workaround for it (a known-good delay), not a
# root-cause fix, since the actual minimum safe gap hasn't been bisected.
#
# Does NOT apply to sync_write_goal_position (see ServoBus.sync_set_joints)
# -- that's a single bus transaction addressing multiple servos at once, not
# a sequence of individual calls, and needs no inter-call gap for that
# reason. Moving several servos via a loop of individual move_to_deg() calls
# is visibly staggered (confirmed: >1 second of lag across 6 servos)
# compared to one sync_write call (~1ms for the same 6) -- sync_set_joints
# is what actually produces simultaneous multi-servo movement.
INTER_CMD_DELAY_S = 0.3


@dataclass
class ServoStatus:
    position_deg: float  # raw servo angle, calibration offset NOT subtracted


class ServoBus:
    """One U2D2 adapter, addressing servo IDs on its shared bus.

    Usage:
        with ServoBus("/dev/ttyUSB0") as bus:
            bus.servo(0).enable()
            bus.servo(0).move_to_deg(15, speed=200)
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 1_000_000, timeout: float = 0.5):
        self._c = _controller_cls()(serial_port=port, baudrate=baudrate, timeout=timeout)
        # The controller owns one half-duplex serial bus.  The web controller has a
        # trajectory writer and an optional telemetry reader in different threads;
        # without one bus-wide lock their packets can overlap.  Keep the lock here,
        # below every caller, rather than relying on each application to remember it.
        self._lock = threading.RLock()
        # The very first command issued right after opening the port
        # reproducibly timed out without this, confirmed empirically on
        # both the CB1 and a Mac -- same class of issue as INTER_CMD_DELAY_S
        # above (and the same fix pattern used for the STM32 side's
        # TMC_Init(), which needed its own settle delay before its first
        # UART transaction for the same kind of reason).
        time.sleep(1.0)
        self._servos: dict[int, Servo] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # rustypot's Scs0009PyController doesn't expose an explicit close()
        # as of this writing -- the underlying serial port is released when
        # the Rust object is dropped (standard PyO3 behavior on garbage
        # collection), so there's nothing to call here. Kept as a context
        # manager anyway so `with ServoBus(...) as bus:` reads the same way
        # as MantaHandDriver, and so a real close() can be added here later
        # without changing any calling code.
        pass

    def servo(self, servo_id: int) -> "Servo":
        if servo_id not in self._servos:
            self._servos[servo_id] = Servo(self._c, servo_id, self._lock)
        return self._servos[servo_id]

    def finger(self, finger_id: int) -> "Finger":
        """finger_id: one of FINGER_JOINTS' keys below (0, 1, 2)."""
        return Finger(self, finger_id)

    def sync_set_joints(self, pose: dict[int, dict[str, float]], speed: int | None = None):
        """Real-time-capable aa/fe1/fe2 update: ONE sync_write_goal_position
        bus transaction for every joint given, across any number of
        fingers -- confirmed empirically at ~0.4ms for all 9 servos
        (~2700Hz raw throughput). This is the fast path for a control loop
        (e.g. a grasping policy) -- Finger.set_joint is NOT suitable for
        that, since its write-verify + INTER_CMD_DELAY_S makes it ~0.9s per
        single joint update, confirmed empirically.

        pose: {finger_id: {joint_name: relative_deg}}, relative_deg in the
        same zero-relative-degrees convention as Finger.set_joint. Every
        value is bounds-checked against FINGER_JOINTS' declared range
        BEFORE any write is issued (cheap, O(1) per joint, all-or-nothing
        -- a single bad value raises and nothing moves).

        speed: if given, sync-writes goal_speed for every joint in `pose`
        BEFORE the position write. This is NOT optional cosmetic tuning --
        a servo whose goal_speed register has never been set (or was left
        at some earlier value, e.g. from a torque-free calibration sweep)
        can fail to move at all on a plain sync_write_goal_position, or
        move only partway, with no error raised (confirmed on real
        hardware: aa servos that had never had set_joint()/zero_joints()
        called on them -- which are what set goal_speed on the verified
        single-write path -- silently didn't move via this call until
        goal_speed was written first; fe1/fe2 on the same fingers moved
        fine, apparently because something upstream had already given them
        a usable speed). Pass speed on at least the first call for any
        joint you're about to drive this way -- e.g. right after
        home()/zero_joints() if you're not relying on those having already
        touched every joint you plan to use here. Once a joint's goal_speed
        is set it persists on the servo, so a real control loop can pass
        speed once and omit it on every subsequent frame for full speed.

        Deliberately has NO time.sleep() after either write -- both
        sync_write_goal_speed and sync_write_goal_position are single bus
        transactions that need no inter-call gap (see INTER_CMD_DELAY_S's
        comment), and a control loop calling this every frame can't afford
        one. Also NOT verified against actual servo position afterward --
        a write that silently doesn't stick on one servo out of several
        wouldn't be caught here; that's the necessary cost of real-time
        speed. Follow up with a plain per-servo .status read if that
        matters for a given frame, not on every frame."""
        ids: list[int] = []
        values: list[float] = []
        for finger_id, joint_t in pose.items():
            joints = FINGER_JOINTS[finger_id]
            for name, relative_deg in joint_t.items():
                if name not in joints:
                    raise ValueError(f"finger {finger_id} has no joint {name!r} (have {list(joints)})")
                servo_id, zero_deg, (lo, hi) = joints[name]
                if not lo <= relative_deg <= hi:
                    raise ValueError(
                        f"finger {finger_id} {name}: {relative_deg} outside declared range [{lo}, {hi}]"
                    )
                ids.append(servo_id)
                values.append(math.radians(zero_deg + relative_deg))
        with self._lock:
            if speed is not None:
                self._c.sync_write_goal_speed(ids=ids, values=[speed] * len(ids))
            self._c.sync_write_goal_position(ids=ids, values=values)

    def sync_read_joint_positions(self) -> dict[int, dict[str, float]]:
        """Read all nine joint positions in one controller operation.

        Returns zero-relative degrees in the same convention accepted by
        :meth:`sync_set_joints`.  This is the only position polling path intended
        for a control/telemetry loop.  Falling back to nine ``Servo.status`` calls
        would also invoke the empirically-required 300 ms inter-command delay and
        could starve trajectory writes for seconds, so lack of sync-read support is
        reported explicitly instead.

        The SCS protocol manual documents READ and SYNC WRITE but not SYNC READ.
        ``rustypot`` nevertheless exposes ``sync_read_present_position`` for its
        SCS0009 controller.  Whether the particular nine-servo chain can sustain a
        useful rate is deliberately left to ``examples/benchmark_servo_telemetry.py``.
        """
        read = getattr(self._c, "sync_read_present_position", None)
        if read is None:
            raise NotImplementedError(
                "this rustypot SCS0009 controller has no sync_read_present_position"
            )
        ordered = []
        for finger_id, joints in FINGER_JOINTS.items():
            for name, (servo_id, zero_deg, _limits) in joints.items():
                ordered.append((finger_id, name, servo_id, zero_deg))
        ids = [x[2] for x in ordered]
        with self._lock:
            values = read(ids)
        if len(values) != len(ids):
            raise RuntimeError(f"sync position read returned {len(values)} values for {len(ids)} ids")
        out: dict[int, dict[str, float]] = {fid: {} for fid in FINGER_JOINTS}
        for (finger_id, name, _servo_id, zero_deg), raw_rad in zip(ordered, values):
            out[finger_id][name] = math.degrees(float(raw_rad)) - zero_deg
        return out

    @property
    def supports_sync_position_read(self) -> bool:
        return callable(getattr(self._c, "sync_read_present_position", None))

    @property
    def controller(self):
        """Escape hatch: the raw Scs0009PyController, for anything this
        wrapper doesn't cover."""
        return self._c


class Servo:
    def __init__(self, controller: "Scs0009PyController", servo_id: int,
                 lock: threading.RLock | None = None):
        self._c = controller
        self.id = servo_id
        self._lock = lock or threading.RLock()

    def _call(self, fn, *args):
        """Every controller RPC goes through here so INTER_CMD_DELAY_S is
        enforced uniformly -- see that constant's comment for why this
        exists at all. The delay is AFTER the call, not before: it's
        enforcing a minimum gap before the *next* call, whatever that turns
        out to be, not rate-limiting this one specifically.

        Retries on timeout: even with INTER_CMD_DELAY_S respected, isolated
        timeouts still happen occasionally on real hardware (confirmed --
        a whole enable/move/read sequence can succeed cleanly and then the
        very next read times out with no other change). Same situation as
        the TMC2209 bit-bang link earlier in this project: rather than
        chase a perfect delay constant that eliminates every last timeout,
        absorb the occasional miss with a bounded retry instead."""
        last_exc = None
        for _attempt in range(3):
            # Include success OR timeout recovery in the critical section: the
            # quiet period exists to protect the *next* packet on this shared bus,
            # regardless of which thread wants to issue it.
            with self._lock:
                try:
                    result = fn(self.id, *args)
                    time.sleep(INTER_CMD_DELAY_S)
                    return result
                except RuntimeError as exc:  # rustypot's "Operation timed out"
                    last_exc = exc
                    time.sleep(INTER_CMD_DELAY_S)
        raise last_exc

    def _call_verified(self, write_fn, read_fn, value, attempts=3, matches=None):
        """Like _call, but also reads back what was written and retries the
        whole write if it doesn't match. A write reporting local success (no
        exception, no timeout) does NOT mean the servo actually applied it
        -- confirmed on real hardware: write_torque_enable(ON) returned
        cleanly while read_torque_enable kept reporting OFF, leaving the
        servo silently ignoring every position command with no error
        anywhere to catch it, and write_goal_position had the identical
        problem. Same lesson as the TMC2209 bit-bang link's
        tmc_write_verified earlier in this project.

        matches: optional callable(readback_value, written_value) -> bool,
        defaults to exact equality. Position writes need a tolerance instead
        (see move_to_deg) since the servo's 10-bit resolution means even a
        genuinely successful write won't read back bit-for-bit identical to
        the float we sent."""
        if matches is None:
            matches = lambda readback, written: readback == written  # noqa: E731
        readback = None
        for _attempt in range(attempts):
            self._call(write_fn, value)
            readback = self._call(read_fn)
            if matches(readback[0], value):
                return
        raise RuntimeError(
            f"servo {self.id}: wrote {value} but read back {readback!r} after {attempts} attempts"
        )

    def enable(self):
        self._call_verified(self._c.write_torque_enable, self._c.read_torque_enable, TORQUE_ON)

    def disable(self):
        self._call_verified(self._c.write_torque_enable, self._c.read_torque_enable, TORQUE_OFF)

    def free(self):
        """Backdrivable by hand, no holding torque -- useful for manually
        posing a finger to find calibration offsets."""
        self._call_verified(self._c.write_torque_enable, self._c.read_torque_enable, TORQUE_FREE)

    def set_speed(self, speed: int):
        # Not verified like enable()/move_to_deg(): read_goal_speed returned
        # a value in a completely different unit/scale than what
        # write_goal_speed takes (a small float like -1.03 back for a
        # written 200), so a naive readback comparison here would just be
        # comparing incompatible numbers, not actually checking anything.
        # Left as a plain call until that conversion is understood.
        self._call(self._c.write_goal_speed, speed)

    def move_to_deg(self, angle_deg: float, speed: int | None = None):
        if speed is not None:
            self.set_speed(speed)
        target_rad = math.radians(angle_deg)
        self._call_verified(
            self._c.write_goal_position,
            self._c.read_goal_position,
            target_rad,
            matches=lambda readback, written: abs(readback - written) < POSITION_TOLERANCE_RAD,
        )

    @property
    def status(self) -> ServoStatus:
        # read_present_position returns a single-element list, e.g.
        # [-1.39...], not a bare float -- confirmed empirically; an earlier
        # version of this wrapper assumed a bare float based on a
        # third-party API summary and crashed with "must be real number,
        # not list" the first time it actually ran against real hardware.
        raw = self._call(self._c.read_present_position)
        return ServoStatus(position_deg=math.degrees(raw[0]))


# finger_id -> {joint_name: (servo_id, zero_deg, (min_rel_deg, max_rel_deg))}.
# One dedicated servo per DOF (aa/fe1/fe2), no shared/differential servos.
# zero_deg is each servo's own measured raw position_deg at its "0 degrees"
# reference (per-servo, not interchangeable between units -- confirmed
# repeatedly this session).
#
# (min_rel_deg, max_rel_deg) is EFFECTIVE range = intersection of the
# originally-declared nominal contract (aa was +/-85, fe1: -15..92, fe2:
# -18..92), the user's later conservative aa cap (+/-70), and each servo's
# REAL measured hardstop, from a manual
# torque-free sweep logged at 10Hz on 2026-08-29 (every servo freed one at
# a time, hand-moved through its full range, logged to
# host/examples/servo_manual_range.csv -- see servo_calibration_notes.md
# for the raw min/max readings this was derived from). Whichever bound is
# more restrictive wins, per instruction: commands should never exceed
# either the declared contract or the physical range, whichever is
# smaller. This REPLACES the earlier declared-only ranges, which were
# frequently wrong in both directions (some servos couldn't reach the
# declared value at all, others could go well past it).
FINGER_JOINTS = {
    # aa/yaw is intentionally capped at the user's conservative +/-70 deg
    # contract even where the manual sweep found a few more physical degrees.
    0: {"aa": (0, -41.0156, (-70.00, 70.00)), "fe1": (1, -34.2773, (-12.30, 89.06)), "fe2": (2, -43.9453, (-18.00, 86.72))},
    1: {"aa": (3, -10.2539, (-70.00, 70.00)), "fe1": (4, 84.9609, (-15.00, 64.75)), "fe2": (5, -146.7773, (-2.93, 92.00))},
    2: {"aa": (6, 12.89, (-70.00, 70.00)), "fe1": (7, 79.98046875, (-15.00, 69.73)), "fe2": (8, 72.6562, (-16.11, 77.05))},
}
DEFAULT_JOINT_SPEED = 80  # matches what's worked reliably all session


class Finger:
    """A finger is whatever named joints its FINGER_JOINTS entry declares
    (aa/fe1/fe2, each one dedicated servo). Direct single-servo access
    (self.servo(id)) is still available underneath for anything beyond
    named-joint moves.
    """

    def __init__(self, bus: ServoBus, finger_id: int):
        self._bus = bus
        self._finger_id = finger_id
        self._servo_ids = sorted(
            servo_id for servo_id, _zero_deg, _range in FINGER_JOINTS[finger_id].values()
        )

    @property
    def servo_ids(self) -> list[int]:
        return self._servo_ids

    def servo(self, servo_id: int) -> Servo:
        """Direct access to one of this finger's servos, e.g. for anything
        beyond named-joint moves."""
        return self._bus.servo(servo_id)

    def enable(self):
        for servo_id in self._servo_ids:
            self._bus.servo(servo_id).enable()

    def disable(self):
        for servo_id in self._servo_ids:
            self._bus.servo(servo_id).disable()

    def set_joint(self, name: str, relative_deg: float, speed: int = DEFAULT_JOINT_SPEED):
        """Move one named joint (aa, fe1, or fe2 -- see FINGER_JOINTS).
        relative_deg is plain degrees relative to this joint's servo's own
        zero_deg reference. Raises ValueError (nothing moves) if
        relative_deg falls outside FINGER_JOINTS' declared nominal range
        for this joint -- that range is a declared contract, not a
        guarantee of real reachability, so a write-verify RuntimeError from
        move_to_deg is still possible and propagates uncaught."""
        joints = FINGER_JOINTS[self._finger_id]
        if name not in joints:
            raise ValueError(f"finger {self._finger_id} has no joint {name!r} (have {list(joints)})")
        servo_id, zero_deg, (lo, hi) = joints[name]
        if not lo <= relative_deg <= hi:
            raise ValueError(f"finger {self._finger_id} {name}: {relative_deg} outside declared range [{lo}, {hi}]")
        self._bus.servo(servo_id).move_to_deg(zero_deg + relative_deg, speed=speed)

    def zero_joints(self, speed: int = DEFAULT_JOINT_SPEED):
        """Return this finger's aa/fe1/fe2 servos to their zero references."""
        for name in FINGER_JOINTS[self._finger_id]:
            self.set_joint(name, 0.0, speed=speed)
