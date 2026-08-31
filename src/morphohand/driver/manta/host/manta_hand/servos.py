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
import os
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

# The value a servo reports when nothing has written the register since it last came
# up. Confirmed on hardware 2026-08-29: a chain that had been explicitly disabled read
# back 2, and after a power cycle the same nine read 0. Functionally 0 and 2 are both
# "not holding", but they are not interchangeable as a diagnostic -- a servo that reads
# 0 AFTER we set it to 1 or 2 has reset itself underneath us, which is the closest
# thing this bus has to the packet-storm/servo-reboot signal a Dynamixel chain gives.
TORQUE_UNSET = 0
TORQUE_NOT_HOLDING = (TORQUE_UNSET, TORQUE_OFF)

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

# MEASURED 2026-08-29 on the real nine-servo chain, which refines the rule above: the
# gap is a WRITE requirement, not a bus-wide one.
#
#   reads only, no gap at all      111 Hz,  0 errors in 6003 transactions
#   sync write then a read sweep    12 Hz, 14 errors in 91 attempts  (15% failure)
#   alternating write/read          46 Hz,  2 errors in 278 attempts
#
# So a transaction issued too soon after a WRITE fails; a read issued straight after
# another read never did. ServoBus therefore keeps the full gap around every write --
# before it as well as after, so a write is never the thing issued too soon either --
# and lets consecutive reads run at bus speed. Every ordering is at least as protected
# as it was under the original blanket delay; only read-after-read got faster.
#
# Re-run examples/benchmark_servo_telemetry.py before trusting these numbers on
# different servos, a different adapter, or a longer chain.
INTER_READ_DELAY_S = 0.0

# Gap a READ waits out after a SYNC write (sync_set_joints), as opposed to after one of
# the slow verified individual writes, which keep the full INTER_CMD_DELAY_S above.
# Measured on the real chain 2026-08-29, writing each servo the pose it was already
# holding (nothing commanded to move), 9 writes + 9 reads per cycle:
#
#   sync writes alone, back to back    2859 Hz,  0 errors / 14298
#   write ->  0ms -> 9 reads             90 Hz,  1 error  / 451
#   write ->  1ms -> 9 reads            100 Hz,  0 errors / 500
#   write ->  2ms -> 9 reads             91 Hz,  0 errors / 454
#   write -> 10ms -> 9 reads             52 Hz,  0 errors / 263
#
# 2ms is the setting: it clears the errors with margin and still leaves ~90Hz, well
# above the 50Hz a closed loop on this hand would want. An earlier measurement under
# different conditions (torque off, writing an extreme target rather than the held
# pose) saw 15% failures at 0ms and has NOT been reproduced -- which is exactly why
# this is 2ms and not 0.
POST_SYNC_WRITE_READ_GAP_S = 0.002

# Settle time after opening the port, before the first transaction. Same empirical
# story as INTER_CMD_DELAY_S; named so a fake bus (fake_hardware.py) and the test
# suite can set it to 0 without patching a literal.
PORT_SETTLE_S = 1.0


# The FT232H inside the U2D2 buffers incoming bytes until either its buffer fills or
# its latency timer expires, and ftdi_sio defaults that timer to 16ms. Every SCS
# response is a handful of bytes, so it never fills anything, and every single read
# therefore costs a flat 16ms no matter what was asked for. Measured on this hardware
# 2026-08-29: nine per-servo position reads took 144.0ms (6.95Hz) at the default, and
# 9.0ms (111Hz) with the timer at 1ms -- 16x, zero errors over 445 bundles. At 1 Mbaud
# the actual wire time for one exchange is ~0.15ms, so the timer was 99% of the cost.
#
# This is the single highest-leverage number on the servo side: it is the difference
# between "the bus polls at 10Hz so closed loop is out of the question" and a 111Hz
# position feed. It is also invisible -- nothing errors, everything just runs slowly.
FTDI_LATENCY_PATH = "/sys/bus/usb-serial/devices/{name}/latency_timer"
FTDI_TARGET_LATENCY_MS = 1


def ftdi_latency_ms(port: str) -> int | None:
    """This port's FTDI latency timer in ms, or None if it is not an FTDI port
    (nothing to tune) or the attribute cannot be read."""
    name = os.path.basename(port)
    try:
        with open(FTDI_LATENCY_PATH.format(name=name)) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def tune_ftdi_latency(port: str, target_ms: int = FTDI_TARGET_LATENCY_MS) -> dict:
    """Try to lower this port's latency timer, and report what happened.

    Writing the sysfs attribute needs root, which a service should not assume it has,
    so a refusal is reported rather than raised -- the bus still works at 16ms, just
    16x slower. Make it permanent with a udev rule (see docs/hardware_control_station.md):

        SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
    """
    before = ftdi_latency_ms(port)
    if before is None:
        return {"supported": False, "before_ms": None, "after_ms": None,
                "note": "not an ftdi_sio port; nothing to tune"}
    if before <= target_ms:
        return {"supported": True, "before_ms": before, "after_ms": before,
                "note": "already tuned"}
    name = os.path.basename(port)
    try:
        with open(FTDI_LATENCY_PATH.format(name=name), "w") as f:
            f.write(str(target_ms))
    except OSError as exc:
        return {"supported": True, "before_ms": before, "after_ms": before,
                "note": (f"cannot write latency_timer ({exc}); servo reads will cost "
                         f"~{before}ms each. Fix with: echo {target_ms} | sudo tee "
                         f"{FTDI_LATENCY_PATH.format(name=name)}")}
    after = ftdi_latency_ms(port)
    return {"supported": True, "before_ms": before, "after_ms": after,
            "note": f"lowered {before}ms -> {after}ms"}


def _torque_matches(readback: int | None, requested: int) -> bool:
    """A servo asked to stop holding satisfies the request whether it reports an
    explicit disable (2) or the power-on default (0). Only 1 counts as holding."""
    if readback is None:
        return False
    if requested in TORQUE_NOT_HOLDING:
        return readback in TORQUE_NOT_HOLDING
    return readback == requested


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
        # Do this BEFORE the first transaction: it is a 16x difference in every read
        # this bus will ever do, and it is silent when it fails.
        self.latency_tuning = tune_ftdi_latency(port)
        time.sleep(PORT_SETTLE_S)
        self._servos: dict[int, Servo] = {}
        # Bus health, for the operator UI. SCS0009 timeouts are absorbed by
        # Servo._call's retry, which means a bus that is degrading looks
        # exactly like a healthy one from the outside until it stops working
        # altogether. Counting them is what makes the degradation visible.
        self.timeouts = 0
        self.transactions = 0
        self.consecutive_timeouts = 0
        # Two independent quiet periods, per the measurements at INTER_READ_DELAY_S and
        # POST_SYNC_WRITE_READ_GAP_S. Reads wait for the first; writes wait only for
        # the second, which only the slow verified write path ever sets -- so a
        # trajectory of consecutive sync writes is never throttled by its own history.
        self._quiet_until_read = 0.0
        self._quiet_until_write = 0.0

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
            self._servos[servo_id] = Servo(self._c, servo_id, self._lock, bus=self)
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
            # NO _wait_for_quiet here, deliberately. This is the one real-time path on
            # the bus and it is a single transaction; making it wait out the write gap
            # caps a 50Hz trajectory at 3.33Hz. The gap protects a transaction issued
            # too soon after a write -- it is not a rate limit on writes themselves,
            # and consecutive sync writes were the collaborator's proven fast path long
            # before the gap existed. _note_call still runs, so a READ that follows
            # still waits; see INTER_READ_DELAY_S and _wait_for_quiet.
            if speed is not None:
                self._c.sync_write_goal_speed(ids=ids, values=[speed] * len(ids))
            self._c.sync_write_goal_position(ids=ids, values=values)
            self._note_call(True, fast=True)

    def sync_read_joint_positions(self) -> dict[int, dict[str, float]]:
        """All nine joint positions, in the zero-relative degrees `sync_set_joints` takes.

        Still named for the SYNC READ it was built on; it no longer uses one, because
        SYNC READ DOES NOT WORK ON THIS HARDWARE. The SCS protocol manual documents READ
        and SYNC WRITE but not SYNC READ, and `rustypot` exposes
        `sync_read_present_position` for the SCS0009 regardless. Measured on the real bus
        2026-08-29: every sync read of every field (position, load, speed, voltage,
        temperature, status, torque_enable) times out after 500ms, every time, while all
        nine servos answer a plain per-servo READ immediately. A telemetry loop built on
        the sync path produces nothing but timeouts -- which is what `--telemetry-hz 0`
        was quietly hiding.

        So this reads per servo. Nine transactions instead of one, which is exactly why
        the FTDI latency timer matters: measured 144ms (6.9Hz) for all nine at the 16ms
        ftdi_sio default, 9.0ms (111Hz) at 1ms. See tune_ftdi_latency."""
        out: dict[int, dict[str, float]] = {fid: {} for fid in FINGER_JOINTS}
        for finger_id, joints in FINGER_JOINTS.items():
            for name, (servo_id, zero_deg, _limits) in joints.items():
                raw = self.servo(servo_id)._call(self._c.read_present_position)[0]
                out[finger_id][name] = math.degrees(float(raw)) - zero_deg
        return out

    def read_alarms(self) -> dict[int, int] | None:
        """Each servo's status/error byte, or None if unavailable. Non-zero means the
        servo has latched a fault (overload, over-temperature, over-voltage).

        This is the SCS0009's own answer to the packet-overload watchdog question. It
        costs the same nine transactions as any other field, so poll it occasionally,
        not per frame."""
        read = getattr(self._c, "read_status", None)
        if read is None:
            return None
        return {servo_id: int(self.servo(servo_id)._call(read)[0])
                for joints in FINGER_JOINTS.values()
                for servo_id, _zero, _limits in joints.values()}

    # -- torque ------------------------------------------------------------------------
    def set_torque_all(self, state: int, *, attempts: int = 3) -> dict[int, int]:
        """Put every one of the nine servos into `state` (TORQUE_ON/OFF/FREE) and verify.

        Nine `Servo.enable()` calls cost ~5.5s (each is a verified write plus a read,
        each followed by INTER_CMD_DELAY_S) which is far too slow to run at startup or
        from a UI button. One sync_write_torque_enable is a single bus transaction (SYNC
        WRITE the SCS manual does document, unlike SYNC READ), and nine plain reads
        confirm it -- ~10 transactions instead of ~54.

        Verification is not optional here. A torque_enable write reporting local success
        while the servo ignored it is a documented failure of this exact part (see
        Servo._call_verified), and a servo left with torque OFF accepts goal-position
        writes, reads them back correctly, and never moves. That silent mode is what
        made a home look like it had zeroed nine joints that had not moved at all.

        Returns {servo_id: state actually read back}. Raises RuntimeError naming the
        servos that would not take the state after `attempts` rounds."""
        ids = sorted(sid for joints in FINGER_JOINTS.values()
                     for sid, _zero, _lim in joints.values())
        readback: dict[int, int] = {}
        for _round in range(attempts):
            with self._lock:
                self._wait_for_quiet(True)
                self._c.sync_write_torque_enable(ids=ids, values=[state] * len(ids))
                self._note_call(True)
            readback = self.read_torque_all()
            bad = [sid for sid in ids if not _torque_matches(readback.get(sid), state)]
            if not bad:
                return readback
            # Fall back to the per-servo verified path for the stragglers: it retries
            # on timeout and is the one write path with a proven track record.
            for sid in bad:
                try:
                    self.servo(sid)._call_verified(self._c.write_torque_enable,
                                                    self._c.read_torque_enable, state)
                except Exception:
                    pass
            readback = self.read_torque_all()
            if all(_torque_matches(readback.get(sid), state) for sid in ids):
                return readback
        bad = [sid for sid in ids if not _torque_matches(readback.get(sid), state)]
        raise RuntimeError(
            f"servos {bad} would not accept torque state {state} after {attempts} rounds "
            f"(read back {[readback.get(sid) for sid in bad]}); check servo power and wiring"
        )

    def read_torque_all(self) -> dict[int, int]:
        ids = sorted(sid for joints in FINGER_JOINTS.values()
                     for sid, _zero, _lim in joints.values())
        # Per-servo READ, not sync: sync reads time out on this hardware (see
        # sync_read_joint_positions). Nine transactions, ~9ms total at a 1ms FTDI
        # latency timer, ~144ms at the 16ms default.
        return {sid: int(self.servo(sid)._call(self._c.read_torque_enable)[0])
                for sid in ids}

    def enable_all(self) -> dict[int, int]:
        return self.set_torque_all(TORQUE_ON)

    def disable_all(self) -> dict[int, int]:
        return self.set_torque_all(TORQUE_OFF)

    def free_all(self) -> dict[int, int]:
        """Backdrivable, no holding torque -- for posing a finger by hand."""
        return self.set_torque_all(TORQUE_FREE)

    # -- extra feedback ----------------------------------------------------------------
    def sync_read_field(self, field: str) -> dict[int, float] | None:
        """Deprecated alias for read_field. Kept so existing callers keep working; the
        sync path it was named for does not work on this hardware."""
        return self.read_field(field)

    def read_field(self, field: str) -> dict[int, float] | None:
        """One per-servo READ of `present_<field>` across all nine, or None if this
        rustypot build does not expose it.

        The SCS0009 has NO present-current register -- confirmed against the installed
        rustypot register table. `present_load` is the closest thing and it is an
        uncalibrated duty-cycle-like number, not amps and not newtons. Nothing here converts it to force; it is exposed so a run log can
        carry it as a raw covariate alongside the manual score, which is the only
        honest use for it until someone calibrates it against a load cell.

        Each extra field costs another nine transactions: measured 111Hz for position
        alone and 55.6Hz for position+load, at a 1ms FTDI latency timer."""
        read = getattr(self._c, f"read_present_{field}", None)
        if read is None:
            return None
        return {servo_id: float(self.servo(servo_id)._call(read)[0])
                for joints in FINGER_JOINTS.values()
                for servo_id, _zero, _limits in joints.values()}

    def available_feedback_fields(self) -> list[str]:
        """Which present_* fields the installed rustypot can sync-read. Costs nothing:
        an attribute check, no bus traffic."""
        return [f for f in ("position", "load", "speed", "voltage", "temperature", "current")
                if callable(getattr(self._c, f"read_present_{f}", None))]

    def suspected_resets(self, expected: int) -> list[int]:
        """Servos reporting the power-on default when we last set them to something
        else -- i.e. servos that rebooted. Costs nine reads; call it when something
        looks wrong, not per frame."""
        if expected == TORQUE_UNSET:
            return []
        return sorted(sid for sid, value in self.read_torque_all().items()
                      if value == TORQUE_UNSET)

    def health(self) -> dict:
        """Counters for the operator UI. `consecutive_timeouts` is the one that matters:
        an SCS chain that has started dropping packets keeps working, slowly and
        intermittently, long before it fails outright."""
        return {"transactions": self.transactions, "timeouts": self.timeouts,
                "consecutive_timeouts": self.consecutive_timeouts,
                "timeout_rate": (self.timeouts / self.transactions) if self.transactions else 0.0,
                "ftdi_latency": self.latency_tuning}

    def _wait_for_quiet(self, is_write: bool) -> None:
        """Sleep until this transaction is safe to issue.

        Reads and writes wait on different clocks, because the hazard is asymmetric:
        what fails is a transaction issued too soon after a write, not a write issued
        at any rate. See the measurements at INTER_READ_DELAY_S and
        POST_SYNC_WRITE_READ_GAP_S."""
        until = self._quiet_until_write if is_write else self._quiet_until_read
        wait = until - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def _note_call(self, is_write: bool, *, fast: bool = False) -> None:
        """Record a completed transaction.

        `fast=True` marks the one-transaction sync write path: it makes a following
        READ wait POST_SYNC_WRITE_READ_GAP_S, and imposes nothing at all on a following
        write, so a 50Hz trajectory runs at its own rate. `fast=False` is the slow
        verified individual-write path and keeps the collaborator's full
        INTER_CMD_DELAY_S on everything after it."""
        if not is_write:
            return  # a read imposes nothing on anything
        now = time.monotonic()
        if fast:
            self._quiet_until_read = max(self._quiet_until_read,
                                          now + POST_SYNC_WRITE_READ_GAP_S)
        else:
            self._quiet_until_read = max(self._quiet_until_read, now + INTER_CMD_DELAY_S)
            self._quiet_until_write = max(self._quiet_until_write, now + INTER_CMD_DELAY_S)

    def _note_transaction(self, timed_out: bool) -> None:
        self.transactions += 1
        if timed_out:
            self.timeouts += 1
            self.consecutive_timeouts += 1
        else:
            self.consecutive_timeouts = 0

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
                 lock: threading.RLock | None = None, bus: "ServoBus | None" = None):
        self._c = controller
        self.id = servo_id
        self._lock = lock or threading.RLock()
        self._bus = bus  # for health accounting only; None when built standalone

    def _call(self, fn, *args, is_write: bool | None = None):
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
        if is_write is None:
            # Anything not named read_* is treated as a write, so an unrecognised call
            # gets the conservative timing rather than the fast one.
            is_write = not getattr(fn, "__name__", "write").startswith("read_")
        last_exc = None
        for _attempt in range(3):
            # Include success OR timeout recovery in the critical section: the quiet
            # period exists to protect the next packet on this shared bus, whichever
            # thread wants to issue it.
            with self._lock:
                self._wait_for_quiet(is_write)
                try:
                    result = fn(self.id, *args)
                    self._note(False)
                    self._mark(is_write)
                    return result
                except RuntimeError as exc:  # rustypot's "Operation timed out"
                    last_exc = exc
                    self._note(True)
                    # A timed-out transaction leaves the bus in an unknown state; give
                    # it the full slow-write quiet period before anything else goes out.
                    self._mark(True)
        raise last_exc

    def _mark(self, is_write: bool) -> None:
        if self._bus is not None:
            self._bus._note_call(is_write)

    def _wait_for_quiet(self, is_write: bool) -> None:
        if self._bus is not None:
            self._bus._wait_for_quiet(is_write)
        else:
            time.sleep(INTER_CMD_DELAY_S)

    def _note(self, timed_out: bool) -> None:
        """Report the outcome to the owning bus's health counters, when there is one.
        The retry above is what makes a degrading bus invisible; this is what makes it
        visible again."""
        if self._bus is not None:
            self._bus._note_transaction(timed_out)

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
# declared nominal contract (aa +/-85, fe1: -15..92, fe2: -18..92) and each
# servo's REAL measured hardstop, from a manual
# torque-free sweep logged at 10Hz on 2026-08-29 (every servo freed one at
# a time, hand-moved through its full range, logged to
# host/examples/servo_manual_range.csv -- see servo_calibration_notes.md
# for the raw min/max readings this was derived from). Whichever bound is
# more restrictive wins, per instruction: commands should never exceed
# either the declared contract or the physical range, whichever is
# smaller. This REPLACES the earlier declared-only ranges, which were
# frequently wrong in both directions (some servos couldn't reach the
# declared value at all, others could go well past it).

# aa/yaw's live command cap. HISTORY, because this number has moved twice and
# both moves changed which plans exist:
#
#   +-85  declared contract, and what assets/mjcf/real_v1/real_hand.xml's yaw
#         joints still declare (test_manta_frame_map pins that identification).
#   +-70  a conservative cap the user set on 2026-08-29, before any plan had
#         been driven. It later turned out to be a SCREEN, not a safety margin:
#         it is the sole reason sv1_u0060_b100 (needs 73.88) and sv1_w0116_b100
#         (needs 76.27) could not be loaded, and both overruns are middle_yaw at
#         turn_end -- i.e. the cap was deciding how far the hand may turn.
#   +-85  restored 2026-08-31 at the user's request, so those two plans load.
#
# What the manual sweep actually demonstrated for the three aa servos, in this
# module's own zero-relative degrees:
#
#   servo 0 (thumb  aa)   -70.02 .. +74.71
#   servo 3 (index  aa)   -79.69 .. +74.41
#   servo 6 (middle aa)  -162.60 .. +136.82
#
# Only servo 6 -- the one both restored plans need -- was demonstrated past 85 in
# both directions. Thumb aa has never been shown past -70.02, and that is a gap in
# the evidence rather than a measured hardstop: the sweep is a person moving a freed
# joint, so it records where they stopped, not necessarily where the joint does. No
# deployed plan commands thumb aa past -23.1, so nothing currently rides on it; if a
# future plan does, sweep servo 0 again before trusting the bound.
AA_LIMIT_DEG = (-85.00, 85.00)
FINGER_JOINTS = {
    0: {"aa": (0, -41.0156, AA_LIMIT_DEG), "fe1": (1, -34.2773, (-12.30, 89.06)), "fe2": (2, -43.9453, (-18.00, 86.72))},
    1: {"aa": (3, -10.2539, AA_LIMIT_DEG), "fe1": (4, 84.9609, (-15.00, 64.75)), "fe2": (5, -146.7773, (-2.93, 92.00))},
    2: {"aa": (6, 12.89, AA_LIMIT_DEG), "fe1": (7, 79.98046875, (-15.00, 69.73)), "fe2": (8, 72.6562, (-16.11, 77.05))},
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
