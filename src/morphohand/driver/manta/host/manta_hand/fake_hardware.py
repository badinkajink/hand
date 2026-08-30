"""Hardware fakes: a Manta M8P on a pty and an SCS0009 bus, both faithful enough
to run the real driver stack against.

Why this lives in the package and not in tests/: the two physical links are exclusive
and single-instance, so a second process cannot attach, and every mistake costs a
BOOT0/RESET cycle at the bench. Everything above the serial byte -- MantaHandDriver,
Joint, Gantry, Hand, ServoBus, HandRuntime, the HTTP service and the browser UI -- can
be exercised end to end against these instead, on a workstation, with no hardware and
no risk. `python -m manta_hand.web --fake` is that mode.

This is NOT MockHardwareBackend. The mock replaces the backend and therefore tests
none of the driver stack; these fakes replace the DEVICE, so every line of real driver
code runs. The behaviours modelled here are the ones that have actually bitten this
project, and they are documented at each site.
"""

from __future__ import annotations

import math
import os
import pty
import select
import threading
import time

NUM_AXES = 8


class FakeAxis:
    def __init__(self) -> None:
        self.position = 0
        self.target = 0
        self.enabled = False
        self.homing_result = 0
        self.steps_per_mm = 0.0
        self.velocity = 0.0
        self.homing_deadline = 0.0
        self.homing_stall_at: float | None = None
        self.moving_until = 0.0

    @property
    def moving(self) -> bool:
        return time.monotonic() < self.moving_until


class FakeM8P:
    """Serves the firmware protocol on a pty; `port` is the slave device path."""

    def __init__(self, *, stall_axes: set[int] | None = None, drop_lines: int = 0,
                 reply_delay_s: float = 0.0, time_scale: float = 0.02) -> None:
        self.axes = [FakeAxis() for _ in range(NUM_AXES)]
        # Which axes' StallGuard actually fires. Real hardware right now: J3 and J5
        # do not, and reach their outcome through the timeout path instead.
        self.stall_axes = {0, 1, 2, 4} if stall_axes is None else stall_axes
        self.drop_lines = drop_lines
        self.reply_delay_s = reply_delay_s
        self.time_scale = time_scale  # compress simulated travel so tests stay fast
        self.commands: list[str] = []
        # Set True to model the board vanishing from the USB bus: it accepts bytes and
        # never answers, which is what the host sees as a read timeout. This is the
        # failure that ends with the BOOT0/RESET cycle, and the one the runtime has to
        # latch rather than retry into.
        self.answering = True
        self._master, self._slave = pty.openpty()
        self.port = os.ttyname(self._slave)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="fake-m8p")
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        for fd in (self._master, self._slave):
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> "FakeM8P":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- wire ---------------------------------------------------------------------
    def _write(self, text: str) -> None:
        if not self.answering:
            return
        if self.reply_delay_s:
            time.sleep(self.reply_delay_s)
        os.write(self._master, (text + "\r\n").encode("ascii"))

    def _serve(self) -> None:
        buf = b""
        while not self._stop.is_set():
            try:
                # Non-blocking-ish read so the loop can notice _stop.
                ready, _, _ = select.select([self._master], [], [], 0.05)
                if not ready:
                    continue
                chunk = os.read(self._master, 4096)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    self._execute(line.decode("ascii", "replace").strip())
                except Exception as exc:  # a fake device must never take the test down
                    self._write(f"ERR FAKE {type(exc).__name__}")

    # -- protocol -----------------------------------------------------------------
    def _execute(self, line: str) -> None:
        if not line:
            return
        self.commands.append(line)
        parts = line.split()
        cmd = parts[0]

        if cmd == "STATALL":
            self._write("OK")
            lines = [self._status_body(i) for i in range(NUM_AXES)]
            # Model cdc_send_blocking giving up on a busy IN endpoint: the OK arrives,
            # some of the eight bodies never do.
            for body in lines[: NUM_AXES - self.drop_lines]:
                self._write(body)
            return
        if cmd == "STOPALL":
            for axis in self.axes:
                self._stop_axis(axis)
            self._write("OK")
            return

        if len(parts) < 2 or not parts[1].upper().startswith("J"):
            self._write("ERR BADARG")
            return
        try:
            index = int(parts[1][1:])
        except ValueError:
            self._write("ERR RANGE")
            return
        if not 0 <= index < NUM_AXES:
            self._write("ERR RANGE")
            return
        axis = self.axes[index]
        args = parts[2:]

        if cmd == "STAT":
            self._write("OK " + self._status_body(index))
        elif cmd == "EN":
            axis.enabled = True
            self._write("OK")
        elif cmd == "DIS":
            axis.enabled = False
            axis.moving_until = 0.0
            axis.homing_result = 0  # firmware Stepper_Disable clears this
            self._write("OK")
        elif cmd == "STOP":
            self._stop_axis(axis)
            self._write("OK")
        elif cmd == "ZERO":
            axis.position = 0
            axis.target = 0
            self._write("OK")
        elif cmd == "SETSCALE":
            axis.steps_per_mm = float(args[0])
            self._write("OK")
        elif cmd in ("WREG5160", "WREG", "CUR", "USTEP", "SETRUN5160"):
            self._write("OK")
        elif cmd in ("RREG", "RREG5160", "RREGA"):
            self._write("OK 00000000")
        elif cmd == "MOVEMM":
            if not axis.steps_per_mm:
                self._write("ERR UNCALIBRATED")
                return
            self._start_move(axis, int(float(args[0]) * axis.steps_per_mm))
            self._write("OK")
        elif cmd == "MOVE":
            self._start_move(axis, int(args[0]))
            self._write("OK")
        elif cmd == "JOG":
            self._write("OK")
        elif cmd == "HOME":
            if not axis.enabled:
                self._write("ERR NODIAG")
                return
            timeout_s = float(args[3]) / 1000.0 * self.time_scale
            axis.homing_result = 1
            axis.homing_deadline = time.monotonic() + timeout_s
            # A stalling axis finds its hardstop partway through the window; a
            # non-stalling one runs the window out and reports 3.
            axis.homing_stall_at = (time.monotonic() + timeout_s * 0.4
                                    if index in self.stall_axes else None)
            axis.moving_until = axis.homing_deadline
            self._write("OK")
        else:
            self._write("ERR UNKNOWN")

    def _start_move(self, axis: FakeAxis, target_steps: int) -> None:
        distance = abs(target_steps - axis.position)
        axis.target = target_steps
        axis.position = target_steps  # firmware counts steps out regardless of physics
        axis.moving_until = time.monotonic() + (distance / 12000.0) * self.time_scale

    @staticmethod
    def _stop_axis(axis: FakeAxis) -> None:
        # Exactly the firmware's Stepper_Stop: target follows position, homing_result
        # is deliberately left alone.
        axis.target = axis.position
        axis.moving_until = 0.0

    def _status_body(self, index: int) -> str:
        axis = self.axes[index]
        now = time.monotonic()
        if axis.homing_result == 1:
            if axis.homing_stall_at is not None and now >= axis.homing_stall_at:
                axis.homing_result = 2
                self._stop_axis(axis)
            elif now >= axis.homing_deadline:
                axis.homing_result = 3
                self._stop_axis(axis)
        return (f"{axis.position} {axis.target} {int(axis.moving)} "
                f"{int(axis.enabled)} {axis.homing_result}")



TORQUE_ON = 1
TORQUE_OFF = 2
TORQUE_FREE = 3

NUM_SERVOS = 9


class FakeScs0009Controller:
    """Matches the subset of rustypot's API that manta_hand.servos actually calls."""

    def __init__(self, serial_port: str = "/dev/fake", baudrate: int = 1_000_000,
                 timeout: float = 0.5, **_kwargs) -> None:
        self.port = serial_port
        self._lock = threading.Lock()
        self.goal_position = {i: 0.0 for i in range(NUM_SERVOS)}
        self.present_position = {i: 0.0 for i in range(NUM_SERVOS)}
        self.goal_speed = {i: 0 for i in range(NUM_SERVOS)}
        self.torque_enable = {i: TORQUE_OFF for i in range(NUM_SERVOS)}
        self.present_load = {i: 0.0 for i in range(NUM_SERVOS)}
        self.alarms: dict[int, int] = {}
        self.writes = 0
        self.sync_writes = 0
        self.reads = 0
        # Set to a servo id to make every transaction to it time out, the way a
        # servo that has dropped off the daisy chain does.
        self.dead_ids: set[int] = set()
        self.supports_sync_read = True

    # -- helpers ------------------------------------------------------------------
    def _check(self, ids) -> None:
        for sid in (ids if isinstance(ids, (list, tuple)) else [ids]):
            if sid in self.dead_ids:
                raise RuntimeError("Operation timed out")

    def _settle(self, sid: int) -> None:
        """Torque OFF means the goal register is accepted but the horn does not move."""
        if self.torque_enable[sid] == TORQUE_ON:
            self.present_position[sid] = self.goal_position[sid]

    # -- single-servo API ---------------------------------------------------------
    def write_torque_enable(self, sid: int, value: int) -> None:
        self._check(sid)
        self.writes += 1
        self.torque_enable[sid] = int(value)
        self._settle(sid)

    def read_torque_enable(self, sid: int):
        self._check(sid)
        self.reads += 1
        return [self.torque_enable[sid]]

    def write_goal_position(self, sid: int, radians: float) -> None:
        self._check(sid)
        self.writes += 1
        self.goal_position[sid] = float(radians)
        self._settle(sid)

    def read_goal_position(self, sid: int):
        self._check(sid)
        self.reads += 1
        return [self.goal_position[sid]]

    def read_present_position(self, sid: int):
        self._check(sid)
        self.reads += 1
        return [self.present_position[sid]]

    # The per-servo reads the driver actually uses. SYNC READ is unsupported by the real
    # SCS0009 (it times out on every field), so everything reads one servo at a time --
    # these are the calls that matter, not the sync_* ones below.
    def read_present_load(self, sid: int):
        self._check(sid)
        self.reads += 1
        return [self.present_load[sid]]

    def read_present_speed(self, sid: int):
        self._check(sid)
        self.reads += 1
        return [0.0]

    def read_present_voltage(self, sid: int):
        self._check(sid)
        self.reads += 1
        return [50.0]

    def read_present_temperature(self, sid: int):
        self._check(sid)
        self.reads += 1
        return [20.0]

    def read_status(self, sid: int):
        """Latched alarm byte; 0 is healthy. Set `alarms[sid]` to model a fault."""
        self._check(sid)
        self.reads += 1
        return [self.alarms.get(sid, 0)]

    def write_goal_speed(self, sid: int, value: int) -> None:
        self._check(sid)
        self.writes += 1
        self.goal_speed[sid] = int(value)

    def read_goal_speed(self, sid: int):
        self._check(sid)
        self.reads += 1
        return [-1.03]  # the real one really does answer in an unrelated scale

    # -- sync API -----------------------------------------------------------------
    def sync_write_goal_position(self, ids, values) -> None:
        self._check(ids)
        self.sync_writes += 1
        for sid, value in zip(ids, values):
            self.goal_position[sid] = float(value)
            self._settle(sid)

    def sync_write_goal_speed(self, ids, values) -> None:
        self._check(ids)
        self.sync_writes += 1
        for sid, value in zip(ids, values):
            self.goal_speed[sid] = int(value)

    def sync_write_torque_enable(self, ids, values) -> None:
        self._check(ids)
        self.sync_writes += 1
        for sid, value in zip(ids, values):
            self.torque_enable[sid] = int(value)
            self._settle(sid)

    def sync_read_torque_enable(self, ids):
        self._check(ids)
        self.reads += 1
        return [self.torque_enable[sid] for sid in ids]

    def sync_read_present_position(self, ids):
        if not self.supports_sync_read:
            raise AttributeError("sync_read_present_position")
        self._check(ids)
        self.reads += 1
        return [self.present_position[sid] for sid in ids]

    def sync_read_present_load(self, ids):
        self._check(ids)
        self.reads += 1
        return [self.present_load[sid] for sid in ids]

    # -- test conveniences --------------------------------------------------------
    def position_degrees(self) -> list[float]:
        return [math.degrees(self.present_position[i]) for i in range(NUM_SERVOS)]

    def all_torque_on(self) -> bool:
        return all(v == TORQUE_ON for v in self.torque_enable.values())


def install(monkeypatch) -> type[FakeScs0009Controller]:
    """Point manta_hand.servos at the fake controller for the duration of a test."""
    from manta_hand import servos

    monkeypatch.setattr(servos, "_controller_cls", lambda: FakeScs0009Controller)
    return FakeScs0009Controller
