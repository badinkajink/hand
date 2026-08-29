"""Serial transport to the manta-hand firmware's USB-CDC command port."""

from __future__ import annotations

import threading

import serial

from .protocol import MantaHandError, format_command


class MantaHandDriver:
    """One command in, one reply line out, lock-protected.

    Usage:
        with MantaHandDriver("/dev/ttyACM0") as hand:
            hand.joints[0].move_to(3200, velocity=1600, accel=4000)
    """

    def __init__(self, port: str = "/dev/ttyACM0", timeout: float = 2.0):
        # baudrate is meaningless over USB-CDC but pyserial requires a value
        self._ser = serial.Serial(port, baudrate=115200, timeout=timeout)
        self._lock = threading.Lock()

        from .joint import Joint  # local import to avoid a circular import at module load

        self.joints = [Joint(self, i) for i in range(8)]

    def close(self):
        self._ser.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def send(self, *parts) -> str:
        """Send one command line, return its reply's payload (text after OK/ERR)."""
        line = format_command(*parts)
        with self._lock:
            self._ser.reset_input_buffer()
            self._ser.write((line + "\n").encode("ascii"))
            reply = self._ser.readline().decode("ascii", errors="replace").strip()
        if not reply:
            raise MantaHandError(f"no reply to {line!r} (timeout)")
        if reply.startswith("ERR"):
            raise MantaHandError(f"{line!r} -> {reply}")
        if not reply.startswith("OK"):
            raise MantaHandError(f"{line!r} -> unexpected reply {reply!r}")
        return reply[2:].strip()

    def send_multiline(self, *parts, n_lines: int) -> list[str]:
        """Like send(), but for commands (STATALL) whose OK is followed by
        n_lines more lines of data."""
        line = format_command(*parts)
        with self._lock:
            self._ser.reset_input_buffer()
            self._ser.write((line + "\n").encode("ascii"))
            first = self._ser.readline().decode("ascii", errors="replace").strip()
            if first.startswith("ERR"):
                raise MantaHandError(f"{line!r} -> {first}")
            rest = []
            for _ in range(n_lines):
                l = self._ser.readline().decode("ascii", errors="replace").strip()
                if not l:
                    raise MantaHandError(f"{line!r}: expected {n_lines} lines, got fewer (timeout)")
                rest.append(l)
        return rest

    def stop_all(self):
        self.send("STOPALL")

    def get_all_status(self):
        from .joint import JointStatus

        lines = self.send_multiline("STATALL", n_lines=8)
        out = []
        for line in lines:
            position, target, moving, enabled, homing_result = line.split()
            out.append(JointStatus(int(position), int(target), bool(int(moving)), bool(int(enabled)),
                                    int(homing_result)))
        return out
