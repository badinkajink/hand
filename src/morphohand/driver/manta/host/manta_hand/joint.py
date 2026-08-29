"""Per-joint convenience wrapper. See docs/protocol.md for the wire format."""

from __future__ import annotations

from dataclasses import dataclass

from .protocol import MantaHandError


@dataclass
class JointStatus:
    position: int
    target: int
    moving: bool
    enabled: bool
    homing_result: int
    """0=idle, 1=homing in progress, 2=stalled (position is the home
    reference, call Joint.zero() explicitly if you want that to become 0),
    3=timed out with no stall seen -- don't trust position as a home
    reference in that case."""


class Joint:
    def __init__(self, driver, index: int):
        self._driver = driver
        self.index = index

    def _j(self) -> str:
        return f"J{self.index}"

    def enable(self):
        self._driver.send("EN", self._j())

    def disable(self):
        self._driver.send("DIS", self._j())

    def write_reg5160(self, reg: int, value: int):
        """Raw TMC5160 register write (RAM-only, lost on power-cycle). Used
        for COOLCONF (0x6D) to set this axis's StallGuard2 SGT threshold
        before home() -- see docs/bringup.md for how SGT was tuned per axis."""
        self._driver.send("WREG5160", self._j(), f"{reg:X}", f"{value:X}")

    def set_current(self, run_ma: int, hold_ma: int):
        """run_ma/hold_ma in milliamps -- set these from your NEMA8's actual
        datasheet rating, not a guess. See docs/bringup.md."""
        self._driver.send("CUR", self._j(), run_ma, hold_ma)

    def set_microsteps(self, usteps: int):
        if usteps not in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            raise ValueError(f"microsteps must be a power of two 1-256, got {usteps}")
        self._driver.send("USTEP", self._j(), usteps)

    def move_to(self, steps: int, velocity: int, accel: int):
        """Absolute step position, trapezoidal profile."""
        self._driver.send("MOVE", self._j(), steps, velocity, accel)

    def set_scale(self, steps_per_mm: float):
        """RAM-only mm calibration for move_to_mm -- lost on power-cycle,
        re-set every session before the first move_to_mm call."""
        self._driver.send("SETSCALE", self._j(), steps_per_mm)

    def move_to_mm(self, mm: float, velocity: int, accel: int):
        """Absolute position in mm, trapezoidal profile. 0mm is wherever this
        axis's step-position 0 currently is (home, or ZERO), so home first
        and call set_scale() this session before using this."""
        self._driver.send("MOVEMM", self._j(), mm, velocity, accel)

    def jog(self, velocity: int, accel: int):
        """Continuous velocity move; jog(0, accel) ramps to a stop."""
        self._driver.send("JOG", self._j(), velocity, accel)

    def stop(self):
        self._driver.send("STOP", self._j())

    def zero(self):
        """Zero the step counter at the joint's current physical position.
        Also usable after a successful home() -- see home()'s docstring --
        to make the stall position the new step-0 reference."""
        self._driver.send("ZERO", self._j())

    def home(self, direction: int, velocity: int, accel: int, timeout_ms: int):
        """Sensorless homing via the TMC2209's DIAG/StallGuard4 output --
        needs SGTHRS and TCOOLTHRS already configured AND empirically tuned
        on this joint's driver first (see docs/bringup.md), and a DIAG pin
        physically jumpered + wired for it (only J0/J1 as of this writing --
        raises RuntimeError for any other joint, or one that isn't
        enabled). Non-blocking, same pattern as move_to/jog: poll `status`
        afterward -- .homing_result becomes 2 (stalled, .position is now the
        home reference) or 3 (timed out, don't trust .position) once it's
        done. direction is +1 or -1; sign of anything else is used."""
        d = 1 if direction >= 0 else -1
        try:
            self._driver.send("HOME", self._j(), d, velocity, accel, timeout_ms)
        except MantaHandError as e:
            raise MantaHandError(
                f"{self._j()}: can't home -- no DIAG pin wired for this joint yet, "
                f"or it isn't enabled ({e})"
            ) from e

    @property
    def status(self) -> JointStatus:
        body = self._driver.send("STAT", self._j())
        position, target, moving, enabled, homing_result = body.split()
        return JointStatus(int(position), int(target), bool(int(moving)), bool(int(enabled)),
                            int(homing_result))
