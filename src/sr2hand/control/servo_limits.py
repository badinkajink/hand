"""Command ranges of the SR2 hand's nine servos: the driver's calibrated ranges (manta_hand.servos.FINGER_JOINTS) less a
margin where a range ends at the SCS0009's electrical end (owner, 2026-11-17: "i agree on the HOM target clipping").

An SCS0009 reads 1024 steps over 300 deg; the driver's servo degrees, (steps - 511) x 300 / 1024, run from -149.71 to
+149.99. Four calibrated limits sit on those ends: the index PIP's extension at -2.93 deg from its zero (servo -149.71),
and the flexion of the index MCP (64.75), the middle MCP (69.73) and the middle PIP (77.05), at servo +149.71. The
potentiometer's track ends before them. On 2026-11-16 the index and middle MCPs' readings jumped by about -110 deg at 62
and 68 deg (servo 147.0 and 148.0) as the owner curled them by hand. On 2026-11-17 at 09:21 a HOM twist drove the index
PIP to -2.9 deg; its reading then jumped to +184-189 deg, and the servo, blind, pushed against its stop at its overload
protection. A command here stays within POT_SAFE_DEG of the servo's middle, 5 deg from the ends and 2-3 deg short of
those jumps: the index PIP at +1.78 deg or more, the index MCP at 60.04 deg or less, the middle MCP at 65.02 and the
middle PIP at 72.34. The other five ranges are the calibrated ones.

The index PIP's zero (servo -146.78) lies 1.78 deg past that line. The station's zero and open poses still go there, as
they did many times without a jump; the HOM targets are clipped (hom_bench.HomBench.step and hom_server on the server,
hom_trial.py's frames on corvus as a backstop). Plain Python over the driver's tables: no rustypot, no numpy.
"""
from __future__ import annotations

import sys
from pathlib import Path

POT_SAFE_DEG = 145.0                      # servo degrees from the middle of the 300 deg range a command may reach
FINGERS = ("thumb", "index", "middle")
SIM_JOINTS = ("yaw", "mcp", "pip")
_HOST = Path(__file__).resolve().parents[1] / "driver/manta/host"


def _driver():
    if str(_HOST) not in sys.path:
        sys.path.insert(0, str(_HOST))
    from manta_hand import plan as P
    from manta_hand import servos as SV
    return P, SV


def calibrated_range(finger: str, servo: str) -> tuple[float, float]:
    """The driver's range of one servo (aa, fe1, fe2) in its zero-relative degrees."""
    P, SV = _driver()
    return tuple(SV.FINGER_JOINTS[P.FINGER_ID[finger]][servo][2])


def safe_range(finger: str, servo: str, pot_safe_deg: float = POT_SAFE_DEG) -> tuple[float, float]:
    """The calibrated range cut to servo degrees within pot_safe_deg of the middle of the potentiometer's range."""
    P, SV = _driver()
    _sid, zero, (lo, hi) = SV.FINGER_JOINTS[P.FINGER_ID[finger]][servo]
    return max(lo, -pot_safe_deg - zero), min(hi, pot_safe_deg - zero)


def safe_range_sim(finger: str, sim_joint: str, pot_safe_deg: float = POT_SAFE_DEG) -> tuple[float, float]:
    """safe_range in sim degrees (yaw, mcp, pip): over JOINT_SIGN, whose -1 swaps the ends (manual.sim_joint_limits)."""
    P, _ = _driver()
    lo, hi = safe_range(finger, P.SIM_JOINT_TO_SERVO[sim_joint], pot_safe_deg)
    s = P.JOINT_SIGN[(finger, sim_joint)]
    a, b = lo / s, hi / s
    return (a, b) if a <= b else (b, a)


def clip_servo(targets: dict, pot_safe_deg: float = POT_SAFE_DEG) -> list[str]:
    """Servo targets {finger: {aa, fe1, fe2}} (zero-relative degrees) clipped to safe_range in place; the clipped
    joints as "finger.servo"."""
    out = []
    for f, js in targets.items():
        for sv, v in js.items():
            lo, hi = safe_range(f, sv, pot_safe_deg)
            if not lo <= float(v) <= hi:
                js[sv] = min(max(float(v), lo), hi)
                out.append(f"{f}.{sv}")
    return out


def clip_sim(frame: dict, pot_safe_deg: float = POT_SAFE_DEG) -> list[str]:
    """A stream frame {finger: {yaw, mcp, pip}} (sim degrees) clipped to safe_range_sim in place; the clipped joints as
    "finger.joint"."""
    out = []
    for f, js in frame.items():
        for j, v in js.items():
            lo, hi = safe_range_sim(f, j, pot_safe_deg)
            if not lo <= float(v) <= hi:
                js[j] = min(max(float(v), lo), hi)
                out.append(f"{f}.{j}")
    return out


if __name__ == "__main__":
    for f in FINGERS:
        for j in SIM_JOINTS:
            P, _ = _driver()
            sv = P.SIM_JOINT_TO_SERVO[j]
            print(f"{f:6s} {j:3s} ({sv}): calibrated {calibrated_range(f, sv)}, safe (servo) "
                  f"{tuple(round(x, 2) for x in safe_range(f, sv))}, safe (sim) "
                  f"{tuple(round(x, 2) for x in safe_range_sim(f, j))}")
