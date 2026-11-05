"""Manual joint/mount commands, in `examples/hand_control.py`'s grammar.

That script is the thing you reach for when a plan is not what you want -- posing one
finger, checking a sign, walking a gantry out of the way.  It needs its own two USB
connections, so it cannot run while the control station owns the ports.  This module is
the same grammar over the station's existing link, so the browser can do it without
anyone unplugging anything.

    <finger>_x <mm>     gantry, PALM-FRAME mm
    <finger>_y <mm>     gantry, PALM-FRAME mm
    <finger>_yaw <deg>  servo, SIM-frame degrees (what the plans and the UI speak)
    <finger>_mcp <deg>
    <finger>_pip <deg>
    <finger>_aa <deg>   servo, the SERVO's own zero-relative degrees (hand_control's
    <finger>_fe1 <deg>  convention).  aa/fe1/fe2 are yaw/mcp/pip with the per-finger
    <finger>_fe2 <deg>  sign already applied, so `thumb_aa 5` and `thumb_yaw 5` are the
                        same command but `index_aa 5` and `index_yaw 5` are opposite.

`<finger>` is 0/1/2 or thumb/index/middle, interchangeably.  Segments are comma
separated and the whole line is parsed -- and can fail -- before anything moves, so a
typo in the third segment does not leave the first two applied.

ON THE X/Y FRAME.  `hand_control.py` takes x/y in FIRMWARE mm (raw stepper travel,
0 = wherever that axis homed).  This module takes PALM-FRAME mm instead, because that
is what every other number in the control station is: the plan's mounts, the build
sheets, the design search, the telemetry read-out.  Mixing the two is the mistake
hand_control's own docstring warns about, so a parsed mount command carries the
firmware mm it resolves to and the caller shows both.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .plan import (FINGER_ID, JOINT_SIGN, SIM_JOINT_TO_SERVO, local_from_palm,
                   palm_envelope, stepper_mm)
from .servos import FINGER_JOINTS

FINGER_ORDER = ("thumb", "index", "middle")
JOINT_ORDER = ("yaw", "mcp", "pip")
SERVO_JOINT_TO_SIM = {v: k for k, v in SIM_JOINT_TO_SERVO.items()}

_TOKEN = r"(\d+|thumb|index|middle)"
_AXIS_RE = re.compile(rf"^{_TOKEN}_(x|y|yaw|mcp|pip|aa|fe1|fe2)\s+(-?\d+(?:\.\d+)?)$",
                      re.IGNORECASE)


class ManualCommandError(ValueError):
    """A line that does not parse, or that asks for a position the hand cannot reach."""


@dataclass
class ManualRequest:
    """One parsed line: at most one mount target per finger, at most one value per joint."""
    mounts: dict[str, dict[str, float]] = field(default_factory=dict)
    joints: dict[str, dict[str, float]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.mounts or self.joints)


def resolve_finger(token: str) -> str:
    token = str(token).strip().lower()
    if token.isdigit():
        fid = int(token)
        for name, i in FINGER_ID.items():
            if i == fid:
                return name
        raise ManualCommandError(f"no finger {fid} (have 0, 1, 2)")
    if token not in FINGER_ID:
        raise ManualCommandError(f"no finger {token!r} (have {', '.join(FINGER_ORDER)})")
    return token


def sim_joint_limits(finger: str, sim_joint: str) -> tuple[float, float]:
    """The servo's calibrated range expressed in SIM degrees.

    The bound lives on the servo, and JOINT_SIGN flips three of the nine joints, so a
    negative sign swaps which end of the servo's range is the sim minimum.
    """
    servo_name = SIM_JOINT_TO_SERVO[sim_joint]
    _sid, _zero, (lo, hi) = FINGER_JOINTS[FINGER_ID[finger]][servo_name]
    sign = JOINT_SIGN[(finger, sim_joint)]
    a, b = lo / sign, hi / sign
    return (a, b) if a <= b else (b, a)


def mount_limits(finger: str) -> dict[str, tuple[float, float]]:
    """Reachable palm-frame (x, y) for one finger, inverted from measured stepper travel."""
    (xlo, xhi), (ylo, yhi) = palm_envelope(finger)
    return {"x": (xlo, xhi), "y": (ylo, yhi)}


def limits() -> dict:
    """Everything a UI needs to build bounded controls, in one payload."""
    return {
        "fingers": list(FINGER_ORDER),
        "joints": list(JOINT_ORDER),
        "mounts": {f: {k: list(v) for k, v in mount_limits(f).items()} for f in FINGER_ORDER},
        "joint_deg": {f: {j: list(sim_joint_limits(f, j)) for j in JOINT_ORDER}
                      for f in FINGER_ORDER},
        "servo_alias": SIM_JOINT_TO_SERVO,
        "joint_sign": {f: {j: JOINT_SIGN[(f, j)] for j in JOINT_ORDER} for f in FINGER_ORDER},
    }


def check_mount(finger: str, x_mm: float, y_mm: float) -> dict[int, float]:
    """Bounds-check a palm-frame mount target and return the firmware mm it resolves to.

    Nothing is commanded here -- this is the same split `GantryFinger.stepper_targets`
    makes, so a caller can validate a whole line before moving one axis.
    """
    for name, value in (("x", x_mm), ("y", y_mm)):
        if not math.isfinite(value):
            raise ManualCommandError(f"{finger}_{name} must be a finite number")
        lo, hi = mount_limits(finger)[name]
        if not lo <= value <= hi:
            raise ManualCommandError(
                f"{finger}_{name}={value:+.2f}mm is outside the reachable palm-frame range "
                f"[{lo:.2f}, {hi:.2f}]mm")
    return stepper_mm(finger, *local_from_palm(finger, x_mm, y_mm))


def check_joint(finger: str, sim_joint: str, deg: float) -> float:
    """Bounds-check one sim-frame joint value; returns the servo-frame degrees."""
    if not math.isfinite(deg):
        raise ManualCommandError(f"{finger}_{sim_joint} must be a finite number")
    lo, hi = sim_joint_limits(finger, sim_joint)
    if not lo <= deg <= hi:
        raise ManualCommandError(
            f"{finger}_{sim_joint}={deg:+.2f}deg is outside this joint's calibrated range "
            f"[{lo:.2f}, {hi:.2f}]deg")
    return JOINT_SIGN[(finger, sim_joint)] * deg


def parse(line: str) -> ManualRequest:
    """Parse a whole comma-separated line.  Raises before anything is applied."""
    req = ManualRequest()
    for segment in str(line).split(","):
        segment = segment.strip()
        if not segment:
            continue
        m = _AXIS_RE.match(segment)
        if not m:
            raise ManualCommandError(
                f"could not parse {segment!r} -- expected e.g. 'thumb_mcp 30', "
                f"'0_x -42.5', 'index_fe1 20'")
        finger = resolve_finger(m.group(1))
        axis = m.group(2).lower()
        value = float(m.group(3))
        if axis in ("x", "y"):
            req.mounts.setdefault(finger, {})[axis] = value
        else:
            sim_joint = SERVO_JOINT_TO_SIM.get(axis, axis)
            if axis in SERVO_JOINT_TO_SIM:            # aa/fe1/fe2 arrive servo-signed
                value = value / JOINT_SIGN[(finger, sim_joint)]
            req.joints.setdefault(finger, {})[sim_joint] = value
    return req


def validate(req: ManualRequest, *, current_mounts: dict[str, dict[str, float]] | None = None
             ) -> dict:
    """Bounds-check a parsed request.  A mount command that names only one axis needs the
    other one to check against, which is why `current_mounts` is required for a partial
    move -- a half-specified target is not checkable, and guessing zero would send the
    gantry to the middle of the rail.

    Returns {"mounts": {finger: {"x","y","steppers"}}, "joints": {finger: {joint: deg}}}.
    """
    out: dict = {"mounts": {}, "joints": {}}
    for finger, axes in req.mounts.items():
        have = dict((current_mounts or {}).get(finger) or {})
        target = {"x": axes.get("x", have.get("x")), "y": axes.get("y", have.get("y"))}
        missing = [k for k, v in target.items() if v is None]
        if missing:
            given = ", ".join(f"{finger}_{k}" for k in sorted(axes))
            raise ManualCommandError(
                f"{given} alone is not enough: {finger}_{missing[0]} is unknown, so the target "
                f"cannot be bounds-checked. Give both axes, or apply a morphology first so the "
                f"current position is known.")
        steppers = check_mount(finger, float(target["x"]), float(target["y"]))
        out["mounts"][finger] = {"x": float(target["x"]), "y": float(target["y"]),
                                 "steppers": {str(k): round(v, 3) for k, v in steppers.items()}}
    for finger, joints in req.joints.items():
        for sim_joint, deg in joints.items():
            if sim_joint not in JOINT_ORDER:
                raise ManualCommandError(f"no joint {sim_joint!r} (have {', '.join(JOINT_ORDER)})")
            check_joint(finger, sim_joint, deg)
        out["joints"][finger] = {j: float(v) for j, v in joints.items()}
    return out
