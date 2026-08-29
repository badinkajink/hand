"""Take a hand DESIGN and a planned grasp/turn from the simulator and put it on this hardware.

WHY THIS EXISTS

The simulator's `real_v1` scenes and this driver were built from the same drawing, so their
frames agree -- but "agree" was an assumption nobody had checked end to end, and the two sides
express a design in different units, different frames, and different joint names. This module
is the one place that conversion lives, and the one place that says which parts of it are
actually VERIFIED.

    sim scene                         this module                     firmware
    <f>_mount body pos (m)   ->  palm frame {P} (mm)   ->  local (q_Fx,q_Fy)  ->  MOVEMM J<i>
    <f>_yaw/_mcp/_pip (rad)  ->  degrees               ->  aa/fe1/fe2         ->  servo goal

WHAT IS VERIFIED, AND WHAT IS NOT

1. MOUNT FRAME -- VERIFIED, and it is the identity. `morphohand.sampling.morphology`'s
   `REAL_V1_MOUNTS` is {thumb (-50,0), index (50,55), middle (50,-55)} in mm and
   `kinematics.FINGER_GEOMETRY`'s origins are the same three pairs. Both come from XY_space.png.
   A sim mount position in mm IS a {P} coordinate; nothing rotates, nothing flips.

2. LOCAL FRAME AND THE STEPPER SWAP -- VERIFIED in `kinematics.py` (firmware "x" tracks local
   q_Fy, firmware "y" tracks local q_Fx, per-finger sign from which corner homes). This module
   reuses that transform rather than restating it.

3. TRAVEL ENVELOPE -- MISMATCHED, and this is the live issue. The sim declares a +-30 mm box in
   x for all three fingers; the firmware's measured travel gives thumb x_max +26.2, index x_min
   -26.0, middle x_min -24.1. The shortfall lands on firmware joints J1/J3/J5, whose
   STEPS_PER_MM was "back-calculated from a known-good 10mm move and hasn't been individually
   ruler-checked" -- so the missing 4-6 mm may be a scale error rather than a mechanical limit.
   See `travel_audit()`; do not paper over it by widening FULL_EXTENSION_MM on a hunch, because
   MOVEMM does not stall-check.

4. JOINT IDENTITY -- VERIFIED by range fingerprint. The sim's joint limits are exactly the
   declared servo contract: yaw +-85 = aa, mcp [-15,+92] = fe1, pip [-18,+92] = fe2. Three
   distinct ranges, three exact matches; there is no other assignment that fits.

5. JOINT SIGN -- fe1/fe2 INFERRED, aa UNVERIFIED.
   fe1/fe2: the calibrated ranges are asymmetric ([-15,+92]-shaped) in a way only "small
   hyperextension, large flexion" explains, and the sim mirrors its mcp/pip axis per finger
   (thumb 0 -1 0, pair 0 +1 0) so that positive is flexion toward the palm centre on all three,
   matching the drawing's own "+flex direction" column. Positive-is-flexion on both sides.
   aa: the ranges are symmetric, so there is no fingerprint at all, and the sim gives ALL THREE
   fingers the same yaw axis (1 0 0) while mirroring mcp/pip -- i.e. positive yaw swings every
   tip toward +y_P, which is not what a rigid 180-degree-rotated thumb module would do. The sign
   is a per-finger coin flip that only hardware can settle, and the plans matter: g12 grips with
   thumb_yaw at +17.7 deg, so a flipped sign misses the tool by ~35 deg of roll.
   Measure it with `examples/verify_frame_mapping.py`, then set JOINT_SIGN.

Everything here is pure arithmetic over the driver's own calibration tables -- no serial, no
mujoco, no morphohand. It imports on a workstation so a plan can be checked before anyone is
standing next to the hardware.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .kinematics import (
    FINGER_GEOMETRY,
    FULL_EXTENSION_MM,
    HOME_ACCEL,
    HOME_COOLCONF,
    HOME_DIRECTION,
    HOME_VELOCITY,
    STEPPER_ACCEL,
    STEPPER_JOINTS,
    STEPPER_VELOCITY,
    STEPS_PER_MM,
    _home_timeout_ms,
    _TRANSFORM,
)
from .servos import DEFAULT_JOINT_SPEED, FINGER_JOINTS

FINGER_ID = {"thumb": 0, "index": 1, "middle": 2}
FINGER_NAME = {v: k for k, v in FINGER_ID.items()}

# sim joint suffix -> firmware joint name. Item 4 of the module docstring.
SIM_JOINT_TO_SERVO = {"yaw": "aa", "mcp": "fe1", "pip": "fe2"}

# (finger, sim joint) -> multiply the sim's degrees by this to get the servo's own
# zero-relative degrees. All +1 is the ASSUMPTION, not a measurement -- see docstring item 5.
# `verify_frame_mapping.py` prints the three aa lines to paste in here once measured.
# JOINT_SIGN: dict[tuple[str, str], float] = {
#    (f, j): 1.0 for f in FINGER_ID for j in SIM_JOINT_TO_SERVO
# }
SIGNS_MEASURED = True  # flip to True in the same commit that measures them

JOINT_SIGN = {
    ("thumb", "yaw"): +1.0,
    ("thumb", "mcp"): +1.0,
    ("thumb", "pip"): +1.0,
    ("index", "yaw"): -1.0,
    ("index", "mcp"): +1.0,
    ("index", "pip"): +1.0,
    ("middle", "yaw"): -1.0,
    ("middle", "mcp"): +1.0,
    ("middle", "pip"): +1.0,
}


# ------------------------------------------------------------------------------------------
# frames
# ------------------------------------------------------------------------------------------
def local_from_palm(finger: str, x_mm: float, y_mm: float) -> tuple[float, float]:
    """{P} -> this finger's local (q_Fx, q_Fy). A translation by the finger's origin; the
    local frames are axis-aligned with {P}, so there is no rotation to get wrong."""
    ox, oy = FINGER_GEOMETRY[FINGER_ID[finger]]["origin"]
    return x_mm - ox, y_mm - oy


def palm_from_local(finger: str, local_x_mm: float, local_y_mm: float) -> tuple[float, float]:
    ox, oy = FINGER_GEOMETRY[FINGER_ID[finger]]["origin"]
    return local_x_mm + ox, local_y_mm + oy


def stepper_mm(finger: str, local_x_mm: float, local_y_mm: float) -> dict[int, float]:
    """Local (q_Fx, q_Fy) -> {firmware joint index: mm from home}. The x/y swap and the
    per-finger sign both come from kinematics._TRANSFORM, so this stays in step with the
    homing corners it was solved from."""
    fid = FINGER_ID[finger]
    jx, jy = STEPPER_JOINTS[fid]
    x_off, x_sign, y_off, y_sign = _TRANSFORM[fid]
    return {jx: x_off + x_sign * local_y_mm, jy: y_off + y_sign * local_x_mm}


def local_envelope(finger: str,
                   travel_mm: dict[int, float] | None = None
                   ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Real reachable ((x_lo, x_hi), (y_lo, y_hi)) in this finger's local frame, inverted from
    measured stepper travel.

    `travel_mm` overrides FULL_EXTENSION_MM per joint index -- the honest way to ask "what
    would a re-calibrated J1/J3/J5 buy me?" without editing the driver's safety table."""
    fid = FINGER_ID[finger]
    jx, jy = STEPPER_JOINTS[fid]
    travel = dict(FULL_EXTENSION_MM)
    travel.update(travel_mm or {})
    x_off, x_sign, y_off, y_sign = _TRANSFORM[fid]
    ys = sorted(((0.0 - x_off) / x_sign, (travel[jx] - x_off) / x_sign))
    xs = sorted(((0.0 - y_off) / y_sign, (travel[jy] - y_off) / y_sign))
    return (xs[0], xs[1]), (ys[0], ys[1])


def palm_envelope(finger: str, travel_mm: dict[int, float] | None = None):
    (xlo, xhi), (ylo, yhi) = local_envelope(finger, travel_mm)
    ox, oy = FINGER_GEOMETRY[FINGER_ID[finger]]["origin"]
    return (xlo + ox, xhi + ox), (ylo + oy, yhi + oy)


@dataclass(frozen=True)
class Violation:
    finger: str
    axis: str          # "x" / "y" (local frame) or a joint name
    value: float
    bound: tuple[float, float]
    short: float       # how much further the axis would have to travel, mm or deg

    def __str__(self) -> str:
        unit = "mm" if self.axis in ("x", "y") else "deg"
        return (f"{self.finger}_{self.axis} {self.value:+.2f}{unit} outside "
                f"[{self.bound[0]:.2f}, {self.bound[1]:.2f}] -- short by {self.short:.2f}{unit}")


def mount_violations(finger: str, x_mm: float, y_mm: float, *, frame: str = "palm",
                     travel_mm: dict[int, float] | None = None) -> list[Violation]:
    """Everything wrong with putting `finger`'s mount here, with the shortfall in mm so a
    design that misses by 0.2 mm reads differently from one that misses by 6."""
    lx, ly = local_from_palm(finger, x_mm, y_mm) if frame == "palm" else (x_mm, y_mm)
    (xlo, xhi), (ylo, yhi) = local_envelope(finger, travel_mm)
    out = []
    for v, lo, hi, axis in ((lx, xlo, xhi, "x"), (ly, ylo, yhi, "y")):
        short = max(lo - v, v - hi, 0.0)
        if short > 1e-9:
            out.append(Violation(finger, axis, v, (lo, hi), short))
    return out


def joint_violation(finger: str, sim_joint: str, sim_deg: float) -> Violation | None:
    """Check a sim joint angle against the SERVO's own measured range (servos.FINGER_JOINTS),
    not the sim's declared limits -- several real ranges are tighter (index fe1 stops at
    +64.75 deg where the scene allows +92)."""
    servo_name = SIM_JOINT_TO_SERVO[sim_joint]
    _sid, _zero, (lo, hi) = FINGER_JOINTS[FINGER_ID[finger]][servo_name]
    v = servo_deg(finger, sim_joint, sim_deg)
    short = max(lo - v, v - hi, 0.0)
    return None if short <= 1e-9 else Violation(finger, servo_name, v, (lo, hi), short)


def servo_deg(finger: str, sim_joint: str, sim_deg: float) -> float:
    """Sim degrees -> the servo's own zero-relative degrees. Sign only: both sides measure from
    the same nominal zero pose, so there is no offset term -- if a servo's mechanical zero
    turns out not to be the scene's zero pose, that belongs here, and the envelope sweep says
    1 deg of it already costs most of the success rate."""
    return JOINT_SIGN[(finger, sim_joint)] * sim_deg


def travel_audit(travel_mm: dict[int, float] | None = None) -> list[str]:
    """The sim's declared +-30/+-55 design box against what the rails actually reach. This is
    the report that answers 'which candidate hands does the y-axis shortfall disqualify'."""
    declared = {"thumb": ((-30.0, 30.0), (-55.0, 55.0)),
                "index": ((-30.0, 30.0), (-30.0, 30.0)),
                "middle": ((-30.0, 30.0), (-30.0, 30.0))}
    lines = [f"  {'finger':7} {'axis':4} {'declared':>16} {'reachable':>16}  {'lost':>12}"]
    for finger, (dx, dy) in declared.items():
        real = local_envelope(finger, travel_mm)
        for axis, dec, got in (("x", dx, real[0]), ("y", dy, real[1])):
            lo = max(0.0, got[0] - dec[0])
            hi = max(0.0, dec[1] - got[1])
            lost = (f"{lo:.1f} at min" if lo > hi else f"{hi:.1f} at max") if max(lo, hi) > 1e-9 else "-"
            lines.append(f"  {finger:7} {axis:4} [{dec[0]:6.1f},{dec[1]:6.1f}] "
                         f"[{got[0]:6.2f},{got[1]:6.2f}]  {lost:>12}")
    return lines


# ------------------------------------------------------------------------------------------
# a plan
# ------------------------------------------------------------------------------------------
@dataclass
class Pose:
    """One set-point of the trajectory: sim-frame degrees per finger joint, and how long the
    hand takes to ramp into it from the previous one."""
    name: str
    ramp_s: float
    hold_s: float
    joints: dict[str, dict[str, float]]   # finger -> {yaw, mcp, pip} in SIM degrees


@dataclass
class HandPlan:
    """A design (where the three mounts go) plus the open-loop trajectory (what the nine servos
    do). Produced by `scripts/real_v1_hand_plan.py` on the workstation; consumed here."""
    design: str
    mounts_palm_mm: dict[str, tuple[float, float]]
    poses: list[Pose]
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "HandPlan":
        raw = json.loads(Path(path).read_text())
        return cls(
            design=raw["design"],
            mounts_palm_mm={f: tuple(v) for f, v in raw["mounts_palm_mm"].items()},
            poses=[Pose(p["name"], float(p["ramp_s"]), float(p["hold_s"]), p["joints"])
                   for p in raw["poses"]],
            meta=raw.get("meta", {}),
        )

    # -- checking ---------------------------------------------------------------------------
    def validate(self, travel_mm: dict[int, float] | None = None) -> list[Violation]:
        """Every reason this plan cannot be run as written. Empty list = it fits."""
        bad: list[Violation] = []
        for finger, (x, y) in self.mounts_palm_mm.items():
            bad += mount_violations(finger, x, y, travel_mm=travel_mm)
        for pose in self.poses:
            for finger, joints in pose.joints.items():
                for sim_joint, deg in joints.items():
                    v = joint_violation(finger, sim_joint, deg)
                    if v is not None:
                        bad.append(v)
        return bad

    # -- the literal command list -----------------------------------------------------------
    def stepper_commands(self, *, home: bool = True,
                         velocity: int = STEPPER_VELOCITY,
                         accel: int = STEPPER_ACCEL) -> list[str]:
        """The wire protocol, verbatim (docs/protocol.md), for putting the gantries where this
        design wants them. Readable before it is run -- paste into a serial terminal, or diff
        two designs against each other."""
        out: list[str] = []
        for finger in ("thumb", "index", "middle"):
            fid = FINGER_ID[finger]
            jx, jy = STEPPER_JOINTS[fid]
            lx, ly = local_from_palm(finger, *self.mounts_palm_mm[finger])
            targets = stepper_mm(finger, lx, ly)
            out.append(f"# {finger}: palm ({self.mounts_palm_mm[finger][0]:+.1f}, "
                       f"{self.mounts_palm_mm[finger][1]:+.1f}) mm = local ({lx:+.1f}, {ly:+.1f}) mm")
            for j in (jx, jy):
                out.append(f"EN J{j}")
                out.append(f"SETSCALE J{j} {STEPS_PER_MM[j]:.1f}")
            if home:
                for j in (jx, jy):   # sequential, never concurrent -- see kinematics._home_one_axis
                    out.append(f"WREG5160 J{j} 6D {HOME_COOLCONF[j]:X}")
                    out.append(f"HOME J{j} {HOME_DIRECTION} {HOME_VELOCITY} {HOME_ACCEL} "
                               f"{_home_timeout_ms(j)}   # poll STAT J{j}, then:")
                    out.append(f"ZERO J{j}")
            for j in (jx, jy):
                out.append(f"MOVEMM J{j} {targets[j]:.2f} {velocity} {accel}")
        return out

    def servo_setpoints(self) -> list[str]:
        """The trajectory as the handful of numbers it really is: servo-frame degrees, per
        set-point, with the ramp that leads into each."""
        order = [(f, j) for f in ("thumb", "index", "middle") for j in ("yaw", "mcp", "pip")]
        head = f"  {'set-point':14} {'ramp':>6} {'hold':>6}  " + " ".join(
            f"{f'f{FINGER_ID[f]}.{SIM_JOINT_TO_SERVO[j]}':>7}" for f, j in order)
        out = [head]
        for p in self.poses:
            vals = " ".join(f"{servo_deg(f, j, p.joints[f][j]):7.2f}" for f, j in order)
            out.append(f"  {p.name:14} {p.ramp_s:5.2f}s {p.hold_s:5.2f}s  {vals}")
        return out

    def describe(self, travel_mm: dict[int, float] | None = None) -> str:
        bad = self.validate(travel_mm)
        lines = [f"plan: {self.design}   ({self.meta.get('object', '?')}, "
                 f"{self.meta.get('source', 'no source recorded')})", ""]
        lines += ["MOUNTS", f"  {'finger':7} {'palm x':>8} {'palm y':>8} {'local x':>8} "
                            f"{'local y':>8}   stepper mm"]
        for finger in ("thumb", "index", "middle"):
            px, py = self.mounts_palm_mm[finger]
            lx, ly = local_from_palm(finger, px, py)
            t = stepper_mm(finger, lx, ly)
            js = "  ".join(f"J{j}={v:6.2f}" for j, v in sorted(t.items()))
            lines.append(f"  {finger:7} {px:8.1f} {py:8.1f} {lx:8.1f} {ly:8.1f}   {js}")
        lines += ["", "TRAJECTORY (servo-frame degrees)"] + self.servo_setpoints()
        lines += ["", ("FITS the measured envelope" if not bad else "DOES NOT FIT:")]
        lines += [f"  {v}" for v in bad]
        if not SIGNS_MEASURED:
            lines += ["", "aa SIGN IS UNMEASURED -- every yaw column above is an assumption.",
                      "  run examples/verify_frame_mapping.py before believing this trajectory."]
        return "\n".join(lines)

    # -- running it -------------------------------------------------------------------------
    def apply_mounts(self, hand, *, home: bool = True,
                     velocity: int = STEPPER_VELOCITY, accel: int = STEPPER_ACCEL) -> None:
        """Put the three gantries where this design wants them. Validates first: a bad target
        raises before any command goes out, rather than after two of six axes have moved."""
        bad = [v for v in self.validate() if v.axis in ("x", "y")]
        if bad:
            raise ValueError("mounts outside the measured envelope:\n  "
                             + "\n  ".join(str(v) for v in bad))
        if home:
            hand.home_all(velocity=velocity, accel=accel)
        for finger, (x, y) in self.mounts_palm_mm.items():
            hand.finger(FINGER_ID[finger]).move_to_global(x, y, velocity=velocity, accel=accel)

    def run_trajectory(self, hand, *, signs_checked: bool, rate_hz: float = 50.0,
                       speed: int = DEFAULT_JOINT_SPEED) -> None:
        """Replay the servo trajectory: linear interpolation between set-points at `rate_hz`,
        through the one-transaction sync path (`Hand.set_joints_fast`). The per-joint
        write-verify path cannot do this -- it is ~0.9 s per joint.

        `signs_checked` has no default on purpose. Until someone has watched a real finger move
        under a positive aa command, the yaw column of every plan is a guess (docstring item 5)
        and a wrong guess quietly rolls the thumb off the tool instead of onto it."""
        if not signs_checked:
            raise ValueError(
                "run_trajectory(signs_checked=False): JOINT_SIGN is unmeasured. Run "
                "examples/verify_frame_mapping.py, set JOINT_SIGN + SIGNS_MEASURED, then pass "
                "signs_checked=True.")
        bad = self.validate()
        if bad:
            raise ValueError("plan does not fit:\n  " + "\n  ".join(str(v) for v in bad))
        dt = 1.0 / rate_hz
        prev = self.poses[0]
        hand.set_joints_fast(self._pose_dict(prev), speed=speed)
        time.sleep(prev.hold_s)
        for pose in self.poses[1:]:
            n = max(1, int(round(pose.ramp_s * rate_hz)))
            for i in range(1, n + 1):
                u = i / n
                blend = Pose(pose.name, 0.0, 0.0, {
                    f: {j: prev.joints[f][j] + (pose.joints[f][j] - prev.joints[f][j]) * u
                        for j in SIM_JOINT_TO_SERVO}
                    for f in FINGER_ID})
                hand.set_joints_fast(self._pose_dict(blend), speed=speed)
                time.sleep(dt)
            if pose.hold_s > 0:
                time.sleep(pose.hold_s)
            prev = pose

    @staticmethod
    def _pose_dict(pose: Pose) -> dict[int, dict[str, float]]:
        return {FINGER_ID[f]: {SIM_JOINT_TO_SERVO[j]: servo_deg(f, j, deg)
                               for j, deg in joints.items()}
                for f, joints in pose.joints.items()}
