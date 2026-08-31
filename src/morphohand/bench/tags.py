"""Where the shaft is, in millimetres the simulator can be held to.

Until 2026-08-31 this program had no object-pose sensor. Every bench run was scored
by an operator looking at the shaft and typing a number, `scripts/probe_obs_ablation.py`
called the object-derived observation blocks "hidden", and the whole clip/band
literature was measured in simulation and *predicted* onto hardware. Two AprilTags and
a set of tape-measure readings close that gap.

  id 6, 40 mm   STATIC, mounted vertical. Supplies world up, and -- because its own
                position in the hand's frame was measured -- an absolute datum.
  id 0, 30 mm   on the CYLINDER, its plane containing the shaft axis, on a vane
                sticking out past one end face.

WHAT THIS MODULE IS FOR. `pupil_apriltags` gives each tag's pose in the CAMERA's frame,
which is a frame that moves whenever anyone nudges the tripod. This turns that into two
frames that mean something:

  the reference-tag frame   Height above the bench floor and horizontal distance from
                            the datum, independent of how the camera is aimed.
  the bench frame {B}       origin on the floor under the palm centre, +x toward the
                            index/middle gantries, +z up. The reference tag is rigidly
                            bolted normal to the x_B axis and the camera directly faces it.
                            Therefore the tag's in-plane horizontal axis is +-y_B, not
                            an arbitrary heading. The remaining sign is selected from
                            the observed hand-side offset (the tag is at x=+133.5 and
                            the shaft lies on its smaller-x side). No staged calibration
                            is needed while that mounting is unchanged.

THE MEASUREMENTS, all by the user on 2026-08-31, all in mm.

  reference tag centre    175.0 above the floor
                          133.5 from the palm centre along x (past the index gantry)
                           15.0 from the palm centre along y (toward the thumb side)
  cylinder tag centre     21.0 past the cylinder's flat end face, i.e. 71.0 from the
                          cylinder's centre along the shaft axis, and ON that axis
                          (no lateral offset).
  fingertip datum         with the distal link vertical, its top flat edge is 65.0
                          above the floor. The link is 37.2 long.

WHY THE FINGERTIP DATUM IS THE IMPORTANT ONE. It is the only measurement that ties the
bench floor to the simulator's floor, and it does so through a part both sides model.
In `real_hand_morphology_actuated.xml` at the zero pose, with the palm at 62.5 mm, every
`<finger>_pip_frame` -- the top of the distal link -- sits at z = 21.0 mm, and the link
runs 37.2 mm down from there to 16.2 mm BELOW the floor (which is exactly the "straight
fingers put every pad 16 mm below the floor" note in build_real_v1_scenes.py, from the
other direction). The bench puts that same edge at 65.0. So

    z_bench = z_sim + 44.0 mm

and the physical palm plane stands 106.5 mm over its table where the sim's stands at
62.5 over its own. Anything this module reports as a bench height can be compared to a
simulated one by that single subtraction, and `SIM_TO_BENCH_Z_MM` is the only place it
is written down.

CONVENTIONS. Distances in and out of this module are MILLIMETRES; `pupil_apriltags`
works in the units of the `tag_size` it was given (metres, in this program), and the
conversion happens once, at the boundary, in `reading_from_tags`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---- the printed tags -----------------------------------------------------------------
REF_TAG_ID = 6
CYL_TAG_ID = 0
REF_TAG_SIZE_M = 0.040
CYL_TAG_SIZE_M = 0.030

# ---- the measured geometry ------------------------------------------------------------
#: Reference tag centre in the bench frame {B}: origin on the floor under the palm
#: centre, +x toward the index/middle gantries, +z up. The y SIGN is the one component
#: of this that is not pinned by the measurement -- the user recorded "15 mm toward the
#: thumb side", and the thumb mount sits at y = 0 in {P}, so "thumb side" does not name a
#: y direction on its own. The magnitude is used (it only ever enters as an offset of the
#: datum); if a bench session ever needs the sign, measure which way the tag sits from the
#: index/middle split rather than inferring it from this comment.
REF_TAG_BENCH_MM = (133.5, 15.0, 175.0)
REF_TAG_Y_SIGN_MEASURED = False

#: Rigid mounting facts, recorded explicitly rather than living only in an operator's
#: recollection. The printed reference-tag plane is normal to bench x and faces the camera;
#: the camera faces it along the opposite axis. ``plane_axis='y'`` is consequently one of
#: the two signed bench-y directions. ``heading_from_mounting`` resolves that printed-axis
#: sign from the hand already visible in the frame.
REF_TAG_NORMAL_BENCH_AXIS = "x (signed direction resolved from the facing/scene)"
CAMERA_VIEW_BENCH_AXIS = "x (opposes the reference-tag face)"
REF_TAG_PLANE_HORIZONTAL_BENCH_CANDIDATES = ("+y", "-y")
REF_TAG_MOUNTING_FIXED = True

#: The largest |bench x| at which the tool can plausibly sit while the hand is holding it.
#: The mounts span +-30 mm and the fingers reach roughly 60 mm beyond them. This is deliberately
#: generous because it is used to REJECT an ambiguous mounting heading, never to validate a
#: measurement: a shaft outside it means the scene is not staged, not that the reading is wrong.
OBJECT_X_ENVELOPE_MM = 90.0

#: Cylinder tag centre from the cylinder's centre, along the shaft axis, on the axis.
#: 21 mm past a flat end face + the 50 mm half-length.
CYL_TAG_AXIAL_MM = 71.0
CYL_HALF_LEN_MM = 50.0
CYL_RADIUS_MM = 12.5

#: The floor tie-in. See the module docstring: measured 65.0, simulated 21.0.
FINGERTIP_TOP_BENCH_MM = 65.0
FINGERTIP_LINK_MM = 37.2
SIM_PIP_FRAME_Z_MM = 21.0
SIM_TO_BENCH_Z_MM = FINGERTIP_TOP_BENCH_MM - SIM_PIP_FRAME_Z_MM   # 44.0

#: Where the bench scenes stand the tool before the hand closes (`--bench-height 0.100`
#: sets the cylinder body's z, so this is the CENTRE, not the underside).
BENCH_POST_HEIGHT_SIM_MM = 100.0

AXES = {"x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
        "z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0)}


def axis(flag: str) -> np.ndarray:
    try:
        return np.array(AXES[flag], dtype=float)
    except KeyError:
        raise ValueError(f"axis must be one of {sorted(AXES)}, not {flag!r}") from None


def sim_z_mm(bench_z_mm: float) -> float:
    """A bench height in the simulator's own z. The inverse is `bench_z_mm`."""
    return bench_z_mm - SIM_TO_BENCH_Z_MM


def bench_z_mm(sim_z_mm_: float) -> float:
    return sim_z_mm_ + SIM_TO_BENCH_Z_MM


def deg_from_up(cos_up: float) -> float:
    """Angle from UP unfolded to [0, 180]: 0 = straight up, 90 = horizontal, 180 = down.

    Folding with abs() would score a turn to the WRONG pole identically to a correct one,
    which is the mistake `docs/rl/` records costing a week of policy analysis.
    """
    return math.degrees(math.acos(min(1.0, max(-1.0, float(cos_up)))))


@dataclass(frozen=True)
class Reading:
    """One frame's worth of object state, in the bench frame where it is knowable."""
    t: float                                   # seconds since the trace started
    cos_up: float                              # cos(shaft axis, world up), SIGNED
    deg_from_up: float
    z_bench_mm: float                          # cylinder CENTRE above the bench floor
    radial_mm: float                           # horizontal distance from the reference tag
    tag_z_bench_mm: float                      # the tag itself, for a sanity read
    xy_bench_mm: tuple[float, float] | None    # None until the heading is calibrated
    margin: float                              # detector decision margin on the cylinder tag
    range_mm: float                            # cylinder tag distance from the camera

    @property
    def z_sim_mm(self) -> float:
        return sim_z_mm(self.z_bench_mm)

    def row(self) -> dict:
        x, y = self.xy_bench_mm if self.xy_bench_mm else ("", "")
        return {"t": round(self.t, 4), "cos": round(self.cos_up, 5),
                "deg": round(self.deg_from_up, 3),
                "z_bench_mm": round(self.z_bench_mm, 2),
                "z_sim_mm": round(self.z_sim_mm, 2),
                "radial_mm": round(self.radial_mm, 2),
                "x_bench_mm": (round(x, 2) if x != "" else ""),
                "y_bench_mm": (round(y, 2) if y != "" else ""),
                "tag_z_bench_mm": round(self.tag_z_bench_mm, 2),
                "margin": round(self.margin, 1), "range_mm": round(self.range_mm, 1)}


CSV_FIELDS = ["t", "cos", "deg", "z_bench_mm", "z_sim_mm", "radial_mm",
              "x_bench_mm", "y_bench_mm", "tag_z_bench_mm", "margin", "range_mm"]


@dataclass
class BenchFrame:
    """The static reference tag, resolved into directions in the camera's frame.

    Latched ONCE at the start of a run rather than read every frame: the tag does not
    move, and depending on it per-frame makes a shadowed reference a single point of
    failure for the whole trace. `drift_deg` re-checks it for free whenever it IS seen,
    which is how a bumped tripod gets caught.
    """
    up_cam: np.ndarray                          # unit, world up in camera coordinates
    plane_cam: np.ndarray                       # unit, horizontal, in the tag's plane
    ref_t_cam_mm: np.ndarray                    # reference tag centre, camera frame, mm
    heading_deg: float | None = None            # bench +x measured from `plane_cam`, about up
    ref_bench_mm: tuple[float, float, float] = REF_TAG_BENCH_MM
    n_frames: int = 0
    seen_frames: int = 0

    @classmethod
    def latch(cls, poses, *, up_axis: str = "x", plane_axis: str = "y",
              heading_deg: float | None = None,
              ref_bench_mm: tuple[float, float, float] = REF_TAG_BENCH_MM) -> "BenchFrame":
        """`poses` is a sequence of (pose_R, pose_t_metres) for the reference tag.

        Averaging a handful of frames rather than trusting one: a single detection of a
        40 mm tag at ~500 mm carries a few tenths of a degree of pose noise, and the up
        vector is the datum every angle in the run is measured against.
        """
        poses = list(poses)
        if not poses:
            raise ValueError(f"reference tag id{REF_TAG_ID} was never seen -- light it "
                             "better or re-aim before running")
        u, p = axis(up_axis), axis(plane_axis)
        up = _unit(np.mean([R @ u for R, _ in poses], axis=0))
        plane = np.mean([R @ p for R, _ in poses], axis=0)
        plane = _unit(plane - up * float(np.dot(plane, up)))     # horizontal component only
        t = np.mean([np.asarray(t).flatten() for _, t in poses], axis=0) * 1000.0
        return cls(up_cam=up, plane_cam=plane, ref_t_cam_mm=t,
                   heading_deg=heading_deg, ref_bench_mm=tuple(ref_bench_mm),
                   n_frames=len(poses), seen_frames=len(poses))

    # ---- the two horizontal bench axes, once a heading exists --------------------------
    @property
    def x_cam(self) -> np.ndarray | None:
        if self.heading_deg is None:
            return None
        a = math.radians(self.heading_deg)
        return _unit(math.cos(a) * self.plane_cam
                     + math.sin(a) * np.cross(self.up_cam, self.plane_cam))

    @property
    def y_cam(self) -> np.ndarray | None:
        x = self.x_cam
        return None if x is None else _unit(np.cross(self.up_cam, x))

    def drift_deg(self, pose_R, up_axis: str = "x") -> float:
        """How far the reference has moved since it was latched. A bumped tripod."""
        v = pose_R @ axis(up_axis)
        return math.degrees(math.acos(min(1.0, max(-1.0, float(np.dot(_unit(v), self.up_cam))))))

    def locate(self, t_cam_mm: np.ndarray) -> tuple[float, float, tuple[float, float] | None]:
        """A camera-frame point -> (height above the bench floor, horizontal distance from
        the reference tag, bench (x, y) or None)."""
        d = np.asarray(t_cam_mm, dtype=float).flatten() - self.ref_t_cam_mm
        up = float(np.dot(d, self.up_cam))
        horiz = d - up * self.up_cam
        z = self.ref_bench_mm[2] + up
        radial = float(np.linalg.norm(horiz))
        xy = None
        if self.heading_deg is not None:
            xy = (self.ref_bench_mm[0] + float(np.dot(d, self.x_cam)),
                  self.ref_bench_mm[1] + float(np.dot(d, self.y_cam)))
        return z, radial, xy

    def heading_from_mounting(self, t_cam_mm: np.ndarray) -> tuple[float, str]:
        """The heading, from how the tag is MOUNTED plus one fact about the staged scene.

        The reference tag is bolted facing normal to the gantry x-axis and stays that way, so
        its in-plane horizontal axis is +-y_B and the heading can only be +90 or -90 -- there is
        no continuum to calibrate, only a sign. The two candidates differ by 180 deg, so they
        mirror the shaft about the tag's own x: one lands at 133.5 - d, the other at 133.5 + d.
        The sign is then decided by the hand, which sits at the palm centre, x = 0.

        That discriminator only works when the shaft is staged IN the hand, and the first live
        run showed why the test has to be able to fail. With the shaft parked on its post beside
        the reference tag the two candidates came out at +114 and +153 mm -- both far outside the
        hand, 19 mm either side of the tag -- and an earlier version of this method, which simply
        took the smaller x, returned +114 and called it "the hand's side". It then wrote bench
        x/y into the trace off a coin flip. So the rule is exactly one candidate inside the hand
        envelope, and staging beside the tag is refused rather than guessed.

        What this cannot check is its own premise: a tag that has been re-aimed by hand is no
        longer normal to the gantry x-axis and the true heading is not +-90 at all. That is what
        `--calibrate-heading` is for.
        """
        seen = []
        for cand in (90.0, -90.0):
            keep, self.heading_deg = self.heading_deg, cand
            _, _, xy = self.locate(t_cam_mm)
            self.heading_deg = keep
            seen.append((float(xy[0]), cand))
        inside = [(x, c) for x, c in seen if abs(x) <= OBJECT_X_ENVELOPE_MM]
        if len(inside) != 1:
            where = " and ".join(f"{x:+.0f}" for x, _ in sorted(seen))
            raise ValueError(
                f"the two mounting headings put the shaft at bench x {where} mm and "
                f"{'neither is' if not inside else 'both are'} inside the "
                f"+-{OBJECT_X_ENVELOPE_MM:.0f} mm hand envelope, so the sign is a guess. "
                "Stage the shaft in the hand before latching, or use --calibrate-heading")
        x, cand = inside[0]
        other = next(v for v, c in seen if c != cand)
        return cand, (f"puts the shaft at bench x {x:+.0f} mm, inside the "
                      f"+-{OBJECT_X_ENVELOPE_MM:.0f} mm hand envelope; the other candidate "
                      f"lands at {other:+.0f} mm")

    def heading_for(self, t_cam_mm: np.ndarray, bench_xy_mm: tuple[float, float]) -> float:
        """Solve the one unknown: the heading that puts a point of KNOWN bench (x, y) where
        the camera sees it. Calibration, run once, with the cylinder staged somewhere its
        position is actually known -- not something to guess."""
        d = np.asarray(t_cam_mm, dtype=float).flatten() - self.ref_t_cam_mm
        obs = d - float(np.dot(d, self.up_cam)) * self.up_cam
        if np.linalg.norm(obs) < 1.0:
            raise ValueError("calibration point is within 1 mm of the reference tag's "
                             "vertical axis; its heading is undefined there")
        want = np.array([bench_xy_mm[0] - self.ref_bench_mm[0],
                         bench_xy_mm[1] - self.ref_bench_mm[1]], dtype=float)
        if np.linalg.norm(want) < 1.0:
            raise ValueError("calibration point coincides with the reference tag in x/y")
        # angle of the observed offset from plane_cam, minus the angle the bench wants
        e1, e2 = self.plane_cam, np.cross(self.up_cam, self.plane_cam)
        obs_a = math.atan2(float(np.dot(obs, e2)), float(np.dot(obs, e1)))
        want_a = math.atan2(want[1], want[0])
        return math.degrees((obs_a - want_a + math.pi) % (2 * math.pi) - math.pi)

    def to_json(self) -> dict:
        return {"up_cam": [round(v, 6) for v in self.up_cam],
                "plane_cam": [round(v, 6) for v in self.plane_cam],
                "ref_t_cam_mm": [round(v, 2) for v in self.ref_t_cam_mm],
                "heading_deg": self.heading_deg,
                "ref_bench_mm": list(self.ref_bench_mm),
                "rigid_mounting": {
                    "fixed": REF_TAG_MOUNTING_FIXED,
                    "ref_tag_normal_bench_axis": REF_TAG_NORMAL_BENCH_AXIS,
                    "camera_view_bench_axis": CAMERA_VIEW_BENCH_AXIS,
                    "plane_horizontal_candidates":
                        list(REF_TAG_PLANE_HORIZONTAL_BENCH_CANDIDATES),
                },
                "latched_from_frames": self.n_frames}


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).flatten()
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("cannot normalise a zero-length direction")
    return v / n


def object_center_cam_mm(pose_R, pose_t, *, shaft_axis: str = "x",
                         axial_mm: float = CYL_TAG_AXIAL_MM) -> tuple[np.ndarray, np.ndarray]:
    """(cylinder centre, unit shaft direction) in the camera frame, both mm / unitless.

    `shaft_axis` is the direction in the TAG's frame that points from the cylinder centre
    OUTWARD to the tag. Get that sign backwards and the centre lands 142 mm the wrong way
    down the shaft, which is why `--probe` prints the tag height and the derived centre
    height side by side: with the shaft standing up and the tag above it, the centre must
    read LOWER than the tag.
    """
    s = _unit(np.asarray(pose_R, dtype=float) @ axis(shaft_axis))
    tag = np.asarray(pose_t, dtype=float).flatten() * 1000.0
    return tag - axial_mm * s, s


def reading_from_tags(frame: BenchFrame, pose_R, pose_t, *, t: float,
                      shaft_axis: str = "x", margin: float = 0.0,
                      axial_mm: float = CYL_TAG_AXIAL_MM) -> Reading:
    center, s = object_center_cam_mm(pose_R, pose_t, shaft_axis=shaft_axis, axial_mm=axial_mm)
    z, radial, xy = frame.locate(center)
    tag_z, _, _ = frame.locate(np.asarray(pose_t, dtype=float).flatten() * 1000.0)
    cos_up = float(np.dot(s, frame.up_cam))
    return Reading(t=t, cos_up=cos_up, deg_from_up=deg_from_up(cos_up), z_bench_mm=z,
                   radial_mm=radial, tag_z_bench_mm=tag_z, xy_bench_mm=xy, margin=float(margin),
                   range_mm=float(np.linalg.norm(np.asarray(pose_t).flatten()) * 1000.0))


# ---- what a whole trace means ----------------------------------------------------------
#: A drop is a fall, not a low sample: the shaft has to LOSE this much height against its
#: own baseline. 25 mm is twice the cylinder's radius, so a shaft that has merely rolled in
#: the grip does not read as one.
#: The window at the end of a trace that the study's primary metric averages over. A single
#: last frame is one sample of a shaft that is still settling on its contacts; a second of it is
#: the pose the hand actually ended up holding. Averaging also makes the metric insensitive to
#: exactly when the operator stopped the recording, which is not a property of the hand.
HOLD_WINDOW_S = 1.0

DROP_FALL_MM = 25.0
#: ...and stay down. A single frame this low is a mis-detection; a fall persists.
DROP_HOLD_S = 0.4


@dataclass
class TraceSummary:
    """The scorecard. Every field here is a MEASUREMENT; nothing is a prediction."""
    frames: int = 0
    seen: int = 0
    visibility: float = 0.0
    duration_s: float = 0.0
    cos_start: float | None = None
    cos_peak: float | None = None
    cos_final: float | None = None
    cos_hold: float | None = None
    hold_window_s: float = HOLD_WINDOW_S
    cos_min: float | None = None
    t_peak_s: float | None = None
    deg_turned: float | None = None
    z_start_mm: float | None = None
    z_final_mm: float | None = None
    z_drop_mm: float | None = None
    slip_mm: float | None = None
    dropped: bool = False
    drop_at_s: float | None = None
    wrong_pole: bool = False
    below_floor_mm: float | None = None
    gaps: list[tuple[float, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["gaps"] = [[round(a, 2), round(b, 2)] for a, b in self.gaps]
        return d

    def line(self) -> str:
        if not self.seen:
            return "no cylinder tag in the whole trace -- nothing measured"
        turn = f"{self.deg_turned:+.1f} deg" if self.deg_turned is not None else "--"
        drop = f", DROPPED at {self.drop_at_s:.1f}s" if self.dropped else ""
        return (f"cos {self.cos_start:+.3f} -> {self.cos_final:+.3f} (hold "
                f"{self.cos_hold:+.3f}, peak {self.cos_peak:+.3f} "
                f"at {self.t_peak_s:.1f}s), turned {turn}, height "
                f"{self.z_start_mm:.0f} -> {self.z_final_mm:.0f} mm, slip {self.slip_mm:.1f} mm, "
                f"seen {100 * self.visibility:.0f}%{drop}")


def summarise(readings, *, total_frames: int | None = None,
              drop_fall_mm: float = DROP_FALL_MM,
              drop_hold_s: float = DROP_HOLD_S,
              hold_window_s: float = HOLD_WINDOW_S) -> TraceSummary:
    """Turn a trace into the numbers a bench session records.

    `readings` are the frames where the tag WAS seen; `total_frames` is how many were
    captured, so visibility is honest. Everything is computed against the trace's own
    first sample: the interesting quantity on this bench is the CHANGE the turn produced,
    and the absolute height only says whether the staging matched the plan.
    """
    rs = [r for r in readings if r is not None]
    s = TraceSummary(frames=int(total_frames if total_frames is not None else len(rs)),
                     seen=len(rs))
    s.visibility = s.seen / s.frames if s.frames else 0.0
    if not rs:
        s.notes.append("the cylinder tag was never detected")
        return s
    s.duration_s = rs[-1].t - rs[0].t
    cos = [r.cos_up for r in rs]
    s.cos_start, s.cos_final = cos[0], cos[-1]
    s.cos_peak, s.cos_min = max(cos), min(cos)
    # The primary metric of the transfer study: the alignment the hand ENDED UP holding, not
    # the best instant it passed through. `cos_peak` without a height check scores a shaft that
    # was flicked upright and dropped exactly like one that was carried and kept.
    t_end = rs[-1].t
    held = [r.cos_up for r in rs if r.t >= t_end - hold_window_s]
    s.cos_hold = float(np.mean(held)) if held else cos[-1]
    s.hold_window_s = float(hold_window_s)
    s.t_peak_s = rs[int(np.argmax(cos))].t
    s.deg_turned = deg_from_up(cos[0]) - deg_from_up(cos[-1])   # + = moved toward vertical
    s.z_start_mm, s.z_final_mm = rs[0].z_bench_mm, rs[-1].z_bench_mm
    s.z_drop_mm = s.z_start_mm - min(r.z_bench_mm for r in rs)

    # Slip is horizontal only. A cylinder standing up necessarily RISES (its centre climbs
    # as it comes off the post) so vertical travel is signal, not slip.
    if rs[0].xy_bench_mm is not None:
        x0, y0 = rs[0].xy_bench_mm
        s.slip_mm = max(math.hypot(r.xy_bench_mm[0] - x0, r.xy_bench_mm[1] - y0) for r in rs)
    else:
        s.slip_mm = max(abs(r.radial_mm - rs[0].radial_mm) for r in rs)
        s.notes.append("slip is a RADIAL change only (no heading calibrated); a shaft that "
                       "moved tangentially around the datum reads as zero slip")

    # A drop has to be a sustained fall.
    below = [r for r in rs if r.z_bench_mm < s.z_start_mm - drop_fall_mm]
    if below:
        first = below[0].t
        if below[-1].t - first >= drop_hold_s or first >= rs[-1].t - drop_hold_s:
            s.dropped, s.drop_at_s = True, first

    # A cylinder centre cannot be under the bench. When it reads that way the axial offset is
    # being applied to the WRONG END of the shaft: the centre is computed as tag - 71*u, so a
    # `--shaft-axis` pointing from the tag toward the centre instead of the other way puts the
    # centre 2 x 71 x cos on the far side. It is invisible while the shaft lies flat -- cos ~ 0,
    # so the offset is horizontal and the height is right either way -- and only appears once
    # the tool stands up, which is exactly when the run matters.
    lowest = min(r.z_bench_mm for r in rs)
    if lowest < 0.0:
        s.below_floor_mm = lowest
        s.notes.append(
            f"the cylinder CENTRE reads {lowest:.0f} mm, below the bench floor, which is not a "
            f"pose. Almost always --shaft-axis has the wrong sign: flipping it puts the centre "
            f"at {max(r.z_bench_mm + 2 * CYL_TAG_AXIAL_MM * r.cos_up for r in rs):.0f} mm "
            "instead. Height, slip, x/y and the drop verdict are all wrong until it is fixed; "
            "the ANGLE is unaffected except for its sign. Stand the tool up and re-probe")

    if s.cos_min < -0.15:
        s.wrong_pole = True
        s.notes.append(f"cos reached {s.cos_min:+.3f}: the shaft swung past horizontal to the "
                       "WRONG pole. The signed peak already excludes it; it is not progress")
    # Dropouts are holes in the trace, and a hole during the turn is not a slip.
    for a, b in zip(rs, rs[1:]):
        if b.t - a.t > 0.25:
            s.gaps.append((a.t, b.t))
    if s.gaps:
        s.notes.append(f"{len(s.gaps)} dropout(s) longer than 0.25 s; a gap during the turn "
                       "is missing data, not a measurement")
    if s.visibility < 0.9:
        s.notes.append(f"the tag was visible in only {100 * s.visibility:.0f}% of frames")
    return s
