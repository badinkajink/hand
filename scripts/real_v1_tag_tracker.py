#!/usr/bin/env python3
"""The bench's object-pose sensor: two AprilTags, a RealSense, and the tape-measure geometry.

    # once, before you trust anything
    scripts/real_v1_tag_tracker.py --probe

    # a run, written next to the bench session that produced it
    scripts/real_v1_tag_tracker.py --seconds 20 --out logs/20260831-turn.csv --push

    # only if the normally-fixed reference tag has been re-aimed by hand
    scripts/real_v1_tag_tracker.py --calibrate-heading 0,0 --calibration bench_frame.json

This is the maintained version of `docs/experiments/20260830-apriltag-tracking/track_tags.py`,
which stays as the bare probe it was prototyped as. What is added here is everything that makes
a reading comparable to a simulated one rather than just self-consistent:

  * the geometry lives in `morphohand.bench.tags`, so it is unit-testable without a camera and
    the constants have exactly one home;
  * every frame carries the cylinder CENTRE, not the tag -- the tag is on a vane 71 mm out along
    the shaft, so at 90 degrees of turn the tag moves 71 mm while the centre barely moves at all;
  * heights are absolute, above the bench floor, and convertible to the simulator's own z by one
    subtraction (see `tags.SIM_TO_BENCH_Z_MM`);
  * the trace is summarised by the same code the bench session and the control station use, so
    "it turned 42 degrees" means the same thing in all three places.

INTERPRETER. `pyrealsense2` and `pupil_apriltags` are not in this repo's uv environment; on this
workstation they live in ~/miniconda3. Run this with that interpreter (the shebang path is
resolved for you by --print-interpreter, and real_v1_bench_session.py finds it automatically).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from morphohand.bench import tags as T  # noqa: E402

#: Interpreters that might carry the camera stack, best first. Kept here rather than in the
#: library because it is a fact about this workstation, not about the geometry.
CANDIDATE_PYTHONS = ("/home/humanoid/miniconda3/bin/python", sys.executable,
                     "python3.12", "python3.10", "python3")


def find_interpreter() -> str | None:
    """The first interpreter on this machine that can actually open the camera."""
    probe = "import pyrealsense2, pupil_apriltags, cv2, numpy"
    for cand in CANDIDATE_PYTHONS:
        exe = cand if os.path.isabs(cand) else shutil.which(cand)
        if not exe or not os.path.exists(exe):
            continue
        try:
            if subprocess.run([exe, "-c", probe], capture_output=True, timeout=60).returncode == 0:
                return exe
        except Exception:
            continue
    return None


def _camera():
    try:
        import cv2
        import pyrealsense2 as rs
        from pupil_apriltags import Detector
    except ImportError as exc:
        alt = find_interpreter()
        hint = (f"\n  try:  {alt} {' '.join(sys.argv)}" if alt and alt != sys.executable
                else "\n  install pyrealsense2, pupil_apriltags and opencv-python for this "
                     "interpreter")
        raise SystemExit(f"{exc}\n{sys.executable} cannot open the camera.{hint}") from None
    return rs, cv2, Detector


def open_ir(rs, w, h, fps, warmup=30, exposure=None, gain=None):
    """IR stream 1, emitter off, with exposure stable before the reference latch.

    Three ordering rules the hard way, all from the 2026-08-30 bring-up: the emitter write must
    come AFTER frames are flowing (issuing it between start() and the first frame wedges the
    stream), the projector's dot pattern wrecks the tag decode, and auto-exposure hunts as the
    shaft swings so the motion smear changes frame to frame. By default AE is allowed to settle
    on the actual lighting at the start of EACH run and that value is then locked, so a session
    inherits the room it is actually in rather than a number measured in another one, without
    letting exposure hunt during the trajectory. That is a convenience, not a tag-visibility
    fix: a reference tag that has been bumped is lost at every exposure (see `detect`). A
    positive ``exposure`` still requests an explicit fixed exposure/gain for repeatability
    studies, which is what a cross-session comparison of decision margins needs.
    """
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, fps)
    prof = pipe.start(cfg)
    pipe.wait_for_frames(8000)
    for s in prof.get_device().query_sensors():
        if s.supports(rs.option.emitter_enabled):
            s.set_option(rs.option.emitter_enabled, 0)
    sensor = next((s for s in prof.get_device().query_sensors()
                   if s.supports(rs.option.enable_auto_exposure)
                   and s.supports(rs.option.exposure)), None)
    if sensor is None:
        raise RuntimeError("RealSense IR sensor exposes no exposure controls")
    fixed = exposure is not None and float(exposure) > 0
    if fixed:
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, float(exposure))
        if gain is not None and sensor.supports(rs.option.gain):
            sensor.set_option(rs.option.gain, float(gain))
    else:
        sensor.set_option(rs.option.enable_auto_exposure, 1)
    for _ in range(warmup):
        pipe.wait_for_frames(8000)
    if not fixed:
        # Read the values AE chose for this lighting, then freeze exactly those values. Merely
        # leaving AE on fixed the shadowed reference in RealSense Viewer, but produces a moving
        # measurement transfer function as the bright shaft vane swings through the frame.
        actual_exposure = float(sensor.get_option(rs.option.exposure))
        actual_gain = (float(sensor.get_option(rs.option.gain))
                       if sensor.supports(rs.option.gain) else None)
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, actual_exposure)
        if actual_gain is not None:
            sensor.set_option(rs.option.gain, actual_gain)
        for _ in range(3):
            pipe.wait_for_frames(8000)
    actual = {"mode": "fixed" if fixed else "auto-locked",
              "exposure_us": float(sensor.get_option(rs.option.exposure)),
              "gain": (float(sensor.get_option(rs.option.gain))
                       if sensor.supports(rs.option.gain) else None)}
    I = prof.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()
    return pipe, (I.fx, I.fy, I.ppx, I.ppy), I, actual


def detect(det, img, cam, cv2=None):
    """Detect both bench tags, retrying a missing one on two normalizations of the image.

    Neither normalization is a substitute for a tag that is staged correctly, and the retry
    ladder must not be read as one. Measured on the three probe frames of 2026-08-31, the
    reference tag decodes raw at margin 62 at 14:11, needs global equalization to reach 35
    at 15:20, and at 16:57 decodes under nothing at all -- not raw, not equalized, not CLAHE,
    and not under any of the 24 detector parameter combinations swept over quad_decimate,
    quad_sigma, refine_edges and decode_sharpening. Its contrast is the SAME in all three
    (Michelson 0.68 / 0.72 / 0.71) and so is its sharpness, so the 16:57 loss is not an
    exposure problem; the tag had shifted a few pixels, i.e. something bumped it. The probe
    is the instrument that catches that, and the fix is at the bench, not here.

    What the ladder is for is the marginal middle case, and the two normalizations disagree
    about which frame they rescue -- CLAHE scored 62 on the 14:11 frame where equalization
    scored 37, and equalization was the only one that read the 15:20 frame -- so both are
    tried. Raw stays primary and supplies every healthy frame.
    """
    want = ((T.CYL_TAG_ID, T.CYL_TAG_SIZE_M), (T.REF_TAG_ID, T.REF_TAG_SIZE_M))
    out = {}

    def scan(image):
        for tid, size in want:
            if tid in out:
                continue
            for t in det.detect(image, estimate_tag_pose=True, camera_params=cam,
                                tag_size=size):
                if t.tag_id == tid:
                    out[tid] = t

    scan(img)
    if cv2 is not None and len(out) < len(want):
        scan(cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img))
    if cv2 is not None and len(out) < len(want):
        scan(cv2.equalizeHist(img))
    return out


def meter(v, n=13):
    """Position of cos on [-1, +1]; ':' marks horizontal, right is up."""
    cells = ["."] * n
    cells[(n - 1) // 2] = ":"
    cells[int(round((float(np.clip(v, -1.0, 1.0)) + 1.0) / 2.0 * (n - 1)))] = "|"
    return "[" + "".join(cells) + "]"


class Live:
    """In-place status line on a terminal, throttled plain lines when redirected. Events print
    ABOVE the live line so a dropout survives in scrollback."""

    def __init__(self, enabled, hz):
        self.on, self.tty = enabled, sys.stdout.isatty()
        self.every = (1.0 / hz if hz > 0 else 0.0) if self.tty else max(1.0 / max(hz, 1e-9), 0.5)
        self.last, self.width = 0.0, 0

    def event(self, msg):
        if self.on and self.tty and self.width:
            sys.stdout.write("\r" + " " * self.width + "\r")
            self.width = 0
        print(msg, flush=True)

    def update(self, line, force=False):
        if not self.on:
            return
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        if self.tty:
            sys.stdout.write("\r" + line + " " * max(0, self.width - len(line)))
            sys.stdout.flush()
            self.width = len(line)
        else:
            print(line, flush=True)

    def close(self):
        if self.on and self.tty and self.width:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.width = 0


class Pusher:
    """Throttled POST of the live reading to the control station.

    Deliberately best-effort and silent after the first failure: the trace on disk is the
    measurement, the station display is a convenience, and a bench run must not stop because
    the CB1 is busy or the network hiccuped.
    """

    def __init__(self, enabled: bool, hz: float, run_id: str = "", *,
                 push_url: str = "", push_token: str = ""):
        self.on, self.every, self.run_id = enabled, (1.0 / hz if hz > 0 else 0.0), run_id
        self.last, self.sent, self.failed, self.warned = 0.0, 0, 0, False
        self.push_url = push_url.rstrip("/")
        self.push_token = push_token
        self.mh = None
        if self.on and not self.push_url:
            try:
                import mh
                self.mh = mh
            except Exception as exc:
                print(f"  push disabled: {exc}")
                self.on = False

    def send(self, payload: dict, force: bool = False):
        if not self.on:
            return False
        now = time.time()
        if not force and now - self.last < self.every:
            return False
        self.last = now
        try:
            body = {**payload, "run_id": self.run_id}
            if self.push_url:
                req = urllib.request.Request(
                    self.push_url + "/tracker/sample", json.dumps(body).encode(), method="POST",
                    headers={"Content-Type": "application/json",
                             "X-Manta-Token": self.push_token})
                with urllib.request.urlopen(req, timeout=1.5):
                    pass
            else:
                self.mh.post("/tracker/sample", body, timeout=1.5)
            self.sent += 1
            return True
        except Exception as exc:
            self.failed += 1
            if not self.warned:
                self.warned = True
                print(f"\n  push failing ({exc}); the CSV is unaffected", flush=True)
            return False


def load_calibration(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text())


def probe(a, rs, cv2, Detector):
    pipe, cam, I, camera_settings = open_ir(rs, a.width, a.height, a.fps,
                                            exposure=a.exposure, gain=a.gain)
    try:
        det = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0)
        img = np.asanyarray(pipe.wait_for_frames(5000).get_infrared_frame().get_data())
        out = Path(a.probe_image)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
        found = detect(det, img, cam, cv2)
        print(f"IR1 {I.width}x{I.height} fx={I.fx:.1f}  "
              f"HFOV={2 * np.degrees(np.arctan(I.width / (2 * I.fx))):.1f} deg  emitter OFF  "
              f"{camera_settings['mode']} {camera_settings['exposure_us']:.0f}us/"
              f"{camera_settings['gain']:.0f}")
        print(f"exposure looks {'OK' if 20 < img.mean() < 235 else 'BAD'} (mean {img.mean():.0f})")
        for tid, t in sorted(found.items()):
            e = np.mean([np.linalg.norm(t.corners[k] - t.corners[(k + 1) % 4]) for k in range(4)])
            print(f"  id{tid}: margin {t.decision_margin:5.1f}  {e:5.1f}px edge = {e / 8:.1f} "
                  f"px/cell  range {np.linalg.norm(t.pose_t) * 1000:6.1f} mm")
        for tid in (T.CYL_TAG_ID, T.REF_TAG_ID):
            if tid not in found:
                print(f"  id{tid}: NOT SEEN")
        if len(found) < 2:
            print("\nboth tags are needed before any geometry can be checked.")
            return 1
        cal = load_calibration(a.calibration)
        frame = T.BenchFrame.latch([(found[T.REF_TAG_ID].pose_R, found[T.REF_TAG_ID].pose_t)],
                                   up_axis=a.up_axis, plane_axis=a.plane_axis,
                                   heading_deg=cal.get("heading_deg", a.heading_deg))
        r = T.reading_from_tags(frame, found[T.CYL_TAG_ID].pose_R, found[T.CYL_TAG_ID].pose_t,
                                t=0.0, shaft_axis=a.shaft_axis,
                                margin=found[T.CYL_TAG_ID].decision_margin,
                                axial_mm=a.axial_mm)
        print(f"\n  cos(shaft, up) = {r.cos_up:+.4f}   {r.deg_from_up:.1f} deg from up, "
              f"{abs(90.0 - r.deg_from_up):.1f} deg off horizontal")
        print("    -> lay the shaft flat: expect cos ~0.  Stand it up: expect cos ~+1.")
        print(f"\n  tag centre      {r.tag_z_bench_mm:7.1f} mm above the bench floor")
        print(f"  cylinder centre {r.z_bench_mm:7.1f} mm above the bench floor "
              f"(= {r.z_sim_mm:.1f} in the simulator's z)")
        # The sign check that costs nothing and saves a whole session: the centre is 71 mm from
        # the tag ALONG the shaft, so with the shaft upright and the tag on top the centre must
        # read lower. If it reads higher, --shaft-axis points the wrong way.
        if abs(r.cos_up) > 0.5:
            expect = "LOWER" if r.cos_up > 0 else "HIGHER"
            got = "lower" if r.z_bench_mm < r.tag_z_bench_mm else "higher"
            ok = (expect.lower() == got)
            print(f"  shaft is near vertical, so the centre should read {expect} than the tag; "
                  f"it reads {got}  --> --shaft-axis {a.shaft_axis} is "
                  f"{'CORRECT' if ok else 'BACKWARDS (use its negation)'}")
        else:
            print("  shaft is near horizontal, so this frame cannot check --shaft-axis's sign. "
                  "Stand it up and re-probe.")
        print(f"\n  the bench scene stands the tool at {T.BENCH_POST_HEIGHT_SIM_MM:.0f} mm in "
              f"sim = {T.bench_z_mm(T.BENCH_POST_HEIGHT_SIM_MM):.0f} mm on the bench floor")
        if frame.heading_deg is None and not a.no_mounting_heading:
            try:
                h, why = frame.heading_from_mounting(
                    T.object_center_cam_mm(found[T.CYL_TAG_ID].pose_R,
                                           found[T.CYL_TAG_ID].pose_t,
                                           shaft_axis=a.shaft_axis, axial_mm=a.axial_mm)[0])
                frame.heading_deg = h
                r = T.reading_from_tags(frame, found[T.CYL_TAG_ID].pose_R,
                                        found[T.CYL_TAG_ID].pose_t, t=0.0,
                                        shaft_axis=a.shaft_axis, axial_mm=a.axial_mm)
                print(f"  heading {h:+.0f} deg from the MOUNTING (tag normal to the gantry "
                      f"x-axis; {why})")
            except ValueError as exc:
                print(f"  heading unresolved: {exc}")
        if frame.heading_deg is None:
            print("  no heading: heights and radial distances are real, bench (x, y) is "
                  "withheld. --calibrate-heading fixes that.")
        else:
            print(f"  heading {frame.heading_deg:+.2f} deg -> cylinder centre at bench "
                  f"({r.xy_bench_mm[0]:+.1f}, {r.xy_bench_mm[1]:+.1f}) mm")
        print(f"\nwrote {out}")
        return 0
    finally:
        pipe.stop()


def calibrate(a, rs, cv2, Detector):
    want = tuple(float(v) for v in a.calibrate_heading.split(","))
    if len(want) != 2:
        raise SystemExit("--calibrate-heading takes X,Y in mm in the bench frame")
    pipe, cam, I, _camera_settings = open_ir(rs, a.width, a.height, a.fps,
                                             exposure=a.exposure, gain=a.gain)
    try:
        det = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0)
        refs, cyls = [], []
        for _ in range(a.latch_frames):
            img = np.asanyarray(pipe.wait_for_frames(8000).get_infrared_frame().get_data())
            f = detect(det, img, cam, cv2)
            if T.REF_TAG_ID in f:
                refs.append((f[T.REF_TAG_ID].pose_R, f[T.REF_TAG_ID].pose_t))
            if T.CYL_TAG_ID in f:
                cyls.append((f[T.CYL_TAG_ID].pose_R, f[T.CYL_TAG_ID].pose_t))
        if not cyls:
            raise SystemExit("the cylinder tag was never seen; nothing to calibrate against")
        frame = T.BenchFrame.latch(refs, up_axis=a.up_axis, plane_axis=a.plane_axis)
        centers = [T.object_center_cam_mm(R, t, shaft_axis=a.shaft_axis, axial_mm=a.axial_mm)[0]
                   for R, t in cyls]
        c = np.mean(centers, axis=0)
        heading = frame.heading_for(c, want)
        spread = float(np.max(np.linalg.norm(np.array(centers) - c, axis=1)))
        doc = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "heading_deg": round(heading, 3),
               "calibration_point_bench_mm": list(want),
               "frames": {"reference": len(refs), "cylinder": len(cyls)},
               "cylinder_centre_spread_mm": round(spread, 2),
               "up_axis": a.up_axis, "plane_axis": a.plane_axis, "shaft_axis": a.shaft_axis,
               "note": "heading is the bench +x direction measured from the reference tag's "
                       "in-plane horizontal axis, about world up. On the fixed rig the mounting "
                       "normally supplies this; this staged calibration is for a tag that has "
                       "been re-aimed by hand."}
        print(f"heading {heading:+.3f} deg from {len(cyls)} cylinder frames "
              f"(centre spread {spread:.2f} mm)")
        if spread > 5.0:
            print("  !! the cylinder moved during calibration, or the pose is noisy. "
                  "Re-run with it clamped.")
        if a.calibration:
            a.calibration.parent.mkdir(parents=True, exist_ok=True)
            a.calibration.write_text(json.dumps(doc, indent=2) + "\n")
            print(f"wrote {a.calibration}")
        else:
            print(json.dumps(doc, indent=2))
        return 0
    finally:
        pipe.stop()


def record(a, rs, cv2, Detector):
    pipe, cam, I, camera_settings = open_ir(rs, a.width, a.height, a.fps,
                                            exposure=a.exposure, gain=a.gain)
    live = Live(not a.quiet, a.live_hz)
    push = Pusher(a.push, a.push_hz, a.run_id, push_url=a.push_url,
                  push_token=a.push_token)
    try:
        det = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0)
        cal = load_calibration(a.calibration)
        refs = []
        for _ in range(a.latch_frames):
            img = np.asanyarray(pipe.wait_for_frames(8000).get_infrared_frame().get_data())
            f = detect(det, img, cam, cv2)
            if T.REF_TAG_ID in f:
                refs.append((f[T.REF_TAG_ID].pose_R, f[T.REF_TAG_ID].pose_t))
        frame = T.BenchFrame.latch(refs, up_axis=a.up_axis, plane_axis=a.plane_axis,
                                   heading_deg=cal.get("heading_deg", a.heading_deg))
        live.event(f"latched the reference from {len(refs)}/{a.latch_frames} frames"
                   + ("" if frame.heading_deg is None
                      else f", heading {frame.heading_deg:+.2f} deg"))
        if len(refs) < a.latch_frames:
            live.event("  note: intermittent reference. Latching covers it, but light it "
                       "better if you re-aim.")
        live.event(f"recording {a.seconds:.0f}s -- cos +1 = up, 0 = horizontal, -1 = down "
                   f"(wrong pole).  Ctrl-C stops early and still writes {a.out}")

        rows, readings, t0 = [], [], time.time()
        heading_tries = 0
        peak, lost_since, drift_warned, stamps = -1.0, None, False, []
        push_confirmed = False
        try:
            while time.time() - t0 < a.seconds:
                img = np.asanyarray(pipe.wait_for_frames(5000).get_infrared_frame().get_data())
                found = detect(det, img, cam, cv2)
                t = time.time() - t0
                stamps.append(time.time())
                if len(stamps) > 15:
                    stamps.pop(0)
                fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
                if T.REF_TAG_ID in found and not drift_warned:
                    d = frame.drift_deg(found[T.REF_TAG_ID].pose_R, a.up_axis)
                    if d > a.drift_deg:
                        drift_warned = True
                        live.event(f"  [{t:6.2f}s] WARNING: the reference moved {d:.1f} deg -- "
                                   "camera bumped. Angles after this point are referenced to a "
                                   "stale datum")
                if T.CYL_TAG_ID in found:
                    if lost_since is not None:
                        live.event(f"  [{t:6.2f}s] cylinder tag REACQUIRED after "
                                   f"{t - lost_since:.2f}s")
                        lost_since = None
                    tag = found[T.CYL_TAG_ID]
                    r = T.reading_from_tags(frame, tag.pose_R, tag.pose_t, t=t,
                                            shaft_axis=a.shaft_axis,
                                            margin=tag.decision_margin, axial_mm=a.axial_mm)
                    if frame.heading_deg is None and not a.no_mounting_heading:
                        try:
                            h, why = frame.heading_from_mounting(
                                T.object_center_cam_mm(tag.pose_R, tag.pose_t,
                                                       shaft_axis=a.shaft_axis,
                                                       axial_mm=a.axial_mm)[0])
                            frame.heading_deg = h
                            live.event(f"  heading {h:+.0f} deg from the mounting ({why})")
                            r = T.reading_from_tags(frame, tag.pose_R, tag.pose_t, t=t,
                                                    shaft_axis=a.shaft_axis,
                                                    margin=tag.decision_margin,
                                                    axial_mm=a.axial_mm)
                        except ValueError as exc:
                            # Retry rather than give up on the first frame: the tracker can be
                            # armed a moment before the shaft is finally seated, and the
                            # envelope test legitimately refuses a shaft still on its post.
                            heading_tries += 1
                            if heading_tries in (1, 30) or heading_tries % 300 == 0:
                                live.event(f"  heading unresolved, bench x/y withheld: {exc}")
                            if heading_tries >= 600:
                                a.no_mounting_heading = True
                                live.event("  giving up on the mounting heading for this run")
                    readings.append(r)
                    rows.append(r.row())
                    peak = max(peak, r.cos_up)
                    pushed = push.send({"seen": True, **r.row()})
                    if pushed and not push_confirmed:
                        push_confirmed = True
                        live.event("station push confirmed")
                    live.update(f"{t:6.2f}s  cos {r.cos_up:+.3f} {meter(r.cos_up)} "
                                f"{r.deg_from_up:5.1f}deg  z {r.z_bench_mm:6.1f}mm  "
                                f"peak {peak:+.3f}  {fps:4.1f}fps  m{r.margin:3.0f}")
                else:
                    if lost_since is None:
                        lost_since = t
                        live.event(f"  [{t:6.2f}s] cylinder tag LOST")
                    rows.append({k: "" for k in T.CSV_FIELDS} | {"t": round(t, 4)})
                    push.send({"seen": False, "t": round(t, 4)})
                    live.update(f"{t:6.2f}s  -- TAG LOST for {t - lost_since:4.2f}s --"
                                f"                              {fps:4.1f}fps")
        except KeyboardInterrupt:
            live.close()
            print(f"interrupted at {time.time() - t0:.2f}s -- writing what was captured")
        live.close()

        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=T.CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        summary = T.summarise(readings, total_frames=len(rows))
        doc = {"schema_version": 1, "trace": str(a.out), "run_id": a.run_id,
               "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "frame": frame.to_json(), "summary": summary.to_json(),
               "axes": {"up": a.up_axis, "plane": a.plane_axis, "shaft": a.shaft_axis},
               "axial_mm": a.axial_mm, "camera": {"width": a.width, "height": a.height,
                                                  "fps": a.fps, **camera_settings}}
        sidecar = a.out.with_name(a.out.stem + "_SUMMARY.json")
        sidecar.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"{len(rows)} frames -> {a.out}")
        print(f"  {summary.line()}")
        for n in summary.notes:
            print(f"  NOTE: {n}")
        print(f"  wrote {sidecar}")
        if push.on:
            print(f"  pushed {push.sent} samples to the station ({push.failed} failed)")
            push.send({"seen": False, "final": True, "summary": summary.to_json(),
                       "frame": frame.to_json(), "camera": camera_settings,
                       "axes": {"up": a.up_axis, "plane": a.plane_axis,
                                "shaft": a.shaft_axis}, "axial_mm": a.axial_mm}, force=True)
        return 0
    finally:
        pipe.stop()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", action="store_true", help="one frame, print the geometry, exit")
    p.add_argument("--print-interpreter", action="store_true",
                   help="print an interpreter on this machine that can open the camera")
    p.add_argument("--calibrate-heading", default=None, metavar="X,Y",
                   help="solve the bench-frame heading from a cylinder staged at a KNOWN "
                        "bench (x, y) in mm, and write it to --calibration")
    p.add_argument("--calibration", type=Path,
                   default=ROOT / "docs/experiments/20260830-apriltag-tracking/bench_frame.json")
    p.add_argument("--heading-deg", type=float, default=None,
                   help="use this heading instead of a calibration file")
    p.add_argument("--no-mounting-heading", action="store_true",
                   help="do NOT infer the heading from the tag's mounting (it is bolted normal "
                        "to the gantry x-axis, so the heading is +-90 and only the sign is "
                        "unknown; the sign follows from the hand being at x=0 and the tag at "
                        "x=+133.5). Pass this if the tag has been re-aimed by hand.")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--run-id", default="", help="station run_id these samples belong to")
    p.add_argument("--push", action="store_true", help="POST live samples to the control station")
    p.add_argument("--push-hz", type=float, default=10.0)
    p.add_argument("--push-url", default="",
                   help="control-station API base ending in /api/v1; default uses scripts/mh.py")
    p.add_argument("--push-token", default="",
                   help="X-Manta-Token used with --push-url")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--exposure", type=float, default=0.0,
                   help="us; default 0 lets IR auto-exposure settle at run start and then locks "
                        "its chosen exposure/gain. Positive values force a fixed exposure")
    p.add_argument("--gain", type=float, default=64.0)
    p.add_argument("--shaft-axis", default="x",
                   help="direction in the CYLINDER tag's frame pointing from the cylinder "
                        "centre outward to the tag")
    p.add_argument("--up-axis", default="x", help="world up in the REFERENCE tag's frame")
    p.add_argument("--plane-axis", default="y",
                   help="a second REFERENCE tag axis; its horizontal component is the zero "
                        "of the heading")
    p.add_argument("--axial-mm", type=float, default=T.CYL_TAG_AXIAL_MM,
                   help="cylinder centre to tag centre along the shaft")
    p.add_argument("--latch-frames", type=int, default=20)
    p.add_argument("--drift-deg", type=float, default=2.0)
    p.add_argument("--probe-image", default=None,
                   help="default: logs/<stamp>-probe_ir.png.  Not the repo root -- run "
                        "outputs live under logs/ (gitignored)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--live-hz", type=float, default=12.0)
    a = p.parse_args()

    if a.print_interpreter:
        exe = find_interpreter()
        print(exe or "")
        return 0 if exe else 1
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if a.out is None:
        a.out = ROOT / f"logs/{stamp}-turn_trace.csv"
    if a.probe_image is None:
        a.probe_image = ROOT / f"logs/{stamp}-probe_ir.png"

    rs, cv2, Detector = _camera()
    if a.calibrate_heading:
        return calibrate(a, rs, cv2, Detector)
    if a.probe:
        return probe(a, rs, cv2, Detector)
    return record(a, rs, cv2, Detector)


if __name__ == "__main__":
    sys.exit(main())
