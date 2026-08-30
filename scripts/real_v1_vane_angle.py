#!/usr/bin/env python3
"""Read the cylinder's turn angle out of a bench video, per frame.

The measurement trick this implements (see
docs/experiments/20260830-real_v1_bench_suite/README.md section 3): the turn is a
rotation about the pinch axis (palm X).  A flat vane on the cylinder's end, with
its face normal along X, stays face-on to a camera looking along X for the whole
turn -- so the turn is a pure IN-PLANE image rotation of the tag on that vane.
In-plane angle is the most accurate quantity a tag detector produces: sub-degree,
no pose ambiguity, no camera calibration beyond lens distortion.  Reading a tag on
the cylinder's END FACE instead would be the out-of-plane, ambiguous case.

A second tag on something that never moves gives a reference angle, so the reported
angle is (vane - reference) and camera roll, drift and re-mounting all cancel.

Family is DICT_APRILTAG_36h11, which OpenCV ships.  Frames come from imageio, which
the project already depends on, so the only extra is the detector:

    uv pip install opencv-python-headless

Only the four corner pixels are used -- never a pose estimate.

  python3 scripts/real_v1_vane_angle.py run.mp4 --vane-id 0 --ref-id 1 \
      --t0 <started_unix from the run's meta> --out vane.csv

Align to a step with  frame_t == started_unix + step["t_s"].

Validated on SYNTHETIC footage only (2026-08-30): a 70 deg sweep filmed by a
camera rolled 7 deg reads back to 0.33 deg rms / 0.64 deg max, with the reference
tag absorbing the roll exactly, and a sweep crossing the +-180 wrap reads to
0.51 deg max.  There is no vane and no camera yet, so nothing here has met real
lighting, motion blur or a tag at an angle -- expect a pass once the first real
video exists.
"""
import argparse
import csv
import math
import sys

import numpy as np


def _cv2():
    try:
        import cv2
    except ImportError:
        sys.exit("this needs a tag detector:  uv pip install opencv-python-headless")
    return cv2


def tag_angle_deg(corners):
    """In-plane rotation of one tag, from its top edge.  CCW-positive, degrees.

    corners is the (4, 2) pixel array OpenCV returns, ordered clockwise from the
    tag's own top-left, so corner[1] - corner[0] is the tag's +x edge.  Image y
    grows downward, hence the negation: the result is CCW in the way a person
    looking at the frame would call it.
    """
    c = np.asarray(corners, dtype=float).reshape(4, 2)
    dx, dy = c[1] - c[0]
    return math.degrees(math.atan2(-dy, dx))


def unwrap(prev, cur):
    """atan2 wraps at +-180; the turn does not.  Keep the running angle continuous."""
    if prev is None:
        return cur
    return cur + 360.0 * round((prev - cur) / 360.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--vane-id", type=int, default=0, help="tag on the cylinder's vane")
    ap.add_argument("--ref-id", type=int, default=1,
                    help="tag on something rigid that never moves; -1 to skip")
    ap.add_argument("--t0", type=float, default=0.0,
                    help="started_unix from the run's meta, so t is comparable to the JSONL")
    ap.add_argument("--zero-frame", type=int, default=None,
                    help="frame index to call 0 deg (default: the first frame with both tags)")
    ap.add_argument("--out", default=None, help="CSV out (default: stdout summary only)")
    a = ap.parse_args()

    cv2 = _cv2()
    import imageio.v3 as iio

    try:
        fps = float(iio.immeta(a.video, plugin="pyav")["fps"])
    except Exception:
        fps = 30.0

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

    rows, zero, prev_v, prev_r = [], None, None, None
    i = -1
    for frame in iio.imiter(a.video):
        i += 1
        corners, ids, _ = det.detectMarkers(np.asarray(frame))
        seen = {} if ids is None else {int(k): c for k, c in zip(ids.flatten(), corners)}
        if a.vane_id not in seen:
            continue
        vc = np.asarray(seen[a.vane_id]).reshape(4, 2)
        v = prev_v = unwrap(prev_v, tag_angle_deg(vc))
        r = 0.0
        if a.ref_id >= 0:
            if a.ref_id not in seen:
                continue                      # no reference this frame -> no usable angle
            r = prev_r = unwrap(prev_r, tag_angle_deg(seen[a.ref_id]))
        ang = v - r
        if zero is None and (a.zero_frame is None or i >= a.zero_frame):
            zero = ang
        cx, cy = vc.mean(axis=0)
        rows.append({"frame": i, "t": round(a.t0 + i / fps, 3),
                     "vane_deg": round(v, 3), "ref_deg": round(r, 3),
                     "angle_deg": round(ang - zero, 3),
                     "cx": round(float(cx), 1), "cy": round(float(cy), 1)})
    if not rows:
        print("no frame had the vane tag (and the reference, if required)"); return 1

    ang = [r["angle_deg"] for r in rows]
    # The vane's centroid sliding ALONG the shaft is axial slip -- the defect no
    # reward term has ever measured.  It is a pixel number here, not millimetres,
    # until someone puts a scale in the frame.
    cx = [r["cx"] for r in rows]
    print(f"  frames with a read : {len(rows)} of {i + 1}  ({fps:.1f} fps)")
    print(f"  angle              : start {ang[0]:+.1f}  end {ang[-1]:+.1f}  "
          f"peak {max(ang, key=abs):+.1f} deg")
    print(f"  vane centroid drift: {max(cx) - min(cx):.1f} px along the image x "
          f"(axial slip if the shaft lies along image x)")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
