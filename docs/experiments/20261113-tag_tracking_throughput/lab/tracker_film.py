"""A short film of the live tracker: both cameras' views around the hex block, side by side, each frame with its
detection window (yellow), the tags found (green, with ids), and the fused pose's axes projected into it (x red, y
green, z blue, 15 mm), over a bar with the output rate, the frame age at publish, the tags used and the RMS. Frames are
the tracker's own (luma), cropped to a fixed box around the block found in the first second, and written at the
camera rate; H.264 by ffmpeg.

    /usr/bin/python3 tracker_film.py OUT.mp4 [--seconds 10] [--width 1920 --height 1080 --fps 30] [--crop 520]
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from sr2hand.geometry.rotations import tf_inv  # noqa: E402
from sr2hand.tracking import live, pose as tp  # noqa: E402

CAL = ROOT / "docs/experiments/20261113-camera_arm_calibration/data/fits"
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--seconds", type=float, default=10)
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--crop", type=int, default=520, help="crop box side at 1080p, px (scaled with the mode)")
ap.add_argument("--panel", type=int, default=540, help="displayed panel side, px")
a = ap.parse_args()
cams = live.load_cameras(CAL / "20261113-lens_board_k3_arm.json", CAL / "20261113-handeye_tcp_fitted_lens_moved836.json",
                         a.width, a.height)
tm = live.load_map(CAL / "20261113-hex_block_map.json")
frames, lock = {}, threading.Lock()
samples = []
crop = int(a.crop * a.width / 1920)
box = {}                                                # sn -> (x0, y0) of the crop, set after the warm-up


def on_frame(sn, y, det):
    if sn not in box:
        return
    x0, y0 = box[sn]
    with lock:
        frames[sn] = (y[y0:y0 + crop, x0:x0 + crop].copy(), det, (x0, y0))


def on_sample(s):
    with lock:
        samples.append((s, {sn: frames.get(sn) for sn in cams}))


tr = live.Tracker(cams, tm, fps=a.fps, on_sample=on_sample)
for w in tr.workers:
    w.on_frame = on_frame
tr.start()
time.sleep(1.5)
for w in tr.workers:                                    # the crop: centred on the block's corners in each camera
    if w.last_corners is not None:
        cx, cy = w.last_corners.mean(0)
        box[w.sn] = (int(np.clip(cx - crop / 2, 0, a.width - crop)), int(np.clip(cy - crop / 2, 0, a.height - crop)))
time.sleep(0.5)
with lock:
    samples.clear()
time.sleep(a.seconds)
tr.stop()
S = list(samples)
sns = sorted(cams)
P = a.panel
tmp = Path(a.out).with_suffix(".raw.mp4")
vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (2 * P, P + 60))
t_prev = None
for s, fr in S:
    m = s.msg
    panels = []
    for sn in sns:
        v = fr.get(sn)
        img = np.zeros((P, P, 3), np.uint8)
        if v is not None:
            g, det, (x0, y0) = v
            g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            k = P / crop
            img = cv2.resize(g, (P, P), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
            off = np.array([x0, y0], float)
            if det.roi is not None:
                rx0, ry0, rx1, ry1 = det.roi
                cv2.rectangle(img, tuple(np.int32((np.array([rx0, ry0]) - off) * k)),
                              tuple(np.int32((np.array([rx1, ry1]) - off) * k)), (0, 220, 255), 2)
            for i, c in det.corners.items():
                q = np.int32((c - off) * k)
                cv2.polylines(img, [q], True, (0, 255, 0), 2)
                cv2.putText(img, str(i), tuple(np.int32(q.mean(0) - [8, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
            if s.T_BO is not None:
                cam, T_BC = cams[sn]
                T_CO = tf_inv(T_BC) @ s.T_BO
                pts = tp.project(cam, T_CO, np.array([[0, 0, 0], [15, 0, 0], [0, 15, 0], [0, 0, 15]], float))
                o = tuple(np.int32((pts[0] - off) * k))
                for j, col in ((1, (0, 0, 255)), (2, (0, 200, 0)), (3, (255, 80, 0))):
                    cv2.line(img, o, tuple(np.int32((pts[j] - off) * k)), col, 3)
        cv2.putText(img, sn, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panels.append(img)
    bar = np.full((60, 2 * P, 3), 255, np.uint8)
    rate = "" if t_prev is None else f"{1000.0 / max(m.t_publish_ms - t_prev, 1e-3):5.1f} Hz"
    t_prev = m.t_publish_ms
    txt = (f"{a.width}x{a.height}, {a.fps} fps   output {rate}   frame age {m.t_publish_ms - m.t_capture_ms:4.1f} ms   "
           f"cameras {m.n_cams}   tags {m.n_tags}   rms {m.rms_px:.2f} px")
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(bar, txt, (max(8, (2 * P - tw) // 2), 38), cv2.FONT_HERSHEY_SIMPLEX, min(0.7, 0.7 * (2 * P - 16) / tw),
                (0, 0, 0), 2)
    vw.write(np.vstack([np.hstack(panels), bar]))
vw.release()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "23", "-movflags", "+faststart", a.out], check=True)
tmp.unlink()
mid = len(S) // 2
cap = cv2.VideoCapture(a.out)
cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
ok, img = cap.read()
if ok:
    cv2.imwrite(str(Path(a.out).with_name(Path(a.out).stem + "_poster.jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(f"{len(S)} frames -> {a.out}", flush=True)
os._exit(0)
