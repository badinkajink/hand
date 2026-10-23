"""Overhead D435 time-lapse during a run: one colour frame every PERIOD s as JPEG into OUT, until the file
logs/ur5_lab_stop_timelapse exists or SECONDS pass. 640 x 360 at 15 fps, colour only, so it loads the computer little.
Read-only (the camera).

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/timelapse_d435.py OUT [--period 2] [--seconds 7200]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
STOP = ROOT / "logs/ur5_lab_stop_timelapse"
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--period", type=float, default=2.0)
ap.add_argument("--seconds", type=float, default=7200.0)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
STOP.unlink(missing_ok=True)
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.color, 640, 360, rs.format.rgb8, 15)
pipe.start(cfg)
t0, n = time.time(), 0
try:
    for _ in range(30):
        pipe.wait_for_frames()
    while time.time() - t0 < a.seconds and not STOP.exists():
        fr = pipe.wait_for_frames()
        img = np.asanyarray(fr.get_color_frame().get_data())
        Image.fromarray(img).save(out / f"{time.strftime('%H%M%S')}_{n:05d}.jpg", quality=85)
        n += 1
        time.sleep(max(0.0, a.period - 0.07))
finally:
    pipe.stop()
    print(f"{n} frames in {out}")
