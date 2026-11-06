"""The D435 beside the SR2 hand, colour and depth: every PERIOD s one colour JPEG and one depth PNG aligned to it (uint16,
the median of 5 frames, zeros left out; times depth_scale is metres), until OUT/STOP exists or SECONDS pass. OUT gets
intrinsics.json (the colour stream's pinhole and the depth scale). Read-only (the camera). For checking the gantry's
layout moves when the owner cannot look (11-05: "maybe you can do some intelligent geometry processing with the d435 to
roughly estimate travel of each finger"); lab/layout_travel.py measures them.

    /usr/bin/python3 docs/experiments/20261104-sr2_hand_crosstalk/lab/d435_rgbd.py OUT [--period 2] [--seconds 14400]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--period", type=float, default=2.0)
ap.add_argument("--seconds", type=float, default=14400.0)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 15)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
pipe = rs.pipeline()
prof = pipe.start(cfg)
scale = prof.get_device().first_depth_sensor().get_depth_scale()
intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
(out / "intrinsics.json").write_text(json.dumps({"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
                                                 "width": intr.width, "height": intr.height, "model": str(intr.model),
                                                 "coeffs": list(intr.coeffs), "depth_scale": scale}))
align = rs.align(rs.stream.color)
for _ in range(30):                                   # auto exposure settles
    pipe.wait_for_frames(5000)
t_end, n = time.time() + a.seconds, 0
while time.time() < t_end and not (out / "STOP").exists():
    t0 = time.time()
    try:
        ds, colour = [], None
        for _ in range(5):
            fs = align.process(pipe.wait_for_frames(5000))
            ds.append(np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32))
            colour = np.asanyarray(fs.get_color_frame().get_data()).copy()
        d = np.stack(ds)
        d[d == 0] = np.nan
        with np.errstate(all="ignore"):
            med = np.nan_to_num(np.nanmedian(d, axis=0), nan=0.0).astype(np.uint16)
        stamp = time.strftime("%H%M%S") + f"_{n:05d}"
        Image.fromarray(colour).save(out / f"{stamp}.jpg", quality=90)
        Image.fromarray(med).save(out / f"{stamp}_depth.png")
        n += 1
    except RuntimeError as e:                          # a frame that did not arrive: restart the pipeline
        print(time.strftime("%H:%M:%S"), e, flush=True)
        pipe.stop()
        time.sleep(5)
        prof = pipe.start(cfg)
    time.sleep(max(0.0, a.period - (time.time() - t0)))
pipe.stop()
print(f"{n} frames in {out}", flush=True)
