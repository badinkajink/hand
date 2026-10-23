"""One frame of the overhead RealSense D435: colour PNG and depth aligned to colour (npy, metres), after 30 frames of
auto-exposure. Read-only.

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/grab_d435.py NAME
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
from PIL import Image

HERE = Path(__file__).resolve().parent
name = sys.argv[1]
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 15)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
prof = pipe.start(cfg)
try:
    align = rs.align(rs.stream.color)
    for _ in range(30):
        pipe.wait_for_frames()
    fr = align.process(pipe.wait_for_frames())
    c, d = fr.get_color_frame(), fr.get_depth_frame()
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    img = np.asanyarray(c.get_data())
    depth = np.asanyarray(d.get_data()).astype(np.float32) * scale
    intr = c.profile.as_video_stream_profile().intrinsics
    Image.fromarray(img).save(HERE / f"{name}_color.png")
    np.save(HERE / f"{name}_depth_m.npy", depth)
    dev = prof.get_device()
    meta = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), serial=dev.get_info(rs.camera_info.serial_number),
                intrinsics=dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy,
                                model=str(intr.model), coeffs=list(intr.coeffs)), depth_scale=scale)
    (HERE / f"{name}_meta.json").write_text(json.dumps(meta, indent=1))
    valid = depth[depth > 0]
    print(f"{name}: {img.shape[1]}x{img.shape[0]} colour, depth {valid.size / depth.size:.0%} valid, "
          f"median {np.median(valid):.3f} m; serial {meta['serial']}")
finally:
    pipe.stop()
