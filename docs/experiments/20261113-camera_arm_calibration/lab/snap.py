"""Reference frames of both D435s: the mean of N colour frames per camera (record_pose.py's session settings) as
OUTDIR/<serial>_ref.png, and the hex block's tags found in it (OUTDIR/tags.json). Taken before a camera is moved, they
show afterwards whether the other camera moved too (the static scene's shift, the block's pose).

    /usr/bin/python3 snap.py OUTDIR [N]
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from sr2hand.tracking import pose as tp  # noqa: E402

SETTINGS = {"enable_auto_exposure": 0, "enable_auto_white_balance": 0, "exposure": 166, "gain": 64,
            "white_balance": 4600}
out = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
out.mkdir(parents=True, exist_ok=True)
det = tp.detector()
tags = {}
for dev in rs.context().devices:
    sn = dev.get_info(rs.camera_info.serial_number)
    p, c = rs.pipeline(), rs.config()
    c.enable_device(sn)
    c.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
    prof = p.start(c)
    rgb = [s for s in prof.get_device().query_sensors() if s.get_info(rs.camera_info.name) == "RGB Camera"][0]
    for o, v in SETTINGS.items():
        rgb.set_option(getattr(rs.option, o), v)
    for _ in range(15):
        p.wait_for_frames()
    acc = np.zeros((1080, 1920, 3))
    for _ in range(n):
        acc += np.asanyarray(p.wait_for_frames().get_color_frame().get_data())
    p.stop()
    img = np.clip(acc / n + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / f"{sn}_ref.png"), img)
    tags[sn] = {str(i): c.tolist() for i, c in tp.detect(det, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).items()}
    print(f"{time.strftime('%H:%M:%S')} {sn}: tags {sorted(int(i) for i in tags[sn])}", flush=True)
(out / "tags.json").write_text(json.dumps(dict(t=time.time(), tags=tags)))
import os  # noqa: E402
os._exit(0)
