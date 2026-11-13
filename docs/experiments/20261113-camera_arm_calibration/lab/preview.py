"""Live window of both D435 colour views for the owner while they jog the arm (RealSense Viewer and pyrealsense2 cannot
share the cameras on corvus: the viewer is a source build that claims the USB interfaces itself). It draws the
board's detected ChArUco corners in green with their count, and the corners of every recorded pose (OUTDIR/poses.jsonl)
as small grey dots, so empty parts of each image show.

While OUTDIR/PAUSE exists it releases the cameras and shows "RECORDING - hold still", so record_pose.py can open them;
it streams again once PAUSE is gone. q in the window or OUTDIR/QUIT ends it.

    DISPLAY=:1 XAUTHORITY=/run/user/1002/gdm/Xauthority /usr/bin/python3 preview.py OUTDIR
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

out = Path(sys.argv[1])
A = cv2.aruco
DICT = A.getPredefinedDictionary(A.DICT_5X5_50)
BOARD = A.CharucoBoard_create(7, 7, 15.0, 10.5, DICT)
RGB = {"enable_auto_exposure": 0, "enable_auto_white_balance": 0, "exposure": 166, "gain": 64, "white_balance": 4600}
W, H = 960, 540
WIN = "SR2 calibration cameras (q quits)"


def start():
    cams = {}
    for dev in rs.context().devices:
        sn = dev.get_info(rs.camera_info.serial_number)
        p, c = rs.pipeline(), rs.config()
        c.enable_device(sn)
        c.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
        prof = p.start(c)
        for s in prof.get_device().query_sensors():
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                for o, v in RGB.items():
                    for _ in range(10):
                        try:
                            s.set_option(getattr(rs.option, o), v)
                            break
                        except RuntimeError:
                            time.sleep(0.2)
        cams[sn] = p
    return cams


def recorded():
    pts = {}
    try:
        for line in open(out / "poses.jsonl"):
            r = json.loads(line)
            for sn, c in r["cams"].items():
                pts.setdefault(sn, []).extend(c["color"]["corners_mean"])
    except (OSError, ValueError, KeyError):
        pass
    return pts


cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 2 * W, H)
cams, last, seen_t = None, {}, 0.0
while True:
    if (out / "QUIT").exists():
        break
    if (out / "PAUSE").exists():
        if cams:
            for p in cams.values():
                p.stop()
            cams = None
        img = np.zeros((H, 2 * W, 3), np.uint8)
        cv2.putText(img, "RECORDING - hold still", (W - 420, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        cv2.imshow(WIN, img)
        if cv2.waitKey(200) & 0xFF == ord("q"):
            break
        continue
    if cams is None:
        try:
            cams = start()
        except RuntimeError as e:
            print(time.strftime("%H:%M:%S"), "start failed:", e, flush=True)
            cams = None
            cv2.waitKey(1000)
            continue
        pts = recorded()
    panels = []
    for sn in sorted(cams):
        try:
            f = cams[sn].wait_for_frames(2000)
        except RuntimeError:
            continue
        col = np.asanyarray(f.get_color_frame().get_data()).copy()
        gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
        c, ids, rej = A.detectMarkers(gray, DICT, parameters=A.DetectorParameters_create())
        n = 0
        if ids is not None:
            c, ids, _, _ = A.refineDetectedMarkers(gray, BOARD, c, ids, rej)
            n, cc, ci = A.interpolateCornersCharuco(c, ids, gray, BOARD)
            n = int(n or 0)
        img = cv2.resize(col, (W, H))
        k = W / 1920
        for x, y in pts.get(sn, []):
            cv2.circle(img, (int(x * k), int(y * k)), 2, (160, 160, 160), -1)
        if n:
            for x, y in cc.reshape(-1, 2) * (W / 1280):
                cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
        ok = n >= 12
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), (0, 200, 0) if ok else (0, 0, 255), 4)
        cv2.putText(img, f"{sn}  {n}/36 corners", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if ok else (0, 0, 255), 2)
        panels.append(img)
    if len(panels) == 2:
        cv2.imshow(WIN, np.hstack(panels))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
if cams:
    for p in cams.values():
        p.stop()
cv2.destroyAllWindows()
os._exit(0)
