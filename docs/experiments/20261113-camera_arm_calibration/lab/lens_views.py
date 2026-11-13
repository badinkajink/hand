"""Hand-held views of a ChArUco board for each D435 colour camera's lens calibration (intrinsics with distortion).
Both cameras stream colour 1920 x 1080 at 15 fps with record_pose.py's session settings. About five times a second
per camera the latest frame is searched for the board. A frame is saved (PNG, lossless) when the board shows at least
MIN corners, has moved less than STILL px (1.5) since the camera's previous search (held still, no blur), and differs from
every view already saved for that camera: under 60 % of its corners shared, or the shared ones moved 40 px or more
in the median. At most CAP views per camera.

Boards (owner's choice, tag-block session's specs of 11-13): pla = the 11-12 PLA board, CharucoBoard_create(7, 5, 21.0,
14.0, DICT_5X5_100), ids 0-16; paper = page 1 of the 11-13 paper boards, (7, 7, 15.0, 10.5), ids 0-23. The two share
marker codes 0-16: keep the hand's board covered or out of view while the hand-held one is used.

A window (DISPLAY :1) shows both cameras: the board's corners now in green, every saved view's corners as grey dots,
the saved count per camera, and SAVED for a moment after each save. q in the window also stops it.

Writes OUTDIR/<serial>_NNN.png, one line per saved view in OUTDIR/views.jsonl, and OUTDIR/status.json every second
(views per camera and per cell of a 3 x 3 grid of the image). Stops after SECONDS or when OUTDIR/STOP exists.

    /usr/bin/python3 lens_views.py OUTDIR --board pla [--seconds 600] [--min 8] [--cap 40]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

A = cv2.aruco
BOARDS = {"pla": (7, 5, 21.0, 14.0), "paper": (7, 7, 15.0, 10.5)}
SETTINGS = {"enable_auto_exposure": 0, "enable_auto_white_balance": 0, "exposure": 166, "gain": 64,
            "white_balance": 4600}

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--board", choices=sorted(BOARDS), required=True)
ap.add_argument("--seconds", type=float, default=600)
ap.add_argument("--min", type=int, default=8)
ap.add_argument("--cap", type=int, default=40)
ap.add_argument("--still", type=float, default=1.5, help="px of median corner motion between searches")
a = ap.parse_args()
out = Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)
DICT = A.getPredefinedDictionary(A.DICT_5X5_100)
nx, ny, sq, mk = BOARDS[a.board]
BOARD = A.CharucoBoard_create(nx, ny, sq, mk, DICT)
N_IDS = (nx * ny) // 2


def detect(gray):
    c, ids, rej = A.detectMarkers(gray, DICT, parameters=A.DetectorParameters_create())
    if ids is None:
        return {}
    keep = ids.ravel() < N_IDS
    c = [q for q, k in zip(c, keep) if k]
    ids = ids[keep]
    if not len(ids):
        return {}
    c, ids, _, _ = A.refineDetectedMarkers(gray, BOARD, c, ids, rej)
    n, cc, ci = A.interpolateCornersCharuco(c, ids, gray, BOARD)
    return {int(j): xy for j, xy in zip(ci.ravel(), cc.reshape(-1, 2))} if n else {}


def retry(fn, *args):
    for i in range(10):
        try:
            return fn(*args)
        except RuntimeError:
            if i == 9:
                raise
            time.sleep(0.2)


cams = {}
for dev in rs.context().devices:
    sn = dev.get_info(rs.camera_info.serial_number)
    p, c = rs.pipeline(), rs.config()
    c.enable_device(sn)
    c.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
    prof = p.start(c)
    rgb = [s for s in prof.get_device().query_sensors() if s.get_info(rs.camera_info.name) == "RGB Camera"][0]
    for o, v in SETTINGS.items():
        retry(rgb.set_option, getattr(rs.option, o), v)
    i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    cams[sn] = dict(pipe=p, prev={}, views=[], grid=np.zeros((3, 3), int),
                    intr=dict(w=i.width, h=i.height, fx=i.fx, fy=i.fy, cx=i.ppx, cy=i.ppy, coeffs=list(i.coeffs)))
(out / "session.json").write_text(json.dumps(dict(board=a.board, spec=BOARDS[a.board], dict="DICT_5X5_100",
                                                  settings=SETTINGS, start=time.time(),
                                                  intrinsics={s: c["intr"] for s, c in cams.items()}), indent=1))
WIN, PW, PH = "PLA board lens views (q quits)", 960, 540
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 2 * PW, PH)
flash = {}
t0, t_status = time.time(), 0.0
log = open(out / "views.jsonl", "a")
print(f"{time.strftime('%H:%M:%S')} recording {a.board} views from {sorted(cams)}", flush=True)
try:
    while time.time() - t0 < a.seconds and not (out / "STOP").exists():
        for sn, cam in cams.items():
            fs = cam["pipe"].wait_for_frames()
            col = np.asanyarray(fs.get_color_frame().get_data()).copy()
            cur = detect(cv2.cvtColor(col, cv2.COLOR_BGR2GRAY))
            prev, cam["prev"] = cam["prev"], cur
            img, k = cv2.resize(col, (PW, PH)), PW / 1920
            for v in cam["views"]:
                for x, y in v.values():
                    cv2.circle(img, (int(x * k), int(y * k)), 2, (170, 170, 170), -1)
            for x, y in cur.values():
                cv2.circle(img, (int(x * k), int(y * k)), 4, (0, 255, 0), -1)
            msg = f"{sn}  {len(cur)} corners  saved {len(cam['views'])}/{a.cap}"
            cv2.putText(img, msg, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if cur else (0, 0, 255), 2)
            if time.time() - flash.get(sn, 0) < 0.6:
                cv2.putText(img, "SAVED", (PW // 2 - 90, PH // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
            cam["panel"] = img
            if len(cur) < a.min or len(cam["views"]) >= a.cap:
                continue
            shared = [j for j in cur if j in prev]
            if len(shared) < len(cur) // 2 or np.median([np.linalg.norm(cur[j] - prev[j]) for j in shared]) > a.still:
                continue
            novel = True
            for v in cam["views"]:
                sh = [j for j in cur if j in v]
                if len(sh) >= 0.6 * len(cur) and np.median([np.linalg.norm(cur[j] - v[j]) for j in sh]) < 40:
                    novel = False
                    break
            if not novel:
                continue
            k = len(cam["views"])
            cam["views"].append(cur)
            flash[sn] = time.time()
            cv2.imwrite(str(out / f"{sn}_{k:03d}.png"), col)
            xy = np.array(list(cur.values()))
            for r, q in set(zip(np.clip(xy[:, 1] // 360, 0, 2).astype(int), np.clip(xy[:, 0] // 640, 0, 2).astype(int))):
                cam["grid"][r, q] += 1
            log.write(json.dumps(dict(cam=sn, view=k, t=time.time(), ids=sorted(cur),
                                      corners=[cur[j].tolist() for j in sorted(cur)])) + "\n")
            log.flush()
            print(f"{time.strftime('%H:%M:%S')} {sn} view {k}: {len(cur)} corners, centre "
                  f"({xy[:, 0].mean():.0f}, {xy[:, 1].mean():.0f}), span {np.ptp(xy[:, 0]):.0f} x {np.ptp(xy[:, 1]):.0f} px",
                  flush=True)
        if all("panel" in c for c in cams.values()):
            cv2.imshow(WIN, np.hstack([cams[s]["panel"] for s in sorted(cams)]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if time.time() - t_status > 1.0:
            t_status = time.time()
            (out / "status.json").write_text(json.dumps({sn: dict(views=len(c["views"]), grid=c["grid"].tolist())
                                                         for sn, c in cams.items()}))
finally:
    for cam in cams.values():
        cam["pipe"].stop()
    cv2.destroyAllWindows()
    print(f"{time.strftime('%H:%M:%S')} stopped: " + ", ".join(f"{sn} {len(c['views'])} views" for sn, c in cams.items()),
          flush=True)
    import os
    os._exit(0)
