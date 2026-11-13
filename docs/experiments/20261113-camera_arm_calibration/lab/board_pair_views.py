"""Hand-held views of the paper ChArUco board seen by both D435s at once, to re-solve one camera's pose after it was
moved (stereo_pose_fit.py), the other camera holding its arm-fitted pose.

Both cameras stream colour 1920 x 1080 at 15 fps with record_pose.py's session settings. In turn, each camera's newest
frame is searched for the board (page 1 of the 11-13 paper boards, CharucoBoard_create(7, 7, 15.0, 10.5, DICT_5X5_100),
ids 0-23). When both cameras see at least MIN corners (12), the corners moved less than STILL px (1.0) since the
previous search in both, and the board is new in either camera (under 60 % of its corners shared with every saved view,
or the shared ones moved 40 px or more), a burst of BURST framesets (6) is taken. Each corner is averaged over the
frames that found it; the view is kept when no corner's sd exceeds MAX_SD px (0.35) in either camera. Hand-held on
11-13 the owner found 1.0 / 0.35 too strict; --still 2 --max-sd 0.8 was used (0.8 px is about 0.2 mm at 0.3 m).

A window (DISPLAY :1) shows both cameras: the board's corners now in green, every saved view's corners as grey dots,
the saved count, and SAVED for a moment after each save.

Writes OUTDIR/<serial>_NNN.png (the burst's first frame), OUTDIR/pairs.jsonl and OUTDIR/status.json; a restart on the
same OUTDIR continues. Stops after SECONDS, at CAP views, when OUTDIR/STOP exists, or on q in the window.

    /usr/bin/python3 board_pair_views.py OUTDIR [--seconds 900] [--cap 40] [--still 1.0] [--max-sd 0.35]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

A = cv2.aruco
SETTINGS = {"enable_auto_exposure": 0, "enable_auto_white_balance": 0, "exposure": 166, "gain": 64,
            "white_balance": 4600}
DICT = A.getPredefinedDictionary(A.DICT_5X5_100)
BOARD = A.CharucoBoard_create(7, 7, 15.0, 10.5, DICT)
N_IDS = 24

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--seconds", type=float, default=900)
ap.add_argument("--cap", type=int, default=40)
ap.add_argument("--min", type=int, default=12)
ap.add_argument("--still", type=float, default=1.0)
ap.add_argument("--burst", type=int, default=6)
ap.add_argument("--max-sd", type=float, default=0.35, help="px, largest corner sd over the burst")
a = ap.parse_args()
out = Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)


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


def moved(cur, prev):
    sh = [j for j in cur if j in prev]
    if len(sh) < len(cur) // 2:
        return np.inf
    return float(np.median([np.linalg.norm(cur[j] - prev[j]) for j in sh]))


def novel(cur, views):
    for v in views:
        sh = [j for j in cur if j in v]
        if len(sh) >= 0.6 * len(cur) and np.median([np.linalg.norm(cur[j] - v[j]) for j in sh]) < 40:
            return False
    return True


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
    cams[sn] = dict(pipe=p, prev={}, views=[])
SNS = sorted(cams)
if (out / "pairs.jsonl").exists():
    for line in open(out / "pairs.jsonl"):
        r = json.loads(line)
        for sn in SNS:
            cams[sn]["views"].append({int(j): np.array(x) for j, x in zip(r["cams"][sn]["ids"], r["cams"][sn]["corners"])})
(out / "session.json").write_text(json.dumps(dict(board="paper", spec=[7, 7, 15.0, 10.5], dict="DICT_5X5_100",
                                                  settings=SETTINGS, start=time.time()), indent=1))
WIN, PW, PH = "board in both cameras (q quits)", 960, 540
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 2 * PW, PH)
flash, note = 0.0, ""
t0 = time.time()
log = open(out / "pairs.jsonl", "a")
print(f"{time.strftime('%H:%M:%S')} recording board pairs from {SNS}", flush=True)


def grab(sn):
    col = np.asanyarray(cams[sn]["pipe"].wait_for_frames().get_color_frame().get_data()).copy()
    return col, detect(cv2.cvtColor(col, cv2.COLOR_BGR2GRAY))


def panel(sn, col, cur):
    img, k = cv2.resize(col, (PW, PH)), PW / 1920
    for v in cams[sn]["views"]:
        for x, y in v.values():
            cv2.circle(img, (int(x * k), int(y * k)), 2, (170, 170, 170), -1)
    for x, y in cur.values():
        cv2.circle(img, (int(x * k), int(y * k)), 4, (0, 255, 0), -1)
    ok = len(cur) >= a.min
    cv2.putText(img, f"{sn}  {len(cur)}/36  saved {len(cams[sn]['views'])}/{a.cap}  {note}", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if ok else (0, 0, 255), 2)
    if time.time() - flash < 0.8:
        cv2.putText(img, "SAVED", (PW // 2 - 90, PH // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
    return img


try:
    while time.time() - t0 < a.seconds and not (out / "STOP").exists() and len(cams[SNS[0]]["views"]) < a.cap:
        frames, still = {}, True
        for sn in SNS:
            frames[sn] = grab(sn)
            cur = frames[sn][1]
            if len(cur) < a.min or moved(cur, cams[sn]["prev"]) > a.still:
                still = False
            cams[sn]["prev"] = cur
        if still and any(novel(frames[sn][1], cams[sn]["views"]) for sn in SNS):
            burst = {sn: [frames[sn][1]] for sn in SNS}
            first = {sn: frames[sn][0] for sn in SNS}
            for _ in range(a.burst - 1):
                for sn in SNS:
                    burst[sn].append(grab(sn)[1])
            row, ok, means = dict(view=len(cams[SNS[0]]["views"]), t=time.time(), cams={}), True, {}
            for sn in SNS:
                ids = sorted(set().union(*[set(d) for d in burst[sn]]))
                m, sd = {}, {}
                for j in ids:
                    c = np.array([d[j] for d in burst[sn] if j in d])
                    if len(c) < max(2, a.burst // 2):
                        continue
                    m[j], sd[j] = c.mean(0), float(c.std(0).max())
                ok = ok and len(m) >= a.min and max(sd.values()) <= a.max_sd
                means[sn] = m
                row["cams"][sn] = dict(ids=sorted(m), corners=[np.round(m[j], 3).tolist() for j in sorted(m)],
                                       sd_px=[round(sd[j], 3) for j in sorted(m)])
            if not ok:
                note = "moved: not saved"
            else:
                k = row["view"]
                for sn in SNS:
                    cams[sn]["views"].append(means[sn])
                    cv2.imwrite(str(out / f"{sn}_{k:03d}.png"), first[sn])
                log.write(json.dumps(row) + "\n")
                log.flush()
                flash, note = time.time(), ""
                print(f"{time.strftime('%H:%M:%S')} view {k}: " + ", ".join(f"{sn} {len(means[sn])}" for sn in SNS)
                      + " corners", flush=True)
                (out / "status.json").write_text(json.dumps({"views": k + 1}))
        cv2.imshow(WIN, np.hstack([panel(sn, *frames[sn]) for sn in SNS]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    for cam in cams.values():
        cam["pipe"].stop()
    cv2.destroyAllWindows()
    print(f"{time.strftime('%H:%M:%S')} stopped: {len(cams[SNS[0]]['views'])} views", flush=True)
    import os
    os._exit(0)
