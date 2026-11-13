"""Views of the hex tag block turned by hand in front of both D435s, for a two-camera fit of its tag map
(block_map_fit.py).

Both cameras stream colour 1920 x 1080 at 15 fps with record_pose.py's session settings. In turn, each camera's newest
frame is searched for the block's tags (sr2hand.tracking.pose.detect, tags 10-15). When the tags have moved less than
STILL px (1.0) since the previous search in both cameras, at least MIN_TAGS tags are seen in all (3), and the block's
pose in the arm's base frame differs from every saved view's by 15 deg or 25 mm, a burst of BURST framesets (6) is
taken. The pose comes from the nominal map with 13.0 mm tags and today's camera poses. Each tag's corners are averaged
over the frames that saw it. A view is kept when every tag's corner sd stays under 0.35 px (the block did not move).

A window (DISPLAY :1) shows both cameras with the tags now (green, with ids), the saved count, and per tag the views
that saw it in both cameras at once ("both") and in either. Below the panels: how often each pair of neighbouring tags
was seen together by one camera. SAVED flashes after each save.

Writes OUTDIR/<serial>_NNN.png (the burst's first frame), one line per view in OUTDIR/views.jsonl and
OUTDIR/status.json; a restart on the same OUTDIR continues. Stops after SECONDS, when OUTDIR/STOP exists, or on q in the window.

    /usr/bin/python3 block_views.py OUTDIR LENS.json HANDEYE.json [--seconds 900] [--cap 80]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from sr2hand.geometry.rotations import log_so3, tf_inv  # noqa: E402
from sr2hand.tracking import pose as tp  # noqa: E402
from sr2hand.tracking.tags import HexBlock  # noqa: E402

SETTINGS = {"enable_auto_exposure": 0, "enable_auto_white_balance": 0, "exposure": 166, "gain": 64,
            "white_balance": 4600}
TM = HexBlock(across_flats=29.985, tag_size=13.0).tag_map()
IDS = TM.ids()
PAIRS = [(IDS[k], IDS[(k + 1) % 6]) for k in range(6)]

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("lens")
ap.add_argument("handeye")
ap.add_argument("--seconds", type=float, default=900)
ap.add_argument("--cap", type=int, default=80)
ap.add_argument("--still", type=float, default=1.0)
ap.add_argument("--min-tags", type=int, default=3)
ap.add_argument("--burst", type=int, default=6)
a = ap.parse_args()
out = Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)
LENS = json.load(open(a.lens))["cams"]
HE = json.load(open(a.handeye))["cams"]
DET = tp.detector(nthreads=4)


def camera(sn):
    l = LENS[sn]["lens"]
    K = np.array([[l["fx"], 0, l["cx"]], [0, l["fy"], l["cy"]], [0, 0, 1.0]])
    return tp.Camera(K, np.array([l["k1"], l["k2"], l["p1"], l["p2"], l["k3"]], float), 1920, 1080)


def retry(fn, *args):
    for i in range(10):
        try:
            return fn(*args)
        except RuntimeError:
            if i == 9:
                raise
            time.sleep(0.2)


def grab(cam):
    fs = cam["pipe"].wait_for_frames()
    col = np.asanyarray(fs.get_color_frame().get_data()).copy()
    return col, tp.detect(DET, cv2.cvtColor(col, cv2.COLOR_BGR2GRAY), ids=IDS)


def moved(cur, prev):
    sh = [i for i in cur if i in prev]
    if not sh:
        return np.inf
    return float(np.median(np.concatenate([np.linalg.norm(cur[i] - prev[i], axis=1) for i in sh])))


def block_in_base(seen):
    """The block's pose in the base frame from the camera seeing the most tags (nominal map), or None."""
    sn = max(seen, key=lambda s: len(seen[s]))
    if not seen[sn]:
        return None
    T_CB, _ = tp.solve_body(seen[sn], TM, cams[sn]["cam"])
    return None if T_CB is None else tf_inv(np.array(HE[sn]["T_CB"])) @ T_CB


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
    cams[sn] = dict(pipe=p, prev={}, cam=camera(sn))
SNS = sorted(cams)
(out / "session.json").write_text(json.dumps(dict(map="HexBlock(29.985, 13.0)", ids=IDS, settings=SETTINGS,
                                                  lens=a.lens, handeye=a.handeye, start=time.time()), indent=1))
both = {i: 0 for i in IDS}
either = {i: 0 for i in IDS}
pairs = {p: 0 for p in PAIRS}
poses = []
if (out / "views.jsonl").exists():                       # a restart continues the saved views and their counts
    for line in open(out / "views.jsonl"):
        r = json.loads(line)
        poses.append(np.array(r["T_PB_nominal"]))
        sets = {sn: set(r["cams"][sn]["ids"]) for sn in SNS}
        for i in IDS:
            either[i] += any(i in s for s in sets.values())
            both[i] += all(i in s for s in sets.values())
        for p in PAIRS:
            pairs[p] += any(p[0] in s and p[1] in s for s in sets.values())
WIN, PW, PH = "hex block views (q quits)", 960, 540
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 2 * PW, PH + 90)
flash, note = 0.0, ""
t0 = time.time()
log = open(out / "views.jsonl", "a")
print(f"{time.strftime('%H:%M:%S')} recording hex block views from {SNS}", flush=True)


def panel(sn, col, cur):
    img, k = cv2.resize(col, (PW, PH)), PW / 1920
    for i, c in cur.items():
        cv2.polylines(img, [np.int32(c * k)], True, (0, 255, 0), 2)
        cv2.putText(img, str(i), tuple(np.int32(c.mean(0) * k - [10, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
    cv2.putText(img, f"{sn}  tags {sorted(cur)}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if cur else (0, 0, 255), 2)
    if time.time() - flash < 0.8:
        cv2.putText(img, "SAVED", (PW // 2 - 90, PH // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
    return img


try:
    while time.time() - t0 < a.seconds and not (out / "STOP").exists() and len(poses) < a.cap:
        frames, still = {}, True
        for sn in SNS:
            col, cur = grab(cams[sn])
            frames[sn] = (col, cur)
            if cur and moved(cur, cams[sn]["prev"]) > a.still:
                still = False
            cams[sn]["prev"] = cur
        seen = {sn: frames[sn][1] for sn in SNS}
        n_all = len(set().union(*[set(s) for s in seen.values()]))
        if still and n_all >= a.min_tags:
            T = block_in_base(seen)
            novel = T is not None and all(np.degrees(np.linalg.norm(log_so3(P[:3, :3].T @ T[:3, :3]))) >= 15
                                          or np.linalg.norm(T[:3, 3] - P[:3, 3]) >= 25 for P in poses)
            if novel:
                burst = {sn: [frames[sn][1]] for sn in SNS}
                first = {sn: frames[sn][0] for sn in SNS}
                for _ in range(a.burst - 1):
                    for sn in SNS:
                        burst[sn].append(grab(cams[sn])[1])
                row, ok = dict(view=len(poses), t=time.time(), T_PB_nominal=np.round(T, 4).tolist(), cams={}), True
                for sn in SNS:
                    ids = sorted(set().union(*[set(d) for d in burst[sn]]))
                    mean, sd, n = {}, {}, {}
                    for i in ids:
                        c = np.array([d[i] for d in burst[sn] if i in d])
                        if len(c) < max(2, a.burst // 2):
                            continue
                        mean[i], sd[i], n[i] = c.mean(0), float(c.std(0).max()), len(c)
                        if sd[i] > 0.35:
                            ok = False
                    row["cams"][sn] = dict(ids=sorted(mean), corners={str(i): np.round(mean[i], 3).tolist() for i in mean},
                                           sd_px={str(i): round(sd[i], 3) for i in sd}, n={str(i): n[i] for i in n})
                if not ok:
                    note = "moved during the burst: not saved"
                else:
                    k = len(poses)
                    poses.append(T)
                    for sn in SNS:
                        cv2.imwrite(str(out / f"{sn}_{k:03d}.png"), first[sn])
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    sets = {sn: set(row["cams"][sn]["ids"]) for sn in SNS}
                    for i in IDS:
                        either[i] += any(i in s for s in sets.values())
                        both[i] += all(i in s for s in sets.values())
                    for p in PAIRS:
                        pairs[p] += any(p[0] in s and p[1] in s for s in sets.values())
                    flash = time.time()
                    note = f"view {k} saved"
                    print(f"{time.strftime('%H:%M:%S')} view {k}: " + ", ".join(f"{sn} {sorted(sets[sn])}" for sn in SNS)
                          + f"; block at {np.round(T[:3, 3], 0).tolist()} mm", flush=True)
                    (out / "status.json").write_text(json.dumps(dict(views=len(poses), both=both, either=either,
                                                                     pairs={f"{p[0]}-{p[1]}": v for p, v in pairs.items()})))
        bar = np.full((90, 2 * PW, 3), 255, np.uint8)
        cv2.putText(bar, f"saved {len(poses)}/{a.cap}   per tag, views seen by both cameras / by either:  "
                    + "  ".join(f"{i}: {both[i]}/{either[i]}" for i in IDS), (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2)
        cv2.putText(bar, "neighbours seen together:  " + "  ".join(f"{p[0]}-{p[1]}: {v}" for p, v in pairs.items())
                    + f"     {note}", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imshow(WIN, np.vstack([np.hstack([panel(sn, *frames[sn]) for sn in SNS]), bar]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    for cam in cams.values():
        cam["pipe"].stop()
    cv2.destroyAllWindows()
    print(f"{time.strftime('%H:%M:%S')} stopped: {len(poses)} views; both {both}; pairs "
          + str({f'{p[0]}-{p[1]}': v for p, v in pairs.items()}), flush=True)
    import os
    os._exit(0)
