"""Record one calibration pose of the UR5 with the ChArUco board on the SR2 hand. Nothing is sent to the arm: its
joints and TCP pose are read by read-only RTDE before and after, and the pose is marked moved if a joint changed by
0.01 deg or more.

Each D435 records, at session settings fixed by SETTINGS (auto exposure and auto white balance off; white paper
below saturation, checked on 11-13):
- with the projector on, FRAMES framesets of colour 1920 x 1080, depth 1280 x 720 as streamed (z16), and depth aligned
  to colour by librealsense (through the factory colour model, which has no distortion);
- with the projector off, after DROP frames, FRAMES left infrared images (IR1, 1280 x 720, in the depth stream's
  frame), whose ChArUco corners give the board's pose without the colour lens or the depth-to-colour extrinsic.
It saves each stream's factory intrinsics, the depth-to-colour and IR1-to-depth extrinsics, the depth scale, the
options read back and every frame's supported metadata.

The board is the tag-block session's paper board (assets/tracking/20261113-paper_board/, page 1): 7 x 7 squares,
nominally 15.0 mm with 10.5 mm markers, DICT_5X5_100 (DICT_5X5_50 decodes markers 0-23 the same). The measured square
is set in the calibration; corner ids and pixels do not depend on it. Markers are detected, then recovered from the
board's layout by refineDetectedMarkers, which brings IR1 from 16-21 to 24 markers.

    /usr/bin/python3 record_pose.py OUTDIR [--label TEXT] [--frames 15]
Writes OUTDIR/pNN/ and appends one line to OUTDIR/poses.jsonl.
"""
import argparse
import os
import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

UR5 = "192.168.0.4"
A = cv2.aruco
DICT = A.getPredefinedDictionary(A.DICT_5X5_50)
BOARD = A.CharucoBoard_create(7, 7, 15.0, 10.5, DICT)
SETTINGS = {"RGB Camera": {"enable_auto_exposure": 0, "enable_auto_white_balance": 0, "exposure": 166, "gain": 64,
                           "white_balance": 4600},
            "Stereo Module": {"enable_auto_exposure": 0, "exposure": 7000, "gain": 16, "laser_power": 150}}
DROP = 10

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--label", default="")
ap.add_argument("--frames", type=int, default=15)
a = ap.parse_args()
out = Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)
log = out / "poses.jsonl"
k = sum(1 for _ in open(log)) if log.exists() else 0
pdir = out / f"p{k:02d}"
pdir.mkdir()


def retry(fn, *args):
    """An option read or write right after a stream starts can fail with 'Device or resource busy'; try again."""
    for i in range(10):
        try:
            return fn(*args)
        except RuntimeError:
            if i == 9:
                raise
            time.sleep(0.2)


def arm():
    import rtde_receive
    r = rtde_receive.RTDEReceiveInterface(UR5)
    s = dict(t=time.time(), q=list(r.getActualQ()), tcp=list(r.getActualTCPPose()), mode=r.getRobotMode(),
             safety=r.getSafetyMode())
    r.disconnect()
    return s


def charuco(gray):
    c, ids, rej = A.detectMarkers(gray, DICT, parameters=A.DetectorParameters_create())
    if ids is None:
        return [], [], []
    c, ids, _, _ = A.refineDetectedMarkers(gray, BOARD, c, ids, rej)
    n, cc, ci = A.interpolateCornersCharuco(c, ids, gray, BOARD)
    if not n:
        return ids.ravel().tolist(), [], []
    return ids.ravel().tolist(), cc.reshape(-1, 2), ci.ravel().tolist()


def intr(p):
    i = p.as_video_stream_profile().get_intrinsics()
    return dict(w=i.width, h=i.height, fx=i.fx, fy=i.fy, cx=i.ppx, cy=i.ppy, model=str(i.model), coeffs=list(i.coeffs))


def extr(e):
    return dict(R=np.array(e.rotation).reshape(3, 3).T.tolist(), t=list(e.translation))   # x_to = R x_from + t (m)


def metadata(f):
    m = {}
    for v in rs.frame_metadata_value.__members__.values():
        if f.supports_frame_metadata(v):
            m[str(v).split(".")[-1]] = f.get_frame_metadata(v)
    return dict(t_ms=f.get_timestamp(), domain=str(f.get_frame_timestamp_domain()).split(".")[-1],
                n=f.get_frame_number(), meta=m)


class Track:
    """Per-corner pixels over a burst."""
    def __init__(self):
        self.c, self.markers = {}, []

    def add(self, gray):
        mids, cc, ci = charuco(gray)
        self.markers.append(len(mids))
        for j, xy in zip(ci, cc):
            self.c.setdefault(int(j), []).append(np.asarray(xy, float))

    def summary(self, n):
        keep = {j: np.array(v) for j, v in self.c.items() if len(v) >= n // 2}
        ids = sorted(keep)
        return dict(corner_ids=ids, corners_mean=[keep[j].mean(0).tolist() for j in ids],
                    corners_sd=[float(np.linalg.norm(keep[j].std(0))) for j in ids],
                    corners_seen=[len(keep[j]) for j in ids], markers_per_frame=self.markers)


from pupil_apriltags import Detector  # noqa: E402

TAGS = Detector(families="tag36h11", quad_decimate=1.0, refine_edges=1)   # one detector: a new one per call crashed


def tags(gray):
    return [dict(id=int(t.tag_id), corners=t.corners.tolist(), margin=float(t.decision_margin))
            for t in TAGS.detect(np.ascontiguousarray(gray))]


arm0 = arm()
cams = {}
for dev in rs.context().devices:
    sn = dev.get_info(rs.camera_info.serial_number)
    p, c = rs.pipeline(), rs.config()
    c.enable_device(sn)
    c.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
    c.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    c.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 15)
    prof = p.start(c)
    d = prof.get_device()
    opts = {}
    for s in d.query_sensors():
        name = s.get_info(rs.camera_info.name)
        for o, v in SETTINGS.get(name, {}).items():
            retry(s.set_option, getattr(rs.option, o), v)
        opts[name] = {o: retry(s.get_option, getattr(rs.option, o)) for o in
                      ["enable_auto_exposure", "exposure", "gain", "enable_auto_white_balance", "white_balance",
                       "emitter_enabled", "laser_power"] if s.supports(getattr(rs.option, o))}
    retry(d.first_depth_sensor().set_option, rs.option.emitter_enabled, 1)
    sc, sd, si = (prof.get_stream(rs.stream.color), prof.get_stream(rs.stream.depth),
                  prof.get_stream(rs.stream.infrared, 1))
    cams[sn] = dict(pipe=p, dev=d, align=rs.align(rs.stream.color), streams=dict(
        firmware=d.get_info(rs.camera_info.firmware_version), options=opts,
        depth_scale=d.first_depth_sensor().get_depth_scale(),
        color=intr(sc), depth=intr(sd), ir1=intr(si),
        depth_to_color=extr(sd.get_extrinsics_to(sc)), ir1_to_depth=extr(si.get_extrinsics_to(sd))))
    (pdir / f"{sn}_streams.json").write_text(json.dumps(cams[sn]["streams"], indent=1))
for _ in range(30):
    for cam in cams.values():
        cam["pipe"].wait_for_frames()

rec = dict(pose=k, label=a.label, arm_before=arm0, settings=SETTINGS, cams={})
for sn, cam in cams.items():
    p = cam["pipe"]
    col_t, ir_t = Track(), Track()
    raw, aligned, meta_c, meta_d, meta_i = [], [], [], [], []
    for i in range(a.frames):
        fs = p.wait_for_frames()
        raw.append(np.asanyarray(fs.get_depth_frame().get_data()).copy())
        meta_d.append(metadata(fs.get_depth_frame()))
        al = cam["align"].process(fs)
        aligned.append(np.asanyarray(al.get_depth_frame().get_data()).copy())
        cf = fs.get_color_frame()
        meta_c.append(metadata(cf))
        col = np.asanyarray(cf.get_data()).copy()
        gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
        col_t.add(gray)
        if i == 0:
            cv2.imwrite(str(pdir / f"{sn}_color.png"), col)
            tags_c = tags(gray)
        cv2.imwrite(str(pdir / f"{sn}_color_{i:02d}.jpg"), col, [cv2.IMWRITE_JPEG_QUALITY, 95])
    retry(cam["dev"].first_depth_sensor().set_option, rs.option.emitter_enabled, 0)
    for _ in range(DROP):
        p.wait_for_frames()
    irs = []
    for i in range(a.frames):
        fs = p.wait_for_frames()
        f = fs.get_infrared_frame(1)
        meta_i.append(metadata(f))
        irs.append(np.asanyarray(f.get_data()).copy())
        ir_t.add(irs[-1])
    retry(cam["dev"].first_depth_sensor().set_option, rs.option.emitter_enabled, 1)
    p.stop()
    cv2.imwrite(str(pdir / f"{sn}_ir1_off.png"), irs[0])
    np.savez_compressed(pdir / f"{sn}_ir1_off.npz", ir1=np.stack(irs))
    np.savez_compressed(pdir / f"{sn}_depth_raw.npz", z16=np.stack(raw))
    al = np.stack(aligned).astype(np.float32)
    al[al == 0] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore")
        np.save(pdir / f"{sn}_depth_aligned_m.npy",
                (np.nanmedian(al, 0) * cam["streams"]["depth_scale"]).astype(np.float32))
    rec["cams"][sn] = dict(streams=cam["streams"], color=dict(col_t.summary(a.frames), tags=tags_c, frames=meta_c),
                           ir1_off=dict(ir_t.summary(a.frames), tags=tags(irs[0]), frames=meta_i), depth_frames=meta_d)
rec["arm_after"] = arm()
dq = np.degrees(np.abs(np.subtract(rec["arm_after"]["q"], arm0["q"]))).max()
rec["still"] = bool(dq < 0.01)
rec["dq_max_deg"] = float(dq)
with open(log, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(f"pose {k} {a.label!r}: arm {'still' if rec['still'] else f'MOVED {dq:.3f} deg'}, "
      f"q = {np.round(np.degrees(arm0['q']), 2).tolist()}")
for sn, c in rec["cams"].items():
    for s in ("color", "ir1_off"):
        x = c[s]
        xy = np.array(x["corners_mean"]).reshape(-1, 2)
        box = f"x {xy[:, 0].min():.0f}-{xy[:, 0].max():.0f}, y {xy[:, 1].min():.0f}-{xy[:, 1].max():.0f}" if len(xy) else "-"
        sd = np.median(x["corners_sd"]) if x["corners_sd"] else float("nan")
        print(f"  {sn} {s:7s}: {len(x['corner_ids'])}/36 corners ({box}), sd {sd:.3f} px, "
              f"markers {min(x['markers_per_frame'])}-{max(x['markers_per_frame'])}, tags {[t['id'] for t in x['tags']]}")
sys.stdout.flush()
os._exit(0)   # the interpreter's teardown of librealsense and pupil_apriltags dumped core after a complete record
