"""One film of a session's recordings (owner, 2026-11-16: "we should also be recording the re-centering and the
pre-grasp keyframe ramp. it should be one contiguous, unbroken stream of motion, just sped up between the controls").
The segments of a folder for one camera, or both side by side, in time order: the trials (the controls) at --trial-x
and every other step (ready's recentre and pre-grasp ramp, putdown, layout's reconfiguration, zero, open, release,
place, approach) at --fast. Each segment is re-timed by its measured frame rate (its _frames.csv: the recorder writes at
a nominal 30 fps while the tracker delivers 22-27), loses its first --head-s seconds (the cameras' start), and carries
a caption with its step, layout and speed.

    /usr/bin/python3 film_session.py DIR [--from 20261116-121000] [--to 20261116-123000] [--camera 836612071918 | --both]
        [--fast 6] [--trial-x 1] [--label STAMP=TEXT ...] [--list] [--out FILM.mp4]

Names: hom_setup.py's --video-dir writes <stamp>_<step>_<plan stem>_<serial>.mp4 and hom_trial.py's
<stamp>_trial_<label>_<serial>.mp4; the earlier <stamp>_<task>_video_<serial>.mp4 (a trial) and
<stamp>_reconfig_<plan stem>_<serial>.mp4 (a layout) are read too.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

NAME = re.compile(r"^(\d{8}-\d{6})_(.+)_(\d{12})\.mp4$")
LAYOUT = re.compile(r"x(\d+(?:\.\d+)?)_y(\d+(?:\.\d+)?)")
TASK = re.compile(r"(twist|press|roll|pinpush)")
STEP_TEXT = {"ready": "recentre and pre-grasp", "putdown": "put-down", "layout": "reconfiguration",
             "zero": "fingers straight", "open": "fingers open", "release": "release", "place": "placement",
             "approach": "approach"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SERIALS = ("832412070344", "836612071918")


def segments(d, sn, t_from=None, t_to=None):
    """The folder's segments for camera sn in time order: {stamp, step, rest, file, csv}."""
    out = []
    for f in sorted(Path(d).glob(f"*_{sn}.mp4")):
        m = NAME.match(f.name)
        if not m:
            continue
        stamp, mid, _ = m.groups()
        if (t_from and stamp < t_from) or (t_to and stamp > t_to):
            continue
        if mid.endswith("_video"):                         # hom_trial.py's name before --video-dir
            step, rest = "trial", mid[:-len("_video")]
        else:
            step, _, rest = mid.partition("_")
        step = "layout" if step == "reconfig" else step
        c = f.with_name(f.stem + "_frames.csv")
        out.append({"stamp": stamp, "step": step, "rest": rest, "file": f, "csv": c if c.exists() else None})
    return out


def timing(seg):
    """(measured fps, real seconds) from the camera times of the segment's _frames.csv."""
    if seg["csv"] is None:
        return None, None
    with open(seg["csv"]) as fh:
        t = [float(r["t_camera_ms"]) for r in csv.DictReader(fh)]
    if len(t) < 2:
        return None, None
    span = (t[-1] - t[0]) / 1000.0
    # the median frame interval: a camera gap at a clip's start (1.7 s after frames 31-33 in 125922 and 130318) would
    # otherwise enter the rate and play the clip at 0.75-0.88 of real time (hand-4a, 2026-11-16 15:45)
    return 1000.0 / float(np.median(np.diff(t))), span


def nominal_fps(f):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                        "-of", "json", str(f)], capture_output=True, text=True, check=True)
    num, den = json.loads(r.stdout)["streams"][0]["r_frame_rate"].split("/")
    return float(num) / float(den)


def caption(seg, speed, labels):
    if seg["stamp"] in labels:
        return labels[seg["stamp"]]
    lay = LAYOUT.search(seg["rest"])
    at = f" at ({lay.group(1)}, {lay.group(2)}) mm" if lay else ""
    pace = "real time" if speed == 1 else f"{speed:g}x"
    if seg["step"] == "trial":
        task = TASK.search(seg["rest"])
        return f"control: HOM {task.group(1) if task else 'trial'}{at}, {pace}"
    if seg["step"] == "layout":
        return f"reconfiguration{' to' + at[3:] if lay else ''}, {pace}"
    return f"{STEP_TEXT.get(seg['step'], seg['step'])}{at}, {pace}"


def motion(seg):
    """Each frame's local change from the one before: the third-largest mean absolute change of the 16 x 9 blocks of
    the grey frame at 160 x 90, smoothed over 9 frames. A still scene reads 0.42-0.47, the gantries moving 0.6-3.4, the
    arm 0.8-3 (camera 832412070344, 2026-11-16); the whole frame's mean diluted the hand's motion below the noise."""
    if "motion" not in seg:
        import cv2
        cap, prev, d = cv2.VideoCapture(str(seg["file"])), None, [0.0]
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA).astype(float)
            if prev is not None:
                b = np.abs(g - prev).reshape(9, 10, 16, 10).mean(axis=(1, 3))
                d.append(float(np.sort(b.ravel())[-3]))         # the third-largest of 16 x 9 blocks: local motion
            prev = g
        d = np.asarray(d)
        seg["motion"] = np.convolve(d, np.ones(9) / 9, mode="same") if len(d) > 9 else d
    return seg["motion"]


def keep_ranges(seg, thr, head_s, trial, pad_s=0.15, min_idle_s=0.5):
    """Frame ranges [(n0, n1)] of the segment to show (owner, 2026-11-16: "we can cut the extreme dead time where
    nothing moves so that the 6x is reasonable"): every run of still frames (motion below thr) longer than min_idle_s
    real seconds is cut but for pad_s at each end; a trial (shown in real time) loses only its still head and tail."""
    m = motion(seg)
    fps = timing(seg)[0] or nominal_fps(seg["file"])
    n_head = int(round(head_s * fps))
    moving = m > thr
    moving[:n_head] = False
    idx = np.flatnonzero(moving)
    if len(idx) == 0:
        return [(n_head, min(len(m) - 1, n_head + int(fps)))]            # a still segment: one second of it
    pad, gap = int(round(pad_s * fps)), int(round(min_idle_s * fps))
    if trial:
        return [(max(n_head, idx[0] - 3 * pad), min(len(m) - 1, idx[-1] + 3 * pad))]
    out, a, b = [], idx[0], idx[0]
    for n in idx[1:]:
        if n - b > gap:
            out.append((a, b))
            a = n
        b = n
    out.append((a, b))
    return [(max(n_head, a - pad), min(len(m) - 1, b + pad)) for a, b in out]


def chain(k, seg, speed, head_s, ranges=None):
    """The filter for input k: drop head_s real seconds (or keep only the frame ranges), re-time to real time over
    speed, 30 fps, 960 x 540."""
    fps, _ = timing(seg)
    nom = nominal_fps(seg["file"])
    fps = fps or nom
    f = nom / fps / speed
    if ranges:
        sel = "+".join(f"between(n\\,{a}\\,{b})" for a, b in ranges)
        return (f"[{k}:v]select='{sel}',setpts=N/({nom:g}*TB),setpts=PTS*{f:.5f},fps=30,"
                f"scale=960:540,format=yuv420p")
    return (f"[{k}:v]trim=start={head_s * fps / nom:.3f},setpts=PTS-STARTPTS,setpts=PTS*{f:.5f},fps=30,"
            f"scale=960:540,format=yuv420p")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--from", dest="t_from", default=None, help="the first stamp (YYYYmmdd-HHMMSS) to include")
    ap.add_argument("--to", dest="t_to", default=None, help="the last stamp to include")
    ap.add_argument("--camera", default="836612071918", choices=SERIALS)
    ap.add_argument("--both", action="store_true", help="both cameras side by side (1920 x 540)")
    ap.add_argument("--fast", type=float, default=6.0, help="the speed of every step but the trials")
    ap.add_argument("--trial-x", type=float, default=1.0, help="the speed of the trials")
    ap.add_argument("--head-s", type=float, default=0.5, help="real seconds dropped from each segment's start")
    ap.add_argument("--label", action="append", default=[], help="STAMP=TEXT: a segment's caption")
    ap.add_argument("--speed", action="append", default=[], help="STAMP=X: that segment's speed")
    ap.add_argument("--trim-idle", action="store_true",
                    help="cut the still stretches (motion under --idle-k x the camera's noise floor)")
    ap.add_argument("--idle-k", type=float, default=1.25)
    ap.add_argument("--blur", action="append", default=[],
                    help="STAMP=X:Y:W:H: smear that rectangle (pixels of the 960 x 540 frame) in the segment, as for a "
                         "person in view (people are blurred in published frames)")
    ap.add_argument("--blur-all", default=None, help="X:Y:W:H smeared in every segment (a side where people pass)")
    ap.add_argument("--blur-cam", action="append", default=[],
                    help="SERIAL=X:Y:W:H: smeared in every segment of that camera (with --both too)")
    ap.add_argument("--control", action="append", default=[],
                    help="STAMP: a segment that, like a trial, keeps its still stretches under --trim-idle")
    ap.add_argument("--clock", action="store_true", help="each caption starts with the segment's clock time")
    ap.add_argument("--list", action="store_true", help="list the segments only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    labels = dict(x.split("=", 1) for x in a.label)
    speeds = {k: float(v) for k, v in (x.split("=", 1) for x in a.speed)}
    blurs = {}
    if a.blur_all:
        bx, by, bw, bh = (int(t) for t in a.blur_all.split(":"))
        blurs["*"] = [f"delogo=x={max(1, bx)}:y={max(1, by)}:w={min(bw, 958 - max(1, bx))}:h={min(bh, 538 - max(1, by))}"]
    for x in a.blur:
        k, v = x.split("=", 1)
        bx, by, bw, bh = (int(t) for t in v.split(":"))
        blurs.setdefault(k, []).append(f"delogo=x={max(1, bx)}:y={max(1, by)}:w={min(bw, 958 - max(1, bx))}:"
                                       f"h={min(bh, 538 - max(1, by))}")
    cam_blur = {}
    for x in a.blur_cam:
        k, v = x.split("=", 1)
        bx, by, bw, bh = (int(t) for t in v.split(":"))
        cam_blur.setdefault(k, []).append(f"delogo=x={max(1, bx)}:y={max(1, by)}:w={min(bw, 958 - max(1, bx))}:"
                                          f"h={min(bh, 538 - max(1, by))}")

    def cb(serial):
        return "".join(f",{b}" for b in cam_blur.get(serial, []))
    segs = segments(a.dir, a.camera, a.t_from, a.t_to)
    if a.both:
        other = {s["stamp"] + s["step"]: s for s in segments(a.dir, [s for s in SERIALS if s != a.camera][0],
                                                              a.t_from, a.t_to)}
        segs = [dict(s, pair=other.get(s["stamp"] + s["step"])) for s in segs]
    if not segs:
        sys.exit(f"no segments of camera {a.camera} in {a.dir}")
    total = 0.0
    thr = None
    if a.trim_idle:
        floor = float(np.percentile(np.concatenate([motion(s)[5:] for s in segs]), 10))
        thr = a.idle_k * floor
        print(f"still below {thr:.3f} (the noise floor {floor:.3f} x {a.idle_k:g})")
    for s in segs:
        speed = speeds.get(s["stamp"], a.trial_x if s["step"] == "trial" else a.fast)
        fps, span = timing(s)
        s["speed"] = speed
        out_s = None if span is None else max(0.0, span - a.head_s) / speed
        if thr is not None:
            s["ranges"] = keep_ranges(s, thr, a.head_s, s["step"] == "trial" or s["stamp"] in a.control)
            n = sum(b - a_ + 1 for a_, b in s["ranges"])
            out_s = n / (fps or nominal_fps(s["file"])) / speed
        total += out_s or 0.0
        print(f"{s['stamp']} {s['step']:8s} {s['rest'][:48]:48s} "
              + (f"{span:6.1f} s at {fps:4.1f} fps -> {out_s:5.1f} s at {speed:g}x" if span else "no _frames.csv")
              + ("" if not a.both or s.get("pair") else "  (no second camera)") + f"  [{caption(s, speed, labels)}]")
    print(f"film: {total:.1f} s")
    if a.list:
        return 0
    out = Path(a.out) if a.out else Path(a.dir) / f"{segs[0]['stamp']}_session_film_{'both' if a.both else a.camera}.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, s in enumerate(segs):
            cap = Path(tmp) / f"cap{i}.txt"
            cap.write_text((f"{s['stamp'][9:11]}:{s['stamp'][11:13]}  " if a.clock else "")
                           + caption(s, s["speed"], labels))
            text = (f"drawtext=fontfile={FONT}:textfile={cap}:x=20:y=18:fontsize=26:fontcolor=white:box=1:"
                    "boxcolor=black@0.55:boxborderw=8")
            part = Path(tmp) / f"part{i:03d}.mp4"
            if a.both and s.get("pair"):
                inputs = ["-i", str(s["file"]), "-i", str(s["pair"]["file"])]
                other_sn = [x for x in SERIALS if x != a.camera][0]
                fc = (f"{chain(0, s, s['speed'], a.head_s, s.get('ranges'))}{cb(a.camera)}[a];"
                      f"{chain(1, s['pair'], s['speed'], a.head_s, s.get('ranges'))}{cb(other_sn)}[b];"
                      f"[a][b]hstack=inputs=2:shortest=1,{text}[v]")
            else:
                inputs = ["-i", str(s["file"])]
                bl = "".join(f"{b}," for b in blurs.get(s["stamp"], []) + blurs.get("*", []) + cam_blur.get(a.camera, []))
                fc = (f"{chain(0, s, s['speed'], a.head_s, s.get('ranges'))},{bl}{'pad=1920:540:0:0,' if a.both else ''}"
                      f"{text}[v]")
            subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", fc, "-map", "[v]", "-an",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-r", "30", str(part)], check=True)
            parts.append(part)
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        str(out)], check=True)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                       capture_output=True, text=True)
    print(f"-> {out} ({float(r.stdout.strip()):.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
