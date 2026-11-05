#!/usr/bin/env python3
"""A short cut of the lab's overhead time-lapse of 11-02 to 11-03 (lab/timelapse_d435.py's frames, one every 2 s, made
into lab/20261102-tipcal-timelapse_60fps.mp4: 35,401 frames, 9.8 min). Only the lit stretches are kept (the frame's mean
brightness over 60 of 255; the night frames read about 26 and show nothing); every STRIDE-th frame at FPS, with the clock
and the run at each frame in the corner, and a card for the dark stretch between them.

Clock: each frame's file name on corvus carries its time (HHMMSS_index.jpg; the film's frame list,
/dev/shm/20261102-timelapse_list.txt). The camera ran twice, 3,737 frames from 13:05:37 to 15:06:25 and, after its stall,
31,664 frames from 15:13:37 to 08:13:35, one every 1.933 s; within each run the time is linear in the frame to about a
second (frame 6,003 of the second run is named 18:27:00, and the line gives 18:27:00). The runs shown are the lab's
(QUEUE R31, lab NOTES): seat 1 from 14:50:10, seat 2 from 18:27:10, seat 3 from 22:03:37; the night's obliques on seat 2
from 04:41 and on seat 3 from 06:36.

    python3 scripts/ur5/ur5_timelapse_short.py [--stride 16] [--fps 60] [--out OUT]
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
SRC = LAB / "20261102-tipcal-timelapse_60fps.mp4"
OUT = LAB / "20261102-tipcal-timelapse_short.mp4"
W, H, N = 640, 360, 35401
DAY = dt.datetime(2026, 11, 2)
RUN1 = (DAY.replace(hour=13, minute=5, second=37), DAY.replace(hour=15, minute=6, second=25), 3737)   # first, last, n
RUN2 = (DAY.replace(hour=15, minute=13, second=37), DAY.replace(hour=8, minute=13, second=35) + dt.timedelta(days=1), 31664)
K_GAP = RUN1[2]                                                   # the first frame after the camera's stall
DT = (RUN2[1] - RUN2[0]).total_seconds() / (RUN2[2] - 1)
RUNS = [(DAY.replace(hour=13, minute=5), "set-up: homing by hand, new plans"),
        (DAY.replace(hour=14, minute=50, second=10), "seat 1 (v3 ID 6), 5 mm sphere"),
        (DAY.replace(hour=18, minute=27, second=10), "seat 2 (v3 ID 5), 5 mm sphere"),
        (DAY.replace(hour=22, minute=3, second=37), "seat 3 (v3 ID 4), 5 mm sphere"),
        (DAY.replace(hour=1, minute=43) + dt.timedelta(days=1), "press speeds on seats 3, 2 and 1"),
        (DAY.replace(hour=2, minute=47) + dt.timedelta(days=1), "night obliques, seat 1 (v3 ID 6)"),
        (DAY.replace(hour=4, minute=41) + dt.timedelta(days=1), "night obliques, seat 2 (v3 ID 5)"),
        (DAY.replace(hour=6, minute=36) + dt.timedelta(days=1), "night obliques, seat 3 (v3 ID 4)")]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def when(n):
    r, k = (RUN1, n) if n < K_GAP else (RUN2, n - K_GAP)
    return r[0] + (r[1] - r[0]) * (k / (r[2] - 1))


def what(t):
    return [lab for t0, lab in RUNS if t0 <= t][-1]


def brightness():
    """Mean brightness (0 to 255) of every frame, from a 64 x 36 grey decode."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(SRC), "-vf", "scale=64:36,format=gray",
                          "-f", "rawvideo", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 36 * 64).mean(axis=1)


def lit_runs(y, thr=60.0, least=300):
    m = np.r_[False, y > thr, False]
    d = np.flatnonzero(np.diff(m.astype(int)))
    return [(int(a), int(b)) for a, b in zip(d[::2], d[1::2]) if b - a >= least]


def caption(img, lines, font, small):
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((8, 8, 8 + 12 + max(d.textlength(s, font=font if i == 0 else small) for i, s in enumerate(lines)),
                 8 + 30 + 22 * (len(lines) - 1)), fill=(0, 0, 0, 150))
    for i, s in enumerate(lines):
        d.text((14, 11 + (0 if i == 0 else 26 + 22 * (i - 1))), s, font=font if i == 0 else small, fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    y = brightness()
    if len(y) != N:
        raise SystemExit(f"{SRC.name}: {len(y)} frames, not {N}")
    runs = lit_runs(y)
    picks = [n for s, e in runs for n in range(s, e, a.stride)]
    sel = "+".join(f"between(n\\,{s}\\,{e - 1})*not(mod(n-{s}\\,{a.stride}))" for s, e in runs)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(SRC), "-vf",
                            f"select='{sel}'", "-vsync", "0",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-threads", "4", "-crf", "20", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    font, small, big = (ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 16), ImageFont.truetype(FONT, 22))
    speed = a.stride * DT * a.fps / 60.0
    k, prev_end, written = 0, None, 0
    for s, e in runs:
        if prev_end is not None:                                   # the dark stretch: a card for one second
            card = Image.new("RGB", (W, H), (12, 12, 12))
            d = ImageDraw.Draw(card)
            t0, t1 = when(prev_end), when(s)
            for j, line in enumerate((f"Lights out from {t0:%H:%M} to {t1:%H:%M}.",
                                      "The runs went on in the dark.")):
                d.text((W / 2, H / 2 - 16 + 32 * j), line, font=big, fill=(235, 235, 235), anchor="mm")
            for _ in range(a.fps):
                enc.stdin.write(card.tobytes())
                written += 1
        for n in range(s, e, a.stride):
            buf = dec.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                raise SystemExit(f"the decoder ended at pick {k} of {len(picks)}")
            t = when(n)
            img = Image.frombytes("RGB", (W, H), buf)
            caption(img, [f"{t:%m-%d %H:%M}", what(t), f"1 s of film = {speed:.0f} min"], font, small)
            enc.stdin.write(img.tobytes())
            k += 1
            written += 1
        prev_end = e
    dec.stdout.close()
    dec.wait()
    enc.stdin.close()
    enc.wait()
    print(f"lit stretches (frames): {runs} = {[(f'{when(s):%H:%M}', f'{when(e - 1):%H:%M}') for s, e in runs]}; "
          f"{k} frames and {written - k} card frames at {a.fps} fps = {written / a.fps:.1f} s; frame period {DT:.4f} s, "
          f"the stall after frame {K_GAP - 1} -> {a.out}")


if __name__ == "__main__":
    main()
