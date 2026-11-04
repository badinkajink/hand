"""A short cut of a lab time-lapse (timelapse_d435.py's frames, named HHMMSS_index.jpg, one every PERIOD s), read
straight from the frames' folder: every STRIDE-th lit frame at FPS, each with its clock (from its file name), the run
then on the robot (from RUNS) and the film's speed in the corner. A one-second card stands for each dark stretch (mean
brightness at most 60 of 255 over at least 300 frames: the lab's lights out) and for each gap of over a minute between
frames (the camera off). The same look as scripts/ur5_timelapse_short.py, which cut the film of 11-02 from its mp4.

RUNS is a JSON list of ["HH:MM[:SS]", "label"] in time order, each label shown from its time on (times before the
previous one fall on the next day). Run it at idle priority beside a robot run (it decodes JPEGs and encodes x264).

    nice -n 19 ionice -c3 /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/timelapse_short.py \
        FRAMES OUT --day 2026-11-03 --runs RUNS.json [--stride 16] [--fps 60] [--gap-note TEXT]
"""
import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 640, 360
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def clock(names, day):
    """Each frame's time from its name, a day added at each wrap past midnight."""
    out, add, prev = [], dt.timedelta(0), None
    for s in names:
        t = dt.datetime.combine(day, dt.datetime.strptime(s[:6], "%H%M%S").time()) + add
        if prev is not None and t < prev - dt.timedelta(hours=12):
            add += dt.timedelta(days=1)
            t += dt.timedelta(days=1)
        out.append(t)
        prev = t
    return out


def runs_from(path, day):
    rows, prev = [], None
    for hm, label in json.loads(Path(path).read_text()):
        f = "%H:%M:%S" if hm.count(":") == 2 else "%H:%M"
        t = dt.datetime.combine(day, dt.datetime.strptime(hm, f).time())
        while prev is not None and t < prev:
            t += dt.timedelta(days=1)
        rows.append((t, label))
        prev = t
    return rows


def lum(path):
    im = Image.open(path)
    im.draft("L", (W // 8, H // 8))
    return float(np.asarray(im.convert("L"), float).mean())


def caption(img, lines, font, small):
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((8, 8, 8 + 12 + max(d.textlength(s, font=font if i == 0 else small) for i, s in enumerate(lines)),
                 8 + 30 + 22 * (len(lines) - 1)), fill=(0, 0, 0, 150))
    for i, s in enumerate(lines):
        d.text((14, 11 + (0 if i == 0 else 26 + 22 * (i - 1))), s, font=font if i == 0 else small, fill=(255, 255, 255))
    return img


def card(lines, big):
    im = Image.new("RGB", (W, H), (12, 12, 12))
    d = ImageDraw.Draw(im)
    for j, line in enumerate(lines):
        d.text((W / 2, H / 2 - 16 * (len(lines) - 1) + 32 * j), line, font=big, fill=(235, 235, 235), anchor="mm")
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--day", required=True, help="YYYY-MM-DD of the first frame")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--gap-note", default="", help="a second line on the cards of the camera's gaps")
    a = ap.parse_args()
    day = dt.date.fromisoformat(a.day)
    files = sorted(Path(a.frames).glob("*.jpg"))
    ts = clock([f.name for f in files], day)
    order = np.argsort(np.array([t.timestamp() for t in ts]), kind="stable")
    files, ts = [files[k] for k in order], [ts[k] for k in order]
    sec = np.diff([t.timestamp() for t in ts])
    period = float(np.median(sec))
    runs = runs_from(a.runs, day)

    def what(t):
        hit = [lab for t0, lab in runs if t0 <= t]
        return hit[-1] if hit else ""

    # the camera's runs: frames less than a minute apart
    cuts = [0] + [k + 1 for k in np.flatnonzero(sec > 60.0)] + [len(files)]
    font, small, big = ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 16), ImageFont.truetype(FONT, 22)
    speed = a.stride * period * a.fps / 60.0
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-threads", "2", "-crf", "20",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    shown, cards, dark_from = 0, [], None
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        if c0 > 0:                                                  # the camera's gap
            lines = [f"No frames from {ts[c0 - 1]:%H:%M} to {ts[c0]:%H:%M}."] + ([a.gap_note] if a.gap_note else [])
            im = card(lines, big)
            for _ in range(a.fps):
                enc.stdin.write(im.tobytes())
            cards.append(lines[0])
        for k in range(c0, c1, a.stride):
            y = lum(files[k])
            if y <= 60.0:                                           # dark: held until the lights come back
                dark_from = dark_from or ts[k]
                continue
            if dark_from is not None and (ts[k] - dark_from).total_seconds() >= 300 * period:
                lines = [f"Lights out from {dark_from:%H:%M} to {ts[k]:%H:%M}.", "The runs went on in the dark."]
                im = card(lines, big)
                for _ in range(a.fps):
                    enc.stdin.write(im.tobytes())
                cards.append(lines[0])
            dark_from = None
            img = Image.open(files[k]).convert("RGB")
            if img.size != (W, H):
                img = img.resize((W, H))
            caption(img, [f"{ts[k]:%m-%d %H:%M}", what(ts[k]), f"1 s of film = {speed:.0f} min"], font, small)
            enc.stdin.write(img.tobytes())
            shown += 1
    enc.stdin.close()
    enc.wait()
    n_all = shown + a.fps * len(cards)
    print(f"{len(files)} frames {ts[0]:%m-%d %H:%M:%S} to {ts[-1]:%m-%d %H:%M:%S}, one every {period:.3f} s (median); "
          f"{shown} shown (every {a.stride}th lit one) and {len(cards)} cards {cards} at {a.fps} fps = "
          f"{n_all / a.fps:.1f} s -> {a.out}")


if __name__ == "__main__":
    main()
