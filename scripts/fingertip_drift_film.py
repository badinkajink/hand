#!/usr/bin/env python3
"""Film of the contact viewer through the board's warm-up and two touches, with the settings of 2026-10-20 above and those
of 2026-10-21 below: the recording of 2026-10-21 09:59 (data/20261021-095907_touches_records.npz, written by
scripts/fingertip_contact_estimation_study.py thermal), replayed five times faster than recorded by two viewers in turn,
each captured by headless Chrome (scripts/fingertip_contact_film.py) and stacked under a label bar.

    ~/miniconda3/bin/python scripts/fingertip_drift_film.py
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_contact_film as FILM  # noqa: E402

D = ROOT / "docs/experiments/20261020-fingertip_contact_estimation"
REC = D / "data/20261021-095907_touches_records.npz"
OUT = D / "media/20261021-viewer_drift_replay.mp4"
VIEWER = ROOT / "src/morphohand/driver/reskin5x/host/wrench_viewer.py"
PY = Path.home() / "miniconda3/bin/python"
SPEED, SECONDS, FPS, SIZE, WIDTH = 5.0, 300.0, 10, (1400, 880), 1100
POSTER_S = 245.0                                           # recording time of the poster frame: after the second touch
RUNS = (("Settings of 2026-10-20: no temperature correction, no noise floor, no re-zero",
         ["--no-thermal", "--floor", "0", "--min-force", "0", "--auto-zero", "0"]),
        ("Settings of 2026-10-21: temperature correction, 2 uT noise floor, shown above 0.25 N, re-zero after 3 s at rest",
         []))


def label(text, width, path, h=46):
    im = Image.new("RGB", (width, h), "white")
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype("DejaVuSerif.ttf", 24)
    dr.text((12, (h - 28) // 2), text, fill="black", font=font)
    dr.line([(0, h - 1), (width, h - 1)], fill="black", width=1)
    im.save(path)


def run(args, port, frames):
    """Capture one viewer; returns (frames, recording time of the first frame in s)."""
    p = subprocess.Popen([str(PY), "-u", str(VIEWER), "--replay", str(REC), "--replay-seconds", str(SECONDS), "--speed",
                          str(SPEED), "--http", str(port), "--no-log", *args], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        t_open = None
        for line in p.stdout:                               # the replay starts just before the server opens
            if line.startswith("open http") and t_open is None:
                t_open = time.time()
            if line.startswith("calibrated"):
                break
        else:
            raise RuntimeError("the viewer exited before it calibrated")
        first = []
        n = asyncio.run(FILM.capture(f"http://127.0.0.1:{port}", (SECONDS - 60.0) / SPEED, FPS, SIZE, frames, 1.5,
                                     on_start=first.append))
        return n, (first[0] - t_open) * SPEED
    finally:
        p.terminate()
        p.wait(10)


def main():
    work = Path(tempfile.mkdtemp(prefix="drift_film_"))
    dirs, starts = [], []
    for k, (text, args) in enumerate(RUNS):
        fr = work / f"run{k}"
        fr.mkdir()
        n, rec0 = run(args, 8792 + k, fr)
        print(f"{text}: {n} frames from {rec0:.1f} s of the recording", flush=True)
        label(text, SIZE[0], work / f"label{k}.png")
        dirs.append(fr)
        starts.append(rec0)
    t_common = max(starts)                                  # both films from the same moment of the recording
    for d, s0 in zip(dirs, starts):
        drop = int(round((t_common - s0) / SPEED * FPS))
        files = sorted(d.glob("f*.png"))
        for f in files[:drop]:
            f.unlink()
        for j, f in enumerate(files[drop:]):
            f.rename(d / f"g{j:05d}.png")
        for f in sorted(d.glob("g*.png")):
            f.rename(d / ("f" + f.name[1:]))
    n = min(len(list(d.glob("f*.png"))) for d in dirs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    inputs = []
    for k, d in enumerate(dirs):
        inputs += ["-loop", "1", "-framerate", str(FPS), "-i", str(work / f"label{k}.png"),
                   "-framerate", str(FPS), "-i", str(d / "f%05d.png")]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex",
                    f"[0][1][2][3]vstack=inputs=4:shortest=1,scale={WIDTH}:-2", "-frames:v", str(n), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "26", str(OUT)], check=True)
    i = min(n - 1, int(round((POSTER_S - t_common) / SPEED * FPS)))
    parts = [Image.open(work / f"label{k}.png") for k in range(len(dirs))]
    shots = [Image.open(d / f"f{i:05d}.png").convert("RGB") for d in dirs]
    W = SIZE[0]
    H = sum(p.height for p in parts) + sum(s.height for s in shots)
    poster = Image.new("RGB", (W, H), "white")
    y = 0
    for lab, shot in zip(parts, shots):
        poster.paste(lab, (0, y))
        y += lab.height
        poster.paste(shot, (0, y))
        y += shot.height
    poster = poster.resize((WIDTH, round(H * WIDTH / W)), Image.LANCZOS)
    poster.save(OUT.with_suffix(".png"))
    shutil.rmtree(work)
    print(f"{n} frames from {t_common:.1f} s of the recording -> {OUT.relative_to(ROOT)} "
          f"({OUT.stat().st_size / 1e6:.1f} MB), poster at {POSTER_S:.0f} s")


if __name__ == "__main__":
    main()
