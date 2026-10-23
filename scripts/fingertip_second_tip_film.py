#!/usr/bin/env python3
"""Film of the contact viewer replaying the second fingertip's six centre presses of 2026-10-23 12:27 (the recording
data/20261023-122717_round2_records.npz, written by scripts/fingertip_second_tip_transfer.py --save-records). The viewer
calibrates on the recording's first 2 s at rest, as it does live, so it finds the pad's own poles and strength; headless
Chrome captures it (scripts/fingertip_contact_film.py) and the frames from --from-s to --to-s of the recording go under a
label bar.

    ~/miniconda3/bin/python scripts/fingertip_second_tip_film.py
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_contact_film as FILM  # noqa: E402

D = ROOT / "docs/experiments/20261023-second_fingertip_transfer"
REC = D / "data/20261023-122717_round2_records.npz"
OUT = D / "media/20261023-viewer_centre_presses.mp4"
VIEWER = ROOT / "src/morphohand/driver/reskin5x/host/wrench_viewer.py"
PY = Path.home() / "miniconda3/bin/python"
FPS, SIZE, WIDTH = 10, (1400, 880), 1100
LABEL = "Second fingertip, six presses at the face centre, replayed through the viewer on its own rest fit"


def label(text, width, path, h=46):
    im = Image.new("RGB", (width, h), "white")
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype("DejaVuSerif.ttf", 24)
    dr.text((12, (h - 28) // 2), text, fill="black", font=font)
    dr.line([(0, h - 1), (width, h - 1)], fill="black", width=1)
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rec", default=str(REC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--from-s", type=float, default=21.0, help="recording time of the film's first frame (s)")
    ap.add_argument("--to-s", type=float, default=58.0, help="recording time of its last frame (s)")
    ap.add_argument("--poster-s", type=float, default=24.0, help="recording time of the poster frame (s)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8795)
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="second_tip_film_"))
    frames = work / "run"
    frames.mkdir()
    p = subprocess.Popen([str(PY), "-u", str(VIEWER), "--replay", a.rec, "--speed", str(a.speed), "--http", str(a.port),
                          "--no-log"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        t_open = t_first = None
        for line in p.stdout:                               # the replay starts just before the viewer's first line
            t_first = t_first or time.time()
            if line.startswith("open http") and t_open is None:
                t_open = time.time()
            if line.startswith("calibrated"):
                print(line.strip(), flush=True)
                break
        else:
            raise RuntimeError("the viewer exited before it calibrated")
        t_open = t_open or t_first                          # the calibration can report before the server line
        first = []
        wall = (a.to_s + 0.5) / a.speed - (time.time() - t_open)
        n = asyncio.run(FILM.capture(f"http://127.0.0.1:{a.port}", wall, FPS, SIZE, frames, 1.5, on_start=first.append))
        rec0 = (first[0] - t_open) * a.speed
    finally:
        p.terminate()
        p.wait(10)
    files = sorted(frames.glob("f*.png"))
    rec_t = [rec0 + k * a.speed / FPS for k in range(len(files))]
    keep = [f for f, rt in zip(files, rec_t) if a.from_s <= rt <= a.to_s]
    print(f"{n} frames captured from {rec0:.1f} s of the recording; {len(keep)} kept ({a.from_s:.0f} to {a.to_s:.0f} s)",
          flush=True)
    for j, f in enumerate(keep):
        f.rename(frames / f"g{j:05d}.png")
    label(LABEL, SIZE[0], work / "label.png")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", str(FPS), "-i", str(work / "label.png"),
                    "-framerate", str(FPS), "-i", str(frames / "g%05d.png"), "-filter_complex",
                    f"[0][1]vstack=inputs=2:shortest=1,scale={WIDTH}:-2", "-frames:v", str(len(keep)), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "26", str(out)], check=True)
    i = min(len(keep) - 1, max(0, int(round((a.poster_s - a.from_s) / a.speed * FPS))))
    lab, shot = Image.open(work / "label.png"), Image.open(frames / f"g{i:05d}.png").convert("RGB")
    poster = Image.new("RGB", (SIZE[0], lab.height + shot.height), "white")
    poster.paste(lab, (0, 0))
    poster.paste(shot, (0, lab.height))
    poster = poster.resize((WIDTH, round(poster.height * WIDTH / SIZE[0])), Image.LANCZOS)
    poster.save(out.with_suffix(".png"))
    shutil.rmtree(work)
    print(f"{len(keep)} frames -> {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB), poster at {a.poster_s:.0f} s")


if __name__ == "__main__":
    main()
