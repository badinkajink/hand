#!/usr/bin/env python3
"""Candidate opener films for the SR2 part of the hand-design talk (owner, 2026-11-10 night: "a cool video of the hand
doing something at the very beginning of the sr2 hand section ... a real video, or a video of the whole hand on the ur5?
... ill take candidates that you select"). Writes docs/experiments/20261110-codesign_formulation/media/:

  20261110-opener_timelapse.mp4   the SR2 hand on the UR5, 5 November: three gantry moves to new layouts with finger
                                  sweeps between them, from the lab camera's frames (one every 1.94 s), cropped to the hand
  20261110-opener_eight_hands.mp4 the September bench array: eight real_v1 layouts in a held turn, slowed to 0.5x
  20261110-opener_cad_sweep.mp4   the CAD SR2 hand stepping through measured poses of the crosstalk layouts, the hand only

Sources: logs/20261105-sr2_timelapse_720/ (HHMMSS_index.jpg, 12:14 to 16:14, copied off corvus on 11-10; not in git),
docs/experiments/20260921-bench_video_array/20260921-eight_hands_held_turn_2x4_1920.mp4 and
docs/experiments/20261104-sr2_hand_crosstalk/media/20261105-film_cad_sweep.mp4.

    .venv/bin/python scripts/docs/codesign_opener_film.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/experiments/20261110-codesign_formulation/media"
FRAMES = ROOT / "logs/20261105-sr2_timelapse_720"
# (from, to) as HHMMSS: the gantry moves of the crosstalk chain (runner events of chain5, corvus clock) and a few minutes
# of finger sweeps after the first two; 14:20-14:33, when people walk past the camera, is left out
SEGMENTS = [("141130", "141320"), ("143619", "143930"), ("150220", "150410"), ("154144", "154430"), ("161210", "161422")]
CROP = (0.38, 0.0, 0.96, 0.75)        # fractions of the 1280 x 720 frame: the hand and the gantries; the operator at the left is cut


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def timelapse(fps=20):
    names = sorted(p.name for p in FRAMES.glob("*.jpg"))
    pick = [n for a, b in SEGMENTS for n in names if a <= n[:6] <= b]
    lst = OUT / "_opener_frames.txt"
    lst.write_text("".join(f"file '{FRAMES / n}'\nduration {1 / fps:.4f}\n" for n in pick))
    x0, y0, x1, y1 = CROP
    out = OUT / "20261110-opener_timelapse.mp4"
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(lst), "-vf",
           f"crop=iw*{x1 - x0:.3f}:ih*{y1 - y0:.3f}:iw*{x0:.3f}:ih*{y0:.3f},scale=800:-2,fps={fps}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "slow", "-movflags", "+faststart", str(out))
    lst.unlink()
    return out, len(pick)


def eight_hands():
    src = ROOT / "docs/experiments/20260921-bench_video_array/20260921-eight_hands_held_turn_2x4_1920.mp4"
    out = OUT / "20261110-opener_eight_hands.mp4"
    ffmpeg("-i", str(src), "-vf", "setpts=2.0*PTS,scale=1440:-2", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "23", "-preset", "slow", "-movflags", "+faststart", str(out))
    return out


def cad_sweep():
    src = ROOT / "docs/experiments/20261104-sr2_hand_crosstalk/media/20261105-film_cad_sweep.mp4"
    out = OUT / "20261110-opener_cad_sweep.mp4"
    ffmpeg("-i", str(src), "-vf", "crop=iw*0.64:ih:0:0,setpts=0.5*PTS", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "23", "-preset", "slow", "-movflags", "+faststart", str(out))
    return out


def main():
    out, n = timelapse()
    for f in (out, eight_hands(), cad_sweep()):
        d = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)],
                           capture_output=True, text=True).stdout.strip()
        ffmpeg("-ss", str(0.45 * float(d)), "-i", str(f), "-frames:v", "1", "-q:v", "3", str(f.with_suffix(".jpg")))
        print(f.name, f"{float(d):.1f} s", f"{f.stat().st_size / 1e6:.1f} MB", flush=True)
    print("timelapse frames:", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
