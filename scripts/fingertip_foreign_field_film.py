#!/usr/bin/env python3
"""Film of the contact viewer replaying the owner's six touches of steel on the second fingertip's face (2026-10-23
15:43:17-15:43:42, from data/20261023-153800_touch_records.npz of 20261023-foreign_field_rejection): first with the test
for fields from outside the pad off, then with it on (--foreign-gate, key f). Each run calibrates on the clip's first
2 s at rest, as the viewer does live; headless Chrome captures it (scripts/fingertip_contact_film.py), and the frames
from --from-s to --to-s of the clip go under a label bar. The poster stacks the two runs at --poster-s.

    ~/miniconda3/bin/python scripts/fingertip_foreign_field_film.py
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

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_contact_film as FILM  # noqa: E402
from fingertip_second_tip_film import label  # noqa: E402

D = ROOT / "docs/experiments/20261023-foreign_field_rejection"
REC = D / "data/20261023-153800_touch_records.npz"
OUT = D / "media/20261023-viewer_steel_touches.mp4"
VIEWER = ROOT / "src/morphohand/driver/reskin5x/host/wrench_viewer.py"
PY = Path.home() / "miniconda3/bin/python"
FPS, SIZE, WIDTH = 10, (1400, 880), 1100
CLIP = ("15:43:00.5", "15:43:46.5")                    # 4.5 s at rest, a field change of 50 uT, the six touches
LABELS = {False: "Test off: six touches of steel at the face centre, replayed through the viewer",
          True: "Test on (key f): the same six touches"}


def clock(s, day):
    h, m, sec = s.split(":")
    lt = time.localtime(day)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, int(h), int(m), 0, 0, 0, -1)) + float(sec)


def run(clip, gate, a, work):
    """Captured frames of one replay and the clip time of each."""
    frames = work / ("on" if gate else "off")
    frames.mkdir()
    cmd = [str(PY), "-u", str(VIEWER), "--replay", str(clip), "--http", str(a.port), "--no-log"]
    p = subprocess.Popen(cmd + (["--foreign-gate"] if gate else []), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
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
        t_open = t_open or t_first
        first = []
        wall = a.to_s + 0.5 - (time.time() - t_open)
        n = asyncio.run(FILM.capture(f"http://127.0.0.1:{a.port}", wall, FPS, SIZE, frames, 1.5, on_start=first.append))
        rec0 = first[0] - t_open
    finally:
        p.terminate()
        p.wait(10)
    files = sorted(frames.glob("f*.png"))
    print(f"{'on' if gate else 'off'}: {n} frames from {rec0:.1f} s of the clip", flush=True)
    return files, [rec0 + k / FPS for k in range(len(files))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--from-s", type=float, default=15.5, help="clip time of each run's first frame (s)")
    ap.add_argument("--to-s", type=float, default=45.5, help="clip time of its last frame (s)")
    ap.add_argument("--poster-s", type=float, default=32.3, help="clip time of the poster (s): the fourth touch")
    ap.add_argument("--port", type=int, default=8797)
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="foreign_field_film_"))
    z = np.load(REC)
    t = z["t"]
    day = float(t[0])
    sel = (t >= clock(CLIP[0], day)) & (t <= clock(CLIP[1], day))
    clip = work / "clip.npz"
    np.savez(clip, t=t[sel], r=z["r"][sel])
    seq, poster = [], []
    for gate in (False, True):
        files, rec_t = run(clip, gate, a, work)
        keep = [f for f, rt in zip(files, rec_t) if a.from_s <= rt <= a.to_s]
        label(LABELS[gate], SIZE[0], work / f"label_{gate}.png")
        lab = Image.open(work / f"label_{gate}.png")
        for f in keep:
            im = Image.new("RGB", (SIZE[0], lab.height + SIZE[1]), "white")
            im.paste(lab, (0, 0))
            im.paste(Image.open(f).convert("RGB"), (0, lab.height))
            seq.append(im)
        kept_t = [rt for rt in rec_t if a.from_s <= rt <= a.to_s]
        j = int(np.argmin([abs(rt - a.poster_s) for rt in kept_t]))
        poster.append(seq[len(seq) - len(keep) + j])
    for k, im in enumerate(seq):
        im.save(work / f"g{k:05d}.png")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(work / "g%05d.png"),
                    "-vf", f"scale={WIDTH}:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", str(out)], check=True)
    W, H = poster[0].size
    both = Image.new("RGB", (W, 2 * H), "white")
    both.paste(poster[0], (0, 0))
    both.paste(poster[1], (0, H))
    both = both.resize((WIDTH, round(2 * H * WIDTH / W)), Image.LANCZOS)
    both.save(out.with_suffix(".png"))
    shutil.rmtree(work)
    print(f"{len(seq)} frames -> {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB); poster at {a.poster_s:.1f} s")


if __name__ == "__main__":
    main()
