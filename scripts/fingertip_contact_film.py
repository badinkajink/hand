"""Film of the browser contact viewer (src/morphohand/driver/reskin5x/host/wrench_viewer.py): headless Chrome opens the
viewer's page and the DevTools protocol captures it at a fixed rate; ffmpeg encodes the frames.

    ~/miniconda3/bin/python scripts/fingertip_contact_film.py --url http://127.0.0.1:8791 --seconds 12 --fps 10 \
        --out docs/experiments/20261020-fingertip_contact_estimation/media/20261020-viewer_replay.mp4
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets


async def capture(url, seconds, fps, size, frames_dir, warmup):
    port = 9333
    prof = tempfile.mkdtemp(prefix="chrome_film_")
    chrome = subprocess.Popen(["google-chrome", "--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                               "--hide-scrollbars", f"--window-size={size[0]},{size[1]}", f"--remote-debugging-port={port}",
                               f"--user-data-dir={prof}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list").read())
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=2 ** 26) as ws:
            n = 0

            async def call(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps(dict(id=n, method=method, params=params)))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("id") == n:
                        return m.get("result", {})

            await call("Page.enable")
            await call("Page.navigate", url=url)
            await asyncio.sleep(warmup)
            t0, k = time.time(), 0
            while time.time() - t0 < seconds:
                r = await call("Page.captureScreenshot", format="png")
                (frames_dir / f"f{k:05d}.png").write_bytes(base64.b64decode(r["data"]))
                k += 1
                await asyncio.sleep(max(0.0, t0 + k / fps - time.time()))
            return k
    finally:
        chrome.terminate()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8791")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--size", type=int, nargs=2, default=(1400, 880))
    ap.add_argument("--warmup", type=float, default=6.0, help="seconds to let the page load and connect")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frames = Path(tempfile.mkdtemp(prefix="film_frames_"))
    k = asyncio.run(capture(a.url, a.seconds, a.fps, a.size, frames, a.warmup))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(frames / "f%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                    str(out)], check=True)
    poster = out.with_suffix(".png")
    best = max(frames.glob("f*.png"), key=lambda p: p.stat().st_size)      # the busiest frame as the poster
    poster.write_bytes(best.read_bytes())
    print(f"{k} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB), poster {poster}")


if __name__ == "__main__":
    main()
