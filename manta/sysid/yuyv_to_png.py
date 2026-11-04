#!/usr/bin/env python3
"""Convert raw YUYV frames from v4l2_grab.py to PNGs.
    python3 yuyv_to_png.py <in_dir> <width> <height> [out_dir]
"""

import pathlib
import sys

import numpy as np
from PIL import Image


def yuyv_to_rgb(raw: bytes, w: int, h: int) -> np.ndarray:
    a = np.frombuffer(raw, np.uint8)[: w * h * 2].reshape(h, w // 2, 4).astype(np.float32)
    y = np.empty((h, w), np.float32)
    y[:, 0::2], y[:, 1::2] = a[..., 0], a[..., 2]
    u = np.repeat(a[..., 1], 2, axis=1) - 128
    v = np.repeat(a[..., 3], 2, axis=1) - 128
    rgb = np.stack([y + 1.402 * v, y - 0.344 * u - 0.714 * v, y + 1.772 * u], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def main():
    src = pathlib.Path(sys.argv[1])
    w, h = int(sys.argv[2]), int(sys.argv[3])
    dst = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else src
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.yuyv")):
        Image.fromarray(yuyv_to_rgb(f.read_bytes(), w, h)).save(dst / f"{f.stem}.png")


if __name__ == "__main__":
    main()
