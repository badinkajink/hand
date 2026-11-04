#!/usr/bin/env python3
"""Horizontal shift (px) of the thumb finger between two RealSense frames.

    python3 track_shift.py <ref.png> <img.png> [<img.png> ...]
    TRACK_REGION=r0:r1,c0:c1 TRACK_MAX=15 python3 track_shift.py ...   (2-D search)

With TRACK_REGION set, searches x AND y shifts (the camera got bumped
2026-11-04 and J1's finger now sits at the right edge, rows 156-479, cols 534-639).

Brute-force 1-D search: the shift that best aligns the finger region's
horizontal gradients (robust to the lighting drifting between frames). Positive
= moved right (toward J1's motor / home wall in the current camera view).
"""

import os
import sys

import numpy as np
from PIL import Image

# Finger body below the J1 carriage, in the 640x480 colour frame.
REGION = (slice(200, 420), slice(170, 460))  # rows, cols
MAX_SHIFT = 60


def grad(path):
    a = np.asarray(Image.open(path).convert("L"), np.float32)
    gx = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    return gx


def shift2d(ref, img, rows, cols, m):
    r = ref[rows, cols]
    best = None
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            c = img[slice(rows.start + dy, rows.stop + dy), slice(cols.start + dx, cols.stop + dx)]
            if c.shape != r.shape:
                continue
            score = float(np.mean(np.abs(c - r)))
            if best is None or score < best[0]:
                best = (score, dx, dy)
    return best


def shift(ref, img):
    r = ref[REGION]
    rows, cols = REGION
    best, best_s = None, 0
    for s in range(-MAX_SHIFT, MAX_SHIFT + 1):
        c = img[rows, slice(cols.start + s, cols.stop + s)]
        if c.shape != r.shape:
            continue
        score = float(np.mean(np.abs(c - r)))
        if best is None or score < best:
            best, best_s = score, s
    return best_s, best


def main():
    ref = grad(sys.argv[1])
    env = os.environ.get("TRACK_REGION")
    if env:
        (r0, r1), (c0, c1) = (map(int, part.split(":")) for part in env.split(","))
        m = int(os.environ.get("TRACK_MAX", 15))
        for p in sys.argv[2:]:
            score, dx, dy = shift2d(ref, grad(p), slice(r0, r1), slice(c0, c1), m)
            print(f"{p.rsplit('/', 1)[-1]}: shift dx {dx:+d} dy {dy:+d} px (residual {score:.1f})")
        return
    for p in sys.argv[2:]:
        s, score = shift(ref, grad(p))
        print(f"{p.rsplit('/', 1)[-1]}: shift {s:+d} px (residual {score:.1f})")


if __name__ == "__main__":
    main()
