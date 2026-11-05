#!/usr/bin/env python3
"""Scratch: identify hand solids 000 and 005 with the owner's STL exports of 2026-08-30 (copied from ~/Downloads into
downloads_copy/): try the 24 axis-aligned rotations, align the bounding boxes, and measure the surface distance of
20,000 samples of the STL to 200,000 samples of the solid. Loads meshes: run under resguard (--mem 3G).

Results of this code on 2026-11-04 (run inline before the resguard gate closed at 17:3x):
  d435_plate.stl       vs solid 000: best rotation 180 deg about z, translation (-111.0, 142.875, -4.0) mm
                       (= the solid's STEP placement), surface distance max 0.458 mm, mean 0.126 mm
  static_tag_mount.stl vs solid 005: best rotation +90 deg about z, translation (6.5, 117.0, -7.0) mm
                       (= the solid's STEP placement), surface distance max 0.348 mm, mean 0.090 mm
"""
import itertools
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
EX = HERE.parent / "extract/hand"


def rotations():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                R[i, p] = s
            if np.linalg.det(R) > 0:
                yield R


def match(stl, solid):
    a = trimesh.load(HERE / "downloads_copy" / stl, process=True)
    b = trimesh.load(EX / f"{solid}.stl", process=True)
    tree = cKDTree(b.sample(200000))
    best = None
    for R in rotations():
        v = a.vertices @ R.T
        t = b.bounds[0] - v.min(axis=0)
        s = trimesh.sample.sample_surface(trimesh.Trimesh(v + t, a.faces), 20000)[0]
        d, _ = tree.query(s)
        if best is None or d.mean() < best[0]:
            best = (d.mean(), d.max(), R, t)
    print(f"{stl} vs solid {solid}: mean {best[0]:.3f} mm, max {best[1]:.3f} mm\nR =\n{best[2]}\nt = {best[3]}")


if __name__ == "__main__":
    match("d435_plate.stl", "000")
    match("static_tag_mount.stl", "005")
