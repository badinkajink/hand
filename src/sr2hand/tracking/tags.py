"""AprilTag cells and tag maps.

A tag's cells come from OpenCV's AprilTag dictionaries (tag_bits.json, written once with cv2.aruco.generateImageMarker);
1 is a black cell, rows run top to bottom as the tag is seen from the front, and the black border is included. The tag
size is the edge of the black square (8 cells for tag36h11, 7 for tag25h9), the size pupil_apriltags' pose takes.

Corner order. pupil_apriltags returns the corners of a tag drawn from these cells as right-top, left-top, left-bottom and
right-bottom of the drawn image, for every id of both families (checked on 2026-11-05 for ids 0-31). A tag map lists
each tag's four corners in that order, so a detection's corners pair with the map's without reordering.

The tag block is a hexagonal prism below the object (plan, "Objects and tag blocks"). Its frame B has its origin at the
prism's centre, z_B along the axis toward the object and face k's outward normal at k x 60 deg from x_B. Each face
bears one tag, upright when z_B points up: the tag's top edge is on the object's side.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BITS = json.loads((Path(__file__).with_name("tag_bits.json")).read_text())


def cells(family: str, tag_id: int) -> np.ndarray:
    """(n, n) array of a tag's cells, 1 black, row 0 at the top as seen from the front."""
    rows = BITS["families"][family]["tags"][str(tag_id)]
    return np.array([[int(c) for c in r] for r in rows], dtype=np.uint8)


def n_cells(family: str) -> int:
    return int(BITS["families"][family]["cells"])


def tag_corners(centre, right, down, size) -> np.ndarray:
    """(4, 3) corners of a tag of edge `size` whose drawn image has its right and down directions along the unit
    vectors right and down, in the detector's order (right-top, left-top, left-bottom, right-bottom)."""
    c, r, d = (np.asarray(v, float) for v in (centre, right, down))
    h = size / 2.0
    return np.stack([c + h * r - h * d, c - h * r - h * d, c - h * r + h * d, c + h * r + h * d])


def black_rects(family: str, tag_id: int):
    """The black cells as rectangles (row, col0, col1): runs of black cells in each row, for drawing a tag from boxes."""
    b = cells(family, tag_id)
    out = []
    for i, row in enumerate(b):
        j = 0
        while j < len(row):
            if row[j]:
                k = j
                while k + 1 < len(row) and row[k + 1]:
                    k += 1
                out.append((i, j, k))
                j = k + 1
            else:
                j += 1
    return out


@dataclass
class TagMap:
    """The tags of one rigid body: id -> (4, 3) corners in the body's frame (mm), and per tag its centre, drawing axes
    (right, down) and outward normal, which the renderers and the visibility check use."""
    family: str
    size: float
    corners: dict = field(default_factory=dict)
    frames: dict = field(default_factory=dict)          # id -> (centre, right, down, normal)

    def add(self, tag_id, centre, right, down):
        right, down = np.asarray(right, float), np.asarray(down, float)
        normal = np.cross(down, right)                   # out of the face, toward the viewer
        self.frames[int(tag_id)] = (np.asarray(centre, float), right, down, normal)
        self.corners[int(tag_id)] = tag_corners(centre, right, down, self.size)

    def ids(self):
        return sorted(self.corners)

    def stacked(self, ids=None):
        ids = self.ids() if ids is None else ids
        return np.concatenate([self.corners[i] for i in ids])

    def transformed(self, T) -> "TagMap":
        """The same tags in another frame: T maps this map's frame into it (4 x 4, mm). The corners are moved as they are,
        so a refined map keeps each tag's own size and shape."""
        R, t = T[:3, :3], T[:3, 3]
        out = TagMap(self.family, self.size)
        for i, (c, r, d, n) in self.frames.items():
            out.frames[i] = (R @ c + t, R @ r, R @ d, R @ n)
            out.corners[i] = self.corners[i] @ R.T + t
        return out

    def to_json(self):
        return {"family": self.family, "size_mm": self.size,
                "corners_mm": {str(i): np.round(c, 5).tolist() for i, c in self.corners.items()}}

    @staticmethod
    def from_corners(family, size, corners: dict) -> "TagMap":
        """A map from corners alone (a bundle adjustment's result): each tag's frame is refitted from its corners."""
        out = TagMap(family, size)
        for i, C in corners.items():
            C = np.asarray(C, float)
            c = C.mean(0)
            r = (C[0] - C[1] + C[3] - C[2]) / 2.0
            d = (C[2] - C[1] + C[3] - C[0]) / 2.0
            out.frames[int(i)] = (c, r / np.linalg.norm(r), d / np.linalg.norm(d), np.cross(d, r) / np.linalg.norm(np.cross(d, r)))
            out.corners[int(i)] = C
        return out


@dataclass
class HexBlock:
    """The printed hexagonal tag block (mm): across flats, height, tag edge, family and the ids of faces 0-5."""
    across_flats: float = 30.0
    height: float = 17.0
    tag_size: float = 13.0
    family: str = "tag36h11"
    ids: tuple = (10, 11, 12, 13, 14, 15)

    @property
    def face_width(self):
        return self.across_flats / math.sqrt(3.0)

    @property
    def across_corners(self):
        return 2.0 * self.face_width

    def vertices(self):
        """(12, 3) prism vertices in B: the bottom hexagon (z = -height/2), then the top."""
        r = self.face_width                                       # circumradius
        ang = np.radians(30.0 + 60.0 * np.arange(6))            # corners between the faces
        ring = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros(6)], 1)
        lo, hi = ring.copy(), ring.copy()
        lo[:, 2], hi[:, 2] = -self.height / 2, self.height / 2
        return np.concatenate([lo, hi])

    def tag_map(self) -> TagMap:
        """Tag corners in B. Face k: normal n_k at 60 k deg from x_B, the tag's right z_B x n_k and its down -z_B."""
        border = (self.face_width - self.tag_size) / 2.0
        cell = self.tag_size / n_cells(self.family)
        if border < cell or (self.height - self.tag_size) / 2.0 < cell:
            raise ValueError(f"the white border is narrower than a cell ({border:.2f} mm, cell {cell:.2f} mm)")
        tm = TagMap(self.family, self.tag_size)
        z = np.array([0.0, 0.0, 1.0])
        for k, tid in enumerate(self.ids):
            a = math.radians(60.0 * k)
            n = np.array([math.cos(a), math.sin(a), 0.0])
            tm.add(tid, self.across_flats / 2.0 * n, np.cross(z, n), -z)
        return tm


def plate_map(family, size, ids, centres, normal=(0.0, 0.0, 1.0), right=(1.0, 0.0, 0.0)) -> TagMap:
    """Tags on a flat plate facing along `normal`: each upright (its down along right x normal), centres in the plate's
    frame (mm)."""
    n, r = np.asarray(normal, float), np.asarray(right, float)
    d = np.cross(r, n)
    tm = TagMap(family, size)
    for tid, c in zip(ids, centres):
        tm.add(tid, c, r, d)
    return tm
