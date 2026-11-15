#!/usr/bin/env python3
"""AprilTag stickers for the back of the DIP link b v4, to track the fingers' gantries with the two D435s of the HOM
trials (owner, 2026-11-15: "please design a pdf of apriltags to print on the back of the dip link. we have a nice flat
surface there"; the thumb's tag is seen directly, the index's and middle's with the MCP and PIP flexed about 90 deg and
the thumb out of the way).

The face is the tongue's back, behind the fingertip board: 21.0 mm across the finger by 21.41 mm along it, between a
1.45 mm step at the PIP end and the top cover's end plate, which is flush with it at the tip end. Four holes of 1.9 mm
(the M2 pilot holes) pass through it at 8.0 mm either side of the centre line and 8.0 mm either side of the holes'
mid-plane (z_L -44.26 and -28.26 mm). Measured on assets/mjcf/experimental/20261030-cad_v2/meshes/v4b_dip_link.stl,
the print STL moved into frame L.

Frame L is that mesh's frame, the frame of each finger's <finger>_pip_frame body in the cad_v3 model
(assets/mjcf/experimental/20261104-cad_v3/ur5_full_k0.xml): origin on the PIP axis, y_L along it, x_L dorsal, z_L from
the tip toward the PIP axis (the tip's end at z_L = -46.91 mm). It is frame D of assets/cad_v2/README.md turned 180 deg
about the PIP axis (x_L = -x_D, y_L = y_D, z_L = -z_D). The thumb's DIP link mesh sits in its pip_frame turned 180 deg
about z (quat 0 0 0 1); the index's and middle's without a turn.

Each sticker is a tag36h11 tag with a 13.0 mm black square (the size of the tag blocks' tags) and a white border of one
cell, centred in a cut outline 0.25 mm inside the face. A 14 mm square would put its corners 0.46 mm from the holes'
edges; at 13 mm they are 1.17 mm away, and the holes lie under the outer corners of the white border. Thumb 24, index
25, middle 26: ids 0-4, 6 and 10-23 are on the test plate, the hand's tag plate and the objects' blocks, and
tracking/tag_bits.json has ids 0-31. Upright as seen from the back with the PIP joint up: the tag's right is +y_L, its
down -z_L, its face +x_L.

`build` writes the Letter PDF at actual size, a preview PNG and a JSON with each tag's corners in frame L (the sticker's
thickness, 0.10 mm, added to x_L); `check` rasterizes the sticker grid at CHECK_DPI and detects every tag (camera stack:
pupil_apriltags), comparing ids, black-square edges and centres with the layout.

    ~/.claude/bin/resguard.sh run --mem 2G -- ~/miniconda3/bin/python scripts/hardware/dip_tag_print.py build
    ~/.claude/bin/resguard.sh run --mem 2G -- ~/miniconda3/bin/python scripts/hardware/dip_tag_print.py check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for _paths
import _paths  # noqa: E402,F401

from sr2hand.tracking import tags as TG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATE = "20261115"
OUT = ROOT / f"assets/tracking/{DATE}-dip_link_tags"
PDF = OUT / f"{DATE}-dip_link_tags_letter.pdf"
PREVIEW = OUT / f"{DATE}-dip_link_tags_preview.png"
REPORT = OUT / f"{DATE}-dip_link_tags.json"
PAGE = (215.9, 279.4)                       # mm, US Letter
FAMILY = "tag36h11"
TAG = 13.0                                  # mm, the black square's edge
IDS = {"thumb": 24, "index": 25, "middle": 26}
COPIES = 4                                  # stickers per finger
FACE = (21.0, 21.41)                        # mm, the tongue's back: across the finger (y_L), along it (z_L)
FACE_Z = (-46.91, -25.5)                    # mm, z_L of the face's tip end and of the step at its PIP end
FACE_X = 8.0                                # mm, x_L of the face
STEP = 1.45                                 # mm, the step up to the face at x_L 9.45 on the PIP side
HOLES = [(y, z) for y in (-8.0, 8.0) for z in (-44.26, -28.26)]   # y_L, z_L (mm), radius 0.95
HOLE_R = 0.95
INSET = 0.25                                # mm, the cut outline inside the face's edges
STICKER_T = 0.10                            # mm, paper and glue
GAP = 9.0                                   # mm between neighbouring stickers
CHECK_DPI = 600
GREY = "#8c8c8c"


def face_centre():
    return 0.0, (FACE_Z[0] + FACE_Z[1]) / 2.0


def cut_size():
    return FACE[0] - 2 * INSET, FACE[1] - 2 * INSET


def tag_map(finger: str) -> TG.TagMap:
    """The finger's tag in frame L (mm): centre on the face's centre, the sticker's thickness dorsal of the face."""
    yc, zc = face_centre()
    tm = TG.TagMap(FAMILY, TAG)
    tm.add(IDS[finger], (FACE_X + STICKER_T, yc, zc), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
    return tm


def draw_tag(ax, tag_id, cx, cy, size, zorder=3):
    """The tag's black cells as one compound path, centred at (cx, cy) on a page whose y runs up."""
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    c = size / TG.n_cells(FAMILY)                                # cells across the black square, border included
    x0, y1 = cx - size / 2, cy + size / 2
    verts, codes = [], []
    for row, j0, j1 in TG.black_rects(FAMILY, tag_id):
        a, b = x0 + j0 * c, y1 - (row + 1) * c
        w = (j1 - j0 + 1) * c
        verts += [(a, b), (a + w, b), (a + w, b + c), (a, b + c), (a, b)]
        codes += [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc="black", ec="none", lw=0, zorder=zorder))


def sticker(ax, finger, cx, cy, scale=1.0, label=True):
    """One sticker centred at (cx, cy) mm: the cut outline, the tag and, in the margin at the PIP end, the finger, id and
    an arrow toward the PIP joint."""
    from matplotlib.patches import Rectangle, Polygon
    w, h = (v * scale for v in cut_size())
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec=GREY, lw=0.25, zorder=2))
    draw_tag(ax, IDS[finger], cx, cy, TAG * scale)
    if label:
        quiet = TAG * scale / 2 * (10.0 / 8.0)                  # the white border's outer edge from the centre
        ym = cy + (quiet + h / 2) / 2                             # midway between it and the cut line
        fs = 3.6 * scale
        ax.text(cx, ym, f"{finger} {IDS[finger]}", ha="center", va="center", fontsize=fs, color=GREY,
                family="serif", zorder=3)
        a = 0.55 * scale
        for sx in (-1, 1):                                        # beside the label, clear of the holes (|y_L| <= 5)
            x = cx + sx * 4.6 * scale
            ax.add_patch(Polygon([(x - a, ym - a * 0.8), (x + a, ym - a * 0.8), (x, ym + a * 0.9)], closed=True,
                                 fc=GREY, ec="none", zorder=3))


def grid_layout():
    """Centres (mm, page coordinates with y up) of the stickers: a column per finger, COPIES rows."""
    w, h = cut_size()
    fingers = list(IDS)
    gw = len(fingers) * w + (len(fingers) - 1) * GAP
    x0 = 18.0 + w / 2
    y_top = PAGE[1] - 40.0
    out = []
    for i, f in enumerate(fingers):
        for k in range(COPIES):
            out.append((f, x0 + i * (w + GAP), y_top - h / 2 - k * (h + GAP)))
    return out, gw


def placement(ax, ox, oy, s=3.0):
    """The face at scale s, seen from the back with the PIP joint up: outline, step, top cover, holes, the sticker."""
    from matplotlib.patches import Rectangle, Circle
    yc, zc = face_centre()
    W, H = FACE[0] * s, FACE[1] * s
    # the face (frame L: page x = y_L, page y = z_L), its lower-left corner at (ox, oy)
    ax.add_patch(Rectangle((ox, oy), W, H, fc="#f2f2f2", ec="k", lw=0.6, zorder=1))
    ax.add_patch(Rectangle((ox - 2.0, oy + H), W + 4.0, 5.0, fc="#d9d9d9", ec="k", lw=0.6, zorder=1))   # the step
    ax.add_patch(Rectangle((ox, oy - 1.6 * s), W, 1.6 * s, fc="#e6e6e6", ec="k", lw=0.4, ls=(0, (2, 1)), zorder=1))
    cx, cy = ox + W / 2, oy + (zc - FACE_Z[0]) * s
    sticker(ax, "index", cx, cy, scale=s, label=True)
    for y, z in HOLES:                                            # the holes under the sticker, dashed
        ax.add_patch(Circle((ox + (y + FACE[0] / 2) * s, oy + (z - FACE_Z[0]) * s), HOLE_R * s, fill=False,
                            ec="k", lw=0.4, ls=(0, (1.5, 1.0)), zorder=5))
    t = dict(fontsize=7.5, family="serif", zorder=6)
    ax.text(ox + W / 2, oy + H + 2.5, f"step up {STEP:.2f} mm toward the PIP joint", ha="center", va="center", **t)
    ax.text(ox + W / 2, oy - 0.8 * s, "top cover's end plate (flush)", ha="center", va="center", **t)
    ax.text(ox + W / 2, oy - 1.6 * s - 4.0, f"face {FACE[0]:.2f} × {FACE[1]:.2f} mm; holes {2 * HOLE_R:.1f} mm in diameter at "
            f"±8.0 mm; scale {s:g}:1", ha="center", va="top", **t)
    return W, H


def cmd_build(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(PAGE[0] / 25.4, PAGE[1] / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE[0])
    ax.set_ylim(0, PAGE[1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(PAGE[0] / 2, PAGE[1] - 16.0, "AprilTags for the back of the DIP link b v4", ha="center", va="center",
            fontsize=13, family="serif")
    ax.text(PAGE[0] / 2, PAGE[1] - 23.0, f"{FAMILY}, {TAG:.1f} mm black square; thumb {IDS['thumb']}, index "
            f"{IDS['index']}, middle {IDS['middle']}; cut outline {cut_size()[0]:.2f} × {cut_size()[1]:.2f} mm",
            ha="center", va="center", fontsize=9.5, family="serif")
    cells, gw = grid_layout()
    w, h = cut_size()
    for f, cx, cy in cells:
        sticker(ax, f, cx, cy)
    for f, cx, cy in cells[::COPIES]:
        ax.text(cx, cy + h / 2 + 4.0, f"{f} {IDS[f]}", ha="center", va="bottom", fontsize=10, family="serif")
    # the placement drawing to the right of the grid
    W, H = placement(ax, 18.0 + gw + 22.0, PAGE[1] - 40.0 - 3.0 * FACE[1] - 6.0, s=3.0)
    y = PAGE[1] - 40.0 - COPIES * h - (COPIES - 1) * GAP - 12.0
    lines = [
        "Print at 100 % (actual size, no fit to page) on matte paper. The bar below measures 100.0 mm and each tag's",
        f"black square {TAG:.2f} mm; measure one with calipers.",
        "Cut along the grey outline. Glue the sticker flat on the back of the DIP link, the face behind the fingertip board",
        "between the step at the PIP end and the top cover, with a 0.25 mm reveal on each side. The arrows and the label",
        "point toward the PIP joint: seen from the back with the PIP joint up, the tag reads upright.",
        "Thumb 24, index 25, middle 26. The four screw holes lie under the corners of the white border; press the paper flat",
        "there. Tag centre in the link's frame L (x_L dorsal, y_L along the PIP axis, z_L toward the PIP axis):",
        f"x_L {FACE_X + STICKER_T:.2f}, y_L {face_centre()[0]:.2f}, z_L {face_centre()[1]:.3f} mm "
        f"(scripts/hardware/dip_tag_print.py; {REPORT.name}).",
    ]
    for ln in lines:
        ax.text(18.0, y, ln, ha="left", va="top", fontsize=9.0, family="serif")
        y -= 5.0
    bx, by = 18.0, y - 8.0
    ax.plot([bx, bx + 100], [by, by], color="k", lw=0.6)
    for k in range(11):
        ax.plot([bx + 10 * k, bx + 10 * k], [by, by + (3 if k % 5 == 0 else 1.8)], color="k", lw=0.5)
    ax.text(bx + 103, by, "100 mm", ha="left", va="center", fontsize=9.5, family="serif")
    with PdfPages(PDF) as pdf:
        pdf.savefig(fig)
    fig.savefig(PREVIEW, dpi=110, facecolor="white")
    plt.close(fig)
    maps = {f: tag_map(f).to_json() for f in IDS}
    rep = {"what": "AprilTag stickers for the back of the DIP link b v4 (tongue's back), Letter PDF at actual size",
           "script": "scripts/hardware/dip_tag_print.py", "pdf": str(PDF.relative_to(ROOT)), "page_mm": PAGE,
           "family": FAMILY, "tag_mm": TAG, "ids": IDS, "copies": COPIES, "face_mm": FACE, "face_z_L_mm": FACE_Z,
           "face_x_L_mm": FACE_X, "holes_yz_L_mm": HOLES, "hole_r_mm": HOLE_R, "cut_mm": cut_size(),
           "sticker_t_mm": STICKER_T,
           "frame_L": "the cad_v2 mesh v4b_dip_link.stl's frame = <finger>_pip_frame in assets/mjcf/experimental/"
                      "20261104-cad_v3/ur5_full_k0.xml (thumb's mesh turned 180 deg about z there): origin on the PIP "
                      "axis, y_L along it, x_L dorsal, z_L toward the PIP axis; x_L = -x_D, y_L = y_D, z_L = -z_D",
           "maps_L": maps,
           "stickers_page_mm": [{"finger": f, "id": IDS[f], "centre": [round(cx, 3), round(PAGE[1] - cy, 3)]}
                                for f, cx, cy in cells]}
    REPORT.write_text(json.dumps(rep, indent=1))
    print("wrote", PDF.relative_to(ROOT), PREVIEW.relative_to(ROOT), REPORT.relative_to(ROOT))
    return 0


def cmd_check(a):
    """Rasterize the sticker grid at CHECK_DPI and detect the tags: ids, black-square edges and centres (mm)."""
    import cv2
    from pupil_apriltags import Detector
    rep = json.loads(REPORT.read_text())
    ppm = CHECK_DPI / 25.4
    cs = np.array([s["centre"] for s in rep["stickers_page_mm"]])           # mm from the page's top-left corner
    w, h = rep["cut_mm"]
    lo, hi = cs.min(0) - [w, h], cs.max(0) + [w, h]
    box = [int(round(v * ppm)) for v in (lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1])]
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-r", str(CHECK_DPI), "-gray", "-png", "-x", str(box[0]), "-y", str(box[1]),
                        "-W", str(box[2]), "-H", str(box[3]), "-f", "1", "-l", "1", str(PDF), f"{td}/p"], check=True)
        img = cv2.imread(str(next(Path(td).glob("p*.png"))), cv2.IMREAD_GRAYSCALE)
    det = Detector(families=FAMILY, nthreads=2, quad_decimate=1.0, refine_edges=True)
    found = det.detect(img)
    off = np.array([box[0], box[1]]) / ppm
    rows = []
    for d in found:
        c = d.corners / ppm + off                                 # pupil_apriltags: pixel edges at integers
        edges = np.linalg.norm(np.roll(c, -1, 0) - c, axis=1)
        centre = c.mean(0)
        k = int(np.argmin(np.linalg.norm(cs - centre, axis=1)))
        want = rep["stickers_page_mm"][k]
        up = (c[0] + c[1]) / 2 - centre                           # the top edge's midpoint (right-top, left-top)
        rows.append({"id": int(d.tag_id), "want": want["id"], "edge_mm": [round(float(e), 4) for e in edges],
                     "centre_err_mm": round(float(np.linalg.norm(centre - cs[k])), 4),
                     "upright": bool(up[1] < 0 and abs(up[0]) < 0.05)})
    e = np.array([r["edge_mm"] for r in rows]) if rows else np.zeros((0, 4))
    out = {"dpi": CHECK_DPI, "detected": len(rows), "stickers": len(cs),
           "ids_ok": all(r["id"] == r["want"] for r in rows),
           "edge_mm": [round(float(e.min()), 4), round(float(e.max()), 4)] if len(e) else None,
           "centre_err_mm_max": max((r["centre_err_mm"] for r in rows), default=None),
           "upright": all(r["upright"] for r in rows)}
    out["ok"] = bool(out["detected"] == out["stickers"] and out["ids_ok"] and out["upright"]
                     and abs(out["edge_mm"][0] - TAG) < 0.05 and abs(out["edge_mm"][1] - TAG) < 0.05
                     and out["centre_err_mm_max"] < 0.05)
    rep["check"] = out
    REPORT.write_text(json.dumps(rep, indent=1))
    print(json.dumps(out))
    return 0 if out["ok"] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("check")
    a = ap.parse_args()
    return {"build": cmd_build, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
