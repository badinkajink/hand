#!/usr/bin/env python3
"""Paper calibration boards for the D435 colour cameras of the object tracking plan, sized for the owner's rigid back of
about 140 x 130 mm (owner, 2026-11-13: "i have an approximately 14cm x 13cm rigid back. can you prepare a checkerboard to
print to fit on there, with appropriate amounts of edge space to allow for gluing"; "i mean paper checkerboard").

One Letter PDF at actual size, two pages: a ChArUco board of 7 x 7 squares of 15 mm with 10.5 mm DICT_5X5_100 markers
(ids 0 to 23; both counts odd, so OpenCV 4.6 on corvus and 4.10 draw the same pattern), and a plain checkerboard of
9 x 8 squares of 12 mm (8 x 7 inner corners; one count odd and one even, so the pattern differs when turned 180 deg).
Each pattern is centred in a 140 x 130 mm trim outline with crop marks, which leaves a white border of at least 12.5 mm
for the glue and for screws. A 100 mm bar checks the printer's scale. The black areas are drawn as one filled path per
page from OpenCV's own drawing at 4 px/mm, on which every edge of these boards falls.

`build` writes the PDF and a preview PNG of both boards in their trim outlines; `check` rasterizes the boards' region of the PDF at CHECK_DPI with pdftoppm and detects both boards
with OpenCV, comparing the corners with the board model (camera stack).

    ~/.claude/bin/resguard.sh run --mem 2G -- ~/miniconda3/bin/python scripts/hardware/paper_board_print.py build
    ~/.claude/bin/resguard.sh run --mem 2G -- ~/miniconda3/bin/python scripts/hardware/paper_board_print.py check   # 0.9 GB
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

ROOT = Path(__file__).resolve().parents[2]
DATE = "20261113"
OUT = ROOT / f"assets/tracking/{DATE}-paper_board"
PDF = OUT / f"{DATE}-paper_boards_letter.pdf"
PREVIEW = OUT / f"{DATE}-paper_boards_preview.png"
REPORT = OUT / f"{DATE}-paper_boards.json"
PAGE = (215.9, 279.4)                       # mm, US Letter
TRIM = (140.0, 130.0)                       # mm, the rigid back
PPM = 4                                     # px/mm of OpenCV's drawing: every edge of both boards falls on its grid
CHARUCO = dict(squares=(7, 7), square=15.0, marker=10.5, dictionary="DICT_5X5_100")
CHESS = dict(squares=(9, 8), square=12.0)
CHECK_DPI = 300                             # 3.0 GB peak at 600 dpi (cv2 on 3,800 px pages); 300 dpi is 0.085 mm/px


def charuco_board():
    import cv2
    c = CHARUCO
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, c["dictionary"]))
    return cv2.aruco.CharucoBoard(c["squares"], c["square"], c["marker"], d)


def charuco_image():
    """OpenCV's drawing of the ChArUco board at PPM px/mm (0 black, 255 white), no margin."""
    c = CHARUCO
    w, h = (int(round(n * c["square"] * PPM)) for n in c["squares"])
    img = charuco_board().generateImage((w, h), marginSize=0, borderBits=1)
    assert img.shape == (h, w)
    return img


def chess_image():
    c = CHESS
    s = int(round(c["square"] * PPM))
    nx, ny = c["squares"]
    ii, jj = np.meshgrid(np.arange(ny * s) // s, np.arange(nx * s) // s, indexing="ij")
    return np.where((ii + jj) % 2 == 0, 0, 255).astype(np.uint8)


def rectangles(img):
    """Black pixels of img as rectangles (x0, y0, w, h) in px, from row runs merged downward where a run repeats."""
    black = img < 128
    open_, out = {}, []
    for r in range(black.shape[0] + 1):
        runs = set()
        if r < black.shape[0]:
            row = np.r_[False, black[r], False].astype(np.int8)
            d = np.diff(row)
            runs = set(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))
        for key in list(open_):
            if key not in runs:
                y0 = open_.pop(key)
                out.append((key[0], y0, key[1] - key[0], r - y0))
        for key in runs:
            open_.setdefault(key, r)
    return out


def black_path(img, x0_mm, y0_mm):
    """One compound path of img's black rectangles placed with its top-left corner at (x0_mm, y0_mm) on a page whose y
    runs up: a single fill, so the printer sees no seams between neighbouring rectangles."""
    from matplotlib.path import Path as MPath
    verts, codes = [], []
    for x, y, w, h in rectangles(img):
        a, b = x0_mm + x / PPM, y0_mm - (y + h) / PPM
        verts += [(a, b), (a + w / PPM, b), (a + w / PPM, b + h / PPM), (a, b + h / PPM), (a, b)]
        codes += [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY]
    return MPath(verts, codes)


def page(fig_title, img, lines):
    """A Letter page (mm coordinates, y up) with the pattern centred in the trim outline, crop marks, a 100 mm bar and
    the text lines below."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, Rectangle
    fig = plt.figure(figsize=(PAGE[0] / 25.4, PAGE[1] / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE[0])
    ax.set_ylim(0, PAGE[1])
    ax.set_aspect("equal")
    ax.axis("off")
    tx0, ty1 = (PAGE[0] - TRIM[0]) / 2, PAGE[1] - 25.0          # trim outline: left edge and top edge
    ty0 = ty1 - TRIM[1]
    ax.add_patch(Rectangle((tx0, ty0), TRIM[0], TRIM[1], fill=False, ec="#bbbbbb", lw=0.3, ls=(0, (2, 2))))
    for x in (tx0, tx0 + TRIM[0]):                               # crop marks, 2 mm outside the outline
        for y in (ty0, ty1):
            sx, sy = (1 if x > tx0 else -1), (1 if y > ty0 else -1)
            ax.plot([x + sx * 2, x + sx * 8], [y, y], color="k", lw=0.4)
            ax.plot([x, x], [y + sy * 2, y + sy * 8], color="k", lw=0.4)
    w_mm, h_mm = img.shape[1] / PPM, img.shape[0] / PPM
    px0, py1 = tx0 + (TRIM[0] - w_mm) / 2, ty1 - (TRIM[1] - h_mm) / 2
    ax.add_patch(PathPatch(black_path(img, px0, py1), fc="black", ec="none", lw=0))
    ax.text(PAGE[0] / 2, ty1 + 8, fig_title, ha="center", va="bottom", fontsize=12, family="serif")
    y = ty0 - 14
    for ln in lines:
        ax.text(tx0 - 10, y, ln, ha="left", va="top", fontsize=9.5, family="serif")
        y -= 5.6
    bx, by = tx0 - 10, y - 8                                     # the 100 mm bar, ticks every 10 mm
    ax.plot([bx, bx + 100], [by, by], color="k", lw=0.6)
    for k in range(11):
        ax.plot([bx + 10 * k, bx + 10 * k], [by, by + (3 if k % 5 == 0 else 1.8)], color="k", lw=0.5)
    ax.text(bx + 103, by, "100 mm", ha="left", va="center", fontsize=9.5, family="serif")
    margins = (round((TRIM[0] - w_mm) / 2, 2), round((TRIM[1] - h_mm) / 2, 2))
    return fig, {"pattern_mm": [w_mm, h_mm], "margin_mm": list(margins),
                 "pattern_top_left_mm": [round(px0, 3), round(PAGE[1] - py1, 3)]}


def cmd_build(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    OUT.mkdir(parents=True, exist_ok=True)
    c, k = CHARUCO, CHESS
    pages = [
        ("ChArUco board, 7 × 7 squares of 15 mm", charuco_image(), [
            f"ChArUco: {c['squares'][0]} × {c['squares'][1]} squares of {c['square']:g} mm, {c['marker']:g} mm markers "
            f"of {c['dictionary']}, ids 0 to 23; 36 inner corners.",
            f"OpenCV: CharucoBoard(({c['squares'][0]}, {c['squares'][1]}), {c['square']:g}, {c['marker']:g}, "
            f"{c['dictionary']}), lengths in mm.",
            "Print at 100 % (actual size, no fit to page).",
            "Seven squares measure 105.0 mm along each edge, and the bar 100.0 mm.",
            "Measure both edges with calipers and use the measured square size in the calibration.",
            "Trim at the crop marks (140 × 130 mm) or to the back's own edges; glue over the whole back.",
        ]),
        ("Checkerboard, 9 × 8 squares of 12 mm", chess_image(), [
            f"Checkerboard: {k['squares'][0]} × {k['squares'][1]} squares of {k['square']:g} mm; 8 × 7 inner "
            f"corners (OpenCV patternSize (8, 7)).",
            "Every inner corner must be in view. With one odd and one even count the pattern differs when turned 180\u00b0.",
            "Print at 100 % (actual size, no fit to page).",
            "Nine squares measure 108.0 mm and eight 96.0 mm, and the bar 100.0 mm.",
            "Measure both edges with calipers and use the measured square size in the calibration.",
            "Trim at the crop marks (140 × 130 mm) or to the back's own edges; glue over the whole back.",
        ]),
    ]
    rep = {"what": "paper calibration boards for a 140 x 130 mm rigid back, Letter PDF at actual size",
           "script": "scripts/hardware/paper_board_print.py", "pdf": str(PDF.relative_to(ROOT)), "page_mm": PAGE,
           "trim_mm": TRIM, "charuco": CHARUCO, "checkerboard": CHESS, "pages": []}
    figs = []
    with PdfPages(PDF) as pdf:
        for title, img, lines in pages:
            fig, info = page(title, img, lines)
            pdf.savefig(fig)
            figs.append(fig)
            rep["pages"].append({"title": title, **info, "rectangles": len(rectangles(img))})
    import io
    from PIL import Image
    shots, dpi = [], 100
    tx0 = (PAGE[0] - TRIM[0]) / 2
    box_mm = (tx0 - 10.0, 17.5, tx0 + TRIM[0] + 10.0, 25.0 + TRIM[1] + 10.0)   # the trim outline and its marks
    for f in figs:
        buf = io.BytesIO()
        f.savefig(buf, dpi=dpi, facecolor="white", format="png")
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        shots.append(im.crop(tuple(int(round(v * dpi / 25.4)) for v in box_mm)))
        plt.close(f)
    gap = 12
    both = Image.new("RGB", (sum(i.size[0] for i in shots) + gap, shots[0].size[1]), (255, 255, 255))
    both.paste(shots[0], (0, 0))
    both.paste(shots[1], (shots[0].size[0] + gap, 0))
    both.save(PREVIEW, optimize=True)
    REPORT.write_text(json.dumps(rep, indent=1))
    print("wrote", PDF.relative_to(ROOT), PREVIEW.relative_to(ROOT), json.dumps(rep["pages"]))
    return 0


def cmd_check(a):
    """Rasterize the PDF at CHECK_DPI and detect both boards; corner spacing and positions against the model (mm)."""
    import cv2
    rep = json.loads(REPORT.read_text())
    ppm = CHECK_DPI / 25.4
    # only the trim outline's region (with 10 mm round it) is rasterized: the whole page at 600 dpi paged at 2 GB
    tx0 = (PAGE[0] - TRIM[0]) / 2
    x0_mm, y0_mm = tx0 - 10.0, 15.0
    box = [int(round(v * ppm)) for v in (x0_mm, y0_mm, TRIM[0] + 20.0, TRIM[1] + 20.0)]
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-r", str(CHECK_DPI), "-gray", "-png", "-x", str(box[0]), "-y", str(box[1]),
                        "-W", str(box[2]), "-H", str(box[3]), str(PDF), f"{td}/p"], check=True)
        imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sorted(Path(td).glob("p-*.png"))]
    off = np.array([box[0], box[1]]) / ppm                     # the region's corner on the page (mm)
    out = {}
    # page 1: ChArUco corners against the board model placed at the pattern's top-left corner
    board = charuco_board()
    det = cv2.aruco.CharucoDetector(board)
    cc, ids, _mc, _mi = det.detectBoard(imgs[0])
    x0, y0 = rep["pages"][0]["pattern_top_left_mm"]
    model = board.getChessboardCorners()[ids.ravel(), :2]
    found = cc.reshape(-1, 2) / ppm + off - np.array([x0, y0])
    err = np.linalg.norm(found - model, axis=1)
    out["charuco"] = {"corners_found": int(len(ids)), "corners": int(len(board.getChessboardCorners())),
                      "corner_err_mm": [round(float(np.median(err)), 4), round(float(err.max()), 4)]}
    # page 2: checkerboard corners, their spacing along rows and columns
    ok, corners = cv2.findChessboardCornersSB(imgs[1], (8, 7), flags=cv2.CALIB_CB_ACCURACY)
    if ok:
        g = corners.reshape(7, 8, 2) / ppm
        dx = np.linalg.norm(np.diff(g, axis=1), axis=2)
        dy = np.linalg.norm(np.diff(g, axis=0), axis=2)
        out["checkerboard"] = {"found": True, "corners": int(len(corners)),
                               "spacing_mm": [round(float(np.mean(dx)), 4), round(float(np.mean(dy)), 4)],
                               "spacing_sd_mm": [round(float(np.std(dx)), 4), round(float(np.std(dy)), 4)]}
    else:
        out["checkerboard"] = {"found": False}
    out["dpi"] = CHECK_DPI
    out["ok"] = bool(out["charuco"]["corners_found"] == out["charuco"]["corners"]
                     and out["charuco"]["corner_err_mm"][1] < 0.05 and out["checkerboard"].get("found"))
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
