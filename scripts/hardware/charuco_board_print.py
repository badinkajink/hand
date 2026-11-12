#!/usr/bin/env python3
"""A printed ChArUco board for the D435 colour cameras of the object tracking plan
(docs/notes/20261105-object_seating_and_tracking/). It is moved by hand in front of each camera to calibrate the lens,
focal length and distortion, which the factory model lacks (bench of 2026-11-09), and then fastened to the hand for the
hand-eye poses (owner, 2026-11-12: one ChArUco board for both). 7 x 5 squares of 21 mm with 14 mm DICT_5X5_100 markers,
ids 0-16; both counts are odd, so OpenCV 4.6 (corvus) and 4.10 draw the same pattern. The board's corners lie where
black squares meet, so the black filament's spread on the tag block (0.16 mm per edge, 2026-11-09) does not move them.

The pattern comes from OpenCV once (`layout`, written to src/sr2hand/tracking/charuco_7x5_5x5_100.json). `build` makes
the white plate and its black cells in one frame with manifold3d, face up on z = 0: the board in an 8 mm white margin,
163 x 121 x 4 mm, the black 0.6 mm deep, four M3 clearance holes 4 mm in from the corners. `check` draws the black
body's top faces at 10 px/mm and runs OpenCV's ChArUco detector on the drawing and on OpenCV's own drawing of the board: all
24 corners, with their ids, must agree. `bambu` writes the Bambu Studio project, the white on filament 2 and the black on filament 3.

    ~/.claude/bin/resguard.sh run --mem 2G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/bin/python scripts/hardware/charuco_board_print.py layout          (OpenCV)
    ~/.claude/bin/resguard.sh run --mem 2G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/envs/femtip/bin/python scripts/hardware/charuco_board_print.py build   (manifold3d)
    ~/.claude/bin/resguard.sh run --mem 2G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/bin/python scripts/hardware/charuco_board_print.py check [--preview PNG]
    ~/.claude/bin/resguard.sh run --mem 3G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/bin/python scripts/hardware/charuco_board_print.py bambu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for _paths
import _paths  # noqa: E402,F401
import object_tracking_print as OP  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATE = "20261112"
OUT = ROOT / f"assets/tracking/{DATE}-charuco_board"
REPORT = OUT / f"{DATE}-charuco_board.json"
LAYOUT = ROOT / "src/sr2hand/tracking/charuco_7x5_5x5_100.json"
NX, NY, SQUARE, MARKER, DICT = 7, 5, 21.0, 14.0, "DICT_5X5_100"
LAYOUT_PPM = 2                 # px/mm of the layout image: every edge of this board falls on its 0.5 mm grid
MARGIN, THICK, DEPTH = 8.0, 4.0, 0.6     # mm: white margin round the board, plate thickness, black depth
HOLE_IN = 4.0                  # mm: the M3 clearance holes' centres from the plate's edges
PROJECT = OUT / f"{DATE}-charuco_board_p1s.3mf"
PLACE, TOWER = (105.0, 95.0), (205.0, 200.0)   # mm on the P1S bed: the board's centre and the prime tower
CHECK_PPM = 10.0               # px/mm of check's drawing (4 x 4 supersampled)


def stl(colour):
    return OUT / f"{DATE}-charuco_board_7x5_{colour}.stl"


def board_wh():
    return NX * SQUARE, NY * SQUARE


def opencv_board():
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT))
    return cv2, cv2.aruco.CharucoBoard((NX, NY), SQUARE, MARKER, d)


def cmd_layout(a):
    """The board's black pixels from OpenCV's own drawing, as runs per row of the LAYOUT_PPM image."""
    cv2, b = opencv_board()
    W, H = board_wh()
    img = b.generateImage((int(W * LAYOUT_PPM), int(H * LAYOUT_PPM)), marginSize=0, borderBits=1)
    b.setLegacyPattern(True)
    same = bool((img == b.generateImage(img.shape[::-1], marginSize=0, borderBits=1)).all())
    runs = []
    for r, row in enumerate(img < 128):
        c = 0
        while c < row.size:
            if row[c]:
                e = c
                while e + 1 < row.size and row[e + 1]:
                    e += 1
                runs.append([r, c, e])
                c = e + 1
            else:
                c += 1
    out = {"source": f"OpenCV {cv2.__version__} cv2.aruco.CharucoBoard(({NX}, {NY}), {SQUARE}, {MARKER}, {DICT})"
                     f".generateImage at {LAYOUT_PPM} px/mm, marginSize 0, borderBits 1; written once by "
                     "scripts/hardware/charuco_board_print.py layout",
           "dictionary": DICT, "squares": [NX, NY], "square_mm": SQUARE, "marker_mm": MARKER, "px_per_mm": LAYOUT_PPM,
           "image_px": list(img.shape[::-1]), "legacy_pattern_identical": same,
           "ids": [int(i) for i in np.asarray(b.getIds()).ravel()], "black_runs": runs}
    LAYOUT.write_text(json.dumps(out))
    print(f"wrote {LAYOUT.relative_to(ROOT)}: {len(runs)} runs, ids {out['ids'][0]}-{out['ids'][-1]}, "
          f"legacy pattern identical {same}")


def cmd_build(a):
    import manifold3d as M
    lay = json.loads(LAYOUT.read_text())
    W, H = board_wh()
    PW, PH = W + 2 * MARGIN, H + 2 * MARGIN
    S = M.Manifold.cube([PW, PH, THICK]).translate([-PW / 2, -PH / 2, 0.0])
    for sx in (-1, 1):
        for sy in (-1, 1):
            S = S - M.Manifold.cylinder(THICK + 2.0, OP.HOLE_D / 2, OP.HOLE_D / 2, OP.HOLE_SEG).translate(
                [sx * (PW / 2 - HOLE_IN), sy * (PH / 2 - HOLE_IN), -1.0])
    px = 1.0 / lay["px_per_mm"]
    runs = [M.CrossSection.square([(c1 - c0 + 1) * px, px]).translate([c0 * px - W / 2, H / 2 - (r + 1) * px])
            for r, c0, c1 in lay["black_runs"]]
    region = M.CrossSection.batch_boolean(runs, M.OpType.Add).offset(OP.EPS, M.JoinType.Miter)
    cells = M.Manifold.extrude(region, DEPTH + OP.PROUD).translate([0.0, 0.0, THICK - DEPTH])
    black, white = S ^ cells, S - cells
    expected = sum(c1 - c0 + 1 for _r, c0, c1 in lay["black_runs"]) * px * px * DEPTH
    vol = {"solid": S.volume(), "white": white.volume(), "black": black.volume(), "black_expected": expected,
           "overlap": (white ^ black).volume()}
    ok = (abs(vol["white"] + vol["black"] - vol["solid"]) < 1e-3 and vol["overlap"] < 1e-6
          and abs(vol["black"] / expected - 1.0) < 0.03)
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for colour, m in (("white", white), ("black", black)):
        t = OP.to_trimesh(m)
        ok &= bool(t.is_watertight and t.is_winding_consistent and t.volume > 0)
        t.export(stl(colour))
        files[colour] = {"file": str(stl(colour).relative_to(ROOT)), "triangles": int(len(t.faces)),
                         "watertight": bool(t.is_watertight), "shells": len(t.split(only_watertight=False)),
                         "bounds_mm": np.round(t.bounds, 3).tolist()}
    rep = {"what": "ChArUco board for the D435 colour cameras' lens and hand-eye calibrations, white and black STLs",
           "script": "scripts/hardware/charuco_board_print.py", "board": {k: lay[k] for k in
                                                                       ("dictionary", "squares", "square_mm", "marker_mm", "ids")},
           "plate_mm": [PW, PH, THICK], "black_depth_mm": DEPTH, "files": files,
           "volume_mm3": {k: round(v, 4) for k, v in vol.items()}, "ok": bool(ok)}
    REPORT.write_text(json.dumps(rep, indent=1))
    print("board", "ok" if ok else "FAILED", {k: round(v, 3) for k, v in vol.items()}, "->", REPORT.relative_to(ROOT))
    if not ok:
        raise SystemExit(1)


def cmd_check(a):
    import trimesh
    cv2, b = opencv_board()
    W, H = board_wh()
    mesh = {k: trimesh.load(stl(k), force="mesh") for k in ("white", "black")}
    top, up = np.array([0.0, 0.0, THICK]), np.array([0.0, 0.0, 1.0])
    half = max(W, H) / 2 + MARGIN
    # the drawing: x along the board's right, y down; OpenCV's board points have their origin at its top-left corner
    img = OP.draw(OP.in_plane(mesh["black"], top, up), top, np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]),
                  half, ppm=CHECK_PPM)
    det = cv2.aruco.CharucoDetector(b)
    corners, ids, _mc, _mi = det.detectBoard(img)
    found = {int(i): c for i, c in zip(np.ravel(ids), corners.reshape(-1, 2))} if ids is not None else {}
    # the reference: the same detector on OpenCV's own drawing of the board, with a margin that puts the board where the
    # mesh drawing has it (the detector's corners sit 0.49 px off the board's points in both drawings, its convention)
    m = int(round(MARGIN * CHECK_PPM))
    ref = b.generateImage((int(W * CHECK_PPM) + 2 * m, int(H * CHECK_PPM) + 2 * m), marginSize=m, borderBits=1)
    rc, rids, _a, _b = det.detectBoard(ref)
    shift = np.array([half - W / 2 - MARGIN, half - H / 2 - MARGIN]) * CHECK_PPM
    want = {int(i): c + shift for i, c in zip(np.ravel(rids), rc.reshape(-1, 2))}
    err = [float(np.linalg.norm(found[k] - want[k])) / CHECK_PPM for k in want if k in found]
    aw, ab = OP.area(OP.in_plane(mesh["white"], top, up)), OP.area(OP.in_plane(mesh["black"], top, up))
    PW, PH = W + 2 * MARGIN, H + 2 * MARGIN
    face = PW * PH - 4 * 0.5 * OP.HOLE_SEG * (OP.HOLE_D / 2) ** 2 * np.sin(2 * np.pi / OP.HOLE_SEG)
    ok = len(found) == len(want) and max(err) < 0.05 and abs(face - aw - ab) < 1e-3
    rep = json.loads(REPORT.read_text())
    rep["check"] = {"ok": bool(ok), "ppm": CHECK_PPM, "corners_found": len(found), "corners": len(want),
                    "corner_err_mm": [round(min(err), 4), round(float(np.sqrt(np.mean(np.square(err)))), 4),
                                      round(max(err), 4)],
                    "face_mm2": round(face, 3), "white_mm2": round(aw, 3), "black_mm2": round(ab, 3)}
    REPORT.write_text(json.dumps(rep, indent=1))
    print(f"check {'ok' if ok else 'FAILED'}: {len(found)} of {len(want)} corners, error min/rms/max "
          f"{rep['check']['corner_err_mm']} mm; white {aw:.2f} + black {ab:.2f} of {face:.2f} mm2")
    if a.preview:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for k, p in want.items():
            cv2.circle(vis, tuple(int(round(v)) for v in p), 9, (0, 160, 0), 2, cv2.LINE_AA)
        for p in found.values():
            cv2.circle(vis, tuple(int(round(v)) for v in p), 3, (0, 0, 220), -1, cv2.LINE_AA)
        cv2.imwrite(a.preview, cv2.resize(vis, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA))
        print("wrote", a.preview)
    write_readme(rep)
    if not ok:
        raise SystemExit(1)


def cmd_bambu(a):
    item = {"name": "board", "white": stl("white"), "black": stl("black"), "place": PLACE}
    q = OP.bambu_project([item], PROJECT, ROOT / f"logs/{DATE}-charuco_board/bambu", "ChArUco board", tower=TOWER)
    rep = json.loads(REPORT.read_text())
    rep["bambu"] = q
    REPORT.write_text(json.dumps(rep, indent=1))
    write_readme(rep)
    print(f"{PROJECT.relative_to(ROOT)}: {q['prediction_s'] / 60:.0f} min, {q['layers']} layers, "
          f"{q['filament_loads']} filament loads, used {q['used_g']} g, boxes {q['boxes_mm']}")
    print("problems:", q["problems"] or "none")
    if q["problems"]:
        raise SystemExit(1)


def write_readme(rep):
    c = rep.get("check", {})
    L = [f"# ChArUco board, {DATE}",
         "",
         "Lens and hand-eye calibration target for the D435 colour cameras of "
         "`docs/notes/20261105-object_seating_and_tracking/`, written by `scripts/hardware/charuco_board_print.py`.",
         "",
         f"OpenCV board: `cv2.aruco.CharucoBoard(({NX}, {NY}), {SQUARE}, {MARKER}, cv2.aruco.getPredefinedDictionary("
         f"cv2.aruco.{DICT}))`, lengths in mm, marker ids 0-16. Both square counts are odd, so OpenCV 4.6 "
         "(`cv2.aruco.CharucoBoard_create(7, 5, 21.0, 14.0, d)` on corvus) and 4.10 draw the same pattern.",
         "",
         f"Files: `{stl('white').name}` and `{stl('black').name}`, one frame, face up on z = 0: the board in an "
         f"{MARGIN:g} mm white margin, {rep['plate_mm'][0]:g} x {rep['plate_mm'][1]:g} x {THICK:g} mm, the black "
         f"{DEPTH} mm deep, four M3 clearance holes {HOLE_IN:g} mm in from the corners. Load both at once as one "
         "object with two parts and give the black part the black filament. Matte white and matte black PLA, 0.2 mm "
         "layers, at least 3 top layers, textured PEI plate."]
    if c:
        L += ["", f"Check: drawn from the black body at {c['ppm']:.0f} px/mm, OpenCV's ChArUco detector finds "
                  f"{c['corners_found']} of {c['corners']} corners, {c['corner_err_mm'][1]:.4f} mm RMS and at most "
                  f"{c['corner_err_mm'][2]:.4f} mm from those found on OpenCV's own drawing of the board; on the top "
                  "face the white and black "
                  "areas sum to the face's area."]
    if "bambu" in rep:
        q = rep["bambu"]
        L += ["", f"Bambu Studio project: `{Path(q['project']).name}`, the white on filament 2 and the black on "
                  f"filament 3 (AMS slots 2 and 3), {q['presets']['process']} and {q['presets']['filament']}. "
                  f"Sliced by the command line: {q['prediction_s'] / 3600:.1f} h, "
                  f"{q['used_g'].get('2', 0):.1f} g white and {q['used_g'].get('3', 0):.1f} g black, "
                  f"{q['filament_loads']} filament loads. It is not in git; `charuco_board_print.py bambu` rebuilds "
                  "it."]
    (OUT / "README.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("layout", "build", "bambu"):
        sub.add_parser(name)
    p = sub.add_parser("check")
    p.add_argument("--preview", help="PNG of the drawing with the board's corners (green) and the detected (red)")
    a = ap.parse_args()
    {"layout": cmd_layout, "build": cmd_build, "check": cmd_check, "bambu": cmd_bambu}[a.cmd](a)


if __name__ == "__main__":
    main()
