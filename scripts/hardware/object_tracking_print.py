#!/usr/bin/env python3
"""Print files of steps 1 and 3 of the object seating and tracking plan (docs/notes/20261105-object_seating_and_tracking/):
the tag test plate and the hexagonal tag block. Each part is two STL files in one frame, a white body and its black
cells, which together fill the part; the black cells are flush with the tag faces. The tags are those of the maps that
the simulation and the tracker use (object_tracking_mjcf.plate_maps and block_map), so a print matches them. The files
lie in print orientation on z = 0 (owner, 2026-10-19). STLs are the deliverable (owner, 2026-11-07: "i dont need the
3mfs, i can plate things myself"); `bambu` writes a Bambu Studio project of the two-colour parts, which the owner asked
for the same day.

  plate   170 x 70 x 3 mm, printed face up. Upper row tag36h11 ids 0-4, lower row tag25h9 ids 0-4, at 10, 12, 13, 14
          and 16 mm (the edge of the black square) on a 30 mm pitch. Black cells 0.6 mm deep, three 0.2 mm layers. Two
          M3 clearance holes 156 mm apart. The frame of plate_maps is the file's frame moved down by 3 mm.
  block   30 mm across flats and 17 mm tall, printed upright: tag36h11 ids 10-15 (the cylinder's) at 13 mm on faces 0-5.
          Black cells 1.0 mm deep. A 4.2 mm bore along the axis takes a 4 mm pin on the turntable from below, or the
          object's 4 mm stem from above. The block frame B (tags.HexBlock) is the file's frame moved down by 8.5 mm.
          Walls and infill are the slicer's.

    ~/.claude/bin/resguard.sh run --mem 2G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/envs/femtip/bin/python scripts/hardware/object_tracking_print.py build
    ~/.claude/bin/resguard.sh run --mem 2G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/bin/python scripts/hardware/object_tracking_print.py check [--preview PNG]
    ~/.claude/bin/resguard.sh run --mem 3G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/bin/python scripts/hardware/object_tracking_print.py bambu

build needs manifold3d (the femtip environment) and checks that the two bodies fill the part without overlap. check
needs pupil_apriltags and OpenCV (the camera stack). It draws each tag face of the black body at 20 px/mm from the
mesh's triangles in the face's plane, detects the tag and compares its id and corners with the map. It also sums the
white and black areas in each face's plane against the face's area, which fails if a cell stands proud or sinks.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for _paths
import _paths  # noqa: E402,F401
import object_tracking_mjcf as OM  # noqa: E402
from sr2hand.tracking import tags as TG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATE = "20261107"
OUT = ROOT / f"assets/tracking/{DATE}-tag_prints"
REPORT = OUT / f"{DATE}-tag_prints.json"
PLATE_DEPTH = 0.6              # mm: the plate's black cells, three 0.2 mm layers
BLOCK_DEPTH = 1.0              # mm: the block's black cells
BORE_D = 4.2                   # mm: the block's axial bore
HOLE_D, HOLE_X = 3.4, 78.0     # mm: the plate's M3 clearance holes at (+-HOLE_X, 0)
HOLE_SEG, BORE_SEG = 48, 64    # polygon segments of the hole and the bore
EPS = 0.005                    # mm: each tag's black region grows by this much in the face's plane
PROUD = 0.1                    # mm: the black regions stand this far out of the face before the cut
BLOCK_SHAPE = "cylinder"       # the block bears the cylinder's ids, 10-15
LAYER = 0.2                    # mm: the layer height of the settings lines
PPM, SS = 20.0, 4              # px/mm of check's drawings, drawn at SS x SS and averaged
NAMES = {"plate": "tag_test_plate", "block": "tag_block_ids10-15"}


def shifted(tm, dz):
    """The map moved up by dz (mm), each tag keeping its own size (a plate map holds several sizes)."""
    out = TG.TagMap(tm.family, tm.size)
    t = np.array([0.0, 0.0, dz])
    for i, (c, r, d, n) in tm.frames.items():
        out.frames[i] = (c + t, r, d, n)
        out.corners[i] = tm.corners[i] + t
    return out


def maps(part):
    """The tag maps of a part in its file's frame (mm)."""
    if part == "plate":
        return [shifted(tm, OM.PLATE_DIMS[2]) for tm in OM.plate_maps().values()]
    return [shifted(OM.block_map(BLOCK_SHAPE), OM.BLOCK.height / 2)]


def edge(tm, tid):
    """A tag's own edge (mm), from its right-top and left-top corners."""
    C = tm.corners[tid]
    return float(np.linalg.norm(C[0] - C[1]))


def black_cells(tm, depth):
    """Manifold of the black cells of every tag of tm, from the tag's face to depth below it. Each tag's black region is
    the union of its rows' runs of black cells, made in the face's plane and extruded along the normal. Cells that meet
    only at a corner share one edge, which is not manifold once an STL's vertices are merged, so the region is grown by
    EPS: such cells join by a neck 2 EPS wide, and the black square grows by 2 EPS (0.01 mm). The region also stands
    PROUD out of the face, and the caller cuts the part's solid with it: on a slanted face of the hexagon a region that
    ended in the face fell a rounding error short of it and left a white skin over the cells, and boxes per run left
    zero-area triangles where they overlapped."""
    import manifold3d as M
    out = []
    for tid, (c, r, d, n) in tm.frames.items():
        size = edge(tm, tid)
        cell, h = size / TG.n_cells(tm.family), size / 2.0
        runs = [M.CrossSection.square([(j1 - j0 + 1) * cell, cell]).translate([-h + j0 * cell, h - (i + 1) * cell])
                for i, j0, j1 in TG.black_rects(tm.family, tid)]
        region = M.CrossSection.batch_boolean(runs, M.OpType.Add).offset(EPS, M.JoinType.Miter)
        A = np.hstack([np.stack([r, -d, n], 1), c[:, None]])     # (right, up, out of the face): right-handed, n = d x r
        out.append(M.Manifold.extrude(region, depth + PROUD).translate([0.0, 0.0, -depth]).transform(A))
    return M.Manifold.batch_boolean(out, M.OpType.Add)


def solid(part):
    """The part's solid in its file's frame (mm), on z = 0."""
    import manifold3d as M
    if part == "plate":
        W, H, T = OM.PLATE_DIMS
        s = M.Manifold.cube([W, H, T]).translate([-W / 2, -H / 2, 0.0])
        for x in (-HOLE_X, HOLE_X):
            s = s - M.Manifold.cylinder(T + 2.0, HOLE_D / 2, HOLE_D / 2, HOLE_SEG).translate([x, 0.0, -1.0])
        return s
    B = OM.BLOCK
    s = M.Manifold.extrude(M.CrossSection([B.vertices()[:6, :2].tolist()]), B.height)
    return s - M.Manifold.cylinder(B.height + 2.0, BORE_D / 2, BORE_D / 2, BORE_SEG).translate([0.0, 0.0, -1.0])


def to_trimesh(m):
    import trimesh
    mm = m.to_mesh64()
    t = trimesh.Trimesh(np.asarray(mm.vert_properties, float)[:, :3], np.asarray(mm.tri_verts), process=True)
    t.merge_vertices()
    return t


def stl(part, colour):
    return OUT / f"{DATE}-{NAMES[part]}_{colour}.stl"


def cmd_build(a):
    import manifold3d as M
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"what": "tag test plate (step 1) and tag block (step 3) as white and black STLs",
           "script": "scripts/hardware/object_tracking_print.py"}
    for part in ("plate", "block"):
        tms, depth = maps(part), (PLATE_DEPTH if part == "plate" else BLOCK_DEPTH)
        S = solid(part)
        cells = M.Manifold.batch_boolean([black_cells(tm, depth) for tm in tms], M.OpType.Add)
        black, white = S ^ cells, S - cells       # the proud boxes cross each face plane; neither lies in it
        cells_area = sum(int(TG.cells(tm.family, i).sum()) * (edge(tm, i) / TG.n_cells(tm.family)) ** 2
                         for tm in tms for i in tm.ids())
        vol = {"solid": S.volume(), "white": white.volume(), "black": black.volume(),
               "black_expected": cells_area * depth, "overlap": (white ^ black).volume()}
        ok = (abs(vol["white"] + vol["black"] - vol["solid"]) < 1e-3 and vol["overlap"] < 1e-6
              and abs(vol["black"] / vol["black_expected"] - 1.0) < 0.03)
        files = {}
        for colour, m in (("white", white), ("black", black)):
            t = to_trimesh(m)
            ok &= bool(t.is_watertight and t.is_winding_consistent and t.volume > 0)
            t.export(stl(part, colour))
            files[colour] = {"file": str(stl(part, colour).relative_to(ROOT)), "triangles": int(len(t.faces)),
                             "watertight": bool(t.is_watertight), "shells": len(t.split(only_watertight=False)),
                             "bounds_mm": np.round(t.bounds, 3).tolist()}
        z = np.round(to_trimesh(black).bounds[:, 2], 3)
        mid = LAYER * (np.arange(int(z[1] / LAYER) + 2) + 0.5)       # layer mid-heights, the first layer LAYER thick
        rep[part] = {"files": files, "volume_mm3": {k: round(v, 4) for k, v in vol.items()}, "ok": bool(ok),
                     "black_z_mm": z.tolist(), "two_colour_layers": int(((mid > z[0]) & (mid < z[1])).sum()),
                     "tags": {tm.family: {str(i): round(edge(tm, i), 3) for i in tm.ids()} for tm in tms}}
        print(part, "ok" if ok else "FAILED", {k: round(v, 3) for k, v in vol.items()}, flush=True)
    REPORT.write_text(json.dumps(rep, indent=1))
    print("wrote", REPORT.relative_to(ROOT))
    if not all(rep[p]["ok"] for p in ("plate", "block")):
        raise SystemExit(1)


def in_plane(mesh, c, n, tol=1e-3):
    """(k, 3, 3) triangles of mesh in the plane through c whose outward normal is n."""
    V, F = mesh.vertices, mesh.faces
    on = (mesh.face_normals @ n > 0.9999) & (np.abs((V[F].mean(1) - c) @ n) < tol)
    return V[F[on]]


def area(tris):
    return float(0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum())


def to_px(P, c, r, d, half):
    """Points (mm) to pixels of a drawing about c, x along r and y along d, pixel centres at integers (OpenCV)."""
    Q = np.asarray(P) - c
    return (np.stack([Q @ r, Q @ d], -1) + half) * PPM - 0.5


def draw(tris, c, r, d, half, ppm=PPM):
    """The triangles drawn black on white over the square of half-width `half` (mm) about c, at PPM px/mm, drawn at
    SS times the resolution and averaged."""
    import cv2
    N = int(round(2 * half * ppm))
    big = np.full((N * SS, N * SS), 255, np.uint8)
    Q = tris - c
    uv = (np.stack([Q @ r, Q @ d], -1) + half) * ppm * SS - 0.5
    for t in np.round(uv * 256).astype(np.int32):
        cv2.fillConvexPoly(big, t, 0, lineType=cv2.LINE_8, shift=8)
    return cv2.resize(big, (N, N), interpolation=cv2.INTER_AREA)


def cmd_check(a):
    import cv2
    import trimesh
    from sr2hand.tracking import pose as TP
    rep = json.loads(REPORT.read_text())
    tiles = []
    for part in ("plate", "block"):
        mesh = {k: trimesh.load(stl(part, k), force="mesh") for k in ("white", "black")}
        rows, planes = [], {}
        for tm in maps(part):
            det = TP.detector(tm.family)
            cell = {i: edge(tm, i) / TG.n_cells(tm.family) for i in tm.ids()}
            for tid, (c, r, d, n) in tm.frames.items():
                half = edge(tm, tid) / 2 + 2 * cell[tid]
                img = draw(in_plane(mesh["black"], c, n), c, r, d, half)
                found = TP.detect(det, img)
                want = to_px(tm.corners[tid], c, r, d, half)
                err = (float(np.sqrt(((found[tid] - want) ** 2).sum(1).mean())) / PPM) if tid in found else None
                rows.append({"family": tm.family, "id": tid, "edge_mm": round(edge(tm, tid), 3),
                             "found": sorted(found), "corner_rms_mm": None if err is None else round(err, 4)})
                planes.setdefault((round(float(c @ n), 6), tuple(np.round(n, 6))), (c, n))
                tiles.append((f"{tm.family} {tid}", img, found.get(tid), want))
        # the white and black areas in each tag face's plane against the face's area
        W, H, T = OM.PLATE_DIMS
        face_area = (W * H - 2 * 0.5 * HOLE_SEG * (HOLE_D / 2) ** 2 * math.sin(2 * math.pi / HOLE_SEG)
                     if part == "plate" else OM.BLOCK.face_width * OM.BLOCK.height)
        faces = []
        for c, n in planes.values():
            aw, ab = area(in_plane(mesh["white"], c, n)), area(in_plane(mesh["black"], c, n))
            faces.append({"normal": np.round(n, 4).tolist(), "white_mm2": round(aw, 4), "black_mm2": round(ab, 4),
                          "face_mm2": round(face_area, 4), "gap_mm2": round(face_area - aw - ab, 6)})
        ok = (all(r_["found"] == [r_["id"]] and r_["corner_rms_mm"] is not None and r_["corner_rms_mm"] < 0.05
                  for r_ in rows) and all(abs(f["gap_mm2"]) < 1e-3 for f in faces))
        rep[part]["check"] = {"ok": bool(ok), "ppm": PPM, "tags": rows, "faces": faces}
        print(part, "check", "ok" if ok else "FAILED", flush=True)
        for r_ in rows:
            print(f"  {r_['family']:9s} id {r_['id']:2d} {r_['edge_mm']:5.1f} mm  found {r_['found']}  "
                  f"corner rms {r_['corner_rms_mm']} mm")
        for f in faces:
            print(f"  face {f['normal']}: white {f['white_mm2']:.2f} + black {f['black_mm2']:.2f} of {f['face_mm2']:.2f} mm2")
    REPORT.write_text(json.dumps(rep, indent=1))
    write_readme(rep)
    if a.preview:
        sheet(tiles, a.preview)
    if not all(rep[p]["check"]["ok"] for p in ("plate", "block")):
        raise SystemExit(1)


def sheet(tiles, path, side=240):
    """The drawn faces side by side, the map's corners as green circles and the detected ones as red dots."""
    import cv2
    out = []
    for name, img, found, want in tiles:
        s = side / img.shape[0]
        t = cv2.cvtColor(cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        for p in want:
            cv2.circle(t, tuple(int(round(v)) for v in (p + 0.5) * s - 0.5), 6, (0, 160, 0), 1, cv2.LINE_AA)
        for p in ([] if found is None else found):
            cv2.circle(t, tuple(int(round(v)) for v in (p + 0.5) * s - 0.5), 2, (0, 0, 220), -1, cv2.LINE_AA)
        cv2.putText(t, name, (6, side - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 60, 0), 1, cv2.LINE_AA)
        out.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(120, 120, 120)))
    cols = 8
    out += [np.full_like(out[0], 255)] * (-len(out) % cols)
    grid = np.vstack([np.hstack(out[k:k + cols]) for k in range(0, len(out), cols)])
    cv2.imwrite(str(path), grid)
    print("wrote", path)


def write_readme(rep):
    p, b = rep["plate"], rep["block"]
    rms = [t["corner_rms_mm"] for part in (p, b) for t in part["check"]["tags"]]
    L = [f"# Tag test plate and tag block, {DATE}",
         "",
         "Print files of steps 1 and 3 of `docs/notes/20261105-object_seating_and_tracking/20261105-object_seating_and_"
         "tracking_plan.md`, written by `scripts/hardware/object_tracking_print.py`. Each part is a white body and its "
         "black cells in one frame, lying on z = 0. Load both files of a part at once as one object with two parts, and "
         "give the black part the black filament.",
         "",
         "| Part | Files | Print |",
         "|---|---|---|",
         f"| Tag test plate, 170 x 70 x 3 mm: tag36h11 ids 0-4 (upper row) and tag25h9 ids 0-4 at 10, 12, 13, 14 and "
         f"16 mm; two M3 clearance holes | `{Path(p['files']['white']['file']).name}`, "
         f"`{Path(p['files']['black']['file']).name}` | face up; black cells {PLATE_DEPTH} mm deep, "
         f"{p['two_colour_layers']} layers in both colours |",
         f"| Tag block, 30 mm across flats, 17 mm tall: tag36h11 ids 10-15 at 13 mm; 4.2 mm bore along the axis | "
         f"`{Path(b['files']['white']['file']).name}`, `{Path(b['files']['black']['file']).name}` | upright; black "
         f"cells {BLOCK_DEPTH} mm deep, {b['two_colour_layers']} layers in both colours |",
         "",
         f"Settings for both: matte white and matte black PLA on the P1S with the AMS, 0.4 mm nozzle, {LAYER} mm layers "
         "with a 0.2 mm first layer, textured PEI plate, prime tower on. The plate: 15 % infill and at least 3 top "
         "layers. The block: 2 walls (the plan's 0.8 mm), 15 % gyroid infill. On the block the black squares' top and "
         "bottom edges, 2.0 and 15.0 mm above the bed, fall on layer boundaries.",
         "",
         "Frames: the plate's tag map (`object_tracking_mjcf.plate_maps`) is the file's frame moved down by 3 mm; the "
         "block frame B (`sr2hand.tracking.tags.HexBlock`) is the file's frame moved down by 8.5 mm, z_B up, face k's "
         "normal at 60k deg from x_B.",
         "",
         f"Check: each tag face of the black body drawn at {PPM:.0f} px/mm from the mesh decodes as its own id, with "
         f"corners {min(rms):.3f} to {max(rms):.3f} mm RMS from the map; in each face's plane the white and black "
         "areas sum to the face's area. Numbers in `" + REPORT.name + "`."]
    if "bambu" in rep:
        q = rep["bambu"]
        L += ["",
              f"Bambu Studio project: `{Path(q['project']).name}`, both parts on one plate, the white bodies on filament "
              f"{q['filament']['white']} and the black cells on filament {q['filament']['black']} (AMS slots "
              f"{q['filament']['white']} and {q['filament']['black']}), presets {q['presets']['process']} and "
              f"{q['presets']['filament']} for the {q['presets']['machine']}. Sliced by the command line: "
              f"{q['prediction_s'] / 3600:.1f} h, {q['used_g'].get(str(q['filament']['white']), 0):.1f} g white and "
              f"{q['used_g'].get(str(q['filament']['black']), 0):.1f} g black, "
              f"{q['filament_loads']} filament loads. It is not in git (3MF files stay off GitHub); "
              "`object_tracking_print.py bambu` rebuilds it."]
    (OUT / "README.md").write_text("\n".join(L) + "\n")
    print("wrote", (OUT / "README.md").relative_to(ROOT))


# The Bambu Studio project (owner, 2026-11-07: "im gonna put white filament on amshead 2, and black filament on ams head
# 3. can you prepare the 3mf for bambustudio"). Filaments 1-3 of the project stand for AMS slots 1-3: the white bodies
# print with filament 2 and the black cells with filament 3; filament 1 is unused. The project is not committed (3MF files
# stay off GitHub, owner 2026-11-06); `bambu` rebuilds it.
PROJECT = OUT / f"{DATE}-tag_prints_p1s.3mf"
P1S = dict(machine="Bambu Lab P1S 0.4 nozzle", process="0.20mm Standard @BBL X1C",
           filament="Bambu PLA Basic @BBL P1S 0.4 nozzle")
P1S_PROCESS = dict(curr_bed_type="Textured PEI Plate", brim_type="no_brim", sparse_infill_pattern="gyroid",
                   wipe_tower_x=["175"], wipe_tower_y=["160"])     # the prime tower on the bed and clear of both parts
COLOURS = ("#898989", "#FFFFFF", "#000000")
FILAMENT = {"white": 2, "black": 3}
PLACE = {"plate": (110.0, 70.0), "block": (60.0, 170.0)}           # mm on the bed: the centre of each white body
BED = 256.0                                                        # mm: the P1S's printable square


def extrusion_bounds(gcode):
    """(x0, y0, x1, y1) of the extruding moves of a Bambu G-code's printed features and prime tower (absolute XY,
    relative E). The printer's own purges are left out: the start sequence's line at y = -3 mm (feature "Custom") and the
    flushes at the purge chute behind the bed inside each filament change (M620 ... M621)."""
    import re
    x = y = feat = None
    change = False
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for line in gcode.splitlines():
        if line.startswith("; FEATURE:"):
            feat = line[10:].strip()
        elif re.match(r"\s*M620 S\d+A", line):
            change = True
        elif re.match(r"\s*M621 S", line):
            change = False
        m = re.match(r"G[01] ([^;]*)", line)
        if not m:
            continue
        w = dict((t[0], float(t[1:])) for t in m.group(1).split() if t[0] in "XYE" and len(t) > 1)
        x, y = w.get("X", x), w.get("Y", y)
        if ("X" in w or "Y" in w) and w.get("E", 0.0) > 0 and x is not None and y is not None \
                and feat not in (None, "Custom") and not change:        # a G1 E alone is an unretraction in place
            lo, hi = np.minimum(lo, (x, y)), np.maximum(hi, (x, y))
    return np.concatenate([lo, hi])


def bambu_project(items, project, work, plate_name, tower=(175.0, 160.0)):
    """A Bambu Studio project of two-colour parts through the command line. Each item {name, white, black, place} is
    assembled from its white and black STLs as one object with two parts, on filaments FILAMENT["white"] and
    FILAMENT["black"], its white body centred at `place` (mm on the bed); the prime tower goes to `tower`. The project is
    sliced once so that its G-code can be checked (the owner slices again in the GUI); an item whose black is None is a
    part in white only. Returns the slice's numbers and
    the problems found: the P1S start sequence, temperatures, filaments loaded, each black part's offset from its white
    part against the STLs, every printed move on the bed and the prime tower clear of every part."""
    import re
    import shutil
    import subprocess
    import zipfile

    import trimesh

    import dip_fit_print_plates as DF        # bbl_preset (presets with the printer's G-code templates), BAMBU, the check
    work.mkdir(parents=True, exist_ok=True)
    process = dict(P1S_PROCESS, wipe_tower_x=[f"{tower[0]:g}"], wipe_tower_y=[f"{tower[1]:g}"])
    (work / "machine.json").write_text(json.dumps(DF.bbl_preset("machine", P1S["machine"]), indent=1))
    (work / "process.json").write_text(json.dumps(DF.bbl_preset("process", P1S["process"], process), indent=1))
    fils = []
    for k, col in enumerate(COLOURS, 1):
        fils.append(work / f"filament{k}.json")
        fils[-1].write_text(json.dumps(DF.bbl_preset("filament", P1S["filament"], {"filament_colour": [col]}), indent=1))
    objs = [{"path": str(it[colour]), "count": 1, "filaments": [FILAMENT[colour]], "assemble_index": [k],
             "pos_x": [it["place"][0]], "pos_y": [it["place"][1]], "pos_z": [0.0]}
            for k, it in enumerate(items, 1) for colour in ("white", "black") if it.get(colour) is not None]
    (work / "assemble.json").write_text(json.dumps(
        {"plates": [{"plate_name": plate_name, "need_arrange": False, "objects": objs}]}, indent=1))
    # --export-3mf takes a bare name: the command line joins it to --outputdir
    cmd = DF.BAMBU + ["--debug", "1", "--arrange", "0", "--orient", "0",
                      "--load-settings", f"{work / 'machine.json'};{work / 'process.json'}",
                      "--load-filaments", ";".join(str(f) for f in fils),
                      "--load-assemble-list", str(work / "assemble.json"),
                      "--slice", "0", "--export-3mf", project.name, "--outputdir", str(work)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    (work / "cli.log").write_text(r.stdout + r.stderr)
    res = json.loads((work / "result.json").read_text())
    if res.get("return_code") != 0:
        raise SystemExit(f"Bambu Studio: {res.get('error_string')} (log {work / 'cli.log'})")
    shutil.copy(work / project.name, project)
    with zipfile.ZipFile(project) as z:
        msets = z.read("Metadata/model_settings.config").decode()
        info = z.read("Metadata/slice_info.config").decode()
        gcode = z.read("Metadata/plate_1.gcode").decode(errors="replace")
        proj = json.loads(z.read("Metadata/project_settings.config"))
    problems = DF.check_bambu_gcode(gcode, "P1S", 220, 55)
    # each object's white part on filament 2 and its black on 3, offset from each other as in the STLs
    parts = re.findall(r'<part id="\d+"[^>]*>.*?key="matrix" value="([^"]+)".*?key="source_file" value="([^"]+)"'
                       r'.*?key="extruder" value="(\d+)"', msets, re.S)
    placed = {f: (np.array(m.split(), float).reshape(4, 4)[:3, 3], int(e)) for m, f, e in parts}
    offsets = {}
    for it in items:
        if it.get("black") is None:                     # a part in white only
            if placed[Path(it["white"]).name][1] != FILAMENT["white"]:
                problems.append(f"{it['name']}: filament {placed[Path(it['white']).name][1]}")
            continue
        t = {c: placed[Path(it[c]).name] for c in ("white", "black")}
        ctr = {c: trimesh.load(it[c], force="mesh").bounds.mean(0) for c in ("white", "black")}
        off = (t["black"][0] - t["white"][0]) - (ctr["black"] - ctr["white"])
        offsets[it["name"]] = float(np.abs(off).max())
        if t["white"][1] != FILAMENT["white"] or t["black"][1] != FILAMENT["black"] or offsets[it["name"]] > 1e-4:
            problems.append(f"{it['name']}: filaments {t['white'][1]} and {t['black'][1]}, black part off by {off} mm")
    loads = [int(v) for v in re.findall(r"^\s*M620 S(\d+)A", gcode, re.M)]
    if sorted(set(loads)) != [FILAMENT["white"] - 1, FILAMENT["black"] - 1]:
        problems.append(f"filaments loaded {sorted(set(loads))}")
    box = extrusion_bounds(gcode)
    if box[:2].min() < 0 or box[2:].max() > BED:
        problems.append(f"extrusion outside the bed {box}")
    boxes = feature_boxes(gcode)
    for name, b in boxes.items():
        if name != "tower" and "tower" in boxes and overlap(boxes["tower"], b, 2.0):
            problems.append(f"prime tower {np.round(boxes['tower'], 1)} meets object {name} {np.round(b, 1)}")
    used = {i: float(g) for i, g in re.findall(r'<filament id="(\d+)"[^>]*used_g="([\d.]+)"', info)}
    return {"project": str(project.relative_to(ROOT)), "presets": P1S, "process": process,
            "colours": COLOURS, "filament": FILAMENT, "place_mm": {it["name"]: it["place"] for it in items},
            "prediction_s": float(re.search(r'key="prediction" value="([\d.]+)"', info).group(1)),
            "used_g": used, "filament_loads": len(loads),
            "layers": int(re.search(r"; total layer number: (\d+)", gcode).group(1)),
            "extrusion_box_mm": np.round(box, 2).tolist(), "part_offset_err_mm": offsets,
            "boxes_mm": {k: np.round(v, 1).tolist() for k, v in boxes.items()},
            "flush_volumes_matrix": proj.get("flush_volumes_matrix"), "problems": problems}


def feature_boxes(gcode):
    """{object label, "object" when the G-code has none, or "tower": (x0, y0, x1, y1)} of a Bambu G-code's extrusions
    outside filament changes. The tower is what lies between WIPE_TOWER_START and WIPE_TOWER_END; the feature labels
    run on for a few moves into the next feature in both directions, so they do not separate tower and parts."""
    import re
    x = y = feat = obj = None
    change = tower = False
    pts = {}
    for line in gcode.splitlines():
        if line.startswith("; FEATURE:"):
            feat = line[10:].strip()
        elif line.startswith("; WIPE_TOWER_START"):
            tower = True
        elif line.startswith("; WIPE_TOWER_END"):
            tower = False
        elif line.startswith("M624 "):
            obj = line.split()[1]
        elif line.startswith("M625"):
            obj = None
        elif re.match(r"\s*M620 S\d+A", line):
            change = True
        elif re.match(r"\s*M621 S", line):
            change = False
        m = re.match(r"G[01] ([^;]*)", line)
        if not m:
            continue
        w = dict((t[0], float(t[1:])) for t in m.group(1).split() if t[0] in "XYE" and len(t) > 1)
        x, y = w.get("X", x), w.get("Y", y)
        if not ("X" in w or "Y" in w) or w.get("E", 0.0) <= 0 or x is None or y is None or change \
                or feat in (None, "Custom"):                            # a G1 E alone is an unretraction in place
            continue
        if tower:
            pts.setdefault("tower", []).append((x, y))
        elif feat != "Prime tower":
            pts.setdefault(obj or "object", []).append((x, y))
    return {k: np.concatenate([np.min(v, 0), np.max(v, 0)]) for k, v in pts.items()}


def overlap(a, b, gap):
    """Whether boxes (x0, y0, x1, y1) a and b come closer than gap (mm)."""
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def cmd_bambu(a):
    items = [{"name": p, "white": stl(p, "white"), "black": stl(p, "black"), "place": PLACE[p]} for p in ("plate", "block")]
    b = bambu_project(items, PROJECT, ROOT / f"logs/{DATE}-tag_prints/bambu", "tag plate and block")
    rep = json.loads(REPORT.read_text())
    rep["bambu"] = b
    REPORT.write_text(json.dumps(rep, indent=1))
    write_readme(rep)
    print(f"{PROJECT.relative_to(ROOT)}: {b['prediction_s'] / 60:.0f} min, {b['layers']} layers, "
          f"{b['filament_loads']} filament loads, used {b['used_g']} g, extrusion box {b['extrusion_box_mm']}")
    print("problems:", b["problems"] or "none")
    if b["problems"]:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p = sub.add_parser("check")
    p.add_argument("--preview", help="PNG of the drawn faces with the map's and the detected corners")
    sub.add_parser("bambu")
    a = ap.parse_args()
    {"build": cmd_build, "check": cmd_check, "bambu": cmd_bambu}[a.cmd](a)


if __name__ == "__main__":
    main()
